"""Multi-constraint PPO-Lagrangian for CMDP-core PowerZooJax envs."""

from __future__ import annotations

import time
from functools import partial
from typing import Callable, NamedTuple

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from jax.experimental import io_callback

from powerzoojax.rl.config import TrainConfig
from powerzoojax.rl.wrappers import SafeRLWrapper


class SafeActorCritic(nn.Module):
    """Actor-critic with one reward head and ``k`` cost critics."""

    action_dim: int
    n_constraints: int
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)

        mean = nn.tanh(nn.Dense(self.action_dim)(x))
        log_std_param = self.param(
            "log_std", nn.initializers.constant(-0.5), (self.action_dim,)
        )
        log_std = jnp.clip(log_std_param, -5.0, 2.0)
        v_reward = jnp.squeeze(nn.Dense(1)(x), axis=-1)
        v_cost = nn.Dense(self.n_constraints)(x)
        return mean, log_std, v_reward, v_cost


def _gaussian_log_prob(mean, log_std, action):
    std = jnp.exp(log_std)
    var = std ** 2
    log_p = -0.5 * (((action - mean) ** 2) / var + 2 * log_std + jnp.log(2 * jnp.pi))
    return jnp.sum(log_p, axis=-1)


def _gaussian_entropy(log_std):
    return jnp.sum(0.5 + 0.5 * jnp.log(2 * jnp.pi) + log_std, axis=-1)


def _gaussian_sample(key, mean, log_std):
    std = jnp.exp(log_std)
    return mean + std * jax.random.normal(key, shape=mean.shape)


class _SafeTransition(NamedTuple):
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    cost: chex.Array
    done: chex.Array
    v_reward: chex.Array
    v_cost: chex.Array
    log_prob: chex.Array


def _resolved_cost_thresholds(env: SafeRLWrapper, config: TrainConfig) -> tuple[float, ...]:
    if config.cost_thresholds or config.cost_threshold != 0.0:
        return config.resolved_cost_thresholds(env.num_constraints)
    return tuple(float(x) for x in env.cost_thresholds)


def make_cmdp_train(env, config: TrainConfig) -> Callable:
    """Create a JIT-compiled multi-constraint PPO-Lagrangian trainer."""

    if not isinstance(env, SafeRLWrapper):
        raise TypeError(
            f"make_cmdp_train requires a SafeRLWrapper, got {type(env).__name__}. "
            "Use: env = SafeRLWrapper(your_env, params, ...)"
        )
    if env.num_constraints <= 0:
        raise ValueError("SafeRLWrapper must expose at least one selected constraint.")

    hidden_dim = config.hidden_dims[0] if config.hidden_dims else 128
    n_updates = max(1, config.total_timesteps // (config.num_envs * config.n_steps))
    n_constraints = env.num_constraints
    constraint_names = tuple(env.selected_constraint_names)
    cost_scale = jnp.asarray(config.resolved_cost_scale(n_constraints), dtype=jnp.float32)
    # Divide by cost_scale to match rollout.cost (line ~159), which is also divided by
    # cost_scale before GAE and the dual update (episode_cost_est - thresholds).
    thresholds = jnp.asarray(
        _resolved_cost_thresholds(env, config), dtype=jnp.float32
    ) / cost_scale
    lambda_lr = jnp.asarray(config.resolved_lambda_lr(n_constraints), dtype=jnp.float32)

    def train(key):
        key, init_key = jax.random.split(key)
        dummy_obs = jnp.zeros((env.obs_size,), dtype=jnp.float32)
        network = SafeActorCritic(
            action_dim=env.num_actions,
            n_constraints=n_constraints,
            hidden_dim=hidden_dim,
        )
        net_params = network.init(init_key, dummy_obs)

        tx = optax.chain(
            optax.zero_nans(),
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.learning_rate),
        )
        opt_state = tx.init(net_params)
        log_lambda = jnp.zeros((n_constraints,), dtype=jnp.float32)

        key, env_key = jax.random.split(key)
        env_keys = jax.random.split(env_key, config.num_envs)
        obs_batch, env_state_batch = jax.vmap(env.reset)(env_keys)

        def _update_step(runner_state, _):
            net_params, opt_state, log_lambda, env_state_batch, obs_batch, key = runner_state
            lam = jnp.exp(log_lambda)

            def _env_step(carry, _):
                env_state, obs, key = carry
                key, act_key, step_key = jax.random.split(key, 3)

                mean, log_std, v_reward, v_cost = jax.vmap(
                    partial(network.apply, net_params)
                )(obs)
                action = jax.vmap(_gaussian_sample)(
                    jax.random.split(act_key, config.num_envs), mean, log_std
                )
                action = jnp.clip(action, -1.0, 1.0)
                log_prob = jax.vmap(_gaussian_log_prob)(mean, log_std, action)

                step_keys = jax.random.split(step_key, config.num_envs)
                next_obs, next_env_state, reward, cost, done, info = jax.vmap(
                    env.step
                )(step_keys, env_state, action)

                transition = _SafeTransition(
                    obs=obs,
                    action=action,
                    reward=reward,
                    cost=cost,
                    done=done,
                    v_reward=v_reward,
                    v_cost=v_cost,
                    log_prob=log_prob,
                )
                return (next_env_state, next_obs, key), transition

            (env_state_batch, obs_batch, key), rollout = jax.lax.scan(
                _env_step,
                (env_state_batch, obs_batch, key),
                None,
                length=config.n_steps,
            )

            rollout = rollout._replace(cost=rollout.cost / cost_scale[None, None, :])

            _, _, last_v_reward, last_v_cost = jax.vmap(
                partial(network.apply, net_params)
            )(obs_batch)

            def _gae_reward(carry, t):
                gae, next_v = carry
                done_mask = 1.0 - t.done.astype(jnp.float32)
                delta = t.reward + config.gamma * next_v * done_mask - t.v_reward
                gae = delta + config.gamma * config.gae_lambda * done_mask * gae
                return (gae, t.v_reward), gae

            _, adv_reward = jax.lax.scan(
                _gae_reward,
                (jnp.zeros(config.num_envs, dtype=jnp.float32), last_v_reward),
                rollout,
                reverse=True,
            )
            returns_reward = adv_reward + rollout.v_reward

            def _gae_cost(carry, t):
                gae, next_v = carry
                done_mask = (1.0 - t.done.astype(jnp.float32))[:, None]
                delta = t.cost + config.cost_gamma * next_v * done_mask - t.v_cost
                gae = delta + config.cost_gamma * config.gae_lambda * done_mask * gae
                return (gae, t.v_cost), gae

            _, adv_cost = jax.lax.scan(
                _gae_cost,
                (
                    jnp.zeros((config.num_envs, n_constraints), dtype=jnp.float32),
                    last_v_cost,
                ),
                rollout,
                reverse=True,
            )
            returns_cost = adv_cost + rollout.v_cost

            adv_reward = (adv_reward - adv_reward.mean()) / (adv_reward.std() + 1e-8)
            adv_cost_mean = adv_cost.mean(axis=(0, 1), keepdims=True)
            adv_cost_std = adv_cost.std(axis=(0, 1), keepdims=True) + 1e-8
            adv_cost = (adv_cost - adv_cost_mean) / adv_cost_std

            def _ppo_loss(net_params, batch):
                obs, action, old_log_prob, adv_r, adv_c, ret_r, ret_c = batch
                mean, log_std, v_r, v_c = network.apply(net_params, obs)
                log_prob = _gaussian_log_prob(mean, log_std, action)
                entropy = _gaussian_entropy(log_std)

                log_diff = jnp.clip(log_prob - old_log_prob, -20.0, 20.0)
                ratio = jnp.exp(log_diff)
                combined_adv = adv_r - jnp.sum(lam * adv_c, axis=-1)
                clipped = jnp.clip(ratio, 1 - config.clip_eps, 1 + config.clip_eps)
                pi_loss = -jnp.minimum(ratio * combined_adv, clipped * combined_adv).mean()

                vf_loss_reward = 0.5 * ((v_r - ret_r) ** 2).mean()
                vf_loss_cost = 0.5 * ((v_c - ret_c) ** 2).mean()
                vf_loss = vf_loss_reward + vf_loss_cost
                loss = pi_loss + config.vf_coef * vf_loss - config.ent_coef * entropy.mean()
                return loss, (pi_loss, vf_loss, entropy.mean())

            def _update_epoch(carry, _):
                net_params, opt_state, key = carry
                key, perm_key = jax.random.split(key)

                batch_size = config.n_steps * config.num_envs
                flat = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]),
                    (
                        rollout.obs,
                        rollout.action,
                        rollout.log_prob,
                        adv_reward,
                        adv_cost,
                        returns_reward,
                        returns_cost,
                    ),
                )
                permutation = jax.random.permutation(perm_key, batch_size)
                shuffled = jax.tree.map(lambda x: x[permutation], flat)
                minibatch_size = batch_size // config.n_minibatches
                minibatches = jax.tree.map(
                    lambda x: x.reshape(
                        (config.n_minibatches, minibatch_size) + x.shape[1:]
                    ),
                    shuffled,
                )

                def _update_minibatch(carry, batch):
                    net_params, opt_state = carry
                    grad_fn = jax.value_and_grad(_ppo_loss, has_aux=True)
                    (loss, _), grads = grad_fn(net_params, batch)
                    grads = jax.tree.map(
                        lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0),
                        grads,
                    )
                    updates, opt_state = tx.update(grads, opt_state, net_params)
                    new_params = optax.apply_updates(net_params, updates)
                    new_params = jax.tree.map(
                        lambda new, old: jnp.where(jnp.isfinite(new), new, old),
                        new_params,
                        net_params,
                    )
                    return (new_params, opt_state), loss

                (net_params, opt_state), losses = jax.lax.scan(
                    _update_minibatch, (net_params, opt_state), minibatches
                )
                return (net_params, opt_state, key), losses.mean()

            key, epoch_key = jax.random.split(key)
            (net_params, opt_state, _), epoch_losses = jax.lax.scan(
                _update_epoch,
                (net_params, opt_state, epoch_key),
                None,
                length=config.n_epochs,
            )

            mean_cost = jnp.mean(rollout.cost, axis=(0, 1))
            episode_cost_est = mean_cost * config.n_steps
            log_lambda = jnp.clip(
                log_lambda + lambda_lr * (episode_cost_est - thresholds),
                -float(config.log_lambda_max),
                float(config.log_lambda_max),
            )

            step_mean_cost_scaled = rollout.cost.mean(axis=1)
            step_mean_reward = rollout.reward.mean(axis=1)
            step_done_rate = rollout.done.astype(jnp.float32).mean(axis=1)

            metrics = {
                "loss": epoch_losses.mean(),
                "mean_reward": jnp.mean(rollout.reward),
                "mean_cost_total": jnp.sum(mean_cost),
                "episode_cost_est_total": jnp.sum(episode_cost_est),
                "lambda_total": jnp.sum(jnp.exp(log_lambda)),
                # Compatibility aliases for legacy readers.
                "mean_cost": jnp.sum(mean_cost),
                "episode_cost_est": jnp.sum(episode_cost_est),
                "lambda": jnp.sum(jnp.exp(log_lambda)),
                "step_mean_reward": step_mean_reward,
                "step_done_rate": step_done_rate,
            }
            for idx, name in enumerate(constraint_names):
                metrics[f"mean_cost_{name}"] = mean_cost[idx]
                metrics[f"episode_cost_est_{name}"] = episode_cost_est[idx]
                metrics[f"lambda_{name}"] = jnp.exp(log_lambda[idx])
                metrics[f"step_mean_cost_scaled_{name}"] = step_mean_cost_scaled[:, idx]

            # ── Greedy eval pass ─────────────────────────────────────────────
            # When n_eval_envs > 0: run a short deterministic rollout every
            # update step to produce eval_returns and eval_cost_<name> that are
            # directly comparable to rejax PPO's eval_returns (greedy, no
            # exploration noise).  This branch is resolved at trace time so it
            # adds zero overhead when disabled (n_eval_envs == 0).
            if config.n_eval_envs > 0:
                key, eval_key = jax.random.split(key)
                eval_obs0, eval_state0 = jax.vmap(env.reset)(
                    jax.random.split(eval_key, config.n_eval_envs)
                )

                def _eval_env_step(carry, _):
                    e_state, e_obs, e_key = carry
                    e_key, step_key = jax.random.split(e_key)
                    # Greedy: use distribution mean, no sampling
                    mean_a, _, _, _ = jax.vmap(
                        partial(network.apply, net_params)
                    )(e_obs)
                    action = jnp.clip(mean_a, -1.0, 1.0)
                    e_obs_new, e_state_new, e_rew, e_cost, _, _ = jax.vmap(
                        env.step
                    )(jax.random.split(step_key, config.n_eval_envs), e_state, action)
                    return (e_state_new, e_obs_new, e_key), (e_rew, e_cost)

                _, (eval_rews, eval_costs_raw) = jax.lax.scan(
                    _eval_env_step,
                    (eval_state0, eval_obs0, eval_key),
                    None,
                    length=config.n_steps,
                )
                # eval_rews:      (n_steps, n_eval_envs)
                # eval_costs_raw: (n_steps, n_eval_envs, n_constraints)
                eval_return = eval_rews.sum(axis=0).mean()
                eval_cost_ps = (eval_costs_raw / cost_scale).mean(axis=(0, 1))

                metrics["eval_returns"] = eval_return
                for idx, name in enumerate(constraint_names):
                    metrics[f"eval_cost_{name}"] = eval_cost_ps[idx]

            runner_state = (net_params, opt_state, log_lambda, env_state_batch, obs_batch, key)
            return runner_state, metrics

        runner_state = (net_params, opt_state, log_lambda, env_state_batch, obs_batch, key)
        return runner_state, _update_step

    from powerzoojax.rl.trainer import TrainResult

    n_ckpt = max(1, int(getattr(config, "n_checkpoints", 1)))

    def _make_segment_fn(update_step_fn, seg_len: int):
        @jax.jit
        def run_segment(runner_state):
            return jax.lax.scan(update_step_fn, runner_state, None, length=seg_len)

        return run_segment

    def train_fn(key):
        runner_state, update_step_fn = train(key)
        wall_times: list[float] = []
        t0 = time.perf_counter()

        if config.record_eval_wall_time:
            def _stamp(_):
                wall_times.append(time.perf_counter() - t0)

            def _timed_update_step(carry, xs):
                new_carry, metrics = update_step_fn(carry, xs)
                io_callback(_stamp, (), jnp.array(0.0, dtype=jnp.float32))
                return new_carry, metrics

            update_step_fn_local = _timed_update_step
        else:
            update_step_fn_local = update_step_fn

        if n_ckpt <= 1:
            run_full = _make_segment_fn(update_step_fn_local, n_updates)
            runner_state, metrics = run_full(runner_state)
            net_params = runner_state[0]
            if wall_times:
                import numpy as _np

                metrics = {
                    **metrics,
                    "eval_wall_time_s": _np.asarray(wall_times, dtype=_np.float32),
                }
            return TrainResult(params=net_params, metrics=metrics, config=config)

        base = n_updates // n_ckpt
        seg_lens = [base] * n_ckpt
        seg_lens[-1] += n_updates - base * n_ckpt

        run_seg = _make_segment_fn(update_step_fn_local, base)
        run_seg_last = _make_segment_fn(update_step_fn_local, seg_lens[-1])

        checkpoints = []
        checkpoint_walltimes = []
        all_metrics: list[dict] = []
        cur_state = runner_state
        total_steps_done = 0
        env_steps_per_update = config.num_envs * config.n_steps
        t0 = time.perf_counter()
        for i, seg_len in enumerate(seg_lens):
            if seg_len <= 0:
                continue
            fn = run_seg if (i < n_ckpt - 1 or seg_lens[-1] == base) else run_seg_last
            cur_state, seg_metrics = fn(cur_state)
            total_steps_done += seg_len * env_steps_per_update
            checkpoints.append((total_steps_done, cur_state[0]))
            checkpoint_walltimes.append(time.perf_counter() - t0)
            all_metrics.append(seg_metrics)

        # Concatenate per-update metrics across all segments so the full
        # training trajectory is preserved (not just the last segment).
        import numpy as _np
        combined: dict = {}
        for key in all_metrics[0].keys():
            try:
                arrays = [_np.atleast_1d(_np.asarray(m[key])) for m in all_metrics]
                combined[key] = _np.concatenate(arrays, axis=0)
            except Exception:
                combined[key] = all_metrics[-1][key]
        combined["checkpoint_walltime_s"] = _np.asarray(
            checkpoint_walltimes, dtype=_np.float32
        )
        if wall_times:
            combined["eval_wall_time_s"] = _np.asarray(
                wall_times, dtype=_np.float32
            )

        return TrainResult(
            params=cur_state[0],
            metrics=combined,
            config=config,
            checkpoints=checkpoints,
        )

    return train_fn
