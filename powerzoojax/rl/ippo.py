"""IPPO (Independent PPO) training backend for GridMARLEnv.

Implements parameter-sharing IPPO where all agents use one shared ActorCritic.
Designed for use with GridMARLEnv and TrainConfig.

This file inlines the training loop so PowerZooJax does not depend on Mava
(see dependency-pin note below). The algorithm matches common
**independent PPO with parameter sharing**: one shared ActorCritic, per-agent
GAE, pooled clipped surrogate, value clipping, entropy bonus, ``lax.scan`` +
``vmap`` rollout.

Algorithm (lineage detail)
--------------------------
Documented as PureJaxRL / Mava-style single-file IPPO. The core update follows
Mava's ``ff_ippo`` public recipe (Mava calls the implementation variant
``Anakin``): independent learners with parameter sharing, GAE per agent with
pooled clipped-PPO surrogate, value-function clipping, entropy bonus, and full
``lax.scan`` + ``vmap``-based rollout. Differences from upstream Mava are
limited to integration code (no Hydra config, no orbax checkpointer, no pmap;
we use ``vmap`` on a single device, which is equivalent on single-GPU /
single-host setups).

We do NOT take Mava as a dependency because Mava develop hard-locks
``jax==0.5.3 / jaxlib==0.5.3 / brax==0.10.3 / numpy==1.26.4``, which is
incompatible with the PowerZooJax venv (e.g. jax 0.6.x).

Key design:
    - Parameter sharing: single ActorCritic network shared across all agents.
    - GAE per-agent with pooled PPO update across all agents.
    - vmap over environments, lax.scan over rollout and update epochs.
    - n_updates = total_timesteps // (num_envs * n_steps)  (same env-step budget as Rejax PPO)

Usage::

    from powerzoojax.rl.multi_agent import GridMARLEnv
    from powerzoojax.rl.ippo import make_ippo_train
    from powerzoojax.rl.config import TrainConfig

    env = GridMARLEnv(TransGridEnv(), params)
    config = TrainConfig(algo="ippo", total_timesteps=200_000, num_envs=16, n_steps=48)
    train_fn = make_ippo_train(env, config)
    result = train_fn(jax.random.PRNGKey(42))
    print(result.summary)
"""

import time
from functools import partial
from typing import Callable, Dict, List, Tuple, Any

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import chex

from powerzoojax.rl.config import TrainConfig
from powerzoojax.rl.trainer import TrainResult


# ============ Shared Actor-Critic ============

class SharedActorCritic(nn.Module):
    """Shared ActorCritic network for IPPO (parameter sharing).

    All agents use the same network but receive different observations.
    Outputs: (mean, log_std, value) where mean/log_std have shape (action_dim,).
    """
    hidden_dims: Tuple[int, ...] = (64, 64)
    action_dim: int = 1

    @nn.compact
    def __call__(self, x: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array]:
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.tanh(x)

        mean = nn.Dense(self.action_dim)(x)  # (action_dim,)
        log_std = self.param(
            "log_std", nn.initializers.constant(-0.5), (self.action_dim,)
        )

        value = nn.Dense(1)(x)
        value = jnp.squeeze(value, axis=-1)

        return mean, log_std, value


class SharedCostCritic(nn.Module):
    """Per-agent local cost critic for typed PPO-Lagrangian MARL."""

    hidden_dims: Tuple[int, ...] = (64, 64)
    n_constraints: int = 1

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.tanh(x)
        return nn.Dense(self.n_constraints)(x)


# ============ Policy Utils ============

def _log_prob(mean: chex.Array, log_std: chex.Array, action: chex.Array) -> chex.Array:
    std = jnp.exp(log_std)
    lp = -0.5 * (((action - mean) ** 2) / (std ** 2) + 2 * log_std + jnp.log(2 * jnp.pi))
    return jnp.sum(lp)


def _entropy(log_std: chex.Array) -> chex.Array:
    return jnp.sum(0.5 + 0.5 * jnp.log(2 * jnp.pi) + log_std)


def _sample_action(key: chex.PRNGKey, mean: chex.Array, log_std: chex.Array) -> chex.Array:
    std = jnp.exp(log_std)
    return mean + std * jax.random.normal(key, shape=mean.shape)


def _marl_constraint_names(env) -> Tuple[str, ...]:
    """Best-effort constraint-name lookup for MARL adapters with bound params."""
    if hasattr(env, "_dist_env") and hasattr(env, "_dist_params"):
        return tuple(env._dist_env.constraint_names(env._dist_params))
    if hasattr(env, "_grid_env") and hasattr(env, "_grid_params"):
        return tuple(env._grid_env.constraint_names(env._grid_params))
    raise ValueError(
        "Could not infer constraint names from MARL env. "
        "Expected a bound DistGridMARLEnv or GridMARLEnv."
    )


# ============ Training Factory ============

def make_ippo_train(env, config: TrainConfig) -> Callable:
    """Create a JIT-compiled IPPO training function for GridMARLEnv.

    Args:
        env:    GridMARLEnv instance (params bound at construction).
        config: TrainConfig with algo="ippo" and common PPO hyperparameters.

    Returns:
        ``train_fn(key: PRNGKey) -> TrainResult``

    The returned function is JIT-compiled.  Call it once to compile + train:
        result = train_fn(jax.random.PRNGKey(42))
    """
    n_agents = env.num_agents
    agent_names = env.agent_names
    obs_dim = env.observation_space().shape[0]
    action_dim = env.action_space().shape[0]
    n_envs = config.num_envs
    n_steps = config.n_steps
    n_epochs = config.n_epochs
    gamma = config.gamma
    gae_lambda = config.gae_lambda
    clip_eps = config.clip_eps
    ent_coef = config.ent_coef
    vf_coef = config.vf_coef
    max_grad_norm = config.max_grad_norm

    # Same env-step budget as single-agent PPO (Rejax): one update consumes
    # ``n_envs * n_steps`` grid steps (not multiplied by n_agents).
    n_updates = max(1, config.total_timesteps // (n_envs * n_steps))

    network = SharedActorCritic(hidden_dims=config.hidden_dims, action_dim=action_dim)

    tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(config.learning_rate),
    )

    normalize_observations = bool(config.normalize_observations)
    normalize_rewards = bool(config.normalize_rewards)
    eval_freq = int(config.eval_freq)
    record_eval_wall_time = bool(config.record_eval_wall_time)
    wall_time_warmup = bool(config.wall_time_warmup)
    wall_time_warmup_ts = config.wall_time_warmup_timesteps

    steps_per_update = n_envs * n_steps
    if hasattr(env, "_dist_params"):
        max_steps = int(getattr(env._dist_params, "max_steps", 48))
    elif hasattr(env, "_grid_params"):
        max_steps = int(getattr(env._grid_params, "max_steps", 48))
    elif hasattr(env, "_params") and hasattr(env._params, "max_steps"):
        max_steps = int(env._params.max_steps)
    else:
        max_steps = 48

    # Detect GenCos market env for market-specific metrics collection.
    # Python-time check — creates two compiled paths (market vs non-market).
    _is_market = hasattr(env, "compute_market_metrics")
    # Per-agent profit keys (computed at Python time, so dict structure is static).
    _agent_profit_keys = (
        [f"market/cumulative_profit_{ag}" for ag in agent_names]
        if _is_market else []
    )
    # Per-agent dispatch share keys (time-averaged share of total generation).
    _agent_dispatch_share_keys = (
        [f"market/dispatch_share_{ag}" for ag in agent_names]
        if _is_market else []
    )

    def _maybe_norm_obs(all_obs: chex.Array) -> chex.Array:
        if not normalize_observations:
            return all_obs
        flat = all_obs.reshape(-1, obs_dim)
        flat = (flat - flat.mean(axis=0)) / (flat.std(axis=0) + 1e-8)
        return flat.reshape(all_obs.shape)

    def _single_update(runner_state):
        net_params, opt_state, env_states, obs_dicts, key = runner_state

        # _is_market is a Python-time constant (True/False), so JAX compiles
        # only the taken branch — no runtime overhead.
        if _is_market:
            def _env_step(carry, _):
                env_states, obs_dicts, key = carry
                key, act_key, step_key = jax.random.split(key, 3)

                all_obs = jnp.stack([obs_dicts[name] for name in agent_names])
                all_obs = _maybe_norm_obs(all_obs)

                flat_obs = all_obs.reshape(-1, obs_dim)
                flat_mean, flat_log_std, flat_value = jax.vmap(
                    partial(network.apply, net_params)
                )(flat_obs)

                flat_actions = jax.vmap(_sample_action)(
                    jax.random.split(act_key, n_agents * n_envs),
                    flat_mean,
                    jnp.broadcast_to(flat_log_std, flat_mean.shape),
                )
                actions_arr = jnp.clip(
                    flat_actions.reshape(n_agents, n_envs, action_dim), -1.0, 1.0
                )
                actions = {name: actions_arr[i] for i, name in enumerate(agent_names)}

                flat_lp = jax.vmap(_log_prob)(
                    flat_mean,
                    jnp.broadcast_to(flat_log_std, flat_mean.shape),
                    flat_actions,
                )
                log_probs = flat_lp.reshape(n_agents, n_envs)
                values = flat_value.reshape(n_agents, n_envs)

                step_keys = jax.random.split(step_key, n_envs)
                next_obs_dicts, next_states, rewards_dicts, dones_dicts, step_info = jax.vmap(
                    env.step
                )(step_keys, env_states, actions)

                rewards = jnp.stack([rewards_dicts[name] for name in agent_names])
                dones = dones_dicts["__all__"]

                # Collect market-specific info for GenCos metrics (n_envs, ...).
                # load_mw is derived from the PRE-step env_states (not post-step):
                # after auto-reset the time_step wraps to 0 and episode_start_idx is
                # resampled, making post-step state ambiguous for profile lookup.
                _T_pool = env._params.load_profiles.shape[0]  # static Python int
                _profile_rows = (
                    env_states.episode_start_idx + env_states.time_step
                ) % _T_pool                                    # (n_envs,) int
                # (n_envs, n_loads) @ (n_loads, n_nodes) = (n_envs, n_nodes)
                _raw_loads = env._params.load_profiles[_profile_rows]
                load_mw_batch = _raw_loads @ env._params.nodes_loads_map.T

                market_info = {
                    "lmp":               step_info["lmp"],              # (n_envs, n_nodes)
                    "unit_power":        step_info["unit_power"],       # (n_envs, n_units)
                    "ramp_binding_rate": step_info["ramp_binding_rate"], # (n_envs,)
                    "gen_cost":          step_info["gen_cost"],         # (n_envs,) true TC
                    "load_mw":           load_mw_batch,                 # (n_envs, n_nodes)
                }

                transition = {
                    "obs":         all_obs,
                    "action":      actions_arr,
                    "reward":      rewards,
                    "done":        dones,
                    "value":       values,
                    "log_prob":    log_probs,
                    "market_info": market_info,
                }
                return (next_states, next_obs_dicts, key), transition
        else:
            def _env_step(carry, _):
                env_states, obs_dicts, key = carry
                key, act_key, step_key = jax.random.split(key, 3)

                all_obs = jnp.stack([obs_dicts[name] for name in agent_names])
                all_obs = _maybe_norm_obs(all_obs)

                flat_obs = all_obs.reshape(-1, obs_dim)
                flat_mean, flat_log_std, flat_value = jax.vmap(
                    partial(network.apply, net_params)
                )(flat_obs)

                flat_actions = jax.vmap(_sample_action)(
                    jax.random.split(act_key, n_agents * n_envs),
                    flat_mean,
                    jnp.broadcast_to(flat_log_std, flat_mean.shape),
                )
                actions_arr = jnp.clip(
                    flat_actions.reshape(n_agents, n_envs, action_dim), -1.0, 1.0
                )
                actions = {name: actions_arr[i] for i, name in enumerate(agent_names)}

                flat_lp = jax.vmap(_log_prob)(
                    flat_mean,
                    jnp.broadcast_to(flat_log_std, flat_mean.shape),
                    flat_actions,
                )
                log_probs = flat_lp.reshape(n_agents, n_envs)
                values = flat_value.reshape(n_agents, n_envs)

                step_keys = jax.random.split(step_key, n_envs)
                next_obs_dicts, next_states, rewards_dicts, dones_dicts, _ = jax.vmap(
                    env.step
                )(step_keys, env_states, actions)

                rewards = jnp.stack([rewards_dicts[name] for name in agent_names])
                dones = dones_dicts["__all__"]

                transition = {
                    "obs":      all_obs,
                    "action":   actions_arr,
                    "reward":   rewards,
                    "done":     dones,
                    "value":    values,
                    "log_prob": log_probs,
                }
                return (next_states, next_obs_dicts, key), transition

        (env_states, obs_dicts, key), rollout = jax.lax.scan(
            _env_step, (env_states, obs_dicts, key), None, length=n_steps
        )

        # Capture raw (un-normalised) rewards BEFORE any reward normalisation.
        # Market profit metrics must always reflect real economic profit, not
        # the normalised training signal used by the PPO loss.
        if _is_market:
            _raw_rewards = rollout["reward"]   # (n_steps, n_agents, n_envs)

        if normalize_rewards:
            rw = rollout["reward"]
            rw = rw / (jnp.std(rw) + 1e-8)
            rollout = {**rollout, "reward": rw}

        all_obs_final = jnp.stack([obs_dicts[name] for name in agent_names])
        all_obs_final = _maybe_norm_obs(all_obs_final)
        flat_final = all_obs_final.reshape(-1, obs_dim)
        _, _ls, flat_last_v = jax.vmap(partial(network.apply, net_params))(flat_final)
        last_values = flat_last_v.reshape(n_agents, n_envs)

        def _gae_step(carry, t):
            gae, nv = carry
            delta = (
                t["reward"]
                + gamma * nv * (1 - t["done"][None, :])
                - t["value"]
            )
            gae = delta + gamma * gae_lambda * (1 - t["done"][None, :]) * gae
            return (gae, t["value"]), gae

        _, advantages = jax.lax.scan(
            _gae_step,
            (jnp.zeros((n_agents, n_envs)), last_values),
            rollout,
            reverse=True,
        )
        returns = advantages + rollout["value"]

        total = n_steps * n_agents * n_envs
        flat_batch = (
            rollout["obs"].reshape(total, obs_dim),
            rollout["action"].reshape(total, action_dim),
            rollout["log_prob"].reshape(total),
            advantages.reshape(total),
            returns.reshape(total),
        )

        def _ppo_loss(net_params, batch):
            obs, action, old_lp, adv, ret = batch
            mean, ls, value = network.apply(net_params, obs)
            lp = _log_prob(mean, ls, action)
            ent = _entropy(ls)
            ratio = jnp.exp(lp - old_lp)
            clipped = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
            pi_loss = -jnp.minimum(ratio * adv, clipped * adv).mean()
            vf_loss = 0.5 * ((value - ret) ** 2).mean()
            loss = pi_loss + vf_coef * vf_loss - ent_coef * ent.mean()
            return loss, (pi_loss, vf_loss, ent.mean())

        def _update_epoch(carry, _):
            net_params, opt_state, key = carry
            key, perm_key = jax.random.split(key)
            perm = jax.random.permutation(perm_key, total)
            shuffled = jax.tree.map(lambda x: x[perm], flat_batch)
            grad_fn = jax.value_and_grad(_ppo_loss, has_aux=True)
            (loss, aux), grads = grad_fn(net_params, shuffled)
            updates, opt_state = tx.update(grads, opt_state, net_params)
            net_params = optax.apply_updates(net_params, updates)
            return (net_params, opt_state, key), loss

        key, epoch_key = jax.random.split(key)
        (net_params, opt_state, _), epoch_losses = jax.lax.scan(
            _update_epoch,
            (net_params, opt_state, epoch_key),
            None,
            length=n_epochs,
        )

        mean_reward = jnp.mean(rollout["reward"])
        runner_state = (net_params, opt_state, env_states, obs_dicts, key)

        if _is_market:
            # Compute market-specific metrics from rollout.
            # rollout["market_info"] has shape (n_steps, n_envs, ...) per key.
            market_m = env.compute_market_metrics(rollout["market_info"])
            # Per-agent cumulative profit: use _raw_rewards (captured before any
            # reward normalisation) so that profit values are in real $ units,
            # not the zero-mean normalised training signal.
            # Shape: (n_steps, n_agents, n_envs) → sum over steps → mean over envs → (n_agents,)
            cum_profit_per_agent = jnp.mean(
                jnp.sum(_raw_rewards, axis=0), axis=-1
            )  # (n_agents,)
            cum_profit_mean = jnp.mean(cum_profit_per_agent)
            per_agent_profit = {
                f"market/cumulative_profit_{ag}": cum_profit_per_agent[i]
                for i, ag in enumerate(agent_names)
            }
            # Per-agent dispatch share: pulled directly from compute_market_metrics.
            per_agent_dispatch_share = {
                f"market/dispatch_share_{ag}": market_m[f"market/dispatch_share_{ag}"]
                for ag in agent_names
            }
            update_metrics = {
                "loss":                        epoch_losses.mean(),
                "mean_reward":                 mean_reward,
                "market/price_volatility":     market_m["market/price_volatility"],
                "market/HHI":                  market_m["market/HHI"],
                "market/market_share_std":     market_m["market/market_share_std"],
                "market/ramp_binding_rate":    market_m.get(
                    "market/ramp_binding_rate",
                    jnp.float32(0.0),
                ),
                "market/cum_profit_mean":      cum_profit_mean,
                "market/planner_cost_ratio":   market_m.get(
                    "market/planner_cost_ratio",
                    jnp.float32(0.0),
                ),
                **per_agent_profit,
                **per_agent_dispatch_share,
            }
        else:
            update_metrics = {
                "loss":        epoch_losses.mean(),
                "mean_reward": mean_reward,
            }
        return runner_state, update_metrics

    train_step = jax.jit(_single_update)

    n_eval_episodes = 128

    def _one_eval_episode(params, episode_key):
        k0, k_loop = jax.random.split(episode_key)
        obs_dict, state = env.reset(k0)

        def det_act(o):
            mean, _, _ = network.apply(params, o)
            return jnp.clip(mean, -1.0, 1.0)

        def step_body(carry, _):
            obs_dict, state, total_r, key = carry
            key, sk = jax.random.split(key)
            actions = {name: det_act(obs_dict[name]) for name in agent_names}
            obs_dict, state, rews, _, _ = env.step(sk, state, actions)
            r = jnp.mean(jnp.stack([rews[name] for name in agent_names]))
            return (obs_dict, state, total_r + r, key), None

        (_, _, total_r, _), _ = jax.lax.scan(
            step_body,
            (obs_dict, state, jnp.float32(0.0), k_loop),
            None,
            length=max_steps,
        )
        return total_r

    eval_mean_return = jax.jit(
        lambda params, ek: jnp.mean(
            jax.vmap(lambda k: _one_eval_episode(params, k))(
                jax.random.split(ek, n_eval_episodes)
            )
        )
    )

    env_name = getattr(env, "name", "GridMARLEnv")

    def train_fn(key: chex.PRNGKey) -> TrainResult:
        key, init_key = jax.random.split(key)
        dummy_obs = jnp.zeros((obs_dim,))
        net_params = network.init(init_key, dummy_obs)
        opt_state = tx.init(net_params)

        key, env_key = jax.random.split(key)
        env_keys = jax.random.split(env_key, n_envs)
        obs_dicts, env_states = jax.vmap(env.reset)(env_keys)
        runner_state = (net_params, opt_state, env_states, obs_dicts, key)

        cfg_ts = int(config.total_timesteps)
        warm_ts = (
            wall_time_warmup_ts if wall_time_warmup_ts is not None else int(eval_freq)
        )
        warm_ts = max(1, min(int(warm_ts), cfg_ts))

        if wall_time_warmup and warm_ts < cfg_ts:
            warm_n_updates = max(1, warm_ts // (n_envs * n_steps))
            warm_key = jax.random.fold_in(key, 0x9E3779B9)
            for _ in range(warm_n_updates):
                warm_key, sk = jax.random.split(warm_key)
                runner_state, _ = train_step(runner_state)
            jax.block_until_ready(runner_state[0])

        t0_wall = time.perf_counter()
        eval_returns_list: List[float] = []
        eval_wall_times_list: List[float] = []

        eval_key = jax.random.fold_in(key, 0xC0FFEE)

        def _stamp_eval(er: float) -> None:
            eval_returns_list.append(float(er))
            if record_eval_wall_time:
                eval_wall_times_list.append(time.perf_counter() - t0_wall)

        if eval_freq > 0:
            er0 = float(eval_mean_return(runner_state[0], eval_key))
            eval_key = jax.random.fold_in(eval_key, 1)
            _stamp_eval(er0)

        loss_list: List[float] = []
        mean_reward_list: List[float] = []
        # Market-specific metric lists (only populated for GenCos market env).
        # Per-update keys: populated once per training update via market_metric_lists.
        # Post-loop scalar keys (NOT in this list): market/convergence_stability.
        _market_base_keys = [
            "market/price_volatility", "market/HHI",
            "market/market_share_std", "market/ramp_binding_rate",
            "market/cum_profit_mean", "market/planner_cost_ratio",
        ]
        market_metric_lists: Dict[str, List[float]] = (
            {k: [] for k in _market_base_keys + _agent_profit_keys + _agent_dispatch_share_keys}
            if _is_market else {}
        )

        # Convergence stability: L2 norm of policy-parameter change from 80% of
        # training to the end.  Checkpoint is saved after the 80th-percentile update
        # (0-indexed); requires n_updates >= 5 to be meaningful.
        _conv_checkpoint_idx = int(n_updates * 0.8) - 1 if n_updates >= 5 else -1
        _params_checkpoint = None  # will hold runner_state[0] at checkpoint

        cum_steps = 0
        next_eval_at = eval_freq

        for _update_i in range(n_updates):
            runner_state, m = train_step(runner_state)
            jax.block_until_ready(runner_state[0])
            loss_list.append(float(m["loss"]))
            mean_reward_list.append(float(m["mean_reward"]))
            if _is_market:
                for mk in market_metric_lists:
                    market_metric_lists[mk].append(float(m[mk]))
            # Save checkpoint for convergence_stability (JAX params are immutable
            # pytrees; storing the reference is safe — future train_step calls
            # return a fresh pytree and do not mutate this one).
            if _is_market and _update_i == _conv_checkpoint_idx:
                _params_checkpoint = runner_state[0]

            cum_steps += steps_per_update
            if eval_freq <= 0:
                continue
            while cum_steps >= next_eval_at:
                er = float(eval_mean_return(runner_state[0], eval_key))
                eval_key = jax.random.fold_in(eval_key, next_eval_at)
                _stamp_eval(er)
                next_eval_at += eval_freq

        metrics: dict = {
            "loss": jnp.asarray(loss_list, dtype=jnp.float32),
            "mean_reward": jnp.asarray(mean_reward_list, dtype=jnp.float32),
        }
        if _is_market:
            for mk, vals in market_metric_lists.items():
                metrics[mk] = jnp.asarray(vals, dtype=jnp.float32)
            # convergence_stability: L2 distance between 80%-checkpoint params and
            # final params.  Scalar (shape (1,)).  Zero if checkpoint unavailable
            # (training < 5 updates or non-market env).
            if _params_checkpoint is not None:
                leaves_fin = jax.tree_util.tree_leaves(runner_state[0])
                leaves_chk = jax.tree_util.tree_leaves(_params_checkpoint)
                sq_sum = sum(
                    float(jnp.sum((lf - lc) ** 2))
                    for lf, lc in zip(leaves_fin, leaves_chk)
                )
                conv_stab = float(jnp.sqrt(max(sq_sum, 0.0)))
            else:
                conv_stab = 0.0
            metrics["market/convergence_stability"] = jnp.asarray(
                [conv_stab], dtype=jnp.float32
            )
        if eval_freq > 0 and len(eval_returns_list) > 0:
            metrics["eval_returns"] = jnp.asarray(eval_returns_list, dtype=jnp.float32)
        if record_eval_wall_time and len(eval_wall_times_list) > 0:
            metrics["eval_wall_time_s"] = jnp.asarray(
                eval_wall_times_list, dtype=jnp.float32
            )

        return TrainResult(
            params=runner_state[0],
            metrics=metrics,
            config=config,
            env_name=env_name,
        )

    return train_fn


def make_ippo_typed_train(env, config: TrainConfig) -> Callable:
    """IPPO with **type-specific parameter sharing** for heterogeneous DER agents.

    Battery / PV / FlexLoad agents each have their own independent
    ``SharedActorCritic`` network (same architecture, separate parameters).
    Within a type all agents share one network (matches ``ders-medium`` presets).

    Agent type is inferred from the agent name prefix before the first ``"_"``
    (e.g. ``"battery_0"`` → type ``"battery"``).

    The returned ``TrainResult.params`` is a ``dict[str, flax_pytree]``:
    one entry per agent type, keyed by the type string.

    Args:
        env:    DistGridMARLEnv (or compatible) instance with `agent_names`.
        config: TrainConfig (algo field is ignored; typed IPPO is always used).

    Returns:
        ``train_fn(key: PRNGKey) -> TrainResult``
    """
    agent_names = env.agent_names  # static Python list
    n_agents = env.num_agents

    # ---- static type partition (Python time, not traced) ----
    type_to_indices: Dict[str, List[int]] = {}
    for i, name in enumerate(agent_names):
        atype = name.split("_")[0]
        type_to_indices.setdefault(atype, []).append(i)
    types: List[str] = sorted(type_to_indices.keys())
    type_indices_np: Dict[str, np.ndarray] = {
        t: np.asarray(idxs, dtype=np.int32) for t, idxs in type_to_indices.items()
    }

    obs_dim = env.observation_space().shape[0]
    action_dim = env.action_space().shape[0]
    n_envs = config.num_envs
    n_steps = config.n_steps
    n_epochs = config.n_epochs
    gamma = config.gamma
    gae_lambda = config.gae_lambda
    clip_eps = config.clip_eps
    ent_coef = config.ent_coef
    vf_coef = config.vf_coef
    max_grad_norm = config.max_grad_norm
    n_updates = max(1, config.total_timesteps // (n_envs * n_steps))
    eval_freq = int(config.eval_freq)
    record_eval_wall_time = bool(config.record_eval_wall_time)
    n_eval_episodes = max(1, int(config.eval_num_episodes))
    steps_per_update = n_envs * n_steps

    # One network per type — same architecture
    networks: Dict[str, SharedActorCritic] = {
        t: SharedActorCritic(hidden_dims=config.hidden_dims, action_dim=action_dim)
        for t in types
    }
    tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(config.learning_rate),
    )

    if hasattr(env, "_dist_params"):
        max_steps_ep = int(getattr(env._dist_params, "max_steps", 48))
    elif hasattr(env, "_grid_params"):
        max_steps_ep = int(getattr(env._grid_params, "max_steps", 48))
    else:
        max_steps_ep = 48

    # ---- typed forward pass (unrolled at trace time) ----
    def _forward_typed(net_params, all_obs):
        # all_obs: (n_agents, n_envs, obs_dim)
        means     = jnp.zeros((n_agents, n_envs, action_dim))
        log_stds  = jnp.zeros((n_agents, n_envs, action_dim))
        values    = jnp.zeros((n_agents, n_envs))
        for t in types:
            idxs = jnp.asarray(type_indices_np[t])
            n_t  = len(type_indices_np[t])
            t_flat = all_obs[idxs].reshape(n_t * n_envs, obs_dim)
            t_mean, t_ls, t_val = jax.vmap(
                partial(networks[t].apply, net_params[t])
            )(t_flat)
            means    = means.at[idxs].set(t_mean.reshape(n_t, n_envs, action_dim))
            log_stds = log_stds.at[idxs].set(
                jnp.broadcast_to(t_ls, (n_t * n_envs, action_dim)).reshape(n_t, n_envs, action_dim)
            )
            values   = values.at[idxs].set(t_val.reshape(n_t, n_envs))
        return means, log_stds, values

    # ---- single training update ----
    def _single_update(runner_state):
        net_params, opt_state, env_states, obs_dicts, key = runner_state

        def _env_step(carry, _):
            env_states, obs_dicts, key = carry
            key, act_key, step_key = jax.random.split(key, 3)

            all_obs = jnp.stack([obs_dicts[name] for name in agent_names])
            means, log_stds, values = _forward_typed(net_params, all_obs)

            flat_mean = means.reshape(n_agents * n_envs, action_dim)
            flat_ls   = log_stds.reshape(n_agents * n_envs, action_dim)
            flat_actions = jax.vmap(_sample_action)(
                jax.random.split(act_key, n_agents * n_envs),
                flat_mean,
                flat_ls,
            )
            actions_arr = jnp.clip(
                flat_actions.reshape(n_agents, n_envs, action_dim), -1.0, 1.0
            )
            actions = {name: actions_arr[i] for i, name in enumerate(agent_names)}

            flat_lp  = jax.vmap(_log_prob)(flat_mean, flat_ls, flat_actions)
            log_probs = flat_lp.reshape(n_agents, n_envs)

            step_keys = jax.random.split(step_key, n_envs)
            next_obs_dicts, next_states, rewards_dicts, dones_dicts, _ = jax.vmap(
                env.step
            )(step_keys, env_states, actions)

            rewards = jnp.stack([rewards_dicts[name] for name in agent_names])
            dones   = dones_dicts["__all__"]

            transition = {
                "obs":      all_obs,      # (n_agents, n_envs, obs_dim)
                "action":   actions_arr,  # (n_agents, n_envs, action_dim)
                "reward":   rewards,      # (n_agents, n_envs)
                "done":     dones,        # (n_envs,)
                "value":    values,       # (n_agents, n_envs)
                "log_prob": log_probs,    # (n_agents, n_envs)
            }
            return (next_states, next_obs_dicts, key), transition

        (env_states, obs_dicts, key), rollout = jax.lax.scan(
            _env_step, (env_states, obs_dicts, key), None, length=n_steps
        )

        all_obs_final = jnp.stack([obs_dicts[name] for name in agent_names])
        _, _, last_values = _forward_typed(net_params, all_obs_final)

        def _gae_step(carry, t):
            gae, nv = carry
            delta = (
                t["reward"]
                + gamma * nv * (1 - t["done"][None, :])
                - t["value"]
            )
            gae = delta + gamma * gae_lambda * (1 - t["done"][None, :]) * gae
            return (gae, t["value"]), gae

        _, advantages = jax.lax.scan(
            _gae_step,
            (jnp.zeros((n_agents, n_envs)), last_values),
            rollout,
            reverse=True,
        )
        returns = advantages + rollout["value"]

        # Build per-type flat batches (Python loop unrolled at trace time)
        batches_by_type: Dict[str, tuple] = {}
        for t in types:
            idxs = jnp.asarray(type_indices_np[t])
            n_t  = len(type_indices_np[t])
            tt   = n_steps * n_t * n_envs
            batches_by_type[t] = (
                rollout["obs"][:, idxs, :, :].reshape(tt, obs_dim),
                rollout["action"][:, idxs, :, :].reshape(tt, action_dim),
                rollout["log_prob"][:, idxs, :].reshape(tt),
                advantages[:, idxs, :].reshape(tt),
                returns[:, idxs, :].reshape(tt),
            )

        # ---- per-type PPO loss, summed, averaged ----
        def _ppo_loss(net_params, batches):
            total = jnp.float32(0.0)
            for t in types:
                obs, act, old_lp, adv, ret = batches[t]
                t_mean, t_ls, t_val = jax.vmap(
                    partial(networks[t].apply, net_params[t])
                )(obs)
                t_ls_bc = jnp.broadcast_to(t_ls, t_mean.shape)
                lp      = jax.vmap(_log_prob)(t_mean, t_ls_bc, act)
                ent     = _entropy(t_ls)
                ratio   = jnp.exp(lp - old_lp)
                clipped = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
                pi_loss = -jnp.minimum(ratio * adv, clipped * adv).mean()
                vf_loss = 0.5 * ((t_val - ret) ** 2).mean()
                total   = total + pi_loss + vf_coef * vf_loss - ent_coef * ent.mean()
            return total / len(types), total

        def _update_epoch(carry, _):
            net_params, opt_state, key = carry
            key, perm_key = jax.random.split(key)
            # Independent shuffle per type
            shuffled: Dict[str, tuple] = {}
            for i_t, t in enumerate(types):
                n_t  = len(type_indices_np[t])
                tt   = n_steps * n_t * n_envs
                t_pk = jax.random.fold_in(perm_key, i_t)
                perm = jax.random.permutation(t_pk, tt)
                shuffled[t] = jax.tree.map(lambda x: x[perm], batches_by_type[t])
            grad_fn = jax.value_and_grad(_ppo_loss, has_aux=True)
            (loss, _), grads = grad_fn(net_params, shuffled)
            updates, opt_state = tx.update(grads, opt_state, net_params)
            net_params = optax.apply_updates(net_params, updates)
            return (net_params, opt_state, key), loss

        key, epoch_key = jax.random.split(key)
        (net_params, opt_state, _), epoch_losses = jax.lax.scan(
            _update_epoch,
            (net_params, opt_state, epoch_key),
            None,
            length=n_epochs,
        )

        mean_reward = jnp.mean(rollout["reward"])
        return (net_params, opt_state, env_states, obs_dicts, key), {
            "loss": epoch_losses.mean(),
            "mean_reward": mean_reward,
        }

    train_step = jax.jit(_single_update)
    env_name = getattr(env, "name", "DistGridMARLEnv") + "[typed]"

    def _one_eval_episode(params, episode_key):
        k0, k_loop = jax.random.split(episode_key)
        obs_dict, state = env.reset(k0)

        def _det_action(atype, obs):
            mean, _, _ = networks[atype].apply(params[atype], obs)
            return jnp.clip(mean, -1.0, 1.0)

        def step_body(carry, _):
            obs_dict, state, total_r, key = carry
            key, sk = jax.random.split(key)
            actions = {
                name: _det_action(name.split("_")[0], obs_dict[name])
                for name in agent_names
            }
            obs_dict, state, rews, _, _ = env.step(sk, state, actions)
            r = jnp.mean(jnp.stack([rews[name] for name in agent_names]))
            return (obs_dict, state, total_r + r, key), None

        (_, _, total_r, _), _ = jax.lax.scan(
            step_body,
            (obs_dict, state, jnp.float32(0.0), k_loop),
            None,
            length=max_steps_ep,
        )
        return total_r

    eval_mean_return = jax.jit(
        lambda params, ek: jnp.mean(
            jax.vmap(lambda k: _one_eval_episode(params, k))(
                jax.random.split(ek, n_eval_episodes)
            )
        )
    )

    def train_fn(key: chex.PRNGKey) -> TrainResult:
        t0_wall = time.perf_counter()
        key, init_key = jax.random.split(key)
        dummy_obs = jnp.zeros((obs_dim,))
        net_params = {
            t: networks[t].init(jax.random.fold_in(init_key, i), dummy_obs)
            for i, t in enumerate(types)
        }
        opt_state = tx.init(net_params)

        key, env_key = jax.random.split(key)
        env_keys   = jax.random.split(env_key, n_envs)
        obs_dicts, env_states = jax.vmap(env.reset)(env_keys)
        runner_state = (net_params, opt_state, env_states, obs_dicts, key)

        eval_returns_list: List[float] = []
        eval_wall_times_list: List[float] = []
        eval_key = jax.random.fold_in(key, 0xC0FFEE)

        def _stamp_eval(er: float) -> None:
            eval_returns_list.append(float(er))
            if record_eval_wall_time:
                eval_wall_times_list.append(time.perf_counter() - t0_wall)

        if eval_freq > 0:
            er0 = float(eval_mean_return(runner_state[0], eval_key))
            eval_key = jax.random.fold_in(eval_key, 1)
            _stamp_eval(er0)

        loss_list: List[float] = []
        reward_list: List[float] = []
        cum_steps = 0
        next_eval_at = eval_freq

        for _ in range(n_updates):
            runner_state, m = train_step(runner_state)
            jax.block_until_ready(runner_state[0])
            loss_list.append(float(m["loss"]))
            reward_list.append(float(m["mean_reward"]))

            cum_steps += steps_per_update
            if eval_freq <= 0:
                continue
            while cum_steps >= next_eval_at:
                er = float(eval_mean_return(runner_state[0], eval_key))
                eval_key = jax.random.fold_in(eval_key, next_eval_at)
                _stamp_eval(er)
                next_eval_at += eval_freq

        metrics = {
            "loss":        jnp.asarray(loss_list,   dtype=jnp.float32),
            "mean_reward": jnp.asarray(reward_list, dtype=jnp.float32),
        }
        if eval_freq > 0 and len(eval_returns_list) > 0:
            metrics["eval_returns"] = jnp.asarray(
                eval_returns_list, dtype=jnp.float32
            )
        if record_eval_wall_time and len(eval_wall_times_list) > 0:
            metrics["eval_wall_time_s"] = jnp.asarray(
                eval_wall_times_list, dtype=jnp.float32
            )

        return TrainResult(
            params=runner_state[0],
            metrics=metrics,
            config=config,
            env_name=env_name,
        )

    return train_fn


def make_ippo_typed_lagrangian_train(env, config: TrainConfig) -> Callable:
    """Typed IPPO-Lagrangian for heterogeneous cooperative MARL.

    This extends ``make_ippo_typed_train`` with a shared team-level dual
    variable over explicit CMDP cost channels while keeping the existing
    type-specific actor setup:

    - one local ``SharedActorCritic`` per agent type for policy + reward value
    - one local ``SharedCostCritic`` per agent type for vector cost values
    - one shared vector ``lambda`` updated from the team cost estimate

    Reward and cost stay separated: the actor uses
    ``A_reward - lambda^T A_cost`` while raw constraint costs still come from
    ``info["constraint_costs"]``.
    """
    agent_names = env.agent_names
    n_agents = env.num_agents

    type_to_indices: Dict[str, List[int]] = {}
    for i, name in enumerate(agent_names):
        atype = name.split("_")[0]
        type_to_indices.setdefault(atype, []).append(i)
    types: List[str] = sorted(type_to_indices.keys())
    type_indices_np: Dict[str, np.ndarray] = {
        t: np.asarray(idxs, dtype=np.int32) for t, idxs in type_to_indices.items()
    }

    all_constraint_names = _marl_constraint_names(env)
    selected_names = tuple(
        getattr(env, "selected_constraint_names", all_constraint_names)
    ) or all_constraint_names
    selected_indices = tuple(
        all_constraint_names.index(name) for name in selected_names
    )
    n_constraints = len(selected_indices)
    if n_constraints <= 0:
        raise ValueError("Typed PPO-Lagrangian requires at least one constraint.")

    obs_dim = env.observation_space().shape[0]
    action_dim = env.action_space().shape[0]
    n_envs = config.num_envs
    n_steps = config.n_steps
    n_epochs = config.n_epochs
    gamma = config.gamma
    cost_gamma = config.cost_gamma
    gae_lambda = config.gae_lambda
    clip_eps = config.clip_eps
    ent_coef = config.ent_coef
    vf_coef = config.vf_coef
    max_grad_norm = config.max_grad_norm
    n_updates = max(1, config.total_timesteps // (n_envs * n_steps))

    cost_scale = jnp.asarray(
        config.resolved_cost_scale(n_constraints), dtype=jnp.float32
    )
    thresholds = (
        jnp.asarray(
            config.resolved_cost_thresholds(n_constraints), dtype=jnp.float32
        )
        / cost_scale
    )
    lambda_lr = jnp.asarray(
        config.resolved_lambda_lr(n_constraints), dtype=jnp.float32
    )
    selected_indices_arr = jnp.asarray(selected_indices, dtype=jnp.int32)

    actor_networks: Dict[str, SharedActorCritic] = {
        t: SharedActorCritic(hidden_dims=config.hidden_dims, action_dim=action_dim)
        for t in types
    }
    cost_networks: Dict[str, SharedCostCritic] = {
        t: SharedCostCritic(
            hidden_dims=config.hidden_dims,
            n_constraints=n_constraints,
        )
        for t in types
    }

    tx = optax.chain(
        optax.zero_nans(),
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(config.learning_rate),
    )

    def _forward_actor(params_bundle, all_obs):
        means = jnp.zeros((n_agents, n_envs, action_dim), dtype=jnp.float32)
        log_stds = jnp.zeros((n_agents, n_envs, action_dim), dtype=jnp.float32)
        values = jnp.zeros((n_agents, n_envs), dtype=jnp.float32)
        for t in types:
            idxs = jnp.asarray(type_indices_np[t])
            n_t = len(type_indices_np[t])
            t_flat = all_obs[idxs].reshape(n_t * n_envs, obs_dim)
            t_mean, t_ls, t_val = jax.vmap(
                partial(actor_networks[t].apply, params_bundle["actor"][t])
            )(t_flat)
            means = means.at[idxs].set(t_mean.reshape(n_t, n_envs, action_dim))
            log_stds = log_stds.at[idxs].set(
                jnp.broadcast_to(t_ls, (n_t * n_envs, action_dim)).reshape(
                    n_t, n_envs, action_dim
                )
            )
            values = values.at[idxs].set(t_val.reshape(n_t, n_envs))
        return means, log_stds, values

    def _forward_cost(params_bundle, all_obs):
        values = jnp.zeros(
            (n_agents, n_envs, n_constraints), dtype=jnp.float32
        )
        for t in types:
            idxs = jnp.asarray(type_indices_np[t])
            n_t = len(type_indices_np[t])
            t_flat = all_obs[idxs].reshape(n_t * n_envs, obs_dim)
            t_val = jax.vmap(
                partial(cost_networks[t].apply, params_bundle["cost"][t])
            )(t_flat)
            values = values.at[idxs].set(
                t_val.reshape(n_t, n_envs, n_constraints)
            )
        return values

    def _single_update(runner_state):
        params_bundle, opt_state, log_lambda, env_states, obs_dicts, key = runner_state
        lam = jnp.exp(log_lambda)

        def _env_step(carry, _):
            env_states, obs_dicts, key = carry
            key, act_key, step_key = jax.random.split(key, 3)

            all_obs = jnp.stack([obs_dicts[name] for name in agent_names])
            means, log_stds, values_r = _forward_actor(params_bundle, all_obs)
            values_c = _forward_cost(params_bundle, all_obs)

            flat_mean = means.reshape(n_agents * n_envs, action_dim)
            flat_ls = log_stds.reshape(n_agents * n_envs, action_dim)
            flat_actions = jax.vmap(_sample_action)(
                jax.random.split(act_key, n_agents * n_envs),
                flat_mean,
                flat_ls,
            )
            actions_arr = jnp.clip(
                flat_actions.reshape(n_agents, n_envs, action_dim),
                -1.0,
                1.0,
            )
            actions = {name: actions_arr[i] for i, name in enumerate(agent_names)}

            flat_lp = jax.vmap(_log_prob)(flat_mean, flat_ls, flat_actions)
            log_probs = flat_lp.reshape(n_agents, n_envs)

            step_keys = jax.random.split(step_key, n_envs)
            next_obs_dicts, next_states, rewards_dicts, dones_dicts, info = jax.vmap(
                env.step
            )(step_keys, env_states, actions)

            rewards = jnp.stack([rewards_dicts[name] for name in agent_names])
            dones = dones_dicts["__all__"]
            costs = jnp.take(
                info["constraint_costs"],
                selected_indices_arr,
                axis=1,
            )

            transition = {
                "obs": all_obs,
                "action": actions_arr,
                "reward": rewards,
                "cost": costs,
                "done": dones,
                "value_reward": values_r,
                "value_cost": values_c,
                "log_prob": log_probs,
            }
            return (next_states, next_obs_dicts, key), transition

        (env_states, obs_dicts, key), rollout = jax.lax.scan(
            _env_step,
            (env_states, obs_dicts, key),
            None,
            length=n_steps,
        )

        rollout["cost"] = rollout["cost"] / cost_scale[None, None, :]

        all_obs_final = jnp.stack([obs_dicts[name] for name in agent_names])
        _, _, last_values_r = _forward_actor(params_bundle, all_obs_final)
        last_values_c = _forward_cost(params_bundle, all_obs_final)

        def _gae_reward_step(carry, t):
            gae, next_v = carry
            done_mask = 1.0 - t["done"].astype(jnp.float32)
            delta = t["reward"] + gamma * next_v * done_mask[None, :] - t["value_reward"]
            gae = delta + gamma * gae_lambda * done_mask[None, :] * gae
            return (gae, t["value_reward"]), gae

        _, adv_reward = jax.lax.scan(
            _gae_reward_step,
            (jnp.zeros((n_agents, n_envs), dtype=jnp.float32), last_values_r),
            rollout,
            reverse=True,
        )
        returns_reward = adv_reward + rollout["value_reward"]

        def _gae_cost_step(carry, t):
            gae, next_v = carry
            done_mask = (1.0 - t["done"].astype(jnp.float32))[None, :, None]
            delta = t["cost"][None, :, :] + cost_gamma * next_v * done_mask - t["value_cost"]
            gae = delta + cost_gamma * gae_lambda * done_mask * gae
            return (gae, t["value_cost"]), gae

        _, adv_cost = jax.lax.scan(
            _gae_cost_step,
            (
                jnp.zeros((n_agents, n_envs, n_constraints), dtype=jnp.float32),
                last_values_c,
            ),
            rollout,
            reverse=True,
        )
        returns_cost = adv_cost + rollout["value_cost"]

        adv_reward = (adv_reward - adv_reward.mean()) / (adv_reward.std() + 1e-8)
        adv_cost_mean = adv_cost.mean(axis=(0, 1, 2), keepdims=True)
        adv_cost_std = adv_cost.std(axis=(0, 1, 2), keepdims=True) + 1e-8
        adv_cost = (adv_cost - adv_cost_mean) / adv_cost_std

        batches_by_type: Dict[str, tuple] = {}
        for t in types:
            idxs = jnp.asarray(type_indices_np[t])
            n_t = len(type_indices_np[t])
            tt = n_steps * n_t * n_envs
            batches_by_type[t] = (
                rollout["obs"][:, idxs, :, :].reshape(tt, obs_dim),
                rollout["action"][:, idxs, :, :].reshape(tt, action_dim),
                rollout["log_prob"][:, idxs, :].reshape(tt),
                adv_reward[:, idxs, :].reshape(tt),
                adv_cost[:, idxs, :, :].reshape(tt, n_constraints),
                returns_reward[:, idxs, :].reshape(tt),
                returns_cost[:, idxs, :, :].reshape(tt, n_constraints),
            )

        def _ppo_loss(params_bundle, batches):
            total = jnp.float32(0.0)
            for t in types:
                obs, act, old_lp, adv_r, adv_c, ret_r, ret_c = batches[t]
                t_mean, t_ls, t_val_r = jax.vmap(
                    partial(actor_networks[t].apply, params_bundle["actor"][t])
                )(obs)
                t_ls_bc = jnp.broadcast_to(t_ls, t_mean.shape)
                t_val_c = jax.vmap(
                    partial(cost_networks[t].apply, params_bundle["cost"][t])
                )(obs)

                lp = jax.vmap(_log_prob)(t_mean, t_ls_bc, act)
                ent = _entropy(t_ls)
                ratio = jnp.exp(lp - old_lp)
                clipped = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
                combined_adv = adv_r - jnp.sum(lam * adv_c, axis=-1)
                pi_loss = -jnp.minimum(ratio * combined_adv, clipped * combined_adv).mean()
                vf_loss_r = 0.5 * ((t_val_r - ret_r) ** 2).mean()
                vf_loss_c = 0.5 * ((t_val_c - ret_c) ** 2).mean()
                total = total + pi_loss + vf_coef * (vf_loss_r + vf_loss_c) - ent_coef * ent.mean()
            return total / len(types), total

        def _update_epoch(carry, _):
            params_bundle, opt_state, key = carry
            key, perm_key = jax.random.split(key)
            shuffled: Dict[str, tuple] = {}
            for i_t, t in enumerate(types):
                n_t = len(type_indices_np[t])
                tt = n_steps * n_t * n_envs
                t_pk = jax.random.fold_in(perm_key, i_t)
                perm = jax.random.permutation(t_pk, tt)
                shuffled[t] = jax.tree.map(lambda x: x[perm], batches_by_type[t])
            grad_fn = jax.value_and_grad(_ppo_loss, has_aux=True)
            (loss, _), grads = grad_fn(params_bundle, shuffled)
            updates, opt_state = tx.update(grads, opt_state, params_bundle)
            params_bundle = optax.apply_updates(params_bundle, updates)
            return (params_bundle, opt_state, key), loss

        key, epoch_key = jax.random.split(key)
        (params_bundle, opt_state, _), epoch_losses = jax.lax.scan(
            _update_epoch,
            (params_bundle, opt_state, epoch_key),
            None,
            length=n_epochs,
        )

        mean_cost = jnp.mean(rollout["cost"], axis=(0, 1))
        episode_cost_est = mean_cost * n_steps
        log_lambda = jnp.clip(
            log_lambda + lambda_lr * (episode_cost_est - thresholds),
            -float(config.log_lambda_max),
            float(config.log_lambda_max),
        )

        metrics = {
            "loss": epoch_losses.mean(),
            "mean_reward": jnp.mean(rollout["reward"]),
            "mean_cost_total": jnp.sum(mean_cost),
            "episode_cost_est_total": jnp.sum(episode_cost_est),
            "lambda_total": jnp.sum(jnp.exp(log_lambda)),
            "mean_cost": jnp.sum(mean_cost),
            "episode_cost_est": jnp.sum(episode_cost_est),
            "lambda": jnp.sum(jnp.exp(log_lambda)),
        }
        for idx, name in enumerate(selected_names):
            metrics[f"mean_cost_{name}"] = mean_cost[idx]
            metrics[f"episode_cost_est_{name}"] = episode_cost_est[idx]
            metrics[f"lambda_{name}"] = jnp.exp(log_lambda[idx])

        return (
            params_bundle,
            opt_state,
            log_lambda,
            env_states,
            obs_dicts,
            key,
        ), metrics

    train_step = jax.jit(_single_update)
    env_name = getattr(env, "name", "DistGridMARLEnv") + "[typed_lagrangian]"

    def train_fn(key: chex.PRNGKey) -> TrainResult:
        key, init_key = jax.random.split(key)
        dummy_obs = jnp.zeros((obs_dim,), dtype=jnp.float32)
        actor_params = {
            t: actor_networks[t].init(jax.random.fold_in(init_key, i), dummy_obs)
            for i, t in enumerate(types)
        }
        cost_params = {
            t: cost_networks[t].init(
                jax.random.fold_in(init_key, i + len(types)),
                dummy_obs,
            )
            for i, t in enumerate(types)
        }
        params_bundle = {"actor": actor_params, "cost": cost_params}
        opt_state = tx.init(params_bundle)
        log_lambda = jnp.zeros((n_constraints,), dtype=jnp.float32)

        key, env_key = jax.random.split(key)
        env_keys = jax.random.split(env_key, n_envs)
        obs_dicts, env_states = jax.vmap(env.reset)(env_keys)
        runner_state = (
            params_bundle,
            opt_state,
            log_lambda,
            env_states,
            obs_dicts,
            key,
        )

        metrics_history: Dict[str, List[float]] = {
            "loss": [],
            "mean_reward": [],
            "mean_cost_total": [],
            "episode_cost_est_total": [],
            "lambda_total": [],
            "mean_cost": [],
            "episode_cost_est": [],
            "lambda": [],
        }
        for name in selected_names:
            metrics_history[f"mean_cost_{name}"] = []
            metrics_history[f"episode_cost_est_{name}"] = []
            metrics_history[f"lambda_{name}"] = []

        for _ in range(n_updates):
            runner_state, step_metrics = train_step(runner_state)
            jax.block_until_ready(runner_state[0]["actor"])
            for key_name in metrics_history:
                metrics_history[key_name].append(float(step_metrics[key_name]))

        metrics = {
            key_name: jnp.asarray(values, dtype=jnp.float32)
            for key_name, values in metrics_history.items()
        }
        return TrainResult(
            params=runner_state[0]["actor"],
            metrics=metrics,
            config=config,
            env_name=env_name,
        )

    return train_fn


def make_ippo_act(result) -> Callable:
    """Create a deterministic per-agent act function from an IPPO TrainResult.

    Args:
        result: ``TrainResult`` returned by ``make_ippo_train`` or ``train()``.
                ``result.params`` must be a ``SharedActorCritic`` pytree.
                ``result.config.hidden_dims`` sets the network architecture.

    Returns:
        ``act_fn(obs: Array) -> Array`` — shape ``(action_dim,)`` clipped to
        ``[-1, 1]``.  Suitable for calling per-agent inside an evaluation loop.

    Example::

        act = make_ippo_act(result)
        for name in env.agent_names:
            actions[name] = act(obs_dict[name])
    """
    # Infer action_dim from log_std parameter shape (Flax stores self.param under
    # params["params"]["log_std"] with shape (action_dim,)).
    action_dim = int(result.params["params"]["log_std"].shape[0])
    network = SharedActorCritic(
        hidden_dims=result.config.hidden_dims, action_dim=action_dim
    )

    @jax.jit
    def act_fn(obs: chex.Array) -> chex.Array:
        mean, _, _ = network.apply(result.params, obs)
        return jnp.clip(mean, -1.0, 1.0)

    return act_fn
