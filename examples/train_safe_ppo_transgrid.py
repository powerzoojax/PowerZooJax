"""CMDP PPO-Lagrangian — TransGrid Safe Economic Dispatch.

Demonstrates constrained MDP (CMDP) training with PowerZooJax:
- SafeRLWrapper exposes selected CMDP costs directly
- PPO-Lagrangian: max reward subject to E[episode_cost] <= cost_threshold
- Lagrange multiplier lambda auto-adjusts cost penalty weight
- 6-tuple step output: (obs, state, reward, costs, done, info)

The key insight: reward and cost are INDEPENDENT channels.
- reward = -generation_cost (economic objective)
- costs = [thermal_overload] for this single-constraint example
- PPO-Lagrangian balances both via adaptive lambda

Dependencies: jax, flax, optax, chex (all included with powerzoojax)
No external SafeRL library needed.

Run:
    python examples/train_safe_ppo_transgrid.py
"""

import os
import sys
import time
from functools import partial
from typing import NamedTuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import chex

from powerzoojax.case import create_case5
from powerzoojax.envs.grid.trans import TransGridEnv, make_trans_params
from powerzoojax.rl import SafeRLWrapper


# ============ Hyperparameters ============

class CMDPConfig(NamedTuple):
    lr: float = 3e-4
    lambda_lr: float = 5e-3       # Lagrange multiplier learning rate
    cost_threshold: float = 0.0   # strict max allowable episode cost
    n_envs: int = 32
    n_steps: int = 48
    n_updates: int = 300
    n_minibatches: int = 4
    n_epochs: int = 4
    gamma: float = 0.99
    cost_gamma: float = 1.0       # cost discount (typically 1.0 for cumulative)
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


# ============ Networks ============

class SafeActorCritic(nn.Module):
    """Actor-Critic with dual value heads (reward + cost)."""
    action_dim: int
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)

        # Actor
        mean = nn.Dense(self.action_dim)(x)
        log_std = self.param(
            "log_std", nn.initializers.constant(-0.5), (self.action_dim,)
        )

        # Reward critic
        v_reward = nn.Dense(1)(x)
        v_reward = jnp.squeeze(v_reward, axis=-1)

        # Cost critic
        v_cost = nn.Dense(1)(x)
        v_cost = jnp.squeeze(v_cost, axis=-1)

        return mean, log_std, v_reward, v_cost


# ============ Gaussian Policy Utils ============

def gaussian_log_prob(mean, log_std, action):
    std = jnp.exp(log_std)
    var = std ** 2
    log_p = -0.5 * (((action - mean) ** 2) / var + 2 * log_std + jnp.log(2 * jnp.pi))
    return jnp.sum(log_p, axis=-1)


def gaussian_entropy(log_std):
    return jnp.sum(0.5 + 0.5 * jnp.log(2 * jnp.pi) + log_std, axis=-1)


def gaussian_sample(key, mean, log_std):
    std = jnp.exp(log_std)
    return mean + std * jax.random.normal(key, shape=mean.shape)


# ============ Transition ============

class SafeTransition(NamedTuple):
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    cost: chex.Array              # per-step cost signal
    done: chex.Array
    v_reward: chex.Array
    v_cost: chex.Array
    log_prob: chex.Array


# ============ PPO-Lagrangian Training ============

def make_train(config: CMDPConfig):

    case = create_case5()
    profiles = jnp.ones((48, case.n_loads), dtype=jnp.float32) * 0.5
    params = make_trans_params(case, load_profiles=profiles, max_steps=48)
    env = SafeRLWrapper(
        TransGridEnv(),
        params,
        selected_names=("thermal_overload",),
        cost_thresholds=(config.cost_threshold,),
    )

    def train(key):
        key, init_key = jax.random.split(key)
        dummy_obs = jnp.zeros((env.obs_size,))
        network = SafeActorCritic(action_dim=env.num_actions)
        net_params = network.init(init_key, dummy_obs)

        tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.lr),
        )
        opt_state = tx.init(net_params)

        # Lagrange multiplier (log-parameterized for positivity)
        log_lambda = jnp.float32(0.0)

        key, env_key = jax.random.split(key)
        env_keys = jax.random.split(env_key, config.n_envs)
        obs_batch, env_state_batch = jax.vmap(env.reset)(env_keys)

        def _update_step(runner_state, _):
            net_params, opt_state, log_lambda, env_state_batch, obs_batch, key = runner_state
            lam = jnp.exp(log_lambda)  # Current lambda

            # --- Rollout ---
            def _env_step(carry, _):
                env_state, obs, key = carry
                key, act_key, step_key = jax.random.split(key, 3)

                mean, log_std, v_reward, v_cost = jax.vmap(
                    partial(network.apply, net_params)
                )(obs)
                action = jax.vmap(gaussian_sample)(
                    jax.random.split(act_key, config.n_envs), mean, log_std
                )
                action = jnp.clip(action, -1.0, 1.0)
                log_prob = jax.vmap(gaussian_log_prob)(mean, log_std, action)

                step_keys = jax.random.split(step_key, config.n_envs)
                # SafeRLWrapper returns selected CMDP costs as a vector.
                next_obs, next_env_state, reward, costs, done, info = jax.vmap(
                    env.step
                )(step_keys, env_state, action)
                cost = jnp.squeeze(costs, axis=-1)

                transition = SafeTransition(
                    obs=obs, action=action, reward=reward, cost=cost,
                    done=done, v_reward=v_reward, v_cost=v_cost,
                    log_prob=log_prob,
                )
                return (next_env_state, next_obs, key), transition

            (env_state_batch, obs_batch, key), rollout = jax.lax.scan(
                _env_step,
                (env_state_batch, obs_batch, key),
                None,
                length=config.n_steps,
            )

            # Bootstrap values
            _, _, last_v_reward, last_v_cost = jax.vmap(
                partial(network.apply, net_params)
            )(obs_batch)

            # --- GAE for reward ---
            def _gae_reward(carry, t):
                gae, nv = carry
                delta = t.reward + config.gamma * nv * (1 - t.done) - t.v_reward
                gae = delta + config.gamma * config.gae_lambda * (1 - t.done) * gae
                return (gae, t.v_reward), gae

            _, adv_reward = jax.lax.scan(
                _gae_reward,
                (jnp.zeros(config.n_envs), last_v_reward),
                rollout,
                reverse=True,
            )
            returns_reward = adv_reward + rollout.v_reward

            # --- GAE for cost ---
            def _gae_cost(carry, t):
                gae, nv = carry
                delta = t.cost + config.cost_gamma * nv * (1 - t.done) - t.v_cost
                gae = delta + config.cost_gamma * config.gae_lambda * (1 - t.done) * gae
                return (gae, t.v_cost), gae

            _, adv_cost = jax.lax.scan(
                _gae_cost,
                (jnp.zeros(config.n_envs), last_v_cost),
                rollout,
                reverse=True,
            )
            returns_cost = adv_cost + rollout.v_cost

            # --- PPO-Lagrangian loss ---
            def _ppo_loss(net_params, batch):
                obs, action, old_log_prob, adv_r, adv_c, ret_r, ret_c = batch
                mean, log_std, v_r, v_c = network.apply(net_params, obs)
                log_prob = gaussian_log_prob(mean, log_std, action)
                entropy = gaussian_entropy(log_std)

                ratio = jnp.exp(log_prob - old_log_prob)
                # Lagrangian advantage: A_reward - lambda * A_cost
                combined_adv = adv_r - lam * adv_c
                clipped = jnp.clip(ratio, 1 - config.clip_eps, 1 + config.clip_eps)
                pi_loss = -jnp.minimum(ratio * combined_adv, clipped * combined_adv).mean()

                vf_loss_r = 0.5 * ((v_r - ret_r) ** 2).mean()
                vf_loss_c = 0.5 * ((v_c - ret_c) ** 2).mean()
                vf_loss = vf_loss_r + vf_loss_c

                loss = pi_loss + config.vf_coef * vf_loss - config.ent_coef * entropy.mean()
                return loss, (pi_loss, vf_loss, entropy.mean())

            def _update_epoch(carry, _):
                net_params, opt_state, key = carry
                key, perm_key = jax.random.split(key)

                batch_size = config.n_steps * config.n_envs
                flat = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]),
                    (rollout.obs, rollout.action, rollout.log_prob,
                     adv_reward, adv_cost, returns_reward, returns_cost),
                )

                permutation = jax.random.permutation(perm_key, batch_size)
                shuffled = jax.tree.map(lambda x: x[permutation], flat)
                minibatch_size = batch_size // config.n_minibatches
                minibatches = jax.tree.map(
                    lambda x: x.reshape((config.n_minibatches, minibatch_size) + x.shape[1:]),
                    shuffled,
                )

                def _update_minibatch(carry, batch):
                    net_params, opt_state = carry
                    grad_fn = jax.value_and_grad(_ppo_loss, has_aux=True)
                    (loss, aux), grads = grad_fn(net_params, batch)
                    updates, opt_state = tx.update(grads, opt_state, net_params)
                    net_params = optax.apply_updates(net_params, updates)
                    return (net_params, opt_state), loss

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

            # --- Update Lagrange multiplier ---
            # lambda increases if mean episode cost > threshold, decreases otherwise
            mean_cost = jnp.mean(rollout.cost)  # mean per-step cost
            episode_cost_est = mean_cost * config.n_steps  # estimated episode cost
            # Dual gradient ascent: log_lambda += lr * (episode_cost - threshold)
            log_lambda = log_lambda + config.lambda_lr * (episode_cost_est - config.cost_threshold)
            # Clamp to prevent extreme values
            log_lambda = jnp.clip(log_lambda, -5.0, 5.0)

            metrics = {
                "loss": epoch_losses.mean(),
                "mean_reward": jnp.mean(rollout.reward),
                "mean_cost": mean_cost,
                "episode_cost_est": episode_cost_est,
                "lambda": jnp.exp(log_lambda),
            }

            runner_state = (net_params, opt_state, log_lambda, env_state_batch, obs_batch, key)
            return runner_state, metrics

        runner_state = (net_params, opt_state, log_lambda, env_state_batch, obs_batch, key)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, length=config.n_updates
        )
        return runner_state, metrics

    return jax.jit(train)


# ============ Main ============

def main():
    print("=" * 60)
    print("CMDP PPO-Lagrangian — TransGrid Safe Economic Dispatch")
    print("=" * 60)

    config = CMDPConfig()
    print(f"\nCost threshold: {config.cost_threshold}")
    print(f"Config: n_envs={config.n_envs}, n_updates={config.n_updates}")
    print(f"Lambda LR: {config.lambda_lr}")

    train_fn = make_train(config)
    key = jax.random.PRNGKey(42)

    print("\nCompiling + training...")
    t0 = time.time()
    runner_state, metrics = train_fn(key)
    jax.block_until_ready(metrics)
    total_time = time.time() - t0

    total_steps = config.n_envs * config.n_steps * config.n_updates
    print(f"\nTraining complete!")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Wall time: {total_time:.2f}s")
    print(f"  Throughput: {total_steps / total_time:,.0f} steps/s")

    rewards = metrics["mean_reward"]
    costs = metrics["mean_cost"]
    ep_costs = metrics["episode_cost_est"]
    lambdas = metrics["lambda"]

    print(f"\n{'Update':>6} | {'Reward':>8} | {'Cost/step':>10} | {'Ep Cost':>8} | {'Lambda':>8}")
    print("-" * 55)
    n = len(rewards)
    for i in [0, n // 4, n // 2, 3 * n // 4, n - 1]:
        print(f"{i:>6} | {float(rewards[i]):>8.4f} | {float(costs[i]):>10.4f} | "
              f"{float(ep_costs[i]):>8.2f} | {float(lambdas[i]):>8.4f}")

    print(f"\nFinal: reward={float(rewards[-1]):.4f}, "
          f"ep_cost={float(ep_costs[-1]):.2f} (threshold={config.cost_threshold}), "
          f"lambda={float(lambdas[-1]):.4f}")

    # Safety check
    final_safe = float(ep_costs[-1]) <= config.cost_threshold * 1.5
    print(f"Safety: {'WITHIN budget' if final_safe else 'EXCEEDS budget'}")


if __name__ == "__main__":
    main()
