"""PureJaxRL PPO — TransGrid Economic Dispatch (OPF).

Demonstrates single-agent RL for power system OPF:
- TransGridEnv: multi-dimensional continuous action (n_units generators)
- Reward: -generation_cost * reward_scale (economic objective)
- LogWrapper: auto-reset + episode return tracking
- Full GPU pipeline: lax.scan rollout + vmap batch environments

Dependencies: jax, flax, optax, chex (all included with powerzoojax)

Run:
    python examples/train_ppo_transgrid.py
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
from powerzoojax.rl import LogWrapper


# ============ Hyperparameters ============

class PPOConfig(NamedTuple):
    lr: float = 3e-4
    n_envs: int = 32
    n_steps: int = 48             # one episode = 48 steps (24h @ 30min)
    n_updates: int = 300
    n_minibatches: int = 4
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


# ============ Networks ============

class ActorCritic(nn.Module):
    """Multi-dimensional continuous actor-critic for OPF dispatch."""
    action_dim: int
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, x):
        # Shared trunk
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)

        # Actor head
        mean = nn.Dense(self.action_dim)(x)
        log_std = self.param(
            "log_std", nn.initializers.constant(-0.5), (self.action_dim,)
        )

        # Critic head
        value = nn.Dense(1)(x)
        value = jnp.squeeze(value, axis=-1)

        return mean, log_std, value


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

class Transition(NamedTuple):
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    value: chex.Array
    log_prob: chex.Array


# ============ PPO Training ============

def make_train(config: PPOConfig):

    # Create environment
    case = create_case5()
    profiles = jnp.ones((48, case.n_loads), dtype=jnp.float32) * 0.5
    params = make_trans_params(case, load_profiles=profiles, max_steps=48)
    env = LogWrapper(TransGridEnv(), params)

    def train(key):
        key, init_key = jax.random.split(key)
        dummy_obs = jnp.zeros((env.obs_size,))
        network = ActorCritic(action_dim=env.num_actions)
        net_params = network.init(init_key, dummy_obs)

        tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.lr),
        )
        opt_state = tx.init(net_params)

        key, env_key = jax.random.split(key)
        env_keys = jax.random.split(env_key, config.n_envs)
        obs_batch, env_state_batch = jax.vmap(env.reset)(env_keys)

        def _update_step(runner_state, _):
            net_params, opt_state, env_state_batch, obs_batch, key = runner_state

            # Collect rollout
            def _env_step(carry, _):
                env_state, obs, key = carry
                key, act_key, step_key = jax.random.split(key, 3)

                mean, log_std, value = jax.vmap(
                    partial(network.apply, net_params)
                )(obs)
                action = jax.vmap(gaussian_sample)(
                    jax.random.split(act_key, config.n_envs), mean, log_std
                )
                action = jnp.clip(action, -1.0, 1.0)
                log_prob = jax.vmap(gaussian_log_prob)(mean, log_std, action)

                step_keys = jax.random.split(step_key, config.n_envs)
                next_obs, next_env_state, reward, done, info = jax.vmap(
                    env.step
                )(step_keys, env_state, action)

                transition = Transition(
                    obs=obs, action=action, reward=reward,
                    done=done, value=value, log_prob=log_prob,
                )
                return (next_env_state, next_obs, key), transition

            (env_state_batch, obs_batch, key), rollout = jax.lax.scan(
                _env_step,
                (env_state_batch, obs_batch, key),
                None,
                length=config.n_steps,
            )

            # Bootstrap
            _, _, last_value = jax.vmap(
                partial(network.apply, net_params)
            )(obs_batch)

            # GAE
            def _compute_gae(carry, transition):
                gae, next_value = carry
                delta = (
                    transition.reward
                    + config.gamma * next_value * (1 - transition.done)
                    - transition.value
                )
                gae = delta + config.gamma * config.gae_lambda * (1 - transition.done) * gae
                return (gae, transition.value), gae

            _, advantages = jax.lax.scan(
                _compute_gae,
                (jnp.zeros(config.n_envs), last_value),
                rollout,
                reverse=True,
            )
            returns = advantages + rollout.value

            # PPO update
            def _ppo_loss(net_params, batch):
                obs, action, old_log_prob, advantage, return_ = batch
                mean, log_std, value = network.apply(net_params, obs)
                log_prob = gaussian_log_prob(mean, log_std, action)
                entropy = gaussian_entropy(log_std)

                ratio = jnp.exp(log_prob - old_log_prob)
                clipped = jnp.clip(ratio, 1 - config.clip_eps, 1 + config.clip_eps)
                pi_loss = -jnp.minimum(ratio * advantage, clipped * advantage).mean()
                vf_loss = 0.5 * ((value - return_) ** 2).mean()
                loss = pi_loss + config.vf_coef * vf_loss - config.ent_coef * entropy.mean()
                return loss, (pi_loss, vf_loss, entropy.mean())

            def _update_epoch(carry, _):
                net_params, opt_state, key = carry
                key, perm_key = jax.random.split(key)

                batch_size = config.n_steps * config.n_envs
                flat = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]),
                    (rollout.obs, rollout.action, rollout.log_prob,
                     advantages, returns),
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

            mean_return = jnp.mean(rollout.reward)
            runner_state = (net_params, opt_state, env_state_batch, obs_batch, key)
            return runner_state, {"loss": epoch_losses.mean(), "mean_reward": mean_return}

        runner_state = (net_params, opt_state, env_state_batch, obs_batch, key)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, length=config.n_updates
        )
        return runner_state, metrics

    return jax.jit(train)


# ============ Main ============

def main():
    print("=" * 60)
    print("PureJaxRL PPO — TransGrid Economic Dispatch")
    print("=" * 60)

    config = PPOConfig()
    case = create_case5()
    print(f"\nGrid: case5 ({case.n_nodes} buses, {case.n_lines} lines, {case.n_units} units)")
    print(f"Config: n_envs={config.n_envs}, n_steps={config.n_steps}, n_updates={config.n_updates}")

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
    losses = metrics["loss"]
    print(f"\n{'Update':>6} | {'Mean Reward':>12} | {'Loss':>10}")
    print("-" * 35)
    n = len(rewards)
    indices = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    for i in indices:
        print(f"{i:>6} | {float(rewards[i]):>12.4f} | {float(losses[i]):>10.4f}")

    print(f"\nReward: {float(rewards[0]):.4f} -> {float(rewards[-1]):.4f}")


if __name__ == "__main__":
    main()
