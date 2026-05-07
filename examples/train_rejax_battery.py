"""Rejax SAC — Battery SOC Tracking.

Demonstrates PowerZooJax integration with Rejax (https://github.com/keraJLi/rejax).
Rejax provides pre-built JAX RL algorithms (PPO, SAC, DQN, TD3, etc.)
that work directly with LogWrapper's interface.

Prerequisites:
    pip install rejax

If rejax is not installed, this script shows the equivalent manual setup.

Run:
    python examples/train_rejax_battery.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax
import jax.numpy as jnp

from powerzoojax.envs.resource.battery import BatteryEnv, make_battery_params
from powerzoojax.rl import LogWrapper


def main():
    print("=" * 60)
    print("Rejax SAC — Battery SOC Tracking")
    print("=" * 60)

    # Create wrapped environment (Rejax-compatible interface)
    env = LogWrapper(BatteryEnv(), make_battery_params(max_steps=48))

    print(f"\nEnvironment:")
    print(f"  obs_size: {env.obs_size}")
    print(f"  num_actions: {env.num_actions}")
    print(f"  action_size: {env.action_size}")

    # Verify Rejax compatibility
    print(f"\nRejax compatibility check:")
    print(f"  env.obs_size = {env.obs_size}  (int, required)")
    print(f"  env.num_actions = {env.num_actions}  (int, required)")
    print(f"  env.action_size = {env.action_size}  (int, required)")

    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    print(f"  reset(key) -> obs.shape={obs.shape}, state=LogEnvState  (OK)")

    k1, k2 = jax.random.split(key)
    action = env.action_space().sample(k2)
    obs, state, reward, done, info = env.step(k1, state, action)
    print(f"  step(key, state, action) -> 5-tuple  (OK)")
    print(f"  info keys: {list(info.keys())}")

    try:
        import rejax
        print(f"\n  rejax version: {rejax.__version__} (installed)")
        _run_rejax_sac(env)
    except ImportError:
        print(f"\n  rejax not installed. Install with: pip install rejax")
        print("  Showing manual SAC-compatible rollout instead...")
        _run_manual_rollout(env)


def _run_rejax_sac(env):
    """Run SAC training with Rejax."""
    from rejax import SAC

    print("\nTraining SAC with Rejax...")
    algo = SAC.create(
        env=env,
        total_timesteps=50_000,
        eval_freq=10_000,
        num_envs=16,
        learning_rate=3e-4,
    )

    key = jax.random.PRNGKey(42)
    t0 = time.time()
    train_state, logs = algo.train(key)
    elapsed = time.time() - t0

    print(f"\nTraining complete in {elapsed:.2f}s")
    if "eval_returns" in logs:
        print(f"Final eval return: {float(logs['eval_returns'][-1]):.4f}")


def _run_manual_rollout(env):
    """Demonstrate LogWrapper's compatibility with JAX RL patterns."""
    print("\nRunning manual rollout (lax.scan + vmap)...")

    n_envs = 32
    n_steps = 480  # 10 episodes of 48 steps

    @jax.jit
    def batch_rollout(key):
        env_keys = jax.random.split(key, n_envs)
        obs_batch, state_batch = jax.vmap(env.reset)(env_keys)

        def scan_step(carry, _):
            state, obs, key = carry
            key, k1, k2 = jax.random.split(key, 3)
            # Random policy
            actions = jax.vmap(lambda k: env.action_space().sample(k))(
                jax.random.split(k2, n_envs)
            )
            next_obs, next_state, reward, done, info = jax.vmap(env.step)(
                jax.random.split(k1, n_envs), state, actions
            )
            return (next_state, next_obs, key), {
                "reward": reward,
                "done": done,
                "returned_episode_returns": info["returned_episode_returns"],
                "returned_episode": info["returned_episode"],
            }

        _, data = jax.lax.scan(
            scan_step, (state_batch, obs_batch, key), None, length=n_steps
        )
        return data

    key = jax.random.PRNGKey(0)
    t0 = time.time()
    data = batch_rollout(key)
    jax.block_until_ready(data["reward"])
    elapsed = time.time() - t0

    total_steps = n_envs * n_steps
    print(f"  {total_steps:,} steps in {elapsed:.4f}s")
    print(f"  Throughput: {total_steps / elapsed:,.0f} steps/s")

    # Episode statistics
    ep_mask = data["returned_episode"]  # (n_steps, n_envs)
    ep_returns = data["returned_episode_returns"]
    completed = ep_mask.sum()
    if completed > 0:
        mean_return = jnp.where(ep_mask, ep_returns, 0.0).sum() / completed
        print(f"  Completed episodes: {int(completed)}")
        print(f"  Mean episode return: {float(mean_return):.4f}")


if __name__ == "__main__":
    main()
