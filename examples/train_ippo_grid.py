"""Multi-Agent IPPO — Grid Economic Dispatch.

Demonstrates multi-agent RL with PowerZooJax GridMARLEnv:
- Each generator unit is an independent agent
- Independent PPO (IPPO): parameter-sharing, single ActorCritic network
- JaxMARL-compatible dict-based obs/actions/rewards interface
- Full GPU pipeline: lax.scan rollout + vmap

Resources (batteries etc.) can be added via make_battery_bundle() and
passed to make_trans_params(resources=(bundle,)).

Quick start (preset-based):
    python -c "from powerzoojax.rl import train; result = train('case5-ippo')"

Run full example:
    python examples/train_ippo_grid.py

With JaxMARL installed, the GridMARLEnv can be used directly with
JaxMARL's IPPO/MAPPO/QMIX algorithms.
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax

from powerzoojax.case import create_case5
from powerzoojax.envs.grid.trans import TransGridEnv, make_trans_params
from powerzoojax.rl import GridMARLEnv
from powerzoojax.rl.ippo import make_ippo_train
from powerzoojax.rl.config import TrainConfig


def main():
    print("=" * 60)
    print("Multi-Agent IPPO — Grid Economic Dispatch")
    print("=" * 60)

    import jax.numpy as jnp
    case = create_case5()
    profiles = jnp.ones((48, case.n_loads), dtype=jnp.float32) * 0.5
    grid_params = make_trans_params(case, load_profiles=profiles, max_steps=48)
    env = GridMARLEnv(TransGridEnv(), grid_params)

    config = TrainConfig(
        algo="ippo",
        total_timesteps=200_000,
        num_envs=16,
        n_steps=48,
        n_epochs=4,
        hidden_dims=(64, 64),
    )

    print(f"\nGrid: case5 ({case.n_nodes} buses, {case.n_lines} lines, {env.num_agents} agents)")
    print(f"Config: num_envs={config.num_envs}, n_steps={config.n_steps}, "
          f"total_timesteps={config.total_timesteps:,}")

    train_fn = make_ippo_train(env, config)
    key = jax.random.PRNGKey(42)

    print("\nCompiling + training...")
    t0 = time.time()
    result = train_fn(key)
    jax.block_until_ready(result.metrics)
    total_time = time.time() - t0

    total_steps = config.total_timesteps
    print(f"\nTraining complete!")
    print(f"  Total timesteps: {total_steps:,}")
    print(f"  Wall time: {total_time:.2f}s")
    print(f"  Throughput: {total_steps / total_time:,.0f} timesteps/s")

    rewards = result.metrics["mean_reward"]
    losses = result.metrics["loss"]
    print(f"\n{'Update':>6} | {'Mean Reward':>12} | {'Loss':>10}")
    print("-" * 35)
    n = len(rewards)
    for i in [0, n // 4, n // 2, 3 * n // 4, n - 1]:
        print(f"{i:>6} | {float(rewards[i]):>12.4f} | {float(losses[i]):>10.4f}")

    print(f"\nReward: {float(rewards[0]):.4f} -> {float(rewards[-1]):.4f}")
    print(f"\nFor battery+MARL, see preset 'case5-ippo-battery':")
    print("  from powerzoojax.rl import train")
    print("  result = train('case5-ippo-battery')")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
