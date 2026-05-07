"""JAX utility functions for GPU-accelerated RL pipelines.

Core functions:
- split_key_for_envs: Split PRNG key for parallel environments.
- batch_reset: vmap-based parallel reset.
- batch_step: vmap-based parallel step (with auto-reset).
- scan_rollout: lax.scan-based trajectory collection.
"""

from typing import Tuple, Any, Dict

import jax
import chex


def split_key_for_envs(key: chex.PRNGKey, n_envs: int) -> chex.PRNGKey:
    """Split a PRNG key for multiple parallel environments.

    Args:
        key: Base PRNG key.
        n_envs: Number of parallel environments.

    Returns:
        Array of keys with shape (n_envs, 2).
    """
    return jax.random.split(key, n_envs)


def batch_reset(env, keys: chex.PRNGKey, params) -> Tuple[chex.Array, Any]:
    """Reset multiple environments in parallel using vmap.

    Args:
        env: Environment instance with .reset(key, params).
        keys: PRNG keys (n_envs, 2).
        params: Shared environment parameters (broadcast).

    Returns:
        obs: Observations (n_envs, obs_dim).
        states: Batched environment states.
    """
    return jax.vmap(env.reset, in_axes=(0, None))(keys, params)


def batch_step(
    env,
    keys: chex.PRNGKey,
    states,
    actions: chex.Array,
    params,
) -> Tuple[chex.Array, Any, chex.Array, chex.Array, chex.Array, Dict[str, Any]]:
    """Step multiple environments in parallel using vmap (with auto-reset).

    Uses env.step_auto_reset so that done episodes are automatically reset,
    making this compatible with lax.scan fixed-length rollout.

    Args:
        env: Environment instance with .step_auto_reset(key, state, action, params).
        keys: PRNG keys (n_envs, 2).
        states: Batched states.
        actions: Batched actions (n_envs, action_dim).
        params: Shared environment parameters (broadcast).

    Returns:
        obs, states, rewards, costs, dones, infos  — all batched along axis 0.
    """
    return jax.vmap(env.step_auto_reset, in_axes=(0, 0, 0, None))(
        keys, states, actions, params
    )


def scan_rollout(
    env,
    key: chex.PRNGKey,
    init_state,
    params,
    actions: chex.Array,
) -> Tuple[Any, chex.Array, chex.Array, chex.Array, chex.Array, Dict[str, Any]]:
    """Collect a trajectory with lax.scan (no Python loop).

    Uses env.step_auto_reset so episodes auto-reset at done boundaries.

    Args:
        env: Environment instance.
        key: PRNG key (will be split per step).
        init_state: Initial environment state (from env.reset).
        params: Environment parameters.
        actions: Pre-defined action sequence, shape (T, *action_shape).

    Returns:
        final_state: State after the last step.
        obs_traj: Observations (T, obs_dim).
        reward_traj: Rewards (T,).
        cost_traj: Constraint-cost vectors (T, k).
        done_traj: Done flags (T,).
        info_traj: Info dicts with batched leaves (T, ...).
    """
    T = actions.shape[0]
    keys = jax.random.split(key, T)

    def scan_fn(state, xs):
        k, action = xs
        obs, new_state, reward, costs, done, info = env.step_auto_reset(
            k, state, action, params
        )
        return new_state, (obs, reward, costs, done, info)

    final_state, (obs_traj, reward_traj, cost_traj, done_traj, info_traj) = jax.lax.scan(
        scan_fn, init_state, (keys, actions)
    )
    return final_state, obs_traj, reward_traj, cost_traj, done_traj, info_traj
