"""Tests for DSO PenaltyRewardWrapper integration (without frozen train YAMLs).

Test groups:
  1. Wrapper construction — PenaltyRewardWrapper wraps DistGridEnv cleanly.
  2. Reward penalisation — wrapper correctly subtracts λ*scale*Σ(costs).
"""

from __future__ import annotations

import numpy as np


# ── 1. Wrapper construction ────────────────────────────────────────────────────


def test_dso_penalty_wrapper_construction():
    """PenaltyRewardWrapper wraps DistGridEnv; reset returns correct obs shape."""
    import jax
    from powerzoojax.envs.grid.dist import DistGridEnv
    from powerzoojax.rl.wrappers import LogWrapper, PenaltyRewardWrapper
    from powerzoojax.tasks.dso import make_dso_params

    params = make_dso_params()
    env = DistGridEnv()
    wrapper = LogWrapper(
        PenaltyRewardWrapper(env, penalty_lambda=10.0, reward_scale=0.001),
        params,
    )

    key = jax.random.PRNGKey(0)
    obs, _state = wrapper.reset(key)
    assert obs.shape[0] == wrapper.obs_size


def test_dso_penalty_wrapper_step_runs():
    """PenaltyRewardWrapper.step returns 6-tuple without crashing."""
    import jax
    import jax.numpy as jnp
    from powerzoojax.envs.grid.dist import DistGridEnv
    from powerzoojax.rl.wrappers import PenaltyRewardWrapper
    from powerzoojax.tasks.dso import make_dso_params

    params = make_dso_params()
    env = DistGridEnv()
    wrapped = PenaltyRewardWrapper(env, penalty_lambda=10.0, reward_scale=0.001)

    key = jax.random.PRNGKey(7)
    key_r, key_s = jax.random.split(key)
    _, state = env.reset(key_r, params)

    action = jnp.zeros(env.action_space(params).shape[0], dtype=jnp.float32)
    result = wrapped.step(key_s, state, action, params)
    assert len(result) == 6, "step should return (obs, state, reward, costs, done, info)"


# ── 2. Reward penalisation ─────────────────────────────────────────────────────


def test_dso_penalty_wrapper_subtracts_costs():
    """PenaltyRewardWrapper reduces the reward by exactly lambda*scale*sum(costs)."""
    import jax
    import jax.numpy as jnp
    from powerzoojax.envs.grid.dist import DistGridEnv
    from powerzoojax.rl.wrappers import PenaltyRewardWrapper
    from powerzoojax.tasks.dso import make_dso_params

    params = make_dso_params()
    env = DistGridEnv()

    penalty_lambda = 10.0
    reward_scale = 0.001
    wrapped = PenaltyRewardWrapper(env, penalty_lambda=penalty_lambda, reward_scale=reward_scale)

    key = jax.random.PRNGKey(42)
    key_r, key_s = jax.random.split(key)
    _, state = env.reset(key_r, params)

    # Zero-action provokes non-trivial constraint costs (all FlexLoads idle).
    action = jnp.zeros(env.action_space(params).shape[0], dtype=jnp.float32)

    _ow, _os, rew_w, _cw, _dw, info_w = wrapped.step(key_s, state, action, params)
    _rw, _rs, rew_r, costs_r, _dr, _ir = env.step(key_s, state, action, params)

    expected_penalty = float(penalty_lambda * reward_scale * jnp.sum(costs_r))
    actual_gap = float(info_w["unpenalized_reward"]) - float(rew_w)

    assert np.isclose(actual_gap, expected_penalty, atol=1e-5), (
        f"reward gap {actual_gap:.6f} != expected penalty {expected_penalty:.6f}"
    )


def test_dso_penalty_zero_lambda_leaves_reward_unchanged():
    """With penalty_lambda=0, penalised reward == unpenalised reward."""
    import jax
    import jax.numpy as jnp
    from powerzoojax.envs.grid.dist import DistGridEnv
    from powerzoojax.rl.wrappers import PenaltyRewardWrapper
    from powerzoojax.tasks.dso import make_dso_params

    params = make_dso_params()
    env = DistGridEnv()
    wrapped = PenaltyRewardWrapper(env, penalty_lambda=0.0, reward_scale=0.001)

    key = jax.random.PRNGKey(99)
    key_r, key_s = jax.random.split(key)
    _, state = env.reset(key_r, params)
    action = jnp.zeros(env.action_space(params).shape[0], dtype=jnp.float32)

    _ow, _os, rew_w, _cw, _dw, info_w = wrapped.step(key_s, state, action, params)
    assert np.isclose(
        float(info_w["unpenalized_reward"]),
        float(rew_w),
        atol=1e-6,
    ), "zero lambda should leave reward unchanged"


def test_dso_penalty_higher_lambda_lowers_reward():
    """Higher λ produces strictly lower (or equal) penalised reward given same actions."""
    import jax
    import jax.numpy as jnp
    from powerzoojax.envs.grid.dist import DistGridEnv
    from powerzoojax.rl.wrappers import PenaltyRewardWrapper
    from powerzoojax.tasks.dso import make_dso_params

    params = make_dso_params()
    env = DistGridEnv()

    key = jax.random.PRNGKey(11)
    key_r, key_s = jax.random.split(key)
    _, state = env.reset(key_r, params)
    action_size = env.action_space(params).shape[0]
    action = jnp.zeros(action_size, dtype=jnp.float32)

    results = {}
    for lam in [1.0, 10.0, 100.0]:
        wrapped = PenaltyRewardWrapper(env, penalty_lambda=lam, reward_scale=0.001)
        _, _, rew, costs, _, _ = wrapped.step(key_s, state, action, params)
        results[lam] = float(rew)

    # If any costs are non-zero, rewards should be strictly ordered.
    _ow, _os, _rw, costs_r, _dw, _iw = env.step(key_s, state, action, params)
    if float(jnp.sum(costs_r)) > 0.0:
        assert results[1.0] >= results[10.0] >= results[100.0], (
            f"Higher lambda should produce lower/equal reward; got {results}"
        )
