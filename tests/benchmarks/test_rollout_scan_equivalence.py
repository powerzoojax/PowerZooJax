"""Numerical equivalence tests for scan-based rollout helpers.

When eval helpers were converted from a Python ``for`` loop to ``jax.lax.scan``,
each step's scalar collection moved from per-step host syncs to a final fused
launch. These tests pin per-step values to ensure the new path produces
bit-identical (within float32 tolerance) trajectories vs the env's own
``reset``/``step`` API driven by an explicit Python loop.

Pattern:

  1. Fix ``key`` and a deterministic policy.
  2. Run env.reset/step in a tight Python loop, collecting per-step scalars.
  3. Run the production rollout helper (now scan-based).
  4. Assert all per-step series match within ``atol=1e-5``.

Add new tasks here as their rollouts are converted.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ATOL = 1e-5


# ---------------------------------------------------------------------------
# DSO
# ---------------------------------------------------------------------------

def _python_rollout_dso(env, params, key, policy_fn, max_steps):
    """Reference Python-loop rollout, mirrors the pre-scan implementation."""
    obs, state = env.reset(key, params)
    out = {
        "rewards": [],
        "losses": [],
        "violations": [],
        "curtailed": [],
        "shifted": [],
        "shift_in": [],
    }
    k = key
    for _ in range(max_steps):
        k, k_step, k_pol = jax.random.split(k, 3)
        action = policy_fn(obs, state, k_pol)
        obs, state, reward, _costs, done, info = env.step(k_step, state, action, params)
        out["rewards"].append(np.asarray(reward))
        out["losses"].append(np.asarray(info["p_loss_MW"]))
        out["violations"].append(np.asarray(info["n_violations"]))
        out["curtailed"].append(np.asarray(info["resource_curtailed_mw"]))
        out["shifted"].append(np.asarray(info["resource_shift_out_mw"]))
        out["shift_in"].append(np.asarray(info["resource_shift_in_mw"]))
    return {k_: np.stack(v) for k_, v in out.items()}


def test_rollout_dso_scan_matches_python_loop():
    from powerzoojax.envs.grid.dist import DistGridEnv
    from powerzoojax.tasks.dso import make_dso_params, rollout_dso

    env = DistGridEnv()
    params = make_dso_params()
    action_dim = sum(b.action_dim for b in params.resources)

    def zero_policy(obs, state, key):
        return jnp.zeros(action_dim)

    key = jax.random.PRNGKey(0xD50)
    max_steps = int(params.max_steps)

    expected = _python_rollout_dso(env, params, key, zero_policy, max_steps)
    actual = rollout_dso(env, params, key, zero_policy, max_steps=max_steps)

    for field in expected:
        a = np.asarray(actual[field])
        e = expected[field]
        assert a.shape == e.shape, f"{field}: shape {a.shape} vs {e.shape}"
        np.testing.assert_allclose(
            a, e, atol=ATOL,
            err_msg=f"DSO rollout mismatch on '{field}'",
        )


def test_rollout_dso_scan_jits_and_runs_under_vmap():
    """L0 contract: scan-based rollout must be jit + vmap clean."""
    from powerzoojax.envs.grid.dist import DistGridEnv
    from powerzoojax.tasks.dso import make_dso_params, rollout_dso

    env = DistGridEnv()
    params = make_dso_params()
    action_dim = sum(b.action_dim for b in params.resources)

    def zero_policy(obs, state, key):
        return jnp.zeros(action_dim)

    @jax.jit
    def run_one(key):
        return rollout_dso(env, params, key, zero_policy)

    keys = jax.random.split(jax.random.PRNGKey(7), 4)
    batched = jax.vmap(run_one)(keys)
    assert batched["rewards"].shape == (4, int(params.max_steps))
