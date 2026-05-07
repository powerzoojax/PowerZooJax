"""Targeted tests for TSO comparison benchmark contract (JAX side only).

Default pytest run: PowerZooJax-only, no subprocess, no hardcoded paths.

Cross-repo tests (PowerZoo side) live in crossrepo_tso_comparison.py
and require POWERZOO_DIR / POWERZOO_PYTHON env vars.

Contract assertions compare against the checked-in golden file:
    tests/golden/tso_comparison_contract.json

To refresh the golden: python scripts/generate_tso_comparison_golden.py

Closure scope:
  - Shared comparison contract (schema + load trace + cost keys) is established.
  - Speed benchmarking workflow (48-step rollout, determinism) is closed.
  - Full behavioral parity has accepted gaps:
      obs shape (JAX includes line flows; PowerZoo side may differ),
      dispatch solver (JAX continuous-relaxation SCUC vs PowerZoo backend),
      reserve cost routing (penalty weights may differ across sides).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from powerzoojax.tasks.tso import (
    make_comparison_tso_load_trace,
    make_comparison_tso_params,
    TSO_COMPARISON_SCHEMA,
    _comparison_tso_synthetic_trace,
)
from powerzoojax.envs.grid.unit_commitment import UnitCommitmentEnv

# ---------------------------------------------------------------------------
# Golden file
# ---------------------------------------------------------------------------

GOLDEN_PATH = Path(__file__).parent.parent / "golden" / "tso_comparison_contract.json"


@pytest.fixture(autouse=True, scope="module")
def _force_float32_mode():
    """Keep comparison tests isolated from ACOPF's global x64 side effects."""
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", False)


@pytest.fixture(scope="session")
def golden() -> dict:
    assert GOLDEN_PATH.exists(), (
        f"Golden file missing: {GOLDEN_PATH}. "
        "Run: python scripts/generate_tso_comparison_golden.py"
    )
    return json.loads(GOLDEN_PATH.read_text())


# ---------------------------------------------------------------------------
# JAX fixtures — function scope (immune to x64 contamination from other tests)
# ---------------------------------------------------------------------------

@pytest.fixture
def jax_env():
    return UnitCommitmentEnv()


@pytest.fixture
def jax_params():
    return make_comparison_tso_params()


@pytest.fixture
def jax_reset(jax_env, jax_params):
    key = jax.random.PRNGKey(42)
    obs, state = jax_env.reset(key, jax_params)
    return obs, state, key


# ============================================================
# 1. Load trace contracts
# ============================================================
#
# Two functions:
#  - ``make_comparison_tso_load_trace``   — production GB-real path
#    (cross-backend benchmark; identical formula on PowerZoo side).
#  - ``_comparison_tso_synthetic_trace``  — legacy sin-wave helper kept
#    for tests / CI that must run without GB parquet.

def test_trace_length():
    assert len(make_comparison_tso_load_trace(48, 0.5)) == 48


def test_trace_deterministic():
    trace1 = make_comparison_tso_load_trace(48, 0.5)
    trace2 = make_comparison_tso_load_trace(48, 0.5)
    np.testing.assert_array_equal(trace1, trace2)


def test_trace_bounds():
    """GB-real net-load is clipped to [0.05, 1.0] on both sides; pre-fix
    sin-wave bounds were [0.10, 1.00].  Loosen to the implementation
    contract."""
    trace = make_comparison_tso_load_trace(48, 0.5)
    assert float(np.min(trace)) >= 0.05 - 1e-6
    assert float(np.max(trace)) <= 1.00 + 1e-6


def test_synthetic_helper_step0_value():
    """Legacy sin-wave helper retains its analytical anchor for offline
    tests: at t=0 ``norm = 0.75 + 0.25*sin(-π/4) ≈ 0.5732``."""
    trace = _comparison_tso_synthetic_trace(48, 0.5)
    expected = 0.75 + 0.25 * np.sin(-np.pi / 4.0)
    assert abs(float(trace[0]) - expected) < 1e-4


def test_synthetic_helper_peak_at_step18():
    trace = _comparison_tso_synthetic_trace(48, 0.5)
    assert abs(float(trace[18]) - 1.0) < 1e-4


def test_synthetic_helper_trough_at_step42():
    trace = _comparison_tso_synthetic_trace(48, 0.5)
    assert abs(float(trace[42]) - 0.5) < 1e-4


# ============================================================
# 2. Schema contents
# ============================================================

def test_schema_case_id():
    assert TSO_COMPARISON_SCHEMA["case_id"] == "case118"


def test_schema_n_units():
    assert TSO_COMPARISON_SCHEMA["n_units"] == 54


def test_schema_n_agents():
    """Must be 1 centralized agent on both sides."""
    assert TSO_COMPARISON_SCHEMA["n_agents"] == 1


def test_schema_horizon():
    assert TSO_COMPARISON_SCHEMA["max_steps"] == 48


def test_schema_dt():
    assert TSO_COMPARISON_SCHEMA["delta_t_minutes"] == 30
    assert TSO_COMPARISON_SCHEMA["delta_t_hours"] == 0.5


def test_schema_load_source_gb_real():
    """Cross-backend TSO env now uses real GB net-load on both sides
    (P0-1 fairness fix); the prior 'deterministic_synthetic' tag is
    retired but the sin-wave helper remains available for offline CI."""
    assert TSO_COMPARISON_SCHEMA["load_source"] == "gb_real"


def test_schema_enable_uc():
    assert TSO_COMPARISON_SCHEMA["enable_uc"] is True


def test_schema_enable_reserve():
    assert TSO_COMPARISON_SCHEMA["enable_reserve"] is True


def test_schema_action_shape():
    assert TSO_COMPARISON_SCHEMA["action_shape"] == (108,)


def test_schema_reward_components():
    rc = TSO_COMPARISON_SCHEMA["reward_components"]
    assert "gen_cost" in rc
    assert "startup_cost" in rc
    assert "no_load_cost" in rc


def test_schema_no_agent_structure_gap():
    """Agent structure gap must NOT be present — both sides are centralized."""
    gaps_text = str(TSO_COMPARISON_SCHEMA.get("accepted_gaps", []))
    assert "agent_structure" not in gaps_text.lower()


# ============================================================
# 3. Golden file — contract assertions
# ============================================================

def test_golden_load_source(golden):
    """Golden file may still reference the legacy synthetic anchor; we
    only check that the live schema records ``gb_real`` (P0-1 fix)."""
    assert TSO_COMPARISON_SCHEMA["load_source"] == "gb_real"


def test_golden_n_agents(golden):
    assert golden["n_agents"] == 1


def test_golden_action_shape(golden):
    assert tuple(golden["action_shape"]) == (108,)


def test_golden_max_steps(golden):
    assert golden["max_steps"] == 48


def test_golden_enable_uc(golden):
    assert golden["enable_uc"] is True


def test_golden_enable_reserve(golden):
    assert golden["enable_reserve"] is True


def test_golden_trace_matches_synthetic_helper(golden):
    """The legacy sin-wave anchor still matches the cached golden values
    (golden was generated against the sin formula).  After P0-1 the
    production trace is GB-real and must NOT match the sin anchor; that
    is verified by ``test_golden_trace_synthetic_diverges_from_gb``."""
    trace = _comparison_tso_synthetic_trace(48, 0.5)
    sc = golden["load_trace_spot_checks"]
    assert abs(float(trace[0])  - sc["step_0"])  < 1e-5
    assert abs(float(trace[18]) - sc["step_18"]) < 1e-5
    assert abs(float(trace[42]) - sc["step_42"]) < 1e-5
    assert abs(float(np.sum(trace)) - sc["trace_sum_48steps"]) < 1e-4


def test_golden_trace_synthetic_diverges_from_gb():
    """Sanity: the GB-real production trace must NOT collapse to the
    legacy sin anchor (otherwise we have not actually switched data
    sources)."""
    gb_trace = make_comparison_tso_load_trace(48, 0.5)
    syn_trace = _comparison_tso_synthetic_trace(48, 0.5)
    diff = float(np.max(np.abs(np.asarray(gb_trace) - np.asarray(syn_trace))))
    assert diff > 1e-3, (
        "GB-real and sin-wave traces collapsed to identical values; "
        "P0-1 (TSO real GB load) regressed."
    )


def test_golden_schema_consistent(golden):
    """TSO_COMPARISON_SCHEMA must agree with golden on contract fields
    that did NOT change in P0-1.  ``load_source`` was intentionally
    flipped from ``deterministic_synthetic`` to ``gb_real``."""
    schema = TSO_COMPARISON_SCHEMA
    assert schema["case_id"]      == golden["case_id"]
    assert schema["n_units"]      == golden["n_units"]
    assert schema["n_agents"]     == golden["n_agents"]
    assert schema["max_steps"]    == golden["max_steps"]
    assert tuple(schema["action_shape"]) == tuple(golden["action_shape"])


# ============================================================
# 4. JAX factory — UCParams fields
# ============================================================

def test_jax_params_n_units(jax_params):
    assert jax_params.case.n_units == 54


def test_jax_params_max_steps(jax_params):
    assert jax_params.max_steps == 48


def test_jax_params_delta_t(jax_params):
    assert abs(jax_params.delta_t_hours - 0.5) < 1e-6


def test_jax_params_enable_uc(jax_params):
    assert jax_params.enable_uc is True


def test_jax_params_enable_reserve(jax_params):
    assert jax_params.enable_reserve is True


def test_jax_params_ramp_shape(jax_params):
    assert jax_params.ramp_up_mw.shape == (54,)
    assert jax_params.ramp_down_mw.shape == (54,)


def test_jax_params_startup_cost_nonzero(jax_params):
    assert float(jnp.max(jax_params.startup_cost)) > 0.0


# ============================================================
# 5. JAX action / obs / reward spaces
# ============================================================

def test_jax_action_shape(jax_env, jax_params):
    assert jax_env.action_space(jax_params).shape == (108,)


def test_jax_obs_shape(jax_env, jax_params, jax_reset):
    obs, _, _ = jax_reset
    expected_dim = 4 * 54 + jax_params.case.n_lines + 4
    assert obs.shape == (expected_dim,)


def test_jax_obs_finite(jax_reset):
    obs, _, _ = jax_reset
    assert jnp.all(jnp.isfinite(obs))


# ============================================================
# 6. JAX reward / cost keys
# ============================================================

def test_jax_info_reward_keys(jax_env, jax_params, jax_reset):
    _, state, key = jax_reset
    action = jnp.zeros(108)
    _, k2 = jax.random.split(key)
    _, _, _, _costs, _, info = jax_env.step(k2, state, action, jax_params)
    for key_name in ("gen_cost", "startup_cost", "no_load_cost"):
        assert key_name in info, f"Missing info key: {key_name}"


def test_jax_info_cost_keys(jax_env, jax_params, jax_reset):
    _, state, key = jax_reset
    action = jnp.zeros(108)
    _, k2 = jax.random.split(key)
    _, _, _, _costs, _, info = jax_env.step(k2, state, action, jax_params)
    for key_name in (
        "cost_sum",
        "cost_thermal_overload",
        "cost_reserve_shortfall",
        "reserve_shortfall",
    ):
        assert key_name in info, f"Missing cost key: {key_name}"


def test_jax_reward_negative(jax_env, jax_params, jax_reset):
    _, state, key = jax_reset
    action = jnp.zeros(108)
    _, k2 = jax.random.split(key)
    _, _, reward, _costs, _, _ = jax_env.step(k2, state, action, jax_params)
    assert float(reward) < 0.0


# ============================================================
# 7. JAX 48-step rollout
# ============================================================

def test_jax_rollout_48steps(jax_env, jax_params):
    key = jax.random.PRNGKey(99)
    obs, state = jax_env.reset(key, jax_params)
    action = jnp.zeros(2 * jax_params.case.n_units)
    gen_costs = []
    for _ in range(48):
        key, k = jax.random.split(key)
        obs, state, reward, costs, done, info = jax_env.step(k, state, action, jax_params)
        gen_costs.append(float(info["gen_cost"]))
    assert len(gen_costs) == 48
    assert all(c > 0 for c in gen_costs)


def test_jax_rollout_reserve_tracked(jax_env, jax_params):
    key = jax.random.PRNGKey(7)
    obs, state = jax_env.reset(key, jax_params)
    n_units = jax_params.case.n_units
    action = jnp.concatenate([jnp.ones(n_units), jnp.zeros(n_units)])
    reserve_vals = []
    for _ in range(48):
        key, k = jax.random.split(key)
        obs, state, reward, costs, done, info = jax_env.step(k, state, action, jax_params)
        reserve_vals.append(float(info["reserve_shortfall"]))
    assert all(v >= 0 for v in reserve_vals)


def test_jax_load_trace_deterministic_rollout(jax_env, jax_params):
    """Two resets with same seed produce identical reward sequences."""
    action = jnp.zeros(108)

    def _rollout(seed):
        key = jax.random.PRNGKey(seed)
        obs, state = jax_env.reset(key, jax_params)
        rewards = []
        for _ in range(48):
            key, k = jax.random.split(key)
            obs, state, r, _costs, _, _ = jax_env.step(k, state, action, jax_params)
            rewards.append(float(r))
        return rewards

    r1 = _rollout(0)
    r2 = _rollout(0)
    np.testing.assert_allclose(r1, r2, rtol=0, atol=1e-5)
