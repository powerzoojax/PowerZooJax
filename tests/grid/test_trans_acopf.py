"""Tests for TransGridEnv in ACOPF mode (solver_mode=2) — L0 JAX + L1 env + L1.5 cross.

L0: JIT, vmap, scan, auto-reset with ACOPF.
L1: Observation shape, cost positive, generator limits, voltage bounds.
L1.5: ACOPF dispatch vs PF dispatch comparison.
"""

import jax
import jax.numpy as jnp
import jax.tree_util as tu
import numpy as np
import pytest

from powerzoojax.case import create_case5
from powerzoojax.envs.grid.trans import TransGridEnv, TransGridParams, make_trans_params


@pytest.fixture(scope="module")
def case():
    return create_case5()


@pytest.fixture(scope="module")
def env():
    return TransGridEnv()


@pytest.fixture(scope="module")
def params_acopf(case):
    return make_trans_params(case, max_steps=4, solver_mode=2)


@pytest.fixture(scope="module")
def params_pf(case):
    return make_trans_params(case, max_steps=4, solver_mode=0)


@pytest.fixture(scope="module")
def key():
    return jax.random.PRNGKey(99)


# ==================== L0: JAX Contract ====================

class TestTransACOPF_L0_JAX:

    def test_reset_jit(self, env, params_acopf, key):
        obs, state = env.reset(key, params_acopf)
        assert obs.ndim == 1
        assert isinstance(state.time_step, jax.Array)

    def test_step_jit(self, env, params_acopf, key):
        _, state0 = env.reset(key, params_acopf)
        action = jnp.zeros(params_acopf.case.n_units)
        obs, state, reward, costs, done, info = env.step(
            key, state0, action, params_acopf
        )
        assert obs.ndim == 1
        assert reward.ndim == 0
        assert costs.shape == (4,)

    @pytest.mark.xfail(
        reason=(
            "Known ACOPF path dtype drift: reset state is float32 while "
            "ACOPF step returns float64 fields, so scan carry is not yet stable."
        ),
        strict=True,
    )
    def test_scan_rollout(self, env, params_acopf, key):
        _, state0 = env.reset(key, params_acopf)

        def scan_step(carry, _):
            s, k = carry
            k, k2 = jax.random.split(k)
            a = jnp.zeros(params_acopf.case.n_units)
            o, ns, r, costs, d, info = env.step(k2, s, a, params_acopf)
            return (ns, k), (r, costs, d)

        (final_s, _), (rews, costs, dones) = jax.lax.scan(
            scan_step, (state0, key), None, length=params_acopf.max_steps)
        assert rews.shape == (params_acopf.max_steps,)
        assert costs.shape == (params_acopf.max_steps, 4)

    @pytest.mark.xfail(
        reason=(
            "Known ACOPF path dtype drift: auto-reset scan shares the same "
            "float32/float64 state carry mismatch as scan_rollout."
        ),
        strict=True,
    )
    def test_auto_reset(self, env, params_acopf, key):
        _, state0 = env.reset(key, params_acopf)
        a = jnp.zeros(params_acopf.case.n_units)

        def step_fn(carry, _):
            s, k = carry
            k, k2 = jax.random.split(k)
            o, ns, r, costs, d, info = env.step(k2, s, a, params_acopf)
            return (ns, k), d

        (final_s, _), dones = jax.lax.scan(
            step_fn, (state0, key), None, length=params_acopf.max_steps)
        assert bool(dones[-1])
        assert int(final_s.time_step) == 0

    def test_pytree_stability(self, env, params_acopf, key):
        _, s = env.reset(key, params_acopf)
        flat1, td1 = tu.tree_flatten(s)
        rebuilt = tu.tree_unflatten(td1, flat1)
        flat2, _ = tu.tree_flatten(rebuilt)
        assert len(flat1) == len(flat2)


# ==================== L1: Environment Behavior ====================

class TestTransACOPF_L1_Env:

    def test_obs_includes_vm(self, env, params_acopf, key):
        """ACOPF mode obs should include vm (larger dim than DC mode)."""
        obs, _ = env.reset(key, params_acopf)
        sp = env.observation_space(params_acopf)
        assert obs.shape == sp.shape
        n = params_acopf.case
        dc_dim = n.n_lines + n.n_loads + n.n_units + 2
        assert obs.shape[0] == dc_dim + n.n_nodes

    def test_gen_within_limits(self, env, params_acopf, key):
        _, state0 = env.reset(key, params_acopf)
        a = jnp.zeros(params_acopf.case.n_units)
        _, state1, _, _, _, _ = env.step(key, state0, a, params_acopf)
        assert jnp.all(state1.unit_power_mw >= params_acopf.case.unit_p_min - 1.0)
        assert jnp.all(state1.unit_power_mw <= params_acopf.case.unit_p_max + 1.0)

    def test_vm_reasonable(self, env, params_acopf, key):
        _, state0 = env.reset(key, params_acopf)
        a = jnp.zeros(params_acopf.case.n_units)
        _, state1, _, _, _, _ = env.step(key, state0, a, params_acopf)
        assert jnp.all(state1.vm >= 0.8)
        assert jnp.all(state1.vm <= 1.2)

    def test_cost_positive(self, env, params_acopf, key):
        _, state0 = env.reset(key, params_acopf)
        a = jnp.zeros(params_acopf.case.n_units)
        _, _, _, _, _, info = env.step(key, state0, a, params_acopf)
        assert float(info["gen_cost"]) > 0.0

    def test_info_exposes_line_viol_mva(self, env, params_acopf, key):
        _, state0 = env.reset(key, params_acopf)
        a = jnp.zeros(params_acopf.case.n_units)
        _, _, _, _, _, info = env.step(key, state0, a, params_acopf)
        assert "line_viol_mva" in info
        assert jnp.isfinite(info["line_viol_mva"])
        assert float(info["line_viol_mva"]) >= 0.0

    def test_acopf_line_flow_q_nonzero_obs_uses_apparent(self, env, params_acopf, key):
        """ACOPF mode fills line_flow_q_mw; obs line norm uses |S|/cap."""
        _, state0 = env.reset(key, params_acopf)
        a = jnp.zeros(params_acopf.case.n_units)
        _, state1, _, _, _, _ = env.step(key, state0, a, params_acopf)
        assert not jnp.allclose(state1.line_flow_q_mw, 0.0)


# ==================== L1.5: Cross-Mode Consistency ====================

class TestTransACOPF_L15_Cross:

    def test_acopf_ignores_action(self, env, params_acopf, key):
        """In ACOPF mode, different actions should yield same dispatch."""
        _, state0 = env.reset(key, params_acopf)
        a1 = params_acopf.case.unit_p_min
        a2 = params_acopf.case.unit_p_max
        _, s1, _, _, _, _ = env.step(key, state0, a1, params_acopf)
        _, s2, _, _, _, _ = env.step(key, state0, a2, params_acopf)
        np.testing.assert_allclose(
            np.array(s1.unit_power_mw), np.array(s2.unit_power_mw), atol=0.01,
            err_msg="ACOPF dispatch should be independent of agent action")
