"""Tests for TransGridEnv in OPF mode (solver_mode=1) — L0 JAX + L1 env + L1.5 cross.

L0: JIT, vmap, scan, auto-reset with DCOPF.
L1: Observation shape, cost positive, generator limits.
L1.5: OPF dispatch vs PF dispatch comparison.
"""

import jax
import jax.numpy as jnp
import jax.tree_util as tu
import numpy as np
import pytest

from powerzoojax.case import create_case5
from powerzoojax.envs.grid.trans import TransGridEnv, TransGridParams, make_trans_params


@pytest.fixture(autouse=True, scope="module")
def _force_float32_mode():
    """Keep DCOPF-backed TransGrid tests isolated from ACOPF's global x64 side effects."""
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", False)


@pytest.fixture(scope="module")
def case():
    return create_case5()


@pytest.fixture(scope="module")
def env():
    return TransGridEnv()


@pytest.fixture(scope="module")
def params_opf(case):
    return make_trans_params(case, max_steps=8, solver_mode=1)


@pytest.fixture(scope="module")
def params_pf(case):
    return make_trans_params(case, max_steps=8, solver_mode=0)


@pytest.fixture(scope="module")
def key():
    return jax.random.PRNGKey(123)


# ==================== L0: JAX Contract ====================

class TestTransOPF_L0_JAX:

    def test_reset_jit(self, env, params_opf, key):
        obs, state = env.reset(key, params_opf)
        assert obs.ndim == 1
        assert isinstance(state.time_step, jax.Array)

    def test_step_jit(self, env, params_opf, key):
        obs0, state0 = env.reset(key, params_opf)
        action = jnp.zeros(params_opf.case.n_units)
        obs, state, reward, costs, done, info = env.step(
            key, state0, action, params_opf
        )
        assert obs.ndim == 1
        assert reward.ndim == 0
        assert costs.shape == (4,)

    def test_vmap_batch(self, env, params_opf):
        keys = jax.random.split(jax.random.PRNGKey(0), 3)
        batch_p = tu.tree_map(lambda x: jnp.stack([x] * 3), params_opf)
        vfn = jax.vmap(env.reset)
        obs_b, state_b = vfn(keys, batch_p)
        assert obs_b.shape[0] == 3

    def test_scan_rollout(self, env, params_opf, key):
        obs0, state0 = env.reset(key, params_opf)

        def scan_step(carry, _):
            s, k = carry
            k, k2 = jax.random.split(k)
            a = jnp.zeros(params_opf.case.n_units)
            o, ns, r, costs, d, info = env.step(k2, s, a, params_opf)
            return (ns, k), (r, costs, d)

        (final_s, _), (rews, costs, dones) = jax.lax.scan(
            scan_step, (state0, key), None, length=params_opf.max_steps)
        assert rews.shape == (params_opf.max_steps,)
        assert costs.shape == (params_opf.max_steps, 4)

    def test_auto_reset(self, env, params_opf, key):
        _, state0 = env.reset(key, params_opf)
        a = jnp.zeros(params_opf.case.n_units)

        def step_fn(carry, _):
            s, k = carry
            k, k2 = jax.random.split(k)
            o, ns, r, costs, d, info = env.step(k2, s, a, params_opf)
            return (ns, k), d

        (final_s, _), dones = jax.lax.scan(
            step_fn, (state0, key), None, length=params_opf.max_steps)
        assert bool(dones[-1])
        assert int(final_s.time_step) == 0

    def test_pytree_stability(self, env, params_opf, key):
        _, s = env.reset(key, params_opf)
        flat1, td1 = tu.tree_flatten(s)
        rebuilt = tu.tree_unflatten(td1, flat1)
        flat2, _ = tu.tree_flatten(rebuilt)
        assert len(flat1) == len(flat2)


# ==================== L1: Environment Behavior ====================

class TestTransOPF_L1_Env:

    def test_obs_shape(self, env, params_opf, key):
        obs, _ = env.reset(key, params_opf)
        sp = env.observation_space(params_opf)
        assert obs.shape == sp.shape

    def test_gen_within_limits(self, env, params_opf, key):
        """OPF dispatch must respect generator limits."""
        _, state0 = env.reset(key, params_opf)
        a = jnp.zeros(params_opf.case.n_units)
        _, state1, _, _, _, _ = env.step(key, state0, a, params_opf)
        assert jnp.all(state1.unit_power_mw >= params_opf.case.unit_p_min - 1.0)
        assert jnp.all(state1.unit_power_mw <= params_opf.case.unit_p_max + 1.0)

    def test_cost_positive(self, env, params_opf, key):
        """Generation cost should be positive (there is load)."""
        _, state0 = env.reset(key, params_opf)
        a = jnp.zeros(params_opf.case.n_units)
        _, _, _, _, _, info = env.step(key, state0, a, params_opf)
        assert float(info["gen_cost"]) > 0.0

    def test_reward_negative(self, env, params_opf, key):
        """Reward = -scale * gen_cost, should be negative."""
        _, state0 = env.reset(key, params_opf)
        a = jnp.zeros(params_opf.case.n_units)
        _, _, reward, _, _, _ = env.step(key, state0, a, params_opf)
        assert float(reward) < 0.0


# ==================== L1.5: Cross-Mode Consistency ====================

class TestTransOPF_L15_Cross:

    def test_opf_ignores_action(self, env, params_opf, key):
        """In OPF mode, different actions should yield same dispatch."""
        _, state0 = env.reset(key, params_opf)
        a1 = params_opf.case.unit_p_min
        a2 = params_opf.case.unit_p_max
        _, s1, _, _, _, _ = env.step(key, state0, a1, params_opf)
        _, s2, _, _, _, _ = env.step(key, state0, a2, params_opf)
        np.testing.assert_allclose(
            np.array(s1.unit_power_mw), np.array(s2.unit_power_mw), atol=0.01,
            err_msg="OPF dispatch should be independent of agent action")

    def test_opf_lower_cost_than_naive(self, env, params_opf, params_pf, key):
        """OPF should produce lower or equal cost vs naive proportional dispatch."""
        _, state0_opf = env.reset(key, params_opf)
        a = jnp.zeros(params_opf.case.n_units)
        _, _, _, _, _, info_opf = env.step(key, state0_opf, a, params_opf)

        _, state0_pf = env.reset(key, params_pf)
        mid = (params_pf.case.unit_p_min + params_pf.case.unit_p_max) / 2
        _, _, _, _, _, info_pf = env.step(key, state0_pf, mid, params_pf)

        cost_opf = float(info_opf["gen_cost"])
        cost_pf = float(info_pf["gen_cost"])
        assert cost_opf <= cost_pf * 1.2, (
            f"OPF cost {cost_opf:.1f} should be <= PF cost {cost_pf:.1f} (with margin)")
