"""Tests for ResourceBundle attachment to Grid Envs.

Validates that BatteryBundle integrates correctly with TransGridEnv,
DistGridEnv, and DistGrid3PhaseEnv — obs/action dims, SOC advancement,
injection effects, JIT/vmap/scan compatibility, and backward compat.
"""

import jax
import jax.numpy as jnp
import jax.tree_util as tu
import numpy as np
import pytest

from powerzoojax.case import create_case5, create_case33bw
from powerzoojax.envs.resource.battery import (
    BatteryBundle,
    BatteryBundleState,
    make_battery_bundle,
)
from powerzoojax.envs.grid.trans import (
    TransGridEnv,
    make_trans_params,
)
from powerzoojax.envs.grid.dist import (
    DistGridEnv,
    make_dist_params,
)
from powerzoojax.envs.grid.dist_3phase import (
    DistGrid3PhaseEnv,
    DistGrid3PhParams,
    make_dist_3phase_params,
)
from powerzoojax.envs.grid.bfs_3phase_power_flow import build_3phase_topology


# ========================== Fixtures ==========================

@pytest.fixture(scope="module")
def case5():
    return create_case5()


@pytest.fixture(scope="module")
def case33():
    return create_case33bw()


@pytest.fixture(scope="module")
def trans_env():
    return TransGridEnv()


@pytest.fixture(scope="module")
def dist_env():
    return DistGridEnv()


@pytest.fixture(scope="module")
def env_3ph():
    return DistGrid3PhaseEnv()


@pytest.fixture(scope="module")
def bundle_case5(case5):
    return make_battery_bundle(case5, bus_ids=[2, 4], power_mw=20.0, capacity_mwh=50.0)


@pytest.fixture(scope="module")
def trans_params_with_bundle(case5, bundle_case5):
    return make_trans_params(case5, resources=(bundle_case5,))


@pytest.fixture(scope="module")
def trans_params_empty(case5):
    return make_trans_params(case5)


@pytest.fixture(scope="module")
def bundle_case33(case33):
    return make_battery_bundle(case33, bus_ids=[18, 25], power_mw=0.2, capacity_mwh=0.4)


@pytest.fixture(scope="module")
def dist_params_with_bundle(case33, bundle_case33):
    return make_dist_params(case33, resources=(bundle_case33,))


@pytest.fixture(scope="module")
def params_3ph():
    n_nodes = 4
    n_lines = 3
    from_nodes = np.array([0, 1, 2])
    to_nodes = np.array([1, 2, 3])
    Z = np.zeros((n_lines, 3, 3), dtype=complex)
    for i in range(n_lines):
        Z[i] = (0.05 + 0.1j) * np.eye(3)

    T = 4
    P_base = jnp.ones(n_lines * 3) * 0.02
    Q_base = jnp.ones(n_lines * 3) * 0.01
    load_P = jnp.tile(P_base[None, :], (T, 1))
    load_Q = jnp.tile(Q_base[None, :], (T, 1))

    bundle = BatteryBundle(
        n_devices=1,
        per_device_action_dim=1,
        per_device_obs_dim=2,
        dt_hours=0.5,
        bus_idx=jnp.array([2], dtype=jnp.int32),
        power_max=jnp.array([0.5]),
        capacity=jnp.array([1.0]),
        soc_min=jnp.array([0.1]),
        soc_max=jnp.array([0.9]),
        eta_charge=jnp.array([0.95]),
        eta_discharge=jnp.array([0.95]),
        initial_soc=jnp.array([0.5]),
    )

    return make_dist_3phase_params(
        n_nodes, from_nodes, to_nodes, Z,
        load_P, load_Q,
        ref_bus=0, max_steps=T, base_mva=10.0,
        resources=(bundle,),
    )


# ====================== TransGridEnv ======================

class TestTransGridBundleAttachment:

    def test_empty_resources_backward_compat(self, trans_env, trans_params_empty, case5):
        key = jax.random.PRNGKey(0)
        obs, state = trans_env.reset(key, trans_params_empty)
        expected_obs_dim = case5.n_lines + case5.n_loads + case5.n_units + 2
        assert obs.shape == (expected_obs_dim,)
        act_space = trans_env.action_space(trans_params_empty)
        assert act_space.shape == (case5.n_units,)

    def test_with_bundle_reset_shapes(self, trans_env, trans_params_with_bundle):
        key = jax.random.PRNGKey(0)
        obs, state = trans_env.reset(key, trans_params_with_bundle)
        assert len(state.resource_states) == 1
        assert state.resource_states[0].soc.shape == (2,)

    def test_with_bundle_step_shapes(self, trans_env, trans_params_with_bundle, case5):
        key = jax.random.PRNGKey(0)
        obs, state = trans_env.reset(key, trans_params_with_bundle)
        act_space = trans_env.action_space(trans_params_with_bundle)
        obs_space = trans_env.observation_space(trans_params_with_bundle)
        expected_act = case5.n_units + 2  # 2 batteries, action_dim=1 each
        assert act_space.shape == (expected_act,)
        action = jnp.zeros(expected_act)
        obs2, state2, reward, costs, done, info = trans_env.step(key, state, action, trans_params_with_bundle)
        assert obs2.shape == (obs_space.shape[0],)
        assert "cost_resource" in info

    def test_injection_reduces_load(self, trans_env, trans_params_with_bundle, case5):
        key = jax.random.PRNGKey(0)
        obs, state = trans_env.reset(key, trans_params_with_bundle)
        n_units = case5.n_units
        action_discharge = jnp.concatenate([jnp.zeros(n_units), jnp.ones(2)])
        _, state_d, _, _costs, _, _ = trans_env.step(key, state, action_discharge, trans_params_with_bundle)
        action_zero = jnp.zeros(n_units + 2)
        _, state_z, _, _costs, _, _ = trans_env.step(key, state, action_zero, trans_params_with_bundle)
        # With discharge injection, generation cost should differ
        assert not jnp.allclose(state_d.total_cost, state_z.total_cost)

    def test_soc_advances(self, trans_env, trans_params_with_bundle, case5):
        key = jax.random.PRNGKey(0)
        obs, state = trans_env.reset(key, trans_params_with_bundle)
        n_units = case5.n_units
        action = jnp.concatenate([jnp.zeros(n_units), jnp.ones(2)])  # discharge
        new_obs, new_state, _, _costs, _, _ = trans_env.step(key, state, action, trans_params_with_bundle)
        initial_soc = state.resource_states[0].soc
        new_soc = new_state.resource_states[0].soc
        assert jnp.all(new_soc < initial_soc), "SOC should decrease after discharge"

    def test_scan_rollout(self, trans_env, trans_params_with_bundle, case5):
        key = jax.random.PRNGKey(0)
        _, init_state = trans_env.reset(key, trans_params_with_bundle)
        n_act = case5.n_units + 2

        def scan_step(carry, _):
            state, key = carry
            key, subkey = jax.random.split(key)
            action = jnp.zeros(n_act)
            obs, new_state, r, costs, d, info = trans_env.step(subkey, state, action, trans_params_with_bundle)
            return (new_state, key), r

        (final_state, _), rewards = jax.lax.scan(scan_step, (init_state, key), None, length=10)
        assert rewards.shape == (10,)
        assert final_state.resource_states[0].soc.shape == (2,)

    def test_vmap(self, trans_env, trans_params_with_bundle, case5):
        batch = 4
        keys = jax.random.split(jax.random.PRNGKey(0), batch)
        obs_b, states_b = jax.vmap(trans_env.reset, in_axes=(0, None))(keys, trans_params_with_bundle)
        assert obs_b.shape[0] == batch
        n_act = case5.n_units + 2
        actions = jnp.zeros((batch, n_act))
        obs2, states2, r, costs, d, info = jax.vmap(trans_env.step, in_axes=(0, 0, 0, None))(
            keys, states_b, actions, trans_params_with_bundle)
        assert states2.resource_states[0].soc.shape == (batch, 2)

    def test_is_opf_bug_regression(self, case5, trans_env):
        """solver_mode=1 (DCOPF) + resources=() should not NameError."""
        params = make_trans_params(case5, solver_mode=1)
        key = jax.random.PRNGKey(0)
        obs, state = trans_env.reset(key, params)
        action = jnp.zeros(case5.n_units)
        obs2, state2, r, costs, d, info = trans_env.step(key, state, action, params)
        assert obs2.shape[0] > 0

    def test_dcopf_with_bundle(self, case5, trans_env):
        bundle = make_battery_bundle(case5, bus_ids=[2], power_mw=10.0, capacity_mwh=20.0)
        params = make_trans_params(case5, solver_mode=1, resources=(bundle,))
        key = jax.random.PRNGKey(0)
        obs, state = trans_env.reset(key, params)
        n_act = case5.n_units + 1
        action = jnp.zeros(n_act)
        obs2, state2, r, costs, d, info = trans_env.step(key, state, action, params)
        assert obs2.shape[0] > 0

    def test_cycle_cost_included_in_cost_resource(self, case5, trans_env):
        """TransGridEnv includes BatteryBundle's cost_sum in info['cost_resource']."""
        rate = 10.0
        dt = 0.5
        p_rated = 10.0
        bundle = make_battery_bundle(
            case5,
            bus_ids=[2],
            power_mw=p_rated,
            capacity_mwh=20.0,
            initial_soc=0.5,
            dt_hours=dt,
            cycle_cost_per_mwh=rate,
        )
        params = make_trans_params(case5, resources=(bundle,), delta_t_hours=dt)
        key = jax.random.PRNGKey(0)
        obs, state = trans_env.reset(key, params)
        n_act = case5.n_units + 1
        action = jnp.zeros(n_act).at[-1].set(1.0)  # max discharge last segment
        _, _, _, _costs, _, info = trans_env.step(key, state, action, params)
        expected_cycle = p_rated * dt * rate
        assert float(info["cost_resource"]) == pytest.approx(expected_cycle, rel=1e-4)


# ====================== DistGridEnv ======================

class TestDistGridBundleAttachment:

    def test_with_bundle_pu_conversion(self, dist_env, dist_params_with_bundle, case33):
        key = jax.random.PRNGKey(0)
        obs, state = dist_env.reset(key, dist_params_with_bundle)
        n_der = case33.n_nodes - 1
        n_act = n_der + 2
        action = jnp.concatenate([jnp.zeros(n_der), jnp.ones(2)])  # discharge
        obs2, state2, r, costs, d, info = dist_env.step(key, state, action, dist_params_with_bundle)
        assert obs2.shape[0] > 0
        assert state2.resource_states[0].soc.shape == (2,)
        assert jnp.all(state2.resource_states[0].soc < 0.5), "SOC should decrease after discharge"

    def test_dist_empty_backward_compat(self, dist_env, case33):
        params = make_dist_params(case33)
        key = jax.random.PRNGKey(0)
        obs, state = dist_env.reset(key, params)
        n = case33.n_nodes
        assert obs.shape[0] > 0
        action = jnp.zeros(n - 1)
        obs2, state2, r, costs, d, info = dist_env.step(key, state, action, params)
        assert obs2.shape[0] > 0


# ====================== DistGrid3PhaseEnv ======================

class TestDistGrid3PhBundleAttachment:

    def test_3phase_with_bundle_balanced(self, env_3ph, params_3ph):
        key = jax.random.PRNGKey(0)
        obs, state = env_3ph.reset(key, params_3ph)
        assert len(state.resource_states) == 1
        assert state.resource_states[0].soc.shape == (1,)

        act_space = env_3ph.action_space(params_3ph)
        n_3ph_der = params_3ph.topo.n_lines * 3
        expected_act = n_3ph_der + 1
        assert act_space.shape == (expected_act,)

        action = jnp.concatenate([jnp.zeros(n_3ph_der), jnp.array([1.0])])
        obs2, state2, r, costs, d, info = env_3ph.step(key, state, action, params_3ph)
        assert obs2.shape[0] > 0
        assert "cost_resource" in info

    def test_3phase_empty_backward_compat(self, env_3ph):
        n_nodes = 4
        n_lines = 3
        from_nodes = np.array([0, 1, 2])
        to_nodes = np.array([1, 2, 3])
        Z = np.zeros((n_lines, 3, 3), dtype=complex)
        for i in range(n_lines):
            Z[i] = (0.05 + 0.1j) * np.eye(3)
        T = 4
        P_base = jnp.ones(n_lines * 3) * 0.02
        Q_base = jnp.ones(n_lines * 3) * 0.01
        load_P = jnp.tile(P_base[None, :], (T, 1))
        load_Q = jnp.tile(Q_base[None, :], (T, 1))

        params = make_dist_3phase_params(
            n_nodes, from_nodes, to_nodes, Z, load_P, load_Q, max_steps=T)
        key = jax.random.PRNGKey(0)
        obs, state = env_3ph.reset(key, params)
        assert obs.shape[0] > 0
        action = jnp.zeros(n_lines * 3)
        obs2, state2, r, costs, d, info = env_3ph.step(key, state, action, params)
        assert obs2.shape[0] > 0


# ====================== Multi-bundle ======================

# ====================== include_der=False ======================

class TestBundleOnlyAction:
    """When include_der=False, action space = bundle dims only."""

    def test_dist_include_der_false_action_dim(self, dist_env, case33, bundle_case33):
        params = make_dist_params(case33, resources=(bundle_case33,), include_der=False)
        act_space = dist_env.action_space(params)
        assert act_space.shape == (bundle_case33.action_dim,), \
            f"Expected action_dim={bundle_case33.action_dim}, got {act_space.shape}"

    def test_dist_include_der_false_step(self, dist_env, case33, bundle_case33):
        params = make_dist_params(case33, resources=(bundle_case33,), include_der=False)
        key = jax.random.PRNGKey(0)
        obs, state = dist_env.reset(key, params)
        action = jnp.ones(bundle_case33.action_dim)  # discharge
        obs2, state2, r, costs, d, info = dist_env.step(key, state, action, params)
        assert obs2.shape[0] > 0
        assert jnp.all(state2.resource_states[0].soc < 0.5)

    def test_dist_include_der_false_scan(self, dist_env, case33, bundle_case33):
        params = make_dist_params(case33, resources=(bundle_case33,), include_der=False)
        key = jax.random.PRNGKey(0)
        _, init_state = dist_env.reset(key, params)
        act_dim = bundle_case33.action_dim

        def scan_step(carry, _):
            state, key = carry
            key, sk = jax.random.split(key)
            action = jnp.zeros(act_dim)
            obs, ns, r, costs, d, info = dist_env.step(sk, state, action, params)
            return (ns, key), r

        (final, _), rewards = jax.lax.scan(scan_step, (init_state, key), None, length=10)
        assert rewards.shape == (10,)

    def test_trans_include_unit_dispatch_false(self, trans_env, case5):
        bundle = make_battery_bundle(case5, bus_ids=[2], power_mw=10.0, capacity_mwh=20.0)
        params = make_trans_params(case5, solver_mode=1, resources=(bundle,),
                                   include_unit_dispatch=False)
        act_space = trans_env.action_space(params)
        assert act_space.shape == (bundle.action_dim,), \
            f"Expected {bundle.action_dim}, got {act_space.shape}"
        key = jax.random.PRNGKey(0)
        obs, state = trans_env.reset(key, params)
        action = jnp.zeros(bundle.action_dim)
        obs2, state2, r, costs, d, info = trans_env.step(key, state, action, params)
        assert obs2.shape[0] > 0


# ====================== Multi-bundle ======================

class TestMultiBundleAttachment:

    def test_multi_bundle_trans(self, case5, trans_env):
        b1 = make_battery_bundle(case5, bus_ids=[2], power_mw=20.0, capacity_mwh=50.0)
        b2 = make_battery_bundle(case5, bus_ids=[4, 5], power_mw=10.0, capacity_mwh=20.0)
        params = make_trans_params(case5, resources=(b1, b2))
        key = jax.random.PRNGKey(0)
        obs, state = trans_env.reset(key, params)
        assert len(state.resource_states) == 2
        assert state.resource_states[0].soc.shape == (1,)
        assert state.resource_states[1].soc.shape == (2,)

        act_dim = case5.n_units + b1.action_dim + b2.action_dim
        assert trans_env.action_space(params).shape == (act_dim,)

        action = jnp.zeros(act_dim)
        obs2, state2, r, costs, d, info = trans_env.step(key, state, action, params)
        assert obs2.shape[0] > 0
        assert len(state2.resource_states) == 2
