"""Tests for DistGrid3PhaseEnv — L0 JAX + L1 env + L1.5 cross-module.

L0: JIT, vmap, scan, auto-reset.
L1: Observation shape, reward, cost, voltage safety.
L1.5: Cross-module consistency (zero DER = baseline, DER reduces deviation).
Physics: ``line_phase_mask`` / per-phase thermal vs ``line_cap``, bundle ``q_inj``
in ``Q_net`` (aligned with standalone ``bfs_3phase_power_flow``).
"""

import jax
import jax.numpy as jnp
import jax.tree_util as tu
import numpy as np
import pytest

from powerzoojax.envs.grid.bfs_3phase_power_flow import (
    build_3phase_topology,
    bfs_3phase_power_flow,
)
from powerzoojax.envs.grid.dist_3phase import (
    DistGrid3PhaseEnv,
    DistGrid3PhParams,
    _compute_3ph_thermal_metrics,
    make_dist_3phase_params,
)
from powerzoojax.envs.resource.battery import BatteryBundle


@pytest.fixture(scope="module")
def env():
    return DistGrid3PhaseEnv()


@pytest.fixture(scope="module")
def params():
    n_nodes = 4
    n_lines = 3
    from_nodes = np.array([0, 1, 2])
    to_nodes = np.array([1, 2, 3])
    Z = np.zeros((n_lines, 3, 3), dtype=complex)
    for i in range(n_lines):
        Z[i] = (0.05 + 0.1j) * np.eye(3)

    topo = build_3phase_topology(n_nodes, from_nodes, to_nodes, Z, ref_bus=0)

    T = 4
    P_base = jnp.ones(n_lines * 3) * 0.02
    Q_base = jnp.ones(n_lines * 3) * 0.01
    load_P = jnp.tile(P_base[None, :], (T, 1))
    load_Q = jnp.tile(Q_base[None, :], (T, 1))

    p_load_ref = max(float(jnp.sum(load_P[0])), 1e-8)
    q_load_ref = max(float(jnp.sum(load_Q[0])), 1e-8)

    return DistGrid3PhParams(
        topo=topo,
        load_P_3ph=load_P,
        load_Q_3ph=load_Q,
        p_load_ref=p_load_ref,
        q_load_ref=q_load_ref,
        base_mva=10.0,
        v_ref_mag=1.0,
        v_min=0.90,
        v_max=1.10,
        vuf_max=2.0,
        max_steps=T,
        steps_per_day=T,
        loss_penalty_weight=0.1,
    )


@pytest.fixture(scope="module")
def key():
    return jax.random.PRNGKey(42)


# ==================== L0: JAX Contract ====================

class TestDistGrid3Ph_L0_JAX:

    def test_reset_jit(self, env, params, key):
        obs, state = env.reset(key, params)
        assert obs.ndim == 1
        assert isinstance(state.time_step, jax.Array)

    def test_step_jit(self, env, params, key):
        obs0, state0 = env.reset(key, params)
        action = jnp.zeros(params.topo.n_lines * 3)
        obs, state, reward, costs, done, info = env.step(key, state0, action, params)
        assert obs.ndim == 1
        assert reward.ndim == 0
        assert costs.shape == (4,)

    def test_vmap_batch_reset(self, env, params):
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        batch_params = tu.tree_map(lambda x: jnp.stack([x] * 4), params)
        vfn = jax.vmap(env.reset)
        obs_b, state_b = vfn(keys, batch_params)
        assert obs_b.shape[0] == 4

    def test_scan_rollout(self, env, params, key):
        obs0, state0 = env.reset(key, params)
        act_dim = params.topo.n_lines * 3

        def scan_step(carry, _):
            s, k = carry
            k, k_step = jax.random.split(k)
            a = jnp.zeros(act_dim)
            obs, ns, r, costs, d, info = env.step(k_step, s, a, params)
            return (ns, k), (obs, r, d)

        (final_s, _), (obs_seq, r_seq, d_seq) = jax.lax.scan(
            scan_step, (state0, key), None, length=params.max_steps)
        assert r_seq.shape == (params.max_steps,)

    def test_auto_reset(self, env, params, key):
        obs0, state0 = env.reset(key, params)
        act = jnp.zeros(params.topo.n_lines * 3)

        def step_fn(carry, _):
            s, k = carry
            k, k2 = jax.random.split(k)
            o, ns, r, costs, d, info = env.step(k2, s, act, params)
            return (ns, k), d

        (final_s, _), dones = jax.lax.scan(
            step_fn, (state0, key), None, length=params.max_steps)
        assert bool(dones[-1])
        assert int(final_s.time_step) == 0

    def test_pytree_stability(self, env, params, key):
        _, state = env.reset(key, params)
        flat1, td1 = tu.tree_flatten(state)
        rebuilt = tu.tree_unflatten(td1, flat1)
        flat2, td2 = tu.tree_flatten(rebuilt)
        assert len(flat1) == len(flat2)


# ==================== L1: Environment Behavior ====================

class TestDistGrid3Ph_L1_Env:

    def test_obs_shape(self, env, params, key):
        obs, _ = env.reset(key, params)
        sp = env.observation_space(params)
        assert obs.shape == sp.shape

    def test_action_shape(self, env, params):
        sp = env.action_space(params)
        assert sp.shape == (params.topo.n_lines * 3,)

    def test_reward_negative(self, env, params, key):
        """With non-zero load, reward should be <= 0."""
        _, state = env.reset(key, params)
        act = jnp.zeros(params.topo.n_lines * 3)
        _, _, reward, _, _, _ = env.step(key, state, act, params)
        assert float(reward) <= 0.0

    def test_reward_cost_separation(self, env, params, key):
        """Reward = -loss_penalty_weight * p_loss_MW; cost = count-based (safety only)."""
        _, state = env.reset(key, params)
        act = jnp.zeros(params.topo.n_lines * 3)
        _, _, reward, costs, _, info = env.step(key, state, act, params)
        # Reward should equal -loss_penalty_weight * p_loss_MW
        expected_reward = -params.loss_penalty_weight * info["p_loss_MW"]
        np.testing.assert_allclose(
            float(reward), float(expected_reward), atol=1e-5,
            err_msg="Reward should be -loss_penalty_weight * p_loss_MW")
        np.testing.assert_allclose(
            float(info["cost_sum"]), float(jnp.sum(costs)), atol=1e-5,
            err_msg="cost_sum should equal sum(costs)")

    def test_cost_nonneg(self, env, params, key):
        _, state = env.reset(key, params)
        act = jnp.zeros(params.topo.n_lines * 3)
        _, _, _, costs, _, info = env.step(key, state, act, params)
        assert bool(jnp.all(costs >= 0.0))

    def test_obs_normalisation_scale(self, env, params, key):
        """Power obs features should be normalised by total base-case load."""
        _, state = env.reset(key, params)
        act = jnp.zeros(params.topo.n_lines * 3)
        obs, *_ = env.step(key, state, act, params)

        n = params.topo.n_nodes
        nl = params.topo.n_lines
        # p_load slice: after V(3n) + P_flow(3nl) + Q_flow(3nl) → starts at 3n+6nl
        p_load_start = 3 * n + 6 * nl
        p_load_obs = obs[p_load_start:p_load_start + 3 * n]
        q_load_start = p_load_start + 3 * n
        q_load_obs = obs[q_load_start:q_load_start + 3 * n]
        # Loads sum to 1.0 (ref bus contributes 0)
        np.testing.assert_allclose(
            float(p_load_obs.sum()), 1.0, atol=1e-4,
            err_msg="p_load_norm should sum to 1.0")
        np.testing.assert_allclose(
            float(q_load_obs.sum()), 1.0, atol=1e-4,
            err_msg="q_load_norm should sum to 1.0")

    def test_vuf_violation_triggered(self, env, key):
        """Severe phase imbalance on loads → VUF over limit → cost_vuf_violation > 0."""
        n_nodes = 4
        n_lines = 3
        from_nodes = np.array([0, 1, 2])
        to_nodes = np.array([1, 2, 3])
        Z = np.zeros((n_lines, 3, 3), dtype=complex)
        for i in range(n_lines):
            Z[i] = (0.05 + 0.1j) * np.eye(3)
        T = 4
        n_nonref = n_nodes - 1
        p_row = np.tile(np.array([0.15, 0.001, 0.001], dtype=np.float32), n_nonref)
        P_base = jnp.asarray(p_row)
        Q_base = jnp.ones(n_lines * 3) * 0.005
        load_P = jnp.tile(P_base[None, :], (T, 1))
        load_Q = jnp.tile(Q_base[None, :], (T, 1))

        params = make_dist_3phase_params(
            n_nodes, from_nodes, to_nodes, Z, load_P, load_Q,
            ref_bus=0, max_steps=T, base_mva=10.0,
            vuf_max=0.5,
        )
        _, state = env.reset(key, params)
        act = jnp.zeros(n_lines * 3)
        _, _, _, _, _, info = env.step(key, state, act, params)
        assert float(info["cost_vuf_violation"]) > 0.0
        assert float(info["max_vuf_percent"]) > 0.5

    def test_action_space_no_der(self, env, params):
        """include_der=False → action dim = bundle only (no per-phase DER slot)."""
        bundle = BatteryBundle(
            n_devices=1,
            per_device_action_dim=2,
            per_device_obs_dim=3,
            dt_hours=0.5,
            enable_q_control=True,
            bus_idx=jnp.array([2], dtype=jnp.int32),
            power_max=jnp.array([0.3], dtype=jnp.float32),
            s_rated=jnp.array([0.5], dtype=jnp.float32),
            capacity=jnp.array([1.0], dtype=jnp.float32),
            soc_min=jnp.array([0.1], dtype=jnp.float32),
            soc_max=jnp.array([0.9], dtype=jnp.float32),
            eta_charge=jnp.array([0.95], dtype=jnp.float32),
            eta_discharge=jnp.array([0.95], dtype=jnp.float32),
            initial_soc=jnp.array([0.5], dtype=jnp.float32),
        )
        params_no_der = params.replace(resources=(bundle,), include_der=False)
        sp = env.action_space(params_no_der)
        assert sp.shape == (2,)
        assert int(params.topo.n_lines * 3) != 2


# ==================== L1.5: Cross-Module Consistency ====================

class TestDistGrid3Ph_L15_Cross:

    def test_zero_action_matches_baseline(self, env, params, key):
        """Zero DER action = pure load power flow → should match reset state approximately."""
        obs0, state0 = env.reset(key, params)
        act = jnp.zeros(params.topo.n_lines * 3)
        obs1, state1, *_ = env.step(key, state0, act, params)
        np.testing.assert_allclose(
            np.array(state0.v_mag), np.array(state1.v_mag), atol=0.01,
            err_msg="Zero action should not change voltage much from reset (same load)")

    def test_der_changes_voltage(self, env, params, key):
        """Non-zero DER should change the voltage profile."""
        _, state0 = env.reset(key, params)
        act_zero = jnp.zeros(params.topo.n_lines * 3)
        _, s1, *_ = env.step(key, state0, act_zero, params)
        act_der = jnp.ones(params.topo.n_lines * 3) * 0.01
        _, s2, *_ = env.step(key, state0, act_der, params)
        diff = jnp.max(jnp.abs(s1.v_mag - s2.v_mag))
        assert float(diff) > 1e-5, "DER injection should change voltage profile"


# ---------------------------------------------------------------------------
# Physics: topology phase masks, thermal limits, bundle Q injection
# ---------------------------------------------------------------------------

def _thermal_violation_count_reference(
    result,
    params,
    line_cap: jnp.ndarray,
    base_mva: float,
) -> jnp.ndarray:
    """Same count logic as ``DistGrid3PhaseEnv.step`` (thermal block)."""
    n_viol, _, _ = _compute_3ph_thermal_metrics(
        result, params.topo, line_cap,
        base_mva, params.ref_bus, params.v_ref_mag,
    )
    return n_viol


class TestDistGrid3Ph_PhysicsTopology:

    def test_line_phase_mask_single_phase_lateral(self):
        """Sparse diagonal Z → one energized phase; count = 1."""
        n_nodes = 4
        n_lines = 3
        from_nodes = np.array([0, 1, 2])
        to_nodes = np.array([1, 2, 3])
        Z = np.zeros((n_lines, 3, 3), dtype=complex)
        Z[0] = (0.05 + 0.1j) * np.eye(3)
        # Line 1: phase A only (single-phase lateral)
        Z[1, 0, 0] = 0.05 + 0.1j
        Z[2] = (0.05 + 0.1j) * np.eye(3)

        topo = build_3phase_topology(n_nodes, from_nodes, to_nodes, Z, ref_bus=0)
        m = np.array(topo.line_phase_mask)
        assert m[0].tolist() == [1.0, 1.0, 1.0]
        assert m[1].tolist() == [1.0, 0.0, 0.0]
        assert float(topo.line_phase_count[0]) == 3.0
        assert float(topo.line_phase_count[1]) == 1.0


class TestDistGrid3Ph_ThermalOverload:

    def test_zero_line_cap_skips_thermal(self, env, key):
        """``cap=0`` → no thermal violations counted (matches Python case data)."""
        n_nodes = 4
        n_lines = 3
        from_nodes = np.array([0, 1, 2])
        to_nodes = np.array([1, 2, 3])
        Z = np.zeros((n_lines, 3, 3), dtype=complex)
        for i in range(n_lines):
            Z[i] = (0.05 + 0.1j) * np.eye(3)
        T = 4
        P_base = jnp.ones(n_lines * 3) * 0.05
        Q_base = jnp.ones(n_lines * 3) * 0.03
        load_P = jnp.tile(P_base[None, :], (T, 1))
        load_Q = jnp.tile(Q_base[None, :], (T, 1))

        params = make_dist_3phase_params(
            n_nodes, from_nodes, to_nodes, Z, load_P, load_Q,
            ref_bus=0, max_steps=T, base_mva=10.0,
            line_cap=jnp.zeros(n_lines, dtype=jnp.float32),
        )
        _, state = env.reset(key, params)
        act = jnp.zeros(n_lines * 3)
        _, _, _, _, _, info = env.step(key, state, act, params)
        assert float(info["cost_thermal_overload"]) == 0.0

    def test_no_line_cap_means_no_thermal(self, env, key):
        """Default ``line_cap=None`` (no MVA limits) → thermal cost stays zero even heavy load."""
        n_nodes = 4
        n_lines = 3
        from_nodes = np.array([0, 1, 2])
        to_nodes = np.array([1, 2, 3])
        Z = np.zeros((n_lines, 3, 3), dtype=complex)
        for i in range(n_lines):
            Z[i] = (0.05 + 0.1j) * np.eye(3)
        T = 4
        P_base = jnp.ones(n_lines * 3) * 0.5
        Q_base = jnp.ones(n_lines * 3) * 0.01
        load_P = jnp.tile(P_base[None, :], (T, 1))
        load_Q = jnp.tile(Q_base[None, :], (T, 1))

        params = make_dist_3phase_params(
            n_nodes, from_nodes, to_nodes, Z, load_P, load_Q,
            ref_bus=0, max_steps=T, base_mva=10.0,
        )
        _, state = env.reset(key, params)
        act = jnp.zeros(n_lines * 3)
        _, _, _, _, _, info = env.step(key, state, act, params)
        assert float(info["cost_thermal_overload"]) == 0.0

    def test_tiny_line_cap_triggers_thermal(self, env, key):
        """Very small positive ``cap`` should produce thermal overload counts."""
        n_nodes = 4
        n_lines = 3
        from_nodes = np.array([0, 1, 2])
        to_nodes = np.array([1, 2, 3])
        Z = np.zeros((n_lines, 3, 3), dtype=complex)
        for i in range(n_lines):
            Z[i] = (0.05 + 0.1j) * np.eye(3)
        T = 4
        P_base = jnp.ones(n_lines * 3) * 0.15
        Q_base = jnp.ones(n_lines * 3) * 0.10
        load_P = jnp.tile(P_base[None, :], (T, 1))
        load_Q = jnp.tile(Q_base[None, :], (T, 1))

        line_cap = jnp.full((n_lines,), 1e-4, dtype=jnp.float32)  # 100 µVA / phase
        params = make_dist_3phase_params(
            n_nodes, from_nodes, to_nodes, Z, load_P, load_Q,
            ref_bus=0, max_steps=T, base_mva=10.0,
            line_cap=line_cap,
        )
        _, state = env.reset(key, params)
        act = jnp.zeros(n_lines * 3)
        _, _, _, _, _, info = env.step(key, state, act, params)
        assert float(info["cost_thermal_overload"]) > 0.0

    def test_thermal_count_matches_reference_formula(self, env, key):
        """``info['cost_thermal_overload']`` == independent reference count."""
        n_nodes = 4
        n_lines = 3
        from_nodes = np.array([0, 1, 2])
        to_nodes = np.array([1, 2, 3])
        Z = np.zeros((n_lines, 3, 3), dtype=complex)
        for i in range(n_lines):
            Z[i] = (0.05 + 0.1j) * np.eye(3)
        T = 4
        P_base = jnp.ones(n_lines * 3) * 0.08
        Q_base = jnp.ones(n_lines * 3) * 0.05
        load_P = jnp.tile(P_base[None, :], (T, 1))
        load_Q = jnp.tile(Q_base[None, :], (T, 1))
        base_mva = 10.0
        line_cap = jnp.array([0.5, 0.2, 0.3], dtype=jnp.float32)

        params = make_dist_3phase_params(
            n_nodes, from_nodes, to_nodes, Z, load_P, load_Q,
            ref_bus=0, max_steps=T, base_mva=base_mva,
            line_cap=line_cap,
        )
        _, state = env.reset(key, params)
        act = jnp.zeros(n_lines * 3)
        _, new_state, _, _, _, info = env.step(key, state, act, params)

        # Re-run PF with same net load as step (zero DER)
        P_net = params.load_P_3ph[0]
        Q_net = params.load_Q_3ph[0]
        result = bfs_3phase_power_flow(
            params.topo, P_net, Q_net,
            params.bfs_max_iter, params.bfs_tol,
        )
        ref_n = _thermal_violation_count_reference(
            result, params, line_cap, base_mva)
        assert int(info["cost_thermal_overload"]) == int(ref_n)
        np.testing.assert_allclose(
            np.array(new_state.Q_branch),
            np.array(result.Q_branch),
            atol=1e-5,
            err_msg="State branch Q should match standalone PF for same net load",
        )

    def test_cost_continuous_includes_thermal_excess(self, env, key):
        """Continuous cost should include exact per-phase thermal overload magnitude."""
        n_nodes = 4
        n_lines = 3
        from_nodes = np.array([0, 1, 2])
        to_nodes = np.array([1, 2, 3])
        Z = np.zeros((n_lines, 3, 3), dtype=complex)
        for i in range(n_lines):
            Z[i] = (0.05 + 0.1j) * np.eye(3)
        T = 4
        P_base = jnp.ones(n_lines * 3) * 0.15
        Q_base = jnp.ones(n_lines * 3) * 0.10
        load_P = jnp.tile(P_base[None, :], (T, 1))
        load_Q = jnp.tile(Q_base[None, :], (T, 1))
        line_cap = jnp.array([0.25, 0.20, 0.20], dtype=jnp.float32)

        params = make_dist_3phase_params(
            n_nodes, from_nodes, to_nodes, Z, load_P, load_Q,
            ref_bus=0, max_steps=T, base_mva=10.0, line_cap=line_cap,
        )
        _, state = env.reset(key, params)
        act = jnp.zeros(n_lines * 3)
        _, _, _, _, _, info = env.step(key, state, act, params)
        assert float(info["cost_thermal_overload"]) > 0.0
        assert float(info["cost_continuous"]) > 0.0


class TestDistGrid3Ph_QBundleInjection:

    def _params_with_q_bundle(self, bus_idx: int = 2):
        """Single battery at ``bus_idx`` with P+Q control; DER disabled."""
        n_nodes = 4
        n_lines = 3
        from_nodes = np.array([0, 1, 2])
        to_nodes = np.array([1, 2, 3])
        Z = np.zeros((n_lines, 3, 3), dtype=complex)
        for i in range(n_lines):
            Z[i] = (0.05 + 0.1j) * np.eye(3)
        T = 4
        P_base = jnp.ones(n_lines * 3) * 0.02
        Q_base = jnp.ones(n_lines * 3) * 0.015
        load_P = jnp.tile(P_base[None, :], (T, 1))
        load_Q = jnp.tile(Q_base[None, :], (T, 1))

        bundle = BatteryBundle(
            n_devices=1,
            per_device_action_dim=2,
            per_device_obs_dim=3,
            dt_hours=0.5,
            enable_q_control=True,
            bus_idx=jnp.array([bus_idx], dtype=jnp.int32),
            power_max=jnp.array([0.3], dtype=jnp.float32),
            s_rated=jnp.array([0.5], dtype=jnp.float32),
            capacity=jnp.array([1.0], dtype=jnp.float32),
            soc_min=jnp.array([0.1], dtype=jnp.float32),
            soc_max=jnp.array([0.9], dtype=jnp.float32),
            eta_charge=jnp.array([0.95], dtype=jnp.float32),
            eta_discharge=jnp.array([0.95], dtype=jnp.float32),
            initial_soc=jnp.array([0.5], dtype=jnp.float32),
        )

        return make_dist_3phase_params(
            n_nodes, from_nodes, to_nodes, Z,
            load_P, load_Q,
            ref_bus=0, max_steps=T, base_mva=10.0,
            resources=(bundle,),
            include_der=False,
        )

    def test_nonzero_q_action_changes_branch_reactive_flow(self, env, key):
        """Reactive injection from bundle must change ``Q_branch`` vs Q_norm=0."""
        params = self._params_with_q_bundle(bus_idx=2)
        _, s0 = env.reset(key, params)
        act_q0 = jnp.array([0.0, 0.0], dtype=jnp.float32)
        _, s_q0, *_ = env.step(key, s0, act_q0, params)
        _, s1 = env.reset(key, params)
        act_qmax = jnp.array([0.0, 1.0], dtype=jnp.float32)
        _, s_q1, *_ = env.step(key, s1, act_qmax, params)
        assert not jnp.allclose(s_q0.Q_branch, s_q1.Q_branch, atol=1e-4)

    def test_q_net_matches_standalone_pf(self, env, key):
        """Env PF solution equals ``bfs_3phase_power_flow(P_net, Q_net)`` with same net Q."""
        params = self._params_with_q_bundle(bus_idx=2)
        ref_bus = 0
        _, s0 = env.reset(key, params)
        bundle = params.resources[0]
        act = jnp.array([0.1, 0.7], dtype=jnp.float32)
        _, p_inj, q_inj, _, _ = bundle.step(s0.resource_states[0], act, {})

        nonref_idx = int(
            np.where(
                np.asarray(bundle.bus_idx) < ref_bus,
                np.asarray(bundle.bus_idx),
                np.asarray(bundle.bus_idx) - 1,
            )[0])
        n_nonref_3ph = 3 * (params.topo.n_nodes - 1)
        bundle_p_pu = jnp.zeros((n_nonref_3ph,), dtype=jnp.float32)
        bundle_q_pu = jnp.zeros((n_nonref_3ph,), dtype=jnp.float32)
        p_per = p_inj[0] / 3.0 / params.base_mva
        q_per = q_inj[0] / 3.0 / params.base_mva
        for ph in range(3):
            bundle_p_pu = bundle_p_pu.at[3 * nonref_idx + ph].add(p_per)
            bundle_q_pu = bundle_q_pu.at[3 * nonref_idx + ph].add(q_per)

        P_net = params.load_P_3ph[0] - bundle_p_pu
        Q_net = params.load_Q_3ph[0] - bundle_q_pu

        ref = bfs_3phase_power_flow(
            params.topo, P_net, Q_net, params.bfs_max_iter, params.bfs_tol,
        )
        _, s_env, *_ = env.step(key, s0, act, params)
        np.testing.assert_allclose(
            np.array(s_env.Q_branch), np.array(ref.Q_branch), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(
            np.array(s_env.v_mag), np.array(ref.v_mag), rtol=1e-4, atol=1e-4)
