"""Tests for DSO task — L0 JAX contract + L1 physics + L1.5 integration.

Covers D1 (feeder-shape mapping) and D2 (6× FlexLoad preset).

L0: JIT, vmap, scan rollout, auto-reset, pytree stability.
L1: Action dim = 12, load profile shape, FlexLoad injection physics.
L1.5: Feeder shape → voltage effect, FlexLoad curtailment → loss change.
"""

import jax
import jax.numpy as jnp
import jax.tree_util as tu
import numpy as np
import pytest

from powerzoojax.case import create_case33bw
from powerzoojax.envs.grid.dist import DistGridEnv, make_dist_params
from powerzoojax.tasks.dso import (
    DSO_FEEDER_BUS_MAP,
    DSO_FLEXLOAD_CONFIG,
    DSO_V_MIN,
    DSO_V_MAX,
    DSOTask,
    load_dso_feeder_shapes,
    make_dso_flexload_bundle,
    make_dso_load_profiles,
    make_dso_params,
    make_dso_params_from_split,
    make_dso_1flex_params,
    make_synthetic_feeder_shapes,
)


@pytest.fixture(scope="module")
def case33():
    return create_case33bw()


@pytest.fixture(scope="module")
def dso_params(case33):
    return make_dso_params(case33)


@pytest.fixture(scope="module")
def env():
    return DistGridEnv()


@pytest.fixture(scope="module")
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture(scope="module")
def init(env, key, dso_params):
    return env.reset(key, dso_params)


# ==================== D1: Feeder-Shape Mapping ====================

class TestDSO_D1_FeederMapping:

    class _FakeAusgridLoader:
        def load_signals(
            self,
            signals,
            *,
            source,
            region,
            start_date,
            end_date,
            resample,
        ):
            assert signals == ["load.actual_mw"]
            assert source == "ausgrid"
            seed = sum(ord(c) for c in region) % 5
            pattern = np.array([1.0, 1.4, 0.8, 1.2], dtype=np.float32)
            rolled = np.roll(pattern, seed)
            import pandas as pd

            idx = pd.date_range(
                start="2024-05-01", periods=len(rolled), freq="30min", tz="UTC"
            )
            return pd.DataFrame(
                {
                    "datetime": idx,
                    "region": region,
                    "load.actual_mw": rolled,
                }
            )

    def test_feeder_bus_map_covers_all_non_slack(self):
        """Feeders A+B+C should cover buses 2-33 (all except slack bus 1)."""
        all_buses = set()
        for buses in DSO_FEEDER_BUS_MAP.values():
            all_buses.update(buses)
        expected = set(range(2, 34))
        assert all_buses == expected

    def test_feeder_bus_map_no_overlap(self):
        """No bus should belong to more than one feeder."""
        seen = set()
        for buses in DSO_FEEDER_BUS_MAP.values():
            overlap = seen & set(buses)
            assert len(overlap) == 0, f"Overlapping buses: {overlap}"
            seen.update(buses)

    def test_synthetic_shapes_mean_one(self):
        """Synthetic feeder shapes should be normalised to mean ~= 1."""
        shapes = make_synthetic_feeder_shapes(max_steps=48)
        for name, shape in shapes.items():
            assert shape.shape == (48,), f"{name} wrong shape"
            np.testing.assert_allclose(
                shape.mean(), 1.0, atol=0.01,
                err_msg=f"{name} mean should be ~1.0")

    def test_synthetic_shapes_positive(self):
        """All shape values should be positive (load cannot be negative)."""
        shapes = make_synthetic_feeder_shapes(max_steps=48)
        for name, shape in shapes.items():
            assert np.all(shape > 0), f"{name} has non-positive values"

    def test_load_dso_feeder_shapes(self):
        """Real-data split loader should return all 3 mean-normalised feeder shapes."""
        shapes = load_dso_feeder_shapes(
            data_loader=self._FakeAusgridLoader(),
            role="iid",
            resample="30min",
        )
        assert set(shapes.keys()) == set(DSO_FEEDER_BUS_MAP.keys())
        for shape in shapes.values():
            assert shape.ndim == 1
            assert len(shape) == 4
            assert np.all(shape > 0)
            np.testing.assert_allclose(shape.mean(), 1.0, atol=1e-6)

    def test_make_dso_load_profiles_shape(self, case33):
        """Load profiles should have shape (max_steps, n_bus)."""
        shapes = make_synthetic_feeder_shapes(max_steps=48)
        load_p, load_q = make_dso_load_profiles(case33, shapes, max_steps=48)
        assert load_p.shape == (48, 33)
        assert load_q.shape == (48, 33)

    def test_load_profiles_slack_constant(self, case33):
        """Slack bus (bus 1, index 0) load should be constant (zero in case33bw)."""
        shapes = make_synthetic_feeder_shapes(max_steps=48)
        load_p, _ = make_dso_load_profiles(case33, shapes, max_steps=48)
        slack_load = load_p[:, 0]
        np.testing.assert_allclose(
            float(slack_load.std()), 0.0, atol=1e-8,
            err_msg="Slack bus load should be constant across time")

    def test_load_profiles_time_varying(self, case33):
        """Non-slack bus loads should vary over time (driven by feeder shapes)."""
        shapes = make_synthetic_feeder_shapes(max_steps=48)
        load_p, _ = make_dso_load_profiles(case33, shapes, max_steps=48)
        # Bus 2 (index 1) is in feeder A, should have time variation
        bus2_std = float(load_p[:, 1].std())
        assert bus2_std > 0, "Bus 2 load should vary over time"

    def test_load_profiles_power_factor_preserved(self, case33):
        """Q/P ratio should be preserved from case33bw base data."""
        shapes = make_synthetic_feeder_shapes(max_steps=48)
        load_p, load_q = make_dso_load_profiles(case33, shapes, max_steps=48)
        base_mva = float(case33.base_mva)
        # Bus 2 (index 1): base pd=0.1, qd=0.06 → ratio = 0.6
        base_p = float(case33.node_pd[1]) / base_mva
        base_q = float(case33.node_qd[1]) / base_mva
        if base_p > 1e-8:
            expected_ratio = base_q / base_p
            actual_ratio = float(load_q[0, 1]) / float(load_p[0, 1])
            np.testing.assert_allclose(
                actual_ratio, expected_ratio, atol=1e-4,
                err_msg="Q/P ratio should be preserved from base case")

    def test_episode_start_offset(self, case33):
        """Different episode_start should give different load profiles."""
        shapes = make_synthetic_feeder_shapes(max_steps=96)
        load_p_0, _ = make_dso_load_profiles(
            case33, shapes, max_steps=48, episode_start=0)
        load_p_48, _ = make_dso_load_profiles(
            case33, shapes, max_steps=48, episode_start=48)
        # They should differ (different windows of the same shape)
        assert not jnp.allclose(load_p_0, load_p_48)

    def test_make_dso_params_from_split(self, case33):
        """Canonical split factory should build real-data-driven params directly."""
        params = make_dso_params_from_split(
            case33,
            role="summer_ood",
            data_loader=self._FakeAusgridLoader(),
            max_steps=4,
        )
        assert params.load_profiles_p.shape == (4, 33)
        assert params.load_profiles_q.shape == (4, 33)
        assert params.cost_mode == "voltage_only"
        assert float(params.load_profiles_p[:, 1].std()) > 0


# ==================== D2: FlexLoad Bundle ====================

class TestDSO_D2_FlexLoadBundle:

    def test_flexload_config_6_devices(self):
        """Standard DSO preset uses exactly six FlexLoad devices."""
        assert len(DSO_FLEXLOAD_CONFIG) == 6

    def test_flexload_bus_ids(self):
        """FlexLoad buses should be [6, 14, 18, 22, 28, 33]."""
        bus_ids = [cfg["bus_id"] for cfg in DSO_FLEXLOAD_CONFIG]
        assert bus_ids == [6, 14, 18, 22, 28, 33]

    def test_make_dso_flexload_bundle(self, case33):
        """Bundle should have correct action/obs dimensions."""
        bundle = make_dso_flexload_bundle(case33)
        assert bundle.n_devices == 6
        assert bundle.action_dim == 12  # 6 devices × 2 actions
        assert bundle.obs_dim == 30     # 6 devices × 5 obs

    def test_dso_params_action_dim(self, dso_params):
        """Action space should be Box(12) — 6 FlexLoads × 2 actions, no DER."""
        env = DistGridEnv()
        act_space = env.action_space(dso_params)
        assert act_space.shape == (12,)

    def test_dso_params_voltage_limits(self, dso_params):
        """Voltage limits should be [0.94, 1.06] per DSO spec."""
        assert dso_params.v_min == DSO_V_MIN
        assert dso_params.v_max == DSO_V_MAX

    def test_dso_params_no_legacy_der(self, dso_params):
        """include_der should be False — agent controls only FlexLoads."""
        assert dso_params.include_der is False

    def test_dso_params_one_resource(self, dso_params):
        """Should have exactly 1 resource bundle (FlexLoadBundle)."""
        assert len(dso_params.resources) == 1

    def test_dso_params_load_profiles_set(self, dso_params):
        """Load profiles should have been set (not flat default)."""
        lp = dso_params.load_profiles_p
        assert lp.shape == (48, 33)
        # Should have time variation (not constant)
        assert float(lp[:, 1].std()) > 0


# ==================== L0: JAX Contract ====================

class TestDSO_L0_JAX:

    def test_reset_jit(self, env, key, dso_params):
        obs, state = env.reset(key, dso_params)
        assert obs.shape[0] > 0
        assert state.v_mag.shape == (33,)

    def test_step_jit(self, env, key, init, dso_params):
        obs, state = init
        action = jnp.zeros(12)
        obs2, state2, reward, costs, done, info = env.step(
            key, state, action, dso_params
        )
        assert obs2.shape == obs.shape
        assert reward.shape == ()
        assert costs.shape == (3,)

    def test_vmap_reset(self, env, dso_params):
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        obs_b, state_b = jax.vmap(lambda k: env.reset(k, dso_params))(keys)
        assert obs_b.shape[0] == 4
        assert state_b.v_mag.shape == (4, 33)

    def test_vmap_step(self, env, dso_params):
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        obs_b, state_b = jax.vmap(lambda k: env.reset(k, dso_params))(keys)
        actions = jnp.zeros((4, 12))
        keys2 = jax.random.split(jax.random.PRNGKey(1), 4)
        obs2, state2, rew, costs, done, info = jax.vmap(
            lambda k, s, a: env.step(k, s, a, dso_params)
        )(keys2, state_b, actions)
        assert obs2.shape[0] == 4
        assert rew.shape == (4,)
        assert costs.shape == (4, 3)

    def test_scan_rollout(self, env, key, dso_params):
        obs, state = env.reset(key, dso_params)
        action = jnp.zeros(12)

        def step_fn(carry, _):
            s = carry
            k = jax.random.PRNGKey(0)
            obs, new_s, rew, costs, done, info = env.step(k, s, action, dso_params)
            return new_s, (obs, rew, costs, done)

        final_state, (obs_traj, rew_traj, cost_traj, done_traj) = jax.lax.scan(
            step_fn, state, None, length=dso_params.max_steps)
        assert obs_traj.shape[0] == dso_params.max_steps
        assert rew_traj.shape == (dso_params.max_steps,)
        assert cost_traj.shape == (dso_params.max_steps, 3)

    def test_auto_reset(self, env, key, dso_params):
        """After max_steps, state should be auto-reset."""
        obs, state = env.reset(key, dso_params)
        action = jnp.zeros(12)

        for _ in range(dso_params.max_steps):
            obs, state, reward, costs, done, info = env.step(
                key, state, action, dso_params
            )

        assert int(state.time_step) == 0

    def test_state_pytree(self, init):
        obs, state = init
        leaves = tu.tree_leaves(state)
        assert len(leaves) > 0
        # Should be flattenable without error
        flat, treedef = tu.tree_flatten(state)
        restored = treedef.unflatten(flat)
        assert restored.time_step.shape == state.time_step.shape

    def test_make_jaxpr(self, env, key, init, dso_params):
        obs, state = init
        action = jnp.zeros(12)
        jaxpr = jax.make_jaxpr(
            lambda k, s, a: env.step(k, s, a, dso_params)
        )(key, state, action)
        assert jaxpr is not None


# ==================== L1: Physics ====================

class TestDSO_L1_Physics:

    def test_obs_shape(self, env, init, dso_params):
        """Obs should include grid features + FlexLoad bundle obs."""
        obs, _ = init
        n = 33
        m = dso_params.topo.n_lines
        # grid obs: 3*n + 2*m + 2 (v_norm + p_flow + q_flow + p_load + q_load + sin/cos)
        # bundle obs: 30 (6 devices × 5)
        expected = 3 * n + 2 * m + 2 + 30
        assert obs.shape == (expected,)

    def test_reward_negative(self, env, key, init, dso_params):
        """Reward = -loss_penalty * loss_MW, should be <= 0."""
        _, state = init
        action = jnp.zeros(12)
        _, _, reward, _, _, _ = env.step(key, state, action, dso_params)
        assert float(reward) <= 0.0

    def test_reward_cost_separation(self, env, key, init, dso_params):
        """Reward = -weight * loss; cost = violation counts (separate channels)."""
        _, state = init
        action = jnp.zeros(12)
        _, _, reward, _, _, info = env.step(key, state, action, dso_params)
        expected_reward = -dso_params.loss_penalty_weight * info["p_loss_MW"]
        np.testing.assert_allclose(
            float(reward), float(expected_reward), atol=1e-5)

    def test_loss_positive(self, env, key, init, dso_params):
        """Network loss should be positive in base case."""
        _, state = init
        action = jnp.zeros(12)
        _, _, _, _, _, info = env.step(key, state, action, dso_params)
        assert float(info["p_loss_MW"]) > 0

    def test_zero_action_passthrough(self, env, key, init, dso_params):
        """With action=0, FlexLoads should be passthrough (no injection)."""
        _, state = init
        action = jnp.zeros(12)
        _, new_state, _, _, _, _ = env.step(key, state, action, dso_params)
        # Bundle state: all curtailed/shift should be ~0
        bs = new_state.resource_states[0]
        np.testing.assert_allclose(
            float(jnp.sum(bs.curtailed_mw)), 0.0, atol=1e-6)
        np.testing.assert_allclose(
            float(jnp.sum(bs.shift_out_mw)), 0.0, atol=1e-6)

    def test_curtail_action_reduces_load(self, env, key, init, dso_params):
        """Curtailment action should inject positive power (reduce net load)."""
        _, state = init
        # Full curtailment on all 6 devices, no shift
        action = jnp.array([1.0, 0.0] * 6)
        _, new_state, _, _, _, _ = env.step(key, state, action, dso_params)
        bs = new_state.resource_states[0]
        total_curtail = float(jnp.sum(bs.curtailed_mw))
        assert total_curtail > 0, "Curtailment action should produce positive curtail"

    def test_info_contains_dso_metrics(self, env, key, init, dso_params):
        """Info dict should contain DSO-relevant metrics."""
        _, state = init
        action = jnp.zeros(12)
        _, _, _, costs, _, info = env.step(key, state, action, dso_params)
        assert "p_loss_MW" in info
        assert "cost_voltage_violation" in info
        assert "cost_thermal_overload" in info
        assert "cost_resource" in info
        assert "cost_sum" in info
        assert "bfs_converged" in info
        assert costs.shape == (3,)

    def test_constraint_vector_matches_named_costs(self, env, key, init, dso_params):
        """Explicit CMDP vector must align with named info entries.

        Regression test for the bundle_cost aggregation bug where iterating over
        cost_info.values() included per-device vectors, making bundle_cost a vector.
        """
        _, state = init
        action = jnp.array([1.0, 0.5] * 6)  # non-zero action to trigger cost path
        _, _, _, costs, _, info = env.step(key, state, action, dso_params)
        cost_resource = info["cost_resource"]
        cost_sum = info["cost_sum"]
        assert costs.shape == (3,), f"cost vector should have shape (3,), got {costs.shape}"
        assert cost_resource.shape == (), (
            f"info['cost_resource'] should be scalar, got shape {cost_resource.shape}"
        )
        assert cost_sum.shape == (), (
            f"info['cost_sum'] should be scalar, got shape {cost_sum.shape}"
        )
        np.testing.assert_allclose(
            np.asarray(costs),
            np.asarray(
                [
                    info["cost_voltage_violation"],
                    info["cost_thermal_overload"],
                    info["cost_resource"],
                ]
            ),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            float(info["cost_sum"]), float(jnp.sum(costs)), atol=1e-6
        )
        # cost_resource should be positive when FlexLoad is active
        assert float(cost_resource) >= 0.0


# ==================== L1.5: Cross-Module Integration ====================

class TestDSO_L15_Integration:

    def test_curtail_changes_loss(self, env, key, init, dso_params):
        """Curtailing load should change network loss vs no-control."""
        _, state = init

        # No-control baseline
        action_zero = jnp.zeros(12)
        _, _, _, _, _, info_zero = env.step(key, state, action_zero, dso_params)
        loss_zero = float(info_zero["p_loss_MW"])

        # Full curtailment
        action_curtail = jnp.array([1.0, 0.0] * 6)
        _, _, _, _, _, info_curtail = env.step(key, state, action_curtail, dso_params)
        loss_curtail = float(info_curtail["p_loss_MW"])

        # Loss should differ
        assert abs(loss_zero - loss_curtail) > 1e-6, \
            "Curtailment should change network loss"

    def test_default_case_creation(self):
        """make_dso_params() with no args should auto-create case33bw."""
        params = make_dso_params()
        assert params.case.n_nodes == 33
        assert len(params.resources) == 1
        env = DistGridEnv()
        act_space = env.action_space(params)
        assert act_space.shape == (12,)

    def test_full_episode_rollout(self, env, key, dso_params):
        """Full 48-step episode should complete without errors."""
        obs, state = env.reset(key, dso_params)
        total_reward = 0.0
        for t in range(dso_params.max_steps):
            action = jnp.zeros(12)
            obs, state, reward, costs, done, info = env.step(
                key, state, action, dso_params
            )
            total_reward += float(reward)
        assert total_reward < 0, "Total reward should be negative (loss penalty)"


# ==================== CMDP Constraint Selection ====================

class TestDSO_CMDPSelection:
    """L1: Verify DistGridEnv always exposes the full CMDP vector.

    The benchmark-level DSO task then selects only ``voltage_violation`` from
    that full vector via ``ConstraintSpec``.
    """

    def test_constraint_names_fixed_order(self, env, dso_params):
        assert env.constraint_names(dso_params) == (
            "voltage_violation",
            "thermal_overload",
            "resource",
        )

    def test_cost_vector_matches_named_info(self, env, key, dso_params):
        _, state = env.reset(key, dso_params)
        action = jnp.zeros(12)
        _, _, _, costs, _, info = env.step(key, state, action, dso_params)
        np.testing.assert_allclose(
            np.asarray(costs),
            np.asarray(
                [
                    info["cost_voltage_violation"],
                    info["cost_thermal_overload"],
                    info["cost_resource"],
                ]
            ),
            atol=1e-6,
        )
        np.testing.assert_allclose(float(info["cost_sum"]), float(jnp.sum(costs)), atol=1e-6)

    def test_legacy_cost_mode_field_does_not_change_core_costs(self, env, key, case33):
        """Deprecated ``cost_mode`` may survive in params, but core costs stay identical."""
        params_voltage = make_dso_params(case33, cost_mode="voltage_only")
        params_aggregate = make_dso_params(case33, cost_mode="aggregate")
        _, state_voltage = env.reset(key, params_voltage)
        _, state_aggregate = env.reset(key, params_aggregate)
        action = jnp.zeros(12)
        _, _, _, costs_voltage, _, info_voltage = env.step(
            key, state_voltage, action, params_voltage
        )
        _, _, _, costs_aggregate, _, info_aggregate = env.step(
            key, state_aggregate, action, params_aggregate
        )
        np.testing.assert_allclose(
            np.asarray(costs_voltage), np.asarray(costs_aggregate), atol=1e-6
        )
        np.testing.assert_allclose(
            float(info_voltage["cost_sum"]),
            float(info_aggregate["cost_sum"]),
            atol=1e-6,
        )

    def test_dso_task_constraint_spec_selects_voltage_only(self):
        spec = DSOTask().constraint_spec()
        assert spec.selected_names == ("voltage_violation",)
        assert spec.thresholds == (0.0,)
        assert spec.fallback_weights == (1.0,)

    def test_cost_sum_is_scalar(self, env, key, dso_params):
        _, state = env.reset(key, dso_params)
        action = jnp.zeros(12)
        _, _, _, costs, _, info = env.step(key, state, action, dso_params)
        assert info["cost_sum"].shape == (), (
            f"info['cost_sum'] shape should be () got {info['cost_sum'].shape}"
        )
        assert costs.shape == (3,)

    def test_constraint_vector_jit_compatible(self, env, key, dso_params):
        _, state = env.reset(key, dso_params)
        action = jnp.zeros(12)
        step_jit = jax.jit(lambda k, s, a: env.step(k, s, a, dso_params))
        _, _, _, costs, _, info = step_jit(key, state, action)
        assert costs.shape == (3,)
        assert info["cost_sum"].shape == ()

    def test_cost_sum_always_present(self, env, key, dso_params):
        """Aggregate diagnostics should always be present in core info."""
        _, state = env.reset(key, dso_params)
        action = jnp.zeros(12)
        _, _, _, costs, _, info = env.step(key, state, action, dso_params)
        assert "cost_sum" in info
        assert "cost_voltage_violation" in info
        assert "cost_thermal_overload" in info
        assert "cost_resource" in info
        assert costs.shape == (3,)

    def test_1flex_uses_same_constraint_layout(self, env, key):
        """1-flex variant keeps the same CMDP constraint order."""
        params = make_dso_1flex_params()
        assert params.resources[0].n_devices == 1
        _, state = env.reset(key, params)
        action = jnp.zeros(2)  # 1 device × 2 actions
        _, _, _, costs, _, info = env.step(key, state, action, params)
        assert env.constraint_names(params) == (
            "voltage_violation",
            "thermal_overload",
            "resource",
        )
        np.testing.assert_allclose(
            np.asarray(costs),
            np.asarray(
                [
                    info["cost_voltage_violation"],
                    info["cost_thermal_overload"],
                    info["cost_resource"],
                ]
            ),
            atol=1e-6,
        )
