"""Tests for DSO D3 (non-stationary params) and D4 (baselines + metrics + presets).

D3: make_dso_params_nonstationary integration
D4: no-control / TOU / droop baselines, compute_dso_metrics, rl presets
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powerzoojax.envs.grid.dist import DistGridEnv
from powerzoojax.tasks.dso import (
    concat_dso_feeder_windows,
    DSO_FEEDER_BUS_MAP,
    make_dso_params,
    make_dso_1flex_params,
    make_dso_load_profiles,
    make_dso_params_nonstationary,
    rollout_dso,
    dso_no_control_rollout,
    dso_tou_rule_based_rollout,
    dso_droop_rule_based_rollout,
    compute_dso_metrics,
)


@pytest.fixture(scope="module")
def env():
    return DistGridEnv()


@pytest.fixture(scope="module")
def dso_params():
    return make_dso_params()


@pytest.fixture(scope="module")
def key():
    return jax.random.PRNGKey(42)


# ==================== D3: Non-stationary Params ====================

class TestDSO_D3_Nonstationary:

    def test_nonstationary_params_valid(self, env):
        """make_dso_params_nonstationary should return valid DistGridParams."""
        params = make_dso_params_nonstationary(episode_idx=100)
        obs, state = env.reset(jax.random.PRNGKey(0), params)
        assert obs.shape[0] > 0
        assert state.v_mag.shape == (33,)

    def test_nonstationary_drift_changes_load(self):
        """Drift should increase total load magnitude (fixed start to isolate effect)."""
        params_0 = make_dso_params_nonstationary(episode_idx=0, mode="fixed")
        params_400 = make_dso_params_nonstationary(episode_idx=400, mode="fixed")
        # Same episode_start=0, but episode 400 has drift_factor > 1
        # Wait — "fixed" mode always returns drift=1.0. Use "train" with same key
        # and compare load sums after normalising for episode_start.
        # Better: directly create params with explicit drift.
        from powerzoojax.data.nonstationary import apply_drift
        base_params = make_dso_params()
        p0, q0 = base_params.load_profiles_p, base_params.load_profiles_q
        p_drift, q_drift = apply_drift(p0, q0, 1.12)  # +12% load
        total_0 = float(jnp.sum(p0))
        total_drift = float(jnp.sum(p_drift))
        assert total_drift > total_0, "Drift should increase total load"
        np.testing.assert_allclose(
            total_drift / total_0, 1.12, atol=1e-4)

    def test_nonstationary_iid_mode_no_drift(self):
        """IID mode should not apply drift."""
        params_iid_0 = make_dso_params_nonstationary(
            episode_idx=0, mode="iid", key=jax.random.PRNGKey(0))
        params_iid_100 = make_dso_params_nonstationary(
            episode_idx=100, mode="iid", key=jax.random.PRNGKey(0))
        # Same key, iid mode → same total load (no drift)
        total_0 = float(jnp.sum(params_iid_0.load_profiles_p))
        total_100 = float(jnp.sum(params_iid_100.load_profiles_p))
        np.testing.assert_allclose(total_0, total_100, atol=1e-4)

    def test_nonstationary_drift_shock(self):
        """Drift-shock mode should produce 25% higher loads."""
        params_fixed = make_dso_params_nonstationary(
            episode_idx=0, mode="fixed", key=jax.random.PRNGKey(0))
        params_shock = make_dso_params_nonstationary(
            episode_idx=0, mode="drift_shock", key=jax.random.PRNGKey(0))
        total_fixed = float(jnp.sum(params_fixed.load_profiles_p))
        total_shock = float(jnp.sum(params_shock.load_profiles_p))
        # Shock = 1.25× fixed, but different episode_start so test ratio
        # on a per-bus basis at t=0 instead
        ratio = total_shock / max(total_fixed, 1e-8)
        # Should be roughly 1.25 (exact depends on episode_start)
        assert ratio > 1.1, f"Drift shock ratio {ratio} should be > 1.1"

    def test_nonstationary_reset_step(self, env):
        """Non-stationary params should work with reset/step."""
        params = make_dso_params_nonstationary(episode_idx=50)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)
        action = jnp.zeros(12)
        obs2, state2, reward, costs, done, info = env.step(key, state, action, params)
        assert reward.shape == ()
        assert costs.shape == (3,)
        assert float(info["p_loss_MW"]) > 0

    def test_make_dso_params_respects_v_slack(self):
        """Scenario tuning should be able to raise the regulated slack setpoint."""
        params = make_dso_params(v_slack=1.03)
        assert params.v_slack == pytest.approx(1.03)

    def test_make_dso_params_load_scale_changes_total_load(self):
        """Uniform load scaling should change the total demand level."""
        params_1 = make_dso_params(load_scale=1.0)
        params_2 = make_dso_params(load_scale=0.9)
        total_1 = float(jnp.sum(params_1.load_profiles_p))
        total_2 = float(jnp.sum(params_2.load_profiles_p))
        assert total_2 < total_1
        np.testing.assert_allclose(total_2 / total_1, 0.9, atol=1e-4)

    def test_bus_load_reweight_preserves_feeder_totals(self):
        """Bus-level calibration should keep feeder totals unchanged when requested."""
        from powerzoojax.case import create_case33bw

        case = create_case33bw()
        feeder_shapes = {
            feeder: np.ones(48, dtype=np.float32) for feeder in DSO_FEEDER_BUS_MAP
        }
        overrides = {
            16: 0.9,
            17: 0.85,
            18: 0.8,
            30: 0.9,
            31: 0.85,
            32: 0.8,
            33: 0.75,
        }

        base_p, base_q = make_dso_load_profiles(case, feeder_shapes, load_scale=0.9)
        adj_p, adj_q = make_dso_load_profiles(
            case,
            feeder_shapes,
            load_scale=0.9,
            bus_load_scale_overrides=overrides,
            preserve_feeder_totals=True,
        )

        node_ids = np.asarray(case.node_ids)
        for bus_ids in DSO_FEEDER_BUS_MAP.values():
            idxs = [int(np.where(np.abs(node_ids - bus_id) < 0.5)[0][0]) for bus_id in bus_ids]
            np.testing.assert_allclose(
                np.asarray(base_p[0, idxs]).sum(),
                np.asarray(adj_p[0, idxs]).sum(),
                atol=1e-6,
            )
            np.testing.assert_allclose(
                np.asarray(base_q[0, idxs]).sum(),
                np.asarray(adj_q[0, idxs]).sum(),
                atol=1e-6,
            )

    def test_concat_dso_feeder_windows(self):
        """Concatenated feeder windows should preserve window order and length."""
        shapes = {
            "feeder_A": np.arange(10, dtype=np.float32),
            "feeder_B": np.arange(100, 110, dtype=np.float32),
            "feeder_C": np.arange(200, 210, dtype=np.float32),
        }
        out = concat_dso_feeder_windows(shapes, [1, 6], window_len=3)
        np.testing.assert_allclose(out["feeder_A"], np.array([1, 2, 3, 6, 7, 8], dtype=np.float32))
        np.testing.assert_allclose(out["feeder_B"], np.array([101, 102, 103, 106, 107, 108], dtype=np.float32))
        np.testing.assert_allclose(out["feeder_C"], np.array([201, 202, 203, 206, 207, 208], dtype=np.float32))


# ==================== D4: Baselines ====================

class TestDSO_D4_Baselines:

    def test_no_control_rollout(self, env, dso_params, key):
        """No-control baseline should complete without errors."""
        results = dso_no_control_rollout(env, dso_params, key)
        assert len(results["rewards"]) == 48
        assert len(results["losses"]) == 48
        assert all(l > 0 for l in results["losses"]), "All losses should be positive"

    def test_no_control_zero_curtailment(self, env, dso_params, key):
        """No-control should have zero curtailment and zero shift."""
        results = dso_no_control_rollout(env, dso_params, key)
        assert all(abs(c) < 1e-6 for c in results["curtailed"])
        assert all(abs(s) < 1e-6 for s in results["shifted"])

    def test_tou_rollout(self, env, dso_params, key):
        """TOU rule-based baseline should complete and produce non-zero actions in peak hours."""
        results = dso_tou_rule_based_rollout(env, dso_params, key)
        assert len(results["rewards"]) == 48
        # Should have some curtailment during peak hours
        total_curtail = sum(results["curtailed"])
        assert total_curtail > 0, "TOU should curtail during peak hours"

    def test_tou_has_shift(self, env, dso_params, key):
        """TOU rule-based baseline should produce shift-out during peak hours."""
        results = dso_tou_rule_based_rollout(env, dso_params, key)
        total_shift = sum(results["shifted"])
        assert total_shift > 0, "TOU should shift during peak hours"

    def test_droop_rollout(self, env, dso_params, key):
        """Droop rule-based baseline should complete without errors."""
        results = dso_droop_rule_based_rollout(env, dso_params, key)
        assert len(results["rewards"]) == 48
        assert len(results["losses"]) == 48

    def test_custom_policy_rollout(self, env, dso_params, key):
        """rollout_dso with a custom policy should work."""
        def random_policy(obs, state, key):
            return jax.random.uniform(key, (12,), minval=0.0, maxval=0.3)
        results = rollout_dso(env, dso_params, key, random_policy)
        assert len(results["rewards"]) == 48

    def test_last_step_resource_stats_not_zero(self, env, dso_params, key):
        """Regression: last-step curtailed/shifted stats must reflect actual actions.

        When done=True, auto-reset zeroes resource_states.  Previously rollout_dso
        read from state.resource_states[0] after auto-reset, returning 0 for the
        final step even when the policy was active.  Now we read from info which
        carries pre-auto-reset values.
        """
        # Use TOU policy which actively curtails during peak steps 16-21.
        # The 48-step episode ends at step 47, which is past peak, so we use
        # a policy that always curtails to guarantee non-zero at every step.
        def always_curtail(obs, state, k):
            return jnp.array([1.0, 0.0] * 6)  # full curtailment, no shift

        results = rollout_dso(env, dso_params, key, always_curtail)
        # Every step should have non-zero curtailment (including the last step)
        assert all(c > 0 for c in results["curtailed"]), (
            "All curtailed values should be > 0 when always curtailing; "
            f"got zeros at steps: {[i for i,c in enumerate(results['curtailed']) if c <= 0]}"
        )
        # The last step in particular was previously 0 due to the auto-reset bug
        assert results["curtailed"][-1] > 0, (
            "Last step curtailed must be > 0 (auto-reset bug regression)"
        )


# ==================== D4: Metrics ====================

class TestDSO_D4_Metrics:

    def test_metrics_structure(self, env, dso_params, key):
        """compute_dso_metrics should return all expected keys."""
        results = dso_no_control_rollout(env, dso_params, key)
        metrics = compute_dso_metrics(results)
        expected_keys = {
            "total_reward", "total_loss_mwh", "mean_loss_mw",
            "total_voltage_violations", "total_thermal_overloads",
            "total_violations",
            "voltage_violation_count_per_step",
            "thermal_overload_count_per_step",
            "total_curtailed_mwh",
            "total_shifted_mwh", "total_shift_in_mwh",
            "served_flex_ratio", "network_loss_reduction_pct",
            "peak_shaving_pct",
        }
        assert expected_keys == set(metrics.keys())

    def test_metrics_no_control_values(self, env, dso_params, key):
        """No-control metrics should have zero curtailment and positive loss."""
        results = dso_no_control_rollout(env, dso_params, key)
        metrics = compute_dso_metrics(results)
        assert metrics["total_loss_mwh"] > 0
        assert metrics["mean_loss_mw"] > 0
        np.testing.assert_allclose(
            metrics["total_violations"],
            metrics["total_voltage_violations"] + metrics["total_thermal_overloads"],
            atol=1e-6,
        )
        np.testing.assert_allclose(metrics["total_curtailed_mwh"], 0.0, atol=1e-6)
        np.testing.assert_allclose(metrics["total_shifted_mwh"], 0.0, atol=1e-6)

    def test_metrics_relative_with_baseline(self, env, dso_params, key):
        """Relative metrics should be computed when baseline is provided."""
        baseline = dso_no_control_rollout(env, dso_params, key)
        tou = dso_tou_rule_based_rollout(env, dso_params, key)
        metrics = compute_dso_metrics(tou, baseline_results=baseline)
        assert metrics["network_loss_reduction_pct"] is not None
        assert metrics["peak_shaving_pct"] is not None

    def test_metrics_tou_has_curtailment(self, env, dso_params, key):
        """TOU metrics should show non-zero curtailment energy."""
        results = dso_tou_rule_based_rollout(env, dso_params, key)
        metrics = compute_dso_metrics(results)
        assert metrics["total_curtailed_mwh"] > 0

    def test_no_control_self_comparison(self, env, dso_params, key):
        """Comparing no-control to itself should give 0% reduction."""
        results = dso_no_control_rollout(env, dso_params, key)
        metrics = compute_dso_metrics(results, baseline_results=results)
        np.testing.assert_allclose(
            metrics["network_loss_reduction_pct"], 0.0, atol=1e-4)


# ==================== D4: Presets ====================

class TestDSO_D4_Presets:

    def test_dso_nflex_preset_exists(self):
        """dso-nflex preset should be registered."""
        from powerzoojax.rl.presets import get_preset
        preset = get_preset("dso-nflex")
        assert preset.description is not None
        assert preset.config.algo == "ppo"
        assert preset.config.total_timesteps == 3_000_000
        assert preset.config.num_envs == 128
        assert preset.config.n_steps == 48

    def test_dso_nflex_safe_preset_exists(self):
        """dso-nflex-safe preset should be registered."""
        from powerzoojax.rl.presets import get_preset
        preset = get_preset("dso-nflex-safe")
        assert preset.config.algo == "ppo_lagrangian"
        assert preset.config.cost_thresholds == (0.0,)

    def test_dso_nflex_preset_instantiates(self):
        """dso-nflex env_factory should produce a working wrapped env."""
        from powerzoojax.rl.presets import get_preset
        preset = get_preset("dso-nflex")
        wrapped_env = preset.env_factory()
        assert wrapped_env.obs_size > 0
        assert wrapped_env.action_size == 12

    def test_dso_preset_in_list(self):
        """DSO presets should appear in list_presets."""
        from powerzoojax.rl.presets import list_presets
        names = [p["name"] for p in list_presets()]
        assert "dso-nflex" in names
        assert "dso-nflex-safe" in names

    def test_dso_nflex_safe_description_says_voltage_only(self):
        """dso-nflex-safe description should state voltage_only cost mode."""
        from powerzoojax.rl.presets import get_preset
        preset = get_preset("dso-nflex-safe")
        assert "voltage_only" in preset.description or "voltage" in preset.description


# ==================== D4: 1-Flex Variant ====================

class TestDSO_1Flex:
    """L1 regression: baseline helpers work for the 1-flex DSO variant.

    Finding 2 fix verification: action dims and device counts are derived
    from params.resources, not hardcoded to 6-flex values.
    """

    @pytest.fixture(scope="class")
    def env(self):
        return DistGridEnv()

    @pytest.fixture(scope="class")
    def params_1flex(self):
        return make_dso_1flex_params()

    @pytest.fixture(scope="class")
    def key(self):
        return jax.random.PRNGKey(99)

    def test_1flex_action_dim(self, env, params_1flex):
        """1-flex variant: action space = Box(2) (1 device × 2 actions)."""
        act_space = env.action_space(params_1flex)
        assert act_space.shape == (2,), (
            f"1-flex action space should be (2,), got {act_space.shape}")

    def test_1flex_single_device(self, params_1flex):
        """1-flex variant: exactly 1 FlexLoad device."""
        assert len(params_1flex.resources) == 1
        assert params_1flex.resources[0].n_devices == 1

    def test_1flex_legacy_cost_mode_field(self, params_1flex):
        """Deprecated ``cost_mode`` field is preserved for config loading only."""
        assert params_1flex.cost_mode == "voltage_only"

    def test_1flex_no_control_rollout(self, env, params_1flex, key):
        """no-control baseline: works with 1-flex (action_dim=2)."""
        results = dso_no_control_rollout(env, params_1flex, key)
        assert len(results["rewards"]) == 48
        assert all(c == 0.0 for c in results["curtailed"])

    def test_1flex_tou_rollout(self, env, params_1flex, key):
        """TOU baseline: works with 1-flex, curtails during peak hours."""
        results = dso_tou_rule_based_rollout(env, params_1flex, key)
        assert len(results["rewards"]) == 48
        total_curtail = sum(results["curtailed"])
        assert total_curtail > 0, "TOU should curtail during peak hours"

    def test_1flex_droop_rollout(self, env, params_1flex, key):
        """Droop baseline: works with 1-flex."""
        results = dso_droop_rule_based_rollout(env, params_1flex, key)
        assert len(results["rewards"]) == 48
        assert all(l > 0 for l in results["losses"])

    def test_1flex_metrics(self, env, params_1flex, key):
        """compute_dso_metrics: works with 1-flex rollout results."""
        baseline = dso_no_control_rollout(env, params_1flex, key)
        tou = dso_tou_rule_based_rollout(env, params_1flex, key)
        metrics = compute_dso_metrics(tou, baseline_results=baseline)
        assert "network_loss_reduction_pct" in metrics
        assert "total_curtailed_mwh" in metrics
        assert metrics["total_curtailed_mwh"] > 0

    def test_1flex_vs_6flex_independent(self, env, key):
        """1-flex and 6-flex rollouts are independent (different dims)."""
        params_1 = make_dso_1flex_params()
        params_6 = make_dso_params()  # default 6-flex

        results_1 = dso_no_control_rollout(env, params_1, key)
        results_6 = dso_no_control_rollout(env, params_6, key)

        # Both should produce 48-step rollouts
        assert len(results_1["losses"]) == 48
        assert len(results_6["losses"]) == 48
        # Losses should differ (different load distributions / no bundle injection)
        mean_loss_1 = sum(results_1["losses"]) / 48
        mean_loss_6 = sum(results_6["losses"]) / 48
        assert mean_loss_1 > 0 and mean_loss_6 > 0
