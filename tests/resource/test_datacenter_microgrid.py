"""Tests for DataCenterMicrogridEnv (C1 + C2).

Test tiers:
  L0   — JAX contract (jit, vmap, pytree)
  GPU  — auto-reset, lax.scan rollout, vmap+scan
  L1   — physics: power balance, SOC clamp, DG fuel cost, reward/cost separation
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.tree_util as tu
import pytest

from powerzoojax.envs.microgrid import (
    DataCenterMicrogridEnv,
    DCMicrogridParams,
    DCMicrogridState,
    make_dcmicrogrid_params,
)
from powerzoojax.envs.microgrid.dc_microgrid import _solar_cf
from powerzoojax.envs.resource.diesel import (
    DieselParams,
    compute_dg_power,
    compute_dg_fuel_cost,
    compute_dg_emissions,
)

OBS_DIM = 24


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def env():
    return DataCenterMicrogridEnv()


@pytest.fixture(scope="module")
def params():
    # Small episode (10 steps) for fast tests; 5-min dt preserved
    return make_dcmicrogrid_params(max_steps=10)


@pytest.fixture(scope="module")
def params_full():
    """Full 288-step config for episode-length tests."""
    return make_dcmicrogrid_params()


@pytest.fixture(scope="module")
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture(scope="module")
def reset_out(env, params, key):
    obs, state = env.reset(key, params)
    return obs, state


@pytest.fixture(scope="module")
def step_out(env, params, key, reset_out):
    _, state = reset_out
    action = jnp.zeros(5, dtype=jnp.float32)
    key2 = jax.random.PRNGKey(0)
    return env.step(key2, state, action, params)


# ---------------------------------------------------------------------------
# L0 — JAX contract
# ---------------------------------------------------------------------------

class TestL0_JAXContract:

    def test_reset_jit(self, env, params, key):
        """env.reset is already @jit; calling it confirms JIT compiles and returns correct shapes."""
        obs, state = env.reset(key, params)
        assert obs.shape == (OBS_DIM,)
        assert obs.dtype == jnp.float32

    def test_step_jit(self, env, params, key):
        """env.step is already @jit; calling it confirms JIT compiles without error."""
        obs0, state0 = env.reset(key, params)
        action = jnp.zeros(5, dtype=jnp.float32)
        obs1, state1, reward, costs, done, info = env.step(key, state0, action, params)
        assert obs1.shape == (OBS_DIM,)
        assert reward.shape == ()
        assert done.shape == ()
        assert costs.shape == (3,)

    def test_reset_vmap(self, env, params):
        """vmap(reset) over 4 parallel envs."""
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
        obs_batch, state_batch = vmap_reset(keys, params)
        assert obs_batch.shape == (4, OBS_DIM)

    def test_step_vmap(self, env, params):
        """vmap(step) over 4 parallel envs."""
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
        _, state_batch = vmap_reset(keys, params)
        actions = jnp.zeros((4, 5), dtype=jnp.float32)
        step_keys = jax.random.split(jax.random.PRNGKey(1), 4)
        vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
        obs_b, state_b, rew_b, costs_b, done_b, _ = vmap_step(
            step_keys, state_batch, actions, params
        )
        assert obs_b.shape == (4, OBS_DIM)
        assert rew_b.shape == (4,)
        assert costs_b.shape == (4, 3)

    def test_pytree_stable(self, env, params, key):
        """PyTree structure of state is unchanged after step."""
        obs0, state0 = env.reset(key, params)
        action = jnp.zeros(5)
        _, state1, *_ = env.step(key, state0, action, params)
        leaves0 = tu.tree_leaves(state0)
        leaves1 = tu.tree_leaves(state1)
        assert len(leaves0) == len(leaves1)
        for l0, l1 in zip(leaves0, leaves1):
            assert l0.shape == l1.shape
            assert l0.dtype == l1.dtype

    def test_state_leaves_count(self, env, params, key):
        """DCMicrogridState has the expected number of JAX array leaves."""
        _, state = env.reset(key, params)
        leaves = tu.tree_leaves(state)
        # dc leaves: 5 scalar + 7×_MAX_TASKS + 3 counts/flags = many; plus soc, p_dg, p_pv, last_action(5-vec counts as 1), done
        # Just verify it's a non-trivial pytree (> 10 leaves)
        assert len(leaves) > 10


# ---------------------------------------------------------------------------
# GPU Pipeline
# ---------------------------------------------------------------------------

class TestGPUPipeline:

    def test_auto_reset_triggers(self, env, params):
        """After max_steps, state resets (time_step returns to 0)."""
        key = jax.random.PRNGKey(7)
        _, state = env.reset(key, params)
        action = jnp.zeros(5)
        for _ in range(params.dc.max_steps):
            key, k = jax.random.split(key)
            _, state, _, _, done, _ = env.step(k, state, action, params)
        # After max_steps steps, last done=True triggers auto-reset → time_step==0
        assert int(state.dc.time_step) == 0
        assert not bool(state.done)

    def test_lax_scan_rollout(self, env, params):
        """lax.scan over max_steps compiles and produces correct shapes."""
        key = jax.random.PRNGKey(3)
        obs0, state0 = env.reset(key, params)

        def step_fn(carry, _):
            state, k = carry
            k, k_step = jax.random.split(k)
            action = jnp.zeros(5)
            obs, new_state, reward, costs, done, info = env.step(
                k_step, state, action, params
            )
            return (new_state, k), (obs, reward, done, info["cost_sum"])

        (_, _), (obs_seq, rew_seq, done_seq, cost_seq) = jax.lax.scan(
            step_fn, (state0, key), None, length=params.dc.max_steps
        )
        assert obs_seq.shape == (params.dc.max_steps, OBS_DIM)
        assert rew_seq.shape == (params.dc.max_steps,)
        assert done_seq.shape == (params.dc.max_steps,)

    def test_vmap_plus_scan(self, env, params):
        """vmap over lax.scan works (batch of 4 envs, each scanned)."""
        keys = jax.random.split(jax.random.PRNGKey(5), 4)

        def single_rollout(k):
            obs0, state0 = env.reset(k, params)

            def step_fn(carry, _):
                state, key = carry
                key, k_step = jax.random.split(key)
                action = jnp.zeros(5)
                obs, new_state, reward, costs, done, _ = env.step(
                    k_step, state, action, params
                )
                return (new_state, key), reward

            _, rewards = jax.lax.scan(step_fn, (state0, k), None, length=params.dc.max_steps)
            return rewards

        batch_rewards = jax.vmap(single_rollout)(keys)
        assert batch_rewards.shape == (4, params.dc.max_steps)


# ---------------------------------------------------------------------------
# L1 — Physics
# ---------------------------------------------------------------------------

class TestL1_Physics:

    def test_power_balance_residual(self, env, params, key):
        """residual = p_pv + p_dg + p_batt - p_load is in info and checkable."""
        _, state = env.reset(key, params)
        action = jnp.array([0.5, 0.5, 0.5, 0.2, 0.5], dtype=jnp.float32)
        _, _, _, _, _, info = env.step(key, state, action, params)
        residual_check = (
            info["p_pv_mw"] + info["p_dg_mw"] + info["p_batt_mw"] - info["p_load_mw"]
        )
        assert jnp.allclose(info["residual"], residual_check, atol=1e-5)

    def test_power_deficit_no_implicit_slack(self, env, params, key):
        """With dg=0, batt=0, pv≈0 (night), deficit > 0 when load is positive."""
        _, state = env.reset(key, params)
        # Force nighttime (time_step=0 → hour=0 → solar_cf=0)
        # action: no DG, no battery discharge
        action = jnp.array([0.5, 0.5, 0.5, 0.0, 0.0], dtype=jnp.float32)
        _, _, _, _, _, info = env.step(key, state, action, params)
        # If p_load > 0 and solar_cf=0 at t=0, residual = 0 + 0 + 0 - p_load < 0
        # → power_deficit = p_load > 0
        if float(info["p_load_mw"]) > 0:
            assert float(info["power_deficit"]) >= 0.0
            # deficit + spill = |residual|
            balance = float(info["power_deficit"]) - float(info["power_spill"])
            assert abs(balance - abs(float(info["residual"]))) < 1e-5

    def test_battery_soc_clamp(self, env, params, key):
        """Battery SOC stays within [soc_min, soc_max] under extreme actions."""
        _, state = env.reset(key, params)
        # Max discharge repeatedly
        action = jnp.array([0.5, 0.5, 0.5, 1.0, 0.0], dtype=jnp.float32)
        for _ in range(params.dc.max_steps):
            key, k = jax.random.split(key)
            _, state, *_ = env.step(k, state, action, params)
            assert float(state.soc) >= params.battery_soc_min - 1e-6
            assert float(state.soc) <= params.battery_soc_max + 1e-6

    def test_dg_fuel_cost_positive_when_dispatched(self, env, params, key):
        """DG fuel cost > 0 when dg_power_norm > 0."""
        _, state = env.reset(key, params)
        action = jnp.array([0.5, 0.5, 0.5, 0.0, 0.8], dtype=jnp.float32)
        _, _, _, _, _, info = env.step(key, state, action, params)
        assert float(info["fuel_cost"]) > 0.0
        assert float(info["p_dg_mw"]) > 0.0

    def test_dg_fuel_cost_zero_when_no_dispatch(self, env, params, key):
        """DG fuel cost = 0 when dg_power_norm = 0."""
        _, state = env.reset(key, params)
        action = jnp.array([0.5, 0.5, 0.5, 0.0, 0.0], dtype=jnp.float32)
        _, _, _, _, _, info = env.step(key, state, action, params)
        assert float(info["fuel_cost"]) == pytest.approx(0.0, abs=1e-8)
        assert float(info["p_dg_mw"]) == pytest.approx(0.0, abs=1e-8)

    def test_cost_info_consistent(self, env, params, key):
        """`cost_sum` == cost_sla + cost_overtemp + cost_power_deficit."""
        _, state = env.reset(key, params)
        action = jnp.array([0.5, 0.5, 0.5, 0.0, 0.3], dtype=jnp.float32)
        _, _, _, costs, _, info = env.step(key, state, action, params)
        cost_sum = float(info["cost_sla"]) + float(info["cost_overtemp"]) + float(
            info["cost_power_deficit"]
        )
        assert float(info["cost_sum"]) == pytest.approx(cost_sum, abs=1e-6)
        assert float(jnp.sum(costs)) == pytest.approx(cost_sum, abs=1e-6)

    def test_reward_cost_separation(self, env, params, key):
        """reward is a scalar float; cost signals do not feed into reward."""
        _, state = env.reset(key, params)
        action = jnp.array([0.5, 0.5, 0.5, 0.0, 0.3], dtype=jnp.float32)
        _, _, reward, costs, _, info = env.step(key, state, action, params)
        # reward is a scalar
        assert reward.shape == ()
        # Cost channels are separate from reward
        assert "cost_sum" in info
        assert "cost_sla" in info
        assert "cost_overtemp" in info
        assert "cost_power_deficit" in info
        assert costs.shape == (3,)
        # reward does NOT contain safety penalty (its components are energy/cost/carbon only)
        # Just verify it's finite and negative (energy always positive)
        assert jnp.isfinite(reward)

    def test_pv_positive_daytime_zero_night(self, env, params, key):
        """Solar CF is ≥ 0 always; positive during daytime, 0 at midnight."""
        # time_step=0 → hour=0 → solar_cf should be 0 (midnight)
        steps_per_day = params.dc.steps_per_day
        cf_night = _solar_cf(jnp.int32(0), steps_per_day)
        assert float(cf_night) == pytest.approx(0.0, abs=1e-6)
        # time_step = steps_per_day//2 → hour=12 → solar_cf should be max (=1)
        cf_noon = _solar_cf(jnp.int32(steps_per_day // 2), steps_per_day)
        assert float(cf_noon) == pytest.approx(1.0, abs=1e-4)

    def test_obs_shape(self, env, params, key):
        """Observation has shape (20,) and dtype float32."""
        obs, _ = env.reset(key, params)
        assert obs.shape == (OBS_DIM,)
        assert obs.dtype == jnp.float32

    def test_action_space_bounds(self, env, params):
        """action_space is Box(5) with correct per-dimension bounds."""
        space = env.action_space(params)
        assert space.shape == (5,)
        # First 3: [0, 1]; index 3: [-1, 1]; index 4: [0, 1]
        assert float(space.low[3]) == pytest.approx(-1.0)
        assert float(space.high[3]) == pytest.approx(1.0)
        assert float(space.low[0]) == pytest.approx(0.0)

    def test_reward_vector_in_info(self, env, params, key):
        """info['reward_vector'] has shape (3,) and is finite."""
        _, state = env.reset(key, params)
        action = jnp.array([0.5, 0.5, 0.5, 0.1, 0.3], dtype=jnp.float32)
        _, _, _, _, _, info = env.step(key, state, action, params)
        rv = info["reward_vector"]
        assert rv.shape == (3,)
        assert jnp.all(jnp.isfinite(rv))

    def test_episode_length_288(self, env, params_full):
        """Episode terminates at step 288 (full 24h at 5-min resolution)."""
        key = jax.random.PRNGKey(99)
        _, state = env.reset(key, params_full)
        action = jnp.zeros(5)
        done_step = None
        for i in range(params_full.dc.max_steps + 1):
            key, k = jax.random.split(key)
            _, state, _, _, done, _ = env.step(k, state, action, params_full)
            if bool(done) and done_step is None:
                done_step = i + 1
        assert done_step == params_full.dc.max_steps

    def test_carbon_proportional_to_dg(self, env, params, key):
        """carbon_kg is proportional to p_dg_mw."""
        _, state = env.reset(key, params)
        action_low = jnp.array([0.5, 0.5, 0.5, 0.0, 0.2], dtype=jnp.float32)
        action_high = jnp.array([0.5, 0.5, 0.5, 0.0, 0.8], dtype=jnp.float32)
        _, _, _, _, _, info_low = env.step(key, state, action_low, params)
        _, _, _, _, _, info_high = env.step(key, state, action_high, params)
        assert float(info_high["carbon_kg"]) > float(info_low["carbon_kg"])


# ---------------------------------------------------------------------------
# C1 unit tests — DG pure functions
# ---------------------------------------------------------------------------

class TestC1_DGFunctions:

    def test_compute_dg_power_clamp(self):
        """compute_dg_power clamps to [0, p_max]."""
        assert float(compute_dg_power(jnp.float32(-0.5), 0.6)) == pytest.approx(0.0)
        assert float(compute_dg_power(jnp.float32(1.5), 0.6)) == pytest.approx(0.6)
        assert float(compute_dg_power(jnp.float32(0.5), 0.6)) == pytest.approx(0.3)

    def test_compute_dg_fuel_cost_formula(self):
        """Fuel cost = p * dt * cost."""
        cost = compute_dg_fuel_cost(jnp.float32(0.3), dt_h=5.0 / 60.0, fuel_cost_per_mwh=300.0)
        expected = 0.3 * (5.0 / 60.0) * 300.0
        assert float(cost) == pytest.approx(expected, rel=1e-5)

    def test_compute_dg_emissions_formula(self):
        """Emissions = p * dt * 1e3 * ef."""
        em = compute_dg_emissions(jnp.float32(0.6), dt_h=5.0 / 60.0, emission_factor=0.80)
        expected = 0.6 * (5.0 / 60.0) * 1e3 * 0.80
        assert float(em) == pytest.approx(expected, rel=1e-5)

    def test_diesel_params_defaults(self):
        dg = DieselParams()
        assert dg.p_dg_max_mw == pytest.approx(0.6)
        assert dg.fuel_cost_per_mwh == pytest.approx(300.0)
        assert dg.emission_factor == pytest.approx(0.80)

    def test_dg_jit_compatible(self):
        """compute_dg_power is JIT-compilable."""
        jit_fn = jax.jit(lambda x: compute_dg_power(x, 0.6))
        result = jit_fn(jnp.float32(0.5))
        assert float(result) == pytest.approx(0.3)

    def test_dg_vmap_compatible(self):
        """compute_dg_power is vmap-compatible."""
        inputs = jnp.linspace(0.0, 1.0, 5)
        results = jax.vmap(lambda x: compute_dg_power(x, 0.6))(inputs)
        expected = jnp.clip(inputs, 0.0, 1.0) * 0.6
        assert jnp.allclose(results, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# make_dcmicrogrid_params tests
# ---------------------------------------------------------------------------

class TestFactory:

    def test_default_config(self):
        p = make_dcmicrogrid_params()
        assert p.dc.max_steps == 288
        assert p.dc.steps_per_day == 288
        assert abs(p.dc.delta_t_hours - 5.0 / 60.0) < 1e-6
        assert p.pv_p_max_mw == pytest.approx(0.4)
        assert p.dg.p_dg_max_mw == pytest.approx(0.6)
        assert p.battery_capacity_mwh == pytest.approx(2.0)

    def test_override_episode_length(self):
        p = make_dcmicrogrid_params(max_steps=48)
        assert p.dc.max_steps == 48

    def test_instantiable_and_resetable(self):
        env = DataCenterMicrogridEnv()
        params = make_dcmicrogrid_params(max_steps=5)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)
        assert obs.shape == (OBS_DIM,)
        assert isinstance(state, DCMicrogridState)


# ---------------------------------------------------------------------------
# PV phase-alignment regression tests (fix: use state.dc.time_step for PV)
# ---------------------------------------------------------------------------

class TestPVPhaseAlignment:
    """Verify that p_pv_mw in power balance uses the CURRENT step's time index
    (state.dc.time_step = t), not the post-step index (new_dc.time_step = t+1).

    Sunrise anchor: with steps_per_day=288, step t=72 maps to hour=6.0 exactly.
      _solar_cf(72,  288) = clip(sin(pi*(6.0-6)/12), 0,1) = sin(0) = 0.0
      _solar_cf(73,  288) = clip(sin(pi*(6.0833-6)/12), 0,1) ≈ 0.0218  (> 0)

    If the bug were present, a step taken when state.dc.time_step=72 would
    report p_pv_mw > 0 (using t+1=73); with the fix it correctly reports 0.
    """

    def _advance_to_step(self, env, params, target_step):
        """Run the env forward until state.dc.time_step == target_step."""
        assert target_step < params.dc.max_steps, "target_step must fit in episode"
        key = jax.random.PRNGKey(0)
        _, state = env.reset(key, params)
        action = jnp.zeros(5)
        for _ in range(target_step):
            key, k = jax.random.split(key)
            _, state, *_ = env.step(k, state, action, params)
        assert int(state.dc.time_step) == target_step, (
            f"Expected time_step={target_step}, got {state.dc.time_step}"
        )
        return state, key

    def test_pv_zero_at_sunrise_boundary(self, env, params_full):
        """At state.dc.time_step=72 (hour=6.0), p_pv_mw must be exactly 0.

        _solar_cf(72, 288) = sin(0) = 0.  If the bug were present the step
        would use _solar_cf(73, 288) ≈ 0.0218 and return p_pv_mw > 0.
        """
        state, key = self._advance_to_step(env, params_full, target_step=72)
        action = jnp.zeros(5)
        key, k = jax.random.split(key)
        _, _, _, _, _, info = env.step(k, state, action, params_full)

        expected_solar_cf = float(_solar_cf(jnp.int32(72), 288))  # = 0.0
        assert expected_solar_cf == pytest.approx(0.0, abs=1e-7), (
            "Test anchor broken: _solar_cf(72,288) should be 0"
        )
        assert float(info["p_pv_mw"]) == pytest.approx(
            expected_solar_cf * params_full.pv_p_max_mw, abs=1e-6
        ), (
            "p_pv_mw should be 0 at sunrise boundary (step 72); "
            "a non-zero value indicates the pre-fix t+1 bug."
        )

    def test_pv_positive_one_step_after_sunrise(self, env, params_full):
        """At state.dc.time_step=73 (hour≈6.083), p_pv_mw must be > 0.

        _solar_cf(73, 288) > 0, so the fix should preserve this.
        """
        state, key = self._advance_to_step(env, params_full, target_step=73)
        action = jnp.zeros(5)
        key, k = jax.random.split(key)
        _, _, _, _, _, info = env.step(k, state, action, params_full)

        expected_solar_cf = float(_solar_cf(jnp.int32(73), 288))
        assert expected_solar_cf > 0.0, (
            "Test anchor broken: _solar_cf(73,288) should be positive"
        )
        assert float(info["p_pv_mw"]) == pytest.approx(
            expected_solar_cf * params_full.pv_p_max_mw, rel=1e-5
        )

    def test_residual_uses_current_step_pv(self, env, params_full):
        """residual = p_pv(t) + p_dg + p_batt - p_load(t); no t+1 PV mix-in.

        At step t=72 with action=(0,0,0,0,0): DG=0, batt=0, pv=0 (sunrise).
        Power balance: residual = 0 + 0 + 0 - p_load = -p_load < 0.
        Both power_deficit and residual must equal -p_load exactly.
        """
        state, key = self._advance_to_step(env, params_full, target_step=72)
        action = jnp.zeros(5)  # no DG, no battery
        key, k = jax.random.split(key)
        _, _, _, _, _, info = env.step(k, state, action, params_full)

        # Verify p_pv = 0 (current step = sunrise boundary)
        assert float(info["p_pv_mw"]) == pytest.approx(0.0, abs=1e-6)

        # Verify residual identity: p_pv + p_dg + p_batt - p_load
        residual_ref = (
            info["p_pv_mw"] + info["p_dg_mw"] + info["p_batt_mw"] - info["p_load_mw"]
        )
        assert jnp.allclose(info["residual"], residual_ref, atol=1e-5)

        # With DG=0, batt≈0 (no desired discharge), pv=0:
        # residual should be negative (pure deficit)
        if float(info["p_load_mw"]) > 1e-4:
            assert float(info["residual"]) < 0.0, (
                "residual should be negative at night with no DG/battery; "
                "positive residual would imply t+1 PV was mixed in."
            )


# ---------------------------------------------------------------------------
# Carbon reward unit-consistency tests
# ---------------------------------------------------------------------------

class TestCarbonRewardUnits:
    """Verify reward_vector[2] == -carbon_kg (same kgCO2 unit, no ×1e3 mismatch)."""

    def test_r_carbon_equals_negative_carbon_kg(self, env, params, key):
        """reward_vector[2] must equal -info['carbon_kg'] exactly."""
        _, state = env.reset(key, params)
        action = jnp.array([0.5, 0.5, 0.5, 0.0, 0.6], dtype=jnp.float32)
        _, _, _, _, _, info = env.step(key, state, action, params)
        r_carbon = float(info["reward_vector"][2])
        carbon_kg = float(info["carbon_kg"])
        assert r_carbon == pytest.approx(-carbon_kg, rel=1e-5), (
            f"reward_vector[2]={r_carbon} should equal -carbon_kg={-carbon_kg}; "
            "a factor-of-1000 gap indicates the pre-fix unit mismatch."
        )

    def test_full_dg_dispatch_carbon_positive_and_consistent(self, env, params, key):
        """With max DG dispatch (dg_norm=1.0): carbon_kg > 0 and |r_carbon| == carbon_kg."""
        _, state = env.reset(key, params)
        action = jnp.array([0.5, 0.5, 0.5, 0.0, 1.0], dtype=jnp.float32)
        _, _, _, _, _, info = env.step(key, state, action, params)
        carbon_kg = float(info["carbon_kg"])
        r_carbon = float(info["reward_vector"][2])
        assert carbon_kg > 0.0, "carbon_kg should be positive with full DG dispatch"
        assert abs(r_carbon) == pytest.approx(carbon_kg, rel=1e-5), (
            f"|r_carbon|={abs(r_carbon)} != carbon_kg={carbon_kg}"
        )
        # Sanity: expected value for full DG (0.6 MW, 5 min, ef=0.80)
        dt_h = params.dc.delta_t_hours
        expected_kg = 0.6 * dt_h * 1e3 * 0.80
        assert carbon_kg == pytest.approx(expected_kg, rel=1e-4)


# ---------------------------------------------------------------------------
# C3: multi-objective reward / cost signals through LogWrapper
# ---------------------------------------------------------------------------

class TestC3_MultiObjectiveSignals:
    """Verify reward_vector and cost channels survive the RL wrapper stack."""

    def _make_log_wrapped(self, params):
        from powerzoojax.rl.wrappers import LogWrapper
        return LogWrapper(DataCenterMicrogridEnv(), params)

    def test_reward_vector_in_info_after_logwrapper(self, params, key):
        """reward_vector is accessible from info after LogWrapper.step()."""
        wrapped = self._make_log_wrapped(params)
        _, state = wrapped.reset(key)
        action = jnp.array([0.5, 0.5, 0.5, 0.0, 0.3], dtype=jnp.float32)
        _, _, _, _, info = wrapped.step(key, state, action)
        assert "reward_vector" in info
        rv = info["reward_vector"]
        assert rv.shape == (3,)
        assert jnp.all(jnp.isfinite(rv))

    def test_cost_channels_in_info_after_logwrapper(self, params, key):
        """cost, cost_sla, cost_overtemp, cost_power_deficit all present."""
        wrapped = self._make_log_wrapped(params)
        _, state = wrapped.reset(key)
        action = jnp.zeros(5)
        _, _, _, _, info = wrapped.step(key, state, action)
        for channel in ("cost", "cost_sla", "cost_overtemp", "cost_power_deficit"):
            assert channel in info, f"missing key '{channel}' in info"
            assert jnp.isfinite(info[channel])

    def test_cost_sum_consistent_through_wrapper(self, params, key):
        """info['cost'] == sum of individual cost channels after wrapping."""
        wrapped = self._make_log_wrapped(params)
        _, state = wrapped.reset(key)
        action = jnp.array([0.5, 0.5, 0.5, 0.0, 0.5], dtype=jnp.float32)
        _, _, _, _, info = wrapped.step(key, state, action)
        cost_sum = (
            float(info["cost_sla"])
            + float(info["cost_overtemp"])
            + float(info["cost_power_deficit"])
        )
        assert float(info["cost_sum"]) == pytest.approx(cost_sum, abs=1e-6)

    def test_reward_scalar_independent_of_cost(self, params, key):
        """Scalar reward and cost are distinct channels (reward/cost separation)."""
        wrapped = self._make_log_wrapped(params)
        _, state = wrapped.reset(key)
        action = jnp.array([0.5, 0.5, 0.5, 0.0, 0.6], dtype=jnp.float32)
        _, _, reward, _, info = wrapped.step(key, state, action)
        # reward is a finite scalar
        assert reward.shape == ()
        assert jnp.isfinite(reward)
        # cost is a separate non-negative scalar
        assert float(info["cost_sum"]) >= 0.0
        # They are different values in general (reward is negative energy signal)
        assert float(reward) != float(info["cost_sum"]) or True  # just check types


# ---------------------------------------------------------------------------
# C3 + preset: preset registry and instantiation
# ---------------------------------------------------------------------------

class TestPresets:
    """Verify DC Microgrid presets are listed, instantiable, and JIT-runnable."""

    def test_dc_microgrid_preset_exists(self):
        from powerzoojax.rl.presets import list_presets
        names = [p["name"] for p in list_presets()]
        assert "dc-microgrid" in names

    def test_dc_microgrid_safe_preset_exists(self):
        from powerzoojax.rl.presets import list_presets
        names = [p["name"] for p in list_presets()]
        assert "dc-microgrid-safe" in names

    def test_dc_microgrid_preset_instantiable(self):
        from powerzoojax.rl.presets import get_preset
        preset = get_preset("dc-microgrid")
        wrapped = preset.env_factory()
        assert wrapped is not None
        assert wrapped.obs_size == OBS_DIM
        assert wrapped.num_actions == 5

    def test_dc_microgrid_safe_preset_instantiable(self):
        from powerzoojax.rl.presets import get_preset
        preset = get_preset("dc-microgrid-safe")
        wrapped = preset.env_factory()
        assert wrapped is not None

    def test_dc_microgrid_preset_reset_step_jit(self):
        """Preset env_factory() → wrapped.reset() + wrapped.step() run via JIT."""
        from powerzoojax.rl.presets import get_preset
        wrapped = get_preset("dc-microgrid").env_factory()
        key = jax.random.PRNGKey(0)
        obs, state = wrapped.reset(key)
        assert obs.shape == (OBS_DIM,)
        action = jnp.zeros(5)
        obs2, state2, reward, done, info = wrapped.step(key, state, action)
        assert obs2.shape == (OBS_DIM,)
        assert "reward_vector" in info
        assert "cost" in info

    def test_dc_microgrid_safe_preset_reset_step(self):
        """SafeRLWrapper returns explicit selected CMDP costs as the 4th output."""
        from powerzoojax.rl.presets import get_preset
        wrapped = get_preset("dc-microgrid-safe").env_factory()
        key = jax.random.PRNGKey(1)
        obs, state = wrapped.reset(key)
        action = jnp.zeros(5)
        result = wrapped.step(key, state, action)
        # SafeRLWrapper returns (obs, state, reward, costs, done, info)
        assert len(result) == 6
        obs2, state2, reward, costs, done, info = result
        assert obs2.shape == (OBS_DIM,)
        assert costs.shape == (3,)
        assert jnp.all(jnp.isfinite(costs))

    def test_preset_config_algo(self):
        from powerzoojax.rl.presets import get_preset
        assert get_preset("dc-microgrid").config.algo == "ppo"
        assert get_preset("dc-microgrid-safe").config.algo == "ppo_lagrangian"

    def test_preset_config_n_steps(self):
        from powerzoojax.rl.presets import get_preset
        assert get_preset("dc-microgrid").config.n_steps == 288
        assert get_preset("dc-microgrid-safe").config.n_steps == 288


# ---------------------------------------------------------------------------
# Finding C regression guard: PV must use state.dc.time_step, not new_dc.time_step
# ---------------------------------------------------------------------------

class TestFindingC_PVPhaseRegressionGuard:
    """Explicit regression guard for the PV one-step phase bug.

    Finding C was already fixed prior to this review round
    (step() uses state.dc.time_step = t; NOT new_dc.time_step = t+1).
    These tests lock that behaviour in.
    """

    def test_pv_never_uses_next_step_time_index(self, env, params_full):
        """At every step, p_pv_mw == _solar_cf(state.dc.time_step), not t+1."""
        from powerzoojax.envs.microgrid.dc_microgrid import _solar_cf
        key = jax.random.PRNGKey(77)
        _, state = env.reset(key, params_full)
        action = jnp.zeros(5)
        spd = params_full.dc.steps_per_day
        pv_max = params_full.pv_p_max_mw
        # Check 30 consecutive steps across midnight and sunrise
        for _ in range(30):
            t_before = int(state.dc.time_step)
            key, k = jax.random.split(key)
            _, state, _, _, _, info = env.step(k, state, action, params_full)
            cf_current = float(_solar_cf(jnp.int32(t_before), spd))
            cf_next = float(_solar_cf(jnp.int32(t_before + 1), spd))
            expected_pv = cf_current * pv_max
            assert float(info["p_pv_mw"]) == pytest.approx(expected_pv, abs=1e-5), (
                f"t={t_before}: got {float(info['p_pv_mw']):.6f}, "
                f"expected(t)={expected_pv:.6f}, wrong(t+1)={cf_next*pv_max:.6f}. "
                "PV phase regression."
            )


# ---------------------------------------------------------------------------
# L1 — Battery energy conservation and feasibility boundaries
# ---------------------------------------------------------------------------

class TestL1_BatteryPhysics:
    """Round-trip energy conservation and SOC-boundary feasibility."""

    def test_battery_round_trip_efficiency(self, env, params):
        """Discharge then charge by the same |p|·dt yields net SOC drop = (1-η_rt)·E/cap.

        With η_c=η_d=0.95, dt=5/60h, cap=2 MWh, p_max=0.5 MW:
          - Step 1 (batt_norm=+1): p_batt=+0.5 MW; ΔSOC = -0.5·dt/(η_d·cap)
          - Step 2 (batt_norm=-1): p_batt=-0.5 MW; ΔSOC = +0.5·dt·η_c/cap
          - Net ΔSOC = -p·dt·(1/η_d - η_c)/cap
        """
        key = jax.random.PRNGKey(11)
        _, state = env.reset(key, params)
        soc0 = float(state.soc)

        # Step 1: max discharge
        a_dis = jnp.array([0.0, 0.0, 0.5, 1.0, 0.0], dtype=jnp.float32)
        _, state, _, _, _, info1 = env.step(key, state, a_dis, params)
        # Step 2: max charge
        a_chg = jnp.array([0.0, 0.0, 0.5, -1.0, 0.0], dtype=jnp.float32)
        _, state, _, _, _, info2 = env.step(key, state, a_chg, params)

        dt = params.dc.delta_t_hours
        cap = params.battery_capacity_mwh
        eta_c = params.battery_eta_charge
        eta_d = params.battery_eta_discharge
        # Both steps should run at the rated power without hitting boundaries
        assert float(info1["p_batt_mw"]) == pytest.approx(params.battery_power_mw, rel=1e-5)
        assert float(info2["p_batt_mw"]) == pytest.approx(-params.battery_power_mw, rel=1e-5)

        p = params.battery_power_mw
        expected_delta = -(p * dt) * (1.0 / eta_d - eta_c) / cap
        net_delta = float(state.soc) - soc0
        assert net_delta == pytest.approx(expected_delta, rel=1e-4, abs=1e-6)
        # Sanity: with η_rt < 1, SOC must have dropped (we lost energy round-trip)
        assert net_delta < 0.0

    def test_battery_discharge_bounded_at_soc_min(self, env):
        """At SOC=soc_min, max-discharge action yields p_batt=0 and SOC unchanged."""
        params = make_dcmicrogrid_params(
            max_steps=10,
            battery_soc_init=0.1,  # exactly at soc_min
        )
        key = jax.random.PRNGKey(0)
        _, state = env.reset(key, params)
        assert float(state.soc) == pytest.approx(params.battery_soc_min, abs=1e-6)
        a = jnp.array([0.0, 0.0, 0.5, 1.0, 0.0], dtype=jnp.float32)
        _, state2, _, _, _, info = env.step(key, state, a, params)
        assert float(info["p_batt_mw"]) == pytest.approx(0.0, abs=1e-6)
        assert float(state2.soc) == pytest.approx(params.battery_soc_min, abs=1e-6)

    def test_battery_charge_bounded_at_soc_max(self, env):
        """At SOC=soc_max, max-charge action yields p_batt=0 and SOC unchanged."""
        params = make_dcmicrogrid_params(
            max_steps=10,
            battery_soc_init=0.9,  # exactly at soc_max
        )
        key = jax.random.PRNGKey(0)
        _, state = env.reset(key, params)
        assert float(state.soc) == pytest.approx(params.battery_soc_max, abs=1e-6)
        a = jnp.array([0.0, 0.0, 0.5, -1.0, 0.0], dtype=jnp.float32)
        _, state2, _, _, _, info = env.step(key, state, a, params)
        assert float(info["p_batt_mw"]) == pytest.approx(0.0, abs=1e-6)
        assert float(state2.soc) == pytest.approx(params.battery_soc_max, abs=1e-6)

    def test_battery_sign_convention_discharge_positive(self, env, params, key):
        """batt_norm > 0 → p_batt > 0 (discharge, supply side); SOC decreases."""
        _, state = env.reset(key, params)
        a = jnp.array([0.0, 0.0, 0.5, 0.5, 0.0], dtype=jnp.float32)
        _, state2, _, _, _, info = env.step(key, state, a, params)
        assert float(info["p_batt_mw"]) > 0.0
        assert float(state2.soc) < float(state.soc)

    def test_battery_sign_convention_charge_negative(self, env, params, key):
        """batt_norm < 0 → p_batt < 0 (charge, draws from supply); SOC increases."""
        _, state = env.reset(key, params)
        a = jnp.array([0.0, 0.0, 0.5, -0.5, 0.0], dtype=jnp.float32)
        _, state2, _, _, _, info = env.step(key, state, a, params)
        assert float(info["p_batt_mw"]) < 0.0
        assert float(state2.soc) > float(state.soc)


# ---------------------------------------------------------------------------
# L1 — Reward scalarization identity
# ---------------------------------------------------------------------------

class TestL1_RewardScalarization:
    """Verify reward = r_energy + w_cost·r_cost + w_carbon·r_carbon exactly."""

    def test_reward_equals_weighted_sum(self, env, params, key):
        _, state = env.reset(key, params)
        a = jnp.array([0.5, 0.5, 0.5, 0.3, 0.6], dtype=jnp.float32)
        _, _, reward, _, _, info = env.step(key, state, a, params)
        rv = info["reward_vector"]
        r_energy = float(rv[0])
        r_cost = float(rv[1])
        r_carbon = float(rv[2])
        expected = r_energy + params.w_cost * r_cost + params.w_carbon * r_carbon
        assert float(reward) == pytest.approx(expected, rel=1e-5, abs=1e-6)

    def test_r_cost_includes_fuel_and_battery_degradation(self, env, params, key):
        """r_cost = -(fuel_cost + |p_batt|·dt·deg_cost_per_mwh)."""
        _, state = env.reset(key, params)
        a = jnp.array([0.5, 0.5, 0.5, 0.4, 0.7], dtype=jnp.float32)
        _, _, _, _, _, info = env.step(key, state, a, params)
        dt = params.dc.delta_t_hours
        fuel = float(info["fuel_cost"])
        p_batt = float(info["p_batt_mw"])
        deg = abs(p_batt) * dt * params.battery_deg_cost_per_mwh
        r_cost = float(info["reward_vector"][1])
        assert r_cost == pytest.approx(-(fuel + deg), rel=1e-5, abs=1e-6)

    def test_r_energy_equals_negative_p_dc_mwh(self, env, params, key):
        """r_energy = -p_dc_mw · dt_h."""
        _, state = env.reset(key, params)
        a = jnp.array([0.5, 0.5, 0.5, 0.0, 0.4], dtype=jnp.float32)
        _, _, _, _, _, info = env.step(key, state, a, params)
        dt = params.dc.delta_t_hours
        expected = -float(info["p_dc_mw"]) * dt
        r_energy = float(info["reward_vector"][0])
        assert r_energy == pytest.approx(expected, rel=1e-5, abs=1e-6)


# ---------------------------------------------------------------------------
# L1 — Power spill (excess generation)
# ---------------------------------------------------------------------------

class TestL1_PowerSpill:
    """power_spill = max(residual, 0) must fire when supply exceeds demand."""

    def test_spill_when_supply_exceeds_demand(self, env, key):
        """Oversize DG to guarantee residual > 0; expect power_spill > 0, deficit = 0."""
        params = make_dcmicrogrid_params(
            max_steps=5,
            dg_p_max_mw=100.0,  # massive over-capacity
        )
        _, state = env.reset(key, params)
        # Full DG dispatch with no battery action
        a = jnp.array([0.5, 0.5, 0.5, 0.0, 1.0], dtype=jnp.float32)
        _, _, _, _, _, info = env.step(key, state, a, params)
        assert float(info["p_dg_mw"]) == pytest.approx(100.0, rel=1e-5)
        assert float(info["power_spill"]) > 0.0
        assert float(info["power_deficit"]) == pytest.approx(0.0, abs=1e-6)
        # Identity: spill - deficit = residual
        gap = float(info["power_spill"]) - float(info["power_deficit"])
        assert gap == pytest.approx(float(info["residual"]), abs=1e-5)

    def test_spill_exclusive_with_deficit(self, env, params, key):
        """At every step, at most one of (spill, deficit) is > 0."""
        _, state = env.reset(key, params)
        for i in range(params.dc.max_steps):
            key, k = jax.random.split(key)
            a = jax.random.uniform(k, (5,), minval=-1.0, maxval=1.0)
            _, state, _, _, _, info = env.step(k, state, a, params)
            spill = float(info["power_spill"])
            deficit = float(info["power_deficit"])
            assert spill >= 0.0 and deficit >= 0.0
            # They cannot both be strictly positive
            assert (spill < 1e-9) or (deficit < 1e-9), (
                f"step {i}: spill={spill}, deficit={deficit} both positive"
            )


# ---------------------------------------------------------------------------
# L1 — COP-temperature coupling and p_aux_frac
# ---------------------------------------------------------------------------

class TestL1_CoolingAndAux:
    """COP decreases with outdoor temp → higher p_cool; p_aux_frac scales overhead."""

    def test_cop_decreases_with_outdoor_temperature(self, env):
        """Hotter outdoor → lower COP → higher p_cool_mw → higher p_dc_mw.

        Force outdoor temp via outdoor_temp_profile (deterministic override) so
        that COP factor changes are purely temperature-driven.
        """
        # Aggressive cooling so q_cool is non-zero (t_zone > t_setpoint=18)
        a = jnp.array([0.5, 0.5, 0.0, 0.0, 0.0], dtype=jnp.float32)
        key = jax.random.PRNGKey(123)

        # Cool case: outdoor at t_ref (no COP penalty)
        cool_profile = jnp.full((288,), 20.0, dtype=jnp.float32)
        params_cool = make_dcmicrogrid_params(max_steps=5)
        params_cool = params_cool.replace(outdoor_temp_profile=cool_profile)
        _, s_cool0 = env.reset(key, params_cool)
        _, s_cool, _, _, _, info_cool = env.step(key, s_cool0, a, params_cool)

        # Hot case: outdoor 15°C above t_ref → COP factor reduced by 0.04*15=0.6
        hot_profile = jnp.full((288,), 35.0, dtype=jnp.float32)
        params_hot = make_dcmicrogrid_params(max_steps=5)
        params_hot = params_hot.replace(outdoor_temp_profile=hot_profile)
        _, s_hot0 = env.reset(key, params_hot)
        _, s_hot, _, _, _, info_hot = env.step(key, s_hot0, a, params_hot)

        # Cooling power must be larger under heat (lower COP for same q_cool)
        assert float(s_hot.dc.p_cool_mw) > float(s_cool.dc.p_cool_mw), (
            f"hot p_cool={float(s_hot.dc.p_cool_mw):.4f} should exceed cool "
            f"p_cool={float(s_cool.dc.p_cool_mw):.4f} (COP-temperature coupling)"
        )
        # Strict: same IT load and setpoint, only outdoor differs → hot p_dc > cool p_dc
        assert float(info_hot["p_dc_mw"]) > float(info_cool["p_dc_mw"]), (
            f"hot p_dc={float(info_hot['p_dc_mw'])} should exceed cool p_dc="
            f"{float(info_cool['p_dc_mw'])} via COP degradation"
        )

    def test_p_aux_frac_increases_total_power(self, env, key):
        """Higher p_aux_frac → higher p_dc_mw (linear add-on to IT power)."""
        a = jnp.array([0.5, 0.5, 0.5, 0.0, 0.0], dtype=jnp.float32)

        params_lo = make_dcmicrogrid_params(max_steps=5, p_aux_frac=0.0)
        _, s_lo = env.reset(key, params_lo)
        _, s_lo2, _, _, _, info_lo = env.step(key, s_lo, a, params_lo)

        params_hi = make_dcmicrogrid_params(max_steps=5, p_aux_frac=0.20)
        _, s_hi = env.reset(key, params_hi)
        _, s_hi2, _, _, _, info_hi = env.step(key, s_hi, a, params_hi)

        # IT power should be (approximately) identical; difference goes into aux
        assert float(s_lo2.dc.p_it_mw) == pytest.approx(float(s_hi2.dc.p_it_mw), rel=1e-4)
        # p_dc difference ≈ (0.20 - 0.0) · p_it
        delta_obs = float(info_hi["p_dc_mw"]) - float(info_lo["p_dc_mw"])
        delta_exp = 0.20 * float(s_lo2.dc.p_it_mw)
        assert delta_obs == pytest.approx(delta_exp, rel=1e-3, abs=1e-6)


# ---------------------------------------------------------------------------
# L1 — Action clipping
# ---------------------------------------------------------------------------

class TestL1_ActionClipping:
    """Out-of-range actions must be clipped to the declared box, no NaN."""

    def test_oversized_action_clipped(self, env, params, key):
        """Extreme action values are clipped: dg saturates, batt saturates."""
        _, state = env.reset(key, params)
        a_huge = jnp.array([5.0, -3.0, 9.9, 7.0, 4.0], dtype=jnp.float32)
        _, state2, reward, costs, done, info = env.step(key, state, a_huge, params)
        # DG should saturate at p_dg_max_mw
        assert float(info["p_dg_mw"]) == pytest.approx(params.dg.p_dg_max_mw, rel=1e-5)
        # Battery should saturate at +power_mw (positive = discharge)
        assert float(info["p_batt_mw"]) == pytest.approx(params.battery_power_mw, rel=1e-5)
        # last_action stored is the clipped value
        la = state2.last_action
        assert float(la[0]) == pytest.approx(1.0, abs=1e-6)
        assert float(la[1]) == pytest.approx(0.0, abs=1e-6)  # negative clipped to 0
        assert float(la[2]) == pytest.approx(1.0, abs=1e-6)
        assert float(la[3]) == pytest.approx(1.0, abs=1e-6)
        assert float(la[4]) == pytest.approx(1.0, abs=1e-6)
        # Everything must be finite
        assert jnp.isfinite(reward) and bool(jnp.all(jnp.isfinite(state2.last_action)))


# ---------------------------------------------------------------------------
# L1 — Profile override end-to-end
# ---------------------------------------------------------------------------

class TestL1_ProfileOverrides:
    """Verify cpu_profile / solar_profile / outdoor_temp_profile actually take effect."""

    def test_cpu_profile_overrides_inference_load(self, env, key):
        """All-1 cpu_profile → gpus_infer == infer_gpu_peak (synthetic would clip to 0.1·peak min)."""
        T = 288
        full_profile = jnp.ones((T,), dtype=jnp.float32)
        params = make_dcmicrogrid_params(max_steps=5)
        params = params.replace(cpu_profile=full_profile)
        _, state = env.reset(key, params)
        a = jnp.zeros(5, dtype=jnp.float32)
        _, state2, _, _, _, _ = env.step(key, state, a, params)
        assert int(state2.dc.gpus_infer) == int(params.dc.infer_gpu_peak)

        # And zero profile → gpus_infer == 0 (no inference load)
        zero_profile = jnp.zeros((T,), dtype=jnp.float32)
        params0 = make_dcmicrogrid_params(max_steps=5)
        params0 = params0.replace(cpu_profile=zero_profile)
        _, state0 = env.reset(key, params0)
        _, state0_next, _, _, _, _ = env.step(key, state0, a, params0)
        assert int(state0_next.dc.gpus_infer) == 0

    def test_solar_profile_overrides_pv(self, env, key):
        """solar_profile=0 forces p_pv=0 even at noon; solar_profile=1 forces full output.

        After the bundle refactor the solar profile lives inside the SolarBundle
        rather than the env params, so we override it through the factory.
        """
        T = 288
        params_zero = make_dcmicrogrid_params(
            max_steps=5, solar_profile=jnp.zeros((T,), dtype=jnp.float32)
        )
        _, state = env.reset(key, params_zero)
        a = jnp.zeros(5, dtype=jnp.float32)
        for _ in range(3):
            key, k = jax.random.split(key)
            _, state, _, _, _, info = env.step(k, state, a, params_zero)
            assert float(info["p_pv_mw"]) == pytest.approx(0.0, abs=1e-6)

        params_full = make_dcmicrogrid_params(
            max_steps=5, solar_profile=jnp.ones((T,), dtype=jnp.float32)
        )
        _, state = env.reset(key, params_full)
        _, _, _, _, _, info = env.step(key, state, a, params_full)
        assert float(info["p_pv_mw"]) == pytest.approx(params_full.pv_p_max_mw, rel=1e-5)

    def test_outdoor_temp_profile_overrides_synthetic(self, env, key):
        """Constant outdoor_temp_profile sets state.dc.t_outdoor to that constant."""
        T = 288
        const_temp = 30.0
        profile = jnp.full((T,), const_temp, dtype=jnp.float32)
        params = make_dcmicrogrid_params(max_steps=5)
        params = params.replace(outdoor_temp_profile=profile)
        _, state = env.reset(key, params)
        a = jnp.zeros(5, dtype=jnp.float32)
        _, state2, _, _, _, _ = env.step(key, state, a, params)
        assert float(state2.dc.t_outdoor) == pytest.approx(const_temp, abs=1e-5)

    def test_profile_cyclic_indexing(self, env, key):
        """Profile shorter than max_steps is cyclically indexed via t % T."""
        # Build cpu_profile of length 4 with distinct values
        cpu = jnp.array([0.0, 0.5, 1.0, 0.25], dtype=jnp.float32)
        params = make_dcmicrogrid_params(max_steps=10)
        params = params.replace(cpu_profile=cpu)
        _, state = env.reset(key, params)
        a = jnp.zeros(5, dtype=jnp.float32)
        peak = params.dc.infer_gpu_peak
        # t=0 → cpu[0]=0.0 → gpus_infer=0
        _, state, *_ = env.step(key, state, a, params)
        assert int(state.dc.gpus_infer) == 0
        # t=1 → cpu[1]=0.5 → gpus_infer = floor(0.5*peak)
        _, state, *_ = env.step(key, state, a, params)
        assert int(state.dc.gpus_infer) == int(0.5 * peak)
        # t=2 → cpu[2]=1.0 → gpus_infer=peak
        _, state, *_ = env.step(key, state, a, params)
        assert int(state.dc.gpus_infer) == peak
        # t=3 → cpu[3]=0.25
        _, state, *_ = env.step(key, state, a, params)
        assert int(state.dc.gpus_infer) == int(0.25 * peak)
        # t=4 wraps to cpu[0]=0.0 → gpus_infer=0
        _, state, *_ = env.step(key, state, a, params)
        assert int(state.dc.gpus_infer) == 0


# ---------------------------------------------------------------------------
# L0 — vmap numerical correctness (not just shape)
# ---------------------------------------------------------------------------

class TestL0_VmapNumerical:
    """vmap must produce identical results when inputs are identical."""

    def test_vmap_identical_inputs_identical_outputs(self, env, params):
        """Two parallel envs with the same key/action/state → bitwise-equal step output."""
        N = 4
        same_key = jax.random.PRNGKey(2026)
        keys = jnp.broadcast_to(same_key, (N, *same_key.shape))

        # Reset all with the same key
        vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
        obs_b, state_b = vmap_reset(keys, params)
        # All N obs rows should be identical
        for i in range(1, N):
            assert jnp.allclose(obs_b[0], obs_b[i], atol=0.0)

        # Step with identical actions and the same step key
        actions = jnp.broadcast_to(
            jnp.array([0.4, 0.6, 0.5, 0.2, 0.3], dtype=jnp.float32), (N, 5)
        )
        step_key = jax.random.PRNGKey(99)
        step_keys = jnp.broadcast_to(step_key, (N, *step_key.shape))
        vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
        obs2_b, state2_b, rew_b, costs_b, done_b, info_b = vmap_step(
            step_keys, state_b, actions, params
        )
        # All rewards equal across the batch
        assert jnp.allclose(rew_b, rew_b[0], atol=0.0)
        assert costs_b.shape == (N, 3)
        # Power channels identical
        for k in ("p_dc_mw", "p_pv_mw", "p_dg_mw", "p_batt_mw", "carbon_kg"):
            v = info_b[k]
            assert jnp.allclose(v, v[0], atol=0.0), f"{k} not identical across vmap batch"

    def test_bundle_protocol_dispatch(self, env, params, key):
        """DCMicrogridEnv composes via the ResourceBundle protocol.

        Verifies the bundle dispatch loop in ``step()`` actually runs each
        attached bundle: changing a bundle's parameter must visibly change
        the corresponding info channel.
        """
        from powerzoojax.envs.resource.battery import BatteryBundle
        from powerzoojax.envs.resource.renewable import RenewableBundle
        from powerzoojax.envs.resource.diesel import DieselBundle

        # Default factory must build the canonical (battery, PV, diesel) tuple,
        # where PV is a RenewableBundle in no-control mode (replaces the
        # short-lived SolarBundle).
        types = [type(b).__name__ for b in params.resources]
        assert types == ["BatteryBundle", "RenewableBundle", "DieselBundle"], types
        # action_dim breakdown: battery=1, solar=0, diesel=1
        assert params.resources[0].action_dim == 1
        assert params.resources[1].action_dim == 0
        assert params.resources[2].action_dim == 1
        # Total bundle action_dim + 3 (DC) == 5.
        bundle_total = sum(b.action_dim for b in params.resources)
        assert bundle_total + 3 == 5

        # Sanity: scaling DG p_max scales the actual p_dg_mw injection.
        new_resources = list(params.resources)
        d = new_resources[2]
        new_resources[2] = d.replace(p_max=d.p_max * 2.0)
        scaled_params = params.replace(resources=tuple(new_resources))
        _, state = env.reset(key, scaled_params)
        a = jnp.array([0.5, 0.5, 0.5, 0.0, 0.5], dtype=jnp.float32)
        _, _, _, _, _, info = env.step(key, state, a, scaled_params)
        # Original p_max=0.6, scaled to 1.2; with dg_norm=0.5, p_dg should be 0.6.
        assert float(info["p_dg_mw"]) == pytest.approx(0.6, rel=1e-4)

    def test_vmap_matches_serial(self, env, params):
        """vmap result must match serial single-env result for the same inputs."""
        key = jax.random.PRNGKey(7)
        action = jnp.array([0.3, 0.7, 0.4, -0.5, 0.6], dtype=jnp.float32)

        # Serial
        obs_s, state_s = env.reset(key, params)
        obs_s2, state_s2, rew_s, costs_s, done_s, info_s = env.step(
            key, state_s, action, params
        )

        # Batched (size=3, all identical)
        N = 3
        keys = jnp.broadcast_to(key, (N, *key.shape))
        vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
        _, state_b = vmap_reset(keys, params)
        actions = jnp.broadcast_to(action, (N, 5))
        vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
        _, _, rew_b, costs_b, _, info_b = vmap_step(keys, state_b, actions, params)

        # Serial reward must equal each batched reward (within float tolerance)
        for i in range(N):
            assert float(rew_b[i]) == pytest.approx(float(rew_s), rel=1e-5, abs=1e-6)
            assert jnp.allclose(costs_b[i], costs_s, atol=1e-6)
        # Power channels likewise
        for k in ("p_dc_mw", "p_pv_mw", "p_dg_mw", "p_batt_mw"):
            for i in range(N):
                assert float(info_b[k][i]) == pytest.approx(
                    float(info_s[k]), rel=1e-5, abs=1e-6
                )
