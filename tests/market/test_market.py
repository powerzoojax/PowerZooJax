"""Tests for powerzoojax.envs.market — L0 JAX + L1 Physics + L1 Market.

Test hierarchy:
    L0 JAX Contract: JIT compilation, vmap parallelism, pytree stability.
    L1 Physics: Power balance, SOC dynamics, cost monotonicity.
    L1 Market: LMP sign convention, reward = LMP × P × Δt, safety/cost separation.

Domain knowledge:
    LMP = ∂(total_cost)/∂(demand_at_bus)
    Battery arbitrage: buy low (charge when LMP low), sell high (discharge when LMP high)
    Revenue = LMP × P_net × Δt
    Round-trip efficiency loss: η_rt < 1
"""

import jax
import jax.numpy as jnp
import jax.tree_util as tu
import numpy as np
import pytest

from powerzoojax.case import create_case5
from powerzoojax.envs.market.clearing import (
    CostSegments,
    PiecewiseEDSetup,
    PiecewiseEDResult,
    make_cost_segments,
    prepare_piecewise_ed,
    piecewise_ed,
)
from powerzoojax.envs.resource.battery import make_battery_bundle
from powerzoojax.envs.market.cost_based_market import (
    CostBasedMarketEnv,
    CostMarketState,
    CostMarketParams,
    make_cost_market_params,
)
from powerzoojax.envs.market.bid_based_market import (
    BidBasedMarketEnv,
    BidMarketState,
    BidMarketParams,
    make_bid_market_params,
)


# ════════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def case5():
    return create_case5()


@pytest.fixture(scope="module")
def cost_params(case5):
    return make_cost_market_params(case5, max_steps=48)


@pytest.fixture(scope="module")
def bid_params(case5):
    return make_bid_market_params(
        case5,
        max_steps=48,
        n_segments=5,
        markup_std=0.05,
    )


@pytest.fixture(scope="module")
def prng_key():
    return jax.random.PRNGKey(42)


# ════════════════════════════════════════════════════════════════════
# Cost Segments (Piecewise Clearing)
# ════════════════════════════════════════════════════════════════════

class TestMakeCostSegments:
    """Unit tests for piecewise cost segment generation."""

    def test_shape(self, case5):
        segs = make_cost_segments(case5, n_segments=5)
        n_u = case5.n_units
        assert segs.seg_widths.shape == (n_u, 5)
        assert segs.seg_prices.shape == (n_u, 5)

    def test_prices_monotonic(self, case5):
        """Prices must be non-decreasing per generator (convex cost)."""
        segs = make_cost_segments(case5, n_segments=5)
        for i in range(case5.n_units):
            prices = np.asarray(segs.seg_prices[i])
            diffs = np.diff(prices)
            assert np.all(diffs >= 0), f"Non-monotonic prices for gen {i}: {prices}"

    def test_widths_sum_to_range(self, case5):
        """Sum of segment widths = p_max - p_min for each generator."""
        segs = make_cost_segments(case5, n_segments=5)
        expected = np.asarray(case5.unit_p_max - case5.unit_p_min)
        actual = np.asarray(segs.seg_widths.sum(axis=1))
        np.testing.assert_allclose(actual, expected, atol=1e-5)

    def test_prices_positive(self, case5):
        """Offer prices should be non-negative for reasonable cost data."""
        segs = make_cost_segments(case5, n_segments=5)
        assert jnp.all(segs.seg_prices >= 0)


# ════════════════════════════════════════════════════════════════════
# Piecewise ED Solver
# ════════════════════════════════════════════════════════════════════

class TestPiecewiseED:
    """L0 + L1 tests for the piecewise-linear ED solver."""

    def test_jit_compiles(self, case5):
        """piecewise_ed must be JIT-compilable."""
        setup = prepare_piecewise_ed(case5, n_segments=5)
        node_load = jnp.array([100.0, 150.0, 200.0, 180.0, 120.0])
        fn = jax.jit(lambda ld: piecewise_ed(setup, ld))
        result = fn(node_load)
        assert result.unit_power.shape == (case5.n_units,)

    def test_power_balance(self, case5):
        """Total generation must equal total demand."""
        setup = prepare_piecewise_ed(case5, n_segments=5)
        node_load = jnp.array([100.0, 150.0, 200.0, 180.0, 120.0])
        result = piecewise_ed(setup, node_load)
        total_gen = float(jnp.sum(result.unit_power))
        total_demand = float(jnp.sum(node_load))
        np.testing.assert_allclose(total_gen, total_demand, atol=1.0)

    def test_unit_within_limits(self, case5):
        """Generator dispatch within [p_min, p_max]."""
        setup = prepare_piecewise_ed(case5, n_segments=5)
        node_load = jnp.array([100.0, 150.0, 200.0, 180.0, 120.0])
        result = piecewise_ed(setup, node_load)
        assert jnp.all(result.unit_power >= setup.p_min - 0.1)
        assert jnp.all(result.unit_power <= setup.p_max + 0.1)

    def test_lmp_shape(self, case5):
        """LMP vector must have n_nodes entries."""
        setup = prepare_piecewise_ed(case5, n_segments=5)
        node_load = jnp.array([100.0, 150.0, 200.0, 180.0, 120.0])
        result = piecewise_ed(setup, node_load)
        assert result.lmp.shape == (case5.n_nodes,)

    def test_lmp_positive(self, case5):
        """LMP should be non-negative for reasonable loads."""
        setup = prepare_piecewise_ed(case5, n_segments=5)
        node_load = jnp.array([100.0, 150.0, 200.0, 180.0, 120.0])
        result = piecewise_ed(setup, node_load)
        assert jnp.all(result.lmp >= -1.0)  # allow small numerical noise

    def test_total_cost_finite(self, case5):
        """True nonlinear cost must be finite."""
        setup = prepare_piecewise_ed(case5, n_segments=5)
        node_load = jnp.array([100.0, 150.0, 200.0, 180.0, 120.0])
        result = piecewise_ed(setup, node_load)
        assert jnp.isfinite(result.total_cost)
        assert float(result.total_cost) > 0

    def test_offer_cost_finite(self, case5):
        """Offer cost must be finite."""
        setup = prepare_piecewise_ed(case5, n_segments=5)
        node_load = jnp.array([100.0, 150.0, 200.0, 180.0, 120.0])
        result = piecewise_ed(setup, node_load)
        assert jnp.isfinite(result.offer_cost)

    def test_custom_offer_prices(self, case5):
        """piecewise_ed must accept custom offer prices."""
        setup = prepare_piecewise_ed(case5, n_segments=5)
        node_load = jnp.array([100.0, 150.0, 200.0, 180.0, 120.0])
        # Double all prices — should still solve
        custom_prices = setup.base_seg_prices * 2.0
        result = piecewise_ed(setup, node_load, custom_prices)
        assert result.unit_power.shape == (case5.n_units,)
        total_gen = float(jnp.sum(result.unit_power))
        total_demand = float(jnp.sum(node_load))
        np.testing.assert_allclose(total_gen, total_demand, atol=1.0)


# ════════════════════════════════════════════════════════════════════
# CostBasedMarketEnv
# ════════════════════════════════════════════════════════════════════

class TestCostBasedInit:
    """Constructor, spaces, and parameter factory."""

    def test_obs_space_shape(self, cost_params):
        env = CostBasedMarketEnv()
        space = env.observation_space(cost_params)
        assert space.shape == (6,)

    def test_action_space_shape(self, cost_params):
        env = CostBasedMarketEnv()
        space = env.action_space(cost_params)
        assert space.shape == (1,)

    def test_params_dimensions(self, cost_params, case5):
        assert cost_params.n_nodes == case5.n_nodes
        assert cost_params.n_units == case5.n_units
        assert cost_params.n_lines == case5.n_lines

    def test_default_params_raises(self):
        env = CostBasedMarketEnv()
        with pytest.raises(NotImplementedError):
            env.default_params()


class TestCostResourceOverride:
    """Explicit BatteryBundle via ``resources=`` overrides factory default."""

    def test_custom_bundle_runs(self, case5):
        bundle = make_battery_bundle(
            case5,
            bus_ids=[2],
            power_mw=100.0,
            capacity_mwh=400.0,
            dt_hours=0.5,
        )
        params = make_cost_market_params(case5, resources=(bundle,))
        assert int(params.resources[0].bus_idx[0]) == 1
        env = CostBasedMarketEnv()
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)
        assert obs.shape[0] == 4 + params.resources[0].obs_dim
        _, _, _, _, _, _ = env.step(key, state, jnp.array(0.0), params)


class TestCostBasedReset:
    """Reset returns valid obs and state."""

    def test_reset_returns_obs_state(self, prng_key, cost_params):
        env = CostBasedMarketEnv()
        obs, state = env.reset(prng_key, cost_params)
        assert obs.shape == (6,)
        assert isinstance(state, CostMarketState)

    def test_obs_soc_in_range(self, prng_key, cost_params):
        env = CostBasedMarketEnv()
        obs, _ = env.reset(prng_key, cost_params)
        # [lmp_norm, sin, cos, demand_norm, soc, p_norm]
        soc = float(obs[4])
        assert 0.0 <= soc <= 1.0

    def test_obs_time_bounded(self, prng_key, cost_params):
        env = CostBasedMarketEnv()
        obs, _ = env.reset(prng_key, cost_params)
        assert -1.0 <= float(obs[1]) <= 1.0  # sin
        assert -1.0 <= float(obs[2]) <= 1.0  # cos

    def test_state_initial_soc(self, prng_key, cost_params):
        env = CostBasedMarketEnv()
        _, state = env.reset(prng_key, cost_params)
        np.testing.assert_allclose(
            float(state.resource_states[0].soc[0]), 0.5, atol=1e-5)

    def test_lmp_populated(self, prng_key, cost_params):
        """LMP vector must be populated after reset."""
        env = CostBasedMarketEnv()
        _, state = env.reset(prng_key, cost_params)
        assert state.lmp.shape == (cost_params.n_nodes,)
        assert jnp.any(state.lmp != 0.0)


class TestCostBasedStep:
    """Step mechanics and market reward."""

    def test_step_returns_five_tuple(self, prng_key, cost_params):
        env = CostBasedMarketEnv()
        _, state = env.reset(prng_key, cost_params)
        result = env.step(prng_key, state, jnp.array(0.0), cost_params)
        assert len(result) == 6

    def test_zero_action_near_zero_reward(self, prng_key, cost_params):
        """Zero battery action → zero market revenue."""
        env = CostBasedMarketEnv()
        _, state = env.reset(prng_key, cost_params)
        _, _, reward, _, _, _ = env.step(prng_key, state, jnp.array(0.0), cost_params)
        np.testing.assert_allclose(float(reward), 0.0, atol=1e-3)

    def test_info_contains_keys(self, prng_key, cost_params):
        env = CostBasedMarketEnv()
        _, state = env.reset(prng_key, cost_params)
        _, _, _, costs, _, info = env.step(prng_key, state, jnp.array(0.0), cost_params)
        assert "bus_lmp" in info
        assert "is_safe" in info
        assert "cost_sum" in info
        assert "realized_p_mw" in info
        assert "requested_p_mw" in info
        assert "opf_converged" in info
        assert costs.shape == (1,)

    def test_discharge_positive_revenue(self, prng_key, cost_params):
        """Discharging (P > 0) with positive LMP → positive reward."""
        env = CostBasedMarketEnv()
        _, state = env.reset(prng_key, cost_params)
        # action=1.0 → max discharge
        _, _, reward, _, _, info = env.step(prng_key, state, jnp.array(1.0), cost_params)
        bus_lmp = float(info["bus_lmp"])
        realized = float(info["realized_p_mw"])
        if bus_lmp > 0 and realized > 0:
            assert float(reward) > 0

    def test_charge_negative_revenue(self, prng_key, cost_params):
        """Charging (P < 0) with positive LMP → negative reward."""
        env = CostBasedMarketEnv()
        _, state = env.reset(prng_key, cost_params)
        _, _, reward, _, _, info = env.step(prng_key, state, jnp.array(-1.0), cost_params)
        bus_lmp = float(info["bus_lmp"])
        realized = float(info["realized_p_mw"])
        if bus_lmp > 0 and realized < 0:
            assert float(reward) < 0

    def test_reward_formula(self, prng_key, cost_params):
        """reward = LMP × P_realized × Δt."""
        env = CostBasedMarketEnv()
        _, state = env.reset(prng_key, cost_params)
        _, _, reward, _, _, info = env.step(prng_key, state, jnp.array(0.5), cost_params)
        expected = float(info["bus_lmp"]) * float(info["realized_p_mw"]) * cost_params.delta_t_hours
        np.testing.assert_allclose(float(reward), expected, atol=1e-3)

    def test_soc_changes_on_charge(self, prng_key, cost_params):
        """Charging should increase SOC."""
        env = CostBasedMarketEnv()
        _, state = env.reset(prng_key, cost_params)
        _, new_state, _, _, _, _ = env.step(prng_key, state, jnp.array(-1.0), cost_params)
        assert float(new_state.resource_states[0].soc[0]) >= float(
            state.resource_states[0].soc[0])

    def test_soc_changes_on_discharge(self, prng_key, cost_params):
        """Discharging should decrease SOC."""
        env = CostBasedMarketEnv()
        _, state = env.reset(prng_key, cost_params)
        _, new_state, _, _, _, _ = env.step(prng_key, state, jnp.array(1.0), cost_params)
        assert float(new_state.resource_states[0].soc[0]) <= float(
            state.resource_states[0].soc[0])

    def test_safety_cost_not_in_reward(self, prng_key, cost_params):
        """Safety penalties flow through cost channel, not reward."""
        env = CostBasedMarketEnv()
        _, state = env.reset(prng_key, cost_params)
        _, _, reward, costs, _, info = env.step(prng_key, state, jnp.array(0.0), cost_params)
        # Reward is purely LMP×P×Δt (which is 0 for zero action)
        # Cost channel carries thermal overload
        assert "cost_thermal_overload" in info
        assert float(info["cost_sum"]) == pytest.approx(float(costs[0]), abs=1e-6)


class TestCostBasedJAXContract:
    """L0: JIT, vmap, pytree stability."""

    def test_jit_reset(self, prng_key, cost_params):
        env = CostBasedMarketEnv()
        fn = jax.jit(lambda k: env.reset(k, cost_params))
        obs, state = fn(prng_key)
        assert obs.shape == (6,)

    def test_jit_step(self, prng_key, cost_params):
        env = CostBasedMarketEnv()
        _, state = env.reset(prng_key, cost_params)
        fn = jax.jit(lambda k, s, a: env.step(k, s, a, cost_params))
        obs, new_state, reward, costs, done, info = fn(prng_key, state, jnp.array(0.0))
        assert obs.shape == (6,)
        assert costs.shape == (1,)

    def test_vmap_reset(self, cost_params):
        """vmap over reset with different keys."""
        env = CostBasedMarketEnv()
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        obs_batch, state_batch = jax.vmap(lambda k: env.reset(k, cost_params))(keys)
        assert obs_batch.shape == (4, 6)

    def test_vmap_step(self, prng_key, cost_params):
        """vmap over step with batch of states."""
        env = CostBasedMarketEnv()
        keys = jax.random.split(prng_key, 4)
        obs_b, state_b = jax.vmap(lambda k: env.reset(k, cost_params))(keys)
        actions = jnp.zeros(4)

        step_keys = jax.random.split(jax.random.PRNGKey(1), 4)
        obs_b2, state_b2, rew_b, cost_b, done_b, info_b = jax.vmap(
            lambda k, s, a: env.step(k, s, a, cost_params)
        )(step_keys, state_b, actions)
        assert obs_b2.shape == (4, 6)
        assert rew_b.shape == (4,)
        assert cost_b.shape == (4, 1)

    def test_pytree_structure_stable(self, prng_key, cost_params):
        """State pytree structure must not change across steps."""
        env = CostBasedMarketEnv()
        _, state0 = env.reset(prng_key, cost_params)
        _, state1, *_ = env.step(prng_key, state0, jnp.array(0.5), cost_params)

        struct0 = tu.tree_structure(state0)
        struct1 = tu.tree_structure(state1)
        assert struct0 == struct1

    def test_scan_rollout(self, prng_key, cost_params):
        """lax.scan rollout must work without errors."""
        env = CostBasedMarketEnv()
        _, init_state = env.reset(prng_key, cost_params)

        def scan_step(carry, x):
            key, state = carry
            key, k_step = jax.random.split(key)
            action = jnp.float32(0.0)
            obs, new_state, reward, costs, done, info = env.step(
                k_step, state, action, cost_params
            )
            return (key, new_state), (obs, reward, done)

        carry = (prng_key, init_state)
        (_, final_state), (obs_traj, rew_traj, done_traj) = jax.lax.scan(
            scan_step, carry, None, length=10)
        assert obs_traj.shape == (10, 6)
        assert rew_traj.shape == (10,)


class TestCostBasedAutoReset:
    """Auto-reset behaviour for lax.scan compatibility."""

    def test_auto_reset_on_done(self, cost_params):
        """When done=True, state should be reset."""
        env = CostBasedMarketEnv()
        short_params = cost_params.replace(max_steps=2)
        key = jax.random.PRNGKey(0)
        _, state = env.reset(key, short_params)

        # Step twice to trigger done
        _, state, *_ = env.step(key, state, jnp.array(0.0), short_params)
        _, state2, _, _, done, _ = env.step(key, state, jnp.array(0.0), short_params)

        # After auto-reset, time_step should be 0
        assert bool(done) is True
        assert int(state2.time_step) == 0

    def test_soc_reset_on_done(self, cost_params):
        """SOC should reset to initial_soc when episode ends."""
        env = CostBasedMarketEnv()
        short_params = cost_params.replace(max_steps=2)
        key = jax.random.PRNGKey(0)
        _, state = env.reset(key, short_params)

        # Discharge to lower SOC, then finish episode
        _, state, *_ = env.step(key, state, jnp.array(1.0), short_params)
        _, state2, _, _, done, _ = env.step(key, state, jnp.array(1.0), short_params)
        assert bool(done) is True
        init_soc = float(short_params.resources[0].initial_soc[0])
        np.testing.assert_allclose(
            float(state2.resource_states[0].soc[0]), init_soc, atol=1e-4)


# ════════════════════════════════════════════════════════════════════
# BidBasedMarketEnv
# ════════════════════════════════════════════════════════════════════

class TestBidBasedInit:
    """Constructor and spaces."""

    def test_obs_space_shape(self, bid_params):
        env = BidBasedMarketEnv()
        space = env.observation_space(bid_params)
        assert space.shape == (7,)

    def test_action_space_shape(self, bid_params):
        env = BidBasedMarketEnv()
        space = env.action_space(bid_params)
        assert space.shape == (1,)

    def test_default_params_raises(self):
        env = BidBasedMarketEnv()
        with pytest.raises(NotImplementedError):
            env.default_params()


class TestBidBasedReset:
    """Reset generates offer curves and runs clearing."""

    def test_reset_returns_obs_state(self, prng_key, bid_params):
        env = BidBasedMarketEnv()
        obs, state = env.reset(prng_key, bid_params)
        assert obs.shape == (7,)
        assert isinstance(state, BidMarketState)

    def test_offer_prices_populated(self, prng_key, bid_params):
        env = BidBasedMarketEnv()
        _, state = env.reset(prng_key, bid_params)
        assert state.offer_prices.shape == (bid_params.n_units, bid_params.n_segments)
        assert jnp.all(state.offer_prices > 0)

    def test_lmp_available(self, prng_key, bid_params):
        env = BidBasedMarketEnv()
        _, state = env.reset(prng_key, bid_params)
        assert state.lmp.shape == (bid_params.n_nodes,)


class TestBidBasedStep:
    """Step mechanics."""

    def test_step_returns_five_tuple(self, prng_key, bid_params):
        env = BidBasedMarketEnv()
        _, state = env.reset(prng_key, bid_params)
        result = env.step(prng_key, state, jnp.array(0.0), bid_params)
        assert len(result) == 6

    def test_info_contains_market_fields(self, prng_key, bid_params):
        env = BidBasedMarketEnv()
        _, state = env.reset(prng_key, bid_params)
        _, _, _, costs, _, info = env.step(prng_key, state, jnp.array(0.0), bid_params)
        assert "offer_cost" in info
        assert "true_cost" in info
        assert "cost_model" in info
        assert "ed_converged" in info
        assert costs.shape == (1,)

    def test_zero_action_near_zero_reward(self, prng_key, bid_params):
        env = BidBasedMarketEnv()
        _, state = env.reset(prng_key, bid_params)
        _, _, reward, _, _, _ = env.step(prng_key, state, jnp.array(0.0), bid_params)
        np.testing.assert_allclose(float(reward), 0.0, atol=1e-3)

    def test_reward_formula(self, prng_key, bid_params):
        """reward = LMP × P_realized × Δt."""
        env = BidBasedMarketEnv()
        _, state = env.reset(prng_key, bid_params)
        _, _, reward, _, _, info = env.step(prng_key, state, jnp.array(0.5), bid_params)
        expected = float(info["bus_lmp"]) * float(info["realized_p_mw"]) * bid_params.delta_t_hours
        np.testing.assert_allclose(float(reward), expected, atol=1e-3)

    def test_offer_prices_persist_across_steps(self, prng_key, bid_params):
        """Offer prices should not change within an episode."""
        env = BidBasedMarketEnv()
        _, state0 = env.reset(prng_key, bid_params)
        _, state1, *_ = env.step(prng_key, state0, jnp.array(0.0), bid_params)
        np.testing.assert_allclose(
            np.asarray(state0.offer_prices),
            np.asarray(state1.offer_prices),
            atol=1e-7,
        )


class TestBidBasedJAXContract:
    """L0: JIT, vmap, pytree stability."""

    def test_jit_reset(self, prng_key, bid_params):
        env = BidBasedMarketEnv()
        fn = jax.jit(lambda k: env.reset(k, bid_params))
        obs, state = fn(prng_key)
        assert obs.shape == (7,)

    def test_jit_step(self, prng_key, bid_params):
        env = BidBasedMarketEnv()
        _, state = env.reset(prng_key, bid_params)
        fn = jax.jit(lambda k, s, a: env.step(k, s, a, bid_params))
        obs, _, reward, costs, done, info = fn(prng_key, state, jnp.array(0.0))
        assert obs.shape == (7,)
        assert costs.shape == (1,)

    def test_vmap_reset(self, bid_params):
        env = BidBasedMarketEnv()
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        obs_b, state_b = jax.vmap(lambda k: env.reset(k, bid_params))(keys)
        assert obs_b.shape == (4, 7)

    def test_vmap_step(self, prng_key, bid_params):
        env = BidBasedMarketEnv()
        keys = jax.random.split(prng_key, 4)
        _, state_b = jax.vmap(lambda k: env.reset(k, bid_params))(keys)
        actions = jnp.zeros(4)
        step_keys = jax.random.split(jax.random.PRNGKey(1), 4)
        obs_b2, _, rew_b, cost_b, done_b, _ = jax.vmap(
            lambda k, s, a: env.step(k, s, a, bid_params)
        )(step_keys, state_b, actions)
        assert obs_b2.shape == (4, 7)
        assert rew_b.shape == (4,)
        assert cost_b.shape == (4, 1)

    def test_pytree_structure_stable(self, prng_key, bid_params):
        env = BidBasedMarketEnv()
        _, state0 = env.reset(prng_key, bid_params)
        _, state1, *_ = env.step(prng_key, state0, jnp.array(0.0), bid_params)
        assert tu.tree_structure(state0) == tu.tree_structure(state1)

    def test_scan_rollout(self, prng_key, bid_params):
        env = BidBasedMarketEnv()
        _, init_state = env.reset(prng_key, bid_params)

        def scan_step(carry, x):
            key, state = carry
            key, k_step = jax.random.split(key)
            obs, new_state, reward, costs, done, info = env.step(
                k_step, state, jnp.float32(0.0), bid_params)
            return (key, new_state), (obs, reward, done)

        (_, final_state), (obs_traj, rew_traj, done_traj) = jax.lax.scan(
            scan_step, (prng_key, init_state), None, length=10)
        assert obs_traj.shape == (10, 7)
        assert rew_traj.shape == (10,)


class TestBidBasedAutoReset:
    """Auto-reset for lax.scan compatibility."""

    def test_auto_reset_on_done(self, bid_params):
        env = BidBasedMarketEnv()
        short_params = bid_params.replace(max_steps=2)
        key = jax.random.PRNGKey(0)
        _, state = env.reset(key, short_params)

        _, state, *_ = env.step(key, state, jnp.array(0.0), short_params)
        _, state2, _, _, done, _ = env.step(key, state, jnp.array(0.0), short_params)
        assert bool(done) is True
        assert int(state2.time_step) == 0


# ════════════════════════════════════════════════════════════════════
# Multi-step episode rollout
# ════════════════════════════════════════════════════════════════════

class TestMultiStepEpisode:
    """Full episode rollout for both envs."""

    def test_lmp_episode_completes(self, prng_key, cost_params):
        env = CostBasedMarketEnv()
        short_params = cost_params.replace(max_steps=5)
        _, state = env.reset(prng_key, short_params)
        key = prng_key
        done_seen = False
        for _ in range(5):
            key, k_step = jax.random.split(key)
            _, state, _, _, done, _ = env.step(k_step, state, jnp.array(0.0), short_params)
            if bool(done):
                done_seen = True
                break
        assert done_seen

    def test_bid_episode_completes(self, prng_key, bid_params):
        env = BidBasedMarketEnv()
        short_params = bid_params.replace(max_steps=5)
        _, state = env.reset(prng_key, short_params)
        key = prng_key
        done_seen = False
        for _ in range(5):
            key, k_step = jax.random.split(key)
            _, state, _, _, done, _ = env.step(k_step, state, jnp.array(0.0), short_params)
            if bool(done):
                done_seen = True
                break
        assert done_seen
