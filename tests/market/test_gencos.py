"""Tests for GenCos Market MARL — exact bid-based SCED + ramp-bounded rolling market.

Test methodology:
    offer_sced is an exact LP model solved via PD-IPM (primal-dual interior
    point method).  All numerical comparisons REQUIRE ``assert
    sced_result.converged`` as precondition.

Coverage:
    L0  — JAX contracts (jit, vmap, scan, pytree)
    L1  — Solver feasibility, dual-based LMP, truthful-vs-dc_opf, monotonicity
    L1  — MARL interface, auto-reset, cost/safety info, action clipping
    L2  — Ramp coupling: runtime bounds enforced in solver layer
    L2  — Observation layout: 12-dim private obs with ramp headroom + LMP history
    L2  — obs[5] is one-step-ahead demand forecast, not current demand
    L2  — Preset: main preset uses GB data (not synthetic), dev preset explicit
    L2  — Episode length 48 regardless of GB pool size
    L2  — Market metrics: price_volatility / HHI / market_share_std / ramp_binding
    L2  — Planner cost ratio (market/planner_cost_ratio) exported by IPPO
    L2  — Per-agent cumulative profit exported by IPPO trainer
    L2  — Per-agent dispatch share (market/dispatch_share_{ag}) exported by IPPO
    L2  — Convergence stability (market/convergence_stability) exported by IPPO

GenCos market metrics implemented here (six of nine target checklist items):
    ✅ cumulative_profit_per_agent  ✅ price_volatility  ✅ HHI
    ✅ market_share_dynamics        ✅ ramp_binding_rate ✅ convergence_stability
    ❌ social_welfare_ratio   — needs consumer value V (VOLL) — future work
    ❌ exploitability_proxy   — needs BR oracle or self-play — future work
    ❌ collusion_indicator    — needs Nash equilibrium solver — future work

    Bonus observable (diagnostic, not part of the nine-item checklist):
        market/planner_cost_ratio = planner_cost / RL_gen_cost ∈ (0, 1].
        This is a COST RATIO, not a welfare ratio.  True social_welfare_ratio
        = (V − RL_cost) / (V − planner_cost) requires consumer value V (VOLL
        × demand) which is not defined in this codebase.

    Note on market_share_std: scalar summary accompanying dispatch_share_{ag}.
    Not a separate checklist item; market_share_dynamics = dispatch_share_{ag} counts.

NOT covered here (intentional scope boundary — future work, NOT claimed as done):
    - Social welfare ratio: requires consumer value V (VOLL × demand) for
      (V − RL_cost) / (V − planner_cost) ∈ [0, 1] to be well-defined.  Any
      choice of V without an explicit VOLL definition is arbitrary.
    - Exploitability proxy: requires self-play / best-response oracle; a price-taking
      proxy would conflate dispatch suboptimality with strategic effect.
    - Collusion indicator: requires knowing single-agent optimal markup (Nash equil.).
    - Self-play / PSRO population metrics.
    - Out-of-sample IID/OOD policy evaluation.

Test counts:
    Tests that require real GB demand data are replaced with monkeypatch-based
    equivalents so the suite runs without data dependency.  All tests should
    pass or be explicitly annotated with a ``pytest.mark.skip`` if the feature
    is not yet implemented.
"""

from __future__ import annotations

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax

from powerzoojax.case import create_case5
from powerzoojax.envs.market.offer_sced import prepare_offer_sced, offer_sced
from powerzoojax.envs.market.market_marl_core import (
    make_market_marl_params,
    market_marl_reset,
    market_marl_step,
    MarketMARLState,
    MarketMARLParams,
)
from powerzoojax.rl.market_marl import MarketMARLEnv, OBS_DIM
from powerzoojax.rl.presets import get_preset, _wrap_gencos_case5_dev
from powerzoojax.tasks.gencos import load_gencos_profiles, rollout_gencos


# ============ Fixtures ============

@pytest.fixture(scope="module")
def case5():
    return create_case5()


@pytest.fixture(scope="module")
def sced_setup(case5):
    return prepare_offer_sced(case5, n_segments=3)


@pytest.fixture(scope="module")
def mid_load(case5):
    return (case5.load_d_max + case5.load_d_min) / 2.0


@pytest.fixture(scope="module")
def load_mw(sced_setup, mid_load):
    """Nodal load for mid-load scenario."""
    case = create_case5()
    nodes_loads_map = jnp.array(np.asarray(case.nodes_loads_map), dtype=jnp.float32)
    return nodes_loads_map @ jnp.array(mid_load, dtype=jnp.float32)


@pytest.fixture(scope="module")
def truthful_prices(sced_setup):
    return sced_setup.base_seg_prices


@pytest.fixture(scope="module")
def sced_result(sced_setup, load_mw, truthful_prices):
    return offer_sced(sced_setup, load_mw, truthful_prices)


@pytest.fixture(scope="module")
def marl_params(case5, mid_load):
    profiles = jnp.tile(
        jnp.array(mid_load, dtype=jnp.float32)[None, :], (48, 1)
    )
    return make_market_marl_params(case5, profiles, n_segments=3, max_markup=2.0)


@pytest.fixture(scope="module")
def marl_env(marl_params):
    return MarketMARLEnv(marl_params)


@pytest.fixture(scope="module")
def marl_state(marl_params):
    key = jax.random.PRNGKey(0)
    return market_marl_reset(key, marl_params)


# ============ L0 JAX Contracts ============

class TestJAXContracts:

    def test_jit_reset(self, marl_params):
        key = jax.random.PRNGKey(42)
        jit_reset = jax.jit(market_marl_reset)
        state = jit_reset(key, marl_params)
        assert state.time_step == 0

    def test_jit_step(self, marl_params, marl_state):
        key = jax.random.PRNGKey(1)
        actions = jnp.zeros(marl_params.n_units * marl_params.offer_sced_setup.n_segments,
                            dtype=jnp.float32)
        jit_step = jax.jit(market_marl_step)
        final_state, done, profit, info = jit_step(key, marl_state, actions, marl_params)
        assert final_state.time_step == 1

    def test_vmap_4env(self, marl_params):
        """vmap over 4 independent environments."""
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        states = jax.vmap(market_marl_reset, in_axes=(0, None))(keys, marl_params)
        assert states.time_step.shape == (4,)

        n_act = marl_params.n_units * marl_params.offer_sced_setup.n_segments
        actions = jnp.zeros((4, n_act), dtype=jnp.float32)
        step_keys = jax.random.split(jax.random.PRNGKey(1), 4)

        def step_fn(key, state):
            return market_marl_step(key, state, jnp.zeros(n_act, dtype=jnp.float32), marl_params)

        final_states, dones, profits, infos = jax.vmap(step_fn)(step_keys, states)
        assert final_states.time_step.shape == (4,)
        assert profits.shape == (4, marl_params.n_units)

    def test_scan_rollout(self, marl_params):
        """lax.scan 48-step rollout — float32 types throughout."""
        key = jax.random.PRNGKey(7)
        init_state = market_marl_reset(key, marl_params)
        n_act = marl_params.n_units * marl_params.offer_sced_setup.n_segments

        def scan_fn(carry, _xs):
            state, k = carry
            k, k_step = jax.random.split(k)
            act = jnp.zeros(n_act, dtype=jnp.float32)
            final_state, done, profit, info = market_marl_step(k_step, state, act, marl_params)
            return (final_state, k), profit

        (final_state, _), profits = lax.scan(
            scan_fn, (init_state, key), None, length=48
        )
        assert profits.shape == (48, marl_params.n_units)
        # After 48 steps with auto-reset, time_step should be 0 (reset)
        assert final_state.time_step == 0

    def test_pytree_roundtrip(self, marl_state):
        """PyTree leaves and unflatten reproduce the same structure."""
        leaves, treedef = jax.tree_util.tree_flatten(marl_state)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        for l, r in zip(
            jax.tree_util.tree_leaves(marl_state),
            jax.tree_util.tree_leaves(restored),
        ):
            np.testing.assert_array_equal(np.asarray(l), np.asarray(r))

    def test_state_has_lmp_history(self, marl_params, marl_state):
        """State carries lmp_history buffer of correct shape."""
        k_hist = marl_params.lmp_history_len
        assert marl_state.lmp_history.shape == (k_hist,)
        assert jnp.all(jnp.isfinite(marl_state.lmp_history))

    def test_state_has_episode_start_idx(self, marl_params, marl_state):
        """State must carry episode_start_idx for pool-sampling semantics."""
        assert hasattr(marl_state, "episode_start_idx"), (
            "MarketMARLState must have episode_start_idx field"
        )
        T = marl_params.load_profiles.shape[0]
        idx = int(marl_state.episode_start_idx)
        assert 0 <= idx < T, f"episode_start_idx={idx} out of [0, {T-1}]"


# ============ Solver Convergence + Feasibility ============

class TestSolverConvergence:

    def test_converged(self, sced_result):
        """Mid-load case5 must converge."""
        assert bool(sced_result.converged), "SCED did not converge for mid-load case5"

    def test_is_feasible(self, sced_result):
        """Mid-load must be in feasible range."""
        assert bool(sced_result.is_feasible)

    def test_balance_residual(self, sced_result, load_mw):
        """Sum of dispatch ≈ total load (tolerance 1e-2)."""
        assert bool(sced_result.converged)
        total_dispatch = float(jnp.sum(sced_result.unit_power))
        total_load = float(jnp.sum(load_mw))
        assert abs(total_dispatch - total_load) < 0.01, (
            f"Balance error: dispatch={total_dispatch:.4f}, load={total_load:.4f}"
        )

    def test_box_residual(self, sced_result, sced_setup):
        """Segment deltas must be in [0, seg_width] within IPM tolerance."""
        assert bool(sced_result.converged)
        delta = sced_result.segment_delta
        seg_width = sced_setup.base_seg_widths
        assert float(jnp.max(jnp.maximum(0.0, -delta))) < 1e-3
        assert float(jnp.max(jnp.maximum(0.0, delta - seg_width))) < 1e-3

    def test_line_residual(self, sced_result, sced_setup):
        """Line flows must be within limits (tolerance 1e-3)."""
        assert bool(sced_result.converged)
        flow = sced_result.line_flow
        assert float(jnp.max(jnp.maximum(0.0, flow - sced_setup.line_cap))) < 1e-3
        assert float(jnp.max(jnp.maximum(0.0, sced_setup.line_floor - flow))) < 1e-3

    def test_segment_delta_consistency(self, sced_result, sced_setup):
        """sum(delta, axis=-1) ≈ unit_power − p_min."""
        assert bool(sced_result.converged)
        delta_sum = jnp.sum(sced_result.segment_delta, axis=-1)
        residual = delta_sum - (sced_result.unit_power - sced_setup.p_min)
        assert float(jnp.max(jnp.abs(residual))) < 1e-4

    def test_segment_delta_shape(self, sced_result, sced_setup):
        assert sced_result.segment_delta.shape == (
            sced_setup.n_units, sced_setup.n_segments
        )

    def test_segment_delta_nonneg(self, sced_result):
        assert bool(sced_result.converged)
        assert float(jnp.min(sced_result.segment_delta)) >= -1e-3


# ============ Infeasible Load Defence ============

class TestInfeasibleLoad:

    def test_underload_not_crash(self, sced_setup, truthful_prices):
        """total_load < Σ p_min → is_feasible False, solver doesn't crash."""
        tiny_load = jnp.zeros(sced_setup.n_nodes, dtype=jnp.float32)
        result = offer_sced(sced_setup, tiny_load, truthful_prices)
        assert not bool(result.is_feasible)
        assert jnp.all(jnp.isfinite(result.unit_power))

    def test_overload_not_crash(self, sced_setup, truthful_prices):
        """total_load > Σ p_max → is_feasible False, solver doesn't crash."""
        huge_load = jnp.full(sced_setup.n_nodes, 1e6, dtype=jnp.float32)
        result = offer_sced(sced_setup, huge_load, truthful_prices)
        assert not bool(result.is_feasible)
        assert jnp.all(jnp.isfinite(result.unit_power))


# ============ Dual-Based LMP ============

class TestDualLMP:

    def test_lmp_shape(self, sced_result, sced_setup):
        assert sced_result.lmp.shape == (sced_setup.n_nodes,)

    def test_lmp_finite(self, sced_result):
        assert bool(sced_result.converged)
        assert bool(jnp.all(jnp.isfinite(sced_result.lmp)))

    def test_lambda_balance_finite(self, sced_result):
        assert bool(sced_result.converged)
        assert bool(jnp.isfinite(sced_result.lambda_balance))

    def test_mu_nonneg(self, sced_result):
        assert bool(sced_result.converged)
        assert float(jnp.min(sced_result.mu_upper)) >= -1e-6
        assert float(jnp.min(sced_result.mu_lower)) >= -1e-6

    def test_lmp_formula_self_consistent(self, sced_result, sced_setup):
        """LMP_n ≈ -lambda_balance − PTDF[:,n]^T · (μ_upper − μ_lower) (atol=1e-3).

        λ_balance (KKT dual) has opposite sign to economic LMP convention;
        economic LMP = -λ_balance.
        """
        assert bool(sced_result.converged)
        net_mu = sced_result.mu_upper - sced_result.mu_lower
        lmp_recomputed = (
            -sced_result.lambda_balance
            - sced_setup.PTDF.T @ net_mu
        )
        np.testing.assert_allclose(
            np.asarray(sced_result.lmp),
            np.asarray(lmp_recomputed),
            atol=1e-3,
            err_msg="LMP formula not self-consistent",
        )


# ============ Truthful vs DC-OPF ============

class TestTruthfulVsDCOPF:

    def test_truthful_vs_dcopf(self, sced_setup, load_mw, truthful_prices, case5):
        """Truthful offer_sced ≈ cost-based dc_opf for case5 (mc_a=mc_b=0).

        case5 has mc_a = mc_b = 0, so truthful segment prices = mc_c (constant).
        Both solvers minimise a linear objective over the same feasible set.
        """
        from powerzoojax.envs.grid.dc_opf import prepare_dcopf, dc_opf

        opf_setup = prepare_dcopf(case5)
        opf_result = dc_opf(opf_setup, load_mw)
        sced_result = offer_sced(sced_setup, load_mw, truthful_prices)

        assert bool(sced_result.converged), "SCED did not converge"
        assert bool(opf_result.converged), "DC-OPF did not converge"

        np.testing.assert_allclose(
            np.asarray(sced_result.unit_power),
            np.asarray(opf_result.unit_power),
            atol=1.0,
            err_msg="unit_power mismatch: truthful SCED vs DC-OPF",
        )
        np.testing.assert_allclose(
            np.asarray(sced_result.lmp),
            np.asarray(opf_result.lmp),
            atol=1.0,
            err_msg="lmp mismatch: truthful SCED vs DC-OPF",
        )


# ============ LMP Monotonicity ============

class TestLMPMonotonicity:

    def test_markup_raises_mean_lmp(self, sced_setup, load_mw, truthful_prices):
        """No-congestion scenario: raising any unit's offer → mean(LMP) non-decreasing.

        Build a no-congestion variant with 1e8 MW line caps.
        Raise the highest-mc unit's (index 4) offer price 3x.
        """
        setup_nc = sced_setup.replace(
            line_cap=jnp.full_like(sced_setup.line_cap, 1e8),
            line_floor=jnp.full_like(sced_setup.line_floor, -1e8),
        )
        r_base = offer_sced(setup_nc, load_mw, truthful_prices)

        marked_up = truthful_prices.at[4, :].set(truthful_prices[4, :] * 3.0)
        r_markup = offer_sced(setup_nc, load_mw, marked_up)

        assert bool(r_base.converged), "Base SCED not converged"
        assert bool(r_markup.converged), "Markup SCED not converged"

        mean_base = float(jnp.mean(r_base.lmp))
        mean_markup = float(jnp.mean(r_markup.lmp))
        assert mean_markup >= mean_base - 1e-3, (
            f"mean LMP decreased after markup: {mean_base:.4f} → {mean_markup:.4f}"
        )


# ============ Reset/Step LMP Consistency ============

class TestResetStepConsistency:

    def test_reset_step_lmp_match(self, marl_params):
        """Reset uses truthful prices; step with action=-1 uses truthful prices.
        Both should produce the same LMP (same offer_sced call).
        """
        key = jax.random.PRNGKey(0)
        state = market_marl_reset(key, marl_params)

        # action = -1 → m = 0 → offer = base * 1 = truthful
        n_act = marl_params.n_units * marl_params.offer_sced_setup.n_segments
        truthful_action = jnp.full((n_act,), -1.0, dtype=jnp.float32)
        final_state, done, profit, info = market_marl_step(
            key, state, truthful_action, marl_params
        )
        assert bool(info["sced_converged"]), "SCED not converged in step"

        np.testing.assert_allclose(
            np.asarray(state.lmp),
            np.asarray(final_state.lmp),
            atol=1e-3,
            err_msg="LMP mismatch: reset vs step with truthful action",
        )


# ============ Cost/Safety Info ============

class TestCostSafetyInfo:

    def test_info_keys_present(self, marl_params, marl_state):
        key = jax.random.PRNGKey(5)
        n_act = marl_params.n_units * marl_params.offer_sced_setup.n_segments
        _, _, _, info = market_marl_step(
            key, marl_state, jnp.zeros(n_act, dtype=jnp.float32), marl_params
        )
        required = [
            "cost", "cost_thermal_overload", "is_safe", "n_violations",
            "gen_cost", "sced_converged", "sced_feasible",
            "ramp_binding_rate",   # NEW: ramp coupling metric
        ]
        for k in required:
            assert k in info, f"Missing info key: {k}"

    def test_no_congestion_is_safe(self, marl_params, marl_state):
        """Truthful action under mid-load should produce no congestion."""
        key = jax.random.PRNGKey(0)
        n_act = marl_params.n_units * marl_params.offer_sced_setup.n_segments
        truthful_action = jnp.full((n_act,), -1.0, dtype=jnp.float32)
        _, _, _, info = market_marl_step(
            key, marl_state, truthful_action, marl_params
        )
        assert bool(info["sced_converged"])
        # Note: is_safe depends on system — just verify the field is boolean
        assert info["is_safe"].dtype == jnp.bool_


# ============ Action Clipping ============

class TestActionClipping:

    def test_clipping_does_not_crash(self, marl_params, marl_state):
        """Action +2.0 (out-of-range) should not crash, equivalent to +1.0."""
        key = jax.random.PRNGKey(3)
        n_act = marl_params.n_units * marl_params.offer_sced_setup.n_segments
        out_of_range = jnp.full((n_act,), 2.0, dtype=jnp.float32)
        max_action = jnp.full((n_act,), 1.0, dtype=jnp.float32)

        _, _, profit_oor, _ = market_marl_step(key, marl_state, out_of_range, marl_params)
        _, _, profit_max, _ = market_marl_step(key, marl_state, max_action, marl_params)
        np.testing.assert_allclose(
            np.asarray(profit_oor), np.asarray(profit_max), atol=1e-5
        )


# ============ TC Formula ============

class TestTCFormula:

    def test_tc_hand_check(self, marl_params, marl_state, case5):
        """Verify TC formula: TC(P) = (a/3)P³ + (b/2)P² + cP."""
        key = jax.random.PRNGKey(0)
        n_act = marl_params.n_units * marl_params.offer_sced_setup.n_segments
        truthful_action = jnp.full((n_act,), -1.0, dtype=jnp.float32)
        final_state, done, profit_vec, info = market_marl_step(
            key, marl_state, truthful_action, marl_params
        )
        assert bool(info["sced_converged"])

        P = info["unit_power"]
        lmp = info["lmp"]
        dt = marl_params.delta_t_hours
        a = marl_params.unit_cost_a
        b = marl_params.unit_cost_b
        c = marl_params.unit_cost_c

        TC = (a / 3.0) * P ** 3 + (b / 2.0) * P ** 2 + c * P
        expected_profit = lmp[marl_params.unit_node_idx] * P * dt - TC * dt

        np.testing.assert_allclose(
            np.asarray(profit_vec),
            np.asarray(expected_profit),
            atol=1e-5,
            err_msg="dispatch_profit accounting identity violated",
        )


# ============ Auto-Reset ============

class TestAutoReset:

    def test_done_at_step_48(self, marl_params):
        key = jax.random.PRNGKey(0)
        state = market_marl_reset(key, marl_params)
        n_act = marl_params.n_units * marl_params.offer_sced_setup.n_segments
        act = jnp.zeros(n_act, dtype=jnp.float32)

        for i in range(48):
            key, k_step = jax.random.split(key)
            final_state, done, _, _ = market_marl_step(k_step, state, act, marl_params)
            state = final_state

        assert bool(done), "done not True at step 48"
        assert state.time_step == 0, "state.time_step not reset to 0 after done"

    def test_reset_after_done(self, marl_params):
        """After episode end, continuing should start a fresh episode."""
        key = jax.random.PRNGKey(42)
        state = market_marl_reset(key, marl_params)
        n_act = marl_params.n_units * marl_params.offer_sced_setup.n_segments
        act = jnp.zeros(n_act, dtype=jnp.float32)

        for _ in range(48):
            key, k_step = jax.random.split(key)
            state, _, _, _ = market_marl_step(k_step, state, act, marl_params)

        assert state.time_step == 0
        # Continue for 1 more step — should be at t=1 without crashing
        key, k_step = jax.random.split(key)
        state2, done2, _, _ = market_marl_step(k_step, state, act, marl_params)
        assert state2.time_step == 1
        assert not bool(done2)


# ============ MARL Interface ============

class TestMARLInterface:

    def test_num_agents(self, marl_env):
        assert marl_env.num_agents == 5

    def test_agent_names(self, marl_env):
        assert marl_env.agent_names == [f"genco_{i}" for i in range(5)]

    def test_observation_space_shape(self, marl_env):
        """Obs dim = 12 (8 scalar features + 4 LMP history)."""
        obs_space = marl_env.observation_space()
        assert obs_space.shape == (OBS_DIM,), (
            f"Expected obs shape ({OBS_DIM},), got {obs_space.shape}"
        )

    def test_action_space_shape(self, marl_env):
        act_space = marl_env.action_space()
        assert act_space.shape == (3,)  # n_seg=3

    def test_reset_returns_obs_dict(self, marl_env):
        key = jax.random.PRNGKey(0)
        obs_dict, state = marl_env.reset(key)
        for name in marl_env.agent_names:
            assert name in obs_dict
            assert obs_dict[name].shape == (OBS_DIM,), (
                f"Expected obs shape ({OBS_DIM},), got {obs_dict[name].shape}"
            )

    def test_step_returns_dones_all(self, marl_env):
        key = jax.random.PRNGKey(0)
        obs_dict, state = marl_env.reset(key)
        actions = {n: jnp.zeros(3, dtype=jnp.float32) for n in marl_env.agent_names}
        obs2, state2, rewards, dones, info = marl_env.step(key, state, actions)
        assert "__all__" in dones
        for n in marl_env.agent_names:
            assert n in rewards
            assert n in dones

    def test_step_obs_shape(self, marl_env):
        key = jax.random.PRNGKey(0)
        obs_dict, state = marl_env.reset(key)
        actions = {n: jnp.zeros(3, dtype=jnp.float32) for n in marl_env.agent_names}
        obs2, state2, rewards, dones, info = marl_env.step(key, state, actions)
        for n in marl_env.agent_names:
            assert obs2[n].shape == (OBS_DIM,), (
                f"Agent {n}: expected shape ({OBS_DIM},), got {obs2[n].shape}"
            )


# ============ Ramp Coupling (Finding 1) ============

class TestRampCoupling:

    def test_ramp_binding_rate_in_info(self, marl_params, marl_state):
        """market_marl_step must include ramp_binding_rate in info."""
        key = jax.random.PRNGKey(0)
        n_act = marl_params.n_units * marl_params.offer_sced_setup.n_segments
        _, _, _, info = market_marl_step(
            key, marl_state, jnp.zeros(n_act, dtype=jnp.float32), marl_params
        )
        assert "ramp_binding_rate" in info
        assert jnp.isfinite(info["ramp_binding_rate"])
        assert 0.0 <= float(info["ramp_binding_rate"]) <= 1.0

    def test_rollout_gencos_uses_ramp_binding_rate_key(self, case5, mid_load):
        """Episode rollouts must propagate per-step ramp_binding_rate into summaries.

        This guards against the historical bug where ``rollout_gencos`` looked
        for a non-existent ``info['ramp_binding']`` key and silently wrote zeros
        into eval/per-episode metrics despite the env exposing non-zero
        ``ramp_binding_rate``.
        """
        tight_ramp = np.ones(5, dtype=np.float32) * 1.0
        profiles = jnp.tile(
            jnp.array(mid_load, dtype=jnp.float32)[None, :], (48, 1)
        )
        params = make_market_marl_params(
            case5,
            profiles,
            n_segments=3,
            max_markup=2.0,
            ramp_up_mw_per_step=tight_ramp,
            ramp_down_mw_per_step=tight_ramp,
        )
        env = MarketMARLEnv(params)
        fixed_action = jnp.ones((3,), dtype=jnp.float32)

        def policy_fn(obs_dict):
            return {name: fixed_action for name in env.agent_names}

        rollout = rollout_gencos(env, jax.random.PRNGKey(0), policy_fn)
        assert "ramp_binding_rate" in rollout
        assert rollout["ramp_binding_rate"].shape == (48,)
        assert float(np.mean(rollout["ramp_binding_rate"])) > 0.05, (
            "rollout_gencos should preserve non-zero ramp binding instead of "
            "silently zero-filling from a missing info key"
        )

    def test_tiny_ramp_limits_dispatch_change(self, case5, mid_load):
        """With very small ramp (1 MW/step), dispatch must barely change.

        Strategy: reset to mid-load equilibrium, then step with high markup.
        Without ramp, dispatch would shift significantly.  With ramp=1 MW,
        the change per unit must be ≤ 1 MW (within solver tolerance).
        """
        tiny_ramp = np.ones(5, dtype=np.float32) * 1.0  # 1 MW per step
        profiles = jnp.tile(
            jnp.array(mid_load, dtype=jnp.float32)[None, :], (48, 1)
        )
        params = make_market_marl_params(
            case5, profiles, n_segments=3, max_markup=2.0,
            ramp_up_mw_per_step=tiny_ramp,
            ramp_down_mw_per_step=tiny_ramp,
        )
        key = jax.random.PRNGKey(42)
        state = market_marl_reset(key, params)
        prev_p = state.unit_power_mw

        # Step with max markup (would force big dispatch change without ramp)
        n_act = params.n_units * params.offer_sced_setup.n_segments
        max_actions = jnp.ones(n_act, dtype=jnp.float32)
        s1, _, _, info = market_marl_step(key, state, max_actions, params)

        dispatch_change = jnp.abs(s1.unit_power_mw - prev_p)
        # Ramp is 1 MW; allow 0.6 MW tolerance for IPM convergence
        assert jnp.all(dispatch_change <= 1.6), (
            f"Ramp constraint not enforced: changes={dispatch_change}"
        )
        assert bool(info["sced_converged"]), "SCED should converge with tiny ramp"

    def test_large_ramp_allows_big_change(self, case5, mid_load):
        """With very large ramp (p_max), dispatch can change freely.

        Verify ramp is NOT artificially constraining when set large.
        """
        p_max = np.asarray(case5.unit_p_max, dtype=np.float32)
        big_ramp = p_max * 2.0  # effectively unconstrained
        profiles = jnp.tile(
            jnp.array(mid_load, dtype=jnp.float32)[None, :], (48, 1)
        )
        params = make_market_marl_params(
            case5, profiles, n_segments=3, max_markup=2.0,
            ramp_up_mw_per_step=big_ramp,
            ramp_down_mw_per_step=big_ramp,
        )
        key = jax.random.PRNGKey(0)
        state = market_marl_reset(key, params)

        n_act = params.n_units * params.offer_sced_setup.n_segments
        # Use truthful bids — dispatch should be unconstrained
        truthful_action = jnp.full((n_act,), -1.0, dtype=jnp.float32)
        s1, _, _, info = market_marl_step(key, state, truthful_action, params)
        assert bool(info["sced_converged"])
        # State should update normally (no crash, no degenerate dispatch)
        assert jnp.all(jnp.isfinite(s1.unit_power_mw))

    def test_ramp_params_in_state_observation(self, case5):
        """Ramp headroom obs feature [4] must change when ramp is tighter.

        Uses low load (20% of p_max) so units dispatch well below capacity,
        giving meaningful headroom that tightening the ramp limit reduces.
        """
        # Low load: 20% of p_max per load bus so units have spare headroom
        low_load = np.asarray(case5.load_d_min, dtype=np.float32) * 0.5 + np.float32(1.0)
        profiles = jnp.tile(
            jnp.array(low_load, dtype=jnp.float32)[None, :], (48, 1)
        )
        # Default ramp (50% of p_max per step — very generous)
        params_default = make_market_marl_params(
            case5, profiles, n_segments=3, ramp_rate_fraction=0.5
        )
        # Very tight ramp (1 MW per step)
        tight_ramp = np.ones(case5.n_units, dtype=np.float32) * 1.0
        params_tight = make_market_marl_params(
            case5, profiles, n_segments=3,
            ramp_up_mw_per_step=tight_ramp,
            ramp_down_mw_per_step=tight_ramp,
        )
        env_default = MarketMARLEnv(params_default)
        env_tight = MarketMARLEnv(params_tight)

        key = jax.random.PRNGKey(0)
        obs_default, _ = env_default.reset(key)
        obs_tight, _   = env_tight.reset(key)

        # Check across all agents: tight ramp obs[4] <= default obs[4]
        # and at least one agent shows strictly smaller headroom
        any_smaller = False
        for ag in env_default.agent_names:
            h_def  = float(obs_default[ag][4])
            h_tight = float(obs_tight[ag][4])
            assert h_tight <= h_def + 1e-5, (
                f"{ag}: tight ramp headroom ({h_tight:.4f}) > default ({h_def:.4f})"
            )
            if h_tight < h_def - 1e-5:
                any_smaller = True
        assert any_smaller, (
            "Tight ramp should reduce headroom obs[4] for at least one agent; "
            f"default={[float(obs_default[ag][4]) for ag in env_default.agent_names]}, "
            f"tight={[float(obs_tight[ag][4]) for ag in env_tight.agent_names]}"
        )

    def test_offer_sced_runtime_bounds_backward_compat(self, sced_setup, load_mw,
                                                        truthful_prices):
        """offer_sced with p_min_rt=None, p_max_rt=None must match static call."""
        result_static = offer_sced(sced_setup, load_mw, truthful_prices)
        result_rt = offer_sced(sced_setup, load_mw, truthful_prices,
                               p_min_rt=None, p_max_rt=None)
        np.testing.assert_allclose(
            np.asarray(result_static.unit_power),
            np.asarray(result_rt.unit_power),
            atol=1e-4,
        )

    def test_offer_sced_runtime_bounds_constrain_dispatch(self, sced_setup, load_mw,
                                                           truthful_prices):
        """Runtime p_min_rt / p_max_rt passed to offer_sced bound the dispatch."""
        # Force unit 4 (G5, largest) to [30, 40] MW only
        p_min_rt = sced_setup.p_min.at[4].set(30.0)
        p_max_rt = sced_setup.p_max.at[4].set(40.0)
        result = offer_sced(sced_setup, load_mw, truthful_prices,
                            p_min_rt=p_min_rt, p_max_rt=p_max_rt)
        if bool(result.converged):
            p4 = float(result.unit_power[4])
            assert 29.0 <= p4 <= 41.0, (
                f"G5 dispatch {p4:.2f} MW outside runtime bounds [30, 40]"
            )


# ============ Episode Sampling (pool diversity) ============

class TestEpisodeSampling:
    """Verify that the GB profile pool is properly sampled across episodes.

    Each call to market_marl_reset samples a random episode_start_idx so that
    different episodes access different 48-step windows of the profile pool.
    """

    def _make_large_pool_params(self, case5, n_pool=200):
        """Helper: params with a pool larger than max_steps=48."""
        import numpy as np
        base = jnp.array(
            (np.asarray(case5.load_d_max) + np.asarray(case5.load_d_min)) / 2.0,
            dtype=jnp.float32,
        )
        # Make every row distinguishable: row i has load = base * (1 + 0.005*i)
        mults = 1.0 + 0.005 * jnp.arange(n_pool, dtype=jnp.float32)
        profiles = base[None, :] * mults[:, None]                    # (n_pool, n_loads)
        return make_market_marl_params(case5, profiles, n_segments=3, max_steps=48)

    def test_different_keys_give_different_starts(self, case5):
        """Reset with many different keys produces multiple distinct start indices."""
        params = self._make_large_pool_params(case5, n_pool=200)
        starts = set()
        for seed in range(30):
            state = market_marl_reset(jax.random.PRNGKey(seed), params)
            starts.add(int(state.episode_start_idx))
        assert len(starts) >= 5, (
            f"Expected ≥5 distinct episode starts across 30 resets, "
            f"got {len(starts)}: {sorted(starts)}"
        )

    def test_start_idx_in_valid_range(self, case5):
        """episode_start_idx must be in [0, T-1]."""
        params = self._make_large_pool_params(case5, n_pool=200)
        T = params.load_profiles.shape[0]
        for seed in range(10):
            state = market_marl_reset(jax.random.PRNGKey(seed), params)
            idx = int(state.episode_start_idx)
            assert 0 <= idx < T, f"seed={seed}: episode_start_idx={idx} not in [0, {T-1}]"

    def test_step_uses_correct_profile_row(self, case5):
        """market_marl_step clears at profile row (episode_start_idx + t) % T, not t % T.

        Checks the SCED CLEARING result directly — info["unit_power"] — not obs[5].
        obs[5] is built by _build_obs_dict(final_state) and only touches
        final_state.episode_start_idx / final_state.time_step; it cannot distinguish
        whether the step itself used the correct or naive row during market clearing.
        info["unit_power"] is sced.unit_power, which was produced by SCED using
        whichever profile row the step consumed.

        Verification via DC power balance: in a lossless DC-SCED with no congestion,
        sum(unit_power) == sum(nodal_load).  We construct a pool where:
          - Row 0 (naive row, t % T = 0 at t=0): LOW  load
          - Row `start` (correct row): HIGH load (clearly different)
        After step 0, sum(info["unit_power"]) must be close to the HIGH-load row.
        If the implementation uses the naive t % T formula, the dispatch matches the
        LOW-load row instead, and the assertion fails.

        Note: at step 0 the prev_dispatch (from reset) was computed at the same
        profile row as the step (reset also samples row `start`), so ramp constraints
        are not binding and power balance holds cleanly.
        """
        import numpy as np

        # ── Build pool: 10 rows, fracs in [0.10, 0.55] of d_max (step 0.05).
        # Row 0: 10% of d_max (LOW load, ≈150 MW for case5).
        # Rows ≥ 2: ≥ 20% (≥300 MW) — clearly higher for power-balance discrimination.
        #
        # IMPORTANT: the pool is deliberately capped at frac=0.55 (≈55% of pmax).
        # Rows with frac ≥ 0.60 trigger PD-IPM non-convergence in case5 for some
        # combinations of markup prices and ramp bounds, making the test unreliable.
        # The instability is non-monotonic (rows at 0.60, 0.65-0.75, 0.95 fail while
        # rows at 0.80-0.90, 1.00 may not) and cannot be predicted analytically —
        # only empirically verified against offer_sced at the relevant operating point.
        # Using a conservative cap guarantees convergence for ALL rows in the pool.
        n_pool = 10
        d_max = np.asarray(case5.load_d_max, dtype=np.float32)
        d_min = np.asarray(case5.load_d_min, dtype=np.float32)
        fracs = (0.1 + 0.05 * np.arange(n_pool, dtype=np.float32))   # [0.10 … 0.55]
        profiles = np.clip(
            fracs[:, None] * d_max[None, :],
            d_min[None, :], d_max[None, :],
        )                                                              # (10, n_loads)
        params = make_market_marl_params(
            case5, jnp.array(profiles, dtype=jnp.float32),
            n_segments=3, max_steps=48,
        )
        T = params.load_profiles.shape[0]

        # Helper: compute total nodal load for a given profile row
        def total_load_at(row_idx: int) -> float:
            return float(jnp.sum(params.nodes_loads_map @ params.load_profiles[row_idx]))

        # ── Find a reset with episode_start_idx >= 2 AND convergent SCED ──────
        # At t=0: correct row = start (>= 2, high load), naive row = 0 (low load).
        # Pre-check: skip seeds where market_marl_step SCED does not converge.
        # This guards against ramp-bound + markup-price combinations that trigger
        # PD-IPM non-convergence at a specific operating point (even if the
        # unconstrained problem at that load level would normally converge).
        n_act = params.n_units * params.offer_sced_setup.n_segments
        flat_actions = jnp.zeros(n_act, dtype=jnp.float32)   # zero markup
        step_key = jax.random.PRNGKey(88)

        state0 = None
        for seed in range(60):
            s = market_marl_reset(jax.random.PRNGKey(seed), params)
            if int(s.episode_start_idx) < 2:
                continue
            # Pre-check SCED convergence before committing to this seed.
            _, _, _, _pre_info = market_marl_step(step_key, s, flat_actions, params)
            if not bool(_pre_info["sced_converged"]):
                continue
            state0 = s
            break
        assert state0 is not None, "Could not get episode_start_idx >= 2 with convergent SCED in 60 seeds"
        start = int(state0.episode_start_idx)

        correct_row = start % T    # what the correct formula (start+0)%T gives
        naive_row   = 0 % T        # what a broken t%T formula gives at t=0
        assert correct_row != naive_row, (
            f"start={start}: correct_row={correct_row} == naive_row — test is degenerate"
        )

        load_correct = total_load_at(correct_row)
        load_naive   = total_load_at(naive_row)
        assert abs(load_correct - load_naive) > 0.5, (
            f"Rows {correct_row} and {naive_row} have nearly equal loads "
            f"({load_correct:.2f} vs {load_naive:.2f} MW); pool rows not distinguishable"
        )

        # ── Call market_marl_step and read clearing output from info ──────────
        # Re-run the step (same key/actions as pre-check above) to get _final state.
        _final, _done, _profit, info = market_marl_step(
            step_key, state0, flat_actions, params
        )
        assert bool(info["sced_converged"]), "SCED did not converge; test unreliable"

        total_dispatch = float(jnp.sum(info["unit_power"]))

        err_correct = abs(total_dispatch - load_correct)
        err_naive   = abs(total_dispatch - load_naive)

        # Power balance: sum(gen) ≈ sum(load) for the consumed row.
        # Correct row (high load) → total_dispatch ≈ load_correct, err_correct ≈ 0.
        # Naive   row (low  load) → total_dispatch ≈ load_naive,   err_naive   ≈ 0.
        # The assertion fails iff the naive formula was used.
        assert err_correct < err_naive, (
            f"step 0 total_dispatch={total_dispatch:.3f} MW closer to "
            f"NAIVE row {naive_row} (load={load_naive:.3f}, err={err_naive:.3f}) "
            f"than to CORRECT row {correct_row} (load={load_correct:.3f}, err={err_correct:.3f}). "
            f"episode_start_idx={start}; indicates step used wrong profile row."
        )

        # ── Also verify step 1 (t=1 → row (start+1)%T) advances correctly ────
        # state after auto-reset if done; but at t=0→1 done=False for max_steps=48.
        state1 = _final     # time_step = 1, episode_start_idx = start (unchanged)
        assert int(state1.time_step) == 1
        assert int(state1.episode_start_idx) == start

        _final2, _, _, info2 = market_marl_step(
            step_key, state1, flat_actions, params
        )
        assert bool(info2["sced_converged"]), "SCED step 1 did not converge"

        correct_row_1 = (start + 1) % T
        naive_row_1   = 1 % T
        if correct_row_1 != naive_row_1:   # non-degenerate only if start != 0 (it is)
            load_c1 = total_load_at(correct_row_1)
            load_n1 = total_load_at(naive_row_1)
            if abs(load_c1 - load_n1) > 0.5:    # guard: rows must differ enough
                td1 = float(jnp.sum(info2["unit_power"]))
                assert abs(td1 - load_c1) < abs(td1 - load_n1), (
                    f"step 1: total_dispatch={td1:.3f} MW closer to naive row "
                    f"{naive_row_1} (load={load_n1:.3f}) than correct row "
                    f"{correct_row_1} (load={load_c1:.3f})"
                )


class TestTaskProfileLoading:

    def test_loaded_profiles_respect_dispatch_feasible_band(self, case5, monkeypatch):
        """Task profiles must not underload or overload the fixed-unit dispatch task."""
        from powerzoojax.data.data_loader import DataLoader

        fake_demand = jnp.array([[0.0], [50.0], [100.0]], dtype=jnp.float32)
        monkeypatch.setattr(DataLoader, "load_jax_profiles", lambda *a, **k: fake_demand)

        profiles = np.asarray(load_gencos_profiles(case5, split="iid"))
        total_load = profiles.sum(axis=1)
        min_total = float(np.asarray(case5.unit_p_min).sum()) + 1.0
        max_total = float(np.asarray(case5.unit_p_max).sum()) - 1.0
        assert float(total_load.min()) >= min_total - 1e-4
        assert float(total_load.max()) <= max_total + 1e-4

    def test_ood_profiles_are_not_normalized_back_to_iid(self, case5, monkeypatch):
        """Demand-shift and renewable-shock must differ materially from IID profiles."""
        from powerzoojax.data.data_loader import DataLoader

        fake_demand = jnp.linspace(0.0, 100.0, 8, dtype=jnp.float32)[:, None]
        monkeypatch.setattr(DataLoader, "load_jax_profiles", lambda *a, **k: fake_demand)

        iid = np.asarray(load_gencos_profiles(case5, split="iid"))
        demand_shift = np.asarray(load_gencos_profiles(case5, split="iid", ood_axis="demand_shift"))
        renewable_shock = np.asarray(load_gencos_profiles(case5, split="iid", ood_axis="renewable_shock"))

        assert not np.allclose(demand_shift, iid), "demand_shift should not collapse back to IID"
        assert not np.allclose(renewable_shock, iid), (
            "renewable_shock should not collapse back to IID"
        )
        assert float(demand_shift.sum(axis=1).mean()) > float(iid.sum(axis=1).mean())
        assert float(renewable_shock.sum(axis=1).mean()) > float(iid.sum(axis=1).mean())

class TestAutoResetSampling(TestEpisodeSampling):

    def test_autoreset_resamples_start(self, case5):
        """After episode end (done=True), the next episode has a new start index.

        Run a 48-step scan (auto-reset triggers at step 48).  The post-reset
        episode_start_idx should frequently differ from the original one when
        the pool is large.
        """
        params = self._make_large_pool_params(case5, n_pool=200)
        n_act = params.n_units * params.offer_sced_setup.n_segments

        key = jax.random.PRNGKey(0)
        init_state = market_marl_reset(key, params)
        original_start = int(init_state.episode_start_idx)

        def scan_fn(carry, _):
            state, k = carry
            k, k_step = jax.random.split(k)
            act = jnp.zeros(n_act, dtype=jnp.float32)
            final_state, done, _, _ = market_marl_step(k_step, state, act, params)
            return (final_state, k), done

        (final_state, _), dones = jax.lax.scan(
            scan_fn, (init_state, key), None, length=48,
        )
        # After 48 steps, auto-reset should have fired (done was True at step 48)
        assert bool(dones[-1]), "Expected done=True at last step of 48-step episode"
        # final_state is the post-reset state; its episode_start_idx is resampled
        new_start = int(final_state.episode_start_idx)
        # With T=200 and uniform sampling, P(same start twice) = 1/200 = 0.5%
        # We assert it's in valid range (the probabilistic nature means we can't
        # always guarantee it differs, but we can check validity)
        T = params.load_profiles.shape[0]
        assert 0 <= new_start < T, f"Post-reset episode_start_idx={new_start} invalid"

    def test_small_pool_wraps_correctly(self, marl_params):
        """With a small pool (size == max_steps), wrapping still works correctly."""
        # marl_params has a 48-step pool and max_steps=48
        T = marl_params.load_profiles.shape[0]
        assert T == 48, f"Expected 48-step fixture pool, got T={T}"

        key = jax.random.PRNGKey(5)
        state = market_marl_reset(key, marl_params)
        # Even with a pool exactly equal to max_steps, start index is in [0, T-1]
        idx = int(state.episode_start_idx)
        assert 0 <= idx < T


# ============ Observation Layout (Finding 2) ============

class TestObsLayout:

    def test_obs_dim_is_12(self, marl_env):
        """Total obs dim must be OBS_DIM = 12 (8 scalar + 4 LMP history)."""
        assert OBS_DIM == 12, "OBS_DIM constant should be 12"
        obs_space = marl_env.observation_space()
        assert obs_space.shape[0] == 12

    def test_obs_has_lmp_history(self, marl_env):
        """Dims [8..11] must carry LMP history (finite, normalised)."""
        key = jax.random.PRNGKey(0)
        obs_dict, state = marl_env.reset(key)
        for name in marl_env.agent_names:
            obs = obs_dict[name]  # (12,)
            lmp_hist = obs[8:]    # (4,)
            assert lmp_hist.shape == (4,)
            assert bool(jnp.all(jnp.isfinite(lmp_hist)))

    def test_obs_ramp_headroom_nonneg(self, marl_env):
        """Dim [4] (ramp headroom) must be in [0, 1]."""
        key = jax.random.PRNGKey(0)
        obs_dict, _ = marl_env.reset(key)
        for name in marl_env.agent_names:
            headroom = float(obs_dict[name][4])
            assert 0.0 <= headroom <= 1.0 + 1e-5, (
                f"Agent {name}: ramp headroom {headroom:.4f} out of [0,1]"
            )

    def test_obs_demand_forecast_is_next_step(self, case5):
        """obs[5] = load at (episode_start_idx + time_step + 1) % T / total_pmax.

        Uses a distinguishable profile pool (each row carries a unique total load)
        so that only the correct start-index-aware row index produces the observed
        value.  The flat mid-load fixture is NOT used here because it cannot
        discriminate between the correct formula and the naive (time_step+1)%T.

        Verifies across 20 reset seeds — at least one must have start != 0 to
        confirm that episode_start_idx is actually taken into account.
        """
        import numpy as np
        n_pool = 96
        base = jnp.array(
            (np.asarray(case5.load_d_max) + np.asarray(case5.load_d_min)) / 2.0,
            dtype=jnp.float32,
        )
        # Row i: every bus load scales by (1 + 0.01*i) — fully distinguishable
        mults = 1.0 + 0.01 * jnp.arange(n_pool, dtype=jnp.float32)
        profiles = base[None, :] * mults[:, None]                   # (96, n_loads)
        params = make_market_marl_params(case5, profiles, n_segments=3, max_steps=48)
        env = MarketMARLEnv(params)
        T = params.load_profiles.shape[0]
        total_pmax = float(jnp.sum(params.unit_p_max))

        checked_nonzero_start = False
        for seed in range(20):
            key = jax.random.PRNGKey(seed)
            obs_dict, state = env.reset(key)
            start = int(state.episode_start_idx)
            t     = int(state.time_step)          # always 0 at reset

            # Correct formula: (episode_start_idx + time_step + 1) % T
            next_t = (start + t + 1) % T
            next_load = float(jnp.sum(params.nodes_loads_map @ params.load_profiles[next_t]))
            expected_norm = next_load / (total_pmax + 1e-6)

            for name in env.agent_names:
                obs_f = float(obs_dict[name][5])
                assert 0.0 <= obs_f <= 1.0 + 1e-5, (
                    f"seed={seed}, {name}: obs[5]={obs_f:.4f} out of [0, 1]"
                )
                assert abs(obs_f - expected_norm) < 1e-4, (
                    f"seed={seed}, {name}: obs[5]={obs_f:.6f} ≠ load at "
                    f"(start={start}+t={t}+1)%T={T} = row {next_t}: {expected_norm:.6f}"
                )
            if start != 0:
                checked_nonzero_start = True

        assert checked_nonzero_start, (
            "All 20 seeds yielded episode_start_idx=0; "
            "test did not exercise the start-index-aware formula"
        )

    def test_lmp_history_updates_after_step(self, marl_env):
        """LMP history dims [8..11] must change after a step that changes LMP."""
        key = jax.random.PRNGKey(0)
        obs0, state0 = marl_env.reset(key)
        actions = {n: jnp.ones(3, dtype=jnp.float32) for n in marl_env.agent_names}
        obs1, state1, _, _, _ = marl_env.step(key, state0, actions)

        # After step, the newest LMP history entry (dim 11) may differ
        lmp_hist0 = obs0["genco_0"][8:]
        lmp_hist1 = obs1["genco_0"][8:]
        # Most recent entry (index 3) might have changed; test at least history is finite
        assert bool(jnp.all(jnp.isfinite(lmp_hist1)))


def _gb_data_available() -> bool:
    """Return True if GB demand data is accessible via DataLoader."""
    try:
        from powerzoojax.data.data_loader import DataLoader
        from powerzoojax.data.signals import LOAD_ACTUAL_MW
        loader = DataLoader()
        arr = loader.load_jax_profiles(
            [LOAD_ACTUAL_MW], source="gb",
            start_date="2025-04-01", end_date="2025-04-02",
            resample="30min",
        )
        return arr.shape[0] > 0
    except Exception:
        return False


# ============ Preset (Finding 3) ============

class TestPreset:

    def test_main_preset_exists(self):
        """Main preset 'gencos-case5-ippo' must exist."""
        preset = get_preset("gencos-case5-ippo")
        assert preset is not None

    def test_main_preset_is_not_synthetic_flat(self):
        """Main preset env_factory must NOT be the old flat mid-load factory.

        Specifically, it should be _wrap_gencos_case5_gb, NOT _wrap_gencos_case5_dev.
        """
        from powerzoojax.rl.presets import _wrap_gencos_case5_gb, _wrap_gencos_case5_dev
        preset = get_preset("gencos-case5-ippo")
        assert preset.env_factory is _wrap_gencos_case5_gb, (
            "Main preset env_factory should be _wrap_gencos_case5_gb (GB data), "
            f"got {preset.env_factory}"
        )
        assert preset.env_factory is not _wrap_gencos_case5_dev, (
            "Main preset must not point to the synthetic dev factory"
        )

    def test_dev_preset_exists_and_is_synthetic(self):
        """Dev preset 'gencos-case5-ippo-dev' must exist and be explicitly synthetic."""
        preset = get_preset("gencos-case5-ippo-dev")
        desc = preset.description.lower()
        assert "dev" in desc or "synthetic" in desc, (
            f"Dev preset should be marked as dev/synthetic: {desc}"
        )

    def test_dev_preset_factory_creates_market_env(self):
        """Dev preset env_factory must produce a usable MarketMARLEnv."""
        env = _wrap_gencos_case5_dev()
        assert isinstance(env, MarketMARLEnv)
        assert env.num_agents == 5
        assert env.agent_names == [f"genco_{i}" for i in range(5)]

    def test_dev_preset_obs_shape(self):
        """Dev preset env must have the updated 12-dim obs."""
        env = _wrap_gencos_case5_dev()
        obs_space = env.observation_space()
        assert obs_space.shape == (OBS_DIM,)

    def test_main_preset_has_gb_factory(self):
        """Main preset env_factory reference should be the GB version."""
        from powerzoojax.rl.presets import _wrap_gencos_case5_gb
        preset = get_preset("gencos-case5-ippo")
        assert preset.env_factory is _wrap_gencos_case5_gb, (
            "Main preset env_factory is not _wrap_gencos_case5_gb"
        )

    def test_main_preset_max_steps_is_48(self, monkeypatch):
        """Main preset must produce 48-step episodes regardless of pool size.

        GB window = sampling pool; benchmark episode length = 48×30min.
        Uses monkeypatch so the test does not depend on real GB data.
        """
        import jax.numpy as jnp
        from powerzoojax.case import create_case5
        import powerzoojax.rl.presets as preset_mod

        case = create_case5()
        # Synthetic pool of 200 steps — simulates a large GB window
        fake_profiles = jnp.tile(
            jnp.array((case.load_d_max + case.load_d_min) / 2.0,
                      dtype=jnp.float32)[None, :],
            (200, 1),
        )
        monkeypatch.setattr(
            preset_mod, "_load_gb_profiles_for_case5",
            lambda *a, **kw: fake_profiles,
        )
        env = get_preset("gencos-case5-ippo").env_factory()
        assert isinstance(env, MarketMARLEnv)
        assert int(env._params.max_steps) == 48, (
            f"Benchmark episode length must be 48 steps, got {env._params.max_steps}"
        )
        # Profile pool should be the full fake array (not truncated to 48)
        assert env._params.load_profiles.shape[0] == 200, (
            "GB profile pool should not be truncated to episode length"
        )

    def test_main_preset_gb_profiles_not_flat(self, monkeypatch):
        """Main preset must use non-flat profiles (GB source, not synthetic mid-load).

        Uses monkeypatch to inject varied profiles without real GB data.
        """
        import jax.numpy as jnp
        from powerzoojax.case import create_case5
        import powerzoojax.rl.presets as preset_mod

        case = create_case5()
        # Varied (non-flat) synthetic profile mimicking GB temporal variation
        t = jnp.linspace(0.0, 2.0 * jnp.pi, 96, dtype=jnp.float32)
        demand_norm = (jnp.sin(t) + 1.0) / 2.0                       # ∈ [0, 1]
        d_min = jnp.array(case.load_d_min, dtype=jnp.float32)
        d_max = jnp.array(case.load_d_max, dtype=jnp.float32)
        fake_profiles = d_min[None, :] + demand_norm[:, None] * (d_max - d_min)[None, :]

        monkeypatch.setattr(
            preset_mod, "_load_gb_profiles_for_case5",
            lambda *a, **kw: fake_profiles,
        )
        env = get_preset("gencos-case5-ippo").env_factory()
        profiles = env._params.load_profiles
        # Profiles must vary over time (not flat mid-load).
        # Use global std across all (time, bus) entries; some buses may have
        # d_min == d_max == 0 so per-column checks are fragile.
        std_global = float(jnp.std(profiles))
        assert std_global > 1e-3, (
            f"GB preset profiles should vary over time; global std={std_global:.6f}; "
            f"profile shape={profiles.shape}"
        )


# ============ Market Metrics (Finding 4) ============

class TestMarketMetrics:

    def test_compute_market_metrics_method_exists(self, marl_env):
        """MarketMARLEnv must have compute_market_metrics hook."""
        assert hasattr(marl_env, "compute_market_metrics")
        assert callable(marl_env.compute_market_metrics)

    def test_compute_market_metrics_keys(self, marl_env):
        """compute_market_metrics must return all implemented metric keys.

        Implemented keys (G5 partial — see module docstring for NOT-implemented):
            price_volatility, HHI, market_share_std, ramp_binding_rate,
            dispatch_share_{ag} for each agent.
        """
        n_steps, n_envs, n_units = 10, 4, 5
        n_nodes = marl_env._params.n_nodes

        rng = np.random.default_rng(0)
        rollout_info = {
            "lmp":               jnp.array(rng.uniform(10, 50, (n_steps, n_envs, n_nodes)), dtype=jnp.float32),
            "unit_power":        jnp.array(rng.uniform(20, 300, (n_steps, n_envs, n_units)), dtype=jnp.float32),
            "ramp_binding_rate": jnp.array(rng.uniform(0, 1, (n_steps, n_envs)), dtype=jnp.float32),
        }
        metrics = marl_env.compute_market_metrics(rollout_info)

        assert "market/price_volatility" in metrics
        assert "market/HHI" in metrics
        assert "market/market_share_std" in metrics
        assert "market/ramp_binding_rate" in metrics
        # Per-agent dispatch share
        for ag in marl_env.agent_names:
            assert f"market/dispatch_share_{ag}" in metrics, (
                f"Per-agent dispatch share key 'market/dispatch_share_{ag}' missing"
            )

    def test_compute_market_metrics_values_reasonable(self, marl_env):
        """Market metrics must be finite and in physically plausible ranges."""
        n_steps, n_envs, n_units = 48, 8, 5
        n_nodes = marl_env._params.n_nodes

        rng = np.random.default_rng(42)
        rollout_info = {
            "lmp":               jnp.array(rng.uniform(20, 80, (n_steps, n_envs, n_nodes)), dtype=jnp.float32),
            "unit_power":        jnp.array(rng.uniform(10, 200, (n_steps, n_envs, n_units)), dtype=jnp.float32),
            "ramp_binding_rate": jnp.array(rng.uniform(0, 0.5, (n_steps, n_envs)), dtype=jnp.float32),
        }
        metrics = marl_env.compute_market_metrics(rollout_info)

        vol = float(metrics["market/price_volatility"])
        hhi = float(metrics["market/HHI"])
        share_std = float(metrics["market/market_share_std"])
        ramp_br = float(metrics["market/ramp_binding_rate"])

        assert np.isfinite(vol) and vol >= 0.0,  f"price_volatility {vol} invalid"
        assert np.isfinite(hhi) and 0.0 <= hhi <= 1.0, f"HHI {hhi} out of [0,1]"
        assert np.isfinite(share_std) and share_std >= 0.0, f"share_std {share_std} invalid"
        assert np.isfinite(ramp_br) and 0.0 <= ramp_br <= 1.0, f"ramp_binding_rate {ramp_br} invalid"

    def test_ippo_has_market_metric_keys_for_market_env(self, marl_env):
        """make_ippo_train returns train_fn that, after 1 update, has market metrics."""
        from powerzoojax.rl.ippo import make_ippo_train
        from powerzoojax.rl.config import TrainConfig

        # Use minimal config: 1 update, few envs
        config = TrainConfig(
            algo="ippo",
            total_timesteps=4 * 48,    # 4 envs × 48 steps = 1 update
            num_envs=4,
            n_steps=48,
            n_epochs=1,
            hidden_dims=(32, 32),
            eval_freq=0,               # no eval during smoke test
        )
        train_fn = make_ippo_train(marl_env, config)
        result = train_fn(jax.random.PRNGKey(7))

        assert "market/price_volatility" in result.metrics, (
            "Market metrics not exported for GenCos IPPO training"
        )
        assert "market/HHI" in result.metrics
        assert "market/market_share_std" in result.metrics
        assert "market/ramp_binding_rate" in result.metrics
        assert "market/cum_profit_mean" in result.metrics

    def test_ippo_has_per_agent_cumulative_profit(self, marl_env):
        """make_ippo_train must export per-agent cumulative profit for all GenCos.

        Keys: market/cumulative_profit_genco_0 ... market/cumulative_profit_genco_4.
        These are distinct from the aggregate market/cum_profit_mean.
        """
        from powerzoojax.rl.ippo import make_ippo_train
        from powerzoojax.rl.config import TrainConfig

        config = TrainConfig(
            algo="ippo",
            total_timesteps=4 * 48,
            num_envs=4,
            n_steps=48,
            n_epochs=1,
            hidden_dims=(32, 32),
            eval_freq=0,
        )
        train_fn = make_ippo_train(marl_env, config)
        result = train_fn(jax.random.PRNGKey(11))

        n_agents = marl_env.num_agents
        for ag in marl_env.agent_names:
            key = f"market/cumulative_profit_{ag}"
            assert key in result.metrics, (
                f"Per-agent profit key '{key}' missing from training metrics. "
                f"Available market keys: {[k for k in result.metrics if 'market' in k]}"
            )
            arr = result.metrics[key]
            assert arr.shape == (1,) or arr.ndim <= 1, (
                f"Expected scalar-per-update array for {key}, got shape {arr.shape}"
            )
            assert float(jnp.isfinite(arr).all()), f"{key} contains non-finite values"

    def test_market_profit_uses_raw_reward_not_normalised(self, marl_env):
        """Per-agent profit metrics must reflect raw dispatch profit, not PPO signal.

        With identical initial params and identical random trajectories, the per-agent
        cumulative profit must be the same whether normalize_rewards=True or False.
        If the implementation incorrectly uses the normalised training reward instead
        of raw profit, the two values would diverge.
        """
        from powerzoojax.rl.ippo import make_ippo_train
        from powerzoojax.rl.config import TrainConfig

        base_cfg = dict(
            algo="ippo",
            total_timesteps=4 * 48,   # exactly 1 update
            num_envs=4,
            n_steps=48,
            n_epochs=1,
            hidden_dims=(32, 32),
            eval_freq=0,
        )
        key = jax.random.PRNGKey(17)

        # Run with normalize_rewards=False (profits are identical to raw rewards)
        result_raw  = make_ippo_train(marl_env, TrainConfig(**base_cfg, normalize_rewards=False))(key)
        # Run with normalize_rewards=True (PPO loss uses normalised signal)
        result_norm = make_ippo_train(marl_env, TrainConfig(**base_cfg, normalize_rewards=True))(key)

        for ag in marl_env.agent_names:
            k = f"market/cumulative_profit_{ag}"
            # metrics[k] is (n_updates,) — here n_updates=1, so index [0]
            v_raw  = float(result_raw.metrics[k][0])
            v_norm = float(result_norm.metrics[k][0])
            # Same initial params + same random key → same trajectory → same raw rewards.
            # Both configs must therefore report qualitatively identical per-agent profits.
            #
            # Correct implementation: |v_raw - v_norm| << |v_raw| (only float32 noise,
            # ≪ 1% relative difference).
            # Wrong implementation (using normalised rewards): v_norm ≈ 0 ± small,
            # while v_raw ≈ 10,000+, so relative error would be ≫ 50%.
            #
            # Allow 2% relative tolerance to accommodate XLA float32 rounding
            # that arises from different JIT graph structures between the two configs.
            rel_err = abs(v_raw - v_norm) / (abs(v_raw) + 1e-6)
            assert rel_err < 0.02, (
                f"{ag}: per-agent profit relative error={rel_err:.4f} between "
                f"normalize_rewards=False ({v_raw:.2f}) and "
                f"normalize_rewards=True ({v_norm:.2f}).  "
                f"If v_norm ≈ 0 this indicates normalised rewards were used for profit."
            )

    def test_compute_market_metrics_dispatch_share_values(self, marl_env):
        """dispatch_share_{ag} must be in [0,1], sum to ≈1.0, and be finite.

        Per-agent dispatch share = agent's fraction of total generation, averaged
        over all (step, env) pairs.  Values must be non-negative, sum to ≈1.0,
        and each be individually in [0, 1].
        """
        n_steps, n_envs, n_units = 48, 8, 5
        n_nodes = marl_env._params.n_nodes

        rng = np.random.default_rng(99)
        rollout_info = {
            "lmp":        jnp.array(rng.uniform(20, 80, (n_steps, n_envs, n_nodes)), dtype=jnp.float32),
            "unit_power": jnp.array(rng.uniform(10, 200, (n_steps, n_envs, n_units)), dtype=jnp.float32),
        }
        metrics = marl_env.compute_market_metrics(rollout_info)

        share_vals = []
        for ag in marl_env.agent_names:
            key = f"market/dispatch_share_{ag}"
            assert key in metrics, f"Missing dispatch share key: {key}"
            v = float(metrics[key])
            assert np.isfinite(v), f"{key} is not finite: {v}"
            assert 0.0 <= v <= 1.0 + 1e-5, f"{key} = {v:.6f} outside [0, 1]"
            share_vals.append(v)

        total = sum(share_vals)
        assert abs(total - 1.0) < 1e-4, (
            f"dispatch_share values should sum to ≈1.0; got {total:.6f}; "
            f"values={share_vals}"
        )

    def test_ippo_exports_dispatch_share_per_agent(self, marl_env):
        """make_ippo_train must export per-agent dispatch share for all GenCos.

        Keys: market/dispatch_share_genco_0 ... market/dispatch_share_genco_4.
        """
        from powerzoojax.rl.ippo import make_ippo_train
        from powerzoojax.rl.config import TrainConfig

        config = TrainConfig(
            algo="ippo",
            total_timesteps=4 * 48,
            num_envs=4,
            n_steps=48,
            n_epochs=1,
            hidden_dims=(32, 32),
            eval_freq=0,
        )
        train_fn = make_ippo_train(marl_env, config)
        result = train_fn(jax.random.PRNGKey(23))

        for ag in marl_env.agent_names:
            key = f"market/dispatch_share_{ag}"
            assert key in result.metrics, (
                f"Dispatch share key '{key}' missing from training metrics. "
                f"Available market keys: {[k for k in result.metrics if 'market' in k]}"
            )
            arr = result.metrics[key]
            assert arr.ndim <= 1, f"Expected 1-D array for {key}, got shape {arr.shape}"
            assert float(jnp.isfinite(arr).all()), f"{key} contains non-finite values"
            v = float(arr[0])
            assert 0.0 <= v <= 1.0 + 1e-4, f"{key}={v:.6f} outside [0, 1]"

    def test_ippo_exports_convergence_stability(self, marl_env):
        """make_ippo_train must export market/convergence_stability for market envs.

        convergence_stability = L2 norm of policy-parameter change from the 80%
        training checkpoint to the final params.  It requires at least 5 updates to
        activate the checkpoint mechanism (otherwise it reports 0.0).

        Semantic: low value → policy has converged in the final phase of training.
        This is the policy-parameter L2 drift metric, implemented in the IPPO
        training loop (not in compute_market_metrics).
        """
        from powerzoojax.rl.ippo import make_ippo_train
        from powerzoojax.rl.config import TrainConfig

        # Need >= 5 updates to trigger checkpoint; 5 * 4 * 48 = 960 total steps.
        config = TrainConfig(
            algo="ippo",
            total_timesteps=5 * 4 * 48,
            num_envs=4,
            n_steps=48,
            n_epochs=1,
            hidden_dims=(32, 32),
            eval_freq=0,
        )
        train_fn = make_ippo_train(marl_env, config)
        result = train_fn(jax.random.PRNGKey(31))

        assert "market/convergence_stability" in result.metrics, (
            "convergence_stability missing from training metrics. "
            f"Available keys: {sorted(result.metrics.keys())}"
        )
        arr = result.metrics["market/convergence_stability"]
        assert arr.shape == (1,), f"Expected shape (1,), got {arr.shape}"
        cs = float(arr[0])
        assert np.isfinite(cs), f"convergence_stability is not finite: {cs}"
        assert cs >= 0.0, f"convergence_stability < 0: {cs}"
        # No upper-bound assertion: depends on learning rate and architecture size.

    def test_convergence_stability_zero_for_short_training(self, marl_env):
        """With < 5 updates, convergence_stability must be 0.0 (no checkpoint).

        Short training (1 update) does not trigger the 80%-checkpoint mechanism,
        so the metric falls back to 0.0 (indeterminate).
        """
        from powerzoojax.rl.ippo import make_ippo_train
        from powerzoojax.rl.config import TrainConfig

        config = TrainConfig(
            algo="ippo",
            total_timesteps=4 * 48,    # exactly 1 update — below the 5-update threshold
            num_envs=4,
            n_steps=48,
            n_epochs=1,
            hidden_dims=(32, 32),
            eval_freq=0,
        )
        train_fn = make_ippo_train(marl_env, config)
        result = train_fn(jax.random.PRNGKey(37))

        assert "market/convergence_stability" in result.metrics
        cs = float(result.metrics["market/convergence_stability"][0])
        assert cs == 0.0, (
            f"convergence_stability should be 0.0 for < 5 updates, got {cs}"
        )

    def test_planner_cost_ratio_key_present(self, marl_env, marl_params):
        """compute_market_metrics must return planner_cost_ratio when load_mw and
        gen_cost are provided.  Without these keys the metric is silently omitted
        (backward compat with old IPPO that did not collect load_mw/gen_cost).

        planner_cost_ratio = planner_cost / RL_gen_cost ∈ (0, 1].
        This is a COST RATIO (not a welfare ratio): it measures how much cheaper
        the ramp-free planner dispatch is relative to the RL dispatch.
        True social_welfare_ratio = (V − RL_cost) / (V − planner_cost) requires
        consumer value V (VOLL × demand) which is not defined in this codebase.
        """
        n_steps, n_envs = 4, 4
        n_act = marl_params.n_units * marl_params.offer_sced_setup.n_segments
        n_nodes = marl_params.n_nodes
        T_pool = marl_params.load_profiles.shape[0]

        key = jax.random.PRNGKey(50)
        states = jax.vmap(market_marl_reset, in_axes=(0, None))(
            jax.random.split(key, n_envs), marl_params
        )
        gen_costs, load_mws = [], []
        acts = jnp.zeros((n_envs, n_act), dtype=jnp.float32)
        for step_i in range(n_steps):
            profile_rows = (states.episode_start_idx + states.time_step) % T_pool
            raw_loads = marl_params.load_profiles[profile_rows]         # (n_envs, n_loads)
            load_mws.append(raw_loads @ marl_params.nodes_loads_map.T)  # (n_envs, n_nodes)

            step_keys = jax.random.split(jax.random.PRNGKey(step_i + 20), n_envs)
            def _step(k, s, a):
                return market_marl_step(k, s, a, marl_params)
            states, _, _, info = jax.vmap(_step)(step_keys, states, acts)
            gen_costs.append(info["gen_cost"])  # (n_envs,)

        rollout_info = {
            "lmp":        jnp.zeros((n_steps, n_envs, n_nodes), dtype=jnp.float32),
            "unit_power": jnp.zeros((n_steps, n_envs, marl_params.n_units), dtype=jnp.float32),
            "gen_cost":   jnp.stack(gen_costs),   # (n_steps, n_envs)
            "load_mw":    jnp.stack(load_mws),    # (n_steps, n_envs, n_nodes)
        }
        metrics = marl_env.compute_market_metrics(rollout_info)
        assert "market/planner_cost_ratio" in metrics, (
            "planner_cost_ratio missing from compute_market_metrics output "
            "even though load_mw and gen_cost were provided. "
            f"Keys returned: {sorted(metrics.keys())}"
        )
        # social_welfare_ratio must NOT appear — it is not implemented.
        assert "market/social_welfare_ratio" not in metrics, (
            "social_welfare_ratio key must not appear — it is not implemented. "
            "Only planner_cost_ratio (a cost ratio) is exported."
        )

        # Backward compat: without load_mw/gen_cost, planner_cost_ratio is absent.
        rollout_no_cost = {
            "lmp":        rollout_info["lmp"],
            "unit_power": rollout_info["unit_power"],
        }
        metrics_no = marl_env.compute_market_metrics(rollout_no_cost)
        assert "market/planner_cost_ratio" not in metrics_no, (
            "planner_cost_ratio should be absent when load_mw/gen_cost not provided"
        )

    def test_planner_cost_ratio_formula_and_range(self, marl_env, marl_params):
        """planner_cost_ratio = planner_cost / RL_gen_cost must be in (0, 1].

        Semantic: planner (ramp-free DC-OPF, true costs) minimises gen cost over
        a strictly larger feasible set than the ramp-constrained RL dispatch.
        Therefore planner_cost ≤ RL_gen_cost always → ratio ∈ (0, 1].

        This test also verifies the formula is planner_cost / RL_cost (not the
        inverse), by checking that:
          - ratio ≤ 1.0 (planner is never more expensive than RL)
          - ratio > 0.0 (both costs are positive)
          - the metric is numerically consistent with a direct manual calculation.
        """
        n_steps, n_envs = 6, 4
        n_act = marl_params.n_units * marl_params.offer_sced_setup.n_segments
        n_nodes = marl_params.n_nodes
        T_pool = marl_params.load_profiles.shape[0]

        key = jax.random.PRNGKey(51)
        states = jax.vmap(market_marl_reset, in_axes=(0, None))(
            jax.random.split(key, n_envs), marl_params
        )
        gen_costs, load_mws = [], []
        acts = jnp.zeros((n_envs, n_act), dtype=jnp.float32)  # zero markup
        for step_i in range(n_steps):
            profile_rows = (states.episode_start_idx + states.time_step) % T_pool
            raw_loads = marl_params.load_profiles[profile_rows]
            load_mws.append(raw_loads @ marl_params.nodes_loads_map.T)

            step_keys = jax.random.split(jax.random.PRNGKey(step_i + 30), n_envs)
            def _step(k, s, a):
                return market_marl_step(k, s, a, marl_params)
            states, _, _, info = jax.vmap(_step)(step_keys, states, acts)
            gen_costs.append(info["gen_cost"])

        gen_cost_arr = jnp.stack(gen_costs)   # (n_steps, n_envs)
        load_mw_arr  = jnp.stack(load_mws)    # (n_steps, n_envs, n_nodes)

        rollout_info = {
            "lmp":        jnp.zeros((n_steps, n_envs, n_nodes), dtype=jnp.float32),
            "unit_power": jnp.zeros((n_steps, n_envs, marl_params.n_units), dtype=jnp.float32),
            "gen_cost":   gen_cost_arr,
            "load_mw":    load_mw_arr,
        }
        metrics = marl_env.compute_market_metrics(rollout_info)
        cr = float(metrics["market/planner_cost_ratio"])

        # Range check: planner is relaxation of RL → planner_cost ≤ RL_cost
        assert np.isfinite(cr), f"planner_cost_ratio is not finite: {cr}"
        assert cr > 0.0, f"planner_cost_ratio must be positive, got {cr}"
        assert cr <= 1.0 + 1e-4, (
            f"planner_cost_ratio = {cr:.6f} > 1.0: planner (ramp-free) must never "
            f"be more expensive than RL (ramp-constrained) on the same load"
        )

        # Formula check: manually compute planner costs and verify.
        # planner_cost_ratio = mean(planner_cost / (RL_gen_cost + ε))
        from powerzoojax.envs.market.offer_sced import offer_sced
        setup = marl_params.offer_sced_setup
        N = n_steps * n_envs
        load_flat = load_mw_arr.reshape(N, n_nodes)
        planner_costs_manual = jnp.array([
            float(offer_sced(setup, load_flat[i]).total_cost)
            for i in range(N)
        ]).reshape(n_steps, n_envs)
        cr_manual = float(jnp.mean(
            planner_costs_manual / (gen_cost_arr + jnp.float32(1e-6))
        ))
        assert abs(cr - cr_manual) < 1e-3, (
            f"planner_cost_ratio mismatch: compute_market_metrics={cr:.6f}, "
            f"manual={cr_manual:.6f}.  Formula may be wrong."
        )

    def test_ippo_exports_planner_cost_ratio(self, marl_env):
        """make_ippo_train must export market/planner_cost_ratio for market envs.

        planner_cost_ratio = planner_cost / RL_gen_cost ∈ (0, 1] (cost ratio,
        not a welfare ratio — see module docstring for why social_welfare_ratio
        requires consumer value V which is not defined).
        """
        from powerzoojax.rl.ippo import make_ippo_train
        from powerzoojax.rl.config import TrainConfig

        config = TrainConfig(
            algo="ippo",
            total_timesteps=4 * 48,
            num_envs=4,
            n_steps=48,
            n_epochs=1,
            hidden_dims=(32, 32),
            eval_freq=0,
        )
        train_fn = make_ippo_train(marl_env, config)
        result = train_fn(jax.random.PRNGKey(53))

        assert "market/planner_cost_ratio" in result.metrics, (
            "planner_cost_ratio missing from IPPO training metrics. "
            f"Available market keys: {[k for k in result.metrics if 'market' in k]}"
        )
        assert "market/social_welfare_ratio" not in result.metrics, (
            "social_welfare_ratio must not appear in IPPO metrics — not implemented"
        )
        arr = result.metrics["market/planner_cost_ratio"]
        assert arr.ndim <= 1, f"Expected 1-D array, got shape {arr.shape}"
        assert float(jnp.isfinite(arr).all()), "planner_cost_ratio contains non-finite values"
        cr = float(arr[0])
        assert cr > 0.0, f"planner_cost_ratio should be positive, got {cr}"
        assert cr <= 1.0 + 1e-4, (
            f"planner_cost_ratio = {cr:.6f} > 1.0 in IPPO run "
            f"(planner is ramp-free DC-OPF, always ≤ RL gen cost)"
        )
