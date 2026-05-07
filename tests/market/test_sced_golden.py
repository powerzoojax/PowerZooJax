"""Golden equivalence tests: JAX PD-IPM offer_sced vs Python HiGHS LP.

These tests encode "golden" reference values produced by
``PowerZoo.powerzoo.envs.grid.cal_opf_trans.solve_bid_sced_scipy`` (exact LP via
scipy/HiGHS) and assert that the JAX PD-IPM ``offer_sced`` matches them.

Both solvers use the same LP formulation (segment-space):
    min  Σ_{i,k} offer_price_{i,k} · δ_{i,k}
    s.t.  Σ δ = D_delta            (system balance above p_min)
          0 ≤ δ ≤ seg_widths       (box)
          line_floor ≤ M_S δ + flow_pmin ≤ line_cap   (line limits)

Golden values were computed with ``scipy.optimize.linprog(method='highs')`` on
case5, n_segments=3, truthful prices (MC at segment midpoints).

Scenario 1 — Uncongested, load = [0, 250, 300, 200, 0] MW (total 750 MW):
    HiGHS: unit=[40, 80, 20, 10, 600], LMP=[15, 15, 15, 15, 15]

Scenario 2 — Congested (line 6 binding), load = [0, 400, 450, 400, 0] MW (total 1250 MW):
    HiGHS: unit=[40, 170, 520, 23.66, 496.34], LMP=[16.99, 26.42, 30.04, 40.0, 10.0]

Note on LP degeneracy:
    In the congested scenario the LP is degenerate (multiple primal optima on
    the face where line 6 and balance are simultaneously binding).  HiGHS
    (simplex) finds a vertex; IPM converges to the analytic center of the
    optimal face — so unit_power may differ, but offer_cost and LMP match.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest
from scipy.optimize import linprog

from powerzoojax.case import load_case
from powerzoojax.envs.market.offer_sced import prepare_offer_sced, offer_sced


# ---------------------------------------------------------------------------
# HiGHS LP reference (segment-space SCED, float64)
# ---------------------------------------------------------------------------

def _solve_offer_sced_highs(sced_setup, node_load_mw, offer_prices: np.ndarray):
    """Solve the same LP as ``offer_sced`` via SciPy HiGHS (exact simplex LP).

    Returns primal segment deltas, costs, line flows, and LMP from scipy duals:
    ``lmp_n = ν + PTDF[:,n]^T · (λ⁺ - λ⁻)`` where ν is the equality marginal and
    ``λ⁺, λ⁻`` are the first / second block of inequality marginals for
    ``M_S δ ≤ …`` and ``-M_S δ ≤ …``.  This matches ``OfferSCEDResult.lmp`` from
    ``λ_balance = -ν`` (sign convention in the PD-IPM).

    Parameters
    ----------
    sced_setup
        Result of ``prepare_offer_sced(case, n_segments=…)``.
    node_load_mw
        (n_nodes,) nodal load [MW].
    offer_prices
        (n_units, n_segments) offer prices [$/MWh], float32 or float64.

    Returns
    -------
    dict with keys: ``success``, ``segment_delta``, ``unit_power``, ``offer_cost``,
    ``lmp``, ``line_flow``, ``mu_line_upper``, ``mu_line_lower``, ``linprog_result``.
    """
    PTDF = np.asarray(sced_setup.PTDF)
    M_u = np.asarray(sced_setup.M_u)
    n_u = sced_setup.n_units
    n_seg = sced_setup.n_segments
    n_l = sced_setup.n_lines
    n_dim = n_u * n_seg

    M_S = np.repeat(M_u, n_seg, axis=1)
    p_min = np.asarray(sced_setup.p_min, dtype=np.float64)
    delta_bar = np.asarray(sced_setup.base_seg_widths, dtype=np.float64).ravel()

    line_cap = np.asarray(sced_setup.line_cap, dtype=np.float64)
    line_floor = np.asarray(sced_setup.line_floor, dtype=np.float64)

    offer_prices = np.asarray(offer_prices, dtype=np.float64)
    c = offer_prices.ravel()

    node_load_mw = np.asarray(node_load_mw, dtype=np.float64)
    total_load = float(np.sum(node_load_mw))
    flow_pmin = M_u @ p_min - PTDF @ node_load_mw
    d_delta = total_load - float(np.sum(p_min))

    A_eq = np.ones((1, n_dim))
    b_eq = np.array([d_delta])
    A_ub = np.vstack([M_S, -M_S])
    b_ub = np.concatenate([line_cap - flow_pmin, flow_pmin - line_floor])
    bounds = [(0.0, float(delta_bar[i])) for i in range(n_dim)]

    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        return {
            "success": False,
            "segment_delta": None,
            "unit_power": None,
            "offer_cost": np.nan,
            "lmp": None,
            "line_flow": None,
            "mu_line_upper": None,
            "mu_line_lower": None,
            "linprog_result": res,
        }

    x = res.x
    seg_delta = x.reshape(n_u, n_seg)
    unit_power = p_min + np.sum(seg_delta, axis=1)
    offer_cost = float(c @ x)

    nu_eq = res.eqlin["marginals"][0]
    mu_all = res.ineqlin["marginals"]
    mu_up = mu_all[:n_l]
    mu_lo = mu_all[n_l : 2 * n_l]
    lmp = nu_eq + PTDF.T @ (mu_up - mu_lo)

    line_flow = M_S @ x + flow_pmin

    return {
        "success": True,
        "segment_delta": seg_delta,
        "unit_power": unit_power,
        "offer_cost": offer_cost,
        "lmp": lmp,
        "line_flow": line_flow,
        "mu_line_upper": mu_up,
        "mu_line_lower": mu_lo,
        "linprog_result": res,
    }


# ---------------------------------------------------------------------------
# Golden constants (produced by HiGHS, exact LP)
# ---------------------------------------------------------------------------

# Scenario 1: uncongested, total load = 750 MW
LOAD_750 = np.array([0., 250., 300., 200., 0.], dtype=np.float32)
GOLDEN_750_UNIT_POWER = np.array([40., 80., 20., 10., 600.], dtype=np.float32)
GOLDEN_750_LMP        = np.array([15., 15., 15., 15., 15.], dtype=np.float32)
GOLDEN_750_OFFER_COST = 7340.0
GOLDEN_750_LINE_FLOW  = np.array(
    [335.684, 150.885, -366.568, 85.684, -194.316, -233.432],
    dtype=np.float32,
)

# Scenario 2: congested (line 6 at -240 MW cap), total load = 1250 MW
LOAD_1250 = np.array([0., 400., 450., 400., 0.], dtype=np.float32)
GOLDEN_1250_LMP        = np.array(
    [16.991, 26.416, 30.038, 40.0, 10.0], dtype=np.float32
)
GOLDEN_1250_UNIT_POWER = np.array(
    [40., 170., 520., 23.657, 496.343], dtype=np.float32
)
GOLDEN_1250_LINE_6_FLOW = -240.0          # line 6 exactly at lower limit

# Tiered offers: all 15 block prices globally distinct (no cross-unit tie),
# strictly increasing within each unit.  Merit order at D_delta=685 MW (750 MW
# load) fills blocks in price order until G5 seg2 is partially used (~168.33 MW),
# giving a unique uncongested primal (HiGHS vertex = merit-order fill).
OFFER_PRICES_TIERED = np.array(
    [
        [12.0, 15.5, 21.0],
        [13.0, 17.0, 23.0],
        [25.0, 33.0, 46.0],
        [36.0, 43.0, 56.0],
        [8.0, 11.5, 18.0],
    ],
    dtype=np.float32,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sced_setup():
    c = load_case("case5")
    return prepare_offer_sced(c, n_segments=3)


@pytest.fixture(scope="module")
def result_750(sced_setup):
    load = jnp.array(LOAD_750)
    return offer_sced(sced_setup, load, sced_setup.base_seg_prices)


@pytest.fixture(scope="module")
def result_1250(sced_setup):
    load = jnp.array(LOAD_1250)
    return offer_sced(sced_setup, load, sced_setup.base_seg_prices)


# ---------------------------------------------------------------------------
# Scenario 1: Uncongested — strict unit_power + LMP comparison
# ---------------------------------------------------------------------------

class TestUncongested750:
    """JAX PD-IPM vs HiGHS for case5 with 750 MW load (all lines uncongested)."""

    def test_converged(self, result_750):
        assert result_750.converged, "IPM must converge for uncongested case"

    def test_is_feasible(self, result_750):
        assert result_750.is_feasible, "750 MW load is within [p_min_sum, p_max_sum]"

    def test_unit_power_vs_golden(self, result_750):
        """Unit dispatch must match HiGHS within 0.5 MW."""
        np.testing.assert_allclose(
            np.array(result_750.unit_power),
            GOLDEN_750_UNIT_POWER,
            atol=0.5,
            err_msg="unit_power mismatch vs HiGHS golden (uncongested)",
        )

    def test_lmp_vs_golden(self, result_750):
        """LMP must match HiGHS within 0.05 $/MWh."""
        np.testing.assert_allclose(
            np.array(result_750.lmp),
            GOLDEN_750_LMP,
            atol=0.05,
            err_msg="LMP mismatch vs HiGHS golden (uncongested)",
        )

    def test_lmp_uniform(self, result_750):
        """Uncongested → all nodal LMPs should be nearly equal."""
        lmp = np.array(result_750.lmp)
        assert lmp.max() - lmp.min() < 0.05, (
            f"LMP spread too large for uncongested case: {lmp}"
        )

    def test_offer_cost_vs_golden(self, result_750):
        np.testing.assert_allclose(
            float(result_750.offer_cost),
            GOLDEN_750_OFFER_COST,
            atol=1.0,
            err_msg="offer_cost mismatch vs HiGHS golden",
        )

    def test_balance(self, result_750):
        total_load = float(LOAD_750.sum())
        total_gen = float(jnp.sum(result_750.unit_power))
        assert abs(total_gen - total_load) < 0.01, (
            f"Balance violated: gen={total_gen:.3f}, load={total_load:.3f}"
        )

    def test_line_flow_vs_golden(self, result_750):
        """Line flows must match HiGHS within 0.5 MW."""
        np.testing.assert_allclose(
            np.array(result_750.line_flow),
            GOLDEN_750_LINE_FLOW,
            atol=0.5,
            err_msg="line_flow mismatch vs HiGHS golden (uncongested)",
        )


# ---------------------------------------------------------------------------
# Scenario 2: Congested — strict LMP + cost check, feasibility for dispatch
# ---------------------------------------------------------------------------

class TestCongested1250:
    """JAX PD-IPM vs HiGHS for case5 with 1250 MW load (line 6 binding at -240 MW).

    The LP is degenerate: multiple primal optima exist.  HiGHS finds a vertex;
    IPM converges to the analytic center of the optimal face.  unit_power may
    differ, but offer_cost and LMP match closely.
    """

    def test_converged(self, result_1250):
        assert result_1250.converged, "IPM must converge for congested case"

    def test_is_feasible(self, result_1250):
        assert result_1250.is_feasible, "1250 MW load is within [p_min_sum, p_max_sum]"

    def test_lmp_vs_golden(self, result_1250):
        """LP dual-based LMP must match HiGHS within 0.2 $/MWh."""
        np.testing.assert_allclose(
            np.array(result_1250.lmp),
            GOLDEN_1250_LMP,
            atol=0.2,
            err_msg="LMP mismatch vs HiGHS golden (congested)",
        )

    def test_lmp_nodal_spread(self, result_1250):
        """Congested case must show non-trivial nodal LMP spread (> 5 $/MWh)."""
        lmp = np.array(result_1250.lmp)
        assert lmp.max() - lmp.min() > 5.0, (
            f"Expected LMP spread due to congestion, got: {lmp}"
        )

    def test_balance(self, result_1250):
        total_load = float(LOAD_1250.sum())
        total_gen = float(jnp.sum(result_1250.unit_power))
        assert abs(total_gen - total_load) < 0.1, (
            f"Balance violated: gen={total_gen:.3f}, load={total_load:.3f}"
        )

    def test_line6_at_limit(self, result_1250, sced_setup):
        """Line 6 should be binding at its lower limit (-240 MW)."""
        line_flow = np.array(result_1250.line_flow)
        assert abs(line_flow[5] - GOLDEN_1250_LINE_6_FLOW) < 0.5, (
            f"Line 6 flow {line_flow[5]:.2f} not at expected limit {GOLDEN_1250_LINE_6_FLOW}"
        )

    def test_unit_power_box_feasible(self, result_1250, sced_setup):
        """Unit dispatch must respect [p_min, p_max] bounds."""
        p = np.array(result_1250.unit_power)
        p_min = np.array(sced_setup.p_min)
        p_max = np.array(sced_setup.p_max)
        assert np.all(p >= p_min - 0.1), f"Some units below p_min: {p}"
        assert np.all(p <= p_max + 0.1), f"Some units above p_max: {p}"

    def test_offer_cost_near_optimal(self, result_1250):
        """IPM offer cost must be within 0.1% of HiGHS optimal.

        Both IPM and HiGHS should find the same objective value because
        the LP dual (and hence objective on the optimal face) is unique.
        """
        highs_cost = float(
            GOLDEN_1250_UNIT_POWER @ np.array([14., 15., 30., 40., 10.])
            - np.array([5., 10., 20., 10., 20.]) @ np.array([14., 15., 30., 40., 10.])
        )
        jax_cost = float(result_1250.offer_cost)
        rel_diff = abs(jax_cost - highs_cost) / max(abs(highs_cost), 1.0)
        assert rel_diff < 0.001, (
            f"IPM offer_cost {jax_cost:.2f} deviates from HiGHS {highs_cost:.2f} "
            f"by {rel_diff*100:.3f}% (>0.1%)"
        )


# ---------------------------------------------------------------------------
# Segment setup equivalence (Python vs JAX)
# ---------------------------------------------------------------------------

class TestSegmentSetup:
    """Verify that Python _make_cost_segments_np and JAX make_cost_segments agree."""

    def test_seg_prices(self, sced_setup):
        """Truthful segment prices for case5 (mc_a=mc_b=0 → flat mc_c per unit)."""
        expected_prices = np.array([
            [14., 14., 14.],  # unit 1: mc_c=14
            [15., 15., 15.],  # unit 2: mc_c=15
            [30., 30., 30.],  # unit 3: mc_c=30
            [40., 40., 40.],  # unit 4: mc_c=40
            [10., 10., 10.],  # unit 5: mc_c=10
        ], dtype=np.float32)
        np.testing.assert_allclose(
            np.array(sced_setup.base_seg_prices),
            expected_prices,
            atol=0.01,
        )

    def test_seg_widths(self, sced_setup):
        """Segment widths must equal (p_max - p_min) / n_seg for each unit."""
        p_min = np.array([5., 10., 20., 10., 20.])
        p_max = np.array([40., 170., 520., 200., 600.])
        expected_widths = np.tile(
            ((p_max - p_min) / 3.0).reshape(-1, 1), (1, 3)
        ).astype(np.float32)
        np.testing.assert_allclose(
            np.array(sced_setup.base_seg_widths),
            expected_widths,
            atol=0.01,
        )

    def test_seg_widths_sum_equals_operating_range(self, sced_setup):
        """Sum of widths per unit = p_max - p_min."""
        p_min = np.array([5., 10., 20., 10., 20.], dtype=np.float32)
        p_max = np.array([40., 170., 520., 200., 600.], dtype=np.float32)
        width_sum = np.array(sced_setup.base_seg_widths).sum(axis=1)
        np.testing.assert_allclose(width_sum, p_max - p_min, atol=0.01)


# ---------------------------------------------------------------------------
# Tiered offer prices: JAX PD-IPM primal vs HiGHS; LMP from HiGHS dual (see offer_sced)
# ---------------------------------------------------------------------------


class TestTieredOfferPrices:
    """15 globally distinct block prices (no merit-order ties).

    ``offer_sced.lmp`` is recovered via HiGHS on the same LP (host callback); primal
    ``δ`` comes from PD-IPM and is compared to HiGHS at 1 MW.
    """

    def test_uncongested_tiered(self, sced_setup):
        """Scenario A: unique merit-order path; primal + dual vs HiGHS."""
        load = np.asarray(LOAD_750, dtype=np.float32)
        hi = _solve_offer_sced_highs(sced_setup, load, OFFER_PRICES_TIERED)
        assert hi["success"], hi["linprog_result"].message

        r = offer_sced(
            sced_setup,
            jnp.array(load),
            jnp.array(OFFER_PRICES_TIERED),
        )
        assert bool(r.converged) and bool(r.is_feasible)

        np.testing.assert_allclose(
            float(r.offer_cost),
            hi["offer_cost"],
            atol=1.0,
            err_msg="offer_cost vs HiGHS (tiered, 750 MW)",
        )
        np.testing.assert_allclose(
            np.array(r.lmp),
            hi["lmp"],
            atol=0.5,
            err_msg="LMP vs HiGHS dual recovery (tiered, 750 MW)",
        )
        np.testing.assert_allclose(
            np.array(r.unit_power),
            hi["unit_power"],
            atol=1.0,
            err_msg="unit_power vs HiGHS (tiered uncongested)",
        )
        np.testing.assert_allclose(
            np.array(r.segment_delta),
            hi["segment_delta"],
            atol=1.0,
            err_msg="segment_delta vs HiGHS (tiered uncongested)",
        )

    def test_congested_tiered(self, sced_setup):
        """Scenario B: offer_cost, LMP, primal vs HiGHS (line 6 binding)."""
        load = np.asarray(LOAD_1250, dtype=np.float32)
        hi = _solve_offer_sced_highs(sced_setup, load, OFFER_PRICES_TIERED)
        assert hi["success"], hi["linprog_result"].message

        r = offer_sced(
            sced_setup,
            jnp.array(load),
            jnp.array(OFFER_PRICES_TIERED),
        )
        assert bool(r.converged) and bool(r.is_feasible)

        np.testing.assert_allclose(
            float(r.offer_cost),
            hi["offer_cost"],
            atol=1.0,
            err_msg="offer_cost vs HiGHS (tiered, 1250 MW)",
        )
        np.testing.assert_allclose(
            np.array(r.lmp),
            hi["lmp"],
            atol=0.5,
            err_msg="LMP vs HiGHS dual recovery (tiered, congested)",
        )
        np.testing.assert_allclose(
            np.array(r.unit_power),
            hi["unit_power"],
            atol=1.0,
            err_msg="unit_power vs HiGHS (tiered congested)",
        )
        np.testing.assert_allclose(
            np.array(r.segment_delta),
            hi["segment_delta"],
            atol=1.0,
            err_msg="segment_delta vs HiGHS (tiered congested)",
        )
        assert abs(float(r.line_flow[5]) - (-240.0)) < 0.5, (
            f"line 6 should sit at lower limit: {float(r.line_flow[5])}"
        )
        assert abs(float(hi["mu_line_lower"][5])) > 1.0, (
            "expected line 6 lower-limit dual (shadow price) in tiered congested case"
        )

    def test_segment_assignment(self, sced_setup):
        """Merit-order structure; segment_delta vs HiGHS; offer_cost identity."""
        load = jnp.array(LOAD_750, dtype=np.float32)
        hi = _solve_offer_sced_highs(sced_setup, load, OFFER_PRICES_TIERED)
        assert hi["success"]

        r = offer_sced(
            sced_setup,
            load,
            jnp.array(OFFER_PRICES_TIERED),
        )
        assert bool(r.converged)
        seg = np.array(r.segment_delta)
        widths = np.array(sced_setup.base_seg_widths)

        np.testing.assert_allclose(seg, hi["segment_delta"], atol=1.0)

        # G5 (unit 4) has globally cheapest first block — should be full before using G4.
        assert seg[4, 0] >= widths[4, 0] - 0.5, (
            f"G5 segment 0 should fill (merit order): got {seg[4, 0]} vs width {widths[4, 0]}"
        )
        assert seg[3, 2] < 1.0, (
            f"G4 seg2 (56 $/MWh) unused at 750 MW: {seg[3, 2]}"
        )
        assert 160.0 < seg[4, 2] < widths[4, 2] - 1.0, (
            f"G5 seg2 should be marginal partial block: {seg[4, 2]}"
        )
        # Offer cost matches manual sum(price * delta) with row-major flatten as solver.
        c = np.asarray(OFFER_PRICES_TIERED).ravel()
        d = seg.ravel()
        manual_cost = float(np.sum(c * d))
        np.testing.assert_allclose(
            float(r.offer_cost),
            manual_cost,
            atol=0.01,
            err_msg="offer_cost must equal sum(flat_offer_prices * flat segment_delta)",
        )
