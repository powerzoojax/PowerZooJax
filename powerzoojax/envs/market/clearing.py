"""Piecewise-linear Economic Dispatch — pure-JAX market clearing.

Approximate network-constrained dispatch used by ``BidBasedMarketEnv``.
Each generator's operating range ``[p_min, p_max]`` is divided into K
equal-width segments with per-segment marginal-cost prices; clearing
allocates segments by a three-step heuristic:

    1. Merit-order fill: sort all segments across generators by offer
       price, allocate cheapest first until total demand is met.
    2. PTDF penalty: penalise segments whose dispatch loads a congested
       line, iteratively shifting allocation toward less-loaded paths.
    3. Balance repair: enforce total dispatch = total demand exactly
       after the PTDF iterations settle.

This is **not** an exact LP/QP SCED. ``PiecewiseEDResult.converged``
flags whether the heuristic settled within tolerance — always check it
before consuming the result.

LMP recovery (``_compute_lmp_kkt_ed``):
    Because the heuristic does not produce LP dual variables, LMP is a
    KKT-style fit on each generator's *true* marginal cost at the
    dispatched operating point. Compatible with offer-based clearing
    (offer prices affect dispatch ordering; LMP reflects underlying
    economic costs). For exact LMPs see ``offer_sced`` (PD-IPM, used by
    GenCos MARL) or ``grid/dc_opf.dc_opf`` (ADMM, used by
    ``CostBasedMarketEnv``).

Idealised SCED this approximates (target physics, not the actual
implementation):
    minimise   Σ offer_price · delta
    subject to Σ (p_min + delta) = total_demand
               0 ≤ delta_k ≤ seg_width_k
               line_floor ≤ PTDF @ net_injection ≤ line_cap
    Nodal price (idealised): LMP_n = λ − Σ_l (μ⁺_l − μ⁻_l) · PTDF_{l,n}

Architecture:
    Setup-time (NumPy, called once):
        make_cost_segments    → CostSegments
        prepare_piecewise_ed  → PiecewiseEDSetup
    Runtime (pure JAX, JIT-compilable):
        piecewise_ed          → PiecewiseEDResult

I/O contract:
    Input  : PiecewiseEDSetup, node_load_mw (n_nodes,),
             offer_prices (n_units, K)
    Output : PiecewiseEDResult { unit_power, line_flow, lmp, total_cost,
                                  converged, ... }

Translated from PowerZoo's ``cal_dcopf_trans.py``
(``make_cost_segments`` and ``solve_piecewise_ed_opf``).
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
import chex
from flax import struct

from powerzoojax.case.case_data import CaseData


# ============ Data Structures ============

@struct.dataclass
class CostSegments:
    """Piecewise-linear cost segments for all generators.

    Attributes:
        seg_widths: (n_units, n_segments) MW width per block.
        seg_prices: (n_units, n_segments) $/MWh price per block.
    """
    seg_widths: chex.Array   # (n_units, n_segments)
    seg_prices: chex.Array   # (n_units, n_segments)


@struct.dataclass
class PiecewiseEDSetup:
    """Precomputed data for piecewise-linear economic dispatch.

    Created once at setup time (NumPy), used at runtime (JAX).
    """
    PTDF: chex.Array             # (n_lines, n_nodes)
    M_u: chex.Array              # (n_lines, n_units)  = PTDF @ nodes_units_map
    nodes_units_map: chex.Array  # (n_nodes, n_units)
    mc_a: chex.Array             # (n_units,) quadratic cost coeff
    mc_b: chex.Array             # (n_units,) linear cost coeff
    mc_c: chex.Array             # (n_units,) constant cost coeff
    p_min: chex.Array            # (n_units,)
    p_max: chex.Array            # (n_units,)
    line_cap: chex.Array         # (n_lines,)
    line_floor: chex.Array       # (n_lines,)
    base_seg_widths: chex.Array  # (n_units, n_segments)
    base_seg_prices: chex.Array  # (n_units, n_segments)
    n_units: int = struct.field(pytree_node=False, default=0)
    n_lines: int = struct.field(pytree_node=False, default=0)
    n_nodes: int = struct.field(pytree_node=False, default=0)
    n_segments: int = struct.field(pytree_node=False, default=5)


@struct.dataclass
class PiecewiseEDResult:
    """Output of piecewise-linear economic dispatch."""
    unit_power: chex.Array       # (n_units,) MW
    line_flow: chex.Array        # (n_lines,) MW
    node_injection: chex.Array   # (n_nodes,) MW
    lmp: chex.Array              # (n_nodes,) $/MWh — MC-based KKT recovery; see _compute_lmp_kkt_ed
    total_cost: chex.Array       # scalar $ (true nonlinear cost)
    offer_cost: chex.Array       # scalar $ (offer-based cost)
    converged: chex.Array        # bool
    iterations: chex.Array       # int32


# ============ Setup Functions (NumPy, called once) ============

def make_cost_segments(
    case: CaseData,
    n_segments: int = 5,
) -> CostSegments:
    """Build piecewise-linear offer segments from quadratic cost data.

    For each generator with MC(P) = mc_a·P² + mc_b·P + mc_c, the
    operating range [p_min, p_max] is divided into n_segments equal-width
    blocks.  Price of each block = MC evaluated at block midpoint.

    Args:
        case: CaseData with unit cost coefficients and power limits.
        n_segments: Number of segments per generator.

    Returns:
        CostSegments with seg_widths (n_units, K) and seg_prices (n_units, K).
    """
    p_min = np.asarray(case.unit_p_min)
    p_max = np.asarray(case.unit_p_max)
    mc_a = np.asarray(case.unit_cost_a)
    mc_b = np.asarray(case.unit_cost_b)
    mc_c = np.asarray(case.unit_cost_c)
    n_u = len(p_min)

    seg_widths = np.zeros((n_u, n_segments), dtype=np.float32)
    seg_prices = np.zeros((n_u, n_segments), dtype=np.float32)

    for i in range(n_u):
        width = (p_max[i] - p_min[i]) / n_segments
        seg_widths[i, :] = width
        for k in range(n_segments):
            p_mid = p_min[i] + (k + 0.5) * width
            # MC(p_mid) = mc_a * p_mid^2 + mc_b * p_mid + mc_c
            seg_prices[i, k] = mc_a[i] * p_mid ** 2 + mc_b[i] * p_mid + mc_c[i]

        # Enforce monotonicity
        for k in range(1, n_segments):
            # Numerical patch: if true MC is non-monotone (e.g. flat cost), nudge by 0.01 $/MWh.
            # This is a data-cleaning heuristic, not a market mechanism.
            if seg_prices[i, k] < seg_prices[i, k - 1]:
                seg_prices[i, k] = seg_prices[i, k - 1] + 0.01

    return CostSegments(
        seg_widths=jnp.array(seg_widths, dtype=jnp.float32),
        seg_prices=jnp.array(seg_prices, dtype=jnp.float32),
    )


def prepare_piecewise_ed(
    case: CaseData,
    n_segments: int = 5,
) -> PiecewiseEDSetup:
    """Prepare PiecewiseEDSetup from CaseData.

    Called once at setup time.  All heavy matrix operations use NumPy.
    The result is a frozen pytree passed to JIT-compiled functions.

    Args:
        case: CaseData instance.
        n_segments: Number of offer-curve segments per generator.

    Returns:
        PiecewiseEDSetup ready for ``piecewise_ed()``.
    """
    PTDF = np.asarray(case.PTDF)
    A_u = np.asarray(case.nodes_units_map)
    M_u = PTDF @ A_u

    line_cap = np.asarray(case.line_cap).copy()
    line_floor = np.asarray(case.line_floor).copy()
    NO_LIMIT = 1e5
    line_cap[line_cap == 0] = NO_LIMIT
    line_floor[line_floor == 0] = -NO_LIMIT

    segments = make_cost_segments(case, n_segments)

    return PiecewiseEDSetup(
        PTDF=jnp.array(PTDF, dtype=jnp.float32),
        M_u=jnp.array(M_u, dtype=jnp.float32),
        nodes_units_map=jnp.array(A_u, dtype=jnp.float32),
        mc_a=jnp.array(case.unit_cost_a, dtype=jnp.float32),
        mc_b=jnp.array(case.unit_cost_b, dtype=jnp.float32),
        mc_c=jnp.array(case.unit_cost_c, dtype=jnp.float32),
        p_min=jnp.array(case.unit_p_min, dtype=jnp.float32),
        p_max=jnp.array(case.unit_p_max, dtype=jnp.float32),
        line_cap=jnp.array(line_cap, dtype=jnp.float32),
        line_floor=jnp.array(line_floor, dtype=jnp.float32),
        base_seg_widths=segments.seg_widths,
        base_seg_prices=segments.seg_prices,
        n_units=case.n_units,
        n_lines=case.n_lines,
        n_nodes=case.n_nodes,
        n_segments=n_segments,
    )


# ============ Runtime (Pure JAX, JIT-compilable) ============


def _compute_lmp_kkt_ed(
    setup: PiecewiseEDSetup,
    unit_power: chex.Array,
    flow: chex.Array,
) -> chex.Array:
    """Derive LMP from KKT stationarity of the piecewise-linear ED.

    Same approach as ``dc_opf._compute_lmp_kkt``: for interior generators
    MC(P_i) = λ − net_mu^T · M_u[:,i], solved via weighted least-squares.

    LMP_n = λ − PTDF[:,n]^T · net_mu

    Note: ``mc`` is evaluated from the *true* quadratic cost coefficients
    (``mc_a``, ``mc_b``, ``mc_c``), not from the offer prices used in
    dispatch.  When offer prices deviate from true costs (markup ≠ 0 in
    ``BidBasedMarketEnv``), the recovered LMP is an approximation based on
    true marginal costs at the dispatched operating point, not a strict
    dual of the offer-based clearing problem.  This is a deliberate
    Market Lite simplification suitable for RL reward signals.
    """
    n_units = setup.n_units
    n_lines = setup.n_lines

    mc = setup.mc_a * unit_power ** 2 + setup.mc_b * unit_power + setup.mc_c

    eps_gen = 0.01
    eps_line = 0.1
    at_lower = unit_power <= setup.p_min + eps_gen
    at_upper = unit_power >= setup.p_max - eps_gen
    interior = ~(at_lower | at_upper)
    w = jnp.where(interior, 1.0, 0.01)

    upper_bind = flow >= setup.line_cap - eps_line
    lower_bind = flow <= setup.line_floor + eps_line
    congested = upper_bind | lower_bind

    A = jnp.concatenate(
        [jnp.ones((n_units, 1)), -setup.M_u.T], axis=1
    )

    Aw = w[:, None] * A
    AtWA = A.T @ Aw
    AtWb = A.T @ (w * mc)

    reg_lambda = jnp.array([1e-6])
    reg_mu = jnp.where(congested, 1e-6, 1e4)
    reg = jnp.concatenate([reg_lambda, reg_mu])

    x = jnp.linalg.solve(AtWA + jnp.diag(reg), AtWb)
    lam = x[0]
    net_mu = x[1:]
    net_mu = jnp.where(congested, net_mu, 0.0)

    lmp = lam - setup.PTDF.T @ net_mu
    return lmp


def piecewise_ed(
    setup: PiecewiseEDSetup,
    node_load_mw: chex.Array,
    offer_prices: chex.Array = None,
    max_iter: int = 50,
    tol: float = 1e-3,
) -> PiecewiseEDResult:
    """Solve piecewise-linear economic dispatch with congestion management.

    This is an approximate network-constrained economic dispatch, not an
    exact LP/QP SCED.  Dispatch uses merit-order + iterative PTDF penalty
    (heuristic, not globally optimal).  LMP is recovered via weighted
    least-squares KKT stationarity, not from LP dual variables.
    Convergence is not guaranteed for all cases; check ``result.converged``
    before using the result in downstream analysis.

    Merit-order dispatch: segments are sorted by price, filled cheapest-first.
    Congestion managed by iterative penalty on line violations (same approach
    as ``dc_opf.py``).

    Args:
        setup: Precomputed PiecewiseEDSetup.
        node_load_mw: (n_nodes,) nodal load [MW].
        offer_prices: (n_units, n_segments) $/MWh. If None, uses base prices.
        max_iter: Maximum iterations for congestion management.
        tol: Convergence tolerance [MW].

    Returns:
        PiecewiseEDResult.
    """
    seg_widths = setup.base_seg_widths  # (n_units, n_segments)
    seg_prices = jnp.where(
        offer_prices is not None,
        offer_prices,
        setup.base_seg_prices,
    ) if offer_prices is not None else setup.base_seg_prices

    n_u = setup.n_units
    n_seg = setup.n_segments

    # Flatten segments: (n_units * n_segments,)
    flat_widths = seg_widths.reshape(-1)
    flat_prices = seg_prices.reshape(-1)

    # M_S: (n_lines, n_units * n_segments) — each segment maps through its unit's PTDF column
    # M_S[:, i*K+k] = M_u[:, i] for all k
    M_S = jnp.repeat(setup.M_u, n_seg, axis=1)

    total_load = jnp.sum(node_load_mw)
    c0 = setup.PTDF @ node_load_mw

    # Flow contribution from p_min dispatch
    flow_pmin = setup.M_u @ setup.p_min - c0

    # Demand to fill with segments: total_demand - sum(p_min)
    demand_delta = total_load - jnp.sum(setup.p_min)
    demand_delta = jnp.maximum(demand_delta, 0.0)

    # Merit-order initialization: sort segments by price, fill cheapest first
    sort_idx = jnp.argsort(flat_prices)
    sorted_widths = flat_widths[sort_idx]
    sorted_prices = flat_prices[sort_idx]

    # Fill segments cumulatively
    cumulative_capacity = jnp.cumsum(sorted_widths)
    # Each segment gets: min(width, max(0, demand - cumulative_before))
    cumulative_before = cumulative_capacity - sorted_widths
    sorted_delta = jnp.clip(
        jnp.minimum(sorted_widths, jnp.maximum(demand_delta - cumulative_before, 0.0)),
        0.0, sorted_widths,
    )

    # Unsort back to original order
    inv_idx = jnp.argsort(sort_idx)
    delta0 = sorted_delta[inv_idx]

    # Iterative congestion management
    penalty = jnp.float32(100.0)
    tol_arr = jnp.float32(tol)
    # Capacity-adaptive learning rate (same logic as dc_opf)
    total_headroom = jnp.maximum(jnp.sum(flat_widths), 1.0)
    lr = jnp.maximum(total_headroom / 100.0, 5.0)

    def _cond(state):
        delta, it, conv = state
        return jnp.logical_and(it < max_iter, jnp.logical_not(conv))

    def _body(state):
        delta, it, _ = state

        # Current line flows
        flow = M_S @ delta + flow_pmin

        over = jnp.maximum(flow - setup.line_cap, 0.0)
        under = jnp.maximum(setup.line_floor - flow, 0.0)

        # If no violations, converged
        line_viol = jnp.sum(over + under)

        # Cost gradient: offer prices
        g_cost = flat_prices

        # Penalty gradient from line violations
        g_line = penalty * M_S.T @ (over - under)

        g = g_cost + g_line

        # Normalised step
        g_scale = jnp.maximum(jnp.max(jnp.abs(g)), 1.0)
        step = g / g_scale

        # Descent (capacity-adaptive learning rate)
        delta_new = delta - step * lr
        delta_new = jnp.clip(delta_new, 0.0, flat_widths)

        # Enforce power balance: sum(delta) = demand_delta
        # When err > 0 (under-generation), allocate proportionally to remaining capacity.
        # When err < 0 (over-generation), reduce proportionally to current dispatch.
        gen_delta = jnp.sum(delta_new)
        err = demand_delta - gen_delta
        remaining = flat_widths - delta_new
        weight_up = remaining / jnp.maximum(jnp.sum(remaining), 1e-6)
        weight_down = delta_new / jnp.maximum(jnp.sum(delta_new), 1e-6)
        balance_weight = jnp.where(err > 0, weight_up, weight_down)
        delta_new = delta_new + err * balance_weight
        delta_new = jnp.clip(delta_new, 0.0, flat_widths)

        # Second pass for residual
        err2 = demand_delta - jnp.sum(delta_new)
        remaining2 = flat_widths - delta_new
        weight_up2 = remaining2 / jnp.maximum(jnp.sum(remaining2), 1e-6)
        weight_down2 = delta_new / jnp.maximum(jnp.sum(delta_new), 1e-6)
        balance_weight2 = jnp.where(err2 > 0, weight_up2, weight_down2)
        delta_new = delta_new + err2 * balance_weight2
        delta_new = jnp.clip(delta_new, 0.0, flat_widths)

        change = jnp.max(jnp.abs(delta_new - delta))
        balance = jnp.abs(jnp.sum(delta_new) - demand_delta)
        converged = jnp.logical_and(
            change < tol_arr,
            jnp.logical_and(balance < 1.0, line_viol < tol_arr))

        return (delta_new, it + 1, converged)

    init = (delta0, jnp.int32(0), jnp.bool_(False))
    delta_f, iters, conv = lax.while_loop(_cond, _body, init)

    # Reconstruct unit power: (n_units,) = p_min + sum over segments
    delta_2d = delta_f.reshape(n_u, n_seg)
    unit_power = setup.p_min + jnp.sum(delta_2d, axis=1)

    # Line flows
    flow = M_S @ delta_f + flow_pmin
    node_inj = setup.nodes_units_map @ unit_power - node_load_mw

    # True nonlinear cost
    total_cost = jnp.sum(
        (setup.mc_a / 3.0) * unit_power ** 3
        + (setup.mc_b / 2.0) * unit_power ** 2
        + setup.mc_c * unit_power
    )

    # Offer cost
    offer_cost = jnp.sum(flat_prices * delta_f)

    # LMP from KKT stationarity conditions
    lmp = _compute_lmp_kkt_ed(setup, unit_power, flow)

    return PiecewiseEDResult(
        unit_power=unit_power,
        line_flow=flow,
        node_injection=node_inj,
        lmp=lmp,
        total_cost=total_cost,
        offer_cost=offer_cost,
        converged=conv,
        iterations=iters,
    )
