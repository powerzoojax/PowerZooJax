"""DC Optimal Power Flow — pure-JAX ADMM solver.

A JAX-native DC-OPF used by ``CostBasedMarketEnv`` (post-clear LMP) and
any caller that needs cost-minimising dispatch on a DC grid. The primal
objective is **linear** in dispatch (``mc_c @ p``), matching the LP
formulation of PowerZoo's ``_solve_ed_opf_scipy`` (HiGHS backend). For
comparison, ``clearing.piecewise_ed`` is a heuristic piecewise-linear
ED on offer curves (faster, approximate LMP) and
``market.offer_sced`` is an exact PD-IPM on segment-space LPs (used by
GenCos MARL).

ADMM consensus decomposition — primal variables ``p`` (dispatch),
``z1`` (box shadow), ``z2`` (line-flow shadow); dual variables
``y1``, ``y2``:

    1. **p-update** — solve the (n_units+1) × (n_units+1) KKT system::

           (ρ(I + M_uᵀ M_u))  p  +  λ · 1  =
                ρ z1 + ρ M_uᵀ(z2+c0) − mc_c − y1 − M_uᵀ y2
           1ᵀ p                                =  total_load

    2. **z1-update** — project onto box::

           z1 ← clip(p + y1/ρ, p_min, p_max)

    3. **z2-update** — project onto line bounds::

           z2 ← clip(M_u p − c0 + y2/ρ, floor, cap)

    4. **Dual updates** — ``y1 += ρ(p − z1)``,
       ``y2 += ρ(M_u p − c0 − z2)``

The p-update enforces balance exactly (it's part of the KKT system);
z2-projection enforces line feasibility. At convergence the primal
residuals ``‖p − z1‖`` and ``‖M_u p − c0 − z2‖`` both vanish and the
solution matches the LP optimal (including interior-point solutions
that a merit-order oracle alone cannot represent).

LMP recovery (post-convergence):
    LP line-constraint duals come from y2::

        μ_u = max(0,  y2),   μ_l = max(0, −y2)
        LMP_n = λ − PTDF[:,n]ᵀ · (μ_u − μ_l)

    where λ is the power-balance dual recovered from KKT stationarity
    on interior generators. Sign convention matches PowerZoo's
    ``LMP = λ_sys + PTDFᵀ · (μ_upper − μ_lower)``.

Performance:
    ``DCOPFSetup.A_aug_inv`` is the precomputed inverse of the
    (n_units+1)² ADMM KKT matrix, constant across all ADMM iterations
    *and* all env steps. Storing the inverse turns an O(n³) solve into
    an O(n²) matmul per iteration — a major win for vmap'd training
    rollouts.

Architecture:
    Setup-time (NumPy, called once):    prepare_dcopf → DCOPFSetup
    Runtime (pure JAX, JIT-compilable): dc_opf        → DCOPFResult

I/O contract:
    Input  : DCOPFSetup, node_load_mw (n_nodes,)
    Output : DCOPFResult { unit_power, line_flow, lmp, total_cost,
                            converged, iterations, residuals }

Translated from PowerZoo's ``cal_opf_trans.py``.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import jax.lax as lax
import chex
from flax import struct

from powerzoojax.case.case_data import CaseData


@struct.dataclass
class DCOPFSetup:
    """Precomputed data for DCOPF.

    Cost convention: ``mc_a, mc_b, mc_c`` are **marginal** cost coefficients.
    MC(p) = mc_a * p² + mc_b * p + mc_c   [$/MWh]
    TC(p) = (mc_a/3)*p³ + (mc_b/2)*p² + mc_c*p   [$]

    The dispatch objective uses only ``mc_c`` (linear), matching the
    PowerZoo LP.  ``mc_a`` and ``mc_b`` are retained for total-cost reporting.

    ``A_aug_inv`` is the precomputed inverse of the (n_units+1) × (n_units+1)
    ADMM KKT matrix (constant across all ADMM iterations and all env steps).
    Storing the inverse replaces an O(n³) solve with an O(n²) matmul per
    ADMM iteration, giving a large speedup for vmap'd training rollouts.
    """
    M_u: chex.Array            # (n_lines, n_units) PTDF @ nodes_units_map
    PTDF: chex.Array           # (n_lines, n_nodes)
    mc_a: chex.Array           # (n_units,) quadratic marginal cost $/MW²h
    mc_b: chex.Array           # (n_units,) linear marginal cost $/MWh
    mc_c: chex.Array           # (n_units,) constant marginal cost $/h
    p_min: chex.Array          # (n_units,)
    p_max: chex.Array          # (n_units,)
    line_cap: chex.Array       # (n_lines,)
    line_floor: chex.Array     # (n_lines,)
    nodes_units_map: chex.Array  # (n_nodes, n_units)
    A_aug_inv: chex.Array      # (n_units+1, n_units+1) inverse of ADMM KKT matrix
    n_units: int = struct.field(pytree_node=False, default=0)
    n_lines: int = struct.field(pytree_node=False, default=0)
    n_nodes: int = struct.field(pytree_node=False, default=0)


@struct.dataclass
class DCOPFResult:
    """Output of DC OPF solver."""
    unit_power: chex.Array     # (n_units,) MW
    line_flow: chex.Array      # (n_lines,) MW
    node_injection: chex.Array # (n_nodes,) MW
    lmp: chex.Array            # (n_nodes,) $/MWh
    total_cost: chex.Array     # scalar $
    converged: chex.Array      # bool
    iterations: chex.Array     # int32
    max_box_residual_mw: chex.Array      # scalar
    max_line_residual_mw: chex.Array     # scalar
    balance_residual_mw: chex.Array      # scalar


def prepare_dcopf(case: CaseData, rho: float = 1.0) -> DCOPFSetup:
    """Prepare DCOPF data from CaseData."""
    PTDF = np.asarray(case.PTDF)
    A_u = np.asarray(case.nodes_units_map)
    M_u = PTDF @ A_u

    line_cap = np.asarray(case.line_cap).copy()
    line_floor = np.asarray(case.line_floor).copy()
    NO_LIMIT = 1e5
    line_cap[line_cap == 0] = NO_LIMIT
    line_floor[line_floor == 0] = -NO_LIMIT

    n_u = M_u.shape[1]
    # Precompute the ADMM KKT matrix inverse (constant across all env steps and
    # all ADMM iterations).  Replacing solve(A_aug, rhs) with A_aug_inv @ rhs
    # reduces per-iteration cost from O(n³) LU to O(n²) matmul.
    A = rho * (np.eye(n_u, dtype=np.float64) + M_u.T @ M_u)
    A_aug = np.block([
        [A,                              np.ones((n_u, 1), dtype=np.float64)],
        [np.ones((1, n_u), dtype=np.float64), np.zeros((1, 1), dtype=np.float64)],
    ])
    A_aug_inv = np.linalg.inv(A_aug).astype(np.float32)

    return DCOPFSetup(
        M_u=jnp.array(M_u, dtype=jnp.float32),
        PTDF=jnp.array(PTDF, dtype=jnp.float32),
        mc_a=jnp.array(case.unit_cost_a, dtype=jnp.float32),
        mc_b=jnp.array(case.unit_cost_b, dtype=jnp.float32),
        mc_c=jnp.array(case.unit_cost_c, dtype=jnp.float32),
        p_min=jnp.array(case.unit_p_min, dtype=jnp.float32),
        p_max=jnp.array(case.unit_p_max, dtype=jnp.float32),
        line_cap=jnp.array(line_cap, dtype=jnp.float32),
        line_floor=jnp.array(line_floor, dtype=jnp.float32),
        nodes_units_map=jnp.array(A_u, dtype=jnp.float32),
        A_aug_inv=jnp.array(A_aug_inv, dtype=jnp.float32),
        n_units=case.n_units,
        n_lines=case.n_lines,
        n_nodes=case.n_nodes,
    )


def _merit_dispatch(
    d: chex.Array,
    p_min: chex.Array,
    p_max: chex.Array,
    total_load: chex.Array,
) -> chex.Array:
    """LP-optimal (merit-order) dispatch for linear cost vector ``d``.

    Solves:  min  dᵀ p   s.t.  sum(p) = total_load,  p_min ≤ p ≤ p_max

    The unique LP optimal is the vertex obtained by loading units in ascending
    order of ``d`` until demand is met (cheapest-first / merit order).

    Args:
        d: (n_units,) effective cost coefficients.
        p_min: (n_units,) lower bounds [MW].
        p_max: (n_units,) upper bounds [MW].
        total_load: scalar total demand [MW].

    Returns:
        (n_units,) dispatch array satisfying sum(p) = total_load and
        p_min ≤ p ≤ p_max (clipped to feasible range when infeasible).

    JIT- and vmap-compatible.
    """
    order = jnp.argsort(d)          # ascending: cheapest first
    inv_order = jnp.argsort(order)  # inverse permutation

    p_min_s = p_min[order]
    p_max_s = p_max[order]
    headroom_s = p_max_s - p_min_s  # >= 0

    # Amount of load that must be allocated above all-at-p_min baseline
    load_above_min = jnp.clip(
        total_load - jnp.sum(p_min),
        0.0,
        jnp.sum(headroom_s),
    )

    # Cumulative headroom of units 0..i-1 in sorted order (shifted right by 1)
    prev_cumsum = jnp.concatenate([
        jnp.zeros(1, dtype=jnp.float32),
        jnp.cumsum(headroom_s)[:-1],
    ])

    # Each unit absorbs min(remaining_above_min, its_headroom)
    additional = jnp.clip(load_above_min - prev_cumsum, 0.0, headroom_s)

    return (p_min_s + additional)[inv_order]


def _compute_lmp_alm(
    setup: DCOPFSetup,
    p_f: chex.Array,
    mu_u: chex.Array,
    mu_l: chex.Array,
    p_min: chex.Array,
    p_max: chex.Array,
) -> chex.Array:
    """Derive nodal LMP from ALM dual variables at primal convergence.

    At ALM convergence (and matching LP KKT), for an interior generator i
    (p_min_i < p_i < p_max_i):

        mc_c_i  =  λ  −  M_u[:,i]ᵀ · (μ_u − μ_l)

    λ is recovered via weighted mean over interior generators.  Then:

        LMP_n  =  λ  −  PTDF[:,n]ᵀ · (μ_u − μ_l)

    Sign convention is consistent with PowerZoo:
        LMP = λ_sys + PTDFᵀ · (mu_upper_scipy − mu_lower_scipy)
    with  mu_upper_scipy = −μ_u,  mu_lower_scipy = −μ_l.
    """
    eps_gen = jnp.float32(0.1)
    at_lower = p_f <= p_min + eps_gen
    at_upper = p_f >= p_max - eps_gen
    interior = ~(at_lower | at_upper)

    net_mu = mu_u - mu_l  # (n_lines,) net ALM shadow price

    # For interior generators: kkt_rhs_i = mc_c_i + M_u[:,i]ᵀ·net_mu = λ
    kkt_rhs = setup.mc_c + setup.M_u.T @ net_mu  # (n_units,)

    # Weighted mean for λ: full weight on interior, fallback to all-equal
    w = jnp.where(interior, 1.0, 0.0)
    w_sum = jnp.sum(w)
    w_safe = jnp.where(w_sum > jnp.float32(1e-6), w, jnp.ones_like(w) * jnp.float32(0.01))
    lam = jnp.sum(w_safe * kkt_rhs) / jnp.sum(w_safe)

    return lam - setup.PTDF.T @ net_mu


def dc_opf(
    setup: DCOPFSetup,
    node_load_mw: chex.Array,
    commitment: chex.Array = None,
    max_iter: int = 100,
    tol: float = 1e-3,
) -> DCOPFResult:
    """Solve DC-OPF via ADMM (Alternating Direction Method of Multipliers).

    Primal objective: ``mc_c @ p`` (linear), matching PowerZoo HiGHS LP.

    The ADMM consensus decomposition introduces auxiliary variables ``z1``
    (box projection) and ``z2`` (line-flow projection) so that each update
    step is cheap and JAX-native:

    - **p-update**: exact KKT solve (n_units+1 linear system) — balance is
      enforced exactly at every iteration.
    - **z1-update**: element-wise clip to ``[p_min, p_max]``.
    - **z2-update**: element-wise clip to ``[line_floor, line_cap]``.

    This handles the "degenerate LP optimal" case (e.g. congested line where
    the LP optimum requires two generators at non-bound values simultaneously)
    which merit-order / subgradient methods cannot represent exactly.

    Args:
        setup: Precomputed DCOPFSetup.
        node_load_mw: (n_nodes,) load [MW].
        commitment: (n_units,) binary mask. None = all on.
        max_iter: Max ADMM iterations (100 suffices for most cases; use 200
            for heavily congested systems).
        tol: Convergence tolerance for primal residuals [MW].

    Returns:
        DCOPFResult.
    """
    p_min = setup.p_min
    p_max = setup.p_max
    if commitment is not None:
        p_min = p_min * commitment
        p_max = p_max * commitment

    total_load = jnp.sum(node_load_mw)
    c0 = setup.PTDF @ node_load_mw   # load contribution to line flows

    n_u = setup.n_units  # Python int (static)
    n_l = setup.n_lines  # Python int (static)

    # ADMM penalty.  rho=1 balances MW-scale constraints and $/MWh costs.
    rho = jnp.float32(1.0)
    tol_arr = jnp.float32(tol)

    # ── Initialise ────────────────────────────────────────────────────────
    p0 = _merit_dispatch(setup.mc_c, p_min, p_max, total_load)
    z1_0 = p0
    z2_0 = jnp.clip(setup.M_u @ p0 - c0, setup.line_floor, setup.line_cap)
    y1_0 = jnp.zeros(n_u, dtype=jnp.float32)
    y2_0 = jnp.zeros(n_l, dtype=jnp.float32)

    # ── ADMM iterations via fori_loop (fixed count, GPU-friendly) ────────
    # Uses the precomputed A_aug_inv stored in setup (constant across all
    # iterations and env steps): A_aug_inv @ rhs_aug replaces linalg.solve,
    # reducing per-iteration cost from O(n³) to O(n²).
    def _body(i, state):
        p, z1, z2, y1, y2 = state

        # ── p-update: KKT solve via precomputed inverse ──────────────────
        rhs = (rho * z1
               + rho * setup.M_u.T @ (z2 + c0)
               - setup.mc_c
               - y1
               - setup.M_u.T @ y2)
        rhs_aug = jnp.append(rhs, total_load)
        sol = setup.A_aug_inv @ rhs_aug
        p_new = sol[:n_u]

        # ── z1-update: project onto box ──────────────────────────────────
        z1_new = jnp.clip(p_new + y1 / rho, p_min, p_max)

        # ── z2-update: project onto line constraint set ──────────────────
        z2_new = jnp.clip(
            setup.M_u @ p_new - c0 + y2 / rho,
            setup.line_floor,
            setup.line_cap,
        )

        # ── Dual updates ─────────────────────────────────────────────────
        y1_new = y1 + rho * (p_new - z1_new)
        y2_new = y2 + rho * (setup.M_u @ p_new - c0 - z2_new)

        return (p_new, z1_new, z2_new, y1_new, y2_new)

    p_f, z1_f, z2_f, y1_f, y2_f = lax.fori_loop(
        0, max_iter, _body, (p0, z1_0, z2_0, y1_0, y2_0)
    )

    # ── Post-loop convergence check ───────────────────────────────────────
    r_box  = jnp.max(jnp.abs(p_f - z1_f))
    r_line = jnp.max(jnp.abs(setup.M_u @ p_f - c0 - z2_f))
    balance = jnp.abs(jnp.sum(p_f) - total_load)
    conv = (r_box < tol_arr) & (r_line < tol_arr) & (balance < jnp.float32(1.0))
    iters = jnp.int32(max_iter)

    # LP line-constraint duals from ADMM:
    #   upper-bound binding → y2 > 0 → μ_u = y2
    #   lower-bound binding → y2 < 0 → μ_l = −y2
    mu_u_f = jnp.maximum(jnp.float32(0.0),  y2_f)
    mu_l_f = jnp.maximum(jnp.float32(0.0), -y2_f)

    flow = setup.M_u @ p_f - c0
    node_inj = setup.nodes_units_map @ p_f - node_load_mw
    total_cost = jnp.sum(
        (setup.mc_a / 3.0) * p_f ** 3
        + (setup.mc_b / 2.0) * p_f ** 2
        + setup.mc_c * p_f
    )

    lmp = _compute_lmp_alm(setup, p_f, mu_u_f, mu_l_f, p_min, p_max)

    return DCOPFResult(
        unit_power=p_f,
        line_flow=flow,
        node_injection=node_inj,
        lmp=lmp,
        total_cost=total_cost,
        converged=conv,
        iterations=iters,
        max_box_residual_mw=r_box,
        max_line_residual_mw=r_line,
        balance_residual_mw=balance,
    )
