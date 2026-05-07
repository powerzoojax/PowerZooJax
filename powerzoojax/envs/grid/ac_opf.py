"""AC Optimal Power Flow — reduced-space, IFT-differentiable.

A JAX-native AC-OPF solver used by ``TransGridEnv`` when
``solver_mode=2`` (ACOPF dispatches generators *and* solves AC PF in
one pass; the ``physics`` flag is ignored in this mode). Three-module
pipeline:

  A. Newton-Raphson power flow (``lax.while_loop``)
  B. ``jax.custom_vjp`` wrapping NR with Implicit Function Theorem
     gradients (no AD through ``while_loop`` → memory-bounded, exact
     O(n²) sensitivity)
  C. ``jaxopt.LBFGSB`` outer optimisation over controls ``u`` with
     PHR augmented Lagrangian for line limits / bus Q / PQ-bus Vm /
     slack P-balance.

Variable split:
    u (control) : all generator ``Pg`` [p.u.] + ``Vm`` at generator buses
    x (state)   : ``Va`` at non-slack buses + ``Vm`` at PQ (load-only) buses

The slack-bus active-power row is **not** solved inside NR; it is
enforced as the scalar equality
    h_bal = P_inj[slack] − (Cg Pg)[slack] + Pd[slack] = 0
via equality ALM, so losses spread across all generators (no single
"balance machine"). Outer problem::

    min_u  C(Pg) + ALM(line, Qg, Vm_pq, h_bal)
    s.t.   lb_u ≤ u ≤ ub_u

Bus types: slack (Va=0, Vm controlled); PV (Va solved, Vm controlled);
PQ (both solved). For case5: n_pq = 0 → 4×4 NR Jacobian.

Cost convention divergence (read carefully):
    AC-OPF uses the **standard MATPOWER quadratic-TC** form:
        TC(P) = cost_a · P²  +  cost_c · P            [$ at the dt]
        MC(P) = 2 · cost_a · P  +  cost_c              [$/MWh]
    ``cost_b`` is **not** used. This diverges from the project-wide
    quadratic-MC convention
        MC = a · P² + b · P + c
        TC = (a/3) P³ + (b/2) P² + c · P
    used by ``compute_generation_cost``, DC-OPF, and the market envs.
    When importing a CaseData prepared for the project convention,
    only ``cost_a`` and ``cost_c`` carry through to the AC-OPF
    objective with a different physical interpretation — verify your
    case file supplies the right form for the AC path before mixing
    DC-mode and AC-OPF runs.

Caveats:
  * ``jax_enable_x64`` is required for ALM + L-BFGS-B numerical stability.
  * ``vmap`` over ``ac_opf`` makes nested ``lax.while_loop`` instances
    wait for the slowest to converge.
  * LMP from ``_recover_lmp_reduced_kkt`` is an approximate diagnostic
    from PF Jacobian stationarity + bus-level marginal-cost matching.
    It is **not** the full NLP KKT multiplier (no branch / ALM
    inequality duals). Do not compare pointwise to NLP solver output.

Interface (unchanged from the previous AL-based version)::

    prepare_acopf(case) -> ACOPFSetup
    ac_opf(setup, ...)  -> ACOPFResult
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import jax

import jax.numpy as jnp
import jax.lax as lax
import chex
from flax import struct

from jaxopt import LBFGSB

from powerzoojax.case.case_data import CaseData
from powerzoojax.envs.grid.ac_power_flow import (
    branch_admittances,
    build_ybus,
)

# Last slack-balance ALM scalars from the most recent eager ``ac_opf`` execution path
# (``jax.debug.callback`` under ``jit``/``vmap``: last batch element wins; not thread-safe).
_last_acopf_alm_balance_state: dict[str, float] | None = None


def _store_last_acopf_alm_balance_state_host(
    lam_bal: np.ndarray, rho_bal: np.ndarray, h_bal_pu: np.ndarray
) -> None:
    """Host callback: updates ``_last_acopf_alm_balance_state`` with concrete scalars."""
    global _last_acopf_alm_balance_state
    _last_acopf_alm_balance_state = {
        "lam_bal": float(np.asarray(lam_bal)),
        "rho_bal": float(np.asarray(rho_bal)),
        "h_bal_pu": float(np.asarray(h_bal_pu)),
    }


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@struct.dataclass
class ACOPFSetup:
    """Precomputed AC-OPF data including bus-type classification."""

    # Admittance matrix (n_bus × n_bus)
    G: chex.Array
    B: chex.Array
    # Generator-bus incidence matrix (n_bus × n_gen)
    Cg: chex.Array
    # Nominal loads [p.u.]
    Pd: chex.Array
    Qd: chex.Array
    # Generator cost: sum_i (cost_a_i·P_i² + cost_c_i·P_i) in MW — matches PowerZoo
    # ``acopf_solver`` (mc_b is not used in AC-OPF objective there).
    cost_a: chex.Array
    cost_b: chex.Array
    cost_c: chex.Array
    # Generator active / reactive power limits [p.u.]
    Pg_min: chex.Array
    Pg_max: chex.Array
    Qg_min: chex.Array
    Qg_max: chex.Array
    # Nodal voltage magnitude limits [p.u.]
    Vm_min: chex.Array
    Vm_max: chex.Array
    # Static topology (pytree_node=False → baked into JIT compiled code)
    ref_bus: int = struct.field(pytree_node=False, default=0)
    n_bus: int = struct.field(pytree_node=False, default=0)
    n_gen: int = struct.field(pytree_node=False, default=0)
    base_mva: float = struct.field(pytree_node=False, default=100.0)
    # Branch and auxiliary data
    gen_bus_idx: chex.Array = None
    Yff_real: chex.Array = None
    Yff_imag: chex.Array = None
    Yft_real: chex.Array = None
    Yft_imag: chex.Array = None
    Ytf_real: chex.Array = None
    Ytf_imag: chex.Array = None
    Ytt_real: chex.Array = None
    Ytt_imag: chex.Array = None
    br_from_idx: chex.Array = None
    br_to_idx: chex.Array = None
    PTDF: chex.Array = None
    nodes_units_map: chex.Array = None
    # Apparent-power line limit [p.u.]; 0 = unlimited
    line_cap_pu: chex.Array = None

    # ------ Reduced-space bus-type classification (new static fields) ------
    # Counts of PV buses (non-slack buses with generators) and PQ buses
    n_pv: int = struct.field(pytree_node=False, default=0)
    n_pq: int = struct.field(pytree_node=False, default=0)
    # Python tuples of indices (static, baked into JIT)
    pv_bus_idx: tuple = struct.field(pytree_node=False, default=())
    pq_bus_idx: tuple = struct.field(pytree_node=False, default=())
    # non_slack_bus_idx = pv_bus_idx + pq_bus_idx (PV first, then PQ)
    non_slack_bus_idx: tuple = struct.field(pytree_node=False, default=())
    # ctrl_gen_idx: all ``0..n_gen-1`` (every unit is a Pg decision variable)
    ctrl_gen_idx: tuple = struct.field(pytree_node=False, default=())
    # slack_gen_idx: legacy empty tuple (single balance machine removed)
    slack_gen_idx: tuple = struct.field(pytree_node=False, default=())
    # gen_bus_ctrl_idx: Vm control bus ordering = pv_buses + (slack_bus,)
    gen_bus_ctrl_idx: tuple = struct.field(pytree_node=False, default=())
    # Cg submatrix: (n_ns, n_ctrl_gen) = Cg[non_slack_buses, :][:, ctrl_gens]
    # Used in NR mismatch: P_mismatch[non_slack] = P_inj - Cg_ns_ctrl @ Pg_ctrl + Pd
    Cg_ns_ctrl: chex.Array = None
    # Slack-bus row of ``Cg`` (shape ``(n_gen,)``): ``(Cg @ Pg)[slack]``.
    # Used for slack active-balance equality ``h_bal`` in ``ac_opf`` (full
    # generator set — not an abbreviated ``ctrl`` subvector).
    Cg_slack_row: chex.Array = None


@struct.dataclass
class ACOPFResult:
    """Output of AC OPF solver (interface identical to previous AL version)."""
    unit_power: chex.Array      # (n_gen,)  active power [MW]
    q_gen: chex.Array           # (n_gen,)  reactive power [MVAr]
    vm: chex.Array              # (n_bus,)  voltage magnitude [p.u.]
    va: chex.Array              # (n_bus,)  voltage angle [rad]
    line_flow_p: chex.Array     # (n_lines,) active branch flow [MW]
    line_flow_q: chex.Array     # (n_lines,) reactive branch flow [MVAr]
    lmp: chex.Array             # (n_bus,)  nodal LMP [$/MWh]
    total_cost: chex.Array      # scalar    generation cost [$/h]
    converged: chex.Array       # bool      True if L-BFGS-B + NR both converged
    iterations: chex.Array      # int       L-BFGS-B outer iteration count
    line_viol_mva: chex.Array   # scalar    max thermal violation [MVA]; 0 = feasible


@struct.dataclass
class ACOPFDiagnostics:
    """Outer ALM diagnostics (not part of public ``ACOPFResult``; for debugging / tooling).

    Populated at the end of ``ac_opf`` from the final ALM carry; safe under ``jit``.
    """

    objective_trace: chex.Array       # (_n_alm_outer,)  generation cost [$/h] per outer step
    max_viol_trace: chex.Array        # (_n_alm_outer,)  max constraint violation [p.u.]
    pf_residual_trace: chex.Array     # (_n_alm_outer,)  max |PF mismatch|
    max_line_viol: chex.Array           # scalar  final max line thermal viol [p.u.] (sqrt form)
    max_qg_viol: chex.Array             # scalar  final max Qg bound viol [p.u.]
    max_slack_pg_viol: chex.Array       # scalar  final max slack Pg bound viol [p.u.]
    max_viol: chex.Array                # scalar  max of the three above
    alm_converged: chex.Array           # bool    True if early-stop criteria met
    delta_lambda_final: chex.Array      # scalar  last measured max |Δλ| (0 if never updated)


# ---------------------------------------------------------------------------
# Setup-time helpers (NumPy; called once from prepare_acopf)
# ---------------------------------------------------------------------------

def _classify_buses(
    nb: int,
    ng: int,
    slack: int,
    gen_bus: np.ndarray,
):
    """Classify buses into Slack / PV / PQ; **all** generators are Pg controls.

    * PV buses: non-slack buses with ≥1 generator.
    * PQ buses: non-slack buses with no generator (pure load).

    **Generator classification (no single balance machine)**

    Every unit ``0..ng-1`` is a decision variable in ``ac_opf``.  Slack-bus
    active power balance is enforced by an explicit equality
    ``h_bal = 0`` (ALM), not by solving for one ``Pg`` from the slack equation.
    ``slack_gen_idx`` is left empty for backward compatibility of the struct.

    Returns
    -------
    pv_buses, pq_buses, non_slack_buses : sorted tuples of bus indices
    ctrl_gens   : ``tuple(range(ng))`` — all generators
    slack_gens  : empty tuple (legacy field; balance-gen concept removed)
    gen_bus_ctrl: Vm-control ordering = pv_buses + (slack_bus,)
    """
    gen_bus_set = {int(b) for b in gen_bus}
    pv_buses = tuple(sorted(b for b in gen_bus_set if b != slack))
    pq_buses = tuple(sorted(
        b for b in range(nb) if b != slack and b not in gen_bus_set
    ))
    non_slack_buses = pv_buses + pq_buses   # PV first

    ctrl_gens = tuple(range(ng))
    slack_gens = ()

    gen_bus_ctrl = pv_buses + (slack,)
    return pv_buses, pq_buses, non_slack_buses, ctrl_gens, slack_gens, gen_bus_ctrl


def prepare_acopf(
    case: CaseData,
    v_min: float = 0.95,
    v_max: float = 1.05,
    q_factor: float = 0.75,
) -> ACOPFSetup:
    """Prepare AC-OPF data from CaseData (called once, pure NumPy).

    Enables jax_enable_x64 so that float64 arrays created here (and used by
    ``ac_opf``) are preserved through JIT.  This is a global, sticky change.
    Note: ``offer_sced`` uses pure float32 with engineering stabilization and
    does NOT require x64 — only ac_opf needs this setting.
    """
    jax.config.update("jax_enable_x64", True)
    nb = case.n_nodes
    ng = case.n_units
    nl = case.n_lines
    bMVA = case.base_mva

    br_from = np.asarray(case.line_from_idx, dtype=int)
    br_to = np.asarray(case.line_to_idx, dtype=int)
    br_r = np.asarray(case.line_r if case.line_r is not None else np.zeros(nl))
    br_x = np.asarray(case.line_x)
    br_b = np.asarray(case.line_b if case.line_b is not None else np.zeros(nl))
    br_ratio = np.asarray(
        case.line_ratio if case.line_ratio is not None else np.zeros(nl))
    br_angle = np.asarray(
        case.line_angle if case.line_angle is not None else np.zeros(nl))
    br_status = np.asarray(
        case.line_status if case.line_status is not None else np.ones(nl))
    bus_gs = np.asarray(case.node_gs if case.node_gs is not None else np.zeros(nb))
    bus_bs = np.asarray(case.node_bs if case.node_bs is not None else np.zeros(nb))

    Ybus = build_ybus(
        nb, br_from, br_to, br_r, br_x, br_b,
        br_ratio, br_angle, br_status, bus_gs, bus_bs, bMVA)

    gen_bus = np.asarray(case.unit_node_idx, dtype=int)
    Cg = np.zeros((nb, ng))
    Cg[gen_bus, np.arange(ng)] = 1.0

    Pd = np.asarray(case.node_pd) / bMVA
    Qd = np.asarray(case.node_qd) / bMVA

    cost_a = np.asarray(case.unit_cost_a)
    cost_b = np.asarray(case.unit_cost_b)
    cost_c = np.asarray(case.unit_cost_c)

    Pg_min = np.asarray(case.unit_p_min) / bMVA
    Pg_max = np.asarray(case.unit_p_max) / bMVA

    if case.unit_q_min is not None:
        Qg_min = np.asarray(case.unit_q_min) / bMVA
        Qg_max = np.asarray(case.unit_q_max) / bMVA
    else:
        Qg_min = -Pg_max * q_factor
        Qg_max = Pg_max * q_factor

    if case.node_v_min is not None:
        Vm_min = np.asarray(case.node_v_min)
        Vm_max = np.asarray(case.node_v_max)
    else:
        Vm_min = np.full(nb, v_min)
        Vm_max = np.full(nb, v_max)

    Yff, Yft, Ytf, Ytt = branch_admittances(br_r, br_x, br_b, br_ratio, br_angle)

    PTDF = np.asarray(case.PTDF) if case.PTDF is not None else np.zeros((nl, nb))
    A_u = (np.asarray(case.nodes_units_map)
           if case.nodes_units_map is not None else np.zeros((nb, ng)))

    if case.line_cap is not None:
        raw_cap = np.asarray(case.line_cap, dtype=float).copy()
        raw_cap[raw_cap < 0] = 0.0
        raw_cap[raw_cap >= 1e5] = 0.0
        line_cap_pu = raw_cap / bMVA
    else:
        line_cap_pu = np.zeros(nl)

    # Bus-type classification for the reduced-space formulation
    slack = int(case.slack_bus_idx)
    pv_buses, pq_buses, non_slack_buses, ctrl_gens, slack_gens, gen_bus_ctrl = \
        _classify_buses(nb, ng, slack, gen_bus)

    # Cg_ns_ctrl: (n_ns, n_gen) — NR P mismatch at non-slack buses: full ``Cg @ Pg``.
    n_ns = len(non_slack_buses)
    ng_loc = ng
    if n_ns > 0 and ng_loc > 0:
        Cg_ns_ctrl = Cg[np.array(non_slack_buses), :]
    else:
        Cg_ns_ctrl = np.zeros((n_ns, max(ng_loc, 1)))

    # Slack-bus row of Cg (n_gen,) for equality h_bal in ac_opf
    if ng_loc > 0:
        Cg_slack_row_np = Cg[slack, :]
    else:
        Cg_slack_row_np = np.zeros(1)

    return ACOPFSetup(
        G=jnp.array(Ybus.real, dtype=jnp.float64),
        B=jnp.array(Ybus.imag, dtype=jnp.float64),
        Cg=jnp.array(Cg, dtype=jnp.float64),
        Pd=jnp.array(Pd, dtype=jnp.float64),
        Qd=jnp.array(Qd, dtype=jnp.float64),
        cost_a=jnp.array(cost_a, dtype=jnp.float64),
        cost_b=jnp.array(cost_b, dtype=jnp.float64),
        cost_c=jnp.array(cost_c, dtype=jnp.float64),
        Pg_min=jnp.array(Pg_min, dtype=jnp.float64),
        Pg_max=jnp.array(Pg_max, dtype=jnp.float64),
        Qg_min=jnp.array(Qg_min, dtype=jnp.float64),
        Qg_max=jnp.array(Qg_max, dtype=jnp.float64),
        Vm_min=jnp.array(Vm_min, dtype=jnp.float64),
        Vm_max=jnp.array(Vm_max, dtype=jnp.float64),
        ref_bus=slack,
        n_bus=nb,
        n_gen=ng,
        base_mva=float(bMVA),
        gen_bus_idx=jnp.array(gen_bus, dtype=jnp.int32),
        Yff_real=jnp.array(Yff.real, dtype=jnp.float64),
        Yff_imag=jnp.array(Yff.imag, dtype=jnp.float64),
        Yft_real=jnp.array(Yft.real, dtype=jnp.float64),
        Yft_imag=jnp.array(Yft.imag, dtype=jnp.float64),
        Ytf_real=jnp.array(Ytf.real, dtype=jnp.float64),
        Ytf_imag=jnp.array(Ytf.imag, dtype=jnp.float64),
        Ytt_real=jnp.array(Ytt.real, dtype=jnp.float64),
        Ytt_imag=jnp.array(Ytt.imag, dtype=jnp.float64),
        br_from_idx=jnp.array(br_from, dtype=jnp.int32),
        br_to_idx=jnp.array(br_to, dtype=jnp.int32),
        PTDF=jnp.array(PTDF, dtype=jnp.float64),
        nodes_units_map=jnp.array(A_u, dtype=jnp.float64),
        line_cap_pu=jnp.array(line_cap_pu, dtype=jnp.float64),
        n_pv=len(pv_buses),
        n_pq=len(pq_buses),
        pv_bus_idx=pv_buses,
        pq_bus_idx=pq_buses,
        non_slack_bus_idx=non_slack_buses,
        ctrl_gen_idx=ctrl_gens,
        slack_gen_idx=slack_gens,
        gen_bus_ctrl_idx=gen_bus_ctrl,
        Cg_ns_ctrl=jnp.array(Cg_ns_ctrl, dtype=jnp.float64),
        Cg_slack_row=jnp.array(Cg_slack_row_np, dtype=jnp.float64),
    )


# ---------------------------------------------------------------------------
# Runtime helpers (pure JAX; JIT-compilable)
# ---------------------------------------------------------------------------

def _power_injections(Va, Vm, G, B):
    """Compute nodal P and Q injections from voltages and admittance matrix."""
    Vr = Vm * jnp.cos(Va)
    Vi = Vm * jnp.sin(Va)
    Ir = G @ Vr - B @ Vi
    Ii = G @ Vi + B @ Vr
    P = Vr * Ir + Vi * Ii
    Q = Vi * Ir - Vr * Ii
    return P, Q


def _branch_flows(Va, Vm, setup):
    """Compute branch active / reactive power flows [MW, MVAr]."""
    bMVA = setup.base_mva
    f = setup.br_from_idx
    t = setup.br_to_idx
    Vf_r = Vm[f] * jnp.cos(Va[f])
    Vf_i = Vm[f] * jnp.sin(Va[f])
    Vt_r = Vm[t] * jnp.cos(Va[t])
    Vt_i = Vm[t] * jnp.sin(Va[t])
    If_r = (setup.Yff_real * Vf_r - setup.Yff_imag * Vf_i
            + setup.Yft_real * Vt_r - setup.Yft_imag * Vt_i)
    If_i = (setup.Yff_real * Vf_i + setup.Yff_imag * Vf_r
            + setup.Yft_real * Vt_i + setup.Yft_imag * Vt_r)
    pf = (Vf_r * If_r + Vf_i * If_i) * bMVA
    qf = (Vf_i * If_r - Vf_r * If_i) * bMVA
    return pf, qf


def _branch_s2_pu(Va, Vm, setup):
    """Squared apparent power |S|² [p.u.²] at both ends of each branch."""
    f = setup.br_from_idx
    t = setup.br_to_idx
    Vf_r = Vm[f] * jnp.cos(Va[f])
    Vf_i = Vm[f] * jnp.sin(Va[f])
    Vt_r = Vm[t] * jnp.cos(Va[t])
    Vt_i = Vm[t] * jnp.sin(Va[t])
    If_r = (setup.Yff_real * Vf_r - setup.Yff_imag * Vf_i
            + setup.Yft_real * Vt_r - setup.Yft_imag * Vt_i)
    If_i = (setup.Yff_real * Vf_i + setup.Yff_imag * Vf_r
            + setup.Yft_real * Vt_i + setup.Yft_imag * Vt_r)
    Sf2 = (Vf_r * If_r + Vf_i * If_i) ** 2 + (Vf_i * If_r - Vf_r * If_i) ** 2
    It_r = (setup.Ytf_real * Vf_r - setup.Ytf_imag * Vf_i
            + setup.Ytt_real * Vt_r - setup.Ytt_imag * Vt_i)
    It_i = (setup.Ytf_real * Vf_i + setup.Ytf_imag * Vf_r
            + setup.Ytt_real * Vt_i + setup.Ytt_imag * Vt_r)
    St2 = (Vt_r * It_r + Vt_i * It_i) ** 2 + (Vt_i * It_r - Vt_r * It_i) ** 2
    return Sf2, St2


def _recover_lmp_reduced_kkt(
    setup: ACOPFSetup,
    Va: chex.Array,
    Vm: chex.Array,
    Pg: chex.Array,
    Pg_min: chex.Array,
    Pg_max: chex.Array,
) -> chex.Array:
    """Approximate nodal LMP (λ_P) from reduced-space **stationarity** (diagnostic only).

    **Reduced-space AC-OPF** (this module) partitions::

        u = [Pg_ctrl, Vm_ctrl]   — primal controls (L-BFGS-B)
        x = [Va_non_slack, Vm_PQ] — implicit PF state (NR)

    Full NLP KKT would couple u, x, PF equalities, and *all* inequality /
    ALM duals.  This routine does **not** recover those multipliers; it solves
    a **small WLS surrogate** motivated by:

    1. **Dual of nodal injections** (same as before): stationarity of the
       Lagrangian w.r.t. *full* (Va, Vm) implies rows::

           [J_P^T  J_Q^T] [λ_P; λ_Q] = 0   (stacked Va and Vm blocks)

       i.e. ``A_net @ λ = 0`` with λ = [λ_P; λ_Q] in stacked 2n form.

    2. **Bus-level marginal matching** (fixes multi-unit buses): instead of one
       row per generator ``λ_P[bus_i] = MC_i`` (over-constraining when several
       units share a bus), we impose **one row per bus** with generation::

           λ_P[b] = (Σ_{i at b} MC_i) / (#units at b)

       weighted heavily only when **all** units at that bus are in the
       interior of ``[Pg_min, Pg_max]``.  If any unit at bus ``b`` is at a
       bound, that row is down-weighted — a placeholder for **active Pg
       bounds**; a full treatment would add explicit complementarity / KKT
       multipliers for active inequalities.

    **Extension (not implemented)**: branch / Qg / Vm inequality duals from the
    outer ALM would enter additional rows; keep this hook in mind when
    comparing to SLSQP KKT vectors.

    **Important**: treat ``result.lmp`` as an **approximate LMP diagnostic**,
    not interchangeable with SciPy SLSQP constraint multipliers.

    Returns
    -------
    (n_bus,) nodal λ_P in $/MWh (same units as PowerZoo ``lmp`` field).
    """
    nb = setup.n_bus
    ng = setup.n_gen
    bMVA = setup.base_mva
    gi = setup.gen_bus_idx

    Pg_mw = Pg * bMVA
    mc = jnp.float64(2.0) * setup.cost_a * Pg_mw + setup.cost_c

    eps = jnp.float64(0.01) / bMVA
    at_lower = Pg <= Pg_min + eps
    at_upper = Pg >= Pg_max - eps
    at_bound = at_lower | at_upper

    gen_per_bus = jnp.zeros(nb, dtype=jnp.float64).at[gi].add(jnp.float64(1.0))
    bound_per_bus = jnp.zeros(nb, dtype=jnp.float64).at[gi].add(
        jnp.where(at_bound, jnp.float64(1.0), jnp.float64(0.0))
    )
    all_interior_bus = (gen_per_bus > jnp.float64(0.0)) & (
        bound_per_bus == jnp.float64(0.0)
    )

    sum_mc = jnp.zeros(nb, dtype=jnp.float64).at[gi].add(mc)
    mc_bus = jnp.where(
        gen_per_bus > jnp.float64(0.0),
        sum_mc / jnp.maximum(gen_per_bus, jnp.float64(1e-12)),
        jnp.float64(0.0),
    )

    (J_P_Va, J_Q_Va), (J_P_Vm, J_Q_Vm) = jax.jacobian(
        lambda va, vm: _power_injections(va, vm, setup.G, setup.B),
        argnums=(0, 1),
    )(Va, Vm)

    A_net = jnp.concatenate([
        jnp.concatenate([J_P_Va.T, J_Q_Va.T], axis=1),
        jnp.concatenate([J_P_Vm.T, J_Q_Vm.T], axis=1),
    ], axis=0)
    b_net = jnp.zeros(2 * nb)
    w_net = jnp.ones(2 * nb)

    ii = jnp.arange(nb, dtype=jnp.int32)
    has_gen_bus = gen_per_bus > jnp.float64(0.0)
    E_bus = jnp.zeros((nb, 2 * nb), dtype=jnp.float64).at[ii, ii].set(
        jnp.where(has_gen_bus, jnp.float64(1.0), jnp.float64(0.0))
    )
    b_bus = jnp.where(has_gen_bus, mc_bus, jnp.float64(0.0))
    w_bus = jnp.where(
        has_gen_bus & all_interior_bus,
        jnp.float64(1e4),
        jnp.where(has_gen_bus, jnp.float64(1.0), jnp.float64(0.0)),
    )

    A = jnp.concatenate([A_net, E_bus], axis=0)
    b = jnp.concatenate([b_net, b_bus], axis=0)
    w = jnp.concatenate([w_net, w_bus], axis=0)

    Aw = w[:, None] * A
    AtWA = A.T @ Aw
    AtWb = A.T @ (w * b)
    reg = jnp.float64(1e-6) * jnp.eye(2 * nb)

    x_sol = jnp.linalg.solve(AtWA + reg, AtWb)
    return x_sol[:nb]


# Main solver: ac_opf

def ac_opf(
    setup: ACOPFSetup,
    node_load_mw: chex.Array = None,
    node_load_mvar: chex.Array = None,
    commitment: chex.Array = None,
    max_iter: int = 200,
    tol: float = 1e-3,
    warm_start_unit_power_mw: chex.Array = None,
    alm_outer_steps: int = 12,
) -> ACOPFResult:
    """Solve AC-OPF via reduced-space L-BFGS-B with IFT-differentiable NR.

    Args:
        setup: ACOPFSetup from ``prepare_acopf``.
        node_load_mw: (n_bus,) active load [MW]. ``None`` → use setup.Pd.
        node_load_mvar: (n_bus,) reactive load [MVAr]. ``None`` → use setup.Qd.
        commitment: (n_gen,) binary. ``None`` → all units on.
        max_iter: Max L-BFGS-B outer iterations (default 200; typically
            converges in 20–60 iterations for case5).
        tol: L-BFGS-B convergence tolerance.
        warm_start_unit_power_mw: Optional (n_gen,) dispatch guess [MW]
            (e.g. DC-OPF result).  Used to initialise Pg_ctrl.
        alm_outer_steps: Number of augmented-Lagrangian outer iterations
            (``lax.scan`` length).  Default 12.

    Returns:
        ACOPFResult with identical field layout to the previous AL-based solver.
    """
    nb = setup.n_bus
    ng = setup.n_gen
    bMVA = setup.base_mva
    n_pv = setup.n_pv
    n_pq = setup.n_pq
    # n_ns = number of non-slack buses = nb - 1 = n_pv + n_pq
    n_ns = n_pv + n_pq
    # Dimension of control u = [Pg_all (n_gen), Vm_ctrl (n_gen_bus)]
    n_ctrl_gen = ng  # all generators are Pg decision variables
    n_gen_ctrl = len(setup.gen_bus_ctrl_idx)   # gen buses = n_pv + 1
    n_u = n_ctrl_gen + n_gen_ctrl
    # Dimension of NR state x = [Va_non_slack, Vm_PQ]
    n_x = n_ns + n_pq
    ref_bus = setup.ref_bus

    # Compile-time index arrays (treated as constants under JIT)
    ns_arr = jnp.array(setup.non_slack_bus_idx, dtype=jnp.int32)
    pv_arr = jnp.array(setup.pv_bus_idx, dtype=jnp.int32)
    pq_arr = jnp.array(setup.pq_bus_idx, dtype=jnp.int32)
    gcb_arr = jnp.array(setup.gen_bus_ctrl_idx, dtype=jnp.int32)

    # Regularisation identity for Jacobian solve
    I_n = jnp.eye(n_x, dtype=jnp.float64)

    # Load preprocessing
    Pd = node_load_mw / bMVA if node_load_mw is not None else setup.Pd
    Qd = node_load_mvar / bMVA if node_load_mvar is not None else setup.Qd

    # Generator limits (adjusted for unit commitment)
    Pg_min = setup.Pg_min
    Pg_max = setup.Pg_max
    Qg_min_all = setup.Qg_min
    Qg_max_all = setup.Qg_max
    if commitment is not None:
        Pg_min = Pg_min * commitment
        Pg_max = Pg_max * commitment
        Qg_min_all = Qg_min_all * commitment
        Qg_max_all = Qg_max_all * commitment

    # Bus-level Q limits: sum of per-unit Qg bounds on each bus (matches equal-split
    # reporting: total bus Q is one PF quantity; per-gen ALM duplicates were inconsistent).
    gi = setup.gen_bus_idx
    Qg_bus_max_agg = jnp.zeros(nb, dtype=jnp.float64).at[gi].add(Qg_max_all)
    Qg_bus_min_agg = jnp.zeros(nb, dtype=jnp.float64).at[gi].add(Qg_min_all)
    has_gen_bus = jnp.zeros(nb, dtype=jnp.float64).at[gi].add(jnp.float64(1.0)) > jnp.float64(0.0)

    # Decision u = [Pg_all, Vm_ctrl].  Slack-bus active balance: h_bal=0 (ALM), not substitution.
    # -------------------------------------------------------------------------

    cg_idx = jnp.arange(ng, dtype=jnp.int32)

    n_u = n_ctrl_gen + n_gen_ctrl

    lb = jnp.concatenate([Pg_min, setup.Vm_min[gcb_arr]])
    ub = jnp.concatenate([Pg_max, setup.Vm_max[gcb_arr]])

    if warm_start_unit_power_mw is not None:
        ws_pu = jnp.asarray(warm_start_unit_power_mw, dtype=jnp.float64) / bMVA
        Pg0_all = jnp.clip(ws_pu, Pg_min, Pg_max)
    else:
        total_Pd = jnp.sum(Pd)
        safe_cap = jnp.maximum(jnp.sum(Pg_max), jnp.float64(1e-8))
        frac = jnp.clip(total_Pd / safe_cap, jnp.float64(0.0), jnp.float64(1.0))
        Pg0_all = jnp.clip(
            Pg_min + (Pg_max - Pg_min) * frac,
            Pg_min,
            Pg_max,
        )
    Vm0_ctrl = jnp.ones(n_gen_ctrl, dtype=jnp.float64)
    u0 = jnp.clip(jnp.concatenate([Pg0_all, Vm0_ctrl]), lb, ub)

    # -------------------------------------------------------------------------
    # Module A — Mismatch function and Newton-Raphson solver
    # -------------------------------------------------------------------------

    def _mismatch(x_state, u_ctrl):
        """AC power flow mismatch g(x, u) = 0  →  vector of shape (n_x,).

        x_state: (n_x,) = [Va_non_slack (n_ns), Vm_pq (n_pq)]
        u_ctrl:  (n_u,) = [Pg_all (n_gen), Vm_ctrl (n_gen_bus)]

        Non-slack P balance uses full ``Cg @ Pg`` at non-slack rows.  Slack-bus P
        is **not** in NR; it is enforced by ``h_bal = 0`` in the outer objective.
        """
        Va_ns = x_state[:n_ns]
        Vm_pq = x_state[n_ns:]         # empty when n_pq == 0

        Pg_all = u_ctrl[:n_ctrl_gen]
        Vm_ctrl = u_ctrl[n_ctrl_gen:]  # Vm for gen buses (PV then slack)

        Va = jnp.zeros(nb).at[ns_arr].set(Va_ns)
        Vm = (
            jnp.ones(nb)
            .at[pv_arr].set(Vm_ctrl[:n_pv])
            .at[ref_bus].set(Vm_ctrl[n_pv])
        )
        if n_pq > 0:
            Vm = Vm.at[pq_arr].set(Vm_pq)

        P_inj, Q_inj = _power_injections(Va, Vm, setup.G, setup.B)

        Cg_Pg_ns = setup.Cg_ns_ctrl @ Pg_all
        g_P = P_inj[ns_arr] - Cg_Pg_ns + Pd[ns_arr]

        if n_pq > 0:
            g_Q = Q_inj[pq_arr] + Qd[pq_arr]
            return jnp.concatenate([g_P, g_Q])
        return g_P

    def _nr_raw(u_ctrl):
        """Damped Newton-Raphson: returns (x_converged, final_residual, n_iters).

        Uses ``lax.while_loop`` (JIT/vmap safe). Jacobian is computed via
        ``jax.jacobian`` (automatic differentiation of _mismatch w.r.t. x).
        A small diagonal shift (1e-7·I) regularises near-singular Jacobians.
        Step damping avoids divergence when starting far from the solution.
        """
        # Flat start: Va = 0 for all non-slack buses, Vm = 1.0 for PQ buses
        x0 = jnp.concatenate([jnp.zeros(n_ns), jnp.ones(n_pq)])

        def body(state):
            x, res, it = state
            g = _mismatch(x, u_ctrl)
            Jx = jax.jacobian(_mismatch, argnums=0)(x, u_ctrl)
            dx = jax.scipy.linalg.solve(
                Jx + jnp.float64(1e-7) * I_n, -g)
            # Limit step size to improve robustness far from the solution
            step = jnp.minimum(
                jnp.float64(1.0),
                jnp.float64(1.0) / (jnp.linalg.norm(dx) + jnp.float64(1e-8)))
            return x + step * dx, jnp.max(jnp.abs(g)), it + jnp.int32(1)

        def cond(state):
            _, res, it = state
            return (res > jnp.float64(1e-6)) & (it < jnp.int32(30))

        return lax.while_loop(
            cond, body, (x0, jnp.float64(jnp.inf), jnp.int32(0)))

    # -------------------------------------------------------------------------
    # Module B — IFT custom_vjp: exact gradients via implicit function theorem
    #
    # Direct AD through while_loop would:
    #   (a) expand to 30× the computation graph (memory explosion)
    #   (b) accumulate floating-point error across iterations
    #   (c) give misleading gradients when NR doesn't fully converge
    #
    # IFT avoids all three: solve ONE adjoint linear system at the converged
    # point, giving O(n²) exact sensitivity at the price of a second Jx solve.
    # -------------------------------------------------------------------------

    @jax.custom_vjp
    def pf_solve(u_ctrl):
        """Power flow solve: given u, return converged state x = [Va_ns, Vm_pq].

        The gradient of any downstream computation w.r.t. u is computed by
        the IFT rule defined in ``_pf_fwd`` / ``_pf_bwd`` below.
        """
        x, _, _ = _nr_raw(u_ctrl)
        return x

    def _pf_fwd(u_ctrl):
        """Forward pass: run NR and stash Jacobians as residuals for IFT bwd."""
        x, _, _ = _nr_raw(u_ctrl)
        # Evaluate Jacobians at the converged point (single evaluation, no loop)
        Jx = jax.jacobian(_mismatch, argnums=0)(x, u_ctrl)  # (n_x, n_x)
        Ju = jax.jacobian(_mismatch, argnums=1)(x, u_ctrl)  # (n_x, n_u)
        return x, (Jx, Ju)

    def _pf_bwd(res, x_bar):
        """Backward pass (IFT adjoint):
            Jx^T λ = x_bar    (adjoint equation)
            ū = −λ^T Ju       (control sensitivity)
        """
        Jx, Ju = res
        # Small diagonal shift for numerical stability (does not affect
        # gradients significantly when NR converged tightly)
        lam = jax.scipy.linalg.solve(
            Jx.T + jnp.float64(1e-6) * I_n, x_bar)
        return (-lam @ Ju,)   # (n_u,) — cotangent of u_ctrl

    pf_solve.defvjp(_pf_fwd, _pf_bwd)

    # -------------------------------------------------------------------------
    # Module C — Augmented Lagrangian (PHR) + L-BFGS-B
    #
    # Slack-bus **active** balance: scalar equality ``h_bal = 0`` via
    # ``λ·h + (ρ/2)·h²`` (not PHR).  All ``Pg`` are box-bounded; no single
    # balance machine absorbs losses.
    # Line limits: inequality and penalty use **squared** apparent power
    #   Sf² − Smax² ≤ 0 (p.u.²)  (``line_cap_pu == 0`` → unlimited, unchanged).
    # Final reported ``line_viol_mva`` still uses √(Sf²) − Smax in MVA.
    #
    # Outer ALM uses ``lax.scan`` with adaptive ρ (freeze when violation drops
    # quickly), early stop when primal/dual tolerances are met, and per-step
    # traces (objective, max violation, PF residual).  After ``done``, inner
    # L-BFGS-B is skipped via ``lax.cond`` while traces are padded.
    # -------------------------------------------------------------------------

    nl = int(setup.line_cap_pu.shape[0])

    def _alm_ineq_sum(rho: jnp.ndarray, lam: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
        """Sum of PHR ALM terms for vector inequality constraints h ≤ 0."""
        return jnp.sum(
            (jnp.float64(1.0) / (jnp.float64(2.0) * rho))
            * (jnp.maximum(jnp.float64(0.0), lam + rho * h) ** 2 - lam ** 2)
        )

    def _alm_eq_scalar(lam: jnp.ndarray, rho: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
        """Scalar equality h = 0: augmented term ``λ·h + (ρ/2)·h²`` (minimisation)."""
        return lam * h + (jnp.float64(0.5) * rho) * h ** 2

    def _reduced_obj(
        u_ctrl,
        rho_line,
        rho_Qg,
        rho_bal,
        rho_vm,
        lam_line_f,
        lam_line_t,
        lam_Qg_max,
        lam_Qg_min,
        lam_bal,
        lam_vm_pq_max,
        lam_vm_pq_min,
    ):
        """Scalar ALM objective for L-BFGS-B; slack P balance equality + inequalities."""
        Pg_all = u_ctrl[:n_ctrl_gen]
        Vm_ctrl = u_ctrl[n_ctrl_gen:]

        x = pf_solve(u_ctrl)
        Va_ns = x[:n_ns]

        Va = jnp.zeros(nb).at[ns_arr].set(Va_ns)
        Vm = (
            jnp.ones(nb)
            .at[pv_arr].set(Vm_ctrl[:n_pv])
            .at[ref_bus].set(Vm_ctrl[n_pv])
        )
        if n_pq > 0:
            Vm = Vm.at[pq_arr].set(x[n_ns:])

        P_inj, Q_inj = _power_injections(Va, Vm, setup.G, setup.B)

        Pg_all_mw = Pg_all * bMVA
        # Same objective as PowerZoo ``acopf_solver``: sum(a P^2 + c P); mc_b unused.
        cost = jnp.sum(
            setup.cost_a * Pg_all_mw ** 2 + setup.cost_c * Pg_all_mw
        )

        h_bal = (
            P_inj[ref_bus]
            - jnp.dot(setup.Cg_slack_row, Pg_all)
            + Pd[ref_bus]
        )
        pen_bal = _alm_eq_scalar(lam_bal, rho_bal, h_bal)

        # Line thermal limits in **squared apparent power** form (p.u.²):
        #   Sf2 - Smax² ≤ 0,  St2 - Smax² ≤ 0
        # This avoids sqrt (nondifferentiable at 0) and is smoother for L-BFGS-B + PHR-ALM
        # than sqrt(Sf2) - Smax ≤ 0.  Reporting ``line_viol_mva`` still uses sqrt(S) - Smax.
        has_cap = setup.line_cap_pu > jnp.float64(0.0)
        Sf2, St2 = _branch_s2_pu(Va, Vm, setup)
        Smax = setup.line_cap_pu
        Smax2 = Smax ** 2
        h_line_f = jnp.where(has_cap, Sf2 - Smax2, jnp.float64(0.0))
        h_line_t = jnp.where(has_cap, St2 - Smax2, jnp.float64(0.0))
        pen_line = _alm_ineq_sum(rho_line, lam_line_f, h_line_f) + _alm_ineq_sum(
            rho_line, lam_line_t, h_line_t
        )

        Qg_bus = Q_inj + Qd
        h_Qg_max = jnp.where(
            has_gen_bus, Qg_bus - Qg_bus_max_agg, jnp.float64(0.0))
        h_Qg_min = jnp.where(
            has_gen_bus, Qg_bus_min_agg - Qg_bus, jnp.float64(0.0))
        pen_Qg = _alm_ineq_sum(rho_Qg, lam_Qg_max, h_Qg_max) + _alm_ineq_sum(
            rho_Qg, lam_Qg_min, h_Qg_min
        )

        if n_pq > 0:
            Vm_pq = x[n_ns:]
            h_vm_pq_max = Vm_pq - setup.Vm_max[pq_arr]
            h_vm_pq_min = setup.Vm_min[pq_arr] - Vm_pq
            pen_vm = _alm_ineq_sum(
                rho_vm, lam_vm_pq_max, h_vm_pq_max
            ) + _alm_ineq_sum(rho_vm, lam_vm_pq_min, h_vm_pq_min)
        else:
            pen_vm = jnp.float64(0.0)

        return cost + pen_bal + pen_line + pen_Qg + pen_vm

    lam_line_f0 = jnp.zeros(nl, dtype=jnp.float64)
    lam_line_t0 = jnp.zeros(nl, dtype=jnp.float64)
    lam_Qg_max0 = jnp.zeros(nb, dtype=jnp.float64)
    lam_Qg_min0 = jnp.zeros(nb, dtype=jnp.float64)
    lam_bal0 = jnp.float64(0.0)
    lam_vm_pq_max0 = jnp.zeros(n_pq, dtype=jnp.float64)
    lam_vm_pq_min0 = jnp.zeros(n_pq, dtype=jnp.float64)

    rho_line0 = jnp.float64(1e3)
    rho_Qg0 = jnp.float64(1e3)
    rho_bal0 = jnp.float64(1e3)
    rho_vm0 = jnp.float64(1e3)

    inner_max = max(150, max_iter // 2)
    solver = LBFGSB(fun=_reduced_obj, maxiter=inner_max, tol=float(tol))

    _n_alm_outer = int(alm_outer_steps)

    # Outer ALM early-stop tolerances (p.u. / $ scales); JIT-safe scalars.
    feas_tol = jnp.float64(1e-4)
    obj_tol = jnp.float64(1e-3)
    dual_tol = jnp.float64(1e-2)

    z_trace = jnp.zeros(_n_alm_outer, dtype=jnp.float64)
    init_carry = (
        u0,
        lam_line_f0,
        lam_line_t0,
        lam_Qg_max0,
        lam_Qg_min0,
        lam_bal0,
        lam_vm_pq_max0,
        lam_vm_pq_min0,
        rho_line0,
        rho_Qg0,
        rho_bal0,
        rho_vm0,
        jnp.int32(0),
        jnp.array(True, dtype=jnp.bool_),
        jnp.inf,  # obj_prev
        jnp.inf,  # prev_max_viol
        jnp.array(False, dtype=jnp.bool_),  # done
        jnp.float64(0.0),  # last_obj (for trace padding when ALM finished early)
        jnp.float64(0.0),  # last_max_viol
        jnp.float64(0.0),  # last_pf_res
        z_trace,  # objective_trace
        z_trace,  # max_viol_trace
        z_trace,  # pf_residual_trace
        jnp.float64(0.0),  # delta_lambda_last
    )

    def _dual_delta_max(
        lam_line_f_a,
        lam_line_t_a,
        lam_Qg_max_a,
        lam_Qg_min_a,
        lam_bal_a,
        lam_vm_pq_max_a,
        lam_vm_pq_min_a,
        lam_line_f_b,
        lam_line_t_b,
        lam_Qg_max_b,
        lam_Qg_min_b,
        lam_bal_b,
        lam_vm_pq_max_b,
        lam_vm_pq_min_b,
    ) -> jnp.ndarray:
        """Max |Δλ| over all ALM multipliers (JIT-safe; handles nl==0)."""
        parts = [jnp.abs(lam_bal_b - lam_bal_a)]
        if nl > 0:
            parts.append(jnp.max(jnp.abs(lam_line_f_b - lam_line_f_a)))
            parts.append(jnp.max(jnp.abs(lam_line_t_b - lam_line_t_a)))
        if n_pq > 0:
            parts.append(jnp.max(jnp.abs(lam_vm_pq_max_b - lam_vm_pq_max_a)))
            parts.append(jnp.max(jnp.abs(lam_vm_pq_min_b - lam_vm_pq_min_a)))
        parts.append(jnp.max(jnp.abs(lam_Qg_max_b - lam_Qg_max_a)))
        parts.append(jnp.max(jnp.abs(lam_Qg_min_b - lam_Qg_min_a)))
        return jnp.max(jnp.stack(parts))

    def _alm_step(carry, scan_i):
        (
            u_curr,
            lam_line_f,
            lam_line_t,
            lam_Qg_max,
            lam_Qg_min,
            lam_bal,
            lam_vm_pq_max,
            lam_vm_pq_min,
            rho_line,
            rho_Qg,
            rho_bal,
            rho_vm,
            n_iters,
            prev_failed_ls,
            obj_prev,
            prev_max_viol,
            done_flag,
            last_obj,
            last_max_viol,
            last_pf_res,
            obj_trace,
            max_viol_trace,
            pf_res_trace,
            delta_lambda_last,
        ) = carry

        def _step_skip(c):
            """ALM already converged: skip L-BFGS-B, freeze carry, pad traces."""
            (
                u_curr,
                lam_line_f,
                lam_line_t,
                lam_Qg_max,
                lam_Qg_min,
                lam_bal,
                lam_vm_pq_max,
                lam_vm_pq_min,
                rho_line,
                rho_Qg,
                rho_bal,
                rho_vm,
                n_iters,
                prev_failed_ls,
                obj_prev,
                prev_max_viol,
                done_flag,
                last_obj,
                last_max_viol,
                last_pf_res,
                obj_trace,
                max_viol_trace,
                pf_res_trace,
                delta_lambda_last,
            ) = c
            obj_trace_n = obj_trace.at[scan_i].set(last_obj)
            max_viol_trace_n = max_viol_trace.at[scan_i].set(last_max_viol)
            pf_res_trace_n = pf_res_trace.at[scan_i].set(last_pf_res)
            return (
                u_curr,
                lam_line_f,
                lam_line_t,
                lam_Qg_max,
                lam_Qg_min,
                lam_bal,
                lam_vm_pq_max,
                lam_vm_pq_min,
                rho_line,
                rho_Qg,
                rho_bal,
                rho_vm,
                n_iters,
                prev_failed_ls,
                obj_prev,
                prev_max_viol,
                done_flag,
                last_obj,
                last_max_viol,
                last_pf_res,
                obj_trace_n,
                max_viol_trace_n,
                pf_res_trace_n,
                delta_lambda_last,
            )

        def _step_active(c):
            """One PHR-ALM outer iteration: L-BFGS-B on reduced objective, then λ update."""
            (
                u_curr,
                lam_line_f,
                lam_line_t,
                lam_Qg_max,
                lam_Qg_min,
                lam_bal,
                lam_vm_pq_max,
                lam_vm_pq_min,
                rho_line,
                rho_Qg,
                rho_bal,
                rho_vm,
                n_iters,
                prev_failed_ls,
                obj_prev,
                prev_max_viol,
                done_prev,
                last_obj,
                last_max_viol,
                last_pf_res,
                obj_trace,
                max_viol_trace,
                pf_res_trace,
                _delta_unused,
            ) = c

            opt_result = solver.run(
                u_curr,
                bounds=(lb, ub),
                rho_line=rho_line,
                rho_Qg=rho_Qg,
                rho_bal=rho_bal,
                rho_vm=rho_vm,
                lam_line_f=lam_line_f,
                lam_line_t=lam_line_t,
                lam_Qg_max=lam_Qg_max,
                lam_Qg_min=lam_Qg_min,
                lam_bal=lam_bal,
                lam_vm_pq_max=lam_vm_pq_max,
                lam_vm_pq_min=lam_vm_pq_min,
            )
            u_next = opt_result.params
            n_iters_next = n_iters + opt_result.state.iter_num
            last_failed_linesearch = opt_result.state.failed_linesearch

            x_v = pf_solve(u_next)
            Va_ns_v = x_v[:n_ns]
            Va_v = jnp.zeros(nb).at[ns_arr].set(Va_ns_v)
            Pg_all_v = u_next[:n_ctrl_gen]
            Vm_ctrl_v = u_next[n_ctrl_gen:]
            Vm_v = (
                jnp.ones(nb)
                .at[pv_arr].set(Vm_ctrl_v[:n_pv])
                .at[ref_bus].set(Vm_ctrl_v[n_pv])
            )
            if n_pq > 0:
                Vm_v = Vm_v.at[pq_arr].set(x_v[n_ns:])
            P_inj_v, Q_inj_v = _power_injections(Va_v, Vm_v, setup.G, setup.B)
            Pg_all_mw_v = Pg_all_v * bMVA
            objective_value = jnp.sum(
                setup.cost_a * Pg_all_mw_v ** 2 + setup.cost_c * Pg_all_mw_v
            )

            h_bal_v = (
                P_inj_v[ref_bus]
                - jnp.dot(setup.Cg_slack_row, Pg_all_v)
                + Pd[ref_bus]
            )
            viol_pg_up = jnp.max(jnp.maximum(Pg_all_v - Pg_max, jnp.float64(0.0)))
            viol_pg_lo = jnp.max(jnp.maximum(Pg_min - Pg_all_v, jnp.float64(0.0)))
            max_pg_viol = jnp.maximum(viol_pg_up, viol_pg_lo)

            Sf2_v, St2_v = _branch_s2_pu(Va_v, Vm_v, setup)
            Sf_pu_v = jnp.sqrt(jnp.maximum(Sf2_v, jnp.float64(0.0)))
            St_pu_v = jnp.sqrt(jnp.maximum(St2_v, jnp.float64(0.0)))
            has_cap = setup.line_cap_pu > jnp.float64(0.0)
            Smax2 = setup.line_cap_pu ** 2
            # Multiplier update uses the **same** squared line constraints as ``_reduced_obj``.
            h_line_f_alm = jnp.where(has_cap, Sf2_v - Smax2, jnp.float64(0.0))
            h_line_t_alm = jnp.where(has_cap, St2_v - Smax2, jnp.float64(0.0))

            # Diagnostics / feasibility: sqrt-form line violation [p.u.] (matches ``line_viol_mva``).
            if nl > 0:
                viol_f = jnp.where(
                    has_cap,
                    jnp.maximum(jnp.float64(0.0), Sf_pu_v - setup.line_cap_pu),
                    jnp.float64(0.0),
                )
                viol_t = jnp.where(
                    has_cap,
                    jnp.maximum(jnp.float64(0.0), St_pu_v - setup.line_cap_pu),
                    jnp.float64(0.0),
                )
                max_line_viol = jnp.maximum(jnp.max(viol_f), jnp.max(viol_t))
            else:
                max_line_viol = jnp.float64(0.0)

            Qg_bus_v = Q_inj_v + Qd
            h_Qg_max = jnp.where(
                has_gen_bus, Qg_bus_v - Qg_bus_max_agg, jnp.float64(0.0))
            h_Qg_min = jnp.where(
                has_gen_bus, Qg_bus_min_agg - Qg_bus_v, jnp.float64(0.0))
            viol_q_up = jnp.maximum(jnp.float64(0.0), h_Qg_max)
            viol_q_lo = jnp.maximum(jnp.float64(0.0), h_Qg_min)
            max_qg_viol = jnp.maximum(jnp.max(viol_q_up), jnp.max(viol_q_lo))

            max_viol = jnp.maximum(
                max_line_viol,
                jnp.maximum(
                    max_qg_viol,
                    jnp.maximum(jnp.abs(h_bal_v), max_pg_viol),
                ),
            )

            g_pf = _mismatch(x_v, u_next)
            pf_res = jnp.max(jnp.abs(g_pf))

            lam_line_f_n = jnp.maximum(
                jnp.float64(0.0), lam_line_f + rho_line * h_line_f_alm
            )
            lam_line_t_n = jnp.maximum(
                jnp.float64(0.0), lam_line_t + rho_line * h_line_t_alm
            )
            lam_Qg_max_n = jnp.maximum(jnp.float64(0.0), lam_Qg_max + rho_Qg * h_Qg_max)
            lam_Qg_min_n = jnp.maximum(jnp.float64(0.0), lam_Qg_min + rho_Qg * h_Qg_min)
            lam_bal_n = lam_bal + rho_bal * h_bal_v

            if n_pq > 0:
                Vm_pq_v = x_v[n_ns:]
                h_vm_pq_max = Vm_pq_v - setup.Vm_max[pq_arr]
                h_vm_pq_min = setup.Vm_min[pq_arr] - Vm_pq_v
                lam_vm_pq_max_n = jnp.maximum(
                    jnp.float64(0.0), lam_vm_pq_max + rho_vm * h_vm_pq_max
                )
                lam_vm_pq_min_n = jnp.maximum(
                    jnp.float64(0.0), lam_vm_pq_min + rho_vm * h_vm_pq_min
                )
            else:
                lam_vm_pq_max_n = lam_vm_pq_max
                lam_vm_pq_min_n = lam_vm_pq_min

            delta_lambda = _dual_delta_max(
                lam_line_f,
                lam_line_t,
                lam_Qg_max,
                lam_Qg_min,
                lam_bal,
                lam_vm_pq_max,
                lam_vm_pq_min,
                lam_line_f_n,
                lam_line_t_n,
                lam_Qg_max_n,
                lam_Qg_min_n,
                lam_bal_n,
                lam_vm_pq_max_n,
                lam_vm_pq_min_n,
            )

            obj_ok = (scan_i > jnp.int32(0)) & (
                jnp.abs(objective_value - obj_prev) < obj_tol
            )
            alm_done_new = (
                (max_viol < feas_tol)
                & obj_ok
                & (delta_lambda < dual_tol)
            )

            rho_factor = jnp.float64(2.0)
            rho_cap = jnp.float64(5e5)
            improve = (scan_i > jnp.int32(0)) & (
                max_viol < jnp.float64(0.3) * prev_max_viol
            )
            rho_line_n = jnp.where(
                scan_i == jnp.int32(0),
                jnp.minimum(rho_line * rho_factor, rho_cap),
                jnp.where(
                    improve,
                    rho_line,
                    jnp.minimum(rho_line * rho_factor, rho_cap),
                ),
            )
            rho_Qg_n = jnp.where(
                scan_i == jnp.int32(0),
                jnp.minimum(rho_Qg * rho_factor, rho_cap),
                jnp.where(
                    improve,
                    rho_Qg,
                    jnp.minimum(rho_Qg * rho_factor, rho_cap),
                ),
            )
            rho_bal_n = jnp.where(
                scan_i == jnp.int32(0),
                jnp.minimum(rho_bal * rho_factor, rho_cap),
                jnp.where(
                    improve,
                    rho_bal,
                    jnp.minimum(rho_bal * rho_factor, rho_cap),
                ),
            )
            rho_vm_n = jnp.where(
                scan_i == jnp.int32(0),
                jnp.minimum(rho_vm * rho_factor, rho_cap),
                jnp.where(
                    improve,
                    rho_vm,
                    jnp.minimum(rho_vm * rho_factor, rho_cap),
                ),
            )

            obj_prev_new = objective_value
            prev_max_viol_new = max_viol
            done_new = jnp.logical_or(done_prev, alm_done_new)

            obj_trace_n = obj_trace.at[scan_i].set(objective_value)
            max_viol_trace_n = max_viol_trace.at[scan_i].set(max_viol)
            pf_res_trace_n = pf_res_trace.at[scan_i].set(pf_res)

            return (
                u_next,
                lam_line_f_n,
                lam_line_t_n,
                lam_Qg_max_n,
                lam_Qg_min_n,
                lam_bal_n,
                lam_vm_pq_max_n,
                lam_vm_pq_min_n,
                rho_line_n,
                rho_Qg_n,
                rho_bal_n,
                rho_vm_n,
                n_iters_next,
                last_failed_linesearch,
                obj_prev_new,
                prev_max_viol_new,
                done_new,
                objective_value,
                max_viol,
                pf_res,
                obj_trace_n,
                max_viol_trace_n,
                pf_res_trace_n,
                delta_lambda,
            )

        carry_after = lax.cond(done_flag, _step_skip, _step_active, carry)
        return carry_after, None

    final_carry, _ = lax.scan(_alm_step, init_carry, jnp.arange(_n_alm_outer))

    (
        u_opt,
        _lam_line_f,
        _lam_line_t,
        _lam_Qg_max,
        _lam_Qg_min,
        _lam_bal,
        _lam_vm_pq_max,
        _lam_vm_pq_min,
        _rho_line,
        _rho_qg,
        _rho_bal,
        _rho_vm,
        n_iters,
        last_failed_linesearch,
        _obj_prev_f,
        _prev_max_viol_f,
        alm_done_flag,
        _last_obj_f,
        _last_max_viol_f,
        _last_pf_res_f,
        objective_trace,
        max_viol_trace,
        pf_residual_trace,
        delta_lambda_final,
    ) = final_carry

    # Pack diagnostics (internal only; ``ACOPFResult`` unchanged). Uncomment to print under jit:
    # jax.debug.print("alm traces obj={}", objective_trace)
    x_diag = pf_solve(u_opt)
    Va_ns_diag = x_diag[:n_ns]
    Pg_all_d = u_opt[:n_ctrl_gen]
    Vm_ctrl_d = u_opt[n_ctrl_gen:]
    Va_diag = jnp.zeros(nb).at[ns_arr].set(Va_ns_diag)
    Vm_diag = (
        jnp.ones(nb)
        .at[pv_arr].set(Vm_ctrl_d[:n_pv])
        .at[ref_bus].set(Vm_ctrl_d[n_pv])
    )
    if n_pq > 0:
        Vm_diag = Vm_diag.at[pq_arr].set(x_diag[n_ns:])
    P_inj_d, Q_inj_d = _power_injections(Va_diag, Vm_diag, setup.G, setup.B)
    Sf2_d, St2_d = _branch_s2_pu(Va_diag, Vm_diag, setup)
    Sf_pu_d = jnp.sqrt(jnp.maximum(Sf2_d, jnp.float64(0.0)))
    St_pu_d = jnp.sqrt(jnp.maximum(St2_d, jnp.float64(0.0)))
    has_cap_d = setup.line_cap_pu > jnp.float64(0.0)
    if nl > 0:
        viol_f_d = jnp.where(
            has_cap_d,
            jnp.maximum(jnp.float64(0.0), Sf_pu_d - setup.line_cap_pu),
            jnp.float64(0.0),
        )
        viol_t_d = jnp.where(
            has_cap_d,
            jnp.maximum(jnp.float64(0.0), St_pu_d - setup.line_cap_pu),
            jnp.float64(0.0),
        )
        d_max_line = jnp.maximum(jnp.max(viol_f_d), jnp.max(viol_t_d))
    else:
        d_max_line = jnp.float64(0.0)
    Qg_bus_d = Q_inj_d + Qd
    h_Qg_max_d = jnp.where(
        has_gen_bus, Qg_bus_d - Qg_bus_max_agg, jnp.float64(0.0))
    h_Qg_min_d = jnp.where(
        has_gen_bus, Qg_bus_min_agg - Qg_bus_d, jnp.float64(0.0))
    d_max_qg = jnp.maximum(
        jnp.max(jnp.maximum(jnp.float64(0.0), h_Qg_max_d)),
        jnp.max(jnp.maximum(jnp.float64(0.0), h_Qg_min_d)),
    )
    d_max_slack = jnp.maximum(
        jnp.max(jnp.maximum(Pg_all_d - Pg_max, jnp.float64(0.0))),
        jnp.max(jnp.maximum(Pg_min - Pg_all_d, jnp.float64(0.0))),
    )
    d_max_viol = jnp.maximum(d_max_line, jnp.maximum(d_max_qg, d_max_slack))
    _acopf_diagnostics = ACOPFDiagnostics(
        objective_trace=objective_trace,
        max_viol_trace=max_viol_trace,
        pf_residual_trace=pf_residual_trace,
        max_line_viol=d_max_line,
        max_qg_viol=d_max_qg,
        max_slack_pg_viol=d_max_slack,
        max_viol=d_max_viol,
        alm_converged=alm_done_flag,
        delta_lambda_final=delta_lambda_final,
    )
    # Internal: inspect ``_acopf_diagnostics`` in a debugger, or temporarily return it.
    _ = _acopf_diagnostics

    # -------------------------------------------------------------------------
    # Extract full primal: u = [Pg_all, Vm_ctrl]
    # -------------------------------------------------------------------------

    Pg_all = u_opt[:n_ctrl_gen]
    Vm_ctrl_opt = u_opt[n_ctrl_gen:]

    x_opt = pf_solve(u_opt)
    Va_ns_opt = x_opt[:n_ns]

    Va_opt = jnp.zeros(nb).at[ns_arr].set(Va_ns_opt)
    Vm_opt = (
        jnp.ones(nb)
        .at[pv_arr].set(Vm_ctrl_opt[:n_pv])
        .at[ref_bus].set(Vm_ctrl_opt[n_pv])
    )
    if n_pq > 0:
        Vm_opt = Vm_opt.at[pq_arr].set(x_opt[n_ns:])

    P_inj_opt, Q_inj_opt = _power_injections(Va_opt, Vm_opt, setup.G, setup.B)

    # Full Qg array (Qg_i = Q_inj_i + Qd_i; for buses with multiple gens: equal split)
    Qg_bus_opt = Q_inj_opt + Qd                  # (nb,)
    n_gens_per_bus = jnp.zeros(nb).at[setup.gen_bus_idx].add(jnp.float64(1.0))
    Qg_all = Qg_bus_opt[setup.gen_bus_idx] / n_gens_per_bus[setup.gen_bus_idx]

    # Branch flows [MW, MVAr]
    pf_mw, qf_mvar = _branch_flows(Va_opt, Vm_opt, setup)

    # Total generation cost [$/h] (PowerZoo AC-OPF: a·P² + c·P in MW)
    Pg_all_mw = Pg_all * bMVA
    total_cost = jnp.sum(
        setup.cost_a * Pg_all_mw ** 2 + setup.cost_c * Pg_all_mw
    )

    # Line thermal violation [MVA]
    Sf2_opt, St2_opt = _branch_s2_pu(Va_opt, Vm_opt, setup)
    has_cap = setup.line_cap_pu > jnp.float64(0.0)
    Sf_pu = jnp.sqrt(jnp.maximum(Sf2_opt, jnp.float64(0.0)))
    St_pu = jnp.sqrt(jnp.maximum(St2_opt, jnp.float64(0.0)))
    viol_f = jnp.where(
        has_cap, jnp.maximum(jnp.float64(0.0), Sf_pu - setup.line_cap_pu),
        jnp.float64(0.0))
    viol_t = jnp.where(
        has_cap, jnp.maximum(jnp.float64(0.0), St_pu - setup.line_cap_pu),
        jnp.float64(0.0))
    line_viol_mva = jnp.maximum(jnp.max(viol_f), jnp.max(viol_t)) * bMVA

    # Physical feasibility check (power flow residual at the optimal point)
    g_check = _mismatch(x_opt, u_opt)
    pf_residual = jnp.max(jnp.abs(g_check))
    pf_ok = pf_residual < jnp.float64(1e-3)
    lbfgsb_ok = jnp.logical_not(last_failed_linesearch)
    converged = jnp.logical_and(pf_ok, lbfgsb_ok)

    # Approximate nodal LMP (reduced-space dual recovery; see ``_recover_lmp_reduced_kkt``).
    lmp = _recover_lmp_reduced_kkt(setup, Va_opt, Vm_opt, Pg_all, Pg_min, Pg_max)

    h_bal_final = (
        P_inj_opt[ref_bus]
        - jnp.dot(setup.Cg_slack_row, Pg_all)
        + Pd[ref_bus]
    )
    # Host-side snapshot only (``float(tracer)`` breaks ``jit`` / ``vmap``).
    jax.debug.callback(
        _store_last_acopf_alm_balance_state_host,
        _lam_bal,
        _rho_bal,
        h_bal_final,
    )

    return ACOPFResult(
        unit_power=Pg_all_mw,
        q_gen=Qg_all * bMVA,
        vm=Vm_opt,
        va=Va_opt,
        line_flow_p=pf_mw,
        line_flow_q=qf_mvar,
        lmp=lmp,
        total_cost=total_cost,
        converged=converged,
        iterations=n_iters,
        line_viol_mva=line_viol_mva,
    )


# ---------------------------------------------------------------------------
# PF residual, benchmarks, SLSQP-vs-JAX diagnosis
# ---------------------------------------------------------------------------


def compute_acopf_pf_residual(
    setup: ACOPFSetup,
    Va: chex.Array,
    Vm: chex.Array,
    Pg_pu: chex.Array,
    Pd: chex.Array,
    Qd: chex.Array,
) -> jnp.ndarray:
    """Max absolute reduced-space PF mismatch ‖g(x,u)‖∞ at explicit (Va, Vm, Pg) [p.u.].

    Matches the ``_mismatch`` definition inside ``ac_opf`` (full ``Cg @ Pg`` at non-slack).
    """
    nb = setup.n_bus
    n_pv = setup.n_pv
    n_pq = setup.n_pq
    n_ns = n_pv + n_pq
    ns_arr = jnp.array(setup.non_slack_bus_idx, dtype=jnp.int32)
    pq_arr = jnp.array(setup.pq_bus_idx, dtype=jnp.int32)

    P_inj, Q_inj = _power_injections(Va, Vm, setup.G, setup.B)
    Cg_Pg_ns = setup.Cg_ns_ctrl @ Pg_pu
    g_P = P_inj[ns_arr] - Cg_Pg_ns + Pd[ns_arr]
    if n_pq > 0:
        g_Q = Q_inj[pq_arr] + Qd[pq_arr]
        g = jnp.concatenate([g_P, g_Q])
    else:
        g = g_P
    return jnp.max(jnp.abs(g))


def compute_acopf_pf_residual_from_result(
    setup: ACOPFSetup,
    result: ACOPFResult,
    node_load_mw: chex.Array = None,
    node_load_mvar: chex.Array = None,
    commitment: chex.Array = None,
) -> jnp.ndarray:
    """``compute_acopf_pf_residual`` for an ``ACOPFResult`` (loads match ``ac_opf`` defaults)."""
    bMVA = setup.base_mva
    Pd = node_load_mw / bMVA if node_load_mw is not None else setup.Pd
    Qd = node_load_mvar / bMVA if node_load_mvar is not None else setup.Qd
    Va = result.va
    Vm = result.vm
    Pg_pu = result.unit_power / bMVA
    if commitment is not None:
        Pg_pu = jnp.where(commitment > 0, Pg_pu, jnp.float64(0.0))
    return compute_acopf_pf_residual(setup, Va, Vm, Pg_pu, Pd, Qd)


def compute_acopf_slack_balance_residual(
    setup: ACOPFSetup,
    Va: chex.Array,
    Vm: chex.Array,
    Pg_pu: chex.Array,
    Pd: chex.Array,
) -> jnp.ndarray:
    """Slack-bus active balance residual ``h_bal`` [p.u.] (same sign as inside ``ac_opf``).

    ``h_bal = P_inj[slack] - (Cg Pg)[slack] + Pd[slack]`` — should be ≈ 0 at a feasible ALM point.
    """
    P_inj, _ = _power_injections(Va, Vm, setup.G, setup.B)
    return P_inj[setup.ref_bus] - jnp.dot(setup.Cg_slack_row, Pg_pu) + Pd[setup.ref_bus]


def compute_acopf_slack_balance_residual_from_result(
    setup: ACOPFSetup,
    result: ACOPFResult,
    node_load_mw: chex.Array = None,
    commitment: chex.Array = None,
) -> jnp.ndarray:
    """``compute_acopf_slack_balance_residual`` for an ``ACOPFResult``."""
    bMVA = setup.base_mva
    Pd = node_load_mw / bMVA if node_load_mw is not None else setup.Pd
    Va = result.va
    Vm = result.vm
    Pg_pu = result.unit_power / bMVA
    if commitment is not None:
        Pg_pu = jnp.where(commitment > 0, Pg_pu, jnp.float64(0.0))
    return compute_acopf_slack_balance_residual(setup, Va, Vm, Pg_pu, Pd)


def get_last_acopf_alm_balance_state() -> dict[str, float] | None:
    """Return ``{lam_bal, rho_bal, h_bal_pu}`` from the last completed host callback, or ``None``.

    Populated via ``jax.debug.callback`` at the end of ``ac_opf`` so values stay concrete under
    ``jit``/``vmap``. For batched runs, the last updated slice is retained.
    """
    return _last_acopf_alm_balance_state


def compute_acopf_inequality_violations(
    setup: ACOPFSetup,
    Va: chex.Array,
    Vm: chex.Array,
    Pg_pu: chex.Array,
    Pd: chex.Array,
    Qd: chex.Array,
    Qg_max_all: chex.Array,
    Qg_min_all: chex.Array,
    Pg_max: chex.Array,
    Pg_min: chex.Array,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return (max_line_viol_pu, max_qg_bus_viol_pu, max_pg_bound_viol_pu).

    Third component is max Pg bound violation over **all** units (legacy name
    ``max_slack_pg_viol`` in some diagnostics).
    """
    nb = setup.n_bus
    gi = setup.gen_bus_idx
    Qg_bus_max_agg = jnp.zeros(nb, dtype=jnp.float64).at[gi].add(Qg_max_all)
    Qg_bus_min_agg = jnp.zeros(nb, dtype=jnp.float64).at[gi].add(Qg_min_all)
    has_gen_bus = jnp.zeros(nb, dtype=jnp.float64).at[gi].add(jnp.float64(1.0)) > jnp.float64(0.0)

    P_inj, Q_inj = _power_injections(Va, Vm, setup.G, setup.B)

    nl = int(setup.line_cap_pu.shape[0])
    if nl > 0:
        Sf2, St2 = _branch_s2_pu(Va, Vm, setup)
        Sf_pu = jnp.sqrt(jnp.maximum(Sf2, jnp.float64(0.0)))
        St_pu = jnp.sqrt(jnp.maximum(St2, jnp.float64(0.0)))
        has_cap = setup.line_cap_pu > jnp.float64(0.0)
        viol_f = jnp.where(
            has_cap,
            jnp.maximum(jnp.float64(0.0), Sf_pu - setup.line_cap_pu),
            jnp.float64(0.0),
        )
        viol_t = jnp.where(
            has_cap,
            jnp.maximum(jnp.float64(0.0), St_pu - setup.line_cap_pu),
            jnp.float64(0.0),
        )
        max_line = jnp.maximum(jnp.max(viol_f), jnp.max(viol_t))
    else:
        max_line = jnp.float64(0.0)

    Qg_bus = Q_inj + Qd
    h_Qg_max = jnp.where(has_gen_bus, Qg_bus - Qg_bus_max_agg, jnp.float64(0.0))
    h_Qg_min = jnp.where(has_gen_bus, Qg_bus_min_agg - Qg_bus, jnp.float64(0.0))
    max_qg = jnp.maximum(
        jnp.max(jnp.maximum(jnp.float64(0.0), h_Qg_max)),
        jnp.max(jnp.maximum(jnp.float64(0.0), h_Qg_min)),
    )

    max_slack = jnp.maximum(
        jnp.max(jnp.maximum(jnp.float64(0.0), Pg_pu - Pg_max)),
        jnp.max(jnp.maximum(jnp.float64(0.0), Pg_min - Pg_pu)),
    )
    return max_line, max_qg, max_slack


@struct.dataclass
class ACOPFBenchmarkConfig:
    """One AC-OPF benchmark run (non-pytree; for Python driver loops only)."""

    name: str = struct.field(pytree_node=False, default="cfg")
    tol: float = struct.field(pytree_node=False, default=1e-3)
    max_iter: int = struct.field(pytree_node=False, default=200)
    alm_outer_steps: int = struct.field(pytree_node=False, default=12)
    warm: str = struct.field(pytree_node=False, default="none")  # none | dcopf | provided


@dataclass(frozen=True)
class ACOPFBenchmarkRow:
    """One row of benchmark output (sorted by ``total_cost`` ascending)."""

    name: str
    tol: float
    max_iter: int
    alm_outer_steps: int
    warm: str
    total_cost: float
    line_viol_mva: float
    pf_residual: float
    iterations: int
    runtime_sec: float


def make_acopf_benchmark_grid(
    *,
    tols: tuple[float, ...] = (1e-3, 1e-4, 1e-5),
    max_iters: tuple[int, ...] = (200, 400),
    alm_outer_steps_list: tuple[int, ...] = (8, 12, 20),
    warms: tuple[str, ...] = ("none", "dcopf"),
) -> list[ACOPFBenchmarkConfig]:
    """Cartesian product of sweep knobs (default excludes ``provided`` warm)."""
    out: list[ACOPFBenchmarkConfig] = []
    for t in tols:
        for m in max_iters:
            for a in alm_outer_steps_list:
                for w in warms:
                    name = f"t{t:g}_m{m}_a{a}_w{w}"
                    out.append(
                        ACOPFBenchmarkConfig(
                            name=name,
                            tol=float(t),
                            max_iter=int(m),
                            alm_outer_steps=int(a),
                            warm=str(w),
                        )
                    )
    return out


def benchmark_acopf_configs(
    setup: ACOPFSetup,
    config_list: Sequence[ACOPFBenchmarkConfig],
    warm_starts: Mapping[str, np.ndarray] | None = None,
    *,
    case: CaseData | None = None,
    node_load_mw: chex.Array | None = None,
    node_load_mvar: chex.Array | None = None,
    commitment: chex.Array | None = None,
    warmup_runs: int = 1,
) -> list[ACOPFBenchmarkRow]:
    """Run ``ac_opf`` under several configs; record cost, violations, PF residual, time (warm).

    **Warm start**
    - ``none``: internal flat / proportional guess.
    - ``dcopf``: ``prepare_dcopf(case)`` + ``dc_opf`` → ``unit_power`` as warm start (requires ``case``).
    - ``provided``: ``warm_starts[config.name]`` must be an array-like (n_gen,) MW dispatch.

    Timing: after optional ``warmup_runs`` JIT/XLA compilations, one timed execution uses
    ``jax.block_until_ready`` on scalars read from the result.

    Returns rows sorted by ``total_cost`` ascending.
    """
    from powerzoojax.envs.grid.dc_opf import dc_opf, prepare_dcopf

    warm_starts = warm_starts or {}
    rows: list[ACOPFBenchmarkRow] = []

    for cfg in config_list:
        ws_mw: np.ndarray | None = None
        if cfg.warm == "none":
            ws_mw = None
        elif cfg.warm == "dcopf":
            if case is None:
                raise ValueError("benchmark_acopf_configs: case=... required when warm='dcopf'")
            d_setup = prepare_dcopf(case)
            if node_load_mw is None:
                nw = jnp.asarray(setup.Pd * setup.base_mva, dtype=jnp.float64)
            else:
                nw = jnp.asarray(node_load_mw, dtype=jnp.float64)
            d_res = dc_opf(d_setup, node_load_mw=nw)
            ws_mw = np.asarray(d_res.unit_power, dtype=np.float64)
        elif cfg.warm == "provided":
            if cfg.name not in warm_starts:
                raise KeyError(
                    f"warm='provided' requires warm_starts[{cfg.name!r}] (MW array)"
                )
            ws_mw = np.asarray(warm_starts[cfg.name], dtype=np.float64)
        else:
            raise ValueError(f"unknown warm mode {cfg.warm!r}")

        ws_arg = None if ws_mw is None else jnp.asarray(ws_mw)

        def _run():
            return ac_opf(
                setup,
                node_load_mw=node_load_mw,
                node_load_mvar=node_load_mvar,
                commitment=commitment,
                max_iter=cfg.max_iter,
                tol=cfg.tol,
                warm_start_unit_power_mw=ws_arg,
                alm_outer_steps=cfg.alm_outer_steps,
            )

        for _ in range(max(0, int(warmup_runs))):
            r0 = _run()
            jax.block_until_ready(r0.total_cost)

        t0 = time.perf_counter()
        r = _run()
        jax.block_until_ready(r.total_cost)
        jax.block_until_ready(r.line_viol_mva)
        jax.block_until_ready(r.iterations)
        t1 = time.perf_counter()

        pf_res = compute_acopf_pf_residual_from_result(
            setup, r, node_load_mw=node_load_mw, node_load_mvar=node_load_mvar,
            commitment=commitment,
        )
        jax.block_until_ready(pf_res)

        rows.append(
            ACOPFBenchmarkRow(
                name=cfg.name,
                tol=cfg.tol,
                max_iter=cfg.max_iter,
                alm_outer_steps=cfg.alm_outer_steps,
                warm=cfg.warm,
                total_cost=float(np.asarray(r.total_cost)),
                line_viol_mva=float(np.asarray(r.line_viol_mva)),
                pf_residual=float(np.asarray(pf_res)),
                iterations=int(np.asarray(r.iterations)),
                runtime_sec=float(t1 - t0),
            )
        )

    rows.sort(key=lambda row: row.total_cost)
    return rows


@dataclass(frozen=True)
class ACOPFDiagnosisReport:
    """Structured comparison of a JAX ``ACOPFResult`` vs an SLSQP (or other) reference."""

    # Primal mismatch (golden vs JAX)
    max_abs_unit_power_mw: float
    rms_unit_power_mw: float
    max_abs_vm: float
    rms_vm: float
    max_abs_lmp: float
    rms_lmp: float
    l2_delta_unit_power_mw: float
    l2_delta_vm: float
    total_cost_jax: float
    total_cost_golden: float
    total_cost_gap: float
    # Feasibility / constraint activity on JAX primal
    pf_residual: float
    line_viol_mva: float
    max_qg_bus_viol_pu: float
    slack_pg_viol_pu: float
    # Slack P balance (Prompt A equality ALM)
    abs_h_bal_pu: float
    h_bal_per_sum_abs_pd: float
    lam_bal: float | None
    rho_bal: float | None
    # Pure generation-cost gradient ‖∂C/∂P_MW‖₂ in $/MWh (not projected KKT grad of full ALM objective)
    cost_grad_norm_mwh: float
    # Optimality heuristics
    mc_std_interior: float
    mc_max_min_interior: float
    n_interior_gens: int
    flag_likely_feasible_not_kkt: bool
    notes: tuple[str, ...] = field(default_factory=tuple)


def diagnose_acopf_vs_golden(
    setup: ACOPFSetup,
    result: ACOPFResult,
    golden_unit_power_mw: np.ndarray,
    golden_vm: np.ndarray,
    golden_lmp: np.ndarray,
    *,
    golden_total_cost: float | None = None,
    node_load_mw: chex.Array | None = None,
    node_load_mvar: chex.Array | None = None,
    commitment: chex.Array | None = None,
    use_last_acopf_alm_state: bool = True,
    lam_bal: float | None = None,
    rho_bal: float | None = None,
) -> ACOPFDiagnosisReport:
    """Explain gaps between JAX AC-OPF and a reference (e.g. SLSQP golden).

    **Primal mismatch**: dispatch / voltage differences (no claim of which solver is ``true`` OPF).

    **Dual mismatch**: JAX uses ALM + box-constrained L-BFGS-B; reference KKT multipliers are not
    recovered — ``result.lmp`` is only an approximate dual diagnostic (see ``_recover_lmp_reduced_kkt``).

    **Approximate LMP limitation**: bus-level WLS surrogate ignores branch / inequality duals; do not
    expect pointwise agreement with SLSQP shadow prices.

    **Balance diagnostics**: ``abs_h_bal_pu`` and ``lam_bal`` / ``rho_bal`` (from optional kwargs, or
    from ``get_last_acopf_alm_balance_state()`` if ``use_last_acopf_alm_state`` and the last call was
    ``ac_opf`` on the same process).  **Not** the projected gradient of the full ALM+L-BFGS objective;
    ``cost_grad_norm_mwh`` is only ‖∂C/∂P_MW‖₂ for the generation cost ``C``.
    """
    bMVA = float(setup.base_mva)
    Pg_min = setup.Pg_min
    Pg_max = setup.Pg_max
    Qg_min_all = setup.Qg_min
    Qg_max_all = setup.Qg_max
    if commitment is not None:
        Pg_min = Pg_min * commitment
        Pg_max = Pg_max * commitment
        Qg_min_all = Qg_min_all * commitment
        Qg_max_all = Qg_max_all * commitment

    u_j = np.asarray(result.unit_power, dtype=np.float64).ravel()
    u_g = np.asarray(golden_unit_power_mw, dtype=np.float64).ravel()
    vm_j = np.asarray(result.vm, dtype=np.float64).ravel()
    vm_g = np.asarray(golden_vm, dtype=np.float64).ravel()
    lmp_j = np.asarray(result.lmp, dtype=np.float64).ravel()
    lmp_g = np.asarray(golden_lmp, dtype=np.float64).ravel()

    du = u_j - u_g
    dvm = vm_j - vm_g
    dlmp = lmp_j - lmp_g
    l2_dp = float(np.linalg.norm(du))
    l2_dvm = float(np.linalg.norm(dvm))

    tc_j = float(np.asarray(result.total_cost))
    if golden_total_cost is not None:
        tc_g = float(golden_total_cost)
    else:
        tc_g = float(
            np.sum(
                np.asarray(setup.cost_a) * (u_g ** 2)
                + np.asarray(setup.cost_c) * u_g
            )
        )

    Pd_np = (
        np.asarray(node_load_mw, dtype=np.float64) / bMVA
        if node_load_mw is not None
        else np.asarray(setup.Pd)
    )
    Qd_np = (
        np.asarray(node_load_mvar, dtype=np.float64) / bMVA
        if node_load_mvar is not None
        else np.asarray(setup.Qd)
    )

    Va = result.va
    Vm = result.vm
    Pg_pu = result.unit_power / setup.base_mva
    pf_res = float(
        np.asarray(
            compute_acopf_pf_residual_from_result(
                setup, result, node_load_mw=node_load_mw, node_load_mvar=node_load_mvar,
                commitment=commitment,
            )
        )
    )
    line_vm = float(np.asarray(result.line_viol_mva))
    _ml, max_qg, max_slack = compute_acopf_inequality_violations(
        setup, Va, Vm, Pg_pu,
        jnp.asarray(Pd_np),
        jnp.asarray(Qd_np),
        Qg_max_all,
        Qg_min_all,
        Pg_max,
        Pg_min,
    )
    max_qg_f = float(np.asarray(max_qg))
    max_slack_f = float(np.asarray(max_slack))

    h_bal_np = float(
        np.asarray(
            compute_acopf_slack_balance_residual_from_result(
                setup, result, node_load_mw=node_load_mw, commitment=commitment
            )
        )
    )
    abs_h = abs(h_bal_np)
    sum_abs_pd = float(np.sum(np.abs(Pd_np)))
    h_ratio = abs_h / max(sum_abs_pd, 1e-12)

    lam_b = lam_bal
    rho_b = rho_bal
    if use_last_acopf_alm_state:
        st = get_last_acopf_alm_balance_state()
        if st is not None:
            if lam_b is None:
                lam_b = st.get("lam_bal")
            if rho_b is None:
                rho_b = st.get("rho_bal")

    Pg_mw = np.asarray(result.unit_power)
    mc = np.asarray(setup.cost_a) * (2.0 * Pg_mw) + np.asarray(setup.cost_c)
    cost_grad_norm = float(np.linalg.norm(mc))
    eps_pu = 0.01
    Pgmin = np.asarray(Pg_min) * bMVA
    Pgmax = np.asarray(Pg_max) * bMVA
    interior = (Pg_mw > Pgmin + eps_pu) & (Pg_mw < Pgmax - eps_pu)
    n_int = int(np.sum(interior))
    if n_int > 1:
        mc_i = mc[interior]
        mc_std = float(np.std(mc_i))
        mc_mm = float(np.max(mc_i) - np.min(mc_i))
    else:
        mc_std = 0.0
        mc_mm = 0.0

    line_viol_pu = line_vm / max(bMVA, 1e-6)
    viol_sum = pf_res + line_viol_pu + max_qg_f + max_slack_f
    likely_subopt = viol_sum < 0.05 and (mc_mm > 1.0)

    note_primal = (
        "Primal mismatch: compares MW / p.u. / $/MWh vectors; differences can come from "
        "different KKT stationarity, ALM tolerance, or local minima of the reduced problem."
    )
    note_dual = (
        "Dual mismatch: JAX ALM+L-BFGS-B does not export full NLP multipliers; "
        "reference LMP from SLSQP includes inequality duals that our surrogate omits."
    )
    note_lmp = (
        "Approximate LMP: _recover_lmp_reduced_kkt uses PF Jacobian + bus-level MC rows only; "
        "not comparable to SLSQP λ for line/Qg/slack constraints."
    )
    note_interpret = (
        "If PF residual and violations are small but dispatch/LMP differ from SLSQP, the JAX "
        "point is likely **feasible but not identical to the SLSQP KKT point** (different "
        "constraint handling / non-convexity)."
    )
    if likely_subopt:
        note_interpret += (
            " Interior marginal costs show non-negligible spread — consistent with "
            "**near-feasible, not fully economic-dispatch-stationary** behaviour."
        )
    note_balance = (
        "Interpret |h_bal|: if tiny, slack active balance is already satisfied by ALM; if not, "
        "raise rho_bal0 / alm_outer_steps before blaming Q-side modeling."
    )

    notes = (note_primal, note_dual, note_lmp, note_interpret, note_balance)

    return ACOPFDiagnosisReport(
        max_abs_unit_power_mw=float(np.max(np.abs(du))),
        rms_unit_power_mw=float(np.sqrt(np.mean(du ** 2))),
        max_abs_vm=float(np.max(np.abs(dvm))),
        rms_vm=float(np.sqrt(np.mean(dvm ** 2))),
        max_abs_lmp=float(np.max(np.abs(dlmp))),
        rms_lmp=float(np.sqrt(np.mean(dlmp ** 2))),
        l2_delta_unit_power_mw=l2_dp,
        l2_delta_vm=l2_dvm,
        total_cost_jax=tc_j,
        total_cost_golden=tc_g,
        total_cost_gap=tc_j - tc_g,
        pf_residual=pf_res,
        line_viol_mva=line_vm,
        max_qg_bus_viol_pu=max_qg_f,
        slack_pg_viol_pu=max_slack_f,
        abs_h_bal_pu=abs_h,
        h_bal_per_sum_abs_pd=h_ratio,
        lam_bal=lam_b,
        rho_bal=rho_b,
        cost_grad_norm_mwh=cost_grad_norm,
        mc_std_interior=mc_std,
        mc_max_min_interior=mc_mm,
        n_interior_gens=n_int,
        flag_likely_feasible_not_kkt=bool(likely_subopt),
        notes=notes,
    )


def format_acopf_diagnosis_report(rep: ACOPFDiagnosisReport) -> str:
    """Human-readable multi-line string (for notebooks / logs)."""
    lam_s = f"{rep.lam_bal:.8g}" if rep.lam_bal is not None else "n/a"
    rho_s = f"{rep.rho_bal:.8g}" if rep.rho_bal is not None else "n/a"
    lines = [
        "=== AC-OPF JAX vs golden diagnosis ===",
        f"max|ΔP|={rep.max_abs_unit_power_mw:.6g} MW  rms|ΔP|={rep.rms_unit_power_mw:.6g}  ||ΔP||_2={rep.l2_delta_unit_power_mw:.6g}",
        f"max|ΔVm|={rep.max_abs_vm:.6g} p.u.  rms|ΔVm|={rep.rms_vm:.6g}  ||ΔVm||_2={rep.l2_delta_vm:.6g}",
        f"max|ΔLMP|={rep.max_abs_lmp:.6g} $/MWh  rms|ΔLMP|={rep.rms_lmp:.6g}",
        f"total_cost jax={rep.total_cost_jax:.8g}  golden={rep.total_cost_golden:.8g}  gap={rep.total_cost_gap:.8g}",
        f"PF residual (max |g|)={rep.pf_residual:.6g} p.u.",
        f"line_viol_mva={rep.line_viol_mva:.6g}  max_qg_bus_viol_pu={rep.max_qg_bus_viol_pu:.6g}  Pg_bound_viol_pu={rep.slack_pg_viol_pu:.6g}",
        f"|h_bal|={rep.abs_h_bal_pu:.6g} p.u.   |h_bal|/sum|Pd|={rep.h_bal_per_sum_abs_pd:.6g}   lam_bal={lam_s}   rho_bal={rho_s}",
        f"||dC/dP_MW||_2 (cost only)={rep.cost_grad_norm_mwh:.6g} $/MWh   (not full ALM projected grad)",
        f"MC interior: std={rep.mc_std_interior:.6g}  max-min={rep.mc_max_min_interior:.6g}  n_int={rep.n_interior_gens}",
        f"heuristic 'feasible_not_kkt'={rep.flag_likely_feasible_not_kkt}",
        "--- notes ---",
    ]
    lines.extend(rep.notes)
    return "\n".join(lines)
