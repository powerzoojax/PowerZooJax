"""AC Power Flow (Newton-Raphson) — pure-JAX algorithm module.

A JAX-native AC power flow solver used by ``TransGridEnv`` when
``physics=1`` and ``solver_mode ∈ {0, 1}`` (raw PF, or DCOPF dispatch
followed by AC verification). Implements Newton-Raphson on the
MATPOWER pi-model (complex tap ratio); supports PV → PQ switching for
generator Q-limit enforcement (``Qg_min_bus`` / ``Qg_max_bus`` /
``Vm_setpoint`` fields on ``ACPFSetup``).

Translated from PowerZoo's ``cal_pf_trans.py``.

Architecture:
    Setup-time (NumPy, called once):
        branch_admittances, build_ybus, prepare_acpf
    Runtime (pure JAX, JIT-compilable):
        ac_power_flow, calc_branch_flows, ac_power_flow_with_check

I/O contract:
    Input  : ACPFSetup (precomputed from CaseData), max_iter, tol
    Output : ACPFResult { vm, va, p_calc, q_calc, pf_from, qf_from,
                           pf_to, qf_to, converged, iterations }

    ``ac_power_flow_with_check`` returns a 9-tuple:
        (line_flow_mw, node_injection_mw, vm, va, is_safe,
         n_violations, cost_thermal, cost_voltage, converged)

vmap note:
    The Newton-Raphson loop uses ``lax.while_loop``, so under ``vmap``
    every instance waits for the slowest to converge. Extreme
    exploration actions may push a few instances to ``max_iter``,
    dragging the whole batch down. Mitigations: tighten action bounds,
    lower ``max_iter``, or monitor ``converged`` flags downstream.

float32 precision:
    The default ``tol=1e-5`` sits near the float32 noise floor. For
    grids with > 30 buses, consider ``tol=1e-4`` or enable
    ``jax.config.update('jax_enable_x64', True)``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import jax.numpy as jnp
import jax.lax as lax
import chex
from flax import struct

from powerzoojax.case.case_data import CaseData
from powerzoojax.envs.grid.power_flow import safety_check


# ---------------------------------------------------------------------------
# Result containers (pytree-compatible)
# ---------------------------------------------------------------------------

@struct.dataclass
class ACPFSetup:
    """Precomputed data for AC power flow (built once at env construction)."""
    Y_bus_real: chex.Array       # (n_bus, n_bus) real part of Ybus
    Y_bus_imag: chex.Array       # (n_bus, n_bus) imag part of Ybus
    P_sched: chex.Array          # (n_bus,) scheduled net active injection [p.u.]
    Q_sched: chex.Array          # (n_bus,) scheduled net reactive injection [p.u.]
    Vm_init: chex.Array          # (n_bus,) initial voltage magnitudes [p.u.]
    Va_init: chex.Array          # (n_bus,) initial voltage angles [rad]
    pv_idx: chex.Array           # (n_pv,) PV bus indices
    pq_idx: chex.Array           # (n_pq,) PQ bus indices
    pv_pq_idx: chex.Array        # (n_pv+n_pq,) concatenated PV+PQ indices
    n_pv: int = struct.field(pytree_node=False, default=0)
    n_pq: int = struct.field(pytree_node=False, default=0)
    n_pvpq: int = struct.field(pytree_node=False, default=0)
    n_bus: int = struct.field(pytree_node=False, default=0)
    base_mva: float = struct.field(pytree_node=False, default=100.0)
    # Branch admittance data for branch flow calculation
    Yff_real: chex.Array = None  # (n_branch,)
    Yff_imag: chex.Array = None
    Yft_real: chex.Array = None
    Yft_imag: chex.Array = None
    Ytf_real: chex.Array = None
    Ytf_imag: chex.Array = None
    Ytt_real: chex.Array = None
    Ytt_imag: chex.Array = None
    br_from_idx: chex.Array = None  # (n_branch,) int
    br_to_idx: chex.Array = None    # (n_branch,) int
    br_status: chex.Array = None    # (n_branch,) float
    gen_bus_idx: chex.Array = None  # (n_gen,) int — maps gen to bus
    Pd_mw: chex.Array = None        # (n_bus,) bus active load [MW]
    Qd_mw: chex.Array = None        # (n_bus,) bus reactive load [MW]
    Pg_sched_per_gen: chex.Array = None  # (n_gen,) scheduled Pg [MW]
    # Q-limit fields for PV→PQ switching (added for Q-limit enforcement)
    Vm_setpoint: chex.Array = None  # (n_bus,) voltage setpoints for PV buses [p.u.]
    Qg_min_bus: chex.Array = None   # (n_bus,) per-bus net-injection Qmin [p.u.] = Qg_min-Qd; PQ/slack = -inf
    Qg_max_bus: chex.Array = None   # (n_bus,) per-bus net-injection Qmax [p.u.] = Qg_max-Qd; PQ/slack = +inf


@struct.dataclass
class ACPFResult:
    """Output of AC power flow solver."""
    vm: chex.Array              # (n_bus,) voltage magnitudes [p.u.]
    va: chex.Array              # (n_bus,) voltage angles [rad]
    p_calc: chex.Array          # (n_bus,) calculated P injection [p.u.]
    q_calc: chex.Array          # (n_bus,) calculated Q injection [p.u.]
    pf_from: chex.Array         # (n_branch,) from-end active flow [MW]
    qf_from: chex.Array         # (n_branch,) from-end reactive flow [MW]
    pf_to: chex.Array           # (n_branch,) to-end active flow [MW]
    qf_to: chex.Array           # (n_branch,) to-end reactive flow [MW]
    converged: chex.Array       # bool scalar
    iterations: chex.Array      # int32 scalar


# ---------------------------------------------------------------------------
# Setup-time functions (NumPy — called once)
# ---------------------------------------------------------------------------

def branch_admittances(
    br_r: np.ndarray,
    br_x: np.ndarray,
    br_b: np.ndarray,
    br_ratio: np.ndarray,
    br_angle: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-branch pi-model admittances (MATPOWER model).

    Args:
        br_r: Branch resistance (n_branch,) [p.u.]
        br_x: Branch reactance (n_branch,) [p.u.]
        br_b: Branch charging susceptance (n_branch,) [p.u.]
        br_ratio: Transformer tap ratio (0 → regular line = 1)
        br_angle: Phase shift [degrees]

    Returns:
        (Yff, Yft, Ytf, Ytt) — complex arrays (n_branch,)
    """
    z = br_r + 1j * br_x
    Ys = np.where(z != 0, 1.0 / z, 0j)

    ratio = br_ratio.copy().astype(float)
    ratio[ratio == 0] = 1.0
    tap = ratio * np.exp(1j * np.radians(br_angle))
    tap_mag2 = (tap * np.conj(tap)).real

    Yff = (Ys + 0.5j * br_b) / tap_mag2
    Yft = -Ys / np.conj(tap)
    Ytf = -Ys / tap
    Ytt = Ys + 0.5j * br_b

    return Yff, Yft, Ytf, Ytt


def build_ybus(
    n_bus: int,
    br_from: np.ndarray,
    br_to: np.ndarray,
    br_r: np.ndarray,
    br_x: np.ndarray,
    br_b: np.ndarray,
    br_ratio: np.ndarray,
    br_angle: np.ndarray,
    br_status: np.ndarray,
    bus_gs: np.ndarray,
    bus_bs: np.ndarray,
    base_mva: float = 100.0,
) -> np.ndarray:
    """Build the complex bus admittance matrix (Ybus).

    Returns:
        Ybus: complex (n_bus, n_bus)
    """
    on = br_status > 0
    f = br_from[on].astype(int)
    t = br_to[on].astype(int)

    Yff, Yft, Ytf, Ytt = branch_admittances(
        br_r[on], br_x[on], br_b[on], br_ratio[on], br_angle[on])

    Ybus = np.zeros((n_bus, n_bus), dtype=complex)
    np.add.at(Ybus, (f, f), Yff)
    np.add.at(Ybus, (f, t), Yft)
    np.add.at(Ybus, (t, f), Ytf)
    np.add.at(Ybus, (t, t), Ytt)

    diag = np.arange(n_bus)
    Ybus[diag, diag] += (bus_gs + 1j * bus_bs) / base_mva

    return Ybus


def prepare_acpf(case: CaseData) -> ACPFSetup:
    """Prepare all precomputed data needed for AC power flow.

    Extracts bus classification, generator schedules, and builds Y_bus.
    Called once at env construction time.

    Args:
        case: CaseData with AC fields populated.

    Returns:
        ACPFSetup: frozen struct ready for JIT-compiled NR solver.
    """
    n_bus = case.n_nodes
    n_gen = case.n_units
    n_branch = case.n_lines
    base_mva = case.base_mva

    # Extract arrays as numpy
    br_from_idx = np.asarray(case.line_from_idx, dtype=int)
    br_to_idx = np.asarray(case.line_to_idx, dtype=int)
    br_r = np.asarray(case.line_r if case.line_r is not None else np.zeros(n_branch))
    br_x = np.asarray(case.line_x)
    br_b = np.asarray(case.line_b if case.line_b is not None else np.zeros(n_branch))
    br_ratio = np.asarray(case.line_ratio if case.line_ratio is not None else np.zeros(n_branch))
    br_angle = np.asarray(case.line_angle if case.line_angle is not None else np.zeros(n_branch))
    br_status = np.asarray(case.line_status if case.line_status is not None else np.ones(n_branch))

    bus_type = np.asarray(case.node_type, dtype=int)
    Pd = np.asarray(case.node_pd) / base_mva
    Qd = np.asarray(case.node_qd) / base_mva
    Gs = np.asarray(case.node_gs if case.node_gs is not None else np.zeros(n_bus))
    Bs = np.asarray(case.node_bs if case.node_bs is not None else np.zeros(n_bus))

    # Generator → bus mapping (default to midpoint / zero / 1.0 when AC fields absent)
    gen_bus_idx = np.asarray(case.unit_node_idx, dtype=int)
    _p_mid = (np.asarray(case.unit_p_min) + np.asarray(case.unit_p_max)) / 2.0
    Pg_gen = np.asarray(case.unit_pg if case.unit_pg is not None else _p_mid) / base_mva
    Qg_gen = np.asarray(case.unit_qg if case.unit_qg is not None else np.zeros(n_gen)) / base_mva
    Vg_gen = np.asarray(case.unit_vg if case.unit_vg is not None else np.ones(n_gen))
    Pg_sched_mw = np.asarray(case.unit_pg if case.unit_pg is not None else _p_mid)

    # Aggregate generators to bus level
    Pg_bus = np.zeros(n_bus)
    Qg_bus = np.zeros(n_bus)
    Vm_setpoint = np.ones(n_bus)

    np.add.at(Pg_bus, gen_bus_idx, Pg_gen)
    np.add.at(Qg_bus, gen_bus_idx, Qg_gen)
    Vm_setpoint[gen_bus_idx] = Vg_gen

    P_sched = Pg_bus - Pd
    Q_sched = Qg_bus - Qd

    # Build Ybus
    Ybus = build_ybus(
        n_bus, br_from_idx, br_to_idx,
        br_r, br_x, br_b, br_ratio, br_angle, br_status,
        Gs, Bs, base_mva)

    # Bus classification
    pv_idx = np.where(bus_type == 2)[0]
    pq_idx = np.where(bus_type == 1)[0]
    slack_idx = np.where(bus_type == 3)[0]
    pv_pq_idx = np.concatenate([pv_idx, pq_idx])
    n_pv = len(pv_idx)
    n_pq = len(pq_idx)
    n_pvpq = n_pv + n_pq

    # Initial guess
    Vm = np.ones(n_bus)
    Va = np.zeros(n_bus)
    Vm[pv_idx] = Vm_setpoint[pv_idx]
    Vm[bus_type == 3] = Vm_setpoint[bus_type == 3]

    # Q-limit data: aggregate per-gen limits to bus level.
    # PV buses get the union of all connected generators' Q ranges.
    # PQ and slack buses use ±inf (Q-limits never enforced there).
    if case.unit_q_min is not None and case.unit_q_max is not None:
        Qg_min_gen = np.asarray(case.unit_q_min, dtype=float) / base_mva
        Qg_max_gen = np.asarray(case.unit_q_max, dtype=float) / base_mva
    else:
        Qg_min_gen = np.full(n_gen, -np.inf)
        Qg_max_gen = np.full(n_gen,  np.inf)

    # init to sentinel values so minimum.at / maximum.at work correctly
    Qg_min_bus = np.full(n_bus,  np.inf)
    Qg_max_bus = np.full(n_bus, -np.inf)
    np.minimum.at(Qg_min_bus, gen_bus_idx, Qg_min_gen)
    np.maximum.at(Qg_max_bus, gen_bus_idx, Qg_max_gen)
    # PQ and slack buses: no Q-limit enforcement
    Qg_min_bus[pq_idx] = -np.inf
    Qg_max_bus[pq_idx] =  np.inf
    Qg_min_bus[slack_idx] = -np.inf
    Qg_max_bus[slack_idx] =  np.inf
    # Safety: PV buses with no assigned generator keep unlimited range
    pv_no_gen = Qg_min_bus[pv_idx] == np.inf
    Qg_min_bus[pv_idx[pv_no_gen]] = -np.inf
    Qg_max_bus[pv_idx[pv_no_gen]] =  np.inf

    # Convert generator Q limits to net-injection limits:
    #   Q_net = Q_gen - Q_load  →  Q_net_max = Qg_max - Qd
    # The NR computes Q_calc = net injection. Comparing Q_calc against
    # these corrected limits is equivalent to checking Q_gen = Q_calc + Qd
    # against the original generator limits.
    # PQ/slack buses already have ±inf; subtracting Qd doesn't affect them.
    Qg_min_bus -= Qd
    Qg_max_bus -= Qd

    # Branch admittances for post-processing
    Yff, Yft, Ytf, Ytt = branch_admittances(br_r, br_x, br_b, br_ratio, br_angle)

    return ACPFSetup(
        Y_bus_real=jnp.array(Ybus.real, dtype=jnp.float32),
        Y_bus_imag=jnp.array(Ybus.imag, dtype=jnp.float32),
        P_sched=jnp.array(P_sched, dtype=jnp.float32),
        Q_sched=jnp.array(Q_sched, dtype=jnp.float32),
        Vm_init=jnp.array(Vm, dtype=jnp.float32),
        Va_init=jnp.array(Va, dtype=jnp.float32),
        pv_idx=jnp.array(pv_idx, dtype=jnp.int32),
        pq_idx=jnp.array(pq_idx, dtype=jnp.int32),
        pv_pq_idx=jnp.array(pv_pq_idx, dtype=jnp.int32),
        n_pv=n_pv,
        n_pq=n_pq,
        n_pvpq=n_pvpq,
        n_bus=n_bus,
        base_mva=base_mva,
        Yff_real=jnp.array(Yff.real, dtype=jnp.float32),
        Yff_imag=jnp.array(Yff.imag, dtype=jnp.float32),
        Yft_real=jnp.array(Yft.real, dtype=jnp.float32),
        Yft_imag=jnp.array(Yft.imag, dtype=jnp.float32),
        Ytf_real=jnp.array(Ytf.real, dtype=jnp.float32),
        Ytf_imag=jnp.array(Ytf.imag, dtype=jnp.float32),
        Ytt_real=jnp.array(Ytt.real, dtype=jnp.float32),
        Ytt_imag=jnp.array(Ytt.imag, dtype=jnp.float32),
        br_from_idx=jnp.array(br_from_idx, dtype=jnp.int32),
        br_to_idx=jnp.array(br_to_idx, dtype=jnp.int32),
        br_status=jnp.array(br_status, dtype=jnp.float32),
        gen_bus_idx=jnp.array(gen_bus_idx, dtype=jnp.int32),
        Pd_mw=jnp.array(Pd * base_mva, dtype=jnp.float32),
        Qd_mw=jnp.array(Qd * base_mva, dtype=jnp.float32),
        Pg_sched_per_gen=jnp.array(Pg_sched_mw, dtype=jnp.float32),
        Vm_setpoint=jnp.array(Vm_setpoint, dtype=jnp.float32),
        Qg_min_bus=jnp.array(Qg_min_bus, dtype=jnp.float32),
        Qg_max_bus=jnp.array(Qg_max_bus, dtype=jnp.float32),
    )


# ---------------------------------------------------------------------------
# Runtime functions (pure JAX, JIT-compilable)
# ---------------------------------------------------------------------------

def _build_jacobian(
    Vm: chex.Array,
    Va: chex.Array,
    G: chex.Array,
    B: chex.Array,
    P_calc: chex.Array,
    Q_calc: chex.Array,
    pv_pq_idx: chex.Array,
    bus_is_pq: chex.Array,
) -> chex.Array:
    """Build the NR Jacobian with fixed shape (2*n_pvpq, 2*n_pvpq).

    The second block row is masked by ``bus_is_pq``:
    - PQ buses (or switched PV buses): use the standard Q equations.
    - Unswitched PV buses: use an identity row (forcing ΔVm = 0, voltage held
      at setpoint).

    This keeps the Jacobian shape static across Q-limit switching iterations,
    which is required by ``lax.while_loop``.

    Full-bus matrices computed first, then sliced to relevant rows/columns.
    Diagonal correction follows MATPOWER (v7) makeJac / dSbus_dV derivation.
    """
    n = Vm.shape[0]
    n_pvpq = pv_pq_idx.shape[0]

    dVa = Va[:, None] - Va[None, :]
    VmVm = Vm[:, None] * Vm[None, :]

    sin_dVa = jnp.sin(dVa)
    cos_dVa = jnp.cos(dVa)

    H = VmVm * (G * sin_dVa - B * cos_dVa)
    N = VmVm * (G * cos_dVa + B * sin_dVa)
    M = -N
    L = H

    diag = jnp.arange(n)
    Vm_sq = Vm ** 2
    H = H.at[diag, diag].set(-Q_calc - B[diag, diag] * Vm_sq)
    N = N.at[diag, diag].set(P_calc + G[diag, diag] * Vm_sq)
    M = M.at[diag, diag].set(P_calc - G[diag, diag] * Vm_sq)
    L = L.at[diag, diag].set(Q_calc - B[diag, diag] * Vm_sq)

    # dP/dVm and dQ/dVm: divide off-diag columns by Vm[j]
    N_col = N / Vm[None, :]
    L_col = L / Vm[None, :]

    N_col = N_col.at[diag, diag].set(P_calc / Vm + G[diag, diag] * Vm)
    L_col = L_col.at[diag, diag].set(Q_calc / Vm - B[diag, diag] * Vm)

    J11 = H[jnp.ix_(pv_pq_idx, pv_pq_idx)]           # (n_pvpq, n_pvpq)
    J12 = N_col[jnp.ix_(pv_pq_idx, pv_pq_idx)]       # (n_pvpq, n_pvpq) — full pvpq columns

    J21_full = M[jnp.ix_(pv_pq_idx, pv_pq_idx)]      # (n_pvpq, n_pvpq)
    J22_full = L_col[jnp.ix_(pv_pq_idx, pv_pq_idx)]  # (n_pvpq, n_pvpq)

    # Unswitched PV buses (bus_is_pq=False): replace row with identity
    # (J21=0, J22=I) → forces ΔVm = 0 for that bus during the solve.
    eye_pvpq = jnp.eye(n_pvpq)
    zero_pvpq = jnp.zeros((n_pvpq, n_pvpq))
    mask = bus_is_pq[:, None]  # (n_pvpq, 1) broadcast over columns

    J21_masked = jnp.where(mask, J21_full, zero_pvpq)
    J22_masked = jnp.where(mask, J22_full, eye_pvpq)

    top = jnp.concatenate([J11, J12], axis=1)
    bot = jnp.concatenate([J21_masked, J22_masked], axis=1)
    return jnp.concatenate([top, bot], axis=0)  # (2*n_pvpq, 2*n_pvpq)


def ac_power_flow(
    setup: ACPFSetup,
    max_iter: int = 30,
    tol: float = 1e-5,
) -> ACPFResult:
    """Newton-Raphson AC power flow solver (pure JAX, JIT-compilable).

    Uses lax.while_loop for the NR iteration.

    **Q-limit enforcement**: PV buses whose calculated Q exceeds Qg_min/Qg_max
    (stored in setup) are switched to PQ mid-iteration (``bus_is_pq`` mask).
    The Jacobian stays fixed at ``(2*n_pvpq, 2*n_pvpq)`` regardless of how many
    buses switch, satisfying the static-shape requirement of ``lax.while_loop``.
    PV→PQ switching is one-way within a single solve (no PQ→PV recovery).
    If ``Qg_min_bus``/``Qg_max_bus`` are ±inf (default when Q data absent),
    switching never triggers and the solver behaves identically to before.

    Note: Under ``jax.vmap``, all batch elements run for max_iter iterations
    (already-converged elements still execute but do not update). Each
    iteration contains ``jnp.linalg.solve(J, mismatch)`` which is O(n³).
    For large networks with large batch sizes, consider ``lax.fori_loop``
    with a fixed iteration count as a performance-tuning alternative.

    Args:
        setup: Precomputed ACPFSetup from prepare_acpf().
        max_iter: Maximum NR iterations.
        tol: Convergence tolerance on max |mismatch| in p.u.
            Default 1e-5 (float32 precision floor is ~2e-6).

    Returns:
        ACPFResult with voltage, power, branch flow, and convergence info.
    """
    G = setup.Y_bus_real
    B = setup.Y_bus_imag
    P_sched = setup.P_sched
    pv_pq_idx = setup.pv_pq_idx
    # Derive dimensions from array shapes (static at trace time)
    n_pvpq = pv_pq_idx.shape[0]

    Vm0 = setup.Vm_init
    Va0 = setup.Va_init

    tol_arr = jnp.float32(tol)

    # Initial Q-limit state
    init_bus_is_pq = jnp.concatenate([
        jnp.zeros(setup.n_pv, dtype=jnp.bool_),   # PV buses start as non-PQ
        jnp.ones(setup.n_pq, dtype=jnp.bool_),    # PQ buses always PQ
    ])
    init_Q_sched_eff = setup.Q_sched[pv_pq_idx]   # (n_pvpq,) initial Q schedule

    # NR state: (Vm, Va, Q_sched_eff, bus_is_pq, iteration, converged)
    def _cond(state):
        Vm, Va, Q_sched_eff, bus_is_pq, it, conv = state
        return jnp.logical_and(it < max_iter, jnp.logical_not(conv))

    def _body(state):
        Vm, Va, Q_sched_eff, bus_is_pq, it, _conv = state

        cos_Va = jnp.cos(Va)
        sin_Va = jnp.sin(Va)
        Vr = Vm * cos_Va
        Vi = Vm * sin_Va

        Ir = G @ Vr - B @ Vi
        Ii = G @ Vi + B @ Vr

        P_calc = Vr * Ir + Vi * Ii
        Q_calc = Vi * Ir - Vr * Ii

        # --- Q-limit check: update bus_is_pq and Q_sched_eff ---
        Q_at_pvpq = Q_calc[pv_pq_idx]
        Qmin_pvpq = setup.Qg_min_bus[pv_pq_idx]
        Qmax_pvpq = setup.Qg_max_bus[pv_pq_idx]

        over  = Q_at_pvpq > Qmax_pvpq
        under = Q_at_pvpq < Qmin_pvpq

        bus_is_pq = bus_is_pq | over | under   # one-way: once switched stays PQ
        Q_sched_eff = jnp.where(over,  Qmax_pvpq,
                      jnp.where(under, Qmin_pvpq, Q_sched_eff))

        # --- Mismatch: PQ buses use Q equation; PV buses RHS = 0 (identity rows) ---
        dP = P_sched[pv_pq_idx] - P_calc[pv_pq_idx]
        dQ_full = Q_sched_eff - Q_at_pvpq
        dQ_masked = jnp.where(bus_is_pq, dQ_full, 0.0)
        mismatch = jnp.concatenate([dP, dQ_masked])

        max_mis = jnp.max(jnp.abs(mismatch))
        converged = max_mis < tol_arr

        J = _build_jacobian(Vm, Va, G, B, P_calc, Q_calc,
                            pv_pq_idx, bus_is_pq)
        dx = jnp.linalg.solve(J, mismatch)

        Va = Va.at[pv_pq_idx].add(dx[:n_pvpq])
        Vm = Vm.at[pv_pq_idx].add(dx[n_pvpq:])  # identity rows → dx ≈ 0 for PV

        # Belt-and-suspenders: clamp non-switched PV buses back to setpoints
        pv_not_switched = ~bus_is_pq[:setup.n_pv]
        Vm = Vm.at[setup.pv_idx].set(
            jnp.where(pv_not_switched,
                      setup.Vm_setpoint[setup.pv_idx],
                      Vm[setup.pv_idx]))

        return (Vm, Va, Q_sched_eff, bus_is_pq, it + 1, converged)

    init_state = (Vm0, Va0, init_Q_sched_eff, init_bus_is_pq,
                  jnp.int32(0), jnp.bool_(False))
    Vm_f, Va_f, _Q_eff_f, _bus_is_pq_f, iters, conv = lax.while_loop(
        _cond, _body, init_state)

    # Final power calculations
    cos_Va = jnp.cos(Va_f)
    sin_Va = jnp.sin(Va_f)
    Vr = Vm_f * cos_Va
    Vi = Vm_f * sin_Va
    Ir = G @ Vr - B @ Vi
    Ii = G @ Vi + B @ Vr
    P_calc = Vr * Ir + Vi * Ii
    Q_calc = Vi * Ir - Vr * Ii

    # Branch flows
    pf_from, qf_from, pf_to, qf_to = calc_branch_flows(
        Vm_f, Va_f, setup)

    return ACPFResult(
        vm=Vm_f,
        va=Va_f,
        p_calc=P_calc,
        q_calc=Q_calc,
        pf_from=pf_from,
        qf_from=qf_from,
        pf_to=pf_to,
        qf_to=qf_to,
        converged=conv,
        iterations=iters,
    )


def calc_branch_flows(
    Vm: chex.Array,
    Va: chex.Array,
    setup: ACPFSetup,
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    """Compute branch power flows from solved voltages.

    Returns (pf_from, qf_from, pf_to, qf_to) in MW/MVAr.
    """
    base_mva = setup.base_mva
    f = setup.br_from_idx
    t = setup.br_to_idx

    # Voltage at terminals (real arithmetic)
    Vf_r = Vm[f] * jnp.cos(Va[f])
    Vf_i = Vm[f] * jnp.sin(Va[f])
    Vt_r = Vm[t] * jnp.cos(Va[t])
    Vt_i = Vm[t] * jnp.sin(Va[t])

    # Yff*Vf + Yft*Vt  (complex multiply via real arithmetic)
    If_r = (setup.Yff_real * Vf_r - setup.Yff_imag * Vf_i +
            setup.Yft_real * Vt_r - setup.Yft_imag * Vt_i)
    If_i = (setup.Yff_real * Vf_i + setup.Yff_imag * Vf_r +
            setup.Yft_real * Vt_i + setup.Yft_imag * Vt_r)

    It_r = (setup.Ytf_real * Vf_r - setup.Ytf_imag * Vf_i +
            setup.Ytt_real * Vt_r - setup.Ytt_imag * Vt_i)
    It_i = (setup.Ytf_real * Vf_i + setup.Ytf_imag * Vf_r +
            setup.Ytt_real * Vt_i + setup.Ytt_imag * Vt_r)

    # Sf = Vf * conj(If) = (Vf_r + j Vf_i)(If_r - j If_i)
    pf_from = (Vf_r * If_r + Vf_i * If_i) * base_mva
    qf_from = (Vf_i * If_r - Vf_r * If_i) * base_mva
    pf_to = (Vt_r * It_r + Vt_i * It_i) * base_mva
    qf_to = (Vt_i * It_r - Vt_r * It_i) * base_mva

    # Zero out inactive branches
    active = setup.br_status > 0
    pf_from = jnp.where(active, pf_from, 0.0)
    qf_from = jnp.where(active, qf_from, 0.0)
    pf_to = jnp.where(active, pf_to, 0.0)
    qf_to = jnp.where(active, qf_to, 0.0)

    return pf_from, qf_from, pf_to, qf_to


def _voltage_safety_check(
    vm: chex.Array,
    v_min: chex.Array,
    v_max: chex.Array,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Check voltage magnitude limits.

    Returns:
        (v_safe, n_v_violations, cost_voltage)
    """
    over = jnp.maximum(vm - v_max, 0.0)
    under = jnp.maximum(v_min - vm, 0.0)
    violation = over + under
    n_viol = jnp.sum(violation > 0.0).astype(jnp.int32)
    v_safe = n_viol == 0
    cost_v = jnp.sum(violation)
    return v_safe, n_viol, cost_v


def ac_power_flow_with_check(
    setup: ACPFSetup,
    line_cap: chex.Array,
    line_floor: chex.Array,
    v_min: chex.Array,
    v_max: chex.Array,
    max_iter: int = 30,
    tol: float = 1e-5,
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array,
           chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
    """AC power flow + safety check (thermal + voltage).

    Args:
        setup: Precomputed ACPFSetup.
        line_cap: Line capacity upper limits (n_lines,) [MW].
        line_floor: Line capacity lower limits (n_lines,) [MW].
        v_min: Voltage lower limits (n_nodes,) [p.u.].
        v_max: Voltage upper limits (n_nodes,) [p.u.].
        max_iter: NR max iterations.
        tol: NR convergence tolerance.

    Returns:
        A 9-tuple: line_flow_mw, node_injection_mw, vm, va, is_safe,
        n_violations, cost_thermal, cost_voltage, converged.

        ``line_flow_mw`` is the directed from-end active flow (pf_from) for
        use by callers.  Thermal safety uses the maximum of the absolute
        values at both ends to catch losses on the to-side:
            thermal_flow = max(|pf_from|, |pf_to|)

        ``converged`` (bool scalar) indicates whether the NR solver converged.
        When False the returned voltages and flows are non-physical intermediate
        values — callers should handle this flag accordingly.
    """
    result = ac_power_flow(setup, max_iter=max_iter, tol=tol)

    # Directed from-end flow returned to callers (signed, for dispatch logic).
    line_flow_mw = result.pf_from
    node_injection_mw = result.p_calc * setup.base_mva

    # Thermal check: AC losses make |pf_from| ≠ |pf_to|; use the larger end.
    thermal_flow = jnp.maximum(jnp.abs(result.pf_from), jnp.abs(result.pf_to))
    is_thermal_safe, n_thermal, cost_thermal = safety_check(
        thermal_flow, line_cap, line_floor)

    v_safe, n_v_viol, cost_v = _voltage_safety_check(result.vm, v_min, v_max)

    is_safe = jnp.logical_and(is_thermal_safe, v_safe)
    n_violations = n_thermal + n_v_viol

    return (line_flow_mw, node_injection_mw, result.vm, result.va,
            is_safe, n_violations, cost_thermal, cost_v, result.converged)
