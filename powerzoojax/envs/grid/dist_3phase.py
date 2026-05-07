"""Three-Phase Distribution Grid Environment — radial 3-phase BFS.

A radial 3-phase distribution env where the agent controls per-phase
DER injections at non-reference buses and / or attached
``ResourceBundle`` instances; the env runs the BIBC/BCBV/DLF-based
3-phase BFS each step. Differs from ``DistGridEnv`` (single-phase) in
that it preserves per-phase imbalance — necessary for handling
unbalanced loads, single-phase laterals, and voltage-unbalance
constraints, all of which collapse out of a single-phase model.

Power flow: ``bfs_3phase_power_flow`` (BIBC/BCBV/DLF on complex
voltages and currents). All loads, injections, and flows in **per-unit**
relative to ``base_mva``; default ``base_mva = 10`` reflects the small
feeders this env typically runs on (cf. 100 in the single-phase env).
Bundle injections are scattered as **balanced three-phase** (1/3 of
bundle P / Q on each phase) at the bundle's buses — single-phase DER
bundles need a different injection model.

Action (all entries in [−1, 1]):
    Concatenated layout
        ``[der_per_phase (3·(n_nodes − 1)) | bundle_0 | bundle_1 | ...]``
    where ``der_per_phase`` is denormalised to ``[der_low, der_high]``
    in p.u.

Observation:
    [V_A(n), V_B(n), V_C(n),
     P_A/B/C_branch (n_lines each), Q_A/B/C_branch (n_lines each),
     p_load_A/B/C (n each), q_load_A/B/C (n each),
     sin t, cos t, <bundle_obs concatenated>]

Reward:
    reward = −loss_penalty_weight · p_loss_MW

CMDP costs (stacked vector, four channels — one more than single phase):
    cost_voltage_violation  count of buses with v ∉ [v_min, v_max] on
                            any phase.
    cost_thermal_overload   count of lines with per-phase apparent
                            power above ``line_cap``.
    cost_vuf_violation      count of buses where the Fortescue
                            voltage-unbalance factor exceeds
                            ``vuf_max`` (%, default 2.0). Specific to
                            3-phase: VUF measures negative-sequence /
                            positive-sequence voltage magnitude ratio
                            and is typically capped at 2–3 % by
                            distribution codes.
    cost_resource           Σ over bundles of each bundle's
                            ``info["cost_sum"]``.

Resource attachment:
    ``make_dist_3phase_params(resources=(make_battery_bundle(...),))``.
    Bundle states live in ``DistGrid3PhState.resource_states`` and
    auto-reset with the grid; bundle injections are split equally
    across phases at each attached bus.
"""

from functools import partial
from typing import Tuple, Dict, Any

import jax
import jax.numpy as jnp
import jax.tree_util as tu
import chex
from flax import struct

from powerzoojax.envs.base import (
    Environment,
    denormalize_action,
    stack_costs,
    time_features,
)
from powerzoojax.envs.spaces import Box
from powerzoojax.envs.grid.bfs_3phase_power_flow import (
    ThreePhaseTopoData,
    build_3phase_topology,
    bfs_3phase_power_flow,
)


@struct.dataclass
class DistGrid3PhState:
    """3-phase distribution grid state — pure grid quantities only.

    Resource state (battery SOC, etc.) is NOT stored here.
    """
    time_step: chex.Array
    done: chex.Array
    v_mag: chex.Array          # (3*(n_nodes-1),) per-phase voltage magnitudes [p.u.]
    V_real: chex.Array         # (3*(n_nodes-1),) voltage real parts [p.u.]
    V_imag: chex.Array         # (3*(n_nodes-1),) voltage imag parts [p.u.]
    P_branch: chex.Array       # (3*n_lines,) per-phase branch active power [p.u.]
    Q_branch: chex.Array       # (3*n_lines,) per-phase branch reactive power [p.u.]
    p_loss_total: chex.Array   # scalar total active loss [p.u.]
    is_safe: chex.Array
    n_violations: chex.Array
    total_cost: chex.Array
    resource_states: tuple = ()  # tuple[BundleState, ...]; empty = no resources


@struct.dataclass
class DistGrid3PhParams:
    """3-phase distribution grid parameters.

    Scalar configuration fields are marked ``pytree_node=False`` so they
    are treated as static compile-time constants by ``jax.jit``.

    **Action scaling note**: ``der_low`` and ``der_high`` are in **per-unit**
    (p.u.) relative to ``base_mva``.  With the default ``base_mva = 10``,
    ``der_low = -0.1`` corresponds to −1 MW.
    """
    topo: ThreePhaseTopoData = None
    load_P_3ph: chex.Array = None  # (T, 3*(n_nodes-1)) per-phase active load [p.u.]
    load_Q_3ph: chex.Array = None  # (T, 3*(n_nodes-1)) per-phase reactive load [p.u.]
    line_cap: chex.Array = None    # (n_lines,) aggregate MVA capacity; 0 = no limit
    p_load_ref: float = struct.field(pytree_node=False, default=1.0)
    q_load_ref: float = struct.field(pytree_node=False, default=1.0)
    base_mva: float = struct.field(pytree_node=False, default=10.0)
    ref_bus: int = struct.field(pytree_node=False, default=0)
    v_ref_mag: float = struct.field(pytree_node=False, default=1.0)
    v_min: float = struct.field(pytree_node=False, default=0.90)
    v_max: float = struct.field(pytree_node=False, default=1.10)
    vuf_max: float = struct.field(pytree_node=False, default=2.0)
    max_steps: int = struct.field(pytree_node=False, default=48)
    steps_per_day: int = struct.field(pytree_node=False, default=48)
    loss_penalty_weight: float = struct.field(pytree_node=False, default=0.1)
    der_low: float = struct.field(pytree_node=False, default=-0.1)
    der_high: float = struct.field(pytree_node=False, default=0.1)
    bfs_max_iter: int = struct.field(pytree_node=False, default=100)
    bfs_tol: float = struct.field(pytree_node=False, default=1e-6)
    include_der: bool = struct.field(pytree_node=False, default=True)
    resources: tuple = ()


def make_dist_3phase_params(
    n_nodes: int,
    from_nodes,
    to_nodes,
    Z_3ph_pu,
    load_P_3ph: jnp.ndarray,
    load_Q_3ph: jnp.ndarray,
    *,
    ref_bus: int = 0,
    v_ref_mag: float = 1.0,
    v_min: float = 0.90,
    v_max: float = 1.10,
    vuf_max: float = 2.0,
    max_steps: int = 48,
    steps_per_day: int = 48,
    base_mva: float = 10.0,
    loss_penalty_weight: float = 0.1,
    line_cap=None,
    resources: tuple = (),
    include_der: bool = True,
) -> DistGrid3PhParams:
    """Create DistGrid3PhParams from raw 3-phase network data.

    Args:
        n_nodes: Number of buses.
        from_nodes: (n_lines,) from-bus indices.
        to_nodes: (n_lines,) to-bus indices.
        Z_3ph_pu: (n_lines, 3, 3) complex impedance matrices per branch [p.u.].
        load_P_3ph: (T, 3*(n_nodes-1)) per-phase active load [p.u.].
        load_Q_3ph: (T, 3*(n_nodes-1)) per-phase reactive load [p.u.].
        ref_bus: Reference (slack) bus index.
        v_ref_mag: Reference voltage magnitude.
        v_min: Voltage lower bound.
        v_max: Voltage upper bound.
        vuf_max: Maximum voltage unbalance factor (%).
        max_steps: Maximum steps per episode.
        steps_per_day: Steps per day for time encoding.
        base_mva: System base power (MVA).
        loss_penalty_weight: Weight on active-loss reward shaping.
        line_cap: (n_lines,) aggregate apparent-power capacity per line [MVA].
            0 means no limit (thermal check skipped for that line).
            None defaults to all zeros (no thermal limits).
        resources: Tuple of ResourceBundle instances (e.g. BatteryBundle).
            Bundle ``bus_idx`` should be 0-based bus indices (not ref_bus).
            Bundle injection is balanced three-phase (1/3 per phase).

    Returns:
        DistGrid3PhParams ready for DistGrid3PhaseEnv.
    """
    topo = build_3phase_topology(
        n_nodes, from_nodes, to_nodes, Z_3ph_pu,
        ref_bus=ref_bus, v_ref_mag=v_ref_mag,
    )
    load_P = jnp.asarray(load_P_3ph, dtype=jnp.float32)
    load_Q = jnp.asarray(load_Q_3ph, dtype=jnp.float32)
    p_load_ref = max(float(jnp.sum(load_P[0])), 1e-8)
    q_load_ref = max(float(jnp.sum(load_Q[0])), 1e-8)

    n_lines = len(from_nodes)
    if line_cap is None:
        lc = jnp.zeros(n_lines, dtype=jnp.float32)
    else:
        lc = jnp.asarray(line_cap, dtype=jnp.float32)

    return DistGrid3PhParams(
        topo=topo,
        load_P_3ph=load_P,
        load_Q_3ph=load_Q,
        line_cap=lc,
        p_load_ref=p_load_ref,
        q_load_ref=q_load_ref,
        base_mva=base_mva,
        ref_bus=ref_bus,
        v_ref_mag=v_ref_mag,
        v_min=v_min,
        v_max=v_max,
        vuf_max=vuf_max,
        max_steps=max_steps,
        steps_per_day=steps_per_day,
        loss_penalty_weight=loss_penalty_weight,
        include_der=include_der,
        resources=tuple(resources),
    )


class DistGrid3PhaseEnv(Environment):
    """Three-phase distribution grid environment — BFS 3-phase power flow.

    Observation layout:
        V_A (n_nodes,)         — (v - 1) / 0.1
        V_B (n_nodes,)
        V_C (n_nodes,)
        P_A (n_lines,)         — P_branch / p_load_ref
        P_B (n_lines,)
        P_C (n_lines,)
        Q_A (n_lines,)         — Q_branch / q_load_ref
        Q_B (n_lines,)
        Q_C (n_lines,)
        p_load_A (n_nodes,)    — p_load / p_load_ref
        p_load_B (n_nodes,)
        p_load_C (n_nodes,)
        q_load_A (n_nodes,)    — q_load / q_load_ref
        q_load_B (n_nodes,)
        q_load_C (n_nodes,)
        [time_sin, time_cos]  (2,)
        battery_soc (n_batteries,)  — omitted if no resources

    """

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: DistGrid3PhParams,
    ) -> Tuple[chex.Array, DistGrid3PhState]:
        P = params.load_P_3ph[0]
        Q = params.load_Q_3ph[0]
        result = bfs_3phase_power_flow(params.topo, P, Q, params.bfs_max_iter, params.bfs_tol)

        resource_states = tuple(b.reset(key) for b in params.resources)

        state = DistGrid3PhState(
            time_step=jnp.int32(0),
            done=jnp.bool_(False),
            v_mag=result.v_mag,
            V_real=result.V_real,
            V_imag=result.V_imag,
            P_branch=result.P_branch,
            Q_branch=result.Q_branch,
            p_loss_total=_compute_3ph_loss(result, params.topo),
            is_safe=jnp.bool_(True),
            n_violations=jnp.int32(0),
            total_cost=jnp.float32(0.0),
            resource_states=resource_states,
        )
        obs = self._get_obs(state, params)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, key: chex.PRNGKey, state: DistGrid3PhState,
        action: chex.Array, params: DistGrid3PhParams,
    ) -> Tuple[chex.Array, DistGrid3PhState, chex.Array, chex.Array, chex.Array, Dict[str, Any]]:
        """Run one environment step.

        Args:
            key: PRNG key.
            state: Current grid state.
            action: When ``include_der=True``:
                ``(3*(n_nodes-1) + sum(bundle.action_dim),)`` in ``[-1, 1]``.
                When ``include_der=False``:
                ``(sum(bundle.action_dim),)`` — bundle actions only.
            params: Environment parameters.
        """
        t_idx = state.time_step % params.load_P_3ph.shape[0]
        P_load = params.load_P_3ph[t_idx]
        Q_load = params.load_Q_3ph[t_idx]
        n_nonref_3ph = P_load.shape[0]  # 3*(n_nodes-1), static

        if params.include_der:
            der_action = action[:n_nonref_3ph]
            der_low = jnp.full((n_nonref_3ph,), params.der_low, dtype=jnp.float32)
            der_high = jnp.full((n_nonref_3ph,), params.der_high, dtype=jnp.float32)
            physical_action = denormalize_action(der_action, der_low, der_high)
            offset = n_nonref_3ph
        else:
            physical_action = jnp.zeros(n_nonref_3ph, dtype=jnp.float32)
            offset = 0

        # --- Bundle processing (balanced 3-phase scatter) ---
        ref_bus = params.ref_bus  # Python int (pytree_node=False)
        ctx = {}
        new_resource_states_list = []
        bundle_injection_p_3ph_mw = jnp.zeros(n_nonref_3ph, dtype=jnp.float32)
        bundle_injection_q_3ph_mvar = jnp.zeros(n_nonref_3ph, dtype=jnp.float32)
        bundle_cost = jnp.float32(0.0)
        for i, bundle in enumerate(params.resources):
            a_slice = action[offset: offset + bundle.action_dim]
            offset += bundle.action_dim
            new_bs, p_inj, q_inj, _, cost_info = bundle.step(
                state.resource_states[i], a_slice, ctx)
            new_resource_states_list.append(new_bs)
            nonref_idx = jnp.where(
                bundle.bus_idx < ref_bus, bundle.bus_idx, bundle.bus_idx - 1)
            p_per_phase = p_inj / 3.0
            q_per_phase = q_inj / 3.0
            for ph in range(3):
                bundle_injection_p_3ph_mw = bundle_injection_p_3ph_mw.at[
                    3 * nonref_idx + ph].add(p_per_phase)
                bundle_injection_q_3ph_mvar = bundle_injection_q_3ph_mvar.at[
                    3 * nonref_idx + ph].add(q_per_phase)
            bundle_cost = bundle_cost + cost_info.get(
                "cost_sum",
                cost_info.get("cost", jnp.float32(0.0)),
            )
        new_resource_states = tuple(new_resource_states_list)
        bundle_p_pu = bundle_injection_p_3ph_mw / params.base_mva
        bundle_q_pu = bundle_injection_q_3ph_mvar / params.base_mva

        P_net = P_load - physical_action - bundle_p_pu
        Q_net = Q_load - bundle_q_pu

        result = bfs_3phase_power_flow(params.topo, P_net, Q_net, params.bfs_max_iter, params.bfs_tol)

        # --- Voltage violations (per-phase, count-based) ---
        v_violation = jnp.logical_or(
            result.v_mag < params.v_min,
            result.v_mag > params.v_max,
        )
        n_v_violations = jnp.sum(v_violation).astype(jnp.int32)

        # --- VUF check (Fortescue symmetrical components) ---
        vuf_percent = _compute_vuf(result.V_real, result.V_imag)
        n_vuf_violations = jnp.sum(vuf_percent > params.vuf_max).astype(jnp.int32)

        # --- Thermal: exact per-phase apparent power from endpoint voltages and branch current ---
        # ``bfs_3phase_power_flow`` stores branch P/Q diagnostics using the
        # receiving-end voltage approximation.  For thermal checks we instead
        # rebuild per-phase apparent power from the branch current and both
        # endpoint voltages, then compare the larger endpoint magnitude against
        # the per-phase share of the line MVA rating.
        if params.line_cap is not None:
            n_l_violations, s_over, _ = _compute_3ph_thermal_metrics(
                result, params.topo, params.line_cap,
                params.base_mva, params.ref_bus, params.v_ref_mag,
            )
        else:
            n_l_violations = jnp.int32(0)
            s_over = jnp.float32(0.0)

        n_violations = n_v_violations + n_l_violations + n_vuf_violations
        is_safe = n_violations == 0

        # --- Losses ---
        p_loss_pu = _compute_3ph_loss(result, params.topo)
        p_loss_mw = p_loss_pu * params.base_mva

        I_sq = result.I_branch_real ** 2 + result.I_branch_imag ** 2
        I_sq_mat = I_sq.reshape(-1, 3)
        q_loss_pu = jnp.sum(I_sq_mat * params.topo.Z_diag_x)
        q_loss_mvar = q_loss_pu * params.base_mva

        reward = -params.loss_penalty_weight * p_loss_mw

        new_time = state.time_step + 1
        done = new_time >= params.max_steps

        new_state = DistGrid3PhState(
            time_step=new_time,
            done=done,
            v_mag=result.v_mag,
            V_real=result.V_real,
            V_imag=result.V_imag,
            P_branch=result.P_branch,
            Q_branch=result.Q_branch,
            p_loss_total=p_loss_pu,
            is_safe=is_safe,
            n_violations=n_violations,
            total_cost=state.total_cost + n_violations.astype(jnp.float32),
            resource_states=new_resource_states,
        )

        # Auto-reset
        _, k_reset = jax.random.split(key)
        _, reset_state = self.reset(k_reset, params)
        final_state = tu.tree_map(
            lambda n, r: jnp.where(done, r, n), new_state, reset_state)
        final_state = jax.lax.stop_gradient(final_state)
        final_obs = self._get_obs(final_state, params)

        cost_voltage_violation = n_v_violations.astype(jnp.float32)
        cost_thermal_overload = n_l_violations.astype(jnp.float32)
        cost_vuf_violation = n_vuf_violations.astype(jnp.float32)
        cost_resource = bundle_cost
        cost_sum = cost_voltage_violation + cost_thermal_overload + cost_vuf_violation + cost_resource

        v_over = jnp.sum(jnp.maximum(result.v_mag - params.v_max, 0.0))
        v_under = jnp.sum(jnp.maximum(params.v_min - result.v_mag, 0.0))
        vuf_over = jnp.sum(jnp.maximum(vuf_percent - params.vuf_max, 0.0))
        cost_continuous = v_over + v_under + vuf_over + s_over

        costs = stack_costs(
            cost_voltage_violation,
            cost_thermal_overload,
            cost_vuf_violation,
            cost_resource,
        )
        info = {
            "cost_voltage_violation": cost_voltage_violation,
            "cost_thermal_overload": cost_thermal_overload,
            "cost_vuf_violation": cost_vuf_violation,
            "cost_resource": cost_resource,
            "cost_continuous": cost_continuous,
            "cost_sum": cost_sum,
            "is_safe": is_safe,
            "goal_met": is_safe,
            "n_violations": n_violations,
            "p_loss_MW": p_loss_mw,
            "q_loss_MVAr": q_loss_mvar,
            "max_vuf_percent": jnp.max(vuf_percent),
            "bfs_converged": result.converged,
        }
        return final_obs, final_state, reward, costs, done, info

    def _get_obs(self, state: DistGrid3PhState, params: DistGrid3PhParams) -> chex.Array:
        """Build per-phase observation vector — grid quantities + bundle obs.

        Layout:
            V_A, V_B, V_C  (n_nodes each)  — (v - 1) / 0.1
            P_A, P_B, P_C  (n_lines each)  — P_branch / p_load_ref
            Q_A, Q_B, Q_C  (n_lines each)  — Q_branch / q_load_ref
            p_load_A/B/C   (n_nodes each)  — p_load / p_load_ref
            q_load_A/B/C   (n_nodes each)  — q_load / q_load_ref
            [time_sin, time_cos]  (2,)
            <bundle_obs>
        """
        n_nodes = params.topo.n_nodes
        n_lines = params.topo.n_lines

        v_nonref = state.v_mag.reshape(-1, 3)
        v_ref = jnp.full((1, 3), params.v_ref_mag)
        v_all = jnp.concatenate([v_ref, v_nonref], axis=0)
        v_norm = (v_all - 1.0) / 0.1
        V_A = v_norm[:, 0]
        V_B = v_norm[:, 1]
        V_C = v_norm[:, 2]

        p_mat = state.P_branch.reshape(-1, 3) / params.p_load_ref
        q_mat = state.Q_branch.reshape(-1, 3) / params.q_load_ref
        P_A, P_B, P_C = p_mat[:, 0], p_mat[:, 1], p_mat[:, 2]
        Q_A, Q_B, Q_C = q_mat[:, 0], q_mat[:, 1], q_mat[:, 2]

        t_idx = state.time_step % params.load_P_3ph.shape[0]
        pload_nonref = params.load_P_3ph[t_idx].reshape(-1, 3)
        qload_nonref = params.load_Q_3ph[t_idx].reshape(-1, 3)
        pload_all = jnp.concatenate([jnp.zeros((1, 3)), pload_nonref], axis=0) / params.p_load_ref
        qload_all = jnp.concatenate([jnp.zeros((1, 3)), qload_nonref], axis=0) / params.q_load_ref
        pA, pB, pC = pload_all[:, 0], pload_all[:, 1], pload_all[:, 2]
        qA, qB, qC = qload_all[:, 0], qload_all[:, 1], qload_all[:, 2]

        t_sin, t_cos = time_features(state.time_step, params.steps_per_day)

        obs_parts = [
            V_A, V_B, V_C,
            P_A, P_B, P_C,
            Q_A, Q_B, Q_C,
            pA, pB, pC,
            qA, qB, qC,
            jnp.stack([t_sin, t_cos]),
        ]

        ctx = {}
        for i, bundle in enumerate(params.resources):
            obs_parts.append(bundle.observe(state.resource_states[i], ctx))

        return jnp.concatenate(obs_parts)

    def observation_space(self, params: DistGrid3PhParams) -> Box:
        n = params.topo.n_nodes
        nl = params.topo.n_lines
        obs_dim = 9 * n + 6 * nl + 2
        obs_dim += sum(b.obs_dim for b in params.resources)
        return Box(
            low=jnp.full((obs_dim,), -jnp.inf),
            high=jnp.full((obs_dim,), jnp.inf),
            shape=(obs_dim,),
            dtype=jnp.float32,
        )

    def action_space(self, params: DistGrid3PhParams) -> Box:
        dim = sum(b.action_dim for b in params.resources)
        if params.include_der:
            dim += params.topo.n_lines * 3
        return Box(
            low=jnp.full((dim,), -1.0, dtype=jnp.float32),
            high=jnp.full((dim,), 1.0, dtype=jnp.float32),
            shape=(dim,),
            dtype=jnp.float32,
        )

    @property
    def name(self):
        return "DistGrid3PhaseEnv"

    def constraint_names(self, params: DistGrid3PhParams) -> tuple[str, ...]:
        return (
            "voltage_violation",
            "thermal_overload",
            "vuf_violation",
            "resource",
        )


# ---------------------------------------------------------------------------
# Helper functions (pure JAX, JIT-friendly)
# ---------------------------------------------------------------------------

def _compute_3ph_loss(result, topo):
    """Compute total 3-phase active loss from I²R (p.u.)."""
    I_sq = result.I_branch_real ** 2 + result.I_branch_imag ** 2
    I_sq_mat = I_sq.reshape(-1, 3)  # (n_lines, 3)
    p_loss_mat = I_sq_mat * topo.Z_diag_r  # (n_lines, 3)
    return jnp.sum(p_loss_mat)


def _compute_vuf(V_real, V_imag):
    """Compute Voltage Unbalance Factor (VUF) per node via Fortescue.

    VUF = |V_neg| / |V_pos| * 100%

    Args:
        V_real: (3*(n_nodes-1),) real part of voltages, node-major ABC order.
        V_imag: (3*(n_nodes-1),) imag part of voltages.

    Returns:
        vuf_percent: (n_nodes-1,) VUF in percent for each non-ref node.
    """
    Vr = V_real.reshape(-1, 3)  # (n_nodes-1, 3)
    Vi = V_imag.reshape(-1, 3)

    # Fortescue operator: α = exp(j*2π/3)
    alpha_r = jnp.float32(-0.5)
    alpha_i = jnp.float32(0.8660254037844386)  # sqrt(3)/2
    alpha2_r = jnp.float32(-0.5)
    alpha2_i = jnp.float32(-0.8660254037844386)

    # V_pos = (Va + α*Vb + α²*Vc) / 3
    Va_r, Va_i = Vr[:, 0], Vi[:, 0]
    Vb_r, Vb_i = Vr[:, 1], Vi[:, 1]
    Vc_r, Vc_i = Vr[:, 2], Vi[:, 2]

    # α * Vb
    aVb_r = alpha_r * Vb_r - alpha_i * Vb_i
    aVb_i = alpha_r * Vb_i + alpha_i * Vb_r
    # α² * Vc
    a2Vc_r = alpha2_r * Vc_r - alpha2_i * Vc_i
    a2Vc_i = alpha2_r * Vc_i + alpha2_i * Vc_r

    Vpos_r = (Va_r + aVb_r + a2Vc_r) / 3.0
    Vpos_i = (Va_i + aVb_i + a2Vc_i) / 3.0

    # V_neg = (Va + α²*Vb + α*Vc) / 3
    a2Vb_r = alpha2_r * Vb_r - alpha2_i * Vb_i
    a2Vb_i = alpha2_r * Vb_i + alpha2_i * Vb_r
    aVc_r = alpha_r * Vc_r - alpha_i * Vc_i
    aVc_i = alpha_r * Vc_i + alpha_i * Vc_r

    Vneg_r = (Va_r + a2Vb_r + aVc_r) / 3.0
    Vneg_i = (Va_i + a2Vb_i + aVc_i) / 3.0

    v_pos_mag = jnp.sqrt(Vpos_r ** 2 + Vpos_i ** 2)
    v_neg_mag = jnp.sqrt(Vneg_r ** 2 + Vneg_i ** 2)

    # VUF = |V_neg| / |V_pos| * 100, with safe division
    vuf = jnp.where(v_pos_mag > 1e-6, v_neg_mag / v_pos_mag * 100.0, 0.0)
    return vuf


def _reconstruct_3ph_bus_voltages(
    V_real: chex.Array,
    V_imag: chex.Array,
    n_nodes: int,
    ref_bus: int,
    v_ref_mag: float,
):
    """Rebuild full-bus ABC voltages from the non-reference BFS state."""
    V_nonref_r = V_real.reshape(n_nodes - 1, 3)
    V_nonref_i = V_imag.reshape(n_nodes - 1, 3)

    nonref_bus_idx = jnp.asarray(
        [bus for bus in range(n_nodes) if bus != ref_bus], dtype=jnp.int32)

    V_bus_r = jnp.zeros((n_nodes, 3), dtype=jnp.float32)
    V_bus_i = jnp.zeros((n_nodes, 3), dtype=jnp.float32)
    V_bus_r = V_bus_r.at[nonref_bus_idx].set(V_nonref_r)
    V_bus_i = V_bus_i.at[nonref_bus_idx].set(V_nonref_i)

    ref_angles = jnp.deg2rad(jnp.array([0.0, -120.0, 120.0], dtype=jnp.float32))
    V_ref_r = jnp.float32(v_ref_mag) * jnp.cos(ref_angles)
    V_ref_i = jnp.float32(v_ref_mag) * jnp.sin(ref_angles)
    V_bus_r = V_bus_r.at[ref_bus].set(V_ref_r)
    V_bus_i = V_bus_i.at[ref_bus].set(V_ref_i)

    return V_bus_r, V_bus_i


def _compute_3ph_thermal_metrics(
    result,
    topo,
    line_cap: chex.Array,
    base_mva: float,
    ref_bus: int,
    v_ref_mag: float,
):
    """Thermal metrics from exact endpoint apparent power per phase.

    Branch currents are exact outputs of the three-phase BFS solver.  Rebuild
    endpoint complex power as ``S = V * conj(I)`` at both ends and use the
    larger endpoint magnitude as the thermal loading for each energized phase.
    """
    V_bus_r, V_bus_i = _reconstruct_3ph_bus_voltages(
        result.V_real, result.V_imag, topo.n_nodes, ref_bus, v_ref_mag)

    I_r = result.I_branch_real.reshape(-1, 3)
    I_i = result.I_branch_imag.reshape(-1, 3)

    V_send_r = V_bus_r[topo.from_nodes]
    V_send_i = V_bus_i[topo.from_nodes]
    V_recv_r = V_bus_r[topo.to_nodes]
    V_recv_i = V_bus_i[topo.to_nodes]

    P_send = V_send_r * I_r + V_send_i * I_i
    Q_send = V_send_i * I_r - V_send_r * I_i
    P_recv = V_recv_r * I_r + V_recv_i * I_i
    Q_recv = V_recv_i * I_r - V_recv_r * I_i

    S_send = jnp.sqrt(jnp.maximum(P_send ** 2 + Q_send ** 2, 0.0)) * jnp.float32(base_mva)
    S_recv = jnp.sqrt(jnp.maximum(P_recv ** 2 + Q_recv ** 2, 0.0)) * jnp.float32(base_mva)
    S_phase = jnp.maximum(S_send, S_recv)

    per_phase_cap = line_cap / topo.line_phase_count
    has_cap = (line_cap > 0.0).astype(jnp.float32)
    thermal_mask = has_cap[:, None] * topo.line_phase_mask
    overloaded = (S_phase > per_phase_cap[:, None]).astype(jnp.float32)
    n_l_violations = jnp.sum(overloaded * thermal_mask).astype(jnp.int32)
    s_over = jnp.sum(
        jnp.maximum(S_phase - per_phase_cap[:, None], 0.0) * thermal_mask)
    return n_l_violations, s_over, S_phase
