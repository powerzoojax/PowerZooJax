"""Transmission Grid Environment — DC / AC power flow with optional OPF.

A transmission-network env where the agent dispatches generators each
step (or the env's OPF solver does, depending on ``solver_mode``), an
optional set of ``ResourceBundle`` instances inject DER power at named
buses, and the resulting flows / voltages are checked against thermal
and voltage limits. Reward is the (negative, scaled) cost of
generation; constraint violations flow through a 4-channel CMDP cost
vector.

Mode matrix (two static flags from ``GridParams``):

                  | physics = 0 (DC)              | physics = 1 (AC)
    ──────────────┼───────────────────────────────┼──────────────────────────────
    solver_mode=0 | agent dispatches, DC PF only  | agent dispatches, AC PF
                  |                               | overwrites DC results
    solver_mode=1 | DCOPF dispatches, DC PF       | DCOPF dispatches, AC PF
                  | results retained              | overwrites DC results
    solver_mode=2 | ACOPF dispatches AND solves AC PF (physics flag ignored)

In all modes the slack bus absorbs the system imbalance via
``dc_power_flow``'s ``actual_unit_power`` adjustment (so the agent
cannot exploit free slack energy).

``CaseData.line_cap`` (MATPOWER ``rateA``) is in MVA end-to-end. DC
mode treats it as an active-power cap (MW, Q ≈ 0). AC PF and ACOPF use
apparent power ``√(P² + Q²)`` against that limit; ACOPF prepares
``line_cap_pu = line_cap / base_mva`` for the solver and post-checks
match AC PF semantics.

Action (all entries in [−1, 1]):
    Concatenated layout
        ``[unit_dispatch (n_units) | bundle_0 | bundle_1 | ...]``
    Setting ``include_unit_dispatch=False`` drops the leading
    ``n_units`` block — useful when a separate dispatcher / OPF owns
    generation setpoints and the agent only controls DERs.

Observation:
    DC : [line_flow / cap, load / total_cap, unit_p / p_max, sin t,
          cos t, <bundle_obs concatenated>]
    AC : [|S| / cap, vm, load / total_cap, unit_p / p_max, sin t,
          cos t, <bundle_obs concatenated>]
    where ``|S| = √(P² + Q²)`` per line. Bundle observations are
    appended in resource declaration order.

Reward:
    reward = −reward_scale · generation_cost(unit_power)
    where TC is the ``a/3 · p³ + b/2 · p² + c · p`` integral of the
    project-wide quadratic-MC convention.

CMDP costs (stacked vector, four channels):
    cost_thermal_overload   ``safety_check`` (MW, DC) or
                            ``ac_thermal_check`` (MVA, AC), weighted by
                            ``cost_thermal_weight``.
    cost_voltage_violation  voltage out-of-band (AC modes only; 0 in DC).
    cost_power_balance      slack overload — actual unit power above
                            p_max or below p_min, summed in MW.
    cost_resource           Σ over bundles of each bundle's
                            ``info["cost_sum"]``; the env does not
                            interpret bundle costs further.

Resource attachment:
    ``make_trans_params(resources=(make_battery_bundle(case,
    bus_ids=[1,3]), ...))``. Bundle states live inside
    ``TransGridState.resource_states`` and auto-reset with the grid;
    bundle injections are subtracted from node load before clearing /
    flow so the OPF / PF sees the residual demand.
"""

from __future__ import annotations

from functools import partial
from typing import Tuple, Dict, Any

import jax
import jax.numpy as jnp
import jax.tree_util as tu
import chex
from flax import struct
from powerzoojax.envs.grid.power_flow import (
    ac_thermal_check,
    dc_power_flow,
    dc_power_flow_with_check,
    safety_check,
    compute_generation_cost,
    proportional_dispatch,
)
from powerzoojax.envs.grid.ac_power_flow import (
    ACPFSetup,
    prepare_acpf,
    ac_power_flow,
    _voltage_safety_check,
)
from powerzoojax.envs.grid.dc_opf import (
    DCOPFSetup,
    prepare_dcopf,
    dc_opf,
)
from powerzoojax.envs.base import (
    Environment,
    denormalize_action,
    stack_costs,
    time_features,
)
from powerzoojax.envs.spaces import Box
from powerzoojax.case.case_data import CaseData
from powerzoojax.envs.grid.base import GridState, GridParams


# ============ State / Params ============

@struct.dataclass
class TransGridState(GridState):
    """TransGridEnv state (extends GridState) — pure grid quantities only.

    Resource state (battery SOC, etc.) is NOT stored here; it lives in the
    caller's scan carry.  The grid env only sees ``external_injection_mw``
    passed through ``step()``.
    """
    pass


@struct.dataclass
class TransGridParams(GridParams):
    """TransGridEnv parameters (extends GridParams).

    Attributes:
        case: CaseData with PTDF, limits, cost coefficients, etc.
        reward_scale: Scaling factor for generation cost in reward.
        acpf_setup: Precomputed ACPF data (None for DC-only mode).
        dcopf_setup: Precomputed DCOPF data (None unless solver_mode=1).
        acopf_setup: Precomputed ACOPF data (None unless solver_mode=2).
        resources: Tuple of ResourceBundle instances for grid attachment.
            Each bundle's ``bus_idx`` must be valid internal node indices.
            Action layout: ``[unit_actions | bundle_0_actions | bundle_1_actions | ...]``.
    """
    case: CaseData = None
    reward_scale: float = 0.01
    acpf_setup: ACPFSetup = None
    dcopf_setup: DCOPFSetup = None
    acopf_setup: ACOPFSetup = None
    include_unit_dispatch: bool = struct.field(pytree_node=False, default=True)
    resources: tuple = ()


# ============ Factory ============

def make_trans_params(
    case: CaseData,
    load_profiles: chex.Array = None,
    max_steps: int = 48,
    delta_t_hours: float = 0.5,
    steps_per_day: int = 48,
    cost_thermal_weight: float = 1.0,
    reward_scale: float = 0.01,
    physics: int = 0,
    solver_mode: int = 0,
    resources: tuple = (),
    include_unit_dispatch: bool = True,
) -> TransGridParams:
    """Create TransGridParams from a CaseData and optional load profiles.

    If load_profiles is None, a flat profile at (d_max + d_min) / 2 is used.

    Args:
        case: CaseData instance.
        load_profiles: (T, n_loads) load demand array [MW].
        max_steps: Maximum steps per episode.
        delta_t_hours: Time step duration in hours.
        steps_per_day: Steps per day.
        cost_thermal_weight: Weight for thermal overload cost.
        reward_scale: Multiplier for generation cost in reward.
        physics: 0=DC, 1=AC.
        solver_mode: 0=PF, 1=DCOPF, 2=ACOPF.
        resources: Tuple of ResourceBundle instances (e.g. BatteryBundle).
            Use ``make_battery_bundle(case, bus_ids=[...])`` to create bundles.
    """
    if load_profiles is None:
        mid_load = (case.load_d_max + case.load_d_min) / 2.0
        load_profiles = jnp.tile(mid_load[None, :], (max_steps, 1))

    acpf_setup = None
    if physics == 1:
        acpf_setup = prepare_acpf(case)

    dcopf_setup = None
    if solver_mode == 1:
        dcopf_setup = prepare_dcopf(case)

    acopf_setup = None
    if solver_mode == 2:
        from powerzoojax.envs.grid.ac_opf import prepare_acopf as prepare_acopf_opf
        acopf_setup = prepare_acopf_opf(case)

    return TransGridParams(
        case=case,
        load_profiles=load_profiles,
        max_steps=max_steps,
        delta_t_hours=delta_t_hours,
        steps_per_day=steps_per_day,
        cost_thermal_weight=cost_thermal_weight,
        reward_scale=reward_scale,
        physics=physics,
        solver_mode=solver_mode,
        acpf_setup=acpf_setup,
        dcopf_setup=dcopf_setup,
        acopf_setup=acopf_setup,
        include_unit_dispatch=include_unit_dispatch,
        resources=tuple(resources),
    )


# ============ Helper pure functions ============

def _get_node_load(case: CaseData, load_demand: chex.Array) -> chex.Array:
    """Map per-load demand to per-node demand."""
    return case.nodes_loads_map @ load_demand


def _run_dcopf_step(
    dcopf_setup: DCOPFSetup,
    node_load_mw: chex.Array,
    case: CaseData,
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
    """Run DCOPF for a single step.

    Returns:
        (unit_power_mw, line_flow_mw, node_inj, is_safe, n_violations, cost_thermal)
    """
    result = dc_opf(dcopf_setup, node_load_mw)
    unit_power_mw = result.unit_power
    line_flow_mw = result.line_flow
    node_inj = result.node_injection

    is_safe, n_violations, cost_thermal = safety_check(
        line_flow_mw, case.line_cap, case.line_floor)

    return unit_power_mw, line_flow_mw, node_inj, is_safe, n_violations, cost_thermal


def _run_acopf_step(
    acopf_setup: ACOPFSetup,
    node_load_mw: chex.Array,
    case: CaseData,
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array,
           chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
    """Run ACOPF for a single step.

    Returns:
        (unit_power_mw, line_flow_mw, line_flow_q_mw, node_inj, is_safe,
         n_violations, cost_thermal, vm, va, q_gen, cost_voltage, line_viol_mva)
    """
    node_qd = case.node_qd if case.node_qd is not None else jnp.zeros_like(node_load_mw)
    from powerzoojax.envs.grid.ac_opf import ac_opf
    result = ac_opf(acopf_setup, node_load_mw, node_qd, max_iter=100)

    unit_power_mw = result.unit_power
    q_gen = result.q_gen
    vm = result.vm
    va = result.va
    line_flow_mw = result.line_flow_p
    line_flow_q_mw = result.line_flow_q

    node_inj = case.nodes_units_map @ unit_power_mw - node_load_mw

    zlines = jnp.zeros_like(line_flow_mw)
    is_thermal_safe, n_thermal, cost_thermal = ac_thermal_check(
        result.line_flow_p, result.line_flow_q, case.line_cap,
        zlines, zlines, use_both_ends=False)

    v_safe, n_v_viol, cost_voltage = _voltage_safety_check(
        vm, case.node_v_min, case.node_v_max)

    is_safe = jnp.logical_and(is_thermal_safe, v_safe)
    n_violations = n_thermal + n_v_viol

    return (unit_power_mw, line_flow_mw, line_flow_q_mw, node_inj, is_safe, n_violations,
            cost_thermal, vm, va, q_gen, cost_voltage, result.line_viol_mva)


def _run_ac_step(
    acpf_setup: ACPFSetup,
    unit_power_mw: chex.Array,
    node_load_mw: chex.Array,
    case: CaseData,
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array,
           chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
    """Run AC power flow for a single step.

    Modifies the ACPF setup's P/Q schedules based on current unit dispatch
    and load, then runs NR.

    Returns:
        (line_flow_mw, node_inj, is_safe, n_violations, cost_thermal,
         vm, va, q_flow, cost_voltage)
    """
    base_mva = acpf_setup.base_mva

    # Aggregate unit power to bus-level P schedule
    Pg_bus = jnp.zeros(acpf_setup.Y_bus_real.shape[0])
    Pg_bus = Pg_bus.at[acpf_setup.gen_bus_idx].add(unit_power_mw / base_mva)
    Pd_bus = node_load_mw / base_mva

    # Q schedule from original setup (load reactive power)
    # Sign convention: Qd_bus is positive for consumption (MATPOWER convention),
    # but Q_sched needs generation-positive form, hence the negation.
    Qd_bus = acpf_setup.Qd_mw / base_mva

    P_sched_new = Pg_bus - Pd_bus
    Q_sched_new = -Qd_bus  # Q_gen_sched is 0 for PQ buses, solved for PV/slack

    setup_updated = acpf_setup.replace(
        P_sched=P_sched_new,
        Q_sched=Q_sched_new,
    )

    result = ac_power_flow(setup_updated, max_iter=30, tol=1e-5)

    line_flow_mw = result.pf_from
    node_inj = result.p_calc * base_mva

    # Thermal safety (apparent power MVA, both ends; matches ACOPF |S| semantics)
    is_thermal_safe, n_thermal, cost_thermal = ac_thermal_check(
        result.pf_from, result.qf_from, case.line_cap,
        result.pf_to, result.qf_to, use_both_ends=True)

    # Voltage safety
    v_safe, n_v_viol, cost_voltage = _voltage_safety_check(
        result.vm, case.node_v_min, case.node_v_max)

    is_safe = jnp.logical_and(is_thermal_safe, v_safe)
    n_violations = n_thermal + n_v_viol

    return (line_flow_mw, node_inj, is_safe, n_violations, cost_thermal,
            result.vm, result.va, result.qf_from, cost_voltage)


# ============ Environment ============

class TransGridEnv(Environment):
    """Transmission grid environment — DC / AC Power Flow mode.

    The agent dispatches generators every time step. The environment
    computes power flow (DC or AC), checks safety, and returns an
    observation of normalised flows, loads, and unit outputs.
    """

    # ====== RL Interface Methods ======

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: TransGridParams,
    ) -> Tuple[chex.Array, TransGridState]:
        case = params.case

        load_demand = params.load_profiles[0]
        node_load_mw = _get_node_load(case, load_demand)
        total_load = jnp.sum(node_load_mw)

        unit_power_mw = proportional_dispatch(total_load, case.unit_p_min, case.unit_p_max)

        # DC power flow (always computed for baseline)
        line_flow_mw, node_inj, actual_unit_power = dc_power_flow(
            case, unit_power_mw, node_load_mw)

        resource_states = tuple(b.reset(key) for b in params.resources)

        state = TransGridState(
            time_step=jnp.int32(0),
            done=jnp.bool_(False),
            unit_power_mw=actual_unit_power,
            line_flow_mw=line_flow_mw,
            node_injection_mw=node_inj,
            is_safe=jnp.bool_(True),
            n_violations=jnp.int32(0),
            total_cost=jnp.float32(0.0),
            vm=jnp.ones_like(node_inj),
            va=jnp.zeros_like(node_inj),
            q_gen=jnp.zeros_like(unit_power_mw),
            line_flow_q_mw=jnp.zeros_like(line_flow_mw),
            resource_states=resource_states,
        )
        obs = self._get_obs(state, params)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: TransGridState,
        action: chex.Array,
        params: TransGridParams,
    ) -> Tuple[chex.Array, TransGridState, chex.Array, chex.Array, chex.Array, Dict[str, Any]]:
        """Run one environment step.

        Args:
            key: PRNG key.
            state: Current grid state.
            action: When ``include_unit_dispatch=True``:
                ``(n_units + sum(bundle.action_dim),)`` in ``[-1, 1]``.
                When ``include_unit_dispatch=False``:
                ``(sum(bundle.action_dim),)`` — bundle actions only.
            params: Environment parameters.
        """
        case = params.case

        t_idx = state.time_step % params.load_profiles.shape[0]
        load_demand = params.load_profiles[t_idx]
        node_load_mw = _get_node_load(case, load_demand)

        # --- Bundle processing (trace-time unrolled) ---
        n_units = case.unit_p_min.shape[0]  # static shape
        if params.include_unit_dispatch:
            unit_action = action[:n_units]
            offset = n_units
        else:
            unit_action = jnp.zeros(n_units, dtype=jnp.float32)
            offset = 0

        ctx = {"dt_hours": params.delta_t_hours}
        new_resource_states_list = []
        bundle_injection = jnp.zeros(node_load_mw.shape[0], dtype=jnp.float32)
        bundle_cost = jnp.float32(0.0)
        for i, bundle in enumerate(params.resources):
            a_slice = action[offset: offset + bundle.action_dim]
            offset += bundle.action_dim
            new_bs, p_inj, q_inj, _, cost_info = bundle.step(
                state.resource_states[i], a_slice, ctx)
            new_resource_states_list.append(new_bs)
            bundle_injection = bundle_injection.at[bundle.bus_idx].add(p_inj)
            bundle_cost = bundle_cost + cost_info.get(
                "cost_sum",
                cost_info.get("cost", jnp.float32(0.0)),
            )
        new_resource_states = tuple(new_resource_states_list)
        node_load_mw = node_load_mw - bundle_injection

        # --- Dispatch (Python if on static solver_mode) ---
        if params.dcopf_setup is not None:
            unit_power_mw, line_flow_mw, node_inj, is_safe, n_violations, cost_thermal = (
                _run_dcopf_step(params.dcopf_setup, node_load_mw, case))
        else:
            agent_dispatch = denormalize_action(unit_action, case.unit_p_min, case.unit_p_max)
            line_flow_mw, node_inj, unit_power_mw, is_safe, n_violations, cost_thermal = (
                dc_power_flow_with_check(case, agent_dispatch, node_load_mw))

        # Default DC-mode voltage fields
        vm = jnp.ones_like(node_inj)
        va = jnp.zeros_like(node_inj)
        qflow = jnp.zeros_like(line_flow_mw)
        cost_voltage = jnp.float32(0.0)

        # AC PF path (physics=1, solver_mode in {0,1})
        if params.acpf_setup is not None:
            (ac_flow, ac_inj, ac_safe, ac_nviol, ac_cost_t,
             ac_vm, ac_va, ac_qflow, ac_cost_v) = _run_ac_step(
                params.acpf_setup, unit_power_mw, node_load_mw, case)

            is_ac = params.physics == 1
            line_flow_mw = jnp.where(is_ac, ac_flow, line_flow_mw)
            node_inj = jnp.where(is_ac, ac_inj, node_inj)
            is_safe = jnp.where(is_ac, ac_safe, is_safe)
            n_violations = jnp.where(is_ac, ac_nviol, n_violations)
            cost_thermal = jnp.where(is_ac, ac_cost_t, cost_thermal)
            vm = jnp.where(is_ac, ac_vm, vm)
            va = jnp.where(is_ac, ac_va, va)
            qflow = jnp.where(is_ac, ac_qflow, qflow)
            cost_voltage = jnp.where(is_ac, ac_cost_v, cost_voltage)

        # ACOPF path (solver_mode=2): OPF determines dispatch with AC model
        q_gen = jnp.zeros_like(unit_power_mw)
        acopf_line_viol_mva = jnp.float32(0.0)
        if params.acopf_setup is not None:
            (acopf_up, acopf_flow, acopf_qflow, acopf_inj, acopf_safe, acopf_nviol,
             acopf_cost_t, acopf_vm, acopf_va, acopf_qgen, acopf_cost_v,
             acopf_line_viol_mva) = (
                _run_acopf_step(params.acopf_setup, node_load_mw, case))

            is_acopf = params.solver_mode == 2
            unit_power_mw = jnp.where(is_acopf, acopf_up, unit_power_mw)
            line_flow_mw = jnp.where(is_acopf, acopf_flow, line_flow_mw)
            node_inj = jnp.where(is_acopf, acopf_inj, node_inj)
            is_safe = jnp.where(is_acopf, acopf_safe, is_safe)
            n_violations = jnp.where(is_acopf, acopf_nviol, n_violations)
            cost_thermal = jnp.where(is_acopf, acopf_cost_t, cost_thermal)
            vm = jnp.where(is_acopf, acopf_vm, vm)
            va = jnp.where(is_acopf, acopf_va, va)
            q_gen = jnp.where(is_acopf, acopf_qgen, q_gen)
            cost_voltage = jnp.where(is_acopf, acopf_cost_v, cost_voltage)
            qflow = jnp.where(is_acopf, acopf_qflow, qflow)

        gen_cost = compute_generation_cost(
            unit_power_mw, case.unit_cost_a, case.unit_cost_b, case.unit_cost_c)

        # Slack overload: penalise when actual unit power exceeds p_max
        slack_over = jnp.sum(jnp.maximum(unit_power_mw - case.unit_p_max, 0.0))
        slack_under = jnp.sum(jnp.maximum(case.unit_p_min - unit_power_mw, 0.0))
        cost_power_balance = slack_over + slack_under

        new_total_cost = state.total_cost + gen_cost

        new_time = state.time_step + 1
        done = new_time >= params.max_steps

        new_state = TransGridState(
            time_step=new_time,
            done=done,
            unit_power_mw=unit_power_mw,
            line_flow_mw=line_flow_mw,
            node_injection_mw=node_inj,
            is_safe=is_safe,
            n_violations=n_violations,
            total_cost=new_total_cost,
            vm=vm,
            va=va,
            q_gen=q_gen,
            line_flow_q_mw=qflow,
            resource_states=new_resource_states,
        )

        # Auto-reset: when done, reset to initial state.
        _, k_reset = jax.random.split(key)
        _, reset_state = self.reset(k_reset, params)
        final_state = tu.tree_map(
            lambda new, rst: jnp.where(done, rst, new),
            new_state, reset_state)
        final_state = jax.lax.stop_gradient(final_state)
        final_obs = self._get_obs(final_state, params)

        reward = -params.reward_scale * gen_cost

        cost_thermal_weighted = cost_thermal * params.cost_thermal_weight
        cost_resource = bundle_cost
        cost_sum = cost_thermal_weighted + cost_voltage + cost_power_balance + cost_resource
        costs = stack_costs(
            cost_thermal_weighted,
            cost_voltage,
            cost_power_balance,
            cost_resource,
        )
        info = {
            "gen_cost": gen_cost,
            "cost_thermal_overload": cost_thermal_weighted,
            "cost_voltage_violation": cost_voltage,
            "cost_power_balance": cost_power_balance,
            "cost_resource": cost_resource,
            "cost_load_shedding": jnp.float32(0.0),
            "cost_sum": cost_sum,
            "is_safe": is_safe,
            "n_violations": n_violations,
            "goal_met": is_safe,
            "line_viol_mva": jnp.where(
                params.solver_mode == 2, acopf_line_viol_mva, jnp.float32(0.0)),
        }

        return final_obs, final_state, reward, costs, done, info

    # ====== Spaces & Observation ======

    def _get_obs(self, state: TransGridState, params: TransGridParams) -> chex.Array:
        """Observation vector — grid quantities + bundle observations.

        DC layout: [P/cap, load/total_cap, unit_p/p_max, sin(t), cos(t), <bundle_obs>]
        AC layout: [|S|/cap, vm, load/total_cap, unit_p/p_max, sin(t), cos(t), <bundle_obs>]
        """
        case = params.case

        safe_cap = jnp.maximum(jnp.abs(case.line_cap), 1.0)
        if params.acpf_setup is not None or params.acopf_setup is not None:
            line_s = jnp.sqrt(
                state.line_flow_mw ** 2 + state.line_flow_q_mw ** 2)
            flow_norm = line_s / safe_cap
        else:
            flow_norm = state.line_flow_mw / safe_cap

        t_idx = state.time_step % params.load_profiles.shape[0]
        load_demand = params.load_profiles[t_idx]
        total_p_max = jnp.sum(case.unit_p_max)
        safe_total = jnp.maximum(total_p_max, 1.0)
        load_norm = load_demand / safe_total

        safe_p_max = jnp.maximum(case.unit_p_max, 1.0)
        unit_norm = state.unit_power_mw / safe_p_max

        t_sin, t_cos = time_features(state.time_step, params.steps_per_day)

        if params.acpf_setup is not None or params.acopf_setup is not None:
            obs_parts = [flow_norm, state.vm, load_norm, unit_norm,
                         jnp.stack([t_sin, t_cos])]
        else:
            obs_parts = [flow_norm, load_norm, unit_norm,
                         jnp.stack([t_sin, t_cos])]

        ctx = {"dt_hours": params.delta_t_hours}
        for i, bundle in enumerate(params.resources):
            obs_parts.append(bundle.observe(state.resource_states[i], ctx))

        return jnp.concatenate(obs_parts)

    def observation_space(self, params: TransGridParams) -> Box:
        case = params.case
        obs_dim = case.n_lines + case.n_loads + case.n_units + 2
        if params.acpf_setup is not None or params.acopf_setup is not None:
            obs_dim += case.n_nodes  # vm in observation
        obs_dim += sum(b.obs_dim for b in params.resources)
        low = jnp.full((obs_dim,), -jnp.inf, dtype=jnp.float32)
        high = jnp.full((obs_dim,), jnp.inf, dtype=jnp.float32)
        return Box(low=low, high=high, shape=(obs_dim,), dtype=jnp.float32)

    def action_space(self, params: TransGridParams) -> Box:
        case = params.case
        act_dim = sum(b.action_dim for b in params.resources)
        if params.include_unit_dispatch:
            act_dim += case.n_units
        return Box(
            low=jnp.full((act_dim,), -1.0, dtype=jnp.float32),
            high=jnp.full((act_dim,), 1.0, dtype=jnp.float32),
            shape=(act_dim,),
            dtype=jnp.float32,
        )

    # ====== Status & Diagnostics ======

    @property
    def name(self) -> str:
        return "TransGridEnv"

    def constraint_names(self, params: TransGridParams) -> tuple[str, ...]:
        return (
            "thermal_overload",
            "voltage_violation",
            "power_balance",
            "resource",
        )
