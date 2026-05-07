"""Unit Commitment Environment — TSO SCUC benchmark task.

A security-constrained unit commitment env on top of the DC power-flow
infrastructure: each step the agent both commits units (on / off) and
sets dispatch intents, the env applies ramp / min-up-down / reserve
constraints, runs DC PF or DC-OPF redispatch, and charges operating
cost (variable + startup + no-load) plus CMDP penalties for thermal,
reserve, and min-up-down violations.

State machine (extends ``GridState``, per unit unless noted):
    unit_status         on (1) / off (0).
    time_in_state       steps since the last status change — drives
                        min-up / min-down enforcement.
    last_dispatch       previous step's actual dispatch [MW] — drives
                        ramp constraints.
    startup_cost_accum  episode-accumulated startup cost [$].
    episode_start_idx   profile-pool start index sampled on reset
                        when ``sample_start_on_reset=True`` (diverse
                        temporal coverage, same idea as GenCos MARL).

Action (flattened ``Box(2 · n_units)`` in [−1, 1]):
    first half   commit intent — sign threshold yields on / off,
                  subject to min-up / min-down masking.
    second half  dispatch intent — denormalised to ``[p_min, p_max]``,
                  then clipped to the per-unit ramp window around
                  ``last_dispatch``.
    With ``enable_uc=False`` the commit half is ignored and all units
    are forced ON (pure economic dispatch baseline).

Solver mode:
    0 = direct DC PF on the agent's clipped dispatch.
    1 = DCOPF redispatch on the committed subset (recommended for
        SCUC): ADMM finds a balanced dispatch within physical ramp
        and commitment bounds, ignoring the agent's exact dispatch
        values within those bounds.

Reward:
    reward = −reward_scale · (gen_cost + startup_cost + no_load_cost)

CMDP costs (stacked vector, three channels):
    cost_thermal_overload   ``safety_check`` (MW), weighted by
                            ``cost_thermal_weight``.
    cost_reserve_shortfall  max(0, reserve_required − reserve_available),
                            with ``reserve_required = reserve_margin_frac
                            · total_load``. Returns 0 when
                            ``enable_reserve=False``.
    cost_min_updown         soft penalty per unit for any commit
                            transition that would violate its min-up
                            or min-down.

Conversions from CaseData (handled in ``make_uc_params``):
    Ramp:        ``unit_ramp_up / unit_ramp_down`` are fractions per
                 hour (e.g. 0.7 = 70 %/h); converted to MW / step via
                 ``frac · p_max · delta_t_hours``.
    Min-up/down: ``unit_min_up_time / unit_min_down_time`` already in
                 steps (typical 30-min res: coal 4, nuclear 96, gas 1).
    Initial:     ``|unit_keep_time|`` seeds ``time_in_state``;
                 ``unit_init_state``, ``unit_init_power`` seed the
                 rest.

Optional ``forecast_horizon_steps > 0`` appends that many future
total-load values to the observation, letting the agent anticipate
ramping needs. Default 0 keeps the legacy no-forecast layout.
"""

from __future__ import annotations

from functools import partial
from typing import Tuple, Dict, Any

import numpy as np
import jax
import jax.numpy as jnp
import jax.tree_util as tu
import chex
from flax import struct

from powerzoojax.envs.grid.power_flow import (
    dc_power_flow,
    safety_check,
    compute_generation_cost,
    proportional_dispatch,
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
from powerzoojax.envs.grid.trans import TransGridParams


# ============ State / Params ============

@struct.dataclass
class UCState(GridState):
    """Unit Commitment environment state (extends GridState).

    Additional UC state-machine fields beyond GridState:
        unit_status: Current on/off for each unit (1=on, 0=off).
        time_in_state: Steps in current on/off state (≥ 1 after first step).
        last_dispatch: Previous step actual dispatch [MW] (0 for off units).
        startup_cost_accum: Episode-accumulated startup cost [$].
    """
    unit_status: chex.Array = None       # (n_units,) int32
    time_in_state: chex.Array = None     # (n_units,) int32
    last_dispatch: chex.Array = None     # (n_units,) float32
    startup_cost_accum: chex.Array = None  # float32 scalar
    episode_start_idx: chex.Array = None   # () int32


@struct.dataclass
class UCParams(TransGridParams):
    """Unit Commitment environment parameters (extends TransGridParams).

    UC-specific fields:
        min_up_steps: Minimum consecutive ON steps per unit.
        min_down_steps: Minimum consecutive OFF steps per unit.
        startup_cost: One-time startup cost per unit [$].
        no_load_cost_per_step: Fixed running cost per step per unit [$/step].
        ramp_up_mw: Max ramp-up per step per unit [MW/step].
        ramp_down_mw: Max ramp-down per step per unit [MW/step].
        reserve_margin_frac: Required headroom above total load (e.g. 0.05).
        enable_uc: If False, all units always ON — pure economic dispatch mode.
        enable_reserve: If False, reserve shortfall not charged as cost.
        forecast_horizon_steps: Number of future total-load forecast steps
            appended to the observation. 0 keeps the legacy no-forecast layout.
        init_unit_status: Initial on/off status for reset.
        init_time_in_state: Initial time-in-state for reset [steps].
        init_dispatch: Initial dispatch for reset [MW].
        sample_start_on_reset: If True, reset samples a fresh 48-step episode
            start uniformly from the bound ``load_profiles`` pool.
        fixed_episode_start_idx: Deterministic start index used when sampling
            is disabled (the eval/default path).
    """
    min_up_steps: chex.Array = None        # (n_units,) int32
    min_down_steps: chex.Array = None      # (n_units,) int32
    startup_cost: chex.Array = None        # (n_units,) float32
    no_load_cost_per_step: chex.Array = None  # (n_units,) float32
    ramp_up_mw: chex.Array = None          # (n_units,) float32
    ramp_down_mw: chex.Array = None        # (n_units,) float32
    reserve_margin_frac: float = struct.field(pytree_node=False, default=0.05)
    enable_uc: bool = struct.field(pytree_node=False, default=True)
    enable_reserve: bool = struct.field(pytree_node=False, default=True)
    forecast_horizon_steps: int = struct.field(pytree_node=False, default=0)
    dispatch_preference_weight: float = struct.field(pytree_node=False, default=0.01)
    dcopf_max_iter: int = struct.field(pytree_node=False, default=100)
    dcopf_tol: float = struct.field(pytree_node=False, default=1e-3)
    init_unit_status: chex.Array = None    # (n_units,) int32
    init_time_in_state: chex.Array = None  # (n_units,) int32
    init_dispatch: chex.Array = None       # (n_units,) float32
    sample_start_on_reset: bool = struct.field(pytree_node=False, default=False)
    fixed_episode_start_idx: int = struct.field(pytree_node=False, default=0)


# ============ Factory ============

def make_uc_params(
    case: CaseData,
    load_profiles: chex.Array = None,
    max_steps: int = 48,
    delta_t_hours: float = 0.5,
    steps_per_day: int = 48,
    reserve_margin_frac: float = 0.05,
    enable_uc: bool = True,
    enable_reserve: bool = True,
    forecast_horizon_steps: int = 0,
    reward_scale: float = 1e-4,
    cost_thermal_weight: float = 1.0,
    solver_mode: int = 1,
    dcopf_max_iter: int = 100,
    dcopf_tol: float = 1e-3,
    sample_start_on_reset: bool = False,
    fixed_episode_start_idx: int = 0,
) -> UCParams:
    """Create UCParams from a CaseData with UC fields.

    Ramp conversion:
        unit_ramp_up / unit_ramp_down in CaseData are fractions per hour.
        ramp_mw_per_step = ramp_frac * p_max * delta_t_hours

    min_up_time / min_down_time in CaseData are in steps (30-min res).

    keep_time in CaseData is in steps; abs value = initial time-in-state.

    Args:
        case: CaseData with UC fields populated (unit_ramp_up, unit_min_up_time,
            unit_startup_cost, unit_no_load_cost, unit_init_state, unit_keep_time).
        load_profiles: (T, n_loads) MW load profiles. None → flat midpoint.
        max_steps: Episode length.
        delta_t_hours: Step duration in hours (0.5 = 30-min).
        steps_per_day: Steps per day (48 for 30-min resolution).
        reserve_margin_frac: Required reserve headroom fraction.
        enable_uc: If False, all units always ON (ED mode).
        enable_reserve: If False, no reserve cost.
        forecast_horizon_steps: Number of future total-load steps appended to
            the observation.
        reward_scale: Scalar for reward = -reward_scale * operating_cost.
        cost_thermal_weight: Weight for thermal overload in CMDP cost.
        solver_mode: 0 = direct dispatch PF, 1 = DC-OPF (recommended for SCUC).
        dcopf_max_iter: Max ADMM iterations for the DC-OPF redispatch.
        dcopf_tol: ADMM convergence tolerance for the DC-OPF redispatch.
        sample_start_on_reset: Whether reset samples a new episode start
            uniformly from the bound load-profile pool.
        fixed_episode_start_idx: Deterministic episode start used when
            ``sample_start_on_reset`` is False.
    """
    if int(forecast_horizon_steps) < 0:
        raise ValueError(
            "forecast_horizon_steps must be non-negative, "
            f"got {forecast_horizon_steps!r}"
        )
    n_units = case.n_units

    if load_profiles is None:
        mid_load = (case.load_d_max + case.load_d_min) / 2.0
        load_profiles = jnp.tile(mid_load[None, :], (max_steps, 1))

    # --- Ramp: fraction/hour → MW/step ---
    if case.unit_ramp_up is not None:
        ramp_up_mw = jnp.array(
            np.asarray(case.unit_ramp_up) * np.asarray(case.unit_p_max) * delta_t_hours,
            dtype=jnp.float32,
        )
        ramp_down_mw = jnp.array(
            np.asarray(case.unit_ramp_down) * np.asarray(case.unit_p_max) * delta_t_hours,
            dtype=jnp.float32,
        )
    else:
        # No ramp limit: allow full p_max change per step
        ramp_up_mw = jnp.array(case.unit_p_max, dtype=jnp.float32)
        ramp_down_mw = jnp.array(case.unit_p_max, dtype=jnp.float32)

    # --- Min-up / min-down steps (direct from CaseData, already in steps) ---
    if case.unit_min_up_time is not None:
        min_up_steps = jnp.array(case.unit_min_up_time, dtype=jnp.int32)
    else:
        min_up_steps = jnp.ones(n_units, dtype=jnp.int32)

    if case.unit_min_down_time is not None:
        min_down_steps = jnp.array(case.unit_min_down_time, dtype=jnp.int32)
    else:
        min_down_steps = jnp.ones(n_units, dtype=jnp.int32)

    # --- Costs ---
    if case.unit_startup_cost is not None:
        startup_cost = jnp.array(case.unit_startup_cost, dtype=jnp.float32)
    else:
        startup_cost = jnp.zeros(n_units, dtype=jnp.float32)

    if case.unit_no_load_cost is not None:
        no_load_cost_per_step = jnp.array(
            np.asarray(case.unit_no_load_cost) * delta_t_hours,
            dtype=jnp.float32,
        )
    else:
        no_load_cost_per_step = jnp.zeros(n_units, dtype=jnp.float32)

    # ED mode: zero out UC-specific costs so reward = gen_cost only
    if not enable_uc:
        startup_cost = jnp.zeros(n_units, dtype=jnp.float32)
        no_load_cost_per_step = jnp.zeros(n_units, dtype=jnp.float32)
        min_up_steps = jnp.ones(n_units, dtype=jnp.int32)
        min_down_steps = jnp.ones(n_units, dtype=jnp.int32)

    # --- Initial state ---
    if case.unit_init_state is not None:
        init_unit_status = jnp.array(case.unit_init_state, dtype=jnp.int32)
    else:
        init_unit_status = jnp.ones(n_units, dtype=jnp.int32)  # all on

    if case.unit_keep_time is not None:
        raw_keep = np.abs(np.asarray(case.unit_keep_time)).clip(0, 1000)
        init_time_in_state = jnp.array(raw_keep, dtype=jnp.int32)
    else:
        # Default: well past any min-up/down requirement
        init_time_in_state = jnp.full((n_units,), 200, dtype=jnp.int32)

    if case.unit_init_power is not None:
        init_dispatch = jnp.array(case.unit_init_power, dtype=jnp.float32)
    else:
        # ON units start at p_min, OFF units at 0
        init_dispatch = (
            jnp.array(case.unit_p_min, dtype=jnp.float32)
            * init_unit_status.astype(jnp.float32)
        )

    if not enable_uc:
        # ED: economic dispatch always uses full commitment at reset (ignore case UC init).
        init_unit_status = jnp.ones(n_units, dtype=jnp.int32)
        init_time_in_state = jnp.full((n_units,), 200, dtype=jnp.int32)
        init_dispatch = jnp.array(case.unit_p_min, dtype=jnp.float32)

    # --- DCOPF setup ---
    dcopf_setup = None
    if solver_mode == 1:
        dcopf_setup = prepare_dcopf(case)

    return UCParams(
        # GridParams
        load_profiles=load_profiles,
        max_steps=max_steps,
        delta_t_hours=delta_t_hours,
        steps_per_day=steps_per_day,
        cost_thermal_weight=cost_thermal_weight,
        physics=0,
        solver_mode=solver_mode,
        # TransGridParams
        case=case,
        reward_scale=reward_scale,
        acpf_setup=None,
        dcopf_setup=dcopf_setup,
        acopf_setup=None,
        include_unit_dispatch=True,
        resources=(),
        # UCParams
        min_up_steps=min_up_steps,
        min_down_steps=min_down_steps,
        startup_cost=startup_cost,
        no_load_cost_per_step=no_load_cost_per_step,
        ramp_up_mw=ramp_up_mw,
        ramp_down_mw=ramp_down_mw,
        reserve_margin_frac=reserve_margin_frac,
        enable_uc=enable_uc,
        enable_reserve=enable_reserve,
        forecast_horizon_steps=int(forecast_horizon_steps),
        dcopf_max_iter=int(dcopf_max_iter),
        dcopf_tol=float(dcopf_tol),
        init_unit_status=init_unit_status,
        init_time_in_state=init_time_in_state,
        init_dispatch=init_dispatch,
        sample_start_on_reset=bool(sample_start_on_reset),
        fixed_episode_start_idx=int(fixed_episode_start_idx),
    )


# ============ Environment ============

class UnitCommitmentEnv(Environment):
    """Unit Commitment environment — DC power flow with UC state machine.

    Agent action: Box(2 * n_units) in [-1, 1].
        First n_units  → commitment intent; > 0.0 → commit (if enable_uc).
        Last  n_units  → dispatch intent.
            Direct-PF mode: maps directly to [p_min, p_max] then clipped to ramp.
            DC-OPF mode   : used as a *preference bias* — OPF cost is shifted by
                            −dispatch_preference_weight × target_mw so the OPF
                            solution moves toward the agent's preferred dispatch
                            while still respecting physical constraints.

    Env enforces:
        - Min-up / min-down constraints via commitment masking
        - Ramp constraints via modified p_min/p_max in DCOPF
        - Startup cost accounting (off→on transition)
        - No-load cost (per ON unit per step)
        - Reserve margin check

    Reward: -reward_scale * (gen_cost + startup_cost + no_load_cost)
    Constraint costs: thermal_overload, reserve_shortfall, min_updown
    """

    # ====== RL Interface Methods ======

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: UCParams,
    ) -> Tuple[chex.Array, UCState]:
        case = params.case
        pool_steps = params.load_profiles.shape[0]
        max_start = max(pool_steps - int(params.max_steps), 0)
        if params.sample_start_on_reset and max_start > 0:
            episode_start_idx = jax.random.randint(
                key,
                shape=(),
                minval=0,
                maxval=max_start + 1,
                dtype=jnp.int32,
            )
        else:
            episode_start_idx = jnp.int32(params.fixed_episode_start_idx)
        load_demand = params.load_profiles[episode_start_idx]
        node_load_mw = case.nodes_loads_map @ load_demand

        commit = params.init_unit_status.astype(jnp.float32)

        if params.dcopf_setup is not None:
            result = dc_opf(
                params.dcopf_setup,
                node_load_mw,
                commitment=commit,
                max_iter=int(params.dcopf_max_iter),
                tol=float(params.dcopf_tol),
            )
            unit_power_mw = result.unit_power
            line_flow_mw = result.line_flow
            node_inj = result.node_injection
        else:
            # Proportional dispatch among committed units
            committed_p_max = case.unit_p_max * commit
            committed_p_min = case.unit_p_min * commit
            unit_power_mw = proportional_dispatch(
                jnp.sum(node_load_mw), committed_p_min, committed_p_max)
            line_flow_mw, node_inj, unit_power_mw = dc_power_flow(
                case, unit_power_mw, node_load_mw)

        is_thermal_safe, n_violations, _ = safety_check(
            line_flow_mw, case.line_cap, case.line_floor)
        total_committed_cap = jnp.sum(commit * case.unit_p_max)
        total_load = jnp.sum(node_load_mw)
        reserve_shortfall = jnp.maximum(
            jnp.float32(0.0),
            total_load * (jnp.float32(1.0) + params.reserve_margin_frac)
            - total_committed_cap,
        )
        reserve_violation = jnp.logical_and(
            jnp.bool_(params.enable_reserve),
            reserve_shortfall > jnp.float32(1e-6),
        )
        is_safe = jnp.logical_and(is_thermal_safe, jnp.logical_not(reserve_violation))
        n_violations = n_violations + reserve_violation.astype(jnp.int32)

        state = UCState(
            time_step=jnp.int32(0),
            done=jnp.bool_(False),
            unit_power_mw=unit_power_mw,
            line_flow_mw=line_flow_mw,
            node_injection_mw=node_inj,
            is_safe=is_safe,
            n_violations=n_violations,
            total_cost=jnp.float32(0.0),
            vm=jnp.ones_like(node_inj),
            va=jnp.zeros_like(node_inj),
            q_gen=jnp.zeros_like(unit_power_mw),
            line_flow_q_mw=jnp.zeros_like(line_flow_mw),
            resource_states=(),
            unit_status=params.init_unit_status,
            time_in_state=params.init_time_in_state,
            last_dispatch=unit_power_mw,
            startup_cost_accum=jnp.float32(0.0),
            episode_start_idx=episode_start_idx,
        )
        obs = self._get_obs(state, params)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: UCState,
        action: chex.Array,
        params: UCParams,
    ) -> Tuple[chex.Array, UCState, chex.Array, chex.Array, chex.Array, Dict[str, Any]]:
        """Run one UC environment step.

        Action layout: [commit_intent(n_units) | dispatch_intent(n_units)]
            commit_intent ∈ [-1, 1]: > 0 → want to commit (if enable_uc).
            dispatch_intent ∈ [-1, 1]: maps to [eff_p_min, eff_p_max].
                Direct-PF mode: used as dispatch setpoint.
                DC-OPF  mode  : biases OPF cost toward preferred dispatch.
        """
        case = params.case
        n_units = state.unit_status.shape[0]  # static Python int from array shape

        commit_signal = action[:n_units]
        dispatch_signal = action[n_units:]

        # --- 1. Determine target commitment ---
        if params.enable_uc:
            raw_commit = jnp.where(
                commit_signal > jnp.float32(0.0),
                jnp.int32(1), jnp.int32(0),
            )
        else:
            raw_commit = jnp.ones_like(state.unit_status)

        # --- 2. Min-up / min-down masking (force feasibility) ---
        if params.enable_uc:
            must_stay_on = jnp.logical_and(
                state.unit_status == 1,
                state.time_in_state < params.min_up_steps,
            )
            must_stay_off = jnp.logical_and(
                state.unit_status == 0,
                state.time_in_state < params.min_down_steps,
            )
            actual_commit = jnp.where(must_stay_on, jnp.int32(1), raw_commit)
            actual_commit = jnp.where(must_stay_off, jnp.int32(0), actual_commit)
        else:
            actual_commit = raw_commit

        commit_float = actual_commit.astype(jnp.float32)

        # --- 3. Startup cost (off→on transition) ---
        switched_on = jnp.logical_and(
            actual_commit == 1, state.unit_status == 0)
        startup_cost_step = jnp.sum(
            switched_on.astype(jnp.float32) * params.startup_cost)

        # --- 4. No-load cost (per ON unit per step) ---
        no_load_cost_step = jnp.sum(commit_float * params.no_load_cost_per_step)

        # --- 5. Ramp-adjusted dispatch bounds ---
        # Allow newly started units (last_dispatch=0) to ramp up to ramp_up_mw
        ramp_p_min = jnp.maximum(
            case.unit_p_min,
            state.last_dispatch - params.ramp_down_mw,
        )
        ramp_p_max = jnp.minimum(
            case.unit_p_max,
            state.last_dispatch + params.ramp_up_mw,
        )
        # Apply commitment mask: OFF units get [0, 0]
        eff_p_min = ramp_p_min * commit_float
        eff_p_max = ramp_p_max * commit_float

        # Load for this time step
        t_idx = state.episode_start_idx + state.time_step
        load_demand = params.load_profiles[t_idx]
        node_load_mw = case.nodes_loads_map @ load_demand

        # --- 6. Power dispatch ---
        if params.dcopf_setup is not None:
            # DC-OPF with ramp-adjusted bounds
            setup_ramp = params.dcopf_setup.replace(
                p_min=eff_p_min, p_max=eff_p_max)
            # Apply dispatch preference bias: steer OPF solution toward agent's
            # preferred dispatch without overriding physical constraints.
            # target ∈ [eff_p_min, eff_p_max]; OFF units: both bounds = 0 → bias = 0
            target_dispatch_mw = denormalize_action(
                dispatch_signal, eff_p_min, eff_p_max)
            alpha = jnp.float32(params.dispatch_preference_weight)
            mc_c_biased = setup_ramp.mc_c - alpha * target_dispatch_mw
            setup_biased = setup_ramp.replace(mc_c=mc_c_biased)
            result = dc_opf(
                setup_biased,
                node_load_mw,
                max_iter=int(params.dcopf_max_iter),
                tol=float(params.dcopf_tol),
            )
            unit_power_mw = result.unit_power
            line_flow_mw = result.line_flow
            node_inj = result.node_injection
            opf_converged = result.converged
            opf_iterations = result.iterations
            opf_box_residual_mw = result.max_box_residual_mw
            opf_line_residual_mw = result.max_line_residual_mw
            opf_balance_residual_mw = result.balance_residual_mw
        else:
            # Direct dispatch: denormalize agent's dispatch_signal, apply ramp
            target_dispatch = denormalize_action(
                dispatch_signal, case.unit_p_min, case.unit_p_max)
            clipped_dispatch = jnp.clip(target_dispatch, eff_p_min, eff_p_max)
            line_flow_mw, node_inj, unit_power_mw = dc_power_flow(
                case, clipped_dispatch, node_load_mw)
            # Enforce commitment: zero out OFF units' attributed power
            # (DC PF slack adjustment may assign power to the slack bus unit
            #  even when it is OFF; this override ensures OFF → 0 MW reported)
            unit_power_mw = unit_power_mw * commit_float
            opf_converged = jnp.bool_(True)
            opf_iterations = jnp.int32(0)
            opf_box_residual_mw = jnp.float32(0.0)
            opf_line_residual_mw = jnp.float32(0.0)
            opf_balance_residual_mw = jnp.float32(0.0)

        # --- 7. Safety check ---
        is_thermal_safe, n_violations, cost_thermal = safety_check(
            line_flow_mw, case.line_cap, case.line_floor)

        # --- 8. Reserve margin check ---
        total_committed_cap = jnp.sum(commit_float * case.unit_p_max)
        total_load = jnp.sum(node_load_mw)
        required_reserve = params.reserve_margin_frac * total_load
        # Shortfall: how much headroom is missing
        reserve_shortfall = jnp.maximum(
            jnp.float32(0.0),
            total_load * (jnp.float32(1.0) + params.reserve_margin_frac)
            - total_committed_cap,
        )
        reserve_violation = jnp.logical_and(
            jnp.bool_(params.enable_reserve),
            reserve_shortfall > jnp.float32(1e-6),
        )
        is_safe = jnp.logical_and(is_thermal_safe, jnp.logical_not(reserve_violation))
        n_violations = n_violations + reserve_violation.astype(jnp.int32)

        if params.enable_reserve:
            cost_reserve = reserve_shortfall
        else:
            cost_reserve = jnp.float32(0.0)

        # --- 9. Generation cost ---
        gen_cost = compute_generation_cost(
            unit_power_mw,
            case.unit_cost_a, case.unit_cost_b, case.unit_cost_c,
        )

        # --- 10. Update UC state ---
        status_changed = (actual_commit != state.unit_status).astype(jnp.int32)
        new_time_in_state = jnp.where(
            status_changed, jnp.int32(1), state.time_in_state + jnp.int32(1))

        commitment_switches = jnp.sum(status_changed).astype(jnp.float32)
        new_startup_cost_accum = state.startup_cost_accum + startup_cost_step
        new_total_cost = state.total_cost + gen_cost
        new_time = state.time_step + 1
        done = new_time >= params.max_steps

        new_state = UCState(
            time_step=new_time,
            done=done,
            unit_power_mw=unit_power_mw,
            line_flow_mw=line_flow_mw,
            node_injection_mw=node_inj,
            is_safe=is_safe,
            n_violations=n_violations,
            total_cost=new_total_cost,
            vm=jnp.ones_like(node_inj),
            va=jnp.zeros_like(node_inj),
            q_gen=jnp.zeros_like(unit_power_mw),
            line_flow_q_mw=jnp.zeros_like(line_flow_mw),
            resource_states=(),
            unit_status=actual_commit,
            time_in_state=new_time_in_state,
            last_dispatch=unit_power_mw,
            startup_cost_accum=new_startup_cost_accum,
            episode_start_idx=state.episode_start_idx,
        )

        # --- 11. Auto-reset ---
        _, k_reset = jax.random.split(key)
        _, reset_state = self.reset(k_reset, params)
        final_state = tu.tree_map(
            lambda nw, rs: jnp.where(done, rs, nw),
            new_state, reset_state,
        )
        final_state = jax.lax.stop_gradient(final_state)
        final_obs = self._get_obs(final_state, params)

        # --- 12. Reward / cost separation ---
        total_step_operating_cost = gen_cost + startup_cost_step + no_load_cost_step
        reward = -params.reward_scale * total_step_operating_cost

        cost_thermal_w = cost_thermal * params.cost_thermal_weight
        cost_min_updown = jnp.float32(0.0)  # masking guarantees feasibility
        cost_sum = cost_thermal_w + cost_reserve + cost_min_updown

        costs = stack_costs(cost_thermal_w, cost_reserve, cost_min_updown)
        info = {
            "gen_cost": gen_cost,
            "startup_cost": startup_cost_step,
            "no_load_cost": no_load_cost_step,
            "reserve_shortfall": reserve_shortfall,
            "cost_thermal_overload": cost_thermal_w,
            "cost_reserve_shortfall": cost_reserve,
            "cost_min_updown": cost_min_updown,
            "cost_sum": cost_sum,
            "commitment_switches": commitment_switches,
            "is_safe": is_safe,
            "n_violations": n_violations,
            "goal_met": is_safe,
            "opf_converged": opf_converged,
            "opf_iterations": opf_iterations,
            "opf_box_residual_mw": opf_box_residual_mw,
            "opf_line_residual_mw": opf_line_residual_mw,
            "opf_balance_residual_mw": opf_balance_residual_mw,
        }

        return final_obs, final_state, reward, costs, done, info

    # ====== Spaces & Observation ======

    def _get_obs(self, state: UCState, params: UCParams) -> chex.Array:
        """Observation vector for UC environment.

        Layout: [unit_status | time_in_state_norm | last_dispatch_norm |
                 unit_cost_b_norm | line_flow_norm | load_norm | reserve_norm |
                 sin(t) | cos(t) | future_total_load_norm[t+1:t+H]]

        Dimensions: 4*n_units + n_lines + 4 + forecast_horizon_steps
        """
        case = params.case

        # Unit features
        unit_status_f = state.unit_status.astype(jnp.float32)
        # Normalise time_in_state by a large constant (max plausible horizon)
        time_norm = state.time_in_state.astype(jnp.float32) / jnp.float32(200.0)
        safe_p_max = jnp.maximum(case.unit_p_max, jnp.float32(1.0))
        dispatch_norm = state.last_dispatch / safe_p_max
        # Marginal cost b normalised by max
        cost_b_abs = jnp.abs(case.unit_cost_b)
        safe_cb = jnp.maximum(jnp.max(cost_b_abs), jnp.float32(1.0))
        cost_b_norm = case.unit_cost_b / safe_cb

        # Line flow normalised by capacity
        safe_cap = jnp.maximum(jnp.abs(case.line_cap), jnp.float32(1.0))
        flow_norm = state.line_flow_mw / safe_cap

        # Scalar features
        t_idx = state.episode_start_idx + state.time_step
        load_demand = params.load_profiles[t_idx]
        node_load_mw = case.nodes_loads_map @ load_demand
        total_load = jnp.sum(node_load_mw)
        total_p_max = jnp.maximum(jnp.sum(case.unit_p_max), jnp.float32(1.0))
        load_norm = total_load / total_p_max

        # Reserve margin (committed headroom / load)
        commit_float = state.unit_status.astype(jnp.float32)
        total_committed_cap = jnp.sum(commit_float * case.unit_p_max)
        safe_load = jnp.maximum(total_load, jnp.float32(1.0))
        reserve_ratio = (total_committed_cap - total_load) / safe_load

        t_sin, t_cos = time_features(state.time_step, params.steps_per_day)
        pieces = [
            unit_status_f,
            time_norm,
            dispatch_norm,
            cost_b_norm,
            flow_norm,
            jnp.stack([load_norm, reserve_ratio, t_sin, t_cos]),
        ]
        forecast_h = int(params.forecast_horizon_steps)
        if forecast_h > 0:
            total_load_profile = jnp.sum(params.load_profiles, axis=1)
            future_offsets = jnp.minimum(
                state.time_step + jnp.arange(1, forecast_h + 1, dtype=jnp.int32),
                jnp.int32(params.max_steps - 1),
            )
            future_idx = state.episode_start_idx + future_offsets
            pieces.append(total_load_profile[future_idx] / total_p_max)
        return jnp.concatenate(pieces)

    def observation_space(self, params: UCParams) -> Box:
        case = params.case
        obs_dim = 4 * case.n_units + case.n_lines + 4 + int(params.forecast_horizon_steps)
        return Box(
            low=jnp.full((obs_dim,), -jnp.inf, dtype=jnp.float32),
            high=jnp.full((obs_dim,), jnp.inf, dtype=jnp.float32),
            shape=(obs_dim,),
            dtype=jnp.float32,
        )

    def action_space(self, params: UCParams) -> Box:
        act_dim = 2 * params.case.n_units
        return Box(
            low=jnp.full((act_dim,), -1.0, dtype=jnp.float32),
            high=jnp.full((act_dim,), 1.0, dtype=jnp.float32),
            shape=(act_dim,),
            dtype=jnp.float32,
        )

    @property
    def name(self) -> str:
        return "UnitCommitmentEnv"

    def constraint_names(self, params: UCParams) -> tuple[str, ...]:
        return (
            "thermal_overload",
            "reserve_shortfall",
            "min_updown",
        )
