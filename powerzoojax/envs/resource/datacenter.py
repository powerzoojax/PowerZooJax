"""AI Data Center Resource — pure-JAX implementation.

A grid-attached AI data center: the agent splits free GPUs between a
training and a finetuning queue and chooses a cooling setpoint; the env
runs scheduled compute, integrates first-order zone thermals, and
returns the total electrical draw to the grid. The interesting coupling
is thermal-electrical:

    more compute    → more IT heat → hotter zone → more cooling power
    cooler setpoint → more cooling power but lower overtemp risk

so the agent trades throughput (meeting task SLAs) against electricity
cost and thermal safety, against a stochastic backdrop of task arrivals,
diurnal inference load, and outdoor temperature.

Sign convention (consistent with all ResourceEnv):
    current_p_mw < 0 ⇒ absorbing from grid (this resource is only a load).

Three physical layers:
  1. IT      — p_it = idle_GPUs·idle_w
                    + scheduled_GPUs·(idle_w + (active_w − idle_w)·η)
                    + infer_GPUs·active_w·0.5  + p_base_mw
               (the 0.5 factor models 50% average inference utilisation.)
  2. Cooling — q_cool = ua·max(t_zone − t_setpoint, 0)             [kW heat]
               p_cool = q_cool / COP                                [MW elec]
               COP    = cop_ref·clip(1 − cop_decay·(t_out − t_ref)⁺, 0.4, 1.2)
  3. Thermal — c_thermal·dT/dt = p_it − q_cool + h_wall·(t_out − t_zone)

Task lifecycle (one buffer of _MAX_TASKS slots; status 0/1/2 = inactive
/ waiting / running):
    Arrive (Poisson per type)            : 0 → 1
    Force-schedule slack-to-start ≤ 0    : 1 → 2  (overrides agent split)
    RL-schedule within r_train / r_ft × free_GPUs, EDF-greedy : 1 → 2
    Complete (remaining hits 0)          : 2 → 0
    Expire (now ≥ deadline while waiting): 1 → 0  (SLA violation)

Control problem (CMDP):
    Action      : [r_train, r_ft, cool_norm] ∈ [0, 1]³
                  r_train  : fraction of free GPUs → training queue
                  r_ft     : fraction of *remaining* free GPUs → finetune
                  cool_norm: setpoint = t_set_min + cool_norm·(t_set_max − t_set_min)
    Reward      : 0. Define the task externally (energy cost, revenue, …).
    Costs       : (sla, overtemp), stacked vector.
                  cost_sla      = n_expired_this_step / n_gpus  (PowerZoo convention)
                  cost_overtemp = (t_zone − t_critical)⁺ / (T_MAX − t_critical)
    Observation : 11-D
                  [gpu_util, infer_util, q_train_fill, q_ft_fill, queue_urgency,
                   cool_ratio, zone_norm, outdoor_norm, setpoint_norm,
                   sin t, cos t]
    Termination : t ≥ max_steps (auto-resets).

The env force-schedules any waiting task with slack-to-start ≤ 0 *before*
honouring r_train / r_ft, so a r_train = r_ft = 0 policy still draws
compute power whenever urgent tasks remain. This caps worst-case SLA
damage but limits the agent's ability to fully shed load to save energy.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as tu
from flax import struct

from powerzoojax.envs.base import Environment, stack_costs
from powerzoojax.envs.spaces import Box
from powerzoojax.envs.resource.base import ResourceState, ResourceParams, time_features

# Task buffer capacity (waiting + running combined, compile-time constant)
_MAX_TASKS: int = 64

# Max new tasks per step per type (Poisson lambda << this value for standard configs)
_MAX_ARRIVALS: int = 5

# Task status codes (int32)
_INACTIVE: int = 0
_WAITING: int = 1
_RUNNING: int = 2

# Task type codes (int32)
_TRAIN: int = 0
_FINETUNE: int = 1

# Zone temperature physical bounds [°C] — shared by clip and normalisation
_T_ZONE_MIN: float = 15.0
_T_ZONE_MAX: float = 45.0


# State / Params
@struct.dataclass
class DataCenterState(ResourceState):
    """DataCenter state extending ResourceState.

    Attributes:
        t_zone: Zone temperature [°C].
        t_setpoint: Current cooling setpoint [°C].
        t_outdoor: Outdoor temperature [°C].
        p_it_mw: IT (compute) power consumption [MW].
        p_cool_mw: Cooling power consumption [MW].
        gpus_infer: GPUs occupied by inference (exogenous diurnal load).
        gpus_active: Total active GPUs (inference + scheduled tasks).
        task_gpus: GPU count for each task slot [_MAX_TASKS].
        task_deadline: Deadline (absolute step) for each slot [_MAX_TASKS].
        task_duration: Total required duration (steps) for each slot [_MAX_TASKS].
        task_remaining: Steps remaining (only meaningful when running) [_MAX_TASKS].
        task_type: Task type — 0=training, 1=finetuning [_MAX_TASKS].
        task_eta: GPU utilisation efficiency for each slot [_MAX_TASKS].
        task_status: Slot status — 0=inactive, 1=waiting, 2=running [_MAX_TASKS].
        sla_violations: Cumulative count of expired tasks (SLA violations).
        done: Episode termination flag.
    """

    # Thermal
    t_zone: chex.Array         # float32 scalar [°C]
    t_setpoint: chex.Array     # float32 scalar [°C]
    t_outdoor: chex.Array      # float32 scalar [°C]

    # Power breakdown (stored for obs / diagnostics)
    p_it_mw: chex.Array        # float32 scalar [MW]
    p_cool_mw: chex.Array      # float32 scalar [MW]

    # GPU summary
    gpus_infer: chex.Array     # int32 scalar
    gpus_active: chex.Array    # int32 scalar

    # Task buffer — fixed shape [_MAX_TASKS]
    task_gpus: chex.Array      # int32
    task_deadline: chex.Array  # int32
    task_duration: chex.Array  # int32
    task_remaining: chex.Array # int32
    task_type: chex.Array      # int32  (0=train, 1=ft)
    task_eta: chex.Array       # float32
    task_status: chex.Array    # int32  (0=inactive, 1=wait, 2=run)

    # Stats
    sla_violations: chex.Array # int32 scalar
    done: chex.Array           # bool scalar


@struct.dataclass
class DataCenterParams(ResourceParams):
    """DataCenter parameters.

    Attributes:
        n_gpus: Total GPU count in the data center.
        gpu_idle_w: Per-GPU idle power [W] (H100 measured ~55 W).
        gpu_active_w: Per-GPU active power [W] (H100 training ~580 W).
        p_base_mw: Non-GPU baseline IT power [MW] (networking, storage).
        infer_gpu_peak: Peak GPU count consumed by inference (diurnal profile).
        cop_ref: Reference COP at t_ref outdoor temperature.
        cop_decay: COP fractional decay per °C above t_ref.
        t_ref: Reference outdoor temperature for COP [°C].
        c_thermal: Thermal capacitance [kWh/°C].
        ua_cooling: Cooling heat-transfer coefficient [kW/°C].
        h_wall: Building envelope heat-transfer coefficient [kW/°C].
        t_set_min: Minimum cooling setpoint [°C].
        t_set_max: Maximum cooling setpoint [°C] (ASHRAE upper bound).
        t_initial: Initial zone temperature at reset [°C].
        t_critical: Over-temperature safety threshold [°C].
        p_aux_frac: Auxiliary power as fraction of IT power.
        train_arrival_interval: Mean steps between training task arrivals (Poisson rate = 1/interval).
        train_gpu_{lo,hi}: Uniform GPU count range for training tasks.
        train_dur_{lo,hi}: Uniform duration range [steps] for training tasks.
        train_deadline_slack: deadline = arrive + duration × slack (training).
        train_gpu_eta: GPU utilisation efficiency for training.
        ft_*: Same parameters for finetuning tasks.

    All fields are pytree_node=False — static compile-time constants under JIT.
    """

    # GPU & IT
    n_gpus: int = struct.field(pytree_node=False, default=1000)
    gpu_idle_w: float = struct.field(pytree_node=False, default=55.0)
    gpu_active_w: float = struct.field(pytree_node=False, default=580.0)
    p_base_mw: float = struct.field(pytree_node=False, default=0.5)
    infer_gpu_peak: int = struct.field(pytree_node=False, default=400)

    # COP / cooling
    cop_ref: float = struct.field(pytree_node=False, default=5.0)
    cop_decay: float = struct.field(pytree_node=False, default=0.04)
    t_ref: float = struct.field(pytree_node=False, default=20.0)

    # Thermal zone
    c_thermal: float = struct.field(pytree_node=False, default=500.0)
    ua_cooling: float = struct.field(pytree_node=False, default=200.0)
    h_wall: float = struct.field(pytree_node=False, default=5.0)
    t_set_min: float = struct.field(pytree_node=False, default=18.0)
    t_set_max: float = struct.field(pytree_node=False, default=27.0)
    t_initial: float = struct.field(pytree_node=False, default=22.0)
    t_critical: float = struct.field(pytree_node=False, default=35.0)
    p_aux_frac: float = struct.field(pytree_node=False, default=0.05)

    # Training task generation
    train_arrival_interval: int = struct.field(pytree_node=False, default=8)
    train_gpu_lo: int = struct.field(pytree_node=False, default=50)
    train_gpu_hi: int = struct.field(pytree_node=False, default=200)
    train_dur_lo: int = struct.field(pytree_node=False, default=10)
    train_dur_hi: int = struct.field(pytree_node=False, default=50)
    train_deadline_slack: float = struct.field(pytree_node=False, default=2.0)
    train_gpu_eta: float = struct.field(pytree_node=False, default=0.90)

    # Finetuning task generation
    ft_arrival_interval: int = struct.field(pytree_node=False, default=4)
    ft_gpu_lo: int = struct.field(pytree_node=False, default=10)
    ft_gpu_hi: int = struct.field(pytree_node=False, default=50)
    ft_dur_lo: int = struct.field(pytree_node=False, default=5)
    ft_dur_hi: int = struct.field(pytree_node=False, default=20)
    ft_deadline_slack: float = struct.field(pytree_node=False, default=3.0)
    ft_gpu_eta: float = struct.field(pytree_node=False, default=0.75)


# Pure Helper Functions
def _diurnal_factor(time_step: chex.Array, steps_per_day: int) -> chex.Array:
    """Inference load factor: sinusoidal diurnal profile, peak at ~14h, trough at ~4h."""
    hour = (time_step % steps_per_day).astype(jnp.float32) / float(steps_per_day) * 24.0
    return jnp.clip(
        0.5 + 0.5 * jnp.sin(2.0 * jnp.pi * (hour - 8.0) / 24.0),
        0.1, 1.0,
    )


def _outdoor_temp(time_step: chex.Array, steps_per_day: int) -> chex.Array:
    """Synthetic outdoor temperature [°C]: mean 20, amplitude 8, peak at ~14h."""
    hour = (time_step % steps_per_day).astype(jnp.float32) / float(steps_per_day) * 24.0
    return jnp.float32(20.0) + jnp.float32(8.0) * jnp.sin(2.0 * jnp.pi * (hour - 8.0) / 24.0)


def _insert_tasks(
    task_status: chex.Array,
    task_gpus: chex.Array,
    task_deadline: chex.Array,
    task_duration: chex.Array,
    task_remaining: chex.Array,
    task_type: chex.Array,
    task_eta: chex.Array,
    n_arrive: chex.Array,
    gpus_samples: chex.Array,
    dur_samples: chex.Array,
    deadline_samples: chex.Array,
    type_id: int,
    eta_val: float,
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
    """Insert up to n_arrive new tasks into inactive buffer slots.

    Uses lax.scan over _MAX_ARRIVALS iterations; each iteration conditionally
    inserts one task if (i < n_arrive) and a free slot exists.  JIT/vmap safe.
    """
    _indices = jnp.arange(_MAX_TASKS, dtype=jnp.int32)

    def insert_one(carry, i):
        ts, tg, td, tdu, tr, tt, te = carry
        should = i < n_arrive

        # First inactive slot: map inactive→index, occupied→_MAX_TASKS; take min
        free_indices = jnp.where(ts == jnp.int32(_INACTIVE), _indices, jnp.int32(_MAX_TASKS))
        first_free = jnp.min(free_indices)
        has_free = first_free < jnp.int32(_MAX_TASKS)
        do = should & has_free

        # Clamp slot to valid range (dummy slot=0 when not inserting; update is gated by `do`)
        slot = jnp.where(do, first_free, jnp.int32(0))

        tg = tg.at[slot].set(jnp.where(do, gpus_samples[i], tg[slot]))
        td = td.at[slot].set(jnp.where(do, deadline_samples[i], td[slot]))
        tdu = tdu.at[slot].set(jnp.where(do, dur_samples[i], tdu[slot]))
        tr = tr.at[slot].set(jnp.where(do, dur_samples[i], tr[slot]))
        tt = tt.at[slot].set(jnp.where(do, jnp.int32(type_id), tt[slot]))
        te = te.at[slot].set(jnp.where(do, jnp.float32(eta_val), te[slot]))
        ts = ts.at[slot].set(jnp.where(do, jnp.int32(_WAITING), ts[slot]))

        return (ts, tg, td, tdu, tr, tt, te), None

    init = (task_status, task_gpus, task_deadline, task_duration, task_remaining, task_type, task_eta)
    (ts_out, tg_out, td_out, tdu_out, tr_out, tt_out, te_out), _ = jax.lax.scan(
        insert_one, init, jnp.arange(_MAX_ARRIVALS, dtype=jnp.int32)
    )
    return ts_out, tg_out, td_out, tdu_out, tr_out, tt_out, te_out


def _greedy_edf_schedule(
    task_status: chex.Array,
    task_gpus: chex.Array,
    task_deadline: chex.Array,
    task_duration: chex.Array,
    task_remaining: chex.Array,
    eligible: chex.Array,
    sort_key: chex.Array,
    gpu_budget: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """Greedy EDF-style scheduling: schedule eligible tasks by ascending sort_key within gpu_budget.

    Algorithm (JIT/vmap safe):
        1. Sort eligible tasks by ascending sort_key (non-eligible slots → INF).
        2. Cumulative sum of GPU demands in sorted order.
        3. Mark tasks whose cumulative sum fits within budget as scheduled.
        4. Unsort via inverse permutation to recover per-slot scheduled mask.
        5. Update task_status (1→2) and task_remaining (=task_duration) for scheduled.

    Returns:
        updated task_status, updated task_remaining
    """
    _INF = jnp.int32(1_000_000)
    effective_key = jnp.where(eligible, sort_key, _INF)
    sort_idx = jnp.argsort(effective_key)

    sorted_gpus = jnp.where(eligible[sort_idx], task_gpus[sort_idx], jnp.int32(0))
    cum_gpus = jnp.cumsum(sorted_gpus)
    fits_sorted = (cum_gpus <= gpu_budget) & eligible[sort_idx]

    # Inverse permutation: map fits_sorted back to original task indices
    inv_sort = jnp.argsort(sort_idx)
    should_schedule = fits_sorted[inv_sort]

    new_status = jnp.where(should_schedule, jnp.int32(_RUNNING), task_status)
    new_remaining = jnp.where(should_schedule, task_duration, task_remaining)
    return new_status, new_remaining


# Environment
class DataCenterEnv(Environment):
    """AI Data Center resource environment (pure-JAX).

    Action: 3-D in [0, 1]:
        action[0] r_train    — GPU fraction for training
        action[1] r_finetune — GPU fraction for finetuning (of remaining capacity)
        action[2] cool_norm  — cooling setpoint fraction

    Observation: 11-D (see module docstring).
    Info keys: ``cost`` (SLA density), ``cost_overtemp``, ``n_expired``, ``p_dc_mw``.
    """

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: DataCenterParams
    ) -> Tuple[chex.Array, DataCenterState]:
        state = DataCenterState(
            current_p_mw=jnp.float32(0.0),
            current_q_mvar=jnp.float32(0.0),
            time_step=jnp.int32(0),
            t_zone=jnp.float32(params.t_initial),
            t_setpoint=jnp.float32(params.t_initial),
            t_outdoor=jnp.float32(params.t_ref),
            p_it_mw=jnp.float32(0.0),
            p_cool_mw=jnp.float32(0.0),
            gpus_infer=jnp.int32(0),
            gpus_active=jnp.int32(0),
            task_gpus=jnp.zeros(_MAX_TASKS, dtype=jnp.int32),
            task_deadline=jnp.zeros(_MAX_TASKS, dtype=jnp.int32),
            task_duration=jnp.zeros(_MAX_TASKS, dtype=jnp.int32),
            task_remaining=jnp.zeros(_MAX_TASKS, dtype=jnp.int32),
            task_type=jnp.zeros(_MAX_TASKS, dtype=jnp.int32),
            task_eta=jnp.zeros(_MAX_TASKS, dtype=jnp.float32),
            task_status=jnp.zeros(_MAX_TASKS, dtype=jnp.int32),
            sla_violations=jnp.int32(0),
            done=jnp.bool_(False),
        )
        return self._get_obs(state, params), state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: DataCenterState,
        action: chex.Array,
        params: DataCenterParams,
    ) -> Tuple[chex.Array, DataCenterState, chex.Array, chex.Array, chex.Array, Dict[str, Any]]:

        # 1. Parse action
        action = jnp.asarray(action, dtype=jnp.float32).reshape(3)
        r_train = jnp.clip(action[0], 0.0, 1.0)
        r_ft = jnp.clip(action[1], 0.0, 1.0)
        cool_norm = jnp.clip(action[2], 0.0, 1.0)

        # 2. Cooling setpoint
        t_setpoint = jnp.float32(params.t_set_min) + cool_norm * jnp.float32(
            params.t_set_max - params.t_set_min
        )

        # 3. Outdoor temperature (synthetic diurnal curve)
        t_outdoor = _outdoor_temp(state.time_step, params.steps_per_day)

        # 4. Inference GPUs (exogenous diurnal load)
        gpus_infer = jnp.int32(jnp.floor(
            jnp.float32(params.infer_gpu_peak) * _diurnal_factor(state.time_step, params.steps_per_day)
        ))

        # 5. Generate arrivals (Poisson, per type)
        key, k_tn, k_tg, k_td, k_fn, k_fg, k_fd = jax.random.split(key, 7)

        lam_train = 1.0 / params.train_arrival_interval
        n_train = jnp.int32(
            jnp.minimum(
                jax.random.poisson(k_tn, lam_train).astype(jnp.int32),
                jnp.int32(_MAX_ARRIVALS),
            )
        )
        train_gpus = jax.random.randint(
            k_tg,
            (_MAX_ARRIVALS,),
            params.train_gpu_lo,
            params.train_gpu_hi + 1,
            dtype=jnp.int32,
        )
        train_durs = jax.random.randint(
            k_td,
            (_MAX_ARRIVALS,),
            params.train_dur_lo,
            params.train_dur_hi + 1,
            dtype=jnp.int32,
        )
        train_deadlines = state.time_step + (
            train_durs.astype(jnp.float32) * jnp.float32(params.train_deadline_slack)
        ).astype(jnp.int32)

        task_status, task_gpus, task_deadline, task_duration, task_remaining, task_type, task_eta = \
            _insert_tasks(
                state.task_status, state.task_gpus, state.task_deadline, state.task_duration,
                state.task_remaining, state.task_type, state.task_eta,
                n_train, train_gpus, train_durs, train_deadlines, _TRAIN, params.train_gpu_eta,
            )

        lam_ft = 1.0 / params.ft_arrival_interval
        n_ft = jnp.int32(
            jnp.minimum(
                jax.random.poisson(k_fn, lam_ft).astype(jnp.int32),
                jnp.int32(_MAX_ARRIVALS),
            )
        )
        ft_gpus = jax.random.randint(
            k_fg,
            (_MAX_ARRIVALS,),
            params.ft_gpu_lo,
            params.ft_gpu_hi + 1,
            dtype=jnp.int32,
        )
        ft_durs = jax.random.randint(
            k_fd,
            (_MAX_ARRIVALS,),
            params.ft_dur_lo,
            params.ft_dur_hi + 1,
            dtype=jnp.int32,
        )
        ft_deadlines = state.time_step + (
            ft_durs.astype(jnp.float32) * jnp.float32(params.ft_deadline_slack)
        ).astype(jnp.int32)

        task_status, task_gpus, task_deadline, task_duration, task_remaining, task_type, task_eta = \
            _insert_tasks(
                task_status, task_gpus, task_deadline, task_duration,
                task_remaining, task_type, task_eta,
                n_ft, ft_gpus, ft_durs, ft_deadlines, _FINETUNE, params.ft_gpu_eta,
            )

        # 6. Force-schedule urgent tasks (slack-to-start <= 0)
        gpus_running_pre = jnp.int32(
            jnp.sum(
                jnp.where(task_status == jnp.int32(_RUNNING), task_gpus, jnp.int32(0))
            )
        )
        gpus_capacity_urgent = jnp.maximum(
            jnp.int32(0), jnp.int32(params.n_gpus) - gpus_infer - gpus_running_pre
        )

        # slack = deadline - time_step - duration (same as PowerZoo _schedule_urgent)
        slack_start = task_deadline - state.time_step - task_duration
        urgent_eligible = (task_status == jnp.int32(_WAITING)) & (slack_start <= jnp.int32(0))
        task_status, task_remaining = _greedy_edf_schedule(
            task_status, task_gpus, task_deadline, task_duration, task_remaining,
            urgent_eligible, task_deadline, gpus_capacity_urgent,
        )

        # 7. RL-controlled scheduling (train then finetuning)
        gpus_running_post_urgent = jnp.int32(
            jnp.sum(
                jnp.where(task_status == jnp.int32(_RUNNING), task_gpus, jnp.int32(0))
            )
        )
        gpus_avail = jnp.maximum(
            jnp.int32(0), jnp.int32(params.n_gpus) - gpus_infer - gpus_running_post_urgent
        )

        gpu_budget_train = (r_train * gpus_avail.astype(jnp.float32)).astype(jnp.int32)
        train_eligible = (task_status == jnp.int32(_WAITING)) & (task_type == jnp.int32(_TRAIN))
        task_status, task_remaining = _greedy_edf_schedule(
            task_status, task_gpus, task_deadline, task_duration, task_remaining,
            train_eligible, task_deadline - state.time_step - task_duration, gpu_budget_train,
        )

        gpus_running_post_train = jnp.int32(
            jnp.sum(
                jnp.where(task_status == jnp.int32(_RUNNING), task_gpus, jnp.int32(0))
            )
        )
        gpus_avail_ft = jnp.maximum(
            jnp.int32(0), jnp.int32(params.n_gpus) - gpus_infer - gpus_running_post_train
        )
        gpu_budget_ft = (r_ft * gpus_avail_ft.astype(jnp.float32)).astype(jnp.int32)
        ft_eligible = (task_status == jnp.int32(_WAITING)) & (task_type == jnp.int32(_FINETUNE))
        task_status, task_remaining = _greedy_edf_schedule(
            task_status, task_gpus, task_deadline, task_duration, task_remaining,
            ft_eligible, task_deadline - state.time_step - task_duration, gpu_budget_ft,
        )

        # 8. Advance running tasks (remaining -= 1, complete at 0)
        is_running_mask = task_status == jnp.int32(_RUNNING)
        new_remaining = jnp.where(is_running_mask, task_remaining - 1, task_remaining)
        new_remaining = jnp.maximum(jnp.int32(0), new_remaining)
        completed = is_running_mask & (new_remaining <= jnp.int32(0))
        task_status = jnp.where(completed, jnp.int32(_INACTIVE), task_status)
        task_remaining = new_remaining

        # 9. Expire waiting tasks whose deadline has passed
        expired = (task_status == jnp.int32(_WAITING)) & (state.time_step >= task_deadline)
        n_expired = jnp.int32(jnp.sum(expired.astype(jnp.int32)))
        task_status = jnp.where(expired, jnp.int32(_INACTIVE), task_status)
        new_sla_violations = jnp.int32(state.sla_violations + n_expired)

        # 10. Compute power
        is_running_final = task_status == jnp.int32(_RUNNING)
        gpus_infer_eff = jnp.minimum(gpus_infer, jnp.int32(params.n_gpus))

        p_infer_w = gpus_infer_eff.astype(jnp.float32) * params.gpu_active_w * jnp.float32(0.5)
        task_power_w = task_gpus.astype(jnp.float32) * (
            params.gpu_idle_w
            + (params.gpu_active_w - params.gpu_idle_w) * task_eta
        )
        p_running_w = jnp.sum(jnp.where(is_running_final, task_power_w, jnp.float32(0.0)))

        gpus_running_final = jnp.int32(
            jnp.sum(jnp.where(is_running_final, task_gpus, jnp.int32(0)))
        )
        gpus_active = jnp.int32(
            jnp.minimum(gpus_infer_eff + gpus_running_final, jnp.int32(params.n_gpus))
        )
        p_idle_w = jnp.maximum(
            jnp.float32(0.0),
            (jnp.int32(params.n_gpus) - gpus_active).astype(jnp.float32),
        ) * params.gpu_idle_w

        p_it_mw = (p_infer_w + p_running_w + p_idle_w) / jnp.float32(1e6) + jnp.float32(params.p_base_mw)

        cop = jnp.float32(params.cop_ref) * jnp.clip(
            jnp.float32(1.0)
            - jnp.float32(params.cop_decay) * jnp.maximum(t_outdoor - jnp.float32(params.t_ref), jnp.float32(0.0)),
            jnp.float32(0.4),
            jnp.float32(1.2),
        )

        # 11. Thermal dynamics (first-order ODE)
        t_zone = state.t_zone
        dt_h = jnp.float32(params.delta_t_hours)
        p_it_kw = p_it_mw * jnp.float32(1e3)

        # q_cool [kW]: heat removed by cooling system, depends on setpoint
        q_cool = jnp.float32(params.ua_cooling) * jnp.maximum(t_zone - t_setpoint, jnp.float32(0.0))
        q_wall = jnp.float32(params.h_wall) * (t_outdoor - t_zone)

        # p_cool_mw derived from q_cool via COP: W_elec = Q_removed / COP
        p_cool_mw = q_cool / (cop * jnp.float32(1e3))
        p_dc_mw = p_it_mw + p_cool_mw + jnp.float32(params.p_aux_frac) * p_it_mw
        t_zone_new = t_zone + dt_h * (p_it_kw - q_cool + q_wall) / jnp.maximum(
            jnp.float32(params.c_thermal), jnp.float32(1e-6)
        )
        t_zone_new = jnp.clip(t_zone_new, jnp.float32(_T_ZONE_MIN), jnp.float32(_T_ZONE_MAX))

        # 12. Grid power (negative = absorbing)
        current_p_mw = -p_dc_mw

        # 13. Time step and done
        new_time = state.time_step + jnp.int32(1)
        done = new_time >= jnp.int32(params.max_steps)

        new_state = DataCenterState(
            current_p_mw=current_p_mw,
            current_q_mvar=jnp.float32(0.0),
            time_step=new_time,
            t_zone=t_zone_new,
            t_setpoint=t_setpoint,
            t_outdoor=t_outdoor,
            p_it_mw=p_it_mw,
            p_cool_mw=p_cool_mw,
            gpus_infer=gpus_infer,
            gpus_active=gpus_active,
            task_gpus=task_gpus,
            task_deadline=task_deadline,
            task_duration=task_duration,
            task_remaining=task_remaining,
            task_type=task_type,
            task_eta=task_eta,
            task_status=task_status,
            sla_violations=new_sla_violations,
            done=done,
        )

        # 14. Auto-reset: swap in fresh state when episode ends
        _, k_reset = jax.random.split(key)
        _, reset_state = self.reset(k_reset, params)
        final_state = tu.tree_map(
            lambda n, r: jnp.where(done, r, n), new_state, reset_state
        )
        final_obs = self._get_obs(final_state, params)

        reward = jnp.float32(0.0)
        cost_sla = n_expired.astype(jnp.float32) / jnp.float32(max(params.n_gpus, 1))
        cost_overtemp = jnp.maximum(
            t_zone_new - jnp.float32(params.t_critical), jnp.float32(0.0)
        ) / jnp.float32(_T_ZONE_MAX - params.t_critical)
        costs = stack_costs(cost_sla, cost_overtemp)
        info: Dict[str, Any] = {
            "cost_sla": cost_sla,
            "cost_overtemp": cost_overtemp,
            "cost_sum": jnp.sum(costs),
            "n_expired": n_expired,
            "p_dc_mw": p_dc_mw,
            "p_cool_mw": p_cool_mw,
        }
        return final_obs, final_state, reward, costs, done, info

    # Spaces & Observation
    def _get_obs(self, state: DataCenterState, params: DataCenterParams) -> chex.Array:
        """11-D observation matching PowerZoo DataCenterEnv.

        [gpu_util, infer_util, q_train_fill, q_ft_fill, queue_urgency,
         cool_ratio, zone_temp_norm, outdoor_temp_norm, setpoint_norm,
         time_sin, time_cos]
        """
        n_gpus_f = jnp.float32(max(params.n_gpus, 1))

        gpu_util = state.gpus_active.astype(jnp.float32) / n_gpus_f
        infer_util = state.gpus_infer.astype(jnp.float32) / n_gpus_f

        is_waiting = state.task_status == jnp.int32(_WAITING)
        is_wt_train = is_waiting & (state.task_type == jnp.int32(_TRAIN))
        is_wt_ft = is_waiting & (state.task_type == jnp.int32(_FINETUNE))

        train_demand = jnp.sum(jnp.where(is_wt_train, state.task_gpus, jnp.int32(0)))
        ft_demand = jnp.sum(jnp.where(is_wt_ft, state.task_gpus, jnp.int32(0)))
        q_train_fill = jnp.clip(train_demand.astype(jnp.float32) / n_gpus_f, 0.0, 1.0)
        q_ft_fill = jnp.clip(ft_demand.astype(jnp.float32) / n_gpus_f, 0.0, 1.0)

        # Queue urgency: mirrors PowerZoo slack = (deadline - now - duration) / duration.
        # Positive → enough lead time; 0 → must start now; negative → already overdue.
        # Clipped to [-1, 1]; default 5.0 when no tasks are waiting (→ urgency = 1.0).
        slack_norm = (
            state.task_deadline - state.time_step - state.task_duration
        ).astype(jnp.float32) / jnp.maximum(
            state.task_duration.astype(jnp.float32), jnp.float32(1.0)
        )
        min_slack = jnp.min(jnp.where(is_waiting, slack_norm, jnp.float32(5.0)))
        queue_urgency = jnp.clip(min_slack / jnp.float32(5.0), -1.0, 1.0)

        p_total = state.p_it_mw + state.p_cool_mw
        cool_ratio = state.p_cool_mw / jnp.maximum(p_total, jnp.float32(1e-9))

        zone_norm = (state.t_zone - jnp.float32(_T_ZONE_MIN)) / jnp.float32(_T_ZONE_MAX - _T_ZONE_MIN)
        outdoor_norm = (state.t_outdoor - jnp.float32(10.0)) / jnp.float32(20.0)  # maps operating range [10, 30] → [0, 1]
        setpoint_norm = (state.t_setpoint - jnp.float32(params.t_set_min)) / jnp.maximum(
            jnp.float32(params.t_set_max - params.t_set_min), jnp.float32(1e-6)
        )

        t_sin, t_cos = time_features(state.time_step, params.steps_per_day)

        return jnp.stack([
            jnp.clip(gpu_util, 0.0, 1.0),
            jnp.clip(infer_util, 0.0, 1.0),
            q_train_fill,
            q_ft_fill,
            queue_urgency,
            jnp.clip(cool_ratio, 0.0, 1.0),
            jnp.clip(zone_norm, 0.0, 1.0),
            jnp.clip(outdoor_norm, 0.0, 1.0),
            jnp.clip(setpoint_norm, 0.0, 1.0),
            t_sin,
            t_cos,
        ])

    def observation_space(self, params: DataCenterParams) -> Box:

        # obs[4] = queue_urgency ∈ [-1, 1]; obs[9:11] = time sin/cos ∈ [-1, 1]
        low = jnp.array([0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0], dtype=jnp.float32)
        high = jnp.ones(11, dtype=jnp.float32)
        return Box(low=low, high=high, shape=(11,), dtype=jnp.float32)

    def action_space(self, params: DataCenterParams) -> Box:
        return Box(
            low=jnp.zeros(3, dtype=jnp.float32),
            high=jnp.ones(3, dtype=jnp.float32),
            shape=(3,),
            dtype=jnp.float32,
        )

    # Diagnostics
    @property
    def name(self) -> str:
        return "DataCenterEnv"

    def default_params(self) -> DataCenterParams:
        return DataCenterParams()

    def constraint_names(self, params: DataCenterParams) -> tuple[str, ...]:
        return ("sla", "overtemp")


# Factory Function
def make_datacenter_params(
    n_gpus: int = 1000,
    gpu_idle_w: float = 55.0,
    gpu_active_w: float = 580.0,
    p_base_mw: float = 0.5,
    infer_gpu_peak: int = 400,
    cop_ref: float = 5.0,
    cop_decay: float = 0.04,
    t_ref: float = 20.0,
    c_thermal: float = 500.0,
    ua_cooling: float = 200.0,
    h_wall: float = 5.0,
    t_set_min: float = 18.0,
    t_set_max: float = 27.0,
    t_initial: float = 22.0,
    t_critical: float = 35.0,
    p_aux_frac: float = 0.05,
    train_arrival_interval: int = 8,
    train_gpu_lo: int = 50,
    train_gpu_hi: int = 200,
    train_dur_lo: int = 10,
    train_dur_hi: int = 50,
    train_deadline_slack: float = 2.0,
    train_gpu_eta: float = 0.90,
    ft_arrival_interval: int = 4,
    ft_gpu_lo: int = 10,
    ft_gpu_hi: int = 50,
    ft_dur_lo: int = 5,
    ft_dur_hi: int = 20,
    ft_deadline_slack: float = 3.0,
    ft_gpu_eta: float = 0.75,
    delta_t_hours: float = 0.5,
    steps_per_day: int = 48,
    max_steps: int = 48,
) -> DataCenterParams:
    """Build DataCenterParams with physical defaults matching PowerZoo DataCenterEnv."""
    return DataCenterParams(
        n_gpus=n_gpus,
        gpu_idle_w=gpu_idle_w,
        gpu_active_w=gpu_active_w,
        p_base_mw=p_base_mw,
        infer_gpu_peak=infer_gpu_peak,
        cop_ref=cop_ref,
        cop_decay=cop_decay,
        t_ref=t_ref,
        c_thermal=c_thermal,
        ua_cooling=ua_cooling,
        h_wall=h_wall,
        t_set_min=t_set_min,
        t_set_max=t_set_max,
        t_initial=t_initial,
        t_critical=t_critical,
        p_aux_frac=p_aux_frac,
        train_arrival_interval=train_arrival_interval,
        train_gpu_lo=train_gpu_lo,
        train_gpu_hi=train_gpu_hi,
        train_dur_lo=train_dur_lo,
        train_dur_hi=train_dur_hi,
        train_deadline_slack=train_deadline_slack,
        train_gpu_eta=train_gpu_eta,
        ft_arrival_interval=ft_arrival_interval,
        ft_gpu_lo=ft_gpu_lo,
        ft_gpu_hi=ft_gpu_hi,
        ft_dur_lo=ft_dur_lo,
        ft_dur_hi=ft_dur_hi,
        ft_deadline_slack=ft_deadline_slack,
        ft_gpu_eta=ft_gpu_eta,
        delta_t_hours=delta_t_hours,
        steps_per_day=steps_per_day,
        max_steps=max_steps,
    )
