"""DataCenter Microgrid Environment — pure-JAX implementation.

A self-contained behind-the-meter microgrid: an AI data-center load
on one DC bus, served by a battery, a PV plant, and a diesel generator,
with optional capped grid import. The agent jointly decides GPU
scheduling and cooling setpoint for the data-center sub-env *and*
battery / diesel dispatch on the supply side, minimising total energy,
$ cost (fuel + grid + battery degradation + terminal-SOC drift), and
CO2 emissions, subject to SLA, thermal, and power-balance constraints
under stochastic workload arrivals and exogenous sun / temperature /
price profiles.

Architecture: a *composite* env. The DataCenter physics live in a
sub-env (not a bundle — it owns task-queue state that doesn't fit the
ResourceBundle signature). Battery / PV / Diesel attach as
``ResourceBundle`` instances in busless mode: injections sum directly
into the single bus rather than scattering into a node vector. The DC
sub-step is the private ``_dc_step_inner`` (no auto-reset); a single
unified auto-reset wraps DC + all bundle states at the env level.

Power balance (no losses)::

    p_supply      = Σ p_inj_b      (batt + PV + DG)
    p_load        = −dc.current_p_mw                       (DC absorbs)
    residual      = p_supply − p_load
    p_grid_import = clip(−residual, 0, grid_import_p_max_mw)
    power_deficit = max(−(residual + p_grid_import), 0)    → CMDP cost
    power_spill   = max( residual + p_grid_import, 0)      → info only

``grid_import_p_max_mw`` defaults to 0 (true island); > 0 enables a
hybrid with capped grid backup.

Action (5-D, Box)::

    [0] train_sched_rate       ∈ [0, 1]    → DC sub-env
    [1] ft_sched_rate          ∈ [0, 1]    → DC sub-env
    [2] cooling_setpoint_norm  ∈ [0, 1]    → DC sub-env
    [3] battery_power_norm     ∈ [−1, 1]   → BatteryBundle (+ = discharge)
    [4] dg_power_norm          ∈ [0, 1]    → DieselBundle
    PV is profile-driven (no action; bundle built in no-control mode).

Observation (24-D)::

    [cpu_util, mem_util, q_train_fill, q_ft_fill, queue_urgency,
     zone_temp_norm, outdoor_temp_norm, cop_ratio,
     solar_cf, soc, dg_margin_norm, p_load_norm, net_load_norm,
     batt_dis_headroom_norm, batt_chg_headroom_norm,
     grid_price_norm, grid_price_6h_max_norm,
     last_action_norm(5), sin(t), cos(t)]

The 5-channel last-action echo lets the policy learn smooth dispatch
without thrashing across consecutive steps.

Scalarised reward (component vector also in info["reward_vector"])::

    r_energy = −(p_dc_mw · dt_h)                                   [MWh]
    r_cost   = −(fuel + grid + |p_batt|·dt·battery_deg_cost_per_mwh
                 + terminal_soc_cost)                               [$]
    r_carbon = −(diesel_carbon_kg + grid_carbon_kg)               [kgCO2]
    reward   = r_energy + w_cost·r_cost + w_carbon·r_carbon

terminal_soc_cost = penalty·(soc − target)² at episode end (default
penalty = 0, opt-in). Battery degradation is applied at this env-level
reward; the inner BatteryBundle's cycle_cost is set to 0 to avoid
double counting.

Constraint costs (CMDP, separate from reward, all ∈ [0, 1])::

    cost_sla           = n_expired / n_gpus       (PowerZoo convention)
    cost_overtemp      = (t_zone − t_critical)⁺ / (T_MAX − t_critical)
    cost_power_deficit = power_deficit / max(p_load, ε)

Exogenous profiles (cyclically indexed via ``arr[t % T]``; see
``powerzoojax.data.dc_microgrid_profiles`` for loaders)::

    cpu_profile          (T,) ∈ [0,1]   inference load — None → synthetic
    outdoor_temp_profile (T,) [°C]                      None → synthetic
    price_profile        (T,) [$/MWh]                   None → no market signal
    PV profile lives on the SolarBundle (single source of truth).

Default config: 288 steps × 5 min = 24 h episode.

``DCMicrogridParams`` / ``DCMicrogridState`` expose backward-compat
properties (``params.battery_capacity_mwh``, ``state.soc``, ...) for
analysis / test code written against the old flat schema.
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
from powerzoojax.envs.resource.base import time_features

# Bundle protocol re-exports
from powerzoojax.envs.resource.battery import (
    BatteryBundle,
    BatteryBundleState,
    make_battery_bundle,
)
from powerzoojax.envs.resource.renewable import (
    RenewableBundle,
    RenewableBundleState,
    make_renewable_bundle,
)
from powerzoojax.envs.resource.diesel import (
    DieselParams,
    DieselBundle,
    DieselBundleState,
    compute_dg_power,
    compute_dg_fuel_cost,
    compute_dg_emissions,
    make_diesel_bundle,
)

# DC core (sub-env)
from powerzoojax.envs.resource.datacenter import (
    DataCenterState,
    DataCenterParams,
    _insert_tasks,
    _greedy_edf_schedule,
    _diurnal_factor,
    _outdoor_temp,
    _MAX_TASKS,
    _MAX_ARRIVALS,
    _INACTIVE,
    _WAITING,
    _RUNNING,
    _TRAIN,
    _FINETUNE,
    _T_ZONE_MIN,
    _T_ZONE_MAX,
)


def _solar_cf(time_step, steps_per_day):
    """Synthetic clip-sin solar CF reference profile.

    Used as the default CF when no profile is supplied to the SolarBundle
    factory and as the deterministic anchor in tests / parity comparisons.
    Matches PowerZoo's old SolarBundle helper bit-for-bit so that runs without
    real PV traces stay reproducible across both backends.
    """
    hour = (
        (time_step % steps_per_day).astype(jnp.float32)
        / jnp.float32(steps_per_day)
        * jnp.float32(24.0)
    )
    return jnp.clip(jnp.sin(jnp.pi * (hour - 6.0) / 12.0), 0.0, 1.0)


# ---------------------------------------------------------------------------
# DC reset helper (mirrors DataCenterEnv.reset without the env object)
# ---------------------------------------------------------------------------

def _dc_reset(params: DataCenterParams) -> DataCenterState:
    """Create a fresh DataCenterState (functional equivalent of DataCenterEnv.reset)."""
    return DataCenterState(
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


# ---------------------------------------------------------------------------
# DC step without auto-reset (replicates DataCenterEnv.step steps 1–13)
# ---------------------------------------------------------------------------

def _dc_step_inner(
    key: chex.PRNGKey,
    state: DataCenterState,
    action: chex.Array,
    params: DataCenterParams,
    *,
    infer_frac: Optional[chex.Array] = None,
    t_outdoor: Optional[chex.Array] = None,
) -> Tuple[DataCenterState, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
    """Execute one DataCenter step without the auto-reset wrapper.

    Replicates the physics of DataCenterEnv.step() up to but not including the
    auto-reset block.  This lets DataCenterMicrogridEnv apply a single unified
    auto-reset covering all resource states simultaneously.

    Args:
        key: PRNG key (consumed for task arrival sampling).
        state: Current DataCenterState.
        action: 3-D array [train_sched_rate, ft_sched_rate, cooling_norm].
        params: DataCenterParams.
        infer_frac: Optional override for inference GPU fraction ∈ [0, 1].
            When None, the synthetic ``_diurnal_factor()`` is used.
        t_outdoor: Optional override for outdoor temperature [°C].
            When None, the synthetic ``_outdoor_temp()`` is used.

    Returns:
        new_dc_state: Updated DataCenterState (done=True at episode end).
        p_dc_mw: Total facility power draw [MW] (positive value).
        cost_sla: SLA violation density (n_expired / n_gpus).
        cost_overtemp: Over-temperature excess, normalised to [0, 1].
        n_expired: Raw count of task expirations this step.
        cop: Cooling coefficient of performance at this step.
    """
    # 1. Parse action
    action = jnp.asarray(action, dtype=jnp.float32).reshape(3)
    r_train = jnp.clip(action[0], 0.0, 1.0)
    r_ft = jnp.clip(action[1], 0.0, 1.0)
    cool_norm = jnp.clip(action[2], 0.0, 1.0)

    # 2. Cooling setpoint
    t_setpoint = jnp.float32(params.t_set_min) + cool_norm * jnp.float32(
        params.t_set_max - params.t_set_min
    )

    # 3. Outdoor temperature
    if t_outdoor is None:
        t_outdoor = _outdoor_temp(state.time_step, params.steps_per_day)

    # 4. Inference GPUs
    if infer_frac is None:
        infer_frac = _diurnal_factor(state.time_step, params.steps_per_day)
    gpus_infer = jnp.int32(jnp.floor(
        jnp.float32(params.infer_gpu_peak) * infer_frac
    ))

    # 5. Generate task arrivals
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

    ts, tg, tdeadl, tdur, trem, ttype, teta = _insert_tasks(
        state.task_status, state.task_gpus, state.task_deadline,
        state.task_duration, state.task_remaining, state.task_type, state.task_eta,
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

    ts, tg, tdeadl, tdur, trem, ttype, teta = _insert_tasks(
        ts, tg, tdeadl, tdur, trem, ttype, teta,
        n_ft, ft_gpus, ft_durs, ft_deadlines, _FINETUNE, params.ft_gpu_eta,
    )

    # 6. Force-schedule urgent tasks
    gpus_running_pre = jnp.int32(
        jnp.sum(jnp.where(ts == jnp.int32(_RUNNING), tg, jnp.int32(0)))
    )
    gpus_cap_urgent = jnp.maximum(
        jnp.int32(0), jnp.int32(params.n_gpus) - gpus_infer - gpus_running_pre
    )
    slack_start = tdeadl - state.time_step - tdur
    urgent_elig = (ts == jnp.int32(_WAITING)) & (slack_start <= jnp.int32(0))
    ts, trem = _greedy_edf_schedule(ts, tg, tdeadl, tdur, trem, urgent_elig, tdeadl, gpus_cap_urgent)

    # 7. RL-controlled scheduling (train then ft)
    gpus_post_urgent = jnp.int32(
        jnp.sum(jnp.where(ts == jnp.int32(_RUNNING), tg, jnp.int32(0)))
    )
    gpus_avail = jnp.maximum(
        jnp.int32(0), jnp.int32(params.n_gpus) - gpus_infer - gpus_post_urgent
    )

    gpu_budget_train = (r_train * gpus_avail.astype(jnp.float32)).astype(jnp.int32)
    train_elig = (ts == jnp.int32(_WAITING)) & (ttype == jnp.int32(_TRAIN))
    ts, trem = _greedy_edf_schedule(
        ts, tg, tdeadl, tdur, trem,
        train_elig, tdeadl - state.time_step - tdur, gpu_budget_train,
    )

    gpus_post_train = jnp.int32(
        jnp.sum(jnp.where(ts == jnp.int32(_RUNNING), tg, jnp.int32(0)))
    )
    gpus_avail_ft = jnp.maximum(
        jnp.int32(0), jnp.int32(params.n_gpus) - gpus_infer - gpus_post_train
    )
    gpu_budget_ft = (r_ft * gpus_avail_ft.astype(jnp.float32)).astype(jnp.int32)
    ft_elig = (ts == jnp.int32(_WAITING)) & (ttype == jnp.int32(_FINETUNE))
    ts, trem = _greedy_edf_schedule(
        ts, tg, tdeadl, tdur, trem,
        ft_elig, tdeadl - state.time_step - tdur, gpu_budget_ft,
    )

    # 8. Advance running tasks
    is_running_mask = ts == jnp.int32(_RUNNING)
    new_remaining = jnp.where(is_running_mask, trem - 1, trem)
    new_remaining = jnp.maximum(jnp.int32(0), new_remaining)
    completed = is_running_mask & (new_remaining <= jnp.int32(0))
    ts = jnp.where(completed, jnp.int32(_INACTIVE), ts)
    trem = new_remaining

    # 9. Expire waiting tasks
    expired_mask = (ts == jnp.int32(_WAITING)) & (state.time_step >= tdeadl)
    n_expired = jnp.int32(jnp.sum(expired_mask.astype(jnp.int32)))
    ts = jnp.where(expired_mask, jnp.int32(_INACTIVE), ts)
    new_sla_violations = jnp.int32(state.sla_violations + n_expired)

    # 10. Compute power
    is_running_final = ts == jnp.int32(_RUNNING)
    gpus_infer_eff = jnp.minimum(gpus_infer, jnp.int32(params.n_gpus))

    p_infer_w = (
        gpus_infer_eff.astype(jnp.float32) * jnp.float32(params.gpu_active_w) * jnp.float32(0.5)
    )
    task_power_w = tg.astype(jnp.float32) * (
        jnp.float32(params.gpu_idle_w)
        + (jnp.float32(params.gpu_active_w) - jnp.float32(params.gpu_idle_w)) * teta
    )
    p_running_w = jnp.sum(jnp.where(is_running_final, task_power_w, jnp.float32(0.0)))

    gpus_running_final = jnp.int32(
        jnp.sum(jnp.where(is_running_final, tg, jnp.int32(0)))
    )
    gpus_active = jnp.int32(
        jnp.minimum(gpus_infer_eff + gpus_running_final, jnp.int32(params.n_gpus))
    )
    p_idle_w = jnp.maximum(
        jnp.float32(0.0),
        (jnp.int32(params.n_gpus) - gpus_active).astype(jnp.float32),
    ) * jnp.float32(params.gpu_idle_w)

    p_it_mw = (
        (p_infer_w + p_running_w + p_idle_w) / jnp.float32(1e6)
        + jnp.float32(params.p_base_mw)
    )

    cop = jnp.float32(params.cop_ref) * jnp.clip(
        jnp.float32(1.0)
        - jnp.float32(params.cop_decay)
        * jnp.maximum(t_outdoor - jnp.float32(params.t_ref), jnp.float32(0.0)),
        jnp.float32(0.4),
        jnp.float32(1.2),
    )

    # 11. Thermal dynamics
    t_zone = state.t_zone
    dt_h_f = jnp.float32(params.delta_t_hours)
    p_it_kw = p_it_mw * jnp.float32(1e3)
    q_cool = jnp.float32(params.ua_cooling) * jnp.maximum(
        t_zone - t_setpoint, jnp.float32(0.0)
    )
    q_wall = jnp.float32(params.h_wall) * (t_outdoor - t_zone)
    p_cool_mw = q_cool / (cop * jnp.float32(1e3))
    p_dc_mw = p_it_mw + p_cool_mw + jnp.float32(params.p_aux_frac) * p_it_mw
    t_zone_new = t_zone + dt_h_f * (p_it_kw - q_cool + q_wall) / jnp.maximum(
        jnp.float32(params.c_thermal), jnp.float32(1e-6)
    )
    t_zone_new = jnp.clip(t_zone_new, jnp.float32(_T_ZONE_MIN), jnp.float32(_T_ZONE_MAX))

    # 12. Grid-side power (negative = absorbing)
    current_p_mw = -p_dc_mw

    # 13. Time step and done
    new_time = state.time_step + jnp.int32(1)
    done = new_time >= jnp.int32(params.max_steps)

    new_dc_state = DataCenterState(
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
        task_gpus=tg,
        task_deadline=tdeadl,
        task_duration=tdur,
        task_remaining=trem,
        task_type=ttype,
        task_eta=teta,
        task_status=ts,
        sla_violations=new_sla_violations,
        done=done,
    )

    cost_sla = n_expired.astype(jnp.float32) / jnp.float32(max(params.n_gpus, 1))
    cost_overtemp = jnp.maximum(
        t_zone_new - jnp.float32(params.t_critical), jnp.float32(0.0)
    ) / jnp.float32(_T_ZONE_MAX - params.t_critical)

    return new_dc_state, p_dc_mw, cost_sla, cost_overtemp, n_expired, cop


# ---------------------------------------------------------------------------
# State / Params
# ---------------------------------------------------------------------------

@struct.dataclass
class DCMicrogridState:
    """Full state of the DataCenterMicrogridEnv.

    Attributes:
        dc: DataCenterState — task buffer, thermal, IT power.
        resource_states: tuple of bundle states, one per attached
            ResourceBundle in ``params.resources``.  By convention the
            default factory builds them in the order
            ``(BatteryBundleState, SolarBundleState, DieselBundleState)``.
        last_action: Last 5-D normalised action (history feature in obs).
        done: Episode termination flag.
    """
    dc: DataCenterState
    resource_states: tuple
    last_action: chex.Array  # float32 (5,)
    done: chex.Array         # bool scalar

    # ------------------------------------------------------------------
    # Backward-compatible scalar accessors that look up nested bundle states.
    # ``state.soc`` etc. used to be top-level fields; analyses and tests
    # still reach for them.
    # ------------------------------------------------------------------

    def _bundle_state(self, idx):
        if idx < 0 or idx >= len(self.resource_states):
            return None
        return self.resource_states[idx]

    @property
    def soc(self) -> chex.Array:
        for i, bs in enumerate(self.resource_states):
            if isinstance(bs, BatteryBundleState):
                return bs.soc[0]
        return jnp.float32(0.0)

    @property
    def p_dg_mw(self) -> chex.Array:
        for i, bs in enumerate(self.resource_states):
            if isinstance(bs, DieselBundleState):
                return bs.p_dg_mw[0]
        return jnp.float32(0.0)

    @property
    def p_pv_mw(self) -> chex.Array:
        for i, bs in enumerate(self.resource_states):
            if isinstance(bs, RenewableBundleState):
                return bs.p_mw[0]
        return jnp.float32(0.0)


@struct.dataclass
class DCMicrogridParams:
    """Parameters for DataCenterMicrogridEnv.

    Attributes:
        dc: DataCenterParams sub-config (IT + thermal + task generation).
        resources: tuple of attached ``ResourceBundle`` instances.
            Default factory builds ``(BatteryBundle, SolarBundle, DieselBundle)``.
            The env's ``step()`` dispatches actions to bundles in this order.
        battery_deg_cost_per_mwh: Degradation cost [$/MWh of throughput].
            Applied at the env-level reward layer (NOT inside BatteryBundle).
        w_cost: Weight on r_cost in scalarized reward.
        w_carbon: Weight on r_carbon in scalarized reward.
        grid_import_p_max_mw: Maximum external grid import [MW].
        grid_price_ref_per_mwh: Price scale used to normalize observation
            features [currency/MWh].
        grid_carbon_kg_per_kwh: Grid-import carbon intensity.
        terminal_soc_target: End-of-episode SOC inventory target.
        terminal_soc_penalty: Quadratic terminal SOC inventory penalty.
        cpu_profile: Optional (T,) float32 ∈ [0,1] inference load fraction.
            None → DataCenterEnv's synthetic ``_diurnal_factor()``.
        outdoor_temp_profile: Optional (T,) float32 [°C] outdoor temperature.
            None → DataCenterEnv's synthetic ``_outdoor_temp()``.
        price_profile: Optional (T,) float32 grid-import price [currency/MWh].
            None → price signal is zero.

    Note:
        The ``solar_profile`` is now owned by the SolarBundle (see
        ``make_solar_bundle(profile=...)``) — there is no env-level
        ``solar_profile`` field, eliminating a previous source-of-truth
        ambiguity.
    """
    dc: DataCenterParams = struct.field(pytree_node=False)
    resources: tuple = ()
    battery_deg_cost_per_mwh: float = struct.field(pytree_node=False, default=5.0)
    w_cost: float = struct.field(pytree_node=False, default=0.5)
    w_carbon: float = struct.field(pytree_node=False, default=0.3)
    grid_import_p_max_mw: float = struct.field(pytree_node=False, default=0.0)
    grid_price_ref_per_mwh: float = struct.field(pytree_node=False, default=150.0)
    grid_carbon_kg_per_kwh: float = struct.field(pytree_node=False, default=0.18)
    terminal_soc_target: float = struct.field(pytree_node=False, default=0.5)
    terminal_soc_penalty: float = struct.field(pytree_node=False, default=0.0)
    cpu_profile: chex.Array = None
    outdoor_temp_profile: chex.Array = None
    price_profile: chex.Array = None

    # ------------------------------------------------------------------
    # Backward-compatible accessors that look up nested bundles.
    # These let analysis / test / OOD code that was written against the
    # old flat schema (e.g. ``params.battery_capacity_mwh``) keep working.
    # ------------------------------------------------------------------

    def _find(self, cls):
        for b in self.resources:
            if isinstance(b, cls):
                return b
        return None

    @property
    def battery(self):
        return self._find(BatteryBundle)

    @property
    def solar(self):
        return self._find(RenewableBundle)

    @property
    def diesel(self):
        return self._find(DieselBundle)

    @property
    def battery_capacity_mwh(self) -> float:
        b = self.battery
        return float(b.capacity[0]) if b is not None else 0.0

    @property
    def battery_power_mw(self) -> float:
        b = self.battery
        return float(b.power_max[0]) if b is not None else 0.0

    @property
    def battery_eta_charge(self) -> float:
        b = self.battery
        return float(b.eta_charge[0]) if b is not None else 1.0

    @property
    def battery_eta_discharge(self) -> float:
        b = self.battery
        return float(b.eta_discharge[0]) if b is not None else 1.0

    @property
    def battery_soc_min(self) -> float:
        b = self.battery
        return float(b.soc_min[0]) if b is not None else 0.0

    @property
    def battery_soc_max(self) -> float:
        b = self.battery
        return float(b.soc_max[0]) if b is not None else 1.0

    @property
    def battery_soc_init(self) -> float:
        b = self.battery
        return float(b.initial_soc[0]) if b is not None else 0.5

    @property
    def pv_p_max_mw(self) -> float:
        s = self.solar
        return float(s.capacity_mw[0]) if s is not None else 0.0

    @property
    def solar_profile(self):
        """Backward-compat: returns the (T,) PV CF profile.

        The RenewableBundle stores profiles as ``(T, n_devices)``; for the
        single-device microgrid PV we squeeze along axis 1 so callers see the
        same shared (T,) shape they used in the SolarBundle / inline era.
        """
        s = self.solar
        if s is None or s.profiles is None:
            return None
        if s.profiles.ndim == 2:
            return s.profiles[:, 0]
        return s.profiles

    @property
    def dg(self):
        """Backward-compat shim: returns a single-device DieselParams view."""
        d = self.diesel
        if d is None:
            return None
        return DieselParams(
            p_dg_max_mw=float(d.p_max[0]),
            fuel_cost_per_mwh=float(d.fuel_cost_per_mwh[0]),
            emission_factor=float(d.emission_factor[0]),
        )


# ---------------------------------------------------------------------------
# Bundle index helpers (Python-level type lookup; safe at trace time)
# ---------------------------------------------------------------------------

def _find_bundle_index(resources: tuple, cls) -> int:
    """Return the position of the first bundle of type ``cls`` in ``resources``.

    Returns -1 when no such bundle is attached.  Pure Python at trace time.
    """
    for i, b in enumerate(resources):
        if isinstance(b, cls):
            return i
    return -1


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class DataCenterMicrogridEnv(Environment):
    """Self-contained data-center microgrid environment (pure-JAX).

    Combines DataCenter compute/thermal dynamics with a battery, PV, and
    diesel generator attached via the ``ResourceBundle`` protocol, with
    optional capped grid import.

    Action: Box(5) — [train_sched, ft_sched, cooling_norm, batt_norm, dg_norm]
    Obs:    Box(24) — see module docstring.
    """

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: DCMicrogridParams
    ) -> Tuple[chex.Array, DCMicrogridState]:
        dc_state = _dc_reset(params.dc)
        # Bundle resets are trace-time unrolled (params.resources is a static tuple).
        keys = jax.random.split(key, len(params.resources)) if len(params.resources) > 0 else ()
        resource_states = tuple(
            b.reset(keys[i]) for i, b in enumerate(params.resources)
        )
        state = DCMicrogridState(
            dc=dc_state,
            resource_states=resource_states,
            last_action=jnp.zeros(5, dtype=jnp.float32),
            done=jnp.bool_(False),
        )
        return self._get_obs(state, params), state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: DCMicrogridState,
        action: chex.Array,
        params: DCMicrogridParams,
    ) -> Tuple[chex.Array, DCMicrogridState, chex.Array, chex.Array, chex.Array, Dict[str, Any]]:
        dt_h = params.dc.delta_t_hours

        # --- 1. Parse and clip action ---
        action = jnp.asarray(action, dtype=jnp.float32).reshape(5)
        train_sched = jnp.clip(action[0], 0.0, 1.0)
        ft_sched = jnp.clip(action[1], 0.0, 1.0)
        cool_norm = jnp.clip(action[2], 0.0, 1.0)
        batt_norm = jnp.clip(action[3], -1.0, 1.0)
        dg_norm = jnp.clip(action[4], 0.0, 1.0)
        clipped_action = jnp.stack([train_sched, ft_sched, cool_norm, batt_norm, dg_norm])

        # --- 2. DataCenter sub-step (no auto-reset) ---
        key, k_dc, k_reset = jax.random.split(key, 3)
        dc_action = jnp.stack([train_sched, ft_sched, cool_norm])

        t = state.dc.time_step
        _infer_frac = None
        _t_outdoor = None
        if params.cpu_profile is not None:
            T_cpu = params.cpu_profile.shape[0]
            _infer_frac = params.cpu_profile[t % T_cpu]
        if params.outdoor_temp_profile is not None:
            T_tmp = params.outdoor_temp_profile.shape[0]
            _t_outdoor = params.outdoor_temp_profile[t % T_tmp]

        new_dc, p_dc_mw, cost_sla, cost_overtemp, n_expired, cop = _dc_step_inner(
            k_dc, state.dc, dc_action, params.dc,
            infer_frac=_infer_frac,
            t_outdoor=_t_outdoor,
        )
        p_load_mw = -new_dc.current_p_mw  # positive load (DC absorbs)

        # --- 3. Bundle dispatch (mirrors DistGridEnv.step:299-308) ---
        # Action layout for bundles, after slot 0..2 (DC):
        #   battery_slice = action[3 : 3 + battery.action_dim]
        #   solar_slice   = empty (zero-action)
        #   diesel_slice  = action[3 + 1 + 0 : 3 + 1 + 0 + diesel.action_dim]
        # We rely on the standard action layout produced by make_dcmicrogrid_params
        # but support arbitrary attached bundles via per-type slot extraction:
        # battery uses slot 3, diesel uses slot 4 (one scalar each in the
        # default config).  Solar consumes 0 action.  Other bundle types are
        # passed an empty slice (defensive default).
        bundle_actions = []
        for b in params.resources:
            if isinstance(b, BatteryBundle):
                bundle_actions.append(jnp.array([batt_norm], dtype=jnp.float32))
            elif isinstance(b, DieselBundle):
                bundle_actions.append(jnp.array([dg_norm], dtype=jnp.float32))
            elif isinstance(b, RenewableBundle):
                # No-control PV mode → zero-length action; otherwise pass zeros
                # (DCMicrogrid leaves PV exogenous regardless of policy).
                bundle_actions.append(jnp.zeros((b.action_dim,), dtype=jnp.float32))
            else:
                bundle_actions.append(jnp.zeros((b.action_dim,), dtype=jnp.float32))

        ctx = {"t": state.dc.time_step}
        new_resource_states = []
        p_supply_mw = jnp.float32(0.0)
        # Per-source injections + economics (default zero when bundle absent)
        p_batt_mw = jnp.float32(0.0)
        p_pv_mw = jnp.float32(0.0)
        p_dg_mw = jnp.float32(0.0)
        fuel_cost = jnp.float32(0.0)
        carbon_kg = jnp.float32(0.0)

        for i, bundle in enumerate(params.resources):
            new_bs, p_inj, _q_inj, _obs_slice, info_b = bundle.step(
                state.resource_states[i], bundle_actions[i], ctx
            )
            new_resource_states.append(new_bs)
            p_supply_mw = p_supply_mw + jnp.sum(p_inj)
            if isinstance(bundle, BatteryBundle):
                p_batt_mw = jnp.sum(p_inj)
            elif isinstance(bundle, RenewableBundle):
                p_pv_mw = jnp.sum(p_inj)
            elif isinstance(bundle, DieselBundle):
                p_dg_mw = jnp.sum(p_inj)
                fuel_cost = fuel_cost + info_b.get("fuel_cost", jnp.float32(0.0))
                carbon_kg = carbon_kg + info_b.get("carbon_kg", jnp.float32(0.0))

        new_resource_states_tuple = tuple(new_resource_states)

        # --- 4. Power balance (behind-the-meter bus with optional grid import) ---
        residual = p_supply_mw - p_load_mw
        raw_power_deficit = jnp.maximum(-residual, jnp.float32(0.0))
        p_grid_import_mw = jnp.minimum(
            raw_power_deficit,
            jnp.float32(max(params.grid_import_p_max_mw, 0.0)),
        )
        residual_after_grid = residual + p_grid_import_mw
        power_deficit = jnp.maximum(-residual_after_grid, jnp.float32(0.0))
        power_spill = jnp.maximum(residual_after_grid, jnp.float32(0.0))
        grid_price_per_mwh = _read_grid_price_from_time(t, params)
        grid_cost = p_grid_import_mw * jnp.float32(dt_h) * grid_price_per_mwh
        grid_carbon_kg = (
            p_grid_import_mw
            * jnp.float32(dt_h)
            * jnp.float32(1e3)
            * jnp.float32(params.grid_carbon_kg_per_kwh)
        )

        # --- 5. Done & raw new state ---
        done = new_dc.done
        new_state_raw = DCMicrogridState(
            dc=new_dc,
            resource_states=new_resource_states_tuple,
            last_action=clipped_action,
            done=done,
        )

        # --- 6. Reward (scalarized; vector in info) ---
        soc_after_step = _read_soc_from_state(new_state_raw, params)
        terminal_soc_gap = soc_after_step - jnp.float32(params.terminal_soc_target)
        terminal_soc_cost = jnp.where(
            done,
            jnp.float32(params.terminal_soc_penalty) * terminal_soc_gap * terminal_soc_gap,
            jnp.float32(0.0),
        )
        r_energy = -(p_dc_mw * jnp.float32(dt_h))
        r_cost = -(
            fuel_cost
            + grid_cost
            + jnp.abs(p_batt_mw) * jnp.float32(dt_h) * jnp.float32(params.battery_deg_cost_per_mwh)
            + terminal_soc_cost
        )
        r_carbon = -(carbon_kg + grid_carbon_kg)
        reward = (
            r_energy
            + jnp.float32(params.w_cost) * r_cost
            + jnp.float32(params.w_carbon) * r_carbon
        )

        # --- 7. CMDP costs ---
        cost_power_deficit = power_deficit / jnp.maximum(p_load_mw, jnp.float32(1e-6))
        total_cost = cost_sla + cost_overtemp + cost_power_deficit

        # --- 8. Auto-reset: replace state with fresh reset state when done ---
        _, reset_state = self.reset(k_reset, params)
        final_state = tu.tree_map(
            lambda n, r: jnp.where(done, r, n), new_state_raw, reset_state
        )
        final_obs = self._get_obs(final_state, params)

        costs = stack_costs(cost_sla, cost_overtemp, cost_power_deficit)
        info: Dict[str, Any] = {
            # Cost channels
            "cost_sla": cost_sla,
            "cost_overtemp": cost_overtemp,
            "cost_power_deficit": cost_power_deficit,
            "cost_sum": total_cost,
            # Power balance
            "p_load_mw": p_load_mw,
            "p_pv_mw": p_pv_mw,
            "p_dg_mw": p_dg_mw,
            "p_batt_mw": p_batt_mw,
            "p_grid_import_mw": p_grid_import_mw,
            "power_deficit": power_deficit,
            "power_spill": power_spill,
            "residual": residual_after_grid,
            "raw_residual": residual,
            # Economics
            "fuel_cost": fuel_cost,
            "grid_cost": grid_cost,
            "grid_price_per_mwh": grid_price_per_mwh,
            "terminal_soc_cost": terminal_soc_cost,
            "carbon_kg": carbon_kg,
            "grid_carbon_kg": grid_carbon_kg,
            # Diagnostics
            "n_expired": n_expired,
            "p_dc_mw": p_dc_mw,
            # ``soc`` reports the post-step SOC BEFORE auto-reset, matching the
            # Python backend's behaviour where info reflects the step that just
            # ran (not the freshly reset state of the next episode).
            "soc": soc_after_step,
            # Vector reward
            "reward_vector": jnp.stack([r_energy, r_cost, r_carbon]),
        }

        return final_obs, final_state, reward, costs, done, info

    # ------------------------------------------------------------------ obs

    def _get_obs(
        self, state: DCMicrogridState, params: DCMicrogridParams
    ) -> chex.Array:
        """24-D observation with load, storage headroom, and price signals."""
        dc = state.dc
        dc_p = params.dc
        n_gpus_f = jnp.float32(max(dc_p.n_gpus, 1))

        # DC obs
        cpu_util = jnp.clip(dc.gpus_active.astype(jnp.float32) / n_gpus_f, 0.0, 1.0)
        mem_util = jnp.clip(dc.gpus_infer.astype(jnp.float32) / n_gpus_f, 0.0, 1.0)

        is_waiting = dc.task_status == jnp.int32(_WAITING)
        is_wt_train = is_waiting & (dc.task_type == jnp.int32(_TRAIN))
        is_wt_ft = is_waiting & (dc.task_type == jnp.int32(_FINETUNE))
        train_demand = jnp.sum(jnp.where(is_wt_train, dc.task_gpus, jnp.int32(0)))
        ft_demand = jnp.sum(jnp.where(is_wt_ft, dc.task_gpus, jnp.int32(0)))
        q_train_fill = jnp.clip(train_demand.astype(jnp.float32) / n_gpus_f, 0.0, 1.0)
        q_ft_fill = jnp.clip(ft_demand.astype(jnp.float32) / n_gpus_f, 0.0, 1.0)

        slack_norm = (
            (dc.task_deadline - dc.time_step - dc.task_duration).astype(jnp.float32)
            / jnp.maximum(dc.task_duration.astype(jnp.float32), jnp.float32(1.0))
        )
        min_slack = jnp.min(jnp.where(is_waiting, slack_norm, jnp.float32(5.0)))
        queue_urgency = jnp.clip(min_slack / jnp.float32(5.0), -1.0, 1.0)

        zone_norm = (dc.t_zone - jnp.float32(_T_ZONE_MIN)) / jnp.float32(
            _T_ZONE_MAX - _T_ZONE_MIN
        )
        outdoor_norm = (dc.t_outdoor - jnp.float32(10.0)) / jnp.float32(20.0)

        cop_factor = jnp.clip(
            jnp.float32(1.0)
            - jnp.float32(dc_p.cop_decay)
            * jnp.maximum(dc.t_outdoor - jnp.float32(dc_p.t_ref), jnp.float32(0.0)),
            jnp.float32(0.4),
            jnp.float32(1.2),
        )
        cop_ratio = (cop_factor - jnp.float32(0.4)) / jnp.float32(0.8)

        # Resource obs (from bundle states)
        soc = _read_soc_from_state(state, params)
        solar_cf = _read_solar_cf_from_state(state, params)
        dg_margin_norm = _read_dg_margin_from_state(state, params)
        p_pv_mw = _read_pv_power_from_state(state, params)
        p_load_mw = jnp.maximum(-dc.current_p_mw, jnp.float32(0.0))
        batt_p_max = _read_battery_power_max_from_params(params)
        dg_p_max = _read_dg_power_max_from_params(params)
        pv_p_max = _read_pv_capacity_from_params(params)
        total_supply_scale = jnp.maximum(
            batt_p_max + dg_p_max + pv_p_max, jnp.float32(1e-6)
        )
        dispatchable_scale = jnp.maximum(
            batt_p_max + dg_p_max, jnp.float32(1e-6)
        )
        p_load_norm = jnp.clip(p_load_mw / total_supply_scale, 0.0, 1.0)
        net_load_norm = jnp.clip(
            (p_load_mw - p_pv_mw) / dispatchable_scale,
            -1.0,
            1.0,
        )
        batt_dis_headroom_norm = _read_battery_discharge_headroom_norm(state, params)
        batt_chg_headroom_norm = _read_battery_charge_headroom_norm(state, params)
        grid_price_norm = _read_grid_price_norm_from_state(state, params)
        grid_price_6h_max_norm = _read_grid_price_future_max_norm(state, params)

        t_sin, t_cos = time_features(dc.time_step, dc_p.steps_per_day)

        return jnp.stack([
            cpu_util,                              # 0
            mem_util,                              # 1
            q_train_fill,                          # 2
            q_ft_fill,                             # 3
            queue_urgency,                         # 4  ∈ [-1, 1]
            jnp.clip(zone_norm, 0.0, 1.0),        # 5
            jnp.clip(outdoor_norm, 0.0, 1.0),     # 6
            jnp.clip(cop_ratio, 0.0, 1.0),        # 7
            solar_cf,                              # 8  ∈ [0, 1]
            soc,                                   # 9
            jnp.clip(dg_margin_norm, 0.0, 1.0),  # 10
            p_load_norm,                          # 11
            net_load_norm,                        # 12  ∈ [-1, 1]
            batt_dis_headroom_norm,               # 13
            batt_chg_headroom_norm,               # 14
            grid_price_norm,                       # 15
            grid_price_6h_max_norm,                # 16
            state.last_action[0],                 # 17
            state.last_action[1],                 # 18
            state.last_action[2],                 # 19
            state.last_action[3],                 # 20  ∈ [-1, 1]
            state.last_action[4],                 # 21
            t_sin,                                # 22
            t_cos,                                # 23
        ])

    # ------------------------------------------------------------------ spaces

    def observation_space(self, params: DCMicrogridParams) -> Box:
        low = jnp.array(
            [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1, -1],
            dtype=jnp.float32,
        )
        high = jnp.ones(24, dtype=jnp.float32)
        return Box(low=low, high=high, shape=(24,), dtype=jnp.float32)

    def action_space(self, params: DCMicrogridParams) -> Box:
        low = jnp.array([0.0, 0.0, 0.0, -1.0, 0.0], dtype=jnp.float32)
        high = jnp.ones(5, dtype=jnp.float32)
        return Box(low=low, high=high, shape=(5,), dtype=jnp.float32)

    def constraint_names(self, params: DCMicrogridParams) -> tuple[str, ...]:
        return ("sla", "overtemp", "power_deficit")

    # ------------------------------------------------------------------ misc

    @property
    def name(self) -> str:
        return "DataCenterMicrogridEnv"

    def default_params(self) -> DCMicrogridParams:
        return make_dcmicrogrid_params()


# ---------------------------------------------------------------------------
# State-readers (Python-level isinstance lookups; trace-time static)
# ---------------------------------------------------------------------------

def _read_soc_from_state(
    state: DCMicrogridState, params: DCMicrogridParams
) -> chex.Array:
    """Battery SOC scalar (first device of the first BatteryBundle), 0.0 if none."""
    idx = _find_bundle_index(params.resources, BatteryBundle)
    if idx < 0:
        return jnp.float32(0.0)
    bs = state.resource_states[idx]
    return bs.soc[0]


def _read_solar_cf_from_state(
    state: DCMicrogridState, params: DCMicrogridParams
) -> chex.Array:
    """Solar CF scalar at the env's current time index, 0.0 if no PV bundle.

    Uses ``state.dc.time_step`` (post-step in info, post-reset in obs) to look
    up the bundle's profile, matching the previous SolarBundle obs semantics.
    """
    idx = _find_bundle_index(params.resources, RenewableBundle)
    if idx < 0:
        return jnp.float32(0.0)
    bundle = params.resources[idx]
    T = bundle.profiles.shape[0]
    cf_now = bundle.profiles[state.dc.time_step % T]  # shape (n_devices,)
    return cf_now[0]


def _read_dg_margin_from_state(
    state: DCMicrogridState, params: DCMicrogridParams
) -> chex.Array:
    """DG headroom = 1 - p_dg / p_max (first device of the first DieselBundle)."""
    idx = _find_bundle_index(params.resources, DieselBundle)
    if idx < 0:
        return jnp.float32(1.0)
    bundle = params.resources[idx]
    bs = state.resource_states[idx]
    safe_p = jnp.maximum(bundle.p_max[0], jnp.float32(1e-9))
    return jnp.float32(1.0) - bs.p_dg_mw[0] / safe_p


def _read_pv_power_from_state(
    state: DCMicrogridState, params: DCMicrogridParams
) -> chex.Array:
    """Current PV active power [MW] (first device), 0.0 if absent."""
    idx = _find_bundle_index(params.resources, RenewableBundle)
    if idx < 0:
        return jnp.float32(0.0)
    bs = state.resource_states[idx]
    return bs.p_mw[0]


def _read_battery_power_max_from_params(params: DCMicrogridParams) -> chex.Array:
    """Battery max charge/discharge power [MW] (first device), 0.0 if absent."""
    idx = _find_bundle_index(params.resources, BatteryBundle)
    if idx < 0:
        return jnp.float32(0.0)
    bundle = params.resources[idx]
    return jnp.asarray(bundle.power_max[0], dtype=jnp.float32)


def _read_battery_discharge_headroom_norm(
    state: DCMicrogridState, params: DCMicrogridParams
) -> chex.Array:
    """Battery discharge headroom normalised by converter power rating."""
    idx = _find_bundle_index(params.resources, BatteryBundle)
    if idx < 0:
        return jnp.float32(0.0)
    bundle = params.resources[idx]
    bs = state.resource_states[idx]
    safe_p = jnp.maximum(bundle.power_max[0], jnp.float32(1e-6))
    max_dis = jnp.clip(
        (bs.soc[0] - bundle.soc_min[0])
        * bundle.capacity[0]
        * bundle.eta_discharge[0]
        / jnp.float32(bundle.dt_hours),
        0.0,
        safe_p,
    )
    return max_dis / safe_p


def _read_battery_charge_headroom_norm(
    state: DCMicrogridState, params: DCMicrogridParams
) -> chex.Array:
    """Battery charge headroom normalised by converter power rating."""
    idx = _find_bundle_index(params.resources, BatteryBundle)
    if idx < 0:
        return jnp.float32(0.0)
    bundle = params.resources[idx]
    bs = state.resource_states[idx]
    safe_p = jnp.maximum(bundle.power_max[0], jnp.float32(1e-6))
    max_chg = jnp.clip(
        (bundle.soc_max[0] - bs.soc[0])
        * bundle.capacity[0]
        / (bundle.eta_charge[0] * jnp.float32(bundle.dt_hours)),
        0.0,
        safe_p,
    )
    return max_chg / safe_p


def _read_pv_capacity_from_params(params: DCMicrogridParams) -> chex.Array:
    """PV nameplate capacity [MW] (first device), 0.0 if absent."""
    idx = _find_bundle_index(params.resources, RenewableBundle)
    if idx < 0:
        return jnp.float32(0.0)
    bundle = params.resources[idx]
    return jnp.asarray(bundle.capacity_mw[0], dtype=jnp.float32)


def _read_dg_power_max_from_params(params: DCMicrogridParams) -> chex.Array:
    """DG nameplate capacity [MW] (first device), 0.0 if absent."""
    idx = _find_bundle_index(params.resources, DieselBundle)
    if idx < 0:
        return jnp.float32(0.0)
    bundle = params.resources[idx]
    return jnp.asarray(bundle.p_max[0], dtype=jnp.float32)


def _read_grid_price_from_time(t: chex.Array, params: DCMicrogridParams) -> chex.Array:
    """Current grid-import price [currency/MWh], 0.0 if no profile is configured."""
    if params.price_profile is None:
        return jnp.float32(0.0)
    T = params.price_profile.shape[0]
    return jnp.asarray(params.price_profile[t % T], dtype=jnp.float32)


def _read_grid_price_norm_from_state(
    state: DCMicrogridState, params: DCMicrogridParams
) -> chex.Array:
    price = _read_grid_price_from_time(state.dc.time_step, params)
    ref = jnp.float32(max(params.grid_price_ref_per_mwh, 1e-6))
    return jnp.clip(price / ref, 0.0, 1.0)


def _read_grid_price_future_max_norm(
    state: DCMicrogridState, params: DCMicrogridParams
) -> chex.Array:
    """Max grid price over the next 6 hours, normalized by the configured reference."""
    if params.price_profile is None:
        return jnp.float32(0.0)
    T = params.price_profile.shape[0]
    idx = (state.dc.time_step + jnp.arange(72, dtype=jnp.int32)) % T
    future_max = jnp.max(params.price_profile[idx])
    ref = jnp.float32(max(params.grid_price_ref_per_mwh, 1e-6))
    return jnp.clip(future_max / ref, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_dcmicrogrid_params(
    # DC overrides
    n_gpus: int = 1000,
    gpu_idle_w: float = 55.0,
    gpu_active_w: float = 580.0,
    p_base_mw: float = 0.4,
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
    # Time config
    delta_t_hours: float = 5.0 / 60.0,
    steps_per_day: int = 288,
    max_steps: int = 288,
    # Battery
    battery_capacity_mwh: float = 2.0,
    battery_power_mw: float = 0.5,
    battery_eta_charge: float = 0.95,
    battery_eta_discharge: float = 0.95,
    battery_soc_min: float = 0.1,
    battery_soc_max: float = 0.9,
    battery_soc_init: float = 0.5,
    battery_deg_cost_per_mwh: float = 5.0,
    # PV
    pv_p_max_mw: float = 0.4,
    solar_profile: Optional[chex.Array] = None,
    # Grid import / market price
    grid_import_p_max_mw: float = 0.0,
    grid_price_ref_per_mwh: float = 150.0,
    grid_carbon_kg_per_kwh: float = 0.18,
    terminal_soc_target: float = 0.5,
    terminal_soc_penalty: float = 0.0,
    price_profile: Optional[chex.Array] = None,
    # DG
    dg_p_max_mw: float = 0.6,
    dg_fuel_cost_per_mwh: float = 300.0,
    dg_emission_factor: float = 0.80,
    dg_p_min_norm: float = 0.0,
    # Reward weights
    w_cost: float = 0.5,
    w_carbon: float = 0.3,
    # Optional exogenous profiles
    cpu_profile: Optional[chex.Array] = None,
    outdoor_temp_profile: Optional[chex.Array] = None,
) -> DCMicrogridParams:
    """Build DCMicrogridParams with default ``(BatteryBundle, SolarBundle, DieselBundle)``.

    All numeric defaults preserved from the previous in-line implementation;
    the change is structural — physics now lives in the dedicated bundle
    modules and is composed via the ResourceBundle protocol.
    """
    from powerzoojax.envs.resource.datacenter import make_datacenter_params

    dc = make_datacenter_params(
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

    battery_bundle = make_battery_bundle(
        n_devices=1,
        power_mw=battery_power_mw,
        capacity_mwh=battery_capacity_mwh,
        soc_min=battery_soc_min,
        soc_max=battery_soc_max,
        eta_charge=battery_eta_charge,
        eta_discharge=battery_eta_discharge,
        initial_soc=battery_soc_init,
        dt_hours=delta_t_hours,
        cycle_cost_per_mwh=0.0,  # degradation handled at outer reward layer
    )
    # PV: zero-action, profile-driven (RenewableBundle's no-control mode).
    # Synthetic clip-sin curve when no profile is supplied, matching the
    # historical ``_solar_cf`` behaviour bit-for-bit.
    if solar_profile is None:
        T = int(steps_per_day)
        steps_arr = jnp.arange(T, dtype=jnp.float32)
        hour = steps_arr / jnp.float32(T) * jnp.float32(24.0)
        solar_profile_eff = jnp.clip(jnp.sin(jnp.pi * (hour - 6.0) / 12.0), 0.0, 1.0)
    else:
        solar_profile_eff = jnp.asarray(solar_profile, dtype=jnp.float32)
    solar_bundle = make_renewable_bundle(
        n_devices=1,
        capacity_mw=pv_p_max_mw,
        profiles=solar_profile_eff,
        max_steps=steps_per_day,
        enable_curtailment=False,
        enable_q_control=False,
        dt_hours=delta_t_hours,
    )
    diesel_bundle = make_diesel_bundle(
        n_devices=1,
        p_max_mw=dg_p_max_mw,
        fuel_cost_per_mwh=dg_fuel_cost_per_mwh,
        emission_factor=dg_emission_factor,
        p_min_norm=dg_p_min_norm,
        dt_hours=delta_t_hours,
    )

    return DCMicrogridParams(
        dc=dc,
        resources=(battery_bundle, solar_bundle, diesel_bundle),
        battery_deg_cost_per_mwh=battery_deg_cost_per_mwh,
        w_cost=w_cost,
        w_carbon=w_carbon,
        grid_import_p_max_mw=grid_import_p_max_mw,
        grid_price_ref_per_mwh=grid_price_ref_per_mwh,
        grid_carbon_kg_per_kwh=grid_carbon_kg_per_kwh,
        terminal_soc_target=terminal_soc_target,
        terminal_soc_penalty=terminal_soc_penalty,
        cpu_profile=cpu_profile,
        outdoor_temp_profile=outdoor_temp_profile,
        price_profile=None if price_profile is None else jnp.asarray(price_profile, dtype=jnp.float32),
    )


def make_dcmicrogrid_params_with_profiles(
    source: str = "google",
    episode_start_step: int = 0,
    data_dir: Optional[str] = None,
    manifest_dir: Optional[str] = None,
    strict: bool = False,
    require_real_data: bool = False,
    **kwargs,
) -> DCMicrogridParams:
    """Build DCMicrogridParams with exogenous workload / solar / temp profiles.

    Loads the DC workload profile (``datacenter.cpu_util``) from the specified
    data source via :mod:`powerzoojax.data.dc_microgrid_profiles`.  Solar uses
    the committed GB generation-by-type real trace, normalised to a capacity
    factor.  Grid-import price uses the committed GB MID market trace. Outdoor
    temperature remains a deterministic adapter because no weather manifest
    exists yet.

    Args:
        source: Workload source key — ``"google"``, ``"azure"``, or ``"alibaba"``.
        episode_start_step: Cyclic offset into the source profile.
        data_dir: Override for parquet directory.
        manifest_dir: Override for manifests directory.
        strict: Error handling for profile loading.
        require_real_data: If True, missing parquet data or parquet-engine
            imports raise instead of falling back to synthetic workload/solar.
        **kwargs: Forwarded to :func:`make_dcmicrogrid_params`.

    Returns:
        DCMicrogridParams with ``cpu_profile``, ``outdoor_temp_profile``
        populated and the SolarBundle's ``profile`` set from the loader.
    """
    from powerzoojax.data.dc_microgrid_profiles import load_workload_profiles

    episode_len = kwargs.get("max_steps", 288)
    profiles = load_workload_profiles(
        source=source,
        episode_len=episode_len,
        start_step=episode_start_step,
        data_dir=data_dir,
        manifest_dir=manifest_dir,
        strict=strict,
        require_real_data=require_real_data,
    )
    # Solar profile is now owned by SolarBundle: pass it through the factory.
    return make_dcmicrogrid_params(
        cpu_profile=profiles["cpu_profile"],
        outdoor_temp_profile=profiles["outdoor_temp_profile"],
        solar_profile=profiles["solar_profile"],
        price_profile=profiles["price_profile"],
        **kwargs,
    )


__all__ = [
    "DCMicrogridState",
    "DCMicrogridParams",
    "DataCenterMicrogridEnv",
    "make_dcmicrogrid_params",
    "make_dcmicrogrid_params_with_profiles",
    # Re-exports for backward compat
    "DieselParams",
    "compute_dg_power",
    "compute_dg_fuel_cost",
    "compute_dg_emissions",
    "_solar_cf",
]
