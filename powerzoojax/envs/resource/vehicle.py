"""Electric Vehicle (EV) resource — pure-JAX implementation.

A grid-connected EV that follows a fixed daily commute schedule, charges
or V2G-discharges only when at home, and consumes battery energy on
each trip. The control problem is planning-under-fixed-schedule:
the agent must build SOC up to ``soc_departure_min`` by the next
scheduled departure (optionally V2G-ing back to the grid in between)
while respecting SOC and inverter limits. While the EV is away the
action has no effect — feasible power is multiplied by ``is_home``
after clipping.

The schedule is exogenous, deterministic, and stored in ``VehicleParams``
as fixed-length arrays of (departure, arrival, energy) with up to
``MAX_TRIPS`` slots. Trips deduct energy from SOC at departure (floored
at ``soc_min``); arrivals only flip ``is_home`` back to 1.

Sign convention:
    current_p_mw > 0 ⇒ V2G discharge / inject to grid
    current_p_mw < 0 ⇒ G2V charge   / draw from grid

Action mapping (asymmetric, since p_charge_max ≠ p_discharge_max in
general):
    a ∈ [−1, 1], piecewise linear about 0:
        a ≥ 0 → P = a · p_discharge_max   (discharge / V2G)
        a < 0 → P = a · p_charge_max      (charge)
    P is then clipped to the SOC-feasible envelope (same form as
    Battery: ±(soc − bound)·C·η / dt, capped at rated power) and
    multiplied by ``is_home`` so an "away" command silently zeroes out.

Dynamics (memoryless apart from SOC, is_home, time_of_day):
    Step boundary : check trips first; departure deducts trip energy
                    and sets is_home = 0; arrival sets is_home = 1.
    SOC update    : Coulomb counting on the *feasible* power, identical
                    to Battery (Δsoc = −P·dt/(η_d·C) on discharge,
                                Δsoc = −P·dt·η_c/C  on charge),
                    defensively clipped to [soc_min, soc_max].
    time_of_day   : (time_of_day + dt) mod 24.

Control problem (CMDP):
    Action      : a ∈ [−1, 1] (1-D scalar).
    Reward      : 0 — define externally (energy cost, V2G revenue, …).
    Costs       : (departure_soc,) = max(0, soc_departure_min − soc),
                  evaluated *at the moment of departure*. This soft
                  penalty exists specifically because trip energy is
                  silently floored at ``soc_min`` — without the cost,
                  an RL agent learns to V2G to zero and rely on that
                  floor to top up the trip.
    Observation : 9-D
                  [soc, p_norm, is_home, departure_ready,
                   time_to_dep_norm, time_to_arr_norm,
                   sin t, cos t, soc_dep_min]
    Termination : t ≥ max_steps (auto-resets).

Single-device only — no SoA Bundle is provided. Multi-EV scenarios
should be built by the caller via vmap or a custom bundle.
"""

from functools import partial
from typing import Tuple, Dict, Any

import jax
import jax.numpy as jnp
import jax.tree_util as tu
import chex
from flax import struct

from powerzoojax.envs.base import Environment, denormalize_action, stack_costs
from powerzoojax.envs.spaces import Box
from powerzoojax.envs.resource.base import (
    ResourceState, ResourceParams, time_features,
)

MAX_TRIPS = 8  # Maximum number of daily commute trips


# State / Params
@struct.dataclass
class VehicleState(ResourceState):
    """Vehicle state.

    Attributes:
        soc: State of charge [0, 1].
        is_home: 1.0 if at home (available for charge/discharge), 0.0 if away.
        time_of_day: Current time in hours [0, 24).
        done: Episode termination flag.
    """
    soc: chex.Array          # float32 scalar [0, 1]
    is_home: chex.Array      # float32 scalar (0.0 or 1.0)
    time_of_day: chex.Array  # float32 scalar [0, 24)
    done: chex.Array         # bool scalar


@struct.dataclass
class VehicleParams(ResourceParams):
    """Vehicle parameters.

    Attributes:
        capacity_mwh: Battery capacity [MWh].
        soc_init: Initial SOC.
        soc_min / soc_max: SOC bounds.
        soc_departure_min: Required SOC at departure.
        p_charge_max_mw: Max charging power [MW].
        p_discharge_max_mw: Max V2G discharge power [MW].
        eta_charge / eta_discharge: One-way efficiencies.
        trip_departures: Departure times [hours], shape (MAX_TRIPS,). Unused = inf.
        trip_arrivals: Arrival times [hours], shape (MAX_TRIPS,). Unused = inf.
        trip_energies: Energy consumed per trip [MWh], shape (MAX_TRIPS,). Unused = 0.
        n_trips: Number of actual trips (int32).

    Scalar fields are ``pytree_node=False`` — static under JIT.
    Trip schedule arrays remain traced pytree leaves.
    """
    capacity_mwh: float = struct.field(pytree_node=False, default=0.06)
    soc_init: float = struct.field(pytree_node=False, default=0.8)
    soc_min: float = struct.field(pytree_node=False, default=0.1)
    soc_max: float = struct.field(pytree_node=False, default=0.95)
    soc_departure_min: float = struct.field(pytree_node=False, default=0.8)
    p_charge_max_mw: float = struct.field(pytree_node=False, default=0.007)
    p_discharge_max_mw: float = struct.field(pytree_node=False, default=0.007)
    eta_charge: float = struct.field(pytree_node=False, default=0.95)
    eta_discharge: float = struct.field(pytree_node=False, default=0.95)
    trip_departures: chex.Array = struct.field(default_factory=lambda: jnp.full((MAX_TRIPS,), jnp.inf, dtype=jnp.float32))
    trip_arrivals: chex.Array = struct.field(default_factory=lambda: jnp.full((MAX_TRIPS,), jnp.inf, dtype=jnp.float32))
    trip_energies: chex.Array = struct.field(default_factory=lambda: jnp.zeros((MAX_TRIPS,), dtype=jnp.float32))
    n_trips: int = struct.field(pytree_node=False, default=1)


# Pure Functions
def _is_time_in_window(time_of_day: chex.Array, target: chex.Array, window: chex.Array) -> chex.Array:
    """Check if time_of_day is in [target, target + window) with wrap at 24h."""
    end = (target + window) % 24.0

    # Normal case: start < end
    in_normal = (time_of_day >= target) & (time_of_day < end)

    # Wrap case: crosses midnight
    in_wrap = (time_of_day >= target) | (time_of_day < end)
    return jnp.where(target < end, in_normal, in_wrap)


def _update_availability(
    soc: chex.Array,
    is_home: chex.Array,
    time_of_day: chex.Array,
    params: VehicleParams,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Check departure/arrival for all trips, update is_home and soc.

    Scans trips in order. At most one event fires per step.

    Returns:
        new_soc: SOC after commute event (if any).
        new_home: Updated is_home flag.
        departure_soc_violation: ``max(0, soc_departure_min - soc)`` at the
            moment of departure. Zero when no departure occurs or SOC is
            sufficient.  Used as a cost signal to discourage RL from
            exploiting the silent clipping guard.
    """
    dt_hours = params.delta_t_hours

    def scan_trip(carry, trip_idx):
        soc_c, home_c, changed, violation = carry
        dep = params.trip_departures[trip_idx]
        arr = params.trip_arrivals[trip_idx]
        energy = params.trip_energies[trip_idx]
        valid = trip_idx < params.n_trips

        # Check departure: at home, departure time matches, not yet changed
        is_departing = valid & (home_c > 0.5) & _is_time_in_window(time_of_day, dep, dt_hours) & (~changed)

        # Record SOC shortfall before departure
        soc_gap = jnp.maximum(0.0, params.soc_departure_min - soc_c)
        violation = jnp.where(is_departing, violation + soc_gap, violation)

        # On departure: consume trip energy, leave home
        new_soc_dep = jnp.maximum(params.soc_min, soc_c - energy / params.capacity_mwh)
        soc_c = jnp.where(is_departing, new_soc_dep, soc_c)
        home_c = jnp.where(is_departing, 0.0, home_c)
        changed = changed | is_departing

        # Check arrival: away from home, arrival time matches, not yet changed
        is_arriving = valid & (home_c < 0.5) & _is_time_in_window(time_of_day, arr, dt_hours) & (~changed)
        home_c = jnp.where(is_arriving, 1.0, home_c)
        changed = changed | is_arriving

        return (soc_c, home_c, changed, violation), None

    init_carry = (soc, is_home, jnp.bool_(False), jnp.float32(0.0))
    (new_soc, new_home, _, violation), _ = jax.lax.scan(
        scan_trip, init_carry, jnp.arange(MAX_TRIPS)
    )
    return new_soc, new_home, violation


def _time_to_next_departure(time_of_day: chex.Array, params: VehicleParams) -> chex.Array:
    """Compute hours until next scheduled departure."""
    def scan_fn(best, trip_idx):
        dep = params.trip_departures[trip_idx]
        valid = trip_idx < params.n_trips
        diff = jnp.where(dep > time_of_day, dep - time_of_day, 24.0 - time_of_day + dep)
        diff = jnp.where(valid, diff, 24.0)
        best = jnp.minimum(best, diff)
        return best, None

    best, _ = jax.lax.scan(scan_fn, jnp.float32(24.0), jnp.arange(MAX_TRIPS))
    return best


def _time_to_next_arrival(time_of_day: chex.Array, is_home: chex.Array, params: VehicleParams) -> chex.Array:
    """Compute hours until next scheduled arrival.

    When the vehicle is away, returns time until the nearest future arrival.
    When at home, returns 0.0 (already arrived).
    """
    def scan_fn(best, trip_idx):
        arr = params.trip_arrivals[trip_idx]
        valid = trip_idx < params.n_trips
        diff = jnp.where(arr > time_of_day, arr - time_of_day, 24.0 - time_of_day + arr)
        diff = jnp.where(valid, diff, 24.0)
        best = jnp.minimum(best, diff)
        return best, None

    best, _ = jax.lax.scan(scan_fn, jnp.float32(24.0), jnp.arange(MAX_TRIPS))

    # 0.0 when at home
    return jnp.where(is_home > 0.5, 0.0, best)


# Environment Class
class VehicleEnv(Environment):
    """Electric Vehicle environment.

    Action: scalar in [-1, 1], denormalized internally to [-p_charge_max, p_discharge_max] MW.
        Positive = V2G discharge, negative = G2V charge.
        Uses piecewise linear mapping: 0 → 0 (idle), +1 → +p_discharge_max, -1 → -p_charge_max.
    Observation: 9-D [soc, p_norm, is_home, departure_ready,
                      time_to_dep_norm, time_to_arr_norm,
                      time_sin, time_cos, soc_dep_min].
    """

    # RL Interface Methods
    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: VehicleParams
    ) -> Tuple[chex.Array, VehicleState]:
        state = VehicleState(
            current_p_mw=jnp.float32(0.0),
            current_q_mvar=jnp.float32(0.0),
            time_step=jnp.int32(0),
            soc=jnp.float32(params.soc_init),
            is_home=jnp.float32(1.0),
            time_of_day=jnp.float32(0.0),
            done=jnp.bool_(False),
        )
        obs = self._get_obs(state, params)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: VehicleState,
        action: chex.Array,
        params: VehicleParams,
    ) -> Tuple[chex.Array, VehicleState, chex.Array, chex.Array, chex.Array, Dict[str, Any]]:

        # Piecewise linear denormalization: 0→0, +1→+p_discharge_max, -1→-p_charge_max
        action = jnp.float32(action).reshape(())
        action = jnp.clip(action, -1.0, 1.0)
        action = jnp.where(
            action >= 0,
            action * params.p_discharge_max_mw,
            action * params.p_charge_max_mw,
        )

        # 1. Check commute events at current time
        soc_after_commute, new_home, departure_soc_violation = _update_availability(
            state.soc, state.is_home, state.time_of_day, params
        )

        # 2. Apply charge/discharge action (only when at home)
        dt = params.delta_t_hours

        # SOC-limited discharge power
        max_discharge = jnp.clip(
            (soc_after_commute - params.soc_min) * params.capacity_mwh * params.eta_discharge / dt,
            0.0, params.p_discharge_max_mw)

        # SOC-limited charge power
        max_charge = jnp.clip(
            (params.soc_max - soc_after_commute) * params.capacity_mwh / (params.eta_charge * dt),
            0.0, params.p_charge_max_mw)

        # Clip action to feasible range
        feasible = jnp.clip(action, -max_charge, max_discharge)

        # Zero out power when not at home
        feasible = feasible * new_home

        # Update SOC
        delta_discharge = -feasible * dt / (params.eta_discharge * params.capacity_mwh)
        delta_charge = -feasible * dt * params.eta_charge / params.capacity_mwh
        delta_soc = jnp.where(feasible >= 0, delta_discharge, delta_charge)
        new_soc = jnp.clip(soc_after_commute + delta_soc, params.soc_min, params.soc_max)

        # 3. Advance time
        new_time_of_day = (state.time_of_day + dt) % 24.0
        new_time = state.time_step + 1
        done = new_time >= params.max_steps

        new_state = VehicleState(
            current_p_mw=feasible,
            current_q_mvar=jnp.float32(0.0),
            time_step=new_time,
            soc=new_soc,
            is_home=new_home,
            time_of_day=new_time_of_day,
            done=done,
        )

        # Auto-reset: when done, swap in fresh reset state
        _, k_reset = jax.random.split(key)
        _, reset_state = self.reset(k_reset, params)
        final_state = tu.tree_map(
            lambda n, r: jnp.where(done, r, n), new_state, reset_state)
        final_obs = self._get_obs(final_state, params)

        reward = jnp.float32(0.0)

        # Departure SOC violation: penalises leaving with SOC below
        # soc_departure_min.  Without this the agent can V2G-discharge
        # endlessly and rely on the silent soc_min clip at departure.
        costs = stack_costs(departure_soc_violation)
        info = {
            "cost_departure_soc": departure_soc_violation,
            "cost_sum": departure_soc_violation,
        }
        return final_obs, final_state, reward, costs, done, info

    # Spaces & Observation
    def _get_obs(self, state: VehicleState, params: VehicleParams) -> chex.Array:
        """9-D observation: [soc, p_norm, is_home, departure_ready,
        time_to_dep_norm, time_to_arr_norm, time_sin, time_cos, soc_dep_min]."""
        power_scale = jnp.maximum(params.p_charge_max_mw, params.p_discharge_max_mw)
        power_scale = jnp.maximum(power_scale, 1e-6)
        p_norm = jnp.clip(state.current_p_mw / power_scale, -1.0, 1.0)
        departure_ready = (state.soc >= params.soc_departure_min).astype(jnp.float32)

        time_to_dep = _time_to_next_departure(state.time_of_day, params)
        time_to_dep_norm = jnp.minimum(time_to_dep / 24.0, 1.0)

        time_to_arr = _time_to_next_arrival(state.time_of_day, state.is_home, params)
        time_to_arr_norm = jnp.minimum(time_to_arr / 24.0, 1.0)

        t_sin, t_cos = time_features(state.time_step, params.steps_per_day)
        return jnp.stack([
            state.soc, p_norm, state.is_home, departure_ready,
            time_to_dep_norm, time_to_arr_norm, t_sin, t_cos,
            params.soc_departure_min,
        ])

    def observation_space(self, params: VehicleParams) -> Box:
        low = jnp.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0], dtype=jnp.float32)
        high = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
        return Box(low=low, high=high, shape=(9,), dtype=jnp.float32)

    def action_space(self, params: VehicleParams) -> Box:
        return Box(
            low=jnp.array([-1.0], dtype=jnp.float32),
            high=jnp.array([1.0], dtype=jnp.float32),
            shape=(1,), dtype=jnp.float32,
        )

    # Status & Diagnostics
    @property
    def name(self) -> str:
        return "VehicleEnv"

    def default_params(self) -> VehicleParams:
        return make_vehicle_params()

    def constraint_names(self, params: VehicleParams) -> tuple[str, ...]:
        return ("departure_soc",)


def make_vehicle_params(
    E_max_kWh: float = 60.0,
    soc_init: float = 0.8,
    soc_min: float = 0.1,
    soc_max: float = 0.95,
    soc_departure_min: float = 0.8,
    p_charge_max_kW: float = 7.0,
    p_discharge_max_kW: float = 7.0,
    eta_charge: float = 0.95,
    eta_discharge: float = 0.95,
    commute_schedule: list = None,
    delta_t_minutes: float = 15.0,
    steps_per_day: int = None,
    max_steps: int = None,
) -> VehicleParams:
    """Create VehicleParams with kWh/kW inputs (PowerZoo-compatible).

    Args:
        commute_schedule: List of dicts with 'departure', 'arrival', 'energy_kWh'.
            Default: single trip 8am-6pm, 15kWh.
    """
    if commute_schedule is None:
        commute_schedule = [{'departure': 8.0, 'arrival': 18.0, 'energy_kWh': 15.0}]

    # Sort by departure
    commute_schedule = sorted(commute_schedule, key=lambda x: x['departure'])
    n_trips = len(commute_schedule)
    assert n_trips <= MAX_TRIPS, f"Max {MAX_TRIPS} trips supported, got {n_trips}"

    departures = jnp.full((MAX_TRIPS,), jnp.inf, dtype=jnp.float32)
    arrivals = jnp.full((MAX_TRIPS,), jnp.inf, dtype=jnp.float32)
    energies = jnp.zeros((MAX_TRIPS,), dtype=jnp.float32)

    dep_list = [float(t['departure']) for t in commute_schedule]
    arr_list = [float(t['arrival']) for t in commute_schedule]
    eng_list = [float(t['energy_kWh']) / 1000.0 for t in commute_schedule]  # kWh -> MWh

    departures = departures.at[:n_trips].set(jnp.array(dep_list, dtype=jnp.float32))
    arrivals = arrivals.at[:n_trips].set(jnp.array(arr_list, dtype=jnp.float32))
    energies = energies.at[:n_trips].set(jnp.array(eng_list, dtype=jnp.float32))

    dt_hours = delta_t_minutes / 60.0
    if steps_per_day is None:
        steps_per_day = int(24.0 / dt_hours)
    if max_steps is None:
        max_steps = steps_per_day  # default 1-day episode

    return VehicleParams(
        capacity_mwh=E_max_kWh / 1000.0,
        soc_init=soc_init,
        soc_min=soc_min,
        soc_max=soc_max,
        soc_departure_min=soc_departure_min,
        p_charge_max_mw=p_charge_max_kW / 1000.0,
        p_discharge_max_mw=p_discharge_max_kW / 1000.0,
        eta_charge=eta_charge,
        eta_discharge=eta_discharge,
        trip_departures=departures,
        trip_arrivals=arrivals,
        trip_energies=energies,
        n_trips=n_trips,
        delta_t_hours=dt_hours,
        steps_per_day=steps_per_day,
        max_steps=max_steps,
    )
