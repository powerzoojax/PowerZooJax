"""Cost-Based LMP Market Environment — pure-JAX implementation.

A cost-based DC-OPF market on a transmission grid where the agent
operates one or more ``ResourceBundle`` instances (typically a battery)
and earns revenue at the cleared LMP. Unlike ``BidBasedMarketEnv``,
generators do not bid — the clearing minimises *true* generator cost
each step. This makes the env well-suited for studying battery arbitrage
under cost-reflective nodal pricing, decoupled from strategic-bidding
effects.

Like the bid-based env, the bundle's injection enters clearing as
``node_load − total_p_inject``, so the agent's actions shift residual
demand and (in principle) move the LMP — strategic dispatch, not pure
price-taking.

Clearing & LMP:
    Dispatch comes from ``grid/dc_opf.dc_opf`` — an ADMM solver on the
    DC-OPF problem with thermal limits, aligned with PowerZoo's HiGHS LP
    semantics. The solver is heuristic and may leave residuals under
    severe congestion; ``info["opf_converged"]`` exposes the flag.
    Nodal LMP is recovered post-convergence by ``_compute_lmp_alm`` —
    line-limit shadow prices feed
        LMP = λ · 1 − PTDFᵀ · net_mu
    with λ recovered from the balance rows using the linear cost
    intercept ``mc_c``. **Not** ``clearing._compute_lmp_kkt_ed``.

Sign convention:
    unit_power_mw > 0  ⇒  generator injecting at its node.
    Bundle injections enter clearing as ``node_load − total_p_inject``.

Control problem (mixed reward + CMDP):
    Action      : concatenated bundle actions, ``(Σ b.action_dim,)`` in
                  [−1, 1]. Default config = one battery scalar.
    Reward      : revenue = Σ_i LMP[bus_i] · P_i · Δt over bundle
                  devices. Non-zero by design (reward-driven, unlike
                  the pure-CMDP resource layer).
    Costs       : (thermal_overload,) weighted by ``cost_thermal_weight``.
    Observation : 4 base channels + Σ b.obs_dim, layout
                  [lmp_norm@first_bundle_bus, sin t, cos t,
                   total_demand_norm, <bundle_obs concatenated>]
    Termination : t ≥ max_steps (auto-resets).

Default resources: ``make_cost_market_params(case, resources=())``
attaches a single 50 MW / 200 MWh BatteryBundle at external bus id 1
(legacy convention).

Determinism: clearing is fully deterministic given (load profile, bundle
injections) — no offer markup randomisation (contrast with
``BidBasedMarketEnv``, where offers carry per-episode noise). Episode
randomness comes only from bundle internals
(e.g. ``randomize_initial_soc``).
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

from powerzoojax.envs.base import Environment, stack_costs, time_features
from powerzoojax.envs.spaces import Box
from powerzoojax.case.case_data import CaseData
from powerzoojax.envs.market.base import MarketState, MarketParams
from powerzoojax.envs.resource.battery import make_battery_bundle
from powerzoojax.envs.grid.dc_opf import DCOPFSetup, prepare_dcopf, dc_opf
from powerzoojax.envs.grid.power_flow import safety_check, compute_generation_cost


# ============ State / Params ============

@struct.dataclass
class CostMarketState(MarketState):
    """State for CostBasedMarketEnv."""
    is_safe: chex.Array           # bool scalar
    n_violations: chex.Array      # int32 scalar
    total_gen_cost: chex.Array    # float32 — cumulative generation cost


@struct.dataclass
class CostMarketParams(MarketParams):
    """Parameters for CostBasedMarketEnv."""
    dcopf_setup: DCOPFSetup = None
    cost_thermal_weight: float = struct.field(pytree_node=False, default=1.0)


# ============ Factory ============

def make_cost_market_params(
    case: CaseData,
    load_profiles: chex.Array = None,
    resources: tuple = (),
    lmp_scale: float = 100.0,
    max_steps: int = 48,
    delta_t_hours: float = 0.5,
    steps_per_day: int = 48,
    cost_thermal_weight: float = 1.0,
) -> CostMarketParams:
    """Create CostMarketParams from a CaseData.

    Args:
        case: CaseData instance (must have PTDF, cost coefficients, etc.).
        load_profiles: (T, n_loads) load demand array [MW].
            If None, flat mid-range profile is generated.
        resources: Tuple of ResourceBundle instances (e.g. BatteryBundle).
            If empty, a default single-device battery at external bus id 1
            is created via ``make_battery_bundle`` (same defaults as the
            legacy flat battery fields).
        lmp_scale: LMP normalisation factor for observations.
        max_steps: Episode length.
        delta_t_hours: Time step duration [hours].
        steps_per_day: Steps per day (for time encoding).
        cost_thermal_weight: Weight for thermal overload cost.

    Returns:
        CostMarketParams.
    """
    if load_profiles is None:
        # Default: flat profile at mid-range demand — suitable for debugging only.
        # For realistic arbitrage training, pass time-varying load_profiles.
        mid_load = (case.load_d_max + case.load_d_min) / 2.0
        load_profiles = jnp.tile(mid_load[None, :], (max_steps, 1))

    line_cap = np.asarray(case.line_cap).copy()
    line_floor = np.asarray(case.line_floor).copy()
    NO_LIMIT = 1e5
    line_cap[line_cap == 0] = NO_LIMIT
    line_floor[line_floor == 0] = -NO_LIMIT

    dcopf_setup = prepare_dcopf(case)

    if len(resources) == 0:
        resources = (
            make_battery_bundle(
                case,
                bus_ids=[1],
                power_mw=50.0,
                capacity_mwh=200.0,
                eta_charge=0.95,
                eta_discharge=0.95,
                soc_min=0.1,
                soc_max=0.9,
                initial_soc=0.5,
                dt_hours=delta_t_hours,
            ),
        )
    else:
        resources = tuple(resources)

    # Static sanity checks (CPU-time only, does not affect JAX tracing)
    n_nodes_case = case.n_nodes
    for _b in resources:
        for _idx in np.asarray(_b.bus_idx).tolist():
            if not (0 <= _idx < n_nodes_case):
                raise ValueError(
                    f"ResourceBundle bus_idx={_idx} out of range "
                    f"[0, {n_nodes_case}) for case with {n_nodes_case} nodes."
                )

    return CostMarketParams(
        PTDF=jnp.array(case.PTDF, dtype=jnp.float32),
        nodes_units_map=jnp.array(case.nodes_units_map, dtype=jnp.float32),
        line_cap=jnp.array(line_cap, dtype=jnp.float32),
        line_floor=jnp.array(line_floor, dtype=jnp.float32),
        load_profiles=load_profiles,
        nodes_loads_map=jnp.array(case.nodes_loads_map, dtype=jnp.float32),
        unit_cost_a=jnp.array(case.unit_cost_a, dtype=jnp.float32),
        unit_cost_b=jnp.array(case.unit_cost_b, dtype=jnp.float32),
        unit_cost_c=jnp.array(case.unit_cost_c, dtype=jnp.float32),
        unit_p_min=jnp.array(case.unit_p_min, dtype=jnp.float32),
        unit_p_max=jnp.array(case.unit_p_max, dtype=jnp.float32),
        resources=resources,
        lmp_scale=lmp_scale,
        max_steps=max_steps,
        delta_t_hours=delta_t_hours,
        steps_per_day=steps_per_day,
        n_nodes=case.n_nodes,
        n_units=case.n_units,
        n_lines=case.n_lines,
        n_loads=case.n_loads,
        slack_bus_idx=int(case.slack_bus_idx),
        dcopf_setup=dcopf_setup,
        cost_thermal_weight=cost_thermal_weight,
    )


def _get_node_load(params: CostMarketParams, load_demand: chex.Array) -> chex.Array:
    """Map per-load demand to per-node demand."""
    return params.nodes_loads_map @ load_demand


# ============ Environment ============

class CostBasedMarketEnv(Environment):
    """Cost-based LMP arbitrage environment for battery storage.

    The agent controls batteries (via attached bundles) on the grid.
    Generator dispatch comes from ``grid/dc_opf.dc_opf`` (ADMM).  Nodal LMP
    follows ``dc_opf._compute_lmp_alm`` — not ``clearing._compute_lmp_kkt_ed``.

    Action: concatenated bundle actions (see each ResourceBundle).
    Observation: [lmp_norm, time_sin, time_cos, demand_norm, <bundle_obs>].
    Reward: sum_i LMP_{bus_i} × P_i × Δt (revenue-based).
    Cost: thermal overload (CMDP channel).
    """

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self,
        key: chex.PRNGKey,
        params: CostMarketParams,
    ) -> Tuple[chex.Array, CostMarketState]:
        # Initial load
        load_demand = params.load_profiles[0]
        node_load_mw = _get_node_load(params, load_demand)

        # Run DC-OPF for initial dispatch and LMP
        opf_result = dc_opf(params.dcopf_setup, node_load_mw)

        resource_states = tuple(b.reset(key) for b in params.resources)

        state = CostMarketState(
            time_step=jnp.int32(0),
            done=jnp.bool_(False),
            lmp=opf_result.lmp,
            unit_power_mw=opf_result.unit_power,
            line_flow_mw=opf_result.line_flow,
            resource_states=resource_states,
            is_safe=jnp.bool_(True),
            n_violations=jnp.int32(0),
            total_gen_cost=jnp.float32(0.0),
        )
        obs = self._get_obs(state, params)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: CostMarketState,
        action: chex.Array,
        params: CostMarketParams,
    ) -> Tuple[chex.Array, CostMarketState, chex.Array, chex.Array, chex.Array, Dict[str, Any]]:
        """Step the market environment.

        1. Run resource bundles (feasible P, SOC update).
        2. Apply net injections as load modification.
        3. Run DC-OPF → dispatch, flows, LMP.
        4. Reward = sum_i LMP[bus_i] × P_i × Δt.
        """
        action_float = jnp.asarray(action, dtype=jnp.float32).reshape(-1)

        # --- Bundles: physics before OPF (same pattern as DistGridEnv) ---
        ctx: Dict[str, Any] = {}
        new_resource_states_list = []
        p_inj_per_bundle = []
        total_p_inject = jnp.zeros(params.n_nodes, dtype=jnp.float32)
        requested_sum = jnp.float32(0.0)
        offset = 0
        for i, bundle in enumerate(params.resources):
            a_slice = action_float[offset: offset + bundle.action_dim]
            offset += bundle.action_dim
            new_bs, p_inj, _q_inj, _obs_sl, _cost_info = bundle.step(
                state.resource_states[i], a_slice, ctx)
            new_resource_states_list.append(new_bs)
            p_inj_per_bundle.append(p_inj)
            total_p_inject = total_p_inject.at[bundle.bus_idx].add(p_inj)
            # Requested grid-side P (pre-feasibility) for diagnostics
            p_des = jnp.clip(a_slice, -1.0, 1.0) * bundle.power_max
            requested_sum = requested_sum + jnp.sum(p_des)
        new_resource_states = tuple(new_resource_states_list)

        # Current load profile
        t_idx = state.time_step % params.load_profiles.shape[0]
        load_demand = params.load_profiles[t_idx]
        node_load_mw = _get_node_load(params, load_demand)

        # Battery injection: discharge reduces net nodal load
        node_load_mw_mod = node_load_mw - total_p_inject

        # Run DC-OPF for dispatch and LMP
        opf_result = dc_opf(params.dcopf_setup, node_load_mw_mod)
        # Note: opf_result.converged is not checked; dc_opf uses a heuristic penalty
        # solver that may leave residuals under congestion.  Verify with converged flag
        # in analysis or evaluation scripts.

        # Reward after OPF (LMP reflects cleared system given injection)
        reward = jnp.float32(0.0)
        for i, bundle in enumerate(params.resources):
            p_inj = p_inj_per_bundle[i]
            lmp_per_device = opf_result.lmp[bundle.bus_idx]
            reward = reward + jnp.sum(lmp_per_device * p_inj * params.delta_t_hours)

        # Safety check
        is_safe, n_violations, cost_thermal = safety_check(
            opf_result.line_flow, params.line_cap, params.line_floor)

        # Generation cost
        gen_cost = compute_generation_cost(
            opf_result.unit_power,
            params.unit_cost_a, params.unit_cost_b, params.unit_cost_c)

        # Diagnostics: first attached bus LMP (single-battery compat)
        first_bus = params.resources[0].bus_idx[0]
        bus_lmp = opf_result.lmp[first_bus]

        # Time step
        new_time = state.time_step + 1
        done = new_time >= params.max_steps

        new_state = CostMarketState(
            time_step=new_time,
            done=done,
            lmp=opf_result.lmp,
            unit_power_mw=opf_result.unit_power,
            line_flow_mw=opf_result.line_flow,
            resource_states=new_resource_states,
            is_safe=is_safe,
            n_violations=n_violations,
            total_gen_cost=state.total_gen_cost + gen_cost,
        )

        # Auto-reset
        _, k_reset = jax.random.split(key)
        _, reset_state = self.reset(k_reset, params)
        final_state = tu.tree_map(
            lambda n, r: jnp.where(done, r, n), new_state, reset_state)
        final_state = jax.lax.stop_gradient(final_state)
        final_obs = self._get_obs(final_state, params)

        cost_thermal_weighted = cost_thermal * params.cost_thermal_weight
        realized_total = jnp.sum(total_p_inject)
        costs = stack_costs(cost_thermal_weighted)
        info = {
            "gen_cost": gen_cost,
            "cost_thermal_overload": cost_thermal_weighted,
            "cost_sum": cost_thermal_weighted,
            "is_safe": is_safe,
            "n_violations": n_violations,
            "bus_lmp": bus_lmp,
            "realized_p_mw": realized_total,
            "requested_p_mw": requested_sum,
            "opf_converged": opf_result.converged,
        }

        return final_obs, final_state, reward, costs, done, info

    # ====== Observation ======

    def _get_obs(self, state: CostMarketState, params: CostMarketParams) -> chex.Array:
        """Build [lmp_norm, time_sin, time_cos, demand_norm, <bundle_obs>]."""
        first_bus = params.resources[0].bus_idx[0]
        bus_lmp = state.lmp[first_bus]
        lmp_norm = bus_lmp / jnp.maximum(params.lmp_scale, 1e-6)

        t_sin, t_cos = time_features(state.time_step, params.steps_per_day)

        t_idx = state.time_step % params.load_profiles.shape[0]
        load_demand = params.load_profiles[t_idx]
        total_demand = jnp.sum(_get_node_load(params, load_demand))
        p_max_sum = jnp.maximum(jnp.sum(params.unit_p_max), 1.0)
        demand_norm = total_demand / p_max_sum

        ctx: Dict[str, Any] = {}
        parts = [
            jnp.array([lmp_norm], dtype=jnp.float32),
            jnp.stack([t_sin, t_cos]),
            jnp.array([demand_norm], dtype=jnp.float32),
        ]
        for i, bundle in enumerate(params.resources):
            parts.append(bundle.observe(state.resource_states[i], ctx))
        return jnp.concatenate(parts)

    # ====== Spaces ======

    def observation_space(self, params: CostMarketParams) -> Box:
        n_base = 4
        n_b = sum(b.obs_dim for b in params.resources)
        obs_dim = n_base + n_b
        return Box(
            low=jnp.full((obs_dim,), -jnp.inf, dtype=jnp.float32),
            high=jnp.full((obs_dim,), jnp.inf, dtype=jnp.float32),
            shape=(obs_dim,),
            dtype=jnp.float32,
        )

    def action_space(self, params: CostMarketParams) -> Box:
        act_dim = sum(b.action_dim for b in params.resources)
        return Box(
            low=jnp.full((act_dim,), -1.0, dtype=jnp.float32),
            high=jnp.full((act_dim,), 1.0, dtype=jnp.float32),
            shape=(act_dim,),
            dtype=jnp.float32,
        )

    def default_params(self) -> CostMarketParams:
        """Cannot provide sensible defaults without a CaseData.

        Use ``make_cost_market_params(case)`` instead.
        """
        raise NotImplementedError(
            "CostBasedMarketEnv requires a CaseData. "
            "Use make_cost_market_params(case) to create params."
        )

    def constraint_names(self, params: CostMarketParams) -> tuple[str, ...]:
        return ("thermal_overload",)
