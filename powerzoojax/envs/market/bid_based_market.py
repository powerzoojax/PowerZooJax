"""Bid-Based LMP Market Environment — pure-JAX implementation.

A competitive electricity market on a DC-OPF transmission grid where the
agent operates one or more ``ResourceBundle`` instances (typically a
battery) as a price-aware market participant. Generators submit stepped
offer curves; an approximate piecewise economic dispatch clears the
market each step; the agent earns revenue at the cleared LMP. The
bundle's injection is netted out of node load before clearing, so its
actions *do* shift residual demand and the LMP — this is a single-agent
strategic-dispatch problem, not pure price-taking.

Generator offers (one episode):
    offer = base · (1 + |N(0, markup_std)|)    one-sided upward markup
                                                (no underbidding allowed)
    Monotonicity is enforced per generator across segments. Offers are
    generated once at reset and held fixed for the entire episode
    (markup is episode-level uncertainty, not step-level).

Clearing:
    ``clearing.piecewise_ed`` — approximate piecewise-linear ED, **not**
    an exact LP/QP SCED. May fail to converge on hard inputs;
    ``info["ed_converged"]`` exposes the flag.

LMP recovery (Market-Lite simplification):
    LMP is computed by ``clearing._compute_lmp_kkt_ed`` — a KKT-style
    fit on each generator's *true* marginal cost at the dispatched
    operating point, **not** on the cleared offer prices. When
    markup_std > 0, LMP and offer-clearing prices diverge; this gives a
    cleaner reward signal for RL while still letting markup affect the
    dispatch ordering.

Sign convention:
    unit_power_mw > 0  ⇒  generator injecting at its node.
    Bundle injections enter clearing as ``node_load − total_p_inject``.

Control problem (mixed reward + CMDP):
    Action      : concatenated bundle actions, ``(Σ b.action_dim,)``
                  in [−1, 1]. Default config = one battery scalar.
    Reward      : revenue = Σ_i LMP[bus_i] · P_i · Δt over bundle
                  devices. Non-zero by design (reward-driven, unlike
                  the pure-CMDP resource layer).
    Costs       : (thermal_overload,) weighted by ``cost_thermal_weight``.
    Observation : 5 base channels + Σ b.obs_dim, layout
                  [lmp_norm@first_bundle_bus, sin t, cos t,
                   total_demand_norm, mean_offer_price_norm,
                   <bundle_obs concatenated>]
    Termination : t ≥ max_steps (auto-resets, including a fresh draw
                  of episode offer prices).

Default resources: ``make_bid_market_params(case, resources=())``
attaches a single 50 MW / 200 MWh BatteryBundle at external bus id 1
(legacy convention).

Stays within the **Market Lite** scope: piecewise offers, KKT-style
LMP, no full unit commitment.
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
from powerzoojax.envs.grid.power_flow import safety_check, compute_generation_cost
from powerzoojax.envs.market.clearing import (
    PiecewiseEDSetup,
    prepare_piecewise_ed,
    piecewise_ed,
    make_cost_segments,
)


# ============ State / Params ============

@struct.dataclass
class BidMarketState(MarketState):
    """State for BidBasedMarketEnv.

    Extends MarketState with offer-based fields.
    """
    is_safe: chex.Array            # bool scalar
    n_violations: chex.Array       # int32 scalar
    total_gen_cost: chex.Array     # float32 — cumulative true cost
    offer_prices: chex.Array       # (n_units, n_segments) — current episode offers


@struct.dataclass
class BidMarketParams(MarketParams):
    """Parameters for BidBasedMarketEnv."""
    ed_setup: PiecewiseEDSetup = None
    cost_thermal_weight: float = struct.field(pytree_node=False, default=1.0)
    n_segments: int = struct.field(pytree_node=False, default=5)
    markup_std: float = struct.field(pytree_node=False, default=0.05)


# ============ Factory ============

def make_bid_market_params(
    case: CaseData,
    load_profiles: chex.Array = None,
    resources: tuple = (),
    lmp_scale: float = 100.0,
    max_steps: int = 48,
    delta_t_hours: float = 0.5,
    steps_per_day: int = 48,
    cost_thermal_weight: float = 1.0,
    n_segments: int = 5,
    markup_std: float = 0.05,
) -> BidMarketParams:
    """Create BidMarketParams from a CaseData.

    Args:
        case: CaseData instance.
        load_profiles: (T, n_loads) load demand [MW]. If None, flat mid-range.
        resources: ResourceBundle tuple. If empty, default BatteryBundle at
            external bus id 1 (legacy defaults).
        lmp_scale: LMP normalisation factor.
        max_steps: Episode length.
        delta_t_hours: Time step duration [hours].
        steps_per_day: Steps per day.
        cost_thermal_weight: Weight for thermal overload cost.
        n_segments: Number of offer curve segments per generator.
        markup_std: Std dev of random markup (0 = truthful bidding).

    Returns:
        BidMarketParams.
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

    ed_setup = prepare_piecewise_ed(case, n_segments)

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

    return BidMarketParams(
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
        ed_setup=ed_setup,
        cost_thermal_weight=cost_thermal_weight,
        n_segments=n_segments,
        markup_std=markup_std,
    )


def _get_node_load(params: BidMarketParams, load_demand: chex.Array) -> chex.Array:
    """Map per-load demand to per-node demand."""
    return params.nodes_loads_map @ load_demand


def _generate_offer_prices(
    key: chex.PRNGKey,
    base_prices: chex.Array,
    markup_std: float,
) -> chex.Array:
    """Generate episode offer prices: base × (1 + |N(0, markup_std)|).

    Args:
        key: PRNG key for random markup.
        base_prices: (n_units, n_segments) base offer prices.
        markup_std: Standard deviation of multiplicative markup.

    Note: markup is one-sided (upward only, using ``abs``).  Generators
    cannot underbid below true cost in this model.  Offers are fixed for
    the entire episode (generated once at ``reset``).

    Returns:
        offer_prices: (n_units, n_segments) with random markup applied.
    """
    noise = jax.random.normal(key, shape=base_prices.shape, dtype=base_prices.dtype)
    markup = 1.0 + jnp.abs(noise * markup_std)
    offer_prices = base_prices * markup

    # Enforce monotonicity per generator using scan over segments
    def _mono_step(prev_price, cur_price):
        fixed = jnp.maximum(cur_price, prev_price + 0.01)
        return fixed, fixed

    def _mono_row(row):
        first = row[0:1]
        _, fixed_rest = jax.lax.scan(_mono_step, row[0], row[1:])
        return jnp.concatenate([first, fixed_rest])

    offer_prices = jax.vmap(_mono_row)(offer_prices)
    return offer_prices


# ============ Environment ============

class BidBasedMarketEnv(Environment):
    """Competitive electricity market with piecewise-linear offer curves.

    Battery agent participates in a market where generators submit offers.
    Clearing uses ``piecewise_ed`` (approximate piecewise ED, not exact SCED).
    LMP is MC-based KKT recovery at dispatch point; not a strict offer-price dual.

    Action: concatenated bundle actions.
    Observation: [lmp_norm, sin, cos, demand_norm, mean_offer_norm, <bundle_obs>].
    Reward: sum_i LMP × P_i × Δt (revenue-based).
    Cost: thermal overload (CMDP channel).
    """

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self,
        key: chex.PRNGKey,
        params: BidMarketParams,
    ) -> Tuple[chex.Array, BidMarketState]:
        # Generate offer prices for this episode
        key, k_offers = jax.random.split(key)
        offer_prices = _generate_offer_prices(
            k_offers, params.ed_setup.base_seg_prices, params.markup_std)

        # Initial load
        load_demand = params.load_profiles[0]
        node_load_mw = _get_node_load(params, load_demand)

        # Run piecewise ED clearing
        ed_result = piecewise_ed(params.ed_setup, node_load_mw, offer_prices)

        resource_states = tuple(b.reset(key) for b in params.resources)

        state = BidMarketState(
            time_step=jnp.int32(0),
            done=jnp.bool_(False),
            lmp=ed_result.lmp,
            unit_power_mw=ed_result.unit_power,
            line_flow_mw=ed_result.line_flow,
            resource_states=resource_states,
            is_safe=jnp.bool_(True),
            n_violations=jnp.int32(0),
            total_gen_cost=jnp.float32(0.0),
            offer_prices=offer_prices,
        )
        obs = self._get_obs(state, params)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: BidMarketState,
        action: chex.Array,
        params: BidMarketParams,
    ) -> Tuple[chex.Array, BidMarketState, chex.Array, chex.Array, chex.Array, Dict[str, Any]]:
        """Step the bid-based market environment."""
        action_float = jnp.asarray(action, dtype=jnp.float32).reshape(-1)

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
            p_des = jnp.clip(a_slice, -1.0, 1.0) * bundle.power_max
            requested_sum = requested_sum + jnp.sum(p_des)
        new_resource_states = tuple(new_resource_states_list)

        t_idx = state.time_step % params.load_profiles.shape[0]
        load_demand = params.load_profiles[t_idx]
        node_load_mw = _get_node_load(params, load_demand)

        node_load_mw_mod = node_load_mw - total_p_inject

        ed_result = piecewise_ed(
            params.ed_setup, node_load_mw_mod, state.offer_prices)
        # Note: ed_result.converged is not checked here; piecewise_ed is heuristic and
        # may not converge for all inputs.  Use ed_result.converged in analysis scripts.

        reward = jnp.float32(0.0)
        for i, bundle in enumerate(params.resources):
            p_inj = p_inj_per_bundle[i]
            lmp_per_device = ed_result.lmp[bundle.bus_idx]
            reward = reward + jnp.sum(lmp_per_device * p_inj * params.delta_t_hours)

        is_safe, n_violations, cost_thermal = safety_check(
            ed_result.line_flow, params.line_cap, params.line_floor)

        gen_cost = compute_generation_cost(
            ed_result.unit_power,
            params.unit_cost_a, params.unit_cost_b, params.unit_cost_c)

        first_bus = params.resources[0].bus_idx[0]
        bus_lmp = ed_result.lmp[first_bus]

        new_time = state.time_step + 1
        done = new_time >= params.max_steps

        new_state = BidMarketState(
            time_step=new_time,
            done=done,
            lmp=ed_result.lmp,
            unit_power_mw=ed_result.unit_power,
            line_flow_mw=ed_result.line_flow,
            resource_states=new_resource_states,
            is_safe=is_safe,
            n_violations=n_violations,
            total_gen_cost=state.total_gen_cost + gen_cost,
            offer_prices=state.offer_prices,
        )

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
            "offer_cost": ed_result.offer_cost,
            "true_cost": ed_result.total_cost,
            "cost_model": jnp.float32(1.0),  # marker for piecewise
            "ed_converged": ed_result.converged,
        }

        return final_obs, final_state, reward, costs, done, info

    # ====== Observation ======

    def _get_obs(self, state: BidMarketState, params: BidMarketParams) -> chex.Array:
        """Build [lmp_norm, sin, cos, demand_norm, mean_offer_norm, <bundle_obs>]."""
        first_bus = params.resources[0].bus_idx[0]
        bus_lmp = state.lmp[first_bus]
        lmp_norm = bus_lmp / jnp.maximum(params.lmp_scale, 1e-6)

        t_sin, t_cos = time_features(state.time_step, params.steps_per_day)

        t_idx = state.time_step % params.load_profiles.shape[0]
        load_demand = params.load_profiles[t_idx]
        total_demand = jnp.sum(_get_node_load(params, load_demand))
        p_max_sum = jnp.maximum(jnp.sum(params.unit_p_max), 1.0)
        demand_norm = total_demand / p_max_sum

        w = params.ed_setup.base_seg_widths
        p = state.offer_prices
        total_w = jnp.maximum(jnp.sum(w), 1e-6)
        mean_offer = jnp.sum(p * w) / total_w
        mean_offer_norm = mean_offer / jnp.maximum(params.lmp_scale, 1e-6)

        ctx: Dict[str, Any] = {}
        parts = [
            jnp.array([lmp_norm], dtype=jnp.float32),
            jnp.stack([t_sin, t_cos]),
            jnp.array([demand_norm], dtype=jnp.float32),
            jnp.array([mean_offer_norm], dtype=jnp.float32),
        ]
        for i, bundle in enumerate(params.resources):
            parts.append(bundle.observe(state.resource_states[i], ctx))
        return jnp.concatenate(parts)

    # ====== Spaces ======

    def observation_space(self, params: BidMarketParams) -> Box:
        n_base = 5
        n_b = sum(b.obs_dim for b in params.resources)
        obs_dim = n_base + n_b
        low_base = jnp.array([-5.0, -1.0, -1.0, 0.0, 0.0], dtype=jnp.float32)
        high_base = jnp.array([5.0, 1.0, 1.0, 2.0, 5.0], dtype=jnp.float32)
        low = jnp.concatenate([low_base, jnp.full((n_b,), -jnp.inf, dtype=jnp.float32)])
        high = jnp.concatenate([high_base, jnp.full((n_b,), jnp.inf, dtype=jnp.float32)])
        return Box(
            low=low,
            high=high,
            shape=(obs_dim,),
            dtype=jnp.float32,
        )

    def action_space(self, params: BidMarketParams) -> Box:
        act_dim = sum(b.action_dim for b in params.resources)
        return Box(
            low=jnp.full((act_dim,), -1.0, dtype=jnp.float32),
            high=jnp.full((act_dim,), 1.0, dtype=jnp.float32),
            shape=(act_dim,),
            dtype=jnp.float32,
        )

    def default_params(self) -> BidMarketParams:
        """Cannot provide sensible defaults without a CaseData.

        Use ``make_bid_market_params(case)`` instead.
        """
        raise NotImplementedError(
            "BidBasedMarketEnv requires a CaseData. "
            "Use make_bid_market_params(case) to create params."
        )

    def constraint_names(self, params: BidMarketParams) -> tuple[str, ...]:
        return ("thermal_overload",)
