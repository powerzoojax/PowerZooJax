"""GenCos Market MARL — core pure-JAX functions.

A multi-agent bid-based electricity market where each generator is an
agent that submits a piecewise-linear offer curve and is paid at the
cleared LMP minus its true production cost. The temporal coupling is
ramp limits: each step's dispatch is bounded by the previous step's
dispatch ± ramp_up / ramp_down, making this a true rolling sequential
market rather than independent single-shot auctions.

This module exposes the *pure functional* core — ``market_marl_reset``
and ``market_marl_step`` — for use by trainers in ``powerzoojax.rl``
or other higher-level wrappers. There is no ``Environment``-subclass
wrapper here on purpose (envs/ must not depend on rl/).

Action (per agent, ``n_seg`` segments):
    a ∈ [−1, 1]^{n_seg}                     raw actions per agent
    m = (a + 1) / 2 ∈ [0, 1]                normalised markup magnitudes
    sorted_m = sort(m)                      enforce monotone offer curve
    offer = base · (1 + sorted_m · max_markup)
    The flat action vector concatenates all units' segment markups,
    shape ``(n_units · n_seg,)``.

Clearing (``offer_sced`` — exact PD-IPM in segment space):
    Solves the SCED LP with thermal limits and **runtime ramp bounds**
        p_min_rt = max(p_min, prev_dispatch − ramp_down)
        p_max_rt = min(p_max, prev_dispatch + ramp_up)
    enforced *inside* the LP, not by post-hoc clipping. LMPs are exact
    duals (no KKT-recovery approximation, unlike ``BidBasedMarketEnv``).

Reward (per-agent vector, shape ``(n_units,)``):
    dispatch_profit_i = LMP[node_i] · P_i · Δt  −  TC(P_i) · Δt
    TC(P) = (a/3) P³ + (b/2) P² + c P                    (true cost)
    Note: TC is the *true* production cost, not the offer-cleared cost.
    Strategic markup affects dispatch ordering and clearing price but
    each agent's TC is unchanged — that's what keeps the MARL signal
    meaningful.

Costs (CMDP, single channel):
    cost_thermal_overload (× ``cost_thermal_weight``) reports line
    overloads. It is *not* subtracted from reward — the env is mixed
    reward + CMDP, like the bid / cost market envs.

State fields beyond ``MarketState``:
    - ``last_dispatch_profit``  per-unit profit at last step, (n_units,).
    - ``lmp_history``           circular buffer of the last
                                ``lmp_history_len`` mean-LMP values,
                                oldest → newest. Used as a private
                                observation feature so each agent can
                                condition on recent price trends.
    - ``episode_start_idx``     sampled uniformly from ``[0, T)`` at
                                each reset; the actual profile row at
                                step t is ``(episode_start_idx + t) % T``.
                                Gives diverse temporal coverage across
                                training episodes when ``load_profiles``
                                is a large pool (e.g. a year of GB
                                half-hourly demand).

Resources:
    None. ``resources = ()`` is enforced — this env is pure
    generator-side bidding, no DERs.

Auto-reset is internal (with ``stop_gradient`` on the merged final
state); each auto-reset draws a fresh ``episode_start_idx``.

Do NOT import from ``powerzoojax.rl`` here (no training-framework
dependencies in ``envs/``).
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import chex
from flax import struct

from powerzoojax.case.case_data import CaseData
from powerzoojax.envs.base import EnvState, EnvParams
from powerzoojax.envs.market.base import MarketState, MarketParams
from powerzoojax.envs.market.offer_sced import (
    OfferSCEDSetup, prepare_offer_sced, offer_sced,
)
from powerzoojax.envs.grid.power_flow import safety_check


# ============ State / Params ============

@struct.dataclass
class MarketMARLState(MarketState):
    """Market MARL environment state.

    Extends MarketState with per-unit dispatch profit from the last step,
    a fixed-length circular buffer of mean-LMP history for observations,
    and an episode start index for sampling diverse 48-step windows from a
    large profile pool.

    episode_start_idx:
        Sampled uniformly from [0, T-1] at each reset (where T is the number
        of rows in load_profiles).  The actual profile row used at step t is
        (episode_start_idx + t) % T.  This gives diverse temporal coverage
        across training episodes when load_profiles is a large GB dataset pool.
    """
    last_dispatch_profit: chex.Array   # (n_units,) $ — dispatch profit at last step
    lmp_history: chex.Array            # (lmp_history_len,) $/MWh — mean LMP per step,
                                       #   oldest→newest.  Initialised to the reset LMP.
    episode_start_idx: chex.Array      # () int32 — profile row index for start of episode


@struct.dataclass
class MarketMARLParams(MarketParams):
    """Market MARL environment parameters.

    Extends MarketParams with GenCos-specific fields.
    """
    offer_sced_setup: OfferSCEDSetup = None   # exact solver setup; contains seg widths/prices
    unit_node_idx: chex.Array = None          # (n_units,) int32 — generator node indices
    ramp_up_mw_per_step: chex.Array = None    # (n_units,) MW — max ramp-up per step
    ramp_down_mw_per_step: chex.Array = None  # (n_units,) MW — max ramp-down per step
    max_markup: float = struct.field(pytree_node=False, default=2.0)
    cost_thermal_weight: float = struct.field(pytree_node=False, default=1.0)
    lmp_history_len: int = struct.field(pytree_node=False, default=4)


# ============ Factory ============

def make_market_marl_params(
    case: CaseData,
    load_profiles,
    *,
    n_segments: int = 3,
    max_markup: float = 2.0,
    max_steps: int = 48,
    delta_t_hours: float = 0.5,
    steps_per_day: int = 48,
    lmp_scale: float = 100.0,
    cost_thermal_weight: float = 1.0,
    lmp_history_len: int = 4,
    ramp_up_mw_per_step: Optional[np.ndarray] = None,
    ramp_down_mw_per_step: Optional[np.ndarray] = None,
    ramp_rate_fraction: float = 0.5,
) -> MarketMARLParams:
    """Build MarketMARLParams from a CaseData and load profile array.

    Args:
        case: CaseData instance (e.g. from create_case5()).
        load_profiles: (T, n_loads) array [MW] — absolute MW values.
        n_segments: Number of offer-curve segments per generator.
        max_markup: Maximum fractional markup (offer = base * (1 + m * max_markup)).
        max_steps: Episode length in steps.
        delta_t_hours: Time resolution [hours].
        steps_per_day: Steps per day (for sin/cos obs features).
        lmp_scale: LMP normalisation scale [$/MWh] for observations.
        cost_thermal_weight: Weight on thermal overload cost signal.
        lmp_history_len: Number of past mean-LMP values in the obs buffer.
        ramp_up_mw_per_step: (n_units,) MW — per-step ramp-up limit.
            If None, defaults to ramp_rate_fraction × p_max.
        ramp_down_mw_per_step: (n_units,) MW — per-step ramp-down limit.
            If None, defaults to ramp_rate_fraction × p_max.
        ramp_rate_fraction: Fraction of p_max used as default ramp rate when
            ramp_up/down_mw_per_step are not provided.  Default 0.5 (50% of
            p_max per step, i.e. 100% per hour at 30-min resolution).

    Returns:
        MarketMARLParams.
    """
    setup = prepare_offer_sced(case, n_segments=n_segments)

    p_max_np = np.asarray(case.unit_p_max, dtype=np.float32)  # (n_units,)
    if ramp_up_mw_per_step is None:
        ramp_up = (ramp_rate_fraction * p_max_np).astype(np.float32)
    else:
        ramp_up = np.asarray(ramp_up_mw_per_step, dtype=np.float32)
    if ramp_down_mw_per_step is None:
        ramp_down = (ramp_rate_fraction * p_max_np).astype(np.float32)
    else:
        ramp_down = np.asarray(ramp_down_mw_per_step, dtype=np.float32)

    return MarketMARLParams(
        # --- EnvParams ---
        max_steps=max_steps,
        delta_t_hours=delta_t_hours,
        # --- MarketParams ---
        PTDF=jnp.array(np.asarray(case.PTDF), dtype=jnp.float32),
        nodes_units_map=jnp.array(np.asarray(case.nodes_units_map), dtype=jnp.float32),
        nodes_loads_map=jnp.array(np.asarray(case.nodes_loads_map), dtype=jnp.float32),
        load_profiles=jnp.array(load_profiles, dtype=jnp.float32),
        line_cap=setup.line_cap,
        line_floor=setup.line_floor,
        unit_cost_a=jnp.array(np.asarray(case.unit_cost_a), dtype=jnp.float32),
        unit_cost_b=jnp.array(np.asarray(case.unit_cost_b), dtype=jnp.float32),
        unit_cost_c=jnp.array(np.asarray(case.unit_cost_c), dtype=jnp.float32),
        unit_p_min=jnp.array(np.asarray(case.unit_p_min), dtype=jnp.float32),
        unit_p_max=jnp.array(np.asarray(case.unit_p_max), dtype=jnp.float32),
        lmp_scale=lmp_scale,
        steps_per_day=steps_per_day,
        n_nodes=int(case.n_nodes),
        n_units=int(case.n_units),
        n_lines=int(case.n_lines),
        n_loads=int(case.n_loads),
        slack_bus_idx=int(case.slack_bus_idx),
        resources=(),
        # --- MarketMARLParams ---
        offer_sced_setup=setup,
        unit_node_idx=jnp.array(np.asarray(case.unit_node_idx), dtype=jnp.int32),
        ramp_up_mw_per_step=jnp.array(ramp_up, dtype=jnp.float32),
        ramp_down_mw_per_step=jnp.array(ramp_down, dtype=jnp.float32),
        max_markup=max_markup,
        cost_thermal_weight=cost_thermal_weight,
        lmp_history_len=lmp_history_len,
    )


# ============ Reset / Step ============

def market_marl_reset(
    key: chex.PRNGKey,
    params: MarketMARLParams,
) -> MarketMARLState:
    """Reset the market MARL environment.

    Samples a random episode start index from [0, T-1] where T is the number
    of rows in load_profiles.  Each episode then accesses profile rows
    (episode_start_idx + t) % T for t=0..max_steps-1, giving diverse temporal
    windows across training episodes when load_profiles is a large pool (e.g.
    a full year of GB half-hourly demand data).

    Runs the exact SCED with truthful prices at the sampled start row to
    initialise the unit dispatch and LMP history.

    Args:
        key: PRNG key used to sample episode_start_idx.
        params: MarketMARLParams.

    Returns:
        MarketMARLState at t=0 with sampled start index, truthful dispatch
        and initialised LMP history.
    """
    key_start, _ = jax.random.split(key)
    T = params.load_profiles.shape[0]   # Python int (static)
    # Sample start uniformly from [0, T-1]; wrap-around at episode end is fine
    # because the pool is large (GB data) and the circular nature is acceptable.
    episode_start_idx = jax.random.randint(
        key_start, shape=(), minval=0, maxval=T, dtype=jnp.int32,
    )

    truthful_prices = params.offer_sced_setup.base_seg_prices
    load_mw = params.nodes_loads_map @ params.load_profiles[episode_start_idx]
    sced = offer_sced(params.offer_sced_setup, load_mw, truthful_prices)

    init_mean_lmp = jnp.mean(sced.lmp)
    lmp_history = jnp.full(
        (params.lmp_history_len,), init_mean_lmp, dtype=jnp.float32
    )

    return MarketMARLState(
        time_step=jnp.int32(0),
        done=jnp.bool_(False),
        lmp=sced.lmp,
        unit_power_mw=sced.unit_power,
        line_flow_mw=sced.line_flow,
        resource_states=(),
        last_dispatch_profit=jnp.zeros(params.n_units, dtype=jnp.float32),
        lmp_history=lmp_history,
        episode_start_idx=episode_start_idx,
    )


def market_marl_step(
    key: chex.PRNGKey,
    state: MarketMARLState,
    flat_actions: chex.Array,
    params: MarketMARLParams,
):
    """Single-step market MARL transition with ramp-bounded rolling market.

    Action mapping:
        flat_actions ∈ [-1, 1]^{n_units * n_seg}
        m = (a + 1) / 2 ∈ [0, 1]
        sorted_m = sort(m, axis=-1)   (enforce monotone offers)
        offer_prices = base_seg_prices * (1 + sorted_m * max_markup)

    Ramp coupling:
        p_min_rt[i] = max(p_min[i], prev_dispatch[i] - ramp_down[i])
        p_max_rt[i] = min(p_max[i], prev_dispatch[i] + ramp_up[i])
        offer_sced is called with p_min_rt / p_max_rt to enforce these bounds
        in the LP itself (solver-layer enforcement, not post-hoc clipping).

    Reward: dispatch_profit_i = LMP[node_i] * P_i * dt − TC(P_i) * dt
    where TC(P) = (a/3) P³ + (b/2) P² + c P.

    Auto-reset: when done, state is replaced with reset state (stop_gradient).

    Args:
        key: PRNG key (split for auto-reset).
        state: Current MarketMARLState.
        flat_actions: (n_units * n_seg,) flattened actions in [-1, 1].
        params: MarketMARLParams.

    Returns:
        (final_state, done, reward_vec, info)
        where reward_vec is (n_units,) dispatch profit per agent.
    """
    n_units = params.n_units
    n_seg = params.offer_sced_setup.n_segments
    f32 = jnp.float32

    # 0. Action clipping (in case upstream doesn't enforce Box)
    flat_actions = jnp.clip(flat_actions.astype(f32), -1.0, 1.0)

    # 1. Action → monotone offer prices
    m = (flat_actions.reshape(n_units, n_seg) + 1.0) / 2.0
    sorted_m = jnp.sort(m, axis=-1)
    offer_prices = (params.offer_sced_setup.base_seg_prices
                    * (1.0 + sorted_m * params.max_markup))

    # 2. Load at current timestep.
    # True profile row = (episode_start_idx + within-episode step) % T.
    T = params.load_profiles.shape[0]   # Python int, static
    t_idx = (state.episode_start_idx + state.time_step) % T
    load_mw = params.nodes_loads_map @ params.load_profiles[t_idx]

    # 3. Compute ramp-bounded runtime dispatch bounds
    prev_p = state.unit_power_mw                              # (n_units,)
    p_min_rt = jnp.maximum(params.unit_p_min,
                           prev_p - params.ramp_down_mw_per_step)
    p_max_rt = jnp.minimum(params.unit_p_max,
                           prev_p + params.ramp_up_mw_per_step)

    # 4. Exact SCED with runtime ramp bounds
    sced = offer_sced(
        params.offer_sced_setup, load_mw, offer_prices,
        p_min_rt=p_min_rt, p_max_rt=p_max_rt,
    )

    # 5. Safety check
    is_safe, n_violations, cost_thermal = safety_check(
        sced.line_flow,
        params.offer_sced_setup.line_cap,
        params.offer_sced_setup.line_floor,
    )

    # 6. Reward: dispatch profit per unit
    P = sced.unit_power
    TC_per_unit = (
        (params.unit_cost_a / 3.0) * P ** 3
        + (params.unit_cost_b / 2.0) * P ** 2
        + params.unit_cost_c * P
    )
    dispatch_profit = (
        sced.lmp[params.unit_node_idx] * P * params.delta_t_hours
        - TC_per_unit * params.delta_t_hours
    )

    # 7. Ramp binding rate: fraction of units at their ramp-constrained limit
    up_lim   = jnp.minimum(params.unit_p_max, prev_p + params.ramp_up_mw_per_step)
    down_lim = jnp.maximum(params.unit_p_min, prev_p - params.ramp_down_mw_per_step)
    ramp_binding = jnp.mean(
        (P >= up_lim - f32(0.5)) | (P <= down_lim + f32(0.5))
    )

    # 8. Update LMP history (circular buffer, oldest dropped, newest appended)
    new_mean_lmp = jnp.mean(sced.lmp)
    new_lmp_history = jnp.roll(state.lmp_history, -1).at[-1].set(new_mean_lmp)

    # 9. State transition — carry episode_start_idx forward unchanged within episode.
    new_time = state.time_step + 1
    done = new_time >= params.max_steps
    new_state = MarketMARLState(
        time_step=new_time,
        done=done,
        lmp=sced.lmp,
        unit_power_mw=P,
        line_flow_mw=sced.line_flow,
        resource_states=(),
        last_dispatch_profit=dispatch_profit,
        lmp_history=new_lmp_history,
        episode_start_idx=state.episode_start_idx,   # unchanged within episode
    )

    # 10. Auto-reset with stop_gradient.
    # k_reset is used by market_marl_reset to sample a NEW episode_start_idx,
    # so each auto-reset begins at a freshly sampled position in the pool.
    _, k_reset = jax.random.split(key)
    reset_state = market_marl_reset(k_reset, params)
    final_state = jax.tree_util.tree_map(
        lambda n, r: jnp.where(done, r, n), new_state, reset_state
    )
    final_state = jax.lax.stop_gradient(final_state)

    cost_thermal_weighted = cost_thermal * params.cost_thermal_weight
    gen_cost = jnp.sum(TC_per_unit)
    info = {
        "cost": cost_thermal_weighted,
        "offer_cost": sced.offer_cost,
        "lmp": sced.lmp,
        "lambda_balance": sced.lambda_balance,
        "unit_power": P,
        "segment_delta": sced.segment_delta,
        "gen_cost": gen_cost,
        "cost_thermal_overload": cost_thermal_weighted,
        "cost_sum": cost_thermal_weighted,
        "is_safe": is_safe,
        "n_violations": n_violations,
        "sced_converged": sced.converged,
        "sced_feasible": sced.is_feasible,
        "ramp_binding_rate": ramp_binding,
    }
    return final_state, done, dispatch_profit, info
