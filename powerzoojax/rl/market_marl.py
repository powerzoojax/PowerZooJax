"""GenCos Market MARL — JaxMARL-compatible wrapper.

MarketMARLEnv wraps the pure-functional market_marl_step/market_marl_reset
(from envs/market/market_marl_core.py) behind the MultiAgentEnvironment
interface.

Design notes:
    - 5 GenCo agents (one per generator in case5)
    - Each agent observes its own private 12-dimensional obs (see layout below)
    - Actions: Box([-1,1]^{n_seg}) per agent
    - Reward:  dispatch profit (LMP * P * dt - TC(P) * dt)
    - Episode length: 48 steps (configurable via params.max_steps)

Private obs layout (dims = 8 + lmp_history_len; benchmark default lmp_history_len=4 → 12 dims):

    [0]  own_cost_b_norm        base_seg_prices[i,0] / lmp_scale
    [1]  own_p_max_norm         pmax_i / max(pmax)
    [2]  own_last_dispatch_norm unit_power_mw[i] / pmax_i
    [3]  own_last_profit_norm   last_dispatch_profit[i] / (lmp_scale * pmax_i * dt)
    [4]  own_ramp_headroom_norm min(pmax_i - unit_power_mw[i], ramp_up_i) / pmax_i
    [5]  demand_forecast_norm   total_load at (t+1) / total_pmax  (one-step-ahead forecast)
    [6]  sin(2π * t / steps_per_day)
    [7]  cos(2π * t / steps_per_day)
    [8..]  lmp_history_norm     state.lmp_history / lmp_scale  (oldest → newest)

Import constraint:
    This file imports from:
        - powerzoojax.rl.multi_agent (base class)
        - powerzoojax.envs.market.market_marl_core
        - powerzoojax.envs.market.offer_sced (for planner baseline in welfare ratio)
    No other rl/ imports.
"""

from __future__ import annotations

from functools import partial
from typing import Dict, List, Tuple, Any

import jax
import jax.numpy as jnp
import chex

from powerzoojax.rl.multi_agent import MultiAgentEnvironment
from powerzoojax.envs.spaces import Box
from powerzoojax.envs.market.market_marl_core import (
    MarketMARLState,
    MarketMARLParams,
    market_marl_reset,
    market_marl_step,
)
from powerzoojax.envs.market.offer_sced import offer_sced as _offer_sced

# Fixed number of scalar features per agent (dims 0–7).
_OBS_SCALAR_DIM = 8
# Benchmark default for lmp_history_len (params.lmp_history_len).
# OBS_DIM = 12 documents the canonical benchmark value (lmp_history_len=4).
# Individual MarketMARLEnv instances derive their obs dimension from params
# via self._obs_dim = _OBS_SCALAR_DIM + params.lmp_history_len.
OBS_DIM = 12  # = _OBS_SCALAR_DIM + 4 (benchmark default)


class MarketMARLEnv(MultiAgentEnvironment):
    """GenCos bid-based market — multi-agent RL environment.

    Agents are generator companies (GenCos).  Each step:
        1. Each agent submits n_seg-dimensional bid (markup levels).
        2. Exact SCED clears the market with ramp-bounded dispatch bounds.
        3. Each agent receives dispatch profit as reward.

    Args:
        params: MarketMARLParams (from make_market_marl_params).

    Usage::

        from powerzoojax.case import create_case5
        from powerzoojax.envs.market.market_marl_core import make_market_marl_params
        from powerzoojax.rl.market_marl import MarketMARLEnv
        import jax.numpy as jnp

        case = create_case5()
        mid_load = (case.load_d_max + case.load_d_min) / 2.0
        profiles = jnp.tile(mid_load[None, :], (48, 1))
        params = make_market_marl_params(case, profiles, n_segments=3)
        env = MarketMARLEnv(params)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)
        obs2, state2, rew, dones, info = env.step(key, state, {n: jnp.zeros(3) for n in env.agent_names})
    """

    def __init__(self, params: MarketMARLParams):
        self._params = params
        self._n_seg = params.offer_sced_setup.n_segments
        self._n_units = params.n_units
        self._agent_names = [f"genco_{i}" for i in range(params.n_units)]
        # Obs dimension derived from params so lmp_history_len is truly honoured.
        self._lmp_hist_len = int(params.lmp_history_len)
        self._obs_dim = _OBS_SCALAR_DIM + self._lmp_hist_len

    @property
    def num_agents(self) -> int:
        return self._n_units

    @property
    def agent_names(self) -> List[str]:
        return self._agent_names

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self,
        key: chex.PRNGKey,
    ) -> Tuple[Dict[str, chex.Array], MarketMARLState]:
        state = market_marl_reset(key, self._params)
        return self._build_obs_dict(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: MarketMARLState,
        actions: Dict[str, chex.Array],
    ) -> Tuple[
        Dict[str, chex.Array],
        MarketMARLState,
        Dict[str, chex.Array],
        Dict[str, chex.Array],
        Dict[str, Any],
    ]:
        flat = jnp.concatenate(
            [actions[n] for n in self._agent_names], axis=-1
        )
        final_state, done, profit_vec, info = market_marl_step(
            key, state, flat, self._params
        )
        # Rewards
        rewards = {n: profit_vec[i] for i, n in enumerate(self._agent_names)}
        # Dones — use returned `done` flag (final_state.done may be False after auto-reset)
        dones = {n: done for n in self._agent_names}
        dones["__all__"] = done
        # Obs from final_state
        obs_dict = self._build_obs_dict(final_state)
        return obs_dict, final_state, rewards, dones, info

    def observation_space(self, agent: str = None) -> Box:
        """Private observation space per agent: shape (self._obs_dim,).

        Dimension = 8 scalar features + params.lmp_history_len LMP history values.
        For the benchmark default (lmp_history_len=4) this equals OBS_DIM=12.

        Layout: [own_cost_b, own_p_max, own_dispatch, own_profit,
                 own_ramp_headroom, demand_forecast_t+1, sin(t), cos(t),
                 lmp_hist[0..lmp_history_len-1]]
        """
        d = self._obs_dim
        return Box(
            low=jnp.full((d,), -jnp.inf, dtype=jnp.float32),
            high=jnp.full((d,), jnp.inf, dtype=jnp.float32),
            shape=(d,),
            dtype=jnp.float32,
        )

    def action_space(self, agent: str = None) -> Box:
        """n_seg-dimensional action space in [-1, 1] (same for all agents)."""
        d = self._n_seg
        return Box(
            low=jnp.full((d,), -1.0, dtype=jnp.float32),
            high=jnp.full((d,), 1.0, dtype=jnp.float32),
            shape=(d,),
            dtype=jnp.float32,
        )

    # ── Private helpers ────────────────────────────────────────────────────

    def _build_obs_dict(
        self,
        state: MarketMARLState,
    ) -> Dict[str, chex.Array]:
        """Build per-agent private observations from state.

        Obs layout (8 scalar dims + lmp_history_len LMP history dims):
            [0] own_cost_b_norm        — base_seg_prices[i,0] / lmp_scale
            [1] own_p_max_norm         — pmax_i / max(pmax)
            [2] own_last_dispatch_norm — unit_power_mw[i] / pmax_i
            [3] own_last_profit_norm   — last_profit[i] / (lmp_scale * pmax_i * dt)
            [4] own_ramp_headroom_norm — min(pmax_i - dispatch_i, ramp_up_i) / pmax_i
            [5] demand_forecast_norm   — total_load at (t+1) / total_pmax  (one-step-ahead)
                                         At episode boundary (t+1 wraps), the next profile
                                         row is used (circular).  Semantically this is a
                                         perfect one-step forecast, consistent with the
                                         design intent: agents see next period's demand.
            [6] sin(2π * t / steps_per_day)
            [7] cos(2π * t / steps_per_day)
            [8..] lmp_history_norm     — state.lmp_history / lmp_scale, oldest→newest
        """
        params = self._params
        lmp_scale = params.lmp_scale
        pmax = params.unit_p_max                             # (n_units,)
        pmax_norm = jnp.max(pmax)
        dt = params.delta_t_hours
        t = state.time_step.astype(jnp.float32)
        spd = params.steps_per_day

        total_pmax = jnp.sum(pmax)
        # One-step-ahead demand forecast (obs[5]).
        # True next profile row = (episode_start_idx + time_step + 1) % T.
        # This correctly accounts for the randomly sampled episode start index,
        # so the forecast reflects the actual next half-hour in the GB pool.
        T_profile = params.load_profiles.shape[0]
        next_t = (state.episode_start_idx + state.time_step + 1) % T_profile
        total_load_forecast = jnp.sum(
            params.nodes_loads_map @ params.load_profiles[next_t]
        )
        t_sin = jnp.sin(2.0 * jnp.pi * t / spd)
        t_cos = jnp.cos(2.0 * jnp.pi * t / spd)

        # LMP history (4 values, oldest→newest), normalised by lmp_scale
        lmp_hist_norm = state.lmp_history / lmp_scale       # (lmp_history_len,)

        obs_dict = {}
        for i, name in enumerate(self._agent_names):
            base_price = params.offer_sced_setup.base_seg_prices[i, 0]
            pmax_i = pmax[i]
            profit_scale = lmp_scale * pmax_i * dt + jnp.float32(1e-6)

            # Ramp headroom: how far this unit can still ramp up
            ramp_up_i = params.ramp_up_mw_per_step[i]
            disp_i = state.unit_power_mw[i]
            ramp_headroom = jnp.minimum(pmax_i - disp_i, ramp_up_i)
            ramp_headroom_norm = ramp_headroom / (pmax_i + jnp.float32(1e-6))

            obs_scalar = jnp.array([
                base_price / lmp_scale,                                     # [0]
                pmax_i / (pmax_norm + jnp.float32(1e-6)),                  # [1]
                disp_i / (pmax_i + jnp.float32(1e-6)),                     # [2]
                state.last_dispatch_profit[i] / profit_scale,              # [3]
                ramp_headroom_norm,                                         # [4]
                total_load_forecast / (total_pmax + jnp.float32(1e-6)),   # [5] t+1 forecast
                t_sin,                                                      # [6]
                t_cos,                                                      # [7]
            ], dtype=jnp.float32)                                           # (8,)

            obs_i = jnp.concatenate([obs_scalar, lmp_hist_norm])    # (12,)
            obs_dict[name] = obs_i
        return obs_dict

    # ── Market metrics hook (used by IPPO for GenCos-specific logging) ────

    def compute_market_metrics(
        self, rollout_market_info: Dict[str, chex.Array]
    ) -> Dict[str, chex.Array]:
        """Compute market-specific metrics from collected rollout info.

        Called by make_ippo_train after each training update when this env
        is detected.  rollout_market_info values have shapes:

            lmp          : (n_steps, n_envs, n_nodes)
            unit_power   : (n_steps, n_envs, n_units)
            ramp_binding : (n_steps, n_envs)   — fraction of binding units

        Implemented metrics (all keys prefixed ``"market/"``):
            price_volatility       — mean CV (std/mean) of per-env LMP time series
            HHI                    — mean Herfindahl-Hirschman Index across steps/envs
            market_share_std       — std of time-averaged dispatch shares across agents
                                     (summary statistic accompanying dispatch_share_{ag})
            ramp_binding_rate      — mean fraction of ramp-constrained units per step
            dispatch_share_{ag}    — per-agent time-averaged dispatch share (values ≈ 1/n
                                     for equal share; sum ≈ 1.0 across agents).
                                     Together these form the ``market_share_dynamics``
                                     time-series signal: one value per agent per training
                                     update, suitable for plotting dispatch evolution.
            planner_cost_ratio     — planner_cost / RL_gen_cost ∈ (0, 1].
                                     Planner = ramp-free DC-OPF with true costs
                                     (offer_sced with base_seg_prices, no ramp bounds).
                                     Measures dispatch cost efficiency relative to the
                                     cost-minimising planner.  A value close to 1.0
                                     means RL dispatch is nearly as cheap as the planner;
                                     lower values indicate higher RL generation cost.
                                     NOTE: this is a COST RATIO, not a welfare ratio.
                                     True social_welfare_ratio = (V − RL_cost) /
                                     (V − planner_cost) where V is consumer value
                                     (VOLL × demand).  V is not defined in this codebase,
                                     so a proper welfare ratio cannot be computed without
                                     an arbitrary consumer-side assumption.  This metric
                                     is therefore not a social-welfare ratio.
                                     Requires rollout_market_info to contain ``load_mw``
                                     (n_steps, n_envs, n_nodes) and ``gen_cost``
                                     (n_steps, n_envs); omitted silently if absent
                                     (backward-compatible with old IPPO versions).

        NOT implemented (explicit future work, G5 partial — not claimed as done):
            social_welfare_ratio   — Requires consumer value V (VOLL ×
                                     demand) for the ratio (V − RL_cost) / (V − planner_cost)
                                     to be well-defined and ∈ [0, 1].  V is absent from
                                     the current codebase; any choice would be arbitrary.
            exploitability_proxy   — requires a best-response oracle or self-play;
                                     a price-taking approximation would conflate dispatch
                                     suboptimality with strategic effect.
            collusion_indicator    — requires knowing the single-agent optimal markup,
                                     which is undefined without a Nash equilibrium solver.
            convergence_stability  — implemented separately in the IPPO training loop
                                     (ippo.py), NOT computable from rollout data alone;
                                     exported as ``market/convergence_stability``.
        """
        lmp = rollout_market_info["lmp"]           # (n_steps, n_envs, n_nodes)
        power = rollout_market_info["unit_power"]  # (n_steps, n_envs, n_units)

        # Price volatility: mean over envs of (std / mean) of mean_LMP over time
        mean_lmp_t = jnp.mean(lmp, axis=-1)        # (n_steps, n_envs)
        lmp_mean_env = jnp.mean(mean_lmp_t, axis=0) + jnp.float32(1e-8)  # (n_envs,)
        lmp_std_env  = jnp.std(mean_lmp_t, axis=0)                        # (n_envs,)
        price_volatility = jnp.mean(lmp_std_env / lmp_mean_env)           # scalar

        # HHI: mean Herfindahl-Hirschman Index across steps and envs
        total_p = jnp.sum(power, axis=-1, keepdims=True) + jnp.float32(1e-8)
        shares = power / total_p                   # (n_steps, n_envs, n_units)
        hhi = jnp.mean(jnp.sum(shares ** 2, axis=-1))                     # scalar

        # Market share std: std of time-averaged dispatch shares across agents
        mean_shares = jnp.mean(shares, axis=(0, 1))  # (n_units,)
        market_share_std = jnp.std(mean_shares)

        metrics: Dict[str, chex.Array] = {
            "market/price_volatility":  price_volatility,
            "market/HHI":               hhi,
            "market/market_share_std":  market_share_std,
        }

        if "ramp_binding_rate" in rollout_market_info:
            metrics["market/ramp_binding_rate"] = jnp.mean(
                rollout_market_info["ramp_binding_rate"]
            )

        # Per-agent time-averaged dispatch share (market_share_dynamics proxy).
        # mean_shares[i] is agent i's fraction of total dispatch, averaged over
        # all (step, env) pairs.  Values are in [0, 1] and sum ≈ 1.0 across agents.
        for i, ag in enumerate(self._agent_names):
            metrics[f"market/dispatch_share_{ag}"] = mean_shares[i]

        # Planner cost ratio: planner_cost / RL_gen_cost ∈ (0, 1].
        # Planner = ramp-free DC-OPF with true costs (offer_sced, base_seg_prices).
        # This is a COST RATIO, not a welfare ratio (see docstring above for why
        # a true welfare ratio requires consumer value V which is not defined here).
        # ratio = 1.0 when RL dispatch is as cheap as the planner; < 1.0 otherwise.
        # The planner minimises gen cost without ramp constraints, so planner_cost
        # ≤ RL_gen_cost always → ratio ∈ (0, 1].
        if "load_mw" in rollout_market_info and "gen_cost" in rollout_market_info:
            load_mw_all  = rollout_market_info["load_mw"]   # (n_steps, n_envs, n_nodes)
            gen_cost_rl  = rollout_market_info["gen_cost"]  # (n_steps, n_envs)
            n_steps_r, n_envs_r = gen_cost_rl.shape
            N_total   = n_steps_r * n_envs_r
            load_flat = load_mw_all.reshape(N_total, -1)     # (N, n_nodes)

            # Planner cost per (step, env): offer_sced with truthful prices,
            # no runtime ramp bounds.  Returns OfferSCEDResult.total_cost.
            # Subsampled to N_PLANNER_SUB entries via uniform stride to reduce
            # SCED calls from N_total×50 iters to N_sub×20 iters (~80× fewer).
            # Statistical accuracy is sufficient for monitoring trends.
            _N_PLANNER_SUB = 512
            N_sub  = jnp.minimum(jnp.int32(_N_PLANNER_SUB), jnp.int32(N_total))
            stride = N_total // _N_PLANNER_SUB
            indices = jnp.arange(_N_PLANNER_SUB, dtype=jnp.int32) * stride
            load_sub     = load_flat[indices]                # (_N_PLANNER_SUB, n_nodes)
            gen_cost_sub = gen_cost_rl.reshape(-1)[indices]  # (_N_PLANNER_SUB,)

            def _planner_cost(lmw):  # (n_nodes,) → scalar
                res = _offer_sced(self._params.offer_sced_setup, lmw)
                return res.total_cost

            planner_costs_sub = jax.vmap(_planner_cost)(load_sub)  # (_N_PLANNER_SUB,)

            cost_ratio = jnp.mean(
                planner_costs_sub / (gen_cost_sub + jnp.float32(1e-6))
            )
            metrics["market/planner_cost_ratio"] = cost_ratio

        return metrics
