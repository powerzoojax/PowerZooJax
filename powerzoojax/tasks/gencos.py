"""GenCos MARL market task — case5 + GB demand profiles.

Task-level factories and metrics for the five-agent competitive market
GenCos benchmark (case5 + GB demand profiles).

Main entry-points
-----------------
* ``load_gencos_profiles(case, split, ood_axis=None)`` — load GB demand profiles
  scaled to case5 load buses; raises ``FileNotFoundError`` when data unavailable.
* ``make_gencos_params(case, profiles, ...)`` — canonical factory returning
  ``MarketMARLParams`` with task preset defaults.
* ``make_gencos_env(case, profiles, ...)`` — full factory returning a
  ``MarketMARLEnv`` ready for rollout or IPPO training.
* ``compute_gencos_metrics(rollout, agent_names)`` — episode-level market metrics
  (total_profit, HHI, price_volatility, sced_convergence_rate, …).

Physical core (``MarketMARLState``, ``MarketMARLParams``, ``market_marl_step``)
lives in ``powerzoojax.envs.market.market_marl_core`` (framework-neutral).
The RL wrapper ``MarketMARLEnv`` lives in ``powerzoojax.rl.market_marl``.
This module is the task recipe that connects the two layers.

Splits
------
    train           2025-04-01 → 2025-12-31  (GB demand actuals)
    iid             2026-01-01 → 2026-03-31  (GB demand actuals)
    demand_shift    train window × 1.10      (OOD: high demand)
    renewable_shock train window × 1.05      (OOD: net-load shock proxy)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from powerzoojax.tasks.base import ConstraintSpec


# ============ Profile loading ============

def load_gencos_profiles(case, split: str = "train", ood_axis: Optional[str] = None):
    """Load GB demand profiles normalised and scaled to case5 load buses.

    Args:
        case:     CaseData for case5.
        split:    ``"train"`` (2025-04-01→2025-12-31) or
                  ``"iid"``  (2026-01-01→2026-03-31).
        ood_axis: ``None``, ``"demand_shift"`` (case-scaled load × 1.10), or
                  ``"renewable_shock"`` (net-load proxy +5% after case scaling).

    Returns:
        jnp.array of shape (T, n_loads) in MW.

    Raises:
        FileNotFoundError: if GB demand data is not available.
    """
    import jax.numpy as jnp
    from powerzoojax.data.data_loader import DataLoader
    from powerzoojax.data.signals import LOAD_ACTUAL_MW
    from powerzoojax.data.splits import (
        GB_TRAIN_START, GB_TRAIN_END,
        GB_IID_START, GB_IID_END,
    )

    if split == "train":
        start, end = GB_TRAIN_START, GB_TRAIN_END
    elif split == "iid":
        start, end = GB_IID_START, GB_IID_END
    else:
        raise ValueError(f"Unknown split '{split}'; use 'train' or 'iid'.")

    try:
        loader = DataLoader()
        arr = loader.load_jax_profiles(
            [LOAD_ACTUAL_MW],
            source="gb",
            start_date=start,
            end_date=end,
            resample="30min",
        )  # (T, 1) float32
        demand_mw = np.asarray(arr).ravel().astype(np.float32)  # (T,)
        if len(demand_mw) == 0:
            raise ValueError("Empty dataset: no rows returned for the requested window.")
    except Exception as exc:
        raise FileNotFoundError(
            f"GB demand data unavailable for split='{split}': {exc}\n"
            "Use synthetic flat profiles as fallback."
        ) from exc

    # Normalise demand to [0, 1] over the window, then affine-map per load bus.
    #   demand_norm = (d - d.min()) / (d.max() - d.min())
    #   load_profiles[t, i] = d_min[i] + demand_norm[t] * (d_max[i] - d_min[i])
    d_min = np.asarray(case.load_d_min, dtype=np.float32)  # (n_loads,)
    d_max = np.asarray(case.load_d_max, dtype=np.float32)  # (n_loads,)

    denom = float(demand_mw.max() - demand_mw.min()) + 1e-8
    demand_norm = (demand_mw - demand_mw.min()) / denom     # (T,) ∈ [0, 1]

    profiles = (
        d_min[None, :]
        + demand_norm[:, None] * (d_max - d_min)[None, :]
    )  # (T, n_loads)
    profiles = np.clip(profiles, d_min[None, :], d_max[None, :])

    # Apply OOD stress *after* case scaling so the split actually differs from IID.
    if ood_axis == "demand_shift":
        profiles = profiles * np.float32(1.10)
    elif ood_axis == "renewable_shock":
        # Proxy: net-load increase from wind collapse (+5% gross demand effect)
        profiles = profiles * np.float32(1.05)
    elif ood_axis is not None:
        raise ValueError(
            f"Unknown ood_axis='{ood_axis}'. "
            "Supported: 'demand_shift', 'renewable_shock'."
        )

    # GenCos is a dispatch-only task with fixed online units, so total demand
    # must stay between aggregate p_min and p_max. Without this projection,
    # very low-demand windows can underload the system (all units stuck at p_min,
    # SCED marked non-converged) and post-scaling OOD demand can overshoot total
    # generation capacity. Project the row totals back into the feasible band
    # while preserving each row's spatial load mix as much as possible.
    min_total_load = float(np.asarray(case.unit_p_min, dtype=np.float32).sum()) + 1.0
    max_total_load = float(np.asarray(case.unit_p_max, dtype=np.float32).sum()) - 1.0
    total_load = profiles.sum(axis=1, keepdims=True)
    base_weights = np.clip(d_max, 0.0, None)
    base_weight_sum = float(base_weights.sum())
    if base_weight_sum <= 0.0:
        raise ValueError("GenCos case has no positive load buses for profile projection.")
    base_weights = base_weights / base_weight_sum
    row_weights = np.where(
        total_load > 1e-6,
        profiles / (total_load + 1e-8),
        base_weights[None, :],
    )
    profiles = np.where(total_load < min_total_load, row_weights * min_total_load, profiles)
    total_load = profiles.sum(axis=1, keepdims=True)
    scale_down = np.minimum(1.0, max_total_load / (total_load + 1e-8))
    profiles = profiles * scale_down

    return jnp.array(profiles, dtype=jnp.float32)


# ============ Param / env factories ============

def make_gencos_params(
    case,
    load_profiles,
    n_segments: int = 3,
    max_markup: float = 2.0,
    max_steps: int = 48,
):
    """Build ``MarketMARLParams`` for the GenCos benchmark.

    Thin wrapper around :func:`~powerzoojax.envs.market.market_marl_core.make_market_marl_params`
    that names the task preset defaults explicitly.

    Args:
        case:          CaseData (typically ``create_case5()``).
        load_profiles: (T, n_loads) array of MW demand profiles.
        n_segments:    Cost-curve segments per generator (default 3).
        max_markup:    Maximum price markup multiplier (default 2.0).
        max_steps:     Episode length in half-hour steps (default 48 = 24 h).

    Returns:
        ``MarketMARLParams`` ready for ``market_marl_reset`` / ``market_marl_step``.
    """
    from powerzoojax.envs.market.market_marl_core import make_market_marl_params

    return make_market_marl_params(
        case,
        load_profiles,
        n_segments=n_segments,
        max_markup=max_markup,
        max_steps=max_steps,
    )


def make_gencos_env(
    case,
    load_profiles,
    n_segments: int = 3,
    max_markup: float = 2.0,
    max_steps: int = 48,
):
    """Build a ``MarketMARLEnv`` for the GenCos benchmark.

    Combines :func:`make_gencos_params` with the ``MarketMARLEnv`` RL wrapper.
    The env is ready for IPPO training or deterministic rollout.

    Returns:
        Tuple ``(env, params)``.
    """
    from powerzoojax.rl.market_marl import MarketMARLEnv

    params = make_gencos_params(
        case, load_profiles,
        n_segments=n_segments,
        max_markup=max_markup,
        max_steps=max_steps,
    )
    return MarketMARLEnv(params), params


# ============ Metrics ============

def compute_gencos_metrics(rollout: dict, agent_names: list) -> dict:
    """Aggregate a single-episode rollout into scalar market metrics.

    Args:
        rollout:     Dict with keys ``profits`` (dict name→(T,) array),
                     ``gen_cost`` (T,), ``lmp`` (T, n_nodes),
                     ``unit_power`` (T, n_units), ``sced_converged`` (T,),
                     ``ramp_binding_rate`` (T,).
        agent_names: Ordered list of agent name strings.

    Returns:
        Dict of scalar metrics: ``total_profit``, ``mean_profit_per_agent``,
        ``total_gen_cost``, ``mean_lmp``, ``price_volatility``, ``hhi``,
        ``sced_convergence_rate``, ``ramp_binding_rate``.
    """
    profits_per_agent = rollout["profits"]          # dict name→(T,)
    lmp = rollout["lmp"]                            # (T, n_nodes)
    unit_power = rollout["unit_power"]              # (T, n_units)

    total_profit = float(sum(
        profits_per_agent[n].sum() for n in agent_names
    ))
    mean_profit = total_profit / max(len(agent_names), 1)

    # Price volatility: CV of mean_LMP time series
    mean_lmp_t = lmp.mean(axis=-1)                 # (T,)
    lmp_mean = float(mean_lmp_t.mean()) + 1e-8
    price_volatility = float(mean_lmp_t.std()) / lmp_mean

    # HHI from dispatch shares
    total_p = unit_power.sum(axis=-1, keepdims=True) + 1e-8
    shares = unit_power / total_p
    hhi = float(np.mean(np.sum(shares ** 2, axis=-1)))

    return {
        "total_profit":          total_profit,
        "mean_profit_per_agent": mean_profit,
        "total_gen_cost":        float(rollout["gen_cost"].sum()),
        "mean_lmp":              float(lmp.mean()),
        "price_volatility":      price_volatility,
        "hhi":                   hhi,
        "sced_convergence_rate": float(rollout["sced_converged"].mean()),
        "ramp_binding_rate":     float(rollout["ramp_binding_rate"].mean()),
    }


# ============ TaskSpec implementation ============

def rollout_gencos(env_marl, key, policy_fn) -> dict:
    """Single-episode Python-loop MARL rollout for a GenCos policy.

    Args:
        env_marl: MarketMARLEnv (params embedded).
        key: JAX PRNGKey.
        policy_fn: Callable ``obs_dict -> actions_dict``.

    Returns:
        Dict with numpy arrays ``profits`` (dict name→(T,)), ``gen_cost`` (T,),
        ``lmp`` (T, n_nodes), ``unit_power`` (T, n_units),
        ``sced_converged`` (T,), ``ramp_binding_rate`` (T,).
    """
    import jax
    obs_dict, state = env_marl.reset(key)
    agent_names = env_marl.agent_names
    max_steps = int(env_marl._params.max_steps)

    profits_hist = {n: [] for n in agent_names}
    gen_cost_hist, lmp_hist, unit_power_hist = [], [], []
    sced_hist, ramp_hist = [], []

    for _ in range(max_steps):
        key, k_step = jax.random.split(key)
        actions = policy_fn(obs_dict)
        obs_dict, state, rewards_d, _, info = env_marl.step(k_step, state, actions)
        for n in agent_names:
            profits_hist[n].append(float(rewards_d[n]))
        gen_cost_hist.append(float(info.get("gen_cost", 0.0)))
        lmp_hist.append(np.asarray(info.get("lmp", np.zeros(1))))
        unit_power_hist.append(np.asarray(info.get("unit_power", np.zeros(1))))
        sced_hist.append(float(info.get("sced_converged", 1.0)))
        ramp_hist.append(float(info.get("ramp_binding_rate", 0.0)))

    return {
        "profits":         {n: np.array(profits_hist[n]) for n in agent_names},
        "gen_cost":        np.array(gen_cost_hist),
        "lmp":             np.stack(lmp_hist),
        "unit_power":      np.stack(unit_power_hist),
        "sced_converged":  np.array(sced_hist),
        "ramp_binding_rate": np.array(ramp_hist),
    }


def rollout_gencos_baseline(env_marl, key, strategy: str = "truthful") -> dict:
    """Non-learning baseline rollout for GenCos: fixed **bidding** strategies (no learning).

    Official benchmark names:

    ``"truthful"``    — all agents bid at true segment costs (zero markup).
    ``"uniform_mid"`` — all agents bid at the midpoint of the markup range.
    ``"max_markup"``  — all agents bid at the maximum allowed markup.

    Legacy aliases ``"zero_markup"`` and ``"uniform_markup"`` are accepted for
    older notebooks, but new benchmark records must use the official names above.
    """
    import jax
    import jax.numpy as jnp
    agent_names = env_marl.agent_names
    n_agents = len(agent_names)
    n_seg = int(env_marl._params.offer_sced_setup.n_segments)

    # market_marl_step maps action a ∈ [-1, 1] to markup fraction
    # m=(a+1)/2, then offer = base * (1 + m * max_markup).
    if strategy in ("truthful", "zero_markup"):
        action_value = -1.0
    elif strategy in ("uniform_mid", "uniform_markup"):
        action_value = 0.0
    elif strategy == "max_markup":
        action_value = 1.0
    else:
        raise ValueError(f"Unknown GenCos baseline: {strategy!r}")
    fixed_action = jnp.full((n_agents, n_seg), action_value, dtype=jnp.float32)

    def policy_fn(obs_dict):
        return {n: fixed_action[i] for i, n in enumerate(agent_names)}

    return rollout_gencos(env_marl, key, policy_fn)


class GencosTask:
    """TaskSpec for the GenCos benchmark (case5, 5 competing generator agents).

    ``MarketMARLEnv`` embeds params at construction time, so ``make_env(split)``
    returns a split-specific env.  ``episode_params()`` returns the embedded
    ``MarketMARLParams`` for use by callers that need param access.
    """

    task_name = "gencos"
    default_splits: tuple = ("train", "iid", "demand_shift", "renewable_shock")

    def __init__(
        self,
        *,
        n_segments: int = 3,
        max_markup: float = 2.0,
        max_steps: int = 48,
    ):
        self._n_segments = n_segments
        self._max_markup = max_markup
        self._max_steps = max_steps
        # _cache[split] = (MarketMARLEnv, MarketMARLParams)
        self._cache: dict = {}

    def _ensure_cache(self, split: str):
        if split in self._cache:
            return
        ood_axis = None
        if split == "demand_shift":
            ood_axis = "demand_shift"
        elif split == "renewable_shock":
            ood_axis = "renewable_shock"

        from powerzoojax.case import load_case
        case = load_case("5")
        load_role = "iid" if split in ("demand_shift", "renewable_shock") else split
        profiles = load_gencos_profiles(case, split=load_role, ood_axis=ood_axis)
        env_marl, params = make_gencos_env(
            case, profiles,
            n_segments=self._n_segments,
            max_markup=self._max_markup,
            max_steps=self._max_steps,
        )
        self._cache[split] = (env_marl, params)

    def make_env(self, split: str = "train"):
        self._ensure_cache(split)
        return self._cache[split][0]

    def episode_params(
        self,
        split: str,
        episode_idx: int,
        n_episodes: int,
        max_steps: int,
        *,
        strategy: str = "uniform",
        seed: int = 0,
    ):
        self._ensure_cache(split)
        return self._cache[split][1]

    def rollout(self, env, params, key, policy_fn) -> dict:
        return rollout_gencos(env, key, policy_fn)

    def baseline_rollout(self, env, params, key, baseline_name: str) -> dict:
        # "no_control" is the generic reference used by the shared eval_loop;
        # map to "truthful" (competitive bidding) as the GenCos analogue.
        if baseline_name == "no_control":
            baseline_name = "truthful"
        return rollout_gencos_baseline(env, key, strategy=baseline_name)

    def compute_metrics(self, agent_rollout, baseline_rollout) -> dict:
        agent_names = list(agent_rollout["profits"].keys())
        return compute_gencos_metrics(agent_rollout, agent_names)

    def baseline_names(self) -> tuple:
        return ("truthful", "uniform_mid", "max_markup")

    def constraint_spec(self) -> ConstraintSpec:
        return ConstraintSpec(
            selected_names=("thermal_overload",),
            thresholds=(0.0,),
            fallback_weights=(1.0,),
        )
