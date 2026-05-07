"""TSO environment factories and utilities for SCUC benchmark task.

Provides:
  - make_tso_net_load_profiles(): net-load construction from GB data or synthetic
  - make_tso_case118_params(): main case118 UCParams factory
  - make_tso_case14_params(): case14 UCParams with injected UC defaults
  - make_case14_with_uc_defaults(): add minimal UC metadata to case14 CaseData
  - tso_all_on_rollout(): "all units on" baseline
  - tso_merit_order_rollout(): merit-order (priority-list) rule-based baseline
  - compute_tso_metrics(): episode metrics aggregation

Net-load mapping (default must-take fractions 0.7 wind / 0.5 solar):
    net_load = gross_demand - wind_frac * wind_available - solar_frac * solar_available
    Per-series inputs are peak-normalised before composition; see ``make_tso_net_load_profiles``.

Data source: GB demand and generation-by-type parquet time series (train/iid date windows in ``data.splits``).
When real data arrays are omitted, synthetic intraday profiles are generated.
"""

from __future__ import annotations

from typing import Dict, Optional, Any

import numpy as np
import jax
import jax.numpy as jnp
import chex

from powerzoojax.case.case_data import CaseData
from powerzoojax.envs.grid.unit_commitment import (
    UCParams,
    UnitCommitmentEnv,
    make_uc_params,
)
from powerzoojax.tasks.base import ConstraintSpec


# ============ Net-load construction ============

_TSO_SYNTHETIC_LINE_LIMIT_DEGREES = 8.0
_TSO_UNLIMITED_CAP_THRESHOLD_MVA = 1e5


def _case118_needs_benchmark_line_caps(case: CaseData) -> bool:
    """Return True when case118 lacks executable thermal ratings."""
    line_cap = np.asarray(case.line_cap, dtype=np.float32)
    if case.line_rate_a is None:
        line_rate_a = np.zeros_like(line_cap)
    else:
        line_rate_a = np.asarray(case.line_rate_a, dtype=np.float32)
    return bool(
        np.max(line_rate_a) <= 1e-6
        and np.min(line_cap) >= _TSO_UNLIMITED_CAP_THRESHOLD_MVA
    )


def _with_tso_case118_benchmark_line_caps(case: CaseData) -> CaseData:
    """Inject finite benchmark line caps when case118 has no thermal ratings.

    Legacy ``raw_cases/case118.py`` exposed a per-line ``s`` column equal to
    ``baseMVA * deg2rad(1) / x``. We preserve that ordering but widen it to an
    8-degree transfer envelope so ``iid`` remains feasible for ``all_on``
    while the benchmark's frozen ``line_tightening`` split still creates a
    meaningful congestion shift.
    """
    if not _case118_needs_benchmark_line_caps(case):
        return case

    line_x = np.asarray(case.line_x, dtype=np.float32)
    theta_limit_rad = float(np.deg2rad(_TSO_SYNTHETIC_LINE_LIMIT_DEGREES))
    synth_cap = np.where(
        np.abs(line_x) > 1e-9,
        np.float32(float(case.base_mva) * theta_limit_rad) / np.abs(line_x),
        np.float32(1e6),
    ).astype(np.float32)
    synth_cap_jax = jnp.asarray(synth_cap, dtype=jnp.float32)
    return case.replace(
        line_cap=synth_cap_jax,
        line_floor=-synth_cap_jax,
        line_rate_a=synth_cap_jax,
    )


def make_tso_net_load_profiles(
    n_steps: int = 48,
    wind_frac: float = 0.7,
    solar_frac: float = 0.5,
    load_d_max: Optional[chex.Array] = None,
    gross_demand_mw: Optional[chex.Array] = None,
    wind_mw: Optional[chex.Array] = None,
    solar_mw: Optional[chex.Array] = None,
    delta_t_hours: float = 0.5,
) -> chex.Array:
    """Construct (n_steps, n_loads) net-load profiles.

    Net-load = gross_demand - wind_frac * wind - solar_frac * solar
    Then rescaled per bus by load_d_max.

    When real data arrays are not provided, generates synthetic intraday profiles:
      - gross_demand: sinusoidal daily curve (peak at hour 18)
      - wind: higher at night / lower midday
      - solar: bell curve around noon

    Args:
        n_steps: Number of time steps.
        wind_frac: Fraction of wind that is must-take (default 0.7).
        solar_frac: Fraction of solar that is must-take (default 0.5).
        load_d_max: (n_loads,) per-bus maximum demand [MW]. If None, returns
            normalised profiles in [0, 1].
        gross_demand_mw: (n_steps,) total system demand [MW]. None → synthetic.
        wind_mw: (n_steps,) wind available [MW]. None → synthetic.
        solar_mw: (n_steps,) solar available [MW]. None → synthetic.
        delta_t_hours: Step duration (used for synthetic profile phase).

    Returns:
        (n_steps, n_loads) float32 JAX array of load profiles [MW].
        If load_d_max is None, shape is (n_steps, 1) with values in [0, 1].
    """
    t = np.arange(n_steps, dtype=np.float32) * delta_t_hours  # hours

    if gross_demand_mw is None:
        # Synthetic: sinusoidal with peak at 18:00, min at 03:00
        phase = 2 * np.pi * (t - 3.0) / 24.0
        demand_norm = 0.75 + 0.25 * np.sin(phase)
        gross = demand_norm.astype(np.float32)
    else:
        g = np.asarray(gross_demand_mw, dtype=np.float32)[:n_steps]
        safe_max = np.maximum(g.max(), 1.0)
        gross = g / safe_max

    if wind_mw is None:
        wind_phase = 2 * np.pi * (t - 2.0) / 24.0
        wind_norm = 0.3 + 0.2 * np.sin(wind_phase + np.pi)  # higher at night
        wind = wind_norm.astype(np.float32)
    else:
        w = np.asarray(wind_mw, dtype=np.float32)[:n_steps]
        safe_max = np.maximum(w.max(), 1.0)
        wind = w / safe_max

    if solar_mw is None:
        # Bell curve: peak noon (12:00), zero at night
        solar_phase = 2 * np.pi * (t - 12.0) / 24.0
        solar_raw = np.cos(solar_phase) * (np.abs(t % 24.0 - 12.0) < 6.0)
        solar = np.maximum(solar_raw, 0.0).astype(np.float32) * 0.4
    else:
        s = np.asarray(solar_mw, dtype=np.float32)[:n_steps]
        safe_max = np.maximum(s.max(), 1.0)
        solar = s / safe_max

    # Net load (normalised, clamped to [0.05, 1.0])
    net_norm = np.clip(gross - wind_frac * wind - solar_frac * solar, 0.05, 1.0)

    if load_d_max is not None:
        ld = np.asarray(load_d_max, dtype=np.float32)  # (n_loads,)
        profiles = net_norm[:, None] * ld[None, :]      # (n_steps, n_loads)
    else:
        profiles = net_norm[:, None].astype(np.float32)  # (n_steps, 1)

    return jnp.array(profiles, dtype=jnp.float32)


def make_tso_net_load_profiles_from_data(
    data_loader,
    case: CaseData,
    split: str = "train",
    episode_start_idx: int = 0,
    n_steps: int = 48,
    wind_frac: float = 0.7,
    solar_frac: float = 0.5,
    allow_synthetic_fallback: bool = False,
) -> chex.Array:
    """Load net-load profiles from GB data via DataLoader.

    Uses the DataLoader JAX API (``load_jax_profiles``) with semantic signal
    names ``load.actual_mw``, ``wind.available_mw``, ``solar.available_mw``.

    Args:
        data_loader: DataLoader instance.
        case: CaseData for load_d_max scaling.
        split: "train" or "iid" (selects GB_TRAIN_* or GB_IID_* date range).
        episode_start_idx: Starting row index within the loaded window.
        n_steps: Number of steps to extract.
        wind_frac: Must-take wind fraction (default 0.7).
        solar_frac: Must-take solar fraction (default 0.5).
        allow_synthetic_fallback: If True, fall back to synthetic profiles when
            data loading fails instead of raising.  Default False — callers
            must opt in to the fallback explicitly so failures are not silent.

    Returns:
        (n_steps, n_loads) float32 JAX array.

    Raises:
        RuntimeError: If data loading fails and allow_synthetic_fallback=False.
    """
    from powerzoojax.data import signals as S
    from powerzoojax.data.splits import (
        GB_TRAIN_START, GB_TRAIN_END,
        GB_IID_START, GB_IID_END,
    )

    start = GB_TRAIN_START if split == "train" else GB_IID_START
    end = GB_TRAIN_END if split == "train" else GB_IID_END

    try:
        # (T_total, 3): columns = [load.actual_mw, wind.available_mw, solar.available_mw]
        raw = data_loader.load_jax_profiles(
            [S.LOAD_ACTUAL_MW, S.WIND_AVAILABLE_MW, S.SOLAR_AVAILABLE_MW],
            start_date=start,
            end_date=end,
            resample="30min",
        )
    except Exception as exc:
        if allow_synthetic_fallback:
            return make_tso_net_load_profiles(
                n_steps=n_steps,
                load_d_max=case.load_d_max,
                wind_frac=wind_frac,
                solar_frac=solar_frac,
            )
        raise RuntimeError(
            f"Failed to load GB profiles from DataLoader "
            f"(split={split!r}, start={start}, end={end}): {exc}"
        ) from exc

    t0 = episode_start_idx
    t1 = t0 + n_steps
    available = raw.shape[0]
    if available < t1:
        msg = (
            f"DataLoader returned {available} rows but episode_start_idx={t0} "
            f"+ n_steps={n_steps} = {t1} rows required "
            f"(split={split!r}, start={start}, end={end})."
        )
        if allow_synthetic_fallback:
            return make_tso_net_load_profiles(
                n_steps=n_steps,
                load_d_max=case.load_d_max,
                wind_frac=wind_frac,
                solar_frac=solar_frac,
            )
        raise RuntimeError(msg)

    raw_np = np.asarray(raw)
    demand_arr = raw_np[t0:t1, 0]   # load.actual_mw
    wind_arr   = raw_np[t0:t1, 1]   # wind.available_mw
    solar_arr  = raw_np[t0:t1, 2]   # solar.available_mw

    return make_tso_net_load_profiles(
        n_steps=n_steps,
        load_d_max=case.load_d_max,
        gross_demand_mw=demand_arr,
        wind_mw=wind_arr,
        solar_mw=solar_arr,
        wind_frac=wind_frac,
        solar_frac=solar_frac,
    )


# ============ Case14 UC metadata helper ============

def make_case14_with_uc_defaults(base_case: CaseData) -> CaseData:
    """Ensure UC columns exist on case14 (normally already set in ``create_case14()``).

    Legacy helper for tests that call ``load_case('14')`` on older snapshots.
    If ``unit_ramp_up`` is present, returns *base_case* unchanged.
    """
    if base_case.unit_ramp_up is not None:
        return base_case
    n = base_case.n_units
    return base_case.replace(
        unit_ramp_up=jnp.full((n,), 0.5, dtype=jnp.float32),
        unit_ramp_down=jnp.full((n,), 0.5, dtype=jnp.float32),
        unit_min_up_time=jnp.full((n,), 2, dtype=jnp.float32),
        unit_min_down_time=jnp.full((n,), 2, dtype=jnp.float32),
        unit_startup_cost=jnp.full((n,), 1000.0, dtype=jnp.float32),
        unit_no_load_cost=jnp.full((n,), 100.0, dtype=jnp.float32),
        unit_init_state=jnp.ones((n,), dtype=jnp.float32),
        unit_keep_time=jnp.full((n,), 10.0, dtype=jnp.float32),
        unit_init_power=jnp.array(base_case.unit_p_min, dtype=jnp.float32),
    )


# ============ Main case factories ============

def make_tso_case118_params(
    load_profiles: Optional[chex.Array] = None,
    max_steps: int = 48,
    delta_t_hours: float = 0.5,
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
    data_loader=None,
    split: str = "train",
) -> UCParams:
    """Create UCParams for the case118 TSO-SCUC main task.

    Uses IEEE 118-bus system with 54 generators and UC fields from case data.
    Load profiles default to synthetic net-load (intraday sinusoidal).
    Pass data_loader to use real GB demand / generation data.

    Args:
        load_profiles: Pre-built (T, n_loads) profiles [MW]. If None, generated
            from GB data (if data_loader given) or synthetic.
        max_steps: Episode length (48 steps = 24 hours at 30-min).
        delta_t_hours: Step duration in hours.
        reserve_margin_frac: Required reserve headroom fraction of load.
        enable_uc: True=SCUC mode, False=pure ED.
        enable_reserve: True=charge reserve shortfall as cost.
        forecast_horizon_steps: Number of future total-load steps appended to
            the observation.
        reward_scale: Multiplier for reward = -reward_scale * operating_cost.
        cost_thermal_weight: Weight on thermal overload in CMDP cost.
        solver_mode: 0=direct PF, 1=DC-OPF (recommended).
        dcopf_max_iter: Max ADMM iterations for the DC-OPF redispatch.
        dcopf_tol: ADMM convergence tolerance for the DC-OPF redispatch.
        sample_start_on_reset: Whether reset samples a fresh episode start
            uniformly from the bound load-profile pool.
        fixed_episode_start_idx: Deterministic start index used when sampling
            is disabled.
        data_loader: Optional DataLoader for GB demand / wind / solar data.
        split: "train" or "iid" split for data_loader.

    Returns:
        UCParams ready for use with UnitCommitmentEnv.
    """
    from powerzoojax.case import load_case
    case = _with_tso_case118_benchmark_line_caps(load_case("118"))

    if load_profiles is None:
        if data_loader is not None:
            load_profiles = make_tso_net_load_profiles_from_data(
                data_loader, case, split=split, n_steps=max_steps,
            )
        else:
            load_profiles = make_tso_net_load_profiles(
                n_steps=max_steps,
                load_d_max=case.load_d_max,
                delta_t_hours=delta_t_hours,
            )

    return make_uc_params(
        case=case,
        load_profiles=load_profiles,
        max_steps=max_steps,
        delta_t_hours=delta_t_hours,
        steps_per_day=int(round(24.0 / delta_t_hours)),
        reserve_margin_frac=reserve_margin_frac,
        enable_uc=enable_uc,
        enable_reserve=enable_reserve,
        forecast_horizon_steps=forecast_horizon_steps,
        reward_scale=reward_scale,
        cost_thermal_weight=cost_thermal_weight,
        solver_mode=solver_mode,
        dcopf_max_iter=dcopf_max_iter,
        dcopf_tol=dcopf_tol,
        sample_start_on_reset=sample_start_on_reset,
        fixed_episode_start_idx=fixed_episode_start_idx,
    )


def make_tso_case14_params(
    load_profiles: Optional[chex.Array] = None,
    max_steps: int = 48,
    delta_t_hours: float = 0.5,
    reserve_margin_frac: float = 0.05,
    enable_uc: bool = True,
    enable_reserve: bool = True,
    forecast_horizon_steps: int = 0,
    reward_scale: float = 1e-4,
    solver_mode: int = 1,
    dcopf_max_iter: int = 100,
    dcopf_tol: float = 1e-3,
) -> UCParams:
    """Create UCParams for the case14 small-scale sanity / debug task.

    IEEE 14-bus system with 5 generators.  UC fields come from ``create_case14()``
    (same table layout as case118).  Suitable for small-scale UC / ED tests.
    """
    from powerzoojax.case import load_case
    base_case = load_case("14")
    case = make_case14_with_uc_defaults(base_case)

    if load_profiles is None:
        load_profiles = make_tso_net_load_profiles(
            n_steps=max_steps,
            load_d_max=case.load_d_max,
            delta_t_hours=delta_t_hours,
        )

    return make_uc_params(
        case=case,
        load_profiles=load_profiles,
        max_steps=max_steps,
        delta_t_hours=delta_t_hours,
        steps_per_day=int(round(24.0 / delta_t_hours)),
        reserve_margin_frac=reserve_margin_frac,
        enable_uc=enable_uc,
        enable_reserve=enable_reserve,
        forecast_horizon_steps=forecast_horizon_steps,
        reward_scale=reward_scale,
        cost_thermal_weight=1.0,
        solver_mode=solver_mode,
        dcopf_max_iter=dcopf_max_iter,
        dcopf_tol=dcopf_tol,
    )


def make_tso_ed_params(
    case_id: str = "118",
    max_steps: int = 48,
    delta_t_hours: float = 0.5,
) -> UCParams:
    """Variant: TSO-ED — all units always ON, pure economic dispatch.

    No UC constraints, no startup/no_load cost, no reserve check.
    Equivalent to TransGridEnv in DCOPF mode but wrapped in UCParams.
    """
    if case_id == "118":
        return make_tso_case118_params(
            max_steps=max_steps,
            delta_t_hours=delta_t_hours,
            enable_uc=False,
            enable_reserve=False,
            solver_mode=1,
        )
    else:
        return make_tso_case14_params(
            max_steps=max_steps,
            delta_t_hours=delta_t_hours,
            enable_uc=False,
            enable_reserve=False,
            solver_mode=1,
        )


def make_tso_uc_params(
    case_id: str = "118",
    max_steps: int = 48,
    delta_t_hours: float = 0.5,
) -> UCParams:
    """Variant: TSO-UC — commitment + min-up/down + startup cost, NO reserve cost.

    enable_uc=True, enable_reserve=False.
    Reserve shortfall is still tracked in info but not charged as CMDP cost.
    """
    if case_id == "118":
        return make_tso_case118_params(
            max_steps=max_steps,
            delta_t_hours=delta_t_hours,
            enable_uc=True,
            enable_reserve=False,
            solver_mode=1,
        )
    else:
        return make_tso_case14_params(
            max_steps=max_steps,
            delta_t_hours=delta_t_hours,
            enable_uc=True,
            enable_reserve=False,
            solver_mode=1,
        )


def make_tso_scuc_params(
    case_id: str = "118",
    max_steps: int = 48,
    delta_t_hours: float = 0.5,
    reserve_margin_frac: float = 0.05,
) -> UCParams:
    """Variant: TSO-SCUC-Safe — full SCUC with reserve cost in CMDP cost channel.

    enable_uc=True, enable_reserve=True.
    Task-level CMDP selection consumes thermal overload + reserve shortfall.
    Intended for PPO-Lagrangian / SafeRLWrapper training.
    """
    if case_id == "118":
        return make_tso_case118_params(
            max_steps=max_steps,
            delta_t_hours=delta_t_hours,
            enable_uc=True,
            enable_reserve=True,
            reserve_margin_frac=reserve_margin_frac,
            solver_mode=1,
        )
    else:
        return make_tso_case14_params(
            max_steps=max_steps,
            delta_t_hours=delta_t_hours,
            enable_uc=True,
            enable_reserve=True,
            reserve_margin_frac=reserve_margin_frac,
            solver_mode=1,
        )


# ============ Baseline rollout utilities ============

def tso_all_on_rollout(
    env: UnitCommitmentEnv,
    params: UCParams,
    key: chex.PRNGKey,
) -> Dict[str, Any]:
    """Reference baseline: all units always ON; economic dispatch by DC-OPF.

    Returns episode metrics dict.
    """
    obs, state = env.reset(key, params)
    n_units = params.case.n_units

    action = jnp.concatenate([
        jnp.ones(n_units, dtype=jnp.float32),
        jnp.zeros(n_units, dtype=jnp.float32),
    ])

    def body(carry, _):
        obs, state, k = carry
        k, k_step = jax.random.split(k)
        obs2, state2, _reward, _costs, _done, info = env.step(k_step, state, action, params)
        step_out = {kk: jnp.asarray(info[kk]) for kk in _TSO_INFO_KEYS}
        return (obs2, state2, k), step_out

    _, traj = jax.lax.scan(body, (obs, state, key), xs=None, length=int(params.max_steps))
    return {k: jnp.asarray(v) for k, v in traj.items()}


def tso_merit_order_rollout(
    env: UnitCommitmentEnv,
    params: UCParams,
    key: chex.PRNGKey,
    n_commit: Optional[int] = None,
) -> Dict[str, Any]:
    """Priority-list (merit-order) commitment, then OPF dispatch (cheapest units first).

    Args:
        n_commit: Number of units to commit each step. None → choose a count from
            total load (commit just enough nameplate capacity to cover load + reserve).
    """
    case = params.case
    n_units = int(case.n_units)
    n_steps = int(params.max_steps)

    # Precompute commitment schedule in numpy (outside JAX trace).
    mc_b_np = np.asarray(case.unit_cost_b)
    p_max_np = np.asarray(case.unit_p_max)
    merit_order = np.argsort(mc_b_np)
    load_profiles_np = np.asarray(params.load_profiles)
    nodes_loads_map_np = np.asarray(case.nodes_loads_map)
    reserve_frac = float(params.reserve_margin_frac)
    n_load_rows = load_profiles_np.shape[0]

    actions_list = []
    for step in range(n_steps):
        t_idx = step % n_load_rows
        load_demand = load_profiles_np[t_idx]
        total_load = float(np.sum(nodes_loads_map_np @ load_demand))
        required_cap = total_load * (1.0 + reserve_frac)
        commit_np = np.zeros(n_units, dtype=np.float32)
        cumcap = 0.0
        for i in merit_order:
            commit_np[i] = 1.0
            cumcap += p_max_np[i]
            if cumcap >= required_cap:
                break
        commit_signal = commit_np * 2.0 - 1.0
        actions_list.append(
            np.concatenate([commit_signal, np.zeros(n_units, dtype=np.float32)])
        )
    actions_arr = jnp.array(np.stack(actions_list), dtype=jnp.float32)  # (n_steps, 2*n_units)

    obs, state = env.reset(key, params)

    def body(carry, action_t):
        obs, state, k = carry
        k, k_step = jax.random.split(k)
        obs2, state2, _reward, _costs, _done, info = env.step(k_step, state, action_t, params)
        step_out = {kk: jnp.asarray(info[kk]) for kk in _TSO_INFO_KEYS}
        return (obs2, state2, k), step_out

    _, traj = jax.lax.scan(body, (obs, state, key), xs=actions_arr)
    return {k: jnp.asarray(v) for k, v in traj.items()}


# ============ Metrics ============

def _comparison_tso_synthetic_trace(
    n_steps: int = 48,
    delta_t_hours: float = 0.5,
) -> np.ndarray:
    """Sin-based deterministic trace for tests that omit GB parquet.

    Parity and production load traces use ``make_comparison_tso_load_trace``.

    Formula:
        t[i]     = i * delta_t_hours
        phase[i] = 2π * (t[i] - 3.0) / 24.0
        norm[i]  = clip(0.75 + 0.25 * sin(phase[i]), 0.10, 1.00)
    """
    t = np.arange(n_steps, dtype=np.float64) * delta_t_hours
    phase = 2.0 * np.pi * (t - 3.0) / 24.0
    return np.clip(0.75 + 0.25 * np.sin(phase), 0.10, 1.00).astype(np.float32)


def make_comparison_tso_load_trace(
    n_steps: int = 48,
    delta_t_hours: float = 0.5,
    *,
    split: str = "train",
    episode_start_idx: int = 0,
    wind_frac: float = 0.7,
    solar_frac: float = 0.5,
    data_loader: Optional[Any] = None,
) -> np.ndarray:
    """Net-load factor time series from GB parquet (load, wind, solar columns).

    Loads the requested split window, takes rows ``episode_start_idx`` :
    ``episode_start_idx + n_steps``, peak-normalises demand / wind / solar
    each with its own max (≥1), forms
    ``clip(gross - wind_frac*wind - solar_frac*solar, 0.05, 1.0)`` — same
    composition as ``make_tso_net_load_profiles`` for real arrays.

    Returns
    -------
    float32 array of shape ``(n_steps,)`` with values in ``[0.05, 1.0]``.
    Regression coverage: ``tests/benchmarks/test_tso_comparison_parity.py``.

    Notes
    -----
    ``_comparison_tso_synthetic_trace`` is the offline sin substitute when
    parquet is unavailable.
    """
    if data_loader is None:
        from powerzoojax.data.data_loader import DataLoader
        data_loader = DataLoader()

    from powerzoojax.data import signals as S
    from powerzoojax.data.splits import (
        GB_TRAIN_START, GB_TRAIN_END,
        GB_IID_START, GB_IID_END,
    )

    split_windows = {
        "train": (GB_TRAIN_START, GB_TRAIN_END),
        "iid":   (GB_IID_START, GB_IID_END),
    }
    if split not in split_windows:
        raise ValueError(
            f"comparison_tso load trace supports split in {sorted(split_windows)}, "
            f"got {split!r}.  No summer_ood / zone_holdout window is defined here."
        )
    start, end = split_windows[split]

    raw = data_loader.load_jax_profiles(
        [S.LOAD_ACTUAL_MW, S.WIND_AVAILABLE_MW, S.SOLAR_AVAILABLE_MW],
        start_date=start,
        end_date=end,
        resample="30min",
    )
    raw_np = np.asarray(raw)
    t0 = int(episode_start_idx)
    t1 = t0 + int(n_steps)
    if raw_np.shape[0] < t1:
        raise RuntimeError(
            f"GB parquet returned {raw_np.shape[0]} rows but episode_start_idx "
            f"({t0}) + n_steps ({n_steps}) requires {t1} (split={split!r}, "
            f"window={start}..{end}).  Either lower episode_start_idx or extend "
            f"the GB parquet."
        )
    demand = np.asarray(raw_np[t0:t1, 0], dtype=np.float32)
    wind = np.asarray(raw_np[t0:t1, 1], dtype=np.float32)
    solar = np.asarray(raw_np[t0:t1, 2], dtype=np.float32)

    # Same per-series peak normalisation and net composition as
    # ``make_tso_net_load_profiles`` for real arrays.
    gross = demand / max(float(demand.max()), 1.0)
    wind_n = wind / max(float(wind.max()), 1.0)
    solar_n = solar / max(float(solar.max()), 1.0)
    net_norm = np.clip(
        gross - wind_frac * wind_n - solar_frac * solar_n, 0.05, 1.0
    )
    return net_norm.astype(np.float32)


def make_comparison_tso_params(
    max_steps: int = 48,
    delta_t_hours: float = 0.5,
    reserve_margin_frac: float = 0.05,
    *,
    split: str = "train",
    episode_start_idx: int = 0,
    data_loader: Optional[Any] = None,
) -> UCParams:
    """UCParams for TSO case118 with GB-backed comparison load trace.

    Fixed recipe:
        case118; single agent; 48 steps × 0.5 h; UC and reserve on;
        ``reserve_margin_frac=0.05``; load from ``make_comparison_tso_load_trace``;
        reward scale ``1e-4``; DC-OPF solver (``solver_mode=1``).

    Action: Box(108) in ``[-1, 1]`` — ``commit_intent(54) | dispatch(54)``;
    ``commit_intent > 0`` means unit ON. Reward sums gen / startup / no-load
    cost; the task-level CMDP channels are thermal overload and reserve shortfall.

    See ``TSO_COMPARISON_SCHEMA`` for observation size and solver labels used
    when pairing with a non-JAX reference env.
    """
    from powerzoojax.case import load_case
    case = load_case("118")

    trace = make_comparison_tso_load_trace(
        n_steps=max_steps,
        delta_t_hours=delta_t_hours,
        split=split,
        episode_start_idx=episode_start_idx,
        data_loader=data_loader,
    )
    load_profiles = jnp.array(
        trace[:, None] * np.asarray(case.load_d_max)[None, :],
        dtype=jnp.float32,
    )

    return make_tso_case118_params(
        load_profiles=load_profiles,
        max_steps=max_steps,
        delta_t_hours=delta_t_hours,
        reserve_margin_frac=reserve_margin_frac,
        enable_uc=True,
        enable_reserve=True,
        reward_scale=1e-4,
        solver_mode=1,
    )


TSO_COMPARISON_SCHEMA: Dict[str, Any] = {
    # --- Task configuration ---
    "case_id": "case118",
    "n_units": 54,
    "n_agents": 1,
    "max_steps": 48,
    "delta_t_minutes": 30,
    "delta_t_hours": 0.5,
    "load_source": "gb_real",
    "load_signal_keys": (
        "load.actual_mw, wind.available_mw, solar.available_mw via DataLoader; "
        "net factor from make_comparison_tso_load_trace."
    ),
    "enable_uc": True,
    "enable_reserve": True,
    "reserve_margin_frac": 0.05,
    # --- Unified action contract ---
    "action_shape": (108,),
    "action_range": "[-1, 1]",
    "action_layout": "[commit_intent(54) | dispatch(54)]",
    "action_semantics": "commit_intent > 0 → unit ON; dispatch → [p_min, p_max] scale",
    # --- Implementation labels (JAX vs reference) ---
    "solver_mode_jax": "dc_opf",
    "solver_mode_python": "score_based_allocation",
    "obs_shape_jax": (410,),
    "obs_shape_python": (249,),
    "reward_components": ["gen_cost", "startup_cost", "no_load_cost"],
    "reward_scale": 1e-4,
    "cost_channels_jax": ["thermal_overload", "reserve_shortfall"],
    "cost_channels_python": ["thermal_overload", "reserve_shortfall"],
    # --- Known interface differences vs compact reference stack ---
    "accepted_gaps": [
        "obs_shape: JAX (410,) vs reference (249,) — different feature layout",
        "dispatch_solver: JAX DC-OPF vs reference score-based allocation",
        "reserve_cost_routing: JAX CMDP cost channel vs reference info-only",
    ],
}


def compute_tso_metrics(
    rollout_results: Dict[str, Any],
    baseline_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Aggregate episode-level TSO metrics.

    Args:
        rollout_results: Dict from tso_*_rollout or custom policy rollout.
        baseline_results: Optional reference (e.g. all_on) for relative metrics.

    Returns:
        Dict of scalar metrics.
    """
    r = rollout_results
    reserve_shortfall = np.asarray(r.get("reserve_shortfall", [0]), dtype=np.float32)
    thermal_overload = np.asarray(
        r.get("cost_thermal_overload", [0]), dtype=np.float32
    )
    is_safe = np.asarray(r.get("is_safe", [1]), dtype=np.float32)
    metrics: Dict[str, float] = {
        "total_gen_cost": float(np.sum(r.get("gen_cost", [0]))),
        "total_startup_cost": float(np.sum(r.get("startup_cost", [0]))),
        "total_no_load_cost": float(np.sum(r.get("no_load_cost", [0]))),
        "total_operating_cost": float(
            np.sum(r.get("gen_cost", [0]))
            + np.sum(r.get("startup_cost", [0]))
            + np.sum(r.get("no_load_cost", [0]))
        ),
        "mean_reserve_shortfall": float(np.mean(reserve_shortfall)),
        "total_reserve_shortfall": float(np.sum(reserve_shortfall)),
        "reserve_shortfall_rate": float(np.mean(reserve_shortfall > 1e-6)),
        "mean_thermal_cost": float(np.mean(thermal_overload)),
        "total_thermal_cost": float(np.sum(thermal_overload)),
        "thermal_violation_rate": float(np.mean(thermal_overload > 1e-6)),
        "total_commitment_switches": float(np.sum(r.get("commitment_switches", [0]))),
        "feasibility_rate": float(np.mean(is_safe)),
        "total_cost_violations": float(np.sum(r.get("cost_sum", [0]))),
    }

    if baseline_results is not None:
        b = baseline_results
        b_total = (
            float(np.sum(b.get("gen_cost", [0])))
            + float(np.sum(b.get("startup_cost", [0])))
            + float(np.sum(b.get("no_load_cost", [0])))
        )
        if b_total > 0:
            metrics["cost_vs_baseline_pct"] = (
                (metrics["total_operating_cost"] - b_total) / b_total * 100.0
            )

    return metrics


# ============ Scan-based rollout (trained policy) ============

_TSO_INFO_KEYS = (
    "gen_cost", "startup_cost", "no_load_cost",
    "reserve_shortfall", "cost_thermal_overload",
    "cost_sum", "commitment_switches", "is_safe",
    "opf_converged", "opf_iterations",
    "opf_box_residual_mw", "opf_line_residual_mw", "opf_balance_residual_mw",
)


def rollout_tso(
    env: UnitCommitmentEnv,
    params: UCParams,
    key: chex.PRNGKey,
    policy_fn,
) -> Dict[str, Any]:
    """Single-episode lax.scan rollout for a trained TSO policy.

    Args:
        env: UnitCommitmentEnv instance.
        params: UCParams for this episode.
        key: JAX PRNGKey.
        policy_fn: Callable ``(obs, state, key) -> action``.

    Returns:
        Dict with arrays matching the structure returned by
        ``tso_all_on_rollout`` / ``tso_merit_order_rollout``.
    """
    obs0, state0 = env.reset(key, params)

    def body(carry, _):
        obs, state, k = carry
        k, k_step, k_pol = jax.random.split(k, 3)
        action = policy_fn(obs, state, k_pol)
        obs2, state2, _reward, _costs, _done, info = env.step(k_step, state, action, params)
        stepwise = {kk: jnp.asarray(info[kk]) for kk in _TSO_INFO_KEYS}
        return (obs2, state2, k), stepwise

    _, traj = jax.lax.scan(
        body, (obs0, state0, key), xs=None, length=int(params.max_steps)
    )
    return {k: jnp.asarray(v) for k, v in traj.items()}


# ============ TaskSpec implementation ============

class TSOTask:
    """TaskSpec implementation for the TSO SCUC benchmark.

    Single-agent unit commitment task on case118 with GB demand data.

    Args:
        max_steps: Episode length in 30-min steps (default: 48).
        dt_hours: Duration of one step in hours (default: 0.5).
        reserve_margin_frac: Reserve margin fraction (default: 0.05).
        reward_scale: Multiplier for reward = -reward_scale * operating_cost.
        cost_thermal_weight: Weight on thermal-overload cost channel.
        solver_mode: DC-OPF solver mode (default: 1).
        forecast_horizon_steps: Number of future total-load forecast steps in
            the observation.
        load_scale: Load scaling factor for OOD evaluation (default: 1.0).
        line_rating_scale: Line rating scaling for OOD evaluation (default: 1.0).
    """

    task_name = "tso"
    default_splits = ("train", "iid", "load_stress", "line_tightening")

    def __init__(
        self,
        max_steps: int = 48,
        dt_hours: float = 0.5,
        reserve_margin_frac: float = 0.05,
        reward_scale: float = 1e-4,
        cost_thermal_weight: float = 1.0,
        solver_mode: int = 1,
        dcopf_max_iter: int = 100,
        dcopf_tol: float = 1e-3,
        forecast_horizon_steps: int = 0,
        load_scale: float = 1.0,
        line_rating_scale: float = 1.0,
        gb_train_start: Optional[str] = None,
        gb_train_end: Optional[str] = None,
        gb_iid_start: Optional[str] = None,
        gb_iid_end: Optional[str] = None,
    ) -> None:
        from powerzoojax.data.splits import (
            GB_IID_END,
            GB_IID_START,
            GB_TRAIN_END,
            GB_TRAIN_START,
        )

        self._max_steps = max_steps
        self._dt_hours = dt_hours
        self._reserve_margin_frac = reserve_margin_frac
        self._reward_scale = reward_scale
        self._cost_thermal_weight = cost_thermal_weight
        self._solver_mode = solver_mode
        self._dcopf_max_iter = int(dcopf_max_iter)
        self._dcopf_tol = float(dcopf_tol)
        self._forecast_horizon_steps = int(forecast_horizon_steps)
        self._load_scale = load_scale
        self._line_rating_scale = line_rating_scale
        self._gb_windows = {
            "train": (
                str(gb_train_start or GB_TRAIN_START),
                str(gb_train_end or GB_TRAIN_END),
            ),
            "iid": (
                str(gb_iid_start or GB_IID_START),
                str(gb_iid_end or GB_IID_END),
            ),
        }
        self._raw_cache: Dict = {}  # gb_split → np.ndarray [T, 3]
        self._case = None

    def _get_case(self):
        if self._case is None:
            from powerzoojax.case import load_case
            self._case = load_case("118")
        return self._case

    def _gb_split_for(self, split: str) -> str:
        split_map = {
            "train": "train",
            "iid": "iid",
            "load_stress": "iid",
            "line_tightening": "iid",
        }
        if split not in split_map:
            raise ValueError(
                f"Unknown TSO split {split!r}. Expected one of {tuple(split_map)}"
            )
        return split_map[split]

    def _gb_window_for(self, split: str) -> tuple[str, str]:
        gb_split = self._gb_split_for(split)
        return self._gb_windows[gb_split]

    def _get_raw(self, split: str) -> np.ndarray:
        gb = self._gb_split_for(split)
        if gb in self._raw_cache:
            return self._raw_cache[gb]
        from powerzoojax.data.data_loader import DataLoader
        from powerzoojax.data import signals as S
        start, end = self._gb_window_for(split)
        try:
            raw = DataLoader().load_jax_profiles(
                [S.LOAD_ACTUAL_MW, S.WIND_AVAILABLE_MW, S.SOLAR_AVAILABLE_MW],
                start_date=start, end_date=end, resample="30min",
            )
        except Exception as exc:
            raise RuntimeError(
                f"TSOTask requires real GB data for gb_split={gb!r}: {exc}"
            ) from exc
        self._raw_cache[gb] = np.asarray(raw)
        return self._raw_cache[gb]

    def _finalize_params(
        self,
        profiles: chex.Array,
        *,
        sample_start_on_reset: bool = False,
    ) -> UCParams:
        params = make_tso_case118_params(
            load_profiles=profiles,
            max_steps=self._max_steps,
            delta_t_hours=self._dt_hours,
            reserve_margin_frac=self._reserve_margin_frac,
            enable_uc=True,
            enable_reserve=True,
            forecast_horizon_steps=self._forecast_horizon_steps,
            reward_scale=self._reward_scale,
            cost_thermal_weight=self._cost_thermal_weight,
            solver_mode=self._solver_mode,
            dcopf_max_iter=self._dcopf_max_iter,
            dcopf_tol=self._dcopf_tol,
            sample_start_on_reset=sample_start_on_reset,
        )
        if abs(self._line_rating_scale - 1.0) > 1e-6:
            new_cap = jnp.array(
                np.asarray(params.case.line_cap) * self._line_rating_scale,
                dtype=jnp.float32,
            )
            new_floor = jnp.array(
                np.asarray(params.case.line_floor) * self._line_rating_scale,
                dtype=jnp.float32,
            )
            replace_kwargs = {
                "line_cap": new_cap,
                "line_floor": new_floor,
            }
            if params.case.line_rate_a is not None:
                replace_kwargs["line_rate_a"] = jnp.array(
                    np.asarray(params.case.line_rate_a) * self._line_rating_scale,
                    dtype=jnp.float32,
                )
            params = params.replace(case=params.case.replace(**replace_kwargs))
        return params

    def _build_params(self, raw: np.ndarray, t0: int, t1: int) -> UCParams:
        case = self._get_case()
        profiles = make_tso_net_load_profiles(
            n_steps=self._max_steps,
            load_d_max=np.asarray(case.load_d_max) * self._load_scale,
            gross_demand_mw=raw[t0:t1, 0],
            wind_mw=raw[t0:t1, 1],
            solar_mw=raw[t0:t1, 2],
        )
        return self._finalize_params(profiles, sample_start_on_reset=False)

    def training_params(self, *, max_steps: int | None = None) -> UCParams:
        raw = self._get_raw("train")
        case = self._get_case()
        episode_steps = self._max_steps if max_steps is None else int(max_steps)
        profiles = make_tso_net_load_profiles(
            n_steps=int(len(raw)),
            load_d_max=np.asarray(case.load_d_max) * self._load_scale,
            gross_demand_mw=raw[:, 0],
            wind_mw=raw[:, 1],
            solar_mw=raw[:, 2],
        )
        params = self._finalize_params(profiles, sample_start_on_reset=True)
        if episode_steps != int(params.max_steps):
            params = params.replace(max_steps=episode_steps)
        return params

    def make_env(self, split: str = "train") -> UnitCommitmentEnv:
        return UnitCommitmentEnv()

    def episode_params(
        self,
        split: str,
        episode_idx: int,
        n_episodes: int,
        max_steps: int,
        *,
        strategy: str = "uniform",
        seed: int = 0,
    ) -> UCParams:
        raw = self._get_raw(split)
        T = len(raw)
        if strategy == "seeded":
            rng = np.random.default_rng(seed * 997 + 7)
            start = int(rng.integers(0, max(1, T - max_steps)))
        else:
            starts = np.linspace(0, max(0, T - max_steps), n_episodes).astype(int)
            start = int(starts[episode_idx])
        t1 = min(start + max_steps, T)
        t0 = max(0, t1 - max_steps)
        return self._build_params(raw, t0, t1)

    def rollout(self, env, params, key, policy_fn):
        return rollout_tso(env, params, key, policy_fn)

    def baseline_rollout(self, env, params, key, baseline_name: str):
        _map = {
            "all_on":      tso_all_on_rollout,
            "merit_order": tso_merit_order_rollout,
            "no_control":  tso_all_on_rollout,  # TSO has no off-state; all_on is the safe reference
        }
        return _map[baseline_name](env, params, key)

    def compute_metrics(self, agent_rollout, baseline_rollout):
        return compute_tso_metrics(agent_rollout, baseline_rollout)

    def baseline_names(self) -> tuple:
        return ("all_on", "merit_order")

    def constraint_spec(self) -> ConstraintSpec:
        # Physical SCUC security constraints are hard targets: line overloads
        # and reserve shortfall should both be zero in the CMDP budget.
        return ConstraintSpec(
            selected_names=("thermal_overload", "reserve_shortfall"),
            thresholds=(0.0, 0.0),
            fallback_weights=(1.0, 1.0),
        )
