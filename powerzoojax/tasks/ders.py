"""DERs environment factory — case141 + 12 heterogeneous DER agents.

Main configuration (12 heterogeneous agents on ``case141`` / registry ``141``,
141-bus Caracas-area distribution, Khodr et al. 2008):
    4× Battery (P+Q)     — bus 9, 55, 17, 122  — 0.3 MWh / 0.1 MW
    4× PV inverter       — bus 6, 73, 72, 82   — 0.2 MW nameplate
    4× FlexLoad          — bus 41, 70, 135, 24 — 0.10 MW curtail/shift

Buses are chosen on the slack-rooted tree by BFS depth to spread control along
the feeder (see benchmark docs).

Total: 12 agents, each with action Box(2).  All variants use Dec-POMDP
local observations (≈15-dim) via DistGridMARLEnv(observation_mode="local").

Training status (see presets.py):
    - Current training path: reward-shaped IPPO (``ders-medium`` / ``ders-medium-safe``)
    - Benchmark extension path: typed IPPO-Lagrangian (explicit MARL cost channel)
    - MAPPO-style centralized critics are still not implemented

Key differences from DSO task:
    - Larger network: case141 (141-bus) vs case33bw (33-bus)
    - Heterogeneous resources: Battery + PV + FlexLoad
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from powerzoojax.envs.grid.dist import DistGridEnv, DistGridParams, make_dist_params
from powerzoojax.envs.resource.battery import make_battery_bundle, BatteryBundle
from powerzoojax.envs.resource.renewable import make_renewable_bundle, RenewableBundle
from powerzoojax.envs.resource.flexload import make_flexload_bundle, FlexLoadBundle
from powerzoojax.case.case_data import CaseData
from powerzoojax.tasks.base import ConstraintSpec


# ============ Agent deployment (buses and device parameters) ============

#: ``load_case`` argument for the formal DERs benchmark (registry key ``141``).
DERS_BENCHMARK_LOAD_CASE = "case141"


def _default_ders_case() -> CaseData:
    from powerzoojax.case import load_case

    return load_case(DERS_BENCHMARK_LOAD_CASE)


#: 4 Battery agent buses (spread by BFS depth on case141)
DERS_BATTERY_BUSES = [9, 55, 17, 122]

#: 4 PV agent buses
DERS_PV_BUSES = [6, 73, 72, 82]

#: 4 FlexLoad agent buses
DERS_FLEXLOAD_BUSES = [41, 70, 135, 24]

#: Battery config: power_mw, capacity_mwh, s_rated_mva
DERS_BATTERY_CONFIG = {
    "power_mw": 0.10,
    "capacity_mwh": 0.30,
    "s_rated_mva": 0.15,    # sqrt(0.10² + 0.08²) ≈ 0.128 → 0.15 for headroom
    "enable_q_control": True,
    "soc_min": 0.1,
    "soc_max": 0.9,
    "initial_soc": 0.5,
}

#: PV config: capacity_mw, s_rated_mva
DERS_PV_CONFIG = {
    "capacity_mw": 0.20,
    "s_rated_mva": 0.22,    # sqrt(0.20² + 0.06²) ≈ 0.209 → 0.22 for headroom
    "allow_curtailment": True,
    "curtail_cost_per_mwh": 0.0,
}

#: FlexLoad config: curtail_cap_mw, shift_cap_mw
DERS_FLEXLOAD_CONFIG = {
    "curtail_cap_mw": 0.10,
    "shift_cap_mw": 0.10,
    "shift_horizon": 4,
}

#: Voltage limits [p.u.] — strict for safe MARL
DERS_V_MIN = 0.94
DERS_V_MAX = 1.06


# ============ Bundle factories ============

def make_ders_battery_bundle(case, *, max_steps: int = 48, dt_hours: float = 0.5) -> BatteryBundle:
    """Create the 4-device Battery bundle for DERs (buses in ``DERS_BATTERY_BUSES``)."""
    return make_battery_bundle(
        case,
        bus_ids=DERS_BATTERY_BUSES,
        power_mw=DERS_BATTERY_CONFIG["power_mw"],
        capacity_mwh=DERS_BATTERY_CONFIG["capacity_mwh"],
        s_rated_mva=DERS_BATTERY_CONFIG["s_rated_mva"],
        enable_q_control=DERS_BATTERY_CONFIG["enable_q_control"],
        soc_min=DERS_BATTERY_CONFIG["soc_min"],
        soc_max=DERS_BATTERY_CONFIG["soc_max"],
        initial_soc=DERS_BATTERY_CONFIG["initial_soc"],
        dt_hours=dt_hours,
    )


def _default_pv_profiles(max_steps: int, n: int) -> jnp.ndarray:
    """Bell-curve solar profiles, shape (max_steps, n)."""
    steps = jnp.arange(max_steps, dtype=jnp.float32)
    hour = steps / float(max_steps) * 24.0
    shape = jnp.clip(jnp.sin(jnp.pi * (hour - 6.0) / 12.0), 0.0, 1.0)
    return jnp.broadcast_to(shape[:, None], (max_steps, n))


def _window_with_wraparound(
    series: np.ndarray,
    *,
    max_steps: int,
    episode_start: int = 0,
) -> np.ndarray:
    """Extract a ``max_steps`` window with wraparound from a 1-D series."""
    arr = np.asarray(series, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise ValueError("Cannot slice an empty profile series")
    idx = (np.arange(max_steps, dtype=np.int32) + int(episode_start)) % arr.size
    return arr[idx].astype(np.float32)


def _gb_window_for_role(role: str) -> tuple[str, str]:
    """Map a DERs split role to the corresponding GB renewables window."""
    from powerzoojax.data.splits import (
        GB_IID_END,
        GB_IID_START,
        GB_TRAIN_END,
        GB_TRAIN_START,
    )

    if role in ("train", "zone_holdout"):
        return GB_TRAIN_START, GB_TRAIN_END
    if role in ("iid", "summer_ood"):
        return GB_IID_START, GB_IID_END
    raise KeyError(
        f"Unknown DERs split role '{role}'. "
        "Expected one of: 'train', 'iid', 'summer_ood', 'zone_holdout'."
    )


def load_ders_load_shape(
    data_loader=None,
    *,
    role: str = "train",
    resample: str = "30min",
) -> np.ndarray:
    """Load an Ausgrid demand shape for DERs and normalise it by split peak."""
    from powerzoojax.data.ausgrid_utils import (
        filter_ausgrid_role_days,
        get_ausgrid_split,
    )

    if data_loader is None:
        from powerzoojax.data.data_loader import DataLoader
        data_loader = DataLoader()

    start, end, substations = get_ausgrid_split(role)
    if not substations:
        raise ValueError(f"No Ausgrid substations configured for role='{role}'")

    profiles = []
    for substation in substations:
        if hasattr(data_loader, "load_signals"):
            df = data_loader.load_signals(
                ["load.actual_mw"],
                source="ausgrid",
                region=substation,
                start_date=start,
                end_date=end,
                resample=resample,
            )
            df = filter_ausgrid_role_days(df, role)
            profiles.append(np.asarray(df["load.actual_mw"], dtype=np.float32).reshape(-1))
        else:
            arr = data_loader.load_jax_profiles(
                ["load.actual_mw"],
                source="ausgrid",
                region=substation,
                start_date=start,
                end_date=end,
                resample=resample,
            )
            profiles.append(np.asarray(arr, dtype=np.float32).reshape(-1))

    min_len = min(len(p) for p in profiles)
    stacked = np.stack([p[:min_len] for p in profiles], axis=0)
    avg = stacked.mean(axis=0)
    peak = max(float(np.max(avg)), 1e-8)
    return (avg / peak).astype(np.float32)


def load_ders_pv_profile(
    data_loader=None,
    *,
    role: str = "train",
    resample: str = "30min",
) -> np.ndarray:
    """Load a GB solar-availability profile for DERs and normalise by peak."""
    if data_loader is None:
        from powerzoojax.data.data_loader import DataLoader
        data_loader = DataLoader()

    start, end = _gb_window_for_role(role)
    arr = data_loader.load_jax_profiles(
        ["solar.available_mw"],
        source="gb",
        start_date=start,
        end_date=end,
        resample=resample,
    )
    pv = np.asarray(arr, dtype=np.float32).reshape(-1)
    peak = max(float(np.max(pv)), 1e-8)
    return (pv / peak).astype(np.float32)


def load_ders_split_profiles(
    data_loader=None,
    *,
    role: str = "train",
    resample: str = "30min",
) -> tuple[np.ndarray, np.ndarray]:
    """Load the split-level load and PV profiles used by the DER benchmark."""
    load_shape = load_ders_load_shape(
        data_loader=data_loader,
        role=role,
        resample=resample,
    )
    pv_profile = load_ders_pv_profile(
        data_loader=data_loader,
        role=role,
        resample=resample,
    )
    return load_shape, pv_profile


def select_ders_episode_start(
    total_len: int,
    *,
    max_steps: int,
    episode_idx: int,
    n_episodes: int,
    strategy: str = "uniform",
    seed: int = 0,
    salt: int = 11,
) -> int:
    """Select a deterministic episode start for DER split windows."""
    if total_len <= 0:
        raise ValueError("total_len must be positive")
    usable = max(0, int(total_len) - int(max_steps))
    if strategy == "seeded":
        rng = np.random.default_rng(seed * 997 + salt)
        return int(rng.integers(0, max(1, usable + 1)))
    starts = np.linspace(0, usable, max(int(n_episodes), 1)).astype(int)
    return int(starts[int(episode_idx)])


def apply_ders_profile_window(
    base_params: DistGridParams,
    *,
    load_shape: np.ndarray,
    pv_profile: np.ndarray,
    episode_start: int = 0,
    load_scale: float = 1.0,
    pv_scale: float = 1.0,
) -> DistGridParams:
    """Inject a temporal load/PV window into an existing DERs params object."""
    max_steps = int(base_params.max_steps)
    load_window = _window_with_wraparound(
        load_shape, max_steps=max_steps, episode_start=episode_start
    )
    pv_window = _window_with_wraparound(
        pv_profile, max_steps=max_steps, episode_start=episode_start
    )
    load_window = np.asarray(load_window, dtype=np.float32) * float(load_scale)
    pv_window = np.asarray(pv_window, dtype=np.float32) * float(pv_scale)

    base_load_p = base_params.load_profiles_p[:1]
    base_load_q = base_params.load_profiles_q[:1]
    load_profiles_p = jnp.asarray(base_load_p * load_window[:, None], dtype=jnp.float32)
    load_profiles_q = jnp.asarray(base_load_q * load_window[:, None], dtype=jnp.float32)

    new_resources = []
    for bundle in base_params.resources:
        if isinstance(bundle, RenewableBundle):
            pv_profiles = np.broadcast_to(
                pv_window[:, None], (max_steps, bundle.n_devices)
            ).astype(np.float32)
            new_resources.append(
                bundle.replace(profiles=jnp.asarray(pv_profiles, dtype=jnp.float32))
            )
        else:
            new_resources.append(bundle)

    return base_params.replace(
        load_profiles_p=load_profiles_p,
        load_profiles_q=load_profiles_q,
        resources=tuple(new_resources),
    )


def make_ders_params_from_split(
    case=None,
    *,
    role: str = "train",
    data_loader=None,
    resample: str = "30min",
    episode_start: int = 0,
    max_steps: int = 48,
    dt_hours: float = 0.5,
    v_min: float = DERS_V_MIN,
    v_max: float = DERS_V_MAX,
    loss_penalty_weight: float = 0.1,
    load_scale: float = 1.0,
    pv_scale: float = 1.0,
) -> DistGridParams:
    """Create real-data-driven DERs params for a named split."""
    base_params = make_ders_params(
        case,
        max_steps=max_steps,
        dt_hours=dt_hours,
        v_min=v_min,
        v_max=v_max,
        loss_penalty_weight=loss_penalty_weight,
    )
    load_shape, pv_profile = load_ders_split_profiles(
        data_loader=data_loader,
        role=role,
        resample=resample,
    )
    return apply_ders_profile_window(
        base_params,
        load_shape=load_shape,
        pv_profile=pv_profile,
        episode_start=episode_start,
        load_scale=load_scale,
        pv_scale=pv_scale,
    )


def make_ders_pv_bundle(
    case,
    *,
    max_steps: int = 48,
    profiles: Optional[jnp.ndarray] = None,
    dt_hours: float = 0.5,
) -> RenewableBundle:
    """Create the 4-device PV bundle for DERs (buses in ``DERS_PV_BUSES``).

    Args:
        case: CaseData (default benchmark: :func:`create_case141` from ``powerzoojax.case``).
        max_steps: Episode length (default 48).
        profiles: Capacity factor array, shape ``(max_steps, 4)`` or ``(max_steps,)``.
            If None, a synthetic bell-curve PV profile is used.
        dt_hours: Time-step duration.
    """
    n = len(DERS_PV_BUSES)
    if profiles is None:
        profiles = _default_pv_profiles(max_steps, n)

    return make_renewable_bundle(
        case,
        bus_ids=DERS_PV_BUSES,
        capacity_mw=DERS_PV_CONFIG["capacity_mw"],
        s_rated_mva=DERS_PV_CONFIG["s_rated_mva"],
        profiles=profiles,
        allow_curtailment=DERS_PV_CONFIG["allow_curtailment"],
        curtail_cost_per_mwh=DERS_PV_CONFIG["curtail_cost_per_mwh"],
        dt_hours=dt_hours,
    )


def make_ders_flexload_bundle(case, *, dt_hours: float = 0.5) -> FlexLoadBundle:
    """Create the 4-device FlexLoad bundle for DERs (buses in ``DERS_FLEXLOAD_BUSES``)."""
    return make_flexload_bundle(
        case,
        bus_ids=DERS_FLEXLOAD_BUSES,
        curtail_cap_mw=DERS_FLEXLOAD_CONFIG["curtail_cap_mw"],
        shift_cap_mw=DERS_FLEXLOAD_CONFIG["shift_cap_mw"],
        shift_horizon=DERS_FLEXLOAD_CONFIG["shift_horizon"],
        dt_hours=dt_hours,
    )


# ============ Main params factory ============

def make_ders_params(
    case=None,
    *,
    max_steps: int = 48,
    dt_hours: float = 0.5,
    v_min: float = DERS_V_MIN,
    v_max: float = DERS_V_MAX,
    loss_penalty_weight: float = 0.1,
    pv_profiles: Optional[jnp.ndarray] = None,
    load_profiles_p: Optional[jnp.ndarray] = None,
    load_profiles_q: Optional[jnp.ndarray] = None,
) -> DistGridParams:
    """Create ``DistGridParams`` for the DERs task on case141.

    Attaches 3 resource bundles totalling 12 DER agents:
        - 4× Battery with Q-control (buses in ``DERS_BATTERY_BUSES``)
        - 4× PV inverter
        - 4× FlexLoad

    Args:
        case: CaseData.  If None, :func:`_default_ders_case` is used.
        max_steps: Episode length.
        dt_hours: Time-step duration.
        v_min: Minimum allowed bus voltage [p.u.].
        v_max: Maximum allowed bus voltage [p.u.].
        loss_penalty_weight: Weight on active-power-loss reward term.
        pv_profiles: Capacity factor array ``(max_steps, 4)`` for the 4 PV devices.
            If None, a synthetic bell-curve profile is used.
        load_profiles_p: Active load time series ``(max_steps, n_nodes)`` [p.u.].
            If None, flat profiles from the case base load are used.
        load_profiles_q: Reactive load time series.  Defaults to flat.

    Returns:
        :class:`DistGridParams` ready for :class:`DistGridEnv` or
        :class:`~powerzoojax.rl.multi_agent.DistGridMARLEnv`.
    """
    if case is None:
        case = _default_ders_case()

    bat_bundle = make_ders_battery_bundle(case, max_steps=max_steps, dt_hours=dt_hours)
    pv_bundle = make_ders_pv_bundle(case, max_steps=max_steps, profiles=pv_profiles, dt_hours=dt_hours)
    fl_bundle = make_ders_flexload_bundle(case, dt_hours=dt_hours)

    params = make_dist_params(
        case,
        max_steps=max_steps,
        steps_per_day=48,
        v_min=v_min,
        v_max=v_max,
        loss_penalty_weight=loss_penalty_weight,
        resources=(bat_bundle, pv_bundle, fl_bundle),
        include_der=False,
    )

    # Optionally override load profiles
    if load_profiles_p is not None or load_profiles_q is not None:
        p = load_profiles_p if load_profiles_p is not None else params.load_profiles_p
        q = load_profiles_q if load_profiles_q is not None else params.load_profiles_q
        params = params.replace(load_profiles_p=p, load_profiles_q=q)

    return params


# ============ MARL environment factory ============

def make_ders_marl_env(
    case=None,
    *,
    max_steps: int = 48,
    v_min: float = DERS_V_MIN,
    v_max: float = DERS_V_MAX,
    voltage_penalty: float = 8.0,
    pv_profiles: Optional[jnp.ndarray] = None,
    observation_mode: str = "local",
):
    """Create a :class:`~powerzoojax.rl.multi_agent.DistGridMARLEnv` for DERs.

    Returns ``(env, params)`` where ``env`` is a
    :class:`~powerzoojax.rl.multi_agent.DistGridMARLEnv` with 12 agents
    (battery_0…3, renewable_0…3, flexload_0…3).

    Args:
        case: CaseData.  Defaults to the DERs benchmark :func:`_default_ders_case`.
        max_steps: Episode length.
        v_min / v_max: Voltage limits.
        voltage_penalty: Weight on voltage-deviation cost added to reward.
        pv_profiles: Optional PV capacity factor profiles.
        observation_mode: ``"local"`` (default) — each agent sees only its own
            bus voltage, K-hop neighbours, global summary stats, time features,
            and own device state (Dec-POMDP design intent).
            ``"global"`` — legacy full-grid shared obs (backward compat).
    """
    from powerzoojax.rl.multi_agent import DistGridMARLEnv

    if case is None:
        case = _default_ders_case()

    params = make_ders_params(
        case,
        max_steps=max_steps,
        v_min=v_min,
        v_max=v_max,
        pv_profiles=pv_profiles,
    )
    env = DistGridMARLEnv(
        DistGridEnv(), params,
        voltage_penalty=voltage_penalty,
        observation_mode=observation_mode,
    )
    return env, params


# ============ Baseline rollouts ============

def rollout_ders(env, params, key, policy_fn):
    """Single-episode rollout returning per-step metrics.

    Args:
        env: DistGridEnv instance.
        params: DistGridParams from make_ders_params.
        key: JAX PRNGKey.
        policy_fn: Callable ``(obs, key) -> action``.

    Returns:
        Dict with arrays ``reward``, ``cost_continuous``, ``p_loss_MW``,
        ``v_min_episode``, ``v_max_episode``, each shape ``(max_steps,)``.
    """
    obs0, state0 = env.reset(key, params)
    max_steps = int(params.max_steps)

    def body(carry, _):
        obs, state, k = carry
        k, k_step = jax.random.split(k)
        action = policy_fn(obs, k_step)
        obs2, state2, rew, _costs, done, info = env.step(k_step, state, action, params)
        out = {
            "reward": rew,
            "cost_continuous": info["cost_continuous"],
            "p_loss_MW": info["p_loss_MW"],
            "v_min_episode": jnp.asarray(info.get("v_min_step", jnp.min(state2.v_mag))),
            "v_max_episode": jnp.asarray(info.get("v_max_step", jnp.max(state2.v_mag))),
        }
        return (obs2, state2, k), out

    _, traj = jax.lax.scan(body, (obs0, state0, key), xs=None, length=max_steps)
    # Keep rollout outputs as JAX arrays so callers can safely wrap this
    # function in ``jax.jit``. Metric aggregation converts to NumPy outside JIT.
    traj = dict(traj)
    traj["metric_v_min"] = jnp.asarray(params.v_min, dtype=jnp.float32)
    traj["metric_v_max"] = jnp.asarray(params.v_max, dtype=jnp.float32)
    return traj


def ders_noop_action(params) -> jnp.ndarray:
    """Device-wise no-op action for the heterogeneous DER benchmark.

    Battery:
        ``[P, Q] = [0, 0]``.
    Renewable:
        ``[curtail, Q] = [+1, 0]`` so PV stays at zero curtailment with zero Q.
    FlexLoad:
        ``[curtail, shift] = [0, 0]``.
    """
    parts: list[jnp.ndarray] = []
    for bundle in params.resources:
        if isinstance(bundle, BatteryBundle):
            parts.append(jnp.zeros(bundle.action_dim, dtype=jnp.float32))
            continue
        if isinstance(bundle, RenewableBundle):
            if bundle.enable_curtailment and bundle.enable_q_control:
                curtail = jnp.ones((bundle.n_devices,), dtype=jnp.float32)
                q = jnp.zeros((bundle.n_devices,), dtype=jnp.float32)
                parts.append(jnp.stack([curtail, q], axis=-1).reshape(-1))
            elif bundle.enable_curtailment:
                parts.append(jnp.ones(bundle.action_dim, dtype=jnp.float32))
            elif bundle.enable_q_control:
                parts.append(jnp.zeros(bundle.action_dim, dtype=jnp.float32))
            else:
                parts.append(jnp.zeros((0,), dtype=jnp.float32))
            continue
        if isinstance(bundle, FlexLoadBundle):
            parts.append(jnp.zeros(bundle.action_dim, dtype=jnp.float32))
            continue
        parts.append(jnp.zeros(bundle.action_dim, dtype=jnp.float32))
    return jnp.concatenate(parts) if parts else jnp.zeros((0,), dtype=jnp.float32)


def ders_no_control_rollout(env, params, key):
    """No-control baseline with bundle-aware no-op actions."""
    return rollout_ders(env, params, key, lambda obs, k: ders_noop_action(params))


def ders_volt_droop_rollout(env, params, key):
    """Voltage-droop **rule-based** baseline (IEEE 1547–inspired local Q–V response).

    Policy:
      - Battery: P=0, Q ∝ (1 - v) * droop_gain, clipped to ±1.
      - PV: curtail if overvoltage (v > 1.03), Q same droop as battery.
      - FlexLoad: curtail load if undervoltage (v < 0.97).

    The voltage at each agent's bus is read directly from the obs vector
    (first n_nodes entries: (v - 1.0) / 0.1).
    """
    bat_bundle, pv_bundle, fl_bundle = (
        params.resources[0], params.resources[1], params.resources[2]
    )
    n_nodes = params.case.n_nodes
    bat_bus = jnp.asarray(bat_bundle.bus_idx, dtype=jnp.int32)
    pv_bus  = jnp.asarray(pv_bundle.bus_idx, dtype=jnp.int32)
    fl_bus  = jnp.asarray(fl_bundle.bus_idx, dtype=jnp.int32)

    n_bat = bat_bundle.n_devices
    n_pv  = pv_bundle.n_devices
    n_fl  = fl_bundle.n_devices

    droop_gain = 10.0  # Q = gain x (1 - v), clipped to [-1, 1]

    def _droop(obs, k):
        # obs[:n_nodes] = (v - 1) / 0.1; recover per-unit voltages.
        v_pu = 1.0 + obs[:n_nodes] * 0.1
        v_bat = v_pu[bat_bus]
        v_pv  = v_pu[pv_bus]
        v_fl  = v_pu[fl_bus]

        # Battery: P=0, Q-droop interleaved [P, Q] per device.
        bat_q = jnp.clip((1.0 - v_bat) * droop_gain, -1.0, 1.0)
        bat_act = jnp.stack([jnp.zeros((n_bat,), dtype=jnp.float32), bat_q], axis=-1).reshape(-1)

        # PV: [curtail, Q] per device. curtail = -0.5 if overvoltage else 1.0.
        pv_curtail = jnp.where(v_pv > 1.03, -0.5, 1.0)
        pv_q = jnp.clip((1.0 - v_pv) * droop_gain, -1.0, 1.0)
        pv_act = jnp.stack([pv_curtail, pv_q], axis=-1).reshape(-1)

        # FlexLoad actions are already fractions in [0, 1].
        fl_curtail = jnp.where(v_fl < 0.97, 0.5, 0.0)
        fl_act = jnp.stack(
            [fl_curtail.astype(jnp.float32), jnp.zeros((n_fl,), dtype=jnp.float32)],
            axis=-1,
        ).reshape(-1)

        return jnp.concatenate([bat_act, pv_act, fl_act])

    return rollout_ders(env, params, key, _droop)


def compute_ders_safety_metrics(
    rollout: dict,
    *,
    v_min: float = DERS_V_MIN,
    v_max: float = DERS_V_MAX,
) -> dict:
    """Compute detailed voltage safety metrics from a DERs rollout.

    Args:
        rollout: Result dict from ``rollout_ders``.
        v_min: Voltage lower bound used for violation detection.  Defaults to
            ``DERS_V_MIN`` (0.94 p.u.).  Pass ``params.v_min`` when evaluating
            under OOD scenarios with tightened voltage bounds.
        v_max: Voltage upper bound.  Defaults to ``DERS_V_MAX`` (1.06 p.u.).

    Returns:
        Dict with:
            ``voltage_safety_rate``: fraction of steps with all buses in [v_min, v_max].
            ``undervoltage_steps``:  steps where min_v < v_min.
            ``overvoltage_steps``:   steps where max_v > v_max.
            ``max_undervoltage_dev``: worst |v_min - ep_v_min| (positive means violation).
            ``max_overvoltage_dev``:  worst |ep_v_max - v_max| (positive means violation).
            ``mean_v_min``:          mean of per-step minimum voltage.
            ``mean_v_max``:          mean of per-step maximum voltage.
    """
    v_min_ep = rollout["v_min_episode"]
    v_max_ep = rollout["v_max_episode"]

    under_mask = v_min_ep < v_min
    over_mask  = v_max_ep > v_max
    safe_mask  = ~under_mask & ~over_mask

    return {
        "voltage_safety_rate":    float(np.mean(safe_mask)),
        "voltage_violation_rate": float(np.mean(~safe_mask)),
        "undervoltage_steps":     int(np.sum(under_mask)),
        "overvoltage_steps":      int(np.sum(over_mask)),
        "max_undervoltage_dev":   float(np.max(np.maximum(v_min - v_min_ep, 0.0))),
        "max_overvoltage_dev":    float(np.max(np.maximum(v_max_ep - v_max, 0.0))),
        "mean_v_min":             float(np.mean(v_min_ep)),
        "mean_v_max":             float(np.mean(v_max_ep)),
    }


# ============ R5 — DERs-large (20 agents, tight voltage) ============

#: Extended battery buses for the 20-agent DERs-large variant (case141)
DERS_LARGE_BATTERY_BUSES  = [33, 7, 39, 11, 92, 54, 44]  # 7 batteries
DERS_LARGE_PV_BUSES       = [15, 104, 57, 106, 62, 136, 83]  # 7 PV inverters
DERS_LARGE_FLEXLOAD_BUSES = [121, 84, 123, 125, 128, 30]  # 6 FlexLoads
# Total: 7 + 7 + 6 = 20 agents

#: Tight voltage limits for DERs-large stress test
DERS_LARGE_V_MIN = 0.96
DERS_LARGE_V_MAX = 1.04


def make_ders_large_params(
    case=None,
    *,
    max_steps: int = 48,
    dt_hours: float = 0.5,
    v_min: float = DERS_LARGE_V_MIN,
    v_max: float = DERS_LARGE_V_MAX,
    pv_profiles: Optional[jnp.ndarray] = None,
) -> DistGridParams:
    """Create ``DistGridParams`` for the DERs-large scalability variant.

    20 agents on case141 with tighter voltage limits [0.96, 1.04]:
        - 7× Battery with Q-control (buses in ``DERS_LARGE_BATTERY_BUSES``)
        - 7× PV inverter
        - 6× FlexLoad
    """
    if case is None:
        case = _default_ders_case()

    n_pv = len(DERS_LARGE_PV_BUSES)
    if pv_profiles is None:
        pv_profiles = _default_pv_profiles(max_steps, n_pv)

    bat_bundle = make_battery_bundle(
        case,
        bus_ids=DERS_LARGE_BATTERY_BUSES,
        power_mw=DERS_BATTERY_CONFIG["power_mw"],
        capacity_mwh=DERS_BATTERY_CONFIG["capacity_mwh"],
        s_rated_mva=DERS_BATTERY_CONFIG["s_rated_mva"],
        enable_q_control=True,
        soc_min=DERS_BATTERY_CONFIG["soc_min"],
        soc_max=DERS_BATTERY_CONFIG["soc_max"],
        initial_soc=DERS_BATTERY_CONFIG["initial_soc"],
        dt_hours=dt_hours,
    )
    pv_bundle = make_renewable_bundle(
        case,
        bus_ids=DERS_LARGE_PV_BUSES,
        capacity_mw=DERS_PV_CONFIG["capacity_mw"],
        s_rated_mva=DERS_PV_CONFIG["s_rated_mva"],
        profiles=pv_profiles,
        allow_curtailment=True,
        dt_hours=dt_hours,
    )
    fl_bundle = make_flexload_bundle(
        case,
        bus_ids=DERS_LARGE_FLEXLOAD_BUSES,
        curtail_cap_mw=DERS_FLEXLOAD_CONFIG["curtail_cap_mw"],
        shift_cap_mw=DERS_FLEXLOAD_CONFIG["shift_cap_mw"],
        shift_horizon=DERS_FLEXLOAD_CONFIG["shift_horizon"],
        dt_hours=dt_hours,
    )

    return make_dist_params(
        case,
        max_steps=max_steps,
        steps_per_day=48,
        v_min=v_min,
        v_max=v_max,
        resources=(bat_bundle, pv_bundle, fl_bundle),
        include_der=False,
    )


def make_ders_large_marl_env(
    case=None,
    *,
    max_steps: int = 48,
    voltage_penalty: float = 12.0,
):
    """Create a :class:`~powerzoojax.rl.multi_agent.DistGridMARLEnv` for DERs-large.

    Returns ``(env, params)`` with 20 agents and tight voltage [0.96, 1.04].
    """
    from powerzoojax.rl.multi_agent import DistGridMARLEnv

    if case is None:
        case = _default_ders_case()

    params = make_ders_large_params(case, max_steps=max_steps)
    env = DistGridMARLEnv(
        DistGridEnv(), params,
        voltage_penalty=voltage_penalty,
        observation_mode="local",
    )
    return env, params


# ============ R5 — OOD scenario utilities ============

def make_ders_ood_params(
    base_params: DistGridParams,
    *,
    scenario: str,
    pv_scale: float = 1.0,
    v_min: Optional[float] = None,
    v_max: Optional[float] = None,
) -> DistGridParams:
    """Return a modified ``DistGridParams`` for an OOD stress test.

    Supported scenarios:

    ``"voltage_tightening"``
        Tighten voltage limits to [0.96, 1.04] (or provide explicit ``v_min``/``v_max``).

    ``"pv_penetration_shift"``
        Scale PV capacity by ``pv_scale`` (default 2.0 for double penetration).
        Requires ``pv_scale`` argument.  Replaces the RenewableBundle in-place.

    ``"load_stress"``
        Scale both load profiles by ``pv_scale`` (reused as general scale).

    Args:
        base_params: A ``DistGridParams`` returned by ``make_ders_params``.
        scenario: One of ``"voltage_tightening"``, ``"pv_penetration_shift"``,
            ``"load_stress"``.
        pv_scale: Scale factor for PV capacity / load, depending on scenario.
        v_min: Override voltage lower bound.
        v_max: Override voltage upper bound.

    Returns:
        Modified ``DistGridParams`` (only the targeted field is changed).
    """
    if scenario == "voltage_tightening":
        new_v_min = v_min if v_min is not None else DERS_LARGE_V_MIN
        new_v_max = v_max if v_max is not None else DERS_LARGE_V_MAX
        return base_params.replace(v_min=new_v_min, v_max=new_v_max)

    elif scenario == "pv_penetration_shift":
        # Scale up the PV bundle capacity (higher renewable penetration → more OV risk)
        bundles = list(base_params.resources)
        new_bundles = []
        for b in bundles:
            if isinstance(b, RenewableBundle):
                new_cap = b.capacity_mw * float(pv_scale if pv_scale != 1.0 else 2.0)
                b = b.replace(capacity_mw=new_cap)
            new_bundles.append(b)
        return base_params.replace(resources=tuple(new_bundles))

    elif scenario == "load_stress":
        scale = float(pv_scale if pv_scale != 1.0 else 1.15)
        return base_params.replace(
            load_profiles_p=base_params.load_profiles_p * scale,
            load_profiles_q=base_params.load_profiles_q * scale,
        )

    elif scenario == "phase_swap":
        raise NotImplementedError(
            "phase_swap OOD requires case123 three-phase parameter extraction. "
            "Blocker: CaseData does not store Z_3ph_pu matrices needed by "
            "make_dist_3phase_params().  Use make_ders_3phase_eval() for the "
            "scaffolded 3-phase evaluation harness (balanced diagonal approximation)."
        )

    else:
        raise ValueError(
            f"Unknown scenario: '{scenario}'. "
            "Choose from: 'voltage_tightening', 'pv_penetration_shift', 'load_stress', "
            "'phase_swap'."
        )


def agent_dropout_rollout(
    env,
    params,
    key,
    policy_fn=None,
    n_dropout: int = 2,
    seed: int = 0,
) -> dict:
    """Rollout with random agent dropout: N agents' actions are zeroed at test time.

    Simulates the scenario where some DER devices go offline unexpectedly.
    The default ``policy_fn`` is no-control (all zeros).

    Args:
        env: DistGridEnv instance.
        params: DistGridParams from make_ders_params.
        key: JAX PRNGKey.
        policy_fn: Base policy ``(obs, key) -> action``.  If None, uses no-control.
        n_dropout: Number of agents to disable (randomly chosen, reproducible via seed).
        seed: NumPy random seed for selecting which agents to drop.

    Returns:
        Result dict (same structure as ``rollout_ders``) plus
        ``"dropped_agent_indices"`` list.
    """
    n_act = sum(b.action_dim for b in params.resources)
    per_agent_act = 2   # all bundles use per_device_action_dim=2
    n_agents = n_act // per_agent_act

    rng = np.random.default_rng(seed)
    drop_agents = sorted(rng.choice(n_agents, size=n_dropout, replace=False).tolist())

    # Build mask: 1.0 for active agents, 0.0 for dropped
    mask = np.ones(n_act, dtype=np.float32)
    for a in drop_agents:
        mask[a * per_agent_act: (a + 1) * per_agent_act] = 0.0
    mask_jnp = jnp.asarray(mask)

    base_fn = policy_fn if policy_fn is not None else (
        lambda obs, k: jnp.zeros(n_act, dtype=jnp.float32)
    )

    def _masked_policy(obs, k):
        return base_fn(obs, k) * mask_jnp

    result = rollout_ders(env, params, key, _masked_policy)
    result["dropped_agent_indices"] = drop_agents
    return result


def compute_ders_metrics(
    rollout: dict,
    baseline: dict,
    *,
    v_min: float = DERS_V_MIN,
    v_max: float = DERS_V_MAX,
) -> dict:
    """Compute DERs evaluation metrics relative to a no-control baseline.

    Args:
        rollout: Result dict from rollout_ders.
        baseline: No-control result dict.
        v_min: Voltage lower bound for violation counting.  Defaults to
            ``DERS_V_MIN`` (0.94 p.u.).  Pass ``params.v_min`` when evaluating
            under OOD scenarios with tightened voltage bounds.
        v_max: Voltage upper bound.  Defaults to ``DERS_V_MAX`` (1.06 p.u.).

    Returns:
        Dict of scalar metrics:
            ``total_reward``, ``total_cost``, ``mean_p_loss_mw``,
            ``voltage_violation_steps``, ``loss_reduction_pct``,
            ``cost_reduction_pct``.
    """
    v_lo = v_min
    v_hi = v_max

    viol_steps = int(np.sum(
        (rollout["v_min_episode"] < v_lo) | (rollout["v_max_episode"] > v_hi)
    ))
    baseline_loss = float(np.mean(baseline["p_loss_MW"]))
    rl_loss = float(np.mean(rollout["p_loss_MW"]))
    loss_red = (baseline_loss - rl_loss) / max(baseline_loss, 1e-8) * 100.0

    baseline_cost = float(np.sum(baseline["cost_continuous"]))
    rl_cost = float(np.sum(rollout["cost_continuous"]))
    cost_red = (baseline_cost - rl_cost) / max(baseline_cost, 1e-8) * 100.0

    return {
        "total_reward": float(np.sum(rollout["reward"])),
        "total_cost": float(np.sum(rollout["cost_continuous"])),
        "mean_p_loss_mw": rl_loss,
        "voltage_violation_steps": viol_steps,
        "loss_reduction_pct": loss_red,
        "cost_reduction_pct": cost_red,
    }


# ============ R5 — DERs-3phase eval scaffold (case123) ============

def make_ders_3phase_eval(
    case=None,
    *,
    max_steps: int = 48,
    battery_buses: Optional[Sequence[int]] = None,
    v_min: float = DERS_V_MIN,
    v_max: float = DERS_V_MAX,
    voltage_penalty: float = 8.0,
    compat_obs_dim: Optional[int] = None,
):
    """Create a DistGrid3PhaseMARLEnv for DERs 3-phase OOD evaluation on case123.

    Builds the ``DERs-3phase-ood`` evaluation preset:
    case123 (114-bus), **6 battery agents**, local phase-averaged observations
    (≈15-dim, compatible with 1-phase local obs for zero-shot transfer).

    Uses **balanced diagonal** 3-phase impedance (Z_3ph_pu[i] = (r+jx)·I₃)
    built from CaseData scalar r/x.

    Known blocker: true phase-unbalance requires full per-phase Z_3ph_pu
    matrices.  CaseData stores only scalar line_r/line_x; off-diagonal
    impedance coupling from case123's ``line_config`` is discarded during
    ``build_case_from_tables()``.  The ``phase_swap`` OOD scenario remains
    blocked by this.  This scaffold uses the balanced-diagonal approximation.

    Returns:
        ``(env, params)`` where ``env`` is a
        :class:`~powerzoojax.rl.multi_agent.DistGrid3PhaseMARLEnv` with
        6 agents.  ``env._obs_dim`` equals ``compat_obs_dim`` (default 15),
        matching 1-phase medium/large so a trained 1-phase battery policy
        can run a forward pass on 3-phase observations without shape error.

    Args:
        case: CaseData for case123.  If None, ``create_case123()`` is called.
        max_steps: Episode length.
        battery_buses: 1-indexed bus IDs for battery placement (matching
            case123 node_ids 1–114).  Defaults to [10, 30, 50, 70, 90, 110]
            (6 agents, matching DERs-3phase-ood spec).
        v_min: Voltage lower bound [p.u.].
        v_max: Voltage upper bound [p.u.].
        voltage_penalty: Voltage cost weight in MARL reward shaping.
        compat_obs_dim: Target obs_dim for zero-shot compat with 1-phase policy.
            Defaults to 15 (= 1 + K + 3 + 2 + FlexLoad_dev_obs(5)) which is the
            obs_dim of DERs-medium and DERs-large.  Shortfall is zero-padded.
            Pass ``None`` to disable padding and use the natural 3-phase dim.
    """
    from powerzoojax.case import create_case123
    from powerzoojax.envs.grid.dist_3phase import DistGrid3PhaseEnv, make_dist_3phase_params
    from powerzoojax.envs.resource.battery import make_battery_bundle
    from powerzoojax.rl.multi_agent import DistGrid3PhaseMARLEnv, _LOCAL_K

    # Default: pad to match DERs-medium/large obs_dim (15 = 10 + FlexLoad_dev_obs=5)
    # so zero-shot transfer from a 1-phase battery policy does not hit shape mismatch.
    if compat_obs_dim is None:
        _FLEXLOAD_DEV_OBS = 5  # FlexLoadBundle.per_device_obs_dim in 1-phase medium
        compat_obs_dim = 1 + _LOCAL_K + 3 + 2 + _FLEXLOAD_DEV_OBS  # = 15

    if case is None:
        case = create_case123()

    if battery_buses is None:
        battery_buses = [10, 30, 50, 70, 90, 110]  # 6 agents for DERs-3phase-ood

    n_nodes = case.n_nodes
    n_lines = case.n_lines

    # 0-based internal indices from CaseData
    from_nodes = np.asarray(case.line_from_idx, dtype=np.int32)
    to_nodes   = np.asarray(case.line_to_idx,   dtype=np.int32)

    # Balanced diagonal Z_3ph from scalar r/x — balanced 3-phase approximation.
    line_r = np.asarray(case.line_r if case.line_r is not None else
                        np.zeros(n_lines), dtype=np.float64)
    line_x_arr = np.asarray(case.line_x if case.line_x is not None else
                            np.ones(n_lines) * 0.01, dtype=np.float64)
    Z_3ph_pu = np.zeros((n_lines, 3, 3), dtype=complex)
    for i in range(n_lines):
        Z_3ph_pu[i] = (line_r[i] + 1j * line_x_arr[i]) * np.eye(3)

    base_mva = float(case.base_mva) if case.base_mva is not None else 10.0
    n_nonref = n_nodes - 1   # number of non-ref buses

    # Per-phase load arrays: interleaved [A0, B0, C0, A1, B1, C1, ...]
    # CaseData.node_pd_a has shape (n_nodes,), index 0 = ref bus (slack).
    p_per_node = np.zeros((n_nonref, 3), dtype=np.float32)
    q_per_node = np.zeros((n_nonref, 3), dtype=np.float32)
    if getattr(case, "node_pd_a", None) is not None:
        p_per_node[:, 0] = np.asarray(case.node_pd_a)[1:] / base_mva
        p_per_node[:, 1] = np.asarray(case.node_pd_b)[1:] / base_mva
        p_per_node[:, 2] = np.asarray(case.node_pd_c)[1:] / base_mva
        q_per_node[:, 0] = np.asarray(case.node_qd_a)[1:] / base_mva
        q_per_node[:, 1] = np.asarray(case.node_qd_b)[1:] / base_mva
        q_per_node[:, 2] = np.asarray(case.node_qd_c)[1:] / base_mva
    elif getattr(case, "node_pd", None) is not None:
        total_p = np.asarray(case.node_pd)[1:] / base_mva
        total_q = (np.asarray(case.node_qd)[1:] / base_mva
                   if getattr(case, "node_qd", None) is not None
                   else np.zeros(n_nonref))
        p_per_node[:] = (total_p / 3.0)[:, None]
        q_per_node[:] = (total_q / 3.0)[:, None]

    # C-order reshape: (n_nonref, 3) → (3*n_nonref,) = [A0, B0, C0, A1, B1, C1, ...]
    load_p_base = jnp.asarray(p_per_node.reshape(-1), dtype=jnp.float32)
    load_q_base = jnp.asarray(q_per_node.reshape(-1), dtype=jnp.float32)
    load_P = jnp.tile(load_p_base[None, :], (max_steps, 1))
    load_Q = jnp.tile(load_q_base[None, :], (max_steps, 1))

    # Battery bundle — battery_buses are 1-indexed node IDs for case123
    bat_bundle = make_battery_bundle(
        case,
        bus_ids=list(battery_buses),
        power_mw=DERS_BATTERY_CONFIG["power_mw"],
        capacity_mwh=DERS_BATTERY_CONFIG["capacity_mwh"],
        s_rated_mva=DERS_BATTERY_CONFIG["s_rated_mva"],
        enable_q_control=True,
        soc_min=DERS_BATTERY_CONFIG["soc_min"],
        soc_max=DERS_BATTERY_CONFIG["soc_max"],
        initial_soc=DERS_BATTERY_CONFIG["initial_soc"],
    )

    params = make_dist_3phase_params(
        n_nodes=n_nodes,
        from_nodes=from_nodes,
        to_nodes=to_nodes,
        Z_3ph_pu=Z_3ph_pu,
        load_P_3ph=load_P,
        load_Q_3ph=load_Q,
        v_min=v_min,
        v_max=v_max,
        max_steps=max_steps,
        base_mva=base_mva,
        resources=(bat_bundle,),
        include_der=False,
    )

    dist3ph_env = DistGrid3PhaseEnv()
    env = DistGrid3PhaseMARLEnv(
        dist3ph_env, params,
        voltage_penalty=voltage_penalty,
        min_obs_dim=int(compat_obs_dim) if compat_obs_dim is not None else 0,
    )
    return env, params


# ============ E — Data-driven DERs factory hook ============

def make_ders_params_with_profiles(
    case=None,
    *,
    max_steps: int = 48,
    dt_hours: float = 0.5,
    v_min: float = DERS_V_MIN,
    v_max: float = DERS_V_MAX,
    loss_penalty_weight: float = 0.1,
    pv_profiles: Optional[jnp.ndarray] = None,
    load_profiles_p: Optional[jnp.ndarray] = None,
    load_profiles_q: Optional[jnp.ndarray] = None,
) -> DistGridParams:
    """Create DERs params from externally-supplied time-series profiles.

    Thin wrapper around :func:`make_ders_params` that accepts real data arrays
    (e.g. from Ausgrid substations or GB renewable traces) as profiles.

    Default is still synthetic (bell-curve PV, flat load) if profiles are None.
    This hook lets callers wire in :func:`powerzoojax.data.ausgrid_utils`
    outputs without coupling the data layer into the env core.

    Args:
        case: CaseData.  If None, :func:`_default_ders_case` is used.
        max_steps: Episode length.
        dt_hours: Step duration [h].
        v_min / v_max: Voltage limits.
        loss_penalty_weight: Loss penalty weight.
        pv_profiles: Capacity factor array ``(max_steps, 4)`` for 4 PV devices.
            None → synthetic bell-curve.
        load_profiles_p: Active load array ``(max_steps, n_nodes)`` [p.u.].
            None → flat base-load from case.
        load_profiles_q: Reactive load array.  None → flat.

    Returns:
        :class:`~powerzoojax.envs.grid.dist.DistGridParams` for DERs.
    """
    return make_ders_params(
        case,
        max_steps=max_steps,
        dt_hours=dt_hours,
        v_min=v_min,
        v_max=v_max,
        loss_penalty_weight=loss_penalty_weight,
        pv_profiles=pv_profiles,
        load_profiles_p=load_profiles_p,
        load_profiles_q=load_profiles_q,
    )


# ============ TaskSpec implementation ============

def rollout_ders_marl(env_marl, params, key, policy_fn):
    """Single-episode lax.scan MARL rollout for IPPO DERs policy.

    Args:
        env_marl: DistGridMARLEnv (params embedded; reset/step take no params).
        params: DistGridParams — used only to read max_steps.
        key: JAX PRNGKey.
        policy_fn: Callable ``obs_dict -> actions_dict``.

    Returns:
        Dict with numpy arrays ``reward``, ``cost_continuous``, ``p_loss_MW``,
        ``v_min_episode``, ``v_max_episode``, each shape ``(max_steps,)``.
    """
    obs0, state0 = env_marl.reset(key)
    max_steps = int(params.max_steps)

    def body(carry, _):
        obs_dict, state, k = carry
        k, k_step = jax.random.split(k)
        actions = policy_fn(obs_dict)
        obs2, state2, rewards_d, _, info = env_marl.step(k_step, state, actions)
        out = {
            "reward": jnp.sum(jnp.stack(list(rewards_d.values()))),
            "cost_continuous": jnp.asarray(info.get("cost_continuous", 0.0)),
            "p_loss_MW": jnp.asarray(info.get("p_loss_MW", 0.0)),
            "v_min_episode": jnp.asarray(
                info.get("v_min_step", jnp.min(state2.grid_state.v_mag))
            ),
            "v_max_episode": jnp.asarray(
                info.get("v_max_step", jnp.max(state2.grid_state.v_mag))
            ),
        }
        return (obs2, state2, k), out

    _, traj = jax.lax.scan(body, (obs0, state0, key), xs=None, length=max_steps)
    # Keep rollout outputs as JAX arrays so eval can JIT the full episode.
    traj = dict(traj)
    traj["metric_v_min"] = jnp.asarray(params.v_min, dtype=jnp.float32)
    traj["metric_v_max"] = jnp.asarray(params.v_max, dtype=jnp.float32)
    return traj


class DERsTask:
    """TaskSpec for the DERs benchmark (case141, 12 heterogeneous DER agents).

    The MARL env embeds params at construction time, so ``make_env(split)``
    returns a split-specific ``DistGridMARLEnv``.  ``episode_params()``
    returns the embedded ``DistGridParams`` for use by baselines that call
    ``DistGridEnv`` (single-agent) directly.
    """

    task_name = "ders"
    default_splits: tuple = (
        "train",
        "iid",
        "voltage_tightening",
        "pv_penetration_shift",
        "load_stress",
    )

    def __init__(
        self,
        *,
        case: Optional[CaseData] = None,
        v_min: float = DERS_V_MIN,
        v_max: float = DERS_V_MAX,
        voltage_penalty: float = 8.0,
        max_steps: int = 48,
    ):
        self._case = case
        self._v_min = v_min
        self._v_max = v_max
        self._voltage_penalty = voltage_penalty
        self._max_steps = max_steps
        # _cache[split] stores split-level static assets. Episode windows are
        # derived lazily from the cached profiles so repeated eval episodes
        # do not collapse onto one fixed params instance.
        self._cache: dict[str, dict[str, object]] = {}

    @staticmethod
    def _split_spec(split: str) -> tuple[str, str | None]:
        ood_map = {
            "voltage_tightening": "voltage_tightening",
            "pv_penetration_shift": "pv_penetration_shift",
            "pv_penetration": "pv_penetration_shift",
            "load_stress": "load_stress",
        }
        scenario = ood_map.get(split)
        base_role = "iid" if scenario is not None else split
        return base_role, scenario

    def _ensure_cache(self, split: str):
        if split in self._cache:
            return
        base_role, scenario = self._split_spec(split)
        case = self._case if self._case is not None else _default_ders_case()
        base_params = make_ders_params(
            case,
            max_steps=self._max_steps,
            v_min=self._v_min,
            v_max=self._v_max,
        )
        load_shape, pv_profile = load_ders_split_profiles(role=base_role)
        self._cache[split] = {
            "case": case,
            "base_params": base_params,
            "base_role": base_role,
            "ood_scenario": scenario,
            "load_shape": load_shape,
            "pv_profile": pv_profile,
            "total_profile_len": min(len(load_shape), len(pv_profile)),
            "env": None,
            "env_params_start": None,
        }

    def _episode_params_from_cache(
        self,
        split: str,
        episode_idx: int,
        n_episodes: int,
        *,
        strategy: str,
        seed: int,
    ) -> DistGridParams:
        self._ensure_cache(split)
        cached = self._cache[split]
        total_len = int(cached["total_profile_len"])
        episode_start = select_ders_episode_start(
            total_len,
            max_steps=self._max_steps,
            episode_idx=episode_idx,
            n_episodes=n_episodes,
            strategy=strategy,
            seed=seed,
            salt=17,
        )
        params = apply_ders_profile_window(
            cached["base_params"],
            load_shape=cached["load_shape"],
            pv_profile=cached["pv_profile"],
            episode_start=episode_start,
        )
        scenario = cached["ood_scenario"]
        if scenario is not None:
            params = make_ders_ood_params(params, scenario=scenario)
        return params

    def total_profile_len(self, split: str) -> int:
        """Return the split's underlying 1-D profile length before windowing."""
        self._ensure_cache(split)
        return int(self._cache[split]["total_profile_len"])

    def episode_start(
        self,
        split: str,
        episode_idx: int,
        n_episodes: int,
        *,
        strategy: str = "uniform",
        seed: int = 0,
    ) -> int:
        """Deterministic profile start used for a given episode index."""
        self._ensure_cache(split)
        return select_ders_episode_start(
            int(self._cache[split]["total_profile_len"]),
            max_steps=self._max_steps,
            episode_idx=episode_idx,
            n_episodes=n_episodes,
            strategy=strategy,
            seed=seed,
            salt=17,
        )

    def params_from_start(
        self,
        split: str,
        episode_start: int,
    ) -> DistGridParams:
        """Build split params for an explicit profile window start."""
        self._ensure_cache(split)
        cached = self._cache[split]
        params = apply_ders_profile_window(
            cached["base_params"],
            load_shape=cached["load_shape"],
            pv_profile=cached["pv_profile"],
            episode_start=episode_start,
        )
        scenario = cached["ood_scenario"]
        if scenario is not None:
            params = make_ders_ood_params(params, scenario=scenario)
        return params

    def make_env(self, split: str = "train"):
        self._ensure_cache(split)
        cached = self._cache[split]
        env = cached["env"]
        if env is not None:
            return env
        params = self._episode_params_from_cache(
            split,
            episode_idx=0,
            n_episodes=1,
            strategy="uniform",
            seed=0,
        )
        from powerzoojax.rl.multi_agent import DistGridMARLEnv
        env = DistGridMARLEnv(
            DistGridEnv(),
            params,
            voltage_penalty=self._voltage_penalty,
            observation_mode="local",
        )
        cached["env"] = env
        cached["env_params_start"] = 0
        return env

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
        return self._episode_params_from_cache(
            split,
            episode_idx=episode_idx,
            n_episodes=n_episodes,
            strategy=strategy,
            seed=seed,
        )

    def rollout(self, env, params, key, policy_fn):
        return rollout_ders_marl(env, params, key, policy_fn)

    def baseline_rollout(self, env, params, key, baseline_name: str):
        single_env = DistGridEnv()
        if baseline_name == "no_control":
            return ders_no_control_rollout(single_env, params, key)
        elif baseline_name == "volt_droop":
            return ders_volt_droop_rollout(single_env, params, key)
        else:
            raise ValueError(f"Unknown DERs baseline: {baseline_name!r}")

    def compute_metrics(self, agent_rollout, baseline_rollout) -> dict:
        v_min = float(np.asarray(agent_rollout.get("metric_v_min", self._v_min)))
        v_max = float(np.asarray(agent_rollout.get("metric_v_max", self._v_max)))
        metrics = compute_ders_metrics(
            agent_rollout, baseline_rollout,
            v_min=v_min, v_max=v_max,
        )
        metrics.update(
            compute_ders_safety_metrics(
                agent_rollout,
                v_min=v_min,
                v_max=v_max,
            )
        )
        return metrics

    def baseline_names(self) -> tuple:
        return ("no_control", "volt_droop")

    def constraint_spec(self) -> ConstraintSpec:
        return ConstraintSpec(
            selected_names=("voltage_violation", "thermal_overload", "resource"),
            thresholds=(0.0, 0.0, 0.0),
            fallback_weights=(4.0, 1.0, 1.0),
        )
