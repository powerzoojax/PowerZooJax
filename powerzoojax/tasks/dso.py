"""DSO (Distribution System Operator) task environment.

Wraps ``DistGridEnv`` + ``FlexLoadBundle`` on case33bw to create the DSO
network-loss-minimisation benchmark.

**Load profiles**: ``feeder_shapes=None`` uses *synthetic* diurnal profiles
(dev / testing only).  For the production benchmark, pass real Ausgrid shapes
built via ``load_feeder_shape()``.  Presets tagged ``[synthetic load,
dev/test only]`` use the synthetic path.

Main entry-points
-----------------
* ``load_dso_feeder_shapes(data_loader, role)`` — load all 3 real Ausgrid feeder shapes
* ``make_dso_flexload_bundle(case)`` — 6-device FlexLoad bundle (``DSO_FLEXLOAD_CONFIG``)
* ``make_dso_load_profiles(case, feeder_shapes, ...)`` — map feeder shapes to
  per-bus (T, n_bus) load profiles in p.u.
* ``load_feeder_shape(data_loader, feeder, role)`` — load real Ausgrid data
* ``make_dso_params_from_split(case, role, ...)`` — canonical real-data split factory
* ``make_dso_params(case, ...)`` — one-call factory returning ``DistGridParams``
* ``make_dso_1flex_params(case, ...)`` — single-device 1-flex variant factory
* ``make_dso_params_nonstationary(...)`` — factory with drift + rolling
* ``rollout_dso(env, params, key, policy_fn)`` — single-episode rollout
* ``dso_no_control_rollout(...)`` — no-control baseline
* ``dso_tou_rule_based_rollout(...)`` — time-of-use (TOU) **rule-based** baseline (fixed peak window)
* ``dso_droop_rule_based_rollout(...)`` — volt–var / droop-style **rule-based** baseline (local voltage feedback)
* ``dso_tou_heuristic_rollout`` / ``dso_droop_heuristic_rollout`` — legacy aliases, same callables
* ``compute_dso_metrics(rollout_info)`` — DSO metric computation

All functions run at setup-time on CPU.  The returned ``DistGridParams`` is
used with the existing ``DistGridEnv`` — no new env class is needed.

Constraint selection
--------------------
The DSO benchmark consumes only the ``"voltage_violation"`` constraint from
the full vector exposed by ``DistGridEnv``. The legacy ``cost_mode`` argument
is retained only for backward-compatible config loading; the core env always
returns the full constraint vector.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import jax.numpy as jnp

from powerzoojax.case.case_data import CaseData
from powerzoojax.envs.grid.dist import DistGridParams, make_dist_params
from powerzoojax.envs.resource.flexload import FlexLoadBundle, make_flexload_bundle
from powerzoojax.tasks.base import ConstraintSpec

# Feeder-to-bus mapping for case33bw (IEEE 33-bus Baran-Wu)
# Bus 1 is the slack/substation bus and belongs to no feeder.
DSO_FEEDER_BUS_MAP: Dict[str, List[int]] = {
    "feeder_A": list(range(2, 19)),   # buses 2-18  (17 buses)
    "feeder_B": list(range(19, 23)),  # buses 19-22 (4 buses)
    "feeder_C": list(range(23, 34)),  # buses 23-33 (11 buses)
}

# DSO FlexLoad configuration (six devices on case33bw)
DSO_FLEXLOAD_CONFIG: List[Dict] = [
    {"bus_id": 6,  "curtail_cap_mw": 0.15, "shift_cap_mw": 0.15},  # FL_A1
    {"bus_id": 14, "curtail_cap_mw": 0.10, "shift_cap_mw": 0.10},  # FL_A2
    {"bus_id": 18, "curtail_cap_mw": 0.10, "shift_cap_mw": 0.10},  # FL_A3
    {"bus_id": 22, "curtail_cap_mw": 0.08, "shift_cap_mw": 0.08},  # FL_B1
    {"bus_id": 28, "curtail_cap_mw": 0.12, "shift_cap_mw": 0.12},  # FL_C1
    {"bus_id": 33, "curtail_cap_mw": 0.10, "shift_cap_mw": 0.10},  # FL_C2
]

# Voltage limits for DSO task (tighter than DistGridEnv defaults)
DSO_V_MIN = 0.94
DSO_V_MAX = 1.06


def _normalize_bus_load_scale_overrides(
    bus_load_scale_overrides: Optional[Dict[int, float]],
) -> Optional[Dict[int, float]]:
    """Coerce YAML / JSON bus-scale overrides to ``{int: float}``.

    Values must stay strictly positive because they scale physical base loads.
    """
    if bus_load_scale_overrides is None:
        return None

    normalized: Dict[int, float] = {}
    for raw_bus_id, raw_scale in bus_load_scale_overrides.items():
        bus_id = int(raw_bus_id)
        scale = float(raw_scale)
        if scale <= 0.0:
            raise ValueError(
                f"bus_load_scale_overrides[{bus_id}] must be > 0, got {scale}"
            )
        normalized[bus_id] = scale
    return normalized


def resolve_dso_flexload_config(
    *,
    flexload_config: Optional[List[Dict]] = None,
    flexload_bus_ids: Optional[List[int]] = None,
) -> List[Dict]:
    """Resolve the active FlexLoad layout for a DSO scenario.

    ``flexload_config`` takes priority and should contain full per-device dicts.
    ``flexload_bus_ids`` is a light-weight convenience override: it keeps the
    default device capacities and only swaps the attachment buses.
    """
    if flexload_config is not None and flexload_bus_ids is not None:
        raise ValueError("Pass either flexload_config or flexload_bus_ids, not both.")
    if flexload_config is not None:
        return [
            {
                "bus_id": int(cfg["bus_id"]),
                "curtail_cap_mw": float(cfg["curtail_cap_mw"]),
                "shift_cap_mw": float(cfg["shift_cap_mw"]),
            }
            for cfg in flexload_config
        ]
    if flexload_bus_ids is not None:
        if len(flexload_bus_ids) != len(DSO_FLEXLOAD_CONFIG):
            raise ValueError(
                "flexload_bus_ids must match the default number of DSO devices "
                f"({len(DSO_FLEXLOAD_CONFIG)}), got {len(flexload_bus_ids)}."
            )
        return [
            {
                **cfg,
                "bus_id": int(bus_id),
            }
            for cfg, bus_id in zip(DSO_FLEXLOAD_CONFIG, flexload_bus_ids)
        ]
    return list(DSO_FLEXLOAD_CONFIG)


def dso_task_kwargs_from_config(task_config: Dict) -> Dict:
    """Extract runtime DSO task kwargs from a task config dict."""
    kwargs: Dict = {
        "load_scale": float(task_config.get("load_scale", 1.0)),
        "v_min": float(task_config.get("v_min", DSO_V_MIN)),
        "v_max": float(task_config.get("v_max", DSO_V_MAX)),
        "v_slack": float(task_config.get("v_slack", 1.0)),
        "cost_mode": str(task_config.get("cost_mode", "voltage_only")),
        "shift_horizon": int(task_config.get("shift_horizon", 4)),
        "preserve_feeder_totals": bool(
            task_config.get("preserve_feeder_totals", False)
        ),
    }
    if "flexload_config" in task_config:
        kwargs["flexload_config"] = task_config["flexload_config"]
    elif "flexload_bus_ids" in task_config:
        kwargs["flexload_bus_ids"] = task_config["flexload_bus_ids"]

    bus_load_scale_overrides = _normalize_bus_load_scale_overrides(
        task_config.get("bus_load_scale_overrides")
    )
    if bus_load_scale_overrides:
        kwargs["bus_load_scale_overrides"] = bus_load_scale_overrides
    return kwargs


# D1: Feeder-shape mapping
def load_feeder_shape(
    data_loader,
    feeder: str,
    role: str = "train",
    *,
    resample: str = "30min",
) -> np.ndarray:
    """Load and average Ausgrid substation profiles for one feeder.

    Returns 1-D numpy array of shape multipliers normalised so mean = 1.

    Args:
        data_loader: ``DataLoader`` instance.
        feeder: ``"feeder_A"`` / ``"feeder_B"`` / ``"feeder_C"``.
        role: ``"train"`` / ``"iid"`` / ``"summer_ood"`` / ``"zone_holdout"``.
        resample: Target resolution (``"30min"`` for 48-step episodes).

    Returns:
        1-D ``np.ndarray`` of length T (number of resampled time steps).
    """
    from powerzoojax.data.ausgrid_utils import (
        filter_ausgrid_role_days,
        get_ausgrid_split,
        get_feeder_substations,
    )

    # Determine which substations to use
    pool_role = "zone_holdout" if role == "zone_holdout" else "train"
    substations = get_feeder_substations(feeder, pool_role)

    # Determine time window
    start, end, _ = get_ausgrid_split(role)

    profiles = []
    for sub in substations:
        if hasattr(data_loader, "load_signals"):
            df = data_loader.load_signals(
                ["load.actual_mw"],
                source="ausgrid",
                region=sub,
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
                region=sub,
                start_date=start,
                end_date=end,
                resample=resample,
            )
            profiles.append(np.asarray(arr).flatten())

    # Average across substations, truncate to shortest
    min_len = min(len(p) for p in profiles)
    stacked = np.stack([p[:min_len] for p in profiles])
    avg = stacked.mean(axis=0)

    # Normalise to mean = 1 (shape profile)
    shape = avg / np.maximum(avg.mean(), 1e-8)
    return shape.astype(np.float32)


def load_dso_feeder_shapes(
    data_loader=None,
    role: str = "train",
    *,
    resample: str = "30min",
) -> Dict[str, np.ndarray]:
    """Load all three DSO feeder shapes for a named Ausgrid split.

    This is the canonical real-data entry-point for DSO experiments.  It keeps
    the split logic in one place so callers do not need to manually assemble
    ``{"feeder_A": ..., "feeder_B": ..., "feeder_C": ...}`` each time.

    Args:
        data_loader: Optional ``DataLoader`` instance.  ``None`` → create one.
        role: One of ``"train"``, ``"iid"``, ``"summer_ood"``,
            ``"zone_holdout"``.
        resample: Target resolution (default ``"30min"`` for 48-step episodes).

    Returns:
        Dict mapping feeder name to a mean-normalised 1-D shape array.
    """
    if data_loader is None:
        from powerzoojax.data.data_loader import DataLoader
        data_loader = DataLoader()

    return {
        feeder: load_feeder_shape(
            data_loader,
            feeder,
            role=role,
            resample=resample,
        )
        for feeder in DSO_FEEDER_BUS_MAP
    }


def concat_dso_feeder_windows(
    feeder_shapes: Dict[str, np.ndarray],
    window_starts: List[int],
    *,
    window_len: int = 48,
) -> Dict[str, np.ndarray]:
    """Concatenate selected fixed-length windows from feeder shapes.

    Useful for training on a curated set of day-like windows without changing
    the underlying physical model or evaluation horizon.
    """
    if window_len <= 0:
        raise ValueError(f"window_len must be > 0, got {window_len}")
    if not window_starts:
        raise ValueError("window_starts must be non-empty")

    concatenated: Dict[str, np.ndarray] = {}
    for feeder, raw_shape in feeder_shapes.items():
        arr = np.asarray(raw_shape, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            raise ValueError(f"feeder {feeder!r} has empty shape array")
        segments = []
        for start in window_starts:
            idx = (np.arange(window_len) + int(start)) % arr.size
            segments.append(arr[idx])
        concatenated[feeder] = np.concatenate(segments, axis=0).astype(np.float32)
    return concatenated


def make_dso_load_profiles(
    case: CaseData,
    feeder_shapes: Dict[str, np.ndarray],
    *,
    max_steps: int = 48,
    episode_start: int = 0,
    load_scale: float = 1.0,
    bus_load_scale_overrides: Optional[Dict[int, float]] = None,
    preserve_feeder_totals: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Map per-feeder temporal shapes to per-bus (T, n_bus) load profiles.

    Each bus's load at time t = base_load[bus] * feeder_shape[t].
    Bus 1 (slack) keeps its base load constant (shape = 1).

    Args:
        case: CaseData (must be case33bw-like with 33 nodes, IDs 1-33).
        feeder_shapes: Mapping ``{"feeder_A": shape_A, ...}``.
            Each value is a 1-D array; a window of ``max_steps`` starting
            at ``episode_start`` is extracted (with wraparound).
        max_steps: Episode length in time steps.
        episode_start: Offset into the shape arrays for this episode.
        load_scale: Uniform multiplier on the case's static P/Q loads before
            feeder-shape expansion. Useful when calibrating the feeder-shape
            mapping to a more realistic total demand level.
        bus_load_scale_overrides: Optional per-bus multipliers applied to the
            static base loads before feeder shapes are expanded. Useful for
            task-level calibration of the spatial load split.
        preserve_feeder_totals: If True, re-normalise the adjusted base loads
            within each feeder so every feeder keeps its original total P/Q.

    Returns:
        ``(load_profiles_p, load_profiles_q)`` each ``(max_steps, n_bus)``
        in per-unit (divided by ``base_mva``).
    """
    if load_scale <= 0.0:
        raise ValueError(f"load_scale must be > 0, got {load_scale}")

    base_mva = float(case.base_mva) if case.base_mva is not None else 100.0
    node_ids = np.asarray(case.node_ids, dtype=np.float32)
    p_base_mw = np.asarray(case.node_pd, dtype=np.float32) * float(load_scale)
    q_base_mvar = np.asarray(case.node_qd, dtype=np.float32) * float(load_scale)
    p_target_mw = p_base_mw.copy()
    q_target_mvar = q_base_mvar.copy()

    bus_load_scale_overrides = _normalize_bus_load_scale_overrides(
        bus_load_scale_overrides
    )
    if bus_load_scale_overrides:
        p_base_mw = p_base_mw.copy()
        q_base_mvar = q_base_mvar.copy()
        for bus_id, scale in bus_load_scale_overrides.items():
            matches = np.where(np.abs(node_ids - bus_id) < 0.5)[0]
            if len(matches) > 0:
                idx = int(matches[0])
                p_base_mw[idx] *= scale
                q_base_mvar[idx] *= scale

        if preserve_feeder_totals:
            for bus_ids in DSO_FEEDER_BUS_MAP.values():
                feeder_idx: List[int] = []
                for bus_id in bus_ids:
                    matches = np.where(np.abs(node_ids - bus_id) < 0.5)[0]
                    if len(matches) > 0:
                        feeder_idx.append(int(matches[0]))
                if not feeder_idx:
                    continue
                p_before = float(np.sum(p_target_mw[feeder_idx]))
                q_before = float(np.sum(q_target_mvar[feeder_idx]))
                p_after = float(np.sum(p_base_mw[feeder_idx]))
                q_after = float(np.sum(q_base_mvar[feeder_idx]))
                if p_after > 1e-8:
                    p_base_mw[feeder_idx] *= p_before / p_after
                if q_after > 1e-8:
                    q_base_mvar[feeder_idx] *= q_before / q_after

    p_base_pu = p_base_mw / base_mva  # (n_bus,)
    q_base_pu = q_base_mvar / base_mva  # (n_bus,)
    n_bus = len(p_base_pu)

    # Build per-bus shape multiplier (T, n_bus)
    shape_matrix = np.ones((max_steps, n_bus), dtype=np.float32)

    for feeder_name, bus_ids in DSO_FEEDER_BUS_MAP.items():
        raw_shape = feeder_shapes.get(feeder_name)
        if raw_shape is None:
            continue
        # Extract window with wraparound
        T_total = len(raw_shape)
        indices = np.arange(max_steps) + episode_start
        if T_total > 0:
            indices = indices % T_total
        shape_window = raw_shape[indices]  # (max_steps,)

        for bus_id in bus_ids:
            # Find internal index
            matches = np.where(np.abs(node_ids - bus_id) < 0.5)[0]
            if len(matches) > 0:
                idx = int(matches[0])
                shape_matrix[:, idx] = shape_window

    load_p = shape_matrix * p_base_pu[None, :]  # (T, n_bus) p.u.
    load_q = shape_matrix * q_base_pu[None, :]  # (T, n_bus) p.u.

    return jnp.asarray(load_p, dtype=jnp.float32), jnp.asarray(load_q, dtype=jnp.float32)


def make_synthetic_feeder_shapes(
    max_steps: int = 48,
    *,
    peak_ratio: float = 1.5,
    peak_hour_start: int = 17,
    peak_hour_end: int = 21,
    steps_per_day: int = 48,
) -> Dict[str, np.ndarray]:
    """Create synthetic diurnal feeder shapes for testing without real data.

    Returns dict of 3 feeder shapes, each a 1-D array of length
    ``max_steps`` normalised to mean ~1.  The shapes have a realistic
    diurnal peak in the evening.

    Feeder A: residential evening peak
    Feeder B: flatter industrial profile
    Feeder C: coastal with midday + evening twin peaks
    """
    t = np.arange(max_steps, dtype=np.float32)
    hour = (t / steps_per_day) * 24.0  # fractional hour of day

    # Feeder A: single evening peak
    base_a = 1.0 + (peak_ratio - 1.0) * np.exp(-0.5 * ((hour - 18.5) / 2.0) ** 2)
    base_a = base_a / base_a.mean()

    # Feeder B: flatter, slight morning peak
    base_b = 1.0 + 0.3 * np.exp(-0.5 * ((hour - 10.0) / 3.0) ** 2)
    base_b = base_b / base_b.mean()

    # Feeder C: twin peaks (midday + evening)
    base_c = (1.0
              + 0.4 * np.exp(-0.5 * ((hour - 12.0) / 2.0) ** 2)
              + 0.5 * np.exp(-0.5 * ((hour - 19.0) / 1.5) ** 2))
    base_c = base_c / base_c.mean()

    return {
        "feeder_A": base_a.astype(np.float32),
        "feeder_B": base_b.astype(np.float32),
        "feeder_C": base_c.astype(np.float32),
    }


# D2: DSO FlexLoad bundle factory
def make_dso_flexload_bundle(
    case: CaseData,
    *,
    shift_horizon: int = 4,
    curtail_cost_per_mwh: float = 50.0,
    shift_cost_per_mwh: float = 10.0,
    dt_hours: float = 0.5,
) -> FlexLoadBundle:
    """Create the standard 6-device FlexLoad bundle for DSO task.

    Uses bus IDs and capacities from ``DSO_FLEXLOAD_CONFIG``.

    Args:
        case: CaseData for case33bw (must contain bus IDs 6,14,18,22,28,33).
        shift_horizon: Deferred-release horizon in time steps (default 4 = 2h).
        curtail_cost_per_mwh: Curtailment discomfort cost [$/MWh].
        shift_cost_per_mwh: Shift holding cost [$/MWh].
        dt_hours: Time-step duration [h].

    Returns:
        ``FlexLoadBundle`` with 6 devices, action_dim = 12, obs_dim = 30.
    """
    bus_ids = [cfg["bus_id"] for cfg in DSO_FLEXLOAD_CONFIG]
    curtail_caps = [cfg["curtail_cap_mw"] for cfg in DSO_FLEXLOAD_CONFIG]
    shift_caps = [cfg["shift_cap_mw"] for cfg in DSO_FLEXLOAD_CONFIG]

    return make_flexload_bundle(
        case,
        bus_ids=bus_ids,
        curtail_cap_mw=curtail_caps,
        shift_cap_mw=shift_caps,
        shift_horizon=shift_horizon,
        curtail_cost_per_mwh=curtail_cost_per_mwh,
        shift_cost_per_mwh=shift_cost_per_mwh,
        dt_hours=dt_hours,
    )


# D2: DSO params factory (one-call preset)
def make_dso_params(
    case: Optional[CaseData] = None,
    *,
    feeder_shapes: Optional[Dict[str, np.ndarray]] = None,
    max_steps: int = 48,
    steps_per_day: int = 48,
    episode_start: int = 0,
    load_scale: float = 1.0,
    v_slack: float = 1.0,
    v_min: float = DSO_V_MIN,
    v_max: float = DSO_V_MAX,
    loss_penalty_weight: float = 0.1,
    shift_horizon: int = 4,
    curtail_cost_per_mwh: float = 50.0,
    shift_cost_per_mwh: float = 10.0,
    dt_hours: float = 0.5,
    flexload_config: Optional[List[Dict]] = None,
    flexload_bus_ids: Optional[List[int]] = None,
    bus_load_scale_overrides: Optional[Dict[int, float]] = None,
    preserve_feeder_totals: bool = False,
    cost_mode: str = "voltage_only",
) -> DistGridParams:
    """One-call factory for the DSO task preset.

    Creates ``DistGridParams`` with:
    - case33bw (default) as the physical grid
    - FlexLoad bundle at configurable buses (default: six devices from ``DSO_FLEXLOAD_CONFIG``)
    - Ausgrid-driven (or synthetic) load profiles
    - DSO voltage limits [0.94, 1.06]
    - ``include_der=False`` (agent controls only FlexLoads, no legacy DER)
    - legacy ``cost_mode`` preserved only for backward-compatible config loading

    Args:
        case: CaseData.  If None, creates case33bw automatically.
        feeder_shapes: Per-feeder temporal shape dicts.  If None, uses
            **synthetic** diurnal profiles (dev / testing only — not the
            Ausgrid-driven production path).  Pass shapes built via
            ``load_feeder_shape()`` for the real benchmark.
        max_steps: Episode length (default 48 = 24h @ 30min).
        steps_per_day: Steps per day for time encoding.
        episode_start: Starting offset into feeder shape arrays.
        load_scale: Uniform multiplier on the case's static P/Q loads.
        v_slack: Slack/substation voltage magnitude [p.u.].
        v_min: Minimum voltage [p.u.] (default 0.94).
        v_max: Maximum voltage [p.u.] (default 1.06).
        loss_penalty_weight: Reward = -weight * loss_MW.
        shift_horizon: FlexLoad deferred-release horizon [steps].
        curtail_cost_per_mwh: FlexLoad curtailment cost.
        shift_cost_per_mwh: FlexLoad shift holding cost.
        dt_hours: Time-step duration [h].
        flexload_config: List of per-device dicts with keys
            ``bus_id``, ``curtail_cap_mw``, ``shift_cap_mw``.  If None,
            uses ``DSO_FLEXLOAD_CONFIG`` (six-device default).
            Use a single-element list for the 1-flex variant, e.g.
            ``[{"bus_id": 18, "curtail_cap_mw": 0.10, "shift_cap_mw": 0.10}]``.
        flexload_bus_ids: Optional light-weight bus-only override that keeps
            the default device capacities but relocates the six FlexLoads.
        bus_load_scale_overrides: Optional task-level spatial load calibration
            applied to the static bus loads before feeder shapes are expanded.
        preserve_feeder_totals: Keep each feeder's total static P/Q unchanged
            after applying ``bus_load_scale_overrides``.
        cost_mode: Deprecated legacy field retained for backward-compatible
            config loading. Task-level constraint selection now decides which
            channels benchmark training consumes.

    Returns:
        ``DistGridParams`` ready for ``DistGridEnv``.
    """
    if case is None:
        from powerzoojax.case import create_case33bw
        case = create_case33bw()

    if feeder_shapes is None:
        feeder_shapes = make_synthetic_feeder_shapes(max_steps=max_steps)

    flexload_config = resolve_dso_flexload_config(
        flexload_config=flexload_config,
        flexload_bus_ids=flexload_bus_ids,
    )

    # Build load profiles
    load_p, load_q = make_dso_load_profiles(
        case, feeder_shapes,
        max_steps=max_steps,
        episode_start=episode_start,
        load_scale=load_scale,
        bus_load_scale_overrides=bus_load_scale_overrides,
        preserve_feeder_totals=preserve_feeder_totals,
    )

    # Build FlexLoad bundle from config
    bus_ids = [cfg["bus_id"] for cfg in flexload_config]
    curtail_caps = [cfg["curtail_cap_mw"] for cfg in flexload_config]
    shift_caps = [cfg["shift_cap_mw"] for cfg in flexload_config]

    flex_bundle = make_flexload_bundle(
        case,
        bus_ids=bus_ids,
        curtail_cap_mw=curtail_caps,
        shift_cap_mw=shift_caps,
        shift_horizon=shift_horizon,
        curtail_cost_per_mwh=curtail_cost_per_mwh,
        shift_cost_per_mwh=shift_cost_per_mwh,
        dt_hours=dt_hours,
    )

    # Build DistGridParams via existing factory
    params = make_dist_params(
        case,
        max_steps=max_steps,
        steps_per_day=steps_per_day,
        v_slack=v_slack,
        v_min=v_min,
        v_max=v_max,
        loss_penalty_weight=loss_penalty_weight,
        resources=(flex_bundle,),
        include_der=False,
        cost_mode=cost_mode,
    )

    # Overwrite load profiles with Ausgrid-driven (or synthetic) profiles
    params = params.replace(
        load_profiles_p=load_p,
        load_profiles_q=load_q,
    )

    return params


def make_dso_params_from_split(
    case: Optional[CaseData] = None,
    *,
    role: str = "train",
    data_loader=None,
    resample: str = "30min",
    **kwargs,
) -> DistGridParams:
    """Create real-data DSO params directly from a named Ausgrid split.

    Unlike ``make_dso_params()``, does not rely on synthetic feeder profiles by default.

    Args:
        case: CaseData (default case33bw).
        role: One of ``"train"``, ``"iid"``, ``"summer_ood"``,
            ``"zone_holdout"``.
        data_loader: Optional ``DataLoader`` instance.  ``None`` → create one.
        resample: Target resolution passed to ``load_dso_feeder_shapes``.
        **kwargs: Forwarded to ``make_dso_params`` (``max_steps``,
            ``episode_start``, legacy ``cost_mode``, etc.).

    Returns:
        ``DistGridParams`` with real Ausgrid-driven feeder shapes.
    """
    feeder_shapes = load_dso_feeder_shapes(
        data_loader=data_loader,
        role=role,
        resample=resample,
    )
    return make_dso_params(case, feeder_shapes=feeder_shapes, **kwargs)


def make_dso_1flex_params(
    case: Optional[CaseData] = None,
    *,
    bus_id: int = 18,
    curtail_cap_mw: float = 0.10,
    shift_cap_mw: float = 0.10,
    **kwargs,
) -> DistGridParams:
    """Convenience factory for the 1-device DSO variant.

    Returns ``DistGridParams`` with a single FlexLoad device.  All other
    parameters are forwarded to ``make_dso_params``.  Useful for fast
    unit tests and 1-flex baseline comparisons.

    Args:
        case: CaseData (default case33bw).
        bus_id: Bus to attach the single FlexLoad device (default 18).
        curtail_cap_mw: Curtailment capacity [MW].
        shift_cap_mw: Shift capacity [MW].
        **kwargs: Forwarded to ``make_dso_params`` (feeder_shapes,
            max_steps, legacy ``cost_mode``, etc.).

    Returns:
        ``DistGridParams`` with 1 FlexLoad, action_dim=2.
    """
    config = [{"bus_id": bus_id, "curtail_cap_mw": curtail_cap_mw,
               "shift_cap_mw": shift_cap_mw}]
    return make_dso_params(case, flexload_config=config, **kwargs)


# D3: Non-stationary DSO params factory
def make_dso_params_nonstationary(
    case: Optional[CaseData] = None,
    *,
    feeder_shapes: Optional[Dict[str, np.ndarray]] = None,
    episode_idx: int = 0,
    total_episodes: int = 500,
    key=None,
    drift_rate: float = 0.0003,
    rolling_sigma_steps: float = 720.0,
    mode: str = "train",
    max_steps: int = 48,
    **kwargs,
) -> DistGridParams:
    """Create DSO params with non-stationary episode sampling.

    Wraps ``make_dso_params`` with ``NonstationarySampler`` to set
    ``episode_start`` and ``drift_factor`` based on ``episode_idx``.

    Args:
        case: CaseData (default case33bw).
        feeder_shapes: Per-feeder shapes.  Must cover the full training
            window (not just ``max_steps``).  If None, uses synthetic shapes
            of length ``total_episodes * 2`` steps (for testing).
        episode_idx: Current episode index.
        total_episodes: Total expected episodes in the training run.
        key: JAX PRNG key.  If None, uses ``PRNGKey(episode_idx)``.
        drift_rate: Per-episode drift rate.
        rolling_sigma_steps: Gaussian σ for month-rolling.
        mode: Sampler mode (``"train"`` / ``"iid"`` / ``"drift_shock"`` / ``"fixed"``).
        max_steps: Episode length.
        **kwargs: Passed to ``make_dso_params``.

    Returns:
        ``DistGridParams`` with drift-scaled load profiles.
    """
    import jax

    from powerzoojax.data.nonstationary import NonstationarySampler, apply_drift

    if key is None:
        key = jax.random.PRNGKey(episode_idx)

    if feeder_shapes is None:
        # Longer synthetic shapes for multi-episode sampling
        total_steps = max(total_episodes * 2, max_steps * 10)
        feeder_shapes = make_synthetic_feeder_shapes(max_steps=total_steps)
    else:
        total_steps = min(len(s) for s in feeder_shapes.values())

    sampler = NonstationarySampler(
        total_steps=total_steps,
        total_episodes=total_episodes,
        max_steps=max_steps,
        drift_rate=drift_rate,
        rolling_sigma_steps=rolling_sigma_steps,
    )

    cfg = sampler.sample(episode_idx, key, mode=mode)

    params = make_dso_params(
        case,
        feeder_shapes=feeder_shapes,
        max_steps=max_steps,
        episode_start=cfg.episode_start,
        **kwargs,
    )

    # Apply drift factor to load profiles
    load_p, load_q = apply_drift(
        params.load_profiles_p,
        params.load_profiles_q,
        cfg.drift_factor,
    )
    params = params.replace(load_profiles_p=load_p, load_profiles_q=load_q)

    return params


# D4: Rollout utilities
def rollout_dso(env, params, key, policy_fn, max_steps=None):
    """Run a single DSO episode and collect trajectory info.

    Implementation uses ``jax.lax.scan`` so the entire rollout fuses into a
    single JIT call with no per-step Python<->JAX host syncs. Output keys are
    backward-compatible with the previous Python-loop version: each value is a
    1-D array of length ``max_steps``. Downstream consumers (e.g.
    ``compute_dso_metrics``) wrap them with ``np.array(...)`` and work
    unchanged.

    Args:
        env: ``DistGridEnv`` instance.
        params: ``DistGridParams`` (from ``make_dso_params``).
        key: JAX PRNG key.
        policy_fn: ``(obs, state, key) -> action`` callable.
        max_steps: Override episode length (default from params).

    Returns:
        Dict with keys ``rewards``, ``losses``, ``violations``, ``curtailed``,
        ``voltage_violations``, ``thermal_violations``, ``shifted``,
        ``shift_in``. Each is a 1-D JAX array of shape ``(max_steps,)``.
    """
    import jax

    if max_steps is None:
        max_steps = int(params.max_steps)

    obs0, state0 = env.reset(key, params)

    def body(carry, _):
        obs, state, k = carry
        k, k_step, k_policy = jax.random.split(k, 3)
        action = policy_fn(obs, state, k_policy)
        obs2, state2, reward, _costs, done, info = env.step(k_step, state, action, params)
        out = {
            "rewards": reward,
            "losses": info["p_loss_MW"],
            "violations": info["n_violations"],
            "voltage_violations": info["cost_voltage_violation"],
            "thermal_violations": info["cost_thermal_overload"],
            # Pre-auto-reset scalars from info, so final step is recorded
            # correctly even when done=True triggers auto-reset on state.
            "curtailed": info["resource_curtailed_mw"],
            "shifted": info["resource_shift_out_mw"],
            "shift_in": info["resource_shift_in_mw"],
        }
        return (obs2, state2, k), out

    _, results = jax.lax.scan(
        body, (obs0, state0, key), xs=None, length=int(max_steps)
    )
    return results


def _bundle_action_dim(params) -> int:
    """Total action dim for all resource bundles in params (trace-time constant)."""
    return sum(b.action_dim for b in params.resources)


def _first_flexload_bus_idx(params):
    """Internal bus indices for the first FlexLoadBundle in params.

    Returns the bundle's ``bus_idx`` array as-is (``jnp.ndarray`` on device or
    concrete ``np.ndarray`` depending on how ``params`` was constructed).  The
    shape is static and safe to read under ``jax.jit`` tracing, while the
    values themselves are only used for in-graph indexing — so we avoid the
    unconditional ``np.asarray`` conversion that would fail on traced arrays.
    Raises if no bundle is attached.
    """
    if not params.resources:
        raise ValueError("params has no resource bundles")
    return params.resources[0].bus_idx


def dso_no_control_rollout(env, params, key):
    """No-control baseline: zero action at every step.

    Works with any FlexLoadBundle configuration (1-flex, 6-flex, large, ...).
    Action dimension is derived from ``params.resources``.
    """
    action_dim = _bundle_action_dim(params)

    def policy_fn(obs, state, key):
        return jnp.zeros(action_dim)

    return rollout_dso(env, params, key, policy_fn)


def dso_tou_rule_based_rollout(env, params, key, peak_start=16, peak_end=21):
    """Time-of-use (TOU) **rule-based** baseline: curtail + shift in a fixed peak window.

    During peak hours (steps ``peak_start`` to ``peak_end``), all devices
    apply 80% curtailment + 50% shift-out.  Outside peak, all actions = 0.

    Works with any FlexLoadBundle configuration (1-flex, 6-flex, large, ...).
    Action dimension and device count are derived from ``params.resources``.

    Args:
        peak_start: Start of peak period (step index within day).
        peak_end: End of peak period (step index within day).
    """
    action_dim = _bundle_action_dim(params)
    n_devices = sum(b.n_devices for b in params.resources if hasattr(b, "n_devices"))
    peak_action = jnp.array([0.8, 0.5] * n_devices, dtype=jnp.float32)
    zero_action = jnp.zeros(action_dim, dtype=jnp.float32)

    def policy_fn(obs, state, key):
        # Derive the step index from env state so the TOU rule stays
        # JIT/scan-compatible. A Python counter is invisible inside
        # lax.scan and would freeze the policy at the initial branch.
        t = state.time_step % params.steps_per_day
        is_peak = jnp.logical_and(t >= peak_start, t < peak_end)
        return jnp.where(is_peak, peak_action, zero_action)

    return rollout_dso(env, params, key, policy_fn)


def dso_droop_rule_based_rollout(env, params, key, v_low=0.96, v_high=1.04):
    """Voltage-droop **rule-based** baseline: curtail/shift from local voltage vs. band.

    When voltage at a FlexLoad bus drops below ``v_low``, that device applies
    curtailment and shift-out proportional to the voltage deviation.  Above
    ``v_high``, shift-in is allowed (action = 0, passthrough).

    Works with any FlexLoadBundle configuration (1-flex, 6-flex, large, ...).
    Bus indices are read from ``params.resources[0].bus_idx`` rather than
    from the hardcoded ``DSO_FLEXLOAD_CONFIG`` constant, so this policy
    is valid for any DSO variant preset.

    This is a reactive, deterministic local control rule (droop-style scheduling).
    """
    flex_bus_internal_idx = _first_flexload_bus_idx(params)
    n_devices = int(flex_bus_internal_idx.shape[0])
    action_dim = _bundle_action_dim(params)

    def policy_fn(obs, state, key):
        v_mag = state.v_mag
        action = jnp.zeros(action_dim)
        for i in range(n_devices):
            bus_v = v_mag[flex_bus_internal_idx[i]]
            curtail_frac = jnp.clip((v_low - bus_v) / 0.04, 0.0, 1.0)
            shift_frac = jnp.clip((v_low - bus_v) / 0.06, 0.0, 0.5)
            action = action.at[2 * i].set(curtail_frac)
            action = action.at[2 * i + 1].set(shift_frac)
        return action

    return rollout_dso(env, params, key, policy_fn)


# Backward-compatible names (prefer ``*_rule_based_*`` in new code and docs).
dso_tou_heuristic_rollout = dso_tou_rule_based_rollout
dso_droop_heuristic_rollout = dso_droop_rule_based_rollout


# D4: DSO metrics
def compute_dso_metrics(
    rollout_results: Dict,
    baseline_results: Optional[Dict] = None,
) -> Dict[str, float]:
    """Compute DSO evaluation metrics from rollout results.

    Args:
        rollout_results: Dict from ``rollout_dso`` or a baseline rollout.
        baseline_results: No-control baseline results for relative metrics.
            If None, relative metrics are not computed.

    Returns:
        Dict of metric name → value:
        - ``total_reward``: cumulative episode reward
        - ``total_loss_mwh``: total network loss [MWh] (sum of per-step MW × dt)
        - ``mean_loss_mw``: average network loss [MW]
        - ``total_voltage_violations``: count of voltage violations
        - ``total_thermal_overloads``: count of thermal overloads
        - ``total_violations``: total count of voltage + thermal violations
        - ``voltage_violation_count_per_step``: average voltage violations / step
        - ``thermal_overload_count_per_step``: average thermal overloads / step
        - ``total_curtailed_mwh``: total energy curtailed [MWh]
        - ``total_shifted_mwh``: total energy shifted out [MWh]
        - ``total_shift_in_mwh``: total energy released (shift-in) [MWh]
        - ``served_flex_ratio``: shift_in / shift_out (buffer clearance)
        - ``network_loss_reduction_pct``: relative loss reduction vs baseline
        - ``peak_shaving_pct``: relative peak reduction vs baseline
    """
    dt_h = 0.5  # 30-min steps

    losses = np.array(rollout_results["losses"])
    curtailed = np.array(rollout_results["curtailed"])
    shifted = np.array(rollout_results["shifted"])
    shift_in = np.array(rollout_results["shift_in"])

    rewards = np.array(rollout_results["rewards"])
    voltage_violations = np.array(
        rollout_results.get("voltage_violations", rollout_results["violations"])
    )
    thermal_overloads = np.array(
        rollout_results.get(
            "thermal_violations",
            np.zeros_like(voltage_violations, dtype=np.float32),
        )
    )
    total_violations = voltage_violations + thermal_overloads
    max_steps = max(int(losses.shape[0]), 1)

    metrics = {
        "total_reward": float(rewards.sum()),
        "total_loss_mwh": float(losses.sum() * dt_h),
        "mean_loss_mw": float(losses.mean()),
        "total_voltage_violations": float(voltage_violations.sum()),
        "total_thermal_overloads": float(thermal_overloads.sum()),
        "total_violations": float(total_violations.sum()),
        "voltage_violation_count_per_step": float(voltage_violations.sum() / max_steps),
        "thermal_overload_count_per_step": float(thermal_overloads.sum() / max_steps),
        "total_curtailed_mwh": float(curtailed.sum() * dt_h),
        "total_shifted_mwh": float(shifted.sum() * dt_h),
        "total_shift_in_mwh": float(shift_in.sum() * dt_h),
        "served_flex_ratio": (
            float(shift_in.sum() / max(shifted.sum(), 1e-8))
            if shifted.sum() > 0 else 0.0
        ),
    }

    if baseline_results is not None:
        bl_losses = np.array(baseline_results["losses"])
        bl_total = bl_losses.sum()
        if bl_total > 1e-8:
            metrics["network_loss_reduction_pct"] = float(
                (bl_total - losses.sum()) / bl_total * 100.0
            )
        else:
            metrics["network_loss_reduction_pct"] = 0.0

        bl_peak = bl_losses.max()
        rl_peak = losses.max()
        if bl_peak > 1e-8:
            metrics["peak_shaving_pct"] = float(
                (bl_peak - rl_peak) / bl_peak * 100.0
            )
        else:
            metrics["peak_shaving_pct"] = 0.0
    else:
        metrics["network_loss_reduction_pct"] = None
        metrics["peak_shaving_pct"] = None

    return metrics


# ============ TaskSpec implementation ============

class DSOTask:
    """TaskSpec implementation for the DSO benchmark.

    Single-agent non-stationary DistGrid + FlexLoad task on case33bw.

    Args:
        v_min: Minimum voltage limit [p.u.]. Default: ``DSO_V_MIN`` (0.94).
        v_max: Maximum voltage limit [p.u.]. Default: ``DSO_V_MAX`` (1.06).
        cost_mode: Deprecated legacy field retained for backward-compatible
            config loading.
    """

    task_name = "dso"
    default_splits = ("train", "iid", "summer_ood", "zone_holdout")

    def __init__(
        self,
        v_min: float = DSO_V_MIN,
        v_max: float = DSO_V_MAX,
        cost_mode: str = "voltage_only",
        *,
        load_scale: float = 1.0,
        v_slack: float = 1.0,
        shift_horizon: int = 4,
        flexload_config: Optional[List[Dict]] = None,
        flexload_bus_ids: Optional[List[int]] = None,
        bus_load_scale_overrides: Optional[Dict[int, float]] = None,
        preserve_feeder_totals: bool = False,
    ) -> None:
        self._load_scale = load_scale
        self._v_min = v_min
        self._v_max = v_max
        self._cost_mode = cost_mode
        self._v_slack = v_slack
        self._shift_horizon = shift_horizon
        self._flexload_config = flexload_config
        self._flexload_bus_ids = flexload_bus_ids
        self._bus_load_scale_overrides = _normalize_bus_load_scale_overrides(
            bus_load_scale_overrides
        )
        self._preserve_feeder_totals = preserve_feeder_totals
        self._cache: Dict = {}  # split → (base_params, feeder_shapes, case_obj)

    def _ensure_cache(self, split: str) -> None:
        if split in self._cache:
            return
        from powerzoojax.case import create_case33bw
        base_params = make_dso_params_from_split(
            role=split,
            load_scale=self._load_scale,
            v_slack=self._v_slack,
            v_min=self._v_min,
            v_max=self._v_max,
            shift_horizon=self._shift_horizon,
            flexload_config=self._flexload_config,
            flexload_bus_ids=self._flexload_bus_ids,
            bus_load_scale_overrides=self._bus_load_scale_overrides,
            preserve_feeder_totals=self._preserve_feeder_totals,
            cost_mode=self._cost_mode,
        )
        feeder_shapes = load_dso_feeder_shapes(data_loader=None, role=split)
        self._cache[split] = (base_params, feeder_shapes, create_case33bw())

    def make_env(self, split: str = "train"):
        from powerzoojax.envs.grid.dist import DistGridEnv
        return DistGridEnv()

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
        base_params, feeder_shapes, case_obj = self._cache[split]
        T = min(len(s) for s in feeder_shapes.values())
        if strategy == "seeded":
            rng = np.random.default_rng(seed * 997 + 3)
            start = int(rng.integers(0, max(1, T - max_steps)))
        else:
            starts = np.linspace(0, max(0, T - max_steps), n_episodes).astype(int)
            start = int(starts[episode_idx])
        load_p, load_q = make_dso_load_profiles(
            case_obj,
            feeder_shapes,
            max_steps=max_steps,
            episode_start=start,
            load_scale=self._load_scale,
            bus_load_scale_overrides=self._bus_load_scale_overrides,
            preserve_feeder_totals=self._preserve_feeder_totals,
        )
        return base_params.replace(load_profiles_p=load_p, load_profiles_q=load_q)

    def rollout(self, env, params, key, policy_fn):
        return rollout_dso(env, params, key, policy_fn)

    def baseline_rollout(self, env, params, key, baseline_name: str):
        _map = {
            "no_control": dso_no_control_rollout,
            "tou":        dso_tou_rule_based_rollout,
            "droop":      dso_droop_rule_based_rollout,
        }
        return _map[baseline_name](env, params, key)

    def compute_metrics(self, agent_rollout, baseline_rollout):
        return compute_dso_metrics(agent_rollout, baseline_rollout)

    def baseline_names(self) -> tuple:
        return ("no_control", "tou", "droop")

    def constraint_spec(self) -> ConstraintSpec:
        return ConstraintSpec(
            selected_names=("voltage_violation",),
            thresholds=(0.0,),
            fallback_weights=(1.0,),
        )
