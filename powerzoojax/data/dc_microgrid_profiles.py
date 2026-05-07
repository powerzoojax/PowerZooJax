"""DC Microgrid profile utilities: real-trace loading, episode cycling, OOD transforms.

Responsibilities
----------------
- Synthetic deterministic profiles (always available; no data files required).
- Real workload profile loading via :class:`DataLoader` (requires parquet data).
- Episode slicing / tiling / cycling.
- OOD scenario transforms that return modified :class:`DCMicrogridParams`.

Real vs Synthetic (explicit accounting)
-----------------------------------------
Real data (requires parquet files):
    - ``cpu_profile``, ``mem_profile`` from Google / Azure / Alibaba DC traces
      (manifests: google_dc_2019, azure_dc_v2, alibaba_dc_2018).
    - All three sources are 5-min (300 s) interval, ``time_mode="profile"``,
      ``cyclical=True``.
    - ``solar_profile`` from the GB generation-by-type trace
      (manifest: gb_gen_by_type).  The 30-min Solar series is normalised to a
      capacity factor and expanded to the DC task's 5-min grid.
    - ``price_profile`` from the GB MID market trace
      (manifest: gb_market_mid).  The 30-min APX MID price is expanded to the
      DC task's 5-min grid and used as a grid-import price [currency/MWh].

Deterministic adapters (NOT real data; implemented as sine-curve proxies):
    - ``outdoor_temp_profile``: matches ``_outdoor_temp()`` in datacenter_microgrid.py.
    This will be replaced when a weather manifest is available.

Source keys (used in ``load_workload_profiles`` and OOD transforms):
    - ``"google"`` → google_dc_2019
    - ``"azure"``  → azure_dc_v2
    - ``"alibaba"``→ alibaba_dc_2018

If a required parquet file is missing, ``load_workload_profiles`` emits a
``UserWarning`` and falls back to synthetic profiles unless ``require_real_data``
is true.
"""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import warnings
from typing import Dict, Optional, Union

import jax.numpy as jnp
import numpy as np

# Canonical step configuration for DC Microgrid main config
DC_STEPS_PER_DAY: int = 288   # 5-min steps per 24-h day
DC_EPISODE_LEN: int = 288     # one full day per episode

# Mapping from user-facing source names to manifest names
_SOURCE_TO_MANIFEST: Dict[str, str] = {
    "google":  "google_dc_2019",
    "azure":   "azure_dc_v2",
    "alibaba": "alibaba_dc_2018",
}

_DATA_ROOT = Path(__file__).resolve().parent
_DEFAULT_PARQUET_DIR = _DATA_ROOT / "parquet"
_DEFAULT_MANIFEST_DIR = _DATA_ROOT / "manifests"
_SOLAR_MANIFEST = "gb_gen_by_type"
_SOLAR_PROFILE_YEAR = 2025
_MARKET_MANIFEST = "gb_market_mid"

VALID_SOURCES = tuple(_SOURCE_TO_MANIFEST.keys())
VALID_OOD_SCENARIOS = (
    "workload_swap",      # cpu_profile: Google → Azure
    "workload_shock",     # cpu_profile: Google → Alibaba (higher variance)
    "renewable_drought",  # solar_profile × drought_factor (default 0.2)
    "cooling_stress",     # outdoor_temp_profile + temp_delta (default +5°C)
    "dg_derating",        # DieselParams.p_dg_max_mw × dg_derating_factor (0.6)
    "sla_tighten",        # DataCenterParams.{train,ft}_deadline_slack → sla_slack (1.2)
)


# ---------------------------------------------------------------------------
# Synthetic profile generators
# ---------------------------------------------------------------------------

def make_synthetic_cpu_profile(
    episode_len: int = DC_EPISODE_LEN,
    steps_per_day: int = DC_STEPS_PER_DAY,
) -> jnp.ndarray:
    """Deterministic diurnal CPU utilisation ∈ [0.1, 1.0].

    Matches the ``_diurnal_factor()`` function inside datacenter.py:
    ``0.5 + 0.5 * sin(2π*(hour-8)/24)`` clipped to [0.1, 1.0].
    """
    t = jnp.arange(episode_len, dtype=jnp.float32)
    hour = (t % steps_per_day) / float(steps_per_day) * 24.0
    return jnp.clip(0.5 + 0.5 * jnp.sin(2.0 * jnp.pi * (hour - 8.0) / 24.0), 0.1, 1.0)


def make_synthetic_solar_profile(
    episode_len: int = DC_EPISODE_LEN,
    steps_per_day: int = DC_STEPS_PER_DAY,
) -> jnp.ndarray:
    """Deterministic solar capacity factor ∈ [0, 1].

    Matches ``_solar_cf()`` in datacenter_microgrid.py:
    ``clip(sin(π*(hour-6)/12), 0, 1)`` — positive 06:00–18:00.
    """
    t = jnp.arange(episode_len, dtype=jnp.float32)
    hour = (t % steps_per_day) / float(steps_per_day) * 24.0
    return jnp.clip(jnp.sin(jnp.pi * (hour - 6.0) / 12.0), 0.0, 1.0)


@lru_cache(maxsize=8)
def _load_gb_solar_cf_5min(
    data_dir_key: str,
    manifest_dir_key: str,
    year: int,
) -> np.ndarray:
    manifest_path = Path(manifest_dir_key) / f"{_SOLAR_MANIFEST}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing solar manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parquet_file = manifest.get("parquet_file")
    if not parquet_file:
        raise ValueError(f"{manifest_path} has no parquet_file field")
    parquet_path = Path(data_dir_key) / str(parquet_file)
    if not parquet_path.exists():
        raise FileNotFoundError(f"missing solar parquet: {parquet_path}")

    try:
        import pandas as pd
    except ImportError:
        raise

    df = pd.read_parquet(parquet_path, columns=["startTime", "Solar"])
    if "Solar" not in df.columns:
        raise ValueError(f"{parquet_path} has no Solar column")
    if "startTime" in df.columns:
        ts = pd.to_datetime(df["startTime"], utc=True, errors="coerce")
        mask = (ts >= f"{year}-01-01") & (ts < f"{year + 1}-01-01")
        if int(mask.sum()) >= 48:
            df = df.loc[mask]

    values = df["Solar"].to_numpy(dtype=np.float32, copy=True)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, None)
    peak = float(np.max(values)) if values.size else 0.0
    if peak <= 0.0:
        raise ValueError(f"{parquet_path} has no positive Solar values")
    cf_30min = np.clip(values / peak, 0.0, 1.0).astype(np.float32)
    return np.repeat(cf_30min, 6).astype(np.float32)


def make_real_solar_profile(
    episode_len: int = DC_EPISODE_LEN,
    start_step: int = 0,
    data_dir: Optional[str] = None,
    manifest_dir: Optional[str] = None,
    *,
    strict: bool = True,
    require_real_data: bool = False,
) -> jnp.ndarray:
    """Load a real solar generation trace and return a 5-min capacity factor.

    The source is the committed GB generation-by-type parquet
    (``gb_gen_by_type`` manifest).  The 30-min ``Solar`` series is clipped,
    normalised by the selected year's peak, repeated to 5-min resolution, then
    cycled using ``start_step``.
    """
    data_root = str(Path(data_dir).resolve()) if data_dir else str(_DEFAULT_PARQUET_DIR)
    manifest_root = (
        str(Path(manifest_dir).resolve()) if manifest_dir else str(_DEFAULT_MANIFEST_DIR)
    )
    try:
        solar_full = jnp.asarray(
            _load_gb_solar_cf_5min(data_root, manifest_root, _SOLAR_PROFILE_YEAR),
            dtype=jnp.float32,
        )
        return cycle_profile(solar_full, episode_len, start_step)
    except (FileNotFoundError, ImportError) as exc:
        if require_real_data:
            raise
        warnings.warn(
            f"Solar data unavailable ({type(exc).__name__}: {exc}). "
            "Falling back to synthetic solar_profile.",
            UserWarning,
            stacklevel=2,
        )
    except Exception:
        if strict:
            raise
        warnings.warn(
            "Failed to load solar data (strict=False). "
            "Falling back to synthetic solar_profile.",
            UserWarning,
            stacklevel=2,
        )
    return make_synthetic_solar_profile(episode_len, DC_STEPS_PER_DAY)


@lru_cache(maxsize=8)
def _load_gb_mid_price_5min(
    data_dir_key: str,
    manifest_dir_key: str,
) -> np.ndarray:
    manifest_path = Path(manifest_dir_key) / f"{_MARKET_MANIFEST}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing market-price manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parquet_file = manifest.get("parquet_file")
    if not parquet_file:
        raise ValueError(f"{manifest_path} has no parquet_file field")
    parquet_path = Path(data_dir_key) / str(parquet_file)
    if not parquet_path.exists():
        raise FileNotFoundError(f"missing market-price parquet: {parquet_path}")

    try:
        import pandas as pd
    except ImportError:
        raise

    df = pd.read_parquet(
        parquet_path,
        columns=["mid_price_APXMIDP", "mid_price_N2EXMIDP"],
    )
    apx = df["mid_price_APXMIDP"].to_numpy(dtype=np.float32, copy=True)
    n2ex = df["mid_price_N2EXMIDP"].to_numpy(dtype=np.float32, copy=True)
    values = np.where(np.isfinite(apx) & (apx > 0.0), apx, n2ex)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, 500.0).astype(np.float32)
    if values.size == 0 or float(np.max(values)) <= 0.0:
        raise ValueError(f"{parquet_path} has no positive MID price values")
    return np.repeat(values, 6).astype(np.float32)


def make_synthetic_price_profile(
    episode_len: int = DC_EPISODE_LEN,
    steps_per_day: int = DC_STEPS_PER_DAY,
) -> jnp.ndarray:
    """Fallback time-of-use price [currency/MWh], low overnight and high evening."""
    t = jnp.arange(episode_len, dtype=jnp.float32)
    hour = (t % steps_per_day) / float(steps_per_day) * 24.0
    night = jnp.where((hour < 6.0) | (hour >= 23.0), 55.0, 0.0)
    day = jnp.where((hour >= 6.0) & (hour < 16.0), 90.0, 0.0)
    peak = jnp.where((hour >= 16.0) & (hour < 23.0), 180.0, 0.0)
    return jnp.asarray(night + day + peak, dtype=jnp.float32)


def make_real_price_profile(
    episode_len: int = DC_EPISODE_LEN,
    start_step: int = 0,
    data_dir: Optional[str] = None,
    manifest_dir: Optional[str] = None,
    *,
    strict: bool = True,
    require_real_data: bool = False,
) -> jnp.ndarray:
    """Load the GB MID market price trace and return a 5-min price profile."""
    data_root = str(Path(data_dir).resolve()) if data_dir else str(_DEFAULT_PARQUET_DIR)
    manifest_root = (
        str(Path(manifest_dir).resolve()) if manifest_dir else str(_DEFAULT_MANIFEST_DIR)
    )
    try:
        price_full = jnp.asarray(
            _load_gb_mid_price_5min(data_root, manifest_root),
            dtype=jnp.float32,
        )
        return cycle_profile(price_full, episode_len, start_step)
    except (FileNotFoundError, ImportError) as exc:
        if require_real_data:
            raise
        warnings.warn(
            f"Market price data unavailable ({type(exc).__name__}: {exc}). "
            "Falling back to synthetic price_profile.",
            UserWarning,
            stacklevel=2,
        )
    except Exception:
        if strict:
            raise
        warnings.warn(
            "Failed to load market price data (strict=False). "
            "Falling back to synthetic price_profile.",
            UserWarning,
            stacklevel=2,
        )
    return make_synthetic_price_profile(episode_len, DC_STEPS_PER_DAY)


def make_synthetic_outdoor_temp_profile(
    episode_len: int = DC_EPISODE_LEN,
    steps_per_day: int = DC_STEPS_PER_DAY,
    mean_temp: float = 20.0,
    amplitude: float = 8.0,
) -> jnp.ndarray:
    """Deterministic outdoor temperature [°C].

    Matches ``_outdoor_temp()`` in datacenter.py:
    ``20 + 8 * sin(2π*(hour-8)/24)``.
    """
    t = jnp.arange(episode_len, dtype=jnp.float32)
    hour = (t % steps_per_day) / float(steps_per_day) * 24.0
    return jnp.float32(mean_temp) + jnp.float32(amplitude) * jnp.sin(
        2.0 * jnp.pi * (hour - 8.0) / 24.0
    )


def make_all_synthetic_profiles(
    episode_len: int = DC_EPISODE_LEN,
    steps_per_day: int = DC_STEPS_PER_DAY,
) -> Dict[str, jnp.ndarray]:
    """Return all three synthetic profiles as a dict."""
    return {
        "cpu_profile": make_synthetic_cpu_profile(episode_len, steps_per_day),
        "solar_profile": make_synthetic_solar_profile(episode_len, steps_per_day),
        "outdoor_temp_profile": make_synthetic_outdoor_temp_profile(
            episode_len, steps_per_day
        ),
    }


# ---------------------------------------------------------------------------
# Episode slicing / cycling
# ---------------------------------------------------------------------------

def cycle_profile(
    arr: jnp.ndarray,
    episode_len: int,
    start_step: int = 0,
) -> jnp.ndarray:
    """Slice or cyclically tile *arr* to exactly *episode_len* steps.

    Args:
        arr: Source array of shape ``(T,)`` (float32).  If empty (T=0),
            returned as-is.
        episode_len: Desired output length.
        start_step: Starting offset into the cyclically-tiled sequence.

    Returns:
        Array of shape ``(episode_len,)`` with dtype preserved.

    Rules:
        - If ``T >= episode_len`` and ``start_step == 0``: simple slice.
        - Otherwise: tile enough repetitions to cover
          ``start_step + episode_len``, then slice.
    """
    T = arr.shape[0]
    if T == 0 or episode_len == 0:
        return arr
    n_tile = (start_step + episode_len + T - 1) // T
    tiled = jnp.tile(arr, n_tile)
    return tiled[start_step : start_step + episode_len]


# ---------------------------------------------------------------------------
# Real-data workload loader (with synthetic fallback)
# ---------------------------------------------------------------------------

def load_workload_profiles(
    source: str = "google",
    episode_len: int = DC_EPISODE_LEN,
    start_step: int = 0,
    data_dir: Optional[str] = None,
    manifest_dir: Optional[str] = None,
    strict: bool = True,
    require_real_data: bool = False,
) -> Dict[str, jnp.ndarray]:
    """Load DC workload profiles, with controlled fallback behaviour.

    Args:
        source: Data source key — ``"google"``, ``"azure"``, or ``"alibaba"``.
        episode_len: Number of steps in the output profiles (default 288).
        start_step: Cyclic offset into the source profile (default 0).
        data_dir: Override for the parquet directory (``None`` → default).
        manifest_dir: Override for the manifests directory (``None`` → default).
        strict: Schema / manifest error handling (default ``True``).

            * ``strict=True`` (default): non-infra exceptions (schema mismatch,
              signal-resolution failure, …) are re-raised immediately.
            * ``strict=False``: all exceptions warn + synthetic fallback.

        require_real_data: Infrastructure error handling (default ``False``).

            * ``False`` (default): ``FileNotFoundError`` and ``ImportError``
              (parquet engine absent) → ``UserWarning`` + synthetic fallback.
              Suitable for generic env construction and development.
            * ``True``: ``FileNotFoundError`` and ``ImportError`` are ALSO
              re-raised — the caller requires a real workload profile.  Use for
              OOD scenarios (``workload_swap``, ``workload_shock``) where using
              synthetic data would silently corrupt the experiment.

    Returns:
        Dict with keys ``"cpu_profile"``, ``"solar_profile"``,
        ``"outdoor_temp_profile"`` — all ``jnp.float32`` arrays of shape
        ``(episode_len,)``.

    Notes:
        **Real data** (cpu_profile): loaded from the DC workload parquet file
        when available.

        **Real data** (solar_profile): loaded from the GB generation-by-type
        solar trace when available.

        **Real data** (price_profile): loaded from the GB MID market trace
        when available.

        **Deterministic adapter** (outdoor_temp_profile): synthetic sine curve —
        no weather manifest exists yet.
    """
    if source not in _SOURCE_TO_MANIFEST:
        raise ValueError(
            f"Unknown source '{source}'. "
            f"Valid sources: {VALID_SOURCES}"
        )

    cpu_arr: Optional[jnp.ndarray] = None

    try:
        from powerzoojax.data.data_loader import DataLoader
        loader = DataLoader(data_dir=data_dir, manifest_dir=manifest_dir)
        raw: jnp.ndarray = loader.load_jax_profiles(
            ["datacenter.cpu_util"],
            source=source,
        )  # shape (T, 1)
        cpu_full = raw[:, 0]  # (T,)
        cpu_arr = cycle_profile(cpu_full, episode_len, start_step)
    except (FileNotFoundError, ImportError) as exc:
        # FileNotFoundError: parquet data file absent (expected in CI / dev).
        # ImportError:        no parquet engine installed (pyarrow / fastparquet).
        if require_real_data:
            # Caller (e.g. workload_swap / workload_shock) requires a real profile:
            # silently returning synthetic would corrupt the OOD experiment.
            raise
        # Generic path: infra / environment issue → acceptable fallback.
        warnings.warn(
            f"Workload data unavailable for source='{source}' "
            f"({type(exc).__name__}: {exc}). "
            f"Falling back to synthetic cpu_profile.",
            UserWarning,
            stacklevel=2,
        )
    except Exception:
        if strict:
            raise  # re-raise: schema / manifest / signal errors must not be silenced
        warnings.warn(
            f"Failed to load workload data for source='{source}' "
            f"(strict=False). Falling back to synthetic cpu_profile.",
            UserWarning,
            stacklevel=2,
        )

    if cpu_arr is None:
        cpu_arr = make_synthetic_cpu_profile(episode_len, DC_STEPS_PER_DAY)

    return {
        "cpu_profile": cpu_arr,
        "solar_profile": make_real_solar_profile(
            episode_len=episode_len,
            start_step=start_step,
            data_dir=data_dir,
            manifest_dir=manifest_dir,
            strict=strict,
            require_real_data=require_real_data,
        ),
        "price_profile": make_real_price_profile(
            episode_len=episode_len,
            start_step=start_step,
            data_dir=data_dir,
            manifest_dir=manifest_dir,
            strict=strict,
            require_real_data=require_real_data,
        ),
        # Deterministic adapter — explicitly NOT real weather data:
        "outdoor_temp_profile": make_synthetic_outdoor_temp_profile(
            episode_len, DC_STEPS_PER_DAY
        ),
    }


# ---------------------------------------------------------------------------
# OOD transforms
# ---------------------------------------------------------------------------

def apply_ood_transform(
    params: "DCMicrogridParams",  # type: ignore[name-defined]
    scenario: str,
    *,
    drought_factor: float = 0.2,
    temp_delta: float = 5.0,
    dg_derating_factor: float = 0.6,
    sla_slack: float = 1.2,
    data_dir: Optional[str] = None,
    manifest_dir: Optional[str] = None,
) -> "DCMicrogridParams":  # type: ignore[name-defined]
    """Apply an OOD scenario transform, returning a modified ``DCMicrogridParams``.

    Args:
        params: Base params — should have non-None profiles for transforms
            that modify them.  If profiles are None, a synthetic fallback is
            generated automatically.
        scenario: One of ``VALID_OOD_SCENARIOS``.
        drought_factor: Solar scaling for ``renewable_drought`` (default 0.2).
        temp_delta: Temperature offset [°C] for ``cooling_stress`` (default 5.0).
        dg_derating_factor: DG capacity multiplier for ``dg_derating`` (0.6).
        sla_slack: New deadline slack for ``sla_tighten`` (default 1.2).
        data_dir: Override data dir for workload_swap / workload_shock.
        manifest_dir: Override manifest dir for workload_swap / workload_shock.

    Returns:
        New ``DCMicrogridParams`` with the transform applied.

    Raises:
        ValueError: Unknown scenario.
    """
    # Deferred import to avoid circular
    from powerzoojax.envs.microgrid import (
        DCMicrogridParams,
        make_dcmicrogrid_params,
    )
    from powerzoojax.envs.resource.diesel import DieselParams
    from powerzoojax.envs.resource.datacenter import make_datacenter_params

    if scenario not in VALID_OOD_SCENARIOS:
        raise ValueError(
            f"Unknown OOD scenario '{scenario}'. "
            f"Valid: {VALID_OOD_SCENARIOS}"
        )

    # Derive episode_len from base params to preserve the configured horizon.
    # Priority: current cpu_profile.shape[0] → params.dc.max_steps → DC_EPISODE_LEN.
    if params.cpu_profile is not None and params.cpu_profile.shape[0] > 0:
        episode_len = int(params.cpu_profile.shape[0])
    else:
        episode_len = int(params.dc.max_steps)

    # ------------------------------------------------------------------
    # Helper: get current profile or synthetic fallback
    # ------------------------------------------------------------------
    def _get_cpu(p: "DCMicrogridParams") -> jnp.ndarray:
        if p.cpu_profile is not None:
            return p.cpu_profile
        return make_synthetic_cpu_profile(episode_len)

    def _get_solar(p: "DCMicrogridParams") -> jnp.ndarray:
        # Solar profile is owned by the (zero-action) RenewableBundle attached
        # in p.resources.  Profiles are stored as (T, n_devices); the OOD
        # transforms operate on the shared (T,) shape — squeeze accordingly.
        from powerzoojax.envs.resource.renewable import RenewableBundle
        for b in p.resources:
            if isinstance(b, RenewableBundle) and b.profiles is not None:
                if b.profiles.ndim == 2:
                    return b.profiles[:, 0]
                return b.profiles
        return make_synthetic_solar_profile(episode_len)

    def _get_temp(p: "DCMicrogridParams") -> jnp.ndarray:
        if p.outdoor_temp_profile is not None:
            return p.outdoor_temp_profile
        return make_synthetic_outdoor_temp_profile(episode_len)

    # ------------------------------------------------------------------
    # Scenario dispatch
    # ------------------------------------------------------------------
    if scenario == "renewable_drought":
        # Solar × drought_factor (default 0.2 → 20% of normal)
        new_solar = _get_solar(params) * jnp.float32(drought_factor)
        return _replace_profiles(params, solar_profile=new_solar)

    elif scenario == "cooling_stress":
        # Outdoor temperature + temp_delta
        new_temp = _get_temp(params) + jnp.float32(temp_delta)
        return _replace_profiles(params, outdoor_temp_profile=new_temp)

    elif scenario == "dg_derating":
        # DG nameplate capacity × dg_derating_factor (applied to every DG device)
        return _scale_diesel_capacity(params, dg_derating_factor)

    elif scenario == "sla_tighten":
        # Reduce deadline slack: train + ft both → sla_slack
        old_dc = params.dc
        new_dc = make_datacenter_params(
            n_gpus=old_dc.n_gpus,
            gpu_idle_w=old_dc.gpu_idle_w,
            gpu_active_w=old_dc.gpu_active_w,
            p_base_mw=old_dc.p_base_mw,
            infer_gpu_peak=old_dc.infer_gpu_peak,
            cop_ref=old_dc.cop_ref,
            cop_decay=old_dc.cop_decay,
            t_ref=old_dc.t_ref,
            c_thermal=old_dc.c_thermal,
            ua_cooling=old_dc.ua_cooling,
            h_wall=old_dc.h_wall,
            t_set_min=old_dc.t_set_min,
            t_set_max=old_dc.t_set_max,
            t_initial=old_dc.t_initial,
            t_critical=old_dc.t_critical,
            p_aux_frac=old_dc.p_aux_frac,
            train_arrival_interval=old_dc.train_arrival_interval,
            train_gpu_lo=old_dc.train_gpu_lo,
            train_gpu_hi=old_dc.train_gpu_hi,
            train_dur_lo=old_dc.train_dur_lo,
            train_dur_hi=old_dc.train_dur_hi,
            train_deadline_slack=sla_slack,  # tightened
            train_gpu_eta=old_dc.train_gpu_eta,
            ft_arrival_interval=old_dc.ft_arrival_interval,
            ft_gpu_lo=old_dc.ft_gpu_lo,
            ft_gpu_hi=old_dc.ft_gpu_hi,
            ft_dur_lo=old_dc.ft_dur_lo,
            ft_dur_hi=old_dc.ft_dur_hi,
            ft_deadline_slack=sla_slack,  # tightened
            ft_gpu_eta=old_dc.ft_gpu_eta,
            delta_t_hours=old_dc.delta_t_hours,
            steps_per_day=old_dc.steps_per_day,
            max_steps=old_dc.max_steps,
        )
        return _replace_dc(params, new_dc)

    elif scenario in ("workload_swap", "workload_shock"):
        # Replace cpu_profile with data from a different source.
        # episode_len is derived from base params above — preserves current horizon.
        # require_real_data=True: synthetic fallback must NOT happen here — if the
        # workload file is absent or no parquet engine is available, raise immediately
        # so the experiment is not silently run with fake OOD data.
        src_map = {"workload_swap": "azure", "workload_shock": "alibaba"}
        new_source = src_map[scenario]
        profiles = load_workload_profiles(
            new_source,
            episode_len=episode_len,
            data_dir=data_dir,
            manifest_dir=manifest_dir,
            strict=True,
            require_real_data=True,
        )
        return _replace_profiles(params, cpu_profile=profiles["cpu_profile"])

    else:
        raise ValueError(f"Unhandled scenario: '{scenario}'")


# ---------------------------------------------------------------------------
# Internal helpers for param field replacement
# ---------------------------------------------------------------------------

def _replace_profiles(
    params: "DCMicrogridParams",
    *,
    cpu_profile: Optional[jnp.ndarray] = None,
    solar_profile: Optional[jnp.ndarray] = None,
    outdoor_temp_profile: Optional[jnp.ndarray] = None,
) -> "DCMicrogridParams":
    """Return a new DCMicrogridParams with specified profiles replaced.

    ``cpu_profile`` and ``outdoor_temp_profile`` live on the env params.
    ``solar_profile`` lives inside the RenewableBundle (zero-action mode) in
    ``params.resources``; this helper rebuilds that bundle's ``profiles``
    field when ``solar_profile`` is updated.  Accepts either ``(T,)`` or
    ``(T, n_devices)`` inputs and broadcasts to the bundle's shape.
    """
    from powerzoojax.envs.resource.renewable import RenewableBundle

    new_resources = params.resources
    if solar_profile is not None:
        sp = jnp.asarray(solar_profile, dtype=jnp.float32)
        new_list = []
        for b in params.resources:
            if isinstance(b, RenewableBundle):
                if sp.ndim == 1:
                    sp_shaped = jnp.broadcast_to(sp[:, None], (sp.shape[0], b.n_devices))
                else:
                    sp_shaped = sp
                new_list.append(b.replace(profiles=sp_shaped))
            else:
                new_list.append(b)
        new_resources = tuple(new_list)
    new_cpu = cpu_profile if cpu_profile is not None else params.cpu_profile
    new_temp = outdoor_temp_profile if outdoor_temp_profile is not None else params.outdoor_temp_profile

    return params.replace(
        resources=new_resources,
        cpu_profile=new_cpu,
        outdoor_temp_profile=new_temp,
    )


def _scale_diesel_capacity(
    params: "DCMicrogridParams",
    factor: float,
) -> "DCMicrogridParams":
    """Return a new DCMicrogridParams with every DieselBundle's p_max scaled."""
    from powerzoojax.envs.resource.diesel import DieselBundle

    new_resources = tuple(
        b.replace(p_max=b.p_max * jnp.float32(factor))
        if isinstance(b, DieselBundle) else b
        for b in params.resources
    )
    return params.replace(resources=new_resources)


def _replace_dc(
    params: "DCMicrogridParams",
    new_dc: "DataCenterParams",
) -> "DCMicrogridParams":
    """Return a new DCMicrogridParams with DataCenterParams replaced."""
    return params.replace(dc=new_dc)
