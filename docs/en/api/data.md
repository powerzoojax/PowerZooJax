# Data

Setup-time facade for parquet-backed time series. For the architecture-level role, see [Architecture → Data pipeline](../architecture/data-pipeline.md). Nothing in `powerzoojax.data` enters the compiled rollout.

Prefer `load_jax_profiles(...)` for env construction and `load_signals(...)` for inspection. `load_data(...)` covers raw-column layouts and direct table access; benchmark code standardizes on `load_jax_profiles(...)` and the parquet facade documented in [Architecture → Data pipeline](../architecture/data-pipeline.md).

## Semantic loading

```python
from powerzoojax.data import DataLoader, signals as S

loader = DataLoader()
profiles = loader.load_jax_profiles(
    [S.LOAD_ACTUAL_MW, S.SOLAR_AVAILABLE_MW],
    source="gb",
    start_date="2025-04-01",
    end_date="2025-12-31",
    resample="30min",
)
```

## Stable signal names

Power-system signals:

- `LOAD_ACTUAL_MW`
- `LOAD_FORECAST_DA_MW`
- `LOAD_FORECAST_P10_MW`
- `LOAD_FORECAST_P50_MW`
- `LOAD_FORECAST_P90_MW`
- `SOLAR_AVAILABLE_MW`
- `WIND_AVAILABLE_MW`

Market index signals:

- `MARKET_MID_PRICE_APX`
- `MARKET_MID_PRICE_N2EX`
- `MARKET_MID_VOLUME_APX`
- `MARKET_MID_VOLUME_N2EX`

Data-center signals:

- `DC_CPU_UTIL`
- `DC_MEM_UTIL`
- `DC_NET_IN`
- `DC_NET_OUT`
- `DC_DISK_IO`
- `DC_POWER_MW`
- `DC_GPU_UTIL`
- `DC_GPU_MEM_UTIL`

## Loader, manifest, registry

::: powerzoojax.data.data_loader.DataLoader

::: powerzoojax.data.manifest.DatasetManifest

::: powerzoojax.data.registry.DatasetRegistry

::: powerzoojax.data.alignment.TimeAligner

## Frozen splits

```python
from powerzoojax.data import (
    GB_TRAIN_START, GB_TRAIN_END,
    GB_IID_START, GB_IID_END,
    AUSGRID_TRAIN_START, AUSGRID_TRAIN_END,
    AUSGRID_IID_START, AUSGRID_IID_END,
    AUSGRID_SUMMER_START, AUSGRID_SUMMER_END,
    gb_windows,
    ausgrid_windows,
)
```

::: powerzoojax.data.splits.gb_windows

::: powerzoojax.data.splits.ausgrid_windows

## Ausgrid utilities

```python
from powerzoojax.data import (
    AUSGRID_FEEDER_POOLS,
    AUSGRID_TRAIN_POOL,
    AUSGRID_ZONE_HOLDOUT_POOL,
    get_ausgrid_split,
    get_feeder_substations,
    select_full_coverage_substations,
)
```

::: powerzoojax.data.ausgrid_utils.get_ausgrid_split

::: powerzoojax.data.ausgrid_utils.get_feeder_substations

::: powerzoojax.data.ausgrid_utils.select_full_coverage_substations

## Non-stationary sampling

```python
from powerzoojax.data import EpisodeConfig, NonstationarySampler, apply_drift
```

::: powerzoojax.data.nonstationary.EpisodeConfig

::: powerzoojax.data.nonstationary.NonstationarySampler

::: powerzoojax.data.nonstationary.apply_drift

## DC microgrid profiles

```python
from powerzoojax.data import (
    cycle_profile,
    make_synthetic_cpu_profile,
    make_synthetic_solar_profile,
    make_synthetic_outdoor_temp_profile,
    make_all_synthetic_profiles,
    load_workload_profiles,
    apply_ood_transform,
    VALID_SOURCES,
    VALID_OOD_SCENARIOS,
)
```

`apply_ood_transform(params, scenario)` produces the OOD splits used by the [DC Microgrid benchmark](../benchmarks/dc-microgrid.md). Valid scenarios: `workload_swap`, `workload_shock`, `renewable_drought`, `cooling_stress`, `dg_derating`, `sla_tighten`.

::: powerzoojax.data.dc_microgrid_profiles.cycle_profile

::: powerzoojax.data.dc_microgrid_profiles.make_all_synthetic_profiles

::: powerzoojax.data.dc_microgrid_profiles.load_workload_profiles

::: powerzoojax.data.dc_microgrid_profiles.apply_ood_transform
