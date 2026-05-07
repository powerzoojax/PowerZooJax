# Data

!!! note "Python API 签名"
    本页只翻译概览、合约表与示例。完整的 mkdocstrings 自动生成签名（参数、字段、类型）由英文 API 渲染，见 [English API → Data](../../en/api/data.md)。

setup 阶段读取 parquet 时间序列的统一入口。架构层角色见 [Architecture → Data pipeline](../architecture/data-pipeline.md)。`powerzoojax.data` 中的任何内容都不会进入编译后的 rollout。

构造 env 参数时优先用 `load_jax_profiles(...)`，检查曲线用 `load_signals(...)`。`load_data(...)` 面向原始列名布局与直接读表；benchmark 代码以 `load_jax_profiles(...)` 与 [Architecture → Data pipeline](../architecture/data-pipeline.md) 中的 parquet facade 为准。

## 语义加载

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

## 稳定信号名

电力系统信号：

- `LOAD_ACTUAL_MW`
- `LOAD_FORECAST_DA_MW`
- `LOAD_FORECAST_P10_MW`
- `LOAD_FORECAST_P50_MW`
- `LOAD_FORECAST_P90_MW`
- `SOLAR_AVAILABLE_MW`
- `WIND_AVAILABLE_MW`

市场指数信号：

- `MARKET_MID_PRICE_APX`
- `MARKET_MID_PRICE_N2EX`
- `MARKET_MID_VOLUME_APX`
- `MARKET_MID_VOLUME_N2EX`

数据中心信号：

- `DC_CPU_UTIL`
- `DC_MEM_UTIL`
- `DC_NET_IN`
- `DC_NET_OUT`
- `DC_DISK_IO`
- `DC_POWER_MW`
- `DC_GPU_UTIL`
- `DC_GPU_MEM_UTIL`

## 冻结的 split

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

## Ausgrid 工具

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

## 非平稳采样

```python
from powerzoojax.data import EpisodeConfig, NonstationarySampler, apply_drift
```

## 加载器、数据集目录与注册表

`DataLoader` 是 setup 阶段的统一入口；`DatasetManifest` 与 `DatasetRegistry` 管理数据集目录与注册信息；`TimeAligner` 负责时间对齐。完整签名仍由英文 API 页渲染。

## 数据中心微电网曲线

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

`apply_ood_transform(params, scenario)` 产出 [DC Microgrid benchmark](../benchmarks/dc-microgrid.md) 使用的 OOD split。可用场景：`workload_swap`、`workload_shock`、`renewable_drought`、`cooling_stress`、`dg_derating`、`sla_tighten`。
