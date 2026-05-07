# Resources

!!! note "Python API 签名"
    本页只翻译概览、关系表与简要说明。完整的 mkdocstrings 自动生成签名由英文 API 页面渲染，见 [English API → Resources](../../en/api/resource.md)。

公开的是两类资源接口：

- 独立 `*Env`：单设备自己运行
- `*Bundle`：通过 `params.resources` 挂到 grid / market / microgrid env

并不是每种资源都同时提供这两种形式：

| 资源 | 独立 env | Bundle |
| --- | --- | --- |
| Battery | `BatteryEnv` | `BatteryBundle` |
| Renewable | `RenewableEnv` / `SolarEnv` / `WindEnv` | `RenewableBundle` |
| Vehicle | `VehicleEnv` | 无公开 bundle |
| FlexLoad | `FlexLoadEnv` | `FlexLoadBundle` |
| Diesel | 无独立 env | `DieselBundle` |
| Data center | `DataCenterEnv` | 无公开 bundle |

物理语义见 [Physics → Resources](../physics/resources.md)。组合的数据中心微电网见 [API → Microgrid](microgrid.md)。

## 各资源的 cost 语义

| Resource | `reward` | `costs` |
| --- | --- | --- |
| `BatteryEnv` | `0.0` | 循环吞吐 cost |
| `RenewableEnv` | `0.0` | 始终 `0.0` |
| `VehicleEnv` | `0.0` | departure SOC 差额 |
| `FlexLoadEnv` | `0.0` | curtail + 积压 demand 不舒适度 + 同时激活惩罚 |
| `DataCenterEnv` | `0.0` | 过期任务 SLA 密度 |

不并入 `costs` 的诊断量：

- `BatteryEnv`: `cost_action_clip`
- `DataCenterEnv`: `cost_overtemp`
- `DieselBundle`: `fuel_cost`, `carbon_kg`

## 各资源入口

### `BatteryEnv` 与 `BatteryBundle`

```python
from powerzoojax.envs import BatteryEnv
from powerzoojax.envs.resource.battery import BatteryBundle, make_battery_bundle
```

符号约定：`P > 0` 放电注入，`P < 0` 充电吸收。可行功率由变流器额定值和 SOC 余量共同裁剪。

### `RenewableEnv`、`SolarEnv`、`WindEnv` 与 `RenewableBundle`

```python
from powerzoojax.envs import RenewableEnv, SolarEnv, WindEnv
from powerzoojax.envs.resource.renewable import RenewableBundle, make_renewable_bundle
```

动作 `a in [-1, 1]` 映射到限发 `(1 - a) / 2`。若开启 `enable_q_control=True`，则无功受逆变器 PQ-circle 约束。

### `VehicleEnv`

```python
from powerzoojax.envs import VehicleEnv
```

按行程驱动的 SOC 动力学。出发时扣行程能量；仅当 `is_home=1` 时允许充电 / V2G 放电。`info["cost"]` 在出发 SOC 低于 `soc_departure_min` 时惩罚。

### `FlexLoadEnv` 与 `FlexLoadBundle`

```python
from powerzoojax.envs import FlexLoadEnv
from powerzoojax.envs.resource.flexload import FlexLoadBundle, make_flexload_bundle
```

两个动作：现在削减、现在移出需求。被移出的 demand 会在后续 `shift_horizon` 步内释放回来。

### `DieselBundle`

```python
from powerzoojax.envs.resource.diesel import (
    DieselParams,
    DieselBundle,
    DieselBundleState,
    make_diesel_bundle,
    compute_dg_power,
    compute_dg_fuel_cost,
    compute_dg_emissions,
)
```

柴油机当前公开的是 pure helpers 加 bundle，没有独立 `DieselEnv`。每台设备一个 `[0, 1]` 动作，映射到 `[0, p_max]` 的有功出力；`cost_info["cost"]` 为零，燃料成本和碳排单独报告。

### `DataCenterEnv`

```python
from powerzoojax.envs import DataCenterEnv, make_datacenter_params
```

三层耦合：IT 功率、冷却功率、机房热动力学。任务到达进入固定容量 buffer，调度器按 EDF 风格分配 GPU。

## 基础协议

英文 API 页还提供这些基础类型的完整签名：

- `ResourceState`
- `ResourceParams`
- `ResourceBundle`
- `ResourceBundleState`
