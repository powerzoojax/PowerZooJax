# Data center microgrid

!!! note "Python API 签名"
    本页只翻译概览、合约表与示例。完整的 mkdocstrings 自动生成签名（参数、字段、类型）由英文 API 渲染，见 [English API → Data center microgrid](../../en/api/microgrid.md)。

`DataCenterMicrogridEnv` 与它组合的柴油机 helper 的公开 API。物理层见 [Physics → Microgrid](../physics/microgrid.md)。

PowerZooJax 当前公开的微电网 API surface 只有这一条：表后的 `DataCenterMicrogridEnv`。它表示一个由数据中心负荷、电池、PV 和柴油机构成的自给式微电网。

因此，本页讲的是这个具体环境的 API，而不是泛化的 microgrid 概念说明。若看物理语义与系统层描述，请转到 [Physics → Microgrid](../physics/microgrid.md)。

## `DataCenterMicrogridEnv`

```python
from powerzoojax.envs import (
    DataCenterMicrogridEnv,
    DCMicrogridState,
    DCMicrogridParams,
    make_dcmicrogrid_params,
    make_dcmicrogrid_params_with_profiles,
)
```

### Step 合约

- Action：5 维 `Box` `[train_sched, ft_sched, cooling_norm, batt_norm, dg_norm]`。`batt_norm in [-1, 1]`（正 = 放电）；其他在 `[0, 1]`。
- 观测：24 维，包含工作负载、热、能源资产、功率平衡、电池 headroom、电网价格与时间通道（见 [Physics → Microgrid](../physics/microgrid.md#observation-24-d)）。
- Reward：标量 `r_energy + w_cost * r_cost + w_carbon * r_carbon`。`info["reward_vector"] = [r_energy, r_cost, r_carbon]` 另外暴露未加权分量。
- 约束 cost：`costs = (cost_sla, cost_overtemp, cost_power_deficit)`，`info["cost_sum"]` 是聚合诊断量。

其中 `train_sched` 和 `ft_sched` 表示串行的 GPU 预算比例，不是固定 GPU 数量：先由 training 吃掉当前余量的一部分，再由 finetuning 基于“剩余余量”继续分配。精确语义见 [Physics → Microgrid](../physics/microgrid.md#action)。

## 柴油机 helper

```python
from powerzoojax.envs import (
    DieselParams,
    compute_dg_power,
    compute_dg_fuel_cost,
    compute_dg_emissions,
)
```

`DieselParams` 含 `p_dg_max_mw`、`fuel_cost_per_mwh` 与 `emission_factor`（kgCO2 / kWh）。三个 helper 都是纯函数。
