# Grid environments

!!! note "Python API 签名"
    本页只翻译概览、合约表与示例。完整的 mkdocstrings 自动生成签名（参数、字段、类型）由英文 API 渲染，见 [English API → Grid environments](../../en/api/grid.md)。

`TransGridEnv` 与共享的输电网基础设施（DC / AC power flow、OPF helper）的公开 API。物理层见 [Physics → Transmission](../physics/transmission.md)。辐射状配电 env 见 [API → Distribution](distribution.md)。TSO benchmark 用的 SCUC env 见 [API → Unit commitment](grid-uc.md)。

## `TransGridEnv`

```python
from powerzoojax.envs import TransGridEnv, make_trans_params
```

### 有效模式 {#effective-modes}

| `physics` | `solver_mode` | Dispatch 来源 | 网络 | 安全检查 |
| --- | --- | --- | --- | --- |
| `0` | `0` | agent | DC PF | MW 线路上下限 |
| `1` | `0` | agent | Newton-Raphson AC PF | 视在功率热极限 + 电压上下限 |
| `0` | `1` | DCOPF | DC PF | MW 线路上下限 |
| `1` | `1` | DCOPF | DCOPF + AC PF 事后验证 | 视在功率热极限 + 电压上下限 |
| `0` 或 `1` | `2` | ACOPF | ACOPF | 视在功率热极限 + 电压上下限 |

### Step 合约

- `reward = -reward_scale * gen_cost`。
- `costs = [cost_thermal_overload, cost_voltage_violation, cost_power_balance, cost_resource]`。
- `info["cost_sum"] = sum(costs)` 只是聚合诊断量。
- DC 观测：`[line_flow / cap, load / total_cap, unit_p / p_max, sin(t), cos(t), <bundle_obs>]`。
- AC 观测：`[|S| / cap, vm, load / total_cap, unit_p / p_max, sin(t), cos(t), <bundle_obs>]`。

## 共享状态容器

`GridState` 与 `GridParams` 的完整签名由英文 API 页渲染。

## DC power flow

`dc_power_flow`、`dc_power_flow_with_check`、`safety_check`、`compute_generation_cost` 与 `proportional_dispatch` 的完整签名由英文 API 页渲染。

## AC power flow

`prepare_acpf`、`ac_power_flow`、`ACPFSetup` 与 `ACPFResult` 的完整签名由英文 API 页渲染。

## OPF 模块

`prepare_dcopf`、`dc_opf`、`DCOPFSetup`、`DCOPFResult` 以及 `ac_opf`、`ACOPFSetup` 的完整签名由英文 API 页渲染。
