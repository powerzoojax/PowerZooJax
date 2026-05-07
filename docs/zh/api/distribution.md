# Distribution environments

!!! note "Python API 签名"
    本页只翻译概览、合约表与示例。完整的 mkdocstrings 自动生成签名（参数、字段、类型）由英文 API 渲染，见 [English API → Distribution environments](../../en/api/distribution.md)。

辐射状配电 env 与它们的前推回代法求解器的公开 API。物理层见 [Physics → Distribution](../physics/distribution.md)。

## `DistGridEnv`

```python
from powerzoojax.envs import DistGridEnv, make_dist_params
```

### Step 合约

- `reward = -loss_penalty_weight * p_loss_MW`。
- `costs = [cost_voltage_violation, cost_thermal_overload, cost_resource]`。
- `info["cost_sum"] = sum(costs)` 是聚合诊断量。
- DSO benchmark 在 task / wrapper 层只选择 `"voltage_violation"`；`cost_mode` 已弃用，仅保留给旧配置加载。
- `info["cost_continuous"]` 是连续型电压 / 视在功率过载诊断量。
- `info["soc_terminal_sq"]` 在终止 transition 上、且第一个挂上的 bundle 是 battery bundle 时填充。

### 观察布局

`[v_mag_norm | p_branch_norm | q_branch_norm | p_load_norm | q_load_norm | sin(t) | cos(t) | <bundle_obs>]`。
它表示 `DistGridEnv` 每步返回给策略的输入向量。字段语义以及 DSO 里 `FlexLoadBundle` 的 observation 切片见 [Physics → Distribution](../physics/distribution.md#distgridenv-balanced-radial-feeder)。

## `DistGrid3PhaseEnv`

```python
from powerzoojax.envs import DistGrid3PhaseEnv, make_dist_3phase_params
```

### Step 合约

- `reward = -loss_penalty_weight * p_loss_MW`。
- `costs = [cost_voltage_violation, cost_thermal_overload, cost_vuf_violation, cost_resource]`。
- `info["cost_sum"] = sum(costs)` 是聚合诊断量。
- `info["max_vuf_percent"]` 保留每节点最大 VUF（Fortescue 电压不平衡度，% 单位）。

## 平衡 BFS 求解器

平衡配电潮流的 `prepare_bfs`、`bfs_power_flow` 与 `BFSTopoData` 完整签名由英文 API 页渲染。

## 三相 BFS 求解器

三相配电潮流的 `build_3phase_topology`、`bfs_3phase_power_flow` 与 `ThreePhaseTopoData` 完整签名由英文 API 页渲染。
