# Unit commitment (TSO)

!!! note "Python API 签名"
    本页只翻译概览、合约表与示例。完整的 mkdocstrings 自动生成签名（参数、字段、类型）由英文 API 渲染，见 [English API → Unit commitment](../../en/api/grid-uc.md)。

`UnitCommitmentEnv` 与 TSO benchmark 工厂的公开 API。物理层见 [Physics → Transmission](../physics/transmission.md#unitcommitmentenv-scuc-for-the-tso-task)。

## `UnitCommitmentEnv`

```python
from powerzoojax.envs import (
    UnitCommitmentEnv,
    UCState,
    UCParams,
    make_uc_params,
)
```

### Step 合约

- Action：`Box(2 * n_units)`，范围 `[-1, 1]` = `[commitment_signal (n_units) | dispatch_target (n_units)]`。
- Reward：`-reward_scale * (gen_cost + startup_cost + no_load_cost)`。
- 约束 cost：`costs = (w_th * cost_thermal_overload, cost_reserve_shortfall, cost_min_updown)`。当前 `cost_min_updown` 因 mask 始终为 0。

### 智能体观测

`[unit_status | time_in_state_norm | last_dispatch_norm | unit_cost_b_norm | line_flow_norm | load_norm | reserve_ratio | sin(t) | cos(t)]`。总维度 `4 * n_units + n_lines + 4`。
它表示策略每步拿到的输入向量，不是额外的一套原始遥测接口。逐字段语义见 [Physics → Transmission](../physics/transmission.md#智能体观测)。

## TSO 工厂

```python
from powerzoojax.envs import (
    make_tso_case118_params,
    make_tso_case14_params,
    make_tso_ed_params,
    make_tso_uc_params,
    make_tso_scuc_params,
)
```

## 非学习式 baseline

```python
from powerzoojax.envs import tso_all_on_rollout, tso_merit_order_rollout
```

## TSO 净负荷 helper

净负荷曲线由任务工厂在 setup 阶段构造。完整签名见英文 API 页中的 `TSO net-load helpers`。

## 指标

任务级聚合指标（如 `total_operating_cost`、`reserve_shortfall_mw`）由 `compute_tso_metrics` 统一计算；完整签名见英文 API 页中的 `Metrics`。
