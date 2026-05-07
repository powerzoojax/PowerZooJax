# Market Lite

!!! note "Python API 签名"
    本页只翻译概览、合约表与示例。完整的 mkdocstrings 自动生成签名（参数、字段、类型）由英文 API 渲染，见 [English API → Market Lite](../../en/api/market.md)。

储能套利市场 env 的公开 API。两个 env 关注的是价格-物理耦合，不是完整的 ISO 市场栈。物理层见 [Physics → Markets](../physics/markets.md)。精确 bid-based SCED 求解器见 [API → Market SCED](market-sced.md)。GenCos 竞争市场见 [API → Market MARL](market-marl.md)。

## `CostBasedMarketEnv`

```python
from powerzoojax.envs import (
    CostBasedMarketEnv,
    CostMarketState,
    CostMarketParams,
    make_cost_market_params,
)
```

发电机 dispatch 由 `dc_opf` 用真实边际成本求解。Reward 是储能营收 `sum(LMP * P * dt)`。core CMDP 向量是 `costs = (cost_thermal_overload,)`，其静态名称为 `("thermal_overload",)`；`info["cost_sum"]` 是聚合诊断量。

## `BidBasedMarketEnv`

```python
from powerzoojax.envs import (
    BidBasedMarketEnv,
    BidMarketState,
    BidMarketParams,
    make_bid_market_params,
)
```

在 offer 曲线上做**近似（启发式）分段 ED** 出清。反推 LMP 是基于真实边际成本的 KKT 风格近似（不是精确对偶价格）。论文级 LMP 用 [Market SCED](market-sced.md) 中的精确 PD-IPM 求解器。

## 别名导出

`SimpleLMPArbitrageEnv`、`LMPMarketState`、`LMPMarketParams` 与 `make_lmp_market_params` 与 cost-based 类型指向同一实现，仅符号名不同。

## 共享市场状态

共享状态容器 `MarketState` 与 `MarketParams` 的完整签名由英文 API 页渲染；本页只保留概览层说明。

## 近似出清 helper（启发式分段 ED）

```python
from powerzoojax.envs import make_cost_segments, piecewise_ed
from powerzoojax.envs.market.clearing import (
    prepare_piecewise_ed,
    PiecewiseEDSetup,
    PiecewiseEDResult,
)
```
