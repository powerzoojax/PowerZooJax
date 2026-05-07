# Market Lite

Public API for the storage-arbitrage market envs. Both focus on price-physics coupling rather than a full ISO market stack. For physics, see [Physics → Markets](../physics/markets.md). For the exact bid-based SCED solver, see [API → Market SCED](market-sced.md). For the GenCos competitive market, see [API → Market MARL](market-marl.md).

## `CostBasedMarketEnv`

```python
from powerzoojax.envs import (
    CostBasedMarketEnv,
    CostMarketState,
    CostMarketParams,
    make_cost_market_params,
)
```

Generator dispatch comes from `dc_opf` using true marginal costs. Reward is storage revenue `sum(LMP * P * dt)`. The core CMDP vector is `costs = (cost_thermal_overload,)`, with static name `("thermal_overload",)`. `info["cost_sum"]` is the aggregate diagnostic.

::: powerzoojax.envs.market.cost_based_market.CostBasedMarketEnv
    options:
      show_source: false
      members:
        - reset
        - step
        - observation_space
        - action_space

::: powerzoojax.envs.market.cost_based_market.CostMarketState

::: powerzoojax.envs.market.cost_based_market.CostMarketParams

::: powerzoojax.envs.market.cost_based_market.make_cost_market_params

## `BidBasedMarketEnv`

```python
from powerzoojax.envs import (
    BidBasedMarketEnv,
    BidMarketState,
    BidMarketParams,
    make_bid_market_params,
)
```

**Approximate (heuristic) piecewise ED** clearing on offer curves. Recovered LMPs are KKT-style approximations on true marginal cost (not exact dual prices). For paper-quality LMPs use the exact PD-IPM solver in [Market SCED](market-sced.md).

::: powerzoojax.envs.market.bid_based_market.BidBasedMarketEnv
    options:
      show_source: false
      members:
        - reset
        - step
        - observation_space
        - action_space

::: powerzoojax.envs.market.bid_based_market.BidMarketState

::: powerzoojax.envs.market.bid_based_market.BidMarketParams

::: powerzoojax.envs.market.bid_based_market.make_bid_market_params

## Alias exports

`SimpleLMPArbitrageEnv`, `LMPMarketState`, `LMPMarketParams`, and `make_lmp_market_params` are aliases of the cost-based equivalents. Either import path refers to the same implementation.

## Shared market state

::: powerzoojax.envs.market.base.MarketState

::: powerzoojax.envs.market.base.MarketParams

## Approximate clearing helpers (heuristic piecewise ED)

```python
from powerzoojax.envs import make_cost_segments, piecewise_ed
from powerzoojax.envs.market.clearing import (
    prepare_piecewise_ed,
    PiecewiseEDSetup,
    PiecewiseEDResult,
)
```

::: powerzoojax.envs.market.clearing.make_cost_segments

::: powerzoojax.envs.market.clearing.prepare_piecewise_ed

::: powerzoojax.envs.market.clearing.piecewise_ed

::: powerzoojax.envs.market.clearing.PiecewiseEDSetup

::: powerzoojax.envs.market.clearing.PiecewiseEDResult
