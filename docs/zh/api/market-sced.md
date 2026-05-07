# 精确 bid-based SCED

!!! note "Python API 签名"
    本页只翻译概览、合约表与示例。完整的 mkdocstrings 自动生成签名（参数、字段、类型）由英文 API 渲染，见 [English API → Exact bid-based SCED](../../en/api/market-sced.md)。

精确 primal-dual interior-point security-constrained economic dispatch（SCED）求解器的公开 API。物理层见 [Physics → Markets](../physics/markets.md#offer_sced-exact-bid-based-sced)。

## 概览

`offer_sced` 通过 primal-dual interior-point 方法（PD-IPM）**精确**求解基于报价的 SCED LP。它兼容 JIT、可 vmap。LMP 由 IPM 对偶反推，在测试 case 上与 HiGHS LP 边际值差距 `< 1e-4 $/MWh`。

setup 阶段入口是 `prepare_offer_sced(case, n_segments=...)`，它构造分段宽度、基础价格、线路容量与 PTDF 映射矩阵。运行时入口是 `offer_sced(setup, load_mw, offer_prices, p_min_rt=None, p_max_rt=None)`。

## API

```python
from powerzoojax.envs import (
    OfferSCEDSetup,
    OfferSCEDResult,
    prepare_offer_sced,
    offer_sced,
)
```

## 用法示例

```python
import jax
import jax.numpy as jnp

from powerzoojax.case import load_case
from powerzoojax.envs import prepare_offer_sced, offer_sced

case = load_case("5")
setup = prepare_offer_sced(case, n_segments=3)

load_mw = jnp.full((case.n_loads,), 100.0)
nodal_load = case.nodes_loads_map @ load_mw
offer_prices = setup.base_seg_prices  # truthful

result = offer_sced(setup, nodal_load, offer_prices)
print("dispatch:", result.unit_power)
print("lmp:", result.lmp)
print("converged:", result.converged)
```

如需在 LP 层强制每步 ramp 上下限（`MarketMARLEnv` 在使用此功能），传入 `p_min_rt` 与 `p_max_rt` 进行覆盖：

```python
prev = previous_dispatch_mw
p_min_rt = jnp.maximum(setup.p_min, prev - ramp_down_mw)
p_max_rt = jnp.minimum(setup.p_max, prev + ramp_up_mw)
result = offer_sced(setup, nodal_load, offer_prices, p_min_rt=p_min_rt, p_max_rt=p_max_rt)
```
