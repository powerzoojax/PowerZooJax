# Exact bid-based SCED

Public API for the exact primal-dual interior-point security-constrained economic dispatch (SCED) solver. For physics, see [Physics → Markets](../physics/markets.md#offer_sced-exact-bid-based-sced).

## Overview

`offer_sced` solves the offer-based SCED LP exactly via a primal-dual interior-point method (PD-IPM). It is JIT-compatible and `vmap`-able. LMPs are recovered from IPM duals and match HiGHS LP marginals to within `< 1e-4 $/MWh` on tested cases.

The setup-time entry point is `prepare_offer_sced(case, n_segments=...)` which builds the segment widths, base prices, line capacity, and PTDF-based mapping matrices. The runtime entry point is `offer_sced(setup, load_mw, offer_prices, p_min_rt=None, p_max_rt=None)`.

## API

```python
from powerzoojax.envs import (
    OfferSCEDSetup,
    OfferSCEDResult,
    prepare_offer_sced,
    offer_sced,
)
```

::: powerzoojax.envs.market.offer_sced.prepare_offer_sced

::: powerzoojax.envs.market.offer_sced.offer_sced

::: powerzoojax.envs.market.offer_sced.OfferSCEDSetup

::: powerzoojax.envs.market.offer_sced.OfferSCEDResult

## Usage example

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

To enforce per-step ramp bounds at the LP level (used by `MarketMARLEnv`), pass `p_min_rt` and `p_max_rt` overrides:

```python
prev = previous_dispatch_mw
p_min_rt = jnp.maximum(setup.p_min, prev - ramp_down_mw)
p_max_rt = jnp.minimum(setup.p_max, prev + ramp_up_mw)
result = offer_sced(setup, nodal_load, offer_prices, p_min_rt=p_min_rt, p_max_rt=p_max_rt)
```
