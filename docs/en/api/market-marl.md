# GenCos market MARL

Public API for the GenCos benchmark: a 5-agent rolling competitive market with exact PD-IPM clearing and ramp coupling. For physics, see [Physics → Markets](../physics/markets.md#marketmarlenv-gencos-rolling-market). For the benchmark task definition, see [Benchmarks → GenCos](../benchmarks/gencos.md).

## Pure-functional core

```python
from powerzoojax.envs import (
    MarketMARLState,
    MarketMARLParams,
    make_market_marl_params,
    market_marl_reset,
    market_marl_step,
)
```

These are JIT-compatible pure functions. `market_marl_reset` samples a random episode start index from the load-profile pool; `market_marl_step` enforces ramp bounds at the LP level via `offer_sced`.

::: powerzoojax.envs.market.market_marl_core.MarketMARLState

::: powerzoojax.envs.market.market_marl_core.MarketMARLParams

::: powerzoojax.envs.market.market_marl_core.make_market_marl_params

::: powerzoojax.envs.market.market_marl_core.market_marl_reset

::: powerzoojax.envs.market.market_marl_core.market_marl_step

## MARL adapter

```python
from powerzoojax.rl import MarketMARLEnv
```

`MarketMARLEnv` wraps the pure core with the per-agent dict layout used by IPPO trainers. Agents are named `genco_i`, one per generator. The wrapper does not duplicate physics — it only handles action / observation splitting.

Private observation layout per agent is `8 + lmp_history_len`:

- normalized first-segment base price
- normalized own `p_max`
- normalized own last dispatch
- normalized own last dispatch profit
- normalized own ramp-up headroom
- normalized one-step-ahead total-load forecast
- `sin(t)`, `cos(t)`
- mean-LMP history

The wrapper returns per-agent reward and done dictionaries. The underlying pure core returns `(final_state, done, reward_vec, info)` and keeps market safety diagnostics such as `cost_thermal_overload` in `info`.

::: powerzoojax.rl.market_marl.MarketMARLEnv

## Usage example

```python
import jax
import jax.numpy as jnp

from powerzoojax.case import load_case
from powerzoojax.envs import make_market_marl_params, market_marl_reset, market_marl_step
from powerzoojax.rl import MarketMARLEnv

case = load_case("5")
profiles = jnp.full((48, case.n_loads), 100.0, dtype=jnp.float32)
params = make_market_marl_params(case, profiles, n_segments=3, max_steps=48)

# 1. functional core (no per-agent layout)
state = market_marl_reset(jax.random.PRNGKey(0), params)
flat_actions = jnp.zeros(case.n_units * 3)
state, done, reward_vec, info = market_marl_step(
    jax.random.PRNGKey(1), state, flat_actions, params
)

# 2. MARL wrapper (per-agent dict)
env = MarketMARLEnv(params)
obs_dict, state = env.reset(jax.random.PRNGKey(2))
action_dict = {f"genco_{i}": jnp.zeros(3) for i in range(case.n_units)}
obs_dict, state, reward_dict, done_dict, info = env.step(
    jax.random.PRNGKey(3), state, action_dict
)
```
