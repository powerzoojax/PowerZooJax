# RL — multi-agent

Public API for multi-agent wrappers and trainers. For single-agent, see [API → RL](rl.md). For the conceptual overview, see [Training → Trainers](../training/trainers.md#marl-make_ippo_train-and-make_ippo_typed_train).

## Multi-agent base contract

```python
from powerzoojax.rl import MultiAgentEnvironment, MARLState
```

`MultiAgentEnvironment` declares the dict-API contract used by IPPO trainers:

- `reset(key, params)` returns `(obs_dict, state)`.
- `step(key, state, action_dict, params)` returns `(obs_dict, state, reward, done, info)`.

::: powerzoojax.rl.multi_agent.MultiAgentEnvironment

::: powerzoojax.rl.multi_agent.MARLState

## Grid MARL adapters

```python
from powerzoojax.rl import GridMARLEnv, DistGridMARLEnv
```

`GridMARLEnv` wraps `TransGridEnv`. Generator agents are named `unit_i`; bundle devices are named by resource type (`battery_0`, `pv_0`, `flexload_0`, ...). Reward is shared across agents.

`DistGridMARLEnv` wraps `DistGridEnv`. Supports `observation_mode="global"` (default) and `observation_mode="local"` (Dec-POMDP K-hop neighborhood, used by the DERs benchmark).

::: powerzoojax.rl.multi_agent.GridMARLEnv

::: powerzoojax.rl.multi_agent.DistGridMARLEnv

## Market MARL adapter

```python
from powerzoojax.rl import MarketMARLEnv
```

See [API → Market MARL](market-marl.md) for the GenCos task wrapper.

## IPPO trainers

```python
from powerzoojax.rl import make_ippo_train, make_ippo_act
```

- `algo="ippo"` — independent PPO with full parameter sharing.
- `algo="ippo_typed"` — typed parameter sharing: agents partitioned by name prefix (`battery_*`, `renewable_*`, `flexload_*`, ...). Each type has an independent `SharedActorCritic`.

For the DERs benchmark, `algo="ippo_typed"` is required (see [Benchmarks → DERs](../benchmarks/ders.md)).

`make_ippo_act(env, params)` returns a deterministic per-agent action function suitable for evaluation rollouts.

::: powerzoojax.rl.ippo.make_ippo_train

::: powerzoojax.rl.ippo.make_ippo_act
