# 05 — Train MARL

Multi-agent IPPO on a grid env. Each generator and each attached bundle device becomes its own agent. Reference script: [`examples/train_ippo_grid.py`](https://github.com/powerzoojax/PowerZooJax/blob/main/examples/train_ippo_grid.py).

## One-liner via `train`

```python
from powerzoojax.rl import train

result = train("case5-ippo", seed=0)
print(result.summary)
```

The preset wraps `TransGridEnv` with `GridMARLEnv`, exposing 5 unit agents with parameter sharing.

## Explicit env + trainer

```python
import jax
import jax.numpy as jnp

from powerzoojax.case import load_case
from powerzoojax.envs import TransGridEnv, make_trans_params
from powerzoojax.rl import GridMARLEnv, TrainConfig, make_train

case = load_case("5")
env = TransGridEnv()
profiles = jnp.ones((48, case.n_loads), dtype=jnp.float32) * 0.5
params = make_trans_params(case, load_profiles=profiles, max_steps=48)
marl_env = GridMARLEnv(env, params)

config = TrainConfig(
    algo="ippo",
    total_timesteps=200_000,
    num_envs=16,
    n_steps=48,
    hidden_dims=(64, 64),
)

train_fn = make_train(marl_env, config)
result = train_fn(jax.random.PRNGKey(0))
```

## What `GridMARLEnv` exposes

- `obs_dict`: `{"unit_0": ..., "unit_1": ..., ..., "battery_0": ...}`. Generator agents are named `unit_i`; bundle devices are named by resource type.
- `action_dict`: same keys.
- `reward`: scalar shared across all agents.
- `done`: scalar.
- `info`: standard dict, including compatibility diagnostics such as `constraint_costs` and `cost_sum` if the inner env reports constraints.

The wrapper does not duplicate physics. It splits the concatenated action and reassembles per-agent observations.

## Typed parameter sharing for DERs

For heterogeneous agents (e.g. 4 batteries + 4 PVs + 4 flex loads), use `algo="ippo_typed"`. Each agent type has its own `SharedActorCritic`:

```python
from powerzoojax.rl import DistGridMARLEnv

# (constructed via `make_ders_marl_env(...)` in the DERs benchmark)
marl_env = DistGridMARLEnv(env, params, observation_mode="local")

config = TrainConfig(
    algo="ippo_typed",
    total_timesteps=15_000_000,
    num_envs=64,
    n_steps=48,
    hidden_dims=(128, 128),
)
train_fn = make_train(marl_env, config)
result = train_fn(jax.random.PRNGKey(0))
print(list(result.params.keys()))  # ['battery', 'renewable', 'flexload']
```

`observation_mode="local"` exposes a Dec-POMDP K-hop neighborhood per agent, as required by the DERs benchmark spec.

## GenCos competitive market

For the GenCos benchmark, swap the wrapper for `MarketMARLEnv`:

```python
from powerzoojax.envs import make_market_marl_params
from powerzoojax.rl import MarketMARLEnv

profiles = jnp.full((48, case.n_loads), 100.0, dtype=jnp.float32)
mparams = make_market_marl_params(case, profiles, n_segments=3, max_steps=48)
marl_env = MarketMARLEnv(mparams)
```

Per-agent reward is dispatch profit using exact LMPs from the PD-IPM SCED solver. See [Benchmarks → GenCos](../benchmarks/gencos.md).

## What you get back

For IPPO trainers, `TrainResult.params` is either:

- a Flax pytree (full parameter sharing, `algo="ippo"`), or
- a dict keyed by agent type (typed sharing, `algo="ippo_typed"`).

Use `make_ippo_act(env, params)` to get a deterministic per-agent policy callable for evaluation rollouts.
