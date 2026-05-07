# 01 — Single step

The smallest runnable program. It loads a 5-bus transmission case, instantiates `TransGridEnv`, resets, and steps once. Reference script: [`examples/jax_02_grid_env.py`](https://github.com/powerzoojax/PowerZooJax/blob/main/examples/jax_02_grid_env.py).

```python
import jax
import jax.numpy as jnp

from powerzoojax.case import load_case
from powerzoojax.envs import TransGridEnv, make_trans_params

case = load_case("5")
env = TransGridEnv()
params = make_trans_params(case, max_steps=48)

key = jax.random.PRNGKey(0)
obs, state = env.reset(key, params)

action = jnp.zeros(case.n_units, dtype=jnp.float32)
key, k_step = jax.random.split(key)
obs, state, reward, costs, done, info = env.step(k_step, state, action, params)

print("obs shape:", obs.shape)
print("time step:", int(state.time_step))
print("reward:  ", float(reward))
print("names:   ", env.constraint_names(params))
print("costs:   ", costs)
print("cost_sum:", float(info["cost_sum"]))
print("done:    ", bool(done))
```

## What to read in the output

- `reward = -reward_scale * gen_cost`. Negative is normal (the agent minimizes cost).
- `costs` is the core CMDP vector. For `TransGridEnv` the names are `("thermal_overload", "voltage_violation", "power_balance", "resource")`.
- `info["cost_sum"]` is the aggregate diagnostic. Zero means the step was fully feasible.
- `state` already follows the auto-reset semantics: when `done=True`, `state` is the freshly reset initial state of the next episode.

## Switching the solver mode

```python
params_dcopf = make_trans_params(case, max_steps=48, solver_mode=1)  # DCOPF dispatch
params_acopf = make_trans_params(case, max_steps=48, solver_mode=2)  # ACOPF dispatch + AC state
```

`physics=1` activates AC PF for the agent-dispatch path. See [API → Grid](../api/grid.md#effective-modes) for the full mode matrix.

## Attaching a battery bundle

```python
from powerzoojax.envs.resource.battery import make_battery_bundle

bundle = make_battery_bundle(
    case,
    bus_ids=jnp.array([1], dtype=jnp.int32),
    capacity_mwh=jnp.array([4.0]),
    power_mw=jnp.array([2.0]),
)
params = make_trans_params(case, max_steps=48, resources=(bundle,))

action_dim = case.n_units + bundle.action_dim
action = jnp.zeros(action_dim, dtype=jnp.float32)
obs, state, reward, costs, done, info = env.step(k_step, state, action, params)
```

The action layout becomes `[unit actions | bundle_0 actions]`. The env splits it by `bundle.action_dim` automatically. See [Architecture → Environment stack](../architecture/env-stack.md#layer-3-resource-bundles) for how bundles compose with the rest of the stack.

## Next step

[02 — Batched rollout](02_batched_rollout.md) scales this single step into many parallel episodes inside one JIT-compiled program.
