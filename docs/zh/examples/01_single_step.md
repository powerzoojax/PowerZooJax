# 01 — 单步

最小可运行程序。它加载一个 5 节点输电 case，实例化 `TransGridEnv`，reset 一次，再走一步。参考脚本：[`examples/jax_02_grid_env.py`](https://github.com/powerzoojax/PowerZooJax/blob/main/examples/jax_02_grid_env.py)。

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

## 输出怎么读

- `reward = -reward_scale * gen_cost`。负值是常态（agent 最小化 cost）。
- `costs` 是 core CMDP 向量。对 `TransGridEnv` 来说，名称顺序是 `("thermal_overload", "voltage_violation", "power_balance", "resource")`。
- `info["cost_sum"]` 是聚合诊断量。零表示该步完全可行。
- `state` 已经遵循 auto-reset 语义：当 `done=True` 时，`state` 是下一 episode 的初始 state。

## 切换求解器模式

```python
params_dcopf = make_trans_params(case, max_steps=48, solver_mode=1)  # DCOPF dispatch
params_acopf = make_trans_params(case, max_steps=48, solver_mode=2)  # ACOPF dispatch + AC state
```

`physics=1` 在 agent-dispatch 路径上启用 AC PF。完整模式矩阵见 [API → Grid](../api/grid.md#effective-modes)。

## 挂电池 bundle

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

action 布局变为 `[unit actions | bundle_0 actions]`。env 自动按 `bundle.action_dim` 切分。bundle 如何与栈中其余部分组合，见 [Architecture → Environment stack](../architecture/env-stack.md#layer-3-resource-bundles)。

## 下一步

[02 — Batched rollout](02_batched_rollout.md) 把这一步扩展为一个 JIT 程序内的多个并行 episode。
