# 05 — 训练 MARL

在 grid env 上跑多 agent IPPO。每个发电机以及每台挂上的 bundle 设备各成为一个 agent。参考脚本：[`examples/train_ippo_grid.py`](https://github.com/powerzoojax/PowerZooJax/blob/main/examples/train_ippo_grid.py)。

## 一行式 `train`

```python
from powerzoojax.rl import train

result = train("case5-ippo", seed=0)
print(result.summary)
```

这个 preset 用 `GridMARLEnv` 包装 `TransGridEnv`，暴露 5 个机组 agent 并参数共享。

## 显式 env + trainer

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

## `GridMARLEnv` 暴露什么

- `obs_dict`：`{"unit_0": ..., "unit_1": ..., ..., "battery_0": ...}`。发电机 agent 命名为 `unit_i`；bundle 设备按资源类型命名。
- `action_dict`：相同 key。
- `reward`：所有 agent 共享的标量。
- `done`：标量。
- `info`：标准 dict；如果内层 env 报告了约束，还会包含 `constraint_costs`、`cost_sum` 之类的兼容诊断字段。

Wrapper 不复制物理。它把拼接后的 action 拆开，并按 agent 重新组装观测。

## DERs 用 typed 参数共享

异质 agent（例如 4 电池 + 4 PV + 4 柔性负荷）用 `algo="ippo_typed"`。每个 agent 类型有自己的 `SharedActorCritic`：

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

`observation_mode="local"` 给每个 agent 暴露一个 Dec-POMDP K-hop 邻域，是 DERs benchmark 规格要求的设置。

## GenCos 竞争市场

GenCos benchmark 把 wrapper 换成 `MarketMARLEnv`：

```python
from powerzoojax.envs import make_market_marl_params
from powerzoojax.rl import MarketMARLEnv

profiles = jnp.full((48, case.n_loads), 100.0, dtype=jnp.float32)
mparams = make_market_marl_params(case, profiles, n_segments=3, max_steps=48)
marl_env = MarketMARLEnv(mparams)
```

按 agent reward 是用 PD-IPM SCED 求解器精确 LMP 计算的 dispatch 利润。详见 [Benchmarks → GenCos](../benchmarks/gencos.md)。

## 你能拿到什么

对 IPPO trainer，`TrainResult.params` 要么是：

- 一个 Flax pytree（完全参数共享，`algo="ippo"`），
- 或按 agent 类型 key 的 dict（typed 共享，`algo="ippo_typed"`）。

用 `make_ippo_act(env, params)` 取得评测 rollout 用的确定性按 agent policy callable。
