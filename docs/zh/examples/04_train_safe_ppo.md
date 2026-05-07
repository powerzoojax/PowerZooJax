# 04 — 训练 safe PPO

在同一个物理 env 上做 PPO-Lagrangian 训练。Agent 最大化期望回报，同时把选中的 CMDP cost 控制在任务预算之下。参考脚本：[`examples/train_safe_ppo_transgrid.py`](https://github.com/powerzoojax/PowerZooJax/blob/main/examples/train_safe_ppo_transgrid.py)。

## 一行式 `train`

```python
from powerzoojax.rl import train

result = train("case5-safe-dispatch", seed=0)
print(result.summary)
```

这个 preset 用 `SafeRLWrapper`，其中 `selected_names=("thermal_overload",)`、`cost_thresholds=(0.0,)`，并搭配 `algo="ppo_lagrangian"`。dispatcher 自动路由到 `make_cmdp_train`。

## 显式 env + trainer

```python
import jax
import jax.numpy as jnp

from powerzoojax.case import load_case
from powerzoojax.envs import TransGridEnv, make_trans_params
from powerzoojax.rl import SafeRLWrapper, TrainConfig, make_train

case = load_case("5")
env = TransGridEnv()
profiles = jnp.ones((48, case.n_loads), dtype=jnp.float32) * 0.5
params = make_trans_params(case, load_profiles=profiles, max_steps=48)
wrapped = SafeRLWrapper(
    env,
    params,
    selected_names=("thermal_overload",),
    cost_thresholds=(0.0,),
)

config = TrainConfig(
    algo="ppo_lagrangian",
    total_timesteps=200_000,
    num_envs=32,
    n_steps=48,
    cost_thresholds=(0.0,),
    lambda_lr=5e-3,
    cost_gamma=1.0,
    hidden_dims=(64, 64),
)

train_fn = make_train(wrapped, config)
result = train_fn(jax.random.PRNGKey(0))
```

## `SafeRLWrapper` 改了 env 什么

```python
obs, state = wrapped.reset(key)
obs, state, reward, costs, done, info = wrapped.step(key, state, action)
#                       ^^^^^                      <- 选中的 CMDP cost 向量
```

6-元组正是 `make_cmdp_train` 期待的形式。底层 env 物理不变：内层 env 仍计算完整 cost 向量，`SafeRLWrapper` 只挑出任务相关的那一部分。

## 对偶乘子的作用

PPO-Lagrangian 维护一个非负对偶向量 `lambda`，用它惩罚 cost critic：

```
augmented_advantage = A_R - lambda^T A_C
```

每个分量在 log 空间更新：`log_lambda_i += lambda_lr_i * (episode_cost_est_i - threshold_i)`，然后 `lambda_i = exp(log_lambda_i)`。`log_lambda_i` 被裁剪到 `[-log_lambda_max, +log_lambda_max]` 以防发散。`episode_cost_est_i = mean_cost_i * n_steps` 是每次 update 对约束 `i` 累计 episode cost 的估算。这个例子只选了一个约束，因此向量情形会退化成熟悉的标量情形。

## 怎么选 `cost_threshold`

- 先跑无约束 PPO，测量各个约束的 `mean cost_i`。
- 对硬物理约束使用 `cost_thresholds=(0.0, ...)`。只有任务明确声明为松弛约束或机会约束时，才应使用非零 budget。
- benchmark preset 把 `cost_thresholds` 冻结。`tso-scuc-safe`、`dso-nflex-safe`、`dc-microgrid-safe` 用的值见 [Training → Presets](../training/presets.md)。

## 下一步

[05 — 训练 MARL](05_train_marl.md) 展示在多 agent grid env 上跑 IPPO。
