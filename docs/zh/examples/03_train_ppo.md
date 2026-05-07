# 03 — 训练 PPO

通过内置 trainer 跑单 agent PPO。参考脚本：[`examples/train_ppo_battery.py`](https://github.com/powerzoojax/PowerZooJax/blob/main/examples/train_ppo_battery.py) 与 [`examples/train_ppo_transgrid.py`](https://github.com/powerzoojax/PowerZooJax/blob/main/examples/train_ppo_transgrid.py)。

## 一行式 `train`

```python
from powerzoojax.rl import train

result = train("battery-soc-tracking", seed=0)
print(result.summary)
```

`train(preset_name, **overrides)` 解析 preset、构造 env、构造 `TrainConfig`，并运行 `make_train(env, config)`。完整目录见 [Training → Presets](../training/presets.md)。

## 显式 env + trainer

```python
import jax
import jax.numpy as jnp

from powerzoojax.case import load_case
from powerzoojax.envs import TransGridEnv, make_trans_params
from powerzoojax.rl import LogWrapper, TrainConfig, make_train

case = load_case("5")
env = TransGridEnv()
profiles = jnp.ones((48, case.n_loads), dtype=jnp.float32) * 0.5
params = make_trans_params(case, load_profiles=profiles, max_steps=48)
wrapped = LogWrapper(env, params)

config = TrainConfig(
    algo="ppo",
    total_timesteps=200_000,
    num_envs=32,
    n_steps=48,
    learning_rate=3e-4,
    gamma=0.99,
    hidden_dims=(64, 64),
)

train_fn = make_train(wrapped, config)
result = train_fn(jax.random.PRNGKey(0))
print("trained params keys:", list(result.params.keys()))
```

## Trainer 返回什么

`TrainResult` 暴露：

- `params` —— 学到的 policy 参数（Flax pytree）。
- `summary` —— 简短文字总结。
- `learning_curve` —— 每次 eval 的平均回报（当 `eval_freq > 0`）。

用标准 JAX 序列化保存（`flax.serialization.to_bytes` 或 `pickle`），或对相关数组调用 `jnp.save(...)`。

## 自定义 reward

用 `RewardWrapper` 在不修改 env 的前提下注入自定义标量 reward：

```python
from powerzoojax.rl import RewardWrapper

def soc_tracking(state, action, next_state, reward, info):
    return -jnp.abs(next_state.resource_states[0].soc - 0.5)

wrapped = RewardWrapper(LogWrapper(env, params), reward_fn=soc_tracking)
train_fn = make_train(wrapped, config)
```

原始 env reward 保留在 `info["env_reward"]`。

## 下一步

[04 — 训练 safe PPO](04_train_safe_ppo.md) 展示直接使用所选 `costs` 向量的 CMDP 路径。
