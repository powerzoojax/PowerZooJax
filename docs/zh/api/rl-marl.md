# RL —— 多 agent

!!! note "Python API 签名"
    本页只翻译概览、合约表与示例。完整的 mkdocstrings 自动生成签名（参数、字段、类型）由英文 API 渲染，见 [English API → RL — multi-agent](../../en/api/rl-marl.md)。

多 agent wrapper 与 trainer 的公开 API。单 agent 见 [API → RL](rl.md)。概念性概览见 [Training → Trainers](../training/trainers.md#marl-make_ippo_train-and-make_ippo_typed_train)。

## 多 agent 基础合约

```python
from powerzoojax.rl import MultiAgentEnvironment, MARLState
```

`MultiAgentEnvironment` 声明 IPPO trainer 用的 dict-API 合约：

- `reset(key, params)` 返回 `(obs_dict, state)`。
- `step(key, state, action_dict, params)` 返回 `(obs_dict, state, reward, done, info)`。

## Grid MARL 适配器

```python
from powerzoojax.rl import GridMARLEnv, DistGridMARLEnv
```

`GridMARLEnv` 包装 `TransGridEnv`。发电机 agent 命名为 `unit_i`；bundle 设备按资源类型命名（`battery_0`、`pv_0`、`flexload_0`、…）。reward 在 agent 间共享。

`DistGridMARLEnv` 包装 `DistGridEnv`。支持 `observation_mode="global"`（默认）与 `observation_mode="local"`（Dec-POMDP K-hop 邻域，DERs benchmark 用）。

## Market MARL 适配器

```python
from powerzoojax.rl import MarketMARLEnv
```

GenCos 任务的 wrapper 见 [API → Market MARL](market-marl.md)。

## IPPO trainer

```python
from powerzoojax.rl import make_ippo_train, make_ippo_act
```

- `algo="ippo"` —— 完全参数共享的 independent PPO。
- `algo="ippo_typed"` —— Typed 参数共享：按名字前缀（`battery_*`、`renewable_*`、`flexload_*`、…）分组。每类有独立 `SharedActorCritic`。

DERs benchmark 必须用 `algo="ippo_typed"`（见 [Benchmarks → DERs](../benchmarks/ders.md)）。

`make_ippo_act(env, params)` 返回适合评测 rollout 的确定性按 agent 动作函数。
