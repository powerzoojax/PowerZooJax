# GenCos market MARL

!!! note "Python API 签名"
    本页只翻译概览、合约表与示例。完整的 mkdocstrings 自动生成签名（参数、字段、类型）由英文 API 渲染，见 [English API → GenCos market MARL](../../en/api/market-marl.md)。

GenCos benchmark 的公开 API：5 agent 滚动竞争市场，使用精确 PD-IPM 出清与 ramp 耦合。物理层见 [Physics → Markets](../physics/markets.md#marketmarlenv-gencos-rolling-market)。benchmark 任务定义见 [Benchmarks → GenCos](../benchmarks/gencos.md)。

## 纯函数式核心

```python
from powerzoojax.envs import (
    MarketMARLState,
    MarketMARLParams,
    make_market_marl_params,
    market_marl_reset,
    market_marl_step,
)
```

这些是 JIT 兼容的纯函数。`market_marl_reset` 从 load-profile 池里随机采样 episode 起始 index；`market_marl_step` 通过 `offer_sced` 在 LP 层强制 ramp 上下限。

## MARL 适配器

```python
from powerzoojax.rl import MarketMARLEnv
```

`MarketMARLEnv` 用 IPPO trainer 期待的按 agent dict 布局包装纯核心。Agent 命名为 `genco_i`，每家发电公司一个。Wrapper 不复制物理——只处理 action / observation 切分。

每个 agent 的私有观测维度是 `8 + lmp_history_len`，内容依次是：

- 归一化后的首段基础报价
- 归一化后的自身 `p_max`
- 归一化后的上一步自身 dispatch
- 归一化后的上一步自身 dispatch 利润
- 归一化后的自身 ramp 上行余量
- 归一化后的一步前瞻总负荷 forecast
- `sin(t)`、`cos(t)`
- 均值 LMP 历史

Wrapper 返回的是按 agent 的 reward / done 字典。更底层的纯函数核心返回 `(final_state, done, reward_vec, info)`，市场安全诊断量如 `cost_thermal_overload` 保留在 `info` 中。

## 用法示例

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
