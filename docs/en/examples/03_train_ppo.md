# 03 — Train PPO

Single-agent PPO via the built-in trainer. Reference scripts: [`examples/train_ppo_battery.py`](https://github.com/powerzoojax/PowerZooJax/blob/main/examples/train_ppo_battery.py) and [`examples/train_ppo_transgrid.py`](https://github.com/powerzoojax/PowerZooJax/blob/main/examples/train_ppo_transgrid.py).

## One-liner via `train`

```python
from powerzoojax.rl import train

result = train("battery-soc-tracking", seed=0)
print(result.summary)
```

`train(preset_name, **overrides)` resolves the preset, builds the env, builds the `TrainConfig`, and runs `make_train(env, config)`. See [Training → Presets](../training/presets.md) for the full catalog.

## Explicit env + trainer

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

## What the trainer returns

`TrainResult` exposes:

- `params` — the learned policy parameters (Flax pytree).
- `summary` — short text summary.
- `learning_curve` — per-eval mean returns (when `eval_freq > 0`).

Save these with standard JAX serialization (`flax.serialization.to_bytes` or `pickle`), or with `jnp.save(...)` on the relevant arrays.

## Customizing the reward

Use `RewardWrapper` to inject a custom scalar reward without modifying the env:

```python
from powerzoojax.rl import RewardWrapper

def soc_tracking(state, action, next_state, reward, info):
    return -jnp.abs(next_state.resource_states[0].soc - 0.5)

wrapped = RewardWrapper(LogWrapper(env, params), reward_fn=soc_tracking)
train_fn = make_train(wrapped, config)
```

The original env reward is preserved in `info["env_reward"]`.

## Next step

[04 — Train safe PPO](04_train_safe_ppo.md) shows the CMDP path that uses the selected `costs` vector directly.
