# 04 — Train safe PPO

PPO-Lagrangian training on the same physics env. The agent maximizes expected return subject to selected CMDP costs staying below task budgets. Reference script: [`examples/train_safe_ppo_transgrid.py`](https://github.com/powerzoojax/PowerZooJax/blob/main/examples/train_safe_ppo_transgrid.py).

## One-liner via `train`

```python
from powerzoojax.rl import train

result = train("case5-safe-dispatch", seed=0)
print(result.summary)
```

The preset uses `SafeRLWrapper` with `selected_names=("thermal_overload",)` and `cost_thresholds=(0.0,)`, plus `algo="ppo_lagrangian"`. The dispatcher routes to `make_cmdp_train` automatically.

## Explicit env + trainer

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

## How `SafeRLWrapper` changes the env

```python
obs, state = wrapped.reset(key)
obs, state, reward, costs, done, info = wrapped.step(key, state, action)
#                       ^^^^^                      <- selected CMDP cost vector
```

The 6-tuple is what `make_cmdp_train` expects. The underlying env physics is unchanged: the inner env still computes its full cost vector, and `SafeRLWrapper` only selects the task-relevant subset.

## What the dual multiplier does

PPO-Lagrangian maintains a learned non-negative dual vector `lambda` and uses it to penalize the cost critics:

```
augmented_advantage = A_R - lambda^T A_C
```

Each component is updated in log-space: `log_lambda_i += lambda_lr_i * (episode_cost_est_i - threshold_i)`, then `lambda_i = exp(log_lambda_i)`. `log_lambda_i` is clipped to `[-log_lambda_max, +log_lambda_max]` to prevent runaway. `episode_cost_est_i = mean_cost_i * n_steps` is the per-update estimate of the cumulative episode cost for constraint `i`. In this example only one constraint is selected, so the vector reduces to the familiar scalar case.

## Choosing `cost_threshold`

- Run unconstrained PPO first and measure the resulting `mean cost`.
- Use `cost_thresholds=(0.0, ...)` for hard physical constraints. Non-zero budgets should only be used when the task explicitly defines a relaxed or chance-constrained safety target.
- The benchmark presets freeze `cost_thresholds`. See [Training → Presets](../training/presets.md) for the values used by `tso-scuc-safe`, `dso-nflex-safe`, and `dc-microgrid-safe`.

## Next step

[05 — Train MARL](05_train_marl.md) shows IPPO on a multi-agent grid env.
