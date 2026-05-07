# Examples

The pages below are short, runnable recipes. They mirror the scripts in [`examples/`](https://github.com/powerzoojax/PowerZooJax/tree/main/examples) and the patterns used in `tests/`.

## Recipe map

| Recipe | What it shows | Underlying script |
| --- | --- | --- |
| [01 — Single step](01_single_step.md) | reset + one step on `TransGridEnv` | `examples/jax_02_grid_env.py` |
| [02 — Batched rollout](02_batched_rollout.md) | `vmap` + `lax.scan` over many envs | `examples/jax_00_verify_device.py` |
| [03 — Train PPO](03_train_ppo.md) | single-agent PPO on `BatteryEnv` and `TransGridEnv` | `examples/train_ppo_battery.py`, `examples/train_ppo_transgrid.py` |
| [04 — Train safe PPO](04_train_safe_ppo.md) | PPO-Lagrangian / CMDP on `TransGridEnv` | `examples/train_safe_ppo_transgrid.py` |
| [05 — Train MARL](05_train_marl.md) | IPPO on `GridMARLEnv` | `examples/train_ippo_grid.py` |

## Notes before running

- Most recipes use small synthetic profiles for clarity and compile speed. Real-data paths are documented in [Architecture → Data pipeline](../architecture/data-pipeline.md) and the per-task pages under [Benchmarks](../benchmarks/overview.md).
- Each recipe is self-contained: copy-paste into a Python file and run `python script.py`. CPU works; GPU is faster after the first JIT compile.
- For reproducible large-scale rollouts, prefer the helpers in [Architecture → JAX Parallelization Architecture](../architecture/gpu-pipeline.md) over hand-written Python loops.

## Local docs preview

```bash
./run_doc.sh         # mkdocs serve
./run_doc.sh build   # one-shot build, fails on broken links
```
