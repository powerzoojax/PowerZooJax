# Examples

下面是简短可运行的示例。它们与 [`examples/`](https://github.com/powerzoojax/PowerZooJax/tree/main/examples) 中的脚本以及 `tests/` 中的模式对齐。

## 示例索引

| 示例 | 演示内容 | 对应脚本 |
| --- | --- | --- |
| [01 — 单步](01_single_step.md) | `TransGridEnv` 上的 reset + 一步 | `examples/jax_02_grid_env.py` |
| [02 — Batched rollout](02_batched_rollout.md) | 多 env 上的 `vmap` + `lax.scan` | `examples/jax_00_verify_device.py` |
| [03 — 训练 PPO](03_train_ppo.md) | `BatteryEnv` 与 `TransGridEnv` 上的单 agent PPO | `examples/train_ppo_battery.py`、`examples/train_ppo_transgrid.py` |
| [04 — 训练 safe PPO](04_train_safe_ppo.md) | `TransGridEnv` 上的 PPO-Lagrangian / CMDP | `examples/train_safe_ppo_transgrid.py` |
| [05 — 训练 MARL](05_train_marl.md) | `GridMARLEnv` 上的 IPPO | `examples/train_ippo_grid.py` |

## 运行前注意

- 多数示例用小的合成 profile，目的是让代码清楚、编译快。真实数据路径见 [Architecture → Data pipeline](../architecture/data-pipeline.md) 与 [Benchmarks](../benchmarks/overview.md) 下的各任务页。
- 每个示例都自包含：复制粘贴到 Python 文件并运行 `python script.py` 即可。CPU 能跑；GPU 在第一次 JIT 之后会更快。
- 大规模可复现 rollout 优先用 [Architecture → JAX 原生并行计算](../architecture/gpu-pipeline.md) 中的 helper、而不是手写 Python 循环。

## 本地预览文档

```bash
./run_doc.sh         # mkdocs serve
./run_doc.sh build   # one-shot build, fails on broken links
```
