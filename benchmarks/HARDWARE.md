# Hardware Specification for Benchmark Experiments

## Current Status

Verification snapshot: **2026-04-25**

- `python -m benchmarks.common.experiment_ops benchmark_preflight --task all --enforce` passes.
- DSO, DERs, GenCos, and DC Microgrid have complete 5-seed paper-facing evidence in their current summaries.
- TSO has complete 5-seed evidence, but its strict zero-violation safety gate is negative.
- DC Microgrid execution scaling has formal matched-range and JAX-extended artifacts; these are `scaling_only` records and must not be used as final-reward comparisons.
- Formal comparison runs must not use synthetic fallback when real data is required

`RunRecord.env_info` automatically captures `nvidia_gpu_name` per record for
paper hardware-table reproducibility.

## Machine

| Component | Spec |
|-----------|------|
| CPU | AMD Ryzen Threadripper PRO 7985WX, 64 cores / 128 threads |
| RAM | 251 GB |
| GPU | 3× NVIDIA RTX 4500 Ada Generation, 24 GB VRAM each |
| CUDA | 13.0, driver 580.82.07 |
| JAX backend | GPU (CUDA) |

## Parallel vs Serial Execution Policy

### Core Rule

| Dimension | Policy | Reason |
|-----------|--------|--------|
| Seeds within a task | **Parallel in GPU-sized waves** (up to 3 simultaneous seeds on this machine) | Each seed is independent |
| Algorithms within a task | **Serial** | Each algo batch wants the full 3-GPU pool |
| Tasks | **Serial** | Avoids GPU sharing and throughput distortion |
| Baselines | **Upfront, serial per task** | Light jobs, cheap to clear first |
| Eval after training | **Parallel across seeds** | Same isolation rule as training |
| Summarize + plots | **Serial, CPU-only** | Fast and non-blocking |

### Recommended Phase Structure

```bash
Phase 0 — Baselines
  python benchmarks/tso/run.py baseline --seeds 0,1,2,3,4
  python benchmarks/dso/run.py baseline --seeds 0,1,2,3,4
  python benchmarks/ders/run_all.py --only baselines --seeds 0 1 2 3 4
  python benchmarks/gencos/run_all.py --only baselines --seeds 0 1 2 3 4

Phase 1 — RL training
  CUDA_VISIBLE_DEVICES=0 python benchmarks/tso/run.py train --algo ppo --seed 0 &
  CUDA_VISIBLE_DEVICES=1 python benchmarks/tso/run.py train --algo ppo --seed 1 &
  CUDA_VISIBLE_DEVICES=2 python benchmarks/tso/run.py train --algo ppo --seed 2 &
  wait
  CUDA_VISIBLE_DEVICES=0 python benchmarks/tso/run.py train --algo ppo --seed 3 &
  CUDA_VISIBLE_DEVICES=1 python benchmarks/tso/run.py train --algo ppo --seed 4 &
  wait
  [repeat by task / algorithm as needed]

Phase 2 — Evaluation
  for seed in 0 1 2; do
      CUDA_VISIBLE_DEVICES=$seed python benchmarks/<task>/run.py eval \
          --run-id <run_id_seed${seed}> --split iid &
  done
  wait
  [repeat for seeds 3 and 4]

Phase 3 — Summarize + plots
  python benchmarks/<task>/run.py summarize
  python benchmarks/<task>/run.py plots
```

### Why Not Cross-Task Parallelism?

With 3 GPUs and the one-GPU-per-run rule, running two tasks simultaneously
would either share a GPU between jobs or reduce the seed count per task. Both
hurt fairness and make throughput figures incomparable.

### GPU Assignment Policy

One GPU per training or eval run. Always set `CUDA_VISIBLE_DEVICES`
explicitly. Never share a GPU between two simultaneous training jobs.

## Frozen `num_envs` per Task

**Source of truth: `benchmarks/<task>/configs/task.yaml::num_envs`.** The
table below mirrors those values. `tests/benchmarks/test_config_consistency.py`
asserts agreement with `benchmarks/common/analysis.py::FROZEN_NUM_ENVS`, and
separately checks that each `train_*.yaml::num_envs` does not exceed the
frozen capacity.

| Task | num_envs | Reason |
|------|----------|--------|
| DSO | 512 | Small single-agent state, throughput-oriented |
| TSO | 1024 | Wide batch needed for GPU throughput |
| DERs | 128 | MARL overhead but still GPU-scalable |
| DC Microgrid | 256 | Long horizon, larger buffers |
| GenCos | 256 | Market env still fits under budget |

## VRAM Budget

Per-GPU working budget: **≤ 20 GB**. Leave roughly 4 GB headroom.

## CPU / XLA

Do not cap XLA's CPU thread count. Leave `XLA_FLAGS` and
`TF_NUM_INTEROP_THREADS` at default unless a task-specific issue forces a
documented override.

## Reporting Requirement

Every `RunRecord` must include `throughput_sps` when applicable. The paper's
cross-task throughput table uses these recorded values.

## Checklist Before Starting Any Experiment

1. Confirm the target task currently passes the seed-0 readiness check (`seed0_readiness --enforce` exits 0).
2. Confirm the target GPU is free with `nvidia-smi`.
3. Set `CUDA_VISIBLE_DEVICES` explicitly.
4. Confirm the task's `train_*.yaml::num_envs` does not exceed the frozen capacity above.
5. Do not run two training jobs on the same GPU simultaneously.
6. For `dc_microgrid`, run a real-data smoke check first; if real traces are unavailable, stop rather than falling back to synthetic.
