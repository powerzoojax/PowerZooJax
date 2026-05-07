# PowerZooJax Benchmarks

Reproducible experiment entry points for the 5 PowerZooJax benchmark tasks.

## Install

For the paper-facing benchmark workflow, install the benchmark extras instead
of relying on whatever happens to already be present in `.venv`:

```bash
uv sync --extra rl --extra benchmarks
# Linux + CUDA 12:
uv sync --extra cuda12 --extra rl --extra benchmarks
```

The `benchmarks` extra is where the cross-backend stack is declared
(`stable-baselines3`, `pettingzoo`, `sbx-rl`).

## Source Of Truth

- Static task definitions live in `benchmarks/<task>/configs/task.yaml`.
- Auto-derived convergence metadata lives in `benchmarks/<task>/configs/provenance.json`.
- Training and eval hyperparameters live in `train_*.yaml` and `eval_*.yaml`.
- Runtime evidence lives in `results/manifest.json` plus per-run JSONs under `results/runs/`.
- Aggregate reporting evidence lives in `results/summary/latest.json`.
- Large binary training-state artifacts are local regeneration outputs, not
  part of the anonymous benchmark release.

Do not infer readiness from older notes or historical manifests alone. The
authoritative **seed-0 readiness check** before multi-seed runs is:

```bash
python -m benchmarks.common.experiment_ops seed0_readiness --task <task> --enforce
```

Before paper-facing reporting, run the executable-truth preflight:

```bash
python -m benchmarks.common.experiment_ops benchmark_preflight --task all
python -m benchmarks.common.experiment_ops benchmark_preflight --task <task> --enforce
```

See `docs/en/glossary.md` (Benchmark workflow glossary).

Current paper-facing snapshot from task configs and `results/summary/latest.json`
as of 2026-04-25:

| Task | Case | Agents | Steps | RL Paradigm | Evidence Status |
|------|------|--------|-------|-------------|-----------------|
| **TSO** | case118 | 1 | 48 x 30min | Safe RL + hybrid action (SCUC) | 5-seed evidence complete; strict safety gate is negative |
| **DSO** | case33bw | 1 | 48 x 30min | Non-stationary RL (loss min) | 5-seed evidence complete; official eval split is IID only |
| **DERs** | case141 | 12 | 48 x 30min | Cooperative safe MARL | 5-seed Phase-1 evidence complete; mandatory seed-0 backend/device matrix complete |
| **GenCos** | case5 | 5 | 48 x 30min | Competitive MARL | 5-seed evidence complete; IPPO beats truthful/uniform-mid but not `max_markup` |
| **DC Microgrid** | self-contained | 1 | 288 x 5min | Multi-objective robust RL | 5-seed evidence, Phase-2 backend audit, and execution-scaling artifacts complete |

For TSO, the current campaign satisfies the seed and hypothesis-test evidence
requirements, but no primary-split row satisfies both zero-violation safety
thresholds. Treat that as a hard benchmark negative result, not as a reason to
relax the safety gate.

For DC Microgrid, the old seed-0-reference gap has been resolved. Its remaining
paper-facing caveats are interpretive: Python-backend strict exact-zero
`feasibility_rate` is numerically brittle, and execution scaling is a
`scaling_only` artifact that must not be used as a final-reward comparison.

## Design Principles

1. **Task as primary axis.** Each task owns its own configs, scripts, and results.
2. **Config-driven experiments.** Scripts read frozen YAML/JSON configs and write structured RunRecords. No hand-written result numbers.
3. **Shared statistics only in `common/`.** Cross-task utilities live in `common/`; task-specific metrics stay with the task.
4. **No synthetic fallback in formal runs.** If real data is required and missing, the run must fail rather than silently substitute fabricated data.
5. **Historical notes must not override code.** If a README and `task.yaml` disagree, trust `task.yaml`, `provenance.json`, and the readiness checker output.
6. **Protocol honesty.** If a task has fewer than the configured submission seeds or lacks a hypothesis test, summaries should say so explicitly rather than implying submission-grade statistical strength.

## Directory Layout

```text
benchmarks/
  README.md
  HARDWARE.md
  common/
    io.py
    stats.py
    runtime.py
    analysis.py
    powerzoo_bridge.py
    experiment_ops.py
  <task>/
    README.md
    configs/
      task.yaml
      provenance.json
      train_<algo>.yaml
      eval_<split>.yaml
    run.py / run_all.py
    train.py
    eval.py
    baselines.py
    summarize.py
    plots.py
    results/
      manifest.json
      runs/
      artifacts/
      summary/
      figures/
```

Not every task exposes the same top-level CLI:

- `dso`, `tso`, `dc_microgrid` have `run.py`
- `ders`, `gencos`, `dc_microgrid` also provide `run_all.py`

## Unified Result Schema

Every training, eval, and baseline script writes a `RunRecord` JSON defined in
`benchmarks/common/io.py`.

```json
{
  "task": "dso",
  "algo": "ppo",
  "seed": 0,
  "run_id": "dso_ppo_train_s0_20260416_143022",
  "status": "completed",
  "split": "iid",
  "metrics": {},
  "walltime_s": 3600.0,
  "throughput_sps": 125000.0,
  "timestamp": "2026-04-16T14:30:22"
}
```

The outer schema is shared; the inner `metrics` keys are task-specific.

## Cross-Backend Comparison: Irreducible Gaps

The cross-backend driver (`benchmarks/common/powerzoo_bridge.py`) lets the
PowerZoo baselines write into the same per-task manifest as
`backend=jax_rejax`. After the 2026-04 fairness fixes, the remaining caveats
are intentional and must be disclosed:

1. **TSO env implementation gap.** Both backends share Case118, the same
   48-step episode structure, and the same GB net-load trace, but they do not
   share the exact solver or observation encoding.
2. **SB3 IL vs PowerZooJax IPPO gap.** DERs and GenCos are cross-backend
   compared with frozen-self-play IL on the PowerZoo side and parameter-shared
   IPPO on the PowerZooJax side. The Python-side GenCos Phase-2 rows use
   frozen opponent policies after the first round; they are not random-opponent
   baselines.
3. **DERs cooperative MARL gap.** DERs is fundamentally cooperative; IL with
   frozen partners is a weaker training class than parameter-shared IPPO.

These are the only intended caveats. Everything else that is reducible
(data window, case, split propagation, reward scale, safety thresholds,
and baseline plumbing) must stay aligned across backends.

## Submission-Grade Reporting

For benchmark-style benchmark reporting, treat the committed task configs as the
machine-readable protocol:

- `task.yaml` defines the split taxonomy, baseline set, primary split, and benchmark metadata.
- `summary/latest.json` should expose protocol compliance, not just aggregate scores.
- A smaller active rerun campaign can be useful as a checkpoint, but a
  submission-grade table should use the configured seed budget and an explicit paired
  hypothesis test on the primary split.

## Quick Start

Check readiness first:

```bash
python -m benchmarks.common.experiment_ops seed0_readiness --task dso
python -m benchmarks.common.experiment_ops seed0_readiness --task dso --enforce
```

Example task entry points:

```bash
# DSO: run all baselines on the official IID split
python benchmarks/dso/run.py baseline --seeds 0,1,2,3,4

# DSO: train PPO
python benchmarks/dso/run.py train --algo ppo --seed 0

# DSO: evaluate a trained run on IID
python benchmarks/dso/run.py eval --run-id dso_ppo_train_s0_... --split iid

# DERs: run only baselines
python benchmarks/ders/run_all.py --only baselines --seeds 0 1 2 3 4

# GenCos: train IPPO
python benchmarks/gencos/run_all.py --only train --algos ippo --seeds 0 1 2 3 4

# Summarize one task
python benchmarks/dso/run.py summarize
```

## Hardware Specification

See [`HARDWARE.md`](HARDWARE.md) for GPU assignment, frozen throughput
budgets, and the current task readiness snapshot.

**Summary — frozen `num_envs`:**

| Task | num_envs | GPU per run |
|------|----------|-------------|
| DSO | 512 | 1× RTX 4500 Ada (24 GB) |
| TSO | 1024 | 1× RTX 4500 Ada (24 GB) |
| DERs | 128 | 1× RTX 4500 Ada (24 GB) |
| DC Microgrid | 256 | 1× RTX 4500 Ada (24 GB) |
| GenCos | 256 | 1× RTX 4500 Ada (24 GB) |

## Git Policy

- `configs/` : **committed** (frozen experiment definitions)
- `results/manifest.json` : **committed** (run index and paper-facing fact source)
- `results/runs/` : **committed** (per-run records referenced by the manifest)
- `results/summary/` : **committed** (lightweight aggregates)
- `results/figures/`, `results/tables/`, `results/scaling/` : **committed** when used for paper-facing reporting
- `results/artifacts/*.json` : **committed** as ordinary Git files when referenced by the manifest or analysis outputs
- Training-state payloads such as `*.npy`, `*.npz`, `*.pkl`, `*.msgpack`, `*.zip`, and `*_params_orbax/` are **not part of the anonymous benchmark release**
- `results/*.log` : **gitignored**

The repository may locally ignore `benchmarks/*/results` during exploratory
runs. Release snapshots should force-add the curated final result tree after
the stale/debug outputs have been archived. Users should retrieve the release
snapshot with:

```bash
git clone <repo>
cd PowerZooJax
```

## Adding a New Task

1. Copy an existing task directory (e.g. `dso/`) as template.
2. Update `configs/task.yaml` with the new task's frozen definition.
3. Implement `train.py`, `eval.py`, `baselines.py` calling the task's env factory.
4. Wire `run.py` or `run_all.py` to dispatch.
5. Add the task to the matrix above.
