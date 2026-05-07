#!/usr/bin/env python
"""DC Microgrid full-pipeline driver (reward-shaping PPO variant).

Resource-aware orchestrator that:
  1. Runs non-learning baselines on CPU (reward shaping applied for same scale).
  2. Polls nvidia-smi until the GPU pool [1, 2] is idle (< 1 GB used each).
  3. Trains PPO with reward shaping (3 seeds). Chunked schedule:
       chunk 1 = GPU 1 seed 0 || GPU 2 seed 1
       chunk 2 = GPU 1 seed 2
  4. Evaluates the trained policies across main + appendix OOD splits
     (8 splits total, 3 seeds each, chunked the same way).
  5. Summarises results and generates paper figures.
  6. Measures JAX throughput on DC Microgrid (single + num_envs sweep).

All subprocess output streams to per-run log files under results/_logs/.

Usage:
    nohup python benchmarks/dc_microgrid/_drive_pipeline.py \
        >> benchmarks/dc_microgrid/results/_logs/driver.log 2>&1 &
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TASK_DIR = Path(__file__).resolve().parent
LOG_DIR = TASK_DIR / "results" / "_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

PYTHON = str(PROJECT_ROOT / ".venv" / "bin" / "python")
SEEDS = [0, 1, 2]
ALGOS = ["ppo"]
MAIN_SPLITS = ["train", "iid", "cooling_stress", "renewable_drought"]
APPENDIX_SPLITS = ["workload_swap", "workload_shock", "dg_derating", "sla_tighten"]
ALL_SPLITS = MAIN_SPLITS + APPENDIX_SPLITS
GPU_FREE_THRESHOLD_MIB = 1024
GPU_POLL_INTERVAL_S = 120

# GPU 0 reserved for TSO; restrict this pipeline to GPU 1 and GPU 2.
GPU_POOL = [1, 2]

# Baselines: False = always re-run (uses shaped reward, so stale records
# must be regenerated). Set True only for manual re-entry.
SKIP_PHASE_0 = True

# If True, keep existing runs/ and manifest.json (do not stash). Use when
# resuming a pipeline after a manual fix so we don't lose completed work.
KEEP_PRIOR_RESULTS = True

# If True, skip training when all SEEDS already have a completed train
# record in the manifest. Combined with KEEP_PRIOR_RESULTS this lets the
# driver resume from mid-pipeline.
SKIP_IF_TRAINED = True

# Per-subprocess env; close XLA preallocation so parallel seeds don't OOM
# against residual allocations from other processes on the same GPU.
SUBPROC_ENV_BASE = {
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.6",
}


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


# ── GPU utilities ────────────────────────────────────────────────────────


def gpu_used_mib() -> list[int]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    return [int(x) for x in out.strip().splitlines()]


def wait_for_pool_free(pool: list[int]) -> None:
    log(f"Waiting for GPUs {pool} to drop below {GPU_FREE_THRESHOLD_MIB} MiB...")
    while True:
        try:
            used = gpu_used_mib()
        except Exception as exc:
            log(f"  nvidia-smi failed ({exc}); retrying in 60s")
            time.sleep(60)
            continue
        pool_used = [used[i] for i in pool]
        if all(u < GPU_FREE_THRESHOLD_MIB for u in pool_used):
            log(f"  Pool GPUs free: pool={pool} used={pool_used} MiB. Proceeding.")
            return
        log(
            f"  pool={pool} used={pool_used} MiB (all={used}); "
            f"sleep {GPU_POLL_INTERVAL_S}s"
        )
        time.sleep(GPU_POLL_INTERVAL_S)


# ── Subprocess helpers ───────────────────────────────────────────────────


def run_serial(cmd: list[str], log_name: str) -> int:
    log_path = LOG_DIR / log_name
    log(f"  [serial] -> {log_path.name}: {' '.join(cmd)}")
    env = {**os.environ, **SUBPROC_ENV_BASE}
    with open(log_path, "ab") as f:
        f.write(f"\n=== {_ts()} START: {' '.join(cmd)} ===\n".encode())
        f.flush()
        rc = subprocess.call(
            cmd, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=str(PROJECT_ROOT)
        )
        f.write(f"\n=== {_ts()} END rc={rc} ===\n".encode())
    log(f"  [serial] done rc={rc}")
    return rc


def _run_chunk(chunk: list[tuple[list[str], str]], pool: list[int]) -> list[int]:
    procs = []
    for (cmd, log_name), gpu_id in zip(chunk, pool):
        log_path = LOG_DIR / log_name
        env = {
            **os.environ,
            **SUBPROC_ENV_BASE,
            "CUDA_VISIBLE_DEVICES": str(gpu_id),
        }
        f = open(log_path, "ab")
        f.write(f"\n=== {_ts()} START gpu={gpu_id}: {' '.join(cmd)} ===\n".encode())
        f.flush()
        p = subprocess.Popen(
            cmd, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=str(PROJECT_ROOT)
        )
        log(f"  [GPU {gpu_id}] pid={p.pid} -> {log_path.name}")
        procs.append((p, f, log_path, gpu_id))
        time.sleep(2)

    rcs = []
    for p, f, log_path, gpu_id in procs:
        rc = p.wait()
        f.write(f"\n=== {_ts()} END gpu={gpu_id} rc={rc} ===\n".encode())
        f.close()
        rcs.append(rc)
        log(f"  [GPU {gpu_id}] done rc={rc} (log: {log_path.name})")
    return rcs


def run_jobs_on_pool(
    jobs: list[tuple[list[str], str]], pool: list[int]
) -> list[int]:
    rcs: list[int] = []
    n = len(pool)
    n_chunks = (len(jobs) + n - 1) // n
    for ci in range(n_chunks):
        chunk = jobs[ci * n : (ci + 1) * n]
        log(
            f"  -- chunk {ci + 1}/{n_chunks}: "
            f"{len(chunk)} jobs on GPUs {pool[: len(chunk)]}"
        )
        rcs.extend(_run_chunk(chunk, pool[: len(chunk)]))
    return rcs


# ── Manifest helpers ─────────────────────────────────────────────────────


def manifest_records() -> list[dict]:
    mp = TASK_DIR / "results" / "manifest.json"
    if not mp.exists():
        return []
    return json.loads(mp.read_text(encoding="utf-8"))


def find_train_run_ids() -> dict[tuple[str, int], str]:
    """Return the most recent *training* run_id per (algo, seed).

    Note: eval.py also writes records with split=='train' when the eval
    split happens to be 'train'. Those are NOT training runs and lack the
    'params' artifact. Filter them out by looking for 'eval of' in notes.
    """
    mapping: dict[tuple[str, int], tuple[str, str]] = {}
    for rec in manifest_records():
        if (
            rec.get("task") == "dc_microgrid"
            and rec.get("split") == "train"
            and rec.get("algo") in ALGOS
            and rec.get("status") == "completed"
            and "eval of " not in (rec.get("notes") or "")
        ):
            key = (rec["algo"], int(rec["seed"]))
            ts = rec.get("timestamp", "")
            if key not in mapping or ts > mapping[key][1]:
                mapping[key] = (rec["run_id"], ts)
    return {k: v[0] for k, v in mapping.items()}


def find_existing_eval_keys() -> set[tuple[str, int, str]]:
    """Return set of (algo, seed, split) tuples already evaluated and saved."""
    keys: set[tuple[str, int, str]] = set()
    for rec in manifest_records():
        if (
            rec.get("task") == "dc_microgrid"
            and rec.get("split") in ALL_SPLITS
            and rec.get("split") != "train"  # train is reserved for training records
            and rec.get("algo") in ALGOS
            and rec.get("status") == "completed"
        ):
            keys.add((rec["algo"], int(rec["seed"]), rec["split"]))
    # Also count split == "train" eval records as an eval — they are saved by
    # eval.py with split=="train" too.
    for rec in manifest_records():
        if (
            rec.get("task") == "dc_microgrid"
            and rec.get("split") == "train"
            and rec.get("algo") in ALGOS
            and rec.get("status") == "completed"
            and "eval of " in (rec.get("notes") or "")
        ):
            keys.add((rec["algo"], int(rec["seed"]), "train"))
    return keys


def phase1_deficit_ok(threshold: float = 0.80) -> bool:
    """Check PPO seed 0 train record has power_deficit_rate < threshold.

    Threshold is 0.80 because the rule_based baseline itself sits at
    deficit_rate ~ 0.89; PPO under reward shaping is expected to improve
    on rule_based (i.e., lower deficit). Any value below 0.80 thus
    validates that PPO has meaningfully learned to shed load / use DG.
    """
    for rec in manifest_records():
        if (
            rec.get("task") == "dc_microgrid"
            and rec.get("algo") == "ppo"
            and rec.get("split") == "train"
            and int(rec.get("seed", -1)) == 0
        ):
            dr = rec.get("metrics", {}).get("power_deficit_rate")
            if dr is None:
                return True  # no metric: cannot judge, allow to proceed
            if float(dr) < threshold:
                return True
            log(
                f"  [sanity] PPO seed=0 power_deficit_rate={dr:.3f} "
                f">= {threshold}; aborting downstream phases"
            )
            return False
    return True


# ── Pre-run cleanup ──────────────────────────────────────────────────────


def stash_stale_results() -> None:
    """Move all prior manifest/runs/summary/figures into _trash so the new
    reward-shaped pipeline starts from a clean slate. Artifacts (.pkl / .npy)
    are also stashed so plots don't pick up stale learning curves."""
    results = TASK_DIR / "results"
    trash = results / "_trash"
    trash.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for subdir in ("summary", "figures", "runs", "artifacts"):
        src = results / subdir
        if src.exists() and any(src.iterdir()):
            dest = trash / f"{subdir}_{stamp}"
            shutil.move(str(src), str(dest))
            log(f"  [clean] stashed {src.name}/ -> {dest.relative_to(results)}")
            src.mkdir(parents=True, exist_ok=True)
    manifest = results / "manifest.json"
    if manifest.exists():
        dest = trash / f"manifest_{stamp}.json"
        shutil.move(str(manifest), str(dest))
        log(f"  [clean] stashed manifest.json -> {dest.relative_to(results)}")
    manifest.write_text("[]", encoding="utf-8")


# ── Phases ───────────────────────────────────────────────────────────────


def phase0_baselines() -> int:
    log("=" * 60)
    log("Phase 0 — Non-learning baselines (CPU only, shaped reward)")
    log("=" * 60)
    cmd = [
        PYTHON, "benchmarks/dc_microgrid/run.py", "baseline",
        "--seeds", ",".join(str(s) for s in SEEDS),
        "--splits", ",".join(ALL_SPLITS),
    ]
    return run_serial(cmd, "phase0_baselines.log")


def phase1_train(algo: str) -> list[int]:
    log("=" * 60)
    log(f"Phase 1 — Training {algo} (shaped) on GPU pool {GPU_POOL}")
    log("=" * 60)
    jobs = []
    for seed in SEEDS:
        cmd = [
            PYTHON, "benchmarks/dc_microgrid/run.py", "train",
            "--algo", algo, "--seed", str(seed),
        ]
        jobs.append((cmd, f"phase1_{algo}_seed{seed}.log"))
    return run_jobs_on_pool(jobs, GPU_POOL)


def phase2_eval() -> list[int]:
    log("=" * 60)
    log(f"Phase 2 — Evaluation on all {len(ALL_SPLITS)} splits, GPU pool {GPU_POOL}")
    log("=" * 60)
    train_ids = find_train_run_ids()
    log(f"  Found {len(train_ids)} train run-ids in manifest:")
    for k, v in sorted(train_ids.items()):
        log(f"    {k}: {v}")

    existing = find_existing_eval_keys() if SKIP_IF_TRAINED else set()
    if existing:
        log(f"  Skipping {len(existing)} (algo, seed, split) eval combos already in manifest.")

    all_rcs: list[int] = []
    for algo in ALGOS:
        for split in ALL_SPLITS:
            jobs = []
            for seed in SEEDS:
                if (algo, seed, split) in existing:
                    continue
                rid = train_ids.get((algo, seed))
                if rid is None:
                    log(f"  [WARN] no run-id for {algo} seed={seed}, skipping")
                    continue
                cmd = [
                    PYTHON, "benchmarks/dc_microgrid/run.py", "eval",
                    "--run-id", rid, "--split", split,
                ]
                jobs.append((cmd, f"phase2_{algo}_seed{seed}_{split}.log"))
            if jobs:
                log(f"-- eval batch: algo={algo} split={split} ({len(jobs)} pending)")
                all_rcs.extend(run_jobs_on_pool(jobs, GPU_POOL))
    return all_rcs


def phase3_summarize_plots() -> int:
    log("=" * 60)
    log("Phase 3 — Summarize + plots (CPU)")
    log("=" * 60)
    rc1 = run_serial(
        [PYTHON, "benchmarks/dc_microgrid/run.py", "summarize"],
        "phase3_summarize.log",
    )
    rc2 = run_serial(
        [PYTHON, "benchmarks/dc_microgrid/run.py", "plots"],
        "phase3_plots.log",
    )
    return rc1 | rc2


def phase4_throughput() -> int:
    """JAX throughput measurements (single num_envs + sweep)."""
    log("=" * 60)
    log("Phase 4 — Throughput benchmarks (dc_microgrid single + sweep)")
    log("=" * 60)
    # Pin to one GPU in the pool so the measurement is not shared.
    gpu = GPU_POOL[0]
    env_overrides = {**SUBPROC_ENV_BASE, "CUDA_VISIBLE_DEVICES": str(gpu)}
    log_path = LOG_DIR / "phase4_throughput.log"

    def _run(cmd: list[str]) -> int:
        with open(log_path, "ab") as f:
            f.write(f"\n=== {_ts()} START (gpu={gpu}): {' '.join(cmd)} ===\n".encode())
            f.flush()
            rc = subprocess.call(
                cmd,
                env={**os.environ, **env_overrides},
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
            )
            f.write(f"\n=== {_ts()} END rc={rc} ===\n".encode())
        return rc

    rc_single = _run(
        [PYTHON, "-m", "benchmarks.common.analysis", "--task", "dc_microgrid"]
    )
    rc_sweep = _run(
        [PYTHON, "-m", "benchmarks.common.analysis",
         "--task", "dc_microgrid", "--sweep"]
    )
    log(f"  throughput: single rc={rc_single}, sweep rc={rc_sweep}")
    # Non-fatal if sweep fails (paper still has single-num_envs number).
    return rc_single


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> int:
    log("DC Microgrid reward-shaping pipeline starting")
    log(f"  PROJECT_ROOT={PROJECT_ROOT}")
    log(f"  PYTHON={PYTHON}")
    log(f"  GPU_POOL={GPU_POOL}")
    log(f"  ALGOS={ALGOS}")
    log(f"  SPLITS={ALL_SPLITS}")
    log(f"  Logs in {LOG_DIR}")

    t_start = time.time()

    if KEEP_PRIOR_RESULTS:
        log("stash_stale_results — SKIPPED (KEEP_PRIOR_RESULTS=True)")
    else:
        stash_stale_results()

    if SKIP_PHASE_0:
        log("Phase 0 — SKIPPED (SKIP_PHASE_0=True)")
    else:
        rc = phase0_baselines()
        if rc != 0:
            log(f"[ABORT] baselines failed rc={rc}")
            return 1

    wait_for_pool_free(GPU_POOL)

    have_trained = find_train_run_ids()
    for algo in ALGOS:
        fully_trained = all((algo, s) in have_trained for s in SEEDS)
        if SKIP_IF_TRAINED and fully_trained:
            log(f"Phase 1 — SKIPPED for {algo}: "
                f"all seeds already have train records in manifest")
            continue
        rcs = phase1_train(algo)
        if any(rc != 0 for rc in rcs):
            log(f"[ABORT] {algo} training failed rcs={rcs}")
            return 2

    if not phase1_deficit_ok():
        log(
            "[ABORT] Phase 1 sanity check failed: PPO seed=0 has "
            "power_deficit_rate above 0.50. Increase "
            "reward_shaping_weights.power_deficit in configs/task.yaml and "
            "re-run the driver."
        )
        return 3

    rcs = phase2_eval()
    if any(rc != 0 for rc in rcs):
        log(f"[WARN] some eval runs returned non-zero: {rcs}")

    rc = phase3_summarize_plots()
    if rc != 0:
        log(f"[WARN] summarize/plots non-zero rc={rc}")

    rc = phase4_throughput()
    if rc != 0:
        log(f"[WARN] throughput non-zero rc={rc} (non-fatal)")

    elapsed = (time.time() - t_start) / 3600
    log(f"DC Microgrid pipeline complete in {elapsed:.2f} h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
