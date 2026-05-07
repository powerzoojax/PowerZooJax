#!/usr/bin/env python
"""Run the complete TSO benchmark pipeline end-to-end.

Default target when every training/evaluation cell completes:
90 RunRecords = 40 baseline + 10 training (PPO+PPO-Lag × 5 seeds) + 40 evaluation records.
The manifest may contain fewer logical records if a training run fails or an
evaluation cell is skipped.

Usage:
    python benchmarks/tso/run_all.py [--seeds 0,1,2,3,4] [--algos ppo,ppo_lagrangian] [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.common.configs import load_task_config

TASK_DIR = Path(__file__).resolve().parent
DEFAULT_RL_ALGOS = ["ppo", "ppo_lagrangian"]
SUPPORTED_RL_ALGOS = [
    "ppo",
    "sac",
    "saute_ppo",
    "ppo_lagrangian",
    "ppo_penalty_l10",
    "ppo_penalty_l100",
    "ppo_penalty_l1000",
]


def main():
    parser = argparse.ArgumentParser(description="Run full TSO benchmark pipeline")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument(
        "--algos",
        default=",".join(DEFAULT_RL_ALGOS),
        help=(
            "Comma-separated training algos. Default matches the TSO formal "
            "prompt: ppo,ppo_lagrangian. SAC/penalty variants remain "
            "available for explicit ablations."
        ),
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print steps without executing")
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip non-learning baselines (use when baseline records already exist)",
    )
    parser.add_argument(
        "--campaign-start-iso",
        default=None,
        help=(
            "ISO timestamp for the current campaign. When provided, summary "
            "and manifest-backed plots ignore older records."
        ),
    )
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    algos = [s.strip() for s in args.algos.split(",") if s.strip()]
    unknown_algos = [algo for algo in algos if algo not in SUPPORTED_RL_ALGOS]
    if unknown_algos:
        raise ValueError(
            f"Unsupported TSO algos {unknown_algos!r}; "
            f"supported={SUPPORTED_RL_ALGOS!r}"
        )
    task_cfg = load_task_config(TASK_DIR)
    all_splits = list(task_cfg.get("eval_splits", ["train", "iid", "load_stress", "line_tightening"]))

    print("=" * 60)
    print("TSO Benchmark — Full Pipeline")
    print("=" * 60)

    # Step 1: Baselines
    if args.skip_baselines:
        print("\n[Step 1] Skipping baselines (--skip-baselines).")
    else:
        print("\n[Step 1] Running non-learning baselines...")
        from benchmarks.tso.baselines import run_all_baselines
        if not args.dry_run:
            run_all_baselines(task_dir=TASK_DIR, seeds=seeds, splits=all_splits)
        else:
            print(f"  [DRY] run_all_baselines(seeds={seeds}, splits={all_splits})")

    # Step 2: Train RL agents
    print("\n[Step 2] Training RL agents...")
    from benchmarks.tso.train import train_tso
    train_run_ids: dict[tuple[str, int], str] = {}
    failed_runs: set[tuple[str, int]] = set()
    for algo in algos:
        for seed in seeds:
            if not args.dry_run:
                record = train_tso(task_dir=TASK_DIR, algo=algo, seed=seed)
                train_run_ids[(algo, seed)] = record.run_id
                if record.status != "completed":
                    print(f"  [WARN] training failed for {algo} seed={seed}: status={record.status}")
                    failed_runs.add((algo, seed))
            else:
                print(f"  [DRY] train_tso(algo={algo}, seed={seed})")

    # Step 3: Evaluate trained agents
    print("\n[Step 3] Evaluating trained agents on all splits...")
    from benchmarks.tso.eval import eval_tso
    for algo in algos:
        for seed in seeds:
            run_id = train_run_ids.get((algo, seed))
            if run_id is None:
                if not args.dry_run:
                    print(f"  [WARN] No training record for {algo} seed={seed}, skipping eval")
                continue
            if (algo, seed) in failed_runs:
                if not args.dry_run:
                    print(f"  [WARN] Skipping eval for failed train ({algo}, seed={seed})")
                continue
            for split in all_splits:
                if not args.dry_run:
                    eval_tso(task_dir=TASK_DIR, run_id=run_id, split=split)
                else:
                    print(f"  [DRY] eval_tso(algo={algo}, seed={seed}, split={split})")

    # Step 4: Summarize
    print("\n[Step 4] Summarizing results...")
    if not args.dry_run:
        from benchmarks.tso.summarize import summarize_tso
        summarize_tso(task_dir=TASK_DIR, after=args.campaign_start_iso)
    else:
        print("  [DRY] summarize_tso()")

    # Step 5: Plots
    print("\n[Step 5] Generating figures...")
    if not args.dry_run:
        from benchmarks.tso.plots import generate_all_plots
        generate_all_plots(TASK_DIR, after=args.campaign_start_iso)
    else:
        print("  [DRY] generate_all_plots()")

    print("\n" + "=" * 60)
    print("TSO Benchmark pipeline complete.")
    print(f"Results: {TASK_DIR / 'results'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
