#!/usr/bin/env python
"""Run the complete DC Microgrid benchmark pipeline end-to-end.

Produces all RunRecords (baselines + training + eval), then summarizes
results and generates paper figures.

Expected record count (reward-shaped PPO+SAC pipeline, 8 splits, default seeds)
--------------------------------------------------------------------------------
  baselines: 3 algos x 8 splits x 3 seeds = 72
  RL training: 2 algos x 3 seeds = 6
  RL eval: 2 algos x 8 splits x 3 seeds = 48
  Total: 126 (excluding ablations)

Usage:
    python benchmarks/dc_microgrid/run_all.py [--seeds 0,1,2] [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

TASK_DIR = Path(__file__).resolve().parent
ALL_SPLITS = [
    "train", "iid", "cooling_stress", "renewable_drought",
    "workload_swap", "workload_shock", "dg_derating", "sla_tighten",
]
RL_ALGOS = ["ppo", "sac"]


def main():
    parser = argparse.ArgumentParser(
        description="Run full DC Microgrid benchmark pipeline"
    )
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print steps without executing",
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip non-learning baselines (use when baseline records already exist)",
    )
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    print("=" * 60)
    print("DC Microgrid Benchmark — Full Pipeline")
    print("=" * 60)

    # Step 1: Baselines
    if args.skip_baselines:
        print("\n[Step 1] Skipping baselines (--skip-baselines).")
    else:
        print("\n[Step 1] Running non-learning baselines...")
        from benchmarks.dc_microgrid.baselines import run_all_baselines
        if not args.dry_run:
            run_all_baselines(task_dir=TASK_DIR, seeds=seeds, splits=ALL_SPLITS)
        else:
            print(f"  [DRY] run_all_baselines(seeds={seeds}, splits={ALL_SPLITS})")

    # Step 2: Train RL agents
    print("\n[Step 2] Training RL agents...")
    from benchmarks.dc_microgrid.train import train_dcmicrogrid
    train_run_ids: dict[tuple[str, int], str] = {}
    for algo in RL_ALGOS:
        for seed in seeds:
            if not args.dry_run:
                record = train_dcmicrogrid(task_dir=TASK_DIR, algo=algo, seed=seed)
                train_run_ids[(algo, seed)] = record.run_id
            else:
                print(f"  [DRY] train_dcmicrogrid(algo={algo}, seed={seed})")

    # Step 3: Evaluate trained agents
    print("\n[Step 3] Evaluating trained agents on all splits...")
    from benchmarks.dc_microgrid.eval import eval_dcmicrogrid
    for algo in RL_ALGOS:
        for seed in seeds:
            run_id = train_run_ids.get((algo, seed))
            if run_id is None:
                if not args.dry_run:
                    print(
                        f"  [WARN] No training record for {algo} seed={seed}, "
                        "skipping eval"
                    )
                    continue
            for split in ALL_SPLITS:
                if not args.dry_run:
                    eval_dcmicrogrid(task_dir=TASK_DIR, run_id=run_id, split=split)
                else:
                    print(
                        f"  [DRY] eval_dcmicrogrid(algo={algo}, seed={seed}, "
                        f"split={split})"
                    )

    # Step 4: Summarize
    print("\n[Step 4] Summarizing results...")
    if not args.dry_run:
        from benchmarks.dc_microgrid.summarize import summarize_dcmicrogrid
        summarize_dcmicrogrid(task_dir=TASK_DIR)
    else:
        print("  [DRY] summarize_dcmicrogrid()")

    # Step 5: Plots
    print("\n[Step 5] Generating figures...")
    if not args.dry_run:
        from benchmarks.dc_microgrid.plots import generate_all_plots
        generate_all_plots(TASK_DIR)
    else:
        print("  [DRY] generate_all_plots()")

    print("\n" + "=" * 60)
    print("DC Microgrid Benchmark pipeline complete.")
    print(f"Results: {TASK_DIR / 'results'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
