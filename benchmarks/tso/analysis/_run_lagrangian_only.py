#!/usr/bin/env python
"""Re-train + re-eval PPO-Lagrangian only, then summarize + plots.

Used to recover from NaN-divergence after fixing CMDP stability
(see cmdp.py zero_nans / cost_scale / log_lambda_max + train_safe.json).

Steps:
  1. Train ppo_lagrangian for seeds [0, 1, 2]
  2. For each seed whose training produced finite params,
     eval on all 4 splits (train, iid, load_stress, line_tightening)
  3. Run summarize_tso(); regenerate plots

PPO baselines and PPO eval results are left untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

TASK_DIR = Path(__file__).resolve().parent
SPLITS = ["train", "iid", "load_stress", "line_tightening"]
SEEDS = [0, 1, 2]


def main():
    from benchmarks.tso.train import train_tso
    from benchmarks.tso.eval import eval_tso
    from benchmarks.tso.summarize import summarize_tso
    from benchmarks.tso.plots import generate_all_plots

    train_run_ids: dict[int, str] = {}
    failed_seeds: list[int] = []

    print("=" * 60)
    print("[Lagrangian-only] Step 1: Train ppo_lagrangian (3 seeds)")
    print("=" * 60)
    for seed in SEEDS:
        record = train_tso(task_dir=TASK_DIR, algo="ppo_lagrangian", seed=seed)
        train_run_ids[seed] = record.run_id
        if record.status != "completed":
            print(f"  [WARN] seed={seed} train status={record.status}; "
                  f"params will not be evaluated")
            failed_seeds.append(seed)

    print("=" * 60)
    print("[Lagrangian-only] Step 2: Eval on 4 splits")
    print("=" * 60)
    for seed in SEEDS:
        if seed in failed_seeds:
            print(f"  Skipping eval for seed={seed} (training failed)")
            continue
        run_id = train_run_ids[seed]
        for split in SPLITS:
            eval_tso(task_dir=TASK_DIR, run_id=run_id, split=split)

    print("=" * 60)
    print("[Lagrangian-only] Step 3: Summarize + plots")
    print("=" * 60)
    summarize_tso(task_dir=TASK_DIR)
    generate_all_plots(TASK_DIR)

    print("=" * 60)
    print(f"[Lagrangian-only] Done. failed_seeds={failed_seeds}")
    print("=" * 60)


if __name__ == "__main__":
    main()
