#!/usr/bin/env python
"""GenCos full pipeline orchestration.

Runs the complete GenCos experiment in order:
  1. Baselines  (truthful, uniform_mid, max_markup) × 4 splits × seeds
  2. Training   (ippo) × seeds — serial on GPU 2
  3. Evaluation trained runs × 4 eval splits
  4. Summarize  → results/summary/latest.json + console table
  5. Plots      → results/figures/*.pdf + *.png

Usage
-----
  CUDA_VISIBLE_DEVICES=2 python benchmarks/gencos/run_all.py
  CUDA_VISIBLE_DEVICES=2 python benchmarks/gencos/run_all.py --seeds 0 1 2
  CUDA_VISIBLE_DEVICES=2 python benchmarks/gencos/run_all.py --only baselines
  CUDA_VISIBLE_DEVICES=2 python benchmarks/gencos/run_all.py --only train
  CUDA_VISIBLE_DEVICES=2 python benchmarks/gencos/run_all.py --only eval
  CUDA_VISIBLE_DEVICES=2 python benchmarks/gencos/run_all.py --only summarize
  CUDA_VISIBLE_DEVICES=2 python benchmarks/gencos/run_all.py --only plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.common.configs import load_task_config

DEFAULT_TASK_DIR = Path(__file__).resolve().parent
ALL_SPLITS = ["train", "iid", "demand_shift", "renewable_shock"]
EVAL_SPLITS = ["train", "iid", "demand_shift", "renewable_shock"]
TRAIN_ALGOS = ["ippo"]


def stage_baselines(task_dir: Path, seeds: list[int], splits: list[str]) -> None:
    print("\n" + "=" * 60)
    print(f"STAGE 1: Baselines  (seeds={seeds}, splits={splits})")
    print("=" * 60)
    from benchmarks.gencos.baselines import run_all_baselines
    run_all_baselines(task_dir=task_dir, seeds=seeds, splits=splits)


def stage_train(task_dir: Path, algos: list[str], seeds: list[int]) -> list[str]:
    print("\n" + "=" * 60)
    print(f"STAGE 2: Training   (algos={algos}, seeds={seeds})")
    print("=" * 60)
    from benchmarks.gencos.train import train_gencos

    run_ids: list[str] = []
    for algo in algos:
        for seed in seeds:
            try:
                record = train_gencos(task_dir=task_dir, algo=algo, seed=seed)
                run_ids.append(record.run_id)
            except Exception as exc:
                print(f"[run_all] train algo={algo} seed={seed} failed: {exc}")
    return run_ids


def stage_eval(task_dir: Path, run_ids: list[str], splits: list[str]) -> None:
    print("\n" + "=" * 60)
    print(f"STAGE 3: Evaluation (run_ids={run_ids}, splits={splits})")
    print("=" * 60)
    from benchmarks.gencos.eval import eval_gencos

    for run_id in run_ids:
        for split in splits:
            try:
                eval_gencos(task_dir=task_dir, run_id=run_id, split=split)
            except Exception as exc:
                print(f"[run_all] eval {run_id} split={split} failed: {exc}")


def stage_summarize(task_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("STAGE 4: Summarize")
    print("=" * 60)
    from benchmarks.gencos.summarize import summarize_gencos
    summarize_gencos(task_dir=task_dir)


def stage_plots(task_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("STAGE 5: Plots")
    print("=" * 60)
    from benchmarks.gencos.plots import generate_all_plots
    generate_all_plots(task_dir=task_dir)


def _collect_existing_train_run_ids(task_dir: Path) -> list[str]:
    from benchmarks.common.io import load_manifest
    records = load_manifest(task_dir)
    run_ids = [
        r.run_id for r in records
        if r.split == "train"
        and r.algo not in ("truthful", "uniform_mid", "max_markup")
        and r.status == "completed"
        and (r.artifacts or {}).get("params")
    ]
    print(f"[run_all] collected {len(run_ids)} completed training run_ids: {run_ids}")
    return run_ids


def main() -> None:
    default_seeds = load_task_config(DEFAULT_TASK_DIR).get("seeds", [0, 1, 2, 3, 4])
    parser = argparse.ArgumentParser(
        description="Run the full GenCos benchmark pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=default_seeds)
    parser.add_argument("--algos", nargs="+", choices=TRAIN_ALGOS, default=TRAIN_ALGOS)
    parser.add_argument("--baseline-splits", nargs="+", default=ALL_SPLITS, choices=ALL_SPLITS)
    parser.add_argument("--eval-splits", nargs="+", default=EVAL_SPLITS, choices=ALL_SPLITS)
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-summarize", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--task-dir",
        type=Path,
        default=DEFAULT_TASK_DIR,
        help="Task directory containing configs/ and results/ (defaults to benchmarks/gencos).",
    )
    parser.add_argument(
        "--only",
        choices=["baselines", "train", "eval", "summarize", "plots"],
        default=None,
    )
    args = parser.parse_args()
    task_dir = args.task_dir

    if args.only == "baselines":
        stage_baselines(task_dir, args.seeds, args.baseline_splits)
        return
    if args.only == "train":
        stage_train(task_dir, args.algos, args.seeds)
        return
    if args.only == "eval":
        run_ids = _collect_existing_train_run_ids(task_dir)
        if not run_ids:
            print("[run_all] No completed training runs found.")
            sys.exit(1)
        stage_eval(task_dir, run_ids, args.eval_splits)
        return
    if args.only == "summarize":
        stage_summarize(task_dir)
        return
    if args.only == "plots":
        stage_plots(task_dir)
        return

    # ── Full pipeline ─────────────────────────────────────────────────────
    run_ids: list[str] = []
    if not args.skip_baselines:
        stage_baselines(task_dir, args.seeds, args.baseline_splits)
    if not args.skip_train:
        run_ids = stage_train(task_dir, args.algos, args.seeds)
    else:
        run_ids = _collect_existing_train_run_ids(task_dir)
    if not args.skip_eval and run_ids:
        stage_eval(task_dir, run_ids, args.eval_splits)
    if not args.skip_summarize:
        stage_summarize(task_dir)
    if not args.skip_plots:
        stage_plots(task_dir)
    print("\n[run_all] Pipeline complete.")


if __name__ == "__main__":
    main()
