#!/usr/bin/env python
"""DERs full pipeline orchestration.

Runs the complete DERs experiment in order:
  1. Baselines  (no_control, volt_droop) × 5 splits × seeds
  2. Training   (ippo, ippo_safe, ippo_lagrangian) × seeds — serial on GPU 2
  3. Evaluation trained runs × 5 eval splits
  4. Summarize  → results/summary/latest.json + console table
  5. Plots      → results/figures/*.pdf + *.png

Usage
-----
  CUDA_VISIBLE_DEVICES=2 python benchmarks/ders/run_all.py
  CUDA_VISIBLE_DEVICES=2 python benchmarks/ders/run_all.py --seeds 0 1 2 3 4 --algos ippo ippo_safe ippo_lagrangian
  CUDA_VISIBLE_DEVICES=2 python benchmarks/ders/run_all.py --only baselines
  CUDA_VISIBLE_DEVICES=2 python benchmarks/ders/run_all.py --only train --algos ippo
  CUDA_VISIBLE_DEVICES=2 python benchmarks/ders/run_all.py --only eval --eval-splits iid voltage_tightening
  CUDA_VISIBLE_DEVICES=2 python benchmarks/ders/run_all.py --only eval --run-ids <train_run_id> --eval-splits iid
  CUDA_VISIBLE_DEVICES=2 python benchmarks/ders/run_all.py --only summarize
  CUDA_VISIBLE_DEVICES=2 python benchmarks/ders/run_all.py --only plots

All stages can be skipped individually with --skip-{stage}.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.common.runtime import prefer_packaged_cuda_binaries

prefer_packaged_cuda_binaries()

DEFAULT_TASK_DIR = Path(__file__).resolve().parent
ALL_SPLITS = ["train", "iid", "voltage_tightening", "pv_penetration_shift", "load_stress"]
EVAL_SPLITS = ["train", "iid", "voltage_tightening", "pv_penetration_shift", "load_stress"]
TRAIN_ALGOS = ["ippo", "ippo_safe", "ippo_lagrangian"]


# ── Stage helpers ─────────────────────────────────────────────────────────────


def stage_baselines(task_dir: Path, seeds: list[int], splits: list[str]) -> None:
    print("\n" + "=" * 60)
    print(f"STAGE 1: Baselines  (seeds={seeds}, splits={splits})")
    print("=" * 60)
    from benchmarks.ders.baselines import run_all_baselines
    run_all_baselines(task_dir=task_dir, seeds=seeds, splits=splits)


def stage_train(task_dir: Path, algos: list[str], seeds: list[int]) -> list[str]:
    """Train each algo × seed serially; return list of run_ids."""
    print("\n" + "=" * 60)
    print(f"STAGE 2: Training   (algos={algos}, seeds={seeds})")
    print("=" * 60)
    from benchmarks.ders.train import train_ders

    run_ids: list[str] = []
    for algo in algos:
        for seed in seeds:
            try:
                record = train_ders(task_dir=task_dir, algo=algo, seed=seed)
                run_ids.append(record.run_id)
            except Exception as exc:
                print(f"[run_all] train algo={algo} seed={seed} failed: {exc}")
    return run_ids


def stage_eval(task_dir: Path, run_ids: list[str], splits: list[str]) -> None:
    """Evaluate each trained run on each eval split."""
    print("\n" + "=" * 60)
    print(f"STAGE 3: Evaluation (run_ids={run_ids}, splits={splits})")
    print("=" * 60)
    from benchmarks.ders.eval import eval_ders

    for run_id in run_ids:
        for split in splits:
            try:
                eval_ders(task_dir=task_dir, run_id=run_id, split=split)
            except Exception as exc:
                print(f"[run_all] eval {run_id} split={split} failed: {exc}")


def stage_summarize(task_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("STAGE 4: Summarize")
    print("=" * 60)
    from benchmarks.ders.summarize import summarize_ders
    summarize_ders(task_dir=task_dir)


def stage_plots(task_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("STAGE 5: Plots")
    print("=" * 60)
    from benchmarks.ders.plots import generate_all_plots
    generate_all_plots(task_dir=task_dir)


def _collect_existing_train_run_ids(task_dir: Path) -> list[str]:
    """Find run_ids of completed training runs in the manifest."""
    from benchmarks.common.io import has_training_artifact, load_manifest

    records = load_manifest(task_dir)
    run_ids = [
        r.run_id
        for r in records
        if r.split == "train"
        and r.algo not in ("no_control", "volt_droop")
        and r.status == "completed"
        and has_training_artifact(r.artifacts)
    ]
    print(f"[run_all] collected {len(run_ids)} completed training run_ids: {run_ids}")
    return run_ids


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full DERs benchmark pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
        help="Seeds to use (default: 0 1 2 3 4)",
    )
    parser.add_argument(
        "--algos", nargs="+",
        choices=TRAIN_ALGOS,
        default=TRAIN_ALGOS,
        help="RL algorithms to train (default: ippo ippo_safe ippo_lagrangian)",
    )
    parser.add_argument(
        "--baseline-splits", nargs="+",
        default=ALL_SPLITS,
        choices=ALL_SPLITS,
        help="Splits to run baselines on (default: all 5)",
    )
    parser.add_argument(
        "--eval-splits", nargs="+",
        default=EVAL_SPLITS,
        choices=ALL_SPLITS,
        help="Splits to evaluate trained runs on (default: all 5)",
    )
    parser.add_argument(
        "--run-ids", nargs="+", default=None,
        help=(
            "Training run ids to evaluate when --only eval is used. "
            "Default: collect all completed training runs from the manifest."
        ),
    )
    # Skip flags
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-summarize", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--task-dir",
        type=Path,
        default=DEFAULT_TASK_DIR,
        help="Task directory containing configs/ and results/ (defaults to benchmarks/ders).",
    )
    # Jump to a single stage
    parser.add_argument(
        "--only",
        choices=["baselines", "train", "eval", "summarize", "plots"],
        default=None,
        help="Run only this one stage (ignores --skip-* flags)",
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
        run_ids = list(args.run_ids) if args.run_ids else _collect_existing_train_run_ids(task_dir)
        if not run_ids:
            print("[run_all] No completed training runs found. Run stage 'train' first.")
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
        print(f"[run_all] skip_train — using {len(run_ids)} existing runs: {run_ids}")

    if not args.skip_eval:
        if run_ids:
            stage_eval(task_dir, run_ids, args.eval_splits)
        else:
            print("[run_all] No train run_ids available; skipping eval.")

    if not args.skip_summarize:
        stage_summarize(task_dir)

    if not args.skip_plots:
        stage_plots(task_dir)

    print("\n[run_all] Pipeline complete.")


if __name__ == "__main__":
    main()
