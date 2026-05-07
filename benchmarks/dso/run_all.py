#!/usr/bin/env python
"""DSO full pipeline orchestration.

Runs the complete DSO experiment in order:
  1. Baselines  (no_control, tou, droop) × configured eval splits × seeds
  2. Training   (ppo, ppo_lagrangian) × seeds
  3. Evaluation trained runs × configured eval splits
  4. Summarize  → results/summary/latest.json + console table
  5. Plots      → results/figures/*.pdf + *.png

Usage:
    python benchmarks/dso/run_all.py
    python benchmarks/dso/run_all.py --seeds 0 1 2 --algos ppo sac saute_ppo ppo_lagrangian
    python benchmarks/dso/run_all.py --skip-train   # baselines + eval existing runs + summarize + plots
    python benchmarks/dso/run_all.py --only plots   # re-run only the plot step

Stages can be skipped with --skip-{stage} flags.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

TASK_DIR = Path(__file__).resolve().parent
DEFAULT_BASELINE_SPLITS = ["iid"]
DEFAULT_EVAL_SPLITS = ["iid"]


def _configured_eval_splits() -> list[str]:
    from benchmarks.common.configs import load_task_config

    cfg = load_task_config(TASK_DIR)
    splits = [str(s) for s in cfg.get("eval_splits", [])]
    if splits:
        return splits
    return [str(cfg.get("primary_split", "iid"))]


def _validate_splits(label: str, splits: list[str]) -> list[str]:
    allowed = _configured_eval_splits()
    invalid = sorted(set(splits) - set(allowed))
    if invalid:
        raise ValueError(
            f"DSO {label} split(s) {invalid} are not configured. "
            f"Configured eval_splits={allowed}."
        )
    return splits


def stage_baselines(seeds: list[int], splits: list[str]) -> None:
    splits = _validate_splits("baseline", splits)
    print("\n" + "=" * 60)
    print(f"STAGE 1: Baselines  (seeds={seeds}, splits={splits})")
    print("=" * 60)
    from benchmarks.dso.baselines import run_all_baselines
    run_all_baselines(task_dir=TASK_DIR, seeds=seeds, splits=splits)


def stage_train(algos: list[str], seeds: list[int]) -> list[str]:
    """Train each algo × seed; return list of run_ids for the trained runs."""
    print("\n" + "=" * 60)
    print(f"STAGE 2: Training   (algos={algos}, seeds={seeds})")
    print("=" * 60)
    from benchmarks.dso.train import train_dso

    run_ids: list[str] = []
    for algo in algos:
        for seed in seeds:
            try:
                record = train_dso(task_dir=TASK_DIR, algo=algo, seed=seed)
            except Exception as exc:
                print(f"[run_all] train algo={algo} seed={seed} failed: {exc}")
                continue
            run_ids.append(record.run_id)
    return run_ids


def stage_eval(run_ids: list[str], splits: list[str]) -> None:
    """Evaluate each trained run on each eval split."""
    splits = _validate_splits("eval", splits)
    print("\n" + "=" * 60)
    print(f"STAGE 3: Evaluation (run_ids={run_ids}, splits={splits})")
    print("=" * 60)
    from benchmarks.dso.eval import eval_dso

    for run_id in run_ids:
        for split in splits:
            try:
                eval_dso(task_dir=TASK_DIR, run_id=run_id, split=split)
            except Exception as exc:
                print(f"[run_all] eval {run_id} split={split} failed: {exc}")


def stage_summarize(
    *,
    after: str | None = None,
    backend: str = "jax_rejax",
    device: str = "gpu",
) -> None:
    print("\n" + "=" * 60)
    print("STAGE 4: Summarize")
    print("=" * 60)
    from benchmarks.dso.summarize import summarize_dso
    summarize_dso(task_dir=TASK_DIR, after=after, backend=backend, device=device)


def stage_plots(
    *,
    after: str | None = None,
    backend: str = "jax_rejax",
    device: str = "gpu",
) -> None:
    print("\n" + "=" * 60)
    print("STAGE 5: Plots")
    print("=" * 60)
    from benchmarks.dso.plots import generate_all_plots
    generate_all_plots(task_dir=TASK_DIR, after=after, backend=backend, device=device)


def _collect_existing_train_run_ids() -> list[str]:
    """Find run_ids of completed training runs in the manifest."""
    from benchmarks.common.io import load_manifest

    records = load_manifest(TASK_DIR)
    run_ids = [
        r.run_id
        for r in records
        if r.split == "train" and r.algo not in ("no_control", "tou", "droop")
        and r.status == "completed"
        and r.artifacts.get("params")
    ]
    print(f"[run_all] collected {len(run_ids)} completed training run_ids: {run_ids}")
    return run_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full DSO benchmark pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[0, 1, 2],
        help="Seeds to use (default: 0 1 2)",
    )
    parser.add_argument(
        "--algos", nargs="+",
        choices=["ppo", "sac", "saute_ppo", "ppo_lagrangian"],
        default=["ppo", "sac", "saute_ppo", "ppo_lagrangian"],
        help="RL algorithms to train (default: canonical DSO set)",
    )
    parser.add_argument("--after", default=None, help="Campaign start ISO filter for summarize/plots")
    parser.add_argument("--summary-backend", default="jax_rejax")
    parser.add_argument("--summary-device", default="gpu")
    parser.add_argument(
        "--baseline-splits", nargs="+",
        default=DEFAULT_BASELINE_SPLITS,
        help="Splits to run baselines on (default: iid)",
    )
    parser.add_argument(
        "--eval-splits", nargs="+",
        default=DEFAULT_EVAL_SPLITS,
        help="Splits to evaluate trained runs on (default: iid)",
    )
    # Skip flags
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-summarize", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    # Jump to a single stage
    parser.add_argument(
        "--only",
        choices=["baselines", "train", "eval", "summarize", "plots"],
        default=None,
        help="Run only this one stage (ignores --skip-* flags)",
    )

    args = parser.parse_args()

    if args.only == "baselines":
        stage_baselines(args.seeds, args.baseline_splits)
        return
    if args.only == "train":
        stage_train(args.algos, args.seeds)
        return
    if args.only == "eval":
        run_ids = _collect_existing_train_run_ids()
        if not run_ids:
            print("[run_all] No completed training runs found. Run stage 'train' first.")
            sys.exit(1)
        stage_eval(run_ids, args.eval_splits)
        return
    if args.only == "summarize":
        stage_summarize(after=args.after, backend=args.summary_backend, device=args.summary_device)
        return
    if args.only == "plots":
        stage_plots(after=args.after, backend=args.summary_backend, device=args.summary_device)
        return

    # ── Full pipeline ─────────────────────────────────────────────────────
    run_ids: list[str] = []

    if not args.skip_baselines:
        stage_baselines(args.seeds, args.baseline_splits)

    if not args.skip_train:
        run_ids = stage_train(args.algos, args.seeds)
    else:
        run_ids = _collect_existing_train_run_ids()
        print(f"[run_all] skip_train — using {len(run_ids)} existing runs: {run_ids}")

    if not args.skip_eval:
        if run_ids:
            stage_eval(run_ids, args.eval_splits)
        else:
            print("[run_all] No train run_ids available; skipping eval.")

    if not args.skip_summarize:
        stage_summarize(after=args.after, backend=args.summary_backend, device=args.summary_device)

    if not args.skip_plots:
        stage_plots(after=args.after, backend=args.summary_backend, device=args.summary_device)

    print("\n[run_all] Pipeline complete.")


if __name__ == "__main__":
    main()
