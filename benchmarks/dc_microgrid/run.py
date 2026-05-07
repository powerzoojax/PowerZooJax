#!/usr/bin/env python
"""DC Microgrid benchmark CLI entry point.

Usage:
    python benchmarks/dc_microgrid/run.py baseline [--seeds 0,1,2] [--splits train,iid,cooling_stress,renewable_drought]
    python benchmarks/dc_microgrid/run.py train --algo ppo|sac [--seed 0] [--config configs/train_ppo.yaml]
    python benchmarks/dc_microgrid/run.py eval --run-id <id> --split iid
    python benchmarks/dc_microgrid/run.py summarize
    python benchmarks/dc_microgrid/run.py plots [--only normscore|curves|cost|ood]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

TASK_DIR = Path(__file__).resolve().parent


def _task_dir(args) -> Path:
    return Path(args.task_dir).resolve()


def cmd_baseline(args):
    from benchmarks.dc_microgrid.baselines import run_all_baselines

    seeds = [int(s) for s in args.seeds.split(",")]
    splits = [s.strip() for s in args.splits.split(",")]
    run_all_baselines(task_dir=_task_dir(args), seeds=seeds, splits=splits)


def cmd_train(args):
    from benchmarks.dc_microgrid.train import train_dcmicrogrid

    train_dcmicrogrid(
        task_dir=_task_dir(args),
        algo=args.algo,
        seed=args.seed,
        config_path=args.config,
    )


def cmd_eval(args):
    from benchmarks.dc_microgrid.eval import eval_dcmicrogrid

    eval_dcmicrogrid(
        task_dir=_task_dir(args),
        run_id=args.run_id,
        split=args.split,
    )


def cmd_summarize(args):
    from benchmarks.dc_microgrid.summarize import summarize_dcmicrogrid

    summarize_dcmicrogrid(task_dir=_task_dir(args), after=args.after)


def cmd_plots(args):
    from benchmarks.dc_microgrid.plots import (
        generate_all_plots,
        plot_normscore_bars,
        plot_reward_curves,
        plot_cost_decomposition,
        plot_ood_robustness,
    )

    dispatch = {
        "normscore": plot_normscore_bars,
        "curves": plot_reward_curves,
        "cost": plot_cost_decomposition,
        "ood": plot_ood_robustness,
    }
    if args.only:
        dispatch[args.only](_task_dir(args))
    else:
        generate_all_plots(_task_dir(args))


def main():
    parser = argparse.ArgumentParser(
        description="DC Microgrid benchmark runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── baseline ──────────────────────────────────────────────────────
    p_base = sub.add_parser(
        "baseline", help="Run non-learning baselines (no_control, max_renewable, rule_based)"
    )
    p_base.add_argument("--seeds", default="0,1,2", help="Comma-separated seeds")
    p_base.add_argument(
        "--splits",
        default="train,iid,cooling_stress,renewable_drought",
        help="Comma-separated splits to evaluate",
    )
    p_base.add_argument(
        "--task-dir",
        default=str(TASK_DIR),
        help="Benchmark task directory (defaults to benchmarks/dc_microgrid).",
    )

    # ── train ─────────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Train RL agent")
    p_train.add_argument("--algo", required=True, choices=["ppo", "sac"])
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--config", default=None, help="Override config JSON path")
    p_train.add_argument(
        "--task-dir",
        default=str(TASK_DIR),
        help="Benchmark task directory (defaults to benchmarks/dc_microgrid).",
    )

    # ── eval ──────────────────────────────────────────────────────────
    p_eval = sub.add_parser("eval", help="Evaluate a trained run")
    p_eval.add_argument("--run-id", required=True)
    p_eval.add_argument(
        "--split",
        required=True,
        choices=[
            "train", "iid", "cooling_stress", "renewable_drought",
            "workload_swap", "workload_shock", "dg_derating", "sla_tighten",
        ],
    )
    p_eval.add_argument(
        "--task-dir",
        default=str(TASK_DIR),
        help="Benchmark task directory (defaults to benchmarks/dc_microgrid).",
    )

    # ── summarize ─────────────────────────────────────────────────────
    p_summary = sub.add_parser("summarize", help="Aggregate results into summary table")
    p_summary.add_argument(
        "--task-dir",
        default=str(TASK_DIR),
        help="Benchmark task directory (defaults to benchmarks/dc_microgrid).",
    )
    p_summary.add_argument(
        "--after",
        default=None,
        help="Optional ISO timestamp filter to keep only the current campaign.",
    )

    # ── plots ─────────────────────────────────────────────────────────
    p_plots = sub.add_parser("plots", help="Generate paper figures")
    p_plots.add_argument(
        "--only",
        choices=["normscore", "curves", "cost", "ood"],
        default=None,
        help="Generate only one specific figure",
    )
    p_plots.add_argument(
        "--task-dir",
        default=str(TASK_DIR),
        help="Benchmark task directory (defaults to benchmarks/dc_microgrid).",
    )

    args = parser.parse_args()
    {
        "baseline": cmd_baseline,
        "train": cmd_train,
        "eval": cmd_eval,
        "summarize": cmd_summarize,
        "plots": cmd_plots,
    }[args.command](args)


if __name__ == "__main__":
    main()
