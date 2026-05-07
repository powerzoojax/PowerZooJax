#!/usr/bin/env python
"""DSO benchmark CLI entry point.

Usage:
    python benchmarks/dso/run.py baseline [--seeds 0,1,2]
    python benchmarks/dso/run.py train --algo ppo [--seed 0] [--config configs/train_ppo.yaml]
    python benchmarks/dso/run.py eval --run-id <id> --split iid
    python benchmarks/dso/run.py summarize
    python benchmarks/dso/run.py plots [--only normscore|curves|drift|profiles|loss]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

TASK_DIR = Path(__file__).resolve().parent


def cmd_baseline(args):
    from benchmarks.dso.baselines import run_all_baselines

    seeds = [int(s) for s in args.seeds.split(",")]
    run_all_baselines(task_dir=TASK_DIR, seeds=seeds, task_config_path=args.task_config)


def cmd_train(args):
    from benchmarks.dso.train import train_dso

    train_dso(
        task_dir=TASK_DIR,
        algo=args.algo,
        seed=args.seed,
        config_path=args.config,
        task_config_path=args.task_config,
    )


def cmd_eval(args):
    from benchmarks.dso.eval import eval_dso

    eval_dso(
        task_dir=TASK_DIR,
        run_id=args.run_id,
        split=args.split,
        task_config_path=args.task_config,
    )


def cmd_summarize(args):
    from benchmarks.dso.summarize import summarize_dso

    summarize_dso(
        task_dir=TASK_DIR,
        after=args.after,
        backend=args.backend,
        device=args.device,
    )


def cmd_plots(args):
    from benchmarks.dso.plots import (
        generate_all_plots,
        plot_learning_curves,
        plot_learning_curves_walltime,
        plot_drift_gap,
        plot_load_profiles,
        plot_loss_reduction,
    )

    dispatch = {
        "curves": plot_learning_curves,
        "curves_walltime": plot_learning_curves_walltime,
        "drift": plot_drift_gap,
        "profiles": plot_load_profiles,
        "loss": plot_loss_reduction,
    }
    if args.only:
        if args.only in {"curves", "curves_walltime"}:
            dispatch[args.only](
                TASK_DIR,
                after=args.after,
                backend=args.backend,
                device=args.device,
            )
        else:
            dispatch[args.only](TASK_DIR)
    else:
        generate_all_plots(
            TASK_DIR,
            after=args.after,
            backend=args.backend,
            device=args.device,
        )


def main():
    parser = argparse.ArgumentParser(
        description="DSO benchmark runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── baseline ──────────────────────────────────────────────────
    p_base = sub.add_parser("baseline", help="Run non-learning baselines (no_control, tou, droop)")
    p_base.add_argument("--seeds", default="0,1,2", help="Comma-separated seeds")
    p_base.add_argument("--task-config", default=None, help="Override task YAML path")

    # ── train ─────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Train RL agent")
    p_train.add_argument(
        "--algo",
        required=True,
        choices=[
            "ppo",
            "sac",
            "saute_ppo",
            "ppo_lagrangian",
        ],
    )
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--config", default=None, help="Override config JSON path")
    p_train.add_argument("--task-config", default=None, help="Override task YAML path")

    # ── eval ──────────────────────────────────────────────────────
    p_eval = sub.add_parser("eval", help="Evaluate a trained run")
    p_eval.add_argument("--run-id", required=True)
    p_eval.add_argument("--split", required=True, help="Configured eval split; current DSO executable truth is iid")
    p_eval.add_argument("--task-config", default=None, help="Override task YAML path")

    # ── summarize ─────────────────────────────────────────────────
    p_summary = sub.add_parser("summarize", help="Aggregate results into summary table")
    p_summary.add_argument("--after", default=None, help="Campaign start ISO filter")
    p_summary.add_argument("--backend", default="jax_rejax")
    p_summary.add_argument("--device", default="gpu")

    # ── plots ─────────────────────────────────────────────────────
    p_plots = sub.add_parser("plots", help="Generate paper figures")
    p_plots.add_argument(
        "--only",
        choices=["curves", "curves_walltime", "drift", "profiles", "loss"],
        default=None,
        help="Generate only one specific figure",
    )
    p_plots.add_argument("--after", default=None, help="Campaign start ISO filter")
    p_plots.add_argument("--backend", default="jax_rejax")
    p_plots.add_argument("--device", default="gpu")

    args = parser.parse_args()
    {"baseline": cmd_baseline, "train": cmd_train, "eval": cmd_eval,
     "summarize": cmd_summarize, "plots": cmd_plots}[args.command](args)


if __name__ == "__main__":
    main()
