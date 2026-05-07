#!/usr/bin/env python
"""TSO benchmark CLI entry point.

Usage:
    python benchmarks/tso/run.py baseline [--seeds 0,1,2,3,4] [--splits train,iid,load_stress,line_tightening]
    python benchmarks/tso/run.py train --algo ppo [--seed 0] [--config configs/train_ppo.yaml]
    python benchmarks/tso/run.py train --algo sac [--seed 0] [--config configs/train_sac.yaml]
    python benchmarks/tso/run.py train --algo ppo_lagrangian [--seed 0]
    python benchmarks/tso/run.py train --algo ppo_penalty_l10 [--seed 0]
    python benchmarks/tso/run.py train --algo ppo_penalty_l100 [--seed 0]
    python benchmarks/tso/run.py train --algo ppo_penalty_l1000 [--seed 0]
    python benchmarks/tso/run.py eval --run-id <id> --split iid
    python benchmarks/tso/run.py summarize
    python benchmarks/tso/run.py plots [--only normscore|gantt|cost|curves|cross_backend]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

TASK_DIR = Path(__file__).resolve().parent


def cmd_baseline(args):
    from benchmarks.tso.baselines import run_all_baselines

    seeds = [int(s) for s in args.seeds.split(",")]
    splits = [s.strip() for s in args.splits.split(",")]
    run_all_baselines(task_dir=TASK_DIR, seeds=seeds, splits=splits)


def cmd_train(args):
    from benchmarks.tso.train import train_tso

    train_tso(
        task_dir=TASK_DIR,
        algo=args.algo,
        seed=args.seed,
        config_path=args.config,
    )


def cmd_eval(args):
    from benchmarks.tso.eval import eval_tso

    eval_tso(
        task_dir=TASK_DIR,
        run_id=args.run_id,
        split=args.split,
        checkpoint_index=args.checkpoint_index,
    )


def cmd_summarize(args):
    from benchmarks.tso.summarize import summarize_tso

    summarize_tso(
        task_dir=TASK_DIR,
        after=args.after,
        backend=args.backend,
        device=args.device,
    )


def cmd_plots(args):
    from benchmarks.tso.plots import (
        generate_all_plots,
        plot_normscore_bars,
        plot_gantt_commitment,
        plot_cost_decomposition,
        plot_learning_curves,
        plot_cross_backend_learning_walltime,
    )

    dispatch = {
        "normscore": plot_normscore_bars,
        "gantt": plot_gantt_commitment,
        "cost": plot_cost_decomposition,
        "curves": plot_learning_curves,
        "cross_backend": plot_cross_backend_learning_walltime,
    }
    if args.only in ("curves", "cross_backend"):
        # Always refresh both: they share canonical GPU JAX runs; avoids stale PNG pairs.
        plot_learning_curves(TASK_DIR, after=args.after)
        plot_cross_backend_learning_walltime(TASK_DIR, after=args.after)
    elif args.only:
        if args.only in ("normscore", "gantt", "cost"):
            dispatch[args.only](TASK_DIR)
        else:
            dispatch[args.only](TASK_DIR, after=args.after)
    else:
        generate_all_plots(TASK_DIR, after=args.after)


def main():
    parser = argparse.ArgumentParser(
        description="TSO benchmark runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── baseline ──────────────────────────────────────────────────
    p_base = sub.add_parser("baseline", help="Run non-learning baselines (all_on, merit_order)")
    p_base.add_argument("--seeds", default="0,1,2,3,4", help="Comma-separated seeds")
    p_base.add_argument(
        "--splits",
        default="train,iid,load_stress,line_tightening",
        help="Comma-separated splits to evaluate",
    )

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
            "ppo_penalty_l10",
            "ppo_penalty_l100",
            "ppo_penalty_l1000",
        ],
    )
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--config", default=None, help="Override config JSON path")

    # ── eval ──────────────────────────────────────────────────────
    p_eval = sub.add_parser("eval", help="Evaluate a trained run")
    p_eval.add_argument("--run-id", required=True)
    p_eval.add_argument(
        "--split", required=True,
        choices=["train", "iid", "load_stress", "line_tightening"],
    )
    p_eval.add_argument(
        "--checkpoint-index",
        type=int,
        default=None,
        help="Optional PPO-Lagrangian checkpoint index saved during training.",
    )

    # ── summarize ─────────────────────────────────────────────────
    p_summary = sub.add_parser("summarize", help="Aggregate results into summary table")
    p_summary.add_argument("--after", default=None, help="ISO campaign-start timestamp filter")
    p_summary.add_argument("--backend", default="jax_rejax")
    p_summary.add_argument("--device", default="gpu")

    # ── plots ─────────────────────────────────────────────────────
    p_plots = sub.add_parser("plots", help="Generate paper figures")
    p_plots.add_argument(
        "--only",
        choices=["normscore", "gantt", "cost", "curves", "cross_backend"],
        default=None,
        help=(
            "Generate a subset of figures. "
            "Note: curves and cross_backend each run BOTH learning_curves* and "
            "cross_backend_learning_walltime so wall-time dashboards stay in sync."
        ),
    )
    p_plots.add_argument("--after", default=None, help="ISO campaign-start timestamp filter")

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
