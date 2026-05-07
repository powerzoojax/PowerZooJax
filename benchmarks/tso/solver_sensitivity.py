"""TSO formal-eval solver sensitivity without retraining.

This script re-evaluates an existing trained policy under overridden
``dcopf_max_iter`` / ``dcopf_tol`` settings and stores the result outside the
canonical benchmark manifest.

Outputs:

- ``benchmarks/tso/results/solver_sensitivity/<label>.json``
- ``benchmarks/tso/results/solver_sensitivity/<label>_per_episode.json``
- optional comparison figure via ``plot`` mode
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax
import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.common.configs import (
    load_config,
    load_task_config_for_run,
    load_train_config_for_run,
)
from benchmarks.common.io import load_pickle, load_run
from benchmarks.common.runtime import build_train_cfg, make_policy_fn, rollout_bound_wrapper
from benchmarks.common.stats import aggregate_seeds
from benchmarks.tso.config_runtime import get_eval_episodes, get_eval_gb_split, make_task_from_config


TASK_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = TASK_DIR / "results" / "solver_sensitivity"


def _build_saute_wrapper(env, params, train_cfg, selected_names):
    from powerzoojax.rl.wrappers import SauteWrapper

    saute_eval_kwargs = {
        "selected_names": selected_names,
        "horizon": int(train_cfg.saute_horizon or params.max_steps),
        "unsafe_reward": float(train_cfg.saute_unsafe_reward),
        "use_reward_shaping": False,
    }
    if train_cfg.cost_thresholds:
        return SauteWrapper(
            env,
            params,
            cost_thresholds=tuple(float(x) for x in train_cfg.cost_thresholds),
            **saute_eval_kwargs,
        )
    return SauteWrapper(
        env,
        params,
        cost_threshold=float(train_cfg.cost_threshold),
        **saute_eval_kwargs,
    )


def _evaluate_with_overrides(
    *,
    run_id: str,
    split: str,
    dcopf_max_iter: int,
    dcopf_tol: float,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    from powerzoojax.envs.grid.unit_commitment import UnitCommitmentEnv
    from powerzoojax.tasks.tso import rollout_tso

    eval_config = load_config(TASK_DIR / "configs" / f"eval_{split}.yaml")
    original = load_run(run_id, TASK_DIR)
    task_config = load_task_config_for_run(TASK_DIR, original)
    max_steps = int(task_config.get("max_steps", 48))
    n_episodes = get_eval_episodes(task_config, eval_config)
    gb_split = get_eval_gb_split(split, eval_config)

    algo = original.algo
    algo_key = {"ppo_lagrangian": "safe", "saute_ppo": "saute_ppo"}.get(algo, algo)
    train_cfg_dict = load_train_config_for_run(
        TASK_DIR,
        original,
        algo_key_map={"ppo_lagrangian": "safe", "saute_ppo": "saute_ppo"},
        default_key=algo_key,
    )
    train_cfg = build_train_cfg(train_cfg_dict, algo=algo)

    task_with_overrides = {
        **task_config,
        "dcopf_max_iter": int(dcopf_max_iter),
        "dcopf_tol": float(dcopf_tol),
    }
    task = make_task_from_config(
        task_with_overrides,
        load_scale=float(eval_config.get("load_scale", 1.0)),
        line_rating_scale=float(eval_config.get("line_rating_scale", 1.0)),
    )
    env = UnitCommitmentEnv()
    base_params = task.episode_params(split, 0, n_episodes, max_steps)
    constraint_spec = task.constraint_spec()

    rel_p = (original.artifacts or {}).get("params")
    if not rel_p:
        raise FileNotFoundError(f"Run {run_id!r} has no params artifact")
    train_state = load_pickle(TASK_DIR / "results" / rel_p)
    policy_fn = make_policy_fn(
        algo,
        train_state,
        env,
        base_params,
        train_cfg,
        action_dim=2 * base_params.case.n_units,
        selected_names=constraint_spec.selected_names,
    )

    per_episode_metrics: list[dict[str, float]] = []
    episode_diags: list[dict[str, float]] = []
    t0 = time.time()
    for ep in range(n_episodes):
        key = jax.random.PRNGKey(original.seed * 10_000 + ep)
        ref_key = jax.random.PRNGKey(original.seed * 10_000 + ep + 50_000)
        params = task.episode_params(
            split, ep, n_episodes, max_steps, strategy="uniform", seed=original.seed
        )
        if algo == "saute_ppo":
            wrapper = _build_saute_wrapper(
                env, params, train_cfg, constraint_spec.selected_names
            )
            agent_data = rollout_bound_wrapper(
                wrapper,
                key,
                policy_fn,
                max_steps=int(params.max_steps),
                info_keys={
                    "gen_cost": "gen_cost",
                    "startup_cost": "startup_cost",
                    "no_load_cost": "no_load_cost",
                    "reserve_shortfall": "reserve_shortfall",
                    "cost_thermal_overload": "cost_thermal_overload",
                    "cost_sum": "cost_sum",
                    "commitment_switches": "commitment_switches",
                    "is_safe": "is_safe",
                    "opf_converged": "opf_converged",
                    "opf_iterations": "opf_iterations",
                    "opf_box_residual_mw": "opf_box_residual_mw",
                    "opf_line_residual_mw": "opf_line_residual_mw",
                    "opf_balance_residual_mw": "opf_balance_residual_mw",
                },
            )
        else:
            agent_fn = jax.jit(lambda p, k: rollout_tso(env, p, k, policy_fn))
            agent_data = agent_fn(params, key)
        ref_data = task.baseline_rollout(env, params, ref_key, "no_control")
        per_episode_metrics.append(task.compute_metrics(agent_data, ref_data))

        thermal = np.asarray(agent_data["cost_thermal_overload"], dtype=np.float32)
        line_resid = np.asarray(agent_data["opf_line_residual_mw"], dtype=np.float32)
        converged = np.asarray(agent_data["opf_converged"], dtype=np.float32)
        episode_diags.append(
            {
                "episode_idx": float(ep),
                "opf_nonconvergence_rate": float(1.0 - np.mean(converged)),
                "mean_opf_line_residual_mw": float(np.mean(line_resid)),
                "max_opf_line_residual_mw": float(np.max(line_resid)),
                "mean_opf_box_residual_mw": float(
                    np.mean(np.asarray(agent_data["opf_box_residual_mw"], dtype=np.float32))
                ),
                "thermal_step_line_residual_mw": float(
                    np.mean(line_resid[thermal > 1e-6]) if np.any(thermal > 1e-6) else 0.0
                ),
            }
        )
    walltime = time.time() - t0

    flat_metrics = {k: v["mean"] for k, v in aggregate_seeds(per_episode_metrics).items()}
    diag_agg = {
        "opf_nonconvergence_rate": float(np.mean([r["opf_nonconvergence_rate"] for r in episode_diags])),
        "mean_opf_line_residual_mw": float(np.mean([r["mean_opf_line_residual_mw"] for r in episode_diags])),
        "max_opf_line_residual_mw": float(np.max([r["max_opf_line_residual_mw"] for r in episode_diags])),
        "mean_thermal_step_line_residual_mw": float(
            np.mean([r["thermal_step_line_residual_mw"] for r in episode_diags])
        ),
    }
    result = {
        "task": "tso",
        "source_run_id": run_id,
        "algo": algo,
        "seed": int(original.seed),
        "split": split,
        "gb_split": gb_split,
        "dcopf_max_iter": int(dcopf_max_iter),
        "dcopf_tol": float(dcopf_tol),
        "n_eval_episodes": int(n_episodes),
        "walltime_s": float(walltime),
        "metrics": flat_metrics,
        "diagnostics": diag_agg,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return result, per_episode_metrics


def _save_eval(label: str, result: dict[str, Any], per_episode: list[dict[str, float]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / f"{label}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (OUTPUT_DIR / f"{label}_per_episode.json").write_text(
        json.dumps(per_episode, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _plot(labels: list[str], output_name: str) -> Path:
    rows: list[dict[str, Any]] = []
    for label in labels:
        path = OUTPUT_DIR / f"{label}.json"
        rows.append(json.loads(path.read_text(encoding="utf-8")))

    xs = np.arange(len(rows))
    names = [
        f"{row['algo']}\niter={row['dcopf_max_iter']}\ntol={row['dcopf_tol']:.0e}"
        for row in rows
    ]
    costs_m = np.array([row["metrics"]["total_operating_cost"] / 1e6 for row in rows], dtype=float)
    reserve = np.array([row["metrics"]["total_reserve_shortfall"] for row in rows], dtype=float)
    thermal = np.array([row["metrics"]["total_thermal_cost"] for row in rows], dtype=float)
    thermal_rate = np.array([row["metrics"].get("thermal_violation_rate", np.nan) for row in rows], dtype=float)
    walltime = np.array([row["walltime_s"] for row in rows], dtype=float)
    nonconv = np.array([row["diagnostics"]["opf_nonconvergence_rate"] for row in rows], dtype=float)
    max_line_resid = np.array([row["diagnostics"]["max_opf_line_residual_mw"] for row in rows], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.8))

    ax = axes[0, 0]
    bars = ax.bar(xs, costs_m, color="#db3a34", alpha=0.85)
    ax.set_title("Formal IID — operating cost + walltime")
    ax.set_ylabel("Operating cost (million GBP)")
    ax.set_xticks(xs, names)
    for i, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"eval {walltime[i]:.0f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax = axes[0, 1]
    if np.allclose(reserve, 0.0):
        ax.scatter(xs, np.zeros_like(xs), color="#2a9d8f", s=70, zorder=3)
        ax.text(
            0.5,
            0.9,
            "All compared runs have exactly zero reserve shortfall.",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
        )
        ax.set_ylim(-0.05, 0.2)
    else:
        ax.bar(xs, reserve, color="#2a9d8f", alpha=0.85)
    ax.set_title("Formal IID — reserve shortfall")
    ax.set_ylabel("Total reserve shortfall")
    ax.set_xticks(xs, names)

    ax = axes[1, 0]
    bars = ax.bar(xs, thermal, color="#7b2cbf", alpha=0.85)
    ax.set_title("Formal IID — thermal overload")
    ax.set_ylabel("Total thermal cost")
    ax.set_xticks(xs, names)
    for i, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"max resid={max_line_resid[i]:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax = axes[1, 1]
    bars = ax.bar(xs, thermal_rate, color="#3a86ff", alpha=0.85)
    ax.set_title("Formal IID — thermal step rate")
    ax.set_ylabel("Thermal violation rate")
    ax.set_xticks(xs, names)
    for i, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"opf nonconv={nonconv[i]:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.suptitle("TSO solver sensitivity: formal IID outcomes without retraining")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / f"{output_name}.pdf"
    png_path = OUTPUT_DIR / f"{output_name}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    return pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    p_eval = sub.add_parser("eval", help="Run one solver-sensitivity re-eval")
    p_eval.add_argument("--run-id", required=True)
    p_eval.add_argument("--split", default="iid")
    p_eval.add_argument("--dcopf-max-iter", type=int, required=True)
    p_eval.add_argument("--dcopf-tol", type=float, required=True)
    p_eval.add_argument("--label", required=True)

    p_plot = sub.add_parser("plot", help="Plot saved solver-sensitivity rows")
    p_plot.add_argument("--labels", required=True, help="comma-separated labels")
    p_plot.add_argument("--output-name", default="solver_sensitivity_compare")

    args = parser.parse_args()
    if args.mode == "eval":
        result, per_episode = _evaluate_with_overrides(
            run_id=args.run_id,
            split=args.split,
            dcopf_max_iter=args.dcopf_max_iter,
            dcopf_tol=args.dcopf_tol,
        )
        _save_eval(args.label, result, per_episode)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    labels = [s for s in args.labels.split(",") if s]
    out = _plot(labels, args.output_name)
    print(str(out))


if __name__ == "__main__":
    main()
