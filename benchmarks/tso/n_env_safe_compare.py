#!/usr/bin/env python3
"""Compare n=128 vs n=256 safe-RL TSO runs on walltime and formal IID safety.

Usage::

    python benchmarks/tso/n_env_safe_compare.py

Outputs::

    benchmarks/tso/results/figures/n_env_safe_compare.pdf
    benchmarks/tso/results/figures/n_env_safe_compare.png
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parent
FIG_DIR = TASK_DIR / "results" / "figures"


_RUN_SPECS = [
    {
        "family": "Sauté PPO",
        "n_envs": 256,
        "algo": "saute_ppo",
        "note_substr": (
            "TSO Sauté PPO n256 4M with n_steps=24, mb=8, thermal budget=10, "
            "reserve budget=100, unsafe_reward=-30."
        ),
        "color": "#8e24aa",
        "linestyle": "-",
    },
    {
        "family": "Sauté PPO",
        "n_envs": 128,
        "algo": "saute_ppo",
        "note_substr": (
            "TSO Sauté PPO n128 4M with n_steps=24, mb=8, thermal budget=10, "
            "reserve budget=100, unsafe_reward=-30."
        ),
        "color": "#8e24aa",
        "linestyle": "--",
    },
    {
        "family": "PPO-Lag",
        "n_envs": 256,
        "algo": "ppo_lagrangian",
        "note_substr": "TSO PPO-Lagrangian n256 4M with n_steps=24 and n_minibatches=8.",
        "color": "#d81b60",
        "linestyle": "-",
    },
    {
        "family": "PPO-Lag",
        "n_envs": 128,
        "algo": "ppo_lagrangian",
        "note_substr": "TSO PPO-Lagrangian n128 4M with n_steps=24 and n_minibatches=8.",
        "color": "#d81b60",
        "linestyle": "--",
    },
]


def _latest_train_run(task_dir: Path, note_substr: str, algo: str) -> dict | None:
    manifest = task_dir / "results" / "manifest.json"
    if not manifest.exists():
        return None
    data = json.loads(manifest.read_text(encoding="utf-8"))
    best: dict | None = None
    best_ts = ""
    for row in data:
        if row.get("task") != "tso" or row.get("split") != "train":
            continue
        if row.get("algo") != algo:
            continue
        run_id = str(row.get("run_id") or "")
        if not run_id:
            continue

        notes = str(row.get("notes") or "")
        if note_substr not in notes:
            cfg_path = task_dir / "results" / "artifacts" / f"{run_id}_config.json"
            cfg_notes = ""
            if cfg_path.exists():
                try:
                    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                    cfg_notes = str((cfg.get("train_config_raw") or {}).get("notes") or "")
                except Exception:
                    cfg_notes = ""
            if note_substr not in cfg_notes:
                continue

        ts = str(row.get("timestamp") or "")
        if best is None or ts > best_ts:
            best = row
            best_ts = ts
    return best


def _latest_iid_eval_for_train(task_dir: Path, train_run_id: str) -> dict | None:
    runs_dir = task_dir / "results" / "runs"
    best: dict | None = None
    best_ts = ""
    for path in runs_dir.glob("tso_*_iid_*.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        notes = str(row.get("notes") or "")
        if f"eval of {train_run_id}" not in notes:
            continue
        ts = str(row.get("timestamp") or "")
        if best is None or ts > best_ts:
            best = row
            best_ts = ts
    return best


def _load_curve_array(artifacts_dir: Path, run_id: str, stem: str) -> np.ndarray | None:
    path = artifacts_dir / f"{run_id}_{stem}.npy"
    if not path.exists():
        return None
    return np.asarray(np.load(path), dtype=float).ravel()


def _curve_x(train_row: dict, artifacts_dir: Path, run_id: str, n_points: int) -> np.ndarray:
    wall = _load_curve_array(artifacts_dir, run_id, "learning_curve_eval_walltimes")
    if wall is not None and wall.size == n_points:
        return wall
    total = float(train_row.get("walltime_s") or 0.0)
    if total > 0.0:
        return np.linspace(0.0, total, n_points)
    return np.arange(n_points, dtype=float)


def _collect_runs(task_dir: Path) -> list[dict]:
    artifacts_dir = task_dir / "results" / "artifacts"
    rows: list[dict] = []
    for spec in _RUN_SPECS:
        train_row = _latest_train_run(task_dir, spec["note_substr"], spec["algo"])
        if train_row is None:
            print(f"[n_env_safe_compare] missing train run for {spec['note_substr']!r}")
            continue
        train_run_id = str(train_row["run_id"])
        eval_row = _latest_iid_eval_for_train(task_dir, train_run_id)
        if eval_row is None:
            print(f"[n_env_safe_compare] missing iid eval for train run {train_run_id}")
            continue
        cost_curve = _load_curve_array(artifacts_dir, train_run_id, "eval_total_operating_cost")
        if cost_curve is None or cost_curve.size == 0:
            print(f"[n_env_safe_compare] missing eval cost curve for {train_run_id}")
            continue
        rows.append(
            {
                **spec,
                "train": train_row,
                "eval": eval_row,
                "train_run_id": train_run_id,
                "eval_run_id": str(eval_row["run_id"]),
                "cost_curve": cost_curve,
                "wall_curve": _curve_x(train_row, artifacts_dir, train_run_id, int(cost_curve.size)),
            }
        )
    return rows


def _metric_with_legacy_backfill(metrics: dict, key: str) -> float:
    value = metrics.get(key)
    if value is not None:
        return float(value)

    feasibility = metrics.get("feasibility_rate")
    reserve_total = metrics.get("total_reserve_shortfall")
    thermal_total = metrics.get("total_thermal_cost")

    if key == "reserve_shortfall_rate":
        if reserve_total is not None and abs(float(reserve_total)) <= 1e-9:
            return 0.0
        return float("nan")

    if key == "thermal_violation_rate":
        if thermal_total is not None and abs(float(thermal_total)) <= 1e-9:
            return 0.0
        if feasibility is not None and reserve_total is not None and abs(float(reserve_total)) <= 1e-9:
            return max(0.0, 1.0 - float(feasibility))
        return float("nan")

    raise KeyError(key)


def plot_compare(task_dir: Path = TASK_DIR) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _collect_runs(task_dir)
    if not rows:
        print("[n_env_safe_compare] No matching runs found.")
        return Path()

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5), constrained_layout=True)
    ax_curve, ax_cost = axes[0]
    ax_reserve, ax_thermal = axes[1]

    # Panel 1: train-monitor eval cost vs walltime.
    for row in rows:
        x = np.asarray(row["wall_curve"], dtype=float) / 60.0
        y = np.asarray(row["cost_curve"], dtype=float) / 1e6
        label = f"{row['family']} n={row['n_envs']}"
        ax_curve.plot(
            x,
            y,
            color=row["color"],
            linestyle=row["linestyle"],
            linewidth=2.2,
            label=label,
        )
        ax_curve.scatter(x[-1], y[-1], color=row["color"], s=28)
    ax_curve.set_title("Train Monitor — Eval cost vs walltime")
    ax_curve.set_xlabel("Train walltime (minutes)")
    ax_curve.set_ylabel("Eval operating cost (million GBP)")
    ax_curve.grid(True, alpha=0.3)
    ax_curve.legend(fontsize=8, loc="best")
    ax_curve.text(
        0.01,
        0.02,
        "Train-monitor only. Formal IID comparisons are in the other three panels.",
        transform=ax_curve.transAxes,
        fontsize=7,
        color="#5d4037",
        va="bottom",
    )

    # Shared x positions by family and n_envs.
    order = [("Sauté PPO", 256), ("Sauté PPO", 128), ("PPO-Lag", 256), ("PPO-Lag", 128)]
    xpos = np.arange(len(order), dtype=float)
    xlabels = [f"{family}\nn={n}" for family, n in order]
    row_map = {(row["family"], row["n_envs"]): row for row in rows}

    # Panel 2: formal IID cost with train/eval walltime annotations.
    cost_vals: list[float] = []
    cost_colors: list[str] = []
    for key in order:
        row = row_map.get(key)
        if row is None:
            cost_vals.append(np.nan)
            cost_colors.append("#cccccc")
            continue
        cost_vals.append(float((row["eval"].get("metrics") or {}).get("total_operating_cost", np.nan)) / 1e6)
        cost_colors.append(row["color"])
    bars = ax_cost.bar(xpos, cost_vals, color=cost_colors, alpha=0.85)
    for i, key in enumerate(order):
        row = row_map.get(key)
        if row is None or not np.isfinite(cost_vals[i]):
            continue
        train_s = float(row["train"].get("walltime_s") or 0.0)
        eval_s = float(row["eval"].get("walltime_s") or 0.0)
        ax_cost.text(
            xpos[i],
            cost_vals[i] + 0.03,
            f"train {train_s:.0f}s\neval {eval_s:.0f}s",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax_cost.set_title("Formal IID — operating cost + walltime")
    ax_cost.set_ylabel("Operating cost (million GBP)")
    ax_cost.set_xticks(xpos)
    ax_cost.set_xticklabels(xlabels)
    ax_cost.grid(True, axis="y", alpha=0.3)

    # Panel 3: formal IID reserve.
    reserve_totals: list[float] = []
    reserve_rates: list[float] = []
    for key in order:
        row = row_map.get(key)
        metrics = (row or {}).get("eval", {}).get("metrics") or {}
        reserve_totals.append(float(metrics.get("total_reserve_shortfall", np.nan)))
        reserve_rates.append(_metric_with_legacy_backfill(metrics, "reserve_shortfall_rate"))
    ax_reserve.bar(xpos, reserve_totals, color=cost_colors, alpha=0.85)
    finite_reserve = [v for v in reserve_totals if np.isfinite(v)]
    reserve_ymax = max(finite_reserve + [1.0])
    reserve_offset = max(1.0, 0.02 * reserve_ymax)
    if finite_reserve and max(abs(v) for v in finite_reserve) <= 1e-9:
        # Zero-height bars are visually invisible. Keep the true zero values,
        # but add explicit markers and a small positive y-range so the panel
        # communicates "all exactly zero" instead of looking empty/broken.
        ax_reserve.scatter(
            xpos,
            np.zeros_like(xpos, dtype=float),
            s=72,
            facecolors="white",
            edgecolors=cost_colors,
            linewidths=1.8,
            zorder=3,
        )
        ax_reserve.set_ylim(0.0, 1.0)
        reserve_offset = 0.04
        ax_reserve.text(
            0.5,
            0.96,
            "All compared formal IID runs have exactly zero reserve shortfall.",
            transform=ax_reserve.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            color="#5d4037",
        )
    for i, val in enumerate(reserve_totals):
        if not np.isfinite(val):
            continue
        rate = reserve_rates[i]
        ax_reserve.text(
            xpos[i],
            val + reserve_offset,
            f"rate={rate:.4f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax_reserve.set_title("Formal IID — reserve shortfall")
    ax_reserve.set_ylabel("Total reserve shortfall")
    ax_reserve.set_xticks(xpos)
    ax_reserve.set_xticklabels(xlabels)
    ax_reserve.grid(True, axis="y", alpha=0.3)

    # Panel 4: formal IID thermal.
    thermal_totals: list[float] = []
    thermal_rates: list[float] = []
    for key in order:
        row = row_map.get(key)
        metrics = (row or {}).get("eval", {}).get("metrics") or {}
        thermal_totals.append(float(metrics.get("total_thermal_cost", np.nan)))
        thermal_rates.append(_metric_with_legacy_backfill(metrics, "thermal_violation_rate"))
    ax_thermal.bar(xpos, thermal_totals, color=cost_colors, alpha=0.85)
    ymax = max([v for v in thermal_totals if np.isfinite(v)] + [1.0])
    for i, val in enumerate(thermal_totals):
        if not np.isfinite(val):
            continue
        rate = thermal_rates[i]
        ax_thermal.text(
            xpos[i],
            val + 0.03 * ymax,
            f"rate={rate:.4f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax_thermal.set_title("Formal IID — thermal overload")
    ax_thermal.set_ylabel("Total thermal cost")
    ax_thermal.set_xticks(xpos)
    ax_thermal.set_xticklabels(xlabels)
    ax_thermal.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "TSO safe-RL n=128 vs n=256: train-monitor walltime plus formal IID outcomes",
        fontsize=12,
        y=0.98,
    )

    out = FIG_DIR / "n_env_safe_compare.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[n_env_safe_compare] saved {out} (+ .png)")
    return out


if __name__ == "__main__":
    plot_compare(TASK_DIR)
