"""Current-only TSO safe-RL comparison figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TASK_DIR = Path(__file__).resolve().parent


def _load_run(algo: str, split: str) -> dict:
    runs_dir = TASK_DIR / "results" / "runs"
    matches = []
    for path in runs_dir.glob("*.json"):
        obj = json.loads(path.read_text(encoding="utf-8"))
        if obj.get("algo") != algo or obj.get("split") != split:
            continue
        matches.append(obj)
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {algo}/{split} run, got {len(matches)}.")
    return matches[0]


def _load_artifact(run: dict, key: str) -> np.ndarray:
    rel = run["artifacts"].get(key)
    if not rel:
        raise KeyError(f"Run {run['run_id']} missing artifact {key!r}")
    return np.load(TASK_DIR / "results" / rel)


def generate_current_safe_compare() -> tuple[Path, Path]:
    ppo_train = _load_run("ppo_lagrangian", "train")
    ppo_iid = _load_run("ppo_lagrangian", "iid")
    saute_train = _load_run("saute_ppo", "train")
    saute_iid = _load_run("saute_ppo", "iid")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("TSO safe-RL current benchmark: train-monitor plus formal IID")

    colors = {
        "PPO-Lag": "#d81b60",
        "Sauté PPO": "#8e24aa",
    }

    # Train-monitor panel.
    ax = axes[0, 0]
    ppo_x = _load_artifact(ppo_train, "checkpoint_walltime_s") / 60.0
    ppo_y = _load_artifact(ppo_train, "eval_total_operating_cost") / 1e6
    saute_x = _load_artifact(saute_train, "eval_wall_time_s") / 60.0
    saute_y = _load_artifact(saute_train, "eval_total_operating_cost") / 1e6
    ax.plot(ppo_x, ppo_y, color=colors["PPO-Lag"], lw=2, label="PPO-Lag")
    ax.plot(saute_x, saute_y, color=colors["Sauté PPO"], lw=2, label="Sauté PPO")
    ax.set_title("Train Monitor — eval cost vs walltime")
    ax.set_xlabel("Train walltime (minutes)")
    ax.set_ylabel("Eval operating cost (million GBP)")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.text(
        0.01,
        0.02,
        "Train-monitor only. Formal IID is in the other panels.",
        transform=ax.transAxes,
        fontsize=9,
        ha="left",
        va="bottom",
    )

    labels = ["PPO-Lag", "Sauté PPO"]
    iid_runs = [ppo_iid, saute_iid]
    train_runs = [ppo_train, saute_train]
    xs = np.arange(len(labels))

    # Formal IID operating cost.
    ax = axes[0, 1]
    costs = [run["metrics"]["total_operating_cost"] / 1e6 for run in iid_runs]
    bars = ax.bar(xs, costs, color=[colors[label] for label in labels], alpha=0.85)
    ax.set_title("Formal IID — operating cost")
    ax.set_ylabel("Operating cost (million GBP)")
    ax.set_xticks(xs, labels)
    ax.grid(axis="y", alpha=0.3)
    for x, train_run, eval_run, bar in zip(xs, train_runs, iid_runs, bars):
        ax.text(
            x,
            bar.get_height() + 0.03,
            f"train {train_run['walltime_s']:.0f}s\neval {eval_run['walltime_s']:.0f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Formal IID reserve shortfall.
    ax = axes[1, 0]
    reserves = [run["metrics"]["total_reserve_shortfall"] for run in iid_runs]
    max_reserve = max(reserves) if reserves else 0.0
    if max_reserve <= 1e-9:
        ax.scatter(xs, np.zeros_like(xs, dtype=float), color="#2e7d32", s=80, zorder=3)
        for x in xs:
            ax.text(x, 0.0002, "0", ha="center", va="bottom", fontsize=10)
        ax.text(
            0.5,
            0.95,
            "all exactly zero reserve shortfall",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            color="#2e7d32",
        )
        ax.set_ylim(-0.005, 0.01)
    else:
        ax.bar(xs, reserves, color=[colors[label] for label in labels], alpha=0.85)
    ax.set_title("Formal IID — reserve shortfall")
    ax.set_ylabel("Total reserve shortfall")
    ax.set_xticks(xs, labels)
    ax.grid(axis="y", alpha=0.3)

    # Formal IID thermal overload.
    ax = axes[1, 1]
    thermals = [run["metrics"]["total_thermal_cost"] for run in iid_runs]
    bars = ax.bar(xs, thermals, color=[colors[label] for label in labels], alpha=0.85)
    ax.set_title("Formal IID — thermal overload")
    ax.set_ylabel("Total thermal cost")
    ax.set_xticks(xs, labels)
    ax.grid(axis="y", alpha=0.3)
    for x, run, bar in zip(xs, iid_runs, bars):
        ax.text(
            x,
            bar.get_height() + 0.08,
            f"rate={run['metrics']['thermal_violation_rate']:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_dir = TASK_DIR / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "current_safe_compare.pdf"
    png_path = out_dir / "current_safe_compare.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return pdf_path, png_path


if __name__ == "__main__":
    pdf_path, png_path = generate_current_safe_compare()
    print(pdf_path)
    print(png_path)
