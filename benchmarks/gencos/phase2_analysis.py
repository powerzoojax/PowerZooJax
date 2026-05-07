"""GenCos Phase-2 backend/device comparison.

Outputs:

* ``results/phase2_backend_summary.json``
* ``results/figures/phase2_backend_compare.{pdf,png}``

Comparison contract:

* one training row per backend/device cell, always from ``split="train"``
  training-class records;
* one official eval row per split from ``train / iid / demand_shift /
  renewable_shock``;
* training curves come from the training record's canonical artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.common.io import has_training_artifact, load_manifest

DEFAULT_TASK_DIR = Path(__file__).resolve().parent
PLOT_FACE = "#f7f3eb"
AX_FACE = "#fffdf8"
GRID_COLOR = "#d7cfbf"
CELL_ORDER = (
    ("jax_rejax", "gpu", "ippo"),
    ("jax_rejax", "cpu", "ippo"),
    ("sb3", "cuda", "ppo"),
    ("sbx", "cuda", "sbx_ppo"),
)
SPLIT_ORDER = ("train", "iid", "demand_shift", "renewable_shock")
SPLIT_LABELS = {
    "train": "Train",
    "iid": "IID",
    "demand_shift": "Demand +10%",
    "renewable_shock": "RenShock +5%",
}
CELL_COLORS = {
    ("jax_rejax", "gpu"): "#1f6feb",
    ("jax_rejax", "cpu"): "#5e8cff",
    ("sb3", "cuda"): "#d8891c",
    ("sbx", "cuda"): "#c44536",
}
CROSS_BACKEND_EVAL_METRIC = "mean_per_agent_cumulative_profit"
CROSS_BACKEND_EVAL_LABEL = "Profit ($/episode/agent)"


def _figures_dir(task_dir: Path) -> Path:
    return task_dir / "results" / "figures"


def _style_axes(ax) -> None:
    ax.set_facecolor(AX_FACE)
    ax.grid(True, color=GRID_COLOR, alpha=0.45, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(
        path.with_suffix(".png"),
        dpi=150,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )


def _select_latest_record(
    task_dir: Path,
    *,
    backend: str,
    device: str,
    algo: str,
    split: str,
    seed: int,
    training: bool | None,
):
    records = [
        r
        for r in load_manifest(task_dir)
        if r.backend == backend
        and r.device == device
        and r.algo == algo
        and r.split == split
        and r.seed == seed
    ]
    if training is not None:
        records = [
            r for r in records if has_training_artifact(r.artifacts) is bool(training)
        ]
    if not records:
        return None
    completed = [r for r in records if r.status == "completed"]
    chosen = completed if completed else records
    chosen.sort(key=lambda r: (r.timestamp, r.run_id))
    return chosen[-1]


def _artifact_array(task_dir: Path, record, key: str) -> np.ndarray | None:
    if record is None:
        return None
    rel = (record.artifacts or {}).get(key)
    if not rel:
        return None
    path = task_dir / "results" / rel
    if not path.exists():
        return None
    try:
        arr = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if arr.size == 0:
        return None
    return arr


def _curve_xs(task_dir: Path, record, n: int) -> np.ndarray:
    xs = _artifact_array(task_dir, record, "timesteps")
    if xs is None or xs.size != n:
        return np.linspace(0.0, 5.0, n, dtype=np.float64)
    return xs / 1e6


def _metric(record, key: str) -> float | None:
    if record is None:
        return None
    value = record.metrics.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_summary(task_dir: Path, *, seed: int = 0) -> dict:
    rows = []
    for backend, device, algo in CELL_ORDER:
        train_record = _select_latest_record(
            task_dir,
            backend=backend,
            device=device,
            algo=algo,
            split="train",
            seed=seed,
            training=True,
        )
        split_records = {
            split: _select_latest_record(
                task_dir,
                backend=backend,
                device=device,
                algo=algo,
                split=split,
                seed=seed,
                training=False,
            )
            for split in SPLIT_ORDER
        }
        eval_curve = _artifact_array(task_dir, train_record, "learning_curve_eval_return")
        row = {
            "label": f"{backend} / {device}",
            "backend": backend,
            "device": device,
            "algo": algo,
            "train_run_id": None if train_record is None else train_record.run_id,
            "train_status": None if train_record is None else train_record.status,
            "train_walltime_s": None if train_record is None else train_record.walltime_s,
            "train_throughput_sps": None if train_record is None else train_record.throughput_sps,
            "framework_version": None if train_record is None else train_record.framework_version,
            "train_curve_points": 0 if eval_curve is None else int(eval_curve.size),
            "train_curve_final_return": None
            if eval_curve is None
            else float(eval_curve[-1]),
            "train_env_data_source": None
            if train_record is None
            else train_record.env_info.get("data_source"),
            "train_env_benchmark_split": None
            if train_record is None
            else train_record.env_info.get("benchmark_split"),
            "train_env_ood_axis": None
            if train_record is None
            else train_record.env_info.get("ood_axis"),
            "train_env_profile_window": None
            if train_record is None
            else train_record.env_info.get("profile_window"),
        }
        if train_record is not None:
            row["train_total_profit"] = _metric(train_record, "total_profit")
            row["train_mean_lmp"] = _metric(train_record, "mean_lmp")
            row["train_hhi"] = _metric(train_record, "hhi")
            row["train_ramp_binding_rate"] = _metric(train_record, "ramp_binding_rate")
            row["train_sced_convergence_rate"] = _metric(
                train_record, "sced_convergence_rate"
            )
        for split, rec in split_records.items():
            if split == "train":
                row["train_eval_run_id"] = None if rec is None else rec.run_id
                row["train_eval_status"] = None if rec is None else rec.status
                row["train_eval_total_profit"] = _metric(rec, "total_profit")
                row["train_eval_mean_lmp"] = _metric(rec, "mean_lmp")
                row["train_eval_hhi"] = _metric(rec, "hhi")
                row["train_eval_ramp_binding_rate"] = _metric(
                    rec, "ramp_binding_rate"
                )
                row["train_eval_sced_convergence_rate"] = _metric(
                    rec, "sced_convergence_rate"
                )
                row["train_eval_env_data_source"] = (
                    None if rec is None else rec.env_info.get("data_source")
                )
                row["train_eval_env_benchmark_split"] = (
                    None if rec is None else rec.env_info.get("benchmark_split")
                )
                row["train_eval_env_ood_axis"] = (
                    None if rec is None else rec.env_info.get("ood_axis")
                )
                row["train_eval_env_profile_window"] = (
                    None if rec is None else rec.env_info.get("profile_window")
                )
                continue
            prefix = f"{split}_"
            row[f"{prefix}run_id"] = None if rec is None else rec.run_id
            row[f"{prefix}status"] = None if rec is None else rec.status
            row[f"{prefix}total_profit"] = _metric(rec, "total_profit")
            row[f"{prefix}mean_lmp"] = _metric(rec, "mean_lmp")
            row[f"{prefix}hhi"] = _metric(rec, "hhi")
            row[f"{prefix}ramp_binding_rate"] = _metric(rec, "ramp_binding_rate")
            row[f"{prefix}sced_convergence_rate"] = _metric(
                rec, "sced_convergence_rate"
            )
            row[f"{prefix}env_data_source"] = None if rec is None else rec.env_info.get(
                "data_source"
            )
            row[f"{prefix}env_benchmark_split"] = (
                None if rec is None else rec.env_info.get("benchmark_split")
            )
            row[f"{prefix}env_ood_axis"] = None if rec is None else rec.env_info.get(
                "ood_axis"
            )
            row[f"{prefix}env_profile_window"] = (
                None if rec is None else rec.env_info.get("profile_window")
            )
        rows.append(row)

    gpu_row = next(
        (r for r in rows if r["backend"] == "jax_rejax" and r["device"] == "gpu"),
        None,
    )
    cpu_row = next(
        (r for r in rows if r["backend"] == "jax_rejax" and r["device"] == "cpu"),
        None,
    )
    speedup = None
    if gpu_row and cpu_row and gpu_row["train_walltime_s"] and cpu_row["train_walltime_s"]:
        speedup = float(cpu_row["train_walltime_s"]) / float(gpu_row["train_walltime_s"])

    return {
        "task": "gencos",
        "seed": seed,
        "cross_backend_eval_metric": CROSS_BACKEND_EVAL_METRIC,
        "rows": rows,
        "jax_gpu_vs_cpu_speedup": speedup,
    }


def _plot_phase2(task_dir: Path, summary: dict, *, seed: int = 0) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in summary["rows"] if r["train_run_id"]]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor(PLOT_FACE)
    for ax in axes.ravel():
        _style_axes(ax)

    # Curves: eval return / HHI / ramp binding.
    for row in rows:
        train_record = _select_latest_record(
            task_dir,
            backend=row["backend"],
            device=row["device"],
            algo=row["algo"],
            split="train",
            seed=seed,
            training=True,
        )
        color = CELL_COLORS.get((row["backend"], row["device"]), "#666666")
        label = row["label"]
        eval_curve = _artifact_array(task_dir, train_record, "learning_curve_eval_return")
        hhi_curve = _artifact_array(task_dir, train_record, "market/HHI")
        ramp_curve = _artifact_array(task_dir, train_record, "market/ramp_binding_rate")

        if eval_curve is not None:
            xs = _curve_xs(task_dir, train_record, int(eval_curve.size))
            axes[0, 0].plot(xs, eval_curve, color=color, linewidth=2.0, label=label)
        if hhi_curve is not None:
            xs = _curve_xs(task_dir, train_record, int(hhi_curve.size))
            axes[0, 1].plot(xs, hhi_curve, color=color, linewidth=2.0, label=label)
        if ramp_curve is not None:
            xs = _curve_xs(task_dir, train_record, int(ramp_curve.size))
            axes[0, 2].plot(xs, ramp_curve, color=color, linewidth=2.0, label=label)

    axes[0, 0].set_title("Training Eval Return (mean/agent)")
    axes[0, 0].set_xlabel("Timesteps (M)")
    axes[0, 0].set_ylabel(CROSS_BACKEND_EVAL_LABEL)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].set_title("Training Market HHI")
    axes[0, 1].set_xlabel("Timesteps (M)")
    axes[0, 1].set_ylabel("HHI")
    axes[0, 1].axhline(1.0 / 5.0, color="#2d8a34", lw=1.0, ls="--")
    axes[0, 1].axhline(1.0, color="#c44536", lw=1.0, ls=":")

    axes[0, 2].set_title("Training Ramp Binding Rate")
    axes[0, 2].set_xlabel("Timesteps (M)")
    axes[0, 2].set_ylabel("Rate")

    # Bars: throughput + walltime.
    xs = np.arange(len(rows))
    labels = [r["label"] for r in rows]
    colors = [CELL_COLORS.get((r["backend"], r["device"]), "#666666") for r in rows]
    throughput = np.asarray(
        [float(r["train_throughput_sps"]) for r in rows], dtype=np.float64
    )
    walltime_min = np.asarray(
        [float(r["train_walltime_s"]) / 60.0 for r in rows], dtype=np.float64
    )

    bars = axes[1, 0].bar(xs, throughput, color=colors, alpha=0.92)
    axes[1, 0].set_title("Training Throughput")
    axes[1, 0].set_ylabel("Steps / second")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xticks(xs)
    axes[1, 0].set_xticklabels(labels, rotation=18, ha="right")
    for bar, value in zip(bars, throughput):
        axes[1, 0].text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            f"{value:,.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    bars = axes[1, 1].bar(xs, walltime_min, color=colors, alpha=0.92)
    axes[1, 1].set_title("Training Walltime")
    axes[1, 1].set_ylabel("Minutes")
    axes[1, 1].set_xticks(xs)
    axes[1, 1].set_xticklabels(labels, rotation=18, ha="right")
    for bar, value in zip(bars, walltime_min):
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    speedup = summary.get("jax_gpu_vs_cpu_speedup")
    if speedup is not None:
        axes[1, 1].text(
            0.03,
            0.93,
            f"JAX gpu/cpu speedup = {speedup:.2f}x",
            transform=axes[1, 1].transAxes,
            fontsize=9,
            bbox={"facecolor": "#fff8e8", "edgecolor": "#d7cfbf", "boxstyle": "round,pad=0.25"},
        )

    # Final split robustness lines.
    split_x = np.arange(len(SPLIT_ORDER))
    for row, color in zip(rows, colors):
        ys = np.asarray(
            [row.get(f"{split}_total_profit", np.nan) for split in SPLIT_ORDER],
            dtype=np.float64,
        )
        axes[1, 2].plot(
            split_x,
            ys,
            marker="o",
            linewidth=2.0,
            color=color,
            label=row["label"],
        )
    axes[1, 2].set_title("Final Eval Total Profit Across Splits")
    axes[1, 2].set_ylabel("Total profit")
    axes[1, 2].set_xticks(split_x)
    axes[1, 2].set_xticklabels([SPLIT_LABELS[s] for s in SPLIT_ORDER], rotation=18, ha="right")
    axes[1, 2].legend(fontsize=8)

    fig.suptitle("GenCos Phase-2 backend/device comparison", y=0.995, fontsize=15)
    fig.tight_layout()
    out = _figures_dir(task_dir) / "phase2_backend_compare.pdf"
    _save_figure(fig, out)
    plt.close(fig)
    return out


def generate_phase2_backend_compare(
    task_dir: Path = DEFAULT_TASK_DIR,
    *,
    seed: int = 0,
) -> dict:
    """Write the Phase-2 summary JSON and backend comparison figure."""
    task_dir = Path(task_dir)
    summary = _build_summary(task_dir, seed=seed)
    summary_path = task_dir / "results" / "phase2_backend_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    figure_path = _plot_phase2(task_dir, summary, seed=seed)
    print(f"[phase2_analysis] wrote {summary_path}")
    print(f"[phase2_analysis] wrote {figure_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", type=Path, default=DEFAULT_TASK_DIR)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    generate_phase2_backend_compare(args.task_dir, seed=int(args.seed))


if __name__ == "__main__":
    main()
