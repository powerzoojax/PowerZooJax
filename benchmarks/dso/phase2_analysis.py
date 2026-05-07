"""DSO Phase-2 backend/device comparison.

Outputs:

* ``results/phase2_backend_summary.json``
* ``results/figures/phase2_backend_compare.{pdf,png}``

Comparison contract:

* one PPO training curve row per backend/device cell;
* formal IID safety/efficiency metrics from the latest matching IID eval row;
* canonical curves come from ``learning_curve_eval_return``,
  ``eval_cost_voltage_violation``, ``timesteps``, and
  ``learning_curve_eval_walltimes`` when available.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.common.configs import load_task_config
from benchmarks.common.io import has_training_artifact, load_manifest

TASK_DIR = Path(__file__).resolve().parent

CELL_ORDER = (
    ("jax_rejax", "gpu", "ppo", "JAX / GPU"),
    ("jax_rejax", "cpu", "ppo", "JAX / CPU"),
    ("sb3", "cuda", "ppo", "SB3 / CUDA"),
    ("sb3", "cpu", "ppo", "SB3 / CPU"),
    ("sbx", "cuda", "sbx_ppo", "SBX / CUDA"),
)
PLOT_CELL_ORDER = (
    ("jax_rejax", "gpu", "ppo", "JAX / GPU"),
    ("jax_rejax", "cpu", "ppo", "JAX / CPU"),
    ("sb3", "cuda", "ppo", "SB3 / CUDA"),
    ("sbx", "cuda", "sbx_ppo", "SBX / CUDA"),
)
CELL_COLORS = {
    ("jax_rejax", "gpu"): "#3B7DD8",
    ("jax_rejax", "cpu"): "#93c5fd",
    ("sb3", "cuda"): "#E8A33D",
    ("sb3", "cpu"): "#fbbf24",
    ("sbx", "cuda"): "#D87FB6",
}
CELL_LINESTYLES = {
    ("jax_rejax", "gpu"): "-",
    ("jax_rejax", "cpu"): "--",
    ("sb3", "cuda"): "-",
    ("sb3", "cpu"): "--",
    ("sbx", "cuda"): "-",
}


def _figures_dir(task_dir: Path) -> Path:
    d = task_dir / "results" / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sort_key(record) -> tuple[str, str]:
    return (record.timestamp or "", record.run_id)


def _artifact_path(task_dir: Path, record, key: str) -> Path | None:
    if record is None:
        return None
    rel = (record.artifacts or {}).get(key)
    if not rel:
        return None
    path = task_dir / "results" / rel
    return path if path.exists() else None


def _artifact_array(task_dir: Path, record, key: str) -> np.ndarray | None:
    path = _artifact_path(task_dir, record, key)
    if path is None:
        return None
    try:
        arr = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
    except Exception:
        return None
    return arr if arr.size > 0 else None


def _curve_timesteps(task_dir: Path, record, n: int) -> np.ndarray | None:
    xs = _artifact_array(task_dir, record, "timesteps")
    if xs is None or xs.size != n:
        return None
    return xs


def _curve_walltimes(task_dir: Path, record, n: int) -> np.ndarray | None:
    xs = _artifact_array(task_dir, record, "learning_curve_eval_walltimes")
    if xs is None or xs.size != n:
        return None
    return xs


def _metric(record, key: str) -> float | None:
    if record is None:
        return None
    value = (record.metrics or {}).get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_cross_backend_train_record(record) -> bool:
    arts = record.artifacts or {}
    if not arts.get("learning_curve_train_return"):
        return False
    if record.backend == "jax_rejax":
        return False
    return True


def _select_train_record(records: list, backend: str, device: str, algo: str, seed: int):
    candidates = [
        r
        for r in records
        if (r.backend or "jax_rejax") == backend
        and (r.device or "gpu") == device
        and r.algo == algo
        and int(r.seed) == int(seed)
        and r.status == "completed"
    ]
    if backend == "jax_rejax":
        candidates = [
            r for r in candidates
            if r.split == "train" and has_training_artifact(r.artifacts)
        ]
    else:
        candidates = [r for r in candidates if _is_cross_backend_train_record(r)]
    if not candidates:
        return None
    candidates.sort(key=_sort_key)
    return candidates[-1]


def _select_iid_eval_record(
    records: list,
    backend: str,
    device: str,
    algo: str,
    seed: int,
    train_record,
):
    if backend != "jax_rejax" and train_record is not None and train_record.split == "iid":
        if (train_record.artifacts or {}).get("per_episode"):
            return train_record

    candidates = [
        r
        for r in records
        if (r.backend or "jax_rejax") == backend
        and (r.device or "gpu") == device
        and r.algo == algo
        and int(r.seed) == int(seed)
        and r.split == "iid"
        and r.status == "completed"
        and (r.artifacts or {}).get("per_episode")
    ]
    if backend == "jax_rejax":
        candidates = [r for r in candidates if not has_training_artifact(r.artifacts)]
    if not candidates:
        return None
    candidates.sort(key=_sort_key)
    return candidates[-1]


def _mean_std(values: list[float | None]) -> tuple[float | None, float | None]:
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return None, None
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return float(np.mean(arr)), std


def _seed_row(
    task_dir: Path,
    records: list,
    backend: str,
    device: str,
    algo: str,
    label: str,
    seed: int,
) -> dict[str, Any]:
    train_record = _select_train_record(records, backend, device, algo, seed)
    iid_record = _select_iid_eval_record(records, backend, device, algo, seed, train_record)

    eval_return = _artifact_array(task_dir, train_record, "learning_curve_eval_return")
    eval_cost = _artifact_array(task_dir, train_record, "eval_cost_voltage_violation")
    timesteps = None if eval_return is None else _curve_timesteps(task_dir, train_record, len(eval_return))
    walltimes = None if eval_return is None else _curve_walltimes(task_dir, train_record, len(eval_return))
    train_artifacts = {} if train_record is None else dict(train_record.artifacts or {})
    eval_artifacts = {} if iid_record is None else dict(iid_record.artifacts or {})

    artifacts = {
        key: train_artifacts.get(key)
        for key in (
            "learning_curve_train_return",
            "learning_curve_eval_return",
            "learning_curve_eval_walltimes",
            "eval_cost_voltage_violation",
            "timesteps",
        )
        if train_artifacts.get(key)
    }
    if (
        "learning_curve_train_return" not in artifacts
        and train_artifacts.get("learning_curve_eval_return")
    ):
        # ReJAX DSO records save the monitored return curve under the eval
        # alias only. Keep a source tag so the summary does not imply a
        # separate raw artifact exists.
        artifacts["learning_curve_train_return"] = train_artifacts["learning_curve_eval_return"]
        artifacts["learning_curve_train_return_source"] = "learning_curve_eval_return"
    if eval_artifacts.get("per_episode"):
        artifacts["per_episode"] = eval_artifacts["per_episode"]

    return {
        "label": label,
        "backend": backend,
        "device": device,
        "algo": algo,
        "seed": int(seed),
        "train_run_id": None if train_record is None else train_record.run_id,
        "iid_run_id": None if iid_record is None else iid_record.run_id,
        "train_walltime_s": None if train_record is None else train_record.walltime_s,
        "train_throughput_sps": None if train_record is None else train_record.throughput_sps,
        "framework_version": None if train_record is None else train_record.framework_version,
        "curve_points": 0 if eval_return is None else int(eval_return.size),
        "curve_has_eval_return": bool(eval_return is not None),
        "curve_has_eval_cost": bool(eval_cost is not None),
        "curve_has_timesteps": bool(timesteps is not None),
        "curve_has_walltimes": bool(walltimes is not None),
        "curve_final_eval_return": None if eval_return is None else float(eval_return[-1]),
        "curve_final_eval_cost_voltage_violation": None if eval_cost is None else float(eval_cost[-1]),
        "iid_total_reward": _metric(iid_record, "total_reward"),
        "iid_voltage_violation_count_per_step": _metric(iid_record, "voltage_violation_count_per_step"),
        "iid_network_loss_reduction_pct": _metric(iid_record, "network_loss_reduction_pct"),
        "iid_total_voltage_violations": _metric(iid_record, "total_voltage_violations"),
        "artifacts": artifacts,
    }


def _build_summary(task_dir: Path, *, seeds: list[int]) -> dict[str, Any]:
    records = load_manifest(task_dir)
    rows: list[dict[str, Any]] = []

    for backend, device, algo, label in CELL_ORDER:
        seed_rows = [
            _seed_row(task_dir, records, backend, device, algo, label, seed)
            for seed in seeds
        ]
        completed = [r for r in seed_rows if r["train_run_id"] and r["iid_run_id"]]
        row: dict[str, Any] = {
            "label": label,
            "backend": backend,
            "device": device,
            "algo": algo,
            "requested_seeds": [int(seed) for seed in seeds],
            "completed_seeds": [int(r["seed"]) for r in completed],
            "missing_seeds": [
                int(r["seed"]) for r in seed_rows if not (r["train_run_id"] and r["iid_run_id"])
            ],
            "n_seeds": len(completed),
            "seed_rows": seed_rows,
        }
        for key in (
            "train_walltime_s",
            "train_throughput_sps",
            "curve_final_eval_return",
            "curve_final_eval_cost_voltage_violation",
            "iid_total_reward",
            "iid_voltage_violation_count_per_step",
            "iid_network_loss_reduction_pct",
            "iid_total_voltage_violations",
        ):
            mean, std = _mean_std([r.get(key) for r in completed])
            row[f"{key}_mean"] = mean
            row[f"{key}_std"] = std
            if len(seeds) == 1:
                row[key] = mean
        row["all_required_artifacts_present"] = all(
            all(
                r["artifacts"].get(key)
                for key in (
                    "learning_curve_train_return",
                    "learning_curve_eval_return",
                    "learning_curve_eval_walltimes",
                    "eval_cost_voltage_violation",
                    "timesteps",
                    "per_episode",
                )
            )
            for r in completed
        )
        rows.append(row)

    out: dict[str, Any] = {"task": "dso", "seeds": [int(seed) for seed in seeds], "rows": rows}
    if len(seeds) == 1:
        out["seed"] = int(seeds[0])
    return out


def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25)


def _plot_curve_family(
    ax,
    curves: list[tuple[np.ndarray, np.ndarray]],
    *,
    color: str,
    linestyle: str,
    label: str,
    x_scale: float = 1.0,
) -> None:
    """Plot an interpolated across-seed mean curve with a one-std band."""
    if not curves:
        return

    if len(curves) == 1:
        xs, ys = curves[0]
        ax.plot(
            xs / x_scale,
            ys,
            color=color,
            linestyle=linestyle,
            linewidth=2.2,
            label=label,
        )
        return

    starts = [float(np.min(xs)) for xs, _ in curves if xs.size]
    stops = [float(np.max(xs)) for xs, _ in curves if xs.size]
    if not starts or not stops:
        return
    lo = max(starts)
    hi = min(stops)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return

    n = min(240, max(32, min(xs.size for xs, _ in curves)))
    grid = np.linspace(lo, hi, n)
    vals = np.stack([np.interp(grid, xs, ys) for xs, ys in curves], axis=0)
    mean = vals.mean(axis=0)
    std = vals.std(axis=0, ddof=1) if vals.shape[0] > 1 else np.zeros_like(mean)

    ax.plot(
        grid / x_scale,
        mean,
        color=color,
        linestyle=linestyle,
        linewidth=2.4,
        label=label,
    )
    ax.fill_between(
        grid / x_scale,
        mean - std,
        mean + std,
        color=color,
        alpha=0.12,
        linewidth=0,
    )


def _plot_phase2(task_dir: Path, summary: dict[str, Any], *, seeds: list[int]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.25,
            "pdf.fonttype": 42,
        }
    )

    records = load_manifest(task_dir)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.36), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax_ret, ax_wall = axes

    for backend, device, algo, label in PLOT_CELL_ORDER:
        color = CELL_COLORS[(backend, device)]
        ls = CELL_LINESTYLES[(backend, device)]
        ret_curves: list[tuple[np.ndarray, np.ndarray]] = []
        wall_curves: list[tuple[np.ndarray, np.ndarray]] = []

        for seed in seeds:
            train_record = _select_train_record(records, backend, device, algo, seed)
            if train_record is None:
                continue
            eval_return = _artifact_array(task_dir, train_record, "learning_curve_eval_return")
            if eval_return is not None:
                xs = _curve_timesteps(task_dir, train_record, len(eval_return))
                if xs is None:
                    xs = np.linspace(0.0, float(len(eval_return)), len(eval_return))
                ret_curves.append((xs, eval_return))
                wall = _curve_walltimes(task_dir, train_record, len(eval_return))
                if wall is None:
                    total_wall = float(train_record.walltime_s or 0.0)
                    if total_wall > 0:
                        wall = np.linspace(0.0, total_wall, len(eval_return))
                if wall is not None:
                    wall_curves.append((wall, eval_return))

        _plot_curve_family(ax_ret, ret_curves, color=color, linestyle=ls, label=label, x_scale=1e6)
        _plot_curve_family(ax_wall, wall_curves, color=color, linestyle=ls, label=label)

    ax_ret.set_title("Eval return vs environment steps", fontsize=10)
    ax_ret.set_xlabel("Environment steps")
    ax_ret.set_ylabel("Eval return")
    ax_ret.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}M"))
    ax_ret.legend(fontsize=10, framealpha=0.95, loc="lower right")

    ax_wall.set_title("Eval return vs wall time", fontsize=10)
    ax_wall.set_xlabel("Wall time (s)")
    ax_wall.set_ylabel("Eval return")
    ax_wall.legend(fontsize=10, framealpha=0.95, loc="lower right")

    for ax in (ax_ret, ax_wall):
        _style_axes(ax)

    seed_label = str(seeds[0]) if len(seeds) == 1 else f"{min(seeds)}-{max(seeds)}"
    fig.suptitle(
        f"DSO Phase-2 Backend Comparison (PPO, seeds {seed_label})",
        fontsize=10,
    )

    out = _figures_dir(task_dir) / "phase2_backend_compare.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=220)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", default=str(TASK_DIR))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seed list. Defaults to task.yaml::seeds when provided; --seed is kept for seed-0 compatibility.",
    )
    args = parser.parse_args(argv)

    task_dir = Path(args.task_dir)
    if args.seeds:
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    else:
        try:
            cfg_seeds = load_task_config(task_dir).get("seeds")
        except Exception:
            cfg_seeds = None
        seeds = [int(x) for x in cfg_seeds] if cfg_seeds else [int(args.seed)]
    summary = _build_summary(task_dir, seeds=seeds)
    summary_path = task_dir / "results" / "phase2_backend_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    figure_path = _plot_phase2(task_dir, summary, seeds=seeds)
    print(f"[phase2_analysis] wrote {summary_path}")
    print(f"[phase2_analysis] wrote {figure_path}")


if __name__ == "__main__":
    main()
