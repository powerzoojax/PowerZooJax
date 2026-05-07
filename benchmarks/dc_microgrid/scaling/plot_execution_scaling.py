#!/usr/bin/env python
"""Plot DC Microgrid execution-scaling artifacts.

Reads ``results/scaling/execution_scaling.json`` and writes the two figures
required by the execution-scaling addendum.  The plots use execution metrics
only and do not include final reward.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.ticker as mticker

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

TASK_DIR = _PROJECT_ROOT / "benchmarks" / "dc_microgrid"
DEFAULT_INPUT = TASK_DIR / "results" / "scaling" / "execution_scaling.json"
DEFAULT_FIGURE_DIR = TASK_DIR / "results" / "figures"

JAX_COLOR = "#3B7DD8"
SB3_COLOR = "#E8A33D"
GRID_COLOR = "#d5dbe3"


def _figure_rc() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
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
            "grid.linestyle": "-",
            "grid.linewidth": 0.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _format_k(value: float, _pos: int | None = None) -> str:
    if abs(value) >= 1000:
        return f"{value / 1000:g}K"
    return f"{value:g}"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Scaling artifact not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        row for row in payload.get("results", [])
        if row.get("status") == "completed"
        and row.get("steady_state_env_steps_per_second") is not None
    ]
    if not rows:
        raise ValueError(f"No completed scaling rows with throughput in {path}")
    return rows


def _group_mean(
    rows: list[dict[str, Any]],
    *,
    suite: str,
    backends: set[str] | None = None,
) -> dict[tuple[str, int], dict[str, float]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        row_suite = row.get("suite")
        if suite == "matched":
            if row_suite not in ("matched", "all"):
                continue
        elif row_suite != suite:
            continue
        if backends is not None and row.get("backend") not in backends:
            continue
        key = (str(row.get("backend")), int(row.get("nenv")))
        grouped[key].append(row)

    out: dict[tuple[str, int], dict[str, float]] = {}
    for key, vals in grouped.items():
        out[key] = {
            "throughput": mean(float(v["steady_state_env_steps_per_second"]) for v in vals),
            "seconds_per_update": mean(float(v["seconds_per_update"]) for v in vals),
            "wall_clock_per_1m": mean(float(v["wall_clock_per_1M_transitions"]) for v in vals),
        }
    return out


def _plot_matched(rows: list[dict[str, Any]], figure_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    _figure_rc()
    full = _group_mean(rows, suite="matched")
    sb3_max_nenv = max(
        (nenv for (backend, nenv) in full if backend == "sb3"),
        default=None,
    )
    if sb3_max_nenv is None:
        raise ValueError("Matched-range plot needs SB3 measurements to define the like-for-like range")
    data = {(backend, nenv): vals for (backend, nenv), vals in full.items() if nenv <= sb3_max_nenv}
    backends = sorted({backend for backend, _nenv in data})
    if len(backends) < 2:
        raise ValueError("Matched-range plot needs at least two completed backends")

    colors = {"jax_rejax": JAX_COLOR, "sb3": SB3_COLOR}
    labels = {"jax_rejax": "PowerZooJax (JAX-GPU)", "sb3": "SB3/CUDA"}
    markers = {"jax_rejax": "o", "sb3": "s"}

    fig, ax = plt.subplots(figsize=(10.0, 3.6), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for backend in backends:
        points = sorted((nenv, vals["throughput"]) for (b, nenv), vals in data.items() if b == backend)
        ax.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            marker=markers.get(backend, "o"),
            linewidth=2.0,
            color=colors.get(backend, "gray"),
            label=labels.get(backend, backend),
        )
        nenv, sps = points[-1]
        ax.annotate(
            f"{sps / 1000:.0f}K",
            (nenv, sps),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            color=colors.get(backend, "gray"),
        )
    ax.set_xlabel("Parallel environments (nenv)")
    ax.set_ylabel("Steady-state env steps / second")
    ax.set_title("DC Microgrid matched-range execution scaling", loc="left")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_format_k))
    ax.grid(axis="y", color=GRID_COLOR, alpha=0.35, linewidth=0.5)
    ax.margins(x=0.05, y=0.12)
    ax.legend(frameon=False, loc="upper left")
    paths = [
        figure_dir / "execution_scaling_matched_range.png",
        figure_dir / "execution_scaling_matched_range.pdf",
    ]
    for path in paths:
        fig.savefig(path, dpi=200)
    plt.close(fig)
    return paths


def _plot_jax_extended(rows: list[dict[str, Any]], figure_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    _figure_rc()
    data = _group_mean(rows, suite="jax-extended", backends={"jax_rejax"})
    if len(data) < 2:
        raise ValueError("JAX-extended plot needs at least two completed JAX rows")
    points = sorted((nenv, vals["throughput"], vals["wall_clock_per_1m"]) for (_b, nenv), vals in data.items())

    fig, ax1 = plt.subplots(figsize=(10.0, 3.6), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("white")
    ax1.plot(
        [p[0] for p in points],
        [p[1] for p in points],
        marker="o",
        linewidth=2.0,
        color=JAX_COLOR,
        label="env steps / second",
    )
    ax1.set_xlabel("Parallel environments (nenv)")
    ax1.set_ylabel("Steady-state env steps / second")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(_format_k))
    ax1.grid(axis="y", color=GRID_COLOR, alpha=0.35, linewidth=0.5)
    ax2 = ax1.twinx()
    ax2.plot(
        [p[0] for p in points],
        [p[2] for p in points],
        marker="s",
        linewidth=2.0,
        color="#dc2626",
        label="seconds / 1M transitions",
    )
    ax2.set_ylabel("Wall clock seconds / 1M transitions")
    ax1.set_title("DC Microgrid JAX-only extended scaling", loc="left")
    ax1.annotate(
        f"{points[-1][1] / 1000:.0f}K",
        (points[-1][0], points[-1][1]),
        xytext=(5, 0),
        textcoords="offset points",
        va="center",
        color=JAX_COLOR,
    )
    ax2.annotate(
        f"{points[-1][2]:.1f}s",
        (points[-1][0], points[-1][2]),
        xytext=(5, 0),
        textcoords="offset points",
        va="center",
        color="#dc2626",
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper right")
    paths = [
        figure_dir / "execution_scaling_jax_extended.png",
        figure_dir / "execution_scaling_jax_extended.pdf",
    ]
    for path in paths:
        fig.savefig(path, dpi=200)
    plt.close(fig)
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--figure-dir", default=str(DEFAULT_FIGURE_DIR))
    parser.add_argument("--only", choices=["matched", "jax-extended"], default=None)
    args = parser.parse_args(argv)

    rows = _load_rows(Path(args.input))
    figure_dir = Path(args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if args.only in (None, "matched"):
        written.extend(_plot_matched(rows, figure_dir))
    if args.only in (None, "jax-extended"):
        written.extend(_plot_jax_extended(rows, figure_dir))
    for path in written:
        print(f"[execution_scaling_plot] wrote {path}")


if __name__ == "__main__":
    main()
