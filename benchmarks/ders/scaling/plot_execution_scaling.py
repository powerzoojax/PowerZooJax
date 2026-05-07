#!/usr/bin/env python
"""Plot DERs execution-scaling results."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.ticker as mticker

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

TASK_DIR = _PROJECT_ROOT / "benchmarks" / "ders"
DEFAULT_INPUT = TASK_DIR / "results" / "scaling" / "execution_scaling.json"
MATERIALS_FIGURE_DIR = _PROJECT_ROOT / "BenchmarkPaper" / "materials" / "figures"

JAX_COLOR = "#3B7DD8"
SB3_COLOR = "#E8A33D"
GRID_COLOR = "#d5dbe3"


def _figure_rc() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
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


def _load_rows(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Scaling artifact not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        r for r in payload.get("results", [])
        if r.get("status") == "completed" and r.get("steps_per_sec") is not None
    ]


def _group_mean(rows):
    grouped = defaultdict(list)
    for r in rows:
        key = (str(r.get("backend", "jax_rejax")), int(r["nenv"]))
        grouped[key].append(float(r["steps_per_sec"]))
    return {k: mean(v) for k, v in grouped.items()}


def plot(rows, figure_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    _figure_rc()
    data = _group_mean(rows)
    backends = sorted({b for b, _ in data})

    COLORS = {"jax_rejax": JAX_COLOR, "sb3": SB3_COLOR}
    LABELS = {"jax_rejax": "PowerZooJax (JAX-GPU)", "sb3": "SB3 (CPU)"}
    MARKERS = {"jax_rejax": "o", "sb3": "s"}

    fig, ax = plt.subplots(figsize=(10.0, 3.6), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for backend in backends:
        pts = sorted((nenv, sps) for (b, nenv), sps in data.items() if b == backend)
        if not pts:
            continue
        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            marker=MARKERS.get(backend, "o"),
            linewidth=2.0,
            color=COLORS.get(backend, "gray"),
            label=LABELS.get(backend, backend),
        )
        nenv, sps = pts[-1]
        ax.annotate(
            f"{sps / 1000:.0f}K",
            (nenv, sps),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            color=COLORS.get(backend, "gray"),
        )
    ax.set_xlabel("Parallel environments (nenv)")
    ax.set_ylabel("Env steps / second")
    ax.set_title("DERs execution scaling (IPPO)", loc="left")
    ax.grid(axis="y", color=GRID_COLOR, alpha=0.35, linewidth=0.5)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_format_k))
    ax.margins(x=0.05, y=0.12)
    ax.legend(frameon=False, loc="upper left")

    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        figure_dir / "ders_execution_scaling_matched_range.pdf",
        figure_dir / "ders_execution_scaling_matched_range.png",
    ]
    for p in paths:
        fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--figure-dir", default=str(MATERIALS_FIGURE_DIR))
    args = parser.parse_args(argv)

    rows = _load_rows(Path(args.input))
    paths = plot(rows, Path(args.figure_dir))
    for p in paths:
        print(f"[ders_scaling_plot] wrote {p}")


if __name__ == "__main__":
    main()
