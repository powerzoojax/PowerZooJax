#!/usr/bin/env python
"""TSO episode-level scatter: operating_cost vs thermal_cost.

Reads manifest-deduped IID per-episode artifacts (4 algos x 5 seeds x 50 eps =
1000 points) and renders a single-panel scatter with mean +/- 95% CI
confidence ellipse per algorithm.  Complements the mean-frontier figure in
the main text by exposing within-policy episode-level dispersion.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.patches import Ellipse

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks._paper_style import apply_rc  # noqa: E402
from benchmarks.common.io import load_manifest_filtered  # noqa: E402

FS = 12

TASK_DIR = _PROJECT_ROOT / "benchmarks" / "tso"
RESULTS_DIR = TASK_DIR / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
MATERIALS_FIGURE_DIR = _PROJECT_ROOT / "BenchmarkPaper" / "materials" / "figures"

ALGOS = ("ppo", "ppo_lagrangian", "merit_order", "all_on")
ALGO_LABELS = {
    "ppo": "PPO",
    "ppo_lagrangian": "PPO-Lag",
    "merit_order": "Merit-order",
    "all_on": "All-on",
}
ALGO_COLORS = {
    "ppo": "#D6453F",
    "ppo_lagrangian": "#3B7DD8",
    "merit_order": "#7E7E7E",
    "all_on": "#1A1A1A",
}
ALGO_MARKERS = {
    "ppo": "o",
    "ppo_lagrangian": "s",
    "merit_order": "^",
    "all_on": "D",
}

EXPECTED_SEEDS_PER_ALGO = 5
EXPECTED_EPISODES_PER_SEED = 50


def _figure_rc() -> None:
    apply_rc()
    plt.rcParams.update(
        {
            "font.size": FS,
            "axes.titlesize": FS,
            "axes.labelsize": FS,
            "xtick.labelsize": FS,
            "ytick.labelsize": FS,
            "legend.fontsize": FS,
        }
    )


def _collect_points() -> dict[str, np.ndarray]:
    records = load_manifest_filtered(TASK_DIR, backend="jax_rejax", device="gpu")
    iid = [r for r in records if r.split == "iid" and r.status == "completed"]

    points: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    seeds_by_algo: dict[str, set[int]] = defaultdict(set)

    for rec in iid:
        if rec.algo not in ALGOS:
            continue
        pe_path = rec.artifacts.get("per_episode")
        if not pe_path:
            raise RuntimeError(f"Missing per_episode artifact for {rec.run_id}")
        episodes = json.loads((RESULTS_DIR / pe_path).read_text(encoding="utf-8"))
        if len(episodes) != EXPECTED_EPISODES_PER_SEED:
            raise RuntimeError(
                f"{rec.run_id}: expected {EXPECTED_EPISODES_PER_SEED} episodes, "
                f"got {len(episodes)}"
            )
        seeds_by_algo[rec.algo].add(rec.seed)
        for ep in episodes:
            points[rec.algo].append(
                (
                    float(ep["total_operating_cost"]),
                    float(ep["total_thermal_cost"]),
                    float(ep["thermal_violation_rate"]),
                )
            )

    for algo in ALGOS:
        if len(seeds_by_algo[algo]) != EXPECTED_SEEDS_PER_ALGO:
            raise RuntimeError(
                f"{algo}: expected {EXPECTED_SEEDS_PER_ALGO} seeds, "
                f"got {sorted(seeds_by_algo[algo])}"
            )
        n = len(points[algo])
        target = EXPECTED_SEEDS_PER_ALGO * EXPECTED_EPISODES_PER_SEED
        if n != target:
            raise RuntimeError(f"{algo}: expected {target} points, got {n}")

    return {a: np.asarray(points[a], dtype=float) for a in ALGOS}


def _confidence_ellipse(ax, xs, ys, color, n_std: float = 1.96) -> None:
    """Draw a covariance ellipse capturing roughly the central 95% mass."""
    if xs.size < 3:
        return
    cov = np.cov(xs, ys)
    if not np.isfinite(cov).all():
        return
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
    width, height = 2.0 * n_std * np.sqrt(np.maximum(eigvals, 0.0))
    ellipse = Ellipse(
        (xs.mean(), ys.mean()),
        width=width,
        height=height,
        angle=angle,
        edgecolor=color,
        facecolor="none",
        linewidth=1.6,
        alpha=0.85,
        zorder=4,
    )
    ax.add_patch(ellipse)


def plot(data: dict[str, np.ndarray]) -> tuple[Path, Path]:
    _figure_rc()
    fig, ax = plt.subplots(figsize=(10.0, 4.8), constrained_layout=False)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Symlog y-axis: many episodes have thermal_cost = 0 for safe baselines.
    ax.set_yscale("symlog", linthresh=1e-2, linscale=0.6)

    # Size by thermal_violation_rate so the third dimension is legible.
    size_min, size_max = 12.0, 110.0
    for algo in ALGOS:
        pts = data[algo]
        xs, ys, vr = pts[:, 0], pts[:, 1], pts[:, 2]
        sizes = size_min + (size_max - size_min) * np.clip(vr, 0.0, 1.0)
        ax.scatter(
            xs,
            ys,
            s=sizes,
            c=ALGO_COLORS[algo],
            marker=ALGO_MARKERS[algo],
            alpha=0.32,
            linewidths=0.0,
            label=f"{ALGO_LABELS[algo]} (n={len(pts)})",
            zorder=2,
        )
        _confidence_ellipse(ax, xs, ys, ALGO_COLORS[algo])
        ax.scatter(
            [xs.mean()],
            [ys.mean()],
            s=80,
            c=ALGO_COLORS[algo],
            marker=ALGO_MARKERS[algo],
            edgecolors="white",
            linewidths=1.2,
            zorder=5,
        )

    ax.set_xlabel(r"Total operating cost (£)")
    ax.set_ylabel("Total thermal-violation cost (symlog)")

    def _fmt_pounds(value: float, _pos: int | None = None) -> str:
        if value >= 1e6:
            return f"£{value / 1e6:.1f}M"
        if value >= 1e3:
            return f"£{value / 1e3:.0f}K"
        return f"£{value:.0f}"

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_pounds))
    # thermal_cost >= 0 always; clip the symlog axis so we don't waste space
    # on the symmetric negative side.
    y_max = max(data[a][:, 1].max() for a in ALGOS)
    ax.set_ylim(0.0, y_max * 1.6)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        handletextpad=0.4,
        columnspacing=0.9,
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.15, top=0.84)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURE_DIR / "episode_scatter_cost_vs_thermal.pdf"
    png_path = FIGURE_DIR / "episode_scatter_cost_vs_thermal.png"
    fig.savefig(pdf_path, dpi=200, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def _print_summary(data: dict[str, np.ndarray]) -> None:
    print("Per-algo summary (manifest-deduped IID):")
    for algo in ALGOS:
        pts = data[algo]
        xs, ys, vr = pts[:, 0], pts[:, 1], pts[:, 2]
        print(
            f"  {ALGO_LABELS[algo]:<12} n={len(pts):4d}  "
            f"op_cost mean=£{xs.mean() / 1e6:5.2f}M  "
            f"thermal mean={ys.mean():.2f}  viol_rate mean={vr.mean():.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--copy-to-paper",
        action="store_true",
        help="Also copy the rendered PDF/PNG to BenchmarkPaper/materials/figures/.",
    )
    args = parser.parse_args()

    data = _collect_points()
    _print_summary(data)
    pdf_path, png_path = plot(data)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")

    if args.copy_to_paper:
        MATERIALS_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        for src in (pdf_path, png_path):
            dst = MATERIALS_FIGURE_DIR / src.name
            dst.write_bytes(src.read_bytes())
            print(f"Copied -> {dst}")


if __name__ == "__main__":
    main()
