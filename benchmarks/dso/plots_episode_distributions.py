#!/usr/bin/env python
"""DSO behavior distribution: 4-panel box+jitter on IID per-episode metrics.

Renders a 2x2 grid covering curtailment, shift, peak shaving and voltage
violations across 7 algorithms.  Uses manifest-deduped IID records and shows
all 250 points per algo as semi-transparent jitter overlaid on a box plot so
that the distributional tails (where PPO-Lag / SAC voltage violations live)
remain visible.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks._paper_style import apply_rc  # noqa: E402
from benchmarks.common.io import load_manifest_filtered  # noqa: E402

FS = 12

TASK_DIR = _PROJECT_ROOT / "benchmarks" / "dso"
RESULTS_DIR = TASK_DIR / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
MATERIALS_FIGURE_DIR = _PROJECT_ROOT / "BenchmarkPaper" / "materials" / "figures"

ALGO_ORDER = (
    "ppo",
    "saute_ppo",
    "ppo_lagrangian",
    "sac",
    "tou",
    "droop",
    "no_control",
)
ALGO_LABELS = {
    "ppo": "PPO",
    "saute_ppo": "Sauté-PPO",
    "ppo_lagrangian": "PPO-Lag",
    "sac": "SAC",
    "tou": "TOU",
    "droop": "Droop",
    "no_control": "No-control",
}
LEARNING_ALGOS = {"ppo", "saute_ppo", "ppo_lagrangian", "sac"}
ALGO_COLORS = {
    "ppo": "#D6453F",
    "saute_ppo": "#E07A29",
    "ppo_lagrangian": "#3B7DD8",
    "sac": "#7C42B8",
    "tou": "#7E7E7E",
    "droop": "#3F6F4F",
    "no_control": "#1A1A1A",
}

METRICS = (
    ("total_curtailed_mwh", "Curtailment (MWh / day)"),
    ("total_shifted_mwh", "Shift (MWh / day)"),
    ("peak_shaving_pct", "Peak shaving (%)"),
    ("voltage_violation_count_per_step", "Voltage violations (per step)"),
)

EXPECTED_SEEDS = 5
EXPECTED_EPISODES = 50


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


def _collect() -> dict[str, dict[str, np.ndarray]]:
    records = load_manifest_filtered(TASK_DIR, backend="jax_rejax", device="gpu")
    iid = [r for r in records if r.split == "iid" and r.status == "completed"]

    raw: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {m: [] for m, _ in METRICS}
    )
    seeds_by_algo: dict[str, set[int]] = defaultdict(set)

    for rec in iid:
        if rec.algo not in ALGO_ORDER:
            continue
        pe_path = rec.artifacts.get("per_episode")
        if not pe_path:
            raise RuntimeError(f"Missing per_episode artifact for {rec.run_id}")
        episodes = json.loads((RESULTS_DIR / pe_path).read_text(encoding="utf-8"))
        if len(episodes) != EXPECTED_EPISODES:
            raise RuntimeError(
                f"{rec.run_id}: expected {EXPECTED_EPISODES} episodes, "
                f"got {len(episodes)}"
            )
        seeds_by_algo[rec.algo].add(rec.seed)
        for ep in episodes:
            for metric, _ in METRICS:
                raw[rec.algo][metric].append(float(ep[metric]))

    target = EXPECTED_SEEDS * EXPECTED_EPISODES
    for algo in ALGO_ORDER:
        if len(seeds_by_algo[algo]) != EXPECTED_SEEDS:
            raise RuntimeError(
                f"{algo}: expected {EXPECTED_SEEDS} seeds, "
                f"got {sorted(seeds_by_algo[algo])}"
            )
        n = len(raw[algo][METRICS[0][0]])
        if n != target:
            raise RuntimeError(f"{algo}: expected {target} points, got {n}")

    return {a: {m: np.asarray(v) for m, v in raw[a].items()} for a in ALGO_ORDER}


def plot(data: dict[str, dict[str, np.ndarray]]) -> tuple[Path, Path]:
    _figure_rc()
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.0), constrained_layout=True)
    fig.patch.set_facecolor("white")

    rng = np.random.default_rng(0)

    for ax, (metric, label) in zip(axes.flat, METRICS):
        ax.set_facecolor("white")
        # Box plot per algo (horizontal).
        box_data = [data[a][metric] for a in ALGO_ORDER]
        bp = ax.boxplot(
            box_data,
            vert=False,
            widths=0.55,
            showfliers=False,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.2),
            whiskerprops=dict(color="#555", linewidth=0.9),
            capprops=dict(color="#555", linewidth=0.9),
            boxprops=dict(linewidth=0.6, edgecolor="#444"),
        )
        for box, algo in zip(bp["boxes"], ALGO_ORDER):
            box.set_facecolor(ALGO_COLORS[algo])
            box.set_alpha(0.20)

        # Strip plot (jitter) so distributional tails are visible.
        for idx, algo in enumerate(ALGO_ORDER):
            ys = idx + 1 + rng.uniform(-0.18, 0.18, size=len(data[algo][metric]))
            ax.scatter(
                data[algo][metric],
                ys,
                s=8,
                c=ALGO_COLORS[algo],
                alpha=0.45,
                linewidths=0.0,
                zorder=3,
            )

        ax.set_yticks(range(1, len(ALGO_ORDER) + 1))
        ax.set_yticklabels([ALGO_LABELS[a] for a in ALGO_ORDER])
        ax.set_xlabel(label)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3, linewidth=0.4)
        ax.grid(axis="y", visible=False)

    fig.suptitle(
        "DSO IID per-episode behavior distributions "
        "(7 algorithms x 5 seeds x 50 episodes; box + jitter)",
        x=0.01,
        y=1.02,
        ha="left",
        fontsize=FS,
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURE_DIR / "episode_distributions.pdf"
    png_path = FIGURE_DIR / "episode_distributions.png"
    fig.savefig(pdf_path, dpi=200, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def _print_summary(data: dict[str, dict[str, np.ndarray]]) -> None:
    print("Per-algo IID behavior summary (manifest-deduped):")
    header = f"  {'algo':<11} {'n':>4} "
    header += " ".join(f"{label[:18]:>20}" for _, label in METRICS)
    print(header)
    for algo in ALGO_ORDER:
        n = len(data[algo][METRICS[0][0]])
        cells = []
        for metric, _ in METRICS:
            arr = data[algo][metric]
            cells.append(f"mean={arr.mean():7.3f} max={arr.max():6.3f}")
        print(f"  {ALGO_LABELS[algo]:<11} {n:>4} " + " ".join(f"{c:>20}" for c in cells))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--copy-to-paper",
        action="store_true",
        help="Also copy the rendered PDF/PNG to BenchmarkPaper/materials/figures/.",
    )
    args = parser.parse_args()

    data = _collect()
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
