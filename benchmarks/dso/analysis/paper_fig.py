"""Build Fig 5.4 (DSO raw-metric distributions) for the PowerZooJax paper.

Three side-by-side panels showing per-episode IID metrics for the seven
DSO controllers (5 seeds × 50 episodes per controller):
  (a) network loss in MWh -- box plot (raw distribution)
  (b) mean voltage-violation count -- horizontal bar (most are zero)
  (c) percentage loss reduction relative to no-control -- box plot

Run:
    python benchmarks/dso/analysis/paper_fig.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks._paper_style import (  # noqa: E402
    apply_rc, figsize, save_paper_fig,
    COLOR_LEARNED, COLOR_SAFE, FONT_TITLE,
)


ARTIFACTS_DIR = _PROJECT_ROOT / "benchmarks" / "dso" / "results" / "artifacts"

# 7 controllers ordered weakest to strongest, with three grey shades for
# the three non-learning baselines so they are visually distinguishable.
CONTROLLERS = [
    ("no_control",     "No-ctrl",   "#e5e7eb"),
    ("tou",            "Peak-rule", "#cbd5e1"),
    ("droop",          "Droop",     "#9ca3af"),
    ("sac",            "SAC",       COLOR_LEARNED),
    ("saute_ppo",      "Sauté",     COLOR_SAFE),
    ("ppo_lagrangian", "Lag-PPO",   COLOR_SAFE),
    ("ppo",            "PPO",       COLOR_LEARNED),
]


def load_episodes(algo: str) -> dict[str, np.ndarray]:
    files = sorted(ARTIFACTS_DIR.glob(f"dso_{algo}_iid_s*_per_episode.json"))
    if not files:
        raise FileNotFoundError(f"no per_episode files for algo={algo}")
    loss_mwh, vviol, red_pct = [], [], []
    for fp in files:
        for r in json.loads(fp.read_text()):
            loss_mwh.append(float(r.get("total_loss_mwh", np.nan)))
            vviol.append(float(r.get("total_voltage_violations", 0.0)))
            red_pct.append(float(r.get("network_loss_reduction_pct", 0.0)))
    return {
        "loss_mwh":           np.asarray(loss_mwh),
        "voltage_violations": np.asarray(vviol),
        "loss_reduction_pct": np.asarray(red_pct),
    }


def _box(ax, data_list, labels, colors, ylabel, panel_title, dark_text=False):
    bp = ax.boxplot(
        data_list, positions=range(len(data_list)),
        widths=0.62, patch_artist=True, showfliers=False,
        medianprops=dict(color="white" if not dark_text else "#111827",
                         linewidth=1.4),
        whiskerprops=dict(color="#374151", linewidth=0.9),
        capprops=dict(color="#374151", linewidth=0.9),
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_edgecolor(c)
        patch.set_alpha(0.95)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right",
                       rotation_mode="anchor")
    ax.set_ylabel(ylabel)
    ax.set_title(panel_title, fontsize=FONT_TITLE, loc="left")
    ax.grid(True, axis="y", alpha=0.25, linestyle="-", linewidth=0.4)


def _bar(ax, values, labels, colors, ylabel, panel_title):
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, width=0.62, edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right",
                       rotation_mode="anchor")
    ax.set_ylabel(ylabel)
    ax.set_title(panel_title, fontsize=FONT_TITLE, loc="left")
    ax.grid(True, axis="y", alpha=0.25, linestyle="-", linewidth=0.4)


def main() -> None:
    apply_rc()
    data_by_algo = {a: load_episodes(a) for a, _, _ in CONTROLLERS}
    labels = [lab for _, lab, _ in CONTROLLERS]
    colors = [col for _, _, col in CONTROLLERS]

    loss_data  = [data_by_algo[a]["loss_mwh"]            for a, _, _ in CONTROLLERS]
    vviol_mean = [data_by_algo[a]["voltage_violations"].mean() for a, _, _ in CONTROLLERS]
    redp_data  = [data_by_algo[a]["loss_reduction_pct"]  for a, _, _ in CONTROLLERS]

    w, h = figsize(panels=3, total_width_frac=1.0, height=2.0)
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(w, h))

    _box(ax_a, loss_data, labels, colors,
         "Network loss (MWh)", "(a) Per-episode loss")
    _bar(ax_b, vviol_mean, labels, colors,
         "Mean violation count", "(b) Voltage violations")
    _box(ax_c, redp_data, labels, colors,
         "Loss reduction (%)", "(c) Loss reduction")

    fig.tight_layout()
    out = save_paper_fig(fig, "fig_dso")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
