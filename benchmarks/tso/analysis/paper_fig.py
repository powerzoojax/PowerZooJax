"""Build Fig 5.3 (TSO cost--safety frontier) for the PowerZooJax paper.

Two side-by-side scatter panels at consistent body font size:
  (a) cost vs overload rate
  (b) cost vs reserve shortfall rate
Legends sit in a vertical column on the right.

Run:
    python benchmarks/tso/analysis/paper_fig.py
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
    apply_rc,
    save_paper_fig,
    split_marker,
    COLOR_UPPER,
)

FS = 12

# Same order/colours as benchmarks/gencos/analysis/paper_fig.py STRATEGIES.
_MAX_MARKUP_GENCOS = "#4f46e5"

FRONTIER_JSON = _PROJECT_ROOT / "benchmarks" / "tso" / "results" / "phase3_cost_safety_frontier.json"

ALGO_COLORS = {
    "ppo":            "#4f92a8",
    "ppo_lagrangian": COLOR_UPPER,
    "merit_order":    "#655770",
    "all_on":         _MAX_MARKUP_GENCOS,
}

ALGO_ORDER = [
    ("ppo",            "PPO"),
    ("ppo_lagrangian", "PPO-Lagrangian"),
    ("merit_order",    "Merit-order"),
    ("all_on",         "All-on"),
]
SPLIT_ORDER = [
    ("iid",             "In-distribution"),
    ("line_tightening", "Line-tightening"),
    ("load_stress",     "Load-stress"),
]


def _draw_panel(
    ax,
    splits_data,
    y_field: str,
    y_label: str,
    panel_title: str,
    *,
    ylim: tuple[float | None, float | None] | None = None,
    yticks: list[float] | np.ndarray | None = None,
):
    for algo, _ in ALGO_ORDER:
        col = ALGO_COLORS[algo]
        for split_key, _ in SPLIT_ORDER:
            sd = splits_data[split_key]
            point = next((p for p in sd["points"] if p["algo"] == algo), None)
            if point is None:
                continue
            x_mean = point["total_operating_cost"]["mean"] / 1e6
            x_lo = point["total_operating_cost"]["ci_lo"] / 1e6
            x_hi = point["total_operating_cost"]["ci_hi"] / 1e6
            y_mean = point[y_field]["mean"] * 100
            y_lo = point[y_field]["ci_lo"] * 100
            y_hi = point[y_field]["ci_hi"] * 100
            ax.errorbar(
                x_mean, y_mean,
                xerr=[[max(0.0, x_mean - x_lo)], [max(0.0, x_hi - x_mean)]],
                yerr=[[max(0.0, y_mean - y_lo)], [max(0.0, y_hi - y_mean)]],
                fmt=split_marker(split_key), color=col, ecolor=col,
                elinewidth=0.9, markersize=6, capsize=2,
                markeredgecolor="white", markeredgewidth=0.6,
            )
    ax.set_xlabel(r"Cost (£m/day)")
    ax.set_ylabel(y_label)
    # Long rotated y-label is clipped with tight_layout(rect left=0); shift down
    # and nudge outward slightly from the spine.
    ax.yaxis.set_label_coords(-0.14, 0.38)
    ax.set_xlim(1.2, 5.0)
    ax.set_title(panel_title, fontsize=FS, loc="center")
    ax.grid(True, alpha=0.25, linestyle="-", linewidth=0.4)
    if ylim is not None:
        lo, hi = ylim
        if lo is not None and hi is not None:
            ax.set_ylim(lo, hi)
        elif lo is not None:
            ax.set_ylim(bottom=lo)
        elif hi is not None:
            ax.set_ylim(top=hi)
    if yticks is not None:
        ax.set_yticks(yticks)


def main() -> None:
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
    splits_data = json.loads(FRONTIER_JSON.read_text())["splits"]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.0, 2.3))

    _draw_panel(
        ax_a,
        splits_data,
        "thermal_violation_rate",
        "Overload rate (%)",
        "(a) Cost vs thermal overload",
        ylim=(-5.0, 55.0),
        yticks=[0.0, 25.0, 50.0],
    )
    _draw_panel(
        ax_b,
        splits_data,
        "reserve_shortfall_rate",
        "Shortfall rate (%)",
        "(b) Cost vs reserve shortfall",
        ylim=(-1.5, None),
    )

    # Combined legend in a vertical column on the right of the two panels.
    algo_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=ALGO_COLORS[a],
                   markersize=6, label=lab,
                   markeredgecolor="white", markeredgewidth=0.6)
        for a, lab in ALGO_ORDER
    ]
    split_handles = [
        plt.Line2D([0], [0], marker=split_marker(k), color="none",
                   markerfacecolor="#374151",
                   markersize=6, linestyle="none", label=lab,
                   markeredgecolor="none")
        for k, lab in SPLIT_ORDER
    ]
    # Two figure-level legends. Do **not** call fig.add_artist on the first:
    # fig.legend() already puts it in fig.legends; add_artist also appends it to
    # fig.artists, so Matplotlib draws the same legend twice (text looks bold).
    # A second fig.legend() appends another entry to fig.legends (both stay).
    fig.legend(
        handles=algo_handles,
        title="Algorithm",
        loc="upper left",
        bbox_to_anchor=(0.83, 0.96),
        ncol=1,
        frameon=False,
        handletextpad=0.4,
        labelspacing=0.4,
        borderaxespad=0.0,
        alignment="left",
        fontsize=FS,
        title_fontsize=FS,
    )
    fig.legend(
        handles=split_handles,
        title="Split",
        loc="upper left",
        bbox_to_anchor=(0.83, 0.48),
        ncol=1,
        frameon=False,
        handletextpad=0.4,
        labelspacing=0.4,
        borderaxespad=0.0,
        alignment="left",
        fontsize=FS,
        title_fontsize=FS,
    )

    fig.tight_layout(rect=[0.05, 0, 0.82, 1])
    out = save_paper_fig(fig, "fig_tso")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
