"""Build Fig 5.4 (GenCos) for the PowerZooJax paper.

Two horizontal panels at consistent body font size:
  (a) Strategy dot chart: daily profit (linear y, £k) for each strategy
      on in-distribution, high demand, and low renewable. Splits are categorical
      and unrelated, so we draw independent dots (no connecting lines)
      with 95% CI vertical error bars at each (strategy, split).
  (b) LMP-vs-load response on the shared Phase-1 in-distribution episode.

Run:
    python benchmarks/gencos/analysis/paper_fig.py
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
    COLOR_UPPER,
)

FS = 12

SUMMARY_JSON = _PROJECT_ROOT / "benchmarks" / "gencos" / "results" / "summary" / "latest.json"
ARTIFACTS_DIR = _PROJECT_ROOT / "benchmarks" / "gencos" / "results" / "artifacts"
POLICY_COMPARE_LMP = ARTIFACTS_DIR / "policy_compare_lmp.npz"

# Max-markup: indigo-violet (not paper navy) so it separates from Truthful teal in hue
# and lightness; avoids dark green (red–green CVD confusion with IPPO).
_MAX_MARKUP_COLOR = "#4f46e5"

# Strategy palette: teal / mauve baselines; IPPO = paper red; Max-markup = indigo.
STRATEGIES = [
    ("truthful",    "Truthful",    "#4f92a8"),  # muted teal (lighter baseline)
    ("uniform_mid", "Uniform-mid", "#655770"),  # dusty mauve (mid baseline)
    ("ippo",        "IPPO",        COLOR_UPPER),
    ("max_markup",  "Max-markup",  _MAX_MARKUP_COLOR),
]

# Three eval splits, drawn left-to-right on the (a) panel's x-axis.
SPLITS = [
    ("iid",             "In-distribution"),
    ("demand_shift",    "High\ndemand"),
    ("renewable_shock", "Low\nrenewable"),
]

def _row(rows, algo: str, split: str = "iid"):
    for r in rows:
        if r["algo"] == algo and r["split"] == split:
            return r
    return None


def load_lmp_vs_load() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if not POLICY_COMPARE_LMP.exists():
        raise FileNotFoundError(
            f"missing {POLICY_COMPARE_LMP}; run "
            "`python benchmarks/gencos/phase1_analysis.py` to regenerate it"
        )
    with np.load(POLICY_COMPARE_LMP) as cache:
        curves = {}
        for algo, _, _ in STRATEGIES:
            load_key = f"{algo}_total_load_mw"
            lmp_key = f"{algo}_mean_lmp"
            if load_key not in cache or lmp_key not in cache:
                raise KeyError(
                    f"{POLICY_COMPARE_LMP} is missing {load_key!r} or {lmp_key!r}"
                )
            curves[algo] = (
                np.asarray(cache[load_key], dtype=np.float64),
                np.asarray(cache[lmp_key], dtype=np.float64),
            )
    return curves


def _bar_label(value_k: float) -> str:
    if value_k < 100:
        return f"{value_k:.0f}k"
    return f"{value_k:.0f}k"


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
    rows = json.loads(SUMMARY_JSON.read_text())["rows"]
    lmp_curves = load_lmp_vs_load()

    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(10.0, 2.3),
        gridspec_kw={"width_ratios": [1.0, 1.0]},
    )

    # ---------------- (a) Strategy dot chart ----------------
    # 12 independent dots (4 strategies x 3 splits). No connecting lines:
    # splits are categorical evaluation conditions, not a sequence, so a
    # line would imply a dynamic that doesn't exist. Within a split column
    # readers compare four colours; tracking one colour across columns
    # tells them whether the strategy is robust under OOD evaluation.
    n_split = len(SPLITS)
    n_strat = len(STRATEGIES)
    x_pos = np.arange(n_split, dtype=float)
    # Symmetric horizontal jitter so the four dots in each split column
    # don't sit on a perfectly vertical stack.
    jitter_step = 0.10
    jitter = (np.arange(n_strat) - (n_strat - 1) / 2) * jitter_step

    # Faint alternating background bands separate the three split columns,
    # which (in lieu of connecting lines) carries "three independent comparisons".
    for s_i in range(n_split):
        if s_i % 2 == 0:
            ax_a.axvspan(s_i - 0.5, s_i + 0.5, color="#f6f7f9", zorder=0)

    for s_idx, (algo, label, color) in enumerate(STRATEGIES):
        for s_i, (split_key, _) in enumerate(SPLITS):
            r = _row(rows, algo, split_key)
            if r is None:
                continue
            y = r["total_profit_mean"] / 1e3
            lo = (r["total_profit_mean"] - r["total_profit_ci_lo"]) / 1e3
            hi = (r["total_profit_ci_hi"] - r["total_profit_mean"]) / 1e3
            ax_a.errorbar(
                x_pos[s_i] + jitter[s_idx],
                y,
                yerr=[[lo], [hi]],
                fmt="o",
                color=color,
                markersize=7 if algo != "ippo" else 8,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.9,
                capsize=2.2,
                elinewidth=1.0,
                ecolor=color,
                zorder=4 if algo == "ippo" else 3,
            )

    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels([lab for _, lab in SPLITS])
    # Two-line split names vs (b)'s xlabel: modest pad to align baselines.
    ax_a.tick_params(axis="x", which="major", pad=8)
    ax_a.set_xlim(-0.5, n_split - 1 + 0.5)
    ax_a.set_ylabel(r"Daily profit (£k)")
    ax_a.set_ylim(-100, 800)
    # Sparse ticks so short figure height still shows all labels (0–500-only was too coarse).
    ax_a.set_yticks([0, 250, 500, 750])
    ax_a.set_title("(a) Profit per strategy and split", fontsize=FS, loc="center")
    ax_a.grid(True, axis="y", alpha=0.25, linestyle="-", linewidth=0.4)
    ax_a.set_axisbelow(True)

    # ---------------- (b) LMP vs load ----------------
    for algo, label, color in STRATEGIES:
        total_load, mean_lmp = lmp_curves[algo]
        order = np.argsort(total_load)
        x = total_load[order]
        y = mean_lmp[order]
        # IPPO is the protagonist of this plot -> slightly thicker line.
        lw = 2.0 if algo == "ippo" else 1.4
        ax_b.plot(x, y, color=color, lw=lw, zorder=3 if algo == "ippo" else 2)

    ax_b.set_xlabel("System load (MW)")
    ax_b.set_ylabel(r"LMP (£/MWh)")
    ax_b.set_xlim(380, 1080)
    ax_b.set_ylim(0, 85)
    ax_b.set_yticks([0, 25, 50, 75])
    ax_b.set_title("(b) LMP vs load (in-distribution episode)", fontsize=FS, loc="center")
    ax_b.grid(True, axis="both", alpha=0.25, linestyle="-", linewidth=0.4)
    ax_b.set_axisbelow(True)

    # Single Strategy legend on the right -- splits are already on the
    # x-axis of panel (a), so we no longer need a Split legend block.
    strategy_handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            color=color,
            linestyle="-",
            lw=1.6 if algo == "ippo" else 1.2,
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=8,
            label=label,
        )
        for algo, label, color in STRATEGIES
    ]
    fig.legend(
        handles=strategy_handles,
        title="Strategy",
        loc="center left",
        bbox_to_anchor=(0.83, 0.55),
        ncol=1,
        frameon=False,
        handletextpad=0.5,
        labelspacing=0.45,
        borderaxespad=0.0,
        alignment="left",
        fontsize=FS,
        title_fontsize=FS,
    )

    fig.tight_layout(rect=[0, 0, 0.82, 1])
    out = save_paper_fig(fig, "fig_gencos")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
