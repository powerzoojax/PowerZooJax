"""Shared matplotlib style for the PowerZooJax paper figures (§5).

All five task figures (`paper_fig.py` under each benchmark) import the
constants and helpers defined here so that colour, marker, font size, and
figure aspect ratio are coordinated across the paper.

Constraints baked in:
- Each figure renders at no more than ~2.0 inches tall when included at
  0.85--0.95 \\textwidth in the LaTeX source.
- One algorithm has the same colour everywhere; one split has the same
  marker shape everywhere.
- No 3-D, no coloured backgrounds, faint y-grid only.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------
# Output location
# -----------------------------------------------------------------------

PAPER_FIG_DIR = (
    Path(__file__).resolve().parent.parent
    / "BenchmarkPaper"
    / "materials"
    / "figures"
    / "paper"
)


def ensure_outdir() -> Path:
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    return PAPER_FIG_DIR


# -----------------------------------------------------------------------
# Colours: keyed by algorithm class. Each algorithm gets the same colour
# in every figure across the paper.
# -----------------------------------------------------------------------

# Class-level palette
COLOR_LEARNED = "#1f4e79"      # PPO, IPPO, SAC -- learned policies
COLOR_SAFE = "#d97706"         # safe-RL variants (Lagrangian, Saute, penalty)
COLOR_BASELINE = "#6b7280"     # rule-based, no-control, droop, TOU, all-on, merit-order
COLOR_UPPER = "#dc2626"        # oracle / upper-bound reference (max-markup, etc.)
COLOR_HIGHLIGHT = "#0ea5e9"    # secondary highlight (stress overlay, etc.)

# Algorithm-specific lookup. If an algorithm is not listed here the caller
# falls back to one of the class-level colours above.
ALGO_COLOR = {
    # learned, single-agent
    "ppo":             COLOR_LEARNED,
    "sac":             COLOR_LEARNED,
    # learned, multi-agent
    "ippo":            COLOR_LEARNED,
    "ippo_typed":      COLOR_LEARNED,
    # safe-RL family
    "ppo_lagrangian":  COLOR_SAFE,
    "ppo_lag":         COLOR_SAFE,
    "saute_ppo":       COLOR_SAFE,
    "sauté_ppo":       COLOR_SAFE,
    "penalty_ppo":     COLOR_SAFE,
    "ippo_safe":       COLOR_SAFE,
    "ippo_lagrangian": COLOR_SAFE,
    # non-learning baselines
    "no_control":      COLOR_BASELINE,
    "no-control":      COLOR_BASELINE,
    "tou":             COLOR_BASELINE,
    "time_of_use":     COLOR_BASELINE,
    "droop":           COLOR_BASELINE,
    "voltage_droop":   COLOR_BASELINE,
    "volt_var":        COLOR_BASELINE,
    "all_on":          COLOR_BASELINE,
    "merit_order":     COLOR_BASELINE,
    "max_renewable":   COLOR_BASELINE,
    "rule_based":      COLOR_BASELINE,
    "truthful":        COLOR_BASELINE,
    "uniform_mid":     COLOR_BASELINE,
    # upper bound
    "max_markup":      COLOR_UPPER,
}


def algo_color(name: str) -> str:
    """Look up the canonical paper colour for an algorithm name.

    Falls back to a neutral baseline grey for unknown names.
    """
    key = name.lower().replace("-", "_").replace(" ", "_")
    return ALGO_COLOR.get(key, COLOR_BASELINE)


# -----------------------------------------------------------------------
# Markers: keyed by data split. Each split has the same marker shape
# in every figure across the paper.
# -----------------------------------------------------------------------

SPLIT_MARKER = {
    "iid":                  "o",
    "demand_shift":         "D",
    "renewable_shock":      "D",
    "line_tightening":      "^",
    "load_stress":          "s",
    "voltage_tightening":   "^",
    "pv_penetration_shift": "D",
    "cooling_stress":       "D",
    "renewable_drought":    "s",
    "workload_swap":        "v",
    "workload_shock":       "v",
    "dg_derating":          "P",
    "sla_tighten":          "X",
}


def split_marker(name: str) -> str:
    return SPLIT_MARKER.get(name.lower(), "o")


# -----------------------------------------------------------------------
# Default rcParams: small, readable, paper-grade.
# -----------------------------------------------------------------------

FONT_BODY = 10      # all text inside figures (non-main-text figures; main-text unchanged)
FONT_TITLE = 10     # panel titles


def apply_rc():
    mpl.rcParams.update({
        "font.family":            "sans-serif",
        "font.sans-serif":        ["Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset":       "stix",
        "font.size":              FONT_BODY,
        "axes.titlesize":         FONT_TITLE,
        "axes.labelsize":         FONT_BODY,
        "xtick.labelsize":        FONT_BODY,
        "ytick.labelsize":        FONT_BODY,
        "legend.fontsize":        FONT_BODY,
        "axes.spines.top":        False,
        "axes.spines.right":      False,
        "axes.grid":              True,
        "axes.grid.axis":         "y",
        "grid.alpha":             0.25,
        "grid.linestyle":         "-",
        "grid.linewidth":         0.4,
        "lines.linewidth":        1.4,
        "lines.markersize":       5,
        "savefig.bbox":           "tight",
        "savefig.pad_inches":     0.05,
        "pdf.fonttype":           42,   # editable text in vector outputs
        "ps.fonttype":            42,
    })


# -----------------------------------------------------------------------
# Figure-size presets.  Width is the rendered width in inches when the
# PDF is included at the corresponding fraction of \textwidth.
# Height is capped at 2.0 in everywhere.
# -----------------------------------------------------------------------

# Non-main-text appendix figures save at ~10 in width (see paper style plan).
TEXTWIDTH_IN = 10.0

def figsize(panels: int, *, total_width_frac: float = 0.95, height: float = 1.9):
    """Return (width, height) in inches for a one-row figure.

    `panels` controls the conventional layout (1, 2, or 3 panels in a row),
    `total_width_frac` is the LaTeX `width=X\textwidth` we plan to use.
    """
    w = TEXTWIDTH_IN * total_width_frac
    return (w, height)


# -----------------------------------------------------------------------
# Convenience for saving
# -----------------------------------------------------------------------

def save_paper_fig(fig, name: str, *, dpi: int = 300):
    """Save `fig` to BenchmarkPaper/materials/figures/paper/<name>.pdf and
    a same-name PNG sidecar for quick inspection.
    """
    outdir = ensure_outdir()
    pdf = outdir / f"{name}.pdf"
    png = outdir / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=dpi)
    return pdf
