"""Build §6.5 (DERs cooperative MARL across stress splits) for the paper.

Two side-by-side grouped-bar panels with a shared y-axis (loss reduction %):
  (a) All controllers
  (b) IPPO with only one device class active
A right-side stacked legend gives "Controller" and "Active class" groups.

Run:
    python benchmarks/ders/analysis/paper_fig.py
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

from benchmarks.common.io import load_manifest  # noqa: E402
from benchmarks._paper_style import (  # noqa: E402
    apply_rc, figsize, save_paper_fig,
    COLOR_LEARNED, COLOR_SAFE, FONT_BODY,
)


TASK_DIR = _PROJECT_ROOT / "benchmarks" / "ders"
RESULTS_DIR = TASK_DIR / "results"
SUMMARY_DIR = RESULTS_DIR / "summary"

# Five controllers with explicit colours so the legend stays consistent.
CONTROLLERS = [
    ("no_control",      "No-control",    "#cbd5e1"),
    ("volt_droop",      "Volt/VAR droop", "#6b7280"),
    ("ippo",            "IPPO",           COLOR_LEARNED),
    ("ippo_safe",       "IPPO-rs",        COLOR_SAFE),
    ("ippo_lagrangian", "IPPO-Lag.",      "#fbbf24"),
]
SPLITS = [
    ("iid",                  "In-distribution"),
    ("pv_penetration_shift", "PV-shift"),
    ("load_stress",          "Load-stress"),
]
ABLATIONS = [
    ("ippo",           "All classes",  COLOR_LEARNED),
    ("ippo_bat_only",  "Batteries",    "#16a34a"),
    ("ippo_pv_only",   "PV inverters", "#ca8a04"),
    ("ippo_flex_only", "Flex loads",   "#db2777"),
]

_MANIFEST = None


def _manifest_records():
    global _MANIFEST
    if _MANIFEST is None:
        _MANIFEST = load_manifest(TASK_DIR)
    return _MANIFEST


def _metric_seed_values(algo: str, split: str, key: str) -> list[float]:
    """Per-seed mean values for the latest GPU/JAX per-episode records."""
    values: list[float] = []
    records = [
        r for r in _manifest_records()
        if r.algo == algo
        and r.split == split
        and r.status == "completed"
        and r.backend == "jax_rejax"
        and r.device == "gpu"
        and (r.artifacts or {}).get("per_episode")
    ]
    for rec in sorted(records, key=lambda r: r.seed):
        fp = RESULTS_DIR / rec.artifacts["per_episode"]
        rows = json.loads(fp.read_text())
        vals = [float(r.get(key, 0.0)) for r in rows]
        values.append(float(np.mean(vals)))
    return values


def load_metric(algo: str, split: str, key: str) -> tuple[float, float]:
    """Mean and std-error of `key` over seed-level per-episode records."""
    seed_means = np.asarray(_metric_seed_values(algo, split, key), dtype=float)
    if seed_means.size == 0:
        return float("nan"), float("nan")
    return float(seed_means.mean()), float(seed_means.std(ddof=0) / np.sqrt(max(1, len(seed_means))))


def _summary_rows() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for algo, label, _ in ABLATIONS:
        for split, split_label in SPLITS:
            vals = np.asarray(_metric_seed_values(algo, split, "loss_reduction_pct"))
            if vals.size == 0:
                continue
            rows.append({
                "algo": algo,
                "label": label,
                "split": split,
                "split_label": split_label,
                "n_seeds": int(vals.size),
                "loss_reduction_pct_mean": float(vals.mean()),
                "loss_reduction_pct_se": float(vals.std(ddof=0) / np.sqrt(vals.size)),
            })
    return rows


def main() -> None:
    apply_rc()

    n_splits = len(SPLITS)
    group_centers = np.arange(n_splits)

    w, h = figsize(panels=2, total_width_frac=1.0, height=1.75)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(w, h), sharey=True)

    # Panel (a): all five controllers across the three splits.
    bar_w_a = 0.14
    n_a = len(CONTROLLERS)
    for i, (algo, _, col) in enumerate(CONTROLLERS):
        means, errs = [], []
        for split, _ in SPLITS:
            m, e = load_metric(algo, split, "loss_reduction_pct")
            means.append(m)
            errs.append(e)
        offset = (i - (n_a - 1) / 2) * bar_w_a
        ax_a.bar(group_centers + offset, means, width=bar_w_a * 0.95,
                 color=col, edgecolor="none",
                 yerr=errs, ecolor="#374151",
                 error_kw={"elinewidth": 0.7, "capsize": 1.5})

    ax_a.set_xticks(group_centers)
    ax_a.set_xticklabels([lab for _, lab in SPLITS],
                         rotation=20, ha="right",
                         rotation_mode="anchor")
    ax_a.set_ylabel("Loss reduction (%)")
    ax_a.set_title("(a) All controllers", fontsize=FONT_BODY, loc="left")
    ax_a.grid(True, axis="y", alpha=0.25, linestyle="-", linewidth=0.4)
    ax_a.axhline(0, color="#9ca3af", linewidth=0.6)

    # Panel (b): IPPO with only one device class active.
    bar_w_b = 0.17
    n_b = len(ABLATIONS)
    for i, (algo, _, col) in enumerate(ABLATIONS):
        means, errs = [], []
        for split, _ in SPLITS:
            m, e = load_metric(algo, split, "loss_reduction_pct")
            means.append(m)
            errs.append(e)
        offset = (i - (n_b - 1) / 2) * bar_w_b
        ax_b.bar(group_centers + offset, means, width=bar_w_b * 0.95,
                 color=col, edgecolor="none",
                 yerr=errs, ecolor="#374151",
                 error_kw={"elinewidth": 0.7, "capsize": 1.5})

    ax_b.set_xticks(group_centers)
    ax_b.set_xticklabels([lab for _, lab in SPLITS],
                         rotation=20, ha="right",
                         rotation_mode="anchor")
    ax_b.set_title("(b) Only one class active",
                   fontsize=FONT_BODY, loc="left")
    ax_b.grid(True, axis="y", alpha=0.25, linestyle="-", linewidth=0.4)
    ax_b.axhline(0, color="#9ca3af", linewidth=0.6)

    # Right-side stacked legends: "Controller" (panel a) and "Active class"
    # (panel b). Match the TSO/GenCos figures.
    ctrl_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=col, edgecolor="none", label=lab)
        for _, lab, col in CONTROLLERS
    ]
    class_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=col, edgecolor="none", label=lab)
        for _, lab, col in ABLATIONS
    ]
    legend_font = FONT_BODY
    legend_kwargs = dict(
        ncol=1,
        frameon=False,
        handlelength=1.25,
        handleheight=0.55,
        handletextpad=0.25,
        labelspacing=0.08,
        columnspacing=0.5,
        borderaxespad=0.0,
        alignment="left",
        fontsize=legend_font,
        title_fontsize=legend_font,
    )
    leg_ctrl = fig.legend(handles=ctrl_handles, title="Controller",
                          loc="upper left", bbox_to_anchor=(0.78, 0.97),
                          **legend_kwargs)
    fig.add_artist(leg_ctrl)
    fig.legend(handles=class_handles, title="Active class",
               loc="upper left", bbox_to_anchor=(0.78, 0.47),
               **legend_kwargs)

    fig.tight_layout(rect=[0, 0, 0.78, 1])
    out = save_paper_fig(fig, "fig_ders")
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = SUMMARY_DIR / "paper_fig_ders_b2.json"
    summary_path.write_text(json.dumps(_summary_rows(), indent=2), encoding="utf-8")
    print(f"saved: {out}")
    print(f"saved: {summary_path}")


if __name__ == "__main__":
    main()
