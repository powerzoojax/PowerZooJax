"""GenCos benchmark figure generation.

Produces the standard summary figures plus paper-facing Phase-1/Phase-2 figures:
  1. normscore_bars.pdf   — NormScore per (algo, split): 0=truthful, 1=max_markup
  2. learning_curves.pdf — per-episode eval return + HHI vs timestep
  3. market_power.pdf    — HHI and price_volatility bars (algo × split)
  4. ood_robustness.pdf  — NormScore on iid vs OOD splits (avoids raw-dollar scale mixing)
  5. phase1_training_diagnostics.pdf
  6. phase1_episode_mechanism.pdf
  7. phase1_policy_compare.pdf
  8. phase1_iid_metric_distributions.pdf
  9. phase2_backend_compare.pdf

Usage:
    python benchmarks/gencos/plots.py
    python benchmarks/gencos/run_all.py --only plots
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

TASK_DIR = Path(__file__).resolve().parent

SPLIT_ORDER = ["train", "iid", "demand_shift", "renewable_shock"]
SPLIT_LABELS = {
    "train": "Train",
    "iid": "IID",
    "demand_shift": "Demand +10%",
    "renewable_shock": "RenShock +5%",
}

ALGO_ORDER = ["truthful", "uniform_mid", "max_markup", "ippo"]
ALGO_LABELS = {
    "truthful":    "Truthful",
    "uniform_mid": "Uniform Mid",
    "max_markup":  "Max Markup",
    "ippo":        "IPPO",
}
ALGO_COLORS = {
    "truthful":    "#4caf50",
    "uniform_mid": "#ff9800",
    "max_markup":  "#f44336",
    "ippo":        "#2196f3",
}


def _figures_dir(task_dir: Path) -> Path:
    return task_dir / "results" / "figures"


def _load_summary(task_dir: Path) -> Optional[dict]:
    path = task_dir / "results" / "summary" / "latest.json"
    if not path.exists():
        print(f"[GenCos plots] No summary at {path}. Run 'summarize' first.")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _get_row(rows, algo, split):
    return next((r for r in rows if r["algo"] == algo and r["split"] == split), None)


# ── Figure 1: NormScore grouped bar chart ────────────────────────────────────

def plot_normscore_bars(task_dir: Path = TASK_DIR) -> Path:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = _load_summary(task_dir)
    if summary is None:
        return Path()
    rows = summary["rows"]
    splits = [s for s in SPLIT_ORDER if any(r["split"] == s for r in rows)]
    algos  = [a for a in ALGO_ORDER  if any(r["algo"]  == a for r in rows)]
    if not algos:
        print("[GenCos plots] normscore_bars: no algos found, skipping.")
        return Path()

    fig, ax = plt.subplots(figsize=(9, 4))
    n_splits, n_algos = len(splits), len(algos)
    bar_w = 0.7 / n_algos
    x = np.arange(n_splits)

    for j, algo in enumerate(algos):
        vals = [
            (_get_row(rows, algo, s) or {}).get("norm_score") or 0.0
            for s in splits
        ]
        offset = (j - n_algos / 2 + 0.5) * bar_w
        ax.bar(x + offset, vals, bar_w,
               label=ALGO_LABELS.get(algo, algo),
               color=ALGO_COLORS.get(algo, "#999"))

    ax.axhline(0, color="black", lw=0.8, ls="--", label="Truthful (0)")
    ax.axhline(1, color="red",   lw=0.8, ls=":",  label="Max Markup (1)")
    ax.axhline(0.5, color="gray", lw=0.6, ls=":", alpha=0.6, label="Uniform Mid (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels([SPLIT_LABELS.get(s, s) for s in splits], rotation=20, ha="right")
    ax.set_ylabel("NormScore  (0 = truthful, 1 = max markup, >1 = collusion)")
    ax.set_title("GenCos — Strategic Bidding Level (NormScore)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()

    figures_dir = _figures_dir(task_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / "normscore_bars.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[GenCos plots] normscore_bars -> {out}")
    return out


# ── Figure 2: Learning curves (eval return + HHI vs. timestep) ───────────────

def plot_learning_curves(task_dir: Path = TASK_DIR) -> Path:
    """Plot IPPO training convergence: per-episode eval return and HHI vs timestep.

    Uses the ``learning_curve_eval_return`` artifact (true per-episode profit of
    the greedy deterministic policy) rather than the per-step training reward,
    which is harder to interpret out of context.
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from benchmarks.common.io import load_manifest

    records = load_manifest(task_dir)
    train_candidates = [
        r for r in records
        if r.split == "train" and r.algo == "ippo"
        and (r.artifacts or {}).get("params")
    ]
    latest_by_seed = {}
    for record in train_candidates:
        current = latest_by_seed.get(record.seed)
        if current is None or (record.timestamp, record.run_id) > (
            current.timestamp,
            current.run_id,
        ):
            latest_by_seed[record.seed] = record
    train_records = [latest_by_seed[s] for s in sorted(latest_by_seed)]
    if not train_records:
        print("[GenCos plots] No training records found; skipping learning_curves.")
        return Path()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    arts_dir = task_dir / "results" / "artifacts"

    eval_values_for_limits = []
    clipped_eval_points = 0

    def _curve_xs(xs: Optional[np.ndarray], n: int) -> np.ndarray:
        if xs is None or n <= 0:
            return np.arange(n, dtype=np.float64)
        if len(xs) == n:
            return xs
        xmax = float(xs[-1]) if len(xs) else 5.0
        return np.linspace(0.0, xmax, n, dtype=np.float64)

    for r in train_records:
        ts_path = arts_dir / f"{r.run_id}_timesteps.npy"
        xs = np.load(ts_path) / 1e6 if ts_path.exists() else None

        # ── Left panel: per-episode eval return (mean per agent, $/episode) ──
        ax0 = axes[0]
        eval_path = arts_dir / f"{r.run_id}_learning_curve_eval_return.npy"
        if eval_path.exists():
            arr = np.load(eval_path)
            eval_xs = _curve_xs(xs, len(arr))
            ax0.plot(eval_xs, arr, alpha=0.85,
                     label=f"s{r.seed}", color=ALGO_COLORS["ippo"])
            finite = arr[np.isfinite(arr)]
            if finite.size:
                eval_values_for_limits.append(finite)
        ax0.set_xlabel("Timesteps (M)")
        ax0.set_ylabel("Eval Return ($/episode/agent)")
        ax0.set_title("IPPO — Eval Return (mean/agent)")

        # ── Right panel: HHI vs timestep (market power development) ──
        ax1 = axes[1]
        for candidate in ["market_HHI", "market/HHI"]:
            p = arts_dir / f"{r.run_id}_{candidate}.npy"
            if p.exists():
                arr = np.load(p)
                t = _curve_xs(xs, len(arr))
                ax1.plot(t, arr, alpha=0.7,
                         label=f"s{r.seed}", color=ALGO_COLORS["ippo"])
                break
        ax1.set_xlabel("Timesteps (M)" if xs is not None else "Update")
        ax1.set_ylabel("HHI (market concentration)")
        ax1.set_title("IPPO — Market HHI during Training")

    if eval_values_for_limits:
        all_eval = np.concatenate(eval_values_for_limits)
        if all_eval.size >= 20:
            lo, hi = np.quantile(all_eval, [0.02, 0.98])
        else:
            lo, hi = float(np.min(all_eval)), float(np.max(all_eval))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            pad = 0.08 * (hi - lo)
            ymin, ymax = float(lo - pad), float(hi + pad)
            clipped_eval_points = int(np.sum((all_eval < ymin) | (all_eval > ymax)))
            ax0.set_ylim(ymin, ymax)
            if clipped_eval_points:
                ax0.text(
                    0.02,
                    0.04,
                    f"{clipped_eval_points} extreme eval point clipped",
                    transform=ax0.transAxes,
                    fontsize=7,
                    color="#555",
                )

    # Reference lines on HHI panel
    n_agents = 5  # GenCos case5
    ax1 = axes[1]
    ax1.axhline(1.0 / n_agents, color="green", lw=0.9, ls="--",
                label=f"Uniform dispatch (1/{n_agents}={1/n_agents:.2f})")
    ax1.axhline(1.0, color="red", lw=0.9, ls=":", label="Monopoly (1.0)")
    ax1.set_ylim(0, 1.05)

    for ax in axes:
        ax.legend(fontsize=7)
    fig.tight_layout()

    figures_dir = _figures_dir(task_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / "learning_curves.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[GenCos plots] learning_curves -> {out}")
    return out


# ── Figure 3: Market power metrics ───────────────────────────────────────────

def plot_market_power(task_dir: Path = TASK_DIR) -> Path:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = _load_summary(task_dir)
    if summary is None:
        return Path()
    rows = summary["rows"]
    splits = [s for s in SPLIT_ORDER if any(r["split"] == s for r in rows)]
    algos  = [a for a in ALGO_ORDER  if any(r["algo"]  == a for r in rows)]
    if not algos:
        print("[GenCos plots] market_power: no algos found, skipping.")
        return Path()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    n_splits, n_algos = len(splits), len(algos)
    bar_w = 0.7 / n_algos
    x = np.arange(n_splits)

    for ax, metric_key, ylabel, title, ref_lines in [
        (axes[0], "hhi_mean",             "HHI",
         "Market Concentration (HHI)",
         [(1.0 / 5, "green", "--", "Uniform dispatch (0.20)"),
          (1.0, "red", ":", "Monopoly (1.00)")]),
        (axes[1], "price_volatility_mean", "Price CV (σ/μ)",
         "LMP Price Volatility (CV)", []),
    ]:
        for j, algo in enumerate(algos):
            vals = [
                (_get_row(rows, algo, s) or {}).get(metric_key) or 0.0
                for s in splits
            ]
            offset = (j - n_algos / 2 + 0.5) * bar_w
            ax.bar(x + offset, vals, bar_w,
                   label=ALGO_LABELS.get(algo, algo),
                   color=ALGO_COLORS.get(algo, "#999"))
        for yval, color, ls, label in ref_lines:
            ax.axhline(yval, color=color, lw=0.9, ls=ls, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels([SPLIT_LABELS.get(s, s) for s in splits], rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)

    fig.tight_layout()
    figures_dir = _figures_dir(task_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / "market_power.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[GenCos plots] market_power -> {out}")
    return out


# ── Figure 4: OOD robustness — NormScore not raw profit ──────────────────────

def plot_ood_robustness(task_dir: Path = TASK_DIR) -> Path:
    """OOD robustness bar chart using NormScore instead of raw total profit.

    Raw total profit spans three orders of magnitude across baselines
    (truthful ≈ $7k vs max_markup ≈ $589k on IID), making bars unreadable.
    NormScore (0 = truthful, 1 = max markup) puts everything on the same scale
    and directly reflects the strategic positioning of each agent.
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = _load_summary(task_dir)
    if summary is None:
        return Path()
    rows = summary["rows"]
    eval_splits = ["iid", "demand_shift", "renewable_shock"]
    eval_splits = [s for s in eval_splits if any(r["split"] == s for r in rows)]
    algos = [a for a in ALGO_ORDER if any(r["algo"] == a for r in rows)]
    if not algos:
        print("[GenCos plots] ood_robustness: no algos found, skipping.")
        return Path()

    fig, ax = plt.subplots(figsize=(8, 4))
    n_splits, n_algos = len(eval_splits), len(algos)
    bar_w = 0.7 / n_algos
    x = np.arange(n_splits)

    for j, algo in enumerate(algos):
        vals = [
            (_get_row(rows, algo, s) or {}).get("norm_score") or 0.0
            for s in eval_splits
        ]
        offset = (j - n_algos / 2 + 0.5) * bar_w
        ax.bar(x + offset, vals, bar_w,
               label=ALGO_LABELS.get(algo, algo),
               color=ALGO_COLORS.get(algo, "#999"))

    ax.axhline(0, color="black", lw=0.8, ls="--", label="Truthful ref (0)")
    ax.axhline(1, color="red",   lw=0.8, ls=":",  label="Max Markup ref (1)")
    ax.axhline(0.5, color="gray", lw=0.6, ls=":", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([SPLIT_LABELS.get(s, s) for s in eval_splits], rotation=20, ha="right")
    ax.set_ylabel("NormScore  (0 = truthful, 1 = max markup)")
    ax.set_title("GenCos — OOD Robustness (NormScore under Distribution Shift)")
    ax.legend(fontsize=8)
    fig.tight_layout()

    figures_dir = _figures_dir(task_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / "ood_robustness.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[GenCos plots] ood_robustness -> {out}")
    return out


def generate_all_plots(task_dir: Path = TASK_DIR) -> None:
    plot_normscore_bars(task_dir)
    plot_learning_curves(task_dir)
    plot_market_power(task_dir)
    plot_ood_robustness(task_dir)
    try:
        from benchmarks.gencos.phase1_analysis import analyze_phase1_episode

        analyze_phase1_episode(task_dir)
    except Exception as exc:
        print(f"[GenCos plots] phase1_analysis skipped: {exc}")
    try:
        from benchmarks.gencos.phase2_analysis import generate_phase2_backend_compare

        generate_phase2_backend_compare(task_dir)
    except Exception as exc:
        print(f"[GenCos plots] phase2_analysis skipped: {exc}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", default=str(TASK_DIR), type=Path)
    args = parser.parse_args()
    generate_all_plots(args.task_dir)
