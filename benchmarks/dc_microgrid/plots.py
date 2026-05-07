"""DC Microgrid benchmark figure generation.

Reads summary JSON and learning-curve .npy artifacts to produce:
  1. normscore_bars.pdf      -- NormScore per (algo, split) with 95% CI
  2. reward_curves.pdf       -- NormScore vs timestep, seed mean + CI band
  3. cost_decomposition.pdf  -- Energy / fuel / carbon stacked bars per algo
  4. ood_robustness.pdf      -- Per-OOD-scenario NormScore with CI

Usage:
    python benchmarks/dc_microgrid/plots.py [--task-dir benchmarks/dc_microgrid]
    python benchmarks/dc_microgrid/run.py plots
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

TASK_DIR = Path(__file__).resolve().parent

SPLIT_ORDER = [
    "train",
    "iid",
    "cooling_stress",
    "renewable_drought",
    "workload_swap",
    "workload_shock",
    "dg_derating",
    "sla_tighten",
]
SPLIT_LABELS = {
    "train": "Train",
    "iid": "IID",
    "cooling_stress": "Cooling Stress",
    "renewable_drought": "Renew. Drought",
    "workload_swap": "WL Swap",
    "workload_shock": "WL Shock",
    "dg_derating": "DG Derate",
    "sla_tighten": "SLA Tight",
}

ALGO_ORDER = ["no_control", "max_renewable", "rule_based", "ppo", "sac"]
ALGO_LABELS = {
    "no_control": "No Control",
    "max_renewable": "Max Renewable",
    "rule_based": "Rule-Based",
    "ppo": "PPO (shaped)",
    "sac": "SAC (shaped)",
}

ALGO_COLORS = {
    "no_control": "#9e9e9e",
    "max_renewable": "#8bc34a",
    "rule_based": "#4caf50",
    "ppo": "#1976d2",
    "sbx_ppo": "#e91e63",
    "sac": "#ff6f00",
}

# Fair-comparison: same algo, different backend/device → same base color,
# different linestyle so curves stay visually distinct within one panel.
BACKEND_DEVICE_LINESTYLES: dict[tuple[str, str], str] = {
    ("jax_rejax", "gpu"): "-",
    ("jax_rejax", "cpu"): "--",
    ("sb3", "cuda"): "-.",
    ("sb3", "cpu"): ":",
    ("sbx", "gpu"): (0, (3, 1, 1, 1)),   # type: ignore[dict-item]
    ("sbx", "cpu"): (0, (5, 2, 1, 2, 1, 2)),  # type: ignore[dict-item]
}

COST_COLORS = {
    "energy": "#2196f3",
    "fuel": "#ff9800",
    "carbon": "#9c27b0",
}


def _figures_dir(task_dir: Path) -> Path:
    return task_dir / "results" / "figures"


def _load_summary(task_dir: Path) -> Optional[dict]:
    path = task_dir / "results" / "summary" / "latest.json"
    if not path.exists():
        print(f"[DC Microgrid plots] No summary found at {path}. Run 'summarize' first.")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _get_row(rows: list[dict], algo: str, split: str) -> Optional[dict]:
    for r in rows:
        if r["algo"] == algo and r["split"] == split:
            return r
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: NormScore grouped bar chart (with 95% CI error bars)
# ─────────────────────────────────────────────────────────────────────────────

def plot_normscore_bars(task_dir: Path = TASK_DIR) -> Path:
    """Grouped bar chart: NormScore per (algo, split) with 95% bootstrap CI."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = _load_summary(task_dir)
    if summary is None:
        return Path()

    rows = summary["rows"]
    figures_dir = _figures_dir(task_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    algos = [a for a in ALGO_ORDER if any(r["algo"] == a for r in rows)]
    splits = [s for s in SPLIT_ORDER if any(r["split"] == s for r in rows)]

    n_algos = len(algos)
    n_splits = len(splits)
    x = np.arange(n_splits)
    width = 0.8 / max(n_algos, 1)

    fig, ax = plt.subplots(figsize=(max(10, n_splits * 1.5), 5))

    for i, algo in enumerate(algos):
        scores = []
        ci_lo = []
        ci_hi = []
        for split in splits:
            r = _get_row(rows, algo, split)
            ns = r.get("norm_score") if r else None
            scores.append(float(ns) if ns is not None else 0.0)
            lo = r.get("norm_score_ci_lo") if r else None
            hi = r.get("norm_score_ci_hi") if r else None
            ci_lo.append(float(lo) if lo is not None else float(ns) if ns is not None else 0.0)
            ci_hi.append(float(hi) if hi is not None else float(ns) if ns is not None else 0.0)

        scores_a = np.array(scores)
        err_low = np.clip(scores_a - np.array(ci_lo), 0, None)
        err_high = np.clip(np.array(ci_hi) - scores_a, 0, None)
        offset = (i - n_algos / 2 + 0.5) * width
        ax.bar(
            x + offset,
            scores,
            width=width * 0.9,
            label=ALGO_LABELS.get(algo, algo),
            color=ALGO_COLORS.get(algo, "#888888"),
            alpha=0.85,
            yerr=[err_low, err_high],
            capsize=3,
            error_kw={"linewidth": 0.8, "alpha": 0.7},
        )

    ax.axhline(0, color="#9e9e9e", linewidth=0.8, linestyle="--")
    ax.axhline(1, color="#4caf50", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([SPLIT_LABELS.get(s, s) for s in splits], rotation=20, ha="right")
    ax.set_ylabel("NormScore (0 = No-Control, 1 = Rule-Based)")
    ax.set_title("DC Microgrid — NormScore by Algorithm and Split (95% CI)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    out = figures_dir / "normscore_bars.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[DC Microgrid plots] saved {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: RL learning curves — eval return vs env steps (and wall time)
# ─────────────────────────────────────────────────────────────────────────────

def _load_walltime_array(artifacts_dir: Path, run_id: str) -> Optional[np.ndarray]:
    for name in (
        f"{run_id}_learning_curve_eval_walltimes.npy",
        f"{run_id}_eval_wall_time_s.npy",
    ):
        p = artifacts_dir / name
        if p.exists():
            try:
                w = np.load(p).astype(np.float64).ravel()
                if w.size:
                    return w
            except OSError:
                pass
    return None


def _load_curve_array(task_dir: Path, rec: dict) -> Optional[np.ndarray]:
    run_id = rec.get("run_id", "")
    if not run_id:
        return None
    artifacts_dir = task_dir / "results" / "artifacts"
    artifacts = rec.get("artifacts") or {}
    candidates = []
    for key in ("learning_curve_eval_return", "eval_returns", "learning_curve_train_return"):
        rel = artifacts.get(key)
        if rel:
            candidates.append(task_dir / "results" / rel)
    candidates.extend(
        [
            artifacts_dir / f"{run_id}_learning_curve_eval_return.npy",
            artifacts_dir / f"{run_id}_eval_returns.npy",
            artifacts_dir / f"{run_id}_learning_curve_train_return.npy",
        ]
    )
    for path in candidates:
        if not path.exists():
            continue
        try:
            arr = np.load(path).astype(np.float64).ravel()
        except OSError:
            continue
        if arr.size > 0:
            return arr
    return None


def _load_total_timesteps(task_dir: Path, rec: dict) -> Optional[int]:
    run_id = rec.get("run_id", "")
    if not run_id:
        return None
    artifacts = rec.get("artifacts") or {}
    cfg_path = None
    if artifacts.get("config"):
        cfg_path = task_dir / "results" / artifacts["config"]
    else:
        cfg_path = task_dir / "results" / "artifacts" / f"{run_id}_config.json"
    if cfg_path is None or not cfg_path.exists():
        return None
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    candidates = (
        ("train_config", "total_timesteps"),
        ("train_config_raw", "total_timesteps"),
        ("train_config_resolved", "total_timesteps"),
        ("powerzoo_driver_config", "total_timesteps"),
    )
    for outer, inner in candidates:
        block = cfg.get(outer)
        if isinstance(block, dict) and block.get(inner) is not None:
            try:
                return int(block[inner])
            except (TypeError, ValueError):
                continue
    if cfg.get("total_timesteps") is not None:
        try:
            return int(cfg["total_timesteps"])
        except (TypeError, ValueError):
            return None
    return None


def _curve_record_has_learning_trace(rec: dict) -> bool:
    artifacts = rec.get("artifacts") or {}
    return bool(
        artifacts.get("params")
        or artifacts.get("params_orbax")
        or artifacts.get("learning_curve_train_return")
        or artifacts.get("learning_curve_eval_return")
    )


def _train_runs_by_backend_device(
    task_dir: Path,
) -> dict[tuple[str, str, str], list[dict]]:
    """Return curve-backed records grouped by (backend, device, algo).

    Reads the manifest rather than scanning the filesystem so that
    cross-backend / cross-device records for the same (algo, seed) are all
    preserved. Rejax training runs contribute params-backed eval curves, while
    PowerZoo bridge runs contribute train-return curves without params.

    Within each (backend, device, algo, seed) cell the newest curve-backed
    record (by timestamp) wins, then results are regrouped by
    (backend, device, algo) for per-group seed aggregation.
    """
    manifest_path = task_dir / "results" / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        records = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    # Per (backend, device, algo, seed): keep the latest curve-backed run.
    best: dict[tuple[str, str, str, int], dict] = {}
    for rec in records:
        _arts = rec.get("artifacts") or {}
        if not _curve_record_has_learning_trace(rec):
            continue
        backend = rec.get("backend") or "jax_rejax"
        device = rec.get("device") or "gpu"
        algo = rec.get("algo") or "ppo"
        seed = int(rec.get("seed", -1))
        cell_key = (backend, device, algo, seed)
        cur = best.get(cell_key)
        if cur is None or rec.get("timestamp", "") > cur.get("timestamp", ""):
            best[cell_key] = rec

    # Regroup by (backend, device, algo) for cross-seed aggregation.
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for (backend, device, algo, _seed), rec in best.items():
        groups.setdefault((backend, device, algo), []).append(rec)
    return groups


def plot_reward_curves(task_dir: Path = TASK_DIR) -> Path:
    """Learning curves: return vs env timesteps and vs wall time.

    Reads curve-backed records from the manifest and groups them by
    (backend, device, algo). Each group yields one curve: seed mean ± 95% CI
    band. Multiple PPO variants (jax+GPU, jax+CPU, sb3+CUDA, sb3+CPU …)
    appear as separate curves distinguished by linestyle, sharing the algo's
    base color.

    Wall-time panel prefers checkpoint walltimes when available. Older
    cross-backend records without checkpoint walltimes fall back to a linear
    interpolation to the recorded total wall time.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    figures_dir = _figures_dir(task_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = task_dir / "results" / "artifacts"

    groups = _train_runs_by_backend_device(task_dir)
    if not groups:
        print(
            "[DC Microgrid plots] No training records found in manifest. "
            "Run training and save_run first, or check "
            f"{task_dir / 'results' / 'manifest.json'}."
        )
        return Path()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax_steps, ax_wall = axes
    found_any = False
    has_wall = False
    _unk_color_i = 0

    for group_key in sorted(groups.keys()):
        backend, device, algo = group_key
        color = ALGO_COLORS.get(algo, None)
        if color is None:
            color = plt.cm.tab10(_unk_color_i % 10)  # type: ignore[union-attr]
            _unk_color_i += 1
        linestyle = BACKEND_DEVICE_LINESTYLES.get((backend, device), "-")

        series_returns: list[np.ndarray] = []
        series_steps: list[np.ndarray] = []
        series_wall: list[np.ndarray] = []
        series_ep_std: list[np.ndarray] = []

        for rec in groups[group_key]:
            run_id = rec.get("run_id", "")
            if not run_id or not artifacts_dir.exists():
                continue
            y = _load_curve_array(task_dir, rec)
            if y is None:
                continue
            step_path = artifacts_dir / f"{run_id}_timesteps.npy"
            if step_path.exists():
                x_steps = np.load(step_path).astype(np.float64).ravel()
            else:
                total_timesteps = _load_total_timesteps(task_dir, rec)
                if total_timesteps is not None and len(y) > 0:
                    x_steps = np.linspace(0, total_timesteps, len(y), dtype=np.float64)
                else:
                    x_steps = np.arange(len(y), dtype=np.float64)
            n = min(len(x_steps), len(y))
            if n <= 0:
                continue
            x_steps = x_steps[:n]
            y = y[:n]
            series_returns.append(y)
            series_steps.append(x_steps)
            w = _load_walltime_array(artifacts_dir, run_id)
            if w is not None and w.size >= n:
                series_wall.append(w[:n].copy())
            elif rec.get("walltime_s") is not None:
                walltime = float(rec["walltime_s"])
                series_wall.append(np.linspace(0, walltime, n, dtype=np.float64))
            else:
                series_wall.append(np.array([]))
            er_path = artifacts_dir / f"{run_id}_eval_episode_returns.npy"
            ep_std = None
            if er_path.exists():
                try:
                    er = np.load(er_path)
                    if er.ndim == 2 and er.shape[0] >= n:
                        ep_std = np.std(er[:n, :], axis=1)
                except OSError:
                    ep_std = None
            series_ep_std.append(ep_std if ep_std is not None else np.zeros(n))

        if not series_returns:
            continue

        min_len = min(len(a) for a in series_returns)
        R = np.stack([a[:min_len] for a in series_returns], axis=0)
        Xs = np.stack([a[:min_len] for a in series_steps], axis=0)
        x_mean = np.mean(Xs, axis=0)
        y_mean = np.mean(R, axis=0)
        if R.shape[0] >= 2:
            lo = np.percentile(R, 2.5, axis=0)
            hi = np.percentile(R, 97.5, axis=0)
        else:
            lo, hi = y_mean, y_mean

        algo_label = ALGO_LABELS.get(algo, algo.replace("_", " ").title())
        n_seeds = R.shape[0]
        label = (
            f"{algo_label} ({backend}+{device})"
            f" (n={n_seeds} seed{'s' if n_seeds != 1 else ''})"
        )

        ax_steps.fill_between(
            x_mean, lo, hi, color=color, alpha=0.18, linewidth=0, zorder=1,
        )
        ax_steps.plot(
            x_mean, y_mean, color=color, linestyle=linestyle,
            linewidth=2.0, label=label, zorder=3,
        )

        if min_len and np.any(series_ep_std[0] > 0) and R.shape[0] == 1:
            s0 = series_ep_std[0][:min_len]
            ax_steps.fill_between(
                x_mean, y_mean - 1.645 * s0, y_mean + 1.645 * s0,
                color=color, alpha=0.12, linewidth=0, zorder=2,
            )

        # Wall time panel: runs without wall timestamps are excluded gracefully.
        wall_idx = [i for i, w in enumerate(series_wall) if w.size >= min_len]
        if wall_idx:
            R_w = R[wall_idx, :]
            W = np.stack([series_wall[i][:min_len] for i in wall_idx], axis=0)
            xw = np.mean(W, axis=0)
            yw_wall = np.mean(R_w, axis=0)
            if R_w.shape[0] >= 2:
                wlo = np.percentile(R_w, 2.5, axis=0)
                whi = np.percentile(R_w, 97.5, axis=0)
            else:
                wlo, whi = yw_wall, yw_wall
            n_w = R_w.shape[0]
            label_w = (
                f"{algo_label} ({backend}+{device})"
                f" (wall: {n_w}/{R.shape[0]} seeds)"
            )
            ax_wall.fill_between(
                xw, wlo, whi, color=color, alpha=0.18, linewidth=0, zorder=1,
            )
            ax_wall.plot(
                xw, yw_wall, color=color, linestyle=linestyle,
                linewidth=2.0, label=label_w, zorder=3,
            )
            has_wall = True

        found_any = True

    if not found_any:
        print("[DC Microgrid plots] Could not load any eval curves.")
        plt.close(fig)
        return Path()

    def _fmt_millions(x, _pos):
        if x >= 1e6:
            return f"{x/1e6:.1f}M"
        if x >= 1e3:
            return f"{x/1e3:.0f}k"
        return f"{x:.0f}"

    ax_steps.xaxis.set_major_formatter(FuncFormatter(_fmt_millions))
    ax_steps.set_xlabel("Environment timesteps")
    ax_steps.set_ylabel("Return (shaped reward)")
    ax_steps.set_title("DC Microgrid — convergence (vs steps)")
    ax_steps.spines["top"].set_visible(False)
    ax_steps.spines["right"].set_visible(False)
    ax_steps.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.95)
    ax_steps.grid(True, alpha=0.28, linewidth=0.6, linestyle="--")

    if has_wall:
        ax_wall.set_xlabel("Cumulative training wall time at eval (s, host)")
        ax_wall.set_ylabel(ax_steps.get_ylabel())
        ax_wall.set_title("Eval return vs wall time (cross-backend/device comparison)")
        ax_wall.spines["top"].set_visible(False)
        ax_wall.spines["right"].set_visible(False)
        ax_wall.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.95)
        ax_wall.grid(True, alpha=0.28, linewidth=0.6, linestyle="--")
    else:
        ax_wall.set_visible(False)
        fig.set_size_inches(7.5, 4.2)

    fig.tight_layout()
    out = figures_dir / "reward_curves.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[DC Microgrid plots] saved {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: Cost decomposition stacked bar (train split)
# ─────────────────────────────────────────────────────────────────────────────

def plot_cost_decomposition(task_dir: Path = TASK_DIR) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = _load_summary(task_dir)
    if summary is None:
        return Path()

    rows = summary["rows"]
    figures_dir = _figures_dir(task_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    algos = [a for a in ALGO_ORDER if _get_row(rows, a, "train") is not None]
    energy_costs = []
    fuel_costs = []
    carbon_costs = []

    for algo in algos:
        r = _get_row(rows, algo, "train")
        energy_costs.append(r.get("total_energy_cost_mean") or 0.0)
        fuel_costs.append(r.get("total_fuel_cost_mean") or 0.0)
        carbon_costs.append((r.get("total_carbon_kg_mean") or 0.0) / 1000.0)

    x = np.arange(len(algos))
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        x, energy_costs, label="Energy [MWh]",
        color=COST_COLORS["energy"], alpha=0.85,
    )
    ax.bar(
        x, fuel_costs, bottom=energy_costs, label="Fuel Cost [$]",
        color=COST_COLORS["fuel"], alpha=0.85,
    )
    ax.bar(
        x, carbon_costs,
        bottom=[e + f for e, f in zip(energy_costs, fuel_costs)],
        label="Carbon [tCO2]",
        color=COST_COLORS["carbon"], alpha=0.85,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([ALGO_LABELS.get(a, a) for a in algos])
    ax.set_ylabel("Cost (Train Split)")
    ax.set_title("DC Microgrid — Cost Decomposition (Train Split)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    out = figures_dir / "cost_decomposition.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[DC Microgrid plots] saved {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: OOD robustness comparison (all splits with CI)
# ─────────────────────────────────────────────────────────────────────────────

def plot_ood_robustness(task_dir: Path = TASK_DIR) -> Path:
    """Per-split NormScore comparison for PPO, across all OOD scenarios."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = _load_summary(task_dir)
    if summary is None:
        return Path()

    rows = summary["rows"]
    figures_dir = _figures_dir(task_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rl_algos = [a for a in ["ppo", "sac"] if any(r["algo"] == a for r in rows)]
    ood_splits = [s for s in SPLIT_ORDER if s != "train" and any(r["split"] == s for r in rows)]

    if not rl_algos or not ood_splits:
        print("[DC Microgrid plots] Not enough RL results for OOD robustness plot.")
        return Path()

    n_algos = len(rl_algos)
    n_splits = len(ood_splits)
    x = np.arange(n_splits)
    width = 0.7 / max(n_algos, 1)

    fig, ax = plt.subplots(figsize=(max(8, n_splits * 1.2), 5))

    for i, algo in enumerate(rl_algos):
        scores = []
        ci_lo = []
        ci_hi = []
        for split in ood_splits:
            r = _get_row(rows, algo, split)
            ns = r.get("norm_score") if r else None
            scores.append(float(ns) if ns is not None else float("nan"))
            lo = r.get("norm_score_ci_lo") if r else None
            hi = r.get("norm_score_ci_hi") if r else None
            ci_lo.append(float(lo) if lo is not None else float(ns) if ns is not None else 0.0)
            ci_hi.append(float(hi) if hi is not None else float(ns) if ns is not None else 0.0)

        scores_a = np.array(scores)
        err_low = np.clip(scores_a - np.array(ci_lo), 0, None)
        err_high = np.clip(np.array(ci_hi) - scores_a, 0, None)
        offset = (i - n_algos / 2 + 0.5) * width
        ax.bar(
            x + offset,
            scores,
            width=width * 0.9,
            label=ALGO_LABELS.get(algo, algo),
            color=ALGO_COLORS.get(algo, "#888888"),
            alpha=0.85,
            yerr=[err_low, err_high],
            capsize=3,
            error_kw={"linewidth": 0.8, "alpha": 0.7},
        )

    ax.axhline(0, color="#9e9e9e", linewidth=0.8, linestyle="--")
    ax.axhline(1, color="#4caf50", linewidth=0.8, linestyle="--", label="Rule-Based level")
    ax.set_xticks(x)
    ax.set_xticklabels([SPLIT_LABELS.get(s, s) for s in ood_splits], rotation=20, ha="right")
    ax.set_ylabel("NormScore")
    ax.set_title("DC Microgrid — OOD Robustness (95% CI)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    out = figures_dir / "ood_robustness.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[DC Microgrid plots] saved {out}")
    return out


def generate_all_plots(task_dir: Path = TASK_DIR) -> None:
    plot_normscore_bars(task_dir)
    plot_reward_curves(task_dir)
    plot_cost_decomposition(task_dir)
    plot_ood_robustness(task_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", default=str(TASK_DIR))
    args = parser.parse_args()
    generate_all_plots(Path(args.task_dir))
