"""DERs benchmark figure generation.

Reads summary JSON and training-curve .npy artifacts to produce:

  1. normscore_bars.pdf   — NormScore per (algo, split), grouped bar chart
  2. learning_curves.pdf  — Mean reward vs env steps (one curve per train record)
  3. learning_curves_walltime.pdf — Same curves with **wall-clock** on the x-axis
  4. cross_backend_learning_walltime.pdf — Phase-2 style overlay: jax IPPO vs
     SB3/SBX PPO, all trained on the canonical train split
  5. voltage_safety.pdf   — voltage_safety_rate per (algo, split), bar chart
  6. ood_robustness.pdf   — cost_reduction_pct on iid vs OOD splits (spider/bar)

Wall-time x-axis uses ``learning_curve_eval_walltimes.npy`` when present and
length-matched; otherwise a proportional map from ``timesteps.npy`` using each
run's ``walltime_s`` and ``total_timesteps`` (same contract as DSO plots).

Usage:
    python benchmarks/ders/plots.py
    python benchmarks/ders/plots.py --only cross_backend --seed 0
    python benchmarks/ders/run_all.py --only plots
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.common.artifacts import read_parallel_n_envs_from_run_config
from benchmarks.common.io import has_training_artifact, load_manifest, load_manifest_filtered

TASK_DIR = Path(__file__).resolve().parent

SPLIT_ORDER = ["train", "iid", "voltage_tightening", "pv_penetration_shift", "load_stress"]
SPLIT_LABELS = {
    "train": "Train",
    "iid": "IID",
    "voltage_tightening": "Volt Tight",
    "pv_penetration_shift": "PV +100%",
    "load_stress": "Load +15%",
}

ALGO_ORDER = ["no_control", "volt_droop", "ippo", "ippo_safe", "ippo_lagrangian"]
ALGO_LABELS = {
    "no_control": "No Control",
    "volt_droop": "Volt Droop",
    "ippo": "IPPO",
    "ippo_safe": "IPPO-rs",
    "ippo_lagrangian": "IPPO-Lagrangian",
}
ALGO_COLORS = {
    "no_control": "#9e9e9e",
    "volt_droop": "#4caf50",
    "ippo": "#2196f3",
    "ippo_safe": "#e91e63",
    "ippo_lagrangian": "#ff9800",
}

# Phase-2 / cross-backend overlays (backend family — legacy single hue)
BACKEND_COLORS = {
    "jax_rejax": "#1565c0",
    "sb3": "#ef6c00",
    "sbx": "#6a1b9a",
}


def _figures_dir(task_dir: Path) -> Path:
    return task_dir / "results" / "figures"


def _cross_backend_series_color(backend: str, device: str) -> str:
    """Distinct colours per (backend, device) so e.g. JAX GPU vs JAX CPU do not share a hue."""
    be = backend or "jax_rejax"
    dev = (device or "gpu").lower()
    if be == "jax_rejax":
        return "#0d47a1" if dev == "gpu" else "#00695c"  # blue vs teal
    if be == "sb3":
        return "#e65100" if dev == "gpu" else "#f9a825"  # deep orange vs amber
    if be == "sbx":
        return "#4a148c" if dev == "gpu" else "#ab47bc"  # purple vs light purple
    return "#455a64"


def _load_summary(task_dir: Path) -> Optional[dict]:
    path = task_dir / "results" / "summary" / "latest.json"
    if not path.exists():
        print(f"[DERs plots] No summary found at {path}. Run 'summarize' first.")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _get_row(rows: list[dict], algo: str, split: str) -> Optional[dict]:
    for r in rows:
        if r["algo"] == algo and r["split"] == split:
            return r
    return None


def _total_timesteps_from_record(
    metrics: dict,
    artifacts_dir: Path,
    run_id: str,
) -> float:
    v = metrics.get("total_timesteps")
    if v is not None:
        return float(v)
    cfg_f = artifacts_dir / f"{run_id}_config.json"
    if cfg_f.exists():
        try:
            cfg = json.loads(cfg_f.read_text(encoding="utf-8"))
            for key in ("train_config", "powerzoo_driver_config", "task_config"):
                sub = cfg.get(key)
                if isinstance(sub, dict) and sub.get("total_timesteps") is not None:
                    return float(sub["total_timesteps"])
        except Exception:
            pass
    return 0.0


def _load_walltime_checkpoints(
    run_id: str,
    artifacts_dir: Path,
    expected_len: int,
) -> Optional[np.ndarray]:
    f = artifacts_dir / f"{run_id}_learning_curve_eval_walltimes.npy"
    if not f.exists():
        return None
    arr = np.load(f).flatten()
    if len(arr) != expected_len:
        return None
    return arr


def _timesteps_to_walltime(
    ts: np.ndarray,
    *,
    walltime_s: float,
    total_timesteps: float,
    run_id: str,
    artifacts_dir: Path,
) -> tuple[np.ndarray, bool]:
    """Map env-step checkpoints to seconds. Returns (xs_seconds, from_file)."""
    wall = _load_walltime_checkpoints(run_id, artifacts_dir, len(ts))
    if wall is not None:
        return wall, True
    if total_timesteps > 0 and walltime_s > 0:
        return ts * walltime_s / total_timesteps, False
    return ts.astype(float).copy(), False


def _load_train_return_series(
    artifacts_dir: Path,
    run_id: str,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    for name in ("learning_curve_train_return", "mean_reward"):
        y_path = artifacts_dir / f"{run_id}_{name}.npy"
        if not y_path.exists():
            continue
        y = np.load(y_path).flatten()
        ts_path = artifacts_dir / f"{run_id}_timesteps.npy"
        if not ts_path.exists():
            continue
        ts = np.load(ts_path).flatten()
        if len(ts) == len(y) and len(y) > 1:
            return ts, y
    return None, None


# ── Figure 1: NormScore grouped bar chart ────────────────────────────────────

def plot_normscore_bars(task_dir: Path = TASK_DIR) -> Path:
    """Grouped bar chart: NormScore per (algo, split)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = _load_summary(task_dir)
    if summary is None:
        return Path()

    rows = summary["rows"]
    splits = [s for s in SPLIT_ORDER if any(
        r["split"] == s for r in rows
    )]
    algos = [a for a in ALGO_ORDER if any(r["algo"] == a for r in rows)]

    fig, ax = plt.subplots(figsize=(9, 4))
    n_splits = len(splits)
    n_algos = len(algos)
    bar_width = 0.7 / n_algos
    x = np.arange(n_splits)

    for j, algo in enumerate(algos):
        vals = []
        errs = []
        for split in splits:
            row = _get_row(rows, algo, split)
            vals.append(row["norm_score"] if row and row.get("norm_score") is not None else 0.0)
            errs.append(0.0)
        offset = (j - n_algos / 2 + 0.5) * bar_width
        ax.bar(
            x + offset, vals, bar_width,
            label=ALGO_LABELS.get(algo, algo),
            color=ALGO_COLORS.get(algo, "#999"),
            yerr=errs, capsize=3,
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(1, color="green", linewidth=0.8, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels([SPLIT_LABELS.get(s, s) for s in splits], rotation=20, ha="right")
    ax.set_ylabel("NormScore (↑ better)")
    ax.set_title("DERs — NormScore by Algorithm and Split")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    figures_dir = _figures_dir(task_dir)
    out = figures_dir / "normscore_bars.pdf"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[DERs plots] normscore_bars -> {out}")
    return out


# ── Figure 2: Learning curves ─────────────────────────────────────────────────

def plot_learning_curves(task_dir: Path = TASK_DIR) -> Path:
    """Mean-reward learning curves for trained DERs MARL variants."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = load_manifest(task_dir)
    train_records = [
        r for r in records
        if r.split == "train"
        and r.algo in ("ippo", "ippo_safe", "ippo_lagrangian")
        and (r.artifacts or {}).get("params")
    ]
    if not train_records:
        print("[DERs plots] No training records with params found; skipping learning_curves.")
        return Path()

    fig, ax = plt.subplots(figsize=(7, 4))
    arts_dir = task_dir / "results" / "artifacts"

    for r in train_records:
        reward_path = arts_dir / f"{r.run_id}_mean_reward.npy"
        ts_path = arts_dir / f"{r.run_id}_timesteps.npy"
        if not reward_path.exists():
            continue
        rewards = np.load(reward_path)
        xs = np.load(ts_path) if ts_path.exists() else np.arange(len(rewards))
        color = ALGO_COLORS.get(r.algo, "#999")
        ax.plot(
            xs / 1e6, rewards,
            color=color, alpha=0.7,
            label=f"{ALGO_LABELS.get(r.algo, r.algo)} s{r.seed}",
        )

    ax.set_xlabel("Timesteps (M)")
    ax.set_ylabel("Mean reward per update")
    ax.set_title("DERs — MARL Learning Curves")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()

    figures_dir = _figures_dir(task_dir)
    out = figures_dir / "learning_curves.pdf"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[DERs plots] learning_curves -> {out}")
    return out


def plot_learning_curves_walltime(
    task_dir: Path = TASK_DIR,
    *,
    include_jax_cpu: bool = False,
) -> Path:
    """DERs MARL train curves with **wall time (s)** on the x-axis.

    By default **omits** ``jax_rejax`` runs on **CPU** so the x-range matches
    the main GPU reference (Phase-2 JAX-CPU matrix rows would otherwise stretch
    the axis to ~30min). Pass ``include_jax_cpu=True`` to overlay CPU curves too.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = load_manifest(task_dir)
    train_records = [
        r for r in records
        if r.split == "train" and r.algo in ("ippo", "ippo_safe", "ippo_lagrangian")
        and (r.artifacts or {}).get("params")
    ]
    if not include_jax_cpu:
        train_records = [
            r for r in train_records
            if not ((r.backend or "jax_rejax") == "jax_rejax"
                    and (r.device or "gpu") == "cpu")
        ]
    if not train_records:
        print("[DERs plots] No DERs MARL train records; skipping learning_curves_walltime.")
        return Path()

    fig, ax = plt.subplots(figsize=(8, 4.2))
    arts_dir = task_dir / "results" / "artifacts"
    any_estimated = False

    for r in train_records:
        ts, y = _load_train_return_series(arts_dir, r.run_id)
        if ts is None or y is None:
            continue
        total_ts = _total_timesteps_from_record(r.metrics or {}, arts_dir, r.run_id)
        wt = float(r.walltime_s or 0.0)
        xs, from_file = _timesteps_to_walltime(
            ts, walltime_s=wt, total_timesteps=total_ts,
            run_id=r.run_id, artifacts_dir=arts_dir,
        )
        if not from_file and wt > 0 and total_ts > 0:
            any_estimated = True
        color = ALGO_COLORS.get(r.algo, "#999")
        be = r.backend or "jax_rejax"
        dev = r.device or "gpu"
        n_envs = read_parallel_n_envs_from_run_config(arts_dir, r.run_id)
        env_tag = f", n_envs={n_envs}" if n_envs is not None else ""
        ax.plot(
            xs, y, color=color, alpha=0.85, linewidth=1.6,
            label=f"{ALGO_LABELS.get(r.algo, r.algo)} s{r.seed} [{be}/{dev}{env_tag}]",
        )

    ax.set_xlabel("Wall time (s)")
    ax.set_ylabel("Train return (checkpoint)")
    title = "DERs — Learning curves vs wall time (native MARL train)"
    if any_estimated:
        title += "\n(proportional wall time where checkpoint logs are absent)"
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()

    figures_dir = _figures_dir(task_dir)
    out = figures_dir / "learning_curves_walltime.pdf"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[DERs plots] learning_curves_walltime -> {out}")
    return out


def plot_cross_backend_learning_walltime(
    task_dir: Path = TASK_DIR,
    *,
    seed: int = 0,
    after: str | None = None,
) -> Path:
    """Overlay jax IPPO with SB3/SBX PPO IL training curves on the train split."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if after:
        records = load_manifest_filtered(task_dir, after=after)
    else:
        records = load_manifest(task_dir)

    arts_dir = task_dir / "results" / "artifacts"
    picked: list = []
    for r in records:
        if r.status != "completed":
            continue
        if r.seed != seed:
            continue
        if not has_training_artifact(r.artifacts):
            continue
        if r.algo == "ippo" and r.split == "train":
            picked.append(r)
        elif r.algo in ("ppo", "sbx_ppo") and r.split == "train":
            picked.append(r)

    if not picked:
        print(
            "[DERs plots] No cross-backend train records "
            f"(seed={seed}, after={after!r}); skipping cross_backend_learning_walltime."
        )
        return Path()

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    any_estimated = False
    # Longer runs first so shorter curves remain visible
    picked.sort(key=lambda r: float(r.walltime_s or 0.0), reverse=True)

    for r in picked:
        ts, y = _load_train_return_series(arts_dir, r.run_id)
        if ts is None or y is None:
            print(f"[DERs plots] cross_backend: skip {r.run_id} (no curve)")
            continue
        total_ts = _total_timesteps_from_record(r.metrics or {}, arts_dir, r.run_id)
        wt = float(r.walltime_s or 0.0)
        xs, from_file = _timesteps_to_walltime(
            ts, walltime_s=wt, total_timesteps=total_ts,
            run_id=r.run_id, artifacts_dir=arts_dir,
        )
        if not from_file and wt > 0 and total_ts > 0:
            any_estimated = True
        be = r.backend or "jax_rejax"
        dev = (r.device or "gpu").lower()
        color = _cross_backend_series_color(be, dev)
        ls = "-" if dev == "gpu" else "--"
        algo_disp = {"ppo": "PPO", "sbx_ppo": "SBX-PPO", "ippo": "IPPO"}.get(
            r.algo, r.algo.upper()
        )
        n_envs = read_parallel_n_envs_from_run_config(arts_dir, r.run_id)
        env_tag = f", n_envs={n_envs}" if n_envs is not None else ""
        label = f"{be}/{dev} · {algo_disp}{env_tag}"
        ax.plot(xs, y, color=color, linestyle=ls, linewidth=2.0, label=label, alpha=0.9)

    ax.set_xlabel("Wall time (s)")
    ax.set_ylabel("Train return (checkpoint)")
    title = (
        f"DERs — Cross-backend learning vs wall time (seed={seed})\n"
        "All backends trained on the canonical train split"
    )
    if any_estimated:
        title += "\n(proportional wall time where checkpoint logs are absent)"
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="best", framealpha=0.92)
    fig.tight_layout()

    figures_dir = _figures_dir(task_dir)
    out = figures_dir / "cross_backend_learning_walltime.pdf"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[DERs plots] cross_backend_learning_walltime -> {out}")
    return out


# ── Figure 3: Voltage safety rate ────────────────────────────────────────────

def plot_voltage_safety(task_dir: Path = TASK_DIR) -> Path:
    """Bar chart of voltage_safety_rate per (algo, split)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = _load_summary(task_dir)
    if summary is None:
        return Path()

    rows = summary["rows"]
    splits = [s for s in SPLIT_ORDER if any(r["split"] == s for r in rows)]
    algos = [a for a in ALGO_ORDER if any(r["algo"] == a for r in rows)]

    fig, ax = plt.subplots(figsize=(9, 4))
    n_splits = len(splits)
    n_algos = len(algos)
    bar_width = 0.7 / n_algos
    x = np.arange(n_splits)

    for j, algo in enumerate(algos):
        vals = []
        for split in splits:
            row = _get_row(rows, algo, split)
            v = (row.get("voltage_safety_rate_mean") or 0.0) * 100.0 if row else 0.0
            vals.append(v)
        offset = (j - n_algos / 2 + 0.5) * bar_width
        ax.bar(
            x + offset, vals, bar_width,
            label=ALGO_LABELS.get(algo, algo),
            color=ALGO_COLORS.get(algo, "#999"),
        )

    ax.set_xticks(x)
    ax.set_xticklabels([SPLIT_LABELS.get(s, s) for s in splits], rotation=20, ha="right")
    ax.set_ylabel("Voltage Safety Rate (%)")
    ax.set_ylim(0, 110)
    ax.axhline(100, color="green", linewidth=0.8, linestyle=":")
    ax.set_title("DERs — Fraction of Steps with All Voltages in Bounds")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    figures_dir = _figures_dir(task_dir)
    out = figures_dir / "voltage_safety.pdf"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[DERs plots] voltage_safety -> {out}")
    return out


# ── Figure 4: OOD robustness (cost_reduction_pct across splits) ───────────────

def plot_ood_robustness(task_dir: Path = TASK_DIR) -> Path:
    """Bar chart comparing cost_reduction_pct on iid vs OOD splits per algo."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = _load_summary(task_dir)
    if summary is None:
        return Path()

    rows = summary["rows"]
    eval_splits = ["iid", "voltage_tightening", "pv_penetration_shift", "load_stress"]
    eval_splits = [s for s in eval_splits if any(r["split"] == s for r in rows)]
    algos = [a for a in ALGO_ORDER if any(r["algo"] == a for r in rows)]

    fig, ax = plt.subplots(figsize=(9, 4))
    n_splits = len(eval_splits)
    n_algos = len(algos)
    bar_width = 0.7 / n_algos
    x = np.arange(n_splits)

    for j, algo in enumerate(algos):
        vals = []
        for split in eval_splits:
            row = _get_row(rows, algo, split)
            vals.append(row.get("cost_reduction_pct_mean") or 0.0 if row else 0.0)
        offset = (j - n_algos / 2 + 0.5) * bar_width
        ax.bar(
            x + offset, vals, bar_width,
            label=ALGO_LABELS.get(algo, algo),
            color=ALGO_COLORS.get(algo, "#999"),
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([SPLIT_LABELS.get(s, s) for s in eval_splits], rotation=20, ha="right")
    ax.set_ylabel("Cost Reduction vs No-Control (%)")
    ax.set_title("DERs — OOD Robustness")
    ax.legend(fontsize=8)
    fig.tight_layout()

    figures_dir = _figures_dir(task_dir)
    out = figures_dir / "ood_robustness.pdf"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[DERs plots] ood_robustness -> {out}")
    return out


def generate_all_plots(
    task_dir: Path = TASK_DIR,
    *,
    include_jax_cpu_walltime: bool = False,
) -> None:
    """Generate standard DERs figures (incl. wall-time and cross-backend overlays)."""
    plot_normscore_bars(task_dir)
    plot_learning_curves(task_dir)
    plot_learning_curves_walltime(task_dir, include_jax_cpu=include_jax_cpu_walltime)
    plot_cross_backend_learning_walltime(task_dir)
    plot_voltage_safety(task_dir)
    plot_ood_robustness(task_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DERs benchmark plots.")
    parser.add_argument("--task-dir", default=str(TASK_DIR), type=Path)
    parser.add_argument(
        "--only",
        choices=["all", "cross_backend", "walltime"],
        default="all",
        help="all = full suite; cross_backend / walltime = partial regeneration",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed filter for --only cross_backend",
    )
    parser.add_argument(
        "--after",
        type=str, default=None,
        help="ISO timestamp: only manifest records at/after this instant (cross_backend)",
    )
    parser.add_argument(
        "--include-jax-cpu-walltime",
        action="store_true",
        help="Include jax_rejax+CPU native IPPO curves on learning_curves_walltime (stretches x-axis).",
    )
    args = parser.parse_args()
    td = args.task_dir
    if args.only == "all":
        generate_all_plots(td, include_jax_cpu_walltime=args.include_jax_cpu_walltime)
    elif args.only == "walltime":
        plot_learning_curves_walltime(td, include_jax_cpu=args.include_jax_cpu_walltime)
    else:
        plot_cross_backend_learning_walltime(td, seed=args.seed, after=args.after)
