"""DSO benchmark figure generation.

Reads summary JSON and learning-curve .npy artifacts to produce:
  1. learning_curves.pdf  — Reward vs timestep, one curve per (algo, seed)
  2. learning_curves_walltime.pdf — Reward vs wall time
  3. loss_reduction.pdf   — Network loss reduction by algorithm
  4. load_profiles.pdf    — Ausgrid feeder shapes for each configured split

Usage:
    python benchmarks/dso/plots.py [--task-dir benchmarks/dso]
    python benchmarks/dso/run.py plots
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from benchmarks.common.artifacts import read_parallel_n_envs_from_run_config
from benchmarks.common.configs import load_task_config
from benchmarks.common.io import load_manifest_filtered

TASK_DIR = Path(__file__).resolve().parent
FIGURES_DIR = TASK_DIR / "results" / "figures"

SPLIT_ORDER = ["train", "iid"]
SPLIT_LABELS = {
    "train": "Train",
    "iid": "IID",
}

ALGO_ORDER = [
    "no_control", "tou", "droop",
    "ppo", "sac", "saute_ppo", "ppo_lagrangian",
]
ALGO_LABELS = {
    "no_control": "No Control",
    "tou": "TOU",
    "droop": "Droop",
    "ppo": "PPO",
    "sac": "SAC",
    "saute_ppo": "Saute-PPO",
    "ppo_lagrangian": "PPO-Lag",
}

ALGO_COLORS = {
    "no_control": "#9e9e9e",
    "tou": "#4caf50",
    "droop": "#ff9800",
    "ppo": "#2196f3",
    "sac": "#795548",
    "saute_ppo": "#7b1fa2",
    "ppo_lagrangian": "#e91e63",
}


def _load_summary(task_dir: Path) -> Optional[dict]:
    path = task_dir / "results" / "summary" / "latest.json"
    if not path.exists():
        print(f"[DSO plots] No summary found at {path}. Run 'summarize' first.")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _get_row(rows: list[dict], algo: str, split: str) -> Optional[dict]:
    for r in rows:
        if r["algo"] == algo and r["split"] == split:
            return r
    return None


def _aggregated_n_envs_suffix(algo_nenv: dict[str, set[int]], algo: str) -> str:
    vals = algo_nenv.get(algo)
    if not vals:
        return ""
    if len(vals) == 1:
        return f", n_envs={next(iter(vals))}"
    lo, hi = min(vals), max(vals)
    return f", n_envs={lo}–{hi}" if lo != hi else f", n_envs={lo}"


def _task_max_steps(task_dir: Path) -> int:
    try:
        cfg = load_task_config(task_dir)
        return int(cfg.get("max_steps", 48))
    except Exception:
        return 48


def _safe_cost_thresholds(task_dir: Path) -> list[float]:
    """Return per-step cost thresholds for plotting.

    Reads cost_thresholds from configs/train_safe.yaml (episode-level budget)
    and divides by max_steps to convert to per-step units matching the y-axis.
    """
    try:
        import yaml  # type: ignore
        p = task_dir / "configs" / "train_safe.yaml"
        with p.open() as f:
            cfg = yaml.safe_load(f)
        val = cfg.get("cost_thresholds", [None])
        if isinstance(val, (int, float)):
            raw = [float(val)]
        else:
            raw = [float(v) for v in val if v is not None]
        max_steps = _task_max_steps(task_dir)
        # Episode budget → per-step budget for plot display
        return [v / max_steps for v in raw]
    except Exception:
        return []


def _canonical_train_run_ids(
    task_dir: Path,
    *,
    after: str | None = None,
    backend: str = "jax_rejax",
    device: str = "gpu",
) -> dict[tuple[str, int], str]:
    """Map (algo, seed) -> training run_id from manifest (split=train, has params)."""
    records = [
        r.to_dict()
        for r in load_manifest_filtered(
            task_dir,
            after=after,
            backend=backend,
            device=device,
        )
    ]
    out: dict[tuple[str, int], str] = {}
    for r in records:
        if r.get("split") != "train":
            continue
        arts = r.get("artifacts") or {}
        if not arts or (
            "params" not in arts
            and "params_orbax" not in arts
            and "params_flax" not in arts
        ):
            continue
        out[(r.get("algo"), int(r.get("seed", -1)))] = r.get("run_id", "")
    return out


def _canonical_train_runs(
    task_dir: Path,
    *,
    after: str | None = None,
    backend: str = "jax_rejax",
    device: str = "gpu",
) -> dict[tuple[str, int], dict]:
    """Map (algo, seed) -> full manifest record for canonical training runs."""
    records = [
        r.to_dict()
        for r in load_manifest_filtered(
            task_dir,
            after=after,
            backend=backend,
            device=device,
        )
    ]
    out: dict[tuple[str, int], dict] = {}
    for r in records:
        if r.get("split") != "train":
            continue
        arts = r.get("artifacts") or {}
        if not arts or (
            "params" not in arts
            and "params_orbax" not in arts
            and "params_flax" not in arts
        ):
            continue
        key = (r.get("algo"), int(r.get("seed", -1)))
        # Keep latest run_id (lexicographic timestamp suffix wins)
        if key not in out or r.get("run_id", "") > out[key].get("run_id", ""):
            out[key] = r
    return out


def _ema_smooth(arr: np.ndarray, span: int = 20) -> np.ndarray:
    """Exponential moving average smoothing."""
    alpha = 2.0 / (span + 1)
    out = np.empty(len(arr), dtype=float)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * float(arr[i]) + (1.0 - alpha) * out[i - 1]
    return out


def _load_ts(run_id: str, artifacts_dir: Path, expected_len: int) -> np.ndarray:
    ts_f = artifacts_dir / f"{run_id}_timesteps.npy"
    if ts_f.exists():
        ts = np.load(ts_f).flatten()
        if len(ts) == expected_len:
            return ts
    return np.linspace(0, 3_000_000, expected_len)


def _load_reward_curve(run_id: str, artifacts_dir: Path, max_steps: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Load the best available **episode-reward** curve for a training run.

    Priority:
      1. ``learning_curve_eval_return`` or ``eval_unpenalized_returns`` —
         offline greedy eval in the benchmark's native reward units.
      2. ``eval_returns``       — offline greedy eval fallback.
      3. ``mean_reward`` / ``learning_curve_train_return`` — training rollout
         mean reward per step; multiply by ``max_steps`` to approximate
         episode-level reward.

    Returns ``(timesteps, episode_rewards, source_label)``.
    """
    for name in (
        "learning_curve_eval_return",
        "eval_unpenalized_returns",
        "eval_returns",
    ):
        f = artifacts_dir / f"{run_id}_{name}.npy"
        if f.exists():
            arr = np.load(f).flatten()
            return _load_ts(run_id, artifacts_dir, len(arr)), arr, "eval"

    for name in ("learning_curve_train_return", "mean_reward"):
        f = artifacts_dir / f"{run_id}_{name}.npy"
        if f.exists():
            arr = np.load(f).flatten()
            return _load_ts(run_id, artifacts_dir, len(arr)), arr * max_steps, "train (fallback)"

    return np.array([]), np.array([]), "none"


def _load_walltime_curve(
    run_id: str,
    artifacts_dir: Path,
    expected_len: int,
) -> np.ndarray | None:
    f = artifacts_dir / f"{run_id}_learning_curve_eval_walltimes.npy"
    if not f.exists():
        return None
    arr = np.load(f).flatten()
    if len(arr) != expected_len:
        return None
    return arr


def _load_cost_curve(run_id: str, artifacts_dir: Path) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Load per-step voltage violation cost curve. Returns (ts, cost) or (None, None).

    Priority: eval_cost (greedy eval pass) > mean_cost (training rollout).
    """
    for name in (
        "eval_cost_voltage_violation",
        "eval_cost_per_step",
        "mean_cost_voltage_violation",
        "mean_cost",
    ):
        f = artifacts_dir / f"{run_id}_{name}.npy"
        if f.exists():
            arr = np.load(f).flatten()
            return _load_ts(run_id, artifacts_dir, len(arr)), arr
    return None, None


def _load_ppo_reference_violations(
    task_dir: Path,
    max_steps: int,
    *,
    after: str | None = None,
    backend: str = "jax_rejax",
    device: str = "gpu",
) -> dict[str, float]:
    """Load non-CMDP algo (e.g. PPO) final eval per-step violations from per_episode.json.

    Uses the latest per-seed train-split evaluation to avoid double-counting reruns.
    Returns {algo: mean_per_step_violations} for all algos with per_episode data.
    """
    manifest = [
        r.to_dict()
        for r in load_manifest_filtered(
            task_dir,
            after=after,
            backend=backend,
            device=device,
        )
    ]

    # Latest eval record per (algo, seed) on the train split (no params = pure eval record)
    best: dict[tuple[str, int], dict] = {}
    for r in manifest:
        arts = r.get("artifacts") or {}
        if "per_episode" not in arts or "params" in arts:
            continue
        if r.get("split") != "train":
            continue
        key = (r.get("algo"), int(r.get("seed", -1)))
        if key not in best or r.get("run_id", "") > best[key].get("run_id", ""):
            best[key] = r

    algo_violations: dict[str, list[float]] = {}
    for (algo, _seed), r in best.items():
        per_ep_path = task_dir / "results" / r["artifacts"]["per_episode"]
        try:
            episodes = json.loads(per_ep_path.read_text(encoding="utf-8"))
            violations = [
                float(ep.get("total_voltage_violations", ep.get("total_violations", 0.0)))
                / max_steps
                for ep in episodes
            ]
            if violations:
                algo_violations.setdefault(algo, []).extend(violations)
        except Exception:
            pass

    return {algo: float(np.mean(v)) for algo, v in algo_violations.items()}


def _load_lambda_curve(run_id: str, artifacts_dir: Path) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Load Lagrange multiplier curve. Returns (ts, lambda) or (None, None)."""
    for name in ("lambda_voltage_violation", "lambda"):
        f = artifacts_dir / f"{run_id}_{name}.npy"
        if f.exists():
            arr = np.load(f).flatten()
            # Skip if constant (uninformative)
            if np.ptp(arr) < 1e-6:
                return None, None
            ts_f = artifacts_dir / f"{run_id}_timesteps.npy"
            ts = np.load(ts_f).flatten() if ts_f.exists() and len(np.load(ts_f).flatten()) == len(arr) else np.linspace(0, 3_000_000, len(arr))
            return ts, arr
    return None, None


def _aggregate_curves(
    curves: list[tuple[np.ndarray, np.ndarray]],
    n_grid: int = 300,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate multiple (ts, vals) curves onto a common grid and return (grid, mean, std).

    The time grid runs from 0 to ``max(ts[-1])`` **across seeds**.  For wall-time x-axes,
    that means one slow seed stretches the whole x-axis for the mean band (expected if
    seeds are comparable); if older / different-hardware runs are mixed in, the figure
    looks like one algorithm "takes longer" even when all used the same ``total_timesteps``.
    """
    if not curves:
        return np.array([]), np.array([]), np.array([])
    ts_max = max(ts[-1] for ts, _ in curves if len(ts))
    grid = np.linspace(0, ts_max, n_grid)
    interp = np.stack([np.interp(grid, ts, vals) for ts, vals in curves if len(ts) > 1])
    return grid, interp.mean(axis=0), interp.std(axis=0)


# Wall-time x-axis is in **seconds** (checkpoints linearly mapped from env steps using each
# run’s ``walltime_s`` and ``total_timesteps``).  Multi-seed means follow ``_aggregate_curves``:
# grid 0..max end-time in seconds (not normalized), so the figure stays a real-time comparison.

def _walltime_from_proportional(
    ts: np.ndarray,
    total_ts: float,
    total_wt: float,
) -> np.ndarray:
    """Estimate per-checkpoint walltime by linear scaling from total walltime."""
    if total_ts <= 0 or total_wt <= 0:
        return ts.copy()
    return ts * total_wt / total_ts


# ──────────────────────────────────────────────────────────────────────────────
# Fig 2: Learning curves (timestep + walltime variants)
# ──────────────────────────────────────────────────────────────────────────────

def _build_learning_curve_panels(
    task_dir: Path,
    x_mode: str = "timesteps",  # "timesteps" | "walltime"
    *,
    after: str | None = None,
    backend: str = "jax_rejax",
    device: str = "gpu",
) -> tuple["plt.Figure", bool]:  # type: ignore[name-defined]
    """Build the shared multi-panel learning curve figure.

    x_mode="timesteps"  → x-axis in millions of env steps
    x_mode="walltime"   → x-axis in seconds (actual checkpoint times when
                          available, otherwise proportional estimate)

    Panel layout:
      Row 0: Episode reward (PPO / PPO-Lag / penalty PPO)
      Row 1: Per-step voltage-violation cost (CMDP + eval curves; penalty/PPO
             use greedy-eval ``eval_cost_per_step`` when logged)
      Row 2: Lagrange λ (omitted when all runs have constant λ — means λ saturated)

    Multi-seed: mean ± 1-std shaded band; EMA smoothing for noisy training signals.
    """
    import matplotlib.pyplot as plt

    artifacts_dir = task_dir / "results" / "artifacts"
    canonical = _canonical_train_runs(
        task_dir,
        after=after,
        backend=backend,
        device=device,
    )
    if not canonical or not artifacts_dir.exists():
        fig, ax = plt.subplots(1, 1, figsize=(9, 4))
        ax.text(0.5, 0.5, "No training artifacts found.\nRun stage 2 (train) first.",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        return fig, False

    max_steps = _task_max_steps(task_dir)

    # Collect per-algo seed curves
    algo_reward: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    algo_cost:   dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    algo_lam:    dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    # "eval" takes priority: once any seed has greedy eval, label the whole algo "eval"
    algo_source: dict[str, str] = {}
    algo_nenv: dict[str, set[int]] = {}
    used_estimated_walltime = False

    for (algo, seed), rec in sorted(canonical.items()):
        run_id = rec["run_id"]
        _ne = read_parallel_n_envs_from_run_config(artifacts_dir, run_id)
        if _ne is not None:
            algo_nenv.setdefault(algo, set()).add(_ne)
        total_ts = float(rec.get("metrics", {}).get("total_timesteps", 3_000_000) or 3_000_000)
        # Fallback total_ts from config or default
        cfg_f = artifacts_dir / f"{run_id}_config.json"
        if cfg_f.exists():
            try:
                cfg = json.loads(cfg_f.read_text())
                total_ts = float(
                    cfg.get("train_config", cfg).get("total_timesteps", total_ts)
                )
            except Exception:
                pass
        total_wt = float(rec.get("walltime_s") or 0.0)

        ts, reward, source = _load_reward_curve(run_id, artifacts_dir, max_steps)
        if len(ts) < 2:
            print(f"[DSO plots] no reward curve for {algo} seed={seed} ({run_id})")
            continue
        # Only upgrade source label (eval > train fallback), never downgrade
        if algo not in algo_source or source == "eval":
            algo_source[algo] = source

        if x_mode == "walltime":
            wall_ts = _load_walltime_curve(run_id, artifacts_dir, len(ts))
            if wall_ts is not None:
                ts = wall_ts
            else:
                ts = _walltime_from_proportional(ts, total_ts, total_wt)
                used_estimated_walltime = True

        smooth_span = 20 if source.startswith("train") else 5
        reward_s = _ema_smooth(reward, span=smooth_span)

        algo_reward.setdefault(algo, []).append((ts, reward_s))

        ts_c, cost = _load_cost_curve(run_id, artifacts_dir)
        if ts_c is not None and cost is not None:
            if x_mode == "walltime":
                wall_ts_c = _load_walltime_curve(run_id, artifacts_dir, len(ts_c))
                if wall_ts_c is not None:
                    ts_c = wall_ts_c
                else:
                    ts_c = _walltime_from_proportional(ts_c, total_ts, total_wt)
                    used_estimated_walltime = True
            algo_cost.setdefault(algo, []).append((ts_c, _ema_smooth(cost, span=20)))

        ts_l, lam = _load_lambda_curve(run_id, artifacts_dir)
        if ts_l is not None and lam is not None:
            if x_mode == "walltime":
                wall_ts_l = _load_walltime_curve(run_id, artifacts_dir, len(ts_l))
                if wall_ts_l is not None:
                    ts_l = wall_ts_l
                else:
                    ts_l = _walltime_from_proportional(ts_l, total_ts, total_wt)
                    used_estimated_walltime = True
            algo_lam.setdefault(algo, []).append((ts_l, lam))

    if not algo_reward:
        fig, ax = plt.subplots(1, 1, figsize=(9, 4))
        ax.text(0.5, 0.5, "No reward curves loaded.", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        return fig, False

    has_cost = bool(algo_cost)
    has_lam  = bool(algo_lam)
    n_panels = 3

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes_arr = plt.subplots(
        1, n_panels, figsize=(10.0, 3.4),
        sharex=True, squeeze=False, constrained_layout=False,
    )
    axes_list: list = axes_arr[0, :].tolist()
    ax_r = axes_list[0]

    present_algos = [a for a in ALGO_ORDER if a in algo_reward]

    if x_mode == "walltime":
        for algo, seed_curves in algo_reward.items():
            ends = [float(ts[-1]) for ts, _ in seed_curves if len(ts) > 1]
            if len(ends) >= 2:
                lo, hi = min(ends), max(ends)
                if hi > 1.5 * max(lo, 1e-6):
                    print(
                        f"[DSO plots] wall-time: algo={algo!r} per-seed end-time "
                        f"[{lo:.1f}s, {hi:.1f}s] (same total_timesteps; spread usually load/GPU "
                        f"or mixed run batches). Re-train all seeds in comparable conditions if "
                        f"you need a tight time comparison; x-axis is seconds, 0..{hi:.1f}s for mean."
                    )

    # ── Panel 0: Reward ────────────────────────────────────────────────────────
    for algo in present_algos:
        color  = ALGO_COLORS.get(algo, "#607d8b")
        source = algo_source.get(algo, "?")
        curves = algo_reward[algo]
        grid, mean, std = _aggregate_curves(curves)
        n_seeds = len(curves)
        sem = std / max(n_seeds ** 0.5, 1.0)
        label = f"{ALGO_LABELS.get(algo, algo)}"
        if source == "eval":
            label += " (eval)"
        elif source.startswith("train"):
            label += " (train rollout)"
        label += _aggregated_n_envs_suffix(algo_nenv, algo)
        ax_r.plot(grid, mean, color=color, linewidth=2.0, label=label, zorder=3)
        ax_r.fill_between(grid, mean - sem, mean + sem,
                          color=color, alpha=0.18, zorder=2)

    x_label = (
        "Eval checkpoint wall time (s)"
        if x_mode == "walltime"
        else "Environment steps"
    )
    ax_r.set_ylabel("Episode total reward", fontsize=12)

    # ── Panel 1: Cost ─────────────────────────────────────────────────────────
    ax_c = axes_list[1]
    if has_cost:
        # Time-series cost curves for algos that record them during training
        for algo in present_algos:
            if algo not in algo_cost:
                continue
            color = ALGO_COLORS.get(algo, "#607d8b")
            grid, mean, std = _aggregate_curves(algo_cost[algo])
            n_seeds = len(algo_cost[algo])
            sem = std / max(n_seeds ** 0.5, 1.0)
            src_label = " (train rollout)" if algo_source.get(algo) == "train (fallback)" else ""
            ax_c.plot(grid, mean, color=color, linewidth=2.0,
                      label=f"{ALGO_LABELS.get(algo, algo)}{src_label}"
                      f"{_aggregated_n_envs_suffix(algo_nenv, algo)}", zorder=3)
            ax_c.fill_between(grid, mean - sem, mean + sem,
                              color=color, alpha=0.18, zorder=2)

        # Horizontal reference lines for algos without training-time cost curves
        # (e.g. PPO: only final eval violations available)
        ppo_ref = _load_ppo_reference_violations(
            task_dir,
            max_steps,
            after=after,
            backend=backend,
            device=device,
        )
        ref_linestyles = ["--", "-.", ":"]
        ref_idx = 0
        for algo in present_algos:
            if algo in algo_cost:
                continue  # already has time-series
            ref_val = ppo_ref.get(algo)
            if ref_val is None:
                continue
            color = ALGO_COLORS.get(algo, "#607d8b")
            ls = ref_linestyles[ref_idx % len(ref_linestyles)]
            ax_c.axhline(
                ref_val, color=color, linewidth=1.8, linestyle=ls,
                label=f"{ALGO_LABELS.get(algo, algo)} (final eval, mean={ref_val:.1f})"
                f"{_aggregated_n_envs_suffix(algo_nenv, algo)}",
                zorder=4,
            )
            ref_idx += 1

        # Draw the actual configured constraint threshold
        thresholds = _safe_cost_thresholds(task_dir)
        thresh_val = thresholds[0] if thresholds else 0.0
        ax_c.axhline(thresh_val, color="#e53935", linewidth=1.4, linestyle="--",
                     label=f"CMDP threshold ({thresh_val:.1f}/step)", zorder=5)
        ax_c.set_ylabel("Voltage violations / step", fontsize=12)
    else:
        ax_c.set_axis_off()

    # ── Panel 2: Lambda ───────────────────────────────────────────────────────
    ax_l = axes_list[2]
    if has_lam:
        import math as _math
        lambda_cap = _math.exp(5.0)  # exp(log_lambda_max=5.0)
        for algo in present_algos:
            if algo not in algo_lam:
                continue
            color = ALGO_COLORS.get(algo, "#607d8b")
            grid, mean, std = _aggregate_curves(algo_lam[algo])
            n_seeds = len(algo_lam[algo])
            sem = std / max(n_seeds ** 0.5, 1.0)
            ax_l.plot(grid, mean, color=color, linewidth=2.0,
                      label=f"{ALGO_LABELS.get(algo, algo)}"
                      f"{_aggregated_n_envs_suffix(algo_nenv, algo)}", zorder=3)
            ax_l.fill_between(grid, mean - sem, mean + sem,
                              color=color, alpha=0.18, zorder=2)
        ax_l.axhline(lambda_cap, color="#9e9e9e", linewidth=1.2, linestyle=":",
                     label=f"λ cap = exp(5) ≈ {lambda_cap:.0f}", zorder=4)
        ax_l.set_ylabel("Lagrange λ", fontsize=12)
    else:
        ax_l.set_axis_off()

    # ── Shared x-axis formatting ───────────────────────────────────────────────
    for ax in axes_list:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if x_mode == "timesteps":
            import matplotlib.ticker as mticker
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e5 else f"{x:.0f}")
            )

    for ax in axes_list:
        if ax.axison:
            ax.set_xlabel(x_label, fontsize=12)

    from matplotlib.lines import Line2D
    legend_handles: list = []
    for algo in present_algos:
        legend_handles.append(
            Line2D([0], [0], color=ALGO_COLORS.get(algo, "#607d8b"),
                   linewidth=2.4, label=ALGO_LABELS.get(algo, algo))
        )
    if has_cost:
        legend_handles.append(
            Line2D([0], [0], color="#e53935", linewidth=1.6, linestyle="--",
                   label="CMDP threshold")
        )
    if has_lam:
        legend_handles.append(
            Line2D([0], [0], color="#9e9e9e", linewidth=1.6, linestyle=":",
                   label="λ cap = exp(5)")
        )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=min(len(legend_handles), 6),
        fontsize=12,
        frameon=False,
        handlelength=1.4,
        columnspacing=0.8,
    )
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.18, top=0.78, wspace=0.34)
    return fig, True


def plot_learning_curves(
    task_dir: Path = TASK_DIR,
    *,
    after: str | None = None,
    backend: str = "jax_rejax",
    device: str = "gpu",
) -> Path:
    """Learning curves with environment-timesteps x-axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    artifacts_dir = task_dir / "results" / "artifacts"
    if not artifacts_dir.exists():
        print(f"[DSO plots] No artifacts at {artifacts_dir}.")
        return Path()

    fig, ok = _build_learning_curve_panels(
        task_dir,
        x_mode="timesteps",
        after=after,
        backend=backend,
        device=device,
    )
    if not ok:
        print("[DSO plots] No learning-curve data found.")
        plt.close(fig)
        return Path()

    out = FIGURES_DIR / "learning_curves.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[DSO plots] saved {out}")
    return out


def plot_learning_curves_walltime(
    task_dir: Path = TASK_DIR,
    *,
    after: str | None = None,
    backend: str = "jax_rejax",
    device: str = "gpu",
) -> Path:
    """Learning curves with wall-time x-axis.

    Uses actual checkpoint wall time when the training run saved it; otherwise
    falls back to proportional scaling from total wall time.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    artifacts_dir = task_dir / "results" / "artifacts"
    if not artifacts_dir.exists():
        print(f"[DSO plots] No artifacts at {artifacts_dir}.")
        return Path()

    fig, ok = _build_learning_curve_panels(
        task_dir,
        x_mode="walltime",
        after=after,
        backend=backend,
        device=device,
    )
    if not ok:
        print("[DSO plots] No learning-curve data found.")
        plt.close(fig)
        return Path()

    out = FIGURES_DIR / "learning_curves_walltime.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[DSO plots] saved {out}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Fig 3: Drift tracking gap
# ──────────────────────────────────────────────────────────────────────────────

def plot_drift_gap(task_dir: Path = TASK_DIR) -> Path:
    """Horizontal bar chart of drift_tracking_gap per algo."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = _load_summary(task_dir)
    if summary is None:
        return Path()

    rows = summary["rows"]
    # Deduplicate: one gap per algo (same value on all rows)
    seen: dict[str, float] = {}
    for r in rows:
        algo = r["algo"]
        gap = r.get("drift_tracking_gap")
        if gap is not None and algo not in seen:
            seen[algo] = float(gap)

    if not seen:
        print("[DSO plots] No drift_tracking_gap data — current DSO eval_splits only includes iid.")
        return Path()

    algos = [a for a in ALGO_ORDER if a in seen]
    gaps = [seen[a] for a in algos]
    labels = [ALGO_LABELS.get(a, a) for a in algos]
    colors = [ALGO_COLORS.get(a, "#607d8b") for a in algos]

    fig, ax = plt.subplots(figsize=(6, 0.7 * len(algos) + 1.5))
    y = np.arange(len(algos))
    bars = ax.barh(y, gaps, color=colors, alpha=0.88)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.axvline(0, color="#9e9e9e", linewidth=0.9, linestyle="--")
    ax.set_xlabel("Drift Tracking Gap  [NormScore(IID) − NormScore(OOD)]", fontsize=10)
    ax.set_title("DSO Drift Tracking Gap\n(smaller = more robust to distribution shift)", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out = FIGURES_DIR / "drift_gap.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[DSO plots] saved {out}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Fig 4: Load profiles (Ausgrid feeder shapes per split)
# ──────────────────────────────────────────────────────────────────────────────

def plot_load_profiles(task_dir: Path = TASK_DIR) -> Path:
    """Plot Ausgrid feeder aggregate load for each split (7-day window)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from powerzoojax.tasks.dso import load_dso_feeder_shapes
    except ImportError as e:
        print(f"[DSO plots] Cannot import dso_env: {e}")
        return Path()

    task_cfg = load_task_config(task_dir)
    splits = ["train", *[str(s) for s in task_cfg.get("eval_splits", ["iid"])]]
    splits = list(dict.fromkeys(splits))
    split_colors = {
        "train": "#2196f3",
        "iid": "#4caf50",
    }

    # Show 7 days = 336 half-hour intervals
    WINDOW = 336

    n_cols = min(2, len(splits))
    n_rows = int(np.ceil(len(splits) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3.5 * n_rows), sharex=False, sharey=False)
    axes_flat = np.asarray(axes).reshape(-1)
    for ax in axes_flat[len(splits):]:
        ax.axis("off")

    for ax, split in zip(axes_flat, splits):
        try:
            feeder_shapes = load_dso_feeder_shapes(data_loader=None, role=split)
        except Exception as exc:
            ax.set_title(f"{SPLIT_LABELS.get(split, split)}\n(data unavailable: {exc})", fontsize=9)
            continue

        # Sum all zones for total load
        keys = sorted(feeder_shapes.keys())
        if not keys:
            ax.set_title(f"{SPLIT_LABELS.get(split, split)}\n(no zones)", fontsize=9)
            continue

        total = np.sum([np.array(feeder_shapes[k]) for k in keys], axis=0)
        T = len(total)
        window = min(WINDOW, T)
        t = np.arange(window) * 0.5  # hours

        color = split_colors.get(split, "#607d8b")
        ax.plot(t, total[:window], color=color, linewidth=1.2, alpha=0.9)
        ax.fill_between(t, 0, total[:window], alpha=0.12, color=color)
        ax.set_title(f"{SPLIT_LABELS.get(split, split)} ({T} steps total)", fontsize=10, fontweight="bold")
        ax.set_xlabel("Hours", fontsize=9)
        ax.set_ylabel("Aggregate Load (MW)", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Ausgrid Feeder Load Profiles (first 7 days of each split)", fontsize=12, fontweight="bold")
    fig.tight_layout()

    out = FIGURES_DIR / "load_profiles.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[DSO plots] saved {out}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Fig 5: Loss reduction box/violin per split
# ──────────────────────────────────────────────────────────────────────────────

def plot_loss_reduction(task_dir: Path = TASK_DIR) -> Path:
    """Bar chart: network_loss_reduction_pct per algo per split (mean ± std)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = _load_summary(task_dir)
    if summary is None:
        return Path()

    rows = summary["rows"]
    present_algos = [a for a in ALGO_ORDER if any(r["algo"] == a for r in rows)]
    present_splits = [s for s in SPLIT_ORDER if any(r["split"] == s for r in rows)]

    n_splits = len(present_splits)
    n_algos = len(present_algos)
    x = np.arange(n_splits)
    width = 0.8 / n_algos

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, algo in enumerate(present_algos):
        means = []
        stds = []
        for split in present_splits:
            row = _get_row(rows, algo, split)
            if row and row.get("network_loss_reduction_pct_mean") is not None:
                means.append(float(row["network_loss_reduction_pct_mean"]))
                stds.append(float(row.get("network_loss_reduction_pct_std", 0.0)))
            else:
                means.append(float("nan"))
                stds.append(0.0)

        offset = (i - n_algos / 2 + 0.5) * width
        ax.bar(
            x + offset, means,
            yerr=stds,
            width=width * 0.9,
            label=ALGO_LABELS.get(algo, algo),
            color=ALGO_COLORS.get(algo, "#607d8b"),
            alpha=0.88,
            capsize=3,
            error_kw={"linewidth": 1.0},
        )

    ax.axhline(0, color="#9e9e9e", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([SPLIT_LABELS.get(s, s) for s in present_splits], fontsize=11)
    ax.set_ylabel("Network Loss Reduction (%)", fontsize=12)
    ax.set_title("DSO Network Loss Reduction by Algorithm and Split", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out = FIGURES_DIR / "loss_reduction.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[DSO plots] saved {out}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def generate_all_plots(
    task_dir: Path = TASK_DIR,
    *,
    after: str | None = None,
    backend: str = "jax_rejax",
    device: str = "gpu",
) -> None:
    """Generate all DSO paper figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_load_profiles(task_dir)
    plot_drift_gap(task_dir)
    plot_learning_curves(task_dir, after=after, backend=backend, device=device)
    plot_learning_curves_walltime(task_dir, after=after, backend=backend, device=device)
    plot_loss_reduction(task_dir)
    print(f"[DSO plots] All figures written to {FIGURES_DIR}/")


if __name__ == "__main__":
    import argparse
    import sys

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(_PROJECT_ROOT))

    parser = argparse.ArgumentParser(description="Generate DSO paper figures")
    parser.add_argument(
        "--task-dir", type=Path, default=TASK_DIR,
        help="Path to the DSO task directory",
    )
    parser.add_argument(
        "--only",
        choices=["curves", "curves_walltime", "drift", "profiles", "loss"],
        default=None, help="Generate only one specific figure",
    )
    parser.add_argument("--after", default=None, help="Campaign start ISO filter")
    parser.add_argument("--backend", default="jax_rejax", help="Filter manifest backend")
    parser.add_argument("--device", default="gpu", help="Filter manifest device")
    args = parser.parse_args()

    if args.only == "curves":
        plot_learning_curves(
            args.task_dir,
            after=args.after,
            backend=args.backend,
            device=args.device,
        )
    elif args.only == "curves_walltime":
        plot_learning_curves_walltime(
            args.task_dir,
            after=args.after,
            backend=args.backend,
            device=args.device,
        )
    elif args.only == "drift":
        plot_drift_gap(args.task_dir)
    elif args.only == "profiles":
        plot_load_profiles(args.task_dir)
    elif args.only == "loss":
        plot_loss_reduction(args.task_dir)
    else:
        generate_all_plots(
            args.task_dir,
            after=args.after,
            backend=args.backend,
            device=args.device,
        )
