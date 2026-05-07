"""TSO benchmark figure generation.

Reads summary JSON and learning-curve .npy artifacts to produce:
  1. normscore_bars.pdf   — NormScore per (algo, split), grouped bar chart
  2. gantt_commitment.pdf — Unit on/off timeline for one episode (Gantt chart)
  3. cost_decomposition.pdf — Gen / startup / no-load cost stacked bar
  4. learning_curves*.pdf — Timestep/walltime dashboards (cost, reserve, thermal),
     plus final checkpoint cost vs reserve scatter
  5. cross_backend_learning_walltime.{pdf,png} — Overlay eval cost vs wall time for
     every completed **train** row: JAX (PPO / PPO-Lag × GPU/CPU when present) plus
     SB3/SBX PPO rows as you add them to the manifest (same seed).

Usage:
    python benchmarks/tso/plots.py [--task-dir benchmarks/tso]
    python benchmarks/tso/run.py plots
    python benchmarks/tso/run.py plots --only cross_backend  # also runs learning_curves (run.py pairs them)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from benchmarks.common.artifacts import read_parallel_n_envs_from_run_config
from benchmarks.common.configs import load_task_config

TASK_DIR = Path(__file__).resolve().parent
FIGURES_DIR = TASK_DIR / "results" / "figures"


def _figures_dir(task_dir: Path) -> Path:
    """Return the output figure directory for the requested task root."""
    return Path(task_dir) / "results" / "figures"

SPLIT_ORDER = ["train", "iid", "load_stress", "line_tightening"]
SPLIT_LABELS = {
    "train": "Train",
    "iid": "IID",
    "load_stress": "Load Stress",
    "line_tightening": "Line Tight",
}

ALGO_ORDER = [
    "all_on",
    "merit_order",
    "ppo",
    "sac",
    "saute_ppo",
    "ppo_lagrangian",
    "ppo_penalty_l10",
    "ppo_penalty_l100",
    "ppo_penalty_l1000",
]
ALGO_LABELS = {
    "all_on": "All-On",
    "merit_order": "Merit Order",
    "ppo": "PPO",
    "sac": "SAC",
    "saute_ppo": "Sauté PPO",
    "ppo_lagrangian": "PPO-Lag",
    "ppo_penalty_l10": "Penalty PPO (λ=10)",
    "ppo_penalty_l100": "Penalty PPO (λ=100)",
    "ppo_penalty_l1000": "Penalty PPO (λ=1000)",
}

ALGO_COLORS = {
    "all_on": "#9e9e9e",
    "merit_order": "#4caf50",
    "ppo": "#2196f3",
    "sac": "#00acc1",
    "saute_ppo": "#8e24aa",
    "ppo_lagrangian": "#e91e63",
    "ppo_penalty_l10": "#ff8a65",    # orange-300  (weak penalty)
    "ppo_penalty_l100": "#f4511e",   # deep-orange-600 (moderate penalty)
    "ppo_penalty_l1000": "#bf360c",  # deep-orange-900 (strong penalty)
}

# Learning-curve dashboards: legend / draw order (must match ALGO_ORDER for ppo* block).
_LEARNING_CURVE_ALGO_RANK: dict[str, int] = {
    a: i
    for i, a in enumerate(ALGO_ORDER)
    if a in ("ppo", "sac", "saute_ppo", "ppo_lagrangian") or a.startswith("ppo_penalty")
}

COST_COLORS = {
    "gen": "#2196f3",
    "startup": "#ff9800",
    "no_load": "#9c27b0",
}

# Phase-2 overlays: same algo family shares hue; linestyle encodes GPU/CUDA vs CPU.
_CROSS_BACKEND_PPO_PYTHON = "#1565c0"   # JAX PPO
_CROSS_BACKEND_LAG_PYTHON = "#ad1457"  # JAX PPO-Lag
_CROSS_BACKEND_PPO_SB3 = "#ef6c00"
_CROSS_BACKEND_PPO_SBX = "#6a1b9a"
_CROSS_BACKEND_SAC_JAX = "#00acc1"


def _load_summary(task_dir: Path) -> Optional[dict]:
    path = task_dir / "results" / "summary" / "latest.json"
    if not path.exists():
        print(f"[TSO plots] No summary found at {path}. Run 'summarize' first.")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _default_campaign_after(task_dir: Path) -> str | None:
    try:
        task_config = load_task_config(task_dir)
    except Exception:
        return None
    protocol = task_config.get("benchmark_protocol") or {}
    value = protocol.get("current_campaign_start_iso")
    return str(value) if value else None


def _parse_iso_utc(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _record_after_campaign(record: dict, after: str | None) -> bool:
    if after is None:
        return True
    after_dt = _parse_iso_utc(after)
    if after_dt is None:
        raise ValueError(
            f"Invalid ISO timestamp for after={after!r}; "
            "expected e.g. '2026-04-24T02:05:20+00:00'."
        )
    rec_dt = _parse_iso_utc(record.get("timestamp"))
    return rec_dt is not None and rec_dt >= after_dt


def _tso_curve_legend_label(curve: dict, artifacts_dir: Path) -> str:
    algo = curve["algo"]
    seed = curve["seed"]
    run_id = str(curve.get("run_id", ""))
    base = f"{ALGO_LABELS.get(algo, algo)} (seed {seed})"
    n = read_parallel_n_envs_from_run_config(artifacts_dir, run_id)
    parts: list[str] = [base]
    if n is not None:
        parts.append(f"n_envs={n}")
    run = curve.get("run") or {}
    dev = (run.get("device") or "gpu").lower()
    backend = run.get("backend") or "jax_rejax"
    if backend != "jax_rejax" or dev != "gpu":
        parts.append(f"{backend}+{dev}")
    return ", ".join(parts)


def _get_row(rows: list[dict], algo: str, split: str) -> Optional[dict]:
    for r in rows:
        if r["algo"] == algo and r["split"] == split:
            return r
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Fig 1: NormScore grouped bar chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_normscore_bars(task_dir: Path = TASK_DIR) -> Path:
    """Grouped bar chart: NormScore per (algo, split)."""
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

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, algo in enumerate(algos):
        scores: list[float] = []
        lo_errs: list[float] = []
        hi_errs: list[float] = []
        for split in splits:
            r = _get_row(rows, algo, split)
            ns = r.get("norm_score") if r else None
            ns_f = float(ns) if ns is not None else 0.0
            scores.append(ns_f)
            ci_lo = r.get("norm_score_ci_lo") if r else None
            ci_hi = r.get("norm_score_ci_hi") if r else None
            if ci_lo is not None and ci_hi is not None:
                lo_errs.append(max(0.0, ns_f - float(ci_lo)))
                hi_errs.append(max(0.0, float(ci_hi) - ns_f))
            else:
                lo_errs.append(0.0)
                hi_errs.append(0.0)

        offset = (i - n_algos / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, scores,
            width=width * 0.9,
            label=ALGO_LABELS.get(algo, algo),
            color=ALGO_COLORS.get(algo, "#888888"),
            alpha=0.85,
            yerr=[lo_errs, hi_errs],
            capsize=3,
            error_kw={"elinewidth": 1.2, "alpha": 0.7},
        )

    ax.axhline(0, color="#9e9e9e", linewidth=0.8, linestyle="--")
    ax.axhline(1, color="#4caf50", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([SPLIT_LABELS.get(s, s) for s in splits])
    ax.set_ylabel("NormScore (IQM ± 95% bootstrap CI)")
    ax.set_title("TSO — NormScore by Algorithm and Split\n"
                 "(0 = All-On level, 1 = Merit-Order level)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(-0.3, 1.5)
    fig.tight_layout()

    out = figures_dir / "normscore_bars.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[TSO plots] saved {out}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Fig 2: Gantt chart — unit commitment timeline for one episode
# ──────────────────────────────────────────────────────────────────────────────

def plot_gantt_commitment(task_dir: Path = TASK_DIR) -> Path:
    """Gantt chart showing unit on/off timeline for merit_order vs PPO.

    Runs a fresh rollout with the merit_order baseline and (if available)
    the best PPO policy to visualize commitment schedules.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import jax

    figures_dir = _figures_dir(task_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    try:
        from powerzoojax.envs.grid.unit_commitment import UnitCommitmentEnv
        from powerzoojax.tasks.tso import (
            make_tso_case118_params,
            tso_merit_order_rollout,
        )

        params = make_tso_case118_params(max_steps=48)
        env = UnitCommitmentEnv()
        key = jax.random.PRNGKey(42)

        # Collect commitment schedule for merit_order
        obs, state = env.reset(key, params)
        case = params.case
        n_units = case.n_units
        mc_b_np = np.asarray(case.unit_cost_b)
        p_max_np = np.asarray(case.unit_p_max)
        merit_order = np.argsort(mc_b_np)

        schedules = {"merit_order": []}

        for step in range(params.max_steps):
            t_idx = step % params.load_profiles.shape[0]
            load_demand = np.asarray(params.load_profiles[t_idx])
            total_load = float(np.sum(np.asarray(case.nodes_loads_map) @ load_demand))
            required_cap = total_load * (1.0 + params.reserve_margin_frac)

            commit_np = np.zeros(n_units, dtype=np.float32)
            cumcap = 0.0
            for i in merit_order:
                commit_np[i] = 1.0
                cumcap += p_max_np[i]
                if cumcap >= required_cap:
                    break

            schedules["merit_order"].append(commit_np.copy())
            commit_signal = commit_np * 2.0 - 1.0
            import jax.numpy as jnp
            action = jnp.concatenate([
                jnp.array(commit_signal, dtype=jnp.float32),
                jnp.zeros(n_units, dtype=jnp.float32),
            ])
            key, k = jax.random.split(key)
            obs, state, _, _, _, _ = env.step(k, state, action, params)

        schedules_arr = {
            algo: np.stack(v, axis=0)  # (T, n_units)
            for algo, v in schedules.items()
        }

        # Plot: show top 20 units (by merit order) for clarity
        n_show = min(20, n_units)
        units_to_show = merit_order[:n_show]

        fig, axes = plt.subplots(1, 1, figsize=(12, 6))
        ax = axes
        algo = "merit_order"
        schedule = schedules_arr[algo][:, units_to_show]  # (T, n_show)

        T = schedule.shape[0]
        for u_idx, u in enumerate(units_to_show):
            for t in range(T):
                if schedule[t, u_idx] > 0.5:
                    ax.broken_barh(
                        [(t, 1)], (u_idx - 0.4, 0.8),
                        facecolors=ALGO_COLORS[algo],
                        alpha=0.8,
                    )

        ax.set_yticks(range(n_show))
        ax.set_yticklabels([f"Unit {units_to_show[i]+1}" for i in range(n_show)],
                           fontsize=7)
        ax.set_xlabel("Time step (30 min)")
        ax.set_title(f"TSO — Commitment Schedule: {ALGO_LABELS[algo]}")
        ax.set_xlim(0, T)
        ax.set_ylim(-0.5, n_show - 0.5)

        fig.tight_layout()
        out = figures_dir / "gantt_commitment.pdf"
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[TSO plots] saved {out}")
        return out

    except Exception as exc:
        print(f"[TSO plots] Gantt chart failed ({exc}), skipping")
        return Path()


# ──────────────────────────────────────────────────────────────────────────────
# Fig 3: Cost decomposition stacked bar
# ──────────────────────────────────────────────────────────────────────────────

def plot_cost_decomposition(task_dir: Path = TASK_DIR) -> Path:
    """Stacked bar chart: gen / startup / no-load cost per algo on train split."""
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
    gen_costs = []
    startup_costs = []
    noload_costs = []

    for algo in algos:
        r = _get_row(rows, algo, "train")
        gen_costs.append(r.get("total_gen_cost_mean") or 0.0)
        startup_costs.append(r.get("total_startup_cost_mean") or 0.0)
        noload_costs.append(r.get("total_no_load_cost_mean") or 0.0)

    x = np.arange(len(algos))
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(x, gen_costs, label="Generation Cost",
           color=COST_COLORS["gen"], alpha=0.85)
    ax.bar(x, startup_costs, bottom=gen_costs, label="Startup Cost",
           color=COST_COLORS["startup"], alpha=0.85)
    ax.bar(x, noload_costs,
           bottom=[g + s for g, s in zip(gen_costs, startup_costs)],
           label="No-Load Cost", color=COST_COLORS["no_load"], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([ALGO_LABELS.get(a, a) for a in algos])
    ax.set_ylabel("Operating Cost (Train Split)")
    ax.set_title("TSO — Cost Decomposition (Train Split)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    out = figures_dir / "cost_decomposition.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[TSO plots] saved {out}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Fig 4: Learning curves
# ──────────────────────────────────────────────────────────────────────────────

def _canonical_train_runs(
    task_dir: Path,
    *,
    after: str | None = None,
) -> dict[tuple[str, int], dict]:
    """Map ``(algo, seed)`` to the training record used for learning-curve figures.

    **Selection policy (Phase-1 dashboard fairness):**

    - Only ``backend=jax_rejax`` train rows (SB3/SBX cross-backend curves need
      dedicated plots; mixing them here breaks the PPO vs PPO-Lag comparison).
    - When both JAX GPU and JAX CPU exist for the same ``(algo, seed)`` — e.g.
      Phase-2 CPU reruns with capped ``num_envs`` — **always prefer GPU** so
      walltime/n_envs match between algos on the same figure. Among ties,
      prefer the newest timestamp.
    """
    manifest_path = task_dir / "results" / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        records = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    def _rank(record: dict) -> tuple[int, str]:
        dev = (record.get("device") or "gpu").lower()
        gpu_first = 1 if dev == "gpu" else 0
        ts = str(record.get("timestamp") or "")
        return (gpu_first, ts)

    if after is None:
        after = _default_campaign_after(task_dir)

    out: dict[tuple[str, int], dict] = {}
    for record in records:
        if not _record_after_campaign(record, after):
            continue
        if record.get("split") != "train":
            continue
        artifacts = record.get("artifacts") or {}
        if "params" not in artifacts and "params_orbax" not in artifacts:
            continue
        backend = record.get("backend") or "jax_rejax"
        if backend != "jax_rejax":
            continue
        key = (record.get("algo"), int(record.get("seed", -1)))
        current = out.get(key)
        if current is None or _rank(record) > _rank(current):
            out[key] = record
    return out


def _load_curve_array(
    artifacts_dir: Path,
    run_id: str,
    stem: str,
) -> np.ndarray | None:
    path = artifacts_dir / f"{run_id}_{stem}.npy"
    if not path.exists():
        return None
    try:
        return np.asarray(np.load(path))
    except Exception:
        return None


def _curve_x(
    run: dict,
    artifacts_dir: Path,
    n_points: int,
    *,
    axis_kind: str,
) -> np.ndarray | None:
    if n_points <= 0:
        return None

    run_id = str(run.get("run_id", ""))
    if axis_kind == "timesteps":
        timesteps = _load_curve_array(artifacts_dir, run_id, "timesteps")
        if timesteps is not None and timesteps.size == n_points:
            return timesteps.astype(float)
        total_timesteps = float((run.get("metrics") or {}).get("total_timesteps") or 0.0)
        if total_timesteps > 0:
            return np.linspace(0.0, total_timesteps, n_points)
        return np.arange(n_points, dtype=float)

    if axis_kind == "walltime":
        eval_walltimes = _load_curve_array(
            artifacts_dir, run_id, "learning_curve_eval_walltimes"
        )
        if eval_walltimes is not None and eval_walltimes.size == n_points:
            # Rejax: seconds since t0 (after optional warm-up) at each eval — already
            # monotonic; not per-eval Δt.  See ``save_training_artifacts`` contract.
            return eval_walltimes.astype(float)
        walltime_s = float(run.get("walltime_s") or 0.0)
        if walltime_s > 0:
            return np.linspace(0.0, walltime_s, n_points)
        return np.arange(n_points, dtype=float)

    raise ValueError(f"Unknown axis_kind={axis_kind!r}")


def _smooth_curve(values: np.ndarray, window: int | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size < 5:
        return values
    if window is None:
        window = max(5, values.size // 20)
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    if window >= values.size:
        window = values.size - 1 if values.size % 2 == 0 else values.size
    if window < 3:
        return values
    pad = window // 2
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _group_curves_by_algo(curves: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for c in curves:
        out.setdefault(c["algo"], []).append(c)
    return out


def _aligned_metric_stack(
    group: list[dict],
    axis_kind: str,
    y_key: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """If all curves share identical x and y length, return ``x`` and ``Y`` (n_seed, T)."""
    if len(group) < 2:
        return None, None
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    t0: int | None = None
    for c in group:
        x = c.get(f"x_{axis_kind}")
        y = c.get(y_key)
        if x is None or y is None:
            return None, None
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        if t0 is None:
            t0 = int(xa.size)
        elif int(xa.size) != t0 or int(ya.size) != t0:
            return None, None
        xs.append(xa)
        ys.append(ya)
    x0 = xs[0]
    atol = max(1.0, float(np.max(np.abs(x0))) * 1e-9)
    for xa in xs[1:]:
        if not np.allclose(xa, x0, rtol=0.0, atol=atol):
            return None, None
    return x0, np.stack(ys, axis=0)


def _plot_seed_ribbon_eval_cost(
    ax,
    curves: list[dict],
    *,
    axis_kind: str,
    running_best: bool,
) -> None:
    """Mean ± std band per algorithm when multiple seeds share the same eval grid."""
    x_divisor = 1e6 if axis_kind == "timesteps" else 60.0
    for algo, group in _group_curves_by_algo(curves).items():
        x, stack = _aligned_metric_stack(group, axis_kind, "eval_cost")
        if x is None or stack is None:
            continue
        stack_mgbp = stack / 1e6
        if running_best:
            stack_mgbp = np.minimum.accumulate(stack_mgbp, axis=1)
        mean = np.mean(stack_mgbp, axis=0)
        std = np.std(stack_mgbp, axis=0)
        x_plot = np.asarray(x, dtype=float) / x_divisor
        color = ALGO_COLORS.get(algo, "#888888")
        ax.fill_between(
            x_plot,
            mean - std,
            mean + std,
            color=color,
            alpha=0.2,
            linewidth=0.0,
        )


def _eval_cost_curve(
    artifacts_dir: Path,
    run_id: str,
    reward_scale: float,
) -> np.ndarray | None:
    cost_curve = _load_curve_array(artifacts_dir, run_id, "eval_total_operating_cost")
    if cost_curve is not None:
        return cost_curve.astype(float)

    eval_returns = _load_curve_array(
        artifacts_dir, run_id, "learning_curve_eval_return"
    )
    if eval_returns is None:
        eval_returns = _load_curve_array(artifacts_dir, run_id, "eval_returns")
    if eval_returns is None:
        return None
    return -np.asarray(eval_returns, dtype=float) / reward_scale


def _run_config_snapshot(artifacts_dir: Path, run_id: str) -> dict:
    path = artifacts_dir / f"{run_id}_config.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_cost_scale(snapshot: dict) -> float:
    raw = (
        (snapshot.get("train_config_raw") or {}).get("cost_scale")
        or (snapshot.get("train_config_resolved") or {}).get("cost_scale")
        or 1.0
    )
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else 1.0
    return float(raw)


def _style_axis(ax) -> None:
    ax.set_facecolor("#fffaf3")
    ax.grid(True, color="#cfc3b1", alpha=0.35, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_alpha(0.35)


def _annotate_reference_lines(ax, refs_million_gbp: dict[str, float]) -> None:
    reference_specs = (
        ("all_on", "All-On", "#8f8f8f", "--"),
        ("merit_order", "Merit Order", "#2f855a", ":"),
    )
    for key, label, color, linestyle in reference_specs:
        value = refs_million_gbp.get(key)
        if value is None:
            continue
        ax.axhline(value, color=color, linestyle=linestyle, linewidth=1.25, alpha=0.9)
        ax.text(
            0.99,
            value,
            label,
            color=color,
            fontsize=8,
            ha="right",
            va="bottom",
            transform=ax.get_yaxis_transform(),
            bbox={"facecolor": "#fffaf3", "edgecolor": "none", "pad": 0.4},
        )


def _plot_eval_cost_panel(
    ax,
    curves: list[dict],
    *,
    axis_kind: str,
    running_best: bool,
    refs_million_gbp: dict[str, float],
    artifacts_dir: Path,
) -> None:
    _style_axis(ax)
    x_divisor = 1e6 if axis_kind == "timesteps" else 60.0
    x_label = "Timesteps (millions)" if axis_kind == "timesteps" else "Walltime (minutes)"

    _plot_seed_ribbon_eval_cost(
        ax, curves, axis_kind=axis_kind, running_best=running_best
    )

    for curve in curves:
        x = curve.get(f"x_{axis_kind}")
        y = curve.get("eval_cost")
        if x is None or y is None or len(x) != len(y) or len(y) == 0:
            continue
        y_m = np.asarray(y, dtype=float) / 1e6
        x = np.asarray(x, dtype=float) / x_divisor
        viol = curve.get("eval_reserve_shortfall_rate")
        color = ALGO_COLORS.get(curve["algo"], "#888888")
        markevery = max(1, len(y_m) // 12)

        if running_best:
            y_to_plot = np.minimum.accumulate(y_m)
        else:
            ax.plot(x, y_m, color=color, alpha=0.22, linewidth=1.0)
            y_to_plot = _smooth_curve(y_m)

        ax.plot(
            x,
            y_to_plot,
            color=color,
            linewidth=2.3,
            alpha=0.95,
            label=_tso_curve_legend_label(curve, artifacts_dir),
            marker="o",
            markersize=3.2,
            markevery=markevery,
        )
        ax.scatter(x[-1], y_to_plot[-1], color=color, s=24, zorder=3)

        if viol is not None and len(viol) == len(y):
            vm = np.asarray(viol, dtype=float) > 1e-6
            if np.any(vm):
                ax.scatter(
                    x[vm],
                    y_to_plot[vm],
                    s=42,
                    marker="X",
                    color=color,
                    edgecolors="#b71c1c",
                    linewidths=0.6,
                    zorder=6,
                    alpha=0.95,
                )

    _annotate_reference_lines(ax, refs_million_gbp)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Episode operating cost (£M)")
    ax.set_title(
        "Lowest Eval Cost Found So Far"
        if running_best
        else "Eval Cost During Training"
    )
    ax.text(
        0.01,
        0.02,
        "× = checkpoint with reserve shortfall on ≥1 eval episode",
        transform=ax.transAxes,
        fontsize=7,
        color="#5d4037",
        va="bottom",
    )
    # Per-panel legend suppressed: a single shared legend is added at the
    # figure top by plot_learning_curves so the dense dashboard stays readable.


def _plot_metric_panel(
    ax,
    curves: list[dict],
    *,
    axis_kind: str,
    metric_key: str,
    title: str,
    ylabel: str,
    artifacts_dir: Path,
    running_best: bool = False,
    lower_is_better: bool = True,
    ymin: float | None = None,
    ymax: float | None = None,
) -> None:
    _style_axis(ax)
    x_divisor = 1e6 if axis_kind == "timesteps" else 60.0
    x_label = "Timesteps (millions)" if axis_kind == "timesteps" else "Walltime (minutes)"

    for curve in curves:
        x = curve.get(f"x_{axis_kind}")
        y = curve.get(metric_key)
        if x is None or y is None or len(x) != len(y) or len(y) == 0:
            continue
        x = np.asarray(x, dtype=float) / x_divisor
        y = np.asarray(y, dtype=float)
        if running_best:
            if lower_is_better:
                y_to_plot = np.minimum.accumulate(y)
            else:
                y_to_plot = np.maximum.accumulate(y)
        else:
            ax.plot(x, y, color=ALGO_COLORS.get(curve["algo"], "#888888"), alpha=0.22, linewidth=1.0)
            y_to_plot = _smooth_curve(y)

        color = ALGO_COLORS.get(curve["algo"], "#888888")
        markevery = max(1, len(y_to_plot) // 12)
        ax.plot(
            x,
            y_to_plot,
            color=color,
            linewidth=2.3,
            alpha=0.95,
            label=_tso_curve_legend_label(curve, artifacts_dir),
            marker="o",
            markersize=3.2,
            markevery=markevery,
        )
        ax.scatter(x[-1], y_to_plot[-1], color=color, s=24, zorder=3)

    ax.set_xlabel(x_label)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ymin is not None or ymax is not None:
        lo, hi = ax.get_ylim()
        ax.set_ylim(ymin if ymin is not None else lo, ymax if ymax is not None else hi)
    # Per-panel legend suppressed: shared figure-level legend handles labelling.


def _plot_cmdp_diagnostics(
    ax_cost,
    ax_lambda,
    *,
    run: dict,
    artifacts_dir: Path,
) -> None:
    _style_axis(ax_cost)
    _style_axis(ax_lambda)

    run_id = str(run.get("run_id", ""))
    x_cost = None
    x_lambda = None
    snapshot = _run_config_snapshot(artifacts_dir, run_id)
    cost_scale = _safe_cost_scale(snapshot)

    reserve_cost = _load_curve_array(
        artifacts_dir, run_id, "episode_cost_est_reserve_shortfall"
    )
    thermal_cost = _load_curve_array(
        artifacts_dir, run_id, "episode_cost_est_thermal_overload"
    )
    reserve_lambda = _load_curve_array(artifacts_dir, run_id, "lambda_reserve_shortfall")
    thermal_lambda = _load_curve_array(artifacts_dir, run_id, "lambda_thermal_overload")

    if reserve_cost is not None:
        x_cost = _curve_x(run, artifacts_dir, reserve_cost.size, axis_kind="timesteps")
        rc = np.asarray(reserve_cost, dtype=float) * cost_scale
        ax_cost.plot(
            x_cost / 1e6,
            rc,
            color="#ef6c00",
            linewidth=1.0,
            alpha=0.35,
            label="Reserve (raw update)",
        )
        ax_cost.plot(
            x_cost / 1e6,
            _smooth_curve(rc),
            color="#ef6c00",
            linewidth=2.2,
            alpha=0.95,
            label="Reserve (smoothed)",
        )
    if thermal_cost is not None and np.any(np.abs(thermal_cost) > 1e-12):
        if x_cost is None:
            x_cost = _curve_x(run, artifacts_dir, thermal_cost.size, axis_kind="timesteps")
        tc = np.asarray(thermal_cost, dtype=float) * cost_scale
        ax_cost.plot(
            x_cost / 1e6,
            tc,
            color="#c62828",
            linewidth=1.0,
            alpha=0.35,
            label="Thermal (raw)",
        )
        ax_cost.plot(
            x_cost / 1e6,
            _smooth_curve(tc),
            color="#c62828",
            linewidth=2.2,
            alpha=0.95,
            label="Thermal (smoothed)",
        )
    else:
        ax_cost.text(
            0.98,
            0.08,
            "Thermal overload remained zero",
            transform=ax_cost.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#8d6e63",
        )

    if reserve_lambda is not None:
        x_lambda = _curve_x(
            run, artifacts_dir, reserve_lambda.size, axis_kind="timesteps"
        )
        ax_lambda.plot(
            x_lambda / 1e6,
            np.asarray(reserve_lambda, dtype=float),
            color="#ef6c00",
            linewidth=2.0,
            label="Reserve Shortfall",
        )
    if thermal_lambda is not None and np.max(np.abs(thermal_lambda - thermal_lambda[0])) > 1e-8:
        if x_lambda is None:
            x_lambda = _curve_x(
                run, artifacts_dir, thermal_lambda.size, axis_kind="timesteps"
            )
        ax_lambda.plot(
            x_lambda / 1e6,
            np.asarray(thermal_lambda, dtype=float),
            color="#c62828",
            linewidth=2.0,
            label="Thermal Overload",
        )
    else:
        ax_lambda.text(
            0.98,
            0.08,
            "Thermal multiplier stayed at its initial value",
            transform=ax_lambda.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#8d6e63",
        )

    ax_cost.set_title(
        "PPO-Lag — CMDP cost estimate (on-policy rollouts, not eval violation rate)"
    )
    ax_cost.set_xlabel("Timesteps (millions)")
    ax_cost.set_ylabel(
        "Estimated episode constraint cost (reserve / thermal, scaled; arb. units)"
    )
    ax_cost.text(
        0.01,
        0.02,
        "Noisy: stochastic policy during training. Compare dashboard panels for "
        "greedy checkpoint eval.",
        transform=ax_cost.transAxes,
        fontsize=7,
        color="#5d4037",
        va="bottom",
    )
    handles, labels = ax_cost.get_legend_handles_labels()
    if handles:
        ax_cost.legend(fontsize=7, loc="upper right")

    ax_lambda.set_title("PPO-Lag — Lagrange multipliers λ (dual variables)")
    ax_lambda.set_xlabel("Timesteps (millions)")
    ax_lambda.set_ylabel("λ = exp(log λ) (higher = stronger penalty on cost)")
    handles, labels = ax_lambda.get_legend_handles_labels()
    if handles:
        ax_lambda.legend(fontsize=7, loc="upper right")


def _training_update_timesteps(
    artifacts_dir: Path,
    run_id: str,
    n_updates: int,
) -> np.ndarray | None:
    """Env timesteps at the end of each PPO update (num_envs × n_steps per update)."""
    if n_updates <= 0:
        return None
    snapshot = _run_config_snapshot(artifacts_dir, run_id)
    raw = snapshot.get("train_config_raw") or {}
    total = int(raw.get("total_timesteps", 0) or 0)
    num_envs = int(raw.get("num_envs", 256))
    n_steps = int(raw.get("n_steps", 48))
    if total <= 0:
        return None
    per = float(num_envs * n_steps)
    t = np.minimum(
        np.arange(1, n_updates + 1, dtype=np.float64) * per,
        float(total),
    )
    return t


def _plot_ppo_lag_training_cmdp_reserve_row(
    ax,
    curves: list[dict],
    artifacts_dir: Path,
) -> None:
    """Dense on-policy constraint signal during PPO-Lag training (not checkpoint eval)."""
    _style_axis(ax)
    any_plotted = False
    for curve in curves:
        if curve.get("algo") != "ppo_lagrangian":
            continue
        run_id = str(curve.get("run_id", ""))
        est = _load_curve_array(
            artifacts_dir, run_id, "episode_cost_est_reserve_shortfall"
        )
        if est is None or est.size == 0:
            continue
        est = np.asarray(est, dtype=float)
        cost_scale = _safe_cost_scale(_run_config_snapshot(artifacts_dir, run_id))
        y = est * cost_scale
        x_steps = _training_update_timesteps(artifacts_dir, run_id, int(est.size))
        if x_steps is None or x_steps.shape[0] != y.shape[0]:
            snap = _run_config_snapshot(artifacts_dir, run_id)
            total = int((snap.get("train_config_raw") or {}).get("total_timesteps", 0) or 0)
            if total <= 0:
                continue
            x_steps = np.linspace(0.0, float(total), int(est.size), dtype=np.float64)
        x_m = x_steps / 1e6
        color = ALGO_COLORS.get("ppo_lagrangian", "#e91e63")
        _n = read_parallel_n_envs_from_run_config(artifacts_dir, run_id)
        _nt = f", n_envs={_n}" if _n is not None else ""
        label_base = (
            f"{ALGO_LABELS.get('ppo_lagrangian', 'PPO-Lag')} (seed {curve.get('seed', '')}){_nt}"
        )
        ax.plot(x_m, y, color=color, linewidth=1.0, alpha=0.28)
        ax.plot(
            x_m,
            _smooth_curve(y),
            color=color,
            linewidth=2.2,
            alpha=0.95,
        )
        any_plotted = True

    ax.set_title(
        "PPO-Lag only — on-policy training reserve CMDP cost (per policy update)"
    )
    ax.set_xlabel("Environment timesteps (millions)")
    ax.set_ylabel("Reserve CMDP cost (training rollout)")
    if not any_plotted:
        ax.text(
            0.5,
            0.5,
            "No episode_cost_est_reserve_shortfall artifact for canonical PPO-Lag run(s).",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            color="#6d4c41",
        )
    else:
        ax.text(
            0.01,
            0.02,
            "Noisy: parallel envs + stochastic policy. Upper rows use sparse greedy eval "
            "for fair comparison with PPO.",
            transform=ax.transAxes,
            fontsize=7,
            color="#5d4037",
            va="bottom",
        )
        from matplotlib.lines import Line2D
        ax.legend(
            handles=[
                Line2D([0], [0], color=color, linewidth=1.0, alpha=0.6,
                       label="PPO-Lag (raw, 5 seeds)"),
                Line2D([0], [0], color=color, linewidth=2.2,
                       label="PPO-Lag (smoothed, 5 seeds)"),
            ],
            fontsize=10, loc="upper right", frameon=False,
        )


def _plot_final_checkpoint_scatter(
    curves: list[dict],
    out_pdf: Path,
    *,
    dpi: int = 200,
    artifacts_dir: Path,
) -> None:
    """Scatter: last-checkpoint eval cost vs reserve shortfall rate (per run)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    _style_axis(ax)
    seen_labels: set[str] = set()
    for curve in curves:
        cost = curve.get("eval_cost")
        viol = curve.get("eval_reserve_shortfall_rate")
        if cost is None or viol is None or np.asarray(cost).size == 0:
            continue
        cost_a = np.asarray(cost, dtype=float)
        viol_a = np.asarray(viol, dtype=float)
        if viol_a.size != cost_a.size:
            continue
        c_last = float(cost_a[-1]) / 1e6
        v_last = float(viol_a[-1])
        color = ALGO_COLORS.get(curve["algo"], "#888888")
        label = _tso_curve_legend_label(curve, artifacts_dir)
        use_label = label if label not in seen_labels else None
        seen_labels.add(label)
        ax.scatter(
            v_last,
            c_last,
            color=color,
            s=96,
            edgecolors="#263238",
            linewidths=0.5,
            zorder=4,
            label=use_label,
        )
    ax.set_xlabel(
        "Reserve shortfall rate at final checkpoint\n"
        "(fraction of monitored eval episodes with shortfall)"
    )
    ax.set_ylabel("Eval operating cost at final checkpoint (million GBP)")
    ax.set_title("TSO — Final checkpoint: economic cost vs reserve feasibility")
    ax.set_xlim(-0.05, 1.05)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(str(out_pdf).replace(".pdf", ".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[TSO plots] saved {out_pdf}")


def plot_learning_curves(
    task_dir: Path = TASK_DIR,
    *,
    after: str | None = None,
) -> Path:
    """Generate timestep and walltime learning-curve dashboards."""
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

    figures_dir = _figures_dir(task_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = task_dir / "results" / "artifacts"
    if not artifacts_dir.exists():
        print("[TSO plots] No artifacts directory found. Train models first.")
        return Path()

    task_config = load_task_config(task_dir)
    if after is None:
        after = _default_campaign_after(task_dir)
    reward_scale = float(task_config.get("reward_scale", 1e-4))
    canonical = _canonical_train_runs(task_dir, after=after)
    curves: list[dict] = []

    for (algo, seed), run in sorted(canonical.items()):
        run_id = str(run.get("run_id", ""))
        eval_cost = _eval_cost_curve(artifacts_dir, run_id, reward_scale)
        if eval_cost is None or eval_cost.size == 0:
            continue
        curves.append({
            "algo": algo,
            "seed": seed,
            "run_id": run_id,
            "run": run,
            "eval_cost": np.asarray(eval_cost, dtype=float),
            "eval_reserve_shortfall_rate": _load_curve_array(
                artifacts_dir, run_id, "eval_reserve_shortfall_rate"
            ),
            "eval_total_reserve_shortfall": _load_curve_array(
                artifacts_dir, run_id, "eval_total_reserve_shortfall"
            ),
            "eval_thermal_violation_rate": _load_curve_array(
                artifacts_dir, run_id, "eval_thermal_violation_rate"
            ),
            "eval_total_thermal_overload": _load_curve_array(
                artifacts_dir, run_id, "eval_total_thermal_overload"
            ),
            "x_timesteps": _curve_x(
                run, artifacts_dir, len(eval_cost), axis_kind="timesteps"
            ),
            "x_walltime": _curve_x(
                run, artifacts_dir, len(eval_cost), axis_kind="walltime"
            ),
        })

    curves.sort(
        key=lambda c: (_LEARNING_CURVE_ALGO_RANK.get(c["algo"], 99), c["seed"]),
    )

    if not curves:
        print("[TSO plots] No learning-curve artifacts found for canonical runs.")
        return Path()

    summary = _load_summary(task_dir) or {}
    rows = summary.get("rows", [])
    refs_million_gbp = {}
    for algo in ("all_on", "merit_order"):
        row = _get_row(rows, algo, "train")
        if row and row.get("total_operating_cost_mean") is not None:
            refs_million_gbp[algo] = float(row["total_operating_cost_mean"]) / 1e6

    curve_png_dpi = 200
    fig_ts, axes_ts = plt.subplots(
        4, 2, figsize=(14, 13.6), squeeze=False, constrained_layout=True
    )
    _plot_eval_cost_panel(
        axes_ts[0, 0],
        curves,
        axis_kind="timesteps",
        running_best=True,
        refs_million_gbp=refs_million_gbp,
        artifacts_dir=artifacts_dir,
    )
    _plot_eval_cost_panel(
        axes_ts[0, 1],
        curves,
        axis_kind="timesteps",
        running_best=False,
        refs_million_gbp=refs_million_gbp,
        artifacts_dir=artifacts_dir,
    )
    axes_ts[0, 0].text(
        0.01,
        0.98,
        "Left: best eval cost so far. Right: raw checkpoint eval. "
        "Shaded: mean ± std when eval grids align.",
        transform=axes_ts[0, 0].transAxes,
        fontsize=7,
        color="#5d4037",
        va="top",
    )
    _plot_metric_panel(
        axes_ts[1, 0],
        curves,
        axis_kind="timesteps",
        metric_key="eval_reserve_shortfall_rate",
        title="Eval — Reserve Shortfall Incidence",
        ylabel="Reserve shortfall rate [0, 1]",
        artifacts_dir=artifacts_dir,
        lower_is_better=True,
        ymin=-0.02,
        ymax=1.02,
    )
    axes_ts[1, 0].text(
        0.01,
        0.98,
        "PPO-Lag: smooth eval curve vs noisy training rollout → see bottom row.",
        transform=axes_ts[1, 0].transAxes,
        fontsize=7,
        color="#5d4037",
        va="top",
    )
    _plot_metric_panel(
        axes_ts[1, 1],
        curves,
        axis_kind="timesteps",
        metric_key="eval_total_reserve_shortfall",
        title="Eval — Reserve Shortfall Magnitude",
        ylabel="Σ reserve shortfall (MW·step / ep)",
        artifacts_dir=artifacts_dir,
        lower_is_better=True,
        ymin=0.0,
    )
    _plot_metric_panel(
        axes_ts[2, 0],
        curves,
        axis_kind="timesteps",
        metric_key="eval_thermal_violation_rate",
        title="Eval — Thermal Overload Incidence",
        ylabel="Thermal overload rate [0, 1]",
        artifacts_dir=artifacts_dir,
        lower_is_better=True,
        ymin=-0.02,
        ymax=1.02,
    )
    _plot_metric_panel(
        axes_ts[2, 1],
        curves,
        axis_kind="timesteps",
        metric_key="eval_total_thermal_overload",
        title="Eval — Thermal Overload Magnitude",
        ylabel="Σ thermal overload cost / ep",
        artifacts_dir=artifacts_dir,
        lower_is_better=True,
        ymin=0.0,
    )

    gs_ts = axes_ts[3, 0].get_gridspec()
    axes_ts[3, 0].remove()
    axes_ts[3, 1].remove()
    ax_ppo_lag_train = fig_ts.add_subplot(gs_ts[3, :])
    _plot_ppo_lag_training_cmdp_reserve_row(ax_ppo_lag_train, curves, artifacts_dir)

    # Suptitle removed: the LaTeX figure caption already identifies the dashboard,
    # and dropping it frees the top strip for a clean shared legend.
    from matplotlib.lines import Line2D
    algos_in_figure = sorted({c["algo"] for c in curves},
                             key=lambda a: _LEARNING_CURVE_ALGO_RANK.get(a, 99))
    fig_ts.legend(
        handles=[
            Line2D([0], [0], color=ALGO_COLORS.get(a, "#888"),
                   linewidth=2.4, marker="o", markersize=4,
                   label=f"{ALGO_LABELS.get(a, a)} (5 seeds)")
            for a in algos_in_figure
        ],
        loc="outside upper center",
        ncol=len(algos_in_figure),
        fontsize=12,
        frameon=False,
    )

    out = figures_dir / "learning_curves.pdf"
    out_ts = figures_dir / "learning_curves_timesteps.pdf"
    fig_ts.savefig(out, bbox_inches="tight")
    fig_ts.savefig(str(out).replace(".pdf", ".png"), dpi=curve_png_dpi, bbox_inches="tight")
    fig_ts.savefig(out_ts, bbox_inches="tight")
    fig_ts.savefig(str(out_ts).replace(".pdf", ".png"), dpi=curve_png_dpi, bbox_inches="tight")
    plt.close(fig_ts)

    fig_wt, axes_wt = plt.subplots(3, 2, figsize=(14, 10.8), squeeze=False)
    # Lead with best-so-far envelopes: raw checkpoint eval is noisy (esp. PPO-Lag + dual),
    # and non-monotonic curves read as "not converged" in papers.
    _plot_eval_cost_panel(
        axes_wt[0, 0],
        curves,
        axis_kind="walltime",
        running_best=True,
        refs_million_gbp=refs_million_gbp,
        artifacts_dir=artifacts_dir,
    )
    _plot_eval_cost_panel(
        axes_wt[0, 1],
        curves,
        axis_kind="walltime",
        running_best=False,
        refs_million_gbp=refs_million_gbp,
        artifacts_dir=artifacts_dir,
    )
    axes_wt[0, 0].text(
        0.01,
        0.98,
        "Left: best eval cost so far (non-increasing). Right: raw checkpoint eval.",
        transform=axes_wt[0, 0].transAxes,
        fontsize=7,
        color="#5d4037",
        va="top",
    )
    _plot_metric_panel(
        axes_wt[1, 0],
        curves,
        axis_kind="walltime",
        metric_key="eval_reserve_shortfall_rate",
        title="Eval — Reserve Shortfall Incidence",
        ylabel="Reserve shortfall rate [0, 1]",
        artifacts_dir=artifacts_dir,
        lower_is_better=True,
        ymin=-0.02,
        ymax=1.02,
    )
    _plot_metric_panel(
        axes_wt[1, 1],
        curves,
        axis_kind="walltime",
        metric_key="eval_total_reserve_shortfall",
        title="Eval — Reserve Shortfall Magnitude",
        ylabel="Σ reserve shortfall (MW·step / ep)",
        artifacts_dir=artifacts_dir,
        lower_is_better=True,
        ymin=0.0,
    )
    _plot_metric_panel(
        axes_wt[2, 0],
        curves,
        axis_kind="walltime",
        metric_key="eval_thermal_violation_rate",
        title="Eval — Thermal Overload Incidence",
        ylabel="Thermal overload rate [0, 1]",
        artifacts_dir=artifacts_dir,
        lower_is_better=True,
        ymin=-0.02,
        ymax=1.02,
    )
    _plot_metric_panel(
        axes_wt[2, 1],
        curves,
        axis_kind="walltime",
        metric_key="eval_total_thermal_overload",
        title="Eval — Thermal Overload Magnitude",
        ylabel="Σ thermal overload cost / ep",
        artifacts_dir=artifacts_dir,
        lower_is_better=True,
        ymin=0.0,
    )
    fig_wt.suptitle(
        "TSO — Training Dashboard (Walltime)",
        fontsize=16,
        y=1.0,
    )
    fig_wt.tight_layout(rect=(0.0, 0.03, 1.0, 0.96))
    fig_wt.text(
        0.5,
        0.015,
        "X axis is minutes = saved eval wall-clock seconds / 60. With multiple algorithms, "
        "the axis range is set by the longest-walltime run; shorter-budget curves may cluster "
        "on the left. PPO-Lag on-policy safety costs are shown in the Timesteps dashboard.",
        ha="center",
        fontsize=7.5,
        color="#5d4037",
    )

    out_wt = figures_dir / "learning_curves_walltime.pdf"
    fig_wt.savefig(out_wt, bbox_inches="tight")
    fig_wt.savefig(str(out_wt).replace(".pdf", ".png"), dpi=curve_png_dpi, bbox_inches="tight")
    plt.close(fig_wt)

    out_scatter = figures_dir / "learning_curves_final_scatter.pdf"
    _plot_final_checkpoint_scatter(
        curves, out_scatter, dpi=curve_png_dpi, artifacts_dir=artifacts_dir,
    )

    safe_curve = next((c for c in curves if c["algo"] == "ppo_lagrangian"), None)
    if safe_curve is not None:
        fig_diag, axes_diag = plt.subplots(1, 2, figsize=(14, 4.8), squeeze=False)
        _plot_cmdp_diagnostics(
            axes_diag[0, 0],
            axes_diag[0, 1],
            run=safe_curve["run"],
            artifacts_dir=artifacts_dir,
        )
        fig_diag.suptitle(
            "TSO — PPO-Lag Internal Safety Diagnostics",
            fontsize=15,
            y=1.03,
        )
        fig_diag.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        out_diag = figures_dir / "ppo_lag_diagnostics.pdf"
        fig_diag.savefig(out_diag, bbox_inches="tight")
        fig_diag.savefig(str(out_diag).replace(".pdf", ".png"), dpi=180, bbox_inches="tight")
        plt.close(fig_diag)
        print(f"[TSO plots] saved {out_diag}")

    print(f"[TSO plots] saved {out}")
    print(f"[TSO plots] saved {out_wt}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Fig 5: OOD robustness — NormScore across IID / Load Stress / Line Tightening
# ──────────────────────────────────────────────────────────────────────────────

def plot_ood_robustness(task_dir: Path = TASK_DIR) -> Path:
    """Bar chart: NormScore across all OOD splits per algorithm.

    Shows how each algo degrades from IID to Load Stress / Line Tightening.
    Baselines included as reference. 95% bootstrap CI shown.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = _load_summary(task_dir)
    if summary is None:
        return Path()

    rows = summary["rows"]
    figures_dir = _figures_dir(task_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    ood_splits = ["iid", "load_stress", "line_tightening"]
    algos = [a for a in ALGO_ORDER if any(r["algo"] == a for r in rows)]
    if not algos:
        print("[TSO plots] No algos found for OOD robustness figure.")
        return Path()

    x = np.arange(len(ood_splits))
    width = 0.8 / max(len(algos), 1)
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, algo in enumerate(algos):
        scores: list[float] = []
        lo_errs: list[float] = []
        hi_errs: list[float] = []
        for split in ood_splits:
            r = _get_row(rows, algo, split)
            ns = r.get("norm_score") if r else None
            ns_f = float(ns) if ns is not None else 0.0
            scores.append(ns_f)
            ci_lo = r.get("norm_score_ci_lo") if r else None
            ci_hi = r.get("norm_score_ci_hi") if r else None
            if ci_lo is not None and ci_hi is not None:
                lo_errs.append(max(0.0, ns_f - float(ci_lo)))
                hi_errs.append(max(0.0, float(ci_hi) - ns_f))
            else:
                lo_errs.append(0.0)
                hi_errs.append(0.0)

        offset = (i - len(algos) / 2 + 0.5) * width
        ax.bar(
            x + offset, scores, width=width * 0.9,
            label=ALGO_LABELS.get(algo, algo),
            color=ALGO_COLORS.get(algo, "#888888"), alpha=0.85,
            yerr=[lo_errs, hi_errs], capsize=3,
            error_kw={"elinewidth": 1.2, "alpha": 0.7},
        )

    ax.axhline(0, color="#9e9e9e", linewidth=0.8, linestyle="--")
    ax.axhline(1, color="#4caf50", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(["IID", "Load Stress\n(OOD)", "Line Tightening\n(OOD)"])
    ax.set_ylabel("NormScore (IQM ± 95% CI)")
    ax.set_title("TSO — OOD Robustness\n(0 = All-On, 1 = Merit-Order)")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_ylim(-0.3, 1.6)
    fig.tight_layout()

    out = figures_dir / "ood_robustness.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[TSO plots] saved {out}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Fig: Cross-backend / multi-device eval-cost vs wall time (train records)
# ──────────────────────────────────────────────────────────────────────────────

def _tso_cross_backend_curve_key(record: dict) -> tuple[str, str, str] | None:
    """Return (backend, device, series_id) for dedup, or None to skip.

    ``series_id`` distinguishes JAX PPO vs PPO-Lag vs Python PPO (SB3/SBX).
    """
    backend = (record.get("backend") or "jax_rejax").lower()
    dev_raw = (record.get("device") or "gpu").lower()
    device = "cuda" if dev_raw in ("cuda", "gpu") else dev_raw
    algo = (record.get("algo") or "").lower()

    if backend == "jax_rejax":
        if algo == "ppo":
            return (backend, device, "jax_ppo")
        if algo == "sac":
            return (backend, device, "jax_sac")
        if algo == "ppo_lagrangian":
            return (backend, device, "jax_ppo_lagrangian")
        return None
    if backend == "sb3":
        if algo == "ppo":
            return (backend, device, "sb3_ppo")
        return None
    if backend == "sbx":
        if algo in ("sbx_ppo", "ppo"):
            return (backend, device, "sbx_ppo")
        return None
    return None


def _tso_cross_backend_series_color(series_id: str) -> str:
    if series_id == "jax_ppo":
        return _CROSS_BACKEND_PPO_PYTHON
    if series_id == "jax_ppo_lagrangian":
        return _CROSS_BACKEND_LAG_PYTHON
    if series_id == "jax_sac":
        return _CROSS_BACKEND_SAC_JAX
    if series_id == "sb3_ppo":
        return _CROSS_BACKEND_PPO_SB3
    if series_id == "sbx_ppo":
        return _CROSS_BACKEND_PPO_SBX
    return "#455a64"


def _tso_device_linestyle(device: str) -> str:
    """Solid for GPU/CUDA; dashed for CPU (and anything else)."""
    d = (device or "").lower()
    return "-" if d in ("gpu", "cuda") else "--"


def plot_cross_backend_learning_walltime(
    task_dir: Path = TASK_DIR,
    *,
    seed: int = 0,
    after: str | None = None,
) -> Path:
    """Overlay **best eval operating cost so far** (cumulative min over checkpoints) vs wall time.

    For **JAX on GPU**, the PPO and PPO-Lagrangian lines use the same manifest
    rows as :func:`_canonical_train_runs` / ``learning_curves_walltime`` (GPU
    preferred, then newest timestamp), so this figure cannot drift from the main
    dashboard when only one of the two plots is regenerated.

    Other keys (JAX CPU, SB3, SBX) still pick the newest completed train row per
    ``(backend, device, series_id)`` at the given ``seed``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest_path = task_dir / "results" / "manifest.json"
    if not manifest_path.exists():
        print("[TSO plots] No manifest; skip cross_backend_learning_walltime.")
        return Path()

    try:
        records = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        print("[TSO plots] manifest unreadable; skip cross_backend_learning_walltime.")
        return Path()

    task_config = load_task_config(task_dir)
    if after is None:
        after = _default_campaign_after(task_dir)
    reward_scale = float(task_config.get("reward_scale", 1e-4))
    artifacts_dir = task_dir / "results" / "artifacts"

    best: dict[tuple[str, str, str], dict] = {}
    for record in records:
        if not _record_after_campaign(record, after):
            continue
        if record.get("task") != "tso":
            continue
        if record.get("split") != "train":
            continue
        if record.get("status") != "completed":
            continue
        if int(record.get("seed", -1)) != int(seed):
            continue
        arts = record.get("artifacts") or {}
        if "params" not in arts and "params_orbax" not in arts:
            continue
        key = _tso_cross_backend_curve_key(record)
        if key is None:
            continue
        ts = str(record.get("timestamp") or "")
        prev = best.get(key)
        if prev is None or ts > str(prev.get("timestamp") or ""):
            best[key] = record

    # Keep GPU JAX curves identical to `plot_learning_curves` canonical selection.
    canonical_gpu = _canonical_train_runs(task_dir, after=after)
    for algo_name in ("ppo", "sac", "ppo_lagrangian"):
        chosen = canonical_gpu.get((algo_name, int(seed)))
        if chosen is None:
            continue
        ck = _tso_cross_backend_curve_key(chosen)
        if ck is None:
            continue
        _be, dev, _sid = ck
        if dev != "cuda":
            continue
        best[ck] = chosen

    if not best:
        print(
            "[TSO plots] No train rows for cross_backend_learning_walltime "
            f"(seed={seed})."
        )
        return Path()

    rows_plot: list[tuple[tuple[str, str, str], dict]] = sorted(
        best.items(),
        key=lambda kv: float((kv[1].get("walltime_s") or 0.0)),
        reverse=True,
    )

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    _style_axis(ax)
    refs_million_gbp: dict[str, float] = {}
    summary = _load_summary(task_dir) or {}
    for algo in ("all_on", "merit_order"):
        row = _get_row(summary.get("rows", []), algo, "train")
        if row and row.get("total_operating_cost_mean") is not None:
            refs_million_gbp[algo] = float(row["total_operating_cost_mean"]) / 1e6
    _annotate_reference_lines(ax, refs_million_gbp)

    series_labels = {
        "jax_ppo": "PPO",
        "jax_sac": "SAC",
        "jax_ppo_lagrangian": "PPO-Lagrangian",
        "sb3_ppo": "PPO (SB3)",
        "sbx_ppo": "PPO (SBX)",
    }
    backend_labels = {"jax_rejax": "JAX", "sb3": "SB3", "sbx": "SBX"}

    for (_be, dev, series_id), run in rows_plot:
        run_id = str(run.get("run_id", ""))
        eval_cost = _eval_cost_curve(artifacts_dir, run_id, reward_scale)
        if eval_cost is None or eval_cost.size == 0:
            print(f"[TSO plots] cross_backend: skip {run_id} (no eval cost curve)")
            continue
        n_pts = len(eval_cost)
        x_w = _curve_x(run, artifacts_dir, n_pts, axis_kind="walltime")
        if x_w is None:
            continue
        color = _tso_cross_backend_series_color(series_id)
        ls = _tso_device_linestyle(dev)
        n_envs = read_parallel_n_envs_from_run_config(artifacts_dir, run_id)
        env_part = f", n_envs={n_envs}" if n_envs is not None else ""
        be_disp = backend_labels.get(_be, _be)
        dev_disp = "GPU" if dev in ("gpu", "cuda") else dev.upper()
        lab = (
            f"{series_labels.get(series_id, series_id)} · {be_disp} · {dev_disp}"
            f"{env_part}"
        )
        y_m = np.asarray(eval_cost, dtype=float) / 1e6
        y_best = np.minimum.accumulate(y_m)
        ax.plot(
            np.asarray(x_w, dtype=float) / 60.0,
            y_best,
            color=color,
            linestyle=ls,
            linewidth=2.0,
            label=lab,
            alpha=0.92,
        )

    ax.set_xlabel("Walltime (minutes)")
    ax.set_ylabel("Episode operating cost (million GBP, lower is better)")
    ax.set_title(
        f"TSO — Best eval cost so far vs walltime (cross-device / cross-backend, seed={seed})\n"
        "Solid = GPU/CUDA, dashed = CPU · Y = cumulative min per run · add SB3/SBX rows to extend"
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=7.5, loc="best", framealpha=0.92)
    fig.tight_layout()

    figures_dir = _figures_dir(task_dir)
    out = figures_dir / "cross_backend_learning_walltime.pdf"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[TSO plots] saved {out}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Generate all plots
# ──────────────────────────────────────────────────────────────────────────────

def generate_all_plots(
    task_dir: Path = TASK_DIR,
    *,
    after: str | None = None,
) -> None:
    """Generate all TSO benchmark figures."""
    if after is None:
        after = _default_campaign_after(task_dir)
    plot_normscore_bars(task_dir)
    plot_gantt_commitment(task_dir)
    plot_cost_decomposition(task_dir)
    plot_learning_curves(task_dir, after=after)
    plot_cross_backend_learning_walltime(task_dir, after=after)
    plot_ood_robustness(task_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", default=str(TASK_DIR))
    parser.add_argument("--after", default=None, help="ISO campaign-start timestamp filter")
    args = parser.parse_args()
    generate_all_plots(Path(args.task_dir), after=args.after)
