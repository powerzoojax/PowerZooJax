"""DC Microgrid extended figure generation (diagnostic + per-step plots).

Reads:
  - results/trajectories/*.npz   (per-step rollouts dumped by dump_all.py)
  - results/summary/latest.json  (aggregated metrics)
  - results/manifest.json        (raw records)

Produces ~16 figures saved under results/figures/extra/:

Time-series panels (24h day, episode 0):
  1.  power_dispatch_iid.{pdf,png}     -- 4-algo stacked PV+DG+battery vs DC load
  2.  soc_trajectory_iid.{pdf,png}     -- SOC over 24h, all algos overlaid
  3.  action_heatmap_<algo>_iid.{pdf,png}  -- 5 channels x 288 steps heatmap (per algo)
  4.  reward_breakdown_iid.{pdf,png}   -- shaped vs raw reward + penalty over time
  5.  cumulative_costs_iid.{pdf,png}   -- cumulative fuel + carbon + deficit penalty
  6.  power_balance_iid.{pdf,png}      -- residual = pv+dg+batt-load over 24h

Diurnal aggregates (mean across episodes):
  7.  diurnal_deficit_iid.{pdf,png}    -- per-hour deficit rate, all algos
  8.  diurnal_pv_dg_load_iid.{pdf,png} -- PV / DG / load profiles (rule_based)
  9.  hourly_action_means_iid.{pdf,png}-- per-hour mean action by algo

OOD / cross-split summaries:
  10. ood_cost_bars.{pdf,png}          -- energy/fuel/carbon stacked, 4 algos x 8 splits
  11. ood_feasibility.{pdf,png}        -- feasibility & deficit rate per (algo, split)
  12. ood_battery_cycles.{pdf,png}     -- battery throughput per algo per split
  13. ppo_vs_baselines_violin.{pdf,png}-- per-episode metric distributions

Training-side:
  14. learning_curves_zoom_early.{pdf,png}  -- first 10% of training, 3 seeds + mean
  15. learning_curves_log_y.{pdf,png}       -- full curve, log y-axis to see late-stage drift
  16. seed_dispersion.{pdf,png}             -- final eval reward per seed

Usage:
    python benchmarks/dc_microgrid/plots_extra.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

TASK_DIR = Path(__file__).resolve().parent
TRAJ_DIR = TASK_DIR / "results" / "trajectories"
FIG_DIR = TASK_DIR / "results" / "figures" / "extra"
ARTIFACTS_DIR = TASK_DIR / "results" / "artifacts"

ALGOS = ["no_control", "max_renewable", "rule_based", "ppo"]
ALGO_LABELS = {
    "no_control": "No Control",
    "max_renewable": "Max Renewable",
    "rule_based": "Rule-Based",
    "ppo": "PPO (shaped)",
}
ALGO_COLORS = {
    "no_control": "#9e9e9e",
    "max_renewable": "#8bc34a",
    "rule_based": "#4caf50",
    "ppo": "#2196f3",
}
ACTION_NAMES = ["train_sched", "ft_sched", "cooling_norm", "batt_norm", "dg_norm"]
SPLITS = [
    "train", "iid", "cooling_stress", "renewable_drought",
    "workload_swap", "workload_shock", "dg_derating", "sla_tighten",
]
SPLIT_LABELS = {
    "train": "Train", "iid": "IID",
    "cooling_stress": "Cooling Stress", "renewable_drought": "Renew. Drought",
    "workload_swap": "WL Swap", "workload_shock": "WL Shock",
    "dg_derating": "DG Derate", "sla_tighten": "SLA Tight",
}


def _load_traj(algo: str, split: str, seed: int = 0) -> Optional[dict]:
    pattern = f"{algo}_{split}_s{seed}*.npz"
    matches = sorted(TRAJ_DIR.glob(pattern))
    if not matches:
        return None
    return dict(np.load(matches[0]))


def _hour_axis(t_steps: int = 288) -> np.ndarray:
    return np.linspace(0, 24, t_steps, endpoint=False)


def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / f"{name}.pdf"
    png = FIG_DIR / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"[plots_extra] saved {name}.pdf + .png")


# ─────────────────────────────────────────────────────────────────────────────
# 1. 24h power dispatch — 4-panel stacked area
# ─────────────────────────────────────────────────────────────────────────────

def plot_power_dispatch(split: str = "iid") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    h = _hour_axis()
    for ax, algo in zip(axes, ALGOS):
        traj = _load_traj(algo, split)
        if traj is None:
            ax.set_title(f"{ALGO_LABELS[algo]} — no trajectory")
            continue
        # Episode 0
        p_pv = traj["p_pv_mw"][0]
        p_dg = traj["p_dg_mw"][0]
        p_batt = traj["p_batt_mw"][0]   # +discharge / -charge
        p_dc = traj["p_dc_mw"][0]
        batt_dis = np.where(p_batt > 0, p_batt, 0)
        batt_chg = np.where(p_batt < 0, -p_batt, 0)
        ax.fill_between(h, 0, p_pv, label="PV", color="#ffd54f", alpha=0.85)
        ax.fill_between(h, p_pv, p_pv + p_dg, label="DG", color="#a1887f", alpha=0.85)
        ax.fill_between(h, p_pv + p_dg, p_pv + p_dg + batt_dis, label="Battery dis", color="#8bc34a", alpha=0.7)
        ax.plot(h, p_dc, color="#d32f2f", lw=1.6, label="DC load")
        ax.plot(h, -batt_chg, color="#1976d2", lw=1.0, alpha=0.7, label="Battery chg (neg)")
        ax.set_ylabel("Power [MW]")
        ax.set_title(f"{ALGO_LABELS[algo]} (split={split}, ep=0)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, ncol=2)
    axes[-1].set_xlabel("Hour of day")
    fig.suptitle(f"DC Microgrid — 24h Power Dispatch ({SPLIT_LABELS[split]} split)")
    fig.tight_layout()
    _save(fig, f"power_dispatch_{split}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 2. SOC trajectory overlay
# ─────────────────────────────────────────────────────────────────────────────

def plot_soc_trajectory(split: str = "iid") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    h = _hour_axis()
    for algo in ALGOS:
        traj = _load_traj(algo, split)
        if traj is None:
            continue
        soc_mean = traj["soc"].mean(axis=0)
        soc_lo = traj["soc"].min(axis=0)
        soc_hi = traj["soc"].max(axis=0)
        ax.plot(h, soc_mean, label=ALGO_LABELS[algo], color=ALGO_COLORS[algo], lw=1.8)
        ax.fill_between(h, soc_lo, soc_hi, color=ALGO_COLORS[algo], alpha=0.15)
    ax.axhline(0.1, color="red", ls="--", lw=0.7, alpha=0.6, label="SOC min")
    ax.axhline(0.9, color="green", ls="--", lw=0.7, alpha=0.6, label="SOC max")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Battery SOC")
    ax.set_title(f"DC Microgrid — Battery SOC over 24h ({SPLIT_LABELS[split]}, mean ± episode range)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, f"soc_trajectory_{split}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Action heatmap per algo
# ─────────────────────────────────────────────────────────────────────────────

def plot_action_heatmaps(split: str = "iid") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(ALGOS), 1, figsize=(11, 9), sharex=True)
    for ax, algo in zip(axes, ALGOS):
        traj = _load_traj(algo, split)
        if traj is None:
            ax.set_title(f"{ALGO_LABELS[algo]} — no traj")
            continue
        action = traj["action"][0].T  # shape (5, 288)
        im = ax.imshow(action, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
                       extent=[0, 24, len(ACTION_NAMES) - 0.5, -0.5])
        ax.set_yticks(range(len(ACTION_NAMES)))
        ax.set_yticklabels(ACTION_NAMES)
        ax.set_title(f"{ALGO_LABELS[algo]} ({SPLIT_LABELS[split]}, ep=0)")
        plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    axes[-1].set_xlabel("Hour of day")
    fig.suptitle("DC Microgrid — Action Heatmaps")
    fig.tight_layout()
    _save(fig, f"action_heatmaps_{split}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Reward breakdown over time
# ─────────────────────────────────────────────────────────────────────────────

def plot_reward_breakdown(split: str = "iid") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(ALGOS), 1, figsize=(11, 10), sharex=True)
    h = _hour_axis()
    for ax, algo in zip(axes, ALGOS):
        traj = _load_traj(algo, split)
        if traj is None:
            ax.set_title(f"{ALGO_LABELS[algo]} — no traj")
            continue
        raw = traj["raw_reward"][0]
        pen = traj["shaping_penalty"][0]
        shaped = traj["reward"][0]
        ax.plot(h, raw, label="raw_reward", color="#1976d2", lw=1.2)
        ax.plot(h, -pen, label="-shaping_penalty", color="#e53935", lw=1.2)
        ax.plot(h, shaped, label="shaped_reward (raw - penalty)", color="black", lw=1.5)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_ylabel("Reward")
        ax.set_title(ALGO_LABELS[algo])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=8)
    axes[-1].set_xlabel("Hour of day")
    fig.suptitle(f"Reward Breakdown ({SPLIT_LABELS[split]}, ep=0)")
    fig.tight_layout()
    _save(fig, f"reward_breakdown_{split}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Cumulative costs over time
# ─────────────────────────────────────────────────────────────────────────────

def plot_cumulative_costs(split: str = "iid") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    h = _hour_axis()
    for algo in ALGOS:
        traj = _load_traj(algo, split)
        if traj is None:
            continue
        c = ALGO_COLORS[algo]
        axes[0].plot(h, traj["fuel_cost"][0].cumsum(), color=c, label=ALGO_LABELS[algo], lw=1.5)
        axes[1].plot(h, traj["carbon_kg"][0].cumsum(), color=c, label=ALGO_LABELS[algo], lw=1.5)
        axes[2].plot(h, traj["cost_power_deficit"][0].cumsum(), color=c, label=ALGO_LABELS[algo], lw=1.5)
    axes[0].set_title("Cumulative Fuel Cost [$]")
    axes[1].set_title("Cumulative Carbon [kgCO2]")
    axes[2].set_title("Cumulative deficit (units of load)")
    for ax in axes:
        ax.set_xlabel("Hour of day")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Cumulative Operating Costs ({SPLIT_LABELS[split]}, ep=0)")
    fig.tight_layout()
    _save(fig, f"cumulative_costs_{split}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Power balance residual
# ─────────────────────────────────────────────────────────────────────────────

def plot_power_balance(split: str = "iid") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    h = _hour_axis()
    for algo in ALGOS:
        traj = _load_traj(algo, split)
        if traj is None:
            continue
        residual = traj["p_pv_mw"][0] + traj["p_dg_mw"][0] + traj["p_batt_mw"][0] - traj["p_dc_mw"][0]
        ax.plot(h, residual, label=ALGO_LABELS[algo], color=ALGO_COLORS[algo], lw=1.4)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Residual = PV+DG+Batt - DC load [MW]")
    ax.set_title(f"Power Balance Residual (negative = deficit, {SPLIT_LABELS[split]}, ep=0)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, f"power_balance_{split}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Diurnal deficit rate (per-hour mean across episodes)
# ─────────────────────────────────────────────────────────────────────────────

def plot_diurnal_deficit(split: str = "iid") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    h = _hour_axis()
    for algo in ALGOS:
        traj = _load_traj(algo, split)
        if traj is None:
            continue
        deficit_mean = traj["cost_power_deficit"].mean(axis=0)
        ax.plot(h, deficit_mean, label=ALGO_LABELS[algo], color=ALGO_COLORS[algo], lw=1.6)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean deficit / load fraction")
    ax.set_title(f"Diurnal Power Deficit ({SPLIT_LABELS[split]}, mean over {3} eps)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, f"diurnal_deficit_{split}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 8. PV / DG / Load profiles
# ─────────────────────────────────────────────────────────────────────────────

def plot_pv_dg_load(split: str = "iid") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    traj = _load_traj("rule_based", split)
    if traj is None:
        print(f"[plots_extra] no rule_based traj for split={split}, skipping pv_dg_load")
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    h = _hour_axis()
    pv = traj["p_pv_mw"].mean(axis=0)
    dg = traj["p_dg_mw"].mean(axis=0)
    load = traj["p_dc_mw"].mean(axis=0)
    ax.fill_between(h, 0, pv, alpha=0.7, color="#ffd54f", label="PV (MW)")
    ax.plot(h, dg, color="#a1887f", lw=1.5, label="DG output (MW)")
    ax.plot(h, load, color="#d32f2f", lw=2.0, label="DC load (MW)")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("MW")
    ax.set_title(f"PV / DG / Load profiles under Rule-Based ({SPLIT_LABELS[split]})")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, f"pv_dg_load_{split}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Hourly mean action by algo
# ─────────────────────────────────────────────────────────────────────────────

def plot_hourly_action_means(split: str = "iid") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(ACTION_NAMES), 1, figsize=(11, 10), sharex=True)
    h = _hour_axis()
    for ax, ai in zip(axes, range(len(ACTION_NAMES))):
        for algo in ALGOS:
            traj = _load_traj(algo, split)
            if traj is None:
                continue
            mean = traj["action"].mean(axis=0)[:, ai]
            ax.plot(h, mean, label=ALGO_LABELS[algo], color=ALGO_COLORS[algo], lw=1.4)
        ax.set_ylabel(ACTION_NAMES[ai])
        ax.grid(True, alpha=0.3)
        if ai == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=4)
    axes[-1].set_xlabel("Hour of day")
    fig.suptitle(f"Hourly Mean Actions by Algorithm ({SPLIT_LABELS[split]})")
    fig.tight_layout()
    _save(fig, f"hourly_action_means_{split}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 10. OOD cost stacked bars
# ─────────────────────────────────────────────────────────────────────────────

def _load_summary() -> Optional[dict]:
    p = TASK_DIR / "results" / "summary" / "latest.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _row(rows, algo, split):
    for r in rows:
        if r["algo"] == algo and r["split"] == split:
            return r
    return None


def plot_ood_cost_bars() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = _load_summary()
    if summary is None:
        return
    rows = summary["rows"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=False)
    metric_keys = ["total_energy_cost_mean", "total_fuel_cost_mean", "total_carbon_kg_mean"]
    titles = ["Energy [MWh]", "Fuel cost [$]", "Carbon [kgCO2]"]
    n_algos = len(ALGOS)
    n_splits = len(SPLITS)
    width = 0.8 / n_algos
    x = np.arange(n_splits)
    for ax, key, title in zip(axes, metric_keys, titles):
        for i, algo in enumerate(ALGOS):
            vals = []
            for s in SPLITS:
                r = _row(rows, algo, s)
                vals.append(float(r.get(key) or 0.0) if r else 0.0)
            ax.bar(x + (i - n_algos / 2 + 0.5) * width, vals, width=width * 0.9,
                   label=ALGO_LABELS[algo], color=ALGO_COLORS[algo], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([SPLIT_LABELS[s] for s in SPLITS], rotation=25, ha="right")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Operating Cost Breakdown by (Algorithm, Split)")
    fig.tight_layout()
    _save(fig, "ood_cost_bars")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 11. OOD feasibility & deficit
# ─────────────────────────────────────────────────────────────────────────────

def plot_ood_feasibility() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = _load_summary()
    if summary is None:
        return
    rows = summary["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    n_algos = len(ALGOS)
    n_splits = len(SPLITS)
    width = 0.8 / n_algos
    x = np.arange(n_splits)
    for ax, key, title in zip(
        axes,
        ["feasibility_rate_mean", "power_deficit_rate_mean"],
        ["Feasibility rate (higher better)", "Power deficit rate (lower better)"],
    ):
        for i, algo in enumerate(ALGOS):
            vals = []
            for s in SPLITS:
                r = _row(rows, algo, s)
                vals.append(float(r.get(key) or 0.0) if r else 0.0)
            ax.bar(x + (i - n_algos / 2 + 0.5) * width, vals, width=width * 0.9,
                   label=ALGO_LABELS[algo], color=ALGO_COLORS[algo], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([SPLIT_LABELS[s] for s in SPLITS], rotation=25, ha="right")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Constraint Satisfaction across Splits")
    fig.tight_layout()
    _save(fig, "ood_feasibility")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 12. Battery cycles
# ─────────────────────────────────────────────────────────────────────────────

def plot_battery_cycles() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = _load_summary()
    if summary is None:
        return
    rows = summary["rows"]
    fig, ax = plt.subplots(figsize=(11, 5))
    n_algos = len(ALGOS)
    width = 0.8 / n_algos
    x = np.arange(len(SPLITS))
    for i, algo in enumerate(ALGOS):
        vals = []
        for s in SPLITS:
            r = _row(rows, algo, s)
            vals.append(float(r.get("battery_cycles_mean") or 0.0) if r else 0.0)
        ax.bar(x + (i - n_algos / 2 + 0.5) * width, vals, width=width * 0.9,
               label=ALGO_LABELS[algo], color=ALGO_COLORS[algo], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([SPLIT_LABELS[s] for s in SPLITS], rotation=25, ha="right")
    ax.set_ylabel("Battery cycles")
    ax.set_title("Battery throughput per (Algorithm, Split)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "battery_cycles")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 13. Per-episode metric distributions (violin)
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_episode_violin(split: str = "iid") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metric_traj_keys = ["reward", "cost_power_deficit", "fuel_cost"]
    titles = ["Per-step shaped reward", "Per-step deficit", "Per-step fuel cost ($)"]
    for ax, key, title in zip(axes, metric_traj_keys, titles):
        data = []
        labels = []
        colors = []
        for algo in ALGOS:
            traj = _load_traj(algo, split)
            if traj is None:
                continue
            arr = traj[key].flatten()  # all steps from all episodes
            data.append(arr)
            labels.append(ALGO_LABELS[algo])
            colors.append(ALGO_COLORS[algo])
        if not data:
            continue
        parts = ax.violinplot(data, showmeans=True, showmedians=False, widths=0.8)
        for body, c in zip(parts["bodies"], colors):
            body.set_facecolor(c)
            body.set_alpha(0.65)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"Per-step Metric Distributions ({SPLIT_LABELS[split]})")
    fig.tight_layout()
    _save(fig, f"per_episode_violin_{split}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 14. Learning curve — early-zoom (first 10%)
# ─────────────────────────────────────────────────────────────────────────────

def _load_eval_returns_per_seed() -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for f in sorted(ARTIFACTS_DIR.glob("dc_microgrid_ppo_train_s*_eval_returns.npy")):
        name = f.stem
        seed = int(name.split("_s")[1].split("_")[0])
        try:
            out[seed] = np.load(f).flatten()
        except Exception:
            pass
    return out


def plot_learning_curve_early() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = _load_eval_returns_per_seed()
    if not curves:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for seed, arr in sorted(curves.items()):
        head = arr[: max(1, len(arr) // 10)]
        ax.plot(np.linspace(0, 0.1, len(head)), head, label=f"seed {seed}", lw=1.2, alpha=0.9)
    if len(curves) >= 2:
        head_lens = [len(arr) // 10 for arr in curves.values()]
        L = min(head_lens)
        stacked = np.stack([arr[:L] for arr in curves.values()], axis=0)
        ax.plot(np.linspace(0, 0.1, L), stacked.mean(0), color="black", lw=2.0, label="mean")
    ax.set_xlabel("Training Progress (fraction)")
    ax.set_ylabel("Episode Reward (shaped, eval)")
    ax.set_title("PPO Learning Curve — Zoomed to first 10% of training")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "learning_curve_zoom_early")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 15. Learning curve — log y-axis
# ─────────────────────────────────────────────────────────────────────────────

def plot_learning_curve_logy() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = _load_eval_returns_per_seed()
    if not curves:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for seed, arr in sorted(curves.items()):
        ax.plot(np.linspace(0, 1, len(arr)), -arr, label=f"seed {seed}", alpha=0.85, lw=1.0)
    if len(curves) >= 2:
        L = min(len(a) for a in curves.values())
        stacked = np.stack([a[:L] for a in curves.values()], axis=0)
        ax.plot(np.linspace(0, 1, L), -stacked.mean(0), color="black", lw=2.0, label="mean")
    ax.set_yscale("log")
    ax.set_xlabel("Training Progress (fraction)")
    ax.set_ylabel("Negative Episode Reward (log)")
    ax.set_title("PPO Learning Curve — Log scale on |reward|")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    _save(fig, "learning_curve_logy")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 16. Final eval reward dispersion across seeds
# ─────────────────────────────────────────────────────────────────────────────

def plot_seed_dispersion() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = _load_eval_returns_per_seed()
    if not curves:
        return
    seeds = sorted(curves.keys())
    last_quarter_means = [float(curves[s][-len(curves[s]) // 4 :].mean()) for s in seeds]
    final_evals = [float(curves[s][-1]) for s in seeds]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(seeds))
    w = 0.35
    ax.bar(x - w / 2, last_quarter_means, w, label="mean of last 25%", color="#1976d2", alpha=0.85)
    ax.bar(x + w / 2, final_evals, w, label="final eval", color="#43a047", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("Episode reward (shaped)")
    ax.set_title("PPO seed dispersion at end of training")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "seed_dispersion")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Generate everything
# ─────────────────────────────────────────────────────────────────────────────

def generate_all() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # Time-series & diurnal — across multiple representative splits
    for split in ("iid", "cooling_stress", "renewable_drought", "dg_derating"):
        plot_power_dispatch(split)
        plot_soc_trajectory(split)
        plot_action_heatmaps(split)
        plot_reward_breakdown(split)
        plot_cumulative_costs(split)
        plot_power_balance(split)
        plot_diurnal_deficit(split)
        plot_pv_dg_load(split)
        plot_hourly_action_means(split)
        plot_per_episode_violin(split)

    # Cross-split summaries
    plot_ood_cost_bars()
    plot_ood_feasibility()
    plot_battery_cycles()

    # Training-side
    plot_learning_curve_early()
    plot_learning_curve_logy()
    plot_seed_dispersion()


if __name__ == "__main__":
    generate_all()
    print(f"\n[plots_extra] All figures written under {FIG_DIR}")
