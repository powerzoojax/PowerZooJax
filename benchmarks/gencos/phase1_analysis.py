"""Phase-1 GenCos diagnostics for an isolated seed-0 campaign.

Generates a richer figure suite for the final seed-0 Phase-1 campaign:

* ``results/analysis_episode_summary.json``
* ``results/figures/phase1_episode_mechanism.{pdf,png}``
* ``results/figures/phase1_policy_compare.{pdf,png}``
* ``results/figures/phase1_training_diagnostics.{pdf,png}``
* ``results/figures/phase1_iid_metric_distributions.{pdf,png}``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.common.configs import load_config
from benchmarks.common.io import load_manifest, load_pickle, load_run
from powerzoojax.rl.ippo import SharedActorCritic
from powerzoojax.tasks.gencos import GencosTask

DEFAULT_TASK_DIR = Path(__file__).resolve().parent
POLICY_ORDER = ("truthful", "uniform_mid", "max_markup", "ippo")
POLICY_COLORS = {
    "truthful": "#3a7d44",
    "uniform_mid": "#d8891c",
    "max_markup": "#c44536",
    "ippo": "#1f6feb",
}
POLICY_LABELS = {
    "truthful": "Truthful",
    "uniform_mid": "Uniform Mid",
    "max_markup": "Max Markup",
    "ippo": "IPPO",
}
AGENT_COLORS = {
    "genco_0": "#1f77b4",
    "genco_1": "#ff7f0e",
    "genco_2": "#2ca02c",
    "genco_3": "#d62728",
    "genco_4": "#9467bd",
}
PLOT_FACE = "white"
AX_FACE = "white"
GRID_COLOR = "#d5dbe3"
DAY_BAND = "#eceff5"


def _figures_dir(task_dir: Path) -> Path:
    return task_dir / "results" / "figures"


def _save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(
        path.with_suffix(".png"),
        dpi=150,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )


def _style_axes(ax) -> None:
    ax.set_facecolor(AX_FACE)
    ax.grid(True, color=GRID_COLOR, alpha=0.25, linestyle="-", linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _decorate_time_axis(ax, n_steps: int) -> None:
    for start in range(0, n_steps, 12):
        if (start // 12) % 2 == 1:
            ax.axvspan(start - 0.5, min(start + 11.5, n_steps - 0.5), color=DAY_BAND, alpha=0.35)
    ax.set_xlim(0, n_steps - 1)


def _rolling_mean(arr: np.ndarray, window: int = 7) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size < 3:
        return arr
    window = min(window, arr.size if arr.size % 2 == 1 else arr.size - 1)
    if window < 3:
        return arr
    pad = window // 2
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.pad(arr, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _safe_interp(xs_src: np.ndarray, ys_src: np.ndarray, xs_dst: np.ndarray) -> np.ndarray:
    ys_src = np.asarray(ys_src, dtype=np.float64)
    finite_mask = np.isfinite(ys_src)
    if not np.any(finite_mask):
        return np.full_like(xs_dst, np.nan, dtype=np.float64)
    return np.interp(xs_dst, xs_src[finite_mask], ys_src[finite_mask])


def _load_latest_train_run_id(task_dir: Path, *, algo: str = "ippo", seed: int = 0) -> str:
    records = [
        r
        for r in load_manifest(task_dir)
        if r.algo == algo
        and r.seed == seed
        and r.split == "train"
        and (r.backend or "jax_rejax") == "jax_rejax"
        and (r.device or "gpu") == "gpu"
        and (r.artifacts or {}).get("params")
        and r.status == "completed"
    ]
    if not records:
        raise FileNotFoundError(
            f"No completed training run found in {task_dir}/results/manifest.json "
            f"for canonical jax_rejax+gpu algo={algo!r}, seed={seed}."
        )
    records.sort(key=lambda r: (r.timestamp, r.run_id))
    return records[-1].run_id


def _baseline_policy(agent_names: list[str], action_dim: int, action_value: float) -> Callable:
    action = jnp.full((action_dim,), action_value, dtype=jnp.float32)

    def _policy(_obs_dict):
        return {name: action for name in agent_names}

    return _policy


def _load_ippo_policy(task_dir: Path, run_id: str, env) -> Callable:
    train_cfg = load_config(task_dir / "configs" / "train_ippo.yaml")
    hidden_dims = tuple(train_cfg.get("hidden_dims", [128, 128]))
    record = load_run(run_id, task_dir)
    net_params = load_pickle(task_dir / "results" / record.artifacts["params"])
    network = SharedActorCritic(
        hidden_dims=hidden_dims,
        action_dim=env.action_space().shape[0],
    )
    agent_names = list(env.agent_names)

    def _policy(obs_dict):
        return {
            name: jnp.clip(network.apply(net_params, obs_dict[name])[0], -1.0, 1.0)
            for name in agent_names
        }

    return _policy


def _rollout_episode(env, key: jax.Array, policy_fn: Callable) -> dict:
    obs_dict, state = env.reset(key)
    params = env._params
    agent_names = list(env.agent_names)
    n_steps = int(params.max_steps)
    total_profile_rows = int(params.load_profiles.shape[0])

    mean_action_t = {name: [] for name in agent_names}
    profit = {name: [] for name in agent_names}
    load_mw = []
    mean_lmp = []
    lmp = []
    unit_power = []
    dispatch_share = []
    ramp_binding_rate = []
    n_violations = []
    is_safe = []
    sced_converged = []

    episode_start_idx = int(state.episode_start_idx)
    for _ in range(n_steps):
        profile_row = int((state.episode_start_idx + state.time_step) % total_profile_rows)
        nodal_load = np.asarray(
            params.nodes_loads_map @ params.load_profiles[profile_row],
            dtype=np.float32,
        )
        key, step_key = jax.random.split(key)
        actions = policy_fn(obs_dict)
        for name in agent_names:
            mean_action_t[name].append(float(jnp.mean(actions[name])))
        obs_dict, state, rewards, _dones, info = env.step(step_key, state, actions)
        power = np.asarray(info["unit_power"], dtype=np.float32)
        shares = power / (power.sum() + 1e-8)
        load_mw.append(nodal_load)
        lmp_values = np.asarray(info["lmp"], dtype=np.float32)
        mean_lmp.append(float(np.mean(lmp_values)))
        lmp.append(lmp_values)
        unit_power.append(power)
        dispatch_share.append(shares)
        ramp_binding_rate.append(float(info["ramp_binding_rate"]))
        n_violations.append(float(info["n_violations"]))
        is_safe.append(bool(info["is_safe"]))
        sced_converged.append(bool(info["sced_converged"]))
        for name in agent_names:
            profit[name].append(float(rewards[name]))

    unit_power_arr = np.asarray(unit_power, dtype=np.float32)
    dispatch_share_arr = np.asarray(dispatch_share, dtype=np.float32)
    hhi_t = np.sum(dispatch_share_arr ** 2, axis=1)
    profit_arr = {
        name: np.asarray(values, dtype=np.float32) for name, values in profit.items()
    }
    total_profit_t = np.sum(
        np.stack([profit_arr[name] for name in agent_names], axis=0),
        axis=0,
    )
    total_profit_per_agent = {
        name: float(np.sum(values)) for name, values in profit_arr.items()
    }

    return {
        "episode_start_idx": episode_start_idx,
        "agent_names": agent_names,
        "mean_action_t": {
            name: np.asarray(values, dtype=np.float32) for name, values in mean_action_t.items()
        },
        "mean_action_per_agent": {
            name: float(np.mean(values)) for name, values in mean_action_t.items()
        },
        "total_profit_per_agent": total_profit_per_agent,
        "avg_dispatch_share_per_agent": {
            name: float(np.mean(dispatch_share_arr[:, idx]))
            for idx, name in enumerate(agent_names)
        },
        "total_profit": float(sum(total_profit_per_agent.values())),
        "total_profit_t": np.asarray(total_profit_t, dtype=np.float32),
        "total_profit_cumulative": np.cumsum(total_profit_t, dtype=np.float64),
        "cum_profit": {
            name: np.cumsum(values, dtype=np.float64) for name, values in profit_arr.items()
        },
        "load_mw": np.asarray(load_mw, dtype=np.float32),
        "mean_lmp": np.asarray(mean_lmp, dtype=np.float32),
        "lmp": np.asarray(lmp, dtype=np.float32),
        "unit_power": unit_power_arr,
        "dispatch_share": dispatch_share_arr,
        "hhi_t": np.asarray(hhi_t, dtype=np.float32),
        "hhi_mean": float(np.mean(hhi_t)),
        "ramp_binding_rate": np.asarray(ramp_binding_rate, dtype=np.float32),
        "ramp_binding_rate_mean": float(np.mean(ramp_binding_rate)),
        "n_violations": np.asarray(n_violations, dtype=np.float32),
        "n_violations_sum": float(np.sum(n_violations)),
        "unsafe_steps": int(np.sum(~np.asarray(is_safe, dtype=bool))),
        "is_safe": np.asarray(is_safe, dtype=bool),
        "sced_converged": np.asarray(sced_converged, dtype=bool),
        "sced_convergence_rate": float(np.mean(sced_converged)),
        "profit": profit_arr,
    }


def _plot_episode_mechanism(task_dir: Path, episode: dict, *, split: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.arange(len(episode["mean_lmp"]))
    agent_names = episode["agent_names"]
    total_load = np.sum(episode["load_mw"], axis=1)

    fig, axes = plt.subplots(3, 2, figsize=(16, 11), sharex="col")
    fig.patch.set_facecolor(PLOT_FACE)
    for ax in axes.ravel():
        _style_axes(ax)
        _decorate_time_axis(ax, len(steps))

    ax = axes[0, 0]
    ax.plot(steps, total_load, color="#222222", lw=2.4, label="Total load")
    ax.set_ylabel("Load (MW)")
    ax_t = ax.twinx()
    ax_t.set_facecolor("none")
    ax_t.plot(steps, episode["mean_lmp"], color=POLICY_COLORS["ippo"], lw=2.4, label="Mean LMP")
    ax_t.set_ylabel("Mean LMP ($/MWh)")
    ax_t.grid(False)
    lines = ax.get_lines() + ax_t.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], loc="upper right", fontsize=8)
    ax.set_title(
        "IPPO showcase episode: system load and price response\n"
        f"split={split}, episode_start_idx={episode['episode_start_idx']}"
    )
    ax.text(
        0.015,
        0.08,
        (
            f"Episode profit = {episode['total_profit'] / 1e3:.1f}k$\n"
            f"Mean HHI = {episode['hhi_mean']:.3f}\n"
            f"Ramp binding = {episode['ramp_binding_rate_mean']:.3f}"
        ),
        transform=ax.transAxes,
        fontsize=8,
        bbox={"facecolor": "#f3f4f6", "edgecolor": "#c4c9d0", "boxstyle": "round,pad=0.35"},
    )

    ax = axes[0, 1]
    for idx, name in enumerate(agent_names):
        ax.plot(
            steps,
            episode["unit_power"][:, idx],
            color=AGENT_COLORS.get(name, f"C{idx}"),
            lw=2,
            label=name,
        )
    ax.set_title("Physical dispatch by agent")
    ax.set_ylabel("Dispatch (MW)")
    ax.legend(loc="upper right", ncol=3, fontsize=8)

    ax = axes[1, 0]
    for idx, name in enumerate(agent_names):
        ax.plot(
            steps,
            episode["cum_profit"][name],
            color=AGENT_COLORS.get(name, f"C{idx}"),
            lw=2,
            label=name,
        )
    ax.set_title("Cumulative profit accrual")
    ax.set_ylabel("Cumulative profit ($)")

    ax = axes[1, 1]
    action_values = np.concatenate(
        [np.asarray(episode["mean_action_t"][name], dtype=np.float64) for name in agent_names]
    )
    action_center = float(np.mean(action_values))
    action_span = max(0.03, float(np.max(action_values) - np.min(action_values)) * 3.0)
    for idx, name in enumerate(agent_names):
        ax.plot(
            steps,
            episode["mean_action_t"][name],
            color=AGENT_COLORS.get(name, f"C{idx}"),
            lw=2,
            label=name,
        )
    ax.axhline(0.0, color="#666666", ls=":", lw=0.8)
    ax.set_ylim(action_center - action_span, action_center + action_span)
    ax.set_title("Bid aggressiveness (mean action per agent)")
    ax.set_ylabel("Action in [-1, 1]")
    ax.text(
        0.02,
        0.90,
        "Reference: truthful = -1, max markup = +1",
        transform=ax.transAxes,
        fontsize=8,
        bbox={"facecolor": "#f3f4f6", "edgecolor": "#c4c9d0", "boxstyle": "round,pad=0.25"},
    )

    ax = axes[2, 0]
    for idx, name in enumerate(agent_names):
        ax.plot(
            steps,
            episode["dispatch_share"][:, idx],
            color=AGENT_COLORS.get(name, f"C{idx}"),
            lw=2,
            label=name,
        )
    ax.set_title("Competitive market share over the day")
    ax.set_ylabel("Dispatch share")
    ax.set_xlabel("30-min interval")
    ax.set_ylim(0.0, 0.85)

    ax = axes[2, 1]
    ax.plot(steps, episode["hhi_t"], color="#6d4c41", lw=2.2, label="HHI")
    ax.plot(
        steps,
        episode["ramp_binding_rate"],
        color="#6a1b9a",
        lw=2.2,
        label="Ramp binding",
    )
    ax.bar(
        steps,
        episode["n_violations"],
        color="#c44536",
        alpha=0.22,
        width=0.9,
        label="Constraint violations",
    )
    unsafe_mask = ~episode["is_safe"]
    if np.any(unsafe_mask):
        for idx in steps[unsafe_mask]:
            ax.axvspan(idx - 0.5, idx + 0.5, color="#c44536", alpha=0.12)
    else:
        ax.text(
            0.02,
            0.90,
            "No unsafe steps / no violations",
            transform=ax.transAxes,
            fontsize=8,
            bbox={"facecolor": "#eef2ee", "edgecolor": "#aab8ae", "boxstyle": "round,pad=0.25"},
        )
    ax.set_title("Concentration and physical coupling")
    ax.set_ylabel("HHI / ramp / violations")
    ax.set_xlabel("30-min interval")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    out = _figures_dir(task_dir) / "phase1_episode_mechanism.pdf"
    _save_figure(fig, out)
    plt.close(fig)
    return out


def _plot_policy_compare(task_dir: Path, episodes: dict[str, dict]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.25,
            "pdf.fonttype": 42,
        }
    )

    steps = np.arange(len(episodes["ippo"]["mean_lmp"]))
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.67))
    fig.patch.set_facecolor(PLOT_FACE)
    for ax in axes.ravel():
        _style_axes(ax)

    ax = axes[0, 0]
    for policy_name in POLICY_ORDER:
        episode = episodes[policy_name]
        label = POLICY_LABELS.get(policy_name, policy_name.replace("_", " ").title())
        total_load = np.sum(episode["load_mw"], axis=1)
        ax.plot(
            total_load,
            episode["mean_lmp"],
            color=POLICY_COLORS[policy_name],
            lw=2,
            label=label,
        )
        ax.scatter(
            total_load[::6],
            episode["mean_lmp"][::6],
            color=POLICY_COLORS[policy_name],
            s=28,
            alpha=0.9,
        )
    ax.set_title("Same physical load, different price response")
    ax.set_xlabel("Total load (MW)")
    ax.set_ylabel("Mean LMP ($/MWh)")
    policy_handles, policy_labels = ax.get_legend_handles_labels()

    ax = axes[0, 1]
    _decorate_time_axis(ax, len(steps))
    for policy_name in POLICY_ORDER:
        episode = episodes[policy_name]
        line = ax.plot(
            steps,
            episode["total_profit_cumulative"],
            color=POLICY_COLORS[policy_name],
            lw=2.3,
            label=POLICY_LABELS.get(policy_name, policy_name.replace("_", " ").title()),
        )[0]
        ax.scatter(
            [steps[-1]],
            [episode["total_profit_cumulative"][-1]],
            color=line.get_color(),
            s=40,
            zorder=3,
        )
    ax.set_title("Cumulative total profit on the same episode")
    ax.set_ylabel("Cumulative profit ($)")

    ax = axes[1, 0]
    _decorate_time_axis(ax, len(steps))
    for policy_name in POLICY_ORDER:
        episode = episodes[policy_name]
        ax.plot(
            steps,
            episode["hhi_t"],
            color=POLICY_COLORS[policy_name],
            lw=2.1,
            label=POLICY_LABELS.get(policy_name, policy_name.replace("_", " ").title()),
        )
    uniform_line = ax.axhline(1.0 / 5.0, color="#3a7d44", lw=0.9, ls="--", label="Uniform dispatch")
    monopoly_line = ax.axhline(1.0, color="#c44536", lw=0.9, ls=":", label="Monopoly")
    ax.set_title("Market concentration trajectory")
    ax.set_xlabel("30-min interval")
    ax.set_ylabel("HHI")
    ax.set_ylim(0.0, 1.0)
    ax.legend(handles=[uniform_line, monopoly_line], fontsize=10, loc="lower left", framealpha=0.88)

    ax = axes[1, 1]
    _decorate_time_axis(ax, len(steps))
    for policy_name in POLICY_ORDER:
        episode = episodes[policy_name]
        ax.plot(
            steps,
            episode["ramp_binding_rate"],
            color=POLICY_COLORS[policy_name],
            lw=2.1,
            label=POLICY_LABELS.get(policy_name, policy_name.replace("_", " ").title()),
        )
    ax.set_title("Physical coupling intensity")
    ax.set_xlabel("30-min interval")
    ax.set_ylabel("Ramp binding rate")
    ax.set_ylim(0.0, 1.0)

    fig.legend(
        policy_handles,
        policy_labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        handlelength=2.0,
        columnspacing=1.4,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = _figures_dir(task_dir) / "phase1_policy_compare.pdf"
    _save_figure(fig, out)
    plt.close(fig)
    return out


def _plot_training_diagnostics(task_dir: Path, run_id: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    artifacts_dir = task_dir / "results" / "artifacts"
    run_record = load_run(run_id, task_dir)
    train_cfg_path = task_dir / "results" / run_record.artifacts.get("config", "")
    total_timesteps = None
    if train_cfg_path.exists():
        config_blob = json.loads(train_cfg_path.read_text(encoding="utf-8"))
        total_timesteps = (
            config_blob.get("train_config", {}).get("total_timesteps")
            or config_blob.get("task_config", {}).get("total_timesteps")
        )

    def _load_series(key: str) -> np.ndarray:
        return np.asarray(np.load(artifacts_dir / f"{run_id}_{key}.npy"), dtype=np.float64)

    eval_return = _load_series("learning_curve_eval_return")
    market_hhi = _load_series("market_HHI")
    ramp_binding = _load_series("market_ramp_binding_rate")
    share_paths = sorted(artifacts_dir.glob(f"{run_id}_market_dispatch_share_*.npy"))
    share_series = {
        path.stem.split("market_dispatch_share_")[1]: np.asarray(np.load(path), dtype=np.float64)
        for path in share_paths
    }

    if total_timesteps is None:
        total_timesteps = 5_000_000.0
    total_millions = float(total_timesteps) / 1e6
    eval_x = np.linspace(0.0, total_millions, len(eval_return))
    dense_x = np.linspace(0.0, total_millions, len(market_hhi))
    plateau_eval_start = eval_x[max(0, len(eval_x) - 10)]
    plateau_dense_start = dense_x[max(0, len(dense_x) - max(10, len(dense_x) // 8))]

    hhi_on_eval = _safe_interp(dense_x, market_hhi, eval_x)
    ramp_on_eval = _safe_interp(dense_x, ramp_binding, eval_x)
    dominant_agent = max(share_series, key=lambda name: float(np.nanmean(share_series[name])))
    dominant_share_on_eval = _safe_interp(dense_x, share_series[dominant_agent], eval_x)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.patch.set_facecolor(PLOT_FACE)
    for ax in axes.ravel():
        _style_axes(ax)

    ax = axes[0, 0]
    ax.axvspan(plateau_eval_start, eval_x[-1], color="#eef2f8", alpha=0.42)
    ax.plot(eval_x, eval_return, color=POLICY_COLORS["ippo"], alpha=0.35, lw=1.6)
    ax.plot(eval_x, _rolling_mean(eval_return, window=7), color=POLICY_COLORS["ippo"], lw=2.8)
    ax.set_title("Eval return convergence")
    ax.set_ylabel("Eval return ($/episode)")
    ax.text(
        0.03,
        0.88,
        f"Final 10 mean = {np.mean(eval_return[-10:]):.0f}",
        transform=ax.transAxes,
        fontsize=8,
        bbox={"facecolor": "#f3f4f6", "edgecolor": "#c4c9d0", "boxstyle": "round,pad=0.25"},
    )

    ax = axes[0, 1]
    ax.axvspan(plateau_dense_start, dense_x[-1], color="#ebecef", alpha=0.35)
    ax.plot(dense_x, market_hhi, color="#6d4c41", alpha=0.28, lw=1.2)
    ax.plot(dense_x, _rolling_mean(market_hhi, window=21), color="#6d4c41", lw=2.6)
    ax.set_title("Concentration during training")
    ax.set_ylabel("HHI")
    hhi_pad = max(0.0015, float(np.max(market_hhi) - np.min(market_hhi)) * 0.25)
    ax.set_ylim(float(np.min(market_hhi) - hhi_pad), float(np.max(market_hhi) + hhi_pad))
    ax.text(
        0.03,
        0.88,
        "Reference: uniform dispatch = 0.20, monopoly = 1.00",
        transform=ax.transAxes,
        fontsize=8,
        bbox={"facecolor": "#f3f4f6", "edgecolor": "#c4c9d0", "boxstyle": "round,pad=0.25"},
    )

    ax = axes[0, 2]
    ax.axvspan(plateau_dense_start, dense_x[-1], color="#ebecef", alpha=0.35)
    ax.plot(dense_x, ramp_binding, color="#6a1b9a", alpha=0.28, lw=1.2)
    ax.plot(dense_x, _rolling_mean(ramp_binding, window=21), color="#6a1b9a", lw=2.6)
    ax.set_title("Physical coupling during training")
    ax.set_ylabel("Ramp binding rate")
    ramp_pad = max(0.002, float(np.max(ramp_binding) - np.min(ramp_binding)) * 0.25)
    ax.set_ylim(float(np.min(ramp_binding) - ramp_pad), float(np.max(ramp_binding) + ramp_pad))

    ax = axes[1, 0]
    for idx, name in enumerate(sorted(share_series)):
        ax.plot(
            dense_x,
            share_series[name],
            color=AGENT_COLORS.get(name, f"C{idx}"),
            lw=2,
            label=name,
        )
    ax.set_title("Agent market-share convergence")
    ax.set_xlabel("Training timesteps (M)")
    ax.set_ylabel("Dispatch share")
    ax.legend(fontsize=8, ncol=3)

    ax = axes[1, 1]
    sc = ax.scatter(
        hhi_on_eval,
        eval_return,
        c=eval_x,
        cmap="copper",
        s=38,
        edgecolors="none",
    )
    ax.plot(hhi_on_eval, eval_return, color="#8a6d3b", alpha=0.5, lw=1.4)
    ax.scatter([hhi_on_eval[0]], [eval_return[0]], color="#2f2f2f", s=45, marker="s", label="start")
    ax.scatter([hhi_on_eval[-1]], [eval_return[-1]], color="#111111", s=50, marker="*", label="end")
    ax.set_title("Competition phase portrait")
    ax.set_xlabel("HHI")
    ax.set_ylabel("Eval return ($/episode)")
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    ax.scatter(
        dominant_share_on_eval,
        eval_return,
        c=eval_x,
        cmap="copper",
        s=38,
        edgecolors="none",
    )
    ax.plot(dominant_share_on_eval, eval_return, color=AGENT_COLORS.get(dominant_agent, "#444444"), alpha=0.55, lw=1.4)
    ax.scatter([dominant_share_on_eval[0]], [eval_return[0]], color="#2f2f2f", s=45, marker="s")
    ax.scatter([dominant_share_on_eval[-1]], [eval_return[-1]], color="#111111", s=50, marker="*")
    ax.set_title(f"Return vs dominant-agent share ({dominant_agent})")
    ax.set_xlabel("Dominant agent dispatch share")
    ax.set_ylabel("Eval return ($/episode)")

    cbar = fig.colorbar(sc, ax=axes[1, 1:], shrink=0.9, pad=0.02)
    cbar.set_label("Training timesteps (M)")

    for ax in axes[1, :]:
        ax.set_facecolor(AX_FACE)
    axes[1, 0].set_xlim(0.0, total_millions)
    axes[0, 0].set_xlim(0.0, total_millions)
    axes[0, 1].set_xlim(0.0, total_millions)
    axes[0, 2].set_xlim(0.0, total_millions)
    for ax in axes[0, :]:
        ax.set_xlabel("Training timesteps (M)")

    fig.subplots_adjust(left=0.055, right=0.96, top=0.94, bottom=0.08, wspace=0.26, hspace=0.25)
    out = _figures_dir(task_dir) / "phase1_training_diagnostics.pdf"
    _save_figure(fig, out)
    plt.close(fig)
    return out


def _load_per_episode_records(task_dir: Path, *, algo: str, split: str, seed: int = 0) -> list[dict]:
    records = [
        r
        for r in load_manifest(task_dir)
        if r.algo == algo
        and r.split == split
        and r.seed == seed
        and r.status == "completed"
        and (r.artifacts or {}).get("per_episode")
    ]
    if not records:
        raise FileNotFoundError(
            f"No per-episode record found for algo={algo!r}, split={split!r}, seed={seed}."
        )
    records.sort(key=lambda r: (r.timestamp, r.run_id))
    path = task_dir / "results" / records[-1].artifacts["per_episode"]
    return json.loads(path.read_text(encoding="utf-8"))


def _plot_iid_metric_distributions(task_dir: Path, *, seed: int = 0) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        ("total_profit", "Episode total profit ($)"),
        ("mean_lmp", "Mean LMP ($/MWh)"),
        ("hhi", "HHI"),
        ("ramp_binding_rate", "Ramp binding rate"),
    ]
    per_policy = {
        policy: _load_per_episode_records(task_dir, algo=policy, split="iid", seed=seed)
        for policy in POLICY_ORDER
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor(PLOT_FACE)
    for ax in axes.ravel():
        _style_axes(ax)

    for ax, (metric, ylabel) in zip(axes.ravel(), metrics):
        data = [
            np.asarray([row[metric] for row in per_policy[policy]], dtype=np.float64)
            for policy in POLICY_ORDER
        ]
        positions = np.arange(1, len(POLICY_ORDER) + 1)
        box = ax.boxplot(
            data,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
        )
        for patch, policy in zip(box["boxes"], POLICY_ORDER):
            patch.set_facecolor(POLICY_COLORS[policy])
            patch.set_alpha(0.28)
            patch.set_edgecolor(POLICY_COLORS[policy])
            patch.set_linewidth(1.4)
        for median in box["medians"]:
            median.set_color("#202020")
            median.set_linewidth(1.8)
        for pos, series, policy in zip(positions, data, POLICY_ORDER):
            jitter = np.linspace(-0.12, 0.12, len(series))
            ax.scatter(
                np.full_like(series, pos, dtype=np.float64) + jitter,
                series,
                s=26,
                color=POLICY_COLORS[policy],
                alpha=0.65,
                edgecolors="white",
                linewidths=0.35,
            )
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel(ylabel)
        ax.set_xticks(positions)
        ax.set_xticklabels([policy.replace("_", " ").title() for policy in POLICY_ORDER], rotation=15)

    fig.suptitle("IID outcome distributions across 30 evaluation episodes", y=0.995, fontsize=14)
    fig.tight_layout()
    out = _figures_dir(task_dir) / "phase1_iid_metric_distributions.pdf"
    _save_figure(fig, out)
    plt.close(fig)
    return out


def analyze_phase1_episode(
    task_dir: Path,
    *,
    run_id: str | None = None,
    split: str = "iid",
    seed: int = 0,
) -> dict:
    task_dir = Path(task_dir)
    if run_id is None:
        run_id = _load_latest_train_run_id(task_dir, seed=seed)

    task = GencosTask(
        n_segments=3,
        max_markup=2.0,
        max_steps=int(load_config(task_dir / "configs" / "task.yaml").get("max_steps", 48)),
    )
    env = task.make_env(split)
    agent_names = list(env.agent_names)
    action_dim = env.action_space().shape[0]

    policy_fns = {
        "truthful": _baseline_policy(agent_names, action_dim, -1.0),
        "uniform_mid": _baseline_policy(agent_names, action_dim, 0.0),
        "max_markup": _baseline_policy(agent_names, action_dim, 1.0),
        "ippo": _load_ippo_policy(task_dir, run_id, env),
    }

    episodes = {
        name: _rollout_episode(env, jax.random.PRNGKey(seed), policy_fn)
        for name, policy_fn in policy_fns.items()
    }
    showcase_idx = episodes["ippo"]["episode_start_idx"]
    if any(episode["episode_start_idx"] != showcase_idx for episode in episodes.values()):
        raise RuntimeError("Expected all policies to analyze the same episode_start_idx.")

    profit_order = {
        name: episodes[name]["total_profit"]
        for name in ("truthful", "ippo", "max_markup")
    }
    acceptance = {
        "sced_convergence_rate_is_1": episodes["ippo"]["sced_convergence_rate"] == 1.0,
        "unsafe_steps_zero": episodes["ippo"]["unsafe_steps"] == 0,
        "n_violations_zero": episodes["ippo"]["n_violations_sum"] == 0.0,
        "ramp_binding_rate_gt_0_05": episodes["ippo"]["ramp_binding_rate_mean"] > 0.05,
        "profit_order_truthful_lt_ippo_lt_max": (
            profit_order["truthful"] < profit_order["ippo"] < profit_order["max_markup"]
        ),
        "market_hhi_lt_1": episodes["ippo"]["hhi_mean"] < 1.0,
    }

    artifacts_dir = task_dir / "results" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    npz_payload = {}
    for name in POLICY_ORDER:
        npz_payload[f"{name}_total_load_mw"] = np.sum(
            episodes[name]["load_mw"], axis=1
        ).astype(np.float32)
        npz_payload[f"{name}_mean_lmp"] = episodes[name]["mean_lmp"].astype(np.float32)
    np.savez(artifacts_dir / "policy_compare_lmp.npz", **npz_payload)

    figures = {
        "episode_mechanism": str(_plot_episode_mechanism(task_dir, episodes["ippo"], split=split)),
        "policy_compare": str(_plot_policy_compare(task_dir, episodes)),
        "training_diagnostics": str(_plot_training_diagnostics(task_dir, run_id)),
        "iid_metric_distributions": str(_plot_iid_metric_distributions(task_dir, seed=seed)),
    }

    summary = {
        "task": "gencos",
        "split": split,
        "seed": seed,
        "run_id": run_id,
        "episode_start_idx": showcase_idx,
        "all_acceptance_checks_passed": all(acceptance.values()),
        "acceptance": acceptance,
        "profit_order_reference": profit_order,
        "figures": figures,
        "policies": {
            name: {
                "total_profit": episode["total_profit"],
                "total_profit_per_agent": episode["total_profit_per_agent"],
                "avg_dispatch_share_per_agent": episode["avg_dispatch_share_per_agent"],
                "mean_action_per_agent": episode["mean_action_per_agent"],
                "mean_lmp": float(np.mean(episode["mean_lmp"])),
                "hhi_mean": episode["hhi_mean"],
                "ramp_binding_rate_mean": episode["ramp_binding_rate_mean"],
                "unsafe_steps": episode["unsafe_steps"],
                "n_violations_sum": episode["n_violations_sum"],
                "sced_convergence_rate": episode["sced_convergence_rate"],
            }
            for name, episode in episodes.items()
        },
    }

    out_path = task_dir / "results" / "analysis_episode_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[phase1_analysis] wrote {out_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", default=DEFAULT_TASK_DIR, type=Path)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--split", default="iid")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    analyze_phase1_episode(
        args.task_dir,
        run_id=args.run_id,
        split=args.split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
