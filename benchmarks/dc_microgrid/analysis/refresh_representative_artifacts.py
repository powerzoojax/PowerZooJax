#!/usr/bin/env python
"""Refresh representative DC Microgrid physical-audit artifacts.

Produces:
  - results/representative_episode_summary.json
  - results/figures/sac_iid_dispatch_representative.{pdf,png}
  - results/figures/sac_iid_representative_mechanism.{pdf,png}
  - results/figures/iid_sla_deficit_violation.{pdf,png}

Defaults target the latest completed `jax_rejax + gpu` SAC seed-0 train run on
the IID split and refresh the figure set used in the paper text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TASK_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = TASK_DIR / "results" / "figures"


def _figure_rc() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "grid.linewidth": 0.4,
        }
    )


def _latest_train_run_id(algo: str, seed: int, backend: str, device: str) -> str:
    manifest = json.loads((TASK_DIR / "results" / "manifest.json").read_text(encoding="utf-8"))
    best: tuple[str, str] | None = None
    for rec in manifest:
        if (
            rec.get("task") == "dc_microgrid"
            and rec.get("algo") == algo
            and rec.get("split") == "train"
            and rec.get("status") == "completed"
            and rec.get("backend", "jax_rejax") == backend
            and rec.get("device", "gpu") == device
            and int(rec.get("seed", -1)) == seed
            and "eval of " not in (rec.get("notes") or "")
        ):
            ts = rec.get("timestamp", "")
            if best is None or ts > best[1]:
                best = (rec["run_id"], ts)
    if best is None:
        raise FileNotFoundError(
            f"No completed train run found for algo={algo} seed={seed} "
            f"backend={backend} device={device}"
        )
    return best[0]


def _rollout_rows(
    run_id: str,
    algo: str,
    split: str,
    seed: int,
    episode_idx: int,
    n_episodes_span: int,
):
    from benchmarks.common.configs import load_config, load_task_config
    from benchmarks.common.io import load_pickle, load_run
    from benchmarks.common.runtime import build_train_cfg, make_policy_fn
    from benchmarks.dc_microgrid._reward_shaping import wrap_with_shaping
    from benchmarks.dc_microgrid.analysis.export_episode_excel import _rollout_one
    from benchmarks.dc_microgrid.rejax_ckpt import load_sac_train_state
    from powerzoojax.envs.microgrid import DataCenterMicrogridEnv
    from powerzoojax.tasks.dc_microgrid import DCMicrogridTask

    import jax

    task_cfg = load_task_config(TASK_DIR)
    max_steps = int(task_cfg.get("max_steps", 288))

    task = DCMicrogridTask(
        source=task_cfg.get("data_source", "google"),
        max_steps=max_steps,
        case_overrides=task_cfg.get("case_overrides") or {},
    )
    base_env = DataCenterMicrogridEnv()
    env = wrap_with_shaping(base_env, task_cfg)
    params = task.episode_params(
        split,
        episode_idx,
        n_episodes_span,
        max_steps,
        strategy="uniform",
        seed=seed,
    )

    rec = load_run(run_id, TASK_DIR)
    train_path = TASK_DIR / "configs" / f"train_{algo}.yaml"
    train_cfg = build_train_cfg(load_config(train_path), algo=algo)
    if algo == "sac":
        rel = rec.artifacts.get("params_orbax")
        if not rel:
            raise FileNotFoundError(f"SAC run {run_id} missing params_orbax artifact")
        train_state = load_sac_train_state(TASK_DIR / "results" / rel, train_cfg, env, params)
    else:
        rel = rec.artifacts.get("params")
        if not rel:
            raise FileNotFoundError(f"PPO run {run_id} missing params artifact")
        train_state = load_pickle(TASK_DIR / "results" / rel)

    policy_fn = make_policy_fn(algo, train_state, env, params, train_cfg)
    key = jax.random.PRNGKey(seed * 1_000_000 + episode_idx)
    return _rollout_one(env, params, key, max_steps, policy_fn, include_state_diag=True)


def _representative_summary(rows: list[dict[str, Any]], run_id: str, split: str, episode_idx: int) -> dict[str, Any]:
    soc = np.asarray([r.get("soc", r.get("battery_soc", 0.0)) for r in rows], dtype=np.float64)
    temp = np.asarray([r.get("dc_t_zone", r.get("zone_temp_c", 0.0)) for r in rows], dtype=np.float64)
    load = np.asarray([r.get("p_dc_mw", r.get("p_load_mw", 0.0)) for r in rows], dtype=np.float64)
    pv = np.asarray([r.get("p_pv_mw", 0.0) for r in rows], dtype=np.float64)
    dg = np.asarray([r.get("p_dg_mw", 0.0) for r in rows], dtype=np.float64)
    batt = np.asarray([r.get("p_batt_mw", 0.0) for r in rows], dtype=np.float64)
    grid = np.asarray([r.get("p_grid_import_mw", 0.0) for r in rows], dtype=np.float64)
    price = np.asarray([r.get("grid_price_per_mwh", 0.0) for r in rows], dtype=np.float64)
    deficit = np.asarray([r.get("cost_power_deficit", 0.0) for r in rows], dtype=np.float64)
    spill = np.asarray([r.get("power_spill_rate", 0.0) for r in rows], dtype=np.float64)
    balance = np.abs(load - (pv + dg + batt + grid)) / np.maximum(load, 1e-6)
    costs_sla = np.asarray([r.get("cost_sla", 0.0) for r in rows], dtype=np.float64)
    rewards = np.asarray([r.get("reward_shaped", 0.0) for r in rows], dtype=np.float64)
    dt_h = 5.0 / 60.0
    batt_corr = 0.0
    if np.std(price) > 1e-9 and np.std(batt) > 1e-9:
        batt_corr = float(np.corrcoef(price, batt)[0, 1])
    return {
        "representative_run_id": run_id,
        "representative_split": split,
        "representative_episode_idx": int(episode_idx),
        "episode_reward": float(rewards.sum()),
        "soc_min": float(soc.min()),
        "soc_max": float(soc.max()),
        "temperature_c_min": float(temp.min()),
        "temperature_c_max": float(temp.max()),
        "mean_power_balance_error": float(balance.mean()),
        "max_power_balance_error": float(balance.max()),
        "sla_deficit_any": bool(np.any(costs_sla > 0)),
        "mean_cost_power_deficit": float(deficit.mean()),
        "mean_cost_power_spill": float(spill.mean()),
        "pv_peak_mw": float(pv.max()),
        "pv_energy_mwh": float(pv.sum() * dt_h),
        "grid_import_mwh": float(grid.sum() * dt_h),
        "grid_cost": float(np.sum([r.get("grid_cost", 0.0) for r in rows])),
        "price_min_per_mwh": float(price.min()),
        "price_max_per_mwh": float(price.max()),
        "price_mean_per_mwh": float(price.mean()),
        "battery_charge_steps": int(np.sum(batt < -1e-6)),
        "battery_discharge_steps": int(np.sum(batt > 1e-6)),
        "price_battery_power_corr": batt_corr,
    }


def _save_json(summary: dict[str, Any]) -> Path:
    out = TASK_DIR / "results" / "representative_episode_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def _save_fig(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / f"{name}.pdf"
    png = FIG_DIR / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=150, bbox_inches="tight")


def _coarsen_mean(x: np.ndarray, group: int = 6) -> np.ndarray:
    if group <= 1 or x.size < group:
        return x
    trimmed = x[: (x.size // group) * group]
    return trimmed.reshape(-1, group).mean(axis=1)


def _plot_dispatch(
    rows: list[dict[str, Any]],
    *,
    split: str,
    algo: str,
    episode_idx: int,
    pv_capacity_mw: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _figure_rc()

    h_raw = np.arange(len(rows)) * (5.0 / 60.0)
    p_pv = np.asarray([r.get("p_pv_mw", 0.0) for r in rows], dtype=np.float64)
    p_dg = np.asarray([r.get("p_dg_mw", 0.0) for r in rows], dtype=np.float64)
    p_batt = np.asarray([r.get("p_batt_mw", 0.0) for r in rows], dtype=np.float64)
    p_grid = np.asarray([r.get("p_grid_import_mw", 0.0) for r in rows], dtype=np.float64)
    price = np.asarray([r.get("grid_price_per_mwh", 0.0) for r in rows], dtype=np.float64)
    soc_raw = np.asarray([r.get("soc", 0.0) for r in rows], dtype=np.float64)
    p_dc = np.asarray([r.get("p_dc_mw", r.get("p_load_mw", 0.0)) for r in rows], dtype=np.float64)
    batt_dis = np.where(p_batt > 0.0, p_batt, 0.0)
    batt_chg = np.where(p_batt < 0.0, -p_batt, 0.0)

    # Display at 30-minute resolution. The underlying rollout remains 5-minute;
    # this is only a paper-facing visualization choice that matches the solar
    # source resolution and removes workload-control jitter from the figure.
    h = _coarsen_mean(h_raw)
    pv = _coarsen_mean(p_pv)
    dg = _coarsen_mean(p_dg)
    grid = _coarsen_mean(p_grid)
    load = _coarsen_mean(p_dc)
    dis = _coarsen_mean(batt_dis)
    chg = _coarsen_mean(batt_chg)
    price_30 = _coarsen_mean(price)
    soc = _coarsen_mean(soc_raw)
    pv_cf = np.clip(pv / max(float(pv_capacity_mw), 1e-9), 0.0, 1.0)
    net_load = load - pv

    # Single shared fig.legend above the three panels: anchor its bottom just
    # above the subplot tops (axes top ≈ 0.67 in fig coords) so there is no
    # large empty band; loc="lower center" places the legend upward from the anchor.
    fig = plt.figure(figsize=(10.0, 3.65))
    # [left, bottom, width, height] — slightly narrower panels + wider
    # inter-panel gap than 0.226@uniform spacing (was ~0.058, now ~0.072).
    axes = [
        fig.add_axes([0.068, 0.165, 0.218, 0.505]),
        fig.add_axes([0.358, 0.165, 0.218, 0.505]),
        fig.add_axes([0.648, 0.165, 0.218, 0.505]),
    ]
    fig.patch.set_facecolor("white")
    legend_handles = []
    legend_labels = []

    def _add_legend_items(handles, labels):
        for handle, label in zip(handles, labels):
            if label in legend_labels:
                continue
            legend_handles.append(handle)
            legend_labels.append(label)

    ax = axes[0]
    ax.plot(h, load, color="#c62828", lw=2.0, label="DC load")
    ax.set_ylabel("Load [MW]")
    ax.set_xlabel("Hour of day")
    ax.grid(True, alpha=0.26)
    ax2 = ax.twinx()
    ax2.fill_between(h, 0.0, pv_cf, color="#f9c74f", alpha=0.45, step="mid", label="PV avail.")
    ax2.plot(h, pv_cf, color="#d99500", lw=1.4)
    ax2.text(0.97, 0.06, "PV CF", transform=ax2.transAxes, ha="right", va="bottom", fontsize=11)
    ax2.set_ylim(-0.02, 1.02)
    ax2.tick_params(axis="y", right=False, labelright=False)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    _add_legend_items(lines1 + lines2, labels1 + labels2)

    ax = axes[1]
    ax.plot(h, net_load, color="#ad1457", lw=2.0, label="Net load")
    ax.plot(h, grid, color="#455a64", lw=1.9, label="Grid import")
    ax.plot(h, dg, color="#6d4c41", lw=1.9, label="Diesel gen.")
    ax.fill_between(h, 0.0, dis, color="#2e7d32", alpha=0.25, step="mid", label="Batt. discharge")
    ax.plot(h, dis, color="#2e7d32", lw=1.4)
    ax.plot(h, -chg, color="#1565c0", lw=1.4, label="Batt. charge")
    ax.axhline(0.0, color="#9e9e9e", lw=0.7)
    ax.set_ylabel("Power [MW]", labelpad=1.0)
    ax.set_xlabel("Hour of day")
    ax.grid(True, alpha=0.26)
    lines, labels = ax.get_legend_handles_labels()
    _add_legend_items(lines, labels)

    ax = axes[2]
    ax.plot(h, soc, color="#2e7d32", lw=2.0, label="Battery SoC")
    ax.axhline(0.15, color="#c62828", ls="--", lw=0.9, label="SoC bounds")
    ax.axhline(0.90, color="#c62828", ls="--", lw=0.9)
    ax.set_ylim(0.05, 0.98)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("SoC")
    ax.grid(True, alpha=0.26)
    ax2 = ax.twinx()
    ax2.plot(h, price_30, color="#111111", lw=1.4, alpha=0.82, label="Grid price")
    ax2.set_ylabel(r"Price [$\mathrm{\$/MWh}$]", labelpad=6)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    _add_legend_items(lines1 + lines2, labels1 + labels2)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.705),
        handlelength=1.2,
        handletextpad=0.35,
        columnspacing=0.75,
        borderaxespad=0.25,
        fontsize=11,
    )
    for ax in axes:
        ax.set_xticks([0, 6, 12, 18])
        ax.set_xlim(-0.6, 23.6)
    _save_fig(fig, f"{algo}_{split}_dispatch_representative")
    plt.close(fig)


def _plot_mechanism(rows: list[dict[str, Any]], *, split: str, algo: str, episode_idx: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _figure_rc()

    h = np.arange(len(rows)) * (5.0 / 60.0)
    soc = np.asarray([r.get("soc", 0.0) for r in rows], dtype=np.float64)
    price = np.asarray([r.get("grid_price_per_mwh", 0.0) for r in rows], dtype=np.float64)
    temp = np.asarray([r.get("dc_t_zone", 0.0) for r in rows], dtype=np.float64)
    reward = np.asarray([r.get("reward_shaped", 0.0) for r in rows], dtype=np.float64)
    raw = np.asarray([r.get("raw_reward", 0.0) for r in rows], dtype=np.float64)
    penalty = np.asarray([r.get("shaping_penalty", 0.0) for r in rows], dtype=np.float64)
    a_train = np.asarray([r.get("a_train_sched", 0.0) for r in rows], dtype=np.float64)
    a_ft = np.asarray([r.get("a_ft_sched", 0.0) for r in rows], dtype=np.float64)
    a_cool = np.asarray([r.get("a_cool", 0.0) for r in rows], dtype=np.float64)
    a_batt = np.asarray([r.get("a_batt", 0.0) for r in rows], dtype=np.float64)
    a_dg = np.asarray([r.get("a_dg", 0.0) for r in rows], dtype=np.float64)
    deficit = np.asarray([r.get("cost_power_deficit", 0.0) for r in rows], dtype=np.float64)
    spill = np.asarray([r.get("power_spill_rate", 0.0) for r in rows], dtype=np.float64)

    fig, axes = plt.subplots(4, 1, figsize=(10.0, 9.09), sharex=True)
    fig.patch.set_facecolor("white")

    axes[0].plot(h, soc, color="#2e7d32", lw=1.8, label="SOC")
    axes[0].axhline(0.15, color="#c62828", ls="--", lw=0.8)
    axes[0].axhline(0.9, color="#2e7d32", ls="--", lw=0.8)
    ax0b = axes[0].twinx()
    ax0b.plot(h, price, color="#111111", lw=1.2, alpha=0.8, label="Grid price")
    axes[0].set_ylabel("SOC")
    ax0b.set_ylabel("Price [$/MWh]")
    axes[0].set_title(f"{algo.upper()} representative mechanism ({split}, ep={episode_idx})")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(h, raw, color="#1565c0", lw=1.1, label="Raw reward")
    axes[1].plot(h, -penalty, color="#c62828", lw=1.1, label="-Shaping penalty")
    axes[1].plot(h, reward, color="black", lw=1.5, label="Shaped reward")
    axes[1].axhline(0, color="#9e9e9e", lw=0.6)
    axes[1].set_ylabel("Reward")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right", fontsize=10, ncol=3)

    axes[2].plot(h, a_train, label="train_sched", lw=1.0)
    axes[2].plot(h, a_ft, label="ft_sched", lw=1.0)
    axes[2].plot(h, a_cool, label="cool", lw=1.0)
    axes[2].plot(h, a_batt, label="batt", lw=1.0)
    axes[2].plot(h, a_dg, label="dg", lw=1.0)
    axes[2].set_ylabel("Action")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="upper right", fontsize=10, ncol=5)

    axes[3].plot(h, temp, color="#ef6c00", lw=1.2, label="Zone temp [C]")
    axes[3].plot(h, deficit, color="#d32f2f", lw=1.2, label="Deficit cost")
    axes[3].plot(h, spill, color="#6a1b9a", lw=1.2, label="Spill proxy")
    axes[3].set_xlabel("Hour of day")
    axes[3].set_ylabel("Cost")
    axes[3].grid(True, alpha=0.25)
    axes[3].legend(loc="upper right", fontsize=10)

    fig.tight_layout()
    _save_fig(fig, f"{algo}_{split}_representative_mechanism")
    plt.close(fig)


def _plot_iid_sla_deficit() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _figure_rc()

    summary_path = TASK_DIR / "results" / "summary" / "latest.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = summary.get("rows", [])
    wanted = ["no_control", "max_renewable", "rule_based", "ppo", "sac"]
    iid_rows = {
        r["algo"]: r for r in rows
        if r.get("backend") == "jax_rejax" and r.get("split") == "iid" and r.get("algo") in wanted
    }
    algos = [a for a in wanted if a in iid_rows]
    x = np.arange(len(algos))
    sla = np.asarray([iid_rows[a].get("sla_violation_rate_mean", 0.0) or 0.0 for a in algos], dtype=np.float64)
    deficit = np.asarray([iid_rows[a].get("power_deficit_rate_mean", 0.0) or 0.0 for a in algos], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10.0, 5.61))
    fig.patch.set_facecolor("white")
    width = 0.36
    ax.bar(x - width / 2, sla, width=width, color="#c62828", alpha=0.85, label="SLA violation rate")
    ax.bar(x + width / 2, deficit, width=width, color="#1565c0", alpha=0.85, label="Power deficit rate")
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("_", "\n") for a in algos])
    ax.set_ylabel("Rate")
    ax.set_title("IID SLA deficit / violation")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    _save_fig(fig, "iid_sla_deficit_violation")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="sac", choices=["ppo", "sac"])
    parser.add_argument("--split", default="iid")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-idx", type=int, default=5)
    parser.add_argument("--n-episodes-span", type=int, default=20)
    parser.add_argument("--backend", default="jax_rejax")
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    try:
        import yaml
        task_cfg = yaml.safe_load((TASK_DIR / "configs" / "task.yaml").read_text(encoding="utf-8")) or {}
    except Exception:
        task_cfg = {}
    # Match the paper figure (build_dcmg_paper_figure.py): episode 5 of a
    # 20-episode IID partition. Override via CLI for other audits.
    n_episodes_span = int(args.n_episodes_span)
    case_overrides = task_cfg.get("case_overrides") or {}
    pv_capacity_mw = float(case_overrides.get("pv_p_max_mw", 0.4))

    run_id = args.run_id or _latest_train_run_id(
        algo=args.algo,
        seed=args.seed,
        backend=args.backend,
        device=args.device,
    )
    rows = _rollout_rows(
        run_id=run_id,
        algo=args.algo,
        split=args.split,
        seed=args.seed,
        episode_idx=args.episode_idx,
        n_episodes_span=n_episodes_span,
    )
    summary = _representative_summary(rows, run_id, args.split, args.episode_idx)
    summary["representative_n_episodes_span"] = int(n_episodes_span)
    summary["pv_capacity_mw"] = float(pv_capacity_mw)
    summary["pv_peak_capacity_factor"] = float(summary["pv_peak_mw"] / max(pv_capacity_mw, 1e-9))
    out = _save_json(summary)
    _plot_dispatch(
        rows,
        split=args.split,
        algo=args.algo,
        episode_idx=args.episode_idx,
        pv_capacity_mw=pv_capacity_mw,
    )
    _plot_mechanism(rows, split=args.split, algo=args.algo, episode_idx=args.episode_idx)
    _plot_iid_sla_deficit()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[refresh_representative_artifacts] wrote {out}")


if __name__ == "__main__":
    main()
