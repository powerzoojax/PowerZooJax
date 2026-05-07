#!/usr/bin/env python
"""Replay formal Phase-2 DC Microgrid runs into full physical traces.

The PowerZoo cross-backend driver already saves deterministic per-step action
trajectories for each evaluation episode. This script replays those saved
actions through the frozen DC env so we can audit SOC / temperature / power
balance / SLA behavior for representative Phase-2 backend rows without
retraining or relying on transient in-memory models.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.common.powerzoo_bridge import (
    _ensure_powerzoo_path,
    _make_jax_dc_episode_params,
    _powerzoo_dc_env_from_jax_params,
    _scalarize_info,
    _wrap_powerzoo_reward_shaping,
)

TASK_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = TASK_DIR / "results"
RUNS_DIR = RESULTS_DIR / "runs"
ARTIFACTS_DIR = RESULTS_DIR / "artifacts"
FIG_DIR = RESULTS_DIR / "figures"


def _load_run(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cell_run_files(*, backend: str, device: str, algo: str) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(RUNS_DIR.glob("dc_microgrid_*_iid_*.json")):
        obj = _load_run(path)
        if obj.get("task") != "dc_microgrid" or obj.get("split") != "iid":
            continue
        if obj.get("backend") != backend or obj.get("device") != device:
            continue
        run_id = str(obj.get("run_id", ""))
        if algo == "sbx_ppo":
            if not run_id.startswith("dc_microgrid_sbx_ppo_"):
                continue
        elif algo == "ppo":
            if "ppo" not in run_id or run_id.startswith("dc_microgrid_sbx_ppo_"):
                continue
        else:
            continue
        paths.append(path)
    return paths


def _choose_representative_run(*, backend: str, device: str, algo: str) -> dict[str, Any]:
    rows = [_load_run(path) for path in _cell_run_files(backend=backend, device=device, algo=algo)]
    if len(rows) < 1:
        raise FileNotFoundError(f"No runs found for {backend}/{device}/{algo}")
    cell_mean = float(np.mean([float(r["metrics"]["episode_reward"]) for r in rows]))
    rows.sort(
        key=lambda r: (
            abs(float(r["metrics"]["episode_reward"]) - cell_mean),
            int(r.get("seed", 10**9)),
            str(r.get("run_id", "")),
        )
    )
    chosen = rows[0]
    chosen["cell_mean_episode_reward"] = cell_mean
    return chosen


def _choose_representative_episode(run_id: str, run_mean: float) -> tuple[int, dict[str, Any]]:
    per_episode_path = ARTIFACTS_DIR / f"{run_id}_per_episode.json"
    per_episode = json.loads(per_episode_path.read_text(encoding="utf-8"))
    indexed = list(enumerate(per_episode))
    indexed.sort(
        key=lambda item: (
            abs(float(item[1]["episode_reward"]) - float(run_mean)),
            item[0],
        )
    )
    return indexed[0]


def _replay_saved_actions(
    *,
    seed: int,
    episode_idx: int,
    split: str,
    actions: np.ndarray,
    n_episodes: int,
) -> list[dict[str, float]]:
    _ensure_powerzoo_path()
    params = _make_jax_dc_episode_params(
        split,
        seed=seed,
        episode_idx=episode_idx,
        n_episodes=int(n_episodes),
        strategy="uniform",
    )
    env = _wrap_powerzoo_reward_shaping(
        "dc_microgrid",
        _powerzoo_dc_env_from_jax_params(params),
    )
    try:
        obs, _ = env.reset(seed=int(seed) * 10_000 + int(episode_idx))
        del obs
        rows: list[dict[str, float]] = []
        for action in np.asarray(actions, dtype=np.float32):
            _, reward, done, truncated, info = env.step(action)
            rows.append(_scalarize_info(dict(info), float(reward)))
            if done or truncated:
                break
        return rows
    finally:
        try:
            env.close()
        except Exception:
            pass


def _representative_summary(
    *,
    cell_name: str,
    backend: str,
    device: str,
    algo: str,
    run_id: str,
    seed: int,
    split: str,
    episode_idx: int,
    selection_run_mean: float,
    selection_cell_mean: float,
    per_episode_metrics: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    soc = np.asarray([r.get("soc", r.get("battery_soc", 0.0)) for r in rows], dtype=np.float64)
    temp = np.asarray([r.get("t_zone", r.get("dc_t_zone", r.get("zone_temp_c", 0.0))) for r in rows], dtype=np.float64)
    load = np.asarray([r.get("p_load_mw", r.get("p_dc_mw", 0.0)) for r in rows], dtype=np.float64)
    pv = np.asarray([r.get("p_pv_mw", 0.0) for r in rows], dtype=np.float64)
    dg = np.asarray([r.get("p_dg_mw", 0.0) for r in rows], dtype=np.float64)
    batt = np.asarray([r.get("p_batt_mw", 0.0) for r in rows], dtype=np.float64)
    deficit = np.asarray([r.get("cost_power_deficit", 0.0) for r in rows], dtype=np.float64)
    spill = np.asarray([r.get("cost_power_spill", r.get("power_spill", 0.0)) for r in rows], dtype=np.float64)
    costs_sla = np.asarray([r.get("cost_sla", 0.0) for r in rows], dtype=np.float64)
    rewards = np.asarray([r.get("reward_shaped", r.get("reward", 0.0)) for r in rows], dtype=np.float64)
    grid = np.asarray([r.get("p_grid_import_mw", 0.0) for r in rows], dtype=np.float64)
    balance = np.abs(load - (pv + dg + batt + grid)) / np.maximum(load, 1e-6)
    return {
        "cell_name": cell_name,
        "backend": backend,
        "device": device,
        "algo": algo,
        "selection_rule": "representative seed closest to 5-seed cell mean; representative episode closest to selected run mean",
        "representative_run_id": run_id,
        "representative_seed": int(seed),
        "representative_split": split,
        "representative_episode_idx": int(episode_idx),
        "cell_mean_episode_reward": float(selection_cell_mean),
        "run_mean_episode_reward": float(selection_run_mean),
        "selected_episode_reward": float(per_episode_metrics["episode_reward"]),
        "replayed_episode_reward": float(rewards.sum()),
        "soc_min": float(soc.min()),
        "soc_max": float(soc.max()),
        "temperature_c_min": float(temp.min()),
        "temperature_c_max": float(temp.max()),
        "mean_power_balance_error": float(balance.mean()),
        "max_power_balance_error": float(balance.max()),
        "sla_deficit_any": bool(np.any(costs_sla > 0)),
        "mean_cost_power_deficit": float(deficit.mean()),
        "mean_cost_power_spill": float(spill.mean()),
        "soc_within_bounds": bool(np.all((soc >= 0.15 - 1e-9) & (soc <= 0.90 + 1e-9))),
        "power_balance_below_1pct": bool(float(balance.max()) < 0.01),
        "temperature_within_15_40c": bool(float(temp.min()) >= 15.0 and float(temp.max()) <= 40.0),
    }


def _save_fig(fig, name: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")


def _plot_dispatch(rows: list[dict[str, Any]], *, prefix: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = np.arange(len(rows)) * (5.0 / 60.0)
    p_pv = np.asarray([r.get("p_pv_mw", 0.0) for r in rows], dtype=np.float64)
    p_dg = np.asarray([r.get("p_dg_mw", 0.0) for r in rows], dtype=np.float64)
    p_batt = np.asarray([r.get("p_batt_mw", 0.0) for r in rows], dtype=np.float64)
    p_grid = np.asarray([r.get("p_grid_import_mw", 0.0) for r in rows], dtype=np.float64)
    p_dc = np.asarray([r.get("p_dc_mw", r.get("p_load_mw", 0.0)) for r in rows], dtype=np.float64)
    batt_dis = np.where(p_batt > 0.0, p_batt, 0.0)
    batt_chg = np.where(p_batt < 0.0, -p_batt, 0.0)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.fill_between(h, 0, p_pv, label="PV generation", color="#f9c74f", alpha=0.42)
    ax.plot(h, p_pv, color="#d99500", lw=1.5)
    ax.plot(h, p_dg, color="#7b4f43", lw=1.8, label="Diesel generator")
    ax.plot(h, p_grid, color="#5e35b1", lw=1.7, label="Grid import")
    ax.plot(h, batt_dis, color="#2e7d32", lw=1.5, label="Battery discharge")
    ax.plot(h, -batt_chg, color="#1565c0", lw=1.5, label="Battery charge")
    ax.plot(h, p_dc, color="#c62828", lw=2.2, label="DC load")
    ax.axhline(0.0, color="#9e9e9e", lw=0.7)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Power [MW]")
    ax.set_title(title)
    ax.grid(True, alpha=0.28)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        fontsize=9,
        ncol=6,
        frameon=False,
    )
    fig.tight_layout()
    _save_fig(fig, f"{prefix}_dispatch_representative")
    plt.close(fig)


def _plot_mechanism(rows: list[dict[str, Any]], *, prefix: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = np.arange(len(rows)) * (5.0 / 60.0)
    soc = np.asarray([r.get("soc", 0.0) for r in rows], dtype=np.float64)
    temp = np.asarray([r.get("dc_t_zone", 0.0) for r in rows], dtype=np.float64)
    reward = np.asarray([r.get("reward_shaped", r.get("reward", 0.0)) for r in rows], dtype=np.float64)
    raw = np.asarray([r.get("raw_reward", 0.0) for r in rows], dtype=np.float64)
    penalty = np.asarray([r.get("shaping_penalty", 0.0) for r in rows], dtype=np.float64)
    a_train = np.asarray([r.get("a_train_sched", 0.0) for r in rows], dtype=np.float64)
    a_ft = np.asarray([r.get("a_ft_sched", 0.0) for r in rows], dtype=np.float64)
    a_cool = np.asarray([r.get("a_cool", 0.0) for r in rows], dtype=np.float64)
    a_batt = np.asarray([r.get("a_batt", 0.0) for r in rows], dtype=np.float64)
    a_dg = np.asarray([r.get("a_dg", 0.0) for r in rows], dtype=np.float64)
    deficit = np.asarray([r.get("cost_power_deficit", 0.0) for r in rows], dtype=np.float64)
    spill = np.asarray([r.get("power_spill_rate", 0.0) for r in rows], dtype=np.float64)

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(h, soc, color="#2e7d32", lw=1.8, label="SOC")
    axes[0].axhline(0.15, color="#c62828", ls="--", lw=0.8)
    axes[0].axhline(0.9, color="#2e7d32", ls="--", lw=0.8)
    ax0b = axes[0].twinx()
    ax0b.plot(h, temp, color="#ef6c00", lw=1.4, label="Zone temp")
    axes[0].set_ylabel("SOC")
    ax0b.set_ylabel("Temp [C]")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(h, raw, color="#1565c0", lw=1.1, label="Raw reward")
    axes[1].plot(h, -penalty, color="#c62828", lw=1.1, label="-Shaping penalty")
    axes[1].plot(h, reward, color="black", lw=1.5, label="Shaped reward")
    axes[1].axhline(0, color="#9e9e9e", lw=0.6)
    axes[1].set_ylabel("Reward")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right", fontsize=8, ncol=3)

    axes[2].plot(h, a_train, label="train_sched", lw=1.0)
    axes[2].plot(h, a_ft, label="ft_sched", lw=1.0)
    axes[2].plot(h, a_cool, label="cool", lw=1.0)
    axes[2].plot(h, a_batt, label="batt", lw=1.0)
    axes[2].plot(h, a_dg, label="dg", lw=1.0)
    axes[2].set_ylabel("Action")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="upper right", fontsize=8, ncol=5)

    axes[3].plot(h, deficit, color="#d32f2f", lw=1.2, label="Deficit cost")
    axes[3].plot(h, spill, color="#6a1b9a", lw=1.2, label="Spill proxy")
    axes[3].set_xlabel("Hour of day")
    axes[3].set_ylabel("Cost")
    axes[3].grid(True, alpha=0.25)
    axes[3].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    _save_fig(fig, f"{prefix}_mechanism_representative")
    plt.close(fig)


def _audit_one(*, cell_name: str, backend: str, device: str, algo: str) -> dict[str, Any]:
    run = _choose_representative_run(backend=backend, device=device, algo=algo)
    run_id = str(run["run_id"])
    seed = int(run["seed"])
    run_mean = float(run["metrics"]["episode_reward"])
    episode_idx, episode_metrics = _choose_representative_episode(run_id, run_mean)

    actions_path = ARTIFACTS_DIR / f"{run_id}_trajectory.npz"
    action_bank = np.load(actions_path)["actions"]
    rows = _replay_saved_actions(
        seed=seed,
        episode_idx=episode_idx,
        split="iid",
        actions=action_bank[episode_idx],
        n_episodes=int(action_bank.shape[0]),
    )

    summary = _representative_summary(
        cell_name=cell_name,
        backend=backend,
        device=device,
        algo=algo,
        run_id=run_id,
        seed=seed,
        split="iid",
        episode_idx=episode_idx,
        selection_run_mean=run_mean,
        selection_cell_mean=float(run["cell_mean_episode_reward"]),
        per_episode_metrics=episode_metrics,
        rows=rows,
    )
    prefix = f"phase2_{backend}_{device}_{algo}_iid"
    title_prefix = f"Phase-2 {backend}+{device} {algo}"
    _plot_dispatch(rows, prefix=prefix, title=f"{title_prefix} representative dispatch")
    _plot_mechanism(rows, prefix=prefix, title=f"{title_prefix} representative mechanism")
    return summary


def main() -> None:
    payload = {
        "sbx_cuda": _audit_one(cell_name="sbx_cuda", backend="sbx", device="cuda", algo="sbx_ppo"),
        "sb3_cuda": _audit_one(cell_name="sb3_cuda", backend="sb3", device="cuda", algo="ppo"),
    }
    out = RESULTS_DIR / "phase2_backend_physical_audit.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[phase2_physical_audit] wrote {out}")


if __name__ == "__main__":
    main()
