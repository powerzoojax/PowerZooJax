#!/usr/bin/env python3
"""TSO n=256, 4M tuning sweep for PPO / SAC / Sauté.

Does **not** increase ``total_timesteps``.  Compare final eval cost to the plain
``*_n256_t4m`` budget_scan baselines in the manifest.

Usage::

    python benchmarks/tso/n256_epoch_sweep.py run
    python benchmarks/tso/n256_epoch_sweep.py plot   # -> figures/n256_epoch_sweep.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

TASK_DIR = Path(__file__).resolve().parent

RUNS: list[tuple[str, str, str]] = [
    ("ppo", "configs/scan_ppo_n256_4M_epochs8.yaml", "n256_ppo_e8"),
    ("ppo", "configs/scan_ppo_n256_4M_epochs10.yaml", "n256_ppo_e10"),
    ("ppo", "configs/scan_ppo_n256_4M_steps24_mb8.yaml", "n256_ppo_s24_mb8"),
    ("sac", "configs/scan_sac_n256_4M_sac_epochs2.yaml", "n256_sac_se2"),
    ("sac", "configs/scan_sac_n256_4M_sac_epochs3.yaml", "n256_sac_se3"),
    (
        "saute_ppo",
        "configs/scan_saute_n256_4M_steps24_mb8_vecbudget_t10_r100_unsafe30.yaml",
        "n256_saute_s24_mb8_vec",
    ),
]


def main() -> None:
    from benchmarks.tso.train import train_tso

    out: list[dict] = []
    for algo, rel_cfg, tag in RUNS:
        cfg_path = TASK_DIR / rel_cfg
        print(f"[n256_epoch_sweep] {tag} <- {cfg_path.name}")
        rec = train_tso(
            TASK_DIR,
            algo=algo,
            seed=0,
            config_path=str(cfg_path),
            extra_notes=f"n256_epoch_sweep={tag}",
        )
        out.append(
            {
                "tag": tag,
                "algo": algo,
                "run_id": rec.run_id,
                "walltime_s": rec.walltime_s,
                "status": rec.status,
                "final_reward": (rec.metrics or {}).get("final_reward"),
            }
        )
    p = TASK_DIR / "results" / "budget_scan" / "n256_epoch_sweep.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[n256_epoch_sweep] wrote {p}")


def _latest_row_notes_contain(task_dir: Path, substr: str) -> dict | None:
    """Latest ``train`` row whose run metadata contains ``substr`` (by timestamp)."""
    mpath = task_dir / "results" / "manifest.json"
    if not mpath.exists():
        return None
    data = json.loads(mpath.read_text(encoding="utf-8"))
    best: dict | None = None
    best_ts = ""
    for r in data:
        if r.get("task") != "tso" or r.get("split") != "train":
            continue
        notes = str(r.get("notes") or "")
        if substr not in notes:
            run_id = str(r.get("run_id") or "")
            if not run_id:
                continue
            run_path = task_dir / "results" / "runs" / f"{run_id}.json"
            run_data = {}
            if run_path.exists():
                try:
                    run_data = json.loads(run_path.read_text(encoding="utf-8"))
                except Exception:
                    run_data = {}
            cfg_notes = ""
            art_cfg_path = task_dir / "results" / "artifacts" / f"{run_id}_config.json"
            if art_cfg_path.exists():
                try:
                    art_cfg = json.loads(art_cfg_path.read_text(encoding="utf-8"))
                    raw_cfg = art_cfg.get("train_config_raw") or {}
                    cfg_notes = str(raw_cfg.get("notes") or "")
                except Exception:
                    cfg_notes = ""
            run_notes = str(run_data.get("notes") or "")
            if substr not in cfg_notes and substr not in run_notes:
                continue
        ts = str(r.get("timestamp") or "")
        if best is None or ts > best_ts:
            best, best_ts = r, ts
    return best


# (legend label, notes substring, color, linestyle)
_SWEEP_SERIES: list[tuple[str, str, str, str]] = [
    (
        "PPO-Lag  n_steps=24, mb=8",
        "TSO PPO-Lagrangian n256 4M with n_steps=24 and n_minibatches=8.",
        "#c2185b",
        "solid",
    ),
    (
        "PPO-Lag  s24/mb8 + gentle thermal margin",
        "TSO PPO-Lagrangian n256 4M steps24/mb8 with gentle thermal margin:",
        "#ff7043",
        "solid",
    ),
    ("PPO  n_epochs=8", "n256_epoch_sweep=n256_ppo_e8", "#1565c0", "solid"),
    ("PPO  n_epochs=10", "n256_epoch_sweep=n256_ppo_e10", "#42a5f5", "solid"),
    ("PPO  n_steps=24, mb=8", "n256_epoch_sweep=n256_ppo_s24_mb8", "#0d47a1", "solid"),
    ("SAC  sac_num_epochs=2", "n256_epoch_sweep=n256_sac_se2", "#00838f", "solid"),
    ("SAC  sac_num_epochs=3", "n256_epoch_sweep=n256_sac_se3", "#26c6da", "solid"),
    (
        "Sauté  thermal=10, reserve=100",
        "n256_epoch_sweep=n256_saute_s24_mb8_vec",
        "#8e24aa",
        "solid",
    ),
    (
        "Sauté  thermal=8, reserve=100, unsafe=-40",
        "thermal budget=8, reserve budget=100, unsafe_reward=-40.",
        "#6a1b9a",
        "solid",
    ),
    (
        "Sauté  early-stop 3.6M (t=10, r=100, u=-30)",
        "TSO Sauté PPO n256 3.6M early-stop with n_steps=24, mb=8, thermal budget=10, reserve budget=100, unsafe_reward=-30.",
        "#ab47bc",
        "solid",
    ),
    ("PPO-Lag  baseline 4M", "budget_scan=ppo_lagrangian_n256_t4m", "#d81b60", "dotted"),
    ("PPO  baseline 4M (e=4)", "budget_scan=ppo_n256_t4m", "#283593", "dotted"),
    ("SAC  baseline 4M (Rejax def.)", "budget_scan=sac_n256_t4m", "#004d40", "dotted"),
]


def plot_sweep(
    task_dir: Path = TASK_DIR,
) -> Path:
    """Draw n256 sweep cost and safety curves vs timesteps plus cost vs wall time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from benchmarks.common.configs import load_task_config
    from benchmarks.tso.budget_scan import _load_eval_cost_and_axes

    tcfg = load_task_config(task_dir)
    reward_scale = float(tcfg.get("reward_scale", 1e-4))
    fig_dir = task_dir / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    series: list[tuple[str, str, str, str, dict | None]] = []
    for label, sub, color, ls in _SWEEP_SERIES:
        rec = _latest_row_notes_contain(task_dir, sub)
        if rec is None:
            print(f"[n256_epoch_sweep] missing manifest row for: {sub!r}")
            continue
        series.append(
            (label, str(rec.get("run_id", "")), color, ls, rec),
        )

    if not series:
        print("[n256_epoch_sweep] No matching manifest rows. Run: python benchmarks/tso/n256_epoch_sweep.py run")
        return Path()

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.4))
    ax_cost_t, ax_cost_w = axes[0]
    ax_reserve_t, ax_thermal_t = axes[1]

    for label, run_id, color, ls, rec_m in series:
        if not run_id:
            continue
        cost, tsv, wsec = _load_eval_cost_and_axes(task_dir, run_id, reward_scale)
        if cost.size == 0:
            print(f"[n256_epoch_sweep] skip {label}: no eval curve")
            continue
        y = cost / 1e6
        reserve = None
        thermal = None
        adir = task_dir / "results" / "artifacts"
        reserve_path = adir / f"{run_id}_eval_reserve_shortfall_rate.npy"
        thermal_path = adir / f"{run_id}_eval_thermal_violation_rate.npy"
        if reserve_path.exists():
            reserve = np.asarray(np.load(reserve_path), dtype=float).ravel()
        if thermal_path.exists():
            thermal = np.asarray(np.load(thermal_path), dtype=float).ravel()
        if tsv is not None and tsv.size == cost.size:
            x_t = tsv / 1e6
        else:
            x_t = np.linspace(0.0, 4.0, cost.size)
        ax_cost_t.plot(
            x_t, y, color=color, linestyle=ls, linewidth=1.9, label=label, alpha=0.95
        )

        wtot = (rec_m or {}).get("walltime_s")
        if wsec is not None and wsec.size == cost.size:
            ax_cost_w.plot(
                wsec / 60.0,
                y,
                color=color,
                linestyle=ls,
                linewidth=1.9,
                label=label,
                alpha=0.95,
            )
        elif wtot and tsv is not None and tsv.size and float(tsv.max()) > 0:
            x_w = tsv / float(tsv.max()) * (float(wtot) / 60.0)
            ax_cost_w.plot(
                x_w, y, color=color, linestyle=ls, linewidth=1.9, label=label, alpha=0.95
            )

        if reserve is not None and reserve.size == cost.size:
            ax_reserve_t.plot(
                x_t,
                reserve,
                color=color,
                linestyle=ls,
                linewidth=1.9,
                label=label,
                alpha=0.95,
            )
        if thermal is not None and thermal.size == cost.size:
            ax_thermal_t.plot(
                x_t,
                thermal,
                color=color,
                linestyle=ls,
                linewidth=1.9,
                label=label,
                alpha=0.95,
            )

    ax_cost_t.set_xlabel("Env timesteps (millions)")
    ax_cost_t.set_ylabel("Eval operating cost (million GBP)")
    ax_cost_t.set_title("Eval cost vs steps (4M, seed 0)")
    ax_cost_t.grid(True, alpha=0.3)
    ax_cost_t.legend(fontsize=7, loc="best")

    ax_cost_w.set_xlabel("Wall time (minutes)")
    ax_cost_w.set_ylabel("Eval operating cost (million GBP)")
    ax_cost_w.set_title("Eval cost vs wall time")
    ax_cost_w.grid(True, alpha=0.3)
    ax_cost_w.legend(fontsize=7, loc="best")

    ax_reserve_t.set_xlabel("Env timesteps (millions)")
    ax_reserve_t.set_ylabel("Reserve shortfall rate")
    ax_reserve_t.set_title("Safety vs steps — reserve shortfall")
    ax_reserve_t.set_ylim(-0.02, 1.02)
    ax_reserve_t.grid(True, alpha=0.3)
    ax_reserve_t.legend(fontsize=7, loc="best")

    ax_thermal_t.set_xlabel("Env timesteps (millions)")
    ax_thermal_t.set_ylabel("Thermal violation rate")
    ax_thermal_t.set_title("Safety vs steps — thermal violation")
    ax_thermal_t.set_ylim(-0.02, 1.02)
    ax_thermal_t.grid(True, alpha=0.3)
    ax_thermal_t.legend(fontsize=7, loc="best")

    fig.suptitle(
        "Solid = n256 tuning sweep; dotted = 4M baselines. Lower cost is better; lower safety-rate panels mean fewer violations.",
        fontsize=9,
        y=1.02,
    )
    fig.tight_layout()
    out = fig_dir / "n256_epoch_sweep.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[n256_epoch_sweep] saved {out} (+ .png)")
    return out


def _cli() -> None:
    p = argparse.ArgumentParser(description="n256 (4M) PPO / SAC / safeRL tuning sweep")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run", help="Train 4 configs + write n256_epoch_sweep.json")
    sub.add_parser("plot", help="Plot sweep vs 4M ppo_n256_t4m / sac_n256_t4m baselines")
    args = p.parse_args()
    if args.cmd == "run":
        main()
    else:
        plot_sweep(TASK_DIR)


if __name__ == "__main__":
    _cli()
