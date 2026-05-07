#!/usr/bin/env python3
"""TSO resource budget scan: num_envs × total_timesteps vs walltime & learning curves.

Why smaller ``num_envs`` can look better on a *fixed* ``total_timesteps`` budget
---------------------------------------------------------------------------
PPO performs one policy update every ``num_envs * n_steps`` environment steps.
So for the same ``total_timesteps``, **fewer parallel envs ⇒ more optimizer
updates** (more passes over the network), which often improves sample efficiency
on the learning curve, at the cost of lower env-step throughput if the GPU was
already saturated at high ``num_envs``.

SAC is off-policy: ``num_envs`` changes collection rate and replay mixing; it is
not directly comparable to PPO's on-policy update count.

Usage
-----
    # Run all preset experiments (GPU recommended; sequential to avoid thrash)
    python benchmarks/tso/budget_scan.py run

    # 4M-step grid (same n_envs × three algos), eval every 100k
    python benchmarks/tso/budget_scan.py run --timesteps 4000000

    # Plots only; ``--only-t4m`` = curves with tag ``*_t4m`` (omit 2M runs on the same figure)
    python benchmarks/tso/budget_scan.py plot --only-t4m

Outputs: ``results/figures/budget_scan_ppo_sac_lag.png`` (all budget_scan runs), optional
``budget_scan_t4m.png`` with ``--only-t4m``; ``results/budget_scan/registry.json``
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

TASK_DIR = Path(__file__).resolve().parent
FIG_DIR = TASK_DIR / "results" / "figures"
REGISTRY_DIR = TASK_DIR / "results" / "budget_scan"
CONFIG_CACHE = REGISTRY_DIR / "configs"

# (algo, num_envs, total_timesteps). 2M: eval every 50k → 40 eval points.
DEFAULT_EXPERIMENTS: list[tuple[str, int, int]] = [
    ("ppo", 64, 2_000_000),
    ("ppo", 128, 2_000_000),
    ("ppo", 256, 2_000_000),
    ("sac", 64, 2_000_000),
    ("sac", 128, 2_000_000),
    ("sac", 256, 2_000_000),
    ("ppo_lagrangian", 64, 2_000_000),
    ("ppo_lagrangian", 128, 2_000_000),
    ("ppo_lagrangian", 256, 2_000_000),
]


def grid_experiments(total_timesteps: int) -> list[tuple[str, int, int]]:
    """Same 3×3 grid: PPO, SAC, PPO-Lagrangian × n_envs 64,128,256."""
    out: list[tuple[str, int, int]] = []
    for algo in ("ppo", "sac", "ppo_lagrangian"):
        for n in (64, 128, 256):
            out.append((algo, n, int(total_timesteps)))
    return out

_ALGO_KEY = {"ppo_lagrangian": "safe", "ppo": "ppo", "sac": "sac"}


def _tag(algo: str, n: int, t: int) -> str:
    tm = t // 1_000_000
    return f"{algo}_n{n}_t{tm}m"


def build_train_dict(algo: str, num_envs: int, total_timesteps: int) -> dict:
    key = _ALGO_KEY[algo]
    raw = yaml.safe_load((TASK_DIR / "configs" / f"train_{key}.yaml").read_text(encoding="utf-8"))
    d = dict(raw)
    d["num_envs"] = int(num_envs)
    d["total_timesteps"] = int(total_timesteps)
    # ~40 eval checkpoints: 2M/50k, 4M/100k — avoid eval dominating wall time on long runs.
    if total_timesteps <= 2_000_000:
        d["eval_freq"] = 50_000
    else:
        d["eval_freq"] = 100_000
    if algo == "ppo_lagrangian":
        d["n_checkpoints"] = max(2, int(round(200 * total_timesteps / 20_000_000)))
    return d


def run_experiments(
    experiments: list[tuple[str, int, int]] | None = None,
    *,
    seed: int = 0,
    dry_run: bool = False,
    registry_name: str = "registry.json",
) -> list[dict]:
    from benchmarks.tso.train import train_tso

    experiments = experiments or DEFAULT_EXPERIMENTS
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_CACHE.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for algo, n, t in experiments:
        tag = _tag(algo, n, t)
        cfg_path = CONFIG_CACHE / f"{tag}.yaml"
        cfg_dict = build_train_dict(algo, n, t)
        if not dry_run:
            cfg_path.write_text(
                yaml.safe_dump(cfg_dict, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
        print(f"[budget_scan] {tag} -> {cfg_path.name} (dry_run={dry_run})")
        if dry_run:
            rows.append({"tag": tag, "status": "dry_run"})
            continue
        rec = train_tso(
            TASK_DIR,
            algo=algo,
            seed=seed,
            config_path=str(cfg_path),
            extra_notes=f"budget_scan={tag}",
        )
        rows.append(
            {
                "tag": tag,
                "algo": algo,
                "num_envs": n,
                "total_timesteps": t,
                "run_id": rec.run_id,
                "status": rec.status,
                "walltime_s": rec.walltime_s,
                "throughput_sps": rec.throughput_sps,
            }
        )
        (REGISTRY_DIR / registry_name).write_text(
            json.dumps(rows, indent=2), encoding="utf-8"
        )
    return rows


def _scan_manifest_for_budget_runs(task_dir: Path) -> list[dict]:
    manifest = task_dir / "results" / "manifest.json"
    if not manifest.exists():
        return []
    data = json.loads(manifest.read_text(encoding="utf-8"))
    by_tag: dict[str, dict] = {}
    for r in data:
        if r.get("task") != "tso" or r.get("split") != "train":
            continue
        notes = r.get("notes") or ""
        if "budget_scan=" not in notes:
            continue
        tag = notes.split("budget_scan=", 1)[-1].split("|", 1)[0].strip()
        ts = str(r.get("timestamp") or "")
        rec = {
            "run_id": r.get("run_id"),
            "tag": tag,
            "algo": r.get("algo"),
            "walltime_s": r.get("walltime_s"),
            "throughput_sps": r.get("throughput_sps"),
            "metrics": r.get("metrics") or {},
            "timestamp": ts,
        }
        prev = by_tag.get(tag)
        if prev is None or ts > str(prev.get("timestamp") or ""):
            by_tag[tag] = rec
    return list(by_tag.values())


def _load_eval_cost_and_axes(
    task_dir: Path, run_id: str, reward_scale: float
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """eval cost, timesteps, walltime per eval checkpoint.

    ``eval_wall_time_s`` from Rejax is **already** host elapsed seconds since the
    wall-clock origin (``perf_counter - t0``) at each eval — do **not** cumsum.
    """
    adir = task_dir / "results" / "artifacts"
    def _npy(stem: str) -> np.ndarray | None:
        p = adir / f"{run_id}_{stem}.npy"
        if not p.exists():
            return None
        return np.load(p)

    ec = _npy("eval_total_operating_cost")
    if ec is None:
        er = _npy("eval_returns")
        if er is not None:
            ec = -np.asarray(er, dtype=np.float64) / reward_scale
    if ec is None:
        return np.array([]), None, None
    ts = _npy("timesteps")
    wt = _npy("eval_wall_time_s")
    if wt is not None and wt.size:
        wall = np.asarray(wt, dtype=np.float64).ravel()
    else:
        wall = None
    return np.asarray(ec, dtype=np.float64).ravel(), ts, wall


def plot_budget_scan(
    task_dir: Path = TASK_DIR,
    *,
    tag_substr: str | None = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from benchmarks.common.configs import load_task_config
    tcfg = load_task_config(task_dir)
    reward_scale = float(tcfg.get("reward_scale", 1e-4))

    runs = _scan_manifest_for_budget_runs(task_dir)
    if tag_substr:
        runs = [r for r in runs if tag_substr in (r.get("tag") or "")]
    if not runs:
        print(
            "[budget_scan] No runs with budget_scan= in manifest"
            + (f" and tag containing {tag_substr!r}" if tag_substr else "")
            + "; run `budget_scan.py run` first."
        )
        return

    # Style: color by algo, linestyle by num_envs (extract from tag)
    algo_colors = {
        "ppo": "#2196f3",
        "sac": "#00acc1",
        "ppo_lagrangian": "#e91e63",
    }
    n_linestyles = {64: "-", 128: "--", 256: "-."}

    def parse_tag(tag: str) -> tuple[str, int]:
        m = re.search(r"^(?P<algo>.+)_n(?P<n>\d+)_t(?P<tm>\d+)m$", tag)
        if m:
            return m.group("algo"), int(m.group("n"))
        return "unknown", 64

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_t, ax_w = axes

    summary_rows: list[dict] = []
    for r in sorted(runs, key=lambda x: x.get("tag", "")):
        rid = r.get("run_id")
        tag = r.get("tag", "")
        if not rid:
            continue
        cost, tsv, wall = _load_eval_cost_and_axes(task_dir, str(rid), reward_scale)
        if cost.size == 0:
            print(f"[budget_scan] skip {rid}: no eval cost")
            continue
        algo, n_env = parse_tag(tag)
        color = algo_colors.get(algo, "#666666")
        ls = n_linestyles.get(n_env, "-")
        label = f"{tag.replace('_', ' ')}  ({(r.get('walltime_s') or 0) / 60:.1f} min)"

        if tsv is not None and tsv.size == cost.size:
            x_t = tsv / 1e6
        else:
            tmax = float((r.get("metrics") or {}).get("total_timesteps") or 2_000_000.0)
            x_t = np.linspace(0.0, tmax / 1e6, cost.size)
        ax_t.plot(x_t, cost / 1e6, color=color, linestyle=ls, linewidth=1.8, label=label, alpha=0.9)

        if wall is not None and wall.size == cost.size:
            ax_w.plot(
                wall / 60.0, cost / 1e6, color=color, linestyle=ls, linewidth=1.8, label=label, alpha=0.9
            )
        else:
            wtot = r.get("walltime_s")
            if wtot and tsv is not None and tsv.size and float(tsv.max()) > 0:
                x_w = tsv / float(tsv.max()) * (float(wtot) / 60.0)
                ax_w.plot(
                    x_w, cost / 1e6, color=color, linestyle=ls, linewidth=1.8, label=label, alpha=0.9
                )

        summary_rows.append(
            {
                "tag": tag,
                "algo": algo,
                "num_envs": n_env,
                "run_id": rid,
                "walltime_s": r.get("walltime_s"),
                "final_eval_cost_mgbp": float(cost[-1] / 1e6) if cost.size else None,
            }
        )

    if tag_substr == "_t4m":
        t_label = "4M total"
    elif tag_substr is None:
        t_label = "all budget_scan runs"
    else:
        t_label = str(tag_substr).strip("_")
    ax_t.set_xlabel("Env timesteps (millions)")
    ax_t.set_ylabel("Eval operating cost (million GBP)")
    ax_t.set_title(f"TSO budget scan — eval cost vs environment steps ({t_label})")
    ax_t.grid(True, alpha=0.3)
    ax_t.legend(fontsize=6, loc="best")

    ax_w.set_xlabel("Cumulative training wall time (minutes)")
    ax_w.set_ylabel("Eval operating cost (million GBP)")
    ax_w.set_title(f"TSO budget scan — eval cost vs wall time ({t_label})")
    ax_w.grid(True, alpha=0.3)
    if ax_w.get_legend_handles_labels()[0]:
        ax_w.legend(fontsize=6, loc="best")

    fig.suptitle(
        "Solid/– /-. = n_envs 64 / 128 / 256; colors: PPO / SAC / PPO-Lag",
        fontsize=9,
        y=1.02,
    )
    fig.tight_layout()
    stem = "budget_scan_t4m" if tag_substr == "_t4m" else "budget_scan_ppo_sac_lag"
    p_pdf = FIG_DIR / f"{stem}.pdf"
    fig.savefig(p_pdf, bbox_inches="tight")
    fig.savefig(str(p_pdf).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[budget_scan] saved {p_pdf} (+ .png)")

    if summary_rows:
        st_name = "summary_table_4m.json" if tag_substr == "_t4m" else "summary_table.json"
        (REGISTRY_DIR / st_name).write_text(
            json.dumps(summary_rows, indent=2), encoding="utf-8"
        )
        _print_recommendation(summary_rows, filter_note=tag_substr or "all")


def _print_recommendation(rows: list[dict], *, filter_note: str = "all") -> None:
    """Heuristic text summary (single-seed; use for development defaults only)."""
    by_algo: dict[str, list[dict]] = {}
    for r in rows:
        a = r.get("algo") or "unknown"
        by_algo.setdefault(a, []).append(r)
    print(f"\n--- budget_scan recommendation (runs: {filter_note}, seed=0) ---")
    print("PPO: total_updates ≈ total_timesteps / (n_envs * n_steps).")
    print("Larger n_envs ⇒ fewer PPO/SAC–style updates for the same env budget (often lower")
    print("per-step sample quality on the learning curve); higher env throughput (shorter walltime).")
    for a, item in sorted(by_algo.items()):
        best = min(
            item,
            key=lambda x: (x.get("final_eval_cost_mgbp") is None, x.get("final_eval_cost_mgbp") or 1e9),
        )
        c = best.get("final_eval_cost_mgbp")
        w = best.get("walltime_s")
        t = best.get("tag", "")
        wmin = w / 60.0 if isinstance(w, (int, float)) else None
        if c is not None and wmin is not None:
            print(f"  {a}: best final eval cost in this grid → {t}  (≈{c:.2f} MGBP)  [wall {wmin:.1f} min]")
    print("Fast dev (GPU) starting point: try total_timesteps=2e6, eval_freq=5e4, and n_envs=64 for")
    print("PPO/SAC; scale n_envs up (128–256) only when prioritizing walltime over sample efficiency.")
    print("PPO-Lagrangian: n_checkpoints scales with total_timesteps in the generated YAMLs.")
    print("------------------------------------------------------------------\n")


def main() -> None:
    p = argparse.ArgumentParser(description="TSO num_envs × total_timesteps budget scan")
    sub = p.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="Run 3×3 grid (PPO+SAC+PPO-Lag × n=64,128,256)")
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument("--dry-run", action="store_true")
    p_run.add_argument(
        "--timesteps",
        type=int,
        default=2_000_000,
        help="Total environment steps per run (e.g. 2000000 or 4000000).",
    )
    p_plot = sub.add_parser("plot", help="Plot from manifest (budget_scan= notes)")
    p_plot.add_argument(
        "--only-t4m",
        action="store_true",
        help="Only series whose tag contains ``_t4m`` (4M-budget runs).",
    )
    args = p.parse_args()
    if args.cmd == "run":
        ts = int(args.timesteps)
        ex = grid_experiments(ts)
        reg = "registry.json" if ts == 2_000_000 else f"registry_t{ts // 1_000_000}m.json"
        run_experiments(ex, seed=args.seed, dry_run=args.dry_run, registry_name=reg)
    else:
        if args.only_t4m:
            plot_budget_scan(TASK_DIR, tag_substr="_t4m")
        else:
            plot_budget_scan(TASK_DIR, tag_substr=None)


if __name__ == "__main__":
    main()
