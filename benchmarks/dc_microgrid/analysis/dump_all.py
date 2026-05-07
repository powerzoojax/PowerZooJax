#!/usr/bin/env python
"""Batch trajectory dumper for the DC Microgrid benchmark.

Runs ``dump_trajectory`` for the standard set used by ``plots_extra``:
  4 algos x 8 splits x seed 0 x 3 episodes = 96 trajectory files.

Picks the latest PPO train run-id from the manifest automatically.
Reuses the same Python process so JAX JIT compiles only once per algo.

Usage:
    python benchmarks/dc_microgrid/dump_all.py
    python benchmarks/dc_microgrid/dump_all.py --episodes 5
    python benchmarks/dc_microgrid/dump_all.py --splits iid,cooling_stress
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

TASK_DIR = Path(__file__).resolve().parent

ALGOS = ["no_control", "max_renewable", "rule_based", "ppo"]
ALL_SPLITS = [
    "train", "iid", "cooling_stress", "renewable_drought",
    "workload_swap", "workload_shock", "dg_derating", "sla_tighten",
]


def _find_ppo_run_ids(seeds: list[int]) -> dict[int, str]:
    mp = TASK_DIR / "results" / "manifest.json"
    if not mp.exists():
        return {}
    data = json.loads(mp.read_text(encoding="utf-8"))
    mapping: dict[int, tuple[str, str]] = {}
    for r in data:
        if (
            r.get("task") == "dc_microgrid"
            and r.get("algo") == "ppo"
            and r.get("split") == "train"
            and r.get("status") == "completed"
            and "eval of " not in (r.get("notes") or "")
        ):
            seed = int(r["seed"])
            ts = r.get("timestamp", "")
            if seed not in mapping or ts > mapping[seed][1]:
                mapping[seed] = (r["run_id"], ts)
    return {s: v[0] for s, v in mapping.items() if s in seeds}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algos", default=",".join(ALGOS))
    parser.add_argument("--splits", default=",".join(ALL_SPLITS))
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",")]

    ppo_ids = _find_ppo_run_ids(seeds) if "ppo" in algos else {}
    if "ppo" in algos and not ppo_ids:
        print(f"[dump_all] WARNING: no PPO train records found for seeds={seeds}")

    from benchmarks.dc_microgrid.dump_trajectory import dump_trajectory

    t_start = time.time()
    n_done = 0
    n_skipped = 0
    n_total = len(algos) * len(splits) * len(seeds)
    for algo in algos:
        for seed in seeds:
            run_id = ppo_ids.get(seed) if algo == "ppo" else None
            if algo == "ppo" and run_id is None:
                print(f"[dump_all] skip ppo seed={seed} (no run-id)")
                n_skipped += len(splits)
                continue
            for split in splits:
                try:
                    dump_trajectory(
                        algo=algo,
                        split=split,
                        seed=seed,
                        episodes=args.episodes,
                        run_id=run_id,
                    )
                    n_done += 1
                except Exception as exc:
                    print(f"[dump_all] FAIL {algo} {split} seed={seed}: {exc}")
                    n_skipped += 1
    elapsed = time.time() - t_start
    print(
        f"\n[dump_all] done: {n_done}/{n_total} succeeded, "
        f"{n_skipped} skipped, walltime={elapsed:.0f}s"
    )


if __name__ == "__main__":
    main()
