"""One-shot trajectory dump for the hand-crafted rule-based policy.

The shipped ``dump_trajectory.py`` baseline path expects per-step action
functions that no longer exist in ``benchmarks.dc_microgrid.baselines``.
This script bypasses that path: it builds the env exactly as
``baselines.run_single_baseline`` does, calls
``rollout_dcmicrogrid_rule_based`` to get the per-step info dicts, then
saves the relevant fields to an NPZ file shaped the same way as the PPO
trajectory dump.

Run:
    python benchmarks/dc_microgrid/analysis/dump_rule_based_trajectory.py \\
        --split iid --episodes 5 --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

import jax  # noqa: E402

from benchmarks.common.configs import load_config, load_task_config  # noqa: E402
from benchmarks.dc_microgrid._reward_shaping import wrap_with_shaping  # noqa: E402
from powerzoojax.tasks.dc_microgrid import (  # noqa: E402
    DCMicrogridTask,
    rollout_dcmicrogrid_rule_based,
)


TASK_DIR = _PROJECT_ROOT / "benchmarks" / "dc_microgrid"
TRAJ_DIR = TASK_DIR / "results" / "trajectories"

PER_STEP_KEYS = (
    "p_dc_mw", "p_pv_mw", "p_dg_mw", "p_batt_mw", "soc",
    "fuel_cost", "carbon_kg",
    "cost_sla", "cost_overtemp", "cost_power_deficit", "cost",
    "raw_reward", "shaping_penalty",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="iid")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    task_config = load_task_config(TASK_DIR)
    eval_cfg_path = TASK_DIR / "configs" / f"eval_{args.split}.yaml"
    eval_config = load_config(eval_cfg_path) if eval_cfg_path.exists() else {}
    max_steps = task_config.get("max_steps", 288)

    task = DCMicrogridTask(
        source=task_config.get("data_source", "google"),
        max_steps=max_steps,
        case_overrides=task_config.get("case_overrides") or {},
    )
    env = wrap_with_shaping(task.make_env(args.split), task_config)

    cols: dict[str, list[list[float]]] = {k: [] for k in PER_STEP_KEYS}
    cols["reward"] = []
    cols["action"] = []  # (n_episodes, T, 5)
    starts: list[int] = []

    for ep in range(args.episodes):
        params = task.episode_params(
            args.split,
            episode_idx=ep,
            n_episodes=args.episodes,
            max_steps=max_steps,
            strategy="uniform",
            seed=args.seed,
        )
        key = jax.random.PRNGKey(args.seed * 10_000 + ep)
        info_history = rollout_dcmicrogrid_rule_based(env, params, key)
        T = len(info_history)
        for k in PER_STEP_KEYS:
            cols[k].append([float(info.get(k, 0.0)) for info in info_history])
        cols["reward"].append([float(info.get("reward", 0.0)) for info in info_history])
        cols["action"].append([list(info.get("action", [0.0] * 5)) for info in info_history])
        starts.append(int(getattr(params, "episode_start_step", 0)))
        print(f"  ep{ep} steps={T}")

    arrays = {k: np.asarray(v, dtype=np.float32) for k, v in cols.items() if k != "action"}
    arrays["action"] = np.asarray(cols["action"], dtype=np.float32)
    arrays["start_step"] = np.asarray(starts, dtype=np.int32)
    arrays["step_idx"] = np.broadcast_to(
        np.arange(arrays["soc"].shape[1], dtype=np.int32),
        arrays["soc"].shape,
    ).copy()

    TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    out = TRAJ_DIR / f"rule_based_{args.split}_s{args.seed}.npz"
    np.savez(out, **arrays)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
