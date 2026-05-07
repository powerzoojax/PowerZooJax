#!/usr/bin/env python
"""Dump per-step trajectory data for a (policy, split, seed) combination.

Loads either a trained PPO policy (.pkl artifact from train.py) or a
non-learning baseline (no_control / max_renewable / rule_based), rolls out
N episodes on the requested split, and saves per-step arrays to
``results/trajectories/<algo>_<split>_s<seed>.npz``.

Saved arrays (each shape ``(n_episodes, T)`` with T=288):
  step_idx      : 0..T-1 indices
  reward        : shaped reward (after RewardShapingWrapper)
  raw_reward    : original env reward (info['raw_reward'])
  shaping_pen   : shaping penalty subtracted (info['shaping_penalty'])
  p_dc_mw       : DC total power demand
  p_pv_mw       : PV generation
  p_dg_mw       : DG output
  p_batt_mw     : battery power (positive = discharge)
  soc           : battery SOC fraction
  fuel_cost     : per-step fuel cost ($)
  carbon_kg     : per-step CO2 emission (kg)
  cost_sla      : SLA violation cost
  cost_overtemp : overtemperature cost
  cost_power_deficit : normalised power deficit
  cost          : total CMDP cost
  action_*      : 5 action channels (train_sched, ft_sched, cool, batt, dg)

Usage:
    python benchmarks/dc_microgrid/dump_trajectory.py \\
        --algo ppo --run-id <train_run_id> --split iid --episodes 3
    python benchmarks/dc_microgrid/dump_trajectory.py \\
        --algo rule_based --split iid --seed 0 --episodes 3
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

TASK_DIR = Path(__file__).resolve().parents[1]
TRAJ_DIR = TASK_DIR / "results" / "trajectories"

PER_STEP_KEYS = (
    "p_dc_mw", "p_pv_mw", "p_dg_mw", "p_batt_mw", "soc",
    "fuel_cost", "carbon_kg",
    "cost_sla", "cost_overtemp", "cost_power_deficit", "cost",
    "raw_reward", "shaping_penalty",
)


def _build_eval_params(task_config: dict, eval_config: dict, start_step: int):
    """Build env params from real profiles, with optional OOD transform."""
    max_steps = task_config.get("max_steps", 288)
    data_source = eval_config.get("data_source", task_config.get("data_source", "google"))
    ood_scenario = eval_config.get("ood_scenario")

    from powerzoojax.envs.microgrid import make_dcmicrogrid_params_with_profiles
    try:
        params = make_dcmicrogrid_params_with_profiles(
            source=data_source,
            episode_start_step=start_step,
            max_steps=max_steps,
        )
    except Exception as exc:
        raise RuntimeError(
            f"DC Microgrid dump_trajectory requires real {data_source!r} "
            f"profiles: {exc}"
        ) from exc
    data_mode = "real"

    if ood_scenario is not None:
        from powerzoojax.data.dc_microgrid_profiles import apply_ood_transform
        try:
            params = apply_ood_transform(params, ood_scenario)
        except Exception as exc:
            raise RuntimeError(
                f"DC Microgrid OOD transform {ood_scenario!r} failed: {exc}"
            ) from exc

    return params, data_mode


def _load_ppo_policy(run_id: str, train_cfg, env, params):
    """Reload the trained PPO policy from a saved params .pkl.

    Uses ``make_policy_fn`` from ``benchmarks.common.runtime`` so the
    Beta-distribution actor used by the canonical config is honoured
    (stock ``rejax.PPO.make_act`` would assume a Gaussian actor).
    """
    artifacts_path = TASK_DIR / "results" / "artifacts" / f"{run_id}_params.pkl"
    if not artifacts_path.exists():
        from benchmarks.common.io import load_run
        rec = load_run(run_id, TASK_DIR)
        artifacts_path = TASK_DIR / "results" / rec.artifacts.get("params", "")
    if not artifacts_path.exists():
        raise FileNotFoundError(f"PPO params artifact not found for {run_id}")

    from benchmarks.common.io import load_pickle
    from benchmarks.common.runtime import make_policy_fn

    train_state = load_pickle(artifacts_path)
    act = make_policy_fn("ppo", train_state, env, params, train_cfg)
    # `make_policy_fn` returns a (obs, state, key) callable; we expose the
    # historical (obs, key) shape used by the dump loop below.
    return lambda obs, key: act(obs, None, key)


def _load_sac_policy(run_id: str, train_cfg, env, params):
    """Reload the SAC policy from a saved Orbax checkpoint via make_policy_fn."""
    orbax_path = TASK_DIR / "results" / "artifacts" / f"{run_id}_params_orbax"
    if not orbax_path.exists():
        from benchmarks.common.io import load_run
        rec = load_run(run_id, TASK_DIR)
        rel = rec.artifacts.get("params_orbax", "")
        orbax_path = TASK_DIR / "results" / rel
    if not orbax_path.exists():
        raise FileNotFoundError(f"SAC Orbax checkpoint not found for {run_id}")

    from benchmarks.dc_microgrid.rejax_ckpt import load_sac_train_state
    from benchmarks.common.runtime import make_policy_fn
    train_state = load_sac_train_state(orbax_path, train_cfg, env, params)
    act = make_policy_fn("sac", train_state, env, params, train_cfg)
    return lambda obs, key: act(obs, None, key)


def _baseline_action_fn(algo: str):
    """Return a deterministic step-indexed action function for baselines."""
    from benchmarks.dc_microgrid.baselines import (
        _no_control_action,
        _max_renewable_action,
        _rule_based_action,
    )
    if algo == "no_control":
        return lambda step, obs, prev_deficit: _no_control_action(step, obs)
    if algo == "max_renewable":
        return lambda step, obs, prev_deficit: _max_renewable_action(step, obs)
    if algo == "rule_based":
        return lambda step, obs, prev_deficit: _rule_based_action(step, obs, prev_deficit)
    raise ValueError(f"Unknown baseline algo: {algo!r}")


def dump_trajectory(
    algo: str,
    split: str,
    seed: int,
    episodes: int,
    run_id: str | None = None,
    out_path: Path | None = None,
) -> Path:
    from powerzoojax.envs.microgrid import DataCenterMicrogridEnv
    from benchmarks.dc_microgrid._reward_shaping import wrap_with_shaping

    from benchmarks.common.configs import load_config, load_task_config

    task_config = load_task_config(TASK_DIR)
    eval_cfg_path = TASK_DIR / "configs" / f"eval_{split}.yaml"
    eval_config: dict = load_config(eval_cfg_path) if eval_cfg_path.exists() else {}
    eval_config.setdefault("split", split)

    base_env = DataCenterMicrogridEnv()
    env = wrap_with_shaping(base_env, task_config)

    max_steps = task_config.get("max_steps", 288)

    # Spread episodes uniformly across the year-long profile.
    total_profile_steps = 365 * max_steps
    starts = (
        np.linspace(0, max(0, total_profile_steps - max_steps), episodes).astype(int)
        if episodes > 1
        else np.zeros(1, dtype=int)
    )

    # Policy
    if algo in ("ppo", "sac"):
        if run_id is None:
            raise ValueError(f"--run-id required for algo={algo}")
        from benchmarks.common.runtime import build_train_cfg

        train_cfg_dict = load_config(TASK_DIR / "configs" / f"train_{algo}.yaml")
        train_cfg = build_train_cfg(train_cfg_dict, algo=algo)
        first_params, _ = _build_eval_params(task_config, eval_config, int(starts[0]))
        if algo == "ppo":
            policy_fn = _load_ppo_policy(run_id, train_cfg, env, first_params)
        else:
            policy_fn = _load_sac_policy(run_id, train_cfg, env, first_params)
    else:
        policy_fn = _baseline_action_fn(algo)

    cols: dict[str, list[list[float]]] = {k: [] for k in PER_STEP_KEYS}
    cols["reward"] = []
    cols["action"] = []  # nested (n_episodes, T, 5)

    t0 = time.time()
    data_modes = []
    for ep in range(episodes):
        params, data_mode = _build_eval_params(task_config, eval_config, int(starts[ep]))
        data_modes.append(data_mode)
        key = jax.random.PRNGKey(seed * 10_000 + ep)
        obs, state = env.reset(key, params)

        ep_rows: dict[str, list[float]] = {k: [] for k in PER_STEP_KEYS}
        ep_rows["reward"] = []
        ep_rows["action"] = []
        prev_deficit = 0.0

        for step in range(max_steps):
            key, k_step, k_pol = jax.random.split(key, 3)
            if algo in ("ppo", "sac"):
                action_jax = policy_fn(obs, k_pol)
                action_np = np.asarray(action_jax).reshape(5)
            else:
                action_np = policy_fn(step, np.asarray(obs), prev_deficit).astype(np.float32)
            action = jnp.array(action_np)
            obs, state, reward, _costs, done, info = env.step(k_step, state, action, params)

            ep_rows["reward"].append(float(np.asarray(reward)))
            ep_rows["action"].append(action_np.tolist())
            for k in PER_STEP_KEYS:
                v = info.get(k, 0.0)
                ep_rows[k].append(float(np.asarray(v)))
            prev_deficit = ep_rows.get("cost_power_deficit", [0.0])[-1]

        for k, v in ep_rows.items():
            cols[k].append(v)

    # Convert to arrays
    arrays = {}
    for k, v in cols.items():
        arr = np.asarray(v, dtype=np.float32)
        arrays[k] = arr
    arrays["step_idx"] = np.tile(np.arange(max_steps, dtype=np.int32), (episodes, 1))
    arrays["start_step"] = np.asarray(starts, dtype=np.int32)

    walltime = time.time() - t0
    suffix = f"_{run_id}" if (algo in ("ppo", "sac") and run_id is not None) else ""
    out_path = out_path or TRAJ_DIR / f"{algo}_{split}_s{seed}{suffix}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    print(
        f"[dump] {algo} split={split} seed={seed} episodes={episodes} "
        f"data_mode={data_modes[0]} walltime={walltime:.1f}s -> {out_path}"
    )
    return out_path


def main():
    p = argparse.ArgumentParser(description="Dump per-step DC Microgrid trajectory data")
    p.add_argument("--algo", required=True,
                   choices=["no_control", "max_renewable", "rule_based", "ppo", "sac"])
    p.add_argument("--split", required=True,
                   choices=[
                       "train", "iid", "cooling_stress", "renewable_drought",
                       "workload_swap", "workload_shock", "dg_derating", "sla_tighten",
                   ])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--run-id", default=None,
                   help="Required for --algo ppo: training run_id whose .pkl to load")
    p.add_argument("--out", default=None, help="Override output path")
    args = p.parse_args()

    out_path = Path(args.out) if args.out else None
    dump_trajectory(
        algo=args.algo,
        split=args.split,
        seed=args.seed,
        episodes=args.episodes,
        run_id=args.run_id,
        out_path=out_path,
    )


if __name__ == "__main__":
    main()
