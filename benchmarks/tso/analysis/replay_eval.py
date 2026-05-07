"""Replay-eval per-step time-series collector (no retraining).

Loads each finished training run's saved policy params, replays **one
representative episode per split** (the first eval window, deterministic
seed), and dumps the full per-step trajectory needed for plots that
``eval.py`` aggregates away.

Outputs (per (algo, split, seed))::

    benchmarks/tso/results/replay/<algo>_s<seed>_<split>.npz

with arrays:
    - gen_cost, startup_cost, no_load_cost                     (T,)
    - reserve_shortfall, cost_thermal_overload, cost           (T,)
    - commitment_switches, is_safe                             (T,)
    - reward, done                                             (T,)
    - action                                                   (T, 2*n_units)
    - unit_status, unit_power_mw                               (T, n_units)
    - line_flow_mw                                             (T, n_lines)
    - meta keys: algo, split, seed, run_id, episode_start, max_steps

Idempotent: re-running overwrites the same npz files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.common.io import load_manifest, load_pickle  # noqa: E402

TASK_DIR = Path(__file__).resolve().parent
RUNS_DIR = TASK_DIR / "results" / "runs"
ARTIFACTS_DIR = TASK_DIR / "results" / "artifacts"
REPLAY_DIR = TASK_DIR / "results" / "replay"

ALL_SPLITS = ["train", "iid", "load_stress", "line_tightening"]
RL_ALGOS = ["ppo", "ppo_lagrangian", "saute_ppo"]


# ────────────────────────────────────────────────────────────────────────────
# Find latest successful train run per (algo, seed)
# ────────────────────────────────────────────────────────────────────────────

def _latest_train_runs() -> dict[tuple[str, int], dict]:
    """Return {(algo, seed): record_dict} for the latest train record per pair
    that has status=completed and a saved params artifact.

    Scans ``runs/*.json`` directly because the deduplicated manifest collapses
    the actual training record (with ``artifacts.params``) under the eval-on-
    train record (without params).  The params-bearing JSONs are still on disk.
    """
    by_pair: dict[tuple[str, int], dict] = {}
    for f in sorted(RUNS_DIR.glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if d.get("split") != "train":
            continue
        if d.get("algo") not in RL_ALGOS:
            continue
        if d.get("status") != "completed":
            continue
        params_rel = (d.get("artifacts") or {}).get("params")
        if not params_rel:
            continue
        params_path = TASK_DIR / "results" / params_rel
        if not params_path.exists():
            continue
        key = (d["algo"], int(d["seed"]))
        prev = by_pair.get(key)
        if prev is None or d.get("timestamp", "") > prev["timestamp"]:
            by_pair[key] = {
                "algo": d["algo"],
                "seed": int(d["seed"]),
                "run_id": d["run_id"],
                "timestamp": d.get("timestamp", ""),
                "params_path": str(params_path),
            }
    return by_pair


# ────────────────────────────────────────────────────────────────────────────
# One-episode replay
# ────────────────────────────────────────────────────────────────────────────

def _replay_one(
    algo: str,
    seed: int,
    run_id: str,
    params_path: str,
    split: str,
    n_episodes: int = 1,
) -> Path | None:
    """Load policy and run ``n_episodes`` deterministic episodes; dump 1 npz.

    With ``n_episodes=1`` arrays have shape ``(T, ...)`` (back-compat).
    With ``n_episodes>1`` arrays gain a leading episode axis: ``(E, T, ...)``.
    Episode windows are picked the same way as ``eval.py`` (linspace over the
    full GB split window) so the replay is faithful to the manifest run.
    """
    # Local imports keep top-level argparse fast.
    import dataclasses
    from powerzoojax.envs.grid.unit_commitment import UnitCommitmentEnv

    from benchmarks.common.configs import (
        load_config,
        load_task_config_for_run,
        load_train_config_for_run,
    )
    from benchmarks.common.io import load_run
    from benchmarks.common.runtime import build_train_cfg
    from benchmarks.common.runtime import make_policy_fn
    from benchmarks.tso.config_runtime import get_eval_episodes, make_task_from_config
    from powerzoojax.rl.wrappers import SauteWrapper

    eval_config = load_config(TASK_DIR / "configs" / f"eval_{split}.yaml")
    original = load_run(run_id, TASK_DIR)
    task_config = load_task_config_for_run(TASK_DIR, original)
    max_steps = int(task_config.get("max_steps", 48))
    n_eval_episodes = get_eval_episodes(task_config, eval_config)

    task = make_task_from_config(
        task_config,
        load_scale=float(eval_config.get("load_scale", 1.0)),
        line_rating_scale=float(eval_config.get("line_rating_scale", 1.0)),
    )

    algo_key = {"ppo_lagrangian": "safe", "saute_ppo": "saute_ppo"}.get(algo, algo)
    train_config_dict = load_train_config_for_run(
        TASK_DIR,
        original,
        algo_key_map={"ppo_lagrangian": "safe", "saute_ppo": "saute_ppo"},
        default_key=algo_key,
    )
    train_cfg = build_train_cfg(train_config_dict, algo=algo)

    train_state = load_pickle(params_path)
    starts = np.arange(min(int(n_episodes), int(n_eval_episodes)), dtype=np.int32)
    if starts.size == 0:
        starts = np.zeros(1, dtype=np.int32)

    env = UnitCommitmentEnv()
    base_params = task.episode_params(
        split, 0, n_eval_episodes, max_steps, strategy="uniform", seed=seed
    )
    policy_fn = make_policy_fn(
        algo,
        train_state,
        env,
        base_params,
        train_cfg,
        action_dim=2 * base_params.case.n_units,
        selected_names=task.constraint_spec().selected_names,
    )

    n_units = int(base_params.case.n_units)
    n_lines = int(base_params.case.line_rate_a.shape[0])
    info_keys = (
        "gen_cost", "startup_cost", "no_load_cost",
        "reserve_shortfall", "cost_thermal_overload", "cost_sum",
        "commitment_switches", "is_safe",
        "opf_converged", "opf_iterations",
        "opf_box_residual_mw", "opf_line_residual_mw", "opf_balance_residual_mw",
    )

    # Episode-axis buffers, shape (E, T, ...)
    E = len(starts)
    T = max_steps
    info_buf = {k: np.zeros((E, T), dtype=np.float32) for k in info_keys}
    reward_buf = np.zeros((E, T), dtype=np.float32)
    done_buf = np.zeros((E, T), dtype=np.float32)
    action_buf = np.zeros((E, T, 2 * n_units), dtype=np.float32)
    unit_status_buf = np.zeros((E, T, n_units), dtype=np.float32)
    unit_power_buf = np.zeros((E, T, n_units), dtype=np.float32)
    line_flow_buf = np.zeros((E, T, n_lines), dtype=np.float32)

    for ep, episode_start in enumerate(starts):
        ep_params = task.episode_params(
            split,
            int(episode_start),
            n_eval_episodes,
            max_steps,
            strategy="uniform",
            seed=seed,
        )
        key = jax.random.PRNGKey(int(seed) * 10_000 + ep)
        if algo == "saute_ppo":
            saute_eval_kwargs = {
                "selected_names": task.constraint_spec().selected_names,
                "horizon": int(train_cfg.saute_horizon or ep_params.max_steps),
                "unsafe_reward": float(train_cfg.saute_unsafe_reward),
                "use_reward_shaping": False,
            }
            if train_cfg.cost_thresholds:
                wrapper = SauteWrapper(
                    env,
                    ep_params,
                    cost_thresholds=tuple(float(x) for x in train_cfg.cost_thresholds),
                    **saute_eval_kwargs,
                )
            else:
                wrapper = SauteWrapper(
                    env,
                    ep_params,
                    cost_threshold=float(train_cfg.cost_threshold),
                    **saute_eval_kwargs,
                )
            obs, state = wrapper.reset(key)
        else:
            wrapper = None
            obs, state = env.reset(key, ep_params)
        for t in range(T):
            key, k_step, k_pol = jax.random.split(key, 3)
            action = policy_fn(obs, state, k_pol)
            action_buf[ep, t] = np.asarray(action, dtype=np.float32)
            if wrapper is not None:
                obs, state, reward, done, info = wrapper.step(k_step, state, action)
                inner_state = state.env_state
            else:
                obs, state, reward, costs, done, info = env.step(
                    k_step, state, action, ep_params
                )
                inner_state = state
            for k in info_keys:
                info_buf[k][ep, t] = float(info[k])
            reward_buf[ep, t] = float(reward)
            done_buf[ep, t] = float(done)
            unit_status_buf[ep, t] = np.asarray(inner_state.unit_status, dtype=np.float32)
            unit_power_buf[ep, t] = np.asarray(inner_state.unit_power_mw, dtype=np.float32)
            line_flow_buf[ep, t] = np.asarray(inner_state.line_flow_mw, dtype=np.float32)

    # If we only ran 1 episode, drop the leading axis for backward compatibility.
    def _maybe_squeeze(x):
        return x[0] if E == 1 else x

    info_to_save = {k: _maybe_squeeze(v) for k, v in info_buf.items()}

    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    out = REPLAY_DIR / f"{algo}_s{seed}_{split}.npz"
    np.savez_compressed(
        out,
        **info_to_save,
        reward=_maybe_squeeze(reward_buf),
        done=_maybe_squeeze(done_buf),
        action=_maybe_squeeze(action_buf),
        unit_status=_maybe_squeeze(unit_status_buf),
        unit_power_mw=_maybe_squeeze(unit_power_buf),
        line_flow_mw=_maybe_squeeze(line_flow_buf),
        # meta
        algo=np.array(algo),
        split=np.array(split),
        seed=np.array(seed, dtype=np.int32),
        run_id=np.array(run_id),
        episode_indices=np.asarray(starts, dtype=np.int32),
        n_episodes=np.array(E, dtype=np.int32),
        max_steps=np.array(T, dtype=np.int32),
        n_units=np.array(n_units, dtype=np.int32),
        n_lines=np.array(n_lines, dtype=np.int32),
    )
    print(f"  saved {out.name} (E={E}, T={T})")
    return out


# ────────────────────────────────────────────────────────────────────────────
# Driver
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algos", default=",".join(RL_ALGOS),
        help=f"comma-separated, default {','.join(RL_ALGOS)}",
    )
    parser.add_argument(
        "--splits", default=",".join(ALL_SPLITS),
        help=f"comma-separated, default {','.join(ALL_SPLITS)}",
    )
    parser.add_argument("--seeds", default="all", help="'all' or e.g. 0,1,2")
    parser.add_argument(
        "--n-episodes", type=int, default=1,
        help="Episodes per (algo, split, seed) replay; default 1. "
             "Use --n-episodes 50 to match eval.py's window count and dump "
             "the full distribution (~50x file size, still tiny).",
    )
    args = parser.parse_args()

    algos_filter = set(args.algos.split(","))
    splits = [s for s in args.splits.split(",") if s]
    seeds_filter: set[int] | None = None
    if args.seeds != "all":
        seeds_filter = {int(s) for s in args.seeds.split(",")}

    runs = _latest_train_runs()
    print(f"[replay] {len(runs)} train runs eligible for replay")

    n_done = 0
    for (algo, seed), info in sorted(runs.items()):
        if algo not in algos_filter:
            continue
        if seeds_filter is not None and seed not in seeds_filter:
            continue
        print(f"[replay] {algo} seed={seed} run_id={info['run_id']}")
        for split in splits:
            try:
                _replay_one(
                    algo=algo,
                    seed=seed,
                    run_id=info["run_id"],
                    params_path=info["params_path"],
                    split=split,
                    n_episodes=int(args.n_episodes),
                )
                n_done += 1
            except Exception as exc:
                print(f"  [WARN] {algo} s{seed} {split} failed: {exc}")
    print(f"[replay] done — wrote {n_done} npz files to {REPLAY_DIR}")


if __name__ == "__main__":
    main()
