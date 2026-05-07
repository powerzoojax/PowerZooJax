#!/usr/bin/env python
"""Prepare and run DC Microgrid execution-scaling measurements.

This runner writes scaling-only artifacts under
``benchmarks/dc_microgrid/results/scaling/``.  It deliberately does not write
``RunRecord`` rows, update ``manifest.json``, or contribute to the algorithm
effect leaderboard.  The metrics are execution metrics only; final reward is
not recorded or compared.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.common.configs import load_task_config, load_train_config
from benchmarks.common.io import collect_jax_run_contract

TASK_DIR = _PROJECT_ROOT / "benchmarks" / "dc_microgrid"
DEFAULT_OUTPUT_DIR = TASK_DIR / "results" / "scaling"
SCHEMA_VERSION = "dc_microgrid_execution_scaling_v1"
THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "XLA_FLAGS",
    "JAX_PLATFORM_NAME",
    "JAX_PLATFORMS",
    "CUDA_VISIBLE_DEVICES",
)


@dataclass(frozen=True)
class ScalingCell:
    backend: str
    device: str
    nenv: int
    seed: int


def _parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError(f"Expected at least one integer in {raw!r}")
    return values


def _thread_settings() -> dict[str, str]:
    return {key: os.environ[key] for key in THREAD_ENV_KEYS if key in os.environ}


def _cpu_core_budget(args: argparse.Namespace) -> int | None:
    if args.cpu_core_budget is not None:
        return int(args.cpu_core_budget)
    if hasattr(os, "sched_getaffinity"):
        try:
            return len(os.sched_getaffinity(0))
        except Exception:
            pass
    return os.cpu_count()


def _mode_updates(mode: str, max_updates: int | None) -> int:
    if max_updates is not None:
        return int(max_updates)
    if mode == "pilot":
        return 3
    if mode == "formal":
        return 20
    return 0


def _resolve_cells(args: argparse.Namespace) -> list[ScalingCell]:
    seeds = _parse_int_list(args.seeds)
    cells: list[ScalingCell] = []
    if args.suite == "single":
        backend = args.backend
        device = args.device or ("gpu" if backend == "jax_rejax" else "cuda")
        nenvs = [int(args.nenv)]
        backends = [(backend, device)]
    elif args.suite == "matched":
        nenvs = _parse_int_list(args.matched_nenvs)
        python_backend = args.python_backend
        backends = [("jax_rejax", "gpu"), (python_backend, "cuda")]
    elif args.suite == "jax-extended":
        nenvs = _parse_int_list(args.jax_extended_nenvs)
        backends = [("jax_rejax", "gpu")]
    elif args.suite == "all":
        nenvs = sorted(
            set(_parse_int_list(args.matched_nenvs))
            | set(_parse_int_list(args.jax_extended_nenvs))
        )
        python_backend = args.python_backend
        backends = [("jax_rejax", "gpu"), (python_backend, "cuda")]
    else:
        raise ValueError(f"Unknown suite: {args.suite!r}")

    for backend, device in backends:
        for nenv in nenvs:
            if args.suite == "all" and backend != "jax_rejax":
                if nenv not in set(_parse_int_list(args.matched_nenvs)):
                    continue
            for seed in seeds:
                cells.append(ScalingCell(backend=backend, device=device, nenv=int(nenv), seed=seed))
    return cells


def _load_existing(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "task": "dc_microgrid",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "notes": (
                "Scaling-only execution artifact. These rows are not algorithm "
                "effect records and must not enter the final-reward leaderboard."
            ),
            "results": [],
        }
    return json.loads(path.read_text(encoding="utf-8"))


def _write_outputs(payload: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "execution_scaling.json"
    csv_path = output_dir / "execution_scaling_table.csv"
    rows = list(payload.get("results", []))
    _attach_speedups(rows)
    payload["results"] = rows
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    fieldnames = [
        "suite",
        "mode",
        "backend",
        "actual_backend",
        "device",
        "actual_device",
        "seed",
        "nenv",
        "n_steps",
        "transitions_per_update",
        "compile_time_s",
        "warmup_time_s",
        "steady_state_env_steps_per_second",
        "seconds_per_update",
        "wall_clock_per_1M_transitions",
        "speedup",
        "speedup_reference",
        "cpu_core_budget",
        "status",
        "measurement_scope",
        "vec_env_type",
        "vec_env_start_method",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return json_path, csv_path


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0.0 else None


def _attach_speedups(rows: list[dict[str, Any]]) -> None:
    matched_refs: dict[tuple[Any, ...], float] = {}
    extended_refs: dict[tuple[Any, ...], float] = {}
    for row in rows:
        sps = _as_float(row.get("steady_state_env_steps_per_second"))
        if sps is None:
            continue
        suite = row.get("suite")
        if suite in ("matched", "all") and row.get("backend") in ("sb3", "sbx"):
            key = (row.get("mode"), row.get("seed"), row.get("nenv"), row.get("n_steps"))
            matched_refs[key] = sps
        if row.get("backend") == "jax_rejax" and int(row.get("nenv") or 0) == 32:
            key = (row.get("mode"), row.get("backend"), row.get("device"), row.get("seed"), row.get("n_steps"))
            extended_refs[key] = sps

    for row in rows:
        row["speedup"] = None
        row["speedup_reference"] = None
        sps = _as_float(row.get("steady_state_env_steps_per_second"))
        if sps is None:
            continue
        suite = row.get("suite")
        if suite in ("matched", "all"):
            key = (row.get("mode"), row.get("seed"), row.get("nenv"), row.get("n_steps"))
            ref = matched_refs.get(key)
            if ref:
                row["speedup"] = sps / ref
                row["speedup_reference"] = "python_backend_same_nenv_seed"
        elif suite == "jax-extended":
            key = (row.get("mode"), row.get("backend"), row.get("device"), row.get("seed"), row.get("n_steps"))
            ref = extended_refs.get(key)
            if ref:
                row["speedup"] = sps / ref
                row["speedup_reference"] = "jax_rejax_gpu_nenv32_same_seed"


def _replace_result(payload: dict[str, Any], row: dict[str, Any]) -> None:
    key_fields = ("suite", "mode", "backend", "device", "seed", "nenv", "n_steps")
    key = tuple(row.get(k) for k in key_fields)
    kept = [
        old for old in payload.get("results", [])
        if tuple(old.get(k) for k in key_fields) != key
    ]
    kept.append(row)
    payload["results"] = kept
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()


def _make_base_row(
    *,
    args: argparse.Namespace,
    cell: ScalingCell,
    status: str,
    message: str | None = None,
) -> dict[str, Any]:
    n_steps = int(args.n_steps)
    transitions_per_update = int(cell.nenv) * n_steps
    return {
        "schema_version": SCHEMA_VERSION,
        "task": "dc_microgrid",
        "suite": args.suite,
        "mode": args.mode,
        "backend": cell.backend,
        "device": cell.device,
        "seed": int(cell.seed),
        "split": args.split,
        "nenv": int(cell.nenv),
        "n_steps": n_steps,
        "transitions_per_update": transitions_per_update,
        "total_transitions_per_update": transitions_per_update,
        "max_updates": int(args.max_updates_resolved),
        "warmup_updates": int(args.warmup_updates),
        "cpu_core_budget": _cpu_core_budget(args),
        "thread_settings": _thread_settings(),
        "status": status,
        "message": message,
        "records_leaderboard_effect": False,
        "leaderboard_policy": "excluded_scaling_only",
        "compares_final_reward": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _block_until_ready(tree: Any) -> None:
    import jax

    leaves = jax.tree_util.tree_leaves(tree)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
            return


def _run_jax_cell(args: argparse.Namespace, cell: ScalingCell) -> dict[str, Any]:
    import jax

    from benchmarks.common.runtime import build_train_cfg
    from benchmarks.dc_microgrid._reward_shaping import wrap_with_shaping
    from powerzoojax.envs.microgrid import DataCenterMicrogridEnv
    from powerzoojax.rl.trainer import (
        _RejaxAdapter,
        _rejax_actor_override,
        _rejax_create_kwargs,
    )
    from powerzoojax.rl.wrappers import LogWrapper
    from powerzoojax.tasks.dc_microgrid import DCMicrogridTask
    import rejax

    row = _make_base_row(args=args, cell=cell, status="running")
    task_config = load_task_config(Path(args.task_dir))
    ppo_config = load_train_config(Path(args.task_dir), "ppo", None, allowed_algos=("ppo",))
    task = DCMicrogridTask(
        source=task_config.get("data_source", "google"),
        max_steps=int(task_config.get("max_steps", 288)),
        case_overrides=task_config.get("case_overrides") or {},
    )
    params = task.episode_params(
        args.split,
        0,
        1,
        int(task_config.get("max_steps", 288)),
        strategy="seeded",
        seed=int(cell.seed),
    )
    env = wrap_with_shaping(DataCenterMicrogridEnv(), task_config)
    wrapped = LogWrapper(env, params)
    adapted_env = _RejaxAdapter(wrapped)

    requested = "gpu" if cell.device in ("gpu", "cuda") else cell.device
    _record_device, env_info, labels = collect_jax_run_contract(
        requested_device=requested,
        context="dc_microgrid/execution_scaling",
        extra_labels={"record_kind": "scaling_only"},
        fail_fast=not args.allow_device_fallback,
    )

    transitions_per_update = int(cell.nenv) * int(args.n_steps)
    train_cfg = build_train_cfg(ppo_config, algo="ppo").replace(
        num_envs=int(cell.nenv),
        n_steps=int(args.n_steps),
        total_timesteps=max(
            transitions_per_update
            * (int(args.warmup_updates) + int(args.max_updates_resolved)),
            transitions_per_update,
        ),
        record_eval_wall_time=False,
        eval_num_episodes=1,
    )
    create_kwargs = _rejax_create_kwargs(rejax.PPO, train_cfg)
    algo = rejax.PPO.create(env=adapted_env, env_params=None, **create_kwargs)
    actor_override = _rejax_actor_override(train_cfg, adapted_env)
    if actor_override is not None:
        algo = algo.replace(actor=actor_override)

    key = jax.random.PRNGKey(int(cell.seed))
    t0 = time.perf_counter()
    train_state = algo.init_state(key)
    _block_until_ready(train_state)
    init_time = time.perf_counter() - t0

    def one_update(state):
        return algo.train_iteration(state)

    compiled_update = jax.jit(one_update)
    t0 = time.perf_counter()
    executable = compiled_update.lower(train_state).compile()
    compile_time = time.perf_counter() - t0

    warmup_time = 0.0
    for _idx in range(int(args.warmup_updates)):
        t0 = time.perf_counter()
        train_state = executable(train_state)
        _block_until_ready(train_state)
        warmup_time += time.perf_counter() - t0

    steady_updates = int(args.max_updates_resolved)
    t0 = time.perf_counter()
    for _idx in range(steady_updates):
        train_state = executable(train_state)
        _block_until_ready(train_state)
    steady_time = time.perf_counter() - t0

    steady_transitions = max(int(cell.nenv) * int(args.n_steps) * steady_updates, 1)
    sps = steady_transitions / steady_time if steady_time > 0 else None
    seconds_per_update = steady_time / steady_updates if steady_updates > 0 else None
    row.update(
        {
            "status": "completed",
            "measurement_scope": "jax_rejax_ppo_train_iteration_including_gradient_update",
            "init_time_s": init_time,
            "compile_time_s": compile_time,
            "warmup_time_s": warmup_time,
            "steady_state_time_s": steady_time,
            "steady_state_env_steps_per_second": sps,
            "seconds_per_update": seconds_per_update,
            "wall_clock_per_1M_transitions": (1_000_000 / sps) if sps else None,
            "actual_backend": labels.get("actual_backend"),
            "actual_device": labels.get("actual_device"),
            "actual_device_kind": labels.get("actual_device_kind"),
            "env_info": env_info,
        }
    )
    return row


def _run_python_cell(args: argparse.Namespace, cell: ScalingCell) -> dict[str, Any]:
    import numpy as np
    from stable_baselines3.common.callbacks import BaseCallback

    from benchmarks.common import powerzoo_bridge as bridge

    row = _make_base_row(args=args, cell=cell, status="running")
    bridge._ensure_powerzoo_path()
    algorithm = "SBX_PPO" if cell.backend == "sbx" else "PPO"
    algo_cls = bridge._algo_class(cell.backend, algorithm)
    policy_spec, policy_meta = bridge._single_agent_policy_spec(
        "dc_microgrid",
        algorithm,
        backend=cell.backend,
    )
    algo_kwargs, aligned_source = bridge._single_agent_algo_kwargs(
        "dc_microgrid",
        algorithm,
        n_envs=int(cell.nenv),
        extra_config={"n_steps": int(args.n_steps)},
    )
    env_info, labels = bridge._collect_torch_run_contract(
        requested_device=cell.device,
        context="dc_microgrid/execution_scaling",
        meta={},
        extra_labels={"record_kind": "scaling_only"},
        fail_fast=not args.allow_device_fallback,
    )
    handle = bridge._build_powerzoo_vec_env(
        "dc_microgrid",
        split=args.split,
        seed=int(cell.seed),
        n_envs=int(cell.nenv),
        strategy="seeded",
        vec_env=args.python_vec_env,
    )
    vec_env_type = type(handle).__name__
    vec_env_start_method = getattr(handle, "powerzoojax_start_method", None)

    transitions_per_update = int(cell.nenv) * int(args.n_steps)
    total_updates = int(args.warmup_updates) + int(args.max_updates_resolved)
    total_timesteps = max(transitions_per_update * total_updates, transitions_per_update)

    class UpdateTimer(BaseCallback):
        def __init__(self):
            super().__init__()
            self.boundaries = [
                transitions_per_update * (idx + 1) for idx in range(total_updates)
            ]
            self.next_idx = 0
            self.update_times: list[float] = []
            self._last = 0.0

        def _on_training_start(self) -> None:
            self._last = time.perf_counter()

        def _on_step(self) -> bool:
            while self.next_idx < len(self.boundaries) and self.num_timesteps >= self.boundaries[self.next_idx]:
                now = time.perf_counter()
                self.update_times.append(now - self._last)
                self._last = now
                self.next_idx += 1
            return True

        def _on_training_end(self) -> None:
            if len(self.update_times) < total_updates:
                now = time.perf_counter()
                self.update_times.append(now - self._last)

    timer = UpdateTimer()
    try:
        model = algo_cls(
            policy_spec,
            handle,
            verbose=0,
            seed=int(cell.seed),
            device=cell.device,
            **algo_kwargs,
        )
        t0 = time.perf_counter()
        model.learn(total_timesteps=int(total_timesteps), progress_bar=False, callback=timer)
        total_time = time.perf_counter() - t0
    finally:
        try:
            handle.close()
        except Exception:
            pass

    update_times = list(timer.update_times)
    warmup_time = float(np.sum(update_times[: int(args.warmup_updates)])) if update_times else 0.0
    steady_times = update_times[int(args.warmup_updates):]
    if not steady_times:
        steady_times = [max(total_time - warmup_time, 0.0)]
    steady_time = float(np.sum(steady_times))
    steady_updates = max(len(steady_times), 1)
    steady_transitions = max(transitions_per_update * steady_updates, 1)
    sps = steady_transitions / steady_time if steady_time > 0 else None
    seconds_per_update = steady_time / steady_updates if steady_updates > 0 else None
    row.update(
        {
            "status": "completed",
            "measurement_scope": "python_backend_ppo_learn_scaling_only_no_artifact_save",
            "compile_time_s": 0.0,
            "warmup_time_s": warmup_time,
            "steady_state_time_s": steady_time,
            "steady_state_env_steps_per_second": sps,
            "seconds_per_update": seconds_per_update,
            "wall_clock_per_1M_transitions": (1_000_000 / sps) if sps else None,
            "actual_backend": labels.get("actual_backend"),
            "actual_device": labels.get("actual_device"),
            "actual_device_kind": labels.get("actual_device_kind"),
            "env_info": env_info,
            "policy_class": policy_meta.get("policy_class"),
            "effective_continuous_action_dist": policy_meta.get("effective_continuous_action_dist"),
            "aligned_from_train_config": aligned_source,
            "vec_env_type": vec_env_type,
            "vec_env_start_method": vec_env_start_method,
        }
    )
    return row


def _plan_row(args: argparse.Namespace, cell: ScalingCell) -> dict[str, Any]:
    row = _make_base_row(
        args=args,
        cell=cell,
        status="planned",
        message="dry-run only; no training, eval, or scaling measurement executed",
    )
    row.update(
        {
            "actual_backend": None,
            "actual_device": None,
            "measurement_scope": (
                "planned_jax_rejax_ppo_train_iteration"
                if cell.backend == "jax_rejax"
                else "planned_python_ppo_learn_scaling_only"
            ),
            "compile_time_s": None,
            "warmup_time_s": None,
            "steady_state_env_steps_per_second": None,
            "seconds_per_update": None,
            "wall_clock_per_1M_transitions": None,
        }
    )
    return row


def run_cell(args: argparse.Namespace, cell: ScalingCell) -> dict[str, Any]:
    if args.mode == "dry-run":
        return _plan_row(args, cell)
    if cell.backend == "jax_rejax":
        return _run_jax_cell(args, cell)
    if cell.backend in ("sb3", "sbx"):
        return _run_python_cell(args, cell)
    raise ValueError(f"Unsupported backend for scaling: {cell.backend!r}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["dry-run", "pilot", "formal"], default="dry-run")
    parser.add_argument("--suite", choices=["single", "matched", "jax-extended", "all"], default="single")
    parser.add_argument("--backend", choices=["jax_rejax", "sb3", "sbx"], default="jax_rejax")
    parser.add_argument("--python-backend", choices=["sb3", "sbx"], default="sb3")
    parser.add_argument("--device", default=None)
    parser.add_argument("--nenv", type=int, default=32)
    parser.add_argument("--matched-nenvs", default="16,32,64")
    parser.add_argument("--jax-extended-nenvs", default="32,64,128,256")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--split", default="iid")
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--warmup-updates", type=int, default=1)
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--cpu-core-budget", type=int, default=None)
    parser.add_argument(
        "--python-vec-env",
        choices=["subproc", "auto", "dummy"],
        default="subproc",
        help=(
            "Python backend VecEnv policy. Default requires SubprocVecEnv so "
            "scaling rows do not silently fall back to single-process DummyVecEnv."
        ),
    )
    parser.add_argument("--allow-device-fallback", action="store_true")
    parser.add_argument("--write-dry-run", action="store_true")
    parser.add_argument("--task-dir", default=str(TASK_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)

    task_dir = Path(args.task_dir)
    ppo_cfg = load_train_config(task_dir, "ppo", None, allowed_algos=("ppo",))
    if args.n_steps is None:
        args.n_steps = int(ppo_cfg.get("n_steps", 288))
    args.max_updates_resolved = _mode_updates(args.mode, args.max_updates)

    cells = _resolve_cells(args)
    print("[execution_scaling] planned cells:")
    for cell in cells:
        print(
            "  "
            f"mode={args.mode} suite={args.suite} backend={cell.backend} "
            f"device={cell.device} nenv={cell.nenv} seed={cell.seed} "
            f"n_steps={args.n_steps} max_updates={args.max_updates_resolved}"
        )
    if args.mode == "dry-run" and not args.write_dry_run:
        print("[execution_scaling] dry-run only; not writing scaling artifacts")
        return

    output_dir = Path(args.output_dir)
    json_path = output_dir / "execution_scaling.json"
    payload = _load_existing(json_path)
    for idx, cell in enumerate(cells, start=1):
        print(
            "[execution_scaling] running "
            f"{idx}/{len(cells)}: mode={args.mode} suite={args.suite} "
            f"backend={cell.backend} device={cell.device} "
            f"nenv={cell.nenv} seed={cell.seed}",
            flush=True,
        )
        row = run_cell(args, cell)
        _replace_result(payload, row)
        _write_outputs(payload, output_dir)
        print(
            "[execution_scaling] completed "
            f"{idx}/{len(cells)}: status={row.get('status')}",
            flush=True,
        )
    written_json, written_csv = _write_outputs(payload, output_dir)
    print(f"[execution_scaling] wrote {written_json}")
    print(f"[execution_scaling] wrote {written_csv}")


if __name__ == "__main__":
    main()
