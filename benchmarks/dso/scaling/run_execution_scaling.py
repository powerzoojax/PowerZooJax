#!/usr/bin/env python
"""DSO execution-scaling throughput driver.

Mirrors the DC Microgrid scaling driver.  Measures steady-state
env steps/second at nenv in {16,32,64,128,256} for jax_rejax and sb3.
Results are written to benchmarks/dso/results/scaling/.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

TASK_DIR = _PROJECT_ROOT / "benchmarks" / "dso"
DEFAULT_OUTPUT_DIR = TASK_DIR / "results" / "scaling"
SCHEMA_VERSION = "dso_execution_scaling_v1"

_DEFAULT_NENVS = "16,32,64,128,256"
_WARMUP_UPDATES = 2
_STEADY_UPDATES = 20


@dataclass(frozen=True)
class _Cell:
    backend: str
    nenv: int
    seed: int


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def _block(tree: Any) -> None:
    import jax
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
            return


# ---------------------------------------------------------------------------
# JAX / rejax cell
# ---------------------------------------------------------------------------

def _run_jax_cell(cell: _Cell, n_steps: int, cache_dir: str) -> dict[str, Any]:
    import jax
    import rejax

    from benchmarks.common.configs import load_task_config, load_train_config
    from benchmarks.common.runtime import build_train_cfg
    from powerzoojax.envs.grid.dist import DistGridEnv
    from powerzoojax.rl.trainer import (
        _RejaxAdapter,
        _rejax_actor_override,
        _rejax_create_kwargs,
    )
    from powerzoojax.rl.wrappers import LogWrapper
    from powerzoojax.tasks.dso import DSOTask, dso_task_kwargs_from_config

    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_enable_compilation_cache", True)

    task_config = load_task_config(TASK_DIR)
    ppo_config = load_train_config(TASK_DIR, "ppo", None)
    max_steps = task_config.get("max_steps", 48)

    task_kwargs = dso_task_kwargs_from_config(task_config)
    task = DSOTask(**task_kwargs)
    params = task.episode_params("train", 0, 1, max_steps, strategy="seeded", seed=cell.seed)

    env = DistGridEnv()
    wrapped = LogWrapper(env, params)
    adapted_env = _RejaxAdapter(wrapped)

    transitions_per_update = cell.nenv * n_steps

    base_cfg = build_train_cfg(ppo_config, algo="ppo").replace(
        num_envs=cell.nenv,
        n_steps=n_steps,
        eval_num_episodes=1,
        record_eval_wall_time=False,
        total_timesteps=max(transitions_per_update, 1),
    )

    create_kwargs = _rejax_create_kwargs(rejax.PPO, base_cfg)
    algo = rejax.PPO.create(env=adapted_env, env_params=None, **create_kwargs)
    override = _rejax_actor_override(base_cfg, adapted_env)
    if override is not None:
        algo = algo.replace(actor=override)

    key = jax.random.PRNGKey(cell.seed)

    t_init = time.perf_counter()
    ts = algo.init_state(key)
    _block(ts)
    init_time = time.perf_counter() - t_init

    compiled_step = jax.jit(algo.train_iteration)

    t0 = time.perf_counter()
    _ = compiled_step.lower(ts).compile()
    compile_time = time.perf_counter() - t0

    warmup_time = 0.0
    for _ in range(_WARMUP_UPDATES):
        t0 = time.perf_counter()
        ts = compiled_step(ts)
        _block(ts)
        warmup_time += time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(_STEADY_UPDATES):
        ts = compiled_step(ts)
        _block(ts)
    steady_time = time.perf_counter() - t0

    sps = cell.nenv * n_steps * _STEADY_UPDATES / steady_time if steady_time > 0 else None
    spu = steady_time / _STEADY_UPDATES if _STEADY_UPDATES > 0 else None
    return {
        "backend": "jax_rejax",
        "nenv": cell.nenv,
        "seed": cell.seed,
        "n_steps": n_steps,
        "status": "completed",
        "steps_per_sec": sps,
        "run_seconds": steady_time,
        "warmup_seconds": warmup_time,
        "compile_time_s": compile_time,
        "init_time_s": init_time,
        "seconds_per_update": spu,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# SB3 cell — mirrors dc_microgrid/_run_python_cell
# ---------------------------------------------------------------------------

def _run_sb3_cell(
    cell: _Cell,
    n_steps: int,
    *,
    vec_env_policy: str = "subproc",
) -> dict[str, Any]:
    import numpy as np
    from stable_baselines3.common.callbacks import BaseCallback

    from benchmarks.common import powerzoo_bridge as bridge

    bridge._ensure_powerzoo_path()

    algorithm = "PPO"
    algo_cls = bridge._algo_class("sb3", algorithm)
    policy_spec, policy_meta = bridge._single_agent_policy_spec("dso", algorithm, backend="sb3")
    algo_kwargs, aligned_source = bridge._single_agent_algo_kwargs(
        "dso", algorithm,
        n_envs=cell.nenv,
        extra_config={"n_steps": n_steps},
    )

    transitions_per_update = cell.nenv * n_steps
    total_updates = _WARMUP_UPDATES + _STEADY_UPDATES
    total_timesteps = max(transitions_per_update * total_updates, transitions_per_update)

    class _UpdateTimer(BaseCallback):
        def __init__(self):
            super().__init__()
            self.boundaries = [
                transitions_per_update * (i + 1) for i in range(total_updates)
            ]
            self.next_idx = 0
            self.update_times: list[float] = []
            self._last = 0.0

        def _on_training_start(self) -> None:
            self._last = time.perf_counter()

        def _on_step(self) -> bool:
            while (
                self.next_idx < len(self.boundaries)
                and self.num_timesteps >= self.boundaries[self.next_idx]
            ):
                now = time.perf_counter()
                self.update_times.append(now - self._last)
                self._last = now
                self.next_idx += 1
            return True

        def _on_training_end(self) -> None:
            if len(self.update_times) < total_updates:
                self.update_times.append(time.perf_counter() - self._last)

    timer = _UpdateTimer()
    handle = bridge._build_powerzoo_vec_env(
        "dso",
        split="train",
        seed=cell.seed,
        n_envs=cell.nenv,
        strategy="seeded",
        vec_env=vec_env_policy,
    )
    vec_env_type = type(handle).__name__
    vec_env_start = getattr(handle, "powerzoojax_start_method", None)

    t_total = time.perf_counter()
    try:
        model = algo_cls(
            policy_spec,
            handle,
            verbose=0,
            seed=cell.seed,
            device="cpu",
            **algo_kwargs,
        )
        model.learn(
            total_timesteps=int(total_timesteps),
            progress_bar=False,
            callback=timer,
        )
        t_total = time.perf_counter() - t_total
    finally:
        try:
            handle.close()
        except Exception:
            pass

    update_times = list(timer.update_times)
    warmup_time = float(np.sum(update_times[:_WARMUP_UPDATES])) if update_times else 0.0
    steady_times = update_times[_WARMUP_UPDATES:]
    if not steady_times:
        steady_times = [max(t_total - warmup_time, 0.0)]
    steady_time = float(np.sum(steady_times))
    steady_updates = max(len(steady_times), 1)
    steady_transitions = transitions_per_update * steady_updates
    sps = steady_transitions / steady_time if steady_time > 0 else None
    spu = steady_time / steady_updates if steady_updates > 0 else None
    return {
        "backend": "sb3",
        "nenv": cell.nenv,
        "seed": cell.seed,
        "n_steps": n_steps,
        "status": "completed",
        "steps_per_sec": sps,
        "run_seconds": steady_time,
        "warmup_seconds": warmup_time,
        "compile_time_s": 0.0,
        "seconds_per_update": spu,
        "vec_env_type": vec_env_type,
        "vec_env_start_method": vec_env_start,
        "aligned_from_train_config": aligned_source,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Dispatch and I/O
# ---------------------------------------------------------------------------

def _key_fields() -> tuple[str, ...]:
    return ("backend", "nenv", "seed", "n_steps")


def _write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "execution_scaling.json"
    csv_path = output_dir / "scaling_results_extended.csv"

    payload = {
        "schema_version": SCHEMA_VERSION,
        "task": "dso",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "results": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    fieldnames = [
        "backend", "nenv", "seed", "steps_per_sec", "run_seconds",
        "warmup_seconds", "compile_time_s", "status",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return json_path, csv_path


def _upsert(rows: list[dict], row: dict, key_fields: tuple) -> list[dict]:
    key = tuple(row.get(k) for k in key_fields)
    kept = [r for r in rows if tuple(r.get(k) for k in key_fields) != key]
    kept.append(row)
    return kept


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nenvs", default=_DEFAULT_NENVS)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--backends", default="jax_rejax,sb3",
                        help="Comma-separated list: jax_rejax, sb3")
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--mode", choices=["dry-run", "formal"], default="formal")
    parser.add_argument(
        "--python-vec-env",
        choices=["subproc", "auto", "dummy"],
        default="subproc",
        help="VecEnv policy for SB3. 'subproc' requires subprocess workers; "
             "failure is logged as failed_subprocvecenv_* status.",
    )
    parser.add_argument("--allow-device-fallback", action="store_true")
    args = parser.parse_args(argv)

    from benchmarks.common.configs import load_train_config
    ppo_cfg = load_train_config(TASK_DIR, "ppo", None)
    n_steps = args.n_steps or int(ppo_cfg.get("n_steps", 48))

    nenvs = _parse_int_list(args.nenvs)
    seeds = _parse_int_list(args.seeds)
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]

    cells = [
        _Cell(backend=b, nenv=n, seed=s)
        for b in backends
        for n in nenvs
        for s in seeds
    ]
    print(
        f"[dso_scaling] {len(cells)} cells: "
        f"backends={backends} nenvs={nenvs} seeds={seeds} n_steps={n_steps}"
    )

    if args.mode == "dry-run":
        for c in cells:
            print(f"  {c.backend} nenv={c.nenv} seed={c.seed}")
        print("[dso_scaling] dry-run, no measurements")
        return

    cache_dir = tempfile.mkdtemp(prefix="dso_jax_cache_")
    output_dir = Path(args.output_dir)
    rows: list[dict[str, Any]] = []

    existing_json = output_dir / "execution_scaling.json"
    if existing_json.exists():
        existing = json.loads(existing_json.read_text(encoding="utf-8"))
        rows = list(existing.get("results", []))

    kf = _key_fields()

    for idx, cell in enumerate(cells, 1):
        print(
            f"[dso_scaling] {idx}/{len(cells)} "
            f"backend={cell.backend} nenv={cell.nenv} seed={cell.seed}",
            flush=True,
        )
        try:
            if cell.backend == "jax_rejax":
                row = _run_jax_cell(cell, n_steps, cache_dir)
            elif cell.backend == "sb3":
                row = _run_sb3_cell(cell, n_steps, vec_env_policy=args.python_vec_env)
            else:
                raise ValueError(f"Unsupported backend: {cell.backend!r}")
        except Exception as exc:
            # Classify the failure mode for the status column
            exc_str = str(exc)
            if "SubprocVecEnv" in exc_str or "Connection reset" in exc_str or "connection reset" in exc_str.lower():
                status = "failed_subprocvecenv_connection_reset"
            elif "out of memory" in exc_str.lower() or "oom" in exc_str.lower():
                status = "failed_oom"
            else:
                status = f"failed_{type(exc).__name__}"
            print(
                f"[dso_scaling] FAILED backend={cell.backend} "
                f"nenv={cell.nenv} seed={cell.seed}: {status}: {exc}",
                flush=True,
            )
            row = {
                "backend": cell.backend,
                "nenv": cell.nenv,
                "seed": cell.seed,
                "n_steps": n_steps,
                "status": status,
                "steps_per_sec": None,
                "run_seconds": None,
                "warmup_seconds": None,
                "compile_time_s": None,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

        rows = _upsert(rows, row, kf)
        print(
            f"[dso_scaling] done backend={cell.backend} "
            f"nenv={cell.nenv} seed={cell.seed} "
            f"status={row['status']} sps={row.get('steps_per_sec')}",
            flush=True,
        )
        _write_outputs(rows, output_dir)

    json_path, csv_path = _write_outputs(rows, output_dir)
    print(f"[dso_scaling] wrote {json_path}")
    print(f"[dso_scaling] wrote {csv_path}")


if __name__ == "__main__":
    main()
