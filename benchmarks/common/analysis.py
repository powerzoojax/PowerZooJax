"""Cross-task analysis helpers.

This file groups paper-summary plotting and throughput benchmarking so the
analysis surface is easier to discover from one place.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BENCHMARKS_ROOT = Path(__file__).resolve().parent.parent
TASKS = ["tso", "dso", "ders", "dc_microgrid", "gencos"]

def load_all_summaries() -> dict[str, dict]:
    """Load latest summary from each task that has one."""
    summaries = {}
    for task in TASKS:
        path = BENCHMARKS_ROOT / task / "results" / "summary" / "latest.json"
        if path.exists():
            summaries[task] = json.loads(path.read_text(encoding="utf-8"))
    return summaries

def print_comparison_table(summaries: dict[str, dict]) -> None:
    """Print a Markdown comparison table to stdout."""
    if not summaries:
        print("No summaries found. Run each task's summarize.py first.")
        return

    print("| Task | Algo | NormScore (IQM) | Feasibility | Notes |")
    print("|------|------|-----------------|-------------|-------|")
    for task, summary in summaries.items():
        for entry in summary.get("rows", []):
            print(
                f"| {task} | {entry.get('algo', '?')} "
                f"| {entry.get('norm_score_iqm', '—')} "
                f"| {entry.get('feasibility', '—')} "
                f"| {entry.get('notes', '')} |"
            )

def plot_main(argv: "list[str] | None" = None):
    parser = argparse.ArgumentParser(description="Generate cross-task figures")
    parser.add_argument(
        "--output-dir", type=str, default="figures",
        help="Directory for output figures",
    )
    args = parser.parse_args(argv)

    summaries = load_all_summaries()
    print_comparison_table(summaries)

import json
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

_RESULTS_DIR = Path(__file__).resolve().parent / "results"
_BENCHMARKS_DIR = Path(__file__).resolve().parent.parent

# Fallback values used only if benchmarks/<task>/configs/task.{yaml,json} cannot
# be read or does not contain "num_envs". The single source of truth is
# task.yaml; tests/benchmarks/test_config_consistency.py asserts agreement.
_FALLBACK_NUM_ENVS: dict[str, int] = {
    "dso": 512,
    "tso": 1024,
    "ders": 256,
    "dc_microgrid": 256,
    "gencos": 512,
}

def _load_frozen_num_envs() -> dict[str, int]:
    """Read num_envs from each task's configs/task.yaml (source of truth)."""
    from benchmarks.common.configs import load_task_config
    out: dict[str, int] = {}
    for task in _FALLBACK_NUM_ENVS:
        try:
            cfg = load_task_config(_BENCHMARKS_DIR / task)
            out[task] = int(cfg.get("num_envs", _FALLBACK_NUM_ENVS[task]))
        except (FileNotFoundError, ValueError):
            out[task] = _FALLBACK_NUM_ENVS[task]
    return out

# Populated at import time from the merged task config; downstream code can keep importing
# FROZEN_NUM_ENVS as a constant.
FROZEN_NUM_ENVS: dict[str, int] = _load_frozen_num_envs()

def _resolve_max_steps(params, default: int = 48, env=None) -> int:
    """Find ``max_steps`` on a possibly-nested params pytree.

    DC Microgrid stores it under ``params.dc.max_steps`` rather than at the
    top level; without this helper the throughput script silently uses the
    48-step default and under-counts that task by ~6x.

    For MARL tasks where ``params is None`` (env holds its own params), the
    adapter wrapper exposes ``env.max_steps``; we fall back to that.
    """
    if params is not None:
        if hasattr(params, "max_steps"):
            return int(params.max_steps)
        for sub_name in ("dc", "grid", "env"):
            sub = getattr(params, sub_name, None)
            if sub is not None and hasattr(sub, "max_steps"):
                return int(sub.max_steps)
    if env is not None and hasattr(env, "max_steps"):
        return int(env.max_steps)
    return int(default)

# num_envs sweep points for the scaling curve figure
SWEEP_NUM_ENVS = [1, 8, 32, 64, 128, 256]

# Number of timing episodes for steady-state sps measurement
N_TIMING_EPISODES = 5

# Env factory helpers
def _make_env_and_params(task: str):
    """Return (env, params) for the given task using synthetic/default config."""
    if task == "dso":
        from powerzoojax.envs.grid.dist import DistGridEnv
        from powerzoojax.tasks.dso import make_dso_params
        env = DistGridEnv()
        params = make_dso_params()
        return env, params

    elif task == "tso":
        from powerzoojax.envs.grid.unit_commitment import UnitCommitmentEnv
        from powerzoojax.tasks.tso import make_tso_case118_params
        env = UnitCommitmentEnv()
        params = make_tso_case118_params(enable_uc=True, enable_reserve=True)
        return env, params

    elif task == "dc_microgrid":
        from powerzoojax.envs.microgrid import (
            DataCenterMicrogridEnv,
            make_dcmicrogrid_params,
        )
        env = DataCenterMicrogridEnv()
        params = make_dcmicrogrid_params(max_steps=288)
        return env, params

    elif task == "ders":
        from powerzoojax.tasks.ders import make_ders_marl_env
        marl_env, _params = make_ders_marl_env()  # factory returns (env, params)
        return _MARLThroughputAdapter(marl_env), None  # params held inside env

    elif task == "gencos":
        # GenCos has no top-level make_market_marl_env factory; construct
        # MarketMARLEnv from case5 + a flat synthetic load profile (throughput
        # only — semantics don't matter for sps measurement).
        from powerzoojax.case import create_case5
        from powerzoojax.envs.market.market_marl_core import make_market_marl_params
        from powerzoojax.rl.market_marl import MarketMARLEnv

        case = create_case5()
        mid_load = (case.load_d_max + case.load_d_min) / 2.0
        profiles = jnp.tile(mid_load[None, :], (48, 1))
        params = make_market_marl_params(case, profiles, n_segments=3)
        marl_env = MarketMARLEnv(params)
        return _MARLThroughputAdapter(marl_env), None

    else:
        raise ValueError(f"Unknown task: {task!r}")

# MARL env → single-agent throughput adapter (for SB3-style timing helpers)

class _MARLThroughputAdapter:
    """Throughput-only single-agent wrapper around a PowerZooJax MARL env.

    PowerZooJax MARL envs (``DistGridMARLEnv``, ``MarketMARLEnv``) hold
    ``params`` internally and use ``reset(key)`` / ``step(key, state,
    dict_actions)`` returning dicts keyed by agent name.

    ``_measure_jax_vmap_scan`` expects the gymnax-style flat API
    ``(obs, state) = env.reset(key, params)`` /
    ``(obs, state, reward, done, info) = env.step(key, state, action, params)``.

    This adapter:
      - Concatenates per-agent observations into one flat array (for shape).
      - Splits a flat zero action back into the per-agent dict the MARL env
        expects (zero is a valid action for all 5 task envs).
      - Sums per-agent rewards into a scalar (dummy; throughput doesn't care).
      - Forwards ``done["__all__"]``.

    NOT for training — semantics of "single-agent flatten" differ from real
    MARL training. Use only inside throughput.py.
    """

    def __init__(self, marl_env):
        from powerzoojax.envs.spaces import Box

        self._marl = marl_env
        self._agents = list(marl_env.agent_names)
        per_act_space = marl_env.action_space(self._agents[0])
        per_obs_space = marl_env.observation_space(self._agents[0])
        self._action_dim = int(per_act_space.shape[0]) if per_act_space.shape else 1
        self._obs_dim = int(per_obs_space.shape[0]) if per_obs_space.shape else 1
        self._n_agents = len(self._agents)
        self._jnp = jnp
        self._Box = Box
        # Resolve a max_steps surrogate; PowerZooJax MARL envs typically expose
        # this through the underlying single-agent env's params.
        self._max_steps = self._infer_max_steps()

    def _infer_max_steps(self) -> int:
        for attr in ("_grid_env", "_market_env"):
            inner = getattr(self._marl, attr, None)
            if inner is None:
                continue
            params = getattr(self._marl, "_grid_params", None) or getattr(
                self._marl, "_market_params", None
            )
            if params is not None and hasattr(params, "max_steps"):
                return int(params.max_steps)
        return 48  # safe default for all current MARL tasks (DERs/GenCos)

    @property
    def max_steps(self) -> int:
        return self._max_steps

    def reset(self, key, params):
        obs_dict, state = self._marl.reset(key)
        flat = self._jnp.concatenate([obs_dict[a].reshape(-1) for a in self._agents])
        return flat, state

    def step(self, key, state, action, params):
        actions = {
            a: action[i * self._action_dim : (i + 1) * self._action_dim]
            for i, a in enumerate(self._agents)
        }
        obs_dict, new_state, reward_dict, done_dict, info = self._marl.step(
            key, state, actions
        )
        flat_obs = self._jnp.concatenate([obs_dict[a].reshape(-1) for a in self._agents])
        # Sum rewards across agents into a scalar (throughput doesn't care
        # about reward magnitude; it just needs a numeric value to scan over).
        reward_sum = sum(reward_dict.values())
        done = done_dict.get("__all__", next(iter(done_dict.values())))
        return flat_obs, new_state, reward_sum, done, info

    def action_space(self, params):
        return self._Box(
            low=self._jnp.full((self._n_agents * self._action_dim,), -1.0, dtype=self._jnp.float32),
            high=self._jnp.full((self._n_agents * self._action_dim,), 1.0, dtype=self._jnp.float32),
            shape=(self._n_agents * self._action_dim,),
            dtype=self._jnp.float32,
        )

    def observation_space(self, params):
        return self._Box(
            low=self._jnp.full((self._n_agents * self._obs_dim,), -self._jnp.inf, dtype=self._jnp.float32),
            high=self._jnp.full((self._n_agents * self._obs_dim,), self._jnp.inf, dtype=self._jnp.float32),
            shape=(self._n_agents * self._obs_dim,),
            dtype=self._jnp.float32,
        )

# JAX vmap+scan measurement
def _measure_jax_vmap_scan(
    env,
    params,
    num_envs: int,
    n_timing_episodes: int = N_TIMING_EPISODES,
) -> dict[str, float]:
    """Measure JAX vmap+lax.scan throughput.

    Returns:
        {
          "compile_time_s":   time for first JIT compilation (single env),
          "warmup_time_s":    time for first vmap+scan episode (JIT + trace),
          "sps_mean":         steady-state steps per second (mean over timing episodes),
          "sps_std":          std across timing episodes,
          "episode_steps":    total steps per timing batch (num_envs * max_steps),
        }
    """
    action_space = env.action_space(params)
    obs_space = env.observation_space(params)
    max_steps = _resolve_max_steps(params, env=env)

    def reset_fn(key):
        return env.reset(key, params)

    def step_fn(key, state, action):
        return env.step(key, state, action, params)

    batch_reset = jax.vmap(reset_fn)
    batch_step = jax.vmap(step_fn)

    def scan_body(carry, _):
        keys, states = carry
        split_keys = jax.vmap(lambda k: jax.random.split(k, 2))(keys)
        keys = split_keys[:, 0, :]
        k_step = split_keys[:, 1, :]
        # Dummy zero action (for throughput measurement only)
        actions = jnp.zeros((num_envs,) + action_space.shape, dtype=jnp.float32)
        obs, new_states, rewards, dones, infos = batch_step(k_step, states, actions)
        return (keys, new_states), rewards

    def full_episode(key):
        keys = jax.random.split(key, num_envs)
        obs, states = batch_reset(keys)
        (keys, final_states), reward_seq = jax.lax.scan(
            scan_body, (keys, states), None, length=max_steps
        )
        return reward_seq

    full_episode_jit = jax.jit(full_episode)

    # JIT compilation time (single-env reset+step)
    single_reset_jit = jax.jit(lambda k: env.reset(k, params))
    t0 = time.perf_counter()
    _ = single_reset_jit(jax.random.PRNGKey(0))
    jax.block_until_ready(_)
    compile_time_s = time.perf_counter() - t0

    # Warmup (first full vmap+scan episode — includes trace compilation)
    key = jax.random.PRNGKey(42)
    t0 = time.perf_counter()
    rewards = full_episode_jit(key)
    jax.block_until_ready(rewards)
    warmup_time_s = time.perf_counter() - t0

    # Steady-state timing
    episode_steps = num_envs * max_steps
    timing_s = []
    for i in range(n_timing_episodes):
        key, subkey = jax.random.split(key)
        t0 = time.perf_counter()
        rewards = full_episode_jit(subkey)
        jax.block_until_ready(rewards)
        elapsed = time.perf_counter() - t0
        timing_s.append(elapsed)

    sps_per_ep = [episode_steps / t for t in timing_s]
    return {
        "compile_time_s": float(compile_time_s),
        "warmup_time_s": float(warmup_time_s),
        "sps_mean": float(np.mean(sps_per_ep)),
        "sps_std": float(np.std(sps_per_ep)),
        "episode_steps": int(episode_steps),
        "num_envs": int(num_envs),
        "max_steps": int(max_steps),
        "timing_episodes": [float(t) for t in timing_s],
    }

# Python for-loop measurement (baseline — mimics Gymnasium-style)
def _measure_python_forloop(
    env,
    params,
    num_envs: int = 1,
    n_timing_episodes: int = N_TIMING_EPISODES,
) -> dict[str, float]:
    """Measure Python for-loop throughput for a single environment.

    Runs env.reset() + env.step() in a Python for-loop without vmap or lax.scan.
    This mimics the Gymnasium-style usage pattern (the baseline we compare against).

    Note: For fairness, runs with num_envs=1 (no batching). The paper compares
    JAX(num_envs=1, for-loop JIT) vs JAX(num_envs=N, vmap+scan) to isolate the
    parallelism benefit, plus JAX(vmap) vs pure Python to show the execution model benefit.
    """
    action_space = env.action_space(params)
    max_steps = _resolve_max_steps(params, env=env)

    reset_jit = jax.jit(lambda k: env.reset(k, params))
    step_jit = jax.jit(lambda k, s, a: env.step(k, s, a, params))

    # Warmup JIT
    key = jax.random.PRNGKey(0)
    obs, state = reset_jit(key)
    jax.block_until_ready(obs)
    action = jnp.zeros(action_space.shape, dtype=jnp.float32)
    key, k1 = jax.random.split(key)
    out = step_jit(k1, state, action)
    jax.block_until_ready(out[0])

    # Time Python for-loop episodes
    timing_s = []
    for ep in range(n_timing_episodes):
        key, subkey = jax.random.split(key)
        t0 = time.perf_counter()
        obs, state = reset_jit(subkey)
        jax.block_until_ready(obs)
        for _ in range(max_steps):
            subkey, k_step = jax.random.split(subkey)
            action = jnp.zeros(action_space.shape, dtype=jnp.float32)
            obs, state, reward, done, info = step_jit(k_step, state, action)
            jax.block_until_ready(reward)  # sync after each step
        elapsed = time.perf_counter() - t0
        timing_s.append(elapsed)

    sps_per_ep = [max_steps / t for t in timing_s]
    return {
        "sps_mean": float(np.mean(sps_per_ep)),
        "sps_std": float(np.std(sps_per_ep)),
        "num_envs": 1,
        "max_steps": int(max_steps),
        "timing_episodes": [float(t) for t in timing_s],
        "mode": "python_forloop_jit",
    }

# num_envs scaling sweep
def measure_scaling_curve(
    task: str,
    sweep_points: list[int] = SWEEP_NUM_ENVS,
) -> dict[str, Any]:
    """Run vmap+scan throughput at each num_envs point in sweep_points.

    Returns a dict suitable for serialisation and plotting:
      {
        "task": task,
        "sweep": [{"num_envs": N, "sps_mean": ..., "sps_std": ...}, ...]
      }
    """
    env, params = _make_env_and_params(task)
    results = []
    for n in sweep_points:
        print(f"  [{task}] num_envs={n} ...", flush=True)
        try:
            r = _measure_jax_vmap_scan(env, params, num_envs=n, n_timing_episodes=3)
            results.append({
                "num_envs": n,
                "sps_mean": r["sps_mean"],
                "sps_std": r["sps_std"],
                "warmup_time_s": r["warmup_time_s"],
                "compile_time_s": r["compile_time_s"],
            })
        except Exception as exc:
            print(f"  [{task}] num_envs={n} FAILED: {exc}")
            results.append({"num_envs": n, "error": str(exc)})
    return {"task": task, "sweep": results}

# Full comparison for one task
def measure_task(task: str, num_envs: int | None = None) -> dict[str, Any]:
    """Full JAX-vs-Python throughput comparison for one task.

    Returns a dict with:
      - task, num_envs, jax_version, python_version, cuda_version
      - jax_vmap_scan: vmap+scan steady-state sps
      - python_forloop: single-env for-loop sps
      - speedup_ratio: jax_sps / python_sps (normalised to per-env)
    """
    import sys
    n = num_envs or FROZEN_NUM_ENVS.get(task, 64)

    print(f"[throughput] {task} num_envs={n}", flush=True)
    env, params = _make_env_and_params(task)

    jax_result = _measure_jax_vmap_scan(env, params, num_envs=n)
    py_result = _measure_python_forloop(env, params, num_envs=1)

    # Per-env speedup: JAX sps / n  vs Python sps (single env)
    jax_per_env_sps = jax_result["sps_mean"] / n
    speedup = jax_per_env_sps / max(py_result["sps_mean"], 1e-9)

    try:
        import jax as _jax
        jax_ver = _jax.__version__
        cuda_info = str(_jax.devices()[0]).lower()
    except Exception:
        jax_ver, cuda_info = "unknown", "unknown"

    return {
        "task": task,
        "num_envs": n,
        "jax_version": jax_ver,
        "python_version": sys.version.split()[0],
        "cuda_info": cuda_info,
        "jax_vmap_scan": jax_result,
        "python_forloop": py_result,
        "jax_per_env_sps": float(jax_per_env_sps),
        "speedup_ratio_per_env": float(speedup),
        "total_jax_sps": float(jax_result["sps_mean"]),
        "total_python_sps": float(py_result["sps_mean"]),
    }

# All-task table
def measure_all_tasks(save: bool = True) -> dict[str, Any]:
    """Run throughput measurement for all 5 tasks and save a table.

    Output file: benchmarks/common/results/throughput_table.json
    """
    table = {}
    for task in FROZEN_NUM_ENVS:
        try:
            table[task] = measure_task(task)
        except Exception as exc:
            print(f"[throughput] {task} FAILED: {exc}")
            table[task] = {"task": task, "error": str(exc)}

    result = {
        "description": "JAX vmap+scan vs Python for-loop throughput comparison",
        "frozen_num_envs": FROZEN_NUM_ENVS,
        "tasks": table,
    }

    if save:
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = _RESULTS_DIR / f"throughput_table_{ts}.json"
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        # Overwrite canonical latest
        latest = _RESULTS_DIR / "throughput_table.json"
        latest.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[throughput] saved → {latest}")

    return result

def print_summary_table(result: dict[str, Any]) -> None:
    """Print a human-readable throughput comparison table."""
    print("\n" + "=" * 70)
    print("JAX Throughput Comparison (vmap+lax.scan vs Python for-loop)")
    print("=" * 70)
    print(f"{'Task':<14} {'n_envs':>7} {'JAX sps':>12} {'Py sps':>10} {'speedup':>9} {'compile':>9}")
    print("-" * 70)
    for task, res in result.get("tasks", {}).items():
        if "error" in res:
            print(f"{task:<14} ERROR: {res['error']}")
            continue
        jax_sps = res.get("total_jax_sps", 0)
        py_sps = res.get("total_python_sps", 0)
        ratio = res.get("speedup_ratio_per_env", 0)
        n = res.get("num_envs", 0)
        compile_s = res.get("jax_vmap_scan", {}).get("compile_time_s", 0)
        print(
            f"{task:<14} {n:>7d} {jax_sps:>12,.0f} {py_sps:>10,.0f} "
            f"{ratio:>8.1f}x {compile_s:>8.2f}s"
        )
    print("=" * 70)
    print("speedup = JAX-sps-per-env / Python-sps (single env, step-synced)")
    print()

def throughput_main(argv: "list[str] | None" = None) -> None:
    parser = argparse.ArgumentParser(description="JAX throughput benchmark")
    parser.add_argument("--task", choices=list(FROZEN_NUM_ENVS) + ["all"], default="all")
    parser.add_argument("--num-envs", type=int, default=None,
                        help="Override frozen num_envs (for exploration only)")
    parser.add_argument("--sweep", action="store_true",
                        help="Run num_envs scaling sweep")
    parser.add_argument("--all-tasks", action="store_true",
                        help="Run all tasks and save throughput_table.json")
    args = parser.parse_args(argv)

    if args.all_tasks or args.task == "all":
        result = measure_all_tasks(save=True)
        print_summary_table(result)
    elif args.sweep:
        result = measure_scaling_curve(args.task)
        print(json.dumps(result, indent=2))
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = _RESULTS_DIR / f"scaling_{args.task}.json"
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved -> {path}")
    else:
        n = args.num_envs or FROZEN_NUM_ENVS.get(args.task, 64)
        result = measure_task(args.task, num_envs=n)
        print(json.dumps(result, indent=2))

def main(argv: "list[str] | None" = None) -> None:
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        argv = ["plot"]
    cmd, sub_argv = argv[0], argv[1:]
    if cmd == "plot":
        plot_main(sub_argv)
        return
    if cmd == "throughput":
        throughput_main(sub_argv)
        return
    raise SystemExit(f"Unknown analysis command: {cmd}")

if __name__ == "__main__":
    main()
