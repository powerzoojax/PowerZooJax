"""PowerZooJax package entrypoint.

PowerZooJax is the benchmark-first JAX counterpart of PowerZoo for
power-system ML/RL research. It is built around a pure-functional env
contract so performance-critical environment transitions stay inside `jax.jit`:
state and PRNG are threaded explicitly through `reset` and `step`, batched execution is `vmap`-friendly, and fixed-length rollout
collection works with built-in auto-reset plus `lax.scan`.

Features:
- Benchmark-first JAX counterpart of PowerZoo for power-system ML/RL research, not a general simulator
- Pure-functional environment contract with explicit state and PRNG, designed for `jax.jit` and `jax.vmap`
- GPU-native rollout pipeline via built-in auto-reset, `batch_reset`/`batch_step`, and `lax.scan` trajectory collection
- Physics-faithful benchmark semantics with reward/cost separation so safety violations stay in the cost channel
- End-to-end benchmark surface covering grid, resource, market, and microgrid environments plus five-task recipes
- Native case library, time-series data pipeline, and CPU-side inspection tools for reproducible experiments

This top-level package is intentionally lightweight:
- sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` if the user has not set it
- exposes a small lazy-import surface for common case/env symbols
- avoids eagerly importing the full benchmark, RL, and data stacks

For most real work, import from subpackages directly:
- `powerzoojax.case` for built-in cases, PTDF, and inspection helpers
- `powerzoojax.envs` for pure-functional grid/resource/market/microgrid envs
- `powerzoojax.tasks` for the five benchmark task recipes: `tso`, `dso`,
  `ders`, `gencos`, and `dc_microgrid`
- `powerzoojax.data` for manifest-driven real/synthetic time-series loading
- `powerzoojax.rl` for wrappers, trainers, presets, and CLI-facing training
- `powerzoojax.utils` for `batch_reset`, `batch_step`, and `scan_rollout`

Quick start:
    >>> import powerzoojax  # sets XLA prealloc default before JAX warms GPU
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from powerzoojax.case import create_case5
    >>> from powerzoojax.envs import TransGridEnv, make_trans_params
    >>> from powerzoojax.utils import scan_rollout
    >>>
    >>> env = TransGridEnv()
    >>> params = make_trans_params(create_case5())
    >>> key = jax.random.PRNGKey(0)
    >>> obs, state = env.reset(key, params)
    >>> actions = jnp.zeros((48, *env.action_space(params).shape), dtype=jnp.float32)
    >>> final_state, obs_traj, reward_traj, done_traj, info_traj = scan_rollout(
    ...     env, key, state, params, actions
    ... )

CLI:
    >>> # List training presets
    >>> # python -m powerzoojax --list-presets
    >>>
    >>> # Train from a preset; --config is optional and overrides hyperparameters
    >>> # python -m powerzoojax --preset case5-economic-dispatch --seed 0
    >>> # python -m powerzoojax --preset case5-economic-dispatch --config experiment.yaml --output result.json

Package layout:
    powerzoojax/
    ├── case/       # Built-in cases, case registry, PTDF/matrices, CPU-side inspection
    ├── data/       # Manifest-driven time-series loading, alignment, splits, OOD transforms
    ├── envs/       # Pure-functional benchmark core: grid, resource, market, microgrid
    ├── tasks/      # Five-task benchmark recipes and task-specific metrics/baselines
    ├── rl/         # Training wrappers, multi-agent adapters, presets, trainers, CLI backend
    ├── utils/      # JAX rollout helpers and shared typing aliases
    ├── __main__.py # `python -m powerzoojax` training CLI
    └── __init__.py # This file: XLA default + lazy top-level re-exports
"""

from __future__ import annotations

import importlib
import os
from typing import Any

# GPU (XLA): JAX may preallocate ~75% of VRAM on first GPU use. If unset, prefer
# on-demand allocation to reduce OOMs when sharing the GPU. Respects a user-defined env.
# If `import jax` runs before this package initializes the GPU, the allocator is unchanged.
if "XLA_PYTHON_CLIENT_PREALLOCATE" not in os.environ:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

__version__ = "0.1.0"

# Lazy re-exports: ``import powerzoojax`` only sets XLA env + version — no case/env eager load.
_CASE_EXPORTS = (
    "CaseData",
    "create_case5",
    "create_case14",
    "create_case33bw",
    "load_case",
    "list_cases",
    "CaseInfo",
    "CasePlotter",
)
_ENV_EXPORTS = (
    "Environment",
    "EnvState",
    "EnvParams",
    "Space",
    "Box",
    "Discrete",
    "MultiDiscrete",
    "MultiBinary",
    "make_box",
    "make_discrete",
    "make_multi_discrete",
    "make_multi_binary",
)

__all__ = [*_CASE_EXPORTS, *_ENV_EXPORTS]


def __getattr__(name: str) -> Any:
    if name in _CASE_EXPORTS:
        case = importlib.import_module("powerzoojax.case")
        value = getattr(case, name)
        globals()[name] = value
        return value
    if name in _ENV_EXPORTS:
        envs = importlib.import_module("powerzoojax.envs")
        value = getattr(envs, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | {"__version__"})
