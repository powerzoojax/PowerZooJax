"""PowerZoo cross-backend bridge: mapping, driver, and dispatcher.

This keeps all PowerZoo-facing benchmark glue in one file so users only need
to scan a single entrypoint when they care about cross-backend comparisons.
"""

from __future__ import annotations

from typing import Any

JAX_TASK_TO_POWERZOO_TASK: dict[str, dict[str, Any]] = {
    "dso": {
        "powerzoo_task": "make_dso_env",
        "powerzoo_module": "powerzoo.tasks.dso_task",
        # Use the same Ausgrid feeder-shape data path as PowerZooJax DSO.
        # Synthetic fallback removed; train/eval/baselines raise instead of
        # silently switching data source.
        "powerzoo_factory_kwargs": {},
        "framework": "auto",
        "access_path": "direct_env",
        "cross_backend_comparable": True,
        "rl_paradigm": "single_agent_non_stationary",
        "n_agents": 1,
        # Cross-backend data: identical Ausgrid windows on both sides
        # (powerzoojax/data/splits.py is the single source of truth;
        # alignment locked by tests/benchmarks/test_split_alignment.py).
        "data_window": (
            "Ausgrid FY25 feeder shapes with the current formal DSO "
            "eval_splits limited to iid."
        ),
        "notes": (
            "PowerZoo make_dso_env(split=...) returns a single-agent "
            "gymnasium.Env (Box(221,) obs, Box(12,) action; FlattenWrapper + "
            "DSOCostWrapper).  Same case33bw + 6 FlexLoad + Ausgrid real "
            "feeder shapes as the DSO benchmark page in docs/en/benchmarks.  PowerZoo "
            "data_loader.py:890 was patched 2026-04 with numeric_only=True "
            "for pandas >=2.0.  Current formal cross-backend DSO records "
            "train internally on train_split=train and report eval_split=iid; "
            "historical OOD split names are not part of the frozen executable "
            "DSO task config."
        ),
    },
    "tso": {
        "powerzoo_task": "CentralizedComparisonTSOEnv",
        "powerzoo_module": "powerzoo.tasks.middle.comparison_tso",
        "powerzoo_factory_kwargs": {},
        "framework": "auto",
        "access_path": "direct_env",
        "cross_backend_comparable": True,
        "rl_paradigm": "single_agent_safe_rl_hybrid",
        "n_agents": 1,
        # Cross-backend data: GB demand actuals + GB gen-by-type, sliced
        # by the same (split, episode_start_idx) on both sides.  The sin-
        # wave synthetic anchor is retained as a test-only helper.
        "data_window": "GB 2025-04..2025-12 (train) / 2026-01..2026-03 (iid)",
        "notes": (
            "PowerZoo CentralizedComparisonTSOEnv(CentralizedComparisonTSOTask("
            "split=..., episode_start_idx=...)) wraps the case118 UC MARL "
            "env into a single-agent gymnasium.Env (Box(249,) obs, Box(108,) "
            "action: commit_intent[54] + dispatch[54]).  Reward signal is "
            "``-1e-4 * (gen_cost + startup_cost + no_load_cost)`` on BOTH "
            "backends (matches PowerZooJax reward_scale, parity locked by "
            "tests/benchmarks/test_tso_reward_parity.py).  Irreducible "
            "implementation gaps documented in paper supplementary: "
            "obs_shape (JAX 410 vs Py 249) and dispatch solver (JAX DC-OPF "
            "vs Py score-based allocation)."
        ),
    },
    "ders": {
        "powerzoo_task": "marl_ders_benchmark",
        "powerzoo_module": "powerzoo.tasks.registry",
        # voltage_penalty=4.0 mirrors the PowerZooJax DERs frozen benchmark config
        # in benchmarks/ders/configs/train_ippo.yaml.  Cross-backend reward
        # magnitudes are only comparable when both sides use this weight;
        # the alignment is enforced by tests/benchmarks/test_ders_voltage_penalty_parity.py.
        "powerzoo_factory_kwargs": {"case": "Case141", "voltage_penalty": 4.0},
        "framework": "pettingzoo",
        "access_path": "pettingzoo_il",
        "cross_backend_comparable": True,
        "rl_paradigm": "scalable_safe_marl_cooperative",
        "n_agents": 12,
        "notes": (
            "PowerZoo marl_ders_benchmark = 12-agent PettingZoo cooperative "
            "task (Battery + PV + FlexLoad on case141, reward_type='shared' "
            "team reward).  Mirrors PowerZooJax DERs DistGridMARLEnv (12 "
            "agents, shared team reward via voltage_penalty).  voltage_penalty "
            "is set to 4.0 on both backends to match the frozen DERs benchmark "
            "configuration.  Cross-backend SB3 baseline goes through "
            "frozen-self-play IL: one SB3 model per agent, opponents are the "
            "previous round's frozen policies (random in round 0).  This is "
            "still a strictly weaker baseline than parameter-shared IPPO "
            "because SB3 has no native cooperative MARL trainer; this is an "
            "irreducible cross-library gap for SB3-style baselines."
        ),
    },
    "gencos": {
        "powerzoo_task": "gencos_bidding",
        "powerzoo_module": "powerzoo.tasks.registry",
        "powerzoo_factory_kwargs": {},
        "framework": "pettingzoo",
        "access_path": "pettingzoo_il",
        "cross_backend_comparable": True,
        "rl_paradigm": "multi_agent_competitive",
        "n_agents": 5,
        # Cross-backend data: GB demand actuals + GB gen-by-type, sliced
        # by ``benchmarks.gencos.train::data_split``.  No synthetic fallback.
        "data_window": "GB 2025-04..2025-12 (train) / 2026-01..2026-03 (iid)",
        "notes": (
            "PowerZoo gencos_bidding = 5-agent PettingZoo competitive task "
            "(case5 dispatch_profit per-agent reward, see task_gencos.py).  "
            "Mirrors PowerZooJax MarketMARLEnv (5 GenCo agents, private "
            "profit reward, partial observability).  The GenCos benchmark "
            "page marks it as the suite's competitive MARL task.  SB3 cross-"
            "backend baseline runs through ``frozen self-play IL``: one "
            "SB3 model per agent, opponents replaced each round by the "
            "previous round's frozen policies (round 0 opponents are "
            "random because no policies exist yet).  Default n_rounds=4, "
            "per_agent budget configurable via "
            "``extra_config['per_agent_steps_per_round']``.  This is "
            "still a strictly weaker baseline than parameter-shared IPPO "
            "because SB3 has no native PSRO / best-response trainer; the "
            "gap is irreducible at the library level and is documented in "
            "the paper supplementary."
        ),
    },
    "dc_microgrid": {
        "powerzoo_task": "dc_microgrid",
        "powerzoo_module": "powerzoo.tasks.registry",
        "powerzoo_factory_kwargs": {},
        "framework": "auto",
        "access_path": "pz_registry",
        "cross_backend_comparable": True,
        "rl_paradigm": "single_agent_multi_objective_robust",
        "n_agents": 1,
        # DC Microgrid now bypasses PowerZoo's task defaults entirely:
        # the bridge rebuilds the Python env from the JAX benchmark's
        # episode params so data source / split / OOD / case overrides /
        # reward shaping stay aligned with Phase-1.
        "data_window": "Google/Azure/Alibaba trace windows reconstructed from JAX benchmark episode params",
        "notes": (
            "Cross-backend DC Microgrid no longer relies on PowerZoo task "
            "defaults (which use synthetic workload + legacy split semantics). "
            "The bridge reconstructs PowerZoo's Python DCMicrogridEnv from "
            "PowerZooJax DCMicrogridTask.episode_params(...) so real-data "
            "workload windows, OOD scenarios, case overrides, 24-D obs parity, "
            "dg_autobalance, and reward-shaping terms match the frozen "
            "benchmark task."
        ),
    },
}

def is_comparable(jax_task: str) -> bool:
    """True only if comparable explicitly equals True (unknown/False both fail)."""
    entry = JAX_TASK_TO_POWERZOO_TASK.get(jax_task)
    if entry is None:
        return False
    return entry.get("cross_backend_comparable") is True

import argparse
import copy
import importlib
import inspect
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

from benchmarks.common.artifacts import save_eval_artifacts, save_training_artifacts
from benchmarks.common.configs import dump_yaml, load_config, load_task_config, load_train_config
from benchmarks.common.io import (
    CANONICAL_BACKENDS,
    RunRecord,
    collect_dataset_provenance,
    collect_env_info,
    config_hash,
    make_run_id,
    merge_env_info_metadata,
    normalize_device_name,
    save_run,
)
from benchmarks.common.powerzoo_repo import ensure_powerzoo_on_path, find_powerzoo_repo
from benchmarks.common.runtime import build_train_cfg
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BENCHMARKS_DIR = _REPO_ROOT / "benchmarks"
_POWERZOO_PATH = find_powerzoo_repo(_REPO_ROOT)

_repo_root_str = str(_REPO_ROOT)
_pythonpath_parts = [
    part for part in os.environ.get("PYTHONPATH", "").split(os.pathsep) if part
]
if _repo_root_str not in _pythonpath_parts:
    os.environ["PYTHONPATH"] = os.pathsep.join([_repo_root_str, *_pythonpath_parts])


def _ensure_repo_root_first() -> None:
    """Keep this repo's ``benchmarks`` package ahead of PowerZoo's package."""
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _REPO_ROOT]
    sys.path.insert(0, _repo_root_str)


_ensure_repo_root_first()


def _ensure_powerzoo_path() -> None:
    """Make ``import powerzoo`` work without touching the editable install path.

    The system pip-editable install may point to a stale path; we side-step by
    prepending the actual on-disk PowerZoo location into ``sys.path`` and
    dropping any already-imported ``powerzoo.*`` modules that came from a
    different checkout. Idempotent.
    """
    path = ensure_powerzoo_on_path(_REPO_ROOT, append=False)
    if path is None:
        raise FileNotFoundError(
            "Could not locate the PowerZoo repo. Set POWERZOO_DIR or place a "
            "sibling checkout at ../PowerZoo or ../PowerZoo.DEL."
        )
    path = path.resolve()
    _ensure_repo_root_first()
    stale_modules: list[str] = []
    for name, module in list(sys.modules.items()):
        if not (name == "powerzoo" or name.startswith("powerzoo.")):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        try:
            module_path = Path(module_file).resolve()
        except Exception:
            continue
        if path not in module_path.parents and module_path != path:
            stale_modules.append(name)
    for name in stale_modules:
        sys.modules.pop(name, None)

class CrossBackendNotComparable(RuntimeError):
    """Raised when a task is not flagged ``cross_backend_comparable=True``.

    The dispatcher should catch this and skip + log a WARNING; do NOT silently
    record sb3/sbx numbers for unalignable tasks.
    """

# Mapping resolution

def _resolve_task(jax_task: str, *, allow_unknown: bool = False) -> dict[str, Any]:
    """Return the task_mapping entry, enforcing ``cross_backend_comparable``.

    ``allow_unknown=True`` skips that check (dry-run only) so ``unknown``
    tasks can be probed before promoting them to ``True``.
    """
    if jax_task not in JAX_TASK_TO_POWERZOO_TASK:
        raise KeyError(f"Unknown jax_task: {jax_task!r}")
    entry = JAX_TASK_TO_POWERZOO_TASK[jax_task]
    if entry["powerzoo_task"] in (None, "TBD", ""):
        raise CrossBackendNotComparable(
            f"jax_task={jax_task!r} has no PowerZoo counterpart yet "
            f"(powerzoo_task={entry['powerzoo_task']!r}). "
            f"Promote in benchmarks/common/powerzoo_bridge.py after dry-run."
        )
    if not allow_unknown and entry["cross_backend_comparable"] is not True:
        raise CrossBackendNotComparable(
            f"jax_task={jax_task!r} cross_backend_comparable="
            f"{entry['cross_backend_comparable']!r}; refusing to record "
            f"cross-backend numbers. Notes: {entry.get('notes', '')}"
        )
    return entry

# IL wrapper for PettingZoo cooperative / competitive multi-agent tasks

def _agent_obs_space(marl_env, agent_id: str):
    """Get one agent's observation space, handling both PowerZoo conventions.

    PowerZoo's MultiAgentEnv subclasses are inconsistent:
    - DERs (DistributionMARLEnv): ``observation_space`` is a dict / Dict
      space; subscript with ``agent_id``.
    - GenCos (gencos_marl): ``observation_space`` is a method; call with
      ``agent_id``.
    We accept both.
    """
    sp = getattr(marl_env, "observation_space", None)
    if callable(sp):
        return sp(agent_id)
    if hasattr(sp, "__getitem__"):
        return sp[agent_id]
    # Last-resort: try the .observation_spaces (plural) dict attribute
    spaces_dict = getattr(marl_env, "observation_spaces", None)
    if spaces_dict is not None:
        return spaces_dict[agent_id]
    raise AttributeError(
        f"Cannot resolve observation_space for agent {agent_id!r} on "
        f"{type(marl_env).__name__}: not callable, not subscriptable, "
        f"no observation_spaces dict."
    )

def _agent_action_space(marl_env, agent_id: str):
    """Same as ``_agent_obs_space`` but for action spaces."""
    sp = getattr(marl_env, "action_space", None)
    if callable(sp):
        return sp(agent_id)
    if hasattr(sp, "__getitem__"):
        return sp[agent_id]
    spaces_dict = getattr(marl_env, "action_spaces", None)
    if spaces_dict is not None:
        return spaces_dict[agent_id]
    raise AttributeError(
        f"Cannot resolve action_space for agent {agent_id!r} on "
        f"{type(marl_env).__name__}"
    )

def _make_il_env(
    marl_env,
    agent_id: str,
    frozen_opponents: "dict[str, Any] | None" = None,
):
    """Wrap a PettingZoo ParallelEnv as a single-agent gymnasium.Env.

    During ``step()``, ``agent_id`` receives the SB3 action and every
    other agent acts via ``frozen_opponents[other_id]`` if available
    (a callable ``policy(obs) -> action``); otherwise samples uniformly
    from its action space (random opponent fallback).

    Frozen-opponent self-play is the canonical IL upgrade over naïve
    random opponents:

    - For competitive tasks (GenCos): random opponents do NOT exhibit
      market power — they bid uniformly across the offer band, so the
      training agent learns against a "no-strategy" market and never
      sees the price-shading or markup behaviour real GenCos would use.
      Frozen self-play (each round, freeze the previous-iteration
      policies of opponents) approximates fictitious play / best
      response and surfaces oligopolistic dynamics that random play
      hides. This is the usual IL baseline for competitive MARL.
    - For cooperative tasks (DERs, shared team reward): random
      opponents inject high-variance noise into the team reward seen
      by the training agent, drowning out its own learning signal.
      Frozen self-play makes the team reward more stable round-over-
      round and lets each agent's policy actually contribute.

    Implementation note: the ``frozen_opponents`` dict can hold either
    SB3 model objects (callable via ``model.predict(obs, deterministic=True)``)
    or any callable ``policy(obs) -> action`` (e.g. lambdas wrapping a
    JAX policy). The wrapper detects which interface is present.
    """
    import gymnasium as gym

    frozen_opponents = frozen_opponents or {}

    def _act_with_opponent(other_id: str, obs):
        """Resolve an opponent's action: frozen policy if present, else random."""
        op = frozen_opponents.get(other_id)
        if op is None:
            return _agent_action_space(marl_env, other_id).sample()
        # SB3-style: model.predict(obs, deterministic=True) -> (action, state)
        if hasattr(op, "predict"):
            action, _ = op.predict(obs, deterministic=True)
            return action
        # Generic callable policy(obs) -> action
        if callable(op):
            return op(obs)
        return _agent_action_space(marl_env, other_id).sample()

    class _PettingZooILEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self._env = marl_env
            self._target = agent_id
            self.observation_space = _agent_obs_space(marl_env, agent_id)
            self.action_space = _agent_action_space(marl_env, agent_id)
            self._others = [a for a in marl_env.possible_agents if a != agent_id]
            self._last_obs_dict: dict[str, Any] = {}
            self._last_target_obs: Any | None = None

        def _zero_obs(self) -> Any:
            """Return a zero-shaped observation matching ``observation_space``."""
            shape = getattr(self.observation_space, "shape", None)
            if shape is not None:
                return np.zeros(shape, dtype=np.float32)
            # Dict / Tuple obs spaces: return a sampled-then-zeroed instance
            sample = self.observation_space.sample()
            try:
                if isinstance(sample, dict):
                    return {k: np.zeros_like(v) for k, v in sample.items()}
                return np.zeros_like(sample)
            except Exception:
                return sample

        def reset(self, *, seed=None, options=None):
            obs_dict, info_dict = self._env.reset(seed=seed, options=options)
            self._last_obs_dict = obs_dict
            target_obs = obs_dict[self._target]
            self._last_target_obs = target_obs
            return target_obs, info_dict.get(self._target, {})

        def _episode_done(self, obs_d, term_d, trunc_d) -> tuple[bool, bool]:
            """Return ``(terminated, truncated)`` with PettingZoo semantics.

            Mid-episode all-done detection is split into two:
            - ``terminated``: triggered by physical termination (term_d's
              ``__all__`` key, or every per-agent term flag set).
            - ``truncated``: episode-end via time-limit or the wrapper's
              ``__all__`` truncation flag.
            Mapping all-done to ``terminated=True`` indiscriminately (as
            the previous implementation did) breaks SB3's bootstrap-value
            semantics for time-limit truncations.
            """
            term_all = bool(term_d.get("__all__", False))
            trunc_all = bool(trunc_d.get("__all__", False))
            if not obs_d:
                # PettingZoo Parallel API contract: empty obs dict signals
                # end-of-episode.  Default to truncated=True (time limit)
                # unless the upstream env explicitly flagged a physical
                # termination.
                return term_all, (trunc_all or not term_all)
            empty_or_all_term = (
                len(obs_d) == 0
                or term_all
                or (term_d and all(bool(v) for v in term_d.values() if v is not None))
            )
            empty_or_all_trunc = (
                trunc_all
                or (trunc_d and all(bool(v) for v in trunc_d.values() if v is not None))
            )
            return bool(empty_or_all_term), bool(empty_or_all_trunc)

        def step(self, action):
            actions: dict[str, Any] = {}
            for a in self._others:
                obs_a = self._last_obs_dict.get(a)
                if obs_a is None:
                    actions[a] = _agent_action_space(marl_env, a).sample()
                else:
                    actions[a] = _act_with_opponent(a, obs_a)
            actions[self._target] = action
            obs_d, rew_d, term_d, trunc_d, info_d = self._env.step(actions)
            terminated, truncated = self._episode_done(obs_d, term_d, trunc_d)
            if not obs_d or self._target not in obs_d:
                # End of episode: return the last observation we held for
                # the target so SB3's value bootstrap sees a meaningful
                # final state, paired with the right terminated/truncated
                # flags decoded above.
                return (
                    self._last_target_obs
                    if self._last_target_obs is not None
                    else self._zero_obs(),
                    float(rew_d.get(self._target, 0.0)),
                    bool(terminated),
                    bool(truncated),
                    info_d.get(self._target, {}) if info_d else {},
                )
            self._last_obs_dict = obs_d
            self._last_target_obs = obs_d[self._target]
            return (
                obs_d[self._target],
                float(rew_d.get(self._target, 0.0)),
                bool(term_d.get(self._target, False) or terminated),
                bool(trunc_d.get(self._target, False) or truncated),
                info_d.get(self._target, {}),
            )

        def close(self):
            pass

    raw_env = _PettingZooILEnv()
    # Wrap in Monitor so SB3's ep_info_buffer (used by
    # _extract_ep_rew_mean) actually populates with episode totals.
    # Without Monitor the buffer stays empty and final_return ends up
    # ``None`` for every IL run.
    try:
        from stable_baselines3.common.monitor import Monitor
        return Monitor(raw_env)
    except Exception:
        # Fall back to gymnasium's RecordEpisodeStatistics, which writes
        # the ``episode`` key into info dicts in a way SB3 also picks up.
        try:
            from gymnasium.wrappers import RecordEpisodeStatistics
            return RecordEpisodeStatistics(raw_env)
        except Exception:
            return raw_env

# Build env for a given (jax_task, access_path)

# Splits accepted by each PowerZoo factory in the cross-backend dispatcher.
# Mirrors the eval_splits enumerated in each task config; any new split must
# be added here or the driver will refuse to instantiate instead of silently
# claiming that it evaluated a different window.
_VALID_SPLITS_BY_TASK: dict[str, tuple[str, ...]] = {
    "dso": ("train", "iid"),
    "tso": ("train", "iid", "load_stress", "line_tightening"),
    "ders": ("train", "iid", "voltage_tightening", "pv_penetration_shift", "load_stress"),
    "gencos": ("train", "iid", "demand_shift", "renewable_shock"),
    "dc_microgrid": (
        "train", "iid", "cooling_stress", "renewable_drought",
        "workload_swap", "workload_shock", "dg_derating", "sla_tighten",
    ),
}

def _powerzoo_reward_shaping_weights(jax_task: str) -> dict[str, float]:
    """Return cross-backend reward-shaping weights for tasks that use them."""
    if jax_task != "dc_microgrid":
        return {}
    cfg = load_task_config(_BENCHMARKS_DIR / jax_task)
    weights = cfg.get("reward_shaping_weights") or {}
    return {
        "sla": float(weights.get("sla", 0.0)),
        "overtemp": float(weights.get("overtemp", 0.0)),
        "power_deficit": float(weights.get("power_deficit", 0.0)),
        "power_spill": float(weights.get("power_spill", 0.0)),
        "power_balance": float(weights.get("power_balance", 0.0)),
        "dispatch_tracking": float(weights.get("dispatch_tracking", 0.0)),
    }

def _dc_task_config() -> dict[str, Any]:
    return load_task_config(_BENCHMARKS_DIR / "dc_microgrid")


def _ders_task_config() -> dict[str, Any]:
    return load_task_config(_BENCHMARKS_DIR / "ders")


def _tso_powerzoo_task_kwargs(
    split: str,
    *,
    episode_start_idx: int | None = None,
) -> tuple[dict[str, Any], tuple[str, str]]:
    """Return PowerZoo TSO task kwargs aligned to the frozen benchmark window."""
    from benchmarks.tso.config_runtime import get_eval_gb_split, resolve_gb_windows

    task_cfg = load_task_config(_BENCHMARKS_DIR / "tso")
    gb_split = get_eval_gb_split(split)
    window = resolve_gb_windows(task_cfg)[gb_split]
    kwargs: dict[str, Any] = {
        "split": split,
        "start_date": window[0],
        "end_date": window[1],
    }
    if episode_start_idx is not None:
        kwargs["episode_start_idx"] = int(episode_start_idx)
    return kwargs, window


def _annotate_tso_powerzoo_env(
    env: Any,
    *,
    split: str,
    profile_window: tuple[str, str],
) -> None:
    """Attach benchmark-facing provenance metadata to the PowerZoo TSO env."""
    env.data_source = "gb_real"
    env.load_profile_source = "gb_real"
    env.benchmark_split = str(split)
    env.ood_axis = None if split in ("train", "iid") else str(split)
    env.profile_window = tuple(profile_window)


def _instantiate_powerzoo_tso_task(
    split: str,
    *,
    episode_start_idx: int | None = None,
):
    """Build the PowerZoo TSO comparison task across old/new repo signatures."""
    from powerzoo.tasks.middle.comparison_tso import CentralizedComparisonTSOTask

    task_kwargs, profile_window = _tso_powerzoo_task_kwargs(
        split,
        episode_start_idx=episode_start_idx,
    )
    sig = inspect.signature(CentralizedComparisonTSOTask.__init__)
    params = sig.parameters

    if "episode_start_idx" in params:
        modern_kwargs: dict[str, Any] = {"split": split}
        if episode_start_idx is not None:
            modern_kwargs["episode_start_idx"] = int(episode_start_idx)
        return CentralizedComparisonTSOTask(**modern_kwargs), profile_window

    return CentralizedComparisonTSOTask(**task_kwargs), profile_window


def _powerzoo_artifact_rel(artifacts_dir: Path, path: Path) -> str:
    """Return a results-relative artifact path for nested artifact bundles."""
    return str(path.relative_to(artifacts_dir.parent)).replace("\\", "/")


def _cross_backend_record_split(
    jax_task: str,
    *,
    requested_split: str,
    train_split: str,
    env_kind: str,
) -> str:
    """Return the honest split label for the training-class RunRecord.

    DERs / GenCos phase-2 train on the canonical ``train`` split even when
    the CLI request names an eval/report split such as ``iid``.  Keeping the
    training-class record on ``split=train`` avoids manifest collisions with
    the official eval records that are written after training.
    """
    if jax_task in {"ders", "gencos"} and env_kind == "pettingzoo":
        return str(train_split)
    return str(requested_split)


def _build_jax_ders_task(*, voltage_penalty: float):
    """Exact JAX-side DER task used as the cross-backend scenario oracle."""
    from powerzoojax.case import load_case
    from powerzoojax.tasks.ders import DERsTask

    task_cfg = _ders_task_config()
    return DERsTask(
        case=load_case(str(task_cfg.get("case", "case141"))),
        v_min=float(task_cfg.get("v_min", 0.94)),
        v_max=float(task_cfg.get("v_max", 1.06)),
        voltage_penalty=float(voltage_penalty),
        max_steps=int(task_cfg.get("max_steps", 48)),
    )


def _build_ders_train_params_bank() -> tuple[tuple[Any, ...], tuple[int, ...]]:
    """Reuse the canonical Phase-1 train-window bank logic for Phase-2."""
    from benchmarks.ders.train import _build_train_params_bank
    from powerzoojax.case import load_case

    task_cfg = _ders_task_config()
    case = load_case(str(task_cfg.get("case", "case141")))
    params_bank, starts, _meta = _build_train_params_bank(
        case=case,
        task_config=task_cfg,
        max_steps=int(task_cfg.get("max_steps", 48)),
    )
    return tuple(params_bank), tuple(int(s) for s in starts)


def _ders_pv_profiles_from_params(params) -> np.ndarray:
    from powerzoojax.envs.resource.renewable import RenewableBundle

    for bundle in params.resources:
        if isinstance(bundle, RenewableBundle):
            profiles = np.asarray(bundle.profiles, dtype=np.float32)
            if profiles.ndim != 2:
                raise ValueError(
                    "DER renewable profiles must be a 2-D (T, n_pv) array for "
                    "cross-backend episode injection."
                )
            return profiles
    raise ValueError("DER params contain no RenewableBundle to inject into PowerZoo.")


def _apply_ders_episode_params(marl_env, params) -> None:
    """Mutate the Python DER env so the next reset matches one exact JAX episode."""
    from powerzoo.tasks.simple.marl_ders_benchmark import (
        inject_load_profiles,
        inject_pv_profiles,
    )

    load_scale = float(getattr(params.case, "base_mva", 1.0))
    load_profiles_p = np.asarray(params.load_profiles_p, dtype=np.float32) * load_scale
    load_profiles_q = np.asarray(params.load_profiles_q, dtype=np.float32) * load_scale
    inject_load_profiles(marl_env, load_profiles_p, load_profiles_q)
    inject_pv_profiles(marl_env, _ders_pv_profiles_from_params(params))

    grid = marl_env.base_env.grid
    grid.v_min = float(params.v_min)
    grid.v_max = float(params.v_max)
    marl_env._v_min_info = float(params.v_min)
    marl_env._v_max_info = float(params.v_max)
    marl_env._scenario_config.setdefault("grid", {})
    marl_env._scenario_config["grid"]["v_min"] = float(params.v_min)
    marl_env._scenario_config["grid"]["v_max"] = float(params.v_max)


def _compute_ders_continuous_cost_from_state(
    state: dict[str, Any],
    *,
    v_min: float,
    v_max: float,
) -> tuple[float, float, float]:
    """Mirror JAX DER ``cost_continuous`` from PowerZoo's solved grid state."""
    nodes = state.get("nodes")
    lines = state.get("lines")
    if nodes is None:
        return 0.0, float(v_min), float(v_max)

    v_mag = np.asarray(nodes["v_mag"], dtype=np.float64)
    v_under = np.maximum(float(v_min) - v_mag, 0.0)
    v_over = np.maximum(v_mag - float(v_max), 0.0)

    s_over = 0.0
    if lines is not None and {"p_flow_MW", "q_flow_MVAr"}.issubset(lines.columns):
        p_flow = np.asarray(lines["p_flow_MW"], dtype=np.float64)
        q_flow = np.asarray(lines["q_flow_MVAr"], dtype=np.float64)
        if "cap" in lines.columns:
            cap = np.asarray(lines["cap"], dtype=np.float64)
        elif "rateA" in lines.columns:
            cap = np.asarray(lines["rateA"], dtype=np.float64)
        else:
            cap = np.full_like(p_flow, np.inf)
        s_flow = np.sqrt(p_flow * p_flow + q_flow * q_flow)
        s_over = float(np.sum(np.maximum(s_flow - cap, 0.0)))

    return (
        float(np.sum(v_under) + np.sum(v_over) + s_over),
        float(np.min(v_mag)),
        float(np.max(v_mag)),
    )


class _PowerZooDERSExactEpisodeWrapper:
    """Inject exact JAX DER episodes and reward shaping into the PowerZoo env."""

    def __init__(
        self,
        inner,
        *,
        split: str,
        seed: int,
        episode_idx: int,
        n_episodes: int,
        strategy: str,
        voltage_penalty: float,
    ):
        self._inner = inner
        self.base_env = getattr(inner, "base_env", None)
        if self.base_env is None:
            raise ValueError("DER cross-backend wrapper requires a TaskResourceMultiAgentEnv.")
        self._split = str(split)
        self._default_seed = int(seed)
        self._default_episode_idx = int(episode_idx)
        self._default_n_episodes = max(int(n_episodes), 1)
        self._default_strategy = str(strategy)
        self._voltage_penalty = float(voltage_penalty)
        self._task = _build_jax_ders_task(voltage_penalty=float(voltage_penalty))
        self._reset_counter = 0
        self._last_episode_start = 0
        self._last_split = str(split)
        self._last_bank_index: int | None = None
        self._train_params_bank: tuple[Any, ...] = ()
        self._train_window_starts: tuple[int, ...] = ()
        if self._split == "train":
            self._train_params_bank, self._train_window_starts = _build_ders_train_params_bank()

    @property
    def possible_agents(self):
        return self._inner.possible_agents

    def observation_space(self, agent: str | None = None):
        sp = getattr(self._inner, "observation_space")
        if callable(sp):
            target = agent if agent is not None else self.possible_agents[0]
            return sp(target)
        return sp if agent is None else sp[agent]

    def action_space(self, agent: str | None = None):
        sp = getattr(self._inner, "action_space")
        if callable(sp):
            target = agent if agent is not None else self.possible_agents[0]
            return sp(target)
        return sp if agent is None else sp[agent]

    def _sample_train_bank_idx(self, seed: int) -> int:
        if not self._train_params_bank:
            return 0
        rng = np.random.default_rng(int(seed) * 9973 + self._reset_counter * 131 + 17)
        return int(rng.integers(0, len(self._train_params_bank)))

    def reset(self, *, seed=None, options=None):
        opts = dict(options or {})
        split = str(opts.pop("split", self._split))
        seed_value = int(self._default_seed + self._reset_counter if seed is None else seed)
        explicit_start = opts.pop("episode_start", None)
        use_train_bank = bool(
            opts.pop("use_train_bank", split == "train" and bool(self._train_params_bank))
        )

        if explicit_start is not None:
            episode_start = int(explicit_start)
            params = self._task.params_from_start(split, episode_start)
            self._last_bank_index = None
        elif use_train_bank and self._train_params_bank:
            bank_idx = int(opts.pop("train_bank_index", self._sample_train_bank_idx(seed_value)))
            bank_idx = max(0, min(bank_idx, len(self._train_params_bank) - 1))
            params = self._train_params_bank[bank_idx]
            episode_start = int(self._train_window_starts[bank_idx])
            self._last_bank_index = bank_idx
        else:
            episode_idx = int(opts.pop("episode_idx", self._default_episode_idx))
            n_episodes = int(opts.pop("n_episodes", self._default_n_episodes))
            strategy = str(opts.pop("strategy", self._default_strategy))
            episode_seed = int(opts.pop("episode_seed", seed_value))
            episode_start = self._task.episode_start(
                split,
                episode_idx,
                n_episodes,
                strategy=strategy,
                seed=episode_seed,
            )
            params = self._task.params_from_start(split, episode_start)
            self._last_bank_index = None

        _apply_ders_episode_params(self._inner, params)
        self._last_episode_start = int(episode_start)
        self._last_split = split
        inner_options = dict(opts)
        inner_options["day_id"] = 0
        obs, infos = self._inner.reset(seed=seed_value, options=inner_options)
        self._reset_counter += 1
        enriched_infos = {
            agent: {
                **dict(infos.get(agent, {})),
                "episode_start": float(self._last_episode_start),
                "split": self._last_split,
            }
            for agent in self.possible_agents
        }
        return obs, enriched_infos

    def step(self, action_dict):
        obs, rewards, terminateds, truncateds, infos = self._inner.step(action_dict)
        current_state = dict(getattr(self.base_env, "_current_state", {}) or {})
        current_info = dict(getattr(self.base_env, "_current_info", {}) or {})
        grid = self.base_env.grid
        base_reward = float(next(iter(rewards.values()))) if rewards else 0.0
        cost_continuous, v_min_step, v_max_step = _compute_ders_continuous_cost_from_state(
            current_state,
            v_min=float(grid.v_min),
            v_max=float(grid.v_max),
        )
        shaped_reward = base_reward - self._voltage_penalty * cost_continuous
        p_loss_mw = float(current_state.get("p_loss_MW", current_info.get("p_loss_MW", 0.0)))
        cost_voltage = float(current_info.get("cost_voltage_violation", 0.0))
        cost_thermal = float(current_info.get("cost_thermal_overload", 0.0))
        cost_resource = float(current_info.get("cost_resource", 0.0))
        cost_sum = float(current_info.get("cost_sum", cost_voltage + cost_thermal + cost_resource))

        shaped_rewards = {agent: float(shaped_reward) for agent in self.possible_agents}
        enriched_infos: dict[str, dict[str, Any]] = {}
        for agent in self.possible_agents:
            info = dict(infos.get(agent, {}))
            info.update(
                {
                    "reward": float(shaped_reward),
                    "raw_reward": float(base_reward),
                    "p_loss_MW": p_loss_mw,
                    "cost_continuous": float(cost_continuous),
                    "cost_sum": cost_sum,
                    "v_min_step": float(v_min_step),
                    "v_max_step": float(v_max_step),
                    "n_violations": float(cost_voltage + cost_thermal),
                    "episode_start": float(self._last_episode_start),
                    "split": self._last_split,
                }
            )
            enriched_infos[agent] = info
        return obs, shaped_rewards, terminateds, truncateds, enriched_infos

    def close(self):
        close_fn = getattr(self._inner, "close", None)
        if callable(close_fn):
            close_fn()

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


def _dso_task_config() -> dict[str, Any]:
    return load_task_config(_BENCHMARKS_DIR / "dso")


def _resolved_single_agent_n_envs(
    jax_task: str,
    algorithm: str,
    n_envs: int | None,
) -> int:
    """Resolve VecEnv parallelism from the frozen benchmark config when omitted."""
    if n_envs is not None and int(n_envs) > 0:
        return int(n_envs)
    try:
        _cfg_path, cfg = _load_benchmark_train_config(jax_task, algorithm)
        if cfg.get("num_envs") is not None:
            return int(cfg["num_envs"])
    except Exception:
        pass
    try:
        task_cfg = load_task_config(_BENCHMARKS_DIR / jax_task)
        if task_cfg.get("num_envs") is not None:
            return int(task_cfg["num_envs"])
    except Exception:
        pass
    return 1


def _powerzoo_dso_env_kwargs(
    split: str,
    *,
    seed: int,
    task_cfg: dict[str, Any] | None = None,
    use_train_reset_bank: bool,
) -> dict[str, Any]:
    """Translate the frozen PowerZooJax DSO task config into PowerZoo kwargs."""
    cfg = dict(_dso_task_config() if task_cfg is None else task_cfg)
    kwargs: dict[str, Any] = {
        "split": split,
        "max_steps": int(cfg.get("max_steps", 48)),
        "delta_t_minutes": float(cfg.get("dt_hours", 0.5)) * 60.0,
        "load_scale": float(cfg.get("load_scale", 1.0)),
        "v_slack": float(cfg.get("v_slack", 1.0)),
        "v_min": float(cfg.get("v_min", 0.94)),
        "v_max": float(cfg.get("v_max", 1.06)),
        "shift_horizon": int(cfg.get("shift_horizon", 4)),
        "preserve_feeder_totals": bool(cfg.get("preserve_feeder_totals", False)),
    }
    if cfg.get("flexload_config") is not None:
        kwargs["flexload_config"] = [
            {
                "name": str(item.get("name", f"fl_{idx}")),
                "bus_id": int(item["bus_id"]),
                "curtail_cap_mw": float(item["curtail_cap_mw"]),
                "shift_cap_mw": float(item["shift_cap_mw"]),
            }
            for idx, item in enumerate(cfg["flexload_config"])
        ]
    if cfg.get("bus_load_scale_overrides") is not None:
        kwargs["bus_load_scale_overrides"] = cfg["bus_load_scale_overrides"]
    if (
        use_train_reset_bank
        and split == "train"
        and str(cfg.get("train_window_sampling", "concat")) == "reset_bank"
        and cfg.get("train_window_starts")
    ):
        kwargs["reset_episode_starts"] = [int(x) for x in cfg["train_window_starts"]]
        kwargs["reset_sampling"] = "random"
        kwargs["reset_seed"] = int(seed)
    return kwargs


def _dso_eval_episode_starts(
    split: str,
    *,
    n_episodes: int,
    max_steps: int,
    task_cfg: dict[str, Any] | None = None,
) -> list[int]:
    """Episode starts matching the frozen DSO benchmark protocol."""
    cfg = dict(_dso_task_config() if task_cfg is None else task_cfg)
    if (
        split == "train"
        and str(cfg.get("train_window_sampling", "concat")) == "reset_bank"
        and cfg.get("train_window_starts")
    ):
        starts = [int(x) for x in cfg["train_window_starts"]]
        if len(starts) >= int(n_episodes):
            return starts[: int(n_episodes)]
        return [starts[i % len(starts)] for i in range(int(n_episodes))]

    _ensure_powerzoo_path()
    from powerzoo.tasks.dso_task import load_dso_feeder_shapes

    feeder_shapes = load_dso_feeder_shapes(role=split)
    total_steps = min(len(shape) for shape in feeder_shapes.values())
    max_start = max(total_steps - int(max_steps), 0)
    return list(
        np.linspace(
            0,
            max_start,
            num=max(int(n_episodes), 1),
            dtype=np.int64,
        ).astype(int)
    )


def _wrap_single_agent_monitor(env: gym.Env) -> gym.Env:
    """Ensure episode totals reach SB3/SBX callbacks on single-env runs."""
    try:
        from stable_baselines3.common.monitor import Monitor

        return Monitor(env)
    except Exception:
        try:
            from gymnasium.wrappers import RecordEpisodeStatistics

            return RecordEpisodeStatistics(env)
        except Exception:
            return env


def _make_jax_dc_episode_params(
    split: str,
    *,
    seed: int,
    episode_idx: int,
    n_episodes: int,
    strategy: str,
):
    from powerzoojax.tasks.dc_microgrid import DCMicrogridTask

    task_cfg = _dc_task_config()
    max_steps = int(task_cfg.get("max_steps", 288))
    task = DCMicrogridTask(
        source=task_cfg.get("data_source", "google"),
        max_steps=max_steps,
        case_overrides=task_cfg.get("case_overrides") or {},
    )
    return task.episode_params(
        split,
        int(episode_idx),
        int(n_episodes),
        max_steps,
        strategy=strategy,
        seed=int(seed),
    )


def _powerzoo_dc_env_from_jax_params(params):
    """Instantiate PowerZoo's Python DC env from JAX benchmark params."""
    from powerzoo.envs.microgrid.dc_microgrid_env import DCMicrogridEnv

    dc = params.dc
    delta_t_minutes = float(dc.delta_t_hours) * 60.0
    train_cfg = {
        "arrival_interval": int(dc.train_arrival_interval),
        "gpu_range": (int(dc.train_gpu_lo), int(dc.train_gpu_hi)),
        "duration_range": (int(dc.train_dur_lo), int(dc.train_dur_hi)),
        "deadline_slack": float(dc.train_deadline_slack),
        "gpu_eta": float(dc.train_gpu_eta),
    }
    finetune_cfg = {
        "arrival_interval": int(dc.ft_arrival_interval),
        "gpu_range": (int(dc.ft_gpu_lo), int(dc.ft_gpu_hi)),
        "duration_range": (int(dc.ft_dur_lo), int(dc.ft_dur_hi)),
        "deadline_slack": float(dc.ft_deadline_slack),
        "gpu_eta": float(dc.ft_gpu_eta),
    }
    dg = params.dg
    env = DCMicrogridEnv(
        n_gpus=int(dc.n_gpus),
        gpu_idle_w=float(dc.gpu_idle_w),
        gpu_active_w=float(dc.gpu_active_w),
        p_base_mw=float(dc.p_base_mw),
        infer_gpu_peak=int(dc.infer_gpu_peak),
        cop_ref=float(dc.cop_ref),
        cop_decay=float(dc.cop_decay),
        t_ref=float(dc.t_ref),
        c_thermal=float(dc.c_thermal),
        ua_cooling=float(dc.ua_cooling),
        h_wall=float(dc.h_wall),
        t_set_min=float(dc.t_set_min),
        t_set_max=float(dc.t_set_max),
        t_critical=float(dc.t_critical),
        p_aux_frac=float(dc.p_aux_frac),
        train_cfg=train_cfg,
        finetune_cfg=finetune_cfg,
        battery_capacity_mwh=float(params.battery_capacity_mwh),
        battery_power_mw=float(params.battery_power_mw),
        battery_eta_charge=float(params.battery_eta_charge),
        battery_eta_discharge=float(params.battery_eta_discharge),
        battery_soc_min=float(params.battery_soc_min),
        battery_soc_max=float(params.battery_soc_max),
        battery_soc_init=float(params.battery_soc_init),
        battery_deg_cost_per_mwh=float(params.battery_deg_cost_per_mwh),
        pv_capacity_mw=float(params.pv_p_max_mw),
        dg_max_mw=float(dg.p_dg_max_mw) if dg is not None else 0.0,
        dg_fuel_cost_per_mwh=float(dg.fuel_cost_per_mwh) if dg is not None else 300.0,
        dg_emission_factor=float(dg.emission_factor) if dg is not None else 0.80,
        w_cost=float(params.w_cost),
        w_carbon=float(params.w_carbon),
        max_steps=int(dc.max_steps),
        delta_t_minutes=delta_t_minutes,
        cpu_profile=np.asarray(params.cpu_profile, dtype=np.float32)
        if params.cpu_profile is not None else None,
        solar_profile=np.asarray(params.solar_profile, dtype=np.float32)
        if params.solar_profile is not None else None,
        outdoor_temp_profile=np.asarray(params.outdoor_temp_profile, dtype=np.float32)
        if params.outdoor_temp_profile is not None else None,
    )
    env._pzjax_grid_import_p_max_mw = float(getattr(params, "grid_import_p_max_mw", 0.0))
    env._pzjax_grid_price_ref_per_mwh = float(getattr(params, "grid_price_ref_per_mwh", 150.0))
    env._pzjax_grid_carbon_kg_per_kwh = float(getattr(params, "grid_carbon_kg_per_kwh", 0.18))
    env._pzjax_terminal_soc_target = float(getattr(params, "terminal_soc_target", params.battery_soc_init))
    env._pzjax_terminal_soc_penalty = float(getattr(params, "terminal_soc_penalty", 0.0))
    env._pzjax_price_profile = (
        np.asarray(params.price_profile, dtype=np.float32)
        if getattr(params, "price_profile", None) is not None else None
    )
    return env


def _clone_dc_tasks(tasks) -> list[Any]:
    return [copy.deepcopy(task) for task in tasks]


def _snapshot_np_random(obj) -> dict[str, Any] | None:
    rng = getattr(obj, "np_random", None)
    if rng is None:
        return None
    try:
        return copy.deepcopy(rng.bit_generator.state)
    except Exception:
        return None


def _restore_np_random(obj, state: dict[str, Any] | None) -> None:
    if state is None:
        return
    rng = getattr(obj, "np_random", None)
    if rng is None:
        return
    try:
        rng.bit_generator.state = copy.deepcopy(state)
    except Exception:
        pass


def _snapshot_powerzoo_dc_state(env) -> dict[str, Any]:
    dc = env._dc
    batt = env._batt
    pv = env._pv
    dg = env._dg
    return {
        "env": {
            "_step_count": int(env._step_count),
            "time_step": int(env.time_step),
            "_last_action": np.asarray(env._last_action, dtype=np.float32).copy(),
        },
        "dc": {
            "t_zone": float(dc.t_zone),
            "t_setpoint": float(dc.t_setpoint),
            "t_outdoor": float(dc.t_outdoor),
            "p_it_mw": float(dc.p_it_mw),
            "p_cool_mw": float(dc.p_cool_mw),
            "p_dc_mw": float(dc.p_dc_mw),
            "gpus_infer": int(dc.gpus_infer),
            "gpus_active": int(dc.gpus_active),
            "sla_violations": int(dc.sla_violations),
            "step_sla_violations": int(dc.step_sla_violations),
            "is_overtemp": bool(dc.is_overtemp),
            "current_p_mw": float(dc.current_p_mw),
            "current_q_mvar": float(dc.current_q_mvar),
            "time_step": int(dc.time_step),
            "_wait_queue": _clone_dc_tasks(dc._wait_queue),
            "_running": _clone_dc_tasks(dc._running),
            "np_random_state": _snapshot_np_random(dc),
        },
        "batt": {
            "soc": float(batt.soc),
            "current_p_mw": float(batt.current_p_mw),
            "current_q_mvar": float(batt.current_q_mvar),
            "_clipped_power_mw": float(batt._clipped_power_mw),
            "throughput_mwh": float(batt.throughput_mwh),
            "_soc_history": list(batt._soc_history),
            "time_step": int(batt.time_step),
        },
        "pv": {
            "current_p_mw": float(pv.current_p_mw),
            "current_q_mvar": float(pv.current_q_mvar),
            "_capacity_factor": float(pv._capacity_factor),
            "time_step": int(pv.time_step),
        },
        "dg": {
            "current_p_mw": float(dg.current_p_mw),
            "current_q_mvar": float(dg.current_q_mvar),
            "fuel_cost_step": float(dg.fuel_cost_step),
            "carbon_kg_step": float(dg.carbon_kg_step),
            "time_step": int(dg.time_step),
        },
    }


def _restore_powerzoo_dc_state(env, snapshot: dict[str, Any]) -> None:
    env_snap = snapshot["env"]
    env._step_count = int(env_snap["_step_count"])
    env.time_step = int(env_snap["time_step"])
    env._last_action = np.asarray(env_snap["_last_action"], dtype=np.float32).copy()

    dc_snap = snapshot["dc"]
    dc = env._dc
    dc.t_zone = float(dc_snap["t_zone"])
    dc.t_setpoint = float(dc_snap["t_setpoint"])
    dc.t_outdoor = float(dc_snap["t_outdoor"])
    dc.p_it_mw = float(dc_snap["p_it_mw"])
    dc.p_cool_mw = float(dc_snap["p_cool_mw"])
    dc.p_dc_mw = float(dc_snap["p_dc_mw"])
    dc.gpus_infer = int(dc_snap["gpus_infer"])
    dc.gpus_active = int(dc_snap["gpus_active"])
    dc.sla_violations = int(dc_snap["sla_violations"])
    dc.step_sla_violations = int(dc_snap["step_sla_violations"])
    dc.is_overtemp = bool(dc_snap["is_overtemp"])
    dc.current_p_mw = float(dc_snap["current_p_mw"])
    dc.current_q_mvar = float(dc_snap["current_q_mvar"])
    dc.time_step = int(dc_snap["time_step"])
    dc._wait_queue = _clone_dc_tasks(dc_snap["_wait_queue"])
    dc._running = _clone_dc_tasks(dc_snap["_running"])
    _restore_np_random(dc, dc_snap["np_random_state"])

    batt_snap = snapshot["batt"]
    batt = env._batt
    batt.soc = float(batt_snap["soc"])
    batt.current_p_mw = float(batt_snap["current_p_mw"])
    batt.current_q_mvar = float(batt_snap["current_q_mvar"])
    batt._clipped_power_mw = float(batt_snap["_clipped_power_mw"])
    batt.throughput_mwh = float(batt_snap["throughput_mwh"])
    batt._soc_history = list(batt_snap["_soc_history"])
    batt.time_step = int(batt_snap["time_step"])

    pv_snap = snapshot["pv"]
    pv = env._pv
    pv.current_p_mw = float(pv_snap["current_p_mw"])
    pv.current_q_mvar = float(pv_snap["current_q_mvar"])
    pv._capacity_factor = float(pv_snap["_capacity_factor"])
    pv.time_step = int(pv_snap["time_step"])

    dg_snap = snapshot["dg"]
    dg = env._dg
    dg.current_p_mw = float(dg_snap["current_p_mw"])
    dg.current_q_mvar = float(dg_snap["current_q_mvar"])
    dg.fuel_cost_step = float(dg_snap["fuel_cost_step"])
    dg.carbon_kg_step = float(dg_snap["carbon_kg_step"])
    dg.time_step = int(dg_snap["time_step"])


class _PowerZooDCBenchmarkWrapper(gym.Wrapper):
    """Mirror the frozen JAX benchmark semantics on PowerZoo's Python env."""

    _OBS_LOW = np.array(
        [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1, -1],
        dtype=np.float32,
    )
    _OBS_HIGH = np.ones(24, dtype=np.float32)

    def __init__(self, inner, lambdas: dict[str, float], *, dg_autobalance: bool):
        super().__init__(inner)
        self._lambdas = dict(lambdas)
        self._dg_autobalance = bool(dg_autobalance)
        self.observation_space = Box(
            low=self._OBS_LOW,
            high=self._OBS_HIGH,
            shape=(24,),
            dtype=np.float32,
        )
        self.action_space = inner.action_space

    def _grid_price(self, step: int | None = None) -> float:
        profile = getattr(self.env, "_pzjax_price_profile", None)
        if profile is None or len(profile) == 0:
            return float(getattr(self.env, "_pzjax_grid_price_ref_per_mwh", 150.0))
        idx = int(self.env._step_count if step is None else step) % int(len(profile))
        return float(profile[idx])

    def _grid_price_norm(self, value: float) -> float:
        ref = max(float(getattr(self.env, "_pzjax_grid_price_ref_per_mwh", 150.0)), 1e-6)
        return float(np.clip(float(value) / ref, 0.0, 1.0))

    def _grid_price_future_max_norm(self, horizon_steps: int = 72) -> float:
        profile = getattr(self.env, "_pzjax_price_profile", None)
        if profile is None or len(profile) == 0:
            return self._grid_price_norm(self._grid_price())
        n = int(len(profile))
        start = int(self.env._step_count)
        idx = (start + np.arange(int(horizon_steps), dtype=np.int64)) % n
        return self._grid_price_norm(float(np.max(profile[idx])))

    def reset(self, *, seed=None, options=None):
        self.env.reset(seed=seed, options=options)
        info = {"step": 0, "reward_vector": [0.0, 0.0, 0.0]}
        return self._build_obs(), info

    def _battery_headroom(self, soc: float) -> tuple[float, float]:
        batt = self.env._batt
        safe_p = max(float(batt.power_mw), 1e-6)
        dt_h = max(float(batt.dt_hours), 1e-6)
        max_dis = np.clip(
            (soc - float(batt.soc_min))
            * float(batt.capacity_mwh)
            * float(batt.eta_discharge)
            / dt_h,
            0.0,
            safe_p,
        )
        max_chg = np.clip(
            (float(batt.soc_max) - soc)
            * float(batt.capacity_mwh)
            / (max(float(batt.eta_charge), 1e-6) * dt_h),
            0.0,
            safe_p,
        )
        return float(max_dis), float(max_chg)

    def _dispatch_tracking_cost(
        self,
        prev_soc: float,
        p_load: float,
        p_pv: float,
        p_batt: float,
        p_dg: float,
    ) -> tuple[float, float, float]:
        batt = self.env._batt
        dg = self.env._dg
        safe_batt = max(float(batt.power_mw), 1e-6)
        safe_dg = max(float(dg.p_dg_max_mw), 1e-6)
        net_load = float(p_load - p_pv)
        max_dis, max_chg = self._battery_headroom(prev_soc)
        price_ratio = self._grid_price_norm(self._grid_price())
        low_price_signal = max(0.0, 0.55 - price_ratio) / 0.55
        high_price_signal = max(0.0, price_ratio - 0.85) / 0.15
        batt_charge_target = -max_chg * low_price_signal
        batt_discharge_target = min(max_dis, max(net_load, 0.0)) * min(high_price_signal, 1.0)
        batt_target = float(np.clip(batt_charge_target + batt_discharge_target, -max_chg, max_dis))
        diesel_cost = float(getattr(dg, "fuel_cost_per_mwh", 300.0))
        dg_target_if_economic = max(net_load - batt_target, 0.0) if self._grid_price() > diesel_cost else 0.0
        dg_target = float(np.clip(dg_target_if_economic, 0.0, safe_dg))
        batt_cost = abs(float(p_batt) - batt_target) / safe_batt
        dg_cost = abs(float(p_dg) - dg_target) / safe_dg
        return float(batt_cost + dg_cost), batt_target, dg_target

    def _build_obs(self) -> np.ndarray:
        dc = self.env._dc
        batt = self.env._batt
        pv = self.env._pv
        dg = self.env._dg
        dc_obs = dc.obs()
        cpu_util = float(dc_obs.get("gpu_util", 0.0))
        mem_util = float(dc_obs.get("infer_util", 0.0))
        q_train_fill = float(dc_obs.get("queue_train_fill", 0.0))
        q_ft_fill = float(dc_obs.get("queue_ft_fill", 0.0))
        urgency = float(dc_obs.get("queue_urgency", 1.0))
        zone_norm = float(dc_obs.get("zone_temp_norm", 0.5))
        outdoor_norm = float(dc_obs.get("outdoor_temp_norm", 0.5))
        cop_factor = float(np.clip(
            1.0 - float(dc.cop_decay) * max(float(dc.t_outdoor) - float(dc.t_ref), 0.0),
            0.4,
            1.2,
        ))
        cop_ratio = float(np.clip((cop_factor - 0.4) / 0.8, 0.0, 1.0))
        solar_cf = float(pv.available_cf)
        soc = float(batt.soc)
        dg_margin_norm = float(np.clip(
            (float(dg.p_dg_max_mw) - float(dg.current_p_mw))
            / max(float(dg.p_dg_max_mw), 1e-8),
            0.0,
            1.0,
        ))
        p_load_mw = max(float(dc.p_dc_mw), 0.0)
        p_pv_mw = max(float(pv.current_p_mw), 0.0)
        batt_p_max = max(float(batt.power_mw), 0.0)
        dg_p_max = max(float(dg.p_dg_max_mw), 0.0)
        pv_p_max = max(float(pv.capacity_mw), 0.0)
        total_supply_scale = max(batt_p_max + dg_p_max + pv_p_max, 1e-6)
        dispatchable_scale = max(batt_p_max + dg_p_max, 1e-6)
        p_load_norm = float(np.clip(p_load_mw / total_supply_scale, 0.0, 1.0))
        net_load_norm = float(np.clip((p_load_mw - p_pv_mw) / dispatchable_scale, -1.0, 1.0))
        batt_dis_headroom, batt_chg_headroom = self._battery_headroom(soc)
        batt_dis_headroom_norm = float(np.clip(batt_dis_headroom / max(batt_p_max, 1e-6), 0.0, 1.0))
        batt_chg_headroom_norm = float(np.clip(batt_chg_headroom / max(batt_p_max, 1e-6), 0.0, 1.0))
        price_norm = self._grid_price_norm(self._grid_price())
        price_future_norm = self._grid_price_future_max_norm()
        last_action = np.asarray(self.env._last_action, dtype=np.float32).reshape(-1)
        phase = 2.0 * np.pi * float(self.env._step_count) / max(float(self.env.max_steps), 1.0)
        return np.clip(
            np.array(
                [
                    cpu_util,
                    mem_util,
                    q_train_fill,
                    q_ft_fill,
                    urgency,
                    zone_norm,
                    outdoor_norm,
                    cop_ratio,
                    solar_cf,
                    soc,
                    dg_margin_norm,
                    p_load_norm,
                    net_load_norm,
                    batt_dis_headroom_norm,
                    batt_chg_headroom_norm,
                    price_norm,
                    price_future_norm,
                    float(last_action[0]) if last_action.size > 0 else 0.0,
                    float(last_action[1]) if last_action.size > 1 else 0.0,
                    float(last_action[2]) if last_action.size > 2 else 0.0,
                    float(last_action[3]) if last_action.size > 3 else 0.0,
                    float(last_action[4]) if last_action.size > 4 else 0.0,
                    float(np.sin(phase)),
                    float(np.cos(phase)),
                ],
                dtype=np.float32,
            ),
            self._OBS_LOW,
            self._OBS_HIGH,
        )

    def step(self, action):
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        env_action = np.clip(action_arr, self.action_space.low, self.action_space.high)
        dg_command_raw_norm = float(env_action[4]) if env_action.size > 4 else 0.0
        dg_command_balanced_norm = dg_command_raw_norm
        dg_preview_residual_mw = 0.0

        if self._dg_autobalance and env_action.size > 4:
            dg_max = max(float(self.env._dg.p_dg_max_mw), 1e-6)
            snap = _snapshot_powerzoo_dc_state(self.env)
            preview_action = env_action.copy()
            preview_action[4] = 0.0
            _, _, _, _, preview_info = self.env.step(preview_action)
            dg_preview_residual_mw = float(
                preview_info.get("p_pv_mw", 0.0)
                + preview_info.get("p_batt_mw", 0.0)
                + preview_info.get("p_dg_mw", 0.0)
                - preview_info.get("p_load_mw", 0.0)
            )
            grid_cap = max(float(getattr(self.env, "_pzjax_grid_import_p_max_mw", 0.0)), 0.0)
            remaining_deficit = max(-dg_preview_residual_mw - grid_cap, 0.0)
            dg_command_balanced_norm = float(np.clip(remaining_deficit / dg_max, 0.0, 1.0))
            _restore_powerzoo_dc_state(self.env, snap)
            env_action[4] = dg_command_balanced_norm

        prev_soc = float(self.env._batt.soc)
        _, reward, terminated, truncated, info = self.env.step(env_action)
        info = dict(info)
        costs = info.get("costs", {}) or {}
        cost_sla = float(info.get("cost_sla", costs.get("sla", 0.0)))
        cost_overtemp = float(info.get("cost_overtemp", costs.get("overtemp", 0.0)))
        cost_deficit = float(info.get("cost_power_deficit", costs.get("power_deficit", 0.0)))
        p_load = float(info.get("p_load_mw", 0.0))
        p_pv = float(info.get("p_pv_mw", 0.0))
        p_batt = float(info.get("p_batt_mw", 0.0))
        p_dg = float(info.get("p_dg_mw", 0.0))
        raw_residual = float(p_pv + p_batt + p_dg - p_load)
        grid_cap = max(float(getattr(self.env, "_pzjax_grid_import_p_max_mw", 0.0)), 0.0)
        p_grid = float(np.clip(max(-raw_residual, 0.0), 0.0, grid_cap))
        residual = float(raw_residual + p_grid)
        power_spill = float(max(residual, 0.0))
        cost_deficit = float(max(-residual, 0.0) / max(p_load, 1e-6))
        cost_spill = power_spill / max(p_load, 1e-6)
        cost_balance = abs(residual) / max(p_load, 1e-6)
        dt_h = float(getattr(self.env, "delta_t_minutes", 5.0)) / 60.0
        grid_price = self._grid_price(int(getattr(self.env, "_step_count", 1)) - 1)
        grid_cost = float(p_grid * dt_h * grid_price)
        grid_carbon = float(
            p_grid
            * dt_h
            * 1000.0
            * float(getattr(self.env, "_pzjax_grid_carbon_kg_per_kwh", 0.18))
        )
        is_terminal = bool(terminated) or bool(truncated)
        terminal_soc_cost = 0.0
        if is_terminal:
            terminal_soc_cost = float(
                getattr(self.env, "_pzjax_terminal_soc_penalty", 0.0)
                * (float(self.env._batt.soc) - float(getattr(self.env, "_pzjax_terminal_soc_target", 0.55))) ** 2
            )
        dispatch_tracking, batt_target, dg_target = self._dispatch_tracking_cost(
            prev_soc, p_load, p_pv, p_batt, p_dg,
        )
        penalty = (
            float(self._lambdas.get("sla", 0.0)) * cost_sla
            + float(self._lambdas.get("overtemp", 0.0)) * cost_overtemp
            + float(self._lambdas.get("power_deficit", 0.0)) * cost_deficit
            + float(self._lambdas.get("power_spill", 0.0)) * cost_spill
            + float(self._lambdas.get("power_balance", 0.0)) * cost_balance
            + float(self._lambdas.get("dispatch_tracking", 0.0)) * dispatch_tracking
        )
        shaped = (
            float(reward)
            - float(penalty)
            - float(getattr(self.env, "w_cost", 0.0)) * (grid_cost + terminal_soc_cost)
            - float(getattr(self.env, "w_carbon", 0.0)) * grid_carbon
        )
        info["raw_reward"] = float(reward)
        info["reward"] = float(shaped)
        info["shaping_penalty"] = float(penalty)
        info["power_spill"] = float(power_spill)
        info["raw_residual"] = float(raw_residual)
        info["residual"] = float(residual)
        info["cost_power_spill"] = float(cost_spill)
        info["cost_power_balance"] = float(cost_balance)
        info["cost_power_deficit"] = float(cost_deficit)
        info["cost_dispatch_tracking"] = float(dispatch_tracking)
        info["dispatch_target_batt_mw"] = float(batt_target)
        info["dispatch_target_dg_mw"] = float(dg_target)
        info["p_grid_import_mw"] = float(p_grid)
        info["grid_price_per_mwh"] = float(grid_price)
        info["grid_cost"] = float(grid_cost)
        info["grid_carbon_kg"] = float(grid_carbon)
        info["terminal_soc_cost"] = float(terminal_soc_cost)
        if self._dg_autobalance:
            info["dg_command_raw_norm"] = float(dg_command_raw_norm)
            info["dg_command_balanced_norm"] = float(dg_command_balanced_norm)
            info["dg_preview_residual_mw"] = float(dg_preview_residual_mw)
        return self._build_obs(), float(shaped), bool(terminated), bool(truncated), info


def _wrap_powerzoo_reward_shaping(jax_task: str, env):
    """Apply benchmark reward shaping to PowerZoo envs when JAX uses it."""
    weights = _powerzoo_reward_shaping_weights(jax_task)
    if jax_task == "dc_microgrid":
        return _PowerZooDCBenchmarkWrapper(
            env,
            weights,
            dg_autobalance=bool(_dc_task_config().get("dg_autobalance", False)),
        )
    if not weights or all(v == 0.0 for v in weights.values()):
        return env

    class _RewardShapingGymWrapper(gym.Wrapper):
        def __init__(self, inner, lambdas: dict[str, float]):
            super().__init__(inner)
            self._lambdas = dict(lambdas)

        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            info = dict(info)
            costs = info.get("costs", {}) or {}
            cost_sla = float(info.get("cost_sla", costs.get("sla", 0.0)))
            cost_overtemp = float(
                info.get("cost_overtemp", costs.get("overtemp", 0.0))
            )
            cost_deficit = float(
                info.get("cost_power_deficit", costs.get("power_deficit", 0.0))
            )
            penalty = (
                self._lambdas["sla"] * cost_sla
                + self._lambdas["overtemp"] * cost_overtemp
                + self._lambdas["power_deficit"] * cost_deficit
            )
            shaped = float(reward) - float(penalty)
            info["raw_reward"] = float(reward)
            info["reward"] = float(shaped)
            info["shaping_penalty"] = float(penalty)
            return obs, shaped, terminated, truncated, info

    return _RewardShapingGymWrapper(env, weights)

def _build_powerzoo_env(
    jax_task: str,
    *,
    split: str = "train",
    seed: int = 0,
    episode_idx: int = 0,
    n_episodes: int = 1,
    strategy: str = "uniform",
    dry_run_steps: int = 0,
):
    """Construct the PowerZoo env (or marl env handle) for ``jax_task``.

    Parameters
    ----------
    jax_task : str
        PowerZoojax task name (``"dso"`` / ``"tso"`` / ``"ders"`` /
        ``"gencos"`` / ``"dc_microgrid"``).
    split : str
        Data split to instantiate the env on.  Must be listed in
        ``_VALID_SPLITS_BY_TASK[jax_task]``.  Honest split metadata in
        ``RunRecord.split``
        depends on the env actually being built with this split — a
        previous version silently hard-coded ``"train"`` and produced
        records that lied about which window the model trained on.
    seed, episode_idx, n_episodes, strategy : used by tasks whose
        comparable PowerZoo env is reconstructed from JAX benchmark episode
        params (currently DC Microgrid).
    dry_run_steps : int
        Used only by the ``pz_registry`` path to size a dummy Trainer.

    Returns
    -------
    (env_or_marl_handle, env_kind) where ``env_kind`` is ``"single"``
    (gym.Env ready for SB3) or ``"pettingzoo"`` (marl env; caller wraps
    into IL views per agent).
    """
    _ensure_powerzoo_path()
    valid_splits = _VALID_SPLITS_BY_TASK.get(jax_task)
    if valid_splits is None:
        raise ValueError(f"Unknown task {jax_task!r}; cannot validate split")
    if split not in valid_splits:
        raise ValueError(
            f"Unknown split {split!r} for task {jax_task!r}. "
            f"Must be one of {valid_splits}. "
            f"Cross-backend records must declare an honest split; the "
            f"driver refuses to silently fall back."
        )
    entry = JAX_TASK_TO_POWERZOO_TASK[jax_task]
    path = entry["access_path"]
    module = entry["powerzoo_module"]
    factory = entry["powerzoo_task"]
    factory_kwargs = entry.get("powerzoo_factory_kwargs") or {}

    if path == "direct_env":
        if jax_task == "dso":
            # Use Ausgrid real feeder shapes for the requested split
            # (date window + day-level filter mirrored from PowerZooJax
            # `splits.py`; alignment guarded by
            # tests/benchmarks/test_split_alignment.py).
            from powerzoo.tasks.dso_task import make_dso_env
            kwargs = _powerzoo_dso_env_kwargs(
                split,
                seed=seed,
                use_train_reset_bank=True,
            )
            kwargs.update(factory_kwargs)
            env = make_dso_env(**kwargs)
            _verify_powerzoo_env_contract(jax_task, env, split=split)
            return _wrap_powerzoo_reward_shaping(jax_task, env), "single"

        if jax_task == "tso":
            from powerzoo.tasks.middle.comparison_tso import (
                CentralizedComparisonTSOEnv,
            )
            try:
                task, profile_window = _instantiate_powerzoo_tso_task(split)
                env = CentralizedComparisonTSOEnv(task)
            except Exception as exc:
                raise CrossBackendNotComparable(
                    "PowerZoo TSO env failed to build on the frozen benchmark "
                    f"window {profile_window[0]}..{profile_window[1]} for "
                    f"split={split!r}: {exc}"
                ) from exc
            _annotate_tso_powerzoo_env(
                env,
                split=split,
                profile_window=profile_window,
            )
            _verify_powerzoo_env_contract(jax_task, env, split=split)
            return _wrap_powerzoo_reward_shaping(jax_task, env), "single"

        raise ValueError(
            f"direct_env path requested for {jax_task!r} but no factory wired"
        )

    if path == "pz_registry":
        if jax_task == "dc_microgrid":
            params = _make_jax_dc_episode_params(
                split,
                seed=seed,
                episode_idx=episode_idx,
                n_episodes=n_episodes,
                strategy=strategy,
            )
            env = _powerzoo_dc_env_from_jax_params(params)
            _verify_powerzoo_env_contract(jax_task, env, split=split)
            return _wrap_powerzoo_reward_shaping(jax_task, env), "single"

        from powerzoo.rl import Trainer
        trainer_kwargs: dict[str, Any] = dict(
            algorithm="PPO", total_timesteps=max(dry_run_steps, 100),
            framework=entry.get("framework", "auto"),
        )
        trainer = Trainer(factory, **trainer_kwargs)
        env = trainer.get_env()
        _verify_powerzoo_env_contract(jax_task, env, split=split)
        return _wrap_powerzoo_reward_shaping(jax_task, env), "single"

    if path == "pettingzoo_il":
        # DERs / GenCos: build PettingZoo ParallelEnv via PowerZoo's
        # make_env() entry point.  TaskPettingZooWrapper has a long-
        # standing bug (line 39 writes ``self.env.observation_space[agent]``
        # assuming dict access; GenCos exposes it as a method, raising
        # ``TypeError: 'method' object is not subscriptable``).  We side-
        # step by reaching one level inside to the actual MultiAgentEnv
        # which implements the canonical PettingZoo callable signatures.
        #
        # ``factory_kwargs`` is forwarded to the Task constructor so that
        # cross-backend hyperparameters (e.g. DERs voltage_penalty) match
        # the PowerZooJax DERs frozen train config value declared in
        # ``powerzoo_bridge.py::*.powerzoo_factory_kwargs``.
        from powerzoo.rl.env import make_env
        wrapper = make_env(
            factory, framework="pettingzoo", split=split, **factory_kwargs,
        )
        marl_env = getattr(wrapper, "env", wrapper)
        if jax_task == "ders":
            marl_env = _PowerZooDERSExactEpisodeWrapper(
                marl_env,
                split=split,
                seed=seed,
                episode_idx=episode_idx,
                n_episodes=n_episodes,
                strategy=strategy,
                voltage_penalty=float(factory_kwargs.get("voltage_penalty", 0.0)),
            )
        try:
            marl_env.reset(seed=0)
        except TypeError:
            marl_env.reset()
        _verify_powerzoo_env_contract(jax_task, marl_env, split=split)
        return marl_env, "pettingzoo"

    raise ValueError(f"Unknown access_path: {path!r}")

def _build_powerzoo_vec_env(
    jax_task: str,
    *,
    split: str,
    seed: int,
    n_envs: int,
    strategy: str = "uniform",
    vec_env: str = "auto",
    start_methods: tuple[str, ...] | None = None,
):
    """Build a monitored VecEnv for single-agent PowerZoo tasks.

    Cross-backend PPO should be allowed to use the Python backend's native
    environment parallelism just as JAX/Rejax uses batched envs.  The total
    SB3/SBX ``learn(total_timesteps=...)`` budget remains unchanged; this only
    changes how those samples are collected.

    ``vec_env`` controls the fallback policy:

    - ``"auto"``: use ``SubprocVecEnv`` for ``n_envs > 1`` when possible,
      fallback to ``DummyVecEnv`` if every safe start method fails.
    - ``"subproc"``: require ``SubprocVecEnv`` for ``n_envs > 1`` and raise on
      failure.  This is used by execution scaling so Python-backend rows do not
      silently become single-process measurements.
    - ``"dummy"``: force ``DummyVecEnv``.
    """
    if n_envs < 1:
        raise ValueError(f"n_envs must be >= 1, got {n_envs}")
    vec_env = str(vec_env).strip().lower()
    if vec_env not in ("auto", "subproc", "dummy"):
        raise ValueError("vec_env must be one of: auto, subproc, dummy")
    _ensure_powerzoo_path()

    def make_env(rank: int):
        def _init():
            env, kind = _build_powerzoo_env(
                jax_task,
                split=split,
                seed=seed,
                episode_idx=0,
                n_episodes=1,
                strategy=strategy,
            )
            if kind != "single":
                raise RuntimeError(
                    f"VecEnv requested for {jax_task!r}, but env kind is {kind!r}"
                )
            try:
                env.reset(seed=int(seed) + rank)
            except TypeError:
                env.reset()
            return _wrap_single_agent_monitor(env)

        return _init

    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    env_fns = [make_env(i) for i in range(int(n_envs))]
    if n_envs == 1 or vec_env == "dummy":
        return DummyVecEnv(env_fns)

    last_exc: Exception | None = None
    if start_methods is None:
        env_override = os.environ.get("POWERZOOJAX_SUBPROC_START_METHODS")
        if env_override:
            start_methods = tuple(
                item.strip() for item in env_override.split(",") if item.strip()
            )
        elif jax_task == "dc_microgrid":
            # DC Microgrid reconstruction imports JAX-side task/params helpers.
            # Forking after JAX/CUDA has been imported is unsafe, so prefer
            # start methods that create clean child interpreters.
            start_methods = ("forkserver", "spawn")
        else:
            start_methods = ("forkserver", "spawn", "fork")
    if not start_methods:
        start_methods = ("spawn",)

    for method in start_methods:
        try:
            vec = SubprocVecEnv(env_fns, start_method=method)
            setattr(vec, "powerzoojax_start_method", method)
            return vec
        except Exception as exc:
            last_exc = exc
            print(
                f"[powerzoo_driver] SubprocVecEnv(n_envs={n_envs}, "
                f"start_method={method}) failed ({type(exc).__name__}: {exc})."
            )
    if vec_env == "subproc":
        raise RuntimeError(
            f"SubprocVecEnv required for {jax_task} n_envs={n_envs}, but all "
            f"start methods failed: {start_methods}. Last error: {last_exc}"
        )
    print(
        f"[powerzoo_driver] falling back to DummyVecEnv after SubprocVecEnv "
        f"failure: {last_exc}"
    )
    return DummyVecEnv(env_fns)


def _mean_numeric_dicts(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({k for row in rows for k in row.keys()})
    out: dict[str, float] = {}
    for key in keys:
        vals: list[float] = []
        for row in rows:
            value = row.get(key)
            if value is None:
                continue
            try:
                vals.append(float(value))
            except (TypeError, ValueError):
                continue
        if vals:
            out[key] = float(np.mean(np.asarray(vals, dtype=np.float64)))
    return out


def _scalarize_info(info: dict[str, Any], reward: float) -> dict[str, float]:
    row: dict[str, float] = {}
    for key, value in info.items():
        try:
            arr = np.asarray(value)
        except Exception:
            continue
        if arr.ndim == 0:
            row[key] = float(arr)
    row["reward"] = float(reward)
    return row


def _env_contract_metadata(handle) -> dict[str, Any]:
    """Extract task-level provenance metadata exposed by a PowerZoo env."""
    meta: dict[str, Any] = {}
    for key in ("data_source", "load_profile_source", "benchmark_split", "ood_axis"):
        value = getattr(handle, key, None)
        if value is not None:
            meta[key] = value
    profile_window = getattr(handle, "profile_window", None)
    if profile_window is not None:
        meta["profile_window"] = list(profile_window)
    return meta


def _collect_env_info_with_contract_meta(
    meta: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Merge runtime env info with task-contract provenance as strings."""
    return merge_env_info_metadata(collect_env_info(), meta)


def _cross_backend_gap_labels(
    jax_task: str,
    entry: dict[str, Any],
    *,
    train_split: str,
    eval_split: str,
    policy_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Machine-readable provenance for cross-backend result rows."""
    requested_dist = (policy_metadata or {}).get("requested_continuous_action_dist")
    effective_dist = (policy_metadata or {}).get("effective_continuous_action_dist")
    gap_tokens: list[str] = []
    if entry.get("access_path") == "pettingzoo_il":
        gap_tokens.append("frozen_self_play_il")
    if jax_task == "tso":
        gap_tokens.append("obs_dispatch_gap")
    if jax_task == "dc_microgrid":
        gap_tokens.append("episode_params_rebuild")
    if requested_dist and effective_dist and requested_dist != effective_dist:
        gap_tokens.append("action_dist_gap")

    if entry.get("access_path") == "pettingzoo_il":
        algo_family = "frozen_self_play_il"
    elif entry.get("n_agents", 1) > 1:
        algo_family = "multi_agent_rl"
    else:
        algo_family = "single_agent_rl"

    return {
        "backend_family": "python_torch",
        "algo_family": algo_family,
        "rl_paradigm": entry.get("rl_paradigm"),
        "train_split": train_split,
        "eval_split": eval_split,
        "split_status": "aligned" if train_split == eval_split else "train_then_eval",
        "cross_backend_gap_note": "none" if not gap_tokens else ",".join(gap_tokens),
        "requested_continuous_action_dist": requested_dist,
        "effective_continuous_action_dist": effective_dist,
        "policy_class": (policy_metadata or {}).get("policy_class"),
    }


def _collect_torch_run_contract(
    *,
    requested_device: str,
    context: str,
    meta: dict[str, Any] | None = None,
    extra_labels: dict[str, Any] | None = None,
    fail_fast: bool = True,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Collect env info and enforce requested-vs-actual torch device."""
    requested_raw = str(requested_device).strip().lower()
    requested_norm = normalize_device_name(requested_raw)
    actual_runtime_device = "cpu"
    actual_runtime_backend = "cpu"
    actual_runtime_device_kind = "cpu"
    torch_error: str | None = None

    try:
        import torch

        if requested_norm == "gpu":
            if torch.cuda.is_available():
                actual_runtime_device = "gpu"
                actual_runtime_backend = "cuda"
                device_idx = torch.cuda.current_device() if torch.cuda.device_count() > 0 else 0
                actual_runtime_device_kind = torch.cuda.get_device_name(device_idx)
            else:
                actual_runtime_device = "cpu"
                actual_runtime_backend = "cpu"
                actual_runtime_device_kind = "cpu"
        else:
            actual_runtime_device = "cpu"
            actual_runtime_backend = "cpu"
            actual_runtime_device_kind = "cpu"
    except Exception as exc:
        torch_error = f"{type(exc).__name__}: {exc}"
        actual_runtime_device = "unknown"
        actual_runtime_backend = "unknown"
        actual_runtime_device_kind = "unknown"

    contract_ok = requested_norm is None or requested_norm == actual_runtime_device
    if fail_fast and not contract_ok:
        extra = f" torch={torch_error}" if torch_error else ""
        raise RuntimeError(
            f"{context}: requested device={requested_raw!r}, "
            f"but torch resolved backend={actual_runtime_backend!r} "
            f"(device_kind={actual_runtime_device_kind!r}).{extra}"
        )

    env_info = _collect_env_info_with_contract_meta(meta)
    env_info.update(
        {
            "declared_backend": "python_torch",
            "actual_backend": str(actual_runtime_backend),
            "declared_device": requested_raw or "auto",
            "actual_device": str(actual_runtime_device),
            "actual_device_kind": str(actual_runtime_device_kind),
            "requested_device": requested_raw or "auto",
            "requested_device_normalized": requested_norm or "auto",
            "actual_runtime_device": str(actual_runtime_device),
            "actual_runtime_backend": str(actual_runtime_backend),
            "actual_runtime_device_kind": str(actual_runtime_device_kind),
            "runtime_family": "torch",
            "device_contract_ok": str(bool(contract_ok)).lower(),
            "cpu_affinity": (
                ",".join(str(cpu) for cpu in sorted(os.sched_getaffinity(0)))
                if hasattr(os, "sched_getaffinity")
                else "unknown"
            ),
            "OMP_NUM_THREADS": str(os.environ.get("OMP_NUM_THREADS", "")),
            "MKL_NUM_THREADS": str(os.environ.get("MKL_NUM_THREADS", "")),
            "OPENBLAS_NUM_THREADS": str(os.environ.get("OPENBLAS_NUM_THREADS", "")),
            "NUMEXPR_NUM_THREADS": str(os.environ.get("NUMEXPR_NUM_THREADS", "")),
        }
    )
    cpu_affinity = env_info["cpu_affinity"]
    cpu_core_budget = (
        len(cpu_affinity.split(",")) if cpu_affinity and cpu_affinity != "unknown" else None
    )
    labels: dict[str, Any] = {
        "runtime_family": "torch",
        "declared_backend": "python_torch",
        "actual_backend": str(actual_runtime_backend),
        "declared_device": requested_raw or "auto",
        "actual_device": str(actual_runtime_device),
        "actual_device_kind": str(actual_runtime_device_kind),
        "requested_device": requested_raw or "auto",
        "requested_device_normalized": requested_norm or "auto",
        "actual_runtime_device": str(actual_runtime_device),
        "actual_runtime_backend": str(actual_runtime_backend),
        "actual_runtime_device_kind": str(actual_runtime_device_kind),
        "device_contract_ok": bool(contract_ok),
        "device_recorded_as": requested_raw,
        "cpu_affinity": cpu_affinity,
        "cpu_core_budget": cpu_core_budget,
        "thread_settings": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", ""),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", ""),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS", ""),
        },
    }
    if extra_labels:
        labels.update(extra_labels)
    return env_info, labels


def _verify_powerzoo_env_contract(
    jax_task: str,
    handle,
    *,
    split: str,
) -> dict[str, Any]:
    """Refuse to benchmark a PowerZoo env whose data/split contract drifted."""
    meta = _env_contract_metadata(handle)
    if jax_task != "gencos":
        return meta

    expected_ood = {
        "train": None,
        "iid": None,
        "demand_shift": "demand_shift",
        "renewable_shock": "renewable_shock",
    }[split]
    actual_source = meta.get("data_source")
    actual_split = meta.get("benchmark_split")
    actual_ood = meta.get("ood_axis")
    if actual_source != "gb_real":
        raise CrossBackendNotComparable(
            f"GenCos PowerZoo env resolved data_source={actual_source!r}, "
            "expected 'gb_real'. Synthetic/custom fallbacks are not allowed "
            "for the official cross-backend benchmark."
        )
    if actual_split != split:
        raise CrossBackendNotComparable(
            f"GenCos PowerZoo env resolved benchmark_split={actual_split!r}, "
            f"expected requested split={split!r}."
        )
    if actual_ood != expected_ood:
        raise CrossBackendNotComparable(
            f"GenCos PowerZoo env resolved ood_axis={actual_ood!r}, "
            f"expected {expected_ood!r} for split={split!r}."
        )
    return meta


def _predict_policy_action(policy, obs, action_space):
    """Best-effort deterministic action for one policy; random fallback if absent."""
    if policy is None:
        return action_space.sample()
    if hasattr(policy, "predict"):
        action, _ = policy.predict(obs, deterministic=True)
        return action
    if callable(policy):
        return policy(obs)
    return action_space.sample()


def _ders_eval_episodes_for_split(split: str) -> int:
    task_dir = _BENCHMARKS_DIR / "ders"
    eval_cfg_path = task_dir / "configs" / f"eval_{split}.yaml"
    if eval_cfg_path.exists():
        cfg = load_config(eval_cfg_path)
        if cfg.get("n_eval_episodes") is not None:
            return int(cfg["n_eval_episodes"])
    task_cfg = load_task_config(task_dir)
    return int(task_cfg.get("eval_episodes", 30))


def _zero_action_for_space(action_space) -> np.ndarray:
    shape = getattr(action_space, "shape", ()) or ()
    return np.zeros(shape, dtype=np.float32)


def _rollout_ders_policy_bank_once(
    marl_env,
    policy_bank: dict[str, Any],
    *,
    split: str,
    episode_seed: int,
    episode_idx: int | None = None,
    n_episodes: int = 1,
    strategy: str = "uniform",
    episode_start: int | None = None,
) -> dict[str, Any]:
    """Evaluate one joint DER policy bank on one exact JAX-aligned episode."""
    agent_names = list(marl_env.possible_agents)
    reset_options: dict[str, Any] = {
        "split": split,
        "strategy": strategy,
        "n_episodes": int(max(n_episodes, 1)),
        "use_train_bank": False,
    }
    if episode_start is not None:
        reset_options["episode_start"] = int(episode_start)
    else:
        reset_options["episode_idx"] = int(episode_idx or 0)
    obs_dict, _ = marl_env.reset(seed=int(episode_seed), options=reset_options)

    reward_hist: list[float] = []
    cost_hist: list[float] = []
    loss_hist: list[float] = []
    vmin_hist: list[float] = []
    vmax_hist: list[float] = []
    action_hist: list[np.ndarray] = []

    done = False
    while not done:
        action_dict: dict[str, np.ndarray] = {}
        flat_step_actions: list[np.ndarray] = []
        for agent_id in agent_names:
            action_space = _agent_action_space(marl_env, agent_id)
            policy = policy_bank.get(agent_id)
            if policy is None:
                action = _zero_action_for_space(action_space)
            else:
                action = _predict_policy_action(policy, obs_dict[agent_id], action_space)
            action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
            action_dict[agent_id] = action_arr
            flat_step_actions.append(action_arr)

        obs_dict, rewards, terminateds, truncateds, infos = marl_env.step(action_dict)
        first_info = dict(infos.get(agent_names[0], {}))
        reward = float(rewards.get(agent_names[0], 0.0))
        reward_hist.append(reward)
        cost_hist.append(float(first_info.get("cost_continuous", 0.0)))
        loss_hist.append(float(first_info.get("p_loss_MW", 0.0)))
        vmin_hist.append(float(first_info.get("v_min_step", marl_env.base_env.grid.v_min)))
        vmax_hist.append(float(first_info.get("v_max_step", marl_env.base_env.grid.v_max)))
        action_hist.append(np.concatenate(flat_step_actions, axis=0))
        done = bool(terminateds.get("__all__", False) or truncateds.get("__all__", False))

    return {
        "episode_start": int(getattr(marl_env, "_last_episode_start", episode_start or 0)),
        "v_min": float(marl_env.base_env.grid.v_min),
        "v_max": float(marl_env.base_env.grid.v_max),
        "trace": {
            "reward": np.asarray(reward_hist, dtype=np.float64),
            "cost_continuous": np.asarray(cost_hist, dtype=np.float64),
            "p_loss_MW": np.asarray(loss_hist, dtype=np.float64),
            "v_min_episode": np.asarray(vmin_hist, dtype=np.float64),
            "v_max_episode": np.asarray(vmax_hist, dtype=np.float64),
        },
        "actions": np.asarray(action_hist, dtype=np.float32),
        "rewards": np.asarray(reward_hist, dtype=np.float64),
    }


def _evaluate_ders_policy_bank(
    marl_env,
    policy_bank: dict[str, Any],
    *,
    split: str,
    seed: int,
    n_episodes: int,
    strategy: str = "uniform",
) -> dict[str, Any]:
    """Evaluate one learned DER policy bank against the no-control baseline."""
    from powerzoojax.tasks.ders import compute_ders_metrics, compute_ders_safety_metrics

    per_episode_metrics: list[dict[str, float]] = []
    per_episode_actions: list[np.ndarray] = []
    per_episode_rewards: list[np.ndarray] = []
    per_episode_traces: list[dict[str, np.ndarray]] = []
    episode_returns: list[float] = []

    for episode_idx in range(int(n_episodes)):
        learned = _rollout_ders_policy_bank_once(
            marl_env,
            policy_bank,
            split=split,
            episode_seed=int(seed) * 10_000 + episode_idx,
            episode_idx=episode_idx,
            n_episodes=n_episodes,
            strategy=strategy,
        )
        baseline = _rollout_ders_policy_bank_once(
            marl_env,
            {},
            split=split,
            episode_seed=int(seed) * 10_000 + episode_idx + 50_000,
            n_episodes=1,
            strategy=strategy,
            episode_start=learned["episode_start"],
        )
        metrics = compute_ders_metrics(
            learned["trace"],
            baseline["trace"],
            v_min=learned["v_min"],
            v_max=learned["v_max"],
        )
        metrics.update(
            compute_ders_safety_metrics(
                learned["trace"],
                v_min=learned["v_min"],
                v_max=learned["v_max"],
            )
        )
        metrics["episode_idx"] = float(episode_idx)
        metrics["episode_start"] = float(learned["episode_start"])
        metrics["episode_reward"] = float(np.sum(learned["trace"]["reward"]))
        per_episode_metrics.append(metrics)
        per_episode_actions.append(learned["actions"])
        per_episode_rewards.append(learned["rewards"])
        per_episode_traces.append(learned["trace"])
        episode_returns.append(metrics["episode_reward"])

    metrics = _mean_numeric_dicts(per_episode_metrics)
    if episode_returns:
        metrics["episode_reward"] = float(np.mean(np.asarray(episode_returns, dtype=np.float64)))
    return {
        "metrics": metrics,
        "per_episode_metrics": per_episode_metrics,
        "per_episode_actions": per_episode_actions,
        "per_episode_rewards": per_episode_rewards,
        "per_episode_traces": per_episode_traces,
    }


def _save_ders_episode_traces(
    *,
    per_episode_traces: list[dict[str, np.ndarray]],
    run_id: str,
    artifacts_dir: Path,
) -> dict[str, str]:
    """Persist per-step DER physics traces needed for cross-backend plots/replay."""
    if not per_episode_traces:
        return {}
    payload = {
        "p_loss_MW": np.asarray(
            [np.asarray(trace["p_loss_MW"], dtype=np.float32) for trace in per_episode_traces],
            dtype=np.float32,
        ),
        "cost_continuous": np.asarray(
            [np.asarray(trace["cost_continuous"], dtype=np.float32) for trace in per_episode_traces],
            dtype=np.float32,
        ),
        "v_min_episode": np.asarray(
            [np.asarray(trace["v_min_episode"], dtype=np.float32) for trace in per_episode_traces],
            dtype=np.float32,
        ),
        "v_max_episode": np.asarray(
            [np.asarray(trace["v_max_episode"], dtype=np.float32) for trace in per_episode_traces],
            dtype=np.float32,
        ),
    }
    out_path = Path(artifacts_dir) / f"{run_id}_ders_episode_traces.npz"
    np.savez(out_path, **payload)
    return {"ders_episode_traces": _powerzoo_artifact_rel(Path(artifacts_dir), out_path)}


def _gencos_eval_episodes_for_split(split: str) -> int:
    """Official GenCos eval episode budget for one split."""
    task_dir = _BENCHMARKS_DIR / "gencos"
    eval_cfg_path = task_dir / "configs" / f"eval_{split}.yaml"
    if eval_cfg_path.exists():
        cfg = load_config(eval_cfg_path)
        if cfg.get("n_eval_episodes") is not None:
            return int(cfg["n_eval_episodes"])
    task_cfg = load_task_config(task_dir)
    return int(task_cfg.get("eval_episodes", 30))


def _rollout_gencos_policy_bank_once(
    marl_env,
    policy_bank: dict[str, Any],
    *,
    episode_seed: int,
) -> dict[str, Any]:
    """Evaluate one joint GenCos policy bank on one full PowerZoo episode."""
    from powerzoojax.tasks.gencos import compute_gencos_metrics

    agent_names = list(marl_env.possible_agents)
    obs_dict, _ = marl_env.reset(seed=int(episode_seed))
    profits_hist = {name: [] for name in agent_names}
    gen_cost_hist: list[float] = []
    lmp_hist: list[np.ndarray] = []
    unit_power_hist: list[np.ndarray] = []
    sced_hist: list[float] = []
    ramp_hist: list[float] = []
    action_hist: list[np.ndarray] = []
    reward_hist: list[np.ndarray] = []

    done = False
    while not done:
        actions: dict[str, Any] = {}
        for agent_id in agent_names:
            obs = obs_dict.get(agent_id)
            actions[agent_id] = _predict_policy_action(
                policy_bank.get(agent_id),
                obs,
                _agent_action_space(marl_env, agent_id),
            )
        obs_dict, rewards_d, term_d, trunc_d, info_d = marl_env.step(actions)
        shared_info = info_d.get(agent_names[0], {}) if info_d else {}
        for agent_id in agent_names:
            profits_hist[agent_id].append(float(rewards_d.get(agent_id, 0.0)))
        gen_cost_hist.append(float(shared_info.get("gen_cost", 0.0)))
        lmp_hist.append(
            np.asarray(
                shared_info.get("lmp", np.zeros(getattr(marl_env, "_n_nodes", 1))),
                dtype=np.float32,
            )
        )
        unit_power_hist.append(
            np.asarray(
                shared_info.get(
                    "unit_power",
                    np.zeros(getattr(marl_env, "_n_units", len(agent_names))),
                ),
                dtype=np.float32,
            )
        )
        sced_hist.append(float(shared_info.get("sced_success", 1.0)))
        ramp_hist.append(float(shared_info.get("ramp_binding_rate", 0.0)))
        action_hist.append(
            np.stack(
                [np.asarray(actions[name], dtype=np.float32) for name in agent_names],
                axis=0,
            )
        )
        reward_hist.append(
            np.asarray([rewards_d.get(name, 0.0) for name in agent_names], dtype=np.float32)
        )
        done = bool(term_d.get("__all__", False) or trunc_d.get("__all__", False))

    rollout = {
        "profits": {name: np.asarray(vals, dtype=np.float64) for name, vals in profits_hist.items()},
        "gen_cost": np.asarray(gen_cost_hist, dtype=np.float64),
        "lmp": np.stack(lmp_hist).astype(np.float32),
        "unit_power": np.stack(unit_power_hist).astype(np.float32),
        "sced_converged": np.asarray(sced_hist, dtype=np.float32),
        "ramp_binding_rate": np.asarray(ramp_hist, dtype=np.float32),
    }
    metrics = compute_gencos_metrics(rollout, agent_names)
    return {
        "rollout": rollout,
        "metrics": metrics,
        "actions": np.stack(action_hist).astype(np.float32),
        "rewards": np.stack(reward_hist).astype(np.float32),
    }


def _evaluate_gencos_policy_bank(
    policy_bank: dict[str, Any],
    *,
    split: str,
    seed: int,
    n_episodes: int,
) -> dict[str, Any]:
    """Mean GenCos market metrics from official PowerZoo rollouts."""
    marl_env, kind = _build_powerzoo_env("gencos", split=split, seed=seed)
    if kind != "pettingzoo":
        raise RuntimeError(f"Expected pettingzoo env for GenCos, got {kind!r}")

    per_episode_metrics: list[dict[str, float]] = []
    per_episode_actions: list[np.ndarray] = []
    per_episode_rewards: list[np.ndarray] = []
    try:
        for ep in range(int(n_episodes)):
            out = _rollout_gencos_policy_bank_once(
                marl_env,
                policy_bank,
                episode_seed=seed * 10_000 + ep,
            )
            per_episode_metrics.append(out["metrics"])
            per_episode_actions.append(out["actions"])
            per_episode_rewards.append(out["rewards"])
    finally:
        try:
            marl_env.close()
        except Exception:
            pass

    return {
        "env_meta": _env_contract_metadata(marl_env),
        "metrics": _mean_numeric_dicts(per_episode_metrics),
        "per_episode_metrics": per_episode_metrics,
        "per_episode_actions": per_episode_actions,
        "per_episode_rewards": per_episode_rewards,
    }


def _save_il_models_manifest(
    models: dict[str, Any],
    *,
    run_id: str,
    artifacts_dir: Path,
) -> dict[str, str]:
    """Persist one checkpoint per agent and a manifest that references them."""
    manifest: dict[str, Any] = {"run_id": run_id, "agents": {}}
    for agent_id, model in models.items():
        save_base = artifacts_dir / f"{run_id}_{agent_id}"
        model.save(str(save_base))
        model_path = save_base.with_suffix(".zip")
        manifest["agents"][agent_id] = f"artifacts/{model_path.name}"
    manifest_path = artifacts_dir / f"{run_id}_models_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    rel = f"artifacts/{manifest_path.name}"
    return {
        "models_manifest": rel,
        # Legacy consumers still probe ``artifacts['params']`` to decide
        # whether a record is a training-class row.
        "params": rel,
    }


def _save_single_agent_model(
    model: Any,
    *,
    run_id: str,
    artifacts_dir: Path,
) -> dict[str, str]:
    """Persist one SB3/SBX single-agent checkpoint bundle.

    Phase-2 cross-backend rows need more than just curve arrays: the paper
    pipeline and later audit/replay steps should be able to point back to a
    concrete serialized model artifact. We expose it under both ``model_zip``
    and legacy ``params`` so training-row detection stays uniform.
    """
    save_base = artifacts_dir / f"{run_id}_model"
    model.save(str(save_base))
    model_path = save_base.with_suffix(".zip")
    rel = f"artifacts/{model_path.name}"
    return {
        "model_zip": rel,
        "params": rel,
    }


def _model_vecnormalize_env(model: Any) -> Any | None:
    """Return an attached SB3 VecNormalize wrapper, if the model has one."""
    getter = getattr(model, "get_vec_normalize_env", None)
    if not callable(getter):
        return None
    try:
        return getter()
    except Exception:
        return None


def _normalize_obs_for_model(model: Any, obs: np.ndarray) -> np.ndarray:
    """Normalize raw single-env observations with the model's VecNormalize stats."""
    vecnorm = _model_vecnormalize_env(model)
    if vecnorm is None:
        return np.asarray(obs, dtype=np.float32)
    obs_arr = np.asarray(obs, dtype=np.float32)
    batched = obs_arr.reshape((1,) + obs_arr.shape)
    try:
        normalized = vecnorm.normalize_obs(batched)
    except Exception:
        return obs_arr
    return np.asarray(normalized[0], dtype=np.float32)


def _save_vecnormalize_stats(
    model: Any,
    *,
    run_id: str,
    artifacts_dir: Path,
) -> dict[str, str]:
    """Persist VecNormalize observation statistics for replay/audit."""
    vecnorm = _model_vecnormalize_env(model)
    if vecnorm is None:
        return {}
    stats_path = artifacts_dir / f"{run_id}_vecnormalize.pkl"
    vecnorm.save(str(stats_path))
    return {"vecnormalize": f"artifacts/{stats_path.name}"}


def _maybe_wrap_vecnormalize(
    env: Any,
    *,
    enabled: bool,
) -> tuple[Any, bool]:
    """Apply SB3 VecNormalize to vectorized single-agent envs when requested."""
    if not enabled:
        return env, False
    try:
        from stable_baselines3.common.vec_env import VecEnv, VecNormalize
    except Exception:
        return env, False
    if isinstance(env, VecNormalize):
        env.training = True
        env.norm_reward = False
        return env, True
    if not isinstance(env, VecEnv):
        return env, False
    return VecNormalize(env, norm_obs=True, norm_reward=False, training=True), True


def _save_ders_eval_record(
    *,
    algorithm: str,
    backend: str,
    device: str,
    framework_version: str,
    seed: int,
    split: str,
    source_run_id: str,
    eval_result: dict[str, Any],
    task_dir: Path,
) -> RunRecord:
    """Write one official DER cross-backend eval record."""
    eval_run_id = make_run_id("ders", algorithm.lower(), split, seed)
    artifacts_dir = task_dir / "results" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics = dict(eval_result["metrics"])
    if metrics.get("episode_reward") is not None:
        metrics["final_return"] = float(metrics["episode_reward"])
    artifact_paths = save_eval_artifacts(
        per_episode_metrics=eval_result["per_episode_metrics"],
        run_id=eval_run_id,
        split=split,
        artifacts_dir=artifacts_dir,
        per_episode_actions=eval_result["per_episode_actions"],
        per_episode_rewards=eval_result["per_episode_rewards"],
    )
    artifact_paths.update(
        _save_ders_episode_traces(
            per_episode_traces=eval_result["per_episode_traces"],
            run_id=eval_run_id,
            artifacts_dir=artifacts_dir,
        )
    )
    task_cfg = load_task_config(task_dir)
    env_info, labels = _collect_torch_run_contract(
        requested_device=device,
        context="powerzoo_bridge/ders_eval_record",
        meta=collect_dataset_provenance(
            task="ders", task_config=task_cfg, split=split
        ),
        fail_fast=False,
        extra_labels={
            "record_kind": "eval",
            "backend_family": "python_torch",
            "algo_family": "frozen_self_play_il",
            "rl_paradigm": "scalable_safe_marl_cooperative",
            "split_status": "aligned",
            "cross_backend_gap_note": "frozen_self_play_il",
            "source_run_id": source_run_id,
        },
    )
    record = RunRecord(
        task="ders",
        variant="ders_cross_backend_pettingzoo_il_eval",
        algo=algorithm.lower(),
        seed=int(seed),
        run_id=eval_run_id,
        config_hash=config_hash({"source_run_id": source_run_id, "split": split}),
        status="completed",
        split=split,
        backend=backend,
        device=device,
        framework_version=framework_version,
        metrics=metrics,
        notes=f"cross-backend eval of {source_run_id} on split={split}",
        env_info=env_info,
        labels=labels,
        artifacts=artifact_paths,
    )
    save_run(record, task_dir)
    return record


def _save_gencos_eval_record(
    *,
    algorithm: str,
    backend: str,
    device: str,
    framework_version: str,
    seed: int,
    split: str,
    source_run_id: str,
    eval_result: dict[str, Any],
    task_dir: Path,
) -> RunRecord:
    """Write one official GenCos cross-backend eval record."""
    eval_run_id = make_run_id("gencos", algorithm.lower(), split, seed)
    artifacts_dir = task_dir / "results" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics = dict(eval_result["metrics"])
    env_meta = dict(eval_result.get("env_meta") or {})
    if metrics.get("total_profit") is not None:
        metrics["final_return"] = float(metrics["total_profit"])
    artifact_paths = save_eval_artifacts(
        per_episode_metrics=eval_result["per_episode_metrics"],
        run_id=eval_run_id,
        split=split,
        artifacts_dir=artifacts_dir,
        per_episode_actions=eval_result["per_episode_actions"],
        per_episode_rewards=eval_result["per_episode_rewards"],
    )
    env_info, labels = _collect_torch_run_contract(
        requested_device=device,
        context="powerzoo_bridge/gencos_eval_record",
        meta={
            **collect_dataset_provenance(
                task="gencos", task_config=load_task_config(task_dir), split=split
            ),
            **env_meta,
        },
        fail_fast=False,
        extra_labels={
            "record_kind": "eval",
            "backend_family": "python_torch",
            "algo_family": "frozen_self_play_il",
            "rl_paradigm": "multi_agent_competitive",
            "split_status": "aligned",
            "cross_backend_gap_note": "frozen_self_play_il",
            "source_run_id": source_run_id,
        },
    )
    record = RunRecord(
        task="gencos",
        variant="gencos_cross_backend_pettingzoo_il",
        algo=algorithm.lower(),
        seed=int(seed),
        run_id=eval_run_id,
        config_hash=config_hash({"source_run_id": source_run_id, "split": split}),
        status="completed",
        split=split,
        backend=backend,
        device=device,
        framework_version=framework_version,
        metrics=metrics,
        notes=(
            f"cross-backend eval of {source_run_id} on split={split}"
            + (f" env_meta={env_meta}." if env_meta else "")
        ),
        env_info=env_info,
        labels=labels,
        artifacts=artifact_paths,
    )
    save_run(record, task_dir)
    return record


def _evaluate_dc_model(
    model: Any,
    *,
    split: str,
    seed: int,
    n_episodes: int,
) -> dict[str, Any]:
    from powerzoojax.tasks.dc_microgrid import compute_dcmicrogrid_metrics

    per_episode_metrics: list[dict[str, float]] = []
    per_episode_actions: list[np.ndarray] = []
    per_episode_rewards: list[np.ndarray] = []

    for episode_idx in range(int(n_episodes)):
        params = _make_jax_dc_episode_params(
            split,
            seed=seed,
            episode_idx=episode_idx,
            n_episodes=n_episodes,
            strategy="uniform",
        )
        env = _wrap_powerzoo_reward_shaping(
            "dc_microgrid",
            _powerzoo_dc_env_from_jax_params(params),
        )
        obs, _ = env.reset(seed=seed * 10_000 + episode_idx)
        done = False
        truncated = False
        info_history: list[dict[str, float]] = []
        actions: list[np.ndarray] = []
        rewards: list[float] = []
        while not (done or truncated):
            policy_obs = _normalize_obs_for_model(model, obs)
            action, _ = model.predict(policy_obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            actions.append(np.asarray(action, dtype=np.float32).reshape(-1))
            rewards.append(float(reward))
            info_history.append(_scalarize_info(dict(info), float(reward)))
        try:
            env.close()
        except Exception:
            pass
        per_episode_metrics.append(compute_dcmicrogrid_metrics(info_history))
        per_episode_actions.append(np.asarray(actions, dtype=np.float32))
        per_episode_rewards.append(np.asarray(rewards, dtype=np.float64))

    return {
        "metrics": _mean_numeric_dicts(per_episode_metrics),
        "per_episode_metrics": per_episode_metrics,
        "per_episode_actions": per_episode_actions,
        "per_episode_rewards": per_episode_rewards,
    }


def _evaluate_dso_model(
    model: Any,
    *,
    split: str,
    seed: int,
    n_episodes: int,
    task_cfg: dict[str, Any] | None = None,
    episode_starts: list[int] | None = None,
) -> dict[str, Any]:
    """Run the PowerZoo DSO env on fixed windows and compute benchmark metrics."""
    _ensure_powerzoo_path()
    from powerzoo.tasks.dso_task import (
        compute_dso_metrics,
        dso_no_control_rollout,
        make_dso_env,
        rollout_dso,
    )

    cfg = dict(_dso_task_config() if task_cfg is None else task_cfg)
    max_steps = int(cfg.get("max_steps", 48))
    starts = (
        _dso_eval_episode_starts(
            split,
            n_episodes=n_episodes,
            max_steps=max_steps,
            task_cfg=cfg,
        )
        if episode_starts is None
        else [int(x) for x in episode_starts]
    )
    env_kwargs = _powerzoo_dso_env_kwargs(
        split,
        seed=seed,
        task_cfg=cfg,
        use_train_reset_bank=False,
    )

    per_episode_metrics: list[dict[str, float]] = []
    per_episode_rewards: list[np.ndarray] = []
    for episode_idx, start in enumerate(starts):
        env = make_dso_env(**env_kwargs, episode_start=int(start))
        rollout_seed = int(seed) * 10_000 + int(episode_idx)

        def _policy(obs):
            policy_obs = _normalize_obs_for_model(model, obs)
            action, _ = model.predict(policy_obs, deterministic=True)
            return np.asarray(action, dtype=np.float32).reshape(-1)

        agent_rollout = rollout_dso(env, _policy, n_steps=max_steps, seed=rollout_seed)
        baseline_rollout = dso_no_control_rollout(env, seed=rollout_seed)
        per_episode_metrics.append(compute_dso_metrics(agent_rollout, baseline_rollout))
        per_episode_rewards.append(
            np.asarray(agent_rollout["rewards"], dtype=np.float64).reshape(-1)
        )
        try:
            env.close()
        except Exception:
            pass

    return {
        "metrics": _mean_numeric_dicts(per_episode_metrics),
        "per_episode_metrics": per_episode_metrics,
        "per_episode_rewards": per_episode_rewards,
    }


def _evaluate_tso_model(
    model: Any,
    *,
    split: str,
    seed: int,
    episode_starts: list[int],
) -> dict[str, Any]:
    """Run the PowerZoo TSO env on explicit episode starts and aggregate metrics.

    Cross-backend TSO phase-2 learning curves are a **train-monitor** artifact.
    We therefore evaluate on the training split with explicit 48-step windows,
    keeping the monitor semantics separate from the formal IID eval rows.
    """
    _ensure_powerzoo_path()
    from powerzoo.tasks.middle.comparison_tso import (
        CentralizedComparisonTSOEnv,
    )

    per_episode_metrics: list[dict[str, float]] = []
    per_episode_rewards: list[np.ndarray] = []

    for episode_idx, episode_start in enumerate(episode_starts):
        task, profile_window = _instantiate_powerzoo_tso_task(
            split,
            episode_start_idx=int(episode_start),
        )
        env = CentralizedComparisonTSOEnv(task)
        _annotate_tso_powerzoo_env(
            env,
            split=split,
            profile_window=profile_window,
        )
        obs, _ = env.reset(seed=int(seed) * 10_000 + int(episode_idx))
        done = False
        truncated = False
        rewards: list[float] = []
        operating_cost_total = 0.0
        reserve_shortfall_total = 0.0
        thermal_overload_total = 0.0
        reserve_shortfall_steps = 0
        thermal_violation_steps = 0
        length = 0
        while not (done or truncated):
            policy_obs = _normalize_obs_for_model(model, obs)
            action, _ = model.predict(policy_obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            rewards.append(float(reward))
            operating_cost_step = float(info.get("operating_cost", 0.0))
            reserve_shortfall_step = float(info.get("reserve_shortfall", 0.0))
            thermal_overload_step = float(info.get("cost_thermal_overload", 0.0))
            operating_cost_total += operating_cost_step
            reserve_shortfall_total += reserve_shortfall_step
            thermal_overload_total += thermal_overload_step
            reserve_shortfall_steps += int(reserve_shortfall_step > 1e-6)
            thermal_violation_steps += int(thermal_overload_step > 1e-6)
            length += 1
        try:
            env.close()
        except Exception:
            pass
        length_f = float(max(length, 1))
        per_episode_metrics.append(
            {
                "episode_reward": float(np.sum(np.asarray(rewards, dtype=np.float64))),
                "total_operating_cost": float(operating_cost_total),
                "total_reserve_shortfall": float(reserve_shortfall_total),
                "total_thermal_overload": float(thermal_overload_total),
                "reserve_shortfall_rate": float(reserve_shortfall_steps / length_f),
                "thermal_violation_rate": float(thermal_violation_steps / length_f),
            }
        )
        per_episode_rewards.append(np.asarray(rewards, dtype=np.float64))

    return {
        "metrics": _mean_numeric_dicts(per_episode_metrics),
        "per_episode_metrics": per_episode_metrics,
        "per_episode_rewards": per_episode_rewards,
    }

# Public dry-run API

def smoke_load_powerzoo_task(jax_task: str, *, split: str = "train") -> dict[str, Any]:
    """Build the env and return diagnostic info; does not train."""
    entry = _resolve_task(jax_task, allow_unknown=True)
    t0 = time.perf_counter()
    handle, kind = _build_powerzoo_env(jax_task, split=split, dry_run_steps=10)
    load_s = time.perf_counter() - t0

    if kind == "single":
        obs_desc = repr(getattr(handle, "observation_space", None))[:120]
        act_desc = repr(getattr(handle, "action_space", None))[:120]
        n_agents = 1
    else:  # pettingzoo
        agents = list(handle.possible_agents)
        n_agents = len(agents)
        a0 = agents[0]
        obs_desc = f"per_agent obs={_agent_obs_space(handle, a0)}"
        act_desc = f"per_agent action={_agent_action_space(handle, a0)}"

    diag = {
        "jax_task": jax_task,
        "powerzoo_task": entry["powerzoo_task"],
        "split": split,
        "access_path": entry["access_path"],
        "framework": entry["framework"],
        "rl_paradigm": entry["rl_paradigm"],
        "n_agents": n_agents,
        "env_class": type(handle).__name__,
        "obs": obs_desc,
        "action": act_desc,
        "load_time_s": round(load_s, 3),
        "status": "load_ok",
    }
    diag.update(_env_contract_metadata(handle))
    return diag

# Public training API

def _algo_class(backend: str, algorithm: str):
    """Map (backend, algorithm) -> SB3/SBX class. Lazy-imports."""
    from powerzoo.rl import Trainer
    Trainer("battery_arbitrage", total_timesteps=10)  # populate ALGORITHMS
    base = algorithm.upper()
    if backend == "sbx" and not base.startswith("SBX_"):
        base = f"SBX_{base}"
    cls = Trainer.ALGORITHMS.get(base)
    if cls is None:
        raise ValueError(
            f"Algorithm {base!r} not registered in Trainer.ALGORITHMS "
            f"(known: {sorted(Trainer.ALGORITHMS.keys())})"
        )
    return cls

def _base_algo_name(algorithm: str) -> str:
    """Normalize backend-specific algo keys to the benchmark algo name."""
    base = algorithm.upper()
    if base.startswith("SBX_"):
        base = base[4:]
    return base.lower()

def _load_benchmark_train_config(
    jax_task: str,
    algorithm: str,
) -> tuple[Path, dict[str, Any]]:
    """Load the frozen benchmark train config for one task/algo pair."""
    algo = _base_algo_name(algorithm)
    task_dir = _BENCHMARKS_DIR / jax_task
    algo_key_map = {"ppo_lagrangian": "safe"}
    allowed_algos = ("ppo", "sac") if jax_task == "dc_microgrid" else None
    cfg = load_train_config(
        task_dir,
        algo,
        None,
        algo_key_map=algo_key_map,
        allowed_algos=allowed_algos,
    )
    resolved_key = algo_key_map.get(algo, algo)
    return task_dir / "configs" / f"train_{resolved_key}.yaml", cfg

def _resolved_total_timesteps(
    jax_task: str,
    algorithm: str,
    total_timesteps: int | None,
) -> int:
    """Resolve the effective env-step budget for one cross-backend run."""
    if total_timesteps is not None:
        return int(total_timesteps)
    try:
        _, cfg = _load_benchmark_train_config(jax_task, algorithm)
    except Exception:
        return 200_000
    if "total_timesteps" in cfg:
        return int(cfg["total_timesteps"])
    return 200_000


def _tso_curve_n_checkpoints(
    raw_train_cfg: dict[str, Any] | None,
    total_timesteps: int,
) -> int:
    """Match TSO cross-backend monitor density to ``train_ppo.yaml::eval_freq``.

    The Phase-2 TSO walltime plot compares train-monitor curves across JAX /
    SB3 / SBX, so the Python backend must not silently collapse to a coarse
    fixed-20 checkpoint schedule. We derive the checkpoint count from the
    frozen benchmark train config's ``eval_freq`` exactly as the JAX training
    loop does conceptually: one monitor record per ``eval_freq`` env steps.
    """
    cfg = raw_train_cfg or {}
    eval_freq = int(cfg.get("eval_freq", 100_000))
    return max(int(total_timesteps) // max(eval_freq, 1), 1)

def _merge_algo_kwargs(
    defaults: dict[str, Any],
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    """Overlay explicit overrides on aligned defaults, merging policy kwargs."""
    merged = dict(defaults)
    if not overrides:
        return merged
    for key, value in overrides.items():
        if (
            key == "policy_kwargs"
            and isinstance(merged.get(key), dict)
            and isinstance(value, dict)
        ):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged

def _aligned_single_agent_algo_kwargs(
    jax_task: str,
    algorithm: str,
    *,
    n_envs: int,
) -> tuple[dict[str, Any], str | None]:
    """Return task-aligned PPO kwargs for SB3/SBX when a frozen config exists."""
    if _base_algo_name(algorithm) != "ppo":
        return {}, None
    try:
        cfg_path, raw_cfg = _load_benchmark_train_config(jax_task, algorithm)
    except Exception:
        return {}, None
    train_cfg = build_train_cfg(raw_cfg, algo="ppo")
    rollout_batch = max(int(n_envs), 1) * int(train_cfg.n_steps)
    n_minibatches = max(int(train_cfg.n_minibatches), 1)
    batch_size = max(1, rollout_batch // n_minibatches)
    kwargs: dict[str, Any] = {
        "learning_rate": float(train_cfg.learning_rate),
        "n_steps": int(train_cfg.n_steps),
        "batch_size": int(batch_size),
        "n_epochs": int(train_cfg.n_epochs),
        "gamma": float(train_cfg.gamma),
        "gae_lambda": float(train_cfg.gae_lambda),
        "clip_range": float(train_cfg.clip_eps),
        "ent_coef": float(train_cfg.ent_coef),
        "vf_coef": float(train_cfg.vf_coef),
        "max_grad_norm": float(train_cfg.max_grad_norm),
    }
    if train_cfg.hidden_dims:
        kwargs["policy_kwargs"] = {"net_arch": list(train_cfg.hidden_dims)}
    return kwargs, str(cfg_path.relative_to(_REPO_ROOT))

def _python_backend_stability_overrides(
    jax_task: str,
    algorithm: str,
) -> dict[str, Any]:
    """Small backend-specific PPO stabilizers for known cross-backend gaps."""
    return {}

def _requested_continuous_action_dist(
    jax_task: str,
    algorithm: str,
) -> str | None:
    """Return the frozen benchmark's requested continuous actor family."""
    try:
        _, raw_cfg = _load_benchmark_train_config(jax_task, algorithm)
    except Exception:
        return None
    value = raw_cfg.get("continuous_action_dist")
    if value is None:
        return None
    return str(value).strip().lower()

def _single_agent_policy_spec(
    jax_task: str,
    algorithm: str,
    *,
    backend: str,
) -> tuple[str | type[Any], dict[str, Any]]:
    """Resolve the concrete policy class for one single-agent bridge run."""
    requested_dist = _requested_continuous_action_dist(jax_task, algorithm)
    metadata = {
        "requested_continuous_action_dist": requested_dist or "gaussian",
        "policy_class": "MlpPolicy",
        "effective_continuous_action_dist": "gaussian",
    }
    if (
        jax_task == "dc_microgrid"
        and _base_algo_name(algorithm) == "ppo"
        and requested_dist == "beta"
    ):
        if backend == "sb3":
            from benchmarks.common.bounded_beta_policy import SB3BoundedBetaPolicy

            metadata["policy_class"] = "SB3BoundedBetaPolicy"
            metadata["effective_continuous_action_dist"] = "beta"
            return SB3BoundedBetaPolicy, metadata
        if backend == "sbx":
            from benchmarks.common.bounded_beta_sbx_policy import SBXPPOBoundedBetaPolicy

            metadata["policy_class"] = "SBXPPOBoundedBetaPolicy"
            metadata["effective_continuous_action_dist"] = "beta"
            return SBXPPOBoundedBetaPolicy, metadata
    return "MlpPolicy", metadata

def _single_agent_algo_kwargs(
    jax_task: str,
    algorithm: str,
    *,
    n_envs: int,
    extra_config: dict[str, Any] | None,
) -> tuple[dict[str, Any], str | None]:
    """Constructor kwargs for single-agent SB3/SBX runs.

    VecEnv collection speeds Python backends up, but PPO's default rollout
    length is per environment.  If we increase ``n_envs`` without shortening
    ``n_steps``, the update batch grows by the same factor and the run gets
    fewer optimizer updates for the same total timestep budget.  Keep the
    default aggregate rollout batch near 2048 unless the caller explicitly
    overrides ``n_steps``.
    """
    excluded = {
        "il_self_play_rounds",
        "per_agent_steps_per_round",
        "curve_eval_episodes",
        "train_split",
        "vec_env",
        "normalize_observations",
        "model_seed_offset",
    }
    overrides = {
        k: v for k, v in (extra_config or {}).items()
        if k not in excluded
    }
    kwargs, aligned_source = _aligned_single_agent_algo_kwargs(
        jax_task,
        algorithm,
        n_envs=n_envs,
    )
    kwargs = _merge_algo_kwargs(
        kwargs,
        _python_backend_stability_overrides(jax_task, algorithm),
    )
    if "PPO" in algorithm.upper() and int(n_envs) > 1 and "n_steps" not in overrides and "n_steps" not in kwargs:
        kwargs["n_steps"] = max(64, 2048 // int(n_envs))
    return _merge_algo_kwargs(kwargs, overrides), aligned_source

def _write_jax_train_override_config(
    jax_task: str,
    algorithm: str,
    overrides: dict[str, Any],
) -> Path:
    """Write a temporary benchmark train config with the requested overrides."""
    cfg_path, cfg = _load_benchmark_train_config(jax_task, algorithm)
    merged = dict(cfg)
    merged.update(overrides)
    suffix = cfg_path.suffix or ".yaml"
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix=f"pzjx_bridge_{jax_task}_{_base_algo_name(algorithm)}_",
        suffix=suffix,
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
    dump_yaml(merged, tmp_path)
    return tmp_path

def train_with_powerzoo(
    jax_task: str,
    algorithm: str,
    seed: int,
    split: str,
    device: str = "cuda",
    total_timesteps: int | None = None,
    n_envs: int | None = None,
    extra_config: dict[str, Any] | None = None,
) -> RunRecord:
    """Train one PowerZoo run, write a PowerZooJax RunRecord.

    Budget semantics
    ----------------
    For single-agent tasks (DSO / TSO / DC Microgrid) ``total_timesteps`` is
    the SB3 ``learn(total_timesteps=...)`` budget directly.

    For PettingZoo IL tasks (DERs / GenCos), ``total_timesteps`` is the
    aggregate budget that gets divided into per-agent / per-round slices:
    ``per_agent_ts_per_round = max(total_timesteps // (n_agents * n_rounds), 1_000)``.
    For paper-table comparison against parameter-shared IPPO (jax_rejax)
    the right axis is ``steps-to-target``, NOT walltime-to-target — IL
    and parameter-shared MARL have fundamentally different sample
    efficiency, and forcing the same wall-clock budget on both would
    misrepresent the cross-library comparison.  See
    ``benchmarks/HARDWARE.md`` and the cross-backend supplementary
    section of the paper for the full discussion.

    To override the slicing (e.g. to make per-agent budget large enough
    for the SB3 IL baseline to actually learn), pass
    ``extra_config = {"per_agent_steps_per_round": 200_000,
                       "il_self_play_rounds": 4}`` — these take
    precedence over the default split of ``total_timesteps``.

    Parameters
    ----------
    jax_task : str
        PowerZooJax task name.
    algorithm : str
        ``"PPO" / "SAC" / "TD3"`` (SB3) or ``"SBX_PPO" / ...`` (SBX).
    seed, split : ``split`` names the requested report/eval split. Tasks with
        a canonical train/eval separation (DER phase-2) still train on
        ``train_split`` and emit official eval records on the requested split(s).
    device : ``"cuda"`` (default) or ``"cpu"``.
    total_timesteps : env interaction budget.
    n_envs : number of parallel single-agent envs for SB3/SBX VecEnv
             collection. ``None`` / ``<= 0`` resolves to the frozen benchmark
             train config for that task/algo.
    extra_config : merged into hyperparams (passed to SB3/SBX constructor).
        Recognised keys for IL path: ``per_agent_steps_per_round`` (int),
        ``il_self_play_rounds`` (int).
    """
    _ensure_powerzoo_path()
    entry = _resolve_task(jax_task)
    backend = "sbx" if algorithm.upper().startswith("SBX_") else "sb3"

    # Resolve algorithm class via PowerZoo's registry side-effect.
    algo_cls = _algo_class(backend, algorithm)

    # Capture the framework version string for the record.
    framework_version = ""
    try:
        if backend == "sbx":
            import sbx
            framework_version = f"sbx-{sbx.__version__}"
        else:
            import stable_baselines3 as sb3
            framework_version = f"sb3-{sb3.__version__}"
    except Exception:
        pass

    benchmark_train_cfg_path: Path | None = None
    benchmark_train_cfg: dict[str, Any] = {}
    try:
        benchmark_train_cfg_path, benchmark_train_cfg = _load_benchmark_train_config(
            jax_task, algorithm
        )
    except Exception:
        benchmark_train_cfg = {}

    resolved_total_timesteps = _resolved_total_timesteps(
        jax_task,
        algorithm,
        total_timesteps,
    )
    if jax_task == "dc_microgrid":
        task_cfg = _dc_task_config()
    elif jax_task == "ders":
        task_cfg = _ders_task_config()
    elif jax_task == "dso":
        task_cfg = _dso_task_config()
    else:
        task_cfg = {}
    cfg_overrides = {**benchmark_train_cfg, **dict(extra_config or {})}
    train_split = str(cfg_overrides.get("train_split", "train"))
    final_eval_split = split
    model_seed = int(seed) + int(cfg_overrides.get("model_seed_offset", 0) or 0)
    resolved_n_envs = (
        _resolved_single_agent_n_envs(jax_task, algorithm, n_envs)
        if entry["n_agents"] == 1
        else int(n_envs or 0)
    )
    cfg: dict[str, Any] = dict(
        algorithm=algorithm,
        total_timesteps=int(resolved_total_timesteps),
        seed=int(seed),
        model_seed=int(model_seed),
        split=split,
        requested_split=split,
        train_split=train_split,
        eval_split=final_eval_split,
        device=device,
        n_envs=int(resolved_n_envs),
        access_path=entry["access_path"],
        rl_paradigm=entry["rl_paradigm"],
        n_agents=entry["n_agents"],
    )
    shaping_weights = _powerzoo_reward_shaping_weights(jax_task)
    if shaping_weights:
        cfg["reward_shaping_weights"] = shaping_weights
    algo_kwargs, aligned_source = _single_agent_algo_kwargs(
        jax_task,
        algorithm,
        n_envs=max(int(resolved_n_envs), 1),
        extra_config=extra_config,
    )
    if algo_kwargs:
        cfg["algo_kwargs"] = algo_kwargs
    stability_overrides = _python_backend_stability_overrides(jax_task, algorithm)
    if stability_overrides:
        cfg["python_backend_stability_overrides"] = stability_overrides
    if aligned_source is not None:
        cfg["aligned_from_train_config"] = aligned_source
    policy_spec, policy_metadata = _single_agent_policy_spec(
        jax_task,
        algorithm,
        backend=backend,
    )
    cfg.update(policy_metadata)
    requested_normalize_observations = bool(
        cfg_overrides.get("normalize_observations", False)
    )
    if "normalize_observations" in benchmark_train_cfg:
        cfg["benchmark_normalize_observations"] = bool(
            benchmark_train_cfg.get("normalize_observations", False)
        )
    cfg["requested_normalize_observations"] = requested_normalize_observations
    cfg["effective_normalize_observations"] = False
    if extra_config:
        cfg.update(extra_config)
    if benchmark_train_cfg_path is not None:
        cfg["benchmark_train_config"] = str(benchmark_train_cfg_path.relative_to(_REPO_ROOT))
    base_labels = _cross_backend_gap_labels(
        jax_task,
        entry,
        train_split=train_split,
        eval_split=final_eval_split,
        policy_metadata=policy_metadata,
    )

    task_dir = _BENCHMARKS_DIR / jax_task
    artifacts_dir = task_dir / "results" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[powerzoo_driver] start jax_task={jax_task} pz={entry['powerzoo_task']} "
        f"path={entry['access_path']} backend={backend} algo={algorithm} "
        f"seed={seed} train_split={train_split} eval_split={split} "
        f"device={device} timesteps={resolved_total_timesteps}"
    )
    _collect_torch_run_contract(
        requested_device=device,
        context=f"powerzoo_driver/{jax_task}",
        meta=collect_dataset_provenance(
            task=jax_task,
            task_config=load_task_config(task_dir),
            split=train_split,
        ),
        extra_labels={**base_labels, "record_kind": "train"},
    )

    handle, kind = _build_powerzoo_env(
        jax_task,
        split=train_split,
        seed=seed,
        episode_idx=0,
        n_episodes=1,
        strategy="seeded" if jax_task == "dc_microgrid" else "uniform",
    )
    initial_env_meta = _env_contract_metadata(handle)
    if kind == "single" and int(resolved_n_envs) > 1:
        try:
            handle.close()
        except Exception:
            pass
        print(f"[powerzoo_driver] using VecEnv n_envs={int(resolved_n_envs)}")
        handle = _build_powerzoo_vec_env(
            jax_task,
            split=train_split,
            seed=seed,
            n_envs=int(resolved_n_envs),
            strategy="seeded" if jax_task == "dc_microgrid" else "uniform",
            vec_env=str((extra_config or {}).get("vec_env", "auto")),
        )
    elif kind == "single":
        handle = _wrap_single_agent_monitor(handle)
    if kind == "single":
        handle, effective_normalize_observations = _maybe_wrap_vecnormalize(
            handle,
            enabled=requested_normalize_observations,
        )
        cfg["effective_normalize_observations"] = bool(
            effective_normalize_observations
        )
        if effective_normalize_observations:
            cfg["vec_normalize"] = {
                "norm_obs": True,
                "norm_reward": False,
                "training": True,
            }
    record_split = _cross_backend_record_split(
        jax_task,
        requested_split=split,
        train_split=train_split,
        env_kind=kind,
    )
    cfg["split"] = record_split
    run_id = make_run_id(jax_task, algorithm.lower(), record_split, seed)
    t0 = time.perf_counter()
    train_curve_walltimes: list[float] | None = None
    dc_eval_result = None
    ders_train_eval_result = None
    dso_eval_result = None
    ders_eval_records: list[RunRecord] = []
    gencos_train_eval_result = None
    gencos_eval_records: list[RunRecord] = []
    il_model_artifacts: dict[str, str] = {}
    single_model_artifacts: dict[str, str] = {}

    # ── single-agent path: one SB3/SBX model on the env directly ───────
    if kind == "single":
        curve_n_checkpoints = 20
        curve_eval_fn = None
        if jax_task == "dc_microgrid":
            try:
                _, raw_train_cfg = _load_benchmark_train_config(jax_task, algorithm)
            except Exception:
                raw_train_cfg = {}
            eval_freq = int(raw_train_cfg.get("eval_freq", 50_000))
            curve_n_checkpoints = max(int(resolved_total_timesteps) // max(eval_freq, 1), 1)
            curve_eval_episodes = int(
                raw_train_cfg.get("eval_num_episodes", task_cfg.get("eval_episodes", 10))
            )

            def _curve_eval(model_obj, _num_timesteps: int) -> float | None:
                eval_out = _evaluate_dc_model(
                    model_obj,
                    split=train_split,
                    seed=seed,
                    n_episodes=curve_eval_episodes,
                )
                return eval_out["metrics"].get("episode_reward")

            curve_eval_fn = _curve_eval
        elif jax_task == "dso":
            try:
                _, raw_train_cfg = _load_benchmark_train_config(jax_task, algorithm)
            except Exception:
                raw_train_cfg = {}
            curve_eval_episodes = int(raw_train_cfg.get("eval_episodes", 8))
            curve_eval_starts = _dso_eval_episode_starts(
                train_split,
                n_episodes=curve_eval_episodes,
                max_steps=int(task_cfg.get("max_steps", 48)),
                task_cfg=task_cfg,
            )

            def _curve_eval(model_obj, _num_timesteps: int) -> dict[str, float]:
                eval_out = _evaluate_dso_model(
                    model_obj,
                    split=train_split,
                    seed=seed,
                    n_episodes=curve_eval_episodes,
                    task_cfg=task_cfg,
                    episode_starts=curve_eval_starts,
                )
                metrics = eval_out["metrics"]
                return {
                    "total_reward": float(metrics.get("total_reward", 0.0)),
                    "voltage_violation_count_per_step": float(
                        metrics.get("voltage_violation_count_per_step", 0.0)
                    ),
                }

            curve_eval_fn = _curve_eval
        elif jax_task == "tso":
            try:
                _, raw_train_cfg = _load_benchmark_train_config(jax_task, algorithm)
            except Exception:
                raw_train_cfg = {}
            curve_n_checkpoints = _tso_curve_n_checkpoints(
                raw_train_cfg,
                int(resolved_total_timesteps),
            )
            curve_eval_episodes = int(raw_train_cfg.get("eval_episodes", 8))
            task_cfg = load_task_config(_BENCHMARKS_DIR / "tso")
            max_steps = int(task_cfg.get("max_steps", 48))
            from benchmarks.tso.config_runtime import make_task_from_config

            tso_task = make_task_from_config(task_cfg)
            train_params = tso_task.training_params(max_steps=max_steps)
            train_profile_len = int(np.asarray(train_params.load_profiles).shape[0])
            max_start = max(train_profile_len - max_steps, 0)

            def _curve_eval(model_obj, num_timesteps: int) -> dict[str, float]:
                rng = np.random.default_rng(
                    int(seed) * 1_000_003 + int(num_timesteps) * 17 + 11
                )
                if max_start > 0:
                    starts = [
                        int(rng.integers(0, max_start + 1))
                        for _ in range(curve_eval_episodes)
                    ]
                else:
                    starts = [0 for _ in range(curve_eval_episodes)]
                eval_out = _evaluate_tso_model(
                    model_obj,
                    split=train_split,
                    seed=seed,
                    episode_starts=starts,
                )
                return eval_out["metrics"]

            curve_eval_fn = _curve_eval

        curve_cb = _TrainCurveCallback(
            int(resolved_total_timesteps),
            n_checkpoints=curve_n_checkpoints,
            eval_fn=curve_eval_fn,
        )
        model = algo_cls(
            policy_spec,
            handle,
            verbose=0,
            seed=model_seed,
            device=device,
            **algo_kwargs,
        )
        model.learn(
            total_timesteps=int(resolved_total_timesteps),
            progress_bar=False,
            callback=curve_cb,
        )
        walltime = time.perf_counter() - t0
        single_model_artifacts = _save_single_agent_model(
            model,
            run_id=run_id,
            artifacts_dir=artifacts_dir,
        )
        single_model_artifacts.update(
            _save_vecnormalize_stats(
                model,
                run_id=run_id,
                artifacts_dir=artifacts_dir,
            )
        )

        # Try to extract ep_rew_mean from SB3 logger
        ep_rew_mean = _extract_ep_rew_mean(model)
        result_metrics: dict[str, Any] = {}
        train_curve_walltimes: list[float] | None = None
        if curve_cb.train_returns:
            result_metrics["ep_rew_mean"] = np.asarray(
                curve_cb.train_returns, dtype=np.float64
            )
            result_metrics["eval_timesteps"] = np.asarray(
                curve_cb.timesteps, dtype=np.float64
            )
            if curve_cb.eval_returns:
                result_metrics["eval_returns"] = np.asarray(
                    curve_cb.eval_returns, dtype=np.float64
                )
            if curve_cb.eval_cost_voltage_violation:
                result_metrics["eval_cost_voltage_violation"] = np.asarray(
                    curve_cb.eval_cost_voltage_violation, dtype=np.float64
                )
            if curve_cb.eval_total_operating_cost:
                result_metrics["eval_total_operating_cost"] = np.asarray(
                    curve_cb.eval_total_operating_cost, dtype=np.float64
                )
            if curve_cb.eval_total_reserve_shortfall:
                result_metrics["eval_total_reserve_shortfall"] = np.asarray(
                    curve_cb.eval_total_reserve_shortfall, dtype=np.float64
                )
            if curve_cb.eval_total_thermal_overload:
                result_metrics["eval_total_thermal_overload"] = np.asarray(
                    curve_cb.eval_total_thermal_overload, dtype=np.float64
                )
            if curve_cb.eval_reserve_shortfall_rate:
                result_metrics["eval_reserve_shortfall_rate"] = np.asarray(
                    curve_cb.eval_reserve_shortfall_rate, dtype=np.float64
                )
            if curve_cb.eval_thermal_violation_rate:
                result_metrics["eval_thermal_violation_rate"] = np.asarray(
                    curve_cb.eval_thermal_violation_rate, dtype=np.float64
                )
            result_metrics["final_return"] = float(curve_cb.train_returns[-1])
            train_curve_walltimes = list(curve_cb.walltimes)
        if ep_rew_mean is not None:
            if "ep_rew_mean" not in result_metrics:
                result_metrics["ep_rew_mean"] = np.asarray([ep_rew_mean], dtype=np.float64)
            result_metrics["final_return"] = float(ep_rew_mean)
        dc_eval_result = None
        dso_eval_result = None
        if jax_task == "dc_microgrid":
            dc_eval_result = _evaluate_dc_model(
                model,
                split=final_eval_split,
                seed=seed,
                n_episodes=int(task_cfg.get("eval_episodes", 10)),
            )
        elif jax_task == "dso":
            dso_eval_result = _evaluate_dso_model(
                model,
                split=final_eval_split,
                seed=seed,
                n_episodes=int(task_cfg.get("eval_episodes", 50)),
                task_cfg=task_cfg,
            )

    # ── pettingzoo IL path: per-agent SB3/SBX model with frozen self-play ─
    elif kind == "pettingzoo":
        marl_env = handle
        agent_ids = list(marl_env.possible_agents)
        n_agents = len(agent_ids)
        ders_curve_env = None

        # Frozen self-play protocol (cheap fictitious play / lagged best
        # response): split the total budget into N_ROUNDS phases. In each
        # round, train each agent against the FROZEN policies of the
        # other agents (last-round versions; round-0 opponents are
        # random because no policies exist yet). Updated models are
        # held back until the round finishes so all agents see the same
        # opponent generation. This is the standard IL upgrade beyond
        # naïve "random opponents" (see _make_il_env docstring).
        #
        # Budget resolution: ``per_agent_steps_per_round`` in
        # ``extra_config`` overrides the default split of
        # ``total_timesteps``.  Default is bumped to 4 rounds so each
        # agent sees a more-fictitious-play-like opponent distribution
        # (random round + 3 frozen rounds) rather than the prior 2-round
        # 50%-random regime that biased the final policy toward random
        # opponents.
        cfg_overrides = dict(cfg_overrides)
        n_rounds = int(cfg_overrides.get("il_self_play_rounds", 4))
        explicit_per_agent = cfg_overrides.get("per_agent_steps_per_round")
        curve_eval_episodes = max(int(cfg_overrides.get("curve_eval_episodes", 5)), 1)
        if explicit_per_agent is not None:
            per_agent_ts_per_round = int(explicit_per_agent)
        else:
            per_agent_ts_per_round = max(
                int(resolved_total_timesteps) // max(n_agents * n_rounds, 1), 1_000
            )

        models: dict[str, Any] = {}        # current generation
        frozen: dict[str, Any] = {}        # held-back opponents for this round
        per_agent_returns: dict[str, float] = {}
        il_status: str = "completed"
        il_error_msg: str = ""
        checkpoint_train_returns: list[float] = []
        checkpoint_eval_returns: list[float] = []
        checkpoint_gencos_total_profit: list[float] = []
        checkpoint_ders_total_cost: list[float] = []
        checkpoint_ders_mean_p_loss_mw: list[float] = []
        checkpoint_ders_voltage_violation_steps: list[float] = []
        checkpoint_ders_voltage_safety_rate: list[float] = []
        checkpoint_hhi: list[float] = []
        checkpoint_ramp_binding: list[float] = []
        checkpoint_mean_lmp: list[float] = []
        checkpoint_price_volatility: list[float] = []
        checkpoint_sced_convergence: list[float] = []
        checkpoint_timesteps: list[int] = []
        checkpoint_walltimes: list[float] = []
        if jax_task == "ders":
            ders_curve_env, curve_kind = _build_powerzoo_env(
                "ders",
                split=train_split,
                seed=seed,
                episode_idx=0,
                n_episodes=curve_eval_episodes,
                strategy="uniform",
            )
            if curve_kind != "pettingzoo":
                raise RuntimeError(
                    f"Expected pettingzoo env for DER curve eval, got {curve_kind!r}"
                )

        for round_idx in range(n_rounds):
            new_models: dict[str, Any] = {}
            print(
                f"  [IL self-play] round {round_idx + 1}/{n_rounds} "
                f"({per_agent_ts_per_round:,} steps/agent, "
                f"opponents={'frozen' if frozen else 'random'})"
            )
            # Re-seed the inner MARL env each self-play round (seed + round_idx)
            # so RNG state does not carry across rounds unintentionally.
            try:
                marl_env.reset(seed=int(seed) + round_idx)
            except TypeError:
                marl_env.reset()
            for agent_id in agent_ids:
                il_env = _make_il_env(
                    marl_env, agent_id, frozen_opponents=frozen,
                )
                prev = models.get(agent_id)
                # Warm-start: reuse this agent's previous-round model
                # (keeps the policy across rounds; if set_env fails for
                # any SB3 internal reason fall through to fresh init).
                if prev is not None and round_idx > 0:
                    try:
                        prev.set_env(il_env)
                        model = prev
                    except Exception:
                        model = algo_cls(
                            policy_spec,
                            il_env,
                            verbose=0,
                            seed=seed + round_idx,
                            device=device,
                            **algo_kwargs,
                        )
                else:
                    model = algo_cls(
                        policy_spec,
                        il_env,
                        verbose=0,
                        seed=seed + round_idx,
                        device=device,
                        **algo_kwargs,
                    )
                # Per-agent learn() failure: stop the round loop and later
                # write status=failed with partial progress instead of
                # dropping the run entirely.
                try:
                    model.learn(
                        total_timesteps=per_agent_ts_per_round,
                        progress_bar=False,
                        reset_num_timesteps=False if prev is not None else True,
                    )
                except Exception as exc:
                    il_status = "failed"
                    il_error_msg = (
                        f"agent={agent_id} round={round_idx + 1}/{n_rounds} "
                        f"raised {type(exc).__name__}: {exc}"
                    )
                    print(f"    [IL][FAIL] {il_error_msg}")
                    models = {**models, **new_models}
                    break
                er = _extract_ep_rew_mean(model)
                if er is not None:
                    per_agent_returns[agent_id] = float(er)
                new_models[agent_id] = model
                print(
                    f"    [IL] agent={agent_id} round={round_idx + 1} "
                    f"ep_rew_mean={er}"
                )
                checkpoint_bank = {
                    aid: new_models.get(aid, models.get(aid))
                    for aid in agent_ids
                }
                known_train_returns = [
                    per_agent_returns[aid]
                    for aid in agent_ids
                    if aid in per_agent_returns
                ]
                if known_train_returns:
                    checkpoint_train_returns.append(float(np.mean(known_train_returns)))
                checkpoint_timesteps.append(
                    int((round_idx * n_agents + len(new_models)) * per_agent_ts_per_round)
                )
                checkpoint_walltimes.append(float(time.perf_counter() - t0))
                if jax_task == "gencos":
                    ck_eval = _evaluate_gencos_policy_bank(
                        checkpoint_bank,
                        split=train_split,
                        seed=seed,
                        n_episodes=curve_eval_episodes,
                    )
                    ck_metrics = ck_eval["metrics"]
                    checkpoint_eval_returns.append(
                        float(ck_metrics.get("mean_profit_per_agent", 0.0))
                    )
                    checkpoint_gencos_total_profit.append(
                        float(ck_metrics.get("total_profit", 0.0))
                    )
                    checkpoint_hhi.append(float(ck_metrics.get("hhi", 0.0)))
                    checkpoint_ramp_binding.append(
                        float(ck_metrics.get("ramp_binding_rate", 0.0))
                    )
                    checkpoint_mean_lmp.append(float(ck_metrics.get("mean_lmp", 0.0)))
                    checkpoint_price_volatility.append(
                        float(ck_metrics.get("price_volatility", 0.0))
                    )
                    checkpoint_sced_convergence.append(
                        float(ck_metrics.get("sced_convergence_rate", 0.0))
                    )
                    print(
                        "    [IL][curve] "
                        f"agg_steps={checkpoint_timesteps[-1]:,} "
                        f"train_mean_profit_per_agent={checkpoint_eval_returns[-1]:.2f} "
                        f"train_total_profit={checkpoint_gencos_total_profit[-1]:.2f} "
                        f"hhi={checkpoint_hhi[-1]:.4f} "
                        f"ramp={checkpoint_ramp_binding[-1]:.4f}"
                    )
                elif jax_task == "ders":
                    ck_eval = _evaluate_ders_policy_bank(
                        ders_curve_env,
                        checkpoint_bank,
                        split=train_split,
                        seed=seed,
                        n_episodes=curve_eval_episodes,
                    )
                    ck_metrics = ck_eval["metrics"]
                    checkpoint_eval_returns.append(
                        float(ck_metrics.get("episode_reward", 0.0))
                    )
                    checkpoint_ders_total_cost.append(
                        float(ck_metrics.get("total_cost", 0.0))
                    )
                    checkpoint_ders_mean_p_loss_mw.append(
                        float(ck_metrics.get("mean_p_loss_mw", 0.0))
                    )
                    checkpoint_ders_voltage_violation_steps.append(
                        float(ck_metrics.get("voltage_violation_steps", 0.0))
                    )
                    checkpoint_ders_voltage_safety_rate.append(
                        float(ck_metrics.get("voltage_safety_rate", 0.0))
                    )
                    print(
                        "    [IL][curve] "
                        f"agg_steps={checkpoint_timesteps[-1]:,} "
                        f"train_episode_reward={checkpoint_eval_returns[-1]:.3f} "
                        f"p_loss_mw={checkpoint_ders_mean_p_loss_mw[-1]:.4f} "
                        f"viol_steps={checkpoint_ders_voltage_violation_steps[-1]:.3f}"
                    )
            if il_status == "failed":
                break
            models = new_models
            # Freeze for next round
            frozen = dict(models)

        walltime = time.perf_counter() - t0
        result_metrics = {}
        if checkpoint_train_returns:
            result_metrics["ep_rew_mean"] = np.asarray(
                checkpoint_train_returns, dtype=np.float64
            )
        if checkpoint_timesteps:
            result_metrics["eval_timesteps"] = np.asarray(
                checkpoint_timesteps, dtype=np.float64
            )
        if checkpoint_eval_returns:
            result_metrics["eval_returns"] = np.asarray(
                checkpoint_eval_returns, dtype=np.float64
            )
            result_metrics["final_return"] = float(checkpoint_eval_returns[-1])
        if checkpoint_gencos_total_profit:
            result_metrics["eval_total_profit"] = np.asarray(
                checkpoint_gencos_total_profit, dtype=np.float64
            )
        if checkpoint_ders_total_cost:
            result_metrics["eval_total_cost"] = np.asarray(
                checkpoint_ders_total_cost, dtype=np.float64
            )
        if checkpoint_ders_mean_p_loss_mw:
            result_metrics["eval_mean_p_loss_mw"] = np.asarray(
                checkpoint_ders_mean_p_loss_mw, dtype=np.float64
            )
        if checkpoint_ders_voltage_violation_steps:
            result_metrics["eval_voltage_violation_steps"] = np.asarray(
                checkpoint_ders_voltage_violation_steps, dtype=np.float64
            )
        if checkpoint_ders_voltage_safety_rate:
            result_metrics["eval_voltage_safety_rate"] = np.asarray(
                checkpoint_ders_voltage_safety_rate, dtype=np.float64
            )
        if checkpoint_hhi:
            result_metrics["market/HHI"] = np.asarray(checkpoint_hhi, dtype=np.float64)
        if checkpoint_ramp_binding:
            result_metrics["market/ramp_binding_rate"] = np.asarray(
                checkpoint_ramp_binding, dtype=np.float64
            )
        if checkpoint_mean_lmp:
            result_metrics["mean_lmp"] = np.asarray(checkpoint_mean_lmp, dtype=np.float64)
        if checkpoint_price_volatility:
            result_metrics["market/price_volatility"] = np.asarray(
                checkpoint_price_volatility, dtype=np.float64
            )
        if checkpoint_sced_convergence:
            result_metrics["sced_convergence_rate"] = np.asarray(
                checkpoint_sced_convergence, dtype=np.float64
            )
        if per_agent_returns and "final_return" not in result_metrics:
            mean_ret = float(np.mean(list(per_agent_returns.values())))
            result_metrics["ep_rew_mean"] = np.asarray([mean_ret], dtype=np.float64)
            result_metrics["final_return"] = mean_ret
        if per_agent_returns:
            result_metrics["per_agent_final_return"] = per_agent_returns
        if il_status == "failed":
            result_metrics["il_failure_reason"] = il_error_msg
        if checkpoint_walltimes:
            train_curve_walltimes = checkpoint_walltimes
        if il_status == "completed" and models:
            il_model_artifacts = _save_il_models_manifest(
                models,
                run_id=run_id,
                artifacts_dir=artifacts_dir,
            )
        if jax_task == "ders" and il_status == "completed" and models:
            eval_splits = tuple(task_cfg.get("eval_splits", _VALID_SPLITS_BY_TASK["ders"]))
            for eval_split in eval_splits:
                eval_env, eval_kind = _build_powerzoo_env(
                    "ders",
                    split=eval_split,
                    seed=seed,
                    episode_idx=0,
                    n_episodes=_ders_eval_episodes_for_split(eval_split),
                    strategy="uniform",
                )
                if eval_kind != "pettingzoo":
                    raise RuntimeError(
                        f"Expected pettingzoo env for DER eval, got {eval_kind!r}"
                    )
                try:
                    eval_result = _evaluate_ders_policy_bank(
                        eval_env,
                        models,
                        split=eval_split,
                        seed=seed,
                        n_episodes=_ders_eval_episodes_for_split(eval_split),
                    )
                finally:
                    try:
                        eval_env.close()
                    except Exception:
                        pass
                if eval_split == train_split:
                    ders_train_eval_result = eval_result
                ders_eval_records.append(
                    _save_ders_eval_record(
                        algorithm=algorithm,
                        backend=backend,
                        device=device,
                        framework_version=framework_version,
                        seed=seed,
                        split=eval_split,
                        source_run_id=run_id,
                        eval_result=eval_result,
                        task_dir=task_dir,
                    )
                )
        if jax_task == "gencos" and il_status == "completed" and models:
            eval_splits = ("train", "iid", "demand_shift", "renewable_shock")
            for eval_split in eval_splits:
                eval_result = _evaluate_gencos_policy_bank(
                    models,
                    split=eval_split,
                    seed=seed,
                    n_episodes=_gencos_eval_episodes_for_split(eval_split),
                )
                if eval_split == train_split:
                    gencos_train_eval_result = eval_result
                gencos_eval_records.append(
                    _save_gencos_eval_record(
                        algorithm=algorithm,
                        backend=backend,
                        device=device,
                        framework_version=framework_version,
                        seed=seed,
                        split=eval_split,
                        source_run_id=run_id,
                        eval_result=eval_result,
                        task_dir=task_dir,
                    )
                )
        if ders_curve_env is not None:
            try:
                ders_curve_env.close()
            except Exception:
                pass
    else:
        raise RuntimeError(f"Unknown env kind: {kind!r}")

    # ── save artifacts (canonical curve aliases, config snapshot) ─────
    artifact_paths = save_training_artifacts(
        result_metrics={
            k: v for k, v in result_metrics.items() if not isinstance(v, dict)
        },
        run_id=run_id,
        artifacts_dir=artifacts_dir,
        total_timesteps=int(resolved_total_timesteps),
        config_snapshot={
            "powerzoo_driver_config": cfg,
            "powerzoo_task": entry["powerzoo_task"],
            "jax_task": jax_task,
            "backend": backend,
            "task_config": task_cfg if task_cfg else None,
            "train_config_raw": benchmark_train_cfg if benchmark_train_cfg else None,
        },
        eval_walltimes_s=train_curve_walltimes,
        train_curve_source="ep_rew_mean",
        eval_curve_source="eval_returns",
        extra_artifacts=(il_model_artifacts or single_model_artifacts)
        if (il_model_artifacts or single_model_artifacts)
        else None,
    )
    if dc_eval_result is not None:
        artifact_paths.update(
            save_eval_artifacts(
                per_episode_metrics=dc_eval_result["per_episode_metrics"],
                run_id=run_id,
                split=final_eval_split,
                artifacts_dir=artifacts_dir,
                per_episode_actions=dc_eval_result["per_episode_actions"],
                per_episode_rewards=dc_eval_result["per_episode_rewards"],
            )
        )
    if dso_eval_result is not None:
        artifact_paths.update(
            save_eval_artifacts(
                per_episode_metrics=dso_eval_result["per_episode_metrics"],
                run_id=run_id,
                split=final_eval_split,
                artifacts_dir=artifacts_dir,
                per_episode_rewards=dso_eval_result["per_episode_rewards"],
            )
        )

    metrics: dict[str, Any] = {}
    if "final_return" in result_metrics:
        metrics["train_final_return"] = float(result_metrics["final_return"])
    if "per_agent_final_return" in result_metrics:
        # Per-agent breakdown is informational only (variance_check uses
        # the scalar ``final_return`` key); flatten to keep the schema's
        # ``dict[str, float]`` invariant.
        for agent_id, ret in result_metrics["per_agent_final_return"].items():
            metrics[f"per_agent_final_return__{agent_id}"] = float(ret)
    if dc_eval_result is not None:
        metrics.update(dc_eval_result["metrics"])
        if "episode_reward" in metrics:
            metrics["final_return"] = float(metrics["episode_reward"])
    if ders_train_eval_result is not None:
        metrics.update(ders_train_eval_result["metrics"])
        if "episode_reward" in metrics:
            metrics["final_return"] = float(metrics["episode_reward"])
    if dso_eval_result is not None:
        metrics.update(dso_eval_result["metrics"])
        if "total_reward" in metrics:
            metrics["final_return"] = float(metrics["total_reward"])
    if gencos_train_eval_result is not None:
        metrics.update(gencos_train_eval_result["metrics"])
        if "total_profit" in metrics:
            metrics["final_return"] = float(metrics["total_profit"])

    # ── Project ep_rew_mean onto the task-specific metric key from
    # task config so that ``variance_check`` and downstream summarize/plot
    # tools can read cross-backend records without needing a separate
    # branch.  For tasks where the SB3 reward signal IS the canonical
    # metric (DSO total_reward = per-episode reward; DC Microgrid
    # episode_reward = per-episode reward), ep_rew_mean is the metric
    # value directly.  TSO requires the inverse of the reward scaling
    # back to USD.  DERs/GenCos canonical metrics
    # (mean_p_loss_mw / total_profit) cannot be reconstructed from the
    # reward alone, so these tasks must write explicit eval-rollout metrics.
    if "final_return" in metrics:
        ep_rew = metrics["final_return"]
        task_key, task_value = _project_to_task_metric(jax_task, ep_rew)
        if task_key is not None and task_value is not None:
            metrics[task_key] = float(task_value)

    # IL self-play failure → RunRecord.status failed (partial metrics kept).
    record_status = "completed"
    record_notes = (
        f"cross-backend run via powerzoo_driver. "
        f"powerzoo_task={entry['powerzoo_task']} access_path={entry['access_path']} "
        f"rl_paradigm={entry['rl_paradigm']} n_agents={entry['n_agents']} "
        f"algorithm={algorithm} train_split={train_split} eval_split={final_eval_split}."
    )
    if shaping_weights:
        record_notes = (
            f"{record_notes} Applied benchmark reward shaping weights "
            f"{shaping_weights}."
        )
    if "il_failure_reason" in result_metrics:
        record_status = "failed"
        record_notes = (
            f"{record_notes} "
            f"IL self-play failed: {result_metrics['il_failure_reason']}; "
            f"partial per-agent returns retained for diagnostics."
        )
    requested_dist = policy_metadata.get("requested_continuous_action_dist")
    effective_dist = policy_metadata.get("effective_continuous_action_dist")
    if requested_dist and effective_dist and requested_dist != effective_dist:
        record_notes = (
            f"{record_notes} Requested continuous_action_dist={requested_dist}, "
            f"but backend policy used {effective_dist}."
        )
    if (
        cfg.get("requested_normalize_observations")
        and not cfg.get("effective_normalize_observations")
    ):
        record_notes = (
            f"{record_notes} Requested normalize_observations=true, but no "
            f"VecNormalize wrapper was applied."
        )
    env_meta = _env_contract_metadata(handle)
    if not env_meta:
        env_meta = dict(initial_env_meta)
    if env_meta:
        record_notes = f"{record_notes} env_meta={env_meta}."
    if gencos_eval_records:
        record_notes = (
            f"{record_notes} official_eval_run_ids="
            f"{[rec.run_id for rec in gencos_eval_records]}."
        )
    if ders_eval_records:
        record_notes = (
            f"{record_notes} official_eval_run_ids="
            f"{[rec.run_id for rec in ders_eval_records]}."
        )
    env_info, labels = _collect_torch_run_contract(
        requested_device=device,
        context=f"powerzoo_driver/{jax_task}",
        meta={
            **collect_dataset_provenance(
                task=jax_task,
                task_config=load_task_config(task_dir),
                split=record_split,
            ),
            **env_meta,
        },
        extra_labels={**base_labels, "record_kind": "train"},
    )

    record = RunRecord(
        task=jax_task,
        variant=f"{jax_task}_cross_backend_{entry['access_path']}",
        algo=algorithm.lower(),
        seed=int(seed),
        run_id=run_id,
        config_hash=config_hash(cfg),
        status=record_status,
        split=record_split,
        backend=backend,
        device=device,
        framework_version=framework_version,
        metrics=metrics,
        walltime_s=float(walltime),
        throughput_sps=(int(resolved_total_timesteps) / walltime) if walltime > 0 else None,
        notes=record_notes,
        env_info=env_info,
        labels=labels,
        artifacts=artifact_paths,
    )
    path = save_run(record, task_dir)
    print(f"[powerzoo_driver] saved {path}")
    return record

# Helper: project SB3 ep_rew_mean onto the task-specific metric key
# declared in benchmarks/<task>/configs/task.yaml::target_return_metric_key.
# Cross-backend records previously only wrote ``final_return`` and were
# silently skipped by variance_check (which keys on the task-specific
# field).  This helper closes the gap for the tasks where the projection
# is well-defined; DERs/GenCos still require explicit eval rollouts.

# Mapping of task → (task_metric_key, transform).  ``None`` means we
# cannot derive the metric from ep_rew_mean alone — the caller must run
# an eval rollout to compute it.  Update as new task metrics become
# derivable from SB3's reward signal.
_TASK_METRIC_FROM_REWARD: dict[str, tuple[str, Any]] = {
    # DSO: target_return_metric_key=total_reward; reward signal IS the
    # per-step DSO reward, ep_rew_mean = per-episode total_reward.
    "dso": ("total_reward", lambda r: r),
    # TSO: reward = -REWARD_SCALE * operating_cost per step (see
    # comparison_tso.py); ep_rew_mean (per-episode total) maps back to
    # the per-episode operating cost via dividing by REWARD_SCALE.
    "tso": ("total_operating_cost", lambda r: -float(r) / 1e-4),
    # DC Microgrid: target_return_metric_key=episode_reward; reward signal
    # IS the per-episode reward sum.
    "dc_microgrid": ("episode_reward", lambda r: r),
    # DERs / GenCos: canonical metrics need an eval rollout (info-side
    # quantities, not reducible to reward).
}

def _project_to_task_metric(
    jax_task: str, ep_rew_mean: float
) -> tuple[str | None, float | None]:
    entry = _TASK_METRIC_FROM_REWARD.get(jax_task)
    if entry is None:
        return None, None
    key, fn = entry
    try:
        return key, float(fn(ep_rew_mean))
    except Exception:
        return None, None


class _TrainCurveCallback:
    """Collect coarse train-return checkpoints from SB3/SBX Monitor stats."""

    def __new__(
        cls,
        total_timesteps: int,
        n_checkpoints: int = 20,
        eval_fn: Any | None = None,
    ):
        from stable_baselines3.common.callbacks import BaseCallback

        class _Impl(BaseCallback):
            def __init__(
                self,
                total_timesteps: int,
                n_checkpoints: int = 20,
                eval_fn: Any | None = None,
            ):
                super().__init__()
                self.total_timesteps = max(int(total_timesteps), 1)
                n_pts = max(int(n_checkpoints), 1)
                self.checkpoints = np.unique(
                    np.linspace(
                        self.total_timesteps / n_pts,
                        self.total_timesteps,
                        n_pts,
                        dtype=np.int64,
                    )
                )
                self.next_idx = 0
                self.train_returns: list[float] = []
                self.eval_returns: list[float] = []
                self.eval_cost_voltage_violation: list[float] = []
                self.eval_total_operating_cost: list[float] = []
                self.eval_total_reserve_shortfall: list[float] = []
                self.eval_total_thermal_overload: list[float] = []
                self.eval_reserve_shortfall_rate: list[float] = []
                self.eval_thermal_violation_rate: list[float] = []
                self.timesteps: list[int] = []
                self.walltimes: list[float] = []
                self._t0 = 0.0
                self._eval_fn = eval_fn

            def _on_training_start(self) -> None:
                self._t0 = time.perf_counter()
                print(
                    "[powerzoo_driver] train checkpoints="
                    f"{len(self.checkpoints)} total_timesteps={self.total_timesteps}",
                    flush=True,
                )

            def _current_mean_reward(self) -> float | None:
                try:
                    ep_buf = getattr(self.model, "ep_info_buffer", None)
                    if ep_buf is None or len(ep_buf) == 0:
                        return None
                    rews = [float(ep["r"]) for ep in ep_buf if "r" in ep]
                    if not rews:
                        return None
                    return float(np.mean(rews))
                except Exception:
                    return None

            def _parse_eval_payload(
                self,
                payload: Any,
            ) -> dict[str, float]:
                if isinstance(payload, dict):
                    reward = payload.get("total_reward")
                    if reward is None:
                        reward = payload.get("episode_reward")
                    if reward is None:
                        reward = payload.get("reward")
                    if reward is None:
                        reward = payload.get("return")
                    cost = payload.get("voltage_violation_count_per_step")
                    if cost is None:
                        cost = payload.get("cost_voltage_violation")
                    out: dict[str, float] = {}
                    if reward is not None:
                        out["eval_return"] = float(reward)
                    if cost is not None:
                        out["eval_cost_voltage_violation"] = float(cost)
                    for key in (
                        "total_operating_cost",
                        "total_reserve_shortfall",
                        "total_thermal_overload",
                        "reserve_shortfall_rate",
                        "thermal_violation_rate",
                    ):
                        if payload.get(key) is not None:
                            out[key] = float(payload[key])
                    return out
                if payload is None:
                    return {}
                return {"eval_return": float(payload)}

            def _record_eval_payload(
                self,
                payload: dict[str, float],
                *,
                fallback_reward: float,
            ) -> None:
                eval_reward = payload.get("eval_return")
                self.eval_returns.append(
                    float(fallback_reward if eval_reward is None else eval_reward)
                )
                if "eval_cost_voltage_violation" in payload:
                    self.eval_cost_voltage_violation.append(
                        float(payload["eval_cost_voltage_violation"])
                    )
                if "total_operating_cost" in payload:
                    self.eval_total_operating_cost.append(
                        float(payload["total_operating_cost"])
                    )
                if "total_reserve_shortfall" in payload:
                    self.eval_total_reserve_shortfall.append(
                        float(payload["total_reserve_shortfall"])
                    )
                if "total_thermal_overload" in payload:
                    self.eval_total_thermal_overload.append(
                        float(payload["total_thermal_overload"])
                    )
                if "reserve_shortfall_rate" in payload:
                    self.eval_reserve_shortfall_rate.append(
                        float(payload["reserve_shortfall_rate"])
                    )
                if "thermal_violation_rate" in payload:
                    self.eval_thermal_violation_rate.append(
                        float(payload["thermal_violation_rate"])
                    )

            def _maybe_record(self, *, force: bool = False) -> None:
                while self.next_idx < len(self.checkpoints):
                    target_step = int(self.checkpoints[self.next_idx])
                    if not force and int(self.num_timesteps) < target_step:
                        break
                    mean_reward = self._current_mean_reward()
                    if mean_reward is None:
                        if force:
                            self.next_idx += 1
                            continue
                        break
                    self.timesteps.append(
                        min(
                            int(self.num_timesteps) if force else target_step,
                            self.total_timesteps,
                        )
                    )
                    self.train_returns.append(float(mean_reward))
                    if self._eval_fn is not None:
                        try:
                            eval_payload = self._eval_fn(
                                self.model, int(self.num_timesteps)
                            )
                        except Exception as exc:
                            print(f"[powerzoo_driver] eval callback failed: {exc}")
                            eval_payload = None
                        parsed = self._parse_eval_payload(eval_payload)
                        self._record_eval_payload(parsed, fallback_reward=float(mean_reward))
                    self.walltimes.append(float(time.perf_counter() - self._t0))
                    progress_pct = 100.0 * float(self.timesteps[-1]) / float(self.total_timesteps)
                    msg = (
                        "[powerzoo_driver] checkpoint "
                        f"{self.next_idx + 1}/{len(self.checkpoints)} "
                        f"step={self.timesteps[-1]}/{self.total_timesteps} "
                        f"({progress_pct:.1f}%) "
                        f"train_return={self.train_returns[-1]:.6f}"
                    )
                    if self.eval_returns:
                        msg += f" eval_return={self.eval_returns[-1]:.6f}"
                    if self.eval_total_operating_cost:
                        msg += (
                            " eval_operating_cost="
                            f"{self.eval_total_operating_cost[-1]:.2f}"
                        )
                    if self.eval_reserve_shortfall_rate:
                        msg += (
                            " eval_reserve_rate="
                            f"{self.eval_reserve_shortfall_rate[-1]:.6f}"
                        )
                    if self.eval_thermal_violation_rate:
                        msg += (
                            " eval_thermal_rate="
                            f"{self.eval_thermal_violation_rate[-1]:.6f}"
                        )
                    if self.eval_cost_voltage_violation:
                        msg += (
                            " eval_voltage_violations_per_step="
                            f"{self.eval_cost_voltage_violation[-1]:.6f}"
                        )
                    msg += f" elapsed_s={self.walltimes[-1]:.1f}"
                    print(msg, flush=True)
                    self.next_idx += 1

            def _on_step(self) -> bool:
                self._maybe_record(force=False)
                return True

            def _on_training_end(self) -> None:
                self._maybe_record(force=True)
                if self.train_returns:
                    return
                mean_reward = self._current_mean_reward()
                if mean_reward is None:
                    return
                self.timesteps.append(int(self.num_timesteps) or self.total_timesteps)
                self.train_returns.append(float(mean_reward))
                if self._eval_fn is not None:
                    try:
                        eval_payload = self._eval_fn(self.model, int(self.num_timesteps))
                    except Exception as exc:
                        print(f"[powerzoo_driver] eval callback failed: {exc}")
                        eval_payload = None
                    parsed = self._parse_eval_payload(eval_payload)
                    self._record_eval_payload(parsed, fallback_reward=float(mean_reward))
                self.walltimes.append(float(time.perf_counter() - self._t0))

        return _Impl(
            total_timesteps=total_timesteps,
            n_checkpoints=n_checkpoints,
            eval_fn=eval_fn,
        )


# Helper: extract ep_rew_mean from a finished SB3 / SBX model

def _extract_ep_rew_mean(model) -> float | None:
    """Best-effort ep_rew_mean extraction across SB3/SBX versions."""
    try:
        ep_buf = getattr(model, "ep_info_buffer", None)
        if ep_buf is None or len(ep_buf) == 0:
            return None
        rews = [float(ep["r"]) for ep in ep_buf if "r" in ep]
        if not rews:
            return None
        return float(np.mean(rews))
    except Exception:
        return None

# CLI

def powerzoo_driver_main(argv: "list[str] | None" = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True,
                        help="PowerZooJax task name (dso/tso/ders/gencos/dc_microgrid)")
    parser.add_argument("--algorithm", default="PPO",
                        help="PPO/SAC/TD3 (→ SB3) or SBX_PPO/SBX_SAC/SBX_TD3 (→ SBX)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", default="iid")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="default: benchmark train_*.yaml total_timesteps when available; otherwise 200000",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=0,
        help="Parallel VecEnv workers for single-agent SB3/SBX runs; 0 means use the frozen benchmark train config.",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Only do smoke_load_powerzoo_task; don't train.")
    parser.add_argument(
        "--extra-config-json",
        default=None,
        help="Optional JSON object merged into Python backend driver config.",
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        diag = smoke_load_powerzoo_task(args.task, split=args.split)
        print(json.dumps(diag, indent=2))
        return
    extra_config = None
    if args.extra_config_json:
        parsed = json.loads(args.extra_config_json)
        if not isinstance(parsed, dict):
            raise ValueError("--extra-config-json must decode to a JSON object")
        extra_config = parsed

    train_with_powerzoo(
        jax_task=args.task,
        algorithm=args.algorithm,
        seed=args.seed,
        split=args.split,
        device=args.device,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        extra_config=extra_config,
    )

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BENCHMARKS_DIR = _REPO_ROOT / "benchmarks"


class BackendNotImplemented(NotImplementedError):
    pass

def _powerzoo_algorithm_for_backend(backend: str, algo: str) -> str:
    """Map (backend, algo) → PowerZoo Trainer ALGORITHMS key.

    backend="sb3" + algo="ppo"  → "PPO"
    backend="sbx" + algo="ppo"  → "SBX_PPO"
    """
    base = algo.upper()
    if backend == "sbx":
        if not base.startswith("SBX_"):
            base = f"SBX_{base}"
    return base

def run_one(
    jax_task: str,
    backend: str,
    algo: str,
    seed: int,
    split: str,
    device: str,
    total_timesteps: int | None = None,
    n_envs: int | None = None,
    extra_config: "dict[str, Any] | None" = None,
) -> dict[str, Any]:
    """Dispatch one (task, backend, algo, seed, split, device) combo.

    Returns a status dict for the dispatcher's logging; the actual RunRecord
    is written by the underlying driver into the per-task results/ directory.
    """
    if backend not in CANONICAL_BACKENDS:
        raise ValueError(
            f"backend={backend!r} not in canonical taxonomy {CANONICAL_BACKENDS}"
        )

    # ── sb3 / sbx: only run on cross_backend_comparable=True tasks ──────
    if backend in ("sb3", "sbx"):
        if not is_comparable(jax_task):
            entry = JAX_TASK_TO_POWERZOO_TASK.get(jax_task, {})
            print(
                f"[cross_backend] SKIP {jax_task} backend={backend}: "
                f"cross_backend_comparable={entry.get('cross_backend_comparable')!r}; "
                f"refusing to record incomparable cross-backend metrics. "
                f"Notes: {entry.get('notes', '')}"
            )
            return {"status": "skipped", "reason": "not_cross_backend_comparable"}

        pz_algo = _powerzoo_algorithm_for_backend(backend, algo)
        try:
            t0 = time.perf_counter()
            rec = train_with_powerzoo(
                jax_task=jax_task,
                algorithm=pz_algo,
                seed=seed,
                split=split,
                device=device,
                total_timesteps=total_timesteps,
                n_envs=n_envs,
                extra_config=extra_config,
            )
            return {
                "status": "ok",
                "backend": backend,
                "device": device,
                "run_id": rec.run_id,
                "walltime_s": round(time.perf_counter() - t0, 2),
            }
        except CrossBackendNotComparable as e:
            print(f"[cross_backend] SKIP {jax_task} backend={backend}: {e}")
            return {"status": "skipped", "reason": str(e)}
        except Exception as e:
            print(f"[cross_backend] FAIL {jax_task} backend={backend}: {e}")
            return {"status": "failed", "error": repr(e)[:300]}

    # ── jax_rejax: subprocess into the task-specific benchmark CLI ───────
    if backend == "jax_rejax":
        task_dir = _BENCHMARKS_DIR / jax_task
        run_py = task_dir / "run.py"
        run_all_py = task_dir / "run_all.py"
        if not run_py.exists() and not run_all_py.exists():
            raise BackendNotImplemented(
                f"benchmarks/{jax_task}/run.py or run_all.py not found"
            )
        env = dict(os.environ)
        if device == "cpu":
            env["JAX_PLATFORM_NAME"] = "cpu"
            env["JAX_PLATFORMS"] = "cpu"
            env["POWERZOOJAX_REQUESTED_DEVICE"] = "cpu"
            if jax_task == "tso" and "TSO_CPU_NUM_ENVS" not in env:
                try:
                    _cfg_path, bench_cfg = _load_benchmark_train_config(jax_task, algo)
                    if bench_cfg.get("num_envs") is not None:
                        env["TSO_CPU_NUM_ENVS"] = str(int(bench_cfg["num_envs"]))
                except Exception:
                    pass
        override_cfg_path: Path | None = None
        override_fields: dict[str, Any] = {}
        if total_timesteps is not None:
            override_fields["total_timesteps"] = int(total_timesteps)
        if n_envs is not None and int(n_envs) > 0:
            override_fields["num_envs"] = int(n_envs)
        if override_fields:
            if not run_py.exists():
                raise BackendNotImplemented(
                    f"benchmarks/{jax_task}/run.py is required to override JAX train config"
                )
            override_cfg_path = _write_jax_train_override_config(
                jax_task,
                algo,
                override_fields,
            )
        # If user wants a specific GPU id, they pre-set CUDA_VISIBLE_DEVICES
        # before invoking this script; we don't override it here.
        if run_py.exists():
            cmd = [
                sys.executable,
                str(run_py),
                "train",
                "--algo", algo,
                "--seed", str(seed),
            ]
            if override_cfg_path is not None:
                cmd.extend(["--config", str(override_cfg_path)])
        else:
            cmd = [
                sys.executable,
                str(run_all_py),
                "--only", "train",
                "--algos", algo,
                "--seeds", str(seed),
            ]
        print(f"[cross_backend] dispatching jax_rejax: {' '.join(cmd)} (device={device})")
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(cmd, env=env, cwd=str(_REPO_ROOT), check=False)
            walltime = time.perf_counter() - t0
            ok = proc.returncode == 0
            return {
                "status": "ok" if ok else "failed",
                "backend": backend,
                "device": device,
                "returncode": proc.returncode,
                "walltime_s": round(walltime, 2),
            }
        except FileNotFoundError as e:
            return {"status": "failed", "error": repr(e)[:300]}
        finally:
            if override_cfg_path is not None:
                override_cfg_path.unlink(missing_ok=True)

    raise BackendNotImplemented(f"backend={backend} not handled")

def run_matrix(
    jax_task: str,
    backends: "list[str]",
    seeds: "list[int]",
    split: str,
    algo: str = "ppo",
    devices: "dict[str, str] | None" = None,
    total_timesteps: int | None = None,
    n_envs: int | None = None,
) -> "list[dict[str, Any]]":
    """Run a backend × seed matrix for one task. Returns list of status dicts.

    ``devices`` maps backend → device, defaulting to ``gpu`` for jax_*
    and ``cuda`` for sb3 / sbx; pass ``{"jax_rejax": "cpu", ...}`` to override.
    """
    devices = devices or {}
    statuses: list[dict[str, Any]] = []
    for backend in backends:
        default_dev = "gpu" if backend.startswith("jax_") else "cuda"
        device = devices.get(backend, default_dev)
        for seed in seeds:
            print(f"\n=== cross_backend run: {jax_task} × {backend} × seed={seed} × split={split} ===")
            try:
                s = run_one(
                    jax_task=jax_task,
                    backend=backend,
                    algo=algo,
                    seed=seed,
                    split=split,
                    device=device,
                    total_timesteps=total_timesteps,
                    n_envs=n_envs,
                )
            except BackendNotImplemented as e:
                s = {"status": "skipped", "reason": str(e)}
            statuses.append(
                {**s, "task": jax_task, "backend": backend, "seed": seed, "split": split}
            )
    return statuses

def powerzoo_dispatch_main(argv: "list[str] | None" = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd")

    one = sub.add_parser("run", help="(default) one combo")
    one.add_argument("--task", required=True)
    one.add_argument("--backend", required=True, choices=CANONICAL_BACKENDS)
    one.add_argument("--algo", default="ppo")
    one.add_argument("--seed", type=int, default=0)
    one.add_argument("--split", default="iid")
    one.add_argument("--device", default=None,
                     help="default: gpu for jax_*, cuda for sb3/sbx")
    one.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="default: benchmark train_*.yaml total_timesteps when available; otherwise 200000",
    )
    one.add_argument(
        "--n-envs",
        type=int,
        default=0,
        help="Parallel VecEnv workers for single-agent SB3/SBX runs; 0 means use the frozen benchmark train config.",
    )

    mat = sub.add_parser("matrix", help="backend × seed matrix on one task")
    mat.add_argument("--task", required=True)
    mat.add_argument("--backends", nargs="+", default=list(CANONICAL_BACKENDS))
    mat.add_argument("--algo", default="ppo")
    mat.add_argument("--seeds", nargs="+", type=int, default=[0])
    mat.add_argument("--split", default="iid")
    mat.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="default: benchmark train_*.yaml total_timesteps when available; otherwise 200000",
    )
    mat.add_argument(
        "--n-envs",
        type=int,
        default=0,
        help="Parallel VecEnv workers for single-agent SB3/SBX runs; 0 means use the frozen benchmark train config.",
    )

    # If no subcommand provided, treat as "run" so single-combo invocation
    # like `--task ... --backend ...` still works.
    args, _ = parser.parse_known_args(argv)
    if args.cmd is None:
        # Re-parse against the "run" subparser
        args = one.parse_args(argv)
        args.cmd = "run"

    if args.cmd == "run":
        device = args.device or ("gpu" if args.backend.startswith("jax_") else "cuda")
        s = run_one(
            jax_task=args.task,
            backend=args.backend,
            algo=args.algo,
            seed=args.seed,
            split=args.split,
            device=device,
            total_timesteps=args.total_timesteps,
            n_envs=args.n_envs,
        )
        print("\nRESULT:", s)
    elif args.cmd == "matrix":
        results = run_matrix(
            jax_task=args.task,
            backends=args.backends,
            seeds=args.seeds,
            split=args.split,
            algo=args.algo,
            total_timesteps=args.total_timesteps,
            n_envs=args.n_envs,
        )
        n_ok = sum(1 for r in results if r["status"] == "ok")
        n_skip = sum(1 for r in results if r["status"] == "skipped")
        n_fail = sum(1 for r in results if r["status"] == "failed")
        print(f"\nMATRIX SUMMARY: ok={n_ok} skipped={n_skip} failed={n_fail} total={len(results)}")

def main(argv: "list[str] | None" = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "driver":
        powerzoo_driver_main(argv[1:])
        return
    if argv and argv[0] == "dispatch":
        powerzoo_dispatch_main(argv[1:])
        return
    powerzoo_dispatch_main(argv)

if __name__ == "__main__":
    main()
