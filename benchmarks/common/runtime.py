"""Shared runtime helpers for benchmark train/eval scripts.

This collects benchmark config adaptation, policy reconstruction, CUDA PATH
setup, and training-config helpers in one small surface.
"""

from __future__ import annotations

import dataclasses
import os
import site
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import jax.numpy as jnp
from flax import linen as nn

from powerzoojax.rl.policies import BoundedBetaPolicy


def _candidate_cuda_roots() -> list[Path]:
    roots: list[Path] = []

    def _append(path: Path) -> None:
        if path not in roots:
            roots.append(path)

    for root in site.getsitepackages():
        _append(Path(root) / "nvidia" / "cuda_nvcc")

    home = Path.home()
    for path in (home / ".cache" / "uv" / "archive-v0").glob("*/nvidia/cuda_nvcc"):
        _append(path)
    for path in (home / ".conda" / "envs").glob("*/lib/python*/site-packages/nvidia/cuda_nvcc"):
        _append(path)

    _append(Path("/usr/local/cuda"))
    return roots


def _configure_cuda_root(cuda_root: Path) -> str:
    bin_dir = cuda_root / "bin"
    current = os.environ.get("PATH", "")
    prefix = str(bin_dir)
    if current.split(os.pathsep)[0] != prefix:
        os.environ["PATH"] = prefix + os.pathsep + current if current else prefix

    flag = f"--xla_gpu_cuda_data_dir={cuda_root}"
    existing = os.environ.get("XLA_FLAGS", "").strip()
    if flag not in existing.split():
        os.environ["XLA_FLAGS"] = f"{existing} {flag}".strip()
    return prefix


def prefer_packaged_cuda_binaries() -> str | None:
    """Expose a CUDA toolchain with both ``ptxas`` and ``libdevice``.

    Returns the CUDA ``bin`` directory that was prepended to ``PATH``, or
    ``None`` if no suitable packaged CUDA root was found.
    """
    for cuda_root in _candidate_cuda_roots():
        ptxas = cuda_root / "bin" / "ptxas"
        libdevice = cuda_root / "nvvm" / "libdevice" / "libdevice.10.bc"
        if not (ptxas.exists() and libdevice.exists()):
            continue
        return _configure_cuda_root(cuda_root)
    return None

def _actor_dist_mode(actor, obs):
    """Return the distribution mode for deterministic evaluation."""
    return actor._action_dist(obs).mode()


def _actor_dist_loc(actor, obs):
    """Return the unsquashed Gaussian mean for deterministic evaluation."""
    return actor._action_dist(obs).loc


def _actor_det_continuous(actor, obs):
    """Return a deterministic continuous action for evaluation."""
    if hasattr(actor, "mode_action"):
        return actor.mode_action(obs)
    action_dist = actor._action_dist(obs)
    if hasattr(action_dist, "loc"):
        return action_dist.loc
    if hasattr(action_dist, "mode"):
        return action_dist.mode()
    if hasattr(action_dist, "mean"):
        return action_dist.mean()
    raise AttributeError("Actor distribution does not expose loc/mode/mean.")


def _actor_squashed_mean(actor, obs):
    """Return tanh(mean) mapped into the action range for deterministic eval."""
    raw_mean = actor._action_dist(obs).loc
    return actor.action_loc + jnp.tanh(raw_mean) * actor.action_scale


def _make_rejax_deterministic_act(algo_obj, train_state, env, base_params, *, squashed: bool):
    """Build a deterministic eval policy from a Rejax actor.

    Rejax ``make_act`` samples from the policy distribution. That is correct for
    training, but benchmark evaluation and post-train diagnostics should use the
    deterministic action (mean/mode), otherwise rollout physics depend on policy
    noise rather than the learned controller.
    """
    action_space = env.action_space(base_params)

    def act(obs, _rng):
        if getattr(algo_obj, "normalize_observations", False):
            obs = algo_obj.normalize_obs(train_state.obs_rms_state, obs)

        obs = jnp.expand_dims(obs, 0)
        if squashed:
            action = algo_obj.actor.apply(
                train_state.actor_ts.params, obs, method=_actor_squashed_mean
            )
        elif hasattr(action_space, "n"):
            action = algo_obj.actor.apply(
                train_state.actor_ts.params, obs, method=_actor_dist_mode
            )
        else:
            action = algo_obj.actor.apply(
                train_state.actor_ts.params, obs, method=_actor_det_continuous
            )
            action = jnp.clip(action, action_space.low, action_space.high)
        return jnp.squeeze(action)

    return act

def make_policy_fn(
    algo: str,
    train_result_params: Any,
    env,
    base_params: Any,
    train_cfg: Any,
    *,
    action_dim: int | None = None,
    n_constraints: int | None = None,
    selected_names: Sequence[str] | None = None,
):
    """Return a ``(obs, state, key) -> action`` callable.

    Parameters
    ----------
    algo:
        Algorithm key — ``"ppo"``, ``"sac"``, ``"ppo_lagrangian"``, or
        ``"saute_ppo"``.
    train_result_params:
        The ``.params`` attribute of a :class:`~powerzoojax.rl.trainer.TrainResult`.
    env:
        The *unwrapped* environment (e.g. ``DistGridEnv()``, ``UnitCommitmentEnv()``).
        Used only by the PPO path to reconstruct the rejax adapter.
    base_params:
        Environment params pytree.  Used by PPO to build ``LogWrapper(env, params)``
        and by PPO-Lagrangian to infer ``action_dim`` when not given explicitly.
    train_cfg:
        :class:`~powerzoojax.rl.config.TrainConfig`.
    action_dim:
        Override the action dimension for PPO-Lagrangian.  If ``None``, inferred
        from ``base_params.resources`` (DistGridParams) or
        ``2 * base_params.case.n_units`` (UCParams).
    n_constraints:
        Number of cost critics for ``SafeActorCritic``.  If ``None``, inferred
        from ``train_result_params`` by reading the v_cost Dense layer kernel.
    selected_names:
        Optional benchmark-selected constraint names used by wrappers such as
        ``SauteWrapper``.
    """
    if algo == "ppo" or algo.startswith("ppo_penalty_"):
        # Penalty-shaping runs use the same rejax PPO policy as plain PPO; only the
        # training-time reward wrapper differs (see benchmarks/*/train.py).
        import rejax
        from powerzoojax.rl.trainer import _RejaxAdapter
        from powerzoojax.rl.wrappers import LogWrapper

        adapted_env = _RejaxAdapter(LogWrapper(env, base_params))
        create_kwargs = train_cfg.to_rejax_kwargs()
        actor_override = None
        if train_cfg.continuous_action_dist == "beta":
            action_space = adapted_env.action_space(None)
            if not hasattr(action_space, "n"):
                actor_override = BoundedBetaPolicy(
                    int(jnp.prod(jnp.asarray(action_space.shape))),
                    (jnp.asarray(action_space.low), jnp.asarray(action_space.high)),
                    tuple(train_cfg.hidden_dims),
                    nn.swish,
                )
        algo_obj = rejax.PPO.create(env=adapted_env, env_params=None, **create_kwargs)
        if actor_override is not None:
            algo_obj = algo_obj.replace(actor=actor_override)
        act = _make_rejax_deterministic_act(
            algo_obj, train_result_params, env, base_params, squashed=False,
        )
        return lambda obs, state, key: act(obs, key)

    elif algo == "sac":
        import rejax
        from powerzoojax.rl.trainer import _RejaxAdapter, _rejax_create_kwargs
        from powerzoojax.rl.wrappers import LogWrapper

        adapted_env = _RejaxAdapter(LogWrapper(env, base_params))
        sac_algo = rejax.SAC.create(env=adapted_env, env_params=None, **_rejax_create_kwargs(rejax.SAC, train_cfg))
        act = _make_rejax_deterministic_act(
            sac_algo, train_result_params, env, base_params, squashed=True,
        )
        return lambda obs, state, key: act(obs, key)

    elif algo == "ppo_lagrangian":
        from powerzoojax.rl.cmdp import SafeActorCritic

        if action_dim is None:
            action_dim = _infer_action_dim(base_params)
        if n_constraints is None:
            n_constraints = _infer_n_constraints(train_result_params, action_dim)

        hidden_dim = train_cfg.hidden_dims[0] if train_cfg.hidden_dims else 256
        network = SafeActorCritic(
            action_dim=action_dim,
            n_constraints=n_constraints,
            hidden_dim=hidden_dim,
        )
        net_params = train_result_params

        def policy_fn(obs, state, key):
            mean, _, _, _ = network.apply(net_params, obs)
            return jnp.clip(mean, -1.0, 1.0)

        return policy_fn

    elif algo == "saute_ppo":
        import rejax
        from powerzoojax.rl.trainer import _RejaxAdapter
        from powerzoojax.rl.wrappers import SauteWrapper

        horizon = (
            int(train_cfg.saute_horizon)
            if train_cfg.saute_horizon is not None
            else _infer_episode_horizon(base_params)
        )
        if train_cfg.cost_thresholds:
            budgets = tuple(float(x) for x in train_cfg.cost_thresholds)
            if any(x <= 0.0 for x in budgets):
                raise ValueError(
                    f"saute_ppo requires positive budgets, got {budgets!r}."
                )
            saute_env = SauteWrapper(
                env,
                base_params,
                cost_thresholds=budgets,
                selected_names=selected_names,
                horizon=horizon,
                unsafe_reward=float(train_cfg.saute_unsafe_reward),
                use_reward_shaping=False,
            )
        else:
            budget = float(train_cfg.cost_threshold)
            if budget <= 0.0:
                raise ValueError(
                    f"saute_ppo requires a positive budget, got {budget}."
                )
            saute_env = SauteWrapper(
                env,
                base_params,
                cost_threshold=budget,
                selected_names=selected_names,
                horizon=horizon,
                unsafe_reward=float(train_cfg.saute_unsafe_reward),
                use_reward_shaping=False,
            )

        adapted_env = _RejaxAdapter(saute_env)
        create_kwargs = train_cfg.to_rejax_kwargs()
        actor_override = None
        if train_cfg.continuous_action_dist == "beta":
            action_space = adapted_env.action_space(None)
            if not hasattr(action_space, "n"):
                actor_override = BoundedBetaPolicy(
                    int(jnp.prod(jnp.asarray(action_space.shape))),
                    (jnp.asarray(action_space.low), jnp.asarray(action_space.high)),
                    tuple(train_cfg.hidden_dims),
                    nn.swish,
                )
        algo_obj = rejax.PPO.create(env=adapted_env, env_params=None, **create_kwargs)
        if actor_override is not None:
            algo_obj = algo_obj.replace(actor=actor_override)
        # Sauté PPO is still plain PPO under the hood, so benchmark evaluation
        # should use the deterministic actor mode just like vanilla PPO.
        act = _make_rejax_deterministic_act(
            algo_obj, train_result_params, env, base_params, squashed=False,
        )
        return lambda obs, state, key: act(obs, key)

    else:
        raise ValueError(
            f"Unsupported algo: {algo!r}. Choose from 'ppo', 'ppo_penalty_*', "
            "'sac', 'ppo_lagrangian', 'saute_ppo'."
        )


def _infer_n_constraints(net_params: Any, action_dim: int) -> int:
    """Infer ``n_constraints`` from a SafeActorCritic params pytree.

    SafeActorCritic appends heads in fixed order (two hidden blocks, mean
    head, reward critic, cost critic), so the final Dense layer's output
    dim is ``n_constraints``.  Flax stores modules by insertion order, so
    we pick the last Dense-like entry (mapping with a ``kernel``) from the
    params dict.
    """
    from flax.core import unfreeze

    tree = unfreeze(net_params)
    leaves = tree.get("params", tree)
    last_dense = None
    for name, sub in leaves.items():
        if isinstance(sub, dict) and "kernel" in sub:
            last_dense = sub
    if last_dense is None:
        raise ValueError(
            "Could not infer n_constraints from SafeActorCritic params. "
            "Pass n_constraints explicitly to make_policy_fn()."
        )
    return int(last_dense["kernel"].shape[-1])

def _infer_action_dim(params) -> int:
    """Infer action dimension from env params when not provided explicitly."""
    # DistGridParams (DSO, DERs single-agent)
    resources = getattr(params, "resources", None)
    if resources is not None:
        return int(sum(b.action_dim for b in resources))
    # UCParams (TSO)
    case = getattr(params, "case", None)
    if case is not None and hasattr(case, "n_units"):
        return int(2 * case.n_units)
    raise ValueError(
        "Cannot infer action_dim from params. "
        "Pass action_dim explicitly to make_policy_fn()."
    )


def _infer_episode_horizon(params, default: int = 48) -> int:
    """Infer episode horizon from params or nested task config objects."""
    if hasattr(params, "max_steps"):
        return int(params.max_steps)
    for attr in ("dc", "grid", "dist"):
        sub = getattr(params, attr, None)
        if sub is not None and hasattr(sub, "max_steps"):
            return int(sub.max_steps)
    return int(default)

from powerzoojax.rl.config import TrainConfig

_LAGRANGIAN_FIELDS = (
    "cost_threshold",
    "cost_thresholds",
    "lambda_lr",
    "cost_scale",
    "log_lambda_max",
    "n_checkpoints",
)


def _replace_train_cfg(full_cfg: Any, **updates: Any) -> Any:
    if hasattr(full_cfg, "replace"):
        return full_cfg.replace(**updates)
    return dataclasses.replace(full_cfg, **updates)


def make_warmup_cfg(full_cfg: Any, *, warmup_iters: int = 1) -> Any:
    """Return a short training config used only to trigger JIT compilation."""
    n_envs = max(1, int(getattr(full_cfg, "num_envs", 1)))
    n_steps = max(1, int(getattr(full_cfg, "n_steps", 1)))
    one_iter = n_envs * n_steps
    eval_freq = max(0, int(getattr(full_cfg, "eval_freq", 0) or 0))
    warmup_total = max(one_iter * max(1, int(warmup_iters)), eval_freq or one_iter)
    full_total = max(1, int(getattr(full_cfg, "total_timesteps", warmup_total)))
    if full_total > 1:
        warmup_total = min(warmup_total, full_total - 1)
    warmup_total = max(1, int(warmup_total))

    updates: dict[str, Any] = {"total_timesteps": warmup_total}
    if hasattr(full_cfg, "wall_time_warmup"):
        updates["wall_time_warmup"] = False
    return _replace_train_cfg(full_cfg, **updates)


def make_steady_train_cfg(full_cfg: Any) -> Any:
    """Return the full-run config with internal wall-clock warmup disabled."""
    if hasattr(full_cfg, "wall_time_warmup"):
        return _replace_train_cfg(full_cfg, wall_time_warmup=False)
    return full_cfg


def _block_train_result(result: Any) -> None:
    try:
        import jax

        jax.block_until_ready(getattr(result, "params", result))
    except Exception:
        pass


def time_jax_train_with_warmup(
    train_fn: Callable[..., Any],
    *,
    full_cfg: Any,
    warmup_cfg: Any,
    **kwargs: Any,
) -> tuple[Any, float, float]:
    """Run a discarded JAX warmup train, then time the full steady-state run."""
    t_warm = time.perf_counter()
    warmup_result = train_fn(config=warmup_cfg, **kwargs)
    _block_train_result(warmup_result)
    compile_warmup_s = time.perf_counter() - t_warm

    t_main = time.perf_counter()
    result = train_fn(config=full_cfg, **kwargs)
    _block_train_result(result)
    walltime_s = time.perf_counter() - t_main

    return result, walltime_s, compile_warmup_s

def build_train_cfg(
    config: dict[str, Any],
    *,
    algo: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> TrainConfig:
    """Build a :class:`TrainConfig` from a benchmark config dict.

    Parameters
    ----------
    config:
        Parsed ``train_*.yaml`` / ``train_*.json`` config dict.
    algo:
        If ``"ppo_lagrangian"``, Lagrangian fields from ``config`` are
        propagated to the returned ``TrainConfig`` via ``replace``.  Pass
        ``None`` to skip the Lagrangian step (algo is still read from
        ``config["algo"]`` in that case).
    overrides:
        Optional last-step ``replace(**overrides)`` for callers that need
        to pin a specific field (e.g. seed) regardless of config contents.
    """
    cfg_init = dict(config)
    if "learning_rate" not in cfg_init and "lr" in cfg_init:
        cfg_init["learning_rate"] = cfg_init["lr"]
    if "hidden_dims" in cfg_init and isinstance(cfg_init["hidden_dims"], list):
        cfg_init["hidden_dims"] = tuple(cfg_init["hidden_dims"])
    for key in ("cost_thresholds", "lambda_lr", "cost_scale"):
        if key in cfg_init and isinstance(cfg_init[key], list):
            cfg_init[key] = tuple(cfg_init[key])

    valid_fields = {f.name for f in dataclasses.fields(TrainConfig)}
    filtered = {k: v for k, v in cfg_init.items() if k in valid_fields}
    cfg = TrainConfig(**filtered)

    if algo == "ppo_lagrangian":
        replace_kwargs: dict[str, Any] = {"algo": algo}
        for k in _LAGRANGIAN_FIELDS:
            if k in config:
                v = config[k]
                if k == "n_checkpoints":
                    v = int(v)
                replace_kwargs[k] = v
        cfg = cfg.replace(**replace_kwargs)
    elif algo == "saute_ppo":
        replace_kwargs: dict[str, Any] = {"algo": algo}
        if "saute_budget" in config:
            replace_kwargs["cost_threshold"] = float(config["saute_budget"])
        if "saute_horizon" in config:
            replace_kwargs["saute_horizon"] = int(config["saute_horizon"])
        if "saute_unsafe_reward" in config:
            replace_kwargs["saute_unsafe_reward"] = float(
                config["saute_unsafe_reward"]
            )
        if "saute_use_reward_shaping" in config:
            replace_kwargs["saute_use_reward_shaping"] = bool(
                config["saute_use_reward_shaping"]
            )
        cfg = cfg.replace(**replace_kwargs)

    if overrides:
        cfg = cfg.replace(**overrides)

    return cfg


def rollout_bound_wrapper(
    wrapper,
    key,
    policy_fn,
    *,
    max_steps: int,
    info_keys: dict[str, str],
):
    """Roll out a params-bound wrapper and collect selected ``info`` fields."""
    import jax
    import jax.numpy as jnp

    obs0, state0 = wrapper.reset(key)

    def body(carry, _):
        obs, state, k = carry
        k, k_step, k_policy = jax.random.split(k, 3)
        action = policy_fn(obs, state, k_policy)
        step_out = wrapper.step(k_step, state, action)
        if len(step_out) == 5:
            obs2, state2, reward, done, info = step_out
        else:
            obs2, state2, reward, _costs, done, info = step_out
        del done
        out = {"rewards": jnp.asarray(reward)}
        for out_key, info_key in info_keys.items():
            out[out_key] = jnp.asarray(info[info_key])
        return (obs2, state2, k), out

    _, results = jax.lax.scan(
        body, (obs0, state0, key), xs=None, length=int(max_steps)
    )
    return results
