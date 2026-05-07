"""Trainer dispatcher — route to Rejax, CMDP, or MARL backends.

Usage::

    from powerzoojax.rl.trainer import make_train, TrainResult
    from powerzoojax.rl.config import TrainConfig

    train_fn = make_train(env, TrainConfig(algo="ppo", total_timesteps=200_000))
    result = train_fn(jax.random.PRNGKey(42))
    print(result.summary)
"""

import json
import time
from types import SimpleNamespace
from typing import Any, Callable, Dict, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import io_callback
from flax import linen as nn

from powerzoojax.rl.config import TrainConfig
from powerzoojax.rl.policies import BoundedBetaPolicy


class _EvalMetricsState(NamedTuple):
    rng: Any
    env_state: Any
    last_obs: Any
    done: Any
    return_: Any
    length: Any
    total_cost: Any
    voltage_violation_total: Any
    reserve_shortfall_total: Any
    thermal_overload_total: Any
    voltage_violation_steps: Any
    reserve_shortfall_steps: Any
    thermal_violation_steps: Any


# ============ Rejax Adapter ============

def _get_max_steps(env) -> int:
    """Extract max_steps from a wrapped env's bound params (best-effort)."""
    def _from_params(params) -> int | None:
        if params is None:
            return None
        if hasattr(params, "max_steps"):
            return int(params.max_steps)
        # Resource-composition envs (e.g. DC microgrid) often nest the real
        # episode horizon under a sub-config such as ``params.dc.max_steps``.
        for attr in ("dc", "grid", "dist"):
            sub = getattr(params, attr, None)
            if sub is not None and hasattr(sub, "max_steps"):
                return int(sub.max_steps)
        return None

    if hasattr(env, '_params'):
        max_steps = _from_params(env._params)
        if max_steps is not None:
            return max_steps
    # RewardWrapper wraps a LogWrapper
    if hasattr(env, '_env'):
        return _get_max_steps(env._env)
    # SafeRLWrapper delegates to _log_wrapper
    if hasattr(env, '_log_wrapper'):
        return _get_max_steps(env._log_wrapper)
    return 200  # conservative fallback


class _RejaxAdapter:
    """Shim that converts our PureJaxRL-style wrappers to Rejax's gymnax-style API.

    Rejax calls ``env.step(key, state, action, params)`` (4-arg) and
    ``env.reset(key, params)`` (2-arg).  Our ``LogWrapper`` / ``RewardWrapper``
    use ``step(key, state, action)`` and ``reset(key)`` (no params).

    This adapter:
    - Accepts the extra ``params`` argument in step/reset/action_space/observation_space
      and ignores it (params are already bound inside the wrapped env).
    - Exposes ``default_params`` with ``max_steps_in_episode`` so Rejax's built-in
      evaluation callback can determine episode length.
    """

    def __init__(self, env):
        self._wrapped = env
        self.default_params = SimpleNamespace(
            max_steps_in_episode=_get_max_steps(env)
        )

    @property
    def obs_size(self) -> int:
        return self._wrapped.obs_size

    @property
    def num_actions(self) -> int:
        return self._wrapped.num_actions

    @property
    def action_size(self) -> int:
        return self._wrapped.action_size

    def action_space(self, params=None):
        return self._wrapped.action_space()

    def observation_space(self, params=None):
        return self._wrapped.observation_space()

    def reset(self, key, params=None):
        return self._wrapped.reset(key)

    def step(self, key, state, action, params=None):
        return self._wrapped.step(key, state, action)


def _rejax_create_kwargs(algo_cls, config: TrainConfig) -> dict:
    """Build ``AlgoCls.create`` kwargs compatible with the installed Rejax.

    Some Rejax releases expose flags like ``normalize_rewards`` as algorithm
    dataclass fields, while older releases do not. Keep our ``TrainConfig``
    stable and drop only the kwargs unsupported by the local ``algo_cls``.
    """
    kwargs = config.to_rejax_kwargs()
    dataclass_fields = getattr(algo_cls, "__dataclass_fields__", None)
    if not dataclass_fields:
        return kwargs

    # ``agent_kwargs`` is consumed inside ``create_agent`` and is not a field on
    # the final algorithm object, so we always keep it when present.
    supported_keys = set(dataclass_fields) | {"agent_kwargs", "hidden_layer_sizes"}
    return {k: v for k, v in kwargs.items() if k in supported_keys}


def _rejax_actor_override(config: TrainConfig, adapted_env) -> Any | None:
    """Return an explicit actor override when the config requests one."""
    if config.algo not in ("ppo", "saute_ppo"):
        return None
    if config.continuous_action_dist != "beta":
        return None

    action_space = adapted_env.action_space(None)
    if hasattr(action_space, "n"):
        return None

    return BoundedBetaPolicy(
        int(jnp.prod(jnp.asarray(action_space.shape))),
        (jnp.asarray(action_space.low), jnp.asarray(action_space.high)),
        tuple(config.hidden_dims),
        nn.swish,
    )


def _actor_dist_mode(actor, obs):
    """Return a distribution mode/mean for deterministic evaluation."""
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


def _make_rejax_eval_act(algo_obj, train_state, env, env_params, *, deterministic: bool):
    """Build the action function used by Rejax monitor evaluation.

    Rejax PPO's stock ``make_act`` samples from the actor distribution. That is
    correct for training rollouts, but benchmark monitor evaluation must match
    SB3/SBX's deterministic ``model.predict(..., deterministic=True)`` protocol.
    """
    if not deterministic:
        return algo_obj.make_act(train_state)

    action_space = env.action_space(env_params)

    def act(obs, _rng):
        if getattr(algo_obj, "normalize_observations", False):
            obs = algo_obj.normalize_obs(train_state.obs_rms_state, obs)

        obs = jnp.expand_dims(obs, 0)
        action = algo_obj.actor.apply(
            train_state.actor_ts.params,
            obs,
            method=_actor_dist_mode,
        )
        if not hasattr(action_space, "n"):
            action = jnp.clip(action, action_space.low, action_space.high)
        return jnp.squeeze(action)

    return act


def _evaluate_with_metrics(
    act,
    rng,
    env,
    env_params,
    num_seeds: int,
    max_steps_in_episode: int,
):
    """Rejax-style eval with episode safety metrics when env info exposes them."""

    zero = jnp.float32(0.0)

    def _info_scalar(info, key: str):
        if key not in info:
            return zero
        return jnp.asarray(info[key], dtype=jnp.float32).squeeze()

    def evaluate_single(seed):
        rng_reset, rng_eval = jax.random.split(seed)
        obs, env_state = env.reset(rng_reset, env_params)
        state = _EvalMetricsState(
            rng=rng_eval,
            env_state=env_state,
            last_obs=obs,
            done=jnp.bool_(False),
            return_=jnp.float32(0.0),
            length=jnp.int32(0),
            total_cost=zero,
            voltage_violation_total=zero,
            reserve_shortfall_total=zero,
            thermal_overload_total=zero,
            voltage_violation_steps=zero,
            reserve_shortfall_steps=zero,
            thermal_violation_steps=zero,
        )

        def cond_fn(carry):
            return jnp.logical_and(
                carry.length < max_steps_in_episode, jnp.logical_not(carry.done)
            )

        def body_fn(carry):
            rng_step, rng_act, rng_env = jax.random.split(carry.rng, 3)
            action = act(carry.last_obs, rng_act)
            obs2, env_state2, reward, done2, info = env.step(
                rng_env, carry.env_state, action, env_params
            )
            total_cost = _info_scalar(info, "cost")
            voltage_violation = _info_scalar(info, "cost_voltage_violation")
            reserve_shortfall = _info_scalar(info, "reserve_shortfall")
            thermal_overload = _info_scalar(info, "cost_thermal_overload")
            return _EvalMetricsState(
                rng=rng_step,
                env_state=env_state2,
                last_obs=obs2,
                done=done2,
                return_=carry.return_ + jnp.asarray(reward).squeeze(),
                length=carry.length + 1,
                total_cost=carry.total_cost + total_cost,
                voltage_violation_total=carry.voltage_violation_total
                + voltage_violation,
                reserve_shortfall_total=carry.reserve_shortfall_total
                + jnp.asarray(reserve_shortfall).squeeze(),
                thermal_overload_total=carry.thermal_overload_total
                + jnp.asarray(thermal_overload).squeeze(),
                voltage_violation_steps=carry.voltage_violation_steps
                + (voltage_violation > jnp.float32(1e-6)).astype(jnp.float32),
                reserve_shortfall_steps=carry.reserve_shortfall_steps
                + (jnp.asarray(reserve_shortfall).squeeze() > jnp.float32(1e-6)).astype(jnp.float32),
                thermal_violation_steps=carry.thermal_violation_steps
                + (jnp.asarray(thermal_overload).squeeze() > jnp.float32(1e-6)).astype(jnp.float32),
            )

        final = jax.lax.while_loop(cond_fn, body_fn, state)
        return (
            final.length,
            final.return_,
            final.total_cost,
            final.voltage_violation_total,
            final.reserve_shortfall_total,
            final.thermal_overload_total,
            final.voltage_violation_steps,
            final.reserve_shortfall_steps,
            final.thermal_violation_steps,
        )

    seeds = jax.random.split(rng, num_seeds)
    (
        lengths,
        returns,
        total_costs,
        voltage_totals,
        reserve_totals,
        thermal_totals,
        voltage_step_counts,
        reserve_step_counts,
        thermal_step_counts,
    ) = jax.vmap(evaluate_single)(seeds)
    lengths_f = jnp.maximum(lengths.astype(jnp.float32), 1.0)
    total_eval_steps = jnp.maximum(jnp.sum(lengths_f), 1.0)
    return {
        "eval_episode_lengths": lengths,
        "eval_episode_returns": returns,
        "eval_returns": jnp.mean(returns),
        "eval_total_cost": jnp.mean(total_costs),
        "eval_cost_per_step": jnp.mean(total_costs / lengths_f),
        "eval_total_voltage_violation": jnp.mean(voltage_totals),
        "eval_cost_voltage_violation": jnp.mean(voltage_totals / lengths_f),
        "eval_voltage_violation_rate": jnp.sum(voltage_step_counts) / total_eval_steps,
        "eval_voltage_violation_episode_rate": jnp.mean(
            voltage_totals > jnp.float32(1e-6)
        ),
        "eval_total_reserve_shortfall": jnp.mean(reserve_totals),
        "eval_total_thermal_overload": jnp.mean(thermal_totals),
        "eval_reserve_shortfall_rate": jnp.sum(reserve_step_counts) / total_eval_steps,
        "eval_reserve_shortfall_episode_rate": jnp.mean(
            reserve_totals > jnp.float32(1e-6)
        ),
        "eval_thermal_violation_rate": jnp.sum(thermal_step_counts) / total_eval_steps,
        "eval_thermal_violation_episode_rate": jnp.mean(
            thermal_totals > jnp.float32(1e-6)
        ),
    }


# ============ TrainResult ============

class TrainResult:
    """Structured container for training results.

    Attributes:
        params:   Trained network parameters (pytree).
        metrics:  Dict of metric arrays with shape ``(n_updates, ...)``.
        config:   The ``TrainConfig`` used for this run.
        env_name: Optional name of the environment.
    """

    def __init__(
        self,
        params: Any,
        metrics: Dict[str, Any],
        config: TrainConfig,
        env_name: str = "",
        checkpoints: Any = None,
    ):
        self.params = params
        self.metrics = metrics
        self.config = config
        self.env_name = env_name
        # Optional ``[(step, params_or_train_state), ...]`` if intermediate
        # checkpoints were captured.
        self.checkpoints = checkpoints

    @property
    def summary(self) -> dict:
        """Final training metrics as a plain dict (JSON-serializable)."""
        m = self.metrics
        if isinstance(m, dict):
            reward_key = "eval_returns" if "eval_returns" in m else "mean_reward"
            rewards = jnp.asarray(m.get(reward_key, jnp.array([]))).flatten()
            costs = jnp.asarray(m.get("episode_cost_est", jnp.array([]))).flatten()
        else:
            # Rejax raw tuple (lengths, returns) — should not occur after the
            # fix in make_rejax_train, but handle defensively.
            _, episode_returns = m
            rewards = jnp.mean(jnp.asarray(episode_returns), axis=-1).flatten()
            costs = jnp.array([])

        return {
            "env": self.env_name,
            "algo": self.config.algo,
            "total_timesteps": self.config.total_timesteps,
            "seed": self.config.seed,
            "final_reward": float(rewards[-1]) if len(rewards) > 0 else None,
            "initial_reward": float(rewards[0]) if len(rewards) > 0 else None,
            "final_cost": float(costs[-1]) if len(costs) > 0 else None,
            "cost_threshold": self.config.cost_threshold,
            "cost_thresholds": list(self.config.cost_thresholds),
        }

    def save(self, path: str) -> None:
        """Save summary + config to a JSON file."""
        data = {"summary": self.summary, "config": self.config._asdict()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)


# ============ Backend factories ============

def make_rejax_train(env, config: TrainConfig) -> Callable:
    """Create a Rejax training function for single-agent algorithms.

    Args:
        env: ``LogWrapper`` or ``RewardWrapper`` instance (PureJaxRL-style).
        config: ``TrainConfig`` with ``algo in ("ppo", "sac", "td3", "dqn")``.

    Returns:
        ``train_fn(key) -> TrainResult``

    Raises:
        ImportError: If ``rejax`` is not installed.
    """
    try:
        import rejax
    except ImportError:
        raise ImportError(
            "Rejax is required for algo='ppo'/'sac'/'td3'/'dqn'. "
            "Install with: pip install powerzoojax[rl]  or  pip install rejax"
        )

    algo_map = {
        "ppo": rejax.PPO,
        "saute_ppo": rejax.PPO,  # SautéRL uses plain PPO on an augmented-obs env
        "sac": rejax.SAC,
        "td3": rejax.TD3,
        "dqn": rejax.DQN,
    }
    if config.algo not in algo_map:
        raise ValueError(
            f"Rejax backend does not support algo='{config.algo}'. "
            f"Supported: {list(algo_map.keys())}"
        )

    AlgoCls = algo_map[config.algo]
    adapted_env = _RejaxAdapter(env)
    max_steps = _get_max_steps(env)
    n_ckpt = max(1, int(getattr(config, "n_checkpoints", 1)))

    # Rejax's stock eval_callback passes ``env_params`` into a JIT-compiled
    # function.  Our ``env_params=None`` would be fine *at Python level*, but
    # Rejax's default callback reads ``algo.env_params.max_steps_in_episode``
    # and ``_RejaxAdapter.default_params`` is a plain ``SimpleNamespace`` which
    # cannot be traced by XLA.  Workaround: supply a custom callback that
    # calls ``rejax.evaluate`` with ``env_params=None`` and an explicit integer
    # ``max_steps_in_episode``.  This is safe because our adapter already
    # ignores ``env_params`` in ``reset`` / ``step``.
    n_eval_eps = int(
        getattr(
            config,
            "eval_num_episodes",
            getattr(config, "eval_episodes", 128),
        )
    )
    n_eval_eps = max(1, n_eval_eps)

    def _eval_callback_base(algo, ts, rng):
        act = _make_rejax_eval_act(
            algo,
            ts,
            adapted_env,
            None,
            deterministic=config.algo in ("ppo", "saute_ppo"),
        )
        return _evaluate_with_metrics(
            act,
            rng,
            adapted_env,
            None,
            n_eval_eps,
            max_steps,
        )

    env_name = getattr(env, "name", "")

    def _metrics_from_raw(raw_logs):
        if isinstance(raw_logs, dict):
            return raw_logs
        if isinstance(raw_logs, tuple) and len(raw_logs) == 2:
            lengths, returns = raw_logs
            return {
                "eval_episode_lengths": lengths,
                "eval_episode_returns": returns,
                "eval_returns": jnp.mean(returns, axis=-1),
            }
        return raw_logs

    if not config.record_eval_wall_time:
        create_kwargs = _rejax_create_kwargs(AlgoCls, config)
        actor_override = _rejax_actor_override(config, adapted_env)
        algo = AlgoCls.create(env=adapted_env, env_params=None, **create_kwargs)
        if actor_override is not None:
            algo = algo.replace(actor=actor_override)
        algo = algo.replace(eval_callback=_eval_callback_base)

        if n_ckpt > 1:
            iteration_steps = int(config.num_envs * config.n_steps)
            num_iterations = int(np.ceil(config.eval_freq / iteration_steps))
            num_evals = int(np.ceil(config.total_timesteps / config.eval_freq))
            steps_per_eval = int(num_iterations * iteration_steps)

            @jax.jit
            def run_eval_chunk(train_state):
                train_state = jax.lax.fori_loop(
                    0,
                    num_iterations,
                    lambda _, state: algo.train_iteration(state),
                    train_state,
                )
                eval_logs = algo.eval_callback(algo, train_state, train_state.rng)
                return train_state, eval_logs

            capture_iters = sorted({
                int(idx)
                for idx in np.linspace(1, num_evals, n_ckpt)
            })
            capture_set = set(capture_iters)

            def train_fn(key):
                train_state = algo.init_state(key)
                eval_logs_host: list[dict[str, Any]] = []
                checkpoints: list[tuple[int, Any]] = []
                checkpoint_walltimes: list[float] = []
                t0 = time.perf_counter()

                if not algo.skip_initial_evaluation:
                    eval_logs_host.append(
                        jax.tree.map(np.asarray, algo.eval_callback(algo, train_state, train_state.rng))
                    )

                for eval_idx in range(1, num_evals + 1):
                    train_state, eval_logs = run_eval_chunk(train_state)
                    eval_logs_host.append(jax.tree.map(np.asarray, eval_logs))
                    if eval_idx in capture_set:
                        checkpoints.append((eval_idx * steps_per_eval, train_state))
                        checkpoint_walltimes.append(time.perf_counter() - t0)

                metrics = jax.tree.map(
                    lambda *xs: np.stack(xs, axis=0),
                    *eval_logs_host,
                )
                metrics = {
                    **metrics,
                    "checkpoint_walltime_s": np.asarray(
                        checkpoint_walltimes, dtype=np.float32
                    ),
                }
                return TrainResult(
                    params=train_state,
                    metrics=metrics,
                    config=config,
                    env_name=env_name,
                    checkpoints=checkpoints,
                )

            return train_fn

        def train_fn(key):
            train_state, raw_logs = algo.train(key)
            metrics = _metrics_from_raw(raw_logs)
            return TrainResult(
                params=train_state,
                metrics=metrics,
                config=config,
                env_name=env_name,
            )

        return train_fn

    if n_ckpt > 1:
        create_kwargs = _rejax_create_kwargs(AlgoCls, config)
        actor_override = _rejax_actor_override(config, adapted_env)
        algo = AlgoCls.create(env=adapted_env, env_params=None, **create_kwargs)
        if actor_override is not None:
            algo = algo.replace(actor=actor_override)
        algo = algo.replace(eval_callback=_eval_callback_base)

        iteration_steps = int(config.num_envs * config.n_steps)
        num_iterations = int(np.ceil(config.eval_freq / iteration_steps))
        num_evals = int(np.ceil(config.total_timesteps / config.eval_freq))
        steps_per_eval = int(num_iterations * iteration_steps)

        @jax.jit
        def run_eval_chunk(train_state):
            train_state = jax.lax.fori_loop(
                0,
                num_iterations,
                lambda _, state: algo.train_iteration(state),
                train_state,
            )
            eval_logs = algo.eval_callback(algo, train_state, train_state.rng)
            return train_state, eval_logs

        capture_iters = sorted({
            int(idx)
            for idx in np.linspace(1, num_evals, n_ckpt)
        })
        capture_set = set(capture_iters)

        def train_fn(key):
            cfg_ts = int(config.total_timesteps)
            warm_ts = (
                config.wall_time_warmup_timesteps
                if config.wall_time_warmup_timesteps is not None
                else int(config.eval_freq)
            )
            warm_ts = max(1, min(int(warm_ts), cfg_ts))

            if config.wall_time_warmup and warm_ts < cfg_ts:
                warm_cfg = config.replace(
                    total_timesteps=warm_ts,
                    record_eval_wall_time=False,
                    n_checkpoints=1,
                )
                warm_fn = make_rejax_train(env, warm_cfg)
                warm_key = jax.random.fold_in(key, 0x9E3779B9)
                warm_res = warm_fn(warm_key)
                jax.block_until_ready(warm_res.params)

            train_state = algo.init_state(key)
            eval_logs_host: list[dict[str, Any]] = []
            wall_times: list[float] = []
            checkpoints: list[tuple[int, Any]] = []
            checkpoint_walltimes: list[float] = []
            t0 = time.perf_counter()

            if not algo.skip_initial_evaluation:
                eval_logs_host.append(
                    jax.tree.map(np.asarray, algo.eval_callback(algo, train_state, train_state.rng))
                )
                wall_times.append(time.perf_counter() - t0)

            for eval_idx in range(1, num_evals + 1):
                train_state, eval_logs = run_eval_chunk(train_state)
                eval_logs_host.append(jax.tree.map(np.asarray, eval_logs))
                wall_times.append(time.perf_counter() - t0)
                if eval_idx in capture_set:
                    checkpoints.append((eval_idx * steps_per_eval, train_state))
                    checkpoint_walltimes.append(time.perf_counter() - t0)

            metrics = jax.tree.map(
                lambda *xs: np.stack(xs, axis=0),
                *eval_logs_host,
            )
            metrics = {
                **metrics,
                "eval_wall_time_s": np.asarray(wall_times, dtype=np.float32),
                "checkpoint_walltime_s": np.asarray(
                    checkpoint_walltimes, dtype=np.float32
                ),
            }
            return TrainResult(
                params=train_state,
                metrics=metrics,
                config=config,
                env_name=env_name,
                checkpoints=checkpoints,
            )

        return train_fn

    # Wall time at each eval without splitting ``train()``: io_callback runs on host
    # when the device reaches each eval (same fused graph as default Rejax path).
    def train_fn(key):
        wall_times: list[float] = []
        cfg_ts = int(config.total_timesteps)
        warm_ts = (
            config.wall_time_warmup_timesteps
            if config.wall_time_warmup_timesteps is not None
            else int(config.eval_freq)
        )
        warm_ts = max(1, min(int(warm_ts), cfg_ts))

        if (
            config.wall_time_warmup
            and warm_ts < cfg_ts
        ):
            warm_cfg = config.replace(
                total_timesteps=warm_ts,
                record_eval_wall_time=False,
            )
            warm_fn = make_rejax_train(env, warm_cfg)
            warm_key = jax.random.fold_in(key, 0x9E3779B9)
            warm_res = warm_fn(warm_key)
            jax.block_until_ready(warm_res.params)

        t0 = time.perf_counter()

        def _eval_callback(algo, ts, rng):
            eval_logs = _eval_callback_base(algo, ts, rng)

            def _stamp(_):
                wall_times.append(time.perf_counter() - t0)

            io_callback(_stamp, (), jnp.array(0.0, dtype=jnp.float32))
            return eval_logs

        create_kwargs = _rejax_create_kwargs(AlgoCls, config)
        actor_override = _rejax_actor_override(config, adapted_env)
        algo = AlgoCls.create(env=adapted_env, env_params=None, **create_kwargs)
        if actor_override is not None:
            algo = algo.replace(actor=actor_override)
        algo = algo.replace(eval_callback=_eval_callback)
        train_state, raw_logs = algo.train(key)
        metrics = _metrics_from_raw(raw_logs)
        metrics = {
            **metrics,
            "eval_wall_time_s": jnp.asarray(wall_times, dtype=jnp.float32),
        }
        return TrainResult(
            params=train_state,
            metrics=metrics,
            config=config,
            env_name=env_name,
        )

    return train_fn


def make_jaxmarl_train(env, config: TrainConfig) -> Callable:
    """Create an IPPO / typed-IPPO / typed-IPPO-Lagrangian training function.

    Args:
        env:    GridMARLEnv instance (params bound at construction).
        config: TrainConfig with
            ``algo in ("ippo", "ippo_typed", "ippo_typed_lagrangian", "mappo")``.
            ``"ippo_typed"`` uses type-specific parameter sharing
            (``make_ippo_typed_train``); required for the DERs benchmark.

    Returns:
        ``train_fn(key) -> TrainResult``

    Raises:
        NotImplementedError: For algo="mappo" (use JaxMARL adapter directly).
        ValueError: For unknown MARL algorithms.
    """
    if config.algo == "ippo":
        from powerzoojax.rl.ippo import make_ippo_train
        return make_ippo_train(env, config)
    elif config.algo == "ippo_typed":
        from powerzoojax.rl.ippo import make_ippo_typed_train
        return make_ippo_typed_train(env, config)
    elif config.algo == "ippo_typed_lagrangian":
        from powerzoojax.rl.ippo import make_ippo_typed_lagrangian_train
        return make_ippo_typed_lagrangian_train(env, config)
    elif config.algo == "mappo":
        raise NotImplementedError(
            "MAPPO: use JaxMARL adapter (powerzoojax.rl.ippo covers IPPO only)."
        )
    else:
        raise ValueError(
            f"Unknown MARL algo: '{config.algo}'. Supported: 'ippo', "
            "'ippo_typed', 'ippo_typed_lagrangian'."
        )


def make_train(env, config: TrainConfig = TrainConfig()) -> Callable:
    """Unified training factory — dispatch to the appropriate backend.

    Args:
        env:    Wrapped environment (``LogWrapper``, ``SafeRLWrapper``, or
                ``RewardWrapper``).
        config: ``TrainConfig`` specifying the algorithm and hyperparameters.

    Returns:
        ``train_fn(key) -> TrainResult``

    Backend routing:
        - ``"ppo" | "sac" | "td3" | "dqn"``  →  Rejax (via ``_RejaxAdapter``)
        - ``"ppo_lagrangian"``                →  self-implemented CMDP
        - ``"ippo"``                          →  shared-param IPPO (all agents one net)
        - ``"ippo_typed"``                    →  type-specific IPPO (DERs default)
        - ``"ippo_typed_lagrangian"``         →  type-specific IPPO-Lagrangian
        - ``"saute_ppo"``                     →  Rejax PPO on a ``SauteWrapper`` env
        - ``"mappo"``                         →  NotImplementedError (pending)
    """
    if config.algo in ("ppo", "sac", "td3", "dqn", "saute_ppo"):
        return make_rejax_train(env, config)
    elif config.algo == "ppo_lagrangian":
        from powerzoojax.rl.cmdp import make_cmdp_train
        return make_cmdp_train(env, config)
    elif config.algo in ("ippo", "ippo_typed", "ippo_typed_lagrangian", "mappo"):
        return make_jaxmarl_train(env, config)
    else:
        raise ValueError(
            f"Unknown algo: '{config.algo}'. "
            "Supported: 'ppo', 'sac', 'td3', 'dqn', 'saute_ppo', 'ppo_lagrangian', "
            "'ippo', 'ippo_typed', 'ippo_typed_lagrangian', 'mappo'."
        )
