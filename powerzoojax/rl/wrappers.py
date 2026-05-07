"""RL training wrappers for CMDP-core PowerZooJax environments.

Core envs now expose the full CMDP step contract:

    step(key, state, action, params) -> (obs, state, reward, costs, done, info)

This module provides the library-facing adapters:

- ``LogWrapper`` binds params, keeps PureJaxRL/Rejax's 5-tuple interface,
  and projects vector costs into compatibility info fields.
- ``SafeRLWrapper`` binds params and returns the selected CMDP cost vector
  directly for multi-constraint PPO-Lagrangian training.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Dict, Sequence, Tuple, Union

import chex
import jax
import jax.numpy as jnp
from flax import struct

from powerzoojax.envs.base import Environment, EnvParams
from powerzoojax.envs.spaces import Box


def _constraint_names(env: Environment, params: EnvParams) -> tuple[str, ...]:
    names = tuple(env.constraint_names(params))
    if len(names) == 0:
        return ()
    return names


def _selected_indices(
    all_names: Sequence[str],
    selected_names: Sequence[str] | None,
) -> tuple[int, ...]:
    if selected_names is None:
        return tuple(range(len(all_names)))

    indices: list[int] = []
    for name in selected_names:
        if name not in all_names:
            raise ValueError(
                f"Unknown constraint name {name!r}. Available: {tuple(all_names)!r}"
            )
        indices.append(all_names.index(name))
    return tuple(indices)


def _broadcast_thresholds(
    cost_threshold: float | Sequence[float] | None,
    cost_thresholds: Sequence[float] | None,
    n_constraints: int,
) -> tuple[float, ...]:
    if cost_thresholds is not None:
        values = tuple(float(x) for x in cost_thresholds)
    elif cost_threshold is None:
        values = tuple(0.0 for _ in range(n_constraints))
    elif isinstance(cost_threshold, (tuple, list)):
        values = tuple(float(x) for x in cost_threshold)
    else:
        values = tuple(float(cost_threshold) for _ in range(n_constraints))

    if len(values) != n_constraints:
        raise ValueError(
            f"Expected {n_constraints} cost thresholds, got {len(values)}."
        )
    return values


def _select_costs(costs: chex.Array, indices: chex.Array) -> chex.Array:
    if indices.shape[0] == 0:
        return jnp.zeros((0,), dtype=jnp.float32)
    return jnp.take(costs, indices, axis=0)


@struct.dataclass
class LogEnvState:
    """Wraps inner env state with episode tracking fields."""

    env_state: Any
    episode_returns: chex.Array
    episode_lengths: chex.Array
    returned_episode_returns: chex.Array
    returned_episode_lengths: chex.Array


class LogWrapper:
    """PureJaxRL / Rejax adapter for single-agent PowerZooJax envs."""

    def __init__(self, env: Environment, params: EnvParams = None):
        self._env = env
        self._params = params if params is not None else env.default_params()
        self.constraint_names = _constraint_names(env, self._params)

    @property
    def obs_size(self) -> int:
        space = self._env.observation_space(self._params)
        return int(space.shape[0]) if space.shape else 1

    @property
    def num_actions(self) -> int:
        space = self._env.action_space(self._params)
        if hasattr(space, "n"):
            return int(space.n)
        return int(space.shape[0]) if space.shape else 1

    @property
    def action_size(self) -> int:
        return self.num_actions

    def observation_space(self):
        return self._env.observation_space(self._params)

    def action_space(self):
        return self._env.action_space(self._params)

    @property
    def name(self) -> str:
        return self._env.name

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, LogEnvState]:
        obs, env_state = self._env.reset(key, self._params)
        state = LogEnvState(
            env_state=env_state,
            episode_returns=jnp.float32(0.0),
            episode_lengths=jnp.int32(0),
            returned_episode_returns=jnp.float32(0.0),
            returned_episode_lengths=jnp.int32(0),
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: LogEnvState,
        action: chex.Array,
    ) -> Tuple[chex.Array, LogEnvState, chex.Array, chex.Array, Dict[str, Any]]:
        obs, env_state, reward, costs, done, info = self._env.step(
            key, state.env_state, action, self._params
        )

        obs = jax.lax.stop_gradient(obs)
        env_state = jax.lax.stop_gradient(env_state)
        costs = jax.lax.stop_gradient(costs)

        new_returns = state.episode_returns + reward
        new_lengths = state.episode_lengths + 1

        new_state = LogEnvState(
            env_state=env_state,
            episode_returns=new_returns * (1 - done),
            episode_lengths=new_lengths * (1 - done.astype(jnp.int32)),
            returned_episode_returns=jnp.where(
                done, new_returns, state.returned_episode_returns
            ),
            returned_episode_lengths=jnp.where(
                done, new_lengths, state.returned_episode_lengths
            ),
        )

        cost_sum = info.get("cost_sum", jnp.sum(costs))
        info = {
            **info,
            "constraint_costs": costs,
            "cost_sum": cost_sum,
            # Compatibility alias for legacy wrappers / dashboards.
            "cost": cost_sum,
            "returned_episode_returns": new_state.returned_episode_returns,
            "returned_episode_lengths": new_state.returned_episode_lengths,
            "returned_episode": done,
        }
        return obs, new_state, reward, done, info


@struct.dataclass
class SafeRLState:
    """Single-agent training state with vector CMDP cost tracking."""

    env_state: Any
    episode_returns: chex.Array
    episode_lengths: chex.Array
    returned_episode_returns: chex.Array
    returned_episode_lengths: chex.Array
    episode_costs: chex.Array
    returned_episode_costs: chex.Array


class SafeRLWrapper:
    """CMDP wrapper that exposes selected constraint costs as a vector output."""

    def __init__(
        self,
        env: Environment,
        params: EnvParams = None,
        cost_threshold: float | Sequence[float] | None = 25.0,
        *,
        selected_names: Sequence[str] | None = None,
        cost_thresholds: Sequence[float] | None = None,
    ):
        self._env = env
        self._params = params if params is not None else env.default_params()
        self.constraint_names = _constraint_names(env, self._params)
        self.selected_constraint_names = tuple(selected_names or self.constraint_names)
        indices = _selected_indices(self.constraint_names, self.selected_constraint_names)
        self._selected_indices = jnp.asarray(indices, dtype=jnp.int32)
        self.cost_thresholds = _broadcast_thresholds(
            cost_threshold, cost_thresholds, len(indices)
        )
        self.cost_threshold = float(cost_threshold) if not isinstance(
            cost_threshold, (tuple, list)
        ) and cost_threshold is not None else (
            self.cost_thresholds[0] if len(self.cost_thresholds) == 1 else 0.0
        )

    @property
    def obs_size(self) -> int:
        space = self._env.observation_space(self._params)
        return int(space.shape[0]) if space.shape else 1

    @property
    def num_actions(self) -> int:
        space = self._env.action_space(self._params)
        if hasattr(space, "n"):
            return int(space.n)
        return int(space.shape[0]) if space.shape else 1

    @property
    def action_size(self) -> int:
        return self.num_actions

    @property
    def num_constraints(self) -> int:
        return int(self._selected_indices.shape[0])

    def observation_space(self):
        return self._env.observation_space(self._params)

    def action_space(self):
        return self._env.action_space(self._params)

    @property
    def name(self) -> str:
        return self._env.name

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, SafeRLState]:
        obs, env_state = self._env.reset(key, self._params)
        zeros = jnp.zeros((self.num_constraints,), dtype=jnp.float32)
        state = SafeRLState(
            env_state=env_state,
            episode_returns=jnp.float32(0.0),
            episode_lengths=jnp.int32(0),
            returned_episode_returns=jnp.float32(0.0),
            returned_episode_lengths=jnp.int32(0),
            episode_costs=zeros,
            returned_episode_costs=zeros,
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: SafeRLState,
        action: chex.Array,
    ) -> Tuple[chex.Array, SafeRLState, chex.Array, chex.Array, chex.Array, Dict[str, Any]]:
        obs, env_state, reward, costs_all, done, info = self._env.step(
            key, state.env_state, action, self._params
        )
        obs = jax.lax.stop_gradient(obs)
        env_state = jax.lax.stop_gradient(env_state)
        costs_all = jax.lax.stop_gradient(costs_all)

        selected_costs = _select_costs(costs_all, self._selected_indices)
        cost_sum = jnp.sum(selected_costs)

        new_returns = state.episode_returns + reward
        new_lengths = state.episode_lengths + 1
        new_costs = state.episode_costs + selected_costs
        zero_costs = jnp.zeros_like(new_costs)

        new_state = SafeRLState(
            env_state=env_state,
            episode_returns=new_returns * (1 - done),
            episode_lengths=new_lengths * (1 - done.astype(jnp.int32)),
            returned_episode_returns=jnp.where(
                done, new_returns, state.returned_episode_returns
            ),
            returned_episode_lengths=jnp.where(
                done, new_lengths, state.returned_episode_lengths
            ),
            episode_costs=jnp.where(done, zero_costs, new_costs),
            returned_episode_costs=jnp.where(
                done,
                new_costs,
                state.returned_episode_costs,
            ),
        )

        info = {
            **info,
            "constraint_costs_all": costs_all,
            "constraint_costs": selected_costs,
            "cost_sum": cost_sum,
            # Compatibility alias for scalar-cost consumers.
            "cost": cost_sum,
            "returned_episode_returns": new_state.returned_episode_returns,
            "returned_episode_lengths": new_state.returned_episode_lengths,
            "returned_episode": done,
            "returned_episode_costs": new_state.returned_episode_costs,
            "returned_episode_cost_sum": jnp.sum(new_state.returned_episode_costs),
        }
        return obs, new_state, reward, selected_costs, done, info


class SauteWrapper:
    """SautéRL wrapper: augments obs with safety-budget signal, enables plain PPO.

    Appends ``z_i = (d_i - episode_cost_i) / horizon`` for each selected
    constraint to the observation vector.  The augmented obs lets a standard
    PPO agent (``algo="saute_ppo"``) learn safety-aware behaviour without any
    Lagrangian dual variable or secondary optimizer.

    Optional reward shaping (``use_reward_shaping=True``): when any constraint
    budget is exhausted (``z_i ≤ 0``), the step reward is replaced by
    ``unsafe_reward`` (default 0).

    Reference: Sootla et al., "Sauté RL: Almost Surely Safe RL Using State Augmentation"
    (ICML 2022).  The normalization here uses *horizon* instead of budget so
    that ``z`` carries the intuitive unit of "budget slack per remaining step".

    Reuses :class:`SafeRLState` for episode tracking — no extra pytree leaf.
    """

    def __init__(
        self,
        env: Environment,
        params: EnvParams = None,
        cost_threshold: float | Sequence[float] | None = 25.0,
        *,
        selected_names: Sequence[str] | None = None,
        cost_thresholds: Sequence[float] | None = None,
        horizon: int | None = None,
        unsafe_reward: float = 0.0,
        use_reward_shaping: bool = True,
    ):
        self._env = env
        self._params = params if params is not None else env.default_params()
        self.constraint_names = _constraint_names(env, self._params)
        self.selected_constraint_names = tuple(selected_names or self.constraint_names)
        indices = _selected_indices(self.constraint_names, self.selected_constraint_names)
        self._selected_indices = jnp.asarray(indices, dtype=jnp.int32)
        self.cost_thresholds = _broadcast_thresholds(
            cost_threshold, cost_thresholds, len(indices)
        )
        self.cost_threshold = (
            float(cost_threshold)
            if not isinstance(cost_threshold, (tuple, list)) and cost_threshold is not None
            else (self.cost_thresholds[0] if len(self.cost_thresholds) == 1 else 0.0)
        )
        _h = horizon if horizon is not None else getattr(self._params, "max_steps", 48)
        self._horizon = float(_h)
        self.unsafe_reward = float(unsafe_reward)
        self.use_reward_shaping = bool(use_reward_shaping)
        self._thresholds_arr = jnp.asarray(self.cost_thresholds, dtype=jnp.float32)

        # Build augmented observation space so Rejax can read the correct shape.
        base_space = self._env.observation_space(self._params)
        n_aug = len(indices)
        aug_dim = (int(base_space.shape[0]) if base_space.shape else 1) + n_aug
        self._obs_space = Box(
            low=jnp.full((aug_dim,), -jnp.inf, dtype=jnp.float32),
            high=jnp.full((aug_dim,), jnp.inf, dtype=jnp.float32),
            shape=(aug_dim,),
            dtype=jnp.float32,
        )

    @property
    def obs_size(self) -> int:
        return int(self._obs_space.shape[0])

    @property
    def num_actions(self) -> int:
        space = self._env.action_space(self._params)
        if hasattr(space, "n"):
            return int(space.n)
        return int(space.shape[0]) if space.shape else 1

    @property
    def action_size(self) -> int:
        return self.num_actions

    @property
    def num_constraints(self) -> int:
        return int(self._selected_indices.shape[0])

    def observation_space(self):
        return self._obs_space  # augmented: base_obs + K budget scalars

    def action_space(self):
        return self._env.action_space(self._params)

    @property
    def name(self) -> str:
        return self._env.name

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, SafeRLState]:
        obs, env_state = self._env.reset(key, self._params)
        zeros = jnp.zeros((self.num_constraints,), dtype=jnp.float32)
        z = self._thresholds_arr / self._horizon  # full budget at episode start
        aug_obs = jnp.concatenate([obs, z])
        state = SafeRLState(
            env_state=env_state,
            episode_returns=jnp.float32(0.0),
            episode_lengths=jnp.int32(0),
            returned_episode_returns=jnp.float32(0.0),
            returned_episode_lengths=jnp.int32(0),
            episode_costs=zeros,
            returned_episode_costs=zeros,
        )
        return aug_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: SafeRLState,
        action: chex.Array,
    ) -> Tuple[chex.Array, SafeRLState, chex.Array, chex.Array, Dict[str, Any]]:
        obs, env_state, reward, costs_all, done, info = self._env.step(
            key, state.env_state, action, self._params
        )
        obs = jax.lax.stop_gradient(obs)
        env_state = jax.lax.stop_gradient(env_state)
        costs_all = jax.lax.stop_gradient(costs_all)

        selected_costs = _select_costs(costs_all, self._selected_indices)
        cost_sum = jnp.sum(selected_costs)
        new_episode_costs = state.episode_costs + selected_costs

        z = (self._thresholds_arr - new_episode_costs) / self._horizon
        aug_obs = jnp.concatenate([obs, z])

        if self.use_reward_shaping and self.num_constraints > 0:
            reward = jnp.where(jnp.any(z <= 0.0), jnp.float32(self.unsafe_reward), reward)

        new_returns = state.episode_returns + reward
        new_lengths = state.episode_lengths + 1
        zero_costs = jnp.zeros_like(new_episode_costs)

        new_state = SafeRLState(
            env_state=env_state,
            episode_returns=new_returns * (1 - done),
            episode_lengths=new_lengths * (1 - done.astype(jnp.int32)),
            returned_episode_returns=jnp.where(
                done, new_returns, state.returned_episode_returns
            ),
            returned_episode_lengths=jnp.where(
                done, new_lengths, state.returned_episode_lengths
            ),
            episode_costs=jnp.where(done, zero_costs, new_episode_costs),
            returned_episode_costs=jnp.where(
                done, new_episode_costs, state.returned_episode_costs
            ),
        )

        info = {
            **info,
            "constraint_costs_all": costs_all,
            "constraint_costs": selected_costs,
            "cost_sum": cost_sum,
            "cost": cost_sum,
            "safety_budget_remaining": z,
            "returned_episode_returns": new_state.returned_episode_returns,
            "returned_episode_lengths": new_state.returned_episode_lengths,
            "returned_episode": done,
            "returned_episode_costs": new_state.returned_episode_costs,
            "returned_episode_cost_sum": jnp.sum(new_state.returned_episode_costs),
        }
        return aug_obs, new_state, reward, done, info


class PenaltyRewardWrapper:
    """Adds a fixed per-step penalty λ·reward_scale·Σ(costs) to the CMDP reward.

    Shaping formula::

        r' = r − penalty_lambda * reward_scale * Σ(costs)

    where:

    - ``r``              – raw env reward (already scaled by env's ``reward_scale``).
    - ``costs``          – CMDP constraint cost vector in physical units (e.g. MW).
    - ``penalty_lambda`` – dimensionless ablation sweep parameter (e.g. 10 / 100 / 1000).
    - ``reward_scale``   – same factor already applied by the env (default 1.0).
                           Bring MW-scale costs into the same unit space as reward.

    λ calibration for TSO (``reward_scale=1e-4``, step reward ≈ −7.5 GBP,
    CMDP costs 0–500 MW/violated step):

    - λ=10  → effective coeff 10×1e-4=1e-3; max penalty ≈ 0.5   — underfitting
    - λ=100 → coeff 1e-2;  max penalty ≈ 5.0 — comparable to reward
    - λ=1000 → coeff 1e-1; max penalty ≈ 50  — safety dominates (over-penalised)

    Usage::

        LogWrapper(PenaltyRewardWrapper(env, penalty_lambda=100.0, reward_scale=1e-4), params)

    The original ``costs`` and ``info`` are passed through unchanged so that
    eval / cost-channel tracking (safety metrics, per-episode shortfall) remain
    unaffected.  Only the scalar reward returned to the RL trainer changes.
    The unpenalised reward is preserved in ``info["unpenalized_reward"]``.
    """

    def __init__(
        self,
        env: Environment,
        penalty_lambda: float,
        reward_scale: float = 1.0,
    ):
        self._env = env
        self.penalty_lambda = float(penalty_lambda)
        self.reward_scale = float(reward_scale)
        # Pre-compute effective scalar coefficient; captured as a Python float
        # so JIT traces it as a compile-time constant (self is static_argnums).
        self._coeff = float(penalty_lambda * reward_scale)

    # ── Delegation to inner env ───────────────────────────────────────────

    def default_params(self) -> Any:
        return self._env.default_params()

    def constraint_names(self, params: Any) -> tuple[str, ...]:
        return self._env.constraint_names(params)

    def observation_space(self, params: Any):
        return self._env.observation_space(params)

    def action_space(self, params: Any):
        return self._env.action_space(params)

    @property
    def name(self) -> str:
        return self._env.name

    # ── CMDP interface ────────────────────────────────────────────────────

    def reset(
        self,
        key: chex.PRNGKey,
        params: Any,
    ) -> Tuple[chex.Array, Any]:
        return self._env.reset(key, params)

    def step(
        self,
        key: chex.PRNGKey,
        state: Any,
        action: chex.Array,
        params: Any,
    ) -> Tuple[chex.Array, Any, chex.Array, chex.Array, chex.Array, Dict[str, Any]]:
        obs, new_state, reward, costs, done, info = self._env.step(
            key, state, action, params
        )
        penalized_reward = reward - jnp.float32(self._coeff) * jnp.sum(costs)
        info = {**info, "unpenalized_reward": reward}
        return obs, new_state, penalized_reward, costs, done, info


def bind(
    env: Environment,
    params: EnvParams = None,
    safe: bool = False,
    cost_threshold: float | Sequence[float] | None = 25.0,
    *,
    selected_names: Sequence[str] | None = None,
    cost_thresholds: Sequence[float] | None = None,
) -> Union["LogWrapper", "SafeRLWrapper"]:
    """Create a training-compatible wrapper around a core env."""

    if safe:
        return SafeRLWrapper(
            env,
            params,
            cost_threshold=cost_threshold,
            selected_names=selected_names,
            cost_thresholds=cost_thresholds,
        )
    return LogWrapper(env, params)
