"""Base environment classes: EnvState, EnvParams, Environment."""

from abc import ABC, abstractmethod
from typing import Tuple, Any, Dict

import jax
import jax.numpy as jnp
import chex
from flax import struct

from powerzoojax.envs.spaces import Space


@struct.dataclass
class EnvState:
    """Base environment state (immutable PyTree).

    All subclasses should use @struct.dataclass decorator.
    """
    time_step: chex.Array  # int32 scalar
    done: chex.Array       # bool scalar


@struct.dataclass
class EnvParams:
    """Base environment parameters (immutable PyTree).

    Static configuration that doesn't change during an episode.
    ``max_steps`` and ``delta_t_hours`` are marked as non-leaf so they
    are part of the pytree structure (static under JIT).
    """
    max_steps: int = struct.field(pytree_node=False, default=48)
    delta_t_hours: float = struct.field(pytree_node=False, default=0.5)


class Environment(ABC):
    """Abstract base class for JAX-based environments.

    Follows Gymnax-style API: pure functions, explicit state, JIT-compatible.
    """

    # RL interface

    @abstractmethod
    def reset(
        self,
        key: chex.PRNGKey,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment to initial state."""
        pass

    @abstractmethod
    def step(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: chex.Array,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, chex.Array, chex.Array, chex.Array, Dict[str, Any]]:
        """Execute one environment step -> (obs, state, reward, costs, done, info).

        info values must be JAX arrays or Python scalars so that the dict
        remains a valid pytree under vmap / pmap batching.

        ``costs`` is the explicit CMDP constraint vector for this env. Its
        ordering must match ``self.constraint_names(params)`` and remain
        static under JIT / vmap. Zero-constraint envs should return an empty
        float32 vector with shape ``(0,)``.

        **PRNG contract for auto-reset**: subclass implementations that embed
        auto-reset logic (``jnp.where(done, reset_state, next_state)``) MUST
        call ``jax.random.split(key)`` to obtain a *separate* key for
        ``self.reset()``.  Reusing the same ``key`` for both transition noise
        and reset would correlate the new episode's initial state with the
        last step's randomness, breaking MDP independence.
        """
        pass

    def step_auto_reset(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: chex.Array,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, chex.Array, chex.Array, chex.Array, Dict[str, Any]]:
        """Step with automatic reset on done -- enables lax.scan rollouts.

        All env subclasses already embed auto-reset inside ``step()``.
        This wrapper applies ``stop_gradient`` to the returned observation
        and state so gradients do not flow across episode boundaries
        (correct for model-free RL).

        Note: the underlying ``step()`` must split ``key`` before using it
        for reset (see ``step()`` docstring).  Do NOT pass the same ``key``
        to both transition randomness and ``reset()``.
        """
        obs, new_state, reward, costs, done, info = self.step(
            key, state, action, params
        )
        obs = jax.lax.stop_gradient(obs)
        new_state = jax.lax.stop_gradient(new_state)
        return obs, new_state, reward, costs, done, info

    # Spaces & observation

    @abstractmethod
    def observation_space(self, params: EnvParams) -> Space:
        """Return observation space specification."""
        pass

    @abstractmethod
    def action_space(self, params: EnvParams) -> Space:
        """Return action space specification."""
        pass

    def _get_obs(self, state: EnvState, params: EnvParams) -> chex.Array:
        """Extract observation from state. Override in subclass."""
        raise NotImplementedError

    # Helpers (override in subclass)

    def _compute_reward(
        self,
        state: EnvState,
        action: chex.Array,
        next_state: EnvState,
        params: EnvParams,
    ) -> chex.Array:
        """Compute reward for transition. Override in subclass."""
        raise NotImplementedError

    def _compute_cost(
        self,
        state: EnvState,
        action: chex.Array,
        next_state: EnvState,
        params: EnvParams,
    ) -> chex.Array:
        """Compute scalar legacy constraint cost for the transition.

        Reward must not carry safety penalties; physical constraint violations
        (e.g. voltage / thermal / SOC bounds) go into explicit CMDP cost
        channels. New implementations should override ``step()`` directly and
        return a full vector. This helper remains as a backward-compatible
        hook for simple scalar-cost envs.
        """
        return jnp.float32(0.0)

    def constraint_names(self, params: EnvParams) -> tuple[str, ...]:
        """Static names for the explicit CMDP constraint vector.

        The tuple order must match the ``costs`` vector returned by ``step()``.
        Zero-constraint envs should return ``()``.
        """
        return ()

    def _is_done(self, state: EnvState, params: EnvParams) -> chex.Array:
        """Check if episode is done. Default: time_step >= max_steps."""
        return state.time_step >= params.max_steps

    def _build_info(self, state: EnvState, params: EnvParams) -> Dict[str, Any]:
        """Build info dictionary. Override to add environment-specific info."""
        return {}

    # Status & diagnostics

    @property
    def name(self) -> str:
        """Environment name for registration."""
        return self.__class__.__name__

    def default_params(self) -> EnvParams:
        """Return default parameters. Override for environment-specific defaults."""
        return EnvParams()

    # Called at setup time only, never inside JIT.

    @property
    def obs_size(self) -> int:
        """Observation dimension (uses default_params)."""
        space = self.observation_space(self.default_params())
        return int(space.shape[0]) if space.shape else 1

    @property
    def num_actions(self) -> int:
        """Action dimension. Discrete: returns n. Continuous: returns shape[0]."""
        space = self.action_space(self.default_params())
        if hasattr(space, 'n'):
            return int(space.n)
        return int(space.shape[0]) if space.shape else 1

    @property
    def action_size(self) -> int:
        """Alias for num_actions."""
        return self.num_actions


# Utility functions


def denormalize_action(
    action: chex.Array, low: chex.Array, high: chex.Array,
) -> chex.Array:
    """Map normalized action from [-1, 1] to [low, high] (clipped)."""
    a = jnp.clip(action, -1.0, 1.0)
    return 0.5 * (low + high) + 0.5 * a * (high - low)


@jax.jit
def time_features(time_step: chex.Array, steps_per_day: int):
    """Sinusoidal time-of-day encoding -> (sin, cos)."""
    phase = 2.0 * jnp.pi * time_step / jnp.maximum(steps_per_day, 1)
    return jnp.sin(phase), jnp.cos(phase)


def empty_costs() -> chex.Array:
    """Canonical zero-length CMDP cost vector."""
    return jnp.zeros((0,), dtype=jnp.float32)


def stack_costs(*costs: chex.Array) -> chex.Array:
    """Build a float32 CMDP cost vector from scalar cost components."""
    if not costs:
        return empty_costs()
    return jnp.asarray(costs, dtype=jnp.float32)
