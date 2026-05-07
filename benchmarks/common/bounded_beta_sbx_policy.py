"""Bounded Beta PPO policy for the SBX backend.

This mirrors the bounded Beta actor family used by the JAX DC Microgrid PPO
trainer so the ``sbx`` backend can participate in a fair Phase-2 comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Sequence

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from flax import linen as nn
from sbx.common.jax_layers import NatureCNN
from sbx.ppo.policies import PPOPolicy

tfd = tfp.distributions


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class AffineBetaDiagDistribution:
    """Factorized Beta policy on an arbitrary finite Box action space.

    Actions are sampled on ``[0, 1]^d`` and then affinely mapped into the env
    bounds. Log-prob and entropy intentionally follow the JAX benchmark's
    bounded-Beta convention and are computed on the unit-space variables
    without adding the affine Jacobian term.
    """

    alpha: jax.Array
    beta: jax.Array
    action_low: jax.Array
    action_scale: jax.Array
    eps: float = 1e-6

    def tree_flatten(self):
        return (self.alpha, self.beta, self.action_low, self.action_scale), {"eps": self.eps}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        alpha, beta, action_low, action_scale = children
        return cls(alpha=alpha, beta=beta, action_low=action_low, action_scale=action_scale, **aux_data)

    def _unit_dist(self):
        return tfd.Beta(self.alpha, self.beta)

    def _to_env_action(self, unit_action: jax.Array) -> jax.Array:
        return self.action_low + unit_action * self.action_scale

    def _to_unit_action(self, action: jax.Array) -> jax.Array:
        unit = (action - self.action_low) / jnp.maximum(self.action_scale, self.eps)
        return jnp.clip(unit, self.eps, 1.0 - self.eps)

    def sample(self, seed: jax.Array) -> jax.Array:
        unit_action = self._unit_dist().sample(seed=seed)
        return self._to_env_action(unit_action)

    def log_prob(self, action: jax.Array) -> jax.Array:
        unit_action = self._to_unit_action(action)
        return jnp.sum(self._unit_dist().log_prob(unit_action), axis=-1)

    def entropy(self) -> jax.Array:
        return jnp.sum(self._unit_dist().entropy(), axis=-1)

    def mode(self) -> jax.Array:
        denom = jnp.maximum(self.alpha + self.beta - 2.0, self.eps)
        unit_mode = jnp.clip((self.alpha - 1.0) / denom, self.eps, 1.0 - self.eps)
        return self._to_env_action(unit_mode)


class SBXBoundedBetaActor(nn.Module):
    """Flax actor that emits a bounded Beta distribution over Box actions."""

    action_dim: int
    action_low: tuple[float, ...]
    action_high: tuple[float, ...]
    net_arch: Sequence[int]
    log_std_init: float = 0.0
    activation_fn: Callable[[jax.Array], jax.Array] = nn.tanh
    num_discrete_choices: int | Sequence[int] | None = None
    ortho_init: bool = False
    features_extractor: type[NatureCNN] | None = None
    features_dim: int = 512
    min_concentration: float = 1.0
    max_concentration: float = 20.0
    eps: float = 1e-6

    def get_std(self) -> jax.Array:
        return jnp.array(0.0)

    @nn.compact
    def __call__(self, x: jax.Array) -> AffineBetaDiagDistribution:
        if self.num_discrete_choices is not None:
            raise NotImplementedError("SBXBoundedBetaActor only supports continuous Box action spaces")

        if self.features_extractor is not None:
            x = self.features_extractor(self.features_dim, self.activation_fn)(x)
        else:
            x = x.reshape((x.shape[0], -1))

        for n_units in self.net_arch:
            x = nn.Dense(n_units)(x)
            x = self.activation_fn(x)

        dense_kwargs: dict[str, Any] = {}
        if self.ortho_init:
            dense_kwargs.update(
                kernel_init=nn.initializers.orthogonal(scale=0.01),
                bias_init=nn.initializers.zeros,
            )
        alpha_logits = nn.Dense(self.action_dim, **dense_kwargs)(x)
        beta_logits = nn.Dense(self.action_dim, **dense_kwargs)(x)

        alpha = self.min_concentration + nn.softplus(alpha_logits)
        beta = self.min_concentration + nn.softplus(beta_logits)
        alpha = jnp.clip(alpha, self.min_concentration + self.eps, self.max_concentration)
        beta = jnp.clip(beta, self.min_concentration + self.eps, self.max_concentration)

        action_low = jnp.asarray(self.action_low, dtype=alpha.dtype)
        action_high = jnp.asarray(self.action_high, dtype=alpha.dtype)
        action_scale = jnp.maximum(action_high - action_low, self.eps)
        return AffineBetaDiagDistribution(
            alpha=alpha,
            beta=beta,
            action_low=action_low,
            action_scale=action_scale,
            eps=self.eps,
        )


class SBXPPOBoundedBetaPolicy(PPOPolicy):
    """SBX PPOPolicy variant that matches the JAX bounded Beta actor."""

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule,
        *args: Any,
        min_concentration: float = 1.0,
        max_concentration: float = 20.0,
        beta_eps: float = 1e-6,
        **kwargs: Any,
    ):
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError(f"SBXPPOBoundedBetaPolicy requires Box action space, got {type(action_space)!r}")
        self.min_concentration = float(min_concentration)
        self.max_concentration = float(max_concentration)
        self.beta_eps = float(beta_eps)
        action_low = tuple(np.asarray(action_space.low, dtype=np.float32).reshape(-1).tolist())
        action_high = tuple(np.asarray(action_space.high, dtype=np.float32).reshape(-1).tolist())
        kwargs["actor_class"] = partial(
            SBXBoundedBetaActor,
            action_low=action_low,
            action_high=action_high,
            min_concentration=self.min_concentration,
            max_concentration=self.max_concentration,
            eps=self.beta_eps,
        )
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )
