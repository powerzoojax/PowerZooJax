"""Custom policy modules used by PowerZooJax trainers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable

import distrax
from flax import linen as nn
from jax import numpy as jnp


class _MLP(nn.Module):
    hidden_layer_sizes: Sequence[int]
    activation: Callable

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        for size in self.hidden_layer_sizes:
            x = nn.Dense(size)(x)
            x = self.activation(x)
        return x


class BoundedBetaPolicy(nn.Module):
    """Beta policy on a finite Box action space.

    Rejax's default PPO continuous actor samples an unconstrained Gaussian and
    clips it to the action bounds before stepping the environment. That creates
    a likelihood mismatch at the bounds because PPO optimizes the unclipped
    action log-prob while the environment sees the clipped action. This policy
    keeps support inside the action box directly.
    """

    action_dim: int
    action_range: tuple[jnp.ndarray, jnp.ndarray]
    hidden_layer_sizes: Sequence[int]
    activation: Callable
    min_concentration: float = 1.0
    max_concentration: float = 20.0
    eps: float = 1e-6

    def setup(self):
        self.features = _MLP(self.hidden_layer_sizes, self.activation)
        self.alpha_head = nn.Dense(self.action_dim)
        self.beta_head = nn.Dense(self.action_dim)

    @property
    def action_low(self):
        return jnp.asarray(self.action_range[0], dtype=jnp.float32)

    @property
    def action_high(self):
        return jnp.asarray(self.action_range[1], dtype=jnp.float32)

    @property
    def action_scale(self):
        return self.action_high - self.action_low

    def _alpha_beta(self, obs):
        x = self.features(obs)
        alpha = self.min_concentration + nn.softplus(self.alpha_head(x))
        beta = self.min_concentration + nn.softplus(self.beta_head(x))
        alpha = jnp.clip(alpha, self.min_concentration + self.eps, self.max_concentration)
        beta = jnp.clip(beta, self.min_concentration + self.eps, self.max_concentration)
        return alpha, beta

    def _unit_dist(self, obs):
        alpha, beta = self._alpha_beta(obs)
        return distrax.Beta(alpha, beta)

    def _to_env_action(self, unit_action):
        return self.action_low + unit_action * self.action_scale

    def _to_unit_action(self, action):
        unit = (action - self.action_low) / jnp.maximum(self.action_scale, self.eps)
        return jnp.clip(unit, self.eps, 1.0 - self.eps)

    def _action_dist(self, obs):
        return distrax.Independent(self._unit_dist(obs), reinterpreted_batch_ndims=1)

    def __call__(self, obs, rng):
        action, log_prob = self.action_log_prob(obs, rng)
        _, entropy = self.log_prob_entropy(obs, action)
        return action, log_prob, entropy

    def action_log_prob(self, obs, rng):
        dist = self._unit_dist(obs)
        unit_action = dist.sample(seed=rng)
        log_prob = jnp.sum(dist.log_prob(unit_action), axis=-1)
        action = self._to_env_action(unit_action)
        return action, log_prob

    def log_prob_entropy(self, obs, action):
        dist = self._unit_dist(obs)
        unit_action = self._to_unit_action(action)
        log_prob = jnp.sum(dist.log_prob(unit_action), axis=-1)
        entropy = jnp.sum(dist.entropy(), axis=-1)
        return log_prob, entropy

    def mode_action(self, obs):
        alpha, beta = self._alpha_beta(obs)
        unit_action = (alpha - 1.0) / jnp.maximum(alpha + beta - 2.0, self.eps)
        unit_action = jnp.clip(unit_action, self.eps, 1.0 - self.eps)
        return self._to_env_action(unit_action)

    def mean_action(self, obs):
        alpha, beta = self._alpha_beta(obs)
        unit_action = alpha / jnp.maximum(alpha + beta, self.eps)
        unit_action = jnp.clip(unit_action, self.eps, 1.0 - self.eps)
        return self._to_env_action(unit_action)

    def act(self, obs, rng):
        action, _ = self.action_log_prob(obs, rng)
        return action
