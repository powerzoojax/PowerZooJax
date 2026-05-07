"""Bounded Beta policies for cross-backend benchmark alignment.

This module only covers the Python backend path used by ``powerzoo_bridge``.
It mirrors the bounded Beta actor already used by the JAX PPO trainer for
continuous Box action spaces such as DC Microgrid.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common.distributions import Distribution, sum_independent_dims
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn
from torch.distributions import Beta
from torch.nn import functional as F


class AffineBetaDistribution(Distribution):
    """Factorized Beta policy on an arbitrary finite Box action space.

    Actions are sampled on ``[0, 1]^d`` and then affinely mapped into the env
    bounds.  Log-prob and entropy intentionally follow the JAX benchmark's
    bounded-Beta convention and are computed on the unit-space variables
    without adding the affine Jacobian term.
    """

    distribution: Beta

    def __init__(
        self,
        action_space: gym.spaces.Box,
        *,
        min_concentration: float = 1.0,
        max_concentration: float = 20.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError(f"AffineBetaDistribution requires Box action space, got {type(action_space)!r}")
        self.action_space = action_space
        self.action_dim = int(np.prod(action_space.shape))
        self.min_concentration = float(min_concentration)
        self.max_concentration = float(max_concentration)
        self.eps = float(eps)
        self._action_low_np = np.asarray(action_space.low, dtype=np.float32).reshape(-1)
        self._action_high_np = np.asarray(action_space.high, dtype=np.float32).reshape(-1)

    def _bounds(self, ref: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        low = th.as_tensor(self._action_low_np, dtype=ref.dtype, device=ref.device)
        high = th.as_tensor(self._action_high_np, dtype=ref.dtype, device=ref.device)
        return low, high

    def _to_env_action(self, unit_action: th.Tensor) -> th.Tensor:
        low, high = self._bounds(unit_action)
        scale = th.clamp(high - low, min=self.eps)
        return low + unit_action * scale

    def _to_unit_action(self, action: th.Tensor) -> th.Tensor:
        low, high = self._bounds(action)
        scale = th.clamp(high - low, min=self.eps)
        unit = (action - low) / scale
        return th.clamp(unit, self.eps, 1.0 - self.eps)

    def _concentrations(
        self,
        alpha_logits: th.Tensor,
        beta_logits: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        alpha = self.min_concentration + F.softplus(alpha_logits)
        beta = self.min_concentration + F.softplus(beta_logits)
        alpha = th.clamp(alpha, min=self.min_concentration + self.eps, max=self.max_concentration)
        beta = th.clamp(beta, min=self.min_concentration + self.eps, max=self.max_concentration)
        return alpha, beta

    def proba_distribution_net(self, latent_dim: int) -> tuple[nn.Module, nn.Module]:
        alpha_net = nn.Linear(latent_dim, self.action_dim)
        beta_net = nn.Linear(latent_dim, self.action_dim)
        return alpha_net, beta_net

    def proba_distribution(self, alpha_logits: th.Tensor, beta_logits: th.Tensor) -> "AffineBetaDistribution":
        alpha, beta = self._concentrations(alpha_logits, beta_logits)
        self.distribution = Beta(alpha, beta)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        unit_actions = self._to_unit_action(actions)
        log_prob = self.distribution.log_prob(unit_actions)
        return sum_independent_dims(log_prob)

    def entropy(self) -> th.Tensor | None:
        return sum_independent_dims(self.distribution.entropy())

    def sample(self) -> th.Tensor:
        sample_fn = getattr(self.distribution, "rsample", None)
        unit_action = sample_fn() if callable(sample_fn) else self.distribution.sample()
        return self._to_env_action(unit_action)

    def mode(self) -> th.Tensor:
        alpha = self.distribution.concentration1
        beta = self.distribution.concentration0
        denom = th.clamp(alpha + beta - 2.0, min=self.eps)
        unit_mode = th.clamp((alpha - 1.0) / denom, self.eps, 1.0 - self.eps)
        return self._to_env_action(unit_mode)

    def actions_from_params(
        self,
        alpha_logits: th.Tensor,
        beta_logits: th.Tensor,
        deterministic: bool = False,
    ) -> th.Tensor:
        self.proba_distribution(alpha_logits, beta_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(
        self,
        alpha_logits: th.Tensor,
        beta_logits: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(alpha_logits, beta_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class SB3BoundedBetaPolicy(ActorCriticPolicy):
    """SB3 ActorCriticPolicy variant that matches the JAX bounded Beta actor."""

    def __init__(
        self,
        *args: Any,
        min_concentration: float = 1.0,
        max_concentration: float = 20.0,
        beta_eps: float = 1e-6,
        **kwargs: Any,
    ):
        self.min_concentration = float(min_concentration)
        self.max_concentration = float(max_concentration)
        self.beta_eps = float(beta_eps)
        super().__init__(*args, **kwargs)
        if not isinstance(self.action_space, gym.spaces.Box):
            raise TypeError(f"SB3BoundedBetaPolicy requires Box action space, got {type(self.action_space)!r}")

        self.action_dist = AffineBetaDistribution(
            self.action_space,
            min_concentration=self.min_concentration,
            max_concentration=self.max_concentration,
            eps=self.beta_eps,
        )
        if hasattr(self, "action_net"):
            del self.action_net
        if hasattr(self, "log_std"):
            del self.log_std
        self.alpha_net, self.beta_net = self.action_dist.proba_distribution_net(
            latent_dim=self.mlp_extractor.latent_dim_pi
        )
        if self.ortho_init:
            for module in (self.alpha_net, self.beta_net):
                module.apply(lambda m: self.init_weights(m, gain=0.01))
        initial_lr = float(self.optimizer.param_groups[0]["lr"])
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=initial_lr,
            **self.optimizer_kwargs,
        )

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                min_concentration=self.min_concentration,
                max_concentration=self.max_concentration,
                beta_eps=self.beta_eps,
            )
        )
        return data

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        alpha_logits = self.alpha_net(latent_pi)
        beta_logits = self.beta_net(latent_pi)
        return self.action_dist.proba_distribution(alpha_logits, beta_logits)
