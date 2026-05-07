"""RewardWrapper — inject custom reward functions into training wrappers.

Wraps a LogWrapper or SafeRLWrapper to replace the environment reward with a
user-supplied function, while maintaining per-episode tracking with the custom
reward signal.

Usage::

    from powerzoojax.rl import LogWrapper, bind
    from powerzoojax.rl.reward import RewardWrapper

    env = LogWrapper(BatteryEnv(), make_battery_params())
    env = RewardWrapper(env, reward_fn=lambda o, a, no, r, i: -jnp.abs(no[0] - 0.5))
    obs, state = env.reset(key)
    obs, state, reward, done, info = env.step(key, state, action)
"""

from functools import partial
import inspect
from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import chex
from flax import struct

from powerzoojax.rl.wrappers import LogWrapper, SafeRLWrapper


@struct.dataclass
class RewardEnvState:
    """State for RewardWrapper — wraps inner state with custom-reward episode tracking."""
    inner_state: Any                              # LogEnvState or SafeRLState
    prev_obs: chex.Array                          # observation before current step
    episode_returns: chex.Array                   # running custom-reward return
    episode_lengths: chex.Array                   # running step count
    returned_episode_returns: chex.Array          # last completed episode return (custom reward)
    returned_episode_lengths: chex.Array          # last completed episode length


class RewardWrapper:
    """Inject a custom reward function into a LogWrapper or SafeRLWrapper.

    The custom ``reward_fn`` *replaces* (not adds to) the original env reward.
    Episode return statistics in ``info`` are computed from the custom reward.

    Args:
        env: A ``LogWrapper`` or ``SafeRLWrapper`` instance.
        reward_fn: Callable with signature::

            reward_fn(obs, action, next_obs, reward, info) -> scalar
            reward_fn(obs, action, next_obs, reward, costs, info) -> scalar

            obs:      observation before the step (from state.prev_obs)
            action:   action taken this step
            next_obs: observation after the step
            reward:   original environment reward (may be used or ignored)
            costs:    selected constraint vector when available
            info:     environment info dict
            returns:  scalar float32 — the new reward to use

    Note:
        Whether the inner env is a ``SafeRLWrapper`` (6-tuple step) is checked
        once at construction time (``isinstance`` on ``env``), so the static
        Python branch is JIT-safe under ``static_argnums=(0,)``.

    Usage::

        # Custom SOC-tracking reward
        env = LogWrapper(BatteryEnv(), params)
        env = RewardWrapper(env, lambda o, a, no, r, i: -jnp.abs(no[0] - 0.5))

        # For CMDP — inner env is SafeRLWrapper, selected costs flow through unchanged
        safe_env = SafeRLWrapper(
            TransGridEnv(),
            params,
            selected_names=("thermal_overload",),
            cost_thresholds=(0.0,),
        )
        env = RewardWrapper(safe_env, my_reward_fn)
        obs, state, reward, costs, done, info = env.step(key, state, action)
    """

    def __init__(self, env, reward_fn: Callable):
        self._env = env
        self._reward_fn = reward_fn
        # Determine safe mode at construction (static Python branch → JIT-safe)
        self._is_safe = isinstance(env, SafeRLWrapper)
        self._reward_fn_uses_costs = _reward_fn_accepts_costs(reward_fn)

    # ---- Property delegation ----

    @property
    def obs_size(self) -> int:
        return self._env.obs_size

    @property
    def num_actions(self) -> int:
        return self._env.num_actions

    @property
    def action_size(self) -> int:
        return self._env.action_size

    def observation_space(self):
        return self._env.observation_space()

    def action_space(self):
        return self._env.action_space()

    @property
    def name(self) -> str:
        return self._env.name

    # ---- Core methods ----

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, RewardEnvState]:
        obs, inner_state = self._env.reset(key)
        state = RewardEnvState(
            inner_state=inner_state,
            prev_obs=obs,
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
        state: RewardEnvState,
        action: chex.Array,
    ) -> Tuple:
        """Returns 5-tuple or 6-tuple depending on the wrapped env."""
        if self._is_safe:
            next_obs, inner_state, orig_reward, costs, done, info = self._env.step(
                key, state.inner_state, action
            )
        else:
            next_obs, inner_state, orig_reward, done, info = self._env.step(
                key, state.inner_state, action
            )
            costs = info.get("constraint_costs", jnp.zeros((0,), dtype=jnp.float32))

        if self._reward_fn_uses_costs:
            new_reward = self._reward_fn(
                state.prev_obs, action, next_obs, orig_reward, costs, info
            )
        else:
            new_reward = self._reward_fn(
                state.prev_obs, action, next_obs, orig_reward, info
            )

        new_ep_returns = state.episode_returns + new_reward
        new_ep_lengths = state.episode_lengths + 1

        new_state = RewardEnvState(
            inner_state=inner_state,
            prev_obs=next_obs,
            episode_returns=new_ep_returns * (1 - done),
            episode_lengths=new_ep_lengths * (1 - done.astype(jnp.int32)),
            returned_episode_returns=jnp.where(
                done, new_ep_returns, state.returned_episode_returns
            ),
            returned_episode_lengths=jnp.where(
                done, new_ep_lengths, state.returned_episode_lengths
            ),
        )

        info = {
            **info,
            "returned_episode_returns": new_state.returned_episode_returns,
            "returned_episode_lengths": new_state.returned_episode_lengths,
            "returned_episode": done,
        }

        if self._is_safe:
            return next_obs, new_state, new_reward, costs, done, info
        return next_obs, new_state, new_reward, done, info


def _reward_fn_accepts_costs(reward_fn: Callable) -> bool:
    """Whether ``reward_fn`` declares a costs argument (or variadic args)."""
    try:
        sig = inspect.signature(reward_fn)
    except (TypeError, ValueError):
        return False

    positional = [
        p for p in sig.parameters.values()
        if p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    if len(positional) >= 6:
        return True
    return any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
