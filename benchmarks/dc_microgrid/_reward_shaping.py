"""Reward shaping wrapper for the DC Microgrid benchmark.

Wraps ``DataCenterMicrogridEnv`` so the scalar reward returned by ``step``
includes fixed-weight penalties on the benchmark cost diagnostics:

    reward_shaped = reward_raw
                  - lambda_sla            * info['cost_sla']
                  - lambda_overtemp       * info['cost_overtemp']
                  - lambda_power_deficit  * info['cost_power_deficit']
                  - lambda_power_spill    * info['cost_power_spill']
                  - lambda_power_balance  * info['cost_power_balance']
                  - lambda_dispatch_track * info['cost_dispatch_tracking']

The original reward is preserved in ``info['raw_reward']`` and the total
per-step penalty is exposed as ``info['shaping_penalty']``.

When ``dg_autobalance=True``, the wrapper previews the same step with diesel
output set to zero, reads the realised residual, and replays the step with DG
applied as a same-step residual slack actuator.

The wrapper is duck-typed (does not subclass ``Environment``) and forwards
all metadata calls. ``LogWrapper`` can be stacked on top of it unchanged.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Dict, Mapping, Tuple

import chex
import jax
import jax.numpy as jnp
from powerzoojax.envs.resource.battery import BatteryBundle
from powerzoojax.envs.resource.diesel import DieselBundle


_REQUIRED_KEYS = (
    "sla",
    "overtemp",
    "power_deficit",
    "power_spill",
    "power_balance",
    "dispatch_tracking",
)


def _find_bundle_index(resources, cls) -> int:
    for i, bundle in enumerate(resources):
        if isinstance(bundle, cls):
            return i
    return -1


def _dispatch_tracking_cost(state, info, params) -> tuple[chex.Array, chex.Array, chex.Array]:
    """Penalty for deviating from a one-step physically/economically plausible target."""
    p_load = info.get("p_load_mw", jnp.float32(0.0))
    p_pv = info.get("p_pv_mw", jnp.float32(0.0))
    p_batt = info.get("p_batt_mw", jnp.float32(0.0))
    p_dg = info.get("p_dg_mw", jnp.float32(0.0))
    net_load = p_load - p_pv

    batt_idx = _find_bundle_index(params.resources, BatteryBundle)
    if batt_idx < 0:
        batt_target = jnp.float32(0.0)
        batt_cost = jnp.float32(0.0)
    else:
        batt = params.resources[batt_idx]
        batt_state = state.resource_states[batt_idx]
        safe_p = jnp.maximum(batt.power_max[0], jnp.float32(1e-6))
        max_dis = jnp.clip(
            (batt_state.soc[0] - batt.soc_min[0])
            * batt.capacity[0]
            * batt.eta_discharge[0]
            / jnp.float32(batt.dt_hours),
            0.0,
            safe_p,
        )
        max_chg = jnp.clip(
            (batt.soc_max[0] - batt_state.soc[0])
            * batt.capacity[0]
            / (batt.eta_charge[0] * jnp.float32(batt.dt_hours)),
            0.0,
            safe_p,
        )
        if getattr(params, "price_profile", None) is not None and getattr(params, "grid_import_p_max_mw", 0.0) > 0.0:
            T_price = params.price_profile.shape[0]
            price = params.price_profile[state.dc.time_step % T_price]
            price_ref = jnp.float32(max(getattr(params, "grid_price_ref_per_mwh", 150.0), 1e-6))
            price_ratio = price / price_ref
            low_price_signal = jnp.clip((jnp.float32(0.55) - price_ratio) / jnp.float32(0.55), 0.0, 1.0)
            high_price_signal = jnp.clip((price_ratio - jnp.float32(0.85)) / jnp.float32(0.85), 0.0, 1.0)
            charge_target = -max_chg * low_price_signal
            discharge_target = jnp.clip(net_load, 0.0, max_dis) * jnp.maximum(
                high_price_signal,
                jnp.where(net_load > jnp.float32(0.0), jnp.float32(0.25), jnp.float32(0.0)),
            )
            batt_target = jnp.where(
                low_price_signal > high_price_signal,
                charge_target,
                discharge_target,
            )
        else:
            batt_target = jnp.clip(net_load, -max_chg, max_dis)
        batt_cost = jnp.abs(p_batt - batt_target) / safe_p

    dg_idx = _find_bundle_index(params.resources, DieselBundle)
    if dg_idx < 0:
        dg_target = jnp.float32(0.0)
        dg_cost = jnp.float32(0.0)
    else:
        dg = params.resources[dg_idx]
        safe_p = jnp.maximum(dg.p_max[0], jnp.float32(1e-6))
        remaining = net_load - batt_target
        if getattr(params, "price_profile", None) is not None and getattr(params, "grid_import_p_max_mw", 0.0) > 0.0:
            T_price = params.price_profile.shape[0]
            price = params.price_profile[state.dc.time_step % T_price]
            dg_cost = dg.fuel_cost_per_mwh[0]
            dg_target = jnp.where(
                price > dg_cost,
                jnp.clip(remaining, 0.0, safe_p),
                jnp.float32(0.0),
            )
        else:
            dg_target = jnp.clip(remaining, 0.0, safe_p)
        dg_cost = jnp.abs(p_dg - dg_target) / safe_p

    return batt_cost + dg_cost, batt_target, dg_target


def _single_dg_capacity(params) -> chex.Array | None:
    """Return the benchmark DG nameplate when the wrapper can auto-balance it."""
    dg_idx = _find_bundle_index(params.resources, DieselBundle)
    if dg_idx < 0:
        return None
    dg = params.resources[dg_idx]
    if getattr(dg, "n_devices", 1) != 1 or getattr(dg, "action_dim", 1) != 1:
        return None
    return jnp.maximum(dg.p_max[0], jnp.float32(1e-6))


def normalize_lambdas(lambdas: Mapping[str, float] | None) -> Dict[str, float]:
    """Coerce a user lambda dict to the canonical shaping-key float dict."""
    if lambdas is None:
        return {k: 0.0 for k in _REQUIRED_KEYS}
    return {k: float(lambdas.get(k, 0.0)) for k in _REQUIRED_KEYS}


class RewardShapingWrapper:
    """Subtract weighted constraint costs from the env reward.

    Args:
        env: A PowerZooJax ``Environment`` (e.g. ``DataCenterMicrogridEnv``).
        lambdas: Mapping with float weights for keys ``sla`` / ``overtemp`` /
            ``power_deficit`` / ``power_spill`` / ``power_balance`` /
            ``dispatch_tracking``. Missing keys default to 0. ``None`` means
            no shaping while still preserving the wrapper surface.
    """

    def __init__(
        self,
        env,
        lambdas: Mapping[str, float] | None = None,
        *,
        dg_autobalance: bool = False,
    ):
        self._env = env
        normed = normalize_lambdas(lambdas)
        self._lambda_sla = normed["sla"]
        self._lambda_overtemp = normed["overtemp"]
        self._lambda_power_deficit = normed["power_deficit"]
        self._lambda_power_spill = normed["power_spill"]
        self._lambda_power_balance = normed["power_balance"]
        self._lambda_dispatch_tracking = normed["dispatch_tracking"]
        self._dg_autobalance = bool(dg_autobalance)

    @property
    def name(self) -> str:
        return self._env.name

    def default_params(self):
        return self._env.default_params()

    def observation_space(self, params):
        return self._env.observation_space(params)

    def action_space(self, params):
        return self._env.action_space(params)

    def constraint_names(self, params) -> tuple[str, ...]:
        return tuple(self._env.constraint_names(params))

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey, params) -> Tuple[chex.Array, Any]:
        return self._env.reset(key, params)

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: Any,
        action: chex.Array,
        params,
    ) -> Tuple[chex.Array, Any, chex.Array, chex.Array, chex.Array, Dict[str, Any]]:
        action = jnp.asarray(action, dtype=jnp.float32).reshape(-1)
        env_action = action
        dg_command_raw_norm = (
            action[4] if action.shape[0] > 4 else jnp.float32(0.0)
        )
        dg_command_balanced_norm = dg_command_raw_norm
        dg_preview_residual_mw = jnp.float32(0.0)

        dg_p_max = _single_dg_capacity(params)
        if self._dg_autobalance and dg_p_max is not None and action.shape[0] > 4:
            preview_action = action.at[4].set(jnp.float32(0.0))
            _, _, _, _, _, preview_info = self._env.step(
                key, state, preview_action, params
            )
            dg_preview_residual_mw = preview_info.get("residual", jnp.float32(0.0))
            dg_command_balanced_norm = jnp.clip(
                -dg_preview_residual_mw / dg_p_max,
                jnp.float32(0.0),
                jnp.float32(1.0),
            )
            env_action = action.at[4].set(dg_command_balanced_norm)

        obs, new_state, reward, costs, done, info = self._env.step(
            key, state, env_action, params
        )
        cost_sla = info.get("cost_sla", jnp.float32(0.0))
        cost_overtemp = info.get("cost_overtemp", jnp.float32(0.0))
        cost_deficit = info.get("cost_power_deficit", jnp.float32(0.0))
        power_spill = info.get("power_spill", jnp.float32(0.0))
        p_load = info.get("p_load_mw", jnp.float32(0.0))
        cost_spill = power_spill / jnp.maximum(p_load, jnp.float32(1e-6))
        cost_balance = (
            jnp.abs(info.get("residual", jnp.float32(0.0)))
            / jnp.maximum(p_load, jnp.float32(1e-6))
        )
        dispatch_tracking, batt_target, dg_target = _dispatch_tracking_cost(
            state, info, params
        )
        penalty = (
            jnp.float32(self._lambda_sla) * cost_sla
            + jnp.float32(self._lambda_overtemp) * cost_overtemp
            + jnp.float32(self._lambda_power_deficit) * cost_deficit
            + jnp.float32(self._lambda_power_spill) * cost_spill
            + jnp.float32(self._lambda_power_balance) * cost_balance
            + jnp.float32(self._lambda_dispatch_tracking) * dispatch_tracking
        )
        shaped_reward = reward - penalty
        info = dict(info)
        info["raw_reward"] = reward
        info["shaping_penalty"] = penalty
        info["cost_power_spill"] = cost_spill
        info["cost_power_balance"] = cost_balance
        info["cost_dispatch_tracking"] = dispatch_tracking
        info["dispatch_target_batt_mw"] = batt_target
        info["dispatch_target_dg_mw"] = dg_target
        if self._dg_autobalance:
            info["dg_command_raw_norm"] = dg_command_raw_norm
            info["dg_command_balanced_norm"] = dg_command_balanced_norm
            info["dg_preview_residual_mw"] = dg_preview_residual_mw
        return obs, new_state, shaped_reward, costs, done, info


def wrap_with_shaping(env, task_config: Mapping[str, Any] | None):
    """Convenience helper: wrap ``env`` if ``task_config['reward_shaping_weights']`` is set.

    Returns the original env unchanged when weights are absent or all zero.
    """
    task_config = task_config or {}
    weights = task_config.get("reward_shaping_weights")
    normed = normalize_lambdas(weights)
    dg_autobalance = bool(task_config.get("dg_autobalance", False))
    if all(v == 0.0 for v in normed.values()) and not dg_autobalance:
        return env
    return RewardShapingWrapper(env, normed, dg_autobalance=dg_autobalance)
