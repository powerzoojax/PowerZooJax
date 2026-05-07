"""DC Microgrid task — DataCenter + Battery + PV + Diesel benchmark.

Task-level recipe for the data-center microgrid benchmark (single-agent,
multi-objective: energy cost, emissions, and SLA violations).

Main entry-points
-----------------
* ``make_dcmicrogrid_params(...)`` — re-exported from
  ``powerzoojax.envs.microgrid``; canonical factory for synthetic profiles.
* ``make_dcmicrogrid_params_with_profiles(...)`` — re-exported; factory for
  real Google workload + PV profiles.
* ``compute_dcmicrogrid_metrics(info_history)`` — aggregate per-step info
  dicts into episode-level scalar metrics (energy cost, carbon, SLA rates, …).

Physical core (``DataCenterMicrogridEnv``, dynamics, bundles) lives in
``powerzoojax.envs.microgrid``.  This module is the task recipe layer.

Splits
------
    train             Full Google workload profile
    iid               Held-out Google workload window
    cooling_stress    OOD: high ambient temperature profile
    renewable_drought OOD: low PV irradiance profile
"""

from __future__ import annotations

import numpy as np
from powerzoojax.tasks.base import ConstraintSpec

# Re-export canonical env factories so callers can import from tasks.*
from powerzoojax.envs.microgrid import (
    make_dcmicrogrid_params,
    make_dcmicrogrid_params_with_profiles,
    DataCenterMicrogridEnv,
    DCMicrogridParams,
    DCMicrogridState,
)

__all__ = [
    "make_dcmicrogrid_params",
    "make_dcmicrogrid_params_with_profiles",
    "DataCenterMicrogridEnv",
    "DCMicrogridParams",
    "DCMicrogridState",
    "compute_dcmicrogrid_metrics",
]


# ============ Metrics ============

def compute_dcmicrogrid_metrics(info_history: list[dict]) -> dict[str, float]:
    """Aggregate per-step info dicts into episode-level scalar metrics.

    Args:
        info_history: List of per-step ``info`` dicts returned by
                      ``DataCenterMicrogridEnv.step()``.

    Returns:
        Dict of scalar metrics covering energy, cost, carbon, SLA, battery,
        and PV utilisation.
    """
    if not info_history:
        return {}

    def _col(key: str) -> np.ndarray:
        return np.array([float(info.get(key, 0.0)) for info in info_history])

    dt_h = 5.0 / 60.0  # 5-min steps

    p_dc   = _col("p_dc_mw")
    fuel_cost      = _col("fuel_cost")
    grid_cost      = _col("grid_cost")
    grid_price     = _col("grid_price_per_mwh")
    terminal_soc_cost = _col("terminal_soc_cost")
    carbon         = _col("carbon_kg")
    grid_carbon    = _col("grid_carbon_kg")
    cost_sla       = _col("cost_sla")
    cost_overtemp  = _col("cost_overtemp")
    cost_power_deficit = _col("cost_power_deficit")
    cost_power_spill = _col("cost_power_spill")
    cost_power_balance = _col("cost_power_balance")
    cost   = _col("cost_sum")
    p_pv   = _col("p_pv_mw")
    p_grid = _col("p_grid_import_mw")
    soc    = _col("soc")
    p_batt = _col("p_batt_mw")
    reward = (
        _col("reward") if "reward" in info_history[0]
        else np.zeros(len(info_history))
    )

    max_pv = float(np.max(p_pv)) if float(np.max(p_pv)) > 0 else 1.0

    batt_throughput = float(np.sum(np.abs(p_batt)) * dt_h)
    # Diagnostic cycle count for the frozen DC task calibration. Keep this
    # tied to the configured physical SOC window rather than the policy's
    # observed excursion, otherwise a near-idle policy can appear to cycle more.
    usable_energy_mwh = 2.0 * (0.90 - 0.15)
    battery_cycles = batt_throughput / max(2.0 * usable_energy_mwh, 1e-6)

    return {
        "total_energy_cost":     float(np.sum(p_dc) * dt_h),
        "total_fuel_cost":       float(np.sum(fuel_cost)),
        "total_grid_cost":       float(np.sum(grid_cost)),
        "total_terminal_soc_cost": float(np.sum(terminal_soc_cost)),
        "total_carbon_kg":       float(np.sum(carbon + grid_carbon)),
        "total_diesel_carbon_kg": float(np.sum(carbon)),
        "total_grid_carbon_kg":  float(np.sum(grid_carbon)),
        "grid_import_mwh":       float(np.sum(p_grid) * dt_h),
        "mean_grid_price_per_mwh": float(np.mean(grid_price)),
        "sla_violation_rate":    float(np.mean(cost_sla)),
        "overtemp_rate":         float(np.mean(cost_overtemp > 0)),
        "power_deficit_rate":    float(np.mean(cost_power_deficit > 0)),
        "feasibility_rate":      float(np.mean(cost == 0)),
        "pv_utilization":        float(np.mean(p_pv / max_pv)),
        "battery_cycles":        battery_cycles,
        "episode_reward":        float(np.sum(reward)),
        "mean_cost_sla":         float(np.mean(cost_sla)),
        "mean_cost_overtemp":    float(np.mean(cost_overtemp)),
        "mean_cost_power_deficit": float(np.mean(cost_power_deficit)),
        "mean_cost_power_spill": float(np.mean(cost_power_spill)),
        "mean_cost_power_balance": float(np.mean(cost_power_balance)),
        "mean_p_dc_mw":          float(np.mean(p_dc)),
        "mean_soc":              float(np.mean(soc)),
        "soc_min":               float(np.min(soc)),
        "soc_max":               float(np.max(soc)),
        "battery_throughput_mwh": batt_throughput,
        "power_spill_rate":      float(np.mean(cost_power_spill > 0)),
    }


# ============ TaskSpec implementation ============

def rollout_dcmicrogrid(env, params, key, policy_fn) -> list:
    """Single-episode Python-loop rollout for a DC Microgrid policy.

    Returns ``info_history``: list of per-step info dicts, length ``params.dc.max_steps``.
    Pass the result directly to :func:`compute_dcmicrogrid_metrics`.

    Args:
        env: DataCenterMicrogridEnv instance.
        params: DCMicrogridParams.
        key: JAX PRNGKey.
        policy_fn: Callable ``(obs, state, key) -> action``.
    """
    import jax
    obs, state = env.reset(key, params)
    info_history = []
    for _ in range(_episode_length(params)):
        key, k_step, k_pol = jax.random.split(key, 3)
        action = policy_fn(obs, state, k_pol)
        obs, state, reward, _costs, _, info = env.step(k_step, state, action, params)
        info_history.append(_pythonize_step_info(info, reward))
    return info_history


def _episode_length(params) -> int:
    """Return DC Microgrid episode length from the nested params schema."""
    return int(params.dc.max_steps)


def _pythonize_info(info: dict) -> dict:
    """Convert JAX info leaves to JSON-friendly Python values."""
    out = {}
    for k, v in info.items():
        arr = np.asarray(v)
        out[k] = float(arr) if arr.ndim == 0 else arr.tolist()
    return out


def _pythonize_step_info(info: dict, reward) -> dict:
    """Convert step info and attach the scalar reward returned by the env."""
    out = _pythonize_info(info)
    out["reward"] = float(np.asarray(reward))
    return out


def rollout_dcmicrogrid_no_control(env, params, key) -> list:
    """No-control baseline: all actions = 0 (DataCenter runs at default settings)."""
    import jax.numpy as jnp
    return rollout_dcmicrogrid(
        env, params, key,
        policy_fn=lambda obs, state, k: jnp.zeros(5, dtype=jnp.float32),
    )


# Step-index constants for 5-min/step 288-step (24 h) episodes.
_DC_SOLAR_PEAK_START = 72    # 06:00
_DC_SOLAR_PEAK_END = 144     # 12:00
_DC_NIGHT_START_2 = 240      # 20:00
_DC_NIGHT_END_1 = 72         # 06:00


def rollout_dcmicrogrid_max_renewable(env, params, key) -> list:
    """Max-renewable baseline: prioritise PV, charge battery during solar peak."""
    import jax
    import jax.numpy as jnp
    obs, state = env.reset(key, params)
    info_history = []
    for step in range(_episode_length(params)):
        key, k_step = jax.random.split(key)
        if _DC_SOLAR_PEAK_START <= step < _DC_SOLAR_PEAK_END:
            batt = -1.0
        elif step >= _DC_NIGHT_START_2 or step < _DC_NIGHT_END_1:
            batt = 1.0
        else:
            batt = 0.0
        action = jnp.array([0.5, 0.5, 0.5, batt, 0.0], dtype=jnp.float32)
        obs, state, reward, _costs, _, info = env.step(k_step, state, action, params)
        info_history.append(_pythonize_step_info(info, reward))
    return info_history


def rollout_dcmicrogrid_rule_based(env, params, key) -> list:
    """Hand-crafted **rule-based** policy: align workload with solar peak, bring DG on when deficit is high."""
    import jax
    import jax.numpy as jnp
    obs, state = env.reset(key, params)
    info_history = []
    power_deficit = 0.0
    for step in range(_episode_length(params)):
        key, k_step = jax.random.split(key)
        if _DC_SOLAR_PEAK_START <= step < _DC_SOLAR_PEAK_END:
            train_sched, ft_sched = 1.0, 0.5
            batt = 0.0
        elif step < _DC_SOLAR_PEAK_START:
            train_sched, ft_sched = 0.0, 0.0
            batt = -1.0
        else:
            train_sched, ft_sched = 0.0, 0.0
            batt = 0.5 if _DC_NIGHT_START_2 > step else -1.0
        dg = 1.0 if power_deficit > 0.05 else 0.0
        action = jnp.array([train_sched, ft_sched, 0.0, batt, dg], dtype=jnp.float32)
        obs, state, reward, _costs, _, info = env.step(k_step, state, action, params)
        step_info = _pythonize_step_info(info, reward)
        power_deficit = step_info.get("cost_power_deficit", 0.0)
        info_history.append(step_info)
    return info_history


class DCMicrogridTask:
    """TaskSpec for the DC Microgrid benchmark (single-agent, real Google workload).

    Each episode uses a different ``episode_start_step`` window from the Google
    workload profile.  The env is stateless; params are rebuilt per episode.
    """

    task_name = "dc_microgrid"
    default_splits: tuple = (
        "train",
        "iid",
        "cooling_stress",
        "renewable_drought",
        "workload_swap",
        "workload_shock",
        "dg_derating",
        "sla_tighten",
    )

    # Total workload profile length (288 steps/day × ~365 days); episodes sample from [0, T).
    _PROFILE_LEN = 288 * 365

    def __init__(
        self,
        *,
        source: str = "google",
        max_steps: int = 288,
        case_overrides: dict | None = None,
    ):
        self._source = source
        self._max_steps = max_steps
        self._case_overrides = dict(case_overrides or {})

    def make_env(self, split: str = "train"):
        return DataCenterMicrogridEnv()

    def episode_params(
        self,
        split: str,
        episode_idx: int,
        n_episodes: int,
        max_steps: int,
        *,
        strategy: str = "uniform",
        seed: int = 0,
    ):
        T = self._PROFILE_LEN
        if strategy == "uniform":
            start = int(episode_idx / max(n_episodes, 1) * max(T - max_steps, 1))
        else:
            rng = np.random.default_rng(seed * 997 + episode_idx)
            start = int(rng.integers(0, max(1, T - max_steps)))

        scenario_map = {
            "cooling_stress": "cooling_stress",
            "renewable_drought": "renewable_drought",
            "workload_swap": "workload_swap",
            "workload_shock": "workload_shock",
            "dg_derating": "dg_derating",
            "sla_tighten": "sla_tighten",
        }
        ood_scenario = scenario_map.get(split)

        params = make_dcmicrogrid_params_with_profiles(
            source=self._source,
            episode_start_step=start,
            max_steps=max_steps,
            strict=True,
            require_real_data=True,
            **self._case_overrides,
        )
        if ood_scenario is not None:
            from powerzoojax.data.dc_microgrid_profiles import apply_ood_transform
            params = apply_ood_transform(params, ood_scenario)
        return params

    def rollout(self, env, params, key, policy_fn) -> list:
        return rollout_dcmicrogrid(env, params, key, policy_fn)

    def baseline_rollout(self, env, params, key, baseline_name: str) -> list:
        if baseline_name == "no_control":
            return rollout_dcmicrogrid_no_control(env, params, key)
        elif baseline_name == "max_renewable":
            return rollout_dcmicrogrid_max_renewable(env, params, key)
        elif baseline_name == "rule_based":
            return rollout_dcmicrogrid_rule_based(env, params, key)
        else:
            raise ValueError(f"Unknown DC Microgrid baseline: {baseline_name!r}")

    def compute_metrics(self, agent_rollout, baseline_rollout) -> dict:
        return compute_dcmicrogrid_metrics(agent_rollout)

    def baseline_names(self) -> tuple:
        return ("no_control", "max_renewable", "rule_based")

    def constraint_spec(self) -> ConstraintSpec:
        return ConstraintSpec(
            selected_names=("sla", "overtemp", "power_deficit"),
            thresholds=(0.0, 0.0, 0.0),
            fallback_weights=(1.0, 1.0, 1.0),
        )
