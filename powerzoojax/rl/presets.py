"""Presets (recipes) — one-liner environment + config bundles for common tasks.

Each preset combines an env factory, an optional custom reward function, and a
default training config.  Users can override any config field via ``train()``.

Usage::

    from powerzoojax.rl.presets import list_presets, get_preset

    for p in list_presets():
        print(p["name"], "-", p["description"])

    preset = get_preset("battery-soc-tracking")
    env = preset.env_factory()
    config = preset.config
"""

from typing import Callable, List, NamedTuple, Optional

from powerzoojax.rl.config import TrainConfig


# ============ PresetDef ============

class PresetDef(NamedTuple):
    """Definition of a training preset."""
    description: str
    env_factory: Callable       # () -> wrapped env (LogWrapper or SafeRLWrapper)
    reward_fn: Optional[Callable]  # None → use env's default reward
    config: TrainConfig


# ============ Env factories ============

def _wrap_battery():
    """BatteryEnv wrapped with LogWrapper (max_steps=48)."""
    from powerzoojax.envs.resource.battery import BatteryEnv, make_battery_params
    from powerzoojax.rl.wrappers import LogWrapper
    return LogWrapper(BatteryEnv(), make_battery_params(max_steps=48))


def _wrap_transgrid_case5():
    """TransGridEnv (case5) wrapped with LogWrapper (constant 50% load, max_steps=48)."""
    import jax.numpy as jnp
    from powerzoojax.case import create_case5
    from powerzoojax.envs.grid.trans import TransGridEnv, make_trans_params
    from powerzoojax.rl.wrappers import LogWrapper

    case = create_case5()
    profiles = jnp.ones((48, case.n_loads), dtype=jnp.float32) * 0.5
    params = make_trans_params(case, load_profiles=profiles, max_steps=48)
    return LogWrapper(TransGridEnv(), params)


def _wrap_transgrid_case5_safe():
    """TransGridEnv (case5) wrapped with SafeRLWrapper (thermal-only CMDP)."""
    import jax.numpy as jnp
    from powerzoojax.case import create_case5
    from powerzoojax.envs.grid.trans import TransGridEnv, make_trans_params
    from powerzoojax.rl.wrappers import SafeRLWrapper

    case = create_case5()
    profiles = jnp.ones((48, case.n_loads), dtype=jnp.float32) * 0.5
    params = make_trans_params(case, load_profiles=profiles, max_steps=48)
    return SafeRLWrapper(
        TransGridEnv(),
        params,
        selected_names=("thermal_overload",),
        cost_thresholds=(0.0,),
    )


def _wrap_transgrid_case5_marl():
    """TransGridEnv (case5) wrapped with GridMARLEnv for multi-agent training."""
    import jax.numpy as jnp
    from powerzoojax.case import create_case5
    from powerzoojax.envs.grid.trans import TransGridEnv, make_trans_params
    from powerzoojax.rl.multi_agent import GridMARLEnv

    case = create_case5()
    profiles = jnp.ones((48, case.n_loads), dtype=jnp.float32) * 0.5
    params = make_trans_params(case, load_profiles=profiles, max_steps=48)
    return GridMARLEnv(TransGridEnv(), params)


def _wrap_dso_nflex():
    """DSO case33bw + 6× FlexLoad wrapped with LogWrapper (48 steps)."""
    from powerzoojax.envs.grid.dist import DistGridEnv
    from powerzoojax.tasks.dso import make_dso_params
    from powerzoojax.rl.wrappers import LogWrapper

    params = make_dso_params()
    return LogWrapper(DistGridEnv(), params)


def _wrap_dso_nflex_safe():
    """DSO case33bw + 6× FlexLoad wrapped with SafeRLWrapper."""
    from powerzoojax.envs.grid.dist import DistGridEnv
    from powerzoojax.tasks.dso import DSOTask, make_dso_params
    from powerzoojax.rl.wrappers import SafeRLWrapper

    params = make_dso_params()
    spec = DSOTask().constraint_spec()
    return SafeRLWrapper(
        DistGridEnv(),
        params,
        selected_names=spec.selected_names,
        cost_thresholds=spec.thresholds,
    )


def _wrap_ders_medium():
    """DERs case141 + 12 agents (4 Battery + 4 PV + 4 FlexLoad), IPPO."""
    from powerzoojax.tasks.ders import make_ders_marl_env
    env, _params = make_ders_marl_env(voltage_penalty=8.0)
    return env


def _wrap_ders_medium_safe():
    """DERs case141 + 12 agents, IPPO + voltage-penalty reward shaping.

    NOTE: This is reward-shaped IPPO only — NOT constrained MARL /
    primal-dual CMDP.  ``cost_threshold`` in the config is a logging target;
    the IPPO training loop does NOT enforce a constraint multiplier.
    Replace with a true CMDP implementation when available.
    """
    from powerzoojax.tasks.ders import make_ders_marl_env
    env, _params = make_ders_marl_env(voltage_penalty=8.0)
    return env


def _wrap_transgrid_case5_marl_battery():
    """TransGridEnv (case5) + 2 BatteryBundle devices as MARL agents."""
    import jax.numpy as jnp
    from powerzoojax.case import create_case5
    from powerzoojax.envs.grid.trans import TransGridEnv, make_trans_params
    from powerzoojax.envs.resource.battery import make_battery_bundle
    from powerzoojax.rl.multi_agent import GridMARLEnv

    case = create_case5()
    profiles = jnp.ones((48, case.n_loads), dtype=jnp.float32) * 0.5
    bundle = make_battery_bundle(
        case, bus_ids=[1, 3],
        power_mw=20.0, capacity_mwh=50.0,
        soc_min=0.1, soc_max=0.9,
    )
    params = make_trans_params(case, load_profiles=profiles, max_steps=48,
                               resources=(bundle,))
    return GridMARLEnv(TransGridEnv(), params)


def _wrap_tso_ed():
    """UnitCommitmentEnv case118, ED mode (enable_uc=False), 48 steps, LogWrapper."""
    from powerzoojax.envs.grid.unit_commitment import UnitCommitmentEnv
    from powerzoojax.tasks.tso import make_tso_ed_params
    from powerzoojax.rl.wrappers import LogWrapper

    params = make_tso_ed_params()
    return LogWrapper(UnitCommitmentEnv(), params)


def _wrap_tso_uc():
    """UnitCommitmentEnv case118, UC mode (enable_uc=True), 48 steps, LogWrapper."""
    from powerzoojax.envs.grid.unit_commitment import UnitCommitmentEnv
    from powerzoojax.tasks.tso import make_tso_uc_params
    from powerzoojax.rl.wrappers import LogWrapper

    params = make_tso_uc_params()
    return LogWrapper(UnitCommitmentEnv(), params)


def _wrap_tso_scuc_safe():
    """UnitCommitmentEnv case118, SCUC with reserve, SafeRLWrapper (PPO-Lagrangian).

    enable_uc=True, enable_reserve=True.
    Thermal overload and reserve shortfall are selected into the CMDP channel.
    PPO-Lagrangian enforces vector cost thresholds via dual multipliers.
    Action space remains Box(2*n_units) — continuous commitment relaxation.
    """
    from powerzoojax.envs.grid.unit_commitment import UnitCommitmentEnv
    from powerzoojax.tasks.tso import TSOTask, make_tso_scuc_params
    from powerzoojax.rl.wrappers import SafeRLWrapper

    params = make_tso_scuc_params()
    spec = TSOTask().constraint_spec()
    return SafeRLWrapper(
        UnitCommitmentEnv(),
        params,
        selected_names=spec.selected_names,
        cost_thresholds=spec.thresholds,
    )


def _load_gb_profiles_for_case5(case, split: str = "train", ood_axis=None):
    """Shim — delegates to :func:`powerzoojax.tasks.gencos.load_gencos_profiles`."""
    from powerzoojax.tasks.gencos import load_gencos_profiles
    return load_gencos_profiles(case, split=split, ood_axis=ood_axis)


def _wrap_gencos_case5_gb(split: str = "train", ood_axis=None):
    """GenCos case5 with real GB 2025Q2–2026Q1 profiles (benchmark split).

    Uses LOAD_ACTUAL_MW from GB demand data, normalised to case5 load buses.
    The full GB window is stored in ``load_profiles`` as the temporal sampling
    pool; individual episodes are always 48 steps (48×30min = 24 h).  Step
    indexing wraps around the pool via modulo, so agents encounter the full
    demand distribution across training episodes.

    Raises FileNotFoundError if GB data is not installed — in that case use
    'gencos-case5-ippo-dev' which uses synthetic flat mid-load profiles.
    """
    from powerzoojax.case import create_case5
    from powerzoojax.envs.market.market_marl_core import make_market_marl_params
    from powerzoojax.rl.market_marl import MarketMARLEnv

    case = create_case5()
    profiles = _load_gb_profiles_for_case5(case, split=split, ood_axis=ood_axis)
    # Episode length is 48 steps (24 h at 30 min per step).
    # The full GB window (several thousand rows) lives in load_profiles as the
    # sampling pool.  Each reset samples a random episode_start_idx ∈ [0, T-1];
    # step t accesses row (episode_start_idx + t) % T, giving diverse temporal
    # windows across training episodes without repeating the same 48-row prefix.
    params = make_market_marl_params(
        case, profiles, n_segments=3, max_markup=2.0,
        max_steps=48,
    )
    return MarketMARLEnv(params)


def _wrap_gencos_case5_dev():
    """GenCos case5 with synthetic flat mid-load profiles (dev/smoke only).

    NOT the benchmark configuration.  Use for quick local development and CI
    when GB data is unavailable.  The benchmark preset is 'gencos-case5-ippo'.
    """
    import jax.numpy as jnp
    from powerzoojax.case import create_case5
    from powerzoojax.envs.market.market_marl_core import make_market_marl_params
    from powerzoojax.rl.market_marl import MarketMARLEnv

    case = create_case5()
    mid_load = (case.load_d_max + case.load_d_min) / 2.0
    profiles = jnp.tile(mid_load[None, :], (48, 1))
    params = make_market_marl_params(case, profiles, n_segments=3, max_markup=2.0)
    return MarketMARLEnv(params)


def _wrap_dc_microgrid():
    """DC Microgrid 1-agent 288-step env, LogWrapper (PPO)."""
    from powerzoojax.envs.microgrid import (
        DataCenterMicrogridEnv,
        make_dcmicrogrid_params,
    )
    from powerzoojax.rl.wrappers import LogWrapper

    params = make_dcmicrogrid_params(max_steps=288)
    return LogWrapper(DataCenterMicrogridEnv(), params)


def _wrap_dc_microgrid_safe():
    """DC Microgrid 1-agent env wrapped for vector-cost PPO-Lagrangian."""
    from powerzoojax.envs.microgrid import (
        DataCenterMicrogridEnv,
        make_dcmicrogrid_params,
    )
    from powerzoojax.tasks.dc_microgrid import DCMicrogridTask
    from powerzoojax.rl.wrappers import SafeRLWrapper

    params = make_dcmicrogrid_params(max_steps=288)
    spec = DCMicrogridTask(max_steps=288).constraint_spec()
    return SafeRLWrapper(
        DataCenterMicrogridEnv(),
        params,
        selected_names=spec.selected_names,
        cost_thresholds=spec.thresholds,
    )




PRESETS: dict = {
    "battery-soc-tracking": PresetDef(
        description="Battery SOC tracking with PPO (Rejax) — custom reward -|soc - 0.5|",
        env_factory=_wrap_battery,
        reward_fn=None,  # set at train() time via lambda
        config=TrainConfig(
            algo="ppo",
            total_timesteps=100_000,
            num_envs=64,
            n_steps=48,
        ),
    ),
    "case5-economic-dispatch": PresetDef(
        description="TransGrid case5 economic dispatch with PPO (Rejax)",
        env_factory=_wrap_transgrid_case5,
        reward_fn=None,
        config=TrainConfig(
            algo="ppo",
            total_timesteps=200_000,
            num_envs=32,
            n_steps=48,
        ),
    ),
    "case5-safe-dispatch": PresetDef(
        description="TransGrid case5 CMDP safe dispatch (PPO-Lagrangian)",
        env_factory=_wrap_transgrid_case5_safe,
        reward_fn=None,
        config=TrainConfig(
            algo="ppo_lagrangian",
            total_timesteps=200_000,
            num_envs=32,
            n_steps=48,
            cost_thresholds=(0.0,),
        ),
    ),
    "case5-ippo": PresetDef(
        description="TransGrid case5 multi-agent IPPO (5 unit agents, parameter sharing)",
        env_factory=_wrap_transgrid_case5_marl,
        reward_fn=None,
        config=TrainConfig(
            algo="ippo",
            total_timesteps=200_000,
            num_envs=16,
            n_steps=48,
        ),
    ),
    "case5-ippo-battery": PresetDef(
        description="TransGrid case5 IPPO + 2 battery devices (7 agents, SOC tracking)",
        env_factory=_wrap_transgrid_case5_marl_battery,
        reward_fn=None,
        config=TrainConfig(
            algo="ippo",
            total_timesteps=200_000,
            num_envs=16,
            n_steps=48,
        ),
    ),
    "dso-nflex": PresetDef(
        description="DSO case33bw + 6× FlexLoad, single agent PPO (network loss minimisation) [synthetic load, dev/test only]",
        env_factory=_wrap_dso_nflex,
        reward_fn=None,
        config=TrainConfig(
            algo="ppo",
            total_timesteps=3_000_000,
            num_envs=128,
            n_steps=48,
            hidden_dims=(128, 128),
            gamma=0.995,
            normalize_observations=True,
        ),
    ),
    "dso-nflex-safe": PresetDef(
        description="DSO case33bw + 6× FlexLoad, PPO-Lagrangian — task-selected CMDP cost = voltage violations only [synthetic load, dev/test only]",
        env_factory=_wrap_dso_nflex_safe,
        reward_fn=None,
        config=TrainConfig(
            algo="ppo_lagrangian",
            total_timesteps=3_000_000,
            num_envs=128,
            n_steps=48,
            hidden_dims=(128, 128),
            gamma=0.995,
            cost_thresholds=(0.0,),
            normalize_observations=True,
        ),
    ),
    "ders-medium": PresetDef(
        description=(
            "DERs case141 + 12 agents (4 Battery + 4 PV + 4 FlexLoad), "
            "typed-IPPO voltage regulation (Battery / Renewable / FlexLoad each "
            "have a separate SharedActorCritic — type-specific parameter sharing)"
        ),
        env_factory=_wrap_ders_medium,
        reward_fn=None,
        config=TrainConfig(
            algo="ippo_typed",
            total_timesteps=15_000_000,
            num_envs=64,
            n_steps=48,
            hidden_dims=(128, 128),
            gamma=0.995,
            normalize_observations=True,
        ),
    ),
    "ders-medium-safe": PresetDef(
        description=(
            "DERs case141 + 12 agents, typed-IPPO + voltage-penalty reward shaping "
            "(NOTE: reward shaping only — NOT constrained MARL / primal-dual; "
            "strict zero cost_thresholds are logging targets only, not enforced)"
        ),
        env_factory=_wrap_ders_medium_safe,
        reward_fn=None,
        config=TrainConfig(
            algo="ippo_typed",
            total_timesteps=15_000_000,
            num_envs=64,
            n_steps=48,
            hidden_dims=(128, 128),
            gamma=0.995,
            cost_thresholds=(0.0, 0.0, 0.0),
            normalize_observations=True,
        ),
    ),
    "tso-ed": PresetDef(
        description=(
            "TSO case118 economic dispatch (UC disabled), single-agent PPO, 48×30min steps. "
            "Action = Box(2*54): [commit_logit(ignored) | dispatch_preference] per unit. "
            "dispatch_preference biases the DC-OPF cost toward the agent's preferred "
            "dispatch level; physical constraints are always enforced by the solver."
        ),
        env_factory=_wrap_tso_ed,
        reward_fn=None,
        config=TrainConfig(
            algo="ppo",
            total_timesteps=5_000_000,
            num_envs=64,
            n_steps=48,
            hidden_dims=(256, 256),
            gamma=0.995,
            normalize_observations=True,
        ),
    ),
    "tso-uc": PresetDef(
        description=(
            "TSO case118 unit commitment (SCUC, continuous relaxation), single-agent PPO, "
            "48×30min steps. Action = Box(2*54): [commit_logit | dispatch_preference] per unit. "
            "commit_logit > 0 → commit unit (subject to min-up/down masking). "
            "dispatch_preference biases the DC-OPF cost toward preferred dispatch. "
            "Min-up/down, startup/no-load, ramp, and reserve constraints enforced in env."
        ),
        env_factory=_wrap_tso_uc,
        reward_fn=None,
        config=TrainConfig(
            algo="ppo",
            total_timesteps=10_000_000,
            num_envs=64,
            n_steps=48,
            hidden_dims=(256, 256),
            gamma=0.995,
            normalize_observations=True,
        ),
    ),
    "tso-scuc-safe": PresetDef(
        description=(
            "TSO case118 SCUC with safety constraints via PPO-Lagrangian (CMDP). "
            "Thermal overload + reserve shortfall enter the selected cost vector; "
            "vector thresholds are enforced by dual multipliers. "
            "Continuous relaxation — no hybrid-action policy head required."
        ),
        env_factory=_wrap_tso_scuc_safe,
        reward_fn=None,
        config=TrainConfig(
            algo="ppo_lagrangian",
            total_timesteps=10_000_000,
            num_envs=64,
            n_steps=48,
            hidden_dims=(256, 256),
            gamma=0.995,
            cost_thresholds=(0.0, 0.0),
            normalize_observations=True,
        ),
    ),
    "gencos-case5-ippo": PresetDef(
        description=(
            "GenCos case5, 5-agent rolling market (exact SCED + ramp coupling), "
            "IPPO — benchmark config using GB 2025Q2–2026Q1 demand profiles. "
            "Raises FileNotFoundError if GB data is not installed; "
            "use 'gencos-case5-ippo-dev' for synthetic profiles."
        ),
        env_factory=_wrap_gencos_case5_gb,
        reward_fn=None,
        config=TrainConfig(
            algo="ippo",
            total_timesteps=5_000_000,
            num_envs=256,
            n_steps=48,
            hidden_dims=(128, 128),
            gamma=0.995,
            eval_freq=100_000,  # steps_per_update=12288 > default 10k → eval every update
        ),
    ),
    "gencos-case5-ippo-dev": PresetDef(
        description=(
            "GenCos case5, 5-agent rolling market (exact SCED + ramp coupling), "
            "IPPO — DEV/SMOKE preset using synthetic flat mid-load profiles. "
            "NOT the benchmark config; use for CI and local dev without GB data."
        ),
        env_factory=_wrap_gencos_case5_dev,
        reward_fn=None,
        config=TrainConfig(
            algo="ippo",
            total_timesteps=5_000_000,
            num_envs=256,
            n_steps=48,
            hidden_dims=(128, 128),
            gamma=0.995,
            eval_freq=100_000,  # steps_per_update=12288 > default 10k → eval every update
        ),
    ),
    "dc-microgrid": PresetDef(
        description=(
            "Self-contained DC Microgrid (1 agent, 288×5min). "
            "Scalarized multi-objective reward (energy + cost + carbon). "
            "Synthetic diurnal workload/solar/temp profiles. "
            "PPO via LogWrapper."
        ),
        env_factory=_wrap_dc_microgrid,
        reward_fn=None,
        config=TrainConfig(
            algo="ppo",
            total_timesteps=5_000_000,
            num_envs=128,
            n_steps=288,
            hidden_dims=(256, 256),
            gamma=0.999,
            normalize_observations=True,
        ),
    ),
    "dc-microgrid-safe": PresetDef(
        description=(
            "Self-contained DC Microgrid (1 agent, 288×5min). "
            "PPO-Lagrangian (CMDP): SLA + over-temperature + power-deficit "
            "vector cost channel with frozen task thresholds. "
            "Synthetic diurnal profiles."
        ),
        env_factory=_wrap_dc_microgrid_safe,
        reward_fn=None,
        config=TrainConfig(
            algo="ppo_lagrangian",
            total_timesteps=5_000_000,
            num_envs=128,
            n_steps=288,
            hidden_dims=(256, 256),
            gamma=0.999,
            cost_thresholds=(0.0, 0.0, 0.0),
            normalize_observations=True,
        ),
    ),
}


def list_presets() -> List[dict]:
    """Return all presets as a list of dicts (AI-friendly JSON output).

    Each dict has keys: ``name``, ``description``, ``algo``, ``total_timesteps``.
    """
    return [
        {
            "name": name,
            "description": p.description,
            "algo": p.config.algo,
            "total_timesteps": p.config.total_timesteps,
        }
        for name, p in PRESETS.items()
    ]


def get_preset(name: str) -> PresetDef:
    """Get a preset by name.

    Args:
        name: Preset name (e.g. ``"battery-soc-tracking"``).

    Returns:
        ``PresetDef`` namedtuple.

    Raises:
        KeyError: If the name is not found.
    """
    if name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise KeyError(
            f"Unknown preset: '{name}'. Available: {available}"
        )
    return PRESETS[name]
