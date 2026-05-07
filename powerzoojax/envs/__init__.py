"""Pure-functional environment surface for PowerZooJax.

`powerzoojax.envs` is the framework-neutral benchmark core: grid, resource,
market, and microgrid environments plus common spaces and base dataclasses.
Most symbols are exposed lazily so importing this package does not eagerly load
every submodule or optional dependency.

The intent is:
- physics and env contracts live here
- benchmark task assembly lives in `powerzoojax.tasks`
- training adapters stay in `powerzoojax.rl`
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    # Base
    "Environment",
    "EnvState",
    "EnvParams",
    # Spaces
    "Space",
    "Box",
    "Discrete",
    "MultiDiscrete",
    "MultiBinary",
    "make_box",
    "make_discrete",
    "make_multi_discrete",
    "make_multi_binary",
    # Grid
    "GridState",
    "GridParams",
    "TransGridEnv",
    "TransGridState",
    "TransGridParams",
    "make_trans_params",
    "dc_power_flow",
    "dc_power_flow_with_check",
    "DistGridEnv",
    "DistGridState",
    "DistGridParams",
    "make_dist_params",
    "DistGrid3PhaseEnv",
    "DistGrid3PhState",
    "DistGrid3PhParams",
    "make_dist_3phase_params",
    # Resource
    "BatteryEnv",
    "BatteryState",
    "BatteryParams",
    "RenewableEnv",
    "RenewableState",
    "RenewableParams",
    "SolarEnv",
    "WindEnv",
    "VehicleEnv",
    "VehicleState",
    "VehicleParams",
    "FlexLoadEnv",
    "FlexLoadState",
    "FlexLoadParams",
    "DataCenterEnv",
    "DataCenterState",
    "DataCenterParams",
    "make_datacenter_params",
    # DataCenter Microgrid (C1 + C2)
    "DieselParams",
    "compute_dg_power",
    "compute_dg_fuel_cost",
    "compute_dg_emissions",
    "DCMicrogridState",
    "DCMicrogridParams",
    "DataCenterMicrogridEnv",
    "make_dcmicrogrid_params",
    "make_dcmicrogrid_params_with_profiles",
    # Market
    "MarketState",
    "MarketParams",
    "CostBasedMarketEnv",
    "CostMarketState",
    "CostMarketParams",
    "make_cost_market_params",
    "SimpleLMPArbitrageEnv",
    "LMPMarketState",
    "LMPMarketParams",
    "make_lmp_market_params",
    "BidBasedMarketEnv",
    "BidMarketState",
    "BidMarketParams",
    "make_bid_market_params",
    "make_cost_segments",
    "piecewise_ed",
    # Exact SCED solver
    "OfferSCEDSetup",
    "OfferSCEDResult",
    "prepare_offer_sced",
    "offer_sced",
    # GenCos Market MARL core
    "MarketMARLState",
    "MarketMARLParams",
    "make_market_marl_params",
    "market_marl_reset",
    "market_marl_step",
    # TSO / Unit Commitment task
    "UCState",
    "UCParams",
    "UnitCommitmentEnv",
    "make_uc_params",
    "make_tso_case118_params",
    "make_tso_case14_params",
    "make_tso_ed_params",
    "make_tso_uc_params",
    "make_tso_scuc_params",
    "tso_all_on_rollout",
    "tso_merit_order_rollout",
    "compute_tso_metrics",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Base
    "Environment": ("powerzoojax.envs.base", "Environment"),
    "EnvState": ("powerzoojax.envs.base", "EnvState"),
    "EnvParams": ("powerzoojax.envs.base", "EnvParams"),
    # Spaces
    "Space": ("powerzoojax.envs.spaces", "Space"),
    "Box": ("powerzoojax.envs.spaces", "Box"),
    "Discrete": ("powerzoojax.envs.spaces", "Discrete"),
    "MultiDiscrete": ("powerzoojax.envs.spaces", "MultiDiscrete"),
    "MultiBinary": ("powerzoojax.envs.spaces", "MultiBinary"),
    "make_box": ("powerzoojax.envs.spaces", "make_box"),
    "make_discrete": ("powerzoojax.envs.spaces", "make_discrete"),
    "make_multi_discrete": ("powerzoojax.envs.spaces", "make_multi_discrete"),
    "make_multi_binary": ("powerzoojax.envs.spaces", "make_multi_binary"),
    # Grid (leaf modules — avoids importing all of grid/__init__.py)
    "GridState": ("powerzoojax.envs.grid.base", "GridState"),
    "GridParams": ("powerzoojax.envs.grid.base", "GridParams"),
    "dc_power_flow": ("powerzoojax.envs.grid.power_flow", "dc_power_flow"),
    "dc_power_flow_with_check": ("powerzoojax.envs.grid.power_flow", "dc_power_flow_with_check"),
    "TransGridEnv": ("powerzoojax.envs.grid.trans", "TransGridEnv"),
    "TransGridState": ("powerzoojax.envs.grid.trans", "TransGridState"),
    "TransGridParams": ("powerzoojax.envs.grid.trans", "TransGridParams"),
    "make_trans_params": ("powerzoojax.envs.grid.trans", "make_trans_params"),
    "DistGridEnv": ("powerzoojax.envs.grid.dist", "DistGridEnv"),
    "DistGridState": ("powerzoojax.envs.grid.dist", "DistGridState"),
    "DistGridParams": ("powerzoojax.envs.grid.dist", "DistGridParams"),
    "make_dist_params": ("powerzoojax.envs.grid.dist", "make_dist_params"),
    "DistGrid3PhaseEnv": ("powerzoojax.envs.grid.dist_3phase", "DistGrid3PhaseEnv"),
    "DistGrid3PhState": ("powerzoojax.envs.grid.dist_3phase", "DistGrid3PhState"),
    "DistGrid3PhParams": ("powerzoojax.envs.grid.dist_3phase", "DistGrid3PhParams"),
    "make_dist_3phase_params": ("powerzoojax.envs.grid.dist_3phase", "make_dist_3phase_params"),
    # Resource (delegate to resource subpackage __getattr__)
    "BatteryEnv": ("powerzoojax.envs.resource", "BatteryEnv"),
    "BatteryState": ("powerzoojax.envs.resource", "BatteryState"),
    "BatteryParams": ("powerzoojax.envs.resource", "BatteryParams"),
    "RenewableEnv": ("powerzoojax.envs.resource", "RenewableEnv"),
    "RenewableState": ("powerzoojax.envs.resource", "RenewableState"),
    "RenewableParams": ("powerzoojax.envs.resource", "RenewableParams"),
    "SolarEnv": ("powerzoojax.envs.resource", "SolarEnv"),
    "WindEnv": ("powerzoojax.envs.resource", "WindEnv"),
    "VehicleEnv": ("powerzoojax.envs.resource", "VehicleEnv"),
    "VehicleState": ("powerzoojax.envs.resource", "VehicleState"),
    "VehicleParams": ("powerzoojax.envs.resource", "VehicleParams"),
    "FlexLoadEnv": ("powerzoojax.envs.resource", "FlexLoadEnv"),
    "FlexLoadState": ("powerzoojax.envs.resource", "FlexLoadState"),
    "FlexLoadParams": ("powerzoojax.envs.resource", "FlexLoadParams"),
    "DataCenterEnv": ("powerzoojax.envs.resource", "DataCenterEnv"),
    "DataCenterState": ("powerzoojax.envs.resource", "DataCenterState"),
    "DataCenterParams": ("powerzoojax.envs.resource", "DataCenterParams"),
    "make_datacenter_params": ("powerzoojax.envs.resource", "make_datacenter_params"),
    "DieselParams": ("powerzoojax.envs.resource", "DieselParams"),
    "compute_dg_power": ("powerzoojax.envs.resource", "compute_dg_power"),
    "compute_dg_fuel_cost": ("powerzoojax.envs.resource", "compute_dg_fuel_cost"),
    "compute_dg_emissions": ("powerzoojax.envs.resource", "compute_dg_emissions"),
    # DC microgrid is a composite (1-bus, no-PF) env, lives in envs.microgrid.
    "DCMicrogridState": ("powerzoojax.envs.microgrid", "DCMicrogridState"),
    "DCMicrogridParams": ("powerzoojax.envs.microgrid", "DCMicrogridParams"),
    "DataCenterMicrogridEnv": ("powerzoojax.envs.microgrid", "DataCenterMicrogridEnv"),
    "make_dcmicrogrid_params": ("powerzoojax.envs.microgrid", "make_dcmicrogrid_params"),
    "make_dcmicrogrid_params_with_profiles": ("powerzoojax.envs.microgrid", "make_dcmicrogrid_params_with_profiles"),
    # Market (leaf modules)
    "MarketState": ("powerzoojax.envs.market.base", "MarketState"),
    "MarketParams": ("powerzoojax.envs.market.base", "MarketParams"),
    "CostBasedMarketEnv": ("powerzoojax.envs.market.cost_based_market", "CostBasedMarketEnv"),
    "CostMarketState": ("powerzoojax.envs.market.cost_based_market", "CostMarketState"),
    "CostMarketParams": ("powerzoojax.envs.market.cost_based_market", "CostMarketParams"),
    "make_cost_market_params": ("powerzoojax.envs.market.cost_based_market", "make_cost_market_params"),
    "SimpleLMPArbitrageEnv": ("powerzoojax.envs.market.cost_based_market", "CostBasedMarketEnv"),
    "LMPMarketState": ("powerzoojax.envs.market.cost_based_market", "CostMarketState"),
    "LMPMarketParams": ("powerzoojax.envs.market.cost_based_market", "CostMarketParams"),
    "make_lmp_market_params": ("powerzoojax.envs.market.cost_based_market", "make_cost_market_params"),
    "BidBasedMarketEnv": ("powerzoojax.envs.market.bid_based_market", "BidBasedMarketEnv"),
    "BidMarketState": ("powerzoojax.envs.market.bid_based_market", "BidMarketState"),
    "BidMarketParams": ("powerzoojax.envs.market.bid_based_market", "BidMarketParams"),
    "make_bid_market_params": ("powerzoojax.envs.market.bid_based_market", "make_bid_market_params"),
    "make_cost_segments": ("powerzoojax.envs.market.clearing", "make_cost_segments"),
    "piecewise_ed": ("powerzoojax.envs.market.clearing", "piecewise_ed"),
    # Exact SCED solver
    "OfferSCEDSetup": ("powerzoojax.envs.market.offer_sced", "OfferSCEDSetup"),
    "OfferSCEDResult": ("powerzoojax.envs.market.offer_sced", "OfferSCEDResult"),
    "prepare_offer_sced": ("powerzoojax.envs.market.offer_sced", "prepare_offer_sced"),
    "offer_sced": ("powerzoojax.envs.market.offer_sced", "offer_sced"),
    # GenCos Market MARL core
    "MarketMARLState": ("powerzoojax.envs.market.market_marl_core", "MarketMARLState"),
    "MarketMARLParams": ("powerzoojax.envs.market.market_marl_core", "MarketMARLParams"),
    "make_market_marl_params": ("powerzoojax.envs.market.market_marl_core", "make_market_marl_params"),
    "market_marl_reset": ("powerzoojax.envs.market.market_marl_core", "market_marl_reset"),
    "market_marl_step": ("powerzoojax.envs.market.market_marl_core", "market_marl_step"),
    # TSO / Unit Commitment task
    "UCState": ("powerzoojax.envs.grid.unit_commitment", "UCState"),
    "UCParams": ("powerzoojax.envs.grid.unit_commitment", "UCParams"),
    "UnitCommitmentEnv": ("powerzoojax.envs.grid.unit_commitment", "UnitCommitmentEnv"),
    "make_uc_params": ("powerzoojax.envs.grid.unit_commitment", "make_uc_params"),
    "make_tso_case118_params": ("powerzoojax.tasks.tso", "make_tso_case118_params"),
    "make_tso_case14_params": ("powerzoojax.tasks.tso", "make_tso_case14_params"),
    "make_tso_ed_params": ("powerzoojax.tasks.tso", "make_tso_ed_params"),
    "make_tso_uc_params": ("powerzoojax.tasks.tso", "make_tso_uc_params"),
    "make_tso_scuc_params": ("powerzoojax.tasks.tso", "make_tso_scuc_params"),
    "tso_all_on_rollout": ("powerzoojax.tasks.tso", "tso_all_on_rollout"),
    "tso_merit_order_rollout": ("powerzoojax.tasks.tso", "tso_merit_order_rollout"),
    "compute_tso_metrics": ("powerzoojax.tasks.tso", "compute_tso_metrics"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        mod_path, attr = _LAZY_IMPORTS[name]
        mod = importlib.import_module(mod_path)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
