"""Resource models and bundle protocol for PowerZooJax.

This package contains standalone resource envs and the bundle components that
attach to grid or microgrid envs:
- battery, renewable, diesel, vehicle, flex-load, and data-center resources
- shared resource/base dataclasses
- the `ResourceBundle` protocol used by composite environments

Imports are lazy so battery-only or data-center-only workflows do not pay to
load the whole resource stack up front.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    # Base
    "ResourceState",
    "ResourceParams",
    "time_features",
    # Battery
    "BatteryState",
    "BatteryParams",
    "BatteryEnv",
    "make_battery_params",
    "compute_feasible_power",
    "update_soc",
    "compute_feasible_power_batch",
    "update_soc_batch",
    # Battery Bundle
    "BatteryBundle",
    "BatteryBundleState",
    "make_battery_bundle",
    # Bundle protocol
    "ResourceBundle",
    "ResourceBundleState",
    # Renewable
    "RenewableState",
    "RenewableParams",
    "RenewableEnv",
    "SolarEnv",
    "WindEnv",
    # Renewable Bundle
    "RenewableBundleState",
    "RenewableBundle",
    "make_renewable_bundle",
    # Diesel
    "DieselParams",
    "compute_dg_power",
    "compute_dg_fuel_cost",
    "compute_dg_emissions",
    "DieselBundleState",
    "DieselBundle",
    "make_diesel_bundle",
    # Vehicle
    "VehicleState",
    "VehicleParams",
    "VehicleEnv",
    "make_vehicle_params",
    # FlexLoad
    "FlexLoadState",
    "FlexLoadParams",
    "FlexLoadEnv",
    "FlexLoadBundleState",
    "FlexLoadBundle",
    "make_flexload_bundle",
    # DataCenter
    "DataCenterState",
    "DataCenterParams",
    "DataCenterEnv",
    "make_datacenter_params",
    # NOTE: DC Microgrid is a composite env (1-bus, no PF) and lives in
    # ``powerzoojax.envs.microgrid``.  It used to be re-exported here for
    # backward compat; the alias was removed because no caller relied on it.
]

# (module_path, attribute_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "ResourceState": ("powerzoojax.envs.resource.base", "ResourceState"),
    "ResourceParams": ("powerzoojax.envs.resource.base", "ResourceParams"),
    "time_features": ("powerzoojax.envs.resource.base", "time_features"),
    "BatteryState": ("powerzoojax.envs.resource.battery", "BatteryState"),
    "BatteryParams": ("powerzoojax.envs.resource.battery", "BatteryParams"),
    "BatteryEnv": ("powerzoojax.envs.resource.battery", "BatteryEnv"),
    "make_battery_params": ("powerzoojax.envs.resource.battery", "make_battery_params"),
    "compute_feasible_power": ("powerzoojax.envs.resource.battery", "compute_feasible_power"),
    "update_soc": ("powerzoojax.envs.resource.battery", "update_soc"),
    "compute_feasible_power_batch": ("powerzoojax.envs.resource.battery", "compute_feasible_power_batch"),
    "update_soc_batch": ("powerzoojax.envs.resource.battery", "update_soc_batch"),
    "BatteryBundle": ("powerzoojax.envs.resource.battery", "BatteryBundle"),
    "BatteryBundleState": ("powerzoojax.envs.resource.battery", "BatteryBundleState"),
    "make_battery_bundle": ("powerzoojax.envs.resource.battery", "make_battery_bundle"),
    "ResourceBundle": ("powerzoojax.envs.resource.base", "ResourceBundle"),
    "ResourceBundleState": ("powerzoojax.envs.resource.base", "ResourceBundleState"),
    "RenewableState": ("powerzoojax.envs.resource.renewable", "RenewableState"),
    "RenewableParams": ("powerzoojax.envs.resource.renewable", "RenewableParams"),
    "RenewableEnv": ("powerzoojax.envs.resource.renewable", "RenewableEnv"),
    "SolarEnv": ("powerzoojax.envs.resource.renewable", "SolarEnv"),
    "WindEnv": ("powerzoojax.envs.resource.renewable", "WindEnv"),
    "RenewableBundleState": ("powerzoojax.envs.resource.renewable", "RenewableBundleState"),
    "RenewableBundle": ("powerzoojax.envs.resource.renewable", "RenewableBundle"),
    "make_renewable_bundle": ("powerzoojax.envs.resource.renewable", "make_renewable_bundle"),
    "DieselParams": ("powerzoojax.envs.resource.diesel", "DieselParams"),
    "compute_dg_power": ("powerzoojax.envs.resource.diesel", "compute_dg_power"),
    "compute_dg_fuel_cost": ("powerzoojax.envs.resource.diesel", "compute_dg_fuel_cost"),
    "compute_dg_emissions": ("powerzoojax.envs.resource.diesel", "compute_dg_emissions"),
    "DieselBundleState": ("powerzoojax.envs.resource.diesel", "DieselBundleState"),
    "DieselBundle": ("powerzoojax.envs.resource.diesel", "DieselBundle"),
    "make_diesel_bundle": ("powerzoojax.envs.resource.diesel", "make_diesel_bundle"),
    "VehicleState": ("powerzoojax.envs.resource.vehicle", "VehicleState"),
    "VehicleParams": ("powerzoojax.envs.resource.vehicle", "VehicleParams"),
    "VehicleEnv": ("powerzoojax.envs.resource.vehicle", "VehicleEnv"),
    "make_vehicle_params": ("powerzoojax.envs.resource.vehicle", "make_vehicle_params"),
    "FlexLoadState": ("powerzoojax.envs.resource.flexload", "FlexLoadState"),
    "FlexLoadParams": ("powerzoojax.envs.resource.flexload", "FlexLoadParams"),
    "FlexLoadEnv": ("powerzoojax.envs.resource.flexload", "FlexLoadEnv"),
    "FlexLoadBundleState": ("powerzoojax.envs.resource.flexload", "FlexLoadBundleState"),
    "FlexLoadBundle": ("powerzoojax.envs.resource.flexload", "FlexLoadBundle"),
    "make_flexload_bundle": ("powerzoojax.envs.resource.flexload", "make_flexload_bundle"),
    "DataCenterState": ("powerzoojax.envs.resource.datacenter", "DataCenterState"),
    "DataCenterParams": ("powerzoojax.envs.resource.datacenter", "DataCenterParams"),
    "DataCenterEnv": ("powerzoojax.envs.resource.datacenter", "DataCenterEnv"),
    "make_datacenter_params": ("powerzoojax.envs.resource.datacenter", "make_datacenter_params"),
    # DC Microgrid lives in powerzoojax.envs.microgrid (composite env with
    # bundle-attached resources, no power flow).  Import from there directly.
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        mod_path, attr = _LAZY_IMPORTS[name]
        mod = importlib.import_module(mod_path)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
