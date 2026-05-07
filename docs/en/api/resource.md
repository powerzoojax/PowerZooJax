# Resources

Public API for standalone resource environments and the bundle protocol used to attach devices to grid, market, and microgrid envs. For physics, see [Physics → Resources](../physics/resources.md). For the composed data-center microgrid, see [API → Microgrid](microgrid.md).

## Quick orientation

Use the resource surface in two modes:

- standalone `*Env`: run one device by itself
- `*Bundle`: attach many devices to a parent env through `params.resources`

Not every resource provides both forms:

| Resource | Standalone env | Bundle |
| --- | --- | --- |
| Battery | `BatteryEnv` | `BatteryBundle` |
| Renewable | `RenewableEnv` / `SolarEnv` / `WindEnv` | `RenewableBundle` |
| Vehicle | `VehicleEnv` | no public bundle |
| FlexLoad | `FlexLoadEnv` | `FlexLoadBundle` |
| Diesel | no standalone env | `DieselBundle` |
| Data center | `DataCenterEnv` | no public bundle |

## Cost semantics by resource

| Resource | `reward` | `costs` |
| --- | --- | --- |
| `BatteryEnv` | `0.0` | cycle-throughput cost |
| `RenewableEnv` | `0.0` | always `0.0` |
| `VehicleEnv` | `0.0` | departure SOC shortfall |
| `FlexLoadEnv` | `0.0` | curtailment + deferred-demand discomfort + simultaneous activation penalty |
| `DataCenterEnv` | `0.0` | expired-task SLA density |
| `DieselBundle` | no standalone reward | `cost_info["cost"] = 0.0`; fuel and carbon reported separately |

Diagnostics not folded into `costs`:

- `BatteryEnv`: `cost_action_clip`
- `DataCenterEnv`: `cost_overtemp`
- `DieselBundle`: `fuel_cost`, `carbon_kg`

## `BatteryEnv` and `BatteryBundle`

```python
from powerzoojax.envs import BatteryEnv
from powerzoojax.envs.resource.battery import BatteryBundle, make_battery_bundle
```

Sign convention: `P > 0` discharge (inject), `P < 0` charge (draw). Feasible power is clipped by converter rating and SOC headroom; SOC update uses one-way charge / discharge efficiency.

::: powerzoojax.envs.resource.battery.BatteryEnv
    options:
      show_source: false
      members:
        - reset
        - step
        - observation_space
        - action_space

::: powerzoojax.envs.resource.battery.BatteryState

::: powerzoojax.envs.resource.battery.BatteryParams

::: powerzoojax.envs.resource.battery.make_battery_params

::: powerzoojax.envs.resource.battery.compute_feasible_power

::: powerzoojax.envs.resource.battery.update_soc

::: powerzoojax.envs.resource.battery.BatteryBundle

::: powerzoojax.envs.resource.battery.BatteryBundleState

::: powerzoojax.envs.resource.battery.make_battery_bundle

## `RenewableEnv`, `SolarEnv`, `WindEnv`, and `RenewableBundle`

```python
from powerzoojax.envs import RenewableEnv, SolarEnv, WindEnv
from powerzoojax.envs.resource.renewable import RenewableBundle, make_renewable_bundle
```

Action `a in [-1, 1]` maps to curtailment `(1 - a) / 2`. Output equals `capacity_mw * capacity_factor * (1 - curtailment)`. With `enable_q_control=True`, reactive power is clipped by inverter PQ-circle.

::: powerzoojax.envs.resource.renewable.RenewableEnv

::: powerzoojax.envs.resource.renewable.RenewableState

::: powerzoojax.envs.resource.renewable.RenewableParams

::: powerzoojax.envs.resource.renewable.SolarEnv

::: powerzoojax.envs.resource.renewable.WindEnv

::: powerzoojax.envs.resource.renewable.RenewableBundle

::: powerzoojax.envs.resource.renewable.RenewableBundleState

::: powerzoojax.envs.resource.renewable.make_renewable_bundle

## `VehicleEnv`

```python
from powerzoojax.envs import VehicleEnv
```

Schedule-driven SOC dynamics. Trip energy is subtracted at departure; charge / V2G discharge is allowed only when `is_home=1`. `info["cost"]` penalizes leaving below `soc_departure_min`.

::: powerzoojax.envs.resource.vehicle.VehicleEnv

::: powerzoojax.envs.resource.vehicle.VehicleState

::: powerzoojax.envs.resource.vehicle.VehicleParams

::: powerzoojax.envs.resource.vehicle.make_vehicle_params

## `FlexLoadEnv` and `FlexLoadBundle`

```python
from powerzoojax.envs import FlexLoadEnv
from powerzoojax.envs.resource.flexload import FlexLoadBundle, make_flexload_bundle
```

Two actions: curtail now and shift demand out now. Buffered demand is released over the next `shift_horizon` steps. Net injection follows `curtail + shift_out - shift_in`. Optional `lmp=` makes price visible in the observation but does not change physics.

::: powerzoojax.envs.resource.flexload.FlexLoadEnv

::: powerzoojax.envs.resource.flexload.FlexLoadState

::: powerzoojax.envs.resource.flexload.FlexLoadParams

::: powerzoojax.envs.resource.flexload.FlexLoadBundle

::: powerzoojax.envs.resource.flexload.FlexLoadBundleState

::: powerzoojax.envs.resource.flexload.make_flexload_bundle

## `DieselBundle`

```python
from powerzoojax.envs.resource.diesel import (
    DieselParams,
    DieselBundle,
    DieselBundleState,
    make_diesel_bundle,
    compute_dg_power,
    compute_dg_fuel_cost,
    compute_dg_emissions,
)
```

Diesel is currently exposed as pure helpers plus a grid-attachable bundle, not as a standalone `Environment`. Action per device is a scalar in `[0, 1]` mapped to active power in `[0, p_max]`. `cost_info["cost"]` stays zero; fuel and carbon are reported separately.

::: powerzoojax.envs.resource.diesel.DieselParams

::: powerzoojax.envs.resource.diesel.compute_dg_power

::: powerzoojax.envs.resource.diesel.compute_dg_fuel_cost

::: powerzoojax.envs.resource.diesel.compute_dg_emissions

::: powerzoojax.envs.resource.diesel.DieselBundle

::: powerzoojax.envs.resource.diesel.DieselBundleState

::: powerzoojax.envs.resource.diesel.make_diesel_bundle

## `DataCenterEnv`

```python
from powerzoojax.envs import DataCenterEnv, make_datacenter_params
```

Three coupled layers: IT power, cooling power, and zone thermal dynamics. Poisson task arrivals go into a fixed-size masked buffer; an EDF-style greedy scheduler allocates GPUs. Active power is always load-like (`current_p_mw <= 0`).

::: powerzoojax.envs.resource.datacenter.DataCenterEnv

::: powerzoojax.envs.resource.datacenter.DataCenterState

::: powerzoojax.envs.resource.datacenter.DataCenterParams

::: powerzoojax.envs.resource.datacenter.make_datacenter_params

## Base resource protocol

::: powerzoojax.envs.resource.base.ResourceState

::: powerzoojax.envs.resource.base.ResourceParams

::: powerzoojax.envs.resource.base.ResourceBundle

::: powerzoojax.envs.resource.base.ResourceBundleState
