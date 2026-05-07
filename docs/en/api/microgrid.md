# Data center microgrid

Public API for PowerZooJax's current microgrid environment surface. At the moment, that surface is the single behind-the-meter `DataCenterMicrogridEnv`, which models a self-contained microgrid composed of a data-center load, battery, PV, and diesel generation.

This page documents that concrete environment API rather than the generic microgrid concept. For the physical interpretation and system-level description, see [Physics / Microgrid](../physics/microgrid.md).

## `DataCenterMicrogridEnv`

```python
from powerzoojax.envs import (
    DataCenterMicrogridEnv,
    DCMicrogridState,
    DCMicrogridParams,
    make_dcmicrogrid_params,
    make_dcmicrogrid_params_with_profiles,
)
```

### Step contract

- Action: 5-D `Box` `[train_sched, ft_sched, cooling_norm, batt_norm, dg_norm]`. `batt_norm in [-1, 1]` (positive = discharge); others in `[0, 1]`.
- Observation: 24-D, with workload, thermal, energy-asset, power-balance, battery-headroom, grid-price, and time channels (see [Physics / Microgrid](../physics/microgrid.md#observation-24-d)).
- Reward: scalar `r_energy + w_cost * r_cost + w_carbon * r_carbon`. `info["reward_vector"] = [r_energy, r_cost, r_carbon]` exposes the unscaled components separately.
- Constraint costs: `costs = (cost_sla, cost_overtemp, cost_power_deficit)`, with `info["cost_sum"]` as the aggregate diagnostic.

Here `train_sched` and `ft_sched` are sequential GPU-budget fractions, not fixed GPU counts: training first consumes a fraction of the current headroom, then finetuning consumes a fraction of the remaining headroom. See [Physics / Microgrid](../physics/microgrid.md#action) for the exact scheduling semantics.

::: powerzoojax.envs.microgrid.dc_microgrid.DataCenterMicrogridEnv
    options:
      show_source: false
      members:
        - reset
        - step
        - observation_space
        - action_space

::: powerzoojax.envs.microgrid.dc_microgrid.DCMicrogridState

::: powerzoojax.envs.microgrid.dc_microgrid.DCMicrogridParams

::: powerzoojax.envs.microgrid.dc_microgrid.make_dcmicrogrid_params

::: powerzoojax.envs.microgrid.dc_microgrid.make_dcmicrogrid_params_with_profiles

## Diesel generator helpers

```python
from powerzoojax.envs import (
    DieselParams,
    compute_dg_power,
    compute_dg_fuel_cost,
    compute_dg_emissions,
)
```

`DieselParams` carries `p_dg_max_mw`, `fuel_cost_per_mwh`, and `emission_factor` (kgCO2 / kWh). All three helpers are pure functions.

::: powerzoojax.envs.microgrid.dc_microgrid.DieselParams

::: powerzoojax.envs.microgrid.dc_microgrid.compute_dg_power

::: powerzoojax.envs.microgrid.dc_microgrid.compute_dg_fuel_cost

::: powerzoojax.envs.microgrid.dc_microgrid.compute_dg_emissions
