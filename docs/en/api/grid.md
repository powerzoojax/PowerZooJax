# Grid environments

Public API for `TransGridEnv` and shared transmission infrastructure (DC / AC power flow, OPF helpers). For physics, see [Physics → Transmission](../physics/transmission.md). For radial distribution envs, see [API → Distribution](distribution.md). For the SCUC env used by the TSO benchmark, see [API → Unit commitment](grid-uc.md).

## `TransGridEnv`

```python
from powerzoojax.envs import TransGridEnv, make_trans_params
```

### Effective modes

| `physics` | `solver_mode` | Dispatch source | Network | Safety |
| --- | --- | --- | --- | --- |
| `0` | `0` | agent | DC PF | MW line limits |
| `1` | `0` | agent | Newton-Raphson AC PF | apparent-power thermal + voltage bounds |
| `0` | `1` | DCOPF | DC PF | MW line limits |
| `1` | `1` | DCOPF | DCOPF + AC PF ex-post check | apparent-power thermal + voltage bounds |
| `0` or `1` | `2` | ACOPF | ACOPF | apparent-power thermal + voltage bounds |

### Step contract

- `reward = -reward_scale * gen_cost`.
- `costs = [cost_thermal_overload, cost_voltage_violation, cost_power_balance, cost_resource]`.
- `info["cost_sum"] = sum(costs)` is the aggregate diagnostic.
- DC observation: `[line_flow / cap, load / total_cap, unit_p / p_max, sin(t), cos(t), <bundle_obs>]`.
- AC observation: `[|S| / cap, vm, load / total_cap, unit_p / p_max, sin(t), cos(t), <bundle_obs>]`.

::: powerzoojax.envs.grid.trans.TransGridEnv
    options:
      show_source: false
      members:
        - reset
        - step
        - observation_space
        - action_space

::: powerzoojax.envs.grid.trans.TransGridState

::: powerzoojax.envs.grid.trans.TransGridParams

::: powerzoojax.envs.grid.trans.make_trans_params

## Shared state containers

::: powerzoojax.envs.grid.base.GridState

::: powerzoojax.envs.grid.base.GridParams

## DC power flow

::: powerzoojax.envs.grid.power_flow.dc_power_flow

::: powerzoojax.envs.grid.power_flow.dc_power_flow_with_check

::: powerzoojax.envs.grid.power_flow.safety_check

::: powerzoojax.envs.grid.power_flow.compute_generation_cost

::: powerzoojax.envs.grid.power_flow.proportional_dispatch

## AC power flow

::: powerzoojax.envs.grid.ac_power_flow.prepare_acpf

::: powerzoojax.envs.grid.ac_power_flow.ac_power_flow

::: powerzoojax.envs.grid.ac_power_flow.ACPFSetup

::: powerzoojax.envs.grid.ac_power_flow.ACPFResult

## OPF modules

### DCOPF

::: powerzoojax.envs.grid.dc_opf.prepare_dcopf

::: powerzoojax.envs.grid.dc_opf.dc_opf

::: powerzoojax.envs.grid.dc_opf.DCOPFSetup

::: powerzoojax.envs.grid.dc_opf.DCOPFResult

### ACOPF

::: powerzoojax.envs.grid.ac_opf.ac_opf

::: powerzoojax.envs.grid.ac_opf.ACOPFSetup
