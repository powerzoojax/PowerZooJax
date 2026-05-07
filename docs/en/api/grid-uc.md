# Unit commitment (TSO)

Public API for `UnitCommitmentEnv` and the TSO benchmark factories. For physics, see [Physics → Transmission](../physics/transmission.md#unitcommitmentenv-scuc-for-the-tso-task).

## `UnitCommitmentEnv`

```python
from powerzoojax.envs import (
    UnitCommitmentEnv,
    UCState,
    UCParams,
    make_uc_params,
)
```

### Step contract

- Action: `Box(2 * n_units)` in `[-1, 1]` = `[commitment_signal (n_units) | dispatch_target (n_units)]`.
- Reward: `-reward_scale * (gen_cost + startup_cost + no_load_cost)`.
- Constraint costs: `costs = (w_th * cost_thermal_overload, cost_reserve_shortfall, cost_min_updown)`. `cost_min_updown` is currently always 0 because masking guarantees feasibility.

### Observation layout

`[unit_status | time_in_state_norm | last_dispatch_norm | unit_cost_b_norm | line_flow_norm | load_norm | reserve_ratio | sin(t) | cos(t) | future_total_load_norm[t+1:t+H]]`. Total dim `4 * n_units + n_lines + 4 + H`, where `H = forecast_horizon_steps`.
This is the policy input vector, not a separate raw telemetry API. For field-by-field semantics, see [Physics → Transmission](../physics/transmission.md#observation).

::: powerzoojax.envs.grid.unit_commitment.UnitCommitmentEnv
    options:
      show_source: false
      members:
        - reset
        - step
        - observation_space
        - action_space

::: powerzoojax.envs.grid.unit_commitment.UCState

::: powerzoojax.envs.grid.unit_commitment.UCParams

::: powerzoojax.envs.grid.unit_commitment.make_uc_params

## TSO factories

```python
from powerzoojax.envs import (
    make_tso_case118_params,
    make_tso_case14_params,
    make_tso_ed_params,
    make_tso_uc_params,
    make_tso_scuc_params,
)
```

::: powerzoojax.tasks.tso.make_tso_case118_params

::: powerzoojax.tasks.tso.make_tso_case14_params

::: powerzoojax.tasks.tso.make_tso_ed_params

::: powerzoojax.tasks.tso.make_tso_uc_params

::: powerzoojax.tasks.tso.make_tso_scuc_params

## TSO net-load helpers

::: powerzoojax.tasks.tso.make_tso_net_load_profiles

::: powerzoojax.tasks.tso.make_tso_net_load_profiles_from_data

## Non-learning baselines

```python
from powerzoojax.envs import tso_all_on_rollout, tso_merit_order_rollout
```

::: powerzoojax.tasks.tso.tso_all_on_rollout

::: powerzoojax.tasks.tso.tso_merit_order_rollout

## Metrics

::: powerzoojax.tasks.tso.compute_tso_metrics
