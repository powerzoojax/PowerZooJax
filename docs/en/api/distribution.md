# Distribution environments

Public API for the radial distribution envs and their backward / forward sweep solvers. For physics, see [Physics → Distribution](../physics/distribution.md).

## `DistGridEnv`

```python
from powerzoojax.envs import DistGridEnv, make_dist_params
```

### Step contract

- `reward = -loss_penalty_weight * p_loss_MW`.
- `costs = [cost_voltage_violation, cost_thermal_overload, cost_resource]`.
- `info["cost_sum"] = sum(costs)` is the aggregate diagnostic.
- The DSO benchmark selects only `"voltage_violation"` at the task / wrapper layer. `cost_mode` is deprecated and kept only for backward-compatible config loading.
- `info["cost_continuous"]` is a continuous voltage / apparent-power overload diagnostic.
- `info["soc_terminal_sq"]` is populated on the terminal transition when a battery bundle is the first attached bundle.

### Observation layout

`[v_mag_norm | p_branch_norm | q_branch_norm | p_load_norm | q_load_norm | sin(t) | cos(t) | <bundle_obs>]`.
This is the policy input vector seen by `DistGridEnv`. For field semantics and the DSO-specific `FlexLoadBundle` slice, see [Physics → Distribution](../physics/distribution.md#distgridenv-balanced-radial-feeder).

::: powerzoojax.envs.grid.dist.DistGridEnv
    options:
      show_source: false
      members:
        - reset
        - step
        - observation_space
        - action_space

::: powerzoojax.envs.grid.dist.DistGridState

::: powerzoojax.envs.grid.dist.DistGridParams

::: powerzoojax.envs.grid.dist.make_dist_params

## `DistGrid3PhaseEnv`

```python
from powerzoojax.envs import DistGrid3PhaseEnv, make_dist_3phase_params
```

### Step contract

- `reward = -loss_penalty_weight * p_loss_MW`.
- `costs = [cost_voltage_violation, cost_thermal_overload, cost_vuf_violation, cost_resource]`.
- `info["cost_sum"] = sum(costs)` is the aggregate diagnostic.
- `info["max_vuf_percent"]` keeps the worst per-node VUF (Fortescue voltage unbalance factor in %).

::: powerzoojax.envs.grid.dist_3phase.DistGrid3PhaseEnv
    options:
      show_source: false
      members:
        - reset
        - step
        - observation_space
        - action_space

::: powerzoojax.envs.grid.dist_3phase.DistGrid3PhState

::: powerzoojax.envs.grid.dist_3phase.DistGrid3PhParams

::: powerzoojax.envs.grid.dist_3phase.make_dist_3phase_params

## Balanced BFS solver

::: powerzoojax.envs.grid.bfs_power_flow.prepare_bfs

::: powerzoojax.envs.grid.bfs_power_flow.bfs_power_flow

::: powerzoojax.envs.grid.bfs_power_flow.BFSTopoData

## Three-phase BFS solver

::: powerzoojax.envs.grid.bfs_3phase_power_flow.build_3phase_topology

::: powerzoojax.envs.grid.bfs_3phase_power_flow.bfs_3phase_power_flow

::: powerzoojax.envs.grid.bfs_3phase_power_flow.ThreePhaseTopoData
