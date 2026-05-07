# Physics Overview

PowerZooJax models the integrated electricity system across multiple interconnected layers, from generation through transmission, distribution, end-use loads, and demand response. This page provides a high-level spatial and functional overview of the modeled domains and their relationships.

## System Architecture

The integrated power system model in PowerZooJax spans five interconnected layers:

![PowerZooJax physics overview diagram](../../assets/images/physics/fig1.png){ width="100%" }

*Figure 1. Cross-layer overview of the modeled physics domains in PowerZooJax.*

## How to read Figure 1

Figure 1 organizes PowerZooJax into three **physical system models** (transmission, distribution, microgrid), two **resource layers** (generation, end-use DER), and one **coordination layer** (market). Power flows left → right (generation to loads), while information signals (e.g., prices) propagate across layers.

## Layer-by-layer model boundaries (code-aligned)

### Market Layer — Coordination & price signals

The market layer is implemented under `powerzoojax.envs.market` as the coordination layer between economic signals and physical constraints. It provides **price and clearing signals** that influence upstream/downstream actions:

- **Upstream**: generator bidding / commitment and dispatch incentives
- **Grid operation**: system-level congestion and balancing incentives
- **Downstream**: end-use flexibility and DER response incentives

See [Markets Layer](markets.md) for clearing and pricing abstractions (`cost_based_market.py`, `bid_based_market.py`, `clearing.py`).

---

### Generation — Supply-side resources

The generation layer represents utility-scale supply that feeds the transmission network:

- **Renewables**: wind/solar injection following exogenous profiles
- **Thermal**: controllable plants with ramping, startup, and minimum output costs
- **Storage**: grid-scale batteries with efficiency and cycling constraints
- **Commitment**: optional unit commitment decisions in day-ahead style benchmarks

In code, supply-side decisions are realized through transmission environments and factories:
- [TransGridEnv](transmission.md)
- [UnitCommitmentEnv](transmission.md)

Key entry points:
- `make_trans_params(...)` in `powerzoojax.envs.grid.trans`
- `make_uc_params(...)` in `powerzoojax.envs.grid.unit_commitment`

---

### Transmission system model — Meshed bulk network (TSO)

The **transmission system model** is `TransGridEnv` (`powerzoojax.envs.grid.trans`) plus shared base structs in `powerzoojax.envs.grid.base`:

- **Physics switch**: `physics=0` (DC PTDF PF), `physics=1` (AC PF)
- **Dispatch switch**: `solver_mode=0` (direct PF), `1` (DCOPF), `2` (ACOPF)
- **Safety/cost outputs**: thermal overload, voltage violation, power-balance residual, and resource costs (CMDP-style vector)
- **UC extension**: `UnitCommitmentEnv` extends the grid state with commitment-specific variables while reusing the transmission core

See [Transmission Layer](transmission.md).

---

### Distribution system model — Radial feeders (DSO)

The **distribution system model** is implemented by two environments:
- `DistGridEnv` (`powerzoojax.envs.grid.dist`) for balanced radial feeders
- `DistGrid3PhaseEnv` (`powerzoojax.envs.grid.dist_3phase`) for unbalanced three-phase feeders

- **Radial topology**: prepared by `prepare_bfs(...)`
- **Power flow kernels**: `bfs_power_flow(...)` and `bfs_3phase_power_flow(...)`
- **Three-phase safety**: explicit VUF tracking (`vuf_max`) in the 3-phase env
- **Factory**: `make_dist_params(...)` configures solver tolerances, limits, and attached bundles

See [Distribution Layer](distribution.md).

---

### End-use DER — Flexible demand and behind-the-meter assets

The end-use DER layer is represented by reusable resource envs/bundles under `powerzoojax.envs.resource`:

- **Flexible demand**: `FlexLoadBundle`
- **EV behavior**: `VehicleEnv`
- **Storage**: `BatteryBundle`
- **PV/renewables**: `RenewableBundle`
- **Dispatchable local thermal**: `DieselBundle`

All bundles follow the same attachable interface (`reset/step/observe`) and are passed through `params.resources=(...)` to grid/microgrid envs.

See [Resources & Bundles](resources.md) for the reusable asset models used across environments.

---

### Data Center Microgrid system model — Coupled local power + computing

The **microgrid system model** is `DataCenterMicrogridEnv` in `powerzoojax.envs.microgrid.dc_microgrid`, with params built via `make_dcmg_params(...)`.

- **Default resource stack**: `(BatteryBundle, RenewableBundle, DieselBundle)`
- **Coupled dynamics**: battery SOC, diesel output constraints, PV profile-driven injection
- **Joint objective**: energy cost + data-center workload/thermal terms in one step loop

See [Microgrid Layer](microgrid.md) and `DataCenterMicrogridEnv`.

## What gets coupled across layers (implementation view)

- **Energy coupling**: generation dispatch and DER injections are assembled into net nodal injections before PF/OPF solves.
- **Constraint coupling**: line thermal limits, voltage bounds, and (3-phase) VUF limits become explicit cost/safety outputs.
- **Resource coupling**: attached bundles contribute both injections and per-bundle costs through the shared `resources` interface.
- **Signal coupling**: market outputs and task-level wrappers map economic signals to env actions/rewards without duplicating physics kernels.

