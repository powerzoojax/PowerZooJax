# Power systems primer

This page is a quick glossary for ML readers. It is not a full electrical engineering course; it gives just enough vocabulary to read the rest of the documentation. Each term is one paragraph and is referenced from the physics and benchmark pages.

## Networks and topology

- Bus / node. A connection point in the network. Generators, loads, and storage are attached to buses.
- Line / branch. A wire between two buses. Each line has a thermal limit (maximum power it can carry).
- Transmission grid. High-voltage backbone (hundreds of kV) connecting power plants to substations. Roughly meshed topology, modeled with full AC equations or a DC linearization.
- Distribution grid. Medium / low voltage feeders that serve customers. Often radial (tree-shaped) and unbalanced across phases.
- Three-phase power. AC power systems usually deliver electricity through three voltage/current waveforms offset by 120°, commonly called phases A/B/C. In an ideally balanced three-phase system, the phase magnitudes are similar and the power transfer is smoother; distribution grids often host many uneven single-phase loads, so phase imbalance is common, which is why `DistGrid3PhaseEnv` models it explicitly.
- Slack bus / reference bus. The single bus whose voltage angle (and magnitude in AC) is fixed. The slack generator is the balancing generator: it increases or decreases its net injection so the power-flow equations close and total generation, load, and losses stay balanced.

## Power flow (PF)

Power flow describes how electrical power is distributed through the network once the grid topology, generation, load, and device parameters are given. Running a power-flow solve means solving the network balance equations to determine where power goes, how much each line carries, and whether the bus voltages stay within a normal range.

- DC power flow. A linearization that ignores reactive power and voltage magnitude variation. Line flow is a linear function of net nodal injection. Solved in one matrix multiply with the [PTDF matrix](https://ps-wiki.github.io/wiki/power-transfer-distribution-factor/).
- AC power flow. Full nonlinear power balance with active power `P`, reactive power `Q`, voltage magnitude `vm`, and angle `va`. Solved iteratively with [Newton-Raphson](https://matpower.app/manual/matpower/ACPowerFlow.html).
- Active power `P`. The part of electrical power that transfers net energy and can become useful work or heat. In power-system intuition, `P` is the "real energy delivery" channel and is more closely tied to frequency balance.
- Reactive power `Q`. The part of electrical power that oscillates between electric and magnetic fields to support AC equipment and voltage. Over one AC cycle it does not produce net work, but it is still essential for voltage support and for motors, transformers, and other inductive devices to operate normally.
- Voltage magnitude `vm`. The size `|V|` of the bus voltage, usually reported in per unit (p.u.) where `1.0` is nominal voltage. "Per-bus voltage magnitude" simply means the vector of voltage magnitudes at all buses.
- Backward / forward sweep (BFS). A specialized solver for radial distribution networks. It sweeps from leaves to root accumulating power, then from root to leaves updating voltages. Cheaper than [Newton-Raphson](https://matpower.app/manual/matpower/ACPowerFlow.html) on radial topologies.
- [PTDF](https://ps-wiki.github.io/wiki/power-transfer-distribution-factor/) (power transfer distribution factor). A precomputed matrix `PTDF[l, n]` giving the change in flow on line `l` per unit of injection at bus `n`. In DC PF, line flows are `PTDF @ p_inj`, where `p_inj` is the vector of net active-power injections at each bus (generation minus load, plus any resource injections).

## Generation, dispatch, and OPF

- Generator / unit. A controllable source with min and max output, a marginal-cost curve, and (for thermal units) ramp limits and on/off transitions.
- Dispatch. The active-power output of each generator at one moment in time.
- Economic dispatch (ED). Pick generator outputs to minimize total cost subject to power balance.
- Optimal power flow (OPF). ED plus network constraints (line limits, voltages).
- DCOPF / ACOPF. OPF using the DC PF or full AC PF model respectively.
- Merit order. The cost-priority ordering of generators from lowest to highest marginal cost. A **merit-order (priority-list) rule** dispatches or commits cheaper units first, then more expensive ones only if additional capacity is needed; it is a common deterministic engineering shortcut, not a global UC optimum.
- Unit commitment (UC) and SCUC. UC chooses which generators are ON over a multi-step horizon. SCUC adds security constraints (line limits, reserve). Both add discrete on/off decisions and intertemporal constraints (min-up / min-down time, startup costs, ramp limits).
- Reserve. Spare generation kept available in case of contingencies. SCUC must keep enough headroom to cover a fraction of demand.
- Locational marginal price (LMP). The marginal cost at each bus of supplying one extra MWh, taking network constraints into account. In a market context, LMPs are the prices used for settlement.

## Resources

- Distributed energy resource (DER). A small generator, storage unit, or controllable load attached at the distribution level. Examples: rooftop PV, residential battery, EV charger, flexible HVAC load.
- Battery / storage. A device with a state of charge (SOC) bounded between `soc_min` and `soc_max`. Discharging injects power, charging draws power. One-way charge/discharge efficiencies mean energy is lost on each round trip.
- Renewable (PV / wind). A profile-driven generator. Output is `capacity * capacity_factor(t) * (1 - curtailment)`. Curtailment is the agent's choice to reduce output below the available level.
- Vehicle / EV. A battery with a schedule. Available to charge or discharge only when the vehicle is at home; trips remove energy from the SOC and impose a minimum departure SOC.
- Flexible load. A controllable load that can be curtailed (reduce demand now and accept a discomfort cost) or shifted (defer demand to be released over a horizon). Sign convention: positive `current_p_mw` is load reduction.
- Data center. A behind-the-meter load that consumes power for IT (compute) and cooling. The agent can reschedule training and finetuning jobs and adjust the cooling setpoint.

## Quality and safety constraints

- Thermal limit. Maximum apparent or active power on a line. In DC PF, `|P_l| <= P_l^max`. In AC PF, `sqrt(P^2 + Q^2) <= S^max`.
- Voltage limits. `vm_min <= vm <= vm_max`, typically `[0.94, 1.06]` per unit (a normalized scale where 1.0 is nominal). Mostly enforced in distribution.
- Voltage unbalance factor (VUF). A measure of how much three-phase voltages differ from a balanced set. The Fortescue VUF is `|V_negative_sequence| / |V_positive_sequence|` in percent.
- Power balance. Total generation = total load + losses. The slack bus absorbs any residual mismatch.
- Per unit (p.u.). A unitless normalization. Powers are divided by a base power (`base_mva`); voltages by a base voltage. Equations become dimensionless and easier to compare.

## Markets

- Cost-based clearing. The market operator clears dispatch using the true generator cost curves (no strategic bidding).
- Bid-based clearing. Generators submit offer curves (price-quantity steps). The market operator clears using offers, not true costs. Recovered LMPs reflect offer prices.
- SCED (security-constrained economic dispatch). A single-period clearing problem that respects line limits.
- Storage arbitrage. Buying power when LMP is low, storing it, and selling when LMP is high. Revenue is `sum(LMP * P * dt)`.
- Strategic bidding. A generator submits offers above its true marginal cost to push the cleared price up. Multi-agent settings can produce non-cooperative equilibria.

## Time and units

- Step length. The simulation time per `step` call, `delta_t_hours`. Common settings are `0.5` h (30 min) for transmission/distribution/markets and `1/12` h (5 min) for the data-center microgrid.
- Episode length. `max_steps`. Most benchmarks use 48 steps × 30 min = 24 h. The data-center microgrid uses 288 steps × 5 min = 24 h.
- Active power `P` is measured in MW; reactive power `Q` in MVAr; energy in MWh. Paper-facing benchmark numbers are reported in GBP (£), with GB-based data sources (NESO demand, Elexon BMRS generation and grid-import prices, Ausgrid distribution feeders). Code-side cost coefficients in `case_data.py` are dimensioned generically with a `$` sign, but the canonical experiments use GB pricing.

## Common abbreviations

| Abbreviation | Meaning |
| --- | --- |
| AC / DC PF | Alternating-current / direct-current power flow |
| ACOPF / DCOPF | Optimal power flow under AC / DC PF model |
| BFS | Backward / forward sweep (radial PF solver) |
| CMDP | Constrained Markov decision process |
| DER | Distributed energy resource |
| ED / SCED | Economic dispatch / security-constrained ED |
| IPPO / MAPPO | Independent / centralized PPO for multi-agent RL |
| LMP | Locational marginal price |
| OPF | Optimal power flow |
| PPO | Proximal policy optimization |
| PTDF | Power transfer distribution factor |
| p.u. | Per unit (normalized) |
| RL / MARL | Reinforcement learning / multi-agent RL |
| SCUC | Security-constrained unit commitment |
| SLA | Service-level agreement (in DataCenter task: deadline obligation) |
| SOC | State of charge (battery energy as fraction of capacity) |
| TSO / DSO | Transmission / distribution system operator |
| VUF | Voltage unbalance factor |

The next layer ([Architecture](../architecture/repo-map.md)) shows how these concepts map onto the code modules.
