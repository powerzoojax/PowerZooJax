# DERs - Heterogeneous Multi-Agent Voltage Regulation

The DERs benchmark is a cooperative multi-agent reinforcement learning (MARL) task on a distribution feeder. Twelve heterogeneous agents on `case141` coordinate to keep voltages inside `[0.94, 1.06]` per-unit (`p.u.`) while reducing network loss. The agents are 4 batteries, 4 PV inverters, and 4 flexible loads placed at different buses along the feeder. For power-system background terms such as DER, `p.u.`, voltage regulation, and radial BFS power flow, see the [Power systems primer](../concepts/power-systems-primer.md).

This page describes the benchmark contract. A key point up front: the benchmark task exposes an explicit constraint specification, but the current benchmark training path is still reward-shaped IPPO rather than an online constrained MARL optimizer with dual updates.

## At A Glance

- Physical env: distribution-grid env on `case141` with 12 heterogeneous resources.
- Benchmark task: 12-agent cooperative voltage-regulation and loss-reduction MARL.
- What is actually trained: typed-parameter-sharing IPPO variants with fixed reward shaping; `ippo_lagrangian` is present in the result surface but is not the DERs headline.
- Primary leaderboard quantity: `mean_p_loss_mw`.
- Safety gate: zero voltage-violation target in the task config.

For a first pass, read this page in four layers: the summary above, the task-spec table, the reward-and-cost section, and finally the training / eval workflow. That order is especially important here because the task definition is constraint-aware, while the current formal training path is still reward-shaped.

## Task specification

| Field | Value |
| --- | --- |
| Task type | 12-agent cooperative Dec-POMDP with local observations |
| Agents | 12 (`4 battery + 4 pv + 4 flexload`) |
| State \(\mathcal{S}\) | bus voltages, branch flows, resource operating status, time-of-day phase |
| Observation \(\mathcal{O}_i\) | local K-hop neighborhood, 15-D (`1 + K + 3 + 2 + 5` with `K=4`) |
| Action \(\mathcal{A}_i\) | `Box(2)` per agent |
| Transition \(\mathcal{P}\) | resource actions change injections and flexible demand, then `case141` distribution power flow and safety checks are applied |
| Reward \(r_t\) | shared team reward \(r_t = -w_{\mathrm{loss}}\, P_{\mathrm{loss}, t}^{\mathrm{MW}}\) |
| Cost \(\mathbf{c}_t\) | task constraint vector \(\mathbf{c}_t = \left(C_t^{\mathrm{volt}}, C_t^{\mathrm{therm}}, C_t^{\mathrm{resource}}\right)\) |
| Threshold \(\mathbf{b}\) | task-level safety gate is `voltage_violation_rate <= 0.0` (single voltage gate, from `task.yaml::safety_thresholds`); the IPPO-Lagrangian config additionally uses per-channel `cost_thresholds: [0.0, 0.0, 0.0]` over `(voltage, thermal, resource)` for its dual variable, but those are training-side budgets and not the leaderboard gate |
| Discount \(\gamma\) | `0.995` |
| Horizon \(T\) | 48 steps x 30 min = 24 h |
| Initial \(\mu_0\) | split-driven real-data episode on `case141` |

`Dec-POMDP` means decentralized partially observable Markov decision process: each agent acts from a local observation, but all agents share one team-level return.

## Underlying physics

The task is built on the distribution-grid stack described in [Physics -> Distribution](../physics/distribution.md) and [Physics -> Resources](../physics/resources.md). The formal benchmark uses `case141`, a 141-bus Caracas-area feeder, and attaches 12 heterogeneous resources.

At the physical env level, voltage safety, thermal overload, and resource-side violations are tracked separately. The DERs task keeps those channels in its task-level constraint specification:

\[
\mathbf{c}_t = \left(C_t^{\mathrm{volt}},\, C_t^{\mathrm{therm}},\, C_t^{\mathrm{resource}}\right)
\]

However, the current benchmark training path does not run an online constrained MARL algorithm on this vector. Instead, the training presets use reward shaping with fixed voltage penalties. That distinction is important: DERs is constraint-aware at the task-definition level, but the current formal trainer is still a shaped-reward MARL path.

## Benchmark task parameters

| Parameter | Value |
| --- | --- |
| Case | `case141` |
| Agents | 12 (`4 battery + 4 pv + 4 flexload`) |
| Action per agent | `Box(2)` |
| Observation per agent | local K-hop neighborhood, 15-D (`1 + K + 3 + 2 + 5` with `K=4`) |
| Episode | 48 steps x 30 min = 24 h |
| Voltage limits | `[0.94, 1.06]` p.u. |
| Primary metric | `mean_p_loss_mw` (`lower_is_better`) |
| Safety gate | zero voltage-violation target in the frozen task config |

## Agent deployment

| Type | Count | Buses | Notes |
| --- | --- | --- | --- |
| Battery | 4 | 9, 55, 17, 122 | 0.10 MW / 0.30 MWh |
| PV | 4 | 6, 73, 72, 82 | 0.20 MW nameplate |
| FlexLoad | 4 | 41, 70, 135, 24 | 0.10 MW curtail / shift cap |

Each agent has a 2-D action, but the physical meaning depends on the resource type. The benchmark uses typed parameter sharing in training, so batteries, PV inverters, and flexible loads need not share one policy head. See [Training -> Trainers](../training/trainers.md) for `IPPO` and typed parameter sharing.

## Reward and cost

The shared team reward is the feeder-loss objective:

\[
r_t = -w_{\mathrm{loss}}\, P_{\mathrm{loss}, t}^{\mathrm{MW}}
\]

where \(P_{\mathrm{loss}, t}^{\mathrm{MW}}\) is the total network active-power loss at step \(t\), shared by all 12 agents.

The task-level constraint vector is

\[
\mathbf{c}_t = \left(C_t^{\mathrm{volt}},\, C_t^{\mathrm{therm}},\, C_t^{\mathrm{resource}}\right)
\]

matching the task `ConstraintSpec` and the underlying distribution env diagnostics.

Constraint channel definitions:

| Symbol | Constraint name | Typical info / metric key | Meaning |
| --- | --- | --- | --- |
| \(C_t^{\mathrm{volt}}\) | `voltage_violation` | `cost_voltage_violation`, `voltage_violation_steps`, `voltage_violation_rate` | Voltage-safety channel; counts whether any bus is outside the configured voltage band on a step, and is summarized as violation steps/rate. |
| \(C_t^{\mathrm{therm}}\) | `thermal_overload` | `cost_thermal_overload` | Thermal branch-overload diagnostic from the underlying distribution power-flow env. |
| \(C_t^{\mathrm{resource}}\) | `resource` | `cost_resource`, `total_cost` | Aggregate resource-side constraint cost from batteries, PV inverters, and flexible-load bundles. |

The current benchmark training path then shapes the reward with a fixed voltage penalty rather than learning a dual variable:

\[
r^{\mathrm{train}}_t = r_t - \lambda_{\mathrm{volt}} C_t^{\mathrm{volt}}
\]

where \(\lambda_{\mathrm{volt}}\) is the configured `voltage_penalty`. In the frozen benchmark setup:

- `ippo` uses the base DERs training config with a moderate fixed voltage penalty.
- `ippo_safe` uses a stronger fixed voltage penalty.

So `safe` here means stronger fixed shaping, not a separate constrained optimizer.

## What "safe" means here

`ders-medium-safe` does not change the underlying physical task. It uses the same env, the same agents, and the same task-level constraint channels. What changes is only the training objective:

- `ippo`: reward-shaped team training with a moderate fixed voltage penalty (`voltage_penalty=4.0`)
- `ippo_safe`: reward-shaped team training with a stronger fixed voltage penalty (`voltage_penalty=8.0`, twice the unconstrained 4.0; paper Appendix H.2 calls this baseline IPPO-rs)

`ippo_lagrangian` is reported in the formal result surface as a CMDP variant,
but the current evidence does not make it the DERs headline. This is why the
DERs benchmark page should not describe the current formal training path as
online CMDP / Lagrangian MARL.

## Baselines

| Name | Description |
| --- | --- |
| `no_control` | all DER actions = 0 |
| `volt_droop` | voltage-droop rule-based baseline with local reactive / curtailment response |

The `volt_droop` baseline is the stronger hand-crafted anchor used in the benchmark summaries.

## Algorithms

| Algo | Preset | Notes |
| --- | --- | --- |
| `ippo` | `ders-medium` | typed-parameter-sharing cooperative MARL baseline |
| `ippo_safe` | `ders-medium-safe` | same MARL path with stronger fixed voltage penalty |
| `ippo_lagrangian` | `ders-medium-safe` | CMDP variant retained in the formal matrix; not the headline row because it remains brittle on PV shift |

Hidden dims `(128, 128)`; gamma `0.995`; total timesteps `10e6`; 5 seeds
for Phase-1 JAX/GPU reporting. The mandatory seed-0 backend/device matrix has
official eval rows on `train`, `iid`, `voltage_tightening`,
`pv_penetration_shift`, and `load_stress`.

## Eval splits

| Split | Description |
| --- | --- |
| `train` | training distribution |
| `iid` | held-out in-distribution episodes |
| `voltage_tightening` | tighter voltage band |
| `pv_penetration_shift` | shifted PV penetration |
| `load_stress` | higher demand stress |

The 3-phase wrapper remains available for evaluation experiments, but it is not part of the frozen benchmark split list.

## Metrics

The DERs page is easiest to read if metrics are separated into layers:

| Layer | Key | Description |
| --- | --- | --- |
| Step reward | `reward` | shared per-step team reward |
| Step constraint channel | `voltage_violation` | voltage-safety channel; summarized by `voltage_violation_steps` and `voltage_violation_rate` |
| Step constraint channel | `thermal_overload` | branch thermal-overload diagnostic from the grid env |
| Step constraint channel | `resource` | aggregate resource-side constraint diagnostic from attached DER bundles |
| Episode aggregate | `total_reward` | cumulative shared reward |
| Episode aggregate | `total_cost` | cumulative continuous safety-cost diagnostic |
| Episode aggregate | `mean_p_loss_mw` | mean network active-power loss |
| Episode aggregate | `voltage_violation_steps` | number of steps with voltage outside limits |
| Episode aggregate | `voltage_violation_rate` | fraction of evaluated steps with any voltage violation |
| Relative evaluation summary | `loss_reduction_pct` | loss reduction vs `no_control` |
| Relative evaluation summary | `cost_reduction_pct` | continuous safety-cost reduction vs `no_control` |
| Relative evaluation summary | `NormScore` | benchmark-normalized score against frozen baselines |

!!! note "Convergence target is loss, not reward"
    The frozen DERs convergence target is based on `mean_p_loss_mw`, **not** on `total_reward`. That is deliberate: the benchmark treats physical loss as the primary leaderboard quantity, so convergence is bound directly to that physical metric. Do not compare episode reward across leaderboard entries.

## Quick start

```bash
python benchmarks/ders/run_all.py --only baselines --seeds 0 1 2 3 4
python benchmarks/ders/run_all.py --only train --algos ippo ippo_safe ippo_lagrangian --seeds 0 1 2 3 4
python benchmarks/ders/run_all.py --only eval
python benchmarks/ders/run_all.py --only summarize
python benchmarks/ders/run_all.py --only plots
```

Shared workflow terms are defined in the [Benchmark workflow glossary](../glossary.md).

## Cross references

- [Physics -> Distribution](../physics/distribution.md)
- [Physics -> Resources](../physics/resources.md)
- [API -> RL MARL](../api/rl-marl.md)
- [Training -> Trainers](../training/trainers.md)
