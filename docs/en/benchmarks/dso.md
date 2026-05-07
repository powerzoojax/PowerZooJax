# DSO - Network Loss Minimization

The DSO benchmark is a single-agent distribution-level demand-response task. One centralized controller schedules 6 flexible loads on the IEEE 33-bus Baran-Wu radial feeder to reduce network loss while keeping bus voltages inside a tight `[0.94, 1.06]` per-unit (`p.u.`) band. For background on radial feeders, backward/forward sweep (BFS), and `p.u.`, see the [Power systems primer](../concepts/power-systems-primer.md).

This page is the DSO task guide: it defines the benchmark contract, the frozen splits, the reward and CMDP semantics, and the generated outputs. Shared workflow vocabulary is in the [Benchmark workflow glossary](../glossary.md).

## At A Glance

- Physical env: `DistGridEnv` on IEEE `case33bw` with 6 `FlexLoad` devices.
- Benchmark task: single-agent demand response for loss minimization under voltage limits.
- What is actually trained: PPO, SAC, Sauté PPO, or PPO-Lagrangian on the DSO task wrapper, not the bare grid env alone.
- Main benchmark quantities: `total_reward` is the current convergence target, while `total_loss_mwh` is the most direct physical episode metric.
- Safety gate: zero voltage-violation rate in the task config.

If you are reading DSO for the first time, use this order: start with the summary above, then the MDP / CMDP table, then the reward-and-cost section, and only then the command examples. That keeps the physical task separate from the training workflow and reporting details.

## MDP / CMDP specification

| Field | Value |
| --- | --- |
| MDP class | MDP (`dso-nflex`) / CMDP (`dso-nflex-safe`) with task-selected constraint `("voltage_violation",)` |
| Agents | 1 (centralized) |
| State \(\mathcal{S}\) | bus voltages, branch flows, nodal loads, flexible-load status, time-of-day phase |
| Observation \(\mathcal{O}\) | `Box(195)`; exact field layout in [Physics -> Distribution](../physics/distribution.md#distgridenv-balanced-radial-feeder) |
| Action \(\mathcal{A}\) | `Box(12) = 6 x [curtail, shift_out]` |
| Transition \(\mathcal{P}\) | flexible-load actions modify feeder demand, then balanced radial BFS power flow is solved on `case33bw` |
| Reward \(r_t\) | \(r_t = -w_{\mathrm{loss}}\, P_{\mathrm{loss}, t}^{\mathrm{MW}}\) |
| Cost \(c_t\) | \(c_t = \left(C_t^{\mathrm{volt}}\right)\) |
| Threshold \(d\) | `cost_thresholds = (0.0,)` for `dso-nflex-safe`; N/A for `dso-nflex` |
| Discount \(\gamma\) | `0.995` |
| Horizon \(T\) | 48 steps x 30 min = 24 h |
| Initial \(\mu_0\) | data-driven episode sampled from Ausgrid FY25 load profiles |

`CMDP` means constrained Markov decision process: reward is the economic training signal, while safety violations are reported through a separate cost channel instead of being folded back into reward. In DSO, the task-level CMDP constraint is voltage only.

## Underlying physics

The benchmark is built on [`DistGridEnv`](../physics/distribution.md#distgridenv-balanced-radial-feeder), a balanced radial distribution-grid environment on the IEEE 33-bus Baran-Wu feeder (`case33bw`) with a [FlexLoad bundle](../physics/resources.md#flexible-load-flexloadenv). The controller does not dispatch batteries or direct DER injections in this task; it only schedules flexible demand.

The task intuition is simple: moving or curtailing demand away from stressed periods can reduce branch current, reduce voltage drops, and often reduce resistive \(I^2R\) loss. That makes DSO a clean benchmark for demand response, distinct from DERs, which adds heterogeneous distributed resources.

At the env level, `DistGridEnv` always exposes the full fixed-shape cost vector

\[
\mathbf{c}_t^{\mathrm{env}} = \left(C_t^{\mathrm{volt}},\, C_t^{\mathrm{therm}},\, C_t^{\mathrm{resource}}\right)
\]

with names `("voltage_violation", "thermal_overload", "resource")`; see [Physics -> Distribution](../physics/distribution.md) and [API -> Distribution](../api/distribution.md). The DSO benchmark does not change those env semantics. Instead, the DSO task layer selects only the voltage channel for its CMDP constraint spec:

\[
\mathbf{c}_t = \left(C_t^{\mathrm{volt}}\right)
\]

Env-level cost channel definitions:

| Symbol | Env constraint name | Info key | Meaning |
| --- | --- | --- | --- |
| \(C_t^{\mathrm{volt}}\) | `voltage_violation` | `cost_voltage_violation` | Count of buses outside the configured voltage band `[0.94, 1.06]` p.u. |
| \(C_t^{\mathrm{therm}}\) | `thermal_overload` | `cost_thermal_overload` | Thermal overload magnitude/count diagnostic for overloaded feeder branches. |
| \(C_t^{\mathrm{resource}}\) | `resource` | `cost_resource` | Aggregate resource-side constraint cost from attached bundles, if any. |

This distinction matters: DSO training and safety reporting are task-level, but the underlying grid env still computes the full diagnostic vector.

## Benchmark task parameters

| Parameter | Value |
| --- | --- |
| Case | `case33bw` |
| Resources | 6 FlexLoad devices across 3 feeder segments |
| Episode | 48 steps x 30 min = 24 h |
| Action space | `Box(12) = 6 x [curtail, shift_out]` |
| Observation | `Box(195)` |
| Voltage limits | `[0.94, 1.06]` p.u. |
| Data source | Ausgrid zone-substation load (FY25) |
| Official eval split | `iid` |
| Safety gate | `voltage_violation_rate <= 0.0` in frozen task config |

There is no battery in this task. That is the main benchmark difference from DERs: DSO isolates the value of pure demand response.

## Resource layout

`make_dso_flexload_bundle(case)` places six flexible loads at fixed buses:

| Device | Bus | `curtail_cap_mw` | `shift_cap_mw` |
| --- | --- | --- | --- |
| FL_A1 | 6 | 0.15 | 0.15 |
| FL_A2 | 14 | 0.10 | 0.10 |
| FL_A3 | 18 | 0.10 | 0.10 |
| FL_B1 | 22 | 0.08 | 0.08 |
| FL_C1 | 28 | 0.12 | 0.12 |
| FL_C2 | 33 | 0.10 | 0.10 |

These devices span three feeder branches, so the policy must coordinate globally rather than solve one purely local voltage-control problem.

## Action and observation

Each device has two controls: curtail load now, or shift load away from the current step into a short buffer. The benchmark observation combines feeder electrical state, current load state, time-of-day features, and per-device flexible-load status.

For the exact observation slice order and normalization rules, see [Physics -> Distribution](../physics/distribution.md#distgridenv-balanced-radial-feeder).

## Reward and CMDP cost

The step reward is the loss-minimization signal:

\[
r_t = -w_{\mathrm{loss}}\, P_{\mathrm{loss}, t}^{\mathrm{MW}}
\]

where \(P_{\mathrm{loss}, t}^{\mathrm{MW}}\) is total feeder active-power loss at step \(t\), and \(w_{\mathrm{loss}}\) is the reward weight (`loss_penalty_weight` in the implementation; see [API -> Distribution](../api/distribution.md)).

The task-level CMDP cost is

\[
\mathbf{c}_t = \left(C_t^{\mathrm{volt}}\right)
\]

where \(C_t^{\mathrm{volt}}\) is the count of buses whose voltage leaves the allowed band `[0.94, 1.06]` p.u. In other words, DSO uses a count-based voltage-safety channel, not a continuous distance-from-limit penalty, as its benchmark CMDP cost.

At the episode level, two different summaries are useful and should not be conflated:

\[
R_{\mathrm{ep}} = \sum_{t=0}^{T-1} r_t
\]

\[
J_{\mathrm{loss}} = \sum_{t=0}^{T-1} P_{\mathrm{loss}, t}^{\mathrm{MW}} \Delta t
\]

- \(R_{\mathrm{ep}}\) is reported as `total_reward` and is the frozen convergence target used by the current benchmark pipeline.
- \(J_{\mathrm{loss}}\) is reported as `total_loss_mwh` and is the more directly interpretable physical episode-loss metric.

This benchmark therefore trains on a reward-equivalent loss objective, but it reports both the trainer-facing return and the physical energy-loss aggregate.

## Baselines

| Name | Description |
| --- | --- |
| `no_control` | all FlexLoad actions = 0; no active scheduling |
| `tou` | time-of-use rule-based baseline: curtail and shift in fixed clock peak windows |
| `droop` | voltage-droop rule-based baseline: deterministic local response from bus voltage relative to a fixed band |

NormScore for DSO is defined on the physical loss metric:

\[
\mathrm{NormScore} = \frac{J_{\mathrm{loss}}^{\mathrm{no\_control}} - J_{\mathrm{loss}}^{\mathrm{algo}}}{J_{\mathrm{loss}}^{\mathrm{no\_control}} - J_{\mathrm{loss}}^{\mathrm{best\ baseline}}}
\]

Here the best frozen non-learning baseline is the stronger of the rule-based baselines, currently `droop` in the summarization pipeline. Because this is a loss-minimization task, lower network loss means higher `NormScore`.

## RL algorithms

| Algo | Preset | Notes |
| --- | --- | --- |
| `ppo` | `dso-nflex` | standard PPO via [Rejax](../training/trainers.md) |
| `sac` | `dso-nflex` | SAC via [Rejax](../training/trainers.md) |
| `saute_ppo` | `dso-nflex-safe` | Sauté PPO using the voltage-safety cost channel |
| `ppo_lagrangian` | `dso-nflex-safe` | PPO-Lagrangian CMDP with zero voltage-violation budget |

Total timesteps `3e6`; 5 seeds; observation normalization on. The Phase-2
backend/device rows are also 5-seed IID reporting rows; DSO is not the
execution-scaling primary task.

## Cross-backend notes

The backend/device comparison keeps the DSO task contract fixed and changes
only the trainer backend or device. The Python bridge trains on the frozen
`train` reset bank and reports the formal `iid` split, matching the current
Phase-2 protocol.

For PPO and SBX-PPO, the current bridge aligns the visible PPO configuration
with `benchmarks/dso/configs/train_ppo.yaml`: `3e6` environment steps,
`hidden_dims=[128, 128]`, Gaussian continuous-action policy, and observation
normalization with reward normalization disabled. The wallclock training-budget
figure uses `learning_curve_eval_return`, which is an in-training monitor curve
rather than the 50-episode formal IID leaderboard metric.

A 2026-04-29 SB3/CUDA seed-0 smoke with the aligned bridge produced a formal
IID `total_reward` near the JAX train-monitor level, but its in-training
monitor curve still converged substantially below the JAX `learning_curve_eval_return`.
This remaining DSO monitor-curve gap should therefore be treated as a
backend/algorithm or monitor-protocol gap, not as evidence that the Python rows
used the default `[64, 64]` network.

## Eval splits

| Split | Description |
| --- | --- |
| `iid` | formal held-out evaluation episodes under the current Ausgrid reset-bank protocol |

!!! caution "DSO currently only runs `iid` officially"
    Executable truth for the current frozen DSO config is `eval_splits: [iid]`.

    - Do **not** treat legacy OOD split names (left over from previous revisions) as configured formal DSO results in this revision.
    - Do **not** use DSO as the execution-scaling primary task — that role belongs to other benchmarks.

    OOD exploration runs are fine, but any numbers outside of `iid` must be clearly labelled as non-frozen results.

## Metrics

It helps to separate four layers of reporting:

| Layer | Key | Description |
| --- | --- | --- |
| Step training signal | `reward` | per-step reward \(r_t = -w_{\mathrm{loss}} P_{\mathrm{loss}, t}^{\mathrm{MW}}\) |
| Step CMDP cost | `voltage_violation` | task-selected voltage-safety channel, equivalent to `cost_voltage_violation` |
| Episode aggregate | `total_reward` | cumulative reward \(R_{\mathrm{ep}}\); current frozen convergence target |
| Episode aggregate | `total_loss_mwh` | physical episode loss \(J_{\mathrm{loss}}\) |
| Episode aggregate | `mean_loss_mw` | average power loss per step |
| Episode aggregate | `total_violations` | total episode voltage-violation count |
| Episode aggregate | `total_voltage_violations` | total count of bus-voltage violations across the episode |
| Episode aggregate | `total_thermal_overloads` | total thermal-overload diagnostic across the episode |
| Episode aggregate | `voltage_violation_count_per_step` | `total_voltage_violations / T` |
| Episode aggregate | `thermal_overload_count_per_step` | `total_thermal_overloads / T` |
| Episode aggregate | `total_curtailed_mwh` | total curtailed energy |
| Episode aggregate | `total_shifted_mwh` | total deferred energy |
| Episode aggregate | `served_flex_ratio` | released deferred energy / deferred energy |
| Relative evaluation summary | `network_loss_reduction_pct` | loss reduction vs `no_control` |
| Relative evaluation summary | `peak_shaving_pct` | peak reduction vs `no_control` |
| Relative evaluation summary | `NormScore` | normalized performance against frozen non-learning baselines |

The key point is that `total_reward` and `total_loss_mwh` are related but not interchangeable: one is the trainer-facing return, the other is the directly interpretable physical aggregate.

## Quick start

```bash
python benchmarks/dso/run.py baseline --seeds 0,1,2,3,4
python benchmarks/dso/run.py train --algo ppo --seed 0
python benchmarks/dso/run.py eval --run-id <run_id> --split iid
python benchmarks/dso/run.py summarize
python benchmarks/dso/run.py plots
python benchmarks/dso/phase2_analysis.py --seeds 0,1,2,3,4
```

## Output files

```text
benchmarks/dso/results/
  manifest.json
  runs/
  summary/latest.json
  phase2_backend_summary.json
  figures/
    normscore_bars.png
    learning_curves.png
    loss_reduction.png
    load_profiles.png
    phase2_backend_compare.png
```

## Cross references

- [Physics -> Distribution](../physics/distribution.md)
- [Physics -> Resources](../physics/resources.md#flexible-load-flexloadenv)
- [API -> Distribution](../api/distribution.md)
- [Training -> Trainers](../training/trainers.md)
- [Training -> Presets](../training/presets.md)
