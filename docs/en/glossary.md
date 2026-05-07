# Benchmark workflow glossary

This page defines benchmark terms used across task pages, result tables, and experiment configs in PowerZooJax.

For power-system concepts such as OPF, SOC, and PTDF, see [Power systems primer](concepts/power-systems-primer.md).

---

## Task

A benchmark task is a clearly defined power-system control problem with fixed:

- **Case**: the power-grid topology and parameters
- **Agents**: the number and type of decision-makers
- **Horizon**: the episode length in timesteps
- **RL paradigm**: single-agent safe RL, cooperative MARL, competitive MARL, and so on
- **Reward and cost**: the objective and any safety-violation channels
- **Splits**: the train / iid / OOD data settings used for evaluation

PowerZooJax currently defines five core benchmark tasks: TSO, DSO, DERs, GenCos, and DC Microgrid.

Each task is defined in `benchmarks/<task>/configs/task.yaml` and must pass `seed0_readiness --enforce` before the full multi-seed runs begin.

## Campaign

A campaign is one coherent benchmark run window for a task: a specific round of baselines, training runs, eval runs, summaries, and figures that belong together under one frozen task setup.

In practice, a new campaign usually starts when you intentionally reset the benchmark workflow and declare that older manifest rows should no longer count as the current formal evidence. That is why readiness checks are often scoped to the current campaign rather than to the entire historical manifest.

You can think of a campaign as answering: "which set of runs belongs to the current benchmark round?" Terms such as [seed-0 readiness](#seed-0-readiness), [campaign seed budget](#campaign-seed-budget), and submission-grade reruns all refer to this scoped benchmark round.

## Seed-0 readiness

`seed0_readiness` is the benchmark gate that checks whether a task has a complete, valid seed-0 reference chain before formal multi-seed runs begin.

In practice, this means the benchmark expects the seed-0 path to have completed the required baseline, train, eval, and summary steps for the current campaign scope. It is a workflow-validity check, not a claim that the method is already good enough scientifically.

## Multi-seed benchmark run

A multi-seed benchmark run is a formal evaluation of one task across several seeds and all required splits. It starts only after the seed-0 reference run is complete, and its results are recorded in `benchmarks/<task>/results/manifest.json`.

## Campaign seed budget

The campaign seed budget is the number of random seeds used in the currently active benchmark campaign for that task.

It is the practical budget for the run that is happening now, and it can be smaller than the final paper-grade budget. For example, a task may temporarily run a 3-seed campaign for active iteration, then later rerun a 5-seed submission-grade campaign.

## Submission-grade minimum

The submission-grade minimum is the minimum seed count expected for the final paper-facing benchmark table or figure.

This is stricter than the active campaign seed budget. It tells readers that a task may be temporarily reported with fewer seeds during iteration, but that the final benchmark claim should use at least this many seeds.

## Run Record

A run record is the saved result of one training run, baseline run, or evaluation run. It includes:

- Basic info: task, algorithm, seed, split, and run ID
- Metrics: episode reward, cost, constraint violations, normalized score
- Status and completion time

All run records are collected into task results and used to generate summaries and figures.

## Split

A split is a named data setting used for training or evaluation:

- **train**: the full training distribution
- **iid**: a held-out test set from the same broad distribution
- **OOD (Out-of-Distribution)**: an intentional distribution shift. Canonical paper-side names per task include: TSO `load_stress` and `line_tightening`; DERs `voltage_tightening`, `pv shift`, `load_stress`; GenCos `demand shift`, `renewable shock`; DCMG `cooling_stress`, `renewable_drought`, `workload_swap`, `workload_shock`, `dg_derating`, `sla_tighten`

Every agent trained on the `train` split should be evaluated on all task splits.

## Baseline

A baseline is a non-learning reference policy used to anchor performance comparisons:

- **no_control**: passive operation with zero action
- **rule-based**: a fixed, deterministic if–then or schedule, such as TOU curtailment, local **volt–var** / droop control, or a **priority-list (merit-order)** commitment
- **heuristic (approximate solver)**: an approximate dispatch or market solve that trades global optimality for speed—used in code and docs for things like the piecewise economic-dispatch path in [Market](api/market.md) clearing, not for simple TOU/droop schedules
- **simple greedy**: one-step optimization without long-term planning

Many baselines are deterministic and reproducible, and they provide the reference IQM values used in score normalization.

## Training run

A training run learns a policy on the training data distribution:

- It starts from a fixed random seed
- It collects experience from multiple parallel environments
- It updates the policy with an RL algorithm such as PPO, IPPO, or a safe-RL variant
- It saves the trained policy for later evaluation

Training configs are fixed and should stay consistent across runs of the same task.

## Evaluation run

An evaluation run measures how well a trained policy or baseline performs on one split:

- It loads a trained policy or creates a baseline
- It runs episodes with that split's data distribution
- It records reward, cost, and constraint metrics
- It saves the result in the standard record format

Evaluation should be reproducible with the same seed.

## Bootstrap CI

Bootstrap CI means bootstrap confidence interval, usually reported as something like a 95% CI.

It is an uncertainty interval estimated by repeatedly resampling the available runs or episodes with replacement. In benchmark reporting, it helps show how stable an aggregate statistic such as mean or IQM is across seeds.

## Primary metric

The primary metric is the task's main benchmark-facing scalar written into the task config and reporting pipeline. It is the metric the benchmark treats as the default quantity to summarize and compare for that task.

It is often, but not always, the same quantity that appears most prominently in tables. Examples:

- TSO: `total_operating_cost`
- DERs: `mean_p_loss_mw`
- GenCos: `total_profit`
- DC Microgrid: `episode_reward`

The primary metric is a reporting contract term, not a statement about how the trainer updates its policy.

## Convergence target

The convergence target is the scalar quantity currently used by the training-and-summary pipeline to decide whether a run has reached its benchmark target.

This is a workflow concept, not necessarily the most physically interpretable quantity. In some tasks it matches the main physical benchmark quantity; in others it is a trainer-facing return-like quantity instead.

Examples:

- In DSO, the current convergence target is `total_reward`, while the more directly interpretable physical aggregate is `total_loss_mwh`.
- In DC Microgrid, the current convergence target is `episode_reward`, which is reward-shaped rather than a pure physical cost total.

## Leaderboard quantity

The leaderboard quantity is the scalar used to rank methods on the benchmark's main split once any safety or audit gates are applied.

This term matters because a method can:

- optimize one per-step reward during training,
- be monitored by a different convergence target during benchmarking,
- and still be ranked by a separate episode-level quantity on the leaderboard.

For example:

- TSO trains on per-step reward but is ranked by episode `total_operating_cost` on the primary split.
- DERs currently treats `mean_p_loss_mw` as the main leaderboard quantity even though training uses reward-shaped IPPO.

## Normalized Score (NormScore)

NormScore turns raw task metrics into a dimensionless comparison scale:

$$
\text{NormScore} = \frac{\text{agent performance} - \text{baseline floor}}{\text{baseline ceiling} - \text{baseline floor}}
$$

- **Baseline floor**: IQM of the weakest baseline, such as `no_control`
- **Baseline ceiling**: IQM of the strongest baseline
- **Agent performance**: IQM over seeds and split episodes

NormScore above `1.0` means the method beats the strongest baseline. NormScore below `0.0` means it is worse than the weakest baseline.

NormScore is a diagnostic relative metric, not the default headline ranking metric. Summaries expose `norm_score_status`, anchor values, and anchor-gap warnings; only rows with `norm_score_status=ok` are eligible for NormScore-based comparisons.

In benchmark pages, `NormScore` usually names the concept, while `norm_score` is often the serialized metric key written into `manifest.json`, summaries, or result tables.

## Manifest

The manifest is the master log for one task. Each training, baseline, and evaluation run writes one record, and all records are collected into a single file. That file is the source of truth for summaries and figures.

## Summary and figures

- **Summary**: aggregated statistics across seeds, splits, and algorithms
- **Figures**: plots such as reward curves or constraint-satisfaction heatmaps

These are generated automatically from the manifest and should not be edited by hand.

## Cross-backend comparability

Results from different implementations, such as PowerZoo and PowerZooJax, are only fair to compare when all of the following stay the same:

- task definition
- data sources
- random seeds
- reward and cost formulas
- safety constraints
- training duration

If any of these differ, the numbers should not be placed in one direct comparison table.

## JAX execution model

PowerZooJax environments are designed to run efficiently on GPU with immutable state and pure functions. This supports fast training and evaluation with many parallel environments.

## Seed

A seed is a fixed random number used to make runs reproducible. It affects policy initialization, environment randomness, and data-split shuffling. Benchmark reports use multiple seeds and aggregate the results.
