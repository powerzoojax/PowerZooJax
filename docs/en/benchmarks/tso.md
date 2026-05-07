# TSO - Security-Constrained Unit Commitment

The TSO benchmark is centralized [unit commitment and economic dispatch](../concepts/power-systems-primer.md#generation-dispatch-and-opf) on the IEEE 118-bus system. At each step, the agent decides which generators should be on and how dispatch should be biased, while the solver enforces physical feasibility.

This page is the TSO task guide: it defines the task, the command flow, required data, metrics, and generated outputs. Shared workflow vocabulary, including what counts as the current [campaign](../glossary.md#campaign), is in the [Benchmark workflow glossary](../glossary.md).

## At A Glance

- Physical env: `UnitCommitmentEnv` on IEEE `case118`.
- Benchmark task: centralized security-constrained unit commitment and dispatch.
- What is actually trained: PPO for the unconstrained path, or PPO-Lagrangian for the safe CMDP path.
- Primary leaderboard quantity: `total_operating_cost` on the `iid` split.
- Safety gate: leaderboard-eligible runs must satisfy zero thermal-overload and zero reserve-shortfall thresholds.

Read this page in this order if you are new to the task: first the plain-language summary above, then the MDP / CMDP table, then the reward-and-cost section, and only then the quick-start commands. That order separates the physical problem, the benchmark contract, and the runnable workflow.

## MDP / CMDP specification {#mdp-cmdp-spec}

| Field | Value |
| --- | --- |
| [MDP class](#tso-rl-algos) | MDP (`tso-uc`) / CMDP (`tso-scuc-safe`, a constrained Markov decision process) |
| [Agents](#tso-benchmark-params) | 1 (centralized) |
| [State \(\mathcal{S}\)](#tso-physics) | unit on/off status, time-in-state, last dispatch, line flows, current load, reserve ratio, time-of-day phase, short-horizon future total-load preview |
| [Observation \(\mathcal{O}\)](#tso-action-obs) | `Box(4 * 54 + 186 + 4 + 4) = Box(410)` (legacy UC state plus a 4-step future total-load forecast; field semantics in [Physics → Transmission](../physics/transmission.md#observation)) |
| [Action \(\mathcal{A}\)](#tso-action-obs) | `Box(108) = [commitment_intent (54) | dispatch_preference (54)]`, range `[-1, 1]` |
| [Transition \(\mathcal{P}\)](#tso-physics) | min-up / min-down masking on commit, then [DC OPF](../concepts/power-systems-primer.md#generation-dispatch-and-opf) dispatch with ramp limits |
| [Reward \(r_t\)](#tso-reward-cost) | \(r_t = -\lambda_{\mathrm{reward}} C^{\mathrm{op}}_t\), where \(C^{\mathrm{op}}_t\) is step operating cost |
| [Cost \(\mathbf{c}_t\)](#tso-reward-cost) | \(\mathbf{c}_t = (C^{\mathrm{th}}_t, C^{\mathrm{res}}_t)\): thermal-overload and reserve-shortfall channels for the safe variant |
| [Threshold \(d\)](#tso-reward-cost) | zero thresholds for thermal overload and reserve shortfall in `tso-scuc-safe` |
| [Discount \(\gamma\)](#tso-rl-algos) | `0.995` |
| [Horizon \(T\)](#tso-benchmark-params) | 48 steps x 30 min = 24 h |
| [Initial \(\mu_0\)](#tso-benchmark-params) | data-driven episode sampled from GB demand + generation profiles |

## Underlying physics {#tso-physics}

The env is `UnitCommitmentEnv` (see [Physics → Transmission](../physics/transmission.md#unitcommitmentenv-scuc-for-the-tso-task)). In power-system terms, this is a [security-constrained unit commitment (SCUC)](../concepts/power-systems-primer.md#generation-dispatch-and-opf) task: a multi-step commitment problem with network security and reserve constraints.

Key features:

- a continuous commitment signal that is thresholded to the binary on/off decision inside the env,
- min-up / min-down masking enforces commitment feasibility inside `step`,
- ramp-bounded dispatch enforced inside the [DC OPF](../concepts/power-systems-primer.md#generation-dispatch-and-opf) solver,
- one-time startup cost and per-step no-load cost,
- system-wide [reserve](../concepts/power-systems-primer.md#generation-dispatch-and-opf) margin requirement (optional).

## Benchmark task parameters {#tso-benchmark-params}

| Parameter | Value |
| --- | --- |
| Case | IEEE case118 |
| Generators | 54 units |
| Buses | 118 |
| Lines | 186 |
| Episode length | 48 steps x 30 min = 24 h |
| Agents | 1 (centralized) |
| Action space | `Box(108) = [commitment_intent (54) | dispatch_preference (54)]` |
| Observation | `Box(4 * 54 + 186 + 4 + 4) = Box(410)`; field semantics in [Physics → Transmission](../physics/transmission.md#observation) |
| Data source | GB demand + gen-by-type (real data for benchmark results) |
| Train split | 2025-04-01 to 2025-12-31 |
| IID split | 2026-01-01 to 2026-03-31 |

Synthetic profiles may still exist for dev or CI, but benchmark reporting should use the real GB data path selected by the task config.

Training resets sample fresh 48-step windows from the full GB training split.
Formal evaluation keeps the frozen fixed-window protocol.

The task config also carries machine-readable benchmark protocol metadata:

- current [campaign](../glossary.md#campaign) seed budget: 5,
- [submission-grade minimum](../glossary.md#submission-grade-minimum): 5 seeds,
- required statistics: mean, std, IQM ([interquartile mean](../glossary.md#normalized-score-normscore)), and 95% [bootstrap CI](../glossary.md#bootstrap-ci),
- primary leaderboard: `iid` split ranked by `total_operating_cost`,
- unsafe policies are reported but not [leaderboard-eligible](../glossary.md#leaderboard-quantity).

The current 5-seed evidence is complete, but the hard safety gate is negative:
no primary-split row has both `reserve_shortfall_rate == 0.0` and
`thermal_violation_rate == 0.0`. This is reported as a benchmark-hardness
result rather than a request to relax the zero-violation thresholds.

## Action and observation {#tso-action-obs}

The action has two parts for each generator: whether to push the unit toward ON, and what dispatch level to prefer if it is available. The observation combines generator status, recent dispatch, network loading, current system demand, reserve information, time-of-day features, and a short future total-load forecast.

- entries `0..53`: commitment intent (`commit_intent`, paper notation \(\mathbf{u}^{\mathrm{cmd}}_{t}\)). After thresholding, `> 0` means request unit ON.
- entries `54..107`: dispatch preference (`dispatch_preference`, paper notation \(\mathbf{P}^{\mathrm{pref}}_{t}\)). Denormalized to `[ramp_p_min, ramp_p_max]` after the commit mask is applied. This is a per-unit preferred feasible dispatch target, not the final dispatch commanded to the grid.

Observation: `[unit_status (54) | time_in_state_norm (54) | last_dispatch_norm (54) | unit_cost_b_norm (54) | line_flow_norm (186) | load_norm | reserve_ratio | sin(t) | cos(t) | future_total_load_norm(t+1:t+4)]`.

For a field-by-field explanation of each observation block, see [Physics → Transmission](../physics/transmission.md#observation).

## Reward and CMDP cost {#tso-reward-cost}

\[
r_t = -\lambda_{\mathrm{reward}}\, C^{\mathrm{op}}_t
\]

\[
C^{\mathrm{op}}_t =
C^{\mathrm{gen}}_t +
C^{\mathrm{start}}_t +
C^{\mathrm{no\mbox{-}load}}_t
\]

\[
\mathbf{c}_t = \left(C^{\mathrm{th}}_t, C^{\mathrm{res}}_t\right)
\]

Here \(C^{\mathrm{op}}_t\) is the realized step operating cost of the SCUC dispatch, split into generation cost, startup cost, and no-load cost. In the implementation these correspond to `gen_cost`, `startup_cost`, and `no_load_cost`, while \(\lambda_{\mathrm{reward}}\) corresponds to `reward_scale`.

For the safe variant, \(C^{\mathrm{th}}_t\) is the thermal-overload magnitude and \(C^{\mathrm{res}}_t\) is the reserve-shortfall magnitude. The fixed CMDP channel names are `("thermal_overload", "reserve_shortfall")`.

CMDP cost channel definitions:

| Symbol | Constraint name | Info key | Meaning |
| --- | --- | --- | --- |
| \(C^{\mathrm{th}}_t\) | `thermal_overload` | `cost_thermal_overload` | Weighted thermal-overload magnitude from line-flow safety checks after dispatch. |
| \(C^{\mathrm{res}}_t\) | `reserve_shortfall` | `cost_reserve_shortfall`, `reserve_shortfall` | Missing committed capacity headroom relative to load plus reserve margin; zero when reserve is disabled or sufficient. |

The underlying `UnitCommitmentEnv` exposes a third channel `min_updown` (`cost_min_updown`) for fixed-shape compatibility with downstream wrappers; it is identically zero because min-up / min-down constraints are enforced by a hard mask inside `step` (see [Physics → Transmission](../physics/transmission.md#unitcommitmentenv-scuc-for-the-tso-task)). The TSO benchmark and the paper Eq. for `\mathbf{c}_t` (Appendix E.2) both use only the first two channels.

The primary benchmark objective is the episode aggregate

\[
J^{\mathrm{op}} = \sum_{t=0}^{T-1} C^{\mathrm{op}}_t
\]

reported as `total_operating_cost`. This episode metric is the quantity used for the leaderboard and is distinct from the per-step reward used during training.

## Baselines

| Name | Description |
| --- | --- |
| `all_on` | every unit always committed; pure dispatch through OPF. Worst-case cost upper bound. |
| `merit_order` | [priority-list (merit-order)](../concepts/power-systems-primer.md#generation-dispatch-and-opf) commitment until load + reserve is covered, then OPF; a common rule-based SCUC approximation and a strong **cost reference** (often a loose lower bound in simplified settings). |

Both baselines are deterministic **non-learning** rollouts (no training) and run on CPU in seconds. They produce `RunRecord` entries with `algo="all_on"` or `algo="merit_order"`, sharing the same `manifest.json` run index as the RL runs.

For score normalization, TSO reports [NormScore](../glossary.md#normalized-score-normscore):

\[
\text{NormScore} = \frac{\text{cost}_{\text{all\_on}} - \text{cost}_{\text{algo}}}{\text{cost}_{\text{all\_on}} - \text{cost}_{\text{merit\_order}}}
\]

So `all_on` scores 0, `merit_order` scores 1, and a strong RL policy can score above 1.

## RL algorithms {#tso-rl-algos}

| Algo | Preset | Notes |
| --- | --- | --- |
| `ppo` | `tso-uc` | Unconstrained baseline |
| `ppo_lagrangian` | `tso-scuc-safe` | PPO-Lagrangian CMDP baseline with thermal-overload and reserve-shortfall costs |
| `ppo_penalty_l10` | `tso-scuc-safe` | Appendix penalty PPO ablation: fixed \(\lambda_{\mathrm{pen}}=10\) (effective coeff 1e-3; under-penalised) |
| `ppo_penalty_l100` | `tso-scuc-safe` | Appendix penalty PPO ablation: fixed \(\lambda_{\mathrm{pen}}=100\) (effective coeff 1e-2; comparable) |
| `ppo_penalty_l1000` | `tso-scuc-safe` | Appendix penalty PPO ablation: fixed \(\lambda_{\mathrm{pen}}=1000\) (effective coeff 1e-1; over-penalised) |

The current paper-facing TSO campaign uses `ppo` and `ppo_lagrangian` as the
primary learned rows. Historical Sauté / penalty sweeps may be discussed as
appendix analysis, but they are not part of the current primary campaign
leaderboard.

For the mandatory phase-2 backend/device comparison, the comparable path is
the canonical unconstrained PPO config from `benchmarks/tso/configs/train_ppo.yaml`
only: `20M` timesteps, `n_steps=48`, `num_envs=256`. Those curves are
train-monitor curves on the `train` split. Formal IID / OOD eval remains a
separate protocol and should not be mixed into the phase-2 learning-curve panel.

### Known backend gap {#tso-known-backend-gap}

TSO is not used as evidence that all PPO backends converge to the same final
return. After aligning reward scale, `total_timesteps=20M`, network width
`[256, 256]`, PPO clip range, horizon, and deterministic monitor evaluation,
the ReJAX JAX/GPU monitor curve still finishes below the Python PPO monitors
on this task. In the 2026-04-29 rerun, five JAX/GPU PPO seeds had mean final
train-monitor return about `-179.7` (best per-seed maxima mean about `-163.0`),
whereas the current seed-0 Python monitors are about `-97.1` for SB3/CUDA and
`-130.8` for SBX/CUDA.

This is treated as an algorithm/backend gap for the train-monitor diagnostic,
not as proof that TSO safe RL is solved. The main wall-clock speed figure uses
same-budget completion time; paper-facing TSO quality claims should use the
formal IID / OOD cost-safety summaries and the hard-safety frontier.

### Penalty PPO ablation {#tso-penalty-ablation}

`ppo_penalty_l*` applies a fixed reward-penalty shaping before training:

$$
r'_t = r_t - \lambda_{\mathrm{pen}} \, \lambda_{\mathrm{reward}} \sum_i c_{i,t}
$$

Here \(r_t\) is the base task reward, \(\lambda_{\mathrm{reward}}\) is the reward-scaling factor, \(\lambda_{\mathrm{pen}}\) is the fixed penalty coefficient, and \(c_{i,t}\) are the physical CMDP cost channels at step \(t\). In the implementation these correspond to the env reward, `reward_scale = 1e-4` (from `task.yaml`), the fixed penalty weight in the penalty ablation config, and the selected CMDP costs from the same `UnitCommitmentEnv`.

Implemented by `PenaltyRewardWrapper` wrapping the same `UnitCommitmentEnv`; the underlying trainer is unchanged standard PPO via [Rejax](../training/trainers.md), the single-agent training backend used by the non-Lagrangian PPO path. Three \(\lambda_{\mathrm{pen}}\) values span the under/comparable/over-penalised regimes:

| Key | \(\lambda_{\mathrm{pen}}\) | Effective coeff | Regime |
| --- | --- | --- | --- |
| `ppo_penalty_l10` | 10 | 1e-3 | Max penalty about 0.5 while the step reward is about -0.5; the safety signal is drowned out |
| `ppo_penalty_l100` | 100 | 1e-2 | Max penalty about 5, on the same order as the step reward; comparable scale |
| `ppo_penalty_l1000` | 1000 | 1e-1 | Max penalty about 50, much larger than the step reward; the cost signal dominates |

All three are expected to be outperformed by `ppo_lagrangian` (adaptive dual) on the safety-gated leaderboard, demonstrating that a static \(\lambda_{\mathrm{pen}}\) cannot reliably satisfy zero-threshold constraints. These runs belong in the TSO appendix and do not replace `ppo` / `ppo_lagrangian` as the primary results.

## Eval splits

| Split | Description |
| --- | --- |
| `train` | same window as training |
| `iid` | held-out months in the same regime |
| `load_stress` | higher-demand [OOD](../glossary.md#split) split |
| `line_tightening` | lower-thermal-rating [OOD](../glossary.md#split) split |

The two OOD ([out-of-distribution](../glossary.md#split)) splits stress different aspects of the task: one raises demand pressure, the other raises congestion pressure.

Capability taxonomy:

- `iid` tests routine day-ahead SCUC competence on held-out same-regime demand windows.
- `load_stress` tests reserve adequacy and cost control under a demand surge.
- `line_tightening` tests congestion-aware commitment and redispatch when transfer headroom shrinks.

Interpretation:

- strong `iid`, weak `load_stress` suggests brittle reserve margins or overly aggressive commitment trimming;
- strong `iid`, weak `line_tightening` suggests the policy relies on uncongested merit-order patterns and lacks transmission-security robustness;
- weak `iid` already means the method is not competitive on standard SCUC before any deliberate shift.

## Metrics

| Key | Description |
| --- | --- |
| `total_operating_cost` | episode aggregate \(J^{\mathrm{op}} = \sum_t C^{\mathrm{op}}_t\) |
| `feasibility_rate` | fraction of steps with neither thermal overload nor reserve shortfall |
| `thermal_violation_rate` | evaluation summary: fraction of steps with \(C^{\mathrm{th}}_t > 0\) |
| `reserve_shortfall_rate` | evaluation summary: fraction of steps with \(C^{\mathrm{res}}_t > 0\) |
| `mean_thermal_cost` | mean per-step `cost_thermal_overload` |
| `total_thermal_cost` | episode sum of `cost_thermal_overload` |
| `mean_reserve_shortfall` | mean per-step reserve shortfall |
| `total_reserve_shortfall` | episode sum of reserve shortfall |
| `commitment_switching_frequency` | episode-level switching summary (off-to-on transitions per episode) |
| `norm_score` | normalized episode-cost score |
| `ood_degradation` | `NormScore(IID) - NormScore(load_stress)` |

Reward-hacking guardrail:

- the primary leaderboard is not cost-only;
- a primary-split row must also satisfy the declared safety thresholds (`reserve_shortfall_rate <= 0`, `thermal_violation_rate <= 0`) to be leaderboard-eligible.

## Quick start

```bash
python benchmarks/tso/run.py baseline --seeds 0,1,2,3,4

for seed in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=<gpu_id> python benchmarks/tso/run.py train \
        --algo ppo --seed $seed &
done
wait

for split in train iid load_stress line_tightening; do
    python benchmarks/tso/run.py eval --run-id <ppo_run_id> --split $split
done

python benchmarks/tso/run.py summarize
python benchmarks/tso/run.py plots
```

For the full pipeline:

```bash
python benchmarks/tso/run_all.py
```

For the current 5-seed [campaign](../glossary.md#campaign):

```bash
python benchmarks/tso/run_all.py --seeds 0,1,2,3,4
```

!!! warning "Do not mix campaigns in the same table"
    Records from different [campaigns](../glossary.md#campaign) come from different reset banks, different OOD protocols, and different code revisions. Do **not** combine them in one results table.

    Always add a `campaign_start_iso` filter so a single table represents a single campaign — otherwise the leaderboard numbers look comparable but are not.

## Output files

```text
benchmarks/tso/results/
  manifest.json
  runs/
  artifacts/
  summary/latest.json         <- aggregated metrics plus protocol_status,
                                 leaderboard_primary_split, split_taxonomy
  figures/
    normscore_bars.{pdf,png}
    gantt_commitment.{pdf,png}
    cost_decomposition.{pdf,png}
    learning_curves.{pdf,png}
```

## Cross references

- [Physics → Transmission](../physics/transmission.md#unitcommitmentenv-scuc-for-the-tso-task)
- [API → Unit commitment](../api/grid-uc.md)
- [Training → Presets](../training/presets.md)
