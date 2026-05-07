# TSO Benchmark — Security-Constrained Unit Commitment (SCUC)

Reproducible experiment pipeline for the TSO task: centralized unit commitment
and economic dispatch on the IEEE 118-bus system with 54 generators.

## Task Definition

| Parameter         | Value                        |
|-------------------|------------------------------|
| Case              | IEEE case118                 |
| Generators        | 54 units                     |
| Buses             | 118                          |
| Lines             | 186                          |
| Episode length    | 48 steps × 30 min = 24 h    |
| Agents            | 1 (centralized)              |
| Action space      | Box(108): [commit(54); dispatch(54)] |
| Observation space | Box(410): legacy UC state + 4-step future total-load forecast |
| Data source       | GB demand + gen-by-type      |
| Train split       | 2025-04-01 → 2025-12-31      |
| IID split         | 2026-01-01 → 2026-03-31      |

## Experiment Matrix

| Algorithm       | Train | IID | Load Stress | Line Tightening |
|-----------------|-------|-----|-------------|-----------------|
| all_on          | ✓     | ✓   | ✓           | ✓               |
| merit_order     | ✓     | ✓   | ✓           | ✓               |
| PPO (relax)     | ✓     | ✓   | ✓           | ✓               |
| PPO-Lagrangian  | ✓     | ✓   | ✓           | ✓               |

Each cell = mean ± std across 5 seeds (seed 0, 1, 2, 3, 4).

Current campaign note: seed and hypothesis-test evidence are complete for the
current 5-seed `jax_rejax+gpu` campaign, but the strict safety gate is
negative. `summary/latest.json::protocol_status.current_campaign_submission_ready`
is `false` because no primary-split row satisfies both
`reserve_shortfall_rate == 0.0` and `thermal_violation_rate == 0.0`.

## What This Benchmark Measures

- `iid`: routine SCUC competence on held-out same-regime GB demand windows.
- `load_stress`: reserve adequacy and cost control when demand surges by 15%.
- `line_tightening`: congestion-aware commitment and redispatch when thermal headroom shrinks by 15%.

Interpretation guide:

- strong `iid` but weak `load_stress` usually means the policy trims reserve too aggressively.
- strong `iid` but weak `line_tightening` usually means the policy depends on uncongested merit-order patterns and lacks transmission-security robustness.
- low `iid` already means the method is not competitive on standard SCUC, even before deliberate OOD shift.

## Metrics

| Metric                        | Description                                               |
|-------------------------------|-----------------------------------------------------------|
| `total_operating_cost`        | gen\_cost + startup\_cost + no\_load\_cost (primary)      |
| `feasibility_rate`            | Fraction of steps with no thermal/reserve violations      |
| `thermal_violation_rate`      | Fraction of steps with thermal overloads                  |
| `reserve_shortfall_rate`      | Fraction of steps with reserve shortfall                  |
| `commitment_switching_frequency` | off→on transitions per episode                         |
| `norm_score`                  | (all\_on\_cost − algo\_cost) / (all\_on\_cost − merit\_order\_cost) |
| `ood_degradation`             | NormScore(IID) − NormScore(load\_stress)                  |

Leaderboard rule:

- the primary leaderboard is the `iid` split sorted by `total_operating_cost`;
- rows violating declared safety thresholds are still reported, but they are not leaderboard-eligible.

## Quick Start

```bash
# Step 1: Run all baselines.
# Default target: 40 baseline records = 2 non-learning baseline algos × 4 splits × 5 seeds.
python benchmarks/tso/run.py baseline --seeds 0,1,2,3,4

# Step 2: Train RL agents.
# Default target: 10 training records = 2 RL algos × 5 seeds.
for seed in 0 1 2 3 4; do
    python benchmarks/tso/run.py train --algo ppo --seed $seed
    python benchmarks/tso/run.py train --algo ppo_lagrangian --seed $seed
done

# Step 3: Evaluate trained agents on all splits.
# Default target: 40 evaluation records = 2 RL algos × 4 splits × 5 seeds.
# Replace <run-id> with the run IDs printed by the train step.
python benchmarks/tso/run.py eval --run-id <ppo_run_id> --split train
python benchmarks/tso/run.py eval --run-id <ppo_run_id> --split iid
python benchmarks/tso/run.py eval --run-id <ppo_run_id> --split load_stress
python benchmarks/tso/run.py eval --run-id <ppo_run_id> --split line_tightening

# Step 4: Summarize all results
python benchmarks/tso/run.py summarize

# Step 5: Generate paper figures
python benchmarks/tso/run.py plots
```

Or run everything at once:
```bash
python benchmarks/tso/run_all.py
```

For the current 5-seed campaign:
```bash
python benchmarks/tso/run_all.py --seeds 0,1,2,3,4
```

## Expected Record Counts

| Category              | Count |
|-----------------------|-------|
| Non-learning baseline records | 40 = 2 algos × 4 splits × 5 seeds |
| RL training records | 10 = 2 algos × 5 seeds |
| RL evaluation records | 40 = 2 algos × 4 splits × 5 seeds |
| **Current Phase-1 campaign target** | **90 records** |

The target count assumes every training run completes and every evaluation cell
is written. Failed training runs or skipped evaluation cells reduce the logical
record count in the manifest.

## Output Files

```
results/
  manifest.json          <- auto-maintained run index; current Phase-1 target is 90 records
  runs/                  <- individual JSON records (gitignored for large files)
  artifacts/             <- policy params .pkl + learning curves .npy (gitignored)
  summary/
    latest.json          <- aggregated summary with NormScore, ood_degradation,
                             protocol_status, leaderboard_primary_split, split_taxonomy
  figures/
    normscore_bars.pdf   <- NormScore grouped bar chart
    gantt_commitment.pdf <- Unit on/off Gantt timeline
    cost_decomposition.pdf <- Gen/startup/no-load stacked bars
    learning_curves.pdf  <- RL learning curves
```

## OOD Splits

- **Load Stress**: IID demand profiles × 1.15 — simulates peak load events.
- **Line Tightening**: Line thermal ratings × 0.85 — simulates congestion increase.

## Notes

- Actions use continuous relaxation: `commit_intent > 0` → unit ON.
  No discrete/hybrid policy head required.
- Observation includes a 4-step future total-load forecast, appended after the
  legacy `[load_norm | reserve_ratio | sin(t) | cos(t)]` block.
- Training resets sample fresh 48-step windows from the full GB training split;
  formal eval keeps the frozen fixed-window protocol.
- The phase-2 backend/device matrix compares the canonical PPO training path
  only (`benchmarks/tso/configs/train_ppo.yaml`: `20M` timesteps, `n_steps=48`,
  `num_envs=256`). Those cross-backend curves are train-monitor curves on the
  `train` split; formal IID eval is separate.
- The phase-2 backend/device matrix is a PPO backend comparison only. The
  audit command is `python benchmarks/tso/analysis/phase2_backend_audit.py --enforce`;
  it checks the four canonical cells, train-window contract, checkpoint
  density, curve payload, and checkpoint artifacts.
- PPO uses `enable_reserve=False`; PPO-Lagrangian uses `enable_reserve=True`
  with reserve shortfall in the CMDP cost channel.
- Physical constraints (min-up/down, ramp, startup/no-load cost) are always
  enforced in the environment regardless of the RL algo.
