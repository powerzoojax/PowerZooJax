# GenCos — Market Bidding Benchmark

Competitive MARL benchmark on `case5`: five generation companies bid markup
curves into a rolling market with network-constrained clearing.

## Source Of Truth

- Static task definition: `configs/task.yaml`
- Frozen cross-backend PPO config: `configs/train_ppo.yaml`
- Frozen convergence metadata: `configs/provenance.json`
- Full pipeline entry point: `run_all.py`

Current evidence status: Phase 1 has complete 5-seed `jax_rejax+gpu`
evidence with paired primary-split tests. IPPO significantly beats `truthful`
and `uniform_mid`, but remains below the strong `max_markup` heuristic; present
that as a benchmark-hardness result, not as "learning solves GenCos."

Phase 2 has the mandatory seed-0 backend/device matrix. Python backend rows
use PowerZoo frozen self-play IL: round 1 starts without frozen policies
because none exist yet, while later rounds use frozen opponent policies. These
are not random-opponent baselines.

## Frozen Task Summary

- **Case**: `case5`
- **Agents**: `5`
- **Episode**: `48 x 30min`
- **Algorithm**: `ippo`
- **Baselines**: `truthful`, `uniform_mid`, `max_markup`
- **Eval splits**: `train`, `iid`, `demand_shift`, `renewable_shock`
- **Data**: GB demand + generation-by-type traces
- **Primary metric**: `total_profit` (`higher_is_better`)

## How To Run

```bash
# Baselines only
python benchmarks/gencos/run_all.py --only baselines --seeds 0 1 2 3 4

# Train IPPO
python benchmarks/gencos/run_all.py --only train --algos ippo --seeds 0 1 2 3 4

# Evaluate completed training runs
python benchmarks/gencos/run_all.py --only eval

# Summaries and plots
python benchmarks/gencos/run_all.py --only summarize
python benchmarks/gencos/run_all.py --only plots
```

`run_all.py` is the supported top-level orchestration entry point for this
task. There is no `run.py` wrapper here.

## Fairness Notes

- Formal runs must use the same case, split set, seed set, and timestep budget across backends.
- Cross-backend comparison uses frozen-self-play IL on the PowerZoo side; do not regress to random-opponent baselines.
- Cross-backend GenCos runs must use the real GB demand windows on both backends; synthetic fallback is not a valid benchmark result.
- Market metrics such as `total_profit`, `mean_lmp`, and `market_HHI` must come from real rollouts, never fabricated placeholders.

## Main Code Paths

- `train.py`
- `eval.py`
- `baselines.py`
- `summarize.py`
- `plots.py`
