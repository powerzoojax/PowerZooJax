# DERs — Voltage Regulation Benchmark

Cooperative MARL benchmark on `case141`: 12 heterogeneous agents
(`4 battery + 4 PV + 4 flexload`) share a team objective while keeping
distribution voltages feasible.

## Source Of Truth

- Static task definition: `configs/task.yaml`
- Frozen convergence metadata: `configs/provenance.json`
- Full pipeline entry point: `run_all.py`
- Current campaign readiness must be checked from current records.
- Do not assume older manifest rows remain valid after DER config / fairness fixes.

## Frozen Task Summary

- **Case**: `case141`
- **Agents**: `12`
- **Episode**: `48 x 30min`
- **Algorithms**: `ippo`, `ippo_safe`, `ippo_lagrangian`
- **Baselines**: `no_control`, `volt_droop`
- **Eval splits**: `train`, `iid`, `voltage_tightening`, `pv_penetration_shift`, `load_stress`
- **Data**: real Ausgrid load + GB solar traces
- **Primary metric**: `mean_p_loss_mw` (`lower_is_better`)

Current evidence status: Phase 1 has complete 5-seed `jax_rejax+gpu`
evidence with paired primary-split tests. Phase 2 has the mandatory seed-0
backend/device pilot rows (`jax_rejax+cpu`, `sb3+cuda`, `sbx+cuda`) with
official eval rows on all configured splits. DERs is not the execution-scaling
primary task.

## How To Run

```bash
# Baselines only
python benchmarks/ders/run_all.py --only baselines --seeds 0 1 2 3 4

# Train all RL algos
python benchmarks/ders/run_all.py --only train --algos ippo ippo_safe ippo_lagrangian --seeds 0 1 2 3 4

# Evaluate completed training runs on the default split set
python benchmarks/ders/run_all.py --only eval

# Evaluate one specific training run, useful for Phase-2 CPU rows
python benchmarks/ders/run_all.py --only eval --run-ids <train_run_id> \
  --eval-splits train iid voltage_tightening pv_penetration_shift load_stress

# Summaries and plots
python benchmarks/ders/run_all.py --only summarize
python benchmarks/ders/run_all.py --only plots
```

`run_all.py` is the supported top-level orchestration entry point for this
task. There is no `run.py` wrapper here.

## Fairness Notes

- Formal runs must use the same split, seed set, timestep budget, and real-data window across backends.
- For cross-backend comparison, the PowerZoo side uses frozen-self-play IL rather than random-opponent evaluation.
- DER phase-2 training is always on the canonical `train` split. The bridge CLI `--split` selects the requested official eval split, not the training split.
- PowerZoo-side DER phase-2 must match JAX on `case141`, the canonical 12 agent buses, `voltage_penalty=4.0`, and the split-driven real-data episode windows.
- Training-class records must keep canonical curves plus a training artifact (`params*` or `models_manifest`); official DER eval rows must also keep `per_episode`, `trajectory`, and `ders_episode_traces`.
- Do not substitute synthetic traces if real inputs fail to load; fail the run and fix the data path first.
- Treat `ippo` / `ippo_safe` as the useful Phase-1 learned-policy rows; do not present `ippo_lagrangian` as the DERs headline because it remains brittle on PV shift.

## Main Code Paths

- `train.py`
- `eval.py`
- `baselines.py`
- `summarize.py`
- `plots.py`
