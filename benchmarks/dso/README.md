# DSO

Single-agent distribution-grid benchmark on `case33bw` with 6 `FlexLoad`
devices and Ausgrid half-hourly load data.

## Task

- Horizon: `48` steps (`24 h`, `dt_hours = 0.5`)
- Voltage band: `[0.94, 1.06] p.u.`
- Slack voltage: `1.045 p.u.`
- Load scaling: `0.83`
- Objective: minimize network loss
- Safety metric: `voltage_violation_rate = 0`

## Data and Splits

- Data source: `ausgrid`
- Training windows: `8` fixed `iid` windows sampled via reset-bank
- Official evaluation split: `iid`
- Protocol seeds: `0, 1, 2, 3, 4`
- Eval episodes per split: `50`

Current evidence status: Phase 1 `jax_rejax+gpu` and the Phase-2 PPO
backend/device matrix both have 5-seed IID evidence. DSO remains a
backend/device algorithm-effect reporting task, not the execution-scaling
primary task.

## Algorithms

- Baselines: `no_control`, `tou`, `droop`
- RL: `ppo`, `sac`, `saute_ppo`, `ppo_lagrangian`

## Entry Points

```bash
python benchmarks/dso/run.py baseline --seeds 0,1,2,3,4

python benchmarks/dso/run.py train --algo ppo --seed 0
python benchmarks/dso/run.py train --algo sac --seed 0
python benchmarks/dso/run.py train --algo saute_ppo --seed 0
python benchmarks/dso/run.py train --algo ppo_lagrangian --seed 0

python benchmarks/dso/run.py eval --run-id <run_id> --split iid
python benchmarks/dso/run.py summarize
python benchmarks/dso/run.py plots

python benchmarks/dso/run_all.py --seeds 0 1 2 3 4
```

## Outputs

```text
benchmarks/dso/results/
  manifest.json
  runs/
  artifacts/
  summary/latest.json
  figures/
```
