# DC Microgrid — Data Center Operation Benchmark

Single-agent benchmark for a grid-connected data-center microgrid:
joint scheduling of workload, cooling, battery, PV, diesel, and grid import
over `288 x 5min` steps.

## Canonical Source

- Task directory: `benchmarks/dc_microgrid`
- Frozen task definition: `configs/task.yaml`
- Seed-0 target and provenance: `configs/provenance.json`
- Aggregate benchmark summary: `results/summary/latest.json`
- Paper figures: `results/figures/`
- Representative physical audit: `results/representative_episode_summary.json`
- Phase-2 backend physical audit: `results/phase2_backend_physical_audit.json` (rerun required after the grid-price semantics update)
- Phase-2 paper-facing audit: `results/phase2_paper_audit.json` (rerun required after the grid-price semantics update)

## Current Benchmark Facts

| Parameter | Value |
|-----------|-------|
| Environment | `DataCenterMicrogridEnv` |
| Episode length | `288 x 5min = 24h` |
| Action space | `Box(5)` |
| Observation | `24-D` |
| Primary data source | real Google workload + GB solar trace + GB MID market price / deterministic temperature profile |
| Required real-data smoke | `google`, `azure`, `alibaba` with `strict=True, require_real_data=True` |
| Baselines | `no_control`, `max_renewable`, `rule_based` |
| RL algorithms | `ppo`, `sac` |
| Main splits | `train`, `iid`, `cooling_stress`, `renewable_drought` |
| Appendix splits | `workload_swap`, `workload_shock`, `dg_derating`, `sla_tighten` |
| Benchmark throughput cap | `task.yaml::num_envs = 256` |
| Canonical train configs | `train_{ppo,sac}.yaml::num_envs = 64`, `total_timesteps = 1_000_000` |
| Phase-2 formal matrix | `jax_rejax+gpu`, `jax_rejax+cpu`, `sb3+cuda`, `sb3+cpu`, `sbx+cuda` (pending rerun under current grid-price semantics) |
| Primary split | `iid` |
| Current campaign start | `2026-04-27T02:17:12+00:00` |

Current evidence status: Phase 1 has complete 5-seed `jax_rejax+gpu`
evidence for PPO/SAC and baselines under the grid-connected MID-price task.
Previous Phase-2 and execution-scaling artifacts were archived because the task
semantics changed; they must be rerun before paper-facing cross-backend or
scaling claims are refreshed.

## Objective And Shaping

The env exposes a scalar base reward \(r_t\) plus explicit per-step cost
channels. Formal PPO/SAC training uses the shaped objective

\[
r_t^{\mathrm{train}} =
r_t
- \lambda_{\mathrm{sla}} C_t^{\mathrm{sla}}
- \lambda_{\mathrm{temp}} C_t^{\mathrm{temp}}
- \lambda_{\mathrm{def}} C_t^{\mathrm{deficit}}
- \lambda_{\mathrm{spill}} C_t^{\mathrm{spill}}
- \lambda_{\mathrm{track}} C_t^{\mathrm{track}}
- \lambda_{\mathrm{bal}} C_t^{\mathrm{balance}}.
\]

Current implementation mapping:

- \(C_t^{\mathrm{sla}}\) -> `info["cost_sla"]`
- \(C_t^{\mathrm{temp}}\) -> `info["cost_overtemp"]`
- \(C_t^{\mathrm{deficit}}\) -> `info["cost_power_deficit"]`
- \(C_t^{\mathrm{spill}}\) -> `info["cost_power_spill"]`
- \(C_t^{\mathrm{track}}\) -> `info["cost_dispatch_tracking"]`
- \(C_t^{\mathrm{balance}}\) -> `info["cost_power_balance"]`

Current weights from `configs/task.yaml`:

- `sla = 50`
- `overtemp = 30`
- `power_deficit = 200`
- `power_spill = 100`
- `dispatch_tracking = 40`

The shaping wrapper lives in [`_reward_shaping.py`](./_reward_shaping.py).
With `task.yaml::dg_autobalance=true`, DG is benchmark-wrapped as a same-step
residual slack actuator after workload, cooling, and battery decisions are
realised.

## Real-Data Rules

- Formal runs must read `google`, `azure`, and `alibaba` successfully with `strict=True, require_real_data=True`.
- `workload_swap` and `workload_shock` require real Azure / Alibaba traces.
- If required real data is unavailable, stop the run. Do not fall back to synthetic data.

## Canonical Commands

```bash
# Real-data smoke
python - <<'PY'
from powerzoojax.data.dc_microgrid_profiles import load_workload_profiles
for src in ['google', 'azure', 'alibaba']:
    load_workload_profiles(src, episode_len=16, strict=True, require_real_data=True)
print('real-data smoke OK')
PY

# Seed-0 readiness check
python -m benchmarks.common.experiment_ops seed0_readiness \
  --task dc_microgrid \
  --after 2026-04-23T08:39:52+00:00 \
  --enforce

# Formal 5-seed campaign rerun or refresh
CUDA_VISIBLE_DEVICES=2 python benchmarks/dc_microgrid/run_all.py --seeds 0,1,2,3,4

# Aggregate refresh
python benchmarks/dc_microgrid/run.py summarize
python benchmarks/dc_microgrid/run.py plots

# Phase-2 backend physical audit
PYTHONPATH=. python benchmarks/dc_microgrid/analysis/phase2_physical_audit.py

# Phase-2 paper-facing fairness/significance audit
PYTHONPATH=. python benchmarks/dc_microgrid/analysis/phase2_paper_audit.py
```

## Execution Scaling

Execution scaling is tracked separately from the algorithm-effect leaderboard.
These artifacts do not compare final reward and must not be used as a
fairness claim between non-overlapping `nenv` ranges.

Current scaling artifacts contain the completed matched-range sweep
(`jax_rejax/gpu` vs `sb3/cuda`, `nenv={16,32,64}`, seeds `0..4`) and the
JAX-only extended sweep (`nenv={32,64,128,256}`, seeds `0..4`). Treat the
extended range as a scaling artifact, not as a matched cross-backend win.

Output locations:

- `results/scaling/execution_scaling.json`
- `results/scaling/execution_scaling_table.csv`
- `results/figures/execution_scaling_matched_range.{png,pdf}`
- `results/figures/execution_scaling_jax_extended.{png,pdf}`

Preparation / dry-run:

```bash
python benchmarks/dc_microgrid/scaling/run_execution_scaling.py \
  --mode dry-run \
  --suite all \
  --seeds 0
```

Pilot examples, after selecting the least-busy GPU:

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> \
python benchmarks/dc_microgrid/scaling/run_execution_scaling.py \
  --mode pilot \
  --suite single \
  --backend jax_rejax \
  --device gpu \
  --nenv 32 \
  --seeds 0 \
  --max-updates 3

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=<gpu_id> \
taskset -c <core_range> \
python benchmarks/dc_microgrid/scaling/run_execution_scaling.py \
  --mode pilot \
  --suite single \
  --backend sb3 \
  --device cuda \
  --nenv 32 \
  --seeds 0 \
  --cpu-core-budget <n_cores> \
  --python-vec-env subproc \
  --max-updates 3
```

Formal commands for rerun or extension:

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> \
python benchmarks/dc_microgrid/scaling/run_execution_scaling.py \
  --mode formal \
  --suite matched \
  --python-backend sb3 \
  --python-vec-env subproc \
  --matched-nenvs 16,32,64 \
  --seeds 0,1,2,3,4

CUDA_VISIBLE_DEVICES=<gpu_id> \
python benchmarks/dc_microgrid/scaling/run_execution_scaling.py \
  --mode formal \
  --suite jax-extended \
  --jax-extended-nenvs 32,64,128,256 \
  --seeds 0,1,2,3,4

python benchmarks/dc_microgrid/scaling/plot_execution_scaling.py
```

## Primary Metrics

- `episode_reward`
- `sla_violation_rate`
- `overtemp_rate`
- `power_deficit_rate`
- `feasibility_rate`
- `total_fuel_cost`
- `total_grid_cost`
- `grid_import_mwh`
- `mean_grid_price_per_mwh`
- `total_carbon_kg`
- `pv_utilization`
- `battery_cycles`

`episode_reward` is the convergence target recorded in `configs/provenance.json`.

For cross-backend Phase-2 interpretation after rerun, `feasibility_rate` is a
**strict** exact-zero metric on per-step `cost_sum`. On the Python backends this
can be numerically brittle: machine-precision power-balance tails may depress
strict `feasibility_rate` even when replay residuals and SLA metrics are
acceptable. For paper claims, prefer:

- `episode_reward`
- `sla_violation_rate`
- `mean_cost_power_balance`
- `results/phase2_backend_physical_audit.json`
- `results/phase2_paper_audit.json`
