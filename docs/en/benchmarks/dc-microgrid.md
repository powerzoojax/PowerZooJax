# DC Microgrid - Multi-Objective Robust Microgrid Control

The DC Microgrid benchmark is a single-agent control task on a grid-connected,
behind-the-meter data-center microgrid. The controller jointly schedules
workload, cooling, battery dispatch, diesel backup, and capped grid import
while balancing energy use, fuel cost, carbon emissions, and service quality.
For the physical env and action semantics, see
[Physics -> Microgrid](../physics/microgrid.md).

## At A Glance

- Physical env: `DataCenterMicrogridEnv`
- Benchmark task: single-agent microgrid control over workload, cooling,
  battery, and diesel backup
- Formal trainer: reward-shaped PPO and SAC
- Primary convergence target: `episode_reward`
- Current canonical benchmark directory: `benchmarks/dc_microgrid`
- Current seed-0 provenance file:
  `benchmarks/dc_microgrid/configs/provenance.json`
- Current evidence: 5-seed Phase-1 and Phase-2 backend/device evidence are
  complete; execution scaling is complete but remains separate from
  algorithm-effect reporting.

## Task Specification

| Field | Value |
| --- | --- |
| Task type | single-agent multi-objective microgrid control |
| Agents | 1 |
| State \(\mathcal{S}\) | task queue, temperatures, battery SOC, PV and load profiles, diesel state, grid price, time-of-day phase |
| Observation \(\mathcal{O}\) | 24-D |
| Action \(\mathcal{A}\) | `Box(5) = [train_sched, ft_sched, cooling_norm, batt_norm, dg_norm]` |
| Transition \(\mathcal{P}\) | workload, cooling, battery, diesel, and capped grid-import decisions update the data center and microgrid balance |
| Reward \(r_t\) | scalarized energy / cost / carbon objective |
| Cost vector \(\mathbf{c}_t\) | \(\left(C_t^{\mathrm{sla}}, C_t^{\mathrm{temp}}, C_t^{\mathrm{deficit}}\right)\) at the env level |
| Discount \(\gamma\) | `0.99` |
| Horizon \(T\) | 288 steps x 5 min = 24 h |
| Initial \(\mu_0\) | data-driven episode from real workload, solar, and weather profiles |

## Underlying Physics

The env combines:

- a data-center workload and thermal model
- battery storage with explicit feasibility
- exogenous PV generation
- a dispatchable diesel generator
- an explicit power-balance constraint with capped grid import

If PV, battery, diesel, and capped grid import cannot cover the data-center
load, the shortfall appears as the unmet-load channel
\(C_t^{\mathrm{deficit}}\). If supply exceeds load after grid import, the
surplus appears as the spill diagnostic \(C_t^{\mathrm{spill}}\).

The benchmark task uses `dg_autobalance=true`, so the DG command is applied as
a same-step residual slack actuator after workload, cooling, and battery
decisions are realised.

## Benchmark Task Parameters

| Parameter | Value |
| --- | --- |
| Environment | `DataCenterMicrogridEnv` |
| Episode | 288 steps x 5 min = 24 h |
| Action space | `Box(5)` |
| Observation | 24-D |
| Primary data source | Google DC workload + GB solar + GB MID grid price + deterministic outdoor temperature |
| Required real-data smoke | `google`, `azure`, `alibaba` with `strict=True, require_real_data=True` |
| Main table splits | `train`, `iid`, `cooling_stress`, `renewable_drought` |
| Appendix splits | `workload_swap`, `workload_shock`, `dg_derating`, `sla_tighten` |
| Primary split | `iid` |
| Primary metric | `episode_reward` (`higher_is_better`) |
| Train budget | PPO and SAC both use `1_000_000` timesteps |
| Phase-2 backend matrix | `jax_rejax+gpu`, `jax_rejax+cpu`, `sb3+cuda`, `sb3+cpu`, `sbx+cuda` |

## Objective, Constraint Channels, And Training Reward

At the env level, the base scalar objective is

\[
r_t
= r_t^{\mathrm{energy}}
+ w_{\mathrm{cost}} r_t^{\mathrm{cost}}
+ w_{\mathrm{carbon}} r_t^{\mathrm{carbon}},
\]

with

\[
r_t^{\mathrm{energy}} = -P_{\mathrm{dc},t}\,\Delta t,
\quad
r_t^{\mathrm{cost}} = -(C_t^{\mathrm{fuel}} + C_t^{\mathrm{deg}}),
\quad
r_t^{\mathrm{carbon}} = -\mathrm{carbon}_t.
\]

The env also exposes the physical cost vector

\[
\mathbf{c}_t
= \left(C_t^{\mathrm{sla}}, C_t^{\mathrm{temp}}, C_t^{\mathrm{deficit}}\right),
\]

which maps to `cost_sla`, `cost_overtemp`, and `cost_power_deficit`.

Env-level cost channel definitions:

| Symbol | Code key | Meaning |
| --- | --- | --- |
| \(C_t^{\mathrm{sla}}\) | `cost_sla` | SLA violation density, computed as expired jobs normalized by the number of GPUs. |
| \(C_t^{\mathrm{temp}}\) | `cost_overtemp` | Temperature safety excess, computed from zone temperature above the critical threshold and normalized by the allowed temperature range. |
| \(C_t^{\mathrm{deficit}}\) | `cost_power_deficit` | Unserved electrical load after PV, diesel, battery, and capped grid import, normalized by current load. |

The benchmark training reward is the fixed shaped objective

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

The first three terms (`sla`, `temp`, `deficit`) are the formal CMDP cost channels declared in the paper Appendix E.5 (Eq. for `\mathbf{c}_t`). The remaining three (`spill`, `track`, `balance`) are **shaping-only diagnostics** introduced by the benchmark reward-shaping wrapper to guide PPO/SAC under the converted plain-MDP objective; they do not appear in the paper-side CMDP definition and are not reported as safety channels in evaluation. They are listed here so the reproducible training reward is unambiguous.

Implementation mapping:

| Symbol | Code key | Meaning |
| --- | --- | --- |
| \(C_t^{\mathrm{spill}}\) | `cost_power_spill` | Shaping-only. Excess supply after serving load and applying capped grid import, normalized by current load. Derived from `power_spill` by the benchmark reward-shaping wrapper. |
| \(C_t^{\mathrm{track}}\) | `cost_dispatch_tracking` | Shaping-only. Deviation from the benchmark wrapper's price-aware battery / diesel dispatch targets. |
| \(C_t^{\mathrm{balance}}\) | `cost_power_balance` | Shaping-only. Absolute residual power imbalance after the same-step balancing logic, normalized by current load. |

Current frozen shaping weights from `benchmarks/dc_microgrid/configs/task.yaml`:

- \(\lambda_{\mathrm{sla}} = 50\)
- \(\lambda_{\mathrm{temp}} = 30\)
- \(\lambda_{\mathrm{def}} = 200\)
- \(\lambda_{\mathrm{spill}} = 100\)
- \(\lambda_{\mathrm{track}} = 80\)
- \(\lambda_{\mathrm{bal}} = 0\)

This benchmark therefore uses:

- env semantics: scalarized base objective plus explicit service / safety channels
- training semantics: reward-shaped PPO / SAC using fixed penalty weights
- evaluation semantics: shaped `episode_reward` plus separately reported
  physical metrics

## Baselines

| Name | Description |
| --- | --- |
| `no_control` | fixed defaults; no scheduling logic |
| `max_renewable` | use PV first, then battery, then diesel |
| `rule_based` | hand-crafted policy for solar alignment and backup generation |

## Algorithms

| Algo | Preset | Notes |
| --- | --- | --- |
| `ppo` | `dc-microgrid` | reward-shaped PPO baseline (Beta-distribution policy head for bounded actions) |
| `sac` | `dc-microgrid` | reward-shaped SAC baseline (tanh-squashed Gaussian) |

Hidden dims `(256, 256)`; gamma `0.99`; `num_envs=64`; `n_steps=288` (one episode per update); total timesteps `1e6`; 5 seeds. PPO uses `lr=1e-4`, `clip_eps=0.1`, `ent_coef=0.001`; SAC uses `lr=3e-4`. Observation normalization is enabled in both canonical train configs. Hyperparameters match paper Appendix H.2 (`tab:hparams`).

## Eval Splits

| Split | Description |
| --- | --- |
| `train` | training window |
| `iid` | held-out days from the same workload pool |
| `cooling_stress` | higher outdoor temperature |
| `renewable_drought` | lower solar availability |

Appendix scenarios are `workload_swap`, `workload_shock`, `dg_derating`, and
`sla_tighten`.

## Metrics

| Layer | Key | Description |
| --- | --- | --- |
| Step base reward | `reward_vector` | decomposed `energy / cost / carbon` terms |
| Step cost channel | `cost_sla` | SLA violation density, `n_expired / n_gpus` |
| Step cost channel | `cost_overtemp` | normalized thermal excess above the critical temperature |
| Step cost channel | `cost_power_deficit` | unmet load after PV, diesel, battery, and capped grid import, normalized by load |
| Step shaping diagnostic | `cost_power_spill` | surplus power after load and grid-import balancing, normalized by load |
| Step shaping diagnostic | `cost_dispatch_tracking` | price-aware battery / diesel dispatch-target tracking error |
| Step shaping diagnostic | `cost_power_balance` | absolute normalized residual power-balance error |
| Episode aggregate | `episode_reward` | cumulative shaped reward used for convergence and summary |
| Episode aggregate | `total_fuel_cost` | total diesel fuel cost |
| Episode aggregate | `total_grid_cost` | total external grid-import cost |
| Episode aggregate | `total_carbon_kg` | total carbon emissions |
| Episode aggregate | `grid_import_mwh` | total imported grid energy |
| Episode aggregate | `mean_grid_price_per_mwh` | average grid-import price over the evaluated episode |
| Episode aggregate | `sla_violation_rate` | mean SLA violation rate |
| Episode aggregate | `overtemp_rate` | fraction of overheating steps |
| Episode aggregate | `power_deficit_rate` | fraction of unmet-load steps |
| Episode aggregate | `feasibility_rate` | fraction of steps without service / safety violation |
| Episode aggregate | `pv_utilization` | PV utilization ratio |
| Episode aggregate | `battery_cycles` | battery throughput / capacity proxy |
| Relative evaluation summary | `norm_score` | normalized score computed on `episode_reward` |
| Relative evaluation summary | `ood_robustness_gap` | `NormScore(iid) - NormScore(cooling_stress)` |

!!! caution "`feasibility_rate` has a numerical tail on the Python backend"
    Be careful when comparing Phase-2 numbers across backends: `feasibility_rate` is a **strict exact-zero** test on per-step `cost_sum`.

    On the Python backend, machine-precision (~$10^{-8}$) power-balance tails can push this strict value toward about `0.5` **even when** the replay audit shows `mean_cost_power_balance` ≈ `1e-8`, tiny maximum power-balance error, and no SLA deficit. In other words, that `0.5` does not mean half the steps actually violate a constraint.

    Paper claims should lean on `episode_reward`, `sla_violation_rate`, `mean_cost_power_balance`, and the Phase-2 physical / paper audits as the primary metrics, **not** on strict `feasibility_rate` as a cross-backend safety measure.

## Quick Start

```bash
python - <<'PY'
from powerzoojax.data.dc_microgrid_profiles import load_workload_profiles
for src in ['google', 'azure', 'alibaba']:
    load_workload_profiles(src, episode_len=16, strict=True, require_real_data=True)
print('real-data smoke OK')
PY

python -m benchmarks.common.experiment_ops seed0_readiness \
  --task dc_microgrid \
  --after 2026-04-23T08:39:52+00:00 \
  --enforce

CUDA_VISIBLE_DEVICES=<gpu_id> python benchmarks/dc_microgrid/run_all.py --seeds 0,1,2,3,4
```

The current Phase-1 canonical campaign is already submission-ready:
`seed0_readiness --after 2026-04-23T08:39:52+00:00 --enforce` passes and
`variance_check --task dc_microgrid` reports `0 warnings, 0 blockers`.

## Execution Scaling

DC Microgrid owns the execution-scaling addendum. These artifacts are separate
from the algorithm-effect leaderboard: they do not compare final reward, and
the JAX-only extended range must not be interpreted as a matched fairness
comparison against Python backends.

The completed scaling artifact has 30 matched-range rows
(`jax_rejax/gpu` vs `sb3/cuda`, `nenv={16,32,64}`, seeds `0..4`) and 20
JAX-only extended rows (`nenv={32,64,128,256}`, seeds `0..4`). It is an
execution artifact, not a leaderboard update.

```bash
# matched-range cross-backend scaling
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=<gpu_id> \
taskset -c <core_range> \
python benchmarks/dc_microgrid/scaling/run_execution_scaling.py \
  --mode formal \
  --suite matched \
  --python-backend sb3 \
  --python-vec-env subproc \
  --matched-nenvs 16,32,64 \
  --seeds 0,1,2,3,4 \
  --cpu-core-budget <n_cores>

# JAX-only extended scaling
CUDA_VISIBLE_DEVICES=<gpu_id> \
python benchmarks/dc_microgrid/scaling/run_execution_scaling.py \
  --mode formal \
  --suite jax-extended \
  --jax-extended-nenvs 32,64,128,256 \
  --seeds 0,1,2,3,4

python benchmarks/dc_microgrid/scaling/plot_execution_scaling.py
```

## Output Files

```text
benchmarks/dc_microgrid/results/
  manifest.json
  runs/
  artifacts/
  summary/latest.json
  figures/
  scaling/execution_scaling.json
  scaling/execution_scaling_table.csv
  representative_episode_summary.json
  phase2_backend_physical_audit.json
  phase2_paper_audit.json
```

## Cross References

- [Physics -> Microgrid](../physics/microgrid.md)
- [Architecture -> Data pipeline](../architecture/data-pipeline.md)
- [API -> Microgrid](../api/microgrid.md)
