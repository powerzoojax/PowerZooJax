# GenCos - Competitive Rolling Market

The GenCos benchmark is a competitive multi-agent electricity-market task. Five generator companies bid into a 48-step rolling market on `case5`, and each agent tries to maximize its own dispatch profit. Clearing uses exact security-constrained economic dispatch (`SCED`), and prices are locational marginal prices (`LMPs`). For market terms such as SCED, LMP, dispatch, and markup, see [Physics -> Markets](../physics/markets.md).

## At A Glance

- Physical env: `MarketMARLEnv` with exact SCED clearing on `case5`.
- Benchmark task: 5-agent competitive rolling electricity market.
- What is actually trained: IPPO on the benchmark market task using real GB demand in the formal path.
- Primary leaderboard quantity: `total_profit`.
- Audit gate: the reporting pipeline also audits market concentration via `market_HHI`.
- Current evidence: 5-seed JAX/GPU evidence is complete; IPPO beats `truthful`
  and `uniform_mid` but remains below the strong `max_markup` heuristic.

If this is your first read, go in this order: summary, task table, reward-and-cost section, then quick-start commands. That makes it easier to separate the market game itself from the benchmark reporting and from the runnable scripts.

## MDP / CMDP specification

| Field | Value |
| --- | --- |
| MDP class | partially observed Markov game |
| Agents | 5 |
| State \(\mathcal{S}\) | last dispatch and profit, ramp headroom, nodal prices, recent price history, system load, time-of-day phase |
| Observation \(\mathcal{O}_i\) | 12-dim per agent (default `lmp_history_len=4`): own dispatch / profit state, ramp headroom, one-step-ahead total-load forecast, time features, and 4 mean-LMP history entries |
| Action \(\mathcal{A}_i\) | `Box(n_segments)` in `[-1, 1]` (default `n_segments=3`) |
| Transition \(\mathcal{P}\) | bids map to offers, the market is cleared by exact SCED, and cleared dispatch affects the next step through ramp limits |
| Reward \(r_{i,t}\) | per-agent dispatch profit |
| Cost \(c_t\) | \(c_t = \left(C_t^{\mathrm{therm}}\right)\) |
| Threshold \(d\) | `ConstraintSpec` threshold `(0.0,)`; no separate safe trainer is used in the current benchmark path |
| Discount \(\gamma\) | `0.995` |
| Horizon \(T\) | 48 steps x 30 min = 24 h |
| Initial \(\mu_0\) | random episode start sampled from the GB demand history |

This is a partially observed Markov game rather than a CMDP-style safe RL task. The thermal-overload channel still exists as a physical feasibility diagnostic, but the benchmark objective is strategic profit.

## Underlying physics

The env is built on `MarketMARLEnv`, described in [Physics -> Markets](../physics/markets.md#marketmarlenv-gencos-rolling-market), with exact offer-based SCED clearing at each step. Ramp limits couple adjacent market steps, so the task is intertemporal rather than a collection of independent one-shot auctions.

At the physics layer, the market core still exposes a thermal-overload diagnostic channel. At the benchmark layer, however, the main quantity of interest is market outcome:

\[
r_{i,t} = \Pi_{i,t}
\]

for each generator company \(i\).

## Benchmark task parameters

| Parameter | Value |
| --- | --- |
| Case | `case5` |
| Agents | 5 |
| Action per agent | `Box(n_segments)` |
| Episode | 48 steps x 30 min = 24 h |
| Clearing | exact offer-based SCED |
| Ramp coupling | enforced across steps |
| LMP history | last 4 mean LMPs in observation |
| Data source | GB demand pool |
| Primary metric | `total_profit` (`higher_is_better`) |

The small `case5` size is deliberate: it is cheap enough for large batched experiments, but still large enough for congestion and market power to matter.

Like the other benchmark tasks, GenCos also has a frozen benchmark task config in `benchmarks/gencos/configs/task.yaml`. That config fixes the split list, seeds, primary metric, and audit thresholds used by the reporting pipeline.

## Action and offer mapping

Each agent action controls the markup on that generator's offer segments. The benchmark-level meaning is simple:

- lower action means lower markup
- higher action means higher markup
- the frozen bidding baselines `truthful`, `uniform_mid`, and `max_markup` correspond to action values `-1`, `0`, and `1`

The exact monotone offer construction and LP clearing are solver details and are documented in [Physics -> Markets](../physics/markets.md#marketmarlenv-gencos-rolling-market).

## Per-agent observation

A typical observation includes:

- own recent dispatch and profit
- remaining ramp headroom
- local price signal
- recent mean-LMP history
- one-step-ahead load context
- time-of-day features

The current wrapper keeps the full nodal LMP vector in the underlying core state and `info`, but the private per-agent observation is a compact bidding-context vector rather than the full market state. See [API -> Market MARL](../api/market-marl.md).

## Reward and cost

Per-agent reward is dispatch profit:

\[
\Pi_{i,t} = \mathrm{LMP}_{b(i),t}\, P_{i,t}\, \Delta t - \mathrm{TC}_i(P_{i,t})\, \Delta t
\]

where:

- \(b(i)\) is the bus of generator company \(i\)
- \(P_{i,t}\) is cleared generation
- \(\mathrm{LMP}_{b(i),t}\) is the nodal price at that generator's bus
- \(\mathrm{TC}_i(P)\) is the true generation cost curve

The task-level physical-feasibility channel is

\[
c_t = \left(C_t^{\mathrm{therm}}\right)
\]

which corresponds to thermal overload. The benchmark does not train a separate safe-RL optimizer on this channel; it remains a feasibility and audit diagnostic.

Feasibility channel definition:

| Symbol | Constraint name | Typical key | Meaning |
| --- | --- | --- | --- |
| \(C_t^{\mathrm{therm}}\) | `thermal_overload` | `thermal_overload`, `cost_thermal_overload` | Thermal-overload diagnostic from SCED clearing when line-flow limits are exceeded. It is audited separately from strategic profit. |

At the episode level, the main leaderboard quantity is

\[
\Pi_{\mathrm{ep}} = \sum_{t=0}^{T-1} \sum_{i=1}^{5} \Pi_{i,t}
\]

reported as `total_profit`.

## Baselines

| Name | Action value | Description |
| --- | --- | --- |
| `truthful` | `-1` | bid true segment costs |
| `uniform_mid` | `0` | bid midpoint markup |
| `max_markup` | `1` | bid the maximum allowed markup |

These are fixed bidding strategies, not learning algorithms.

## Algorithms

| Algo | Preset | Notes |
| --- | --- | --- |
| `ippo` | `gencos-case5-ippo` | benchmark preset using real GB demand |
| `ippo` (synthetic) | `gencos-case5-ippo-dev` | development-only preset, not for benchmark reporting |

Hidden dims `(128, 128)`; gamma `0.995`; total timesteps `5e6`; 5 seeds for
Phase-1 JAX/GPU reporting.

For Phase-2 Python backend rows, the PowerZoo side uses frozen self-play IL
rather than random-opponent training. The first round has no frozen policies to
sample from; later rounds use frozen opponent policies.

## Eval splits

| Split | Description |
| --- | --- |
| `train` | `2025-04-01` to `2025-12-31` |
| `iid` | `2026-01-01` to `2026-03-31` |
| `demand_shift` | upward demand shift |
| `renewable_shock` | stressed net-load / renewable-availability proxy |

The OOD splits hold market rules fixed and perturb the demand-side environment.

## Metrics

It helps to separate profit metrics from market-structure diagnostics:

| Layer | Key | Description |
| --- | --- | --- |
| Step reward | `profit_i` | per-agent dispatch profit |
| Step feasibility channel | `thermal_overload` | thermal-overload diagnostic from SCED line-limit checks |
| Episode aggregate | `total_profit` | sum of all agents' episode profits |
| Episode aggregate | `mean_profit_per_agent` | `total_profit / 5` |
| Episode aggregate | `total_gen_cost` | total realized generation cost |
| Episode aggregate | `mean_lmp` | mean locational marginal price |
| Episode aggregate | `price_volatility` | variability of the mean-LMP series |
| Episode aggregate | `hhi` | Herfindahl-Hirschman Index of dispatch shares, a concentration metric |
| Episode aggregate | `sced_convergence_rate` | fraction of steps where exact SCED converged |
| Episode aggregate | `ramp_binding_rate` | fraction of steps where ramp constraints bind |
| Relative evaluation summary | `NormScore` | normalized profit score against bidding baselines |

The frozen task config audit threshold `benchmarks/gencos/configs/task.yaml::safety_thresholds.market_HHI` uses `market_HHI`, not because HHI is a physical safety constraint like line overload, but because the benchmark also tracks market concentration and monopoly-style behavior as an outcome audit.

## Quick start

```bash
python benchmarks/gencos/run_all.py --only baselines --seeds 0 1 2 3 4
python benchmarks/gencos/run_all.py --only train --algos ippo --seeds 0 1 2 3 4
python benchmarks/gencos/run_all.py --only eval
python benchmarks/gencos/run_all.py --only summarize
python benchmarks/gencos/run_all.py --only plots
```

## Caveats

- The formal benchmark uses exact offer-based SCED, not a heuristic clearing shortcut.
- The dev preset `gencos-case5-ippo-dev` exists only for local quick checks and CI.
- Because the reward is selfish dispatch profit, the benchmark studies strategic market behavior rather than direct system-welfare optimization.

## Cross references

- [Physics -> Markets](../physics/markets.md)
- [API -> Market SCED](../api/market-sced.md)
- [API -> Market MARL](../api/market-marl.md)
- [Training -> Trainers](../training/trainers.md)
