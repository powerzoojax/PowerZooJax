# GenCos - 竞争性滚动市场

GenCos benchmark 是一个竞争性的多智能体电力市场任务。5 个 generator company 在 `case5` 上参与 48 步滚动市场，每个 agent 都希望最大化自己的 dispatch profit。市场清算使用精确的 security-constrained economic dispatch（`SCED`），价格则是 locational marginal prices（`LMPs`）。像 SCED、LMP、dispatch、markup 这些市场术语，可见 [Physics -> Markets](../physics/markets.md)。

## 一眼看懂

- 物理 env：`case5` 上带精确 SCED 清算的 `MarketMARLEnv`。
- benchmark 任务：5 智能体竞争性滚动电力市场。
- 实际训练对象：在正式 benchmark 路径里，基于真实 GB demand 的市场任务上训练 IPPO。
- 主要 leaderboard quantity：`total_profit`。
- 审计门槛：报告流程还会用 `market_HHI` 审计市场集中度。
- 当前 evidence：JAX/GPU 5-seed evidence 已完整；IPPO 显著高于 `truthful` 和 `uniform_mid`，但仍低于强 `max_markup` heuristic。

第一次阅读 GenCos 时，建议按这个顺序看：先看摘要，再看任务定义表，再看 reward 与 cost，最后再看 quick start。这样更容易把市场博弈本身、benchmark 报告口径和脚本工作流分开理解。

## MDP / CMDP 定义

| 字段 | 内容 |
| --- | --- |
| MDP class | partially observed Markov game |
| Agents | 5 |
| State \(\mathcal{S}\) | 上一步 dispatch / profit、ramp headroom、节点价格、近期价格历史、系统负荷、时间相位 |
| Observation \(\mathcal{O}_i\) | 每个 agent 12 维（默认 `lmp_history_len=4`）：自身 dispatch / profit 状态、ramp headroom、未来一步总负荷预测、时间特征，加 4 个系统平均 LMP 历史 |
| Action \(\mathcal{A}_i\) | `Box(n_segments)`，范围 `[-1, 1]`（默认 `n_segments=3`） |
| Transition \(\mathcal{P}\) | 出价映射为 offer，市场通过精确 SCED 清算，清算后的 dispatch 通过 ramp limits 影响下一步 |
| Reward \(r_{i,t}\) | 每个 agent 的 dispatch profit |
| Cost \(c_t\) | \(c_t = \left(C_t^{\mathrm{therm}}\right)\) |
| Threshold \(d\) | `ConstraintSpec` 阈值为 `(0.0,)`；当前 benchmark 路径并没有单独的 safe trainer |
| Discount \(\gamma\) | `0.995` |
| Horizon \(T\) | 48 steps x 30 min = 24 h |
| Initial \(\mu_0\) | 从 GB demand history 中随机采样 episode 起点 |

这更像一个 partially observed Markov game，而不是 CMDP 风格的 safe RL 任务。thermal-overload 通道依然存在，作为物理可行性诊断；但 benchmark 的主要目标是市场利润。

## 底层物理

该环境建立在 `MarketMARLEnv` 之上，详见 [Physics -> Markets](../physics/markets.md#marketmarlenv-gencos-rolling-market)。每一步都做精确的 offer-based SCED 清算，ramp limits 让相邻市场步骤相互耦合，因此这不是一组彼此独立的一次性拍卖。

在物理层，市场 core 仍然会暴露 thermal-overload 诊断通道。但在 benchmark 层，核心关注的是市场结果：

\[
r_{i,t} = \Pi_{i,t}
\]

对每个 generator company \(i\) 都成立。

## Benchmark 任务参数

| 参数 | 内容 |
| --- | --- |
| Case | `case5` |
| Agents | 5 |
| Action per agent | `Box(n_segments)` |
| Episode | 48 steps x 30 min = 24 h |
| Clearing | 精确 offer-based SCED |
| Ramp coupling | 跨步生效 |
| LMP history | 观测中保留最近 4 个 mean LMP |
| Data source | GB demand pool |
| Primary metric | `total_profit`（`higher_is_better`） |

`case5` 之所以被选中，是因为它足够小，便于大批量实验；但又足够让拥塞与 market power 真的产生作用。

和其他 benchmark 任务一样，GenCos 也有一份已定版的任务配置文件，位于 `benchmarks/gencos/configs/task.yaml`。这份配置固定了 split 列表、seeds、primary metric，以及报告流程里使用的审计阈值。

## 动作与 offer 映射

每个 agent 的动作控制该发电机各个 offer segment 的 markup。benchmark 层面的含义很简单：

- 动作越低，markup 越低
- 动作越高，markup 越高
- 固定 bidding baselines `truthful`、`uniform_mid`、`max_markup` 分别对应动作值 `-1`、`0`、`1`

至于单调 offer 是如何构造出来的、以及 LP 清算细节，则属于 solver 层内容，见 [Physics -> Markets](../physics/markets.md#marketmarlenv-gencos-rolling-market)。

## 单个 agent 的观测

典型观测包括：

- 自身最近的 dispatch 与 profit
- 剩余 ramp headroom
- 本地价格信号
- 最近 mean-LMP 历史
- 一步前瞻的负荷上下文
- 时间特征

当前 wrapper 会在底层 core state 和 `info` 中保留完整 nodal LMP 向量，但给每个 agent 的 private observation 是一个压缩后的 bidding-context 向量，而不是完整市场状态。详见 [API -> Market MARL](../api/market-marl.md)。

## Reward 与 cost

每个 agent 的 reward 是 dispatch profit：

\[
\Pi_{i,t} = \mathrm{LMP}_{b(i),t}\, P_{i,t}\, \Delta t - \mathrm{TC}_i(P_{i,t})\, \Delta t
\]

其中：

- \(b(i)\) 是 generator company \(i\) 所在的母线
- \(P_{i,t}\) 是清算得到的发电功率
- \(\mathrm{LMP}_{b(i),t}\) 是该母线处的节点价格
- \(\mathrm{TC}_i(P)\) 是真实发电成本曲线

任务层的物理可行性通道为

\[
c_t = \left(C_t^{\mathrm{therm}}\right)
\]

它对应 thermal overload。benchmark 并不会在这个通道上训练单独的 safe-RL optimizer；它主要作为可行性与审计诊断存在。

在 episode 层，主要 leaderboard quantity 是

\[
\Pi_{\mathrm{ep}} = \sum_{t=0}^{T-1} \sum_{i=1}^{5} \Pi_{i,t}
\]

对应 `total_profit`。

## Baselines

| 名称 | Action value | 说明 |
| --- | --- | --- |
| `truthful` | `-1` | 按真实 segment cost 出价 |
| `uniform_mid` | `0` | 按中间 markup 出价 |
| `max_markup` | `1` | 按允许的最大 markup 出价 |

这些都是固定 bidding 策略，而不是学习算法。

## 算法

| Algo | Preset | 说明 |
| --- | --- | --- |
| `ippo` | `gencos-case5-ippo` | 使用真实 GB demand 的 benchmark preset |
| `ippo`（synthetic） | `gencos-case5-ippo-dev` | 仅用于开发检查，不能用于 benchmark 报告 |

隐藏层 `(128, 128)`；gamma `0.995`；总训练步数 `5e6`；Phase-1 JAX/GPU 报告使用 5 个 seeds。

Phase-2 Python backend 行使用 PowerZoo frozen self-play IL，不是 random-opponent training。第 1 轮因为还没有 frozen policies 可采样，会从无冻结对手开始；后续轮次使用 frozen opponent policies。

## Eval splits

| Split | 说明 |
| --- | --- |
| `train` | `2025-04-01` 到 `2025-12-31` |
| `iid` | `2026-01-01` 到 `2026-03-31` |
| `demand_shift` | 上移的 demand 水平 |
| `renewable_shock` | 更紧张的净负荷 / 可再生可用性代理冲击 |

这些 OOD split 保持市场规则不变，只扰动需求侧环境。

## Metrics

把利润指标和市场结构诊断分开理解会更清楚：

| 层级 | Key | 说明 |
| --- | --- | --- |
| Step reward | `profit_i` | 每个 agent 的 per-step dispatch profit |
| Step feasibility channel | `thermal_overload` | thermal-overload 诊断 |
| Episode aggregate | `total_profit` | 所有 agent 的 episode 总利润 |
| Episode aggregate | `mean_profit_per_agent` | `total_profit / 5` |
| Episode aggregate | `total_gen_cost` | 实际总发电成本 |
| Episode aggregate | `mean_lmp` | 平均 locational marginal price |
| Episode aggregate | `price_volatility` | mean-LMP 序列的波动性 |
| Episode aggregate | `hhi` | dispatch share 的 Herfindahl-Hirschman Index，即集中度指标 |
| Episode aggregate | `sced_convergence_rate` | 精确 SCED 成功收敛的步占比 |
| Episode aggregate | `ramp_binding_rate` | ramp constraints 生效的步占比 |
| Relative evaluation summary | `NormScore` | 相对固定 bidding baselines 的归一化利润分数 |

这份已定版任务配置里的审计阈值 `benchmarks/gencos/configs/task.yaml::safety_thresholds.market_HHI` 使用的是 `market_HHI`。这不是因为 HHI 像线路过载那样是物理安全约束，而是因为 benchmark 也会把市场集中度和近似垄断行为作为结果审计的一部分。

## Quick start

```bash
python benchmarks/gencos/run_all.py --only baselines --seeds 0 1 2 3 4
python benchmarks/gencos/run_all.py --only train --algos ippo --seeds 0 1 2 3 4
python benchmarks/gencos/run_all.py --only eval
python benchmarks/gencos/run_all.py --only summarize
python benchmarks/gencos/run_all.py --only plots
```

## 注意事项

- 正式 benchmark 使用的是精确 offer-based SCED，不是启发式清算近似。
- 开发 preset `gencos-case5-ippo-dev` 仅用于本地快速检查和 CI。
- 因为 reward 是自利的 dispatch profit，所以这个 benchmark 研究的是策略性市场行为，而不是直接优化系统福利。

## 交叉引用

- [Physics -> Markets](../physics/markets.md)
- [API -> Market SCED](../api/market-sced.md)
- [API -> Market MARL](../api/market-marl.md)
- [Training -> Trainers](../training/trainers.md)
