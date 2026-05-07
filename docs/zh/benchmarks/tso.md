# TSO - 安全约束机组组合

TSO benchmark 是 IEEE 118-bus 系统上的集中式 [unit commitment 与 economic dispatch](../concepts/power-systems-primer.md) 任务。每一步中，智能体决定哪些发电机应当开机，以及 dispatch 应向什么方向偏置；物理可行性由求解器保证。

本页是 TSO 任务说明：定义任务、命令流程、数据要求、指标和生成结果。共享术语见 [Benchmark workflow glossary](../glossary.md)。

## 一眼看懂

- 物理 env：IEEE `case118` 上的 `UnitCommitmentEnv`。
- benchmark 任务：集中式的安全约束机组组合与调度。
- 实际训练对象：非安全路径训练 PPO；安全路径训练 PPO-Lagrangian。
- 主要 leaderboard quantity：`iid` split 上的 `total_operating_cost`。
- 安全门槛：只有同时满足零热过载和零备用不足的结果才有 leaderboard 资格。

如果你是第一次读这个任务，建议按这个顺序看：先看上面的直观摘要，再看 MDP / CMDP 表，再看 reward 与 cost 小节，最后再看 quick start 命令。这样可以把物理问题、benchmark 合约和可运行工作流分开理解。

## MDP / CMDP 定义 {#mdp-cmdp-spec}

| 字段 | 内容 |
| --- | --- |
| [MDP class](#tso-rl-algos) | MDP (`tso-uc`) / CMDP (`tso-scuc-safe`，即 constrained Markov decision process) |
| [Agents](#tso-benchmark-params) | 1（集中式） |
| [State \(\mathcal{S}\)](#tso-physics) | 机组开关状态、持续时间、上一步出力、线路潮流、当前负荷、备用比例、时间相位 |
| [Observation \(\mathcal{O}\)](#tso-action-obs) | `Box(4 * 54 + 186 + 4 + 4) = Box(410)`（legacy UC state 加 4-step future total-load forecast；字段语义见 [Physics → Transmission](../physics/transmission.md)） |
| [Action \(\mathcal{A}\)](#tso-action-obs) | `Box(108) = [commitment_intent (54) | dispatch_preference (54)]`，范围 `[-1, 1]` |
| [Transition \(\mathcal{P}\)](#tso-physics) | 先应用最小开停机时间约束，再在 ramp 约束下做 [DC OPF](../concepts/power-systems-primer.md) dispatch |
| [Reward \(r_t\)](#tso-reward-cost) | \(r_t = -\lambda_{\mathrm{reward}} C^{\mathrm{op}}_t\)，其中 \(C^{\mathrm{op}}_t\) 为单步运行成本 |
| [Cost \(\mathbf{c}_t\)](#tso-reward-cost) | \(\mathbf{c}_t = (C^{\mathrm{th}}_t, C^{\mathrm{res}}_t)\)：safe 版本使用热越限与备用不足两个约束通道 |
| [Threshold \(d\)](#tso-reward-cost) | `tso-scuc-safe` 对 thermal overload 和 reserve shortfall 都使用 0 阈值 |
| [Discount \(\gamma\)](#tso-rl-algos) | `0.995` |
| [Horizon \(T\)](#tso-benchmark-params) | 48 steps x 30 min = 24 h |
| [Initial \(\mu_0\)](#tso-benchmark-params) | 从 GB demand + generation profiles 采样 episode |

## 物理背景 {#tso-physics}

环境基于 `UnitCommitmentEnv`（见 [Physics → Transmission](../physics/transmission.md)）。从电力系统语义上看，这是一个 [SCUC](../concepts/power-systems-primer.md) 任务：多步机组组合，同时满足网络安全与备用约束。

主要物理与运行约束包括：

- 连续的 commitment signal 会在环境内部阈值化为二元开关机决策；
- 最小开机 / 停机时间在 `step` 内部直接保证可行；
- dispatch 由带 ramp 限制的 [DC OPF](../concepts/power-systems-primer.md) 求解；
- 成本包含一次性 startup cost 和逐步累积的 no-load cost；
- 可选系统级 [reserve](../concepts/power-systems-primer.md) margin 约束。

## 基准任务参数 {#tso-benchmark-params}

| 参数 | 内容 |
| --- | --- |
| Case | IEEE case118 |
| Generators | 54 台机组 |
| Buses | 118 |
| Lines | 186 |
| Episode length | 48 steps x 30 min = 24 h |
| Agents | 1（集中式） |
| Action space | `Box(108) = [commitment_intent (54) | dispatch_preference (54)]` |
| Observation | `Box(4 * 54 + 186 + 4 + 4) = Box(410)`；字段语义见 [Physics → Transmission](../physics/transmission.md) |
| Data source | GB demand + gen-by-type（benchmark 使用真实数据） |
| Train split | 2025-04-01 到 2025-12-31 |
| IID split | 2026-01-01 到 2026-03-31 |

开发或 CI 可以保留 synthetic profiles，但 benchmark 报告必须使用任务配置指定的真实 GB 数据。

任务配置还携带机器可读的 benchmark protocol 元数据：

- 当前 [campaign](../glossary.md#campaign) seed budget：5；
- [submission-grade minimum](../glossary.md#submission-grade-minimum)：5 seeds；
- 必要统计量：mean、std、IQM（interquartile mean，四分位均值）和 95% [bootstrap CI](../glossary.md#bootstrap-ci)；
- 主 leaderboard：`iid` split 按 `total_operating_cost` 排序；
- 不安全策略仍会被报告，但没有 [leaderboard 资格](../glossary.md#leaderboard-quantity)。

当前 5-seed evidence 已完整，但 hard safety gate 是 negative：没有任何 primary-split 行同时满足 `reserve_shortfall_rate == 0.0` 和 `thermal_violation_rate == 0.0`。这应被写成 benchmark-hardness 负结果，而不是降低零违反阈值的理由。

## 动作与观测 {#tso-action-obs}

动作分为两部分：

- `0..53`：commitment intent（`commit_intent`，论文符号 \(\mathbf{u}^{\mathrm{cmd}}_{t}\)）。阈值化后，`> 0` 表示请求机组开机。
- `54..107`：dispatch preference（`dispatch_preference`，论文符号 \(\mathbf{P}^{\mathrm{pref}}_{t}\)）。在 commit mask 应用后，反归一化到可行出力区间 `[ramp_p_min, ramp_p_max]`。它表示每台机组的偏好可行出力目标，而不是最终直接下发到电网的出力值。

观测为：

`[unit_status (54) | time_in_state_norm (54) | last_dispatch_norm (54) | unit_cost_b_norm (54) | line_flow_norm (186) | load_norm | reserve_ratio | sin(t) | cos(t)]`

各观测分块的逐字段说明见 [Physics → Transmission](../physics/transmission.md)。

## Reward 与 CMDP cost {#tso-reward-cost}

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

其中，\(C^{\mathrm{op}}_t\) 表示 SCUC 实际执行后对应的单步运行成本，由发电成本、启停成本和空载成本组成。在实现中，它们分别对应 `gen_cost`、`startup_cost`、`no_load_cost`，而 \(\lambda_{\mathrm{reward}}\) 对应 `reward_scale`。

对于 safe 版本，\(C^{\mathrm{th}}_t\) 表示热越限幅度，\(C^{\mathrm{res}}_t\) 表示备用不足幅度。固定的 CMDP 通道名为 `("thermal_overload", "reserve_shortfall")`。rollout 诊断中，对应量通过 `cost_thermal_overload`、`reserve_shortfall` 等字段暴露。

底层 `UnitCommitmentEnv` 还提供第三个通道 `min_updown`（`cost_min_updown`），仅用于让下游 wrapper 拿到固定 shape 的 cost 向量，因为最短开机 / 最短停机约束在 `step` 内由 hard mask 强制满足，所以它恒为 0（见 [Physics → Transmission](../physics/transmission.md#unitcommitmentenv-scuc-for-the-tso-task)）。TSO benchmark 与论文中关于 `\mathbf{c}_t` 的定义（Appendix E.2）都只采用前两个通道。

benchmark 的主目标是 episode 聚合量

\[
J^{\mathrm{op}} = \sum_{t=0}^{T-1} C^{\mathrm{op}}_t
\]

在结果中记为 `total_operating_cost`。这个 episode 指标才是排行榜使用的目标量，和训练时逐步返回的 reward 不是同一个层级。

## Baselines

| 名称 | 说明 |
| --- | --- |
| `all_on` | 所有机组始终开机，仅通过 OPF 做 dispatch；可看作较差成本上界 |
| `merit_order` | 按 [merit order（优先顺序）](../concepts/power-systems-primer.md) 从低到高启机直至覆盖 demand 与 reserve，再用 OPF 做 dispatch；工程上常见的 **规则型** SCUC 近似，也是一个强 **成本参考**（在简化设定下常可视为松的下界，而非全局最优） |

两个 baseline 都是确定性的、**无学习** 的 rollout，不需要训练，CPU 上即可快速跑完。它们与 RL 结果共用同一个 `manifest.json` 索引。

TSO 用 [NormScore](../glossary.md) 做归一化评分：

\[
\text{NormScore} = \frac{\text{cost}_{\text{all\_on}} - \text{cost}_{\text{algo}}}{\text{cost}_{\text{all\_on}} - \text{cost}_{\text{merit\_order}}}
\]

因此 `all_on` 得分为 0，`merit_order` 得分为 1，强 RL 策略可以超过 1。

## RL 算法 {#tso-rl-algos}

| Algo | Preset | 说明 |
| --- | --- | --- |
| `ppo` | `tso-uc` | 无约束 baseline |
| `ppo_lagrangian` | `tso-scuc-safe` | 带 thermal-overload 与 reserve-shortfall 成本的 PPO-Lagrangian CMDP baseline |
| `ppo_penalty_l10` | `tso-scuc-safe` | Penalty PPO 消融：固定 \(\lambda_{\mathrm{pen}}=10\)（有效系数 1e-3；欠惩罚） |
| `ppo_penalty_l100` | `tso-scuc-safe` | Penalty PPO 消融：固定 \(\lambda_{\mathrm{pen}}=100\)（有效系数 1e-2；量级相当） |
| `ppo_penalty_l1000` | `tso-scuc-safe` | Penalty PPO 消融：固定 \(\lambda_{\mathrm{pen}}=1000\)（有效系数 1e-1；过惩罚） |

当前 paper-facing TSO campaign 的主要学习行是 `ppo` 与 `ppo_lagrangian`。历史 Sauté / penalty sweep 可作为 appendix 分析，但不属于当前 primary campaign leaderboard。

### Penalty PPO 消融 {#tso-penalty-ablation}

`ppo_penalty_l*` 在训练前对 reward 施加固定的惩罚整形：

$$
r'_t = r_t - \lambda_{\mathrm{pen}} \, \lambda_{\mathrm{reward}} \sum_i c_{i,t}
$$

其中，\(r_t\) 是基础任务 reward，\(\lambda_{\mathrm{reward}}\) 是 reward 缩放系数，\(\lambda_{\mathrm{pen}}\) 是固定惩罚权重，\(c_{i,t}\) 是第 \(t\) 步的物理 CMDP cost 通道。在实现中，它们分别对应环境原始 reward、`reward_scale = 1e-4`（来自 `task.yaml`）、penalty ablation 配置里的固定惩罚权重，以及同一个 `UnitCommitmentEnv` 选出的 CMDP cost 通道。

实现层由 `PenaltyRewardWrapper` 包装同一个 `UnitCommitmentEnv`；底层训练器不变，仍为通过 [Rejax](../training/trainers.md) 运行的标准 PPO。这里的 Rejax 是本项目单智能体 PPO 路径使用的训练后端。三个 \(\lambda_{\mathrm{pen}}\) 值覆盖欠惩罚 / 量级相当 / 过惩罚三个区间：

| Key | \(\lambda_{\mathrm{pen}}\) | 有效系数 | 区间 |
| --- | --- | --- | --- |
| `ppo_penalty_l10` | 10 | 1e-3 | 最大惩罚约 0.5，而单步 reward 约为 -0.5；安全信号被淹没 |
| `ppo_penalty_l100` | 100 | 1e-2 | 最大惩罚约 5，与单步 reward 同量级；量级相当 |
| `ppo_penalty_l1000` | 1000 | 1e-1 | 最大惩罚约 50，远大于单步 reward；成本信号主导 |

在安全门控 leaderboard 上，三档都预期被 `ppo_lagrangian`（自适应对偶）超越，说明固定的 \(\lambda_{\mathrm{pen}}\) 无法可靠满足 zero-threshold 约束。这些结果属于 TSO appendix，不替代 `ppo` / `ppo_lagrangian` 的主结果。

## Eval splits

| Split | 说明 |
| --- | --- |
| `train` | 与训练相同窗口 |
| `iid` | 同一运行机制下的留出月份 |
| `load_stress` | 更高 demand 压力的 [OOD](../glossary.md#split) split |
| `line_tightening` | 更低线路热容量的 [OOD](../glossary.md#split) split |

这里的 OOD 指 [out-of-distribution](../glossary.md#split)，即有意施加的分布外测试设置。两个 OOD split 分别强调不同难点：一个提升负荷压力，另一个提升拥塞压力。

能力划分：

- `iid` 检验同分布留出窗口上的常规 day-ahead SCUC 能力；
- `load_stress` 检验 demand surge 下的备用充足性与成本控制；
- `line_tightening` 检验传输裕度变小时的拥塞感知 commitment 与 redispatch 能力。

解释方式：

- `iid` 强、`load_stress` 弱，通常表示 reserve 留得过于激进或过于脆弱；
- `iid` 强、`line_tightening` 弱，通常表示策略依赖无拥塞下的 merit-order 模式，缺乏 transmission-security 鲁棒性；
- `iid` 本身就弱，则说明方法在标准 SCUC 上都还不具竞争力，更不用说 OOD。

## Metrics

| Key | 说明 |
| --- | --- |
| `total_operating_cost` | episode 聚合量 \(J^{\mathrm{op}} = \sum_t C^{\mathrm{op}}_t\) |
| `feasibility_rate` | 没有热越限且没有备用不足的步数占比 |
| `thermal_violation_rate` | 评估汇总指标：\(C^{\mathrm{th}}_t > 0\) 的步数占比 |
| `reserve_shortfall_rate` | 评估汇总指标：\(C^{\mathrm{res}}_t > 0\) 的步数占比 |
| `commitment_switching_frequency` | episode 层面的切换统计（每个 episode 的 off-to-on 次数） |
| `norm_score` | 归一化的 episode 成本分数 |
| `ood_degradation` | `NormScore(IID) - NormScore(load_stress)` |

Reward-hacking 防护规则：

- 主排行榜不是只看成本；
- 一个主 split 结果若要进入 leaderboard，还必须满足声明的安全阈值（`reserve_shortfall_rate <= 0` 且 `thermal_violation_rate <= 0`）。

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

完整流程：

```bash
python benchmarks/tso/run_all.py
```

当前 5-seed [campaign](../glossary.md#campaign) 使用：

```bash
python benchmarks/tso/run_all.py --seeds 0,1,2,3,4
```

!!! warning "不要跨 campaign 混表"
    不同 [campaign](../glossary.md#campaign) 的记录用的是不同 reset-bank、不同 OOD 协议、不同代码版本，**不要**把它们混进同一张结果表里。

    出表时务必加一个 `campaign_start_iso` filter，让一张表只代表一次 campaign。否则 leaderboard 数字看起来对得上，实际背后跑的不是同一个 benchmark。

## 输出文件

```text
benchmarks/tso/results/
  manifest.json
  runs/
  artifacts/
  summary/latest.json         <- 聚合指标，外加 protocol_status、
                                 leaderboard_primary_split、split_taxonomy
  figures/
    normscore_bars.{pdf,png}
    gantt_commitment.{pdf,png}
    cost_decomposition.{pdf,png}
    learning_curves.{pdf,png}
```

## 交叉引用

- [Physics → Transmission](../physics/transmission.md#unitcommitmentenv-scuc-for-the-tso-task)
- [API → Unit commitment](../api/grid-uc.md)
- [Training → Presets](../training/presets.md)
