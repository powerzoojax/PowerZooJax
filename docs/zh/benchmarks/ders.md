# DERs - 异构多智能体电压调节

DERs benchmark 是一个 cooperative multi-agent reinforcement learning（MARL）任务，运行在配电 feeder 上。12 个异构智能体在 `case141` 上协同工作，把电压维持在 `[0.94, 1.06]` 标幺值（`p.u.`）区间内，同时尽量降低网络损耗。这 12 个智能体由 4 个 battery、4 个 PV inverter 和 4 个 flexible load 组成，分布在不同母线上。像 DER、`p.u.`、电压调节、径向 BFS 潮流这些术语，可先看 [Power systems primer](../concepts/power-systems-primer.md)。

本页描述的是 benchmark 合约。有一点需要先说明：这个任务在定义层面暴露了明确的 constraint specification，但当前 benchmark 训练路径仍然是 reward-shaped IPPO，而不是带在线对偶更新的 constrained MARL optimizer。

## 一眼看懂

- 物理 env：`case141` 上挂 12 个异构资源的配电网环境。
- benchmark 任务：12 智能体 cooperative 的电压调节与降损 MARL 任务。
- 实际训练对象：带 typed parameter sharing 的 IPPO variants，加固定 reward shaping；`ippo_lagrangian` 在结果面中保留，但不是 DERs headline。
- 主要 leaderboard quantity：`mean_p_loss_mw`。
- 安全门槛：任务配置里的零电压违反目标。

如果你是第一次读 DERs，最稳妥的顺序是：先看摘要，再看任务定义表，再看 reward 与 cost，最后再看训练和评估流程。这个顺序尤其重要，因为任务定义本身是 constraint-aware 的，但当前正式训练路径仍然是 reward-shaped。

## 任务定义

| 字段 | 内容 |
| --- | --- |
| 任务类型 | 12 智能体 cooperative Dec-POMDP，本地观测 |
| Agents | 12（`4 battery + 4 pv + 4 flexload`） |
| State \(\mathcal{S}\) | 母线电压、支路潮流、资源运行状态、时间相位 |
| Observation \(\mathcal{O}_i\) | 本地 K-hop 邻域，15 维（`1 + K + 3 + 2 + 5`，其中 `K=4`） |
| Action \(\mathcal{A}_i\) | 每个 agent 一个 `Box(2)` |
| Transition \(\mathcal{P}\) | 资源动作改变注入与柔性负荷，然后在 `case141` 上求解配电潮流并检查安全性 |
| Reward \(r_t\) | 共享 team reward，\(r_t = -w_{\mathrm{loss}}\, P_{\mathrm{loss}, t}^{\mathrm{MW}}\) |
| Cost \(\mathbf{c}_t\) | 任务约束向量 \(\mathbf{c}_t = \left(C_t^{\mathrm{volt}}, C_t^{\mathrm{therm}}, C_t^{\mathrm{resource}}\right)\) |
| Threshold \(\mathbf{b}\) | 任务级安全 gate 是 `voltage_violation_rate <= 0.0`（单一电压通道，来自 `task.yaml::safety_thresholds`）；IPPO-Lagrangian 配置另有按通道的 `cost_thresholds: [0.0, 0.0, 0.0]`，分别对应 `(voltage, thermal, resource)`，但那是训练侧的对偶预算，不是 leaderboard 的安全 gate |
| Discount \(\gamma\) | `0.995` |
| Horizon \(T\) | 48 steps x 30 min = 24 h |
| Initial \(\mu_0\) | `case141` 上由真实数据 split 驱动的 episode |

`Dec-POMDP` 指 decentralized partially observable Markov decision process：每个 agent 只看局部观测，但所有 agent 共享一个 team-level return。

## 底层物理

该任务建立在 [Physics -> Distribution](../physics/distribution.md) 和 [Physics -> Resources](../physics/resources.md) 所描述的配电与资源层之上。正式 benchmark 使用 `case141`，并在上面挂接 12 个异构资源。

在物理 env 层，电压安全、热过载和资源侧违反是分开跟踪的。DERs task 在任务层保留了这些通道：

\[
\mathbf{c}_t = \left(C_t^{\mathrm{volt}},\, C_t^{\mathrm{therm}},\, C_t^{\mathrm{resource}}\right)
\]

但当前 benchmark 训练路径并不会在这个向量上运行在线 constrained MARL。相反，训练 preset 采用的是固定电压 penalty 的 reward shaping。也就是说：DERs 在任务定义层面是 constraint-aware 的，但当前正式 trainer 仍然是 shaped-reward MARL 路径。

## Benchmark 任务参数

| 参数 | 内容 |
| --- | --- |
| Case | `case141` |
| Agents | 12（`4 battery + 4 pv + 4 flexload`） |
| Action per agent | `Box(2)` |
| Observation per agent | 本地 K-hop 邻域，约 15 维 |
| Episode | 48 steps x 30 min = 24 h |
| Voltage limits | `[0.94, 1.06]` p.u. |
| Primary metric | `mean_p_loss_mw`（`lower_is_better`） |
| Safety gate | 已定版任务配置中的零电压违反目标 |

## Agent 部署

| Type | Count | Buses | 说明 |
| --- | --- | --- | --- |
| Battery | 4 | 9, 55, 17, 122 | 0.10 MW / 0.30 MWh |
| PV | 4 | 6, 73, 72, 82 | 0.20 MW nameplate |
| FlexLoad | 4 | 41, 70, 135, 24 | 0.10 MW curtail / shift cap |

每个 agent 都是 2 维动作，但物理含义依资源类型而异。benchmark 训练时使用 typed parameter sharing，因此 battery、PV inverter 和 flexible load 不需要共用同一个 policy head。关于 `IPPO` 和 typed parameter sharing，可见 [Training -> Trainers](../training/trainers.md)。

## Reward 与 cost

共享的 team reward 是 feeder 损耗目标：

\[
r_t = -w_{\mathrm{loss}}\, P_{\mathrm{loss}, t}^{\mathrm{MW}}
\]

其中 \(P_{\mathrm{loss}, t}^{\mathrm{MW}}\) 是时刻 \(t\) 的总网络有功损耗，12 个 agent 共享同一个 reward。

任务层约束向量为

\[
\mathbf{c}_t = \left(C_t^{\mathrm{volt}},\, C_t^{\mathrm{therm}},\, C_t^{\mathrm{resource}}\right)
\]

它与 task `ConstraintSpec` 以及底层配电 env 的诊断通道一致。

当前 benchmark 训练路径随后会用固定电压 penalty 对 reward 做 shaping，而不是学习对偶变量：

\[
r^{\mathrm{train}}_t = r_t - \lambda_{\mathrm{volt}} C_t^{\mathrm{volt}}
\]

其中 \(\lambda_{\mathrm{volt}}\) 对应配置里的 `voltage_penalty`。在当前定版的 benchmark 设置里：

- `ippo` 使用基础 DERs 训练配置，对电压违反施加中等强度的固定 penalty。
- `ippo_safe` 使用更强的固定电压 penalty。

所以这里的 `safe` 指的是更强的固定 shaping，而不是单独的 constrained optimizer。

## 这里的 “safe” 是什么意思

`ders-medium-safe` 并不会改变底层物理任务。它用的是同一个 env、同一组 agents、同一套任务约束通道。变化只在训练目标上：

- `ippo`：带中等固定电压 penalty 的 reward-shaped team training（`voltage_penalty=4.0`）
- `ippo_safe`：带更强固定电压 penalty 的 reward-shaped team training（`voltage_penalty=8.0`，是无约束 4.0 的两倍；论文 Appendix H.2 把这一基线称为 IPPO-rs）

因此，DERs benchmark 页不应把当前正式训练路径写成在线 CMDP / Lagrangian MARL。

## Baselines

| 名称 | 说明 |
| --- | --- |
| `no_control` | 所有 DER 动作为 0 |
| `volt_droop` | 带本地无功 / 削减响应的 voltage-droop 规则 baseline |

`volt_droop` 是 benchmark summary 中更强的手工规则锚点。

## 算法

| Algo | Preset | 说明 |
| --- | --- | --- |
| `ippo` | `ders-medium` | typed-parameter-sharing cooperative MARL baseline |
| `ippo_safe` | `ders-medium-safe` | 同一 MARL 路径，但使用更强的固定电压 penalty |
| `ippo_lagrangian` | `ders-medium-safe` | 保留在正式矩阵中的 CMDP variant；由于 PV shift 仍然脆弱，不作为 headline row |

隐藏层 `(128, 128)`；gamma `0.995`；总训练步数 `10e6`；Phase-1 JAX/GPU 报告使用 5 个 seeds。mandatory seed-0 backend/device matrix 已完成，并在 `train`、`iid`、`voltage_tightening`、`pv_penetration_shift` 和 `load_stress` 上保留 official eval rows。

## Eval splits

| Split | 说明 |
| --- | --- |
| `train` | 训练分布 |
| `iid` | 留出的 in-distribution episodes |
| `voltage_tightening` | 更严格的电压带 |
| `pv_penetration_shift` | PV 渗透率变化 |
| `load_stress` | 更高负荷压力 |

3-phase wrapper 仍然可用于评估实验，但不属于正式 benchmark split 列表。

## Metrics

DERs 页里的指标最好按层理解：

| 层级 | Key | 说明 |
| --- | --- | --- |
| Step reward | `reward` | 共享的 per-step team reward |
| Step constraint channels | `voltage_violation`, `thermal_overload`, `resource` | 任务层 cost 向量 |
| Episode aggregate | `total_reward` | 累积共享 reward |
| Episode aggregate | `total_cost` | 累积连续型安全 cost 诊断 |
| Episode aggregate | `mean_p_loss_mw` | 平均网络有功损耗 |
| Episode aggregate | `voltage_violation_steps` | 电压越界步数 |
| Relative evaluation summary | `loss_reduction_pct` | 相对 `no_control` 的降损比例 |
| Relative evaluation summary | `cost_reduction_pct` | 相对 `no_control` 的连续安全 cost 降低比例 |
| Relative evaluation summary | `NormScore` | 相对固定 baselines 的 benchmark 归一化分数 |

!!! note "convergence target 用的是损耗，不是 reward"
    当前定版 DERs 的 convergence target 基于 `mean_p_loss_mw`，**不是** `total_reward`。这是有意为之：DERs benchmark 把物理网损视为主要 leaderboard quantity，所以收敛判定也直接绑定到这一物理量上。看 leaderboard 时不要拿 episode reward 做横比。

## Quick start

```bash
python benchmarks/ders/run_all.py --only baselines --seeds 0 1 2 3 4
python benchmarks/ders/run_all.py --only train --algos ippo ippo_safe ippo_lagrangian --seeds 0 1 2 3 4
python benchmarks/ders/run_all.py --only eval
python benchmarks/ders/run_all.py --only summarize
python benchmarks/ders/run_all.py --only plots
```

共享 workflow 术语见 [Benchmark workflow glossary](../glossary.md)。

## 交叉引用

- [Physics -> Distribution](../physics/distribution.md)
- [Physics -> Resources](../physics/resources.md)
- [API -> RL MARL](../api/rl-marl.md)
- [Training -> Trainers](../training/trainers.md)
