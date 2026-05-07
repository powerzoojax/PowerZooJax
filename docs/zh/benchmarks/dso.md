# DSO - 配电网损耗最小化

DSO benchmark 是一个单智能体的配电网需求响应任务。一个集中式控制器在 IEEE 33-bus Baran-Wu 径向配电网上调度 6 个柔性负荷，在把母线电压维持在严格的 `[0.94, 1.06]` 标幺值（`p.u.`）区间内的同时，尽量降低网络损耗。关于径向 feeder、后推前代潮流（BFS）和 `p.u.` 的背景，可见 [Power systems primer](../concepts/power-systems-primer.md)。

本页是 DSO 任务说明：定义 benchmark 合约、固定采用的 split、reward 与 CMDP 语义，以及输出结果。共享实验术语见 [Benchmark workflow glossary](../glossary.md)。

## 一眼看懂

- 物理 env：IEEE `case33bw` 上挂 6 个 `FlexLoad` 的 `DistGridEnv`。
- benchmark 任务：在电压约束下做网络损耗最小化的单智能体需求响应。
- 实际训练对象：在 DSO task wrapper 上训练 PPO、SAC、Sauté PPO 或 PPO-Lagrangian，而不是只对裸 grid env 训练。
- 主要 benchmark quantity：`total_reward` 是当前收敛目标，`total_loss_mwh` 是最直接的物理 episode 指标。
- 安全门槛：任务配置要求零电压违反率。

第一次阅读 DSO 时，建议按这个顺序走：先看上面的摘要，再看 MDP / CMDP 表，再看 reward 与 cost，小节都看明白后再看命令示例。这样能把物理任务本身和训练、汇总流程区分开。

## MDP / CMDP 定义

| 字段 | 内容 |
| --- | --- |
| MDP class | MDP (`dso-nflex`) / CMDP (`dso-nflex-safe`)，任务层选中的约束为 `("voltage_violation",)` |
| Agents | 1（集中式） |
| State \(\mathcal{S}\) | 母线电压、支路潮流、节点负荷、柔性负荷状态、时间相位 |
| Observation \(\mathcal{O}\) | `Box(195)`；精确字段布局见 [Physics -> Distribution](../physics/distribution.md#distgridenv-balanced-radial-feeder) |
| Action \(\mathcal{A}\) | `Box(12) = 6 x [curtail, shift_out]` |
| Transition \(\mathcal{P}\) | 柔性负荷动作先改变 feeder demand，然后在 `case33bw` 上求解平衡径向 BFS 潮流 |
| Reward \(r_t\) | \(r_t = -w_{\mathrm{loss}}\, P_{\mathrm{loss}, t}^{\mathrm{MW}}\) |
| Cost \(c_t\) | \(c_t = \left(C_t^{\mathrm{volt}}\right)\) |
| Threshold \(d\) | `dso-nflex-safe` 使用 `cost_thresholds = (0.0,)`；`dso-nflex` 不适用 |
| Discount \(\gamma\) | `0.995` |
| Horizon \(T\) | 48 steps x 30 min = 24 h |
| Initial \(\mu_0\) | 从 Ausgrid FY25 负荷 profile 中采样得到的数据驱动 episode |

`CMDP` 指 constrained Markov decision process：reward 是经济训练信号，安全违反通过单独的 cost 通道报告，而不是重新塞回 reward。对 DSO 来说，任务层 CMDP 只约束电压。

## 底层物理

该 benchmark 建立在 [`DistGridEnv`](../physics/distribution.md#distgridenv-balanced-radial-feeder) 之上：这是一个运行在 IEEE 33-bus Baran-Wu 配电网上、并挂接了 [FlexLoad bundle](../physics/resources.md#flexible-load-flexloadenv) 的平衡径向配电环境。本任务里控制器不调度电池，也不直接注入 DER，只调度柔性需求。

它的任务直觉很简单：把负荷从压力较大的时段移开或削减，通常会降低支路电流、减轻压降，并进一步降低阻性 \(I^2R\) 损耗。因此 DSO 是一个很干净的 demand response benchmark；而 DERs 则在此基础上继续引入异构分布式资源。

在 env 层，`DistGridEnv` 总是暴露完整的固定形状 cost 向量

\[
\mathbf{c}_t^{\mathrm{env}} = \left(C_t^{\mathrm{volt}},\, C_t^{\mathrm{therm}},\, C_t^{\mathrm{resource}}\right)
\]

其名称为 `("voltage_violation", "thermal_overload", "resource")`；详见 [Physics -> Distribution](../physics/distribution.md) 和 [API -> Distribution](../api/distribution.md)。DSO benchmark 并不改变这些 env 语义，而是在任务层只选择电压通道作为 CMDP 约束：

\[
\mathbf{c}_t = \left(C_t^{\mathrm{volt}}\right)
\]

这一点需要明确区分：DSO 的训练与安全报告发生在 task 层，但底层 grid env 仍然计算完整的诊断向量。

## Benchmark 任务参数

| 参数 | 内容 |
| --- | --- |
| Case | `case33bw` |
| Resources | 6 个 FlexLoad，分布在 3 段 feeder 上 |
| Episode | 48 steps x 30 min = 24 h |
| Action space | `Box(12) = 6 x [curtail, shift_out]` |
| Observation | `Box(195)` |
| Voltage limits | `[0.94, 1.06]` p.u. |
| Data source | Ausgrid zone-substation load (FY25) |
| Official eval split | `iid` |
| Safety gate | 已定版任务配置中要求 `voltage_violation_rate <= 0.0` |

这个任务没有 battery。这正是它与 DERs 的主要 benchmark 区别：DSO 单独隔离“纯 demand response”的价值。

## 资源布局

`make_dso_flexload_bundle(case)` 会把 6 个柔性负荷固定放在如下母线上：

| Device | Bus | `curtail_cap_mw` | `shift_cap_mw` |
| --- | --- | --- | --- |
| FL_A1 | 6 | 0.15 | 0.15 |
| FL_A2 | 14 | 0.10 | 0.10 |
| FL_A3 | 18 | 0.10 | 0.10 |
| FL_B1 | 22 | 0.08 | 0.08 |
| FL_C1 | 28 | 0.12 | 0.12 |
| FL_C2 | 33 | 0.10 | 0.10 |

这些设备跨越三段 feeder，所以策略必须做全局协调，而不是只解决一个纯局部的电压控制问题。

## 动作与观测

每个设备都有两个控制量：当前步直接削减负荷，或把当前负荷的一部分延后到短期缓冲区。benchmark observation 综合了 feeder 电气状态、当前负荷状态、时间特征，以及每个柔性负荷设备的状态。

观测向量的精确字段顺序与归一化规则，见 [Physics -> Distribution](../physics/distribution.md#distgridenv-balanced-radial-feeder)。

## Reward 与 CMDP cost

step reward 是损耗最小化信号：

\[
r_t = -w_{\mathrm{loss}}\, P_{\mathrm{loss}, t}^{\mathrm{MW}}
\]

其中 \(P_{\mathrm{loss}, t}^{\mathrm{MW}}\) 表示时刻 \(t\) 的 feeder 总有功损耗，\(w_{\mathrm{loss}}\) 是 reward 权重（实现里对应 `loss_penalty_weight`；见 [API -> Distribution](../api/distribution.md)）。

任务层 CMDP cost 为

\[
\mathbf{c}_t = \left(C_t^{\mathrm{volt}}\right)
\]

其中 \(C_t^{\mathrm{volt}}\) 是电压越出允许区间 `[0.94, 1.06]` p.u. 的母线数量。换句话说，DSO 采用的是“计数型”的电压安全通道，而不是“离限制多远”的连续距离惩罚作为 benchmark CMDP cost。

在 episode 层，有两个不同的汇总量，而且不应该混为一谈：

\[
R_{\mathrm{ep}} = \sum_{t=0}^{T-1} r_t
\]

\[
J_{\mathrm{loss}} = \sum_{t=0}^{T-1} P_{\mathrm{loss}, t}^{\mathrm{MW}} \Delta t
\]

- \(R_{\mathrm{ep}}\) 对应 `total_reward`，也是当前 benchmark pipeline 固定采用的 convergence target。
- \(J_{\mathrm{loss}}\) 对应 `total_loss_mwh`，是更直接可解释的物理 episode 损耗指标。

因此，这个 benchmark 在训练时使用 reward 形式的损耗目标，但在报告中会同时保留 trainer-facing return 和物理能量损耗汇总。

## Baselines

| 名称 | 说明 |
| --- | --- |
| `no_control` | 所有 FlexLoad 动作为 0；没有主动调度 |
| `tou` | 分时规则 baseline：在固定时钟峰值窗口中做削减与移位 |
| `droop` | 电压 droop 规则 baseline：根据母线电压相对固定 band 的偏离做确定性的局部响应 |

DSO 的 NormScore 定义在物理损耗指标上：

\[
\mathrm{NormScore} = \frac{J_{\mathrm{loss}}^{\mathrm{no\_control}} - J_{\mathrm{loss}}^{\mathrm{algo}}}{J_{\mathrm{loss}}^{\mathrm{no\_control}} - J_{\mathrm{loss}}^{\mathrm{best\ baseline}}}
\]

这里的最佳固定非学习 baseline 是规则 baseline 中更强的那个，当前 summarization pipeline 里是 `droop`。因为这是一个损耗最小化任务，所以网络损耗越低，`NormScore` 越高。

## RL 算法

| Algo | Preset | 说明 |
| --- | --- | --- |
| `ppo` | `dso-nflex` | 通过 [Rejax](../training/trainers.md) 运行的标准 PPO |
| `sac` | `dso-nflex` | 通过 [Rejax](../training/trainers.md) 运行的 SAC |
| `saute_ppo` | `dso-nflex-safe` | 使用电压安全 cost 通道的 Sauté PPO |
| `ppo_lagrangian` | `dso-nflex-safe` | 带零电压违反预算的 PPO-Lagrangian CMDP |

总训练步数 `3e6`；5 个 seeds；开启 observation normalization。Phase-2 backend/device 行也是 5-seed IID 报告行；DSO 不是 execution-scaling primary task。

## Eval splits

| Split | 说明 |
| --- | --- |
| `iid` | 当前 Ausgrid reset-bank 协议下的正式留出评估 episodes |

!!! caution "DSO 当前正式只跑 `iid`"
    当前冻结 DSO 配置的 executable truth 是 `eval_splits: [iid]`。

    - **不要**把历史 OOD split 名称（previous revision 留下的）当作本 revision 的正式 DSO 结果
    - **不要**把 DSO 当成 execution scaling 的 primary task（那一职责在其他 benchmark 上）

    跑 OOD 分析是允许的，但任何走出 `iid` 的数字都要明确标记为非定版结果。

## Metrics

DSO 页里的指标最好分成四层来理解：

| 层级 | Key | 说明 |
| --- | --- | --- |
| Step 训练信号 | `reward` | per-step reward，\(r_t = -w_{\mathrm{loss}} P_{\mathrm{loss}, t}^{\mathrm{MW}}\) |
| Step CMDP cost | `voltage_violation` | 任务层选中的电压安全通道 |
| Episode aggregate | `total_reward` | 累积 reward，\(R_{\mathrm{ep}}\)；当前定版的 convergence target |
| Episode aggregate | `total_loss_mwh` | 物理 episode 损耗，\(J_{\mathrm{loss}}\) |
| Episode aggregate | `mean_loss_mw` | 每步平均功率损耗 |
| Episode aggregate | `total_violations` | episode 总电压违反次数 |
| Episode aggregate | `total_curtailed_mwh` | 总削减能量 |
| Episode aggregate | `total_shifted_mwh` | 总延后能量 |
| Episode aggregate | `served_flex_ratio` | 已释放延后能量 / 已延后能量 |
| Relative evaluation summary | `network_loss_reduction_pct` | 相对 `no_control` 的损耗下降比例 |
| Relative evaluation summary | `peak_shaving_pct` | 相对 `no_control` 的峰值下降比例 |
| Relative evaluation summary | `NormScore` | 相对固定非学习 baselines 的归一化表现 |

关键点在于：`total_reward` 和 `total_loss_mwh` 虽然相关，但并不等价。前者是 trainer-facing return，后者是直接可解释的物理汇总量。

## Quick start

```bash
python benchmarks/dso/run.py baseline --seeds 0,1,2,3,4
python benchmarks/dso/run.py train --algo ppo --seed 0
python benchmarks/dso/run.py eval --run-id <run_id> --split iid
python benchmarks/dso/run.py summarize
python benchmarks/dso/run.py plots
python benchmarks/dso/phase2_analysis.py --seeds 0,1,2,3,4
```

## 输出文件

```text
benchmarks/dso/results/
  manifest.json
  runs/
  summary/latest.json
  phase2_backend_summary.json
  figures/
    normscore_bars.png
    learning_curves.png
    loss_reduction.png
    load_profiles.png
    phase2_backend_compare.png
```

## 交叉引用

- [Physics -> Distribution](../physics/distribution.md)
- [Physics -> Resources](../physics/resources.md#flexible-load-flexloadenv)
- [API -> Distribution](../api/distribution.md)
- [Training -> Trainers](../training/trainers.md)
- [Training -> Presets](../training/presets.md)
