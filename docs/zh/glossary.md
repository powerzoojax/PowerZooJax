# Benchmark 工作流术语表

本页解释 PowerZooJax 中 benchmark 页面、结果表和实验配置里反复出现的术语。

如果你还不熟悉电力系统概念，比如 OPF、SOC、PTDF，可以先看 [Power systems primer](concepts/power-systems-primer.md)。

---

## Task

benchmark task 是一个定义清楚的电力系统控制问题，它固定了：

- **Case**：电网拓扑和参数
- **Agents**：决策者的数量和类型
- **Horizon**：episode 的时间步长度
- **RL paradigm**：单智能体 safe RL、协作式 MARL、竞争式 MARL 等
- **Reward and cost**：优化目标和安全违约通道
- **Splits**：用于评测的 train / iid / OOD 数据设置

PowerZooJax 目前有五个核心 benchmark tasks：TSO、DSO、DERs、GenCos 和 DC Microgrid。

每个 task 都定义在 `benchmarks/<task>/configs/task.yaml` 中，并且必须先通过 `seed0_readiness --enforce`，才能进入完整的 multi-seed 正式运行。

## Campaign

campaign 指的是一轮有边界的 benchmark 运行：在同一版冻结的 task setup 下，同一轮 baseline、train、eval、summary 和 figures 运行结果共同组成一个 campaign。

实践里，开始一个 new campaign 往往意味着你明确重置当前 benchmark 工作流，并声明旧的 manifest 记录不再算作“当前这一轮”的正式证据。所以 readiness 检查常常问的是“当前 campaign 内是否完成”，而不是“历史上是否做过”。

可以把 campaign 理解成一句话：“哪些 runs 属于当前这一轮 benchmark。” 后面的 [Seed-0 readiness](#seed-0-readiness)、[campaign seed budget](#campaign-seed-budget) 和 submission-grade rerun，都是围绕这个有范围的当前轮次来说的。

## Seed-0 readiness

`seed0_readiness` 是 benchmark 在正式进入 multi-seed 运行前使用的那一道 seed-0 就绪门槛。

它的含义是：当前 campaign 范围内，seed-0 这条链路必须已经完整跑通要求的 baseline、train、eval 和 summary。它是工作流有效性检查，不等于“这个方法的科学结果已经足够强”。

## Multi-seed benchmark run

multi-seed benchmark run 指的是：围绕同一个 task，在多个随机种子上做正式评测，并覆盖要求的全部 splits。它只会在 seed-0 reference run 完成后开始，结果记录在 `benchmarks/<task>/results/manifest.json` 中。

## Campaign seed budget

campaign seed budget 指当前这一轮 benchmark campaign 实际打算使用的随机种子数量。

它描述的是“这轮正在运行的实验预算”，可以小于最终论文版的种子数。比如某个任务当前先跑 3 个 seeds 做正式迭代，之后再补跑 5 个 seeds 的 submission-grade 结果。

## Submission-grade minimum

submission-grade minimum 指最终论文投稿级 benchmark 表格或图里，至少应该使用多少个 seeds。

它通常比当前 campaign seed budget 更严格。这个术语是在告诉读者：当前阶段可能先用较少 seeds 推进实验，但最终 benchmark 结论至少要达到这个种子数标准。

## Run Record

run record 是一次 training、baseline 或 evaluation 的保存结果。它通常包括：

- 基本信息：task、algorithm、seed、split、run ID
- 指标：episode reward、cost、constraint violations、normalized score
- 状态与完成时间

所有 run records 会被汇总成任务结果，再用于生成 summary 和 figures。

## Split

split 是训练或评测时使用的一种数据设置：

- **train**：训练所用的完整分布
- **iid**：与训练大体同分布的留出测试集
- **OOD (Out-of-Distribution)**：有意引入分布偏移的测试设置。论文里按任务给出的标准 split 名包括：TSO 的 `load_stress` 和 `line_tightening`；DERs 的 `voltage_tightening`、`pv shift`、`load_stress`；GenCos 的 `demand shift`、`renewable shock`；DCMG 的 `cooling_stress`、`renewable_drought`、`workload_swap`、`workload_shock`、`dg_derating`、`sla_tighten`

任何在 `train` split 上训练出的 agent，都应该在该任务的全部 splits 上评估。

## Baseline

baseline 是不用学习的参考策略，用来给性能对比提供参照：

- **no_control**：被动运行，动作恒为 0
- **rule-based（基于规则）**：按固定的 if–then 或时刻表执行，例如分时段 TOU 削减/转移、局部 **Volt–Var / droop** 响应，或**优先顺序（merit order）** 开停机列表
- **heuristic（近似求解）**：在可接受时间内用近似方法换最优性（如市场中分段 ED 的启发式出清），与上一条「按时钟与阈值运行的规则基线」不是同一类说法
- **simple greedy**：只看单步、不做长期规划的贪心策略

许多 baseline 是确定性的、可复现的，并提供用于归一化评分的 IQM 参考值。

## Training run

training run 指在训练分布上学习一个策略：

- 从固定随机种子开始
- 通过多个并行环境收集经验
- 用 PPO、IPPO 或 safe-RL 变体更新策略
- 保存训练后的策略，供后续评估使用

同一个 task 的训练配置应该保持一致，不应随意更改。

## Evaluation run

evaluation run 指在某个 split 上评估一个已训练策略或 baseline：

- 加载 checkpoint 或构造 baseline
- 在该 split 对应的数据分布上跑 episodes
- 记录 reward、cost 和 constraint 指标
- 用统一格式保存结果

如果种子相同，evaluation 应该可以重复得到一致结果。

## Bootstrap CI

bootstrap CI 是 bootstrap confidence interval 的缩写，通常写成 95% CI 之类的形式。

它表示：通过对已有 runs 或 episodes 做“有放回重采样”，估计某个统计量（比如 mean 或 IQM）的不确定性区间。在 benchmark 报告里，它用来表达跨 seeds 汇总结果到底有多稳定。

## Primary metric

primary metric 是写进 task config 和报告流程里的任务主指标。它表示 benchmark 默认用哪个标量作为该任务最主要的汇总与比较对象。

它通常会出现在主表里，但不一定等同于训练器真正用来更新策略的量。比如：

- TSO：`total_operating_cost`
- DERs：`mean_p_loss_mw`
- GenCos：`total_profit`
- DC Microgrid：`episode_reward`

所以，primary metric 是报告契约里的术语，不等于“trainer 实际在优化什么”。

## Convergence target

convergence target 是当前 training-and-summary pipeline 用来判断一次 run 是否达到 benchmark 目标的那个标量。

它是工作流层的概念，不一定是物理上最直观的量。有些任务里它和主要物理指标一致；有些任务里它更像 trainer-facing return。

例如：

- 在 DSO 里，当前 convergence target 是 `total_reward`，而更直接可解释的物理汇总量是 `total_loss_mwh`。
- 在 DC Microgrid 里，当前 convergence target 是 `episode_reward`，它本身已经包含 reward shaping。

## Leaderboard quantity

leaderboard quantity 是在安全门槛或审计门槛通过之后，benchmark 在主 split 上真正拿来给方法排序的那个标量。

这个术语很重要，因为一个方法可能会同时对应三层不同的量：

- 训练时优化的 per-step reward
- benchmark 流程里监控的 convergence target
- leaderboard 最终排序用的 episode-level quantity

例如：

- TSO 训练时用 per-step reward，但主 split 排名看的是真正的 episode `total_operating_cost`。
- DERs 当前把 `mean_p_loss_mw` 视为主要 leaderboard quantity，尽管训练路径是 reward-shaped IPPO。

## Normalized Score（NormScore）

NormScore 用来把不同任务的原始指标变成可比较的无量纲分数：

$$
\text{NormScore} = \frac{\text{agent performance} - \text{baseline floor}}{\text{baseline ceiling} - \text{baseline floor}}
$$

- **Baseline floor**：最弱 baseline 的 IQM，例如 `no_control`
- **Baseline ceiling**：最强 baseline 的 IQM
- **Agent performance**：跨 seeds 和 split episodes 汇总后的 IQM

NormScore 高于 `1.0` 表示比最强 baseline 还好。低于 `0.0` 表示比最弱 baseline 还差。

NormScore 是相对诊断指标，不是默认 headline 排名指标。summary 会暴露 `norm_score_status`、anchor 数值和 anchor-gap warning；只有 `norm_score_status=ok` 的 row 才能用于 NormScore 比较。

在 benchmark 页面里，`NormScore` 通常是概念名，而 `norm_score` 往往是写进 `manifest.json`、summary 或结果表的 metric key。

## Manifest

manifest 是某个任务的总记录文件。每次 training、baseline 和 evaluation 都会写入一条记录，所有记录汇总成一个文件。这个文件是 summary 和 figures 的唯一可信来源。

## Summary 和 figures

- **Summary**：跨 seeds、splits 和 algorithms 的汇总统计
- **Figures**：例如 reward 曲线、constraint 满足情况热图等可视化结果

它们都应从 manifest 自动生成，而不应手工编辑。

## Cross-backend comparability

如果要公平比较不同实现版本，例如 PowerZoo 和 PowerZooJax，那么以下条件必须一致：

- task 定义
- 数据来源
- 随机种子
- reward 和 cost 公式
- 安全约束
- 训练时长

只要这些条件有任何一项不一致，就不应该把结果放进同一个直接对比表里。

## JAX execution model

PowerZooJax 环境使用不可变状态和纯函数设计，以便高效运行在 GPU 上。这让大规模并行训练和评估成为可能。

## Seed

seed 是一个固定随机数，用来保证结果可复现。它会影响策略初始化、环境随机性以及数据切分打乱。benchmark 报告通常会使用多个 seeds，并对结果做汇总。
