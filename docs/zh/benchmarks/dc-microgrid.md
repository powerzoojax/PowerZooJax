# DC Microgrid - 多目标鲁棒微电网控制

DC Microgrid benchmark 是一个单智能体控制任务，运行在孤岛型、表后数据中心微电网上。控制器需要联合调度 workload、cooling、电池 dispatch 和柴油机备用功率，在能耗、燃料成本、碳排放和服务质量之间做权衡。物理环境和动作语义详见 [Physics -> Microgrid](../physics/microgrid.md)。

这页最需要区分的一点是：底层 env 会同时暴露分解后的 reward vector 和单独的 cost vector，但当前 benchmark 训练路径是 reward-shaped PPO，而不是原生 CMDP optimizer。

## 一眼看懂

- 物理 env：孤岛型数据中心微电网的 `DataCenterMicrogridEnv`。
- benchmark 任务：围绕 workload、cooling、电池和柴油备份做单智能体多目标控制。
- 实际训练对象：reward-shaped PPO / SAC，不是原生 constrained optimizer。
- 主要 leaderboard quantity：`episode_reward`。
- 审计门槛：SLA、过热和功率缺口通道仍会单独报告并审计。
- 当前 evidence：5-seed Phase-1 和 Phase-2 backend/device evidence 已完整；execution scaling 已完成，但与 algorithm-effect 报告分开。

这页最清晰的阅读顺序是：先看摘要，再看任务定义表，再看 reward 与 cost，最后再看运行命令。这样可以先建立微电网物理图景，再理解当前 reward-shaped benchmark workflow。

## 任务定义

| 字段 | 内容 |
| --- | --- |
| 任务类型 | 单智能体多目标微电网控制 |
| Agents | 1 |
| State \(\mathcal{S}\) | 任务队列、温度、电池 SOC、PV 与负荷 profile、柴油机状态、时间相位 |
| Observation \(\mathcal{O}\) | 18 维 |
| Action \(\mathcal{A}\) | `Box(5) = [train_sched, ft_sched, cooling_norm, batt_norm, dg_norm]` |
| Transition \(\mathcal{P}\) | workload、cooling、电池和柴油机决策共同更新数据中心与孤岛微电网平衡 |
| Reward \(r_t\) | 标量化的 energy / cost / carbon 目标 |
| Cost \(\mathbf{c}_t\) | \(\mathbf{c}_t = \left(C_t^{\mathrm{sla}}, C_t^{\mathrm{temp}}, C_t^{\mathrm{deficit}}\right)\) |
| Threshold \(d\) | task constraint spec 使用零阈值；但当前正式 trainer 仍是 reward shaping，而不是在线 CMDP 更新 |
| Discount \(\gamma\) | `0.99` |
| Horizon \(T\) | 288 steps x 5 min = 24 h |
| Initial \(\mu_0\) | 从 Google DC 2019 workload、solar 和 weather 中采样的数据驱动 episode |

## 底层物理

底层环境是 `DataCenterMicrogridEnv`，详见 [Physics -> Microgrid](../physics/microgrid.md)。它组合了：

- 数据中心 workload 与热模型
- 带显式可行域的 battery storage
- 外生 PV 发电
- 可调度 diesel generator
- 显式的孤岛型功率平衡约束

这里没有外部电网连接。如果 PV、电池和柴油机总和仍然覆盖不了数据中心负荷，缺口就会进入 power-deficit cost channel。

## Benchmark 任务参数

| 参数 | 内容 |
| --- | --- |
| Environment | `DataCenterMicrogridEnv` |
| Episode | 288 steps x 5 min = 24 h |
| Action space | `Box(5)` |
| Observation | 24 维 |
| Data source | Google DC workload + solar + outdoor temperature |
| Main table splits | `train`, `iid`, `cooling_stress`, `renewable_drought` |
| Appendix splits | `workload_swap`, `workload_shock`, `dg_derating`, `sla_tighten` |
| Phase-2 backend matrix | `jax_rejax+gpu`, `jax_rejax+cpu`, `sb3+cuda`, `sb3+cpu`, `sbx+cuda` |
| Primary metric | `episode_reward`（`higher_is_better`） |

5 分钟步长非常关键，因为只有在这个时间尺度上，热动态和电池可行性才会显式体现出来。

## Reward 与 CMDP cost

在 env 层，基础标量 reward 为

\[
r_t = r_t^{\mathrm{energy}} + w_{\mathrm{cost}}\, r_t^{\mathrm{cost}} + w_{\mathrm{carbon}}\, r_t^{\mathrm{carbon}}
\]

其中：

\[
r_t^{\mathrm{energy}} = -P_{\mathrm{dc},t}\, \Delta t
\]

\[
r_t^{\mathrm{cost}} = -(C^{\mathrm{fuel}}_t + C^{\mathrm{deg}}_t)
\]

\[
r_t^{\mathrm{carbon}} = -\mathrm{carbon}_{t}
\]

env 同时还暴露单独的 cost 向量

\[
\mathbf{c}_t = \left(C_t^{\mathrm{sla}},\, C_t^{\mathrm{temp}},\, C_t^{\mathrm{deficit}}\right)
\]

其中：

- \(C_t^{\mathrm{sla}}\)：SLA deficit 通道
- \(C_t^{\mathrm{temp}}\)：过热通道
- \(C_t^{\mathrm{deficit}}\)：未满足负荷 / 功率缺口通道

它们在实现里分别对应 `cost_sla`、`cost_overtemp` 和 `cost_power_deficit`。

当前 benchmark PPO/SAC 路径随后对 reward 再施加固定 shaping：

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

前三项 (`sla`、`temp`、`deficit`) 就是论文 Appendix E.5 中正式声明的 CMDP cost 通道。后三项 (`spill`、`track`、`balance`) 是 **shaping-only diagnostics**，由 benchmark 的 reward-shaping wrapper 引入，目的是在转换成普通 MDP 训练 PPO/SAC 时给出更密集的引导信号；它们既不属于论文里的 CMDP 定义，也不会作为安全通道被汇报。这里列出来只是为了让训练 reward 完全可重现。

| 符号 | 代码 key | 含义 |
| --- | --- | --- |
| \(C_t^{\mathrm{spill}}\) | `cost_power_spill` | shaping-only。本步供给超出负荷 + 购电后剩余的部分，按当前负荷归一化。由 reward-shaping wrapper 从 `power_spill` 派生。 |
| \(C_t^{\mathrm{track}}\) | `cost_dispatch_tracking` | shaping-only。相对 wrapper 的价格感知电池 / 柴油机调度目标的偏离。 |
| \(C_t^{\mathrm{balance}}\) | `cost_power_balance` | shaping-only。同步骤平衡逻辑后的功率不平衡残差绝对值，按当前负荷归一化。 |

当前定版的 shaping weights 来自 `benchmarks/dc_microgrid/configs/task.yaml`：

- \(\lambda_{\mathrm{sla}} = 50\)
- \(\lambda_{\mathrm{temp}} = 30\)
- \(\lambda_{\mathrm{def}} = 200\)
- \(\lambda_{\mathrm{spill}} = 100\)
- \(\lambda_{\mathrm{track}} = 80\)
- \(\lambda_{\mathrm{bal}} = 0\)

因此，这个 benchmark 最好理解成两层：

- env 语义：标量化基础目标 + 显式 cost vector
- 正式训练路径：使用固定 penalty 权重的 reward-shaped PPO/SAC（含三项 shaping-only diagnostics）

## Baselines

| 名称 | 说明 |
| --- | --- |
| `no_control` | 固定默认设置，没有调度逻辑 |
| `max_renewable` | 优先使用 PV，再用 battery，最后用 diesel |
| `rule_based` | 更强的手工 rule-based 策略，用于太阳能对齐和备用发电 |

## 算法

| Algo | Preset | 说明 |
| --- | --- | --- |
| `ppo` | `dc-microgrid` | reward-shaped、标量化的 PPO baseline（用 Beta 分布 actor 来约束有界动作） |
| `sac` | `dc-microgrid` | reward-shaped、标量化的 SAC baseline（用 tanh-squashed Gaussian） |

Hidden dims `(256, 256)`；gamma `0.99`；`num_envs=64`；`n_steps=288`（每次更新一个完整 episode）；total timesteps `1e6`；5 个 seeds。PPO 用 `lr=1e-4`、`clip_eps=0.1`、`ent_coef=0.001`；SAC 用 `lr=3e-4`。两个 canonical train configs 都开启 observation normalization。超参数与论文 Appendix H.2（`tab:hparams`）一致。

## Eval splits

| Split | 说明 |
| --- | --- |
| `train` | 训练窗口 |
| `iid` | 来自同一 workload pool 的留出天 |
| `cooling_stress` | 更高的室外温度 |
| `renewable_drought` | 更低的太阳能可用性 |

附录中的其他场景包括 `workload_swap`、`workload_shock`、`dg_derating` 和 `sla_tighten`。

## Metrics

把物理汇总量和 reward-facing 汇总量分开看会更清楚：

| 层级 | Key | 说明 |
| --- | --- | --- |
| Step base reward | `reward_vector` | 分解后的 `energy / cost / carbon` reward 分量 |
| Step cost channels | `cost_sla`, `cost_overtemp`, `cost_power_deficit` | env 层服务 / 安全 cost |
| Episode aggregate | `episode_reward` | 累积 shaped reward，也是当前定版的 convergence target |
| Episode aggregate | `total_energy_cost` | 数据中心总能耗代理量 |
| Episode aggregate | `total_fuel_cost` | 总柴油燃料成本 |
| Episode aggregate | `total_carbon_kg` | 总碳排放 |
| Episode aggregate | `sla_violation_rate` | 平均 SLA 违反率 |
| Episode aggregate | `overtemp_rate` | 过热步占比 |
| Episode aggregate | `power_deficit_rate` | 未满足负荷步占比 |
| Episode aggregate | `feasibility_rate` | 无服务 / 安全违反的步占比 |
| Episode aggregate | `pv_utilization` | PV 利用率 |
| Episode aggregate | `battery_cycles` | 电池吞吐量 / 容量代理量 |
| Relative evaluation summary | `norm_score` | 当前 summary pipeline 中基于 `episode_reward` 计算的归一化分数 |
| Relative evaluation summary | `ood_robustness_gap` | `NormScore(iid) - NormScore(cooling_stress)` |

这里最重要的细节是：DC Microgrid 当前的 `norm_score` 是 reward-based 的，因为当前已定版任务目标就是 `episode_reward`。而像 `total_fuel_cost`、`total_carbon_kg` 这样的物理量仍然会单独报告，不能把它们与 leaderboard score 直接等同。

!!! caution "`feasibility_rate` 在 Python backend 上有数值尾巴"
    跨 backend 比较 Phase-2 数据时尤其要注意：`feasibility_rate` 是对 per-step `cost_sum` 的**严格 exact-zero** 检查。

    在 Python backend 上，机器精度级别（约 $10^{-8}$）的 power-balance 数值尾巴会把这个严格指标推到约 `0.5`——**即使** replay audit 显示 `mean_cost_power_balance` ≈ `1e-8`、最大功率平衡误差很小、且没有任何 SLA deficit。换言之，这个 0.5 不代表真有一半步在违反约束。

    论文主张：用 `episode_reward`、`sla_violation_rate`、`mean_cost_power_balance` 以及 Phase-2 physical / paper audits 作为主比较口径，**不要**把 `feasibility_rate` 当成跨 backend 的安全度量。

## Quick start

```bash
python benchmarks/dc_microgrid/run.py baseline \
  --seeds 0,1,2,3,4 \
  --splits train,iid,cooling_stress,renewable_drought,workload_swap,workload_shock,dg_derating,sla_tighten
python benchmarks/dc_microgrid/run.py train --algo ppo --seed 0
python benchmarks/dc_microgrid/run.py eval --run-id <run_id> --split iid
python benchmarks/dc_microgrid/run.py summarize
python benchmarks/dc_microgrid/run.py plots
```

完整默认流程：

```bash
python benchmarks/dc_microgrid/run_all.py
```

## Execution Scaling

DC Microgrid 是 execution-scaling addendum 的主任务。该 artifact 与 algorithm-effect leaderboard 分开：它不比较最终 reward，JAX-only extended range 也不能被写成与 Python backend 的 matched fairness 胜利。

当前 scaling artifact 包含 30 行 matched-range 结果（`jax_rejax/gpu` 对 `sb3/cuda`，`nenv={16,32,64}`，seeds `0..4`）以及 20 行 JAX-only extended 结果（`nenv={32,64,128,256}`，seeds `0..4`）。这些都是 execution artifact，不更新算法效果排行榜。

## 输出文件

```text
benchmarks/dc_microgrid/results/
  manifest.json
  runs/
  artifacts/
  summary/latest.json
  figures/
    normscore_bars.{pdf,png}
    reward_curves.{pdf,png}
    cost_decomposition.{pdf,png}
    ood_robustness.{pdf,png}
  scaling/execution_scaling.json
  scaling/execution_scaling_table.csv
```

## 交叉引用

- [Physics -> Microgrid](../physics/microgrid.md)
- [Architecture -> Data pipeline](../architecture/data-pipeline.md)
- [API -> Microgrid](../api/microgrid.md)
