# Power 系统入门

这一页是给 ML 读者用的工作术语表。它不是完整的电气工程课程；提供的词汇足够你看懂剩余文档。每个词条一个段落，物理与 benchmark 页面会反复引用。

## 网络与拓扑

- Bus / node（母线 / 节点）。网络中的连接点。发电机、负荷、储能都挂在 bus 上。
- Line / branch（支路）。两个 bus 之间的电力线路。每条 line 有一个 thermal limit（最大输送功率）。
- Transmission grid（输电网）。高压主干网（数百 kV），连接发电厂与变电站。整体网状拓扑，用完整 AC 方程或其 DC 线性化建模。
- Distribution grid（配电网）。中低压馈线，直接服务用户。常常是辐射状（树形），三相不平衡。
- Three-phase power（三相电）。交流电系统通常由三条相位彼此错开 120° 的电压 / 电流波形共同供电，记作 A/B/C 三相。理想平衡三相时，三相幅值接近、相位均匀错开，总功率传输更平稳；配电网里单相负荷和不均匀接入很多，所以经常出现三相不平衡，这也是 `DistGrid3PhaseEnv` 需要显式建模的原因。
- Slack bus / reference bus（平衡 / 参考节点）。电压相角（在 AC 下还包括幅值）固定的那个 bus。slack 上的发电机是平衡机组：它通过增加或减少自己的净注入来补齐剩余不平衡，使潮流方程闭合，并让总发电、总负荷与损耗保持平衡。

## 潮流（PF）

“潮流”或 power flow，说的是：给定电网拓扑、发电、负荷和设备参数之后，电功率会如何在网络里分布，以及每个 bus 上的电压、每条 line 上的传输功率分别是多少。工程上做“跑一次潮流”，就是解这样一组网络平衡方程，得到“电从哪里来、经过哪些线路、到哪里去、各处电压是否正常”。

- DC power flow（直流潮流）。一种线性化，忽略无功与电压幅值变化。支路潮流是节点净注入的线性函数，用 [PTDF 矩阵](https://ps-wiki.github.io/wiki/power-transfer-distribution-factor/) 一次矩阵乘即可求解。
- AC power flow（交流潮流）。完整非线性的功率平衡，含有功 `P`、无功 `Q`、电压幅值 `vm`、相角 `va`。用 [Newton-Raphson](https://matpower.app/manual/matpower/ACPowerFlow.html) 迭代求解。
- Active power `P`（有功）。会转化成净能量传递、机械功或热的那部分电功率。直觉上，`P` 对应“真正送达并被使用的能量”，也更直接关系到系统的频率平衡。
- Reactive power `Q`（无功）。在电场和磁场之间往返交换、用于支撑交流设备与电压的那部分电功率。它在一个交流周期平均下来不产生净做功，但对电压支撑，以及电机、变压器等感性设备的正常运行仍然必不可少。
- Voltage magnitude `vm`（电压幅值）。bus 电压的大小 `|V|`，通常用标幺值（p.u.）表示，其中 `1.0` 代表额定电压。“per-bus voltage magnitude” 指的就是所有 bus 电压幅值组成的向量。
- Backward / forward sweep (BFS)（前推回代法）。专为辐射状配电网设计的求解器。它从叶节点向根累加功率，再从根向叶更新电压。在辐射拓扑上比 [Newton-Raphson](https://matpower.app/manual/matpower/ACPowerFlow.html) 便宜。
- [PTDF](https://ps-wiki.github.io/wiki/power-transfer-distribution-factor/) (power transfer distribution factor)（功率转移分布因子）。预计算矩阵 `PTDF[l, n]`，表示在 bus `n` 注入单位功率引起 line `l` 上的潮流变化。在 DC PF 下，支路潮流 `f = PTDF @ p_inj`，其中 `p_inj` 是每个 bus 的净有功注入向量（发电减去负荷，再加上各类资源注入）。

## 发电、调度与 OPF {#发电调度与-opf}

- Generator / unit（发电机组）。一台可控功率源，有功率上下限、边际成本曲线，以及（火电机组的）爬坡限制和开停机切换。
- Dispatch（调度）。某一时刻每台发电机的有功功率。
- Economic dispatch (ED)（经济调度）。在功率平衡约束下，选发电机功率最小化总成本。
- Optimal power flow (OPF)（最优潮流）。ED 加上网络约束（线路容量、电压）。
- DCOPF / ACOPF。分别在 DC PF / 完整 AC PF 模型下的 OPF。
- Merit order（成本优先顺序）。按边际成本从低到高排列发电机组的顺序。**Merit order（优先顺序）规则**会先启用或调度便宜机组，只有在还需要更多容量时才使用更贵的机组；这是常见工程化短流程，不保证是全局最优 UC/SCED。
- Unit commitment (UC) 与 SCUC。UC 在多步时段上选哪些机组开机；SCUC 加上安全约束（线路容量、备用）。两者都引入离散开停机决策与跨时段约束（最短开/停机时间、启动费、爬坡限制）。
- Reserve（备用）。为应对偶发事故而保留的发电余量。SCUC 必须留够覆盖一定比例需求的余量。
- Locational marginal price (LMP)（节点边际电价）。在每个 bus 多供 1 MWh 的边际成本，已计入网络约束。在市场语境下，LMP 用于结算。

## 资源

- Distributed energy resource (DER)（分布式能源）。挂在配电侧的小型发电机、储能或可控负荷。例如屋顶 PV、户用储能、EV 充电桩、柔性 HVAC 负荷。
- Battery / storage（储能 / 电池）。设备状态由 SOC（state of charge）描述，范围在 `soc_min` 与 `soc_max` 之间。放电向电网注入功率，充电吸收功率。单向充放电效率意味着每个循环都有损耗。
- Renewable (PV / wind)（可再生）。曲线驱动的发电机。功率 = `capacity * capacity_factor(t) * (1 - curtailment)`。Curtailment（功率削减）是 agent 主动把发电功率压到可用值之下的选择。
- Vehicle / EV（电动车）。带行程表的电池，仅当车在家时才能充放电；行程会消耗 SOC，并要求出门前满足最低 SOC。
- Flexible load（柔性负荷）。可被削减（现在少用电、付出 discomfort cost）或被搬移（把需求推迟到后续若干时段执行）的可控负荷。符号约定：`current_p_mw > 0` 表示削减负荷量。
- Data center（数据中心）。位于表后侧的负载，消耗功率用于 IT（计算）与冷却。Agent 可以重排训练 / 微调任务，并调整冷却温度设定值。

## 质量与安全约束

- Thermal limit（热极限）。线路允许的最大视在功率或有功。DC PF 下 `|P_l| <= P_l^max`；AC PF 下 `sqrt(P^2 + Q^2) <= S^max`。
- Voltage limits（电压上下限）。`vm_min <= vm <= vm_max`，配电网中常用 `[0.94, 1.06]` 的 p.u.（标幺值，1.0 表示额定值）。
- Voltage unbalance factor (VUF)（电压不平衡度）。三相电压偏离平衡的程度。Fortescue 定义：`|V_negative_sequence| / |V_positive_sequence|`，以百分比表示。
- Power balance（功率平衡）。总发电 = 总负荷 + 损耗。slack bus 吸收任何残差。
- Per unit (p.u.)（标幺值）。无量纲归一化。功率除以基准功率（`base_mva`），电压除以基准电压。方程因此无量纲且便于比较。

## 市场

- Cost-based clearing（基于成本的出清）。市场运营者用真实发电成本曲线出清，没有策略性报价。
- Bid-based clearing（基于报价的出清）。发电机提交报价曲线（价格-数量分段），市场运营者按报价（不是真实成本）出清。算出的 LMP 反映报价。
- SCED (security-constrained economic dispatch)（安全约束经济调度）。单时段、考虑线路容量的出清问题。
- Storage arbitrage（储能套利）。LMP 低时买电存起来、LMP 高时卖出。收益 `sum(LMP * P * dt)`。
- Strategic bidding（策略性报价）。发电机报价高于真实边际成本以抬高出清价。多 agent 设置下可能出现非合作均衡。

## 时间与单位

- Step length（步长）。一次 `step` 调用对应的仿真时间 `delta_t_hours`。常用 `0.5` h（30 min，输电 / 配电 / 市场）和 `1/12` h（5 min，data-center 微电网）。
- Episode length（一集长度）。`max_steps`。多数 benchmark 用 48 步 × 30 min = 24 h；data-center 微电网用 288 步 × 5 min = 24 h。
- 有功 `P` 单位 MW；无功 `Q` 单位 MVAr；能量单位 MWh。论文 benchmark 的成本以英镑（GBP，£）报告，数据来源都是英国相关的（NESO 负荷、Elexon BMRS 发电与电价、Ausgrid 配电）。代码侧 `case_data.py` 里的成本系数仍用通用的 `$` 量纲符号，但跑论文实验时按 GB 电价解读。

## 常见缩写

| 缩写 | 含义 |
| --- | --- |
| AC / DC PF | 交流 / 直流 power flow |
| ACOPF / DCOPF | AC / DC PF 模型下的 OPF |
| BFS | Backward / forward sweep（辐射 PF 求解器） |
| CMDP | Constrained Markov decision process（约束 MDP） |
| DER | Distributed energy resource（分布式能源） |
| ED / SCED | Economic dispatch / security-constrained ED |
| IPPO / MAPPO | Independent / centralized PPO（多 agent RL） |
| LMP | Locational marginal price（节点边际电价） |
| OPF | Optimal power flow（最优潮流） |
| PPO | Proximal policy optimization |
| PTDF | Power transfer distribution factor |
| p.u. | Per unit（标幺值） |
| RL / MARL | Reinforcement learning / multi-agent RL |
| SCUC | Security-constrained unit commitment（安全约束机组组合） |
| SLA | Service-level agreement（DataCenter 任务里的截止时间义务） |
| SOC | State of charge（电池剩余能量比例） |
| TSO / DSO | Transmission / distribution system operator（输 / 配电系统运营者） |
| VUF | Voltage unbalance factor（电压不平衡度） |

下一层（[Architecture](../architecture/repo-map.md)）展示这些概念如何映射到代码模块。
