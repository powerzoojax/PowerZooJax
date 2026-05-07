# Getting started

这一页用大约 5 分钟，从空仓库走到一个 JIT 编译过的 rollout。读完后按页底的阅读路线继续深入。

## 安装

PowerZooJax 需要 Python 3.10+，依赖管理用 `uv`。

```bash
git clone https://github.com/powerzoojax/PowerZooJax.git
cd PowerZooJax
uv sync
```

默认配置依赖 `jax[cuda12]`。如果只用 CPU，先在 `pyproject.toml` 里把 JAX 的依赖换掉再 `uv sync`。

> **Windows 用户**：原生 Windows 上的 JAX 只能跑 CPU，要 GPU 加速请走 WSL2 + Ubuntu。完整的配置流程（含 VS Code 连 WSL、连 `uv` 的 `.venv`、运行项目示例、CUDA Toolkit 装不装的说明、故障排查表）见 [Running on Windows](setup/windows.md)。

## 验证运行环境

```python
import jax
from powerzoojax.case import load_case

print("JAX version:", jax.__version__)
print("Devices:", jax.devices())

case = load_case("5")
print(case.n_nodes, case.n_lines, case.n_units, case.n_loads)
```

## 第一次环境 step

```python
import jax
import jax.numpy as jnp

from powerzoojax.case import load_case
from powerzoojax.envs import TransGridEnv, make_trans_params

case = load_case("5")
env = TransGridEnv()
params = make_trans_params(case, max_steps=48)

key = jax.random.PRNGKey(0)
obs, state = env.reset(key, params)

action = jnp.zeros(case.n_units, dtype=jnp.float32)
key, sk = jax.random.split(key)
obs, state, reward, costs, done, info = env.step(sk, state, action, params)

print("obs shape:", obs.shape)
print("time step:", int(state.time_step))
print("reward:  ", float(reward))
print("names:   ", env.constraint_names(params))
print("costs:   ", costs)
print("done:    ", bool(done))
print("cost_sum:", float(info["cost_sum"]))
```

## 返回值怎么读

- `reset(key, params) -> (obs, state)`。
- `step(key, state, action, params) -> (obs, state, reward, costs, done, info)`。
- `costs` 是 core CMDP 向量；`env.constraint_names(params)` 给出它的静态分量名。
- `done` 描述的是刚刚发生的那一次 transition。
- 当 `done=True` 时，`step` 返回的 `state` 已经是 auto-reset 后的新 episode 初始 state。
- `info["cost_sum"]` 只是聚合诊断量，不是 core 约束通道。
- 在 scan 风格的训练循环里用 `step_auto_reset(...)`（或 [`powerzoojax.utils.jax_utils`](architecture/gpu-pipeline.md) 里的 helper）。它在返回的 obs 与 state 上加了 `stop_gradient`，作为防御性保护防止梯度跨 episode 边界（当前采样无梯度，但代码改动时会自动生效）。

## 用 scan 跑一段 rollout

理解了单步合约后，下一步是把整段 rollout 也 JIT 编译进来：

```python
import jax
import jax.numpy as jnp

from powerzoojax.case import load_case
from powerzoojax.envs import TransGridEnv, make_trans_params
from powerzoojax.utils.jax_utils import scan_rollout

case = load_case("5")
env = TransGridEnv()
params = make_trans_params(case, max_steps=48)

@jax.jit
def episode_return(key, action_seq):
    key, k_reset, k_scan = jax.random.split(key, 3)
    _, state = env.reset(k_reset, params)
    final_state, obs_traj, reward_traj, cost_traj, done_traj, info_traj = scan_rollout(
        env, k_scan, state, params, action_seq
    )
    return reward_traj.sum()

actions = jnp.zeros((48, case.n_units), dtype=jnp.float32)
print(float(episode_return(jax.random.PRNGKey(0), actions)))
```

如果想在一个程序里同时跑 256 个并行 env，参考 [Examples → Batched rollout](examples/02_batched_rollout.md)。

## 选一个起点环境

| 你想研究的内容 | 从哪个 env 入手 |
| --- | --- |
| 输电网 dispatch 与拥塞 | `TransGridEnv` |
| 安全约束机组组合 | `UnitCommitmentEnv`（TSO benchmark） |
| 平衡辐射状配电网控制 | `DistGridEnv`（DSO / DERs benchmark） |
| 不平衡三相馈线控制 | `DistGrid3PhaseEnv` |
| 单独研究储能物理 | `BatteryEnv` |
| 储能在电价下的套利 | `CostBasedMarketEnv`（或 `BidBasedMarketEnv`） |
| 多 agent 竞争性市场 | `MarketMARLEnv`（GenCos benchmark） |
| 多目标微电网控制 | `DataCenterMicrogridEnv`（DC Microgrid benchmark） |
| PPO / CMDP / MARL 集成 | `powerzoojax.rl` 下的 wrapper 与 preset |

需要精确的状态转移方程与约束定义，见 [Physics](physics/transmission.md)。

## 示例脚本

仓库里有可直接运行的脚本，位于 [`examples/`](https://github.com/powerzoojax/PowerZooJax/tree/main/examples)：

- `examples/jax_00_verify_device.py` —— JAX 设备检测、JIT ????、batched rollout 模板。
- `examples/jax_01_create_case.py` —— case 加载、检查、拓扑绘图。
- `examples/jax_02_grid_env.py` —— `TransGridEnv` reset / step、JIT、vmap。
- `examples/jax_03_load_profiles.py` —— `DataLoader` 接到 `TransGridEnv` 的端到端流程。
- `examples/train_ppo_transgrid.py` —— 单 agent PPO + `LogWrapper`。
- `examples/train_safe_ppo_transgrid.py` —— PPO-Lagrangian / CMDP + `SafeRLWrapper`。
- `examples/train_ippo_grid.py` —— 多 agent IPPO + `GridMARLEnv`。
- `examples/train_ppo_battery.py` —— 在 `BatteryEnv` 上自定义 PPO 循环。
- `examples/train_rejax_battery.py` —— 通过高层 Rejax 适配器训练 battery。

本地预览文档：

```bash
./run_doc.sh         # mkdocs serve
./run_doc.sh build   # 一次性 build（链接坏了会失败）
```

## 阅读顺序

读完本页后建议依次：

1. [Concepts → Overview](concepts/overview.md) —— 这套 benchmark 为什么要存在。
2. [Concepts → JAX + RL 环境实现规范](concepts/jax-contract.md) —— 本仓库每个 env 都遵守的 10 条规范。
3. [Architecture → Repo map](architecture/repo-map.md) —— 各模块在哪里。
4. [Physics → Transmission](physics/transmission.md) —— 你的第一个 env 实际在计算什么。
5. [Benchmarks → Overview](benchmarks/overview.md) —— 5 个论文任务。
6. [Training → Wrappers](training/wrappers.md) 与 [Trainers](training/trainers.md) —— 怎么训练 policy。
7. [API reference](api/grid.md) —— 找函数签名时来这。
