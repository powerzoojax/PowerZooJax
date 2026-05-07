# JAX + RL 环境实现规范

在 JAX 上写强化学习环境——尤其是要支撑端到端 GPU 训练的环境——会反复遇到同一组设计选择：state 放在哪里、随机性怎么传递、`step` 怎么处理 episode 边界、编译期和运行期的边界划在哪。这一页把 PowerZooJax 在实现自家环境时沉淀下来的 10 条规范汇总在一起。

它面向两类读者。**PowerZooJax 用户**可以把它当成"本仓库每个 env 都对外保证的东西"——wrapper、trainer、rollout 都依赖这套统一形状。**其他在 JAX 上做 RL 环境的开发者**可以把它当成参考清单：这些是让代码对 `jit` / `vmap` / `lax.scan` 友好的常见模式。

下面每一条都给出规范本身，以及不满足时常见的失败方式（哪个 JAX API 会报错、为什么 batch 并行会退化等），方便按需取用。

## 简短术语表（全文档通用）

- `jit`：[JAX](https://github.com/jax-ml/jax) 的 just-in-time（JIT，即时）编译器，把一个 Python 函数 trace 一次后编译成单个 [XLA（Accelerated Linear Algebra）](https://openxla.org/xla) 程序——XLA 是 JAX、TensorFlow、PyTorch/XLA 共用的线性代数编译器后端。第一次调用之后，再调用就跳过 Python，直接在 device 上跑。
- `trace`：JAX 在真正编译前，先“走一遍”函数并记录数组运算结构的过程。它关心的是计算图的形状、dtype、控制流和静态配置，而不是某一次调用里的具体数值本身。文档里说“进入 JIT trace”“被 trace 的配置”“trace-time static”，指的都是这个阶段。
- `static`：指“在编译视角下固定不变”的部分。比如循环长度、模式开关、bundle 个数这类值，JAX 会把它们当成程序结构的一部分，而不是普通数据。文档里说 `shape static`、`static field`、`trace-time static` 时，强调的都是“这些东西一变，就可能触发重新 trace / 重新编译”。
- `vmap`：自动把函数沿一个前导 batch 轴向量化。不必手写遍历环境的循环，把单环境函数 map 到一批 state 与 key 上即可。
- `batch axis`：批量维，也就是“一批样本排在一起的那一维”。例如 `obs.shape = (N, d)` 里最前面的 `N` 通常就是 batch axis；`vmap` 就是在这条轴上把单样本函数自动扩成批量函数。
- `lax.scan`：带 carry 的 Python `for` 循环的编译版本，用来替换强化学习（RL，reinforcement learning）trainer 里的 rollout 循环。
- `carry`：`lax.scan` 每一步都会接收、更新、再传给下一步的那份“循环状态包”。在环境 rollout 里，它通常就是当前 `state`，或者 `state` 加上一些累计量。
- `PRNGKey`：JAX 显式管理的伪随机数生成器（PRNG，pseudo-random-number generator）状态。用到随机性的函数必须接收一个 key 并在复用前 split。
- `shape`：数组每一维的长度结构，例如 `(64, 128, 32)`。JAX 很多编译缓存都是按 shape 区分的，因此“shape 保持静态”是高频要求。
- `dtype`：数组元素的数据类型，例如 `float32`、`int32`、`bool`。它和 shape 一样，属于 JAX 编译时要区分的重要信息。
- pytree：JAX 对嵌套数组容器（dict、dataclass、tuple）的统称。PowerZooJax 里所有 `state` 和 `params` 都是 pytree。下方的小附注解释了什么算 pytree、`pytree_node` 又是做什么的。
- [`flax`](https://github.com/google/flax)：Google 在 JAX 之上的神经网络库。PowerZooJax 只用了它的 `flax.struct.dataclass`——一种注册为 pytree 的不可变 dataclass，用来声明 `EnvState` 与 `EnvParams`，并用 `pytree_node=False` 标记静态字段。
- [`chex`](https://github.com/google-deepmind/chex)：DeepMind 的 JAX 工具库。PowerZooJax 只用了它的 `chex.Array` 类型别名，给上述 dataclass 的数组字段做类型注解。
- `hot path`：训练或采样时会被高频重复执行、最影响吞吐和延迟的那段代码路径。PowerZooJax 里通常指 rollout、policy forward、gradient update 这些 device 上反复运行的部分。
- `stop_gradient`：JAX 提供的“把某个值当常量看待”的操作。它不会改数值本身，但会阻止梯度继续穿过这个值向前传播。
- `compiled program`：JAX 把 Python 函数 trace 完后生成的整段设备端程序。文档里说“single compiled program”时，意思是采样或训练逻辑被并进一份连续执行的 XLA 程序，而不是每步都回到 Python。
- `struct-of-arrays`：一种批量数据布局方式：同一种字段放在同一个数组里，而不是每个对象各自存一份完整结构。对 JAX / `vmap` 来说，这种布局通常更容易批量化，也更适合静态 shape。

!!! note "附注：pytree 与 `pytree_node`（已熟悉可跳过）"

    **pytree** 指任何"嵌套容器"：叶子是数组（或标量），节点是**已注册的容器类型**——`dict`、`list`、`tuple`、`namedtuple`，以及通过 `flax.struct.dataclass` 注册的 dataclass。JAX 顺着 pytree 一片叶子一片叶子地处理——`jit`、`vmap`、`grad`、`lax.scan` 的 carry、`tree_map` 全都按这种方式工作。普通 Python `class` **不是 pytree**：JAX 没有办法拆开它的字段。所以 PowerZooJax 里每个 `EnvState` / `EnvParams` 都用 `flax.struct.dataclass` 声明。

    一个典型的 PowerZooJax dataclass 在 JAX 眼里长这样（实线 = pytree 叶子，虚线 = 静态字段）：

    ```mermaid
    %%{init: {'flowchart': {'nodeSpacing': 14, 'rankSpacing': 28}}}%%
    flowchart TB
        BS["BatteryState<br/><i>flax.struct.dataclass</i>"]:::ds
        BS --> b1["current_p_mw<br/>chex.Array"]:::leaf
        BS --> b2["soc<br/>chex.Array"]:::leaf
        BS --> b3["time_step<br/>chex.Array"]:::leaf
        BS --> b4["done<br/>chex.Array"]:::leaf

        b1 ~~~ EP

        EP["EnvParams<br/><i>flax.struct.dataclass</i>"]:::ds
        EP --> p1["p_max<br/>chex.Array"]:::leaf
        EP --> p2["soc_init<br/>chex.Array"]:::leaf
        EP --> p3["max_steps : int<br/>pytree_node=False"]:::static

        classDef ds fill:#e8f7f4,stroke:#0f766e,stroke-width:1.5px,color:#123c3a;
        classDef leaf fill:#e8f7f4,stroke:#0f766e,stroke-width:1.5px,color:#123c3a;
        classDef static fill:#e8f7f4,stroke:#0f766e,stroke-width:1.5px,stroke-dasharray:4 2,color:#123c3a;
    ```

    字段开关 `pytree_node` 决定字段拿到哪个符号：

    - **◆ 默认 `pytree_node=True`** —— 字段是 pytree 的**叶子**，必须是 JAX 数组（或另一棵 pytree）。`jit` 会 trace 它，`vmap` 会沿 batch 轴推开它，`grad` 会对它求导。**值在调用之间变化不会触发重编译**。
    - **□ `pytree_node=False`** —— 写成 `max_steps: int = struct.field(pytree_node=False)`。该字段被从 pytree 里剔除，作为**静态元数据**：JAX 把它的值固化进编译后的 XLA 程序，**改这个值就会重编译**。Python `int` / `bool` / 模式开关 / 循环长度用它——这些是程序"形状"的一部分，不是"数据"。

## 规则 1 —— 状态外置，不存进 `self`

每个环境对象只持有静态配置（Python int、函数引用、模式开关）。运行时的仿真状态单独放在 `EnvState` 这个 pytree 里，由 `reset` 返回，并在 `step` 调用之间作为输入和输出传递：

```python
env = TransGridEnv()                       # static namespace
obs, state = env.reset(key, params)        # state is the dynamic data
obs, state, reward, costs, done, info = env.step(key, state, action, params)
```

如果把 state 存在 `self` 上，两条 `vmap` 下的并行 rollout 会隐式共享这份状态。把 state 做成 pytree 之后，batch 化是自然的。

## 规则 2 —— state 与 params 都是 pytree

state 与 params 类用 `flax.struct.dataclass`。字段都是 JAX 数组，例外只有静态设置（如 `max_steps`），用 `pytree_node=False` 标记：

```python
from flax import struct
import chex

@struct.dataclass
class BatteryState:
    current_p_mw: chex.Array
    soc: chex.Array
    time_step: chex.Array
    done: chex.Array
```

带来的约束：

- 不要原地修改 state，用 `state.replace(...)` 产生新 state。
- `step` 调用之间数组的 shape 与 dtype 必须保持静态，这样 JAX 可以复用同一份编译产物。
- 不要在被 trace 的 state 里放 Python 字符串或动态长度列表。
- 如果任务语义需要队列或历史（例如被延后服务的 flexible demand、LMP 历史），JAX state 用固定容量数组加索引 / mask 表达，而不是 Python `deque` 或动态 `list`。这是实现约束；benchmark 页应描述物理量本身，不把容器名当成任务语义。

## 规则 3 —— 随机性显式

没有全局 RNG。任何带随机性的操作都接收一个 `key`；任何用到随机性的分支必须先 split：

```python
key, k_step, k_reset = jax.random.split(key, 3)
obs, state, reward, costs, done, info = env.step(k_step, state, action, params)
new_obs, new_state = env.reset(k_reset, params)
```

当 auto-reset 内嵌在 `step` 里时，实现内部会自己 split 入参 key：一次给状态转移、一次给重置。复用同一个 key 会让下一 episode 的初始状态与上一步的随机性产生关联。

## 规则 4 —— `step` 已经内置 auto-reset

PowerZooJax 里每个 `step` 末尾都等价于：

```python
final_state = jax.tree_util.tree_map(
    lambda new, rst: jnp.where(done, rst, new),
    new_state,
    reset_state,
)
```

`done` 是刚刚结束那一次转移的终止标志。每当 `done=True` 时，返回的 `state` 就是下一 episode 重置后的初始 state。这正是定长 `lax.scan` rollout 不需要 Python 条件判断的原因。

基类还提供 `step_auto_reset(key, state, action, params)`，等价于 `step` 加上对返回 obs 与 state 的 `jax.lax.stop_gradient`。在 `lax.scan` 里用它作为防御性保护：虽然采样阶段本身没有梯度追踪，但若未来代码改动添加了梯度追踪，`stop_gradient` 会自动防止梯度跨 episode 边界传播（model-free RL 下这样的传播是错误的）。

## 规则 5 —— `vmap` 下优先 `jnp.where`，少用 `lax.cond`

`vmap` 之下，`lax.cond` 会把两个分支都执行一遍（因为每个 batch 元素可能选不同分支）。简单数值选择用 `jnp.where` 既便宜又清晰：

```python
soc_next = jnp.where(is_charging, soc + dsoc_charge, soc - dsoc_discharge)
```

`lax.cond` 仅保留给两个分支计算代价差距很大的情形。

## 规则 6 —— 循环用 `lax.scan` 或 `lax.while_loop`

PowerZooJax 在 rollout 和固定容量 buffer 逻辑中使用 `lax.scan`，在迭代求解器中使用 `lax.while_loop`（[Newton-Raphson AC power flow](https://matpower.app/manual/matpower/ACPowerFlow.html)、[backward / forward sweep distribution power flow](https://jesit.springeropen.com/articles/10.1186/s43067-021-00031-0)、精确 [security-constrained economic dispatch (SCED)](https://ps-wiki.github.io/wiki/security-constrained-economic-dispatch/) 求解器中的 [primal-dual interior-point](https://optimization.cbe.cornell.edu/index.php?title=Interior-point_method_for_LP)）。热路径里的运行期循环必须进入 JAX control flow。只有一种 Python 循环可以保留在 jitted 代码里：它必须是 trace-time static 的，也就是只遍历固定配置，例如长度在 setup 阶段就已经固定的 `resource bundle` 元组。

## 规则 7 —— batch 是核心设计目标

下面两种用法必须始终可用：

```python
obs, states = jax.vmap(env.reset, in_axes=(0, None))(keys, params)
obs, states, rewards, costs, dones, infos = jax.vmap(
    env.step_auto_reset, in_axes=(0, 0, 0, None)
)(step_keys, states, actions, params)
```

正因为这一点，state pytree 才必须静态、resource bundle 才采用 struct-of-arrays、动态列表才用定长 mask 数组替代。

## 规则 8 —— resource bundle 在 trace 期是静态的

电网环境内部会用 Python 循环遍历 `params.resources`，但循环长度属于被 trace 的配置：bundle 元组在 setup 阶段就固定了。结果是：代码可读，且编译产物里没有按步执行的 Python 开销。

## 规则 9 —— setup 工作放在 JIT 之外

有些工作只在 setup 阶段在 CPU 上做一次，最终只把数值数组传入 JAX：

- 从原始网络表构建 `CaseData`。
- 准备各类求解器的 setup：[PTDF](https://ps-wiki.github.io/wiki/power-transfer-distribution-factor/)（power transfer distribution factor，功率转移分布因子）、BFS（backward / forward sweep，前推回代配电潮流）、AC PF（AC power flow，交流潮流）、DCOPF（DC optimal power flow，直流最优潮流）、分段 ED（economic dispatch，经济调度）、精确 SCED（security-constrained economic dispatch，安全约束经济调度）。
- 用 `DataLoader` 加载 parquet 时间序列。

编译后的 hot path 只看到数值 pytree。例如，[PTDF](https://ps-wiki.github.io/wiki/power-transfer-distribution-factor/)（power transfer distribution factor，功率转移分布因子）是一个预计算的矩阵，把节点净注入映射到 DC PF 近似下的支路潮流；构造它需要一次稀疏求解，不应该放进 rollout 里。

## 规则 10 —— reward 与安全约束分离

这是一个接口合约：`step` 返回的 `reward` 表示任务目标，`costs` 表示安全或运行约束违反。

!!! warning "不要把安全惩罚揉进 reward"
    把约束违反加权后塞回 `reward` 是 RL 实践里最常见的反模式：训练完之后再无法从单一标量里拆出"目标好坏 vs 安全程度"，benchmark 也无法横比。

    PowerZooJax 强制 `reward` 与 `costs` 走两条独立通道：

    - `reward` 回答"目标做得怎么样"
    - `costs` 回答"是否违反了约束"

    这样环境同时兼容普通 RL、Safe RL / CMDP 和 benchmark 报告，而不需要改写底层物理逻辑。要把 cost 转回 reward 是 wrapper / trainer 的事，不是 env 的事。

更完整的形式化定义、语义解释和报告约定，见下一页 [MDP / CMDP](reward-cost-split.md)。

## 把规则组合起来

一段满足上述全部规则的最小 rollout 是这样的：

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
def rollout(key, actions):
    key, k_reset, k_scan = jax.random.split(key, 3)
    _, init_state = env.reset(k_reset, params)
    final_state, obs_traj, reward_traj, cost_traj, done_traj, info_traj = scan_rollout(
        env, k_scan, init_state, params, actions
    )
    return reward_traj.sum()

actions = jnp.zeros((48, case.n_units), dtype=jnp.float32)
returns = jax.vmap(rollout, in_axes=(0, None))(
    jax.random.split(jax.random.PRNGKey(0), 256),
    actions,
)
print(returns.mean())
```

256 个并行环境，每个跑 48 步，hot path 没有任何 Python 循环。
