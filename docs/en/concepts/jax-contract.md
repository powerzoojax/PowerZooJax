# JAX + RL environment implementation rules

Writing reinforcement-learning environments in JAX — especially environments designed for GPU-native rollout and, where possible, fully fused training — keeps surfacing the same handful of design choices: where state lives, how randomness flows, how `step` handles episode boundaries, where compilation ends and runtime begins. This page collects the ten rules PowerZooJax has settled on as it implements its environments.

They serve two audiences. For PowerZooJax users they document what every environment in the repo guarantees, so wrappers, trainers, and rollouts can rely on a single shape. For anyone else building JAX-based RL environments they double as a reference checklist of patterns that keep code `jit` / `vmap` / `lax.scan` friendly.

Each rule below states the convention and the typical failure mode when it is broken (which JAX API trips, why batched parallelism degrades, etc.), so you can take what you need.

## Brief glossary (used throughout the docs)

- `jit` is [JAX](https://github.com/jax-ml/jax)'s just-in-time (JIT) compiler. It traces a Python function once and compiles it to a single [XLA (Accelerated Linear Algebra)](https://openxla.org/xla) program — XLA is the linear-algebra compiler backend JAX shares with TensorFlow and PyTorch/XLA. After the first call, repeated calls skip Python and run on the device.
- `trace` is JAX's pre-compilation pass where it runs through a function once to record the array program structure. What matters at this stage is shape, dtype, control flow, and static configuration, not the specific runtime values from one call. When the docs say "JIT trace", "traced configuration", or "trace-time static", they refer to this stage.
- `static` means "fixed from the compiler's point of view". Loop bounds, mode flags, and bundle counts are treated as part of the program structure rather than ordinary data. When the docs say `shape static`, `static field`, or `trace-time static`, the key idea is that changing such values can trigger a new trace / recompilation.
- `vmap` automatically vectorizes a function over a leading batch axis. Instead of writing a loop over environments, you map a single-env function over a batch of states and keys.
- `batch axis` is the dimension that stacks many samples together. For example, in `obs.shape = (N, d)`, the leading `N` is usually the batch axis; `vmap` lifts a single-sample function across that axis.
- `lax.scan` is a compiled equivalent of a Python `for` loop with carry. It replaces the rollout loop in a reinforcement-learning (RL) trainer.
- `carry` is the loop-state package that `lax.scan` takes in, updates, and passes to the next step. In environment rollouts it is usually the current `state`, or `state` plus a few accumulators.
- `PRNGKey` is JAX's explicit pseudo-random-number-generator (PRNG) state. Functions that use randomness must take a key as input and split it before reusing.
- `shape` is the per-dimension size structure of an array, such as `(64, 128, 32)`. JAX compilation caches are often keyed by shape, which is why "keep shapes static" shows up so often.
- `dtype` is the element type of an array, such as `float32`, `int32`, or `bool`. Like shape, dtype is part of what JAX distinguishes at compile time.
- pytree is JAX's term for a nested container of arrays (dicts, dataclasses, tuples). All `state` and `params` objects in PowerZooJax are pytrees. See the aside below for what counts as a pytree and what `pytree_node` does.
- [`flax`](https://github.com/google/flax) is Google's NN library on top of JAX. PowerZooJax only uses one piece of it — `flax.struct.dataclass` — to declare immutable, pytree-registered dataclasses for `EnvState` and `EnvParams`, with `pytree_node=False` marking static fields.
- [`chex`](https://github.com/google-deepmind/chex) is DeepMind's JAX utility library. PowerZooJax only uses its `chex.Array` type alias to annotate array fields on those dataclasses.
- `hot path` is the part of the code that runs at high frequency during sampling or training and therefore dominates throughput and latency. In PowerZooJax that usually means rollout, policy forward, and gradient update logic on device.
- `stop_gradient` is JAX's operation for treating a value as a constant for differentiation. It does not change the numeric value itself; it only blocks gradients from flowing through it.
- `compiled program` is the device-side program JAX produces after tracing a Python function. When the docs say "single compiled program", they mean the sampling or training logic has been fused into one continuous XLA program instead of bouncing back to Python each step.
- `struct-of-arrays` is a batch-friendly data layout where the same field from many objects is stored in one array, instead of each object storing its own full record. For JAX / `vmap`, this is usually easier to batch and easier to keep shape-static.

!!! note "Aside — pytree and `pytree_node` (skip if you know them)"

    A **pytree** is any nested container whose *leaves* are arrays (or scalars) and whose *nodes* are *registered* container types: `dict`, `list`, `tuple`, `namedtuple`, and dataclasses registered through `flax.struct.dataclass`. JAX walks pytrees leaf-by-leaf — that is how `jit`, `vmap`, `grad`, `lax.scan` carry, and `tree_map` all work. A plain Python `class` is **not** a pytree: JAX has no recipe for taking it apart, which is why every `EnvState` / `EnvParams` in PowerZooJax is declared with `flax.struct.dataclass`.

    A typical PowerZooJax dataclass therefore looks like this to JAX (solid = pytree leaf, dashed = static field):

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

    The per-field flag `pytree_node` decides which symbol a field gets:

    - **◆ Default `pytree_node=True`** — the field is a *leaf*. It must be a JAX array (or another pytree). `jit` traces it, `vmap` pushes a batch axis through it, `grad` differentiates against it. Changing its *value* between calls does **not** trigger recompilation.
    - **□ `pytree_node=False`** — declared as `max_steps: int = struct.field(pytree_node=False)`. The field is removed from the pytree and treated as *static metadata*: JAX bakes its value into the compiled XLA program, so changing it **does** trigger recompilation. Use it for Python `int` / `bool` / mode flags / loop bounds — things that are part of the program's *shape*, not its *data*.

## Rule 1 — state is explicit

Every environment object holds only static configuration (Python ints, function references, mode flags). Runtime simulation state lives in a separate `EnvState` pytree returned by `reset` and threaded through `step`:

```python
env = TransGridEnv()                       # stateless: only methods
obs, state = env.reset(key, params)        # state is the dynamic data
obs, state, reward, costs, done, info = env.step(key, state, action, params)
```

If state were stored on `self`, two parallel rollouts would silently share it under `vmap`. Pytree state makes batching trivial.

## Rule 2 — state and params are pytrees

State and parameter classes use `flax.struct.dataclass`. Fields are JAX arrays, except for static settings like `max_steps`, which are marked `pytree_node=False`:

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

Consequences:

- never mutate state in place; use `state.replace(...)` to create a new state.
- array shapes and dtypes must be static across `step` calls, so JAX can reuse one compiled program.
- avoid Python strings or dynamically sized lists inside traced state.
- when task semantics need a queue or history (for example deferred flexible demand or LMP history), JAX state represents it with fixed-capacity arrays plus indices / masks, not Python `deque` objects or dynamic `list`s. That is an implementation constraint; benchmark pages should describe the physical quantity, not the container.

## Rule 3 — randomness is explicit

There is no global RNG. Every stochastic operation takes a `key`, and any branch that uses randomness must split the key first:

```python
key, k_step, k_reset = jax.random.split(key, 3)
obs, state, reward, costs, done, info = env.step(k_step, state, action, params)
new_obs, new_state = env.reset(k_reset, params)
```

When auto-reset is built into `step`, the implementation splits the incoming key internally: once for the transition and once for the next-episode reset. Reusing the same key would correlate the next episode's initial state with the last step's noise.

## Rule 4 — `step` already auto-resets

Every PowerZooJax `step` ends with the equivalent of:

```python
final_state = jax.tree_util.tree_map(
    lambda new, rst: jnp.where(done, rst, new),
    new_state,
    reset_state,
)
```

`done` is the terminal flag for the transition that just finished. Whenever `done=True`, the returned `state` is the freshly reset initial state of the next episode. This is what lets fixed-length `lax.scan` rollouts work without Python conditionals.

The base class also exposes `step_auto_reset(key, state, action, params)`, which is `step` plus `jax.lax.stop_gradient` on the returned observation and state. Use it inside `lax.scan` as defensive protection: while sampling has no gradient tracking currently, if code is later modified to enable gradient tracking during sampling, `stop_gradient` will automatically prevent gradients from flowing across episode boundaries (which would violate model-free RL semantics).

## Rule 5 — prefer `jnp.where` to `lax.cond` under `vmap`

Under `vmap`, `lax.cond` evaluates both branches because each batch element can choose differently. For simple numerical selection, `jnp.where` is cheaper and clearer:

```python
soc_next = jnp.where(is_charging, soc + dsoc_charge, soc - dsoc_discharge)
```

`lax.cond` is reserved for branches where one side is materially more expensive than the other.

## Rule 6 — loops become `lax.scan` or `lax.while_loop`

PowerZooJax uses `lax.scan` for rollouts and fixed-capacity buffer logic, and `lax.while_loop` for iterative solvers ([Newton-Raphson AC power flow](https://matpower.app/manual/matpower/ACPowerFlow.html), [backward / forward sweep distribution power flow](https://jesit.springeropen.com/articles/10.1186/s43067-021-00031-0), [primal-dual interior-point](https://optimization.cbe.cornell.edu/index.php?title=Interior-point_method_for_LP) in the exact [security-constrained economic dispatch (SCED)](https://ps-wiki.github.io/wiki/security-constrained-economic-dispatch/) solver). Runtime loops in the hot path must move into JAX control flow. The only Python loops allowed inside jitted code are trace-time static loops over fixed configuration, such as iterating over a resource-bundle tuple whose length is fixed at setup.

## Rule 7 — batch is a first-class design target

These two patterns must always work:

```python
obs, states = jax.vmap(env.reset, in_axes=(0, None))(keys, params)
obs, states, rewards, costs, dones, infos = jax.vmap(
    env.step_auto_reset, in_axes=(0, 0, 0, None)
)(step_keys, states, actions, params)
```

That is why state pytrees are static, resource bundles are struct-of-arrays, and dynamic lists are replaced by fixed-size masked arrays.

## Rule 8 — resource bundles are trace-time static

Grid environments iterate over `params.resources` with a Python loop, but the loop length is part of the traced configuration: the bundle tuple is fixed at setup. The result is readable code with no per-step Python overhead inside the compiled program.

## Rule 9 — setup work stays outside JIT

Some work runs once on CPU at setup time; only the resulting numeric arrays enter JAX:

- building `CaseData` from raw network tables.
- preparing solver setups: [PTDF](https://ps-wiki.github.io/wiki/power-transfer-distribution-factor/) (power transfer distribution factors), BFS (backward / forward sweep distribution power flow), AC PF (AC power flow), DCOPF (DC optimal power flow), piecewise ED (economic dispatch), and exact SCED.
- loading parquet time series through `DataLoader`.

The compiled hot path then sees only numeric pytrees. For example, [PTDF](https://ps-wiki.github.io/wiki/power-transfer-distribution-factor/) (power transfer distribution factor) is a precomputed matrix that maps net nodal injection to line flow under the DC power flow approximation; building it requires a sparse solve that should not run inside the rollout.

## Rule 10 — separate reward from safety constraints

This is an interface contract: the `reward` returned by `step` represents the task objective, while `costs` represent safety or operational constraint violations.

!!! warning "Do not fold safety penalties into reward"
    Mixing weighted constraint violations back into `reward` is one of the most common anti-patterns in RL practice: after training, you can no longer separate "objective performance" from "safety" out of a single scalar, and benchmarks become incomparable.

    PowerZooJax forces `reward` and `costs` onto two independent channels:

    - `reward` answers "how well was the objective solved?"
    - `costs` answer "were constraints violated?"

    This is what lets the same env support standard RL, Safe RL / CMDP training, and benchmark reporting without changing the underlying physics. Folding cost back into reward is a wrapper / trainer concern, not an env concern.

For the fuller formalization, semantic explanation, and reporting conventions, see the next page: [MDP / CMDP](reward-cost-split.md).

## Putting it together

A minimal correct rollout that respects every rule above looks like this:

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

256 parallel environments, 48 steps each, no Python loops in the hot path.
