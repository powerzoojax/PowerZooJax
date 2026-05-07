# Getting started

This page goes from a fresh clone to a JIT-compiled rollout in about five minutes. After that, follow the reading map at the bottom to go deeper.

## Installation

PowerZooJax supports Python 3.10-3.12 and uses `uv` for dependency management.

Start from a fresh clone:

```bash
git clone https://github.com/powerzoojax/PowerZooJax.git
cd PowerZooJax
```

### Windows

Native Windows runs JAX in CPU-only mode (`uv sync` from PowerShell is enough). For GPU acceleration, JAX requires Linux, so the recommended path is **WSL2 + Ubuntu**. The full setup — enabling WSL2, exposing the NVIDIA GPU, deciding whether you need a system CUDA toolkit, opening the WSL project from VS Code, pointing VS Code at the `uv`-managed `.venv`, running the project examples, and a project-specific troubleshooting table — is on its own page: [Running on Windows](setup/windows.md).

### Linux + CPU

On Linux, the default install also targets CPU-compatible JAX:

```bash
uv sync
```

### Linux + GPU (CUDA 12)

For Linux machines with CUDA 12, install the CUDA extra:

```bash
uv sync --extra cuda12
```

If you prefer `pip`, the equivalent commands are:

```bash
pip install -e .
# Linux + CUDA 12
pip install -e ".[cuda12]"
```

If JAX preallocates too much GPU memory during debugging, set:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

(PowerZooJax already sets this to `false` at import time unless the user has set it explicitly.)

## Verify the runtime

```python
import jax
from powerzoojax.case import load_case

print("JAX version:", jax.__version__)
print("Devices:", jax.devices())

case = load_case("5")
print(case.n_nodes, case.n_lines, case.n_units, case.n_loads)
```

## First environment step

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

## How to read the return values

- `reset(key, params) -> (obs, state)`.
- `step(key, state, action, params) -> (obs, state, reward, costs, done, info)`.
- `costs` is the core CMDP vector; `env.constraint_names(params)` gives its static component names.
- `done` belongs to the transition that just happened.
- `state` returned by `step` is already auto-reset when `done=True`.
- `info["cost_sum"]` is an aggregate diagnostic, not the core constraint channel.
- Use `step_auto_reset(...)` inside scan-based training loops (or the helpers in [`powerzoojax.utils.jax_utils`](architecture/gpu-pipeline.md)). It adds `stop_gradient` on the returned obs and state as defensive protection against accidental gradient flow across episode boundaries (sampling currently has no gradients, but the protection auto-activates if code changes in the future).

## A scan-based rollout

After the single-step contract, the next step is a fully JIT-compiled rollout:

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

For 256 parallel envs in one program, see [Examples → Batched rollout](examples/02_batched_rollout.md).

## Choose a starting environment

| If you want to study | Start with |
| --- | --- |
| Transmission dispatch and congestion | `TransGridEnv` |
| Security-constrained unit commitment | `UnitCommitmentEnv` (TSO benchmark) |
| Balanced radial distribution control | `DistGridEnv` (DSO / DERs benchmarks) |
| Unbalanced three-phase feeder control | `DistGrid3PhaseEnv` |
| Standalone storage physics | `BatteryEnv` |
| Storage arbitrage under prices | `CostBasedMarketEnv` (or `BidBasedMarketEnv`) |
| Competitive multi-agent market | `MarketMARLEnv` (GenCos benchmark) |
| Multi-objective microgrid control | `DataCenterMicrogridEnv` (DC Microgrid benchmark) |
| PPO / CMDP / MARL integration | wrappers and presets in `powerzoojax.rl` |

For exact state-transition equations and constraint semantics, see [Physics](physics/transmission.md).

## Example scripts

The repository includes runnable examples in [`examples/`](https://github.com/powerzoojax/PowerZooJax/tree/main/examples):

- `examples/jax_00_verify_device.py` — JAX device check, JIT quick checks, batched rollout patterns.
- `examples/jax_01_create_case.py` — case loading, inspection, topology plotting.
- `examples/jax_02_grid_env.py` — `TransGridEnv` reset / step, JIT, vmap.
- `examples/jax_03_load_profiles.py` — `DataLoader` to `TransGridEnv` end-to-end.
- `examples/train_ppo_transgrid.py` — single-agent PPO with `LogWrapper`.
- `examples/train_safe_ppo_transgrid.py` — PPO-Lagrangian / CMDP with `SafeRLWrapper`.
- `examples/train_ippo_grid.py` — multi-agent IPPO with `GridMARLEnv`.
- `examples/train_ppo_battery.py` — custom PPO loop on `BatteryEnv`.
- `examples/train_rejax_battery.py` — battery training via the high-level Rejax adapter.

For local docs preview:

```bash
./run_doc.sh         # mkdocs serve
./run_doc.sh build   # one-shot build (fails on broken links)
```

## Reading order

After this page, the recommended order is:

1. [Concepts → Overview](concepts/overview.md) — why this suite exists.
2. [Concepts → JAX + RL environment implementation rules](concepts/jax-contract.md) — the ten conventions every env in the repo follows.
3. [Architecture → Repo map](architecture/repo-map.md) — where everything lives.
4. [Physics → Transmission](physics/transmission.md) — what your first env computes.
5. [Benchmarks → Overview](benchmarks/overview.md) — the 5 paper tasks.
6. [Training → Wrappers](training/wrappers.md) and [Trainers](training/trainers.md) — how to train policies.
7. [API reference](api/grid.md) — when you need a signature.
