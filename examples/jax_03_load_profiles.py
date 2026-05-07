"""Build a transmission-grid env from real load profiles.

Pipeline: DataLoader → jnp.ndarray → make_trans_params → TransGridEnv.
"""

import jax
import jax.numpy as jnp

from powerzoojax.case import create_case5
from powerzoojax.data import DataLoader, signals as S
from powerzoojax.envs.grid.trans import TransGridEnv, make_trans_params

# --- 1. Load load profiles -------------------------------------------------
loader = DataLoader()

# GB one-week load (30 min resolution, 336 steps)
load_profile = loader.load_jax_profiles(
    [S.LOAD_ACTUAL_MW],
    source="gb",
    start_date="2024-06-01",
    end_date="2024-06-07",
)
print(f"Load profile: {load_profile.shape}, dtype={load_profile.dtype}")
# → (336, 1), float32

# Optional: multiple signals in one call
multi = loader.load_jax_profiles(
    [S.LOAD_ACTUAL_MW, S.SOLAR_AVAILABLE_MW, S.WIND_AVAILABLE_MW],
    source="gb",
    start_date="2024-06-01",
    end_date="2024-06-07",
)
print(f"Multi-signal: {multi.shape}")  # → (336, 3)

# Datacenter trace (profile mode tiles to the requested window)
dc_trace = loader.load_jax_profiles(
    [S.DC_CPU_UTIL, S.DC_MEM_UTIL],
    source="google",
    start_date="2024-01-01",
    end_date="2024-01-07",
)
print(f"DC trace: {dc_trace.shape}")

# --- 2. Build env ---------------------------------------------------------
case = create_case5()
T = load_profile.shape[0]

# case5 has 3 load buses; profile has 1 column → broadcast to all loads
# Scale each step by peak-normalized shape × per-bus rated midpoint
load_scale = load_profile[:, 0:1] / load_profile[:, 0:1].max()
load_rated = (case.load_d_max + case.load_d_min) / 2.0
profiles_per_load = load_scale * load_rated[None, :]

params = make_trans_params(
    case,
    load_profiles=profiles_per_load,
    max_steps=T,
)
print(f"\nEnv params: max_steps={params.max_steps}, "
      f"load_profiles={params.load_profiles.shape}")

# --- 3. Run one episode ---------------------------------------------------
env = TransGridEnv()
key = jax.random.PRNGKey(42)

obs, state = env.reset(key, params)
print(f"\nInitial obs: shape={obs.shape}")

# JIT step
@jax.jit
def step_fn(key, state, action, params):
    return env.step(key, state, action, params)

total_reward = 0.0
for t in range(min(48, T)):
    key, subkey = jax.random.split(key)
    action = jax.random.uniform(subkey, shape=(case.n_units,), minval=-1.0, maxval=1.0)
    obs, state, reward, costs, done, info = step_fn(subkey, state, action, params)
    total_reward += float(reward)

print(f"Cumulative reward over 48 steps: {total_reward:.2f}")

# --- 4. Registry overview -------------------------------------------------
print("\n=== Registered datasets ===")
for name in loader.registry.list_datasets():
    m = loader.registry.get_manifest(name)
    print(f"  {name}: {m.time_mode}, {m.resolution}, "
          f"signals={m.signals}")
