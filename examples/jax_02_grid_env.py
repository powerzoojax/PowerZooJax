"""
Example 02: Grid Environment Usage (PowerZooJAX)

This example demonstrates:
1. Creating a TransGridEnv with CaseData
2. Basic reset/step operations
3. JIT compilation for fast execution
4. vmap for parallel environments
5. Running a simple rollout

Run:
    python examples/jax_02_grid_env.py
"""

import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax
import jax.numpy as jnp


def example_basic_usage():
    """Example 1: Basic environment usage."""
    print("=" * 60)
    print("Example 1: Basic TransGridEnv Usage")
    print("=" * 60)
    
    from powerzoojax.case import create_case5
    from powerzoojax.envs import TransGridEnv, make_trans_params

    # Create environment and parameters
    env = TransGridEnv()
    case = create_case5()
    params = make_trans_params(case)

    print(f"Environment: {env.name}")
    print(f"  Nodes: {case.n_nodes}")
    print(f"  Lines: {case.n_lines}")
    print(f"  Units: {case.n_units}")
    print(f"  Max steps: {params.max_steps}")
    print(f"  Action dim: {case.n_units}")
    obs_dim = case.n_lines + case.n_loads + case.n_units + 2
    print(f"  Obs dim: {obs_dim}")
    
    # Reset
    key = jax.random.PRNGKey(42)
    obs, state = env.reset(key, params)
    
    print(f"\nInitial state:")
    print(f"  Obs shape: {obs.shape}")
    print(f"  Time step: {state.time_step}")
    print(f"  Is safe: {state.is_safe}")
    print(f"  N violations: {state.n_violations}")
    print(f"  Total generation: {float(jnp.sum(state.unit_power_mw)):.2f} MW")
    tidx = int(state.time_step) % int(params.load_profiles.shape[0])
    node_ld = case.nodes_loads_map @ params.load_profiles[tidx]
    print(f"  Total load: {float(jnp.sum(node_ld)):.2f} MW")
    
    # Step with a random action
    action = jax.random.uniform(key, shape=(case.n_units,), minval=-1.0, maxval=1.0)
    print(f"\nAction (in [-1, 1]): {action}")
    
    obs, new_state, reward, costs, done, info = env.step(key, state, action, params)
    
    print(f"\nAfter step:")
    print(f"  Reward: {float(reward):.4f}")
    print(f"  Constraint names: {env.constraint_names(params)}")
    print(f"  Costs: {costs}")
    print(f"  Done: {done}")
    print(f"  Is safe: {new_state.is_safe}")
    print(f"  Info: {info}")


def example_jit_performance():
    """Example 2: JIT compilation performance."""
    print("\n" + "=" * 60)
    print("Example 2: JIT Compilation Performance")
    print("=" * 60)
    
    from powerzoojax.case import create_case5
    from powerzoojax.envs import TransGridEnv, make_trans_params

    env = TransGridEnv()
    params = make_trans_params(create_case5())
    case = params.case

    # Create JIT-compiled functions
    @jax.jit
    def reset_fn(key):
        return env.reset(key, params)
    
    @jax.jit
    def step_fn(key, state, action):
        return env.step(key, state, action, params)
    
    key = jax.random.PRNGKey(0)
    
    # Warm up (first call compiles)
    print("Warming up (compiling)...")
    obs, state = reset_fn(key)
    action = jnp.ones(case.n_units) * 0.5
    _ = step_fn(key, state, action)
    
    # Benchmark
    n_steps = 1000
    
    # Non-JIT timing
    print(f"\nRunning {n_steps} steps (non-JIT)...")
    start = time.time()
    obs, state = env.reset(key, params)
    for i in range(n_steps):
        key, subkey = jax.random.split(key)
        action = jax.random.uniform(subkey, shape=(case.n_units,), minval=-1.0, maxval=1.0)
        obs, state, _, _, _, _ = env.step(subkey, state, action, params)
    non_jit_time = time.time() - start
    print(f"  Time: {non_jit_time:.4f}s ({n_steps/non_jit_time:.1f} steps/s)")
    
    # JIT timing
    print(f"\nRunning {n_steps} steps (JIT)...")
    key = jax.random.PRNGKey(0)
    start = time.time()
    obs, state = reset_fn(key)
    for i in range(n_steps):
        key, subkey = jax.random.split(key)
        action = jax.random.uniform(subkey, shape=(case.n_units,), minval=-1.0, maxval=1.0)
        obs, state, _, _, _, _ = step_fn(subkey, state, action)
    jit_time = time.time() - start
    print(f"  Time: {jit_time:.4f}s ({n_steps/jit_time:.1f} steps/s)")
    print(f"  Speedup: {non_jit_time/jit_time:.2f}x")


def example_parallel_envs():
    """Example 3: Parallel environments with vmap."""
    print("\n" + "=" * 60)
    print("Example 3: Parallel Environments (vmap)")
    print("=" * 60)
    
    from powerzoojax.case import create_case5
    from powerzoojax.envs import TransGridEnv, make_trans_params

    env = TransGridEnv()
    params = make_trans_params(create_case5())
    case = params.case

    n_envs = 128
    n_steps = 100
    
    print(f"Running {n_envs} parallel environments for {n_steps} steps...")
    
    # Vectorized reset
    @jax.jit
    def batch_reset(keys):
        return jax.vmap(lambda k: env.reset(k, params))(keys)
    
    # Vectorized step
    @jax.jit
    def batch_step(keys, states, actions):
        return jax.vmap(lambda k, s, a: env.step(k, s, a, params))(keys, states, actions)
    
    # Initialize
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, n_envs)
    obs, states = batch_reset(keys)
    
    print(f"  Batch obs shape: {obs.shape}")
    
    # Run rollout
    start = time.time()
    total_rewards = jnp.zeros(n_envs)
    
    for step in range(n_steps):
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, n_envs)
        actions = jax.random.uniform(subkey, shape=(n_envs, case.n_units), minval=-1.0, maxval=1.0)
        
        obs, states, rewards, costs, dones, infos = batch_step(keys, states, actions)
        total_rewards = total_rewards + rewards
    
    elapsed = time.time() - start
    total_steps = n_envs * n_steps
    
    print(f"  Total steps: {total_steps}")
    print(f"  Time: {elapsed:.4f}s")
    print(f"  Throughput: {total_steps/elapsed:,.0f} steps/s")
    print(f"  Mean reward: {float(jnp.mean(total_rewards)):.2f}")


def example_rollout():
    """Example 4: Complete episode rollout."""
    print("\n" + "=" * 60)
    print("Example 4: Episode Rollout")
    print("=" * 60)
    
    from powerzoojax.case import create_case5
    from powerzoojax.envs import TransGridEnv, make_trans_params

    env = TransGridEnv()
    case = create_case5()
    params = make_trans_params(case)

    # Override max_steps for shorter episode
    params = params.replace(max_steps=24)
    
    @jax.jit
    def rollout(key):
        """Run a complete episode."""
        obs, state = env.reset(key, params)
        
        def step_fn(carry, _):
            key, state, total_reward = carry
            key, subkey = jax.random.split(key)
            
            # Simple policy: random action
            action = jax.random.uniform(subkey, shape=(case.n_units,), minval=-1.0, maxval=1.0)

            obs, new_state, reward, costs, done, info = env.step(
                subkey, state, action, params
            )
            return (key, new_state, total_reward + reward), (obs, reward, done)
        
        (_, final_state, total_reward), (obs_history, reward_history, done_history) = jax.lax.scan(
            step_fn,
            (key, state, jnp.array(0.0)),
            jnp.arange(params.max_steps)
        )
        
        return total_reward, final_state, obs_history, reward_history
    
    # Run rollout
    key = jax.random.PRNGKey(0)
    total_reward, final_state, obs_history, reward_history = rollout(key)
    
    print(f"Episode length: {params.max_steps}")
    print(f"Total reward: {float(total_reward):.2f}")
    print(f"Final time step: {final_state.time_step}")
    print(f"Obs history shape: {obs_history.shape}")
    print(f"Reward history shape: {reward_history.shape}")
    print(f"Min reward: {float(jnp.min(reward_history)):.2f}")
    print(f"Max reward: {float(jnp.max(reward_history)):.2f}")


def example_case33():
    """Example 5: Using Case33bw (larger grid)."""
    print("\n" + "=" * 60)
    print("Example 5: Larger Grid (Case33bw)")
    print("=" * 60)
    
    from powerzoojax.case import create_case33bw
    from powerzoojax.envs import TransGridEnv, make_trans_params

    env = TransGridEnv()
    case = create_case33bw()
    params = make_trans_params(case)

    print(f"Case33bw:")
    print(f"  Nodes: {case.n_nodes}")
    print(f"  Lines: {case.n_lines}")
    print(f"  Units: {case.n_units}")
    print(f"  Loads: {case.n_loads}")
    obs_dim = case.n_lines + case.n_loads + case.n_units + 2
    print(f"  Obs dim: {obs_dim}")
    
    # Reset and step
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key, params)
    
    print(f"\nInitial state:")
    print(f"  Obs shape: {obs.shape}")
    print(f"  Total generation: {float(jnp.sum(state.unit_power_mw)):.4f} MW")
    tidx = int(state.time_step) % int(params.load_profiles.shape[0])
    node_ld = case.nodes_loads_map @ params.load_profiles[tidx]
    print(f"  Total load: {float(jnp.sum(node_ld)):.4f} MW")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PowerZooJAX - Grid Environment Examples")
    print("=" * 60 + "\n")
    
    example_basic_usage()
    example_jit_performance()
    example_parallel_envs()
    example_rollout()
    example_case33()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
