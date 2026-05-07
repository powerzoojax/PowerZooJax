"""
Example 00: Verify JAX Device and GPU Acceleration

This script helps you verify:
1. Which devices JAX can see (CPU, GPU, TPU)
2. Where arrays are actually stored
3. Performance comparison between CPU and GPU
4. Whether your code is truly GPU-accelerated

Run:
    python examples/jax_00_verify_device.py
"""

import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def check_jax_installation():
    """Step 1: Check JAX installation and available devices."""
    print("=" * 70)
    print("Step 1: JAX Installation & Device Check")
    print("=" * 70)
    
    import jax
    import jax.numpy as jnp
    
    print(f"JAX version: {jax.__version__}")
    print(f"Default backend: {jax.default_backend()}")
    print(f"Available devices: {jax.devices()}")
    print()
    
    # Check for GPU
    try:
        gpu_devices = jax.devices('gpu')
        print(f"✅ GPU detected: {gpu_devices}")
        has_gpu = True
    except RuntimeError:
        print("❌ No GPU detected. Running on CPU only.")
        print("   To use GPU, install jax with CUDA support:")
        print("   pip install --upgrade 'jax[cuda12]'")
        has_gpu = False
    
    print()
    return has_gpu


def check_array_device():
    """Step 2: Check where arrays are stored."""
    print("=" * 70)
    print("Step 2: Array Device Location")
    print("=" * 70)
    
    import jax
    import jax.numpy as jnp
    
    # Create an array
    x = jnp.ones((1000, 1000))
    
    # Check device
    print(f"Array shape: {x.shape}")
    print(f"Array dtype: {x.dtype}")
    print(f"Array devices: {x.devices()}")  # Shows which device(s) the array is on
    
    # For JAX arrays, you can also check:
    print(f"Is sharded: {x.is_fully_addressable}")
    print()
    
    # Explicit device placement
    if len(jax.devices()) > 1 or jax.default_backend() != 'cpu':
        cpu = jax.devices('cpu')[0]
        print(f"CPU device: {cpu}")
        
        # Put array on CPU explicitly
        x_cpu = jax.device_put(x, cpu)
        print(f"Array on CPU: {x_cpu.devices()}")
        
        try:
            gpu = jax.devices('gpu')[0]
            x_gpu = jax.device_put(x, gpu)
            print(f"Array on GPU: {x_gpu.devices()}")
        except:
            print("(No GPU available for comparison)")
    
    print()


def benchmark_matmul():
    """Step 3: Benchmark matrix multiplication on CPU vs GPU."""
    print("=" * 70)
    print("Step 3: Performance Benchmark (Matrix Multiplication)")
    print("=" * 70)
    
    import jax
    import jax.numpy as jnp
    
    # Matrix size for benchmark
    n = 2000
    print(f"Matrix size: {n} x {n}")
    print()
    
    # Create random matrices
    key = jax.random.PRNGKey(0)
    A = jax.random.normal(key, (n, n))
    B = jax.random.normal(key, (n, n))
    
    # JIT-compiled matmul
    @jax.jit
    def matmul(a, b):
        return a @ b
    
    # Warmup (compilation)
    _ = matmul(A, B).block_until_ready()
    
    # Benchmark on default device
    n_runs = 10
    start = time.time()
    for _ in range(n_runs):
        result = matmul(A, B).block_until_ready()  # block_until_ready ensures GPU sync
    elapsed = time.time() - start
    
    print(f"Default device ({jax.default_backend()}):")
    print(f"  {n_runs} runs in {elapsed:.4f}s")
    print(f"  {elapsed/n_runs*1000:.2f} ms per matmul")
    print(f"  {n**3 * 2 * n_runs / elapsed / 1e9:.2f} GFLOPS")
    print()
    
    # If GPU available, also test CPU explicitly
    try:
        gpu_devices = jax.devices('gpu')
        cpu = jax.devices('cpu')[0]
        
        # Put on CPU explicitly
        A_cpu = jax.device_put(A, cpu)
        B_cpu = jax.device_put(B, cpu)
        
        # Warmup on CPU
        @jax.jit
        def matmul_cpu(a, b):
            return a @ b
        
        _ = matmul_cpu(A_cpu, B_cpu).block_until_ready()
        
        # Benchmark CPU
        start = time.time()
        for _ in range(n_runs):
            result = matmul_cpu(A_cpu, B_cpu).block_until_ready()
        cpu_elapsed = time.time() - start
        
        print(f"CPU (explicit):")
        print(f"  {n_runs} runs in {cpu_elapsed:.4f}s")
        print(f"  {cpu_elapsed/n_runs*1000:.2f} ms per matmul")
        print()
        
        print(f"🚀 GPU Speedup: {cpu_elapsed/elapsed:.1f}x faster")
        
    except RuntimeError:
        print("(No GPU available for CPU comparison)")
    
    print()


def benchmark_powerzoojax_env():
    """Step 4: Benchmark PowerZooJAX environment."""
    print("=" * 70)
    print("Step 4: PowerZooJAX Environment Benchmark")
    print("=" * 70)
    
    import jax
    import jax.numpy as jnp
    
    from powerzoojax.case import create_case5
    from powerzoojax.envs import TransGridEnv, make_trans_params

    env = TransGridEnv()
    case = create_case5()
    params = make_trans_params(case)
    
    print(f"Environment: {env.name}")
    print(f"Running on: {jax.default_backend()}")
    print()
    
    # JIT-compiled functions
    @jax.jit
    def reset_fn(key):
        return env.reset(key, params)
    
    @jax.jit
    def step_fn(key, state, action):
        return env.step(key, state, action, params)
    
    # Warmup
    key = jax.random.PRNGKey(0)
    obs, state = reset_fn(key)
    action = jnp.ones(case.n_units) * 0.5
    _ = step_fn(key, state, action)
    
    # Single environment benchmark
    n_steps = 10000
    key = jax.random.PRNGKey(42)
    obs, state = reset_fn(key)
    
    start = time.time()
    for _ in range(n_steps):
        key, subkey = jax.random.split(key)
        action = jax.random.uniform(subkey, shape=(case.n_units,))
        obs, state, reward, costs, done, info = step_fn(subkey, state, action)
        # Force synchronization to ensure accurate timing
        obs.block_until_ready()
    single_time = time.time() - start
    
    print(f"Single environment ({n_steps} steps):")
    print(f"  Time: {single_time:.4f}s")
    print(f"  Throughput: {n_steps/single_time:,.0f} steps/s")
    print()
    
    # Parallel environments benchmark (vmap)
    n_envs = 256
    n_steps_parallel = 100
    
    @jax.jit
    def batch_reset(keys):
        return jax.vmap(lambda k: env.reset(k, params))(keys)
    
    @jax.jit
    def batch_step(keys, states, actions):
        return jax.vmap(lambda k, s, a: env.step(k, s, a, params))(keys, states, actions)
    
    # Warmup
    keys = jax.random.split(key, n_envs)
    obs, states = batch_reset(keys)
    actions = jnp.ones((n_envs, case.n_units)) * 0.5
    _ = batch_step(keys, states, actions)
    
    # Benchmark
    key = jax.random.PRNGKey(123)
    keys = jax.random.split(key, n_envs)
    obs, states = batch_reset(keys)
    
    start = time.time()
    for _ in range(n_steps_parallel):
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, n_envs)
        actions = jax.random.uniform(subkey, shape=(n_envs, case.n_units))
        obs, states, rewards, costs, dones, infos = batch_step(keys, states, actions)
        obs.block_until_ready()
    parallel_time = time.time() - start
    
    total_steps = n_envs * n_steps_parallel
    print(f"Parallel environments ({n_envs} envs × {n_steps_parallel} steps = {total_steps:,} total):")
    print(f"  Time: {parallel_time:.4f}s")
    print(f"  Throughput: {total_steps/parallel_time:,.0f} steps/s")
    print()
    
    # Compare sequential vs parallel
    sequential_equivalent_time = (total_steps / n_steps) * single_time
    print(f"📊 Parallel speedup: {sequential_equivalent_time/parallel_time:.1f}x vs sequential")
    print()


def show_device_tips():
    """Step 5: Show tips for using GPU."""
    print("=" * 70)
    print("Step 5: Tips for GPU Acceleration")
    print("=" * 70)
    
    import jax
    
    print("""
🔧 Installation (if no GPU detected):
   
   # For CUDA 12.x:
   pip install --upgrade "jax[cuda12]"
   
   # For CUDA 11.x:
   pip install --upgrade "jax[cuda11_pip]"
   
   # Check CUDA version:
   nvcc --version

🔧 Environment Variables:
   
   # Force CPU (for testing):
   JAX_PLATFORMS=cpu python your_script.py
   
   # Specify GPU:
   CUDA_VISIBLE_DEVICES=0 python your_script.py
   
   # Enable memory preallocation (faster but uses more memory):
   XLA_PYTHON_CLIENT_PREALLOCATE=true python your_script.py
   
   # Set memory fraction:
   XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 python your_script.py

🔧 Code Best Practices:
   
   1. Use @jax.jit for all hot paths
   2. Use .block_until_ready() for accurate timing
   3. Avoid Python loops over JAX arrays - use jax.lax.scan instead
   4. Batch operations with jax.vmap for parallelism
   5. Minimize CPU-GPU transfers (avoid frequent .numpy() calls)

🔧 Verify GPU Usage:
   
   # Check GPU memory in another terminal:
   watch -n 1 nvidia-smi
   
   # Or in Python:
   import jax
   print(jax.devices('gpu'))  # Should show GPU device
""")
    
    print(f"Current status: Running on {jax.default_backend().upper()}")
    print()


def force_cpu_comparison():
    """Bonus: Force CPU mode for comparison."""
    print("=" * 70)
    print("Bonus: Force CPU Mode Comparison")
    print("=" * 70)
    
    print("""
To compare CPU vs GPU performance:

1. Run normally (uses GPU if available):
   python examples/jax_00_verify_device.py

2. Force CPU mode:
   JAX_PLATFORMS=cpu python examples/jax_00_verify_device.py

Compare the throughput numbers to see the GPU speedup!
""")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PowerZooJAX - Device Verification & Benchmark")
    print("=" * 70 + "\n")
    
    has_gpu = check_jax_installation()
    check_array_device()
    benchmark_matmul()
    benchmark_powerzoojax_env()
    show_device_tips()
    force_cpu_comparison()
    
    print("=" * 70)
    print("Verification complete!")
    print("=" * 70)
