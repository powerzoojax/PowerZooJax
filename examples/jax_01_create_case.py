"""
Example 01: Creating and Inspecting Power System Cases (PowerZooJAX)

This example demonstrates:
1. Creating a case using JAX-native data structures
2. Printing case info (PowerZoo-compatible format)
3. Plotting network topology
4. Testing JAX operations (PTDF, power flow)

⚠️ Note: Printing and plotting run on CPU.
   Only the JAX computations (step 4) utilize GPU.

Run:
    python examples/jax_01_create_case.py
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax
import jax.numpy as jnp


# Output directory for plots
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "x_jax01_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def example_create_case():
    """Example 1: Create a case and inspect basic info."""
    print("=" * 60)
    print("Example 1: Creating a Case")
    print("=" * 60)
    
    from powerzoojax.case import load_case, CaseInfo
    
    # Create Case5 (5-bus test system)
    case = load_case("5")
    
    # Print summary (CPU operation)
    info = CaseInfo(case, warn=True, name="case5")
    print(info.summary())
    print()


def example_print_tables():
    """Example 2: Print full tables (PowerZoo format)."""
    print("=" * 60)
    print("Example 2: Printing Tables (PowerZoo Format)")
    print("=" * 60)
    
    from powerzoojax.case import create_case5, CaseInfo
    
    case = create_case5()
    info = CaseInfo(case, warn=True, name="case5")

    # Print individual tables
    print(info.nodes)
    print()
    print(info.units)
    print()
    print(info.lines)
    print()
    print(info.loads)
    print()


def example_plot_topology():
    """Example 3: Plot network topology."""
    print("=" * 60)
    print("Example 3: Plotting Topology")
    print("=" * 60)
    
    try:
        import matplotlib.pyplot as plt
        from powerzoojax.case import create_case5, CasePlotter
        
        case = create_case5()
        plotter = CasePlotter(case, warn=True)
        
        # Plot topology
        fig, ax = plotter.plot_topology(
            layout='case',  # Use case-defined coordinates
            title="Case5 - 5 Bus Test System"
        )
        
        # Save figure
        save_path = os.path.join(OUTPUT_DIR, "case5_topology.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved topology plot to: {save_path}")
        
        plt.close()
        
    except ImportError as e:
        print(f"Plotting requires matplotlib and networkx: {e}")


def example_jax_computations():
    """Example 4: JAX computations (GPU-accelerated)."""
    print("=" * 60)
    print("Example 4: JAX Computations (GPU)")
    print("=" * 60)
    
    from powerzoojax.case import create_case5
    
    case = create_case5()
    
    # Check JAX device
    print(f"JAX default backend: {jax.default_backend()}")
    print(f"Available devices: {jax.devices()}")
    print()
    
    # PTDF is already computed and stored as JAX array
    print(f"PTDF shape: {case.PTDF.shape}")
    print(f"PTDF dtype: {case.PTDF.dtype}")
    print()
    
    # Example: Compute line flows from node injections
    # This runs on GPU if available!
    @jax.jit
    def compute_line_flows(case_ptdf, node_injection):
        """Compute line flows from node power injections."""
        return case_ptdf @ node_injection
    
    # Create example node injection (generation - load)
    # Assume all generators at p_min, loads at d_max
    unit_power = case.unit_p_min.copy()
    
    # Map unit power to nodes
    node_gen = case.nodes_units_map @ unit_power
    node_load = case.nodes_loads_map @ case.load_d_max
    node_injection = node_gen - node_load
    
    # Balance at slack bus
    slack_idx = case.slack_bus_idx
    imbalance = node_injection.sum()
    node_injection = node_injection.at[slack_idx].add(-imbalance)
    
    print("Node injection (gen - load):")
    print(f"  {node_injection}")
    print(f"  Sum: {float(node_injection.sum()):.6f} (should be ~0)")
    print()
    
    # Compute line flows (JIT-compiled, GPU)
    line_flows = compute_line_flows(case.PTDF, node_injection)
    
    print("Line flows (MW):")
    for i, flow in enumerate(line_flows):
        print(f"  Line {i+1}: {float(flow):>8.2f} MW")
    print()
    
    # Check constraints
    safe = jnp.all((line_flows >= case.line_floor) & (line_flows <= case.line_cap))
    print(f"All lines within limits: {bool(safe)}")


def example_case33bw():
    """Example 5: Larger case (33-bus distribution system)."""
    print("=" * 60)
    print("Example 5: Case33bw (33-Bus Distribution System)")
    print("=" * 60)
    
    from powerzoojax.case import create_case33bw, CaseInfo
    
    case = create_case33bw()
    info = CaseInfo(case, warn=True, name="case33bw")
    print(info.summary())
    print()
    
    # Plot if matplotlib available
    try:
        import matplotlib.pyplot as plt
        from powerzoojax.case import CasePlotter
        
        plotter = CasePlotter(case, warn=False)
        fig, ax = plotter.plot_topology(
            layout='feeder',  # Feeder layout for radial networks
            figsize=(16, 10),
            title="IEEE 33-Bus Distribution System"
        )
        
        save_path = os.path.join(OUTPUT_DIR, "case33bw_topology.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved topology plot to: {save_path}")
        plt.close()
        
    except ImportError:
        print("Skipping plot (matplotlib not available)")


def example_ptdf_heatmap():
    """Example 6: Visualize PTDF matrix."""
    print("=" * 60)
    print("Example 6: PTDF Heatmap")
    print("=" * 60)
    
    try:
        import matplotlib.pyplot as plt
        from powerzoojax.case import create_case5, CasePlotter
        
        case = create_case5()
        plotter = CasePlotter(case, warn=True)
        
        fig, ax = plotter.plot_ptdf_heatmap(
            figsize=(10, 6),
            title="Case5 - PTDF Matrix"
        )
        
        save_path = os.path.join(OUTPUT_DIR, "case5_ptdf.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved PTDF heatmap to: {save_path}")
        plt.close()
        
    except ImportError as e:
        print(f"Plotting requires matplotlib: {e}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PowerZooJAX - Case Examples")
    print("=" * 60 + "\n")
    
    # Run all examples
    example_create_case()
    print()
    
    example_print_tables()
    print()
    
    example_jax_computations()
    print()
    
    example_case33bw()
    print()
    
    example_plot_topology()
    print()
    
    example_ptdf_heatmap()
    print()
    
    print("=" * 60)
    print("All examples completed!")
    print(f"Output saved to: {OUTPUT_DIR}")
    print("=" * 60)
