"""Grid layer base classes for PowerZooJax.

Foundation for the **grid layer** — environments that simulate the
power-flow physics of a transmission or distribution network and let
the agent interact with it through unit dispatch (and optionally
attached DER bundles). Subclasses (``TransGridEnv``, ``DistGridEnv``,
``DistGridEnv3Phase``, …) plug in concrete network topologies and
either DC or AC physics.

Two static flags configure what runs each step:

  - ``physics``     0 = DC linear flow (PTDF-based)
                    1 = AC nonlinear flow (Newton, BFS, or 3-phase BFS)
  - ``solver_mode`` 0 = pure PF  — agent dispatches; solver computes
                                   flows and voltages.
                    1 = DCOPF    — DC OPF determines dispatch.
                    2 = ACOPF    — AC OPF determines dispatch and
                                   solves AC PF (physics flag ignored
                                   in this mode).

Both flags are ``pytree_node=False`` so the solver branches stay
outside the JIT trace.

Resources:
    ``GridState.resource_states`` is a tuple of ``ResourceBundle`` states
    (e.g. ``BatteryBundleState``), one per attached bundle. Battery /
    DER physics live inside the bundles, not as flat fields on the grid
    state — same pattern as the market layer.

Sign convention:
    unit_power_mw > 0      ⇒ generator injecting (active power)
    q_gen > 0              ⇒ generator injecting reactive power (AC)
    node_injection_mw > 0  ⇒ net injection at node (gen − load + DER)
    line_flow_mw > 0       ⇒ flow from the "from" bus to the "to" bus

Time convention:
    ``load_profiles[t % T]`` gives the load vector at step t. The
    strict modulo cycle is deterministic — an RL agent can overfit to
    the exact period. For generalisation benchmarks, add random
    start-offset sampling or Gaussian noise to the profiles before
    passing them in.

Costs:
    ``cost_thermal_weight`` weights the line-overload signal in the
    CMDP cost channel; same convention as the market layer. Subclasses
    may stack additional cost components.

Design note — static pytree across DC / AC modes:
    AC fields (``vm``, ``va``, ``q_gen``, ``line_flow_q_mw``) are
    always present and filled with zeros / ones in DC mode, so the
    pytree shape is invariant. This is required by ``jax.vmap`` /
    ``jax.lax.scan`` over heterogeneous batches.
"""

import chex
from flax import struct


@struct.dataclass
class GridState:
    """State for grid environments (immutable pytree).

    Attributes:
        time_step: Current simulation step (int32).
        done: Whether the episode has terminated.
        unit_power_mw: Active power of each unit [MW] (n_units,).
        line_flow_mw: Active power flow on each line [MW] (n_lines,).
        node_injection_mw: Net injection at each node [MW] (n_nodes,).
        is_safe: True if all line flows are within limits.
        n_violations: Number of line-limit violations.
        total_cost: Cumulative generation cost this episode [$].
        vm: Voltage magnitudes (n_nodes,) [p.u.] — AC mode only (ones in DC).
        va: Voltage angles (n_nodes,) [rad] — AC mode only (zeros in DC).
        q_gen: Reactive power output (n_units,) [MVAr] — AC mode only (zeros in DC).
        line_flow_q_mw: Reactive line flows (n_lines,) [MVAr] — AC mode only (zeros in DC).
    """
    time_step: chex.Array        # int32 scalar
    done: chex.Array             # bool scalar
    unit_power_mw: chex.Array    # (n_units,) float32
    line_flow_mw: chex.Array     # (n_lines,) float32
    node_injection_mw: chex.Array  # (n_nodes,) float32
    is_safe: chex.Array          # bool scalar
    n_violations: chex.Array     # int32 scalar
    total_cost: chex.Array       # float32 scalar
    # AC fields (always present for static pytree; zeros/ones in DC mode)
    vm: chex.Array               # (n_nodes,) float32
    va: chex.Array               # (n_nodes,) float32
    q_gen: chex.Array            # (n_units,) float32
    line_flow_q_mw: chex.Array   # (n_lines,) float32
    resource_states: tuple = ()  # tuple[BundleState, ...]; empty = no resources


@struct.dataclass
class GridParams:
    """Parameters for grid environments (immutable pytree).

    Attributes:
        load_profiles: Load profiles (T, n_loads) [MW].
            Row t gives each load's demand at time step t.
            Access should always use ``state.time_step % T`` to prevent
            out-of-bounds indexing when ``max_steps > T``.

            **RL note**: the default modulo indexing produces a strictly
            deterministic cycle.  Agents can overfit to this exact period.
            For generalization benchmarks, consider adding random start-offset
            sampling or Gaussian noise to the profiles before passing them in.
        max_steps: Maximum steps per episode.
        delta_t_hours: Time step duration in hours (default 0.5 = 30 min).
        steps_per_day: Steps in one day (e.g. 48 for 30-min resolution).
        cost_thermal_weight: Per-MW penalty weight for line overload in cost channel.
        physics: 0=DC, 1=AC (determines which solver to use).
        solver_mode: 0=PF, 1=OPF (determines action→dispatch mapping).

    ``max_steps``, ``delta_t_hours``, ``steps_per_day``, ``physics``, and
    ``solver_mode`` are marked ``pytree_node=False`` so they become static
    under JIT.  This prevents ``ConcretizationTypeError`` when they are used
    in control-flow decisions and ensures no wasted tracing.
    """
    load_profiles: chex.Array     # (T, n_loads) float32
    max_steps: int = struct.field(pytree_node=False, default=48)
    delta_t_hours: float = struct.field(pytree_node=False, default=0.5)
    steps_per_day: int = struct.field(pytree_node=False, default=48)
    cost_thermal_weight: float = struct.field(pytree_node=False, default=1.0)
    physics: int = struct.field(pytree_node=False, default=0)
    solver_mode: int = struct.field(pytree_node=False, default=0)
