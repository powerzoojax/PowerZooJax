"""
CaseData: Pure JAX Arrays for GPU Training

This module defines the core data structure for power system cases.
All data is stored as JAX arrays for efficient GPU computation.

Design Principles:
1. CaseData is a frozen dataclass (immutable)
2. All arrays are JAX arrays (jnp.ndarray)
3. No Python objects or DataFrame - pure numerical data
4. Can be passed through JIT-compiled functions

Usage:
    >>> case_data = create_case5()
    >>> # Use in JIT-compiled training loop
    >>> @jax.jit
    ... def train_step(case_data, state):
    ...     line_flows = case_data.PTDF @ node_injection
    ...     return line_flows
"""

from typing import NamedTuple, Optional
import jax.numpy as jnp
import chex
from flax import struct


@struct.dataclass
class CaseData:
    """Power system case data (pure JAX arrays).
    
    This is the core data structure used in training.
    All data is stored as JAX arrays for GPU acceleration.
    
    Note: This is a frozen dataclass - all fields are immutable.
    Use `case_data.replace(field=new_value)` to create modified copies.
    
    Note: ``name`` was removed because Python ``str`` cannot be traced by
    JAX JIT.  If you need a display name, store it alongside the CaseData.
    
    Attributes:
        n_nodes: Number of buses/nodes (problem dimension).
        n_lines: Number of transmission lines.
        n_units: Number of generators.
        n_loads: Number of loads.
        node_ids: Node IDs (for reference); shape ``(n_nodes,)``.
        node_x: Node x coordinates for plotting; shape ``(n_nodes,)``.
        node_y: Node y coordinates for plotting; shape ``(n_nodes,)``.
        unit_ids: Unit IDs; shape ``(n_units,)``.
        unit_bus_ids: Bus ID where each unit is connected; shape ``(n_units,)``.
        unit_p_min: Minimum power output [MW]; shape ``(n_units,)``.
        unit_p_max: Maximum power output [MW]; shape ``(n_units,)``.
        unit_cost_a: Quadratic cost coefficient [$/MW²h]; shape ``(n_units,)``.
        unit_cost_b: Linear cost coefficient [$/MWh]; shape ``(n_units,)``.
        unit_cost_c: Constant cost coefficient [$/h]; shape ``(n_units,)``.
        line_ids: Line IDs; shape ``(n_lines,)``.
        line_from: From bus ID; shape ``(n_lines,)``.
        line_to: To bus ID; shape ``(n_lines,)``.
        line_x: Line reactance [p.u.]; shape ``(n_lines,)``.
        line_cap: Branch thermal rating [MVA], from MATPOWER ``rateA``.
            **Convention:** In DC mode this is treated as an active-power limit [MW]
            (``Q≈0`` approximation). In AC PF / ACOPF it is used as an apparent-power
            limit [MVA] with thermal magnitude ``√(P²+Q²)``. Do not treat this field
            as pure MW on AC paths when checking thermal limits.
        line_floor: Line capacity lower bound [MW]; shape ``(n_lines,)``.
        load_ids: Load IDs; shape ``(n_loads,)``.
        load_bus_ids: Bus ID where each load is connected; shape ``(n_loads,)``.
        load_d_max: Maximum demand [MW]; shape ``(n_loads,)``.
        load_d_min: Minimum demand [MW]; shape ``(n_loads,)``.
        PTDF: Power Transfer Distribution Factor; shape ``(n_lines, n_nodes)``.
        nodes_units_map: Node-to-unit mapping; shape ``(n_nodes, n_units)``.
        nodes_loads_map: Node-to-load mapping; shape ``(n_nodes, n_loads)``.
        unit_node_idx: Internal node index for each unit (0-indexed); shape ``(n_units,)``.
        load_node_idx: Internal node index for each load; shape ``(n_loads,)``.
        line_from_idx: Internal from-node index for lines; shape ``(n_lines,)``.
        line_to_idx: Internal to-node index for lines; shape ``(n_lines,)``.
        slack_bus_idx: Slack bus internal index (default 0).
    """
    # Dimensions
    n_nodes: int = 0
    n_lines: int = 0
    n_units: int = 0
    n_loads: int = 0
    
    # Node data
    node_ids: chex.Array = None      # (n_nodes,) float or int
    node_x: chex.Array = None        # (n_nodes,) float
    node_y: chex.Array = None        # (n_nodes,) float
    
    # Unit data
    unit_ids: chex.Array = None      # (n_units,)
    unit_bus_ids: chex.Array = None  # (n_units,)
    unit_p_min: chex.Array = None    # (n_units,)
    unit_p_max: chex.Array = None    # (n_units,)
    unit_cost_a: chex.Array = None   # (n_units,) quadratic
    unit_cost_b: chex.Array = None   # (n_units,) linear
    unit_cost_c: chex.Array = None   # (n_units,) constant
    
    # Line data
    line_ids: chex.Array = None      # (n_lines,)
    line_from: chex.Array = None     # (n_lines,)
    line_to: chex.Array = None       # (n_lines,)
    line_x: chex.Array = None        # (n_lines,) reactance
    line_cap: chex.Array = None      # (n_lines,)
    line_floor: chex.Array = None    # (n_lines,)
    
    # Load data
    load_ids: chex.Array = None      # (n_loads,)
    load_bus_ids: chex.Array = None  # (n_loads,)
    load_d_max: chex.Array = None    # (n_loads,)
    load_d_min: chex.Array = None    # (n_loads,)
    
    # Pre-computed matrices
    PTDF: chex.Array = None          # (n_lines, n_nodes)
    nodes_units_map: chex.Array = None  # (n_nodes, n_units)
    nodes_loads_map: chex.Array = None  # (n_nodes, n_loads)
    
    # Internal indices (0-based)
    unit_node_idx: chex.Array = None    # (n_units,) int
    load_node_idx: chex.Array = None    # (n_loads,) int
    line_from_idx: chex.Array = None    # (n_lines,) int
    line_to_idx: chex.Array = None      # (n_lines,) int
    
    # Configuration
    slack_bus_idx: int = 0

    # ========== AC fields (optional, backward-compatible) ==========

    # AC line data (n_lines,)
    line_r: chex.Array = None           # Resistance [p.u.]
    line_b: chex.Array = None           # Total charging susceptance [p.u.]
    line_ratio: chex.Array = None       # Transformer tap ratio (0 = regular line → treated as 1)
    line_angle: chex.Array = None       # Phase shift [degrees]
    line_status: chex.Array = None      # 1 = active, 0 = inactive

    # AC unit data (n_units,)
    unit_q_min: chex.Array = None       # Min reactive power [MVAr]
    unit_q_max: chex.Array = None       # Max reactive power [MVAr]
    unit_pg: chex.Array = None          # Scheduled active power [MW]
    unit_qg: chex.Array = None          # Scheduled reactive power [MVAr]
    unit_vg: chex.Array = None          # Voltage setpoint [p.u.]

    # AC node data (n_nodes,)
    node_type: chex.Array = None        # 1=PQ, 2=PV, 3=Slack (MATPOWER convention)
    node_pd: chex.Array = None          # Active load at bus [MW]
    node_qd: chex.Array = None          # Reactive load at bus [MVAr]
    node_gs: chex.Array = None          # Bus shunt conductance [MW at 1 p.u. V]
    node_bs: chex.Array = None          # Bus shunt susceptance [MVAr at 1 p.u. V]
    node_v_min: chex.Array = None       # Min voltage magnitude [p.u.]
    node_v_max: chex.Array = None       # Max voltage magnitude [p.u.]

    # AC line rating (n_lines,)
    line_rate_a: chex.Array = None       # Long-term thermal rating [MVA]

    # AC unit operational (n_units,)
    unit_status: chex.Array = None       # 1 = in-service, 0 = out-of-service
    unit_mbase: chex.Array = None        # Machine base [MVA]

    # ========== UC fields (optional, backward-compatible) ==========

    # Unit commitment data (n_units,)
    unit_ramp_up: chex.Array = None      # Ramp-up limit [MW/step]
    unit_ramp_down: chex.Array = None    # Ramp-down limit [MW/step]
    unit_min_up_time: chex.Array = None  # Minimum on time [hours]
    unit_min_down_time: chex.Array = None # Minimum off time [hours]
    unit_init_power: chex.Array = None   # Initial power output [MW]
    unit_init_state: chex.Array = None   # Initial on/off (1=on, 0=off)
    unit_startup_cost: chex.Array = None # Startup cost [$]
    unit_no_load_cost: chex.Array = None # No-load cost [$/h]
    unit_keep_time: chex.Array = None    # Initial keep-time [hours]
    unit_fuel_type: chex.Array = None    # Fuel type int (0=unknown,1=nuclear,2=coal,3=gas)

    # ========== Three-phase fields (optional, backward-compatible) ==========

    # Per-phase node loads (n_nodes,) — only populated for three-phase cases
    node_pd_a: chex.Array = None         # Phase-A active load [MW]
    node_qd_a: chex.Array = None         # Phase-A reactive load [MVAr]
    node_pd_b: chex.Array = None         # Phase-B active load [MW]
    node_qd_b: chex.Array = None         # Phase-B reactive load [MVAr]
    node_pd_c: chex.Array = None         # Phase-C active load [MW]
    node_qd_c: chex.Array = None         # Phase-C reactive load [MVAr]

    # System
    base_mva: float = 100.0

    # Distribution topology (populated by BFS setup)
    path_matrix: chex.Array = None       # (n_lines, n_nodes) BFS path indicator
    downstream_matrix: chex.Array = None # (n_lines, n_nodes) downstream node indicator
    sending_node_idx: chex.Array = None  # (n_lines,) sending-end bus index
    receiving_node_idx: chex.Array = None # (n_lines,) receiving-end bus index

    @property
    def gen_cost_coeffs(self) -> chex.Array:
        """Backward-compatible generator cost coefficients alias.

        Older smoke scripts expect ``case.gen_cost_coeffs`` to expose the
        per-generator quadratic cost triplets. The canonical storage remains
        ``unit_cost_a`` / ``unit_cost_b`` / ``unit_cost_c``.
        """
        return jnp.stack(
            [self.unit_cost_a, self.unit_cost_b, self.unit_cost_c], axis=-1
        )


class CaseArrays(NamedTuple):
    """Raw arrays for case definition (before computing matrices).
    
    Used as intermediate format when defining cases.
    """
    nodes: jnp.ndarray   # (n_nodes, n_cols) - id, x, y, ...
    units: jnp.ndarray   # (n_units, n_cols) - id, bus_id, costs, limits, ...
    lines: jnp.ndarray   # (n_lines, n_cols) - id, from, to, x, limits, ...
    loads: jnp.ndarray   # (n_loads, n_cols) - id, bus_id, limits, ...


def validate_case_data(case_data: CaseData) -> bool:
    """Validate CaseData consistency.
    
    Checks:
    - Dimensions match array shapes
    - All required arrays are present
    - PTDF shape matches (n_lines, n_nodes)
    
    Args:
        case_data: CaseData to validate
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If validation fails
    """
    # Check dimensions
    if case_data.node_ids is not None:
        assert len(case_data.node_ids) == case_data.n_nodes, \
            f"node_ids length {len(case_data.node_ids)} != n_nodes {case_data.n_nodes}"
    
    if case_data.unit_ids is not None:
        assert len(case_data.unit_ids) == case_data.n_units, \
            f"unit_ids length {len(case_data.unit_ids)} != n_units {case_data.n_units}"
    
    if case_data.line_ids is not None:
        assert len(case_data.line_ids) == case_data.n_lines, \
            f"line_ids length {len(case_data.line_ids)} != n_lines {case_data.n_lines}"
    
    if case_data.load_ids is not None:
        assert len(case_data.load_ids) == case_data.n_loads, \
            f"load_ids length {len(case_data.load_ids)} != n_loads {case_data.n_loads}"
    
    # Check PTDF shape
    if case_data.PTDF is not None:
        assert case_data.PTDF.shape == (case_data.n_lines, case_data.n_nodes), \
            f"PTDF shape {case_data.PTDF.shape} != ({case_data.n_lines}, {case_data.n_nodes})"

    # Check UC unit arrays (n_units,) when present
    _uc_unit_fields = [
        'unit_ramp_up', 'unit_ramp_down', 'unit_min_up_time',
        'unit_min_down_time', 'unit_init_power', 'unit_init_state',
        'unit_startup_cost', 'unit_no_load_cost', 'unit_keep_time',
        'unit_fuel_type', 'unit_status', 'unit_mbase',
    ]
    for fname in _uc_unit_fields:
        arr = getattr(case_data, fname, None)
        if arr is not None:
            assert len(arr) == case_data.n_units, \
                f"{fname} length {len(arr)} != n_units {case_data.n_units}"

    # Check three-phase node arrays (n_nodes,) when present
    _3ph_node_fields = [
        'node_pd_a', 'node_qd_a', 'node_pd_b', 'node_qd_b',
        'node_pd_c', 'node_qd_c',
    ]
    for fname in _3ph_node_fields:
        arr = getattr(case_data, fname, None)
        if arr is not None:
            assert len(arr) == case_data.n_nodes, \
                f"{fname} length {len(arr)} != n_nodes {case_data.n_nodes}"

    return True
