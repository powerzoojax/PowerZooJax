"""Backward/Forward Sweep (BFS) Power Flow — pure-JAX algorithm module.

A JAX-native BFS power flow used by ``DistGridEnv`` for **radial
(tree-topology) distribution networks**, implementing the DistFlow
equations on voltage-squared (``v_sq``) as the natural variable. BFS
exploits the tree structure: branch flows are computed leaf-to-root in
the backward sweep, then voltages are updated root-to-leaf in the
forward sweep — typically converging in 1–2 iterations on physical
cases, much cheaper than Newton-Raphson on the same topology.

Translated from PowerZoo's ``cal_pf_dist.py``.

Assumptions:
  - The network is a connected spanning tree rooted at the slack bus.
  - Branch direction is from sending (closer to slack) to receiving (leaf).
  - ``build_radial_topology`` raises if any node is unreachable.

Architecture:
    Setup-time (NumPy, called once):
        build_radial_topology → BFSTopoData
    Runtime (pure JAX, JIT-compilable):
        backward_sweep, forward_sweep, bfs_power_flow → BFSResult

I/O contract:
    Input  : BFSTopoData, p_load_pu, q_load_pu, v_slack
    Output : BFSResult { v_sq, v_mag, p_branch, q_branch, p_loss,
                          q_loss, converged, iterations }

Numerical guards:
    Voltage-squared is floored at ``0.25`` (≡ 0.5 p.u.²) inside both
    sweeps to prevent division-by-zero in loss terms and to keep the
    iteration stable under extreme exploration actions.
"""

from __future__ import annotations

import collections
from typing import Tuple

import numpy as np
import jax.numpy as jnp
import jax.lax as lax
import chex
from flax import struct

from powerzoojax.case.case_data import CaseData


# ---------------------------------------------------------------------------
# Numerical guard constants
# ---------------------------------------------------------------------------

# Minimum voltage squared used as denominator guard in loss calculations.
# 0.25 corresponds to 0.5 p.u. — an extreme undervoltage that prevents
# division by zero while still being physically plausible.
_MIN_V_SQ_GUARD: float = 0.25

# Floor for voltage-squared results in the forward sweep.  Same physical
# interpretation as above: prevents negative / near-zero squared voltages
# from destabilising the iteration.
_MIN_V_SQ_FLOOR: float = 0.25


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@struct.dataclass
class BFSTopoData:
    """Precomputed radial topology data (built once at env construction).

    All matrices are dense JAX arrays for JIT/vmap compatibility.
    """
    path_matrix: chex.Array       # (n_nodes, n_lines) float32
    downstream_matrix: chex.Array # (n_lines, n_nodes) float32
    r_pu: chex.Array              # (n_lines,) float32
    x_pu: chex.Array              # (n_lines,) float32
    sending_node_idx: chex.Array  # (n_lines,) int32
    receiving_node_idx: chex.Array # (n_lines,) int32
    n_nodes: int = struct.field(pytree_node=False, default=0)
    n_lines: int = struct.field(pytree_node=False, default=0)


@struct.dataclass
class BFSResult:
    """Output of BFS power flow solver."""
    v_sq: chex.Array       # (n_nodes,) voltage squared [p.u.^2]
    v_mag: chex.Array      # (n_nodes,) voltage magnitude [p.u.]
    p_branch: chex.Array   # (n_lines,) active branch flow [p.u.]
    q_branch: chex.Array   # (n_lines,) reactive branch flow [p.u.]
    p_loss: chex.Array     # (n_lines,) active branch loss [p.u.]
    q_loss: chex.Array     # (n_lines,) reactive branch loss [p.u.]
    converged: chex.Array  # bool scalar
    iterations: chex.Array # int32 scalar


# ---------------------------------------------------------------------------
# Setup-time functions (NumPy)
# ---------------------------------------------------------------------------

def build_radial_topology(
    n_nodes: int,
    from_nodes: np.ndarray,
    to_nodes: np.ndarray,
    r_pu: np.ndarray,
    x_pu: np.ndarray,
    slack_bus_id: int = 0,
) -> BFSTopoData:
    """Build radial topology from graph data via BFS.

    Args:
        n_nodes: Number of buses.
        from_nodes: (n_lines,) from-bus index for each line.
        to_nodes: (n_lines,) to-bus index for each line.
        r_pu: (n_lines,) resistance [p.u.].
        x_pu: (n_lines,) reactance [p.u.].
        slack_bus_id: Root bus index.

    Returns:
        BFSTopoData with dense path/downstream matrices as JAX arrays.
    """
    n_lines = len(from_nodes)
    from_nodes = np.asarray(from_nodes, dtype=int)
    to_nodes = np.asarray(to_nodes, dtype=int)
    r_pu = np.asarray(r_pu, dtype=float)
    x_pu = np.asarray(x_pu, dtype=float)

    # Build adjacency list
    adj: dict = {i: [] for i in range(n_nodes)}
    for i in range(n_lines):
        adj[from_nodes[i]].append((to_nodes[i], i))
        adj[to_nodes[i]].append((from_nodes[i], i))

    # BFS to build spanning tree
    parent = np.full(n_nodes, -1, dtype=int)
    parent_line = np.full(n_nodes, -1, dtype=int)
    visited = np.zeros(n_nodes, dtype=bool)
    queue = collections.deque([slack_bus_id])
    visited[slack_bus_id] = True

    while queue:
        node = queue.popleft()
        for neighbor, line_idx in adj[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                parent[neighbor] = node
                parent_line[neighbor] = line_idx
                queue.append(neighbor)

    if visited.sum() < n_nodes:
        unreachable = np.where(~visited)[0]
        raise ValueError(f"Unreachable nodes from slack bus {slack_bus_id}: {unreachable}")

    # Sending/receiving node direction in the tree
    sending_nodes = np.where(parent[from_nodes] == to_nodes, to_nodes, from_nodes)
    receiving_nodes = np.where(sending_nodes == from_nodes, to_nodes, from_nodes)

    # Path matrix: path_matrix[node, line] = 1 if line is on root→node path
    path_matrix = np.zeros((n_nodes, n_lines), dtype=np.float32)
    for node in range(n_nodes):
        curr = node
        while parent[curr] != -1:
            line = parent_line[curr]
            if line != -1:
                path_matrix[node, line] = 1.0
            curr = parent[curr]

    # Downstream matrix = path_matrix.T
    downstream_matrix = path_matrix.T.copy()

    return BFSTopoData(
        path_matrix=jnp.array(path_matrix),
        downstream_matrix=jnp.array(downstream_matrix),
        r_pu=jnp.array(r_pu, dtype=jnp.float32),
        x_pu=jnp.array(x_pu, dtype=jnp.float32),
        sending_node_idx=jnp.array(sending_nodes, dtype=jnp.int32),
        receiving_node_idx=jnp.array(receiving_nodes, dtype=jnp.int32),
        n_nodes=n_nodes,
        n_lines=n_lines,
    )


def prepare_bfs(case: CaseData) -> BFSTopoData:
    """Build BFS topology data from a CaseData with distribution parameters.

    Inactive lines (status=0, e.g. normally-open tie switches) are excluded
    so that the BFS spanning tree reflects the actual operating topology.

    Args:
        case: CaseData with line_r, line_x, line_from_idx, line_to_idx.

    Returns:
        BFSTopoData ready for JIT-compiled BFS solver.
    """
    r_pu = np.asarray(case.line_r)
    x_pu = np.asarray(case.line_x)
    from_idx = np.asarray(case.line_from_idx, dtype=int)
    to_idx = np.asarray(case.line_to_idx, dtype=int)

    if case.line_status is not None:
        active = np.asarray(case.line_status) > 0
        r_pu = r_pu[active]
        x_pu = x_pu[active]
        from_idx = from_idx[active]
        to_idx = to_idx[active]

    slack = int(case.slack_bus_idx)

    return build_radial_topology(
        n_nodes=case.n_nodes,
        from_nodes=from_idx,
        to_nodes=to_idx,
        r_pu=r_pu,
        x_pu=x_pu,
        slack_bus_id=slack,
    )


# ---------------------------------------------------------------------------
# Runtime functions (pure JAX, JIT-compilable)
# ---------------------------------------------------------------------------

def backward_sweep(
    topo: BFSTopoData,
    p_load_pu: chex.Array,
    q_load_pu: chex.Array,
    v_sq: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """Backward sweep: branch flows from downstream loads + losses.

    Args:
        topo: BFSTopoData with downstream_matrix, r_pu, x_pu, etc.
        p_load_pu: (n_nodes,) active load [p.u.].
        q_load_pu: (n_nodes,) reactive load [p.u.].
        v_sq: (n_nodes,) voltage squared from previous iteration.

    Returns:
        (p_branch, q_branch) in p.u.
    """
    n_nodes = p_load_pu.shape[0]

    # Base branch flow = sum of downstream loads
    p_branch = topo.downstream_matrix @ p_load_pu
    q_branch = topo.downstream_matrix @ q_load_pu

    # Line losses from previous iteration voltages
    v_sending_sq = jnp.maximum(v_sq[topo.sending_node_idx], _MIN_V_SQ_GUARD)
    i_sq = (p_branch ** 2 + q_branch ** 2) / v_sending_sq
    p_loss = topo.r_pu * i_sq
    q_loss = topo.x_pu * i_sq

    # Losses at receiving nodes, propagated upstream
    p_loss_at_node = jnp.zeros(n_nodes).at[topo.receiving_node_idx].add(p_loss)
    q_loss_at_node = jnp.zeros(n_nodes).at[topo.receiving_node_idx].add(q_loss)

    p_branch = p_branch + topo.downstream_matrix @ p_loss_at_node
    q_branch = q_branch + topo.downstream_matrix @ q_loss_at_node

    return p_branch, q_branch


def forward_sweep(
    topo: BFSTopoData,
    p_branch: chex.Array,
    q_branch: chex.Array,
    v_sq: chex.Array,
    v_slack: float = 1.0,
) -> chex.Array:
    """Forward sweep: voltage drops from root to leaves (DistFlow).

    Args:
        topo: BFSTopoData.
        p_branch: (n_lines,) active branch flow [p.u.].
        q_branch: (n_lines,) reactive branch flow [p.u.].
        v_sq: (n_nodes,) voltage squared from previous iteration.
        v_slack: Slack bus voltage magnitude [p.u.].

    Returns:
        v_sq_new: (n_nodes,) updated voltage squared.
    """
    v_sending_sq = jnp.maximum(v_sq[topo.sending_node_idx], _MIN_V_SQ_GUARD)

    z_sq = topo.r_pu ** 2 + topo.x_pu ** 2
    s_sq = p_branch ** 2 + q_branch ** 2

    term1 = 2.0 * (topo.r_pu * p_branch + topo.x_pu * q_branch)
    term2 = z_sq * s_sq / v_sending_sq
    delta_v_sq = term1 - term2

    total_drop = topo.path_matrix @ delta_v_sq
    v_sq_new = v_slack ** 2 - total_drop

    return jnp.maximum(v_sq_new, _MIN_V_SQ_FLOOR)


def bfs_power_flow(
    topo: BFSTopoData,
    p_load_pu: chex.Array,
    q_load_pu: chex.Array,
    v_slack: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> BFSResult:
    """Run BFS power flow via lax.while_loop (JIT-compilable).

    Args:
        topo: BFSTopoData from build_radial_topology or prepare_bfs.
        p_load_pu: (n_nodes,) active load [p.u.].
        q_load_pu: (n_nodes,) reactive load [p.u.].
        v_slack: Slack bus voltage [p.u.].
        max_iter: Maximum BFS iterations.
        tol: Convergence tolerance on max|ΔV²|.

    Returns:
        BFSResult with voltages, branch flows, losses, convergence info.
    """
    n_nodes = p_load_pu.shape[0]
    n_lines = topo.r_pu.shape[0]
    tol_arr = jnp.float32(tol)

    v_sq0 = jnp.ones(n_nodes)
    p_br0 = jnp.zeros(n_lines)
    q_br0 = jnp.zeros(n_lines)

    # State: (v_sq, p_branch, q_branch, iteration, converged)
    def _cond(state):
        v_sq, p_br, q_br, it, conv = state
        return jnp.logical_and(it < max_iter, jnp.logical_not(conv))

    def _body(state):
        v_sq, _p_br, _q_br, it, _conv = state

        p_branch, q_branch = backward_sweep(topo, p_load_pu, q_load_pu, v_sq)
        v_sq_new = forward_sweep(topo, p_branch, q_branch, v_sq, v_slack)

        max_diff = jnp.max(jnp.abs(v_sq_new - v_sq))
        converged = max_diff < tol_arr

        return (v_sq_new, p_branch, q_branch, it + 1, converged)

    init = (v_sq0, p_br0, q_br0, jnp.int32(0), jnp.bool_(False))
    v_sq_f, p_br_f, q_br_f, iters, conv = lax.while_loop(_cond, _body, init)

    # Compute final losses
    v_sending_sq = jnp.maximum(v_sq_f[topo.sending_node_idx], _MIN_V_SQ_GUARD)
    i_sq = (p_br_f ** 2 + q_br_f ** 2) / v_sending_sq
    p_loss = topo.r_pu * i_sq
    q_loss = topo.x_pu * i_sq

    return BFSResult(
        v_sq=v_sq_f,
        v_mag=jnp.sqrt(jnp.maximum(v_sq_f, 0.0)),
        p_branch=p_br_f,
        q_branch=q_br_f,
        p_loss=p_loss,
        q_loss=q_loss,
        converged=conv,
        iterations=iters,
    )
