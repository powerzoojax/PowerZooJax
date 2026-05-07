"""DC power-flow primitives shared by grid and market envs.

A small set of pure JAX functions used by both ``TransGridEnv`` /
``DistGridEnv`` and the market envs (``CostBasedMarketEnv``,
``BidBasedMarketEnv``, GenCos MARL) for DC-side physics, safety
checking, and quadratic-cost evaluation. All functions are pure,
JIT-compilable, and vmap-safe.

DC power flow (``dc_power_flow`` / ``dc_power_flow_with_check``):
    node_injection = nodes_units_map @ unit_power − node_load
    line_flow      = PTDF @ node_injection
    The slack bus absorbs the system-wide imbalance; the function
    returns ``actual_unit_power_mw`` with the slack adjustment folded
    back into the generator outputs. Load-side correction is
    numerically equivalent but exposing the post-slack dispatch lets
    cost / constraint checks see the *true* generation, not the
    pre-slack RL setpoint — agents cannot exploit "free" slack energy.
    Slack-bus generators are treated as an infinite-capacity reference;
    their p_min / p_max are *not* clamped here — enforce them in the
    env or agent if needed.

Safety checks:
    ``safety_check``      DC active-power limit; returns
                          (is_safe, n_violations, cost_thermal_MW).
                          ``cost_thermal`` is in MW (sum of overruns
                          and underruns), not currency — multiply by
                          a $/MW penalty weight in the env layer.
    ``ac_thermal_check``  AC apparent-power (MVA) limit; parallel
                          triple in MVA. Uses
                          ``max(|S_from|, |S_to|)`` when
                          ``use_both_ends=True`` (ACPF semantics),
                          else ``|S_from|`` only (ACOPF from-end).
    Do not reuse the DC ``safety_check`` for AC flows — different
    units (MW vs MVA) and different physics.

Generation cost (``compute_generation_cost``):
    Marginal cost  MC(p) = a·p² + b·p + c   (project-wide convention)
    Total cost     TC(p) = (a/3)·p³ + (b/2)·p² + c·p     (∫ MC dp)
    Note the deliberate divergence from MATPOWER, which stores TC
    coefficients directly with linear MC; the function-level docstring
    documents the conversion.

Baseline dispatch (``proportional_dispatch``):
    Allocates load above Σ p_min proportionally to per-unit headroom
    ``(p_max − p_min)``. Used by simple baselines and as an OPF
    warm-start; clamps silently to ``p_max`` under over-demand to
    stay JIT-safe.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
import chex

from powerzoojax.case.case_data import CaseData


# ============ DC Power Flow ============

# Note: Not decorated with @jax.jit because CaseData is a pytree of JAX arrays.
# Callers may compile explicitly: jit_fn = jax.jit(dc_power_flow)
# Same pattern applies to dc_power_flow_with_check.
def dc_power_flow(
    case: CaseData,
    unit_power_mw: chex.Array,
    node_load_mw: chex.Array,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Compute DC power flow.

    The slack bus absorbs the power imbalance.  The returned
    ``actual_unit_power_mw`` reflects the true generation including slack
    adjustment — use it for cost computation and constraint checking so
    that RL agents cannot exploit "free" slack energy.

    Args:
        case: CaseData with PTDF, nodes_units_map, slack_bus_idx.
        unit_power_mw: Generator outputs (n_units,) [MW].
        node_load_mw: Nodal load demand (n_nodes,) [MW].

    Returns:
        line_flow_mw: Line active power flows (n_lines,) [MW].
        node_injection_mw: Net nodal injection (n_nodes,) [MW]. Imbalance is
            absorbed by adjusting slack-bus load (load-side adjustment); numerically
            equivalent to adjusting slack-bus generation.
        actual_unit_power_mw: Generator outputs with slack adjustment (n_units,) [MW].
            Slack-bus units are treated as an infinite-capacity reference; p_min/p_max
            are not clamped here. Enforce slack limits in the env or agent if needed.
    """
    # Aggregate unit power to nodes: (n_nodes, n_units) @ (n_units,) -> (n_nodes,)
    node_gen_mw = case.nodes_units_map @ unit_power_mw

    # Slack bus absorbs imbalance
    total_gen = jnp.sum(unit_power_mw)
    total_load = jnp.sum(node_load_mw)
    slack_imbalance = total_gen - total_load
    node_load_balanced = node_load_mw.at[case.slack_bus_idx].add(slack_imbalance)

    # Attribute slack power deficit/surplus to slack-bus units.
    # slack_delta > 0 means the slack unit must produce MORE than dispatched.
    slack_delta = total_load - total_gen
    slack_unit_weight = case.nodes_units_map[case.slack_bus_idx]  # (n_units,)
    slack_total_weight = jnp.sum(slack_unit_weight)
    slack_share = jnp.where(
        slack_total_weight > 0.0,
        slack_unit_weight / slack_total_weight,
        0.0,
    )
    actual_unit_power_mw = unit_power_mw + slack_delta * slack_share

    # Net injection
    node_injection_mw = node_gen_mw - node_load_balanced

    # Line flows via PTDF: (n_lines, n_nodes) @ (n_nodes,) -> (n_lines,)
    line_flow_mw = case.PTDF @ node_injection_mw

    return line_flow_mw, node_injection_mw, actual_unit_power_mw


# ============ Safety Check ============

@jax.jit
def safety_check(
    line_flow_mw: chex.Array,
    line_cap: chex.Array,
    line_floor: chex.Array,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Check line flow safety and compute thermal overload cost.

    Uses active-power (MW) limit semantics for DC power flow. cost_thermal is in MW,
    not currency; when folding into reward, apply a penalty weight in the env layer
    (units: $/MW).

    Args:
        line_flow_mw: Line flows (n_lines,) [MW].
        line_cap: Upper capacity limits (n_lines,) [MW].
        line_floor: Lower capacity limits (n_lines,) [MW].

    Returns:
        is_safe: True if all lines within limits (bool scalar).
        n_violations: Number of violated lines (int32 scalar).
        cost_thermal: Total active-power violation [MW] (sum of overrun above cap and
            underrun below floor); DC active-power limits, not AC MVA thermal limits.

    Note:
        For AC power flow with MVA or current thermal limits, do not reuse this
        function; implement a separate check to avoid semantic confusion.
    """
    over = jnp.maximum(line_flow_mw - line_cap, 0.0)
    under = jnp.maximum(line_floor - line_flow_mw, 0.0)
    violation = over + under
    n_violations = jnp.sum(violation > 0.0).astype(jnp.int32)
    is_safe = n_violations == 0
    cost_thermal = jnp.sum(violation)
    return is_safe, n_violations, cost_thermal


def ac_thermal_check(
    pf_from: chex.Array,
    qf_from: chex.Array,
    line_cap: chex.Array,
    pf_to: chex.Array,
    qf_to: chex.Array,
    use_both_ends: bool,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """AC thermal safety check on apparent power (MVA).

    ``line_cap`` is the MVA thermal rating (see ``CaseData.line_cap`` convention).

    ``thermal_flow`` is ``max(|Sf|, |St|)`` when ``use_both_ends`` is True (ACPF),
    or ``|Sf|`` when False (ACOPF from-end flows only).

    ``cost_thermal`` is ``sum(max(0, thermal_flow - line_cap))`` [MVA].

    Args:
        pf_from: From-end active branch flow [MW].
        qf_from: From-end reactive branch flow [MVAr].
        line_cap: Thermal limit [MVA] per line.
        pf_to: To-end active flow [MW] (ignored when ``use_both_ends`` is False).
        qf_to: To-end reactive flow [MVAr] (ignored when ``use_both_ends`` is False).
        use_both_ends: If True, use ``max(sqrt(Pf²+Qf²), sqrt(Pt²+Qt²))``; if False,
            only ``sqrt(Pf²+Qf²)``.

    Returns:
        is_safe: True if no line exceeds its thermal cap.
        n_violations: Count of violated lines.
        cost_thermal: Sum of positive overloads [MVA].
    """
    sf = jnp.sqrt(pf_from ** 2 + qf_from ** 2)
    st = jnp.sqrt(pf_to ** 2 + qf_to ** 2)
    thermal_flow = jnp.where(use_both_ends, jnp.maximum(sf, st), sf)
    over = jnp.maximum(thermal_flow - line_cap, 0.0)
    n_violations = jnp.sum(over > 0.0).astype(jnp.int32)
    cost_thermal = jnp.sum(over)
    is_safe = n_violations == 0
    return is_safe, n_violations, cost_thermal


# ============ Combined Flow + Check ============

# Note: Not decorated with @jax.jit because CaseData is a pytree of JAX arrays.
# Callers may compile explicitly: jit_fn = jax.jit(dc_power_flow_with_check)
def dc_power_flow_with_check(
    case: CaseData,
    unit_power_mw: chex.Array,
    node_load_mw: chex.Array,
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
    """DC power flow followed by safety check.

    Args:
        case: CaseData.
        unit_power_mw: Generator outputs (n_units,) [MW].
        node_load_mw: Nodal loads (n_nodes,) [MW].

    Returns:
        line_flow_mw: Line active power flows (n_lines,) [MW].
        node_injection_mw: Net nodal injection (n_nodes,) [MW].
        actual_unit_power_mw: Generator outputs with slack adjustment (n_units,) [MW].
            Use this (not the raw ``unit_power_mw``) for cost computation and
            constraint checking so that RL agents cannot exploit "free" slack energy.
        is_safe: True if all lines within limits (bool scalar).
        n_violations: Number of violated lines (int32 scalar).
        cost_thermal: Active-power overload [MW] (DC MW-limit semantics); see
            :func:`safety_check` for unit and AC-reuse caveats.
    """
    line_flow_mw, node_injection_mw, actual_unit_power_mw = dc_power_flow(
        case, unit_power_mw, node_load_mw)
    is_safe, n_violations, cost_thermal = safety_check(
        line_flow_mw, case.line_cap, case.line_floor
    )
    return line_flow_mw, node_injection_mw, actual_unit_power_mw, is_safe, n_violations, cost_thermal


# ============ Generation Cost ============

@jax.jit
def compute_generation_cost(
    unit_power_mw: chex.Array,
    cost_a: chex.Array,
    cost_b: chex.Array,
    cost_c: chex.Array,
) -> chex.Array:
    """Compute total generation cost by integrating the marginal cost curve.

    Marginal cost: MC(p) = cost_a * p^2 + cost_b * p + cost_c
    Total cost:    TC(p) = (cost_a/3) * p^3 + (cost_b/2) * p^2 + cost_c * p

    Args:
        unit_power_mw: Generator outputs (n_units,) [MW].
        cost_a: Quadratic coefficient in MC(p), i.e. coefficient of p^2 (n_units,).
            These are marginal-cost (MC) polynomial coefficients, not MATPOWER-style
            total-cost (TC) coefficients. MATPOWER uses TC = a·p² + b·p + c and
            MC = 2a·p + b (linear); here MC(p) = cost_a·p² + cost_b·p + cost_c
            (quadratic) and TC = (a/3)p³ + (b/2)p² + c·p. Convert when importing
            from MATPOWER.
        cost_b: Linear coefficient of MC(p) (n_units,).
        cost_c: Constant term of MC(p) [$/MWh], i.e. marginal cost at zero output (n_units,).

    Returns:
        total_cost: Scalar generation cost [$/h] (assuming a 1-hour dispatch
            interval; cost_c [$/MWh] × p [MW] = [$/h]).
    """
    p = unit_power_mw
    per_unit = (cost_a / 3.0) * p ** 3 + (cost_b / 2.0) * p ** 2 + cost_c * p
    return jnp.sum(per_unit)


# ============ Proportional Dispatch ============

@jax.jit
def proportional_dispatch(
    total_load_mw: chex.Array,
    p_min: chex.Array,
    p_max: chex.Array,
) -> chex.Array:
    """Dispatch units proportionally to their capacity.

    Each unit first produces p_min, then the remaining demand is
    allocated proportionally to each unit's headroom (p_max - p_min).

    If ``total_load_mw`` exceeds the total generation capacity
    (``sum(p_max)``), all units are clipped to ``p_max`` — no error
    is raised, as this must stay JIT-safe.

    Args:
        total_load_mw: Total system load [MW] (scalar).
        p_min: Minimum output per unit (n_units,) [MW].
        p_max: Maximum output per unit (n_units,) [MW].

    Returns:
        unit_power_mw: Dispatched power (n_units,) [MW], clipped to [p_min, p_max].
    """
    base = jnp.sum(p_min)
    headroom = p_max - p_min
    total_headroom = jnp.sum(headroom)
    remaining = jnp.clip(total_load_mw - base, 0.0, total_headroom)
    # Proportional allocation of remaining load to headroom
    ratio = jnp.where(total_headroom > 0.0, remaining / total_headroom, 0.0)
    unit_power_mw = p_min + ratio * headroom
    return jnp.clip(unit_power_mw, p_min, p_max)
