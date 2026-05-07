"""Market layer base classes for PowerZooJax.

Foundation for the **market layer** — ISO-style economic dispatch
environments that clear a wholesale electricity market on a linearised
DC transmission grid. Subclasses (``CostBasedMarketEnv``,
``BidBasedMarketEnv``, ``LMPMarketEnv``, …) plug in concrete clearing
mechanisms on top of this shared data model.

Grid model (linearised DC OPF):
    line_flow = PTDF @ node_injection
    node_injection = nodes_units_map @ unit_power
                   − nodes_loads_map @ load_profile_t
    ``line_cap`` / ``line_floor`` are upper / lower thermal limits per
    line; a value of 0 is treated as ±1e5 (sentinel for "unconstrained").
    ``slack_bus_idx`` is the reference bus for the DC PF model.

Generator cost model (per unit, units $/MWh):
    MC(P) = a · P² + b · P + c       marginal cost (default a = 0
                                     recovers the standard linear-MC
                                     / quadratic-C(P) form)
    C(P)  = a/3·P³ + b/2·P² + c·P    production cost (∫ MC dP)

Resources:
    ``MarketParams.resources`` is a tuple of ``ResourceBundle`` instances
    (e.g. ``BatteryBundle``) attached at nodes via ``bus_idx``. Battery
    / DER physics — SOC update, PQ projection — live inside the bundles,
    not as flat fields on the market state.
    ``MarketState.resource_states`` carries one bundle state per attached
    bundle (in declaration order).

Pricing convention:
    ``MarketState.lmp`` is the last nodal locational marginal price in
    $/MWh, shape ``(n_nodes,)``. Multiply by P [MW] · dt [h] to get
    revenue [$] for an injection at a node. ``params.lmp_scale`` is an
    observation-normalisation reference; it does not change the unit
    of ``lmp`` itself.

Time convention:
    ``params.steps_per_day`` defines the cyclic profile index;
    ``load_profiles[t % steps_per_day]`` is the load vector at step t.

Design note (struct inheritance):
    ``MarketState`` extends ``EnvState`` (shares ``time_step`` / ``done``);
    ``MarketParams`` extends ``EnvParams`` (shares ``max_steps`` /
    ``delta_t_hours``). Subclasses add domain-specific fields after the
    base ones; non-default fields must come before any default-valued
    ones in the dataclass declaration — that's why
    ``MarketState.resource_states`` has no default.
"""

from __future__ import annotations

import chex
from flax import struct

from powerzoojax.envs.base import EnvState, EnvParams


@struct.dataclass
class MarketState(EnvState):
    """Market environment state.

    Extends EnvState with last LMP, dispatch fields, and attached resource
    bundle states (SOC, etc.) in ``resource_states``.
    """
    lmp: chex.Array  # (n_nodes,) $/MWh — last nodal LMP; multiply by P_MW × Δt_h for revenue [$]
    unit_power_mw: chex.Array     # (n_units,) — generator dispatch
    line_flow_mw: chex.Array      # (n_lines,) — line flows
    # No default here: subclasses add non-default fields after this; use () when empty.
    resource_states: tuple        # tuple[BundleState, ...]; e.g. BatteryBundleState


@struct.dataclass
class MarketParams(EnvParams):
    """Market environment parameters.

    Holds grid and market configuration.  Battery / DER physics live in
    ``resources`` (e.g. ``BatteryBundle``), not as flat fields on this struct.
    Array fields (PTDF, costs, etc.) are JAX leaves; scalar config
    fields are ``pytree_node=False`` (static under JIT).
    """
    # Grid topology
    PTDF: chex.Array = None               # (n_lines, n_nodes)
    nodes_units_map: chex.Array = None     # (n_nodes, n_units)
    line_cap: chex.Array = None    # (n_lines,) MW thermal upper limit; 0 → replaced by 1e5 (unconstrained)
    line_floor: chex.Array = None  # (n_lines,) MW thermal lower limit; 0 → replaced by −1e5 (unconstrained)
    load_profiles: chex.Array = None       # (T, n_loads) MW
    nodes_loads_map: chex.Array = None     # (n_nodes, n_loads)

    # Generator costs.  Marginal cost MC(P) = a·P² + b·P + c [$/MWh] with P in MW.
    unit_cost_a: chex.Array = None   # (n_units,) coeff a — units $/(MWh·MW²)
    unit_cost_b: chex.Array = None   # (n_units,) coeff b — units $/(MWh·MW)
    unit_cost_c: chex.Array = None   # (n_units,) coeff c — units $/MWh
    # Total production cost (same energy units as case): ∫ MC dP = a/3·P³ + b/2·P² + c·P
    unit_p_min: chex.Array = None    # (n_units,)
    unit_p_max: chex.Array = None    # (n_units,)

    resources: tuple = ()            # ResourceBundle instances (e.g. BatteryBundle)

    # Market config
    lmp_scale: float = struct.field(pytree_node=False, default=100.0)  # obs normalisation only; same unit as lmp ($/MWh)
    steps_per_day: int = struct.field(pytree_node=False, default=48)

    # Dimensions (static)
    n_nodes: int = struct.field(pytree_node=False, default=0)
    n_units: int = struct.field(pytree_node=False, default=0)
    n_lines: int = struct.field(pytree_node=False, default=0)
    n_loads: int = struct.field(pytree_node=False, default=0)
    slack_bus_idx: int = struct.field(pytree_node=False, default=0)
