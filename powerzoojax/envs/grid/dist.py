"""Distribution Grid Environment — radial BFS power flow.

A radial-distribution env where the agent controls DER active-power
injections at non-slack buses and / or attached ``ResourceBundle``
instances; the env runs the BFS power flow each step. Reward is the
(negative, weighted) active line loss; constraint violations on
voltage and thermal limits flow through a 3-channel CMDP cost vector.

Power flow: ``bfs_power_flow`` (DistFlow on ``v_sq`` for radial trees,
typically converging in 1–2 sweeps). All loads, injections, and flows
are stored in **per-unit** (p.u.) relative to ``base_mva``; the env
converts to MW only at the cost / reward layer.

Action (all entries in [−1, 1]):
    Concatenated layout
        ``[der_injection (n_nodes − 1) | bundle_0 | bundle_1 | ...]``
    where ``der_injection`` is denormalised to ``[der_low, der_high]``
    in **p.u.** — with ``base_mva = 100``, ``der_low = −0.005`` ⇒
    −500 kW, ``der_high = 0.005`` ⇒ +500 kW. Set physical MW values
    via the right p.u. ratio, **not** by passing MW directly. Setting
    ``include_der=False`` drops the DER block entirely (agent only
    controls bundles).

Observation:
    [v_norm, p_flow_norm, q_flow_norm, p_load_norm, q_load_norm,
     sin t, cos t, <bundle_obs concatenated>]
    where ``v_norm = (v − 1) / 0.1`` and flows / loads are normalised
    by the episode-summed reference ``p_load_ref`` / ``q_load_ref``.

Reward:
    reward = −loss_penalty_weight · p_loss_MW
    Encourages the agent to cancel reverse flows and minimise
    resistive losses.

CMDP costs (stacked vector, three channels):
    cost_voltage_violation  count of nodes with v ∉ [v_min, v_max].
    cost_thermal_overload   count of lines with apparent power
                            |S| = √(P² + Q²) > line_cap (same MVA
                            convention as AC paths).
    cost_resource           Σ over bundles of each bundle's
                            ``info["cost_sum"]``.

Diagnostics in info:
    ``soc_terminal_sq`` : squared deviation between the first bundle's
                          SOC at episode end and at episode start,
                          charged only on the terminal step. Use in a
                          reward-shaping wrapper for storage tasks.
    ``resource_*_mw``   : pre-auto-reset FlexLoad statistics
                          (curtailed / shift_out / shift_in) so
                          rollout loggers don't read zeroed state
                          after auto-reset on the final step.

Resource attachment:
    ``make_dist_params(resources=(make_battery_bundle(case,
    bus_ids=[18, 25]), ...))``. Bundle states live in
    ``DistGridState.resource_states`` and auto-reset with the grid;
    bundle injections are converted to p.u. and subtracted from node
    load before BFS so the solver sees the residual demand.
"""

from functools import partial
from typing import Tuple, Dict, Any

import numpy as np
import jax
import jax.numpy as jnp
import jax.tree_util as tu
import chex
from flax import struct

from powerzoojax.envs.base import (
    Environment,
    denormalize_action,
    stack_costs,
    time_features,
)
from powerzoojax.envs.spaces import Box
from powerzoojax.case.case_data import CaseData
from powerzoojax.envs.grid.bfs_power_flow import (
    BFSTopoData,
    prepare_bfs,
    bfs_power_flow,
)


def _episode_batt_soc_init(resource_states: tuple) -> chex.Array:
    """SOC vector of the first resource bundle at episode start.

    This terminal-SOC diagnostic is intended for battery-like bundles.  When
    the first bundle does not expose ``soc``, return an empty vector so the
    terminal penalty path becomes a no-op instead of assuming battery state.
    """
    if len(resource_states) == 0:
        return jnp.zeros((0,), dtype=jnp.float32)
    first = resource_states[0]
    if hasattr(first, "soc"):
        return first.soc
    return jnp.zeros((0,), dtype=jnp.float32)


@struct.dataclass
class DistGridState:
    """DistGridEnv state — pure grid quantities only.

    Resource state (battery SOC, etc.) lives in ``resource_states``.
    ``episode_batt_soc_init`` stores SOC at episode start for terminal penalties.
    """
    time_step: chex.Array
    done: chex.Array
    v_mag: chex.Array         # (n_nodes,) voltage magnitudes [p.u.]
    p_branch: chex.Array      # (n_lines,) active branch flow [p.u.]
    q_branch: chex.Array      # (n_lines,) reactive branch flow [p.u.]
    p_loss_total: chex.Array  # scalar total active loss [p.u.]
    is_safe: chex.Array       # bool
    n_violations: chex.Array  # int32
    total_cost: chex.Array    # cumulative episode cost
    resource_states: tuple = ()  # tuple[BundleState, ...]; empty = no resources
    # First battery bundle SOC at episode start (for terminal SOC penalty in reward_fn)
    episode_batt_soc_init: chex.Array = struct.field(
        default_factory=lambda: jnp.zeros((0,), dtype=jnp.float32))


@struct.dataclass
class DistGridParams:
    """DistGridEnv parameters.

    Scalar configuration fields are marked ``pytree_node=False`` so they
    are treated as static compile-time constants by ``jax.jit``.

    **Action scaling note**: ``der_low`` and ``der_high`` are in **per-unit**
    (p.u.) relative to ``base_mva``.  With the default ``base_mva = 100``,
    ``der_low = -0.005`` corresponds to −500 kW and ``der_high = 0.005``
    to +500 kW.  Do **not** set them to physical MW values directly.
    """
    case: CaseData = None
    topo: BFSTopoData = None
    load_profiles_p: chex.Array = None  # (T, n_nodes) active load [p.u.]
    load_profiles_q: chex.Array = None  # (T, n_nodes) reactive load [p.u.]
    line_cap: chex.Array = None         # (n_active_lines,) thermal limit [MVA]
    p_load_ref: float = struct.field(pytree_node=False, default=1.0)
    q_load_ref: float = struct.field(pytree_node=False, default=1.0)
    base_mva: float = struct.field(pytree_node=False, default=100.0)
    v_slack: float = struct.field(pytree_node=False, default=1.0)
    v_min: float = struct.field(pytree_node=False, default=0.90)
    v_max: float = struct.field(pytree_node=False, default=1.10)
    max_steps: int = struct.field(pytree_node=False, default=48)
    steps_per_day: int = struct.field(pytree_node=False, default=48)
    loss_penalty_weight: float = struct.field(pytree_node=False, default=0.1)
    der_low: float = struct.field(pytree_node=False, default=-0.005)
    der_high: float = struct.field(pytree_node=False, default=0.005)
    bfs_max_iter: int = struct.field(pytree_node=False, default=100)
    bfs_tol: float = struct.field(pytree_node=False, default=1e-6)
    include_der: bool = struct.field(pytree_node=False, default=True)
    # Deprecated legacy field kept for backward-compatible config loading.
    # Core CMDP envs now always return the full vector cost layout and tasks /
    # wrappers decide which constraint subset to consume.
    cost_mode: str = struct.field(pytree_node=False, default="aggregate")
    resources: tuple = ()


def make_dist_params(
    case: CaseData,
    max_steps: int = 48,
    steps_per_day: int = 48,
    v_slack: float = 1.0,
    v_min: float = 0.90,
    v_max: float = 1.10,
    loss_penalty_weight: float = 0.1,
    resources: tuple = (),
    include_der: bool = True,
    cost_mode: str = "aggregate",
) -> DistGridParams:
    """Create DistGridParams from a CaseData.

    Args:
        case: CaseData instance.
        max_steps: Maximum steps per episode.
        steps_per_day: Steps per day for time encoding.
        v_slack: Slack bus voltage magnitude [p.u.].
        v_min: Minimum allowed voltage [p.u.].
        v_max: Maximum allowed voltage [p.u.].
        loss_penalty_weight: Weight on active-loss reward shaping.
        resources: Tuple of ResourceBundle instances (e.g. BatteryBundle).
            Use ``make_battery_bundle(case, bus_ids=[...])`` to create bundles.
        include_der: If False, action space contains only bundle actions
            (no legacy DER injection at each bus).  Use this when the agent
            should only control attached resources (e.g. batteries).
        cost_mode: Deprecated legacy field kept for backward-compatible config
            loading. Core env semantics no longer depend on it; task-level
            constraint selection decides which channels a benchmark uses.

    Returns:
        DistGridParams ready for DistGridEnv.
    """
    topo = prepare_bfs(case)

    base_mva = float(case.base_mva) if case.base_mva is not None else 100.0

    # Default load profiles: flat at node_pd, node_qd (in p.u.)
    p_load_pu = case.node_pd / base_mva
    q_load_pu = case.node_qd / base_mva
    load_p = jnp.tile(p_load_pu[None, :], (max_steps, 1))
    load_q = jnp.tile(q_load_pu[None, :], (max_steps, 1))

    # Extract active-line thermal limits (matching BFS topo line ordering)
    all_cap = np.asarray(case.line_cap)
    if case.line_status is not None:
        active = np.asarray(case.line_status) > 0
        active_cap = all_cap[active]
    else:
        active_cap = all_cap
    line_cap = jnp.asarray(active_cap, dtype=jnp.float32)

    p_load_ref = max(float(jnp.sum(p_load_pu)), 1e-8)
    q_load_ref = max(float(jnp.sum(q_load_pu)), 1e-8)

    return DistGridParams(
        case=case,
        topo=topo,
        load_profiles_p=load_p,
        load_profiles_q=load_q,
        line_cap=line_cap,
        p_load_ref=p_load_ref,
        q_load_ref=q_load_ref,
        base_mva=base_mva,
        v_slack=v_slack,
        v_min=v_min,
        v_max=v_max,
        max_steps=max_steps,
        steps_per_day=steps_per_day,
        loss_penalty_weight=loss_penalty_weight,
        include_der=include_der,
        cost_mode=cost_mode,
        resources=tuple(resources),
    )


class DistGridEnv(Environment):
    """Distribution grid environment — BFS power flow.

    Observation layout:
        node voltages (n_nodes,)         — (v - 1) / 0.1
        line active flows (n_lines,)     — p_branch / p_load_ref
        line reactive flows (n_lines,)   — q_branch / q_load_ref
        node active loads (n_nodes,)     — p_load / p_load_ref
        node reactive loads (n_nodes,)   — q_load / q_load_ref
        [time_sin, time_cos]  (2,)
        <bundle_obs>                     — appended per bundle

    Reward:
        -loss_penalty_weight * p_loss_MW  (operational cost only)

    Constraint cost channels:
        cost_voltage_violation: count of nodes violating voltage limits
        cost_thermal_overload: count of lines violating thermal limits (|S| > cap)
        cost_resource: bundle clipping costs
        cost_sum: total violation count
    """

    # ====== RL Interface Methods ======

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: DistGridParams,
    ) -> Tuple[chex.Array, DistGridState]:
        p_load = params.load_profiles_p[0]
        q_load = params.load_profiles_q[0]

        result = bfs_power_flow(
            params.topo, p_load, q_load,
            params.v_slack, params.bfs_max_iter, params.bfs_tol,
        )

        resource_states = tuple(b.reset(key) for b in params.resources)
        ep_soc0 = _episode_batt_soc_init(resource_states)

        state = DistGridState(
            time_step=jnp.int32(0),
            done=jnp.bool_(False),
            v_mag=result.v_mag,
            p_branch=result.p_branch,
            q_branch=result.q_branch,
            p_loss_total=jnp.sum(result.p_loss),
            is_safe=jnp.bool_(True),
            n_violations=jnp.int32(0),
            total_cost=jnp.float32(0.0),
            resource_states=resource_states,
            episode_batt_soc_init=ep_soc0,
        )
        obs = self._get_obs(state, params)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, key: chex.PRNGKey, state: DistGridState,
        action: chex.Array, params: DistGridParams,
    ) -> Tuple[chex.Array, DistGridState, chex.Array, chex.Array, chex.Array, Dict[str, Any]]:
        """Run one environment step.

        Args:
            key: PRNG key.
            state: Current grid state.
            action: When ``include_der=True``:
                ``(n_nodes-1 + sum(bundle.action_dim),)`` in ``[-1, 1]``.
                When ``include_der=False``:
                ``(sum(bundle.action_dim),)`` — bundle actions only.
            params: Environment parameters.
        """
        t_idx = state.time_step % params.load_profiles_p.shape[0]
        p_load = params.load_profiles_p[t_idx]
        q_load = params.load_profiles_q[t_idx]
        n_nodes = p_load.shape[0]

        if params.include_der:
            n_der = n_nodes - 1  # static
            der_action = action[:n_der]
            der_low = jnp.full((n_der,), params.der_low, dtype=jnp.float32)
            der_high = jnp.full((n_der,), params.der_high, dtype=jnp.float32)
            physical_action = denormalize_action(der_action, der_low, der_high)
            der_injection = jnp.zeros(n_nodes).at[1:1 + n_der].set(physical_action)
            offset = n_der
        else:
            der_injection = jnp.zeros(n_nodes, dtype=jnp.float32)
            offset = 0

        # --- Bundle processing (trace-time unrolled) ---
        ctx = {}
        new_resource_states_list = []
        bundle_injection_mw = jnp.zeros(n_nodes, dtype=jnp.float32)
        bundle_q_injection_mvar = jnp.zeros(n_nodes, dtype=jnp.float32)
        bundle_cost = jnp.float32(0.0)
        # Resource stats before auto-reset (scalars, for correct last-step rollout accounting)
        bundle_curtailed_mw = jnp.float32(0.0)
        bundle_shift_out_mw = jnp.float32(0.0)
        bundle_shift_in_mw = jnp.float32(0.0)
        for i, bundle in enumerate(params.resources):
            a_slice = action[offset: offset + bundle.action_dim]
            offset += bundle.action_dim
            new_bs, p_inj, q_inj, _, cost_info = bundle.step(
                state.resource_states[i], a_slice, ctx)
            new_resource_states_list.append(new_bs)
            bundle_injection_mw = bundle_injection_mw.at[bundle.bus_idx].add(p_inj)
            bundle_q_injection_mvar = bundle_q_injection_mvar.at[bundle.bus_idx].add(q_inj)
            # Only accumulate the scalar total cost, not per-device vectors
            bundle_cost = bundle_cost + cost_info.get(
                "cost_sum",
                cost_info.get("cost", jnp.float32(0.0)),
            )
            # Accumulate FlexLoad-style resource stats (Python-level hasattr, trace-time static)
            if hasattr(new_bs, "curtailed_mw"):
                bundle_curtailed_mw = bundle_curtailed_mw + jnp.sum(new_bs.curtailed_mw)
                bundle_shift_out_mw = bundle_shift_out_mw + jnp.sum(new_bs.shift_out_mw)
                bundle_shift_in_mw = bundle_shift_in_mw + jnp.sum(new_bs.shift_in_mw)
        new_resource_states = tuple(new_resource_states_list)
        bundle_injection_pu = bundle_injection_mw / params.base_mva
        bundle_q_injection_pu = bundle_q_injection_mvar / params.base_mva

        p_net = p_load - der_injection - bundle_injection_pu
        q_net = q_load - bundle_q_injection_pu

        result = bfs_power_flow(
            params.topo, p_net, q_net,
            params.v_slack, params.bfs_max_iter, params.bfs_tol,
        )

        # Voltage violations (count-based, consistent with PowerZoo)
        v_violation = jnp.logical_or(
            result.v_mag < params.v_min,
            result.v_mag > params.v_max,
        )
        n_v_violations = jnp.sum(v_violation).astype(jnp.int32)

        # Thermal limit: apparent power |S| = sqrt(P² + Q²) > cap
        p_flow_mw = result.p_branch * params.base_mva
        q_flow_mvar = result.q_branch * params.base_mva
        s_flow = jnp.sqrt(p_flow_mw ** 2 + q_flow_mvar ** 2)
        line_cap = jnp.where(params.line_cap > 0, params.line_cap, jnp.float32(1e6))
        n_l_violations = jnp.sum(s_flow > line_cap).astype(jnp.int32)

        n_violations = n_v_violations + n_l_violations
        is_safe = n_violations == 0

        p_loss_total_pu = jnp.sum(result.p_loss)
        p_loss_mw = p_loss_total_pu * params.base_mva

        reward = -params.loss_penalty_weight * p_loss_mw

        new_time = state.time_step + 1
        done = new_time >= params.max_steps

        new_state = DistGridState(
            time_step=new_time,
            done=done,
            v_mag=result.v_mag,
            p_branch=result.p_branch,
            q_branch=result.q_branch,
            p_loss_total=p_loss_total_pu,
            is_safe=is_safe,
            n_violations=n_violations,
            total_cost=state.total_cost + n_violations.astype(jnp.float32),
            resource_states=new_resource_states,
            episode_batt_soc_init=state.episode_batt_soc_init,
        )

        # Auto-reset
        _, k_reset = jax.random.split(key)
        _, reset_state = self.reset(k_reset, params)
        final_state = tu.tree_map(
            lambda n, r: jnp.where(done, r, n), new_state, reset_state)
        final_state = jax.lax.stop_gradient(final_state)
        final_obs = self._get_obs(final_state, params)

        cost_voltage_violation = n_v_violations.astype(jnp.float32)
        cost_thermal_overload = n_l_violations.astype(jnp.float32)
        cost_resource = bundle_cost
        cost_sum = cost_voltage_violation + cost_thermal_overload + cost_resource

        v_over = jnp.sum(jnp.maximum(result.v_mag - params.v_max, 0.0))
        v_under = jnp.sum(jnp.maximum(params.v_min - result.v_mag, 0.0))
        s_over = jnp.sum(jnp.maximum(s_flow - line_cap, 0.0))
        cost_continuous = v_over + v_under + s_over

        soc_final = _episode_batt_soc_init(new_resource_states)
        soc_pen_sq = jnp.sum((soc_final - state.episode_batt_soc_init) ** 2)
        # Only charge terminal penalty on the last transition (episode boundary)
        soc_terminal_sq = jnp.where(done, soc_pen_sq, jnp.float32(0.0))

        costs = stack_costs(
            cost_voltage_violation,
            cost_thermal_overload,
            cost_resource,
        )
        info = {
            "cost_voltage_violation": cost_voltage_violation,
            "cost_thermal_overload": cost_thermal_overload,
            "cost_resource": cost_resource,
            "cost_continuous": cost_continuous,
            "cost_sum": cost_sum,
            "is_safe": is_safe,
            "goal_met": is_safe,
            "n_violations": n_violations,
            "v_min_step": jnp.min(result.v_mag),
            "v_max_step": jnp.max(result.v_mag),
            "p_loss_MW": p_loss_mw,
            "q_loss_MVAr": jnp.sum(result.q_loss) * params.base_mva,
            "bfs_converged": result.converged,
            # Terminal SOC deviation (episode end vs episode start); use in reward_fn when done
            "soc_terminal_sq": soc_terminal_sq,
            # Resource action stats from this step (pre-auto-reset); use in rollout to avoid
            # reading zeroed state after auto-reset on the final step
            "resource_curtailed_mw": bundle_curtailed_mw,
            "resource_shift_out_mw": bundle_shift_out_mw,
            "resource_shift_in_mw": bundle_shift_in_mw,
        }

        return final_obs, final_state, reward, costs, done, info

    # ====== Spaces & Observation ======

    def _get_obs(self, state: DistGridState, params: DistGridParams) -> chex.Array:
        """Build observation vector — grid quantities + bundle observations.

        Layout:
            node voltages (n_nodes,)         — (v - 1.0) / 0.1
            line active flows (n_lines,)     — p_branch / p_load_ref
            line reactive flows (n_lines,)   — q_branch / q_load_ref
            node active loads (n_nodes,)     — p_load / p_load_ref
            node reactive loads (n_nodes,)   — q_load / q_load_ref
            [time_sin, time_cos]  (2,)
            <bundle_obs>
        """
        t_idx = state.time_step % params.load_profiles_p.shape[0]
        p_load = params.load_profiles_p[t_idx]
        q_load = params.load_profiles_q[t_idx]

        v_norm = (state.v_mag - 1.0) / 0.1
        p_flow_norm = state.p_branch / params.p_load_ref
        q_flow_norm = state.q_branch / params.q_load_ref
        p_load_norm = p_load / params.p_load_ref
        q_load_norm = q_load / params.q_load_ref

        t_sin, t_cos = time_features(state.time_step, params.steps_per_day)

        obs_parts = [v_norm, p_flow_norm, q_flow_norm, p_load_norm, q_load_norm,
                     jnp.stack([t_sin, t_cos])]

        ctx = {}
        for i, bundle in enumerate(params.resources):
            obs_parts.append(bundle.observe(state.resource_states[i], ctx))

        return jnp.concatenate(obs_parts)

    def observation_space(self, params: DistGridParams) -> Box:
        n = params.case.n_nodes
        nl = params.topo.n_lines
        obs_dim = 3 * n + 2 * nl + 2
        obs_dim += sum(b.obs_dim for b in params.resources)
        return Box(
            low=jnp.full((obs_dim,), -jnp.inf),
            high=jnp.full((obs_dim,), jnp.inf),
            shape=(obs_dim,),
            dtype=jnp.float32,
        )

    def action_space(self, params: DistGridParams) -> Box:
        act_dim = sum(b.action_dim for b in params.resources)
        if params.include_der:
            act_dim += params.case.n_nodes - 1
        return Box(
            low=jnp.full((act_dim,), -1.0, dtype=jnp.float32),
            high=jnp.full((act_dim,), 1.0, dtype=jnp.float32),
            shape=(act_dim,),
            dtype=jnp.float32,
        )

    # ====== Status & Diagnostics ======

    @property
    def name(self) -> str:
        return "DistGridEnv"

    def constraint_names(self, params: DistGridParams) -> tuple[str, ...]:
        return (
            "voltage_violation",
            "thermal_overload",
            "resource",
        )
