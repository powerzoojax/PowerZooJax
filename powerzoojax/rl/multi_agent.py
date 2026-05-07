"""Multi-Agent RL Environment — JaxMARL Compatible.

Two adapters:
- GridMARLEnv:     wraps TransGridEnv (transmission grid, units + resources)
- DistGridMARLEnv: wraps DistGridEnv (distribution grid, resources only)

Design:
    - Thin adapter layer: delegates all physics to the underlying env.
    - JaxMARL protocol: obs/actions/rewards/dones are Dict[str, Array]
    - dones["__all__"] indicates episode termination
    - Resources read from grid_params.resources (ResourceBundle API)
    - JIT/vmap compatible

Agent naming:
    - GridMARLEnv:     "unit_0", "unit_1", ..., "battery_0", "battery_1", ...
    - DistGridMARLEnv: "battery_0", "battery_1", ... (no unit agents)

Note: This module lives in rl/ (training adapters), not envs/ (pure physics).
It composes existing envs without modifying them.
"""

import collections
from functools import partial
from typing import Dict, Tuple, Any, List
from abc import ABC, abstractmethod

import numpy as np
import jax
import jax.numpy as jnp
import chex
from flax import struct

from powerzoojax.envs.spaces import Box
from powerzoojax.envs.grid.trans import TransGridEnv, TransGridState, TransGridParams
from powerzoojax.envs.grid.dist import DistGridEnv, DistGridState, DistGridParams
from powerzoojax.envs.grid.dist_3phase import DistGrid3PhaseEnv, DistGrid3PhParams


# ============ Local obs K-hop helper (CPU, runs at construction time) ============

_LOCAL_K = 4  # number of K-hop neighbours per agent (excluding own bus)


def _compute_local_neighbor_idx(
    sending_node_idx,
    receiving_node_idx,
    n_nodes: int,
    agent_bus_indices: List[int],
    K: int = _LOCAL_K,
) -> List[np.ndarray]:
    """BFS from each agent's bus to find K nearest neighbour bus indices.

    Runs at Python time (construction only, never inside JIT).

    Returns:
        List of np.ndarray(K,) int32, one per agent.
        Own bus is included as first element if fewer than K neighbours exist
        (padding).  Own bus is NOT included when K neighbours are found.
    """
    # Build undirected adjacency list from BFS topology
    adj: Dict[int, List[int]] = {i: [] for i in range(n_nodes)}
    for s, r in zip(np.asarray(sending_node_idx), np.asarray(receiving_node_idx)):
        s, r = int(s), int(r)
        if r not in adj[s]:
            adj[s].append(r)
        if s not in adj[r]:
            adj[r].append(s)

    result: List[np.ndarray] = []
    for start_bus in agent_bus_indices:
        visited: List[int] = []
        visited_set = {start_bus}
        queue: collections.deque = collections.deque([start_bus])
        while queue and len(visited) < K:
            node = queue.popleft()
            for nb in adj[node]:
                if nb not in visited_set:
                    visited_set.add(nb)
                    queue.append(nb)
                    visited.append(nb)
                    if len(visited) >= K:
                        break
        # Pad with own bus if network is smaller than K
        while len(visited) < K:
            visited.append(start_bus)
        result.append(np.array(visited[:K], dtype=np.int32))
    return result


# ============ Multi-Agent Base ============

class MultiAgentEnvironment(ABC):
    """JaxMARL-compatible multi-agent environment base class.

    Key differences from single-agent Environment:
    - obs, rewards, dones are Dict[str, Array]
    - actions are Dict[str, Array]
    - dones must include "__all__" key
    - params are bound at construction time
    """

    @property
    @abstractmethod
    def num_agents(self) -> int:
        ...

    @property
    @abstractmethod
    def agent_names(self) -> List[str]:
        ...

    @abstractmethod
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], Any]:
        ...

    @abstractmethod
    def step(
        self,
        key: chex.PRNGKey,
        state: Any,
        actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], Any, Dict[str, chex.Array], Dict[str, chex.Array], Dict[str, Any]]:
        """Returns (obs_dict, state, rewards_dict, dones_dict, info)."""
        ...

    @abstractmethod
    def observation_space(self, agent: str) -> Box:
        ...

    @abstractmethod
    def action_space(self, agent: str) -> Box:
        ...


# ============ MARL State ============

@struct.dataclass
class MARLState:
    """Multi-agent environment state — thin wrapper around TransGridState.

    All physical quantities (unit_power_mw, resource SOC, time_step, done, ...)
    are accessible via ``state.grid_state``.
    """
    grid_state: TransGridState


# ============ GridMARLEnv ============

class GridMARLEnv(MultiAgentEnvironment):
    """Multi-agent power grid environment.

    Each generator unit and each resource device is an independent agent.
    Resources are read from ``grid_params.resources`` (ResourceBundle API).

    Args:
        grid_env: TransGridEnv instance.
        grid_params: TransGridParams with case data, load profiles, and
            optional resource bundles attached via ``resources=(bundle,)``.
        reward_mode: 'shared' (all agents get same reward).

    Agent layout (action/obs):
        - Unit agents: action shape (1,) in [-1,1] normalised dispatch.
        - Resource agents: action shape (1,) per device in [-1,1].

    Obs layout per agent:
        - Shared obs: grid core obs (line flows, loads, unit dispatch, time)
        - Unit agent i: shared + unit_norm[i] (own dispatch fraction)
        - Resource device j: shared + device_obs_slice (e.g. [soc, p_norm])

    Usage::

        bundle = make_battery_bundle(case, bus_ids=[1, 3], power_mw=20)
        params = make_trans_params(case, resources=(bundle,), max_steps=48)
        env = GridMARLEnv(TransGridEnv(), params)
        obs_dict, state = env.reset(key)
        obs_dict, state, rewards, dones, info = env.step(key, state, actions_dict)
    """

    def __init__(
        self,
        grid_env: TransGridEnv,
        grid_params: TransGridParams,
        reward_mode: str = "shared",
    ):
        self._grid_env = grid_env
        self._grid_params = grid_params
        self._reward_mode = reward_mode

        case = grid_params.case
        self._n_units = case.n_units

        # Unit agent names
        self._unit_names = [f"unit_{i}" for i in range(self._n_units)]

        # Resource agent names: one agent per device, globally indexed per type
        type_device_count: Dict[str, int] = {}
        self._bundle_agent_names_list: List[List[str]] = []

        for bundle in grid_params.resources:
            bundle_type = type(bundle).__name__.replace("Bundle", "").lower()
            count = type_device_count.get(bundle_type, 0)
            agent_names = [f"{bundle_type}_{count + j}" for j in range(bundle.n_devices)]
            type_device_count[bundle_type] = count + bundle.n_devices
            self._bundle_agent_names_list.append(agent_names)

        self._resource_names = [
            name for names in self._bundle_agent_names_list for name in names
        ]
        self._all_names = self._unit_names + self._resource_names

        # Precompute obs structure (static, based on grid_params)
        total_obs_dim = grid_env.observation_space(grid_params).shape[0]
        bundle_total_obs_dim = sum(b.obs_dim for b in grid_params.resources)
        self._grid_core_dim = total_obs_dim - bundle_total_obs_dim

        # Offset of unit_norm within obs_flat (depends on AC vs DC)
        if grid_params.acpf_setup is not None or grid_params.acopf_setup is not None:
            # AC layout: [flow_norm(L), vm(N), load_norm(n_loads), unit_norm(U), sin, cos]
            self._unit_norm_offset = case.n_lines + case.n_nodes + case.n_loads
        else:
            # DC layout: [flow_norm(L), load_norm(n_loads), unit_norm(U), sin, cos]
            self._unit_norm_offset = case.n_lines + case.n_loads

        # Per-bundle obs start offset and per-device obs dim
        self._bundle_obs_offsets: List[int] = []
        self._bundle_per_device_obs_dims: List[int] = []
        offset = self._grid_core_dim
        for bundle in grid_params.resources:
            self._bundle_obs_offsets.append(offset)
            self._bundle_per_device_obs_dims.append(bundle.per_device_obs_dim)
            offset += bundle.obs_dim

        # Uniform obs dim across all agents (required for JaxMARL vmap)
        unit_obs_dim = self._grid_core_dim + 1
        max_device_obs_dim = max(
            (b.per_device_obs_dim for b in grid_params.resources), default=0
        )
        resource_obs_dim = (self._grid_core_dim + max_device_obs_dim
                            if max_device_obs_dim > 0 else 0)
        self._obs_dim = max(unit_obs_dim, resource_obs_dim) if resource_obs_dim > 0 else unit_obs_dim

    # ---- Properties ----

    @property
    def num_agents(self) -> int:
        return len(self._all_names)

    @property
    def agent_names(self) -> List[str]:
        return self._all_names

    def observation_space(self, agent: str = None) -> Box:
        return Box(
            low=jnp.full((self._obs_dim,), -jnp.inf, dtype=jnp.float32),
            high=jnp.full((self._obs_dim,), jnp.inf, dtype=jnp.float32),
            shape=(self._obs_dim,),
            dtype=jnp.float32,
        )

    def action_space(self, agent: str = None) -> Box:
        return Box(
            low=jnp.full((1,), -1.0, dtype=jnp.float32),
            high=jnp.full((1,), 1.0, dtype=jnp.float32),
            shape=(1,),
            dtype=jnp.float32,
        )

    @property
    def name(self) -> str:
        return "GridMARLEnv"

    # ---- Core Methods ----

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], MARLState]:
        obs_flat, grid_state = self._grid_env.reset(key, self._grid_params)
        state = MARLState(grid_state=grid_state)
        obs_dict = self._build_obs_dict(grid_state, obs_flat)
        return obs_dict, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: MARLState,
        actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], MARLState, Dict[str, chex.Array], Dict[str, chex.Array], Dict[str, Any]]:
        flat_action = self._pack_actions(actions)

        obs_flat, new_grid_state, reward, costs, done, info = self._grid_env.step(
            key, state.grid_state, flat_action, self._grid_params
        )
        info = {
            **info,
            "constraint_costs": costs,
            "cost": info.get("cost_sum", jnp.sum(costs)),
        }

        new_state = MARLState(grid_state=new_grid_state)
        obs_dict = self._build_obs_dict(new_grid_state, obs_flat)

        rewards = {name: reward for name in self._all_names}
        dones = {name: done for name in self._all_names}
        dones["__all__"] = done

        return obs_dict, new_state, rewards, dones, info

    # ---- Internal Helpers ----

    def _pack_actions(self, actions: Dict[str, chex.Array]) -> chex.Array:
        """Pack per-agent action dict into flat action array for TransGridEnv."""
        unit_acts = jnp.stack([actions[name].squeeze() for name in self._unit_names])

        bundle_parts = [
            jnp.concatenate([actions[name].reshape(-1) for name in bundle_names])
            for bundle_names in self._bundle_agent_names_list
        ]

        if bundle_parts:
            return jnp.concatenate([unit_acts] + bundle_parts)
        return unit_acts

    def _build_obs_dict(
        self, grid_state: TransGridState, obs_flat: chex.Array
    ) -> Dict[str, chex.Array]:
        """Unpack flat TransGridEnv obs into per-agent observation dict.

        Obs layout (DC mode):
            [flow_norm(L) | load_norm(D) | unit_norm(U) | sin | cos | bundle_obs...]
        Each agent gets: shared_obs (grid core) + local_feature, padded to obs_dim.
        """
        shared = obs_flat[:self._grid_core_dim]

        obs_dict: Dict[str, chex.Array] = {}

        # Unit agents: shared + own unit_norm[i]
        for i, name in enumerate(self._unit_names):
            own = obs_flat[self._unit_norm_offset + i: self._unit_norm_offset + i + 1]
            local = jnp.concatenate([shared, own])
            pad = self._obs_dim - local.shape[0]
            if pad > 0:
                local = jnp.concatenate([local, jnp.zeros(pad, dtype=jnp.float32)])
            obs_dict[name] = local

        # Resource agents: shared + per-device slice from bundle obs
        for bundle_idx, (bundle_names, bundle_start, dev_obs_dim) in enumerate(
            zip(
                self._bundle_agent_names_list,
                self._bundle_obs_offsets,
                self._bundle_per_device_obs_dims,
            )
        ):
            for j, name in enumerate(bundle_names):
                start = bundle_start + j * dev_obs_dim
                device_obs = obs_flat[start: start + dev_obs_dim]
                local = jnp.concatenate([shared, device_obs])
                pad = self._obs_dim - local.shape[0]
                if pad > 0:
                    local = jnp.concatenate([local, jnp.zeros(pad, dtype=jnp.float32)])
                obs_dict[name] = local

        return obs_dict


# ============ DistGridMARLEnv ============

class DistGridMARLEnv(MultiAgentEnvironment):
    """Multi-agent distribution grid environment — one agent per resource device.

    Wraps DistGridEnv so each attached battery (or other resource) becomes an
    independent agent.  Requires ``include_der=False`` in ``dist_params``.

    All physics are delegated to ``DistGridEnv.step()``.  Reward is shared:
    all agents receive the same scalar reward.  An optional ``voltage_penalty``
    adds a continuous voltage-deviation term on top of the base loss reward::

        effective_reward = base_reward - voltage_penalty * cost_continuous

    Args:
        dist_env:        DistGridEnv instance.
        dist_params:     DistGridParams with resources attached and
                         ``include_der=False``.
        voltage_penalty: Weight on the continuous voltage-deviation cost
                         (v_under + v_over summed across buses).  Set > 0 to
                         shape rewards toward voltage regulation.
        soc_penalty:     Weight on the terminal SOC deviation penalty
                         (``info["soc_terminal_sq"]``, only non-zero at episode
                         end).  Set > 0 to discourage extreme terminal SOC.
        reward_mode:     ``"shared"`` — all agents get the same reward.

    Agent naming:
        One agent per device, globally indexed per resource type:
        ``"battery_0"``, ``"battery_1"``, …

    Obs layout per agent:
        ``[grid_core | own_device_obs]`` padded to uniform ``obs_dim``.
        Grid core = v_norm(N) + p_flow_norm(L) + q_flow_norm(L) +
                    p_load_norm(N) + q_load_norm(N) + sin + cos.

    Action per agent:
        Shape ``(per_device_action_dim,)`` in ``[-1, 1]``.
        P-only (default): ``(1,)``; P+Q (``enable_q_control=True``): ``(2,)``.

    Usage::

        bundle = make_battery_bundle(case, bus_ids=[18, 25, 33], power_mw=0.75)
        params = make_dist_params(case, resources=[bundle], include_der=False)
        env = DistGridMARLEnv(DistGridEnv(), params, voltage_penalty=8.0)
        obs_dict, state = env.reset(key)
        obs_dict, state, rewards, dones, info = env.step(key, state, actions)
    """

    def __init__(
        self,
        dist_env: DistGridEnv,
        dist_params: DistGridParams,
        voltage_penalty: float = 0.0,
        soc_penalty: float = 0.0,
        reward_mode: str = "shared",
        observation_mode: str = "global",
    ):
        self._dist_env = dist_env
        self._dist_params = dist_params
        self._voltage_penalty = float(voltage_penalty)
        self._soc_penalty = float(soc_penalty)
        self._reward_mode = reward_mode
        self._observation_mode = observation_mode

        # Grid core obs dimension (static, independent of resources)
        n = dist_params.case.n_nodes
        nl = dist_params.topo.n_lines
        self._grid_core_dim: int = 3 * n + 2 * nl + 2  # v(N)+pf(L)+qf(L)+pl(N)+ql(N)+t(2)
        # Used by local obs mode: offset of time sin/cos in flat obs
        self._n_nodes: int = n
        self._sincos_start: int = 3 * n + 2 * nl  # = grid_core_dim - 2

        # Build agent names and bundle info
        type_count: Dict[str, int] = {}
        all_names: List[str] = []
        # List of (agent_names_for_bundle, bundle_obs_start_in_flat, per_device_obs_dim)
        bundle_info: List[tuple] = []
        # Per-agent bus indices (0-based internal indices), for local obs
        agent_bus_indices: List[int] = []

        offset = self._grid_core_dim
        for bundle in dist_params.resources:
            bundle_type = type(bundle).__name__.replace("Bundle", "").lower()
            count = type_count.get(bundle_type, 0)
            names = [f"{bundle_type}_{count + j}" for j in range(bundle.n_devices)]
            type_count[bundle_type] = count + bundle.n_devices
            all_names.extend(names)
            bundle_info.append((names, offset, bundle.per_device_obs_dim))
            offset += bundle.obs_dim
            bus_arr = np.asarray(bundle.bus_idx, dtype=np.int32)
            for j in range(bundle.n_devices):
                agent_bus_indices.append(int(bus_arr[j]))

        self._all_names = all_names
        self._bundle_info = bundle_info

        max_dev_obs = max(
            (b.per_device_obs_dim for b in dist_params.resources), default=0
        )

        if observation_mode == "local":
            # Precompute K-hop neighbour indices for each agent (Python time, not JAX)
            self._agent_neighbor_idx: List[np.ndarray] = _compute_local_neighbor_idx(
                dist_params.topo.sending_node_idx,
                dist_params.topo.receiving_node_idx,
                n,
                agent_bus_indices,
                K=_LOCAL_K,
            )
            self._agent_bus_indices: List[int] = agent_bus_indices
            # Local obs = own_v(1) + neighbor_v(K) + global_stats(3) + sin_cos(2) + device_obs
            self._obs_dim = 1 + _LOCAL_K + 3 + 2 + max_dev_obs
        else:
            # Global obs = grid_core + max per-device obs dim (backward compat)
            self._agent_neighbor_idx = []
            self._agent_bus_indices = []
            self._obs_dim = self._grid_core_dim + max_dev_obs

        # Uniform action dim per agent: max per_device_action_dim across bundles.
        self._per_device_action_dim: int = max(
            (b.per_device_action_dim for b in dist_params.resources), default=1
        )

    # ---- Properties ----

    @property
    def num_agents(self) -> int:
        return len(self._all_names)

    @property
    def agent_names(self) -> List[str]:
        return self._all_names

    @property
    def name(self) -> str:
        return "DistGridMARLEnv"

    def observation_space(self, agent: str = None) -> Box:
        return Box(
            low=jnp.full((self._obs_dim,), -jnp.inf, dtype=jnp.float32),
            high=jnp.full((self._obs_dim,), jnp.inf, dtype=jnp.float32),
            shape=(self._obs_dim,),
            dtype=jnp.float32,
        )

    def action_space(self, agent: str = None) -> Box:
        d = self._per_device_action_dim
        return Box(
            low=jnp.full((d,), -1.0, dtype=jnp.float32),
            high=jnp.full((d,), 1.0, dtype=jnp.float32),
            shape=(d,),
            dtype=jnp.float32,
        )

    # ---- Core Methods ----

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], MARLState]:
        obs_flat, grid_state = self._dist_env.reset(key, self._dist_params)
        state = MARLState(grid_state=grid_state)
        obs_dict = self._build_obs_dict(obs_flat)
        return obs_dict, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: MARLState,
        actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], MARLState, Dict[str, chex.Array], Dict[str, chex.Array], Dict[str, Any]]:
        flat_action = self._pack_actions(actions)

        obs_flat, new_grid_state, reward, costs, done, info = self._dist_env.step(
            key, state.grid_state, flat_action, self._dist_params
        )
        info = {
            **info,
            "constraint_costs": costs,
            "cost": info.get("cost_sum", jnp.sum(costs)),
        }

        effective_reward = (
            reward
            - self._voltage_penalty * info["cost_continuous"]
            - self._soc_penalty * info.get("soc_terminal_sq", jnp.float32(0.0))
        )

        new_state = MARLState(grid_state=new_grid_state)
        obs_dict = self._build_obs_dict(obs_flat)

        rewards = {name: effective_reward for name in self._all_names}
        dones = {name: done for name in self._all_names}
        dones["__all__"] = done

        return obs_dict, new_state, rewards, dones, info

    # ---- Internal Helpers ----

    def _pack_actions(self, actions: Dict[str, chex.Array]) -> chex.Array:
        """Pack per-agent action dict into flat action array for DistGridEnv."""
        return jnp.concatenate([actions[name].reshape(-1) for name in self._all_names])

    def _build_obs_dict(self, obs_flat: chex.Array) -> Dict[str, chex.Array]:
        """Unpack flat DistGridEnv obs into per-agent observation dict."""
        if self._observation_mode == "local":
            return self._build_local_obs_dict(obs_flat)
        return self._build_global_obs_dict(obs_flat)

    def _build_global_obs_dict(self, obs_flat: chex.Array) -> Dict[str, chex.Array]:
        """Global obs: full grid_core + own device state (legacy / backward compat)."""
        shared = obs_flat[:self._grid_core_dim]

        obs_dict: Dict[str, chex.Array] = {}
        for names, bundle_start, dev_obs_dim in self._bundle_info:
            for j, name in enumerate(names):
                start = bundle_start + j * dev_obs_dim
                device_obs = obs_flat[start: start + dev_obs_dim]
                local = jnp.concatenate([shared, device_obs])
                pad = self._obs_dim - local.shape[0]
                if pad > 0:
                    local = jnp.concatenate([local, jnp.zeros(pad, dtype=jnp.float32)])
                obs_dict[name] = local

        return obs_dict

    def _build_local_obs_dict(self, obs_flat: chex.Array) -> Dict[str, chex.Array]:
        """Local (Dec-POMDP) obs: own bus v + K-hop neighbours + global stats + time + device.

        Layout (fixed, uniform across all agents):
            own_v          (1,)          — own bus voltage normalised (v-1)/0.1
            neighbor_v     (_LOCAL_K,)   — K nearest neighbour bus voltages
            global_stats   (3,)          — [v_min, v_max, v_mean] across all buses
            sin_cos        (2,)          — time features from grid core obs
            device_obs     (max_dev_obs,) — own resource state (padded to uniform dim)

        Total = 1 + K + 3 + 2 + max_dev_obs.
        """
        v_all = obs_flat[:self._n_nodes]                             # (N,)
        global_stats = jnp.array(
            [jnp.min(v_all), jnp.max(v_all), jnp.mean(v_all)],
            dtype=jnp.float32,
        )
        sin_cos = obs_flat[self._sincos_start: self._sincos_start + 2]  # (2,)

        obs_dict: Dict[str, chex.Array] = {}
        agent_idx = 0
        for names, bundle_start, dev_obs_dim in self._bundle_info:
            for j, name in enumerate(names):
                own_bus = self._agent_bus_indices[agent_idx]
                neighbor_idx = self._agent_neighbor_idx[agent_idx]  # np.ndarray(K,) int32

                own_v = v_all[own_bus: own_bus + 1]                      # (1,)
                neighbor_v = v_all[jnp.asarray(neighbor_idx)]             # (K,)

                start = bundle_start + j * dev_obs_dim
                device_obs = obs_flat[start: start + dev_obs_dim]         # (dev_obs_dim,)

                local = jnp.concatenate([own_v, neighbor_v, global_stats, sin_cos, device_obs])
                pad = self._obs_dim - local.shape[0]
                if pad > 0:
                    local = jnp.concatenate([local, jnp.zeros(pad, dtype=jnp.float32)])
                obs_dict[name] = local
                agent_idx += 1

        return obs_dict


# ============ DistGrid3PhaseMARLEnv ============

class DistGrid3PhaseMARLEnv(MultiAgentEnvironment):
    """Multi-agent 3-phase distribution grid env — one agent per resource device.

    Wraps :class:`~powerzoojax.envs.grid.dist_3phase.DistGrid3PhaseEnv`.
    Always emits **phase-averaged local** Dec-POMDP observations so the obs_dim
    matches :class:`DistGridMARLEnv` in local mode (``1 + K + 3 + 2 + dev_obs``),
    enabling zero-shot transfer of policies trained on 1-phase networks.

    Phase averaging: ``v_avg[b] = (V_A[b] + V_B[b] + V_C[b]) / 3``.

    Local obs layout per agent (same as DistGridMARLEnv local mode):
        own_v         (1,)         — phase-avg voltage at own bus
        neighbor_v    (_LOCAL_K,)  — K nearest neighbour phase-avg voltages
        global_stats  (3,)         — [v_min, v_max, v_mean] phase-averaged
        sin_cos       (2,)         — time features
        device_obs    (max_dev_obs,)
    """

    def __init__(
        self,
        dist3ph_env: DistGrid3PhaseEnv,
        params: DistGrid3PhParams,
        voltage_penalty: float = 0.0,
        min_obs_dim: int = 0,
    ):
        """
        Args:
            dist3ph_env:   DistGrid3PhaseEnv instance.
            params:        DistGrid3PhParams with resources attached.
            voltage_penalty: Weight on voltage cost in reward shaping.
            min_obs_dim:   Minimum obs_dim.  Set to the 1-phase training
                           env's obs_dim to guarantee shape compatibility for
                           zero-shot transfer.  The shortfall is zero-padded.
        """
        self._dist_env = dist3ph_env
        self._dist_params = params
        self._voltage_penalty = float(voltage_penalty)

        n = params.topo.n_nodes
        nl = params.topo.n_lines
        self._n_nodes = n
        self._n_lines = nl
        # Flat obs layout: [V_A(n) | V_B(n) | V_C(n) | P_A(nl)...Q_C(nl) |
        #                   pA(n)...qC(n) | sin | cos | bundle_obs]
        # Positions: V_A at 0, V_B at n, V_C at 2n; sincos at 9n+6nl
        self._sincos_start_3ph: int = 9 * n + 6 * nl
        self._bundle_obs_start: int = 9 * n + 6 * nl + 2

        type_count: Dict[str, int] = {}
        all_names: List[str] = []
        bundle_info: List[tuple] = []
        agent_bus_indices: List[int] = []

        offset = self._bundle_obs_start
        for bundle in params.resources:
            bundle_type = type(bundle).__name__.replace("Bundle", "").lower()
            count = type_count.get(bundle_type, 0)
            names = [f"{bundle_type}_{count + j}" for j in range(bundle.n_devices)]
            type_count[bundle_type] = count + bundle.n_devices
            all_names.extend(names)
            bundle_info.append((names, offset, bundle.per_device_obs_dim))
            offset += bundle.obs_dim
            bus_arr = np.asarray(bundle.bus_idx, dtype=np.int32)
            for j in range(bundle.n_devices):
                agent_bus_indices.append(int(bus_arr[j]))

        self._all_names = all_names
        self._bundle_info = bundle_info

        max_dev_obs = max(
            (b.per_device_obs_dim for b in params.resources), default=0
        )

        self._agent_neighbor_idx: List[np.ndarray] = _compute_local_neighbor_idx(
            params.topo.from_nodes,
            params.topo.to_nodes,
            n,
            agent_bus_indices,
            K=_LOCAL_K,
        )
        self._agent_bus_indices: List[int] = agent_bus_indices
        # obs_dim: local base + device_obs, padded up to min_obs_dim if given.
        # min_obs_dim is used to match the 1-phase training env obs_dim so
        # zero-shot policy transfer does not hit a shape mismatch.
        self._obs_dim: int = max(1 + _LOCAL_K + 3 + 2 + max_dev_obs, int(min_obs_dim))
        self._per_device_action_dim: int = max(
            (b.per_device_action_dim for b in params.resources), default=1
        )

    @property
    def num_agents(self) -> int:
        return len(self._all_names)

    @property
    def agent_names(self) -> List[str]:
        return self._all_names

    @property
    def name(self) -> str:
        return "DistGrid3PhaseMARLEnv"

    def observation_space(self, agent: str = None) -> Box:
        return Box(
            low=jnp.full((self._obs_dim,), -jnp.inf, dtype=jnp.float32),
            high=jnp.full((self._obs_dim,), jnp.inf, dtype=jnp.float32),
            shape=(self._obs_dim,),
            dtype=jnp.float32,
        )

    def action_space(self, agent: str = None) -> Box:
        d = self._per_device_action_dim
        return Box(
            low=jnp.full((d,), -1.0, dtype=jnp.float32),
            high=jnp.full((d,), 1.0, dtype=jnp.float32),
            shape=(d,),
            dtype=jnp.float32,
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], MARLState]:
        obs_flat, grid_state = self._dist_env.reset(key, self._dist_params)
        state = MARLState(grid_state=grid_state)
        return self._build_local_obs_dict_3ph(obs_flat), state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: MARLState,
        actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], MARLState, Dict[str, chex.Array], Dict[str, chex.Array], Dict[str, Any]]:
        flat_action = jnp.concatenate(
            [actions[name].reshape(-1) for name in self._all_names]
        )
        obs_flat, new_grid_state, reward, costs, done, info = self._dist_env.step(
            key, state.grid_state, flat_action, self._dist_params
        )
        info = {
            **info,
            "constraint_costs": costs,
            "cost": info.get("cost_sum", jnp.sum(costs)),
        }
        effective_reward = reward - self._voltage_penalty * info["cost_continuous"]
        new_state = MARLState(grid_state=new_grid_state)
        obs_dict = self._build_local_obs_dict_3ph(obs_flat)
        rewards = {name: effective_reward for name in self._all_names}
        dones = {name: done for name in self._all_names}
        dones["__all__"] = done
        return obs_dict, new_state, rewards, dones, info

    def _build_local_obs_dict_3ph(
        self, obs_flat: chex.Array
    ) -> Dict[str, chex.Array]:
        """Phase-averaged local obs from 3-phase flat observation.

        V_A = obs[:n], V_B = obs[n:2n], V_C = obs[2n:3n].
        v_avg[b] = (V_A[b] + V_B[b] + V_C[b]) / 3  (already normalised).
        """
        n = self._n_nodes
        v_avg = (obs_flat[:n] + obs_flat[n: 2 * n] + obs_flat[2 * n: 3 * n]) / 3.0
        global_stats = jnp.array(
            [jnp.min(v_avg), jnp.max(v_avg), jnp.mean(v_avg)],
            dtype=jnp.float32,
        )
        sin_cos = obs_flat[self._sincos_start_3ph: self._sincos_start_3ph + 2]

        obs_dict: Dict[str, chex.Array] = {}
        agent_idx = 0
        for names, bundle_start, dev_obs_dim in self._bundle_info:
            for j, name in enumerate(names):
                own_bus = self._agent_bus_indices[agent_idx]
                neighbor_idx = self._agent_neighbor_idx[agent_idx]
                own_v = v_avg[own_bus: own_bus + 1]
                neighbor_v = v_avg[jnp.asarray(neighbor_idx)]
                start = bundle_start + j * dev_obs_dim
                device_obs = obs_flat[start: start + dev_obs_dim]
                local = jnp.concatenate(
                    [own_v, neighbor_v, global_stats, sin_cos, device_obs]
                )
                pad = self._obs_dim - local.shape[0]
                if pad > 0:
                    local = jnp.concatenate(
                        [local, jnp.zeros(pad, dtype=jnp.float32)]
                    )
                obs_dict[name] = local
                agent_idx += 1
        return obs_dict
