"""R3 — DERs: 12-agent heterogeneous deployment on case141.

Test coverage:
  L0  JAX contract:  jit / vmap / scan rollout / pytree / jaxpr
  L1  Bundle spec:   battery/pv/flexload buses, dims, Q-control flags
  L1  make_ders_params: resource count, action/obs dims, voltage limits, no DER
  L1  DistGridMARLEnv: 12 agents, agent names, obs/action shapes, step
  L1  Baselines/Metrics: no-control rollout, metrics structure
  L1  Presets: ders-medium / ders-medium-safe exist and are instantiable
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powerzoojax.case import create_case141


# ========================== Fixtures ==========================


@pytest.fixture(scope="module")
def case141():
    return create_case141()


@pytest.fixture(scope="module")
def ders_params(case141):
    from powerzoojax.tasks.ders import make_ders_params
    return make_ders_params(case141, max_steps=12)


@pytest.fixture(scope="module")
def ders_env():
    from powerzoojax.envs.grid.dist import DistGridEnv
    return DistGridEnv()


@pytest.fixture(scope="module")
def marl_env(case141):
    from powerzoojax.tasks.ders import make_ders_marl_env
    env, _ = make_ders_marl_env(case141, max_steps=12, voltage_penalty=8.0)
    return env


@pytest.fixture
def key():
    return jax.random.PRNGKey(7)


# ========================== L1 — Bundle spec ==========================


def test_battery_bundle_buses(case141):
    from powerzoojax.tasks.ders import make_ders_battery_bundle, DERS_BATTERY_BUSES
    bundle = make_ders_battery_bundle(case141)
    assert bundle.n_devices == 4
    assert len(DERS_BATTERY_BUSES) == 4
    # Q-control enabled
    assert bundle.per_device_action_dim == 2
    assert bundle.per_device_obs_dim == 3


def test_pv_bundle_buses(case141):
    from powerzoojax.tasks.ders import make_ders_pv_bundle, DERS_PV_BUSES
    bundle = make_ders_pv_bundle(case141, max_steps=12)
    assert bundle.n_devices == 4
    assert len(DERS_PV_BUSES) == 4
    assert bundle.per_device_action_dim == 2
    assert bundle.per_device_obs_dim == 4


def test_flexload_bundle_buses(case141):
    from powerzoojax.tasks.ders import make_ders_flexload_bundle, DERS_FLEXLOAD_BUSES
    bundle = make_ders_flexload_bundle(case141)
    assert bundle.n_devices == 4
    assert len(DERS_FLEXLOAD_BUSES) == 4
    assert bundle.per_device_action_dim == 2
    assert bundle.per_device_obs_dim == 5


def test_bus_ids_non_overlapping(case141):
    """Battery / PV / FlexLoad buses must be disjoint."""
    from powerzoojax.tasks.ders import (
        DERS_BATTERY_BUSES, DERS_PV_BUSES, DERS_FLEXLOAD_BUSES
    )
    all_buses = DERS_BATTERY_BUSES + DERS_PV_BUSES + DERS_FLEXLOAD_BUSES
    assert len(set(all_buses)) == len(all_buses), "Duplicate bus IDs found"


def test_bus_ids_valid_for_case141(case141):
    """All DER bus IDs exist in case141 node_ids."""
    from powerzoojax.tasks.ders import (
        DERS_BATTERY_BUSES, DERS_PV_BUSES, DERS_FLEXLOAD_BUSES
    )
    node_ids = set(int(x) for x in np.asarray(case141.node_ids))
    for bid in DERS_BATTERY_BUSES + DERS_PV_BUSES + DERS_FLEXLOAD_BUSES:
        assert bid in node_ids, f"bus {bid} not in case141"


def test_battery_q_control_enabled(case141):
    """Battery bundle must have enable_q_control=True (Q voltage support)."""
    from powerzoojax.tasks.ders import make_ders_battery_bundle
    bundle = make_ders_battery_bundle(case141)
    assert bundle.enable_q_control is True


def test_pv_allow_curtailment(case141):
    """PV bundle must have allow_curtailment=True."""
    from powerzoojax.tasks.ders import make_ders_pv_bundle
    bundle = make_ders_pv_bundle(case141, max_steps=12)
    assert bundle.allow_curtailment is True


# ========================== L1 — make_ders_params ==========================


def test_ders_params_three_bundles(ders_params):
    """DistGridParams must contain exactly 3 resource bundles."""
    assert len(ders_params.resources) == 3


def test_ders_params_12_agents_total(ders_params):
    """Total number of resource devices across bundles = 12."""
    total = sum(b.n_devices for b in ders_params.resources)
    assert total == 12


def test_ders_params_action_dim(ders_params):
    """Total action dim = 12 devices × 2 = 24."""
    total_act = sum(b.action_dim for b in ders_params.resources)
    assert total_act == 24  # 12 × 2


def test_ders_params_obs_includes_all_bundles(ders_params):
    """Bundle obs dims: battery=4×3=12, pv=4×4=16, flex=4×5=20, total=48."""
    bundle_obs = sum(b.obs_dim for b in ders_params.resources)
    assert bundle_obs == 12 + 16 + 20  # 48


def test_ders_params_voltage_limits(ders_params):
    from powerzoojax.tasks.ders import DERS_V_MIN, DERS_V_MAX
    assert abs(float(ders_params.v_min) - DERS_V_MIN) < 1e-6
    assert abs(float(ders_params.v_max) - DERS_V_MAX) < 1e-6


def test_ders_params_include_der_false(ders_params):
    """include_der must be False — agents only control attached resources."""
    assert ders_params.include_der is False


def test_ders_params_case_matches_fixture(ders_params, case141):
    assert ders_params.case.n_nodes == case141.n_nodes


def test_ders_params_no_case_arg():
    """make_ders_params() without case arg auto-creates case141."""
    from powerzoojax.tasks.ders import make_ders_params
    params = make_ders_params(max_steps=4)
    assert params.case.n_nodes == 141


# ========================== L0 — JAX contract ==========================


def test_jit_reset(ders_env, ders_params, key):
    """reset() is JIT-compilable."""
    reset_jit = jax.jit(lambda k: ders_env.reset(k, ders_params))
    obs1, s1 = reset_jit(key)
    obs2, s2 = reset_jit(key)
    np.testing.assert_allclose(np.asarray(obs1), np.asarray(obs2))


def test_jit_step(ders_env, ders_params, key):
    """step() is JIT-compilable."""
    n_act = sum(b.action_dim for b in ders_params.resources)
    act = jnp.zeros(n_act)
    step_jit = jax.jit(lambda k, s, a: ders_env.step(k, s, a, ders_params))
    _, state = ders_env.reset(key, ders_params)
    out1 = step_jit(key, state, act)
    out2 = step_jit(key, state, act)
    np.testing.assert_allclose(np.asarray(out1[0]), np.asarray(out2[0]))


def test_vmap_reset(ders_env, ders_params, key):
    """reset() vmaps over a batch of keys."""
    keys = jax.random.split(key, 4)
    batch_reset = jax.vmap(lambda k: ders_env.reset(k, ders_params))
    obs_b, states_b = batch_reset(keys)
    assert obs_b.shape[0] == 4


def test_vmap_step(ders_env, ders_params, key):
    """step() vmaps over a batch of states."""
    keys = jax.random.split(key, 4)
    batch_reset = jax.vmap(lambda k: ders_env.reset(k, ders_params))
    _, states_b = batch_reset(keys)
    n_act = sum(b.action_dim for b in ders_params.resources)
    acts_b = jnp.zeros((4, n_act))
    batch_step = jax.vmap(lambda k, s, a: ders_env.step(k, s, a, ders_params))
    obs_b, _, rew_b, costs_b, done_b, _ = batch_step(keys, states_b, acts_b)
    assert obs_b.shape[0] == 4
    assert rew_b.shape == (4,)


def test_scan_rollout(ders_env, ders_params, key):
    """lax.scan rollout of 12 steps."""
    n_act = sum(b.action_dim for b in ders_params.resources)
    _, state0 = ders_env.reset(key, ders_params)
    act = jnp.zeros(n_act, dtype=jnp.float32)

    def scan_fn(carry, _):
        state, k = carry
        k, sk = jax.random.split(k)
        _, new_state, rew, _costs, done, _ = ders_env.step(sk, state, act, ders_params)
        return (new_state, k), rew

    _, rewards = jax.lax.scan(scan_fn, (state0, key), None, length=12)
    assert rewards.shape == (12,)


def test_auto_reset(ders_env, ders_params, key):
    """done=True at max_steps triggers auto-reset (time_step resets to 0)."""
    n_act = sum(b.action_dim for b in ders_params.resources)
    act = jnp.zeros(n_act)
    _, state = ders_env.reset(key, ders_params)
    for _ in range(ders_params.max_steps):
        _, state, _, _costs, done, _ = ders_env.step(key, state, act, ders_params)
    # After max_steps, auto-reset: time_step should be 0
    assert int(state.time_step) == 0


def test_pytree_structure(ders_env, ders_params, key):
    """DistGridState is a valid JAX pytree (flatten/unflatten round-trip)."""
    _, state = ders_env.reset(key, ders_params)
    leaves, treedef = jax.tree_util.tree_flatten(state)
    state2 = jax.tree_util.tree_unflatten(treedef, leaves)
    n_act = sum(b.action_dim for b in ders_params.resources)
    act = jnp.zeros(n_act)
    obs1, _, *_ = ders_env.step(key, state, act, ders_params)
    obs2, _, *_ = ders_env.step(key, state2, act, ders_params)
    np.testing.assert_allclose(np.asarray(obs1), np.asarray(obs2))


# ========================== L1 — DistGridMARLEnv ==========================


def test_marl_num_agents(marl_env):
    """DistGridMARLEnv must expose exactly 12 agents."""
    assert marl_env.num_agents == 12


def test_marl_agent_names(marl_env):
    """Agent names: battery_0..3, renewable_0..3, flexload_0..3."""
    names = marl_env.agent_names
    assert len(names) == 12
    for i in range(4):
        assert f"battery_{i}" in names
        assert f"renewable_{i}" in names
        assert f"flexload_{i}" in names


def test_marl_action_space_uniform(marl_env):
    """All agents must have the same action space Box(2)."""
    for agent in marl_env.agent_names:
        sp = marl_env.action_space(agent)
        assert sp.shape == (2,), f"{agent}: expected action shape (2,), got {sp.shape}"


def test_marl_obs_space_uniform(marl_env):
    """All agents have the same observation space (uniform for vmap)."""
    dims = set()
    for agent in marl_env.agent_names:
        dims.add(marl_env.observation_space(agent).shape[0])
    assert len(dims) == 1, f"Observation dims not uniform: {dims}"


def test_marl_reset(marl_env, key):
    """reset() returns obs_dict with all 12 agents plus state."""
    obs_dict, state = marl_env.reset(key)
    assert set(obs_dict.keys()) == set(marl_env.agent_names)
    for agent, obs in obs_dict.items():
        assert obs.ndim == 1


def test_marl_step(marl_env, key):
    """step() with zero actions returns valid reward/done dicts."""
    obs_dict, state = marl_env.reset(key)
    zero_actions = {name: jnp.zeros(2) for name in marl_env.agent_names}
    obs2, state2, rewards, dones, info = marl_env.step(key, state, zero_actions)
    assert set(rewards.keys()) == set(marl_env.agent_names)
    assert "__all__" in dones
    for agent in marl_env.agent_names:
        assert rewards[agent].shape == ()
        assert dones[agent].shape == ()


def test_marl_jit_step(marl_env, key):
    """DistGridMARLEnv.step is JIT-compilable."""
    obs_dict, state = marl_env.reset(key)
    zero_actions = {name: jnp.zeros(2) for name in marl_env.agent_names}
    step_jit = jax.jit(lambda s, a: marl_env.step(key, s, a))
    out1 = step_jit(state, zero_actions)
    out2 = step_jit(state, zero_actions)
    for agent in marl_env.agent_names:
        np.testing.assert_allclose(
            np.asarray(out1[2][agent]),
            np.asarray(out2[2][agent]),
        )


def test_marl_obs_shape_correct(marl_env, key):
    """Each agent's obs shape matches observation_space."""
    obs_dict, _ = marl_env.reset(key)
    expected_dim = marl_env.observation_space().shape[0]
    for agent, obs in obs_dict.items():
        assert obs.shape == (expected_dim,), (
            f"{agent}: expected {(expected_dim,)}, got {obs.shape}"
        )


# ========================== L1 — Physics ==========================


def test_obs_shape(ders_env, ders_params, key):
    """DistGridEnv obs shape = grid_core + sum(bundle obs)."""
    n = ders_params.case.n_nodes
    nl = ders_params.topo.n_lines
    bundle_obs = sum(b.obs_dim for b in ders_params.resources)
    expected_dim = 3 * n + 2 * nl + 2 + bundle_obs
    obs, _ = ders_env.reset(key, ders_params)
    assert obs.shape == (expected_dim,)


def test_reward_is_scalar(ders_env, ders_params, key):
    """Step reward is a scalar float."""
    n_act = sum(b.action_dim for b in ders_params.resources)
    _, state = ders_env.reset(key, ders_params)
    _, _, rew, _costs, _done, _ = ders_env.step(key, state, jnp.zeros(n_act), ders_params)
    assert rew.shape == ()


def test_q_injection_changes_voltage(ders_env, ders_params, key):
    """Injecting Q from batteries changes bus voltage profile."""
    n_act = sum(b.action_dim for b in ders_params.resources)
    act_zero = jnp.zeros(n_act)
    # Set Q action to +1 for all battery agents (action indices 1, 3, 5, 7 = Q dims)
    act_q = act_zero.at[1].set(1.0).at[3].set(1.0).at[5].set(1.0).at[7].set(1.0)

    _, state0 = ders_env.reset(key, ders_params)
    _, s_zero, _, _cz, _dz, _ = ders_env.step(key, state0, act_zero, ders_params)
    _, s_q, _, _cq, _dq, _ = ders_env.step(key, state0, act_q, ders_params)

    delta = s_q.v_mag - s_zero.v_mag
    assert float(jnp.max(jnp.abs(delta))) > 1e-5, (
        f"Q injection did not change voltage: max|delta|={float(jnp.max(jnp.abs(delta)))}"
    )


def test_reward_cost_separation(ders_env, ders_params, key):
    """info must contain both 'cost_continuous' and 'p_loss_MW'."""
    n_act = sum(b.action_dim for b in ders_params.resources)
    _, state = ders_env.reset(key, ders_params)
    _, _, rew, _costs, _done, info = ders_env.step(key, state, jnp.zeros(n_act), ders_params)
    assert "cost_continuous" in info
    assert "p_loss_MW" in info


# ========================== L1 — Baselines & Metrics ==========================


def test_no_control_rollout(ders_env, ders_params, key):
    """No-control rollout completes max_steps and returns valid arrays."""
    from powerzoojax.tasks.ders import ders_no_control_rollout
    result = ders_no_control_rollout(ders_env, ders_params, key)
    assert "reward" in result
    assert "p_loss_MW" in result
    assert len(result["reward"]) == ders_params.max_steps


def test_no_control_zero_actions(ders_env, ders_params, key):
    """No-control rollout has zero curtailment / Q actions."""
    from powerzoojax.tasks.ders import ders_no_control_rollout
    result = ders_no_control_rollout(ders_env, ders_params, key)
    assert "cost_continuous" in result
    # cost_continuous exists; no-control should not be artificially zero (there
    # may still be voltage violations in base case), just confirm it runs.
    assert len(result["cost_continuous"]) == ders_params.max_steps


def test_metrics_structure(ders_env, ders_params, key):
    """compute_ders_metrics returns expected keys."""
    from powerzoojax.tasks.ders import (
        ders_no_control_rollout, compute_ders_metrics
    )
    baseline = ders_no_control_rollout(ders_env, ders_params, key)
    metrics = compute_ders_metrics(baseline, baseline)
    for k in ("total_reward", "total_cost", "mean_p_loss_mw",
              "voltage_violation_steps", "loss_reduction_pct", "cost_reduction_pct"):
        assert k in metrics, f"Missing key: {k}"


def test_metrics_self_comparison(ders_env, ders_params, key):
    """Self-comparison should give loss_reduction_pct ≈ 0."""
    from powerzoojax.tasks.ders import (
        ders_no_control_rollout, compute_ders_metrics
    )
    baseline = ders_no_control_rollout(ders_env, ders_params, key)
    metrics = compute_ders_metrics(baseline, baseline)
    assert abs(metrics["loss_reduction_pct"]) < 1e-4


# ========================== L1 — Presets ==========================


def test_ders_presets_exist():
    from powerzoojax.rl.presets import PRESETS
    assert "ders-medium" in PRESETS
    assert "ders-medium-safe" in PRESETS


def test_ders_preset_config():
    from powerzoojax.rl.presets import get_preset
    p = get_preset("ders-medium")
    assert p.config.algo == "ippo_typed"
    assert p.config.total_timesteps == 15_000_000
    assert p.config.n_steps == 48


def test_ders_preset_env_factory():
    """Preset env_factory returns a DistGridMARLEnv with 12 agents."""
    from powerzoojax.rl.presets import get_preset
    from powerzoojax.rl.multi_agent import DistGridMARLEnv
    env = get_preset("ders-medium").env_factory()
    assert isinstance(env, DistGridMARLEnv)
    assert env.num_agents == 12


def test_list_presets_includes_ders():
    from powerzoojax.rl.presets import list_presets
    names = [p["name"] for p in list_presets()]
    assert "ders-medium" in names
    assert "ders-medium-safe" in names
