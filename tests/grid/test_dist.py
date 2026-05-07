"""Tests for DistGridEnv — L0 JAX + L1 env behavior + L1.5 cross-module.

L0: JIT, vmap, scan rollout, auto_reset (incl. PQ battery bundle).
L1: obs shape, DER injection, voltage safety, cost separation.
L1.5: DER + load interaction consistency.
"""

import jax
import jax.numpy as jnp
import jax.tree_util as tu
import numpy as np
import pytest

from powerzoojax.case import create_case33bw
from powerzoojax.envs.resource.battery import make_battery_bundle
from powerzoojax.envs.grid.dist import (
    DistGridEnv,
    DistGridState,
    DistGridParams,
    make_dist_params,
)


@pytest.fixture(scope="module")
def case33():
    return create_case33bw()


@pytest.fixture(scope="module")
def params(case33):
    return make_dist_params(case33, max_steps=48)


@pytest.fixture(scope="module")
def env():
    return DistGridEnv()


@pytest.fixture(scope="module")
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture(scope="module")
def init(env, key, params):
    return env.reset(key, params)


# ==================== L0: JAX Contract ====================

class TestDistGridEnv_L0_JAX:

    def test_reset_jit(self, env, key, params):
        obs, state = env.reset(key, params)
        assert obs.shape[0] > 0
        assert state.v_mag.shape == (params.case.n_nodes,)

    def test_step_jit(self, env, key, init, params):
        obs, state = init
        action = jnp.zeros(params.case.n_nodes - 1)
        obs2, state2, reward, costs, done, info = env.step(key, state, action, params)
        assert obs2.shape == obs.shape
        assert reward.shape == ()

    def test_vmap_reset(self, env, params):
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        obs_b, state_b = jax.vmap(lambda k: env.reset(k, params))(keys)
        assert obs_b.shape[0] == 4
        assert state_b.v_mag.shape == (4, params.case.n_nodes)

    def test_vmap_step(self, env, params):
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        obs_b, state_b = jax.vmap(lambda k: env.reset(k, params))(keys)
        actions = jnp.zeros((4, params.case.n_nodes - 1))
        keys2 = jax.random.split(jax.random.PRNGKey(1), 4)
        obs2, state2, rew, costs, done, info = jax.vmap(
            lambda k, s, a: env.step(k, s, a, params)
        )(keys2, state_b, actions)
        assert obs2.shape[0] == 4
        assert rew.shape == (4,)
        assert costs.shape == (4, 3)

    def test_scan_rollout(self, env, key, params):
        obs, state = env.reset(key, params)
        action = jnp.zeros(params.case.n_nodes - 1)

        def step_fn(carry, _):
            s = carry
            k = jax.random.PRNGKey(0)
            obs, new_s, rew, costs, done, info = env.step(k, s, action, params)
            return new_s, (obs, rew, done)

        final_state, (obs_traj, rew_traj, done_traj) = jax.lax.scan(
            step_fn, state, None, length=params.max_steps)
        assert obs_traj.shape[0] == params.max_steps
        assert rew_traj.shape == (params.max_steps,)

    def test_auto_reset(self, env, key, params):
        """After max_steps, state should be reset."""
        obs, state = env.reset(key, params)
        action = jnp.zeros(params.case.n_nodes - 1)

        for _ in range(params.max_steps):
            obs, state, reward, costs, done, info = env.step(key, state, action, params)

        # After max_steps, auto-reset should have happened
        assert int(state.time_step) == 0

    def test_state_pytree(self, init):
        obs, state = init
        leaves = tu.tree_leaves(state)
        assert len(leaves) == 10  # DistGridState fields (incl. episode_batt_soc_init)

    def test_make_jaxpr(self, env, key, init, params):
        obs, state = init
        action = jnp.zeros(params.case.n_nodes - 1)
        jaxpr = jax.make_jaxpr(
            lambda k, s, a: env.step(k, s, a, params)
        )(key, state, action)
        assert jaxpr is not None


# ==================== L1: Physical Correctness ====================

class TestDistGridEnv_L1_Physics:

    def test_obs_shape(self, env, init, params):
        obs, state = init
        n = params.case.n_nodes
        m = params.topo.n_lines
        expected = 3 * n + 2 * m + 2  # v_norm + p_flow_norm + q_flow_norm + p_load_norm + q_load_norm + sin/cos
        assert obs.shape == (expected,)

    def test_obs_normalisation_scale(self, env, key, init, params):
        """All obs features should be on a similar scale after normalisation."""
        obs, state = init
        action = jnp.zeros(params.case.n_nodes - 1)
        obs1, _, _, _, _, _ = env.step(key, state, action, params)

        n = params.case.n_nodes
        m = params.topo.n_lines
        v_norm = obs1[:n]
        p_flow_norm = obs1[n:n + m]
        p_load_norm = obs1[n + 2 * m:2 * n + 2 * m]
        q_load_norm = obs1[2 * n + 2 * m:3 * n + 2 * m]

        # Trunk line flow ≈ 1.0 (sum of loads / ref ≈ 1 + losses)
        np.testing.assert_allclose(
            float(p_flow_norm[0]), 1.0, atol=0.2,
            err_msg="Trunk p_flow_norm should be ≈ 1.0")
        # Load features sum to 1.0
        np.testing.assert_allclose(
            float(p_load_norm.sum()), 1.0, atol=1e-4,
            err_msg="p_load_norm should sum to 1.0 (base-case)")
        np.testing.assert_allclose(
            float(q_load_norm.sum()), 1.0, atol=1e-4,
            err_msg="q_load_norm should sum to 1.0 (base-case)")
        # v_norm at slack bus = 0 (v_slack = 1.0)
        np.testing.assert_allclose(
            float(v_norm[0]), 0.0, atol=1e-5,
            err_msg="v_norm at slack bus should be 0")

    def test_reward_negative(self, env, key, init, params):
        """Reward should be negative (loss penalty only, no safety penalty)."""
        obs, state = init
        action = jnp.zeros(params.case.n_nodes - 1)
        _, _, reward, _, _, _ = env.step(key, state, action, params)
        assert float(reward) <= 0.0

    def test_reward_cost_separation(self, env, key, init, params):
        """Reward = -loss_penalty_weight * p_loss_MW (economic only); cost = count-based (safety only)."""
        obs, state = init
        action = jnp.zeros(params.case.n_nodes - 1)
        _, _, reward, costs, _, info = env.step(key, state, action, params)
        # Reward should equal -loss_penalty_weight * p_loss_MW
        expected_reward = -params.loss_penalty_weight * info["p_loss_MW"]
        np.testing.assert_allclose(
            float(reward), float(expected_reward), atol=1e-5,
            err_msg="Reward should be -loss_penalty_weight * p_loss_MW")
        np.testing.assert_allclose(
            float(info["cost_sum"]), float(jnp.sum(costs)), atol=1e-5,
            err_msg="cost_sum should equal sum(costs)")

    def test_cost_nonneg(self, env, key, init, params):
        obs, state = init
        action = jnp.zeros(params.case.n_nodes - 1)
        _, _, _, costs, _, info = env.step(key, state, action, params)
        assert bool(jnp.all(costs >= 0.0))
        assert float(info["cost_voltage_violation"]) >= 0.0
        assert float(info["cost_thermal_overload"]) >= 0.0
        assert float(info["cost_sum"]) >= 0.0

    def test_zero_der_baseline(self, env, key, init, params):
        """Zero DER injection → same as no-action baseline."""
        obs, state = init
        action = jnp.zeros(params.case.n_nodes - 1)
        _, state_z, *_ = env.step(key, state, action, params)
        # Voltage profile should reflect pure load scenario
        assert jnp.all(state_z.v_mag > 0.0)
        assert float(state_z.v_mag[0]) > float(state_z.v_mag.min())

    def test_der_injection_raises_voltage(self, env, key, init, params):
        """Positive DER injection should raise voltage at injection bus."""
        obs, state = init
        action_zero = jnp.zeros(params.case.n_nodes - 1)
        _, state_z, *_ = env.step(key, state, action_zero, params)

        # Inject at the last non-slack bus
        action_inject = jnp.zeros(params.case.n_nodes - 1)
        action_inject = action_inject.at[-1].set(0.05)
        _, state_i, *_ = env.step(key, state, action_inject, params)

        # Voltage at the far end should be higher with injection
        assert float(state_i.v_mag[-1]) > float(state_z.v_mag[-1]) - 1e-4

    def test_loss_positive(self, env, key, init, params):
        obs, state = init
        action = jnp.zeros(params.case.n_nodes - 1)
        _, state_n, _, _, _, info = env.step(key, state, action, params)
        assert float(state_n.p_loss_total) > 0.0


# ==================== L1.5: Cross-Module Interaction ====================

class TestDistGridEnv_L15_CrossModule:

    def test_heavy_load_more_violations(self, env, case33, key):
        """Heavier load profile → more voltage violations or higher cost."""
        params_normal = make_dist_params(case33, max_steps=48)
        obs_n, state_n = env.reset(key, params_normal)
        action = jnp.zeros(case33.n_nodes - 1)
        _, _, _, _, _, info_n = env.step(key, state_n, action, params_normal)

        # Create heavy load params (2x)
        heavy_p = params_normal.load_profiles_p * 2.0
        heavy_q = params_normal.load_profiles_q * 2.0
        params_heavy = params_normal.replace(
            load_profiles_p=heavy_p, load_profiles_q=heavy_q)
        obs_h, state_h = env.reset(key, params_heavy)
        _, _, _, _, _, info_h = env.step(key, state_h, action, params_heavy)

        assert float(info_h["cost_sum"]) >= float(info_n["cost_sum"]) - 1e-6

    def test_der_compensation_reduces_loss(self, env, case33, key):
        """DER injection at the lowest-voltage bus should reduce total losses."""
        # Use 2x load to create meaningful losses
        params = make_dist_params(case33, max_steps=48)
        heavy_p = params.load_profiles_p * 2.0
        heavy_q = params.load_profiles_q * 2.0
        params_h = params.replace(load_profiles_p=heavy_p, load_profiles_q=heavy_q)
        obs, state = env.reset(key, params_h)

        action_zero = jnp.zeros(case33.n_nodes - 1)
        _, state_z, _, _, _, info_z = env.step(key, state, action_zero, params_h)

        # Inject a small amount at the far-end bus (worst voltage)
        min_v_idx = int(jnp.argmin(state_z.v_mag))
        action_targeted = jnp.zeros(case33.n_nodes - 1)
        # Only inject at far-end node (non-slack index = bus_idx - 1)
        if min_v_idx > 0:
            action_targeted = action_targeted.at[min_v_idx - 1].set(0.02)
        _, state_i, _, _, _, info_i = env.step(key, state, action_targeted, params_h)

        # Targeted injection should reduce losses
        assert float(state_i.p_loss_total) < float(state_z.p_loss_total) + 1e-3


# ==================== L0: PQ bundle (compile / batch) ====================

class TestDistGridEnv_L0_PQBundle:

    def test_vmap_reset_step_pq_bundle(self, env, case33):
        """vmap(reset) + vmap(step) with enable_q_control=True — batch parallel contract."""
        bundle = make_battery_bundle(
            case33, bus_ids=[18, 25], power_mw=0.2, capacity_mwh=0.4,
            enable_q_control=True,
        )
        params = make_dist_params(case33, max_steps=48, resources=(bundle,))
        n_der = case33.n_nodes - 1
        act_dim = n_der + bundle.action_dim
        batch = 4
        keys = jax.random.split(jax.random.PRNGKey(7), batch)
        obs_b, state_b = jax.vmap(lambda k: env.reset(k, params))(keys)
        assert obs_b.shape == (batch, env.observation_space(params).shape[0])
        assert state_b.v_mag.shape == (batch, case33.n_nodes)
        actions = jnp.zeros((batch, act_dim))
        keys2 = jax.random.split(jax.random.PRNGKey(8), batch)
        obs2, state2, rew, costs, done, info = jax.vmap(
            lambda k, s, a: env.step(k, s, a, params)
        )(keys2, state_b, actions)
        assert obs2.shape[0] == batch
        assert rew.shape == (batch,)
        assert state2.resource_states[0].soc.shape == (batch, 2)
        assert costs.shape == (batch, 3)

    def test_jit_step_pq_bundle(self, env, case33, key):
        """Explicit jit closure over params — PQ bundle action shape must compile."""
        bundle = make_battery_bundle(
            case33, bus_ids=[18, 25], power_mw=0.2, capacity_mwh=0.4,
            enable_q_control=True,
        )
        params = make_dist_params(case33, max_steps=48, resources=(bundle,))
        n_der = case33.n_nodes - 1
        act_dim = n_der + bundle.action_dim
        obs, state = env.reset(key, params)
        action = jnp.zeros(act_dim)
        step_jit = jax.jit(lambda k, s, a: env.step(k, s, a, params))
        obs2, state2, rew, costs, done, info = step_jit(key, state, action)
        assert obs2.shape == obs.shape
        assert rew.shape == ()


# ==================== L1.5: Battery bundle + Q control (Dist) ====================

class TestDistGridEnv_BatteryBundleQ:

    def test_battery_bundle_q_injection_changes_voltage(self, env, case33, key):
        """P-only vs P+Q BESS at same P: voltage profile must differ if q_inj reaches BFS."""
        bundle_p = make_battery_bundle(
            case33, bus_ids=[18, 25], power_mw=0.2, capacity_mwh=0.4,
            enable_q_control=False,
        )
        bundle_pq = make_battery_bundle(
            case33, bus_ids=[18, 25], power_mw=0.2, capacity_mwh=0.4,
            enable_q_control=True,
        )
        params_p = make_dist_params(case33, max_steps=48, resources=(bundle_p,))
        params_pq = make_dist_params(case33, max_steps=48, resources=(bundle_pq,))
        n_der = case33.n_nodes - 1
        key_p, key_q = jax.random.split(key)
        _, state_p = env.reset(key_p, params_p)
        _, state_q = env.reset(key_q, params_pq)
        # Same P commands for both devices; PQ run adds substantial Q (device-major [P,Q,...])
        p_cmd = 0.75
        q_cmd = 0.85
        act_p = jnp.concatenate(
            [jnp.zeros(n_der), jnp.array([p_cmd, p_cmd], dtype=jnp.float32)])
        act_pq = jnp.concatenate(
            [jnp.zeros(n_der),
             jnp.array([p_cmd, q_cmd, p_cmd, q_cmd], dtype=jnp.float32)])
        _, state_p2, *_ = env.step(key_p, state_p, act_p, params_p)
        _, state_q2, *_ = env.step(key_q, state_q, act_pq, params_pq)
        assert not jnp.allclose(state_p2.v_mag, state_q2.v_mag, rtol=1e-5, atol=1e-5), (
            "Voltage profile should differ when Q injection is present; "
            "possible bug: q_inj ignored in BFS net injection"
        )

    def test_bess_bundle_obs_shape(self, env, case33, key):
        """enable_q_control=True → per-device obs is n×3; action_dim = 2 * n_devices."""
        n_dev = 2
        bundle = make_battery_bundle(
            case33, bus_ids=[18, 25], power_mw=0.2, capacity_mwh=0.4,
            enable_q_control=True,
        )
        assert bundle.per_device_obs_dim == 3
        assert bundle.action_dim == 2 * n_dev
        params = make_dist_params(case33, max_steps=48, resources=(bundle,))
        n = params.case.n_nodes
        m = params.topo.n_lines
        base = 3 * n + 2 * m + 2
        assert env.observation_space(params).shape == (base + n_dev * 3,)
        assert env.action_space(params).shape == ((n - 1) + 2 * n_dev,)
        obs, _ = env.reset(key, params)
        assert obs.shape == (base + n_dev * 3,)

    def test_single_device_pq_step(self, env, case33, key):
        """n_devices=1, enable_q_control=True: action slice [P,Q], reshape (1,2) path."""
        bundle = make_battery_bundle(
            case33, bus_ids=[18], power_mw=0.2, capacity_mwh=0.4,
            enable_q_control=True,
        )
        assert bundle.n_devices == 1
        assert bundle.action_dim == 2
        params = make_dist_params(case33, max_steps=48, resources=(bundle,))
        n_der = case33.n_nodes - 1
        obs, state = env.reset(key, params)
        action = jnp.concatenate(
            [jnp.zeros(n_der), jnp.array([0.5, 0.4], dtype=jnp.float32)])
        obs2, state2, _, _, _, info = env.step(key, state, action, params)
        assert obs2.shape == obs.shape
        assert state2.resource_states[0].soc.shape == (1,)
        assert info["bfs_converged"]


class TestDistGridEnv_IncludeDerFalseWithBundle:

    def test_include_der_false_with_bundle(self, env, case33, key):
        """include_der=False: no legacy DER slice; same bundle action as zeros DER + bundle."""
        bundle = make_battery_bundle(
            case33, bus_ids=[18, 25], power_mw=0.2, capacity_mwh=0.4,
            enable_q_control=False,
        )
        params_bundle_only = make_dist_params(
            case33, max_steps=48, resources=(bundle,), include_der=False)
        params_with_der = make_dist_params(
            case33, max_steps=48, resources=(bundle,), include_der=True)
        n_der = case33.n_nodes - 1
        assert env.action_space(params_bundle_only).shape == (bundle.action_dim,)
        b_act = jnp.array([0.6, 0.6], dtype=jnp.float32)
        act_only = b_act
        act_full = jnp.concatenate([jnp.zeros(n_der), b_act])
        k1, k2 = jax.random.split(key)
        _, s1 = env.reset(k1, params_bundle_only)
        _, s2 = env.reset(k2, params_with_der)
        obs1, st1, *_ = env.step(k1, s1, act_only, params_bundle_only)
        obs2, st2, *_ = env.step(k2, s2, act_full, params_with_der)
        np.testing.assert_allclose(
            np.asarray(obs1), np.asarray(obs2), rtol=1e-5, atol=1e-5,
            err_msg="Obs: bundle-only should match zeros DER + same bundle",
        )
        np.testing.assert_allclose(
            np.asarray(st1.v_mag), np.asarray(st2.v_mag), rtol=1e-5, atol=1e-5,
            err_msg="v_mag: bundle-only should match zeros DER + same bundle (offset=0)",
        )
        np.testing.assert_allclose(
            np.asarray(st1.resource_states[0].soc),
            np.asarray(st2.resource_states[0].soc),
            rtol=1e-5, atol=1e-5,
        )

    def test_include_der_false_with_pq_bundle(self, env, case33, key):
        """include_der=False + enable_q_control=True: action_dim=2*n_dev only; same as zero-DER + bundle."""
        bundle = make_battery_bundle(
            case33, bus_ids=[18, 25], power_mw=0.2, capacity_mwh=0.4,
            enable_q_control=True,
        )
        assert bundle.action_dim == 4
        params_only = make_dist_params(
            case33, max_steps=48, resources=(bundle,), include_der=False)
        params_full = make_dist_params(
            case33, max_steps=48, resources=(bundle,), include_der=True)
        n_der = case33.n_nodes - 1
        assert env.action_space(params_only).shape == (4,)
        b_act = jnp.array([0.4, 0.3, 0.5, 0.35], dtype=jnp.float32)
        k1, k2 = jax.random.split(key)
        _, s1 = env.reset(k1, params_only)
        _, s2 = env.reset(k2, params_full)
        obs1, st1, *_ = env.step(k1, s1, b_act, params_only)
        obs2, st2, *_ = env.step(
            k2, s2, jnp.concatenate([jnp.zeros(n_der), b_act]), params_full)
        np.testing.assert_allclose(np.asarray(obs1), np.asarray(obs2), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            np.asarray(st1.v_mag), np.asarray(st2.v_mag), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            np.asarray(st1.resource_states[0].soc),
            np.asarray(st2.resource_states[0].soc),
            rtol=1e-5, atol=1e-5,
        )
