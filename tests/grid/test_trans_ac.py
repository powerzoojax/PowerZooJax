"""Tests for TransGridEnv AC mode — L0 JAX + L1 env behavior + L1.5 cross-mode.

L0: JIT, vmap, scan rollout, auto_reset in AC mode.
L1: obs shape, reward structure, cost separation, voltage in obs.
L1.5: DC vs AC mode on same case5 — gen cost similar, flow directions agree.
"""

import jax
import jax.numpy as jnp
import jax.tree_util as tu
import numpy as np
import pytest

from powerzoojax.case import create_case5
from powerzoojax.envs.grid.trans import TransGridEnv, TransGridParams, make_trans_params


@pytest.fixture(scope="module")
def case5():
    return create_case5()


@pytest.fixture(scope="module")
def ac_params(case5):
    return make_trans_params(case5, physics=1, max_steps=48)


@pytest.fixture(scope="module")
def dc_params(case5):
    return make_trans_params(case5, physics=0, max_steps=48)


@pytest.fixture(scope="module")
def env():
    return TransGridEnv()


# ==================== L0: JAX Contract ====================

class TestTransAC_L0_JAX:
    """L0: JIT, vmap, scan, auto_reset in AC mode."""

    def test_jit_reset(self, env, ac_params):
        key = jax.random.PRNGKey(0)
        obs, state = jax.jit(env.reset, static_argnums=())(key, ac_params)
        assert obs.shape[0] > 0
        assert state.vm is not None

    def test_jit_step(self, env, ac_params, case5):
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, ac_params)
        action = jnp.zeros(case5.n_units)
        jit_step = jax.jit(lambda k, s, a: env.step(k, s, a, ac_params))
        obs2, state2, reward, costs, done, info = jit_step(
            jax.random.PRNGKey(1), state, action
        )
        assert obs2.shape == obs.shape
        assert state2.time_step == 1
        assert costs.shape == (4,)

    def test_vmap_step(self, env, ac_params, case5):
        n_envs = 4
        keys = jax.random.split(jax.random.PRNGKey(0), n_envs)
        batch_params = tu.tree_map(lambda x: jnp.stack([x] * n_envs), ac_params)

        batch_reset = jax.vmap(env.reset)
        obs_batch, state_batch = batch_reset(keys, batch_params)
        assert obs_batch.shape[0] == n_envs

        actions = jnp.tile(jnp.zeros(case5.n_units), (n_envs, 1))
        step_keys = jax.random.split(jax.random.PRNGKey(42), n_envs)
        batch_step = jax.vmap(lambda k, s, a, p: env.step(k, s, a, p))
        obs2, state2, rew, costs, done, info = batch_step(
            step_keys, state_batch, actions, batch_params)
        assert obs2.shape[0] == n_envs
        assert rew.shape == (n_envs,)
        assert costs.shape == (n_envs, 4)

    def test_scan_rollout(self, env, ac_params, case5):
        """lax.scan rollout of full episode."""
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, ac_params)
        action = jnp.zeros(case5.n_units)

        def scan_step(carry, _):
            key, state = carry
            key, subkey = jax.random.split(key)
            obs, state, reward, costs, done, info = env.step(
                subkey, state, action, ac_params
            )
            return (key, state), (reward, costs, done, obs)

        (_, final_state), (rewards, costs, dones, obses) = jax.lax.scan(
            scan_step, (jax.random.PRNGKey(1), state), None, length=48)
        assert rewards.shape == (48,)
        assert costs.shape == (48, 4)
        assert dones.shape == (48,)
        assert obses.shape[0] == 48

    def test_auto_reset(self, env, case5):
        """After done, state resets to t=0, obs matches reset obs."""
        short_params = make_trans_params(case5, physics=1, max_steps=2)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, short_params)
        action = jnp.zeros(case5.n_units)

        _, s1, _, _, d1, _ = env.step(jax.random.PRNGKey(1), state, action, short_params)
        assert not bool(d1)
        assert int(s1.time_step) == 1

        _, s2, _, _, d2, _ = env.step(jax.random.PRNGKey(2), s1, action, short_params)
        assert bool(d2)
        assert int(s2.time_step) == 0  # auto-reset

    def test_pytree_state(self, env, ac_params):
        key = jax.random.PRNGKey(0)
        _, state = env.reset(key, ac_params)
        leaves = tu.tree_leaves(state)
        assert len(leaves) > 0
        flat, td = tu.tree_flatten(state)
        state_rebuilt = tu.tree_unflatten(td, flat)
        assert int(state_rebuilt.time_step) == 0


# ==================== L1: Env Behavior ====================

class TestTransAC_L1_Behavior:
    """L1: Observation shape, reward, cost, voltage in obs."""

    def test_obs_shape(self, env, ac_params, case5):
        """AC obs includes voltage magnitudes."""
        key = jax.random.PRNGKey(0)
        obs, _ = env.reset(key, ac_params)
        expected_dim = case5.n_lines + case5.n_nodes + case5.n_loads + case5.n_units + 2
        assert obs.shape == (expected_dim,), f"Got {obs.shape}, expected ({expected_dim},)"

    def test_obs_space_match(self, env, ac_params):
        space = env.observation_space(ac_params)
        key = jax.random.PRNGKey(0)
        obs, _ = env.reset(key, ac_params)
        assert obs.shape == space.shape

    def test_reward_negative(self, env, ac_params, case5):
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, ac_params)
        action = jnp.zeros(case5.n_units)
        _, _, reward, _, _, _ = env.step(
            jax.random.PRNGKey(1), state, action, ac_params
        )
        assert float(reward) <= 0.0

    def test_cost_separation(self, env, ac_params, case5):
        """Named cost components and explicit vector costs are kept separate from reward."""
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, ac_params)
        action = jnp.zeros(case5.n_units)
        _, _, reward, costs, _, info = env.step(
            jax.random.PRNGKey(1), state, action, ac_params
        )
        assert env.constraint_names(ac_params) == (
            "thermal_overload",
            "voltage_violation",
            "power_balance",
            "resource",
        )
        assert "cost_thermal_overload" in info
        assert "cost_voltage_violation" in info
        assert "cost_power_balance" in info
        assert "cost_resource" in info
        assert "cost_sum" in info
        np.testing.assert_allclose(
            np.asarray(costs),
            np.asarray(
                [
                    info["cost_thermal_overload"],
                    info["cost_voltage_violation"],
                    info["cost_power_balance"],
                    info["cost_resource"],
                ]
            ),
            atol=1e-6,
        )
        assert float(info["cost_sum"]) >= 0.0

    def test_vm_in_state(self, env, ac_params, case5):
        """AC mode should populate vm in state with non-trivial values."""
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, ac_params)
        action = jnp.ones(case5.n_units) * 0.5
        _, state2, _, _, _, _ = env.step(
            jax.random.PRNGKey(1), state, action, ac_params
        )
        assert not jnp.allclose(state2.vm, 1.0, atol=1e-3), \
            "AC mode vm should differ from flat 1.0"


# ==================== L1.5: DC vs AC Cross-mode ====================

class TestTransAC_L15_CrossMode:
    """L1.5: DC vs AC on same case5 — consistency checks."""

    def test_gen_cost_similar(self, env, dc_params, ac_params, case5):
        """Same dispatch should produce similar gen cost in DC and AC modes."""
        key = jax.random.PRNGKey(0)
        action = jnp.zeros(case5.n_units)

        _, dc_state = env.reset(key, dc_params)
        _, _, _, _, _, dc_info = env.step(
            jax.random.PRNGKey(1), dc_state, action, dc_params
        )

        _, ac_state = env.reset(key, ac_params)
        _, _, _, _, _, ac_info = env.step(
            jax.random.PRNGKey(1), ac_state, action, ac_params
        )

        dc_cost = float(dc_info["gen_cost"])
        ac_cost = float(ac_info["gen_cost"])
        assert abs(dc_cost - ac_cost) < 1.0, f"DC={dc_cost}, AC={ac_cost}"

    def test_flow_direction_agree(self, env, dc_params, ac_params, case5):
        """DC and AC line flows should have the same sign (same direction)."""
        key = jax.random.PRNGKey(0)
        action = jnp.zeros(case5.n_units)

        _, dc_state = env.reset(key, dc_params)
        _, dc_s, _, _, _, _ = env.step(
            jax.random.PRNGKey(1), dc_state, action, dc_params
        )

        _, ac_state = env.reset(key, ac_params)
        _, ac_s, _, _, _, _ = env.step(
            jax.random.PRNGKey(1), ac_state, action, ac_params
        )

        dc_sign = jnp.sign(dc_s.line_flow_mw)
        ac_sign = jnp.sign(ac_s.line_flow_mw)
        # At least 4 of 6 lines should agree on direction
        agree = jnp.sum(dc_sign == ac_sign)
        assert int(agree) >= 4, f"Only {agree}/6 flow directions agree"
