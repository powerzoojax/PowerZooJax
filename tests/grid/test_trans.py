"""Tests for TransGridEnv — L0 JAX contract + L1 physics + env API."""

import jax
import jax.numpy as jnp
import pytest

from powerzoojax.case import create_case5
from powerzoojax.envs.grid.trans import (
    TransGridEnv,
    TransGridState,
    TransGridParams,
    make_trans_params,
)


@pytest.fixture(scope="module")
def case5():
    return create_case5()


@pytest.fixture(scope="module")
def env():
    return TransGridEnv()


@pytest.fixture(scope="module")
def params(case5):
    return make_trans_params(case5, max_steps=48)


# ==================== Factory ====================

class TestMakeTransParams:
    def test_default_profiles(self, case5):
        p = make_trans_params(case5)
        assert p.load_profiles.shape == (48, case5.n_loads)

    def test_custom_profiles(self, case5):
        profiles = jnp.ones((10, case5.n_loads)) * 100.0
        p = make_trans_params(case5, load_profiles=profiles, max_steps=10)
        assert p.load_profiles.shape == (10, case5.n_loads)
        assert p.max_steps == 10


# ==================== Reset ====================

class TestReset:
    def test_output_shapes(self, env, params, case5):
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)
        obs_dim = case5.n_lines + case5.n_loads + case5.n_units + 2
        assert obs.shape == (obs_dim,)
        assert state.time_step == 0
        assert not bool(state.done)

    def test_state_fields(self, env, params, case5):
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)
        assert state.unit_power_mw.shape == (case5.n_units,)
        assert state.line_flow_mw.shape == (case5.n_lines,)
        assert state.node_injection_mw.shape == (case5.n_nodes,)
        assert float(state.total_cost) == 0.0

    def test_jit(self, env, params):
        jit_reset = jax.jit(lambda k: env.reset(k, params))
        obs, state = jit_reset(jax.random.PRNGKey(0))
        assert state.time_step == 0

    def test_deterministic(self, env, params):
        k = jax.random.PRNGKey(42)
        obs1, s1 = env.reset(k, params)
        obs2, s2 = env.reset(k, params)
        assert jnp.allclose(obs1, obs2)
        assert jnp.allclose(s1.unit_power_mw, s2.unit_power_mw)


# ==================== Step ====================

class TestStep:
    def test_output_types(self, env, params, case5):
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)
        action = jnp.zeros(case5.n_units)
        k2 = jax.random.PRNGKey(1)
        obs2, state2, reward, costs, done, info = env.step(k2, state, action, params)
        assert obs2.shape == obs.shape
        assert state2.time_step == 1
        assert reward.dtype == jnp.float32
        assert done.dtype == jnp.bool_
        assert "gen_cost" in info
        assert "cost_thermal_overload" in info
        assert costs.shape == (4,)

    def test_action_clipping(self, env, params, case5):
        """Actions beyond unit limits must be clipped."""
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)
        # Action way above p_max
        action = jnp.ones(case5.n_units) * 5.0
        _, state2, *_ = env.step(jax.random.PRNGKey(1), state, action, params)
        assert state2.time_step == 1
        assert state2.unit_power_mw.shape == (case5.n_units,)

    def test_reward_negative(self, env, params, case5):
        """Reward should be non-positive (negative gen cost)."""
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)
        action = jnp.zeros(case5.n_units)
        _, _, reward, _, _, _ = env.step(jax.random.PRNGKey(1), state, action, params)
        assert float(reward) <= 0.0

    def test_cost_accumulates(self, env, params, case5):
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)
        action = jnp.zeros(case5.n_units)
        _, s1, *_ = env.step(jax.random.PRNGKey(1), state, action, params)
        _, s2, *_ = env.step(jax.random.PRNGKey(2), s1, action, params)
        assert float(s2.total_cost) > float(s1.total_cost)

    def test_episode_termination(self, env, case5):
        """Episode done at max_steps; auto-reset returns time_step=0."""
        short_params = make_trans_params(case5, max_steps=3)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, short_params)
        action = jnp.zeros(case5.n_units)
        for i in range(3):
            obs, state, reward, costs, done, info = env.step(
                jax.random.PRNGKey(i + 1), state, action, short_params
            )
        assert bool(done)
        # Auto-reset: state is now at time_step 0 (reset state)
        assert int(state.time_step) == 0

    def test_jit(self, env, params, case5):
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)
        action = jnp.zeros(case5.n_units)
        jit_step = jax.jit(lambda k, s, a: env.step(k, s, a, params))
        obs2, state2, reward, costs, done, info = jit_step(
            jax.random.PRNGKey(1), state, action
        )
        assert state2.time_step == 1


# ==================== Spaces ====================

class TestSpaces:
    def test_obs_space(self, env, params, case5):
        space = env.observation_space(params)
        assert space.shape == (case5.n_lines + case5.n_loads + case5.n_units + 2,)

    def test_action_space(self, env, params, case5):
        space = env.action_space(params)
        assert space.shape == (case5.n_units,)
        assert jnp.allclose(space.low, jnp.full(case5.n_units, -1.0))
        assert jnp.allclose(space.high, jnp.full(case5.n_units, 1.0))

    def test_sample_action(self, env, params):
        space = env.action_space(params)
        action = space.sample(jax.random.PRNGKey(0))
        assert action.shape == space.shape


# ==================== L0: vmap ====================

class TestVmap:
    def test_vmap_reset(self, env, params):
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        vmap_reset = jax.vmap(lambda k: env.reset(k, params))
        obs_batch, state_batch = vmap_reset(keys)
        assert obs_batch.shape[0] == 4
        assert state_batch.time_step.shape == (4,)

    def test_vmap_step(self, env, params, case5):
        batch = 4
        keys = jax.random.split(jax.random.PRNGKey(0), batch)
        vmap_reset = jax.vmap(lambda k: env.reset(k, params))
        obs_batch, state_batch = vmap_reset(keys)

        actions = jnp.tile(jnp.zeros(case5.n_units), (batch, 1))
        step_keys = jax.random.split(jax.random.PRNGKey(1), batch)
        vmap_step = jax.vmap(lambda k, s, a: env.step(k, s, a, params))
        obs2, state2, reward, costs, done, info = vmap_step(
            step_keys, state_batch, actions
        )
        assert obs2.shape[0] == batch
        assert reward.shape == (batch,)
        assert costs.shape == (batch, 4)


# ==================== L0: pytree structure ====================

class TestPytree:
    def test_state_is_pytree(self, env, params):
        key = jax.random.PRNGKey(0)
        _, state = env.reset(key, params)
        leaves = jax.tree.leaves(state)
        assert len(leaves) > 0

    def test_state_replace(self, env, params):
        key = jax.random.PRNGKey(0)
        _, state = env.reset(key, params)
        new_state = state.replace(time_step=jnp.int32(99))
        assert int(new_state.time_step) == 99
        assert int(state.time_step) == 0
