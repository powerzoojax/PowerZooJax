"""GPU pipeline tests for TransGridEnv — step_auto_reset, scan_rollout, vmap+scan."""

import jax
import jax.numpy as jnp
import pytest

from powerzoojax.case import create_case5
from powerzoojax.envs.grid.trans import TransGridEnv, make_trans_params
from powerzoojax.utils.jax_utils import batch_reset, batch_step, scan_rollout


@pytest.fixture(scope="module")
def case5():
    return create_case5()


@pytest.fixture(scope="module")
def env():
    return TransGridEnv()


@pytest.fixture(scope="module")
def params(case5):
    return make_trans_params(case5, max_steps=8)


# ==================== step_auto_reset ====================

class TestStepAutoReset:
    def test_auto_reset_resets_on_done(self, env, case5):
        """When done, step_auto_reset returns reset state."""
        short_params = make_trans_params(case5, max_steps=2)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, short_params)
        action = jnp.zeros(case5.n_units)

        # Step to done
        k1, k2, k3 = jax.random.split(key, 3)
        obs, state, _, _costs, _, _ = env.step(k1, state, action, short_params)
        obs, state, _, _costs, done, _ = env.step_auto_reset(k2, state, action, short_params)
        # After auto-reset, done should be True (from terminal step),
        # but state should be reset
        assert bool(done)
        assert int(state.time_step) == 0

    def test_auto_reset_no_reset_before_done(self, env, params, case5):
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)
        action = jnp.zeros(case5.n_units)
        obs, state, _, _costs, done, _ = env.step_auto_reset(
            jax.random.PRNGKey(1), state, action, params
        )
        assert not bool(done)
        assert int(state.time_step) == 1

    def test_jit(self, env, params, case5):
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)
        action = jnp.zeros(case5.n_units)
        jit_fn = jax.jit(lambda k, s, a: env.step_auto_reset(k, s, a, params))
        obs2, state2, reward, costs, done, info = jit_fn(jax.random.PRNGKey(1), state, action)
        assert state2.time_step == 1


# ==================== scan_rollout ====================

class TestScanRollout:
    def test_scan_basic(self, env, params, case5):
        """scan_rollout should collect T steps of trajectory."""
        T = 8
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)
        actions = jnp.tile(jnp.zeros(case5.n_units), (T, 1))
        final_state, obs_traj, reward_traj, cost_traj, done_traj, info_traj = scan_rollout(
            env, jax.random.PRNGKey(1), state, params, actions
        )
        assert obs_traj.shape[0] == T
        assert reward_traj.shape == (T,)
        assert cost_traj.shape[0] == T
        assert done_traj.shape == (T,)

    def test_scan_episode_boundary(self, env, case5):
        """Scan should cross episode boundary via auto-reset."""
        short_params = make_trans_params(case5, max_steps=3)
        T = 10  # longer than one episode
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, short_params)
        actions = jnp.tile(jnp.zeros(case5.n_units), (T, 1))
        final_state, obs_traj, reward_traj, cost_traj, done_traj, info_traj = scan_rollout(
            env, jax.random.PRNGKey(1), state, short_params, actions
        )
        # Should have at least one done=True in the trajectory
        assert jnp.any(done_traj)
        assert obs_traj.shape[0] == T

    def test_scan_jit(self, env, params, case5):
        T = 4
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)
        actions = jnp.tile(jnp.zeros(case5.n_units), (T, 1))
        jit_scan = jax.jit(
            lambda s, a, k: scan_rollout(env, k, s, params, a)
        )
        final_state, obs_traj, reward_traj, cost_traj, done_traj, info_traj = jit_scan(
            state, actions, jax.random.PRNGKey(1)
        )
        assert obs_traj.shape[0] == T


# ==================== batch (vmap) ====================

class TestBatch:
    def test_batch_reset(self, env, params):
        n_envs = 4
        keys = jax.random.split(jax.random.PRNGKey(0), n_envs)
        obs, states = batch_reset(env, keys, params)
        assert obs.shape[0] == n_envs
        assert states.time_step.shape == (n_envs,)

    def test_batch_step(self, env, params, case5):
        n_envs = 4
        keys = jax.random.split(jax.random.PRNGKey(0), n_envs)
        obs, states = batch_reset(env, keys, params)
        actions = jnp.tile(jnp.zeros(case5.n_units), (n_envs, 1))
        step_keys = jax.random.split(jax.random.PRNGKey(1), n_envs)
        obs2, states2, reward, costs, done, info = batch_step(
            env, step_keys, states, actions, params
        )
        assert obs2.shape[0] == n_envs
        assert reward.shape == (n_envs,)
        assert costs.shape[0] == n_envs


# ==================== vmap + scan combo ====================

class TestVmapScanCombo:
    def test_vmap_scan(self, env, params, case5):
        """vmap over multiple environments, each running scan_rollout."""
        n_envs = 3
        T = 4
        keys = jax.random.split(jax.random.PRNGKey(0), n_envs)
        obs_batch, state_batch = batch_reset(env, keys, params)
        actions = jnp.tile(jnp.zeros(case5.n_units), (T, 1))

        def single_rollout(state, key):
            return scan_rollout(env, key, state, params, actions)

        rollout_keys = jax.random.split(jax.random.PRNGKey(1), n_envs)
        final_states, obs_trajs, reward_trajs, cost_trajs, done_trajs, info_trajs = jax.vmap(
            single_rollout
        )(state_batch, rollout_keys)
        assert obs_trajs.shape == (n_envs, T, obs_batch.shape[-1])
        assert reward_trajs.shape == (n_envs, T)
        assert cost_trajs.shape[:2] == (n_envs, T)
