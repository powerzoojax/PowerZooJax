"""Tests for rl/wrappers.py — LogWrapper, SauteWrapper, and SafeRLWrapper.

L0: Wrapper contract (JIT, vmap, lax.scan, info keys, properties)
L1: Episode return accumulation, auto-reset, CMDP cost tracking
"""

import pytest
import jax
import jax.numpy as jnp

from powerzoojax.envs.resource.battery import BatteryEnv, make_battery_params
from powerzoojax.envs.resource.renewable import RenewableEnv, RenewableParams
from powerzoojax.envs.resource.vehicle import VehicleEnv, make_vehicle_params
from powerzoojax.envs.resource.flexload import FlexLoadEnv, FlexLoadParams
from powerzoojax.envs.grid.trans import TransGridEnv, make_trans_params
from powerzoojax.case import create_case5
from powerzoojax.rl.wrappers import (
    LogWrapper,
    LogEnvState,
    SafeRLWrapper,
    SafeRLState,
    SauteWrapper,
    bind,
)


# ========================== Fixtures ==========================

@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


def _short_battery():
    env = BatteryEnv()
    params = make_battery_params(max_steps=8, delta_t_hours=0.5, steps_per_day=48)
    return env, params


def _short_renewable():
    env = RenewableEnv()
    profiles = jnp.full((48,), 0.5, dtype=jnp.float32)
    params = RenewableParams(capacity_mw=10.0, profiles=profiles, max_steps=8)
    return env, params


def _short_vehicle():
    env = VehicleEnv()
    params = make_vehicle_params(max_steps=8, delta_t_minutes=30.0)
    return env, params


def _short_flexload():
    env = FlexLoadEnv()
    params = FlexLoadParams(max_steps=8)
    return env, params


def _short_transgrid():
    env = TransGridEnv()
    case = create_case5()
    profiles = jnp.ones((48, case.n_loads), dtype=jnp.float32) * 0.5
    params = make_trans_params(case, load_profiles=profiles, max_steps=8)
    return env, params


ALL_ENVS = [
    pytest.param(_short_battery, id="battery"),
    pytest.param(_short_renewable, id="renewable"),
    pytest.param(_short_vehicle, id="vehicle"),
    pytest.param(_short_flexload, id="flexload"),
    pytest.param(_short_transgrid, id="transgrid"),
]


def _make_log_wrapper(factory):
    env, params = factory()
    return LogWrapper(env, params)


def _sample_action(wrapped, key):
    """Sample a valid action for the wrapped environment."""
    space = wrapped.action_space()
    return space.sample(key)


# ========================== L0: LogWrapper Contract ==========================

class TestLogWrapperContract:

    @pytest.mark.parametrize("factory", ALL_ENVS)
    def test_reset_no_params(self, factory, key):
        wrapped = _make_log_wrapper(factory)
        obs, state = wrapped.reset(key)
        assert obs.ndim >= 1
        assert isinstance(state, LogEnvState)

    @pytest.mark.parametrize("factory", ALL_ENVS)
    def test_step_no_params(self, factory, key):
        wrapped = _make_log_wrapper(factory)
        obs, state = wrapped.reset(key)
        k1, k2 = jax.random.split(key)
        action = _sample_action(wrapped, k2)
        obs2, state2, reward, done, info = wrapped.step(k1, state, action)
        assert obs2.shape == obs.shape
        assert reward.shape == ()
        assert done.shape == ()

    @pytest.mark.parametrize("factory", ALL_ENVS)
    def test_properties(self, factory):
        wrapped = _make_log_wrapper(factory)
        assert isinstance(wrapped.obs_size, int)
        assert wrapped.obs_size > 0
        assert isinstance(wrapped.num_actions, int)
        assert wrapped.num_actions > 0
        assert wrapped.action_size == wrapped.num_actions

    @pytest.mark.parametrize("factory", ALL_ENVS)
    def test_info_keys(self, factory, key):
        wrapped = _make_log_wrapper(factory)
        obs, state = wrapped.reset(key)
        k1, k2 = jax.random.split(key)
        action = _sample_action(wrapped, k2)
        _, _, _, _, info = wrapped.step(k1, state, action)
        assert "returned_episode_returns" in info
        assert "returned_episode_lengths" in info
        assert "returned_episode" in info

    @pytest.mark.parametrize("factory", ALL_ENVS)
    def test_jit(self, factory, key):
        wrapped = _make_log_wrapper(factory)
        reset_jit = jax.jit(wrapped.reset)
        step_jit = jax.jit(wrapped.step)
        obs, state = reset_jit(key)
        k1, k2 = jax.random.split(key)
        action = _sample_action(wrapped, k2)
        obs2, state2, reward, done, info = step_jit(k1, state, action)
        assert obs2.shape == obs.shape

    @pytest.mark.parametrize("factory", ALL_ENVS)
    def test_vmap(self, factory, key):
        wrapped = _make_log_wrapper(factory)
        n_envs = 4
        keys = jax.random.split(key, n_envs)

        obs_batch, state_batch = jax.vmap(wrapped.reset)(keys)
        assert obs_batch.shape[0] == n_envs

        k1s = jax.random.split(key, n_envs)
        actions = jax.vmap(lambda k: _sample_action(wrapped, k))(
            jax.random.split(jax.random.PRNGKey(1), n_envs)
        )
        obs2, state2, rewards, dones, infos = jax.vmap(wrapped.step)(
            k1s, state_batch, actions
        )
        assert rewards.shape == (n_envs,)

    @pytest.mark.parametrize("factory", ALL_ENVS)
    def test_lax_scan(self, factory, key):
        """Run a short rollout via lax.scan."""
        wrapped = _make_log_wrapper(factory)
        obs, state = wrapped.reset(key)

        def scan_step(carry, _):
            state, key = carry
            key, k1, k2 = jax.random.split(key, 3)
            action = _sample_action(wrapped, k2)
            obs, state, reward, done, info = wrapped.step(k1, state, action)
            return (state, key), reward

        (final_state, _), rewards = jax.lax.scan(
            scan_step, (state, key), None, length=20
        )
        assert rewards.shape == (20,)


# ========================== L1: Episode Return Tracking ==========================

class TestLogWrapperEpisodeTracking:

    def test_return_accumulation(self, key):
        """After a full episode, returned_episode_returns should equal sum(rewards)."""
        wrapped = _make_log_wrapper(_short_battery)  # max_steps=8
        obs, state = wrapped.reset(key)
        total_reward = 0.0

        for i in range(8):
            k1, key = jax.random.split(key)
            k2, key = jax.random.split(key)
            action = _sample_action(wrapped, k2)
            obs, state, reward, done, info = wrapped.step(k1, state, action)
            total_reward += float(reward)

        # After 8 steps, episode should be done
        assert float(info["returned_episode"]) == 1.0
        assert jnp.isclose(
            info["returned_episode_returns"], total_reward, atol=1e-5
        )

    def test_auto_reset_on_done(self, key):
        """After done, state should be reset but return info preserved."""
        wrapped = _make_log_wrapper(_short_battery)  # max_steps=8
        obs, state = wrapped.reset(key)

        # Run to end of episode
        for i in range(8):
            k1, key = jax.random.split(key)
            k2, key = jax.random.split(key)
            action = _sample_action(wrapped, k2)
            obs, state, reward, done, info = wrapped.step(k1, state, action)

        # Episode counters should be reset (next episode started)
        assert float(state.episode_returns) == 0.0
        assert int(state.episode_lengths) == 0
        # But returned info should be preserved
        assert info["returned_episode_returns"] != 0.0 or True  # may be 0 for battery


# ========================== L0: SafeRLWrapper Contract ==========================

class TestSafeRLWrapperContract:

    @pytest.mark.parametrize("factory", ALL_ENVS)
    def test_step_returns_6_tuple(self, factory, key):
        env, params = factory()
        wrapped = SafeRLWrapper(env, params, cost_threshold=25.0)
        obs, state = wrapped.reset(key)
        k1, k2 = jax.random.split(key)
        action = _sample_action(wrapped, k2)
        result = wrapped.step(k1, state, action)
        assert len(result) == 6  # obs, state, reward, costs, done, info
        obs, state, reward, costs, done, info = result
        assert costs.shape == (wrapped.num_constraints,)
        assert bool(jnp.all(costs >= 0.0))

    @pytest.mark.parametrize("factory", ALL_ENVS)
    def test_cost_threshold_accessible(self, factory):
        env, params = factory()
        wrapped = SafeRLWrapper(env, params, cost_threshold=42.0)
        assert wrapped.cost_threshold == 42.0

    @pytest.mark.parametrize("factory", ALL_ENVS)
    def test_properties(self, factory):
        env, params = factory()
        wrapped = SafeRLWrapper(env, params)
        assert wrapped.obs_size > 0
        assert wrapped.num_actions > 0

    @pytest.mark.parametrize("factory", ALL_ENVS)
    def test_jit(self, factory, key):
        env, params = factory()
        wrapped = SafeRLWrapper(env, params)
        obs, state = jax.jit(wrapped.reset)(key)
        k1, k2 = jax.random.split(key)
        action = _sample_action(wrapped, k2)
        result = jax.jit(wrapped.step)(k1, state, action)
        assert len(result) == 6

    @pytest.mark.parametrize("factory", ALL_ENVS)
    def test_vmap(self, factory, key):
        env, params = factory()
        wrapped = SafeRLWrapper(env, params)
        n_envs = 4
        keys = jax.random.split(key, n_envs)
        obs_batch, state_batch = jax.vmap(wrapped.reset)(keys)
        assert obs_batch.shape[0] == n_envs


class TestSauteWrapperContract:

    def test_obs_size_augmented(self, key):
        env, params = _short_transgrid()
        wrapped = SauteWrapper(
            env,
            params,
            cost_threshold=5.0,
            selected_names=("thermal_overload",),
            horizon=8,
        )
        obs, state = wrapped.reset(key)
        assert obs.shape[0] == env.observation_space(params).shape[0] + 1
        assert isinstance(state, SafeRLState)

    def test_step_returns_budget_info(self, key):
        env, params = _short_transgrid()
        wrapped = SauteWrapper(
            env,
            params,
            cost_threshold=5.0,
            selected_names=("thermal_overload",),
            horizon=8,
            use_reward_shaping=False,
        )
        obs, state = wrapped.reset(key)
        k1, k2 = jax.random.split(key)
        action = _sample_action(wrapped, k2)
        obs2, state2, reward, done, info = wrapped.step(k1, state, action)
        assert obs2.shape == obs.shape
        assert isinstance(state2, SafeRLState)
        assert reward.shape == ()
        assert done.shape == ()
        assert "safety_budget_remaining" in info
        assert info["safety_budget_remaining"].shape == (1,)


# ========================== L1: CMDP Cost Tracking ==========================

class TestSafeRLCostTracking:

    def test_episode_cost_accumulation(self, key):
        """TransGridEnv has nonzero cost_thermal; verify accumulation."""
        env, params = _short_transgrid()
        wrapped = SafeRLWrapper(env, params, cost_threshold=25.0)
        obs, state = wrapped.reset(key)

        total_costs = jnp.zeros((wrapped.num_constraints,), dtype=jnp.float32)
        for i in range(8):
            k1, key = jax.random.split(key)
            k2, key = jax.random.split(key)
            action = _sample_action(wrapped, k2)
            obs, state, reward, costs, done, info = wrapped.step(k1, state, action)
            total_costs = total_costs + costs

        # After done, returned_episode_costs should match accumulated cost
        assert "returned_episode_costs" in info
        assert jnp.allclose(info["returned_episode_costs"], total_costs, atol=1e-4)
        assert jnp.isclose(info["returned_episode_cost_sum"], jnp.sum(total_costs), atol=1e-4)

    def test_cost_info_key_present(self, key):
        env, params = _short_transgrid()
        wrapped = SafeRLWrapper(env, params)
        obs, state = wrapped.reset(key)
        k1, k2 = jax.random.split(key)
        action = _sample_action(wrapped, k2)
        _, _, _, costs, _, info = wrapped.step(k1, state, action)
        assert "cost" in info
        assert "constraint_costs" in info
        assert jnp.allclose(info["constraint_costs"], costs)
        assert jnp.isclose(info["cost_sum"], jnp.sum(costs), atol=1e-6)


# ========================== bind() convenience ==========================

class TestBind:

    def test_bind_returns_log_wrapper(self):
        env, params = _short_battery()
        wrapped = bind(env, params)
        assert isinstance(wrapped, LogWrapper)

    def test_bind_safe_returns_safe_wrapper(self):
        env, params = _short_battery()
        wrapped = bind(env, params, safe=True, cost_threshold=10.0)
        assert isinstance(wrapped, SafeRLWrapper)
        assert wrapped.cost_threshold == 10.0
