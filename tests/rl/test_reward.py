"""Tests for RewardWrapper.

Covers:
    L0 JAX Contract: JIT compilation, vmap batching, lax.scan rollout
    L1 Functional: reward_fn substitution, episode_returns with custom reward,
                   info fields, safe mode (6-tuple) pass-through
"""

import jax
import jax.numpy as jnp
import pytest

from powerzoojax.envs.resource.battery import BatteryEnv, make_battery_params
from powerzoojax.envs.grid.trans import TransGridEnv, make_trans_params
from powerzoojax.case import create_case5
from powerzoojax.rl.wrappers import LogWrapper, SafeRLWrapper
from powerzoojax.rl.reward import RewardWrapper, RewardEnvState


# ============ Fixtures ============

@pytest.fixture
def battery_log_env():
    return LogWrapper(BatteryEnv(), make_battery_params(max_steps=48))


@pytest.fixture
def battery_reward_env(battery_log_env):
    reward_fn = lambda o, a, no, r, i: -jnp.abs(no[0] - 0.5)
    return RewardWrapper(battery_log_env, reward_fn)


@pytest.fixture
def safe_env():
    case = create_case5()
    profiles = jnp.ones((48, case.n_loads), dtype=jnp.float32) * 0.5
    params = make_trans_params(case, load_profiles=profiles, max_steps=48)
    return SafeRLWrapper(TransGridEnv(), params, cost_threshold=5.0)


@pytest.fixture
def safe_reward_env(safe_env):
    reward_fn = lambda o, a, no, r, i: r * 2.0
    return RewardWrapper(safe_env, reward_fn)


# ============ L0: JAX Contract ============

class TestRewardWrapperL0:
    """L0 JAX contract: JIT, vmap, scan."""

    def test_reset_jit(self, battery_reward_env):
        key = jax.random.PRNGKey(0)
        obs, state = jax.jit(battery_reward_env.reset)(key)
        assert obs.shape == (battery_reward_env.obs_size,)
        assert isinstance(state, RewardEnvState)

    def test_step_jit(self, battery_reward_env):
        key = jax.random.PRNGKey(0)
        obs, state = battery_reward_env.reset(key)
        k1, k2 = jax.random.split(key)
        action = battery_reward_env.action_space().sample(k2)
        result = jax.jit(battery_reward_env.step)(k1, state, action)
        assert len(result) == 5  # 5-tuple for non-safe

    def test_step_jit_safe(self, safe_reward_env):
        key = jax.random.PRNGKey(0)
        obs, state = safe_reward_env.reset(key)
        k1, k2 = jax.random.split(key)
        action = safe_reward_env.action_space().sample(k2)
        result = jax.jit(safe_reward_env.step)(k1, state, action)
        assert len(result) == 6  # 6-tuple for safe

    def test_pytree_structure_stable(self, battery_reward_env):
        """RewardEnvState pytree structure must be stable across steps."""
        key = jax.random.PRNGKey(0)
        obs, state = battery_reward_env.reset(key)
        tree0 = jax.tree.structure(state)

        k1, k2 = jax.random.split(key)
        action = battery_reward_env.action_space().sample(k2)
        _, state2, *_ = battery_reward_env.step(k1, state, action)
        tree1 = jax.tree.structure(state2)

        assert tree0 == tree1

    def test_vmap_reset(self, battery_reward_env):
        keys = jax.random.split(jax.random.PRNGKey(0), 8)
        obs_batch, state_batch = jax.vmap(battery_reward_env.reset)(keys)
        assert obs_batch.shape == (8, battery_reward_env.obs_size)

    def test_vmap_step(self, battery_reward_env):
        n = 8
        keys = jax.random.split(jax.random.PRNGKey(0), n)
        obs_batch, state_batch = jax.vmap(battery_reward_env.reset)(keys)

        step_keys = jax.random.split(jax.random.PRNGKey(1), n)
        actions = jax.vmap(lambda k: battery_reward_env.action_space().sample(k))(
            jax.random.split(jax.random.PRNGKey(2), n)
        )
        results = jax.vmap(battery_reward_env.step)(step_keys, state_batch, actions)
        obs_next, state_next, reward, done, info = results
        assert obs_next.shape == (n, battery_reward_env.obs_size)
        assert reward.shape == (n,)

    def test_scan_rollout(self, battery_reward_env):
        """lax.scan rollout must compile and run without error."""
        n_envs = 4
        n_steps = 48

        @jax.jit
        def rollout(key):
            env_keys = jax.random.split(key, n_envs)
            obs_batch, state_batch = jax.vmap(battery_reward_env.reset)(env_keys)

            def scan_step(carry, _):
                state, obs, key = carry
                key, k1, k2 = jax.random.split(key, 3)
                actions = jax.vmap(
                    lambda k: battery_reward_env.action_space().sample(k)
                )(jax.random.split(k2, n_envs))
                step_keys = jax.random.split(k1, n_envs)
                next_obs, next_state, reward, done, info = jax.vmap(
                    battery_reward_env.step
                )(step_keys, state, actions)
                return (next_state, next_obs, key), reward

            (final_state, final_obs, _), rewards = jax.lax.scan(
                scan_step, (state_batch, obs_batch, key), None, length=n_steps
            )
            return rewards

        rewards = rollout(jax.random.PRNGKey(0))
        assert rewards.shape == (n_steps, n_envs)


# ============ L1: Functional Correctness ============

class TestRewardWrapperL1:
    """L1 functional correctness: reward substitution, episode tracking, info fields."""

    def test_reward_fn_replaces_original(self, battery_log_env):
        """Custom reward should replace (not add to) the original reward."""
        # Constant reward function for easy comparison
        const_reward = 99.0
        env = RewardWrapper(battery_log_env, lambda o, a, no, r, i: jnp.float32(const_reward))
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)
        k1, k2 = jax.random.split(key)
        action = env.action_space().sample(k2)
        _, _, reward, done, info = env.step(k1, state, action)
        assert float(reward) == pytest.approx(const_reward)

    def test_reward_fn_uses_obs_transition(self, battery_log_env):
        """reward_fn receives prev_obs (before step) and next_obs (after step)."""
        # SOC-based reward: should differ from original reward
        import jax.numpy as jnp
        soc_reward_fn = lambda o, a, no, r, i: -jnp.abs(no[0] - 0.5)
        env = RewardWrapper(battery_log_env, soc_reward_fn)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)
        k1, k2 = jax.random.split(key)
        action = env.action_space().sample(k2)
        _, _, reward, done, info = env.step(k1, state, action)
        # reward must be in [-0.5, 0] (SOC in [0,1])
        assert float(reward) >= -0.5
        assert float(reward) <= 0.0

    def test_episode_returns_uses_custom_reward(self, battery_log_env):
        """Episode tracking in RewardEnvState should use the custom reward, not original."""
        const_custom = 5.0
        env = RewardWrapper(battery_log_env, lambda o, a, no, r, i: jnp.float32(const_custom))
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)
        assert float(state.episode_returns) == 0.0

        k1, k2 = jax.random.split(key)
        action = env.action_space().sample(k2)
        _, state2, reward, done, info = env.step(k1, state, action)

        if not bool(done):
            # episode not done: running accumulation
            expected = const_custom
            assert float(state2.episode_returns) == pytest.approx(expected, abs=1e-4)

    def test_returned_episode_returns_on_done(self, battery_log_env):
        """returned_episode_returns must update when episode ends."""
        const_custom = 1.0
        env = RewardWrapper(battery_log_env, lambda o, a, no, r, i: jnp.float32(const_custom))
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        # Run until done
        for i in range(50):
            key, k1, k2 = jax.random.split(key, 3)
            action = env.action_space().sample(k2)
            obs, state, reward, done, info = env.step(k1, state, action)
            if bool(done):
                assert float(state.returned_episode_returns) > 0.0
                break

    def test_info_contains_episode_keys(self, battery_reward_env):
        """info must contain returned_episode_returns, returned_episode_lengths, returned_episode."""
        key = jax.random.PRNGKey(0)
        obs, state = battery_reward_env.reset(key)
        k1, k2 = jax.random.split(key)
        action = battery_reward_env.action_space().sample(k2)
        _, _, _, _, info = battery_reward_env.step(k1, state, action)
        assert "returned_episode_returns" in info
        assert "returned_episode_lengths" in info
        assert "returned_episode" in info

    def test_safe_mode_cost_passthrough(self, safe_reward_env):
        """In safe mode, cost from SafeRLWrapper must pass through unchanged."""
        key = jax.random.PRNGKey(0)
        obs, state = safe_reward_env.reset(key)
        k1, k2 = jax.random.split(key)
        action = safe_reward_env.action_space().sample(k2)
        next_obs, state2, reward, costs, done, info = safe_reward_env.step(k1, state, action)
        assert costs.shape == (safe_reward_env._env.num_constraints,)
        assert jnp.allclose(costs, info["constraint_costs"])

    def test_is_safe_flag(self, battery_log_env, safe_env):
        """_is_safe flag should correctly identify SafeRLWrapper."""
        r = lambda o, a, no, r, i: r
        env_log = RewardWrapper(battery_log_env, r)
        env_safe = RewardWrapper(safe_env, r)
        assert env_log._is_safe is False
        assert env_safe._is_safe is True

    def test_property_delegation(self, battery_reward_env, battery_log_env):
        """Properties should match the inner LogWrapper."""
        assert battery_reward_env.obs_size == battery_log_env.obs_size
        assert battery_reward_env.num_actions == battery_log_env.num_actions
        assert battery_reward_env.action_size == battery_log_env.action_size
