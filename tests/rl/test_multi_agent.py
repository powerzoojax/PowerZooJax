"""Tests for rl/multi_agent.py — GridMARLEnv.

L0: MARL contract (JIT, vmap, dict obs/rewards/dones, agent spaces)
L1: Physical coupling (resource SOC tracking, reward modes, episode termination)
"""

import pytest
import jax
import jax.numpy as jnp

from powerzoojax.case import create_case5
from powerzoojax.envs.grid.trans import TransGridEnv, make_trans_params
from powerzoojax.envs.resource.battery import make_battery_bundle
from powerzoojax.rl.multi_agent import GridMARLEnv, MARLState, MultiAgentEnvironment


# ========================== Fixtures ==========================

@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


def _make_grid_only_env():
    """Grid-only MARL env (no resources)."""
    case = create_case5()
    profiles = jnp.ones((48, case.n_loads), dtype=jnp.float32) * 0.5
    params = make_trans_params(case, load_profiles=profiles, max_steps=8)
    return GridMARLEnv(TransGridEnv(), params)


def _make_grid_with_battery():
    """Grid + battery bundle MARL env (1 device at bus 1)."""
    case = create_case5()
    profiles = jnp.ones((48, case.n_loads), dtype=jnp.float32) * 0.5
    bundle = make_battery_bundle(case, bus_ids=[1], power_mw=20.0, capacity_mwh=50.0)
    params = make_trans_params(case, load_profiles=profiles, max_steps=8,
                               resources=(bundle,))
    return GridMARLEnv(TransGridEnv(), params)


def _sample_actions(env, key):
    """Sample random actions for all agents."""
    actions = {}
    for name in env.agent_names:
        key, subkey = jax.random.split(key)
        actions[name] = env.action_space(name).sample(subkey)
    return actions


# ========================== L0: MARL Contract ==========================

class TestGridMARLContract:

    def test_is_multi_agent_env(self):
        env = _make_grid_only_env()
        assert isinstance(env, MultiAgentEnvironment)

    def test_reset_returns_dict_obs(self, key):
        env = _make_grid_only_env()
        obs_dict, state = env.reset(key)
        assert isinstance(obs_dict, dict)
        assert isinstance(state, MARLState)

    def test_agent_names_format(self):
        env = _make_grid_only_env()
        names = env.agent_names
        assert all(n.startswith("unit_") for n in names)
        assert len(names) == env.num_agents

    def test_agent_names_with_resources(self):
        env = _make_grid_with_battery()
        names = env.agent_names
        unit_names = [n for n in names if n.startswith("unit_")]
        resource_names = [n for n in names if not n.startswith("unit_")]
        assert len(unit_names) == 5  # case5 has 5 units
        assert len(resource_names) == 1  # 1 battery device
        assert "battery_0" in names

    def test_num_agents(self):
        env = _make_grid_only_env()
        assert env.num_agents == 5  # case5 has 5 units

    def test_num_agents_with_resources(self):
        env = _make_grid_with_battery()
        assert env.num_agents == 6  # 5 units + 1 battery device

    def test_obs_keys_match_agents(self, key):
        env = _make_grid_only_env()
        obs_dict, _ = env.reset(key)
        assert set(obs_dict.keys()) == set(env.agent_names)

    def test_obs_uniform_dim(self, key):
        env = _make_grid_only_env()
        obs_dict, _ = env.reset(key)
        dims = {name: obs.shape for name, obs in obs_dict.items()}
        shapes = list(dims.values())
        assert all(s == shapes[0] for s in shapes), f"Non-uniform obs dims: {dims}"

    def test_step_returns_5_tuple(self, key):
        env = _make_grid_only_env()
        obs_dict, state = env.reset(key)
        k1, k2 = jax.random.split(key)
        actions = _sample_actions(env, k2)
        result = env.step(k1, state, actions)
        assert len(result) == 5  # obs, state, rewards, dones, info

    def test_step_rewards_dict(self, key):
        env = _make_grid_only_env()
        obs_dict, state = env.reset(key)
        k1, k2 = jax.random.split(key)
        actions = _sample_actions(env, k2)
        _, _, rewards, dones, info = env.step(k1, state, actions)
        assert isinstance(rewards, dict)
        assert set(rewards.keys()) == set(env.agent_names)

    def test_step_dones_has_all(self, key):
        env = _make_grid_only_env()
        obs_dict, state = env.reset(key)
        k1, k2 = jax.random.split(key)
        actions = _sample_actions(env, k2)
        _, _, _, dones, _ = env.step(k1, state, actions)
        assert "__all__" in dones
        for name in env.agent_names:
            assert name in dones

    def test_spaces_per_agent(self):
        env = _make_grid_only_env()
        for name in env.agent_names:
            obs_space = env.observation_space(name)
            act_space = env.action_space(name)
            assert obs_space.shape[0] > 0
            assert act_space.shape[0] > 0

    def test_info_has_cost(self, key):
        env = _make_grid_only_env()
        obs_dict, state = env.reset(key)
        k1, k2 = jax.random.split(key)
        actions = _sample_actions(env, k2)
        _, _, _, _, info = env.step(k1, state, actions)
        assert "cost_sum" in info
        assert "constraint_costs" in info
        assert float(info["cost_sum"]) >= 0.0

    def test_jit_compatible(self, key):
        env = _make_grid_only_env()

        @jax.jit
        def reset_and_step(key):
            obs, state = env.reset(key)
            k1, k2 = jax.random.split(key)
            actions = {}
            keys = jax.random.split(k2, env.num_agents)
            for i, name in enumerate(env.agent_names):
                actions[name] = env.action_space(name).sample(keys[i])
            return env.step(k1, state, actions)

        result = reset_and_step(key)
        assert len(result) == 5

    def test_lax_scan_rollout(self, key):
        """Run a short rollout via lax.scan."""
        env = _make_grid_only_env()
        obs_dict, state = env.reset(key)

        def scan_step(carry, _):
            state, key = carry
            key, k1, k2 = jax.random.split(key, 3)
            actions = {}
            keys = jax.random.split(k2, env.num_agents)
            for i, name in enumerate(env.agent_names):
                actions[name] = env.action_space(name).sample(keys[i])
            obs, state, rewards, dones, info = env.step(k1, state, actions)
            return (state, key), rewards[env.agent_names[0]]

        (final_state, _), rewards = jax.lax.scan(
            scan_step, (state, key), None, length=10
        )
        assert rewards.shape == (10,)


# ========================== L1: Physical Coupling ==========================

class TestGridMARLPhysics:

    def test_shared_reward_equal(self, key):
        """In shared mode, all agents get the same reward."""
        env = _make_grid_only_env()
        obs_dict, state = env.reset(key)
        k1, k2 = jax.random.split(key)
        actions = _sample_actions(env, k2)
        _, _, rewards, _, _ = env.step(k1, state, actions)

        reward_values = [float(rewards[name]) for name in env.agent_names]
        assert all(r == reward_values[0] for r in reward_values)

    def test_done_all_consistent(self, key):
        """__all__ done matches individual agent dones."""
        env = _make_grid_only_env()
        obs_dict, state = env.reset(key)

        for i in range(8):
            k1, key = jax.random.split(key)
            k2, key = jax.random.split(key)
            actions = _sample_actions(env, k2)
            obs_dict, state, rewards, dones, info = env.step(k1, state, actions)

        # After 8 steps (max_steps=8), should be done
        assert bool(dones["__all__"])
        for name in env.agent_names:
            assert bool(dones[name]) == bool(dones["__all__"])

    def test_different_actions_produce_different_state(self, key):
        """Different dispatch actions change dispatch distribution and flows."""
        env = _make_grid_only_env()
        _, state = env.reset(key)
        k1, k2 = jax.random.split(key)

        # Action set 1: all at low dispatch
        actions1 = {name: jnp.array([-0.8]) for name in env.agent_names}
        _, state1, _, _, _ = env.step(k1, state, actions1)

        # Action set 2: all at high dispatch
        actions2 = {name: jnp.array([0.8]) for name in env.agent_names}
        _, state2, _, _, _ = env.step(k2, state, actions2)

        # Total generation is load-balanced by the DC slack unit, so compare
        # the actual dispatch allocation and resulting network flows instead.
        assert not bool(jnp.allclose(
            state1.grid_state.unit_power_mw,
            state2.grid_state.unit_power_mw,
            atol=1.0,
        )), (
            "Different actions should change the post-slack unit dispatch "
            f"allocation: {state1.grid_state.unit_power_mw} vs "
            f"{state2.grid_state.unit_power_mw}"
        )
        assert not bool(jnp.allclose(
            state1.grid_state.line_flow_mw,
            state2.grid_state.line_flow_mw,
            atol=1.0,
        )), (
            "Different actions should change line flows: "
            f"{state1.grid_state.line_flow_mw} vs "
            f"{state2.grid_state.line_flow_mw}"
        )

    def test_resource_agent_present(self, key):
        """Battery resource agent is included in observations."""
        env = _make_grid_with_battery()
        obs_dict, state = env.reset(key)
        assert "battery_0" in obs_dict
        assert obs_dict["battery_0"].shape == obs_dict["unit_0"].shape

    def test_episode_terminates_at_max_steps(self, key):
        """Episode terminates after max_steps."""
        env = _make_grid_only_env()  # max_steps=8
        obs_dict, state = env.reset(key)

        for i in range(8):
            k1, key = jax.random.split(key)
            k2, key = jax.random.split(key)
            actions = _sample_actions(env, k2)
            obs_dict, state, rewards, dones, info = env.step(k1, state, actions)
            if i < 7:
                assert not bool(dones["__all__"])

        assert bool(dones["__all__"])

    def test_auto_reset_after_done(self, key):
        """State resets after episode ends (auto-reset for lax.scan)."""
        env = _make_grid_only_env()  # max_steps=8
        obs_dict, state = env.reset(key)

        for i in range(9):  # One step past episode end
            k1, key = jax.random.split(key)
            k2, key = jax.random.split(key)
            actions = _sample_actions(env, k2)
            obs_dict, state, rewards, dones, info = env.step(k1, state, actions)

        # After auto-reset, time_step should be 1 (reset + one step)
        assert int(state.grid_state.time_step) == 1

    def test_resource_soc_tracking(self, key):
        """Battery SOC changes after step (verifies resource state propagates)."""
        env = _make_grid_with_battery()
        obs_dict, state = env.reset(key)

        initial_soc = float(state.grid_state.resource_states[0].soc[0])

        k1, k2 = jax.random.split(key)
        # Charge action (positive = discharge in battery convention, negative = charge)
        # Use max charge action
        actions = {name: jnp.array([-1.0]) for name in env.agent_names}
        _, new_state, _, _, _ = env.step(k1, state, actions)

        new_soc = float(new_state.grid_state.resource_states[0].soc[0])
        # SOC should change with a non-zero power action
        assert new_soc != initial_soc, \
            f"Battery SOC should change after charging: {initial_soc} -> {new_soc}"

    def test_multi_battery_devices(self, key):
        """Bundle with 2 battery devices creates 2 separate agents."""
        case = create_case5()
        profiles = jnp.ones((48, case.n_loads), dtype=jnp.float32) * 0.5
        bundle = make_battery_bundle(case, bus_ids=[1, 3], power_mw=20.0, capacity_mwh=50.0)
        params = make_trans_params(case, load_profiles=profiles, max_steps=8,
                                   resources=(bundle,))
        env = GridMARLEnv(TransGridEnv(), params)

        assert env.num_agents == 7  # 5 units + 2 battery devices
        assert "battery_0" in env.agent_names
        assert "battery_1" in env.agent_names

        obs_dict, state = env.reset(key)
        assert set(obs_dict.keys()) == set(env.agent_names)
        # All obs dims should be equal
        shapes = [obs.shape for obs in obs_dict.values()]
        assert all(s == shapes[0] for s in shapes)


# ========================== L2: IPPO Training ==========================

class TestIPPOTraining:

    def test_ippo_train_smoke(self, key):
        """IPPO backend compiles and runs 2 updates without error."""
        from powerzoojax.rl.ippo import make_ippo_train
        from powerzoojax.rl.config import TrainConfig

        env = _make_grid_only_env()
        # Very small config: 2 updates × 4 envs × 8 steps × 5 agents = 320 timesteps
        config = TrainConfig(
            algo="ippo",
            total_timesteps=320,
            num_envs=4,
            n_steps=8,
            n_epochs=2,
            hidden_dims=(16, 16),
        )
        train_fn = make_ippo_train(env, config)
        result = train_fn(key)

        assert result.params is not None
        assert "mean_reward" in result.metrics
        assert "loss" in result.metrics
        rewards = result.metrics["mean_reward"]
        assert rewards.shape[0] >= 1

    def test_ippo_train_with_battery(self, key):
        """IPPO training works with battery resource agents."""
        from powerzoojax.rl.ippo import make_ippo_train
        from powerzoojax.rl.config import TrainConfig

        env = _make_grid_with_battery()
        config = TrainConfig(
            algo="ippo",
            total_timesteps=480,
            num_envs=4,
            n_steps=8,
            n_epochs=2,
            hidden_dims=(16, 16),
        )
        train_fn = make_ippo_train(env, config)
        result = train_fn(key)
        assert result.params is not None

    def test_preset_case5_ippo(self):
        """case5-ippo preset can be loaded and its env_factory invoked."""
        from powerzoojax.rl.presets import get_preset
        preset = get_preset("case5-ippo")
        assert preset.config.algo == "ippo"
        env = preset.env_factory()
        assert env.num_agents == 5  # units only

    def test_preset_case5_ippo_battery(self):
        """case5-ippo-battery preset creates env with battery agents."""
        from powerzoojax.rl.presets import get_preset
        preset = get_preset("case5-ippo-battery")
        env = preset.env_factory()
        assert env.num_agents == 7  # 5 units + 2 battery devices
        assert "battery_0" in env.agent_names

    def test_make_train_dispatches_ippo(self, key):
        """make_train() dispatcher routes ippo to IPPO backend."""
        from powerzoojax.rl.trainer import make_train
        from powerzoojax.rl.config import TrainConfig

        env = _make_grid_only_env()
        config = TrainConfig(
            algo="ippo",
            total_timesteps=160,
            num_envs=4,
            n_steps=8,
            n_epochs=1,
            hidden_dims=(16, 16),
        )
        train_fn = make_train(env, config)
        result = train_fn(key)
        assert result.config.algo == "ippo"
