"""L1.5 Integration Test — Action Normalization Pipeline.

Verifies that all envs:
1. Declare action_space with bounds [-1, 1]
2. Map action=0 to the midpoint of the physical range
3. Map action=1 to the physical max, action=-1 to the physical min
4. Clip out-of-range actions gracefully
5. space.sample() returns values in [-1, 1]
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powerzoojax.envs.base import denormalize_action
from powerzoojax.envs.grid.trans import TransGridEnv, make_trans_params
from powerzoojax.envs.grid.power_flow import dc_power_flow
from powerzoojax.envs.grid.dist import DistGridEnv, make_dist_params
from powerzoojax.envs.resource.battery import BatteryEnv, make_battery_params
from powerzoojax.envs.resource.renewable import RenewableEnv, SolarEnv
from powerzoojax.envs.resource.vehicle import VehicleEnv, make_vehicle_params
from powerzoojax.envs.resource.flexload import FlexLoadEnv, FlexLoadParams
from powerzoojax.case import create_case5, create_case33bw


# ========================== denormalize_action ==========================


class TestDenormalizeAction:

    def test_midpoint(self):
        result = denormalize_action(jnp.float32(0.0), jnp.float32(10.0), jnp.float32(100.0))
        np.testing.assert_allclose(float(result), 55.0, atol=1e-5)

    def test_max(self):
        result = denormalize_action(jnp.float32(1.0), jnp.float32(10.0), jnp.float32(100.0))
        np.testing.assert_allclose(float(result), 100.0, atol=1e-5)

    def test_min(self):
        result = denormalize_action(jnp.float32(-1.0), jnp.float32(10.0), jnp.float32(100.0))
        np.testing.assert_allclose(float(result), 10.0, atol=1e-5)

    def test_clip_above(self):
        result = denormalize_action(jnp.float32(5.0), jnp.float32(0.0), jnp.float32(100.0))
        np.testing.assert_allclose(float(result), 100.0, atol=1e-5)

    def test_clip_below(self):
        result = denormalize_action(jnp.float32(-5.0), jnp.float32(0.0), jnp.float32(100.0))
        np.testing.assert_allclose(float(result), 0.0, atol=1e-5)

    def test_vectorized(self):
        low = jnp.array([0.0, 10.0])
        high = jnp.array([100.0, 500.0])
        action = jnp.array([0.0, 1.0])
        result = denormalize_action(action, low, high)
        np.testing.assert_allclose(np.asarray(result), [50.0, 500.0], atol=1e-5)

    def test_symmetric(self):
        low = jnp.float32(-20.0)
        high = jnp.float32(20.0)
        np.testing.assert_allclose(
            float(denormalize_action(jnp.float32(0.5), low, high)), 10.0, atol=1e-5)
        np.testing.assert_allclose(
            float(denormalize_action(jnp.float32(-0.5), low, high)), -10.0, atol=1e-5)

    def test_jit_compatible(self):
        f = jax.jit(denormalize_action)
        result = f(jnp.float32(0.0), jnp.float32(0.0), jnp.float32(100.0))
        np.testing.assert_allclose(float(result), 50.0, atol=1e-5)


# ========================== Action space declarations ==========================


@pytest.fixture
def trans_params():
    return make_trans_params(create_case5())


@pytest.fixture
def dist_params():
    return make_dist_params(create_case33bw())


@pytest.fixture
def battery_params():
    return make_battery_params()


@pytest.fixture
def renewable_params():
    return SolarEnv().default_params()


@pytest.fixture
def vehicle_params():
    return make_vehicle_params()


@pytest.fixture
def flexload_params():
    return FlexLoadParams()


ALL_ENVS = [
    ("TransGridEnv", TransGridEnv, "trans_params"),
    ("BatteryEnv", BatteryEnv, "battery_params"),
    ("RenewableEnv", RenewableEnv, "renewable_params"),
    ("VehicleEnv", VehicleEnv, "vehicle_params"),
    ("FlexLoadEnv", FlexLoadEnv, "flexload_params"),
]


class TestActionSpaceBounds:

    @pytest.mark.parametrize("name,env_cls,params_fixture", ALL_ENVS)
    def test_action_space_is_neg1_to_1(self, name, env_cls, params_fixture, request):
        params = request.getfixturevalue(params_fixture)
        env = env_cls()
        space = env.action_space(params)
        if env_cls is FlexLoadEnv:
            # FlexLoad uses unit-scaled [0, 1] actions (curtail_frac, shift_out_frac)
            np.testing.assert_allclose(np.asarray(space.low), 0.0, atol=1e-6)
            np.testing.assert_allclose(np.asarray(space.high), 1.0, atol=1e-6)
        else:
            np.testing.assert_allclose(np.asarray(space.low), -1.0, atol=1e-6)
            np.testing.assert_allclose(np.asarray(space.high), 1.0, atol=1e-6)

    @pytest.mark.parametrize("name,env_cls,params_fixture", ALL_ENVS)
    def test_sample_in_range(self, name, env_cls, params_fixture, request):
        params = request.getfixturevalue(params_fixture)
        env = env_cls()
        space = env.action_space(params)
        for seed in range(5):
            action = space.sample(jax.random.PRNGKey(seed))
            assert jnp.all(action >= -1.0), f"{name}: sample below -1"
            assert jnp.all(action <= 1.0), f"{name}: sample above 1"


# ========================== TransGridEnv action mapping ==========================


class TestTransGridActionMapping:

    def test_zero_action_is_midpoint(self, trans_params):
        env = TransGridEnv()
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, trans_params)
        case = trans_params.case

        action = jnp.zeros(case.n_units)
        _, state2, _, _, _, _ = env.step(key, state, action, trans_params)

        target_mid = denormalize_action(action, case.unit_p_min, case.unit_p_max)
        node_load_mw = case.nodes_loads_map @ trans_params.load_profiles[0]
        _, _, expected_mid = dc_power_flow(case, target_mid, node_load_mw)
        np.testing.assert_allclose(
            np.asarray(state2.unit_power_mw),
            np.asarray(expected_mid),
            atol=1e-4,
        )

    def test_plus_one_is_pmax(self, trans_params):
        env = TransGridEnv()
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, trans_params)
        case = trans_params.case

        action = jnp.ones(case.n_units)
        _, state2, _, _, _, _ = env.step(key, state, action, trans_params)

        target_max = denormalize_action(action, case.unit_p_min, case.unit_p_max)
        node_load_mw = case.nodes_loads_map @ trans_params.load_profiles[0]
        _, _, expected_max = dc_power_flow(case, target_max, node_load_mw)
        np.testing.assert_allclose(
            np.asarray(state2.unit_power_mw),
            np.asarray(expected_max),
            atol=1e-4,
        )

    def test_minus_one_is_pmin(self, trans_params):
        env = TransGridEnv()
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, trans_params)
        case = trans_params.case

        action = -jnp.ones(case.n_units)
        _, state2, _, _, _, _ = env.step(key, state, action, trans_params)

        target_min = denormalize_action(action, case.unit_p_min, case.unit_p_max)
        node_load_mw = case.nodes_loads_map @ trans_params.load_profiles[0]
        _, _, expected_min = dc_power_flow(case, target_min, node_load_mw)
        np.testing.assert_allclose(
            np.asarray(state2.unit_power_mw),
            np.asarray(expected_min),
            atol=1e-4,
        )

    def test_out_of_range_clipped(self, trans_params):
        env = TransGridEnv()
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, trans_params)
        case = trans_params.case

        action = jnp.ones(case.n_units) * 10.0
        _, state2, _, _, _, _ = env.step(key, state, action, trans_params)

        target_clip = denormalize_action(action, case.unit_p_min, case.unit_p_max)
        node_load_mw = case.nodes_loads_map @ trans_params.load_profiles[0]
        _, _, expected_clip = dc_power_flow(case, target_clip, node_load_mw)
        np.testing.assert_allclose(
            np.asarray(state2.unit_power_mw),
            np.asarray(expected_clip),
            atol=1e-4,
        )


# ========================== BatteryEnv action mapping ==========================


class TestBatteryActionMapping:

    def test_zero_action_is_idle(self, battery_params):
        env = BatteryEnv()
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, battery_params)

        _, state2, _, _, _, _ = env.step(key, state, jnp.float32(0.0), battery_params)
        np.testing.assert_allclose(float(state2.current_p_mw), 0.0, atol=1e-4)

    def test_plus_one_is_max_discharge(self, battery_params):
        env = BatteryEnv()
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, battery_params)

        _, state2, _, _, _, _ = env.step(key, state, jnp.float32(1.0), battery_params)
        assert float(state2.current_p_mw) > 0  # discharge is positive

    def test_minus_one_is_max_charge(self, battery_params):
        env = BatteryEnv()
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, battery_params)

        _, state2, _, _, _, _ = env.step(key, state, jnp.float32(-1.0), battery_params)
        assert float(state2.current_p_mw) < 0  # charge is negative


# ========================== Full pipeline: tanh output → env ==================


class TestTanhPolicyPipeline:
    """Verify that a tanh policy output works end-to-end."""

    def test_tanh_output_transgrid(self, trans_params):
        env = TransGridEnv()
        key = jax.random.PRNGKey(42)
        obs, state = env.reset(key, trans_params)
        case = trans_params.case

        key, subkey = jax.random.split(key)
        raw = jax.random.normal(subkey, shape=(case.n_units,))
        action = jnp.tanh(raw)

        obs2, state2, reward, costs, done, info = env.step(
            key, state, action, trans_params
        )
        assert obs2.shape[0] > 0
        target_dispatch = denormalize_action(action, case.unit_p_min, case.unit_p_max)
        assert jnp.all(target_dispatch >= case.unit_p_min - 1e-4)
        assert jnp.all(target_dispatch <= case.unit_p_max + 1e-4)
        assert jnp.all(jnp.isfinite(state2.unit_power_mw))
        assert abs(float(jnp.sum(state2.node_injection_mw))) < 1e-3

    def test_tanh_output_battery(self, battery_params):
        env = BatteryEnv()
        key = jax.random.PRNGKey(42)
        obs, state = env.reset(key, battery_params)

        key, subkey = jax.random.split(key)
        raw = jax.random.normal(subkey, shape=(1,))
        action = jnp.tanh(raw)

        obs2, state2, reward, costs, done, info = env.step(
            key, state, action, battery_params
        )
        assert obs2.shape[0] > 0
        assert abs(float(state2.current_p_mw)) <= battery_params.power_mw + 1e-4
