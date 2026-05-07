"""Shared fixtures for resource environment tests."""

import pytest
import jax
import jax.numpy as jnp

from powerzoojax.envs.resource.battery import BatteryEnv, BatteryParams, make_battery_params
from powerzoojax.envs.resource.renewable import (
    RenewableEnv, SolarEnv, WindEnv, RenewableParams,
)
from powerzoojax.envs.resource.vehicle import VehicleEnv, VehicleParams, make_vehicle_params
from powerzoojax.envs.resource.flexload import FlexLoadEnv, FlexLoadParams


@pytest.fixture
def key():
    """Deterministic PRNG key."""
    return jax.random.PRNGKey(42)
