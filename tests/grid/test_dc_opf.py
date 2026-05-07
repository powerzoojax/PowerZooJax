"""Tests for dc_opf.py — L0 JAX + L1 physics.

L0: JIT, vmap, pytree stability for DCOPF.
L1: Power balance, gen within limits, cost ordering, line flows.
"""

import jax
import jax.numpy as jnp
import jax.tree_util as tu
import pytest

from powerzoojax.case import create_case5
from powerzoojax.envs.grid.dc_opf import (
    DCOPFSetup,
    DCOPFResult,
    prepare_dcopf,
    dc_opf,
)


@pytest.fixture(autouse=True, scope="module")
def _force_float32_mode():
    """Keep DCOPF tests isolated from ACOPF's global x64 side effects."""
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", False)


@pytest.fixture(scope="module")
def case5():
    return create_case5()


@pytest.fixture(scope="module")
def setup(case5):
    return prepare_dcopf(case5)


@pytest.fixture(scope="module")
def load():
    return jnp.array([100.0, 150.0, 200.0, 180.0, 120.0], dtype=jnp.float32)


@pytest.fixture(scope="module")
def result(setup, load):
    return dc_opf(setup, load)


# ==================== L0: JAX Contract ====================

class TestDCOPF_L0_JAX:

    def test_jit_compile(self, setup, load):
        jit_fn = jax.jit(lambda s, l: dc_opf(s, l, None, 50, 1e-3))
        r = jit_fn(setup, load)
        assert r.unit_power.shape == (setup.n_units,)

    def test_vmap_batch(self, setup, load):
        batch = jnp.stack([load, load * 0.5, load * 1.5])
        vfn = jax.vmap(lambda l: dc_opf(setup, l, None, 50, 1e-3))
        results = vfn(batch)
        assert results.unit_power.shape == (3, setup.n_units)

    def test_result_pytree(self, result):
        leaves = tu.tree_leaves(result)
        assert len(leaves) == 10  # DCOPFResult has 10 fields

    def test_setup_pytree(self, setup):
        flat, td = tu.tree_flatten(setup)
        rebuilt = tu.tree_unflatten(td, flat)
        assert rebuilt.n_units == setup.n_units

    def test_make_jaxpr(self, setup, load):
        jaxpr = jax.make_jaxpr(
            lambda s, l: dc_opf(s, l, None, 50, 1e-3)
        )(setup, load)
        assert jaxpr is not None


# ==================== L1: Physical Correctness ====================

class TestDCOPF_L1_Physics:

    def test_power_balance(self, result, load):
        """Total generation should match total load (inner rebalance in dc_opf)."""
        total_gen = float(jnp.sum(result.unit_power))
        total_load = float(jnp.sum(load))
        assert abs(total_gen - total_load) < 1e-2, (
            f"Power balance: gen={total_gen:.6f}, load={total_load:.6f}"
        )

    def test_gen_within_limits(self, result, case5):
        """Generation should be within [p_min, p_max]."""
        p_min = case5.unit_p_min
        p_max = case5.unit_p_max
        assert jnp.all(result.unit_power >= p_min - 1.0)
        assert jnp.all(result.unit_power <= p_max + 1.0)

    def test_cost_positive(self, result):
        """Total cost should be positive."""
        assert float(result.total_cost) > 0.0

    def test_cheapest_unit_produces_most(self, result, case5):
        """Cheapest unit should produce the most (or be at max)."""
        mc = case5.unit_cost_c
        cheapest = int(jnp.argmin(mc))
        # Cheapest unit should produce at or near its max
        assert float(result.unit_power[cheapest]) > float(case5.unit_p_max[cheapest]) * 0.5

    def test_line_flow_shape(self, result, case5):
        assert result.line_flow.shape == (case5.n_lines,)

    def test_lmp_positive(self, result):
        """LMP should be non-negative."""
        assert jnp.all(result.lmp >= -1.0)

    def test_commitment(self, setup, load):
        """Commitment should deactivate units."""
        commit = jnp.array([1.0, 1.0, 0.0, 0.0, 1.0])
        r = dc_opf(setup, load, commitment=commit)
        # Decommitted units should produce near zero
        assert float(r.unit_power[2]) < 1.0
        assert float(r.unit_power[3]) < 1.0

    def test_lower_load_lower_cost(self, setup, load):
        """Lower load should give lower cost."""
        r_high = dc_opf(setup, load)
        r_low = dc_opf(setup, load * 0.5)
        assert float(r_low.total_cost) < float(r_high.total_cost)
