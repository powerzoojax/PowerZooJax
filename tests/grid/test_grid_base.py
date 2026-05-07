"""L0 JAX contract for `GridState` / `GridParams` (grid/base.py).

These are shared pytree containers; physics lives elsewhere. Tests lock:
static pytree structure, JIT, and vmap over batched state.
"""

import jax
import jax.numpy as jnp
import jax.tree_util as tu
import pytest

from powerzoojax.envs.grid.base import GridParams, GridState


def _minimal_grid_state(n_units: int = 2, n_lines: int = 3, n_nodes: int = 4) -> GridState:
    return GridState(
        time_step=jnp.int32(0),
        done=jnp.bool_(False),
        unit_power_mw=jnp.zeros((n_units,), dtype=jnp.float32),
        line_flow_mw=jnp.zeros((n_lines,), dtype=jnp.float32),
        node_injection_mw=jnp.zeros((n_nodes,), dtype=jnp.float32),
        is_safe=jnp.bool_(True),
        n_violations=jnp.int32(0),
        total_cost=jnp.float32(0.0),
        vm=jnp.ones((n_nodes,), dtype=jnp.float32),
        va=jnp.zeros((n_nodes,), dtype=jnp.float32),
        q_gen=jnp.zeros((n_units,), dtype=jnp.float32),
        line_flow_q_mw=jnp.zeros((n_lines,), dtype=jnp.float32),
        resource_states=(),
    )


def _minimal_grid_params() -> GridParams:
    return GridParams(load_profiles=jnp.zeros((4, 1), dtype=jnp.float32))


class TestGridBaseL0:
    def test_grid_state_leaf_count(self):
        s = _minimal_grid_state()
        assert len(tu.tree_leaves(s)) == 12

    def test_grid_params_only_load_profiles_are_dynamic_leaves(self):
        p = _minimal_grid_params()
        assert len(tu.tree_leaves(p)) == 1

    def test_jit_replace_time_step(self):
        s = _minimal_grid_state()

        @jax.jit
        def bump(s: GridState) -> GridState:
            return s.replace(time_step=s.time_step + 1)

        out = bump(s)
        assert int(out.time_step) == 1

    def test_vmap_batch_first_dim(self):
        s0 = _minimal_grid_state()
        s1 = _minimal_grid_state()
        batch = jax.tree.map(lambda a, b: jnp.stack([a, b], axis=0), s0, s1)

        @jax.jit
        def row_sum_unit_power(s):
            return jnp.sum(s.unit_power_mw)

        out = jax.vmap(row_sum_unit_power)(batch)
        assert out.shape == (2,)
        assert (out == 0.0).all()


@pytest.mark.parametrize("physics,solver_mode", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_grid_params_static_fields_roundtrip(physics: int, solver_mode: int):
    p = GridParams(
        load_profiles=jnp.zeros((2, 1), dtype=jnp.float32),
        physics=physics,
        solver_mode=solver_mode,
    )
    assert p.physics == physics
    assert p.solver_mode == solver_mode
