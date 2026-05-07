"""L2 Equivalence: 3-phase BFS JAX vs PowerZoo on small test case.

Tolerance: atol=1e-2 (float32 + different DLF construction).
"""

import sys
import os
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_POWERZOO_PATH = Path(os.environ.get("POWERZOO_ROOT", str(_REPO_ROOT.parent / "PowerZoo"))).expanduser()

try:
    sys.path.insert(0, str(_POWERZOO_PATH))
    from powerzoo.envs.grid.cal_pf_dist_3phase import (
        build_3phase_topology as pz_build_3phase,
        run_3phase_bfs_power_flow as pz_run_3phase,
    )
    _HAS_POWERZOO = True
except ImportError:
    _HAS_POWERZOO = False

from powerzoojax.envs.grid.bfs_3phase_power_flow import (
    build_3phase_topology,
    bfs_3phase_power_flow,
)

pytestmark = pytest.mark.skipif(not _HAS_POWERZOO, reason="PowerZoo not available")


@pytest.fixture(autouse=True, scope="module")
def _force_float32_mode():
    """Keep this float32 equivalence test isolated from ACOPF's global x64 side effects."""
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", False)


@pytest.fixture(scope="module")
def small_case():
    """4-bus test case with balanced impedances."""
    n_nodes = 4
    from_nodes = np.array([0, 1, 2])
    to_nodes = np.array([1, 2, 3])
    Z = np.zeros((3, 3, 3), dtype=complex)
    for i in range(3):
        Z[i] = (0.1 + 0.2j) * np.eye(3)
    P = np.ones((3, 3)) * 0.01  # 3 buses * 3 phases
    Q = np.ones((3, 3)) * 0.005
    return n_nodes, from_nodes, to_nodes, Z, P, Q


@pytest.fixture(scope="module")
def pz_result(small_case):
    n_nodes, from_nodes, to_nodes, Z, P, Q = small_case
    topo = pz_build_3phase(n_nodes, from_nodes, to_nodes, Z, ref_bus=0)
    return pz_run_3phase(topo, P.flatten(), Q.flatten())


@pytest.fixture(scope="module")
def jax_result(small_case):
    n_nodes, from_nodes, to_nodes, Z, P, Q = small_case
    topo = build_3phase_topology(n_nodes, from_nodes, to_nodes, Z, ref_bus=0)
    P_flat = jnp.array(P.flatten())
    Q_flat = jnp.array(Q.flatten())
    return bfs_3phase_power_flow(topo, P_flat, Q_flat)


class TestBFS3PhEquivalence:
    ATOL = 1e-2

    def test_both_converge(self, pz_result, jax_result):
        assert pz_result['converged']
        assert bool(jax_result.converged)

    def test_v_mag_eq(self, pz_result, jax_result):
        v_pz = pz_result['V_mag']
        v_jax = np.asarray(jax_result.v_mag)
        np.testing.assert_allclose(v_jax, v_pz, atol=self.ATOL,
                                   err_msg="3-phase voltage magnitudes differ")

    def test_p_branch_eq(self, pz_result, jax_result):
        p_pz = pz_result['P_branch']
        p_jax = np.asarray(jax_result.P_branch)
        np.testing.assert_allclose(p_jax, p_pz, atol=self.ATOL,
                                   err_msg="3-phase branch P flows differ")
