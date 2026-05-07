"""L2 Equivalence: DCOPF JAX vs PowerZoo HiGHS LP on case5.

The JAX solver uses ADMM; primal objective is linear ``mc_c @ p``, matching
``powerzoo.envs.grid.cal_opf_trans.solve_ed_opf_detailed(..., solver_type="scipy")``.

**Golden references** (below) were produced once from PowerZoo HiGHS so that CI
can assert numerical parity **without** installing PowerZoo.  Optional tests
``TestDCOPFEquivalencePowerZooLive`` re-query PowerZoo when available and check
that live output still matches the golden (guards against drift).

To regenerate goldens (developer machine with PowerZoo)::

    python -c "
    import sys, numpy as np
    sys.path.insert(0, '/path/to/PowerZoo')
    from powerzoo.case import load_case as pz_load_case
    from powerzoo.envs.grid.cal_opf_trans import solve_ed_opf_detailed
    L = np.array([100.,150.,200.,180.,120.])
    r = solve_ed_opf_detailed(pz_load_case(5), L, solver_type='scipy')
    print('p', r['unit_power_mw']); print('lmp', r['lmp'])
    case = pz_load_case(5); case.init(); case.lines = case.lines.copy()
    i0 = case.lines.index[0]
    case.lines.loc[i0,'cap']=150.; case.lines.loc[i0,'floor']=-150.
    r2 = solve_ed_opf_detailed(case, L, solver_type='scipy')
    print('p2', r2['unit_power_mw']); print('lmp2', r2['lmp'])
    "
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_POWERZOO_PATH = Path(os.environ.get("POWERZOO_ROOT", str(_REPO_ROOT.parent / "PowerZoo"))).expanduser()


def _try_import_powerzoo():
    """Return (load_case, solve_ed_opf_detailed) or None if PowerZoo unavailable."""
    try:
        sys.path.insert(0, str(_POWERZOO_PATH))
        from powerzoo.case import load_case as pz_load_case
        from powerzoo.envs.grid.cal_opf_trans import solve_ed_opf_detailed
        return pz_load_case, solve_ed_opf_detailed
    except ImportError:
        return None


_PZ = _try_import_powerzoo()
_HAS_POWERZOO = _PZ is not None

from powerzoojax.case import create_case5
from powerzoojax.envs.grid.dc_opf import prepare_dcopf, dc_opf

# Shared load vector (MW)
CASE5_LOAD_MW = np.array([100.0, 150.0, 200.0, 180.0, 120.0], dtype=float)

# ---------------------------------------------------------------------------
# Golden HiGHS outputs (PowerZoo solve_ed_opf_detailed, solver_type="scipy")
# Captured 2026-04-06; regenerate if PowerZoo LP formulation changes.
# ---------------------------------------------------------------------------

CASE5_HIGHS_NOMINAL_UNIT_POWER_MW = np.array(
    [40.0, 80.0, 20.0, 10.0, 600.0], dtype=np.float64
)
CASE5_HIGHS_NOMINAL_LMP = np.array(
    [15.0, 15.0, 15.0, 15.0, 15.0], dtype=np.float64
)
CASE5_HIGHS_NOMINAL_LINE0_FLOW_MW = 214.65890015443495
CASE5_HIGHS_NOMINAL_TOTAL_COST = 8760.0

CASE5_HIGHS_CONGESTED_UNIT_POWER_MW = np.array(
    [
        5.0,
        10.0,
        140.0508351488743,
        10.0,
        584.9491648511258,
    ],
    dtype=np.float64,
)
CASE5_HIGHS_CONGESTED_LMP = np.array(
    [
        8.647917079289627,
        34.99108734402853,
        30.0,
        16.27450980392157,
        10.0,
    ],
    dtype=np.float64,
)
CASE5_HIGHS_CONGESTED_LINE0_FLOW_MW = 150.0
CASE5_HIGHS_CONGESTED_TOTAL_COST = 10671.016702977488


@pytest.fixture(autouse=True, scope="module")
def _force_float32_mode():
    """Keep DCOPF equivalence checks isolated from ACOPF's global x64 side effects."""
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", False)


@pytest.fixture(scope="module")
def jax_result_nominal():
    case = create_case5()
    setup = prepare_dcopf(case)
    load = jnp.array(CASE5_LOAD_MW, dtype=jnp.float32)
    return dc_opf(setup, load)


@pytest.fixture(scope="module")
def jax_result_congested():
    case = create_case5()
    lc = np.array(case.line_cap, dtype=float)
    lf = np.array(case.line_floor, dtype=float)
    lc[0] = 150.0
    lf[0] = -150.0
    case_tight = case.replace(
        line_cap=jnp.array(lc, dtype=jnp.float32),
        line_floor=jnp.array(lf, dtype=jnp.float32),
    )
    setup = prepare_dcopf(case_tight)
    load = jnp.array(CASE5_LOAD_MW, dtype=jnp.float32)
    return dc_opf(setup, load, max_iter=200, tol=0.05)


# ==================== Core: JAX vs hardcoded HiGHS (no PowerZoo) ====================

class TestDCOPFEquivalenceGoldenNominal:
    """JAX vs frozen HiGHS reference — runs in CI without PowerZoo."""

    def test_generation_matches_load(self, jax_result_nominal):
        total_load = float(CASE5_LOAD_MW.sum())
        gen_jax = float(jnp.sum(jax_result_nominal.unit_power))
        np.testing.assert_allclose(gen_jax, total_load, rtol=0.0, atol=1e-2)

    def test_unit_power_allclose_highs(self, jax_result_nominal):
        np.testing.assert_allclose(
            np.asarray(jax_result_nominal.unit_power, dtype=float),
            CASE5_HIGHS_NOMINAL_UNIT_POWER_MW,
            rtol=0.0,
            atol=0.5,
        )

    def test_lmp_allclose_highs(self, jax_result_nominal):
        np.testing.assert_allclose(
            np.asarray(jax_result_nominal.lmp, dtype=float),
            CASE5_HIGHS_NOMINAL_LMP,
            rtol=0.0,
            atol=0.5,
        )

    def test_line0_flow_reasonable(self, jax_result_nominal):
        f0 = float(jax_result_nominal.line_flow[0])
        np.testing.assert_allclose(
            f0, CASE5_HIGHS_NOMINAL_LINE0_FLOW_MW, rtol=0.0, atol=0.5
        )

    def test_total_cost_same_order_as_highs(self, jax_result_nominal):
        cj = float(jax_result_nominal.total_cost)
        cr = CASE5_HIGHS_NOMINAL_TOTAL_COST
        assert cj > 0 and cr > 0
        ratio = max(cj, cr) / max(min(cj, cr), 1.0)
        assert ratio < 3.0


class TestDCOPFEquivalenceGoldenCongested:
    """Congested line0 ±150 MW — JAX vs frozen HiGHS."""

    def test_line0_near_limit(self, jax_result_congested):
        line0_flow = float(jax_result_congested.line_flow[0])
        np.testing.assert_allclose(
            line0_flow,
            CASE5_HIGHS_CONGESTED_LINE0_FLOW_MW,
            rtol=0.0,
            atol=1.0,
        )

    def test_unit_power_allclose_highs(self, jax_result_congested):
        np.testing.assert_allclose(
            np.asarray(jax_result_congested.unit_power, dtype=float),
            CASE5_HIGHS_CONGESTED_UNIT_POWER_MW,
            rtol=0.0,
            atol=1.0,
        )

    def test_lmp_allclose_highs(self, jax_result_congested):
        np.testing.assert_allclose(
            np.asarray(jax_result_congested.lmp, dtype=float),
            CASE5_HIGHS_CONGESTED_LMP,
            rtol=0.0,
            atol=1.0,
        )

    def test_total_cost_same_order_as_highs(self, jax_result_congested):
        cj = float(jax_result_congested.total_cost)
        cr = CASE5_HIGHS_CONGESTED_TOTAL_COST
        assert cj > 0 and cr > 0
        ratio = max(cj, cr) / max(min(cj, cr), 1.0)
        assert ratio < 1.05, (
            f"Cost drift vs golden: JAX={cj:.4f}, golden={cr:.4f}"
        )


# ==================== Optional: live PowerZoo agrees with golden ====================


@pytest.fixture(scope="module")
def pz_result_nominal():
    if not _HAS_POWERZOO:
        pytest.skip("PowerZoo not available")
    pz_load_case, solve_ed_opf_detailed = _PZ
    case = pz_load_case(5)
    return solve_ed_opf_detailed(case, CASE5_LOAD_MW, solver_type="scipy")


@pytest.fixture(scope="module")
def pz_result_congested():
    if not _HAS_POWERZOO:
        pytest.skip("PowerZoo not available")
    pz_load_case, solve_ed_opf_detailed = _PZ
    case = pz_load_case(5)
    case.init()
    case.lines = case.lines.copy()
    idx0 = case.lines.index[0]
    case.lines.loc[idx0, "cap"] = 150.0
    case.lines.loc[idx0, "floor"] = -150.0
    return solve_ed_opf_detailed(case, CASE5_LOAD_MW, solver_type="scipy")


@pytest.mark.external
class TestDCOPFEquivalencePowerZooLive:
    """When PowerZoo is installed, verify live HiGHS still matches frozen goldens."""

    @pytest.mark.skipif(not _HAS_POWERZOO, reason="PowerZoo not available")
    def test_nominal_live_matches_golden(self, pz_result_nominal):
        assert pz_result_nominal["success"]
        assert pz_result_nominal.get("slack_violation", 1.0) == 0.0
        np.testing.assert_allclose(
            np.asarray(pz_result_nominal["unit_power_mw"], dtype=float),
            CASE5_HIGHS_NOMINAL_UNIT_POWER_MW,
            rtol=0.0,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(pz_result_nominal["lmp"], dtype=float),
            CASE5_HIGHS_NOMINAL_LMP,
            rtol=0.0,
            atol=1e-6,
        )

    @pytest.mark.skipif(not _HAS_POWERZOO, reason="PowerZoo not available")
    def test_congested_live_matches_golden(self, pz_result_congested):
        assert pz_result_congested["success"]
        np.testing.assert_allclose(
            np.asarray(pz_result_congested["unit_power_mw"], dtype=float),
            CASE5_HIGHS_CONGESTED_UNIT_POWER_MW,
            rtol=0.0,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(pz_result_congested["lmp"], dtype=float),
            CASE5_HIGHS_CONGESTED_LMP,
            rtol=0.0,
            atol=1e-6,
        )
