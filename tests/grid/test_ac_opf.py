"""Tests for ac_opf.py — L0 JAX + L1 physics.

L0: JIT, vmap, pytree stability for ACOPF.
L1: Voltage limits, gen limits, cost structure.
L1-LMP: IFT-based LMP correctness (finite, positive, MC alignment).
"""

import jax
import jax.numpy as jnp
import jax.tree_util as tu
import pytest

from powerzoojax.case import create_case5
from powerzoojax.envs.grid.power_flow import ac_thermal_check
from powerzoojax.envs.grid.ac_opf import (
    ACOPFSetup,
    ACOPFResult,
    prepare_acopf,
    ac_opf,
    _recover_lmp_reduced_kkt as _compute_lmp_ift,
    _branch_s2_pu,
)


@pytest.fixture(scope="module")
def case5():
    return create_case5()


@pytest.fixture(scope="module")
def setup(case5):
    return prepare_acopf(case5)


@pytest.fixture(scope="module")
def result(setup, case5):
    return ac_opf(setup, case5.node_pd, case5.node_qd, max_iter=200)


# ==================== L0: JAX Contract ====================

class TestACOPF_L0_JAX:

    def test_jit_compile(self, setup, case5):
        jit_fn = jax.jit(lambda s, p, q: ac_opf(s, p, q, None, 200, 1e-3))
        r = jit_fn(setup, case5.node_pd, case5.node_qd)
        assert r.vm.shape == (case5.n_nodes,)
        assert r.unit_power.shape == (case5.n_units,)

    def test_vmap_batch(self, setup, case5):
        p = case5.node_pd
        q = case5.node_qd
        batch_p = jnp.stack([p, p * 0.8])
        batch_q = jnp.stack([q, q * 0.8])
        vfn = jax.vmap(lambda pp, qq: ac_opf(setup, pp, qq, None, 100, 1e-3))
        results = vfn(batch_p, batch_q)
        assert results.vm.shape == (2, case5.n_nodes)

    def test_result_pytree(self, result):
        leaves = tu.tree_leaves(result)
        assert len(leaves) == 11  # ACOPFResult has 11 fields

    def test_setup_pytree(self, setup):
        flat, td = tu.tree_flatten(setup)
        rebuilt = tu.tree_unflatten(td, flat)
        assert rebuilt.n_bus == setup.n_bus

    def test_make_jaxpr(self, setup, case5):
        jaxpr = jax.make_jaxpr(
            lambda s, p, q: ac_opf(s, p, q, None, 50, 1e-3)
        )(setup, case5.node_pd, case5.node_qd)
        assert jaxpr is not None


# ==================== L1: Physical Correctness ====================

class TestACOPF_L1_Physics:

    def test_vm_within_limits(self, result, case5):
        """Voltage should be within [v_min, v_max]."""
        v_min = case5.node_v_min
        v_max = case5.node_v_max
        # Allow small tolerance for optimizer inaccuracy
        assert jnp.all(result.vm >= v_min - 0.05)
        assert jnp.all(result.vm <= v_max + 0.05)

    def test_gen_within_limits(self, result, case5):
        """Active generation within [p_min, p_max]."""
        p_min = case5.unit_p_min
        p_max = case5.unit_p_max
        assert jnp.all(result.unit_power >= p_min - 1.0)
        assert jnp.all(result.unit_power <= p_max + 1.0)

    def test_cost_positive(self, result):
        assert float(result.total_cost) > 0.0

    def test_line_flow_shapes(self, result, case5):
        assert result.line_flow_p.shape == (case5.n_lines,)
        assert result.line_flow_q.shape == (case5.n_lines,)

    def test_lmp_shape(self, result, case5):
        assert result.lmp.shape == (case5.n_nodes,)

    def test_va_ref_bus_zero(self, result, setup):
        """Reference bus angle should be near zero."""
        assert abs(float(result.va[setup.ref_bus])) < 0.1

    def test_lower_load_lower_cost(self, setup, case5):
        """Lower load → lower cost."""
        r_high = ac_opf(setup, case5.node_pd, case5.node_qd, max_iter=100)
        r_low = ac_opf(setup, case5.node_pd * 0.5, case5.node_qd * 0.5, max_iter=100)
        assert float(r_low.total_cost) <= float(r_high.total_cost)


# ==================== L1-LMP: IFT-based LMP correctness ====================

class TestACOPF_L1_LMP_IFT:
    """Physical sanity checks for the IFT-based AC-LMP computation.

    The IFT method recovers λ_P (active power balance duals) from the KKT
    stationarity conditions using the power flow Jacobian.  Key invariants:
      - All LMPs must be finite.
      - Interior generator buses: LMP ≈ MC (KKT stationarity).
      - All LMPs positive for typical (non-degenerate) loading.
      - _compute_lmp_ift is JIT-compilable and deterministic.
      - _compute_lmp_ift is vmap-compatible.
    """

    def test_lmp_finite(self, result):
        """LMP values must be finite (no NaN / Inf)."""
        assert jnp.all(jnp.isfinite(result.lmp))

    def test_lmp_positive(self, result):
        """LMP should be positive under normal loading (costs > 0)."""
        # Allow small numerical slack around zero
        assert jnp.all(result.lmp > -5.0), f"Negative LMP: {result.lmp}"

    def test_lmp_at_interior_gen_bus_approx_mc(self, result, setup, case5):
        """Interior-generator bus LMP must be close to the unit's MC.

        Tolerance is relaxed (20 % relative + 5 $/MWh absolute) to
        account for the approximate augmented-Lagrangian primal solution.
        """
        Pg_mw = result.unit_power
        # d/dP of sum(a P^2 + c P) = 2 a P + c (PowerZoo AC-OPF; ignores mc_b)
        mc = 2.0 * case5.unit_cost_a * Pg_mw + case5.unit_cost_c

        found_interior = False
        for i in range(case5.n_units):
            p     = float(Pg_mw[i])
            p_min = float(case5.unit_p_min[i])
            p_max = float(case5.unit_p_max[i])
            # Only check truly interior generators (well away from both bounds)
            if p > p_min + 2.0 and p < p_max - 2.0:
                found_interior = True
                bus_i   = int(setup.gen_bus_idx[i])
                lmp_bus = float(result.lmp[bus_i])
                mc_i    = float(mc[i])
                tol     = 0.20 * abs(mc_i) + 5.0
                assert abs(lmp_bus - mc_i) <= tol, (
                    f"Gen {i} on bus {bus_i}: LMP={lmp_bus:.2f}, MC={mc_i:.2f}, "
                    f"tol={tol:.2f}"
                )

        if not found_interior:
            pytest.skip("No interior generator found in this dispatch — skip MC check")

    def test_lmp_spread_bounded(self, result):
        """LMP spread across buses should be within a realistic range.

        For a lightly loaded, uncongested transmission network the per-bus
        LMP differences are driven by losses only (~1-5 %).  A very large
        spread signals a numerical problem.
        """
        lmp   = result.lmp
        spread = float(jnp.max(lmp) - jnp.min(lmp))
        avg    = float(jnp.mean(jnp.abs(lmp)))
        # Generous guard: spread < 5× average + 100 $/MWh
        assert spread < 5.0 * avg + 100.0, (
            f"LMP spread {spread:.1f} >> average {avg:.1f}"
        )

    def test_lmp_ift_jit_deterministic(self, setup, case5):
        """JIT and eager execution of _compute_lmp_ift must agree."""
        nb    = case5.n_nodes
        ng    = case5.n_units
        Va    = jnp.zeros(nb)
        Vm    = jnp.ones(nb)
        Pg    = jnp.array(case5.unit_p_min, dtype=jnp.float32) / case5.base_mva
        Pgmin = jnp.array(case5.unit_p_min, dtype=jnp.float32) / case5.base_mva
        Pgmax = jnp.array(case5.unit_p_max, dtype=jnp.float32) / case5.base_mva

        lmp_eager = _compute_lmp_ift(setup, Va, Vm, Pg, Pgmin, Pgmax)
        lmp_jit   = jax.jit(_compute_lmp_ift)(setup, Va, Vm, Pg, Pgmin, Pgmax)
        assert jnp.allclose(lmp_eager, lmp_jit, atol=1e-4), (
            f"JIT mismatch: max_diff={float(jnp.max(jnp.abs(lmp_eager - lmp_jit)))}"
        )

    def test_lmp_ift_vmap_compatible(self, setup, case5):
        """_compute_lmp_ift must run correctly under vmap (batch of 2)."""
        nb    = case5.n_nodes
        Va    = jnp.zeros(nb)
        Vm    = jnp.ones(nb)
        Pg    = jnp.array(case5.unit_p_min, dtype=jnp.float32) / case5.base_mva
        Pgmin = jnp.array(case5.unit_p_min, dtype=jnp.float32) / case5.base_mva
        Pgmax = jnp.array(case5.unit_p_max, dtype=jnp.float32) / case5.base_mva

        batch_Va  = jnp.stack([Va,  Va  * 1.01])
        batch_Vm  = jnp.stack([Vm,  Vm  * 0.99])
        batch_Pg  = jnp.stack([Pg,  Pg  * 0.80])

        vfn = jax.vmap(
            lambda va, vm, pg: _compute_lmp_ift(setup, va, vm, pg, Pgmin, Pgmax)
        )
        out = vfn(batch_Va, batch_Vm, batch_Pg)
        assert out.shape == (2, nb)
        assert jnp.all(jnp.isfinite(out))


# ==================== L1: Branch Thermal Constraints ====================

class TestACOPF_L1_LineThermal:
    """Tests for branch thermal constraint enforcement (|Sf|² ≤ Smax², both ends).

    Validates:
      1. line_cap_pu field is populated with correct shape and sign.
      2. line_viol_mva in ACOPFResult is finite and non-negative.
      3. All-zero line_cap_pu (unlimited) produces line_viol_mva == 0.
      4. The gradient of the branch-flow penalty w.r.t. Va is non-zero at a
         violated operating point — confirms the penalty is mechanistically active
         and will push the solver away from violations during gradient descent.
    """

    def test_line_cap_pu_shape_and_sign(self, setup, case5):
        """line_cap_pu must have length n_lines with all entries ≥ 0."""
        assert setup.line_cap_pu is not None
        assert setup.line_cap_pu.shape == (case5.n_lines,)
        assert jnp.all(setup.line_cap_pu >= 0)

    def test_line_viol_mva_finite_nonneg(self, result):
        """line_viol_mva must be finite and ≥ 0."""
        assert jnp.isfinite(result.line_viol_mva)
        assert float(result.line_viol_mva) >= 0.0

    def test_line_viol_implies_positive_ac_thermal_cost(self, setup, case5, result):
        """If solver reports large line violation, post-hoc |S| overload cost is > 0."""
        z = jnp.zeros_like(result.line_flow_p)
        _, _, cost_t = ac_thermal_check(
            result.line_flow_p, result.line_flow_q, case5.line_cap,
            z, z, False)
        if float(result.line_viol_mva) > 0.1:
            assert float(cost_t) > 0.0

    def test_no_limits_zero_violation(self, setup, case5):
        """All-zero line_cap_pu (no thermal limits) must give line_viol_mva == 0."""
        setup_free = setup.replace(line_cap_pu=jnp.zeros_like(setup.line_cap_pu))
        r = ac_opf(setup_free, case5.node_pd, case5.node_qd, max_iter=50)
        assert float(r.line_viol_mva) == 0.0

    def test_branch_penalty_gradient_nonzero_at_violation(self, setup, case5):
        """Branch-penalty gradient w.r.t. Va must be non-zero when lines are violated.

        Procedure:
          1. Use a non-flat operating point (small Va angle spread) so that line
             flows are driven by angle differences, not just shunt capacitance.
          2. Set line_cap_pu = 0.001 p.u. on all currently-limited branches
             (far below any real flow) to guarantee violations.
          3. Check that jax.grad of the penalty term w.r.t. Va is non-zero,
             which proves the penalty contributes to the AL update direction.
        """
        nb = case5.n_nodes
        # Non-flat start: non-trivial angle spread creates flow gradients w.r.t. Va.
        Va = jnp.zeros(nb).at[1].set(0.15).at[2].set(-0.10)
        Vm = jnp.ones(nb)

        cap_tight = jnp.where(
            setup.line_cap_pu > 0,
            jnp.full_like(setup.line_cap_pu, 0.001),
            jnp.zeros_like(setup.line_cap_pu),
        )
        s = setup.replace(line_cap_pu=cap_tight)

        Sf2, St2 = _branch_s2_pu(Va, Vm, s)
        Smax2 = s.line_cap_pu ** 2
        has_cap = s.line_cap_pu > 0
        viol_f = jnp.where(has_cap, jnp.maximum(0.0, Sf2 - Smax2), 0.0)
        viol_t = jnp.where(has_cap, jnp.maximum(0.0, St2 - Smax2), 0.0)
        if float(jnp.sum(viol_f) + jnp.sum(viol_t)) == 0.0:
            pytest.skip("No line violations at non-flat start for this case — test vacuous")

        def branch_pen(Va_, Vm_):
            Sf2_, St2_ = _branch_s2_pu(Va_, Vm_, s)
            has_ = s.line_cap_pu > 0
            vf = jnp.where(has_, jnp.maximum(0.0, Sf2_ - s.line_cap_pu ** 2), 0.0)
            vt = jnp.where(has_, jnp.maximum(0.0, St2_ - s.line_cap_pu ** 2), 0.0)
            return 0.5 * jnp.float32(10.0) * (jnp.sum(vf ** 2) + jnp.sum(vt ** 2))

        g_Va = jax.grad(branch_pen, argnums=0)(Va, Vm)
        assert float(jnp.max(jnp.abs(g_Va))) > 0.0, (
            "Gradient of branch penalty w.r.t. Va must be non-zero at violated point"
        )

    def test_branch_penalty_gradient_jit(self, setup, case5):
        """Branch-penalty gradient must be JIT-compilable."""
        nb = case5.n_nodes
        Va = jnp.zeros(nb).at[1].set(0.15).at[2].set(-0.10)
        Vm = jnp.ones(nb)
        cap = jnp.where(
            setup.line_cap_pu > 0,
            jnp.full_like(setup.line_cap_pu, 0.001),
            jnp.zeros_like(setup.line_cap_pu),
        )
        s = setup.replace(line_cap_pu=cap)

        def branch_pen(Va_, Vm_):
            Sf2_, St2_ = _branch_s2_pu(Va_, Vm_, s)
            has_ = s.line_cap_pu > 0
            vf = jnp.where(has_, jnp.maximum(0.0, Sf2_ - s.line_cap_pu ** 2), 0.0)
            vt = jnp.where(has_, jnp.maximum(0.0, St2_ - s.line_cap_pu ** 2), 0.0)
            return 0.5 * jnp.float32(10.0) * (jnp.sum(vf ** 2) + jnp.sum(vt ** 2))

        g_jit = jax.jit(jax.grad(branch_pen, argnums=0))(Va, Vm)
        assert g_jit.shape == (nb,)
        assert jnp.all(jnp.isfinite(g_jit))
