#!/usr/bin/env python3
"""DC-OPF on case5: PowerZoo HiGHS (LP) vs PowerZooJax ALM.

Usage (PowerZoo importable on this machine)::

    python examples/dc_opf_congestion_compare.py

Notes
-----
- **No congestion** (default line 1–2 limit ±400 MW): JAX matches HiGHS at the same load.
- **Artificial congestion**: tighten line 0 (bus1–bus2) to ±150 MW. HiGHS still solves the
  global LP optimum; the JAX path uses ALM + a single merit-order inner loop and stops when
  sum of line violations < tol, so it may stop at a **feasible but suboptimal** point and
  disagree with the LP.

If PowerZoo is not installed, the script runs the JAX side only and prints a notice.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np

# Repo root (for imports when run as a script)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import jax.numpy as jnp

from powerzoojax.case import create_case5
from powerzoojax.envs.grid.dc_opf import prepare_dcopf, dc_opf

POWERZOO = Path(os.environ.get("POWERZOO_ROOT", str(_ROOT.parent / "PowerZoo"))).expanduser()
CASE5_LOAD_MW = np.array([100.0, 150.0, 200.0, 180.0, 120.0], dtype=float)


def case5_tight_line0(cap_half: float):
    """Clone case5 with line 0 floor/cap set to symmetric ±cap_half."""
    c = create_case5()
    lc = np.array(c.line_cap, dtype=float)
    lf = np.array(c.line_floor, dtype=float)
    lc[0] = cap_half
    lf[0] = -cap_half
    return c.replace(
        line_cap=jnp.array(lc, dtype=jnp.float32),
        line_floor=jnp.array(lf, dtype=jnp.float32),
    )


def pz_tight_line0(cap_half: float):
    try:
        sys.path.insert(0, str(POWERZOO))
        from powerzoo.case import load_case as pz_load_case

        case = pz_load_case(5)
        case.init()
        case.lines = case.lines.copy()
        idx0 = case.lines.index[0]
        case.lines.loc[idx0, "cap"] = cap_half
        case.lines.loc[idx0, "floor"] = -cap_half
        return case
    except ImportError:
        return None


def main():
    load_j = jnp.array(CASE5_LOAD_MW)

    print("=" * 72)
    print("Scenario A: baseline case5 (line0 ±400 MW) — same as equivalence tests")
    print("=" * 72)
    _run_one(case5_tight_line0(400.0), "400", load_j)

    print("\n" + "=" * 72)
    print("Scenario B: congested — line0 ±150 MW (merit unconstrained flow violates)")
    print("=" * 72)
    _run_one(case5_tight_line0(150.0), "150", load_j)

    print("\nNote: under B, HiGHS line0 flow should sit at the +150 MW cap;")
    print("if JAX reports converged=True but unit_power/LMP differ greatly from HiGHS,")
    print("that matches known ALM+Frank–Wolfe limitation under congestion (see docstring).")


def _run_one(case_jax, label: str, load_j):
    setup = prepare_dcopf(case_jax)
    rj = dc_opf(setup, load_j, max_iter=200, tol=1e-4)
    p_j = np.asarray(rj.unit_power, dtype=float)
    lmp_j = np.asarray(rj.lmp, dtype=float)
    flow_j = np.asarray(setup.M_u @ rj.unit_power - setup.PTDF @ load_j)

    print(f"  [JAX] converged={bool(rj.converged)}  iters={int(rj.iterations)}")
    print(f"  [JAX] unit_power (MW): {np.round(p_j, 2)}")
    print(f"  [JAX] total_cost (MC objective): {float(rj.total_cost):.4f}")
    print(f"  [JAX] LMP: {np.round(lmp_j, 4)}")
    print(f"  [JAX] line0 flow: {flow_j[0]:.4f}  (limit ±{label})")

    case_pz = pz_tight_line0(float(label))
    if case_pz is None:
        print("  [PowerZoo] not installed — skipping HiGHS comparison.")
        return

    sys.path.insert(0, str(POWERZOO))
    from powerzoo.envs.grid.cal_opf_trans import solve_ed_opf_detailed

    rp = solve_ed_opf_detailed(case_pz, CASE5_LOAD_MW, solver_type="scipy")
    p_p = np.asarray(rp["unit_power_mw"], dtype=float)
    lmp_p = np.asarray(rp["lmp"], dtype=float)
    flow_p = np.asarray(rp["line_flow_mw"], dtype=float)

    print(f"  [HiGHS] success={rp['success']}  slack={rp.get('slack_violation', 'n/a')}")
    print(f"  [HiGHS] unit_power (MW): {np.round(p_p, 2)}")
    print(f"  [HiGHS] total_cost: {rp['total_cost']:.4f}")
    print(f"  [HiGHS] LMP: {np.round(lmp_p, 4)}")
    print(f"  [HiGHS] line0 flow: {flow_p[0]:.4f}")

    print(f"  --- diff --- max|Δp|={np.max(np.abs(p_j - p_p)):.4f}  "
          f"max|ΔLMP|={np.max(np.abs(lmp_j - lmp_p)):.4f}")


if __name__ == "__main__":
    main()
