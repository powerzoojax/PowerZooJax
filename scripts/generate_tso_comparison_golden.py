"""Generate / refresh the TSO comparison golden file.

Writes: tests/golden/tso_comparison_contract.json

Usage (JAX side only — no PowerZoo needed):
    python scripts/generate_tso_comparison_golden.py

Usage (with PowerZoo cross-repo verification):
    python scripts/generate_tso_comparison_golden.py \
        --powerzoo-dir /path/to/PowerZoo

    # Or via env vars:
    POWERZOO_DIR=/path/to/PowerZoo python scripts/generate_tso_comparison_golden.py

If PowerZoo is requested but unavailable, the script fails explicitly.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

# Ensure PowerZooJax is importable when run from repo root
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from powerzoojax.tasks.tso import (
    TSO_COMPARISON_SCHEMA,
    make_comparison_tso_load_trace,
    make_comparison_tso_params,
)
from powerzoojax.envs.grid.unit_commitment import UnitCommitmentEnv


GOLDEN_PATH = REPO_ROOT / "tests" / "golden" / "tso_comparison_contract.json"


def _build_jax_golden() -> dict:
    trace = make_comparison_tso_load_trace(48, 0.5)
    schema = TSO_COMPARISON_SCHEMA

    # Verify JAX action shape
    import jax
    env = UnitCommitmentEnv()
    params = make_comparison_tso_params()
    action_shape = list(env.action_space(params).shape)

    return {
        "_comment": (
            "TSO comparison benchmark contract. "
            "JAX side generated from PowerZooJax. "
            "See _golden_type / _python_verified for cross-repo status. "
            "Refresh with: python scripts/generate_tso_comparison_golden.py [--powerzoo-dir PATH]"
        ),
        "_golden_type": "jax_snapshot",
        "_python_verified": False,
        "generated_by": (
            "powerzoojax/tasks/tso.py — "
            "TSO_COMPARISON_SCHEMA + make_comparison_tso_load_trace"
        ),
        "case_id":             schema["case_id"],
        "n_units":             schema["n_units"],
        "n_agents":            schema["n_agents"],
        "max_steps":           schema["max_steps"],
        "delta_t_minutes":     schema["delta_t_minutes"],
        "delta_t_hours":       schema["delta_t_hours"],
        "load_source":         schema["load_source"],
        "load_trace_formula":  schema["load_trace_formula"],
        "enable_uc":           schema["enable_uc"],
        "enable_reserve":      schema["enable_reserve"],
        "reserve_margin_frac": schema["reserve_margin_frac"],
        "action_shape":        action_shape,
        "action_range":        schema["action_range"],
        "action_layout":       schema["action_layout"],
        "action_semantics":    schema["action_semantics"],
        "reward_components":   schema["reward_components"],
        "reward_scale":        schema["reward_scale"],
        "cost_channels_jax":   schema["cost_channels_jax"],
        "load_trace_spot_checks": {
            "step_0":             round(float(trace[0]), 8),
            "step_18":            round(float(trace[18]), 8),
            "step_42":            round(float(trace[42]), 8),
            "trace_sum_48steps":  round(float(np.sum(trace)), 6),
            "trace_min":          round(float(np.min(trace)), 8),
            "trace_max":          round(float(np.max(trace)), 8),
        },
        "accepted_gaps": schema["accepted_gaps"],
    }


def _verify_powerzoo(powerzoo_dir: Path, powerzoo_python: Path | None) -> dict:
    """Run PowerZoo side checks, return a summary dict."""
    if powerzoo_python is None:
        for candidate in [
            powerzoo_dir / ".venv" / "bin" / "python",
            powerzoo_dir / "venv" / "bin" / "python",
        ]:
            if candidate.exists():
                powerzoo_python = candidate
                break
    if powerzoo_python is None or not powerzoo_python.exists():
        sys.exit(
            f"ERROR: Could not find a Python interpreter for PowerZoo at {powerzoo_dir}.\n"
            "Set --powerzoo-python explicitly."
        )

    env = {**os.environ, "PYTHONPATH": str(powerzoo_dir)}

    def _run(script: str) -> str:
        result = subprocess.run(
            [str(powerzoo_python), "-c", textwrap.dedent(script)],
            capture_output=True, text=True, timeout=120, env=env,
        )
        if result.returncode != 0:
            sys.exit(
                f"PowerZoo subprocess failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        return result.stdout.strip()

    print(f"  PowerZoo dir:    {powerzoo_dir}")
    print(f"  PowerZoo python: {powerzoo_python}")

    # Check what tasks are actually registered
    registered_str = _run("""
        from powerzoo.tasks.registry import list_tasks
        print(','.join(sorted(list_tasks())))
    """)
    registered = [t for t in registered_str.split(',') if t]
    print(f"  Registered tasks: {registered}")

    has_centralized = 'comparison_tso_centralized' in registered

    if not has_centralized:
        print(
            "  WARNING: comparison_tso_centralized is NOT registered in PowerZoo.\n"
            "  The golden will record registered tasks but cannot capture a full\n"
            "  Python-side reference. Register the task in PowerZoo and re-run\n"
            "  to produce a Python-verified golden."
        )
        return {
            "powerzoo_reachable": True,
            "powerzoo_registered_tasks": registered,
            "comparison_tso_centralized_registered": False,
            "status": (
                "INCOMPLETE: comparison_tso_centralized not registered in PowerZoo. "
                "Python-side reference not captured. "
                "Register the task and re-run to complete cross-repo verification."
            ),
        }

    # comparison_tso_centralized exists — capture full reference
    action_shape_str = _run("""
        from powerzoo.tasks.middle.comparison_tso import CentralizedComparisonTSOTask
        env = CentralizedComparisonTSOTask().create_env()
        print(env.action_space.shape)
    """)

    py_trace_csv = _run("""
        from powerzoo.tasks.middle.comparison_tso import _make_synthetic_load_trace
        import numpy as np
        trace = _make_synthetic_load_trace(48, 0.5)
        print(','.join(f'{v:.8f}' for v in trace))
    """)
    py_trace = np.array([float(v) for v in py_trace_csv.split(',')], dtype=np.float32)
    jax_trace = make_comparison_tso_load_trace(48, 0.5)
    max_diff = float(np.max(np.abs(py_trace - jax_trace)))
    if max_diff > 1e-5:
        sys.exit(f"ERROR: JAX/Python trace mismatch, max diff={max_diff:.2e}")

    print(f"  action_shape: {action_shape_str}")
    print(f"  load trace max diff vs JAX: {max_diff:.2e} ✓")

    return {
        "powerzoo_reachable": True,
        "powerzoo_registered_tasks": registered,
        "comparison_tso_centralized_registered": True,
        "status": "VERIFIED: comparison_tso_centralized exists and load trace matches JAX.",
        "python_action_shape": action_shape_str,
        "python_load_trace_max_diff_vs_jax": max_diff,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate TSO comparison golden file.")
    parser.add_argument(
        "--powerzoo-dir",
        default=os.environ.get("POWERZOO_DIR", ""),
        help="Path to PowerZoo repo (or set POWERZOO_DIR env var)",
    )
    parser.add_argument(
        "--powerzoo-python",
        default=os.environ.get("POWERZOO_PYTHON", ""),
        help="Path to PowerZoo Python interpreter (or set POWERZOO_PYTHON env var)",
    )
    args = parser.parse_args()

    print("Building JAX golden...")
    golden = _build_jax_golden()
    print(f"  n_agents:     {golden['n_agents']}")
    print(f"  action_shape: {golden['action_shape']}")
    print(f"  load_source:  {golden['load_source']}")
    print(f"  trace step_0: {golden['load_trace_spot_checks']['step_0']:.6f}")

    if args.powerzoo_dir:
        print("\nVerifying PowerZoo side...")
        pz_info = _verify_powerzoo(
            Path(args.powerzoo_dir),
            Path(args.powerzoo_python) if args.powerzoo_python else None,
        )
        golden["_powerzoo_verification"] = pz_info
        # Upgrade golden type only when full verification passed
        if pz_info.get("comparison_tso_centralized_registered"):
            golden["_golden_type"] = "python_verified_reference"
            golden["_python_verified"] = True
    else:
        print("\nNo --powerzoo-dir provided; skipping cross-repo verification.")
        golden["_powerzoo_verification"] = None

    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(golden, indent=2) + "\n")
    print(f"\nWrote: {GOLDEN_PATH}")


if __name__ == "__main__":
    main()
