"""Cross-repo TSO comparison tests — PowerZoo side (opt-in only).

These tests require a live PowerZoo installation.
They are NOT part of the default pytest run.

Usage:
    # Via env vars:
    POWERZOO_DIR=/path/to/PowerZoo pytest tests/grid/crossrepo_tso_comparison.py

    # Or with explicit Python interpreter:
    POWERZOO_DIR=/path/to/PowerZoo \
    POWERZOO_PYTHON=/path/to/PowerZoo/.venv/bin/python \
    pytest tests/grid/crossrepo_tso_comparison.py

If POWERZOO_DIR is not set, all tests in this file FAIL explicitly.
Silent skip is not allowed.

Closure scope:
  - Shared comparison contract (action shape, load trace, cost keys, n_units) is
    established between JAX and PowerZoo sides.
  - Speed benchmarking workflow (48-step rollout via subprocess) is closed.
  - Full behavioral parity has accepted gaps:
      obs shape (JAX includes line flows; PowerZoo obs vector may differ),
      dispatch solver (JAX continuous-relaxation SCUC vs PowerZoo backend),
      reserve cost routing (penalty weights may differ across sides).
"""
from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest

from powerzoojax.tasks.tso import make_comparison_tso_load_trace


# ---------------------------------------------------------------------------
# PowerZoo discovery — fails explicitly, never silently skips
# ---------------------------------------------------------------------------

def _get_powerzoo_dir() -> Path:
    val = os.environ.get("POWERZOO_DIR", "").strip()
    if not val:
        pytest.fail(
            "POWERZOO_DIR env var is not set. "
            "These cross-repo tests require a live PowerZoo installation.\n"
            "Set POWERZOO_DIR to the root of the PowerZoo repo and re-run."
        )
    d = Path(val)
    if not d.exists():
        pytest.fail(f"POWERZOO_DIR={d} does not exist.")
    return d


def _get_powerzoo_python() -> Path:
    val = os.environ.get("POWERZOO_PYTHON", "").strip()
    if val:
        p = Path(val)
        if not p.exists():
            pytest.fail(f"POWERZOO_PYTHON={p} does not exist.")
        return p

    # Fall back: sibling .venv
    d = _get_powerzoo_dir()
    candidates = [
        d / ".venv" / "bin" / "python",
        d / "venv" / "bin" / "python",
    ]
    for c in candidates:
        if c.exists():
            return c

    pytest.fail(
        f"Could not find a Python interpreter for PowerZoo at {d}. "
        "Set POWERZOO_PYTHON explicitly."
    )


def _run(script: str, timeout: int = 120) -> subprocess.CompletedProcess:
    python = _get_powerzoo_python()
    powerzoo_dir = _get_powerzoo_dir()
    env = {**os.environ, "PYTHONPATH": str(powerzoo_dir)}
    return subprocess.run(
        [str(python), "-c", textwrap.dedent(script)],
        capture_output=True, text=True, timeout=timeout, env=env,
    )


def _assert_run(script: str, timeout: int = 120) -> None:
    result = _run(script, timeout)
    assert result.returncode == 0, (
        f"PowerZoo subprocess failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


# ============================================================
# Cross-repo contract tests
# ============================================================

def test_crossrepo_registry():
    """comparison_tso_centralized must be registered in PowerZoo."""
    _assert_run("""
        from powerzoo.tasks.registry import list_tasks
        tasks = list_tasks()
        assert 'comparison_tso_centralized' in tasks, (
            f"comparison_tso_centralized not registered in PowerZoo. Found: {tasks}"
        )
    """)


def test_crossrepo_no_old_marl_task():
    """Old 54-agent comparison_tso_uc must NOT be registered."""
    _assert_run("""
        from powerzoo.tasks.registry import list_tasks
        tasks = list_tasks()
        assert 'comparison_tso_uc' not in tasks, (
            f"comparison_tso_uc (old 54-agent MARL task) must not be registered. Found: {tasks}"
        )
    """)


def test_crossrepo_action_space_shape():
    """Python action space must be Box(108,) — identical to JAX side."""
    _assert_run("""
        from powerzoo.tasks.middle.comparison_tso import CentralizedComparisonTSOTask
        env = CentralizedComparisonTSOTask().create_env()
        shape = env.action_space.shape
        assert shape == (108,), f"Expected Box(108,), got Box{shape}"
    """)


def test_crossrepo_action_space_bounds():
    _assert_run("""
        import numpy as np
        from powerzoo.tasks.middle.comparison_tso import CentralizedComparisonTSOTask
        env = CentralizedComparisonTSOTask().create_env()
        assert float(env.action_space.low.min()) == -1.0
        assert float(env.action_space.high.max()) == 1.0
    """)


def test_crossrepo_n_units():
    _assert_run("""
        from powerzoo.tasks.middle.comparison_tso import CentralizedComparisonTSOTask
        env = CentralizedComparisonTSOTask().create_env()
        assert env._n_units == 54, f"Expected 54 units, got {env._n_units}"
    """)


def test_crossrepo_reset_obs_finite():
    _assert_run("""
        import numpy as np
        from powerzoo.tasks.middle.comparison_tso import CentralizedComparisonTSOTask
        env = CentralizedComparisonTSOTask().create_env()
        obs, info = env.reset(seed=0)
        assert np.all(np.isfinite(obs)), "obs contains non-finite values after reset"
    """)


def test_crossrepo_step_info_keys():
    _assert_run("""
        import numpy as np
        from powerzoo.tasks.middle.comparison_tso import CentralizedComparisonTSOTask
        env = CentralizedComparisonTSOTask().create_env()
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(np.zeros(108, dtype=np.float32))
        for k in ('gen_cost', 'startup_cost', 'no_load_cost', 'reserve_shortfall'):
            assert k in info, f"Missing info key: {k}"
    """)


def test_crossrepo_rollout_48steps():
    _assert_run("""
        import numpy as np
        from powerzoo.tasks.middle.comparison_tso import CentralizedComparisonTSOTask
        env = CentralizedComparisonTSOTask().create_env()
        env.reset(seed=0)
        action = np.ones(108, dtype=np.float32) * 0.1
        steps = 0
        for _ in range(48):
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            if terminated or truncated:
                break
        assert steps == 48, f"Expected 48 steps, got {steps}"
    """, timeout=180)


def test_crossrepo_load_trace_identical():
    """JAX and Python synthetic load traces must be numerically identical."""
    jax_trace = np.array(make_comparison_tso_load_trace(48, 0.5), dtype=np.float32)

    result = _run("""
        import numpy as np
        from powerzoo.tasks.middle.comparison_tso import _make_synthetic_load_trace
        trace = _make_synthetic_load_trace(48, 0.5)
        print(','.join(f'{v:.8f}' for v in trace))
    """)
    assert result.returncode == 0, result.stderr

    py_trace = np.array(
        [float(v) for v in result.stdout.strip().split(',')],
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        jax_trace, py_trace, rtol=0, atol=1e-5,
        err_msg="JAX and Python synthetic load traces differ",
    )


def test_crossrepo_load_injected_not_gb_data():
    _assert_run("""
        import numpy as np
        from powerzoo.tasks.middle.comparison_tso import (
            CentralizedComparisonTSOTask, _make_synthetic_load_trace
        )
        env = CentralizedComparisonTSOTask().create_env()
        env.reset(seed=0)
        trace = _make_synthetic_load_trace(48, 0.5)
        d_max = env._inner.case.loads['d_max'].values.astype(np.float64)
        expected_total = float(np.sum(trace[0] * d_max))
        actual_total = float(np.sum(env._inner.grid._get_node_loads_p_current()))
        assert abs(actual_total - expected_total) < 0.01, (
            f"Load mismatch: expected {expected_total:.2f} MW, got {actual_total:.2f} MW"
        )
    """)
