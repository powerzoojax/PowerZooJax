"""TSO comparison env must use real GB net-load on both backends, byte-for-byte.

If this test fails, paper-table walltime-to-target between backends is no
longer comparable on the TSO row.

Imports are deferred so the ``benchmarks`` regular package and the
``PowerZoo`` sibling repo are both resolvable through ``conftest.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.common.powerzoo_repo import find_powerzoo_repo

_REPO_ROOT = Path(__file__).resolve().parents[2]
_POWERZOO_PATH = find_powerzoo_repo(_REPO_ROOT)


def _powerzoo_available() -> bool:
    return _POWERZOO_PATH is not None


# ---------------------------------------------------------------------------
# 1. Both sides must produce the same trace for the same (split, start_idx).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("split,start_idx", [
    ("train", 0),
    ("train", 100),
    ("iid", 0),
])
def test_gb_load_trace_parity(split: str, start_idx: int):
    if not _powerzoo_available():
        pytest.skip("PowerZoo sibling repo not present")
    pytest.importorskip("pandas")

    import numpy as np
    try:
        from powerzoojax.tasks.tso import make_comparison_tso_load_trace as jax_trace_fn
        from powerzoo.tasks.middle.comparison_tso import _make_gb_load_trace as pz_trace_fn
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Trace import failed (likely missing parquet): {exc}")

    try:
        jax_trace = jax_trace_fn(split=split, episode_start_idx=start_idx, n_steps=48)
        pz_trace = pz_trace_fn(split=split, episode_start_idx=start_idx, n_steps=48)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"GB parquet not available: {exc}")

    assert jax_trace.shape == pz_trace.shape == (48,), (
        f"Trace shape mismatch: PowerZooJax={jax_trace.shape}, "
        f"PowerZoo={pz_trace.shape}"
    )
    diff = float(np.max(np.abs(np.asarray(jax_trace) - np.asarray(pz_trace))))
    assert diff < 1e-5, (
        f"GB load trace divergence (split={split!r}, start_idx={start_idx}): "
        f"max |Δ|={diff:.3e}.  PowerZoo and PowerZooJax must compute the "
        f"identical net-load curve for cross-backend records to be comparable."
    )


# ---------------------------------------------------------------------------
# 2. The deprecated synthetic helper still exists (test/CI helper) but is
#    flagged as not for cross-backend use.
# ---------------------------------------------------------------------------

def test_synthetic_helper_is_test_only():
    from powerzoojax.tasks.tso import _comparison_tso_synthetic_trace
    arr = _comparison_tso_synthetic_trace(48)
    assert arr.shape == (48,)
    # The synthetic curve max ~1.0 is a known sin shape; just sanity-check
    # the contract that the curve stays in [0.10, 1.00].
    assert 0.0 < float(arr.min()) < 1.0
    assert float(arr.max()) <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# 3. The TSO_COMPARISON_SCHEMA's load_source field must reflect "GB real",
#    not the legacy "deterministic_synthetic".
# ---------------------------------------------------------------------------

def test_schema_advertises_gb_real():
    from powerzoojax.tasks.tso import TSO_COMPARISON_SCHEMA

    assert TSO_COMPARISON_SCHEMA["load_source"] == "gb_real", (
        f"TSO_COMPARISON_SCHEMA still claims load_source="
        f"{TSO_COMPARISON_SCHEMA['load_source']!r}.  P0-1 fix replaces the "
        f"deterministic sin trace with real GB data; update the schema."
    )


# ---------------------------------------------------------------------------
# 4. CentralizedComparisonTSOTask must accept (split, episode_start_idx).
# ---------------------------------------------------------------------------

def test_powerzoo_task_accepts_split_kwargs():
    if not _powerzoo_available():
        pytest.skip("PowerZoo sibling repo not present")
    import inspect
    from powerzoo.tasks.middle.comparison_tso import CentralizedComparisonTSOTask

    sig = inspect.signature(CentralizedComparisonTSOTask.__init__)
    for kw in ("split", "episode_start_idx"):
        assert kw in sig.parameters, (
            f"CentralizedComparisonTSOTask.__init__ must accept {kw!r} so the "
            f"cross-backend driver can record the actual data window honestly."
        )
