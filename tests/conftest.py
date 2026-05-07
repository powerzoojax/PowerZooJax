"""Shared fixtures for PowerZooJax tests.

Also seeds ``sys.path`` so the ``benchmarks/`` namespace package (no
``__init__.py`` at repo root) and the sibling ``PowerZoo`` repo are
importable from any test under ``tests/``.  This must happen at module
import time (not in a fixture) because some tests do
``from benchmarks.common.* import ...`` at module scope.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Repo root MUST come first so our namespace package ``./benchmarks``
# wins over PowerZoo's regular ``PowerZoo/benchmarks/`` package
# (which has an ``__init__.py`` and would otherwise shadow ``./benchmarks``
# and break ``benchmarks.common.*`` imports).
_repo_str = str(_REPO_ROOT)
if _REPO_ROOT.is_dir() and _repo_str not in sys.path:
    sys.path.insert(0, _repo_str)

from benchmarks.common.powerzoo_repo import ensure_powerzoo_on_path

# PowerZoo must be appended (low priority) so cross-backend tests can
# ``import powerzoo.*`` without overriding the in-repo regular package
# ``benchmarks/__init__.py``.  Note that ``benchmarks/`` IS a regular
# package (not a namespace package) precisely so it cannot be shadowed
# by ``PowerZoo/benchmarks/__init__.py`` even if PowerZoo were ahead of
# us on sys.path.
ensure_powerzoo_on_path(_REPO_ROOT, append=True)


import pytest
import jax
import jax.numpy as jnp

@pytest.fixture
def prng_key():
    """Deterministic PRNG key for reproducible tests."""
    return jax.random.PRNGKey(42)
