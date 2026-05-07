"""
Case Adapter: Convert PowerZoo Cases to JAX CaseData

Two paths — both end at the same ``build_case_from_tables()`` core:

Path A (standalone):  case .py files define tables → build_case_from_tables()
Path B (runtime):     PowerZoo ClearCase instance → case_to_jax() → build_case_from_tables()

Usage:
    >>> # Path B — from live PowerZoo object
    >>> from powerzoo.case import load_case
    >>> from powerzoojax.case import case_to_jax
    >>> case_data = case_to_jax(load_case("5"))
    >>>
    >>> # Path A — standalone (no PowerZoo dependency)
    >>> from powerzoojax.case import load_case
    >>> case_data = load_case("5")
"""

# Re-export from the builder module so existing imports keep working
from powerzoojax.case.case_builder import (  # noqa: F401
    build_case_from_tables,
    case_to_jax,
)

convert_case = case_to_jax
