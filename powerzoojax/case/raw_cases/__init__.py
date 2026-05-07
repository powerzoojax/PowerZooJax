"""
Raw Cases: Original OOP-style case definitions and SCUC JSON helpers.

``case14.json`` and ``scuc_json_to_case_units.py`` are mirrored under
``PowerZoo/powerzoo/case/raw_cases/``; keep both copies identical.

These cases use the same DataFrame-oriented layout as PowerZoo (via the
local ``powerzoojax.case.dataframe.DataFrame`` helper). They can be converted to JAX format
using the case_adapter.

Example:
    >>> from powerzoojax.case.raw_cases import Case5
    >>> from powerzoojax.case import case_to_jax
    >>> 
    >>> case = Case5()
    >>> case_data = case_to_jax(case)
"""

from powerzoojax.case.raw_cases.case5 import Case5
from powerzoojax.case.raw_cases.case33bw import Case33bw
from powerzoojax.case.raw_cases.case118 import Case118

__all__ = ['Case5', 'Case33bw', 'Case118']
