"""Composite microgrid environments.

Microgrid envs in PowerZooJax are composite envs built from the same bundle
protocol used by grid environments, but on a degenerate one-bus/no-power-flow
topology. The current public surface is the data-center microgrid benchmark
and its parameter factories.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "DCMicrogridState",
    "DCMicrogridParams",
    "DataCenterMicrogridEnv",
    "make_dcmicrogrid_params",
    "make_dcmicrogrid_params_with_profiles",
]

_LAZY_IMPORTS = {
    "DCMicrogridState": ("powerzoojax.envs.microgrid.dc_microgrid", "DCMicrogridState"),
    "DCMicrogridParams": ("powerzoojax.envs.microgrid.dc_microgrid", "DCMicrogridParams"),
    "DataCenterMicrogridEnv": ("powerzoojax.envs.microgrid.dc_microgrid", "DataCenterMicrogridEnv"),
    "make_dcmicrogrid_params": ("powerzoojax.envs.microgrid.dc_microgrid", "make_dcmicrogrid_params"),
    "make_dcmicrogrid_params_with_profiles": (
        "powerzoojax.envs.microgrid.dc_microgrid", "make_dcmicrogrid_params_with_profiles",
    ),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        mod_path, attr = _LAZY_IMPORTS[name]
        mod = importlib.import_module(mod_path)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
