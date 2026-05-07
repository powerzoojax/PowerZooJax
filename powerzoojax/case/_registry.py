"""Case registry for discovering and filtering available JAX case data.

Mirrors PowerZoo's ``_registry.py`` but adapted for JAX factory functions.
Each case has a :class:`CaseMeta` record and a factory ``() -> CaseData``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


@dataclass
class CaseMeta:
    """Lightweight metadata for a case (no JAX dependency)."""

    name: str
    factory: Callable  # () -> CaseData
    grid_type: str = ""
    bus_count: int = 0
    phase: str = "1"
    voltage_level: str = ""
    source: str = ""
    description: str = ""


_REGISTRY: Dict[str, CaseMeta] = {}


def _register(
    case_id: str,
    factory: Callable,
    *,
    grid_type: str = "",
    bus_count: int = 0,
    phase: str = "1",
    voltage_level: str = "",
    source: str = "",
    description: str = "",
    aliases: Optional[List[str]] = None,
) -> None:
    meta = CaseMeta(
        name=case_id,
        factory=factory,
        grid_type=grid_type,
        bus_count=bus_count,
        phase=phase,
        voltage_level=voltage_level,
        source=source,
        description=description,
    )
    _REGISTRY[case_id] = meta
    if aliases:
        for alias in aliases:
            _REGISTRY[alias] = meta


def _populate():
    """Lazily import and register all built-in cases."""
    if _REGISTRY:
        return

    # ---- Transmission ----
    from powerzoojax.case.cases.transmission.case5 import create_case5
    _register("5", create_case5,
              grid_type="transmission", bus_count=5,
              voltage_level="HV", source="MATPOWER",
              description="IEEE 5-bus test system")

    from powerzoojax.case.cases.transmission.case14 import create_case14
    _register("14", create_case14,
              grid_type="transmission", bus_count=14,
              voltage_level="HV", source="MATPOWER",
              description="IEEE 14-bus test system")

    from powerzoojax.case.cases.transmission.case118 import create_case118
    _register("118", create_case118,
              grid_type="transmission", bus_count=118,
              voltage_level="HV", source="MATPOWER",
              description="IEEE 118-bus test system")

    from powerzoojax.case.cases.transmission.case300 import create_case300
    _register("300", create_case300,
              grid_type="transmission", bus_count=300,
              voltage_level="HV", source="MATPOWER",
              description="IEEE 300-bus test system")

    from powerzoojax.case.cases.transmission.case29gb import create_case29gb
    _register("29gb", create_case29gb,
              grid_type="transmission", bus_count=29,
              voltage_level="HV", source="custom",
              description="GB reduced 29-bus transmission network (Case29GB)")

    from powerzoojax.case.cases.transmission.case552gb import create_case552gb
    _register("552gb", create_case552gb,
              grid_type="transmission", bus_count=552,
              voltage_level="HV", source="GB",
              description="Great Britain 552-bus transmission (Case552GB)")

    from powerzoojax.case.cases.transmission.case1354pegase import create_case1354pegase
    _register("1354pegase", create_case1354pegase, aliases=["1354"],
              grid_type="transmission", bus_count=1354,
              voltage_level="HV", source="MATPOWER",
              description="PEGASE 1354-bus system")

    from powerzoojax.case.cases.transmission.case2383wp import create_case2383wp
    _register("2383wp", create_case2383wp, aliases=["2383"],
              grid_type="transmission", bus_count=2383,
              voltage_level="HV", source="MATPOWER",
              description="Polish 2383-bus winter peak")

    # ---- Distribution ----
    from powerzoojax.case.cases.distribution.case33bw import create_case33bw
    _register("33bw", create_case33bw, aliases=["33"],
              grid_type="distribution", bus_count=33,
              voltage_level="MV", source="MATPOWER",
              description="IEEE 33-bus Baran & Wu radial distribution")

    from powerzoojax.case.cases.distribution.case118zh import create_case118zh
    _register("118zh", create_case118zh,
              grid_type="distribution", bus_count=118,
              voltage_level="MV", source="MATPOWER",
              description="118-bus distribution system")

    from powerzoojax.case.cases.distribution.case123 import create_case123
    _register("123", create_case123,
              grid_type="distribution", bus_count=123, phase="3",
              voltage_level="MV", source="MATPOWER",
              description="IEEE 123-bus three-phase distribution")

    from powerzoojax.case.cases.distribution.case123_1ph import create_case123_1ph
    _register("123_1ph", create_case123_1ph,
              grid_type="distribution", bus_count=123, phase="1",
              voltage_level="MV", source="MATPOWER",
              description="IEEE 123-bus single-phase equivalent (loads merged per bus)")

    from powerzoojax.case.cases.distribution.case141 import create_case141
    _register("141", create_case141,
              grid_type="distribution", bus_count=141,
              voltage_level="MV", source="MATPOWER",
              description="141-bus distribution system")

    from powerzoojax.case.cases.distribution.case533mt_hi import create_case533mt_hi
    _register("533mt_hi", create_case533mt_hi,
              grid_type="distribution", bus_count=533,
              voltage_level="MV", source="MATPOWER",
              description="533-bus medium tension (hi)")

    from powerzoojax.case.cases.distribution.case533mt_lo import create_case533mt_lo
    _register("533mt_lo", create_case533mt_lo,
              grid_type="distribution", bus_count=533,
              voltage_level="MV", source="MATPOWER",
              description="533-bus medium tension (lo)")


def get_registry() -> Dict[str, CaseMeta]:
    _populate()
    return _REGISTRY


def get_meta(case_id: str) -> CaseMeta:
    """Get metadata for a specific case."""
    reg = get_registry()
    key = str(case_id).lower().replace("case", "")
    if key not in reg:
        raise KeyError(f"Unknown case '{case_id}'. Available: {sorted(reg.keys())}")
    return reg[key]


def list_cases(
    *,
    grid_type: Optional[str] = None,
    min_buses: Optional[int] = None,
    max_buses: Optional[int] = None,
    phase: Optional[str] = None,
    voltage_level: Optional[str] = None,
) -> List[CaseMeta]:
    """Return metadata for all cases, optionally filtered.

    Args:
        grid_type: ``"transmission"`` or ``"distribution"``.
        min_buses: Minimum bus count (inclusive).
        max_buses: Maximum bus count (inclusive).
        phase: ``"1"`` or ``"3"``.
        voltage_level: ``"HV"``, ``"MV"``, or ``"LV"``.

    Returns:
        List of :class:`CaseMeta` sorted by bus count, deduplicated (aliases excluded).
    """
    reg = get_registry()
    seen_names = set()
    results = []
    for meta in reg.values():
        if meta.name in seen_names:
            continue
        seen_names.add(meta.name)
        results.append(meta)

    if grid_type is not None:
        results = [m for m in results if m.grid_type == grid_type]
    if min_buses is not None:
        results = [m for m in results if m.bus_count >= min_buses]
    if max_buses is not None:
        results = [m for m in results if m.bus_count <= max_buses]
    if phase is not None:
        results = [m for m in results if m.phase == phase]
    if voltage_level is not None:
        results = [m for m in results if m.voltage_level == voltage_level]

    results.sort(key=lambda m: m.bus_count)
    return results
