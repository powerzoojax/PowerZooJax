"""Built-in case library for PowerZooJax.

This package is the setup-time entrypoint for benchmark cases. It provides:
- `CaseData`, the JAX-native array container used by env params
- a registry-backed `load_case()` / `list_cases()` interface
- built-in transmission and distribution cases used by the benchmark tasks
- matrix helpers such as PTDF and graph-derived case matrices
- CPU-side inspection and plotting helpers for offline analysis

The case layer is intentionally separate from the env layer: load and inspect
cases here, then pass the resulting `CaseData` into `make_*_params(...)`
factories under `powerzoojax.envs` or `powerzoojax.tasks`.

Quick start:
    >>> from powerzoojax.case import load_case, list_cases
    >>> case = load_case("5")
    >>> metas = list_cases(grid_type="distribution")
"""

import warnings

# Core data structure (GPU)
from powerzoojax.case.case_data import CaseData, validate_case_data

# Matrix computations
from powerzoojax.case.case_matrices import (
    compute_ptdf,
    compute_adjacency_matrix,
    compute_degree_matrix,
    compute_laplacian_matrix,
    build_case_matrices,
)

# Built-in cases — aggregated from transmission/ + distribution/
from powerzoojax.case.cases import (
    create_case5,
    create_case14,
    create_case33bw,
    create_case118,
    create_case118zh,
    create_case123,
    create_case123_1ph,
    create_case141,
    create_case300,
    create_case533mt_hi,
    create_case533mt_lo,
    create_case1354pegase,
    create_case2383wp,
    create_case29gb,
    create_case552gb,
)

# CPU utilities (printing, plotting)
from powerzoojax.case.case_info import CaseInfo, print_case, print_summary
from powerzoojax.case.case_plotter import CasePlotter, plot_case

# Case adapter (for converting raw cases)
from powerzoojax.case.case_adapter import case_to_jax, convert_case

# Registry
from powerzoojax.case._registry import CaseMeta, list_cases, get_meta


def load_case(case_id: str = "5", *, grid_type: str = None) -> CaseData:
    """Load a built-in case by ID.

    Mirrors PowerZoo's ``load_case()`` API.  If *grid_type* is given and
    the loaded case's metadata does not match, a :class:`UserWarning` is
    emitted (same behaviour as PowerZoo).

    Args:
        case_id: Case identifier (e.g. ``"5"``, ``"33bw"``, ``"118"``).
        grid_type: Optional ``"transmission"`` or ``"distribution"``.

    Returns:
        CaseData with all arrays populated.
    """
    from powerzoojax.case._registry import get_registry

    key = str(case_id).lower().replace("case", "")
    reg = get_registry()

    if key not in reg:
        available = sorted(set(reg.keys()))
        raise ValueError(f"Unknown case '{case_id}'. Available: {available}")

    meta = reg[key]

    if grid_type and meta.grid_type and meta.grid_type != grid_type:
        warnings.warn(
            f"Case '{meta.name}' has grid_type='{meta.grid_type}', "
            f"but grid_type='{grid_type}' was requested.",
            UserWarning,
            stacklevel=2,
        )

    return meta.factory()


__all__ = [
    # Core
    "CaseData",
    "validate_case_data",
    # Matrix computations
    "compute_ptdf",
    "compute_adjacency_matrix",
    "compute_degree_matrix",
    "compute_laplacian_matrix",
    "build_case_matrices",
    # Built-in cases (native JAX)
    "create_case5",
    "create_case14",
    "create_case33bw",
    "create_case118",
    "create_case118zh",
    "create_case123",
    "create_case123_1ph",
    "create_case141",
    "create_case300",
    "create_case533mt_hi",
    "create_case533mt_lo",
    "create_case1354pegase",
    "create_case2383wp",
    "create_case29gb",
    "create_case552gb",
    "load_case",
    "list_cases",
    "get_meta",
    "CaseMeta",
    # CPU utilities
    "CaseInfo",
    "print_case",
    "print_summary",
    "CasePlotter",
    "plot_case",
    # Case adapter (raw -> JAX)
    "case_to_jax",
    "convert_case",
]
