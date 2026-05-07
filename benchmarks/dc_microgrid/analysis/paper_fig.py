"""Build Fig 6.6 (DC Microgrid 24h dispatch) for the PowerZooJax paper.

The canonical builder lives at
``BenchmarkPaper/materials/scripts/build_dcmg_paper_figure.py``: it produces
the 2-panel figure with stacked grid/PV/battery supply (a) and SOC + GB grid
price (b) on SAC IID episode 5. This module exists only so the per-task
``paper_fig.py`` invocation pattern (``python benchmarks/<task>/analysis/paper_fig.py``)
keeps working for DC Microgrid; it is a thin wrapper around the canonical
builder.

Run:
    python benchmarks/dc_microgrid/analysis/paper_fig.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "BenchmarkPaper" / "materials" / "scripts"))

from build_dcmg_paper_figure import build_figure  # noqa: E402


def main() -> None:
    build_figure()


if __name__ == "__main__":
    main()
