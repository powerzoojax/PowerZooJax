#!/usr/bin/env python
"""Plot DC Microgrid extended execution-scaling (nenv 16-256).

Reads ``results/scaling/execution_scaling.json`` (which contains both
"matched" suite rows for 16/32/64 and "jax-extended" rows for 64/128/256)
and produces a single matched-range figure covering all five nenv values.

Outputs:
  - materials/figures/dc_microgrid_execution_scaling_matched_range.{pdf,png}
  - benchmarks/dc_microgrid/results/scaling/scaling_results_extended.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.ticker as mticker

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

TASK_DIR = _PROJECT_ROOT / "benchmarks" / "dc_microgrid"
DEFAULT_INPUT = TASK_DIR / "results" / "scaling" / "execution_scaling.json"
MATERIALS_FIGURE_DIR = _PROJECT_ROOT / "BenchmarkPaper" / "materials" / "figures"

JAX_COLOR = "#3B7DD8"
SB3_COLOR = "#E8A33D"
SBX_COLOR = "#D87FB6"
GRID_COLOR = "#d5dbe3"


def _figure_rc() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "grid.linewidth": 0.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _format_k(value: float, _pos: int | None = None) -> str:
    if abs(value) >= 1000:
        return f"{value / 1000:g}K"
    return f"{value:g}"


def _load_all_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        r for r in payload.get("results", [])
        if r.get("status") == "completed"
        and r.get("steady_state_env_steps_per_second") is not None
    ]


def _group_mean(rows: list[dict]) -> dict[tuple, float]:
    """Return {(backend, nenv): mean_sps} over seeds.

    Priority rules (to avoid mixing measurements from different sessions):
    - SB3: only matched/all suite rows
    - JAX nenv in {16,32,64}: prefer matched suite rows only
    - JAX nenv in {128,256}: prefer jax-extended suite rows, fall back to matched
    """
    # Separate JAX by suite preference
    jax_extended: dict[int, list[float]] = defaultdict(list)
    jax_matched: dict[int, list[float]] = defaultdict(list)
    sb3_data: dict[int, list[float]] = defaultdict(list)

    for r in rows:
        backend = r.get("backend", "jax_rejax")
        nenv = int(r["nenv"])
        sps = float(r["steady_state_env_steps_per_second"])
        suite = r.get("suite", "matched")

        if backend in ("sb3", "sbx"):
            if suite in ("matched", "all"):
                sb3_data[nenv].append(sps)
        elif backend == "jax_rejax":
            if suite == "jax-extended":
                jax_extended[nenv].append(sps)
            elif suite in ("matched", "all"):
                jax_matched[nenv].append(sps)

    out: dict[tuple, float] = {}
    # JAX: prefer jax-extended for 128/256, matched for 16/32/64
    all_jax_nenvs = set(jax_extended) | set(jax_matched)
    for nenv in all_jax_nenvs:
        if nenv >= 128 and jax_extended.get(nenv):
            out[("jax_rejax", nenv)] = mean(jax_extended[nenv])
        elif jax_matched.get(nenv):
            out[("jax_rejax", nenv)] = mean(jax_matched[nenv])
        elif jax_extended.get(nenv):
            out[("jax_rejax", nenv)] = mean(jax_extended[nenv])
    # SB3
    for nenv, vs in sb3_data.items():
        out[("sb3", nenv)] = mean(vs)
    return out


def _write_extended_csv(rows: list[dict], output_dir: Path) -> Path:
    """Write scaling_results_extended.csv per task spec."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "scaling_results_extended.csv"
    fieldnames = ["backend", "nenv", "steps_per_sec", "run_seconds",
                  "warmup_seconds", "status"]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            row_out = {
                "backend": r.get("backend"),
                "nenv": r.get("nenv"),
                "steps_per_sec": r.get("steady_state_env_steps_per_second"),
                "run_seconds": r.get("steady_state_time_s"),
                "warmup_seconds": r.get("warmup_time_s"),
                "status": r.get("status"),
            }
            writer.writerow(row_out)
    return csv_path


def plot_extended(rows: list[dict], figure_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    _figure_rc()

    data = _group_mean(rows)
    all_backends = sorted({b for b, _ in data}, key=lambda b: (0 if b == "jax_rejax" else 1))

    COLORS = {"jax_rejax": JAX_COLOR, "sb3": SB3_COLOR, "sbx": SBX_COLOR}
    LABELS = {
        "jax_rejax": "PowerZooJax (JAX-GPU)",
        "sb3": "SB3 (CPU)",
        "sbx": "SBX (GPU)",
    }
    MARKERS = {"jax_rejax": "o", "sb3": "s", "sbx": "^"}

    fig, ax = plt.subplots(figsize=(10.0, 3.6), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for backend in all_backends:
        pts = sorted((nenv, sps) for (b, nenv), sps in data.items() if b == backend)
        if not pts:
            continue
        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            marker=MARKERS.get(backend, "o"),
            linewidth=2.0,
            color=COLORS.get(backend, "gray"),
            label=LABELS.get(backend, backend),
        )
        nenv, sps = pts[-1]
        ax.annotate(
            f"{sps / 1000:.0f}K",
            (nenv, sps),
            textcoords="offset points",
            xytext=(5, 0),
            va="center",
            color=COLORS.get(backend, "gray"),
        )

    ax.set_xlabel("Parallel environments (nenv)")
    ax.set_ylabel("Steady-state env steps / second")
    ax.set_title("DC Microgrid execution scaling (nenv 16-256)", loc="left")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_format_k))
    ax.grid(axis="y", color=GRID_COLOR, alpha=0.35, linewidth=0.5)
    ax.margins(x=0.05, y=0.12)
    ax.legend(frameon=False, loc="upper left")

    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        figure_dir / "dc_microgrid_execution_scaling_matched_range.pdf",
        figure_dir / "dc_microgrid_execution_scaling_matched_range.png",
    ]
    for p in paths:
        fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--figure-dir", default=str(MATERIALS_FIGURE_DIR))
    parser.add_argument("--scaling-dir", default=str(TASK_DIR / "results" / "scaling"))
    args = parser.parse_args(argv)

    rows = _load_all_rows(Path(args.input))
    print(f"[dc_mg_ext_plot] loaded {len(rows)} completed rows")

    # Write extended CSV
    csv_path = _write_extended_csv(rows, Path(args.scaling_dir))
    print(f"[dc_mg_ext_plot] wrote {csv_path}")

    # Write figures
    paths = plot_extended(rows, Path(args.figure_dir))
    for p in paths:
        print(f"[dc_mg_ext_plot] wrote {p}")


if __name__ == "__main__":
    main()
