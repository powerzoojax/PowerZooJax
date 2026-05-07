#!/usr/bin/env python
"""Build the TSO Phase-3 paper-facing cost-safety frontier.

This script is intentionally read-only with respect to experiment records: it
loads the current campaign summary and manifest, writes a derived analysis JSON,
and creates figures. It never appends to the algorithm leaderboard or manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.common.configs import load_task_config

TASK_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = TASK_DIR / "results"
SUMMARY_PATH = RESULTS_DIR / "summary" / "latest.json"
MANIFEST_PATH = RESULTS_DIR / "manifest.json"
OUT_PATH = RESULTS_DIR / "phase3_cost_safety_frontier.json"
FIGURES_DIR = RESULTS_DIR / "figures"

ALGOS = ("all_on", "merit_order", "ppo", "ppo_lagrangian")
LEARNED_ALGOS = {"ppo", "ppo_lagrangian"}
PRIMARY_SPLIT = "iid"
APPENDIX_SPLITS = ("iid", "load_stress", "line_tightening")

ALGO_LABELS = {
    "all_on": "All-On",
    "merit_order": "Merit Order",
    "ppo": "PPO",
    "ppo_lagrangian": "PPO-Lag",
}
ALGO_COLORS = {
    "all_on": "#4f46e5",
    "merit_order": "#655770",
    "ppo": "#4f92a8",
    "ppo_lagrangian": "#dc2626",
}
CLAIM_MARKERS = {
    "leaderboard_eligible_safe": "*",
    "cost_efficient_unsafe": "o",
    "conservative_safe_or_near_safe": "s",
    "hard_negative_result": "X",
}

COST_AXIS_LABEL = "Total operating cost (million GBP, lower is better)"


def _extras_for_bbox(fig: Any, *base_artists: Any) -> list[Any]:
    """Include point annotations plus figure-level artists when computing tight bbox."""
    from matplotlib.text import Annotation

    out: list[Any] = []
    out.extend([a for a in base_artists if a is not None])
    for ax in fig.axes:
        for child in ax.get_children():
            if isinstance(child, Annotation):
                out.append(child)
    return out


@dataclass(frozen=True)
class SafetyThresholds:
    reserve_shortfall_rate: float
    thermal_violation_rate: float


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_iso_utc(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _default_campaign_after(task_dir: Path) -> str | None:
    cfg = load_task_config(task_dir)
    protocol = cfg.get("benchmark_protocol") or {}
    value = protocol.get("current_campaign_start_iso")
    return str(value) if value else None


def _after_filter(record: dict[str, Any], after: str | None) -> bool:
    if not after:
        return True
    threshold = _parse_iso_utc(after)
    if threshold is None:
        raise ValueError(f"Invalid --after timestamp: {after!r}")
    timestamp = _parse_iso_utc(record.get("timestamp"))
    return timestamp is not None and timestamp >= threshold


def _as_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def _metric_bundle(row: dict[str, Any], prefix: str) -> dict[str, float | None]:
    return {
        "mean": row.get(f"{prefix}_mean"),
        "std": row.get(f"{prefix}_std"),
        "iqm": row.get(f"{prefix}_iqm"),
        "ci_lo": row.get(f"{prefix}_ci_lo"),
        "ci_hi": row.get(f"{prefix}_ci_hi"),
    }


def _is_jax_gpu(record: dict[str, Any]) -> bool:
    return (record.get("backend") or "jax_rejax") == "jax_rejax" and (
        record.get("device") or "gpu"
    ) == "gpu"


def _seed_records(
    manifest: list[dict[str, Any]],
    *,
    algo: str,
    split: str,
    after: str | None,
) -> list[dict[str, Any]]:
    records = [
        rec
        for rec in manifest
        if rec.get("task") == "tso"
        and rec.get("status", "completed") == "completed"
        and rec.get("algo") == algo
        and rec.get("split") == split
        and _is_jax_gpu(rec)
        and _after_filter(rec, after)
        and (rec.get("artifacts") or {}).get("per_episode")
    ]
    latest: dict[int, dict[str, Any]] = {}
    for rec in records:
        seed = int(rec.get("seed", -1))
        if seed < 0:
            continue
        old = latest.get(seed)
        if old is None or str(rec.get("timestamp") or "") > str(old.get("timestamp") or ""):
            latest[seed] = rec
    return [latest[seed] for seed in sorted(latest)]


def _hard_gate_pass(
    reserve_rate: float,
    thermal_rate: float,
    thresholds: SafetyThresholds,
) -> bool:
    return (
        reserve_rate <= thresholds.reserve_shortfall_rate
        and thermal_rate <= thresholds.thermal_violation_rate
    )


def _claim_status(
    *,
    algo: str,
    cost: float,
    all_on_cost: float | None,
    reserve_rate: float,
    thermal_rate: float,
    gate_pass: bool,
    learned_policy_gate_exists: bool,
    thresholds: SafetyThresholds,
) -> str:
    if gate_pass:
        return "leaderboard_eligible_safe"
    if algo in LEARNED_ALGOS and not learned_policy_gate_exists and reserve_rate <= thresholds.reserve_shortfall_rate:
        return "hard_negative_result"
    if all_on_cost is not None and cost < all_on_cost:
        return "cost_efficient_unsafe"
    return "conservative_safe_or_near_safe"


def _claim_tags(status: str, *, algo: str, gate_pass: bool, learned_policy_gate_exists: bool) -> list[str]:
    tags = [status]
    if algo in LEARNED_ALGOS and not learned_policy_gate_exists:
        tags.append("hard_negative_result")
    if gate_pass:
        tags.append("leaderboard_eligible_safe")
    return sorted(set(tags))


def build_frontier(task_dir: Path = TASK_DIR, *, after: str | None = None) -> dict[str, Any]:
    if after is None:
        after = _default_campaign_after(task_dir)

    summary = _load_json(task_dir / "results" / "summary" / "latest.json")
    manifest = _load_json(task_dir / "results" / "manifest.json")
    cfg = load_task_config(task_dir)
    safety_cfg = cfg.get("safety_thresholds") or {}
    thresholds = SafetyThresholds(
        reserve_shortfall_rate=float(safety_cfg.get("reserve_shortfall_rate", 0.0)),
        thermal_violation_rate=float(safety_cfg.get("thermal_violation_rate", 0.0)),
    )

    rows = [
        row
        for row in summary.get("rows", [])
        if row.get("algo") in ALGOS and row.get("split") in APPENDIX_SPLITS
    ]
    by_split_algo = {(row["split"], row["algo"]): row for row in rows}

    learned_policy_gate_exists = any(
        _hard_gate_pass(
            _as_float(row.get("reserve_shortfall_rate_mean")),
            _as_float(row.get("thermal_violation_rate_mean")),
            thresholds,
        )
        for row in rows
        if row.get("algo") in LEARNED_ALGOS and row.get("split") == PRIMARY_SPLIT
    )

    splits: dict[str, dict[str, Any]] = {}
    any_gate = False
    for split in APPENDIX_SPLITS:
        all_on = by_split_algo.get((split, "all_on"))
        all_on_cost = (
            _as_float(all_on.get("total_operating_cost_mean")) if all_on else None
        )
        points: list[dict[str, Any]] = []
        for algo in ALGOS:
            row = by_split_algo.get((split, algo))
            if row is None:
                continue
            cost = _as_float(row.get("total_operating_cost_mean"))
            reserve_rate = _as_float(row.get("reserve_shortfall_rate_mean"))
            thermal_rate = _as_float(row.get("thermal_violation_rate_mean"))
            gate_pass = _hard_gate_pass(reserve_rate, thermal_rate, thresholds)
            any_gate = any_gate or gate_pass
            status = _claim_status(
                algo=algo,
                cost=cost,
                all_on_cost=all_on_cost,
                reserve_rate=reserve_rate,
                thermal_rate=thermal_rate,
                gate_pass=gate_pass,
                learned_policy_gate_exists=learned_policy_gate_exists,
                thresholds=thresholds,
            )
            seed_points = []
            for rec in _seed_records(manifest, algo=algo, split=split, after=after):
                metrics = rec.get("metrics") or {}
                seed_reserve = _as_float(metrics.get("reserve_shortfall_rate"))
                seed_thermal = _as_float(metrics.get("thermal_violation_rate"))
                seed_points.append(
                    {
                        "seed": rec.get("seed"),
                        "run_id": rec.get("run_id"),
                        "total_operating_cost": metrics.get("total_operating_cost"),
                        "reserve_shortfall_rate": metrics.get("reserve_shortfall_rate"),
                        "thermal_violation_rate": metrics.get("thermal_violation_rate"),
                        "hard_safety_gate_pass": _hard_gate_pass(
                            seed_reserve,
                            seed_thermal,
                            thresholds,
                        ),
                    }
                )
            points.append(
                {
                    "algo": algo,
                    "algo_label": ALGO_LABELS.get(algo, algo),
                    "split": split,
                    "n_seeds": row.get("n_seeds"),
                    "total_operating_cost": _metric_bundle(row, "total_operating_cost"),
                    "reserve_shortfall_rate": _metric_bundle(row, "reserve_shortfall_rate"),
                    "thermal_violation_rate": _metric_bundle(row, "thermal_violation_rate"),
                    "hard_safety_gate_pass": gate_pass,
                    "claim_status": status,
                    "claim_tags": _claim_tags(
                        status,
                        algo=algo,
                        gate_pass=gate_pass,
                        learned_policy_gate_exists=learned_policy_gate_exists,
                    ),
                    "seed_points": seed_points,
                }
            )
        splits[split] = {
            "points": points,
            "n_points": len(points),
            "any_leaderboard_eligible_safe": any(p["hard_safety_gate_pass"] for p in points),
        }

    return {
        "task": "tso",
        "phase": "phase3_cost_safety_frontier",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "summary": str((task_dir / "results" / "summary" / "latest.json").relative_to(_PROJECT_ROOT)),
            "manifest": str((task_dir / "results" / "manifest.json").relative_to(_PROJECT_ROOT)),
            "campaign_after": after,
            "summary_filters": summary.get("filters"),
            "writes_manifest": False,
            "writes_algorithm_leaderboard": False,
        },
        "hard_safety_gate": {
            "reserve_shortfall_rate": thresholds.reserve_shortfall_rate,
            "thermal_violation_rate": thresholds.thermal_violation_rate,
            "definition": "reserve_shortfall_rate <= threshold and thermal_violation_rate <= threshold",
        },
        "global_claim": {
            "claim_status": "hard_negative_result" if not learned_policy_gate_exists else "leaderboard_eligible_safe",
            "any_point_leaderboard_eligible_safe": any_gate,
            "learned_policy_leaderboard_eligible_safe": learned_policy_gate_exists,
            "paper_facing_interpretation": (
                "Current TSO evidence is a hard safety frontier / negative result: "
                "PPO is cost efficient but unsafe, PPO-Lag removes reserve shortfall "
                "but retains thermal violations, and no learned policy satisfies the "
                "strict zero-violation hard gate."
            ),
        },
        "splits": splits,
    }


def _cost_x_bounds(
    points: list[dict[str, Any]],
    *,
    pad_lo_frac: float = 0.04,
    pad_hi_frac: float = 0.22,
    pad_floor_lo: float = 0.02,
    pad_floor_hi: float = 0.055,
) -> tuple[float, float]:
    """Finite x-span (million GBP) with asymmetric padding — extra margin on cost-high
    side so markers, CI caps and text offsets are less likely to clip."""

    vals: list[float] = []
    for point in points:
        cost = point.get("total_operating_cost") or {}
        for key in ("mean", "ci_lo", "ci_hi"):
            v = cost.get(key)
            if v is None:
                continue
            try:
                vals.append(float(v) / 1e6)
            except (TypeError, ValueError):
                continue
        for seed in point.get("seed_points") or []:
            seed_cost = seed.get("total_operating_cost")
            if seed_cost is None:
                continue
            try:
                vals.append(float(seed_cost) / 1e6)
            except (TypeError, ValueError):
                continue
    vals_f = [v for v in vals if v == v]
    if not vals_f:
        return 0.0, 1.0
    lo, hi = min(vals_f), max(vals_f)
    span = max(hi - lo, 1e-9)
    return (
        lo - max(pad_lo_frac * span, pad_floor_lo),
        hi + max(pad_hi_frac * span, pad_floor_hi),
    )


def _save_figure_white(
    fig: Any,
    pdf_out: Path,
    *,
    dpi_png: int = 220,
    pad_inches: float = 0.1,
    extra_artists: list[Any] | None = None,
) -> None:
    kw: dict[str, Any] = {
        "facecolor": "white",
        "edgecolor": "none",
        "bbox_inches": "tight",
        "pad_inches": pad_inches,
    }
    extra = tuple(extra_artists or [])
    if extra:
        kw["bbox_extra_artists"] = extra
    fig.savefig(pdf_out, **kw)
    fig.savefig(pdf_out.with_suffix(".png"), dpi=dpi_png, **kw)


def _plot_panel(ax: Any, points: list[dict[str, Any]], metric_key: str, *, title: str) -> None:
    ax.axhline(0.0, color="#263238", linewidth=1.0, linestyle="--", alpha=0.8)
    for point in points:
        algo = point["algo"]
        cost = point["total_operating_cost"]
        metric = point[metric_key]
        x = _as_float(cost.get("mean")) / 1e6
        y = _as_float(metric.get("mean"))
        xerr = [
            [max(0.0, x - _as_float(cost.get("ci_lo"), x * 1e6) / 1e6)],
            [max(0.0, _as_float(cost.get("ci_hi"), x * 1e6) / 1e6 - x)],
        ]
        color = ALGO_COLORS.get(algo, "#757575")
        marker = CLAIM_MARKERS.get(point["claim_status"], "o")
        for seed in point.get("seed_points", []):
            seed_cost = seed.get("total_operating_cost")
            seed_y = seed.get(metric_key)
            if seed_cost is None or seed_y is None:
                continue
            ax.scatter(
                float(seed_cost) / 1e6,
                float(seed_y),
                color=color,
                s=22,
                alpha=0.28,
                linewidths=0,
                zorder=2,
            )
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            fmt=marker,
            markersize=10 if marker != "*" else 13,
            color=color,
            markeredgecolor="#263238",
            markeredgewidth=0.7,
            capsize=4,
            linewidth=1.2,
            label=ALGO_LABELS.get(algo, algo),
            zorder=4,
        )
        ax.annotate(
            ALGO_LABELS.get(algo, algo),
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
            clip_on=False,
        )
    ax.set_title(title)
    ylabel = "Reserve shortfall" if metric_key == "reserve_shortfall_rate" else "Thermal violation"
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linestyle="-", linewidth=0.4, alpha=0.25)


def plot_frontier(frontier: dict[str, Any], task_dir: Path = TASK_DIR) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    figures_dir = task_dir / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    iid_points = frontier["splits"][PRIMARY_SPLIT]["points"]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.0, 4.06),
        sharex=True,
        layout=None,
        gridspec_kw={"wspace": 0.16},
    )
    fig.patch.set_facecolor("white")
    _plot_panel(axes[0], iid_points, "reserve_shortfall_rate", title="IID reserve frontier")
    _plot_panel(axes[1], iid_points, "thermal_violation_rate", title="IID thermal frontier")
    x_lo, x_hi = _cost_x_bounds(iid_points)
    axes[0].set_xlim(x_lo, x_hi)
    axes[1].set_xlim(x_lo, x_hi)
    axes[0].margins(x=0, y=0.12)
    axes[1].margins(x=0, y=0.12)
    fig.subplots_adjust(left=0.078, right=0.964, bottom=0.215, top=0.608, wspace=0.16)
    st = fig.suptitle(
        "TSO Phase 3 — Cost-safety frontier (hard gate: reserve=0 and thermal=0)",
        fontsize=10,
        y=0.97,
    )
    handles, labels = axes[1].get_legend_handles_labels()
    leg = fig.legend(
        handles,
        labels,
        ncol=4,
        fontsize=10,
        frameon=False,
        handlelength=2.65,
        handletextpad=0.62,
        columnspacing=1.12,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.805),
        bbox_transform=fig.transFigure,
    )
    xl = fig.text(0.5, 0.092, COST_AXIS_LABEL, fontsize=10, ha="center", va="bottom")
    out = figures_dir / "phase3_cost_safety_frontier.pdf"
    _save_figure_white(
        fig,
        out,
        pad_inches=0.22,
        extra_artists=_extras_for_bbox(fig, st, xl, leg),
    )
    plt.close(fig)
    outputs.append(out)

    fig, axes = plt.subplots(
        len(APPENDIX_SPLITS),
        2,
        figsize=(10.0, 9.08),
        sharex=False,
        layout=None,
        gridspec_kw={"wspace": 0.16, "hspace": 0.34},
    )
    fig.patch.set_facecolor("white")
    for row_idx, split in enumerate(APPENDIX_SPLITS):
        points = frontier["splits"][split]["points"]
        _plot_panel(
            axes[row_idx, 0],
            points,
            "reserve_shortfall_rate",
            title=f"{split}: reserve",
        )
        _plot_panel(
            axes[row_idx, 1],
            points,
            "thermal_violation_rate",
            title=f"{split}: thermal",
        )
        xr_lo, xr_hi = _cost_x_bounds(points)
        axes[row_idx, 0].set_xlim(xr_lo, xr_hi)
        axes[row_idx, 1].set_xlim(xr_lo, xr_hi)
    for ax in axes.ravel():
        ax.margins(x=0, y=0.12)
    handles, labels = axes[-1, 1].get_legend_handles_labels()
    fig.subplots_adjust(left=0.079, right=0.965, bottom=0.202, top=0.878, hspace=0.34, wspace=0.16)
    st2 = fig.suptitle(
        "TSO Phase 3 appendix — Cost-safety frontier across IID and stress splits",
        fontsize=10,
        y=0.966,
    )
    leg2 = fig.legend(
        handles,
        labels,
        ncol=4,
        fontsize=10,
        frameon=False,
        handlelength=2.65,
        handletextpad=0.62,
        columnspacing=1.12,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.132),
        bbox_transform=fig.transFigure,
    )
    xl2 = fig.text(0.5, 0.068, COST_AXIS_LABEL, fontsize=10, ha="center", va="bottom")
    out = figures_dir / "phase3_cost_safety_frontier_appendix.pdf"
    _save_figure_white(
        fig,
        out,
        pad_inches=0.22,
        extra_artists=_extras_for_bbox(fig, st2, xl2, leg2),
    )
    plt.close(fig)
    outputs.append(out)

    return outputs


def validate_frontier(frontier: dict[str, Any], task_dir: Path = TASK_DIR) -> list[str]:
    errors: list[str] = []
    if frontier.get("task") != "tso":
        errors.append("task must be tso")
    if frontier.get("phase") != "phase3_cost_safety_frontier":
        errors.append("phase must be phase3_cost_safety_frontier")
    for split in APPENDIX_SPLITS:
        split_data = (frontier.get("splits") or {}).get(split)
        if not split_data:
            errors.append(f"missing split {split}")
            continue
        algos = {point.get("algo") for point in split_data.get("points", [])}
        missing = set(ALGOS) - algos
        if missing:
            errors.append(f"{split} missing algos: {sorted(missing)}")
        for point in split_data.get("points", []):
            for key in (
                "total_operating_cost",
                "reserve_shortfall_rate",
                "thermal_violation_rate",
                "hard_safety_gate_pass",
                "claim_status",
                "seed_points",
            ):
                if key not in point:
                    errors.append(f"{split}/{point.get('algo')} missing {key}")
            if point.get("claim_status") not in CLAIM_MARKERS:
                errors.append(f"{split}/{point.get('algo')} has invalid claim_status")
            if int(point.get("n_seeds") or 0) < 5:
                errors.append(f"{split}/{point.get('algo')} has n_seeds < 5")
    for rel in (
        "phase3_cost_safety_frontier.json",
        "figures/phase3_cost_safety_frontier.pdf",
        "figures/phase3_cost_safety_frontier.png",
        "figures/phase3_cost_safety_frontier_appendix.pdf",
        "figures/phase3_cost_safety_frontier_appendix.png",
    ):
        path = task_dir / "results" / rel
        if not path.exists() or path.stat().st_size <= 0:
            errors.append(f"missing or empty output: {path}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", default=str(TASK_DIR))
    parser.add_argument("--after", default=None, help="ISO campaign-start timestamp filter")
    parser.add_argument("--check", action="store_true", help="Validate generated fields and files")
    args = parser.parse_args()

    task_dir = Path(args.task_dir)
    frontier = build_frontier(task_dir, after=args.after)
    out_path = task_dir / "results" / "phase3_cost_safety_frontier.json"
    out_path.write_text(json.dumps(frontier, indent=2, sort_keys=True), encoding="utf-8")
    figures = plot_frontier(frontier, task_dir)

    print(f"[TSO phase3] wrote {out_path}")
    for figure in figures:
        print(f"[TSO phase3] wrote {figure} and {figure.with_suffix('.png')}")

    if args.check:
        errors = validate_frontier(frontier, task_dir)
        if errors:
            for error in errors:
                print(f"[TSO phase3] ERROR: {error}", file=sys.stderr)
            raise SystemExit(2)
        print("[TSO phase3] validation passed")


if __name__ == "__main__":
    main()
