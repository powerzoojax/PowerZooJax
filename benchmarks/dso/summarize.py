"""DSO results summarisation.

Reads all RunRecords from the manifest, aggregates across seeds, computes
paper-required metrics, and writes results/summary/latest.json.

Metrics produced
----------------
Per (algo, split) row:
  network_loss_reduction_pct   — primary objective (mean ± std across seeds)
  served_flexible_demand_ratio — shift_in / shift_out (buffer clearance)
  peak_shaving_pct             — relative peak reduction vs no-control
  voltage_violation_rate       — total_violations / max_steps (per step)
  norm_score                   — (nc_loss - algo_loss) / (nc_loss - droop_loss)
                                  [0 = no_control level, 1 = droop level, >1 beats droop]

Per-algo summary:
  drift_tracking_gap           — only populated if a future configured OOD
                                  split is present alongside iid.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
DEFAULT_TASK_DIR = Path(__file__).resolve().parent

from benchmarks.common.configs import load_task_config
from benchmarks.common.io import (
    has_training_artifact,
    load_manifest_filtered,
    save_summary,
)
from benchmarks.common.protocol import build_protocol_status
from benchmarks.common.reporting import summarize_group_evidence
from benchmarks.common.stats import (
    aggregate_seeds,
    compute_norm_score_with_status,
    fmt_metric,
    get_row_mean,
    metric_sensitivity_report,
    numeric_only,
)

# From the frozen task config — fixed for DSO benchmark.
_MAX_STEPS = 48
_PRIMARY_TEST_BASELINE = "no_control"
_PRIMARY_TEST_LEARNERS = ("ppo", "sac", "saute_ppo", "ppo_lagrangian")


def _summary_task_config(task_dir: Path) -> dict:
    try:
        return load_task_config(task_dir)
    except FileNotFoundError:
        return {
            "primary_split": "iid",
            "seeds": [],
            "benchmark_protocol": {
                "protocol_formalization_pending": True,
                "submission_min_seeds": 5,
                "hypothesis_test": {"status": "required_not_yet_implemented"},
                "leaderboard": {
                    "split": "iid",
                    "primary_metric": "total_loss_mwh",
                    "direction": "lower_is_better",
                },
            },
        }


def _aggregate_rows(records, *, by_device: bool) -> list[dict]:
    groups: dict[tuple[str, ...], list[dict]] = {}
    group_records: dict[tuple[str, ...], list] = {}
    for r in records:
        backend = r.backend or "jax_rejax"
        device = r.device or "gpu"
        if by_device:
            key = (backend, device, r.algo, r.split)
        else:
            key = (r.algo, r.split)
        groups.setdefault(key, []).append(numeric_only(r.metrics))
        group_records.setdefault(key, []).append(r)

    rows: list[dict] = []
    for key, metrics_list in sorted(groups.items()):
        agg = aggregate_seeds(metrics_list)
        if by_device:
            backend, device, algo, split = key
            row: dict = {
                "backend": backend,
                "device": device,
                "algo": algo,
                "split": split,
                "n_seeds": len(metrics_list),
            }
        else:
            algo, split = key
            row = {"algo": algo, "split": split, "n_seeds": len(metrics_list)}
        row.update(summarize_group_evidence(group_records.get(key, [])))
        for metric_name, stats in agg.items():
            row[f"{metric_name}_mean"] = stats["mean"]
            row[f"{metric_name}_std"] = stats["std"]
            row[f"{metric_name}_iqm"] = stats["iqm"]
            row[f"{metric_name}_ci_lo"] = stats["ci_lo"]
            row[f"{metric_name}_ci_hi"] = stats["ci_hi"]

        if "total_voltage_violations_mean" in row:
            row["voltage_violation_count_per_step_mean"] = (
                row["total_voltage_violations_mean"] / _MAX_STEPS
            )
            row["voltage_violation_count_per_step_std"] = (
                row.get("total_voltage_violations_std", 0.0) / _MAX_STEPS
            )
        if "total_thermal_overloads_mean" in row:
            row["thermal_overload_count_per_step_mean"] = (
                row["total_thermal_overloads_mean"] / _MAX_STEPS
            )
            row["thermal_overload_count_per_step_std"] = (
                row.get("total_thermal_overloads_std", 0.0) / _MAX_STEPS
            )
        if "served_flex_ratio_mean" in row:
            row["served_flexible_demand_ratio_mean"] = row["served_flex_ratio_mean"]
            row["served_flexible_demand_ratio_std"] = row.get(
                "served_flex_ratio_std", 0.0
            )
        rows.append(row)
    return rows


def _attach_normscore(rows: list[dict], *, by_device: bool) -> None:
    for row in rows:
        split = row["split"]
        if by_device:
            prefix = {
                "backend": row["backend"],
                "device": row["device"],
            }

            def _metric(algo: str) -> float | None:
                return next(
                    (
                        r.get("total_loss_mwh_mean")
                        for r in rows
                        if r.get("backend") == prefix["backend"]
                        and r.get("device") == prefix["device"]
                        and r["algo"] == algo
                        and r["split"] == split
                    ),
                    None,
                )
        else:
            _metric = lambda algo: get_row_mean(rows, algo, split, "total_loss_mwh_mean")

        nc_loss = _metric("no_control")
        droop_loss = _metric("droop")
        algo_loss = row.get("total_loss_mwh_mean")
        row.update(
            compute_norm_score_with_status(
                value=algo_loss,
                anchor_a=nc_loss,
                anchor_b=droop_loss,
                anchor_metric="total_loss_mwh",
                metric_direction="lower_is_better",
            )
        )

    if by_device:
        all_keys = sorted({(r["backend"], r["device"], r["algo"]) for r in rows})
        for backend_label, device_label, algo in all_keys:
            iid_row = next(
                (
                    r for r in rows
                    if r["backend"] == backend_label
                    and r["device"] == device_label
                    and r["algo"] == algo
                    and r["split"] == "iid"
                ),
                None,
            )
            ood_split = next(
                (
                    r["split"]
                    for r in rows
                    if r["backend"] == backend_label
                    and r["device"] == device_label
                    and r["algo"] == algo
                    and r["split"] not in {"iid", "train"}
                ),
                None,
            )
            ood_row = next(
                (
                    r for r in rows
                    if r["backend"] == backend_label
                    and r["device"] == device_label
                    and r["algo"] == algo
                    and r["split"] == ood_split
                ),
                None,
            )
            gap = (
                iid_row["norm_score"] - ood_row["norm_score"]
                if iid_row is not None
                and ood_row is not None
                and iid_row.get("norm_score") is not None
                and ood_row.get("norm_score") is not None
                else None
            )
            for row in rows:
                if (
                    row.get("backend") == backend_label
                    and row.get("device") == device_label
                    and row["algo"] == algo
                ):
                    row["drift_tracking_gap"] = gap
        return

    all_algos = sorted({r["algo"] for r in rows})
    algo_gaps: dict[str, float | None] = {}
    for algo in all_algos:
        iid_row = next(
            (r for r in rows if r["algo"] == algo and r["split"] == "iid"), None
        )
        ood_split = next(
            (
                r["split"]
                for r in rows
                if r["algo"] == algo and r["split"] not in {"iid", "train"}
            ),
            None,
        )
        ood_row = next(
            (r for r in rows if r["algo"] == algo and r["split"] == ood_split), None
        )
        if (
            iid_row is not None
            and ood_row is not None
            and iid_row.get("norm_score") is not None
            and ood_row.get("norm_score") is not None
        ):
            algo_gaps[algo] = iid_row["norm_score"] - ood_row["norm_score"]
        else:
            algo_gaps[algo] = None
    for row in rows:
        row["drift_tracking_gap"] = algo_gaps.get(row["algo"])


def _collect_seed_metric(
    records,
    *,
    algo: str,
    split: str,
    metric: str,
) -> dict[int, float]:
    values: dict[int, float] = {}
    for r in records:
        if r.algo != algo or r.split != split:
            continue
        if (r.backend or "jax_rejax") != "jax_rejax":
            continue
        if (r.device or "gpu") != "gpu":
            continue
        value = (r.metrics or {}).get(metric)
        if value is None:
            continue
        try:
            f_value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f_value):
            values[int(r.seed)] = f_value
    return values


def _collect_seed_episodes(
    records,
    *,
    algo: str,
    split: str,
    metric: str,
    task_dir: Path,
) -> dict[int, np.ndarray]:
    """Per (algo, seed) load the per_episode artifact and return the metric vector.

    Restricted to canonical jax_rejax/gpu rows so that the resulting test
    pairs episodes from the same backend the headline metric is reported on.
    """
    out: dict[int, np.ndarray] = {}
    for r in records:
        if r.algo != algo or r.split != split:
            continue
        if (r.backend or "jax_rejax") != "jax_rejax":
            continue
        if (r.device or "gpu") != "gpu":
            continue
        rel = (r.artifacts or {}).get("per_episode")
        if not rel:
            continue
        # Artifact paths in the manifest are relative to the per-task results
        # directory (e.g. "artifacts/<run_id>_per_episode.json"); resolve via
        # task_dir/results first, then fall back to task_dir for legacy layouts.
        for candidate in (task_dir / "results" / rel, task_dir / rel):
            if candidate.exists():
                path = candidate
                break
        else:
            continue
        try:
            episodes = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        values = []
        for ep in episodes:
            v = ep.get(metric)
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(fv):
                values.append(fv)
        if values:
            out[int(r.seed)] = np.asarray(values, dtype=np.float64)
    return out


def _paired_signflip_permutation_test(
    left: np.ndarray,
    right: np.ndarray,
    *,
    lower_is_better: bool,
    n_perm: int = 10000,
    rng_seed: int = 42,
) -> dict[str, Any]:
    """Two-sided paired sign-flip test on seed-level metric deltas."""
    diff = left - right
    n = int(diff.size)
    if n == 0:
        return {
            "test": "paired_signflip_permutation",
            "n_pairs": 0,
            "mean_diff": None,
            "improvement_mean": None,
            "p_value_two_sided": None,
        }

    observed = float(np.mean(diff))
    if n <= 20:
        bits = np.arange(2**n, dtype=np.uint64)[:, None]
        masks = (bits >> np.arange(n, dtype=np.uint64)) & 1
        signs = masks.astype(np.float64) * 2.0 - 1.0
        perm_stats = np.mean(signs * diff[None, :], axis=1)
    else:
        rng = np.random.default_rng(rng_seed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, n))
        perm_stats = np.mean(signs * diff[None, :], axis=1)
    p_value = float(np.mean(np.abs(perm_stats) >= abs(observed)))
    improvement = -observed if lower_is_better else observed
    return {
        "test": "paired_signflip_permutation",
        "n_pairs": n,
        "mean_diff": observed,
        "improvement_mean": float(improvement),
        "p_value_two_sided": p_value,
    }


def _build_hypothesis_tests(
    records,
    task_cfg: dict[str, Any],
    *,
    task_dir: Path,
) -> list[dict[str, Any]]:
    protocol = task_cfg.get("benchmark_protocol") or {}
    leaderboard = protocol.get("leaderboard") or {}
    split = str(leaderboard.get("split") or task_cfg.get("primary_split", "iid"))
    metric = str(leaderboard.get("primary_metric") or "total_loss_mwh")
    direction = str(leaderboard.get("direction") or "lower_is_better")
    lower_is_better = direction == "lower_is_better"
    baseline_eps = _collect_seed_episodes(
        records,
        algo=_PRIMARY_TEST_BASELINE,
        split=split,
        metric=metric,
        task_dir=task_dir,
    )
    tests: list[dict[str, Any]] = []
    for algo in _PRIMARY_TEST_LEARNERS:
        learner_eps = _collect_seed_episodes(
            records, algo=algo, split=split, metric=metric, task_dir=task_dir,
        )
        seeds = sorted(set(baseline_eps) & set(learner_eps))
        if not seeds:
            continue
        # Pair on (seed, episode_idx); skip seeds whose episode counts disagree.
        paired_seeds: list[int] = []
        left_list: list[np.ndarray] = []
        right_list: list[np.ndarray] = []
        for s in seeds:
            n = min(learner_eps[s].size, baseline_eps[s].size)
            if n == 0:
                continue
            paired_seeds.append(s)
            left_list.append(learner_eps[s][:n])
            right_list.append(baseline_eps[s][:n])
        if not paired_seeds:
            continue
        left = np.concatenate(left_list)
        right = np.concatenate(right_list)
        test = _paired_signflip_permutation_test(
            left,
            right,
            lower_is_better=lower_is_better,
            n_perm=20000,
        )
        test.update(
            {
                "algo": algo,
                "baseline_algo": _PRIMARY_TEST_BASELINE,
                "split": split,
                "metric": metric,
                "metric_direction": direction,
                "paired_seeds": paired_seeds,
                "pairing": "episode",
                "n_episodes_per_seed": int(left.size // len(paired_seeds))
                if paired_seeds and left.size % len(paired_seeds) == 0
                else None,
            }
        )
        tests.append(test)
    return tests


def summarize_dso(
    task_dir: Path,
    *,
    after: str | None = None,
    backend: str = "jax_rejax",
    device: str = "gpu",
) -> dict:
    """Aggregate DSO results and save summary JSON; print paper comparison table."""
    all_records = load_manifest_filtered(task_dir, after=after)
    records = [
        r
        for r in all_records
        if (r.backend or "jax_rejax") == backend and (r.device or "gpu") == device
    ]
    if not records:
        print("[DSO summarize] No records found in manifest.")
        return {}

    all_records = [
        r for r in all_records
        if not (r.split == "train" and has_training_artifact(r.artifacts))
    ]
    records = [
        r for r in records
        if not (r.split == "train" and has_training_artifact(r.artifacts))
    ]

    rows = _aggregate_rows(records, by_device=False)
    _attach_normscore(rows, by_device=False)
    phase2_rows = _aggregate_rows(all_records, by_device=True)
    _attach_normscore(phase2_rows, by_device=True)
    all_algos = sorted({r["algo"] for r in rows})
    task_cfg = _summary_task_config(task_dir)
    hypothesis_tests = _build_hypothesis_tests(records, task_cfg, task_dir=task_dir)
    protocol_status = build_protocol_status(
        rows,
        task_cfg,
        default_primary_metric="total_loss_mwh",
        default_direction="lower_is_better",
    )

    summary = {
        "task": "dso",
        "rows": rows,
        "phase2_rows": phase2_rows,
        "hypothesis_tests": hypothesis_tests,
        "protocol_status": protocol_status,
        "reporting": {
            "suppressed_rows": [],
            "legacy_default_evidence_tier": "legacy_unknown",
            "cross_backend_gate": "comparison_audited=true and audit_suite_version is required",
        },
        "metric_sensitivity": metric_sensitivity_report(
            [row for row in rows if row.get("split") == task_cfg.get("primary_split", "iid")],
            primary_metric_key="total_loss_mwh_mean",
            primary_direction="lower_is_better",
        ),
        "filters": {
            "after": after,
            "backend": backend,
            "device": device,
        },
    }
    path = save_summary(summary, task_dir)
    print(f"[DSO summarize] {len(rows)} entries -> {path}")

    header = (
        "| {:<18} | {:<12} | {:>10} | {:>11} | {:>11} | {:>9} | {:>9} | {:>9} |"
    ).format(
        "Algo", "Split",
        "Loss Red %", "Served Flex", "Peak Shav %",
        "Volt/step", "NormScore", "Drift Gap",
    )
    sep = "|" + "-" * 20 + "|" + "-" * 14 + "|" + ("-" * 12 + "|") * 6
    print()
    print(header)
    print(sep)

    for row in rows:
        print(
            "| {:<18} | {:<12} | {:>10} | {:>11} | {:>11} | {:>9} | {:>9} | {:>9} |".format(
                row["algo"],
                row["split"],
                fmt_metric(row.get("network_loss_reduction_pct_mean"), ".2f"),
                fmt_metric(row.get("served_flexible_demand_ratio_mean"), ".3f"),
                fmt_metric(row.get("peak_shaving_pct_mean"), ".2f"),
                fmt_metric(row.get("voltage_violation_count_per_step_mean"), ".3f"),
                fmt_metric(row.get("norm_score"), ".3f"),
                fmt_metric(row.get("drift_tracking_gap"), ".3f"),
            )
        )

    # Print per-algo drift_tracking_gap summary if a configured OOD split exists.
    print()
    print("drift_tracking_gap by algo (requires iid plus a configured OOD split):")
    for algo in all_algos:
        gap = next(
            (row.get("drift_tracking_gap") for row in rows if row["algo"] == algo),
            None,
        )
        gap_str = f"{gap:.3f}" if gap is not None else "N/A"
        print(f"  {algo:<20} {gap_str}")

    proto_mark = "READY" if protocol_status["current_campaign_submission_ready"] else "NOT READY"
    print(
        "submission protocol: "
        f"{proto_mark} "
        f"(observed min seeds={protocol_status['observed_min_seed_count']}, "
        f"required={protocol_status['submission_min_seeds']}, "
        f"hypothesis_test={protocol_status['hypothesis_test'].get('status', 'unspecified')})"
    )
    if hypothesis_tests:
        print("primary-split paired hypothesis tests:")
        for test in hypothesis_tests:
            print(
                "  {algo} vs {baseline_algo}: n_pairs={n_pairs} "
                "improvement={improvement_mean} p={p_value_two_sided}".format(**test)
            )

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", default=DEFAULT_TASK_DIR, type=Path)
    parser.add_argument("--after", default=None)
    parser.add_argument("--backend", default="jax_rejax")
    parser.add_argument("--device", default="gpu")
    args = parser.parse_args()
    summarize_dso(
        args.task_dir,
        after=args.after,
        backend=args.backend,
        device=args.device,
    )
