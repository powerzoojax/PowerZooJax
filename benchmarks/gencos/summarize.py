"""GenCos results summarisation.

Reads all RunRecords from the manifest, aggregates across seeds, computes
paper-required metrics, and writes results/summary/latest.json.

Metrics produced
----------------
Per (algo, split) row:
  total_profit_mean / _std      — sum of all agents' dispatch profit per episode
  mean_lmp_mean                 — average locational marginal price [$/MWh]
  price_volatility_mean         — CV (std/mean) of mean-LMP time series
  hhi_mean                      — Herfindahl-Hirschman Index of dispatch shares
  total_gen_cost_mean           — sum of generation costs per episode [$/MWh·h]
  sced_convergence_rate_mean    — fraction of steps with converged SCED
  ramp_binding_rate_mean        — fraction of ramp-constrained steps
  norm_score                    — (algo_profit − truthful_profit) /
                                  (max_markup_profit − truthful_profit)
                                  [0=competitive, 1=max-markup, >1=collusion]

Per-algo summary:
  drift_tracking_gap            — norm_score(iid) − norm_score(demand_shift)
                                  Positive = more OOD-robust profit.
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

_OOD_SPLIT = "demand_shift"
DEFAULT_TASK_DIR = Path(__file__).resolve().parent
_PERMUTATION_N = 20000
_PERMUTATION_SEED = 42
_PRIMARY_TEST_LEARNERS = ("ippo",)
_PRIMARY_TEST_BASELINES = ("truthful", "uniform_mid", "max_markup")


def _aggregate_rows(records, *, by_device: bool) -> list[dict]:
    keyed_records: dict[tuple[str, ...], dict[int, Any]] = {}
    for r in records:
        backend = r.backend or "jax_rejax"
        device = r.device or "gpu"
        key = (backend, device, r.algo, r.split) if by_device else (r.algo, r.split)
        by_seed = keyed_records.setdefault(key, {})
        previous = by_seed.get(int(r.seed))
        if previous is None or (r.timestamp, r.run_id) > (previous.timestamp, previous.run_id):
            by_seed[int(r.seed)] = r

    rows: list[dict] = []
    for key, seed_records in sorted(keyed_records.items()):
        chosen_records = [seed_records[s] for s in sorted(seed_records)]
        metrics_list = [numeric_only(r.metrics) for r in chosen_records]
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
        row.update(summarize_group_evidence(chosen_records))
        for metric_name, stats in agg.items():
            row[f"{metric_name}_mean"] = stats["mean"]
            row[f"{metric_name}_std"] = stats["std"]
            row[f"{metric_name}_iqm"] = stats["iqm"]
            row[f"{metric_name}_ci_lo"] = stats["ci_lo"]
            row[f"{metric_name}_ci_hi"] = stats["ci_hi"]
        rows.append(row)
    return rows


def _dedupe_latest_records(records) -> list:
    """Keep one completed row per backend/device/algo/split/seed for statistics."""
    latest: dict[tuple[str, str, str, str, int], Any] = {}
    for rec in records:
        key = (
            rec.backend or "jax_rejax",
            rec.device or "gpu",
            rec.algo,
            rec.split,
            int(rec.seed),
        )
        previous = latest.get(key)
        if previous is None or str(rec.timestamp) > str(previous.timestamp):
            latest[key] = rec
    return list(latest.values())


def _attach_normscore(rows: list[dict], *, by_device: bool) -> None:
    for row in rows:
        split = row["split"]
        if by_device:
            def _metric(algo: str) -> float | None:
                return next(
                    (
                        r.get("total_profit_mean")
                        for r in rows
                        if r.get("backend") == row["backend"]
                        and r.get("device") == row["device"]
                        and r["algo"] == algo
                        and r["split"] == split
                    ),
                    None,
                )
        else:
            _metric = lambda algo: get_row_mean(rows, algo, split, "total_profit_mean")

        truthful_profit = _metric("truthful")
        max_profit = _metric("max_markup")
        algo_profit = row.get("total_profit_mean")
        row.update(
            compute_norm_score_with_status(
                value=algo_profit,
                anchor_a=truthful_profit,
                anchor_b=max_profit,
                anchor_metric="total_profit",
                metric_direction="higher_is_better",
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
            ood_row = next(
                (
                    r for r in rows
                    if r["backend"] == backend_label
                    and r["device"] == device_label
                    and r["algo"] == algo
                    and r["split"] == _OOD_SPLIT
                ),
                None,
            )
            gap = (
                iid_row["norm_score"] - ood_row["norm_score"]
                if iid_row and ood_row
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
        iid_row = next((r for r in rows if r["algo"] == algo and r["split"] == "iid"), None)
        ood_row = next((r for r in rows if r["algo"] == algo and r["split"] == _OOD_SPLIT), None)
        if (
            iid_row and ood_row
            and iid_row.get("norm_score") is not None
            and ood_row.get("norm_score") is not None
        ):
            algo_gaps[algo] = iid_row["norm_score"] - ood_row["norm_score"]
        else:
            algo_gaps[algo] = None
    for row in rows:
        row["drift_tracking_gap"] = algo_gaps.get(row["algo"])


def _per_episode_metric(task_dir: Path, rec, metric: str) -> list[float]:
    rel = (rec.artifacts or {}).get("per_episode")
    if not rel:
        return []
    path = task_dir / "results" / rel
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    vals: list[float] = []
    for item in raw:
        value = item.get(metric)
        if value is None:
            continue
        try:
            vals.append(float(value))
        except (TypeError, ValueError):
            continue
    return vals


def _collect_paired_episode_values(
    task_dir: Path,
    records,
    *,
    left_algo: str,
    right_algo: str,
    split: str,
    metric: str,
    backend: str = "jax_rejax",
    device: str = "gpu",
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    idx: dict[tuple[str, int], list[float]] = {}
    for rec in records:
        rec_backend = rec.backend or "jax_rejax"
        rec_device = rec.device or "gpu"
        if rec_backend != backend or rec_device != device or rec.split != split:
            continue
        if rec.algo not in (left_algo, right_algo):
            continue
        vals = _per_episode_metric(task_dir, rec, metric)
        if vals:
            idx[(rec.algo, int(rec.seed))] = vals

    common_seeds = sorted(
        set(seed for algo, seed in idx if algo == left_algo)
        & set(seed for algo, seed in idx if algo == right_algo)
    )
    left: list[float] = []
    right: list[float] = []
    for seed in common_seeds:
        left_values = idx.get((left_algo, seed), [])
        right_values = idx.get((right_algo, seed), [])
        n = min(len(left_values), len(right_values))
        if n <= 0:
            continue
        left.extend(left_values[:n])
        right.extend(right_values[:n])
    return (
        np.asarray(left, dtype=np.float64),
        np.asarray(right, dtype=np.float64),
        common_seeds,
    )


def _paired_signflip_permutation_test(
    left: np.ndarray,
    right: np.ndarray,
    *,
    higher_is_better: bool,
    n_perm: int = _PERMUTATION_N,
    rng_seed: int = _PERMUTATION_SEED,
) -> dict[str, Any]:
    diff = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return {
            "test": "paired_signflip_permutation",
            "n_pairs": 0,
            "mean_left_minus_right": None,
            "improvement_mean": None,
            "p_value_two_sided": None,
        }

    observed = float(np.mean(diff))
    n = int(diff.size)
    if n <= 16:
        masks = np.arange(1 << n, dtype=np.uint32)[:, None]
        bits = (masks >> np.arange(n, dtype=np.uint32)) & 1
        signs = bits.astype(np.float64) * 2.0 - 1.0
        perm_stats = np.mean(signs * diff[None, :], axis=1)
    else:
        rng = np.random.default_rng(rng_seed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, n))
        perm_stats = np.mean(signs * diff[None, :], axis=1)

    p_value = float(
        (np.count_nonzero(np.abs(perm_stats) >= abs(observed)) + 1)
        / (len(perm_stats) + 1)
    )
    return {
        "test": "paired_signflip_permutation",
        "n_pairs": n,
        "mean_left_minus_right": observed,
        "improvement_mean": observed if higher_is_better else -observed,
        "p_value_two_sided": p_value,
    }


def _build_hypothesis_tests(
    task_dir: Path,
    records,
    task_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    protocol = task_cfg.get("benchmark_protocol") or {}
    leaderboard = protocol.get("leaderboard") or {}
    split = str(leaderboard.get("split") or task_cfg.get("primary_split", "iid"))
    metric = str(leaderboard.get("primary_metric") or "total_profit")
    direction = str(leaderboard.get("direction") or "higher_is_better")
    higher_is_better = direction == "higher_is_better"
    tests: list[dict[str, Any]] = []
    for left_algo in _PRIMARY_TEST_LEARNERS:
        for right_algo in _PRIMARY_TEST_BASELINES:
            left, right, seeds = _collect_paired_episode_values(
                task_dir,
                records,
                left_algo=left_algo,
                right_algo=right_algo,
                split=split,
                metric=metric,
            )
            test = _paired_signflip_permutation_test(
                left,
                right,
                higher_is_better=higher_is_better,
            )
            tests.append(
                {
                    "backend": "jax_rejax",
                    "device": "gpu",
                    "split": split,
                    "metric": metric,
                    "metric_direction": direction,
                    "left_algo": left_algo,
                    "right_algo": right_algo,
                    "common_seeds": seeds,
                    **test,
                }
            )
    return tests


def _build_gencos_protocol_status(
    rows: list[dict[str, Any]],
    task_cfg: dict[str, Any],
    hypothesis_tests: list[dict[str, Any]],
) -> dict[str, Any]:
    status = build_protocol_status(
        rows,
        task_cfg,
        default_primary_metric="total_profit",
        default_direction="higher_is_better",
    )
    evidence_available = any(int(t.get("n_pairs") or 0) > 0 for t in hypothesis_tests)
    significant_pairs = [
        t
        for t in hypothesis_tests
        if t.get("p_value_two_sided") is not None
        and float(t["p_value_two_sided"]) < 0.05
    ]
    status["hypothesis_test_evidence_available"] = evidence_available
    status["n_hypothesis_tests"] = len(hypothesis_tests)
    status["n_hypothesis_tests_with_pairs"] = sum(
        1 for t in hypothesis_tests if int(t.get("n_pairs") or 0) > 0
    )
    status["n_significant_pairs_p_lt_0_05"] = len(significant_pairs)
    if status["hypothesis_test_implemented"] and not evidence_available:
        status["next_actions"] = list(status.get("next_actions") or [])
        status["next_actions"].append(
            "run primary-split eval records with per_episode artifacts for paired tests"
        )
    status["current_campaign_submission_ready"] = bool(
        status["minimum_seed_requirement_met"]
        and status["hypothesis_test_implemented"]
        and evidence_available
        and not status["protocol_formalization_pending"]
    )
    return status


def summarize_gencos(task_dir: Path, *, after: str | None = None) -> dict:
    """Aggregate GenCos results and save summary JSON; print comparison table."""
    all_records = load_manifest_filtered(task_dir, after=after)
    records = [
        r for r in all_records
        if (r.backend or "jax_rejax") == "jax_rejax" and (r.device or "gpu") == "gpu"
    ]
    if not records:
        print("[GenCos summarize] No records found in manifest.")
        return {}

    all_records = [
        r for r in all_records
        if not (r.split == "train" and has_training_artifact(r.artifacts))
    ]
    records = [
        r for r in records
        if not (r.split == "train" and has_training_artifact(r.artifacts))
    ]
    all_records = _dedupe_latest_records(all_records)
    records = _dedupe_latest_records(records)

    rows = _aggregate_rows(records, by_device=False)
    _attach_normscore(rows, by_device=False)
    phase2_rows = _aggregate_rows(all_records, by_device=True)
    _attach_normscore(phase2_rows, by_device=True)
    all_algos = sorted({r["algo"] for r in rows})
    task_cfg = load_task_config(task_dir)
    hypothesis_tests = _build_hypothesis_tests(task_dir, records, task_cfg)
    protocol_status = _build_gencos_protocol_status(
        rows,
        task_cfg,
        hypothesis_tests,
    )

    summary = {
        "task": "gencos",
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
            primary_metric_key="total_profit_mean",
            primary_direction="higher_is_better",
        ),
        "filters": {"after": after, "backend": "jax_rejax", "device": "gpu"},
    }
    path = save_summary(summary, task_dir)
    print(f"[GenCos summarize] {len(rows)} entries -> {path}")

    # ── Print table ───────────────────────────────────────────────────────
    header = (
        "| {:<18} | {:<18} | {:>11} | {:>9} | {:>7} | {:>7} | {:>9} | {:>9} |"
    ).format(
        "Algo", "Split",
        "TotProfit$", "MeanLMP", "HHI", "PriceCV",
        "NormScore", "DriftGap",
    )
    sep = "|" + "-" * 20 + "|" + "-" * 20 + "|" + ("-" * 13 + "|") * 6
    print()
    print(header)
    print(sep)

    for row in rows:
        print(
            "| {:<18} | {:<18} | {:>11} | {:>9} | {:>7} | {:>7} | {:>9} | {:>9} |".format(
                row["algo"],
                row["split"],
                fmt_metric(row.get("total_profit_mean"), ".2f"),
                fmt_metric(row.get("mean_lmp_mean"), ".2f"),
                fmt_metric(row.get("hhi_mean"), ".3f"),
                fmt_metric(row.get("price_volatility_mean"), ".3f"),
                fmt_metric(row.get("norm_score"), ".3f"),
                fmt_metric(row.get("drift_tracking_gap"), ".3f"),
            )
        )

    print()
    print(f"drift_tracking_gap by algo (NormScore(iid) - NormScore({_OOD_SPLIT})):")
    for algo in all_algos:
        gap = next(
            (row.get("drift_tracking_gap") for row in rows if row["algo"] == algo),
            None,
        )
        print(f"  {algo:<22} {f'{gap:.3f}' if gap is not None else 'N/A'}")

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
                "  {left_algo} vs {right_algo}: n_pairs={n_pairs} "
                "improvement={improvement_mean} p={p_value_two_sided}".format(**test)
            )

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", default=DEFAULT_TASK_DIR, type=Path)
    args = parser.parse_args()
    summarize_gencos(args.task_dir)
