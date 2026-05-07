"""TSO results summarisation.

Reads all RunRecords from the manifest, aggregates across seeds, computes
paper-required metrics, and writes results/summary/latest.json.

Metrics produced
----------------
Per (algo, split) row:
  total_operating_cost_mean   — primary objective (mean ± std across seeds)
  total_gen_cost_mean         — generation cost component
  total_startup_cost_mean     — startup cost component
  total_no_load_cost_mean     — no-load cost component
  feasibility_rate_mean       — fraction of steps with no violations
  thermal_violation_rate_mean — fraction of steps with thermal overloads
  reserve_shortfall_rate_mean — fraction of steps with reserve shortfall
  commitment_switching_frequency_mean — off→on switches / episode
  norm_score                  — (all_on_cost - algo_cost) / (all_on_cost - merit_order_cost)
                                [0 = all_on level, 1 = merit_order level]

Per-algo summary:
  ood_degradation             — norm_score(iid) - norm_score(load_stress)
                                Measures performance degradation under load stress.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import json
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
from benchmarks.common.reporting import summarize_group_evidence
from benchmarks.common.stats import (
    aggregate_seeds,
    bootstrap_ci,
    compute_norm_score_with_status,
    fmt_metric,
    iqm,
    metric_sensitivity_report,
    norm_score,
    numeric_only,
)

_MAX_STEPS = 48
_BOOTSTRAP_SEED = 42
_PRIMARY_HYPOTHESIS_PAIRS = (
    ("ppo_lagrangian", "all_on"),
    ("ppo_lagrangian", "ppo"),
    ("ppo", "all_on"),
)


def _default_campaign_after(task_cfg: dict[str, Any]) -> str | None:
    protocol = task_cfg.get("benchmark_protocol") or {}
    value = protocol.get("current_campaign_start_iso")
    return str(value) if value else None


def _per_episode_metric(task_dir: Path, record, metric: str) -> list[float]:
    """Load one metric from a run's per-episode artifact."""
    artifacts_dir = task_dir / "results" / "artifacts"
    rel = (record.artifacts or {}).get("per_episode")
    path = task_dir / "results" / rel if rel else artifacts_dir / f"{record.run_id}_per_episode.json"
    if not path.exists():
        return []
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    vals: list[float] = []
    for row in rows:
        if not isinstance(row, dict) or metric not in row:
            continue
        try:
            vals.append(float(row[metric]))
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
    backend: str = "jax_rejax",
    device: str = "gpu",
    metric: str = "total_operating_cost",
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    idx: dict[tuple[str, int], list[float]] = {}
    for rec in records:
        if rec.backend != backend or rec.device != device or rec.split != split:
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
    used_seeds: list[int] = []
    for seed in common_seeds:
        lv = idx.get((left_algo, seed), [])
        rv = idx.get((right_algo, seed), [])
        n = min(len(lv), len(rv))
        if n <= 0:
            continue
        left.extend(lv[:n])
        right.extend(rv[:n])
        used_seeds.append(seed)
    return np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64), used_seeds


def _paired_signflip_permutation_test(
    left: np.ndarray,
    right: np.ndarray,
    *,
    n_perm: int = 20000,
    rng_seed: int = _BOOTSTRAP_SEED,
) -> dict[str, Any]:
    diff = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return {
            "test": "paired_signflip_permutation",
            "n_pairs": 0,
            "mean_left_minus_right": None,
            "p_value_two_sided": None,
        }

    obs = float(np.mean(diff))
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

    p = float((np.count_nonzero(np.abs(perm_stats) >= abs(obs)) + 1) / (len(perm_stats) + 1))
    return {
        "test": "paired_signflip_permutation",
        "n_pairs": n,
        "mean_left_minus_right": obs,
        "p_value_two_sided": p,
    }


def build_tso_hypothesis_tests(
    task_dir: Path,
    records,
    task_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build primary-split paired tests for canonical JAX/GPU TSO rows."""
    primary_split = task_cfg.get("primary_split", "iid")
    metric = task_cfg.get("target_return_metric_key", "total_operating_cost")
    direction = task_cfg.get("target_metric_direction", "lower_is_better")
    tests: list[dict[str, Any]] = []
    for left_algo, right_algo in _PRIMARY_HYPOTHESIS_PAIRS:
        left, right, seeds = _collect_paired_episode_values(
            task_dir,
            records,
            left_algo=left_algo,
            right_algo=right_algo,
            split=primary_split,
            metric=metric,
        )
        test = _paired_signflip_permutation_test(left, right)
        mean_diff = test.get("mean_left_minus_right")
        left_better = None
        if mean_diff is not None:
            left_better = (
                float(mean_diff) < 0.0
                if direction == "lower_is_better"
                else float(mean_diff) > 0.0
            )
        tests.append(
            {
                "backend": "jax_rejax",
                "device": "gpu",
                "split": primary_split,
                "metric": metric,
                "metric_direction": direction,
                "left_algo": left_algo,
                "right_algo": right_algo,
                "common_seeds": seeds,
                "left_better": left_better,
                **test,
            }
        )
    return tests


def build_tso_leaderboard(rows: list[dict[str, Any]], task_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a primary-split leaderboard with explicit safety gating."""
    protocol = task_cfg.get("benchmark_protocol") or {}
    leaderboard_cfg = protocol.get("leaderboard") or {}
    primary_split = leaderboard_cfg.get("split") or task_cfg.get("primary_split", "iid")
    primary_metric = leaderboard_cfg.get("primary_metric") or task_cfg.get(
        "target_return_metric_key", "total_operating_cost"
    )
    metric_direction = leaderboard_cfg.get("direction") or task_cfg.get(
        "target_metric_direction", "lower_is_better"
    )
    include_baselines = bool(leaderboard_cfg.get("include_baselines", True))
    baseline_set = set(task_cfg.get("baseline_set") or task_cfg.get("baseline_algos") or [])
    safety_thresholds = task_cfg.get("safety_thresholds") or {}
    required_safety = leaderboard_cfg.get("require_safety_thresholds") or list(safety_thresholds)

    entries: list[dict[str, Any]] = []
    for row in rows:
        if row.get("split") != primary_split:
            continue
        if not include_baselines and row.get("algo") in baseline_set:
            continue

        metric_key = f"{primary_metric}_mean"
        metric_value = row.get(metric_key)
        exclusion_reasons: list[str] = []
        safety_status: dict[str, dict[str, Any] | None] = {}

        if metric_value is None:
            exclusion_reasons.append(f"missing_{primary_metric}")

        for metric_name in required_safety:
            threshold = safety_thresholds.get(metric_name)
            observed = row.get(f"{metric_name}_mean")
            if threshold is None:
                safety_status[metric_name] = None
                continue
            if observed is None:
                safety_status[metric_name] = {
                    "observed": None,
                    "threshold": float(threshold),
                    "meets_threshold": None,
                }
                exclusion_reasons.append(f"missing_{metric_name}")
                continue
            meets = float(observed) <= float(threshold)
            safety_status[metric_name] = {
                "observed": float(observed),
                "threshold": float(threshold),
                "meets_threshold": bool(meets),
            }
            if not meets:
                exclusion_reasons.append(
                    f"{metric_name}={float(observed):.3f}>{float(threshold):.3f}"
                )

        eligible = metric_value is not None and not exclusion_reasons
        entries.append({
            "algo": row["algo"],
            "split": primary_split,
            "primary_metric": primary_metric,
            "primary_metric_value": float(metric_value) if metric_value is not None else None,
            "metric_direction": metric_direction,
            "n_seeds": int(row.get("n_seeds", 0)),
            "norm_score": row.get("norm_score"),
            "is_baseline": row["algo"] in baseline_set,
            "leaderboard_eligible": eligible,
            "exclusion_reasons": exclusion_reasons,
            "safety_status": safety_status,
        })

    reverse = metric_direction == "higher_is_better"
    entries.sort(
        key=lambda entry: (
            0 if entry["leaderboard_eligible"] else 1,
            (
                entry["primary_metric_value"]
                if entry["primary_metric_value"] is not None
                else (float("-inf") if reverse else float("inf"))
            ),
            entry["algo"],
        ),
        reverse=reverse,
    )

    rank = 1
    for entry in entries:
        if entry["leaderboard_eligible"]:
            entry["leaderboard_rank"] = rank
            rank += 1
        else:
            entry["leaderboard_rank"] = None
    return entries


def build_tso_protocol_status(
    rows: list[dict[str, Any]],
    task_cfg: dict[str, Any],
    leaderboard: list[dict[str, Any]],
    hypothesis_tests: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Report whether the current summary satisfies submission-grade protocol."""
    protocol = task_cfg.get("benchmark_protocol") or {}
    submission_min_seeds = int(protocol.get("submission_min_seeds", 5))
    configured_seed_budget = len(task_cfg.get("seeds") or [])
    primary_split = (protocol.get("leaderboard") or {}).get("split") or task_cfg.get(
        "primary_split", "iid"
    )
    primary_rows = [row for row in rows if row.get("split") == primary_split]
    primary_algorithms = set(protocol.get("primary_algorithms") or [])
    baseline_algorithms = set(task_cfg.get("baseline_set") or task_cfg.get("baseline_algos") or [])
    if primary_algorithms:
        formal_algorithms = primary_algorithms | baseline_algorithms
        primary_rows = [
            row for row in primary_rows
            if row.get("algo") in formal_algorithms
        ]
    per_algo_seed_count = {
        row["algo"]: int(row.get("n_seeds", 0))
        for row in primary_rows
    }
    observed_min_seed_count = min(per_algo_seed_count.values(), default=0)
    minimum_seed_requirement_met = bool(per_algo_seed_count) and all(
        count >= submission_min_seeds for count in per_algo_seed_count.values()
    )

    hypothesis_cfg = protocol.get("hypothesis_test") or {}
    hypothesis_status = hypothesis_cfg.get("status", "unspecified")
    hypothesis_test_implemented = hypothesis_status == "implemented"
    hypothesis_tests = list(hypothesis_tests or [])
    hypothesis_tests_available = any(
        int(test.get("n_pairs") or 0) > 0 for test in hypothesis_tests
    )

    top_entry = next((entry for entry in leaderboard if entry["leaderboard_eligible"]), None)
    reward_hacking_guardrail = top_entry is not None
    guardrail_note = (
        "Unsafe primary-split rows are excluded from the leaderboard."
        if top_entry is not None
        else "No primary-split row satisfies the declared safety thresholds."
    )

    return {
        "benchmark_question": protocol.get("benchmark_question"),
        "primary_split": primary_split,
        "configured_seed_budget": configured_seed_budget,
        "submission_min_seeds": submission_min_seeds,
        "observed_min_seed_count": observed_min_seed_count,
        "per_algo_seed_count_primary_split": per_algo_seed_count,
        "minimum_seed_requirement_met": minimum_seed_requirement_met,
        "hypothesis_test": hypothesis_cfg,
        "hypothesis_test_implemented": hypothesis_test_implemented,
        "hypothesis_tests_available": hypothesis_tests_available,
        "n_hypothesis_tests": len(hypothesis_tests),
        "reward_hacking_guardrail": reward_hacking_guardrail,
        "reward_hacking_guardrail_note": guardrail_note,
        "current_campaign_submission_ready": (
            minimum_seed_requirement_met
            and hypothesis_test_implemented
            and hypothesis_tests_available
            and reward_hacking_guardrail
        ),
        "next_actions": [
            action
            for action, needed in [
                (
                    f"rerun primary benchmark cells with at least {submission_min_seeds} seeds",
                    not minimum_seed_requirement_met,
                ),
                (
                    "add the declared paired hypothesis test to the primary-split table",
                    not hypothesis_test_implemented,
                ),
                (
                    "rerun summarize after primary-split per-episode artifacts exist",
                    hypothesis_test_implemented and not hypothesis_tests_available,
                ),
            ]
            if needed
        ],
    }


def summarize_tso(
    task_dir: Path,
    *,
    after: str | None = None,
    backend: str = "jax_rejax",
    device: str = "gpu",
) -> dict:
    """Aggregate TSO results and save summary JSON; print paper comparison table."""
    task_cfg = load_task_config(task_dir)
    if after is None:
        after = _default_campaign_after(task_cfg)
    records = load_manifest_filtered(
        task_dir,
        after=after,
        backend=backend,
        device=device,
    )
    if not records:
        print("[TSO summarize] No records found in manifest.")
        return {}
    records = [
        r for r in records
        if r.status == "completed"
        and not (r.split == "train" and has_training_artifact(r.artifacts))
    ]
    if not records:
        print("[TSO summarize] No completed eval/baseline records found.")
        return {}

    groups: dict[tuple[str, str], list[dict]] = {}
    group_records: dict[tuple[str, str], list] = {}
    group_run_ids: dict[tuple[str, str], list[str]] = {}
    for r in records:
        key = (r.algo, r.split)
        groups.setdefault(key, []).append(numeric_only(r.metrics))
        group_records.setdefault(key, []).append(r)
        group_run_ids.setdefault(key, []).append(r.run_id)

    artifacts_dir = task_dir / "results" / "artifacts"

    rows: list[dict] = []
    for (algo, split), metrics_list in sorted(groups.items()):
        agg = aggregate_seeds(metrics_list)
        row: dict = {"algo": algo, "split": split, "n_seeds": len(metrics_list)}
        row.update(summarize_group_evidence(group_records.get((algo, split), [])))
        for metric_name, stats in agg.items():
            row[f"{metric_name}_mean"] = stats["mean"]
            row[f"{metric_name}_std"] = stats["std"]
            row[f"{metric_name}_iqm"] = stats["iqm"]
            row[f"{metric_name}_ci_lo"] = stats["ci_lo"]
            row[f"{metric_name}_ci_hi"] = stats["ci_hi"]

        # Derived: feasibility_rate = 1 - total_violations / max_steps
        # Back-fills the field for runs recorded before is_safe was tracked directly.
        if "feasibility_rate_mean" not in row and "total_violations_mean" in row:
            row["feasibility_rate_mean"] = (
                1.0 - row["total_violations_mean"] / _MAX_STEPS
            )

        # Safety-rate back-fills for legacy records that predate the explicit
        # per-step rate keys in ``compute_tso_metrics``.
        run_ids = group_run_ids.get((algo, split), [])
        thermal_rates: list[float] = []
        shortfall_rates: list[float] = []
        shortfall_episode_incidence: list[float] = []
        for rid in run_ids:
            pe_file = artifacts_dir / f"{rid}_per_episode.json"
            if pe_file.exists():
                episodes = json.loads(pe_file.read_text(encoding="utf-8"))
                if episodes and "thermal_violation_rate" in episodes[0]:
                    thermal_rates.append(
                        float(
                            np.mean(
                                [
                                    float(ep.get("thermal_violation_rate", 0.0))
                                    for ep in episodes
                                ]
                            )
                        )
                    )
                if episodes and "reserve_shortfall_rate" in episodes[0]:
                    shortfall_rates.append(
                        float(
                            np.mean(
                                [
                                    float(ep.get("reserve_shortfall_rate", 0.0))
                                    for ep in episodes
                                ]
                            )
                        )
                    )
                else:
                    shortfall_episode_incidence.append(
                        sum(
                            1
                            for ep in episodes
                            if ep.get("total_reserve_shortfall", 0.0) > 1e-6
                        )
                        / max(len(episodes), 1)
                    )

        if "thermal_violation_rate_mean" not in row:
            if thermal_rates:
                row["thermal_violation_rate_mean"] = float(np.mean(thermal_rates))
            elif "mean_thermal_cost_mean" in row:
                row["thermal_violation_rate_mean"] = min(
                    1.0, max(0.0, row["mean_thermal_cost_mean"])
                )

        if "reserve_shortfall_rate_mean" not in row:
            if shortfall_rates:
                row["reserve_shortfall_rate_mean"] = float(np.mean(shortfall_rates))
            elif shortfall_episode_incidence:
                row["reserve_shortfall_rate_mean"] = float(
                    np.mean(shortfall_episode_incidence)
                )
            elif "mean_reserve_shortfall_mean" in row:
                row["reserve_shortfall_rate_mean"] = min(
                    1.0, max(0.0, row["mean_reserve_shortfall_mean"])
                )

        rows.append(row)

    # ── Per-seed NormScore + bootstrap CI (Agarwal et al. 2021) ─────────
    # NormScore_i = (all_on_mean - algo_cost_i) / (all_on_mean - mo_mean)
    # We compute one NormScore per seed and report IQM with 95% bootstrap CI.
    # 0 = all_on level; 1 = merit_order level; >1 = better than merit_order.
    def _collect_per_seed_cost(algo: str, split: str) -> list[float]:
        out: list[float] = []
        for r in records:
            if r.algo == algo and r.split == split:
                c = r.metrics.get("total_operating_cost")
                if c is not None:
                    out.append(float(c))
        return out

    for row in rows:
        algo = row["algo"]
        split = row["split"]
        algo_costs = _collect_per_seed_cost(algo, split)
        ref_costs = _collect_per_seed_cost("all_on", split)
        ora_costs = _collect_per_seed_cost("merit_order", split)

        if algo_costs and ref_costs and ora_costs:
            ref_mean = float(np.mean(ref_costs))
            ora_mean = float(np.mean(ora_costs))
            status = compute_norm_score_with_status(
                value=row.get("total_operating_cost_mean"),
                anchor_a=ref_mean,
                anchor_b=ora_mean,
                anchor_metric="total_operating_cost",
                metric_direction="lower_is_better",
            )
            row.update(status)
            if status["norm_score_status"] == "ok":
                per_seed_ns = np.array([
                    float(norm_score(c, ref_mean, ora_mean)) for c in algo_costs
                ])
                _, ci_lo, ci_hi = bootstrap_ci(per_seed_ns)
                ns_iqm = float(iqm(per_seed_ns))
                row["norm_score"] = ns_iqm
                row["norm_score_iqm"] = ns_iqm
                row["norm_score_ci_lo"] = float(ci_lo)
                row["norm_score_ci_hi"] = float(ci_hi)
            else:
                row["norm_score_iqm"] = None
                row["norm_score_ci_lo"] = None
                row["norm_score_ci_hi"] = None
        else:
            row.update(
                compute_norm_score_with_status(
                    value=None,
                    anchor_a=None,
                    anchor_b=None,
                    anchor_metric="total_operating_cost",
                    metric_direction="lower_is_better",
                )
            )
            row["norm_score_iqm"] = None
            row["norm_score_ci_lo"] = None
            row["norm_score_ci_hi"] = None

    # ── ood_degradation per algo ───────────────────────────────────────────
    # degradation = norm_score(iid) - norm_score(load_stress)
    # Positive = policy degrades under load stress vs IID.
    all_algos = sorted({r["algo"] for r in rows})
    algo_degradations: dict[str, float | None] = {}
    for algo in all_algos:
        iid_row = next(
            (r for r in rows if r["algo"] == algo and r["split"] == "iid"), None
        )
        ood_row = next(
            (r for r in rows if r["algo"] == algo and r["split"] == "load_stress"), None
        )
        if (
            iid_row is not None
            and ood_row is not None
            and iid_row.get("norm_score") is not None
            and ood_row.get("norm_score") is not None
        ):
            algo_degradations[algo] = iid_row["norm_score"] - ood_row["norm_score"]
        else:
            algo_degradations[algo] = None

    for row in rows:
        row["ood_degradation"] = algo_degradations.get(row["algo"])

    # ── Absolute cost increase under OOD ─────────────────────────────────
    # cost_increase_pct = (cost(split) - cost(iid)) / cost(iid) * 100 [%].
    # Reviewer-friendly raw signal that does NOT depend on the all_on / merit_order
    # normalisation, so it is immune to the "denominator inflation" effect that
    # makes ood_degradation negative on this benchmark.
    def _cost_for(algo: str, split: str):
        for r in rows:
            if r["algo"] == algo and r["split"] == split:
                return r.get("total_operating_cost_mean")
        return None

    for row in rows:
        iid_cost = _cost_for(row["algo"], "iid")
        split_cost = row.get("total_operating_cost_mean")
        if (
            iid_cost is not None
            and split_cost is not None
            and iid_cost > 0
        ):
            row["cost_increase_pct"] = (split_cost - iid_cost) / iid_cost * 100.0
        else:
            row["cost_increase_pct"] = None

    hypothesis_tests = build_tso_hypothesis_tests(task_dir, records, task_cfg)
    leaderboard = build_tso_leaderboard(rows, task_cfg)
    leaderboard_by_algo = {
        entry["algo"]: entry
        for entry in leaderboard
        if entry.get("split") == task_cfg.get("primary_split", "iid")
    }
    for row in rows:
        entry = leaderboard_by_algo.get(row["algo"]) if row.get("split") == task_cfg.get("primary_split", "iid") else None
        row["leaderboard_eligible"] = bool(entry["leaderboard_eligible"]) if entry else False
        row["exclusion_reasons"] = list(entry["exclusion_reasons"]) if entry else []
        row["safety_status"] = dict(entry["safety_status"]) if entry else {}
    protocol_status = build_tso_protocol_status(
        rows,
        task_cfg,
        leaderboard,
        hypothesis_tests,
    )
    summary = {
        "task": "tso",
        "rows": rows,
        "hypothesis_tests": hypothesis_tests,
        "protocol_status": protocol_status,
        "leaderboard_primary_split": leaderboard,
        "split_taxonomy": task_cfg.get("split_taxonomy", {}),
        "reporting": {
            "suppressed_rows": [],
            "legacy_default_evidence_tier": "legacy_unknown",
            "cross_backend_gate": "comparison_audited=true and audit_suite_version is required",
        },
        "metric_sensitivity": metric_sensitivity_report(
            [row for row in rows if row.get("split") == task_cfg.get("primary_split", "iid")],
            primary_metric_key="total_operating_cost_mean",
            primary_direction="lower_is_better",
        ),
        "filters": {
            "after": after,
            "backend": backend,
            "device": device,
        },
    }
    path = save_summary(summary, task_dir)
    print(f"[TSO summarize] {len(rows)} entries -> {path}")

    header = (
        "| {:<18} | {:<16} | {:>11} | {:>9} | {:>11} | {:>9} | {:>9} |"
    ).format(
        "Algo", "Split",
        "Op Cost",
        "Feasib %",
        "Thermal Viol",
        "NormScore",
        "OOD Degr",
    )
    sep = (
        "|" + "-" * 20 + "|" + "-" * 18 + "|"
        + ("-" * 13 + "|") * 5
    )
    print()
    print(header)
    print(sep)

    split_order = ["train", "iid", "load_stress", "line_tightening"]
    algo_order = ["all_on", "merit_order", "ppo", "ppo_lagrangian"]

    def _sort_key(r):
        a = r["algo"]
        s = r["split"]
        return (
            algo_order.index(a) if a in algo_order else 99,
            split_order.index(s) if s in split_order else 99,
        )

    for row in sorted(rows, key=_sort_key):
        print(
            "| {:<18} | {:<16} | {:>11} | {:>9} | {:>11} | {:>9} | {:>9} |".format(
                row["algo"],
                row["split"],
                fmt_metric(row.get("total_operating_cost_mean"), ".1f"),
                fmt_metric(row.get("feasibility_rate_mean"), ".3f"),
                fmt_metric(row.get("thermal_violation_rate_mean"), ".3f"),
                fmt_metric(row.get("norm_score"), ".3f"),
                fmt_metric(row.get("ood_degradation"), ".3f"),
            )
        )

    # Print cost decomposition for each algo on train split
    print()
    print("Cost decomposition (train split, mean across seeds):")
    print(
        "| {:<18} | {:>12} | {:>12} | {:>12} |".format(
            "Algo", "Gen Cost", "Startup Cost", "No-Load Cost"
        )
    )
    print("|" + "-" * 20 + "|" + ("-" * 14 + "|") * 3)
    for row in sorted(rows, key=_sort_key):
        if row["split"] != "train":
            continue
        print(
            "| {:<18} | {:>12} | {:>12} | {:>12} |".format(
                row["algo"],
                fmt_metric(row.get("total_gen_cost_mean"), ".1f"),
                fmt_metric(row.get("total_startup_cost_mean"), ".1f"),
                fmt_metric(row.get("total_no_load_cost_mean"), ".1f"),
            )
        )

    # Print ood_degradation summary
    print()
    print("ood_degradation by algo (NormScore(IID) - NormScore(load_stress)):")
    for algo in all_algos:
        d = algo_degradations.get(algo)
        d_str = f"{d:.3f}" if d is not None else "N/A"
        print(f"  {algo:<20} {d_str}")

    # Print absolute cost increase relative to IID for each OOD split.
    # This is denominator-free and is the metric reviewers can verify by hand.
    print()
    print("cost_increase_pct (vs IID, mean across seeds):")
    print(
        "| {:<18} | {:>14} | {:>14} | {:>14} |".format(
            "Algo", "train", "load_stress", "line_tightening"
        )
    )
    print("|" + "-" * 20 + "|" + ("-" * 16 + "|") * 3)
    for algo in [a for a in algo_order if a in all_algos]:
        cells = []
        for split in ("train", "load_stress", "line_tightening"):
            row = next(
                (r for r in rows if r["algo"] == algo and r["split"] == split),
                None,
            )
            v = row.get("cost_increase_pct") if row else None
            cells.append(f"{v:+.2f}%" if v is not None else "  N/A ")
        print(
            "| {:<18} | {:>14} | {:>14} | {:>14} |".format(algo, *cells)
        )

    print()
    proto_mark = "READY" if protocol_status["current_campaign_submission_ready"] else "NOT READY"
    print(
        "submission protocol: "
        f"{proto_mark} "
        f"(observed min seeds={protocol_status['observed_min_seed_count']}, "
        f"required={protocol_status['submission_min_seeds']}, "
        f"hypothesis_test={protocol_status['hypothesis_test'].get('status', 'unspecified')})"
    )
    if leaderboard:
        print("primary-split leaderboard (IID):")
        for entry in leaderboard:
            rank = entry["leaderboard_rank"] if entry["leaderboard_rank"] is not None else "-"
            status = "eligible" if entry["leaderboard_eligible"] else "reported-only"
            metric_value = entry["primary_metric_value"]
            metric_str = f"{metric_value:.1f}" if metric_value is not None else "N/A"
            print(
                f"  [{rank}] {entry['algo']:<18} {status:<13} "
                f"{entry['primary_metric']}={metric_str}"
            )
            if entry["exclusion_reasons"]:
                print(f"      exclusion_reasons={entry['exclusion_reasons']}")

    if hypothesis_tests:
        print("primary-split paired hypothesis tests:")
        for test in hypothesis_tests:
            print(
                "  {left_algo} vs {right_algo}: n_pairs={n_pairs} "
                "mean_diff={mean_diff} p={p_value}".format(
                    left_algo=test["left_algo"],
                    right_algo=test["right_algo"],
                    n_pairs=test["n_pairs"],
                    mean_diff=fmt_metric(test.get("mean_left_minus_right"), ".3f"),
                    p_value=fmt_metric(test.get("p_value_two_sided"), ".4f"),
                )
            )

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", default=DEFAULT_TASK_DIR, type=Path)
    parser.add_argument("--after", default=None)
    parser.add_argument("--backend", default="jax_rejax")
    parser.add_argument("--device", default="gpu")
    args = parser.parse_args()
    summarize_tso(
        args.task_dir,
        after=args.after,
        backend=args.backend,
        device=args.device,
    )
