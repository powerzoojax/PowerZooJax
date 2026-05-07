"""DC Microgrid results summarisation."""

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
from benchmarks.common.io import load_manifest_filtered, save_summary
from benchmarks.common.reporting import mark_dc_phase2_audited, summarize_group_evidence
from benchmarks.common.stats import (
    aggregate_seeds,
    compute_norm_score_with_status,
    fmt_metric,
    metric_sensitivity_report,
    numeric_only,
)


_BOOTSTRAP_N = 10000
_BOOTSTRAP_SEED = 42
_DEFAULT_SUBMISSION_MIN_SEEDS = 5
_CANONICAL_DEVICE = {
    "jax_rejax": "gpu",
    "sb3": "cuda",
    "sbx": "cuda",
}
_PRIMARY_TEST_BASELINE = "rule_based"
_PRIMARY_TEST_LEARNERS = ("ppo", "sac")


def _canonical_device_for_backend(backend: str) -> str:
    return _CANONICAL_DEVICE.get(backend, "gpu")


def _row_key(
    backend: str,
    device: str,
    algo: str,
    split: str,
    *,
    by_device: bool,
) -> tuple[str, ...]:
    if by_device:
        return (backend, device, algo, split)
    return (backend, algo, split)


def _paired_bootstrap_norm_score(
    algo_vals: np.ndarray,
    no_ctrl_vals: np.ndarray,
    rb_vals: np.ndarray,
    n_boot: int = _BOOTSTRAP_N,
    rng_seed: int = _BOOTSTRAP_SEED,
) -> tuple[float | None, float | None]:
    """Paired-by-seed bootstrap CI for NormScore."""
    n = len(algo_vals)
    if n == 0:
        return None, None
    rng = np.random.default_rng(rng_seed)
    samples: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        denom = float(rb_vals[idx].mean() - no_ctrl_vals[idx].mean())
        if abs(denom) < 1e-8:
            continue
        samples.append(float((algo_vals[idx].mean() - no_ctrl_vals[idx].mean()) / denom))
    if not samples:
        return None, None
    s = np.asarray(samples)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def _aggregate_rows(records, *, by_device: bool) -> tuple[list[dict], dict]:
    groups: dict[tuple[str, ...], list[dict[str, float]]] = {}
    group_records: dict[tuple[str, ...], list] = {}
    seed_rewards: dict[tuple[str, ...], dict[int, float]] = {}

    for r in records:
        backend = r.backend or "jax_rejax"
        device = r.device or _canonical_device_for_backend(backend)
        key = _row_key(backend, device, r.algo, r.split, by_device=by_device)
        groups.setdefault(key, []).append(numeric_only(r.metrics))
        group_records.setdefault(key, []).append(r)
        reward = r.metrics.get("episode_reward")
        if reward is not None and np.isfinite(reward):
            seed_rewards.setdefault(key, {})[r.seed] = float(reward)

    rows: list[dict[str, Any]] = []
    for key, metrics_list in sorted(groups.items()):
        agg = aggregate_seeds(metrics_list)
        if by_device:
            backend, device, algo, split = key
        else:
            backend, algo, split = key
            device = _canonical_device_for_backend(backend)
        row: dict[str, Any] = {
            "backend": backend,
            "device": device,
            "algo": algo,
            "split": split,
            "n_seeds": len(metrics_list),
        }
        row.update(summarize_group_evidence(group_records.get(key, [])))
        for metric_name, stats in agg.items():
            row[f"{metric_name}_mean"] = stats["mean"]
            row[f"{metric_name}_std"] = stats["std"]
            row[f"{metric_name}_iqm"] = stats["iqm"]
            row[f"{metric_name}_ci_lo"] = stats["ci_lo"]
            row[f"{metric_name}_ci_hi"] = stats["ci_hi"]
        rows.append(row)
    return rows, seed_rewards


def _attach_normscore_and_ood(
    rows: list[dict[str, Any]],
    seed_rewards: dict[tuple[str, ...], dict[int, float]],
    *,
    by_device: bool,
) -> None:
    for row in rows:
        backend = row["backend"]
        device = row["device"]
        split = row["split"]
        algo = row["algo"]
        key_algo = _row_key(backend, device, algo, split, by_device=by_device)
        rewards_algo = seed_rewards.get(key_algo, {})

        baseline_backend = backend
        baseline_device = device
        if backend != "jax_rejax" or device != "gpu":
            baseline_backend = "jax_rejax"
            baseline_device = "gpu"

        rewards_nc = seed_rewards.get(
            _row_key(
                baseline_backend,
                baseline_device,
                "no_control",
                split,
                by_device=by_device,
            ),
            {},
        )
        rewards_rb = seed_rewards.get(
            _row_key(
                baseline_backend,
                baseline_device,
                "rule_based",
                split,
                by_device=by_device,
            ),
            {},
        )

        common = sorted(set(rewards_algo) & set(rewards_nc) & set(rewards_rb))
        if not common:
            row.update(
                compute_norm_score_with_status(
                    value=None,
                    anchor_a=None,
                    anchor_b=None,
                    anchor_metric="episode_reward",
                    metric_direction="higher_is_better",
                )
            )
            row["norm_score_ci_lo"] = None
            row["norm_score_ci_hi"] = None
            continue

        a = np.array([rewards_algo[s] for s in common], dtype=np.float64)
        nc = np.array([rewards_nc[s] for s in common], dtype=np.float64)
        rb = np.array([rewards_rb[s] for s in common], dtype=np.float64)
        row.update(
            compute_norm_score_with_status(
                value=float(a.mean()),
                anchor_a=float(nc.mean()),
                anchor_b=float(rb.mean()),
                anchor_metric="episode_reward",
                metric_direction="higher_is_better",
            )
        )
        if row["norm_score_status"] == "ok" and len(common) >= 2:
            lo, hi = _paired_bootstrap_norm_score(a, nc, rb)
            row["norm_score_ci_lo"] = lo
            row["norm_score_ci_hi"] = hi
        else:
            row["norm_score_ci_lo"] = None
            row["norm_score_ci_hi"] = None

    algo_gaps: dict[tuple[str, str], float | None] = {}
    all_algo_backends = sorted({(r["backend"], r["algo"]) for r in rows})
    for backend, algo in all_algo_backends:
        iid_row = next(
            (
                r for r in rows
                if r["backend"] == backend and r["algo"] == algo and r["split"] == "iid"
            ),
            None,
        )
        ood_row = next(
            (
                r for r in rows
                if r["backend"] == backend
                and r["algo"] == algo
                and r["split"] == "cooling_stress"
            ),
            None,
        )
        if (
            iid_row is not None
            and ood_row is not None
            and iid_row.get("norm_score") is not None
            and ood_row.get("norm_score") is not None
        ):
            algo_gaps[(backend, algo)] = iid_row["norm_score"] - ood_row["norm_score"]
        else:
            algo_gaps[(backend, algo)] = None

    for row in rows:
        row["ood_robustness_gap"] = algo_gaps.get((row["backend"], row["algo"]))


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
        v = item.get(metric)
        if v is None:
            continue
        try:
            vals.append(float(v))
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
    metric: str = "episode_reward",
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    idx: dict[tuple[str, int], list[float]] = {}
    for rec in records:
        rec_backend = rec.backend or "jax_rejax"
        rec_device = rec.device or _canonical_device_for_backend(rec_backend)
        if rec_backend != backend or rec_device != device or rec.split != split:
            continue
        if rec.algo not in (left_algo, right_algo):
            continue
        vals = _per_episode_metric(task_dir, rec, metric)
        if vals:
            idx[(rec.algo, rec.seed)] = vals

    common_seeds = sorted(
        set(seed for algo, seed in idx if algo == left_algo)
        & set(seed for algo, seed in idx if algo == right_algo)
    )
    left: list[float] = []
    right: list[float] = []
    for seed in common_seeds:
        lv = idx.get((left_algo, seed), [])
        rv = idx.get((right_algo, seed), [])
        n = min(len(lv), len(rv))
        if n <= 0:
            continue
        left.extend(lv[:n])
        right.extend(rv[:n])
    return (
        np.asarray(left, dtype=np.float64),
        np.asarray(right, dtype=np.float64),
        common_seeds,
    )


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
            "n_pairs": 0,
            "mean_left_minus_right": None,
            "p_value_two_sided": None,
            "test": "paired_signflip_permutation",
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


def _build_hypothesis_tests(
    task_dir: Path,
    records,
    task_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    primary_split = task_cfg.get("primary_split", "iid")
    tests: list[dict[str, Any]] = []
    pairs = [
        ("sac", "ppo"),
        ("ppo", _PRIMARY_TEST_BASELINE),
        ("sac", _PRIMARY_TEST_BASELINE),
    ]
    for left_algo, right_algo in pairs:
        left, right, seeds = _collect_paired_episode_values(
            task_dir,
            records,
            left_algo=left_algo,
            right_algo=right_algo,
            split=primary_split,
            backend="jax_rejax",
            device="gpu",
        )
        test = _paired_signflip_permutation_test(left, right)
        tests.append(
            {
                "backend": "jax_rejax",
                "device": "gpu",
                "split": primary_split,
                "metric": "episode_reward",
                "left_algo": left_algo,
                "right_algo": right_algo,
                "common_seeds": seeds,
                **test,
            }
        )
    return tests


def _build_protocol_status(
    rows: list[dict[str, Any]],
    task_cfg: dict[str, Any],
    hypothesis_tests: list[dict[str, Any]],
) -> dict[str, Any]:
    protocol = task_cfg.get("benchmark_protocol") or {}
    submission_min_seeds = int(
        protocol.get("submission_min_seeds", _DEFAULT_SUBMISSION_MIN_SEEDS)
    )
    primary_split = (protocol.get("leaderboard") or {}).get("split") or task_cfg.get(
        "primary_split", "iid"
    )
    primary_rows = [
        row
        for row in rows
        if row["split"] == primary_split and row["backend"] == "jax_rejax"
    ]
    per_algo_seed_count = {
        row["algo"]: int(row.get("n_seeds", 0))
        for row in primary_rows
    }
    observed_min_seed_count = min(per_algo_seed_count.values(), default=0)
    minimum_seed_requirement_met = bool(per_algo_seed_count) and all(
        count >= submission_min_seeds for count in per_algo_seed_count.values()
    )
    hypothesis_test_implemented = bool(hypothesis_tests)
    significant_pairs = [
        t for t in hypothesis_tests
        if t.get("p_value_two_sided") is not None and t["p_value_two_sided"] < 0.05
    ]
    return {
        "primary_split": primary_split,
        "submission_min_seeds": submission_min_seeds,
        "observed_min_seed_count": observed_min_seed_count,
        "per_algo_seed_count_primary_split": per_algo_seed_count,
        "minimum_seed_requirement_met": minimum_seed_requirement_met,
        "hypothesis_test_implemented": hypothesis_test_implemented,
        "n_significant_pairs_p_lt_0_05": len(significant_pairs),
        "current_campaign_submission_ready": (
            minimum_seed_requirement_met and hypothesis_test_implemented
        ),
        "next_actions": [
            action
            for action, needed in [
                (
                    f"rerun primary benchmark cells to at least {submission_min_seeds} seeds",
                    not minimum_seed_requirement_met,
                ),
                (
                    "add primary-split paired hypothesis tests",
                    not hypothesis_test_implemented,
                ),
            ]
            if needed
        ],
    }


def summarize_dcmicrogrid(task_dir: Path, *, after: str | None = None) -> dict:
    """Aggregate DC Microgrid results and save summary JSON; print comparison table."""
    task_cfg = load_task_config(task_dir)
    records = load_manifest_filtered(task_dir, after=after)
    if not records:
        print("[DC Microgrid summarize] No records found in manifest.")
        return {}

    canonical_records = [
        r
        for r in records
        if (r.device or _canonical_device_for_backend(r.backend or "jax_rejax"))
        == _canonical_device_for_backend(r.backend or "jax_rejax")
    ]
    rows, seed_rewards = _aggregate_rows(canonical_records, by_device=False)
    _attach_normscore_and_ood(rows, seed_rewards, by_device=False)

    phase2_rows, phase2_seed_rewards = _aggregate_rows(records, by_device=True)
    _attach_normscore_and_ood(phase2_rows, phase2_seed_rewards, by_device=True)
    audit_available = (task_dir / "results" / "phase2_paper_audit.json").exists()
    for row in phase2_rows:
        mark_dc_phase2_audited(row, audit_available=audit_available)

    hypothesis_tests = _build_hypothesis_tests(task_dir, records, task_cfg)
    protocol_status = _build_protocol_status(rows, task_cfg, hypothesis_tests)

    summary = {
        "task": "dc_microgrid",
        "grouping": "backend_algo_split",
        "phase2_grouping": "backend_device_algo_split",
        "normscore_basis": "episode_reward",
        "rows": rows,
        "phase2_rows": phase2_rows,
        "protocol_status": protocol_status,
        "hypothesis_tests": hypothesis_tests,
        "reporting": {
            "suppressed_rows": [],
            "legacy_default_evidence_tier": "legacy_unknown",
            "cross_backend_gate": "comparison_audited=true and audit_suite_version is required",
            "dc_phase2_audit_source": (
                "results/phase2_paper_audit.json"
                if (task_dir / "results" / "phase2_paper_audit.json").exists()
                else None
            ),
        },
        "metric_sensitivity": metric_sensitivity_report(
            [row for row in rows if row.get("split") == task_cfg.get("primary_split", "iid")],
            primary_metric_key="episode_reward_mean",
            primary_direction="higher_is_better",
        ),
        "filters": {"after": after},
    }
    path = save_summary(summary, task_dir)
    print(f"[DC Microgrid summarize] {len(rows)} canonical rows -> {path}")

    header = (
        "| {:<10} | {:<20} | {:<18} | {:>12} | {:>9} | {:>10} | {:>9} | {:>10} |"
    ).format(
        "Backend", "Algo", "Split", "Reward", "Feasib %", "SLA Viol %", "NormScore", "OOD Gap"
    )
    sep = (
        "|" + "-" * 12 + "|" + "-" * 22 + "|" + "-" * 20 + "|" + ("-" * 14 + "|") * 5
    )
    print()
    print(header)
    print(sep)

    split_order = [
        "train", "iid", "cooling_stress", "renewable_drought",
        "workload_swap", "workload_shock", "dg_derating", "sla_tighten",
    ]
    algo_order = ["no_control", "max_renewable", "rule_based", "ppo", "sac", "sbx_ppo"]
    backend_order = ["jax_rejax", "sb3", "sbx"]

    def _sort_key(r):
        return (
            backend_order.index(r["backend"]) if r["backend"] in backend_order else 99,
            algo_order.index(r["algo"]) if r["algo"] in algo_order else 99,
            split_order.index(r["split"]) if r["split"] in split_order else 99,
        )

    for row in sorted(rows, key=_sort_key):
        print(
            "| {:<10} | {:<20} | {:<18} | {:>12} | {:>9} | {:>10} | {:>9} | {:>10} |".format(
                row["backend"],
                row["algo"],
                row["split"],
                fmt_metric(row.get("episode_reward_mean"), ".2f"),
                fmt_metric(row.get("feasibility_rate_mean"), ".3f"),
                fmt_metric(row.get("sla_violation_rate_mean"), ".3f"),
                fmt_metric(row.get("norm_score"), ".3f"),
                fmt_metric(row.get("ood_robustness_gap"), ".3f"),
            )
        )

    print()
    print("Phase-2 device rows:")
    for row in sorted(
        phase2_rows,
        key=lambda r: (
            backend_order.index(r["backend"]) if r["backend"] in backend_order else 99,
            r["device"],
            algo_order.index(r["algo"]) if r["algo"] in algo_order else 99,
            split_order.index(r["split"]) if r["split"] in split_order else 99,
        ),
    ):
        if row["backend"] == "jax_rejax" and row["device"] == "gpu":
            continue
        print(
            "  {:<10} {:<4} {:<16} {:<18} reward={} n={}".format(
                row["backend"],
                row["device"],
                row["algo"],
                row["split"],
                fmt_metric(row.get("episode_reward_mean"), ".2f"),
                row.get("n_seeds"),
            )
        )

    print()
    print("Cost decomposition (train split, canonical rows):")
    print(
        "| {:<10} | {:<20} | {:>13} | {:>12} | {:>13} |".format(
            "Backend", "Algo", "Energy [MWh]", "Fuel [$]", "Carbon [kgCO2]"
        )
    )
    print("|" + "-" * 12 + "|" + "-" * 22 + "|" + ("-" * 15 + "|") * 3)
    for row in sorted(rows, key=_sort_key):
        if row["split"] != "train":
            continue
        print(
            "| {:<10} | {:<20} | {:>13} | {:>12} | {:>13} |".format(
                row["backend"],
                row["algo"],
                fmt_metric(row.get("total_energy_cost_mean"), ".3f"),
                fmt_metric(row.get("total_fuel_cost_mean"), ".2f"),
                fmt_metric(row.get("total_carbon_kg_mean"), ".2f"),
            )
        )

    print()
    print(
        "[protocol] primary_split={} observed_min_seed_count={} submission_min_seeds={} "
        "submission_ready={}".format(
            protocol_status["primary_split"],
            protocol_status["observed_min_seed_count"],
            protocol_status["submission_min_seeds"],
            protocol_status["current_campaign_submission_ready"],
        )
    )
    for test in hypothesis_tests:
        print(
            "[hypothesis] {} vs {}: n_pairs={} mean_diff={} p={}".format(
                test["left_algo"],
                test["right_algo"],
                test["n_pairs"],
                fmt_metric(test.get("mean_left_minus_right"), ".3f"),
                fmt_metric(test.get("p_value_two_sided"), ".4f"),
            )
        )

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", default=DEFAULT_TASK_DIR, type=Path)
    parser.add_argument("--after", default=None)
    args = parser.parse_args()
    summarize_dcmicrogrid(args.task_dir, after=args.after)
