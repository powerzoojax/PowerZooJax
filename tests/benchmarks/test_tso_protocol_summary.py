"""TSO benchmark protocol helpers should gate unsafe or underpowered results."""

from __future__ import annotations

import json

from benchmarks.common.io import RunRecord
from benchmarks.tso.summarize import (
    build_tso_hypothesis_tests,
    build_tso_leaderboard,
    build_tso_protocol_status,
)


def _task_cfg() -> dict:
    return {
        "primary_split": "iid",
        "baseline_set": ["all_on", "merit_order"],
        "safety_thresholds": {
            "reserve_shortfall_rate": 0.0,
            "thermal_violation_rate": 0.0,
        },
        "benchmark_protocol": {
            "submission_min_seeds": 5,
            "hypothesis_test": {
                "name": "paired_wilcoxon_or_permutation",
                "status": "required_not_yet_implemented",
            },
            "leaderboard": {
                "split": "iid",
                "primary_metric": "total_operating_cost",
                "direction": "lower_is_better",
                "include_baselines": True,
                "require_safety_thresholds": [
                    "reserve_shortfall_rate",
                    "thermal_violation_rate",
                ],
            },
        },
        "target_return_metric_key": "total_operating_cost",
        "target_metric_direction": "lower_is_better",
    }


def test_tso_leaderboard_reports_unsafe_rows_but_does_not_rank_them():
    rows = [
        {
            "algo": "ppo",
            "split": "iid",
            "n_seeds": 5,
            "total_operating_cost_mean": 100.0,
            "reserve_shortfall_rate_mean": 1.0,
            "thermal_violation_rate_mean": 0.0,
            "norm_score": 1.4,
        },
        {
            "algo": "ppo_lagrangian",
            "split": "iid",
            "n_seeds": 5,
            "total_operating_cost_mean": 110.0,
            "reserve_shortfall_rate_mean": 0.0,
            "thermal_violation_rate_mean": 0.0,
            "norm_score": 1.2,
        },
    ]

    leaderboard = build_tso_leaderboard(rows, _task_cfg())

    assert leaderboard[0]["algo"] == "ppo_lagrangian"
    assert leaderboard[0]["leaderboard_rank"] == 1
    assert leaderboard[0]["leaderboard_eligible"] is True

    unsafe = next(entry for entry in leaderboard if entry["algo"] == "ppo")
    assert unsafe["leaderboard_eligible"] is False
    assert unsafe["leaderboard_rank"] is None
    assert any("reserve_shortfall_rate" in reason for reason in unsafe["exclusion_reasons"])


def test_tso_protocol_status_flags_seed_gap_and_missing_hypothesis_test():
    rows = [
        {
            "algo": "ppo_lagrangian",
            "split": "iid",
            "n_seeds": 3,
            "total_operating_cost_mean": 110.0,
            "reserve_shortfall_rate_mean": 0.0,
            "thermal_violation_rate_mean": 0.0,
            "norm_score": 1.2,
        },
    ]
    cfg = _task_cfg()
    leaderboard = build_tso_leaderboard(rows, cfg)

    protocol = build_tso_protocol_status(rows, cfg, leaderboard)

    assert protocol["minimum_seed_requirement_met"] is False
    assert protocol["observed_min_seed_count"] == 3
    assert protocol["hypothesis_test_implemented"] is False
    assert protocol["current_campaign_submission_ready"] is False


def test_tso_protocol_status_requires_hypothesis_artifacts_even_when_implemented():
    rows = [
        {
            "algo": "ppo_lagrangian",
            "split": "iid",
            "n_seeds": 5,
            "total_operating_cost_mean": 110.0,
            "reserve_shortfall_rate_mean": 0.0,
            "thermal_violation_rate_mean": 0.0,
            "norm_score": 1.2,
        },
    ]
    cfg = _task_cfg()
    cfg["seeds"] = [0, 1, 2, 3, 4]
    cfg["benchmark_protocol"]["hypothesis_test"]["status"] = "implemented"
    leaderboard = build_tso_leaderboard(rows, cfg)

    protocol = build_tso_protocol_status(rows, cfg, leaderboard, hypothesis_tests=[])

    assert protocol["minimum_seed_requirement_met"] is True
    assert protocol["hypothesis_test_implemented"] is True
    assert protocol["hypothesis_tests_available"] is False
    assert protocol["current_campaign_submission_ready"] is False


def test_tso_protocol_status_ignores_non_primary_ablation_algorithms_for_seed_floor():
    rows = [
        {
            "algo": "all_on",
            "split": "iid",
            "n_seeds": 5,
            "total_operating_cost_mean": 120.0,
            "reserve_shortfall_rate_mean": 0.0,
            "thermal_violation_rate_mean": 0.0,
            "norm_score": 0.0,
        },
        {
            "algo": "ppo_lagrangian",
            "split": "iid",
            "n_seeds": 5,
            "total_operating_cost_mean": 110.0,
            "reserve_shortfall_rate_mean": 0.0,
            "thermal_violation_rate_mean": 0.0,
            "norm_score": 1.2,
        },
        {
            "algo": "saute_ppo",
            "split": "iid",
            "n_seeds": 1,
            "total_operating_cost_mean": 105.0,
            "reserve_shortfall_rate_mean": 0.0,
            "thermal_violation_rate_mean": 0.0,
            "norm_score": 1.4,
        },
    ]
    cfg = _task_cfg()
    cfg["benchmark_protocol"]["primary_algorithms"] = ["ppo_lagrangian"]
    cfg["benchmark_protocol"]["hypothesis_test"]["status"] = "implemented"
    leaderboard = build_tso_leaderboard(rows, cfg)

    protocol = build_tso_protocol_status(
        rows,
        cfg,
        leaderboard,
        hypothesis_tests=[{"n_pairs": 1}],
    )

    assert protocol["per_algo_seed_count_primary_split"] == {
        "all_on": 5,
        "ppo_lagrangian": 5,
    }
    assert protocol["observed_min_seed_count"] == 5
    assert protocol["minimum_seed_requirement_met"] is True


def test_tso_hypothesis_tests_use_paired_per_episode_costs(tmp_path):
    task_dir = tmp_path / "tso"
    artifacts_dir = task_dir / "results" / "artifacts"
    artifacts_dir.mkdir(parents=True)

    records = []
    for algo, costs in {
        "ppo_lagrangian": [10.0, 11.0, 12.0],
        "all_on": [20.0, 21.0, 22.0],
        "ppo": [15.0, 16.0, 17.0],
    }.items():
        run_id = f"{algo}_s0"
        rel = f"artifacts/{run_id}_per_episode.json"
        (artifacts_dir / f"{run_id}_per_episode.json").write_text(
            json.dumps(
                [{"total_operating_cost": cost} for cost in costs],
                indent=2,
            ),
            encoding="utf-8",
        )
        records.append(
            RunRecord(
                task="tso",
                variant="tso_scuc",
                algo=algo,
                seed=0,
                run_id=run_id,
                split="iid",
                backend="jax_rejax",
                device="gpu",
                artifacts={"per_episode": rel},
            )
        )

    tests = build_tso_hypothesis_tests(task_dir, records, _task_cfg())
    lag_vs_all_on = next(
        test
        for test in tests
        if test["left_algo"] == "ppo_lagrangian" and test["right_algo"] == "all_on"
    )

    assert lag_vs_all_on["n_pairs"] == 3
    assert lag_vs_all_on["common_seeds"] == [0]
    assert lag_vs_all_on["mean_left_minus_right"] < 0.0
    assert lag_vs_all_on["left_better"] is True
