import json
from pathlib import Path

from benchmarks.common.io import RunRecord
from benchmarks.gencos.summarize import (
    _build_gencos_protocol_status,
    _build_hypothesis_tests,
)


def test_gencos_hypothesis_tests_pair_primary_per_episode_profit(tmp_path: Path):
    task_dir = tmp_path / "gencos"
    artifacts_dir = task_dir / "results" / "artifacts"
    artifacts_dir.mkdir(parents=True)

    def _write(run_id: str, vals: list[float]) -> str:
        rel = f"artifacts/{run_id}_per_episode.json"
        (task_dir / "results" / rel).write_text(
            json.dumps([{"total_profit": v} for v in vals], indent=2),
            encoding="utf-8",
        )
        return rel

    records = [
        RunRecord(
            task="gencos",
            variant="case5",
            algo="ippo",
            seed=0,
            run_id="ippo_s0",
            split="iid",
            backend="jax_rejax",
            device="gpu",
            artifacts={"per_episode": _write("ippo_s0", [30.0, 40.0])},
        ),
        RunRecord(
            task="gencos",
            variant="case5",
            algo="truthful",
            seed=0,
            run_id="truthful_s0",
            split="iid",
            backend="jax_rejax",
            device="gpu",
            artifacts={"per_episode": _write("truthful_s0", [10.0, 20.0])},
        ),
    ]
    cfg = {
        "primary_split": "iid",
        "benchmark_protocol": {
            "leaderboard": {
                "split": "iid",
                "primary_metric": "total_profit",
                "direction": "higher_is_better",
            }
        },
    }

    tests = _build_hypothesis_tests(task_dir, records, cfg)
    ippo_vs_truthful = next(
        t for t in tests if t["left_algo"] == "ippo" and t["right_algo"] == "truthful"
    )

    assert ippo_vs_truthful["n_pairs"] == 2
    assert ippo_vs_truthful["common_seeds"] == [0]
    assert ippo_vs_truthful["mean_left_minus_right"] > 0.0
    assert ippo_vs_truthful["improvement_mean"] > 0.0


def test_gencos_protocol_status_requires_hypothesis_evidence():
    rows = [
        {"algo": "ippo", "split": "iid", "n_seeds": 5},
        {"algo": "truthful", "split": "iid", "n_seeds": 5},
    ]
    cfg = {
        "primary_split": "iid",
        "benchmark_protocol": {
            "submission_min_seeds": 5,
            "current_campaign_seed_budget": 5,
            "protocol_formalization_pending": False,
            "hypothesis_test": {"status": "implemented"},
            "leaderboard": {
                "split": "iid",
                "primary_metric": "total_profit",
                "direction": "higher_is_better",
            },
        },
    }

    no_evidence = _build_gencos_protocol_status(rows, cfg, [])
    with_evidence = _build_gencos_protocol_status(
        rows,
        cfg,
        [{"n_pairs": 2, "p_value_two_sided": 0.5}],
    )

    assert no_evidence["current_campaign_submission_ready"] is False
    assert "run primary-split eval records" in no_evidence["next_actions"][-1]
    assert with_evidence["current_campaign_submission_ready"] is True
