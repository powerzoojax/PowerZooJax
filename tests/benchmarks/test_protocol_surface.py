from pathlib import Path

from benchmarks.common.configs import load_task_config
from benchmarks.common.protocol import build_protocol_status


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"


def test_submission_protocol_surface_declares_implemented_primary_test(
):
    expected = {
        "ders": ("mean_p_loss_mw", "lower_is_better"),
        "gencos": ("total_profit", "higher_is_better"),
    }
    for task, (primary_metric, direction) in expected.items():
        cfg = load_task_config(BENCHMARKS_DIR / task)
        protocol = cfg["benchmark_protocol"]
        assert protocol["protocol_formalization_pending"] is False
        assert protocol["submission_min_seeds"] == 5
        assert protocol["current_campaign_seed_budget"] == 5
        assert protocol["hypothesis_test"]["status"] == "implemented"
        assert protocol["hypothesis_test"]["metric"] == primary_metric
        assert protocol["leaderboard"]["split"] == cfg["primary_split"]
        assert protocol["leaderboard"]["primary_metric"] == primary_metric
        assert protocol["leaderboard"]["direction"] == direction


def test_pending_protocol_status_is_not_submission_ready():
    cfg = {
        "primary_split": "iid",
        "seeds": [0, 1, 2],
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
    rows = [
        {"algo": "ppo", "split": "iid", "n_seeds": 3},
        {"algo": "no_control", "split": "iid", "n_seeds": 3},
    ]

    status = build_protocol_status(
        rows,
        cfg,
        default_primary_metric="total_loss_mwh",
        default_direction="lower_is_better",
    )

    assert status["configured_seed_budget"] == 3
    assert status["observed_min_seed_count"] == 3
    assert status["minimum_seed_requirement_met"] is False
    assert status["hypothesis_test_implemented"] is False
    assert status["current_campaign_submission_ready"] is False
    assert status["protocol_formalization_pending"] is True
