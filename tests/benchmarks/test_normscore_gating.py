from benchmarks.common.stats import (
    compute_norm_score_with_status,
    metric_sensitivity_report,
)


def test_normscore_missing_anchor_is_null():
    out = compute_norm_score_with_status(
        value=1.0,
        anchor_a=None,
        anchor_b=2.0,
        anchor_metric="reward",
        metric_direction="higher_is_better",
    )

    assert out["norm_score"] is None
    assert out["norm_score_status"] == "missing_anchor"


def test_normscore_unstable_anchor_gap_is_not_headline_candidate():
    out = compute_norm_score_with_status(
        value=90.0,
        anchor_a=100.0,
        anchor_b=99.0,
        anchor_metric="loss",
        metric_direction="lower_is_better",
    )

    assert out["norm_score"] is None
    assert out["norm_score_status"] == "unstable_anchor_gap"
    assert out["anchor_gap"] == 1.0


def test_normscore_anchor_order_mismatch_is_flagged():
    out = compute_norm_score_with_status(
        value=-10.0,
        anchor_a=-20.0,
        anchor_b=-30.0,
        anchor_metric="episode_reward",
        metric_direction="higher_is_better",
    )

    assert out["norm_score"] is None
    assert out["norm_score_status"] == "anchor_order_mismatch"


def test_metric_sensitivity_reports_ranking_flip():
    rows = [
        {"algo": "a", "episode_reward_mean": 10.0, "norm_score": 0.1, "norm_score_status": "ok"},
        {"algo": "b", "episode_reward_mean": 9.0, "norm_score": 0.2, "norm_score_status": "ok"},
    ]

    report = metric_sensitivity_report(
        rows,
        primary_metric_key="episode_reward_mean",
        primary_direction="higher_is_better",
    )

    assert report["raw_ranking"] == ["a", "b"]
    assert report["norm_ranking"] == ["b", "a"]
    assert report["ranking_flip"] is True

