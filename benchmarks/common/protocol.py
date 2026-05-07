"""Shared benchmark protocol-status helpers."""

from __future__ import annotations

from typing import Any


def build_protocol_status(
    rows: list[dict[str, Any]],
    task_cfg: dict[str, Any],
    *,
    default_primary_metric: str,
    default_direction: str,
) -> dict[str, Any]:
    """Return conservative submission-readiness status for a task summary."""
    protocol = task_cfg.get("benchmark_protocol") or {}
    leaderboard = protocol.get("leaderboard") or {}
    primary_split = leaderboard.get("split") or task_cfg.get("primary_split", "iid")
    primary_metric = leaderboard.get("primary_metric") or default_primary_metric
    direction = leaderboard.get("direction") or default_direction
    submission_min_seeds = int(protocol.get("submission_min_seeds", 5))
    configured_seed_budget = int(
        protocol.get("current_campaign_seed_budget")
        or len(task_cfg.get("seeds") or [])
    )
    primary_rows = [row for row in rows if row.get("split") == primary_split]
    per_algo_seed_count = {
        str(row["algo"]): int(row.get("n_seeds", 0))
        for row in primary_rows
    }
    observed_min_seed_count = min(per_algo_seed_count.values(), default=0)
    minimum_seed_requirement_met = bool(primary_rows) and all(
        count >= submission_min_seeds for count in per_algo_seed_count.values()
    )

    hypothesis_cfg = protocol.get("hypothesis_test") or {}
    hypothesis_status = str(hypothesis_cfg.get("status", "required_not_yet_implemented"))
    hypothesis_test_implemented = hypothesis_status == "implemented"
    protocol_pending = bool(protocol.get("protocol_formalization_pending", False))

    next_actions: list[str] = []
    if protocol_pending:
        next_actions.append("formalize benchmark protocol and statistical test")
    if not minimum_seed_requirement_met:
        next_actions.append(
            f"collect at least {submission_min_seeds} seeds for every primary-split row"
        )
    if not hypothesis_test_implemented:
        next_actions.append("implement the declared primary-split hypothesis test")

    return {
        "primary_split": primary_split,
        "primary_metric": primary_metric,
        "metric_direction": direction,
        "configured_seed_budget": configured_seed_budget,
        "submission_min_seeds": submission_min_seeds,
        "observed_min_seed_count": observed_min_seed_count,
        "per_algo_seed_count_primary_split": per_algo_seed_count,
        "minimum_seed_requirement_met": minimum_seed_requirement_met,
        "hypothesis_test": hypothesis_cfg,
        "hypothesis_test_implemented": hypothesis_test_implemented,
        "protocol_formalization_pending": protocol_pending,
        "current_campaign_submission_ready": (
            minimum_seed_requirement_met
            and hypothesis_test_implemented
            and not protocol_pending
        ),
        "next_actions": next_actions,
    }

