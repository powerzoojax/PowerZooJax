"""Runtime config helpers for the TSO benchmark.

This keeps the benchmark scripts honest about which task-level fields are
actually executable inputs versus metadata-only protocol fields.
"""

from __future__ import annotations

from typing import Any


def validate_task_config(task_config: dict[str, Any]) -> None:
    """Fail fast when task.yaml claims a TSO variant this pipeline cannot run."""
    from powerzoojax.case import load_case

    if task_config.get("task", "tso") != "tso":
        raise ValueError(f"Expected task='tso', got {task_config.get('task')!r}")
    if task_config.get("case", "case118") != "case118":
        raise ValueError(
            "TSO benchmark scripts currently support only case118; "
            f"got case={task_config.get('case')!r}"
        )
    if task_config.get("data_source", "gb") != "gb":
        raise ValueError(
            "TSO benchmark scripts currently support only real GB data; "
            f"got data_source={task_config.get('data_source')!r}"
        )

    case = load_case("118")
    expected_counts = {
        "n_units": int(case.n_units),
        "n_buses": int(case.n_nodes),
        "n_lines": int(case.n_lines),
    }
    for key, expected in expected_counts.items():
        if key in task_config and int(task_config[key]) != expected:
            raise ValueError(
                f"TSO task config mismatch: {key}={task_config[key]!r}, "
                f"but case118 requires {expected}"
            )


def resolve_gb_windows(task_config: dict[str, Any]) -> dict[str, tuple[str, str]]:
    """Resolve executable GB date windows from task.yaml with frozen fallback.

    ``task.yaml`` is the benchmark-facing declaration point. When a field is
    absent, the frozen defaults from ``powerzoojax.data.splits`` are used so
    the current benchmark window remains reproducible by default.
    """
    from powerzoojax.data.splits import (
        GB_IID_END,
        GB_IID_START,
        GB_TRAIN_END,
        GB_TRAIN_START,
        _parse,
    )

    windows = {
        "train": (
            str(task_config.get("gb_train_start", GB_TRAIN_START)),
            str(task_config.get("gb_train_end", GB_TRAIN_END)),
        ),
        "iid": (
            str(task_config.get("gb_iid_start", GB_IID_START)),
            str(task_config.get("gb_iid_end", GB_IID_END)),
        ),
    }

    for gb_split, (start, end) in windows.items():
        if _parse(start) >= _parse(end):
            raise ValueError(
                f"TSO GB window for {gb_split!r} must satisfy start < end, "
                f"got start={start!r}, end={end!r}"
            )

    train_end = windows["train"][1]
    iid_start = windows["iid"][0]
    if _parse(train_end) >= _parse(iid_start):
        raise ValueError(
            "TSO GB windows must be non-overlapping with train before iid, "
            f"got train_end={train_end!r} and iid_start={iid_start!r}"
        )

    return windows


def _expected_gb_split_for(split: str) -> str:
    """Return the logical GB window name backing a benchmark split."""
    split_map = {
        "train": "train",
        "iid": "iid",
        "load_stress": "iid",
        "line_tightening": "iid",
    }
    if split not in split_map:
        raise ValueError(
            f"Unknown TSO split {split!r}. Expected one of {tuple(split_map)}"
        )
    return split_map[split]


def get_eval_gb_split(split: str, eval_config: dict[str, Any] | None = None) -> str:
    """Return the GB window role for an eval split and reject YAML drift.

    ``eval_*.yaml`` may still carry legacy ``role`` / ``gb_split`` fields.
    They are now treated as executable guardrails instead of dead comments.
    """
    expected = _expected_gb_split_for(split)
    if eval_config is None:
        return expected

    declared_split = eval_config.get("split")
    if declared_split is not None and str(declared_split) != split:
        raise ValueError(
            f"TSO eval split drift: requested split={split!r}, "
            f"but eval config declares split={declared_split!r}"
        )

    declared_role = eval_config.get("role")
    if declared_role is not None and str(declared_role) != expected:
        raise ValueError(
            f"TSO eval role drift for split={split!r}: expected role={expected!r}, "
            f"but eval config declares role={declared_role!r}"
        )

    declared_gb_split = eval_config.get("gb_split")
    if declared_gb_split is not None and str(declared_gb_split) != expected:
        raise ValueError(
            f"TSO eval gb_split drift for split={split!r}: expected gb_split="
            f"{expected!r}, but eval config declares gb_split="
            f"{declared_gb_split!r}"
        )
    return expected


def make_task_from_config(
    task_config: dict[str, Any],
    *,
    load_scale: float = 1.0,
    line_rating_scale: float = 1.0,
):
    """Construct TSOTask from the benchmark task config."""
    from powerzoojax.tasks.tso import TSOTask

    validate_task_config(task_config)
    gb_windows = resolve_gb_windows(task_config)
    return TSOTask(
        max_steps=int(task_config.get("max_steps", 48)),
        dt_hours=float(task_config.get("dt_hours", 0.5)),
        reserve_margin_frac=float(task_config.get("reserve_margin_frac", 0.05)),
        reward_scale=float(task_config.get("reward_scale", 1e-4)),
        cost_thermal_weight=float(task_config.get("cost_thermal_weight", 1.0)),
        solver_mode=int(task_config.get("solver_mode", 1)),
        dcopf_max_iter=int(task_config.get("dcopf_max_iter", 100)),
        dcopf_tol=float(task_config.get("dcopf_tol", 1e-3)),
        forecast_horizon_steps=int(task_config.get("forecast_horizon_steps", 0)),
        load_scale=float(load_scale),
        line_rating_scale=float(line_rating_scale),
        gb_train_start=gb_windows["train"][0],
        gb_train_end=gb_windows["train"][1],
        gb_iid_start=gb_windows["iid"][0],
        gb_iid_end=gb_windows["iid"][1],
    )


def get_eval_episodes(task_config: dict[str, Any], eval_config: dict[str, Any] | None = None) -> int:
    """Return the single source-of-truth evaluation episode count.

    ``task.yaml::eval_episodes`` is the executable source. Split configs may
    still carry a legacy ``n_eval_episodes`` field; when present it must match.
    """
    episodes = int(task_config.get("eval_episodes", 50))
    if eval_config is not None and "n_eval_episodes" in eval_config:
        split_episodes = int(eval_config["n_eval_episodes"])
        if split_episodes != episodes:
            split_name = eval_config.get("split", "<unknown>")
            raise ValueError(
                f"TSO eval episode drift: task.yaml eval_episodes={episodes}, "
                f"but eval_{split_name}.yaml declares n_eval_episodes={split_episodes}"
            )
    return episodes


def get_eval_splits(task_config: dict[str, Any]) -> tuple[str, ...]:
    """Return the executable split list from task.yaml."""
    splits = task_config.get("eval_splits")
    if not splits:
        return ("train", "iid", "load_stress", "line_tightening")
    return tuple(str(s) for s in splits)


def get_baseline_set(task_config: dict[str, Any]) -> tuple[str, ...]:
    """Return the non-learning baseline list from task.yaml."""
    baselines = task_config.get("baseline_set") or ("all_on", "merit_order")
    return tuple(str(name) for name in baselines)
