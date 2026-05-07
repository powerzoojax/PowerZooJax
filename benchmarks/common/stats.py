"""Cross-task statistics: NormScore, IQM, bootstrap CI.

These are pure functions operating on arrays of scalar scores.
They do NOT know about task-specific metrics — that logic belongs
in each task's summarize.py.
"""

from __future__ import annotations

import numpy as np


def norm_score(
    score: float | np.ndarray,
    random_score: float,
    oracle_score: float,
) -> float | np.ndarray:
    """Normalize score to [0, 1] using random and oracle baselines.

    NormScore = (score - random) / (oracle - random)

    Returns 0.0 if oracle == random (degenerate case).
    """
    denom = oracle_score - random_score
    if isinstance(denom, (int, float)) and denom == 0:
        return 0.0
    return (score - random_score) / np.where(denom == 0, 1.0, denom)


def compute_norm_score_with_status(
    *,
    value: float | None,
    anchor_a: float | None,
    anchor_b: float | None,
    anchor_metric: str,
    metric_direction: str,
    min_anchor_gap_abs: float = 1e-8,
    min_anchor_gap_rel: float = 0.05,
) -> dict:
    """Compute a diagnostic NormScore with anchor sanity checks.

    ``anchor_a`` is the floor baseline and ``anchor_b`` is the ceiling
    baseline.  For ``higher_is_better`` the ceiling must be numerically larger;
    for ``lower_is_better`` it must be numerically smaller.
    """
    out = {
        "norm_score": None,
        "norm_score_status": "ok",
        "anchor_a": anchor_a,
        "anchor_b": anchor_b,
        "anchor_gap": None,
        "anchor_metric": anchor_metric,
        "metric_direction": metric_direction,
        "norm_score_warning": None,
    }
    if value is None or anchor_a is None or anchor_b is None:
        out["norm_score_status"] = "missing_anchor"
        out["norm_score_warning"] = "value or anchor baseline is missing"
        return out

    value_f = float(value)
    a = float(anchor_a)
    b = float(anchor_b)
    gap = abs(b - a)
    out["anchor_a"] = a
    out["anchor_b"] = b
    out["anchor_gap"] = gap

    if metric_direction == "higher_is_better":
        signed_gap = b - a
        if signed_gap <= 0.0:
            out["norm_score_status"] = "anchor_order_mismatch"
            out["norm_score_warning"] = "ceiling anchor is not better than floor anchor"
            return out
        score = (value_f - a) / signed_gap
    elif metric_direction == "lower_is_better":
        signed_gap = a - b
        if signed_gap <= 0.0:
            out["norm_score_status"] = "anchor_order_mismatch"
            out["norm_score_warning"] = "ceiling anchor is not better than floor anchor"
            return out
        score = (a - value_f) / signed_gap
    else:
        out["norm_score_status"] = "direction_mismatch"
        out["norm_score_warning"] = f"unknown metric_direction={metric_direction!r}"
        return out

    scale = max(abs(a), abs(b), min_anchor_gap_abs)
    if gap < min_anchor_gap_abs or gap / scale < min_anchor_gap_rel:
        out["norm_score_status"] = "unstable_anchor_gap"
        out["norm_score_warning"] = (
            f"anchor gap {gap:.6g} is too small relative to scale {scale:.6g}"
        )
        return out

    out["norm_score"] = float(score)
    return out


def metric_sensitivity_report(
    rows: list[dict],
    *,
    primary_metric_key: str,
    primary_direction: str,
) -> dict:
    """Return raw-vs-NormScore ranking diagnostics for a summary table."""
    reverse = primary_direction == "higher_is_better"
    raw_candidates = [r for r in rows if r.get(primary_metric_key) is not None]
    raw_ranking = [
        r.get("algo")
        for r in sorted(
            raw_candidates,
            key=lambda r: float(r[primary_metric_key]),
            reverse=reverse,
        )
    ]
    norm_candidates = [r for r in rows if r.get("norm_score") is not None]
    norm_ranking = [
        r.get("algo")
        for r in sorted(
            norm_candidates,
            key=lambda r: float(r["norm_score"]),
            reverse=True,
        )
    ]
    gated_candidates = [
        r for r in rows
        if r.get("norm_score_status") == "ok" and r.get("norm_score") is not None
    ]
    gated_norm_ranking = [
        r.get("algo")
        for r in sorted(
            gated_candidates,
            key=lambda r: float(r["norm_score"]),
            reverse=True,
        )
    ]
    return {
        "primary_metric_key": primary_metric_key,
        "primary_direction": primary_direction,
        "raw_ranking": raw_ranking,
        "norm_ranking": norm_ranking,
        "gated_norm_ranking": gated_norm_ranking,
        "ranking_flip": bool(
            raw_ranking and norm_ranking and raw_ranking[: len(norm_ranking)] != norm_ranking
        ),
    }


def iqm(scores: np.ndarray) -> float:
    """Inter-Quartile Mean: mean of the middle 50% of scores.

    More robust than mean, less noisy than median.
    Recommended by Agarwal et al. (2021) for RL evaluation.
    """
    scores = np.asarray(scores)
    if scores.size == 0:
        return float("nan")
    q25 = np.percentile(scores, 25)
    q75 = np.percentile(scores, 75)
    mask = (scores >= q25) & (scores <= q75)
    if not mask.any():
        return float(np.median(scores))
    return float(np.mean(scores[mask]))


def bootstrap_ci(
    scores: np.ndarray,
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    stat_fn=np.mean,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for a statistic.

    Returns (point_estimate, ci_low, ci_high).
    """
    scores = np.asarray(scores)
    rng = rng or np.random.default_rng(42)
    n = len(scores)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))

    point = float(stat_fn(scores))
    if stat_fn is np.mean:
        sample_idx = rng.integers(0, n, size=(n_bootstrap, n))
        boot_stats = np.mean(scores[sample_idx], axis=1)
    else:
        boot_stats = np.array([
            stat_fn(rng.choice(scores, size=n, replace=True))
            for _ in range(n_bootstrap)
        ])
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_stats, 100 * alpha))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha)))
    return (point, lo, hi)


def numeric_only(d: dict) -> dict:
    """Return only numeric (int/float), non-bool, non-None values from ``d``.

    Used by each task's summarize.py to drop string/dict entries from
    RunRecord.metrics before passing to :func:`aggregate_seeds`.
    """
    return {
        k: v for k, v in d.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v is not None
    }


def get_row_mean(rows: list[dict], algo: str, split: str, key: str):
    """Return ``rows[key]`` for the row matching ``(algo, split)``, or ``None``."""
    for r in rows:
        if r["algo"] == algo and r["split"] == split:
            return r.get(key)
    return None


def fmt_metric(val, fmt: str = ".3f") -> str:
    """Format a metric value for console tables; returns ``'  N/A  '`` when absent."""
    if val is None:
        return "  N/A  "
    try:
        return format(float(val), fmt)
    except (TypeError, ValueError):
        return "  N/A  "


def aggregate_seeds(
    metrics_per_seed: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Aggregate metric dicts across seeds.

    Returns {metric_name: {"mean": ..., "std": ..., "iqm": ...,
                           "ci_lo": ..., "ci_hi": ...}}.
    """
    if not metrics_per_seed:
        return {}

    keys = metrics_per_seed[0].keys()
    result = {}
    for k in keys:
        vals = np.array([m[k] for m in metrics_per_seed if k in m])
        if vals.size == 0:
            continue
        point, ci_lo, ci_hi = bootstrap_ci(vals)
        result[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "iqm": iqm(vals),
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "n_seeds": int(vals.size),
        }
    return result
