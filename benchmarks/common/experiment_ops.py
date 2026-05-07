"""Benchmark experiment-ops helpers and CLIs.

This file intentionally keeps only the benchmark-ops code that is still
wired into the current experiment workflow: target derivation, seed-0
readiness, variance checks, and a small amount of record repair tooling.
"""

import argparse
import json
import math
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.common.configs import dump_json, load_task_config
from benchmarks.common.io import (
    RunRecord,
    dedup_keep_artifacts,
    load_manifest,
    load_manifest_filtered,
)
from benchmarks.common.powerzoo_bridge import JAX_TASK_TO_POWERZOO_TASK

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BENCHMARKS_DIR = _REPO_ROOT / "benchmarks"
_PREFLIGHT_TASKS = ("dso", "tso", "ders", "gencos", "dc_microgrid")
_KNOWN_SPLIT_TOKENS = {
    "train",
    "iid",
    "summer_ood",
    "zone_holdout",
    "load_stress",
    "line_tightening",
    "voltage_tightening",
    "pv_penetration_shift",
    "demand_shift",
    "renewable_shock",
    "cooling_stress",
    "renewable_drought",
    "workload_swap",
    "workload_shock",
    "dg_derating",
    "sla_tighten",
}


def _resolve_task_dir(task: str, task_dir: str | Path | None = None) -> Path:
    """Resolve the benchmark task directory, allowing isolated campaign roots."""
    if task_dir is None:
        return _BENCHMARKS_DIR / task
    return Path(task_dir)


def _json_file(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _configured_splits(cfg: dict[str, Any]) -> set[str]:
    splits = set(cfg.get("eval_splits") or [])
    splits.update(cfg.get("eval_splits_main") or [])
    splits.update(cfg.get("eval_splits_appendix") or [])
    primary = cfg.get("primary_split")
    if primary:
        splits.add(primary)
    return splits


def _summary_splits(summary: dict[str, Any]) -> set[str]:
    splits: set[str] = set()
    for key in ("rows", "phase2_rows", "leaderboard_primary_split"):
        for row in summary.get(key) or []:
            split = row.get("split")
            if split:
                splits.add(str(split))
    return splits


def _doc_paths_for_task(task: str, task_dir: Path) -> list[Path]:
    doc_name = "dc-microgrid" if task == "dc_microgrid" else task
    return [
        _REPO_ROOT / "docs" / "en" / "benchmarks" / f"{doc_name}.md",
        _REPO_ROOT / "docs" / "zh" / "benchmarks" / f"{doc_name}.md",
        task_dir / "README.md",
    ]


def _readiness_words(text: str) -> set[str]:
    """Detect current-status claims without matching words like ``already``.

    Phrases such as "submission-grade minimum" or "for submission-grade
    reporting after the campaign" describe requirements, not current
    readiness.  They should not block a task whose summary correctly says
    ``current_campaign_submission_ready=false``.
    """
    lower = text.lower()
    found: set[str] = set()
    if re.search(r"\bblocked\b", lower):
        found.add("blocked")
    if re.search(r"\bready\b", lower):
        found.add("ready")
    if re.search(r"\bsubmission[- ]ready\b", lower):
        found.add("submission-ready")
    if re.search(r"\b(current|currently|now|is|are|already)\s+submission[- ]grade\b", lower):
        found.add("submission-grade")
    return found


def benchmark_preflight(
    *,
    task: str,
    task_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Check executable benchmark truth against docs and summary surfaces."""
    tasks = list(_PREFLIGHT_TASKS) if task == "all" else [task]
    report: dict[str, Any] = {
        "ok": True,
        "errors": [],
        "warnings": [],
        "tasks": {},
    }
    for task_name in tasks:
        td = _resolve_task_dir(task_name, task_dir if task != "all" else None)
        task_report: dict[str, Any] = {
            "task_dir": str(td),
            "errors": [],
            "warnings": [],
        }
        report["tasks"][task_name] = task_report

        cfg_path = td / "configs" / "task.yaml"
        provenance_path = td / "configs" / "provenance.json"
        manifest_path = td / "results" / "manifest.json"
        summary_path = td / "results" / "summary" / "latest.json"
        for label, path in (
            ("task_config", cfg_path),
            ("provenance", provenance_path),
            ("manifest", manifest_path),
            ("summary", summary_path),
        ):
            if not path.exists():
                task_report["errors"].append(f"missing_{label}:{path}")

        if task_report["errors"]:
            continue

        cfg = load_task_config(td)
        summary = _json_file(summary_path)
        if not isinstance(summary, dict):
            task_report["errors"].append(f"summary_not_object:{summary_path}")
            continue

        eval_splits = _configured_splits(cfg)
        primary_split = cfg.get("primary_split")
        if primary_split not in eval_splits:
            task_report["errors"].append(
                f"primary_split_not_configured:{primary_split}"
            )

        for split in sorted(_summary_splits(summary) - eval_splits):
            task_report["errors"].append(f"summary_split_not_configured:{split}")

        docs_text = ""
        doc_allowed_split_tokens = set(eval_splits)
        # Documentation must be allowed to describe the training role even
        # when formal evaluation is limited to the primary split.
        doc_allowed_split_tokens.add("train")
        for path in _doc_paths_for_task(task_name, td):
            if not path.exists():
                task_report["warnings"].append(f"missing_doc:{path}")
                continue
            text = path.read_text(encoding="utf-8")
            docs_text += "\n" + text
            mentioned = {tok for tok in _KNOWN_SPLIT_TOKENS if tok in text}
            for split in sorted(mentioned - doc_allowed_split_tokens):
                task_report["errors"].append(
                    f"doc_mentions_unconfigured_split:{path}:{split}"
                )

        protocol_status = summary.get("protocol_status") or {}
        ready = bool(protocol_status.get("current_campaign_submission_ready", False))
        doc_words = _readiness_words(docs_text)
        if ready and "blocked" in doc_words:
            task_report["errors"].append("docs_say_blocked_but_summary_ready")
        if not ready and (
            "ready" in doc_words
            or "submission-ready" in doc_words
            or "submission grade" in doc_words
            or "submission-grade" in doc_words
        ):
            task_report["errors"].append("docs_claim_ready_but_summary_not_ready")

    for task_name, task_report in report["tasks"].items():
        for err in task_report["errors"]:
            report["errors"].append(f"{task_name}: {err}")
        for warn in task_report["warnings"]:
            report["warnings"].append(f"{task_name}: {warn}")
    report["ok"] = not report["errors"]
    return report


def benchmark_preflight_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check benchmark executable truth surfaces.")
    parser.add_argument("--task", required=True, help="Task name or 'all'.")
    parser.add_argument("--task-dir", default=None)
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args(argv)

    report = benchmark_preflight(task=args.task, task_dir=args.task_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.enforce and not report["ok"]:
        return 2
    return 0

def _iqm(values: list[float]) -> float:
    """Interquartile mean (drop top/bottom 25%)."""
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 4:
        return float(np.mean(arr))
    q25, q75 = np.quantile(arr, [0.25, 0.75])
    mask = (arr >= q25) & (arr <= q75)
    if not bool(np.any(mask)):
        return float(np.mean(arr))
    return float(np.mean(arr[mask]))

def _load_reference_final_return(task_dir: Path, run_id: str) -> float:
    """Find the reference run's final_return.

    Priority:
      1. If a canonical eval curve artifact exists, take its last value.
      2. Else any non-canonical eval curve artifact (eval_returns / mean_reward).
      3. Else fall back to RunRecord.metrics' final_return-ish key.
    """
    runs = load_manifest(task_dir)
    rec = next((r for r in runs if r.run_id == run_id), None)
    if rec is None:
        raise ValueError(
            f"Reference run_id '{run_id}' not found in {task_dir}/results/manifest.json"
        )

    # 1. Preferred: canonical eval learning-curve artifact
    art_rel = rec.artifacts.get("learning_curve_eval_return")
    if art_rel is not None:
        path = task_dir / "results" / art_rel
        if path.exists():
            arr = np.load(path)
            if arr.size > 0:
                return float(arr.flatten()[-1])

    # 2. Legacy artifact keys from older RunRecords
    for art_key in ("eval_returns", "mean_reward", "ep_rew_mean", "eval_episode_returns"):
        art_rel = rec.artifacts.get(art_key)
        if art_rel is None:
            continue
        path = task_dir / "results" / art_rel
        if path.exists():
            arr = np.load(path)
            if arr.size > 0:
                return float(arr.flatten()[-1])

    # 3. Fallback: scan metrics for plausible final-return keys
    for key in ("final_return", "eval_return", "mean_reward", "return", "ep_rew_mean", "final_reward"):
        v = rec.metrics.get(key)
        if v is None:
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue

    raise ValueError(
        f"Reference run '{run_id}' has neither learning_curve_eval_return artifact, "
        f"nor a non-canonical eval curve artifact (eval_returns / mean_reward / "
        f"ep_rew_mean / eval_episode_returns), nor a recognised final-return metric key."
    )

# Canonical priority list of metric keys that count as "per-episode return"
# (in descending preference). Both reference run and no_control runs need to
# expose the SAME key to make the alpha-interpolated target meaningful;
# derive_target picks the first key that is present in BOTH sets.
_RETURN_KEY_PRIORITY = (
    "final_return",
    "eval_return",
    "mean_reward",
    "return",
    "ep_rew_mean",
    "total_reward",   # DSO baselines / older records
    "final_reward",   # some older train records
)

def _pick_return_key(*records) -> str:
    """Pick the highest-priority return-ish metric key present in all records.

    Raises ValueError if no key is shared.
    """
    common: "set[str] | None" = None
    for r in records:
        present = {k for k in _RETURN_KEY_PRIORITY if k in r.metrics}
        common = present if common is None else common & present
    common = common or set()
    for k in _RETURN_KEY_PRIORITY:
        if k in common:
            return k
    keys_seen = [sorted(r.metrics.keys()) for r in records]
    raise ValueError(
        f"No shared return-ish metric key across records. Seen keys per record: "
        f"{keys_seen}. Recognised keys: {_RETURN_KEY_PRIORITY}"
    )

def _baseline_iqm(
    task_dir: Path,
    split: str,
    baseline_algo: str,
    return_key: str | None = None,
) -> tuple[float, list[str], str]:
    """Compute IQM of the specified baseline algo's records on the given split.

    Returns ``(iqm, run_ids, used_metric_key)``. If ``return_key`` is None,
    autodetects via :func:`_pick_return_key`; the chosen key is returned so
    the caller can use the same key when reading the reference run.

    The baseline algo is task-specific (``no_control`` for DSO/DERs/DC Microgrid;
    ``all_on`` for TSO; ``truthful`` for GenCos) and must be set explicitly
    by the caller — usually via task config ``derive_target_baseline_algo`` or
    falling back to baseline_set[0].
    """
    runs = load_manifest(task_dir)
    nc_runs = [
        r for r in runs
        if r.algo == baseline_algo
        and r.split == split
        and r.status in ("completed", "converged_warning")
    ]
    if not nc_runs:
        raise ValueError(
            f"No baseline records (algo={baseline_algo!r}) found for split={split} "
            f"in {task_dir}/results/manifest.json. Run baselines first."
        )
    if return_key is None:
        return_key = _pick_return_key(*nc_runs)
    values: list[float] = []
    for r in nc_runs:
        v = r.metrics.get(return_key)
        if v is None:
            continue
        try:
            values.append(float(v))
        except (TypeError, ValueError):
            continue
    if not values:
        raise ValueError(
            f"{baseline_algo!r} records exist on split={split} but key "
            f"{return_key!r} is missing from all of them."
        )
    return _iqm(values), [r.run_id for r in nc_runs], return_key

def derive_target(
    task: str,
    reference_run_id: str,
    split: str | None = None,
    alpha: float = 0.8,
    force: bool = False,
    dry_run: bool = False,
    task_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Derive and write convergence target back to provenance.json.

    Returns the patch dict (target + provenance) that was written.
    """
    task_dir_path = _resolve_task_dir(task, task_dir)
    configs_dir = task_dir_path / "configs"
    provenance_path = configs_dir / "provenance.json"
    # Read merged (task.yaml + provenance.json) so logic works on the same
    # dict shape as before the YAML migration.  Writes go ONLY to
    # provenance.json (task.yaml is human-edited and must stay untouched).
    cfg = load_task_config(task_dir_path)

    # Resolve split: default to primary_split
    if split is None:
        split = cfg.get("primary_split")
        if split is None:
            raise ValueError(
                f"task config for {task} has no primary_split; pass --split explicitly."
            )

    # alpha override resolution
    alpha_used = alpha
    convergence_target_cfg = cfg.get("convergence_target") or {}
    if "alpha" in convergence_target_cfg:
        alpha_used = float(convergence_target_cfg["alpha"])

    # Resolve the reference baseline algo (no_control / all_on / truthful / ...)
    # from the merged task config. baseline_set[0] is the default; an explicit
    # derive_target_baseline_algo overrides.
    baseline_set = cfg.get("baseline_set") or cfg.get("baseline_algos") or []
    baseline_algo = (
        cfg.get("derive_target_baseline_algo")
        or (baseline_set[0] if baseline_set else None)
    )
    if not baseline_algo:
        raise ValueError(
            f"task config for {task} has neither derive_target_baseline_algo "
            f"nor a non-empty baseline_set; cannot compute baseline IQM."
        )

    # Resolve the target return metric key + direction.
    # If the task config declares them, use as-is; otherwise auto-detect via
    # _pick_return_key (works for tasks with standard reward keys).
    explicit_metric_key = cfg.get("target_return_metric_key")
    metric_direction = cfg.get("target_metric_direction", "higher_is_better")
    if metric_direction not in ("higher_is_better", "lower_is_better"):
        raise ValueError(
            f"task config target_metric_direction must be 'higher_is_better' or "
            f"'lower_is_better', got {metric_direction!r}"
        )

    # Pull baseline IQM first; this also fixes the canonical return key that
    # the reference run must agree on.
    nc_iqm, nc_run_ids, return_key = _baseline_iqm(
        task_dir_path, split, baseline_algo, return_key=explicit_metric_key,
    )

    # Try to pull reference run's final_return using the SAME key. If absent
    # in reference (e.g. only available as a curve artifact), fall back to
    # the curve-aware loader.
    runs = load_manifest(task_dir_path)
    ref_rec = next((r for r in runs if r.run_id == reference_run_id), None)
    if ref_rec is None:
        raise ValueError(
            f"Reference run_id '{reference_run_id}' not found in "
            f"{task_dir_path}/results/manifest.json"
        )
    reference_metric_record = ref_rec
    reference_eval_run_id = None
    reference_metric_source = "reference_run"
    if _is_train_record(ref_rec) and ref_rec.split != split:
        eval_candidates = [
            r for r in runs
            if not _is_train_record(r)
            and r.algo == ref_rec.algo
            and r.seed == ref_rec.seed
            and r.split == split
            and r.status in ("completed", "converged_warning")
        ]
        linked = [
            r for r in eval_candidates
            if f"eval of {reference_run_id}" in (r.notes or "")
        ]
        if linked or eval_candidates:
            reference_metric_record = (linked or eval_candidates)[0]
            reference_eval_run_id = reference_metric_record.run_id
            reference_metric_source = "primary_eval_record"

    if (
        return_key in reference_metric_record.metrics
        and reference_metric_record.metrics[return_key] is not None
    ):
        reference_final = float(reference_metric_record.metrics[return_key])
    else:
        reference_final = _load_reference_final_return(
            task_dir_path, reference_metric_record.run_id
        )
        print(
            f"[derive_target] WARN: reference run does not expose metric "
            f"{return_key!r} (used by no_control). Falling back to curve-based "
            f"final_return; the resulting target may be biased if curve and "
            f"baseline metric differ in semantics. Verify the value below."
        )

    target = nc_iqm + alpha_used * (reference_final - nc_iqm)

    # ref_rec already loaded above for return_key resolution.

    # Refuse overwrite unless --force
    existing = (cfg.get("convergence_threshold_per_split") or {}).get(split) or {}
    existing_status = existing.get("status")
    existing_run_id = (cfg.get("convergence_provenance") or {}).get("reference_run_id")
    if (
        existing_status == "frozen"
        and existing_run_id is not None
        and existing_run_id != reference_run_id
        and not force
    ):
        raise SystemExit(
            f"task config convergence_threshold_per_split[{split}] is already frozen "
            f"by reference run '{existing_run_id}'. Overwriting with '{reference_run_id}' "
            f"will invalidate all formal records using the previous target. "
            f"Pass --force only if you intend a full rerun."
        )

    # Build the patch
    threshold_block = cfg.setdefault("convergence_threshold_per_split", {})
    threshold_block[split] = {
        "target_return": float(target),
        "alpha": float(alpha_used),
        "status": "frozen",
        "return_metric_key": return_key,
        "metric_direction": metric_direction,
    }
    # Mark non-target splits as optional (only if not already present)
    for s in cfg.get("eval_splits", []):
        if s == split:
            continue
        if s not in threshold_block:
            threshold_block[s] = {
                "target_return": None,
                "alpha": None,
                "status": "optional",
            }

    cfg["convergence_provenance"] = {
        "reference_run_id":          reference_run_id,
        "reference_config_hash":     ref_rec.config_hash,
        "reference_train_split":     ref_rec.split,   # the data split the ref run trained on
        "target_eval_split":         split,            # the eval split this target governs
        "reference_backend":         ref_rec.backend,
        "reference_device":          ref_rec.device,
        "reference_seed":            int(ref_rec.seed),
        "reference_total_timesteps": int(ref_rec.metrics.get("total_timesteps", 0)) or None,
        "reference_eval_run_id":     reference_eval_run_id,
        "reference_metric_source":   reference_metric_source,
        "baseline_algo":             baseline_algo,
        "baseline_iqm":              float(nc_iqm),
        "return_metric_key":         return_key,
    }

    patch = {
        "split":            split,
        "target_return":    float(target),
        "alpha_used":       float(alpha_used),
        "reference_run_id": reference_run_id,
        "baseline_algo":    baseline_algo,
        "baseline_iqm":     float(nc_iqm),
        "baseline_n":       len(nc_run_ids),
        "reference_final":  float(reference_final),
        "reference_metric_run_id": reference_metric_record.run_id,
        "reference_metric_source": reference_metric_source,
    }

    if dry_run:
        print("[derive_target] DRY RUN — not writing provenance.json")
        print(json.dumps(patch, indent=2, ensure_ascii=False))
        return patch

    # Backup + write provenance only (task.yaml is human-edited; never auto-written).
    if provenance_path.exists():
        backup = provenance_path.with_suffix(
            f".bak.{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        )
        shutil.copy2(provenance_path, backup)
        backup_msg = f" (backup at {backup.name})"
    else:
        backup_msg = ""
    provenance = {
        "convergence_threshold_per_split": cfg["convergence_threshold_per_split"],
        "convergence_provenance":          cfg["convergence_provenance"],
    }
    dump_json(provenance, provenance_path)
    print(f"[derive_target] wrote {provenance_path}{backup_msg}")
    print(json.dumps(patch, indent=2, ensure_ascii=False))
    return patch

def derive_target_main(argv: "list[str] | None" = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="dso / tso / ders / gencos / dc_microgrid")
    parser.add_argument(
        "--task-dir",
        default=None,
        help="Optional isolated benchmark task directory (defaults to benchmarks/<task>).",
    )
    parser.add_argument("--reference-run-id", required=True)
    parser.add_argument("--split", default=None, help="defaults to task config primary_split")
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing frozen target (USE WITH CARE)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    derive_target(
        task=args.task,
        reference_run_id=args.reference_run_id,
        split=args.split,
        alpha=args.alpha,
        force=args.force,
        dry_run=args.dry_run,
        task_dir=args.task_dir,
    )

def _load_task_cfg(task: str, task_dir: str | Path | None = None) -> dict[str, Any]:
    return load_task_config(_resolve_task_dir(task, task_dir))

def _default_campaign_after_from_cfg(cfg: dict[str, Any]) -> str | None:
    protocol = cfg.get("benchmark_protocol") or {}
    value = protocol.get("current_campaign_start_iso")
    return str(value) if value else None

def _records_matching(manifest, **filters) -> list:
    """Filter manifest records by attribute equality."""
    out = []
    for r in manifest:
        if all(getattr(r, k, None) == v for k, v in filters.items()):
            out.append(r)
    return out

def _parse_campaign_after(value: str) -> datetime:
    """Parse a campaign-start timestamp for readiness filtering."""
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(
            "--after must be an ISO timestamp, e.g. "
            "2026-04-20T00:00:00+00:00"
        ) from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def _parse_record_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def _path_mtime_at_or_after(path: Path, after_dt: datetime | None) -> bool:
    if after_dt is None:
        return path.exists()
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return mtime >= after_dt

def _any_file_at_or_after(
    paths: list[Path],
    after_dt: datetime | None,
) -> bool:
    if after_dt is None:
        return bool(paths)
    return any(_path_mtime_at_or_after(p, after_dt) for p in paths)

def _load_readiness_manifest(
    task_dir: Path,
    after: str | None,
) -> tuple[list[RunRecord], dict[str, Any]]:
    """Load manifest records, optionally scoped to a new campaign window.

    ``load_manifest()`` deduplicates before returning records.  For campaign
    filtering we must filter the raw append-only manifest first, otherwise an
    older artifact-rich record could win deduplication and hide a newer record
    from the same benchmark cell.
    """
    meta: dict[str, Any] = {
        "record_filter_after": after,
        "manifest_records_total": None,
        "manifest_records_after_filter": None,
        "manifest_records_excluded_by_filter": 0,
    }
    if after is None:
        records = load_manifest(task_dir)
        meta["manifest_records_total"] = len(records)
        meta["manifest_records_after_filter"] = len(records)
        return records, meta

    after_dt = _parse_campaign_after(after)
    meta["record_filter_after_utc"] = after_dt.isoformat()
    manifest_path = task_dir / "results" / "manifest.json"
    if not manifest_path.exists():
        meta["manifest_records_total"] = 0
        meta["manifest_records_after_filter"] = 0
        return [], meta

    raw_records = json.loads(manifest_path.read_text(encoding="utf-8"))
    kept: list[dict[str, Any]] = []
    excluded_missing_timestamp = 0
    excluded_before_after = 0
    for rec in raw_records:
        rec_dt = _parse_record_timestamp(rec.get("timestamp"))
        if rec_dt is None:
            excluded_missing_timestamp += 1
            continue
        if rec_dt < after_dt:
            excluded_before_after += 1
            continue
        kept.append(rec)

    deduped = dedup_keep_artifacts(kept)
    meta.update({
        "manifest_records_total": len(raw_records),
        "manifest_records_after_filter": len(deduped),
        "manifest_records_excluded_by_filter": len(raw_records) - len(kept),
        "manifest_records_excluded_missing_timestamp": excluded_missing_timestamp,
        "manifest_records_excluded_before_after": excluded_before_after,
    })
    return [RunRecord.from_dict(d) for d in deduped], meta

def check_seed0_readiness(
    task: str,
    *,
    after: str | None = None,
    task_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Return a structured status dict for the seed-0 readiness checks."""
    task_dir_path = _resolve_task_dir(task, task_dir)
    cfg = _load_task_cfg(task, task_dir_path)
    if after is None:
        after = _default_campaign_after_from_cfg(cfg)
    primary_split = cfg.get("primary_split")

    report: dict[str, Any] = {
        "task": task,
        "primary_split": primary_split,
        "steps": {},
        "ready_for_formal_runs": False,
        "record_filter_after": after,
    }

    if primary_split is None:
        report["error"] = "task config missing primary_split"
        return report

    try:
        manifest, manifest_meta = _load_readiness_manifest(task_dir_path, after)
    except ValueError as exc:
        report["error"] = str(exc)
        return report
    report.update(manifest_meta)
    after_dt = None
    if after is not None:
        after_dt = _parse_campaign_after(after)
    baseline_set = cfg.get("baseline_set") or cfg.get("baseline_algos") or []
    reference_baseline = (
        cfg.get("derive_target_baseline_algo")
        or (baseline_set[0] if baseline_set else None)
    )

    ref_runs = _records_matching(
        manifest, algo=reference_baseline, split=primary_split, seed=0,
    ) if reference_baseline else []
    other_baselines = [a for a in baseline_set if a != reference_baseline]
    has_other = False
    for h_algo in other_baselines:
        if _records_matching(manifest, algo=h_algo, split=primary_split, seed=0):
            has_other = True
            break
    step1_ok = bool(ref_runs) and (has_other or len(baseline_set) <= 1)
    report["steps"]["1_baseline_seed0"] = {
        "ok": step1_ok,
        "reference_baseline_algo": reference_baseline,
        "reference_baseline_count": len(ref_runs),
        "other_baselines_present": has_other,
        "other_baselines_any_of": other_baselines,
    }

    train_runs = [
        r for r in manifest
        if r.seed == 0
        and (r.artifacts or {}).get("params") is not None
        and r.algo not in tuple(baseline_set)
    ]
    step2_ok = bool(train_runs)
    report["steps"]["2_train_seed0_reference"] = {
        "ok": step2_ok,
        "candidate_run_ids": [r.run_id for r in train_runs],
        "candidate_splits": sorted({r.split for r in train_runs}),
    }

    eval_runs = [
        r for r in manifest
        if r.seed == 0
        and r.split == primary_split
        and (r.artifacts or {}).get("params") is None
        and r.algo not in tuple(baseline_set)
    ]
    step3_ok = bool(eval_runs)
    report["steps"]["3_eval_seed0_primary"] = {
        "ok": step3_ok,
        "eval_run_ids": [r.run_id for r in eval_runs],
    }

    statuses = {r.run_id: r.status for r in train_runs + eval_runs}
    step4_ok = all(s in ("completed", "converged_warning") for s in statuses.values())
    report["steps"]["4_effect_check_pass"] = {
        "ok": step4_ok and bool(statuses),
        "statuses": statuses,
    }

    summary_dir = task_dir_path / "results" / "summary"
    figures_dir = task_dir_path / "results" / "figures"
    summary_files = (
        [p for p in summary_dir.glob("*.json") if p.is_file()]
        if summary_dir.exists() else []
    )
    figure_files = (
        [p for p in figures_dir.iterdir() if p.is_file()]
        if figures_dir.exists() else []
    )
    has_summary = _any_file_at_or_after(summary_files, after_dt)
    has_figures = _any_file_at_or_after(figure_files, after_dt)
    step5_ok = has_summary and has_figures
    report["steps"]["5_summary_and_plots"] = {
        "ok": step5_ok,
        "has_summary": has_summary,
        "has_figures": has_figures,
        "campaign_mtime_filter": after is not None,
        "summary_dir": str(summary_dir),
        "figures_dir": str(figures_dir),
    }

    threshold = (cfg.get("convergence_threshold_per_split") or {}).get(primary_split) or {}
    provenance = cfg.get("convergence_provenance") or {}
    reference_run_id = provenance.get("reference_run_id")
    reference_in_filtered_train = reference_run_id in {r.run_id for r in train_runs}
    provenance_path = task_dir_path / "configs" / "provenance.json"
    provenance_new_enough = _path_mtime_at_or_after(provenance_path, after_dt)
    step6_ok = (
        threshold.get("status") == "frozen"
        and threshold.get("target_return") is not None
        and reference_run_id
        and (after_dt is None or reference_in_filtered_train)
        and provenance_new_enough
    )
    report["steps"]["6_derive_target"] = {
        "ok": bool(step6_ok),
        "target_status": threshold.get("status"),
        "target_return": threshold.get("target_return"),
        "reference_run_id": reference_run_id,
        "reference_in_filtered_train": reference_in_filtered_train,
        "campaign_mtime_filter": after is not None,
        "provenance_path": str(provenance_path),
    }

    report["ready_for_formal_runs"] = all(
        v["ok"] for v in report["steps"].values()
    )
    return report

def print_seed0_readiness_report(report: dict[str, Any]) -> None:
    task = report["task"]
    print(
        f"\n=== Seed-0 readiness checklist: {task} "
        f"(primary_split={report.get('primary_split')}) ==="
    )
    if "error" in report:
        print(f"  ERROR: {report['error']}")
        return
    if report.get("record_filter_after"):
        print(
            "  record filter: "
            f"timestamp >= {report.get('record_filter_after_utc', report['record_filter_after'])} "
            f"({report.get('manifest_records_after_filter')} kept / "
            f"{report.get('manifest_records_total')} raw records)"
        )
    for step, info in report["steps"].items():
        mark = "✓" if info["ok"] else "✗"
        print(f"  [{mark}] {step}")
        for k, v in info.items():
            if k == "ok":
                continue
            print(f"        {k}: {v}")
    overall = (
        "READY: safe to start multi-seed runs"
        if report["ready_for_formal_runs"]
        else "NOT READY"
    )
    print(f"\n  → {overall}\n")

def seed0_readiness_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check on-disk evidence for completing the seed-0 benchmark pipeline "
            "before starting multi-seed runs."
        ),
        epilog="See docs/en/glossary.md (Benchmark workflow glossary).",
    )
    parser.add_argument(
        "--task", required=True,
        help="Task name: dso / tso / ders / gencos / dc_microgrid",
    )
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="Exit with status 2 if any checklist step is incomplete.",
    )
    parser.add_argument(
        "--after",
        default=None,
        help=(
            "Only count manifest records at or after this ISO timestamp. "
            "Use this when starting a new formal campaign so legacy records "
            "cannot satisfy readiness."
        ),
    )
    parser.add_argument(
        "--task-dir",
        default=None,
        help="Optional isolated benchmark task directory (defaults to benchmarks/<task>).",
    )
    args = parser.parse_args(argv)

    report = check_seed0_readiness(args.task, after=args.after, task_dir=args.task_dir)
    print_seed0_readiness_report(report)

    if "error" in report:
        return 2
    if args.enforce and not report["ready_for_formal_runs"]:
        print(
            f"[seed0_readiness] Checklist incomplete for task={args.task}. "
            f"Complete every checklist step before starting multi-seed runs.",
            file=sys.stderr,
        )
        return 2
    return 0

def _task_cfg(task: str) -> dict[str, Any]:
    return load_task_config(_BENCHMARKS_DIR / task)

def _is_train_record(rec) -> bool:
    artifacts = rec.artifacts or {}
    if artifacts.get("params") is None:
        return False
    # PowerZoo bridge single-agent rows are combined train+formal-eval records:
    # they carry a serialized model under ``params`` and also formal
    # ``per_episode`` eval artifacts on the requested eval split. Treat those
    # as eval-kind for variance/convergence checks.
    if rec.split != "train" and artifacts.get("per_episode") is not None:
        return False
    return True

def _record_backend_device(rec) -> tuple[str, str]:
    backend = rec.backend or "jax_rejax"
    device = rec.device or "gpu"
    return backend, device

def check_variance(
    task: str,
    *,
    variance_ratio_threshold: float = 0.3,
    primary_only: bool = True,
    after: str | None = None,
    backend: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Run all 3 checks; return structured report.

    ``variance_ratio_threshold`` defaults to 0.3 (conventional CI
    ≤ 30% of the mean); the previous default of 0.5 was loose enough to
    let "RL beats baseline" claims pass when the seeds disagreed by
    almost a factor of two.

    ``target_metric_direction`` is required in the task config so OOD-collapse
    detection knows which direction means "better" — silent fallback to
    ``higher_is_better`` would invert the check on cost-style metrics
    (TSO total_operating_cost, DERs mean_p_loss_mw).
    """
    task_dir = _BENCHMARKS_DIR / task
    cfg = _task_cfg(task)
    if after is None:
        after = _default_campaign_after_from_cfg(cfg)
    primary_split = cfg.get("primary_split")
    return_key = cfg.get("target_return_metric_key")
    metric_dir = cfg.get("target_metric_direction")
    if metric_dir is None:
        # Default with a loud warning rather than silent miscount.  If
        # this becomes a hard error, downstream task configs MUST
        # declare the field explicitly first.
        metric_dir = "higher_is_better"
    runs = load_manifest_filtered(
        task_dir,
        after=after,
        backend=backend,
        device=device,
    )

    report: dict[str, Any] = {
        "task": task,
        "primary_split": primary_split,
        "return_metric_key": return_key,
        "metric_direction": metric_dir,
        "filters": {
            "after": after,
            "backend": backend,
            "device": device,
        },
        "checks": {},
        "warnings": [],
        "blockers": [],
    }

    if return_key is None:
        report["blockers"].append(
            f"task config missing target_return_metric_key; cannot compute variance"
        )
        return report

    benchmark_protocol = cfg.get("benchmark_protocol") or {}
    submission_min_seeds = int(benchmark_protocol.get("submission_min_seeds", 0) or 0)
    configured_seed_budget = len(cfg.get("seeds") or [])
    report["submission_min_seeds"] = submission_min_seeds or None
    report["configured_seed_budget"] = configured_seed_budget
    if submission_min_seeds and configured_seed_budget < submission_min_seeds:
        report["warnings"].append(
            f"[protocol_gap] task config exposes {configured_seed_budget} default seeds, "
            f"but benchmark_protocol requires >= {submission_min_seeds} for "
            f"submission-grade reporting."
        )

    # ── Check 1: cross-seed variance per (algo, split, backend, device) cell ───
    # Group eval-kind records by (algo, split, backend, device); compute mean/std.
    # Cross-backend (sb3/sbx) records may only carry ``final_return``
    # (the canonical task-specific key requires an eval rollout for
    # DERs / GenCos), so we fall back to ``final_return`` when the
    # task-specific key is absent so the variance check actually covers
    # the cross-backend cells.
    by_cell: dict[tuple, list[float]] = defaultdict(list)
    for r in runs:
        if _is_train_record(r):
            continue
        if r.status not in ("completed", "converged_warning"):
            continue
        v = r.metrics.get(return_key)
        if v is None:
            v = r.metrics.get("final_return")
        if v is None:
            continue
        try:
            v_f = float(v)
            if not math.isfinite(v_f):
                continue
            backend, device = _record_backend_device(r)
            key = (r.algo, r.split, backend, device)
            by_cell[key].append(v_f)
        except (TypeError, ValueError):
            continue

    variance_check: dict[str, Any] = {}
    for (algo, split, backend, device), values in sorted(by_cell.items()):
        if primary_only and split != primary_split:
            continue
        cell_id = f"{algo}|{split}|{backend}|{device}"
        if submission_min_seeds and len(values) < submission_min_seeds:
            report["warnings"].append(
                f"[insufficient_seeds] {cell_id}: only {len(values)} seeds; "
                f"benchmark protocol requires >= {submission_min_seeds} for "
                f"submission-grade reporting."
            )
        if len(values) < 2:
            continue  # need ≥ 2 seeds to compute variance
        arr = np.asarray(values)
        mean_v = float(arr.mean())
        std_v = float(arr.std(ddof=1))
        ratio = std_v / max(abs(mean_v), 1e-9)
        variance_check[cell_id] = {
            "n_seeds": len(values),
            "mean": mean_v,
            "std": std_v,
            "ratio": ratio,
            "values": values,
        }
        if ratio > variance_ratio_threshold:
            report["warnings"].append(
                f"[high_variance] {cell_id}: std/|mean|={ratio:.2f} > "
                f"{variance_ratio_threshold} (mean={mean_v:.4f}, std={std_v:.4f}, "
                f"n={len(values)}); bootstrap CI will likely be uninformative."
            )
    report["checks"]["cross_seed_variance"] = variance_check

    # ── Check 2: OOD collapse ──────────────────────────────────────────
    # For each (algo, backend) that has primary_split + at least one OOD
    # split, compare: (a) does primary_split policy beat baseline_set[0]
    # on primary_split? (b) on OOD splits, is RL worse than baseline_set[0]?
    baseline_set = cfg.get("baseline_set") or cfg.get("baseline_algos") or []
    baseline_algo = (
        cfg.get("derive_target_baseline_algo")
        or (baseline_set[0] if baseline_set else None)
    )
    eval_splits = cfg.get("eval_splits", [])
    ood_splits = [s for s in eval_splits if s != primary_split and s != "train"]
    # Per (split, algo, backend) → mean
    cell_mean = {
        cell: float(np.mean(v)) for cell, v in
        {k: vv["values"] for k, vv in variance_check.items()}.items()
    }
    # Need to also pull baseline cells which the primary_only filter dropped.
    by_cell_all: dict[tuple, list[float]] = defaultdict(list)
    for r in runs:
        if _is_train_record(r):
            continue
        if r.status not in ("completed", "converged_warning"):
            continue
        v = r.metrics.get(return_key)
        if v is None:
            v = r.metrics.get("final_return")
        if v is None:
            continue
        try:
            backend, device = _record_backend_device(r)
            by_cell_all[(r.algo, r.split, backend, device)].append(float(v))
        except (TypeError, ValueError):
            continue

    def _better(a: float, b: float) -> bool:
        return a > b if metric_dir == "higher_is_better" else a < b

    ood_check: dict[str, Any] = {}
    if baseline_algo and ood_splits:
        # Group RL train algos: anything not in baseline_set
        rl_cells: set[tuple[str, str, str]] = {
            (r.algo, *_record_backend_device(r))
            for r in runs
            if not _is_train_record(r) and r.algo not in tuple(baseline_set)
            and r.status in ("completed", "converged_warning")
        }
        for rl_algo, backend, device in sorted(rl_cells):
            bl_primary = by_cell_all.get((baseline_algo, primary_split, backend, device))
            rl_primary = by_cell_all.get((rl_algo, primary_split, backend, device))
            if not (bl_primary and rl_primary):
                # baseline not run on this backend/device; allow cross-backend
                bl_primary = (
                    by_cell_all.get((baseline_algo, primary_split, "jax_rejax", "gpu"))
                    or []
                )
            if not bl_primary or not rl_primary:
                continue
            bl_p = float(np.mean(bl_primary))
            rl_p = float(np.mean(rl_primary))
            primary_wins = _better(rl_p, bl_p)
            for ood in ood_splits:
                bl_ood = (
                    by_cell_all.get((baseline_algo, ood, backend, device))
                    or by_cell_all.get((baseline_algo, ood, "jax_rejax", "gpu"))
                    or []
                )
                rl_ood = by_cell_all.get((rl_algo, ood, backend, device))
                if not bl_ood or not rl_ood:
                    continue
                bl_o = float(np.mean(bl_ood))
                rl_o = float(np.mean(rl_ood))
                ood_wins = _better(rl_o, bl_o)
                cell = f"{rl_algo}|{backend}|{device}|{ood}"
                ood_check[cell] = {
                    "rl_primary": rl_p, "baseline_primary": bl_p,
                    "rl_ood": rl_o, "baseline_ood": bl_o,
                    "primary_wins": primary_wins, "ood_wins": ood_wins,
                }
                if primary_wins and not ood_wins:
                    report["warnings"].append(
                        f"[ood_collapse] {cell}: RL beats baseline on "
                        f"{primary_split} ({rl_p:.3f} vs {bl_p:.3f}) but "
                        f"loses on {ood} ({rl_o:.3f} vs {bl_o:.3f}); "
                        f"distribution-shift brittleness signal."
                    )
    report["checks"]["ood_collapse"] = ood_check

    # ── Check 3: convergence target hit on primary-split eval records ──
    threshold = (cfg.get("convergence_threshold_per_split") or {}).get(primary_split) or {}
    target_return = threshold.get("target_return")
    train_hit: dict[str, Any] = {}
    if target_return is not None:
        target_f = float(target_return)
        for (algo, split, backend, device), values in sorted(by_cell_all.items()):
            if split != primary_split or algo in tuple(baseline_set):
                continue
            cell_id = f"{algo}|{backend}|{device}"
            train_hit.setdefault(cell_id, [])
            for value in values:
                train_hit[cell_id].append(_better(float(value), target_f) or float(value) == target_f)
    else:
        for r in runs:
            if not _is_train_record(r):
                continue
            if r.status not in ("completed", "converged_warning"):
                continue
            backend, device = _record_backend_device(r)
            cell_id = f"{r.algo}|{backend}|{device}"
            train_hit.setdefault(cell_id, []).append(bool((r.convergence or {}).get("hit", False)))
    convergence_check: dict[str, Any] = {}
    for cell_id, hits in sorted(train_hit.items()):
        n = len(hits)
        n_hit = sum(1 for h in hits if h)
        convergence_check[cell_id] = {
            "n_seeds": n,
            "n_hit_target": n_hit,
            "target_return": target_return,
        }
        if n_hit == 0 and n >= 2:
            report["warnings"].append(
                f"[no_convergence] {cell_id}: 0/{n} seeds reached "
                f"convergence target. Either re-run derive_target with a "
                f"better reference, or accept that the algorithm cannot "
                f"hit this target."
            )
    report["checks"]["convergence_hit"] = convergence_check

    return report

def print_variance_report(report: dict[str, Any]) -> None:
    task = report["task"]
    print(f"\n=== Variance check: {task} ===")
    print(f"  primary_split={report['primary_split']}")
    print(f"  metric={report['return_metric_key']} ({report['metric_direction']})")
    if report.get("submission_min_seeds") is not None:
        print(
            f"  submission_min_seeds={report['submission_min_seeds']} "
            f"(configured_seed_budget={report.get('configured_seed_budget')})"
        )
    if report["blockers"]:
        print(f"\n  BLOCKERS ({len(report['blockers'])}):")
        for b in report["blockers"]:
            print(f"    - {b}")
    if report["warnings"]:
        print(f"\n  WARNINGS ({len(report['warnings'])}):")
        for w in report["warnings"]:
            print(f"    - {w}")
    if not report["blockers"] and not report["warnings"]:
        print(f"\n  ALL CHECKS PASSED")

    # Compact stats summary
    var = report["checks"].get("cross_seed_variance", {})
    if var:
        print(f"\n  cross-seed variance ({len(var)} cells):")
        for cell, s in sorted(var.items()):
            tag = "OK" if s["ratio"] <= 0.3 else "HI"
            print(
                f"    [{tag}] {cell}: n={s['n_seeds']} mean={s['mean']:.3f} "
                f"std={s['std']:.3f} ratio={s['ratio']:.2f}"
            )

    ood = report["checks"].get("ood_collapse", {})
    if ood:
        print(f"\n  OOD collapse ({len(ood)} cells):")
        for cell, s in sorted(ood.items()):
            tag = "COL" if s["primary_wins"] and not s["ood_wins"] else "OK"
            print(
                f"    [{tag}] {cell}: primary={s['rl_primary']:.3f} vs "
                f"{s['baseline_primary']:.3f} | ood={s['rl_ood']:.3f} vs "
                f"{s['baseline_ood']:.3f}"
            )

    conv = report["checks"].get("convergence_hit", {})
    if conv:
        print(f"\n  convergence hit ({len(conv)} cells):")
        for cell, s in sorted(conv.items()):
            tag = "OK" if s["n_hit_target"] > 0 else "MISS"
            print(f"    [{tag}] {cell}: {s['n_hit_target']}/{s['n_seeds']} hit target")

def variance_check_main(argv: "list[str] | None" = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=None,
                        help="single task; default: all 5 tasks")
    parser.add_argument("--variance-threshold", type=float, default=0.3)
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero if any warnings or blockers")
    parser.add_argument("--all-splits", action="store_true",
                        help="report variance on all splits, not just primary")
    parser.add_argument("--after", default=None,
                        help="ISO campaign-start timestamp filter")
    parser.add_argument("--backend", default=None,
                        help="Optional backend filter, e.g. jax_rejax")
    parser.add_argument("--device", default=None,
                        help="Optional device filter, e.g. gpu")
    args = parser.parse_args(argv)

    tasks = [args.task] if args.task else list(JAX_TASK_TO_POWERZOO_TASK.keys())
    n_warn = 0
    n_block = 0
    for task in tasks:
        report = check_variance(
            task,
            variance_ratio_threshold=args.variance_threshold,
            primary_only=not args.all_splits,
            after=args.after,
            backend=args.backend,
            device=args.device,
        )
        print_variance_report(report)
        n_warn += len(report["warnings"])
        n_block += len(report["blockers"])

    print(f"\n=== TOTAL: {n_warn} warnings, {n_block} blockers across {len(tasks)} tasks ===")
    if args.strict and (n_warn > 0 or n_block > 0):
        sys.exit(2)

def supersede_records(
    task: str,
    *,
    before: str | None = None,
    filter_kv: dict[str, str] | None = None,
    explicit_run_ids: list[str] | None = None,
    archive_root: Path | None = None,
    dry_run: bool = False,
    task_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Move stale manifest rows + their on-disk artifacts to an archive dir.

    A record is considered stale when:
      - its ``timestamp`` is strictly earlier than ``before`` (if given), AND
      - every ``key=value`` in ``filter_kv`` matches the record (if given), OR
      - its ``run_id`` is in ``explicit_run_ids`` (regardless of the above).

    Surviving (kept) records are written back to ``manifest.json``; the
    pre-supersede manifest snapshot is preserved inside ``archive_root``.

    Layout of ``archive_root``::

        <archive_root>/
            manifest_pre_supersede.json
            _archived_index.md
            runs/<run_id>.json            # moved from results/runs/
            artifacts/<...>               # moved from results/artifacts/

    If ``dry_run=True`` the function returns the plan without touching disk.
    """
    resolved_task_dir = _resolve_task_dir(task, task_dir)
    results_dir = resolved_task_dir / "results"
    manifest_path = results_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest at {manifest_path}")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raw = raw.get("records", []) if isinstance(raw, dict) else []

    before_dt = _parse_campaign_after(before) if before else None
    filter_kv = filter_kv or {}
    explicit_set = set(explicit_run_ids or [])

    def _matches_filter(rec: dict[str, Any]) -> bool:
        for k, v in filter_kv.items():
            if str(rec.get(k)) != str(v):
                return False
        return True

    stale_records: list[dict[str, Any]] = []
    kept_records: list[dict[str, Any]] = []
    for rec in raw:
        run_id = rec.get("run_id")
        is_stale = False
        if run_id in explicit_set:
            is_stale = True
        elif before_dt is not None:
            rec_dt = _parse_record_timestamp(rec.get("timestamp"))
            if rec_dt is not None and rec_dt < before_dt and _matches_filter(rec):
                is_stale = True
        if is_stale:
            stale_records.append(rec)
        else:
            kept_records.append(rec)

    if archive_root is None:
        ts_token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive_root = (
            _REPO_ROOT
            / "local_tmp"
            / "archived_experiment_configs_and_results"
            / f"{task}_{ts_token}"
        )
    archive_root = Path(archive_root)

    plan: list[dict[str, Any]] = []
    for rec in stale_records:
        run_id = rec.get("run_id") or ""
        files: list[Path] = []
        run_json = results_dir / "runs" / f"{run_id}.json"
        if run_json.exists():
            files.append(run_json)
        for art_value in (rec.get("artifacts") or {}).values():
            if not isinstance(art_value, str):
                continue
            ap = (results_dir / art_value).resolve()
            try:
                ap.relative_to(results_dir.resolve())
            except ValueError:
                continue  # refuse to move anything outside results_dir
            if ap.exists() and ap not in files:
                files.append(ap)
        plan.append({"run_id": run_id, "files": files, "record": rec})

    summary = {
        "task": task,
        "manifest_path": str(manifest_path),
        "archive_root": str(archive_root),
        "before": before,
        "filter": filter_kv,
        "explicit_run_ids": sorted(explicit_set),
        "n_total_records": len(raw),
        "n_stale": len(stale_records),
        "n_kept": len(kept_records),
        "plan": [
            {
                "run_id": p["run_id"],
                "timestamp": p["record"].get("timestamp"),
                "backend": p["record"].get("backend"),
                "device": p["record"].get("device"),
                "algo": p["record"].get("algo"),
                "seed": p["record"].get("seed"),
                "split": p["record"].get("split"),
                "n_files": len(p["files"]),
            }
            for p in plan
        ],
    }
    if dry_run:
        summary["status"] = "dry-run"
        return summary

    if not stale_records:
        summary["status"] = "noop"
        return summary

    archive_root.mkdir(parents=True, exist_ok=True)
    (archive_root / "manifest_pre_supersede.json").write_text(
        manifest_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    moved_files: list[str] = []
    failed_moves: list[dict[str, str]] = []
    for entry in plan:
        for src in entry["files"]:
            try:
                rel = src.resolve().relative_to(results_dir.resolve())
            except ValueError:
                failed_moves.append(
                    {"src": str(src), "error": "path escapes results_dir"}
                )
                continue
            dst = archive_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(src), str(dst))
                moved_files.append(str(rel))
            except Exception as exc:  # pragma: no cover - best-effort
                failed_moves.append({"src": str(src), "dst": str(dst), "error": str(exc)})

    manifest_path.write_text(
        json.dumps(kept_records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    index_lines = [
        f"# Archived records from `{task}`",
        "",
        f"Archived at: {datetime.now(timezone.utc).isoformat()}",
        f"Source manifest: `{manifest_path.relative_to(_REPO_ROOT)}`",
        f"Pre-supersede manifest snapshot: `manifest_pre_supersede.json` (in this dir)",
        "",
        "## Selection rule",
        "",
    ]
    if before is not None:
        index_lines.append(f"- ``before``: timestamp < `{before}`")
    if filter_kv:
        index_lines.append(
            "- ``filter``: " + ", ".join(f"`{k}={v}`" for k, v in filter_kv.items())
        )
    if explicit_set:
        index_lines.append("- ``explicit_run_ids``:")
        for rid in sorted(explicit_set):
            index_lines.append(f"  - `{rid}`")
    index_lines += [
        "",
        f"## Run IDs archived ({len(plan)})",
        "",
    ]
    for entry in plan:
        rec = entry["record"]
        index_lines.append(
            f"- `{entry['run_id']}` "
            f"(timestamp=`{rec.get('timestamp')}`, "
            f"backend=`{rec.get('backend')}`, "
            f"device=`{rec.get('device')}`, "
            f"algo=`{rec.get('algo')}`, "
            f"seed=`{rec.get('seed')}`, "
            f"split=`{rec.get('split')}`, "
            f"files_moved={len(entry['files'])})"
        )
    if failed_moves:
        index_lines += ["", "## Files that failed to move", ""]
        for fm in failed_moves:
            index_lines.append(f"- `{fm.get('src')}`: {fm.get('error')}")
    (archive_root / "_archived_index.md").write_text(
        "\n".join(index_lines) + "\n", encoding="utf-8"
    )

    summary.update(
        {
            "status": "ok" if not failed_moves else "partial",
            "n_files_moved": len(moved_files),
            "n_failed_moves": len(failed_moves),
            "files_moved": moved_files,
            "failed_moves": failed_moves,
        }
    )
    return summary


def supersede_main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Archive stale manifest rows + their artifacts to "
            "local_tmp/archived_experiment_configs_and_results/<task>_<utc>/, "
            "leaving only the "
            "current single source of truth in results/manifest.json."
        )
    )
    parser.add_argument("--task", required=True,
                        help="benchmark task (dso/tso/ders/gencos/dc_microgrid)")
    parser.add_argument("--task-dir", default=None,
                        help="override task directory (default: benchmarks/<task>)")
    parser.add_argument("--before", default=None,
                        help="ISO timestamp; archive records with timestamp < this")
    parser.add_argument(
        "--filter", nargs="*", default=None, metavar="KEY=VALUE",
        help="archive only records matching every KEY=VALUE pair "
             "(e.g. algo=ppo backend=jax_rejax device=gpu split=train)",
    )
    parser.add_argument(
        "--run-id", action="append", default=None, metavar="RUN_ID",
        help="archive a specific run_id regardless of --before/--filter; can repeat",
    )
    parser.add_argument("--archive-root", default=None,
                        help="custom archive directory; default uses UTC timestamp")
    parser.add_argument("--dry-run", action="store_true",
                        help="show plan without touching files or manifest")
    args = parser.parse_args(argv)

    filter_kv: dict[str, str] = {}
    for tok in args.filter or []:
        if "=" not in tok:
            print(f"--filter expected KEY=VALUE, got {tok!r}", file=sys.stderr)
            return 2
        k, v = tok.split("=", 1)
        filter_kv[k.strip()] = v.strip()

    if args.before is None and not filter_kv and not args.run_id:
        print(
            "supersede needs at least one of --before / --filter / --run-id",
            file=sys.stderr,
        )
        return 2

    summary = supersede_records(
        args.task,
        before=args.before,
        filter_kv=filter_kv or None,
        explicit_run_ids=args.run_id,
        archive_root=Path(args.archive_root) if args.archive_root else None,
        dry_run=args.dry_run,
        task_dir=args.task_dir,
    )

    print(f"[supersede] task={summary['task']} status={summary.get('status')}")
    print(f"  manifest:        {summary['manifest_path']}")
    print(f"  archive_root:    {summary['archive_root']}")
    print(f"  total records:   {summary['n_total_records']}")
    print(f"  archived:        {summary['n_stale']}")
    print(f"  kept:            {summary['n_kept']}")
    if summary.get("n_files_moved") is not None:
        print(f"  files moved:     {summary['n_files_moved']}")
    if summary.get("n_failed_moves"):
        print(f"  failed moves:    {summary['n_failed_moves']}")
    if summary["plan"]:
        print("  archived run_ids:")
        for entry in summary["plan"]:
            print(
                f"    {entry['run_id']:<60} "
                f"ts={entry.get('timestamp')} "
                f"backend={entry.get('backend')}/{entry.get('device')} "
                f"algo={entry.get('algo')} seed={entry.get('seed')} "
                f"split={entry.get('split')} "
                f"files={entry.get('n_files')}"
            )
    if summary.get("status") == "partial":
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(
            "Usage: python -m benchmarks.common.experiment_ops "
            "<seed0_readiness|derive_target|variance_check|benchmark_preflight|supersede> ...",
            file=sys.stderr,
        )
        return 2

    cmd, sub_argv = argv[0], argv[1:]
    if cmd == "seed0_readiness":
        return seed0_readiness_main(sub_argv)
    if cmd == "derive_target":
        derive_target_main(sub_argv)
        return 0
    if cmd == "variance_check":
        variance_check_main(sub_argv)
        return 0
    if cmd == "benchmark_preflight":
        return benchmark_preflight_main(sub_argv)
    if cmd == "supersede":
        return supersede_main(sub_argv)

    print(f"Unknown experiment op: {cmd}", file=sys.stderr)
    return 2

if __name__ == "__main__":
    raise SystemExit(main())
