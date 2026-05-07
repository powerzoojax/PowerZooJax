"""``variance_check`` must include cross-backend (sb3 / sbx) records.

Pre-fix: cross-backend records only carried ``final_return`` while
``variance_check`` keyed on the task-specific ``target_return_metric_key``
from the task config (e.g. ``total_reward`` for DSO).  Cross-backend records
were silently skipped, leaving the user with the false impression that
multi-seed variance had been validated for both backends.

This test builds a fake manifest with one jax_rejax record (using the
task-specific key) plus one sb3 record (only ``final_return``) and
asserts both show up in the variance report.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_record(task_dir: Path, record: dict) -> None:
    """Append ``record`` to the manifest at ``task_dir/results/manifest.json``."""
    mp = task_dir / "results" / "manifest.json"
    mp.parent.mkdir(parents=True, exist_ok=True)
    if mp.exists():
        existing = json.loads(mp.read_text(encoding="utf-8"))
    else:
        existing = []
    existing.append(record)
    mp.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def _write_task_cfg(
    task_dir: Path,
    return_key: str,
    primary_split: str,
    *,
    submission_min_seeds: int | None = None,
) -> None:
    cfg = {
        "task": "fake",
        "primary_split": primary_split,
        "target_return_metric_key": return_key,
        "target_metric_direction": "lower_is_better",
        "baseline_set": ["no_control"],
        "eval_splits": [primary_split],
        "seeds": [0, 1],
        "num_envs": 1,
        "eval_episodes": 1,
        "convergence_threshold_per_split": {primary_split: {}},
    }
    if submission_min_seeds is not None:
        cfg["benchmark_protocol"] = {"submission_min_seeds": submission_min_seeds}
    cfg_path = task_dir / "configs" / "task.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def _make_record(
    *,
    algo: str,
    split: str,
    seed: int,
    backend: str,
    device: str = "cpu",
    metrics: dict,
) -> dict:
    return {
        "task": "fake",
        "variant": "v",
        "algo": algo,
        "seed": seed,
        "run_id": f"{algo}_{split}_{backend}_s{seed}",
        "config_hash": "abc",
        "status": "completed",
        "split": split,
        "backend": backend,
        "device": device,
        "framework_version": "test",
        "metrics": metrics,
        "convergence": {},
        "walltime_s": 1.0,
        "throughput_sps": 1.0,
        "timestamp": "2026-04-19T00:00:00+00:00",
        "notes": "",
        "env_info": {},
        "artifacts": {},
    }


def test_variance_check_includes_cross_backend_records(tmp_path: Path, monkeypatch):
    # Patch _BENCHMARKS_DIR to point at our temp tree
    from benchmarks.common import experiment_ops as vc

    fake_root = tmp_path / "benchmarks"
    task_dir = fake_root / "fake"
    monkeypatch.setattr(vc, "_BENCHMARKS_DIR", fake_root)

    # task config declares a task-specific metric key + primary_split
    _write_task_cfg(task_dir, return_key="total_reward", primary_split="iid")

    # 2 jax_rejax records carrying the task-specific key
    for s in (0, 1):
        _write_record(
            task_dir,
            _make_record(
                algo="ppo",
                split="iid",
                seed=s,
                backend="jax_rejax",
                device="gpu",
                metrics={"total_reward": -0.5 + 0.05 * s},
            ),
        )

    # 2 sb3 records carrying ONLY final_return
    for s in (0, 1):
        _write_record(
            task_dir,
            _make_record(
                algo="ppo",
                split="iid",
                seed=s,
                backend="sb3",
                device="cuda",
                metrics={"final_return": -0.6 + 0.05 * s},
            ),
        )

    report = vc.check_variance("fake", primary_only=True)

    cells = report["checks"]["cross_seed_variance"]
    cell_jax = "ppo|iid|jax_rejax|gpu"
    cell_sb3 = "ppo|iid|sb3|cuda"
    assert cell_jax in cells, (
        f"jax_rejax cell missing from variance report: {sorted(cells)}"
    )
    assert cell_sb3 in cells, (
        f"sb3 (cross-backend) cell missing from variance report despite "
        f"final_return being present: {sorted(cells)}.  "
        f"variance_check is silently dropping cross-backend records — fix the "
        f"fallback to final_return."
    )
    assert cells[cell_sb3]["n_seeds"] == 2


def test_variance_check_warns_when_below_submission_min_seeds(
    tmp_path: Path,
    monkeypatch,
):
    from benchmarks.common import experiment_ops as vc

    fake_root = tmp_path / "benchmarks"
    task_dir = fake_root / "fake"
    monkeypatch.setattr(vc, "_BENCHMARKS_DIR", fake_root)

    _write_task_cfg(
        task_dir,
        return_key="total_reward",
        primary_split="iid",
        submission_min_seeds=5,
    )

    for s in (0, 1, 2):
        _write_record(
            task_dir,
            _make_record(
                algo="ppo",
                split="iid",
                seed=s,
                backend="jax_rejax",
                device="gpu",
                metrics={"total_reward": -1.0 + 0.1 * s},
            ),
        )

    report = vc.check_variance("fake", primary_only=True)

    warnings = "\n".join(report["warnings"])
    assert "[insufficient_seeds] ppo|iid|jax_rejax|gpu" in warnings
    assert "requires >= 5" in warnings
