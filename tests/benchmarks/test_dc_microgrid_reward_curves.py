"""Tests for DC Microgrid learning-curve fair-comparison plotting.

Covers:
  1. dedup_keep_artifacts keeps (backend, device) variants as separate cells.
  2. _train_runs_by_backend_device yields 4 group keys for 4-backend-device
     PPO records; plot_reward_curves completes without raising.
  3. Single-record manifest also completes without raising.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.common.io import dedup_keep_artifacts
from benchmarks.dc_microgrid.plots import (
    _train_runs_by_backend_device,
    plot_reward_curves,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_record(
    backend: str,
    device: str,
    algo: str = "ppo",
    seed: int = 0,
    run_id: str | None = None,
    timestamp: str = "2026-04-22T00:00:00+00:00",
    params_key: str = "params",
) -> dict:
    rid = run_id or f"dc_microgrid_{algo}_train_s{seed}_{backend}_{device}_fake"
    return {
        "task": "dc_microgrid",
        "variant": "default",
        "algo": algo,
        "seed": seed,
        "split": "train",
        "backend": backend,
        "device": device,
        "run_id": rid,
        "timestamp": timestamp,
        "status": "completed",
        "metrics": {},
        "artifacts": {params_key: f"artifacts/{rid}_{params_key}"},
    }


def _write_manifest(task_dir: Path, records: list[dict]) -> None:
    results = task_dir / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "manifest.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )


def _write_min_npy(artifacts_dir: Path, run_id: str, n: int = 5) -> None:
    """Write minimal eval_returns and timesteps npy files for a run."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    np.save(artifacts_dir / f"{run_id}_eval_returns.npy", np.linspace(-1.0, 0.0, n))
    np.save(artifacts_dir / f"{run_id}_timesteps.npy", np.linspace(0, 1_000_000, n))


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: dedup_keep_artifacts keeps device variants as distinct cells
# ─────────────────────────────────────────────────────────────────────────────

def test_dedup_keeps_device_variants():
    """Two jax_rejax records differing only by device must NOT be collapsed."""
    rec_gpu = _make_record("jax_rejax", "gpu", seed=0, run_id="run_jax_gpu")
    rec_cpu = _make_record("jax_rejax", "cpu", seed=0, run_id="run_jax_cpu")

    result = dedup_keep_artifacts([rec_gpu, rec_cpu])

    assert len(result) == 2, (
        "Expected 2 distinct cells (jax_rejax+gpu vs jax_rejax+cpu), "
        f"got {len(result)}: {[r['run_id'] for r in result]}"
    )
    run_ids = {r["run_id"] for r in result}
    assert "run_jax_gpu" in run_ids
    assert "run_jax_cpu" in run_ids


def test_dedup_collapses_same_backend_device_keeps_latest():
    """Two records for the same (backend, device, algo, seed) → only newest survives."""
    old = _make_record("jax_rejax", "gpu", seed=0, run_id="run_old",
                       timestamp="2026-04-01T00:00:00+00:00")
    new = _make_record("jax_rejax", "gpu", seed=0, run_id="run_new",
                       timestamp="2026-04-22T00:00:00+00:00")

    result = dedup_keep_artifacts([old, new])

    assert len(result) == 1
    assert result[0]["run_id"] == "run_new"


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: four distinct PPO backend×device variants → four plot curves
# ─────────────────────────────────────────────────────────────────────────────

_FOUR_RECORDS = [
    _make_record("jax_rejax", "gpu",  seed=0, run_id="dc_ppo_jax_gpu"),
    _make_record("jax_rejax", "cpu",  seed=0, run_id="dc_ppo_jax_cpu"),
    _make_record("sb3",       "cuda", seed=0, run_id="dc_ppo_sb3_cuda"),
    _make_record("sb3",       "cpu",  seed=0, run_id="dc_ppo_sb3_cpu"),
]


def test_train_runs_by_backend_device_four_groups(tmp_path: Path):
    """Four distinct (backend, device, ppo) records → four group keys."""
    _write_manifest(tmp_path, _FOUR_RECORDS)

    groups = _train_runs_by_backend_device(tmp_path)

    assert len(groups) == 4, (
        f"Expected 4 groups, got {len(groups)}: {list(groups.keys())}"
    )
    expected_keys = {
        ("jax_rejax", "gpu",  "ppo"),
        ("jax_rejax", "cpu",  "ppo"),
        ("sb3",       "cuda", "ppo"),
        ("sb3",       "cpu",  "ppo"),
    }
    assert set(groups.keys()) == expected_keys


def test_reward_curves_four_ppo_lines(tmp_path: Path):
    """Four valid train records → plot completes and returns non-empty Path."""
    _write_manifest(tmp_path, _FOUR_RECORDS)
    artifacts_dir = tmp_path / "results" / "artifacts"
    for rec in _FOUR_RECORDS:
        _write_min_npy(artifacts_dir, rec["run_id"])

    out = plot_reward_curves(tmp_path)

    assert out != Path(), "plot_reward_curves returned empty Path (no curves drawn)"


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: single-run manifest does not crash
# ─────────────────────────────────────────────────────────────────────────────

def test_reward_curves_single_run_no_crash(tmp_path: Path):
    """Single jax_rejax+gpu PPO record → plot completes without raising."""
    rec = _make_record("jax_rejax", "gpu", seed=0, run_id="dc_ppo_single")
    _write_manifest(tmp_path, [rec])
    artifacts_dir = tmp_path / "results" / "artifacts"
    _write_min_npy(artifacts_dir, rec["run_id"])

    out = plot_reward_curves(tmp_path)

    assert out != Path()


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: empty / missing manifest → returns empty Path gracefully
# ─────────────────────────────────────────────────────────────────────────────

def test_reward_curves_empty_manifest_returns_empty_path(tmp_path: Path):
    """No manifest at all → returns Path() without raising."""
    out = plot_reward_curves(tmp_path)
    assert out == Path()


def test_train_runs_missing_manifest_returns_empty(tmp_path: Path):
    groups = _train_runs_by_backend_device(tmp_path)
    assert groups == {}


def test_params_orbax_records_included(tmp_path: Path):
    """SAC-style params_orbax artifact must be treated as a genuine training record."""
    rec = _make_record("jax_rejax", "gpu", algo="sac", seed=0,
                       run_id="dc_sac_orbax", params_key="params_orbax")
    _write_manifest(tmp_path, [rec])
    groups = _train_runs_by_backend_device(tmp_path)
    assert ("jax_rejax", "gpu", "sac") in groups


def test_params_absent_records_excluded(tmp_path: Path):
    """Records with neither params nor params_orbax are eval-only, must be excluded."""
    rec = {
        "task": "dc_microgrid", "algo": "ppo", "seed": 0, "split": "train",
        "backend": "jax_rejax", "device": "gpu",
        "run_id": "dc_ppo_no_params", "timestamp": "2026-04-22T00:00:00+00:00",
        "artifacts": {"eval_returns": "artifacts/dc_ppo_no_params_eval_returns.npy"},
    }
    _write_manifest(tmp_path, [rec])
    groups = _train_runs_by_backend_device(tmp_path)
    assert groups == {}
