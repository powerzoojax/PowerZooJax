from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from benchmarks.tso.plots import _canonical_train_runs, _curve_x


def test_canonical_train_runs_prefers_latest_params_backed_record(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)
    manifest_path = results_dir / "manifest.json"
    manifest = [
        {
            "algo": "ppo",
            "seed": 0,
            "split": "train",
            "run_id": "tso_ppo_train_s0_old",
            "timestamp": "2026-04-21T10:00:00+00:00",
            "artifacts": {"params": "artifacts/tso_ppo_train_s0_old_params.pkl"},
        },
        {
            "algo": "ppo",
            "seed": 0,
            "split": "train",
            "run_id": "tso_ppo_train_s0_eval_on_train",
            "timestamp": "2026-04-21T11:00:00+00:00",
            "artifacts": {"per_episode": "artifacts/tso_ppo_train_s0_eval_on_train.json"},
        },
        {
            "algo": "ppo",
            "seed": 0,
            "split": "train",
            "run_id": "tso_ppo_train_s0_new",
            "timestamp": "2026-04-21T12:00:00+00:00",
            "artifacts": {"params": "artifacts/tso_ppo_train_s0_new_params.pkl"},
        },
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    canonical = _canonical_train_runs(tmp_path)

    assert canonical[("ppo", 0)]["run_id"] == "tso_ppo_train_s0_new"


def test_canonical_train_runs_honors_campaign_after_filter(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)
    manifest_path = results_dir / "manifest.json"
    manifest = [
        {
            "algo": "saute_ppo",
            "seed": 0,
            "split": "train",
            "run_id": "old_ablation",
            "backend": "jax_rejax",
            "device": "gpu",
            "timestamp": "2026-04-23T10:00:00+00:00",
            "artifacts": {"params": "artifacts/old_ablation_params.pkl"},
        },
        {
            "algo": "ppo",
            "seed": 0,
            "split": "train",
            "run_id": "current_ppo",
            "backend": "jax_rejax",
            "device": "gpu",
            "timestamp": "2026-04-24T03:00:00+00:00",
            "artifacts": {"params": "artifacts/current_ppo_params.pkl"},
        },
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    canonical = _canonical_train_runs(
        tmp_path,
        after="2026-04-24T02:05:20+00:00",
    )

    assert ("ppo", 0) in canonical
    assert ("saute_ppo", 0) not in canonical


def test_curve_x_falls_back_to_full_training_span_when_lengths_mismatch(tmp_path: Path):
    artifacts_dir = tmp_path / "results" / "artifacts"
    artifacts_dir.mkdir(parents=True)
    np.save(artifacts_dir / "demo_timesteps.npy", np.linspace(0, 5_000_000, 20))
    np.save(artifacts_dir / "demo_learning_curve_eval_walltimes.npy", np.linspace(0, 50, 20))

    run = {
        "run_id": "demo",
        "metrics": {"total_timesteps": 10_000_000},
        "walltime_s": 200.0,
    }

    x_steps = _curve_x(run, artifacts_dir, 406, axis_kind="timesteps")
    x_walltime = _curve_x(run, artifacts_dir, 406, axis_kind="walltime")

    assert x_steps is not None
    assert x_walltime is not None
    assert x_steps.shape == (406,)
    assert x_walltime.shape == (406,)
    assert x_steps[-1] == 10_000_000
    assert x_walltime[-1] == 200.0
