"""Regression tests for DSO benchmark reporting and plotting semantics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.dso.plots import (
    _load_ppo_reference_violations,
    _load_reward_curve,
)
from benchmarks.dso.summarize import summarize_dso


def _write_manifest(task_dir: Path, records: list[dict]) -> None:
    results_dir = task_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "manifest.json").write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )


def _record(
    *,
    run_id: str,
    algo: str,
    split: str,
    seed: int,
    timestamp: str,
    backend: str = "jax_rejax",
    device: str = "gpu",
    metrics: dict | None = None,
    artifacts: dict | None = None,
) -> dict:
    return {
        "task": "dso",
        "variant": "dso_nflex",
        "algo": algo,
        "seed": seed,
        "run_id": run_id,
        "timestamp": timestamp,
        "split": split,
        "backend": backend,
        "device": device,
        "metrics": metrics or {},
        "artifacts": artifacts or {},
        "status": "completed",
    }


def test_summarize_dso_filters_campaign_and_uses_voltage_only_metric(tmp_path: Path):
    task_dir = tmp_path / "dso"
    _write_manifest(
        task_dir,
        [
            _record(
                run_id="old_ppo_iid_s0",
                algo="ppo",
                split="iid",
                seed=0,
                timestamp="2026-04-21T00:00:00+00:00",
                metrics={
                    "total_reward": -9.0,
                    "total_loss_mwh": 9.0,
                    "total_voltage_violations": 999.0,
                    "total_violations": 999.0,
                },
            ),
            _record(
                run_id="no_control_iid_s0",
                algo="no_control",
                split="iid",
                seed=0,
                timestamp="2026-04-22T03:00:00+00:00",
                metrics={
                    "total_reward": -2.0,
                    "total_loss_mwh": 8.0,
                    "total_voltage_violations": 144.0,
                    "total_violations": 192.0,
                },
            ),
            _record(
                run_id="droop_iid_s0",
                algo="droop",
                split="iid",
                seed=0,
                timestamp="2026-04-22T03:00:01+00:00",
                metrics={
                    "total_reward": -1.0,
                    "total_loss_mwh": 4.0,
                    "total_voltage_violations": 48.0,
                    "total_violations": 96.0,
                },
            ),
            _record(
                run_id="ppo_iid_s0",
                algo="ppo",
                split="iid",
                seed=0,
                timestamp="2026-04-22T03:00:02+00:00",
                metrics={
                    "total_reward": -0.8,
                    "total_loss_mwh": 3.0,
                    "total_voltage_violations": 96.0,
                    "total_violations": 192.0,
                },
            ),
            _record(
                run_id="ppo_iid_sb3_s1",
                algo="ppo",
                split="iid",
                seed=1,
                timestamp="2026-04-22T03:00:03+00:00",
                backend="sb3",
                device="cuda",
                metrics={
                    "total_reward": -99.0,
                    "total_loss_mwh": 99.0,
                    "total_voltage_violations": 480.0,
                    "total_violations": 480.0,
                },
            ),
        ],
    )

    summary = summarize_dso(
        task_dir,
        after="2026-04-22T02:59:00+00:00",
        backend="jax_rejax",
        device="gpu",
    )

    ppo_iid = next(
        row for row in summary["rows"]
        if row["algo"] == "ppo" and row["split"] == "iid"
    )
    assert ppo_iid["n_seeds"] == 1
    assert ppo_iid["voltage_violation_count_per_step_mean"] == pytest.approx(2.0)
    assert ppo_iid["voltage_violation_count_per_step_mean"] != pytest.approx(4.0)


def test_load_reward_curve_keeps_episode_units_and_scales_train_fallback(tmp_path: Path):
    artifacts_dir = tmp_path

    eval_run = "demo_eval"
    train_run = "demo_train"

    import numpy as np

    np.save(artifacts_dir / f"{eval_run}_learning_curve_eval_return.npy", np.array([-1.0, -0.5]))
    np.save(artifacts_dir / f"{train_run}_mean_reward.npy", np.array([-0.1, -0.2]))

    _, eval_curve, eval_source = _load_reward_curve(eval_run, artifacts_dir, max_steps=48)
    _, train_curve, train_source = _load_reward_curve(train_run, artifacts_dir, max_steps=48)

    assert eval_source == "eval"
    assert eval_curve.tolist() == [-1.0, -0.5]
    assert train_source == "train (fallback)"
    assert train_curve.tolist() == pytest.approx([-4.8, -9.6])


def test_load_ppo_reference_violations_uses_voltage_specific_metric(tmp_path: Path):
    task_dir = tmp_path / "dso"
    results_dir = task_dir / "results"
    artifacts_dir = results_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    per_episode_rel = "artifacts/ppo_train_eval_per_episode.json"
    (results_dir / per_episode_rel).write_text(
        json.dumps(
            [
                {
                    "total_voltage_violations": 96.0,
                    "total_violations": 192.0,
                }
            ]
        ),
        encoding="utf-8",
    )
    _write_manifest(
        task_dir,
        [
            _record(
                run_id="ppo_train_eval",
                algo="ppo",
                split="train",
                seed=0,
                timestamp="2026-04-22T03:00:00+00:00",
                artifacts={"per_episode": per_episode_rel},
            )
        ],
    )

    per_step = _load_ppo_reference_violations(task_dir, 48)
    assert per_step["ppo"] == pytest.approx(2.0)
