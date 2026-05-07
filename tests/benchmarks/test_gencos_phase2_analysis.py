from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _write_run(task_dir: Path, record: dict) -> None:
    results = task_dir / "results"
    runs = results / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    path = runs / f"{record['run_id']}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")


def test_phase2_summary_keeps_train_record_when_no_train_eval_row(tmp_path: Path):
    from benchmarks.gencos.phase2_analysis import _build_summary

    results = tmp_path / "results"
    arts = results / "artifacts"
    arts.mkdir(parents=True, exist_ok=True)

    train_run_id = "gencos_ippo_train_s0_demo"
    eval_run_id = "gencos_ippo_iid_s0_demo"

    np.save(arts / f"{train_run_id}_learning_curve_eval_return.npy", np.array([1.0, 2.0]))
    np.save(arts / f"{train_run_id}_learning_curve_train_return.npy", np.array([0.5, 1.5]))
    np.save(arts / f"{train_run_id}_timesteps.npy", np.array([0.0, 5_000_000.0]))
    np.save(arts / f"{train_run_id}_market_HHI.npy", np.array([0.6, 0.55]))
    np.save(arts / f"{train_run_id}_market_ramp_binding_rate.npy", np.array([0.2, 0.25]))
    (arts / f"{train_run_id}_config.json").write_text(
        json.dumps({"train_config": {"total_timesteps": 5_000_000}}),
        encoding="utf-8",
    )

    train_record = {
        "task": "gencos",
        "variant": "gencos",
        "algo": "ippo",
        "seed": 0,
        "run_id": train_run_id,
        "status": "completed",
        "split": "train",
        "backend": "jax_rejax",
        "device": "gpu",
        "framework_version": "test",
        "metrics": {
            "total_profit": 123.0,
            "mean_lmp": 34.0,
            "hhi": 0.55,
            "ramp_binding_rate": 0.25,
            "sced_convergence_rate": 1.0,
        },
        "walltime_s": 10.0,
        "throughput_sps": 100.0,
        "timestamp": "2026-04-24T00:00:00+00:00",
        "notes": "",
        "env_info": {},
        "artifacts": {
            "config": f"artifacts/{train_run_id}_config.json",
            "params": f"artifacts/{train_run_id}_params.pkl",
            "learning_curve_eval_return": f"artifacts/{train_run_id}_learning_curve_eval_return.npy",
            "learning_curve_train_return": f"artifacts/{train_run_id}_learning_curve_train_return.npy",
            "timesteps": f"artifacts/{train_run_id}_timesteps.npy",
            "market/HHI": f"artifacts/{train_run_id}_market_HHI.npy",
            "market/ramp_binding_rate": f"artifacts/{train_run_id}_market_ramp_binding_rate.npy",
        },
    }
    iid_eval_record = {
        "task": "gencos",
        "variant": "gencos",
        "algo": "ippo",
        "seed": 0,
        "run_id": eval_run_id,
        "status": "completed",
        "split": "iid",
        "backend": "jax_rejax",
        "device": "gpu",
        "framework_version": "test",
        "metrics": {
            "total_profit": 111.0,
            "mean_lmp": 33.0,
            "hhi": 0.5,
            "ramp_binding_rate": 0.2,
            "sced_convergence_rate": 1.0,
        },
        "walltime_s": 0.0,
        "throughput_sps": None,
        "timestamp": "2026-04-24T00:01:00+00:00",
        "notes": "",
        "env_info": {},
        "artifacts": {},
    }

    _write_run(tmp_path, train_record)
    _write_run(tmp_path, iid_eval_record)
    (results / "manifest.json").write_text(
        json.dumps([train_record, iid_eval_record], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = _build_summary(tmp_path)
    row = next(r for r in summary["rows"] if r["backend"] == "jax_rejax" and r["device"] == "gpu")

    assert row["train_run_id"] == train_run_id
    assert row["train_total_profit"] == 123.0
    assert row["train_curve_points"] == 2
    assert row["train_curve_final_return"] == 2.0
    assert row["train_eval_run_id"] is None
    assert row["iid_run_id"] == eval_run_id
    assert row["iid_total_profit"] == 111.0
