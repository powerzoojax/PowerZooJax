"""Validate GenCos Phase-2 backend/device comparison for paper use.

Checks:

* required backend/device cells exist;
* training rows use ``split=train`` and carry training artifacts;
* training configs match the frozen GenCos protocol;
* official eval rows exist for IID / demand_shift / renewable_shock;
* SB3/SBX rows expose honest GenCos env contract metadata;
* canonical learning-curve artifacts are present.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.common.io import has_training_artifact, load_manifest

DEFAULT_TASK_DIR = Path(__file__).resolve().parent
CELL_ORDER = (
    ("jax_rejax", "gpu", "ippo"),
    ("jax_rejax", "cpu", "ippo"),
    ("sb3", "cuda", "ppo"),
    ("sbx", "cuda", "sbx_ppo"),
)
EVAL_SPLITS = ("iid", "demand_shift", "renewable_shock")


def _select_latest_record(
    task_dir: Path,
    *,
    backend: str,
    device: str,
    algo: str,
    split: str,
    seed: int,
    training: bool | None,
):
    records = [
        r
        for r in load_manifest(task_dir)
        if r.backend == backend
        and r.device == device
        and r.algo == algo
        and r.split == split
        and r.seed == seed
    ]
    if training is not None:
        records = [
            r for r in records if has_training_artifact(r.artifacts) is bool(training)
        ]
    if not records:
        return None
    completed = [r for r in records if r.status == "completed"]
    chosen = completed if completed else records
    chosen.sort(key=lambda r: (r.timestamp, r.run_id))
    return chosen[-1]


def _legacy_candidates(task_dir: Path, *, backend: str, device: str, algo: str) -> list[dict[str, Any]]:
    """Return older same-cell records that are present but not Phase-2 eligible."""
    out: list[dict[str, Any]] = []
    for record in load_manifest(task_dir):
        if record.backend != backend or record.device != device or record.algo != algo:
            continue
        artifacts = record.artifacts or {}
        reasons: list[str] = []
        if record.split != "train":
            reasons.append(f"split={record.split!r}, expected 'train'")
        if not has_training_artifact(artifacts):
            reasons.append("missing training artifact")
        if backend in {"sb3", "sbx"} and not artifacts.get("models_manifest"):
            reasons.append("missing models_manifest")
        cfg = _load_config_snapshot(task_dir, record)
        driver_cfg = (cfg or {}).get("powerzoo_driver_config") or {}
        if int(driver_cfg.get("total_timesteps", -1)) != 5_000_000:
            reasons.append(
                f"driver total_timesteps={driver_cfg.get('total_timesteps')!r}, expected 5000000"
            )
        if driver_cfg.get("train_split") not in (None, "train"):
            reasons.append(f"driver train_split={driver_cfg.get('train_split')!r}")
        if reasons:
            out.append(
                {
                    "run_id": record.run_id,
                    "split": record.split,
                    "status": record.status,
                    "timestamp": record.timestamp,
                    "reasons": reasons,
                }
            )
    out.sort(key=lambda row: (row["timestamp"], row["run_id"]))
    return out


def _artifact_path(task_dir: Path, record, key: str) -> Path | None:
    rel = (record.artifacts or {}).get(key)
    if not rel:
        return None
    return task_dir / "results" / rel


def _load_config_snapshot(task_dir: Path, record) -> dict[str, Any] | None:
    path = _artifact_path(task_dir, record, "config")
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _expect(condition: bool, msg: str, failures: list[str]) -> None:
    if not condition:
        failures.append(msg)


def _expect_metric_finite(record, key: str, failures: list[str], prefix: str) -> None:
    value = None if record is None else record.metrics.get(key)
    try:
        ok = value is not None and math.isfinite(float(value))
    except Exception:
        ok = False
    _expect(ok, f"{prefix}: metric {key!r} missing or non-finite", failures)


def _expect_env_contract(
    record,
    *,
    split: str,
    failures: list[str],
    prefix: str,
) -> None:
    env = {} if record is None else (record.env_info or {})
    _expect(
        env.get("data_source") == "gb_real",
        f"{prefix}: env_info.data_source must be 'gb_real'",
        failures,
    )
    _expect(
        env.get("benchmark_split") == split,
        f"{prefix}: env_info.benchmark_split must equal {split!r}",
        failures,
    )
    expected_ood = {
        "train": None,
        "iid": None,
        "demand_shift": "demand_shift",
        "renewable_shock": "renewable_shock",
    }[split]
    actual_ood = env.get("ood_axis")
    _expect(
        actual_ood == expected_ood or (expected_ood is None and actual_ood in (None, "", "None")),
        f"{prefix}: env_info.ood_axis must equal {expected_ood!r}",
        failures,
    )
    _expect(
        bool(env.get("profile_window")),
        f"{prefix}: env_info.profile_window missing",
        failures,
    )


def validate_phase2(task_dir: Path, *, seed: int = 0) -> dict[str, Any]:
    failures: list[str] = []
    cells: list[dict[str, Any]] = []

    for backend, device, algo in CELL_ORDER:
        prefix = f"{backend}/{device}/{algo}"
        train_record = _select_latest_record(
            task_dir,
            backend=backend,
            device=device,
            algo=algo,
            split="train",
            seed=seed,
            training=True,
        )
        cell = {
            "backend": backend,
            "device": device,
            "algo": algo,
            "train_run_id": None if train_record is None else train_record.run_id,
            "eval_run_ids": {},
            "legacy_ineligible_candidates": [],
        }
        cells.append(cell)

        if train_record is None:
            cell["legacy_ineligible_candidates"] = _legacy_candidates(
                task_dir,
                backend=backend,
                device=device,
                algo=algo,
            )
        _expect(train_record is not None, f"{prefix}: missing eligible train record", failures)
        if train_record is None:
            continue
        _expect(
            train_record.status == "completed",
            f"{prefix}: train record not completed",
            failures,
        )
        _expect(
            train_record.split == "train",
            f"{prefix}: train record split must be 'train'",
            failures,
        )

        required_train_artifacts = [
            "config",
            "learning_curve_train_return",
            "learning_curve_eval_return",
            "market/HHI",
            "market/ramp_binding_rate",
            "timesteps",
        ]
        if backend in {"sb3", "sbx"}:
            required_train_artifacts.append("models_manifest")
        for key in required_train_artifacts:
            path = _artifact_path(task_dir, train_record, key)
            _expect(
                path is not None and path.exists(),
                f"{prefix}: missing artifact {key!r}",
                failures,
            )

        cfg = _load_config_snapshot(task_dir, train_record)
        _expect(cfg is not None, f"{prefix}: missing config snapshot", failures)
        if cfg is not None:
            if backend == "jax_rejax":
                train_cfg = cfg.get("train_config") or {}
                _expect(
                    int(train_cfg.get("total_timesteps", -1)) == 5_000_000,
                    f"{prefix}: JAX total_timesteps must be 5,000,000",
                    failures,
                )
                _expect(
                    int(train_cfg.get("n_steps", -1)) == 48,
                    f"{prefix}: JAX n_steps must be 48",
                    failures,
                )
                _expect(
                    int(train_cfg.get("n_epochs", -1)) == 4,
                    f"{prefix}: JAX n_epochs must be 4",
                    failures,
                )
            else:
                train_cfg = cfg.get("train_config_raw") or {}
                _expect(
                    int(train_cfg.get("total_timesteps", -1)) == 5_000_000,
                    f"{prefix}: PPO total_timesteps must be 5,000,000",
                    failures,
                )
                _expect(
                    train_cfg.get("train_split") == "train",
                    f"{prefix}: train_split must be 'train'",
                    failures,
                )
                _expect(
                    int(train_cfg.get("il_self_play_rounds", -1)) == 4,
                    f"{prefix}: il_self_play_rounds must be 4",
                    failures,
                )
                _expect(
                    int(train_cfg.get("per_agent_steps_per_round", -1)) == 250_000,
                    f"{prefix}: per_agent_steps_per_round must be 250000",
                    failures,
                )
                _expect_env_contract(
                    train_record,
                    split="train",
                    failures=failures,
                    prefix=f"{prefix}/train",
                )

        _expect_metric_finite(train_record, "total_profit", failures, f"{prefix}/train")
        _expect_metric_finite(train_record, "mean_lmp", failures, f"{prefix}/train")
        _expect_metric_finite(train_record, "hhi", failures, f"{prefix}/train")
        _expect_metric_finite(
            train_record, "ramp_binding_rate", failures, f"{prefix}/train"
        )
        sced = train_record.metrics.get("sced_convergence_rate")
        _expect(
            sced is not None and float(sced) == 1.0,
            f"{prefix}/train: sced_convergence_rate must equal 1.0",
            failures,
        )

        for split in EVAL_SPLITS:
            eval_record = _select_latest_record(
                task_dir,
                backend=backend,
                device=device,
                algo=algo,
                split=split,
                seed=seed,
                training=False,
            )
            cell["eval_run_ids"][split] = None if eval_record is None else eval_record.run_id
            _expect(
                eval_record is not None,
                f"{prefix}: missing eval record for split={split}",
                failures,
            )
            if eval_record is None:
                continue
            _expect(
                eval_record.status == "completed",
                f"{prefix}: eval record {split} not completed",
                failures,
            )
            _expect_metric_finite(
                eval_record, "total_profit", failures, f"{prefix}/{split}"
            )
            _expect_metric_finite(
                eval_record, "mean_lmp", failures, f"{prefix}/{split}"
            )
            _expect_metric_finite(eval_record, "hhi", failures, f"{prefix}/{split}")
            _expect_metric_finite(
                eval_record, "ramp_binding_rate", failures, f"{prefix}/{split}"
            )
            sced = eval_record.metrics.get("sced_convergence_rate")
            _expect(
                sced is not None and float(sced) == 1.0,
                f"{prefix}/{split}: sced_convergence_rate must equal 1.0",
                failures,
            )
            if backend in {"sb3", "sbx"}:
                _expect_env_contract(
                    eval_record,
                    split=split,
                    failures=failures,
                    prefix=f"{prefix}/{split}",
                )

    return {
        "task": "gencos",
        "seed": seed,
        "ok": not failures,
        "failures": failures,
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", type=Path, default=DEFAULT_TASK_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()

    report = validate_phase2(Path(args.task_dir), seed=int(args.seed))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.enforce and not report["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
