#!/usr/bin/env python
"""Audit the current TSO Phase-2 cross-backend training matrix.

This checks the four canonical training cells used for the paper's walltime
learning-curve figure:

- jax_rejax + gpu + ppo
- jax_rejax + cpu + ppo
- sb3 + cuda + ppo
- sbx + cuda + sbx_ppo

The audit is intentionally narrow: same train split, same 20M budget, same
48-step rollout horizon, same 256 env parallelism, and the same train-monitor
curve payload required by ``benchmarks/tso/plots.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.common.configs import load_task_config

TASK_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = TASK_DIR / "results"
RUNS_DIR = RESULTS_DIR / "runs"
ARTIFACTS_DIR = RESULTS_DIR / "artifacts"
SUMMARY_DIR = RESULTS_DIR / "summary"
OUT_PATH = SUMMARY_DIR / "phase2_backend_audit.json"

EXPECTED_TOTAL_TIMESTEPS = 20_000_000
EXPECTED_NUM_ENVS = 256
EXPECTED_N_STEPS = 48
EXPECTED_EVAL_FREQ = 100_000
EXPECTED_EVAL_EPISODES = 8
EXPECTED_TRAIN_WINDOW = "2025-04-01..2025-12-31"

REQUIRED_CURVES = (
    "timesteps",
    "learning_curve_eval_return",
    "learning_curve_eval_walltimes",
    "eval_total_operating_cost",
    "eval_reserve_shortfall_rate",
    "eval_thermal_violation_rate",
)
REQUIRED_ARTIFACTS = (
    "config",
    "params",
)


@dataclass(frozen=True)
class Phase2Cell:
    name: str
    backend: str
    device: str
    algo: str


EXPECTED_CELLS: tuple[Phase2Cell, ...] = (
    Phase2Cell("jax_gpu", "jax_rejax", "gpu", "ppo"),
    Phase2Cell("jax_cpu", "jax_rejax", "cpu", "ppo"),
    Phase2Cell("sb3_cuda", "sb3", "cuda", "ppo"),
    Phase2Cell("sbx_cuda", "sbx", "cuda", "sbx_ppo"),
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_runs() -> list[dict[str, Any]]:
    return [_load_json(path) for path in sorted(RUNS_DIR.glob("*.json"))]


def _parse_iso_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _filter_after(rows: list[dict[str, Any]], after: str | None) -> list[dict[str, Any]]:
    if not after:
        return rows
    threshold = _parse_iso_datetime(after)
    filtered = []
    for row in rows:
        timestamp = row.get("timestamp")
        if not timestamp:
            continue
        try:
            if _parse_iso_datetime(str(timestamp)) >= threshold:
                filtered.append(row)
        except ValueError:
            continue
    return filtered


def _default_campaign_after() -> str | None:
    try:
        task_cfg = load_task_config(TASK_DIR)
    except Exception:
        return None
    protocol = task_cfg.get("benchmark_protocol") or {}
    value = protocol.get("current_campaign_start_iso")
    return str(value) if value else None


def _choose_cell_run(cell: Phase2Cell, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    matches = [
        row
        for row in rows
        if row.get("task") == "tso"
        and row.get("split") == "train"
        and row.get("status") == "completed"
        and (
            "params" in (row.get("artifacts") or {})
            or "params_orbax" in (row.get("artifacts") or {})
        )
        and (row.get("backend") or "jax_rejax") == cell.backend
        and (row.get("device") or "gpu") == cell.device
        and (row.get("algo") or "").lower() == cell.algo
    ]
    if not matches:
        return None
    return max(matches, key=lambda row: str(row.get("timestamp") or ""))


def _artifact_path(run: dict[str, Any], key: str) -> Path | None:
    rel = (run.get("artifacts") or {}).get(key)
    if not rel:
        return None
    return (RESULTS_DIR / rel).resolve()


def _load_curve(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    try:
        arr = np.asarray(np.load(path))
    except Exception:
        return None
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.astype(float)


def _config_views(run: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    cfg_path = _artifact_path(run, "config")
    if cfg_path is None or not cfg_path.exists():
        return {}, {}, {}
    raw = _load_json(cfg_path)
    task_cfg = raw.get("task_config") or {}
    train_cfg = raw.get("train_config_raw") or {}
    driver_cfg = raw.get("powerzoo_driver_config") or {}
    return task_cfg, train_cfg, driver_cfg


def _curve_checks(run: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"required_curves_present": {}, "curve_lengths": {}}
    arrays: dict[str, np.ndarray] = {}
    for key in REQUIRED_CURVES:
        path = _artifact_path(run, key)
        arr = _load_curve(path)
        out["required_curves_present"][key] = bool(path is not None and path.exists() and arr is not None)
        if arr is not None:
            arrays[key] = arr
            out["curve_lengths"][key] = int(arr.size)

    timesteps = arrays.get("timesteps")
    walltimes = arrays.get("learning_curve_eval_walltimes")
    out["all_required_curves_present"] = all(out["required_curves_present"].values())
    out["timesteps_monotone_non_decreasing"] = bool(
        timesteps is not None and timesteps.size > 0 and np.all(np.diff(timesteps) >= 0.0)
    )
    out["walltimes_monotone_non_decreasing"] = bool(
        walltimes is not None and walltimes.size > 0 and np.all(np.diff(walltimes) >= 0.0)
    )
    lengths = [int(arr.size) for arr in arrays.values()]
    out["curve_lengths_match"] = bool(lengths and all(n == lengths[0] for n in lengths))
    out["n_monitor_points"] = int(lengths[0]) if out["curve_lengths_match"] and lengths else 0
    if timesteps is not None and timesteps.size:
        out["first_timestep"] = float(timesteps[0])
        out["last_timestep"] = float(timesteps[-1])
    if walltimes is not None and walltimes.size:
        out["first_eval_walltime_s"] = float(walltimes[0])
        out["last_eval_walltime_s"] = float(walltimes[-1])
    for key in ("eval_total_operating_cost", "eval_reserve_shortfall_rate", "eval_thermal_violation_rate"):
        arr = arrays.get(key)
        if arr is not None and arr.size:
            out[f"final_{key}"] = float(arr[-1])
    return out


def _artifact_checks(run: dict[str, Any]) -> dict[str, Any]:
    present = {}
    for key in REQUIRED_ARTIFACTS:
        path = _artifact_path(run, key)
        present[key] = bool(path is not None and path.exists())
    return {
        "required_artifacts_present": present,
        "all_required_artifacts_present": all(present.values()),
    }


def _python_backend_contract(env_info: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Normalize Python-backend task provenance across manifest revisions."""
    data_source = env_info.get("data_source")
    if data_source is None and str(env_info.get("data_provenance_ok")).lower() == "true":
        data_source = "gb_real"
    benchmark_split = env_info.get("benchmark_split") or env_info.get("split_ids")
    profile_window = env_info.get("profile_window") or env_info.get("split_window")
    details = {
        "data_source": data_source,
        "benchmark_split": benchmark_split,
        "profile_window": profile_window,
    }
    ok = (
        data_source == "gb_real"
        and benchmark_split == "train"
        and profile_window == EXPECTED_TRAIN_WINDOW
    )
    return details, ok


def _cell_audit(cell: Phase2Cell, run: dict[str, Any] | None) -> dict[str, Any]:
    if run is None:
        return {
            "present": False,
            "cell_ok": False,
            "expected": {
                "backend": cell.backend,
                "device": cell.device,
                "algo": cell.algo,
            },
        }

    task_cfg, train_cfg, driver_cfg = _config_views(run)
    curves = _curve_checks(run)
    artifacts = _artifact_checks(run)
    metrics = run.get("metrics") or {}
    env_info = run.get("env_info") or {}
    notes = str(run.get("notes") or "")

    total_timesteps = (
        metrics.get("total_timesteps")
        or train_cfg.get("total_timesteps")
        or driver_cfg.get("total_timesteps")
        or 0
    )
    num_envs = train_cfg.get("num_envs")
    if num_envs is None:
        num_envs = driver_cfg.get("n_envs")

    split_ok = run.get("split") == "train"
    total_timesteps_ok = int(total_timesteps or 0) == EXPECTED_TOTAL_TIMESTEPS
    n_steps_ok = int(train_cfg.get("n_steps") or 0) == EXPECTED_N_STEPS
    num_envs_ok = int(num_envs or 0) == EXPECTED_NUM_ENVS
    eval_freq_ok = int(train_cfg.get("eval_freq") or EXPECTED_EVAL_FREQ) == EXPECTED_EVAL_FREQ
    eval_episodes_ok = int(train_cfg.get("eval_episodes") or EXPECTED_EVAL_EPISODES) == EXPECTED_EVAL_EPISODES
    train_window_ok = (
        f"{task_cfg.get('gb_train_start', '')}..{task_cfg.get('gb_train_end', '')}" == EXPECTED_TRAIN_WINDOW
    )
    forecast_ok = int(task_cfg.get("forecast_horizon_steps") or 0) == 4
    env_info_basic_ok = bool(env_info.get("python"))

    contract_ok = True
    contract_details: dict[str, Any] = {}
    if cell.backend in {"sb3", "sbx"}:
        contract_details, contract_ok = _python_backend_contract(env_info)
    else:
        contract_details = {"notes": notes}
        contract_ok = (
            "gb_split=train" in notes
            and "gb_window=2025-04-01..2025-12-31" in notes
        )

    checks = {
        "split_train": split_ok,
        "status_completed": run.get("status") == "completed",
        "total_timesteps_20m": total_timesteps_ok,
        "n_steps_48": n_steps_ok,
        "num_envs_256": num_envs_ok,
        "eval_freq_100k": eval_freq_ok,
        "eval_episodes_8": eval_episodes_ok,
        "task_window_train_2025_04_to_2025_12": train_window_ok,
        "forecast_horizon_steps_4": forecast_ok,
        "env_info_present": env_info_basic_ok,
        "backend_contract_ok": contract_ok,
        "required_artifacts_present": artifacts["all_required_artifacts_present"],
        "required_curves_present": curves["all_required_curves_present"],
        "curve_lengths_match": curves["curve_lengths_match"],
        "timesteps_monotone": curves["timesteps_monotone_non_decreasing"],
        "walltimes_monotone": curves["walltimes_monotone_non_decreasing"],
    }

    return {
        "present": True,
        "cell_ok": all(checks.values()),
        "run_id": run.get("run_id"),
        "backend": run.get("backend"),
        "device": run.get("device"),
        "algo": run.get("algo"),
        "seed": run.get("seed"),
        "timestamp": run.get("timestamp"),
        "walltime_s": float(run.get("walltime_s") or 0.0),
        "throughput_sps": float(run.get("throughput_sps") or 0.0),
        "checks": checks,
        "contract_details": contract_details,
        "artifact_audit": artifacts,
        "curve_audit": curves,
    }


def build_audit(*, after: str | None = None) -> dict[str, Any]:
    if after is None:
        after = _default_campaign_after()
    all_rows = _load_runs()
    rows = _filter_after(all_rows, after)
    cells = {
        cell.name: _cell_audit(cell, _choose_cell_run(cell, rows))
        for cell in EXPECTED_CELLS
    }
    return {
        "task": "tso",
        "phase": "phase2_cross_backend_learning_curves",
        "required_cells": [cell.__dict__ for cell in EXPECTED_CELLS],
        "required_artifacts": list(REQUIRED_ARTIFACTS),
        "required_curves": list(REQUIRED_CURVES),
        "expected_contract": {
            "split": "train",
            "total_timesteps": EXPECTED_TOTAL_TIMESTEPS,
            "num_envs": EXPECTED_NUM_ENVS,
            "n_steps": EXPECTED_N_STEPS,
            "eval_freq": EXPECTED_EVAL_FREQ,
            "eval_episodes": EXPECTED_EVAL_EPISODES,
            "gb_train_window": EXPECTED_TRAIN_WINDOW,
        },
        "filters": {
            "after": after,
            "candidate_rows": len(rows),
            "total_rows_scanned": len(all_rows),
        },
        "cells": cells,
        "all_cells_present": all(cell["present"] for cell in cells.values()),
        "all_cells_ok": all(cell["cell_ok"] for cell in cells.values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit current TSO Phase-2 backend rows.")
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="exit non-zero when any required cell is missing or violates the contract",
    )
    parser.add_argument(
        "--after",
        default=None,
        help=(
            "Only audit runs with timestamp >= this campaign_start_iso. "
            "Defaults to benchmark_protocol.current_campaign_start_iso."
        ),
    )
    args = parser.parse_args(argv)

    after = args.after or _default_campaign_after()
    if args.enforce and not after:
        print(
            "[phase2_backend_audit] campaign_start_iso is required with "
            "--enforce to avoid accepting old campaign rows. Set "
            "benchmark_protocol.current_campaign_start_iso or pass --after."
        )
        return 2

    payload = build_audit(after=after)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[phase2_backend_audit] wrote {OUT_PATH}")

    if args.enforce and not payload["all_cells_ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
