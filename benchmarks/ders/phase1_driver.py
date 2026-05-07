#!/usr/bin/env python
"""DERs Phase-1 isolated rerun driver."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.common.configs import dump_yaml, load_config, load_task_config
from benchmarks.common.gpu_selector import wait_for_idle_gpu
from benchmarks.common.io import load_manifest_filtered
from benchmarks.ders.phase1_analysis import _candidate_rows, generate_phase1_analysis

SOURCE_TASK_DIR = Path(__file__).resolve().parent
DEFAULT_CAMPAIGN_ROOT = _PROJECT_ROOT / "local_tmp"
PYTHON = sys.executable
SUBPROC_ENV = {
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.7",
}
SMOKE_TIMESTEPS = 192_000
SWEEP_TIMESTEPS = 1_000_000
FORMAL_TIMESTEPS = 10_000_000
SMOKE_SPLITS = ("iid", "voltage_tightening")
FORMAL_SPLITS = ("train", "iid", "voltage_tightening", "pv_penetration_shift", "load_stress")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _campaign_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _copy_configs(dst_task_dir: Path) -> None:
    configs_dst = dst_task_dir / "configs"
    configs_dst.mkdir(parents=True, exist_ok=True)
    for src in (SOURCE_TASK_DIR / "configs").iterdir():
        if src.is_file():
            shutil.copy2(src, configs_dst / src.name)


def _load_yaml(path: Path) -> dict:
    return load_config(path)


def _save_yaml(path: Path, payload: dict) -> None:
    dump_yaml(payload, path)


def _set_eval_episodes(task_dir: Path, n_eval_episodes: int) -> None:
    for path in (task_dir / "configs").glob("eval_*.yaml"):
        cfg = _load_yaml(path)
        cfg["n_eval_episodes"] = int(n_eval_episodes)
        _save_yaml(path, cfg)


def _set_total_timesteps(task_dir: Path, total_timesteps: int) -> None:
    for name in ("train_ippo.yaml", "train_ippo_safe.yaml", "train_ippo_lagrangian.yaml"):
        path = task_dir / "configs" / name
        cfg = _load_yaml(path)
        cfg["total_timesteps"] = int(total_timesteps)
        _save_yaml(path, cfg)


def _set_algo_param(task_dir: Path, algo: str, key: str, value) -> None:
    filename = {
        "ippo": "train_ippo.yaml",
        "ippo_safe": "train_ippo_safe.yaml",
        "ippo_lagrangian": "train_ippo_lagrangian.yaml",
    }[algo]
    path = task_dir / "configs" / filename
    cfg = _load_yaml(path)
    cfg[key] = value
    _save_yaml(path, cfg)


def _set_train_window_settings(
    task_dir: Path,
    *,
    count: int,
    selector: str,
    load_scale: float = 1.0,
    pv_scale: float = 1.0,
    score_v_min: float | None = None,
    score_v_max: float | None = None,
) -> None:
    path = task_dir / "configs" / "task.yaml"
    cfg = _load_yaml(path)
    cfg["train_window_count"] = int(count)
    cfg["train_window_selector"] = str(selector)
    cfg["train_load_scale"] = float(load_scale)
    cfg["train_pv_scale"] = float(pv_scale)
    if score_v_min is None:
        cfg.pop("train_window_score_v_min", None)
    else:
        cfg["train_window_score_v_min"] = float(score_v_min)
    if score_v_max is None:
        cfg.pop("train_window_score_v_max", None)
    else:
        cfg["train_window_score_v_max"] = float(score_v_max)
    _save_yaml(path, cfg)


def _run_logged(
    task_dir: Path,
    *,
    stage_name: str,
    cmd: list[str],
    env: dict[str, str],
) -> None:
    logs_dir = task_dir / "results" / "_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{stage_name}.log"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n=== {datetime.now(timezone.utc).isoformat()} START ===\n")
        f.write(" ".join(cmd) + "\n")
        f.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(_PROJECT_ROOT),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        f.write(f"=== END rc={proc.returncode} ===\n")
    if proc.returncode != 0:
        raise RuntimeError(f"{stage_name} failed with rc={proc.returncode}; see {log_path}")


def _run_pipeline(
    task_dir: Path,
    *,
    stage_name: str,
    algos: list[str],
    baseline_splits: tuple[str, ...],
    eval_splits: tuple[str, ...],
    gpu_index: int,
    skip_baselines: bool = False,
    skip_plots: bool = False,
) -> None:
    env = {**os.environ, **SUBPROC_ENV, "CUDA_VISIBLE_DEVICES": str(gpu_index)}
    cmd = [
        PYTHON,
        str(SOURCE_TASK_DIR / "run_all.py"),
        "--task-dir",
        str(task_dir),
        "--seeds",
        "0",
        "--algos",
        *algos,
        "--baseline-splits",
        *baseline_splits,
        "--eval-splits",
        *eval_splits,
    ]
    if skip_baselines:
        cmd.append("--skip-baselines")
    if skip_plots:
        cmd.append("--skip-plots")
    _run_logged(task_dir, stage_name=stage_name, cmd=cmd, env=env)


def _run_baselines_only(
    task_dir: Path,
    *,
    stage_name: str,
    baseline_splits: tuple[str, ...],
    gpu_index: int,
) -> None:
    env = {**os.environ, **SUBPROC_ENV, "CUDA_VISIBLE_DEVICES": str(gpu_index)}
    cmd = [
        PYTHON,
        str(SOURCE_TASK_DIR / "run_all.py"),
        "--task-dir",
        str(task_dir),
        "--only",
        "baselines",
        "--seeds",
        "0",
        "--baseline-splits",
        *baseline_splits,
    ]
    _run_logged(task_dir, stage_name=stage_name, cmd=cmd, env=env)


def _run_command(task_dir: Path, *, stage_name: str, cmd: list[str], gpu_index: int | None = None) -> None:
    env = {**os.environ, **SUBPROC_ENV}
    if gpu_index is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    _run_logged(task_dir, stage_name=stage_name, cmd=cmd, env=env)


def _latest_candidate_rows(task_dir: Path, *, after: str) -> list[dict]:
    return _candidate_rows(load_manifest_filtered(task_dir, after=after), seed=0)


def _lagrangian_has_signal(task_dir: Path, *, after: str) -> bool:
    records = load_manifest_filtered(task_dir, after=after)
    for record in records:
        if record.algo != "ippo_lagrangian" or record.split != "train":
            continue
        path = task_dir / "results" / "artifacts" / f"{record.run_id}_mean_cost_voltage_violation.npy"
        if not path.exists():
            continue
        arr = np.load(path)
        if np.any(arr > 0.0):
            return True
    return False


def _best_candidate(task_dir: Path, *, after: str) -> dict:
    rows = _latest_candidate_rows(task_dir, after=after)
    if not rows:
        raise RuntimeError(f"No candidate rows found in {task_dir} after {after}")
    return rows[0]


def _formalise_candidate(task_dir: Path, candidate: dict) -> None:
    train_run_id = candidate["train_run_id"]
    cfg_snapshot_path = task_dir / "results" / "artifacts" / f"{train_run_id}_config.json"
    if not cfg_snapshot_path.exists():
        raise FileNotFoundError(f"Missing config snapshot for {train_run_id}: {cfg_snapshot_path}")
    snapshot = json.loads(cfg_snapshot_path.read_text(encoding="utf-8"))
    train_cfg = snapshot.get("train_config", {})
    bank_cfg = snapshot.get("train_window_bank", {})
    algo = candidate["algo"]
    cfg_path = task_dir / "configs" / {
        "ippo": "train_ippo.yaml",
        "ippo_safe": "train_ippo_safe.yaml",
        "ippo_lagrangian": "train_ippo_lagrangian.yaml",
    }[algo]
    train_cfg["total_timesteps"] = FORMAL_TIMESTEPS
    _save_yaml(cfg_path, train_cfg)
    _set_train_window_settings(
        task_dir,
        count=int(bank_cfg.get("train_window_count", 8)),
        selector=str(bank_cfg.get("train_window_selector", "uniform")),
        load_scale=float(bank_cfg.get("train_load_scale", 1.0)),
        pv_scale=float(bank_cfg.get("train_pv_scale", 1.0)),
        score_v_min=bank_cfg.get("train_window_score_v_min"),
        score_v_max=bank_cfg.get("train_window_score_v_max"),
    )


def _write_metadata(task_dir: Path, payload: dict) -> None:
    _write_json(task_dir / "results" / "phase1_driver_metadata.json", payload)


def _retune_round(
    task_dir: Path,
    *,
    round_idx: int,
) -> None:
    if round_idx == 1:
        _set_train_window_settings(
            task_dir,
            count=16,
            selector="hardest_voltage_margin",
            score_v_min=0.96,
            score_v_max=1.04,
        )
    elif round_idx == 2:
        _set_train_window_settings(
            task_dir,
            count=16,
            selector="hardest_voltage_margin",
            load_scale=1.15,
            pv_scale=0.9,
            score_v_min=0.96,
            score_v_max=1.04,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="DERs Phase-1 isolated rerun driver.")
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    parser.add_argument("--gpu-index", type=int, default=None)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--skip-formal", action="store_true")
    args = parser.parse_args()

    probe_log_path = args.campaign_root / "ders_gpu_probe.jsonl"
    if args.gpu_index is None:
        gpu = wait_for_idle_gpu(probe_log_path=probe_log_path)
        gpu_index = int(gpu.index)
    else:
        gpu_index = int(args.gpu_index)
    campaign_start_iso = _campaign_iso_now()
    isolated_dir = args.campaign_root / f"ders_gpu{gpu_index}_stage1_{_utc_stamp()}"
    _copy_configs(isolated_dir)

    metadata = {
        "campaign_start_iso": campaign_start_iso,
        "gpu_index": gpu_index,
        "source_task_dir": str(SOURCE_TASK_DIR),
        "isolated_task_dir": str(isolated_dir),
    }
    _write_metadata(isolated_dir, metadata)

    # Smoke
    _set_total_timesteps(isolated_dir, SMOKE_TIMESTEPS)
    _set_eval_episodes(isolated_dir, 8)
    _set_train_window_settings(
        isolated_dir,
        count=8,
        selector="uniform",
    )
    smoke_after = _campaign_iso_now()
    _run_pipeline(
        isolated_dir,
        stage_name="smoke",
        algos=["ippo", "ippo_safe", "ippo_lagrangian"],
        baseline_splits=SMOKE_SPLITS,
        eval_splits=SMOKE_SPLITS,
        gpu_index=gpu_index,
    )
    smoke_candidates = _latest_candidate_rows(isolated_dir, after=smoke_after)
    metadata["smoke_candidates"] = smoke_candidates
    _write_metadata(isolated_dir, metadata)
    if args.smoke_only:
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    # Sweep round(s)
    best_candidate: dict | None = None
    for round_idx in range(3):
        if round_idx > 0:
            _retune_round(isolated_dir, round_idx=round_idx)
        _set_total_timesteps(isolated_dir, SWEEP_TIMESTEPS)
        _set_eval_episodes(isolated_dir, 12)
        sweep_after = _campaign_iso_now()
        _run_baselines_only(
            isolated_dir,
            stage_name=f"sweep_baselines_r{round_idx}",
            baseline_splits=SMOKE_SPLITS,
            gpu_index=gpu_index,
        )

        sweep_specs = [
            ("ippo", "voltage_penalty", 4.0),
            ("ippo", "voltage_penalty", 8.0),
            ("ippo", "voltage_penalty", 12.0),
            ("ippo_safe", "voltage_penalty", 8.0),
            ("ippo_safe", "voltage_penalty", 12.0),
        ]
        if _lagrangian_has_signal(isolated_dir, after=smoke_after):
            sweep_specs.extend(
                [
                    ("ippo_lagrangian", "lambda_lr", 5e-5),
                    ("ippo_lagrangian", "lambda_lr", 1e-4),
                ]
            )

        for algo, key, value in sweep_specs:
            _set_algo_param(isolated_dir, algo, key, value)
            _run_pipeline(
                isolated_dir,
                stage_name=f"sweep_{algo}_{key}_{value}_r{round_idx}",
                algos=[algo],
                baseline_splits=SMOKE_SPLITS,
                eval_splits=SMOKE_SPLITS,
                gpu_index=gpu_index,
                skip_baselines=True,
                skip_plots=True,
            )

        sweep_candidates = _latest_candidate_rows(isolated_dir, after=sweep_after)
        metadata[f"sweep_round_{round_idx}"] = sweep_candidates
        _write_metadata(isolated_dir, metadata)
        if sweep_candidates and sweep_candidates[0]["passes_physics_gate"]:
            best_candidate = sweep_candidates[0]
            break
        if round_idx == 2 and sweep_candidates:
            best_candidate = sweep_candidates[0]

    if best_candidate is None:
        raise RuntimeError("Sweep produced no candidate rows.")

    metadata["best_candidate"] = best_candidate
    _write_metadata(isolated_dir, metadata)

    if args.skip_formal:
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    # Formal seed-0 Phase 1
    _formalise_candidate(isolated_dir, best_candidate)
    _set_total_timesteps(isolated_dir, FORMAL_TIMESTEPS)
    _set_eval_episodes(isolated_dir, int(load_task_config(isolated_dir).get("eval_episodes", 30)))
    formal_after = _campaign_iso_now()
    _run_pipeline(
        isolated_dir,
        stage_name="formal",
        algos=[best_candidate["algo"]],
        baseline_splits=FORMAL_SPLITS,
        eval_splits=FORMAL_SPLITS,
        gpu_index=gpu_index,
    )

    formal_candidate = _best_candidate(isolated_dir, after=formal_after)
    metadata["formal_candidate"] = formal_candidate
    _write_metadata(isolated_dir, metadata)

    _run_command(
        isolated_dir,
        stage_name="derive_target",
        cmd=[
            PYTHON,
            "-m",
            "benchmarks.common.experiment_ops",
            "derive_target",
            "--task",
            "ders",
            "--task-dir",
            str(isolated_dir),
            "--reference-run-id",
            formal_candidate["train_run_id"],
            "--force",
        ],
    )
    summary = generate_phase1_analysis(
        isolated_dir,
        after=campaign_start_iso,
        seed=0,
        train_run_id=formal_candidate["train_run_id"],
    )
    metadata["phase1_analysis"] = summary
    _write_metadata(isolated_dir, metadata)

    _run_command(
        isolated_dir,
        stage_name="seed0_readiness",
        cmd=[
            PYTHON,
            "-m",
            "benchmarks.common.experiment_ops",
            "seed0_readiness",
            "--task",
            "ders",
            "--task-dir",
            str(isolated_dir),
            "--after",
            campaign_start_iso,
            "--enforce",
        ],
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
