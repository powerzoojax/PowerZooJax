#!/usr/bin/env python
"""Paper-facing audit for DC Microgrid Phase-2 backend/device results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.dc_microgrid.analysis.phase2_physical_audit import (  # noqa: E402
    _replay_saved_actions,
)
from benchmarks.dc_microgrid.summarize import (  # noqa: E402
    _paired_signflip_permutation_test,
)


TASK_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = TASK_DIR / "results"
RUNS_DIR = RESULTS_DIR / "runs"
ARTIFACTS_DIR = RESULTS_DIR / "artifacts"
SUMMARY_PATH = RESULTS_DIR / "summary" / "latest.json"


@dataclass(frozen=True)
class CellSpec:
    cell_id: str
    backend: str
    device: str
    algo: str
    curve_source: str  # "train" for jax, "eval" for driver rows


PHASE2_CELLS = (
    CellSpec("jax_gpu_ppo", "jax_rejax", "gpu", "ppo", "train"),
    CellSpec("jax_cpu_ppo", "jax_rejax", "cpu", "ppo", "train"),
    CellSpec("sb3_cuda_ppo", "sb3", "cuda", "ppo", "eval"),
    CellSpec("sb3_cpu_ppo", "sb3", "cpu", "ppo", "eval"),
    CellSpec("sbx_cuda_ppo", "sbx", "cuda", "sbx_ppo", "eval"),
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _latest_runs(*, split: str, backend: str, device: str, algo: str, require_artifacts: set[str] | None = None) -> dict[int, dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    for path in RUNS_DIR.glob("dc_microgrid_*.json"):
        rec = _load_json(path)
        if (
            rec.get("task") != "dc_microgrid"
            or rec.get("status") != "completed"
            or rec.get("split") != split
            or rec.get("backend") != backend
            or rec.get("device") != device
            or rec.get("algo") != algo
        ):
            continue
        arts = set((rec.get("artifacts") or {}).keys())
        if require_artifacts and not require_artifacts.issubset(arts):
            continue
        seed = int(rec["seed"])
        prev = selected.get(seed)
        if prev is None or str(rec.get("timestamp", "")) > str(prev.get("timestamp", "")):
            selected[seed] = rec
    return selected


def _eval_runs(cell: CellSpec) -> dict[int, dict[str, Any]]:
    return _latest_runs(
        split="iid",
        backend=cell.backend,
        device=cell.device,
        algo=cell.algo,
        require_artifacts={"per_episode"},
    )


def _curve_runs(cell: CellSpec) -> dict[int, dict[str, Any]]:
    if cell.curve_source == "train":
        return _latest_runs(
            split="train",
            backend=cell.backend,
            device=cell.device,
            algo=cell.algo,
            require_artifacts={"config", "timesteps", "learning_curve_eval_return", "learning_curve_eval_walltimes"},
        )
    return _latest_runs(
        split="iid",
        backend=cell.backend,
        device=cell.device,
        algo=cell.algo,
        require_artifacts={"config", "timesteps", "learning_curve_eval_return", "learning_curve_eval_walltimes"},
    )


def _config_snapshot(rec: dict[str, Any]) -> dict[str, Any]:
    rel = (rec.get("artifacts") or {}).get("config")
    if not rel:
        raise FileNotFoundError(f"Run {rec.get('run_id')} missing config artifact")
    return _load_json(RESULTS_DIR / rel)


def _contract_view(cell: CellSpec, cfg: dict[str, Any]) -> dict[str, Any]:
    task_cfg = cfg.get("task_config") or {}
    if cell.backend == "jax_rejax":
        train_cfg = cfg.get("train_config") or {}
        return {
            "total_timesteps": int(train_cfg.get("total_timesteps")),
            "n_envs": int(train_cfg.get("num_envs")),
            "train_split": "train",
            "eval_split": "iid",
            "requested_continuous_action_dist": str(train_cfg.get("continuous_action_dist")),
            "effective_continuous_action_dist": str(train_cfg.get("continuous_action_dist")),
            "policy_class": "jax_native_beta_ppo",
            "data_source": str(task_cfg.get("data_source")),
            "case_overrides": task_cfg.get("case_overrides"),
            "dg_autobalance": bool(task_cfg.get("dg_autobalance")),
            "reward_shaping_weights": task_cfg.get("reward_shaping_weights"),
            "eval_episodes": int(task_cfg.get("eval_episodes")),
        }
    pz = cfg.get("powerzoo_driver_config") or {}
    return {
        "total_timesteps": int(pz.get("total_timesteps")),
        "n_envs": int(pz.get("n_envs")),
        "train_split": str(pz.get("train_split")),
        "eval_split": str(pz.get("eval_split")),
        "requested_continuous_action_dist": str(pz.get("requested_continuous_action_dist")),
        "effective_continuous_action_dist": str(pz.get("effective_continuous_action_dist")),
        "policy_class": str(pz.get("policy_class")),
        "data_source": str(task_cfg.get("data_source")),
        "case_overrides": task_cfg.get("case_overrides"),
        "dg_autobalance": bool(task_cfg.get("dg_autobalance")),
        "reward_shaping_weights": task_cfg.get("reward_shaping_weights"),
        "eval_episodes": int(task_cfg.get("eval_episodes")),
    }


def _artifact_audit(cell: CellSpec, eval_rec: dict[str, Any], curve_rec: dict[str, Any]) -> dict[str, Any]:
    eval_artifacts = eval_rec.get("artifacts") or {}
    curve_artifacts = curve_rec.get("artifacts") or {}

    def _exists(rel: str | None) -> bool:
        return bool(rel) and (RESULTS_DIR / rel).exists()

    out = {
        "config_exists": _exists(curve_artifacts.get("config")),
        "timesteps_exists": _exists(curve_artifacts.get("timesteps")),
        "learning_curve_eval_return_exists": _exists(curve_artifacts.get("learning_curve_eval_return")),
        "learning_curve_eval_walltimes_exists": _exists(curve_artifacts.get("learning_curve_eval_walltimes")),
        "learning_curve_train_return_exists": _exists(curve_artifacts.get("learning_curve_train_return")),
        "per_episode_exists": _exists(eval_artifacts.get("per_episode")),
        "trajectory_exists": _exists(eval_artifacts.get("trajectory")),
    }

    for name, rel in (
        ("timesteps", curve_artifacts.get("timesteps")),
        ("learning_curve_eval_return", curve_artifacts.get("learning_curve_eval_return")),
        ("learning_curve_eval_walltimes", curve_artifacts.get("learning_curve_eval_walltimes")),
        ("learning_curve_train_return", curve_artifacts.get("learning_curve_train_return")),
    ):
        if _exists(rel):
            out[f"{name}_points"] = int(np.load(RESULTS_DIR / rel).shape[0])
        else:
            out[f"{name}_points"] = None
    return out


def _per_episode_values(eval_runs: dict[int, dict[str, Any]]) -> np.ndarray:
    vals: list[float] = []
    for seed in sorted(eval_runs):
        rec = eval_runs[seed]
        rel = (rec.get("artifacts") or {}).get("per_episode")
        if not rel:
            continue
        rows = _load_json(RESULTS_DIR / rel)
        vals.extend(float(r["episode_reward"]) for r in rows)
    return np.asarray(vals, dtype=np.float64)


def _timing_stats(source_runs: dict[int, dict[str, Any]], *, source_label: str) -> dict[str, Any]:
    wall = [float(rec["walltime_s"]) for rec in source_runs.values()]
    tps = [float(rec["throughput_sps"]) for rec in source_runs.values() if rec.get("throughput_sps") is not None]
    return {
        "source": source_label,
        "walltime_mean_s": float(np.mean(wall)),
        "walltime_std_s": float(np.std(wall, ddof=0)),
        "throughput_mean_sps": float(np.mean(tps)) if tps else None,
        "throughput_std_sps": float(np.std(tps, ddof=0)) if len(tps) > 1 else (0.0 if tps else None),
    }


def _summary_row(summary: dict[str, Any], cell: CellSpec) -> dict[str, Any]:
    for row in summary.get("phase2_rows", []):
        if (
            row.get("split") == "iid"
            and row.get("backend") == cell.backend
            and row.get("device") == cell.device
            and row.get("algo") == cell.algo
        ):
            return row
    raise KeyError(f"Missing summary row for {cell.cell_id}")


def _paired_test_payload(left: CellSpec, right: CellSpec, cell_payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    left_vals = np.asarray(cell_payloads[left.cell_id]["episode_reward_pairs"], dtype=np.float64)
    right_vals = np.asarray(cell_payloads[right.cell_id]["episode_reward_pairs"], dtype=np.float64)
    n = min(left_vals.size, right_vals.size)
    left_vals = left_vals[:n]
    right_vals = right_vals[:n]
    result = _paired_signflip_permutation_test(left_vals, right_vals)
    return {
        "left_cell": left.cell_id,
        "right_cell": right.cell_id,
        **result,
    }


def _numerical_tail_audit(cell: CellSpec, eval_runs: dict[int, dict[str, Any]], *, tol: float = 1e-6) -> dict[str, Any]:
    if cell.backend not in {"sb3", "sbx"}:
        return {"applicable": False}

    strict_rates: list[float] = []
    tolerant_rates: list[float] = []
    max_balance_errors: list[float] = []
    mean_balance_errors: list[float] = []
    episode_rewards_saved: list[float] = []
    episode_rewards_replayed: list[float] = []

    for seed, rec in sorted(eval_runs.items()):
        run_id = str(rec["run_id"])
        per_episode = _load_json(RESULTS_DIR / (rec["artifacts"]["per_episode"]))
        actions = np.load(RESULTS_DIR / rec["artifacts"]["trajectory"])["actions"]
        for episode_idx, episode_metrics in enumerate(per_episode):
            rows = _replay_saved_actions(
                seed=int(seed),
                episode_idx=int(episode_idx),
                split="iid",
                actions=actions[episode_idx],
                n_episodes=len(per_episode),
            )
            cost_sum = np.asarray([float(r.get("cost_sum", 0.0)) for r in rows], dtype=np.float64)
            load = np.asarray([float(r.get("p_load_mw", 0.0)) for r in rows], dtype=np.float64)
            supply = np.asarray(
                [
                    float(r.get("p_pv_mw", 0.0))
                    + float(r.get("p_dg_mw", 0.0))
                    + float(r.get("p_batt_mw", 0.0))
                    + float(r.get("p_grid_import_mw", 0.0))
                    for r in rows
                ],
                dtype=np.float64,
            )
            balance = np.abs(load - supply) / np.maximum(load, 1e-6)
            strict_rates.append(float(np.mean(cost_sum == 0.0)))
            tolerant_rates.append(float(np.mean(cost_sum <= tol)))
            max_balance_errors.append(float(np.max(balance)))
            mean_balance_errors.append(float(np.mean(balance)))
            episode_rewards_saved.append(float(episode_metrics["episode_reward"]))
            episode_rewards_replayed.append(float(np.sum([float(r.get("reward", 0.0)) for r in rows])))

    return {
        "applicable": True,
        "tolerance_cost_sum": tol,
        "strict_feasibility_rate_mean": float(np.mean(strict_rates)),
        "tolerant_feasibility_rate_mean": float(np.mean(tolerant_rates)),
        "strict_to_tolerant_gap": float(np.mean(tolerant_rates) - np.mean(strict_rates)),
        "mean_power_balance_error_mean": float(np.mean(mean_balance_errors)),
        "max_power_balance_error_max": float(np.max(max_balance_errors)),
        "mean_abs_replay_reward_gap": float(np.mean(np.abs(np.asarray(episode_rewards_saved) - np.asarray(episode_rewards_replayed)))),
    }


def main() -> None:
    summary = _load_json(SUMMARY_PATH)
    cell_payloads: dict[str, dict[str, Any]] = {}

    reference_contract: dict[str, Any] | None = None
    fairness_checks: dict[str, list[Any]] = {
        "total_timesteps": [],
        "n_envs": [],
        "train_split": [],
        "eval_split": [],
        "requested_continuous_action_dist": [],
        "effective_continuous_action_dist": [],
        "data_source": [],
        "case_overrides": [],
        "dg_autobalance": [],
        "reward_shaping_weights": [],
        "eval_episodes": [],
    }

    for cell in PHASE2_CELLS:
        eval_runs = _eval_runs(cell)
        curve_runs = _curve_runs(cell)
        if sorted(eval_runs) != [0, 1, 2, 3, 4]:
            raise RuntimeError(f"{cell.cell_id} missing eval seeds: {sorted(eval_runs)}")
        if sorted(curve_runs) != [0, 1, 2, 3, 4]:
            raise RuntimeError(f"{cell.cell_id} missing curve/config seeds: {sorted(curve_runs)}")

        contracts: list[dict[str, Any]] = []
        artifact_rows: list[dict[str, Any]] = []
        for seed in range(5):
            cfg = _config_snapshot(curve_runs[seed])
            contract = _contract_view(cell, cfg)
            contracts.append(contract)
            artifact_rows.append(_artifact_audit(cell, eval_runs[seed], curve_runs[seed]))

        unique_contract = {
            key: sorted({_stable_json(c[key]) for c in contracts})
            for key in contracts[0]
        }
        for key in fairness_checks:
            fairness_checks[key].extend(unique_contract[key])

        if cell.cell_id == "jax_gpu_ppo":
            reference_contract = contracts[0]

        cell_payloads[cell.cell_id] = {
            "cell": {
                "backend": cell.backend,
                "device": cell.device,
                "algo": cell.algo,
            },
            "summary_metrics": {
                key: _summary_row(summary, cell).get(key)
                for key in [
                    "episode_reward_mean",
                    "episode_reward_std",
                    "n_seeds",
                    "feasibility_rate_mean",
                    "sla_violation_rate_mean",
                    "mean_cost_power_balance_mean",
                ]
            },
            "timing": _timing_stats(curve_runs, source_label=cell.curve_source),
            "contract_unique_values": unique_contract,
            "artifact_audit": {
                "all_config_exists": all(r["config_exists"] for r in artifact_rows),
                "all_timesteps_exists": all(r["timesteps_exists"] for r in artifact_rows),
                "all_eval_curves_exist": all(
                    r["learning_curve_eval_return_exists"] and r["learning_curve_eval_walltimes_exists"]
                    for r in artifact_rows
                ),
                "all_train_curves_exist": all(r["learning_curve_train_return_exists"] for r in artifact_rows),
                "all_per_episode_exists": all(r["per_episode_exists"] for r in artifact_rows),
                "all_trajectory_exists": all(r["trajectory_exists"] for r in artifact_rows),
                "timesteps_points_unique": sorted({r["timesteps_points"] for r in artifact_rows}),
                "learning_curve_eval_return_points_unique": sorted({r["learning_curve_eval_return_points"] for r in artifact_rows}),
                "learning_curve_eval_walltimes_points_unique": sorted({r["learning_curve_eval_walltimes_points"] for r in artifact_rows}),
                "learning_curve_train_return_points_unique": sorted({r["learning_curve_train_return_points"] for r in artifact_rows if r["learning_curve_train_return_points"] is not None}),
                "curve_logging_note": (
                    "JAX train rows include a step-0 point (21 points total); "
                    "PowerZoo driver rows log 20 checkpoints over 1M steps."
                    if cell.backend == "jax_rejax"
                    else "PowerZoo driver rows log 20 checkpoints over 1M steps."
                ),
            },
            "episode_reward_pairs": _per_episode_values(eval_runs).tolist(),
            "selected_run_ids": {
                "eval": {str(seed): str(rec["run_id"]) for seed, rec in sorted(eval_runs.items())},
                "curve_source": {str(seed): str(rec["run_id"]) for seed, rec in sorted(curve_runs.items())},
            },
            "numerical_tail_audit": _numerical_tail_audit(cell, eval_runs),
        }

    assert reference_contract is not None
    reference = reference_contract
    fairness_audit = {
        "reference_cell": "jax_gpu_ppo",
        "reference_contract": reference,
        "all_cells_match_total_timesteps": sorted(set(fairness_checks["total_timesteps"])) == [_stable_json(reference["total_timesteps"])],
        "all_cells_match_num_envs": sorted(set(fairness_checks["n_envs"])) == [_stable_json(reference["n_envs"])],
        "all_cells_train_on_train_split": sorted(set(fairness_checks["train_split"])) == [_stable_json("train")],
        "all_cells_eval_on_iid": sorted(set(fairness_checks["eval_split"])) == [_stable_json("iid")],
        "all_cells_match_data_source": sorted(set(fairness_checks["data_source"])) == [_stable_json(reference["data_source"])],
        "all_cells_match_case_overrides": sorted(set(fairness_checks["case_overrides"])) == [_stable_json(reference["case_overrides"])],
        "all_cells_match_dg_autobalance": sorted(set(fairness_checks["dg_autobalance"])) == [_stable_json(reference["dg_autobalance"])],
        "all_cells_match_reward_shaping": sorted(set(fairness_checks["reward_shaping_weights"])) == [_stable_json(reference["reward_shaping_weights"])],
        "all_cells_match_eval_episodes": sorted(set(fairness_checks["eval_episodes"])) == [_stable_json(reference["eval_episodes"])],
        "all_python_cells_beta_actor_aligned": all(
            cell_payloads[cell.cell_id]["contract_unique_values"]["requested_continuous_action_dist"] == ['"beta"']
            and cell_payloads[cell.cell_id]["contract_unique_values"]["effective_continuous_action_dist"] == ['"beta"']
            for cell in PHASE2_CELLS
            if cell.backend in {"sb3", "sbx"}
        ),
        "notes": [
            "JAX rows store curves on the train run and per-episode metrics on the eval run.",
            "PowerZoo driver rows store curves and per-episode metrics on the IID eval record itself.",
            "JAX curve arrays have 21 points because they include step 0; driver rows have 20 checkpoint points.",
        ],
    }

    paired_tests = [
        _paired_test_payload(PHASE2_CELLS[0], PHASE2_CELLS[1], cell_payloads),
        _paired_test_payload(PHASE2_CELLS[0], PHASE2_CELLS[4], cell_payloads),
        _paired_test_payload(PHASE2_CELLS[4], PHASE2_CELLS[2], cell_payloads),
        _paired_test_payload(PHASE2_CELLS[2], PHASE2_CELLS[3], cell_payloads),
    ]
    tests_by_pair = {(t["left_cell"], t["right_cell"]): t for t in paired_tests}

    interpretation = {
        "phase2_submission_grade_ready": bool(
            summary.get("protocol_status", {}).get("observed_min_seed_count", 0) >= 5
            and fairness_audit["all_cells_match_total_timesteps"]
            and fairness_audit["all_cells_match_num_envs"]
            and fairness_audit["all_python_cells_beta_actor_aligned"]
        ),
        "jax_cpu_vs_gpu_no_detectable_primary_metric_difference": bool(
            tests_by_pair[("jax_gpu_ppo", "jax_cpu_ppo")]["p_value_two_sided"] is not None
            and tests_by_pair[("jax_gpu_ppo", "jax_cpu_ppo")]["p_value_two_sided"] >= 0.05
        ),
        "sbx_vs_jax_gpu_no_detectable_primary_metric_difference": bool(
            tests_by_pair[("jax_gpu_ppo", "sbx_cuda_ppo")]["p_value_two_sided"] is not None
            and tests_by_pair[("jax_gpu_ppo", "sbx_cuda_ppo")]["p_value_two_sided"] >= 0.05
        ),
        "sbx_significantly_outperforms_sb3_cuda": bool(
            tests_by_pair[("sbx_cuda_ppo", "sb3_cuda_ppo")]["p_value_two_sided"] is not None
            and tests_by_pair[("sbx_cuda_ppo", "sb3_cuda_ppo")]["p_value_two_sided"] < 0.05
        ),
        "sb3_cuda_significantly_outperforms_sb3_cpu": bool(
            tests_by_pair[("sb3_cuda_ppo", "sb3_cpu_ppo")]["p_value_two_sided"] is not None
            and tests_by_pair[("sb3_cuda_ppo", "sb3_cpu_ppo")]["p_value_two_sided"] < 0.05
        ),
        "strict_feasibility_has_numerical_tail_caveat_for_python_backends": bool(
            cell_payloads["sb3_cuda_ppo"]["numerical_tail_audit"]["strict_to_tolerant_gap"] > 0.25
            or cell_payloads["sbx_cuda_ppo"]["numerical_tail_audit"]["strict_to_tolerant_gap"] > 0.25
        ),
    }

    payload = {
        "reference_date": "2026-04-27",
        "summary_source": str(SUMMARY_PATH.relative_to(_PROJECT_ROOT)),
        "paper_ready": interpretation["phase2_submission_grade_ready"],
        "fairness_audit": fairness_audit,
        "phase2_cells": cell_payloads,
        "paired_tests": paired_tests,
        "interpretation": interpretation,
    }

    out = RESULTS_DIR / "phase2_paper_audit.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[phase2_paper_audit] wrote {out}")


if __name__ == "__main__":
    main()
