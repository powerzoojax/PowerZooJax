"""DC Microgrid non-learning baselines (no_control, max_renewable, rule_based)."""

from __future__ import annotations

import time
from pathlib import Path

import jax

from benchmarks.common.configs import load_config, load_task_config
from benchmarks.common.artifacts import save_eval_artifacts
from benchmarks.common.eval_loop import run_episodes
from benchmarks.common.io import (
    RunRecord,
    collect_dataset_provenance,
    collect_jax_run_contract,
    config_hash,
    make_run_id,
    save_run,
)
from benchmarks.common.stats import aggregate_seeds
from benchmarks.dc_microgrid._reward_shaping import wrap_with_shaping

BASELINE_NAMES = ("no_control", "max_renewable", "rule_based")
DEFAULT_SPLITS = ("train", "iid", "cooling_stress", "renewable_drought")

# Step-index constants for 5-min/step 288-step (24 h) episodes.
_DC_SOLAR_PEAK_START = 72    # 06:00
_DC_SOLAR_PEAK_END = 144     # 12:00
_DC_NIGHT_START_2 = 240      # 20:00
_DC_NIGHT_END_1 = 72         # 06:00


def _no_control_action(step: int, obs) -> "np.ndarray":
    import numpy as np
    return np.zeros(5, dtype=np.float32)


def _max_renewable_action(step: int, obs) -> "np.ndarray":
    import numpy as np
    if _DC_SOLAR_PEAK_START <= step < _DC_SOLAR_PEAK_END:
        batt = -1.0
    elif step >= _DC_NIGHT_START_2 or step < _DC_NIGHT_END_1:
        batt = 1.0
    else:
        batt = 0.0
    return np.array([0.5, 0.5, 0.5, batt, 0.0], dtype=np.float32)


def _rule_based_action(step: int, obs, prev_deficit: float) -> "np.ndarray":
    import numpy as np
    if _DC_SOLAR_PEAK_START <= step < _DC_SOLAR_PEAK_END:
        train_sched, ft_sched, batt = 1.0, 0.5, 0.0
    elif step < _DC_SOLAR_PEAK_START:
        train_sched, ft_sched, batt = 0.0, 0.0, -1.0
    else:
        train_sched, ft_sched = 0.0, 0.0
        batt = 0.5 if step < _DC_NIGHT_START_2 else -1.0
    dg = 1.0 if prev_deficit > 0.05 else 0.0
    return np.array([train_sched, ft_sched, 0.0, batt, dg], dtype=np.float32)


def run_single_baseline(
    task_dir: Path,
    algo: str,
    seed: int,
    split: str = "train",
) -> RunRecord:
    """Run one DC Microgrid baseline on one split and seed."""
    from powerzoojax.tasks.dc_microgrid import DCMicrogridTask

    task_config = load_task_config(task_dir)
    max_steps = task_config.get("max_steps", 288)

    eval_cfg_path = task_dir / "configs" / f"eval_{split}.yaml"
    n_episodes = 10
    eval_config: dict = {}
    if eval_cfg_path.exists():
        eval_config = load_config(eval_cfg_path)
        n_episodes = eval_config.get("n_eval_episodes", n_episodes)

    task = DCMicrogridTask(
        source=task_config.get("data_source", "google"),
        max_steps=max_steps,
        case_overrides=task_config.get("case_overrides") or {},
    )
    env = wrap_with_shaping(task.make_env(split), task_config)
    agent_fn = lambda p, k: task.baseline_rollout(env, p, k, algo)

    run_id = make_run_id("dc_microgrid", algo, split, seed)
    print(f"[DC Microgrid baseline] {algo} split={split} seed={seed}")
    t0 = time.time()
    all_metrics = run_episodes(task, split, agent_fn, n_episodes, max_steps, seed=seed)
    walltime = time.time() - t0

    artifacts_dir = task_dir / "results" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    arts = save_eval_artifacts(
        per_episode_metrics=all_metrics,
        run_id=run_id, split=split,
        artifacts_dir=artifacts_dir,
    )
    flat_metrics = {k: v["mean"] for k, v in aggregate_seeds(all_metrics).items()}

    device, env_info, labels = collect_jax_run_contract(
        requested_device="gpu",
        context="dc_microgrid/baseline",
        extra_env_meta=collect_dataset_provenance(
            task="dc_microgrid", task_config=task_config, split=split
        ),
        extra_labels={
            "record_kind": "baseline",
            "algo_family": "single_agent_baseline",
            "backend_family": "jax",
            "split_status": "canonical",
        },
    )
    record = RunRecord(
        task="dc_microgrid", variant="dc_microgrid", algo=algo, seed=seed, run_id=run_id,
        config_hash=config_hash({**task_config, "split": split}),
        status="completed", split=split,
        backend="jax_rejax",
        device=device,
        metrics=flat_metrics, walltime_s=walltime,
        notes=f"non-learning baseline episodes={n_episodes}",
        env_info=env_info,
        labels=labels,
        artifacts=arts,
    )
    path = save_run(record, task_dir)
    print(f"[DC Microgrid baseline] saved to {path}")
    return record


def run_all_baselines(
    task_dir: Path,
    seeds: list[int] | None = None,
    splits: list[str] | None = None,
) -> list[RunRecord]:
    seeds = seeds or [0, 1, 2]
    splits = splits or list(DEFAULT_SPLITS)
    records = []
    for split in splits:
        for algo in BASELINE_NAMES:
            for seed in seeds:
                try:
                    records.append(run_single_baseline(task_dir, algo, seed, split))
                except Exception as exc:
                    print(
                        f"[DC Microgrid baseline] {algo} split={split} seed={seed} FAILED: {exc}"
                    )
    return records
