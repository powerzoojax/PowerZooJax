"""GenCos non-learning baselines (truthful, uniform_mid, max_markup)."""

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

BASELINE_NAMES = ("truthful", "uniform_mid", "max_markup")
DEFAULT_SPLITS = ("train", "iid", "demand_shift", "renewable_shock")


def run_single_baseline(
    task_dir: Path,
    algo: str,
    seed: int,
    split: str = "train",
) -> RunRecord:
    """Run one GenCos baseline on one split and seed."""
    from powerzoojax.tasks.gencos import GencosTask

    task_config = load_task_config(task_dir)
    max_steps = task_config.get("max_steps", 48)

    eval_cfg_path = task_dir / "configs" / f"eval_{split}.yaml"
    n_episodes = 10
    eval_config: dict = {}
    if eval_cfg_path.exists():
        eval_config = load_config(eval_cfg_path)
        n_episodes = eval_config.get("n_eval_episodes", n_episodes)

    task = GencosTask(
        n_segments=task_config.get("n_segments", 3),
        max_markup=task_config.get("max_markup", 2.0),
        max_steps=max_steps,
    )
    env = task.make_env(split)
    agent_fn = lambda p, k: task.baseline_rollout(env, p, k, algo)

    run_id = make_run_id("gencos", algo, split, seed)
    print(f"[GenCos baseline] {algo} split={split} seed={seed}")
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
        context="gencos/baseline",
        extra_env_meta=collect_dataset_provenance(
            task="gencos", task_config=task_config, split=split
        ),
        extra_labels={
            "record_kind": "baseline",
            "algo_family": "competitive_baseline",
            "backend_family": "jax",
            "split_status": "canonical",
        },
    )
    record = RunRecord(
        task="gencos", variant="gencos_case5", algo=algo, seed=seed, run_id=run_id,
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
    print(f"[GenCos baseline] saved to {path}")
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
                    print(f"[GenCos baseline] {algo} split={split} seed={seed} FAILED: {exc}")
    return records
