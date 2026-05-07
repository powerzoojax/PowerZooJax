"""DSO non-learning baselines (no_control, tou, droop)."""

from __future__ import annotations

import time
from pathlib import Path

from benchmarks.common.runtime import prefer_packaged_cuda_binaries

prefer_packaged_cuda_binaries()

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

BASELINE_NAMES = ("no_control", "tou", "droop")
DEFAULT_SPLITS = ("iid",)


def _configured_eval_splits(task_config: dict) -> tuple[str, ...]:
    splits = tuple(str(s) for s in task_config.get("eval_splits", ()))
    if splits:
        return splits
    primary = task_config.get("primary_split", "iid")
    return (str(primary),)


def run_single_baseline(
    task_dir: Path,
    algo: str,
    seed: int,
    split: str = "train",
    task_config_path: str | None = None,
) -> RunRecord:
    """Run one DSO baseline on one split and seed."""
    from powerzoojax.tasks.dso import DSOTask, dso_task_kwargs_from_config

    task_config = load_task_config(task_dir, task_config_path)
    allowed_splits = _configured_eval_splits(task_config)
    if split not in allowed_splits:
        raise ValueError(
            f"DSO split {split!r} is not configured for formal evaluation. "
            f"Configured eval_splits={list(allowed_splits)}."
        )
    max_steps = task_config.get("max_steps", 48)

    eval_cfg_path = task_dir / "configs" / f"eval_{split}.yaml"
    n_episodes = 10
    if eval_cfg_path.exists():
        n_episodes = load_config(eval_cfg_path).get("n_eval_episodes", n_episodes)

    task = DSOTask(**dso_task_kwargs_from_config(task_config))
    env = task.make_env(split)
    agent_fn = jax.jit(lambda p, k: task.baseline_rollout(env, p, k, algo))

    run_id = make_run_id("dso", algo, split, seed)
    print(f"[DSO baseline] {algo} split={split} seed={seed}")
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
        context="dso/baseline",
        extra_env_meta=collect_dataset_provenance(
            task="dso", task_config=task_config, split=split
        ),
        extra_labels={
            "record_kind": "baseline",
            "algo_family": "single_agent_baseline",
            "backend_family": "jax",
            "split_status": "canonical",
        },
    )
    record = RunRecord(
        task="dso", variant="dso_nflex", algo=algo, seed=seed, run_id=run_id,
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
    print(f"[DSO baseline] saved to {path}")
    return record


def run_all_baselines(
    task_dir: Path,
    seeds: list[int] | None = None,
    splits: list[str] | None = None,
    task_config_path: str | None = None,
) -> list[RunRecord]:
    seeds = seeds or [0, 1, 2]
    task_config = load_task_config(task_dir, task_config_path)
    configured_splits = _configured_eval_splits(task_config)
    splits = splits or list(configured_splits)
    invalid = sorted(set(splits) - set(configured_splits))
    if invalid:
        raise ValueError(
            f"DSO baseline split(s) {invalid} are not configured. "
            f"Configured eval_splits={list(configured_splits)}."
        )
    records = []
    for split in splits:
        for algo in BASELINE_NAMES:
            for seed in seeds:
                try:
                    records.append(
                        run_single_baseline(
                            task_dir,
                            algo,
                            seed,
                            split,
                            task_config_path=task_config_path,
                        )
                    )
                except Exception as exc:
                    print(f"[DSO baseline] {algo} split={split} seed={seed} FAILED: {exc}")
    return records
