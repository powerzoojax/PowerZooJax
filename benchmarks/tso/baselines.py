"""TSO non-learning baselines (all_on, merit_order)."""

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
from benchmarks.tso.config_runtime import (
    get_baseline_set,
    get_eval_episodes,
    get_eval_gb_split,
    get_eval_splits,
    make_task_from_config,
)


def run_single_baseline(
    task_dir: Path,
    algo: str,
    seed: int,
    split: str = "train",
) -> RunRecord:
    """Run one TSO baseline on one split and seed."""
    task_config = load_task_config(task_dir)
    max_steps = int(task_config.get("max_steps", 48))

    eval_cfg_path = task_dir / "configs" / f"eval_{split}.yaml"
    eval_config: dict = {}
    if eval_cfg_path.exists():
        eval_config = load_config(eval_cfg_path)
    n_episodes = get_eval_episodes(task_config, eval_config or None)
    gb_split = get_eval_gb_split(split, eval_config or None)

    task = make_task_from_config(
        task_config,
        load_scale=float(eval_config.get("load_scale", 1.0)),
        line_rating_scale=float(eval_config.get("line_rating_scale", 1.0)),
    )
    env = task.make_env(split)
    # No outer jax.jit: rollout functions use lax.scan internally which self-compiles.
    # Wrapping a Python-loop rollout in jit causes XLA to unroll all steps, making
    # compilation prohibitively slow for 48-step TSO episodes.
    agent_fn = lambda p, k: task.baseline_rollout(env, p, k, algo)

    run_id = make_run_id("tso", algo, split, seed)
    print(f"[TSO baseline] {algo} split={split} seed={seed}")
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
        context="tso/baseline",
        extra_env_meta=collect_dataset_provenance(
            task="tso", task_config=task_config, split=split
        ),
        extra_labels={
            "record_kind": "baseline",
            "algo_family": "single_agent_baseline",
            "backend_family": "jax",
            "split_status": "canonical",
        },
    )
    record = RunRecord(
        task="tso", variant="tso_scuc", algo=algo, seed=seed, run_id=run_id,
        config_hash=config_hash({**task_config, "split": split}),
        status="completed", split=split,
        backend="jax_rejax",
        device=device,
        metrics=flat_metrics, walltime_s=walltime,
        notes=f"non-learning baseline episodes={n_episodes} | gb_split={gb_split}",
        env_info=env_info,
        labels=labels,
        artifacts=arts,
    )
    path = save_run(record, task_dir)
    print(f"[TSO baseline] saved to {path}")
    return record


def run_all_baselines(
    task_dir: Path,
    seeds: list[int] | None = None,
    splits: list[str] | None = None,
) -> list[RunRecord]:
    task_config = load_task_config(task_dir)
    seeds = seeds or [int(seed) for seed in task_config.get("seeds", [0, 1, 2, 3, 4])]
    splits = splits or list(get_eval_splits(task_config))
    baseline_names = get_baseline_set(task_config)
    records = []
    for split in splits:
        for algo in baseline_names:
            for seed in seeds:
                try:
                    records.append(run_single_baseline(task_dir, algo, seed, split))
                except Exception as exc:
                    print(f"[TSO baseline] {algo} split={split} seed={seed} FAILED: {exc}")
    return records
