"""DERs non-learning baselines (no_control, volt_droop)."""

from __future__ import annotations

import time
from pathlib import Path

from benchmarks.common.runtime import prefer_packaged_cuda_binaries

prefer_packaged_cuda_binaries()

import jax

from benchmarks.common.configs import load_config, load_task_config
from benchmarks.common.artifacts import save_eval_artifacts
from benchmarks.common.io import (
    RunRecord,
    collect_dataset_provenance,
    collect_jax_run_contract,
    config_hash,
    make_run_id,
    save_run,
)
from benchmarks.common.stats import aggregate_seeds

BASELINE_NAMES = ("no_control", "volt_droop")
DEFAULT_SPLITS = (
    "train",
    "iid",
    "voltage_tightening",
    "pv_penetration_shift",
    "load_stress",
)


def _run_baseline_episodes(
    *,
    task,
    split: str,
    algo: str,
    n_episodes: int,
    max_steps: int,
    seed: int,
) -> list[dict[str, float]]:
    per_episode: list[dict[str, float]] = []
    for ep in range(n_episodes):
        key = jax.random.PRNGKey(seed * 10_000 + ep)
        ref_key = jax.random.PRNGKey(seed * 10_000 + ep + 50_000)
        episode_start = task.episode_start(
            split,
            ep,
            n_episodes,
            strategy="uniform",
            seed=seed,
        )
        params = task.params_from_start(split, episode_start)
        agent_data = task.baseline_rollout(None, params, key, algo)
        ref_data = task.baseline_rollout(None, params, ref_key, "no_control")
        metrics = task.compute_metrics(agent_data, ref_data)
        metrics["episode_idx"] = float(ep)
        metrics["episode_start"] = float(episode_start)
        per_episode.append(metrics)
    return per_episode


def run_single_baseline(
    task_dir: Path,
    algo: str,
    seed: int,
    split: str = "train",
) -> RunRecord:
    """Run one DERs baseline on one split and seed."""
    from powerzoojax.tasks.ders import DERsTask

    task_config = load_task_config(task_dir)
    max_steps = task_config.get("max_steps", 48)

    from powerzoojax.case import load_case

    case = load_case(task_config.get("case", "case141"))

    eval_cfg_path = task_dir / "configs" / f"eval_{split}.yaml"
    n_episodes = 10
    if eval_cfg_path.exists():
        n_episodes = load_config(eval_cfg_path).get("n_eval_episodes", n_episodes)

    task = DERsTask(
        case=case,
        v_min=task_config["v_min"],
        v_max=task_config["v_max"],
        max_steps=max_steps,
    )

    run_id = make_run_id("ders", algo, split, seed)
    print(f"[DERs baseline] {algo} split={split} seed={seed}")
    t0 = time.time()
    all_metrics = _run_baseline_episodes(
        task=task,
        split=split,
        algo=algo,
        n_episodes=n_episodes,
        max_steps=max_steps,
        seed=seed,
    )
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
        context="ders/baseline",
        extra_env_meta=collect_dataset_provenance(
            task="ders", task_config=task_config, split=split
        ),
        extra_labels={
            "record_kind": "baseline",
            "algo_family": "cooperative_baseline",
            "backend_family": "jax",
            "split_status": "canonical",
        },
    )
    record = RunRecord(
        task="ders", variant="ders_12agent", algo=algo, seed=seed, run_id=run_id,
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
    print(f"[DERs baseline] saved to {path}")
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
                    print(f"[DERs baseline] {algo} split={split} seed={seed} FAILED: {exc}")
    return records
