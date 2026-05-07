"""GenCos evaluation script — trained IPPO policy rollout."""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp

from benchmarks.common.configs import (
    load_config,
    load_task_config_for_run,
    load_train_config_for_run,
)
from benchmarks.common.artifacts import save_eval_artifacts
from benchmarks.common.eval_loop import run_episodes
from benchmarks.common.io import (
    RunRecord,
    collect_dataset_provenance,
    collect_jax_run_contract,
    config_hash,
    load_pickle,
    load_run,
    make_run_id,
    save_run,
)
from benchmarks.common.stats import aggregate_seeds


def _gencos_eval_env_meta(split: str) -> dict[str, str]:
    if split in ("train",):
        window = "2025-04-01..2025-12-31"
        ood_axis = None
    elif split in ("iid", "demand_shift", "renewable_shock"):
        window = "2026-01-01..2026-03-31"
        ood_axis = None if split == "iid" else split
    else:
        raise ValueError(f"Unknown GenCos split {split!r}")
    meta = {
        "data_source": "gb_real",
        "benchmark_split": split,
        "profile_window": window,
    }
    if ood_axis is not None:
        meta["ood_axis"] = ood_axis
    return meta


def eval_gencos(task_dir: Path, run_id: str, split: str) -> RunRecord:
    """Evaluate a trained GenCos run on the given split."""
    from powerzoojax.rl.ippo import SharedActorCritic
    from powerzoojax.tasks.gencos import GencosTask, rollout_gencos

    original = load_run(run_id, task_dir)
    task_config = load_task_config_for_run(task_dir, original)
    eval_cfg_path = task_dir / "configs" / f"eval_{split}.yaml"
    n_episodes = 10
    eval_config: dict = {}
    if eval_cfg_path.exists():
        eval_config = load_config(eval_cfg_path)
        n_episodes = eval_config.get("n_eval_episodes", n_episodes)

    max_steps = task_config.get("max_steps", 48)

    train_config = load_train_config_for_run(task_dir, original, default_key="ippo")
    hidden_dims = tuple(train_config.get("hidden_dims", [128, 128]))
    net_params = load_pickle(task_dir / "results" / original.artifacts["params"])

    task = GencosTask(
        n_segments=task_config.get("n_segments", 3),
        max_markup=task_config.get("max_markup", 2.0),
        max_steps=max_steps,
    )
    env = task.make_env(split)
    agent_names = list(env.agent_names)
    action_dim = env.action_space().shape[0]
    network = SharedActorCritic(hidden_dims=hidden_dims, action_dim=action_dim)

    def policy_fn(obs_dict):
        return {n: jnp.clip(network.apply(net_params, obs_dict[n])[0], -1.0, 1.0)
                for n in agent_names}

    agent_fn = lambda p, k: rollout_gencos(env, k, policy_fn)  # params unused (embedded in env)

    eval_run_id = make_run_id("gencos", original.algo, split, original.seed)
    print(f"[GenCos eval] run_id={run_id} split={split} episodes={n_episodes}")
    t0 = time.time()
    all_metrics = run_episodes(task, split, agent_fn, n_episodes, max_steps, seed=original.seed)
    walltime = time.time() - t0

    artifacts_dir = task_dir / "results" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    arts = save_eval_artifacts(
        per_episode_metrics=all_metrics,
        run_id=eval_run_id, split=split,
        artifacts_dir=artifacts_dir,
    )
    flat_metrics = {k: v["mean"] for k, v in aggregate_seeds(all_metrics).items()}
    device, env_info, labels = collect_jax_run_contract(
        requested_device=original.device,
        context="gencos/eval",
        record_device=original.device,
        extra_env_meta={
            **collect_dataset_provenance(
                task="gencos", task_config=task_config, split=split
            ),
            **_gencos_eval_env_meta(split),
        },
        extra_labels={
            **dict(original.labels or {}),
            "record_kind": "eval",
            "source_run_id": run_id,
        },
    )

    record = RunRecord(
        task="gencos", variant="gencos_case5", algo=original.algo, seed=original.seed,
        run_id=eval_run_id,
        config_hash=config_hash({**eval_config, **task_config}),
        status="completed", split=split,
        backend=original.backend,
        device=device,
        framework_version=original.framework_version,
        metrics=flat_metrics, walltime_s=walltime,
        notes=f"eval of {run_id}",
        env_info=env_info,
        labels=labels,
        artifacts=arts,
    )
    path = save_run(record, task_dir)
    print(f"[GenCos eval] saved to {path}")
    return record
