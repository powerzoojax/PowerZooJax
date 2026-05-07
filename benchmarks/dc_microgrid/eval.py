"""DC Microgrid evaluation script.

Loads a trained policy, runs n_eval_episodes episode windows
from the requested split, and saves a RunRecord with aggregated metrics.
"""

from __future__ import annotations

import time
from pathlib import Path

import jax

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
from benchmarks.common.runtime import build_train_cfg, make_policy_fn
from benchmarks.common.stats import aggregate_seeds
from benchmarks.dc_microgrid._reward_shaping import wrap_with_shaping


def eval_dcmicrogrid(task_dir: Path, run_id: str, split: str) -> RunRecord:
    """Evaluate a trained DC Microgrid run on the given split."""
    from powerzoojax.envs.microgrid import DataCenterMicrogridEnv
    from powerzoojax.tasks.dc_microgrid import DCMicrogridTask, rollout_dcmicrogrid

    original = load_run(run_id, task_dir)
    task_config = load_task_config_for_run(task_dir, original)
    eval_cfg_path = task_dir / "configs" / f"eval_{split}.yaml"
    n_episodes = 10
    eval_config: dict = {}
    if eval_cfg_path.exists():
        eval_config = load_config(eval_cfg_path)
        n_episodes = eval_config.get("n_eval_episodes", n_episodes)

    max_steps = task_config.get("max_steps", 288)
    algo = original.algo
    if algo not in ("ppo", "sac"):
        raise NotImplementedError(
            f"Algo {algo!r} not supported for DC Microgrid eval (use ppo or sac)."
        )

    train_config = load_train_config_for_run(task_dir, original, default_key="ppo")
    train_cfg = build_train_cfg(train_config, algo=algo)

    task = DCMicrogridTask(
        source=task_config.get("data_source", "google"),
        max_steps=max_steps,
        case_overrides=task_config.get("case_overrides") or {},
    )
    base_env = DataCenterMicrogridEnv()
    shaped_env = wrap_with_shaping(base_env, task_config)
    base_params = task.episode_params(split, 0, n_episodes, max_steps)

    oarts = original.artifacts
    if algo == "sac":
        from benchmarks.dc_microgrid.rejax_ckpt import load_sac_train_state

        rel = oarts.get("params_orbax")
        if not rel:
            raise FileNotFoundError(
                f"SAC run {run_id!r} has no params_orbax in artifacts: {oarts}"
            )
        train_state = load_sac_train_state(
            task_dir / "results" / rel, train_cfg, shaped_env, base_params
        )
    else:
        rel_p = oarts.get("params")
        if not rel_p:
            raise FileNotFoundError(f"PPO run {run_id!r} has no params in artifacts: {oarts}")
        train_state = load_pickle(task_dir / "results" / rel_p)

    policy_fn = make_policy_fn(algo, train_state, shaped_env, base_params, train_cfg)

    agent_fn = lambda p, k: rollout_dcmicrogrid(shaped_env, p, k, policy_fn)

    eval_run_id = make_run_id("dc_microgrid", algo, split, original.seed)
    print(f"[DC Microgrid eval] run_id={run_id} split={split} episodes={n_episodes}")
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
        context="dc_microgrid/eval",
        record_device=original.device,
        extra_env_meta=collect_dataset_provenance(
            task="dc_microgrid", task_config=task_config, split=split
        ),
        extra_labels={
            **dict(original.labels or {}),
            "record_kind": "eval",
            "source_run_id": run_id,
        },
    )
    record = RunRecord(
        task="dc_microgrid", variant="dc_microgrid", algo=algo, seed=original.seed,
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
    print(f"[DC Microgrid eval] saved to {path}")
    return record
