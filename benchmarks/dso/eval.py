"""DSO evaluation script.

Loads a trained policy, runs n_eval_episodes evenly-spaced episode windows
from the requested split, and saves a RunRecord with aggregated metrics.
"""

from __future__ import annotations

import time
from pathlib import Path

from benchmarks.common.runtime import prefer_packaged_cuda_binaries

prefer_packaged_cuda_binaries()

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
from benchmarks.common.runtime import (
    build_train_cfg,
    make_policy_fn,
    rollout_bound_wrapper,
)
from benchmarks.common.stats import aggregate_seeds


def eval_dso(
    task_dir: Path,
    run_id: str,
    split: str,
    task_config_path: str | None = None,
) -> RunRecord:
    """Evaluate a trained DSO run on the given split."""
    from powerzoojax.envs.grid.dist import DistGridEnv
    from powerzoojax.tasks.dso import DSOTask, dso_task_kwargs_from_config, rollout_dso
    from powerzoojax.rl.wrappers import SauteWrapper

    eval_config = load_config(task_dir / "configs" / f"eval_{split}.yaml")
    original = load_run(run_id, task_dir)
    algo = original.algo

    task_config = load_task_config_for_run(task_dir, original, task_config_path)
    configured_splits = tuple(str(s) for s in task_config.get("eval_splits", ()))
    if not configured_splits:
        configured_splits = (str(task_config.get("primary_split", "iid")),)
    if split not in configured_splits:
        raise ValueError(
            f"DSO split {split!r} is not configured for formal evaluation. "
            f"Configured eval_splits={list(configured_splits)}."
        )

    max_steps = task_config.get("max_steps", 48)
    n_episodes = eval_config.get("n_eval_episodes", 50)

    algo_key = {"ppo_lagrangian": "safe", "saute_ppo": "saute_ppo"}.get(algo, algo)
    train_cfg_dict = load_train_config_for_run(
        task_dir,
        original,
        algo_key_map={"ppo_lagrangian": "safe", "saute_ppo": "saute_ppo"},
        default_key=algo_key,
    )
    train_cfg = build_train_cfg(train_cfg_dict, algo=algo)

    task = DSOTask(**dso_task_kwargs_from_config(task_config))
    env = DistGridEnv()
    constraint_spec = task.constraint_spec()
    base_params = task.episode_params(split, 0, n_episodes, max_steps)

    if algo == "sac":
        from benchmarks.dso.rejax_ckpt import load_sac_train_state

        rel = original.artifacts.get("params_flax")
        if not rel:
            raise FileNotFoundError(
                f"SAC run {run_id!r} has no params_flax in artifacts: {original.artifacts}"
            )
        train_state = load_sac_train_state(
            task_dir / "results" / rel, train_cfg, env, base_params
        )
    else:
        rel = original.artifacts.get("params")
        if not rel:
            raise FileNotFoundError(
                f"Run {run_id!r} has no params in artifacts: {original.artifacts}"
            )
        train_state = load_pickle(task_dir / "results" / rel)
    policy_fn = make_policy_fn(
        algo,
        train_state,
        env,
        base_params,
        train_cfg,
        selected_names=constraint_spec.selected_names,
    )

    eval_run_id = make_run_id("dso", algo, split, original.seed)
    print(f"[DSO eval] run_id={run_id} split={split} episodes={n_episodes}")
    t0 = time.time()
    if algo == "saute_ppo":
        all_metrics: list[dict[str, float]] = []
        for ep in range(n_episodes):
            key = jax.random.PRNGKey(original.seed * 10_000 + ep)
            ref_key = jax.random.PRNGKey(original.seed * 10_000 + ep + 50_000)
            params = task.episode_params(
                split, ep, n_episodes, max_steps, strategy="uniform", seed=original.seed
            )
            eval_wrapper = SauteWrapper(
                env,
                params,
                cost_threshold=float(train_cfg.cost_threshold),
                selected_names=constraint_spec.selected_names,
                horizon=int(train_cfg.saute_horizon or max_steps),
                unsafe_reward=float(train_cfg.saute_unsafe_reward),
                use_reward_shaping=False,
            )
            agent_data = rollout_bound_wrapper(
                eval_wrapper,
                key,
                policy_fn,
                max_steps=int(getattr(params, "max_steps", max_steps)),
                info_keys={
                    "losses": "p_loss_MW",
                    "violations": "n_violations",
                    "voltage_violations": "cost_voltage_violation",
                    "thermal_violations": "cost_thermal_overload",
                    "curtailed": "resource_curtailed_mw",
                    "shifted": "resource_shift_out_mw",
                    "shift_in": "resource_shift_in_mw",
                },
            )
            ref_data = task.baseline_rollout(env, params, ref_key, "no_control")
            all_metrics.append(task.compute_metrics(agent_data, ref_data))
    else:
        agent_fn = jax.jit(lambda p, k: rollout_dso(env, p, k, policy_fn))
        all_metrics = run_episodes(
            task, split, agent_fn, n_episodes, max_steps, seed=original.seed
        )
    walltime = time.time() - t0

    artifacts_dir = task_dir / "results" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    arts = save_eval_artifacts(
        per_episode_metrics=all_metrics,
        run_id=eval_run_id,
        split=split,
        artifacts_dir=artifacts_dir,
    )
    flat_metrics = {k: v["mean"] for k, v in aggregate_seeds(all_metrics).items()}

    device, env_info, labels = collect_jax_run_contract(
        requested_device=original.device,
        context="dso/eval",
        record_device=original.device,
        extra_env_meta=collect_dataset_provenance(
            task="dso", task_config=task_config, split=split
        ),
        extra_labels={
            **dict(original.labels or {}),
            "record_kind": "eval",
            "source_run_id": run_id,
        },
    )
    record = RunRecord(
        task="dso", variant="dso_nflex", algo=algo, seed=original.seed,
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
    print(f"[DSO eval] saved to {path}")
    return record
