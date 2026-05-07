"""TSO evaluation script.

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
from benchmarks.tso.config_runtime import (
    get_eval_episodes,
    get_eval_gb_split,
    make_task_from_config,
)
from benchmarks.tso.checkpoints import load_checkpoint_params


def eval_tso(
    task_dir: Path,
    run_id: str,
    split: str,
    checkpoint_index: int | None = None,
) -> RunRecord:
    """Evaluate a trained TSO run on the given split."""
    from powerzoojax.envs.grid.unit_commitment import UnitCommitmentEnv
    from powerzoojax.tasks.tso import rollout_tso
    from powerzoojax.rl.wrappers import SauteWrapper

    eval_config = load_config(task_dir / "configs" / f"eval_{split}.yaml")
    original = load_run(run_id, task_dir)
    task_config = load_task_config_for_run(task_dir, original)
    max_steps = int(task_config.get("max_steps", 48))
    n_episodes = get_eval_episodes(task_config, eval_config)
    gb_split = get_eval_gb_split(split, eval_config)

    algo = original.algo

    algo_key = {"ppo_lagrangian": "safe", "saute_ppo": "saute_ppo"}.get(algo, algo)
    train_cfg_dict = load_train_config_for_run(
        task_dir,
        original,
        algo_key_map={"ppo_lagrangian": "safe", "saute_ppo": "saute_ppo"},
        default_key=algo_key,
    )
    train_cfg = build_train_cfg(train_cfg_dict, algo=algo)

    task = make_task_from_config(
        task_config,
        load_scale=float(eval_config.get("load_scale", 1.0)),
        line_rating_scale=float(eval_config.get("line_rating_scale", 1.0)),
    )
    env = UnitCommitmentEnv()
    base_params = task.episode_params(split, 0, n_episodes, max_steps)
    constraint_spec = task.constraint_spec()

    oarts = original.artifacts
    if algo == "sac":
        from benchmarks.dc_microgrid.rejax_ckpt import load_sac_train_state

        rel = oarts.get("params_orbax")
        if not rel:
            raise FileNotFoundError(
                f"SAC run {run_id!r} has no params_orbax in artifacts: {oarts}"
            )
        train_state = load_sac_train_state(
            task_dir / "results" / rel, train_cfg, env, base_params
        )
    else:
        checkpoint_note = ""
        if checkpoint_index is None:
            rel_p = oarts.get("params")
            if not rel_p:
                raise FileNotFoundError(
                    f"Run {run_id!r} has no params in artifacts: {oarts}"
                )
            train_state = load_pickle(task_dir / "results" / rel_p)
        else:
            checkpoints_rel = oarts.get("checkpoints")
            if not checkpoints_rel:
                raise FileNotFoundError(
                    f"Run {run_id!r} has no checkpoints artifact: {oarts}"
                )
            train_state, checkpoint_spec = load_checkpoint_params(
                task_dir,
                checkpoints_rel,
                checkpoint_index,
            )
            checkpoint_note = (
                f" | checkpoint_index={int(checkpoint_spec['index'])}"
                f" | checkpoint_timesteps={int(checkpoint_spec['timesteps'])}"
            )
    policy_fn = make_policy_fn(
        algo, train_state, env, base_params, train_cfg,
        action_dim=2 * base_params.case.n_units,
        selected_names=constraint_spec.selected_names,
    )

    eval_run_id = make_run_id("tso", algo, split, original.seed)
    print(f"[TSO eval] run_id={run_id} split={split} episodes={n_episodes}")
    t0 = time.time()
    if algo == "saute_ppo":
        all_metrics: list[dict[str, float]] = []
        for ep in range(n_episodes):
            key = jax.random.PRNGKey(original.seed * 10_000 + ep)
            ref_key = jax.random.PRNGKey(original.seed * 10_000 + ep + 50_000)
            params = task.episode_params(
                split, ep, n_episodes, max_steps, strategy="uniform", seed=original.seed
            )
            saute_eval_kwargs = {
                "selected_names": constraint_spec.selected_names,
                "horizon": int(train_cfg.saute_horizon or max_steps),
                "unsafe_reward": float(train_cfg.saute_unsafe_reward),
                "use_reward_shaping": False,
            }
            if train_cfg.cost_thresholds:
                eval_wrapper = SauteWrapper(
                    env,
                    params,
                    cost_thresholds=tuple(float(x) for x in train_cfg.cost_thresholds),
                    **saute_eval_kwargs,
                )
            else:
                eval_wrapper = SauteWrapper(
                    env,
                    params,
                    cost_threshold=float(train_cfg.cost_threshold),
                    **saute_eval_kwargs,
                )
            agent_data = rollout_bound_wrapper(
                eval_wrapper,
                key,
                policy_fn,
                max_steps=int(params.max_steps),
                info_keys={
                    "gen_cost": "gen_cost",
                    "startup_cost": "startup_cost",
                    "no_load_cost": "no_load_cost",
                    "reserve_shortfall": "reserve_shortfall",
                    "cost_thermal_overload": "cost_thermal_overload",
                    "cost_sum": "cost_sum",
                    "commitment_switches": "commitment_switches",
                    "is_safe": "is_safe",
                    "opf_converged": "opf_converged",
                    "opf_iterations": "opf_iterations",
                    "opf_box_residual_mw": "opf_box_residual_mw",
                    "opf_line_residual_mw": "opf_line_residual_mw",
                    "opf_balance_residual_mw": "opf_balance_residual_mw",
                },
            )
            ref_data = task.baseline_rollout(env, params, ref_key, "no_control")
            all_metrics.append(task.compute_metrics(agent_data, ref_data))
    else:
        agent_fn = jax.jit(lambda p, k: rollout_tso(env, p, k, policy_fn))
        all_metrics = run_episodes(
            task, split, agent_fn, n_episodes, max_steps, seed=original.seed
        )
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
        context="tso/eval",
        record_device=original.device,
        extra_env_meta=collect_dataset_provenance(
            task="tso", task_config=task_config, split=split
        ),
        extra_labels={
            **dict(original.labels or {}),
            "record_kind": "eval",
            "source_run_id": run_id,
        },
    )
    record = RunRecord(
        task="tso", variant="tso_scuc", algo=algo, seed=original.seed,
        run_id=eval_run_id,
        config_hash=config_hash({**eval_config, **task_config}),
        status="completed", split=split,
        backend=original.backend,
        device=device,
        framework_version=original.framework_version,
        metrics=flat_metrics, walltime_s=walltime,
        notes=(
            f"eval of {run_id} | gb_split={gb_split}"
            + (checkpoint_note if algo != "sac" else "")
        ),
        env_info=env_info,
        labels=labels,
        artifacts=arts,
    )
    path = save_run(record, task_dir)
    print(f"[TSO eval] saved to {path}")
    return record
