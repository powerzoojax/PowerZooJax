"""DC Microgrid training script."""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np

from benchmarks.common.configs import load_task_config, load_train_config
from benchmarks.common.artifacts import save_training_artifacts
from benchmarks.common.io import (
    RunRecord,
    collect_dataset_provenance,
    collect_jax_run_contract,
    config_hash,
    dump_pickle,
    make_run_id,
    save_run,
)
from benchmarks.common.runtime import (
    build_train_cfg,
    make_policy_fn,
    make_steady_train_cfg,
    make_warmup_cfg,
    time_jax_train_with_warmup,
)
from benchmarks.dc_microgrid._reward_shaping import wrap_with_shaping
from benchmarks.dc_microgrid.rejax_ckpt import save_sac_train_state


def train_dcmicrogrid(
    task_dir: Path,
    algo: str,
    seed: int = 0,
    config_path: str | None = None,
) -> RunRecord:
    """Train a DC Microgrid agent and save the result as a RunRecord."""
    from powerzoojax.envs.microgrid import DataCenterMicrogridEnv
    from powerzoojax.rl import train
    from powerzoojax.rl.wrappers import LogWrapper
    from powerzoojax.tasks.dc_microgrid import (
        DCMicrogridTask, rollout_dcmicrogrid, compute_dcmicrogrid_metrics,
    )

    config = load_train_config(task_dir, algo, config_path, allowed_algos=("ppo", "sac"))
    task_config = load_task_config(task_dir)
    max_steps = task_config.get("max_steps", 288)

    task = DCMicrogridTask(
        source=task_config.get("data_source", "google"),
        max_steps=max_steps,
        case_overrides=task_config.get("case_overrides") or {},
    )
    params = task.episode_params("train", 0, 1, max_steps, strategy="seeded", seed=seed)

    base_env = DataCenterMicrogridEnv()
    shaped_env = wrap_with_shaping(base_env, task_config)
    wrapped = LogWrapper(shaped_env, params)

    train_cfg = build_train_cfg(config, algo=algo)

    run_id = make_run_id("dc_microgrid", algo, "train", seed)
    print(f"[DC Microgrid train] algo={algo} seed={seed} run_id={run_id}")
    warmup_cfg = make_warmup_cfg(train_cfg)
    train_cfg = make_steady_train_cfg(train_cfg)
    result, walltime, compile_warmup = time_jax_train_with_warmup(
        train,
        full_cfg=train_cfg,
        warmup_cfg=warmup_cfg,
        preset_or_env=wrapped,
        seed=seed,
    )
    print(
        f"[DC Microgrid train] compile_warmup_s={compile_warmup:.3f} "
        f"walltime_s={walltime:.3f}"
    )

    artifacts_dir = task_dir / "results" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    params_pkl = artifacts_dir / f"{run_id}_params.pkl"
    params_orbax_dir = artifacts_dir / f"{run_id}_params_orbax"
    extra_artifact_params: dict[str, str] = {}
    if algo == "sac":
        save_sac_train_state(params_orbax_dir, result.params)
        extra_artifact_params["params_orbax"] = f"artifacts/{params_orbax_dir.name}"
    else:
        dump_pickle(result.params, params_pkl)
        extra_artifact_params["params"] = f"artifacts/{params_pkl.name}"

    metrics_dict = result.metrics if isinstance(result.metrics, dict) else {}
    eval_walltimes_s = None
    wt = metrics_dict.get("eval_wall_time_s")
    if wt is not None:
        w_arr = np.asarray(wt).flatten()
        if w_arr.size > 0:
            eval_walltimes_s = w_arr.tolist()

    artifacts = save_training_artifacts(
        result_metrics=metrics_dict,
        run_id=run_id,
        artifacts_dir=artifacts_dir,
        total_timesteps=int(config["total_timesteps"]),
        config_snapshot={"train_config": config, "task_config": task_config},
        extra_artifacts=extra_artifact_params,
        eval_walltimes_s=eval_walltimes_s,
    )

    dc_metrics: dict = {}
    try:
        policy_fn = make_policy_fn(algo, result.params, shaped_env, params, train_cfg)
        info_history = rollout_dcmicrogrid(
            shaped_env, params, jax.random.PRNGKey(seed * 1000 + 1), policy_fn,
        )
        dc_metrics = compute_dcmicrogrid_metrics(info_history)
    except Exception as exc:
        print(f"[DC Microgrid train] post-training rollout failed ({exc}), skipping")

    device, env_info, labels = collect_jax_run_contract(
        requested_device="gpu",
        context="dc_microgrid/train",
        extra_env_meta=collect_dataset_provenance(
            task="dc_microgrid", task_config=task_config, split="train"
        ),
        extra_labels={
            "record_kind": "train",
            "algo_family": "single_agent_rl",
            "backend_family": "jax",
            "split_status": "canonical",
        },
    )
    record = RunRecord(
        task="dc_microgrid", variant="dc_microgrid", algo=algo, seed=seed, run_id=run_id,
        config_hash=config_hash({**config, **task_config}),
        status="completed", split="train",
        backend="jax_rejax",
        device=device,
        metrics={**result.summary, **dc_metrics},
        walltime_s=walltime,
        compile_warmup_s=compile_warmup,
        throughput_sps=config["total_timesteps"] / walltime if walltime > 0 else None,
        notes="data_mode=real",
        env_info=env_info,
        labels=labels,
        artifacts=artifacts,
    )
    path = save_run(record, task_dir)
    print(f"[DC Microgrid train] saved to {path}")
    return record
