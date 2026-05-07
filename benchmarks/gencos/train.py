"""GenCos training script — homogeneous IPPO."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp

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
    make_steady_train_cfg,
    make_warmup_cfg,
    time_jax_train_with_warmup,
)


def train_gencos(
    task_dir: Path,
    algo: str = "ippo",
    seed: int = 0,
    config_path: str | None = None,
) -> RunRecord:
    """Train a GenCos IPPO agent and save the result as a RunRecord."""
    from powerzoojax.rl.ippo import SharedActorCritic, make_ippo_train
    from powerzoojax.tasks.gencos import GencosTask, rollout_gencos, compute_gencos_metrics

    config = load_train_config(task_dir, algo, config_path, default_key="ippo")
    task_config = load_task_config(task_dir)

    task = GencosTask(
        n_segments=task_config.get("n_segments", 3),
        max_markup=task_config.get("max_markup", 2.0),
        max_steps=task_config.get("max_steps", 48),
    )
    env = task.make_env("train")

    train_cfg = build_train_cfg(config)
    warmup_cfg = make_warmup_cfg(train_cfg)
    train_cfg = make_steady_train_cfg(train_cfg)
    hidden_dims = train_cfg.hidden_dims

    run_id = make_run_id("gencos", algo, "train", seed)
    print(f"[GenCos train] algo={algo} seed={seed} run_id={run_id}")

    def _run_train(*, config, key):
        return make_ippo_train(env, config)(key)

    result, walltime, compile_warmup = time_jax_train_with_warmup(
        _run_train,
        full_cfg=train_cfg,
        warmup_cfg=warmup_cfg,
        key=jax.random.PRNGKey(seed),
    )
    print(f"[GenCos train] compile_warmup_s={compile_warmup:.3f} walltime_s={walltime:.3f}")

    artifacts_dir = task_dir / "results" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    params_file = artifacts_dir / f"{run_id}_params.pkl"
    dump_pickle(result.params, params_file)

    artifacts = save_training_artifacts(
        result_metrics=result.metrics if isinstance(result.metrics, dict) else {},
        run_id=run_id,
        artifacts_dir=artifacts_dir,
        total_timesteps=int(config["total_timesteps"]),
        config_snapshot={"train_config": config, "task_config": task_config},
        extra_artifacts={"params": f"artifacts/{params_file.name}"},
    )

    market_metrics: dict = {}
    try:
        agent_names = list(env.agent_names)
        action_dim = env.action_space().shape[0]
        network = SharedActorCritic(hidden_dims=hidden_dims, action_dim=action_dim)
        net_params = result.params

        def policy_fn(obs_dict):
            return {n: jnp.clip(network.apply(net_params, obs_dict[n])[0], -1.0, 1.0)
                    for n in agent_names}

        rollout = rollout_gencos(env, jax.random.PRNGKey(seed * 1000 + 1), policy_fn)
        market_metrics = compute_gencos_metrics(rollout, agent_names)
    except Exception as exc:
        print(f"[GenCos train] post-train market metrics FAILED: {exc!r}")

    device, env_info, labels = collect_jax_run_contract(
        requested_device="gpu",
        context="gencos/train",
        extra_env_meta=collect_dataset_provenance(
            task="gencos", task_config=task_config, split="train"
        ),
        extra_labels={
            "record_kind": "train",
            "algo_family": "competitive_marl",
            "backend_family": "jax",
            "split_status": "canonical",
        },
    )
    record = RunRecord(
        task="gencos", variant="gencos_case5", algo=algo, seed=seed, run_id=run_id,
        config_hash=config_hash({**config, **task_config}),
        status="completed", split="train",
        backend="jax_rejax",
        device=device,
        metrics={**result.summary, **market_metrics},
        walltime_s=walltime,
        compile_warmup_s=compile_warmup,
        throughput_sps=config["total_timesteps"] / walltime if walltime > 0 else None,
        notes="ippo_shared data=real",
        env_info=env_info,
        labels=labels,
        artifacts=artifacts,
    )
    path = save_run(record, task_dir)
    print(f"[GenCos train] saved to {path}")
    return record
