"""TSO training script."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

from benchmarks.common.runtime import prefer_packaged_cuda_binaries

prefer_packaged_cuda_binaries()

import jax
import jax.numpy as jnp
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
    rollout_bound_wrapper,
    time_jax_train_with_warmup,
)
from benchmarks.dc_microgrid.rejax_ckpt import save_sac_train_state
from benchmarks.tso.checkpoints import save_checkpoint_bundle
from benchmarks.tso.config_runtime import make_task_from_config, resolve_gb_windows

_ALGO_KEY_MAP = {"ppo_lagrangian": "safe", "saute_ppo": "saute_ppo"}

# TSO PPO uses num_envs=256 for GPU throughput. On CPU, that batch blows RAM and
# XLA compile time; cap parallel envs while keeping total_timesteps unchanged.
_DEFAULT_CPU_NUM_ENVS_CAP = 32


def _requested_jax_device_for_train() -> str:
    """Infer the intended TSO JAX train device without hiding CPU fallback."""
    explicit = os.environ.get("POWERZOOJAX_REQUESTED_DEVICE")
    if explicit:
        return explicit
    platform = os.environ.get("JAX_PLATFORM_NAME") or os.environ.get("JAX_PLATFORMS")
    if platform and str(platform).strip().lower().startswith("cpu"):
        return "cpu"
    return "gpu"


def train_tso(
    task_dir: Path,
    algo: str,
    seed: int = 0,
    config_path: str | None = None,
    *,
    extra_notes: str = "",
) -> RunRecord:
    """Train a TSO agent and save the result as a RunRecord."""
    from powerzoojax.envs.grid.unit_commitment import UnitCommitmentEnv
    from powerzoojax.rl import train
    from powerzoojax.rl.wrappers import LogWrapper, SafeRLWrapper, SauteWrapper
    from powerzoojax.tasks.tso import rollout_tso, compute_tso_metrics

    config = load_train_config(task_dir, algo, config_path, algo_key_map=_ALGO_KEY_MAP)
    cpu_envs_note = ""
    if jax.default_backend() == "cpu":
        cap = int(os.environ.get("TSO_CPU_NUM_ENVS", str(_DEFAULT_CPU_NUM_ENVS_CAP)))
        ne = int(config.get("num_envs", 256))
        if ne > cap:
            config = dict(config)
            config["num_envs"] = cap
            cpu_envs_note = f" | cpu_num_envs {ne}->{cap} (TSO_CPU_NUM_ENVS)"
            print(
                f"[TSO train] CPU backend: num_envs {ne} -> {cap} "
                f"(override cap with TSO_CPU_NUM_ENVS env var)"
            )
    task_config = load_task_config(task_dir)
    max_steps = int(task_config.get("max_steps", 48))
    train_window = resolve_gb_windows(task_config)["train"]

    task = make_task_from_config(task_config)
    constraint_spec = task.constraint_spec()
    params = task.training_params(max_steps=max_steps)

    env = UnitCommitmentEnv()
    _is_penalty = algo.startswith("ppo_penalty")
    if config.get("wrapper") == "safe" or algo == "ppo_lagrangian":
        wrapped = SafeRLWrapper(
            env,
            params,
            cost_thresholds=config.get("cost_thresholds", constraint_spec.thresholds),
            selected_names=constraint_spec.selected_names,
        )
    elif config.get("wrapper") == "saute" or algo == "saute_ppo":
        saute_cost_thresholds = config.get("cost_thresholds")
        saute_kwargs = {
            "selected_names": constraint_spec.selected_names,
            "horizon": int(config.get("saute_horizon", max_steps)),
            "unsafe_reward": float(config.get("saute_unsafe_reward", 0.0)),
            "use_reward_shaping": bool(config.get("saute_use_reward_shaping", True)),
        }
        if saute_cost_thresholds is not None:
            budgets = tuple(float(x) for x in saute_cost_thresholds)
            if any(x <= 0.0 for x in budgets):
                raise ValueError(
                    f"saute_ppo requires positive cost_thresholds, got {budgets!r}."
                )
            wrapped = SauteWrapper(
                env,
                params,
                cost_thresholds=budgets,
                **saute_kwargs,
            )
        else:
            saute_budget = float(config.get("saute_budget", 100.0))
            if saute_budget <= 0.0:
                raise ValueError(
                    f"saute_ppo requires a positive saute_budget, got {saute_budget}."
                )
            wrapped = SauteWrapper(
                env,
                params,
                cost_threshold=saute_budget,
                **saute_kwargs,
            )
    elif config.get("wrapper") == "penalty" or _is_penalty:
        from powerzoojax.rl.wrappers import PenaltyRewardWrapper
        penalty_lambda = float(config.get("penalty_lambda", 100.0))
        reward_scale = float(task_config.get("reward_scale", 1e-4))
        wrapped = LogWrapper(
            PenaltyRewardWrapper(env, penalty_lambda=penalty_lambda, reward_scale=reward_scale),
            params,
        )
    else:
        wrapped = LogWrapper(env, params)

    # For penalty variants, the underlying trainer is standard PPO.
    effective_algo = "ppo" if _is_penalty else algo
    train_cfg = build_train_cfg(config, algo=effective_algo)

    run_id = make_run_id("tso", algo, "train", seed)
    print(f"[TSO train] algo={algo} seed={seed} run_id={run_id}")
    warmup_cfg = make_warmup_cfg(train_cfg)
    train_cfg = make_steady_train_cfg(train_cfg)
    result, walltime, compile_warmup = time_jax_train_with_warmup(
        train,
        full_cfg=train_cfg,
        warmup_cfg=warmup_cfg,
        preset_or_env=wrapped,
        seed=seed,
    )
    print(f"[TSO train] compile_warmup_s={compile_warmup:.3f} walltime_s={walltime:.3f}")

    leaves = jax.tree_util.tree_leaves(result.params)
    finite_run = all(bool(jnp.all(jnp.isfinite(jnp.asarray(x)))) for x in leaves)

    artifacts_dir = task_dir / "results" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    extra_artifacts: dict[str, str] = {}
    if finite_run:
        if effective_algo == "sac":
            params_orbax_dir = artifacts_dir / f"{run_id}_params_orbax"
            save_sac_train_state(params_orbax_dir, result.params)
            extra_artifacts["params_orbax"] = f"artifacts/{params_orbax_dir.name}"
        else:
            params_file = artifacts_dir / f"{run_id}_params.pkl"
            dump_pickle(result.params, params_file)
            extra_artifacts["params"] = f"artifacts/{params_file.name}"
        if getattr(result, "checkpoints", None):
            extra_artifacts["checkpoints"] = save_checkpoint_bundle(
                run_id=run_id,
                checkpoints=[
                    (int(steps_done), ckpt_params)
                    for steps_done, ckpt_params in result.checkpoints
                ],
                artifacts_dir=artifacts_dir,
            )
    else:
        print(f"[TSO train] WARN: params contain NaN/Inf; skipping save for {run_id}")

    cfg_snapshot = {
        "task_config": task_config,
        "train_config_raw": config,
        "train_config_resolved": dataclasses.asdict(train_cfg),
    }
    result_metrics = dict(result.metrics) if isinstance(result.metrics, dict) else {}
    reward_scale = float(task_config.get("reward_scale", 1e-4))
    monitor_eval_episodes = max(1, int(config.get("eval_episodes", 1)))

    # For PPO-Lagrangian: compute eval returns at each checkpoint so the
    # learning curve uses the same episodic-eval metric as PPO's eval_returns.
    # result.checkpoints is [(steps_done, params), ...] — one entry per segment.
    if finite_run and algo == "ppo_lagrangian" and getattr(result, "checkpoints", None):
        from powerzoojax.rl.cmdp import SafeActorCritic

        ckpt_eval_returns: list[float] = []
        ckpt_eval_costs: list[float] = []
        ckpt_shortfall_rates: list[float] = []
        ckpt_thermal_rates: list[float] = []
        ckpt_mean_shortfalls: list[float] = []
        ckpt_mean_thermals: list[float] = []
        network = SafeActorCritic(
            action_dim=2 * params.case.n_units,
            n_constraints=len(constraint_spec.selected_names),
            hidden_dim=train_cfg.hidden_dims[0] if train_cfg.hidden_dims else 256,
        )

        @jax.jit
        def checkpoint_eval_metrics(net_params, base_key):
            eval_keys = jax.random.split(base_key, monitor_eval_episodes)

            def single_episode(eval_key):
                def policy_fn(obs, state, key):
                    del state, key
                    mean, _, _, _ = network.apply(net_params, obs)
                    return jnp.clip(mean, -1.0, 1.0)

                rollout = rollout_tso(env, params, eval_key, policy_fn)
                total_cost = (
                    jnp.sum(rollout["gen_cost"])
                    + jnp.sum(rollout["startup_cost"])
                    + jnp.sum(rollout["no_load_cost"])
                )
                total_reserve_shortfall = jnp.sum(rollout["reserve_shortfall"])
                total_thermal_overload = jnp.sum(rollout["cost_thermal_overload"])
                reserve_shortfall_rate = jnp.mean(
                    rollout["reserve_shortfall"] > jnp.float32(1e-6)
                )
                thermal_violation_rate = jnp.mean(
                    rollout["cost_thermal_overload"] > jnp.float32(1e-6)
                )
                return (
                    total_cost,
                    total_reserve_shortfall,
                    total_thermal_overload,
                    reserve_shortfall_rate,
                    thermal_violation_rate,
                )

            (
                costs,
                reserve_shortfalls,
                thermal_overloads,
                reserve_rates,
                thermal_rates,
            ) = jax.vmap(single_episode)(eval_keys)
            return (
                jnp.mean(costs),
                jnp.mean(reserve_rates),
                jnp.mean(thermal_rates),
                jnp.mean(reserve_shortfalls),
                jnp.mean(thermal_overloads),
            )

        for ckpt_idx, (_, ckpt_params) in enumerate(result.checkpoints):
            try:
                (
                    mean_cost,
                    shortfall_rate,
                    thermal_rate,
                    mean_shortfall,
                    mean_thermal,
                ) = checkpoint_eval_metrics(
                        ckpt_params,
                        jax.random.PRNGKey(seed * 100000 + ckpt_idx),
                )
                mean_cost = float(mean_cost)
                shortfall_rate = float(shortfall_rate)
                thermal_rate = float(thermal_rate)
                mean_shortfall = float(mean_shortfall)
                mean_thermal = float(mean_thermal)
                ep_return = -mean_cost * reward_scale
            except Exception:
                mean_cost = float("nan")
                ep_return = float("nan")
                shortfall_rate = float("nan")
                thermal_rate = float("nan")
                mean_shortfall = float("nan")
                mean_thermal = float("nan")
            ckpt_eval_costs.append(mean_cost)
            ckpt_eval_returns.append(ep_return)
            ckpt_shortfall_rates.append(shortfall_rate)
            ckpt_thermal_rates.append(thermal_rate)
            ckpt_mean_shortfalls.append(mean_shortfall)
            ckpt_mean_thermals.append(mean_thermal)
        result_metrics["eval_timesteps"] = np.asarray(
            [steps_done for steps_done, _ in result.checkpoints], dtype=np.float32
        )
        result_metrics["eval_returns"] = np.array(ckpt_eval_returns, dtype=np.float32)
        result_metrics["eval_total_operating_cost"] = np.array(
            ckpt_eval_costs, dtype=np.float32
        )
        result_metrics["eval_reserve_shortfall_rate"] = np.array(
            ckpt_shortfall_rates, dtype=np.float32
        )
        result_metrics["eval_thermal_violation_rate"] = np.array(
            ckpt_thermal_rates, dtype=np.float32
        )
        result_metrics["eval_total_reserve_shortfall"] = np.array(
            ckpt_mean_shortfalls, dtype=np.float32
        )
        result_metrics["eval_total_thermal_overload"] = np.array(
            ckpt_mean_thermals, dtype=np.float32
        )

    if "eval_returns" in result_metrics and "eval_total_operating_cost" not in result_metrics:
        result_metrics["eval_total_operating_cost"] = (
            -np.asarray(result_metrics["eval_returns"], dtype=np.float32) / reward_scale
        )

    eval_walltimes_s = None
    if "eval_wall_time_s" in result_metrics:
        eval_walltimes_s = np.asarray(
            result_metrics["eval_wall_time_s"], dtype=np.float32
        )
    elif "checkpoint_walltime_s" in result_metrics:
        eval_walltimes_s = np.asarray(
            result_metrics["checkpoint_walltime_s"], dtype=np.float32
        )

    artifacts = save_training_artifacts(
        result_metrics=result_metrics,
        run_id=run_id,
        artifacts_dir=artifacts_dir,
        total_timesteps=int(config["total_timesteps"]),
        config_snapshot=cfg_snapshot,
        extra_artifacts=extra_artifacts,
        eval_walltimes_s=eval_walltimes_s,
        eval_curve_source="eval_returns",
    )

    tso_metrics: dict = {}
    if finite_run:
        try:
            policy_algo = "ppo" if _is_penalty else algo
            policy_fn = make_policy_fn(
                policy_algo, result.params, env, params, train_cfg,
                action_dim=2 * params.case.n_units,
                selected_names=constraint_spec.selected_names,
            )
            if policy_algo == "saute_ppo":
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
                rollout = rollout_bound_wrapper(
                    eval_wrapper,
                    jax.random.PRNGKey(seed * 1000 + 1),
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
                    },
                )
            else:
                rollout = rollout_tso(
                    env, params, jax.random.PRNGKey(seed * 1000 + 1), policy_fn
                )
            tso_metrics = compute_tso_metrics(rollout)
        except Exception as exc:
            print(f"[TSO train] post-training rollout failed ({exc}), skipping")

    device, env_info, labels = collect_jax_run_contract(
        requested_device=_requested_jax_device_for_train(),
        context="tso/train",
        extra_env_meta=collect_dataset_provenance(
            task="tso", task_config=task_config, split="train"
        ),
        extra_labels={
            "record_kind": "train",
            "algo_family": "single_agent_safe_rl",
            "backend_family": "jax",
            "split_status": "canonical",
        },
    )
    record = RunRecord(
        task="tso", variant="tso_scuc", algo=algo, seed=seed, run_id=run_id,
        config_hash=config_hash({**config, **task_config}),
        status="completed" if finite_run else "failed",
        split="train",
        backend="jax_rejax",
        device=device,
        metrics={**result.summary, **tso_metrics},
        walltime_s=walltime,
        compile_warmup_s=compile_warmup,
        throughput_sps=config["total_timesteps"] / walltime if walltime > 0 else None,
        notes=(
            f"data_mode=real | gb_split=train | gb_window={train_window[0]}..{train_window[1]}"
            + " | train_reset_sampling=uniform_window"
            + (
                f" | saute_cost_thresholds={tuple(config['cost_thresholds'])}"
                if algo == "saute_ppo" and "cost_thresholds" in config
                else (
                    f" | saute_budget={config['saute_budget']}"
                    if algo == "saute_ppo" and "saute_budget" in config
                    else ""
                )
            )
            + cpu_envs_note
            + (f" | penalty_lambda={config['penalty_lambda']}" if _is_penalty and "penalty_lambda" in config else "")
            + ("" if finite_run else " | params_nan=true")
            + (f" | {extra_notes}" if extra_notes else "")
        ),
        env_info=env_info,
        labels=labels,
        artifacts=artifacts,
    )
    path = save_run(record, task_dir)
    print(f"[TSO train] saved to {path}")
    return record
