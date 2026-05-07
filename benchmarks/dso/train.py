"""DSO training script."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import chex
from benchmarks.common.runtime import prefer_packaged_cuda_binaries
from flax import struct

prefer_packaged_cuda_binaries()

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
    make_policy_fn,
    make_warmup_cfg,
    rollout_bound_wrapper,
    time_jax_train_with_warmup,
)
from benchmarks.dso.rejax_ckpt import save_sac_train_state
from powerzoojax.rl.wrappers import SafeRLWrapper, _select_costs

_ALGO_KEY_MAP = {"ppo_lagrangian": "safe", "saute_ppo": "saute_ppo"}


@struct.dataclass
class _SampledSafeRLState:
    """Safe-RL wrapper state with a sampled-params index."""

    env_state: Any
    episode_returns: chex.Array
    episode_lengths: chex.Array
    returned_episode_returns: chex.Array
    returned_episode_lengths: chex.Array
    episode_costs: chex.Array
    returned_episode_costs: chex.Array
    params_idx: chex.Array


class _SampledParamsSafeRLWrapper(SafeRLWrapper):
    """SafeRLWrapper variant that samples one params bank entry per reset.

    This keeps the 48-step episode contract intact while exposing the trainer
    to multiple fixed train windows instead of a single frozen one.
    """

    def __init__(
        self,
        env,
        params_bank,
        *,
        cost_thresholds,
        selected_names,
    ):
        if not params_bank:
            raise ValueError("params_bank must be non-empty")
        self._params_bank = tuple(params_bank)
        self._n_params = len(self._params_bank)
        super().__init__(
            env,
            self._params_bank[0],
            cost_thresholds=cost_thresholds,
            selected_names=selected_names,
        )

    def _params_for_idx(self, idx: chex.Array):
        branches = tuple((lambda _operand, p=p: p) for p in self._params_bank)
        return jax.lax.switch(idx, branches, 0)

    @partial(jax.jit, static_argnums=(0,))
    def _sample_reset(self, key: chex.PRNGKey):
        sample_key, reset_key = jax.random.split(key)
        idx = jax.random.randint(sample_key, (), 0, self._n_params, dtype=jnp.int32)
        params = self._params_for_idx(idx)
        obs, env_state = self._env.reset(reset_key, params)
        return obs, env_state, idx

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey):
        obs, env_state, idx = self._sample_reset(key)
        zeros = jnp.zeros((self.num_constraints,), dtype=jnp.float32)
        state = _SampledSafeRLState(
            env_state=env_state,
            episode_returns=jnp.float32(0.0),
            episode_lengths=jnp.int32(0),
            returned_episode_returns=jnp.float32(0.0),
            returned_episode_lengths=jnp.int32(0),
            episode_costs=zeros,
            returned_episode_costs=zeros,
            params_idx=idx,
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key: chex.PRNGKey, state: _SampledSafeRLState, action: chex.Array):
        step_key, sample_key, reset_key = jax.random.split(key, 3)
        params = self._params_for_idx(state.params_idx)
        obs, env_state, reward, costs_all, done, info = self._env.step(
            step_key, state.env_state, action, params
        )

        next_idx = jax.random.randint(
            sample_key, (), 0, self._n_params, dtype=jnp.int32
        )
        next_params = self._params_for_idx(next_idx)
        reset_obs, reset_env_state = self._env.reset(reset_key, next_params)

        obs = jax.lax.stop_gradient(jnp.where(done, reset_obs, obs))
        env_state = jax.lax.stop_gradient(
            jax.tree_util.tree_map(
                lambda cur, rst: jnp.where(done, rst, cur),
                env_state,
                reset_env_state,
            )
        )
        costs_all = jax.lax.stop_gradient(costs_all)

        selected_costs = _select_costs(costs_all, self._selected_indices)
        cost_sum = jnp.sum(selected_costs)

        new_returns = state.episode_returns + reward
        new_lengths = state.episode_lengths + 1
        new_costs = state.episode_costs + selected_costs
        zero_costs = jnp.zeros_like(new_costs)

        new_state = _SampledSafeRLState(
            env_state=env_state,
            episode_returns=new_returns * (1 - done),
            episode_lengths=new_lengths * (1 - done.astype(jnp.int32)),
            returned_episode_returns=jnp.where(
                done, new_returns, state.returned_episode_returns
            ),
            returned_episode_lengths=jnp.where(
                done, new_lengths, state.returned_episode_lengths
            ),
            episode_costs=jnp.where(done, zero_costs, new_costs),
            returned_episode_costs=jnp.where(
                done,
                new_costs,
                state.returned_episode_costs,
            ),
            params_idx=jnp.where(done, next_idx, state.params_idx),
        )

        info = {
            **info,
            "constraint_costs_all": costs_all,
            "constraint_costs": selected_costs,
            "cost_sum": cost_sum,
            "cost": cost_sum,
            "returned_episode_returns": new_state.returned_episode_returns,
            "returned_episode_lengths": new_state.returned_episode_lengths,
            "returned_episode": done,
            "returned_episode_costs": new_state.returned_episode_costs,
            "returned_episode_cost_sum": jnp.sum(new_state.returned_episode_costs),
        }
        return obs, new_state, reward, selected_costs, done, info


def train_dso(
    task_dir: Path,
    algo: str,
    seed: int = 0,
    config_path: str | None = None,
    task_config_path: str | None = None,
) -> RunRecord:
    """Train a DSO agent and save the result as a RunRecord."""
    from powerzoojax.envs.grid.dist import DistGridEnv
    from powerzoojax.rl import train
    from powerzoojax.rl.wrappers import LogWrapper, SafeRLWrapper, SauteWrapper
    from powerzoojax.tasks.dso import (
        concat_dso_feeder_windows,
        DSOTask,
        compute_dso_metrics,
        dso_no_control_rollout,
        load_dso_feeder_shapes,
        make_dso_params,
        make_dso_params_from_split,
        dso_task_kwargs_from_config,
        rollout_dso,
    )

    config = load_train_config(task_dir, algo, config_path, algo_key_map=_ALGO_KEY_MAP)
    task_config = load_task_config(task_dir, task_config_path)
    max_steps = task_config.get("max_steps", 48)
    task_kwargs = dso_task_kwargs_from_config(task_config)

    task = DSOTask(**task_kwargs)
    constraint_spec = task.constraint_spec()
    train_episode_start = task_config.get("train_episode_start")
    train_window_starts = task_config.get("train_window_starts")
    train_window_sampling = str(task_config.get("train_window_sampling", "concat"))
    sampled_params_bank = None
    if train_window_starts:
        train_window_role = str(task_config.get("train_window_role", "train"))
        train_window_len = int(task_config.get("train_window_len", max_steps))
        train_window_starts = [int(x) for x in train_window_starts]
        feeder_shapes = load_dso_feeder_shapes(role=train_window_role)
        if train_window_sampling == "reset_bank":
            sampled_params_bank = [
                make_dso_params(
                    feeder_shapes=feeder_shapes,
                    max_steps=train_window_len,
                    episode_start=start,
                    **task_kwargs,
                )
                for start in train_window_starts
            ]
            params = sampled_params_bank[0]
        else:
            concat_shapes = concat_dso_feeder_windows(
                feeder_shapes,
                train_window_starts,
                window_len=train_window_len,
            )
            train_max_steps = int(
                task_config.get(
                    "train_max_steps",
                    len(train_window_starts) * train_window_len,
                )
            )
            params = make_dso_params(
                feeder_shapes=concat_shapes,
                max_steps=train_max_steps,
                episode_start=0,
                **task_kwargs,
            )
    elif train_episode_start is not None:
        params = make_dso_params_from_split(
            role="train",
            max_steps=max_steps,
            episode_start=int(train_episode_start),
            **task_kwargs,
        )
    else:
        params = task.episode_params("train", 0, 1, max_steps, strategy="seeded", seed=seed)

    env = DistGridEnv()
    if config.get("wrapper") == "penalty":
        raise ValueError(
            "DSO penalty PPO ablations are deprecated for the canonical "
            "benchmark. Use ppo, sac, saute_ppo, or ppo_lagrangian."
        )
    if config.get("wrapper") == "safe" or algo == "ppo_lagrangian":
        if sampled_params_bank is not None:
            wrapped = _SampledParamsSafeRLWrapper(
                env,
                sampled_params_bank,
                cost_thresholds=config.get(
                    "cost_thresholds", constraint_spec.thresholds
                ),
                selected_names=constraint_spec.selected_names,
            )
        else:
            wrapped = SafeRLWrapper(
                env,
                params,
                cost_thresholds=config.get(
                    "cost_thresholds", constraint_spec.thresholds
                ),
                selected_names=constraint_spec.selected_names,
            )
    elif config.get("wrapper") == "saute" or algo == "saute_ppo":
        saute_budget = float(config.get("saute_budget", 192.0))
        if saute_budget <= 0.0:
            raise ValueError(
                f"saute_ppo requires a positive saute_budget, got {saute_budget}."
            )
        wrapped = SauteWrapper(
            env,
            params,
            cost_threshold=saute_budget,
            selected_names=constraint_spec.selected_names,
            horizon=int(config.get("saute_horizon", max_steps)),
            unsafe_reward=float(config.get("saute_unsafe_reward", 0.0)),
            use_reward_shaping=bool(config.get("saute_use_reward_shaping", True)),
        )
    else:
        wrapped = LogWrapper(env, params)

    effective_algo = algo
    train_cfg = build_train_cfg(config, algo=effective_algo)

    run_id = make_run_id("dso", algo, "train", seed)
    print(f"[DSO train] algo={algo} seed={seed} run_id={run_id}")
    warmup_cfg = make_warmup_cfg(train_cfg)
    train_cfg = make_steady_train_cfg(train_cfg)
    result, walltime, compile_warmup = time_jax_train_with_warmup(
        train,
        full_cfg=train_cfg,
        warmup_cfg=warmup_cfg,
        preset_or_env=wrapped,
        seed=seed,
    )
    print(f"[DSO train] compile_warmup_s={compile_warmup:.3f} walltime_s={walltime:.3f}")

    artifacts_dir = task_dir / "results" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    extra_artifacts: dict[str, str] = {}
    if effective_algo == "sac":
        params_flax_file = artifacts_dir / f"{run_id}_params_flax.msgpack"
        save_sac_train_state(params_flax_file, result.params)
        extra_artifacts["params_flax"] = f"artifacts/{params_flax_file.name}"
    else:
        params_file = artifacts_dir / f"{run_id}_params.pkl"
        dump_pickle(result.params, params_file)
        extra_artifacts["params"] = f"artifacts/{params_file.name}"

    artifacts = save_training_artifacts(
        result_metrics=result.metrics if isinstance(result.metrics, dict) else {},
        run_id=run_id,
        artifacts_dir=artifacts_dir,
        total_timesteps=int(config["total_timesteps"]),
        config_snapshot={"train_config": config, "task_config": task_config},
        extra_artifacts=extra_artifacts,
        eval_walltimes_s=(
            result.metrics.get("eval_wall_time_s")
            if isinstance(result.metrics, dict)
            else None
        ),
        eval_curve_source=None,
    )

    dso_metrics: dict = {}
    try:
        policy_fn = make_policy_fn(
            effective_algo,
            result.params,
            env,
            params,
            train_cfg,
            selected_names=constraint_spec.selected_names,
        )
        if effective_algo == "saute_ppo":
            eval_wrapper = SauteWrapper(
                env,
                params,
                cost_threshold=float(train_cfg.cost_threshold),
                selected_names=constraint_spec.selected_names,
                horizon=int(train_cfg.saute_horizon or max_steps),
                unsafe_reward=float(train_cfg.saute_unsafe_reward),
                use_reward_shaping=False,
            )
            rollout = rollout_bound_wrapper(
                eval_wrapper,
                jax.random.PRNGKey(seed * 1000 + 1),
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
        else:
            rollout = rollout_dso(
                env, params, jax.random.PRNGKey(seed * 1000 + 1), policy_fn
            )
        no_ctrl = dso_no_control_rollout(env, params, jax.random.PRNGKey(seed * 1000 + 2))
        dso_metrics = compute_dso_metrics(rollout, no_ctrl)
    except Exception as exc:
        print(f"[DSO train] post-training rollout failed ({exc}), skipping")

    device, env_info, labels = collect_jax_run_contract(
        requested_device="gpu",
        context="dso/train",
        extra_env_meta=collect_dataset_provenance(
            task="dso", task_config=task_config, split="train"
        ),
        extra_labels={
            "record_kind": "train",
            "algo_family": "single_agent_rl",
            "backend_family": "jax",
            "split_status": "canonical",
        },
    )
    record = RunRecord(
        task="dso", variant="dso_nflex", algo=algo, seed=seed, run_id=run_id,
        config_hash=config_hash({**config, **task_config}),
        status="completed", split="train", backend="jax_rejax", device=device,
        metrics={**result.summary, **dso_metrics},
        walltime_s=walltime,
        compile_warmup_s=compile_warmup,
        throughput_sps=config["total_timesteps"] / walltime if walltime > 0 else None,
        notes=(
            "data_mode=real"
            + (
                f" | saute_budget={config['saute_budget']}"
                if algo == "saute_ppo" and "saute_budget" in config
                else ""
            )
        ),
        env_info=env_info,
        labels=labels,
        artifacts=artifacts,
    )
    path = save_run(record, task_dir)
    print(f"[DSO train] saved to {path}")
    return record
