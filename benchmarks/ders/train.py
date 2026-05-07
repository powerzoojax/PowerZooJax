"""DERs training script — typed IPPO variants for heterogeneous Dec-POMDPs."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from benchmarks.common.configs import load_task_config, load_train_config
from benchmarks.common.runtime import prefer_packaged_cuda_binaries
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

prefer_packaged_cuda_binaries()

# Safe variants keep their own config files; every other algo falls back to plain ippo.
_ALGO_KEY_MAP: dict[str, str] = {
    "ippo_safe": "ippo_safe",
    "ippo_lagrangian": "ippo_lagrangian",
}
_DEFAULT_KEY = "ippo"


def _unique_ints(values: np.ndarray | list[int]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        iv = int(value)
        if iv in seen:
            continue
        seen.add(iv)
        out.append(iv)
    return out


def _uniform_window_starts(total_len: int, max_steps: int, count: int) -> list[int]:
    usable = max(0, int(total_len) - int(max_steps))
    raw = np.linspace(0, usable, max(int(count), 1)).astype(np.int32)
    starts = _unique_ints(raw.tolist())
    return starts or [0]


def _build_train_params_bank(
    *,
    case,
    task_config: dict,
    max_steps: int,
) -> tuple[list, list[int], dict[str, Any]]:
    from powerzoojax.envs.grid.dist import DistGridEnv
    from powerzoojax.tasks.ders import (
        DERS_V_MAX,
        DERS_V_MIN,
        apply_ders_profile_window,
        compute_ders_safety_metrics,
        ders_no_control_rollout,
        load_ders_split_profiles,
        make_ders_params,
    )

    role = str(task_config.get("train_window_role", "train"))
    count = int(task_config.get("train_window_count", 8))
    selector = str(task_config.get("train_window_selector", "uniform"))
    load_scale = float(task_config.get("train_load_scale", 1.0))
    pv_scale = float(task_config.get("train_pv_scale", 1.0))
    v_min = float(task_config.get("v_min", DERS_V_MIN))
    v_max = float(task_config.get("v_max", DERS_V_MAX))
    score_v_min = float(task_config.get("train_window_score_v_min", v_min))
    score_v_max = float(task_config.get("train_window_score_v_max", v_max))

    load_shape, pv_profile = load_ders_split_profiles(role=role)
    total_len = min(len(load_shape), len(pv_profile))
    base_params = make_ders_params(
        case,
        max_steps=max_steps,
        v_min=v_min,
        v_max=v_max,
    )

    if selector in ("uniform", ""):
        starts = _uniform_window_starts(total_len, max_steps, count)
    elif selector in ("hardest", "hardest_window", "hardest_voltage_margin"):
        candidate_count = int(
            task_config.get(
                "train_window_candidate_count",
                max(count, min(32, count * 4)),
            )
        )
        candidate_starts = _uniform_window_starts(total_len, max_steps, candidate_count)
        env = DistGridEnv()
        scored: list[tuple[tuple[float, float, float, float], int]] = []
        for start in candidate_starts:
            params = apply_ders_profile_window(
                base_params,
                load_shape=load_shape,
                pv_profile=pv_profile,
                episode_start=int(start),
                load_scale=load_scale,
                pv_scale=pv_scale,
            )
            rollout = ders_no_control_rollout(env, params, jax.random.PRNGKey(10_000 + int(start)))
            safety = compute_ders_safety_metrics(
                rollout,
                v_min=score_v_min,
                v_max=score_v_max,
            )
            violation_steps = int(safety["undervoltage_steps"]) + int(safety["overvoltage_steps"])
            v_min_ep = np.asarray(rollout["v_min_episode"], dtype=np.float32)
            v_max_ep = np.asarray(rollout["v_max_episode"], dtype=np.float32)
            lower_margin = float(np.min(v_min_ep - score_v_min))
            upper_margin = float(np.min(score_v_max - v_max_ep))
            worst_margin = min(lower_margin, upper_margin)
            mean_margin = min(
                float(np.mean(v_min_ep - score_v_min)),
                float(np.mean(score_v_max - v_max_ep)),
            )
            score = (
                float(violation_steps),
                float(safety["max_undervoltage_dev"]),
                float(safety["max_overvoltage_dev"]),
                -worst_margin,
                -mean_margin,
                -float(safety["voltage_safety_rate"]),
            )
            scored.append((score, int(start)))
        scored.sort(key=lambda item: item[0], reverse=True)
        starts = sorted(_unique_ints(start for _score, start in scored[:count]))
        if not starts:
            starts = _uniform_window_starts(total_len, max_steps, count)
    else:
        raise ValueError(
            f"Unknown DERs train_window_selector={selector!r}; "
            "expected 'uniform' or 'hardest_voltage_margin'."
        )

    params_bank = [
        apply_ders_profile_window(
            base_params,
            load_shape=load_shape,
            pv_profile=pv_profile,
            episode_start=int(start),
            load_scale=load_scale,
            pv_scale=pv_scale,
        )
        for start in starts
    ]
    bank_meta = {
        "train_window_role": role,
        "train_window_selector": selector or "uniform",
        "train_window_count": len(params_bank),
        "train_window_starts": list(starts),
        "train_load_scale": load_scale,
        "train_pv_scale": pv_scale,
        "train_window_score_v_min": score_v_min,
        "train_window_score_v_max": score_v_max,
        "train_total_profile_len": int(total_len),
    }
    return params_bank, starts, bank_meta


@struct.dataclass
class _SampledDERsMARLState:
    """MARL state augmented with the sampled train-window index."""

    grid_state: Any
    params_idx: chex.Array


class _SampledParamsDistGridMARLEnv:
    """DistGrid MARL wrapper that samples one fixed profile window per reset."""

    def __init__(
        self,
        dist_env,
        params_bank,
        *,
        voltage_penalty: float = 0.0,
        soc_penalty: float = 0.0,
        observation_mode: str = "local",
        selected_constraint_names: tuple[str, ...] | None = None,
    ):
        from powerzoojax.rl.multi_agent import DistGridMARLEnv

        if not params_bank:
            raise ValueError("params_bank must be non-empty")
        self._params_bank = tuple(params_bank)
        self._n_params = len(self._params_bank)
        self._base_env = DistGridMARLEnv(
            dist_env,
            self._params_bank[0],
            voltage_penalty=voltage_penalty,
            soc_penalty=soc_penalty,
            observation_mode=observation_mode,
        )
        self._dist_env = self._base_env._dist_env
        self._dist_params = self._base_env._dist_params
        self._voltage_penalty = self._base_env._voltage_penalty
        self._soc_penalty = self._base_env._soc_penalty
        self._reward_mode = self._base_env._reward_mode
        self._observation_mode = self._base_env._observation_mode
        self._grid_core_dim = self._base_env._grid_core_dim
        self._n_nodes = self._base_env._n_nodes
        self._sincos_start = self._base_env._sincos_start
        self._all_names = self._base_env._all_names
        self._bundle_info = self._base_env._bundle_info
        self._agent_neighbor_idx = self._base_env._agent_neighbor_idx
        self._agent_bus_indices = self._base_env._agent_bus_indices
        self._obs_dim = self._base_env._obs_dim
        self._per_device_action_dim = self._base_env._per_device_action_dim
        self._build_obs_dict = self._base_env._build_obs_dict
        self._pack_actions = self._base_env._pack_actions
        self.constraint_names = tuple(
            self._dist_env.constraint_names(self._params_bank[0])
        )
        self.selected_constraint_names = tuple(
            selected_constraint_names or self.constraint_names
        )

    @property
    def num_agents(self) -> int:
        return self._base_env.num_agents

    @property
    def agent_names(self):
        return self._base_env.agent_names

    @property
    def name(self) -> str:
        return "SampledParamsDistGridMARLEnv"

    def observation_space(self, agent: str = None):
        return self._base_env.observation_space(agent)

    def action_space(self, agent: str = None):
        return self._base_env.action_space(agent)

    def _params_for_idx(self, idx: chex.Array):
        branches = tuple((lambda _operand, p=p: p) for p in self._params_bank)
        return jax.lax.switch(idx, branches, 0)

    @partial(jax.jit, static_argnums=(0,))
    def _sample_reset(self, key: chex.PRNGKey):
        sample_key, reset_key = jax.random.split(key)
        idx = jax.random.randint(sample_key, (), 0, self._n_params, dtype=jnp.int32)
        params = self._params_for_idx(idx)
        obs_flat, grid_state = self._dist_env.reset(reset_key, params)
        return self._build_obs_dict(obs_flat), grid_state, idx

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey):
        obs_dict, grid_state, idx = self._sample_reset(key)
        state = _SampledDERsMARLState(grid_state=grid_state, params_idx=idx)
        return obs_dict, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key: chex.PRNGKey, state: _SampledDERsMARLState, actions):
        step_key, sample_key, reset_key = jax.random.split(key, 3)
        params = self._params_for_idx(state.params_idx)
        flat_action = self._pack_actions(actions)
        obs_flat, new_grid_state, reward, costs, done, info = self._dist_env.step(
            step_key, state.grid_state, flat_action, params
        )
        info = {
            **info,
            "constraint_costs": costs,
            "cost": info.get("cost_sum", jnp.sum(costs)),
        }

        effective_reward = (
            reward
            - self._voltage_penalty * info["cost_continuous"]
            - self._soc_penalty * info.get("soc_terminal_sq", jnp.float32(0.0))
        )
        obs_dict = self._build_obs_dict(obs_flat)

        next_idx = jax.random.randint(sample_key, (), 0, self._n_params, dtype=jnp.int32)
        next_params = self._params_for_idx(next_idx)
        reset_obs_flat, reset_grid_state = self._dist_env.reset(reset_key, next_params)
        reset_obs_dict = self._build_obs_dict(reset_obs_flat)

        final_obs = jax.tree_util.tree_map(
            lambda cur, rst: jnp.where(done, rst, cur),
            obs_dict,
            reset_obs_dict,
        )
        final_grid_state = jax.tree_util.tree_map(
            lambda cur, rst: jnp.where(done, rst, cur),
            new_grid_state,
            reset_grid_state,
        )
        final_state = _SampledDERsMARLState(
            grid_state=jax.lax.stop_gradient(final_grid_state),
            params_idx=jnp.where(done, next_idx, state.params_idx),
        )
        rewards = {name: effective_reward for name in self._all_names}
        dones = {name: done for name in self._all_names}
        dones["__all__"] = done
        return final_obs, final_state, rewards, dones, info


def _sanity_rollout(env_marl, net_params, agent_names, type_to_indices, networks, max_steps, key):
    """Single-episode MARL rollout using deterministic (mean) policy."""
    obs_dict, state = env_marl.reset(key)
    rewards, costs, losses, v_mins, v_maxs = [], [], [], [], []
    for _ in range(max_steps):
        actions = {}
        for t, idxs in type_to_indices.items():
            for idx in idxs:
                name = agent_names[idx]
                mean, _, _ = networks[t].apply(net_params[t], obs_dict[name])
                actions[name] = jnp.clip(mean, -1.0, 1.0)
        key, subkey = jax.random.split(key)
        obs_dict, state, rewards_d, _, info = env_marl.step(subkey, state, actions)
        rewards.append(float(rewards_d[agent_names[0]]))
        costs.append(float(np.asarray(info.get("cost_continuous", 0.0))))
        losses.append(float(np.asarray(info.get("p_loss_MW", 0.0))))
        v_mag = state.grid_state.v_mag
        v_mins.append(float(jnp.min(v_mag)))
        v_maxs.append(float(jnp.max(v_mag)))
    return {
        "reward": np.array(rewards),
        "cost_continuous": np.array(costs),
        "p_loss_MW": np.array(losses),
        "v_min_episode": np.array(v_mins),
        "v_max_episode": np.array(v_maxs),
    }


def train_ders(
    task_dir: Path,
    algo: str,
    seed: int = 0,
    config_path: str | None = None,
) -> RunRecord:
    """Train a DERs typed MARL agent and save the result as a RunRecord."""
    from powerzoojax.rl.ippo import SharedActorCritic
    from powerzoojax.rl.multi_agent import DistGridMARLEnv
    from powerzoojax.rl.trainer import make_train
    from powerzoojax.tasks.ders import (
        DERsTask,
        ders_no_control_rollout,
        DistGridEnv,
    )

    config = load_train_config(
        task_dir, algo, config_path, algo_key_map=_ALGO_KEY_MAP, default_key=_DEFAULT_KEY
    )
    task_config = load_task_config(task_dir)
    max_steps = task_config.get("max_steps", 48)
    voltage_penalty = float(config.get("voltage_penalty", task_config.get("voltage_penalty", 4.0)))

    from powerzoojax.case import load_case

    case = load_case(task_config.get("case", "case141"))
    task = DERsTask(
        case=case,
        v_min=task_config["v_min"],
        v_max=task_config["v_max"],
        voltage_penalty=voltage_penalty,
        max_steps=max_steps,
    )
    params_bank, train_window_starts, bank_meta = _build_train_params_bank(
        case=case,
        task_config=task_config,
        max_steps=max_steps,
    )
    base_params = params_bank[0]
    env_marl = _SampledParamsDistGridMARLEnv(
        DistGridEnv(),
        params_bank,
        voltage_penalty=voltage_penalty,
        observation_mode="local",
        selected_constraint_names=task.constraint_spec().selected_names,
    )
    sanity_env = DistGridMARLEnv(
        DistGridEnv(),
        base_params,
        voltage_penalty=voltage_penalty,
        observation_mode="local",
    )
    agent_names = sanity_env.agent_names

    type_to_indices: dict[str, list[int]] = {}
    for i, name in enumerate(agent_names):
        atype = name.split("_")[0]
        type_to_indices.setdefault(atype, []).append(i)

    train_cfg = build_train_cfg(config)
    warmup_cfg = make_warmup_cfg(train_cfg)
    train_cfg = make_steady_train_cfg(train_cfg)
    hidden_dims = train_cfg.hidden_dims

    run_id = make_run_id("ders", algo, "train", seed)
    print(
        f"[DERs train] algo={algo} seed={seed} voltage_penalty={voltage_penalty} "
        f"train_windows={len(params_bank)} selector={bank_meta['train_window_selector']} "
        f"run_id={run_id}"
    )

    def _run_train(*, config, key):
        return make_train(env_marl, config)(key)

    result, walltime, compile_warmup = time_jax_train_with_warmup(
        _run_train,
        full_cfg=train_cfg,
        warmup_cfg=warmup_cfg,
        key=jax.random.PRNGKey(seed),
    )
    print(f"[DERs train] compile_warmup_s={compile_warmup:.3f} walltime_s={walltime:.3f}")

    net_params = result.params
    artifacts_dir = task_dir / "results" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    params_file = artifacts_dir / f"{run_id}_params.pkl"
    dump_pickle(net_params, params_file)

    artifacts = save_training_artifacts(
        result_metrics=result.metrics if isinstance(result.metrics, dict) else {},
        run_id=run_id,
        artifacts_dir=artifacts_dir,
        total_timesteps=int(config["total_timesteps"]),
        config_snapshot={
            "train_config": config,
            "task_config": task_config,
            "train_window_bank": bank_meta,
        },
        extra_artifacts={"params": f"artifacts/{params_file.name}"},
        eval_walltimes_s=(
            result.metrics.get("eval_wall_time_s")
            if isinstance(result.metrics, dict) else None
        ),
        eval_curve_source=None,
    )

    ders_metrics: dict = {}
    try:
        action_dim = sanity_env.action_space().shape[0]
        networks = {
            t: SharedActorCritic(hidden_dims=hidden_dims, action_dim=action_dim)
            for t in net_params
        }
        rollout = _sanity_rollout(
            sanity_env, net_params, agent_names, type_to_indices, networks,
            max_steps, jax.random.PRNGKey(seed * 1000 + 1),
        )
        no_ctrl = ders_no_control_rollout(
            DistGridEnv(), base_params, jax.random.PRNGKey(seed * 1000 + 2),
        )
        ders_metrics = task.compute_metrics(rollout, no_ctrl)
    except Exception as exc:
        print(f"[DERs train] post-training metrics failed ({exc}), skipping")

    device, env_info, labels = collect_jax_run_contract(
        requested_device="gpu",
        context="ders/train",
        extra_env_meta=collect_dataset_provenance(
            task="ders", task_config=task_config, split="train"
        ),
        extra_labels={
            "record_kind": "train",
            "algo_family": "cooperative_marl",
            "backend_family": "jax",
            "split_status": "canonical",
        },
    )
    record = RunRecord(
        task="ders", variant="ders_12agent", algo=algo, seed=seed, run_id=run_id,
        config_hash=config_hash({**config, **task_config}),
        status="completed", split="train",
        backend="jax_rejax",
        device=device,
        metrics={**result.summary, **ders_metrics},
        walltime_s=walltime,
        compile_warmup_s=compile_warmup,
        throughput_sps=config["total_timesteps"] / walltime if walltime > 0 else None,
        notes=(
            f"{algo}->{train_cfg.algo} voltage_penalty={voltage_penalty} "
            f"data_mode=real train_window_selector={bank_meta['train_window_selector']} "
            f"train_window_count={len(train_window_starts)}"
        ),
        env_info=env_info,
        labels=labels,
        artifacts=artifacts,
    )
    path = save_run(record, task_dir)
    print(f"[DERs train] saved to {path}")
    return record
