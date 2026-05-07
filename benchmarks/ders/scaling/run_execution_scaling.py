#!/usr/bin/env python
"""DERs execution-scaling throughput driver.

Measures steady-state env steps/second (per agent-env-step) for IPPO-typed
training at nenv in {16,32,64,128,256} on JAX.  Uses the XLA persistent
compilation cache to avoid re-compilation between the warmup and measurement
passes for each nenv.

Results written to benchmarks/ders/results/scaling/.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

TASK_DIR = _PROJECT_ROOT / "benchmarks" / "ders"
DEFAULT_OUTPUT_DIR = TASK_DIR / "results" / "scaling"
SCHEMA_VERSION = "ders_execution_scaling_v1"

_DEFAULT_NENVS = "16,32,64,128,256"
_WARMUP_UPDATES = 2
_STEADY_UPDATES = 20


@dataclass(frozen=True)
class _Cell:
    nenv: int
    seed: int


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def _block(tree: Any) -> None:
    import jax
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
            return


def _run_jax_cell(cell: _Cell, n_steps: int, cache_dir: str) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import optax
    from functools import partial

    from benchmarks.common.configs import load_task_config, load_train_config
    from benchmarks.common.runtime import build_train_cfg
    from powerzoojax.case import load_case
    from powerzoojax.envs.grid.dist import DistGridEnv
    from powerzoojax.rl.ippo import SharedActorCritic
    from powerzoojax.rl.multi_agent import DistGridMARLEnv
    from powerzoojax.tasks.ders import (
        apply_ders_profile_window,
        load_ders_split_profiles,
        make_ders_params,
    )

    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_enable_compilation_cache", True)

    task_config = load_task_config(TASK_DIR)
    ippo_config = load_train_config(TASK_DIR, "ippo", None, default_key="ippo")
    max_steps_ep = task_config.get("max_steps", 48)
    voltage_penalty = float(
        ippo_config.get("voltage_penalty", task_config.get("voltage_penalty", 4.0))
    )

    case = load_case(task_config.get("case", "case141"))
    base_params = make_ders_params(
        case,
        max_steps=max_steps_ep,
        v_min=float(task_config["v_min"]),
        v_max=float(task_config["v_max"]),
    )
    load_shape, pv_profile = load_ders_split_profiles(role="train")
    params = apply_ders_profile_window(
        base_params,
        load_shape=load_shape,
        pv_profile=pv_profile,
        episode_start=0,
        load_scale=float(task_config.get("train_load_scale", 1.0)),
        pv_scale=float(task_config.get("train_pv_scale", 1.0)),
    )

    env_marl = DistGridMARLEnv(
        DistGridEnv(),
        params,
        voltage_penalty=voltage_penalty,
        observation_mode="local",
    )

    agent_names = env_marl.agent_names
    n_agents = env_marl.num_agents
    obs_dim = env_marl.observation_space().shape[0]
    action_dim = env_marl.action_space().shape[0]
    n_envs = cell.nenv

    type_to_indices: dict[str, list[int]] = {}
    for i, name in enumerate(agent_names):
        atype = name.split("_")[0]
        type_to_indices.setdefault(atype, []).append(i)
    types = sorted(type_to_indices.keys())

    import numpy as np
    type_indices_np = {t: np.asarray(idxs, dtype=np.int32) for t, idxs in type_to_indices.items()}

    train_cfg = build_train_cfg(ippo_config).replace(
        num_envs=n_envs,
        n_steps=n_steps,
        total_timesteps=n_envs * n_steps * (_WARMUP_UPDATES + _STEADY_UPDATES),
        eval_freq=0,
        record_eval_wall_time=False,
    )
    n_epochs = train_cfg.n_epochs
    gamma = train_cfg.gamma
    gae_lambda = train_cfg.gae_lambda
    clip_eps = train_cfg.clip_eps
    ent_coef = train_cfg.ent_coef
    vf_coef = train_cfg.vf_coef
    max_grad_norm = train_cfg.max_grad_norm

    networks = {
        t: SharedActorCritic(hidden_dims=train_cfg.hidden_dims, action_dim=action_dim)
        for t in types
    }
    tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(train_cfg.learning_rate),
    )

    def _sample_action(key, mean, log_std):
        return mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape)

    def _log_prob(mean, log_std, action):
        return jnp.sum(
            -0.5 * ((action - mean) / jnp.exp(log_std)) ** 2
            - log_std - 0.5 * jnp.log(2 * jnp.pi)
        )

    def _entropy(log_std):
        return jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.e * jnp.pi))

    def _forward_typed(net_params, all_obs):
        means = jnp.zeros((n_agents, n_envs, action_dim))
        log_stds = jnp.zeros((n_agents, n_envs, action_dim))
        values = jnp.zeros((n_agents, n_envs))
        for t in types:
            idxs = jnp.asarray(type_indices_np[t])
            n_t = len(type_indices_np[t])
            t_flat = all_obs[idxs].reshape(n_t * n_envs, obs_dim)
            t_mean, t_ls, t_val = jax.vmap(
                partial(networks[t].apply, net_params[t])
            )(t_flat)
            means = means.at[idxs].set(t_mean.reshape(n_t, n_envs, action_dim))
            log_stds = log_stds.at[idxs].set(
                jnp.broadcast_to(t_ls, (n_t * n_envs, action_dim)).reshape(
                    n_t, n_envs, action_dim
                )
            )
            values = values.at[idxs].set(t_val.reshape(n_t, n_envs))
        return means, log_stds, values

    def _single_update(runner_state):
        net_params, opt_state, env_states, obs_dicts, key = runner_state

        def _env_step(carry, _):
            env_states, obs_dicts, key = carry
            key, act_key, step_key = jax.random.split(key, 3)
            all_obs = jnp.stack([obs_dicts[name] for name in agent_names])
            means, log_stds, values = _forward_typed(net_params, all_obs)
            flat_mean = means.reshape(n_agents * n_envs, action_dim)
            flat_ls = log_stds.reshape(n_agents * n_envs, action_dim)
            flat_actions = jax.vmap(_sample_action)(
                jax.random.split(act_key, n_agents * n_envs), flat_mean, flat_ls
            )
            actions_arr = jnp.clip(
                flat_actions.reshape(n_agents, n_envs, action_dim), -1.0, 1.0
            )
            actions = {name: actions_arr[i] for i, name in enumerate(agent_names)}
            flat_lp = jax.vmap(_log_prob)(flat_mean, flat_ls, flat_actions)
            log_probs = flat_lp.reshape(n_agents, n_envs)
            step_keys = jax.random.split(step_key, n_envs)
            next_obs_dicts, next_states, rewards_dicts, dones_dicts, _ = jax.vmap(
                env_marl.step
            )(step_keys, env_states, actions)
            rewards = jnp.stack([rewards_dicts[name] for name in agent_names])
            dones = dones_dicts["__all__"]
            all_next_obs = jnp.stack([next_obs_dicts[name] for name in agent_names])
            _, _, next_values = _forward_typed(net_params, all_next_obs)
            return (next_states, next_obs_dicts, key), (
                all_obs, actions_arr, log_probs, rewards, dones, values, next_values
            )

        (env_states, obs_dicts, key), rollout = jax.lax.scan(
            _env_step, (env_states, obs_dicts, key), None, length=n_steps
        )
        (all_obs, actions_arr, log_probs, rewards, dones, values, next_values) = rollout

        # GAE
        def _gae(carry, inp):
            gae, next_val = carry
            rew, done, val, nv = inp
            delta = rew + gamma * nv * (1 - done) - val
            gae = delta + gamma * gae_lambda * (1 - done) * gae
            return (gae, val), (gae, gae + val)

        _, (advs, rets) = jax.lax.scan(
            _gae,
            (jnp.zeros((n_agents, n_envs)), next_values[-1]),
            (rewards, dones, values, next_values),
            reverse=True,
        )
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        batches_by_type = {}
        for t in types:
            idxs = type_indices_np[t]
            n_t = len(idxs)
            t_obs = all_obs[:, idxs].reshape(n_steps * n_t * n_envs, obs_dim)
            t_act = actions_arr[:, idxs].reshape(n_steps * n_t * n_envs, action_dim)
            t_lp = log_probs[:, idxs].reshape(n_steps * n_t * n_envs)
            t_adv = advs[:, idxs].reshape(n_steps * n_t * n_envs)
            t_ret = rets[:, idxs].reshape(n_steps * n_t * n_envs)
            batches_by_type[t] = (t_obs, t_act, t_lp, t_adv, t_ret)

        def _ppo_loss(net_params, batches):
            total = jnp.float32(0.0)
            for t in types:
                obs, act, old_lp, adv, ret = batches[t]
                t_mean, t_ls, t_val = jax.vmap(
                    partial(networks[t].apply, net_params[t])
                )(obs)
                t_ls_bc = jnp.broadcast_to(t_ls, t_mean.shape)
                lp = jax.vmap(_log_prob)(t_mean, t_ls_bc, act)
                ent = _entropy(t_ls)
                ratio = jnp.exp(lp - old_lp)
                clipped = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
                pi_loss = -jnp.minimum(ratio * adv, clipped * adv).mean()
                vf_loss = 0.5 * ((t_val - ret) ** 2).mean()
                total = total + pi_loss + vf_coef * vf_loss - ent_coef * ent.mean()
            return total / len(types), total

        def _update_epoch(carry, _):
            net_params, opt_state, key = carry
            key, perm_key = jax.random.split(key)
            shuffled = {}
            for i_t, t in enumerate(types):
                n_t = len(type_indices_np[t])
                tt = n_steps * n_t * n_envs
                t_pk = jax.random.fold_in(perm_key, i_t)
                perm = jax.random.permutation(t_pk, tt)
                shuffled[t] = jax.tree.map(lambda x: x[perm], batches_by_type[t])
            grad_fn = jax.value_and_grad(_ppo_loss, has_aux=True)
            (loss, _), grads = grad_fn(net_params, shuffled)
            updates, opt_state = tx.update(grads, opt_state, net_params)
            net_params = optax.apply_updates(net_params, updates)
            return (net_params, opt_state, key), loss

        key, epoch_key = jax.random.split(key)
        (net_params, opt_state, _), _ = jax.lax.scan(
            _update_epoch, (net_params, opt_state, epoch_key), None, length=n_epochs
        )
        return (net_params, opt_state, env_states, obs_dicts, key), {}

    train_step = jax.jit(_single_update)

    # Init
    key = jax.random.PRNGKey(cell.seed)
    key, init_key = jax.random.split(key)
    dummy_obs = jnp.zeros((obs_dim,))
    t0_init = time.perf_counter()
    net_params = {
        t: networks[t].init(jax.random.fold_in(init_key, i), dummy_obs)
        for i, t in enumerate(types)
    }
    opt_state = tx.init(net_params)
    key, env_key = jax.random.split(key)
    env_keys = jax.random.split(env_key, n_envs)
    obs_dicts, env_states = jax.vmap(env_marl.reset)(env_keys)
    _block(env_states)
    init_time = time.perf_counter() - t0_init

    runner_state = (net_params, opt_state, env_states, obs_dicts, key)

    # Compile
    t0 = time.perf_counter()
    _ = train_step.lower(runner_state).compile()
    compile_time = time.perf_counter() - t0

    # Warmup
    warmup_time = 0.0
    for _ in range(_WARMUP_UPDATES):
        t0 = time.perf_counter()
        runner_state, _ = train_step(runner_state)
        _block(runner_state[0])
        warmup_time += time.perf_counter() - t0

    # Steady
    t0 = time.perf_counter()
    for _ in range(_STEADY_UPDATES):
        runner_state, _ = train_step(runner_state)
        _block(runner_state[0])
    steady_time = time.perf_counter() - t0

    steady_transitions = n_envs * n_steps * _STEADY_UPDATES
    sps = steady_transitions / steady_time if steady_time > 0 else None
    spu = steady_time / _STEADY_UPDATES if _STEADY_UPDATES > 0 else None
    return {
        "backend": "jax_rejax",
        "nenv": cell.nenv,
        "seed": cell.seed,
        "n_steps": n_steps,
        "status": "completed",
        "steps_per_sec": sps,
        "steady_state_env_steps_per_second": sps,
        "run_seconds": steady_time,
        "warmup_seconds": warmup_time,
        "compile_time_s": compile_time,
        "init_time_s": init_time,
        "seconds_per_update": spu,
        "wall_clock_per_1M_transitions": (1_000_000 / sps) if sps else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "execution_scaling.json"
    csv_path = output_dir / "scaling_results_extended.csv"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "task": "ders",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "results": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    fieldnames = ["backend", "nenv", "seed", "steps_per_sec", "run_seconds",
                  "warmup_seconds", "compile_time_s", "status"]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return json_path, csv_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nenvs", default=_DEFAULT_NENVS)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--mode", choices=["dry-run", "formal"], default="formal")
    args = parser.parse_args(argv)

    from benchmarks.common.configs import load_train_config
    ippo_cfg = load_train_config(TASK_DIR, "ippo", None, default_key="ippo")
    n_steps = args.n_steps or int(ippo_cfg.get("n_steps", 48))

    nenvs = _parse_int_list(args.nenvs)
    seeds = _parse_int_list(args.seeds)

    cells = [_Cell(nenv=n, seed=s) for n in nenvs for s in seeds]
    print(f"[ders_scaling] {len(cells)} cells: nenvs={nenvs} seeds={seeds} n_steps={n_steps}")

    if args.mode == "dry-run":
        print("[ders_scaling] dry-run, no measurements")
        return

    cache_dir = tempfile.mkdtemp(prefix="ders_jax_cache_")
    output_dir = Path(args.output_dir)
    rows: list[dict[str, Any]] = []

    existing_json = output_dir / "execution_scaling.json"
    if existing_json.exists():
        existing = json.loads(existing_json.read_text(encoding="utf-8"))
        rows = list(existing.get("results", []))

    key_fields = ("backend", "nenv", "seed", "n_steps")

    for idx, cell in enumerate(cells, 1):
        print(f"[ders_scaling] {idx}/{len(cells)} nenv={cell.nenv} seed={cell.seed}", flush=True)
        try:
            row = _run_jax_cell(cell, n_steps, cache_dir)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"[ders_scaling] FAILED nenv={cell.nenv} seed={cell.seed}: {exc}", flush=True)
            row = {
                "backend": "jax_rejax",
                "nenv": cell.nenv,
                "seed": cell.seed,
                "n_steps": n_steps,
                "status": f"failed: {exc}",
                "steps_per_sec": None,
                "run_seconds": None,
                "warmup_seconds": None,
                "compile_time_s": None,
            }
        key = tuple(row.get(k) for k in key_fields)
        rows = [r for r in rows if tuple(r.get(k) for k in key_fields) != key]
        rows.append(row)
        print(f"[ders_scaling] done nenv={cell.nenv} seed={cell.seed} "
              f"sps={row.get('steps_per_sec')}", flush=True)
        _write_outputs(rows, output_dir)

    json_path, csv_path = _write_outputs(rows, output_dir)
    print(f"[ders_scaling] wrote {json_path}")
    print(f"[ders_scaling] wrote {csv_path}")


if __name__ == "__main__":
    main()
