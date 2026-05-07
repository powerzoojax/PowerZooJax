"""DERs evaluation script — Typed IPPO policy rollout."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from benchmarks.common.runtime import prefer_packaged_cuda_binaries

prefer_packaged_cuda_binaries()

import jax
import jax.numpy as jnp

from benchmarks.common.configs import (
    load_config,
    load_task_config_for_run,
    load_train_config_for_run,
)
from benchmarks.common.artifacts import save_eval_artifacts
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


_CLASS_MASK_ALIASES = {
    "battery": "battery",
    "batteries": "battery",
    "bat": "battery",
    "pv": "renewable",
    "renewable": "renewable",
    "renewables": "renewable",
    "flex": "flexload",
    "flexload": "flexload",
    "flexloads": "flexload",
}

_CLASS_SLUGS = {
    "battery": "bat",
    "renewable": "pv",
    "flexload": "flex",
}

_CLASS_MASK_ROLE = {
    frozenset({"renewable", "flexload"}): "bat_only",
    frozenset({"battery", "flexload"}): "pv_only",
    frozenset({"battery", "renewable"}): "flex_only",
}


def _normalize_class_mask(class_mask: object | None) -> tuple[str, ...]:
    """Normalize disabled DER device classes for counterfactual evaluation."""
    if class_mask is None:
        return ()
    if isinstance(class_mask, str):
        raw_items = [class_mask]
    else:
        raw_items = list(class_mask)

    normalized: list[str] = []
    for item in raw_items:
        for token in str(item).replace(",", " ").split():
            key = token.strip().lower().replace("-", "_")
            if not key:
                continue
            if key not in _CLASS_MASK_ALIASES:
                valid = "batteries, pv, flexloads"
                raise ValueError(
                    f"Unknown DER class mask {token!r}; expected one of {valid}."
                )
            normalized.append(_CLASS_MASK_ALIASES[key])
    return tuple(sorted(set(normalized)))


def _class_mask_role(class_mask: tuple[str, ...]) -> str:
    """Return the paper-facing counterfactual role for a class mask."""
    if not class_mask:
        return "full"
    direct = _CLASS_MASK_ROLE.get(frozenset(class_mask))
    if direct is not None:
        return direct
    return "mask_" + "_".join(_CLASS_SLUGS[c] for c in class_mask)


def _agent_class(agent_name: str) -> str:
    """Map DistGridMARLEnv agent names to the normalized DER class key."""
    prefix = agent_name.split("_", 1)[0]
    if prefix == "battery":
        return "battery"
    if prefix == "renewable":
        return "renewable"
    if prefix == "flexload":
        return "flexload"
    raise ValueError(f"Unknown DER agent class in {agent_name!r}.")


def _noop_actions_by_agent(env_marl, params) -> dict[str, jnp.ndarray]:
    """Split the flat DER no-op action into the MARL action dictionary layout."""
    from powerzoojax.tasks.ders import ders_noop_action

    agent_names = env_marl.agent_names
    per_agent_dim = int(env_marl.action_space().shape[0])
    noop_flat = ders_noop_action(params)
    expected = len(agent_names) * per_agent_dim
    if int(noop_flat.shape[0]) != expected:
        raise ValueError(
            "DER no-op action does not match MARL agent action layout: "
            f"flat_dim={int(noop_flat.shape[0])}, expected={expected}."
        )
    noop_matrix = noop_flat.reshape((len(agent_names), per_agent_dim))
    return {name: noop_matrix[i] for i, name in enumerate(agent_names)}


def _apply_class_mask(policy_fn, env_marl, params, class_mask: tuple[str, ...]):
    """Override selected DER classes with their device-wise no-op actions."""
    if not class_mask:
        return policy_fn

    masked_classes = frozenset(class_mask)
    noop_actions = _noop_actions_by_agent(env_marl, params)
    agent_names = tuple(env_marl.agent_names)

    def masked_policy_fn(obs_dict):
        actions = policy_fn(obs_dict)
        return {
            name: (
                noop_actions[name]
                if _agent_class(name) in masked_classes
                else actions[name]
            )
            for name in agent_names
        }

    return masked_policy_fn


def _build_typed_policy_fn(net_params, env_marl, hidden_dims, action_dim):
    """Build obs_dict -> actions_dict callable from per-type IPPO params."""
    from powerzoojax.rl.ippo import SharedActorCritic

    agent_names = env_marl.agent_names
    type_to_indices: dict[str, list[int]] = {}
    for i, name in enumerate(agent_names):
        type_to_indices.setdefault(name.split("_")[0], []).append(i)

    networks = {
        t: SharedActorCritic(hidden_dims=hidden_dims, action_dim=action_dim)
        for t in net_params
    }

    def policy_fn(obs_dict):
        actions = {}
        for t, idxs in type_to_indices.items():
            for idx in idxs:
                name = agent_names[idx]
                mean, _, _ = networks[t].apply(net_params[t], obs_dict[name])
                actions[name] = jnp.clip(mean, -1.0, 1.0)
        return actions

    return policy_fn


def _run_eval_episodes(
    *,
    task,
    split: str,
    net_params,
    hidden_dims: tuple[int, ...],
    voltage_penalty: float,
    n_episodes: int,
    max_steps: int,
    seed: int,
    class_mask: tuple[str, ...] = (),
) -> list[dict[str, float]]:
    from powerzoojax.rl.multi_agent import DistGridMARLEnv
    from powerzoojax.tasks.ders import DistGridEnv, rollout_ders_marl

    per_episode: list[dict[str, float]] = []
    for ep in range(n_episodes):
        key = jax.random.PRNGKey(seed * 10_000 + ep)
        ref_key = jax.random.PRNGKey(seed * 10_000 + ep + 50_000)
        episode_start = task.episode_start(
            split,
            ep,
            n_episodes,
            strategy="uniform",
            seed=seed,
        )
        params = task.params_from_start(split, episode_start)
        env_marl = DistGridMARLEnv(
            DistGridEnv(),
            params,
            voltage_penalty=voltage_penalty,
            observation_mode="local",
        )
        action_dim = env_marl.action_space().shape[0]
        policy_fn = _build_typed_policy_fn(
            net_params,
            env_marl,
            hidden_dims,
            action_dim,
        )
        policy_fn = _apply_class_mask(policy_fn, env_marl, params, class_mask)
        agent_data = rollout_ders_marl(env_marl, params, key, policy_fn)
        ref_data = task.baseline_rollout(env_marl, params, ref_key, "no_control")
        metrics = task.compute_metrics(agent_data, ref_data)
        metrics["episode_idx"] = float(ep)
        metrics["episode_start"] = float(episode_start)
        per_episode.append(metrics)
    return per_episode


def eval_ders(
    task_dir: Path,
    run_id: str,
    split: str,
    class_mask: object | None = None,
) -> RunRecord:
    """Evaluate a trained DERs run on the given split."""
    from powerzoojax.tasks.ders import DERsTask

    class_mask_tuple = _normalize_class_mask(class_mask)
    ablation_role = _class_mask_role(class_mask_tuple)
    original = load_run(run_id, task_dir)
    task_config = load_task_config_for_run(task_dir, original)
    eval_cfg_path = task_dir / "configs" / f"eval_{split}.yaml"
    n_episodes = 10
    eval_config: dict = {}
    if eval_cfg_path.exists():
        eval_config = load_config(eval_cfg_path)
        n_episodes = eval_config.get("n_eval_episodes", n_episodes)

    max_steps = task_config.get("max_steps", 48)

    from powerzoojax.case import load_case

    case = load_case(task_config.get("case", "case141"))

    train_config = load_train_config_for_run(
        task_dir,
        original,
        algo_key_map={
            "ippo_safe": "ippo_safe",
            "ippo_lagrangian": "ippo_lagrangian",
        },
        default_key="ippo",
    )
    hidden_dims = tuple(train_config.get("hidden_dims", [128, 128]))
    voltage_penalty = float(
        train_config.get("voltage_penalty", task_config.get("voltage_penalty", 4.0))
    )

    net_params = load_pickle(task_dir / "results" / original.artifacts["params"])

    task = DERsTask(
        case=case,
        v_min=task_config["v_min"],
        v_max=task_config["v_max"],
        voltage_penalty=voltage_penalty,
        max_steps=max_steps,
    )

    eval_algo = (
        original.algo if not class_mask_tuple else f"{original.algo}_{ablation_role}"
    )
    eval_run_id = make_run_id("ders", eval_algo, split, original.seed)
    mask_text = ",".join(class_mask_tuple) if class_mask_tuple else "none"
    print(
        f"[DERs eval] run_id={run_id} split={split} episodes={n_episodes} "
        f"class_mask={mask_text}"
    )
    t0 = time.time()
    all_metrics = _run_eval_episodes(
        task=task,
        split=split,
        net_params=net_params,
        hidden_dims=hidden_dims,
        voltage_penalty=voltage_penalty,
        n_episodes=n_episodes,
        max_steps=max_steps,
        seed=original.seed,
        class_mask=class_mask_tuple,
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
        context="ders/eval",
        record_device=original.device,
        extra_env_meta=collect_dataset_provenance(
            task="ders", task_config=task_config, split=split
        ),
        extra_labels={
            **dict(original.labels or {}),
            "record_kind": "eval",
            "source_run_id": run_id,
            "source_algo": original.algo,
            "class_mask": mask_text,
            "ablation_role": ablation_role,
            "counterfactual_ablation": bool(class_mask_tuple),
        },
    )
    record = RunRecord(
        task="ders", variant="ders_12agent", algo=eval_algo, seed=original.seed,
        run_id=eval_run_id,
        config_hash=config_hash(
            {**eval_config, **task_config, "class_mask": class_mask_tuple}
        ),
        status="completed", split=split,
        backend=original.backend,
        device=device,
        metrics=flat_metrics, walltime_s=walltime,
        notes=f"eval of {run_id}; class_mask={mask_text}",
        env_info=env_info,
        labels=labels,
        artifacts=arts,
    )
    path = save_run(record, task_dir)
    print(f"[DERs eval] saved to {path}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained DERs IPPO run.")
    parser.add_argument("--task-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--run-id", required=True, help="Training run_id to evaluate.")
    parser.add_argument(
        "--eval-splits",
        nargs="+",
        default=["iid"],
        help="Evaluation splits to run, e.g. iid pv_penetration_shift load_stress.",
    )
    parser.add_argument(
        "--class-mask",
        nargs="+",
        default=None,
        metavar="{batteries,pv,flexloads}",
        help=(
            "DER device classes to force to no-op. Use two classes for one-class-only "
            "counterfactuals, e.g. --class-mask pv flexloads for Bat-only."
        ),
    )
    args = parser.parse_args()

    for split in args.eval_splits:
        eval_ders(
            task_dir=args.task_dir,
            run_id=args.run_id,
            split=split,
            class_mask=args.class_mask,
        )


if __name__ == "__main__":
    main()
