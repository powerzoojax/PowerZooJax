"""Artifact saving utilities shared across all benchmark tasks.

Call ``save_training_artifacts()`` immediately after ``train()`` returns.
Call ``save_eval_artifacts()`` after each eval split's rollout loop.

Path convention
---------------
All returned dict values are relative paths of the form ``"artifacts/<name>"``,
matching the existing ``RunRecord.artifacts`` convention used by TSO and
DC Microgrid pipelines. The relative root is always the task's ``results/``
directory; we do not call ``Path.relative_to`` because it requires the
``artifacts_dir`` to be located exactly under ``results/`` and breaks on
network mounts where ``..`` resolution differs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np


def _rel(p: Path) -> str:
    """Return a portable ``artifacts/<filename>`` relative path."""
    return f"artifacts/{p.name}"


def _safe_artifact_suffix(name: str) -> str:
    """Sanitize metric keys for use in flat artifact filenames."""
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_")


def save_training_artifacts(
    result_metrics: dict,
    run_id: str,
    artifacts_dir: Path,
    total_timesteps: int,
    config_snapshot: dict | None = None,
    extra_artifacts: dict[str, str] | None = None,
    eval_walltimes_s: "list | None" = None,
    train_curve_source: str | None = None,
    eval_curve_source: str | None = None,
) -> dict[str, str]:
    """Save all training curve arrays to ``artifacts_dir``.

    Handles both PPO (``eval_returns``, ``eval_episode_lengths``) and
    PPO-Lagrangian (``mean_reward``, ``mean_cost``, ``lambda``, ``loss``,
    ``episode_cost_est``) output shapes.

    Returns a dict mapping artifact key -> relative path string suitable for
    ``RunRecord.artifacts`` (paths relative to ``results/``).

    ``extra_artifacts`` allows callers to register paths the saver itself does
    not produce (e.g. TSO's ``params`` / ``checkpoints``).

    Canonical learning-curve artifacts
    ------------------------------------
    Three standard artifacts are written when sources are available:

    - ``learning_curve_train_return.npy``: training rollout return (with
      exploration). Source is selected by ``train_curve_source`` if given,
      else best-effort autodetect (``mean_reward`` / ``ep_rew_mean`` /
      ``returned_episode_returns``).
    - ``learning_curve_eval_return.npy``: offline eval return on a fixed
      eval env (no exploration). Source selected by ``eval_curve_source``
      if given, else best-effort autodetect (``eval_returns`` /
      ``eval/mean_reward``).
    - ``learning_curve_eval_walltimes.npy``: host **elapsed** wall time (s) at
      each eval (monotonic, same as Rejax ``eval_wall_time_s``), not a sum of
      per-eval intervals; from ``eval_walltimes_s``.

    Whichever source isn't present on this backend is silently skipped;
    downstream guards report missing canonical artifacts as warnings.
    """
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}

    explicit_timesteps = result_metrics.get("eval_timesteps")
    if explicit_timesteps is not None:
        try:
            timesteps = np.asarray(explicit_timesteps).flatten()
        except Exception:
            timesteps = None
    else:
        timesteps = None

    # Eval / update checkpoints are evenly spaced across total_timesteps when
    # no backend-specific checkpoint schedule is supplied.
    if timesteps is None or timesteps.size == 0:
        n_checkpoints = None
        for key in ("eval_returns", "mean_reward"):
            arr = result_metrics.get(key)
            if arr is None:
                continue
            try:
                n_checkpoints = int(np.asarray(arr).flatten().shape[0])
            except Exception:
                n_checkpoints = None
            if n_checkpoints:
                break

        if n_checkpoints is not None and n_checkpoints > 0:
            timesteps = np.linspace(0, total_timesteps, n_checkpoints)

    if timesteps is not None and np.asarray(timesteps).size > 0:
        p = artifacts_dir / f"{run_id}_timesteps.npy"
        np.save(p, timesteps)
        saved["timesteps"] = _rel(p)

    curve_keys = (
        # PPO
        "eval_returns",
        "eval_cost_per_step",
        "eval_episode_lengths",
        "eval_timesteps",
        # PPO-Lagrangian
        "mean_reward",
        "mean_cost",
        "lambda",
        "loss",
        "episode_cost_est",
    )
    for key in curve_keys:
        arr = result_metrics.get(key)
        if arr is None:
            continue
        try:
            arr_np = np.asarray(arr).flatten()
            if arr_np.ndim != 1 or arr_np.size == 0:
                continue
            p = artifacts_dir / f"{run_id}_{_safe_artifact_suffix(key)}.npy"
            np.save(p, arr_np)
            saved[key] = _rel(p)
        except Exception as exc:
            print(f"[artifacts] could not save {key}: {exc}")

    # ── sweep remaining 1-D float arrays (forward-compat for new metrics) ─
    for key, val in result_metrics.items():
        if key in curve_keys or key in saved:
            continue
        try:
            arr_np = np.asarray(val)
        except Exception:
            continue
        if arr_np.ndim == 0 or arr_np.size <= 1:
            continue
        if not np.issubdtype(arr_np.dtype, np.floating):
            continue
        try:
            p = artifacts_dir / f"{run_id}_{_safe_artifact_suffix(key)}.npy"
            np.save(p, arr_np)
            saved[key] = _rel(p)
        except Exception as exc:
            print(f"[artifacts] could not save {key}: {exc}")

    if config_snapshot is not None:
        try:
            p = artifacts_dir / f"{run_id}_config.json"
            p.write_text(
                json.dumps(
                    config_snapshot, indent=2, ensure_ascii=False, default=str
                ),
                encoding="utf-8",
            )
            saved["config"] = _rel(p)
        except Exception as exc:
            print(f"[artifacts] could not save config snapshot: {exc}")

    if extra_artifacts:
        for k, v in extra_artifacts.items():
            if v is not None:
                saved[k] = v

    # ── Canonical learning-curve artifact aliases ─────────────────────
    # Auto-detect train rollout source: prefer caller-provided key; fall
    # back to common metric names produced by self-implemented IPPO /
    # PPO-Lagrangian / SB3 / SBX.
    train_candidates = (
        [train_curve_source] if train_curve_source else []
    ) + ["mean_reward", "ep_rew_mean", "returned_episode_returns"]
    for key in train_candidates:
        if not key:
            continue
        arr = result_metrics.get(key)
        if arr is None:
            continue
        try:
            arr_np = np.asarray(arr).flatten()
            if arr_np.ndim != 1 or arr_np.size == 0:
                continue
            p = artifacts_dir / f"{run_id}_learning_curve_train_return.npy"
            np.save(p, arr_np)
            saved["learning_curve_train_return"] = _rel(p)
            break
        except Exception as exc:
            print(f"[artifacts] could not save canonical train curve from {key}: {exc}")

    # Auto-detect offline eval source: prefer caller-provided key; fall
    # back to rejax PPO eval_returns / SB3-EvalCallback eval/mean_reward.
    eval_candidates = (
        [eval_curve_source] if eval_curve_source else []
    ) + ["eval_returns", "eval/mean_reward"]
    for key in eval_candidates:
        if not key:
            continue
        arr = result_metrics.get(key)
        if arr is None:
            continue
        try:
            arr_np = np.asarray(arr).flatten()
            if arr_np.ndim != 1 or arr_np.size == 0:
                continue
            p = artifacts_dir / f"{run_id}_learning_curve_eval_return.npy"
            np.save(p, arr_np)
            saved["learning_curve_eval_return"] = _rel(p)
            break
        except Exception as exc:
            print(f"[artifacts] could not save canonical eval curve from {key}: {exc}")

    # Eval walltimes: only written when caller supplies them.
    if eval_walltimes_s is not None:
        try:
            arr_np = np.asarray(eval_walltimes_s).flatten()
            if arr_np.ndim == 1 and arr_np.size > 0:
                p = artifacts_dir / f"{run_id}_learning_curve_eval_walltimes.npy"
                np.save(p, arr_np)
                saved["learning_curve_eval_walltimes"] = _rel(p)
        except Exception as exc:
            print(f"[artifacts] could not save eval walltimes: {exc}")

    return saved


def read_parallel_n_envs_from_run_config(
    artifacts_dir: str | Path,
    run_id: str,
) -> Optional[int]:
    """Rollout / VecEnv parallelism for plot legends (fair comparison).

    Resolution order matches how benchmarks save ``*_config.json``:
    ``train_config.num_envs`` (DSO/DERs), ``train_config_raw`` /
    ``train_config_resolved`` (TSO), ``powerzoo_driver_config.n_envs``, else
    top-level ``n_envs``. Does **not** use ``task_config.num_envs`` (metadata).
    """
    artifacts_dir = Path(artifacts_dir)
    cfg_f = artifacts_dir / f"{run_id}_config.json"
    if not cfg_f.exists():
        return None
    try:
        cfg = json.loads(cfg_f.read_text(encoding="utf-8"))
    except Exception:
        return None
    train_cfg = cfg.get("train_config")
    if isinstance(train_cfg, dict) and train_cfg.get("num_envs") is not None:
        return int(train_cfg["num_envs"])
    raw = cfg.get("train_config_raw")
    if isinstance(raw, dict) and raw.get("num_envs") is not None:
        return int(raw["num_envs"])
    resolved = cfg.get("train_config_resolved")
    if isinstance(resolved, dict) and resolved.get("num_envs") is not None:
        return int(resolved["num_envs"])
    pz = cfg.get("powerzoo_driver_config")
    if isinstance(pz, dict) and pz.get("n_envs") is not None:
        return int(pz["n_envs"])
    if cfg.get("n_envs") is not None:
        return int(cfg["n_envs"])
    return None


def save_eval_artifacts(
    per_episode_metrics: list[dict[str, float]],
    run_id: str,
    split: str,
    artifacts_dir: Path,
    per_episode_actions: list | None = None,
    per_episode_rewards: list | None = None,
) -> dict[str, str]:
    """Save per-episode evaluation data.

    ``per_episode_metrics``
        List of dicts, one per episode, with float values.
    ``per_episode_actions``
        Optional list of action arrays (one per episode).
    ``per_episode_rewards``
        Optional list of reward sequences (one per episode).

    Returns a ``{artifact_key: relative_path}`` dict.
    """
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    # run_id already encodes the split (make_run_id format: task_algo_split_sN_ts),
    # so we do NOT append split again — avoids doubled suffixes like "_iid_..._iid_".
    prefix = run_id

    if per_episode_metrics:
        try:
            serialised = [
                {k: float(v) for k, v in m.items()} for m in per_episode_metrics
            ]
            p = artifacts_dir / f"{prefix}_per_episode.json"
            p.write_text(
                json.dumps(serialised, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            saved["per_episode"] = _rel(p)
        except Exception as exc:
            print(f"[artifacts] could not save per_episode.json: {exc}")

    arrays: dict[str, Any] = {}
    if per_episode_actions is not None:
        try:
            arrays["actions"] = np.array(per_episode_actions)
        except Exception:
            pass
    if per_episode_rewards is not None:
        try:
            arrays["rewards"] = np.array(per_episode_rewards)
        except Exception:
            pass
    if arrays:
        try:
            p = artifacts_dir / f"{prefix}_trajectory.npz"
            np.savez(p, **arrays)
            saved["trajectory"] = _rel(p)
        except Exception as exc:
            print(f"[artifacts] could not save trajectory.npz: {exc}")

    if per_episode_actions is not None:
        try:
            all_actions = np.concatenate(
                [np.atleast_2d(np.asarray(a)) for a in per_episode_actions], axis=0
            )
            stats: dict[str, Any] = {}
            n_dims = all_actions.shape[-1] if all_actions.ndim > 1 else 1
            for dim in range(n_dims):
                col = all_actions[..., dim] if all_actions.ndim > 1 else all_actions
                stats[f"dim_{dim}"] = {
                    "mean": float(np.mean(col)),
                    "std": float(np.std(col)),
                    "min": float(np.min(col)),
                    "max": float(np.max(col)),
                    "p25": float(np.percentile(col, 25)),
                    "p75": float(np.percentile(col, 75)),
                }
            p = artifacts_dir / f"{prefix}_action_stats.json"
            p.write_text(json.dumps(stats, indent=2), encoding="utf-8")
            saved["action_stats"] = _rel(p)
        except Exception as exc:
            print(f"[artifacts] could not save action_stats: {exc}")

    return saved
