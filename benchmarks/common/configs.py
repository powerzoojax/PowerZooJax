"""Unified config loader for benchmark JSON / YAML files.

YAML is the preferred format for human-maintained configs (task.yaml,
train_*.yaml, eval_*.yaml) because it supports inline comments, which
JSON does not.  Auto-generated provenance (e.g. ``provenance.json``
written by ``experiment_ops.py derive_target``) stays in JSON to avoid pyyaml's
comment-stripping behaviour on round-trip.

Usage::

    from benchmarks.common.configs import load_config, load_task_config

    train_cfg = load_config(task_dir / "configs" / "train_ppo.yaml")
    task_cfg  = load_task_config(task_dir)        # merges task.yaml + provenance.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    """Load a config file, dispatching by suffix.

    Supports ``.yaml``/``.yml`` (via PyYAML's safe loader) and ``.json``
    (via the standard library).  Empty YAML files load as ``{}``.
    """
    text = Path(path).read_text(encoding="utf-8")
    suffix = Path(path).suffix.lower()
    if suffix in (".yaml", ".yml"):
        data = yaml.safe_load(text)
        return data if data is not None else {}
    if suffix == ".json":
        return json.loads(text)
    raise ValueError(
        f"load_config: unknown config format {suffix!r} for {path}; "
        f"expected .yaml/.yml/.json"
    )


def load_task_config(
    task_dir: Path,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return a task config dict.

    Default mode loads and merges ``task.yaml`` + ``provenance.json``.
    ``task.yaml`` carries the human-edited, comment-bearing static
    definition. ``provenance.json`` carries the auto-derived convergence
    target + provenance (written by
    ``benchmarks/common/experiment_ops.py``). The provenance keys overlay
    the task keys so legacy code reading
    ``task_config["convergence_threshold_per_split"]`` continues to work
    without changes.

    When ``config_path`` is given, load that file directly and do **not**
    overlay the default ``provenance.json``. This is useful for ad-hoc or
    scenario-specific task variants that should not inherit the benchmark's
    frozen reference provenance.

    For backward compatibility, falls back to ``task.json`` if
    ``task.yaml`` is absent.
    """
    configs_dir = Path(task_dir) / "configs"
    task_yaml = configs_dir / "task.yaml"
    task_json = configs_dir / "task.json"
    provenance_json = configs_dir / "provenance.json"

    if config_path is not None:
        return load_config(Path(config_path))

    if task_yaml.exists():
        cfg = load_config(task_yaml)
    elif task_json.exists():
        cfg = load_config(task_json)
    else:
        raise FileNotFoundError(
            f"No task config found at {task_yaml} or {task_json}"
        )

    if provenance_json.exists():
        cfg.update(load_config(provenance_json))
    return cfg


def load_train_config(
    task_dir: Path,
    algo: str,
    config_path: str | None,
    *,
    algo_key_map: dict[str, str] | None = None,
    default_key: str | None = None,
    allowed_algos: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Load a train config, resolving a per-task algo-to-filename mapping.

    If ``config_path`` is given, load that file directly (CLI override).
    Otherwise, translate ``algo`` through ``algo_key_map`` and load
    ``task_dir/configs/train_<key>.yaml``.  Algo values absent from
    ``algo_key_map`` fall back to ``default_key`` when provided, else to the
    algo name itself.

    ``allowed_algos``
        When provided, raise :class:`NotImplementedError` for any ``algo``
        not in this tuple (e.g. dc_microgrid supports ppo and sac).
    """
    if config_path is not None:
        return load_config(Path(config_path))
    if allowed_algos is not None and algo not in allowed_algos:
        raise NotImplementedError(
            f"Algo {algo!r} is not supported here. Allowed: {allowed_algos}"
        )
    fallback = default_key if default_key is not None else algo
    key = (algo_key_map or {}).get(algo, fallback)
    return load_config(task_dir / "configs" / f"train_{key}.yaml")


def load_train_config_for_run(
    task_dir: Path,
    run_record,
    *,
    algo_key_map: dict[str, str] | None = None,
    default_key: str | None = None,
) -> dict[str, Any]:
    """Load the train config for a historical run, preferring its frozen snapshot."""
    artifacts = getattr(run_record, "artifacts", None) or {}
    rel = artifacts.get("config")
    if rel:
        cfg = load_config(task_dir / "results" / rel)
        for key in ("train_config_raw", "train_config", "train_config_resolved"):
            train_cfg = cfg.get(key)
            if isinstance(train_cfg, dict):
                return train_cfg
    algo = getattr(run_record, "algo")
    return load_train_config(
        task_dir,
        algo,
        config_path=None,
        algo_key_map=algo_key_map,
        default_key=default_key,
    )


def load_task_config_for_run(
    task_dir: Path,
    run_record,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load the task config for a historical run, preferring its frozen snapshot."""
    if config_path is not None:
        return load_task_config(task_dir, config_path)

    artifacts = getattr(run_record, "artifacts", None) or {}
    rel = artifacts.get("config")
    if rel:
        cfg = load_config(task_dir / "results" / rel)
        task_cfg = cfg.get("task_config")
        if isinstance(task_cfg, dict):
            return task_cfg
    return load_task_config(task_dir)


def dump_yaml(data: dict[str, Any], path: Path) -> None:
    """Write ``data`` to ``path`` as YAML.

    Uses block style (``default_flow_style=False``) and preserves the
    insertion order of the input dict (``sort_keys=False``).  Comments
    in the destination file are NOT preserved across reads / writes;
    callers that need to round-trip with comments must use a different
    serialiser.
    """
    Path(path).write_text(
        yaml.safe_dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def dump_json(data: dict[str, Any], path: Path, *, indent: int = 2) -> None:
    """Write ``data`` to ``path`` as pretty-printed JSON."""
    Path(path).write_text(
        json.dumps(data, indent=indent, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
