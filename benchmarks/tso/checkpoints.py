"""TSO PPO-Lagrangian checkpoint artifact helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks.common.io import dump_pickle, load_pickle


def _rel(path: Path) -> str:
    return f"artifacts/{path.name}"


def save_checkpoint_bundle(
    *,
    run_id: str,
    checkpoints: list[tuple[int, Any]],
    artifacts_dir: Path,
) -> str:
    """Persist checkpoint params plus a small JSON manifest."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for index, (timesteps, params) in enumerate(checkpoints):
        params_path = artifacts_dir / f"{run_id}_checkpoint_{index:03d}_params.pkl"
        dump_pickle(params, params_path)
        entries.append(
            {
                "index": int(index),
                "timesteps": int(timesteps),
                "params": _rel(params_path),
            }
        )
    manifest_path = artifacts_dir / f"{run_id}_checkpoints.json"
    manifest_path.write_text(
        json.dumps({"run_id": run_id, "checkpoints": entries}, indent=2),
        encoding="utf-8",
    )
    return _rel(manifest_path)


def load_checkpoint_specs(task_dir: Path, manifest_rel: str) -> list[dict[str, Any]]:
    """Load checkpoint metadata from a saved TSO checkpoint manifest."""
    manifest_path = task_dir / "results" / manifest_rel
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoints = payload.get("checkpoints", [])
    if not isinstance(checkpoints, list):
        raise ValueError(f"Invalid checkpoint manifest: {manifest_path}")
    return checkpoints


def load_checkpoint_params(
    task_dir: Path,
    manifest_rel: str,
    checkpoint_index: int,
) -> tuple[Any, dict[str, Any]]:
    """Load one checkpoint params pytree by index from the manifest."""
    specs = load_checkpoint_specs(task_dir, manifest_rel)
    for spec in specs:
        if int(spec.get("index", -1)) != int(checkpoint_index):
            continue
        params_rel = spec.get("params")
        if not isinstance(params_rel, str):
            raise ValueError(f"Missing params path in checkpoint spec: {spec!r}")
        return load_pickle(task_dir / "results" / params_rel), spec
    raise IndexError(
        f"Checkpoint index {checkpoint_index} not found in {manifest_rel!r}."
    )
