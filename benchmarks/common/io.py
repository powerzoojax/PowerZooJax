"""Run-record schema and result I/O.

Conventions:
  - Each run is saved as ``results/runs/<run_id>.json``.
  - ``results/manifest.json`` is an auto-appended index of all runs.
  - ``results/summary/`` holds aggregated tables (produced by summarize.py).

The manifest is append-only. Re-running with the same run_id overwrites the
individual JSON but updates the manifest entry in-place.
"""

from __future__ import annotations

import hashlib
import json
import os
import fcntl
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Formal cross-backend campaign taxonomy (dispatcher CLI, paper tables).
CANONICAL_BACKENDS: tuple[str, ...] = ("jax_rejax", "sb3", "sbx")

# Historical RunRecord.backend only; not a campaign option. Loads via
# :meth:`RunRecord.from_dict` without normalization.
DEPRECATED_RUN_RECORD_BACKENDS: frozenset[str] = frozenset({"jax_jaxmarl"})

ACCEPTED_MANIFEST_BACKENDS: frozenset[str] = (
    frozenset(CANONICAL_BACKENDS) | DEPRECATED_RUN_RECORD_BACKENDS
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _PROJECT_ROOT / "powerzoojax" / "data"
_PARQUET_DIR = _DATA_DIR / "parquet"
_MANIFEST_DIR = _DATA_DIR / "manifests"

_TASK_DATASETS: dict[str, tuple[str, ...]] = {
    "dc_microgrid": ("google_dc_2019", "gb_gen_by_type", "gb_market_mid"),
    "dso": ("ausgrid_zone_substation_fy25_imputed",),
    "ders": ("ausgrid_zone_substation_fy25_imputed", "gb_gen_by_type"),
    "gencos": ("gb_forecast_actual_demand", "gb_gen_by_type"),
    "tso": ("gb_forecast_actual_demand", "gb_gen_by_type"),
}


@dataclass
class RunRecord:
    """One benchmark run (training, evaluation, or baseline).

    ``backend`` uses :data:`CANONICAL_BACKENDS` for new runs. Values in
    :data:`DEPRECATED_RUN_RECORD_BACKENDS` may appear in older JSON and are
    accepted when loading manifests unchanged.
    """

    task: str
    variant: str
    algo: str
    seed: int
    run_id: str

    config_hash: str = ""
    status: str = "completed"
    split: str = "train"

    backend: str = "jax_rejax"
    device: str = "gpu"
    framework_version: str = ""

    metrics: dict[str, Any] = field(default_factory=dict)
    convergence: dict[str, Any] = field(default_factory=dict)

    walltime_s: float = 0.0
    compile_warmup_s: float | None = None
    throughput_sps: float | None = None

    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    notes: str = ""
    env_info: dict[str, str] = field(default_factory=dict)
    labels: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    evidence_tier: str | None = None
    comparison_audited: bool | None = None
    audit_suite_version: str | None = None
    parity_scope: list[str] | str | None = None
    suppress_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, s: str) -> "RunRecord":
        return cls.from_dict(json.loads(s))


def collect_env_info() -> dict[str, str]:
    """Collect Python/JAX/CUDA/git version strings for reproducibility."""
    import sys

    info: dict[str, str] = {"python": sys.version.split()[0]}

    try:
        import jax

        info["jax"] = jax.__version__
    except Exception:
        info["jax"] = "unknown"
    try:
        import jaxlib

        info["jaxlib"] = jaxlib.__version__
    except Exception:
        info["jaxlib"] = "unknown"

    try:
        import jax

        backend = jax.default_backend()
        info["jax_backend"] = backend
        if backend == "gpu":
            try:
                devs = jax.devices()
                if devs:
                    device_kind = getattr(devs[0], "device_kind", "gpu")
                    info["jax_device_kind"] = device_kind
                    info["cuda"] = device_kind
                else:
                    info["jax_device_kind"] = "gpu"
                    info["cuda"] = "gpu"
            except Exception:
                info["jax_device_kind"] = "gpu"
                info["cuda"] = "gpu"
        else:
            info["jax_device_kind"] = backend
            info["cuda"] = f"n/a ({backend})"
    except Exception:
        info["jax_backend"] = "unknown"
        info["jax_device_kind"] = "unknown"
        info["cuda"] = "unknown"

    try:
        import subprocess

        nvs = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if nvs.returncode == 0 and nvs.stdout.strip():
            info["nvidia_driver"] = nvs.stdout.strip().splitlines()[0]
    except Exception:
        pass

    try:
        import subprocess

        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if gpu.returncode == 0 and gpu.stdout.strip():
            info["nvidia_gpu_name"] = gpu.stdout.strip().splitlines()[0]
    except Exception:
        pass

    return info


def merge_env_info_metadata(
    info: dict[str, str],
    meta: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Merge task/backend metadata into ``env_info`` as strings."""
    merged = dict(info)
    if not meta:
        return merged
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            if len(value) == 2:
                merged[key] = f"{value[0]}..{value[1]}"
            else:
                merged[key] = json.dumps(list(value), ensure_ascii=False)
            continue
        merged[key] = str(value)
    return merged


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_window(task: str, task_config: dict[str, Any], split: str | None) -> str:
    if split is None:
        return "unspecified"
    if task == "tso":
        if split == "train":
            start = task_config.get("gb_train_start")
            end = task_config.get("gb_train_end")
        else:
            start = task_config.get("gb_iid_start")
            end = task_config.get("gb_iid_end")
        if start and end:
            return f"{start}..{end}"
    windows = {
        "train_episode_start": task_config.get("train_episode_start"),
        "train_window_starts": task_config.get("train_window_starts"),
        "eval_episodes": task_config.get("eval_episodes"),
        "max_steps": task_config.get("max_steps"),
    }
    populated = {k: v for k, v in windows.items() if v is not None}
    return json.dumps({"split": split, **populated}, sort_keys=True)


def collect_dataset_provenance(
    *,
    task: str,
    task_config: dict[str, Any] | None = None,
    split: str | None = None,
    require_real_data: bool = True,
    fail_fast: bool = True,
) -> dict[str, Any]:
    """Return hard provenance metadata for a benchmark task's input data.

    The helper records the resolved parquet path(s), sha256 checksum(s), and
    split window used by the task.  It intentionally stays conservative:
    unknown tasks or missing files are marked as non-official, and formal
    real-data runs raise when ``fail_fast`` is true.
    """
    task_config = dict(task_config or {})
    task_key = str(task)
    datasets = list(_TASK_DATASETS.get(task_key, ()))
    if not datasets and task_config.get("data_source"):
        datasets = [str(task_config["data_source"])]

    resolved_paths: list[str] = []
    checksums: dict[str, str] = {}
    failures: list[str] = []
    for name in datasets:
        manifest_path = _MANIFEST_DIR / f"{name}.json"
        if not manifest_path.exists():
            failures.append(f"missing_manifest:{name}")
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            failures.append(f"manifest_read_error:{name}:{type(exc).__name__}")
            continue
        parquet_file = manifest.get("parquet_file")
        if not parquet_file:
            failures.append(f"missing_parquet_field:{name}")
            continue
        parquet_path = (_PARQUET_DIR / str(parquet_file)).resolve()
        if not parquet_path.exists():
            failures.append(f"missing_dataset:{name}:{parquet_path}")
            continue
        resolved_paths.append(str(parquet_path))
        try:
            checksums[name] = _file_sha256(parquet_path)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"checksum_error:{name}:{type(exc).__name__}")

    synthetic_fallback_used = bool(task_config.get("synthetic_fallback_used", False))
    if require_real_data and synthetic_fallback_used:
        failures.append("synthetic_fallback_used")
    if require_real_data and not resolved_paths:
        failures.append("dataset_resolved_path_missing")
    if require_real_data and not checksums:
        failures.append("dataset_checksum_missing")

    if fail_fast and failures:
        raise RuntimeError(
            f"{task_key}: formal data provenance check failed: {', '.join(failures)}"
        )

    checksum_payload = json.dumps(checksums, sort_keys=True)
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "task": task_key,
                "datasets": datasets,
                "paths": resolved_paths,
                "checksums": checksums,
                "split_window": _split_window(task_key, task_config, split),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    official = not failures
    return {
        "dataset_names": datasets,
        "dataset_resolved_path": resolved_paths,
        "dataset_checksum": checksum_payload if checksums else "",
        "dataset_fingerprint": fingerprint,
        "split_ids": split or "unspecified",
        "split_window": _split_window(task_key, task_config, split),
        "synthetic_fallback_used": synthetic_fallback_used,
        "data_provenance_ok": official,
        "data_provenance_failures": failures,
        "official_provenance": official,
    }


def normalize_device_name(device: str | None) -> str | None:
    """Normalize runtime device labels for contract checks."""
    if device is None:
        return None
    value = str(device).strip().lower()
    if value in ("", "auto", "none", "null", "default"):
        return None
    if value.startswith("cuda") or value.startswith("gpu"):
        return "gpu"
    if value.startswith("cpu"):
        return "cpu"
    if value.startswith("tpu"):
        return "tpu"
    return value


def resolve_requested_device(default: str | None = None) -> str | None:
    """Resolve the requested device with env-var override support."""
    override = os.environ.get("POWERZOOJAX_REQUESTED_DEVICE")
    if override is None:
        candidate = default
    else:
        candidate = override
    if candidate is None:
        return None
    value = str(candidate).strip().lower()
    if value in ("", "auto", "none", "null", "default"):
        return None
    return value


def collect_jax_run_contract(
    *,
    requested_device: str | None,
    context: str,
    declared_backend: str = "jax_rejax",
    record_device: str | None = None,
    extra_env_meta: dict[str, Any] | None = None,
    extra_labels: dict[str, Any] | None = None,
    fail_fast: bool = True,
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Collect reproducibility metadata and enforce the JAX device contract."""
    requested_raw = resolve_requested_device(requested_device)
    requested_norm = normalize_device_name(requested_raw)

    env_info = collect_env_info()
    actual_backend = env_info.get("jax_backend", "unknown")
    actual_runtime_device = normalize_device_name(actual_backend) or str(actual_backend)
    actual_device_kind = env_info.get(
        "jax_device_kind",
        env_info.get("cuda", actual_backend),
    )

    contract_ok = requested_norm is None or requested_norm == actual_runtime_device
    if fail_fast and not contract_ok:
        raise RuntimeError(
            f"{context}: requested device={requested_raw!r}, "
            f"but JAX resolved backend={actual_backend!r} "
            f"(device_kind={actual_device_kind!r})."
        )

    resolved_record_device = record_device or (
        "gpu" if actual_runtime_device == "gpu" else actual_runtime_device
    )
    env_info.update(
        {
            "declared_backend": declared_backend,
            "actual_backend": str(actual_backend),
            "declared_device": requested_raw or "auto",
            "actual_device": str(actual_runtime_device),
            "actual_device_kind": str(actual_device_kind),
            "requested_device": requested_raw or "auto",
            "requested_device_normalized": requested_norm or "auto",
            "actual_runtime_device": str(actual_runtime_device),
            "actual_runtime_backend": str(actual_backend),
            "actual_runtime_device_kind": str(actual_device_kind),
            "device_contract_ok": str(bool(contract_ok)).lower(),
        }
    )
    env_info = merge_env_info_metadata(env_info, extra_env_meta)

    labels: dict[str, Any] = {
        "runtime_family": "jax",
        "declared_backend": declared_backend,
        "actual_backend": str(actual_backend),
        "declared_device": requested_raw or "auto",
        "actual_device": str(actual_runtime_device),
        "actual_device_kind": str(actual_device_kind),
        "requested_device": requested_raw or "auto",
        "requested_device_normalized": requested_norm or "auto",
        "actual_runtime_device": str(actual_runtime_device),
        "actual_runtime_backend": str(actual_backend),
        "actual_runtime_device_kind": str(actual_device_kind),
        "device_contract_ok": bool(contract_ok),
        "device_recorded_as": str(resolved_record_device),
    }
    if extra_labels:
        labels.update(extra_labels)
    return resolved_record_device, env_info, labels


def make_run_id(task: str, algo: str, split: str, seed: int) -> str:
    """Generate a readable run id."""
    # Use microseconds so parallel train/eval jobs for the same benchmark cell
    # do not collide when they start in the same second.
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return f"{task}_{algo}_{split}_s{seed}_{ts}"


def config_hash(config: dict[str, Any]) -> str:
    """Deterministic short hash of a config dict."""
    raw = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _pickle_module():
    """Pickle backend used by dump_pickle / load_pickle.

    Prefers cloudpickle (correct for Rejax PPOState and other custom
    PyTreeNodes that the stdlib pickle cannot serialise across module
    boundaries). Falls back to stdlib pickle if cloudpickle is missing.
    """
    try:
        import cloudpickle as _m
    except ImportError:
        import pickle as _m
    return _m


def dump_pickle(obj: Any, path: str | Path) -> None:
    """Pickle ``obj`` to ``path``. Creates the parent directory if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        _pickle_module().dump(obj, f)


def load_pickle(path: str | Path) -> Any:
    """Unpickle from ``path`` using the same backend as :func:`dump_pickle`."""
    with open(path, "rb") as f:
        return _pickle_module().load(f)


def _results_dir(task_dir: str | Path) -> Path:
    return Path(task_dir) / "results"


def _runs_dir(task_dir: str | Path) -> Path:
    d = _results_dir(task_dir) / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _manifest_path(task_dir: str | Path) -> Path:
    return _results_dir(task_dir) / "manifest.json"


def has_training_artifact(artifacts: dict[str, Any] | None) -> bool:
    """Return ``True`` when an artifact dict marks a training-class record.

    Canonical single-model trainers save one of ``params`` / ``params_flax`` /
    ``params_orbax``.  Cross-backend multi-agent IL trainers may instead save a
    ``models_manifest`` that points to the per-agent checkpoint bundle.
    """
    arts = artifacts or {}
    return any(
        arts.get(key)
        for key in ("params", "params_flax", "params_orbax", "models_manifest")
    )


def save_run(record: RunRecord, task_dir: str | Path) -> Path:
    """Save a RunRecord and update the manifest. Returns the saved path."""
    runs = _runs_dir(task_dir)
    path = runs / f"{record.run_id}.json"
    path.write_text(record.to_json(), encoding="utf-8")
    _append_manifest(record, task_dir)
    return path


def load_run(run_id: str, task_dir: str | Path) -> RunRecord:
    """Load a single RunRecord by its run_id."""
    path = _runs_dir(task_dir) / f"{run_id}.json"
    return RunRecord.from_json(path.read_text(encoding="utf-8"))


def load_manifest(task_dir: str | Path) -> list[RunRecord]:
    """Load all records from the manifest, deduplicated by benchmark cell.

    Deduplication is artifact-aware (see ``dedup_keep_artifacts``): the
    record with **more** artifact keys wins; ties broken by latest
    ``timestamp``. This protects against re-runs that produce a fresher
    record but happen to have fewer (or no) artifacts saved.

    Special case: train-split policy evaluation is a different benchmark cell
    from the underlying training run, even though both share
    ``(algo, split="train", seed)``. We therefore keep those as separate
    records by treating train runs with a training artifact bundle
    (``params`` / ``params_flax`` / ``params_orbax`` / ``models_manifest``)
    as ``kind=train`` and train-split eval/baseline rows as ``kind=eval``.
    """
    mp = _manifest_path(task_dir)
    if not mp.exists():
        return []
    raw = json.loads(mp.read_text(encoding="utf-8"))
    deduped = dedup_keep_artifacts(raw)
    return [RunRecord.from_dict(d) for d in deduped]


def _parse_record_timestamp(value: Any) -> datetime | None:
    """Parse an ISO timestamp from a manifest record.

    Returns a timezone-aware UTC datetime, or ``None`` when the timestamp is
    missing / malformed.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_manifest_filtered(
    task_dir: str | Path,
    *,
    after: str | None = None,
    backend: str | None = None,
    device: str | None = None,
) -> list[RunRecord]:
    """Load manifest records filtered on raw rows before deduplication.

    This is the safe path for campaign-scoped summaries / plots: older
    artifact-rich records must not win the dedup race against newer records
    from the current campaign.
    """
    mp = _manifest_path(task_dir)
    if not mp.exists():
        return []

    raw: list[dict[str, Any]] = json.loads(mp.read_text(encoding="utf-8"))
    after_dt: datetime | None = None
    if after is not None:
        after_dt = _parse_record_timestamp(after)
        if after_dt is None:
            raise ValueError(
                f"Invalid ISO timestamp for after={after!r}; "
                "expected e.g. '2026-04-22T03:00:00+00:00'."
            )

    kept: list[dict[str, Any]] = []
    for rec in raw:
        if backend is not None and (rec.get("backend") or "jax_rejax") != backend:
            continue
        if device is not None and (rec.get("device") or "gpu") != device:
            continue
        if after_dt is not None:
            rec_dt = _parse_record_timestamp(rec.get("timestamp"))
            if rec_dt is None or rec_dt < after_dt:
                continue
        kept.append(rec)

    deduped = dedup_keep_artifacts(kept)
    return [RunRecord.from_dict(d) for d in deduped]


def _append_manifest(record: RunRecord, task_dir: str | Path) -> None:
    """Append or update a record in the manifest (keyed by run_id)."""
    mp = _manifest_path(task_dir)
    mp.parent.mkdir(parents=True, exist_ok=True)
    lock_path = mp.with_suffix(mp.suffix + ".lock")
    with lock_path.open("w", encoding="utf-8") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            if mp.exists():
                entries: list[dict[str, Any]] = json.loads(
                    mp.read_text(encoding="utf-8")
                )
            else:
                entries = []

            new_dict = record.to_dict()
            for i, e in enumerate(entries):
                if e.get("run_id") == record.run_id:
                    entries[i] = new_dict
                    break
            else:
                entries.append(new_dict)

            mp.write_text(
                json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


def save_summary(data: dict | list, task_dir: str | Path, name: str = "latest") -> Path:
    """Save a summary JSON/CSV to results/summary/."""
    summary_dir = _results_dir(task_dir) / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    path = summary_dir / f"{name}.json"
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path


def dedup_keep_artifacts(
    records: list[dict[str, Any]],
    key_fn=lambda r: (
        r.get("algo"),
        r.get("split"),
        int(r.get("seed", -1)),
        # Include backend so sb3 / sbx / jax_rejax records for the same
        # (algo, split, seed) are NOT collapsed.  Cross-backend
        # comparison records depend on this — variance_check needs to see
        # both backends as separate cells.
        r.get("backend") or "jax_rejax",
        # Include device so that jax_rejax+gpu and jax_rejax+cpu training
        # runs for the same (algo, seed) are kept as separate fair-comparison
        # cells.  Without this, the cross-device learning-curve experiment
        # would silently collapse two distinct runs into one.
        r.get("device") or "gpu",
        (
            "train"
            if r.get("split") == "train" and has_training_artifact(r.get("artifacts"))
            else "eval"
        ),
    ),
) -> list[dict[str, Any]]:
    """Collapse records by ``key_fn``; prefer the one with **more** artifact keys.

    Priority within each benchmark cell:
      1. Larger ``len(artifacts)`` wins (more saved files = more useful).
      2. Among ties, the newest ``timestamp`` wins.

    Why: train-split eval is distinct from training, but repeated eval/baseline
    reruns on the same cell should still collapse to the richest record.
    Counting artifact keys handles all variants uniformly: an empty
    ``artifacts == {}`` always loses to any record with at least one entry.

    The cell key includes ``backend`` so cross-backend records (sb3 /
    sbx / jax_rejax) for the same (algo, split, seed) survive dedup as
    distinct cells.  The key also includes ``device`` so that two
    jax_rejax runs that differ only in device (gpu vs cpu) are kept as
    separate cells — required for the cross-device learning-curve
    fair-comparison experiment.
    """
    best: dict[Any, tuple[dict[str, Any], int, str]] = {}
    for r in records:
        k = key_fn(r)
        n_arts = len(r.get("artifacts") or {})
        ts = r.get("timestamp", "")
        if k not in best:
            best[k] = (r, n_arts, ts)
            continue
        _, cur_n, cur_ts = best[k]
        if n_arts > cur_n or (n_arts == cur_n and ts > cur_ts):
            best[k] = (r, n_arts, ts)
    return [v[0] for v in best.values()]
