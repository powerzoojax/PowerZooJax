"""Idle-GPU selection helpers for isolated benchmark campaigns."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class GPUStatus:
    index: int
    memory_used_mb: int
    utilization_gpu: int
    compute_process_count: int = 0


def rank_idle_gpus(
    statuses: list[GPUStatus],
    *,
    utilization_threshold: int = 10,
    memory_threshold_mb: int = 2048,
    max_compute_process_count: int | None = None,
) -> list[GPUStatus]:
    """Return idle candidates sorted from most to least free."""
    candidates = [
        status
        for status in statuses
        if status.utilization_gpu <= int(utilization_threshold)
        and status.memory_used_mb <= int(memory_threshold_mb)
        and (
            max_compute_process_count is None
            or status.compute_process_count <= int(max_compute_process_count)
        )
    ]
    return sorted(
        candidates,
        key=lambda status: (
            int(status.compute_process_count),
            int(status.utilization_gpu),
            int(status.memory_used_mb),
            int(status.index),
        ),
    )


def select_idle_gpu(
    statuses: list[GPUStatus],
    *,
    utilization_threshold: int = 10,
    memory_threshold_mb: int = 2048,
    max_compute_process_count: int | None = None,
) -> GPUStatus | None:
    ranked = rank_idle_gpus(
        statuses,
        utilization_threshold=utilization_threshold,
        memory_threshold_mb=memory_threshold_mb,
        max_compute_process_count=max_compute_process_count,
    )
    return ranked[0] if ranked else None


def _run_nvidia_smi(query: str) -> str:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        check=True,
        text=True,
        timeout=10,
    )
    return result.stdout


def _run_nvidia_smi_compute_apps() -> str:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=10,
    )
    return result.stdout if result.returncode == 0 else ""


def query_gpu_statuses() -> list[GPUStatus]:
    """Read current GPU memory / utilisation plus compute-process counts."""
    gpu_rows = _run_nvidia_smi("index,uuid,memory.used,utilization.gpu")
    process_rows = _run_nvidia_smi_compute_apps()

    process_counts: dict[str, int] = {}
    for raw_line in process_rows.splitlines():
        line = raw_line.strip()
        if not line or "No running" in line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        gpu_uuid = parts[0]
        process_counts[gpu_uuid] = process_counts.get(gpu_uuid, 0) + 1

    statuses: list[GPUStatus] = []
    for raw_line in gpu_rows.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            raise ValueError(f"Unexpected nvidia-smi row: {line!r}")
        index_s, uuid_s, memory_s, util_s = parts
        statuses.append(
            GPUStatus(
                index=int(index_s),
                memory_used_mb=int(memory_s),
                utilization_gpu=int(util_s),
                compute_process_count=int(process_counts.get(uuid_s, 0)),
            )
        )
    return statuses


def _append_probe_log(
    log_path: Path,
    *,
    statuses: list[GPUStatus],
    selected: GPUStatus | None,
) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "selected_gpu": None if selected is None else selected.index,
        "gpus": [asdict(status) for status in statuses],
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def wait_for_idle_gpu(
    *,
    utilization_threshold: int = 10,
    memory_threshold_mb: int = 2048,
    max_compute_process_count: int | None = None,
    poll_interval_s: int = 60,
    probe_log_path: str | Path | None = None,
) -> GPUStatus:
    """Poll until a GPU satisfies the idle thresholds, then return it."""
    log_path = Path(probe_log_path) if probe_log_path is not None else None
    while True:
        statuses = query_gpu_statuses()
        selected = select_idle_gpu(
            statuses,
            utilization_threshold=utilization_threshold,
            memory_threshold_mb=memory_threshold_mb,
            max_compute_process_count=max_compute_process_count,
        )
        if log_path is not None:
            _append_probe_log(log_path, statuses=statuses, selected=selected)
        if selected is not None:
            return selected
        time.sleep(int(poll_interval_s))
