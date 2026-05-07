from __future__ import annotations

from benchmarks.common.gpu_selector import GPUStatus, rank_idle_gpus, select_idle_gpu


def test_rank_idle_gpus_filters_and_sorts_by_util_mem_process():
    statuses = [
        GPUStatus(index=0, memory_used_mb=1500, utilization_gpu=5, compute_process_count=1),
        GPUStatus(index=1, memory_used_mb=500, utilization_gpu=12, compute_process_count=0),
        GPUStatus(index=2, memory_used_mb=500, utilization_gpu=5, compute_process_count=2),
        GPUStatus(index=3, memory_used_mb=3000, utilization_gpu=0, compute_process_count=0),
        GPUStatus(index=4, memory_used_mb=700, utilization_gpu=3, compute_process_count=0),
        GPUStatus(index=5, memory_used_mb=400, utilization_gpu=5, compute_process_count=0),
    ]

    ranked = rank_idle_gpus(statuses)

    assert [gpu.index for gpu in ranked] == [4, 5, 0, 2]


def test_select_idle_gpu_can_require_zero_compute_processes():
    statuses = [
        GPUStatus(index=0, memory_used_mb=300, utilization_gpu=2, compute_process_count=1),
        GPUStatus(index=1, memory_used_mb=500, utilization_gpu=4, compute_process_count=0),
    ]

    picked = select_idle_gpu(statuses, max_compute_process_count=0)

    assert picked is not None
    assert picked.index == 1


def test_select_idle_gpu_returns_none_when_everything_is_busy():
    statuses = [
        GPUStatus(index=0, memory_used_mb=2500, utilization_gpu=0, compute_process_count=0),
        GPUStatus(index=1, memory_used_mb=100, utilization_gpu=99, compute_process_count=0),
    ]
    assert select_idle_gpu(statuses) is None
