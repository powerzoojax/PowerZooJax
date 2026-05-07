"""Benchmark test markers.

Cross-backend fairness tests depend on the sibling ``PowerZoo`` repo and, in
some cases, real parquet assets.  Keep them runnable, but exclude them from
the default in-repo regression suite unless explicitly requested.
"""

from __future__ import annotations

from pathlib import Path

import pytest


_EXTERNAL_BENCHMARK_FILES = {
    "test_ders_voltage_penalty_parity.py",
    "test_driver_split_propagation.py",
    "test_split_alignment.py",
    "test_tso_comparison_parity.py",
    "test_tso_reward_parity.py",
}

_LOCAL_BENCHMARK_TESTS = {
    "test_ders_voltage_penalty_parity.py": {
        "test_task_mapping_passes_voltage_penalty_to_factory_kwargs",
    },
    "test_driver_split_propagation.py": {
        "test_build_powerzoo_env_signature_accepts_split",
        "test_ders_iid_episode_window_excludes_incomplete_tail_day",
    },
    "test_tso_comparison_parity.py": {
        "test_synthetic_helper_is_test_only",
        "test_schema_advertises_gb_real",
    },
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        path = Path(str(item.fspath))
        filename = path.name
        if path.parent.name != "benchmarks" or filename not in _EXTERNAL_BENCHMARK_FILES:
            continue
        local_tests = _LOCAL_BENCHMARK_TESTS.get(filename, set())
        if item.name not in local_tests:
            item.add_marker(pytest.mark.external)
