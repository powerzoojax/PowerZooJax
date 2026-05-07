"""Cross-backend DERs voltage_penalty must match between PowerZoo and the
PowerZooJax DERs frozen benchmark train config.

Locked to 4.0 (PowerZooJax ``benchmarks/ders/configs/train_ippo.yaml``).
PowerZoo's pre-fix default was 8.0 which left cross-backend SB3 reward
magnitudes 2x larger than the JAX side.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.common.configs import load_config
from benchmarks.common.powerzoo_repo import find_powerzoo_repo


_REPO_ROOT = Path(__file__).resolve().parents[2]
_POWERZOO_PATH = find_powerzoo_repo(_REPO_ROOT)
_DERS_TRAIN_CFG = (
    _REPO_ROOT / "benchmarks" / "ders" / "configs" / "train_ippo.yaml"
)


def _powerzoo_available() -> bool:
    return _POWERZOO_PATH is not None


def test_powerzoo_default_matches_powerzoojax_train_config():
    if not _powerzoo_available():
        pytest.skip("PowerZoo sibling repo not present")

    from powerzoo.tasks.simple.marl_ders_benchmark import MARLDERBenchmarkTask

    cfg = load_config(_DERS_TRAIN_CFG)
    expected = float(cfg.get("voltage_penalty", 4.0))

    assert MARLDERBenchmarkTask.DEFAULT_VOLTAGE_PENALTY == pytest.approx(
        expected
    ), (
        f"PowerZoo MARLDERBenchmarkTask.DEFAULT_VOLTAGE_PENALTY="
        f"{MARLDERBenchmarkTask.DEFAULT_VOLTAGE_PENALTY} but PowerZooJax "
        f"train_ippo.yaml declares voltage_penalty={expected}.  Update one "
        f"side to match the other before recording cross-backend numbers."
    )


def test_task_mapping_passes_voltage_penalty_to_factory_kwargs():
    """``powerzoo_bridge.py`` must declare voltage_penalty in DERs factory kwargs
    so the cross-backend driver passes it explicitly to PowerZoo's task
    constructor (rather than relying on the default).
    """
    from benchmarks.common.powerzoo_bridge import JAX_TASK_TO_POWERZOO_TASK

    cfg = load_config(_DERS_TRAIN_CFG)
    expected = float(cfg.get("voltage_penalty", 4.0))

    factory_kwargs = JAX_TASK_TO_POWERZOO_TASK["ders"].get(
        "powerzoo_factory_kwargs", {}
    )
    assert "voltage_penalty" in factory_kwargs, (
        "powerzoo_bridge.py::ders.powerzoo_factory_kwargs must declare "
        "voltage_penalty so cross-backend records use the same weight as "
        "the PowerZooJax DERs frozen benchmark train config."
    )
    assert factory_kwargs["voltage_penalty"] == pytest.approx(expected), (
        f"task_mapping declares voltage_penalty="
        f"{factory_kwargs['voltage_penalty']} but train_ippo.json declares "
        f"{expected}."
    )


def test_powerzoo_task_uses_passed_voltage_penalty():
    """Constructor must propagate voltage_penalty through both
    scenario_config (env reward) and agents_config (constraint penalty)."""
    if not _powerzoo_available():
        pytest.skip("PowerZoo sibling repo not present")

    from powerzoo.tasks.simple.marl_ders_benchmark import MARLDERBenchmarkTask

    task = MARLDERBenchmarkTask(voltage_penalty=4.0)
    scenario = task.get_scenario_config()
    agents = task.get_agents_config()

    assert scenario["reward"]["voltage_penalty"] == pytest.approx(4.0)
    assert agents["constraints"]["penalty_weight"] == pytest.approx(4.0)
