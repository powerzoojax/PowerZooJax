"""Cross-backend driver must propagate the user-provided ``split`` to the
underlying PowerZoo factory, otherwise ``RunRecord.split`` lies about which
data the model actually trained on.

Imports are deferred to inside each test function so that the
``benchmarks`` namespace package (no ``__init__.py`` at repo root) is
resolved through the path setup performed by ``conftest.py``.
"""

from __future__ import annotations

import json
import inspect
from pathlib import Path

import numpy as np
import pytest

from benchmarks.common.powerzoo_repo import find_powerzoo_repo

_REPO_ROOT = Path(__file__).resolve().parents[2]
_POWERZOO_PATH = find_powerzoo_repo(_REPO_ROOT)


def _powerzoo_available() -> bool:
    return _POWERZOO_PATH is not None


def test_build_powerzoo_env_signature_accepts_split():
    """``_build_powerzoo_env`` must accept a ``split`` keyword argument."""
    from benchmarks.common.powerzoo_bridge import _build_powerzoo_env

    sig = inspect.signature(_build_powerzoo_env)
    assert "split" in sig.parameters, (
        "_build_powerzoo_env must accept a `split` parameter so cross-backend "
        "training records its actual data window honestly."
    )


def test_build_powerzoo_env_dso_passes_split_through():
    """``split='iid'`` must produce different feeder shapes than
    ``split='train'`` even though the two share the same date window —
    differentiation happens at the calendar-day level."""
    if not _powerzoo_available():
        pytest.skip("PowerZoo sibling repo not present")
    pytest.importorskip("pandas")
    import numpy as np
    from benchmarks.common.powerzoo_bridge import _build_powerzoo_env

    try:
        env_train, _ = _build_powerzoo_env("dso", split="train")
        env_iid, _ = _build_powerzoo_env("dso", split="iid")
    except Exception as exc:  # pragma: no cover - depends on Ausgrid parquet
        pytest.skip(f"DSO env build requires real Ausgrid parquet: {exc}")

    obs_train, _ = env_train.reset(seed=0)
    obs_iid, _ = env_iid.reset(seed=0)

    diff = float(np.linalg.norm(np.asarray(obs_train) - np.asarray(obs_iid)))
    assert diff > 0.0, (
        "DSO obs[train] == obs[iid] — split is being silently dropped "
        "OR day-level filter is not applied on the PowerZoo side."
    )


def test_build_powerzoo_env_rejects_unknown_split():
    """Pre-flight: unknown splits must raise, not silently fall back."""
    if not _powerzoo_available():
        pytest.skip("PowerZoo sibling repo not present")
    from benchmarks.common.powerzoo_bridge import _build_powerzoo_env

    with pytest.raises((ValueError, KeyError)):
        _build_powerzoo_env("dso", split="not_a_real_split")


def test_smoke_load_powerzoo_task_passes_requested_split(monkeypatch):
    """``--dry-run --split`` must smoke-test the requested split, not train."""
    import benchmarks.common.powerzoo_bridge as bridge

    captured = {}

    class _FakeEnv:
        observation_space = "obs-space"
        action_space = "action-space"

    def fake_build(jax_task: str, *, split: str, dry_run_steps: int = 0):
        captured["jax_task"] = jax_task
        captured["split"] = split
        captured["dry_run_steps"] = dry_run_steps
        return _FakeEnv(), "single"

    monkeypatch.setattr(bridge, "_build_powerzoo_env", fake_build)

    diag = bridge.smoke_load_powerzoo_task("tso", split="load_stress")

    assert captured == {
        "jax_task": "tso",
        "split": "load_stress",
        "dry_run_steps": 10,
    }
    assert diag["split"] == "load_stress"


def test_smoke_load_powerzoo_task_reports_env_contract_metadata(monkeypatch):
    import benchmarks.common.powerzoo_bridge as bridge

    class _FakeEnv:
        observation_space = "obs-space"
        action_space = "action-space"
        data_source = "gb_real"
        benchmark_split = "iid"
        ood_axis = "demand_shift"
        profile_window = ("2026-01-01", "2026-03-31")

    def fake_build(jax_task: str, *, split: str, dry_run_steps: int = 0):
        return _FakeEnv(), "single"

    monkeypatch.setattr(bridge, "_build_powerzoo_env", fake_build)

    diag = bridge.smoke_load_powerzoo_task("gencos", split="iid")

    assert diag["data_source"] == "gb_real"
    assert diag["benchmark_split"] == "iid"
    assert diag["ood_axis"] == "demand_shift"
    assert diag["profile_window"] == ["2026-01-01", "2026-03-31"]


def test_save_gencos_eval_record_embeds_env_contract_metadata(tmp_path, monkeypatch):
    import benchmarks.common.powerzoo_bridge as bridge

    monkeypatch.setattr(bridge, "collect_env_info", lambda: {"python": "3.12.3"})

    rec = bridge._save_gencos_eval_record(
        algorithm="PPO",
        backend="sb3",
        device="cuda",
        framework_version="sb3-test",
        seed=0,
        split="renewable_shock",
        source_run_id="source_train_row",
        eval_result={
            "metrics": {"total_profit": 123.0},
            "per_episode_metrics": [{"total_profit": 123.0}],
            "per_episode_actions": [np.zeros((2, 5), dtype=np.float32)],
            "per_episode_rewards": [np.zeros(2, dtype=np.float64)],
            "env_meta": {
                "data_source": "gb_real",
                "benchmark_split": "renewable_shock",
                "ood_axis": "renewable_shock",
                "profile_window": ["2026-01-01", "2026-03-31"],
            },
        },
        task_dir=tmp_path,
    )

    assert rec.env_info["python"] == "3.12.3"
    assert rec.env_info["data_source"] == "gb_real"
    assert rec.env_info["benchmark_split"] == "renewable_shock"
    assert rec.env_info["ood_axis"] == "renewable_shock"
    assert rec.env_info["profile_window"] == "2026-01-01..2026-03-31"
    assert rec.labels["backend_family"] == "python_torch"
    assert rec.labels["algo_family"] == "frozen_self_play_il"
    assert rec.labels["cross_backend_gap_note"] == "frozen_self_play_il"
    assert rec.labels["source_run_id"] == "source_train_row"

    saved = json.loads(
        (tmp_path / "results" / "runs" / f"{rec.run_id}.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved["env_info"]["data_source"] == "gb_real"
    assert saved["env_info"]["benchmark_split"] == "renewable_shock"
    assert saved["env_info"]["profile_window"] == "2026-01-01..2026-03-31"
    assert saved["labels"]["backend_family"] == "python_torch"


def test_ders_iid_episode_window_excludes_incomplete_tail_day():
    """DER iid start-day sampling must ignore the last partial day.

    The PowerZoo DER iid split currently contains 5039 half-hour samples:
    104 full days plus a 47-step tail.  A 48-step benchmark episode cannot
    start on day 104, otherwise the final step reads beyond the PV trace and
    logs ``flat_idx=5039 out of data range``.
    """
    if not _powerzoo_available():
        pytest.skip("PowerZoo sibling repo not present")

    from benchmarks.common.powerzoo_bridge import _build_powerzoo_env

    env, kind = _build_powerzoo_env("ders", split="iid")
    assert kind == "pettingzoo"

    base_env = env.base_env
    grid = base_env.grid
    total_days, steps_per_day, required_days, max_start_day = base_env._episode_day_window()

    assert len(grid._time_series_data) == 5039
    assert steps_per_day == 48
    assert required_days == 1
    assert total_days == 105
    assert max_start_day == 103, (
        "The final iid day is incomplete (5039 = 104*48 + 47), so a full "
        "48-step episode must not start on day 104."
    )
    assert base_env._validate_start_day(103) == 103
    with pytest.raises(ValueError, match=r"valid range is \[0, 103\]"):
        base_env._validate_start_day(104)
