from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.common.powerzoo_repo import find_powerzoo_repo

_REPO_ROOT = Path(__file__).resolve().parents[2]
_POWERZOO_PATH = find_powerzoo_repo(_REPO_ROOT)


def _powerzoo_available() -> bool:
    return _POWERZOO_PATH is not None


def test_ders_task_defaults_match_canonical_cross_backend_config():
    from benchmarks.common.configs import load_task_config

    cfg = load_task_config(_REPO_ROOT / "benchmarks" / "ders")

    assert cfg["case"] == "case141"
    assert cfg["voltage_penalty"] == pytest.approx(4.0)
    assert cfg["num_envs"] == 128


def test_cross_backend_record_split_keeps_il_training_rows_on_train_split():
    from benchmarks.common.powerzoo_bridge import _cross_backend_record_split

    assert _cross_backend_record_split(
        "ders",
        requested_split="iid",
        train_split="train",
        env_kind="pettingzoo",
    ) == "train"
    assert _cross_backend_record_split(
        "gencos",
        requested_split="iid",
        train_split="train",
        env_kind="pettingzoo",
    ) == "train"
    assert _cross_backend_record_split(
        "dso",
        requested_split="iid",
        train_split="train",
        env_kind="single",
    ) == "iid"


def test_save_il_models_manifest_keeps_legacy_params_alias(tmp_path: Path):
    from benchmarks.common.powerzoo_bridge import _save_il_models_manifest

    class _DummyModel:
        def save(self, path: str) -> None:
            Path(f"{path}.zip").write_bytes(b"dummy")

    artifacts = _save_il_models_manifest(
        {"bat_0": _DummyModel()},
        run_id="ders_ppo_train_s0_demo",
        artifacts_dir=tmp_path,
    )

    assert artifacts["params"] == artifacts["models_manifest"]
    manifest_path = tmp_path / "ders_ppo_train_s0_demo_models_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["agents"]["bat_0"] == "artifacts/ders_ppo_train_s0_demo_bat_0.zip"


def test_ders_hypothesis_tests_pair_primary_per_episode_metrics(tmp_path: Path):
    from benchmarks.common.io import RunRecord
    from benchmarks.ders.summarize import _build_hypothesis_tests

    artifacts_dir = tmp_path / "results" / "artifacts"
    artifacts_dir.mkdir(parents=True)

    def _write(run_id: str, vals: list[float]) -> str:
        rel = f"artifacts/{run_id}_per_episode.json"
        path = tmp_path / "results" / rel
        path.write_text(
            json.dumps([{"mean_p_loss_mw": v} for v in vals], indent=2),
            encoding="utf-8",
        )
        return rel

    records = [
        RunRecord(
            task="ders",
            variant="ders_12agent",
            algo="ippo",
            seed=0,
            run_id="ippo_s0",
            split="iid",
            artifacts={"per_episode": _write("ippo_s0", [0.18, 0.19])},
        ),
        RunRecord(
            task="ders",
            variant="ders_12agent",
            algo="no_control",
            seed=0,
            run_id="nc_s0",
            split="iid",
            artifacts={"per_episode": _write("nc_s0", [0.20, 0.21])},
        ),
    ]

    tests = _build_hypothesis_tests(
        tmp_path,
        records,
        {
            "primary_split": "iid",
            "target_return_metric_key": "mean_p_loss_mw",
            "target_metric_direction": "lower_is_better",
        },
    )
    ippo_vs_nc = next(
        t for t in tests if t["left_algo"] == "ippo" and t["right_algo"] == "no_control"
    )
    assert ippo_vs_nc["n_pairs"] == 2
    assert ippo_vs_nc["common_seeds"] == [0]
    assert ippo_vs_nc["mean_left_minus_right"] < 0.0


def test_build_powerzoo_ders_uses_case141_and_canonical_bus_layout():
    if not _powerzoo_available():
        pytest.skip("PowerZoo sibling repo not present")

    from benchmarks.common.powerzoo_bridge import _build_powerzoo_env

    env, kind = _build_powerzoo_env("ders", split="iid", seed=0)
    try:
        assert kind == "pettingzoo"
        assert type(env.base_env.grid.case).__name__ == "Case141"
        scenario = getattr(getattr(env, "_inner", env), "_scenario_config")
        resources = list(scenario["resources"])
        assert [r["bus_id"] for r in resources if r["type"] == "battery"] == [9, 55, 17, 122]
        assert [r["bus_id"] for r in resources if r["type"] == "solar"] == [6, 73, 72, 82]
        assert [r["bus_id"] for r in resources if r["type"] == "flexload"] == [41, 70, 135, 24]
    finally:
        try:
            env.close()
        except Exception:
            pass


def test_build_powerzoo_ders_wrapper_reports_shaped_reward_and_episode_metadata():
    if not _powerzoo_available():
        pytest.skip("PowerZoo sibling repo not present")

    from benchmarks.common.powerzoo_bridge import _build_powerzoo_env

    env, kind = _build_powerzoo_env("ders", split="voltage_tightening", seed=7)
    assert kind == "pettingzoo"
    try:
        _obs, infos = env.reset(seed=123)
        first_agent = env.possible_agents[0]
        assert infos[first_agent]["split"] == "voltage_tightening"
        zero_action = {
            agent: np.zeros(env.action_space(agent).shape, dtype=np.float32)
            for agent in env.possible_agents
        }
        _obs, rewards, _terms, _truncs, infos = env.step(zero_action)
        info = infos[first_agent]
        assert float(info["episode_start"]) >= 0.0
        assert np.isfinite(float(info["p_loss_MW"]))
        assert float(info["v_min_step"]) <= float(info["v_max_step"])
        expected_reward = float(info["raw_reward"]) - 4.0 * float(info["cost_continuous"])
        assert float(rewards[first_agent]) == pytest.approx(expected_reward)
        assert float(info["reward"]) == pytest.approx(expected_reward)
    finally:
        try:
            env.close()
        except Exception:
            pass
