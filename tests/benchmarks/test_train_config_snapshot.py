import json
from types import SimpleNamespace

from benchmarks.common.configs import (
    load_task_config_for_run,
    load_train_config_for_run,
)


def test_load_train_config_for_run_prefers_snapshot(tmp_path):
    task_dir = tmp_path / "tso"
    (task_dir / "configs").mkdir(parents=True)
    (task_dir / "results" / "artifacts").mkdir(parents=True)

    (task_dir / "configs" / "train_saute_ppo.yaml").write_text(
        "algo: saute_ppo\nsaute_budget: 100.0\nsaute_unsafe_reward: -20.0\n",
        encoding="utf-8",
    )
    snapshot = {
        "train_config_raw": {
            "algo": "saute_ppo",
            "cost_thresholds": [10.0, 100.0],
            "saute_unsafe_reward": -30.0,
        }
    }
    (task_dir / "results" / "artifacts" / "run_config.json").write_text(
        json.dumps(snapshot),
        encoding="utf-8",
    )

    run_record = SimpleNamespace(
        algo="saute_ppo",
        artifacts={"config": "artifacts/run_config.json"},
    )
    cfg = load_train_config_for_run(
        task_dir,
        run_record,
        algo_key_map={"saute_ppo": "saute_ppo"},
        default_key="saute_ppo",
    )

    assert cfg["cost_thresholds"] == [10.0, 100.0]
    assert cfg["saute_unsafe_reward"] == -30.0


def test_load_train_config_for_run_falls_back_to_train_config_snapshot(tmp_path):
    task_dir = tmp_path / "gencos"
    (task_dir / "configs").mkdir(parents=True)
    (task_dir / "results" / "artifacts").mkdir(parents=True)

    (task_dir / "configs" / "train_ippo.yaml").write_text(
        "hidden_dims: [64, 64]\n",
        encoding="utf-8",
    )
    snapshot = {"train_config": {"hidden_dims": [128, 128], "total_timesteps": 1234}}
    (task_dir / "results" / "artifacts" / "run_config.json").write_text(
        json.dumps(snapshot),
        encoding="utf-8",
    )

    run_record = SimpleNamespace(
        algo="ippo",
        artifacts={"config": "artifacts/run_config.json"},
    )
    cfg = load_train_config_for_run(task_dir, run_record, default_key="ippo")

    assert cfg["hidden_dims"] == [128, 128]
    assert cfg["total_timesteps"] == 1234


def test_load_task_config_for_run_prefers_task_snapshot(tmp_path):
    task_dir = tmp_path / "dc_microgrid"
    (task_dir / "configs").mkdir(parents=True)
    (task_dir / "results" / "artifacts").mkdir(parents=True)

    (task_dir / "configs" / "task.yaml").write_text(
        "max_steps: 288\ndata_source: google\n",
        encoding="utf-8",
    )
    snapshot = {"task_config": {"max_steps": 96, "data_source": "frozen_snapshot"}}
    (task_dir / "results" / "artifacts" / "run_config.json").write_text(
        json.dumps(snapshot),
        encoding="utf-8",
    )

    run_record = SimpleNamespace(artifacts={"config": "artifacts/run_config.json"})
    cfg = load_task_config_for_run(task_dir, run_record)

    assert cfg["max_steps"] == 96
    assert cfg["data_source"] == "frozen_snapshot"
