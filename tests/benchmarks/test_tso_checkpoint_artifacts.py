from __future__ import annotations

from benchmarks.common.io import load_pickle
from benchmarks.tso.checkpoints import (
    load_checkpoint_params,
    load_checkpoint_specs,
    save_checkpoint_bundle,
)


def test_save_and_load_checkpoint_bundle(tmp_path):
    artifacts_dir = tmp_path / "results" / "artifacts"
    manifest_rel = save_checkpoint_bundle(
        run_id="tso_ppo_lagrangian_train_s0_dummy",
        checkpoints=[
            (100, {"params": {"w": 1.0}}),
            (200, {"params": {"w": 2.0}}),
        ],
        artifacts_dir=artifacts_dir,
    )

    specs = load_checkpoint_specs(tmp_path, manifest_rel)
    assert [spec["index"] for spec in specs] == [0, 1]
    assert [spec["timesteps"] for spec in specs] == [100, 200]

    params0, spec0 = load_checkpoint_params(tmp_path, manifest_rel, 0)
    params1, spec1 = load_checkpoint_params(tmp_path, manifest_rel, 1)

    assert spec0["timesteps"] == 100
    assert spec1["timesteps"] == 200
    assert params0 == {"params": {"w": 1.0}}
    assert params1 == {"params": {"w": 2.0}}

    assert load_pickle(tmp_path / "results" / specs[0]["params"]) == params0
