from pathlib import Path

import numpy as np

from benchmarks.common.artifacts import save_training_artifacts


def test_save_training_artifacts_sanitizes_metric_keys(tmp_path: Path):
    artifacts = save_training_artifacts(
        result_metrics={
            "mean_reward": np.array([1.0, 2.0], dtype=np.float32),
            "market/HHI": np.array([0.2, 0.3], dtype=np.float32),
        },
        run_id="demo_run",
        artifacts_dir=tmp_path,
        total_timesteps=100,
    )

    assert artifacts["mean_reward"] == "artifacts/demo_run_mean_reward.npy"
    assert artifacts["market/HHI"] == "artifacts/demo_run_market_HHI.npy"
    assert (tmp_path / "demo_run_mean_reward.npy").exists()
    assert (tmp_path / "demo_run_market_HHI.npy").exists()
