"""Smoke tests for DC Microgrid episode Excel export."""

from pathlib import Path

import pytest

TASK_DIR = (
    Path(__file__).resolve().parent.parent.parent / "benchmarks" / "dc_microgrid"
)
# Known training run with params.pkl in-repo (may be absent in minimal checkouts).
_PPO_RUN_ID = "dc_microgrid_ppo_train_s0_20260421_214444"


def test_obs_action_names_length():
    from benchmarks.dc_microgrid.analysis import export_episode_excel as m

    assert len(m.OBS_NAMES) == 24
    assert len(m.ACTION_NAMES) == 5


@pytest.mark.external
def test_export_ppo_writes_xlsx(tmp_path):
    pytest.importorskip("openpyxl")

    pkl = TASK_DIR / "results" / "artifacts" / f"{_PPO_RUN_ID}_params.pkl"
    if not pkl.is_file():
        pytest.skip(f"no PPO params artifact at {pkl}")

    from benchmarks.dc_microgrid.analysis.export_episode_excel import (
        export_episode_excel,
    )

    out = tmp_path / "t.xlsx"
    export_episode_excel(
        algo="ppo",
        split="iid",
        seed=0,
        run_id=_PPO_RUN_ID,
        out=out,
        episode_idx=0,
        n_episodes_span=1,
        profile_start=None,
        no_state_diag=False,
    )
    assert out.exists()
    import pandas as pd

    names = set(pd.ExcelFile(out, engine="openpyxl").sheet_names)
    assert names >= {
        "Meta",
        "Step",
        "Cumulative",
        "ObsLegend",
        "ActionLegend",
    }
    step = pd.read_excel(out, sheet_name="Step", engine="openpyxl")
    assert len(step) >= 1
    assert "p_dc_mw" in step.columns or "reward_shaped" in step.columns
