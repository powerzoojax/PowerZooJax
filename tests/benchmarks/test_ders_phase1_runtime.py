from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np
import pytest


def test_ders_baseline_episodes_are_not_all_identical(monkeypatch):
    from benchmarks.ders.baselines import _run_baseline_episodes
    import powerzoojax.tasks.ders as ders_mod
    from powerzoojax.case import create_case141

    def _fake_split_profiles(*_args, **_kwargs):
        load = np.linspace(0.7, 1.3, 24, dtype=np.float32)
        pv = np.linspace(0.0, 1.0, 24, dtype=np.float32)
        return load, pv

    monkeypatch.setattr(ders_mod, "load_ders_split_profiles", _fake_split_profiles)
    task = ders_mod.DERsTask(case=create_case141(), max_steps=4)
    metrics = _run_baseline_episodes(
        task=task,
        split="iid",
        algo="no_control",
        n_episodes=4,
        max_steps=4,
        seed=0,
    )

    starts = {row["episode_start"] for row in metrics}
    unique_rows = {
        (
            round(float(row["total_cost"]), 6),
            round(float(row["mean_p_loss_mw"]), 6),
            round(float(row["voltage_violation_steps"]), 6),
        )
        for row in metrics
    }
    assert len(starts) > 1
    assert len(unique_rows) > 1


def test_ders_plots_write_into_task_dir(tmp_path: Path):
    from benchmarks.ders.plots import plot_normscore_bars

    summary_dir = tmp_path / "results" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "latest.json").write_text(
        json.dumps(
            {
                "task": "ders",
                "rows": [
                    {"algo": "no_control", "split": "iid", "norm_score": 0.0},
                    {"algo": "volt_droop", "split": "iid", "norm_score": 1.0},
                ],
            }
        ),
        encoding="utf-8",
    )

    out = plot_normscore_bars(tmp_path)
    assert out == tmp_path / "results" / "figures" / "normscore_bars.pdf"
    assert out.exists()


def test_phase1_analysis_zero_policy_uses_true_pv_noop():
    from benchmarks.ders.phase1_analysis import _marl_zero_policy
    from powerzoojax.case import create_case141
    from powerzoojax.tasks.ders import make_ders_marl_env

    env, _params = make_ders_marl_env(create_case141(), max_steps=4, observation_mode="local")
    obs_dict, _state = env.reset(jax.random.PRNGKey(0))
    actions = _marl_zero_policy(env)(obs_dict)

    for name, action in actions.items():
        arr = np.asarray(action, dtype=np.float32)
        if name.startswith("renewable_"):
            assert arr[0] == pytest.approx(1.0)
            assert arr[1] == pytest.approx(0.0)
        else:
            np.testing.assert_allclose(arr, 0.0, atol=1e-6)


def test_phase1_analysis_droop_policy_curtailed_flex_on_undervoltage():
    from benchmarks.ders.phase1_analysis import _marl_volt_droop_policy
    from powerzoojax.case import create_case141
    from powerzoojax.tasks.ders import make_ders_marl_env

    env, _params = make_ders_marl_env(create_case141(), max_steps=4, observation_mode="local")
    obs_dim = env.observation_space().shape[0]
    obs_dict = {
        name: np.zeros((obs_dim,), dtype=np.float32)
        for name in env.agent_names
    }
    for name in env.agent_names:
        if name.startswith("flexload_"):
            obs_dict[name][0] = -0.5  # own_v = 0.95 p.u.

    actions = _marl_volt_droop_policy(env)(obs_dict)
    for name, action in actions.items():
        if name.startswith("flexload_"):
            assert float(np.asarray(action)[0]) > 0.0
