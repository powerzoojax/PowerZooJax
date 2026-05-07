from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from powerzoojax.data.splits import (
    GB_IID_END,
    GB_IID_START,
    GB_TRAIN_END,
    GB_TRAIN_START,
)


def _minimal_tso_task_cfg() -> dict:
    return {
        "task": "tso",
        "case": "case118",
        "n_units": 54,
        "n_buses": 118,
        "n_lines": 186,
        "data_source": "gb",
        "max_steps": 48,
        "dt_hours": 0.25,
        "reserve_margin_frac": 0.07,
        "reward_scale": 2.5e-4,
        "cost_thermal_weight": 3.0,
        "solver_mode": 1,
        "forecast_horizon_steps": 4,
        "eval_episodes": 77,
        "eval_splits": ["iid", "line_tightening"],
        "baseline_set": ["merit_order", "all_on"],
    }


def test_make_task_from_config_activates_task_runtime_fields():
    from benchmarks.tso.config_runtime import make_task_from_config

    task = make_task_from_config(_minimal_tso_task_cfg())
    raw = np.zeros((48, 3), dtype=np.float32)
    params = task._build_params(raw, 0, 48)

    assert float(params.delta_t_hours) == 0.25
    assert float(params.reserve_margin_frac) == 0.07
    assert float(params.reward_scale) == 2.5e-4
    assert float(params.cost_thermal_weight) == 3.0
    assert int(params.solver_mode) == 1
    assert int(params.forecast_horizon_steps) == 4


def test_make_tso_case118_params_synthesizes_benchmark_line_caps_when_case_has_none():
    from powerzoojax.case import load_case
    from powerzoojax.tasks.tso import (
        _TSO_SYNTHETIC_LINE_LIMIT_DEGREES,
        make_tso_case118_params,
    )

    raw_case = load_case("118")
    params = make_tso_case118_params(max_steps=4)

    raw_caps = np.asarray(raw_case.line_cap)
    cap = np.asarray(params.case.line_cap)
    floor = np.asarray(params.case.line_floor)
    rate_a = np.asarray(params.case.line_rate_a)
    expected = (
        float(raw_case.base_mva)
        * np.deg2rad(_TSO_SYNTHETIC_LINE_LIMIT_DEGREES)
        / np.abs(np.asarray(raw_case.line_x))
    )

    assert np.min(raw_caps) >= 1e5
    assert np.allclose(cap, expected)
    assert np.allclose(floor, -expected)
    assert np.allclose(rate_a, expected)
    assert np.max(cap) < 1e5


def test_make_task_from_config_scales_executable_line_caps():
    from benchmarks.tso.config_runtime import make_task_from_config

    task = make_task_from_config(_minimal_tso_task_cfg(), line_rating_scale=0.85)
    raw = np.zeros((48, 3), dtype=np.float32)
    params = task._build_params(raw, 0, 48)
    base_task = make_task_from_config(_minimal_tso_task_cfg(), line_rating_scale=1.0)
    base_params = base_task._build_params(raw, 0, 48)

    assert np.allclose(
        np.asarray(params.case.line_cap),
        np.asarray(base_params.case.line_cap) * 0.85,
    )
    assert np.allclose(
        np.asarray(params.case.line_floor),
        np.asarray(base_params.case.line_floor) * 0.85,
    )


def test_resolve_gb_windows_falls_back_to_frozen_defaults():
    from benchmarks.tso.config_runtime import resolve_gb_windows

    assert resolve_gb_windows(_minimal_tso_task_cfg()) == {
        "train": (GB_TRAIN_START, GB_TRAIN_END),
        "iid": (GB_IID_START, GB_IID_END),
    }


@pytest.mark.parametrize(
    ("split", "expected_start", "expected_end"),
    [
        ("train", "2025-05-01", "2025-11-30"),
        ("load_stress", "2026-02-01", "2026-04-15"),
    ],
)
def test_make_task_from_config_routes_resolved_gb_windows_to_dataloader(
    monkeypatch,
    split: str,
    expected_start: str,
    expected_end: str,
):
    from benchmarks.tso.config_runtime import make_task_from_config

    cfg = _minimal_tso_task_cfg()
    cfg.update(
        {
            "gb_train_start": "2025-05-01",
            "gb_train_end": "2025-11-30",
            "gb_iid_start": "2026-02-01",
            "gb_iid_end": "2026-04-15",
        }
    )
    captured: dict[str, object] = {}

    def fake_load_jax_profiles(
        self,
        signals,
        *,
        start_date=None,
        end_date=None,
        resample=None,
        **_kwargs,
    ):
        captured["signals"] = tuple(signals)
        captured["start_date"] = start_date
        captured["end_date"] = end_date
        captured["resample"] = resample
        return np.ones((96, 3), dtype=np.float32)

    monkeypatch.setattr(
        "powerzoojax.data.data_loader.DataLoader.load_jax_profiles",
        fake_load_jax_profiles,
    )

    task = make_task_from_config(cfg)
    monkeypatch.setattr(task, "_build_params", lambda raw, t0, t1: (raw.shape, t0, t1))

    shape, t0, t1 = task.episode_params(split, 0, 1, 48)

    assert captured == {
        "signals": (
            "load.actual_mw",
            "wind.available_mw",
            "solar.available_mw",
        ),
        "start_date": expected_start,
        "end_date": expected_end,
        "resample": "30min",
    }
    assert shape == (96, 3)
    assert (t0, t1) == (0, 48)


def test_validate_task_config_rejects_metadata_drift():
    from benchmarks.tso.config_runtime import validate_task_config

    cfg = _minimal_tso_task_cfg()
    cfg["n_units"] = 999

    with pytest.raises(ValueError, match="n_units"):
        validate_task_config(cfg)


def test_get_eval_episodes_uses_task_yaml_and_rejects_split_drift():
    from benchmarks.tso.config_runtime import get_eval_episodes

    task_cfg = _minimal_tso_task_cfg()

    assert get_eval_episodes(task_cfg, {"split": "iid"}) == 77
    assert get_eval_episodes(task_cfg, {"split": "iid", "n_eval_episodes": 77}) == 77

    with pytest.raises(ValueError, match="eval episode drift"):
        get_eval_episodes(task_cfg, {"split": "iid", "n_eval_episodes": 50})


def test_get_eval_gb_split_rejects_yaml_drift():
    from benchmarks.tso.config_runtime import get_eval_gb_split

    assert get_eval_gb_split("load_stress", {"split": "load_stress", "gb_split": "iid"}) == "iid"

    with pytest.raises(ValueError, match="gb_split drift"):
        get_eval_gb_split("line_tightening", {"split": "line_tightening", "gb_split": "train"})


def test_run_all_baselines_uses_task_config_baselines_and_splits(monkeypatch):
    import benchmarks.tso.baselines as baselines

    calls: list[tuple[str, str, int]] = []

    monkeypatch.setattr(
        baselines,
        "load_task_config",
        lambda _task_dir: _minimal_tso_task_cfg(),
    )
    monkeypatch.setattr(
        baselines,
        "run_single_baseline",
        lambda task_dir, algo, seed, split: calls.append((algo, split, seed)) or {
            "algo": algo,
            "split": split,
            "seed": seed,
        },
    )

    baselines.run_all_baselines(Path("."), seeds=[5], splits=None)

    assert calls == [
        ("merit_order", "iid", 5),
        ("all_on", "iid", 5),
        ("merit_order", "line_tightening", 5),
        ("all_on", "line_tightening", 5),
    ]


# ── Penalty PPO configs ────────────────────────────────────────────────────────

_TSO_TASK_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "tso"


@pytest.mark.parametrize(
    ("yaml_name", "expected_lambda"),
    [
        ("train_ppo_penalty_l10.yaml", 10.0),
        ("train_ppo_penalty_l100.yaml", 100.0),
        ("train_ppo_penalty_l1000.yaml", 1000.0),
    ],
)
def test_penalty_config_fields(yaml_name: str, expected_lambda: float):
    """Penalty configs have correct algo, wrapper, and penalty_lambda fields."""
    from benchmarks.common.configs import load_train_config

    config_path = str(_TSO_TASK_DIR / "configs" / yaml_name)
    config = load_train_config(_TSO_TASK_DIR, "ppo_penalty", config_path)
    assert config["algo"] == "ppo", "penalty config must declare algo: ppo"
    assert config.get("wrapper") == "penalty", "penalty config must have wrapper: penalty"
    assert float(config["penalty_lambda"]) == expected_lambda


def test_penalty_reward_wrapper_construction():
    """PenaltyRewardWrapper can wrap UnitCommitmentEnv and call reset without error."""
    import jax
    from benchmarks.tso.config_runtime import make_task_from_config
    from powerzoojax.envs.grid.unit_commitment import UnitCommitmentEnv
    from powerzoojax.rl.wrappers import LogWrapper, PenaltyRewardWrapper

    task_cfg = _minimal_tso_task_cfg()
    task = make_task_from_config(task_cfg)
    raw = np.zeros((48, 3), dtype=np.float32)
    params = task._build_params(raw, 0, 48)

    env = UnitCommitmentEnv()
    wrapper = LogWrapper(
        PenaltyRewardWrapper(env, penalty_lambda=100.0, reward_scale=1e-4),
        params,
    )

    key = jax.random.PRNGKey(0)
    obs, state = wrapper.reset(key)
    assert obs.shape[0] == wrapper.obs_size


def test_penalty_reward_wrapper_penalises_costs():
    """PenaltyRewardWrapper reduces the reward by lambda*scale*sum(costs)."""
    import jax
    import jax.numpy as jnp
    from benchmarks.tso.config_runtime import make_task_from_config
    from powerzoojax.envs.grid.unit_commitment import UnitCommitmentEnv
    from powerzoojax.rl.wrappers import PenaltyRewardWrapper

    task_cfg = _minimal_tso_task_cfg()
    task = make_task_from_config(task_cfg)
    raw = np.zeros((48, 3), dtype=np.float32)
    params = task._build_params(raw, 0, 48)

    env = UnitCommitmentEnv()
    penalty_lambda = 100.0
    reward_scale = 1e-4
    wrapped = PenaltyRewardWrapper(env, penalty_lambda=penalty_lambda, reward_scale=reward_scale)

    key = jax.random.PRNGKey(42)
    key_r, key_s = jax.random.split(key)
    _, state = env.reset(key_r, params)

    # Use all-off action to provoke non-zero constraint costs.
    action = jnp.zeros(2 * params.case.n_units, dtype=jnp.float32)
    obs_w, _, rew_w, costs_w, _, info_w = wrapped.step(key_s, state, action, params)
    obs_r, _, rew_r, costs_r, _, _ = env.step(key_s, state, action, params)

    expected_penalty = float(penalty_lambda * reward_scale * jnp.sum(costs_r))
    actual_gap = float(info_w["unpenalized_reward"]) - float(rew_w)

    assert np.isclose(actual_gap, expected_penalty, atol=1e-5), (
        f"reward gap {actual_gap:.6f} != expected penalty {expected_penalty:.6f}"
    )


def test_tso_training_params_enable_reset_sampling(monkeypatch):
    from benchmarks.tso.config_runtime import make_task_from_config

    task = make_task_from_config(_minimal_tso_task_cfg())
    raw = np.stack(
        [
            np.linspace(1000.0, 1200.0, 128, dtype=np.float32),
            np.linspace(200.0, 100.0, 128, dtype=np.float32),
            np.linspace(50.0, 20.0, 128, dtype=np.float32),
        ],
        axis=1,
    )
    monkeypatch.setattr(task, "_get_raw", lambda split: raw)

    params = task.training_params(max_steps=48)
    assert params.sample_start_on_reset is True
    assert params.max_steps == 48
    assert params.load_profiles.shape[0] == 128
