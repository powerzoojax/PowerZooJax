from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.common.powerzoo_repo import ensure_powerzoo_on_path, find_powerzoo_repo

_REPO_ROOT = Path(__file__).resolve().parents[2]
_POWERZOO_PATH = find_powerzoo_repo(_REPO_ROOT)


def _ensure_powerzoo_on_path() -> bool:
    return ensure_powerzoo_on_path(_REPO_ROOT, append=True) is not None
import torch as th


def test_tso_ppo_alignment_matches_benchmark_train_config():
    from benchmarks.common.powerzoo_bridge import _single_agent_algo_kwargs

    kwargs, source = _single_agent_algo_kwargs(
        "tso",
        "PPO",
        n_envs=4,
        extra_config=None,
    )

    assert source == "benchmarks/tso/configs/train_ppo.yaml"
    assert kwargs["learning_rate"] == 3e-4
    assert kwargs["n_steps"] == 48
    assert kwargs["batch_size"] == 48
    assert kwargs["n_epochs"] == 4
    assert kwargs["gamma"] == 0.995
    assert kwargs["gae_lambda"] == 0.95
    assert kwargs["clip_range"] == 0.2
    assert kwargs["ent_coef"] == 0.01
    assert kwargs["vf_coef"] == 0.5
    assert kwargs["max_grad_norm"] == 0.5
    assert kwargs["policy_kwargs"] == {"net_arch": [256, 256]}


def test_tso_sbx_ppo_uses_same_aligned_kwargs_as_sb3():
    from benchmarks.common.powerzoo_bridge import _single_agent_algo_kwargs

    sb3_kwargs, sb3_source = _single_agent_algo_kwargs(
        "tso",
        "PPO",
        n_envs=4,
        extra_config=None,
    )
    sbx_kwargs, sbx_source = _single_agent_algo_kwargs(
        "tso",
        "SBX_PPO",
        n_envs=4,
        extra_config=None,
    )

    assert sbx_source == sb3_source == "benchmarks/tso/configs/train_ppo.yaml"
    assert sbx_kwargs == sb3_kwargs


def test_gencos_ppo_alignment_matches_benchmark_train_config():
    from benchmarks.common.powerzoo_bridge import _single_agent_algo_kwargs

    kwargs, source = _single_agent_algo_kwargs(
        "gencos",
        "PPO",
        n_envs=1,
        extra_config=None,
    )

    assert source == "benchmarks/gencos/configs/train_ppo.yaml"
    assert kwargs["learning_rate"] == 3e-4
    assert kwargs["n_steps"] == 48
    assert kwargs["batch_size"] == 48
    assert kwargs["n_epochs"] == 4
    assert kwargs["gamma"] == 0.995
    assert kwargs["gae_lambda"] == 0.95
    assert kwargs["clip_range"] == 0.2
    assert kwargs["ent_coef"] == 0.01
    assert kwargs["vf_coef"] == 0.5
    assert kwargs["max_grad_norm"] == 0.5
    assert kwargs["policy_kwargs"] == {"net_arch": [128, 128]}


def test_explicit_single_agent_overrides_beat_alignment_but_keep_net_arch():
    from benchmarks.common.powerzoo_bridge import _single_agent_algo_kwargs

    kwargs, _ = _single_agent_algo_kwargs(
        "tso",
        "PPO",
        n_envs=4,
        extra_config={
            "n_steps": 96,
            "ent_coef": 0.0,
            "policy_kwargs": {"ortho_init": False},
            "per_agent_steps_per_round": 12345,
        },
    )

    assert kwargs["n_steps"] == 96
    assert kwargs["ent_coef"] == 0.0
    assert kwargs["policy_kwargs"]["net_arch"] == [256, 256]
    assert kwargs["policy_kwargs"]["ortho_init"] is False
    assert "per_agent_steps_per_round" not in kwargs


def test_gencos_preflight_requires_real_gb_split_metadata():
    from benchmarks.common.powerzoo_bridge import (
        CrossBackendNotComparable,
        _verify_powerzoo_env_contract,
    )

    class _BadEnv:
        data_source = "synthetic_flat"
        benchmark_split = "train"
        ood_axis = None

    with pytest.raises(CrossBackendNotComparable, match="data_source"):
        _verify_powerzoo_env_contract("gencos", _BadEnv(), split="train")


def test_resolved_total_timesteps_defaults_to_frozen_task_train_config():
    from benchmarks.common.powerzoo_bridge import _resolved_total_timesteps

    assert _resolved_total_timesteps("tso", "PPO", None) == 20_000_000
    assert _resolved_total_timesteps("tso", "PPO_LAGRANGIAN", None) == 20_000_000


def test_tso_curve_checkpoint_count_follows_eval_freq():
    from benchmarks.common.powerzoo_bridge import _tso_curve_n_checkpoints

    assert _tso_curve_n_checkpoints({"eval_freq": 100_000}, 20_000_000) == 200
    assert _tso_curve_n_checkpoints({"eval_freq": 50_000}, 2_000_000) == 40
    assert _tso_curve_n_checkpoints({}, 20_000_000) == 200


def test_save_single_agent_model_registers_params_artifact(tmp_path: Path):
    from benchmarks.common.powerzoo_bridge import _save_single_agent_model

    class _DummyModel:
        def save(self, path: str) -> None:
            Path(path).with_suffix(".zip").write_bytes(b"demo")

    arts = _save_single_agent_model(
        _DummyModel(),
        run_id="demo_run",
        artifacts_dir=tmp_path,
    )

    assert arts["params"] == "artifacts/demo_run_model.zip"
    assert arts["model_zip"] == "artifacts/demo_run_model.zip"
    assert (tmp_path / "demo_run_model.zip").exists()


def test_dso_resolved_single_agent_n_envs_defaults_to_frozen_train_config():
    from benchmarks.common.powerzoo_bridge import _resolved_single_agent_n_envs

    assert _resolved_single_agent_n_envs("dso", "PPO", 0) == 128


def test_dc_vec_env_prefers_safe_subproc_start_method(monkeypatch):
    import stable_baselines3.common.vec_env as vec_env_mod
    import benchmarks.common.powerzoo_bridge as bridge

    created = {}

    class _FakeSubprocVecEnv:
        def __init__(self, env_fns, start_method=None):
            created["n_envs"] = len(env_fns)
            created["start_method"] = start_method

    monkeypatch.setattr(bridge, "_ensure_powerzoo_path", lambda: None)
    monkeypatch.setattr(vec_env_mod, "SubprocVecEnv", _FakeSubprocVecEnv)

    vec = bridge._build_powerzoo_vec_env(
        "dc_microgrid",
        split="iid",
        seed=0,
        n_envs=2,
        vec_env="subproc",
    )

    assert isinstance(vec, _FakeSubprocVecEnv)
    assert created == {"n_envs": 2, "start_method": "forkserver"}
    assert getattr(vec, "powerzoojax_start_method") == "forkserver"


def test_dc_vec_env_required_subproc_does_not_silently_fallback(monkeypatch):
    import stable_baselines3.common.vec_env as vec_env_mod
    import benchmarks.common.powerzoo_bridge as bridge

    class _FailingSubprocVecEnv:
        def __init__(self, env_fns, start_method=None):
            raise RuntimeError(f"blocked:{start_method}")

    monkeypatch.setattr(bridge, "_ensure_powerzoo_path", lambda: None)
    monkeypatch.setattr(vec_env_mod, "SubprocVecEnv", _FailingSubprocVecEnv)

    with pytest.raises(RuntimeError, match="SubprocVecEnv required"):
        bridge._build_powerzoo_vec_env(
            "dc_microgrid",
            split="iid",
            seed=0,
            n_envs=2,
            vec_env="subproc",
        )


def test_dso_canonical_env_kwargs_include_calibrated_task_and_reset_bank():
    from benchmarks.common.powerzoo_bridge import _powerzoo_dso_env_kwargs

    kwargs = _powerzoo_dso_env_kwargs(
        "train",
        seed=7,
        use_train_reset_bank=True,
    )

    assert kwargs["max_steps"] == 48
    assert kwargs["delta_t_minutes"] == 30.0
    assert kwargs["load_scale"] == pytest.approx(0.83)
    assert kwargs["v_slack"] == pytest.approx(1.045)
    assert kwargs["v_min"] == pytest.approx(0.94)
    assert kwargs["v_max"] == pytest.approx(1.06)
    assert kwargs["shift_horizon"] == 4
    assert kwargs["preserve_feeder_totals"] is True
    assert kwargs["reset_sampling"] == "random"
    assert kwargs["reset_seed"] == 7
    assert kwargs["reset_episode_starts"] == [0, 360, 721, 1081, 1442, 1802, 2163, 2524]
    assert kwargs["flexload_config"][0]["name"] == "fl_0"
    assert kwargs["flexload_config"][0]["curtail_cap_mw"] == pytest.approx(0.165)
    assert kwargs["bus_load_scale_overrides"][18] == pytest.approx(0.56)


def test_powerzoo_dso_load_matrices_preserve_feeder_totals():
    if not _ensure_powerzoo_on_path():
        pytest.skip("PowerZoo sibling repo not present")

    from powerzoo.case.distribution import Case33bw
    from powerzoo.tasks.dso_task import DSO_FEEDER_BUS_MAP, make_dso_load_matrices

    case = Case33bw()
    feeder_shapes = {name: np.ones(96, dtype=np.float32) for name in DSO_FEEDER_BUS_MAP}
    overrides = {18: 0.5, 33: 0.5}

    load_p_base, _ = make_dso_load_matrices(
        case,
        feeder_shapes,
        max_steps=48,
        load_scale=0.83,
    )
    load_p_raw, _ = make_dso_load_matrices(
        case,
        feeder_shapes,
        max_steps=48,
        load_scale=0.83,
        bus_load_scale_overrides=overrides,
        preserve_feeder_totals=False,
    )
    load_p_preserved, _ = make_dso_load_matrices(
        case,
        feeder_shapes,
        max_steps=48,
        load_scale=0.83,
        bus_load_scale_overrides=overrides,
        preserve_feeder_totals=True,
    )

    load_bus_ids = case.loads["bus_id"].to_numpy(dtype=np.int32)
    feeder_a_mask = np.isin(load_bus_ids, np.asarray(DSO_FEEDER_BUS_MAP["feeder_A"], dtype=np.int32))
    feeder_c_mask = np.isin(load_bus_ids, np.asarray(DSO_FEEDER_BUS_MAP["feeder_C"], dtype=np.int32))

    base_a = float(load_p_base[0, feeder_a_mask].sum())
    raw_a = float(load_p_raw[0, feeder_a_mask].sum())
    preserved_a = float(load_p_preserved[0, feeder_a_mask].sum())
    base_c = float(load_p_base[0, feeder_c_mask].sum())
    raw_c = float(load_p_raw[0, feeder_c_mask].sum())
    preserved_c = float(load_p_preserved[0, feeder_c_mask].sum())

    assert raw_a < base_a
    assert raw_c < base_c
    assert preserved_a == pytest.approx(base_a, rel=1e-6, abs=1e-6)
    assert preserved_c == pytest.approx(base_c, rel=1e-6, abs=1e-6)


def test_run_one_jax_rejax_passes_temp_config_for_total_timestep_override(monkeypatch):
    import benchmarks.common.powerzoo_bridge as bridge
    from benchmarks.common.configs import load_config

    captured: dict[str, object] = {}

    def fake_run(cmd, env, cwd, check):
        cfg_path = Path(cmd[cmd.index("--config") + 1])
        captured["cmd"] = cmd
        captured["env"] = dict(env)
        captured["cwd"] = cwd
        captured["cfg"] = load_config(cfg_path)

        class _Proc:
            returncode = 0

        return _Proc()

    monkeypatch.setattr(bridge.subprocess, "run", fake_run)

    result = bridge.run_one(
        jax_task="tso",
        backend="jax_rejax",
        algo="ppo",
        seed=0,
        split="iid",
        device="cpu",
        total_timesteps=123456,
        n_envs=4,
    )

    assert result["status"] == "ok"
    assert "--config" in captured["cmd"]
    assert captured["env"]["JAX_PLATFORM_NAME"] == "cpu"
    assert captured["env"]["JAX_PLATFORMS"] == "cpu"
    assert captured["cfg"]["total_timesteps"] == 123456
    assert captured["cfg"]["n_steps"] == 48


def test_run_one_jax_rejax_tso_cpu_exports_canonical_num_envs(monkeypatch):
    import benchmarks.common.powerzoo_bridge as bridge

    captured: dict[str, object] = {}

    def fake_run(cmd, env, cwd, check):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(env)
        captured["cwd"] = cwd

        class _Proc:
            returncode = 0

        return _Proc()

    monkeypatch.setattr(bridge.subprocess, "run", fake_run)

    result = bridge.run_one(
        jax_task="tso",
        backend="jax_rejax",
        algo="ppo",
        seed=0,
        split="train",
        device="cpu",
    )

    assert result["status"] == "ok"
    assert captured["env"]["JAX_PLATFORM_NAME"] == "cpu"
    assert captured["env"]["JAX_PLATFORMS"] == "cpu"
    assert captured["env"]["TSO_CPU_NUM_ENVS"] == "256"


def test_find_powerzoo_repo_accepts_powerzoo_del_sibling(tmp_path, monkeypatch):
    from benchmarks.common.powerzoo_repo import find_powerzoo_repo

    repo_root = tmp_path / "PowerZooJax"
    repo_root.mkdir()
    alt = tmp_path / "PowerZoo.DEL"
    (alt / "powerzoo").mkdir(parents=True)

    monkeypatch.delenv("POWERZOO_DIR", raising=False)

    assert find_powerzoo_repo(repo_root) == alt.resolve()


def test_tso_powerzoo_task_kwargs_follow_frozen_benchmark_windows():
    import benchmarks.common.powerzoo_bridge as bridge

    train_kwargs, train_window = bridge._tso_powerzoo_task_kwargs("train")
    iid_kwargs, iid_window = bridge._tso_powerzoo_task_kwargs("line_tightening")

    assert train_kwargs["split"] == "train"
    assert train_window == ("2025-04-01", "2025-12-31")
    assert train_kwargs["start_date"] == "2025-04-01"
    assert train_kwargs["end_date"] == "2025-12-31"

    assert iid_kwargs["split"] == "line_tightening"
    assert iid_window == ("2026-01-01", "2026-03-31")
    assert iid_kwargs["start_date"] == "2026-01-01"
    assert iid_kwargs["end_date"] == "2026-03-31"


def test_find_powerzoo_repo_prefers_workspace_powerzoo_over_del_sibling(tmp_path, monkeypatch):
    from benchmarks.common.powerzoo_repo import find_powerzoo_repo

    repo_root = tmp_path / "PowerZooJax"
    repo_root.mkdir()
    local = repo_root / "PowerZoo"
    (local / "powerzoo").mkdir(parents=True)
    alt = tmp_path / "PowerZoo.DEL"
    (alt / "powerzoo").mkdir(parents=True)

    monkeypatch.delenv("POWERZOO_DIR", raising=False)

    assert find_powerzoo_repo(repo_root) == local.resolve()


def test_train_curve_callback_captures_tso_eval_metrics():
    import benchmarks.common.powerzoo_bridge as bridge

    callback = bridge._TrainCurveCallback(
        total_timesteps=100,
        n_checkpoints=2,
        eval_fn=lambda _model, _step: {
            "episode_reward": 2.5,
            "total_operating_cost": 123.0,
            "total_reserve_shortfall": 0.0,
            "total_thermal_overload": 1.25,
            "reserve_shortfall_rate": 0.0,
            "thermal_violation_rate": 0.125,
        },
    )

    class _FakeModel:
        ep_info_buffer = [{"r": 1.0}, {"r": 3.0}]

    callback.model = _FakeModel()
    callback._on_training_start()
    callback.num_timesteps = 50
    assert callback._on_step() is True
    callback.num_timesteps = 100
    assert callback._on_step() is True

    assert callback.eval_returns == pytest.approx([2.5, 2.5])
    assert callback.eval_total_operating_cost == pytest.approx([123.0, 123.0])
    assert callback.eval_total_reserve_shortfall == pytest.approx([0.0, 0.0])
    assert callback.eval_total_thermal_overload == pytest.approx([1.25, 1.25])
    assert callback.eval_reserve_shortfall_rate == pytest.approx([0.0, 0.0])
    assert callback.eval_thermal_violation_rate == pytest.approx([0.125, 0.125])


def test_dc_microgrid_reward_shaping_weights_match_frozen_task():
    from benchmarks.common.powerzoo_bridge import _powerzoo_reward_shaping_weights

    weights = _powerzoo_reward_shaping_weights("dc_microgrid")

    assert weights == {
        "sla": 50.0,
        "overtemp": 30.0,
        "power_deficit": 200.0,
        "power_spill": 100.0,
        "power_balance": 0.0,
        "dispatch_tracking": 40.0,
    }


def test_dc_microgrid_python_wrapper_matches_obs_shape_and_cost_surface():
    if not _ensure_powerzoo_on_path():
        pytest.skip("PowerZoo sibling repo not present")

    from benchmarks.common.powerzoo_bridge import (
        _powerzoo_dc_env_from_jax_params,
        _wrap_powerzoo_reward_shaping,
    )
    from powerzoojax.envs.microgrid import make_dcmicrogrid_params

    params = make_dcmicrogrid_params()
    env = _wrap_powerzoo_reward_shaping(
        "dc_microgrid",
        _powerzoo_dc_env_from_jax_params(params),
    )

    obs, info = env.reset(seed=0)
    assert obs.shape == (24,)
    assert env.observation_space.shape == (24,)
    assert info["step"] == 0

    action = np.zeros(5, dtype=np.float32)
    next_obs, reward, terminated, truncated, step_info = env.step(action)

    assert next_obs.shape == (24,)
    assert isinstance(float(reward), float)
    assert terminated is False
    assert isinstance(bool(truncated), bool)
    for key in (
        "reward",
        "raw_reward",
        "shaping_penalty",
        "cost_power_spill",
        "cost_power_balance",
        "cost_dispatch_tracking",
        "dispatch_target_batt_mw",
        "dispatch_target_dg_mw",
    ):
        assert key in step_info


def test_dc_microgrid_sb3_policy_spec_uses_bounded_beta_actor():
    from benchmarks.common.powerzoo_bridge import _single_agent_policy_spec

    policy_spec, metadata = _single_agent_policy_spec(
        "dc_microgrid",
        "PPO",
        backend="sb3",
    )

    assert getattr(policy_spec, "__name__", None) == "SB3BoundedBetaPolicy"
    assert metadata["requested_continuous_action_dist"] == "beta"
    assert metadata["effective_continuous_action_dist"] == "beta"
    assert metadata["policy_class"] == "SB3BoundedBetaPolicy"


def test_dc_microgrid_sbx_policy_spec_uses_bounded_beta_actor():
    from benchmarks.common.powerzoo_bridge import _single_agent_policy_spec

    policy_spec, metadata = _single_agent_policy_spec(
        "dc_microgrid",
        "SBX_PPO",
        backend="sbx",
    )

    assert getattr(policy_spec, "__name__", None) == "SBXPPOBoundedBetaPolicy"
    assert metadata["requested_continuous_action_dist"] == "beta"
    assert metadata["effective_continuous_action_dist"] == "beta"
    assert metadata["policy_class"] == "SBXPPOBoundedBetaPolicy"


def test_dc_microgrid_affine_beta_distribution_stays_within_env_bounds():
    if not _ensure_powerzoo_on_path():
        pytest.skip("PowerZoo sibling repo not present")

    from benchmarks.common.bounded_beta_policy import AffineBetaDistribution
    from benchmarks.common.powerzoo_bridge import _build_powerzoo_env

    env, kind = _build_powerzoo_env(
        "dc_microgrid",
        split="train",
        seed=0,
        strategy="seeded",
    )
    assert kind == "single"

    dist = AffineBetaDistribution(env.action_space)
    alpha_logits = th.zeros((4, int(np.prod(env.action_space.shape))), dtype=th.float32)
    beta_logits = th.zeros_like(alpha_logits)
    dist.proba_distribution(alpha_logits, beta_logits)
    actions = dist.sample()
    log_prob = dist.log_prob(actions)

    low = th.as_tensor(env.action_space.low.reshape(-1), dtype=actions.dtype)
    high = th.as_tensor(env.action_space.high.reshape(-1), dtype=actions.dtype)
    assert actions.shape == (4, low.numel())
    assert th.all(actions >= low.unsqueeze(0) - 1e-6)
    assert th.all(actions <= high.unsqueeze(0) + 1e-6)
    assert th.isfinite(log_prob).all()
    env.close()


def test_dc_microgrid_sbx_affine_beta_distribution_stays_within_env_bounds():
    if not _ensure_powerzoo_on_path():
        pytest.skip("PowerZoo sibling repo not present")

    from benchmarks.common.bounded_beta_sbx_policy import AffineBetaDiagDistribution
    from benchmarks.common.powerzoo_bridge import _build_powerzoo_env

    env, kind = _build_powerzoo_env(
        "dc_microgrid",
        split="train",
        seed=0,
        strategy="seeded",
    )
    assert kind == "single"

    low = jnp.asarray(env.action_space.low.reshape(-1), dtype=jnp.float32)
    high = jnp.asarray(env.action_space.high.reshape(-1), dtype=jnp.float32)
    alpha = jnp.full((4, low.shape[0]), 2.0, dtype=jnp.float32)
    beta = jnp.full_like(alpha, 2.0)
    dist = AffineBetaDiagDistribution(
        alpha=alpha,
        beta=beta,
        action_low=low,
        action_scale=high - low,
    )
    actions = dist.sample(seed=jax.random.PRNGKey(0))
    log_prob = dist.log_prob(actions)

    assert actions.shape == (4, low.shape[0])
    assert jnp.all(actions >= low[None, :] - 1e-6)
    assert jnp.all(actions <= high[None, :] + 1e-6)
    assert jnp.isfinite(log_prob).all()
    env.close()
