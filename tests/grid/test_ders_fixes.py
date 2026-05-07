"""Targeted regression / alignment tests for the DERs benchmark fixes.

Covers the specific deviations identified in the audit:
  A. Safety metrics no longer hardcode DERS_V_MIN/MAX — they respect params
  B. DistGridMARLEnv local obs is strictly smaller than full-grid obs
  C. ders-medium-safe preset no longer claims IPPO-Lagrangian
  D. phase_swap entry raises NotImplementedError (blocker documented)
     make_ders_3phase_eval() smoke test (3-phase eval scaffold exists)
  E. make_ders_params_with_profiles data hook works
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powerzoojax.case import create_case141


# ========================== Fixtures ==========================


@pytest.fixture(scope="module")
def case141():
    return create_case141()


@pytest.fixture(scope="module")
def base_params(case141):
    from powerzoojax.tasks.ders import make_ders_params
    return make_ders_params(case141, max_steps=6)


@pytest.fixture(scope="module")
def base_env():
    from powerzoojax.envs.grid.dist import DistGridEnv
    return DistGridEnv()


@pytest.fixture
def key():
    return jax.random.PRNGKey(99)


# ========================== A — Safety metric thresholds ==========================


def test_safety_metrics_respects_tighter_v_min(base_env, base_params, key):
    """compute_ders_safety_metrics uses the passed v_min, not the hardcoded constant."""
    from powerzoojax.tasks.ders import (
        ders_no_control_rollout,
        compute_ders_safety_metrics,
        DERS_V_MIN, DERS_V_MAX,
    )
    rollout = ders_no_control_rollout(base_env, base_params, key)

    # Standard thresholds
    m_base = compute_ders_safety_metrics(rollout, v_min=DERS_V_MIN, v_max=DERS_V_MAX)
    # Tighter thresholds — same physical rollout but stricter evaluation
    m_tight = compute_ders_safety_metrics(rollout, v_min=0.96, v_max=1.04)

    # Stricter bounds must produce ≥ as many violations (safety_rate ≤)
    assert m_tight["voltage_safety_rate"] <= m_base["voltage_safety_rate"] + 1e-6, (
        f"Tighter bounds ({0.96},{1.04}) should give ≤ safety rate than "
        f"({DERS_V_MIN},{DERS_V_MAX}) on the same rollout, got "
        f"{m_tight['voltage_safety_rate']:.3f} > {m_base['voltage_safety_rate']:.3f}"
    )


def test_safety_metrics_max_deviation_uses_v_min_param(base_env, base_params, key):
    """max_undervoltage_dev is measured relative to the passed v_min, not DERS_V_MIN."""
    from powerzoojax.tasks.ders import (
        ders_no_control_rollout,
        compute_ders_safety_metrics,
        DERS_V_MIN,
    )
    rollout = ders_no_control_rollout(base_env, base_params, key)
    v_min_ep = np.asarray(rollout["v_min_episode"])

    m_default = compute_ders_safety_metrics(rollout)  # uses DERS_V_MIN
    m_custom = compute_ders_safety_metrics(rollout, v_min=0.97)

    # Recompute expected max_undervoltage_dev from raw arrays
    expected_default = float(np.max(np.maximum(DERS_V_MIN - v_min_ep, 0.0)))
    expected_custom = float(np.max(np.maximum(0.97 - v_min_ep, 0.0)))

    assert abs(m_default["max_undervoltage_dev"] - expected_default) < 1e-6
    assert abs(m_custom["max_undervoltage_dev"] - expected_custom) < 1e-6


def test_compute_ders_metrics_respects_v_min(base_env, base_params, key):
    """compute_ders_metrics violation count changes when v_min changes."""
    from powerzoojax.tasks.ders import (
        ders_no_control_rollout,
        compute_ders_metrics,
        DERS_V_MIN, DERS_V_MAX,
    )
    rollout = ders_no_control_rollout(base_env, base_params, key)

    m_base = compute_ders_metrics(rollout, rollout, v_min=DERS_V_MIN, v_max=DERS_V_MAX)
    m_tight = compute_ders_metrics(rollout, rollout, v_min=0.96, v_max=1.04)

    # Tighter bounds → at least as many violations
    assert m_tight["voltage_violation_steps"] >= m_base["voltage_violation_steps"]


def test_ood_tightening_metrics_with_correct_thresholds(base_env, base_params, key):
    """OOD voltage_tightening + matching metric thresholds → more violations detected."""
    from powerzoojax.tasks.ders import (
        make_ders_ood_params,
        ders_no_control_rollout,
        compute_ders_safety_metrics,
    )
    ood_params = make_ders_ood_params(base_params, scenario="voltage_tightening")

    base_rollout = ders_no_control_rollout(base_env, base_params, key)
    ood_rollout = ders_no_control_rollout(base_env, ood_params, key)

    # Evaluate both with their *own* voltage bounds (correct comparison)
    m_base = compute_ders_safety_metrics(
        base_rollout, v_min=float(base_params.v_min), v_max=float(base_params.v_max)
    )
    m_ood = compute_ders_safety_metrics(
        ood_rollout, v_min=float(ood_params.v_min), v_max=float(ood_params.v_max)
    )

    # Physics are the same (same grid, same loads) but v_min/v_max stricter in OOD
    # → OOD evaluation finds ≥ as many violations as base evaluation
    assert m_ood["voltage_safety_rate"] <= m_base["voltage_safety_rate"] + 1e-6


# ========================== B — Local observation mode ==========================


def test_local_obs_dim_smaller_than_global(case141):
    """Local obs dim must be strictly smaller than global (full-grid) obs dim."""
    from powerzoojax.tasks.ders import make_ders_params
    from powerzoojax.rl.multi_agent import DistGridMARLEnv
    from powerzoojax.envs.grid.dist import DistGridEnv

    params = make_ders_params(case141, max_steps=4)

    env_global = DistGridMARLEnv(DistGridEnv(), params, observation_mode="global")
    env_local = DistGridMARLEnv(DistGridEnv(), params, observation_mode="local")

    global_dim = env_global.observation_space().shape[0]
    local_dim = env_local.observation_space().shape[0]

    assert local_dim < global_dim, (
        f"Local obs dim {local_dim} should be < global obs dim {global_dim}"
    )
    # Global dim should be roughly 3*N + 2*NL + 2 + max_dev_obs (hundreds)
    assert global_dim > 100, f"Global obs seems too small: {global_dim}"
    # Local dim should be 1 + K + 3 + 2 + max_dev_obs = 6 + K + max_dev_obs (small)
    assert local_dim < 50, f"Local obs dim {local_dim} seems too large (should be ~15)"


def test_local_obs_uniform_across_agents(case141, key=jax.random.PRNGKey(0)):
    """All agents get the same obs dimension in local mode (required for vmap)."""
    from powerzoojax.tasks.ders import make_ders_marl_env

    env, _ = make_ders_marl_env(case141, max_steps=4, observation_mode="local")
    obs_dict, _ = env.reset(key)

    dims = {obs.shape[0] for obs in obs_dict.values()}
    assert len(dims) == 1, f"Local obs dims are not uniform: {dims}"


def test_local_obs_matches_observation_space(case141, key=jax.random.PRNGKey(0)):
    """Per-agent obs shape matches observation_space() in local mode."""
    from powerzoojax.tasks.ders import make_ders_marl_env

    env, _ = make_ders_marl_env(case141, max_steps=4, observation_mode="local")
    obs_dict, _ = env.reset(key)

    expected_dim = env.observation_space().shape[0]
    for agent, obs in obs_dict.items():
        assert obs.shape == (expected_dim,), (
            f"{agent}: expected ({expected_dim},), got {obs.shape}"
        )


def test_local_obs_jit_compatible(case141, key=jax.random.PRNGKey(0)):
    """DistGridMARLEnv with local obs is JIT-compilable."""
    from powerzoojax.tasks.ders import make_ders_marl_env

    env, _ = make_ders_marl_env(case141, max_steps=4, observation_mode="local")
    obs_dict, state = env.reset(key)
    zero_actions = {name: jnp.zeros(2, dtype=jnp.float32) for name in env.agent_names}

    step_jit = jax.jit(lambda s, a: env.step(key, s, a))
    out1 = step_jit(state, zero_actions)
    out2 = step_jit(state, zero_actions)

    # Deterministic under same state/action
    for agent in env.agent_names:
        np.testing.assert_allclose(
            np.asarray(out1[2][agent]), np.asarray(out2[2][agent])
        )


def test_default_ders_env_uses_local_obs(case141):
    """make_ders_marl_env default is observation_mode='local' (Dec-POMDP intent)."""
    from powerzoojax.tasks.ders import make_ders_marl_env

    env, _ = make_ders_marl_env(case141, max_steps=4)
    # Default local obs dim should be well under 100
    obs_dim = env.observation_space().shape[0]
    assert obs_dim < 100, (
        f"Default DERs MARL env obs dim {obs_dim} is too large; "
        "expected local obs (<100) by default"
    )


# ========================== C — ders-medium-safe honesty ==========================


def test_safe_preset_not_lagrangian():
    """ders-medium-safe description must NOT claim IPPO-Lagrangian."""
    from powerzoojax.rl.presets import get_preset

    p = get_preset("ders-medium-safe")
    desc_lower = p.description.lower()

    # Must NOT say lagrangian
    assert "lagrangian" not in desc_lower, (
        f"ders-medium-safe description still claims IPPO-Lagrangian: {p.description!r}"
    )
    # Must clarify it's reward shaping
    assert "reward" in desc_lower or "shaping" in desc_lower or "penalty" in desc_lower, (
        f"ders-medium-safe description should mention reward shaping: {p.description!r}"
    )


def test_safe_preset_algo_is_ippo_typed():
    """ders-medium-safe algo must be 'ippo_typed' (type-specific sharing, not 'ppo_lagrangian')."""
    from powerzoojax.rl.presets import get_preset

    p = get_preset("ders-medium-safe")
    assert p.config.algo == "ippo_typed", (
        f"Expected algo='ippo_typed', got {p.config.algo!r}"
    )


def test_safe_preset_cost_threshold_preserved():
    """Strict zero cost_thresholds are logged even though IPPO doesn't enforce them."""
    from powerzoojax.rl.presets import get_preset

    p = get_preset("ders-medium-safe")
    assert p.config.cost_thresholds == pytest.approx((0.0, 0.0, 0.0))


# ========================== D — phase_swap / 3-phase eval ==========================


def test_phase_swap_raises_not_implemented(base_params):
    """make_ders_ood_params(..., scenario='phase_swap') raises NotImplementedError."""
    from powerzoojax.tasks.ders import make_ders_ood_params

    with pytest.raises(NotImplementedError, match="phase_swap"):
        make_ders_ood_params(base_params, scenario="phase_swap")


def test_make_ders_3phase_eval_smoke():
    """make_ders_3phase_eval() returns a DistGrid3PhaseMARLEnv with 6 agents."""
    from powerzoojax.tasks.ders import make_ders_3phase_eval
    from powerzoojax.rl.multi_agent import DistGrid3PhaseMARLEnv

    env, params = make_ders_3phase_eval(max_steps=4)
    assert isinstance(env, DistGrid3PhaseMARLEnv)
    # Default 6 battery buses → 6 agents
    assert env.num_agents == 6
    # Case123 has 114 nodes
    assert params.topo.n_nodes == 114


def test_make_ders_3phase_eval_reset():
    """make_ders_3phase_eval returns env that can reset and gives per-agent obs."""
    from powerzoojax.tasks.ders import make_ders_3phase_eval

    env, params = make_ders_3phase_eval(max_steps=4)
    key = jax.random.PRNGKey(0)
    obs_dict, state = env.reset(key)
    assert set(obs_dict.keys()) == set(env.agent_names)
    # Local obs — same dim for all agents, ≈15-dim
    obs_dim = list(obs_dict.values())[0].shape[0]
    for v in obs_dict.values():
        assert v.shape == (obs_dim,)


def test_make_ders_3phase_eval_step():
    """make_ders_3phase_eval env can step via MARL interface without error."""
    from powerzoojax.tasks.ders import make_ders_3phase_eval

    env, params = make_ders_3phase_eval(max_steps=4)
    key = jax.random.PRNGKey(0)
    obs_dict, state = env.reset(key)

    actions = {
        name: jnp.zeros(env.action_space(name).shape, dtype=jnp.float32)
        for name in env.agent_names
    }
    obs2, state2, rewards, dones, info = env.step(key, state, actions)

    assert set(obs2.keys()) == set(env.agent_names)
    assert "__all__" in dones
    assert "cost_continuous" in info


def test_make_ders_3phase_eval_has_battery_resources():
    """make_ders_3phase_eval attaches 6 battery agents by default."""
    from powerzoojax.tasks.ders import make_ders_3phase_eval

    env, params = make_ders_3phase_eval(max_steps=4)
    assert len(params.resources) == 1
    n_bats = params.resources[0].n_devices
    assert n_bats == 6  # 6-agent DERs-3phase-ood spec


def test_ders_large_uses_local_obs():
    """make_ders_large_marl_env returns env with local observation mode."""
    from powerzoojax.tasks.ders import make_ders_large_marl_env
    from powerzoojax.rl.multi_agent import DistGridMARLEnv, _LOCAL_K

    env, params = make_ders_large_marl_env(max_steps=4)
    assert isinstance(env, DistGridMARLEnv)
    assert env._observation_mode == "local"
    # Local obs dim must be smaller than global (full-grid dim scales with n_nodes)
    n = params.case.n_nodes
    nl = params.topo.n_lines
    global_dim = 3 * n + 2 * nl + 2
    assert env._obs_dim < global_dim
    # Local base is 1 + K + 3 + 2 = 10; obs_dim is in [10, 30] for K=4
    assert env._obs_dim >= 1 + _LOCAL_K + 3 + 2
    assert env._obs_dim <= 30


# ========================== E — Data hook ==========================


def test_make_ders_params_with_profiles_synthetic(case141):
    """make_ders_params_with_profiles returns valid params with no profiles (synthetic)."""
    from powerzoojax.tasks.ders import make_ders_params_with_profiles

    params = make_ders_params_with_profiles(case141, max_steps=6)
    assert params.case.n_nodes == 141
    total = sum(b.n_devices for b in params.resources)
    assert total == 12


def test_make_ders_params_with_profiles_external_pv(case141):
    """make_ders_params_with_profiles accepts external PV profiles."""
    from powerzoojax.tasks.ders import make_ders_params_with_profiles

    # Custom flat PV profile (capacity factor = 0.8 for all steps/devices)
    pv_profiles = jnp.ones((6, 4), dtype=jnp.float32) * 0.8
    params = make_ders_params_with_profiles(case141, max_steps=6, pv_profiles=pv_profiles)

    # PV bundle should use the supplied profiles
    from powerzoojax.envs.resource.renewable import RenewableBundle
    pv_bundle = next(b for b in params.resources if isinstance(b, RenewableBundle))
    np.testing.assert_allclose(
        np.asarray(pv_bundle.profiles), np.asarray(pv_profiles), atol=1e-5
    )


def test_make_ders_params_with_profiles_external_load(case141):
    """make_ders_params_with_profiles accepts external load profiles."""
    from powerzoojax.tasks.ders import make_ders_params_with_profiles

    n_nodes = case141.n_nodes
    load_p = jnp.ones((6, n_nodes), dtype=jnp.float32) * 0.01
    load_q = jnp.ones((6, n_nodes), dtype=jnp.float32) * 0.005
    params = make_ders_params_with_profiles(
        case141, max_steps=6,
        load_profiles_p=load_p,
        load_profiles_q=load_q,
    )
    np.testing.assert_allclose(
        np.asarray(params.load_profiles_p), np.asarray(load_p), atol=1e-6
    )


def test_load_ders_load_shape_peak_normalized():
    """DERs real-data load shape should be peak-normalised, not mean-normalised."""
    from powerzoojax.tasks.ders import load_ders_load_shape

    class _FakeLoader:
        def load_jax_profiles(self, signals, **kwargs):
            region = kwargs["region"]
            scale = 1.0 + (sum(ord(c) for c in region) % 5)
            base = {
                "Broadmeadow 132_11kV": np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32),
                "Charlestown 132_11kV": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                "Jesmond 132_11kV": np.array([3.0, 6.0, 9.0, 12.0], dtype=np.float32),
            }
            if region not in base:
                base[region] = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32) * scale
            return base[region][:, None]

    shape = load_ders_load_shape(data_loader=_FakeLoader(), role="train")
    assert shape.shape == (4,)
    assert np.max(shape) == pytest.approx(1.0)
    assert float(np.mean(shape)) < 1.0


def test_make_ders_params_from_split_improves_reset_voltage(case141):
    """Real-data split factory should lift reset voltage above flat-load baseline."""
    from powerzoojax.tasks.ders import make_ders_params, make_ders_params_from_split
    from powerzoojax.envs.grid.dist import DistGridEnv

    class _FakeLoader:
        def load_jax_profiles(self, signals, **kwargs):
            source = kwargs.get("source")
            region = kwargs.get("region")
            if source == "ausgrid":
                scale = 1.0 + (sum(ord(c) for c in region) % 5)
                base = {
                    "Broadmeadow 132_11kV": np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32),
                    "Charlestown 132_11kV": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                    "Jesmond 132_11kV": np.array([3.0, 6.0, 9.0, 12.0], dtype=np.float32),
                }
                if region not in base:
                    base[region] = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32) * scale
                return base[region][:, None]
            if source == "gb":
                return np.array([[0.0], [1.0], [0.5], [0.0]], dtype=np.float32)
            raise AssertionError(f"Unexpected source: {source}")

    env = DistGridEnv()
    flat = make_ders_params(case141, max_steps=4)
    real = make_ders_params_from_split(
        case141,
        role="train",
        data_loader=_FakeLoader(),
        max_steps=4,
    )

    _, flat_state = env.reset(jax.random.PRNGKey(0), flat)
    _, real_state = env.reset(jax.random.PRNGKey(0), real)

    flat_vmin = float(np.min(np.asarray(flat_state.v_mag)))
    real_vmin = float(np.min(np.asarray(real_state.v_mag)))

    assert real_vmin > flat_vmin
    assert real.resources[1].profiles.shape == (4, 4)


def test_ders_task_episode_params_vary_across_episodes(case141, monkeypatch):
    """DERsTask.episode_params must return different windows across episodes."""
    import powerzoojax.tasks.ders as ders_mod

    def _fake_split_profiles(*_args, **_kwargs):
        load = np.linspace(0.6, 1.4, 16, dtype=np.float32)
        pv = np.linspace(0.0, 1.0, 16, dtype=np.float32)
        return load, pv

    monkeypatch.setattr(ders_mod, "load_ders_split_profiles", _fake_split_profiles)
    task = ders_mod.DERsTask(case=case141, max_steps=4)

    p0 = task.episode_params("iid", 0, 4, 4, strategy="uniform", seed=0)
    p1 = task.episode_params("iid", 1, 4, 4, strategy="uniform", seed=0)

    assert task.episode_start("iid", 0, 4, strategy="uniform", seed=0) != task.episode_start(
        "iid", 1, 4, strategy="uniform", seed=0
    )
    assert not np.allclose(
        np.asarray(p0.load_profiles_p),
        np.asarray(p1.load_profiles_p),
    )
    assert not np.allclose(
        np.asarray(p0.resources[1].profiles),
        np.asarray(p1.resources[1].profiles),
    )


def test_ders_noop_action_respects_bundle_semantics(case141):
    """No-control must mean PV no-curtail, not zero-valued typed action."""
    from powerzoojax.tasks.ders import ders_noop_action, make_ders_params

    params = make_ders_params(case141, max_steps=4)
    action = np.asarray(ders_noop_action(params), dtype=np.float32)

    bat_dim = params.resources[0].action_dim
    pv_dim = params.resources[1].action_dim
    flex_dim = params.resources[2].action_dim

    bat = action[:bat_dim].reshape(params.resources[0].n_devices, 2)
    pv = action[bat_dim: bat_dim + pv_dim].reshape(params.resources[1].n_devices, 2)
    flex = action[bat_dim + pv_dim: bat_dim + pv_dim + flex_dim].reshape(
        params.resources[2].n_devices, 2
    )

    np.testing.assert_allclose(bat, 0.0, atol=1e-6)
    np.testing.assert_allclose(pv[:, 0], 1.0, atol=1e-6)
    np.testing.assert_allclose(pv[:, 1], 0.0, atol=1e-6)
    np.testing.assert_allclose(flex, 0.0, atol=1e-6)


def test_ders_no_control_rollout_matches_explicit_noop_action(base_env, base_params, key):
    """ders_no_control_rollout should use the same bundle-aware no-op action."""
    from powerzoojax.tasks.ders import ders_no_control_rollout, ders_noop_action, rollout_ders

    explicit = rollout_ders(base_env, base_params, key, lambda obs, k: ders_noop_action(base_params))
    actual = ders_no_control_rollout(base_env, base_params, key)

    for field in ("reward", "cost_continuous", "p_loss_MW", "v_min_episode", "v_max_episode"):
        np.testing.assert_allclose(
            np.asarray(actual[field]),
            np.asarray(explicit[field]),
            atol=1e-6,
        )


def test_build_train_params_bank_hardest_selector_uses_margin_scoring(case141, monkeypatch):
    """Hardest-window bank construction should work with tightened score bounds."""
    import benchmarks.ders.train as ders_train
    import powerzoojax.tasks.ders as ders_mod

    def _fake_split_profiles(*_args, **_kwargs):
        load = np.linspace(0.8, 1.3, 16, dtype=np.float32)
        pv = np.linspace(0.1, 0.9, 16, dtype=np.float32)
        return load, pv

    monkeypatch.setattr(ders_mod, "load_ders_split_profiles", _fake_split_profiles)

    params_bank, starts, meta = ders_train._build_train_params_bank(
        case=case141,
        task_config={
            "train_window_role": "train",
            "train_window_count": 2,
            "train_window_candidate_count": 4,
            "train_window_selector": "hardest_voltage_margin",
            "train_window_score_v_min": 0.96,
            "train_window_score_v_max": 1.04,
            "v_min": 0.94,
            "v_max": 1.06,
        },
        max_steps=4,
    )

    assert len(params_bank) == 2
    assert len(starts) == 2
    assert starts == sorted(starts)
    assert meta["train_window_selector"] == "hardest_voltage_margin"
    assert meta["train_window_score_v_min"] == pytest.approx(0.96)
    assert meta["train_window_score_v_max"] == pytest.approx(1.04)


# ========================== F — Zero-shot obs compatibility ==========================


def test_3phase_obs_dim_equals_1phase_medium(case141):
    """DERs-3phase-ood obs_dim must match DERs-medium for zero-shot compat."""
    from powerzoojax.tasks.ders import make_ders_marl_env, make_ders_3phase_eval

    env_m, _ = make_ders_marl_env(case141, max_steps=4)
    env_3, _ = make_ders_3phase_eval(max_steps=4)

    assert env_m._obs_dim == env_3._obs_dim, (
        f"1-phase medium obs_dim={env_m._obs_dim} != "
        f"3-phase obs_dim={env_3._obs_dim}; "
        "zero-shot transfer will fail with shape mismatch"
    )


def test_3phase_obs_dim_equals_1phase_large(case141):
    """DERs-large obs_dim must also match DERs-3phase-ood."""
    from powerzoojax.tasks.ders import make_ders_large_marl_env, make_ders_3phase_eval

    env_l, _ = make_ders_large_marl_env(case141, max_steps=4)
    env_3, _ = make_ders_3phase_eval(max_steps=4)

    assert env_l._obs_dim == env_3._obs_dim, (
        f"DERs-large obs_dim={env_l._obs_dim} != "
        f"3-phase obs_dim={env_3._obs_dim}"
    )


def test_3phase_policy_forward_no_shape_error():
    """A SharedActorCritic initialised on 1-phase obs_dim can forward-pass
    a 3-phase battery obs without shape error — true zero-shot compat.
    """
    from powerzoojax.tasks.ders import make_ders_marl_env, make_ders_3phase_eval
    from powerzoojax.rl.ippo import SharedActorCritic

    case = create_case141()
    env_m, _ = make_ders_marl_env(case, max_steps=4)
    env_3, _ = make_ders_3phase_eval(max_steps=4)

    # Both must have the same obs_dim
    assert env_m._obs_dim == env_3._obs_dim

    obs_dim   = env_m._obs_dim
    action_dim = env_m.action_space().shape[0]

    # Build and init a 1-phase policy
    net = SharedActorCritic(hidden_dims=(64, 64), action_dim=action_dim)
    key = jax.random.PRNGKey(0)
    params = net.init(key, jnp.zeros(obs_dim))

    # Get a real 3-phase obs and run the 1-phase policy on it
    obs_dict_3ph, _ = env_3.reset(key)
    for agent_obs in obs_dict_3ph.values():
        mean, log_std, value = net.apply(params, agent_obs)
        assert mean.shape    == (action_dim,), f"mean shape {mean.shape}"
        assert log_std.shape == (action_dim,), f"log_std shape {log_std.shape}"
        assert value.shape   == ()


# ========================== G — Type-specific parameter sharing ==========================


def test_ippo_typed_creates_separate_params_per_type(case141):
    """make_ippo_typed_train creates distinct params for battery / pv / flexload."""
    from powerzoojax.tasks.ders import make_ders_marl_env
    from powerzoojax.rl.ippo import make_ippo_typed_train
    from powerzoojax.rl.config import TrainConfig

    env, _ = make_ders_marl_env(case141, max_steps=4)
    config = TrainConfig(
        algo="ippo", total_timesteps=2 * 4,  # 1 update
        num_envs=2, n_steps=4, n_epochs=1,
    )
    train_fn = make_ippo_typed_train(env, config)
    result = train_fn(jax.random.PRNGKey(42))

    # params is a dict keyed by agent type.
    # Agent type comes from bundle class name: BatteryBundle→"battery",
    # RenewableBundle→"renewable", FlexLoadBundle→"flexload".
    assert isinstance(result.params, dict)
    assert set(result.params.keys()) == {"battery", "renewable", "flexload"}

    # Different types must have separate (non-aliased) parameter pytrees
    bat_w = result.params["battery"]["params"]["Dense_0"]["kernel"]
    pv_w  = result.params["renewable"]["params"]["Dense_0"]["kernel"]
    fl_w  = result.params["flexload"]["params"]["Dense_0"]["kernel"]

    assert not jnp.allclose(bat_w, pv_w),  "battery and renewable share identical weights — wrong"
    assert not jnp.allclose(bat_w, fl_w),  "battery and flexload share identical weights — wrong"


def test_ippo_typed_smoke(case141):
    """make_ippo_typed_train completes one update without error."""
    from powerzoojax.tasks.ders import make_ders_marl_env
    from powerzoojax.rl.ippo import make_ippo_typed_train
    from powerzoojax.rl.config import TrainConfig

    env, _ = make_ders_marl_env(case141, max_steps=4)
    config = TrainConfig(
        algo="ippo", total_timesteps=2 * 4,
        num_envs=2, n_steps=4, n_epochs=1,
        eval_freq=4, eval_num_episodes=2, record_eval_wall_time=True,
    )
    result = make_ippo_typed_train(env, config)(jax.random.PRNGKey(0))

    assert "loss"        in result.metrics
    assert "mean_reward" in result.metrics
    assert "eval_returns" in result.metrics
    assert "eval_wall_time_s" in result.metrics
    assert result.metrics["loss"].shape == (1,)  # 1 update
    assert result.metrics["eval_returns"].shape == result.metrics["eval_wall_time_s"].shape
    assert result.metrics["eval_returns"].shape[0] >= 1


def test_ippo_typed_dispatch_via_make_train(case141):
    """make_train(env, config) with algo='ippo_typed' routes to make_ippo_typed_train.

    Verifies the full dispatch chain:
      presets → TrainConfig(algo='ippo_typed') → trainer.make_train → make_ippo_typed_train.
    Result.params must be a dict keyed by agent type (not a plain Flax pytree).
    """
    from powerzoojax.tasks.ders import make_ders_marl_env
    from powerzoojax.rl.trainer import make_train
    from powerzoojax.rl.config import TrainConfig

    env, _ = make_ders_marl_env(case141, max_steps=4)
    config = TrainConfig(
        algo="ippo_typed", total_timesteps=2 * 4,
        num_envs=2, n_steps=4, n_epochs=1,
    )
    result = make_train(env, config)(jax.random.PRNGKey(0))

    # Typed dispatch: params is a dict, not a bare Flax pytree
    assert isinstance(result.params, dict), (
        f"Expected dict of per-type params, got {type(result.params)}"
    )
    assert {"battery", "renewable", "flexload"} == set(result.params.keys())


def test_ippo_typed_lagrangian_dispatch_via_make_train(case141):
    """Typed IPPO-Lagrangian should dispatch through make_train and emit CMDP metrics."""
    from powerzoojax.tasks.ders import make_ders_marl_env
    from powerzoojax.rl.trainer import make_train
    from powerzoojax.rl.config import TrainConfig

    env, _ = make_ders_marl_env(case141, max_steps=4, voltage_penalty=0.0)
    config = TrainConfig(
        algo="ippo_typed_lagrangian",
        total_timesteps=2 * 4 * 2,
        num_envs=2,
        n_steps=4,
        n_epochs=1,
        cost_thresholds=(0.0, 0.0, 0.0),
        lambda_lr=0.01,
    )
    result = make_train(env, config)(jax.random.PRNGKey(0))

    assert isinstance(result.params, dict), (
        f"Expected dict of per-type actor params, got {type(result.params)}"
    )
    assert {"battery", "renewable", "flexload"} == set(result.params.keys())
    assert jnp.asarray(result.metrics["lambda"]).shape == (2,)
    assert jnp.asarray(result.metrics["episode_cost_est"]).shape == (2,)
    assert "mean_cost_voltage_violation" in result.metrics
    assert "lambda_resource" in result.metrics


def test_sampled_params_distgrid_marl_wrapper_reset_step_jit(case141):
    """Sampled-params DER MARL wrapper should reset/step/JIT across a 2-window bank."""
    from benchmarks.ders.train import _SampledParamsDistGridMARLEnv
    from powerzoojax.envs.grid.dist import DistGridEnv
    from powerzoojax.tasks.ders import (
        apply_ders_profile_window,
        load_ders_split_profiles,
        make_ders_params,
    )

    load_shape, pv_profile = load_ders_split_profiles(role="train")
    base_params = make_ders_params(case141, max_steps=4)
    params_bank = [
        apply_ders_profile_window(
            base_params,
            load_shape=load_shape,
            pv_profile=pv_profile,
            episode_start=start,
        )
        for start in (0, 3)
    ]
    env = _SampledParamsDistGridMARLEnv(
        DistGridEnv(),
        params_bank,
        voltage_penalty=0.0,
        observation_mode="local",
    )
    reset_jit = jax.jit(lambda k: env.reset(k))
    obs_dict, state = reset_jit(jax.random.PRNGKey(0))
    assert set(obs_dict.keys()) == set(env.agent_names)
    assert int(state.params_idx) in (0, 1)

    zero_actions = {
        name: jnp.zeros(env.action_space().shape, dtype=jnp.float32)
        for name in env.agent_names
    }
    step_jit = jax.jit(lambda s, a: env.step(jax.random.PRNGKey(1), s, a))
    obs2, state2, rewards, dones, info = step_jit(state, zero_actions)
    assert set(obs2.keys()) == set(env.agent_names)
    assert "__all__" in dones
    assert "constraint_costs" in info
    assert rewards[env.agent_names[0]].shape == ()


def test_ders_medium_preset_uses_ippo_typed():
    """ders-medium preset config must use algo='ippo_typed'."""
    from powerzoojax.rl.presets import PRESETS
    assert PRESETS["ders-medium"].config.algo == "ippo_typed", (
        "ders-medium preset should use algo='ippo_typed' for type-specific sharing"
    )


def test_ders_medium_safe_preset_uses_ippo_typed():
    """ders-medium-safe preset config must use algo='ippo_typed'."""
    from powerzoojax.rl.presets import PRESETS
    assert PRESETS["ders-medium-safe"].config.algo == "ippo_typed", (
        "ders-medium-safe should use algo='ippo_typed'"
    )


def test_ippo_typed_jit_compiles_without_python_if(case141):
    """The typed train step must be JIT-compiled (no traced Python if)."""
    from powerzoojax.tasks.ders import make_ders_marl_env
    from powerzoojax.rl.ippo import make_ippo_typed_train
    from powerzoojax.rl.config import TrainConfig

    env, _ = make_ders_marl_env(case141, max_steps=4)
    config = TrainConfig(
        algo="ippo", total_timesteps=4 * 4,
        num_envs=2, n_steps=4, n_epochs=1,
    )
    # Two calls — second should reuse compiled XLA; no Python retracing error
    train_fn = make_ippo_typed_train(env, config)
    r1 = train_fn(jax.random.PRNGKey(1))
    r2 = train_fn(jax.random.PRNGKey(2))
    assert r1.metrics["loss"].shape == r2.metrics["loss"].shape
