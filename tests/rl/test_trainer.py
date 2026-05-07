"""Tests for rl/trainer.py, rl/config.py, and rl/cmdp.py.

Covers:
    L0 config:  TrainConfig fields, replace, YAML round-trip, asdict
    L1 config:  hidden_dims tuple coercion, field validation
    CMDP smoke: make_cmdp_train + short rollout (no external deps)
    Rejax smoke: make_rejax_train + train (requires rejax, else skip)
    Dispatcher: make_train routing, unknown algo error, MARL NotImplementedError
"""

import json
import os
import tempfile

import jax
import jax.numpy as jnp
import pytest

from benchmarks.common.runtime import make_policy_fn
from powerzoojax.rl.config import TrainConfig, load_config, save_config
from powerzoojax.rl.policies import BoundedBetaPolicy
from powerzoojax.rl.trainer import _evaluate_with_metrics, make_train, TrainResult
from powerzoojax.rl.wrappers import LogWrapper, SafeRLWrapper, SauteWrapper
from powerzoojax.envs.resource.battery import BatteryEnv, make_battery_params
from powerzoojax.envs.grid.trans import TransGridEnv, make_trans_params
from powerzoojax.case import create_case5


# ============ Helpers ============

def _make_battery_log():
    return LogWrapper(BatteryEnv(), make_battery_params(max_steps=8))


def _make_trans_safe():
    case = create_case5()
    profiles = jnp.ones((8, case.n_loads), dtype=jnp.float32) * 0.5
    params = make_trans_params(case, load_profiles=profiles, max_steps=8)
    return SafeRLWrapper(TransGridEnv(), params, cost_threshold=5.0)


def _make_trans_saute():
    case = create_case5()
    profiles = jnp.ones((8, case.n_loads), dtype=jnp.float32) * 0.5
    params = make_trans_params(case, load_profiles=profiles, max_steps=8)
    return SauteWrapper(TransGridEnv(), params, cost_threshold=5.0)


_REJAX_AVAILABLE = False
try:
    import rejax as _rejax  # noqa: F401
    _REJAX_AVAILABLE = True
except ImportError:
    pass


# ============ TrainConfig ============

class TestTrainConfig:
    """L0/L1 tests for TrainConfig (frozen dataclass)."""

    def test_default_fields(self):
        c = TrainConfig()
        assert c.algo == "ppo"
        assert c.total_timesteps == 100_000
        assert c.num_envs == 64
        assert c.seed == 42
        assert c.eval_episodes == 128
        assert c.hidden_dims == (64, 64)

    def test_replace(self):
        c = TrainConfig()
        c2 = c.replace(seed=99, total_timesteps=500_000)
        assert c2.seed == 99
        assert c2.total_timesteps == 500_000
        assert c2.algo == "ppo"  # unchanged

    def test_frozen_immutability(self):
        c = TrainConfig()
        with pytest.raises(Exception):  # frozen dataclass raises FrozenInstanceError
            c.seed = 0

    def test_asdict_returns_plain_dict(self):
        c = TrainConfig()
        d = c._asdict()
        assert isinstance(d, dict)
        assert d["algo"] == "ppo"
        assert isinstance(d["hidden_dims"], list)  # tuple → list for YAML

    def test_to_rejax_kwargs_ppo(self):
        c = TrainConfig(algo="ppo", n_steps=64, n_epochs=2, hidden_dims=(128, 128))
        kw = c.to_rejax_kwargs()
        assert kw["num_steps"] == 64    # n_steps → num_steps
        assert kw["num_epochs"] == 2    # n_epochs → num_epochs
        assert "learning_rate" in kw
        assert "total_timesteps" in kw
        assert kw["agent_kwargs"]["hidden_layer_sizes"] == (128, 128)
        assert kw["normalize_observations"] is False

    def test_to_rejax_kwargs_ppo_beta(self):
        c = TrainConfig(algo="ppo", continuous_action_dist="beta")
        kw = c.to_rejax_kwargs()
        assert kw["agent_kwargs"]["hidden_layer_sizes"] == (64, 64)
        assert c.continuous_action_dist == "beta"

    def test_to_rejax_kwargs_saute(self):
        c = TrainConfig(
            algo="saute_ppo",
            n_steps=64,
            n_epochs=2,
            hidden_dims=(128, 128),
            cost_threshold=5.0,
        )
        kw = c.to_rejax_kwargs()
        assert kw["num_steps"] == 64
        assert kw["num_epochs"] == 2
        assert kw["agent_kwargs"]["hidden_layer_sizes"] == (128, 128)

    def test_to_rejax_kwargs_sac(self):
        c = TrainConfig(algo="sac", hidden_dims=(128, 128))
        kw = c.to_rejax_kwargs()
        assert "total_timesteps" in kw
        # SAC doesn't use n_steps / n_epochs
        assert "num_steps" not in kw
        assert kw["hidden_layer_sizes"] == (128, 128)

    def test_to_rejax_kwargs_sac_target_entropy_ratio(self):
        c = TrainConfig(algo="sac", sac_target_entropy_ratio=0.5)
        kw = c.to_rejax_kwargs()
        assert kw["target_entropy_ratio"] == pytest.approx(0.5)

    def test_record_eval_wall_time_default(self):
        assert TrainConfig().record_eval_wall_time is False
        assert TrainConfig().wall_time_warmup is True


class TestTrainConfigYAML:
    """YAML round-trip tests."""

    def test_save_load_roundtrip(self, tmp_path):
        c = TrainConfig(algo="sac", total_timesteps=50_000, seed=7)
        path = str(tmp_path / "config.yaml")
        save_config(c, path)
        c2 = load_config(path)
        assert c2.algo == "sac"
        assert c2.total_timesteps == 50_000
        assert c2.seed == 7

    def test_hidden_dims_list_to_tuple(self, tmp_path):
        """hidden_dims saved as list must be loaded back as tuple."""
        c = TrainConfig(hidden_dims=(128, 64))
        path = str(tmp_path / "config.yaml")
        save_config(c, path)
        c2 = load_config(path)
        assert isinstance(c2.hidden_dims, tuple)
        assert c2.hidden_dims == (128, 64)

    def test_load_from_dict(self):
        d = {"algo": "td3", "total_timesteps": 10_000, "hidden_dims": [32, 32]}
        c = load_config(d)
        assert c.algo == "td3"
        assert c.total_timesteps == 10_000
        assert isinstance(c.hidden_dims, tuple)

    def test_load_ignores_unknown_keys(self):
        """Unknown keys in config dict must be silently ignored."""
        d = {"algo": "ppo", "total_timesteps": 5_000, "_future_field": 42}
        c = load_config(d)
        assert c.algo == "ppo"


# ============ TrainResult ============

class TestTrainResult:
    def test_summary_keys(self):
        metrics = {"mean_reward": jnp.array([1.0, 2.0, 3.0])}
        config = TrainConfig()
        result = TrainResult(params=None, metrics=metrics, config=config, env_name="test")
        s = result.summary
        assert "algo" in s
        assert "final_reward" in s
        assert float(s["final_reward"]) == pytest.approx(3.0)

    def test_save_json(self, tmp_path):
        metrics = {"mean_reward": jnp.array([1.0, 2.0])}
        config = TrainConfig()
        result = TrainResult(params=None, metrics=metrics, config=config)
        path = str(tmp_path / "result.json")
        result.save(path)
        with open(path) as f:
            data = json.load(f)
        assert "summary" in data
        assert "config" in data


# ============ Dispatcher ============

class TestMakeTrain:
    def test_unknown_algo_raises(self):
        env = _make_battery_log()
        with pytest.raises(ValueError, match="Unknown algo"):
            make_train(env, TrainConfig(algo="unknown_algo"))

    def test_marl_ippo_routes_to_ippo_backend(self):
        """make_train with algo='ippo' routes to IPPO backend for GridMARLEnv."""
        from powerzoojax.case import create_case5
        from powerzoojax.envs.grid.trans import TransGridEnv, make_trans_params
        from powerzoojax.rl.multi_agent import GridMARLEnv
        import jax.numpy as jnp
        case = create_case5()
        profiles = jnp.ones((48, case.n_loads), dtype=jnp.float32) * 0.5
        params = make_trans_params(case, load_profiles=profiles, max_steps=8)
        env = GridMARLEnv(TransGridEnv(), params)
        config = TrainConfig(
            algo="ippo", total_timesteps=160, num_envs=4, n_steps=8, n_epochs=1,
            hidden_dims=(16, 16),
        )
        train_fn = make_train(env, config)
        result = train_fn(jax.random.PRNGKey(0))
        assert result.config.algo == "ippo"

    def test_mappo_raises_not_implemented(self):
        from powerzoojax.case import create_case5
        from powerzoojax.envs.grid.trans import TransGridEnv, make_trans_params
        from powerzoojax.rl.multi_agent import GridMARLEnv
        import jax.numpy as jnp
        case = create_case5()
        profiles = jnp.ones((48, case.n_loads), dtype=jnp.float32) * 0.5
        params = make_trans_params(case, load_profiles=profiles, max_steps=8)
        env = GridMARLEnv(TransGridEnv(), params)
        with pytest.raises(NotImplementedError, match="MAPPO"):
            make_train(env, TrainConfig(algo="mappo"))

    def test_rejax_missing_raises_import_error(self, monkeypatch):
        """If rejax is not importable, make_train raises ImportError."""
        import sys
        env = _make_battery_log()
        # Remove rejax from sys.modules so next import attempt fails
        saved = sys.modules.pop("rejax", None)
        # Use monkeypatch to block re-import
        monkeypatch.setitem(sys.modules, "rejax", None)
        try:
            with pytest.raises((ImportError, TypeError)):
                make_train(env, TrainConfig(algo="ppo"))
        finally:
            if saved is not None:
                sys.modules["rejax"] = saved
            else:
                sys.modules.pop("rejax", None)

    def test_cmdp_wrong_env_type_raises(self):
        """CMDP backend must reject non-SafeRLWrapper envs."""
        env = _make_battery_log()  # LogWrapper, not SafeRLWrapper
        from powerzoojax.rl.cmdp import make_cmdp_train
        with pytest.raises(TypeError, match="SafeRLWrapper"):
            make_cmdp_train(env, TrainConfig(algo="ppo_lagrangian"))


# ============ CMDP Smoke Test ============

class TestCMDPSmoke:
    """Short CMDP training smoke test — no external dependencies."""

    def test_cmdp_train_runs(self):
        """make_cmdp_train + 2 update steps must complete without error."""
        env = _make_trans_safe()
        config = TrainConfig(
            algo="ppo_lagrangian",
            total_timesteps=8 * 4 * 2,  # 2 updates
            num_envs=4,
            n_steps=8,
            n_epochs=1,
            n_minibatches=2,
            cost_threshold=5.0,
        )
        from powerzoojax.rl.cmdp import make_cmdp_train
        train_fn = make_cmdp_train(env, config)
        result = train_fn(jax.random.PRNGKey(0))
        assert isinstance(result, TrainResult)
        assert "mean_reward" in result.metrics
        assert "lambda" in result.metrics

    def test_cmdp_metrics_shape(self):
        """Metrics arrays must have shape (n_updates,)."""
        env = _make_trans_safe()
        n_updates = 3
        config = TrainConfig(
            algo="ppo_lagrangian",
            total_timesteps=8 * 4 * n_updates,
            num_envs=4,
            n_steps=8,
            n_epochs=1,
            n_minibatches=2,
            cost_threshold=5.0,
        )
        from powerzoojax.rl.cmdp import make_cmdp_train
        train_fn = make_cmdp_train(env, config)
        result = train_fn(jax.random.PRNGKey(0))
        lam = jnp.asarray(result.metrics["lambda"])
        assert lam.shape == (n_updates,)

    def test_cmdp_lambda_adjusts(self):
        """Lambda should become active if cost deviates from threshold."""
        env = _make_trans_safe()
        config = TrainConfig(
            algo="ppo_lagrangian",
            total_timesteps=8 * 4 * 5,
            num_envs=4,
            n_steps=8,
            n_epochs=1,
            n_minibatches=2,
            cost_threshold=0.0,  # very tight → lambda should increase
            lambda_lr=0.1,
        )
        from powerzoojax.rl.cmdp import make_cmdp_train
        train_fn = make_cmdp_train(env, config)
        result = train_fn(jax.random.PRNGKey(0))
        lambdas = jnp.asarray(result.metrics["lambda"])
        # The logged series starts after the first update, so it need not be
        # monotonic. What matters is that the dual becomes active above its
        # initial exp(0)=1 baseline when threshold violations are present.
        assert float(jnp.max(lambdas)) > 1.0


class _DummyEvalEnv:
    def reset(self, key, params):
        del key, params
        return jnp.array([0.0], dtype=jnp.float32), jnp.int32(0)

    def step(self, key, state, action, params):
        del key, action, params
        next_state = state + 1
        done = next_state >= 4
        info = {
            "cost": jnp.float32(2.0),
            "cost_voltage_violation": jnp.float32(1.5),
            "cost_thermal_overload": jnp.float32(0.25),
            "reserve_shortfall": jnp.float32(0.5),
        }
        obs = jnp.array([next_state], dtype=jnp.float32)
        reward = jnp.float32(-1.0)
        return obs, next_state, reward, done, info


class _DummySparseViolationEnv:
    def reset(self, key, params):
        del key, params
        return jnp.array([0.0], dtype=jnp.float32), jnp.int32(0)

    def step(self, key, state, action, params):
        del key, action, params
        next_state = state + 1
        done = next_state >= 4
        thermal = jnp.where(next_state == 2, jnp.float32(0.25), jnp.float32(0.0))
        reserve = jnp.where(next_state == 3, jnp.float32(0.5), jnp.float32(0.0))
        info = {
            "cost": jnp.float32(2.0),
            "cost_voltage_violation": jnp.float32(0.0),
            "cost_thermal_overload": thermal,
            "reserve_shortfall": reserve,
        }
        obs = jnp.array([next_state], dtype=jnp.float32)
        reward = jnp.float32(-1.0)
        return obs, next_state, reward, done, info


class TestEvalMetrics:
    def test_bounded_beta_policy_multidim_shapes(self):
        policy = BoundedBetaPolicy(
            5,
            (jnp.full((5,), -1.0), jnp.full((5,), 1.0)),
            (32, 32),
            jax.nn.swish,
        )
        obs = jnp.zeros((4, 22), dtype=jnp.float32)
        rng = jax.random.PRNGKey(0)
        params = policy.init(rng, obs, rng)
        action, log_prob = policy.apply(params, obs, rng, method="action_log_prob")
        log_prob2, entropy = policy.apply(params, obs, action, method="log_prob_entropy")
        mode_action = policy.apply(params, obs, method="mode_action")
        assert action.shape == (4, 5)
        assert log_prob.shape == (4,)
        assert log_prob2.shape == (4,)
        assert entropy.shape == (4,)
        assert mode_action.shape == (4, 5)
        assert jnp.all(jnp.isfinite(action))
        assert jnp.all(jnp.isfinite(log_prob))
        assert jnp.all(jnp.isfinite(log_prob2))
        assert jnp.all(jnp.isfinite(entropy))
        assert jnp.all(mode_action >= -1.0)
        assert jnp.all(mode_action <= 1.0)

    def test_bounded_beta_policy_clips_extreme_concentrations(self):
        policy = BoundedBetaPolicy(
            2,
            (jnp.full((2,), -1.0), jnp.full((2,), 1.0)),
            (16,),
            jax.nn.swish,
            max_concentration=5.0,
        )
        obs = jnp.full((1, 4), 1e6, dtype=jnp.float32)
        rng = jax.random.PRNGKey(0)
        params = policy.init(rng, obs, rng)
        alpha, beta = policy.apply(params, obs, method="_alpha_beta")
        assert jnp.all(alpha <= 5.0 + 1e-6)
        assert jnp.all(beta <= 5.0 + 1e-6)

    def test_evaluate_with_metrics_tracks_voltage_and_total_cost(self):
        env = _DummyEvalEnv()

        def _act(obs, key):
            del obs, key
            return jnp.array([0.0], dtype=jnp.float32)

        metrics = _evaluate_with_metrics(
            _act,
            jax.random.PRNGKey(0),
            env,
            None,
            num_seeds=2,
            max_steps_in_episode=8,
        )

        assert float(metrics["eval_returns"]) == pytest.approx(-4.0)
        assert float(metrics["eval_total_cost"]) == pytest.approx(8.0)
        assert float(metrics["eval_cost_per_step"]) == pytest.approx(2.0)
        assert float(metrics["eval_total_voltage_violation"]) == pytest.approx(6.0)
        assert float(metrics["eval_cost_voltage_violation"]) == pytest.approx(1.5)
        assert float(metrics["eval_voltage_violation_rate"]) == pytest.approx(1.0)
        assert float(metrics["eval_voltage_violation_episode_rate"]) == pytest.approx(1.0)
        assert float(metrics["eval_total_thermal_overload"]) == pytest.approx(1.0)
        assert float(metrics["eval_total_reserve_shortfall"]) == pytest.approx(2.0)
        assert float(metrics["eval_thermal_violation_rate"]) == pytest.approx(1.0)
        assert float(metrics["eval_reserve_shortfall_rate"]) == pytest.approx(1.0)

    def test_evaluate_with_metrics_uses_step_level_safety_rates(self):
        env = _DummySparseViolationEnv()

        def _act(obs, key):
            del obs, key
            return jnp.array([0.0], dtype=jnp.float32)

        metrics = _evaluate_with_metrics(
            _act,
            jax.random.PRNGKey(0),
            env,
            None,
            num_seeds=2,
            max_steps_in_episode=8,
        )

        assert float(metrics["eval_thermal_violation_rate"]) == pytest.approx(0.25)
        assert float(metrics["eval_reserve_shortfall_rate"]) == pytest.approx(0.25)
        assert float(metrics["eval_thermal_violation_episode_rate"]) == pytest.approx(1.0)
        assert float(metrics["eval_reserve_shortfall_episode_rate"]) == pytest.approx(1.0)


# ============ Rejax Smoke Test (conditional) ============

@pytest.mark.skipif(not _REJAX_AVAILABLE, reason="rejax not installed")
class TestRejaxSmoke:
    def test_ppo_beta_battery_runs(self):
        env = _make_battery_log()
        config = TrainConfig(
            algo="ppo",
            continuous_action_dist="beta",
            total_timesteps=2_000,
            num_envs=4,
            n_steps=8,
            n_epochs=1,
            n_minibatches=2,
        )
        train_fn = make_train(env, config)
        result = train_fn(jax.random.PRNGKey(0))
        assert isinstance(result, TrainResult)

    def test_ppo_battery_runs(self):
        """PPO on BatteryEnv must complete and return TrainResult."""
        env = _make_battery_log()
        config = TrainConfig(
            algo="ppo",
            total_timesteps=2_000,
            num_envs=4,
            n_steps=8,
            n_epochs=1,
            n_minibatches=2,
        )
        train_fn = make_train(env, config)
        result = train_fn(jax.random.PRNGKey(0))
        assert isinstance(result, TrainResult)

    def test_ppo_result_has_metrics(self):
        env = _make_battery_log()
        config = TrainConfig(
            algo="ppo",
            total_timesteps=1_000,
            num_envs=4,
            n_steps=8,
            n_epochs=1,
        )
        train_fn = make_train(env, config)
        result = train_fn(jax.random.PRNGKey(0))
        assert result.metrics is not None
        assert result.config.algo == "ppo"

    def test_ppo_record_eval_wall_time(self):
        """Per-eval wall times via io_callback (same fused train as default)."""
        env = _make_battery_log()
        config = TrainConfig(
            algo="ppo",
            total_timesteps=2_000,
            num_envs=4,
            n_steps=8,
            n_epochs=1,
            n_minibatches=2,
            eval_freq=1_000,
            record_eval_wall_time=True,
            wall_time_warmup=False,
        )
        train_fn = make_train(env, config)
        result = train_fn(jax.random.PRNGKey(0))
        assert "eval_wall_time_s" in result.metrics
        assert "eval_reserve_shortfall_rate" in result.metrics
        assert "eval_cost_per_step" in result.metrics
        w = jnp.asarray(result.metrics["eval_wall_time_s"]).flatten()
        r = jnp.asarray(result.metrics["eval_returns"]).flatten()
        s = jnp.asarray(result.metrics["eval_reserve_shortfall_rate"]).flatten()
        c = jnp.asarray(result.metrics["eval_cost_per_step"]).flatten()
        assert w.shape == r.shape
        assert s.shape == r.shape
        assert c.shape == r.shape
        assert jnp.all(jnp.diff(w) >= 0.0)

    def test_saute_ppo_checkpoint_capture(self):
        env = _make_trans_saute()
        config = TrainConfig(
            algo="saute_ppo",
            total_timesteps=1_000,
            num_envs=4,
            n_steps=8,
            n_epochs=1,
            n_minibatches=2,
            eval_freq=256,
            n_checkpoints=3,
            cost_threshold=5.0,
        )
        train_fn = make_train(env, config)
        result = train_fn(jax.random.PRNGKey(0))
        assert isinstance(result, TrainResult)
        assert result.checkpoints is not None
        assert len(result.checkpoints) == 3
        timesteps = [int(step) for step, _ in result.checkpoints]
        assert timesteps == sorted(timesteps)
        assert timesteps[-1] > 0
        walltimes = jnp.asarray(result.metrics["checkpoint_walltime_s"]).flatten()
        assert walltimes.shape == (3,)
        assert jnp.all(jnp.diff(walltimes) >= 0.0)

    def test_saute_ppo_transgrid_runs(self):
        env = _make_trans_saute()
        config = TrainConfig(
            algo="saute_ppo",
            total_timesteps=1_000,
            num_envs=4,
            n_steps=8,
            n_epochs=1,
            n_minibatches=2,
            cost_threshold=5.0,
        )
        train_fn = make_train(env, config)
        result = train_fn(jax.random.PRNGKey(0))
        assert isinstance(result, TrainResult)

    def test_saute_ppo_eval_policy_is_deterministic(self):
        env = _make_trans_saute()
        config = TrainConfig(
            algo="saute_ppo",
            total_timesteps=1_000,
            num_envs=4,
            n_steps=8,
            n_epochs=1,
            n_minibatches=2,
            cost_threshold=5.0,
        )
        train_fn = make_train(env, config)
        result = train_fn(jax.random.PRNGKey(0))

        policy_fn = make_policy_fn(
            "saute_ppo",
            result.params,
            env._env,
            env._params,
            config,
            selected_names=env.selected_constraint_names,
        )
        obs, state = env.reset(jax.random.PRNGKey(123))
        action_a = policy_fn(obs, state, jax.random.PRNGKey(1))
        action_b = policy_fn(obs, state, jax.random.PRNGKey(2))
        assert jnp.allclose(action_a, action_b)

    def test_ppo_beta_eval_policy_is_deterministic(self):
        env = _make_battery_log()
        config = TrainConfig(
            algo="ppo",
            continuous_action_dist="beta",
            total_timesteps=1_000,
            num_envs=4,
            n_steps=8,
            n_epochs=1,
            n_minibatches=2,
        )
        train_fn = make_train(env, config)
        result = train_fn(jax.random.PRNGKey(0))
        policy_fn = make_policy_fn("ppo", result.params, env._env, env._params, config)
        obs, state = env.reset(jax.random.PRNGKey(123))
        action_a = policy_fn(obs, state, jax.random.PRNGKey(1))
        action_b = policy_fn(obs, state, jax.random.PRNGKey(2))
        assert jnp.allclose(action_a, action_b)
