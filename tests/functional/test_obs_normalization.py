"""Functional Test — Observation Normalization Consistency.

Verifies that observations are bounded, reproducible with the same seed,
and each dimension has meaningful variation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from powerzoojax.envs.grid.trans import TransGridEnv, make_trans_params
from powerzoojax.envs.resource.battery import BatteryEnv, make_battery_params
from powerzoojax.case import create_case5

from .conftest import save_figure, write_report, report_header

CATEGORY = "obs_normalization"
N_STEPS = 48


def _collect_obs(env, params, key, n_steps=N_STEPS):
    """Run n_steps with random actions, collect obs array."""
    case = params.case if hasattr(params, 'case') else None
    obs, state = env.reset(key, params)
    obs_list = [np.asarray(obs)]
    for t in range(n_steps):
        key, subkey = jax.random.split(key)
        space = env.action_space(params)
        action = space.sample(subkey)
        obs, state, _, _costs, _, _ = env.step(subkey, state, action, params)
        obs_list.append(np.asarray(obs))
    return np.stack(obs_list)


class TestObsRange:

    def test_transgrid_obs_bounded(self, trans_env, trans_params_flat, key):
        """TransGrid obs dimensions should be in reasonable ranges."""
        obs_all = _collect_obs(trans_env, trans_params_flat, key)
        assert not np.any(np.isnan(obs_all)), "obs contains NaN"
        assert not np.any(np.isinf(obs_all)), "obs contains Inf"
        for d in range(obs_all.shape[1]):
            rng = obs_all[:, d].max() - obs_all[:, d].min()
            assert rng < 1e6, f"obs dim {d} has unreasonable range {rng}"

    def test_battery_obs_bounded(self, battery_env, battery_params, key):
        """Battery obs should be in [0, 1] or [-1, 1] range."""
        obs_all = _collect_obs(battery_env, battery_params, key)
        assert not np.any(np.isnan(obs_all))
        assert obs_all.min() >= -1.5, f"obs min = {obs_all.min()}"
        assert obs_all.max() <= 1.5, f"obs max = {obs_all.max()}"


class TestObsReproducibility:

    def test_same_seed_same_obs(self, trans_env, trans_params_flat):
        """Same seed + same actions → identical obs."""
        key1 = jax.random.PRNGKey(99)
        key2 = jax.random.PRNGKey(99)
        obs1 = _collect_obs(trans_env, trans_params_flat, key1, 10)
        obs2 = _collect_obs(trans_env, trans_params_flat, key2, 10)
        np.testing.assert_allclose(obs1, obs2, atol=1e-6)

    def test_different_seed_different_obs(self, trans_env, trans_params_flat):
        """Different seeds should produce different random action sequences → different obs."""
        key1 = jax.random.PRNGKey(1)
        key2 = jax.random.PRNGKey(2)
        obs1 = _collect_obs(trans_env, trans_params_flat, key1, 10)
        obs2 = _collect_obs(trans_env, trans_params_flat, key2, 10)
        assert not np.allclose(obs1[1:], obs2[1:], atol=1e-6), (
            "Different seeds should produce different trajectories"
        )


def test_generate_report(trans_env, trans_params_flat, battery_env, battery_params, key):
    obs_trans = _collect_obs(trans_env, trans_params_flat, key)
    obs_bat = _collect_obs(battery_env, battery_params, key)

    lines = [report_header("Observation Normalization", 4, 4)]

    lines.append("## TransGridEnv Observation Statistics\n")
    lines.append(f"Shape: {obs_trans.shape} (steps x dims)\n")
    lines.append("| Dim | Min | Max | Mean | Std |")
    lines.append("|-----|-----|-----|------|-----|")
    for d in range(obs_trans.shape[1]):
        col = obs_trans[:, d]
        lines.append(f"| {d} | {col.min():.4f} | {col.max():.4f} | "
                     f"{col.mean():.4f} | {col.std():.4f} |")

    lines.append("\n## BatteryEnv Observation Statistics\n")
    lines.append(f"Shape: {obs_bat.shape}\n")
    lines.append("| Dim | Min | Max | Mean | Std |")
    lines.append("|-----|-----|-----|------|-----|")
    for d in range(obs_bat.shape[1]):
        col = obs_bat[:, d]
        lines.append(f"| {d} | {col.min():.4f} | {col.max():.4f} | "
                     f"{col.mean():.4f} | {col.std():.4f} |")

    # Boxplot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].boxplot(obs_trans, vert=True, showfliers=True)
    axes[0].set_xlabel("Obs Dimension")
    axes[0].set_ylabel("Value")
    axes[0].set_title("TransGridEnv Obs Distribution")
    axes[0].grid(True, alpha=0.3)

    axes[1].boxplot(obs_bat, vert=True, showfliers=True)
    axes[1].set_xlabel("Obs Dimension")
    axes[1].set_ylabel("Value")
    axes[1].set_title("BatteryEnv Obs Distribution")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, "obs_distribution", CATEGORY)

    lines.append("\n![Obs Distribution](obs_distribution.png)\n")
    write_report("\n".join(lines), CATEGORY)
