"""Functional Test — Cross-Path Validation.

Verifies that direct env.step and LogWrapper produce consistent results
when given the same actions and seeds.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from powerzoojax.envs.grid.trans import TransGridEnv, make_trans_params
from powerzoojax.rl.wrappers import LogWrapper
from powerzoojax.case import create_case5

from .conftest import save_figure, write_report, report_header

CATEGORY = "cross_path_validation"
N_STEPS = 24


def _run_direct(env, params, key, actions):
    """Path A: env.step directly."""
    obs, state = env.reset(key, params)
    obs_list, rew_list = [], []
    for a in actions:
        key, subkey = jax.random.split(key)
        obs, state, reward, costs, done, info = env.step(subkey, state, a, params)
        obs_list.append(np.asarray(obs))
        rew_list.append(float(reward))
    return np.stack(obs_list), np.array(rew_list)


def _run_logwrapper(env, params, key, actions):
    """Path B: LogWrapper.step (binds params, adds episode tracking)."""
    wrapped = LogWrapper(env, params)
    obs, state = wrapped.reset(key)
    obs_list, rew_list = [], []
    for a in actions:
        key, subkey = jax.random.split(key)
        obs, state, reward, done, info = wrapped.step(subkey, state, a)
        obs_list.append(np.asarray(obs))
        rew_list.append(float(reward))
    return np.stack(obs_list), np.array(rew_list)


class TestCrossPath:

    def test_direct_vs_logwrapper_obs(self):
        case = create_case5()
        params = make_trans_params(case, max_steps=48)
        env = TransGridEnv()
        key = jax.random.PRNGKey(42)

        actions = [jnp.zeros(case.n_units) for _ in range(N_STEPS)]

        obs_a, rew_a = _run_direct(env, params, jax.random.PRNGKey(42), actions)
        obs_b, rew_b = _run_logwrapper(env, params, jax.random.PRNGKey(42), actions)

        np.testing.assert_allclose(obs_a, obs_b, atol=1e-5,
                                   err_msg="Direct vs LogWrapper obs mismatch")

    def test_direct_vs_logwrapper_reward(self):
        case = create_case5()
        params = make_trans_params(case, max_steps=48)
        env = TransGridEnv()

        actions = [jnp.zeros(case.n_units) for _ in range(N_STEPS)]

        _, rew_a = _run_direct(env, params, jax.random.PRNGKey(42), actions)
        _, rew_b = _run_logwrapper(env, params, jax.random.PRNGKey(42), actions)

        np.testing.assert_allclose(rew_a, rew_b, atol=1e-5,
                                   err_msg="Direct vs LogWrapper reward mismatch")


def test_generate_report():
    case = create_case5()
    params = make_trans_params(case, max_steps=48)
    env = TransGridEnv()
    key = jax.random.PRNGKey(42)

    actions = [jnp.zeros(case.n_units) for _ in range(N_STEPS)]

    obs_a, rew_a = _run_direct(env, params, jax.random.PRNGKey(42), actions)
    obs_b, rew_b = _run_logwrapper(env, params, jax.random.PRNGKey(42), actions)

    obs_diff = np.abs(obs_a - obs_b)
    rew_diff = np.abs(rew_a - rew_b)

    lines = [report_header("Cross-Path Validation", 2, 2)]
    lines.append("## Path Comparison\n")
    lines.append("- **Path A**: `TransGridEnv.step(key, state, action, params)` direct")
    lines.append("- **Path B**: `LogWrapper.step(key, state, action)` (params bound)\n")
    lines.append(f"Steps: {N_STEPS}\n")
    lines.append("## Differences\n")
    lines.append(f"| Metric | Max |diff| | Mean |diff| |")
    lines.append(f"|--------|-----------|------------|")
    lines.append(f"| Observation | {obs_diff.max():.2e} | {obs_diff.mean():.2e} |")
    lines.append(f"| Reward | {rew_diff.max():.2e} | {rew_diff.mean():.2e} |")

    obs_ok = obs_diff.max() < 1e-4
    rew_ok = rew_diff.max() < 1e-4
    lines.append(f"\n**Verdict**: {'PASS' if obs_ok and rew_ok else 'FAIL'} — "
                 f"{'paths produce identical results' if obs_ok and rew_ok else 'paths diverge'}\n")

    # Heatmap
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if obs_diff.max() > 0:
        im = axes[0].imshow(obs_diff.T, aspect="auto", cmap="hot", vmin=0)
        plt.colorbar(im, ax=axes[0])
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Obs Dim")
    axes[0].set_title(f"Obs |diff| (max={obs_diff.max():.2e})")

    axes[1].bar(range(len(rew_diff)), rew_diff, color="C3", alpha=0.7)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("|Reward diff|")
    axes[1].set_title(f"Reward |diff| (max={rew_diff.max():.2e})")

    plt.tight_layout()
    save_figure(fig, "path_comparison", CATEGORY)

    lines.append("![Path Comparison](path_comparison.png)\n")
    write_report("\n".join(lines), CATEGORY)
