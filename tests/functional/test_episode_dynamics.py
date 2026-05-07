"""Functional Test 6 — Episode Dynamics with Real Data.

End-to-end test with GB load profiles: load variation, episode rollout
statistics, and auto-reset behavior.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from powerzoojax.envs.grid.trans import TransGridEnv

from .conftest import save_figure, write_report, report_header

CATEGORY = "episode_dynamics"
KEY = jax.random.PRNGKey(777)


def _run_full_episode(env, params, key):
    """Run a full episode with random actions in [-1, 1]."""
    case = params.case
    T = params.max_steps
    obs, state = env.reset(key, params)

    rewards, costs, violations, gen_totals, load_totals = [], [], [], [], []
    for t in range(T):
        key, subkey = jax.random.split(key)
        action = jax.random.uniform(subkey, shape=(case.n_units,), minval=-1.0, maxval=1.0)
        obs, state, reward, costs_vec, done, info = env.step(
            subkey, state, action, params
        )
        rewards.append(float(reward))
        costs.append(float(info["gen_cost"]))
        violations.append(int(info["n_violations"]))
        gen_totals.append(float(jnp.sum(state.unit_power_mw)))
        t_idx = int(state.time_step - 1) % params.load_profiles.shape[0]
        load_totals.append(float(jnp.sum(case.nodes_loads_map @ params.load_profiles[t_idx])))

    return {
        "rewards": np.array(rewards),
        "costs": np.array(costs),
        "violations": np.array(violations),
        "gen_totals": np.array(gen_totals),
        "load_totals": np.array(load_totals),
        "T": T,
    }


class TestLoadVariation:

    def test_load_varies_over_episode(self, trans_env, trans_params_with_profiles):
        """With real profiles, load should vary over time (not flat)."""
        params = trans_params_with_profiles
        case = params.case
        T = min(params.max_steps, 48)
        loads = []
        for t in range(T):
            t_idx = t % params.load_profiles.shape[0]
            total = float(jnp.sum(case.nodes_loads_map @ params.load_profiles[t_idx]))
            loads.append(total)
        loads = np.array(loads)
        assert loads.std() > 1e-3, "Load should vary with real profiles"
        assert loads.min() < loads.max(), "Load should not be constant"

    def test_auto_reset_at_boundary(self, trans_env, trans_params_flat):
        """Running past max_steps should auto-reset."""
        env, params = trans_env, trans_params_flat
        case = params.case
        obs, state = env.reset(KEY, params)
        T = params.max_steps

        for t in range(T + 5):
            k = jax.random.fold_in(KEY, t)
            action = jnp.zeros(case.n_units)
            obs, state, reward, costs, done, info = env.step(k, state, action, params)
            if t == T - 1:
                assert bool(done), f"Should be done at step {T}"
            if t == T:
                assert int(state.time_step) <= 2, "Should have reset after done"


def test_generate_report(trans_env, trans_params_with_profiles, key):
    """Generate episode dynamics report with real data."""
    env = trans_env
    params = trans_params_with_profiles
    T = params.max_steps
    data = _run_full_episode(env, params, key)
    hours = np.arange(T) * 0.5

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # Gen vs Load
    axes[0, 0].plot(hours, data["load_totals"], label="Load", linewidth=0.8)
    axes[0, 0].plot(hours, data["gen_totals"], label="Gen", linewidth=0.8, linestyle="--")
    axes[0, 0].set_ylabel("Power (MW)")
    axes[0, 0].set_title("Generation vs Load (Random Policy)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    # Reward
    axes[0, 1].plot(hours, np.cumsum(data["rewards"]), linewidth=1, color="C2")
    axes[0, 1].set_ylabel("Cumulative Reward")
    axes[0, 1].set_title("Cumulative Reward")
    axes[0, 1].grid(True, alpha=0.3)

    # Cost
    axes[1, 0].fill_between(hours, data["costs"], alpha=0.3, color="C3")
    axes[1, 0].plot(hours, data["costs"], linewidth=0.8, color="C3")
    axes[1, 0].set_xlabel("Hours")
    axes[1, 0].set_ylabel("Gen Cost ($/step)")
    axes[1, 0].set_title("Generation Cost")
    axes[1, 0].grid(True, alpha=0.3)

    # Violations
    colors = ["C2" if v == 0 else "C1" for v in data["violations"]]
    axes[1, 1].bar(hours, data["violations"], width=0.4, color=colors, alpha=0.7)
    axes[1, 1].set_xlabel("Hours")
    axes[1, 1].set_ylabel("# Violations")
    viol_pct = (data["violations"] > 0).sum() / T * 100
    axes[1, 1].set_title(f"Violations ({viol_pct:.0f}% of steps)")
    axes[1, 1].grid(True, alpha=0.3)

    for ax in axes.flat:
        for d in range(1, 8):
            ax.axvline(d * 24, color="grey", linestyle=":", alpha=0.4)

    fig.suptitle(f"Episode Dynamics — {T} steps, GB Load Profiles", fontsize=12)
    plt.tight_layout()
    save_figure(fig, "episode_overview", CATEGORY)

    lines = [report_header("Episode Dynamics", 2, 2)]
    lines.append(f"## Episode Statistics ({T} steps, random policy)\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total reward | {data['rewards'].sum():.2f} |")
    lines.append(f"| Mean reward/step | {data['rewards'].mean():.4f} |")
    lines.append(f"| Mean gen cost | {data['costs'].mean():.2f} $/step |")
    lines.append(f"| Total violations | {data['violations'].sum()} |")
    lines.append(f"| Steps with violations | {(data['violations'] > 0).sum()}/{T} ({viol_pct:.0f}%) |")
    lines.append(f"| Load range | {data['load_totals'].min():.0f} – {data['load_totals'].max():.0f} MW |")
    lines.append(f"| Gen range | {data['gen_totals'].min():.0f} – {data['gen_totals'].max():.0f} MW |")
    lines.append(f"| Mean supply-demand gap | {np.mean(data['gen_totals'] - data['load_totals']):+.1f} MW |")
    lines.append("\n![Episode Overview](episode_overview.png)\n")
    write_report("\n".join(lines), CATEGORY)
