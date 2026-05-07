"""Functional Test — Reward / Cost Separation.

Project contract: reward carries only the economic objective;
safety penalties flow exclusively through the cost / info channel.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from powerzoojax.envs.grid.trans import TransGridEnv, make_trans_params
from powerzoojax.case import create_case5

from .conftest import save_figure, write_report, report_header

CATEGORY = "reward_cost_separation"
KEY = jax.random.PRNGKey(42)
N_STEPS = 48


def _run_strategy(env, params, action_level, n_steps=N_STEPS):
    """Run with constant normalized action, collect reward/cost/safety."""
    case = params.case
    obs, state = env.reset(KEY, params)
    records = []
    for t in range(n_steps):
        k = jax.random.fold_in(KEY, t)
        action = jnp.full((case.n_units,), action_level)
        obs, state, reward, costs, done, info = env.step(k, state, action, params)
        records.append(dict(
            reward=float(reward),
            gen_cost=float(info["gen_cost"]),
            cost_thermal_overload=float(info["cost_thermal_overload"]),
            cost_voltage_violation=float(info.get("cost_voltage_violation", 0.0)),
            cost_power_balance=float(info.get("cost_power_balance", 0.0)),
            cost_resource=float(info.get("cost_resource", 0.0)),
            cost=float(info["cost_sum"]),
            is_safe=bool(info["is_safe"]),
            n_violations=int(info["n_violations"]),
        ))
    return records


class TestRewardCostSeparation:

    def test_reward_contains_no_safety_penalty(self, trans_env, trans_params_flat):
        """Reward should only reflect economic cost, not safety violations."""
        records = _run_strategy(trans_env, trans_params_flat, 0.0, 10)
        for r in records:
            expected_reward = -trans_params_flat.reward_scale * r["gen_cost"]
            np.testing.assert_allclose(
                r["reward"], expected_reward, atol=1e-4,
                err_msg="Reward should be -reward_scale * gen_cost only"
            )

    def test_cost_contains_safety_penalties(self, trans_env, trans_params_flat):
        """Aggregate cost should equal the sum of all routed penalty channels."""
        records_safe = _run_strategy(trans_env, trans_params_flat, 0.2, 5)
        records_extreme = _run_strategy(trans_env, trans_params_flat, -1.0, 5)

        for r in records_safe + records_extreme:
            expected_cost = (
                r["cost_thermal_overload"]
                + r["cost_voltage_violation"]
                + r["cost_power_balance"]
                + r["cost_resource"]
            )
            np.testing.assert_allclose(
                r["cost"], expected_cost, atol=1e-4,
                err_msg="cost_sum should equal the routed aggregate penalty"
            )

    def test_unsafe_steps_have_positive_cost(self, trans_env, trans_params_flat):
        """When violations occur, cost should be > 0."""
        records = _run_strategy(trans_env, trans_params_flat, -1.0, N_STEPS)
        for r in records:
            if r["n_violations"] > 0:
                assert r["cost"] > 0, (
                    f"Violations={r['n_violations']} but cost={r['cost']}"
                )


def test_generate_report(trans_env, trans_params_flat):
    strategies = {
        "Pmin (a=-1)": _run_strategy(trans_env, trans_params_flat, -1.0),
        "Mid (a=0)": _run_strategy(trans_env, trans_params_flat, 0.0),
        "Pmax (a=+1)": _run_strategy(trans_env, trans_params_flat, 1.0),
    }

    lines = [report_header("Reward / Cost Separation", 3, 3)]
    lines.append("## Per-Strategy Statistics\n")
    lines.append("| Strategy | Mean Reward | Mean Gen Cost | Mean Safety Cost | Unsafe Steps |")
    lines.append("|----------|-----------|-------------|----------------|-------------|")

    fig, axes = plt.subplots(len(strategies), 1, figsize=(12, 3 * len(strategies)), sharex=True)

    for idx, (name, records) in enumerate(strategies.items()):
        rews = np.array([r["reward"] for r in records])
        gcosts = np.array([r["gen_cost"] for r in records])
        scosts = np.array([r["cost"] for r in records])
        unsafe = sum(1 for r in records if not r["is_safe"])

        lines.append(f"| {name} | {rews.mean():.2f} | {gcosts.mean():.1f} | "
                     f"{scosts.mean():.4f} | {unsafe}/{len(records)} |")

        steps = np.arange(len(records))
        axes[idx].plot(steps, rews, label="Reward", color="C0")
        ax2 = axes[idx].twinx()
        ax2.plot(steps, scosts, label="Safety Cost", color="C3", linestyle="--")
        axes[idx].set_ylabel("Reward")
        ax2.set_ylabel("Safety Cost")
        axes[idx].set_title(name)
        axes[idx].legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
        axes[idx].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Step")
    plt.tight_layout()
    save_figure(fig, "reward_cost_timeseries", CATEGORY)

    lines.append("\n![Reward vs Cost](reward_cost_timeseries.png)\n")
    lines.append("**Key contract**: `reward = -reward_scale * gen_cost` (economic only). "
                 "`info['cost_sum']` carries the aggregate routed penalty channels.\n")
    write_report("\n".join(lines), CATEGORY)
