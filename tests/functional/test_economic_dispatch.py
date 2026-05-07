"""Functional Test 3 — Economic Dispatch.

Verifies cost/reward signals are physically meaningful:
cost increases with generation, reward is negative of scaled cost.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from powerzoojax.envs.grid.trans import TransGridEnv, make_trans_params
from powerzoojax.case import create_case5

from .conftest import save_figure, write_report, report_header

CATEGORY = "economic_dispatch"
KEY = jax.random.PRNGKey(7)


def _single_step_at_level(env, params, level: float):
    """Step once with constant action level, return cost and reward."""
    case = params.case
    obs, state = env.reset(KEY, params)
    action = jnp.full((case.n_units,), level)
    _, state2, reward, _costs, _, info = env.step(KEY, state, action, params)
    return float(info["gen_cost"]), float(reward), np.asarray(state2.unit_power_mw)


class TestCostMonotonicity:

    def test_cost_increases_with_generation(self, trans_env, trans_params_flat):
        """More generation → higher cost. Test 5 levels from -1 to 1."""
        levels = np.linspace(-1.0, 1.0, 5)
        costs = []
        for lv in levels:
            cost, _, _ = _single_step_at_level(trans_env, trans_params_flat, float(lv))
            costs.append(cost)
        costs = np.array(costs)
        for i in range(len(costs) - 1):
            assert costs[i] <= costs[i + 1] + 1e-4, (
                f"Cost should increase: level {levels[i]:.1f}→{costs[i]:.2f} "
                f"vs level {levels[i+1]:.1f}→{costs[i+1]:.2f}"
            )

    def test_reward_is_negative_cost(self, trans_env, trans_params_flat):
        """Reward = -reward_scale * gen_cost, so higher cost → more negative reward."""
        cost_lo, rew_lo, _ = _single_step_at_level(trans_env, trans_params_flat, -1.0)
        cost_hi, rew_hi, _ = _single_step_at_level(trans_env, trans_params_flat, 1.0)
        assert rew_lo >= rew_hi, "Min generation should yield less negative reward than max"
        assert rew_hi < 0, "Reward should be negative (cost-based)"
        assert cost_hi >= cost_lo, "Max gen should cost at least as much as min gen"

    def test_cost_nonnegative(self, trans_env, trans_params_flat):
        """Generation cost should always be non-negative."""
        for lv in [-1.0, 0.0, 1.0]:
            c, _, _ = _single_step_at_level(trans_env, trans_params_flat, lv)
            assert c >= 0, f"Cost should be non-negative at action level {lv}"


def test_generate_report(trans_env, trans_params_flat):
    """Generate economic dispatch report."""
    env, params = trans_env, trans_params_flat

    levels = np.linspace(-1.0, 1.0, 11)
    costs, rewards, gen_totals = [], [], []
    for lv in levels:
        c, r, p = _single_step_at_level(env, params, float(lv))
        costs.append(c)
        rewards.append(r)
        gen_totals.append(np.sum(p))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].bar(levels, costs, width=0.15, color="C3", alpha=0.7)
    axes[0].set_xlabel("Normalized Action Level")
    axes[0].set_ylabel("Gen Cost ($/step)")
    axes[0].set_title("Cost vs Dispatch Level")
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(levels, rewards, width=0.15, color="C0", alpha=0.7)
    axes[1].set_xlabel("Normalized Action Level")
    axes[1].set_ylabel("Reward")
    axes[1].set_title("Reward vs Dispatch Level")
    axes[1].grid(True, alpha=0.3)

    axes[2].bar(levels, gen_totals, width=0.15, color="C2", alpha=0.7)
    axes[2].set_xlabel("Normalized Action Level")
    axes[2].set_ylabel("Total Gen (MW)")
    axes[2].set_title("Total Generation vs Action Level")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, "cost_reward_gen", CATEGORY)

    lines = [report_header("Economic Dispatch", 3, 3)]
    lines.append("## Cost / Reward / Generation at Different Action Levels\n")
    lines.append("| Action | Gen (MW) | Cost ($/step) | Reward |")
    lines.append("|--------|---------|--------------|--------|")
    for i, lv in enumerate(levels):
        lines.append(f"| {lv:+.1f} | {gen_totals[i]:.1f} | {costs[i]:.2f} | {rewards[i]:.4f} |")
    lines.append("\n![Cost/Reward/Gen](cost_reward_gen.png)\n")
    lines.append("**Key finding**: Cost increases monotonically with generation level. "
                 "Reward is inversely proportional (more negative = higher cost).\n")
    write_report("\n".join(lines), CATEGORY)
