"""Functional Test 2 — Supply-Demand Balance.

Verifies power balance in TransGridEnv under different dispatch levels.
DC power flow should satisfy nodal balance exactly.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pytest

from powerzoojax.envs.grid.trans import TransGridEnv, make_trans_params
from powerzoojax.envs.base import denormalize_action
from powerzoojax.case import create_case5

from .conftest import save_figure, write_report, report_header

CATEGORY = "supply_demand"
KEY = jax.random.PRNGKey(42)


def _run_episode(env, params, action_level: float, n_steps: int = 48):
    """Run n_steps with constant normalized action, collect target/gen/load."""
    case = params.case
    obs, state = env.reset(KEY, params)
    targets, gens, loads = [], [], []
    for t in range(n_steps):
        k = jax.random.fold_in(KEY, t)
        action = jnp.full((case.n_units,), action_level)
        _, state, _, _, _, _ = env.step(k, state, action, params)
        target_dispatch = denormalize_action(action, case.unit_p_min, case.unit_p_max)
        targets.append(float(jnp.sum(target_dispatch)))
        total_gen = float(jnp.sum(state.unit_power_mw))
        t_idx = int(state.time_step - 1) % params.load_profiles.shape[0]
        load_demand = params.load_profiles[t_idx]
        total_load = float(jnp.sum(case.nodes_loads_map @ load_demand))
        gens.append(total_gen)
        loads.append(total_load)
    return np.array(targets), np.array(gens), np.array(loads)


class TestPowerBalance:

    def test_midpoint_dispatch(self, trans_env, trans_params_flat):
        """action=0 → midpoint dispatch. Check gen/load relationship."""
        targets, gens, loads = _run_episode(trans_env, trans_params_flat, 0.0, 48)
        assert len(gens) == 48
        assert np.all(gens > 0), "Generation should be positive"
        assert np.all(loads > 0), "Load should be positive"
        assert np.allclose(gens, loads, atol=1e-3), "Slack-adjusted dispatch should rebalance load"

    def test_max_dispatch_overgen(self, trans_env, trans_params_flat):
        """action=+1 requests over-generation before slack rebalancing."""
        targets, gens, loads = _run_episode(trans_env, trans_params_flat, 1.0, 10)
        excess = targets - loads
        assert np.mean(excess) > 0, "Max dispatch should over-generate on average"
        assert np.allclose(gens, loads, atol=1e-3), "Actual dispatch should rebalance to load"

    def test_min_dispatch_undergen(self, trans_env, trans_params_flat):
        """action=-1 requests under-generation before slack rebalancing."""
        targets, gens, loads = _run_episode(trans_env, trans_params_flat, -1.0, 10)
        deficit = loads - targets
        assert np.mean(deficit) > 0, "Min dispatch should under-generate on average"
        assert np.allclose(gens, loads, atol=1e-3), "Actual dispatch should rebalance to load"


def test_generate_report(trans_env, trans_params_flat):
    """Generate supply-demand report with plots."""
    env = trans_env
    params = trans_params_flat

    levels = {"Pmin (a=-1)": -1.0, "Mid (a=0)": 0.0, "Pmax (a=+1)": 1.0}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    report_lines = []
    report_lines.append(report_header("Supply-Demand Balance", 3, 3))
    report_lines.append("## Dispatch Level Comparison\n")
    report_lines.append("| Level | Mean Gen (MW) | Mean Load (MW) | Mean Imbalance (MW) |")
    report_lines.append("|-------|--------------|---------------|-------------------|")

    for idx, (label, level) in enumerate(levels.items()):
        targets, gens, loads = _run_episode(env, params, level, 48)
        imbalance = gens - loads
        requested_imbalance = targets - loads
        hours = np.arange(len(gens)) * 0.5

        axes[idx].plot(hours, gens, label="Gen", linewidth=1)
        axes[idx].plot(hours, loads, label="Load", linewidth=1)
        axes[idx].fill_between(hours, loads, gens, alpha=0.15,
                               color="red" if np.mean(imbalance) < 0 else "green")
        axes[idx].set_title(label)
        axes[idx].set_xlabel("Hours")
        axes[idx].legend(fontsize=7)
        axes[idx].grid(True, alpha=0.3)

        report_lines.append(
            f"| {label} | {np.mean(gens):.1f} | {np.mean(loads):.1f} | {np.mean(requested_imbalance):+.1f} |"
        )

    axes[0].set_ylabel("Power (MW)")
    fig.suptitle("Supply-Demand Balance at Different Dispatch Levels", fontsize=11)
    plt.tight_layout()
    save_figure(fig, "balance_comparison", CATEGORY)

    report_lines.append("\n![Balance Comparison](balance_comparison.png)\n")
    write_report("\n".join(report_lines), CATEGORY)
