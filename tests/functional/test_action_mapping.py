"""Functional Test 1 — Action Mapping Verification.

Verifies that [-1, 1] normalized actions map correctly to physical
ranges for every env type. Generates sweep plots and boundary tables.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pytest

from powerzoojax.envs.base import denormalize_action
from powerzoojax.envs.grid.trans import TransGridEnv, make_trans_params
from powerzoojax.envs.grid.power_flow import dc_power_flow
from powerzoojax.envs.grid.dist import DistGridEnv, make_dist_params
from powerzoojax.envs.resource.battery import BatteryEnv, make_battery_params
from powerzoojax.envs.resource.renewable import RenewableEnv, SolarEnv
from powerzoojax.envs.resource.vehicle import VehicleEnv, make_vehicle_params
from powerzoojax.envs.resource.flexload import FlexLoadEnv, FlexLoadParams
from powerzoojax.case import create_case5, create_case33bw

from .conftest import save_figure, write_report, report_header, get_report_dir

CATEGORY = "action_mapping"
KEY = jax.random.PRNGKey(0)


def _sweep_transgrid():
    """Sweep TransGridEnv action from -1 to 1, record actual unit dispatch."""
    env = TransGridEnv()
    case = create_case5()
    params = make_trans_params(case, max_steps=48)
    obs, state = env.reset(KEY, params)
    node_load_mw = case.nodes_loads_map @ params.load_profiles[0]

    actions = np.linspace(-1.0, 1.0, 21)
    results = []
    targets = []
    for a_val in actions:
        action = jnp.full((case.n_units,), a_val)
        target = denormalize_action(action, case.unit_p_min, case.unit_p_max)
        _, s, _, _costs, _, _ = env.step(KEY, state, action, params)
        results.append(np.asarray(s.unit_power_mw))
        targets.append(np.asarray(target))
    return actions, np.stack(targets), np.stack(results), case, node_load_mw


def _sweep_battery():
    """Sweep BatteryEnv action, record current_p_mw."""
    env = BatteryEnv()
    params = make_battery_params()
    obs, state = env.reset(KEY, params)

    actions = np.linspace(-1.0, 1.0, 21)
    powers = []
    for a_val in actions:
        _, s, _, _costs, _, _ = env.step(KEY, state, jnp.float32(a_val), params)
        powers.append(float(s.current_p_mw))
    return actions, np.array(powers), params


class TestActionSweep:

    def test_transgrid_sweep_linear(self):
        actions, targets, results, case, node_load_mw = _sweep_transgrid()
        p_min = np.asarray(case.unit_p_min)
        p_max = np.asarray(case.unit_p_max)

        fig, axes = plt.subplots(1, min(case.n_units, 5), figsize=(14, 3.5), sharey=False)
        if case.n_units == 1:
            axes = [axes]
        for i in range(min(case.n_units, 5)):
            axes[i].plot(actions, results[:, i], "o-", markersize=3)
            axes[i].plot(actions, targets[:, i], "--", linewidth=1.0, label="Target")
            axes[i].axhline(p_min[i], color="red", linestyle=":", alpha=0.5, label="Pmin")
            axes[i].axhline(p_max[i], color="green", linestyle=":", alpha=0.5, label="Pmax")
            axes[i].set_xlabel("Normalized action")
            axes[i].set_ylabel("MW")
            axes[i].set_title(f"Unit {i}")
            axes[i].legend(fontsize=7)
            axes[i].grid(True, alpha=0.3)
        fig.suptitle("TransGridEnv: Action → Physical MW", fontsize=11)
        plt.tight_layout()
        save_figure(fig, "transgrid_sweep", CATEGORY)

        for j, target in enumerate(targets):
            _, _, expected_actual = dc_power_flow(
                case,
                jnp.asarray(target),
                node_load_mw,
            )
            np.testing.assert_allclose(results[j], np.asarray(expected_actual), atol=1e-3)

    def test_battery_sweep_linear(self):
        actions, powers, params = _sweep_battery()
        P = float(params.power_mw)

        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(actions, powers, "o-", markersize=3)
        ax.axhline(-P, color="red", linestyle=":", alpha=0.5, label=f"-P ({-P})")
        ax.axhline(P, color="green", linestyle=":", alpha=0.5, label=f"+P ({P})")
        ax.axhline(0, color="grey", linestyle="-", alpha=0.3)
        ax.set_xlabel("Normalized action")
        ax.set_ylabel("Power (MW)")
        ax.set_title("BatteryEnv: Action → Physical Power")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_figure(fig, "battery_sweep", CATEGORY)

        assert powers[0] < 0, "action=-1 should charge (negative)"
        assert powers[-1] > 0, "action=+1 should discharge (positive)"
        np.testing.assert_allclose(powers[10], 0.0, atol=0.5)


class TestActionBoundary:

    def test_transgrid_boundaries(self):
        env = TransGridEnv()
        case = create_case5()
        params = make_trans_params(case, max_steps=48)
        _, state = env.reset(KEY, params)
        node_load_mw = case.nodes_loads_map @ params.load_profiles[0]

        _, s_min, _, _costs, _, _ = env.step(KEY, state, -jnp.ones(case.n_units), params)
        _, s_mid, _, _costs, _, _ = env.step(KEY, state, jnp.zeros(case.n_units), params)
        _, s_max, _, _costs, _, _ = env.step(KEY, state, jnp.ones(case.n_units), params)

        target_min = denormalize_action(-jnp.ones(case.n_units), case.unit_p_min, case.unit_p_max)
        target_mid = denormalize_action(jnp.zeros(case.n_units), case.unit_p_min, case.unit_p_max)
        target_max = denormalize_action(jnp.ones(case.n_units), case.unit_p_min, case.unit_p_max)

        _, _, expected_min = dc_power_flow(case, target_min, node_load_mw)
        _, _, expected_mid = dc_power_flow(case, target_mid, node_load_mw)
        _, _, expected_max = dc_power_flow(case, target_max, node_load_mw)

        np.testing.assert_allclose(np.asarray(s_min.unit_power_mw), np.asarray(expected_min), atol=1e-3)
        np.testing.assert_allclose(np.asarray(s_mid.unit_power_mw), np.asarray(expected_mid), atol=1e-3)
        np.testing.assert_allclose(np.asarray(s_max.unit_power_mw), np.asarray(expected_max), atol=1e-3)

    def test_clip_beyond_range(self):
        env = TransGridEnv()
        case = create_case5()
        params = make_trans_params(case, max_steps=48)
        _, state = env.reset(KEY, params)
        node_load_mw = case.nodes_loads_map @ params.load_profiles[0]

        _, s_clip, _, _costs, _, _ = env.step(KEY, state, jnp.ones(case.n_units) * 10.0, params)
        target_clip = denormalize_action(jnp.ones(case.n_units) * 10.0, case.unit_p_min, case.unit_p_max)
        _, _, expected_clip = dc_power_flow(case, target_clip, node_load_mw)
        np.testing.assert_allclose(np.asarray(s_clip.unit_power_mw), np.asarray(expected_clip), atol=1e-3)


class TestActionSpaceDeclaration:

    @pytest.mark.parametrize("env_cls,params_factory", [
        (TransGridEnv, lambda: make_trans_params(create_case5())),
        (BatteryEnv, make_battery_params),
        (RenewableEnv, lambda: SolarEnv().default_params()),
        (VehicleEnv, make_vehicle_params),
        (FlexLoadEnv, FlexLoadParams),
    ])
    def test_all_envs_declare_neg1_to_1(self, env_cls, params_factory):
        params = params_factory()
        space = env_cls().action_space(params)
        if env_cls is FlexLoadEnv:
            # FlexLoad uses unit-scaled [0, 1] actions (curtail_frac, shift_out_frac)
            np.testing.assert_allclose(np.asarray(space.low), 0.0, atol=1e-6)
            np.testing.assert_allclose(np.asarray(space.high), 1.0, atol=1e-6)
        else:
            np.testing.assert_allclose(np.asarray(space.low), -1.0, atol=1e-6)
            np.testing.assert_allclose(np.asarray(space.high), 1.0, atol=1e-6)


def test_generate_report():
    """Generate the action mapping report (runs last)."""
    actions_tg, targets_tg, results_tg, case, _node_load = _sweep_transgrid()
    actions_bat, powers_bat, bat_params = _sweep_battery()

    lines = []
    lines.append(report_header("Action Mapping Verification", 4, 4))
    lines.append("## TransGridEnv Sweep (case5)\n")
    lines.append("| Action | " + " | ".join(f"Unit {i} (MW)" for i in range(case.n_units)) + " |")
    lines.append("|--------|" + "|".join("--------" for _ in range(case.n_units)) + "|")
    for j in range(0, 21, 5):
        row = f"| {actions_tg[j]:+.1f}   | " + " | ".join(f"{results_tg[j, i]:.1f}" for i in range(case.n_units)) + " |"
        lines.append(row)
    lines.append(f"\nPmin: {np.asarray(case.unit_p_min)}")
    lines.append(f"Pmax: {np.asarray(case.unit_p_max)}")
    lines.append("Reported MW are the env's actual post-slack dispatch, not the raw denormalized action target.")
    lines.append("\n![TransGrid Sweep](transgrid_sweep.png)\n")

    lines.append("## BatteryEnv Sweep\n")
    lines.append(f"Power rating: {float(bat_params.power_mw)} MW\n")
    lines.append("| Action | Power (MW) |")
    lines.append("|--------|-----------|")
    for j in range(0, 21, 5):
        lines.append(f"| {actions_bat[j]:+.1f}   | {powers_bat[j]:+.2f} |")
    lines.append("\n![Battery Sweep](battery_sweep.png)\n")

    write_report("\n".join(lines), CATEGORY)
