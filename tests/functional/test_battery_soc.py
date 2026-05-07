"""Functional Test 5 — Battery SOC Dynamics.

Verifies battery physics: charge/discharge trajectories,
energy conservation, and efficiency accounting.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from powerzoojax.envs.resource.battery import BatteryEnv, make_battery_params

from .conftest import save_figure, write_report, report_header

CATEGORY = "battery_soc"
KEY = jax.random.PRNGKey(123)


def _run_battery_episode(params, action_val: float, n_steps: int = 48):
    """Run battery with constant normalized action, return trajectories."""
    env = BatteryEnv()
    obs, state = env.reset(KEY, params)
    socs, powers = [float(state.soc)], []
    for t in range(n_steps):
        k = jax.random.fold_in(KEY, t)
        _, state, *_ = env.step(k, state, jnp.float32(action_val), params)
        socs.append(float(state.soc))
        powers.append(float(state.current_p_mw))
    return np.array(socs), np.array(powers)


class TestSOCTrajectory:

    def test_charge_increases_soc(self):
        params = make_battery_params(initial_soc=0.3)
        socs, powers = _run_battery_episode(params, -1.0, 20)
        assert socs[-1] > socs[0], "Charging should increase SOC"
        diffs = np.diff(socs[:15])
        assert np.all(diffs >= -1e-6), "SOC should be non-decreasing while charging"

    def test_discharge_decreases_soc(self):
        params = make_battery_params(initial_soc=0.8)
        socs, powers = _run_battery_episode(params, 1.0, 20)
        assert socs[-1] < socs[0], "Discharging should decrease SOC"
        diffs = np.diff(socs[:15])
        assert np.all(diffs <= 1e-6), "SOC should be non-increasing while discharging"

    def test_soc_bounded(self):
        params = make_battery_params(initial_soc=0.5, soc_min=0.1, soc_max=0.9)
        socs_charge, _ = _run_battery_episode(params, -1.0, 48)
        socs_discharge, _ = _run_battery_episode(params, 1.0, 48)
        assert np.all(np.array(socs_charge) <= 0.9 + 1e-5)
        assert np.all(np.array(socs_discharge) >= 0.1 - 1e-5)

    def test_idle_preserves_soc(self):
        params = make_battery_params(initial_soc=0.5)
        socs, powers = _run_battery_episode(params, 0.0, 10)
        np.testing.assert_allclose(socs[0], socs[-1], atol=1e-4)


class TestCycleCostTrajectory:
    """Functional check: `costs[0]` tracks |P|·Δt·λ over an episode."""

    def test_per_step_cost_matches_throughput_times_rate(self):
        rate = 10.0
        dt = 0.5
        params = make_battery_params(
            initial_soc=0.5,
            cycle_cost_per_mwh=rate,
            delta_t_hours=dt,
            power_mw=20.0,
            max_steps=100,
        )
        env = BatteryEnv()
        obs, state = env.reset(KEY, params)
        for t in range(5):
            k = jax.random.fold_in(KEY, t)
            _, state, _, costs, _, info = env.step(k, state, jnp.float32(0.5), params)
            p = float(state.current_p_mw)
            expected = abs(p) * dt * rate
            np.testing.assert_allclose(float(costs[0]), expected, rtol=1e-5)

    def test_accumulated_cost_matches_sum_of_per_step(self):
        rate = 2.0
        params = make_battery_params(
            initial_soc=0.6,
            cycle_cost_per_mwh=rate,
            max_steps=100,
        )
        env = BatteryEnv()
        dt = float(params.delta_t_hours)
        obs, state = env.reset(KEY, params)
        total = 0.0
        for t in range(8):
            k = jax.random.fold_in(KEY, t)
            _, state, _, costs, _, info = env.step(k, state, jnp.float32(0.4), params)
            p = float(state.current_p_mw)
            total += abs(p) * dt * rate
        # Re-run same actions and sum explicit throughput costs
        obs, state = env.reset(KEY, params)
        sum_info = 0.0
        for t in range(8):
            k = jax.random.fold_in(KEY, t)
            _, state, _, costs, _, info = env.step(k, state, jnp.float32(0.4), params)
            sum_info += float(costs[0])
        np.testing.assert_allclose(sum_info, total, rtol=1e-5)


class TestEnergyConservation:

    def test_charge_energy_accounting(self):
        """Integral of power * dt should match SOC change * capacity (with efficiency)."""
        params = make_battery_params(initial_soc=0.3, eta_roundtrip=0.81)
        socs, powers = _run_battery_episode(params, -0.5, 30)

        dt = float(params.delta_t_hours)
        cap = float(params.capacity_mwh)
        eta_c = float(params.eta_charge)

        energy_in = -np.sum(powers) * dt
        soc_change = socs[-1] - socs[0]
        energy_stored = soc_change * cap

        np.testing.assert_allclose(energy_stored, energy_in * eta_c, rtol=0.05)


def test_generate_report():
    """Generate battery SOC dynamics report."""
    params = make_battery_params(initial_soc=0.5, eta_roundtrip=0.81, max_steps=200)
    cap = float(params.capacity_mwh)

    socs_charge, p_charge = _run_battery_episode(params, -1.0, 48)
    socs_discharge, p_discharge = _run_battery_episode(params, 1.0, 48)

    params_cycle = make_battery_params(initial_soc=0.5, eta_roundtrip=0.81, max_steps=200)
    env = BatteryEnv()
    obs, state = env.reset(KEY, params_cycle)
    socs_cycle = [float(state.soc)]
    for t in range(48):
        k = jax.random.fold_in(KEY, t)
        action = jnp.float32(-1.0) if t < 16 else (jnp.float32(1.0) if t < 32 else jnp.float32(-1.0))
        _, state, *_ = env.step(k, state, action, params_cycle)
        socs_cycle.append(float(state.soc))
    socs_cycle = np.array(socs_cycle)

    hours = np.arange(49) * 0.5

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(hours, socs_charge, "b-", label="Charge (a=-1)")
    axes[0].plot(hours, socs_discharge, "r-", label="Discharge (a=+1)")
    axes[0].axhline(float(params.soc_min), color="grey", linestyle=":", alpha=0.5)
    axes[0].axhline(float(params.soc_max), color="grey", linestyle=":", alpha=0.5)
    axes[0].set_xlabel("Hours")
    axes[0].set_ylabel("SOC")
    axes[0].set_title("Charge / Discharge Trajectory")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(hours, socs_cycle, "g-")
    axes[1].axhline(float(params.soc_min), color="grey", linestyle=":", alpha=0.5)
    axes[1].axhline(float(params.soc_max), color="grey", linestyle=":", alpha=0.5)
    axes[1].set_xlabel("Hours")
    axes[1].set_ylabel("SOC")
    axes[1].set_title("Charge-Discharge-Charge Cycle")
    axes[1].grid(True, alpha=0.3)

    dt = float(params.delta_t_hours)
    eta_c = float(params.eta_charge)
    eta_d = float(params.eta_discharge)
    e_in = -np.sum(p_charge) * dt
    soc_delta_c = socs_charge[-1] - socs_charge[0]
    e_out = np.sum(p_discharge) * dt
    soc_delta_d = socs_discharge[0] - socs_discharge[-1]

    labels = ["E_in (MWh)", "Stored (MWh)", "E_out (MWh)", "Released (MWh)"]
    values = [e_in, soc_delta_c * cap, e_out, soc_delta_d * cap]
    colors = ["C0", "C0", "C3", "C3"]
    axes[2].bar(labels, values, color=colors, alpha=0.7)
    axes[2].set_ylabel("Energy (MWh)")
    axes[2].set_title("Energy Accounting")
    axes[2].grid(True, alpha=0.3)
    plt.setp(axes[2].xaxis.get_majorticklabels(), fontsize=8)

    plt.tight_layout()
    save_figure(fig, "soc_dynamics", CATEGORY)

    lines = [report_header("Battery SOC Dynamics", 4, 4)]
    lines.append("## Trajectory Summary\n")
    lines.append(f"- Capacity: {cap} MWh, Power: {float(params.power_mw)} MW")
    lines.append(f"- Efficiency: charge={eta_c:.3f}, discharge={eta_d:.3f}")
    lines.append(f"- Charge: SOC {socs_charge[0]:.3f} → {socs_charge[-1]:.3f}")
    lines.append(f"- Discharge: SOC {socs_discharge[0]:.3f} → {socs_discharge[-1]:.3f}")
    lines.append(f"- Cycle final SOC: {socs_cycle[-1]:.3f} (started at {socs_cycle[0]:.3f})\n")
    lines.append("## Energy Accounting\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Energy in (charge) | {e_in:.2f} MWh |")
    lines.append(f"| SOC increase * cap | {soc_delta_c * cap:.2f} MWh |")
    lines.append(f"| Ratio (should ~ eta_c={eta_c:.3f}) | {(soc_delta_c * cap) / max(e_in, 1e-6):.3f} |")
    lines.append(f"| Energy out (discharge) | {e_out:.2f} MWh |")
    lines.append(f"| SOC decrease * cap | {soc_delta_d * cap:.2f} MWh |")
    lines.append("\n![SOC Dynamics](soc_dynamics.png)\n")
    write_report("\n".join(lines), CATEGORY)
