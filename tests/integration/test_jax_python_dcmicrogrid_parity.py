"""Cross-backend parity test: PowerZooJax vs PowerZoo (Python) DC microgrid.

After the bundle / sub-resource refactor on both sides, we want a defence-in-
depth test that the **deterministic resource-level physics** are identical.

Why we don't compare *every* quantity step by step
--------------------------------------------------
The DataCenter sub-env uses different PRNGs on each backend (``jax.random``
vs ``numpy.random``).  Identical seeds produce different task arrivals, which
ripple into ``p_dc_mw`` / ``p_load_mw`` / ``t_zone`` / ``sla_violations``.
There is no reasonable way to align two different PRNG algorithms.

What we DO compare
------------------
The resource sub-systems (battery, PV, diesel) are fully deterministic given
the same action sequence and identical initial state.  Their per-step outputs
must match across backends to atol=1e-4:

    - ``p_pv_mw``     (profile-driven, no RNG)
    - ``p_batt_mw``   (action + SOC dynamics, no RNG)
    - ``p_dg_mw``     (action + clip, no RNG)
    - ``soc``         (Coulomb counting with losses, no RNG)
    - ``fuel_cost``   (derived from p_dg)
    - ``carbon_kg``   (derived from p_dg)

Stochastic / DC-driven quantities (p_dc_mw, p_load_mw, t_zone, sla, residual,
reward, cost) are intentionally not bit-compared.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import jax
import jax.numpy as jnp

pytestmark = pytest.mark.external


# Make the (nested) PowerZoo Python package importable when running from the
# PowerZooJax repo root.
_POWERZOO_ROOT = Path(__file__).resolve().parents[2] / "PowerZoo"
if str(_POWERZOO_ROOT) not in sys.path:
    sys.path.insert(0, str(_POWERZOO_ROOT))


@pytest.fixture(scope="module")
def fixed_action_sequence():
    """48 deterministic 5-D actions covering charge / discharge / DG idle / dispatch."""
    rng = np.random.default_rng(2026)
    a = rng.uniform(0.0, 1.0, size=(48, 5)).astype(np.float32)
    # Map slot 3 (battery) to [-1, 1]
    a[:, 3] = a[:, 3] * 2.0 - 1.0
    return a


def _build_jax_env(max_steps: int = 48):
    from powerzoojax.envs.microgrid import (
        DataCenterMicrogridEnv, make_dcmicrogrid_params,
    )
    env = DataCenterMicrogridEnv()
    params = make_dcmicrogrid_params(max_steps=max_steps)
    return env, params


def _build_python_env(max_steps: int = 48):
    from powerzoo.envs.microgrid import DCMicrogridEnv
    return DCMicrogridEnv(max_steps=max_steps)


# ---------------------------------------------------------------------------
# Defaults parity
# ---------------------------------------------------------------------------

class TestDefaultsParity:
    """Both sides must construct with identical numerical defaults."""

    def test_battery_defaults_match(self):
        _, jax_p = _build_jax_env()
        py_env = _build_python_env()
        assert jax_p.battery_capacity_mwh == pytest.approx(py_env._batt.capacity_mwh)
        assert jax_p.battery_power_mw == pytest.approx(py_env._batt.power_mw)
        assert jax_p.battery_eta_charge == pytest.approx(py_env._batt.eta_charge)
        assert jax_p.battery_eta_discharge == pytest.approx(py_env._batt.eta_discharge)
        assert jax_p.battery_soc_min == pytest.approx(py_env._batt.soc_min)
        assert jax_p.battery_soc_max == pytest.approx(py_env._batt.soc_max)
        assert jax_p.battery_soc_init == pytest.approx(py_env._batt.initial_soc)

    def test_pv_defaults_match(self):
        _, jax_p = _build_jax_env()
        py_env = _build_python_env()
        assert jax_p.pv_p_max_mw == pytest.approx(py_env._pv.capacity_mw)

    def test_diesel_defaults_match(self):
        _, jax_p = _build_jax_env()
        py_env = _build_python_env()
        assert jax_p.dg.p_dg_max_mw == pytest.approx(py_env._dg.p_dg_max_mw)
        assert jax_p.dg.fuel_cost_per_mwh == pytest.approx(py_env._dg.fuel_cost_per_mwh)
        assert jax_p.dg.emission_factor == pytest.approx(py_env._dg.emission_factor)

    def test_reward_weights_match(self):
        _, jax_p = _build_jax_env()
        py_env = _build_python_env()
        assert jax_p.w_cost == pytest.approx(py_env._w_cost)
        assert jax_p.w_carbon == pytest.approx(py_env._w_carbon)
        assert jax_p.battery_deg_cost_per_mwh == pytest.approx(
            py_env._battery_deg_cost_per_mwh
        )

    def test_dt_and_episode_length_match(self):
        _, jax_p = _build_jax_env()
        py_env = _build_python_env(max_steps=48)
        assert jax_p.dc.delta_t_hours == pytest.approx(py_env._dt_h)
        # episode length on both sides comes from the test fixture (48)
        assert py_env.max_steps == 48


# ---------------------------------------------------------------------------
# Resource-level deterministic parity
# ---------------------------------------------------------------------------

class TestResourceParity:
    """Battery / PV / DG outputs must match step-by-step under identical actions."""

    def _rollout_jax(self, actions):
        env, params = _build_jax_env(max_steps=actions.shape[0])
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)
        traj = []
        for t in range(actions.shape[0]):
            key, k = jax.random.split(key)
            a = jnp.asarray(actions[t])
            _, state, reward, _, _, info = env.step(k, state, a, params)
            traj.append({
                "p_pv": float(info["p_pv_mw"]),
                "p_dg": float(info["p_dg_mw"]),
                "p_batt": float(info["p_batt_mw"]),
                "soc": float(info["soc"]),
                "fuel_cost": float(info["fuel_cost"]),
                "carbon_kg": float(info["carbon_kg"]),
            })
        return traj

    def _rollout_python(self, actions):
        env = _build_python_env(max_steps=actions.shape[0])
        env.reset(seed=0)
        traj = []
        for t in range(actions.shape[0]):
            _, _, _, _, info = env.step(actions[t])
            traj.append({
                "p_pv": float(info["p_pv_mw"]),
                "p_dg": float(info["p_dg_mw"]),
                "p_batt": float(info["p_batt_mw"]),
                "soc": float(info["soc"]),
                "fuel_cost": float(info["fuel_cost"]),
                "carbon_kg": float(info["carbon_kg"]),
            })
        return traj

    def test_pv_trajectory_matches(self, fixed_action_sequence):
        jt = self._rollout_jax(fixed_action_sequence)
        pt = self._rollout_python(fixed_action_sequence)
        for t in range(len(jt)):
            assert jt[t]["p_pv"] == pytest.approx(pt[t]["p_pv"], abs=1e-5), (
                f"step {t}: jax={jt[t]['p_pv']} python={pt[t]['p_pv']}"
            )

    def test_dg_trajectory_matches(self, fixed_action_sequence):
        jt = self._rollout_jax(fixed_action_sequence)
        pt = self._rollout_python(fixed_action_sequence)
        for t in range(len(jt)):
            assert jt[t]["p_dg"] == pytest.approx(pt[t]["p_dg"], abs=1e-5), (
                f"step {t}: jax={jt[t]['p_dg']} python={pt[t]['p_dg']}"
            )

    def test_battery_trajectory_matches(self, fixed_action_sequence):
        jt = self._rollout_jax(fixed_action_sequence)
        pt = self._rollout_python(fixed_action_sequence)
        for t in range(len(jt)):
            assert jt[t]["p_batt"] == pytest.approx(pt[t]["p_batt"], abs=1e-5), (
                f"step {t}: jax p_batt={jt[t]['p_batt']} python p_batt={pt[t]['p_batt']}"
            )
            assert jt[t]["soc"] == pytest.approx(pt[t]["soc"], abs=1e-5), (
                f"step {t}: jax soc={jt[t]['soc']} python soc={pt[t]['soc']}"
            )

    def test_economics_trajectory_matches(self, fixed_action_sequence):
        jt = self._rollout_jax(fixed_action_sequence)
        pt = self._rollout_python(fixed_action_sequence)
        for t in range(len(jt)):
            assert jt[t]["fuel_cost"] == pytest.approx(pt[t]["fuel_cost"], abs=1e-5), (
                f"step {t}: jax fuel={jt[t]['fuel_cost']} python fuel={pt[t]['fuel_cost']}"
            )
            assert jt[t]["carbon_kg"] == pytest.approx(pt[t]["carbon_kg"], abs=1e-5), (
                f"step {t}: jax carbon={jt[t]['carbon_kg']} python carbon={pt[t]['carbon_kg']}"
            )


# ---------------------------------------------------------------------------
# Solar profile injection parity
# ---------------------------------------------------------------------------

class TestProfileInjectionParity:
    """When the same solar profile is injected, both sides produce identical PV output."""

    def test_constant_solar_profile(self):
        T = 288
        cf_const = 0.7
        prof = np.full(T, cf_const, dtype=np.float32)

        from powerzoojax.envs.microgrid import (
            DataCenterMicrogridEnv, make_dcmicrogrid_params,
        )
        from powerzoo.envs.microgrid import DCMicrogridEnv

        jax_env = DataCenterMicrogridEnv()
        jax_params = make_dcmicrogrid_params(max_steps=10, solar_profile=jnp.asarray(prof))
        py_env = DCMicrogridEnv(max_steps=10, solar_profile=prof)

        key = jax.random.PRNGKey(0)
        _, state = jax_env.reset(key, jax_params)
        py_env.reset(seed=0)

        a_phys = np.array([0.5, 0.5, 0.5, 0.0, 0.0], dtype=np.float32)
        for t in range(5):
            key, k = jax.random.split(key)
            _, state, _, _, _, info_j = jax_env.step(k, state, jnp.asarray(a_phys), jax_params)
            _, _, _, _, info_p = py_env.step(a_phys)
            expected_p_pv = cf_const * jax_params.pv_p_max_mw
            assert float(info_j["p_pv_mw"]) == pytest.approx(expected_p_pv, abs=1e-5)
            assert float(info_p["p_pv_mw"]) == pytest.approx(expected_p_pv, abs=1e-5)
            assert float(info_j["p_pv_mw"]) == pytest.approx(
                float(info_p["p_pv_mw"]), abs=1e-5
            )
