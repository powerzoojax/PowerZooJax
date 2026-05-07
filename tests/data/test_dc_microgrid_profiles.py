"""Tests for DC microgrid profile utilities (C4).

Tests are structured in three tiers:
  Synthetic profiles   — always available; no data files required.
  cycle_profile        — pure array operations; always runnable.
  OOD transforms       — work on in-memory DCMicrogridParams; no data files needed.
  DataLoader loading   — skipped if parquet files absent (expected in CI without data).
  Env + profiles       — verifies JIT/vmap work with profile-populated params.

Real vs Synthetic (explicit):
  - cpu_profile: loaded from Google/Azure/Alibaba parquet when available;
    otherwise falls back to the same synthetic diurnal curve.
  - solar_profile: loaded from the GB generation-by-type parquet when
    available; otherwise falls back to the synthetic solar curve.
  - outdoor_temp_profile: synthetic deterministic sine adapter. No weather
    manifest currently available.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import pytest

# ------------------------------------------------------------------ fixtures

@pytest.fixture(scope="module")
def base_params():
    """DCMicrogridParams with synthetic profiles (no real data needed)."""
    from powerzoojax.envs.microgrid import make_dcmicrogrid_params
    return make_dcmicrogrid_params(max_steps=10)


@pytest.fixture(scope="module")
def base_params_with_profiles():
    """DCMicrogridParams with explicit synthetic profile arrays."""
    from powerzoojax.envs.microgrid import make_dcmicrogrid_params_with_profiles
    # Always falls back to synthetic since parquet absent; no real data needed.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return make_dcmicrogrid_params_with_profiles(source="google", max_steps=10)


# ---------------------------------------------------------------------------
# Synthetic profile generators
# ---------------------------------------------------------------------------

class TestSyntheticProfiles:

    def test_cpu_profile_shape_and_dtype(self):
        from powerzoojax.data.dc_microgrid_profiles import make_synthetic_cpu_profile
        arr = make_synthetic_cpu_profile(episode_len=288)
        assert arr.shape == (288,)
        assert arr.dtype == jnp.float32

    def test_cpu_profile_range(self):
        from powerzoojax.data.dc_microgrid_profiles import make_synthetic_cpu_profile
        arr = make_synthetic_cpu_profile(episode_len=288)
        assert float(jnp.min(arr)) >= 0.09
        assert float(jnp.max(arr)) <= 1.01

    def test_solar_profile_noon_is_max(self):
        """Step 144 = noon (12h), solar CF should be maximum = 1.0."""
        from powerzoojax.data.dc_microgrid_profiles import make_synthetic_solar_profile
        arr = make_synthetic_solar_profile(episode_len=288, steps_per_day=288)
        assert float(arr[144]) == pytest.approx(1.0, abs=1e-4)

    def test_solar_profile_night_is_zero(self):
        """Step 0 = midnight, solar CF should be 0."""
        from powerzoojax.data.dc_microgrid_profiles import make_synthetic_solar_profile
        arr = make_synthetic_solar_profile(episode_len=288, steps_per_day=288)
        assert float(arr[0]) == pytest.approx(0.0, abs=1e-6)

    def test_solar_profile_no_negative(self):
        from powerzoojax.data.dc_microgrid_profiles import make_synthetic_solar_profile
        arr = make_synthetic_solar_profile(episode_len=288)
        assert float(jnp.min(arr)) >= 0.0

    def test_outdoor_temp_mean_and_range(self):
        from powerzoojax.data.dc_microgrid_profiles import make_synthetic_outdoor_temp_profile
        arr = make_synthetic_outdoor_temp_profile(episode_len=288)
        assert float(jnp.mean(arr)) == pytest.approx(20.0, abs=0.5)
        assert float(jnp.min(arr)) >= 10.0
        assert float(jnp.max(arr)) <= 30.0

    def test_make_all_synthetic_profiles_keys(self):
        from powerzoojax.data.dc_microgrid_profiles import make_all_synthetic_profiles
        profiles = make_all_synthetic_profiles(episode_len=10)
        assert set(profiles.keys()) == {"cpu_profile", "solar_profile", "outdoor_temp_profile"}
        for k, v in profiles.items():
            assert v.shape == (10,), f"{k} shape mismatch"
            assert v.dtype == jnp.float32


# ---------------------------------------------------------------------------
# cycle_profile
# ---------------------------------------------------------------------------

class TestCycleProfile:

    def test_exact_length_no_tile_needed(self):
        from powerzoojax.data.dc_microgrid_profiles import cycle_profile
        arr = jnp.arange(288, dtype=jnp.float32)
        out = cycle_profile(arr, 288)
        assert out.shape == (288,)
        assert jnp.allclose(out, arr)

    def test_shorter_source_is_tiled(self):
        """Source length 48 < episode_len 288 → must tile."""
        from powerzoojax.data.dc_microgrid_profiles import cycle_profile
        arr = jnp.ones(48, dtype=jnp.float32) * 0.5
        out = cycle_profile(arr, 288)
        assert out.shape == (288,)
        assert jnp.allclose(out, jnp.full(288, 0.5))

    def test_start_offset_shifts_correctly(self):
        """start_step=48 on a 96-step source should give [arr[48:96], arr[0:48]]."""
        from powerzoojax.data.dc_microgrid_profiles import cycle_profile
        arr = jnp.arange(96, dtype=jnp.float32)
        out = cycle_profile(arr, 96, start_step=48)
        expected = jnp.concatenate([arr[48:96], arr[0:48]])
        assert jnp.allclose(out, expected)

    def test_empty_array_passthrough(self):
        from powerzoojax.data.dc_microgrid_profiles import cycle_profile
        arr = jnp.zeros(0, dtype=jnp.float32)
        out = cycle_profile(arr, 288)
        assert out.shape == (0,)

    def test_output_length_always_equals_episode_len(self):
        from powerzoojax.data.dc_microgrid_profiles import cycle_profile
        for T in (10, 100, 288, 300, 1000):
            for ep in (10, 288):
                arr = jnp.arange(T, dtype=jnp.float32)
                out = cycle_profile(arr, ep)
                assert out.shape == (ep,), f"T={T}, ep={ep}: got {out.shape}"

    def test_no_nan_in_output(self):
        from powerzoojax.data.dc_microgrid_profiles import cycle_profile
        arr = jnp.linspace(0, 1, 100)
        out = cycle_profile(arr, 288)
        assert not jnp.any(jnp.isnan(out))


# ---------------------------------------------------------------------------
# load_workload_profiles — synthetic fallback (always available)
# ---------------------------------------------------------------------------

class TestLoadWorkloadProfiles:

    def test_synthetic_fallback_when_no_data(self):
        """load_workload_profiles falls back to synthetic when parquet absent."""
        from powerzoojax.data.dc_microgrid_profiles import load_workload_profiles
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            profiles = load_workload_profiles("google", episode_len=10)
        # Fallback may or may not emit a warning depending on data presence
        assert "cpu_profile" in profiles
        assert "solar_profile" in profiles
        assert "outdoor_temp_profile" in profiles
        for k, v in profiles.items():
            assert v.shape == (10,), f"{k}: {v.shape}"
            assert v.dtype == jnp.float32

    def test_all_sources_return_valid_profiles(self):
        from powerzoojax.data.dc_microgrid_profiles import load_workload_profiles
        for source in ("google", "azure", "alibaba"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                profiles = load_workload_profiles(source, episode_len=5)
            for k, v in profiles.items():
                assert not jnp.any(jnp.isnan(v)), f"{source}/{k} has NaN"

    def test_invalid_source_raises(self):
        from powerzoojax.data.dc_microgrid_profiles import load_workload_profiles
        with pytest.raises(ValueError, match="Unknown source"):
            load_workload_profiles("nonexistent", episode_len=10)

    def test_profiles_no_nan(self):
        from powerzoojax.data.dc_microgrid_profiles import load_workload_profiles
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            profiles = load_workload_profiles("google", episode_len=288)
        for k, v in profiles.items():
            assert not jnp.any(jnp.isnan(v)), f"{k} has NaN"

    def test_real_solar_profile_shape_and_range(self):
        from powerzoojax.data.dc_microgrid_profiles import make_real_solar_profile
        profile = make_real_solar_profile(episode_len=288, require_real_data=True)
        assert profile.shape == (288,)
        assert profile.dtype == jnp.float32
        assert float(jnp.min(profile)) >= 0.0
        assert float(jnp.max(profile)) <= 1.0
        assert float(jnp.max(profile)) > 0.0


# ---------------------------------------------------------------------------
# OOD transforms
# ---------------------------------------------------------------------------

class TestOODTransforms:

    def _make_params_with_profiles(self, episode_len=10):
        from powerzoojax.envs.microgrid import (
            make_dcmicrogrid_params_with_profiles,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return make_dcmicrogrid_params_with_profiles(
                source="google", max_steps=episode_len
            )

    def test_renewable_drought_reduces_solar(self):
        from powerzoojax.data.dc_microgrid_profiles import apply_ood_transform
        params = self._make_params_with_profiles()
        original_solar = params.solar_profile
        drought_params = apply_ood_transform(params, "renewable_drought", drought_factor=0.2)
        if original_solar is not None and drought_params.solar_profile is not None:
            expected = original_solar * 0.2
            assert jnp.allclose(drought_params.solar_profile, expected, atol=1e-6)
        else:
            # If profiles were None, drought still creates a reduced solar profile
            assert drought_params.solar_profile is not None

    def test_cooling_stress_raises_outdoor_temp(self):
        from powerzoojax.data.dc_microgrid_profiles import apply_ood_transform
        params = self._make_params_with_profiles()
        orig_temp = params.outdoor_temp_profile
        stressed = apply_ood_transform(params, "cooling_stress", temp_delta=5.0)
        if orig_temp is not None and stressed.outdoor_temp_profile is not None:
            diff = stressed.outdoor_temp_profile - orig_temp
            assert jnp.allclose(diff, jnp.full_like(diff, 5.0), atol=1e-6)
        else:
            assert stressed.outdoor_temp_profile is not None

    def test_dg_derating_reduces_max_power(self):
        from powerzoojax.data.dc_microgrid_profiles import apply_ood_transform
        params = self._make_params_with_profiles()
        orig_max = params.dg.p_dg_max_mw
        derated = apply_ood_transform(params, "dg_derating", dg_derating_factor=0.6)
        assert derated.dg.p_dg_max_mw == pytest.approx(orig_max * 0.6, rel=1e-5)

    def test_sla_tighten_reduces_deadline_slack(self):
        from powerzoojax.data.dc_microgrid_profiles import apply_ood_transform
        params = self._make_params_with_profiles()
        tightened = apply_ood_transform(params, "sla_tighten", sla_slack=1.2)
        assert tightened.dc.train_deadline_slack == pytest.approx(1.2, rel=1e-5)
        assert tightened.dc.ft_deadline_slack == pytest.approx(1.2, rel=1e-5)

    def test_invalid_scenario_raises(self):
        from powerzoojax.data.dc_microgrid_profiles import apply_ood_transform
        params = self._make_params_with_profiles()
        with pytest.raises(ValueError, match="Unknown OOD"):
            apply_ood_transform(params, "nonexistent_scenario")

    def test_ood_preserves_other_params(self):
        """OOD transforms don't silently change battery/pv params."""
        from powerzoojax.data.dc_microgrid_profiles import apply_ood_transform
        params = self._make_params_with_profiles()
        drought = apply_ood_transform(params, "renewable_drought")
        assert drought.battery_capacity_mwh == pytest.approx(params.battery_capacity_mwh)
        assert drought.pv_p_max_mw == pytest.approx(params.pv_p_max_mw)
        assert drought.w_cost == pytest.approx(params.w_cost)


# ---------------------------------------------------------------------------
# Env + profiles: JIT/vmap compatibility
# ---------------------------------------------------------------------------

class TestEnvWithProfiles:
    """Verify the env works correctly when params carry non-None profiles."""

    @pytest.fixture
    def env(self):
        from powerzoojax.envs.microgrid import DataCenterMicrogridEnv
        return DataCenterMicrogridEnv()

    @pytest.fixture
    def params_with_profiles(self):
        from powerzoojax.envs.microgrid import make_dcmicrogrid_params_with_profiles
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return make_dcmicrogrid_params_with_profiles(source="google", max_steps=5)

    def test_reset_and_step_with_profiles(self, env, params_with_profiles):
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params_with_profiles)
        assert obs.shape == (24,)
        action = jnp.zeros(5)
        obs2, state2, reward, costs, done, info = env.step(
            key, state, action, params_with_profiles
        )
        assert obs2.shape == (24,)
        assert jnp.isfinite(reward)
        assert costs.shape == (3,)

    def test_jit_with_profiles(self, env, params_with_profiles):
        """JIT compilation works with profile-populated params."""
        key = jax.random.PRNGKey(1)
        obs, state = env.reset(key, params_with_profiles)
        action = jnp.zeros(5)
        # The env methods are already @jit-decorated; just calling them verifies
        _, state2, _, _, _, info = env.step(key, state, action, params_with_profiles)
        assert "p_pv_mw" in info

    def test_vmap_with_profiles(self, env, params_with_profiles):
        """vmap over parallel envs works with profile arrays in params."""
        keys = jax.random.split(jax.random.PRNGKey(2), 4)
        vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
        obs_b, state_b = vmap_reset(keys, params_with_profiles)
        assert obs_b.shape == (4, 24)

    def test_solar_profile_used_in_pv(self, env, params_with_profiles):
        """p_pv_mw should match the solar_profile value at current step."""
        key = jax.random.PRNGKey(3)
        _, state = env.reset(key, params_with_profiles)
        action = jnp.zeros(5)
        t = int(state.dc.time_step)
        _, _, _, _, _, info = env.step(key, state, action, params_with_profiles)
        T_sol = params_with_profiles.solar_profile.shape[0]
        expected_cf = float(params_with_profiles.solar_profile[t % T_sol])
        expected_pv = expected_cf * params_with_profiles.pv_p_max_mw
        assert float(info["p_pv_mw"]) == pytest.approx(expected_pv, abs=1e-5)

    def test_power_balance_with_profiles(self, env, params_with_profiles):
        """residual = p_pv + p_dg + p_batt - p_load remains self-consistent."""
        key = jax.random.PRNGKey(4)
        _, state = env.reset(key, params_with_profiles)
        action = jnp.array([0.5, 0.5, 0.5, 0.2, 0.3], dtype=jnp.float32)
        _, _, _, _, _, info = env.step(key, state, action, params_with_profiles)
        residual_check = (
            info["p_pv_mw"] + info["p_dg_mw"] + info["p_batt_mw"] - info["p_load_mw"]
        )
        assert jnp.allclose(info["residual"], residual_check, atol=1e-5)


# ===========================================================================
# Review finding A: strict vs non-strict error handling
# ===========================================================================

class TestFindingA_StrictErrorHandling:
    """load_workload_profiles must NOT silently degrade non-file errors."""

    def test_file_not_found_with_strict_warns_and_falls_back(self):
        """FileNotFoundError in strict mode → UserWarning + synthetic fallback."""
        from powerzoojax.data.dc_microgrid_profiles import load_workload_profiles
        import unittest.mock as mock

        def _raise_fnf(*a, **kw):
            raise FileNotFoundError("parquet/google_dc_2019_300s.parquet not found")

        with mock.patch(
            "powerzoojax.data.data_loader.DataLoader.__init__",
            side_effect=_raise_fnf,
        ):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                profiles = load_workload_profiles("google", episode_len=5, strict=True)
            # Should warn and fall back to synthetic
            assert any("Falling back" in str(w.message) for w in caught)
            for k, v in profiles.items():
                assert v.shape == (5,)

    def test_import_error_falls_back_when_not_required(self):
        """ImportError (no parquet engine) falls back when require_real_data=False.

        This covers the case where a parquet file EXISTS but pyarrow/fastparquet
        are not installed.  With require_real_data=False (generic path), an
        ImportError is an infrastructure issue → acceptable fallback.
        """
        from powerzoojax.data.dc_microgrid_profiles import load_workload_profiles
        import unittest.mock as mock

        def _raise_import_err(*a, **kw):
            raise ImportError(
                "Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'"
            )

        with mock.patch(
            "powerzoojax.data.data_loader.DataLoader.load_jax_profiles",
            side_effect=_raise_import_err,
        ):
            for strict_val in (True, False):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    profiles = load_workload_profiles(
                        "google", episode_len=5, strict=strict_val, require_real_data=False
                    )
                # Must NOT raise; must warn + fallback
                assert any("Falling back" in str(w.message) for w in caught), (
                    f"Expected fallback warning with strict={strict_val}"
                )
                for k, v in profiles.items():
                    assert v.shape == (5,), f"{k}: {v.shape}"

    def test_non_file_error_strict_raises(self):
        """Non-FileNotFoundError in strict=True mode must NOT be swallowed."""
        from powerzoojax.data.dc_microgrid_profiles import load_workload_profiles
        import unittest.mock as mock

        class _SchemaError(ValueError):
            pass

        def _raise_schema(*a, **kw):
            raise _SchemaError("signal 'datacenter.cpu_util' not found in manifest")

        with mock.patch(
            "powerzoojax.data.data_loader.DataLoader.load_jax_profiles",
            side_effect=_raise_schema,
        ):
            with pytest.raises(_SchemaError):
                load_workload_profiles("google", episode_len=5, strict=True)

    def test_non_file_error_non_strict_warns_and_falls_back(self):
        """Non-FileNotFoundError with strict=False → UserWarning + synthetic fallback."""
        from powerzoojax.data.dc_microgrid_profiles import load_workload_profiles
        import unittest.mock as mock

        def _raise_generic(*a, **kw):
            raise RuntimeError("unexpected schema mismatch")

        with mock.patch(
            "powerzoojax.data.data_loader.DataLoader.load_jax_profiles",
            side_effect=_raise_generic,
        ):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                profiles = load_workload_profiles(
                    "google", episode_len=5, strict=False
                )
            assert len(caught) >= 1
            assert any("Falling back" in str(w.message) for w in caught)
            for v in profiles.values():
                assert v.shape == (5,)

    def test_default_strict_is_true(self):
        """By default strict=True — non-file/non-import errors are NOT silently degraded."""
        import inspect
        from powerzoojax.data.dc_microgrid_profiles import load_workload_profiles
        sig = inspect.signature(load_workload_profiles)
        assert sig.parameters["strict"].default is True

    def test_factory_default_strict_is_false(self):
        """make_dcmicrogrid_params_with_profiles defaults to strict=False for dev convenience."""
        import inspect
        from powerzoojax.envs.microgrid import (
            make_dcmicrogrid_params_with_profiles,
        )
        sig = inspect.signature(make_dcmicrogrid_params_with_profiles)
        assert sig.parameters["strict"].default is False

    def test_factory_succeeds_without_parquet_engine(self):
        """Factory with strict=False succeeds even when no parquet engine is available."""
        from powerzoojax.envs.microgrid import (
            make_dcmicrogrid_params_with_profiles,
        )
        import unittest.mock as mock

        def _raise_import_err(*a, **kw):
            raise ImportError("Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'")

        with mock.patch(
            "powerzoojax.data.data_loader.DataLoader.load_jax_profiles",
            side_effect=_raise_import_err,
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                params = make_dcmicrogrid_params_with_profiles(source="google", max_steps=5)
        # Should succeed with synthetic profiles
        assert params.cpu_profile is not None
        assert params.cpu_profile.shape == (5,)


# ===========================================================================
# Review finding B: workload OOD must preserve episode length
# ===========================================================================

class TestFindingB_OODEpisodeLength:
    """workload_swap / workload_shock must match the base params episode length."""

    def _make_params(self, episode_len=10):
        from powerzoojax.envs.microgrid import (
            make_dcmicrogrid_params_with_profiles,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return make_dcmicrogrid_params_with_profiles(
                source="google", max_steps=episode_len
            )

    def _ood_workload_fallback(self, base_params, scenario):
        """Apply workload OOD via monkeypatching load to always return synthetic."""
        from powerzoojax.data.dc_microgrid_profiles import (
            apply_ood_transform,
            make_synthetic_cpu_profile,
        )
        import unittest.mock as mock

        ep = int(base_params.dc.max_steps)

        def _fake_load(source, episode_len, **kw):
            # Return a dict with synthetic profile of the REQUESTED episode_len
            return {
                "cpu_profile": make_synthetic_cpu_profile(episode_len),
                "solar_profile": jnp.zeros(episode_len, dtype=jnp.float32),
                "outdoor_temp_profile": jnp.zeros(episode_len, dtype=jnp.float32),
            }

        with mock.patch(
            "powerzoojax.data.dc_microgrid_profiles.load_workload_profiles",
            side_effect=_fake_load,
        ):
            return apply_ood_transform(base_params, scenario)

    @pytest.mark.parametrize("scenario", ["workload_swap", "workload_shock"])
    def test_workload_ood_preserves_episode_len(self, scenario):
        """After workload OOD, cpu_profile.shape[0] must equal base max_steps."""
        base = self._make_params(episode_len=10)
        assert base.dc.max_steps == 10

        new_params = self._ood_workload_fallback(base, scenario)

        assert new_params.dc.max_steps == 10, "max_steps must not change"
        assert new_params.cpu_profile is not None
        assert new_params.cpu_profile.shape[0] == 10, (
            f"cpu_profile shape {new_params.cpu_profile.shape[0]} != 10 "
            "(was hardcoded DC_EPISODE_LEN=288 before fix)"
        )

    @pytest.mark.parametrize("scenario", ["workload_swap", "workload_shock"])
    def test_workload_ood_does_not_change_288_base(self, scenario):
        """Sanity: 288-step base also stays at 288 after OOD."""
        base = self._make_params(episode_len=288)
        new_params = self._ood_workload_fallback(base, scenario)
        assert new_params.cpu_profile.shape[0] == 288

    def test_episode_len_derived_from_cpu_profile_shape(self):
        """episode_len in OOD is derived from cpu_profile.shape[0] when present."""
        from powerzoojax.data.dc_microgrid_profiles import apply_ood_transform
        import unittest.mock as mock

        base = self._make_params(episode_len=20)
        # Manually set cpu_profile to length 15 (different from max_steps=20)
        # to verify the profile shape takes priority
        # After bundle refactor DCMicrogridParams takes (dc, resources, ...);
        # the simplest way to override cpu_profile is via .replace().
        base_with_short = base.replace(
            cpu_profile=jnp.zeros(15, dtype=jnp.float32),
        )

        requested_ep_lens = []

        def _fake_load(source, episode_len, **kw):
            requested_ep_lens.append(episode_len)
            return {
                "cpu_profile": jnp.zeros(episode_len, dtype=jnp.float32),
                "solar_profile": jnp.zeros(episode_len, dtype=jnp.float32),
                "outdoor_temp_profile": jnp.zeros(episode_len, dtype=jnp.float32),
            }

        with mock.patch(
            "powerzoojax.data.dc_microgrid_profiles.load_workload_profiles",
            side_effect=_fake_load,
        ):
            apply_ood_transform(base_with_short, "workload_swap")

        assert requested_ep_lens == [15], (
            f"Expected episode_len=15 (from cpu_profile.shape[0]), got {requested_ep_lens}"
        )


# ===========================================================================
# Review finding C: PV phase alignment (regression guard)
# ===========================================================================

class TestFindingC_PVPhaseRegression:
    """Regression tests confirming the one-step PV phase bug does not recur.

    Finding C was already fixed (state.dc.time_step used, not new_dc.time_step).
    These tests are explicit guards to prevent regression.
    """

    def test_step_uses_current_time_index_for_pv(self):
        """p_pv_mw must correspond to state.dc.time_step, not state.dc.time_step+1."""
        from powerzoojax.envs.microgrid import (
            DataCenterMicrogridEnv,
            make_dcmicrogrid_params_with_profiles,
        )
        from powerzoojax.data.dc_microgrid_profiles import make_synthetic_solar_profile

        # Build params with a non-trivial solar profile so we can detect off-by-one
        ep = 20
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            params = make_dcmicrogrid_params_with_profiles(source="google", max_steps=ep)

        env = DataCenterMicrogridEnv()
        key = jax.random.PRNGKey(0)
        _, state = env.reset(key, params)
        action = jnp.zeros(5)

        for _ in range(5):
            key, k = jax.random.split(key)
            t_before = int(state.dc.time_step)
            _, state, _, _, _, info = env.step(k, state, action, params)

            T = params.solar_profile.shape[0]
            expected_pv = float(params.solar_profile[t_before % T]) * params.pv_p_max_mw
            wrong_pv = float(params.solar_profile[(t_before + 1) % T]) * params.pv_p_max_mw

            # The actual p_pv_mw should match the CURRENT (not next) step solar
            assert float(info["p_pv_mw"]) == pytest.approx(expected_pv, abs=1e-5), (
                f"t={t_before}: p_pv={info['p_pv_mw']:.6f} != expected={expected_pv:.6f} "
                f"(wrong_pv={wrong_pv:.6f}). Phase bug may have recurred."
            )


# ===========================================================================
# OOD workload strict semantics: workload_swap / workload_shock must NOT
# silently fall back to synthetic when real data is unavailable.
# ===========================================================================

class TestOODWorkloadStrictSemantics:
    """workload_swap / workload_shock must raise if real data is unavailable.

    apply_ood_transform() calls load_workload_profiles(..., require_real_data=True).
    This ensures that a missing parquet file or missing parquet engine causes an
    immediate error rather than silently returning a synthetic profile that would
    corrupt the OOD experiment with fake data.
    """

    def _make_base_params(self, episode_len=5):
        from powerzoojax.envs.microgrid import (
            make_dcmicrogrid_params_with_profiles,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return make_dcmicrogrid_params_with_profiles(
                source="google", max_steps=episode_len
            )

    @pytest.mark.parametrize("scenario", ["workload_swap", "workload_shock"])
    def test_ood_raises_on_file_not_found(self, scenario):
        """workload_swap / workload_shock raise FileNotFoundError if data absent."""
        from powerzoojax.data.dc_microgrid_profiles import apply_ood_transform
        import unittest.mock as mock

        base = self._make_base_params()

        def _raise_fnf(source, episode_len, **kw):
            raise FileNotFoundError(f"{source} parquet not found")

        with mock.patch(
            "powerzoojax.data.dc_microgrid_profiles.load_workload_profiles",
            side_effect=_raise_fnf,
        ):
            with pytest.raises(FileNotFoundError):
                apply_ood_transform(base, scenario)

    @pytest.mark.parametrize("scenario", ["workload_swap", "workload_shock"])
    def test_ood_raises_on_import_error(self, scenario):
        """workload_swap / workload_shock raise ImportError if no parquet engine."""
        from powerzoojax.data.dc_microgrid_profiles import apply_ood_transform
        import unittest.mock as mock

        base = self._make_base_params()

        def _raise_import(source, episode_len, **kw):
            raise ImportError("Unable to find a usable engine")

        with mock.patch(
            "powerzoojax.data.dc_microgrid_profiles.load_workload_profiles",
            side_effect=_raise_import,
        ):
            with pytest.raises(ImportError):
                apply_ood_transform(base, scenario)

    @pytest.mark.parametrize("scenario", ["workload_swap", "workload_shock"])
    def test_ood_does_not_silently_produce_synthetic(self, scenario):
        """When real data unavailable, OOD must NOT return a params with synthetic profile.

        Concretely: if load_workload_profiles raises, apply_ood_transform must
        propagate the error — it must not catch it and return the base params.
        """
        from powerzoojax.data.dc_microgrid_profiles import apply_ood_transform
        import unittest.mock as mock

        base = self._make_base_params()

        def _raise_fnf(source, episode_len, **kw):
            raise FileNotFoundError("data absent")

        with mock.patch(
            "powerzoojax.data.dc_microgrid_profiles.load_workload_profiles",
            side_effect=_raise_fnf,
        ):
            try:
                result = apply_ood_transform(base, scenario)
                pytest.fail(
                    f"apply_ood_transform({scenario!r}) returned silently with synthetic "
                    f"cpu_profile instead of raising. "
                    f"cpu_profile shape: {result.cpu_profile.shape if result.cpu_profile is not None else None}"
                )
            except FileNotFoundError:
                pass  # expected: error propagated correctly

    def test_generic_factory_still_falls_back(self):
        """Generic factory (require_real_data=False) still succeeds without real data."""
        from powerzoojax.envs.microgrid import (
            make_dcmicrogrid_params_with_profiles,
        )
        import unittest.mock as mock

        def _raise_fnf(*a, **kw):
            raise FileNotFoundError("data absent")

        with mock.patch(
            "powerzoojax.data.data_loader.DataLoader.load_jax_profiles",
            side_effect=_raise_fnf,
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                params = make_dcmicrogrid_params_with_profiles(
                    source="google", max_steps=5
                )
        assert params.cpu_profile is not None
        assert params.cpu_profile.shape == (5,)
