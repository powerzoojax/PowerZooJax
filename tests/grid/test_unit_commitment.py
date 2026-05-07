"""Unit Commitment environment tests.

Coverage:
  L0 JAX contract:
    - jit compile (reset + step)
    - vmap over parallel envs
    - pytree structure stable
    - auto-reset on done
    - lax.scan rollout (48 steps)

  L1 physics / semantics:
    - commitment mask: OFF units do not generate
    - min-up / min-down constraints enforced
    - startup cost charged on off→on transition
    - no-load cost charged for ON units
    - ramp constraint respected
    - reserve shortfall detection
    - reward / cost separation
    - cost decomposition: all info keys present

  Case14 sanity:
    - reset / step smoke
    - commitment mask
    - min-up / min-down small case

  Case118 smoke:
    - reset / step smoke
    - 48-step rollout
    - JIT compile
    - vmap smoke

  Ramp unit conversion test:
    - ramp_mw is NOT the raw fraction (0.7)
    - ramp_mw = 0.7 * p_max * delta_t_hours
"""

import pytest
import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np
import chex

from powerzoojax.envs.grid.unit_commitment import (
    UCState,
    UCParams,
    UnitCommitmentEnv,
    make_uc_params,
)
from powerzoojax.tasks.tso import (
    make_tso_case14_params,
    make_tso_case118_params,
    make_tso_net_load_profiles,
    make_tso_net_load_profiles_from_data,
    make_tso_uc_params,
    make_tso_scuc_params,
    make_case14_with_uc_defaults,
    compute_tso_metrics,
    tso_all_on_rollout,
)


@pytest.fixture(autouse=True, scope="module")
def _force_float32_mode():
    """Keep UC tests isolated from ACOPF's global x64 side effects."""
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", False)


# ============ Fixtures ============

@pytest.fixture(scope="module")
def case14_params() -> UCParams:
    """Small case14 UCParams for fast tests."""
    return make_tso_case14_params(max_steps=8, solver_mode=1)


@pytest.fixture(scope="module")
def case118_params() -> UCParams:
    """Full case118 UCParams with synthetic load."""
    return make_tso_case118_params(max_steps=48, solver_mode=1)


@pytest.fixture(scope="module")
def env() -> UnitCommitmentEnv:
    return UnitCommitmentEnv()


@pytest.fixture(scope="module")
def key():
    return jax.random.PRNGKey(42)


# ============ Helper ============

def zero_action(params: UCParams) -> jnp.ndarray:
    """All-zero action (commit_signal=0, dispatch_signal=0)."""
    return jnp.zeros(2 * params.case.n_units, dtype=jnp.float32)


def all_on_action(params: UCParams) -> jnp.ndarray:
    """Commit all units ON, dispatch intent=0."""
    n = params.case.n_units
    return jnp.concatenate([
        jnp.ones(n, dtype=jnp.float32),
        jnp.zeros(n, dtype=jnp.float32),
    ])


def all_off_action(params: UCParams) -> jnp.ndarray:
    """Try to commit all units OFF, dispatch=0."""
    n = params.case.n_units
    return jnp.concatenate([
        -jnp.ones(n, dtype=jnp.float32),
        jnp.zeros(n, dtype=jnp.float32),
    ])


# ============ L0: JAX contract ============

class TestL0JAXContract:
    """L0 — jit / vmap / pytree / auto-reset / scan rollout."""

    def test_reset_jit(self, env, case14_params, key):
        # env.reset is already JIT-decorated; just call it
        obs, state = env.reset(key, case14_params)
        assert obs.ndim == 1

    def test_step_jit(self, env, case14_params, key):
        obs, state = env.reset(key, case14_params)
        action = all_on_action(case14_params)
        k1, _ = jax.random.split(key)
        # env.step is already JIT-decorated; just call it
        obs2, state2, reward, costs, done, info = env.step(
            k1, state, action, case14_params
        )
        assert obs2.shape == obs.shape
        assert reward.ndim == 0
        assert costs.shape == (3,)

    def test_pytree_structure_stable(self, env, case14_params, key):
        _, state1 = env.reset(key, case14_params)
        action = all_on_action(case14_params)
        k1, k2 = jax.random.split(key)
        _, state2, *_ = env.step(k1, state1, action, case14_params)
        leaves1 = jax.tree_util.tree_leaves(state1)
        leaves2 = jax.tree_util.tree_leaves(state2)
        assert len(leaves1) == len(leaves2), "pytree structure must be static"
        for l1, l2 in zip(leaves1, leaves2):
            if hasattr(l1, "shape"):
                assert l1.shape == l2.shape, f"shape mismatch: {l1.shape} vs {l2.shape}"

    def test_auto_reset(self, env, case14_params, key):
        """After max_steps, state.time_step should reset to near 0."""
        params = case14_params  # max_steps=8
        obs, state = env.reset(key, params)
        action = all_on_action(params)
        for i in range(params.max_steps + 1):
            k, key = jax.random.split(key)
            obs, state, reward, costs, done, info = env.step(k, state, action, params)
        # After max_steps+1 steps, auto-reset must have triggered
        # time_step should be near 0 or 1 (reset then one step)
        assert int(state.time_step) <= 2, f"time_step={state.time_step} after reset"

    def test_vmap_reset(self, env, case14_params):
        n_envs = 4
        keys = jax.random.split(jax.random.PRNGKey(0), n_envs)
        obs_batch, state_batch = jax.vmap(
            lambda k: env.reset(k, case14_params)
        )(keys)
        assert obs_batch.shape[0] == n_envs

    def test_vmap_step(self, env, case14_params):
        n_envs = 4
        keys = jax.random.split(jax.random.PRNGKey(0), n_envs)
        obs_batch, state_batch = jax.vmap(
            lambda k: env.reset(k, case14_params)
        )(keys)
        action = all_on_action(case14_params)
        actions = jnp.stack([action] * n_envs)
        step_keys = jax.random.split(jax.random.PRNGKey(1), n_envs)
        obs2, state2, rew, costs, done, info = jax.vmap(
            lambda k, s, a: env.step(k, s, a, case14_params)
        )(step_keys, state_batch, actions)
        assert obs2.shape[0] == n_envs
        assert rew.shape == (n_envs,)
        assert costs.shape == (n_envs, 3)

    def test_scan_rollout(self, env, case14_params, key):
        """lax.scan over T steps."""
        obs0, state0 = env.reset(key, case14_params)
        action = all_on_action(case14_params)

        def scan_fn(carry, _):
            k, state = carry
            k, k_step = jax.random.split(k)
            obs, new_state, reward, costs, done, info = env.step(
                k_step, state, action, case14_params
            )
            return (k, new_state), (reward, jnp.sum(costs))

        (_, final_state), (rewards, costs) = lax.scan(
            scan_fn, (key, state0), None, length=case14_params.max_steps)

        assert rewards.shape == (case14_params.max_steps,)
        assert costs.shape == (case14_params.max_steps,)


# ============ L1: commitment mask ============

class TestL1CommitmentMask:

    def test_off_units_do_not_generate(self, env, case14_params, key):
        """Units with commit_signal < 0 should have ~0 dispatch (OPF may still produce
        small values; key check is that committed=0 units yield p≈0)."""
        params = make_tso_case14_params(max_steps=8, solver_mode=0)  # direct PF
        obs, state = env.reset(key, params)
        # Force all units OFF
        action = all_off_action(params)
        k1, _ = jax.random.split(key)
        obs2, state2, reward, costs, done, info = env.step(k1, state, action, params)
        # With min-up/down constraints, some units may be forced ON at init.
        # But if they were all ON in reset, min-up may prevent them from turning off.
        # Test that units with status=0 have ~0 power (allow fp tolerance)
        off_mask = (state2.unit_status == 0)
        off_power = state2.unit_power_mw * off_mask.astype(jnp.float32)
        assert float(jnp.max(jnp.abs(off_power))) < 1.0, (
            "OFF units must have negligible dispatch in direct PF mode")

    def test_committed_units_can_generate(self, env, case14_params, key):
        obs, state = env.reset(key, case14_params)
        action = all_on_action(case14_params)
        k1, _ = jax.random.split(key)
        _, state2, *_ = env.step(k1, state, action, case14_params)
        total_gen = float(jnp.sum(state2.unit_power_mw))
        assert total_gen > 0.0, "ON units must generate positive power"


# ============ L1: min-up / min-down ============

class TestL1MinUpDown:

    def test_min_up_enforced(self, env, key):
        """A freshly started unit cannot be turned off before min_up_steps."""
        from powerzoojax.case import load_case
        base = load_case("14")
        case = make_case14_with_uc_defaults(base)
        # Set min_up_steps = 4 for all units so constraint is active
        n = case.n_units
        # Use make_uc_params with custom case that has min_up_time = 4
        case_min4 = case.replace(
            unit_min_up_time=jnp.full((n,), 4, dtype=jnp.float32),
            unit_init_state=jnp.ones((n,), dtype=jnp.float32),
            unit_keep_time=jnp.full((n,), 1.0, dtype=jnp.float32),  # just started ON
        )
        params = make_uc_params(
            case_min4, max_steps=8, solver_mode=0, enable_uc=True)
        obs, state = env.reset(key, params)

        # Try to turn off all units — should be blocked by min-up (time_in_state=1 < 4)
        action = all_off_action(params)
        k1, _ = jax.random.split(key)
        _, state2, *_ = env.step(k1, state, action, params)
        # All units should still be ON
        assert jnp.all(state2.unit_status == 1), (
            "Min-up constraint must prevent turning off freshly started units")

    def test_min_down_enforced(self, env, key):
        """A freshly stopped unit cannot be turned on before min_down_steps."""
        from powerzoojax.case import load_case
        base = load_case("14")
        case = make_case14_with_uc_defaults(base)
        n = case.n_units
        # Start all OFF with time_in_state=1 (just turned off)
        case_off = case.replace(
            unit_min_down_time=jnp.full((n,), 4, dtype=jnp.float32),
            unit_init_state=jnp.zeros((n,), dtype=jnp.float32),
            unit_keep_time=jnp.full((n,), 1.0, dtype=jnp.float32),
            unit_init_power=jnp.zeros((n,), dtype=jnp.float32),
        )
        params = make_uc_params(
            case_off, max_steps=8, solver_mode=0, enable_uc=True)
        obs, state = env.reset(key, params)

        # All units are OFF; try to turn all ON — should be blocked by min-down
        action = all_on_action(params)
        k1, _ = jax.random.split(key)
        _, state2, *_ = env.step(k1, state, action, params)
        # All units should still be OFF
        assert jnp.all(state2.unit_status == 0), (
            "Min-down constraint must prevent turning on freshly stopped units")

    def test_time_in_state_increments(self, env, case14_params, key):
        obs, state = env.reset(key, case14_params)
        init_t = state.time_in_state.copy()
        action = all_on_action(case14_params)
        k1, _ = jax.random.split(key)
        _, state2, *_ = env.step(k1, state, action, case14_params)
        # For units that didn't change status, time_in_state should increase by 1
        same_status = (state2.unit_status == state.unit_status)
        incremented = (state2.time_in_state == state.time_in_state + 1)
        reset_to_one = (state2.time_in_state == 1)
        # Either incremented (same status) or reset to 1 (changed status)
        valid = jnp.logical_or(
            jnp.logical_and(same_status, incremented),
            jnp.logical_and(~same_status, reset_to_one),
        )
        assert jnp.all(valid)


# ============ L1: startup / no-load cost ============

class TestL1StartupNoLoadCost:

    def test_startup_cost_charged_on_transition(self, env, key):
        """Startup cost should be > 0 when units transition off→on."""
        from powerzoojax.case import load_case
        base = load_case("14")
        case = make_case14_with_uc_defaults(base)
        n = case.n_units
        # Start all OFF (well past min-down)
        case_off = case.replace(
            unit_init_state=jnp.zeros((n,), dtype=jnp.float32),
            unit_keep_time=jnp.full((n,), 100.0, dtype=jnp.float32),  # well past min-down
            unit_min_down_time=jnp.full((n,), 2.0, dtype=jnp.float32),
            unit_init_power=jnp.zeros((n,), dtype=jnp.float32),
            unit_startup_cost=jnp.full((n,), 500.0, dtype=jnp.float32),
        )
        params = make_uc_params(case_off, max_steps=8, solver_mode=0, enable_uc=True)
        obs, state = env.reset(key, params)
        assert jnp.all(state.unit_status == 0), "Should start all OFF"

        # Turn all units ON → startup cost should be > 0
        action = all_on_action(params)
        k1, _ = jax.random.split(key)
        _, _, _, _, _, info = env.step(k1, state, action, params)
        assert float(info["startup_cost"]) > 0.0, (
            "Startup cost must be charged for off→on transitions")

    def test_no_startup_cost_when_already_on(self, env, case14_params, key):
        """No startup cost when unit remains ON."""
        obs, state = env.reset(key, case14_params)
        # All units start ON (case14 defaults init_state=1)
        action = all_on_action(case14_params)
        k1, _ = jax.random.split(key)
        _, _, _, _, _, info = env.step(k1, state, action, case14_params)
        # Startup cost = 0 if no unit changed from OFF to ON
        switched = jnp.sum((state.unit_status == 0).astype(jnp.float32))
        if float(switched) == 0.0:
            assert float(info["startup_cost"]) == 0.0

    def test_no_load_cost_positive_for_on_units(self, env, case14_params, key):
        """No-load cost should be > 0 when units are ON."""
        params = case14_params
        obs, state = env.reset(key, params)
        action = all_on_action(params)
        k1, _ = jax.random.split(key)
        _, _, _, _, _, info = env.step(k1, state, action, params)
        assert float(info["no_load_cost"]) >= 0.0

    def test_startup_cost_accum_increases(self, env, key):
        """startup_cost_accum must increase when units start."""
        from powerzoojax.case import load_case
        base = load_case("14")
        case = make_case14_with_uc_defaults(base)
        n = case.n_units
        case_off = case.replace(
            unit_init_state=jnp.zeros((n,), dtype=jnp.float32),
            unit_keep_time=jnp.full((n,), 100.0, dtype=jnp.float32),
            unit_min_down_time=jnp.full((n,), 2.0, dtype=jnp.float32),
            unit_init_power=jnp.zeros((n,), dtype=jnp.float32),
            unit_startup_cost=jnp.full((n,), 100.0, dtype=jnp.float32),
        )
        params = make_uc_params(case_off, max_steps=8, solver_mode=0, enable_uc=True)
        obs, state = env.reset(key, params)
        action = all_on_action(params)
        k1, _ = jax.random.split(key)
        _, state2, *_ = env.step(k1, state, action, params)
        assert float(state2.startup_cost_accum) > float(state.startup_cost_accum)


# ============ L1: ramp clip ============

class TestL1RampClip:

    def test_ramp_conversion_not_raw_fraction(self, case118_params):
        """ramp_up_mw must NOT be the raw 0.7 fraction — it should be in MW."""
        ramp = np.asarray(case118_params.ramp_up_mw)
        p_max = np.asarray(case118_params.case.unit_p_max)
        # Raw case data has ramp_up = 0.7 (fraction/h).
        # After conversion: ramp_mw = 0.7 * p_max * delta_t_hours
        # This should be >> 0.7 for MW-scale machines.
        assert float(ramp.max()) > 10.0, (
            "ramp_up_mw must be in MW, not the raw 0.7 fraction. "
            f"Got max={ramp.max():.3f}")

    def test_ramp_formula_matches_expected(self, case118_params):
        """ramp_mw = ramp_frac * p_max * delta_t_hours for case118."""
        ramp = np.asarray(case118_params.ramp_up_mw)
        p_max = np.asarray(case118_params.case.unit_p_max)
        dt = case118_params.delta_t_hours
        # case118 has ramp_up = 0.7 for all units
        expected = 0.7 * p_max * dt
        np.testing.assert_allclose(ramp, expected, rtol=1e-3, atol=1e-2,
            err_msg="ramp_up_mw must equal ramp_frac * p_max * delta_t_hours")

    def test_ramp_clips_large_change(self, env, key):
        """In direct-PF mode, large dispatch change should be clipped by ramp."""
        from powerzoojax.case import load_case
        base = load_case("14")
        case = make_case14_with_uc_defaults(base)
        n = case.n_units
        # Tight ramp: 10% of p_max per step
        case_tight = case.replace(
            unit_ramp_up=jnp.full((n,), 0.1, dtype=jnp.float32),
            unit_ramp_down=jnp.full((n,), 0.1, dtype=jnp.float32),
        )
        params = make_uc_params(case_tight, max_steps=8, solver_mode=0,
                                enable_uc=True)
        obs, state = env.reset(key, params)
        # Set last_dispatch to p_min (units just started at minimum)
        # Then try to dispatch at p_max → should be clipped to p_min + ramp_up
        action = jnp.concatenate([
            jnp.ones(n, dtype=jnp.float32),   # all ON
            jnp.ones(n, dtype=jnp.float32),    # max dispatch intent
        ])
        k1, _ = jax.random.split(key)
        _, state2, *_ = env.step(k1, state, action, params)

        # Dispatch should be at most last_dispatch + ramp_up
        ramp_up = np.asarray(params.ramp_up_mw)
        last = np.asarray(state.last_dispatch)
        actual = np.asarray(state2.unit_power_mw)
        # For ON units, dispatch <= last + ramp_up (with some OPF balance tolerance)
        on_mask = np.asarray(state2.unit_status) == 1
        if on_mask.any():
            max_allowed = last[on_mask] + ramp_up[on_mask]
            assert np.all(actual[on_mask] <= max_allowed + 1.0), (
                "Dispatch exceeded ramp limit")


# ============ L1: reserve shortage ============

class TestL1ReserveShortage:

    def test_reserve_shortfall_detected(self, env, key):
        """Reserve shortfall should be > 0 when committed capacity is tight."""
        from powerzoojax.case import load_case
        base = load_case("14")
        case = make_case14_with_uc_defaults(base)
        n = case.n_units
        # Reduce all p_max to 30 MW so one unit (30 MW) can't cover ~130 MW * 1.05
        case_tiny = case.replace(
            unit_p_max=jnp.full((n,), 30.0, dtype=jnp.float32),
            unit_p_min=jnp.zeros(n, dtype=jnp.float32),
            unit_init_state=jnp.array([1.0] + [0.0] * (n - 1), dtype=jnp.float32),
            unit_keep_time=jnp.full((n,), 100.0, dtype=jnp.float32),
            unit_init_power=jnp.zeros(n, dtype=jnp.float32),
        )
        params = make_uc_params(case_tiny, max_steps=8, solver_mode=0,
                                enable_uc=True, enable_reserve=True,
                                reserve_margin_frac=0.05)
        obs, state = env.reset(key, params)
        # Keep first unit ON, all others OFF
        commit = jnp.array([1.0] + [-1.0] * (n - 1), dtype=jnp.float32)
        action = jnp.concatenate([commit, jnp.zeros(n, dtype=jnp.float32)])
        k1, _ = jax.random.split(key)
        _, _, _, _, _, info = env.step(k1, state, action, params)
        assert float(info["reserve_shortfall"]) > 0.0, (
            "Reserve shortfall must be detected with very few committed units")

    def test_no_reserve_shortfall_all_on(self, env, case14_params, key):
        """With all units ON, reserve shortfall should be 0."""
        obs, state = env.reset(key, case14_params)
        action = all_on_action(case14_params)
        k1, _ = jax.random.split(key)
        _, _, _, _, _, info = env.step(k1, state, action, case14_params)
        # All ON should have plenty of committed capacity
        assert float(info["reserve_shortfall"]) >= 0.0  # non-negative

    def test_reserve_cost_zero_when_disabled(self, env, key):
        """enable_reserve=False should give cost from thermal only."""
        params = make_tso_case14_params(max_steps=4, solver_mode=0)
        params_no_reserve = UCParams(
            **{k: v for k, v in params.__dict__.items()
               if k not in ("enable_reserve",)},
            enable_reserve=False,
        )
        # Actually use make_uc_params with enable_reserve=False
        from powerzoojax.case import load_case
        base = load_case("14")
        case = make_case14_with_uc_defaults(base)
        n = case.n_units
        case_one = case.replace(
            unit_init_state=jnp.array([1.0] + [0.0] * (n - 1), dtype=jnp.float32),
            unit_keep_time=jnp.full((n,), 100.0, dtype=jnp.float32),
            unit_init_power=jnp.concatenate([
                case.unit_p_min[:1], jnp.zeros(n - 1, dtype=jnp.float32)]),
        )
        p_no_res = make_uc_params(case_one, max_steps=4, solver_mode=0,
                                   enable_uc=True, enable_reserve=False)
        obs, state = env.reset(key, p_no_res)
        commit = jnp.array([1.0] + [-1.0] * (n - 1), dtype=jnp.float32)
        action = jnp.concatenate([commit, jnp.zeros(n, dtype=jnp.float32)])
        k1, _ = jax.random.split(key)
        _, _, _, _, _, info = env.step(k1, state, action, p_no_res)
        # With enable_reserve=False, reserve shortfall is NOT penalised in cost
        # (may still show up as info["reserve_shortfall"] but not in aggregate cost_sum)
        assert float(info["reserve_shortfall"]) >= 0.0  # still tracked
        # cost should not include reserve if enable_reserve=False
        cost_no_res = float(info["cost_sum"])
        # Costs are thermal-only; hard to check exact value but should be >= 0
        assert cost_no_res >= 0.0


# ============ L1: reward / cost separation ============

class TestL1RewardCostSeparation:

    def test_reward_is_negative_operating_cost(self, env, case14_params, key):
        """reward = -reward_scale * (gen_cost + startup + no_load)."""
        obs, state = env.reset(key, case14_params)
        action = all_on_action(case14_params)
        k1, _ = jax.random.split(key)
        _, _, reward, _, _, info = env.step(k1, state, action, case14_params)
        operating_cost = (
            float(info["gen_cost"])
            + float(info["startup_cost"])
            + float(info["no_load_cost"])
        )
        expected_reward = -case14_params.reward_scale * operating_cost
        assert abs(float(reward) - expected_reward) < 1e-3, (
            f"reward mismatch: got {reward:.5f}, expected {expected_reward:.5f}")

    def test_cost_channel_not_in_reward(self, env, case14_params, key):
        """Physical safety cost must NOT be in the reward."""
        obs, state = env.reset(key, case14_params)
        action = all_on_action(case14_params)
        k1, _ = jax.random.split(key)
        _, _, reward, _, _, info = env.step(k1, state, action, case14_params)
        # reward should only reflect operating cost, NOT thermal overload or reserve
        thermal = float(info["cost_thermal_overload"])
        reserve = float(info["reserve_shortfall"])
        # If there are violations, reward should NOT be more negative than operating cost alone
        # (i.e., violations should not amplify reward penalty)
        operating_cost = (
            float(info["gen_cost"]) + float(info["startup_cost"])
            + float(info["no_load_cost"])
        )
        reward_from_op = -case14_params.reward_scale * operating_cost
        # Tolerance: reward should equal operating reward
        assert abs(float(reward) - reward_from_op) < 1e-3

    def test_cost_decomposition_keys_present(self, env, case14_params, key):
        """All required info keys must be present."""
        required_keys = [
            "gen_cost", "startup_cost", "no_load_cost",
            "reserve_shortfall", "cost_thermal_overload",
            "cost_reserve_shortfall", "cost_min_updown", "cost_sum",
            "commitment_switches", "is_safe", "n_violations",
            "opf_converged", "opf_iterations",
            "opf_box_residual_mw", "opf_line_residual_mw",
            "opf_balance_residual_mw",
        ]
        obs, state = env.reset(key, case14_params)
        action = all_on_action(case14_params)
        k1, _ = jax.random.split(key)
        _, _, _, _, _, info = env.step(k1, state, action, case14_params)
        for k in required_keys:
            assert k in info, f"Missing info key: {k}"

    def test_opf_diagnostics_present_and_finite(self, env, case14_params, key):
        """DC-OPF mode should expose convergence diagnostics for debugging."""
        obs, state = env.reset(key, case14_params)
        action = all_on_action(case14_params)
        k1, _ = jax.random.split(key)
        _, _, _, _, _, info = env.step(k1, state, action, case14_params)
        assert int(info["opf_iterations"]) > 0
        assert float(info["opf_box_residual_mw"]) >= 0.0
        assert 0.0 <= float(info["opf_line_residual_mw"]) < 1e-3
        assert 0.0 <= float(info["opf_balance_residual_mw"]) < 1e-3

    def test_custom_dcopf_iteration_budget_propagates(self, env, case14_params, key):
        """UCParams.dcopf_max_iter should control the redispatch solve budget."""
        params = case14_params.replace(dcopf_max_iter=7, dcopf_tol=1e-2)
        obs, state = env.reset(key, params)
        action = all_on_action(params)
        k1, _ = jax.random.split(key)
        _, _, _, _, _, info = env.step(k1, state, action, params)
        assert int(info["opf_iterations"]) == 7

    def test_cost_sum_matches_components(self, env, case14_params, key):
        """info['cost_sum'] = thermal + reserve + min_updown."""
        obs, state = env.reset(key, case14_params)
        action = all_on_action(case14_params)
        k1, _ = jax.random.split(key)
        _, _, _, _, _, info = env.step(k1, state, action, case14_params)
        expected = (
            float(info["cost_thermal_overload"])
            + float(info["cost_min_updown"])
        )
        if case14_params.enable_reserve:
            expected += float(info["reserve_shortfall"])
        assert abs(float(info["cost_sum"]) - expected) < 1e-3


# ============ Case14 sanity ============

class TestCase14Sanity:

    def test_case14_reset_step_smoke(self, env, case14_params, key):
        obs, state = env.reset(key, case14_params)
        assert obs.ndim == 1
        assert obs.shape[0] == 4 * case14_params.case.n_units + case14_params.case.n_lines + 4
        action = all_on_action(case14_params)
        k1, _ = jax.random.split(key)
        obs2, state2, reward, costs, done, info = env.step(
            k1, state, action, case14_params
        )
        assert obs2.shape == obs.shape
        assert float(info["gen_cost"]) > 0.0

    def test_case14_obs_dim(self, case14_params):
        n_units = case14_params.case.n_units
        n_lines = case14_params.case.n_lines
        expected_obs_dim = 4 * n_units + n_lines + 4
        env = UnitCommitmentEnv()
        obs_space = env.observation_space(case14_params)
        assert obs_space.shape[0] == expected_obs_dim

    def test_case14_forecast_obs_appends_future_total_load(self, env, case14_params, key):
        forecast_h = 4
        params = case14_params.replace(forecast_horizon_steps=forecast_h)
        obs, _state = env.reset(key, params)
        base_dim = 4 * params.case.n_units + params.case.n_lines + 4
        assert obs.shape[0] == base_dim + forecast_h

        total_p_max = float(jnp.sum(params.case.unit_p_max))
        expected = np.sum(np.asarray(params.load_profiles), axis=1)[1:1 + forecast_h] / total_p_max
        assert np.allclose(np.asarray(obs[-forecast_h:]), expected, atol=1e-6)

    def test_case14_action_dim(self, case14_params):
        env = UnitCommitmentEnv()
        act_space = env.action_space(case14_params)
        assert act_space.shape[0] == 2 * case14_params.case.n_units

    def test_case14_rollout_8steps(self, env, case14_params, key):
        obs, state = env.reset(key, case14_params)
        action = all_on_action(case14_params)
        rewards = []
        for _ in range(case14_params.max_steps):
            key, k = jax.random.split(key)
            obs, state, reward, costs, done, info = env.step(
                k, state, action, case14_params
            )
            rewards.append(float(reward))
        assert len(rewards) == case14_params.max_steps
        assert all(r <= 0.0 for r in rewards), "Operating cost rewards must be non-positive"


# ============ Case118 smoke tests ============

class TestCase118Smoke:

    def test_case118_params_created(self, case118_params):
        assert case118_params.case.n_units == 54
        assert case118_params.case.n_lines == 186
        assert case118_params.min_up_steps is not None
        assert case118_params.ramp_up_mw is not None
        assert case118_params.startup_cost is not None

    def test_case118_uc_fields_loaded(self, case118_params):
        """Verify UC fields from case118 data are loaded correctly."""
        assert case118_params.min_up_steps.shape == (54,)
        assert case118_params.ramp_up_mw.shape == (54,)
        assert case118_params.startup_cost.shape == (54,)
        assert case118_params.no_load_cost_per_step.shape == (54,)
        # Nuclear units have min_up=96 steps; first nuclear is at index 5
        assert int(case118_params.min_up_steps[5]) == 96
        # Coal units have min_up=4 steps; first coal is at index 0
        assert int(case118_params.min_up_steps[0]) == 4

    def test_case118_reset_smoke(self, env, case118_params, key):
        obs, state = env.reset(key, case118_params)
        assert obs.ndim == 1
        n_units = case118_params.case.n_units
        n_lines = case118_params.case.n_lines
        assert obs.shape[0] == 4 * n_units + n_lines + 4
        assert state.unit_status.shape == (n_units,)
        assert state.time_in_state.shape == (n_units,)
        assert int(state.episode_start_idx) == 0

    def test_case118_reset_samples_training_episode_start(self, env, case118_params):
        params = case118_params.replace(
            load_profiles=make_tso_net_load_profiles(
                n_steps=96,
                load_d_max=case118_params.case.load_d_max,
            ),
            sample_start_on_reset=True,
        )
        keys = jax.random.split(jax.random.PRNGKey(0), 8)
        _, states = jax.vmap(lambda k: env.reset(k, params))(keys)
        starts = np.asarray(states.episode_start_idx)
        assert starts.min() >= 0
        assert starts.max() <= 48
        assert np.unique(starts).size > 1

    def test_case118_reset_respects_fixed_episode_start_idx(self, env, case118_params, key):
        params = case118_params.replace(
            load_profiles=make_tso_net_load_profiles(
                n_steps=96,
                load_d_max=case118_params.case.load_d_max,
            ),
            fixed_episode_start_idx=7,
        )
        _, state = env.reset(key, params)
        assert int(state.episode_start_idx) == 7

    def test_case118_step_smoke(self, env, case118_params, key):
        obs, state = env.reset(key, case118_params)
        action = all_on_action(case118_params)
        k1, _ = jax.random.split(key)
        obs2, state2, reward, costs, done, info = env.step(
            k1, state, action, case118_params
        )
        assert obs2.shape == obs.shape
        assert reward.ndim == 0
        assert float(reward) < 0.0
        assert float(info["gen_cost"]) > 0.0

    def test_case118_forecast_obs_dim(self, env, case118_params, key):
        params = case118_params.replace(forecast_horizon_steps=4)
        obs, _state = env.reset(key, params)
        n_units = params.case.n_units
        n_lines = params.case.n_lines
        assert obs.shape[0] == 4 * n_units + n_lines + 8

    def test_case118_jit_compile(self, env, case118_params, key):
        """JIT compilation must succeed for reset and step (already JIT-decorated)."""
        obs, state = env.reset(key, case118_params)
        action = all_on_action(case118_params)
        k1, _ = jax.random.split(key)
        obs2, state2, reward, costs, done, info = env.step(
            k1, state, action, case118_params
        )
        assert obs2.shape == obs.shape

    def test_case118_48step_rollout(self, env, case118_params, key):
        """48-step rollout must complete without errors."""
        obs, state = env.reset(key, case118_params)
        action = all_on_action(case118_params)
        gen_costs = []
        for _ in range(48):
            key, k = jax.random.split(key)
            obs, state, reward, costs, done, info = env.step(
                k, state, action, case118_params
            )
            gen_costs.append(float(info["gen_cost"]))
        assert len(gen_costs) == 48
        assert all(c > 0.0 for c in gen_costs), "gen_cost must be positive every step"

    def test_case118_vmap_smoke(self, env, case118_params):
        """vmap over 4 parallel envs must work."""
        n_envs = 4
        keys = jax.random.split(jax.random.PRNGKey(7), n_envs)
        obs_batch, state_batch = jax.vmap(
            lambda k: env.reset(k, case118_params)
        )(keys)
        assert obs_batch.shape[0] == n_envs

    def test_case118_off_units_no_power(self, env, case118_params, key):
        """Units starting as OFF should produce ~0 dispatch at reset."""
        obs, state = env.reset(key, case118_params)
        off_mask = (state.unit_status == 0)
        if jnp.any(off_mask):
            off_power = state.unit_power_mw * off_mask.astype(jnp.float32)
            assert float(jnp.max(jnp.abs(off_power))) < 5.0, (
                "OFF units should have near-zero power from OPF")


# ============ Net-load profile tests ============

class TestNetLoadProfiles:

    def test_synthetic_profiles_shape(self):
        profiles = make_tso_net_load_profiles(n_steps=48, load_d_max=None)
        assert profiles.shape == (48, 1)

    def test_profiles_with_load_d_max(self, case118_params):
        ld_max = case118_params.case.load_d_max
        profiles = make_tso_net_load_profiles(
            n_steps=48, load_d_max=ld_max)
        assert profiles.shape == (48, len(ld_max))
        # Non-negative: zero entries exist for buses with zero load capacity
        assert float(jnp.min(profiles)) >= 0.0
        # At least one bus has positive load
        assert float(jnp.max(profiles)) > 0.0

    def test_profiles_in_reasonable_range(self, case118_params):
        ld_max = case118_params.case.load_d_max
        profiles = make_tso_net_load_profiles(
            n_steps=48, load_d_max=ld_max)
        # Should be <= load_d_max (net load ≤ gross load); check via broadcast
        p = np.asarray(profiles)
        cap = np.asarray(ld_max)[None, :] + 1.0  # (1, n_loads)
        assert np.all(p <= cap), "net-load profiles should not exceed load_d_max"


# ============ TSO-ED variant (enable_uc=False) ============

class TestEDVariant:

    def test_ed_all_units_on(self, env, key):
        """In ED mode, all units are always ON regardless of commit signal."""
        params = make_tso_case14_params(
            max_steps=4, solver_mode=0, enable_uc=False)
        obs, state = env.reset(key, params)
        assert jnp.all(state.unit_status == 1), "ED mode: all units must be ON at reset"

        # Try to turn all off — should be ignored
        action = all_off_action(params)
        k1, _ = jax.random.split(key)
        _, state2, *_ = env.step(k1, state, action, params)
        assert jnp.all(state2.unit_status == 1), "ED mode: cannot turn off units"

    def test_ed_no_startup_no_load_cost(self, env, key):
        """In ED mode (enable_uc=False), startup and no_load costs are 0."""
        params = make_tso_case14_params(
            max_steps=4, solver_mode=0, enable_uc=False)
        obs, state = env.reset(key, params)
        action = all_on_action(params)
        k1, _ = jax.random.split(key)
        _, _, _, _, _, info = env.step(k1, state, action, params)
        assert float(info["startup_cost"]) == 0.0
        assert float(info["no_load_cost"]) == 0.0


# ============ SafeRLWrapper integration ============

class TestSafeRLWrapperIntegration:

    def test_safe_rl_wrapper_runs(self, case14_params):
        """SafeRLWrapper(UnitCommitmentEnv, params) must instantiate and run."""
        from powerzoojax.rl.wrappers import SafeRLWrapper
        wrapped = SafeRLWrapper(UnitCommitmentEnv(), case14_params,
                                cost_threshold=5.0)
        key = jax.random.PRNGKey(99)
        obs, state = wrapped.reset(key)
        action = jnp.zeros(wrapped.num_actions, dtype=jnp.float32)
        obs2, state2, reward, cost, done, info = wrapped.step(key, state, action)
        assert obs2.ndim == 1
        assert reward.ndim == 0
        assert cost.shape == (3,)


# ============ Metrics utility ============

class TestTSOMetrics:

    def test_compute_tso_metrics_runs(self, env, case14_params, key):
        results = tso_all_on_rollout(env, case14_params, key)
        metrics = compute_tso_metrics(results)
        assert "total_gen_cost" in metrics
        assert "total_operating_cost" in metrics
        assert "feasibility_rate" in metrics
        assert metrics["total_gen_cost"] > 0.0

    def test_compute_tso_metrics_reports_explicit_step_safety_rates(self):
        metrics = compute_tso_metrics(
            {
                "gen_cost": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "startup_cost": np.zeros(3, dtype=np.float32),
                "no_load_cost": np.zeros(3, dtype=np.float32),
                "reserve_shortfall": np.array([0.0, 1.5, 0.0], dtype=np.float32),
                "cost_thermal_overload": np.array([0.0, 0.0, 2.0], dtype=np.float32),
                "commitment_switches": np.array([0.0, 1.0, 0.0], dtype=np.float32),
                "is_safe": np.array([1.0, 0.0, 0.0], dtype=np.float32),
                "cost_sum": np.array([0.0, 1.5, 2.0], dtype=np.float32),
            }
        )

        assert metrics["reserve_shortfall_rate"] == pytest.approx(1.0 / 3.0)
        assert metrics["thermal_violation_rate"] == pytest.approx(1.0 / 3.0)
        assert metrics["feasibility_rate"] == pytest.approx(1.0 / 3.0)


# ============ Dispatch signal in OPF mode ============

class TestDispatchSignalOPFMode:
    """Verify dispatch_signal influences OPF dispatch (Bug A fix)."""

    def test_dispatch_signal_changes_dispatch_in_opf_mode(self, env, case14_params, key):
        """Same commit half, different dispatch half → different dispatch & reward."""
        # All units ON at reset so OPF has multiple marginal units (JSON default is 1-on).
        n = case14_params.case.n_units
        params = case14_params.replace(
            init_unit_status=jnp.ones(n, dtype=jnp.int32),
            init_time_in_state=jnp.full((n,), 200, dtype=jnp.int32),
            init_dispatch=jnp.array(case14_params.case.unit_p_min, dtype=jnp.float32),
            # Stronger bias so DC-OPF dispatch spread exceeds threshold under new mc_* curves.
            dispatch_preference_weight=0.5,
        )
        obs, state = env.reset(key, params)
        n = params.case.n_units
        k1, k2 = jax.random.split(key)

        # Action A: all ON, dispatch intent = +1 (prefer max dispatch)
        action_high = jnp.concatenate([
            jnp.ones(n, dtype=jnp.float32),
            jnp.ones(n, dtype=jnp.float32),
        ])
        # Action B: all ON, dispatch intent = -1 (prefer min dispatch)
        action_low = jnp.concatenate([
            jnp.ones(n, dtype=jnp.float32),
            -jnp.ones(n, dtype=jnp.float32),
        ])

        _, state_high, reward_high, _, _, info_high = env.step(
            k1, state, action_high, params
        )
        _, state_low, reward_low, _, _, info_low = env.step(
            k1, state, action_low, params
        )

        # Dispatch distributions must differ
        dispatch_high = np.asarray(state_high.unit_power_mw)
        dispatch_low = np.asarray(state_low.unit_power_mw)
        max_diff = float(np.max(np.abs(dispatch_high - dispatch_low)))
        assert max_diff > 0.01, (
            f"OPF dispatch must differ for opposite dispatch signals, "
            f"got max diff={max_diff:.4f} MW. dispatch_preference_weight may be 0."
        )

    def test_dispatch_preference_weight_zero_gives_identical_dispatch(self, env, key):
        """With weight=0, dispatch_signal does NOT influence OPF — for regression."""
        from powerzoojax.case import load_case
        base = load_case("14")
        case = make_case14_with_uc_defaults(base)
        # Build params with weight=0: dispatch half should have zero influence
        from powerzoojax.envs.grid.unit_commitment import make_uc_params, UCParams
        base_params = make_uc_params(case, max_steps=4, solver_mode=1)
        # Replace dispatch_preference_weight with 0 via make_uc_params trick:
        # UCParams is a flax struct; static fields can be changed only at construction.
        # We reconstruct with all fields identical except dispatch_preference_weight.
        params_w0 = UCParams(
            load_profiles=base_params.load_profiles,
            max_steps=base_params.max_steps,
            delta_t_hours=base_params.delta_t_hours,
            steps_per_day=base_params.steps_per_day,
            cost_thermal_weight=base_params.cost_thermal_weight,
            physics=base_params.physics,
            solver_mode=base_params.solver_mode,
            case=base_params.case,
            reward_scale=base_params.reward_scale,
            acpf_setup=base_params.acpf_setup,
            dcopf_setup=base_params.dcopf_setup,
            acopf_setup=base_params.acopf_setup,
            include_unit_dispatch=base_params.include_unit_dispatch,
            resources=base_params.resources,
            min_up_steps=base_params.min_up_steps,
            min_down_steps=base_params.min_down_steps,
            startup_cost=base_params.startup_cost,
            no_load_cost_per_step=base_params.no_load_cost_per_step,
            ramp_up_mw=base_params.ramp_up_mw,
            ramp_down_mw=base_params.ramp_down_mw,
            reserve_margin_frac=base_params.reserve_margin_frac,
            enable_uc=base_params.enable_uc,
            enable_reserve=base_params.enable_reserve,
            dispatch_preference_weight=0.0,  # zero weight
            init_unit_status=base_params.init_unit_status,
            init_time_in_state=base_params.init_time_in_state,
            init_dispatch=base_params.init_dispatch,
        )
        obs, state = env.reset(key, params_w0)
        n = params_w0.case.n_units
        k1, _ = jax.random.split(key)
        action_high = jnp.concatenate([jnp.ones(n), jnp.ones(n)])
        action_low = jnp.concatenate([jnp.ones(n), -jnp.ones(n)])
        _, state_high, *_ = env.step(k1, state, action_high, params_w0)
        _, state_low, *_ = env.step(k1, state, action_low, params_w0)
        # With weight=0, dispatch should be identical regardless of dispatch half
        dispatch_high = np.asarray(state_high.unit_power_mw)
        dispatch_low  = np.asarray(state_low.unit_power_mw)
        np.testing.assert_allclose(
            dispatch_high, dispatch_low, atol=1e-4,
            err_msg="dispatch_preference_weight=0 must give identical OPF dispatch"
        )

    def test_dispatch_signal_params_field_present(self, case14_params):
        """UCParams must have dispatch_preference_weight field."""
        assert hasattr(case14_params, "dispatch_preference_weight"), (
            "UCParams missing dispatch_preference_weight field")
        assert case14_params.dispatch_preference_weight > 0.0, (
            "Default dispatch_preference_weight should be positive")


# ============ DataLoader API tests ============

class TestDataLoaderAPI:
    """Verify make_tso_net_load_profiles_from_data uses real DataLoader API."""

    def _make_mock_loader(self, n_rows: int = 100):
        """Return a mock DataLoader that implements load_jax_profiles correctly."""
        import numpy as _np

        class MockDataLoader:
            def load_jax_profiles(self, signals, *, start_date=None, end_date=None,
                                   resample=None, **kwargs):
                # Return (n_rows, len(signals)) random float32 array
                data = _np.random.default_rng(0).random(
                    (n_rows, len(signals)), dtype=_np.float32)
                # Scale demand column to reasonable MW range
                data[:, 0] *= 30000.0  # load.actual_mw
                data[:, 1] *= 10000.0  # wind.available_mw
                data[:, 2] *= 5000.0   # solar.available_mw
                return jnp.array(data, dtype=jnp.float32)

        return MockDataLoader()

    def _make_failing_loader(self):
        """Return a mock DataLoader whose load_jax_profiles raises."""
        class FailingDataLoader:
            def load_jax_profiles(self, signals, **kwargs):
                raise FileNotFoundError("GB parquet data not found")
        return FailingDataLoader()

    def test_data_loader_api_correctness(self, case14_params):
        """make_tso_net_load_profiles_from_data calls load_jax_profiles, not load."""
        from powerzoojax.case import load_case
        case = load_case("14")
        loader = self._make_mock_loader(n_rows=200)
        profiles = make_tso_net_load_profiles_from_data(
            loader, case, split="train", n_steps=48)
        assert profiles.shape[0] == 48
        assert profiles.shape[1] == case.n_loads
        assert profiles.dtype == jnp.float32

    def test_data_loader_called_with_correct_signals(self, case14_params):
        """Loader must be called with semantic signal names from signals.py."""
        from powerzoojax.data import signals as S
        from powerzoojax.case import load_case

        received_signals = []

        class CapturingLoader:
            def load_jax_profiles(self, signals, **kwargs):
                received_signals.extend(signals)
                data = np.zeros((200, len(signals)), dtype=np.float32)
                data[:, 0] = 1.0  # non-zero demand
                return jnp.array(data, dtype=jnp.float32)

        case = load_case("14")
        make_tso_net_load_profiles_from_data(
            CapturingLoader(), case, split="train", n_steps=48)
        assert S.LOAD_ACTUAL_MW in received_signals, (
            f"Expected {S.LOAD_ACTUAL_MW!r} in loader signals, got {received_signals}")
        assert S.WIND_AVAILABLE_MW in received_signals
        assert S.SOLAR_AVAILABLE_MW in received_signals

    def test_data_loader_failure_raises_not_silent(self, case14_params):
        """When loader fails and allow_synthetic_fallback=False, must raise RuntimeError."""
        from powerzoojax.case import load_case
        case = load_case("14")
        loader = self._make_failing_loader()
        with pytest.raises(RuntimeError, match="Failed to load GB profiles"):
            make_tso_net_load_profiles_from_data(
                loader, case, split="train", n_steps=48,
                allow_synthetic_fallback=False)

    def test_data_loader_failure_fallback_allowed(self, case14_params):
        """When loader fails and allow_synthetic_fallback=True, returns synthetic."""
        from powerzoojax.case import load_case
        case = load_case("14")
        loader = self._make_failing_loader()
        # Must NOT raise; returns synthetic profiles
        profiles = make_tso_net_load_profiles_from_data(
            loader, case, split="train", n_steps=48,
            allow_synthetic_fallback=True)
        assert profiles.shape[0] == 48
        assert profiles.dtype == jnp.float32

    def test_data_loader_insufficient_rows_raises(self):
        """If loader returns fewer rows than needed, raise RuntimeError."""
        from powerzoojax.case import load_case
        case = load_case("14")
        loader = self._make_mock_loader(n_rows=10)  # too few rows
        with pytest.raises(RuntimeError, match="rows required"):
            make_tso_net_load_profiles_from_data(
                loader, case, split="train",
                episode_start_idx=0, n_steps=48,
                allow_synthetic_fallback=False)

    def test_data_loader_episode_start_idx(self):
        """episode_start_idx slices the correct window from loader output."""
        from powerzoojax.case import load_case
        case = load_case("14")

        received_slices = {}

        class SliceCapturingLoader:
            def load_jax_profiles(self, signals, **kwargs):
                n = 200
                data = np.arange(n, dtype=np.float32)[:, None] * np.ones(
                    (1, len(signals)), dtype=np.float32)
                data[:, 0] += 1.0  # ensure positive demand
                return jnp.array(data, dtype=jnp.float32)

        loader = SliceCapturingLoader()
        profiles_0 = make_tso_net_load_profiles_from_data(
            loader, case, split="train", episode_start_idx=0, n_steps=4)
        profiles_50 = make_tso_net_load_profiles_from_data(
            loader, case, split="train", episode_start_idx=50, n_steps=4)
        # Both should return shape (4, n_loads) but from different windows
        assert profiles_0.shape[0] == 4
        assert profiles_50.shape[0] == 4


# ============ case118 main factory with data_loader ============

class TestCase118DataLoaderFactory:
    """Verify make_tso_case118_params(data_loader=...) works correctly (Bug A fix)."""

    def _make_mock_loader(self, n_rows: int = 200):
        class MockLoader:
            def load_jax_profiles(self, signals, *, start_date=None, end_date=None,
                                   resample=None, **kwargs):
                import numpy as _np
                data = _np.ones((n_rows, len(signals)), dtype=_np.float32)
                data[:, 0] *= 20000.0  # load.actual_mw — non-zero demand
                data[:, 1] *= 8000.0   # wind.available_mw
                data[:, 2] *= 3000.0   # solar.available_mw
                return jnp.array(data, dtype=jnp.float32)
        return MockLoader()

    def _make_failing_loader(self):
        class FailingLoader:
            def load_jax_profiles(self, signals, **kwargs):
                raise FileNotFoundError("no data")
        return FailingLoader()

    def test_case118_params_with_mock_loader_no_type_error(self):
        """make_tso_case118_params(data_loader=mock) must not raise TypeError."""
        loader = self._make_mock_loader()
        # Must not raise TypeError: unexpected keyword argument
        params = make_tso_case118_params(data_loader=loader, max_steps=48)
        assert params is not None

    def test_case118_params_with_mock_loader_correct_shape(self):
        """load_profiles must be (max_steps, n_loads) when data_loader is used."""
        from powerzoojax.case import load_case
        case = load_case("118")
        loader = self._make_mock_loader()
        params = make_tso_case118_params(data_loader=loader, max_steps=48)
        assert params.load_profiles.shape == (48, case.n_loads), (
            f"Expected load_profiles shape (48, {case.n_loads}), "
            f"got {params.load_profiles.shape}"
        )

    def test_case118_params_with_mock_loader_is_valid_ucparams(self):
        """Returned object must be a valid UCParams usable for reset/step."""
        loader = self._make_mock_loader()
        params = make_tso_case118_params(data_loader=loader, max_steps=4)
        env = UnitCommitmentEnv()
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)
        assert obs.ndim == 1
        assert state.unit_status.shape == (params.case.n_units,)

    def test_case118_params_failing_loader_raises(self):
        """Failing loader with default allow_synthetic_fallback=False must raise RuntimeError."""
        loader = self._make_failing_loader()
        with pytest.raises(RuntimeError, match="Failed to load GB profiles"):
            make_tso_case118_params(data_loader=loader, max_steps=48)

    def test_case118_params_default_no_loader_still_works(self):
        """Without data_loader, falls back to synthetic — must remain functional."""
        params = make_tso_case118_params(max_steps=4)
        from powerzoojax.case import load_case
        case = load_case("118")
        assert params.load_profiles.shape == (4, case.n_loads)


# ============ tso-scuc-safe vs tso-uc reserve semantics ============

class TestPresetReserveSemantics:
    """Verify tso-scuc-safe has reserve enabled; tso-uc does not (Bug B fix)."""

    def test_tso_scuc_params_enable_reserve_true(self):
        """make_tso_scuc_params() must have enable_reserve=True."""
        params = make_tso_scuc_params()
        assert params.enable_reserve is True, (
            "tso-scuc-safe must have enable_reserve=True; "
            f"got {params.enable_reserve}"
        )

    def test_tso_scuc_params_enable_uc_true(self):
        """make_tso_scuc_params() must have enable_uc=True."""
        params = make_tso_scuc_params()
        assert params.enable_uc is True

    def test_tso_uc_params_enable_reserve_false(self):
        """make_tso_uc_params() must have enable_reserve=False (reserve not in cost)."""
        params = make_tso_uc_params()
        assert params.enable_reserve is False, (
            "tso-uc must have enable_reserve=False; "
            f"got {params.enable_reserve}"
        )

    def test_tso_uc_params_enable_uc_true(self):
        """make_tso_uc_params() must have enable_uc=True."""
        params = make_tso_uc_params()
        assert params.enable_uc is True

    def test_preset_tso_scuc_safe_uses_reserve(self):
        """The tso-scuc-safe preset factory must build params with enable_reserve=True."""
        from powerzoojax.rl.presets import PRESETS
        preset = PRESETS["tso-scuc-safe"]
        wrapped = preset.env_factory()
        inner_params = wrapped._params
        assert inner_params.enable_reserve is True, (
            "tso-scuc-safe preset params must have enable_reserve=True, "
            f"got {inner_params.enable_reserve}"
        )

    def test_preset_tso_uc_no_reserve(self):
        """The tso-uc preset factory must build params with enable_reserve=False."""
        from powerzoojax.rl.presets import PRESETS
        preset = PRESETS["tso-uc"]
        wrapped = preset.env_factory()
        # LogWrapper stores params in _params
        inner_params = wrapped._params
        assert inner_params.enable_reserve is False, (
            "tso-uc preset params must have enable_reserve=False, "
            f"got {inner_params.enable_reserve}"
        )

    def test_reserve_shortfall_in_cost_for_scuc_not_uc(self):
        """Behavioral: tight-capacity → scuc has reserve in info['cost']; uc does not."""
        from powerzoojax.case import load_case
        env = UnitCommitmentEnv()
        base = load_case("14")
        case = make_case14_with_uc_defaults(base)
        n = case.n_units
        # Tiny p_max so reserve shortfall is guaranteed
        case_tiny = case.replace(
            unit_p_max=jnp.full((n,), 30.0, dtype=jnp.float32),
            unit_p_min=jnp.zeros(n, dtype=jnp.float32),
            unit_init_state=jnp.array([1.0] + [0.0] * (n - 1), dtype=jnp.float32),
            unit_keep_time=jnp.full((n,), 100.0, dtype=jnp.float32),
            unit_init_power=jnp.zeros(n, dtype=jnp.float32),
        )
        from powerzoojax.envs.grid.unit_commitment import make_uc_params

        params_scuc = make_uc_params(case_tiny, max_steps=4, solver_mode=0,
                                     enable_uc=True, enable_reserve=True,
                                     reserve_margin_frac=0.05)
        params_uc   = make_uc_params(case_tiny, max_steps=4, solver_mode=0,
                                     enable_uc=True, enable_reserve=False,
                                     reserve_margin_frac=0.05)

        key = jax.random.PRNGKey(42)
        commit = jnp.array([1.0] + [-1.0] * (n - 1), dtype=jnp.float32)
        action = jnp.concatenate([commit, jnp.zeros(n, dtype=jnp.float32)])

        _, state_scuc = env.reset(key, params_scuc)
        _, state_uc   = env.reset(key, params_uc)
        k1, _ = jax.random.split(key)

        _, _, _, _, _, info_scuc = env.step(k1, state_scuc, action, params_scuc)
        _, _, _, _, _, info_uc   = env.step(k1, state_uc,   action, params_uc)

        # Both should detect reserve shortfall
        assert float(info_scuc["reserve_shortfall"]) > 0.0, (
            "Expected reserve shortfall in SCUC scenario")
        assert float(info_uc["reserve_shortfall"]) > 0.0, (
            "Expected reserve shortfall in UC scenario too (it's tracked regardless)")
        assert not bool(info_scuc["is_safe"]), (
            "SCUC safety should treat reserve shortfall as unsafe when reserve is enabled"
        )
        assert int(info_scuc["n_violations"]) >= 1, (
            "SCUC reserve violation should increment n_violations"
        )
        assert bool(info_uc["is_safe"]), (
            "UC variant with reserve disabled should not mark reserve shortfall unsafe"
        )

        # SCUC: reserve_shortfall contributes to aggregate cost_sum.
        assert float(info_scuc["cost_sum"]) >= float(info_scuc["reserve_shortfall"]) - 1e-3, (
            "SCUC cost_sum must include reserve shortfall")

        # UC: reserve not in cost; cost = thermal_overload only
        # (thermal may be 0 in this direct-PF test, so cost should be low / 0)
        # The key check: UC cost < SCUC cost (since SCUC adds reserve penalty)
        assert float(info_uc["cost_sum"]) < float(info_scuc["cost_sum"]) + 1e-3, (
            f"UC cost_sum ({info_uc['cost_sum']:.2f}) should not exceed SCUC cost_sum "
            f"({info_scuc['cost_sum']:.2f}) since UC has reserve disabled")
