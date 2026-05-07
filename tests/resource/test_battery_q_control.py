"""R1 — Verify BatteryBundle Q-control path can influence bus voltage.

The ``BatteryBundle`` already exposes ``enable_q_control=True`` and returns
``q_inject_mvar``.  ``DistGridEnv.step`` routes the bundle's reactive
injection into the BFS power flow via ``bundle_q_injection_mvar`` → ``q_net``.
This test is the R1 closure: it asserts that when the bundle is driven with
pure reactive actions (P=0, Q≠0) the resulting ``v_mag`` differs from the
P-only path, confirming that the Q signal actually reaches the power flow and
that voltage support is a real lever for the DERs task.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powerzoojax.case import create_case33bw
from powerzoojax.envs.grid.dist import DistGridEnv, make_dist_params
from powerzoojax.envs.resource.battery import (
    BatteryBundle,
    make_battery_bundle,
)


# ========================== Fixtures ==========================


@pytest.fixture(scope="module")
def case33():
    return create_case33bw()


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


def _make_dist_with_battery(case33, *, enable_q_control: bool) -> tuple:
    """Construct a DistGridEnv + 3-battery bundle on case33bw feeder ends."""
    bundle = make_battery_bundle(
        case33,
        bus_ids=[18, 22, 33],       # feeder ends (most voltage-sensitive)
        power_mw=0.20,
        capacity_mwh=0.40,
        s_rated_mva=0.30,           # extra MVA headroom for Q
        enable_q_control=enable_q_control,
        soc_min=0.1,
        soc_max=0.9,
        initial_soc=0.5,
    )
    params = make_dist_params(
        case33,
        max_steps=12,
        resources=(bundle,),
        include_der=False,
        v_min=0.94,
        v_max=1.06,
    )
    return DistGridEnv(), params, bundle


# ========================== Tests ==========================


def test_bundle_q_dim(case33):
    """enable_q_control=True doubles per-device action/obs dim."""
    _, _, b_p = _make_dist_with_battery(case33, enable_q_control=False)
    _, _, b_pq = _make_dist_with_battery(case33, enable_q_control=True)
    assert b_p.per_device_action_dim == 1
    assert b_p.per_device_obs_dim == 2
    assert b_pq.per_device_action_dim == 2
    assert b_pq.per_device_obs_dim == 3
    assert b_pq.action_dim == 6      # 3 devices × 2
    assert b_pq.obs_dim == 9         # 3 devices × 3


def test_q_action_changes_voltage(case33, key):
    """Pure reactive action (P=0, Q=+full) shifts the voltage profile."""
    env, params, bundle = _make_dist_with_battery(case33, enable_q_control=True)

    # Pure-P zero baseline: P=0, Q=0.
    act_zero = jnp.zeros(bundle.action_dim, dtype=jnp.float32)
    # Pure reactive injection: P=0, Q=+1 (full capacitive support).
    act_q_pos = act_zero.at[1::2].set(1.0)
    # Pure reactive absorption: P=0, Q=-1 (full inductive absorption).
    act_q_neg = act_zero.at[1::2].set(-1.0)

    _, state = env.reset(key, params)

    _, s_zero, _, _costs, _, _ = env.step(key, state, act_zero, params)
    _, s_q_pos, _, _costs, _, _ = env.step(key, state, act_q_pos, params)
    _, s_q_neg, _, _costs, _, _ = env.step(key, state, act_q_neg, params)

    # Capacitive Q injection should raise voltages somewhere (vs zero baseline).
    delta_pos = s_q_pos.v_mag - s_zero.v_mag
    delta_neg = s_q_neg.v_mag - s_zero.v_mag

    # Max absolute voltage change must be clearly nonzero.
    assert float(jnp.max(jnp.abs(delta_pos))) > 1e-4
    assert float(jnp.max(jnp.abs(delta_neg))) > 1e-4

    # The two directions must move voltage in opposite senses at the bus
    # nearest the injection point (reactive support → higher V,
    # reactive absorption → lower V).
    # Pick the largest absolute change from delta_pos; check delta_neg has
    # the opposite sign at that bus.
    idx = int(jnp.argmax(jnp.abs(delta_pos)))
    assert float(delta_pos[idx]) * float(delta_neg[idx]) < 0.0, (
        f"At bus {idx}: delta_pos={float(delta_pos[idx])}, "
        f"delta_neg={float(delta_neg[idx])} (expected opposite signs)"
    )


def test_q_action_changes_losses(case33, key):
    """Pure reactive actions change the reported network losses.

    Active power loss (I²R) depends on both P and Q flows; a non-trivial Q
    injection must produce a measurably different ``p_loss_total`` than the
    zero-action baseline, proving Q hits the BFS solver.
    """
    env, params, bundle = _make_dist_with_battery(case33, enable_q_control=True)

    act_zero = jnp.zeros(bundle.action_dim, dtype=jnp.float32)
    act_q_pos = act_zero.at[1::2].set(1.0)

    _, state = env.reset(key, params)
    _, s_zero, _, _costs, _, info_zero = env.step(key, state, act_zero, params)
    _, s_q, _, _costs, _, info_q = env.step(key, state, act_q_pos, params)

    loss_zero = float(info_zero["p_loss_MW"])
    loss_q = float(info_q["p_loss_MW"])
    assert abs(loss_zero - loss_q) > 1e-4, (
        f"loss_zero={loss_zero}, loss_q={loss_q}"
    )


def test_p_only_vs_pq_baseline_agree_when_q_zero(case33, key):
    """Sanity: P+Q mode with Q=0 should match P-only mode at same P."""
    env_p, params_p, bundle_p = _make_dist_with_battery(case33, enable_q_control=False)
    env_q, params_q, bundle_q = _make_dist_with_battery(case33, enable_q_control=True)

    # Same P command in both modes.
    p_cmd = jnp.array([0.3, -0.2, 0.1], dtype=jnp.float32)
    act_p = p_cmd                                      # shape (3,)
    act_q = jnp.stack([p_cmd, jnp.zeros_like(p_cmd)], axis=-1).reshape(-1)  # (6,)

    _, s0_p = env_p.reset(key, params_p)
    _, s0_q = env_q.reset(key, params_q)

    _, sp, *_ = env_p.step(key, s0_p, act_p, params_p)
    _, sq, *_ = env_q.step(key, s0_q, act_q, params_q)

    np.testing.assert_allclose(
        np.asarray(sp.v_mag), np.asarray(sq.v_mag), atol=1e-5,
    )


def test_jit_and_vmap_q_path(case33, key):
    """JIT and vmap compatibility for the Q-enabled path."""
    env, params, bundle = _make_dist_with_battery(case33, enable_q_control=True)
    act = jnp.zeros(bundle.action_dim, dtype=jnp.float32).at[1::2].set(0.5)

    step_jit = jax.jit(lambda k, s, a: env.step(k, s, a, params))
    _, state = env.reset(key, params)
    out1 = step_jit(key, state, act)
    out2 = step_jit(key, state, act)
    # Determinism
    np.testing.assert_allclose(np.asarray(out1[1].v_mag), np.asarray(out2[1].v_mag))

    # vmap across a batch of 4 independent initial states
    keys = jax.random.split(key, 4)
    resets = jax.vmap(lambda k: env.reset(k, params))(keys)
    _, batched_states = resets
    acts = jnp.broadcast_to(act, (4, act.shape[0]))
    batched_step = jax.vmap(lambda k, s, a: env.step(k, s, a, params))
    obs_b, st_b, r_b, costs_b, d_b, _ = batched_step(keys, batched_states, acts)
    assert st_b.v_mag.shape == (4, case33.n_nodes)
    assert r_b.shape == (4,)
    assert costs_b.shape[0] == 4
