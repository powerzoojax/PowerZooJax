"""Cross-backend TSO reward must use the same scale on both sides.

PowerZooJax sets ``reward_scale=1e-4`` in
``make_tso_case118_params`` (and the comparison env factory).  PowerZoo's
``CentralizedComparisonTSOEnv`` historically used the inner MARL env's
``/1000`` scaling, which left the SB3-facing ep_rew_mean signal 10x
larger on the PowerZoo side and broke same-hyperparameter portability.

This test guards against regression of that fix.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.common.powerzoo_repo import find_powerzoo_repo

_REPO_ROOT = Path(__file__).resolve().parents[2]
_POWERZOO_PATH = find_powerzoo_repo(_REPO_ROOT)


def _powerzoo_available() -> bool:
    return _POWERZOO_PATH is not None


def test_powerzoo_tso_reward_scale_matches_jax():
    """``REWARD_SCALE`` constant must equal PowerZooJax's reward_scale."""
    if not _powerzoo_available():
        pytest.skip("PowerZoo sibling repo not present")

    from powerzoo.tasks.middle.comparison_tso import CentralizedComparisonTSOEnv

    assert CentralizedComparisonTSOEnv.REWARD_SCALE == pytest.approx(1e-4), (
        f"PowerZoo TSO comparison env REWARD_SCALE="
        f"{CentralizedComparisonTSOEnv.REWARD_SCALE}; PowerZooJax canonical "
        f"value is 1e-4 (see make_tso_case118_params docstring).  Update one "
        f"side to match the other before recording cross-backend numbers."
    )


def test_powerzoo_tso_reward_equals_neg_scale_times_cost():
    """``reward_step ≈ -REWARD_SCALE * (gen + startup + no_load)``.

    Pulls one step from the env with a fixed action and checks the
    invariant directly.
    """
    if not _powerzoo_available():
        pytest.skip("PowerZoo sibling repo not present")
    pytest.importorskip("pandas")
    import numpy as np

    try:
        from powerzoo.tasks.middle.comparison_tso import (
            CentralizedComparisonTSOEnv,
            CentralizedComparisonTSOTask,
        )
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"PowerZoo comparison_tso import failed: {exc}")

    try:
        task = CentralizedComparisonTSOTask(split="train", episode_start_idx=0)
        env = CentralizedComparisonTSOEnv(task)
        env.reset(seed=0)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"PowerZoo TSO env build failed (likely missing GB parquet): {exc}")

    action = np.zeros(env.action_space.shape, dtype=np.float32)
    _obs, reward, _term, _trunc, info = env.step(action)

    expected = -env.REWARD_SCALE * (
        info["gen_cost"] + info["startup_cost"] + info["no_load_cost"]
    )
    assert reward == pytest.approx(expected, rel=1e-5, abs=1e-5), (
        f"TSO reward {reward} != -REWARD_SCALE * total_operating_cost "
        f"({expected}).  Step info: gen={info['gen_cost']}, "
        f"startup={info['startup_cost']}, no_load={info['no_load_cost']}."
    )


def test_powerzoo_tso_info_carries_operating_cost():
    """``info['operating_cost']`` must be the sum used for the reward,
    so downstream summarize/plot has the same definition both sides.
    """
    if not _powerzoo_available():
        pytest.skip("PowerZoo sibling repo not present")
    pytest.importorskip("pandas")
    import numpy as np

    from powerzoo.tasks.middle.comparison_tso import (
        CentralizedComparisonTSOEnv,
        CentralizedComparisonTSOTask,
    )
    try:
        env = CentralizedComparisonTSOEnv(CentralizedComparisonTSOTask())
        env.reset(seed=0)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"GB parquet unavailable: {exc}")

    _, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    expected = info["gen_cost"] + info["startup_cost"] + info["no_load_cost"]
    assert info["operating_cost"] == pytest.approx(expected)
