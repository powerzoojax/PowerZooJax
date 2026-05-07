"""Cross-backend split-alignment guardrail.

PowerZoo and PowerZooJax both define Ausgrid (and GB) split windows.  When
they drift, cross-backend records claim ``split="iid"`` while in reality
each backend is reading a different time window or substation pool.  The
fix is in 2026-04 ("cross-backend fairness fix plan", P0-2): make the
PowerZooJax constants the single source of truth and assert that PowerZoo's
mirror agrees.

This test does not import PowerZoo by default; if PowerZoo is not on the
import path, the test is skipped.  The benchmarks driver ensures PowerZoo
is on ``sys.path`` at runtime, so any cross-backend run will exercise the
same constants the test is checking.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from benchmarks.common.powerzoo_repo import ensure_powerzoo_on_path, find_powerzoo_repo

_REPO_ROOT = Path(__file__).resolve().parents[2]
_POWERZOO_PATH = find_powerzoo_repo(_REPO_ROOT)


def _ensure_powerzoo_on_path() -> bool:
    return ensure_powerzoo_on_path(_REPO_ROOT, append=True) is not None


# ---------------------------------------------------------------------------
# Ausgrid date windows must match PowerZooJax's splits.py.
# ---------------------------------------------------------------------------

def test_ausgrid_split_dates_match_powerzoojax():
    if not _ensure_powerzoo_on_path():
        pytest.skip("PowerZoo sibling repo not present; cross-backend tests skipped")

    from powerzoo.tasks.dso_task import _AUSGRID_SPLIT_DATES

    from powerzoojax.data.splits import (
        AUSGRID_TRAIN_START,
        AUSGRID_TRAIN_END,
        AUSGRID_IID_START,
        AUSGRID_IID_END,
        AUSGRID_SUMMER_START,
        AUSGRID_SUMMER_END,
    )

    expected = {
        "train":        (AUSGRID_TRAIN_START, AUSGRID_TRAIN_END),
        "iid":          (AUSGRID_IID_START, AUSGRID_IID_END),
        "summer_ood":   (AUSGRID_SUMMER_START, AUSGRID_SUMMER_END),
        "zone_holdout": (AUSGRID_TRAIN_START, AUSGRID_TRAIN_END),
    }

    for split, (jax_start, jax_end) in expected.items():
        assert split in _AUSGRID_SPLIT_DATES, (
            f"PowerZoo dso_task.py is missing split '{split}'. "
            f"Sync with powerzoojax/data/splits.py."
        )
        pz_start, pz_end = _AUSGRID_SPLIT_DATES[split]
        assert (pz_start, pz_end) == (jax_start, jax_end), (
            f"Ausgrid split '{split}' window mismatch:\n"
            f"  PowerZoo     : ({pz_start}, {pz_end})\n"
            f"  PowerZooJax  : ({jax_start}, {jax_end})\n"
            f"Single source of truth: powerzoojax/data/splits.py. "
            f"Update PowerZoo/powerzoo/tasks/dso_task.py::_AUSGRID_SPLIT_DATES."
        )


# ---------------------------------------------------------------------------
# GenCos GB date windows must match PowerZooJax's splits.py.
# ---------------------------------------------------------------------------

def test_gencos_split_dates_match_powerzoojax():
    if not _ensure_powerzoo_on_path():
        pytest.skip("PowerZoo sibling repo not present; cross-backend tests skipped")

    from powerzoo.tasks.simple.task_gencos import GenCosTask
    from powerzoojax.data.splits import (
        GB_IID_END,
        GB_IID_START,
        GB_TRAIN_END,
        GB_TRAIN_START,
    )

    expected = {
        "train": (GB_TRAIN_START, GB_TRAIN_END),
        "iid": (GB_IID_START, GB_IID_END),
        "demand_shift": (GB_IID_START, GB_IID_END),
        "renewable_shock": (GB_IID_START, GB_IID_END),
    }

    for split, dates in expected.items():
        assert GenCosTask.SPLIT_DATES.get(split) == dates, (
            f"GenCos split '{split}' mismatch:\n"
            f"  PowerZoo    : {GenCosTask.SPLIT_DATES.get(split)}\n"
            f"  PowerZooJax : {dates}\n"
            "PowerZooJax split constants are the source of truth."
        )


# ---------------------------------------------------------------------------
# Feeder substation pools must match PowerZooJax's ausgrid_utils.py.
# ---------------------------------------------------------------------------

def test_ausgrid_feeder_pools_match_powerzoojax():
    if not _ensure_powerzoo_on_path():
        pytest.skip("PowerZoo sibling repo not present; cross-backend tests skipped")

    from powerzoo.tasks.dso_task import _AUSGRID_FEEDER_POOLS as pz_pools
    from powerzoojax.data.ausgrid_utils import AUSGRID_FEEDER_POOLS as jax_pools

    assert set(pz_pools.keys()) == set(jax_pools.keys()), (
        f"Feeder set mismatch:\n"
        f"  PowerZoo    : {sorted(pz_pools.keys())}\n"
        f"  PowerZooJax : {sorted(jax_pools.keys())}"
    )

    for feeder in jax_pools:
        for role in ("train", "zone_holdout"):
            jax_subs = list(jax_pools[feeder][role])
            pz_subs = list(pz_pools[feeder].get(role, []))
            assert pz_subs == jax_subs, (
                f"Feeder '{feeder}' role '{role}' substation list differs:\n"
                f"  PowerZoo    : {pz_subs}\n"
                f"  PowerZooJax : {jax_subs}\n"
                f"PowerZooJax (powerzoojax/data/ausgrid_utils.py) is the source "
                f"of truth."
            )


# ---------------------------------------------------------------------------
# Day-level filter for iid/train must exist on the PowerZoo side too,
# otherwise PowerZoo's iid (after window alignment) collapses to train.
# ---------------------------------------------------------------------------

def test_powerzoo_iid_uses_day_level_filter():
    if not _ensure_powerzoo_on_path():
        pytest.skip("PowerZoo sibling repo not present; cross-backend tests skipped")

    # The fix plan exposes the day-filter as filter_ausgrid_role_days; mirror
    # PowerZooJax's `_IID_DAY_STRIDE` semantics so that train ∩ iid = ∅.
    from powerzoo.tasks.dso_task import filter_ausgrid_role_days  # noqa: F401

    # Smoke-check the filter actually splits a sample DataFrame into disjoint
    # subsets keyed by `role`.
    import pandas as pd

    ts = pd.date_range("2024-05-01", "2024-05-30", freq="30min", tz="UTC")
    df = pd.DataFrame({"datetime": ts, "value": range(len(ts))})

    train_df = filter_ausgrid_role_days(df, "train")
    iid_df = filter_ausgrid_role_days(df, "iid")

    assert len(train_df) > 0 and len(iid_df) > 0, (
        "Day filter returned empty DataFrame for train or iid."
    )
    overlap = set(train_df["value"]).intersection(set(iid_df["value"]))
    assert not overlap, (
        f"PowerZoo train and iid day filters must be disjoint, "
        f"but overlap has {len(overlap)} rows."
    )
