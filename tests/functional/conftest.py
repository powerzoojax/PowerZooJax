"""Shared fixtures and report utilities for functional tests."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from powerzoojax.case import create_case5
from powerzoojax.data import DataLoader, signals as S
from powerzoojax.envs.grid.trans import TransGridEnv, make_trans_params
from powerzoojax.envs.resource.battery import BatteryEnv, make_battery_params

REPORTS_ROOT = Path(__file__).resolve().parents[1] / "reports"


def pytest_collection_modifyitems(config, items):
    """Classify report-generation tests so they stay out of default regressions."""
    for item in items:
        if item.originalname == "test_generate_report":
            item.add_marker(pytest.mark.report)


@pytest.fixture(scope="session")
def reports_root():
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    return REPORTS_ROOT


def get_report_dir(category: str) -> Path:
    d = REPORTS_ROOT / category
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_figure(fig: plt.Figure, name: str, category: str) -> Path:
    d = get_report_dir(category)
    path = d / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def write_report(content: str, category: str) -> Path:
    d = get_report_dir(category)
    path = d / "report.md"
    path.write_text(content, encoding="utf-8")
    return path


def report_header(title: str, n_pass: int, n_total: int) -> str:
    status = "PASS" if n_pass == n_total else "FAIL"
    return (
        f"# {title}\n\n"
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}  "
        f"**Status**: {status} ({n_pass}/{n_total})\n\n"
    )


# ── Shared env fixtures ──

@pytest.fixture(scope="module")
def case5():
    return create_case5()


@pytest.fixture(scope="module")
def trans_env():
    return TransGridEnv()


@pytest.fixture(scope="module")
def trans_params_with_profiles(case5):
    loader = DataLoader()
    load_profile = loader.load_jax_profiles(
        [S.LOAD_ACTUAL_MW], source="gb",
        start_date="2024-06-01", end_date="2024-06-07",
    )
    load_scale = load_profile[:, 0:1] / load_profile[:, 0:1].max()
    load_rated = (case5.load_d_max + case5.load_d_min) / 2.0
    profiles = load_scale * load_rated[None, :]
    T = profiles.shape[0]
    return make_trans_params(case5, load_profiles=profiles, max_steps=T)


@pytest.fixture(scope="module")
def trans_params_flat(case5):
    return make_trans_params(case5, max_steps=48)


@pytest.fixture(scope="module")
def battery_env():
    return BatteryEnv()


@pytest.fixture(scope="module")
def battery_params():
    return make_battery_params()


@pytest.fixture(scope="module")
def key():
    return jax.random.PRNGKey(42)
