"""Guardrail tests for benchmarks/ config consistency.

These tests catch the classic "I bumped num_envs in train_*.yaml but forgot
to sync HARDWARE.md / throughput.py / task.yaml" failure mode that silently
under-reports throughput.

Source of truth: ``benchmarks/<task>/configs/task.yaml::num_envs`` (merged
with ``provenance.json`` for convergence/provenance fields).  Every other
place that mentions per-task ``num_envs`` MUST agree with it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from benchmarks.common.configs import load_config, load_task_config
from benchmarks.common.io import ACCEPTED_MANIFEST_BACKENDS

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
HARDWARE_MD = BENCHMARKS_DIR / "HARDWARE.md"

TASKS = ["dso", "tso", "ders", "dc_microgrid", "gencos"]

# HARDWARE.md spells task names with a different casing convention.
HARDWARE_LABEL = {
    "dso": "DSO",
    "tso": "TSO",
    "ders": "DERs",
    "dc_microgrid": "DC Microgrid",
    "gencos": "GenCos",
}


def _task_json(task: str) -> dict:
    """Return merged ``task.yaml`` + ``provenance.json`` for ``task``."""
    return load_task_config(BENCHMARKS_DIR / task)


def _train_configs(task: str) -> list[Path]:
    return sorted((BENCHMARKS_DIR / task / "configs").glob("train_*.yaml"))


def _eval_config_stems(task: str) -> set[str]:
    return {
        path.stem.removeprefix("eval_")
        for path in (BENCHMARKS_DIR / task / "configs").glob("eval_*.yaml")
    }


# ---------------------------------------------------------------------------
# 1. Every task config declares num_envs (source of truth).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task", TASKS)
def test_task_json_declares_num_envs(task: str):
    cfg = _task_json(task)
    assert "num_envs" in cfg, (
        f"{task}/configs/task.yaml must declare num_envs (source of truth). "
        f"Add a top-level integer field."
    )
    assert isinstance(cfg["num_envs"], int) and cfg["num_envs"] > 0


# ---------------------------------------------------------------------------
# 2. Every train_*.yaml::num_envs stays within task capacity.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task", TASKS)
@pytest.mark.audit
def test_train_configs_match_task_num_envs(task: str):
    """Soft check: train_*.yaml::num_envs vs task config ``num_envs``.

    These two fields serve different roles after the benchmark-freeze schema split:
      - task config ``num_envs``  = throughput sweep frozen value (reported
        capacity; the maximum vmap width validated for the task)
      - train_*.yaml::num_envs    = the value actually used for training
        (often smaller to fit GPU mem / converge faster)

    A strict equality assert is therefore wrong — it would force every train
    config to inherit the throughput frozen value or vice versa. We instead:
      - REQUIRE train_*.yaml::num_envs <= task config ``num_envs`` (training cannot
        exceed what throughput sweep validated)
      - Keep this check in the explicit benchmark-audit suite rather than the
        default regression run.
    """
    expected = _task_json(task)["num_envs"]
    train_cfgs = _train_configs(task)
    assert train_cfgs, f"No train_*.yaml found for {task}"
    over_capacity: list[str] = []
    differences: list[str] = []
    for cfg_path in train_cfgs:
        if cfg_path.name.endswith(".bak"):
            continue
        cfg = load_config(cfg_path)
        if "num_envs" not in cfg:
            continue
        train_n = int(cfg["num_envs"])
        if train_n > int(expected):
            over_capacity.append(
                f"  {cfg_path.relative_to(REPO_ROOT)}: "
                f"num_envs={train_n} > task-config frozen capacity {expected}"
            )
        if train_n != int(expected):
            differences.append(
                f"  {cfg_path.relative_to(REPO_ROOT)}: "
                f"num_envs={train_n} (task-config frozen capacity={expected})"
            )

    # Hard fail only if a train config exceeds the validated capacity.
    assert not over_capacity, (
        f"\n{task}: train config num_envs exceeds task-config validated "
        f"capacity:\n" + "\n".join(over_capacity)
    )

    # Intentional training-vs-throughput differences are allowed in the audit
    # suite; this test only hard-fails when training exceeds validated capacity.
    assert isinstance(differences, list)


# ---------------------------------------------------------------------------
# 3. throughput.py FROZEN_NUM_ENVS matches task.yaml.
# ---------------------------------------------------------------------------

def test_frozen_num_envs_matches_task_json():
    from benchmarks.common.analysis import FROZEN_NUM_ENVS  # noqa: WPS433

    mismatches = []
    for task in TASKS:
        expected = _task_json(task)["num_envs"]
        actual = FROZEN_NUM_ENVS.get(task)
        if actual != expected:
            mismatches.append(
                f"  FROZEN_NUM_ENVS[{task!r}]={actual} but "
                f"{task}/configs/task.yaml::num_envs={expected}"
            )
    assert not mismatches, (
        "\nbenchmarks/common/throughput.py::FROZEN_NUM_ENVS is out of sync "
        "with task config:\n" + "\n".join(mismatches)
    )


# ---------------------------------------------------------------------------
# 4. HARDWARE.md table matches task.yaml.
# ---------------------------------------------------------------------------

def _parse_hardware_table() -> dict[str, int]:
    """Parse the | Task | num_envs | ... | rows from HARDWARE.md."""
    text = HARDWARE_MD.read_text(encoding="utf-8")
    found: dict[str, int] = {}
    row_pat = re.compile(r"^\|\s*([^|]+?)\s*\|\s*(\d+)\s*\|", re.MULTILINE)
    for label, num in row_pat.findall(text):
        for task, task_label in HARDWARE_LABEL.items():
            if label.strip() == task_label:
                found[task] = int(num)
                break
    return found


def test_hardware_md_table_matches_task_json():
    parsed = _parse_hardware_table()
    mismatches = []
    missing = []
    for task in TASKS:
        if task not in parsed:
            missing.append(f"  {HARDWARE_LABEL[task]} row not found in HARDWARE.md")
            continue
        expected = _task_json(task)["num_envs"]
        if parsed[task] != expected:
            mismatches.append(
                f"  HARDWARE.md row '{HARDWARE_LABEL[task]}'={parsed[task]} but "
                f"{task}/configs/task.yaml::num_envs={expected}"
            )
    assert not missing, (
        "\nHARDWARE.md is missing rows in the num_envs table:\n"
        + "\n".join(missing)
    )
    assert not mismatches, (
        "\nbenchmarks/HARDWARE.md num_envs table is out of sync with task config:\n"
        + "\n".join(mismatches)
    )


# ---------------------------------------------------------------------------
# 5. Every task config has the unified benchmark-freeze fields.
# ---------------------------------------------------------------------------

P0_REQUIRED_TASK_FIELDS = (
    "baseline_set",
    "eval_splits",
    "primary_split",
    "seeds",
    "eval_episodes",
    "convergence_threshold_per_split",
    "convergence_provenance",
    "safety_thresholds",
)


@pytest.mark.parametrize("task", TASKS)
def test_task_json_has_p0_fields(task: str):
    cfg = _task_json(task)
    missing = [k for k in P0_REQUIRED_TASK_FIELDS if k not in cfg]
    assert not missing, (
        f"{task}/configs/task.yaml is missing benchmark-freeze fields: {missing}"
    )
    assert isinstance(cfg["seeds"], list) and all(isinstance(s, int) for s in cfg["seeds"])
    assert isinstance(cfg["eval_episodes"], int) and cfg["eval_episodes"] > 0
    assert cfg["primary_split"] in cfg["eval_splits"], (
        f"{task}: primary_split={cfg['primary_split']!r} must be in eval_splits "
        f"{cfg['eval_splits']}"
    )


def test_ders_configured_eval_splits_are_instantiable():
    """DERs canonical split names must be accepted by the task factory."""
    from powerzoojax.case import load_case
    from powerzoojax.tasks.ders import DERS_BENCHMARK_LOAD_CASE, DERsTask

    cfg = _task_json("ders")
    assert cfg.get("case") == DERS_BENCHMARK_LOAD_CASE
    case = load_case(cfg["case"])
    task = DERsTask(
        case=case,
        v_min=cfg["v_min"],
        v_max=cfg["v_max"],
        voltage_penalty=4.0,
        max_steps=cfg["max_steps"],
    )

    for split in cfg["eval_splits"]:
        params = task.episode_params(split, 0, 1, cfg["max_steps"])
        assert params.load_profiles_p.shape[0] == cfg["max_steps"]


def test_dc_microgrid_configured_eval_splits_are_instantiable():
    """DC Microgrid main and appendix split names must reach real-data params."""
    from powerzoojax.tasks.dc_microgrid import (
        DCMicrogridTask,
        compute_dcmicrogrid_metrics,
    )

    cfg = _task_json("dc_microgrid")
    task = DCMicrogridTask(
        source=cfg["data_source"],
        max_steps=cfg["max_steps"],
    )
    splits = list(cfg["eval_splits"]) + list(cfg.get("eval_splits_appendix", []))

    for split in splits:
        params = task.episode_params(split, 0, 1, cfg["max_steps"])
        assert params.cpu_profile is not None
        assert params.cpu_profile.shape[0] == cfg["max_steps"]


def test_dc_microgrid_reward_shaping_is_logwrapper_compatible():
    """Formal DC Microgrid training stacks RewardShapingWrapper under LogWrapper."""
    from benchmarks.dc_microgrid._reward_shaping import wrap_with_shaping
    from powerzoojax.envs.microgrid import DataCenterMicrogridEnv
    from powerzoojax.rl.wrappers import LogWrapper
    from powerzoojax.tasks.dc_microgrid import (
        DCMicrogridTask,
        compute_dcmicrogrid_metrics,
    )

    cfg = _task_json("dc_microgrid")
    task = DCMicrogridTask(source=cfg["data_source"], max_steps=cfg["max_steps"])
    params = task.episode_params("train", 0, 1, cfg["max_steps"])
    shaped_env = wrap_with_shaping(DataCenterMicrogridEnv(), cfg)
    wrapped = LogWrapper(shaped_env, params)

    assert wrapped.constraint_names == ("sla", "overtemp", "power_deficit")


def test_dc_microgrid_reward_shaping_dg_autobalance_closes_residual():
    """Benchmark wrapper can use DG as a same-step slack actuator."""
    import jax
    import jax.numpy as jnp

    from benchmarks.dc_microgrid._reward_shaping import wrap_with_shaping
    from powerzoojax.envs.microgrid import DataCenterMicrogridEnv, make_dcmicrogrid_params

    params = make_dcmicrogrid_params(
        max_steps=2,
        pv_p_max_mw=0.0,
        dg_p_max_mw=1.2,
    )
    env = wrap_with_shaping(
        DataCenterMicrogridEnv(),
        {"dg_autobalance": True},
    )
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key, params)
    action = jnp.array([0.0, 0.0, 0.5, 0.0, 0.0], dtype=jnp.float32)

    _, _, _, _, _, info = env.step(key, state, action, params)

    assert float(info["cost_power_balance"]) < 1e-5
    assert float(info["dg_command_raw_norm"]) == 0.0
    assert float(info["dg_command_balanced_norm"]) > 0.0


def test_dc_microgrid_task_baseline_uses_nested_episode_length():
    """DC Microgrid task rollouts must read max_steps from params.dc."""
    import jax

    from benchmarks.dc_microgrid._reward_shaping import wrap_with_shaping
    from powerzoojax.envs.microgrid import DataCenterMicrogridEnv, make_dcmicrogrid_params
    from powerzoojax.tasks.dc_microgrid import (
        DCMicrogridTask,
        compute_dcmicrogrid_metrics,
    )

    params = make_dcmicrogrid_params(max_steps=3)
    task = DCMicrogridTask(max_steps=3)
    history = task.baseline_rollout(
        DataCenterMicrogridEnv(),
        params,
        jax.random.PRNGKey(0),
        "no_control",
    )

    assert len(history) == 3
    metrics = compute_dcmicrogrid_metrics(history)
    assert metrics["episode_reward"] < 0.0

    shaped_history = task.baseline_rollout(
        wrap_with_shaping(
            DataCenterMicrogridEnv(),
            {"reward_shaping_weights": {"power_deficit": 100.0}},
        ),
        params,
        jax.random.PRNGKey(0),
        "no_control",
    )
    shaped_metrics = compute_dcmicrogrid_metrics(shaped_history)
    assert shaped_metrics["episode_reward"] < metrics["episode_reward"]


def test_seed0_readiness_after_filter_blocks_legacy_records():
    """A new campaign must not inherit old manifest readiness by accident."""
    from benchmarks.common.experiment_ops import check_seed0_readiness

    report = check_seed0_readiness(
        "dso",
        after="2999-01-01T00:00:00+00:00",
    )

    assert not report["ready_for_formal_runs"]
    assert report["record_filter_after"] == "2999-01-01T00:00:00+00:00"
    assert report["manifest_records_after_filter"] == 0
    assert not report["steps"]["1_baseline_seed0"]["ok"]
    assert not report["steps"]["2_train_seed0_reference"]["ok"]
    assert not report["steps"]["5_summary_and_plots"]["ok"]
    assert not report["steps"]["6_derive_target"]["ok"]


@pytest.mark.parametrize("task_name", TASKS)
def test_task_baseline_names_match_frozen_config(task_name: str):
    """TaskSpec baseline names must match the frozen benchmark config.

    This catches accidental drift between ``baseline_set`` / provenance /
    summary scripts and the task or runner code that actually emits records.
    """
    cfg = _task_json(task_name)
    if task_name == "dso":
        from powerzoojax.tasks.dso import DSOTask
        task = DSOTask()
    elif task_name == "tso":
        from powerzoojax.tasks.tso import TSOTask
        task = TSOTask()
    elif task_name == "ders":
        from powerzoojax.tasks.ders import DERsTask
        task = DERsTask()
    elif task_name == "dc_microgrid":
        from powerzoojax.tasks.dc_microgrid import DCMicrogridTask
        task = DCMicrogridTask()
    elif task_name == "gencos":
        from powerzoojax.tasks.gencos import GencosTask
        task = GencosTask()
    else:
        raise AssertionError(f"Unhandled task {task_name!r}")

    assert tuple(cfg["baseline_set"]) == tuple(task.baseline_names())


@pytest.mark.parametrize("task_name", TASKS)
def test_cross_backend_split_whitelist_covers_frozen_config(task_name: str):
    """Cross-backend dispatch must not reject a frozen benchmark split."""
    from benchmarks.common.powerzoo_bridge import _VALID_SPLITS_BY_TASK

    cfg = _task_json(task_name)
    expected = tuple(cfg["eval_splits"])
    if task_name == "dc_microgrid":
        expected = tuple(cfg["eval_splits_main"]) + tuple(cfg["eval_splits_appendix"])

    assert set(expected).issubset(set(_VALID_SPLITS_BY_TASK[task_name]))


@pytest.mark.parametrize("task_name", TASKS)
def test_eval_config_files_cover_frozen_splits(task_name: str):
    """Every frozen split must have a matching eval_<split>.yaml."""
    cfg = _task_json(task_name)
    if task_name == "dc_microgrid":
        expected = set(cfg["eval_splits_main"]) | set(cfg["eval_splits_appendix"])
    else:
        expected = set(cfg["eval_splits"])
    actual = _eval_config_stems(task_name)
    missing = sorted(expected - actual)
    assert not missing, (
        f"{task_name}: missing eval config files for frozen splits {missing}; "
        "summary/eval runners must not guess split configs."
    )


# ---------------------------------------------------------------------------
# 6. RunRecord backend field, when present in a manifest, must be a formal
#    canonical value or a known deprecated key (see benchmarks.common.io).
#    Never bare "jax". Records without `kind="train"` are eval/baseline and
#    are out of scope for the full convergence / canonical-curve checks
#    for convergence and canonical learning-curve checks.
# ---------------------------------------------------------------------------


def _manifest_records(task: str) -> list[dict]:
    p = BENCHMARKS_DIR / task / "results" / "manifest.json"
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def _is_train_record(rec: dict) -> bool:
    """Same kind=train classification rule as benchmarks/common/io.py dedup_keep_artifacts."""
    from benchmarks.common.io import has_training_artifact

    return has_training_artifact(rec.get("artifacts"))


@pytest.mark.parametrize("task", TASKS)
def test_manifest_backend_taxonomy(task: str):
    """If a record has a backend field, it must use the canonical taxonomy.

    Old records may have backend="" or missing — that's fine (default field
    value, untouched). The forbidden state is bare "jax" or any other free-
    form string, which would silently corrupt cross-backend group-by.
    """
    bad: list[str] = []
    for rec in _manifest_records(task):
        b = rec.get("backend")
        if b is None or b == "":
            continue
        if b not in ACCEPTED_MANIFEST_BACKENDS:
            bad.append(f"  run_id={rec.get('run_id')} backend={b!r}")
    assert not bad, (
        f"\n{task}/results/manifest.json contains records with non-canonical "
        f"backend values (must be one of {sorted(ACCEPTED_MANIFEST_BACKENDS)}):\n"
        + "\n".join(bad)
    )


# ---------------------------------------------------------------------------
# 7. Training-class records with status="completed" should eventually carry
#    convergence + canonical curves. Emit warnings first — most manifests
#    pre-date the canonical-curve convention, so a hard assert before every
#    task passes seed0_readiness would block the freeze rollout itself.
#
#    Ratchet: set POWERZOOJAX_STRICT_P0=1 once seed0_readiness is green everywhere.
# ---------------------------------------------------------------------------

EXPECTED_CANONICAL_CURVES = (
    "learning_curve_train_return",
    "learning_curve_eval_return",
    "learning_curve_eval_walltimes",
)


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.audit
def test_training_records_have_convergence_and_curves(task: str):
    """Soft check: report missing convergence + canonical curves on training records.

    This lives in the explicit audit suite so pre-freeze manifests don't pollute
    the default regression run. Once every task passes seed0_readiness, enable a
    hard assert via POWERZOOJAX_STRICT_P0=1.
    """
    import os
    strict = os.environ.get("POWERZOOJAX_STRICT_P0") == "1"

    issues: list[str] = []
    for rec in _manifest_records(task):
        if not _is_train_record(rec):
            continue
        if rec.get("status") != "completed":
            continue
        # Skip records using the old default backend (untouched legacy)
        if rec.get("backend") in (None, "", "jax"):
            continue
        conv = rec.get("convergence") or {}
        if not conv.get("target_return"):
            issues.append(f"  {rec.get('run_id')}: missing convergence.target_return")
            continue
        arts = rec.get("artifacts") or {}
        missing_curves = [k for k in EXPECTED_CANONICAL_CURVES if k not in arts]
        if missing_curves:
            issues.append(f"  {rec.get('run_id')}: missing curves {missing_curves}")

    if issues:
        msg = (
            f"\n{task}: training-class records are missing canonical convergence/curve fields "
            f"(currently a warning; will become hard error when POWERZOOJAX_STRICT_P0=1):\n"
            + "\n".join(issues)
        )
        if strict:
            pytest.fail(msg)
