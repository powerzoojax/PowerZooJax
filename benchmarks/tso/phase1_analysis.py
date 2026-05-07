"""Phase-1 TSO diagnostics for an isolated seed-0 campaign.

Generates a richer, episode-level figure suite for a completed Phase-1 run:

* ``results/analysis_episode_summary.json``
* ``results/figures/phase1_training_diagnostics.{pdf,png}``
* ``results/figures/phase1_policy_compare.{pdf,png}``
* ``results/figures/phase1_episode_mechanism.{pdf,png}``
* ``results/figures/phase1_line_rating_sensitivity.{pdf,png}``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.common.configs import (
    load_config,
    load_task_config,
    load_task_config_for_run,
    load_train_config_for_run,
)
from benchmarks.common.eval_loop import run_episodes
from benchmarks.common.io import RunRecord, load_manifest, load_manifest_filtered, load_pickle
from benchmarks.common.runtime import build_train_cfg, make_policy_fn
from benchmarks.dc_microgrid.rejax_ckpt import load_sac_train_state
from benchmarks.tso.config_runtime import get_eval_episodes, make_task_from_config
from powerzoojax.envs.base import denormalize_action
from powerzoojax.envs.grid.dc_opf import dc_opf
from powerzoojax.envs.grid.power_flow import (
    compute_generation_cost,
    dc_power_flow,
    safety_check,
)
from powerzoojax.envs.grid.unit_commitment import UCState, UnitCommitmentEnv
from powerzoojax.tasks.tso import (
    _TSO_SYNTHETIC_LINE_LIMIT_DEGREES,
    compute_tso_metrics,
    rollout_tso,
)

DEFAULT_TASK_DIR = Path(__file__).resolve().parent
LEARNED_ALGOS = ("ppo", "sac", "ppo_lagrangian")
POLICY_ORDER = ("all_on", "merit_order", "ppo", "sac", "ppo_lagrangian")
POLICY_COLORS = {
    "all_on": "#4e79a7",
    "merit_order": "#59a14f",
    "ppo": "#e15759",
    "sac": "#b07aa1",
    "ppo_lagrangian": "#f28e2b",
}
PLOT_FACE = "#f7f3eb"
AX_FACE = "#fffdf8"
GRID_COLOR = "#d7cfbf"
DAY_BAND = "#efe7d7"


def _figures_dir(task_dir: Path) -> Path:
    return task_dir / "results" / "figures"


def _save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(
        path.with_suffix(".png"),
        dpi=150,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )


def _style_axes(ax) -> None:
    ax.set_facecolor(AX_FACE)
    ax.grid(True, color=GRID_COLOR, alpha=0.45, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _decorate_time_axis(ax, n_steps: int) -> None:
    for start in range(0, n_steps, 12):
        if (start // 12) % 2 == 1:
            ax.axvspan(
                start - 0.5,
                min(start + 11.5, n_steps - 0.5),
                color=DAY_BAND,
                alpha=0.35,
            )
    ax.set_xlim(0, n_steps - 1)


def _rolling_mean(arr: np.ndarray, window: int = 7) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size < 3:
        return arr
    window = min(window, arr.size if arr.size % 2 == 1 else arr.size - 1)
    if window < 3:
        return arr
    pad = window // 2
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.pad(arr, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _load_records(task_dir: Path, after: str | None) -> list[RunRecord]:
    if after:
        return load_manifest_filtered(task_dir, after=after)
    return load_manifest(task_dir)


def _latest_train_records(records: list[RunRecord]) -> dict[str, RunRecord]:
    latest: dict[str, RunRecord] = {}
    for record in records:
        if record.status != "completed" or record.split != "train":
            continue
        if record.algo not in LEARNED_ALGOS:
            continue
        arts = record.artifacts or {}
        if "params" not in arts and "params_orbax" not in arts:
            continue
        cur = latest.get(record.algo)
        if cur is None or (record.timestamp, record.run_id) > (cur.timestamp, cur.run_id):
            latest[record.algo] = record
    return latest


def _linked_eval_record(
    records: list[RunRecord],
    *,
    train_run_id: str,
    split: str,
) -> RunRecord:
    matches = [
        record
        for record in records
        if record.status == "completed"
        and record.split == split
        and f"eval of {train_run_id}" in (record.notes or "")
    ]
    if not matches:
        raise FileNotFoundError(
            f"No completed eval record found for train_run_id={train_run_id!r}, split={split!r}."
        )
    matches.sort(key=lambda record: (record.timestamp, record.run_id))
    return matches[-1]


def _load_per_episode_rows(task_dir: Path, record: RunRecord) -> list[dict[str, Any]]:
    rel = (record.artifacts or {}).get("per_episode")
    if not rel:
        raise FileNotFoundError(
            f"Run {record.run_id!r} has no per_episode artifact; cannot do Phase-1 analysis."
        )
    path = task_dir / "results" / rel
    rows = json.loads(path.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        enriched = dict(row)
        enriched["episode_idx"] = idx
        out.append(enriched)
    return out


def _select_representative_iid_episode(rows: list[dict[str, Any]]) -> dict[str, Any]:
    safe = [
        row
        for row in rows
        if float(row.get("total_reserve_shortfall", 0.0)) <= 1e-6
        and float(row.get("total_thermal_cost", 0.0)) <= 1e-6
    ]
    if safe:
        median_cost = float(np.median([float(row["total_operating_cost"]) for row in safe]))
        return min(
            safe,
            key=lambda row: abs(float(row["total_operating_cost"]) - median_cost),
        )
    median_cost = float(np.median([float(row["total_operating_cost"]) for row in rows]))
    return min(
        rows,
        key=lambda row: (
            float(row.get("total_cost_violations", np.inf)),
            abs(float(row["total_operating_cost"]) - median_cost),
        ),
    )


def _select_worst_episode(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            float(row.get("total_thermal_cost", 0.0)),
            float(row.get("total_cost_violations", 0.0)),
            float(row.get("total_reserve_shortfall", 0.0)),
        ),
    )


def _latest_eval_metric(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row.get(key, np.nan)) for row in rows]
    return float(np.nanmean(np.asarray(values, dtype=np.float64)))


def _load_curve_artifacts(task_dir: Path, train_record: RunRecord) -> dict[str, np.ndarray]:
    arts_dir = task_dir / "results" / "artifacts"
    run_id = train_record.run_id

    def _maybe(name: str) -> np.ndarray | None:
        path = arts_dir / f"{run_id}_{name}.npy"
        if path.exists():
            return np.load(path)
        return None

    eval_steps = _maybe("eval_timesteps")
    if eval_steps is None:
        eval_steps = _maybe("timesteps")
    eval_cost = _maybe("eval_total_operating_cost")
    eval_thermal_total = _maybe("eval_total_thermal_overload")
    eval_reserve = _maybe("eval_reserve_shortfall_rate")
    eval_thermal = _maybe("eval_thermal_violation_rate")
    eval_return = _maybe("eval_returns")
    if eval_return is None:
        eval_return = _maybe("learning_curve_eval_return")

    out = {
        "steps": np.asarray(eval_steps if eval_steps is not None else [], dtype=np.float64),
        "cost": np.asarray(eval_cost if eval_cost is not None else [], dtype=np.float64),
        "thermal_total": np.asarray(
            eval_thermal_total if eval_thermal_total is not None else [],
            dtype=np.float64,
        ),
        "reserve_rate": np.asarray(eval_reserve if eval_reserve is not None else [], dtype=np.float64),
        "thermal_rate": np.asarray(eval_thermal if eval_thermal is not None else [], dtype=np.float64),
        "eval_return": np.asarray(eval_return if eval_return is not None else [], dtype=np.float64),
    }

    if train_record.algo == "ppo_lagrangian":
        for key in (
            "lambda_total",
            "lambda_reserve_shortfall",
            "lambda_thermal_overload",
            "mean_cost_total",
            "mean_reward",
        ):
            arr = _maybe(key)
            if arr is not None:
                out[key] = np.asarray(arr, dtype=np.float64)
    return out


def _base_params_for_policy(task_dir: Path, train_record: RunRecord | None = None):
    task_cfg = (
        load_task_config_for_run(task_dir, train_record)
        if train_record is not None
        else load_task_config(task_dir)
    )
    task = make_task_from_config(task_cfg)
    max_steps = int(task_cfg.get("max_steps", 48))
    n_eval = get_eval_episodes(task_cfg)
    params = task.episode_params("iid", 0, n_eval, max_steps, strategy="uniform", seed=0)
    return task_cfg, task, params


def _load_learned_policy(
    task_dir: Path,
    train_record: RunRecord,
    env: UnitCommitmentEnv,
    base_params,
    selected_names: tuple[str, ...],
) -> Callable:
    cfg_dict = load_train_config_for_run(
        task_dir,
        train_record,
        algo_key_map={"ppo_lagrangian": "safe", "saute_ppo": "saute_ppo"},
        default_key=train_record.algo,
    )
    train_cfg = build_train_cfg(cfg_dict, algo=train_record.algo)
    if train_record.algo == "sac":
        rel = train_record.artifacts["params_orbax"]
        params_obj = load_sac_train_state(
            task_dir / "results" / rel,
            train_cfg,
            env,
            base_params,
        )
    else:
        rel = train_record.artifacts["params"]
        params_obj = load_pickle(task_dir / "results" / rel)

    policy = make_policy_fn(
        train_record.algo,
        params_obj,
        env,
        base_params,
        train_cfg,
        action_dim=2 * base_params.case.n_units,
        selected_names=selected_names,
    )

    return policy


def _all_on_policy(obs, state, key, params):
    del obs, state, key
    n_units = int(params.case.n_units)
    return jnp.concatenate(
        [
            jnp.ones(n_units, dtype=jnp.float32),
            jnp.zeros(n_units, dtype=jnp.float32),
        ]
    )


def _merit_order_policy(obs, state, key, params):
    del obs, key
    case = params.case
    n_units = int(case.n_units)
    load_profiles_np = np.asarray(params.load_profiles)
    nodes_loads_map_np = np.asarray(case.nodes_loads_map)
    p_max_np = np.asarray(case.unit_p_max)
    merit_order = np.argsort(np.asarray(case.unit_cost_b))
    t_idx = int(state.time_step) % load_profiles_np.shape[0]
    load_demand = load_profiles_np[t_idx]
    total_load = float(np.sum(nodes_loads_map_np @ load_demand))
    required_cap = total_load * (1.0 + float(params.reserve_margin_frac))
    commit = np.zeros(n_units, dtype=np.float32)
    cumcap = 0.0
    for idx in merit_order:
        commit[idx] = 1.0
        cumcap += p_max_np[idx]
        if cumcap >= required_cap:
            break
    return jnp.array(
        np.concatenate([commit * 2.0 - 1.0, np.zeros(n_units, dtype=np.float32)]),
        dtype=jnp.float32,
    )


def _simulate_step(env: UnitCommitmentEnv, state: UCState, action: jax.Array, params):
    case = params.case
    n_units = int(case.n_units)

    commit_signal = action[:n_units]
    dispatch_signal = action[n_units:]

    if params.enable_uc:
        raw_commit = jnp.where(commit_signal > jnp.float32(0.0), jnp.int32(1), jnp.int32(0))
        must_stay_on = jnp.logical_and(
            state.unit_status == 1,
            state.time_in_state < params.min_up_steps,
        )
        must_stay_off = jnp.logical_and(
            state.unit_status == 0,
            state.time_in_state < params.min_down_steps,
        )
        actual_commit = jnp.where(must_stay_on, jnp.int32(1), raw_commit)
        actual_commit = jnp.where(must_stay_off, jnp.int32(0), actual_commit)
    else:
        actual_commit = jnp.ones_like(state.unit_status)

    commit_float = actual_commit.astype(jnp.float32)
    switched_on = jnp.logical_and(actual_commit == 1, state.unit_status == 0)
    status_changed = (actual_commit != state.unit_status).astype(jnp.int32)
    commitment_switches = jnp.sum(status_changed).astype(jnp.float32)

    startup_cost_step = jnp.sum(switched_on.astype(jnp.float32) * params.startup_cost)
    no_load_cost_step = jnp.sum(commit_float * params.no_load_cost_per_step)

    ramp_p_min = jnp.maximum(case.unit_p_min, state.last_dispatch - params.ramp_down_mw)
    ramp_p_max = jnp.minimum(case.unit_p_max, state.last_dispatch + params.ramp_up_mw)
    eff_p_min = ramp_p_min * commit_float
    eff_p_max = ramp_p_max * commit_float

    t_idx = state.time_step % params.load_profiles.shape[0]
    load_demand = params.load_profiles[t_idx]
    node_load_mw = case.nodes_loads_map @ load_demand

    if params.dcopf_setup is not None:
        setup_ramp = params.dcopf_setup.replace(p_min=eff_p_min, p_max=eff_p_max)
        target_dispatch_mw = denormalize_action(dispatch_signal, eff_p_min, eff_p_max)
        mc_c_biased = setup_ramp.mc_c - jnp.float32(params.dispatch_preference_weight) * target_dispatch_mw
        result = dc_opf(setup_ramp.replace(mc_c=mc_c_biased), node_load_mw)
        unit_power_mw = result.unit_power
        line_flow_mw = result.line_flow
        node_inj = result.node_injection
    else:
        target_dispatch = denormalize_action(dispatch_signal, case.unit_p_min, case.unit_p_max)
        clipped_dispatch = jnp.clip(target_dispatch, eff_p_min, eff_p_max)
        line_flow_mw, node_inj, unit_power_mw = dc_power_flow(case, clipped_dispatch, node_load_mw)
        unit_power_mw = unit_power_mw * commit_float

    is_thermal_safe, n_violations, cost_thermal = safety_check(
        line_flow_mw, case.line_cap, case.line_floor
    )

    total_load = jnp.sum(node_load_mw)
    required_capacity = total_load * (jnp.float32(1.0) + params.reserve_margin_frac)
    committed_capacity = jnp.sum(commit_float * case.unit_p_max)
    reserve_shortfall = jnp.maximum(jnp.float32(0.0), required_capacity - committed_capacity)
    reserve_violation = jnp.logical_and(
        jnp.bool_(params.enable_reserve),
        reserve_shortfall > jnp.float32(1e-6),
    )
    is_safe = jnp.logical_and(is_thermal_safe, jnp.logical_not(reserve_violation))
    n_violations = n_violations + reserve_violation.astype(jnp.int32)

    cost_reserve = reserve_shortfall if params.enable_reserve else jnp.float32(0.0)
    gen_cost = compute_generation_cost(
        unit_power_mw,
        case.unit_cost_a,
        case.unit_cost_b,
        case.unit_cost_c,
    )
    step_operating_cost = gen_cost + startup_cost_step + no_load_cost_step
    cost_thermal_w = cost_thermal * params.cost_thermal_weight
    cost_sum = cost_thermal_w + cost_reserve

    new_time = state.time_step + 1
    next_state = UCState(
        time_step=new_time,
        done=new_time >= params.max_steps,
        unit_power_mw=unit_power_mw,
        line_flow_mw=line_flow_mw,
        node_injection_mw=node_inj,
        is_safe=is_safe,
        n_violations=n_violations,
        total_cost=state.total_cost + gen_cost,
        vm=jnp.ones_like(node_inj),
        va=jnp.zeros_like(node_inj),
        q_gen=jnp.zeros_like(unit_power_mw),
        line_flow_q_mw=jnp.zeros_like(line_flow_mw),
        resource_states=(),
        unit_status=actual_commit,
        time_in_state=jnp.where(
            status_changed,
            jnp.int32(1),
            state.time_in_state + jnp.int32(1),
        ),
        last_dispatch=unit_power_mw,
        startup_cost_accum=state.startup_cost_accum + startup_cost_step,
    )
    next_obs = env._get_obs(next_state, params)

    safe_cap = jnp.maximum(jnp.abs(case.line_cap), jnp.float32(1.0))
    line_loading_ratio = jnp.abs(line_flow_mw) / safe_cap

    return {
        "next_state": next_state,
        "next_obs": next_obs,
        "gen_cost": gen_cost,
        "startup_cost": startup_cost_step,
        "no_load_cost": no_load_cost_step,
        "reserve_shortfall": reserve_shortfall,
        "cost_thermal_overload": cost_thermal_w,
        "cost_sum": cost_sum,
        "commitment_switches": commitment_switches,
        "is_safe": is_safe,
        "total_load": total_load,
        "required_capacity": required_capacity,
        "committed_capacity": committed_capacity,
        "available_headroom": committed_capacity - total_load,
        "required_reserve": total_load * params.reserve_margin_frac,
        "n_units_on": jnp.sum(actual_commit).astype(jnp.float32),
        "unit_status": actual_commit.astype(jnp.float32),
        "unit_power_mw": unit_power_mw,
        "line_flow_mw": line_flow_mw,
        "max_line_loading_pct": jnp.max(line_loading_ratio) * jnp.float32(100.0),
        "p95_line_loading_pct": jnp.percentile(line_loading_ratio, 95.0) * jnp.float32(100.0),
        "n_overloaded_lines": jnp.sum(line_loading_ratio > jnp.float32(1.0)).astype(jnp.float32),
        "step_operating_cost": step_operating_cost,
        "dispatch_signal_mean": jnp.mean(dispatch_signal),
        "commit_signal_mean": jnp.mean(commit_signal),
    }


def _stack_series(steps: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in steps[0]:
        if key in ("next_state", "next_obs"):
            continue
        out[key] = np.asarray([np.asarray(step[key]) for step in steps])
    out["summary"] = compute_tso_metrics(out)
    return out


def _rollout_episode_diagnostics(
    env: UnitCommitmentEnv,
    params,
    key: jax.Array,
    policy_fn: Callable,
) -> dict[str, Any]:
    obs, state = env.reset(key, params)
    step_details: list[dict[str, Any]] = []
    carry_key = key
    for _ in range(int(params.max_steps)):
        carry_key, _k_step, k_pol = jax.random.split(carry_key, 3)
        action = policy_fn(obs, state, k_pol, params)
        detail = _simulate_step(env, state, action, params)
        detail["action"] = np.asarray(action)
        step_details.append(detail)
        obs = detail["next_obs"]
        state = detail["next_state"]
    return _stack_series(step_details)


def _validate_replay(
    replay_summary: dict[str, Any],
    expected: dict[str, Any],
    *,
    label: str,
    atol: float = 1e-3,
    rtol: float = 1e-6,
) -> None:
    for key in (
        "total_operating_cost",
        "total_reserve_shortfall",
        "total_thermal_cost",
        "feasibility_rate",
    ):
        lhs = float(replay_summary[key])
        rhs = float(expected[key])
        if not np.isclose(lhs, rhs, atol=atol, rtol=rtol):
            raise AssertionError(
                f"{label}: replay mismatch for {key}: replay={lhs}, expected={rhs}, "
                f"atol={atol}, rtol={rtol}"
            )


def _episode_params_for(
    task_dir: Path,
    split: str,
    episode_idx: int,
    train_record: RunRecord | None = None,
):
    task_cfg = (
        load_task_config_for_run(task_dir, train_record)
        if train_record is not None
        else load_task_config(task_dir)
    )
    eval_cfg_path = task_dir / "configs" / f"eval_{split}.yaml"
    eval_cfg = load_config(eval_cfg_path) if eval_cfg_path.exists() else {}
    task = make_task_from_config(
        task_cfg,
        load_scale=float(eval_cfg.get("load_scale", 1.0)),
        line_rating_scale=float(eval_cfg.get("line_rating_scale", 1.0)),
    )
    max_steps = int(task_cfg.get("max_steps", 48))
    n_eval = get_eval_episodes(task_cfg)
    params = task.episode_params(
        split,
        episode_idx,
        n_eval,
        max_steps,
        strategy="uniform",
        seed=0,
    )
    return task_cfg, task, params


def _policy_metric_table(rollouts: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = {}
    for name, rollout in rollouts.items():
        summary = rollout["summary"]
        table[name] = {
            "total_operating_cost": float(summary["total_operating_cost"]),
            "total_thermal_cost": float(summary["total_thermal_cost"]),
            "total_reserve_shortfall": float(summary["total_reserve_shortfall"]),
            "feasibility_rate": float(summary["feasibility_rate"]),
            "total_commitment_switches": float(summary["total_commitment_switches"]),
        }
    return table


def _plot_training_diagnostics(
    task_dir: Path,
    curves: dict[str, dict[str, np.ndarray]],
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor=PLOT_FACE)
    axs = axes.reshape(-1)
    for ax in axs:
        _style_axes(ax)

    for algo in LEARNED_ALGOS:
        curve = curves.get(algo)
        if not curve:
            continue
        steps = curve.get("steps", np.array([]))
        if steps.size == 0:
            continue
        color = POLICY_COLORS[algo]
        label = algo.replace("_", "-")
        if curve["cost"].size:
            n = min(steps.size, curve["cost"].size)
            axs[0].plot(
                steps[:n],
                _rolling_mean(curve["cost"][:n]),
                color=color,
                linewidth=2.2,
                label=label,
            )
        if curve["thermal_total"].size:
            n = min(steps.size, curve["thermal_total"].size)
            axs[1].plot(
                steps[:n],
                _rolling_mean(curve["thermal_total"][:n]),
                color=color,
                linewidth=2.2,
                label=label,
            )
        if curve["reserve_rate"].size:
            n = min(steps.size, curve["reserve_rate"].size)
            axs[2].plot(
                steps[:n],
                _rolling_mean(curve["reserve_rate"][:n]),
                color=color,
                linewidth=2.2,
                label=label,
            )

    lag_curve = curves.get("ppo_lagrangian", {})
    lag_steps = lag_curve.get("steps", np.array([]))
    if lag_steps.size:
        for key, color, label in (
            ("lambda_thermal_overload", "#f28e2b", "lambda thermal"),
            ("lambda_reserve_shortfall", "#4e79a7", "lambda reserve"),
            ("lambda_total", "#2f2f2f", "lambda total"),
        ):
            arr = lag_curve.get(key)
            if arr is None or arr.size == 0:
                continue
            n = min(lag_steps.size, arr.size)
            axs[3].plot(lag_steps[:n], arr[:n], color=color, linewidth=2.2, label=label)

    axs[0].set_title("Eval Operating Cost")
    axs[0].set_ylabel("USD / episode")
    axs[1].set_title("Eval Total Thermal Overload")
    axs[1].set_ylabel("MW-sum / episode")
    axs[2].set_title("Eval Reserve Shortfall Rate")
    axs[2].set_ylabel("Violation rate")
    axs[3].set_title("PPO-Lagrangian Duals")
    axs[3].set_ylabel("Lambda")

    for ax in axs:
        ax.set_xlabel("Timesteps")
        ax.legend(frameon=False, fontsize=9)

    fig.suptitle("TSO Phase-1 training diagnostics", fontsize=15, y=0.995)
    out = _figures_dir(task_dir) / "phase1_training_diagnostics.pdf"
    _save_figure(fig, out)
    plt.close(fig)
    return out


def _plot_policy_compare(
    task_dir: Path,
    iid_metrics: dict[str, dict[str, float]],
    lt_metrics: dict[str, dict[str, float]],
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor=PLOT_FACE)
    for ax in axes.reshape(-1):
        _style_axes(ax)

    labels = [name.replace("_", "\n") for name in POLICY_ORDER if name in iid_metrics]
    colors = [POLICY_COLORS[name] for name in POLICY_ORDER if name in iid_metrics]
    x = np.arange(len(labels))

    def _values(metric_map, key):
        return [metric_map[name][key] for name in POLICY_ORDER if name in metric_map]

    axes[0, 0].bar(x, _values(iid_metrics, "total_operating_cost"), color=colors)
    axes[0, 1].bar(x, _values(lt_metrics, "total_operating_cost"), color=colors)
    axes[1, 0].bar(
        x,
        _values(iid_metrics, "total_thermal_cost"),
        color="#e15759",
        label="thermal overload",
    )
    axes[1, 0].bar(
        x,
        _values(iid_metrics, "total_reserve_shortfall"),
        bottom=_values(iid_metrics, "total_thermal_cost"),
        color="#4e79a7",
        label="reserve shortfall",
    )
    axes[1, 1].bar(
        x,
        _values(lt_metrics, "total_thermal_cost"),
        color="#e15759",
        label="thermal overload",
    )
    axes[1, 1].bar(
        x,
        _values(lt_metrics, "total_reserve_shortfall"),
        bottom=_values(lt_metrics, "total_thermal_cost"),
        color="#4e79a7",
        label="reserve shortfall",
    )

    axes[0, 0].set_title("Representative IID Episode: cost")
    axes[0, 1].set_title("Worst line_tightening Episode: cost")
    axes[1, 0].set_title("Representative IID Episode: violations")
    axes[1, 1].set_title("Worst line_tightening Episode: violations")
    axes[0, 0].set_ylabel("USD / episode")
    axes[1, 0].set_ylabel("MW-sum / episode")

    for ax in axes.reshape(-1):
        ax.set_xticks(x)
        ax.set_xticklabels(labels)

    axes[1, 0].legend(frameon=False, fontsize=9)
    axes[1, 1].legend(frameon=False, fontsize=9)

    fig.suptitle("TSO Phase-1 same-window policy comparison", fontsize=15, y=0.995)
    out = _figures_dir(task_dir) / "phase1_policy_compare.pdf"
    _save_figure(fig, out)
    plt.close(fig)
    return out


def _plot_episode_mechanism(
    task_dir: Path,
    episode_rollouts: dict[str, dict[str, dict[str, Any]]],
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    compare_policies = ("all_on", "ppo", "ppo_lagrangian")
    fig, axes = plt.subplots(4, 2, figsize=(16, 13), sharex="col", facecolor=PLOT_FACE)
    for ax in axes.reshape(-1):
        _style_axes(ax)

    for col, split in enumerate(("iid", "line_tightening")):
        rollouts = episode_rollouts[split]
        n_steps = len(next(iter(rollouts.values()))["total_load"])
        steps = np.arange(n_steps)

        ref = rollouts["ppo_lagrangian"]
        axes[0, col].plot(steps, ref["total_load"], color="#2f2f2f", linewidth=2.2, label="load")
        axes[0, col].plot(
            steps,
            ref["required_capacity"],
            color="#2f2f2f",
            linewidth=1.8,
            linestyle="--",
            label="load + reserve",
        )
        for name in compare_policies:
            axes[0, col].plot(
                steps,
                rollouts[name]["committed_capacity"],
                color=POLICY_COLORS[name],
                linewidth=2.2,
                label=name.replace("_", "-"),
            )

        for name in compare_policies:
            axes[1, col].plot(
                steps,
                rollouts[name]["max_line_loading_pct"],
                color=POLICY_COLORS[name],
                linewidth=2.2,
                label=name.replace("_", "-"),
            )
        axes[1, col].axhline(100.0, color="#2f2f2f", linestyle="--", linewidth=1.4)

        for name in compare_policies:
            axes[2, col].plot(
                steps,
                rollouts[name]["step_operating_cost"],
                color=POLICY_COLORS[name],
                linewidth=2.2,
                label=name.replace("_", "-"),
            )

        for name in compare_policies:
            axes[3, col].plot(
                steps,
                rollouts[name]["n_units_on"],
                color=POLICY_COLORS[name],
                linewidth=2.2,
                label=name.replace("_", "-"),
            )

        _decorate_time_axis(axes[0, col], n_steps)
        _decorate_time_axis(axes[1, col], n_steps)
        _decorate_time_axis(axes[2, col], n_steps)
        _decorate_time_axis(axes[3, col], n_steps)

    axes[0, 0].set_title("Representative IID Episode")
    axes[0, 1].set_title("Worst line_tightening Episode")
    axes[0, 0].set_ylabel("MW")
    axes[1, 0].set_ylabel("Max line loading [%]")
    axes[2, 0].set_ylabel("USD / step")
    axes[3, 0].set_ylabel("Units on")
    axes[3, 0].set_xlabel("Step")
    axes[3, 1].set_xlabel("Step")
    axes[0, 0].legend(frameon=False, fontsize=9, ncol=2)

    fig.suptitle("TSO Phase-1 episode mechanism replay", fontsize=15, y=0.995)
    out = _figures_dir(task_dir) / "phase1_episode_mechanism.pdf"
    _save_figure(fig, out)
    plt.close(fig)
    return out


def _plot_line_rating_sensitivity(
    task_dir: Path,
    sensitivity_rows: list[dict[str, float]],
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.asarray([row["effective_degrees"] for row in sensitivity_rows], dtype=np.float64)
    cost = np.asarray([row["total_operating_cost_mean"] for row in sensitivity_rows], dtype=np.float64)
    feas = np.asarray([row["feasibility_rate_mean"] for row in sensitivity_rows], dtype=np.float64)
    thermal = np.asarray([row["thermal_violation_rate_mean"] for row in sensitivity_rows], dtype=np.float64)

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.5), sharex=True, facecolor=PLOT_FACE)
    for ax in axes:
        _style_axes(ax)

    axes[0].plot(x, cost, color=POLICY_COLORS["ppo_lagrangian"], marker="o", linewidth=2.4)
    axes[0].set_ylabel("USD / episode")
    axes[0].set_title("PPO-Lagrangian sensitivity to effective line limit")

    axes[1].plot(x, feas, color="#59a14f", marker="o", linewidth=2.4, label="feasibility rate")
    axes[1].plot(x, thermal, color="#e15759", marker="o", linewidth=2.4, label="thermal violation rate")
    axes[1].set_ylabel("Rate")
    axes[1].set_xlabel("Effective line limit [deg-equivalent]")
    axes[1].legend(frameon=False, fontsize=9)

    fig.suptitle(
        "TSO Phase-1 line-rating sensitivity (IID, PPO-Lagrangian, 50 windows)",
        fontsize=14,
        y=0.995,
    )
    out = _figures_dir(task_dir) / "phase1_line_rating_sensitivity.pdf"
    _save_figure(fig, out)
    plt.close(fig)
    return out


def _mean_of(metrics_list: list[dict[str, float]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in metrics_list]))


def _run_line_rating_sensitivity(
    task_dir: Path,
    train_record: RunRecord,
    *,
    line_rating_scales: tuple[float, ...],
) -> list[dict[str, float]]:
    task_cfg = load_task_config_for_run(task_dir, train_record)
    env = UnitCommitmentEnv()
    task = make_task_from_config(task_cfg)
    max_steps = int(task_cfg.get("max_steps", 48))
    n_eval = get_eval_episodes(task_cfg)
    base_params = task.episode_params("iid", 0, n_eval, max_steps, strategy="uniform", seed=0)
    policy_fn = _load_learned_policy(
        task_dir,
        train_record,
        env,
        base_params,
        task.constraint_spec().selected_names,
    )
    agent_fn = jax.jit(lambda params, key: rollout_tso(env, params, key, policy_fn))

    rows: list[dict[str, float]] = []
    for scale in line_rating_scales:
        task_scaled = make_task_from_config(task_cfg, line_rating_scale=scale)
        metrics = run_episodes(task_scaled, "iid", agent_fn, n_eval, max_steps, seed=0)
        rows.append(
            {
                "line_rating_scale": float(scale),
                "effective_degrees": float(scale * _TSO_SYNTHETIC_LINE_LIMIT_DEGREES),
                "total_operating_cost_mean": _mean_of(metrics, "total_operating_cost"),
                "total_thermal_cost_mean": _mean_of(metrics, "total_thermal_cost"),
                "thermal_violation_rate_mean": _mean_of(metrics, "thermal_violation_rate"),
                "reserve_shortfall_rate_mean": _mean_of(metrics, "reserve_shortfall_rate"),
                "feasibility_rate_mean": _mean_of(metrics, "feasibility_rate"),
            }
        )
    return rows


def _write_summary(task_dir: Path, payload: dict[str, Any]) -> Path:
    path = task_dir / "results" / "analysis_episode_summary.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _load_existing_sensitivity(task_dir: Path) -> list[dict[str, float]]:
    path = task_dir / "results" / "analysis_episode_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"No existing analysis summary at {path}.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("line_rating_sensitivity")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Existing analysis summary at {path} has no line_rating_sensitivity rows.")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TSO Phase-1 episode-level diagnostics.")
    parser.add_argument("--task-dir", type=Path, default=DEFAULT_TASK_DIR)
    parser.add_argument("--after", type=str, default=None)
    parser.add_argument(
        "--line-rating-scales",
        type=float,
        nargs="*",
        default=(0.80, 0.85, 0.90, 0.95, 1.00),
        help="IID line-rating sensitivity sweep for the final PPO-Lagrangian policy.",
    )
    parser.add_argument(
        "--reuse-sensitivity",
        action="store_true",
        help="Reuse line_rating_sensitivity from an existing analysis summary instead of recomputing it.",
    )
    args = parser.parse_args(argv)

    task_dir = args.task_dir.resolve()
    records = _load_records(task_dir, args.after)
    train_records = _latest_train_records(records)
    if "ppo_lagrangian" not in train_records:
        raise FileNotFoundError("No completed PPO-Lagrangian train record found.")

    lag_train = train_records["ppo_lagrangian"]
    task_cfg, task, base_params = _base_params_for_policy(task_dir, lag_train)
    env = UnitCommitmentEnv()
    selected_names = task.constraint_spec().selected_names

    policies: dict[str, Callable] = {
        "all_on": _all_on_policy,
        "merit_order": _merit_order_policy,
    }
    learned_policies_raw: dict[str, Callable] = {}
    for algo, record in train_records.items():
        raw_policy = _load_learned_policy(task_dir, record, env, base_params, selected_names)
        learned_policies_raw[algo] = raw_policy
        policies[algo] = (
            lambda obs, state, key, params, _policy=raw_policy: _policy(obs, state, key)
        )

    iid_eval = _linked_eval_record(records, train_run_id=lag_train.run_id, split="iid")
    lt_eval = _linked_eval_record(records, train_run_id=lag_train.run_id, split="line_tightening")
    iid_rows = _load_per_episode_rows(task_dir, iid_eval)
    lt_rows = _load_per_episode_rows(task_dir, lt_eval)

    iid_selected = _select_representative_iid_episode(iid_rows)
    lt_selected = _select_worst_episode(lt_rows)
    iid_worst = _select_worst_episode(iid_rows)

    selected_rollouts: dict[str, dict[str, dict[str, Any]]] = {}
    metric_tables: dict[str, dict[str, dict[str, float]]] = {}

    for split, selected_row in (
        ("iid", iid_selected),
        ("line_tightening", lt_selected),
    ):
        _, _, params = _episode_params_for(
            task_dir,
            split,
            int(selected_row["episode_idx"]),
            lag_train,
        )
        episode_key = jax.random.PRNGKey(int(selected_row["episode_idx"]))
        split_rollouts: dict[str, dict[str, Any]] = {}
        for name in POLICY_ORDER:
            if name not in policies:
                continue
            split_rollouts[name] = _rollout_episode_diagnostics(env, params, episode_key, policies[name])
        if "ppo_lagrangian" not in split_rollouts:
            raise RuntimeError(f"Missing PPO-Lagrangian replay for split={split!r}.")
        _validate_replay(
            split_rollouts["ppo_lagrangian"]["summary"],
            selected_row,
            label=f"{split} episode {selected_row['episode_idx']}",
        )
        selected_rollouts[split] = split_rollouts
        metric_tables[split] = _policy_metric_table(split_rollouts)

    curves = {
        algo: _load_curve_artifacts(task_dir, record)
        for algo, record in train_records.items()
    }
    if args.reuse_sensitivity:
        sensitivity_rows = _load_existing_sensitivity(task_dir)
    else:
        sensitivity_rows = _run_line_rating_sensitivity(
            task_dir,
            lag_train,
            line_rating_scales=tuple(float(x) for x in args.line_rating_scales),
        )

    figures = {
        "phase1_training_diagnostics": str(_plot_training_diagnostics(task_dir, curves)),
        "phase1_policy_compare": str(
            _plot_policy_compare(task_dir, metric_tables["iid"], metric_tables["line_tightening"])
        ),
        "phase1_episode_mechanism": str(_plot_episode_mechanism(task_dir, selected_rollouts)),
        "phase1_line_rating_sensitivity": str(
            _plot_line_rating_sensitivity(task_dir, sensitivity_rows)
        ),
    }

    summary_payload = {
        "task_dir": str(task_dir),
        "train_runs": {algo: record.run_id for algo, record in train_records.items()},
        "selected_episodes": {
            "iid_representative": iid_selected,
            "iid_worst": iid_worst,
            "line_tightening_worst": lt_selected,
        },
        "aggregate_eval_summary": {
            "iid_run_id": iid_eval.run_id,
            "iid_ppo_lagrangian_mean_operating_cost": float(iid_eval.metrics["total_operating_cost"]),
            "iid_ppo_lagrangian_mean_feasibility": float(iid_eval.metrics["feasibility_rate"]),
            "iid_ppo_lagrangian_episode_thermal_incidence": float(
                np.mean([float(row["total_thermal_cost"]) > 1e-6 for row in iid_rows])
            ),
            "iid_ppo_lagrangian_episode_reserve_incidence": float(
                np.mean([float(row["total_reserve_shortfall"]) > 1e-6 for row in iid_rows])
            ),
            "line_tightening_run_id": lt_eval.run_id,
            "line_tightening_ppo_lagrangian_episode_thermal_incidence": float(
                np.mean([float(row["total_thermal_cost"]) > 1e-6 for row in lt_rows])
            ),
            "line_tightening_ppo_lagrangian_episode_reserve_incidence": float(
                np.mean([float(row["total_reserve_shortfall"]) > 1e-6 for row in lt_rows])
            ),
            "line_tightening_ppo_lagrangian_mean_feasibility": float(lt_eval.metrics["feasibility_rate"]),
        },
        "same_window_policy_metrics": metric_tables,
        "line_rating_sensitivity": sensitivity_rows,
        "figures": figures,
    }
    summary_path = _write_summary(task_dir, summary_payload)

    print(f"[phase1_analysis] summary: {summary_path}")
    for name, path in figures.items():
        print(f"[phase1_analysis] figure {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
