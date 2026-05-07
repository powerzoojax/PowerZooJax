"""Phase-1 DERs diagnostics for an isolated seed-0 campaign."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.common.configs import load_train_config, load_task_config
from benchmarks.common.io import (
    RunRecord,
    load_manifest,
    load_manifest_filtered,
    load_pickle,
    load_run,
)
from powerzoojax.rl.ippo import SharedActorCritic
from powerzoojax.tasks.ders import DERsTask

DEFAULT_TASK_DIR = Path(__file__).resolve().parent
LEARNED_ALGOS = ("ippo", "ippo_safe", "ippo_lagrangian")
POLICY_ORDER = ("no_control", "volt_droop", "ippo", "ippo_safe", "ippo_lagrangian")
POLICY_COLORS = {
    "no_control": "#7a7a7a",
    "volt_droop": "#2e7d32",
    "ippo": "#1565c0",
    "ippo_safe": "#c2185b",
    "ippo_lagrangian": "#ef6c00",
}
POLICY_LABELS = {
    "no_control": "No control",
    "volt_droop": "Voltage droop",
    "ippo": "IPPO",
    "ippo_safe": "IPPO-rs",
    "ippo_lagrangian": "IPPO-Lag",
}


def _figures_dir(task_dir: Path) -> Path:
    return task_dir / "results" / "figures"


def _save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")


def _style_axes(ax) -> None:
    ax.grid(True, alpha=0.35, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _load_records(task_dir: Path, after: str | None) -> list[RunRecord]:
    if after:
        return load_manifest_filtered(task_dir, after=after)
    return load_manifest(task_dir)


def _latest_records(records: list[RunRecord], predicate: Callable[[RunRecord], bool]) -> list[RunRecord]:
    latest: dict[tuple, RunRecord] = {}
    for record in records:
        if not predicate(record):
            continue
        key = (
            record.algo,
            record.seed,
            record.split,
            bool((record.artifacts or {}).get("params")),
        )
        cur = latest.get(key)
        if cur is None or (record.timestamp, record.run_id) > (cur.timestamp, cur.run_id):
            latest[key] = record
    return list(latest.values())


def _parse_train_run_id_from_notes(notes: str) -> str | None:
    m = re.search(r"eval of ([^ ]+)", notes or "")
    return m.group(1) if m else None


def _load_per_episode_metrics(task_dir: Path, record: RunRecord) -> list[dict]:
    rel = (record.artifacts or {}).get("per_episode")
    if not rel:
        return []
    path = task_dir / "results" / rel
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _train_curve_artifacts(task_dir: Path, run_id: str) -> dict[str, np.ndarray]:
    arts_dir = task_dir / "results" / "artifacts"
    out: dict[str, np.ndarray] = {}
    for key in (
        "timesteps",
        "learning_curve_train_return",
        "mean_reward",
        "mean_cost_voltage_violation",
        "lambda_voltage_violation",
        "lambda",
    ):
        path = arts_dir / f"{run_id}_{key}.npy"
        if path.exists():
            out[key] = np.load(path)
    return out


def _load_policy_builder(task_dir: Path, train_record: RunRecord) -> tuple[Callable, dict]:
    cfg = load_train_config(
        task_dir,
        train_record.algo,
        config_path=None,
        algo_key_map={
            "ippo_safe": "ippo_safe",
            "ippo_lagrangian": "ippo_lagrangian",
        },
        default_key="ippo",
    )
    hidden_dims = tuple(cfg.get("hidden_dims", [128, 128]))
    net_params = load_pickle(task_dir / "results" / train_record.artifacts["params"])

    def _builder(env_marl):
        action_dim = env_marl.action_space().shape[0]
        agent_names = env_marl.agent_names
        type_to_indices: dict[str, list[int]] = {}
        for i, name in enumerate(agent_names):
            type_to_indices.setdefault(name.split("_")[0], []).append(i)
        networks = {
            t: SharedActorCritic(hidden_dims=hidden_dims, action_dim=action_dim)
            for t in net_params
        }

        def _policy(obs_dict):
            actions = {}
            for t, idxs in type_to_indices.items():
                for idx in idxs:
                    name = agent_names[idx]
                    mean, _, _ = networks[t].apply(net_params[t], obs_dict[name])
                    actions[name] = jnp.clip(mean, -1.0, 1.0)
            return actions

        return _policy

    return _builder, cfg


def _candidate_rows(records: list[RunRecord], *, seed: int) -> list[dict]:
    eval_by_train_split: dict[tuple[str, str], RunRecord] = {}
    train_records = [
        record
        for record in records
        if record.split == "train"
        and record.seed == seed
        and record.algo in LEARNED_ALGOS
        and record.status == "completed"
        and (record.artifacts or {}).get("params")
    ]
    for record in records:
        if record.seed != seed or record.status != "completed":
            continue
        train_run_id = _parse_train_run_id_from_notes(record.notes)
        if train_run_id is None:
            continue
        key = (train_run_id, record.split)
        cur = eval_by_train_split.get(key)
        if cur is None or (record.timestamp, record.run_id) > (cur.timestamp, cur.run_id):
            eval_by_train_split[key] = record

    latest_baselines: dict[tuple[str, str], RunRecord] = {}
    for record in records:
        if record.seed != seed or record.status != "completed":
            continue
        if record.algo not in ("no_control", "volt_droop"):
            continue
        key = (record.algo, record.split)
        cur = latest_baselines.get(key)
        if cur is None or (record.timestamp, record.run_id) > (cur.timestamp, cur.run_id):
            latest_baselines[key] = record

    rows: list[dict] = []
    for train_record in train_records:
        iid = eval_by_train_split.get((train_record.run_id, "iid"))
        vt = eval_by_train_split.get((train_record.run_id, "voltage_tightening"))
        if iid is None:
            continue
        vt_no = latest_baselines.get(("no_control", "voltage_tightening"))
        vt_droop = latest_baselines.get(("volt_droop", "voltage_tightening"))
        iid_ok = float(iid.metrics.get("voltage_violation_rate", 1.0)) == 0.0
        vt_vs_no = None
        vt_vs_droop = None
        if vt is not None and vt_no is not None:
            vt_vs_no = float(vt.metrics.get("voltage_violation_steps", np.inf)) < float(
                vt_no.metrics.get("voltage_violation_steps", np.inf)
            )
        if vt is not None and vt_droop is not None:
            vt_vs_droop = float(vt.metrics.get("voltage_violation_steps", np.inf)) <= float(
                vt_droop.metrics.get("voltage_violation_steps", np.inf)
            )
        passes = bool(iid_ok and (vt_vs_no is True) and (vt_vs_droop is True))
        rows.append(
            {
                "algo": train_record.algo,
                "train_run_id": train_record.run_id,
                "iid_run_id": iid.run_id,
                "voltage_tightening_run_id": None if vt is None else vt.run_id,
                "iid_mean_p_loss_mw": float(iid.metrics.get("mean_p_loss_mw", np.inf)),
                "iid_voltage_violation_rate": float(
                    iid.metrics.get("voltage_violation_rate", np.inf)
                ),
                "voltage_tightening_violation_steps": float(
                    np.inf if vt is None else vt.metrics.get("voltage_violation_steps", np.inf)
                ),
                "passes_physics_gate": passes,
            }
        )
    rows.sort(
        key=lambda row: (
            0 if row["passes_physics_gate"] else 1,
            row["iid_mean_p_loss_mw"],
            row["iid_voltage_violation_rate"],
            row["voltage_tightening_violation_steps"],
        )
    )
    return rows


def _canonical_train_run_id(
    records: list[RunRecord],
    *,
    seed: int,
    explicit_run_id: str | None,
) -> str:
    if explicit_run_id is not None:
        return explicit_run_id
    candidates = _candidate_rows(records, seed=seed)
    if not candidates:
        raise FileNotFoundError("No completed DERs train/eval records found for phase1 analysis.")
    return candidates[0]["train_run_id"]


def _select_episode(
    per_episode: list[dict],
    *,
    baseline_per_episode: list[dict] | None = None,
) -> dict:
    """Pick the IID episode that exhibits the most physical stress.

    Selection criterion (in order of preference, applied to ``baseline_per_episode``
    if given, otherwise to ``per_episode``):
      1. lowest ``mean_v_min`` under no-control — the strongest undervoltage stress
         is the one where Q injection / SOC dispatch can actually differentiate
         policies. The original ``median safe loss`` heuristic biased toward
         already-easy episodes where every policy reaches the same baseline.

    The chosen episode is matched back to ``per_episode`` (the canonical algo's
    list) by ``episode_start`` so the trace reproduces the same profile window.
    """
    source = baseline_per_episode if baseline_per_episode else per_episode
    if not source:
        raise ValueError("No per-episode metrics available for episode selection.")
    chosen = min(source, key=lambda row: float(row.get("mean_v_min", 1.0)))
    if baseline_per_episode is None:
        return chosen
    target_start = int(round(float(chosen.get("episode_start", 0.0))))
    for row in per_episode:
        if int(round(float(row.get("episode_start", -1.0)))) == target_start:
            return row
    return chosen


def _marl_zero_policy(env_marl):
    def _action_for(name: str) -> jnp.ndarray:
        agent_type = name.split("_")[0]
        if agent_type == "renewable":
            return jnp.array([1.0, 0.0], dtype=jnp.float32)
        return jnp.zeros(env_marl.action_space().shape, dtype=jnp.float32)

    return lambda _obs_dict: {name: _action_for(name) for name in env_marl.agent_names}


def _marl_volt_droop_policy(env_marl):
    def _policy(obs_dict):
        actions = {}
        for name in env_marl.agent_names:
            agent_type = name.split("_")[0]
            own_v = 1.0 + jnp.asarray(obs_dict[name][0], dtype=jnp.float32) * 0.1
            if agent_type == "battery":
                q = jnp.clip((1.0 - own_v) * 10.0, -1.0, 1.0)
                actions[name] = jnp.array([0.0, q], dtype=jnp.float32)
            elif agent_type == "renewable":
                curtail = jnp.where(own_v > 1.03, -0.5, 1.0)
                q = jnp.clip((1.0 - own_v) * 10.0, -1.0, 1.0)
                actions[name] = jnp.array([curtail, q], dtype=jnp.float32)
            else:
                curtail = jnp.where(own_v < 0.97, 0.5, 0.0)
                actions[name] = jnp.array([curtail, 0.0], dtype=jnp.float32)
        return actions

    return _policy


def _trace_policy_episode(
    task,
    *,
    split: str,
    episode_start: int,
    policy_name: str,
    learned_policy_builder: Callable | None = None,
    seed: int = 0,
) -> dict:
    """Roll one episode and record physically-realized per-device quantities.

    Recording convention: at iteration ``k`` we store the state and info
    *after* the step has been applied with action ``k``. This aligns
    ``soc[k]`` (post-step), ``battery_q_mvar[k]`` (PQ-circle-clipped output of
    step ``k``), ``renewable_p_mw[k]`` (realized PV output), and
    ``v_min[k]/v_max[k]/p_loss[k]`` (info from step ``k``) on a single time
    index. This deliberately replaces the original mixed pre/post-step
    convention so the figure can be read end-to-end.

    The realized battery Q is recovered by calling ``BatteryBundle.step``
    independently on the pre-step bundle state with the same flat action — the
    bundle is deterministic and ``ctx`` is unused, so this exactly reproduces
    the env's PQ-circle-clipped ``feasible_q``.
    """
    from powerzoojax.envs.grid.dist import DistGridEnv
    from powerzoojax.rl.multi_agent import DistGridMARLEnv

    params = task.params_from_start(split, episode_start)
    env_marl = DistGridMARLEnv(
        DistGridEnv(),
        params,
        voltage_penalty=0.0,
        observation_mode="local",
    )
    if policy_name == "no_control":
        policy_fn = _marl_zero_policy(env_marl)
    elif policy_name == "volt_droop":
        policy_fn = _marl_volt_droop_policy(env_marl)
    else:
        if learned_policy_builder is None:
            raise ValueError(f"Missing learned policy builder for {policy_name!r}")
        policy_fn = learned_policy_builder(env_marl)

    key = jax.random.PRNGKey(seed)
    obs_dict, state = env_marl.reset(key)
    n_steps = int(params.max_steps)

    bat_bundle, ren_bundle, flex_bundle = params.resources
    bat_s_rated = np.asarray(bat_bundle.s_rated, dtype=np.float32)
    ren_capacity_mw = np.asarray(ren_bundle.capacity_mw, dtype=np.float32)
    traces = {
        "policy_name": policy_name,
        "episode_start": int(episode_start),
        "v_min": [],
        "v_max": [],
        "p_loss_mw": [],
        "cost_continuous": [],
        "is_safe": [],
        "n_violations": [],
        "reward": [],
        "battery_soc": [],
        "battery_p_mw": [],
        "battery_q_mvar": [],          # realized, PQ-circle-clipped
        "battery_q_cmd_mvar": [],      # commanded (action * s_rated), pre-clip
        "renewable_p_mw": [],          # realized output
        "renewable_p_max_mw": [],      # cf * capacity_mw at this step
        "renewable_q_mvar": [],        # realized, PQ-circle-clipped
        "renewable_curtail_mw": [],    # realized curtailed = p_max - p_mw
        "renewable_curtail_frac_action": [],  # legacy: action-derived (1-a_curt)/2
        "renewable_cf": [],
        "flex_curtail_mw": [],
        "flex_shift_out_mw": [],
        "flex_shift_in_mw": [],
        "battery_action": [],
        "renewable_action": [],
        "flexload_action": [],
    }

    for _ in range(n_steps):
        bat_state_pre, _ren_state_pre, _flex_state_pre = state.grid_state.resource_states

        key, step_key = jax.random.split(key)
        actions = policy_fn(obs_dict)
        battery_action = np.stack(
            [np.asarray(actions[f"battery_{i}"], dtype=np.float32) for i in range(bat_bundle.n_devices)],
            axis=0,
        )
        renewable_action = np.stack(
            [np.asarray(actions[f"renewable_{i}"], dtype=np.float32) for i in range(ren_bundle.n_devices)],
            axis=0,
        )
        flexload_action = np.stack(
            [np.asarray(actions[f"flexload_{i}"], dtype=np.float32) for i in range(flex_bundle.n_devices)],
            axis=0,
        )

        # Realized battery P/Q via direct bundle.step call on the pre-step state.
        flat_bat_action = jnp.asarray(battery_action.reshape(-1), dtype=jnp.float32)
        _, bat_p_realized, bat_q_realized, _, _ = bat_bundle.step(bat_state_pre, flat_bat_action, {})

        # Advance the env one step (this also re-runs the same bundle math
        # internally; bundles are deterministic so the two calls agree).
        obs_dict, state, rewards, _dones, info = env_marl.step(step_key, state, actions)
        bat_state, ren_state, flex_state = state.grid_state.resource_states

        # Post-step physical quantities (all aligned to step `k`).
        traces["battery_soc"].append(np.asarray(bat_state.soc, dtype=np.float32))
        traces["battery_p_mw"].append(np.asarray(bat_p_realized, dtype=np.float32))
        traces["battery_q_mvar"].append(np.asarray(bat_q_realized, dtype=np.float32))
        traces["battery_q_cmd_mvar"].append(
            (battery_action[:, 1] * bat_s_rated).astype(np.float32)
        )

        ren_p_mw = np.asarray(ren_state.p_mw, dtype=np.float32)
        ren_cf = np.asarray(ren_state.cf, dtype=np.float32)
        ren_p_max = ren_cf * ren_capacity_mw
        traces["renewable_p_mw"].append(ren_p_mw)
        traces["renewable_p_max_mw"].append(ren_p_max)
        traces["renewable_q_mvar"].append(np.asarray(ren_state.q_mvar, dtype=np.float32))
        traces["renewable_curtail_mw"].append(np.maximum(ren_p_max - ren_p_mw, 0.0))
        traces["renewable_curtail_frac_action"].append(
            np.asarray(ren_state.curtail_frac, dtype=np.float32)
        )
        traces["renewable_cf"].append(ren_cf)

        traces["flex_curtail_mw"].append(np.asarray(flex_state.curtailed_mw, dtype=np.float32))
        traces["flex_shift_out_mw"].append(np.asarray(flex_state.shift_out_mw, dtype=np.float32))
        traces["flex_shift_in_mw"].append(np.asarray(flex_state.shift_in_mw, dtype=np.float32))

        traces["battery_action"].append(battery_action)
        traces["renewable_action"].append(renewable_action)
        traces["flexload_action"].append(flexload_action)

        traces["v_min"].append(float(np.asarray(info.get("v_min_step", 0.0))))
        traces["v_max"].append(float(np.asarray(info.get("v_max_step", 0.0))))
        traces["p_loss_mw"].append(float(np.asarray(info.get("p_loss_MW", 0.0))))
        traces["cost_continuous"].append(float(np.asarray(info.get("cost_continuous", 0.0))))
        traces["is_safe"].append(bool(np.asarray(info.get("is_safe", False))))
        traces["n_violations"].append(float(np.asarray(info.get("n_violations", 0.0))))
        traces["reward"].append(float(np.asarray(rewards[env_marl.agent_names[0]])))

    for key_name, values in list(traces.items()):
        if key_name in ("policy_name", "episode_start"):
            continue
        traces[key_name] = np.asarray(values)

    traces["battery_soc_delta_max"] = float(
        np.max(np.ptp(np.asarray(traces["battery_soc"]), axis=0))
    )
    traces["renewable_action_abs_max"] = float(np.max(np.abs(traces["renewable_action"])))
    traces["flexload_action_abs_max"] = float(np.max(np.abs(traces["flexload_action"])))
    return traces


def _plot_episode_voltage(task_dir: Path, traces: dict[str, dict], *, v_min: float, v_max: float) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.arange(len(next(iter(traces.values()))["v_min"]))
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.5), sharex=True)
    for ax in axes:
        _style_axes(ax)
    for name in POLICY_ORDER:
        trace = traces.get(name)
        if trace is None:
            continue
        axes[0].plot(steps, trace["v_min"], label=name, color=POLICY_COLORS[name], linewidth=2.0)
        axes[1].plot(steps, trace["v_max"], label=name, color=POLICY_COLORS[name], linewidth=2.0)
    axes[0].axhline(v_min, color="black", linestyle="--", linewidth=1.0)
    axes[1].axhline(v_max, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("v_min (p.u.)")
    axes[1].set_ylabel("v_max (p.u.)")
    axes[1].set_xlabel("Step")
    axes[0].set_title("DERs Phase-1 — representative IID episode voltage envelope")
    axes[0].legend(loc="best", fontsize=8, ncol=2)
    out = _figures_dir(task_dir) / "phase1_episode_voltage.pdf"
    _save_figure(fig, out)
    plt.close(fig)
    return out


def _plot_policy_compare(task_dir: Path, traces: dict[str, dict]) -> Path:
    """Render the 3×2 policy comparison panel using realized physical quantities.

    Aggregation conventions chosen for physical correctness:
      * Battery SOC — per-device mean (intuitive 0–1 scale; aggregating SOC
        across devices is meaningless).
      * Battery Q, PV Q — system-total MVAr (this is the quantity the grid
        actually sees).
      * PV curtailment — system-total realized MW (= Σ max(p_max − p_out, 0)).
        Recording the action-derived ``curtail_frac`` made the original figure
        report fictitious "curtailment" at night when ``p_max = 0``.
      * Flex — system-total MW; plotted as net instantaneous load relief
        (curtail + shift_out − shift_in) so a single line per policy is
        legible.
      * Active loss — system-total MW.
      * Voltage envelope panel overlays v_min trace + the constraint band so
        readers can see when stress actually occurs.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.25,
            "pdf.fonttype": 42,
        }
    )

    first_trace = next(iter(traces.values()))
    n_steps = len(first_trace["v_min"])
    dt_hours = 24.0 / max(n_steps, 1)

    # Power-systems convention: align x=0 with clock 00:00 (midnight) regardless
    # of when the episode's profile window happened to start. The episode is a
    # continuous 24 h trajectory; we just choose where to "cut" the day for
    # display by circular-shifting every per-step trace array.
    episode_start = int(first_trace.get("episode_start", 0))
    midnight_step = (-episode_start) % n_steps  # step k* whose start = clock 00:00
    clock_hours = np.arange(n_steps, dtype=np.float32) * dt_hours  # 0..24

    def _roll(arr: np.ndarray) -> np.ndarray:
        return np.roll(arr, -midnight_step, axis=0)

    # Reference cf from any trace (PV profile is identical across policies).
    ref_cf = _roll(np.mean(first_trace["renewable_cf"], axis=1))
    pv_active = ref_cf > 0.02
    if pv_active.any():
        idx_active = np.where(pv_active)[0]
        sun_h0 = float(clock_hours[idx_active[0]])
        sun_h1 = float(clock_hours[idx_active[-1]] + dt_hours)
    else:
        sun_h0 = sun_h1 = None

    fig, axes = plt.subplots(3, 2, figsize=(10.0, 7.8), sharex=True)
    fig.patch.set_facecolor("white")
    axes = axes.reshape(-1)
    for ax in axes:
        _style_axes(ax)
        if sun_h0 is not None:
            ax.axvspan(sun_h0, sun_h1, color="#fff3b0", alpha=0.45, zorder=0)
        ax.axhline(0.0, color="#9e9e9e", linewidth=0.6, linestyle=":", zorder=1)

    for name in POLICY_ORDER:
        trace = traces.get(name)
        if trace is None:
            continue
        color = POLICY_COLORS[name]
        label = POLICY_LABELS.get(name, name)
        soc_mean = _roll(np.mean(trace["battery_soc"], axis=1))
        bat_q_total = _roll(np.sum(trace["battery_q_mvar"], axis=1))
        pv_curtail_total = _roll(np.sum(trace["renewable_curtail_mw"], axis=1))
        pv_q_total = _roll(np.sum(trace["renewable_q_mvar"], axis=1))
        flex_net = _roll(
            np.sum(trace["flex_curtail_mw"], axis=1)
            + np.sum(trace["flex_shift_out_mw"], axis=1)
            - np.sum(trace["flex_shift_in_mw"], axis=1)
        )
        p_loss = _roll(np.asarray(trace["p_loss_mw"]))
        axes[0].plot(clock_hours, soc_mean, color=color, label=label, linewidth=2.0)
        axes[1].plot(clock_hours, bat_q_total, color=color, label=label, linewidth=2.0)
        axes[2].plot(clock_hours, pv_curtail_total, color=color, label=label, linewidth=2.0)
        axes[3].plot(clock_hours, pv_q_total, color=color, label=label, linewidth=2.0)
        axes[4].plot(clock_hours, flex_net, color=color, linewidth=2.0)
        axes[5].plot(clock_hours, p_loss, color=color, linewidth=2.0)

    axes[0].set_title("Battery SOC (per-device mean)")
    axes[0].set_ylabel("SOC")
    axes[1].set_title("Battery Q realized (system total, MVAr)")
    axes[1].set_ylabel("Q  [MVAr]")
    axes[2].set_title("PV curtailment realized (system total, MW)")
    axes[2].set_ylabel("MW")
    axes[3].set_title("PV Q realized (system total, MVAr)")
    axes[3].set_ylabel("Q  [MVAr]")
    axes[4].set_title("Flex net load relief = curtail + shift-out − shift-in (MW)")
    axes[4].set_ylabel("MW")
    axes[5].set_title("Active loss (system total, MW)")
    axes[5].set_ylabel("MW")

    # X-axis is clock hour-of-day in [0, 24]. After the circular shift above,
    # column 0 of the rolled arrays already represents clock 00:00 — no further
    # tick remapping is needed.
    for ax in axes:
        ax.set_xlim(0, 24)
        ax.set_xticks(np.arange(0, 25, 3))
    for ax in axes[4:]:
        ax.set_xlabel("Hour of day")

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=False))
    if sun_h0 is not None:
        from matplotlib.patches import Patch
        by_label["PV active (cf > 0.02)"] = Patch(
            facecolor="#fff3b0", alpha=0.6, edgecolor="none"
        )
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        ncol=len(by_label),
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),
        handlelength=2.0,
        columnspacing=1.2,
    )
    fig.suptitle(
        "DERs Phase-1 policy behavior on the same in-distribution episode "
        f"(GB in-distribution profile row {episode_start}; x-axis aligned to clock 00:00)",
        fontsize=10,
        y=0.945,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    out = _figures_dir(task_dir) / "phase1_policy_compare.pdf"
    _save_figure(fig, out)
    plt.close(fig)
    return out


def _plot_training_diagnostics(
    task_dir: Path,
    records: list[RunRecord],
    *,
    canonical_train_run_id: str,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    latest_trains = _latest_records(
        records,
        lambda record: record.split == "train"
        and record.algo in LEARNED_ALGOS
        and record.status == "completed"
        and (record.artifacts or {}).get("params"),
    )
    canonical = load_run(canonical_train_run_id, task_dir)
    has_lagrangian = canonical.algo == "ippo_lagrangian"
    nrows = 3 if has_lagrangian else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(10.5, 4.0 + 2.2 * (nrows - 1)), sharex=False)
    axes = np.atleast_1d(axes)
    for ax in axes:
        _style_axes(ax)

    for record in sorted(latest_trains, key=lambda rec: (rec.algo, rec.timestamp)):
        curves = _train_curve_artifacts(task_dir, record.run_id)
        xs = curves.get("timesteps")
        ys = curves.get("learning_curve_train_return")
        if xs is None:
            xs = np.arange(len(curves.get("mean_reward", [])), dtype=np.float32)
        if ys is None:
            ys = curves.get("mean_reward")
        if xs is None or ys is None:
            continue
        xs_arr = np.asarray(xs, dtype=np.float64).reshape(-1)
        ys_arr = np.asarray(ys, dtype=np.float64).reshape(-1)
        n = min(xs_arr.size, ys_arr.size)
        if n == 0:
            continue
        axes[0].plot(xs_arr[:n] / 1e6, ys_arr[:n], color=POLICY_COLORS.get(record.algo, "#555"), linewidth=2.0, label=record.algo)

    axes[0].set_title("Train return curves")
    axes[0].set_xlabel("Timesteps (M)")
    axes[0].set_ylabel("Return")
    axes[0].legend(loc="best", fontsize=8, ncol=3)

    if has_lagrangian:
        curves = _train_curve_artifacts(task_dir, canonical_train_run_id)
        xs = curves.get("timesteps", np.arange(len(curves.get("mean_cost_voltage_violation", []))))
        cost_y = curves.get("mean_cost_voltage_violation")
        lam_y = curves.get("lambda_voltage_violation", curves.get("lambda"))
        if cost_y is not None and len(cost_y) > 0:
            axes[1].plot(xs[: len(cost_y)] / 1e6, cost_y, color=POLICY_COLORS["ippo_lagrangian"], linewidth=2.0)
        if lam_y is not None and len(lam_y) > 0:
            axes[2].plot(xs[: len(lam_y)] / 1e6, lam_y, color=POLICY_COLORS["ippo_lagrangian"], linewidth=2.0)
        axes[1].set_title("Lagrangian mean_cost_voltage_violation")
        axes[2].set_title("Lagrangian lambda")
        axes[1].set_xlabel("Timesteps (M)")
        axes[2].set_xlabel("Timesteps (M)")
        axes[1].set_ylabel("Cost")
        axes[2].set_ylabel("Lambda")

    fig.tight_layout()
    out = _figures_dir(task_dir) / "phase1_training_diagnostics.pdf"
    _save_figure(fig, out)
    plt.close(fig)
    return out


def _run_config_label(task_dir: Path, train_run_id: str) -> str:
    cfg_path = task_dir / "results" / "artifacts" / f"{train_run_id}_config.json"
    if not cfg_path.exists():
        return train_run_id
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    train_cfg = cfg.get("train_config", {})
    bank = cfg.get("train_window_bank", {})
    if train_cfg.get("voltage_penalty") is not None:
        return f"vp={train_cfg['voltage_penalty']},w={bank.get('train_window_count', '?')}"
    if train_cfg.get("lambda_lr") is not None:
        return f"llr={train_cfg['lambda_lr']},w={bank.get('train_window_count', '?')}"
    return train_run_id


def _plot_sweep_tradeoff(task_dir: Path, records: list[RunRecord], *, seed: int) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eval_groups: dict[str, dict[str, RunRecord]] = {}
    train_algo: dict[str, str] = {}
    for record in records:
        if record.seed != seed or record.status != "completed":
            continue
        if record.split == "train" and (record.artifacts or {}).get("params"):
            train_algo[record.run_id] = record.algo
            continue
        train_run_id = _parse_train_run_id_from_notes(record.notes)
        if train_run_id is None:
            continue
        eval_groups.setdefault(train_run_id, {})[record.split] = record

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax in axes:
        _style_axes(ax)

    for train_run_id, group in sorted(eval_groups.items()):
        iid = group.get("iid")
        vt = group.get("voltage_tightening")
        if iid is None or vt is None:
            continue
        algo = train_algo.get(train_run_id)
        if algo not in LEARNED_ALGOS:
            continue
        color = POLICY_COLORS.get(algo, "#555")
        x = float(iid.metrics.get("mean_p_loss_mw", np.nan))
        y1 = float(vt.metrics.get("voltage_violation_steps", np.nan))
        y2 = 1.0 - float(vt.metrics.get("voltage_violation_rate", np.nan))
        label = _run_config_label(task_dir, train_run_id)
        axes[0].scatter([x], [y1], color=color, s=60, alpha=0.9)
        axes[1].scatter([x], [y2], color=color, s=60, alpha=0.9)
        axes[0].annotate(label, (x, y1), fontsize=7, xytext=(4, 3), textcoords="offset points")
        axes[1].annotate(label, (x, y2), fontsize=7, xytext=(4, 3), textcoords="offset points")

    axes[0].set_xlabel("IID mean_p_loss_mw")
    axes[0].set_ylabel("Voltage-tightening violation steps")
    axes[1].set_xlabel("IID mean_p_loss_mw")
    axes[1].set_ylabel("Voltage-tightening safety rate")
    axes[0].set_title("Sweep trade-off")
    axes[1].set_title("Sweep safety trade-off")
    fig.tight_layout()
    out = _figures_dir(task_dir) / "phase1_sweep_tradeoff.pdf"
    _save_figure(fig, out)
    plt.close(fig)
    return out


def generate_phase1_analysis(
    task_dir: Path,
    *,
    after: str | None = None,
    seed: int = 0,
    train_run_id: str | None = None,
) -> dict:
    task_cfg = load_task_config(task_dir)
    records = _load_records(task_dir, after)
    canonical_train_run_id = _canonical_train_run_id(
        records,
        seed=seed,
        explicit_run_id=train_run_id,
    )
    canonical_train = load_run(canonical_train_run_id, task_dir)
    candidate_rows = _candidate_rows(records, seed=seed)

    eval_records = {
        rec.run_id: rec
        for rec in records
        if _parse_train_run_id_from_notes(rec.notes) == canonical_train_run_id
    }
    iid_eval = next(rec for rec in eval_records.values() if rec.split == "iid")
    iid_per_episode = _load_per_episode_metrics(task_dir, iid_eval)

    no_control_iid = _latest_records(
        records,
        lambda rec: rec.split == "iid"
        and rec.algo == "no_control"
        and rec.seed == seed
        and rec.status == "completed",
    )
    baseline_per_episode: list[dict] | None = None
    if no_control_iid:
        baseline_per_episode = _load_per_episode_metrics(task_dir, no_control_iid[0])
    chosen_episode = _select_episode(
        iid_per_episode,
        baseline_per_episode=baseline_per_episode,
    )
    episode_start = int(chosen_episode["episode_start"])

    from powerzoojax.case import load_case

    task = DERsTask(
        case=load_case(task_cfg.get("case", "case141")),
        v_min=float(task_cfg["v_min"]),
        v_max=float(task_cfg["v_max"]),
        voltage_penalty=0.0,
        max_steps=int(task_cfg.get("max_steps", 48)),
    )

    traces: dict[str, dict] = {}
    traces["no_control"] = _trace_policy_episode(
        task,
        split="iid",
        episode_start=episode_start,
        policy_name="no_control",
        seed=seed * 100 + 1,
    )
    traces["volt_droop"] = _trace_policy_episode(
        task,
        split="iid",
        episode_start=episode_start,
        policy_name="volt_droop",
        seed=seed * 100 + 2,
    )

    latest_trains = {
        rec.algo: rec
        for rec in _latest_records(
            records,
            lambda rec: rec.split == "train"
            and rec.seed == seed
            and rec.algo in LEARNED_ALGOS
            and rec.status == "completed"
            and (rec.artifacts or {}).get("params"),
        )
    }
    for algo, record in latest_trains.items():
        builder, _cfg = _load_policy_builder(task_dir, record)
        traces[algo] = _trace_policy_episode(
            task,
            split="iid",
            episode_start=episode_start,
            policy_name=algo,
            learned_policy_builder=builder,
            seed=seed * 100 + 10 + len(traces),
        )

    summary = {
        "canonical_train_run_id": canonical_train_run_id,
        "canonical_algo": canonical_train.algo,
        "candidate_rows": candidate_rows,
        "selected_episode": chosen_episode,
        "policy_behaviour": {
            name: {
                "battery_soc_delta_max": float(trace["battery_soc_delta_max"]),
                "renewable_action_abs_max": float(trace["renewable_action_abs_max"]),
                "flexload_action_abs_max": float(trace["flexload_action_abs_max"]),
                "mean_p_loss_mw": float(np.mean(trace["p_loss_mw"])),
                "voltage_violation_steps": float(np.sum(trace["n_violations"] > 0)),
            }
            for name, trace in traces.items()
        },
    }

    out_json = task_dir / "results" / "analysis_episode_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    _plot_episode_voltage(
        task_dir,
        traces,
        v_min=float(task_cfg["v_min"]),
        v_max=float(task_cfg["v_max"]),
    )
    _plot_policy_compare(task_dir, traces)
    _plot_training_diagnostics(
        task_dir,
        records,
        canonical_train_run_id=canonical_train_run_id,
    )
    _plot_sweep_tradeoff(task_dir, records, seed=seed)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="DERs Phase-1 diagnostics.")
    parser.add_argument("--task-dir", type=Path, default=DEFAULT_TASK_DIR)
    parser.add_argument("--after", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-run-id", type=str, default=None)
    args = parser.parse_args()
    summary = generate_phase1_analysis(
        args.task_dir,
        after=args.after,
        seed=args.seed,
        train_run_id=args.train_run_id,
    )
    print(
        "[DERs phase1_analysis] "
        f"canonical={summary['canonical_train_run_id']} "
        f"episode_start={summary['selected_episode']['episode_start']}"
    )


if __name__ == "__main__":
    main()
