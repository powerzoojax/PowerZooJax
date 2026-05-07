"""Export one DC Microgrid episode to a multi-sheet Excel file.

For *human inspection* of policy behaviour: named observations, actions, power
flows, raw vs shaped reward, and optional DC state scalars (host ``device_get``).

**Timing convention (important)**:

- Row ``t`` is the transition starting from **obs_t** (post-reset or
  post-previous-step), then action **a_t**, then post-step **info** fields and
  **reward_shaped** (``RewardShapingWrapper`` when enabled).  So ``p_dc_mw``,
  ``soc``, costs, and ``reward_shaped`` are **after** ``a_t``, consistent with
  :func:`rollout_dcmicrogrid` / ``dump_trajectory``.

**Dependency**: ``pip install openpyxl`` (``pandas`` is already a project dep).

**Default output** (if ``--out`` is omitted):
``benchmarks/dc_microgrid/results/episode_{algo}_{split}_s{seed}.xlsx``.

Usage::

    PYTHONPATH=. python benchmarks/dc_microgrid/analysis/export_episode_excel.py \\
        --algo sac --run-id <train_run_id> --split iid --seed 0

Only **ppo** and **sac** training runs are supported (no baselines).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Task root: benchmarks/dc_microgrid
TASK_DIR = Path(__file__).resolve().parent.parent
# Default .xlsx output: benchmarks/dc_microgrid/results/episode_*.xlsx
RESULTS_DIR = TASK_DIR / "results"

# Order matches ``DataCenterMicrogridEnv._get_obs`` stack.
OBS_NAMES: tuple[str, ...] = (
    "cpu_util",
    "mem_util",
    "q_train_fill",
    "q_ft_fill",
    "queue_urgency",
    "zone_temp_norm",
    "outdoor_temp_norm",
    "cop_ratio",
    "solar_cf",
    "soc_obs",
    "dg_margin_norm",
    "p_load_norm",
    "net_load_norm",
    "batt_dis_headroom_norm",
    "batt_chg_headroom_norm",
    "grid_price_norm",
    "grid_price_6h_max_norm",
    "last_a0",
    "last_a1",
    "last_a2",
    "last_a3",
    "last_a4",
    "time_sin",
    "time_cos",
)
ACTION_NAMES: tuple[str, ...] = (
    "a_train_sched",
    "a_ft_sched",
    "a_cool",
    "a_batt",
    "a_dg",
)


def _to_float(x) -> float:
    return float(np.asarray(x).reshape(()))


def _pythonize_info(info: dict) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in info.items():
        if k == "reward_vector":
            arr = np.asarray(v).reshape(-1)
            for i, name in enumerate(("r_energy", "r_cost", "r_carbon")):
                out[name] = float(arr[i]) if i < arr.size else 0.0
            continue
        arr = np.asarray(v)
        if arr.ndim == 0:
            out[k] = float(arr)
        else:
            out[k] = arr.tolist() if arr.size <= 8 else f"<array {arr.shape}>"
    return out


def _state_diag(state) -> dict[str, float]:
    import jax

    s = jax.device_get(state)
    try:
        dc = s.dc
    except Exception:
        return {}
    return {
        "dc_t_zone": _to_float(dc.t_zone),
        "dc_t_setpoint": _to_float(dc.t_setpoint),
        "dc_p_it_mw": _to_float(dc.p_it_mw),
        "dc_p_cool_mw": _to_float(dc.p_cool_mw),
        "dc_sla_violations": float(np.asarray(dc.sla_violations).reshape(())),
    }


def _params_from_profile_start(
    task_config: dict, _eval_config: dict, start: int, max_steps: int, split: str
):
    from powerzoojax.data.dc_microgrid_profiles import apply_ood_transform
    from powerzoojax.envs.microgrid import make_dcmicrogrid_params_with_profiles

    ood = {
        "cooling_stress": "cooling_stress",
        "renewable_drought": "renewable_drought",
        "workload_swap": "workload_swap",
        "workload_shock": "workload_shock",
        "dg_derating": "dg_derating",
        "sla_tighten": "sla_tighten",
    }.get(split)
    p = make_dcmicrogrid_params_with_profiles(
        source=task_config.get("data_source", "google"),
        episode_start_step=start,
        max_steps=max_steps,
        strict=True,
        require_real_data=True,
        **(task_config.get("case_overrides") or {}),
    )
    if ood is not None:
        p = apply_ood_transform(p, ood)
    return p


def _rollout_one(
    env, params, key, max_steps, policy_fn, include_state_diag: bool
):
    import jax
    import jax.numpy as jnp

    rows: list[dict[str, Any]] = []
    obs, state = env.reset(key, params)
    for step in range(max_steps):
        k1, k_step, k_pol = jax.random.split(key, 3)
        key = k1
        action = policy_fn(obs, state, k_pol)
        a_vec = np.asarray(action, dtype=np.float32).reshape(5)
        o_vec = np.asarray(obs, dtype=np.float32).reshape(len(OBS_NAMES))
        o2, s2, r, _c, _d, info = env.step(
            k_step, state, jnp.array(action, dtype=jnp.float32), params
        )
        r_total = _to_float(r)
        p_info = _pythonize_info(dict(info))
        row: dict[str, Any] = {
            "step": step,
            "time_min": step * 5.0,
            "time_h": (step * 5.0) / 60.0,
        }
        for j, n in enumerate(OBS_NAMES):
            row[n] = float(o_vec[j])
        for j, n in enumerate(ACTION_NAMES):
            row[n] = float(a_vec[j])
        for k, v in p_info.items():
            if k in row or not isinstance(v, (int, float)):
                continue
            row[k] = v
        row["reward_shaped"] = r_total
        if include_state_diag:
            row.update(_state_diag(s2))
        obs, state = o2, s2
        rows.append(row)
    return rows


def _cumulative_from_rows(
    rows: list[dict[str, Any]], dt_h: float
) -> list[dict[str, float]]:
    s_p, s_f, s_c, s_r = 0.0, 0.0, 0.0, 0.0
    out: list[dict[str, float]] = []
    for i, r in enumerate(rows):
        s_p += float(r.get("p_dc_mw", 0.0) or 0.0) * dt_h
        s_f += float(r.get("fuel_cost", 0.0) or 0.0)
        s_c += float(r.get("carbon_kg", 0.0) or 0.0)
        s_r += float(r.get("reward_shaped", 0.0) or 0.0)
        out.append(
            {
                "step": i,
                "cum_p_dc_mwh": s_p,
                "cum_fuel_usd": s_f,
                "cum_carbon_kg": s_c,
                "cum_reward_shaped": s_r,
            }
        )
    return out


def _meta_rows(
    *,
    algo: str,
    run_id: str | None,
    split: str,
    seed: int,
    episode_idx: int,
    profile_st: int,
    max_steps: int,
    data_mode: str,
    task_config: dict,
    timing_note: str,
) -> list[dict[str, str]]:
    from benchmarks.common.io import config_hash

    h = config_hash({**task_config, "split": split}) if task_config else ""
    return [
        {"key": "algo", "value": algo},
        {"key": "run_id", "value": run_id or ""},
        {"key": "split", "value": split},
        {"key": "seed", "value": str(seed)},
        {"key": "episode_idx", "value": str(episode_idx)},
        {"key": "profile_start_step", "value": str(profile_st)},
        {"key": "max_steps", "value": str(max_steps)},
        {"key": "data_mode", "value": data_mode},
        {"key": "config_hash", "value": h},
        {"key": "export_utc", "value": datetime.now(timezone.utc).isoformat()},
        {"key": "timing_convention", "value": timing_note},
    ]


def _df_obs_legend():
    import pandas as pd

    # Match ``observation_space`` low in ``dc_microgrid.py``.
    lows = [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1, -1]
    desc = [
        "GPUs active / n_gpus",
        "GPUs on inference / n_gpus",
        "Normalised training queue demand",
        "Normalised finetune queue demand",
        "Queue slack urgency in [-1,1]",
        "Zone temp normalised in [0,1]",
        "Outdoor temp normalised in [0,1]",
        "COP factor normalised in [0,1]",
        "Solar capacity factor in [0,1]",
        "Battery SOC in [0,1]",
        "Diesel headroom normalised in [0,1]",
        "Current load normalised by total supply capacity",
        "Current net load after PV, normalised by dispatchable capacity",
        "Battery discharge headroom, normalised by converter rating",
        "Battery charge headroom, normalised by converter rating",
        "Current grid-import price, normalised by configured reference",
        "Maximum grid-import price over next 6h, normalised by configured reference",
        "Previous action: train sched",
        "Previous action: ft sched",
        "Previous action: cooling",
        "Previous action: battery in [-1,1]",
        "Previous action: DG",
        "sin(time in day)",
        "cos(time in day)",
    ]
    return pd.DataFrame(
        {
            "idx": range(len(OBS_NAMES)),
            "name": list(OBS_NAMES),
            "description": desc,
            "low": lows,
            "high": [1] * len(OBS_NAMES),
        }
    )


def _df_action_legend():
    import pandas as pd

    return pd.DataFrame(
        {
            "name": list(ACTION_NAMES),
            "description": [
                "Train workload schedule in [0,1]",
                "Finetune workload schedule in [0,1]",
                "Cooling setpoint normalised in [0,1]",
                "Battery power normalised in [-1,1] (+ = discharge)",
                "Diesel power normalised in [0,1]",
            ],
            "low": [0, 0, 0, -1, 0],
            "high": [1, 1, 1, 1, 1],
        }
    )


def export_episode_excel(
    *,
    algo: str,
    split: str,
    seed: int,
    run_id: str,
    out: Path | None = None,
    episode_idx: int = 0,
    n_episodes_span: int = 1,
    profile_start: int | None = None,
    no_state_diag: bool = False,
) -> Path:
    """Build env, rollout one trained (PPO or SAC) episode, write Excel sheets."""
    import pandas as pd
    from benchmarks.common.configs import load_config, load_task_config
    from benchmarks.common.io import load_pickle, load_run
    from benchmarks.common.runtime import build_train_cfg, make_policy_fn
    from benchmarks.dc_microgrid._reward_shaping import wrap_with_shaping
    from benchmarks.dc_microgrid.rejax_ckpt import load_sac_train_state
    from powerzoojax.envs.microgrid import DataCenterMicrogridEnv
    from powerzoojax.tasks.dc_microgrid import DCMicrogridTask

    import jax

    try:
        import openpyxl  # noqa: F401
    except ImportError as e:
        raise ImportError("Excel export needs: pip install openpyxl") from e

    task_config = load_task_config(TASK_DIR)
    max_steps = int(task_config.get("max_steps", 288))
    dt_h = 5.0 / 60.0

    task = DCMicrogridTask(
        source=task_config.get("data_source", "google"),
        max_steps=max_steps,
        case_overrides=task_config.get("case_overrides") or {},
    )
    base_env = DataCenterMicrogridEnv()
    env = wrap_with_shaping(base_env, task_config)

    n_span = max(1, n_episodes_span)
    t_prof = 288 * 365
    if profile_start is not None:
        ev_path = TASK_DIR / "configs" / f"eval_{split}.yaml"
        ev = load_config(ev_path) if ev_path.exists() else {}
        params = _params_from_profile_start(
            task_config, ev, profile_start, max_steps, split
        )
        profile_st = int(profile_start)
    else:
        params = task.episode_params(
            split, episode_idx, n_span, max_steps, strategy="uniform", seed=seed
        )
        profile_st = int(
            episode_idx / max(n_span, 1) * max(t_prof - max_steps, 1)
        )

    data_mode = "real"
    rec = load_run(run_id, TASK_DIR)
    if rec.algo not in ("ppo", "sac"):
        raise ValueError(
            f"Run {run_id!r} is algo={rec.algo!r}; only ppo and sac training runs are supported"
        )
    if rec.algo != algo:
        raise ValueError(
            f"Run {run_id!r} has algo={rec.algo!r} but --algo {algo!r} was given"
        )
    a = rec.algo
    train_path = TASK_DIR / "configs" / f"train_{a}.yaml"
    if not train_path.exists():
        train_path = TASK_DIR / "configs" / "train_ppo.yaml"
    train_cfg = build_train_cfg(load_config(train_path), algo=a)
    if a == "sac":
        rel = rec.artifacts.get("params_orbax")
        if not rel:
            raise FileNotFoundError("SAC run has no params_orbax in artifacts")
        train_state = load_sac_train_state(
            TASK_DIR / "results" / rel, train_cfg, env, params
        )
    else:
        rel_p = rec.artifacts.get("params")
        if not rel_p:
            raise FileNotFoundError("PPO run has no params in artifacts")
        train_state = load_pickle(TASK_DIR / "results" / rel_p)
    policy_fn = make_policy_fn(a, train_state, env, params, train_cfg)

    key = jax.random.PRNGKey(seed * 1_000_000 + episode_idx)
    rows = _rollout_one(
        env, params, key, max_steps, policy_fn, not no_state_diag
    )
    cum = _cumulative_from_rows(rows, dt_h)
    note = (
        "row t: obs_* is pre-action; a_* is action; p_dc_mw, soc, costs, "
        "reward_shaped are post-step (after a_t), matching rollout_dcmicrogrid."
    )
    meta = _meta_rows(
        algo=algo,
        run_id=run_id,
        split=split,
        seed=seed,
        episode_idx=episode_idx,
        profile_st=profile_st,
        max_steps=max_steps,
        data_mode=data_mode,
        task_config=task_config,
        timing_note=note,
    )
    out_path = Path(out) if out else (RESULTS_DIR / f"episode_{algo}_{split}_s{seed}.xlsx")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        pd.DataFrame(meta).to_excel(writer, sheet_name="Meta", index=False)
        pd.DataFrame(rows).to_excel(writer, sheet_name="Step", index=False)
        pd.DataFrame(cum).to_excel(writer, sheet_name="Cumulative", index=False)
        _df_obs_legend().to_excel(writer, sheet_name="ObsLegend", index=False)
        _df_action_legend().to_excel(writer, sheet_name="ActionLegend", index=False)
    print(f"[export_episode_excel] wrote {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser(description="DC Microgrid single episode -> Excel")
    p.add_argument(
        "--algo",
        required=True,
        choices=["ppo", "sac"],
    )
    p.add_argument("--run-id", required=True, help="Training run_id (PPO pkl or SAC orbax)")
    p.add_argument(
        "--split",
        required=True,
        choices=[
            "train", "iid", "cooling_stress", "renewable_drought",
            "workload_swap", "workload_shock", "dg_derating", "sla_tighten",
        ],
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episode-idx", type=int, default=0)
    p.add_argument("--n-episodes-span", type=int, default=1)
    p.add_argument("--profile-start", type=int, default=None)
    p.add_argument(
        "--out",
        default=None,
        help="Output .xlsx (default: results/episode_{algo}_{split}_s{seed}.xlsx under task dir)",
    )
    p.add_argument(
        "--no-state-diag",
        action="store_true",
        help="Omit dc_t_zone, dc_p_it_mw, etc.",
    )
    args = p.parse_args()
    try:
        export_episode_excel(
            algo=args.algo,
            split=args.split,
            seed=args.seed,
            run_id=args.run_id,
            out=Path(args.out) if args.out else None,
            episode_idx=args.episode_idx,
            n_episodes_span=args.n_episodes_span,
            profile_start=args.profile_start,
            no_state_diag=args.no_state_diag,
        )
    except ImportError as e:
        print(f"[export_episode_excel] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
