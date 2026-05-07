"""Episode-level TSO UC feasibility probe via scipy.optimize.milp.

This is an analysis utility, not a benchmark policy. It solves a single
48-step UC/ED planning problem on one benchmark episode and reports whether a
zero-reserve / zero-thermal schedule exists under the current case and split.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import jax
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.common.configs import load_task_config
from benchmarks.tso.config_runtime import get_eval_episodes, make_task_from_config
from powerzoojax.envs.grid.unit_commitment import UnitCommitmentEnv


TASK_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = TASK_DIR / "results" / "feasibility"


def _safe_limits(line_floor: np.ndarray, line_cap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    no_limit = 1e5
    floor = line_floor.astype(np.float64).copy()
    cap = line_cap.astype(np.float64).copy()
    floor[floor == 0.0] = -no_limit
    cap[cap == 0.0] = no_limit
    return floor, cap


def _total_gen_cost(unit_power_tu: np.ndarray, case) -> float:
    p = unit_power_tu.astype(np.float64)
    a = np.asarray(case.unit_cost_a, dtype=np.float64)[None, :]
    b = np.asarray(case.unit_cost_b, dtype=np.float64)[None, :]
    c = np.asarray(case.unit_cost_c, dtype=np.float64)[None, :]
    per_unit = (a / 3.0) * p**3 + (b / 2.0) * p**2 + c * p
    return float(per_unit.sum())


class _Builder:
    def __init__(self) -> None:
        self.rows: list[int] = []
        self.cols: list[int] = []
        self.data: list[float] = []
        self.lb: list[float] = []
        self.ub: list[float] = []
        self._row = 0

    def add_row(self, entries: list[tuple[int, float]], lb: float, ub: float) -> None:
        for col, value in entries:
            if value == 0.0:
                continue
            self.rows.append(self._row)
            self.cols.append(int(col))
            self.data.append(float(value))
        self.lb.append(float(lb))
        self.ub.append(float(ub))
        self._row += 1

    def build(self, n_vars: int) -> LinearConstraint:
        mat = coo_matrix(
            (self.data, (self.rows, self.cols)),
            shape=(self._row, n_vars),
            dtype=np.float64,
        ).tocsr()
        return LinearConstraint(mat, np.asarray(self.lb), np.asarray(self.ub))


def solve_episode_uc_milp(
    params,
    *,
    ignore_ramp: bool = False,
    ignore_min_updown: bool = False,
    slack_penalty: float = 1e8,
    time_limit_s: float | None = None,
) -> dict[str, float | int | bool]:
    case = params.case
    T = int(params.max_steps)
    U = int(case.n_units)
    L = int(case.n_lines)

    load_profiles = np.asarray(params.load_profiles, dtype=np.float64)  # (T, n_loads)
    nodes_loads_map = np.asarray(case.nodes_loads_map, dtype=np.float64)  # (n_nodes, n_loads)
    nodes_units_map = np.asarray(case.nodes_units_map, dtype=np.float64)  # (n_nodes, n_units)
    ptdf = np.asarray(case.PTDF, dtype=np.float64)  # (n_lines, n_nodes)
    m_u = ptdf @ nodes_units_map  # (n_lines, n_units)

    node_load_tn = load_profiles @ nodes_loads_map.T  # (T, n_nodes)
    total_load_t = node_load_tn.sum(axis=1)
    c0_tl = node_load_tn @ ptdf.T  # (T, n_lines)

    p_min = np.asarray(case.unit_p_min, dtype=np.float64)
    p_max = np.asarray(case.unit_p_max, dtype=np.float64)
    line_floor, line_cap = _safe_limits(
        np.asarray(case.line_floor, dtype=np.float64),
        np.asarray(case.line_cap, dtype=np.float64),
    )
    reserve_margin = float(params.reserve_margin_frac)
    required_cap_t = (1.0 + reserve_margin) * total_load_t

    min_up = np.asarray(params.min_up_steps, dtype=np.int32)
    min_down = np.asarray(params.min_down_steps, dtype=np.int32)
    ramp_up = np.asarray(params.ramp_up_mw, dtype=np.float64)
    ramp_down = np.asarray(params.ramp_down_mw, dtype=np.float64)
    _, reset_state = UnitCommitmentEnv().reset(jax.random.PRNGKey(0), params)
    init_status = np.asarray(reset_state.unit_status, dtype=np.int32)
    init_time = np.asarray(reset_state.time_in_state, dtype=np.int32)
    init_dispatch = np.asarray(reset_state.last_dispatch, dtype=np.float64)
    startup_cost = np.asarray(params.startup_cost, dtype=np.float64)
    no_load_cost = np.asarray(params.no_load_cost_per_step, dtype=np.float64)
    dispatch_proxy_cost = np.asarray(case.unit_cost_c, dtype=np.float64)

    n_p = T * U
    n_y = T * U
    n_su = T * U
    n_sd = T * U
    n_res = T
    n_lpos = T * L
    n_lneg = T * L
    off_p = 0
    off_y = off_p + n_p
    off_su = off_y + n_y
    off_sd = off_su + n_su
    off_res = off_sd + n_sd
    off_lpos = off_res + n_res
    off_lneg = off_lpos + n_lpos
    n_vars = off_lneg + n_lneg

    def p_idx(t: int, u: int) -> int:
        return off_p + t * U + u

    def y_idx(t: int, u: int) -> int:
        return off_y + t * U + u

    def su_idx(t: int, u: int) -> int:
        return off_su + t * U + u

    def sd_idx(t: int, u: int) -> int:
        return off_sd + t * U + u

    def res_idx(t: int) -> int:
        return off_res + t

    def lpos_idx(t: int, l: int) -> int:
        return off_lpos + t * L + l

    def lneg_idx(t: int, l: int) -> int:
        return off_lneg + t * L + l

    c = np.zeros(n_vars, dtype=np.float64)
    lb = np.full(n_vars, -np.inf, dtype=np.float64)
    ub = np.full(n_vars, np.inf, dtype=np.float64)
    integrality = np.zeros(n_vars, dtype=np.int32)

    for t in range(T):
        for u in range(U):
            c[p_idx(t, u)] = dispatch_proxy_cost[u]
            c[y_idx(t, u)] = no_load_cost[u]
            c[su_idx(t, u)] = startup_cost[u]

            lb[p_idx(t, u)] = 0.0
            ub[p_idx(t, u)] = p_max[u]

            for idx_fn in (y_idx, su_idx, sd_idx):
                idx = idx_fn(t, u)
                lb[idx] = 0.0
                ub[idx] = 1.0
                integrality[idx] = 1

        c[res_idx(t)] = slack_penalty
        lb[res_idx(t)] = 0.0

        for l in range(L):
            c[lpos_idx(t, l)] = slack_penalty
            c[lneg_idx(t, l)] = slack_penalty
            lb[lpos_idx(t, l)] = 0.0
            lb[lneg_idx(t, l)] = 0.0

    builder = _Builder()

    # Balance, generation bounds, reserve, and line constraints.
    for t in range(T):
        builder.add_row(
            [(p_idx(t, u), 1.0) for u in range(U)],
            lb=total_load_t[t],
            ub=total_load_t[t],
        )
        builder.add_row(
            [(y_idx(t, u), p_max[u]) for u in range(U)] + [(res_idx(t), 1.0)],
            lb=required_cap_t[t],
            ub=np.inf,
        )

        for u in range(U):
            builder.add_row(
                [(p_idx(t, u), 1.0), (y_idx(t, u), -p_max[u])],
                lb=-np.inf,
                ub=0.0,
            )
            builder.add_row(
                [(p_idx(t, u), 1.0), (y_idx(t, u), -p_min[u])],
                lb=0.0,
                ub=np.inf,
            )

        for l in range(L):
            upper_entries = [(p_idx(t, u), float(m_u[l, u])) for u in range(U)]
            upper_entries.append((lpos_idx(t, l), -1.0))
            builder.add_row(
                upper_entries,
                lb=-np.inf,
                ub=float(line_cap[l] + c0_tl[t, l]),
            )

            lower_entries = [(p_idx(t, u), float(-m_u[l, u])) for u in range(U)]
            lower_entries.append((lneg_idx(t, l), -1.0))
            builder.add_row(
                lower_entries,
                lb=-np.inf,
                ub=float(-line_floor[l] - c0_tl[t, l]),
            )

    # Transition and startup / shutdown logic.
    for t in range(T):
        for u in range(U):
            prev_y_val = float(init_status[u]) if t == 0 else 0.0
            entries = [
                (y_idx(t, u), 1.0),
                (su_idx(t, u), -1.0),
                (sd_idx(t, u), 1.0),
            ]
            if t > 0:
                entries.append((y_idx(t - 1, u), -1.0))
                rhs = 0.0
            else:
                rhs = prev_y_val
            builder.add_row(entries, lb=rhs, ub=rhs)
            builder.add_row(
                [(su_idx(t, u), 1.0), (sd_idx(t, u), 1.0)],
                lb=-np.inf,
                ub=1.0,
            )

    if not ignore_min_updown:
        for u in range(U):
            if int(init_status[u]) == 1 and int(init_time[u]) < int(min_up[u]):
                remain = int(min_up[u] - init_time[u])
                for t in range(min(remain, T)):
                    builder.add_row([(y_idx(t, u), 1.0)], lb=1.0, ub=1.0)
            if int(init_status[u]) == 0 and int(init_time[u]) < int(min_down[u]):
                remain = int(min_down[u] - init_time[u])
                for t in range(min(remain, T)):
                    builder.add_row([(y_idx(t, u), 1.0)], lb=0.0, ub=0.0)

        for t in range(T):
            for u in range(U):
                up_entries = [(su_idx(k, u), 1.0) for k in range(max(0, t - int(min_up[u]) + 1), t + 1)]
                up_entries.append((y_idx(t, u), -1.0))
                builder.add_row(up_entries, lb=-np.inf, ub=0.0)

                down_entries = [(sd_idx(k, u), 1.0) for k in range(max(0, t - int(min_down[u]) + 1), t + 1)]
                down_entries.append((y_idx(t, u), 1.0))
                builder.add_row(down_entries, lb=-np.inf, ub=1.0)

    if not ignore_ramp:
        for u in range(U):
            builder.add_row(
                [(p_idx(0, u), 1.0)],
                lb=-np.inf,
                ub=float(init_dispatch[u] + ramp_up[u]),
            )
            builder.add_row(
                [(p_idx(0, u), -1.0)],
                lb=-np.inf,
                ub=float(ramp_down[u] - init_dispatch[u]),
            )
            for t in range(1, T):
                builder.add_row(
                    [(p_idx(t, u), 1.0), (p_idx(t - 1, u), -1.0)],
                    lb=-np.inf,
                    ub=float(ramp_up[u]),
                )
                builder.add_row(
                    [(p_idx(t - 1, u), 1.0), (p_idx(t, u), -1.0)],
                    lb=-np.inf,
                    ub=float(ramp_down[u]),
                )

    options = {"disp": False}
    if time_limit_s is not None:
        options["time_limit"] = float(time_limit_s)

    res = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=[builder.build(n_vars)],
        options=options,
    )

    if res.x is None:
        return {
            "solver_success": False,
            "solver_status": int(res.status),
            "solver_message": str(res.message),
            "zero_violation_feasible": False,
            "total_reserve_shortfall": math.inf,
            "total_thermal_cost": math.inf,
        }

    x = np.asarray(res.x, dtype=np.float64)
    unit_power_tu = np.array([[x[p_idx(t, u)] for u in range(U)] for t in range(T)])
    commitment_tu = np.array([[x[y_idx(t, u)] for u in range(U)] for t in range(T)])
    startup_tu = np.array([[x[su_idx(t, u)] for u in range(U)] for t in range(T)])
    reserve_slack_t = np.array([x[res_idx(t)] for t in range(T)])
    line_slack_pos_tl = np.array([[x[lpos_idx(t, l)] for l in range(L)] for t in range(T)])
    line_slack_neg_tl = np.array([[x[lneg_idx(t, l)] for l in range(L)] for t in range(T)])

    line_flow_tl = unit_power_tu @ m_u.T - c0_tl
    thermal_violation_tl = np.maximum(line_flow_tl - line_cap[None, :], 0.0) + np.maximum(
        line_floor[None, :] - line_flow_tl, 0.0
    )
    reserve_shortfall_t = np.maximum(required_cap_t - commitment_tu @ p_max, 0.0)
    unsafe_step = (thermal_violation_tl.sum(axis=1) > 1e-6) | (reserve_shortfall_t > 1e-6)

    total_startup_cost = float((startup_tu * startup_cost[None, :]).sum())
    total_no_load_cost = float((commitment_tu * no_load_cost[None, :]).sum())
    total_gen_cost = _total_gen_cost(unit_power_tu, case)

    return {
        "solver_success": bool(res.success),
        "solver_status": int(res.status),
        "solver_message": str(res.message),
        "objective_value": float(res.fun),
        "zero_violation_feasible": bool(
            reserve_shortfall_t.sum() <= 1e-6 and thermal_violation_tl.sum() <= 1e-6
        ),
        "total_gen_cost": total_gen_cost,
        "total_startup_cost": total_startup_cost,
        "total_no_load_cost": total_no_load_cost,
        "total_operating_cost": total_gen_cost + total_startup_cost + total_no_load_cost,
        "total_reserve_shortfall": float(reserve_shortfall_t.sum()),
        "total_thermal_cost": float(thermal_violation_tl.sum()),
        "reserve_shortfall_rate": float(np.mean(reserve_shortfall_t > 1e-6)),
        "thermal_violation_rate": float(np.mean(thermal_violation_tl.sum(axis=1) > 1e-6)),
        "feasibility_rate": float(1.0 - np.mean(unsafe_step)),
        "max_step_thermal_cost": float(thermal_violation_tl.sum(axis=1).max()),
        "max_step_reserve_shortfall": float(reserve_shortfall_t.max()),
        "line_slack_penalty_total": float(line_slack_pos_tl.sum() + line_slack_neg_tl.sum()),
        "reserve_slack_penalty_total": float(reserve_slack_t.sum()),
        "total_commitment_switches": float(np.abs(np.diff(commitment_tu, axis=0)).sum()),
        "ignore_ramp": bool(ignore_ramp),
        "ignore_min_updown": bool(ignore_min_updown),
    }


def run_probe(
    *,
    split: str,
    episodes: list[int],
    seed: int,
    ignore_ramp: bool,
    ignore_min_updown: bool,
    output_name: str,
    time_limit_s: float | None,
) -> Path:
    task_cfg = load_task_config(TASK_DIR)
    task = make_task_from_config(task_cfg)
    max_steps = int(task_cfg.get("max_steps", 48))
    n_eval_episodes = get_eval_episodes(task_cfg, {})

    rows: list[dict[str, float | int | bool]] = []
    for episode_idx in episodes:
        params = task.episode_params(
            split,
            episode_idx,
            n_eval_episodes,
            max_steps,
            strategy="uniform",
            seed=seed,
        )
        result = solve_episode_uc_milp(
            params,
            ignore_ramp=ignore_ramp,
            ignore_min_updown=ignore_min_updown,
            time_limit_s=time_limit_s,
        )
        row = {
            "episode_idx": int(episode_idx),
            "split": split,
            "seed": int(seed),
            **result,
        }
        rows.append(row)
        reserve = row.get("total_reserve_shortfall", math.inf)
        thermal = row.get("total_thermal_cost", math.inf)
        cost = row.get("total_operating_cost", math.inf)
        print(
            f"[MILP] ep={episode_idx} feasible={row['zero_violation_feasible']} "
            f"reserve={reserve:.6f} "
            f"thermal={thermal:.6f} "
            f"cost={cost:.1f}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{output_name}.json"
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="TSO UC feasibility probe via scipy MILP")
    parser.add_argument("--split", default="iid", choices=["train", "iid"])
    parser.add_argument("--episodes", default="0,38", help="Comma-separated episode indices")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ignore-ramp", action="store_true")
    parser.add_argument("--ignore-min-updown", action="store_true")
    parser.add_argument("--output-name", default="latest_milp_probe")
    parser.add_argument("--time-limit-s", type=float, default=120.0)
    args = parser.parse_args()

    episodes = [int(part) for part in args.episodes.split(",") if part.strip()]
    run_probe(
        split=args.split,
        episodes=episodes,
        seed=args.seed,
        ignore_ramp=args.ignore_ramp,
        ignore_min_updown=args.ignore_min_updown,
        output_name=args.output_name,
        time_limit_s=args.time_limit_s,
    )


if __name__ == "__main__":
    main()
