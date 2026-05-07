"""Grid environment entrypoint plus grid-adjacent benchmark helpers.

This package exposes the transmission/distribution env core together with
power-flow / OPF utilities. For convenience it also re-exports the task-layer
grid factories and rollout helpers used by DSO, DERs, and TSO benchmarks.

Lazy imports keep heavyweight modules and optional dependencies cold until a
symbol is actually requested.
"""

from __future__ import annotations

import importlib
from typing import Any

from powerzoojax.envs.grid.base import GridState, GridParams
from powerzoojax.envs.grid.power_flow import (
    dc_power_flow,
    dc_power_flow_with_check,
    safety_check,
    compute_generation_cost,
    proportional_dispatch,
)
from powerzoojax.envs.grid.ac_power_flow import (
    ACPFSetup,
    ACPFResult,
    prepare_acpf,
    ac_power_flow,
    calc_branch_flows,
    ac_power_flow_with_check,
)
from powerzoojax.envs.grid.dc_opf import (
    DCOPFSetup,
    DCOPFResult,
    prepare_dcopf,
    dc_opf,
)
from powerzoojax.envs.grid.trans import (
    TransGridEnv,
    TransGridState,
    TransGridParams,
    make_trans_params,
)

_GRID_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # ac_opf requires jaxopt — lazy so DSO/DistGrid tests don't fail when jaxopt absent
    "ACOPFSetup": ("powerzoojax.envs.grid.ac_opf", "ACOPFSetup"),
    "ACOPFResult": ("powerzoojax.envs.grid.ac_opf", "ACOPFResult"),
    "ACOPFBenchmarkConfig": ("powerzoojax.envs.grid.ac_opf", "ACOPFBenchmarkConfig"),
    "ACOPFBenchmarkRow": ("powerzoojax.envs.grid.ac_opf", "ACOPFBenchmarkRow"),
    "ACOPFDiagnosisReport": ("powerzoojax.envs.grid.ac_opf", "ACOPFDiagnosisReport"),
    "benchmark_acopf_configs": ("powerzoojax.envs.grid.ac_opf", "benchmark_acopf_configs"),
    "compute_acopf_pf_residual_from_result": ("powerzoojax.envs.grid.ac_opf", "compute_acopf_pf_residual_from_result"),
    "compute_acopf_slack_balance_residual_from_result": ("powerzoojax.envs.grid.ac_opf", "compute_acopf_slack_balance_residual_from_result"),
    "diagnose_acopf_vs_golden": ("powerzoojax.envs.grid.ac_opf", "diagnose_acopf_vs_golden"),
    "format_acopf_diagnosis_report": ("powerzoojax.envs.grid.ac_opf", "format_acopf_diagnosis_report"),
    "get_last_acopf_alm_balance_state": ("powerzoojax.envs.grid.ac_opf", "get_last_acopf_alm_balance_state"),
    "make_acopf_benchmark_grid": ("powerzoojax.envs.grid.ac_opf", "make_acopf_benchmark_grid"),
    "prepare_acopf": ("powerzoojax.envs.grid.ac_opf", "prepare_acopf"),
    "ac_opf": ("powerzoojax.envs.grid.ac_opf", "ac_opf"),
    "DistGridEnv": ("powerzoojax.envs.grid.dist", "DistGridEnv"),
    "DistGridState": ("powerzoojax.envs.grid.dist", "DistGridState"),
    "DistGridParams": ("powerzoojax.envs.grid.dist", "DistGridParams"),
    "make_dist_params": ("powerzoojax.envs.grid.dist", "make_dist_params"),
    "DistGrid3PhaseEnv": ("powerzoojax.envs.grid.dist_3phase", "DistGrid3PhaseEnv"),
    "DistGrid3PhState": ("powerzoojax.envs.grid.dist_3phase", "DistGrid3PhState"),
    "DistGrid3PhParams": ("powerzoojax.envs.grid.dist_3phase", "DistGrid3PhParams"),
    "make_dist_3phase_params": ("powerzoojax.envs.grid.dist_3phase", "make_dist_3phase_params"),
    "DSO_FEEDER_BUS_MAP": ("powerzoojax.tasks.dso", "DSO_FEEDER_BUS_MAP"),
    "DSO_FLEXLOAD_CONFIG": ("powerzoojax.tasks.dso", "DSO_FLEXLOAD_CONFIG"),
    "DSO_V_MIN": ("powerzoojax.tasks.dso", "DSO_V_MIN"),
    "DSO_V_MAX": ("powerzoojax.tasks.dso", "DSO_V_MAX"),
    "make_dso_flexload_bundle": ("powerzoojax.tasks.dso", "make_dso_flexload_bundle"),
    "make_dso_load_profiles": ("powerzoojax.tasks.dso", "make_dso_load_profiles"),
    "make_dso_params": ("powerzoojax.tasks.dso", "make_dso_params"),
    "make_synthetic_feeder_shapes": ("powerzoojax.tasks.dso", "make_synthetic_feeder_shapes"),
    "load_feeder_shape": ("powerzoojax.tasks.dso", "load_feeder_shape"),
    "load_dso_feeder_shapes": ("powerzoojax.tasks.dso", "load_dso_feeder_shapes"),
    "make_dso_params_from_split": ("powerzoojax.tasks.dso", "make_dso_params_from_split"),
    "make_dso_1flex_params": ("powerzoojax.tasks.dso", "make_dso_1flex_params"),
    "make_dso_params_nonstationary": ("powerzoojax.tasks.dso", "make_dso_params_nonstationary"),
    "rollout_dso": ("powerzoojax.tasks.dso", "rollout_dso"),
    "dso_no_control_rollout": ("powerzoojax.tasks.dso", "dso_no_control_rollout"),
    "dso_tou_rule_based_rollout": ("powerzoojax.tasks.dso", "dso_tou_rule_based_rollout"),
    "dso_droop_rule_based_rollout": ("powerzoojax.tasks.dso", "dso_droop_rule_based_rollout"),
    "dso_tou_heuristic_rollout": ("powerzoojax.tasks.dso", "dso_tou_heuristic_rollout"),
    "dso_droop_heuristic_rollout": ("powerzoojax.tasks.dso", "dso_droop_heuristic_rollout"),
    "compute_dso_metrics": ("powerzoojax.tasks.dso", "compute_dso_metrics"),
    # DERs task
    "DERS_BATTERY_BUSES": ("powerzoojax.tasks.ders", "DERS_BATTERY_BUSES"),
    "DERS_PV_BUSES": ("powerzoojax.tasks.ders", "DERS_PV_BUSES"),
    "DERS_FLEXLOAD_BUSES": ("powerzoojax.tasks.ders", "DERS_FLEXLOAD_BUSES"),
    "DERS_BATTERY_CONFIG": ("powerzoojax.tasks.ders", "DERS_BATTERY_CONFIG"),
    "DERS_PV_CONFIG": ("powerzoojax.tasks.ders", "DERS_PV_CONFIG"),
    "DERS_FLEXLOAD_CONFIG": ("powerzoojax.tasks.ders", "DERS_FLEXLOAD_CONFIG"),
    "DERS_V_MIN": ("powerzoojax.tasks.ders", "DERS_V_MIN"),
    "DERS_V_MAX": ("powerzoojax.tasks.ders", "DERS_V_MAX"),
    "make_ders_battery_bundle": ("powerzoojax.tasks.ders", "make_ders_battery_bundle"),
    "make_ders_pv_bundle": ("powerzoojax.tasks.ders", "make_ders_pv_bundle"),
    "make_ders_flexload_bundle": ("powerzoojax.tasks.ders", "make_ders_flexload_bundle"),
    "make_ders_params": ("powerzoojax.tasks.ders", "make_ders_params"),
    "make_ders_params_from_split": ("powerzoojax.tasks.ders", "make_ders_params_from_split"),
    "make_ders_params_with_profiles": ("powerzoojax.tasks.ders", "make_ders_params_with_profiles"),
    "make_ders_marl_env": ("powerzoojax.tasks.ders", "make_ders_marl_env"),
    "rollout_ders": ("powerzoojax.tasks.ders", "rollout_ders"),
    "ders_no_control_rollout": ("powerzoojax.tasks.ders", "ders_no_control_rollout"),
    "ders_volt_droop_rollout": ("powerzoojax.tasks.ders", "ders_volt_droop_rollout"),
    "compute_ders_metrics": ("powerzoojax.tasks.ders", "compute_ders_metrics"),
    "compute_ders_safety_metrics": ("powerzoojax.tasks.ders", "compute_ders_safety_metrics"),
    # R5 — DERs-large + OOD
    "DERS_LARGE_BATTERY_BUSES": ("powerzoojax.tasks.ders", "DERS_LARGE_BATTERY_BUSES"),
    "DERS_LARGE_PV_BUSES": ("powerzoojax.tasks.ders", "DERS_LARGE_PV_BUSES"),
    "DERS_LARGE_FLEXLOAD_BUSES": ("powerzoojax.tasks.ders", "DERS_LARGE_FLEXLOAD_BUSES"),
    "DERS_LARGE_V_MIN": ("powerzoojax.tasks.ders", "DERS_LARGE_V_MIN"),
    "DERS_LARGE_V_MAX": ("powerzoojax.tasks.ders", "DERS_LARGE_V_MAX"),
    "make_ders_large_params": ("powerzoojax.tasks.ders", "make_ders_large_params"),
    "make_ders_large_marl_env": ("powerzoojax.tasks.ders", "make_ders_large_marl_env"),
    "make_ders_ood_params": ("powerzoojax.tasks.ders", "make_ders_ood_params"),
    "make_ders_3phase_eval": ("powerzoojax.tasks.ders", "make_ders_3phase_eval"),
    "agent_dropout_rollout": ("powerzoojax.tasks.ders", "agent_dropout_rollout"),
    # TSO / Unit Commitment task
    "UCState": ("powerzoojax.envs.grid.unit_commitment", "UCState"),
    "UCParams": ("powerzoojax.envs.grid.unit_commitment", "UCParams"),
    "UnitCommitmentEnv": ("powerzoojax.envs.grid.unit_commitment", "UnitCommitmentEnv"),
    "make_uc_params": ("powerzoojax.envs.grid.unit_commitment", "make_uc_params"),
    "make_tso_net_load_profiles": ("powerzoojax.tasks.tso", "make_tso_net_load_profiles"),
    "make_tso_net_load_profiles_from_data": ("powerzoojax.tasks.tso", "make_tso_net_load_profiles_from_data"),
    "make_case14_with_uc_defaults": ("powerzoojax.tasks.tso", "make_case14_with_uc_defaults"),
    "make_tso_case118_params": ("powerzoojax.tasks.tso", "make_tso_case118_params"),
    "make_tso_case14_params": ("powerzoojax.tasks.tso", "make_tso_case14_params"),
    "make_tso_ed_params": ("powerzoojax.tasks.tso", "make_tso_ed_params"),
    "make_tso_uc_params": ("powerzoojax.tasks.tso", "make_tso_uc_params"),
    "make_tso_scuc_params": ("powerzoojax.tasks.tso", "make_tso_scuc_params"),
    "tso_all_on_rollout": ("powerzoojax.tasks.tso", "tso_all_on_rollout"),
    "tso_merit_order_rollout": ("powerzoojax.tasks.tso", "tso_merit_order_rollout"),
    "compute_tso_metrics": ("powerzoojax.tasks.tso", "compute_tso_metrics"),
    "make_comparison_tso_load_trace": ("powerzoojax.tasks.tso", "make_comparison_tso_load_trace"),
    "make_comparison_tso_params": ("powerzoojax.tasks.tso", "make_comparison_tso_params"),
    "TSO_COMPARISON_SCHEMA": ("powerzoojax.tasks.tso", "TSO_COMPARISON_SCHEMA"),
}


def __getattr__(name: str) -> Any:
    if name in _GRID_LAZY_IMPORTS:
        mod_path, attr = _GRID_LAZY_IMPORTS[name]
        mod = importlib.import_module(mod_path)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_EAGER_EXPORTS = [
    "GridState",
    "GridParams",
    "dc_power_flow",
    "dc_power_flow_with_check",
    "safety_check",
    "compute_generation_cost",
    "proportional_dispatch",
    "ACPFSetup",
    "ACPFResult",
    "prepare_acpf",
    "ac_power_flow",
    "calc_branch_flows",
    "ac_power_flow_with_check",
    "DCOPFSetup",
    "DCOPFResult",
    "prepare_dcopf",
    "dc_opf",
    "TransGridEnv",
    "TransGridState",
    "TransGridParams",
    "make_trans_params",
]

__all__ = [*_EAGER_EXPORTS, *_GRID_LAZY_IMPORTS.keys()]


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
