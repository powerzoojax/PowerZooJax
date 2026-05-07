"""Five-task benchmark recipe layer.

Each task module packages the benchmark-facing pieces for one task:
- parameter factories wired to the right case and data source
- rollout helpers and non-learning baselines
- task-specific metrics and evaluation helpers
- a `TaskSpec` implementation for experiment scripts

Physical environment dynamics remain in `powerzoojax.envs.*`; this package is
where benchmark semantics become runnable task recipes.
"""

from __future__ import annotations

import importlib
from typing import Any

# ---- TaskSpec Protocol ----
from powerzoojax.tasks.base import TaskSpec

# ---- TaskSpec implementations ----
from powerzoojax.tasks.dso import DSOTask
from powerzoojax.tasks.tso import TSOTask
from powerzoojax.tasks.ders import DERsTask
from powerzoojax.tasks.gencos import GencosTask
from powerzoojax.tasks.dc_microgrid import DCMicrogridTask

TASK_REGISTRY: dict[str, type] = {
    "dso":          DSOTask,
    "tso":          TSOTask,
    "ders":         DERsTask,
    "gencos":       GencosTask,
    "dc_microgrid": DCMicrogridTask,
}

# ---- DSO ----
from powerzoojax.tasks.dso import (
    DSO_FEEDER_BUS_MAP,
    DSO_FLEXLOAD_CONFIG,
    DSO_V_MIN,
    DSO_V_MAX,
    load_feeder_shape,
    load_dso_feeder_shapes,
    make_dso_flexload_bundle,
    make_dso_load_profiles,
    make_dso_params,
    make_dso_params_from_split,
    make_dso_1flex_params,
    make_dso_params_nonstationary,
    make_synthetic_feeder_shapes,
    rollout_dso,
    dso_no_control_rollout,
    dso_tou_rule_based_rollout,
    dso_droop_rule_based_rollout,
    dso_tou_heuristic_rollout,
    dso_droop_heuristic_rollout,
    compute_dso_metrics,
)

# ---- TSO ----
from powerzoojax.tasks.tso import (
    make_tso_net_load_profiles,
    make_tso_net_load_profiles_from_data,
    make_case14_with_uc_defaults,
    make_tso_case118_params,
    make_tso_case14_params,
    make_tso_ed_params,
    make_tso_uc_params,
    make_tso_scuc_params,
    tso_all_on_rollout,
    tso_merit_order_rollout,
    compute_tso_metrics,
    make_comparison_tso_load_trace,
    make_comparison_tso_params,
    TSO_COMPARISON_SCHEMA,
)

# ---- DERs ----
from powerzoojax.tasks.ders import (
    DERS_BENCHMARK_LOAD_CASE,
    DERS_BATTERY_BUSES,
    DERS_PV_BUSES,
    DERS_FLEXLOAD_BUSES,
    DERS_BATTERY_CONFIG,
    DERS_PV_CONFIG,
    DERS_FLEXLOAD_CONFIG,
    DERS_V_MIN,
    DERS_V_MAX,
    DERS_LARGE_BATTERY_BUSES,
    DERS_LARGE_PV_BUSES,
    DERS_LARGE_FLEXLOAD_BUSES,
    DERS_LARGE_V_MIN,
    DERS_LARGE_V_MAX,
    make_ders_battery_bundle,
    make_ders_pv_bundle,
    make_ders_flexload_bundle,
    make_ders_params,
    make_ders_params_from_split,
    make_ders_params_with_profiles,
    make_ders_marl_env,
    make_ders_large_params,
    make_ders_large_marl_env,
    make_ders_ood_params,
    make_ders_3phase_eval,
    load_ders_load_shape,
    rollout_ders,
    ders_no_control_rollout,
    ders_volt_droop_rollout,
    agent_dropout_rollout,
    compute_ders_metrics,
    compute_ders_safety_metrics,
)

# ---- GenCos ----
from powerzoojax.tasks.gencos import (
    load_gencos_profiles,
    make_gencos_params,
    make_gencos_env,
    compute_gencos_metrics,
)

# ---- DC Microgrid ----
from powerzoojax.tasks.dc_microgrid import (
    make_dcmicrogrid_params,
    make_dcmicrogrid_params_with_profiles,
    DataCenterMicrogridEnv,
    DCMicrogridParams,
    DCMicrogridState,
    compute_dcmicrogrid_metrics,
)

__all__ = [
    # TaskSpec
    "TaskSpec",
    "DSOTask",
    "TSOTask",
    "DERsTask",
    "GencosTask",
    "DCMicrogridTask",
    "TASK_REGISTRY",
    # DSO
    "DSO_FEEDER_BUS_MAP",
    "DSO_FLEXLOAD_CONFIG",
    "DSO_V_MIN",
    "DSO_V_MAX",
    "load_feeder_shape",
    "load_dso_feeder_shapes",
    "make_dso_flexload_bundle",
    "make_dso_load_profiles",
    "make_dso_params",
    "make_dso_params_from_split",
    "make_dso_1flex_params",
    "make_dso_params_nonstationary",
    "make_synthetic_feeder_shapes",
    "rollout_dso",
    "dso_no_control_rollout",
    "dso_tou_rule_based_rollout",
    "dso_droop_rule_based_rollout",
    "dso_tou_heuristic_rollout",
    "dso_droop_heuristic_rollout",
    "compute_dso_metrics",
    # TSO
    "make_tso_net_load_profiles",
    "make_tso_net_load_profiles_from_data",
    "make_case14_with_uc_defaults",
    "make_tso_case118_params",
    "make_tso_case14_params",
    "make_tso_ed_params",
    "make_tso_uc_params",
    "make_tso_scuc_params",
    "tso_all_on_rollout",
    "tso_merit_order_rollout",
    "compute_tso_metrics",
    "make_comparison_tso_load_trace",
    "make_comparison_tso_params",
    "TSO_COMPARISON_SCHEMA",
    # DERs
    "DERS_BENCHMARK_LOAD_CASE",
    "DERS_BATTERY_BUSES",
    "DERS_PV_BUSES",
    "DERS_FLEXLOAD_BUSES",
    "DERS_BATTERY_CONFIG",
    "DERS_PV_CONFIG",
    "DERS_FLEXLOAD_CONFIG",
    "DERS_V_MIN",
    "DERS_V_MAX",
    "DERS_LARGE_BATTERY_BUSES",
    "DERS_LARGE_PV_BUSES",
    "DERS_LARGE_FLEXLOAD_BUSES",
    "DERS_LARGE_V_MIN",
    "DERS_LARGE_V_MAX",
    "make_ders_battery_bundle",
    "make_ders_pv_bundle",
    "make_ders_flexload_bundle",
    "make_ders_params",
    "make_ders_params_from_split",
    "make_ders_params_with_profiles",
    "make_ders_marl_env",
    "make_ders_large_params",
    "make_ders_large_marl_env",
    "make_ders_ood_params",
    "make_ders_3phase_eval",
    "load_ders_load_shape",
    "rollout_ders",
    "ders_no_control_rollout",
    "ders_volt_droop_rollout",
    "agent_dropout_rollout",
    "compute_ders_metrics",
    "compute_ders_safety_metrics",
    # GenCos
    "load_gencos_profiles",
    "make_gencos_params",
    "make_gencos_env",
    "compute_gencos_metrics",
    # DC Microgrid
    "make_dcmicrogrid_params",
    "make_dcmicrogrid_params_with_profiles",
    "DataCenterMicrogridEnv",
    "DCMicrogridParams",
    "DCMicrogridState",
    "compute_dcmicrogrid_metrics",
]
