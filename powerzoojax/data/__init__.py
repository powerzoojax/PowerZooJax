"""Data layer for benchmark profiles and split logic.

This package handles setup-time data work only:
- manifest-driven parquet loading
- time alignment and windowing
- real-data split definitions for GB / Ausgrid-style experiments
- controlled nonstationarity / OOD transforms
- DC microgrid profile loading and synthetic fallbacks

Outputs are regular arrays ready to be packed into `EnvParams`; nothing in this
package is meant to run inside jitted training inner loops.
"""

from .data_loader import DataLoader
from . import signals
from .manifest import DatasetManifest
from .registry import DatasetRegistry
from .alignment import TimeAligner
from . import splits
from .splits import (
    GB_TRAIN_START,
    GB_TRAIN_END,
    GB_IID_START,
    GB_IID_END,
    AUSGRID_TRAIN_START,
    AUSGRID_TRAIN_END,
    AUSGRID_IID_START,
    AUSGRID_IID_END,
    AUSGRID_SUMMER_START,
    AUSGRID_SUMMER_END,
    gb_windows,
    ausgrid_windows,
)
from . import ausgrid_utils
from .ausgrid_utils import (
    AUSGRID_FEEDER_POOLS,
    AUSGRID_TRAIN_POOL,
    AUSGRID_ZONE_HOLDOUT_POOL,
    get_ausgrid_split,
    get_feeder_substations,
    select_full_coverage_substations,
)
from .nonstationary import (
    EpisodeConfig,
    NonstationarySampler,
    apply_drift,
)
from . import dc_microgrid_profiles
from .dc_microgrid_profiles import (
    cycle_profile,
    make_synthetic_cpu_profile,
    make_synthetic_solar_profile,
    make_real_solar_profile,
    make_synthetic_outdoor_temp_profile,
    make_all_synthetic_profiles,
    load_workload_profiles,
    apply_ood_transform,
    VALID_SOURCES,
    VALID_OOD_SCENARIOS,
)

__all__ = [
    "DataLoader",
    "signals",
    "DatasetManifest",
    "DatasetRegistry",
    "TimeAligner",
    "splits",
    "GB_TRAIN_START",
    "GB_TRAIN_END",
    "GB_IID_START",
    "GB_IID_END",
    "AUSGRID_TRAIN_START",
    "AUSGRID_TRAIN_END",
    "AUSGRID_IID_START",
    "AUSGRID_IID_END",
    "AUSGRID_SUMMER_START",
    "AUSGRID_SUMMER_END",
    "gb_windows",
    "ausgrid_windows",
    "ausgrid_utils",
    "AUSGRID_FEEDER_POOLS",
    "AUSGRID_TRAIN_POOL",
    "AUSGRID_ZONE_HOLDOUT_POOL",
    "get_ausgrid_split",
    "get_feeder_substations",
    "select_full_coverage_substations",
    "EpisodeConfig",
    "NonstationarySampler",
    "apply_drift",
    # DC Microgrid profiles (C4)
    "dc_microgrid_profiles",
    "cycle_profile",
    "make_synthetic_cpu_profile",
    "make_synthetic_solar_profile",
    "make_real_solar_profile",
    "make_synthetic_outdoor_temp_profile",
    "make_all_synthetic_profiles",
    "load_workload_profiles",
    "apply_ood_transform",
    "VALID_SOURCES",
    "VALID_OOD_SCENARIOS",
]
