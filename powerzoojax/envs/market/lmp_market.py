"""Backward-compatibility shim for lmp_market → cost_based_market.

Deprecated: import from ``powerzoojax.envs.market.cost_based_market`` instead.
"""
import warnings

warnings.warn(
    "powerzoojax.envs.market.lmp_market is deprecated; "
    "use powerzoojax.envs.market.cost_based_market instead.  "
    "Old names (SimpleLMPArbitrageEnv, LMPMarketState, LMPMarketParams, "
    "make_lmp_market_params) are re-exported here for backward compatibility.",
    DeprecationWarning,
    stacklevel=2,
)
from powerzoojax.envs.market.cost_based_market import (  # noqa: F401, E402
    CostBasedMarketEnv as SimpleLMPArbitrageEnv,
    CostMarketState as LMPMarketState,
    CostMarketParams as LMPMarketParams,
    make_cost_market_params as make_lmp_market_params,
)
