"""Market Lite environments and clearing utilities.

This package contains the lightweight price-physics coupling layer used by the
benchmark's market-style tasks. It exposes:
- cost-based and bid-based market envs
- piecewise dispatch / SCED helpers
- the market MARL core used by GenCos-style experiments
- backward-compatible LMP alias names where older callers still expect them

These are benchmark environments, not a full market platform.
"""

from powerzoojax.envs.market.base import MarketState, MarketParams
from powerzoojax.envs.market.clearing import (
    CostSegments,
    PiecewiseEDSetup,
    PiecewiseEDResult,
    make_cost_segments,
    prepare_piecewise_ed,
    piecewise_ed,
)
from powerzoojax.envs.market.cost_based_market import (
    CostBasedMarketEnv,
    CostMarketState,
    CostMarketParams,
    make_cost_market_params,
)
from powerzoojax.envs.market.bid_based_market import (
    BidBasedMarketEnv,
    BidMarketState,
    BidMarketParams,
    make_bid_market_params,
)
from powerzoojax.envs.market.offer_sced import (
    OfferSCEDSetup,
    OfferSCEDResult,
    prepare_offer_sced,
    offer_sced,
)
from powerzoojax.envs.market.market_marl_core import (
    MarketMARLState,
    MarketMARLParams,
    make_market_marl_params,
    market_marl_reset,
    market_marl_step,
)

# Backward-compatible aliases
SimpleLMPArbitrageEnv = CostBasedMarketEnv
LMPMarketState = CostMarketState
LMPMarketParams = CostMarketParams
make_lmp_market_params = make_cost_market_params

__all__ = [
    # Base
    "MarketState",
    "MarketParams",
    # Clearing
    "CostSegments",
    "PiecewiseEDSetup",
    "PiecewiseEDResult",
    "make_cost_segments",
    "prepare_piecewise_ed",
    "piecewise_ed",
    # Cost-based market (new names)
    "CostBasedMarketEnv",
    "CostMarketState",
    "CostMarketParams",
    "make_cost_market_params",
    # Backward-compat aliases (deprecated)
    "SimpleLMPArbitrageEnv",
    "LMPMarketState",
    "LMPMarketParams",
    "make_lmp_market_params",
    # BidBased
    "BidBasedMarketEnv",
    "BidMarketState",
    "BidMarketParams",
    "make_bid_market_params",
    # Exact SCED solver
    "OfferSCEDSetup",
    "OfferSCEDResult",
    "prepare_offer_sced",
    "offer_sced",
    # GenCos Market MARL core
    "MarketMARLState",
    "MarketMARLParams",
    "make_market_marl_params",
    "market_marl_reset",
    "market_marl_step",
]
