"""RL/MARL integration layer for PowerZooJax.

This package sits above the pure-functional env core and contains:
- wrapper adapters for single-agent and safe RL interfaces
- multi-agent adapters for grid and market-style environments
- training config / preset utilities
- trainer entrypoints used by the CLI and benchmark scripts

The rule of thumb is simple: physics belongs in `powerzoojax.envs`, while
library-facing training glue belongs here.
"""

from powerzoojax.rl.wrappers import (
    LogEnvState,
    LogWrapper,
    SafeRLState,
    SafeRLWrapper,
    SauteWrapper,
    bind,
)

from powerzoojax.rl.multi_agent import (
    MultiAgentEnvironment,
    MARLState,
    GridMARLEnv,
    DistGridMARLEnv,
)

from powerzoojax.rl.reward import RewardWrapper, RewardEnvState
from powerzoojax.rl.config import TrainConfig, load_config, save_config
from powerzoojax.rl.trainer import TrainResult, make_train
from powerzoojax.rl.cmdp import make_cmdp_train
from powerzoojax.rl.ippo import (
    make_ippo_train,
    make_ippo_typed_lagrangian_train,
    make_ippo_act,
)
from powerzoojax.rl.train import train
from powerzoojax.rl.presets import list_presets, get_preset
from powerzoojax.rl.market_marl import MarketMARLEnv

__all__ = [
    # Wrappers
    "LogEnvState",
    "LogWrapper",
    "SafeRLState",
    "SafeRLWrapper",
    "SauteWrapper",
    "bind",
    # Custom reward
    "RewardEnvState",
    "RewardWrapper",
    # Multi-agent
    "MultiAgentEnvironment",
    "MARLState",
    "GridMARLEnv",
    "DistGridMARLEnv",
    # Training config
    "TrainConfig",
    "load_config",
    "save_config",
    # Training
    "TrainResult",
    "make_train",
    "make_cmdp_train",
    "make_ippo_train",
    "make_ippo_typed_lagrangian_train",
    "make_ippo_act",
    "train",
    # Presets
    "list_presets",
    "get_preset",
    # GenCos Market MARL
    "MarketMARLEnv",
]
