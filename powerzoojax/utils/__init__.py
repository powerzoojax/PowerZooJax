"""Shared lightweight utilities for PowerZooJax.

The public surface here is intentionally small:
- PRNG splitting for batched env execution
- `batch_reset` / `batch_step` helpers built on `jax.vmap`
- `scan_rollout` for fixed-length trajectory collection without Python loops
- common typing aliases used across env, task, and training code
"""

from powerzoojax.utils.jax_utils import (
    split_key_for_envs,
    batch_reset,
    batch_step,
    scan_rollout,
)
from powerzoojax.utils.typing import (
    PRNGKey,
    Array,
    Scalar,
)

__all__ = [
    "split_key_for_envs",
    "batch_reset",
    "batch_step",
    "scan_rollout",
    "PRNGKey",
    "Array",
    "Scalar",
]
