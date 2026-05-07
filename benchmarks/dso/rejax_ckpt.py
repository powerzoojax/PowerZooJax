"""DSO Rejax SAC train-state I/O.

Orbax chokes on the DSO SAC state because the wrapped env carries a few
legitimate zero-size arrays (e.g. empty battery SOC placeholders). Flax's
native bytes serialization handles the same pytree cleanly, so DSO uses that
path for SAC checkpoints.
"""

from __future__ import annotations

from pathlib import Path

import jax
from flax import serialization


def save_sac_train_state(path: Path, state) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialization.to_bytes(state))


def load_sac_train_state(path: Path, train_cfg, shaped_env, base_params):
    import rejax
    from powerzoojax.rl.trainer import _RejaxAdapter, _rejax_create_kwargs
    from powerzoojax.rl.wrappers import LogWrapper

    adapted = _RejaxAdapter(LogWrapper(shaped_env, base_params))
    sac = rejax.SAC.create(
        env=adapted,
        env_params=None,
        **_rejax_create_kwargs(rejax.SAC, train_cfg),
    )
    template = sac.init_state(jax.random.PRNGKey(0))
    return serialization.from_bytes(template, path.read_bytes())
