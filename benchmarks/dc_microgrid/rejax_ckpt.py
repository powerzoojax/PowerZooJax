"""Rejax SAC train-state I/O: cloudpickle cannot serialize ``SACState``; use Orbax."""

from __future__ import annotations

import shutil
from pathlib import Path

import jax
import orbax.checkpoint as ocp


def save_sac_train_state(orbax_dir: Path, state) -> None:
    orbax_dir = orbax_dir.resolve()
    if orbax_dir.exists():
        shutil.rmtree(orbax_dir)
    orbax_dir.parent.mkdir(parents=True, exist_ok=True)
    ocp.PyTreeCheckpointer().save(str(orbax_dir), state, force=True)


def load_sac_train_state(orbax_dir: Path, train_cfg, shaped_env, base_params):
    orbax_dir = orbax_dir.resolve()
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

    # Checkpoint may have been saved on GPU; restore explicitly onto local CPU.
    cpu_device = jax.local_devices(backend="cpu")[0]
    cpu_sharding = jax.sharding.SingleDeviceSharding(cpu_device)
    sharding_tree = jax.tree_util.tree_map(lambda _: cpu_sharding, template)
    restore_args = ocp.checkpoint_utils.construct_restore_args(template, sharding_tree)
    return ocp.PyTreeCheckpointer().restore(
        str(orbax_dir), item=template, restore_args=restore_args
    )
