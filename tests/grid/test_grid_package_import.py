"""L0: package import path for ``powerzoojax.envs.grid``."""

import subprocess
import sys


def test_grid_pkg_import_lazy_dist_not_loaded_until_access():
    """``import powerzoojax.envs.grid`` must not load dist before first use."""
    code = r"""
import sys
import powerzoojax.envs.grid as g
assert "powerzoojax.envs.grid.dist" not in sys.modules
_ = g.DistGridEnv
assert "powerzoojax.envs.grid.dist" in sys.modules
"""
    r = subprocess.run(
        [sys.executable, "-c", code],
        cwd=None,
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr + r.stdout


def test_grid_pkg_lazy_symbols_match_submodules():
    import powerzoojax.envs.grid as g
    from powerzoojax.envs.grid import dist as dist_mod
    from powerzoojax.envs.grid import dist_3phase as ph_mod

    assert g.DistGridEnv is dist_mod.DistGridEnv
    assert g.make_dist_3phase_params is ph_mod.make_dist_3phase_params
