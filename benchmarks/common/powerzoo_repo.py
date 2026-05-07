"""Helpers for locating the sibling PowerZoo repository.

Cross-backend runs depend on importing ``powerzoo`` from a nearby checkout.
The default sibling name is ``PowerZoo``, but some local workspaces keep the
repo under a different stable directory such as ``PowerZoo.DEL``.  Resolve
that here so benchmark drivers and guardrail tests use the same lookup logic.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_SIBLING_NAMES: tuple[str, ...] = ("PowerZoo", "PowerZoo.DEL")


def _candidate_powerzoo_paths(repo_root: Path) -> list[Path]:
    candidates: list[Path] = []

    def _append(path: Path) -> None:
        if path not in candidates:
            candidates.append(path)

    override = os.environ.get("POWERZOO_DIR", "").strip()
    if override:
        _append(Path(override).expanduser())

    # Prefer the workspace-local vendor checkout (repo_root/PowerZoo) over any
    # sibling fallback such as ../PowerZoo.DEL so benchmark runs use the repo
    # that is versioned alongside the current PowerZooJax workspace.
    for base in (repo_root, repo_root.parent):
        for name in _SIBLING_NAMES:
            _append(base / name)

    return candidates


def find_powerzoo_repo(repo_root: Path) -> Path | None:
    """Return the resolved PowerZoo checkout path, or ``None`` if absent."""
    for candidate in _candidate_powerzoo_paths(repo_root):
        if (candidate / "powerzoo").is_dir():
            return candidate.resolve()
    return None


def ensure_powerzoo_on_path(repo_root: Path, *, append: bool = True) -> Path | None:
    """Resolve the sibling PowerZoo repo and add it to ``sys.path``."""
    path = find_powerzoo_repo(repo_root)
    if path is None:
        return None
    path_str = str(path)
    if path_str not in sys.path:
        if append:
            sys.path.append(path_str)
        else:
            sys.path.insert(0, path_str)
    return path
