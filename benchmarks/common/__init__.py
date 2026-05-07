"""Small shared helpers for benchmark train/eval/summarize scripts.

Keep this package intentionally narrow. Task-specific metric logic and
one-off experiment operations should stay outside the import surface that
day-to-day benchmark scripts need.
"""

from benchmarks.common.io import RunRecord, load_manifest, load_run, save_run

__all__ = ["RunRecord", "save_run", "load_run", "load_manifest"]
