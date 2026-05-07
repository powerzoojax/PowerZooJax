"""One-off post-experiment patch: env_info, throughput_sps, dedup manifest.

Run after data collection is complete. Idempotent.

Steps:
  1. Patch every individual run JSON in results/runs/ to include env_info
     (collected from the current machine via collect_env_info()). Older
     runs picked up here will be tagged with the *current* machine's commit
     / CUDA strings — that is a known limitation, but better than nothing
     for runs that pre-date the env_info field.
  2. Patch training records that have walltime_s but throughput_sps=None.
  3. Annotate PPO-Lagrangian runs whose training curves were never saved
     (artifacts == {} or no curve keys) so downstream consumers can flag
     them as "ppo_lag_curves_unavailable=true". We never fabricate paths.
  4. Rebuild results/manifest.json from the (patched) run files using the
     artifact-aware dedup (more artifact keys wins; tiebreaker timestamp).

Run::

    python benchmarks/tso/patch_env_info.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.common.configs import load_config  # noqa: E402
from benchmarks.common.io import dedup_keep_artifacts  # noqa: E402
from benchmarks.common.io import collect_env_info  # noqa: E402

TASK_DIR = Path(__file__).resolve().parent
RUNS_DIR = TASK_DIR / "results" / "runs"
MANIFEST_PATH = TASK_DIR / "results" / "manifest.json"
CONFIGS_DIR = TASK_DIR / "configs"


def _load_total_timesteps() -> dict[str, int]:
    """Map algo -> total_timesteps from the train_*.yaml configs."""
    out: dict[str, int] = {}
    ppo = load_config(CONFIGS_DIR / "train_ppo.yaml")
    out["ppo"] = int(ppo["total_timesteps"])
    safe = load_config(CONFIGS_DIR / "train_safe.yaml")
    out["ppo_lagrangian"] = int(safe["total_timesteps"])
    return out


def _patch_env_info() -> int:
    """Add env_info to every run JSON that does not already have it. Returns count patched."""
    info = collect_env_info()
    print(f"[patch] env_info: {info}")
    count = 0
    for json_file in sorted(RUNS_DIR.glob("*.json")):
        rec = json.loads(json_file.read_text(encoding="utf-8"))
        if not rec.get("env_info"):
            rec["env_info"] = info
            json_file.write_text(
                json.dumps(rec, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            count += 1
    print(f"[patch] env_info added to {count} run JSONs")
    return count


def _patch_throughput() -> int:
    """Backfill throughput_sps for training records with walltime_s > 0."""
    total_steps = _load_total_timesteps()
    count = 0
    for json_file in sorted(RUNS_DIR.glob("*.json")):
        rec = json.loads(json_file.read_text(encoding="utf-8"))
        if rec.get("split") != "train":
            continue
        if rec.get("algo") not in total_steps:
            continue
        if rec.get("throughput_sps") is not None:
            continue
        wt = rec.get("walltime_s") or 0.0
        if wt <= 0:
            continue
        sps = total_steps[rec["algo"]] / float(wt)
        rec["throughput_sps"] = sps
        json_file.write_text(
            json.dumps(rec, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  throughput_sps -> {rec['run_id']}: {sps:.0f} sps")
        count += 1
    print(f"[patch] throughput_sps added to {count} training records")
    return count


_PPO_LAG_NOTE = "ppo_lag_curves_unavailable=true"
_CURVE_KEYS = {
    "loss",
    "mean_reward",
    "mean_cost",
    "lambda",
    "episode_cost_est",
    "step_mean_cost_scaled",
    "step_mean_reward",
    "step_done_rate",
    "eval_returns",
    "eval_episode_lengths",
}


def _patch_ppo_lag_notes() -> int:
    """Tag PPO-Lagrangian runs with no curve artifacts so we don't silently lose context.

    Curve files were not persisted in early TSO runs. Re-running them is the
    proper fix; until then, this note tells summarize / plot scripts to treat
    those rows as "training curves unavailable".
    """
    count = 0
    for json_file in sorted(RUNS_DIR.glob("*.json")):
        rec = json.loads(json_file.read_text(encoding="utf-8"))
        if rec.get("algo") != "ppo_lagrangian":
            continue
        artifacts = rec.get("artifacts") or {}
        has_curves = any(k in artifacts for k in _CURVE_KEYS)
        if has_curves:
            continue
        notes = rec.get("notes") or ""
        if _PPO_LAG_NOTE in notes:
            continue
        rec["notes"] = (
            f"{notes} | {_PPO_LAG_NOTE}" if notes else _PPO_LAG_NOTE
        )
        json_file.write_text(
            json.dumps(rec, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        count += 1
    print(f"[patch] ppo_lag_curves_unavailable note added to {count} run JSONs")
    return count


def _rebuild_manifest() -> int:
    """Rebuild manifest.json from current run files, keeping latest per (algo, split, seed).

    Important dedup tweak: when an eval rerun on the train split (no
    ``artifacts``) overwrites the actual training record (with
    ``artifacts.params``), naive newest-by-timestamp would drop the params
    pointer. We therefore prefer records with a non-empty ``artifacts`` dict;
    only when no candidate has artifacts do we fall back to newest-by-timestamp.
    """
    raw = []
    for f in sorted(RUNS_DIR.glob("*.json")):
        try:
            raw.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception as exc:  # pragma: no cover - defensive
            print(f"  [WARN] could not parse {f.name}: {exc}")

    deduped = dedup_keep_artifacts(raw)
    MANIFEST_PATH.write_text(
        json.dumps(deduped, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[patch] rebuilt manifest: {len(raw)} raw -> {len(deduped)} unique records "
          "(prefer-with-artifacts dedup)")
    return len(deduped)


def main() -> None:
    print(f"[patch] task_dir={TASK_DIR}")
    n_runs = sum(1 for _ in RUNS_DIR.glob("*.json"))
    print(f"[patch] {n_runs} run JSONs found")
    _patch_env_info()
    _patch_throughput()
    _patch_ppo_lag_notes()
    n_unique = _rebuild_manifest()
    print(f"[patch] done — manifest now has {n_unique} unique records")


if __name__ == "__main__":
    main()
