"""Evidence-tier helpers for manifest-to-summary reporting."""

from __future__ import annotations

from typing import Any


LEGACY_EVIDENCE_TIER = "legacy_unknown"


def record_evidence_fields(record: Any) -> dict[str, Any]:
    """Return conservative evidence metadata for one RunRecord-like object."""
    labels = dict(getattr(record, "labels", None) or {})
    env_info = dict(getattr(record, "env_info", None) or {})

    evidence_tier = (
        getattr(record, "evidence_tier", None)
        or labels.get("evidence_tier")
        or env_info.get("evidence_tier")
        or LEGACY_EVIDENCE_TIER
    )
    comparison_audited = (
        getattr(record, "comparison_audited", None)
        if getattr(record, "comparison_audited", None) is not None
        else labels.get("comparison_audited", env_info.get("comparison_audited", False))
    )
    audit_suite_version = (
        getattr(record, "audit_suite_version", None)
        or labels.get("audit_suite_version")
        or env_info.get("audit_suite_version")
    )
    parity_scope = (
        getattr(record, "parity_scope", None)
        or labels.get("parity_scope")
        or env_info.get("parity_scope")
        or []
    )
    suppress_reason = (
        getattr(record, "suppress_reason", None)
        or labels.get("suppress_reason")
        or env_info.get("suppress_reason")
    )
    if isinstance(parity_scope, str):
        parity_scope = [p for p in parity_scope.split(",") if p]
    return {
        "evidence_tier": str(evidence_tier),
        "comparison_audited": bool(comparison_audited),
        "audit_suite_version": audit_suite_version,
        "parity_scope": parity_scope,
        "suppress_reason": suppress_reason,
    }


def summarize_group_evidence(records: list[Any]) -> dict[str, Any]:
    """Combine per-record evidence metadata for one summary row."""
    if not records:
        return {
            "evidence_tier": LEGACY_EVIDENCE_TIER,
            "comparison_audited": False,
            "audit_suite_version": None,
            "parity_scope": [],
            "suppress_reason": "empty_group",
        }
    fields = [record_evidence_fields(record) for record in records]
    tiers = {f["evidence_tier"] for f in fields}
    if len(tiers) == 1:
        tier = next(iter(tiers))
    elif "official_main" in tiers:
        tier = "mixed_includes_official"
    else:
        tier = "mixed_non_official"

    audit_versions = sorted(
        {f["audit_suite_version"] for f in fields if f["audit_suite_version"]}
    )
    parity_scope = sorted(
        {
            item
            for f in fields
            for item in (f.get("parity_scope") or [])
        }
    )
    suppress_reasons = sorted(
        {f["suppress_reason"] for f in fields if f.get("suppress_reason")}
    )
    comparison_audited = all(
        f["comparison_audited"] and f["audit_suite_version"] for f in fields
    )
    return {
        "evidence_tier": tier,
        "comparison_audited": bool(comparison_audited),
        "audit_suite_version": audit_versions[0] if len(audit_versions) == 1 else None,
        "parity_scope": parity_scope,
        "suppress_reason": ";".join(suppress_reasons) if suppress_reasons else None,
    }


def mark_dc_phase2_audited(row: dict[str, Any], *, audit_available: bool) -> None:
    """Promote already-audited DC Phase-2 rows at summary layer."""
    if not audit_available:
        return
    if row.get("split") != "iid":
        return
    if row.get("backend") not in {"jax_rejax", "sb3", "sbx"}:
        return
    row["evidence_tier"] = "official_main"
    row["comparison_audited"] = True
    row["audit_suite_version"] = "dc_microgrid_phase2_paper_audit_v1"
    row["parity_scope"] = [
        "case_overrides",
        "data_source",
        "reward_shaping",
        "split",
        "timesteps",
        "n_envs",
        "actor_family",
    ]
    row["suppress_reason"] = None


def cross_backend_row_allowed(row: dict[str, Any]) -> bool:
    """Return whether a row may enter a cross-backend main table."""
    return bool(row.get("comparison_audited") and row.get("audit_suite_version"))

