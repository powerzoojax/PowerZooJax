from types import SimpleNamespace

from benchmarks.common.reporting import (
    cross_backend_row_allowed,
    mark_dc_phase2_audited,
    summarize_group_evidence,
)


def test_legacy_manifest_rows_get_conservative_evidence_defaults():
    rec = SimpleNamespace(labels={}, env_info={})

    fields = summarize_group_evidence([rec])

    assert fields["evidence_tier"] == "legacy_unknown"
    assert fields["comparison_audited"] is False
    assert fields["audit_suite_version"] is None
    assert fields["parity_scope"] == []
    assert fields["suppress_reason"] is None


def test_cross_backend_rows_require_audit_version_to_enter_main_table():
    row = {
        "backend": "sb3",
        "device": "cuda",
        "algo": "ppo",
        "split": "iid",
        "comparison_audited": True,
        "audit_suite_version": None,
    }

    assert cross_backend_row_allowed(row) is False

    row["audit_suite_version"] = "audit_v1"
    assert cross_backend_row_allowed(row) is True


def test_dc_phase2_audit_can_promote_summary_row_without_mutating_results():
    row = {
        "backend": "sbx",
        "device": "cuda",
        "algo": "sbx_ppo",
        "split": "iid",
        "episode_reward_mean": -2400.0,
        "evidence_tier": "legacy_unknown",
        "comparison_audited": False,
    }

    mark_dc_phase2_audited(row, audit_available=True)

    assert row["episode_reward_mean"] == -2400.0
    assert row["evidence_tier"] == "official_main"
    assert row["comparison_audited"] is True
    assert row["audit_suite_version"] == "dc_microgrid_phase2_paper_audit_v1"
    assert "data_source" in row["parity_scope"]

