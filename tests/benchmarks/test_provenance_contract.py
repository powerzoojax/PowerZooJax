import pytest

from benchmarks.common import io


def test_jax_contract_records_declared_and_actual_fields(monkeypatch):
    monkeypatch.setattr(
        io,
        "collect_env_info",
        lambda: {
            "jax_backend": "gpu",
            "jax_device_kind": "NVIDIA Test GPU",
            "cuda": "NVIDIA Test GPU",
        },
    )

    device, env_info, labels = io.collect_jax_run_contract(
        requested_device="gpu",
        context="unit-test",
        declared_backend="jax_rejax",
    )

    assert device == "gpu"
    assert env_info["declared_backend"] == "jax_rejax"
    assert env_info["actual_backend"] == "gpu"
    assert env_info["declared_device"] == "gpu"
    assert env_info["actual_device"] == "gpu"
    assert env_info["actual_device_kind"] == "NVIDIA Test GPU"
    assert env_info["device_contract_ok"] == "true"
    assert labels["device_contract_ok"] is True


def test_jax_contract_hard_fails_gpu_cpu_fallback(monkeypatch):
    monkeypatch.setattr(
        io,
        "collect_env_info",
        lambda: {
            "jax_backend": "cpu",
            "jax_device_kind": "cpu",
            "cuda": "n/a (cpu)",
        },
    )

    with pytest.raises(RuntimeError, match="requested device='gpu'"):
        io.collect_jax_run_contract(
            requested_device="gpu",
            context="unit-test",
        )


def test_dataset_provenance_records_real_data_fingerprint():
    meta = io.collect_dataset_provenance(
        task="dc_microgrid",
        task_config={"data_source": "google", "max_steps": 288},
        split="iid",
    )

    assert meta["data_provenance_ok"] is True
    assert meta["synthetic_fallback_used"] is False
    assert meta["dataset_resolved_path"]
    assert "google_dc_2019" in meta["dataset_checksum"]
    assert len(meta["dataset_fingerprint"]) == 64
    assert meta["split_ids"] == "iid"


def test_dataset_provenance_hard_fails_synthetic_fallback():
    with pytest.raises(RuntimeError, match="synthetic_fallback_used"):
        io.collect_dataset_provenance(
            task="dc_microgrid",
            task_config={"data_source": "google", "synthetic_fallback_used": True},
            split="iid",
        )

