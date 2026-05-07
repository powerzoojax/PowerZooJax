"""Tests for powerzoojax.data.registry — DatasetRegistry discovery."""

from pathlib import Path

import pytest

from powerzoojax.data.registry import DatasetRegistry
from powerzoojax.data import signals as S

MANIFEST_DIR = (
    Path(__file__).resolve().parents[2] / "powerzoojax/data/manifests"
)


@pytest.fixture(scope="module")
def registry():
    """Registry that reads the bundled manifests/ directory."""
    return DatasetRegistry(MANIFEST_DIR)


class TestDatasetRegistry:

    def test_discovers_all_manifests(self, registry):
        datasets = registry.list_datasets()
        manifest_files = sorted(MANIFEST_DIR.glob("*.json"))
        assert len(datasets) == len(manifest_files)
        assert {
            "gb_forecast_actual_demand",
            "gb_gen_by_type",
            "gb_market_mid",
            "aemo_5min_demand",
            "aemo_forecast",
            "ausgrid_zone_substation_fy25_imputed",
            "google_dc_2019",
            "alibaba_dc_2018",
            "alibaba_gpu_2020",
            "azure_dc_v2",
        }.issubset(set(datasets))

    def test_list_sources(self, registry):
        sources = registry.list_sources()
        assert "gb" in sources
        assert "aemo" in sources
        assert "google" in sources
        assert "alibaba" in sources
        assert "azure" in sources
        assert "ausgrid" in sources

    def test_list_signals(self, registry):
        sigs = registry.list_signals()
        assert S.LOAD_ACTUAL_MW in sigs
        assert S.SOLAR_AVAILABLE_MW in sigs
        assert S.WIND_AVAILABLE_MW in sigs
        assert S.MARKET_MID_PRICE_APX in sigs
        assert S.DC_CPU_UTIL in sigs

    def test_find_by_signal(self, registry):
        results = registry.find_by_signal(S.LOAD_ACTUAL_MW)
        assert len(results) >= 1
        names = [m.name for m in results]
        assert any("demand" in n or "forecast" in n for n in names)

    def test_find_by_signal_with_source(self, registry):
        results = registry.find_by_signal(S.LOAD_ACTUAL_MW, source="gb")
        for m in results:
            assert m.source == "gb"

    def test_find_by_signal_with_data_type(self, registry):
        results = registry.find_by_signal(
            S.LOAD_ACTUAL_MW, data_type=S.FORECAST_PANEL,
        )
        for m in results:
            assert m.data_type == S.FORECAST_PANEL

    def test_find_by_source(self, registry):
        results = registry.find_by_source("alibaba")
        assert len(results) == 2
        names = {m.name for m in results}
        assert "alibaba_dc_2018" in names
        assert "alibaba_gpu_2020" in names

    def test_get_manifest(self, registry):
        m = registry.get_manifest("google_dc_2019")
        assert m.source == "google"
        assert m.time_mode == "profile"
        assert m.cyclical is True

    def test_get_manifest_unknown(self, registry):
        with pytest.raises(KeyError, match="Unknown dataset"):
            registry.get_manifest("nonexistent")

    def test_resolve_signals(self, registry):
        result = registry.resolve_signals([S.LOAD_ACTUAL_MW, S.DC_CPU_UTIL])
        assert S.LOAD_ACTUAL_MW in result
        assert S.DC_CPU_UTIL in result

    def test_resolve_signals_missing(self, registry):
        with pytest.raises(ValueError, match="Cannot resolve"):
            registry.resolve_signals(["nonexistent.signal"])

    def test_empty_registry(self, tmp_path):
        """Registry with empty dir has no datasets."""
        empty_dir = tmp_path / "empty_manifests"
        empty_dir.mkdir()
        r = DatasetRegistry(empty_dir)
        assert r.list_datasets() == []
        assert r.list_signals() == []
