"""Dataset manifest: describes how a raw data source maps to semantic signals."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class DatasetManifest:
    """Declarative description of one dataset.

    A manifest is the single source of truth that tells the data layer:
    * which parquet file to read,
    * how to map raw columns to semantic signals,
    * how to handle time alignment (calendar vs. profile), and
    * whether data can be tiled cyclically.
    """

    name: str
    source: str                                      # "gb" / "aemo" / "alibaba" / ...
    data_type: str                                   # "actual_series" | "forecast_panel"
    time_mode: str                                   # "calendar" | "profile"
    resolution: str                                  # "30min" / "5min" / "300s"
    parquet_file: str                                # relative to data_dir
    column_map: Dict[str, str] = field(default_factory=dict)
    index_map: Dict[str, str] = field(default_factory=dict)
    derived: Dict[str, str] = field(default_factory=dict)
    normalize: Dict[str, float] = field(default_factory=dict)
    data_epoch: Optional[str] = None
    cyclical: bool = False
    region_values: List[str] = field(default_factory=list)
    date_range: Optional[Tuple[str, str]] = None
    metadata_json: Optional[str] = None
    #: Primary upstream API or landing page (when a single URL suffices).
    source_url: Optional[str] = None
    #: Multiple upstream endpoints (e.g. actual + day-ahead forecast).
    source_urls: Optional[List[str]] = None
    source_organization: Optional[str] = None

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    @property
    def signals(self) -> List[str]:
        """All semantic signals this manifest can provide (mapped + derived)."""
        mapped = list(self.column_map.values())
        derived = list(self.derived.keys())
        return sorted(set(mapped + derived))

    @property
    def raw_columns_needed(self) -> List[str]:
        """Raw columns that must be read from the parquet file."""
        cols: list[str] = list(self.column_map.keys())
        for expr in self.derived.values():
            for token in expr.replace("+", " ").replace("-", " ").split():
                token = token.strip()
                if token and not token.replace(".", "", 1).isdigit():
                    cols.append(token)
        cols.extend(self.index_map.keys())
        return sorted(set(cols))

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, path: Path) -> "DatasetManifest":
        """Load a manifest from a JSON file."""
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        dr = raw.get("date_range")
        if isinstance(dr, list) and len(dr) == 2:
            dr = tuple(dr)
        else:
            dr = None
        return cls(
            name=raw["name"],
            source=raw["source"],
            data_type=raw["data_type"],
            time_mode=raw["time_mode"],
            resolution=raw["resolution"],
            parquet_file=raw["parquet_file"],
            column_map=raw.get("column_map", {}),
            index_map=raw.get("index_map", {}),
            derived=raw.get("derived", {}),
            normalize=raw.get("normalize", {}),
            data_epoch=raw.get("data_epoch"),
            cyclical=raw.get("cyclical", False),
            region_values=raw.get("region_values", []),
            date_range=dr,
            metadata_json=raw.get("metadata_json"),
            source_url=raw.get("source_url"),
            source_urls=list(raw["source_urls"]) if raw.get("source_urls") else None,
            source_organization=raw.get("source_organization"),
        )

    def to_dict(self) -> dict:
        """Serialize to a plain dict (JSON-friendly)."""
        d: dict = {
            "name": self.name,
            "source": self.source,
            "data_type": self.data_type,
            "time_mode": self.time_mode,
            "resolution": self.resolution,
            "parquet_file": self.parquet_file,
            "column_map": self.column_map,
            "index_map": self.index_map,
            "derived": self.derived,
            "normalize": self.normalize,
            "data_epoch": self.data_epoch,
            "cyclical": self.cyclical,
            "region_values": self.region_values,
            "date_range": list(self.date_range) if self.date_range else None,
            "metadata_json": self.metadata_json,
        }
        if self.source_url is not None:
            d["source_url"] = self.source_url
        if self.source_urls is not None:
            d["source_urls"] = self.source_urls
        if self.source_organization is not None:
            d["source_organization"] = self.source_organization
        return d
