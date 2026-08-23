"""Canonical point-cloud processing policy.

This module contains configuration only.  Keeping it outside the sensor
implementation lets realtime, offline, and diagnostic entry points resolve the
same immutable policy without importing camera or geometry dependencies.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PointCloudConfig:
    """Validated native RGB-D point-cloud processing policy."""

    num_points: int = 1024
    depth_min_m: float = 0.30
    depth_max_m: float = 1.50
    edge_jump_m: float = 0.030
    edge_surface_band_m: float = 0.008
    depth_support_min_neighbors: int = 2
    table_clearance_m: float = 0.008
    table_support_min_pixels: int = 5
    workspace: tuple[float, float, float, float, float, float] = (
        0.25,
        -0.60,
        0.005,
        0.85,
        0.60,
        0.80,
    )
    voxel_size_m: float = 0.005
    outlier_radius_m: float = 0.012
    outlier_min_neighbors: int = 2
    outlier_candidate_multiplier: int = 8

    def __post_init__(self) -> None:
        integer_fields = (
            ("num_points", self.num_points, 1),
            ("depth_support_min_neighbors", self.depth_support_min_neighbors, 0),
            ("table_support_min_pixels", self.table_support_min_pixels, 1),
            ("outlier_min_neighbors", self.outlier_min_neighbors, 0),
            ("outlier_candidate_multiplier", self.outlier_candidate_multiplier, 1),
        )
        for name, value, minimum in integer_fields:
            if isinstance(value, bool) or int(value) != value or int(value) < minimum:
                relation = "positive" if minimum == 1 else "non-negative"
                raise ValueError(f"{name} must be a {relation} integer")
            object.__setattr__(self, name, int(value))

        float_fields = (
            "depth_min_m",
            "depth_max_m",
            "edge_jump_m",
            "edge_surface_band_m",
            "table_clearance_m",
            "voxel_size_m",
            "outlier_radius_m",
        )
        for name in float_fields:
            object.__setattr__(self, name, float(getattr(self, name)))
        workspace = tuple(float(value) for value in self.workspace)
        if len(workspace) != 6:
            raise ValueError("workspace must contain six xyz lower/upper bounds")
        object.__setattr__(self, "workspace", workspace)

        values = tuple(getattr(self, name) for name in float_fields) + workspace
        if not all(np.isfinite(value) for value in values):
            raise ValueError("point-cloud configuration values must be finite")
        if not 0.0 < self.depth_min_m < self.depth_max_m:
            raise ValueError("depth range must satisfy 0 < depth_min_m < depth_max_m")
        if self.edge_jump_m <= 0.0 or self.edge_surface_band_m < 0.0:
            raise ValueError("edge thresholds must be non-negative with positive jump")
        if self.depth_support_min_neighbors > 8:
            raise ValueError("depth_support_min_neighbors must be at most 8")
        if self.table_clearance_m < 0.0:
            raise ValueError("table_clearance_m must be non-negative")
        if self.table_support_min_pixels > 9:
            raise ValueError("table_support_min_pixels must be at most 9")
        if self.voxel_size_m <= 0.0 or self.outlier_radius_m <= 0.0:
            raise ValueError("voxel and outlier radii must be positive")
        lower = self.workspace[:3]
        upper = self.workspace[3:]
        if any(low >= high for low, high in zip(lower, upper, strict=True)):
            raise ValueError(
                "workspace lower bounds must be strictly below upper bounds"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the stable persisted processing-policy representation."""
        return {
            "num_points": self.num_points,
            "depth_min_m": self.depth_min_m,
            "depth_max_m": self.depth_max_m,
            "edge_jump_m": self.edge_jump_m,
            "edge_surface_band_m": self.edge_surface_band_m,
            "depth_support_min_neighbors": self.depth_support_min_neighbors,
            "table_clearance_m": self.table_clearance_m,
            "table_support_min_pixels": self.table_support_min_pixels,
            "workspace": list(self.workspace),
            "voxel_size_m": self.voxel_size_m,
            "outlier_radius_m": self.outlier_radius_m,
            "outlier_min_neighbors": self.outlier_min_neighbors,
            "outlier_candidate_multiplier": self.outlier_candidate_multiplier,
        }

    @property
    def sha256(self) -> str:
        """Stable identity for boundary checks and persisted provenance."""
        canonical = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
