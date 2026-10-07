"""Canonical point-cloud processing policy.

This module contains configuration only.  Keeping it outside the sensor
implementation lets realtime, offline, and diagnostic entry points resolve the
same immutable policy without importing camera or geometry dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PointCloudConfig:
    """Depth-to-color recipe; numerical limits are checked by its consumers."""

    remove_table: bool = True
    num_points: int = 1024
    depth_min_m: float = 0.30
    depth_max_m: float = 1.50
    edge_jump_m: float = 0.030
    edge_surface_band_m: float = 0.008
    depth_support_min_neighbors: int = 2
    # Only at a depth discontinuity, require a denser same-surface 3x3
    # neighborhood. One- or two-pixel structures at an unresolved edge are
    # intentionally treated as unreliable depth rather than preserved noise.
    edge_support_min_neighbors: int = 5
    # Pixels at or below the core height are unambiguously table. Components
    # above the core are preserved down to that height only when connected to
    # a sufficiently large 8-connected patch above the object-seed height.
    # Scattered high noise pixels cannot preserve a low table residual island.
    # Keep seeds above the broad 7--13 mm depth residuals seen on distant table
    # surfaces, while retaining the lower core threshold for connected object bases.
    table_core_height_m: float = 0.007
    table_object_seed_height_m: float = 0.016
    table_object_seed_min_pixels: int = 4
    workspace: tuple[float, float, float, float, float, float] = (
        0.0,
        -0.50,
        0.0,
        0.80,
        0.50,
        0.80,
    )
    voxel_size_m: float = 0.005
    outlier_radius_m: float = 0.012
    outlier_min_neighbors: int = 6
    # Radius-neighbor pairs define both local density and connected islands.
    # Ten removes dense 7--9 point fragments that can satisfy the six-neighbor
    # rule, while remaining conservative for small resolved object surfaces.
    outlier_min_component_points: int = 10
    # Persisted checkpoint key: caps sampling candidates AFTER density/component filtering.
    outlier_candidate_multiplier: int = 8
    # Select one fine-voxel representative per coarse 3x3x3 cell before the
    # final deterministic fill. At the default 5 mm voxel size this stratifies sampling
    # over 15 mm cells without the runtime cost of farthest-point sampling.
    sampling_coarse_voxel_stride: int = 3

    def __post_init__(self) -> None:
        if type(self.remove_table) is not bool:
            raise TypeError("remove_table must be bool")
        if isinstance(self.num_points, bool) or not isinstance(self.num_points, (int, np.integer)):
            raise ValueError("num_points must be an integer")
        object.__setattr__(self, "num_points", int(self.num_points))
        integer_fields = (
            "depth_support_min_neighbors",
            "edge_support_min_neighbors",
            "table_object_seed_min_pixels",
            "outlier_min_neighbors",
            "outlier_min_component_points",
            "outlier_candidate_multiplier",
            "sampling_coarse_voxel_stride",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value:
                raise ValueError(f"{name} must be an integer")
            object.__setattr__(self, name, int(value))

        float_fields = (
            "depth_min_m",
            "depth_max_m",
            "edge_jump_m",
            "edge_surface_band_m",
            "table_core_height_m",
            "table_object_seed_height_m",
            "voxel_size_m",
            "outlier_radius_m",
        )
        for name in float_fields:
            object.__setattr__(self, name, float(getattr(self, name)))
        workspace = tuple(float(value) for value in self.workspace)
        if len(workspace) != 6:
            raise ValueError("workspace must contain six xyz lower/upper bounds")
        object.__setattr__(self, "workspace", workspace)

    def validate(self) -> None:
        for name, minimum in (
            ("num_points", 1),
            ("depth_support_min_neighbors", 0),
            ("edge_support_min_neighbors", 0),
            ("table_object_seed_min_pixels", 1),
            ("outlier_min_neighbors", 0),
            ("outlier_min_component_points", 1),
            ("outlier_candidate_multiplier", 1),
            ("sampling_coarse_voxel_stride", 1),
        ):
            if getattr(self, name) < minimum:
                raise ValueError(f"{name} must be >= {minimum}")
        values = (
            self.depth_min_m,
            self.depth_max_m,
            self.edge_jump_m,
            self.edge_surface_band_m,
            self.table_core_height_m,
            self.table_object_seed_height_m,
            self.voxel_size_m,
            self.outlier_radius_m,
            *self.workspace,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("point-cloud configuration values must be finite")
        if not 0.0 < self.depth_min_m < self.depth_max_m:
            raise ValueError("depth range must satisfy 0 < depth_min_m < depth_max_m")
        if self.edge_jump_m <= 0.0 or self.edge_surface_band_m < 0.0:
            raise ValueError("edge thresholds must be non-negative with positive jump")
        if self.depth_support_min_neighbors > 8:
            raise ValueError("depth_support_min_neighbors must be at most 8")
        if self.edge_support_min_neighbors > 8:
            raise ValueError("edge_support_min_neighbors must be at most 8")
        if self.table_core_height_m < 0.0:
            raise ValueError("table_core_height_m must be non-negative")
        if self.table_object_seed_height_m <= self.table_core_height_m:
            raise ValueError("table_object_seed_height_m must exceed table_core_height_m")
        if self.voxel_size_m <= 0.0 or self.outlier_radius_m <= 0.0:
            raise ValueError("voxel and outlier radii must be positive")
        lower = self.workspace[:3]
        upper = self.workspace[3:]
        if any(low >= high for low, high in zip(lower, upper, strict=True)):
            raise ValueError("workspace lower bounds must be strictly below upper bounds")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PointCloudConfig":
        if not isinstance(value, dict):
            raise ValueError("pointcloud config must be a mapping")
        unknown = set(value) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown pointcloud config fields: {sorted(unknown)}")
        config = cls(**value)
        config.validate()
        return config

    def to_dict(self) -> dict[str, Any]:
        """Return the stable persisted processing-policy representation."""
        return {
            "remove_table": self.remove_table,
            "num_points": self.num_points,
            "depth_min_m": self.depth_min_m,
            "depth_max_m": self.depth_max_m,
            "edge_jump_m": self.edge_jump_m,
            "edge_surface_band_m": self.edge_surface_band_m,
            "depth_support_min_neighbors": self.depth_support_min_neighbors,
            "edge_support_min_neighbors": self.edge_support_min_neighbors,
            "table_core_height_m": self.table_core_height_m,
            "table_object_seed_height_m": self.table_object_seed_height_m,
            "table_object_seed_min_pixels": self.table_object_seed_min_pixels,
            "workspace": list(self.workspace),
            "voxel_size_m": self.voxel_size_m,
            "outlier_radius_m": self.outlier_radius_m,
            "outlier_min_neighbors": self.outlier_min_neighbors,
            "outlier_min_component_points": self.outlier_min_component_points,
            "outlier_candidate_multiplier": self.outlier_candidate_multiplier,
            "sampling_coarse_voxel_stride": self.sampling_coarse_voxel_stride,
        }
