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

# These strings are part of the persisted Real-data and deployment contract,
# not sensor implementation details.  Keeping them in the pure config owner
# lets offline artifact checks validate point-cloud semantics without importing
# camera, OpenCV, or geometry code.
POINT_CLOUD_POLICY_ID = "depth_to_color_orthogonal_edge_table_voxel_radius_graph_v9"
POINT_CLOUD_COLOR_SOURCE = "mean_rgb_of_aligned_depth_pixels_per_voxel"
POINT_CLOUD_SAMPLING = "deterministic_coarse_voxel_stratified_hash_or_cyclic_pad"
POINT_CLOUD_TRANSFORM = (
    "depth_gate_and_cardinal_edge_support;depth_to_color_deprojection;"
    "table_plane_height_hysteresis_crop_in_color_frame_before_deprojection;"
    "xarm_base_transform;workspace_crop;mean_voxel_xyz_and_rgb;"
    "single_radius_graph_density_and_component_outlier;spatial_candidate_cap;"
    "coarse_voxel_stratified_hash_or_cyclic_pad"
)


@dataclass(frozen=True)
class PointCloudConfig:
    """Validated depth-to-color aligned RGB-D point-cloud policy."""

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
    # above the core are preserved down to that height only when enough pixels
    # exceed the object-seed height. This removes low table residual islands
    # without dilating the table mask into object boundaries.
    table_core_height_m: float = 0.007
    table_object_seed_height_m: float = 0.013
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
    outlier_candidate_multiplier: int = 8
    # Select one fine-voxel representative per coarse 3x3x3 cell before the
    # final hash fill. At the default 5 mm voxel size this stratifies sampling
    # over 15 mm cells without the runtime cost of farthest-point sampling.
    sampling_coarse_voxel_stride: int = 3

    def __post_init__(self) -> None:
        integer_fields = (
            ("num_points", self.num_points, 1),
            ("depth_support_min_neighbors", self.depth_support_min_neighbors, 0),
            ("edge_support_min_neighbors", self.edge_support_min_neighbors, 0),
            ("table_object_seed_min_pixels", self.table_object_seed_min_pixels, 1),
            ("outlier_min_neighbors", self.outlier_min_neighbors, 0),
            ("outlier_min_component_points", self.outlier_min_component_points, 1),
            ("outlier_candidate_multiplier", self.outlier_candidate_multiplier, 1),
            ("sampling_coarse_voxel_stride", self.sampling_coarse_voxel_stride, 1),
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

        values = tuple(getattr(self, name) for name in float_fields) + workspace
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
            raise ValueError(
                "table_object_seed_height_m must exceed table_core_height_m"
            )
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

    @property
    def sha256(self) -> str:
        """Stable identity for boundary checks and persisted provenance."""
        canonical = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
