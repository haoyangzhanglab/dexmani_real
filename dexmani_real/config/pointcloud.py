"""Shared point-cloud recipe without camera or geometry backend imports."""

from __future__ import annotations

from dataclasses import astuple, dataclass

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
    # Reject unsupported thin structures at unresolved depth edges.
    edge_support_min_neighbors: int = 5
    # Retain low surfaces only with coherent seeds above the observed 7–13 mm table residual.
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
    # Reject 7--9 point islands that can pass the six-neighbor density threshold.
    outlier_min_component_points: int = 10
    # Bounds final sampling work after density/component filtering.
    outlier_candidate_multiplier: int = 8
    # Stratify over 15 mm cells at the default voxel size, then fill deterministically.
    sampling_coarse_voxel_stride: int = 3

    def validate(self) -> None:
        if type(self.remove_table) is not bool:
            raise TypeError("remove_table must be bool")
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
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        workspace = np.asarray(self.workspace, dtype=np.float64)
        if workspace.shape != (6,):
            raise ValueError("workspace must contain six xyz lower/upper bounds")
        if not all(np.all(np.isfinite(value)) for value in astuple(self)):
            raise ValueError("point-cloud configuration values must be finite")
        if not 0.0 < self.depth_min_m < self.depth_max_m:
            raise ValueError("depth range must satisfy 0 < depth_min_m < depth_max_m")
        if self.edge_jump_m <= 0.0 or self.edge_surface_band_m < 0.0:
            raise ValueError("edge thresholds must be non-negative with positive jump")
        if max(self.depth_support_min_neighbors, self.edge_support_min_neighbors) > 8:
            raise ValueError("depth/edge support must use at most eight neighbors")
        if self.table_core_height_m < 0.0:
            raise ValueError("table_core_height_m must be non-negative")
        if self.table_object_seed_height_m <= self.table_core_height_m:
            raise ValueError("table_object_seed_height_m must exceed table_core_height_m")
        if self.voxel_size_m <= 0.0 or self.outlier_radius_m <= 0.0:
            raise ValueError("voxel and outlier radii must be positive")
        if np.any(workspace[:3] >= workspace[3:]):
            raise ValueError("workspace lower bounds must be strictly below upper bounds")
