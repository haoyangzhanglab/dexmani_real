"""Shared point-cloud recipe without camera or geometry backend imports."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

SAMPLING_METHOD = "quickfps"


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
    # Retain above-core surfaces only when connected to a sufficiently large high seed.
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

    def validate(self) -> None:
        if type(self.remove_table) is not bool:
            raise TypeError("remove_table must be bool")
        for name, minimum in (
            ("num_points", 1),
            ("depth_support_min_neighbors", 0),
            ("edge_support_min_neighbors", 0),
            ("outlier_min_neighbors", 0),
            ("outlier_min_component_points", 1),
            ("outlier_candidate_multiplier", 1),
        ):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        workspace = np.asarray(self.workspace, dtype=np.float64)
        if workspace.shape != (6,):
            raise ValueError("workspace must contain six xyz lower/upper bounds")
        numeric_values = (
            self.depth_min_m, self.depth_max_m, self.edge_jump_m, self.edge_surface_band_m,
            self.voxel_size_m, self.outlier_radius_m, *workspace,
        )
        if not np.isfinite(numeric_values).all():
            raise ValueError("point-cloud configuration values must be finite")
        if not 0.0 < self.depth_min_m < self.depth_max_m:
            raise ValueError("depth range must satisfy 0 < depth_min_m < depth_max_m")
        if self.edge_jump_m <= 0.0 or self.edge_surface_band_m < 0.0:
            raise ValueError("edge thresholds must be non-negative with positive jump")
        if max(self.depth_support_min_neighbors, self.edge_support_min_neighbors) > 8:
            raise ValueError("depth/edge support must use at most eight neighbors")
        for name in ("table_core_height_m", "table_object_seed_height_m"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if type(self.table_object_seed_min_pixels) is not int:
            raise ValueError("table_object_seed_min_pixels must be an integer")
        if self.remove_table:
            if self.table_object_seed_min_pixels < 1:
                raise ValueError("table_object_seed_min_pixels must be an integer >= 1")
            if not 0.0 <= self.table_core_height_m < self.table_object_seed_height_m:
                raise ValueError("table heights must satisfy 0 <= core < object_seed")
        if self.voxel_size_m <= 0.0 or self.outlier_radius_m <= 0.0:
            raise ValueError("voxel and outlier radii must be positive")
        if np.any(workspace[:3] >= workspace[3:]):
            raise ValueError("workspace lower bounds must be strictly below upper bounds")


def resolve_saved_pointcloud_recipe(recipe):
    """Restore the complete saved recipe without filling fields from current defaults."""
    if not isinstance(recipe, dict):
        raise ValueError("saved pointcloud recipe must be a mapping")
    recipe = dict(recipe)
    if recipe.pop("sampling_method", None) != SAMPLING_METHOD:
        raise ValueError(f"saved point-cloud sampling_method must be {SAMPLING_METHOD!r}")
    required = {field.name for field in fields(PointCloudConfig)}
    missing = required - recipe.keys()
    if missing:
        raise ValueError(f"Saved pointcloud recipe missing fields: {sorted(missing)}")
    config = PointCloudConfig(**{**recipe, "workspace": tuple(recipe["workspace"])})
    config.validate()
    return config
