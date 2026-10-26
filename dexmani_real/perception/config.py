"""点云处理与手表面采样的实验参数。"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PointCloudConfig:
    remove_table: bool = True
    num_points: int = 1024
    depth_min_m: float = 0.30
    depth_max_m: float = 1.50
    edge_jump_m: float = 0.030
    edge_surface_band_m: float = 0.008
    table_margin_m: float = 0.007
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

    def __post_init__(self):
        if type(self.remove_table) is not bool:
            raise TypeError("remove_table must be bool")
        if type(self.num_points) is not int or self.num_points < 1:
            raise ValueError("num_points must be a positive integer")
        if type(self.outlier_min_neighbors) is not int or self.outlier_min_neighbors < 0:
            raise ValueError("outlier_min_neighbors must be a nonnegative integer")
        workspace = np.asarray(self.workspace, dtype=np.float64)
        if workspace.shape != (6,) or np.any(workspace[:3] >= workspace[3:]):
            raise ValueError("workspace must contain ordered xyz lower/upper bounds")
        if not np.isfinite(
            (
                self.depth_min_m,
                self.depth_max_m,
                self.edge_jump_m,
                self.edge_surface_band_m,
                self.table_margin_m,
                self.voxel_size_m,
                self.outlier_radius_m,
                *workspace,
            )
        ).all():
            raise ValueError("point-cloud parameters must be finite")
        if not 0 < self.depth_min_m < self.depth_max_m:
            raise ValueError("depth range must satisfy 0 < min < max")
        if min(self.edge_jump_m, self.voxel_size_m, self.outlier_radius_m) <= 0:
            raise ValueError("edge jump, voxel size and outlier radius must be positive")
        if min(self.edge_surface_band_m, self.table_margin_m) < 0:
            raise ValueError("surface band and table margin must be nonnegative")
        object.__setattr__(self, "workspace", tuple(float(v) for v in workspace))


@dataclass(frozen=True)
class HandCloudConfig:
    num_points: int = 1024
    seed: int = 0

    def __post_init__(self):
        if type(self.num_points) is not int or self.num_points < 1:
            raise ValueError("hand point count must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("hand point seed must be a nonnegative integer")
