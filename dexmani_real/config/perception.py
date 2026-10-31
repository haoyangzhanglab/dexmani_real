"""点云处理与手表面模板的实验参数。"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PointCloudConfig:
    remove_table: bool = True
    num_points: int = 1024
    table_margin_m: float = 0.010
    workspace: tuple[float, ...] = (0.15, -0.50, 0.0, 0.85, 0.50, 0.80)

    def __post_init__(self):
        if type(self.remove_table) is not bool:
            raise TypeError("remove_table must be bool")
        if type(self.num_points) is not int or self.num_points < 1:
            raise ValueError("num_points must be a positive integer")
        workspace = np.asarray(self.workspace, dtype=np.float64)
        if (
            workspace.shape != (6,)
            or not np.isfinite(workspace).all()
            or np.any(workspace[:3] >= workspace[3:])
        ):
            raise ValueError("workspace must contain finite ordered xyz lower/upper bounds")
        if self.remove_table and (not np.isfinite(self.table_margin_m) or self.table_margin_m < 0):
            raise ValueError("table_margin_m must be finite and nonnegative")
        object.__setattr__(self, "workspace", tuple(float(v) for v in workspace))


@dataclass(frozen=True)
class HandCloudConfig:
    num_points: int = 1024

    def __post_init__(self):
        if type(self.num_points) is not int or self.num_points < 1:
            raise ValueError("hand point count must be a positive integer")
