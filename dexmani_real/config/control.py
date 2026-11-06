"""运动约束、工作空间与环境碰撞配置。"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

import numpy as np

from dexmani_real.config.calibration_data import TABLE_PLANE_RELATIVE_PATH


@dataclass(frozen=True)
class MotionConfig:
    """共享 EEF 目标变化约束；默认值待真机验证。"""

    ik_target_speed_deg_s: float = 75.0
    ik_max_lead_deg: float = 10.0


@dataclass(frozen=True)
class WorkspaceBounds:
    """机器人基座系 XYZ 工作空间，单位 m。"""

    x_min: float = 0.25
    x_max: float = 0.72
    y_min: float = -0.50
    y_max: float = 0.50
    z_min: float = 0.05
    z_max: float = 0.50

    def as_array(self) -> np.ndarray:
        bounds = np.array(
            ((self.x_min, self.x_max), (self.y_min, self.y_max), (self.z_min, self.z_max)),
            dtype=np.float64,
        )
        if not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("workspace bounds must be finite and ordered")
        return bounds


@dataclass(frozen=True)
class StaticCollisionBox:
    """基座系轴对齐静态盒体；尺寸为沿 X/Y/Z 轴的完整边长，单位米。"""

    center_xyz_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    size_xyz_m: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def validate(self) -> None:
        vectors = np.asarray((self.center_xyz_m, self.size_xyz_m), dtype=np.float64)
        if vectors.shape != (2, 3):
            raise ValueError("static collision box center and size must both have shape (3,)")
        if not np.isfinite(vectors).all():
            raise ValueError("static collision box values must be finite")
        if np.any(vectors[1] <= 0.0):
            raise ValueError("static collision box size_xyz_m must contain positive full side lengths")


@dataclass(frozen=True)
class EnvironmentConfig:
    workspace: WorkspaceBounds = field(default_factory=WorkspaceBounds)
    # 最终 EEF 验收允许超出名义工作空间的距离，向外放宽边界。
    eef_workspace_tolerance_m: float = 0.005
    # EEF/jog 使用的静态障碍物。
    static_boxes: tuple[StaticCollisionBox, ...] = ()

    # 点云去桌面所用的标定事实。
    table_plane_path: str | None = TABLE_PLANE_RELATIVE_PATH

    def validate(self) -> None:
        self.workspace.as_array()
        if not isfinite(self.eef_workspace_tolerance_m) or self.eef_workspace_tolerance_m < 0:
            raise ValueError("eef_workspace_tolerance_m must be finite and non-negative")
        for box in self.static_boxes:
            box.validate()
