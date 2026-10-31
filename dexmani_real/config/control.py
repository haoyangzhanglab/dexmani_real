"""运动约束、工作空间与环境碰撞配置。"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, radians

import numpy as np

from dexmani_real.config.calibration_data import TABLE_PLANE_RELATIVE_PATH


@dataclass(frozen=True)
class MotionConfig:
    """遥操作与策略 EEF 逆解的误差容差。"""

    ik_max_pose_error_pos_m: float = 0.02
    ik_max_pose_error_rot_rad: float = radians(5.0)


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
class TableCollisionConfig:
    """HOME 桌面碰撞盒与软间隙；标定平面由环境单独提供。"""

    size_xy_m: tuple[float, float] = (2.0, 2.0)
    thickness_m: float = 0.04
    # 包含标定得到的法兰模型残差。
    soft_clearance_m: float = 0.02
    allowed_contact_links: tuple[str, ...] = ("link_base",)

    def validate(self) -> None:
        size = np.asarray(self.size_xy_m, dtype=np.float64)
        if size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0.0):
            raise ValueError("table size_xy_m must contain two finite positive lengths")
        if not isfinite(self.thickness_m) or self.thickness_m <= 0.0:
            raise ValueError("table thickness_m must be finite and positive")
        if not isfinite(self.soft_clearance_m) or self.soft_clearance_m < 0.0:
            raise ValueError("table soft_clearance_m must be finite and non-negative")
        if any(
            not isinstance(name, str) or not name.strip() for name in self.allowed_contact_links
        ):
            raise TypeError("table allowed_contact_links must contain non-empty link names")


@dataclass(frozen=True)
class EnvironmentConfig:
    workspace: WorkspaceBounds = field(default_factory=WorkspaceBounds)
    eef_workspace_margin_m: float = 0.005
    # HOME 与在线目标检查共用的静态障碍物。
    static_boxes: tuple[StaticCollisionBox, ...] = ()

    # HOME 和点云直接使用同一份标定事实。
    table_plane_path: str | None = TABLE_PLANE_RELATIVE_PATH
    table: TableCollisionConfig = field(default_factory=TableCollisionConfig)

    def validate(self) -> None:
        self.workspace.as_array()
        if not isfinite(self.eef_workspace_margin_m) or self.eef_workspace_margin_m < 0:
            raise ValueError("eef_workspace_margin_m must be finite and non-negative")
        self.table.validate()
        for box in self.static_boxes:
            box.validate()
