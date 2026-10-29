"""运动约束、工作空间与环境碰撞配置。"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import radians

import numpy as np

from dexmani_real.config.calibration_data import TABLE_PLANE_RELATIVE_PATH
from dexmani_real.geometry.pose import validate_unit_quaternion_wxyz


@dataclass(frozen=True)
class MotionConfig:
    ik_max_pose_error_pos_m: float = 0.02
    ik_max_pose_error_rot_rad: float = radians(5.0)


@dataclass(frozen=True)
class WorkspaceBounds:
    x_min: float = 0.25
    x_max: float = 0.72
    y_min: float = -0.50
    y_max: float = 0.50
    z_min: float = 0.05
    z_max: float = 0.50

    def as_array(self) -> np.ndarray:
        bounds = np.array(
            [
                [self.x_min, self.x_max],
                [self.y_min, self.y_max],
                [self.z_min, self.z_max],
            ],
            dtype=np.float64,
        )
        if not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("workspace bounds must be finite and ordered")
        return bounds


@dataclass(frozen=True)
class StaticCollisionBox:
    name: str = "obstacle"
    center_xyz_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    size_xyz_m: tuple[float, float, float] = (1.0, 1.0, 1.0)
    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def validate(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip() or self.name == "table":
            raise ValueError(
                "static collision box name must be non-empty and must not use reserved name 'table'"
            )
        center = np.asarray(self.center_xyz_m, dtype=np.float64)
        size = np.asarray(self.size_xyz_m, dtype=np.float64)
        if center.shape != (3,) or size.shape != (3,):
            raise ValueError("static collision box center and size must both have shape (3,)")
        if not np.all(np.isfinite(np.concatenate((center, size)))):
            raise ValueError("static collision box values must be finite")
        if np.any(size <= 0.0):
            raise ValueError(
                "static collision box size_xyz_m must contain positive full side lengths"
            )
        validate_unit_quaternion_wxyz(self.quat_wxyz, name="static collision box quaternion")


@dataclass(frozen=True)
class TableCollisionConfig:
    enabled: bool = True
    plane_path: str | None = TABLE_PLANE_RELATIVE_PATH
    size_xy_m: tuple[float, float] = (2.0, 2.0)
    thickness_m: float = 0.04
    # 包含标定得到的法兰模型残差。
    soft_clearance_m: float = 0.02
    allowed_contact_links: tuple[str, ...] = ("link_base",)

    def validate(self) -> None:
        if self.plane_path is not None and (
            not isinstance(self.plane_path, str) or not self.plane_path.strip()
        ):
            raise ValueError("table plane_path must be a non-empty string or null")
        size = np.asarray(self.size_xy_m, dtype=np.float64)
        if size.shape != (2,) or not np.isfinite(size).all():
            raise ValueError("table size_xy_m must be finite shape (2,)")
        if np.any(size <= 0.0) or not np.isfinite(self.thickness_m) or self.thickness_m <= 0.0:
            raise ValueError("table size and thickness must be finite and positive")
        if not np.isfinite(self.soft_clearance_m) or self.soft_clearance_m < 0.0:
            raise ValueError("table soft_clearance_m must be finite and non-negative")
        if any(
            not isinstance(name, str) or not name.strip() for name in self.allowed_contact_links
        ):
            raise TypeError("table allowed_contact_links must contain non-empty link names")


@dataclass(frozen=True)
class EnvironmentConfig:
    workspace: WorkspaceBounds = field(default_factory=WorkspaceBounds)
    eef_workspace_margin_m: float = 0.005
    table_z_surface_m: float = 0.022
    hand_safety_margin_m: float = 0.05
    table: TableCollisionConfig = field(default_factory=TableCollisionConfig)
    static_boxes: tuple[StaticCollisionBox, ...] = ()

    def validate(self) -> None:
        self.workspace.as_array()
        if not np.isfinite(self.table_z_surface_m):
            raise ValueError("table_z_surface_m must be finite")
        if not np.isfinite(self.hand_safety_margin_m) or self.hand_safety_margin_m < 0:
            raise ValueError("hand_safety_margin_m must be finite and non-negative")
        if not np.isfinite(self.eef_workspace_margin_m) or self.eef_workspace_margin_m < 0:
            raise ValueError("EEF workspace margin must be finite and nonnegative")
        if self.table.enabled:
            self.table.validate()
        for box in self.static_boxes:
            box.validate()
        names = [box.name for box in self.static_boxes]
        if len(names) != len(set(names)):
            raise ValueError("static collision box names must be unique")
