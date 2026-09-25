"""Workspace bounds and static environment geometry in the robot base frame."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class WorkspaceBounds:
    """EEF workspace bounds in arm-base frame (meters)."""

    x_min: float = 0.25
    x_max: float = 0.72
    y_min: float = -0.50
    y_max: float = 0.50
    z_min: float = 0.05
    z_max: float = 0.50

    def validate(self) -> None:
        bounds = np.asarray(self.as_tuple(), dtype=np.float64)
        if not np.all(np.isfinite(bounds)) or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("workspace bounds must be finite and ordered")

    def as_tuple(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        """Workspace bounds as ((x_min,x_max), (y_min,y_max), (z_min,z_max))."""
        return (
            (self.x_min, self.x_max),
            (self.y_min, self.y_max),
            (self.z_min, self.z_max),
        )

    def as_array(self) -> "np.ndarray":
        """Workspace bounds as (3,2) np.ndarray — returns a mutable copy."""
        return np.array(
            [
                [self.x_min, self.x_max],
                [self.y_min, self.y_max],
                [self.z_min, self.z_max],
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class StaticCollisionBox:
    """One oriented static obstacle in the xArm base frame.

    ``size_xyz_m`` contains full side lengths (not half extents) and
    ``quat_wxyz`` rotates the box-local axes into the base frame.
    """

    name: str = "obstacle"
    center_xyz_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    size_xyz_m: tuple[float, float, float] = (1.0, 1.0, 1.0)
    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def validate(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name.strip()
            or self.name != self.name.strip()
            or self.name == "table"
        ):
            raise ValueError(
                "static collision box name must be non-empty and must not use reserved name 'table'"
            )
        center = np.asarray(self.center_xyz_m, dtype=np.float64)
        size = np.asarray(self.size_xyz_m, dtype=np.float64)
        quat = np.asarray(self.quat_wxyz, dtype=np.float64)
        if center.shape != (3,) or size.shape != (3,) or quat.shape != (4,):
            raise ValueError(
                "static collision box center/size/quaternion must have shapes (3,), (3,), and (4,)"
            )
        if not np.all(np.isfinite(np.concatenate((center, size, quat)))):
            raise ValueError("static collision box values must be finite")
        if np.any(size <= 0.0):
            raise ValueError(
                "static collision box size_xyz_m must contain positive full side lengths"
            )
        if not np.isclose(float(np.linalg.norm(quat)), 1.0, rtol=0.0, atol=1e-6):
            raise ValueError("static collision box quat_wxyz must be a unit quaternion")


@dataclass(frozen=True)
class TableCollisionConfig:
    """Calibrated table represented as a finite box below an upward plane.

    ``plane_abcd`` uses the robot-base/world convention ``ax+by+cz+d=0``.
    Runtime resolution refreshes enabled collision geometry from ``plane_path``.
    Perception reads that file separately when its effective point-cloud config
    requires table removal, regardless of collision-table enablement.
    Fine teleoperation deliberately does not use this geometry to reject
    robot-table contact.
    """

    enabled: bool = True
    plane_path: str | None = "dexmani_real/calibration/state/table_plane.json"
    plane_abcd: tuple[float, float, float, float] = (0.0, 0.0, 1.0, -0.022)
    size_xy_m: tuple[float, float] = (2.0, 2.0)
    thickness_m: float = 0.04
    # Includes the calibrated flange-model residual.
    soft_clearance_m: float = 0.02
    allowed_contact_links: tuple[str, ...] = ("link_base",)

    def validate(self) -> None:
        if self.plane_path is not None and (
            not isinstance(self.plane_path, str) or not self.plane_path.strip()
        ):
            raise ValueError("table plane_path must be a non-empty string or null")
        plane = np.asarray(self.plane_abcd, dtype=np.float64)
        size = np.asarray(self.size_xy_m, dtype=np.float64)
        if plane.shape != (4,) or size.shape != (2,):
            raise ValueError("table plane_abcd and size_xy_m must have shapes (4,) and (2,)")
        if not np.all(np.isfinite(np.concatenate((plane, size)))):
            raise ValueError("table plane and size must be finite")
        normal_norm = float(np.linalg.norm(plane[:3]))
        if normal_norm <= 1e-9 or float(plane[2] / normal_norm) <= 0.0:
            raise ValueError("table plane must have a finite upward-pointing normal")
        if np.any(size <= 0.0) or not np.isfinite(self.thickness_m) or self.thickness_m <= 0.0:
            raise ValueError("table size and thickness must be finite and positive")
        if not np.isfinite(self.soft_clearance_m) or self.soft_clearance_m < 0.0:
            raise ValueError("table soft_clearance_m must be finite and non-negative")
        if any(
            not isinstance(name, str) or not name.strip() for name in self.allowed_contact_links
        ):
            raise TypeError("table allowed_contact_links must contain non-empty link names")
        if len(self.allowed_contact_links) != len(set(self.allowed_contact_links)):
            raise ValueError("table allowed_contact_links must be unique")


@dataclass(frozen=True)
class EnvironmentConfig:
    """Calibrated table plus optional static robot-environment geometry."""

    table: TableCollisionConfig = field(default_factory=TableCollisionConfig)
    static_boxes: tuple[StaticCollisionBox, ...] = ()

    def validate(self) -> None:
        names = [box.name for box in self.static_boxes]
        if len(names) != len(set(names)):
            raise ValueError("environment.static_boxes names must be unique")
