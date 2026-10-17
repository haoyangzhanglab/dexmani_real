"""Workspace bounds and static environment geometry in the robot base frame."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dexmani_real.utils.geometry import validate_unit_quaternion_wxyz


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
        bounds = self.as_array()
        if not np.all(np.isfinite(bounds)) or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("workspace bounds must be finite and ordered")

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
    """Static box in xArm base: full side lengths and box-local→base WXYZ rotation."""

    name: str = "obstacle"
    center_xyz_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    size_xyz_m: tuple[float, float, float] = (1.0, 1.0, 1.0)
    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def validate(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name.strip()
            or self.name == "table"
        ):
            raise ValueError(
                "static collision box name must be non-empty and must not use reserved name 'table'"
            )
        center = np.asarray(self.center_xyz_m, dtype=np.float64)
        size = np.asarray(self.size_xyz_m, dtype=np.float64)
        if center.shape != (3,) or size.shape != (3,):
            raise ValueError(
                "static collision box center and size must both have shape (3,)"
            )
        if not np.all(np.isfinite(np.concatenate((center, size)))):
            raise ValueError("static collision box values must be finite")
        if np.any(size <= 0.0):
            raise ValueError(
                "static collision box size_xyz_m must contain positive full side lengths"
            )
        validate_unit_quaternion_wxyz(self.quat_wxyz, name="static collision box quaternion")


@dataclass(frozen=True)
class TableCollisionConfig:
    """Finite collision box below an upward base-frame plane ax+by+cz+d=0.
    Resolve plane_path for collision or cloud table removal into one session snapshot.
    HOME checks table clearance; online manipulation permits operator-supervised table contact.
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


@dataclass(frozen=True)
class EnvironmentConfig:
    table: TableCollisionConfig = field(default_factory=TableCollisionConfig)
    static_boxes: tuple[StaticCollisionBox, ...] = ()
