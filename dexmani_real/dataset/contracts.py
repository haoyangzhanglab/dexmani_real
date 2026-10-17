"""Canonical multimodal representations and their numerical recipes.

Public float32 fields (leading T in Zarr, N for Policy observation history):
  fingertip_xyz: (5,3), FK fingertip link positions in xarm_base, meters.
  finger_tactile_force: (5,3), XHand SDK aggregate XYZ readings in SDK units.
  taxel_tactile_force: (5,120,3), XHand SDK taxel XYZ readings in SDK units.
Finger order is thumb/index/middle/ring/pinky; taxel/axis order is unchanged.
Tactile values are consumed Raw readings, including acquisition's bounded hold;
export applies no bias, normalization, sum or mean. Real aggregate tactile and
Sim contact force share an interface, not an identical physical measurement.

camera_intrinsic: (9,), color RGB pinhole K=[[fx,0,cx],[0,fy,cy],[0,0,1]].
camera_extrinsic: (12,), inv(T_xarm_base_from_color)[:3,:4], i.e. [R|t].
Both flatten in C-order. p_color = R @ p_base + t, with optical RDF axes
(right/down/forward). Real's reference is xarm_base, not Sim's numerical World.
K/E do not encode distortion: Raw retains the color distortion model,
coefficients and depth_scale; RGB-D and point-cloud distortion handling stay intact.
Each episode uses its own Raw camera geometry; deployment uses the active session.
All 15 arrays share one time axis. Rebuild older canonical caches from Raw.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dexmani_real.config.hardware import HandParams
from dexmani_real.config.pointcloud import PointCloudConfig


def validate_task_identity(value: str) -> str:
    """Validate an ordinary task label, not a path component."""
    if not isinstance(value, str):
        raise TypeError("task_name must be a string")
    if not value.strip():
        raise ValueError("task_name must be non-empty")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("task_name must not contain control characters")
    return value


@dataclass(frozen=True)
class ProcessingConfig:
    """Full-modal processing requires complete physical/derived modalities and finite floats on every row."""

    pointcloud: PointCloudConfig = field(default_factory=PointCloudConfig)
    table_plane_abcd: tuple[float, float, float, float] | None = None
    fingertip_link_names: tuple[str, ...] = HandParams.fingertip_link_names
    handbase_position_eef_m: tuple[float, ...] = HandParams.T_eef_handbase_pos_xyz
    handbase_quat_eef_wxyz: tuple[float, ...] = HandParams.T_eef_handbase_quat_wxyz

    @classmethod
    def from_runtime(cls, runtime, *, table_plane_abcd=None):
        return cls(
            pointcloud=runtime.pointcloud,
            table_plane_abcd=table_plane_abcd,
            fingertip_link_names=tuple(runtime.hand.fingertip_link_names),
            handbase_position_eef_m=tuple(runtime.hand.T_eef_handbase_pos_xyz),
            handbase_quat_eef_wxyz=tuple(runtime.hand.T_eef_handbase_quat_wxyz),
        )

    def __post_init__(self) -> None:
        self.pointcloud.validate()
        if (
            len(self.fingertip_link_names) != 5
            or len(set(self.fingertip_link_names)) != 5
            or any(
                not isinstance(name, str) or not name.strip() for name in self.fingertip_link_names
            )
        ):
            raise ValueError("fingertip geometry requires five distinct link names")
        if not self.pointcloud.remove_table:
            object.__setattr__(self, "table_plane_abcd", None)
        elif self.table_plane_abcd is None:
            raise ValueError("remove_table=True requires an explicit table_plane_abcd")
        if self.table_plane_abcd is not None:
            plane = tuple(float(v) for v in self.table_plane_abcd)
            if len(plane) != 4 or not np.all(np.isfinite(plane)):
                raise ValueError("table_plane_abcd must be finite with upward normal")
            norm = np.linalg.norm(plane[:3])
            if norm <= 0.0 or plane[2] / norm <= 0.0:
                raise ValueError("table_plane_abcd must have an upward non-zero normal")
            object.__setattr__(self, "table_plane_abcd", plane)


_CORE_DATASET_SPECS: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {
    "joint_state": ((19,), np.dtype(np.float32)),
    "arm_qvel": ((7,), np.dtype(np.float32)),
    "arm_effort": ((7,), np.dtype(np.float32)),
    "hand_current": ((12,), np.dtype(np.float32)),
    "action": ((19,), np.dtype(np.float32)),
    "action_ee": ((21,), np.dtype(np.float32)),
    "finger_tactile_force": ((5, 3), np.dtype(np.float32)),
    "taxel_tactile_force": ((5, 120, 3), np.dtype(np.float32)),
    "fingertip_xyz": ((5, 3), np.dtype(np.float32)),
    "eef_pose": ((9,), np.dtype(np.float32)),
    "camera_intrinsic": ((9,), np.dtype(np.float32)),
    "camera_extrinsic": ((12,), np.dtype(np.float32)),
}


def canonical_array_specs(
    num_points: int,
    rgb_height: int,
    rgb_width: int,
) -> dict[str, tuple[tuple[int, ...], np.dtype[Any]]]:
    """Canonical per-row shapes and dtypes for aligned RGB-D."""
    specs = dict(_CORE_DATASET_SPECS)
    specs.update(
        {
            "rgb": ((rgb_height, rgb_width, 3), np.dtype(np.uint8)),
            "depth": ((rgb_height, rgb_width), np.dtype(np.uint16)),
            "point_cloud": ((num_points, 6), np.dtype(np.float32)),
        }
    )
    return specs


CANONICAL_FORMAT = "dexmani.real.canonical"
