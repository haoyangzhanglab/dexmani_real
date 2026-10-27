from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dexmani_real.geometry.pose import validate_unit_quaternion_wxyz
from dexmani_real.perception.config import HandCloudConfig, PointCloudConfig
from dexmani_real.robot.xhand.config import HandConfig


@dataclass(frozen=True)
class ExportConfig:
    pointcloud: PointCloudConfig = field(default_factory=PointCloudConfig)
    hand_pointcloud: HandCloudConfig = field(default_factory=HandCloudConfig)
    table_plane_abcd: tuple[float, float, float, float] | None = None
    handbase_position_eef_m: tuple[float, ...] = HandConfig.T_eef_handbase_pos_xyz
    handbase_quat_eef_wxyz: tuple[float, ...] = HandConfig.T_eef_handbase_quat_wxyz

    @classmethod
    def from_runtime(cls, runtime, *, table_plane_abcd=None):
        return cls(
            pointcloud=runtime.pointcloud,
            hand_pointcloud=runtime.hand_pointcloud,
            table_plane_abcd=table_plane_abcd,
            handbase_position_eef_m=tuple(runtime.hand.T_eef_handbase_pos_xyz),
            handbase_quat_eef_wxyz=tuple(runtime.hand.T_eef_handbase_quat_wxyz),
        )

    def __post_init__(self) -> None:
        position = np.asarray(self.handbase_position_eef_m, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("hand base position must be a finite (3,) vector")
        validate_unit_quaternion_wxyz(self.handbase_quat_eef_wxyz, name="hand base quaternion")
        if not self.pointcloud.remove_table:
            object.__setattr__(self, "table_plane_abcd", None)
        elif self.table_plane_abcd is None:
            raise ValueError("remove_table=True requires an explicit table_plane_abcd")


def canonical_array_specs(
    num_points: int,
    rgb_height: int,
    rgb_width: int,
    hand_num_points: int,
) -> dict[str, tuple[tuple[int, ...], np.dtype[Any]]]:
    """公共导出的固定 13 字段及逐行形状、类型。"""
    return {
        "joint_state": ((19,), np.dtype(np.float32)),
        "action": ((19,), np.dtype(np.float32)),
        "action_ee": ((21,), np.dtype(np.float32)),
        "finger_tactile_force": ((5, 3), np.dtype(np.float32)),
        "taxel_tactile_force": ((5, 120, 3), np.dtype(np.float32)),
        "fingertip_xyz": ((5, 3), np.dtype(np.float32)),
        "eef_pose": ((9,), np.dtype(np.float32)),
        "camera_intrinsic": ((9,), np.dtype(np.float32)),
        "camera_extrinsic": ((12,), np.dtype(np.float32)),
        "rgb": ((rgb_height, rgb_width, 3), np.dtype(np.uint8)),
        "depth": ((rgb_height, rgb_width), np.dtype(np.uint16)),
        "point_cloud": ((num_points, 6), np.dtype(np.float32)),
        "hand_pointcloud": ((hand_num_points, 6), np.dtype(np.float32)),
    }


CANONICAL_FORMAT = "dexmani.real.canonical"
