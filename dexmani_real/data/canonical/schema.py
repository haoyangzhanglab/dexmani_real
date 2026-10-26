"""Canonical arrays and numerical recipes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dexmani_real.perception.hand_config import HandPointCloudConfig
from dexmani_real.perception.pointcloud.config import PointCloudConfig
from dexmani_real.robot.xhand.config import HandParams


@dataclass(frozen=True)
class ProcessingConfig:
    """Full-modal processing requires complete physical/derived modalities and finite floats on every row."""

    pointcloud: PointCloudConfig = field(default_factory=PointCloudConfig)
    hand_pointcloud: HandPointCloudConfig = field(default_factory=HandPointCloudConfig)
    table_plane_abcd: tuple[float, float, float, float] | None = None
    fingertip_link_names: tuple[str, ...] = HandParams.fingertip_link_names
    handbase_position_eef_m: tuple[float, ...] = HandParams.T_eef_handbase_pos_xyz
    handbase_quat_eef_wxyz: tuple[float, ...] = HandParams.T_eef_handbase_quat_wxyz

    @classmethod
    def from_runtime(cls, runtime, *, table_plane_abcd=None):
        return cls(
            pointcloud=runtime.pointcloud,
            hand_pointcloud=runtime.hand_pointcloud,
            table_plane_abcd=table_plane_abcd,
            fingertip_link_names=tuple(runtime.hand.fingertip_link_names),
            handbase_position_eef_m=tuple(runtime.hand.T_eef_handbase_pos_xyz),
            handbase_quat_eef_wxyz=tuple(runtime.hand.T_eef_handbase_quat_wxyz),
        )

    def __post_init__(self) -> None:
        self.pointcloud.validate()
        self.hand_pointcloud.validate()
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
