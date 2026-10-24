"""小尺寸物理行和显式 fake；不实例化设备驱动。"""

from types import SimpleNamespace

import numpy as np


def make_row(stamp=1_000_000_000):
    arm = np.zeros(1, dtype=[("qpos", "f8", (7,)), ("timestamp_ns", "i8")])
    hand = np.zeros(1, dtype=[
        ("qpos", "f8", (12,)), ("timestamp_ns", "i8"),
        ("tactile_aggregate", "f4", (5, 3)), ("tactile_dense", "f4", (5, 120, 3)),
        ("tactile_aggregate_timestamp_ns", "i8"), ("tactile_dense_timestamp_ns", "i8"),
    ])
    arm["timestamp_ns"] = stamp
    for name in hand.dtype.names:
        if "timestamp_ns" in name:
            hand[name] = stamp
    return SimpleNamespace(
        arm=arm, hand=hand, point_cloud=None, pointcloud_timestamp_ns=0, camera_sequence=1,
        camera=dict(rgb=np.zeros((8, 8, 3), dtype=np.uint8),
                    depth=np.full((8, 8), 500, dtype=np.uint16),
                    timestamp_ns=stamp, ring_sequence=1),
        observation_timestamp_ns=stamp, control_timestamp_ns=stamp,
    )


def recording_metadata():
    return dict(camera_color_width=8, camera_color_height=8, camera_serial="synthetic",
                depth_scale=0.001)
