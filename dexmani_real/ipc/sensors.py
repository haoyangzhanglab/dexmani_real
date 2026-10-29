import json

import numpy as np

from dexmani_real.config.sensor import CameraConfig
from dexmani_real.geometry.camera import CameraIntrinsics, CameraSnapshot
from dexmani_real.ipc.ring import SharedRing
from dexmani_real.utils.camera_calibrate.extrinsics import (
    load_camera_extrinsics,
    load_optional_camera_extrinsics,
)

# 右手发布携带最近头部姿态；recv_ts_ns 取腕部与关键点中较早的接收时刻。
VR_DTYPE = np.dtype(
    [
        ("wrist_pos", "<f8", (3,)),
        ("wrist_quat_wxyz", "<f8", (4,)),
        ("landmarks", "<f8", (21, 3)),
        ("head_pos", "<f8", (3,)),
        ("head_quat_wxyz", "<f8", (4,)),
        ("recv_ts_ns", "<u8"),
        ("sequence_id", "<u8"),
        ("head_recv_ts_ns", "<u8"),
    ],
    align=True,
)


def camera_dtype(height: int, width: int) -> np.dtype:
    return np.dtype(
        [
            # 取 RGB 与深度中较早的首次接收时刻，重复帧不刷新时间。
            ("timestamp_ns", "<u8"),
            ("color_frame_number", "<u8"),
            ("rgb", "u1", (height, width, 3)),
            ("depth", "<u2", (height, width)),
        ],
        align=True,
    )


def pointcloud_dtype(num_points: int) -> np.dtype:
    return np.dtype(
        [
            ("source_camera_sequence", "<u8"),
            ("timestamp_ns", "<u8"),
            ("point_cloud", "<f4", (num_points, 6)),
        ],
        align=True,
    )


class SensorChannels:
    def __init__(
        self,
        ctx,
        *,
        camera: CameraConfig | None = None,
        vr: bool = False,
        pointcloud_num_points: int | None = None,
    ):
        self.camera_ring = (
            SharedRing(ctx, camera_dtype(camera.height, camera.width), 5)
            if camera is not None
            else None
        )
        self.vr_ring = SharedRing(ctx, VR_DTYPE, 8) if vr else None
        self.pointcloud_ring = (
            SharedRing(ctx, pointcloud_dtype(pointcloud_num_points), 8)
            if pointcloud_num_points is not None
            else None
        )

        self.is_running = ctx.RawValue("b", True)
        self.vr_ready, self.camera_ready, self.pointcloud_ready = (ctx.Event() for _ in range(3))
        self.camera_depth_scale = ctx.RawValue("d", 0.0)
        self.camera_serial = ctx.RawArray("c", 32)
        self.camera_intrinsics = ctx.RawArray("c", 2048)


def snapshot_camera_parameters(
    channels, config, calibration_path, *, require_extrinsics=False, include_extrinsics=True,
):
    """在 ready 边界复制相机静态参数，之后不再依赖 worker 或共享通道。"""
    if not channels.camera_ready.is_set():
        raise ValueError("camera parameters require a ready camera")
    serial = channels.camera_serial.value.decode("utf-8").strip()
    if config.serial is not None and serial != config.serial:
        raise ValueError("active camera serial differs from configuration")
    color = CameraIntrinsics(**json.loads(channels.camera_intrinsics.value.decode("utf-8")))
    if (color.height, color.width) != (config.height, config.width):
        raise ValueError("camera intrinsics differ from configured resolution")
    transform = None
    if include_extrinsics:
        loader = load_camera_extrinsics if require_extrinsics else load_optional_camera_extrinsics
        transform = loader(serial, calibration_path)
    return CameraSnapshot(serial, color, channels.camera_depth_scale.value, transform)


def read_vr_frame(channels):
    result = channels.vr_ring.read()
    if result is None:
        return None
    record, sequence = result
    return {**{name: record[name] for name in record.dtype.names}, "ring_sequence": sequence}


def read_camera_frame(channels, sequence=None, *, fields=("rgb", "depth")):
    result = channels.camera_ring.read(
        fields=("timestamp_ns", "color_frame_number", *fields), sequence=sequence,
    )
    if result is None:
        return None
    data, sequence = result
    data["ring_sequence"] = sequence
    data["timestamp_ns"] = int(data["timestamp_ns"])
    data["color_frame_number"] = int(data["color_frame_number"])
    return data
