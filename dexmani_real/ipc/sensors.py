import json
from typing import cast

import numpy as np

from dexmani_real.config.perception import HOI_NUM_POINTS
from dexmani_real.config.sensor import CameraConfig
from dexmani_real.geometry.camera import CameraIntrinsics, CameraSnapshot
from dexmani_real.ipc.ring import SharedRing
from dexmani_real.utils.camera_calibrate.extrinsics import (
    load_camera_extrinsics,
    load_optional_camera_extrinsics,
)

# 组合观测的 recv_ts_ns 取腕部与关键点中较早的接收时刻。
VR_DTYPE = np.dtype(
    [
        ("wrist_pos", "<f8", (3,)),
        ("wrist_quat_wxyz", "<f8", (4,)),
        ("landmarks", "<f8", (21, 3)),
        ("recv_ts_ns", "<u8"),
        ("wrist_recv_ts_ns", "<u8"),
        ("landmarks_recv_ts_ns", "<u8"),
        ("sequence_id", "<u8"),
        ("landmarks_sequence_id", "<u8"),
        ("connection_epoch", "<u8"),
    ],
    align=True,
)


def camera_dtype(height: int, width: int, *, depth_enabled: bool = True) -> np.dtype:
    fields: list[tuple] = [
        # 实际采集通道中最早的首次接收时刻，重复帧不刷新时间。
        ("timestamp_ns", "<u8"),
        ("rgb", "u1", (height, width, 3)),
    ]
    if depth_enabled:
        fields.append(("depth", "<u2", (height, width)))
    return np.dtype(fields, align=True)


def pointcloud_dtype(num_points: int, rgb_dtype: np.dtype | None = None) -> np.dtype:
    fields: list[tuple] = [
        ("source_camera_sequence", "<u8"),
        ("timestamp_ns", "<u8"),
        ("point_cloud", "<f4", (num_points, 6)),
    ]
    if rgb_dtype is not None:
        fields.append(("rgb", rgb_dtype))
    return np.dtype(fields, align=True)


class SensorChannels:
    def __init__(
        self,
        ctx,
        *,
        camera: CameraConfig | None = None,
        depth_enabled: bool = True,
        vr: bool = False,
        pointcloud_num_points: int | None = None,
        include_derived_rgb: bool = False,
        hoi: bool = False,
    ):
        self.camera_ring = (
            SharedRing(ctx, camera_dtype(camera.height, camera.width, depth_enabled=depth_enabled), 5)
            if camera is not None
            else None
        )
        self.vr_ring = SharedRing(ctx, VR_DTYPE, 8) if vr else None
        # VR worker 是唯一写者，断开即推进，下一份数据可能尚未发布。
        self.vr_connection_epoch = ctx.RawValue("Q", 0) if vr else None
        rgb_dtype = None
        if include_derived_rgb:
            image_dtype: np.dtype = cast(SharedRing, self.camera_ring).dtype
            rgb_dtype = image_dtype["rgb"]
        self.pointcloud_ring = (
            SharedRing(
                ctx, pointcloud_dtype(pointcloud_num_points, rgb_dtype), 8,
            )
            if pointcloud_num_points is not None
            else None
        )

        self.hoi_ring = self.hoi_request = self.hoi_ready = None
        if hoi:
            fields: list[tuple] = [
                ("episode_id", "u8"), ("timestamp_ns", "u8"), ("source_camera_sequence", "u8"),
                ("valid", "?"), ("object_pointcloud", "f4", (HOI_NUM_POINTS, 3)),
            ]
            if rgb_dtype is not None:
                fields.append(("rgb", rgb_dtype))
            self.hoi_ring = SharedRing(ctx, np.dtype(fields), 3)
            self.hoi_request = SharedRing(ctx, np.dtype([
                ("episode_id", "u8"), ("after_ns", "u8"), ("after_sequence", "u8"),
            ]), 3)
            self.hoi_ready = ctx.RawValue("b", False)

        self.is_running = ctx.RawValue("b", True)
        self.camera_depth_scale = ctx.RawValue("d", 0.0) if camera is not None and depth_enabled else None
        self.camera_serial = ctx.RawArray("c", 32)
        self.camera_intrinsics = ctx.RawArray("c", 2048)


def snapshot_camera_parameters(
    channels: SensorChannels, calibration_path, *, require_extrinsics=False, include_extrinsics=True,
) -> CameraSnapshot:
    """调用方已启用相机通道；首帧发布后复制静态参数，之后不再依赖 worker。"""
    if cast(SharedRing, channels.camera_ring).latest_sequence <= 0:
        raise ValueError("camera parameters require a published camera frame")
    serial = channels.camera_serial.value.decode("utf-8").strip()
    color = CameraIntrinsics(**json.loads(channels.camera_intrinsics.value.decode("utf-8")))
    transform = None
    if include_extrinsics:
        loader = load_camera_extrinsics if require_extrinsics else load_optional_camera_extrinsics
        transform = loader(serial, calibration_path)
    scale = channels.camera_depth_scale.value if channels.camera_depth_scale is not None else None
    return CameraSnapshot(serial, color, scale, transform)


def read_camera_frame(channels, *, fields=None):
    result = channels.camera_ring.read(
        fields=None if fields is None else ("timestamp_ns", *fields),
    )
    if result is None:
        return None
    data, sequence = result
    return dict(data, ring_sequence=sequence, timestamp_ns=int(data["timestamp_ns"]))
