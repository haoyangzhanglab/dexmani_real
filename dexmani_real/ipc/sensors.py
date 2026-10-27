import numpy as np

from dexmani_real.ipc.ring import SharedRing
from dexmani_real.sensor.config import CameraConfig


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
