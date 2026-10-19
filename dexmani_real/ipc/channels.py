"""Sensor channels allocated by the parent and passed directly through spawn."""

from dexmani_real.sensor.camera.config import CameraParams
from dexmani_real.ipc.ring import SharedMemoryRingBuffer
from dexmani_real.ipc.schema import (
    VR_FRAME_DTYPE,
    make_camera_frame_dtype,
    make_pointcloud_frame_dtype,
)


class SensorChannels:
    def __init__(
        self,
        ctx,
        *,
        camera: CameraParams | None = None,
        vr: bool = False,
        pointcloud_num_points: int | None = None,
    ):
        if pointcloud_num_points is not None and camera is None:
            raise ValueError("pointcloud requires camera storage")
        self.camera_ring = (
            SharedMemoryRingBuffer(
                ctx,
                make_camera_frame_dtype(camera.rgb_shape, camera.depth_shape),
                camera.ring_maxlen,
            )
            if camera is not None
            else None
        )
        self.vr_ring = SharedMemoryRingBuffer(ctx, VR_FRAME_DTYPE, 8) if vr else None
        self.pointcloud_ring = (
            SharedMemoryRingBuffer(ctx, make_pointcloud_frame_dtype(pointcloud_num_points), 8)
            if pointcloud_num_points is not None
            else None
        )
        self.is_running = ctx.Value("b", True, lock=False)
        self.vr_ready, self.camera_ready, self.pointcloud_ready = (ctx.Event() for _ in range(3))
        self.camera_depth_scale = ctx.Value("d", 0.0, lock=False)
        self.camera_serial = ctx.Array("c", 32, lock=False)
        self.camera_geometry = ctx.Array("c", 2048, lock=False)
