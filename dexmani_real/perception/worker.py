"""每个最新相机序列处理一次，保留原始来源时间。"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, cast

import numpy as np

from dexmani_real.config.perception import PointCloudConfig
from dexmani_real.geometry.camera import CameraSnapshot
from dexmani_real.perception.pointcloud import PointCloudProcessor
from dexmani_real.utils.log import configure_logging

if TYPE_CHECKING:
    from dexmani_real.ipc.ring import SharedRing
    from dexmani_real.ipc.sensors import SensorChannels

logger = logging.getLogger(__name__)

IDLE_POLL_S = 0.001


def run_pointcloud_worker(
    shared: "SensorChannels",
    pointcloud: PointCloudConfig,
    camera: CameraSnapshot,
    table_plane_abcd: tuple[float, float, float, float] | None,
) -> None:
    configure_logging()
    processor = PointCloudProcessor(
        depth_scale_m=camera.depth_scale_m,
        intrinsics=camera.color,
        T_xarm_base_from_color=camera.T_xarm_base_from_color,
        table_plane_abcd=table_plane_abcd,
        config=pointcloud,
    )
    logger.debug(
        "pointcloud config=%s table_plane_abcd=%s",
        pointcloud,
        table_plane_abcd,
    )

    last_camera_sequence = 0
    # 会话仅在相机和点云通道均已分配时启动本进程。
    camera_ring = cast("SharedRing", shared.camera_ring)
    pointcloud_ring = cast("SharedRing", shared.pointcloud_ring)

    try:
        while shared.is_running.value:
            latest_sequence = camera_ring.latest_sequence
            if latest_sequence <= last_camera_sequence:
                time.sleep(IDLE_POLL_S)
                continue
            result = camera_ring.read(sequence=latest_sequence)
            if result is None:
                time.sleep(IDLE_POLL_S)
                continue
            data, camera_sequence = result
            last_camera_sequence = camera_sequence

            cloud = processor.process(data["depth"], data["rgb"])
            if cloud is None:
                continue
            if not np.isfinite(cloud).all():
                raise ValueError("pointcloud contains nonfinite values")
            pointcloud_ring.write(
                source_camera_sequence=camera_sequence,
                timestamp_ns=data["timestamp_ns"],
                point_cloud=cloud,
            )
            if not shared.pointcloud_ready.is_set():
                shared.pointcloud_ready.set()
                logger.debug(
                    "run_pointcloud_worker: ready (shape=(%d,6), frame=xarm_base)",
                    pointcloud.num_points,
                )
    finally:
        shared.pointcloud_ready.clear()
        logger.debug("run_pointcloud_worker: exited")
