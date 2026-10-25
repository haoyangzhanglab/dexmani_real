"""Publish float32 [N,6] base-frame clouds once per newest camera sequence."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, cast

import numpy as np

from dexmani_real.geometry.camera import CameraSnapshot
from dexmani_real.geometry.plane import validate_table_plane
from dexmani_real.perception.pointcloud.config import PointCloudConfig
from dexmani_real.perception.pointcloud.processing import build_point_cloud
from dexmani_real.utils.log import configure_logging

if TYPE_CHECKING:
    from dexmani_real.ipc.channels import SensorChannels
    from dexmani_real.ipc.ring import SharedMemoryRingBuffer

logger = logging.getLogger(__name__)

IDLE_POLL_S = 0.001


def run_pointcloud_worker(
    shared: "SensorChannels", pointcloud: PointCloudConfig, camera: CameraSnapshot,
    table_plane_abcd: tuple[float, float, float, float] | None,
) -> None:
    configure_logging()
    if camera.T_xarm_base_from_color is None:
        raise ValueError("pointcloud requires camera extrinsics")
    if pointcloud.remove_table:
        validate_table_plane(table_plane_abcd)
    logger.debug(
        "pointcloud config=%s table_plane_abcd=%s",
        pointcloud, table_plane_abcd,
    )

    last_camera_sequence = 0
    ready = False
    # PolicySession starts this worker only with point-cloud storage; SensorChannels
    # requires camera storage for that allocation. Both handles live for the process.
    camera_ring = cast("SharedMemoryRingBuffer", shared.camera_ring)
    pointcloud_ring = cast("SharedMemoryRingBuffer", shared.pointcloud_ring)

    try:
        while shared.is_running.value:
            latest_sequence = camera_ring.latest_sequence
            if latest_sequence <= last_camera_sequence:
                time.sleep(IDLE_POLL_S)
                continue
            result = camera_ring.read_latest()
            if result is None:
                time.sleep(IDLE_POLL_S)
                continue
            data, camera_published_ns, camera_sequence = result
            color, depth_raw = data["rgb"][0], data["depth"][0]
            if camera_sequence <= last_camera_sequence:
                continue
            last_camera_sequence = camera_sequence

            processing_started_ns = time.monotonic_ns()
            cloud = build_point_cloud(
                depth_raw=depth_raw,
                color=color,
                depth_scale_m=camera.depth_scale_m,
                intrinsics=camera.color,
                T_xarm_base_from_color=camera.T_xarm_base_from_color,
                table_plane_abcd=table_plane_abcd,
                config=pointcloud,
            )
            if cloud is None:
                continue
            camera_header = data["header"][0]
            # Processing must not refresh the camera's original observation time.
            pointcloud_ring.write(
                source_camera_sequence=np.uint64(camera_sequence),
                timestamp_ns=camera_header["timestamp_ns"],
                camera_received_ns=camera_header["received_ns"],
                camera_published_ns=np.uint64(camera_published_ns),
                processing_started_ns=np.uint64(processing_started_ns),
                point_cloud=cloud,
            )
            if not ready:
                shared.pointcloud_ready.set()
                ready = True
                logger.debug(
                    "run_pointcloud_worker: ready (shape=(%d,6), frame=xarm_base)",
                    pointcloud.num_points,
                )
    finally:
        shared.pointcloud_ready.clear()
        logger.debug("run_pointcloud_worker: exited")
