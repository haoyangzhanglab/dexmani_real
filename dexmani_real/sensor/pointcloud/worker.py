"""Publish float32 [N,6] base-frame clouds once per newest camera sequence."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np

from dexmani_real.calibration.camera.extrinsics import CameraExtrinsics
from dexmani_real.sensor.camera.geometry import CameraIntrinsics
from dexmani_real.sensor.pointcloud.config import PointCloudConfig
from dexmani_real.sensor.pointcloud.processing import build_point_cloud
from dexmani_real.utils.log import configure_logging

if TYPE_CHECKING:
    from dexmani_real.ipc.channels import SensorChannels

logger = logging.getLogger(__name__)

IDLE_POLL_S = 0.001


@dataclass(frozen=True)
class PointCloudWorkerConfig:
    pointcloud: PointCloudConfig
    camera_calibration: CameraExtrinsics
    table_plane_abcd: tuple[float, float, float, float] | None = None


class PointCloudWorker:
    def __init__(self, shared: "SensorChannels", config: PointCloudWorkerConfig):
        self.shared = shared
        self.config = config

    def run(self):
        configure_logging()
        static_inputs = self.load_static_inputs()
        if static_inputs is None:
            return
        intrinsics, depth_scale_m, base_from_color = static_inputs
        logger.debug(
            "pointcloud config=%s table_plane_abcd=%s",
            json.dumps(asdict(self.config.pointcloud), sort_keys=True, separators=(",", ":")),
            json.dumps(self.config.table_plane_abcd, separators=(",", ":")),
        )

        last_camera_sequence = 0
        ready = False

        try:
            while self.shared.is_running.value:
                latest_sequence = self.shared.camera_ring.latest_sequence
                if latest_sequence <= last_camera_sequence:
                    time.sleep(IDLE_POLL_S)
                    continue
                result = self.shared.camera_ring.read_latest()
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
                    depth_scale_m=depth_scale_m,
                    intrinsics=intrinsics,
                    T_xarm_base_from_color=base_from_color,
                    table_plane_abcd=self.config.table_plane_abcd,
                    config=self.config.pointcloud,
                )
                if cloud is None:
                    continue
                camera_header = data["header"][0]
                # Processing must not refresh the camera's original observation time.
                self.shared.pointcloud_ring.write(
                    source_camera_sequence=np.uint64(camera_sequence),
                    timestamp_ns=camera_header["timestamp_ns"],
                    camera_received_ns=camera_header["received_ns"],
                    camera_published_ns=np.uint64(camera_published_ns),
                    processing_started_ns=np.uint64(processing_started_ns),
                    point_cloud=cloud,
                )
                if not ready:
                    self.shared.pointcloud_ready.set()
                    ready = True
                    logger.debug(
                        "run_pointcloud_worker: ready (shape=(%d,6), frame=xarm_base)",
                        self.config.pointcloud.num_points,
                    )
        finally:
            self.shared.pointcloud_ready.clear()
            logger.debug("run_pointcloud_worker: exited")

    def load_static_inputs(self):
        """Wait for camera-owned geometry and resolve the verified static transform."""
        while self.shared.is_running.value:
            if not self.shared.camera_ready.wait(timeout=0.05):
                continue
            # Geometry/scale are published once, before ready; malformed values cannot recover here.
            intrinsics = CameraIntrinsics(
                **json.loads(self.shared.camera_intrinsics.value.decode("utf-8"))
            )
            depth_scale_m = float(self.shared.camera_depth_scale.value)
            if not np.isfinite(depth_scale_m) or depth_scale_m <= 0.0:
                raise ValueError("camera depth scale must be finite and positive")
            serial = self.shared.camera_serial.value.decode("utf-8")
            base_from_color = self.config.camera_calibration.for_serial(serial)
            # The aligned color pixel grid requires color-camera deprojection.
            return intrinsics, depth_scale_m, base_from_color
        return None


def run_pointcloud_worker(shared: "SensorChannels", config: PointCloudWorkerConfig) -> None:
    PointCloudWorker(shared, config).run()
