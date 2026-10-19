"""Build at most one float32 [N,6] base-frame cloud per newest camera sequence; no camera SDK or frame queue."""

from __future__ import annotations

import logging
import json
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np

from dexmani_real.calibration.camera.extrinsics import CameraExtrinsics
from dexmani_real.sensor.pointcloud.config import PointCloudConfig
from dexmani_real.sensor.camera.geometry import RGBDGeometry
from dexmani_real.sensor.pointcloud.processing import build_point_cloud
from dexmani_real.utils.log import configure_logging

if TYPE_CHECKING:
    from dexmani_real.ipc.channels import SensorChannels

logger = logging.getLogger(__name__)

_IDLE_POLL_S = 0.001


@dataclass(frozen=True)
class PointCloudWorkerConfig:
    """Clouds inherit camera sequence and oldest receive time; RGB/cloud consumers match RGB-D."""

    pointcloud: PointCloudConfig
    camera_calibration: CameraExtrinsics
    table_plane_abcd: tuple[float, float, float, float] | None = None


class PointCloudWorker:
    """Own point-cloud preprocessing state and publish at most once per camera sequence."""

    def __init__(self, shared: "SensorChannels", config: PointCloudWorkerConfig):
        self.shared = shared
        self.config = config

    def run(self):
        """Consume only the newest camera sequence and publish fixed ``[N,6]`` clouds."""
        configure_logging()
        static_inputs = self.load_static_inputs()
        if static_inputs is None:
            return
        self.geometry, self.depth_scale_m, self.base_from_color = static_inputs
        logger.debug(
            "pointcloud config=%s table_plane_abcd=%s",
            json.dumps(asdict(self.config.pointcloud), sort_keys=True, separators=(",", ":")),
            json.dumps(self.config.table_plane_abcd, separators=(",", ":")),
        )

        self.last_camera_sequence = 0
        ready = False

        try:
            while self.shared.is_running.value:
                latest_sequence = self.shared.camera_ring.latest_sequence
                if latest_sequence <= self.last_camera_sequence:
                    time.sleep(_IDLE_POLL_S)
                    continue
                result = self.shared.camera_ring.read_latest()
                if result is None:
                    time.sleep(_IDLE_POLL_S)
                    continue
                data, camera_published_ns, camera_sequence = result
                header = data["header"]
                color, depth_raw = data["rgb"][0], data["depth"][0]
                if camera_sequence <= self.last_camera_sequence:
                    continue
                self.last_camera_sequence = camera_sequence

                processing_started_ns = time.monotonic_ns()
                cloud = build_point_cloud(
                    depth_raw=depth_raw,
                    color=color,
                    depth_scale_m=self.depth_scale_m,
                    geometry=self.geometry,
                    T_xarm_base_from_color=self.base_from_color,
                    table_plane_abcd=self.config.table_plane_abcd,
                    config=self.config.pointcloud,
                )
                if cloud is None:
                    continue
                camera_header = header[0]
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
            geometry = RGBDGeometry.from_dict(
                json.loads(self.shared.camera_geometry.value.decode("utf-8"))
            )
            depth_scale_m = float(self.shared.camera_depth_scale.value)
            if not np.isfinite(depth_scale_m) or depth_scale_m <= 0.0:
                raise ValueError("camera depth scale must be finite and positive")
            serial = self.shared.camera_serial.value.decode("utf-8")
            base_from_color = self.config.camera_calibration.for_serial(serial)
            # The aligned color pixel grid requires color-camera deprojection.
            return geometry.aligned_depth_to_color(), depth_scale_m, base_from_color
        return None


def run_pointcloud_worker(shared: "SensorChannels", config: PointCloudWorkerConfig) -> None:
    PointCloudWorker(shared, config).run()
