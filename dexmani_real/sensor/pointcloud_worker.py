"""Latest-only depth-to-color RGB-D to realtime point-cloud worker.

The worker owns no camera SDK object and never queues camera payloads. It
observes the newest committed camera-ring sequence, builds at most one cloud
for that sequence, and publishes a fixed ``float32[N,6]`` xArm-base payload.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from dexmani_real.calibration.camera.extrinsics import CameraExtrinsics
from dexmani_real.config.pointcloud import PointCloudConfig
from dexmani_real.ipc.schema import (
    make_pointcloud_frame_dtype,
    validate_point_cloud_array,
)
from dexmani_real.sensor.camera.geometry import RGBDGeometry
from dexmani_real.sensor.pointcloud import (
    POINT_CLOUD_COLOR_SOURCE,
    POINT_CLOUD_POLICY_ID,
    POINT_CLOUD_SAMPLING,
    POINT_CLOUD_TRANSFORM,
    build_point_cloud,
)
from dexmani_real.utils.log import get_logger

if TYPE_CHECKING:
    from dexmani_real.config.experiment import ExperimentConfig
    from dexmani_real.ipc.channels import RuntimeChannels

logger = get_logger(__name__)

_IDLE_POLL_S = 0.001


def _validate_transform(value: np.ndarray, *, label: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64).reshape(4, 4)
    if (
        not np.all(np.isfinite(transform))
        or not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9)
        or not np.allclose(
            transform[:3, :3].T @ transform[:3, :3],
            np.eye(3),
            atol=1e-6,
        )
        or not np.isclose(np.linalg.det(transform[:3, :3]), 1.0, atol=1e-6)
    ):
        raise ValueError(f"{label} must be a finite rigid homogeneous transform")
    return transform


@dataclass(frozen=True)
class PointCloudLoopConfig:
    """Resolved processing policy for the realtime worker.

    Each cloud keeps its camera sequence and acquisition time. Consumers
    check cloud freshness; only joint RGB/cloud consumers retrieve the matching
    RGB-D sample. Published clouds are otherwise self-contained.
    """

    pointcloud: PointCloudConfig
    camera_calibration: CameraExtrinsics
    table_plane_abcd: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.pointcloud, PointCloudConfig):
            raise TypeError("pointcloud must be a PointCloudConfig")
        if not isinstance(self.camera_calibration, CameraExtrinsics):
            raise TypeError(
                "camera_calibration must be a preloaded CameraExtrinsics snapshot"
            )
        if self.table_plane_abcd is not None:
            plane = tuple(float(value) for value in self.table_plane_abcd)
            if len(plane) != 4 or not np.all(np.isfinite(plane)):
                raise ValueError("table_plane_abcd must contain four finite values")
            norm = np.linalg.norm(plane[:3])
            if norm <= 0.0 or plane[2] / norm <= 0.0:
                raise ValueError("table_plane_abcd normal must point upward")
            object.__setattr__(self, "table_plane_abcd", plane)

    @classmethod
    def from_runtime(
        cls,
        runtime: "ExperimentConfig",
        *,
        num_points: int,
    ) -> "PointCloudLoopConfig":
        table = runtime.environment.table
        return cls(
            pointcloud=replace(runtime.pointcloud, num_points=num_points),
            camera_calibration=CameraExtrinsics(),
            table_plane_abcd=table.plane_abcd if table.enabled else None,
        )


def _shared_text(value: object) -> str:
    payload = getattr(value, "value", b"")
    if isinstance(payload, bytes):
        return payload.split(b"\x00", 1)[0].decode("utf-8")
    return str(payload).split("\x00", 1)[0]


def _resolve_base_from_color(
    shared: "RuntimeChannels", calibration: CameraExtrinsics
) -> np.ndarray:
    serial = _shared_text(shared.camera_serial).strip()
    if not serial:
        raise RuntimeError(
            "camera did not publish a serial for point-cloud calibration"
        )
    camera_name = calibration.resolve_name_by_serial(serial)
    metadata = calibration.to_meta_dict(camera_name, expected_serial=serial)
    if metadata.get("camera_type") != "eye_to_hand":
        raise ValueError(
            "realtime point-cloud worker currently requires an eye_to_hand camera; "
            "eye_in_hand needs a separately synchronized arm-pose contract"
        )
    return _validate_transform(
        calibration.get_extrinsics(camera_name),
        label="T_xarm_base_from_color",
    )


def _load_static_inputs(
    shared: "RuntimeChannels",
    calibration: CameraExtrinsics,
) -> tuple[RGBDGeometry, float, np.ndarray] | None:
    """Wait for camera-owned geometry and resolve the verified static transform."""
    while shared.is_running.value:
        if not shared.camera_ready.is_set():
            time.sleep(_IDLE_POLL_S)
            continue
        geometry_text = _shared_text(shared.camera_geometry).strip()
        depth_scale_m = float(shared.camera_depth_scale.value)
        if not geometry_text or not np.isfinite(depth_scale_m) or depth_scale_m <= 0.0:
            time.sleep(_IDLE_POLL_S)
            continue
        geometry_payload = json.loads(geometry_text)
        if not isinstance(geometry_payload, dict):
            raise TypeError("camera geometry shared metadata must encode an object")
        geometry = RGBDGeometry.from_dict(geometry_payload)
        base_from_color = _resolve_base_from_color(shared, calibration)
        # The camera ring carries depth aligned onto the color pixel grid.
        # Its deprojection frame is therefore the color-camera frame.
        return geometry.aligned_depth_to_color(), depth_scale_m, base_from_color
    return None


def pointcloud_loop(shared: "RuntimeChannels", config: PointCloudLoopConfig) -> None:
    """Consume only the newest camera sequence and publish fixed ``[N,6]`` clouds."""
    if not isinstance(config, PointCloudLoopConfig):
        raise TypeError("pointcloud_loop requires a PointCloudLoopConfig")
    cfg = config
    expected_dtype = make_pointcloud_frame_dtype(cfg.pointcloud.num_points)
    if shared.pointcloud_ring.dtype != expected_dtype:
        raise RuntimeError(
            "pointcloud ring dtype does not match PointCloudLoopConfig num_points"
        )

    static_inputs = _load_static_inputs(shared, cfg.camera_calibration)
    if static_inputs is None:
        return
    geometry, depth_scale_m, base_from_color = static_inputs
    logger.debug(
        "pointcloud policy: id=%s color_source=%s sampling=%s "
        "transform=%s config=%s table_plane_abcd=%s",
        POINT_CLOUD_POLICY_ID,
        POINT_CLOUD_COLOR_SOURCE,
        POINT_CLOUD_SAMPLING,
        POINT_CLOUD_TRANSFORM,
        json.dumps(cfg.pointcloud.to_dict(), sort_keys=True, separators=(",", ":")),
        json.dumps(cfg.table_plane_abcd, separators=(",", ":")),
    )

    last_camera_sequence = 0
    ready = False

    try:
        while shared.is_running.value:
            latest_sequence = int(shared.camera_ring.latest_sequence)
            if latest_sequence <= last_camera_sequence:
                time.sleep(_IDLE_POLL_S)
                continue
            result = shared.camera_ring.read_latest()
            if result is None:
                time.sleep(_IDLE_POLL_S)
                continue
            header, color, depth_raw, camera_sequence = result
            if camera_sequence <= last_camera_sequence:
                continue
            last_camera_sequence = camera_sequence

            cloud = build_point_cloud(
                depth_raw=depth_raw,
                color=color,
                depth_scale_m=depth_scale_m,
                geometry=geometry,
                T_xarm_base_from_color=base_from_color,
                table_plane_abcd=cfg.table_plane_abcd,
                config=cfg.pointcloud,
            )
            if cloud is None:
                continue
            validate_point_cloud_array(
                cloud,
                num_points=cfg.pointcloud.num_points,
                label="build_point_cloud output",
            )

            record = np.zeros(1, dtype=expected_dtype)
            camera_header = header[0]
            record["source_camera_sequence"][0] = np.uint64(camera_sequence)
            record["timestamp_ns"][0] = camera_header["timestamp_ns"]
            record["point_cloud"][0] = cloud
            shared.pointcloud_ring.write(record)
            if not ready:
                shared.pointcloud_ready.set()
                ready = True
                logger.info(
                    "pointcloud_loop: ready (shape=(%d,6), frame=xarm_base)",
                    cfg.pointcloud.num_points,
                )
    except Exception:
        shared.workflow_failed.value = True
        raise
    finally:
        logger.info("pointcloud_loop: exited")


__all__ = ["PointCloudLoopConfig", "pointcloud_loop"]
