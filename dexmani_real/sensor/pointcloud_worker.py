"""Latest-only native RGB-D to realtime point-cloud worker.

The worker owns no camera SDK object and never queues camera payloads. It
observes the newest committed camera-ring sequence, builds at most one cloud
for that sequence, and publishes a fixed ``float32[N,6]`` xArm-base payload.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

from dexmani_real.config.camera_calib import CameraCalib
from dexmani_real.config.pointcloud import PointCloudConfig
from dexmani_real.ipc.schema import (
    SUPPORTED_POINT_CLOUD_COUNTS,
    make_pointcloud_frame_dtype,
    validate_point_cloud_array,
)
from dexmani_real.sensor.camera_geometry import RGBDGeometry
from dexmani_real.sensor.camera_worker import CameraHealth
from dexmani_real.sensor.pointcloud import POINT_CLOUD_POLICY_ID, build_point_cloud
from dexmani_real.utils.log import get_logger

if TYPE_CHECKING:
    from dexmani_real.ipc.channels import RuntimeChannels

logger = get_logger(__name__)

_IDLE_POLL_S = 0.001
_METRICS_LOG_INTERVAL_S = 5.0


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
    """Resolved processing and freshness policy for the realtime worker."""

    pointcloud: PointCloudConfig = field(default_factory=PointCloudConfig)
    camera_calibration: CameraCalib = field(default_factory=CameraCalib)
    table_plane_abcd: tuple[float, float, float, float] | None = None
    max_input_age_s: float = 0.25

    def __post_init__(self) -> None:
        if not isinstance(self.pointcloud, PointCloudConfig):
            raise TypeError("pointcloud must be a PointCloudConfig")
        if not isinstance(self.camera_calibration, CameraCalib):
            raise TypeError("camera_calibration must be a preloaded CameraCalib snapshot")
        if self.pointcloud.num_points not in SUPPORTED_POINT_CLOUD_COUNTS:
            raise ValueError(
                "realtime point-cloud count must be one of "
                f"{sorted(SUPPORTED_POINT_CLOUD_COUNTS)}"
            )
        if self.table_plane_abcd is not None:
            plane = tuple(float(value) for value in self.table_plane_abcd)
            if len(plane) != 4 or not np.all(np.isfinite(plane)):
                raise ValueError("table_plane_abcd must contain four finite values")
            norm = np.linalg.norm(plane[:3])
            if norm <= 0.0 or plane[2] / norm <= 0.0:
                raise ValueError("table_plane_abcd normal must point upward")
            object.__setattr__(self, "table_plane_abcd", plane)
        if not np.isfinite(self.max_input_age_s) or self.max_input_age_s <= 0.0:
            raise ValueError("max_input_age_s must be finite and positive")

    @classmethod
    def from_runtime(
        cls,
        runtime: object,
        *,
        num_points: int,
    ) -> "PointCloudLoopConfig":
        environment = getattr(runtime, "environment")
        camera = getattr(runtime, "camera")
        table = environment.table
        return cls(
            pointcloud=replace(getattr(runtime, "pointcloud"), num_points=num_points),
            table_plane_abcd=table.plane_abcd if table.enabled else None,
            max_input_age_s=float(camera.max_frame_age_s),
        )


def _shared_text(value: object) -> str:
    payload = getattr(value, "value", b"")
    if isinstance(payload, bytes):
        return payload.split(b"\x00", 1)[0].decode("utf-8")
    return str(payload).split("\x00", 1)[0]


def _resolve_base_from_color(
    shared: "RuntimeChannels", calibration: CameraCalib
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
    calibration: CameraCalib,
) -> tuple[RGBDGeometry, float, np.ndarray] | None:
    """Wait for camera-owned geometry and resolve the verified static transform."""
    while shared.is_running.value:
        shared.set_heartbeat("pointcloud", time.monotonic())
        if not shared.is_ready("camera"):
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
        base_from_depth = _validate_transform(
            base_from_color @ geometry.T_color_from_depth,
            label="T_xarm_base_from_depth",
        )
        return geometry, depth_scale_m, base_from_depth
    return None


def _camera_frame_is_usable(
    header: np.ndarray,
    *,
    now_ns: int,
    max_input_age_ns: int,
) -> bool:
    record = header[0]
    source_ns = int(record["source_monotonic_ns"])
    camera_publish_ns = int(record["publish_monotonic_ns"])
    return bool(
        int(record["camera_generation"]) > 0
        and int(record["camera_health"]) == int(CameraHealth.OK)
        and not bool(record["clock_reset"])
        and 0 < source_ns <= camera_publish_ns <= now_ns
        and now_ns - source_ns <= max_input_age_ns
    )


def pointcloud_loop(shared: "RuntimeChannels", config: PointCloudLoopConfig) -> None:
    """Consume only the newest camera sequence and publish fixed ``[N,6]`` clouds."""
    if not isinstance(config, PointCloudLoopConfig):
        raise TypeError("pointcloud_loop requires a PointCloudLoopConfig")
    cfg = config
    if not bool(shared.pointcloud_requested.value):
        raise RuntimeError("pointcloud_loop started without pointcloud_requested")
    expected_dtype = make_pointcloud_frame_dtype(cfg.pointcloud.num_points)
    if shared.pointcloud_ring.dtype != expected_dtype:
        raise RuntimeError(
            "pointcloud ring dtype does not match PointCloudLoopConfig num_points"
        )

    shared.set_heartbeat("pointcloud", time.monotonic())
    static_inputs = _load_static_inputs(shared, cfg.camera_calibration)
    if static_inputs is None:
        return
    geometry, depth_scale_m, base_from_depth = static_inputs
    logger.info(
        "pointcloud policy: id=%s config_sha256=%s config=%s table_plane_abcd=%s",
        POINT_CLOUD_POLICY_ID,
        cfg.pointcloud.sha256,
        json.dumps(cfg.pointcloud.to_dict(), sort_keys=True, separators=(",", ":")),
        json.dumps(cfg.table_plane_abcd, separators=(",", ":")),
    )

    last_camera_sequence = 0
    frames_processed = 0
    frames_published = 0
    frames_skipped = 0
    empty_clouds = 0
    stale_after_compute = 0
    compute_ms: deque[float] = deque(maxlen=2048)
    source_to_publish_ms: deque[float] = deque(maxlen=2048)
    last_log_s = time.monotonic()
    ready = False
    max_input_age_ns = int(cfg.max_input_age_s * 1e9)

    try:
        while shared.is_running.value:
            shared.set_heartbeat("pointcloud", time.monotonic())
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
            frames_skipped += max(0, camera_sequence - last_camera_sequence - 1)
            last_camera_sequence = camera_sequence
            now_ns = time.monotonic_ns()
            if not _camera_frame_is_usable(
                header,
                now_ns=now_ns,
                max_input_age_ns=max_input_age_ns,
            ):
                continue

            started_ns = time.monotonic_ns()
            cloud = build_point_cloud(
                depth_raw=depth_raw,
                color=color,
                depth_scale_m=depth_scale_m,
                geometry=geometry,
                T_xarm_base_from_depth=base_from_depth,
                table_plane_abcd=cfg.table_plane_abcd,
                config=cfg.pointcloud,
            )
            finished_ns = time.monotonic_ns()
            compute_ms.append((finished_ns - started_ns) / 1e6)
            frames_processed += 1
            if cloud is None:
                empty_clouds += 1
                continue
            validate_point_cloud_array(
                cloud,
                num_points=cfg.pointcloud.num_points,
                label="build_point_cloud output",
            )

            publish_ns = time.monotonic_ns()
            if publish_ns - int(header[0]["source_monotonic_ns"]) > max_input_age_ns:
                stale_after_compute += 1
                continue
            record = np.zeros(1, dtype=expected_dtype)
            camera_header = header[0]
            record["source_camera_sequence"][0] = np.uint64(camera_sequence)
            record["source_monotonic_ns"][0] = camera_header["source_monotonic_ns"]
            record["camera_publish_monotonic_ns"][0] = camera_header[
                "publish_monotonic_ns"
            ]
            record["publish_monotonic_ns"][0] = np.uint64(publish_ns)
            record["camera_generation"][0] = camera_header["camera_generation"]
            record["depth_frame_number"][0] = camera_header["depth_frame_number"]
            record["color_frame_number"][0] = camera_header["color_frame_number"]
            record["point_cloud"][0] = cloud
            shared.pointcloud_ring.write(record)
            committed_ns = time.monotonic_ns()
            source_to_publish_ms.append(
                (committed_ns - int(camera_header["source_monotonic_ns"])) / 1e6
            )
            frames_published += 1
            if not ready:
                shared.set_ready("pointcloud")
                ready = True
                logger.info(
                    "pointcloud_loop: ready (shape=(%d,6), frame=xarm_base)",
                    cfg.pointcloud.num_points,
                )

            now_s = time.monotonic()
            if now_s - last_log_s >= _METRICS_LOG_INTERVAL_S:
                values = np.asarray(compute_ms, dtype=np.float64)
                source_values = np.asarray(
                    source_to_publish_ms,
                    dtype=np.float64,
                )
                p50 = float(np.percentile(values, 50)) if values.size else 0.0
                p95 = float(np.percentile(values, 95)) if values.size else 0.0
                source_p95 = (
                    float(np.percentile(source_values, 95))
                    if source_values.size
                    else 0.0
                )
                logger.info(
                    "pointcloud_loop: processed=%d published=%d skipped=%d "
                    "empty=%d stale_after_compute=%d compute_ms_p50=%.2f p95=%.2f "
                    "source_to_publish_ms_p95=%.2f",
                    frames_processed,
                    frames_published,
                    frames_skipped,
                    empty_clouds,
                    stale_after_compute,
                    p50,
                    p95,
                    source_p95,
                )
                last_log_s = now_s
    finally:
        logger.info(
            "pointcloud_loop: exited processed=%d published=%d skipped=%d empty=%d "
            "stale_after_compute=%d",
            frames_processed,
            frames_published,
            frames_skipped,
            empty_clouds,
            stale_after_compute,
        )


__all__ = ["PointCloudLoopConfig", "pointcloud_loop"]
