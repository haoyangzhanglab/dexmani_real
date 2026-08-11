"""RealSense discovery and color-stream startup for camera calibration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

import numpy as np

from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

_OPENCV_DISTORTION_MODELS = frozenset({"none", "brown_conrady", "distortion.none", "distortion.brown_conrady"})


@dataclass(frozen=True)
class CameraStream:
    pipeline: Any
    serial: str
    intrinsics: np.ndarray
    distortion: np.ndarray
    capture_metadata: dict[str, Any]


def choose_camera_serial(requested_serial: str | None, connected_serials: Sequence[str]) -> str:
    """Resolve one camera identity without silently choosing among devices."""
    serials = [str(serial) for serial in connected_serials]
    if requested_serial is not None:
        if requested_serial not in serials:
            raise RuntimeError(f"RealSense serial {requested_serial!r} not found; connected={serials}")
        return requested_serial
    if len(serials) != 1:
        raise RuntimeError(f"Expected exactly one RealSense or --serial; connected={serials}")
    return serials[0]


def _realsense_module() -> Any:
    import pyrealsense2 as rs

    return rs


def select_camera_serial(requested_serial: str | None, *, rs_module: Any | None = None) -> str:
    rs = _realsense_module() if rs_module is None else rs_module
    devices = rs.context().query_devices()
    serials = [
        str(device.get_info(rs.camera_info.serial_number))
        for device in devices
        if device.supports(rs.camera_info.serial_number)
    ]
    return choose_camera_serial(requested_serial, serials)


def _camera_parameters(intrinsics: Any) -> tuple[np.ndarray, np.ndarray]:
    camera_matrix = np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.ppx],
            [0.0, intrinsics.fy, intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.asarray(intrinsics.coeffs, dtype=np.float64)
    values = np.concatenate((camera_matrix.reshape(-1), distortion.reshape(-1)))
    if not np.all(np.isfinite(values)):
        raise RuntimeError("RealSense returned non-finite camera intrinsics")
    return camera_matrix, distortion


def _capture_metadata(
    *,
    intrinsics: Any,
    firmware: str,
    sdk_version: str,
    actual_fps: int,
    actual_format: str,
    distortion_model: str,
    config_sha256: str,
) -> dict[str, Any]:
    return {
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "firmware": firmware,
        "sdk_version": sdk_version,
        "resolved_config_sha256": config_sha256,
        "color_profile": {
            "width": int(intrinsics.width),
            "height": int(intrinsics.height),
            "fps": actual_fps,
            "format": actual_format,
        },
        "intrinsics": {
            "fx": float(intrinsics.fx),
            "fy": float(intrinsics.fy),
            "ppx": float(intrinsics.ppx),
            "ppy": float(intrinsics.ppy),
            "distortion_model": distortion_model,
            "distortion_coeffs": [float(value) for value in intrinsics.coeffs],
        },
    }


def start_camera_stream(
    selected_serial: str,
    camera_runtime: Any,
    config_sha256: str,
    *,
    rs_module: Any | None = None,
) -> CameraStream:
    """Start, identify, validate, and warm one RealSense color stream."""
    if not selected_serial:
        raise ValueError("selected_serial must be non-empty")
    rs = _realsense_module() if rs_module is None else rs_module
    pipeline = rs.pipeline()
    stream_config = rs.config()
    started = False
    try:
        stream_config.enable_device(selected_serial)
        stream_config.enable_stream(
            rs.stream.color,
            int(camera_runtime.width),
            int(camera_runtime.height),
            rs.format.bgr8,
            int(camera_runtime.fps),
        )
        profile = pipeline.start(stream_config)
        started = True
        device = profile.get_device()
        serial = str(device.get_info(rs.camera_info.serial_number))
        if serial != selected_serial:
            raise RuntimeError(f"Started RealSense {serial}, expected {selected_serial}")

        firmware = str(device.get_info(rs.camera_info.firmware_version))
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = color_profile.get_intrinsics()
        distortion_model = str(intrinsics.model)
        if distortion_model.lower() not in _OPENCV_DISTORTION_MODELS:
            raise RuntimeError(
                f"RealSense distortion model {distortion_model!r} is not directly compatible with OpenCV solvePnP"
            )
        camera_matrix, distortion = _camera_parameters(intrinsics)
        actual_fps = int(color_profile.fps())
        actual_format_value = color_profile.format()
        actual_format = str(actual_format_value)
        expected_format = str(rs.format.bgr8)
        profile_matches = (
            int(intrinsics.width) == int(camera_runtime.width)
            and int(intrinsics.height) == int(camera_runtime.height)
            and actual_fps == int(camera_runtime.fps)
            and actual_format_value == rs.format.bgr8
        )
        if not profile_matches:
            raise RuntimeError(
                "RealSense color profile differs from the resolved runtime: "
                f"actual={intrinsics.width}x{intrinsics.height}@{actual_fps} {actual_format}, "
                f"expected={camera_runtime.width}x{camera_runtime.height}@{camera_runtime.fps} {expected_format}"
            )
        sdk_version = str(getattr(rs, "__version__", "unknown"))
        metadata = _capture_metadata(
            intrinsics=intrinsics,
            firmware=firmware,
            sdk_version=sdk_version,
            actual_fps=actual_fps,
            actual_format=actual_format,
            distortion_model=distortion_model,
            config_sha256=config_sha256,
        )
        print(f"  序列号: {serial}  固件: {firmware}  SDK: {sdk_version}")
        print(
            f"  实际 profile: {intrinsics.width}x{intrinsics.height}@{actual_fps} {actual_format}; "
            f"fx={intrinsics.fx:.1f} fy={intrinsics.fy:.1f} distortion={distortion_model}"
        )
        for _ in range(int(camera_runtime.warmup_frames)):
            pipeline.wait_for_frames()
        return CameraStream(pipeline, serial, camera_matrix, distortion, metadata)
    except Exception:
        if started:
            try:
                pipeline.stop()
            except Exception:
                logger.warning("RealSense cleanup failed during startup abort", exc_info=True)
        raise


__all__ = [
    "CameraStream",
    "choose_camera_serial",
    "select_camera_serial",
    "start_camera_stream",
]
