"""Build the immutable camera/device metadata captured at episode start.

Both recording backends use the same metadata snapshot.  Keeping this helper
outside the RecorderIO process makes the in-process path a transport change
only: it still publishes the identical schema-v17 metadata and camera
geometry contract.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from dexmani_real.config.camera_calib import CameraCalib
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


def shared_text(value: bytes, *, default: str | None) -> str | None:
    """Decode a null-padded shared-memory text value."""
    encoded = value.rstrip(b"\x00")
    return encoded.decode("utf-8") if encoded else default


def validate_camera_geometry(
    camera_profile_json: str,
    *,
    configured_align_mode: str,
    camera_K: np.ndarray,
) -> tuple[str, str, str]:
    """Validate the live RGB-D profile against the v17 geometry contract."""
    if configured_align_mode != "depth_to_color":
        raise ValueError(
            "camera metadata requires align_mode='depth_to_color'"
        )
    try:
        profile = json.loads(camera_profile_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("camera_actual_profile_json is not valid JSON") from exc
    if not isinstance(profile, dict):
        raise RuntimeError("camera_actual_profile_json must contain a JSON object")

    actual_align_mode = str(profile.get("align_mode", ""))
    common_viewport = str(profile.get("common_viewport", ""))
    output_optical_frame = str(profile.get("output_optical_frame", ""))
    if actual_align_mode != configured_align_mode:
        raise RuntimeError(
            "camera alignment does not match recording configuration: "
            f"actual={actual_align_mode!r}, configured={configured_align_mode!r}"
        )
    if common_viewport != "color" or output_optical_frame != "camera_color_optical":
        raise RuntimeError(
            "camera profile must use the color common viewport and "
            "camera_color_optical output frame"
        )
    output_intrinsics = profile.get("output_intrinsics")
    if not isinstance(output_intrinsics, dict):
        raise RuntimeError("camera_actual_profile_json is missing output_intrinsics")
    try:
        profile_K = np.array(
            [
                [float(output_intrinsics["fx"]), 0.0, float(output_intrinsics["cx"])],
                [0.0, float(output_intrinsics["fy"]), float(output_intrinsics["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("camera output_intrinsics are malformed") from exc
    if not np.allclose(profile_K, camera_K, rtol=1e-6, atol=1e-6):
        raise RuntimeError(
            "camera_K does not match the actual common-viewport intrinsics"
        )
    return actual_align_mode, common_viewport, output_optical_frame


def build_start_metadata(
    shared: Any,
    *,
    task_label: str,
    operator: str,
    align_mode: str,
) -> dict[str, Any]:
    """Snapshot the same metadata for direct and RecorderIO episode writes."""
    camera_K_values = list(shared.camera_K)
    try:
        camera_K = np.asarray(camera_K_values, dtype=np.float64).reshape(3, 3)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "camera_K is unavailable or malformed at recorder START"
        ) from exc
    if (
        not np.all(np.isfinite(camera_K))
        or camera_K[0, 0] <= 0.0
        or camera_K[1, 1] <= 0.0
        or not np.allclose(camera_K[2], [0.0, 0.0, 1.0], rtol=0.0, atol=1e-9)
    ):
        raise RuntimeError("camera_K is unavailable or malformed at recorder START")

    depth_scale = (
        float(shared.camera_depth_scale.value)
        if shared.camera_depth_scale.value != 0.0
        else None
    )
    camera_serial = shared_text(shared.camera_serial.value, default=None)
    camera_firmware = (
        shared_text(shared.camera_firmware.value, default="unknown") or "unknown"
    )
    camera_sdk_version = (
        shared_text(shared.camera_sdk_version.value, default="unknown") or "unknown"
    )
    camera_profile_json = shared_text(shared.camera_profile.value, default="{}") or "{}"
    actual_align_mode, common_viewport, output_optical_frame = validate_camera_geometry(
        camera_profile_json,
        configured_align_mode=align_mode,
        camera_K=camera_K,
    )
    camera_pointcloud_config_json = (
        shared_text(shared.camera_pointcloud_config.value, default="{}") or "{}"
    )
    arm_identity_json = (
        shared_text(
            shared.arm_device_identity.value, default='{"status":"unavailable"}'
        )
        or '{"status":"unavailable"}'
    )
    hand_identity_json = shared_text(
        shared.hand_device_identity.value, default='{"status":"unavailable"}'
    )

    calibration = CameraCalib()
    try:
        camera_name = (
            calibration.resolve_name_by_serial(camera_serial) if camera_serial else None
        )
    except (KeyError, FileNotFoundError):
        camera_name = None
        logger.warning(
            "Camera serial %s not found in cameras.json — no extrinsics in /meta",
            camera_serial,
        )

    return {
        "task_label": task_label,
        "operator": operator,
        "calib": calibration,
        "camera_K": camera_K,
        "camera_name": camera_name,
        "camera_serial": camera_serial,
        "depth_scale": depth_scale,
        "camera_metadata": {
            "camera_firmware": camera_firmware,
            "camera_sdk_version": camera_sdk_version,
            "camera_actual_profile_json": camera_profile_json,
            "camera_alignment_mode": actual_align_mode,
            "camera_common_viewport": common_viewport,
            "camera_K_optical_frame": output_optical_frame,
            "camera_output_optical_frame": output_optical_frame,
            "camera_pointcloud_config_json": camera_pointcloud_config_json,
            "arm_device_identity_json": arm_identity_json,
            "hand_device_identity_json": hand_identity_json or '{"status":"disabled"}',
        },
    }
