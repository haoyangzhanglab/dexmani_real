"""Persist and load camera extrinsics (XYZ meters, WXYZ) by device serial."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from dexmani_real.calibration import CAMERAS_PATH
from dexmani_real.utils.atomic_io import atomic_json_dump
from dexmani_real.utils.geometry import validate_rigid_transform, validate_unit_quaternion_wxyz


def pose_to_matrix(position, orientation) -> np.ndarray:
    position = np.asarray(position, dtype=np.float64)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("camera position must be finite XYZ meters")
    quaternion = validate_unit_quaternion_wxyz(orientation, name="camera calibration orientation")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = position
    transform[:3, :3] = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    return transform


class CameraExtrinsics:
    def __init__(self, calib_path: str | Path | None = None):
        self.calib_path = Path(CAMERAS_PATH if calib_path is None else calib_path).resolve()
        raw = json.loads(self.calib_path.read_text())
        if not isinstance(raw, dict):
            raise ValueError("camera calibration root must be an object")
        self.transforms: dict[str, np.ndarray] = {}
        for name, camera in raw.items():
            if not isinstance(camera, dict) or not isinstance(camera.get("pose"), dict):
                raise ValueError(
                    f"camera calibration {name!r} requires pose.position and pose.orientation"
                )
            if camera["type"] != "eye_to_hand":
                raise ValueError(f"camera calibration {name!r} must be eye_to_hand")
            serial = camera["serial"]
            if not isinstance(serial, str) or not serial.strip():
                raise ValueError("camera serial must be a non-empty string")
            if serial in self.transforms:
                raise ValueError(f"camera calibration serial {serial!r} appears more than once")
            pose = camera["pose"]
            self.transforms[serial] = pose_to_matrix(pose["position"], pose["orientation"])

    def for_serial(self, serial: str) -> np.ndarray:
        """Return an owned, read-only T_base_from_camera, including after process spawn."""
        try:
            transform = self.transforms[serial].copy()
        except KeyError:
            raise KeyError(f"No camera in {self.calib_path.name} has serial {serial!r}") from None
        transform.setflags(write=False)
        return transform


def load_optional_camera_extrinsics(path=None):
    """Missing or invalid extrinsics do not block physical Raw recording."""
    try:
        return CameraExtrinsics(path)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        logging.getLogger(__name__).warning("Recording without camera extrinsics: %s", exc)
        return None


def calibration_capture_metadata(
    *,
    width: int,
    height: int,
    requested_fps: int,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
    method: str,
    sample_count: int,
    position_errors_mm: np.ndarray,
    rotation_errors_deg: np.ndarray,
) -> dict[str, object]:
    """Summarize solver residuals and camera geometry for publication."""

    def summary(values: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(values.mean()),
            "rms": float(np.sqrt(np.mean(values**2))),
            "max": float(values.max()),
        }

    return {
        "width": width,
        "height": height,
        "requested_fps": requested_fps,
        "intrinsics": intrinsics.tolist(),
        "distortion": distortion.tolist(),
        "method": method,
        "sample_count": sample_count,
        "position_error_mm": summary(position_errors_mm),
        "rotation_error_deg": summary(rotation_errors_deg),
        "calibrated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def save_camera_calibration(
    T_base_from_camera: np.ndarray,
    serial: str,
    json_path: Path,
    *,
    calibration_capture: dict[str, object] | None = None,
    cancelled=None,
) -> None:
    """Write an xArm-base camera pose to cameras.json, preserving other entries."""
    T_base_from_camera = validate_rigid_transform(T_base_from_camera, label="T_base_from_camera")
    if not serial.strip():
        raise ValueError("camera serial must be non-empty")
    rot = Rotation.from_matrix(T_base_from_camera[:3, :3])
    quat_wxyz = rot.as_quat()[[3, 0, 1, 2]]  # scipy xyzw → wxyz
    pos = T_base_from_camera[:3, 3]

    entry = {
        "serial": serial,
        "type": "eye_to_hand",
        "pose": {
            "position": pos.tolist(),
            "orientation": quat_wxyz.tolist(),
        },
    }
    if calibration_capture is not None:
        entry["calibration_capture"] = calibration_capture

    if json_path.exists():
        with open(json_path) as f:
            existing = json.load(f)
    else:
        existing = {}

    for name, cam_data in existing.items():
        if cam_data.get("serial") == serial:
            cam_name = name
            break
    else:
        idx = 0
        while f"camera_{idx}" in existing:
            idx += 1
        cam_name = f"camera_{idx}"

    existing[cam_name] = entry
    try:
        json.dumps(existing, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("camera calibration payload must contain finite JSON values") from exc
    atomic_json_dump(existing, json_path, ensure_ascii=False, cancelled=cancelled)
    print(f"  calibration written → {json_path} (camera: {cam_name})")
