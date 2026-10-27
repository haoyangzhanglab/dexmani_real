"""按相机 serial 读写基座系外参，位置单位为米，四元数使用 WXYZ。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from dexmani_real.calibration import CAMERAS_PATH
from dexmani_real.geometry.pose import (
    validate_rigid_transform,
    validate_unit_quaternion_wxyz,
)
from dexmani_real.utils.atomic_io import atomic_json_dump


def pose_to_matrix(position, orientation) -> np.ndarray:
    position = np.asarray(position, dtype=np.float64)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("camera position must be finite XYZ meters")
    quaternion = validate_unit_quaternion_wxyz(orientation, name="camera calibration orientation")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = position
    transform[:3, :3] = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    return transform


def load_camera_extrinsics(path: str | Path | None = None) -> dict[str, np.ndarray]:
    path = CAMERAS_PATH if path is None else Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    transforms = {}
    for camera in raw.values():
        if camera["type"] != "eye_to_hand":
            raise ValueError("camera calibration must be eye_to_hand")
        serial = camera["serial"]
        if not isinstance(serial, str) or not serial.strip() or serial in transforms:
            raise ValueError(f"invalid or duplicate camera serial: {serial!r}")
        pose = camera["pose"]
        transforms[serial] = pose_to_matrix(pose["position"], pose["orientation"])
    return transforms


def load_optional_camera_extrinsics(
    path: str | Path | None = None,
) -> dict[str, np.ndarray] | None:
    if path is None and not CAMERAS_PATH.exists():
        return None
    return load_camera_extrinsics(path)


def save_camera_calibration(
    T_base_from_camera: np.ndarray,
    serial: str,
    json_path: Path,
    *,
    calibration_capture: dict[str, object] | None = None,
    cancelled=None,
) -> None:
    T_base_from_camera = validate_rigid_transform(T_base_from_camera, label="T_base_from_camera")
    if not serial.strip():
        raise ValueError("camera serial must be non-empty")
    rot = Rotation.from_matrix(T_base_from_camera[:3, :3])
    quat_wxyz = rot.as_quat()[[3, 0, 1, 2]]
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
        if cam_data["serial"] == serial:
            cam_name = name
            break
    else:
        idx = 0
        while f"camera_{idx}" in existing:
            idx += 1
        cam_name = f"camera_{idx}"

    existing[cam_name] = entry
    atomic_json_dump(existing, json_path, ensure_ascii=False, cancelled=cancelled)
