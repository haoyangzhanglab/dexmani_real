"""按相机 serial 读写 T_base_from_camera，平移单位为米。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dexmani_real.config.calibration_data import CAMERAS_PATH
from dexmani_real.geometry.pose import validate_rigid_transform
from dexmani_real.utils.io import write_json


def _read_cameras(path: Path) -> dict:
    cameras = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cameras, dict):
        raise ValueError("camera calibration JSON must be an object keyed by serial")
    return cameras


def load_camera_extrinsics(serial: str, path: str | Path | None = None) -> np.ndarray:
    if not isinstance(serial, str) or not serial.strip():
        raise ValueError("camera serial must be non-empty")
    cameras = _read_cameras(CAMERAS_PATH if path is None else Path(path))
    return validate_rigid_transform(
        cameras[serial]["T_base_from_camera"], label=f"camera {serial} T_base_from_camera",
    )


def load_optional_camera_extrinsics(
    serial: str,
    path: str | Path | None = None,
) -> np.ndarray | None:
    if path is None and not CAMERAS_PATH.exists():
        return None
    return load_camera_extrinsics(serial, path)


def save_camera_calibration(
    T_base_from_camera: np.ndarray,
    serial: str,
    json_path: Path,
    *,
    calibration_capture: dict[str, object] | None = None,
    cancelled=None,
) -> None:
    if not isinstance(serial, str) or not serial.strip():
        raise ValueError("camera serial must be non-empty")
    transform = validate_rigid_transform(
        T_base_from_camera, label=f"camera {serial} T_base_from_camera",
    )
    cameras = _read_cameras(json_path) if json_path.exists() else {}
    entry = {"T_base_from_camera": transform.tolist()}
    if calibration_capture is not None:
        entry["calibration_capture"] = calibration_capture
    cameras[serial] = entry
    write_json(cameras, json_path, cancelled=cancelled)
