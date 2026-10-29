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
    for serial, camera in cameras.items():
        if not serial.strip():
            raise ValueError("camera serial must be non-empty")
        validate_rigid_transform(
            camera["T_base_from_camera"], label=f"camera {serial} T_base_from_camera",
        )
    return cameras


def load_camera_extrinsics(path: str | Path | None = None) -> dict[str, np.ndarray]:
    cameras = _read_cameras(CAMERAS_PATH if path is None else Path(path))
    return {
        serial: np.asarray(camera["T_base_from_camera"], dtype=np.float64)
        for serial, camera in cameras.items()
    }


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
    cameras = _read_cameras(json_path) if json_path.exists() else {}
    entry = {"T_base_from_camera": T_base_from_camera.tolist()}
    if calibration_capture is not None:
        entry["calibration_capture"] = calibration_capture
    cameras[serial] = entry
    write_json(cameras, json_path, cancelled=cancelled)
