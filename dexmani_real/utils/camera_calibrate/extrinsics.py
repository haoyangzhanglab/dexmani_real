"""按相机 serial 读写 T_base_from_camera，平移单位为米。"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from dexmani_real.config.calibration_data import CAMERAS_PATH
from dexmani_real.geometry.pose import validate_rigid_transform
from dexmani_real.utils.io import write_json


def _read_cameras(path: Path) -> dict:
    cameras = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cameras, dict):
        raise TypeError("camera calibration JSON must be an object keyed by serial")
    return cameras


def load_camera_extrinsics(serial: str, path: str | Path | None = None) -> np.ndarray:
    if not isinstance(serial, str) or not serial.strip():
        raise ValueError("camera serial must be non-empty")
    cameras = _read_cameras(CAMERAS_PATH if path is None else Path(path))
    return _parse_camera_extrinsics(cameras, serial)


def _parse_camera_extrinsics(cameras: dict, serial: str) -> np.ndarray:
    return validate_rigid_transform(
        cameras[serial]["T_base_from_camera"], label=f"camera {serial} T_base_from_camera",
    )


def load_optional_camera_extrinsics(
    serial: str,
    path: str | Path | None = None,
) -> np.ndarray | None:
    """仅默认文件或其中的 serial 条目缺失时省略外参；显式路径与坏条目仍报错。"""
    if path is not None:
        return load_camera_extrinsics(serial, path)
    if not isinstance(serial, str) or not serial.strip():
        raise ValueError("camera serial must be non-empty")
    try:
        cameras = _read_cameras(CAMERAS_PATH)
    except FileNotFoundError:
        return None
    if serial not in cameras:
        return None
    return _parse_camera_extrinsics(cameras, serial)


def save_camera_calibration(
    T_base_from_camera: np.ndarray,
    serial: str,
    json_path: Path,
    *,
    calibration_capture: dict[str, object],
    check_abort: Callable[[], None],
) -> None:
    """保存来自已连接相机、经求解器验证的外参及标定信息。"""
    try:
        cameras = _read_cameras(json_path)
    except FileNotFoundError:
        cameras = {}
    cameras[serial] = {
        "T_base_from_camera": T_base_from_camera.tolist(),
        "calibration_capture": calibration_capture,
    }
    write_json(cameras, json_path, check_abort=check_abort)
