"""Load cameras.json poses (XYZ meters, WXYZ) and bind transforms by device serial."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from dexmani_real.calibration import CAMERAS_PATH
from dexmani_real.utils.geometry import validate_unit_quaternion_wxyz


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
        self._transforms: dict[str, np.ndarray] = {}
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
            if serial in self._transforms:
                raise ValueError(f"camera calibration serial {serial!r} appears more than once")
            pose = camera["pose"]
            self._transforms[serial] = pose_to_matrix(pose["position"], pose["orientation"])

    def for_serial(self, serial: str) -> np.ndarray:
        """Return an owned, read-only T_base_camera, including after process spawn."""
        try:
            transform = self._transforms[serial].copy()
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
