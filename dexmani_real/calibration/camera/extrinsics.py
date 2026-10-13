"""Load cameras.json poses (XYZ meters, WXYZ) as validated transforms for runtime lookup."""

from __future__ import annotations

import logging

__all__ = ["CameraExtrinsics", "CameraExtrinsicsEntry"]

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dexmani_real.calibration import CAMERAS_PATH
from dexmani_real.utils.geometry import validate_rigid_transform, validate_unit_quaternion_wxyz


def _pose_to_matrix(position: list[float], orientation: list[float]) -> np.ndarray:
    """Convert XYZ meters and a WXYZ quaternion to a float64 (4, 4) transform."""
    from scipy.spatial.transform import Rotation as R

    position_value = np.asarray(position, dtype=np.float64)
    orientation_value = np.asarray(orientation, dtype=np.float64)
    if position_value.shape != (3,) or orientation_value.shape != (4,):
        raise ValueError("camera pose position/orientation must have shapes (3,) and (4,)")
    if not np.all(np.isfinite(position_value)) or not np.all(np.isfinite(orientation_value)):
        raise ValueError("camera pose must contain only finite values")
    orientation_value = validate_unit_quaternion_wxyz(
        orientation_value, name="camera calibration orientation"
    )
    orientation_value = orientation_value / np.linalg.norm(orientation_value)
    px, py, pz = (float(value) for value in position_value)
    w, x, y, z = (float(value) for value in orientation_value)
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = [px, py, pz]
    T[:3, :3] = R.from_quat([x, y, z, w]).as_matrix()  # scipy uses xyzw
    return T


@dataclass(frozen=True)
class CameraExtrinsicsEntry:
    serial: str
    T_world_camera: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.serial, str) or not self.serial.strip():
            raise ValueError("camera serial must be a non-empty string")
        owned = validate_rigid_transform(self.T_world_camera, label="T_world_camera")
        owned.setflags(write=False)
        object.__setattr__(self, "T_world_camera", owned)

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        """Re-run validation and restore read-only arrays after process spawn."""
        return (
            type(self),
            (self.serial, self.T_world_camera),
        )


class CameraExtrinsics:
    def __init__(self, calib_path: str | Path | None = None):
        if calib_path is None:
            calib_path = CAMERAS_PATH
        self.calib_path = Path(calib_path).resolve()
        self._entries: dict[str, CameraExtrinsicsEntry] = {}
        self._load()

    def _load(self) -> None:
        if not self.calib_path.exists():
            raise FileNotFoundError(
                f"Calibration file not found: {self.calib_path}\n"
                f"Create it at {self.calib_path} with format:\n"
                '  {"camera_0": {"serial": "...", "type": "eye_to_hand", '
                '"pose": {"position": [x,y,z], "orientation": [w,x,y,z]}}}'
            )
        payload = self.calib_path.read_bytes()
        raw = json.loads(payload)
        if not isinstance(raw, dict):
            raise TypeError("camera calibration root must be an object")

        for cam_name, cam in raw.items():
            if not isinstance(cam, dict):
                raise TypeError(f"camera calibration entry {cam_name!r} must be an object")
            pose = cam.get("pose")
            if not isinstance(pose, dict):
                raise ValueError(
                    f"camera calibration entry {cam_name!r} requires pose.position and pose.orientation"
                )
            if cam["type"] != "eye_to_hand":
                raise ValueError(f"camera calibration entry {cam_name!r} must be eye_to_hand")
            T = _pose_to_matrix(pose["position"], pose["orientation"])

            entry = CameraExtrinsicsEntry(
                serial=cam["serial"],
                T_world_camera=T,
            )
            if any(existing.serial == entry.serial for existing in self._entries.values()):
                raise ValueError(
                    f"camera calibration serial {entry.serial!r} appears more than once"
                )
            self._entries[cam_name] = entry

    def resolve_name_by_serial(self, serial: str) -> str:
        """Bind a connected device to a calibration; serials are unique at load."""
        for name, entry in self._entries.items():
            if entry.serial == serial:
                return name
        known = {name: entry.serial for name, entry in self._entries.items()}
        raise KeyError(
            f"No camera in {self.calib_path.name} has serial '{serial}'. Known: {known}"
        )

    def get_extrinsics(self, cam_name: str) -> np.ndarray:
        """Return a copy of the fixed (4,4) camera pose in xArm base."""
        return self._entries[cam_name].T_world_camera.copy()

    def get_entry(self, cam_name: str) -> CameraExtrinsicsEntry:
        """Return a named fixed calibration with its owned, read-only transform."""
        return self._entries[cam_name]

    def __repr__(self) -> str:
        cameras = ", ".join(f"{n} ({e.serial})" for n, e in self._entries.items())
        return f"CameraExtrinsics({cameras})"


def load_optional_camera_extrinsics(path=None):
    """Resolve once per recording session; missing extrinsics do not block Raw."""

    try:
        return CameraExtrinsics(path)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        logging.getLogger(__name__).warning("Recording without camera extrinsics: %s", exc)
        return None
