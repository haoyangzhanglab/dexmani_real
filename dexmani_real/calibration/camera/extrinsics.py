"""Load eye-to-hand or eye-in-hand extrinsics from cameras.json.

Each camera entry stores serial, type and pose (XYZ meters, WXYZ quaternion),
converted to a 4x4 transform at load time. Recorder START resolves the connected
serial and requires eye-to-hand calibration. Raw stores the static base-from-color
transform and aligned color-grid intrinsics supplied by the live camera.
The calibration file's calibration_capture section is diagnostic provenance.

Usage::

    calib = CameraExtrinsics()
    cam = calib.resolve_name_by_serial(connected_serial)
    T_base_camera = calib.get_extrinsics(cam)  # eye-in-hand also needs T_base_eef
"""

from __future__ import annotations

__all__ = ["CameraExtrinsics", "CameraExtrinsicsEntry"]

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dexmani_real.calibration import CAMERAS_PATH


def _pose_to_matrix(position: list[float], orientation: list[float]) -> np.ndarray:
    """Convert XYZ meters and a WXYZ quaternion to a float64 (4, 4) transform."""
    from scipy.spatial.transform import Rotation as R

    position_value = np.asarray(position, dtype=np.float64)
    orientation_value = np.asarray(orientation, dtype=np.float64)
    if position_value.shape != (3,) or orientation_value.shape != (4,):
        raise ValueError("camera pose position/orientation must have shapes (3,) and (4,)")
    if not np.all(np.isfinite(position_value)) or not np.all(np.isfinite(orientation_value)):
        raise ValueError("camera pose must contain only finite values")
    norm = float(np.linalg.norm(orientation_value))
    if norm <= 1e-12:
        raise ValueError("camera pose orientation must be a non-zero WXYZ quaternion")
    orientation_value = orientation_value / norm
    px, py, pz = (float(value) for value in position_value)
    w, x, y, z = (float(value) for value in orientation_value)
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = [px, py, pz]
    T[:3, :3] = R.from_quat([x, y, z, w]).as_matrix()  # scipy uses xyzw
    return T


@dataclass(frozen=True)
class CameraExtrinsicsEntry:
    serial: str
    type: str  # "eye_to_hand" | "eye_in_hand"
    T_world_camera: np.ndarray | None = None  # (4,4) eye-to-hand, camera in WORLD frame
    T_eef_camera: np.ndarray | None = None  # (4,4) eye-in-hand

    def __post_init__(self) -> None:
        if not isinstance(self.serial, str) or not self.serial.strip():
            raise ValueError("camera serial must be a non-empty string")
        if self.type not in ("eye_to_hand", "eye_in_hand"):
            raise ValueError(
                f"camera_type must be 'eye_to_hand' or 'eye_in_hand', got '{self.type}'"
            )
        if self.type == "eye_to_hand" and self.T_world_camera is None:
            raise ValueError("eye_to_hand camera requires T_world_camera")
        if self.type == "eye_in_hand" and self.T_eef_camera is None:
            raise ValueError("eye_in_hand camera requires T_eef_camera")
        for name in ("T_world_camera", "T_eef_camera"):
            transform = getattr(self, name)
            if transform is None:
                continue
            value = np.asarray(transform, dtype=np.float64)
            if (
                value.shape != (4, 4)
                or not np.all(np.isfinite(value))
                or not np.allclose(value[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9)
                or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(value[:3, :3]), 1.0, atol=1e-6)
            ):
                raise ValueError(f"{name} must be a finite rigid homogeneous transform")
            owned = value.copy()
            owned.setflags(write=False)
            object.__setattr__(self, name, owned)

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        """Re-run validation and restore read-only arrays after process spawn."""
        return (
            type(self),
            (self.serial, self.type, self.T_world_camera, self.T_eef_camera),
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
            T = _pose_to_matrix(pose["position"], pose["orientation"])
            camera_type = cam["type"]
            T_world_camera = T if camera_type == "eye_to_hand" else None
            T_eef_camera = T if camera_type == "eye_in_hand" else None

            entry = CameraExtrinsicsEntry(
                serial=cam["serial"],
                type=camera_type,
                T_world_camera=T_world_camera,
                T_eef_camera=T_eef_camera,
            )
            if any(existing.serial == entry.serial for existing in self._entries.values()):
                raise ValueError(
                    f"camera calibration serial {entry.serial!r} appears more than once"
                )
            self._entries[cam_name] = entry

    def resolve_name_by_serial(self, serial: str) -> str:
        """Select calibration by the connected camera's serial, not a fixed name.

        Raises KeyError if zero or multiple entries match.
        """
        matches = [n for n, e in self._entries.items() if e.serial == serial]
        if not matches:
            known = {n: e.serial for n, e in self._entries.items()}
            raise KeyError(
                f"No camera in {self.calib_path.name} has serial '{serial}'. Known: {known}"
            )
        if len(matches) > 1:
            raise KeyError(f"Multiple cameras share serial '{serial}': {matches}")
        return matches[0]

    def verify_serial(self, cam_name: str, actual_serial: str) -> None:
        """Assert entry ``cam_name`` belongs to the connected camera.

        Raises:
            ValueError: if the entry's serial differs from ``actual_serial``
                (i.e. the named calibration is for a different physical camera).
        """
        entry = self._entries[cam_name]
        if entry.serial != actual_serial:
            raise ValueError(
                f"Camera calibration mismatch: entry '{cam_name}' is for serial "
                f"'{entry.serial}', but the connected camera is '{actual_serial}'. "
                f"Fix cameras.json or select by serial (resolve_name_by_serial)."
            )

    def get_extrinsics(self, cam_name: str, T_base_eef: np.ndarray | None = None) -> np.ndarray:
        """Return the camera extrinsic (4,4).

        For eye_to_hand: returns the static T_world_camera from calibration;
            DexMani calibrates this world frame to xArm base.
        For eye_in_hand: computes T_base_eef @ T_eef_camera (in the frame of the
            passed eef pose), requires T_base_eef.
        """
        entry = self._entries[cam_name]
        if entry.type == "eye_to_hand":
            return entry.T_world_camera.copy()  # type: ignore[union-attr]  # __post_init__ guarantees non-None for eye_to_hand
        if T_base_eef is None:
            raise ValueError(
                f"Camera '{cam_name}' is eye_in_hand; T_base_eef (4,4 FK matrix) is required"
            )
        return T_base_eef @ entry.T_eef_camera

    def to_meta_dict(self, cam_name: str, expected_serial: str | None = None) -> dict:
        """Return calibration identity and pose for runtime calibration lookup.

        The transform is camera_T_world_camera (eye-to-hand) or camera_T_eef_camera
        (eye-in-hand). This dictionary is not the Raw metadata contract.
        Raises ValueError if the entry does not match a supplied expected_serial.
        """
        if expected_serial is not None:
            self.verify_serial(cam_name, expected_serial)
        entry = self._entries[cam_name]
        meta = {
            "camera_serial": entry.serial,
            "camera_type": entry.type,
        }
        if entry.type == "eye_to_hand":
            meta["camera_T_world_camera"] = entry.T_world_camera.flatten().tolist()  # type: ignore[union-attr]  # __post_init__ guarantees non-None for eye_to_hand
        else:
            meta["camera_T_eef_camera"] = entry.T_eef_camera.flatten().tolist()  # type: ignore[union-attr]  # __post_init__ guarantees non-None for eye_in_hand
        return meta

    def __repr__(self) -> str:
        cameras = ", ".join(f"{n} ({e.type}, {e.serial})" for n, e in self._entries.items())
        return f"CameraExtrinsics({cameras})"
