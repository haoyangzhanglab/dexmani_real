"""Camera extrinsics loader.

Loads per-camera extrinsics from a bundled cameras.json data file.
Supports eye-to-hand (static camera) and eye-in-hand (end-effector mounted).

The storage format is a human-readable pose::

       {
         "camera_0": {
           "serial": "241322110633",
           "type": "eye_to_hand",
           "pose": {
             "position": [0.5, -0.35, 0.82],
             "orientation": [1.0, 0.0, 0.0, 0.0]
           }
         }
       }

   ``position`` is XYZ in meters, ``orientation`` is WXYZ quaternion.
   Converted to a 4×4 homogeneous matrix at load time.

Intrinsics (K matrix) are read from the RealSense hardware at runtime and stored
into HDF5 /meta for self-contained episodes.  New calibration entries may also
carry a ``calibration_capture`` snapshot (profile, distortion, firmware, SDK)
for provenance; it is diagnostic and is not used as runtime intrinsics.

Usage:
    calib = CameraCalib()
    cam = calib.resolve_name_by_serial(connected_serial)  # robust: pick by serial
    T_base_camera = calib.get_extrinsics(cam)                                     # eye-to-hand
    T_base_camera = calib.get_extrinsics(cam, T_base_eef=T_base_eef)              # eye-in-hand
    meta = calib.to_meta_dict(cam, expected_serial=connected_serial)  # verified /meta
"""

from __future__ import annotations

__all__ = ["CameraCalib", "CameraCalibEntry"]

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _pose_to_matrix(position: list[float], orientation: list[float]) -> np.ndarray:
    """Convert pose (position XYZ + orientation WXYZ quaternion) to 4×4 homogeneous matrix.

    Args:
        position: [x, y, z] in meters.
        orientation: [w, x, y, z] quaternion (scalar-first).

    Returns:
        (4, 4) homogeneous transformation matrix, float64.
    """
    from scipy.spatial.transform import Rotation as R

    px, py, pz = float(position[0]), float(position[1]), float(position[2])
    w, x, y, z = float(orientation[0]), float(orientation[1]), float(orientation[2]), float(orientation[3])
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = [px, py, pz]
    T[:3, :3] = R.from_quat([x, y, z, w]).as_matrix()  # scipy uses xyzw
    return T


@dataclass
class CameraCalibEntry:
    serial: str
    type: str  # "eye_to_hand" | "eye_in_hand"
    T_world_camera: np.ndarray | None = None  # (4,4) eye-to-hand, camera in WORLD frame
    T_eef_camera: np.ndarray | None = None  # (4,4) eye-in-hand

    def __post_init__(self):
        if self.type not in ("eye_to_hand", "eye_in_hand"):
            raise ValueError(f"camera_type must be 'eye_to_hand' or 'eye_in_hand', got '{self.type}'")
        if self.type == "eye_to_hand" and self.T_world_camera is None:
            raise ValueError("eye_to_hand camera requires T_world_camera")
        if self.type == "eye_in_hand" and self.T_eef_camera is None:
            raise ValueError("eye_in_hand camera requires T_eef_camera")


class CameraCalib:

    def __init__(self, calib_path: str | None = None):
        if calib_path is None:
            calib_path = self._resolve_default_path()
        self.calib_path = Path(calib_path).resolve()
        self._entries: dict[str, CameraCalibEntry] = {}
        self._load()

    @classmethod
    def _resolve_default_path(cls) -> str:
        """Return the single authoritative cameras.json path."""
        pkg_dir = Path(__file__).resolve().parent  # dexmani_real/config/
        return str(pkg_dir / "cameras.json")

    def _load(self) -> None:
        if not self.calib_path.exists():
            raise FileNotFoundError(
                f"Calibration file not found: {self.calib_path}\n"
                "Create it at dexmani_real/config/cameras.json with format:\n"
                '  {"camera_0": {"serial": "...", "type": "eye_to_hand", '
                '"pose": {"position": [x,y,z], "orientation": [w,x,y,z]}}}'
            )
        with open(self.calib_path) as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise TypeError("camera calibration root must be an object")

        for cam_name, cam in raw.items():
            if not isinstance(cam, dict):
                raise TypeError(f"camera calibration entry {cam_name!r} must be an object")
            pose = cam.get("pose")
            if not isinstance(pose, dict):
                raise ValueError(f"camera calibration entry {cam_name!r} requires pose.position and pose.orientation")
            T = _pose_to_matrix(pose["position"], pose["orientation"])
            camera_type = cam["type"]
            T_world_camera = T if camera_type == "eye_to_hand" else None
            T_eef_camera = T if camera_type == "eye_in_hand" else None

            entry = CameraCalibEntry(
                serial=cam["serial"],
                type=camera_type,
                T_world_camera=T_world_camera,
                T_eef_camera=T_eef_camera,
            )
            self._entries[cam_name] = entry


    def resolve_name_by_serial(self, serial: str) -> str:
        """Return the camera_name whose entry matches ``serial``.

        This is the robust way to pick a calibration entry: select by the
        actually-connected camera's serial instead of a hard-coded name, so a
        stale/placeholder entry under a familiar name (e.g. "camera_0") can
        never be used by mistake.

        Raises:
            KeyError: if no entry matches, or if more than one does.
        """
        matches = [n for n, e in self._entries.items() if e.serial == serial]
        if not matches:
            known = {n: e.serial for n, e in self._entries.items()}
            raise KeyError(f"No camera in {self.calib_path.name} has serial '{serial}'. Known: {known}")
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

        For eye_to_hand: returns the static T_world_camera from config (WORLD frame,
            consistent with recorded eef_pos / arm_ee).
        For eye_in_hand: computes T_base_eef @ T_eef_camera (in the frame of the
            passed eef pose), requires T_base_eef.
        """
        entry = self._entries[cam_name]
        if entry.type == "eye_to_hand":
            return entry.T_world_camera.copy()  # type: ignore[union-attr]  # __post_init__ guarantees non-None for eye_to_hand
        if T_base_eef is None:
            raise ValueError(f"Camera '{cam_name}' is eye_in_hand; T_base_eef (4,4 FK matrix) is required")
        return T_base_eef @ entry.T_eef_camera

    def to_meta_dict(self, cam_name: str, expected_serial: str | None = None) -> dict:
        """Return extrinsics values for HDF5 /meta attributes.

        Contains camera_serial, camera_type, and either
        camera_T_world_camera (eye_to_hand, world frame) or camera_T_eef_camera
        (eye_in_hand) as a 4x4 flat list.

        camera_K is not included here — intrinsics are read from the
        RealSense hardware at recording time and written to HDF5 separately.

        If ``expected_serial`` is given, verifies the entry belongs to that
        physical camera and raises ValueError on mismatch — so a wrong
        camera_name can never silently poison recorded data.
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
        return f"CameraCalib({cameras})"
