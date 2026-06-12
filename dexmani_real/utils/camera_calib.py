"""Camera calibration loader.

Loads per-camera intrinsics and extrinsics from config/calib/cameras.json.
Supports eye-to-hand (static camera) and eye-in-hand (end-effector mounted).

Usage:
    calib = CameraCalib("config/calib/cameras.json")
    K = calib.get_K("camera_0")
    T_base_camera = calib.get_extrinsics("camera_0")                              # eye-to-hand
    T_base_camera = calib.get_extrinsics("camera_0", T_base_eef=T_base_eef)       # eye-in-hand
    meta = calib.to_meta_dict("camera_0")  # for HDF5 /meta attributes
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CameraCalibEntry:
    serial: str
    type: str  # "eye_to_hand" | "eye_in_hand"
    K: np.ndarray  # (3,3)
    T_base_camera: np.ndarray | None = None  # (4,4) eye-to-hand
    T_eef_camera: np.ndarray | None = None  # (4,4) eye-in-hand

    def __post_init__(self):
        if self.type not in ("eye_to_hand", "eye_in_hand"):
            raise ValueError(
                f"camera_type must be 'eye_to_hand' or 'eye_in_hand', got '{self.type}'"
            )
        if self.type == "eye_to_hand" and self.T_base_camera is None:
            raise ValueError("eye_to_hand camera requires T_base_camera")
        if self.type == "eye_in_hand" and self.T_eef_camera is None:
            raise ValueError("eye_in_hand camera requires T_eef_camera")


class CameraCalib:
    def __init__(self, calib_path: str | None = None):
        if calib_path is None:
            calib_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "config", "calib", "cameras.json"
            )
        self.calib_path = Path(calib_path).resolve()
        self._entries: dict[str, CameraCalibEntry] = {}
        self._load()

    def _load(self) -> None:
        if not self.calib_path.exists():
            raise FileNotFoundError(
                f"Calibration file not found: {self.calib_path}\n"
                "Create it with format: https://..."
            )
        with open(self.calib_path) as f:
            raw = json.load(f)

        for cam_name, cam in raw.items():
            K = self._build_K(cam["K"])
            entry = CameraCalibEntry(
                serial=cam["serial"],
                type=cam["type"],
                K=K,
                T_base_camera=(
                    np.array(cam["T_base_camera"], dtype=np.float64)
                    if cam.get("T_base_camera") is not None
                    else None
                ),
                T_eef_camera=(
                    np.array(cam["T_eef_camera"], dtype=np.float64)
                    if cam.get("T_eef_camera") is not None
                    else None
                ),
            )
            self._entries[cam_name] = entry

    @staticmethod
    def _build_K(params: list[float]) -> np.ndarray:
        fx, fy, cx, cy = params
        return np.array(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64
        )

    # ---- public API ----

    @property
    def camera_names(self) -> list[str]:
        return list(self._entries.keys())

    def get_K(self, cam_name: str) -> np.ndarray:
        """Return 3x3 intrinsic matrix."""
        return self._entries[cam_name].K.copy()

    def get_intrinsics(self, cam_name: str) -> dict[str, float]:
        """Return {"fx", "fy", "cx", "cy"} as scalars."""
        K = self._entries[cam_name].K
        return {"fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2]}

    def get_extrinsics(
        self, cam_name: str, T_base_eef: np.ndarray | None = None
    ) -> np.ndarray:
        """Return T_base_camera (4,4).

        For eye_to_hand: returns the static T_base_camera from config.
        For eye_in_hand: computes T_base_eef @ T_eef_camera, requires T_base_eef.
        """
        entry = self._entries[cam_name]
        if entry.type == "eye_to_hand":
            return entry.T_base_camera.copy()
        if T_base_eef is None:
            raise ValueError(
                f"Camera '{cam_name}' is eye_in_hand; T_base_eef (4,4 FK matrix) is required"
            )
        return T_base_eef @ entry.T_eef_camera

    def to_meta_dict(self, cam_name: str) -> dict:
        """Return calibration values for HDF5 /meta attributes.

        Contains camera_serial, camera_type, camera_K (flat list),
        and either camera_T_base_camera or camera_T_eef_camera (4x4 flat list).
        """
        entry = self._entries[cam_name]
        K = self.get_intrinsics(cam_name)
        meta = {
            "camera_serial": entry.serial,
            "camera_type": entry.type,
            "camera_K": [float(K["fx"]), float(K["fy"]), float(K["cx"]), float(K["cy"])],
        }
        if entry.type == "eye_to_hand":
            meta["camera_T_base_camera"] = entry.T_base_camera.flatten().tolist()
        else:
            meta["camera_T_eef_camera"] = entry.T_eef_camera.flatten().tolist()
        return meta

    def __repr__(self) -> str:
        cameras = ", ".join(
            f"{n} ({e.type}, {e.serial})" for n, e in self._entries.items()
        )
        return f"CameraCalib({cameras})"


def example():
    import tempfile

    # Write a minimal calib file for demo
    demo = {
        "cam_static": {
            "serial": "000000000001",
            "type": "eye_to_hand",
            "K": [615.0, 615.0, 320.0, 240.0],
            "T_base_camera": [
                [1.0, 0.0, 0.0, 0.50],
                [0.0, 1.0, 0.0, -0.35],
                [0.0, 0.0, 1.0, 0.82],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
        "cam_wrist": {
            "serial": "000000000002",
            "type": "eye_in_hand",
            "K": [610.0, 610.0, 320.0, 240.0],
            "T_eef_camera": [
                [0.0, 1.0, 0.0, 0.05],
                [-1.0, 0.0, 0.0, -0.02],
                [0.0, 0.0, 1.0, 0.10],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(demo, f, indent=2)
        tmp_path = f.name

    try:
        calib = CameraCalib(tmp_path)
        print(calib)
        print(f"Cameras: {calib.camera_names}")

        # eye-to-hand
        K = calib.get_K("cam_static")
        T = calib.get_extrinsics("cam_static")
        print(f"\n[cam_static] K:\n{K}")
        print(f"[cam_static] T_base_camera:\n{T}")

        # eye-in-hand
        T_base_eef = np.eye(4)
        T_base_eef[:3, 3] = [0.3, 0.0, 0.5]
        T = calib.get_extrinsics("cam_wrist", T_base_eef=T_base_eef)
        print(f"\n[cam_wrist] T_base_camera (FK @ T_eef_camera):\n{T}")

        # meta
        print(f"\n[cam_static] meta: {calib.to_meta_dict('cam_static')}")
        print(f"[cam_wrist] meta: {calib.to_meta_dict('cam_wrist')}")
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    example()
