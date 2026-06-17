"""Camera extrinsics loader.

Loads per-camera extrinsics from the bundled calib/cameras.json data file.
Supports eye-to-hand (static camera) and eye-in-hand (end-effector mounted).

Intrinsics (K matrix) are read from the RealSense hardware at runtime, not from
this calibration file. They are stored into HDF5 /meta at recording time for
self-contained episodes.

Usage:
    calib = CameraCalib()
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
            # 优先查找项目根目录下的 configs/cameras.json（外提后位置）
            # fallback 到包内旧路径 calib/cameras.json
            pkg_dir = Path(__file__).resolve().parent
            project_root = pkg_dir.parent.parent
            new_path = project_root / "configs" / "cameras.json"
            old_path = pkg_dir / "calib" / "cameras.json"
            calib_path = str(new_path) if new_path.exists() else str(old_path)
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
            entry = CameraCalibEntry(
                serial=cam["serial"],
                type=cam["type"],
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

    # ---- public API ----

    @property
    def camera_names(self) -> list[str]:
        return list(self._entries.keys())

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
        """Return extrinsics values for HDF5 /meta attributes.

        Contains camera_serial, camera_type, and either
        camera_T_base_camera or camera_T_eef_camera (4x4 flat list).

        camera_K is not included here — intrinsics are read from the
        RealSense hardware at recording time and written to HDF5 separately.
        """
        entry = self._entries[cam_name]
        meta = {
            "camera_serial": entry.serial,
            "camera_type": entry.type,
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
        T = calib.get_extrinsics("cam_static")
        print(f"\n[cam_static] T_base_camera:\n{T}")

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
