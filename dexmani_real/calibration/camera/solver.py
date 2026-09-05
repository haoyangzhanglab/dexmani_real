"""ArUco eye-to-hand calibration math and persistence without device ownership.

This module does not start cameras, open GUI windows, create worker processes,
or publish robot commands. ``motion.py`` owns arm-motion state and command
publication; ``session.py`` owns the
interactive device, GUI, sampling, and cleanup lifecycle.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from dexmani_real import PACKAGE_DIR
from dexmani_real.planning.kinematics.pose import rot6d_to_quat_wxyz
from dexmani_real.utils.atomic_io import atomic_json_dump

CAMERA_CALIBRATION_PATH = PACKAGE_DIR / "config" / "cameras.json"
ARUCO_DICT = cv2.aruco.DICT_7X7_50
ARUCO_DICT_NAME = "7x7_50"

_HAND_EYE_METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def eef_rpy_from_rot6d(rot6d: np.ndarray) -> np.ndarray:
    """Convert an arm EEF rot6d orientation to xyz RPY in radians."""
    quat_wxyz = rot6d_to_quat_wxyz(rot6d)
    rpy_rad = Rotation.from_quat(np.roll(quat_wxyz, -1)).as_euler("xyz", degrees=False)
    return np.asarray(rpy_rad, dtype=np.float64)


@dataclass(frozen=True)
class ArucoConfig:
    """ArUco marker detection parameters."""

    marker_size_m: float = 0.0982
    target_id: int | None = 1
    capture_frames: int = 5

    def __post_init__(self) -> None:
        if not np.isfinite(self.marker_size_m) or self.marker_size_m <= 0.0:
            raise ValueError("marker_size_m must be finite and positive")
        if self.target_id is not None and self.target_id < 0:
            raise ValueError("target_id must be non-negative or None")
        if not isinstance(self.capture_frames, int) or self.capture_frames < 1:
            raise ValueError("capture_frames must be at least 1")


@dataclass(frozen=True)
class CalibrationConfig:
    """Session tuning parameters for interactive camera calibration."""

    min_samples: int = 10
    max_consistency_std_mm: float = 5.0
    max_consistency_rot_std_deg: float = 3.0
    delta_pos_m: float = 0.008
    delta_rpy_rad: float = 0.03
    status_interval_frames: int = 50

    def __post_init__(self) -> None:
        if not isinstance(self.min_samples, int) or self.min_samples < 3:
            raise ValueError("min_samples must be at least 3")
        positive_fields = (
            "max_consistency_std_mm",
            "max_consistency_rot_std_deg",
            "delta_pos_m",
            "delta_rpy_rad",
        )
        for field_name in positive_fields:
            value = float(getattr(self, field_name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive")
        if (
            not isinstance(self.status_interval_frames, int)
            or self.status_interval_frames < 1
        ):
            raise ValueError("status_interval_frames must be at least 1")


def detect_aruco_pose(
    color_image: np.ndarray,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
    *,
    marker_size_m: float,
    target_id: int | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Detect ArUco marker and return (rvec, tvec) in camera frame, or None."""
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return None

    if target_id is not None:
        mask = ids.flatten() == target_id
        if not mask.any():
            return None
        corners = [c for c, m in zip(corners, mask) if m]

    half = marker_size_m / 2.0
    marker_points = np.array(
        [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]],
        dtype=np.float32,
    )

    for c in corners:
        _, rv, tv = cv2.solvePnP(
            marker_points, c, intrinsics, distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        return rv.flatten().astype(np.float64), tv.flatten().astype(np.float64)

    return None


def marker_corners_3d(marker_size_m: float) -> np.ndarray:
    """Marker corners in marker-local frame for drawFrameAxes."""
    s = marker_size_m / 2.0
    return np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float32)


def draw_calibration_overlay(
    image: np.ndarray,
    detector: cv2.aruco.ArucoDetector,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
    *,
    n_samples: int,
    min_samples: int,
    target_id: int | None,
    marker_corners: np.ndarray,
    marker_size_m: float,
) -> tuple[np.ndarray, bool]:
    """Draw ArUco detection boxes, axes, and status text on the color image.

    Returns:
        (annotated_bgr, detected) — detected is True when the target marker is visible.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    detected = False
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        for c, mid in zip(corners, ids.flatten()):
            if target_id is not None and mid != target_id:
                continue
            ok, rv, tv = cv2.solvePnP(
                marker_corners,
                c,
                intrinsics,
                distortion,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if ok:
                cv2.drawFrameAxes(
                    image, intrinsics, distortion, rv, tv, marker_size_m * 0.5
                )
                detected = True

    color = (0, 200, 0) if detected else (0, 0, 255)
    cv2.putText(
        image,
        f"MARKER {'OK' if detected else 'NOT FOUND'}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
    )
    hint = f"samples={n_samples}/{min_samples}  " + (
        "ENTER=calibrate"
        if n_samples >= min_samples
        else f"SPACE=capture (need >={min_samples})"
    )
    cv2.putText(
        image, hint, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
    )
    return image, detected


def _calibrate_eye_to_hand(
    tvec_ee2base: list[np.ndarray],
    rpy_ee2base: list[np.ndarray],
    rvec_marker2camera: list[np.ndarray],
    tvec_marker2camera: list[np.ndarray],
    method: int = cv2.CALIB_HAND_EYE_PARK,
) -> np.ndarray:
    """Solve eye-to-hand transform, returning T_base_camera (4x4).

    cv2.calibrateHandEye expects eye-in-hand inputs (gripper2base +
    target2cam).  For eye-to-hand we invert the robot poses so the return
    value is T_cam2base == T_base_camera.
    """
    rmat_ee2base = [
        Rotation.from_euler("xyz", rpy, degrees=False).as_matrix()
        for rpy in rpy_ee2base
    ]
    rmat_base2ee = [rmat.T for rmat in rmat_ee2base]
    tvec_base2ee = [
        (-rmat.T @ tv).reshape(3, 1) for rmat, tv in zip(rmat_ee2base, tvec_ee2base)
    ]
    rmat_marker2cam = [cv2.Rodrigues(rv)[0] for rv in rvec_marker2camera]
    tvec_marker2cam = [tv.reshape(3, 1) for tv in tvec_marker2camera]

    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        rmat_base2ee,
        tvec_base2ee,
        rmat_marker2cam,
        tvec_marker2cam,
        method=method,
    )

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_cam2base
    T[:3, 3] = t_cam2base.flatten()
    return T


def _compute_closed_loop_errors(
    T_base_camera: np.ndarray,
    tvec_ee2base: list[np.ndarray],
    rpy_ee2base: list[np.ndarray],
    rvec_marker2camera: list[np.ndarray],
    tvec_marker2camera: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-sample T_ee_marker consistency residuals.

    T_ee_marker must be identical across all samples (the marker is rigidly
    attached to the end-effector).  Any scatter indicates calibration error.

    Returns:
        (residuals_mm, residuals_deg) as (N,) arrays.
    """
    positions: list[np.ndarray] = []
    rotations: list[np.ndarray] = []

    for tvec_ee, rpy_ee, rvec_m2c, tvec_m2c in zip(
        tvec_ee2base,
        rpy_ee2base,
        rvec_marker2camera,
        tvec_marker2camera,
    ):
        T_base_ee = np.eye(4)
        T_base_ee[:3, :3] = Rotation.from_euler(
            "xyz", rpy_ee, degrees=False
        ).as_matrix()
        T_base_ee[:3, 3] = tvec_ee

        T_camera_marker = np.eye(4)
        T_camera_marker[:3, :3] = cv2.Rodrigues(rvec_m2c)[0]
        T_camera_marker[:3, 3] = tvec_m2c

        T_ee_marker = np.linalg.inv(T_base_ee) @ T_base_camera @ T_camera_marker
        positions.append(T_ee_marker[:3, 3].copy())
        R_ee_marker = T_ee_marker[:3, :3]
        det_val = float(np.linalg.det(R_ee_marker))
        if not np.isfinite(det_val) or abs(det_val - 1.0) > 0.01:
            raise ValueError(f"degenerate T_ee_marker rotation (det={det_val:.3f})")
        rotations.append(R_ee_marker)

    positions_arr = np.array(positions)
    mean_pos = positions_arr.mean(axis=0)
    residuals_mm = np.linalg.norm(positions_arr - mean_pos, axis=1) * 1000.0

    rots = Rotation.from_matrix(np.array(rotations))
    mean_rot = rots.mean()
    residuals_deg = np.degrees((mean_rot.inv() * rots).magnitude())
    return residuals_mm, residuals_deg


def calibrate_and_select(
    tvec_ee2base: list[np.ndarray],
    rpy_ee2base: list[np.ndarray],
    rvec_marker2camera: list[np.ndarray],
    tvec_marker2camera: list[np.ndarray],
) -> tuple[np.ndarray, str, np.ndarray, np.ndarray, list[tuple[str, float]]]:
    """Run all five hand-eye methods; return the best by position std.

    Returns:
        (T_best, method_name, errors_mm, errors_deg, method_table).
    """
    best: tuple[float, str, np.ndarray, np.ndarray, np.ndarray] | None = None
    table: list[tuple[str, float]] = []

    for name, m in _HAND_EYE_METHODS.items():
        try:
            T = _calibrate_eye_to_hand(
                tvec_ee2base,
                rpy_ee2base,
                rvec_marker2camera,
                tvec_marker2camera,
                method=m,
            )
            errors_mm, errors_deg = _compute_closed_loop_errors(
                T,
                tvec_ee2base,
                rpy_ee2base,
                rvec_marker2camera,
                tvec_marker2camera,
            )
            std_mm = float(errors_mm.std())
        except Exception:
            table.append((name, float("nan")))
            continue
        table.append((name, std_mm))
        if best is None or std_mm < best[0]:
            best = (std_mm, name, T, errors_mm, errors_deg)

    if best is None:
        raise RuntimeError("all hand-eye methods failed")
    _, name_best, T_best, errors_mm_best, errors_deg_best = best
    return T_best, name_best, errors_mm_best, errors_deg_best, table


@dataclass
class CalibrationSamples:
    """Paired calibration observations; delete preserves field alignment."""

    tvec_ee2base: list[np.ndarray] = field(default_factory=list)
    rpy_ee2base: list[np.ndarray] = field(default_factory=list)
    rvec_marker2cam: list[np.ndarray] = field(default_factory=list)
    tvec_marker2cam: list[np.ndarray] = field(default_factory=list)
    _residuals_mm: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.tvec_ee2base)

    def append(
        self, pos_ee: np.ndarray, rpy_ee: np.ndarray, rvec: np.ndarray, tvec: np.ndarray
    ) -> None:
        values = {
            "pos_ee": pos_ee,
            "rpy_ee": rpy_ee,
            "rvec": rvec,
            "tvec": tvec,
        }
        arrays: dict[str, np.ndarray] = {}
        for name, value in values.items():
            array = np.asarray(value, dtype=np.float64)
            if array.shape != (3,) or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be a finite array with shape (3,)")
            arrays[name] = array.copy()
        self.tvec_ee2base.append(arrays["pos_ee"])
        self.rpy_ee2base.append(arrays["rpy_ee"])
        self.rvec_marker2cam.append(arrays["rvec"])
        self.tvec_marker2cam.append(arrays["tvec"])
        self._residuals_mm = None

    def pop_last(self) -> bool:
        if not self.tvec_ee2base:
            return False
        for lst in (
            self.tvec_ee2base,
            self.rpy_ee2base,
            self.rvec_marker2cam,
            self.tvec_marker2cam,
        ):
            lst.pop()
        self._residuals_mm = None
        return True

    def set_residuals(self, residuals_mm: np.ndarray) -> None:
        residuals = np.asarray(residuals_mm, dtype=np.float64)
        if residuals.shape != (len(self),) or not np.all(np.isfinite(residuals)):
            raise ValueError(
                f"residuals_mm must be finite shape ({len(self)},), got {residuals.shape}"
            )
        self._residuals_mm = residuals.copy()

    def pop_worst(self) -> tuple[int, float] | None:
        if self._residuals_mm is None or len(self) == 0:
            return None
        idx = int(np.argmax(self._residuals_mm))
        val = float(self._residuals_mm[idx])
        for lst in (
            self.tvec_ee2base,
            self.rpy_ee2base,
            self.rvec_marker2cam,
            self.tvec_marker2cam,
        ):
            lst.pop(idx)
        self._residuals_mm = None
        return idx, val

    def solver_inputs(
        self,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        return (
            self.tvec_ee2base,
            self.rpy_ee2base,
            self.rvec_marker2cam,
            self.tvec_marker2cam,
        )


def save_camera_calibration(
    T_world_camera: np.ndarray, serial: str, json_path: Path
) -> None:
    """Write calibration result to cameras.json, preserving other entries."""
    T_world_camera = np.asarray(T_world_camera, dtype=np.float64)
    if T_world_camera.shape != (4, 4) or not np.all(np.isfinite(T_world_camera)):
        raise ValueError("T_world_camera must be a finite 4x4 transform")
    if not np.allclose(T_world_camera[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError("T_world_camera must have homogeneous final row [0, 0, 0, 1]")
    if not serial.strip():
        raise ValueError("camera serial must be non-empty")
    rot = Rotation.from_matrix(T_world_camera[:3, :3])
    quat_wxyz = rot.as_quat()[[3, 0, 1, 2]]  # scipy xyzw → wxyz
    pos = T_world_camera[:3, 3]

    entry = {
        "serial": serial,
        "type": "eye_to_hand",
        "pose": {
            "position": [
                round(float(pos[0]), 6),
                round(float(pos[1]), 6),
                round(float(pos[2]), 6),
            ],
            "orientation": [
                round(float(quat_wxyz[0]), 6),
                round(float(quat_wxyz[1]), 6),
                round(float(quat_wxyz[2]), 6),
                round(float(quat_wxyz[3]), 6),
            ],
        },
    }

    if json_path.exists():
        with open(json_path) as f:
            existing = json.load(f)
    else:
        existing = {}

    cam_name = "camera_0"
    for name, cam_data in existing.items():
        if cam_data.get("serial") == serial:
            cam_name = name
            break
    else:
        idx = 0
        while f"camera_{idx}" in existing:
            idx += 1
        cam_name = f"camera_{idx}"

    if json_path.exists():
        backup = json_path.with_suffix(
            f".json.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        # Copy before the atomic write so cameras.json is never briefly absent.
        shutil.copy2(json_path, backup)
        print(f"  backed up previous config → {backup.name}")

    existing[cam_name] = entry
    atomic_json_dump(existing, json_path, ensure_ascii=False)
    print(f"  calibration written → {json_path} (camera: {cam_name})")
