"""ArUco eye-to-hand calibration math and result persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from dexmani_real.planning.kinematics.pose import rot6d_to_quat_wxyz
from dexmani_real.utils.atomic_io import atomic_json_dump
from dexmani_real.utils.geometry import validate_rigid_transform

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
    min_samples: int = 10
    min_relative_rotation_deg: float = 5.0
    min_axis_separation_deg: float = 15.0
    max_consistency_rms_mm: float = 5.0
    max_consistency_rot_rms_deg: float = 3.0
    status_interval_frames: int = 50

    def __post_init__(self) -> None:
        if not isinstance(self.min_samples, int) or self.min_samples < 3:
            raise ValueError("min_samples must be at least 3")
        positive_fields = (
            "min_relative_rotation_deg",
            "min_axis_separation_deg",
            "max_consistency_rms_mm",
            "max_consistency_rot_rms_deg",
        )
        for field_name in positive_fields:
            value = float(getattr(self, field_name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive")
        if self.min_relative_rotation_deg > 180 or self.min_axis_separation_deg > 90:
            raise ValueError("excitation thresholds exceed rotation/axis angle ranges")
        if not isinstance(self.status_interval_frames, int) or self.status_interval_frames < 1:
            raise ValueError("status_interval_frames must be at least 1")


def _marker_poses(corners, ids, marker_points, intrinsics, distortion, *, target_id):
    if ids is None:
        return
    for corner, marker_id in zip(corners, ids.flatten()):
        if target_id is None or marker_id == target_id:
            yield cv2.solvePnP(
                marker_points, corner, intrinsics, distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE
            )


def detect_aruco_pose(
    color_image: np.ndarray,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
    *,
    detector: cv2.aruco.ArucoDetector,
    marker_corners: np.ndarray,
    target_id: int | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Detect ArUco marker and return (rvec, tvec) in camera frame, or None."""
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    for ok, rv, tv in _marker_poses(
        corners, ids, marker_corners, intrinsics, distortion,
        target_id=target_id,
    ):
        if not ok or rv is None or tv is None:
            return None
        rv, tv = np.asarray(rv, dtype=np.float64), np.asarray(tv, dtype=np.float64)
        if rv.size != 3 or tv.size != 3 or not np.isfinite(rv).all() or not np.isfinite(tv).all():
            return None
        return rv.reshape(3), tv.reshape(3)

    return None


def marker_corners_3d(marker_size_m: float) -> np.ndarray:
    """Marker-local corners in meters, ordered for SOLVEPNP_IPPE_SQUARE."""
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
) -> np.ndarray:
    """Annotate BGR preview with ArUco boxes, axes and detection status."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    detected = False
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        for ok, rv, tv in _marker_poses(
            corners, ids, marker_corners, intrinsics, distortion, target_id=target_id,
        ):
            if ok:
                cv2.drawFrameAxes(image, intrinsics, distortion, rv, tv, marker_size_m * 0.5)
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
        "ENTER=calibrate" if n_samples >= min_samples else f"SPACE=capture (need >={min_samples})"
    )
    cv2.putText(image, hint, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return image


def _calibrate_eye_to_hand(
    tvec_ee2base: tuple[np.ndarray, ...],
    rpy_ee2base: tuple[np.ndarray, ...],
    rvec_marker2camera: tuple[np.ndarray, ...],
    tvec_marker2camera: tuple[np.ndarray, ...],
    *,
    method: int,
) -> np.ndarray:
    """Invert robot poses for eye-to-hand cv2.calibrateHandEye, returning T_base_camera (4,4)."""
    rmat_ee2base = [
        Rotation.from_euler("xyz", rpy, degrees=False).as_matrix() for rpy in rpy_ee2base
    ]
    rmat_base2ee = [rmat.T for rmat in rmat_ee2base]
    tvec_base2ee = [(-rmat.T @ tv).reshape(3, 1) for rmat, tv in zip(rmat_ee2base, tvec_ee2base)]
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
    tvec_ee2base: tuple[np.ndarray, ...],
    rpy_ee2base: tuple[np.ndarray, ...],
    rvec_marker2camera: tuple[np.ndarray, ...],
    tvec_marker2camera: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (N,) marker-to-EEF consistency residuals in mm and degrees; the marker is rigidly mounted."""
    positions: list[np.ndarray] = []
    rotations: list[np.ndarray] = []

    for tvec_ee, rpy_ee, rvec_m2c, tvec_m2c in zip(
        tvec_ee2base,
        rpy_ee2base,
        rvec_marker2camera,
        tvec_marker2camera,
    ):
        T_base_ee = np.eye(4)
        T_base_ee[:3, :3] = Rotation.from_euler("xyz", rpy_ee, degrees=False).as_matrix()
        T_base_ee[:3, 3] = tvec_ee

        T_camera_marker = np.eye(4)
        T_camera_marker[:3, :3] = cv2.Rodrigues(rvec_m2c)[0]
        T_camera_marker[:3, 3] = tvec_m2c

        T_ee_marker = np.linalg.inv(T_base_ee) @ T_base_camera @ T_camera_marker
        positions.append(T_ee_marker[:3, 3].copy())
        rotations.append(T_ee_marker[:3, :3])

    positions_arr = np.array(positions)
    mean_pos = positions_arr.mean(axis=0)
    residuals_mm = np.linalg.norm(positions_arr - mean_pos, axis=1) * 1000.0

    rots = Rotation.from_matrix(np.array(rotations))
    mean_rot = rots.mean()
    residuals_deg = np.degrees((mean_rot.inv() * rots).magnitude())
    return residuals_mm, residuals_deg


def _check_hand_eye_excitation(
    rpy_ee2base, *, min_relative_rotation_deg, min_axis_separation_deg
):
    """Require two-axis excitation of admitted samples; do not impose three-axis rank."""
    rotations = Rotation.from_euler("xyz", rpy_ee2base).as_matrix()
    axes, angles = [], []
    for i in range(len(rotations)):
        for j in range(i + 1, len(rotations)):
            vector = Rotation.from_matrix(rotations[j] @ rotations[i].T).as_rotvec()
            angle = np.linalg.norm(vector)
            if np.degrees(angle) >= min_relative_rotation_deg:
                axes.append(vector / angle)
                angles.append(float(np.degrees(angle)))
    # Any qualifying axis pair is sufficient; opposite axes describe the same direction.
    for axis, other in combinations(axes, 2):
        separation = float(np.degrees(np.arccos(np.clip(abs(axis @ other), 0, 1))))
        if separation >= min_axis_separation_deg:
            return dict(
                sample_count=len(rotations),
                valid_relative_motion_count=len(axes),
                relative_rotation_min_deg=min(angles),
                relative_rotation_max_deg=max(angles),
                axis_separation_witness_deg=separation,
                min_relative_rotation_deg=min_relative_rotation_deg,
                min_axis_separation_deg=min_axis_separation_deg,
            )
    raise ValueError("insufficient hand-eye excitation; add rotation about another direction")


@dataclass(frozen=True)
class HandEyeResult:
    T_base_camera: np.ndarray
    method: str
    position_errors_mm: np.ndarray
    rotation_errors_deg: np.ndarray
    method_scores_mm: list[tuple[str, float]]
    position_rms_mm: float
    rotation_rms_deg: float
    excitation: dict[str, float | int]


def calibrate_and_select(samples: CalibrationSamples, config: CalibrationConfig) -> HandEyeResult:
    """Select the lowest-position-RMS candidate passing both limits."""
    if len(samples) < config.min_samples:
        raise ValueError(f"need at least {config.min_samples} samples, have {len(samples)}")
    tvec_ee2base, rpy_ee2base, rvec_marker2camera, tvec_marker2camera = zip(*samples.pairs)
    excitation = _check_hand_eye_excitation(
        rpy_ee2base,
        min_relative_rotation_deg=config.min_relative_rotation_deg,
        min_axis_separation_deg=config.min_axis_separation_deg,
    )
    best = None
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
            # Check OpenCV's candidate once before composing it with the sample rotations.
            T = validate_rigid_transform(T, label="hand-eye candidate")
            errors_mm, errors_deg = _compute_closed_loop_errors(
                T,
                tvec_ee2base,
                rpy_ee2base,
                rvec_marker2camera,
                tvec_marker2camera,
            )
            rms_mm = float(np.sqrt(np.mean(errors_mm**2)))
            rms_deg = float(np.sqrt(np.mean(errors_deg**2)))
            if not np.isfinite(rms_mm) or not np.isfinite(rms_deg):
                raise ValueError("nonfinite hand-eye quality score")
        except (cv2.error, ValueError):
            table.append((name, float("nan")))
            continue
        table.append((name, rms_mm))
        if rms_mm > config.max_consistency_rms_mm or rms_deg > config.max_consistency_rot_rms_deg:
            continue
        if best is None or (rms_mm, rms_deg) < best[:2]:
            best = (rms_mm, rms_deg, name, T, errors_mm, errors_deg)

    if best is None:
        raise ValueError("no hand-eye candidate passed both quality limits")
    rms_mm, rms_deg, name_best, T_best, errors_mm_best, errors_deg_best = best
    return HandEyeResult(
        T_best, name_best, errors_mm_best, errors_deg_best, table, rms_mm, rms_deg, excitation
    )


@dataclass
class CalibrationSamples:
    """Each row owns paired EEF position/RPY and marker rotation/translation vectors."""

    pairs: list[np.ndarray] = field(default_factory=list)
    residuals_mm: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.pairs)

    def append(self, pos_ee, rpy_ee, rvec, tvec) -> None:
        pair = np.array((pos_ee, rpy_ee, rvec, tvec), dtype=np.float64)
        if pair.shape != (4, 3) or not np.isfinite(pair).all():
            raise ValueError("calibration sample must contain four finite 3-vectors")
        self.pairs.append(pair)
        self.residuals_mm = None

    def pop_last(self) -> bool:
        if not self.pairs:
            return False
        self.pairs.pop()
        self.residuals_mm = None
        return True

    def pop_worst(self) -> tuple[int, float] | None:
        if self.residuals_mm is None:
            return None
        index = int(np.argmax(self.residuals_mm))
        residual = float(self.residuals_mm[index])
        self.pairs.pop(index)
        self.residuals_mm = None
        return index, residual


def save_camera_calibration(
    T_world_camera: np.ndarray,
    serial: str,
    json_path: Path,
    *,
    calibration_capture: dict[str, object] | None = None,
    cancelled=None,
) -> None:
    """Write calibration result to cameras.json, preserving other entries."""
    T_world_camera = validate_rigid_transform(T_world_camera, label="T_world_camera")
    if not serial.strip():
        raise ValueError("camera serial must be non-empty")
    rot = Rotation.from_matrix(T_world_camera[:3, :3])
    quat_wxyz = rot.as_quat()[[3, 0, 1, 2]]  # scipy xyzw → wxyz
    pos = T_world_camera[:3, 3]

    entry = {
        "serial": serial,
        "type": "eye_to_hand",
        "pose": {
            "position": pos.tolist(),
            "orientation": quat_wxyz.tolist(),
        },
    }
    if calibration_capture is not None:
        entry["calibration_capture"] = calibration_capture

    if json_path.exists():
        with open(json_path) as f:
            existing = json.load(f)
    else:
        existing = {}

    for name, cam_data in existing.items():
        if cam_data.get("serial") == serial:
            cam_name = name
            break
    else:
        idx = 0
        while f"camera_{idx}" in existing:
            idx += 1
        cam_name = f"camera_{idx}"

    existing[cam_name] = entry
    try:
        json.dumps(existing, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("camera calibration payload must contain finite JSON values") from exc
    atomic_json_dump(existing, json_path, ensure_ascii=False, cancelled=cancelled)
    print(f"  calibration written → {json_path} (camera: {cam_name})")
