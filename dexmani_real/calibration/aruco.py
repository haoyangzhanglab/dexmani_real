"""ArUco marker detection, burst aggregation, and preview overlays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ARUCO_DICTIONARY_NAME = "7x7_50"


@dataclass(frozen=True)
class ArucoConfig:
    """Marker geometry and detection-quality thresholds."""

    dictionary_id: int = cv2.aruco.DICT_7X7_50
    dictionary_name: str = ARUCO_DICTIONARY_NAME
    marker_size_m: float = 0.1228
    target_id: int | None = 1
    capture_frames: int = 5
    frame_timeout_ms: int = 500
    min_corner_area_px2: float = 100.0
    max_reprojection_error_px: float = 2.0
    max_translation_spread_m: float = 0.005
    max_rotation_spread_deg: float = 2.0

    def __post_init__(self) -> None:
        positive = (
            self.marker_size_m,
            self.min_corner_area_px2,
            self.max_reprojection_error_px,
            self.max_translation_spread_m,
            self.max_rotation_spread_deg,
        )
        if not self.dictionary_name or any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("ArUco geometry and quality thresholds must be finite and positive")
        if self.capture_frames <= 0 or self.frame_timeout_ms <= 0:
            raise ValueError("ArUco capture_frames and frame_timeout_ms must be positive")
        if self.target_id is not None and self.target_id < 0:
            raise ValueError("ArUco target_id must be non-negative or None")

    @property
    def minimum_burst_detections(self) -> int:
        return max(1, (self.capture_frames + 1) // 2)


DEFAULT_ARUCO_CONFIG = ArucoConfig()


def marker_corners_3d(marker_size_m: float) -> np.ndarray:
    """Return IPPE-square marker corners in marker coordinates."""
    if not np.isfinite(marker_size_m) or marker_size_m <= 0.0:
        raise ValueError("marker_size_m must be finite and positive")
    half_size_m = marker_size_m / 2.0
    return np.array(
        [
            [-half_size_m, half_size_m, 0.0],
            [half_size_m, half_size_m, 0.0],
            [half_size_m, -half_size_m, 0.0],
            [-half_size_m, -half_size_m, 0.0],
        ],
        dtype=np.float32,
    )


def create_detector(config: ArucoConfig = DEFAULT_ARUCO_CONFIG, *, refine_corners: bool) -> Any:
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = (
        cv2.aruco.CORNER_REFINE_SUBPIX if refine_corners else cv2.aruco.CORNER_REFINE_NONE
    )
    dictionary = cv2.aruco.getPredefinedDictionary(config.dictionary_id)
    return cv2.aruco.ArucoDetector(dictionary, parameters)


def _camera_parameters(intrinsics: object, distortion: object) -> tuple[np.ndarray, np.ndarray]:
    camera_matrix = np.asarray(intrinsics, dtype=np.float64)
    distortion_coefficients = np.asarray(distortion, dtype=np.float64)
    if camera_matrix.shape != (3, 3) or not np.all(np.isfinite(camera_matrix)):
        raise ValueError("intrinsics must be a finite (3, 3) matrix")
    if distortion_coefficients.ndim not in (1, 2) or not np.all(np.isfinite(distortion_coefficients)):
        raise ValueError("distortion must be a finite vector")
    return camera_matrix, distortion_coefficients


def detect_aruco_pose(
    color_image: np.ndarray,
    intrinsics: object,
    distortion: object,
    config: ArucoConfig = DEFAULT_ARUCO_CONFIG,
    *,
    detector: Any | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return a quality-filtered marker-to-camera Rodrigues pose."""
    image = np.asarray(color_image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("color_image must have shape (H, W, 3)")
    camera_matrix, distortion_coefficients = _camera_parameters(intrinsics, distortion)
    active_detector = detector or create_detector(config, refine_corners=True)
    corners, ids, _rejected = active_detector.detectMarkers(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    if ids is None or len(ids) == 0:
        return None

    marker_points = marker_corners_3d(config.marker_size_m)
    for corner, marker_id in zip(corners, ids.flatten()):
        if config.target_id is not None and int(marker_id) != config.target_id:
            continue
        image_points = np.asarray(corner, dtype=np.float32).reshape(4, 2)
        if abs(float(cv2.contourArea(image_points))) < config.min_corner_area_px2:
            continue
        ok, raw_rvec, raw_tvec = cv2.solvePnP(
            marker_points,
            image_points,
            camera_matrix,
            distortion_coefficients,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            continue
        rvec = np.asarray(raw_rvec, dtype=np.float64).reshape(3)
        tvec = np.asarray(raw_tvec, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(np.concatenate((rvec, tvec)))) or tvec[2] <= 0.0:
            continue
        projected, _jacobian = cv2.projectPoints(
            marker_points,
            rvec,
            tvec,
            camera_matrix,
            distortion_coefficients,
        )
        projected_points = np.asarray(projected, dtype=np.float64).reshape(4, 2)
        reprojection_error_px = float(
            np.sqrt(np.mean(np.sum(np.square(projected_points - image_points.astype(np.float64)), axis=1)))
        )
        if np.isfinite(reprojection_error_px) and reprojection_error_px <= config.max_reprojection_error_px:
            return rvec, tvec
    return None


def aggregate_pose_burst(
    rvecs: Sequence[np.ndarray],
    tvecs: Sequence[np.ndarray],
    config: ArucoConfig = DEFAULT_ARUCO_CONFIG,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Aggregate a detection burst on SO(3), rejecting sparse or dispersed data."""
    if len(rvecs) != len(tvecs):
        raise ValueError("ArUco burst rotation and translation counts must match")
    if len(rvecs) < config.minimum_burst_detections:
        return None
    rotations_array = np.asarray(rvecs, dtype=np.float64)
    translations = np.asarray(tvecs, dtype=np.float64)
    if rotations_array.shape != (len(rvecs), 3) or translations.shape != (len(tvecs), 3):
        raise ValueError("ArUco burst poses must have shape (N, 3)")
    if not np.all(np.isfinite(rotations_array)) or not np.all(np.isfinite(translations)):
        raise ValueError("ArUco burst poses must be finite")

    rotations = Rotation.from_rotvec(rotations_array)
    mean_rotation = rotations.mean()
    rotation_spread_deg = float(np.rad2deg(np.max((mean_rotation.inv() * rotations).magnitude())))
    median_translation = np.median(translations, axis=0)
    translation_spread_m = float(np.max(np.linalg.norm(translations - median_translation, axis=1)))
    if rotation_spread_deg > config.max_rotation_spread_deg or translation_spread_m > config.max_translation_spread_m:
        return None
    return mean_rotation.as_rotvec(), median_translation


def capture_stable_pose(
    pipeline: Any,
    intrinsics: object,
    distortion: object,
    config: ArucoConfig = DEFAULT_ARUCO_CONFIG,
    *,
    abort_requested: Callable[[], bool] | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Collect and aggregate one short, interruptible color-frame burst."""
    detector = create_detector(config, refine_corners=True)
    rvecs: list[np.ndarray] = []
    tvecs: list[np.ndarray] = []
    for _ in range(config.capture_frames):
        if abort_requested is not None and abort_requested():
            return None
        frames = pipeline.wait_for_frames(timeout_ms=config.frame_timeout_ms)
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        image = np.asanyarray(color_frame.get_data())
        pose = detect_aruco_pose(image, intrinsics, distortion, config, detector=detector)
        if pose is not None:
            rvecs.append(pose[0])
            tvecs.append(pose[1])
    return aggregate_pose_burst(rvecs, tvecs, config)


def draw_overlay(
    image: np.ndarray,
    detector: Any,
    intrinsics: object,
    distortion: object,
    *,
    sample_count: int,
    minimum_samples: int,
    config: ArucoConfig = DEFAULT_ARUCO_CONFIG,
) -> tuple[np.ndarray, bool]:
    """Draw marker axes and calibration status on a BGR preview frame."""
    camera_matrix, distortion_coefficients = _camera_parameters(intrinsics, distortion)
    corners, ids, _rejected = detector.detectMarkers(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    detected = False
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        marker_points = marker_corners_3d(config.marker_size_m)
        for corner, marker_id in zip(corners, ids.flatten()):
            if config.target_id is not None and int(marker_id) != config.target_id:
                continue
            ok, rvec, tvec = cv2.solvePnP(
                marker_points,
                np.asarray(corner, dtype=np.float32),
                camera_matrix,
                distortion_coefficients,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if ok:
                cv2.drawFrameAxes(
                    image,
                    camera_matrix,
                    distortion_coefficients,
                    rvec,
                    tvec,
                    config.marker_size_m * 0.5,
                )
                detected = True

    status_color = (0, 200, 0) if detected else (0, 0, 255)
    cv2.putText(
        image,
        f"MARKER {'VISIBLE' if detected else 'NOT FOUND'}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2,
    )
    hint = "ENTER=calibrate" if sample_count >= minimum_samples else f"SPACE=capture (need >={minimum_samples})"
    cv2.putText(
        image,
        f"samples={sample_count}/{minimum_samples}  {hint}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    return image, detected


__all__ = [
    "ARUCO_DICTIONARY_NAME",
    "ArucoConfig",
    "DEFAULT_ARUCO_CONFIG",
    "aggregate_pose_burst",
    "capture_stable_pose",
    "create_detector",
    "detect_aruco_pose",
    "draw_overlay",
    "marker_corners_3d",
]
