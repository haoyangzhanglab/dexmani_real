"""ArUco detection and camera-frame marker poses."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

ARUCO_DICT = cv2.aruco.DICT_7X7_50
ARUCO_DICT_NAME = "7x7_50"


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


def marker_poses(corners, ids, marker_points, intrinsics, distortion, *, target_id):
    """Yield camera-frame poses or None without reordering matching markers."""
    if ids is None:
        return
    for corner, marker_id in zip(corners, ids.flatten()):
        if target_id is not None and marker_id != target_id:
            continue
        ok, rvec, tvec = cv2.solvePnP(
            marker_points, corner, intrinsics, distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if not ok or rvec is None or tvec is None:
            yield None
            continue
        rvec, tvec = np.asarray(rvec, dtype=np.float64), np.asarray(tvec, dtype=np.float64)
        if (
            rvec.size != 3
            or tvec.size != 3
            or not np.isfinite(rvec).all()
            or not np.isfinite(tvec).all()
        ):
            yield None
        else:
            yield rvec.reshape(3), tvec.reshape(3)


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

    poses = marker_poses(
        corners,
        ids,
        marker_corners,
        intrinsics,
        distortion,
        target_id=target_id,
    )
    return next(poses, None)


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
        for pose in marker_poses(
            corners,
            ids,
            marker_corners,
            intrinsics,
            distortion,
            target_id=target_id,
        ):
            if pose is not None:
                rvec, tvec = pose
                cv2.drawFrameAxes(image, intrinsics, distortion, rvec, tvec, marker_size_m * 0.5)
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
