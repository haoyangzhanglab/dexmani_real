"""ArUco 检测与静止机械臂上的多帧采样。"""

import time
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from dexmani_real.utils.timing import sample_is_fresh

ARUCO_DICT = cv2.aruco.DICT_7X7_50
ARUCO_DICT_NAME = "7x7_50"
MARKER_ID = 1
MARKER_SIZE_M = 0.0982
CAPTURE_FRAMES = 5


@dataclass(frozen=True)
class ArucoDetection:
    corners: Sequence[np.ndarray]
    ids: np.ndarray | None
    pose: tuple[np.ndarray, np.ndarray] | None


class ArucoSampler:
    def __init__(self, intrinsics, distortion):
        self.intrinsics = intrinsics
        self.distortion = distortion
        half = MARKER_SIZE_M / 2
        # IPPE_SQUARE 要求左上、右上、右下、左下的标记局部坐标。
        self.corners = np.array(
            [
                [-half, half, 0],
                [half, half, 0],
                [half, -half, 0],
                [-half, -half, 0],
            ],
            dtype=np.float32,
        )
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(ARUCO_DICT),
            parameters,
        )

    def detect(self, rgb: np.ndarray) -> ArucoDetection:
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 2:
            raise ValueError("calibration RGB must be uint8 (H, W, 3)")
        corners, ids, _ = self.detector.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        if ids is None:
            return ArucoDetection(corners, ids, None)
        for corner, marker_id in zip(corners, ids.ravel()):
            if marker_id != MARKER_ID:
                continue
            ok, rvec, tvec = cv2.solvePnP(
                self.corners,
                corner,
                self.intrinsics,
                self.distortion,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not ok or not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
                return ArucoDetection(corners, ids, None)
            return ArucoDetection(corners, ids, (rvec.ravel(), tvec.ravel()))
        return ArucoDetection(corners, ids, None)

    def stationary_arm(self, read_arm, arm_config):
        arm = read_arm()
        if np.max(np.abs(arm["qvel"])) > arm_config.homing.velocity_convergence_rad_s:
            raise ValueError("arm is moving during calibration capture")
        return arm

    def capture(self, read_camera, read_arm, *, camera_config, arm_config):
        arm_before = self.stationary_arm(read_arm, arm_config)
        last_stamp, last_color_frame = time.monotonic_ns(), -1
        rotations, translations, frames = [], [], []
        for _ in range(CAPTURE_FRAMES):
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                arm = read_arm()
                if np.max(np.abs(arm["qvel"])) > arm_config.homing.velocity_convergence_rad_s:
                    raise ValueError("arm is moving during calibration capture")
                frame = read_camera()
                if (
                    frame is not None
                    and frame.timestamp_ns > last_stamp
                    and frame.color_frame_number > last_color_frame
                    and sample_is_fresh(frame.timestamp_ns, camera_config.max_frame_age_s)
                ):
                    break
                time.sleep(0.01)
            else:
                raise TimeoutError("fresh calibration image unavailable")
            last_stamp = frame.timestamp_ns
            last_color_frame = frame.color_frame_number
            if not isinstance(frame.rgb, np.ndarray) or frame.rgb.shape != (
                camera_config.height, camera_config.width, 3,
            ):
                raise ValueError("calibration RGB shape differs from configured camera")
            detection = self.detect(frame.rgb)
            pose = detection.pose
            target_corners = np.full((4, 2), np.nan)
            if detection.ids is not None:
                indices = np.flatnonzero(detection.ids.ravel() == MARKER_ID)
                if len(indices):
                    target_corners = np.asarray(detection.corners[indices[0]]).reshape(4, 2)
            frames.append(dict(
                rgb=frame.rgb, timestamp_ns=frame.timestamp_ns,
                color_frame_number=frame.color_frame_number, corners=target_corners,
                rvec=pose[0] if pose is not None else np.full(3, np.nan),
                tvec=pose[1] if pose is not None else np.full(3, np.nan),
                detected=pose is not None, arm=arm,
            ))
            if pose is not None:
                rotations.append(pose[0])
                translations.append(pose[1])

        arm_after = self.stationary_arm(read_arm, arm_config)
        if (
            np.max(np.abs(arm_after["qpos"] - arm_before["qpos"]))
            > arm_config.homing.convergence_rad
        ):
            raise ValueError("arm drifted during calibration capture")
        if len(rotations) < CAPTURE_FRAMES // 2:
            raise ValueError("marker not detected in enough capture frames")
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_rotvec(rotations).mean().as_matrix()
        transform[:3, 3] = np.median(translations, axis=0)
        return dict(
            arm_before=arm_before, arm_after=arm_after,
            frames=frames, T_camera_from_marker=transform,
        )
