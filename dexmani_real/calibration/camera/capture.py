"""静止机械臂上的 RGB 标记采样；运动权限与样本集合由会话持有。"""

from __future__ import annotations

import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from dexmani_real.calibration.camera.aruco import (
    ARUCO_DICT,
    detect_aruco_pose,
    marker_corners_3d,
)
from dexmani_real.geometry.pose import rot6d_to_quat_wxyz
from dexmani_real.ipc.sensors import SensorChannels
from dexmani_real.runtime.observation import read_camera_frame
from dexmani_real.utils.timing import sample_is_fresh


def eef_rpy_from_rot6d(rot6d: np.ndarray) -> np.ndarray:
    """Convert an arm EEF rot6d orientation to xyz RPY in radians."""
    quat_wxyz = rot6d_to_quat_wxyz(rot6d)
    rpy_rad = Rotation.from_quat(np.roll(quat_wxyz, -1)).as_euler("xyz", degrees=False)
    return np.asarray(rpy_rad, dtype=np.float64)


def detect_aruco_stable(
    channels: SensorChannels,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
    *,
    detector: cv2.aruco.ArucoDetector,
    marker_corners: np.ndarray,
    target_id: int | None,
    max_frame_age_s: float,
    n_frames: int,
    not_before_ns: int,
    check_capture,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Aggregate post-barrier RGB poses using an SO(3) mean and median translation."""
    rvecs_all: list[np.ndarray] = []
    tvecs_all: list[np.ndarray] = []
    last_stamp = not_before_ns
    last_sequence = 0
    last_color_frame = -1
    for _ in range(n_frames):
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            check_capture()
            frame = read_camera_frame(channels, payload="rgb")
            if (
                frame
                and frame["timestamp_ns"] > last_stamp
                and frame["ring_sequence"] > last_sequence
                and frame["color_frame_number"] > last_color_frame
                and sample_is_fresh(frame["timestamp_ns"], max_frame_age_s)
            ):
                break
            time.sleep(0.01)
        else:
            raise TimeoutError("fresh calibration image unavailable")
        last_stamp = frame["timestamp_ns"]
        last_sequence = frame["ring_sequence"]
        last_color_frame = frame["color_frame_number"]
        check_capture()
        image = cv2.cvtColor(frame["rgb"], cv2.COLOR_RGB2BGR)
        result = detect_aruco_pose(
            image,
            intrinsics,
            distortion,
            detector=detector,
            marker_corners=marker_corners,
            target_id=target_id,
        )
        check_capture()
        if result is not None:
            rvecs_all.append(result[0])
            tvecs_all.append(result[1])

    if len(rvecs_all) < max(1, n_frames // 2):
        return None
    return Rotation.from_rotvec(np.asarray(rvecs_all)).mean().as_rotvec(), np.median(
        tvecs_all, axis=0
    )


class CalibrationSampler:
    """持有采样检测器和相机几何；反馈回调持续检查取消及设备故障。"""

    def __init__(self, aruco_config, intrinsics, distortion):
        self.aruco_config = aruco_config
        self.intrinsics = intrinsics
        self.distortion = distortion
        self.marker_corners = marker_corners_3d(aruco_config.marker_size_m)
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(ARUCO_DICT), parameters,
        )

    def read_stationary_arm(self, read_arm, arm_config, after_ns):
        deadline_ns = time.monotonic_ns() + int(arm_config.feedback_max_age_s * 1e9)
        while time.monotonic_ns() < deadline_ns:
            arm = read_arm()
            if arm["timestamp_ns"] <= after_ns:
                time.sleep(0.01)
                continue
            speed_rad_s = float(np.max(np.abs(arm["qvel"])))
            bound_rad_s = arm_config.homing.velocity_convergence_rad_s
            if speed_rad_s > bound_rad_s:
                return None, (
                    "arm velocity exceeds stationary bound "
                    f"({speed_rad_s:.4f}rad/s > {bound_rad_s:.4f}rad/s)"
                )
            return arm, ""
        raise TimeoutError("new arm feedback unavailable after calibration capture barrier")

    def capture(self, channels, read_arm, *, camera_config, arm_config):
        """Return stationary joints and marker pose, or None for a rejected observation."""
        arm_before, feedback_issue = self.read_stationary_arm(
            read_arm, arm_config, time.monotonic_ns(),
        )
        if arm_before is None:
            print(f"FAILED — before capture: {feedback_issue}, skipped")
            return None
        aruco_pose = detect_aruco_stable(
            channels,
            self.intrinsics,
            self.distortion,
            detector=self.detector,
            marker_corners=self.marker_corners,
            target_id=self.aruco_config.target_id,
            max_frame_age_s=camera_config.max_frame_age_s,
            n_frames=self.aruco_config.capture_frames,
            not_before_ns=time.monotonic_ns(),
            check_capture=read_arm,
        )
        arm_after, feedback_issue = self.read_stationary_arm(
            read_arm, arm_config, time.monotonic_ns(),
        )
        if arm_after is None:
            print(f"FAILED — after capture: {feedback_issue}, skipped")
            return None
        arm_before_qpos = np.asarray(arm_before["qpos"], dtype=np.float64)
        arm_after_qpos = np.asarray(arm_after["qpos"], dtype=np.float64)
        drift_rad = float(np.max(np.abs(arm_after_qpos - arm_before_qpos)))
        convergence_rad = float(arm_config.homing.convergence_rad)
        if drift_rad > convergence_rad:
            print(
                "FAILED — arm moved during capture "
                f"({drift_rad:.4f}rad > {convergence_rad:.4f}rad), skipped"
            )
            return None
        if aruco_pose is None:
            print("FAILED — marker not detected, skipped")
            return None
        marker_rvec, marker_tvec = aruco_pose
        return arm_after_qpos, marker_rvec, marker_tvec
