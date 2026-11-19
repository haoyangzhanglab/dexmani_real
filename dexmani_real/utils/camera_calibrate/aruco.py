"""ArUco 检测与静止机械臂上的多帧采样。"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from dexmani_real.geometry.camera import CameraIntrinsics
from dexmani_real.utils.timing import sample_is_fresh

if TYPE_CHECKING:
    from dexmani_real.config.sensor import CameraConfig
    from dexmani_real.config.xarm7 import ArmConfig
    from dexmani_real.sensor.realsense import CameraFrame

ARUCO_DICT = cv2.aruco.DICT_7X7_50
ARUCO_DICT_NAME = "7x7_50"
MARKER_ID = 1
MARKER_SIZE_M = 0.0982
CAPTURE_FRAMES = 5
MIN_DETECTED_FRAMES = 4
MAX_REPROJECTION_RMS_PX = 1.0
MAX_POSITION_SPREAD_M = 0.005
MAX_ROTATION_SPREAD_DEG = 3.0


class SampleRejected(ValueError):
    """观测不满足采样条件，可重新摆位后重试。"""


class ArucoSampler:
    def __init__(self, intrinsics: CameraIntrinsics):
        model = intrinsics.distortion_model.removeprefix("distortion.")
        if model not in {"none", "brown_conrady"}:
            raise ValueError(f"calibration PnP does not support distortion model {intrinsics.distortion_model!r}")
        self.intrinsics = intrinsics.matrix()
        self.distortion = np.zeros(5) if model == "none" else np.asarray(intrinsics.distortion_coeffs)
        half = MARKER_SIZE_M / 2
        # IPPE_SQUARE 要求左上、右上、右下、左下的标记局部坐标。
        self.corners: np.ndarray = np.array(
            [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]],
            dtype=np.float32,
        )
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT), parameters)

    def detect(self, rgb: np.ndarray) -> tuple[np.ndarray | None, tuple[np.ndarray, np.ndarray] | None]:
        # RGB 的类型、形状和缓冲所有权由 RealSenseCamera.read 保证。
        corners, ids, _ = self.detector.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        if ids is not None:
            for corner, marker_id in zip(corners, ids.ravel()):
                if marker_id == MARKER_ID:
                    corner = corner.reshape(4, 2)
                    return corner, self._estimate_pose(corner)
        return None, None

    def _estimate_pose(self, corners: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        ok, rvec, tvec = cv2.solvePnP(
            self.corners, corners, self.intrinsics, self.distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok or not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
            return None
        rotation = cv2.Rodrigues(rvec)[0]
        if np.any((self.corners @ rotation.T + tvec.ravel())[:, 2] <= 0):
            return None
        projected = cv2.projectPoints(self.corners, rvec, tvec, self.intrinsics, self.distortion)[0]
        rms = np.sqrt(np.mean(np.sum((projected.reshape(4, 2) - corners) ** 2, axis=1)))
        if not np.isfinite(rms) or rms > MAX_REPROJECTION_RMS_PX:
            return None
        return rvec.ravel(), tvec.ravel()

    def _stationary_qpos(
        self, read_arm: Callable[[], np.void], arm_config: ArmConfig, reference: np.ndarray | None = None,
    ) -> np.ndarray:
        arm = read_arm()
        if np.max(np.abs(arm["qvel"])) > arm_config.standstill_velocity_tolerance_rad_s:
            raise SampleRejected("arm is moving during calibration capture")
        qpos = arm["qpos"]
        if (
            reference is not None
            and np.max(np.abs(qpos - reference)) > arm_config.standstill_position_tolerance_rad
        ):
            raise SampleRejected("arm drifted during calibration capture")
        return qpos

    def capture(
        self, read_camera: Callable[[], CameraFrame | None], read_arm: Callable[[], np.void], *,
        camera_config: CameraConfig, arm_config: ArmConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        """返回停稳时的关节位置与 T_camera_from_marker。"""
        arm_before = self._stationary_qpos(read_arm, arm_config)
        last_stamp, last_color_frame = time.monotonic_ns(), -1
        poses = []
        for _ in range(CAPTURE_FRAMES):
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                self._stationary_qpos(read_arm, arm_config, arm_before)
                frame = read_camera()
                if (
                    frame is not None
                    and frame.timestamp_ns > last_stamp
                    and frame.color_frame_number > last_color_frame
                    and sample_is_fresh(frame.timestamp_ns, camera_config.max_frame_age_s, time.monotonic_ns())
                ):
                    break
                time.sleep(0.01)
            else:
                raise SampleRejected("fresh calibration image unavailable")
            last_stamp, last_color_frame = frame.timestamp_ns, frame.color_frame_number
            _, pose = self.detect(frame.rgb)
            if pose is not None:
                poses.append(pose)
        arm_after = self._stationary_qpos(read_arm, arm_config, arm_before)
        if len(poses) < MIN_DETECTED_FRAMES:
            raise SampleRejected(
                f"marker has only {len(poses)}/{CAPTURE_FRAMES} valid PnP frames; need {MIN_DETECTED_FRAMES}"
            )
        rotation_vectors, translations = (np.stack(vectors) for vectors in zip(*poses))
        rotations = Rotation.from_rotvec(rotation_vectors)
        first, second = np.triu_indices(len(translations), k=1)
        position_spread: np.floating = np.max(np.linalg.norm(translations[first] - translations[second], axis=1))
        rotation_spread = np.rad2deg((rotations[first].inv() * rotations[second]).magnitude()).max()
        # 先检查一致性再平均，避免平面双解或异常帧污染整组姿态。
        if position_spread > MAX_POSITION_SPREAD_M or rotation_spread > MAX_ROTATION_SPREAD_DEG:
            raise SampleRejected(
                f"inconsistent marker poses: {position_spread * 1000:.3f} mm / {rotation_spread:.3f} deg"
            )
        transform = np.eye(4)
        transform[:3, :3] = rotations.mean().as_matrix()
        transform[:3, 3] = np.median(translations, axis=0)
        return arm_after, transform
