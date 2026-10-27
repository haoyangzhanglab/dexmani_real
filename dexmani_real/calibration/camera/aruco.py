"""ArUco 检测与静止机械臂上的多帧采样。"""

import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from dexmani_real.ipc.sensors import read_camera_frame
from dexmani_real.utils.timing import sample_is_fresh

ARUCO_DICT = cv2.aruco.DICT_7X7_50
ARUCO_DICT_NAME = "7x7_50"
MARKER_ID = 1
MARKER_SIZE_M = 0.0982
CAPTURE_FRAMES = 5


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

    def detect(self, image, *, draw=False):
        corners, ids, _ = self.detector.detectMarkers(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        if ids is None:
            return None
        if draw:
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
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
                return None
            if draw:
                cv2.drawFrameAxes(
                    image, self.intrinsics, self.distortion, rvec, tvec, MARKER_SIZE_M / 2
                )
            return rvec.ravel(), tvec.ravel()
        return None

    def stationary_arm(self, read_arm, arm_config, after_ns):
        deadline = time.monotonic_ns() + int(arm_config.feedback_max_age_s * 1e9)
        while time.monotonic_ns() < deadline:
            arm = read_arm()
            if arm["timestamp_ns"] <= after_ns:
                time.sleep(0.01)
                continue
            if np.max(np.abs(arm["qvel"])) > arm_config.homing.velocity_convergence_rad_s:
                raise ValueError("arm is moving during calibration capture")
            return arm
        raise TimeoutError("fresh arm feedback unavailable after capture barrier")

    def capture(self, channels, read_arm, *, camera_config, arm_config):
        arm_before = self.stationary_arm(read_arm, arm_config, time.monotonic_ns())
        last_stamp, last_color_frame = time.monotonic_ns(), -1
        rotations, translations = [], []
        for _ in range(CAPTURE_FRAMES):
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                arm = read_arm()
                if np.max(np.abs(arm["qvel"])) > arm_config.homing.velocity_convergence_rad_s:
                    raise ValueError("arm is moving during calibration capture")
                frame = read_camera_frame(channels, fields=("rgb",))
                if (
                    frame
                    and frame["timestamp_ns"] > last_stamp
                    and frame["color_frame_number"] > last_color_frame
                    and sample_is_fresh(frame["timestamp_ns"], camera_config.max_frame_age_s)
                ):
                    break
                time.sleep(0.01)
            else:
                raise TimeoutError("fresh calibration image unavailable")
            last_stamp = frame["timestamp_ns"]
            last_color_frame = frame["color_frame_number"]
            pose = self.detect(cv2.cvtColor(frame["rgb"], cv2.COLOR_RGB2BGR))
            if pose is not None:
                rotations.append(pose[0])
                translations.append(pose[1])

        arm_after = self.stationary_arm(read_arm, arm_config, time.monotonic_ns())
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
        return arm_after["qpos"], transform
