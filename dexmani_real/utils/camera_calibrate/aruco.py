"""ArUco 检测与静止机械臂上的多帧采样。"""

import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from dexmani_real.geometry.camera import CameraIntrinsics
from dexmani_real.utils.timing import sample_is_fresh

ARUCO_DICT = cv2.aruco.DICT_7X7_50
ARUCO_DICT_NAME = "7x7_50"
MARKER_ID = 1
MARKER_SIZE_M = 0.0982
CAPTURE_FRAMES = 5
MIN_DETECTED_FRAMES = 4
MAX_REPROJECTION_RMS_PX = 1.0
MAX_POSITION_SPREAD_M = 0.005
MAX_ROTATION_SPREAD_DEG = 3.0


class ArucoSampler:
    def __init__(self, intrinsics: CameraIntrinsics):
        model = intrinsics.distortion_model.removeprefix("distortion.")
        if model not in {"none", "brown_conrady"}:
            raise ValueError(f"calibration PnP does not support distortion model {intrinsics.distortion_model!r}")
        self.intrinsics = intrinsics.matrix()
        self.distortion = np.zeros(5) if model == "none" else np.asarray(intrinsics.distortion_coeffs)
        half = MARKER_SIZE_M / 2
        # IPPE_SQUARE 要求左上、右上、右下、左下的标记局部坐标。
        self.corners = np.array(
            [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]],
            dtype=np.float32,
        )
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT), parameters)

    def detect(self, rgb):
        # RGB 的类型、形状和缓冲所有权由 RealSenseCamera.read 保证。
        corners, ids, _ = self.detector.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        if ids is not None:
            for corner, marker_id in zip(corners, ids.ravel()):
                if marker_id == MARKER_ID:
                    corner = corner.reshape(4, 2)
                    return corner, self.estimate_pose(corner)
        return np.full((4, 2), np.nan), None

    def estimate_pose(self, corners):
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

    def stationary_arm(self, read_arm, arm_config, reference=None):
        arm = read_arm()
        if np.max(np.abs(arm["qvel"])) > arm_config.standstill_velocity_tolerance_rad_s:
            raise ValueError("arm is moving during calibration capture")
        if (
            reference is not None
            and np.max(np.abs(arm["qpos"] - reference)) > arm_config.standstill_position_tolerance_rad
        ):
            raise ValueError("arm drifted during calibration capture")
        return arm

    def capture(self, read_camera, read_arm, *, camera_config, arm_config):
        arm_before = self.stationary_arm(read_arm, arm_config)
        last_stamp, last_color_frame = time.monotonic_ns(), -1
        frames = []
        for _ in range(CAPTURE_FRAMES):
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                arm = self.stationary_arm(read_arm, arm_config, arm_before["qpos"])
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
                raise TimeoutError("fresh calibration image unavailable")
            last_stamp, last_color_frame = frame.timestamp_ns, frame.color_frame_number
            corners, pose = self.detect(frame.rgb)
            frames.append({
                "rgb": frame.rgb, "camera_timestamp_ns": frame.timestamp_ns,
                "color_frame_number": frame.color_frame_number, "marker_corners_xy": corners,
                "marker_rvec": pose[0] if pose is not None else np.full(3, np.nan),
                "marker_tvec": pose[1] if pose is not None else np.full(3, np.nan),
                "marker_detected": pose is not None,
                "arm_qpos": arm["qpos"], "arm_qvel": arm["qvel"], "arm_timestamp_ns": arm["timestamp_ns"],
            })
        arm_after = self.stationary_arm(read_arm, arm_config, arm_before["qpos"])
        sample = {name: np.stack([frame[name] for frame in frames]) for name in frames[0]}
        valid = sample["marker_detected"]
        if valid.sum() < MIN_DETECTED_FRAMES:
            raise ValueError(f"marker has only {valid.sum()}/{CAPTURE_FRAMES} valid PnP frames; need {MIN_DETECTED_FRAMES}")
        rotations = Rotation.from_rotvec(sample["marker_rvec"][valid])
        translations = sample["marker_tvec"][valid]
        first, second = np.triu_indices(len(translations), k=1)
        position_spread = np.max(np.linalg.norm(translations[first] - translations[second], axis=1))
        rotation_spread = np.rad2deg((rotations[first].inv() * rotations[second]).magnitude()).max()
        # 先检查一致性再平均，避免平面双解或异常帧污染整组姿态。
        if position_spread > MAX_POSITION_SPREAD_M or rotation_spread > MAX_ROTATION_SPREAD_DEG:
            raise ValueError(f"inconsistent marker poses: {position_spread * 1000:.3f} mm / {rotation_spread:.3f} deg")
        transform = np.eye(4)
        transform[:3, :3] = rotations.mean().as_matrix()
        transform[:3, 3] = np.median(translations, axis=0)
        sample["T_camera_from_marker"] = transform
        for stage, arm in (("before", arm_before), ("after", arm_after)):
            sample.update({f"arm_{stage}_{name}": value for name, value in arm.items()})
        return sample
