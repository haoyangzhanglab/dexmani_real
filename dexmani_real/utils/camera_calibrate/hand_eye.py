"""Eye-to-hand 求解与刚性标记的闭环残差。"""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from dexmani_real.geometry.pose import validate_rigid_transform

MIN_SAMPLES = 10
MIN_ROTATION_DEG = 5.0
MIN_AXIS_SEPARATION_DEG = 15.0
MAX_POSITION_RMS_MM = 5.0
MAX_ROTATION_RMS_DEG = 3.0


@dataclass(frozen=True)
class HandEyeResult:
    T_base_from_camera: np.ndarray
    method: str
    position_errors_mm: np.ndarray
    rotation_errors_deg: np.ndarray

    @property
    def position_rms_mm(self):
        return float(np.sqrt(np.mean(self.position_errors_mm**2)))

    @property
    def rotation_rms_deg(self):
        return float(np.sqrt(np.mean(self.rotation_errors_deg**2)))

    @property
    def quality_ok(self) -> bool:
        return (
            self.position_rms_mm <= MAX_POSITION_RMS_MM
            and self.rotation_rms_deg <= MAX_ROTATION_RMS_DEG
        )

    @property
    def worst_sample(self) -> int:
        # 按各自门限归一化，位置与旋转异常均可定位。
        errors = np.maximum(
            self.position_errors_mm / MAX_POSITION_RMS_MM,
            self.rotation_errors_deg / MAX_ROTATION_RMS_DEG,
        )
        return int(np.argmax(errors))


def check_excitation(rotations):
    first, second = np.triu_indices(len(rotations), k=1)
    relative = rotations[second] @ rotations[first].transpose(0, 2, 1)
    vectors = Rotation.from_matrix(relative).as_rotvec()
    angles = np.linalg.norm(vectors, axis=1)
    valid = np.rad2deg(angles) >= MIN_ROTATION_DEG
    axes = vectors[valid] / angles[valid, None]
    threshold = np.cos(np.deg2rad(MIN_AXIS_SEPARATION_DEG))
    # 旋转轴正反等价；任意两条相隔足够角度的轴即可。
    for index, axis in enumerate(axes[:-1]):
        if np.any(np.abs(axes[index + 1 :] @ axis) <= threshold):
            return
    raise ValueError("insufficient hand-eye excitation; add rotation about another direction")


def closed_loop_errors(transform, eef_inverse, marker_poses):
    marker_in_eef = eef_inverse @ transform @ marker_poses
    positions = marker_in_eef[:, :3, 3]
    position_errors = np.linalg.norm(positions - positions.mean(axis=0), axis=1) * 1000
    rotations = Rotation.from_matrix(marker_in_eef[:, :3, :3])
    rotation_errors = np.rad2deg((rotations.mean().inv() * rotations).magnitude())
    return position_errors, rotation_errors


def solve_hand_eye(eef_poses: np.ndarray, marker_poses: np.ndarray) -> HandEyeResult:
    """由 T_base_from_eef、T_camera_from_marker 求外参与残差，固定使用 HORAUD；质量门槛由写入边界执行。"""
    if len(eef_poses) < MIN_SAMPLES:
        raise ValueError(f"need at least {MIN_SAMPLES} samples, have {len(eef_poses)}")
    check_excitation(eef_poses[:, :3, :3])

    eef_inverse = np.broadcast_to(np.eye(4), eef_poses.shape).copy()
    eef_inverse[:, :3, :3] = eef_poses[:, :3, :3].transpose(0, 2, 1)
    eef_inverse[:, :3, 3] = -np.einsum("nij,nj->ni", eef_inverse[:, :3, :3], eef_poses[:, :3, 3])
    # Eye-to-Hand 必须传入 T_eef_from_base，输出才是 T_base_from_camera。
    rotation, translation = cv2.calibrateHandEye(
        list(eef_inverse[:, :3, :3]), list(eef_inverse[:, :3, 3]),
        list(marker_poses[:, :3, :3]), list(marker_poses[:, :3, 3]),
        method=cv2.CALIB_HAND_EYE_HORAUD,
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation.ravel()
    transform = validate_rigid_transform(transform, label="HORAUD hand-eye result")
    errors_mm, errors_deg = closed_loop_errors(transform, eef_inverse, marker_poses)
    if not np.isfinite(errors_mm).all() or not np.isfinite(errors_deg).all():
        raise ValueError("nonfinite hand-eye residuals")
    return HandEyeResult(transform, "HORAUD", errors_mm, errors_deg)
