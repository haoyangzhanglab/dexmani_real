"""Eye-to-hand 求解与刚性标记的闭环残差。"""

import logging
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
HAND_EYE_METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}
logger = logging.getLogger(__name__)


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
    """由 T_base_from_eef、T_camera_from_marker 求外参与残差，优先返回合格解。"""
    if len(eef_poses) < MIN_SAMPLES:
        raise ValueError(f"need at least {MIN_SAMPLES} samples, have {len(eef_poses)}")
    check_excitation(eef_poses[:, :3, :3])

    eef_inverse = np.broadcast_to(np.eye(4), eef_poses.shape).copy()
    eef_inverse[:, :3, :3] = eef_poses[:, :3, :3].transpose(0, 2, 1)
    eef_inverse[:, :3, 3] = -np.einsum("nij,nj->ni", eef_inverse[:, :3, :3], eef_poses[:, :3, 3])
    eef_rotations, eef_positions, marker_rotations, marker_positions = [
        list(values)
        for values in (
            eef_inverse[:, :3, :3],
            eef_inverse[:, :3, 3],
            marker_poses[:, :3, :3],
            marker_poses[:, :3, 3],
        )
    ]
    best = None
    best_score = (True, float("inf"), float("inf"))
    for name, method in HAND_EYE_METHODS.items():
        try:
            rotation, translation = cv2.calibrateHandEye(
                eef_rotations,
                eef_positions,
                marker_rotations,
                marker_positions,
                method=method,
            )
            transform = np.eye(4)
            transform[:3, :3] = rotation
            transform[:3, 3] = translation.ravel()
            transform = validate_rigid_transform(transform, label=f"{name} hand-eye candidate")
            errors_mm, errors_deg = closed_loop_errors(transform, eef_inverse, marker_poses)
            result = HandEyeResult(transform, name, errors_mm, errors_deg)
            rms = result.position_rms_mm, result.rotation_rms_deg
            if not np.isfinite(rms).all():
                raise ValueError("nonfinite hand-eye residuals")
        except (cv2.error, ValueError) as exc:
            logger.warning("%s: %s", name, exc)
            continue
        score = (not result.quality_ok, *rms)
        if score < best_score:
            best, best_score = result, score

    if best is None:
        raise ValueError("no hand-eye candidate produced a valid transform and residuals")
    return best
