"""Map VR wrist motion to target EEF pose."""

from __future__ import annotations

import logging

__all__ = ["VRWristMapper"]

import numpy as np
from transforms3d.axangles import axangle2mat, mat2axangle
from transforms3d.quaternions import mat2quat, quat2mat

from dexmani_real.utils.geometry import normalize_quat_wxyz, validate_rotation_matrix


logger = logging.getLogger(__name__)


def _finite_vector(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Return a finite float64 copy with the exact expected shape."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite array with shape {shape}")
    return array.copy()


class VRWristMapper:
    """Reset-relative wrist mapper in xarm_base, using one VR calibration."""

    def __init__(
        self,
        pos_scale: float = 1.0,
        rot_scale: float = 1.0,
        vr_to_robot_rot: np.ndarray | None = None,
    ) -> None:
        if not np.isfinite(pos_scale):
            raise ValueError("pos_scale must be finite")
        if not np.isfinite(rot_scale) or rot_scale < 0:
            raise ValueError("rot_scale must be finite and >= 0")
        self.pos_scale = pos_scale
        self.rot_scale = rot_scale
        self.vr_to_robot_rot = (
            np.eye(3)
            if vr_to_robot_rot is None
            else validate_rotation_matrix(vr_to_robot_rot, name="vr_to_robot_rot")
        )

        self.wrist_pos0: np.ndarray | None = None
        self.previous_wrist_rot: np.ndarray | None = None
        self.scaled_delta_rot: np.ndarray | None = None
        self.eef_pos0: np.ndarray | None = None
        self.eef_rot0: np.ndarray | None = None
        self.last_quat_wxyz: np.ndarray | None = None

    def reset(
        self,
        wrist_pos: np.ndarray,
        wrist_quat_wxyz: np.ndarray,
        eef_pos: np.ndarray,
        eef_quat_wxyz: np.ndarray,
    ) -> None:
        # Validate before committing the new anchor; failed resets clear stale state.
        try:
            next_wrist_pos0 = _finite_vector(wrist_pos, (3,), "wrist_pos")
            next_wrist_rot0 = quat2mat(normalize_quat_wxyz(wrist_quat_wxyz, name="wrist_quat_wxyz"))
            next_eef_pos0 = _finite_vector(eef_pos, (3,), "eef_pos")
            next_eef_rot0 = quat2mat(normalize_quat_wxyz(eef_quat_wxyz, name="eef_quat_wxyz"))
        except (TypeError, ValueError):
            self.clear()
            logger.warning(
                "VRWristMapper.reset: invalid pose input — mapper cleared",
                exc_info=True,
            )
            return
        self.wrist_pos0 = next_wrist_pos0
        self.previous_wrist_rot = next_wrist_rot0
        self.scaled_delta_rot = np.eye(3)
        self.eef_pos0 = next_eef_pos0
        self.eef_rot0 = next_eef_rot0
        self.last_quat_wxyz = mat2quat(self.eef_rot0)

    def map(
        self,
        wrist_pos: np.ndarray,
        wrist_quat_wxyz: np.ndarray,
    ) -> dict[str, np.ndarray] | None:
        if not self.is_ready():
            return None

        try:
            current_wrist_pos = _finite_vector(wrist_pos, (3,), "wrist_pos")
            wrist_rot = quat2mat(normalize_quat_wxyz(wrist_quat_wxyz, name="wrist_quat_wxyz"))
        except (TypeError, ValueError):
            logger.warning("VRWristMapper.map: invalid wrist pose — no target", exc_info=True)
            return None

        delta_pos_vr = current_wrist_pos - self.wrist_pos0
        delta_pos_base = self.pos_scale * (self.vr_to_robot_rot @ delta_pos_vr)

        # Scale local SO(3) increments, avoiding the total-angle branch at pi.
        increment = wrist_rot @ self.previous_wrist_rot.T
        self.scaled_delta_rot = self.scale_rot(increment) @ self.scaled_delta_rot
        self.previous_wrist_rot = wrist_rot
        delta_rot_vr = self.scaled_delta_rot
        # Re-express the VR rotation delta in robot-base axes.
        delta_rot_base = self.vr_to_robot_rot @ delta_rot_vr @ self.vr_to_robot_rot.T
        target_pos = self.eef_pos0 + delta_pos_base
        target_rot = delta_rot_base @ self.eef_rot0
        target_quat_wxyz = normalize_quat_wxyz(mat2quat(target_rot))
        if self.last_quat_wxyz is not None and np.dot(target_quat_wxyz, self.last_quat_wxyz) < 0:
            target_quat_wxyz = -target_quat_wxyz
        if not np.all(np.isfinite(target_pos)) or not np.all(np.isfinite(target_quat_wxyz)):
            logger.warning("VRWristMapper.map: non-finite mapped pose — no target")
            return None

        # Retain the quaternion sign for the next target.
        self.last_quat_wxyz = target_quat_wxyz.copy()

        return {
            "pos": target_pos,
            "quat_wxyz": target_quat_wxyz,
        }

    def clear(self) -> None:
        self.wrist_pos0 = None
        self.previous_wrist_rot = None
        self.scaled_delta_rot = None
        self.eef_pos0 = None
        self.eef_rot0 = None
        self.last_quat_wxyz = None

    def is_ready(self) -> bool:
        return all(
            value is not None
            for value in (
                self.wrist_pos0,
                self.previous_wrist_rot,
                self.scaled_delta_rot,
                self.eef_pos0,
                self.eef_rot0,
            )
        )

    def scale_rot(self, rot: np.ndarray) -> np.ndarray:
        if self.rot_scale == 1.0:
            return rot
        axis, angle = mat2axangle(rot)
        return axangle2mat(axis, self.rot_scale * angle, is_normalized=True)
