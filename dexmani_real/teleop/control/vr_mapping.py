from __future__ import annotations

import logging

__all__ = ["VRWristMapper"]

import numpy as np
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
        if not np.isfinite(rot_scale) or rot_scale != 1.0:
            raise ValueError("only rot_scale=1.0 is supported by reset-relative rotation mapping")
        self.pos_scale = pos_scale
        self.vr_to_robot_rot = (
            np.eye(3)
            if vr_to_robot_rot is None
            else validate_rotation_matrix(vr_to_robot_rot, name="vr_to_robot_rot")
        )

        self.wrist_pos0: np.ndarray | None = None
        self.wrist_rot0: np.ndarray | None = None
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
        self.wrist_rot0 = next_wrist_rot0
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

        # C maps VR axes into xarm_base; conjugation expresses the anchored rotation there.
        delta_rot_vr = wrist_rot @ self.wrist_rot0.T
        delta_rot_base = self.vr_to_robot_rot @ delta_rot_vr @ self.vr_to_robot_rot.T
        target_pos = self.eef_pos0 + delta_pos_base
        target_rot = delta_rot_base @ self.eef_rot0
        target_quat_wxyz = normalize_quat_wxyz(mat2quat(target_rot))
        if self.last_quat_wxyz is not None and np.dot(target_quat_wxyz, self.last_quat_wxyz) < 0:
            target_quat_wxyz = -target_quat_wxyz
        if not np.all(np.isfinite(target_pos)) or not np.all(np.isfinite(target_quat_wxyz)):
            logger.warning("VRWristMapper.map: non-finite mapped pose — no target")
            return None

        self.last_quat_wxyz = target_quat_wxyz.copy()

        return {
            "pos": target_pos,
            "quat_wxyz": target_quat_wxyz,
        }

    def clear(self) -> None:
        self.wrist_pos0 = None
        self.wrist_rot0 = None
        self.eef_pos0 = None
        self.eef_rot0 = None
        self.last_quat_wxyz = None

    def is_ready(self) -> bool:
        return all(
            value is not None
            for value in (
                self.wrist_pos0,
                self.wrist_rot0,
                self.eef_pos0,
                self.eef_rot0,
            )
        )
