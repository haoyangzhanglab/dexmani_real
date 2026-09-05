"""Map VR wrist motion to target EEF pose."""

from __future__ import annotations

__all__ = ["VRWristMapper"]

import numpy as np
from transforms3d.axangles import axangle2mat, mat2axangle
from transforms3d.quaternions import mat2quat, quat2mat

from dexmani_real.planning.kinematics.pose import normalize_quat_wxyz
from dexmani_real.teleop.vr_transform import validate_rotation_matrix
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

_QUAT_NORM_EPS = 1e-12


def _finite_vector(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Return a finite float64 copy with the exact expected shape."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite array with shape {shape}")
    return array.copy()


def _unit_quat_wxyz(value: np.ndarray, name: str) -> np.ndarray:
    """Validate and normalize one wxyz quaternion without identity fallback."""
    quat = _finite_vector(value, (4,), name)
    norm = float(np.linalg.norm(quat))
    if norm < _QUAT_NORM_EPS:
        raise ValueError(f"{name} norm is too small")
    return quat / norm


def _clip_signed_axis_angle(
    rot: np.ndarray, max_abs_angle_rad: float
) -> tuple[np.ndarray, float, bool]:
    """Clamp a rotation's signed axis-angle magnitude symmetrically."""
    axis, angle = mat2axangle(rot)
    if abs(angle) <= max_abs_angle_rad:
        return rot, float(angle), False
    clipped_angle = float(np.copysign(max_abs_angle_rad, angle))
    return axangle2mat(axis, clipped_angle, is_normalized=True), float(angle), True


class VRWristMapper:
    """Reset-relative wrist mapper using one fixed VR-to-robot calibration."""

    def __init__(
        self,
        pos_scale: float = 1.0,
        rot_scale: float = 1.0,
        vr_to_robot_rot: np.ndarray | None = None,
        base_to_world_rot: np.ndarray | None = None,
        max_delta_rot_rad: float = 1.0,
        max_per_frame_rot_rad: float = 0.52,
    ) -> None:
        if not np.isfinite(pos_scale):
            raise ValueError("pos_scale must be finite")
        if not np.isfinite(rot_scale) or rot_scale < 0:
            raise ValueError("rot_scale must be finite and >= 0")
        if not np.isfinite(max_delta_rot_rad) or max_delta_rot_rad <= 0:
            raise ValueError("max_delta_rot_rad must be finite and > 0")
        if not np.isfinite(max_per_frame_rot_rad) or max_per_frame_rot_rad <= 0:
            raise ValueError("max_per_frame_rot_rad must be finite and > 0")
        self.pos_scale = pos_scale
        self.rot_scale = rot_scale
        self.vr_to_robot_rot = (
            np.eye(3)
            if vr_to_robot_rot is None
            else validate_rotation_matrix(vr_to_robot_rot, name="vr_to_robot_rot")
        )
        self.base_to_world_rot = (
            np.eye(3)
            if base_to_world_rot is None
            else validate_rotation_matrix(base_to_world_rot, name="base_to_world_rot")
        )
        self.max_delta_rot_rad = max_delta_rot_rad
        self.max_per_frame_rot_rad = max_per_frame_rot_rad

        self.wrist_pos0: np.ndarray | None = None
        self.wrist_rot0: np.ndarray | None = None
        self.eef_pos0: np.ndarray | None = None
        self.eef_rot0: np.ndarray | None = None
        self.last_quat_wxyz: np.ndarray | None = None
        self._accepted_wrist_rot: np.ndarray | None = None

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
            next_wrist_rot0 = quat2mat(
                _unit_quat_wxyz(wrist_quat_wxyz, "wrist_quat_wxyz")
            )
            next_eef_pos0 = _finite_vector(eef_pos, (3,), "eef_pos")
            next_eef_rot0 = quat2mat(_unit_quat_wxyz(eef_quat_wxyz, "eef_quat_wxyz"))
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
        # Seed the quaternion in WORLD coordinates for continuity checks.
        _eef_rot0_world = self.base_to_world_rot @ self.eef_rot0
        self.last_quat_wxyz = mat2quat(_eef_rot0_world)
        self._accepted_wrist_rot = self.wrist_rot0.copy()

    def map(
        self,
        wrist_pos: np.ndarray,
        wrist_quat_wxyz: np.ndarray,
    ) -> dict[str, np.ndarray] | None:
        if not self.is_ready():
            return None

        try:
            current_wrist_pos = _finite_vector(wrist_pos, (3,), "wrist_pos")
            wrist_rot = quat2mat(_unit_quat_wxyz(wrist_quat_wxyz, "wrist_quat_wxyz"))
        except (TypeError, ValueError):
            logger.warning(
                "VRWristMapper.map: invalid wrist pose — holding", exc_info=True
            )
            return None

        # Advance from the previously accepted pose, not from a raw tracking
        # sample. A clamped spike can therefore never bypass the next frame's
        # rotation-rate bound.
        accepted_wrist_rot = wrist_rot
        if self._accepted_wrist_rot is not None:
            frame_delta = wrist_rot @ self._accepted_wrist_rot.T
            frame_delta_clamped, frame_angle, was_clamped = _clip_signed_axis_angle(
                frame_delta, self.max_per_frame_rot_rad
            )
            if was_clamped:
                logger.warning(
                    "Per-frame rotation spike: %.1f° -> clamped to %.1f°",
                    np.rad2deg(frame_angle),
                    np.rad2deg(np.copysign(self.max_per_frame_rot_rad, frame_angle)),
                )
                accepted_wrist_rot = frame_delta_clamped @ self._accepted_wrist_rot

        delta_pos_vr = current_wrist_pos - self.wrist_pos0
        delta_pos_base = self.pos_scale * (self.vr_to_robot_rot @ delta_pos_vr)
        # Rotate the base-frame delta into world coordinates before adding it.
        delta_pos_world = self.base_to_world_rot @ delta_pos_base

        delta_rot_vr = accepted_wrist_rot @ self.wrist_rot0.T  # type: ignore[union-attr]  # is_ready() gate above implies reset() ran (wrist_rot0 set)
        delta_rot_vr = self.scale_rot(delta_rot_vr)
        delta_rot_vr = self._clip_total_delta_rot(delta_rot_vr)
        # Re-express the VR rotation delta in robot-base axes.
        delta_rot_base = self.vr_to_robot_rot @ delta_rot_vr @ self.vr_to_robot_rot.T
        delta_rot_world = (
            self.base_to_world_rot @ delta_rot_base @ self.base_to_world_rot.T
        )

        # Convert the EEF reference to world coordinates before combining terms.
        eef_pos0_world = self.base_to_world_rot @ self.eef_pos0  # type: ignore[operator]  # is_ready() gate
        eef_rot0_world = self.base_to_world_rot @ self.eef_rot0  # type: ignore[operator]  # is_ready() gate

        target_pos = eef_pos0_world + delta_pos_world
        target_rot = delta_rot_world @ eef_rot0_world
        target_quat_wxyz = normalize_quat_wxyz(mat2quat(target_rot))
        if (
            self.last_quat_wxyz is not None
            and np.dot(target_quat_wxyz, self.last_quat_wxyz) < 0
        ):
            target_quat_wxyz = -target_quat_wxyz
        if not np.all(np.isfinite(target_pos)) or not np.all(
            np.isfinite(target_quat_wxyz)
        ):
            logger.warning("VRWristMapper.map: non-finite mapped pose — holding")
            return None

        # Commit temporal state only after input and output validation succeeds.
        self._accepted_wrist_rot = accepted_wrist_rot.copy()
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
        self._accepted_wrist_rot = None

    def is_ready(self) -> bool:
        return all(
            value is not None
            for value in (
                self.wrist_pos0,
                self.wrist_rot0,
                self.eef_pos0,
                self.eef_rot0,
                self._accepted_wrist_rot,
            )
        )

    def scale_rot(self, rot: np.ndarray) -> np.ndarray:
        if self.rot_scale == 1.0:
            return rot
        axis, angle = mat2axangle(rot)
        return axangle2mat(axis, self.rot_scale * angle, is_normalized=True)

    def _clip_total_delta_rot(self, delta_rot: np.ndarray) -> np.ndarray:
        """Clamp total-from-reset rotation delta to prevent VR tracking glitches.

        Catches accumulated drift from the reset pose before it reaches IK.
        Note: this is NOT per-frame — it clips the total rotation since reset().
        """
        clipped, angle, was_clamped = _clip_signed_axis_angle(
            delta_rot, self.max_delta_rot_rad
        )
        if was_clamped:
            logger.warning(
                "Total-from-reset rotation clamped: %.1f° -> %.1f° "
                "(max_delta_rot_rad=%.1f°).  EEF orientation will not track "
                "wrist beyond this limit.  Press B to re-calibrate at new pose.",
                np.rad2deg(angle),
                np.rad2deg(np.copysign(self.max_delta_rot_rad, angle)),
                np.rad2deg(self.max_delta_rot_rad),
            )
            return clipped
        return delta_rot
