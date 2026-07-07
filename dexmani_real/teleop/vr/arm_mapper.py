"""Map VR wrist motion to target EEF pose."""

from __future__ import annotations

__all__ = ["ArmWristMapper"]

import numpy as np
from dexmani_real.utils.log import get_logger
from dexmani_real.planning.pose_utils import normalize_quat_wxyz
from transforms3d.axangles import axangle2mat, mat2axangle
from transforms3d.quaternions import mat2quat, quat2mat

logger = get_logger(__name__)


class ArmWristMapper:
    """Reset-relative wrist mapper."""

    def __init__(
        self,
        pos_scale: float = 1.0,
        rot_scale: float = 1.0,
        vr_to_base_rot: np.ndarray | None = None,
        base_to_world_rot: np.ndarray | None = None,
        eef_delta_bounds: np.ndarray | None = None,
        max_delta_rot_rad: float = 1.0,
    ) -> None:
        self.pos_scale = pos_scale
        self.rot_scale = rot_scale
        # Maps VR-frame deltas into robot-base-frame deltas.
        self.vr_to_base_rot = np.eye(3) if vr_to_base_rot is None else np.asarray(vr_to_base_rot, dtype=np.float64)
        # Maps base-frame deltas into world-frame deltas (accounts for base_pose_world).
        # Default identity means base == world (simulation case).
        self.base_to_world_rot = np.eye(3) if base_to_world_rot is None else np.asarray(base_to_world_rot, dtype=np.float64)
        # Bounds of target_eef_pos - eef_pos0 in robot base frame, shape (3, 2).
        self.eef_delta_bounds = None if eef_delta_bounds is None else np.asarray(eef_delta_bounds, dtype=np.float64)
        # Per-frame rotation delta cap (rad). ~57° default — catches VR tracking glitches.
        self.max_delta_rot_rad = max_delta_rot_rad

        self.wrist_pos0 = None
        self.wrist_rot0 = None
        self.eef_pos0 = None
        self.eef_rot0 = None
        self.last_quat_wxyz = None

    def reset(
        self,
        wrist_pos: np.ndarray,
        wrist_quat_wxyz: np.ndarray,
        eef_pos: np.ndarray,
        eef_quat_wxyz: np.ndarray,
    ) -> None:
        self.wrist_pos0 = np.asarray(wrist_pos, dtype=np.float64).copy()
        self.wrist_rot0 = quat2mat(normalize_quat_wxyz(wrist_quat_wxyz))
        self.eef_pos0 = np.asarray(eef_pos, dtype=np.float64).copy()
        self.eef_rot0 = quat2mat(normalize_quat_wxyz(eef_quat_wxyz))
        self.last_quat_wxyz = normalize_quat_wxyz(eef_quat_wxyz)

    def map(
        self,
        wrist_pos: np.ndarray,
        wrist_quat_wxyz: np.ndarray,
    ) -> dict[str, np.ndarray] | None:
        if not self.is_ready():
            return None

        wrist_pos = np.asarray(wrist_pos, dtype=np.float64)
        wrist_rot = quat2mat(normalize_quat_wxyz(wrist_quat_wxyz))

        delta_pos_vr = wrist_pos - self.wrist_pos0
        delta_pos_base = self.pos_scale * (self.vr_to_base_rot @ delta_pos_vr)
        delta_pos_base = self.clip_delta_pos(delta_pos_base)
        # Transform base-frame delta → world-frame delta before adding to
        # world-frame eef_pos0 (avoids frame mixing when base_pose_world != I).
        delta_pos_world = self.base_to_world_rot @ delta_pos_base

        delta_rot_vr = wrist_rot @ self.wrist_rot0.T
        delta_rot_vr = self.scale_rot(delta_rot_vr)
        delta_rot_vr = self._clip_delta_rot(delta_rot_vr)
        delta_rot_base = self.vr_to_base_rot @ delta_rot_vr @ self.vr_to_base_rot.T
        # Similarity-transform rotation delta from base frame → world frame.
        delta_rot_world = self.base_to_world_rot @ delta_rot_base @ self.base_to_world_rot.T

        target_pos = self.eef_pos0 + delta_pos_world
        target_rot = delta_rot_world @ self.eef_rot0
        target_quat_wxyz = self.continuous_quat(mat2quat(target_rot))

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
        return self.wrist_pos0 is not None and self.eef_pos0 is not None

    def set_heading(self, head_quat_wxyz: np.ndarray) -> None:
        """Calibrate ``vr_to_base_rot`` so the user's facing direction → robot +X.

        Extracts the head's forward direction in FLU, projects it to the
        horizontal (X-Y) plane, computes the yaw angle, and builds a
        rotation around FLU +Z that aligns the user's "forward" with the
        robot's +X axis.

        Call once per teleop session (on B-press), before :meth:`reset`.
        """
        head_q = np.asarray(head_quat_wxyz, dtype=np.float64)
        if not np.all(np.isfinite(head_q)):
            logger.warning("set_heading: head quaternion contains NaN/inf, keeping current heading")
            return
        norm = np.linalg.norm(head_q)
        if norm < 1e-12:
            logger.warning("set_heading: head quaternion is zero, keeping current heading")
            return
        head_q = head_q / norm

        head_rot = quat2mat(head_q)
        # Head forward in FLU: rotation matrix applied to FLU +X
        forward_flu = head_rot @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
        forward_2d = forward_flu[:2].copy()
        norm_2d = np.linalg.norm(forward_2d)

        if norm_2d < 1e-6:
            logger.warning(
                "set_heading: head forward nearly vertical (norm_2d=%.2e), "
                "keeping current heading",
                norm_2d,
            )
            return

        forward_2d /= norm_2d
        theta = np.arctan2(forward_2d[1], forward_2d[0])
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # R_z(-θ): maps head-forward direction → FLU +X → robot +X
        self.vr_to_base_rot = np.array(
            [
                [cos_t, sin_t, 0.0],
                [-sin_t, cos_t, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        logger.info(
            "set_heading: forward_2d=[%.3f, %.3f] theta=%.1f° → vr_to_base_rot set",
            forward_2d[0], forward_2d[1], np.rad2deg(theta),
        )

    def clip_delta_pos(self, delta_pos: np.ndarray) -> np.ndarray:
        if self.eef_delta_bounds is None:
            return delta_pos
        return np.clip(delta_pos, self.eef_delta_bounds[:, 0], self.eef_delta_bounds[:, 1])

    def scale_rot(self, rot: np.ndarray) -> np.ndarray:
        if self.rot_scale == 1.0:
            return rot
        axis, angle = mat2axangle(rot)
        return axangle2mat(axis, self.rot_scale * angle, is_normalized=True)

    def _clip_delta_rot(self, delta_rot: np.ndarray) -> np.ndarray:
        """Clamp per-frame rotation delta to prevent VR tracking glitches.

        Ref: ManiUniCon max_delta_rot=1.0rad (~57°).
        Catches transient VR jumps before they reach IK.
        """
        axis, angle = mat2axangle(delta_rot)
        if angle > self.max_delta_rot_rad:
            logger.debug(
                "clip_delta_rot: clamping %.3f rad -> %.3f rad",
                angle, self.max_delta_rot_rad,
            )
            return axangle2mat(axis, self.max_delta_rot_rad, is_normalized=True)
        return delta_rot

    def continuous_quat(self, quat_wxyz: np.ndarray) -> np.ndarray:
        quat_wxyz = normalize_quat_wxyz(quat_wxyz)
        if self.last_quat_wxyz is not None and np.dot(quat_wxyz, self.last_quat_wxyz) < 0:
            quat_wxyz = -quat_wxyz
        self.last_quat_wxyz = quat_wxyz.copy()
        return quat_wxyz


