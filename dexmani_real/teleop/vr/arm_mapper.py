"""Map VR wrist motion to target EEF pose."""

from __future__ import annotations

__all__ = ["ArmWristMapper"]

import numpy as np
from transforms3d.axangles import axangle2mat, mat2axangle
from transforms3d.quaternions import mat2quat, quat2mat

from dexmani_real.planning.pose_utils import normalize_quat_wxyz
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


class ArmWristMapper:
    """Reset-relative wrist mapper."""

    def __init__(
        self,
        pos_scale: float = 1.0,
        rot_scale: float = 1.0,
        vr_to_base_rot: np.ndarray | None = None,
        base_to_world_rot: np.ndarray | None = None,
        T_vr_to_robot: np.ndarray | None = None,
        eef_delta_bounds: np.ndarray | None = None,
        max_delta_rot_rad: float = 1.0,
        max_per_frame_rot_rad: float = 0.52,  # ~30°/frame @ 16 Hz — VR glitch gate
    ) -> None:
        self.pos_scale = pos_scale
        self.rot_scale = rot_scale
        # ── Position: heading-dependent (set by set_heading) ──
        self.vr_to_base_rot = np.eye(3) if vr_to_base_rot is None else np.asarray(vr_to_base_rot, dtype=np.float64)
        # ── Position + Rotation: base → world ──
        self.base_to_world_rot = (
            np.eye(3) if base_to_world_rot is None else np.asarray(base_to_world_rot, dtype=np.float64)
        )
        # ── Rotation: fixed VR→robot axis mapping (heading-INDEPENDENT) ──
        # LeFranX-style: a constant similarity transform mapping VR hand axes to
        # robot base axes.  Identity means VR FLU axes = robot base axes.
        self.T_vr_to_robot = np.eye(3) if T_vr_to_robot is None else np.asarray(T_vr_to_robot, dtype=np.float64)
        # Bounds of target_eef_pos - eef_pos0 in robot base frame, shape (3, 2).
        self.eef_delta_bounds = None if eef_delta_bounds is None else np.asarray(eef_delta_bounds, dtype=np.float64)
        # Total-from-reset rotation delta cap (rad). ~57° default — catches accumulated
        # drift from the reset pose before it reaches IK.
        self.max_delta_rot_rad = max_delta_rot_rad
        # Per-frame rotation delta cap (rad). ~30°/frame default — catches single-frame
        # VR tracking glitches (spike-and-recover) that the total-delta cap misses.
        self.max_per_frame_rot_rad = max_per_frame_rot_rad

        self.wrist_pos0: np.ndarray | None = None
        self.wrist_rot0: np.ndarray | None = None
        self.eef_pos0: np.ndarray | None = None
        self.eef_rot0: np.ndarray | None = None
        self.last_quat_wxyz: np.ndarray | None = None
        self._last_wrist_rot: np.ndarray | None = None  # F2: per-frame rotation delta gate

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
        self._last_wrist_rot = self.wrist_rot0.copy()

    def map(
        self,
        wrist_pos: np.ndarray,
        wrist_quat_wxyz: np.ndarray,
    ) -> dict[str, np.ndarray] | None:
        if not self.is_ready():
            return None

        wrist_pos = np.asarray(wrist_pos, dtype=np.float64)
        wrist_rot = quat2mat(normalize_quat_wxyz(wrist_quat_wxyz))

        # F2: Per-frame rotation delta gate — catches single-frame VR tracking
        # glitches (spike-and-recover) that the total-from-reset clip misses.
        # Normal human wrist rotation is < 20°/frame at 16 Hz; the 30°/frame
        # default (~480°/s) is ~2× the fastest plausible motion.
        #
        # _last_wrist_rot tracks the RAW wrist orientation as the delta reference
        # for the next frame.  Using the clamped output as reference caused the
        # baseline to drift after a spike, distorting recovery-frame deltas
        # (ref: arm-ik-adversarial-review §3.2 F8).
        wrist_rot_gated = wrist_rot
        if self._last_wrist_rot is not None:
            frame_delta = wrist_rot @ self._last_wrist_rot.T
            _axis, frame_angle = mat2axangle(frame_delta)
            if frame_angle > self.max_per_frame_rot_rad:
                logger.warning(
                    "Per-frame rotation spike: %.1f° -> clamped to %.1f°",
                    np.rad2deg(frame_angle),
                    np.rad2deg(self.max_per_frame_rot_rad),
                )
                frame_delta_clamped = axangle2mat(_axis, self.max_per_frame_rot_rad, is_normalized=True)
                wrist_rot_gated = frame_delta_clamped @ self._last_wrist_rot
        self._last_wrist_rot = wrist_rot.copy()

        delta_pos_vr = wrist_pos - self.wrist_pos0
        delta_pos_base = self.pos_scale * (self.vr_to_base_rot @ delta_pos_vr)
        delta_pos_base = self.clip_delta_pos(delta_pos_base)
        # Transform base-frame delta → world-frame delta before adding to
        # world-frame eef_pos0 (avoids frame mixing when base_pose_world != I).
        delta_pos_world = self.base_to_world_rot @ delta_pos_base

        delta_rot_vr = wrist_rot_gated @ self.wrist_rot0.T  # type: ignore[union-attr]  # is_ready() gate above implies reset() ran (wrist_rot0 set)
        delta_rot_vr = self.scale_rot(delta_rot_vr)
        delta_rot_vr = self._clip_total_delta_rot(delta_rot_vr)
        # Rotation: fixed VR→robot axis mapping (heading-INDEPENDENT, LeFranX-style).
        # Similarity transform re-expresses the VR-frame rotation delta in robot-base axes.
        delta_rot_base = self.T_vr_to_robot @ delta_rot_vr @ self.T_vr_to_robot.T
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
        self._last_wrist_rot = None

    def is_ready(self) -> bool:
        return self.wrist_pos0 is not None and self.eef_pos0 is not None

    def set_heading(self, head_quat_wxyz: np.ndarray) -> None:
        """Calibrate ``vr_to_base_rot`` so the user's facing direction → robot +X.

        **Only affects position mapping.** Rotation uses the fixed
        ``T_vr_to_robot`` transform (heading-independent, LeFranX-style).

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
                "set_heading: head forward nearly vertical (norm_2d=%.2e), " "keeping current heading",
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
            forward_2d[0],
            forward_2d[1],
            np.rad2deg(theta),
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

    def _clip_total_delta_rot(self, delta_rot: np.ndarray) -> np.ndarray:
        """Clamp total-from-reset rotation delta to prevent VR tracking glitches.

        Ref: ManiUniCon max_delta_rot=1.0rad (~57°).
        Catches accumulated drift from the reset pose before it reaches IK.
        Note: this is NOT per-frame — it clips the total rotation since reset().
        """
        axis, angle = mat2axangle(delta_rot)
        if angle > self.max_delta_rot_rad:
            logger.warning(
                "Total-from-reset rotation clamped: %.1f° -> %.1f° "
                "(max_delta_rot_rad=%.1f°).  EEF orientation will not track "
                "wrist beyond this limit.  Press B to re-calibrate at new pose.",
                np.rad2deg(angle),
                np.rad2deg(self.max_delta_rot_rad),
                np.rad2deg(self.max_delta_rot_rad),
            )
            return axangle2mat(axis, self.max_delta_rot_rad, is_normalized=True)
        return delta_rot

    def continuous_quat(self, quat_wxyz: np.ndarray) -> np.ndarray:
        quat_wxyz = normalize_quat_wxyz(quat_wxyz)
        if self.last_quat_wxyz is not None and np.dot(quat_wxyz, self.last_quat_wxyz) < 0:
            quat_wxyz = -quat_wxyz
        self.last_quat_wxyz = quat_wxyz.copy()
        return quat_wxyz
