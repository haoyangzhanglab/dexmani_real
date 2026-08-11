"""Map VR wrist motion to target EEF pose."""

from __future__ import annotations

__all__ = ["ArmWristMapper"]

import numpy as np
from transforms3d.axangles import axangle2mat, mat2axangle
from transforms3d.quaternions import mat2quat, quat2mat

from dexmani_real.planning.pose_utils import normalize_quat_wxyz
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


def _clip_signed_axis_angle(rot: np.ndarray, max_abs_angle_rad: float) -> tuple[np.ndarray, float, bool]:
    """Clamp a rotation's signed axis-angle magnitude symmetrically."""
    axis, angle = mat2axangle(rot)
    if abs(angle) <= max_abs_angle_rad:
        return rot, float(angle), False
    clipped_angle = float(np.copysign(max_abs_angle_rad, angle))
    return axangle2mat(axis, clipped_angle, is_normalized=True), float(angle), True


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
        max_per_frame_rot_rad: float = 0.52,  # ~30°/frame at the default 16 Hz
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
        # ── Position: heading-dependent (set by set_heading) ──
        self.vr_to_base_rot = (
            np.eye(3) if vr_to_base_rot is None else _finite_vector(vr_to_base_rot, (3, 3), "vr_to_base_rot")
        )
        # ── Position + Rotation: base → world ──
        self.base_to_world_rot = (
            np.eye(3) if base_to_world_rot is None else _finite_vector(base_to_world_rot, (3, 3), "base_to_world_rot")
        )
        # ── Rotation: fixed VR→robot axis mapping (heading-INDEPENDENT) ──
        # LeFranX-style: a constant similarity transform mapping VR hand axes to
        # robot base axes.  Identity means VR FLU axes = robot base axes.
        self.T_vr_to_robot = (
            np.eye(3) if T_vr_to_robot is None else _finite_vector(T_vr_to_robot, (3, 3), "T_vr_to_robot")
        )
        # Bounds of target_eef_pos - eef_pos0 in robot base frame, shape (3, 2).
        self.eef_delta_bounds = (
            None if eef_delta_bounds is None else _finite_vector(eef_delta_bounds, (3, 2), "eef_delta_bounds")
        )
        if self.eef_delta_bounds is not None and np.any(self.eef_delta_bounds[:, 0] > self.eef_delta_bounds[:, 1]):
            raise ValueError("eef_delta_bounds lower values must not exceed upper values")
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
        # Validate into locals, then commit atomically.  A failed reset clears
        # the old anchor so callers cannot unknowingly continue from stale data.
        try:
            next_wrist_pos0 = _finite_vector(wrist_pos, (3,), "wrist_pos")
            next_wrist_rot0 = quat2mat(_unit_quat_wxyz(wrist_quat_wxyz, "wrist_quat_wxyz"))
            next_eef_pos0 = _finite_vector(eef_pos, (3,), "eef_pos")
            next_eef_rot0 = quat2mat(_unit_quat_wxyz(eef_quat_wxyz, "eef_quat_wxyz"))
        except (TypeError, ValueError):
            self.clear()
            logger.warning("ArmWristMapper.reset: invalid pose input — mapper cleared", exc_info=True)
            return
        self.wrist_pos0 = next_wrist_pos0
        self.wrist_rot0 = next_wrist_rot0
        self.eef_pos0 = next_eef_pos0
        self.eef_rot0 = next_eef_rot0
        # Seed continuous_quat in WORLD frame so the first map() dot-product
        # compares quaternions in the same coordinate system.
        # (base_to_world_rot is identity — base frame = world frame, zero transform.)
        _eef_rot0_world = self.base_to_world_rot @ self.eef_rot0
        self.last_quat_wxyz = mat2quat(_eef_rot0_world)
        self._last_wrist_rot = self.wrist_rot0.copy()

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
            logger.warning("ArmWristMapper.map: invalid wrist pose — holding", exc_info=True)
            return None

        # F2: Per-frame rotation delta gate — catches single-frame VR tracking
        # glitches (spike-and-recover) that the total-from-reset clip misses.
        # At the default 16 Hz, normal wrist rotation is <20°/frame; the 30°/frame
        # default (~480°/s) is ~2× the fastest plausible motion.
        #
        # _last_wrist_rot tracks the RAW wrist orientation as the delta reference
        # for the next frame.  Using the clamped output as reference caused the
        # baseline to drift after a spike, distorting recovery-frame deltas
        wrist_rot_gated = wrist_rot
        if self._last_wrist_rot is not None:
            frame_delta = wrist_rot @ self._last_wrist_rot.T
            frame_delta_clamped, frame_angle, was_clamped = _clip_signed_axis_angle(
                frame_delta, self.max_per_frame_rot_rad
            )
            if was_clamped:
                logger.warning(
                    "Per-frame rotation spike: %.1f° -> clamped to %.1f°",
                    np.rad2deg(frame_angle),
                    np.rad2deg(np.copysign(self.max_per_frame_rot_rad, frame_angle)),
                )
                wrist_rot_gated = frame_delta_clamped @ self._last_wrist_rot

        delta_pos_vr = current_wrist_pos - self.wrist_pos0
        delta_pos_base = self.pos_scale * (self.vr_to_base_rot @ delta_pos_vr)
        delta_pos_base = self.clip_delta_pos(delta_pos_base)
        # Transform base-frame delta → world frame before adding to
        # world-frame eef_pos0.  eef_pos0 comes from ArmFK (base frame),
        # so it must also be rotated to world frame for a consistent sum.
        delta_pos_world = self.base_to_world_rot @ delta_pos_base

        delta_rot_vr = wrist_rot_gated @ self.wrist_rot0.T  # type: ignore[union-attr]  # is_ready() gate above implies reset() ran (wrist_rot0 set)
        delta_rot_vr = self.scale_rot(delta_rot_vr)
        delta_rot_vr = self._clip_total_delta_rot(delta_rot_vr)
        # Rotation: fixed VR→robot axis mapping (heading-INDEPENDENT, LeFranX-style).
        # Similarity transform re-expresses the VR-frame rotation delta in robot-base axes.
        delta_rot_base = self.T_vr_to_robot @ delta_rot_vr @ self.T_vr_to_robot.T
        # Similarity-transform rotation delta from base frame → world frame.
        delta_rot_world = self.base_to_world_rot @ delta_rot_base @ self.base_to_world_rot.T

        # Convert eef reference from base frame → world frame so both
        # terms are in the same coordinate system before adding.
        # (is_ready() gate above guarantees eef_pos0/eef_rot0 are set.)
        eef_pos0_world = self.base_to_world_rot @ self.eef_pos0  # type: ignore[operator]  # is_ready() gate
        eef_rot0_world = self.base_to_world_rot @ self.eef_rot0  # type: ignore[operator]  # is_ready() gate

        target_pos = eef_pos0_world + delta_pos_world
        target_rot = delta_rot_world @ eef_rot0_world
        target_quat_wxyz = normalize_quat_wxyz(mat2quat(target_rot))
        if self.last_quat_wxyz is not None and np.dot(target_quat_wxyz, self.last_quat_wxyz) < 0:
            target_quat_wxyz = -target_quat_wxyz
        if not np.all(np.isfinite(target_pos)) or not np.all(np.isfinite(target_quat_wxyz)):
            logger.warning("ArmWristMapper.map: non-finite mapped pose — holding")
            return None

        # Mutate temporal state only after the complete input and mapped output
        # have passed validation.  Invalid frames therefore cannot poison the
        # next frame's glitch baseline or quaternion-continuity reference.
        self._last_wrist_rot = wrist_rot.copy()
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
        self._last_wrist_rot = None

    def is_ready(self) -> bool:
        return all(
            value is not None
            for value in (
                self.wrist_pos0,
                self.wrist_rot0,
                self.eef_pos0,
                self.eef_rot0,
                self._last_wrist_rot,
            )
        )

    def set_heading(self, head_quat_wxyz: np.ndarray) -> None:
        """Calibrate ``vr_to_base_rot`` so the user's facing direction → robot +X.

        **Only affects position mapping.** Rotation uses the fixed
        ``T_vr_to_robot`` transform (heading-independent, LeFranX-style).

        Call once per teleop session (on B-press), before :meth:`reset`.
        """
        try:
            head_q = _unit_quat_wxyz(head_quat_wxyz, "head_quat_wxyz")
        except (TypeError, ValueError):
            logger.warning("set_heading: invalid head quaternion, keeping current heading", exc_info=True)
            return

        head_rot = quat2mat(head_q)
        # Head forward in FLU: rotation matrix applied to FLU +X
        forward_flu = head_rot @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
        forward_2d = forward_flu[:2].copy()
        norm_2d = np.linalg.norm(forward_2d)

        if norm_2d < 1e-6:
            logger.warning(
                "set_heading: head forward nearly vertical (norm_2d=%.2e), keeping current heading",
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

        Catches accumulated drift from the reset pose before it reaches IK.
        Note: this is NOT per-frame — it clips the total rotation since reset().
        """
        clipped, angle, was_clamped = _clip_signed_axis_angle(delta_rot, self.max_delta_rot_rad)
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

    def continuous_quat(self, quat_wxyz: np.ndarray) -> np.ndarray:
        quat_wxyz = normalize_quat_wxyz(quat_wxyz)
        if self.last_quat_wxyz is not None and np.dot(quat_wxyz, self.last_quat_wxyz) < 0:
            quat_wxyz = -quat_wxyz
        self.last_quat_wxyz = quat_wxyz.copy()
        return quat_wxyz
