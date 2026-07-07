"""Signal processing utilities: EMA smoothing, jerk limiting, complementary filter."""

from __future__ import annotations

__all__ = ["ComplementaryFilter", "ema_smooth", "ema_smooth_pose", "limit_jerk"]

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def ema_smooth(
    new_val: np.ndarray, prev_val: np.ndarray | None, alpha: float
) -> np.ndarray:
    """Exponential moving average smooth.

    Args:
        new_val: New sample value.
        prev_val: Previous smoothed value (None means no history).
        alpha: Smoothing factor in [0, 1]. 1.0 = no smoothing.

    Returns:
        Smoothed value.
    """
    if prev_val is None:
        return np.asarray(new_val, dtype=np.float64).copy()
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * np.asarray(new_val, dtype=np.float64) + (1.0 - alpha) * prev_val


def limit_jerk(
    cmd_vel: np.ndarray,
    prev_vel: np.ndarray | None,
    prev_accel: np.ndarray | None,
    dt: float,
    max_jerk: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-joint jerk clamp via proportional scaling.

    When |(accel - prev_accel)/dt| > max_jerk on any joint, all joints are
    scaled proportionally to preserve trajectory shape.

    Ref: LeFranX Ruckig jerk limiting.

    Args:
        cmd_vel: Desired joint velocity (rad/s).
        prev_vel: Previous velocity command (None → no limiting).
        prev_accel: Previous acceleration (None → no limiting).
        dt: Time step in seconds.
        max_jerk: Maximum jerk per joint (rad/s³).

    Returns:
        (clamped_vel, new_accel) — caller must save new_accel for next frame.
    """
    vel = np.asarray(cmd_vel, dtype=np.float64).copy()

    if prev_vel is None or prev_accel is None or dt <= 0:
        return vel, np.zeros_like(vel)

    # Current acceleration
    accel = (vel - prev_vel) / dt

    # Jerk = (accel - prev_accel) / dt
    jerk = (accel - prev_accel) / dt

    jerk_abs = np.abs(jerk)
    max_ratio = np.max(jerk_abs) / max_jerk

    if max_ratio > 1.0:
        # Scale jerk proportionally
        jerk = jerk / max_ratio
        # Reconstruct accel from clamped jerk
        accel = prev_accel + jerk * dt
        # Reconstruct vel from clamped accel
        vel = prev_vel + accel * dt

    return vel, accel.copy()


# ═══════════════════════════════════════════════════════════════════════════
# Cartesian-space pose EMA — position R³ + rotation vector so(3)
# ═══════════════════════════════════════════════════════════════════════════


def ema_smooth_pose(
    target_pos: np.ndarray,
    target_quat_wxyz: np.ndarray,
    prev_pos: np.ndarray,
    prev_quat_wxyz: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """EMA in Cartesian space: position R³ + rotation vector so(3).

    Smooths a 6-DOF EEF target pose before IK.  Position uses standard
    Euclidean EMA; orientation converts the quaternion to a rotation
    vector (axis * angle), applies EMA in so(3), then converts back.

    Rotation-vector EMA naturally takes the short geodesic path on S³
    (magnitude = angle ∈ [0, π]) without the overhead of scipy Slerp.

    Args:
        target_pos: (3,) target EEF position in meters.
        target_quat_wxyz: (4,) target EEF orientation quaternion (w, x, y, z).
        prev_pos: (3,) previous smoothed position.
        prev_quat_wxyz: (4,) previous smoothed orientation quaternion.
        alpha: Smoothing factor in [0, 1].  1.0 = no smoothing.

    Returns:
        ``(pos_smoothed, quat_wxyz_smoothed)`` — both float64 copies.
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))

    # Position: standard EMA in R³
    pos = alpha * np.asarray(target_pos, dtype=np.float64) + (1.0 - alpha) * np.asarray(prev_pos, dtype=np.float64)

    # Orientation: quat → rotvec → EMA → quat
    def _quat_to_rotvec(q: np.ndarray) -> np.ndarray:
        """Convert wxyz quaternion to rotation vector (axis * angle)."""
        w, x, y, z = q[0], q[1], q[2], q[3]
        sin_half = np.sqrt(x * x + y * y + z * z)
        if sin_half < 1e-12:
            return np.zeros(3, dtype=np.float64)
        angle = 2.0 * np.arctan2(sin_half, w)
        return angle * np.array([x, y, z], dtype=np.float64) / sin_half

    target_rv = _quat_to_rotvec(np.asarray(target_quat_wxyz, dtype=np.float64))
    prev_rv = _quat_to_rotvec(np.asarray(prev_quat_wxyz, dtype=np.float64))
    rv = alpha * target_rv + (1.0 - alpha) * prev_rv

    angle = float(np.linalg.norm(rv))
    if angle < 1e-12:
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    else:
        axis = rv / angle
        half = angle / 2.0
        quat = np.array([np.cos(half), axis[0] * np.sin(half),
                         axis[1] * np.sin(half), axis[2] * np.sin(half)], dtype=np.float64)

    return pos, quat


# ═══════════════════════════════════════════════════════════════════════════
# Complementary Filter — position EMA + orientation Slerp
# Ref: OpenTeach franka.py:21-33
# ═══════════════════════════════════════════════════════════════════════════


class ComplementaryFilter:
    """Decoupled pose-space smoothing: position EMA + orientation Slerp.

    Unlike joint-space EMA (which couples translation and rotation smoothing
    through the IK Jacobian), this filter operates directly on the EEF target
    pose BEFORE IK, so the two domains can be tuned independently.

    - **Position**: standard exponential moving average in Euclidean R³.
    - **Orientation**: spherical linear interpolation (Slerp) on the unit
      quaternion sphere S³.  Slerp guarantees constant angular velocity along
      the geodesic, unlike naive quaternion Lerp + renormalize which speeds up
      near the midpoint and slows down near the endpoints.

    Ref: OpenTeach ``franka.py:21-33`` — ``Filter`` class with identical
         EMA + Slerp pattern, used at 60 Hz with ``comp_ratio=0.8``.

    Args:
        pos_comp_ratio: Position retention ratio in [0, 1].
            Higher → more smoothing (0.8 = 80% old + 20% new per frame).
            Set to 0.0 to disable position filtering (pass-through).
        rot_comp_ratio: Orientation retention ratio in [0, 1].
            Higher → more smoothing.  Typically set slightly higher than
            ``pos_comp_ratio`` because VR wrist rotation is the dominant
            jitter source.
    """

    def __init__(self, pos_comp_ratio: float = 0.8, rot_comp_ratio: float = 0.9) -> None:
        self.pos_ratio = float(np.clip(pos_comp_ratio, 0.0, 1.0))
        self.rot_ratio = float(np.clip(rot_comp_ratio, 0.0, 1.0))
        self._pos_state: np.ndarray | None = None       # (3,) in meters
        self._ori_xyzw: np.ndarray | None = None         # (4,) xyzw for scipy

    # -- public API --

    def reset(self) -> None:
        """Discard filter state.  Next call will pass through unfiltered."""
        self._pos_state = None
        self._ori_xyzw = None

    @property
    def is_ready(self) -> bool:
        """True after at least one frame has been processed."""
        return self._pos_state is not None

    def __call__(
        self, pos: np.ndarray, quat_wxyz: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the complementary filter to a single pose sample.

        Args:
            pos: (3,) target EEF position in meters.
            quat_wxyz: (4,) target EEF orientation quaternion (w, x, y, z).

        Returns:
            ``(pos_filtered, quat_wxyz_filtered)`` — both float64 copies.
        """
        pos = np.asarray(pos, dtype=np.float64).ravel()[:3]
        quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64).ravel()[:4]

        # -- first frame: seed state, pass through --
        if self._pos_state is None:
            self._pos_state = pos.copy()
            self._ori_xyzw = _wxyz_to_xyzw(quat_wxyz)
            return pos.copy(), quat_wxyz.copy()

        # -- position: EMA in R³ --
        if self.pos_ratio > 0:
            self._pos_state = (
                self.pos_ratio * self._pos_state
                + (1.0 - self.pos_ratio) * pos
            )
        else:
            self._pos_state = pos.copy()

        # -- orientation: Slerp on S³ --
        if self.rot_ratio > 0:
            t = 1.0 - self.rot_ratio  # fraction of "new" orientation
            quat_xyzw_new = _wxyz_to_xyzw(quat_wxyz)

            # Ensure the two quaternions lie on the same hemisphere so Slerp
            # takes the SHORT geodesic path (q and −q represent the same
            # rotation but Slerp would otherwise go the long way around).
            if float(np.dot(self._ori_xyzw, quat_xyzw_new)) < 0:
                quat_xyzw_new = -quat_xyzw_new

            keyframes = Rotation.from_quat(np.stack([self._ori_xyzw, quat_xyzw_new]))
            self._ori_xyzw = Slerp([0.0, 1.0], keyframes)([t])[0].as_quat()
            quat_wxyz_out = _xyzw_to_wxyz(self._ori_xyzw)
        else:
            quat_wxyz_out = quat_wxyz.copy()
            self._ori_xyzw = _wxyz_to_xyzw(quat_wxyz)

        return self._pos_state.copy(), quat_wxyz_out


# -- quaternion convention helpers (wxyz ↔ xyzw) --


def _wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    """Convert [w, x, y, z] → [x, y, z, w] (scipy convention)."""
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)


def _xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    """Convert [x, y, z, w] → [w, x, y, z] (DexMani convention)."""
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)
