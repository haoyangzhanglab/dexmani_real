"""Cartesian pose interpolator — smooths between discrete VR frames.

Receives target poses at VR frame rate (~25-50 Hz) and produces interpolated
poses at the controller's sampling rate (50 Hz), eliminating stale re-use of
the same VR frame across multiple control ticks.

Ref: ManiUniCon PoseTrajectoryInterpolator (pose_trajectory_interpolator.py:78-207).
Key simplifications vs ManiUniCon version:
  - No future waypoint queue — VR is 50 Hz native (vs 30 Hz in ManiUniCon)
  - Integrated with ArmWristMapper output convention (wxyz quaternion)
  - Optional: disabled by default, configurable via TeleopProfile
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


class CartPoseInterpolator:
    """Interpolates between discrete VR-frame poses for smooth robot motion.

    Receives target poses at VR rate and produces interpolated poses at the
    controller's sampling rate via:
      - Linear interpolation for position
      - SLERP (spherical linear interpolation) for rotation
      - Speed-limited temporal scheduling

    Usage in controller._compute_arm_command():
        interpolator.push_target_pose(target_pos, target_quat_wxyz)
        result = interpolator.get_interpolated_pose()
        if result is not None:
            target_pos, target_quat_wxyz = result
        # Then feed into IK as usual
    """

    def __init__(
        self,
        max_pos_speed: float = 0.25,    # m/s — prevents sudden position jumps
        max_rot_speed: float = 0.5,     # rad/s — prevents sudden rotation jumps
        max_history: int = 5,
    ) -> None:
        if max_pos_speed <= 0 or max_rot_speed <= 0:
            raise ValueError(
                f"max_pos_speed and max_rot_speed must be positive, "
                f"got {max_pos_speed}, {max_rot_speed}"
            )
        self.max_pos_speed = float(max_pos_speed)
        self.max_rot_speed = float(max_rot_speed)
        self._waypoints: deque[tuple[float, np.ndarray, np.ndarray]] = (
            deque(maxlen=max_history)
        )
        self._last_pos: np.ndarray | None = None
        self._last_rot: Rotation | None = None
        self._earliest_arrival_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push_target_pose(
        self,
        pos: np.ndarray,
        quat_wxyz: np.ndarray,
        timestamp: float | None = None,
    ) -> None:
        """Enqueue a new target waypoint (called at VR frame rate).

        Args:
            pos: (3,) EEF target position in world frame (meters).
            quat_wxyz: (4,) EEF target orientation quaternion (w,x,y,z).
            timestamp: monotonic time in seconds. Uses time.monotonic() if None.
        """
        ts = timestamp if timestamp is not None else time.monotonic()
        pos = np.asarray(pos, dtype=np.float64).reshape(3)
        quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
        quat_wxyz = quat_wxyz / np.linalg.norm(quat_wxyz)

        rot = Rotation.from_quat(
            np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        )

        # Compute earliest arrival time respecting speed limits
        if self._last_pos is not None and self._last_rot is not None:
            pos_dist = float(np.linalg.norm(pos - self._last_pos))
            rot_angle = self._rotation_angle(rot, self._last_rot)
            pos_time = pos_dist / self.max_pos_speed
            rot_time = rot_angle / self.max_rot_speed
            travel_time = max(pos_time, rot_time)
            self._earliest_arrival_time = max(ts, self._earliest_arrival_time + travel_time)
        else:
            self._earliest_arrival_time = ts

        self._last_pos = pos.copy()
        self._last_rot = rot
        self._waypoints.append((self._earliest_arrival_time, pos.copy(), quat_wxyz.copy()))

    def get_interpolated_pose(
        self, now: float | None = None
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Get interpolated pose at current time (called at controller rate).

        Returns:
            (pos: (3,), quat_wxyz: (4,)) or None if insufficient waypoints.

        Interpolation method:
          - Position: linear interpolation between two adjacent waypoints
          - Rotation: SLERP (spherical linear interpolation) between two
            adjacent waypoint rotations
        """
        if len(self._waypoints) < 2:
            if len(self._waypoints) == 1:
                _, pos, quat = self._waypoints[0]
                return pos.copy(), quat.copy()
            return None

        now = now if now is not None else time.monotonic()

        # Purge stale waypoints (arrival time already passed)
        while len(self._waypoints) > 1 and self._waypoints[1][0] < now:
            self._waypoints.popleft()

        if len(self._waypoints) < 2:
            return None

        t_prev, pos_prev, quat_prev = self._waypoints[0]
        t_next, pos_next, quat_next = self._waypoints[1]

        if t_next <= t_prev:
            return pos_prev.copy(), quat_prev.copy()

        # Clamped interpolation factor
        alpha = (now - t_prev) / (t_next - t_prev)
        alpha = max(0.0, min(1.0, alpha))

        # Linear position interpolation
        interp_pos = pos_prev + alpha * (pos_next - pos_prev)

        # SLERP rotation interpolation
        rot_prev = Rotation.from_quat(
            [quat_prev[1], quat_prev[2], quat_prev[3], quat_prev[0]]
        )
        rot_next = Rotation.from_quat(
            [quat_next[1], quat_next[2], quat_next[3], quat_next[0]]
        )
        slerp = Slerp([t_prev, t_next], Rotation.concatenate([rot_prev, rot_next]))
        interp_rot = slerp(now)
        interp_quat_xyzw = interp_rot.as_quat()
        # Convert back to wxyz convention
        interp_quat = np.array([
            interp_quat_xyzw[3], interp_quat_xyzw[0],
            interp_quat_xyzw[1], interp_quat_xyzw[2],
        ], dtype=np.float64)

        return interp_pos, interp_quat / np.linalg.norm(interp_quat)

    def reset(self) -> None:
        """Clear all waypoints (call on state transitions: IDLE→TELEOP, etc.)."""
        self._waypoints.clear()
        self._last_pos = None
        self._last_rot = None
        self._earliest_arrival_time = 0.0

    @property
    def ready(self) -> bool:
        """Whether enough waypoints exist for interpolation."""
        return len(self._waypoints) >= 2

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rotation_angle(rot_a: Rotation, rot_b: Rotation) -> float:
        """Angular distance between two rotations in radians."""
        delta = rot_a * rot_b.inv()
        return float(np.linalg.norm(delta.as_rotvec()))
