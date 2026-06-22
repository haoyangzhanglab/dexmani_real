"""hold-on-failure: on any pipeline failure, return current joint positions (hold in place).

Simplified design (ref: ufactory_teleop — no cumulative counters, just per-frame hold):
  VR tracking stale → hold
  Wrist mapper not ready → hold
  IK failed → hold
  Retarget failed → hold

Cumulative E-Stop escalation removed — per-frame hold is sufficient safety;
persistent failures are caught by robot.is_error() at the driver level.
"""

from __future__ import annotations

import numpy as np

from dexmani_real.robot.interface import RobotAction


class TeleopErrorHandler:
    """Per-frame hold-on-failure.  No cumulative counters, no E-Stop escalation.

    Keeps the last known-good arm and hand qpos so that any pipeline
    failure returns a ``RobotAction`` that holds the robot in place
    instead of sending a wild command.
    """

    def __init__(self) -> None:
        self._last_good_arm_qpos: np.ndarray | None = None
        self._last_good_hand_qpos: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_failure(self, stage: str, msg: str = "") -> RobotAction:
        """Record a failure at a given stage and return a hold action."""
        return self.hold_action()

    def init_fallback(self, arm_qpos: np.ndarray, hand_qpos: np.ndarray) -> None:
        """Initialize fallback positions from current state. Idempotent — only
        sets positions that haven't been recorded yet."""
        if self._last_good_arm_qpos is None:
            self._last_good_arm_qpos = np.asarray(arm_qpos, dtype=np.float64).copy()
        if self._last_good_hand_qpos is None:
            self._last_good_hand_qpos = np.asarray(hand_qpos, dtype=np.float64).copy()

    def hold_action(self) -> RobotAction:
        """Return a hold-in-place action (last good positions).

        Falls back to zero positions only if no good position has ever been
        recorded — this should never happen in normal operation since
        init_fallback() is called at the top of every _compute_action().
        """
        arm = (
            self._last_good_arm_qpos.copy()
            if self._last_good_arm_qpos is not None
            else np.zeros(7, dtype=np.float64)
        )
        hand = (
            self._last_good_hand_qpos.copy()
            if self._last_good_hand_qpos is not None
            else np.zeros(12, dtype=np.float64)
        )
        return RobotAction(arm_qpos_cmd=arm, hand_qpos_cmd=hand)

    def update_good_positions(self, arm_qpos: np.ndarray, hand_qpos: np.ndarray) -> None:
        """Update last-good positions after a successful arm + hand pipeline tick."""
        self._last_good_arm_qpos = np.asarray(arm_qpos, dtype=np.float64).copy()
        self._last_good_hand_qpos = np.asarray(hand_qpos, dtype=np.float64).copy()

    @property
    def is_ready(self) -> bool:
        """True once fallback positions have been initialised."""
        return self._last_good_arm_qpos is not None

    def clear(self) -> None:
        """Reset last-good positions (e.g. on state transition)."""
        self._last_good_arm_qpos = None
        self._last_good_hand_qpos = None


