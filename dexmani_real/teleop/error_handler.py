"""hold-on-failure + error counting + emergency-stop escalation.

LeFranX pattern: on any failure, return current joint positions (hold in place).
Five fallback points:
  1. VR tracking stale → hold
  2. Wrist mapper not ready → hold
  3. IK failed → hold
  4. Retarget failed → hold
  5. Joint jump abnormal → hold
"""

from __future__ import annotations

import numpy as np

from dexmani_real.robot.interface import RobotAction


class TeleopErrorHandler:
    """Hold-on-failure accumulator with emergency-stop escalation.

    Keeps track of the last known-good arm and hand qpos so that any
    pipeline failure can return a ``RobotAction`` that holds the robot
    in place instead of sending a wild command.
    """

    def __init__(
        self,
        ik_failure_limit: int = 10,
        vr_stale_limit: int = 50,  # frames at 50 Hz ≈ 1 s
        retarget_failure_limit: int = 20,  # hand tracking may be noisier
        wrist_map_failure_limit: int = 10,
        joint_jump_failure_limit: int = 5,
    ) -> None:
        self._last_good_arm_qpos: np.ndarray | None = None
        self._last_good_hand_qpos: np.ndarray | None = None

        self.ik_failures: int = 0
        self.retarget_failures: int = 0
        self.vr_stale_frames: int = 0
        self.wrist_map_failures: int = 0
        self.joint_jump_failures: int = 0

        self.ik_failure_limit = ik_failure_limit
        self.vr_stale_limit = vr_stale_limit
        self.retarget_failure_limit = retarget_failure_limit
        self.wrist_map_failure_limit = wrist_map_failure_limit
        self.joint_jump_failure_limit = joint_jump_failure_limit

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_success(self, arm_qpos: np.ndarray, hand_qpos: np.ndarray) -> None:
        self._last_good_arm_qpos = np.asarray(arm_qpos, dtype=np.float64).copy()
        self._last_good_hand_qpos = np.asarray(hand_qpos, dtype=np.float64).copy()
        self._reset_counters()

    def record_arm_ok(self, arm_qpos: np.ndarray) -> None:
        """IK succeeded: update last good arm position, reset IK counter."""
        self._last_good_arm_qpos = np.asarray(arm_qpos, dtype=np.float64).copy()
        self.ik_failures = 0

    def record_hand_ok(self, hand_qpos: np.ndarray) -> None:
        """Retarget succeeded: update last good hand position, reset retarget counter."""
        self._last_good_hand_qpos = np.asarray(hand_qpos, dtype=np.float64).copy()
        self.retarget_failures = 0

    def record_jump_ok(self) -> None:
        """Joint jump check passed: reset jump counter."""
        self.joint_jump_failures = 0

    def record_failure(self, stage: str, msg: str = "") -> RobotAction:
        """Record a failure at a given stage and return a hold action."""
        if stage == "ik":
            self.ik_failures += 1
        elif stage == "retarget":
            self.retarget_failures += 1
        elif stage == "vr_stale":
            self.vr_stale_frames += 1
        elif stage == "wrist_map":
            self.wrist_map_failures += 1
        elif stage == "joint_jump":
            self.joint_jump_failures += 1
        return self.hold_action()

    def init_fallback(self, arm_qpos: np.ndarray, hand_qpos: np.ndarray) -> None:
        """Initialize fallback positions from current state. Idempotent."""
        if self._last_good_arm_qpos is None:
            self._last_good_arm_qpos = np.asarray(arm_qpos, dtype=np.float64).copy()
        if self._last_good_hand_qpos is None:
            self._last_good_hand_qpos = np.asarray(hand_qpos, dtype=np.float64).copy()

    def hold_action(self) -> RobotAction:
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

    @property
    def should_emergency_stop(self) -> bool:
        return (
            self.ik_failures > self.ik_failure_limit
            or self.vr_stale_frames > self.vr_stale_limit
            or self.retarget_failures > self.retarget_failure_limit
            or self.wrist_map_failures > self.wrist_map_failure_limit
            or self.joint_jump_failures > self.joint_jump_failure_limit
        )

    def summary(self) -> str:
        return (
            f"ik={self.ik_failures}/{self.ik_failure_limit} "
            f"vr_stale={self.vr_stale_frames}/{self.vr_stale_limit} "
            f"retarget={self.retarget_failures}/{self.retarget_failure_limit} "
            f"wrist_map={self.wrist_map_failures}/{self.wrist_map_failure_limit} "
            f"joint_jump={self.joint_jump_failures}/{self.joint_jump_failure_limit}"
        )

    def clear(self) -> None:
        self._reset_counters()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _reset_counters(self) -> None:
        self.ik_failures = 0
        self.retarget_failures = 0
        self.vr_stale_frames = 0
        self.wrist_map_failures = 0
        self.joint_jump_failures = 0


def example() -> None:
    handler = TeleopErrorHandler()
    handler.record_success(
        arm_qpos=np.zeros(7), hand_qpos=np.zeros(12)
    )

    action = handler.record_failure("ik", "IK returned None")
    print(f"hold action arm={action.arm_qpos_cmd[:3]}...")
    print(f"summary: {handler.summary()}")

    # Simulate IK failures > limit
    for _ in range(11):
        handler.record_failure("ik")
    print(f"should_emergency_stop: {handler.should_emergency_stop}")


if __name__ == "__main__":
    example()
