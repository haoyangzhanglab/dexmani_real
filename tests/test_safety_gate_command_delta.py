"""Safety-gate checks for command-rate references versus measured geometry."""

from __future__ import annotations

import unittest

import numpy as np

from dexmani_real.control.action import ActionCandidate
from dexmani_real.control.safety_gate import GateRejectCode, SafetyGate
from dexmani_real.teleop.action_proposal import limit_hand_target_delta


class SafetyGateCommandDeltaTest(unittest.TestCase):
    @staticmethod
    def _gate(*, endpoint_delta_tolerance_rad: float = 1e-12) -> SafetyGate:
        return SafetyGate(
            arm_joint_lower_rad=tuple(np.full(7, -1.0)),
            arm_joint_upper_rad=tuple(np.full(7, 1.0)),
            hand_joint_lower_rad=tuple(np.full(12, -1.0)),
            hand_joint_upper_rad=tuple(np.full(12, 1.0)),
            max_arm_delta_rad=0.1,
            endpoint_delta_tolerance_rad=endpoint_delta_tolerance_rad,
        )

    @staticmethod
    def _arm_candidate(value: float) -> ActionCandidate:
        return ActionCandidate(
            observation_id=1,
            run_generation=3,
            action_id=1,
            created_monotonic_ns=1,
            target_monotonic_ns=2,
            scheduled_target_monotonic_ns=1,
            valid_until_monotonic_ns=3,
            arm_qpos=np.full(7, value),
            hand_qpos=None,
        )

    def test_endpoint_tolerance_accepts_roundoff_but_rejects_real_excess(self) -> None:
        gate = self._gate()
        current = np.zeros(7)

        roundoff = gate.validate(
            self._arm_candidate(np.nextafter(0.1, np.inf)),
            current_arm_qpos=current,
            run_generation=3,
        )
        real_excess = gate.validate(
            self._arm_candidate(0.1 + 2e-12),
            current_arm_qpos=current,
            run_generation=3,
        )

        self.assertTrue(roundoff.accepted)
        self.assertEqual(real_excess.code, GateRejectCode.ARM_DELTA_LIMIT)

    def test_producer_clipped_hand_endpoint_survives_ulp_roundoff(self) -> None:
        previous = np.full(12, 0.3)
        clipped = limit_hand_target_delta(np.full(12, 1.0), previous, 0.1)
        gate = SafetyGate(
            arm_joint_lower_rad=tuple(np.full(7, -1.0)),
            arm_joint_upper_rad=tuple(np.full(7, 1.0)),
            hand_joint_lower_rad=tuple(np.full(12, -1.0)),
            hand_joint_upper_rad=tuple(np.full(12, 1.0)),
            max_hand_delta_rad=0.1,
        )
        candidate = ActionCandidate(
            observation_id=1,
            run_generation=3,
            action_id=1,
            created_monotonic_ns=1,
            target_monotonic_ns=2,
            scheduled_target_monotonic_ns=1,
            valid_until_monotonic_ns=3,
            arm_qpos=None,
            hand_qpos=clipped,
        )

        result = gate.validate(
            candidate,
            current_arm_qpos=np.zeros(7),
            current_hand_qpos=previous,
            run_generation=3,
            hand_delta_reference_qpos=previous,
        )

        self.assertTrue(result.accepted)

    def test_delta_uses_previous_command_while_geometry_uses_feedback(self) -> None:
        starts: dict[str, np.ndarray] = {}

        def workspace(start: np.ndarray, _end: np.ndarray) -> bool:
            starts["arm"] = start.copy()
            return True

        def collision(
            arm_start: np.ndarray,
            _arm_end: np.ndarray,
            hand_start: np.ndarray,
            _hand_end: np.ndarray,
        ) -> bool:
            starts["collision_arm"] = arm_start.copy()
            starts["hand"] = hand_start.copy()
            return True

        gate = SafetyGate(
            arm_joint_lower_rad=tuple(np.full(7, -1.0)),
            arm_joint_upper_rad=tuple(np.full(7, 1.0)),
            hand_joint_lower_rad=tuple(np.full(12, -1.0)),
            hand_joint_upper_rad=tuple(np.full(12, 1.0)),
            workspace_check=workspace,
            max_arm_delta_rad=0.1,
            max_hand_delta_rad=0.1,
            collision_check=collision,
        )
        candidate = ActionCandidate(
            observation_id=1,
            run_generation=3,
            action_id=1,
            created_monotonic_ns=1,
            target_monotonic_ns=2,
            scheduled_target_monotonic_ns=1,
            valid_until_monotonic_ns=3,
            arm_qpos=np.full(7, 0.15),
            hand_qpos=np.full(12, 0.15),
        )
        measured_arm = np.zeros(7)
        measured_hand = np.zeros(12)

        rejected = gate.validate(
            candidate,
            current_arm_qpos=measured_arm,
            current_hand_qpos=measured_hand,
            run_generation=3,
        )
        accepted = gate.validate(
            candidate,
            current_arm_qpos=measured_arm,
            current_hand_qpos=measured_hand,
            arm_delta_reference_qpos=np.full(7, 0.1),
            hand_delta_reference_qpos=np.full(12, 0.1),
            run_generation=3,
        )

        self.assertEqual(rejected.code, GateRejectCode.ARM_DELTA_LIMIT)
        self.assertTrue(accepted.accepted)
        np.testing.assert_array_equal(starts["arm"], measured_arm)
        np.testing.assert_array_equal(starts["collision_arm"], measured_arm)
        np.testing.assert_array_equal(starts["hand"], measured_hand)


if __name__ == "__main__":
    unittest.main()
