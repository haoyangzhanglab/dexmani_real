"""Safety-gate checks for command-rate references versus measured geometry."""

from __future__ import annotations

import unittest

import numpy as np

from dexmani_real.control.action import ActionCandidate
from dexmani_real.control.safety_gate import GateRejectCode, SafetyGate
from dexmani_real.teleop.action_proposal import limit_hand_target_delta
from dexmani_real.utils.limits import (
    POLICY_HAND_ENDPOINT_ROUNDOFF_TOLERANCE_RAD,
    canonicalize_policy_hand_endpoint_roundoff,
)


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
        clipped = limit_hand_target_delta(np.full(12, 1.0), previous, 0.3)
        gate = SafetyGate(
            arm_joint_lower_rad=tuple(np.full(7, -1.0)),
            arm_joint_upper_rad=tuple(np.full(7, 1.0)),
            hand_joint_lower_rad=tuple(np.full(12, -1.0)),
            hand_joint_upper_rad=tuple(np.full(12, 1.0)),
            max_hand_delta_rad=0.3,
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

    def test_hand_limit_rejection_reports_joint_values_and_bounds(self) -> None:
        gate = SafetyGate(
            arm_joint_lower_rad=tuple(np.full(7, -1.0)),
            arm_joint_upper_rad=tuple(np.full(7, 1.0)),
            hand_joint_lower_rad=tuple(np.zeros(12)),
            hand_joint_upper_rad=tuple(np.ones(12)),
        )
        hand_qpos = np.full(12, 0.5)
        hand_qpos[2] = -0.125
        hand_qpos[9] = 1.25
        candidate = ActionCandidate(
            observation_id=1,
            run_generation=3,
            action_id=1,
            created_monotonic_ns=1,
            target_monotonic_ns=2,
            scheduled_target_monotonic_ns=1,
            valid_until_monotonic_ns=3,
            arm_qpos=np.zeros(7),
            hand_qpos=hand_qpos,
        )

        result = gate.validate(
            candidate,
            current_arm_qpos=np.zeros(7),
            current_hand_qpos=np.zeros(12),
            run_generation=3,
        )

        self.assertEqual(result.code, GateRejectCode.HAND_JOINT_LIMIT)
        self.assertEqual(
            result.detail,
            "hand joint limit violation (rad): "
            "j2: target=-0.125, lower=0, delta=-1.250e-01, "
            "j9: target=1.25, upper=1, delta=+2.500e-01",
        )

    def test_policy_hand_roundoff_is_canonicalized_without_relaxing_mechanics(
        self,
    ) -> None:
        operational_lower = np.zeros(12)
        operational_lower[5] = np.pi / 36.0
        operational_upper = np.full(12, 1.9)
        mechanical_lower = np.zeros(12)
        mechanical_upper = np.full(12, 2.0)
        policy_endpoint = operational_lower.copy()
        # The latest deployment checkpoint's j5 normalizer minimum is this far
        # below the float64 operational limit after float32 unnormalization.
        policy_endpoint[5] -= 3.9791546155298896e-8

        canonical, changed = canonicalize_policy_hand_endpoint_roundoff(
            policy_endpoint,
            operational_lower,
            operational_upper,
            mechanical_lower,
            mechanical_upper,
            mechanical_lower,
            mechanical_upper,
        )

        self.assertTrue(changed)
        self.assertEqual(canonical[5], operational_lower[5])
        np.testing.assert_array_equal(canonical[6:], policy_endpoint[6:])

        meaningful_violation = policy_endpoint.copy()
        meaningful_violation[5] = (
            operational_lower[5] - POLICY_HAND_ENDPOINT_ROUNDOFF_TOLERANCE_RAD - 1e-9
        )
        with self.assertRaisesRegex(ValueError, "target=.*lower=.*delta="):
            canonicalize_policy_hand_endpoint_roundoff(
                meaningful_violation,
                operational_lower,
                operational_upper,
                mechanical_lower,
                mechanical_upper,
                mechanical_lower,
                mechanical_upper,
            )

        mechanical_violation = policy_endpoint.copy()
        mechanical_violation[0] = -1e-8
        with self.assertRaisesRegex(ValueError, "rated mechanical"):
            canonicalize_policy_hand_endpoint_roundoff(
                mechanical_violation,
                operational_lower,
                operational_upper,
                mechanical_lower,
                mechanical_upper,
                mechanical_lower,
                mechanical_upper,
            )

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
            max_hand_delta_rad=0.3,
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

    def test_collision_false_and_checker_exception_have_distinct_codes(self) -> None:
        candidate = ActionCandidate(
            observation_id=1,
            run_generation=3,
            action_id=1,
            created_monotonic_ns=1,
            target_monotonic_ns=2,
            scheduled_target_monotonic_ns=1,
            valid_until_monotonic_ns=3,
            arm_qpos=np.full(7, 0.1),
            hand_qpos=np.full(12, 0.1),
        )
        common = {
            "arm_joint_lower_rad": tuple(np.full(7, -1.0)),
            "arm_joint_upper_rad": tuple(np.full(7, 1.0)),
            "hand_joint_lower_rad": tuple(np.full(12, -1.0)),
            "hand_joint_upper_rad": tuple(np.full(12, 1.0)),
        }
        in_collision = SafetyGate(collision_check=lambda *_args: False, **common)

        def checker_failure(*_args):
            raise LookupError("collision backend unavailable")

        broken = SafetyGate(collision_check=checker_failure, **common)
        kwargs = {
            "current_arm_qpos": np.zeros(7),
            "current_hand_qpos": np.zeros(12),
            "run_generation": 3,
        }
        self.assertEqual(
            in_collision.validate(candidate, **kwargs).code,
            GateRejectCode.COLLISION_TRANSITION,
        )
        self.assertEqual(
            broken.validate(candidate, **kwargs).code,
            GateRejectCode.COLLISION_CHECK_FAILED,
        )

    def test_learned_gate_uses_arm_worker_bound_and_disables_hand_delta(self) -> None:
        gate = SafetyGate(
            arm_joint_lower_rad=tuple(np.full(7, -1.0)),
            arm_joint_upper_rad=tuple(np.full(7, 1.0)),
            hand_joint_lower_rad=tuple(np.full(12, -1.0)),
            hand_joint_upper_rad=tuple(np.full(12, 1.0)),
            max_arm_delta_rad=np.deg2rad(20.0),
            max_hand_delta_rad=None,
        )
        accepted = gate.validate(
            ActionCandidate(
                observation_id=1,
                run_generation=3,
                action_id=1,
                created_monotonic_ns=1,
                target_monotonic_ns=2,
                scheduled_target_monotonic_ns=1,
                valid_until_monotonic_ns=3,
                arm_qpos=np.full(7, np.deg2rad(10.0)),
                hand_qpos=np.full(12, 0.9),
            ),
            current_arm_qpos=np.zeros(7),
            current_hand_qpos=np.zeros(12),
            run_generation=3,
        )
        rejected = gate.validate(
            self._arm_candidate(np.deg2rad(21.0)),
            current_arm_qpos=np.zeros(7),
            run_generation=3,
        )

        self.assertTrue(accepted.accepted)
        self.assertEqual(rejected.code, GateRejectCode.ARM_DELTA_LIMIT)


if __name__ == "__main__":
    unittest.main()
