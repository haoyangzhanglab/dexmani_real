"""Offline arm/hand validation checks at the actuator-worker boundary."""

from __future__ import annotations

import time
import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.config.runtime import ArmLoopConfig, resolve_runtime_config
from dexmani_real.ipc.schema import COUPLED_COMMAND_DTYPE
from dexmani_real.robot.arm_worker import _handle_servo_command
from dexmani_real.robot.command_validation import (
    check_worker_arm_command,
    check_worker_hand_command,
)
from dexmani_real.runtime.safety import CoupledCommandTicket, MotionPermit, SafetyState


class _Value:
    def __init__(self, value: int) -> None:
        self.value = value


class WorkerCommandValidationTest(unittest.TestCase):
    def test_arm_uses_twenty_degree_command_jump_fallback(self) -> None:
        runtime = resolve_runtime_config()
        loop_config = ArmLoopConfig.from_runtime(runtime)
        self.assertAlmostEqual(
            float(np.rad2deg(loop_config.max_servo_command_jump_rad)),
            20.0,
        )

        now_ns = time.monotonic_ns()
        command = np.zeros(1, dtype=COUPLED_COMMAND_DTYPE)
        command["action_id"][0] = 1
        command["arm_present"][0] = 1
        command["run_generation"][0] = 7
        command["created_monotonic_ns"][0] = now_ns
        command["scheduled_target_monotonic_ns"][0] = now_ns
        command["target_monotonic_ns"][0] = now_ns
        command["valid_until_monotonic_ns"][0] = now_ns + 1_000_000_000
        kwargs: dict[str, Any] = {
            "expected_run_generation": 7,
            "now_monotonic_ns": now_ns,
            "joint_limit_lower_rad": np.full(7, -2.0 * np.pi),
            "joint_limit_upper_rad": np.full(7, 2.0 * np.pi),
            "previous_command_qpos_rad": np.deg2rad(np.full(7, 30.0)),
            "max_command_jump_rad": loop_config.max_servo_command_jump_rad,
        }

        command["arm_qpos"][0] = np.deg2rad(np.full(7, 49.9))
        self.assertIsNone(check_worker_arm_command(command, **kwargs))
        command["arm_qpos"][0] = np.deg2rad(np.full(7, 50.1))
        issue = check_worker_arm_command(command, **kwargs)

        assert issue is not None
        self.assertTrue(issue.fault)
        self.assertEqual(issue.reason, "command jump limit violation")

    def test_hand_distinguishes_stale_commands_from_contract_faults(self) -> None:
        runtime = resolve_runtime_config()
        now_ns = time.monotonic_ns()
        command = np.zeros(1, dtype=COUPLED_COMMAND_DTYPE)
        command["action_id"][0] = 1
        command["hand_present"][0] = 1
        command["run_generation"][0] = 7
        command["created_monotonic_ns"][0] = now_ns
        command["scheduled_target_monotonic_ns"][0] = now_ns
        command["target_monotonic_ns"][0] = now_ns
        command["valid_until_monotonic_ns"][0] = now_ns + 1_000_000_000
        command["hand_qpos"][0] = np.deg2rad(runtime.hand.home_qpos_deg)
        kwargs: dict[str, Any] = {
            "operational_lower_rad": np.asarray(runtime.hand.qpos_min_rad),
            "operational_upper_rad": np.asarray(runtime.hand.qpos_max_rad),
            "mechanical_lower_rad": np.asarray(runtime.hand.mechanical_qpos_min_rad),
            "mechanical_upper_rad": np.asarray(runtime.hand.mechanical_qpos_max_rad),
            "now_monotonic_ns": now_ns,
        }

        stale = check_worker_hand_command(command, expected_run_generation=8, **kwargs)
        valid = check_worker_hand_command(command, expected_run_generation=7, **kwargs)
        command["hand_qpos"][0] = (
            np.asarray(runtime.hand.mechanical_qpos_max_rad) + 0.01
        )
        unsafe = check_worker_hand_command(command, expected_run_generation=7, **kwargs)

        assert stale is not None and unsafe is not None
        self.assertFalse(stale.fault)
        self.assertIsNone(valid)
        self.assertTrue(unsafe.fault)

    def test_arm_and_hand_share_the_same_delivery_window(self) -> None:
        runtime = resolve_runtime_config()
        created_ns = time.monotonic_ns()
        command = np.zeros(1, dtype=COUPLED_COMMAND_DTYPE)
        command["action_id"][0] = 1
        command["arm_present"][0] = 1
        command["hand_present"][0] = 1
        command["run_generation"][0] = 7
        command["created_monotonic_ns"][0] = created_ns
        command["scheduled_target_monotonic_ns"][0] = created_ns
        command["target_monotonic_ns"][0] = created_ns
        command["valid_until_monotonic_ns"][0] = created_ns + 500_000_000
        command["arm_qpos"][0] = np.zeros(7)
        command["hand_qpos"][0] = np.deg2rad(runtime.hand.home_qpos_deg)
        now_ns = created_ns + 400_000_000

        arm_issue = check_worker_arm_command(
            command,
            expected_run_generation=7,
            now_monotonic_ns=now_ns,
            joint_limit_lower_rad=np.asarray(runtime.arm.joint_limit_lower),
            joint_limit_upper_rad=np.asarray(runtime.arm.joint_limit_upper),
            previous_command_qpos_rad=np.zeros(7),
            max_command_jump_rad=np.deg2rad(20.0),
        )
        hand_issue = check_worker_hand_command(
            command,
            operational_lower_rad=np.asarray(runtime.hand.qpos_min_rad),
            operational_upper_rad=np.asarray(runtime.hand.qpos_max_rad),
            mechanical_lower_rad=np.asarray(runtime.hand.mechanical_qpos_min_rad),
            mechanical_upper_rad=np.asarray(runtime.hand.mechanical_qpos_max_rad),
            expected_run_generation=7,
            now_monotonic_ns=now_ns,
        )

        self.assertIsNone(arm_issue)
        self.assertIsNone(hand_issue)

        expired_ns = int(command["valid_until_monotonic_ns"][0])
        arm_expired = check_worker_arm_command(
            command,
            expected_run_generation=7,
            now_monotonic_ns=expired_ns,
        )
        hand_expired = check_worker_hand_command(
            command,
            operational_lower_rad=np.asarray(runtime.hand.qpos_min_rad),
            operational_upper_rad=np.asarray(runtime.hand.qpos_max_rad),
            mechanical_lower_rad=np.asarray(runtime.hand.mechanical_qpos_min_rad),
            mechanical_upper_rad=np.asarray(runtime.hand.mechanical_qpos_max_rad),
            expected_run_generation=7,
            now_monotonic_ns=expired_ns,
        )
        assert arm_expired is not None and hand_expired is not None
        self.assertEqual(arm_expired.reason, "expired command")
        self.assertEqual(hand_expired.reason, "expired command")
        self.assertFalse(arm_expired.fault)
        self.assertFalse(hand_expired.fault)

    def test_arm_does_not_fault_on_superseded_unsafe_snapshot(self) -> None:
        now_ns = time.monotonic_ns()
        command = np.zeros(1, dtype=COUPLED_COMMAND_DTYPE)
        command["action_id"][0] = 5
        command["arm_present"][0] = 1
        command["run_generation"][0] = 7
        command["created_monotonic_ns"][0] = now_ns
        command["scheduled_target_monotonic_ns"][0] = now_ns
        command["target_monotonic_ns"][0] = now_ns
        command["valid_until_monotonic_ns"][0] = now_ns + 1_000_000_000
        command["arm_qpos"][0] = np.ones(7)

        arm = SimpleNamespace(servo=Mock(return_value=0))
        state = SimpleNamespace(
            cfg=SimpleNamespace(
                joint_limit_lower=tuple(np.full(7, -2.0)),
                joint_limit_upper=tuple(np.full(7, 2.0)),
                max_servo_command_jump_rad=0.1,
            ),
            arm=arm,
            last_target=np.zeros(7),
            last_measured_qpos=np.zeros(7),
            last_command_generation=7,
            last_rejected_ring_sequence=0,
        )
        shared = SimpleNamespace(
            error_state=_Value(0),
            estop_request=_Value(0),
        )
        ticket = CoupledCommandTicket(7, 11)

        with (
            patch(
                "dexmani_real.robot.arm_worker.read_motion_permit",
                return_value=MotionPermit(SafetyState.RUNNING, 7),
            ),
            patch(
                "dexmani_real.robot.arm_worker.coupled_command_ticket_allows_execution",
                side_effect=(True, False),
            ),
        ):
            _handle_servo_command(cast(Any, state), shared, command, ticket)

        arm.servo.assert_not_called()


if __name__ == "__main__":
    unittest.main()
