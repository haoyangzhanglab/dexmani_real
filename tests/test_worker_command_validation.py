"""Offline arm/hand validation checks at the actuator-worker boundary."""

from __future__ import annotations

import sys
import threading
import time
import types
import unittest
from enum import Enum
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
from dexmani_real.robot.hand_worker import hand_loop
from dexmani_real.robot_spec import (
    HAND_CONTACT_SHAPE,
    HAND_JOINT_SHAPE,
    HAND_TACTILE_FORCE_SHAPE,
    HAND_TACTILE_SUM_SHAPE,
)
from dexmani_real.runtime.safety import CoupledCommandTicket, MotionPermit, SafetyState


class _Value:
    def __init__(self, value: int) -> None:
        self.value = value


class _StartupSendStatus(Enum):
    ACCEPTED = "accepted"
    CRC_UNCONFIRMED = "crc_unconfirmed"
    REJECTED = "rejected"


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
            last_processed_ring_sequence=0,
            servo_call_count=0,
            duplicate_command_skip_count=0,
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

    def test_arm_applies_each_ring_sequence_at_most_once(self) -> None:
        now_ns = time.monotonic_ns()
        command = np.zeros(1, dtype=COUPLED_COMMAND_DTYPE)
        command["action_id"][0] = 5
        command["arm_present"][0] = 1
        command["run_generation"][0] = 7
        command["created_monotonic_ns"][0] = now_ns
        command["scheduled_target_monotonic_ns"][0] = now_ns
        command["target_monotonic_ns"][0] = now_ns
        command["valid_until_monotonic_ns"][0] = now_ns + 1_000_000_000
        command["arm_qpos"][0] = np.full(7, 0.05)

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
            last_processed_ring_sequence=0,
            servo_call_count=0,
            duplicate_command_skip_count=0,
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
                return_value=True,
            ),
        ):
            _handle_servo_command(cast(Any, state), shared, command, ticket)
            _handle_servo_command(cast(Any, state), shared, command, ticket)

        arm.servo.assert_called_once()
        self.assertEqual(state.last_processed_ring_sequence, 11)
        self.assertEqual(state.servo_call_count, 1)
        self.assertEqual(state.duplicate_command_skip_count, 1)

    def _run_hand_startup(self, reset_status: _StartupSendStatus) -> SimpleNamespace:
        runtime = resolve_runtime_config()
        mechanical_lower = np.asarray(runtime.hand.mechanical_qpos_min_rad)
        mechanical_upper = np.asarray(runtime.hand.mechanical_qpos_max_rad)
        home_qpos = np.deg2rad(np.asarray(runtime.hand.home_qpos_deg))
        near_lower = mechanical_lower + 0.2 * (mechanical_upper - mechanical_lower)
        near_upper = mechanical_lower + 0.8 * (mechanical_upper - mechanical_lower)
        startup_qpos = max(
            (near_lower, near_upper),
            key=lambda qpos: float(np.linalg.norm(qpos - home_qpos)),
        )
        self.assertTrue(np.all(startup_qpos >= mechanical_lower))
        self.assertTrue(np.all(startup_qpos <= mechanical_upper))
        self.assertGreater(np.linalg.norm(startup_qpos - home_qpos), 0.1)

        startup_events: list[str] = []
        startup_state = SimpleNamespace(
            qpos=startup_qpos,
            current_ma=np.zeros(HAND_JOINT_SHAPE),
            tactile_sum=np.zeros(HAND_TACTILE_SUM_SHAPE),
            tactile_sum_valid=True,
            tactile_contact=np.zeros(HAND_CONTACT_SHAPE, dtype=bool),
            tactile_force=np.zeros(HAND_TACTILE_FORCE_SHAPE),
            tactile_valid=True,
            commboard_err=np.zeros(HAND_JOINT_SHAPE, dtype=np.int32),
            jointboard_err=np.zeros(HAND_JOINT_SHAPE, dtype=np.int32),
            tipboard_err=np.zeros(HAND_JOINT_SHAPE, dtype=np.int32),
        )

        def reset_home() -> _StartupSendStatus:
            startup_events.append("reset_home")
            return reset_status

        def get_initial_state() -> SimpleNamespace:
            startup_events.append("initial_feedback")
            return startup_state

        hand = SimpleNamespace(
            connect=Mock(),
            calibrate_tactile=Mock(),
            reset_home=Mock(side_effect=reset_home),
            get_state=Mock(side_effect=get_initial_state),
            disconnect=Mock(),
            tactile_calibrated=True,
            is_connected=True,
        )
        hand_state_ring = SimpleNamespace(write=Mock())
        shared = SimpleNamespace(
            is_running=_Value(1),
            error_state=_Value(0),
            estop_request=_Value(0),
            hand_state_ring=hand_state_ring,
            hand_tactile_ring=SimpleNamespace(write=Mock()),
            set_heartbeat=Mock(),
        )
        ready_names: list[str] = []

        def set_ready(name: str) -> None:
            ready_names.append(name)
            startup_events.append("set_ready")
            shared.is_running.value = 0

        shared.set_ready = set_ready
        fake_xhand_module = types.ModuleType("dexmani_real.robot.xhand")
        fake_xhand_module.XHand = Mock(return_value=hand)
        fake_xhand_module.XHandSendStatus = _StartupSendStatus

        with (
            patch.dict(sys.modules, {"dexmani_real.robot.xhand": fake_xhand_module}),
            patch("dexmani_real.robot.hand_worker.time.sleep") as startup_sleep,
        ):
            hand_loop(shared, runtime.hand)

        return SimpleNamespace(
            startup_events=startup_events,
            startup_qpos=startup_qpos,
            hand=hand,
            shared=shared,
            ready_names=ready_names,
            hand_state_ring=hand_state_ring,
            startup_sleep=startup_sleep,
        )

    def test_hand_startup_reset_home_statuses(self) -> None:
        cases = (
            (_StartupSendStatus.ACCEPTED, True),
            (_StartupSendStatus.CRC_UNCONFIRMED, True),
            (_StartupSendStatus.REJECTED, False),
        )
        for status, ready in cases:
            with self.subTest(status=status.value):
                if status is _StartupSendStatus.CRC_UNCONFIRMED:
                    with self.assertLogs(
                        "dexmani_real.robot.hand_worker", level="WARNING"
                    ) as captured_logs:
                        result = self._run_hand_startup(status)
                    self.assertTrue(
                        any(
                            "unconfirmed" in message for message in captured_logs.output
                        )
                    )
                else:
                    result = self._run_hand_startup(status)

                result.hand.reset_home.assert_called_once_with()
                result.hand.disconnect.assert_called_once_with()
                result.startup_sleep.assert_not_called()
                if not ready:
                    self.assertEqual(result.startup_events, ["reset_home"])
                    result.hand.get_state.assert_not_called()
                    result.hand_state_ring.write.assert_not_called()
                    self.assertEqual(result.ready_names, [])
                    self.assertEqual(result.shared.error_state.value, 1)
                    continue

                self.assertEqual(
                    result.startup_events,
                    ["reset_home", "initial_feedback", "set_ready"],
                )
                result.hand.get_state.assert_called_once_with()
                result.hand_state_ring.write.assert_called_once()
                self.assertEqual(result.ready_names, ["hand"])
                self.assertEqual(result.shared.error_state.value, 0)
                np.testing.assert_array_equal(
                    result.hand_state_ring.write.call_args.args[0]["qpos"][0],
                    result.startup_qpos,
                )

    def test_hand_exit_stats_survive_post_ack_generation_revoke(self) -> None:
        """H4 evidence must retain the SDK ACK after coordinator disarms."""
        runtime = resolve_runtime_config()
        qpos = np.deg2rad(np.asarray(runtime.hand.home_qpos_deg))
        state = SimpleNamespace(
            qpos=qpos,
            current_ma=np.zeros(HAND_JOINT_SHAPE),
            tactile_sum=np.zeros(HAND_TACTILE_SUM_SHAPE),
            tactile_sum_valid=True,
            tactile_contact=np.zeros(HAND_CONTACT_SHAPE, dtype=bool),
            tactile_force=np.zeros(HAND_TACTILE_FORCE_SHAPE),
            tactile_valid=True,
            commboard_err=np.zeros(HAND_JOINT_SHAPE, dtype=np.int32),
            jointboard_err=np.zeros(HAND_JOINT_SHAPE, dtype=np.int32),
            tipboard_err=np.zeros(HAND_JOINT_SHAPE, dtype=np.int32),
        )
        now_ns = time.monotonic_ns()
        command = np.zeros(1, dtype=COUPLED_COMMAND_DTYPE)
        command["action_id"][0] = 2
        command["hand_present"][0] = 1
        command["run_generation"][0] = 5
        command["created_monotonic_ns"][0] = now_ns
        command["scheduled_target_monotonic_ns"][0] = now_ns
        command["target_monotonic_ns"][0] = now_ns
        command["valid_until_monotonic_ns"][0] = now_ns + 1_000_000_000
        command["hand_qpos"][0] = qpos

        shared = SimpleNamespace(
            is_running=_Value(1),
            error_state=_Value(0),
            estop_request=_Value(0),
            safety_state=_Value(int(SafetyState.RUNNING)),
            run_generation=_Value(5),
            active_coupled_command_sequence=_Value(1),
            motion_lock=threading.RLock(),
            hand_state_ring=SimpleNamespace(write=Mock()),
            hand_tactile_ring=SimpleNamespace(write=Mock()),
            coupled_cmd_ring=SimpleNamespace(
                read_latest=Mock(return_value=(command, now_ns, 1))
            ),
            set_heartbeat=Mock(),
            set_ready=Mock(),
        )
        state_reads = 0

        def get_state() -> SimpleNamespace:
            nonlocal state_reads
            state_reads += 1
            if state_reads == 3:
                # Let the second loop observe the post-ACK disarm generation
                # before it exits and writes its diagnostic summary.
                shared.safety_state.value = int(SafetyState.ARMED)
                shared.run_generation.value = 6
                shared.active_coupled_command_sequence.value = 0
                shared.is_running.value = 0
            return state

        hand = SimpleNamespace(
            connect=Mock(),
            calibrate_tactile=Mock(),
            reset_home=Mock(return_value=_StartupSendStatus.ACCEPTED),
            get_state=Mock(side_effect=get_state),
            send_action=Mock(return_value=_StartupSendStatus.ACCEPTED),
            disconnect=Mock(),
            tactile_calibrated=True,
            is_connected=True,
        )
        fake_xhand_module = types.ModuleType("dexmani_real.robot.xhand")
        fake_xhand_module.XHand = Mock(return_value=hand)
        fake_xhand_module.XHandSendStatus = _StartupSendStatus

        with (
            patch.dict(sys.modules, {"dexmani_real.robot.xhand": fake_xhand_module}),
            patch("dexmani_real.robot.hand_worker.LoopRate.wait"),
            self.assertLogs("dexmani_real.robot.hand_worker", level="INFO") as logs,
        ):
            hand_loop(shared, runtime.hand)

        hand.send_action.assert_called_once()
        np.testing.assert_array_equal(hand.send_action.call_args.args[0], qpos)
        self.assertTrue(
            any(
                "sdk_send_attempts=1, exact_target_accepts=1" in line
                for line in logs.output
            )
        )


if __name__ == "__main__":
    unittest.main()
