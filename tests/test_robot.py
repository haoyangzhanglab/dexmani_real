import multiprocessing as mp
import sys
import threading
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.config import load_experiment_config
from dexmani_real.robot.commands import OperationCancelled, RobotCommand, RobotError
from dexmani_real.robot.system import Robot
from dexmani_real.robot.xarm7 import worker
from dexmani_real.robot.xarm7.config import ArmConfig
from dexmani_real.robot.xarm7.driver import XArm7
from dexmani_real.robot.xarm7.process import ArmProcess
from dexmani_real.robot.xarm7.protocol import (
    GOAL_DTYPE,
    HomeRequest,
    LifecycleReply,
    PrepareRequest,
)
from dexmani_real.robot.xarm7.trajectory import ArmTrajectory
from dexmani_real.robot.xhand import driver as hand_driver
from dexmani_real.robot.xhand.config import HandConfig
from dexmani_real.robot.xhand.feedback import XHandFeedback, XHandState
from dexmani_real.runtime.preflight import validate_robot_config
from scripts.tools.control_xhand import XHandControlExample


def make_arm_process():
    ctx = mp.get_context("spawn")
    return ArmProcess(
        ArmConfig(),
        run_signal=ctx.Value("q", 1),
        stop_signal=ctx.Event(),
        emergency_signal=ctx.Event(),
        signal_lock=ctx.Lock(),
        input_signal=ctx.Value("q", 3),
    )


class Clock:
    def __init__(self):
        self.ns = 1_000_000_000

    def monotonic_ns(self):
        return self.ns

    def monotonic(self):
        return self.ns / 1e9

    def sleep(self, seconds):
        self.ns += round(seconds * 1e9)


class ArmTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        for name in ("monotonic_ns", "monotonic"):
            patcher = patch.object(worker.time, name, getattr(self.clock, name))
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = make_arm_process()
        self.addCleanup(self.client.pipe.close)
        self.addCleanup(self.client.child_pipe.close)
        self.io = self.client.io
        self.io.connected.set()
        self.io.health_until.value = self.clock.ns + 1_000_000_000
        self.driver = Mock(mode=1, is_connected=True)
        self.controller = worker.ArmController(ArmConfig(), self.io, Mock(), self.driver)
        self.qpos = np.asarray(ArmConfig().home_qpos)
        status = self.controller.status
        status["frame"]["qpos"] = self.qpos
        status["frame"]["timestamp_ns"] = self.clock.ns
        status["valid"], status["prepared_run_id"] = True, 1
        self.io.publish_status(status)
        self.reference = Mock()
        self.reference.step.return_value = self.qpos.copy()
        patcher = patch.object(worker, "ArmTrajectory", return_value=self.reference)
        patcher.start()
        self.addCleanup(patcher.stop)

    def publish_goal(self, **changes):
        slot = np.frombuffer(self.io.goal, dtype=GOAL_DTYPE, count=1)
        slot[0] = (1, 1, 3, self.clock.ns + 500_000_000, self.clock.ns + 500_000_000, self.qpos)
        for name, value in changes.items():
            slot[name] = value

    def test_revoked_stale_and_expired_goals_never_send(self):
        for changes, reason in (
            ({"submit_until_ns": self.clock.ns}, "submission_expired"),
            ({"execute_until_ns": self.clock.ns}, "input_decision_expired"),
        ):
            with self.subTest(changes=changes):
                self.publish_goal(**changes)
                with self.assertRaisesRegex(RuntimeError, reason):
                    self.controller.tick()
        self.publish_goal(input_generation=2)
        self.controller.tick()
        self.assertTrue(self.io.stop.is_set())
        self.driver.servo.assert_not_called()

    def test_checks_are_repeated_after_reference_computation(self):
        self.publish_goal()
        self.reference.step.side_effect = lambda *args: self.io.stop.set() or self.qpos.copy()
        self.controller.tick()
        self.driver.servo.assert_not_called()

    def test_reference_and_sdk_stalls_do_not_trigger_catchup_sends(self):
        self.publish_goal()
        self.reference.step.side_effect = lambda *args: self.clock.sleep(0.03) or self.qpos.copy()
        with self.assertRaisesRegex(RuntimeError, "stalled"):
            self.controller.tick()
        self.driver.servo.assert_not_called()
        self.reference.step.side_effect = None
        self.controller.active = None
        self.publish_goal()
        self.driver.servo.side_effect = lambda q: self.clock.sleep(0.03)
        with self.assertRaisesRegex(TimeoutError, "SDK blocked"):
            self.controller.tick()
        self.driver.servo.assert_called_once()

    def test_stale_feedback_and_tracking_error_reject_targets(self):
        self.publish_goal()
        self.controller.status["frame"]["timestamp_ns"] = self.clock.ns - 200_000_000
        with self.assertRaisesRegex(RuntimeError, "arm_feedback_expired"):
            self.controller.tick()
        self.controller.status["frame"]["timestamp_ns"] = self.clock.ns
        self.reference.step.return_value = self.qpos + 0.5
        self.publish_goal()
        with self.assertRaisesRegex(RuntimeError, "tracking_error"):
            self.controller.tick()
        self.driver.servo.assert_not_called()

    def test_execution_limit_applies_only_to_affected_sequences(self):
        self.io.execution_limit[:] = (self.clock.ns, 1)
        self.publish_goal()
        with self.assertRaisesRegex(RuntimeError, "input_decision_expired"):
            self.controller.tick()
        self.publish_goal(sequence=2)
        self.controller.tick()
        self.driver.servo.assert_called_once()

    def test_stop_failure_remains_visible_after_acknowledgment(self):
        self.driver.stop.side_effect = OSError("stop rejected")
        self.publish_goal()
        self.controller.stop()
        self.assertTrue(self.io.stopped.is_set())
        self.assertIn(b"stop rejected", self.io.close_error.value)
        self.assertIsNone(self.io.take_goal())
        self.client.process = Mock(is_alive=Mock(return_value=True))
        with self.assertRaisesRegex(RuntimeError, "stop rejected"):
            self.client.wait_stopped()

    def test_home_deadline_is_checked_by_child_without_parent_service(self):
        self.controller.operation(
            HomeRequest(1, 1, np.array([self.qpos]), self.qpos, self.clock.ns)
        )
        self.driver.enter_mode.assert_not_called()
        self.driver.move_home_milestone.assert_not_called()
        reply = self.controller.pipe.send.call_args.args[0]
        self.assertIsInstance(reply.error, TimeoutError)
        self.assertIn("deadline", str(reply.error))
        self.driver.stop.assert_called_once()

    def test_child_errors_cross_pipe_without_losing_type_or_stop_failure(self):
        self.client.process = Mock(is_alive=Mock(return_value=True))
        self.controller.pipe = self.client.child_pipe
        request = HomeRequest(1, 1, np.array([self.qpos]), self.qpos, self.clock.ns + 1_000_000_000)
        self.controller.wait_stationary = Mock(side_effect=TimeoutError("standstill timed out"))
        self.controller.operation(request)
        with self.assertRaisesRegex(TimeoutError, "standstill timed out"):
            self.client.request_operation(
                request, service=Mock(), abort=lambda: False, deadline_ns=request.deadline_ns
            )
        self.assertFalse(self.client.operation_unresolved)
        self.io.stop.clear()
        self.io.fault.value = b""
        self.driver.stop.side_effect = OSError("stop failed")
        self.controller.operation(request)
        with self.assertRaisesRegex(RobotError, "standstill timed out.*stop failed"):
            self.client.request_operation(
                request, service=Mock(), abort=lambda: False, deadline_ns=request.deadline_ns
            )

    def test_empty_home_confirms_measured_dwell_and_mode_before_preparation(self):
        self.driver.read.return_value = (self.qpos, np.zeros(7))
        self.driver.read_status.return_value = SimpleNamespace(state=0, error_code=0)
        self.io.health_until.value = self.clock.ns + 5_000_000_000
        with patch.object(self.io.stop, "wait", side_effect=self.clock.sleep):
            self.controller.operation(
                HomeRequest(1, 1, np.empty((0, 7)), self.qpos, self.clock.ns + 5_000_000_000)
            )
        self.assertIsNone(self.controller.pipe.send.call_args.args[0].error)
        self.assertEqual(self.controller.status["prepared_run_id"], 1)
        self.assertGreaterEqual(self.clock.ns, 1_900_000_000)
        self.driver.enter_mode.assert_called_once()
        self.driver.move_home_milestone.assert_not_called()

    def test_client_rejects_revoked_input_and_command_owns_array(self):
        self.client.process = Mock(is_alive=Mock(return_value=True))
        source = self.qpos.copy()
        command = RobotCommand(1, source, input_generation=3)
        source[:] = 0
        self.assertFalse(command.arm_qpos.flags.writeable)
        self.io.input_generation.value = 4
        with self.assertRaises(OperationCancelled):
            self.client.submit(
                command, submit_until_ns=2_000_000_000, execute_until_ns=2_000_000_000
            )
        self.assertIsNone(self.io.take_goal())
        self.io.input_generation.value = 3
        self.client.submit(command, submit_until_ns=2_000_000_000, execute_until_ns=2_000_000_000)
        np.testing.assert_array_equal(self.io.take_goal()["q_goal"], self.qpos)

    def test_sdk_return_codes_are_checked_by_driver(self):
        driver = XArm7(ArmConfig())
        driver.api = Mock()
        for code in (None, True, -1):
            with self.subTest(code=code):
                driver.api.set_servo_angle_j.return_value = code
                with self.assertRaises(RuntimeError):
                    driver.servo(self.qpos)
                driver.api.set_state.return_value = code
                with self.assertRaises(RuntimeError):
                    driver.stop()

    def test_full_slot_preserves_previous_command_and_sequence(self):
        self.client.process = Mock(is_alive=Mock(return_value=True))
        first = RobotCommand(1, self.qpos, input_generation=3)
        second = RobotCommand(1, self.qpos + 0.01, input_generation=3)
        deadlines = dict(submit_until_ns=2_000_000_000, execute_until_ns=2_000_000_000)
        self.client.submit(first, **deadlines)
        with self.assertRaisesRegex(RuntimeError, "slot is full"):
            self.client.submit(second, **deadlines)
        self.assertEqual(self.client.sequence, 1)
        np.testing.assert_array_equal(self.io.take_goal()["q_goal"], first.arm_qpos)

    def test_cancellation_while_waiting_for_publication_lock_leaves_slot_empty(self):
        self.client.process = Mock(is_alive=Mock(return_value=True))
        lock = Mock()
        lock.__enter__ = Mock(side_effect=self.io.stop.set)
        lock.__exit__ = Mock(return_value=False)
        with patch.object(self.io, "goal_lock", lock):
            with self.assertRaises(OperationCancelled):
                self.client.submit(
                    RobotCommand(1, self.qpos, input_generation=3),
                    submit_until_ns=2_000_000_000,
                    execute_until_ns=2_000_000_000,
                )
        self.assertEqual(self.client.sequence, 0)
        self.assertIsNone(self.io.take_goal())

    def test_encoding_failure_does_not_publish_or_advance_sequence(self):
        self.client.process = Mock(is_alive=Mock(return_value=True))
        with self.assertRaises(ValueError):
            self.client.submit(
                RobotCommand(1, np.zeros(8), input_generation=3),
                submit_until_ns=2_000_000_000,
                execute_until_ns=2_000_000_000,
            )
        self.assertEqual(self.client.sequence, 0)
        self.assertIsNone(self.io.take_goal())

    def test_cancelled_operation_is_drained_before_reuse(self):
        self.client.process = Mock(is_alive=Mock(return_value=True))
        self.client.pipe = Mock(poll=Mock(return_value=False))
        request = PrepareRequest(1, 1)
        with self.assertRaises(OperationCancelled):
            self.client.request_operation(
                request,
                service=Mock(),
                abort=Mock(side_effect=[False, True]),
                deadline_ns=self.clock.ns + 100_000_000,
            )
        self.assertTrue(self.client.operation_unresolved)
        self.assertTrue(self.io.stop.is_set())
        self.io.stopped.set()
        self.client.pipe.poll.return_value = True
        self.client.pipe.recv.return_value = LifecycleReply(1, OperationCancelled("cancelled"))
        self.client.wait_stopped()
        self.assertFalse(self.client.operation_unresolved)
        self.client.pipe.recv.assert_called_once()

    def test_late_completion_cannot_report_success(self):
        self.client.process = Mock(is_alive=Mock(return_value=True))
        self.client.pipe = Mock(poll=Mock(return_value=True))

        def receive():
            self.clock.sleep(0.2)
            return LifecycleReply(1)

        self.client.pipe.recv.side_effect = receive
        with self.assertRaisesRegex(TimeoutError, "after deadline"):
            self.client.request_operation(
                PrepareRequest(1, 1),
                service=Mock(),
                abort=lambda: False,
                deadline_ns=self.clock.ns + 100_000_000,
            )
        self.assertFalse(self.client.operation_unresolved)
        self.assertTrue(self.io.stop.is_set())


class TrajectoryTests(unittest.TestCase):
    def test_retargeting_preserves_position_velocity_and_duplicate_trajectory(self):
        cfg = ArmConfig()
        q = np.asarray(cfg.home_qpos)
        reference = ArmTrajectory(cfg, q, np.zeros(7), 1_000_000_000)
        reference.step(1_000_000_000, q + 0.1)
        now = 1_050_000_000
        position = reference.step(now).copy()
        velocity = reference.v.copy()
        updated_position = reference.step(now, q - 0.1)
        np.testing.assert_allclose(updated_position, position)
        np.testing.assert_allclose(reference.v, velocity)
        trajectory = reference.trajectory
        reference.step(now + 10_000_000, q - 0.1)
        self.assertIs(reference.trajectory, trajectory)

    def test_braking_extrema_cannot_leave_joint_limits(self):
        cfg = ArmConfig()
        q = np.asarray(cfg.home_qpos)
        q[0] = cfg.joint_limit_upper[0] - 0.001
        velocity = np.zeros(7)
        velocity[0] = 1
        with self.assertRaisesRegex(ValueError, "extremum"):
            ArmTrajectory(cfg, q, velocity, 1_000_000_000)

    def test_retargeting_matches_a_single_sample_of_previous_trajectory(self):
        cfg = ArmConfig()
        q = np.asarray(cfg.home_qpos)
        reference = ArmTrajectory(cfg, q, np.zeros(7), 1_000_000_000)
        reference.step(1_000_000_000, q + 0.1)
        previous = reference.trajectory
        expected_q, expected_v, _ = previous.at_time(0.05)
        sampled = Mock(wraps=previous)
        reference.trajectory = sampled
        actual = reference.step(1_050_000_000, q - 0.1)
        sampled.at_time.assert_called_once_with(0.05)
        np.testing.assert_allclose(actual, expected_q)
        np.testing.assert_allclose(reference.v, expected_v)


def sdk_error(code=0):
    return SimpleNamespace(error_code=code, error_message=f"error {code}")


def raw_hand_state():
    force = SimpleNamespace(fx=1.0, fy=2.0, fz=3.0)
    return SimpleNamespace(
        finger_state=[
            SimpleNamespace(
                id=i,
                position=0.1,
                torque=0.0,
                commboard_err=0,
                jonitboard_err=0,
                tipboard_err=0,
            )
            for i in range(12)
        ],
        sensor_data=[SimpleNamespace(calc_force=force, raw_force=[force] * 120) for _ in range(5)],
    )


class HandTests(unittest.TestCase):
    def make_hand(self, **overrides):
        hand = hand_driver.XHand(replace(HandConfig(), **overrides))
        hand.control = Mock()
        return hand

    def test_both_transports_open_once_and_support_discovery(self):
        for transport, device in (("serial", "/dev/ttyUSB0"), ("ethercat", "eth0")):
            with self.subTest(transport=transport):
                hand = self.make_hand(comm_type=transport, device_name=None)
                control = hand.control
                hand.sdk = SimpleNamespace(XHandControl=Mock(return_value=control))
                control.enumerate_devices.return_value = [device]
                control.open_serial.return_value = control.open_ethercat.return_value = sdk_error()
                hand.open_device()
                hand.sdk.XHandControl.assert_called_once()
                if transport == "serial":
                    control.open_serial.assert_called_once_with(device, hand.cfg.baudrate)
                    control.open_ethercat.assert_not_called()
                else:
                    control.open_ethercat.assert_called_once_with(device)
                    control.open_serial.assert_not_called()

    def test_connection_failure_closes_handle_without_retry_or_masking_primary_error(self):
        hand = self.make_hand()
        control = hand.control
        sdk = SimpleNamespace(XHandControl=Mock(return_value=control))
        control.open_serial.return_value = sdk_error(42)
        control.close_device.side_effect = OSError("close failed")
        with patch.dict(sys.modules, {"xhand_controller": SimpleNamespace(xhand_control=sdk)}):
            with self.assertLogs(hand_driver.logger, level="ERROR"):
                with self.assertRaises(hand_driver.XHandError) as raised:
                    hand.connect()
        self.assertEqual(raised.exception.code, 42)
        control.open_serial.assert_called_once()
        control.close_device.assert_called_once()
        self.assertIs(hand.control, control)

    def test_crc_payload_is_ignored_but_malformed_success_payload_raises(self):
        hand = self.make_hand()
        hand.connected_flag = True
        hand.control.read_state.return_value = (
            sdk_error(hand_driver.COMMUNICATION_CRC_ERROR_CODE),
            object(),
        )
        self.assertIsNone(hand.get_state())
        raw = raw_hand_state()
        raw.sensor_data.pop()
        hand.control.read_state.return_value = (sdk_error(), raw)
        with self.assertRaisesRegex(ValueError, "aggregate tactile"):
            hand.get_state()
        raw = raw_hand_state()
        raw.finger_state[0].jonitboard_err = 1
        hand.control.read_state.return_value = (sdk_error(), raw)
        with self.assertRaisesRegex(hand_driver.XHandError, "board fault"):
            hand.get_state()

    def test_protocol_status_masks_tactile_without_hiding_joint_feedback(self):
        for transport in ("serial", "ethercat"):
            hand = self.make_hand(comm_type=transport)
            hand.connected_flag = True
            hand.control.read_state.return_value = (sdk_error(1_501_019), raw_hand_state())
            state = hand.get_state()
            self.assertTrue(np.isfinite(state.qpos).all())
            self.assertTrue(np.isnan(state.tactile_dense).all())
            self.assertEqual(np.isfinite(state.tactile_aggregate).all(), transport == "serial")

    def test_tip_diagnostics_report_changes_and_keep_per_finger_masking(self):
        hand = self.make_hand()
        hand.connected_flag = True
        raw = raw_hand_state()
        finger_id = hand_driver.XHAND_TACTILE_SENSOR_FINGER_IDS[0]
        raw.finger_state[finger_id].tipboard_err = 201
        hand.control.read_state.return_value = (sdk_error(), raw)
        with self.assertLogs(hand_driver.logger, level="WARNING") as logs:
            state = hand.get_state()
            hand.get_state()
            raw.finger_state[finger_id].tipboard_err = 0
            hand.get_state()
            raw.finger_state[finger_id].tipboard_err = 201
            hand.get_state()
        self.assertEqual(len(logs.output), 2)
        self.assertIn(f"tip[{finger_id}]=201", logs.output[0])
        self.assertTrue(np.isnan(state.tactile_dense[0]).all())
        self.assertTrue(np.isfinite(state.tactile_dense[1:]).all())
        self.assertTrue(np.isfinite(state.tactile_aggregate).all())

    def test_float32_overflow_holds_previous_tactile_vector_and_source_time(self):
        hand = self.make_hand()
        hand.connected_flag = True
        raw = raw_hand_state()
        hand.control.read_state.return_value = (sdk_error(), raw)
        feedback = XHandFeedback(hand.cfg)
        start = 1_000_000_000
        feedback.reset(start)
        first = feedback.consume(hand.get_state(), source_ns=start, now_ns=start)
        raw.sensor_data[0].raw_force[0] = SimpleNamespace(fx=1e50, fy=2.0, fz=3.0)
        state = hand.get_state()
        self.assertEqual(state.tactile_dense.dtype, np.float32)
        self.assertEqual(state.tactile_aggregate.dtype, np.float32)
        with self.assertLogs("dexmani_real.robot.xhand.feedback"):
            frame = feedback.consume(state, source_ns=start + 20_000_000, now_ns=start + 20_000_000)
        np.testing.assert_array_equal(frame["tactile_dense"], first["tactile_dense"])
        self.assertEqual(frame["tactile_dense_timestamp_ns"][0], start)
        self.assertEqual(frame["timestamp_ns"][0], start + 20_000_000)

    def test_initialization_and_partial_reset_use_operation_cancelled(self):
        hand = self.make_hand()
        hand.connected_flag = True
        with self.assertRaises(OperationCancelled):
            hand.initialize_command(cancel_requested=lambda: True)
        hand.control.read_state.assert_not_called()
        hand.control.reset_sensor.return_value = sdk_error()
        with self.assertRaises(OperationCancelled):
            hand.reset_tactile_sensors(cancel_requested=Mock(side_effect=[False, True]))
        hand.control.reset_sensor.assert_called_once()
        self.assertEqual(hand.get_tactile_reset_metadata()["hand_tactile_reset_result"], "cancelled (1/5)")

    def test_diagnostic_cancellation_prevents_sdk_read(self):
        hand = XHandControlExample(HandConfig(), cancel_requested=lambda: True)
        hand.get_state = Mock()
        with self.assertRaises(OperationCancelled):
            hand.read_action_positions()
        hand.get_state.assert_not_called()

    def test_stop_retries_crc_only_until_deadline(self):
        hand = self.make_hand()
        hand.connected_flag = True
        hand.set_passive = Mock(return_value=hand_driver.COMMUNICATION_CRC_ERROR_CODE)
        clock = Clock()
        attempts = []
        hand.set_passive.side_effect = lambda: (
            attempts.append(clock.ns) or hand_driver.COMMUNICATION_CRC_ERROR_CODE
        )
        with (
            patch.object(hand_driver.time, "monotonic", clock.monotonic),
            patch.object(hand_driver.time, "monotonic_ns", clock.monotonic_ns),
            patch.object(hand_driver.time, "sleep", clock.sleep),
        ):
            with self.assertLogs(hand_driver.logger, level="WARNING"):
                with self.assertRaisesRegex(RuntimeError, "timed out"):
                    hand.stop()
        self.assertEqual(clock.ns, 1_250_000_000)
        self.assertGreater(hand.set_passive.call_count, 1)
        self.assertLess(max(attempts), 1_250_000_000)
        hand.set_passive.side_effect = [hand_driver.COMMUNICATION_CRC_ERROR_CODE, 0]
        with patch.object(hand_driver.time, "sleep"), self.assertLogs(hand_driver.logger):
            hand.stop()

    def test_ordinary_send_does_not_retry_crc_and_expires_continuous_failures(self):
        hand = self.make_hand()
        hand.send_position = Mock(return_value=hand_driver.COMMUNICATION_CRC_ERROR_CODE)
        clock = Clock()
        with patch.object(hand_driver.time, "monotonic_ns", clock.monotonic_ns):
            with self.assertLogs(hand_driver.logger):
                hand.send_action(np.zeros(12))
            hand.send_position.assert_called_once()
            clock.sleep(0.2)
            with self.assertRaisesRegex(hand_driver.XHandError, "age budget"):
                hand.send_action(np.zeros(12))
        self.assertEqual(hand.send_position.call_count, 2)

    def test_reset_exception_propagates_with_partial_metadata(self):
        hand = self.make_hand()
        hand.connected_flag = True
        hand.control.reset_sensor.side_effect = [sdk_error(), OSError("reset failed")]
        with self.assertRaisesRegex(OSError, "reset failed"):
            hand.reset_tactile_sensors()
        self.assertIn(
            "exception (1/5)", hand.get_tactile_reset_metadata()["hand_tactile_reset_result"]
        )

    def test_held_feedback_preserves_original_joint_and_tactile_times(self):
        feedback = XHandFeedback(HandConfig())
        start = 1_000_000_000
        feedback.reset(start)
        state = XHandState(
            np.zeros(12),
            np.zeros(12),
            np.ones((5, 3), dtype=np.float32),
            np.ones((5, 120, 3), dtype=np.float32),
            np.zeros(12),
            np.zeros(12),
            np.zeros(12),
        )
        first = feedback.consume(state, source_ns=start, now_ns=start)
        state.tactile_dense[0, 0, 0] = np.nan
        with self.assertLogs("dexmani_real.robot.xhand.feedback"):
            frame = feedback.consume(state, source_ns=start + 20_000_000, now_ns=start + 20_000_000)
            held = feedback.consume(None, source_ns=start + 40_000_000, now_ns=start + 40_000_000)
        self.assertIs(held, frame)
        self.assertEqual(frame["timestamp_ns"][0], start + 20_000_000)
        self.assertEqual(frame["tactile_dense_timestamp_ns"][0], start)
        np.testing.assert_array_equal(first["tactile_dense"], frame["tactile_dense"])
        self.assertFalse(frame.flags.writeable)
        with self.assertRaises(RuntimeError):
            feedback.consume(None, source_ns=start + 200_000_000, now_ns=start + 200_000_000)


class RobotBoundaryTests(unittest.TestCase):
    def setUp(self):
        client = make_arm_process()
        self.addCleanup(client.pipe.close)
        self.addCleanup(client.child_pipe.close)
        self.robot = Robot(
            ArmConfig(),
            HandConfig(),
            run_signal=client.io.run_id,
            stop_signal=client.io.stop,
            emergency_signal=client.io.emergency,
            signal_lock=client.io.signal_lock,
        )
        self.robot.owner = threading.get_ident()
        self.robot.connected = True
        self.robot.arm = Mock(is_connected=True)
        self.robot.hand = Mock(is_connected=True)

    def test_both_targets_are_validated_before_any_device_submission(self):
        command = RobotCommand(1, np.asarray(ArmConfig().home_qpos), np.full(12, np.nan))
        with self.assertRaises(ValueError):
            self.robot.send_action(command, valid_until_ns=10**20, execute_until_ns=10**20)
        self.robot.arm.submit.assert_not_called()
        self.robot.hand.send_action.assert_not_called()

    def test_hand_home_checks_io_owner_before_stopping_arm(self):
        self.robot.owner = -1
        with self.assertRaisesRegex(RuntimeError, "connecting thread"):
            self.robot.send_hand_home(
                RobotCommand(1, hand_qpos=np.zeros(12)), valid_until_ns=10**20
            )
        self.robot.arm.begin_operation.assert_not_called()
        self.robot.hand.send_action.assert_not_called()

    def test_one_device_stop_failure_does_not_skip_other_cleanup(self):
        arm, hand = self.robot.arm, self.robot.hand
        hand.stop.side_effect = OSError("hand stop failed")
        with self.assertRaises(RobotError):
            self.robot.close()
        arm.wait_stopped.assert_called_once()
        arm.close.assert_called_once()
        hand.disconnect.assert_called_once()
        self.assertTrue(self.robot.stop_required)

    def test_arm_close_failure_keeps_handle_and_still_disconnects_hand(self):
        arm, hand = self.robot.arm, self.robot.hand
        arm.close.side_effect = OSError("arm close failed")
        with self.assertRaisesRegex(RobotError, "arm close failed"):
            self.robot.close()
        self.assertIs(self.robot.arm, arm)
        self.assertIsNone(self.robot.hand)
        hand.stop.assert_called()
        hand.disconnect.assert_called_once()
        self.assertTrue(self.robot.stop_required)

    def test_initial_feedback_cancellation_does_not_query_devices(self):
        with self.assertRaises(OperationCancelled):
            self.robot.read_initial_state(cancel_requested=lambda: True)
        self.robot.arm.snapshot.assert_not_called()
        self.robot.hand.get_state.assert_not_called()

    def test_environment_table_geometry_configuration(self):
        changes = {"table_z_surface_m": 0.03, "hand_safety_margin_m": 0.06}
        cfg = load_experiment_config(cli_overrides={f"environment.{k}": v for k, v in changes.items()})
        validate_robot_config(cfg)
        for name, value in changes.items():
            self.assertEqual(getattr(cfg.environment, name), value)
        for name, value in (("table_z_surface_m", float("nan")), ("hand_safety_margin_m", -0.01)):
            with self.assertRaises(ValueError):
                validate_robot_config(load_experiment_config(cli_overrides={f"environment.{name}": value}))

    def test_unknown_configuration_keys_are_rejected(self):
        for section in ("arm", "arm.homing", "hand", "motion", "environment"):
            field = f"{section}.unsupported_option"
            with self.subTest(field=field), self.assertRaises(TypeError):
                load_experiment_config(cli_overrides={field: 1})

    def test_invalid_geometry_and_home_config_fail_before_device_startup(self):
        for changes in (
            {"environment.workspace.x_min": 1.0},
            {"environment.eef_workspace_margin_m": float("nan")},
            {"arm.homing.convergence_rad": 0.0},
            {"arm.homing.velocity_convergence_rad_s": -0.1},
            {"arm.homing.max_speed_deg_s": 181.0},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_robot_config(load_experiment_config(cli_overrides=changes))


if __name__ == "__main__":
    unittest.main()
