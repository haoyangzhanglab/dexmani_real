"""设备所有权和停止交错；SDK 全部为 fake，线程顺序由 Event 控制。"""

import threading
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.config.experiment import ExperimentConfig
from dexmani_real.robot.commands import OperationCancelled, RobotCommand, RobotError
from dexmani_real.robot.robot import DexManiRobot
from dexmani_real.robot.xarm7.executor import ArmExecutor
from dexmani_real.robot.xarm7.protocol import GOAL_DTYPE, OperationStatus, PrepareRequest
from dexmani_real.robot.xarm7.worker import ArmWorker
from dexmani_real.robot.xhand.driver import COMMUNICATION_CRC_ERROR_CODE, XHand, XHandError
from dexmani_real.runtime.state import RuntimeState


class SubmissionTests(unittest.TestCase):
    def robot(self):
        runtime, state = ExperimentConfig(), RuntimeState()
        robot = DexManiRobot(
            runtime, run_signal=state.run_signal, stop_signal=state.stop_signal,
            emergency_signal=state.emergency_signal, signal_lock=state.motion_lock,
        )
        robot.owner = threading.get_ident()
        robot.connected = True
        robot.arm = SimpleNamespace(is_connected=True, submit=Mock(), check_error=Mock(), request_stop=Mock(), wait_stopped=Mock())
        robot.hand = SimpleNamespace(is_connected=True, send_action=Mock(), end_send_interval=Mock(), stop=Mock())
        command = RobotCommand(state.run_id, np.asarray(runtime.arm.home_qpos), np.deg2rad(runtime.hand.home_qpos_deg))
        return robot, state, command

    def test_both_targets_checked_before_any_side_effect(self):
        robot, state, command = self.robot()
        bad = RobotCommand(state.run_id, command.arm_qpos, np.full(12, np.nan))
        with self.assertRaises(ValueError):
            robot.send_action(bad, valid_until_ns=100, execute_until_ns=100)
        robot.arm.submit.assert_not_called()
        robot.hand.send_action.assert_not_called()

    def test_stop_between_arm_and_hand_blocks_later_call(self):
        robot, state, command = self.robot()
        robot.arm.submit.side_effect = lambda *args, **kwargs: state.request_stop()
        with patch("dexmani_real.robot.robot.time.monotonic_ns", return_value=10):
            with self.assertRaises(OperationCancelled):
                robot.send_action(command, valid_until_ns=100, execute_until_ns=100)
        robot.arm.submit.assert_called_once()
        robot.hand.send_action.assert_not_called()
        self.assertTrue(state.stop_signal.is_set())

    def test_normal_return_is_none_and_hand_failure_is_not_retried(self):
        robot, state, command = self.robot()
        with patch("dexmani_real.robot.robot.time.monotonic_ns", return_value=10):
            self.assertIsNone(robot.send_action(command, valid_until_ns=100, execute_until_ns=100))
            robot.hand.send_action.reset_mock()
            robot.hand.send_action.side_effect = OSError("partial send")
            with self.assertRaisesRegex(RobotError, "partial send"):
                robot.send_action(command, valid_until_ns=100, execute_until_ns=100)
        robot.hand.send_action.assert_called_once()

    def test_hand_stop_failure_latches_even_after_cleanup_retry(self):
        robot, state, command = self.robot()
        robot.hand.stop.side_effect = [OSError("stop failed"), None]
        with self.assertRaises(RobotError):
            robot.stop()
        robot.stop()
        with self.assertRaisesRegex(RobotError, "unresolved"):
            robot.prepare_streaming(state.run_id, service=Mock())
        with self.assertRaises(RobotError):
            robot.send_action(command, valid_until_ns=100, execute_until_ns=100)

    def test_arm_stop_failure_without_hand_keeps_the_original_device_error(self):
        robot, state, _ = self.robot()
        robot.hand = None
        robot.arm.wait_stopped.side_effect = [OSError("arm stop failed"), None]
        with self.assertRaisesRegex(RobotError, "arm stop failed"):
            robot.stop()
        robot.stop()
        self.assertFalse(robot.hand_stop_pending)
        self.assertEqual(str(robot.stop_failure), "arm stop failed")
        with self.assertRaisesRegex(RobotError, "previous robot stop remains unresolved"):
            robot.prepare_streaming(state.run_id, service=Mock())

    def test_crc_budget_is_driver_local_and_stop_uses_its_own_retries(self):
        hand = XHand(ExperimentConfig().hand)
        hand.send_position = Mock(return_value=COMMUNICATION_CRC_ERROR_CODE)
        budget = int(hand.cfg.feedback_max_age_s * 1e9)
        with patch("dexmani_real.robot.xhand.driver.time.monotonic_ns", side_effect=[1, budget, budget + 1]):
            self.assertIsNone(hand.send_action(np.zeros(12)))
            hand.send_action(np.zeros(12))
            with self.assertRaisesRegex(XHandError, "continuous send CRC"):
                hand.send_action(np.zeros(12))
        self.assertEqual(hand.send_position.call_count, 3)
        hand.end_send_interval()
        self.assertIsNone(hand.send_crc_started_ns)
        hand.connected_flag = True
        hand.control = object()
        hand.set_passive = Mock(side_effect=[COMMUNICATION_CRC_ERROR_CODE, 0])
        with patch("dexmani_real.robot.xhand.driver.time.sleep"):
            hand.stop()
        self.assertEqual(hand.set_passive.call_count, 2)


class ArmLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.state = RuntimeState()
        self.cfg = replace(ExperimentConfig().arm, executor_timeout_s=.5)
        self.executor = ArmExecutor(
            self.cfg, run_signal=self.state.run_signal, stop_signal=self.state.stop_signal,
            emergency_signal=self.state.emergency_signal, signal_lock=self.state.motion_lock,
        )
        self.addCleanup(self.executor.pipe.close)
        self.addCleanup(self.executor.child_pipe.close)
        self.executor.process = SimpleNamespace(is_alive=lambda: True, exitcode=None)
        self.executor.io.connected.set()
        self.clock = [1_000_000_000]
        self.driver = SimpleNamespace(
            stop=Mock(return_value=0), enter_mode1=Mock(), mode=1,
            servo=Mock(return_value=0), is_connected=True,
        )
        self.worker = ArmWorker(self.cfg, self.executor.io, Mock(), self.driver, clock_ns=lambda: self.clock[0])
        self.worker.status["valid"] = True
        self.worker.status["frame"]["qpos"] = self.cfg.home_qpos
        self.worker.status["frame"]["qvel"] = 0
        self.worker.status["frame"]["timestamp_ns"] = self.clock[0]
        self.worker.status["prepared_run_id"] = self.state.run_id
        self.executor.io.publish(self.worker.status)
        self.executor.update_health(health_until_ns=self.clock[0] + 1_000_000)

    def goal(self):
        goal = np.zeros((), dtype=GOAL_DTYPE)
        goal[()] = (1, self.state.run_id, -1, self.clock[0] + 1000, self.clock[0] + 10000, self.cfg.home_qpos)
        return goal

    def test_health_and_execution_deadlines_cannot_be_extended_by_local_feedback(self):
        goal = self.goal()
        self.assertIsNone(self.worker.command_issue(goal, self.clock[0], first=True))
        self.assertEqual(self.worker.command_issue(goal, self.clock[0] + 1000, first=True), "submission_expired_before_sdk")
        self.assertEqual(self.worker.command_issue(goal, self.clock[0] + 10000), "input_decision_expired")
        self.executor.update_health(health_until_ns=self.clock[0] + 5)
        self.worker.status["frame"]["timestamp_ns"] = self.clock[0] + 6
        self.assertEqual(self.worker.command_issue(goal, self.clock[0] + 6), "upstream_feedback_or_service_expired")
        self.executor.update_health(health_until_ns=self.clock[0] + 100000)
        self.executor.limit_execution_until(self.clock[0] + 3)
        self.executor.io.execution_limit[1] = 1
        self.assertEqual(self.worker.command_issue(goal, self.clock[0] + 4), "input_decision_expired")

    def test_old_ipc_and_changed_jog_proposal_cannot_reach_sdk(self):
        goal = self.goal()
        self.state.revoke_motion()
        self.state.stop_signal.clear()
        self.assertEqual(self.worker.command_issue(goal, self.clock[0], first=True), "revoked")
        goal["run_id"] = self.state.run_id
        goal["input_generation"] = 4
        self.executor.io.input_generation.value = 5
        self.assertEqual(self.worker.command_issue(goal, self.clock[0], first=True), "input_changed")
        self.driver.servo.assert_not_called()

    def test_stop_and_begin_interleave_without_clearing_new_stop(self):
        entered, release, begin_entered = threading.Event(), threading.Event(), threading.Event()
        failures = []
        run_id = self.state.run_id

        def stop():
            entered.set()
            if not release.wait(2):
                raise TimeoutError("test gate")
            return 0

        def begin():
            begin_entered.set()
            try:
                self.executor.begin_operation(run_id)
            except Exception as exc:
                failures.append(exc)

        self.driver.stop.side_effect = stop
        self.state.stop_signal.set()
        stopping = threading.Thread(target=self.worker.stop)
        stopping.start()
        self.assertTrue(entered.wait(1))
        starting = threading.Thread(target=begin)
        starting.start()
        self.assertTrue(begin_entered.wait(1))
        self.assertFalse(self.executor.io.stopped.is_set())
        self.state.arm()
        self.state.request_stop()
        release.set()
        stopping.join(2)
        starting.join(2)
        self.assertFalse(stopping.is_alive() or starting.is_alive())
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], OperationCancelled)
        self.assertTrue(self.state.stop_signal.is_set())
        self.assertIsNone(self.worker.active)
        self.assertIsNone(self.worker.reference)

    def test_cancelled_preparation_never_publishes_prepared_identity(self):
        entered, release = threading.Event(), threading.Event()

        def stationary(abort):
            entered.set()
            if not release.wait(2):
                raise TimeoutError("test gate")
            abort()

        self.worker.wait_stationary = stationary
        operation = threading.Thread(target=self.worker.operation, args=(PrepareRequest(1, self.state.run_id),))
        operation.start()
        self.assertTrue(entered.wait(1))
        self.state.arm()
        self.state.request_stop()
        release.set()
        operation.join(2)
        self.assertFalse(operation.is_alive())
        self.assertEqual(self.worker.status["prepared_run_id"], 0)
        self.assertEqual(self.worker.pipe.send.call_args.args[0].status, OperationStatus.CANCELLED)
        self.driver.enter_mode1.assert_not_called()

    def test_prepare_and_home_use_health_without_streaming_lease(self):
        self.executor.io.execution_limit[:] = (0, 0)
        self.worker.wait_stationary = lambda abort: abort()
        self.worker.operation(PrepareRequest(1, self.state.run_id))
        self.assertEqual(self.worker.status["prepared_run_id"], self.state.run_id)
        self.assertEqual(self.worker.pipe.send.call_args.args[0].status, OperationStatus.COMPLETED)
        self.driver.stop.return_value = 1
        self.state.stop_signal.set()
        self.worker.stop()
        with self.assertRaises(RobotError):
            self.executor.begin_operation(self.state.run_id)
        self.assertTrue(self.state.stop_signal.is_set())

    def test_late_worker_fault_is_reported_and_does_not_renew_health(self):
        deadline = self.executor.io.health_until.value
        self.worker.fault("late servo failure")
        with self.assertRaisesRegex(RobotError, "late servo failure"):
            self.executor.check_error()
        self.worker.stop()
        self.assertIsNone(self.worker.reference)
        self.assertEqual(self.executor.io.health_until.value, deadline)

    def test_stopped_worker_keeps_reading_feedback_without_rearming(self):
        reads = []
        self.worker.wait_stationary = lambda abort: abort()
        self.driver.connect = Mock()
        self.driver.initialize = Mock(side_effect=lambda **kwargs: self.state.stop_signal.set())
        self.driver.close = Mock()
        self.driver.read_status = Mock(return_value=SimpleNamespace(error_code=0, state=0))

        def read():
            reads.append(self.clock[0])
            if len(reads) == 2:
                self.executor.io.closing.set()
            return np.asarray(self.cfg.home_qpos), np.zeros(7)

        def poll(timeout=0):
            self.clock[0] += 20_000_000
            return False

        self.driver.read = read
        self.worker.pipe.poll.side_effect = poll
        self.worker.run()
        self.assertEqual(len(reads), 2)
        self.assertEqual(self.executor.io.fault.value, b"")
        self.assertTrue(self.state.stop_signal.is_set())
        self.assertEqual(self.worker.status["prepared_run_id"], 0)
        self.driver.stop.assert_called_once()
        self.driver.servo.assert_not_called()
        self.driver.close.assert_called_once()

    def test_home_operation_ignores_old_streaming_lease_but_honors_health(self):
        from dexmani_real.robot.xarm7.protocol import HomeRequest

        self.executor.io.execution_limit[:] = (0, 0)
        self.worker.wait_stationary = lambda abort: abort()
        self.worker.home = Mock(side_effect=lambda *args, check_abort: check_abort())
        request = HomeRequest(2, self.state.run_id, np.array([self.cfg.home_qpos]), self.cfg.home_qpos)
        self.worker.operation(request)
        self.assertEqual(self.worker.pipe.send.call_args.args[0].status, OperationStatus.COMPLETED)
        self.executor.update_health(health_until_ns=self.clock[0])
        self.worker.operation(request)
        self.assertEqual(self.worker.pipe.send.call_args.args[0].status, OperationStatus.FAILED)
        self.assertIn(b"hand feedback or parent service expired", self.executor.io.fault.value)
        self.assertIsNone(self.worker.reference)
