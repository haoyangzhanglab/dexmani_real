"""使用实际控制状态和 IPC，模拟设备副作用。"""

import multiprocessing as mp
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.config.real import RealConfig
from dexmani_real.control.actions import ActionResult
from dexmani_real.robot.commands import OperationCancelled, RobotCommand, RobotError
from dexmani_real.robot.xarm7.process import ArmProcess
from dexmani_real.robot.xarm7.protocol import ArmChannels, PrepareRequest
from dexmani_real.robot.xarm7.worker import ArmController
from dexmani_real.runtime.state import SafetyState, SessionState
from dexmani_real.teleop.keyboard import KeyboardTeleop
from dexmani_real.teleop.loop import TeleopLoop


class KeyboardLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.state = SessionState()
        self.epoch = self.state.arm()
        self.teleop = KeyboardTeleop(RealConfig())
        self.teleop.runtime_state = self.state
        self.teleop.robot = Mock()
        self.teleop.jog = Mock()
        self.teleop.jog_config = self.teleop.runtime.keyboard_teleop
        self.teleop.robot.prepare_streaming.side_effect = lambda _: self.state.stop_signal.clear()

    def test_preparation_enters_running_and_publishes_hold_without_keys(self):
        self.teleop.prepare_jog(self.epoch)
        self.assertEqual(self.state.safety_state, SafetyState.RUNNING)
        command = self.teleop.robot.send_action.call_args.args[0]
        self.assertEqual(command.run_id, self.epoch)
        self.assertTrue(command.arm_hold)
        self.assertIsNone(command.arm_qpos)

    def test_quit_during_preparation_cannot_publish_hold(self):
        def cancelled(_):
            self.state.stop_signal.clear()
            self.state.request_quit()
        self.teleop.robot.prepare_streaming.side_effect = cancelled
        with self.assertRaises(OperationCancelled):
            self.teleop.prepare_jog(self.epoch)
        self.teleop.robot.send_action.assert_not_called()

    def test_home_cancellation_reaches_session_boundary_without_repreparing(self):
        with patch("dexmani_real.teleop.keyboard.home_robot", side_effect=OperationCancelled("cancelled")):
            with self.assertRaises(OperationCancelled):
                self.teleop.run_home(self.epoch)
        self.teleop.robot.prepare_streaming.assert_not_called()


class VRSubmissionTests(unittest.TestCase):
    def test_first_submission_is_logged_after_skipped_first_tick(self):
        loop = TeleopLoop.__new__(TeleopLoop)
        loop.runtime_state = SessionState()
        loop.run_id = loop.runtime_state.arm()
        loop.runtime_state.stop_signal.clear()
        loop.runtime_state.begin_motion()
        loop.runtime = RealConfig()
        loop.robot = Mock()
        loop.controller = Mock(hand_retargeter=None)
        loop.controller.compute_target.return_value = ActionResult(np.zeros(7), None)
        loop.recorder = None
        loop.first_target_pending = True
        loop.soft_timeout_count = 0
        loop.tick_index = 1
        loop.execute_control_step(Mock(), 0)
        loop.robot.send_action.assert_not_called()
        self.assertTrue(loop.first_target_pending)
        with self.assertLogs("dexmani_real.teleop.loop", level="INFO") as captured:
            for index in (2, 3):
                loop.tick_index = index
                loop.execute_control_step(Mock(), time.monotonic_ns() + 1_000_000_000)
        self.assertEqual(loop.robot.send_action.call_count, 2)
        self.assertEqual(sum("first target submitted" in line for line in captured.output), 1)


class ArmTargetTests(unittest.TestCase):
    def setUp(self):
        self.config = RealConfig().arm
        self.state = SessionState()
        self.io = ArmChannels(mp.get_context("spawn"), run_signal=self.state.run_signal,
                              stop_signal=self.state.stop_signal, emergency_signal=self.state.emergency_signal)
        self.driver = Mock()
        self.qpos = np.asarray(self.config.home_qpos)
        self.driver.read.return_value = self.qpos, np.zeros(7)
        self.controller = ArmController(self.config, self.io, Mock(poll=Mock(return_value=False)), self.driver)
        self.controller.prepared_run_id = self.state.run_id
        self.controller.read_feedback()

    def test_new_target_supersedes_expired_active_before_watchdog(self):
        command = RobotCommand(self.state.run_id, arm_qpos=self.qpos)
        self.io.publish_goal(command, expires_ns=time.monotonic_ns() + 1_000_000_000)
        self.controller.active = self.io.take_goal()
        self.controller.active["expires_ns"] = 1
        target = self.qpos.copy()
        target[0] += 0.01
        self.io.publish_goal(RobotCommand(self.state.run_id, arm_qpos=target),
                             expires_ns=time.monotonic_ns() + 1_000_000_000)
        self.controller.wait_stationary = Mock()
        self.driver.servo.side_effect = lambda _: self.io.closing.set()
        self.driver.stop.side_effect = self.io.closing.set
        self.controller.run()
        self.driver.servo.assert_called_once()
        self.assertGreater(self.driver.servo.call_args.args[0][0], self.qpos[0])
        self.assertFalse(self.io.fault.value)

    def test_revocation_during_trajectory_computation_prevents_send(self):
        self.io.publish_goal(RobotCommand(self.state.run_id, arm_hold=True),
                             expires_ns=time.monotonic_ns() + 1_000_000_000)
        def revoked(*args, **kwargs):
            self.state.revoke_motion()
            return self.qpos
        self.controller.reference = Mock(step=Mock(side_effect=revoked))
        self.controller.tick()
        self.driver.servo.assert_not_called()
        self.assertTrue(self.io.stop.is_set())


class ArmOperationTests(unittest.TestCase):
    def setUp(self):
        self.state = SessionState()
        self.owner = ArmProcess.__new__(ArmProcess)
        self.owner.io = ArmChannels(
            mp.get_context("spawn"), run_signal=self.state.run_signal,
            stop_signal=self.state.stop_signal, emergency_signal=self.state.emergency_signal,
        )
        self.owner.io.connected.set()
        self.owner.process = Mock(is_alive=Mock(return_value=True))
        self.owner.pipe = Mock(poll=Mock(return_value=True), recv=Mock(return_value=None))
        self.owner.wait_stopped = Mock()
        self.request = PrepareRequest(self.state.run_id, time.monotonic_ns() + 1_000_000_000)

    def test_successful_request_consumes_one_reply(self):
        self.owner.request_operation(self.request, service=Mock(), abort=lambda: False)
        self.owner.pipe.send.assert_called_once_with(self.request)
        self.owner.pipe.recv.assert_called_once()
        self.assertFalse(self.state.stop_signal.is_set())

    def test_cancellation_before_send_does_not_consume_reply(self):
        with self.assertRaises(OperationCancelled):
            self.owner.request_operation(
                self.request, service=Mock(side_effect=OperationCancelled("cancelled")), abort=lambda: False,
            )
        self.owner.pipe.send.assert_not_called()
        self.owner.pipe.poll.assert_not_called()
        self.owner.pipe.recv.assert_not_called()
        self.assertTrue(self.state.stop_signal.is_set())

    def test_cancelled_request_reaps_its_timeout_reply(self):
        self.owner.pipe.poll.side_effect = [False, True]
        self.owner.pipe.recv.return_value = TimeoutError("owner deadline expired")
        with self.assertRaises(OperationCancelled) as captured:
            self.owner.request_operation(self.request, service=Mock(), abort=Mock(side_effect=[False, True]))
        self.owner.pipe.recv.assert_called_once()
        self.assertFalse(self.owner.io.closing.is_set())
        self.assertFalse(getattr(captured.exception, "cleanup_errors", ()))

    def test_sdk_timeout_reply_keeps_controller_fault(self):
        failure = TimeoutError("SDK timeout")
        def reply():
            self.owner.io.fault.value = b"SDK timeout"
            return failure
        self.owner.pipe.recv.side_effect = reply
        with self.assertRaisesRegex(RobotError, "arm operation failed") as captured:
            self.owner.request_operation(self.request, service=Mock(), abort=lambda: False)
        self.assertIs(captured.exception.__cause__, failure)
        self.owner.pipe.recv.assert_called_once()

    def test_expired_request_does_not_reauthorize_or_send(self):
        request = PrepareRequest(self.state.run_id, time.monotonic_ns() - 1)
        with self.assertRaisesRegex(TimeoutError, "deadline expired"):
            self.owner.request_operation(request, service=Mock(), abort=lambda: False)
        self.owner.pipe.send.assert_not_called()
        self.assertTrue(self.state.stop_signal.is_set())


if __name__ == "__main__":
    unittest.main()
