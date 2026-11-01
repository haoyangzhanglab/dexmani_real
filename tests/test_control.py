"""使用实际控制状态和 IPC，模拟设备副作用。"""

import multiprocessing as mp
import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.config.real import RealConfig
from dexmani_real.config.xarm7 import SDK_TIMEOUT_S, ArmConfig
from dexmani_real.control.actions import ActionResult
from dexmani_real.robot.commands import OperationCancelled, RobotCommand, RobotError
from dexmani_real.robot.system import Robot
from dexmani_real.robot.xarm7.driver import XArm7
from dexmani_real.robot.xarm7.process import ArmProcess
from dexmani_real.robot.xarm7.protocol import ArmChannels, PrepareRequest
from dexmani_real.robot.xarm7.trajectory import ArmTrajectory
from dexmani_real.robot.xarm7.worker import ArmController
from dexmani_real.runtime.lifecycle import close_runtime
from dexmani_real.runtime.state import SafetyState, SessionState
from dexmani_real.teleop.keyboard import KeyboardTeleop
from dexmani_real.teleop.loop import TeleopLoop
from dexmani_real.utils.log import attach_cleanup_errors, error_detail


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
        loop.audio = Mock()
        loop.resuming = False
        loop.soft_timeout_count = 0
        loop.tick_index = 1
        loop.execute_control_step(Mock(), 0)
        loop.robot.send_action.assert_not_called()
        self.assertTrue(loop.first_target_pending)
        loop.audio.play.assert_not_called()
        with self.assertLogs("dexmani_real.teleop.loop", level="INFO") as captured:
            for index in (2, 3):
                loop.tick_index = index
                loop.execute_control_step(Mock(), time.monotonic_ns() + 1_000_000_000)
        self.assertEqual(loop.robot.send_action.call_count, 2)
        self.assertEqual(sum("first target submitted" in line for line in captured.output), 1)
        loop.audio.play.assert_called_once_with("begin", interrupt=False)


class ArmSdkTimeoutTests(unittest.TestCase):
    def setUp(self):
        self.driver = XArm7(RealConfig().arm)
        self.api = self.driver.api = Mock(mode=1, state=2)
        self.timeout = SDK_TIMEOUT_S
        self.api.set_timeout.side_effect = lambda value: setattr(self, "timeout", value)
        self.qpos = np.zeros(7)
        self.api.get_err_warn_code.return_value = (0, (0, 0))
        self.api.get_joint_states.return_value = (0, (self.qpos, np.zeros(7)))
        self.api.motion_enable.return_value = 0
        self.api.set_servo_angle_j.return_value = 0
        for name in ("set_state", "set_mode", "set_collision_sensitivity",
                     "set_tcp_load", "set_joint_maxacc", "set_servo_angle"):
            getattr(self.api, name).side_effect = self.slow_reply

    def slow_reply(self, *args, **kwargs):
        # 模拟 50 ms 回复，不使用真实等待或设备。
        return 0 if self.timeout >= 0.050 else 3

    def test_slow_lifecycle_commands_after_stream_calls(self):
        self.driver.read()  # 启动流程先读取反馈、确认停稳，再初始化。
        self.driver.initialize(check_abort=lambda: None)
        self.api.set_joint_maxacc.assert_called_once()
        self.driver.read()
        self.driver.servo(self.qpos)
        self.driver.enter_mode(0, check_abort=lambda: None)
        self.driver.move_home_milestone(self.qpos)
        self.driver.stop()

    def test_stream_timeout_does_not_shorten_stop_timeout(self):
        for operation in ("get_err_warn_code", "get_joint_states", "set_servo_angle_j"):
            with self.subTest(operation=operation):
                method = getattr(self.api, operation)
                payload = method.return_value[1] if operation != "set_servo_angle_j" else None
                slow_response = lambda *args, **kwargs: (
                    (self.slow_reply(), payload) if payload is not None else self.slow_reply()
                )
                with patch.object(self.api, operation, side_effect=slow_response) as call:
                    with self.assertRaisesRegex(RuntimeError, "SDK code 3"):
                        if operation == "set_servo_angle_j":
                            self.driver.servo(self.qpos)
                        else:
                            self.driver.read()
                    call.assert_called_once()  # 超时后不重发旧请求。
                self.driver.stop()

    def test_sdk_exception_does_not_shorten_stop_timeout(self):
        for operation in ("get_err_warn_code", "get_joint_states", "set_servo_angle_j"):
            with self.subTest(operation=operation):
                with patch.object(self.api, operation, side_effect=OSError("SDK transport failure")):
                    with self.assertRaisesRegex(OSError, "SDK transport failure"):
                        if operation == "set_servo_angle_j":
                            self.driver.servo(self.qpos)
                        else:
                            self.driver.read()
                self.driver.stop()


class ArmTrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.config = ArmConfig()
        # 真机同位目标失败时的原始位置与速度；测试使用实际 Ruckig。
        self.qpos = np.array([
            0, 0.0418815091252327, 0.001742985681630671, 0.4171277284622192,
            -3.138098955154419, 1.195550084114075, -5.752428023697576e-06,
        ])
        self.velocity_noise = np.array([
            5.886210886180241e-16, 2.46429857619359e-15, 5.486062992776652e-17,
            -6.423433062646808e-16, -2.093563478366934e-15,
            1.268826344108498e-15, -1.053906450940178e-15,
        ])
        self.velocity_noise.flags.writeable = False

    def test_stationary_feedback_noise_keeps_same_position(self):
        for jerk in (6000.0, 12000.0):
            with self.subTest(jerk=jerk):
                reference = ArmTrajectory(
                    ArmConfig(max_joint_jerk_deg_per_s3=jerk), self.qpos, self.velocity_noise,
                )
                for _ in range(3):
                    np.testing.assert_array_equal(reference.step(self.qpos), self.qpos)
                np.testing.assert_array_equal(reference.input.current_velocity, np.zeros(7))

    def test_small_real_velocity_is_preserved_and_braked(self):
        velocity = np.array([1e-4, -1e-4, 1e-8, -1e-8, 0, 1e-6, -1e-6])
        reference = ArmTrajectory(self.config, self.qpos, velocity)
        np.testing.assert_array_equal(reference.input.current_velocity, velocity)
        stopped = reference.step(hold=True)
        self.assertGreater(stopped[0], self.qpos[0])
        self.assertLess(stopped[1], self.qpos[1])
        np.testing.assert_allclose(reference.input.current_velocity, 0, rtol=0, atol=1e-12)

    def test_retargeting_preserves_moving_reference(self):
        reference = ArmTrajectory(self.config, self.qpos, self.velocity_noise)
        goal = self.qpos.copy()
        goal[0] += 0.2
        first = reference.step(goal)
        velocity = np.array(reference.input.current_velocity)
        acceleration = np.array(reference.input.current_acceleration)
        self.assertGreater(velocity[0], 0)
        goal[0] = self.qpos[0] - 0.1
        following = reference.step(goal)
        _, initial_velocity, initial_acceleration = reference.output.trajectory.at_time(0)
        np.testing.assert_allclose(initial_velocity, velocity, rtol=0, atol=1e-12)
        np.testing.assert_allclose(initial_acceleration, acceleration, rtol=0, atol=1e-12)
        self.assertGreater(following[0], first[0])


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

    def test_first_stop_failure_is_recorded_only_as_primary_fault(self):
        self.driver.stop.side_effect = RuntimeError("stop reply timeout")
        with self.assertLogs("dexmani_real.robot.xarm7.worker", level="ERROR"):
            self.controller.stop()
        self.assertIn(b"stop reply timeout", self.io.fault.value)
        self.assertFalse(self.io.close_error.value)
        self.assertTrue(self.io.stopped.is_set())

    def test_stop_failure_after_runtime_fault_is_an_independent_cleanup_error(self):
        self.io.fault.value = b"servo failed"
        self.driver.stop.side_effect = RuntimeError("stop reply timeout")
        with self.assertLogs("dexmani_real.robot.xarm7.worker", level="ERROR"):
            self.controller.stop()
        self.assertEqual(self.io.fault.value, b"servo failed")
        self.assertIn(b"stop reply timeout", self.io.close_error.value)
        self.assertTrue(self.io.stopped.is_set())


class ArmOperationTests(unittest.TestCase):
    def setUp(self):
        self.state = SessionState()
        self.owner = ArmProcess(
            RealConfig().arm, run_signal=self.state.run_signal,
            stop_signal=self.state.stop_signal, emergency_signal=self.state.emergency_signal,
        )
        self.addCleanup(self.owner.pipe.close)
        self.addCleanup(self.owner.child_pipe.close)
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
        with self.assertRaisesRegex(RobotError, "SDK timeout") as captured:
            self.owner.request_operation(self.request, service=Mock(), abort=lambda: False)
        self.assertIs(captured.exception.__cause__, failure)
        self.owner.pipe.recv.assert_called_once()
        with self.assertRaises(RobotError) as checked:
            self.owner.check_error()
        self.assertIs(checked.exception, captured.exception)

    def test_expired_request_does_not_reauthorize_or_send(self):
        request = PrepareRequest(self.state.run_id, time.monotonic_ns() - 1)
        with self.assertRaisesRegex(TimeoutError, "deadline expired"):
            self.owner.request_operation(request, service=Mock(), abort=lambda: False)
        self.owner.pipe.send.assert_not_called()
        self.assertTrue(self.state.stop_signal.is_set())


class ArmShutdownTests(unittest.TestCase):
    def setUp(self):
        self.config = RealConfig()
        self.state = SessionState()
        self.owner = ArmProcess(
            self.config.arm, run_signal=self.state.run_signal,
            stop_signal=self.state.stop_signal, emergency_signal=self.state.emergency_signal,
        )
        self.addCleanup(self.owner.pipe.close)
        self.addCleanup(self.owner.child_pipe.close)
        self.owner.process = Mock(pid=1, exitcode=0, is_alive=Mock(return_value=False))
        self.owner.io.stopped.set()

    def report_fault(self):
        self.owner.io.fault.value = b"RuntimeError: servo reply timeout"
        with self.assertRaises(RobotError) as captured:
            self.owner.check_error()
        return captured.exception

    def test_known_fault_is_stable_and_does_not_fail_successful_close(self):
        primary = self.report_fault()
        self.owner.close()
        self.assertTrue(self.owner.is_closed)
        self.assertTrue(self.owner.io.fault.value)
        with self.assertRaises(RobotError) as checked:
            self.owner.check_error()
        self.assertIs(checked.exception, primary)

    def test_fault_arriving_during_join_is_reported(self):
        self.owner.process.join.side_effect = lambda _: setattr(
            self.owner.io.fault, "value", b"late SDK timeout",
        )
        with self.assertRaisesRegex(RobotError, "late SDK timeout"):
            self.owner.close()
        self.assertTrue(self.owner.is_closed)

    def test_new_disconnect_error_is_reported_after_known_fault(self):
        primary = self.report_fault()
        self.owner.process.join.side_effect = lambda _: setattr(
            self.owner.io.close_error, "value", b"disconnect failure",
        )
        with self.assertRaisesRegex(RobotError, "disconnect failure") as captured:
            self.owner.close()
        self.assertIsNot(captured.exception, primary)
        self.assertTrue(self.owner.is_closed)

    def test_known_stop_failure_does_not_fail_close_again(self):
        self.report_fault()
        self.owner.io.close_error.value = b"stop SDK failure"
        with self.assertRaisesRegex(RobotError, "stop SDK failure"):
            self.owner.wait_stopped()
        self.owner.close()
        self.assertTrue(self.owner.is_closed)

    def test_disconnect_error_appended_after_reported_stop_is_not_lost(self):
        primary = self.report_fault()
        self.owner.io.close_error.value = b"stop SDK failure"
        with self.assertRaises(RobotError) as stopped:
            self.owner.wait_stopped()
        attach_cleanup_errors(primary, [("stop", stopped.exception)])
        self.owner.process.join.side_effect = lambda _: setattr(
            self.owner.io.close_error, "value", b"stop SDK failure; disconnect failure",
        )
        with self.assertRaisesRegex(RobotError, "disconnect failure") as closed:
            self.owner.close()
        attach_cleanup_errors(primary, [("close", closed.exception)])
        self.assertIs(closed.exception, stopped.exception)
        self.assertEqual(len(primary.cleanup_errors), 1)
        message = error_detail("session", primary)["message"]
        self.assertEqual(message.count("stop SDK failure"), 1)
        self.assertEqual(message.count("disconnect failure"), 1)

    def test_equal_messages_from_distinct_error_channels_are_preserved(self):
        primary = self.report_fault()
        self.owner.io.close_error.value = str(primary).encode()
        with self.assertRaises(RobotError) as captured:
            self.owner.close()
        self.assertEqual(str(captured.exception), str(primary))
        self.assertIsNot(captured.exception, primary)

    def test_abnormal_exit_after_known_sdk_fault_is_reported(self):
        self.report_fault()
        self.owner.process.exitcode = -9
        with self.assertRaisesRegex(RobotError, "owner exited: -9"):
            self.owner.close()

    def test_already_reported_abnormal_exit_is_not_reported_as_cleanup(self):
        self.owner.process.exitcode = -9
        with self.assertRaisesRegex(RobotError, "owner exited: -9"):
            self.owner.check_error()
        self.owner.close()

    def test_normal_exit_without_stop_confirmation_is_reported(self):
        self.owner.io.stopped.clear()
        with self.assertRaisesRegex(RobotError, "without confirming stop"):
            self.owner.close()

    def test_live_owner_keeps_pipe_handles_and_reports_unresolved_close(self):
        self.owner.process.is_alive.return_value = True
        with self.assertRaisesRegex(TimeoutError, "still active"):
            self.owner.close()
        self.assertFalse(self.owner.pipe.closed)
        self.assertFalse(self.owner.child_pipe.closed)

    def test_keyboard_startup_failure_has_no_spurious_cleanup_error(self):
        self.owner.io.fault.value = b"startup SDK timeout"
        robot = Robot(
            self.config.arm, self.config.hand, run_signal=self.state.run_signal,
            stop_signal=self.state.stop_signal, emergency_signal=self.state.emergency_signal,
        )
        robot.owner = threading.get_ident()
        robot.arm = self.owner
        teleop = KeyboardTeleop(self.config)
        keyboard = Mock()
        with (
            patch("dexmani_real.teleop.keyboard.Robot", return_value=robot),
            patch("dexmani_real.teleop.keyboard.Keyboard", return_value=keyboard),
            patch.object(teleop, "start", side_effect=self.owner.check_error),
            self.assertNoLogs("dexmani_real", level="ERROR"),
            self.assertRaisesRegex(RobotError, "startup SDK timeout") as captured,
        ):
            teleop.run()
        self.assertFalse(getattr(captured.exception, "cleanup_errors", ()))
        self.assertIsNone(robot.arm)
        keyboard.stop.assert_called_once()

    def test_runtime_cleanup_attempts_all_resources_and_leaves_reporting_to_caller(self):
        robot_error, keyboard_error = RuntimeError("robot close failed"), OSError("keyboard close failed")
        robot = Mock(close=Mock(side_effect=robot_error))
        keyboard = Mock(stop=Mock(side_effect=keyboard_error))
        sensors = Mock()
        with self.assertNoLogs("dexmani_real.runtime.lifecycle", level="ERROR"):
            with self.assertRaises(RuntimeError) as captured:
                close_runtime(self.state, robot, keyboard=keyboard, sensors=sensors)
        self.assertIs(captured.exception, robot_error)
        self.assertEqual(robot_error.cleanup_errors, [("keyboard_stop", keyboard_error)])
        sensors.shutdown.assert_called_once()

    def test_keyboard_preserves_primary_and_reports_new_cleanup_failures_once(self):
        self.owner.io.fault.value = b"startup SDK timeout"
        self.owner.process.join.side_effect = lambda _: setattr(
            self.owner.io.close_error, "value", b"stop SDK failure; disconnect failure",
        )
        robot = Robot(
            self.config.arm, self.config.hand, run_signal=self.state.run_signal,
            stop_signal=self.state.stop_signal, emergency_signal=self.state.emergency_signal,
        )
        robot.owner = threading.get_ident()
        robot.arm = self.owner
        teleop = KeyboardTeleop(self.config)
        keyboard = Mock(stop=Mock(side_effect=OSError("keyboard close failed")))
        with (
            patch("dexmani_real.teleop.keyboard.Robot", return_value=robot),
            patch("dexmani_real.teleop.keyboard.Keyboard", return_value=keyboard),
            patch.object(teleop, "start", side_effect=self.owner.check_error),
            self.assertLogs("dexmani_real", level="ERROR") as logs,
            self.assertRaisesRegex(RobotError, "startup SDK timeout") as captured,
        ):
            teleop.run()
        self.assertEqual(len(logs.records), 1)
        self.assertIsNone(logs.records[0].exc_info)
        self.assertNotIn("startup SDK timeout", logs.output[0])
        for detail in ("stop SDK failure", "disconnect failure", "keyboard close failed"):
            self.assertIn(detail, logs.output[0])
        self.assertEqual(len(captured.exception.cleanup_errors), 1)
        self.assertIsNone(robot.arm)
        keyboard.stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
