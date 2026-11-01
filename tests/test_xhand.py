"""模拟 SDK、设备发现和单调时钟，覆盖手部初始化、提交与停止边界。"""

import sys
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.config.deployment import ExecutionConfig
from dexmani_real.config.real import RealConfig
from dexmani_real.config.xhand import CONNECT_TIMEOUT_S, INITIAL_STATE_TIMEOUT_S
from dexmani_real.control.actions import ActionResult
from dexmani_real.deployment.rollout import Rollout
from dexmani_real.deployment.schedule import ActionChunk
from dexmani_real.robot.commands import OperationCancelled, RobotCommand, RobotError
from dexmani_real.robot.system import Robot
from dexmani_real.robot.xarm7 import process as arm_process
from dexmani_real.robot.xhand.driver import (
    COMMUNICATION_CRC_ERROR_CODE,
    DISTRIBUTED_FORCE_UNAVAILABLE_CODE,
    XHand,
    XHandError,
)
from dexmani_real.robot.xhand.feedback import XHandFeedback
from dexmani_real.runtime.home import home_hand, initialize_hand
from dexmani_real.runtime.observation import ObservationNeeds
from dexmani_real.runtime.state import RunEndReason, SessionState, device_failure_reason
from dexmani_real.teleop.loop import TeleopLoop
from scripts.tools.control_xhand import XHandControlExample


class Clock:
    def __init__(self):
        self.now_ns = 1_000_000_000
        self.sleep = Mock(side_effect=self.advance)

    def monotonic_ns(self):
        return self.now_ns

    def monotonic(self):
        return self.now_ns / 1e9

    def time_ns(self):
        return self.now_ns

    def advance(self, seconds):
        self.now_ns += max(1, round(seconds * 1e9))


def reply(code=0):
    return SimpleNamespace(error_code=code, error_message=f"SDK code {code}")


class XHandTests(unittest.TestCase):
    def setUp(self):
        self.runtime = RealConfig()
        self.qpos = np.deg2rad(self.runtime.hand.home_qpos_deg)
        force = SimpleNamespace(fx=0., fy=0., fz=0.)
        self.raw = SimpleNamespace(
            finger_state=[
                SimpleNamespace(id=i, position=q, commboard_err=0, jonitboard_err=0, tipboard_err=0)
                for i, q in enumerate(self.qpos)
            ],
            sensor_data=[SimpleNamespace(calc_force=force, raw_force=[force] * 120) for _ in range(5)],
        )
        self.control = Mock()
        self.control.open_serial.return_value = reply()
        self.control.list_hands_id.return_value = [0]
        self.control.read_state.return_value = reply(), self.raw
        self.control.send_command.return_value = reply()
        sdk = SimpleNamespace(
            XHandControl=Mock(return_value=self.control),
            HandCommand_t=lambda: SimpleNamespace(finger_command=[SimpleNamespace() for _ in range(12)]),
        )
        self.clock = Clock()
        original_sdk = sys.modules.get("xhand_controller")
        # 只恢复这一项，避免整个模块表回滚破坏懒加载模块的身份。
        def restore_sdk():
            if original_sdk is None:
                sys.modules.pop("xhand_controller", None)
            else:
                sys.modules["xhand_controller"] = original_sdk
        sys.modules["xhand_controller"] = SimpleNamespace(xhand_control=sdk)
        self.addCleanup(restore_sdk)
        patches = [patch("multiprocessing.process.BaseProcess.start",
                         side_effect=AssertionError("offline tests must not start device processes"))]
        patches.extend(patch(f"{module}.time", self.clock) for module in (
            "dexmani_real.robot.xhand.driver", "dexmani_real.robot.system",
            "dexmani_real.runtime.home", "dexmani_real.runtime.observation",
            "dexmani_real.teleop.loop", "dexmani_real.deployment.rollout",
            "scripts.tools.control_xhand",
        ))
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.hand = XHand(self.runtime.hand)

    def connect_robot(self):
        self.state = SessionState()
        self.arm = Mock(is_connected=True)
        def arm_snapshot():
            result = np.zeros(1, dtype=[("qpos", "<f8", (7,)), ("timestamp_ns", "<u8")])
            result["qpos"] = self.runtime.arm.home_qpos
            result["timestamp_ns"] = self.clock.now_ns
            return result
        self.arm.snapshot.side_effect = arm_snapshot
        self.arm.prepare.side_effect = lambda _, *, service: self.state.stop_signal.clear()
        self.robot = Robot(
            self.runtime.arm, self.runtime.hand, run_signal=self.state.run_signal,
            stop_signal=self.state.stop_signal, emergency_signal=self.state.emergency_signal,
        )
        with patch.object(arm_process, "ArmProcess", return_value=self.arm) as constructor:
            self.robot.connect()
        constructor.assert_called_once()
        return self.state.arm()

    def test_ready_device_needs_one_read_and_no_sleep_or_motion(self):
        state = self.hand.connect()
        self.control.read_state.assert_called_once_with(0, True)
        self.control.send_command.assert_not_called()
        self.clock.sleep.assert_not_called()
        self.assertEqual(state.timestamp_ns, self.clock.now_ns)
        np.testing.assert_array_equal(state.qpos, self.qpos)
        np.testing.assert_array_equal([j.position for j in self.hand.command.finger_command], self.qpos)

    def test_waits_for_configured_device_before_reading(self):
        self.control.list_hands_id.side_effect = [[], [2], [0]]
        self.hand.connect()
        self.control.read_state.assert_called_once_with(0, True)
        self.assertEqual(self.clock.sleep.call_count, 2)

    def test_discovery_and_crc_wait_share_one_deadline(self):
        start = self.clock.now_ns
        self.control.list_hands_id.side_effect = lambda: (
            [] if self.clock.now_ns < start + int((CONNECT_TIMEOUT_S - .02) * 1e9) else [0]
        )
        self.control.read_state.return_value = reply(COMMUNICATION_CRC_ERROR_CODE), self.raw
        with self.assertRaisesRegex(XHandError, "feedback timed out"):
            self.hand.connect()
        self.assertEqual(self.clock.now_ns - start, round(CONNECT_TIMEOUT_S * 1e9))
        self.control.read_state.assert_called_once()
        self.control.close_device.assert_called_once()
        self.assertIsNone(self.hand.command)

    def test_cancellation_before_open_or_after_read_never_sends(self):
        with self.assertRaises(OperationCancelled):
            self.hand.connect(cancel_requested=lambda: True)
        self.control.open_serial.assert_not_called()
        with self.assertRaises(OperationCancelled):
            self.hand.connect(cancel_requested=lambda: self.control.read_state.called)
        self.assertIsNone(self.hand.command)
        self.control.send_command.assert_not_called()
        self.control.close_device.assert_called_once()

    def test_initial_sdk_fault_is_not_retried(self):
        self.control.read_state.return_value = reply(1_501_035), self.raw
        with self.assertRaises(XHandError) as captured:
            self.hand.connect()
        self.assertEqual(captured.exception.code, 1_501_035)
        self.control.read_state.assert_called_once()
        self.clock.sleep.assert_not_called()
        self.control.close_device.assert_called_once()

    def test_source_time_precedes_blocking_sdk_read(self):
        start = self.clock.now_ns
        def delayed_read(*args):
            self.clock.advance(.05)
            return reply(), self.raw
        self.control.read_state.side_effect = delayed_read
        state = self.hand.connect()
        self.assertEqual(state.timestamp_ns, start)
        self.assertLess(state.timestamp_ns, self.clock.now_ns)

    def test_first_target_crc_fails_immediately_without_resending(self):
        self.hand.connect()
        self.control.send_command.return_value = reply(COMMUNICATION_CRC_ERROR_CODE)
        with self.assertRaisesRegex(XHandError, "delivery unconfirmed"):
            self.hand.send_action(self.qpos)
        self.control.send_command.assert_called_once()
        self.clock.sleep.assert_not_called()

    def test_stop_retries_and_uses_passive_when_pose_ages_out(self):
        self.hand.connect()
        modes = []
        def send(*args):
            modes.append([joint.mode for joint in self.hand.command.finger_command])
            return reply(COMMUNICATION_CRC_ERROR_CODE if len(modes) == 1 else 0)
        self.control.send_command.side_effect = send
        stamp = self.clock.now_ns - int((self.runtime.hand.feedback_max_age_s - .005) * 1e9)
        self.hand.stop(self.qpos, timestamp_ns=stamp)
        self.assertEqual(modes, [[3] * 12, [0] * 12])
        self.control.send_command.side_effect = None
        self.hand.send_action(self.qpos)
        self.assertEqual([joint.mode for joint in self.hand.command.finger_command], [3] * 12)

    def test_unconfirmed_stop_remains_an_error(self):
        self.hand.connect()
        self.control.send_command.return_value = reply(COMMUNICATION_CRC_ERROR_CODE)
        with self.assertRaisesRegex(RobotError, "stop timed out") as captured:
            self.hand.stop()
        self.assertGreater(self.control.send_command.call_count, 1)
        self.assertEqual(device_failure_reason(captured.exception), RunEndReason.HARDWARE_FAULT)

    def test_native_read_and_send_errors_remain_device_failures(self):
        self.hand.connect()
        for method, operation in (
            (self.control.read_state, self.hand.get_state),
            (self.control.send_command, lambda: self.hand.send_action(self.qpos)),
        ):
            failure = TimeoutError("native SDK timeout")
            method.side_effect = failure
            with self.assertRaisesRegex(RobotError, "native SDK timeout") as captured:
                operation()
            self.assertIs(captured.exception.__cause__, failure)
            self.assertEqual(device_failure_reason(captured.exception), RunEndReason.HARDWARE_FAULT)
            method.side_effect = None

    def test_tactile_sdk_error_preserves_partial_reset_and_cause(self):
        self.hand.connect()
        failure = TimeoutError("reset timeout")
        self.control.reset_sensor.side_effect = [reply(), failure]
        with self.assertRaisesRegex(RobotError, "reset timeout") as captured:
            self.hand.reset_tactile_sensors()
        self.assertIs(captured.exception.__cause__, failure)
        self.assertIn("exception (1/5)", self.hand.get_tactile_reset_metadata()["hand_tactile_reset_result"])

    def test_crc_and_missing_tactile_do_not_refresh_cached_source(self):
        cache = XHandFeedback()
        first = cache.consume(self.hand.connect())
        self.clock.advance(.02)
        self.control.read_state.return_value = reply(COMMUNICATION_CRC_ERROR_CODE), self.raw
        self.assertIs(cache.consume(self.hand.get_state()), first)
        self.control.read_state.return_value = reply(DISTRIBUTED_FORCE_UNAVAILABLE_CODE), self.raw
        current = cache.consume(self.hand.get_state())
        self.assertEqual(current["timestamp_ns"][0], self.clock.now_ns)
        self.assertEqual(current["tactile_aggregate_timestamp_ns"][0], self.clock.now_ns)
        self.assertEqual(current["tactile_dense_timestamp_ns"][0], first["timestamp_ns"][0])
        self.assertFalse(current.flags.writeable)

    def test_robot_reuses_first_frame_and_first_action_is_direct(self):
        epoch = self.connect_robot()
        self.control.read_state.assert_called_once()
        self.assertEqual(self.robot.hand_feedback.snapshot()["timestamp_ns"][0], self.clock.now_ns)
        self.robot.prepare_streaming(epoch)
        self.robot.send_action(RobotCommand(epoch, hand_qpos=self.qpos),
                               expires_ns=self.clock.now_ns + 1_000_000_000)
        self.control.send_command.assert_called_once()
        self.control.read_state.assert_called_once()
        self.clock.sleep.assert_not_called()

    def test_home_and_tactile_reset_reuse_confirmed_stop(self):
        epoch = self.connect_robot()
        self.control.reset_sensor.side_effect = lambda *args: self.clock.advance(.01) or reply()
        initialize_hand(self.state, self.robot, run_id=epoch, required_tactile=("aggregate", "dense"),
                        cancel_requested=lambda: False)
        self.assertEqual(self.arm.request_stop.call_count, 2)
        self.assertEqual(self.control.send_command.call_count, 3)
        self.assertEqual(self.control.reset_sensor.call_count, 5)
        self.assertFalse(self.robot.stop_required)
        self.assertEqual(self.robot.hand_feedback.snapshot()["tactile_dense_timestamp_ns"][0], self.clock.now_ns)

    def test_repeated_stop_skips_io_but_new_motion_and_close_do_not(self):
        epoch = self.connect_robot()
        self.robot.stop()
        self.robot.stop()
        self.arm.request_stop.assert_called_once()
        self.control.send_command.assert_called_once()
        self.robot.prepare_streaming(epoch)
        self.robot.send_action(RobotCommand(epoch, hand_qpos=self.qpos),
                               expires_ns=self.clock.now_ns + 1_000_000_000)
        self.robot.stop()
        self.assertEqual(self.arm.request_stop.call_count, 2)
        command = self.robot.hand.command
        self.robot.close()
        self.assertEqual(self.control.send_command.call_count, 4)
        self.assertEqual([joint.mode for joint in command.finger_command], [0] * 12)

    def test_disabled_hand_still_confirms_first_arm_stop(self):
        self.runtime = replace(self.runtime, hand=replace(self.runtime.hand, enabled=False))
        self.connect_robot()
        self.robot.stop()
        self.robot.stop()
        self.arm.request_stop.assert_called_once()
        self.arm.wait_stopped.assert_called_once()
        self.control.open_serial.assert_not_called()

    def test_failed_stop_does_not_skip_next_attempt_or_clear_fault(self):
        self.connect_robot()
        self.control.send_command.return_value = reply(1_501_035)
        with self.assertRaises(XHandError) as stopped:
            self.robot.stop()
        self.assertTrue(self.robot.stop_required)
        self.control.send_command.return_value = reply()
        self.robot.stop()
        self.assertEqual(self.arm.request_stop.call_count, 2)
        with self.assertRaises(XHandError) as checked:
            self.robot.check()
        self.assertIs(checked.exception, stopped.exception)

    def test_home_crc_stops_without_waiting_for_convergence(self):
        epoch = self.connect_robot()
        self.robot.stop()
        self.control.send_command.side_effect = [reply(COMMUNICATION_CRC_ERROR_CODE), reply()]
        with self.assertRaises(XHandError):
            home_hand(self.state, self.robot, run_id=epoch, cancel_requested=lambda: False)
        self.assertFalse(self.robot.stop_required)
        self.clock.sleep.assert_not_called()

    def test_diagnostic_crc_keeps_cleanup_required(self):
        hand = XHandControlExample(self.runtime.hand)
        hand.connect()
        self.control.send_command.return_value = reply(COMMUNICATION_CRC_ERROR_CODE)
        with self.assertRaises(XHandError):
            hand.execute_action(self.qpos)
        self.assertTrue(hand.motion_pending)
        self.control.send_command.return_value = reply()
        hand.stop_action()
        self.assertFalse(hand.motion_pending)
        self.assertEqual(self.control.send_command.call_count, 2)

    def test_diagnostic_failed_read_still_sends_passive_and_reports_error(self):
        hand = XHandControlExample(self.runtime.hand)
        hand.connect()
        hand.motion_pending = True
        self.control.read_state.side_effect = RuntimeError("feedback failed")
        with self.assertRaisesRegex(RuntimeError, "feedback failed"):
            hand.stop_action()
        self.control.send_command.assert_called_once()
        self.assertEqual([joint.mode for joint in hand.command.finger_command], [0] * 12)
        self.assertFalse(hand.motion_pending)

    def test_stale_initial_cache_requires_new_feedback(self):
        epoch = self.connect_robot()
        self.clock.advance(self.runtime.hand.feedback_max_age_s)
        self.robot.read_initial_state(epoch)
        self.assertEqual(self.control.read_state.call_count, 2)
        self.assertEqual(self.robot.hand_feedback.snapshot()["timestamp_ns"][0], self.clock.now_ns)

    def test_readiness_deadline_prevents_late_query(self):
        epoch = self.connect_robot()
        self.robot.hand_feedback.reset()
        self.robot.feedback_next_ns = self.clock.now_ns + int((INITIAL_STATE_TIMEOUT_S + 1) * 1e9)
        with self.assertRaisesRegex(RobotError, "feedback acquisition timed out"):
            self.robot.read_initial_state(epoch)
        self.control.read_state.assert_called_once()

    def test_device_fault_takes_priority_over_feedback_cancellation(self):
        epoch = self.connect_robot()
        self.arm.check_error.side_effect = RobotError("owner failed")
        with self.assertRaisesRegex(RobotError, "owner failed"):
            self.robot.read_initial_state(epoch, cancel_requested=lambda: True)

    def test_vr_crc_revokes_motion_without_committing_reference(self):
        epoch = self.connect_robot()
        self.robot.prepare_streaming(epoch)
        self.state.begin_motion()
        loop = TeleopLoop.__new__(TeleopLoop)
        loop.runtime_state, loop.runtime, loop.robot = self.state, self.runtime, self.robot
        loop.run_id, loop.recorder, loop.first_target_pending = epoch, None, True
        loop.controller = Mock(hand_retargeter=None)
        loop.controller.compute_target.return_value = ActionResult(np.asarray(self.runtime.arm.home_qpos), self.qpos)
        loop.record_control_timing = Mock()
        self.control.send_command.return_value = reply(COMMUNICATION_CRC_ERROR_CODE)
        with self.assertRaises(XHandError):
            loop.execute_control_step(Mock(), self.clock.now_ns + 1_000_000_000)
        loop.controller.commit.assert_not_called()
        self.arm.submit.assert_called_once()
        self.assertTrue(loop.first_target_pending)
        self.assertEqual(self.state.run_ended_reason, RunEndReason.HARDWARE_FAULT)

    def test_policy_crc_does_not_advance_cursor_or_target(self):
        epoch = self.connect_robot()
        self.robot.prepare_streaming(epoch)
        self.state.begin_motion()
        mapper = Mock(map_joint=Mock(return_value=ActionResult(np.asarray(self.runtime.arm.home_qpos), self.qpos)))
        runtime = replace(self.runtime, execution=ExecutionConfig(mode="sync", action_steps=1))
        rollout = Rollout(
            self.state, runtime, SimpleNamespace(n_obs_steps=1, control_dt_s=1/30, action_mode="joint"),
            epoch=epoch, robot=self.robot, observations=Mock(), needs=ObservationNeeds(),
            sensors=SimpleNamespace(channels=None), action_mapper=mapper, model=Mock(poll=Mock(return_value=None)),
            max_running_s=None, capture_ids=(), index=0, output_dir=None, record_raw=False,
        )
        rollout.plan = ActionChunk(np.r_[self.runtime.arm.home_qpos, self.qpos][None, :], 0)
        rollout.next_step_ns = self.clock.now_ns
        self.control.send_command.return_value = reply(COMMUNICATION_CRC_ERROR_CODE)
        with self.assertRaises(XHandError):
            rollout.step()
        self.assertEqual(rollout.last_slot, -1)
        self.assertIsNone(rollout.previous_arm)
        self.assertIsNone(rollout.target_expires_ns)
        self.assertEqual(self.state.run_ended_reason, RunEndReason.HARDWARE_FAULT)


if __name__ == "__main__":
    unittest.main()
