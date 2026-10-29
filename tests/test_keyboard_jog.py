"""真实 jog/Robot/arm 控制链的离线回归；仅替换 SDK、IK 和进程调度。"""

import multiprocessing as mp
import threading
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import ANY, Mock, patch

import numpy as np

from dexmani_real.config.real import load_config
from dexmani_real.geometry.pose import Pose
from dexmani_real.robot.commands import OperationCancelled, RobotCommand, RobotError
from dexmani_real.robot.system import Robot
from dexmani_real.robot.xarm7.process import ArmProcess
from dexmani_real.robot.xarm7.protocol import ArmChannels, HomeRequest, PrepareRequest
from dexmani_real.robot.xarm7.worker import ArmController
from dexmani_real.runtime.jog import CartesianJog
from dexmani_real.runtime.keyboard import Keyboard
from dexmani_real.runtime.state import RunEndReason, SafetyState, SessionState
from dexmani_real.teleop.keyboard import KeyboardTeleop


class KeyboardJogTests(unittest.TestCase):
    def setUp(self):
        self.now = 100_000_000_000
        clock = patch('time.monotonic_ns', side_effect=lambda: self.now)
        clock.start()
        self.addCleanup(clock.stop)
        self.runtime = load_config(overrides={'hand': {'enabled': False}})
        self.state = SessionState()
        self.state.arm()
        self.keys = Keyboard(estop_callback=self.state.request_emergency_stop,
                             bindings={'q': self.state.request_quit})
        self.keys.active = True
        self.io = ArmChannels(
            mp.get_context('spawn'), run_signal=self.state.run_signal,
            stop_signal=self.state.stop_signal, emergency_signal=self.state.emergency_signal,
            signal_lock=self.state.motion_lock,
        )
        self.on_wait = lambda: None
        wait = patch.object(self.io.stop, 'wait', side_effect=self.wait)
        wait.start()
        self.addCleanup(wait.stop)
        self.q = np.asarray(self.runtime.arm.home_qpos)
        self.driver = SimpleNamespace(
            mode=1, is_connected=True, stop=Mock(), enter_mode=Mock(), servo=Mock(),
            read=lambda: (self.q.copy(), np.zeros(7)),
            read_status=lambda: SimpleNamespace(error_code=0, state=0),
        )
        self.replies = []
        self.controller = ArmController(
            self.runtime.arm, self.io, SimpleNamespace(send=self.replies.append), self.driver,
        )
        self.controller.read_feedback()
        self.arm = Mock(cfg=self.runtime.arm, io=self.io, is_connected=True,
                        operation_unresolved=False)
        self.arm.request_stop.side_effect = self.io.stop.set
        self.arm.wait_stopped.side_effect = self.wait_stopped
        self.arm.prepare.side_effect = self.prepare_arm
        self.arm.snapshot.side_effect = lambda: ArmProcess.snapshot(self.arm)
        self.arm.submit.side_effect = lambda *args, **kwargs: ArmProcess.submit(self.arm, *args, **kwargs)
        self.robot = Robot(
            self.runtime.arm, self.runtime.hand, run_signal=self.state.run_signal,
            stop_signal=self.state.stop_signal, emergency_signal=self.state.emergency_signal,
            signal_lock=self.state.motion_lock,
        )
        self.robot.owner, self.robot.connected, self.robot.arm = threading.get_ident(), True, self.arm
        self.solver = Mock()
        self.solver.compute_eef_pose_base.return_value = Pose.identity()
        self.solver.solve.return_value = SimpleNamespace(qpos=self.q + 0.001)
        self.jog = CartesianJog(self.solver, self.runtime.keyboard_teleop, (-np.ones(3), np.ones(3)))
        self.jog.prepare(self.robot, self.state)

    def wait(self, seconds):
        self.now += round(seconds * 1e9)
        self.on_wait()
        return self.io.stop.is_set()

    def wait_stopped(self):
        if not self.io.stopped.is_set():
            self.controller.stop()

    def prepare_arm(self, run_id, *, service):
        ArmProcess.begin_operation(self.arm, run_id)
        self.controller.operation(PrepareRequest(run_id))
        error = self.replies.pop()
        if error is not None:
            raise error

    def step(self, *, consume=True):
        self.controller.read_feedback()
        submissions = self.arm.submit.call_count
        self.jog.step(self.robot, self.keys, self.state, self.runtime)
        published = self.arm.submit.call_count != submissions
        if consume and published:
            self.controller.tick()
        return published

    def test_repeated_short_presses_and_release_prepare_only_once(self):
        ready, run_id = self.now, self.state.run_id
        for key in ('w', 'd', 'up', 'i', 'j'):
            self.keys.handle_key(key, True)
            self.assertTrue(self.step())
            self.assertFalse(self.controller.active['hold'])
            self.keys.handle_key(key, False)
            self.assertTrue(self.step())
            self.assertTrue(self.controller.active['hold'])
        self.assertEqual(self.now, ready)
        self.assertEqual(self.state.run_id, run_id)
        self.assertEqual(self.state.safety_state, SafetyState.RUNNING)
        self.assertEqual(self.arm.prepare.call_count, 1)
        self.assertEqual(self.driver.stop.call_count, 1)  # 只有首次准备前的停止。

    def test_change_during_ik_holds_then_accepts_current_direction(self):
        self.keys.handle_key('w', True)
        def change_direction(*args, **kwargs):
            self.keys.handle_key('w', False)
            self.keys.handle_key('d', True)
            return SimpleNamespace(qpos=self.q + 0.001)
        self.solver.solve.side_effect = change_direction
        self.assertTrue(self.step())
        self.assertTrue(self.controller.active['hold'])
        self.solver.solve.side_effect = None
        self.assertTrue(self.step())
        self.assertFalse(self.controller.active['hold'])
        self.assertEqual(self.arm.prepare.call_count, 1)
        self.assertEqual(self.driver.stop.call_count, 1)

    def test_key_change_after_publication_does_not_revoke_session(self):
        self.keys.handle_key('w', True)
        self.step(consume=False)
        self.keys.handle_key('w', False)
        self.keys.handle_key('d', True)
        self.controller.tick()
        self.assertFalse(self.io.stop.is_set())
        self.assertTrue(self.step())
        self.keys.handle_key('d', False)
        self.assertTrue(self.step())
        self.assertTrue(self.controller.active['hold'])
        self.assertEqual(self.arm.prepare.call_count, 1)

    def test_ik_rejection_holds_until_release_without_repreparing(self):
        self.keys.handle_key('w', True)
        self.solver.solve.return_value = SimpleNamespace(qpos=None, reason='collision')
        with self.assertLogs('dexmani_real.runtime.jog', level='WARNING') as logs:
            self.step()
        self.assertIn('collision', logs.output[0])
        self.assertTrue(self.controller.active['hold'])
        self.step()
        self.solver.solve.assert_called_once()
        self.keys.handle_key('w', False)
        self.step()
        self.keys.handle_key('d', True)
        self.solver.solve.return_value = SimpleNamespace(qpos=self.q + 0.001)
        self.step()
        self.assertFalse(self.controller.active['hold'])
        self.assertEqual(self.arm.prepare.call_count, 1)
        self.assertEqual(self.driver.stop.call_count, 1)

    def test_capture_pause_stays_stopped_until_release_then_prepares_once(self):
        self.keys.handle_key('w', True)
        self.step()
        self.jog.pause(self.robot, self.state)
        self.assertEqual(self.state.safety_state, SafetyState.ARMED)
        self.assertIsNone(self.controller.reference)
        self.step()
        self.assertEqual(self.arm.prepare.call_count, 1)
        self.keys.handle_key('w', False)
        self.assertFalse(self.step())  # 单独的准备轮次。
        self.assertTrue(self.step())
        self.assertTrue(self.controller.active['hold'])
        self.assertEqual(self.arm.prepare.call_count, 2)

    def test_changes_during_preparation_are_sampled_after_preparation(self):
        self.jog.stop(self.robot, self.state)
        self.on_wait = lambda: self.keys.handle_key('d', True)
        self.assertFalse(self.step())
        self.assertEqual(self.state.safety_state, SafetyState.RUNNING)
        self.assertTrue(self.step())
        self.assertFalse(self.controller.active['hold'])
        self.assertEqual(self.arm.prepare.call_count, 2)

    def test_home_visits_all_milestones_then_jog_prepares_again(self):
        path = np.stack((self.q, self.q + .01, self.q + .02, self.q))

        def move(target):
            self.q = target.copy()

        self.driver.move_home_milestone = Mock(side_effect=move)
        self.jog.stop(self.robot, self.state)
        ArmProcess.begin_operation(self.arm, self.state.run_id)
        started = self.now
        self.controller.operation(HomeRequest(self.state.run_id, path, self.q, self.now + 5_000_000_000))
        self.assertIsNone(self.replies.pop())
        self.driver.enter_mode.assert_called_with(0, check_abort=ANY)
        np.testing.assert_array_equal(
            [call.args[0] for call in self.driver.move_home_milestone.call_args_list], path[1:],
        )
        self.assertGreater(self.now - started, 900_000_000)
        self.jog.stop(self.robot, self.state)
        self.assertFalse(self.step())
        self.keys.handle_key('w', True)
        self.assertTrue(self.step())
        self.assertEqual(self.arm.prepare.call_count, 2)
        self.driver.enter_mode.assert_called_with(1, check_abort=ANY)

    def test_quit_during_home_stops_before_next_milestone(self):
        path = np.stack((self.q, self.q + .01, self.q))
        self.driver.move_home_milestone = Mock(side_effect=lambda _: self.state.request_quit())
        self.jog.stop(self.robot, self.state)
        ArmProcess.begin_operation(self.arm, self.state.run_id)
        self.controller.operation(HomeRequest(self.state.run_id, path, self.q, self.now + 5_000_000_000))
        self.assertIsInstance(self.replies.pop(), OperationCancelled)
        self.driver.move_home_milestone.assert_called_once()
        self.assertTrue(self.io.stopped.is_set())
        self.assertFalse(self.io.fault.value)
        self.assertEqual(self.controller.status['prepared_run_id'], 0)
        self.assertIsNone(self.io.take_goal())

    def test_estop_during_ik_prevents_publication(self):
        self.keys.handle_key('w', True)
        def estop(*args, **kwargs):
            self.keys.handle_key('esc', True)
            return SimpleNamespace(qpos=self.q + 0.001)
        self.solver.solve.side_effect = estop
        with self.assertRaises(OperationCancelled):
            self.step()
        self.assertTrue(self.io.emergency.is_set())
        self.assertIsNone(self.io.take_goal())
        self.driver.servo.assert_not_called()

    def test_quit_revokes_idle_hold(self):
        self.step()
        self.keys.handle_key('q', True)
        self.assertFalse(self.step())
        self.assertTrue(self.state.quit_requested)
        self.assertTrue(self.io.stopped.is_set())
        self.assertIsNone(self.controller.reference)

    def test_hold_still_requires_connected_arm_and_explicit_deadline(self):
        hold = RobotCommand(self.state.run_id, arm_hold=True)
        with self.assertRaises(ValueError):
            self.robot.validate_command(hold)
        self.robot.connected = False
        with self.assertRaises(RobotError):
            self.robot.validate_command(hold, execute_until_ns=self.now + 1)

    def test_reading_cached_feedback_does_not_extend_target_lifetime(self):
        self.controller.read_feedback()
        source_ns = self.now
        self.now += 40_000_000
        self.jog.step(self.robot, self.keys, self.state, self.runtime)
        goal = self.io.take_goal()
        self.assertEqual(
            int(goal['execute_until_ns']),
            source_ns + int(self.runtime.keyboard_teleop.target_max_age_s * 1e9),
        )
        self.assertEqual(
            int(goal['submit_until_ns']),
            source_ns + int(self.runtime.arm.feedback_max_age_s * 1e9),
        )

    def test_missing_feedback_stops_and_preserves_failure_reason(self):
        self.step()
        self.runtime = replace(self.runtime, hand=replace(self.runtime.hand, enabled=True))
        teleop = KeyboardTeleop(self.runtime, no_hand=False)
        teleop.start = Mock()
        teleop.run_control_loop = self.step
        with (
            patch('dexmani_real.teleop.keyboard.SessionState', return_value=self.state),
            patch('dexmani_real.teleop.keyboard.Robot', return_value=self.robot),
            patch('dexmani_real.teleop.keyboard.Keyboard', return_value=self.keys),
            patch('dexmani_real.teleop.keyboard.configure_logging'),
            self.assertRaisesRegex(RobotError, 'hand_feedback_unavailable'),
        ):
            teleop.run()
        self.assertEqual(self.state.run_ended_reason, RunEndReason.HARDWARE_FAULT)
        self.assertEqual(self.state.safety_state, SafetyState.FAULT)
        self.assertTrue(self.io.stopped.is_set())
        self.assertIsNone(self.controller.reference)
        self.assertIsNone(self.io.take_goal())

    def test_prepare_cannot_resume_after_emergency_stop(self):
        self.keys.handle_key('esc', True)
        with self.assertRaises(OperationCancelled):
            self.jog.prepare(self.robot, self.state)
        self.assertNotEqual(self.state.safety_state, SafetyState.RUNNING)

    def test_quit_during_initial_prepare_closes_without_session_error(self):
        self.jog.stop(self.robot, self.state)
        self.on_wait = lambda: self.keys.handle_key('q', True)
        teleop = KeyboardTeleop(self.runtime, no_hand=True)
        teleop.start = lambda: self.jog.prepare(self.robot, self.state)
        teleop.run_control_loop = Mock()
        with (
            patch('dexmani_real.teleop.keyboard.SessionState', return_value=self.state),
            patch('dexmani_real.teleop.keyboard.Robot', return_value=self.robot),
            patch('dexmani_real.teleop.keyboard.Keyboard', return_value=self.keys),
            patch('dexmani_real.teleop.keyboard.configure_logging'),
        ):
            self.assertEqual(teleop.run(), 0)
        teleop.run_control_loop.assert_not_called()
        self.assertFalse(self.state.error_state)
        self.assertEqual(self.state.safety_state, SafetyState.DISARMED)
        self.assertTrue(self.io.stopped.is_set())
        self.driver.servo.assert_not_called()


if __name__ == '__main__':
    unittest.main()
