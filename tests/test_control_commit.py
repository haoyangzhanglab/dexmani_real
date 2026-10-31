import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from dexmani_real.control.actions import ActionResult
from dexmani_real.data.recorder import RecordingError
from dexmani_real.geometry.pose import Pose
from dexmani_real.robot.commands import OperationCancelled, RobotCommand, RobotError
from dexmani_real.robot.system import Robot
from dexmani_real.runtime.observation import ObservationNeeds, check_observation
from dexmani_real.runtime.state import RunEndReason
from dexmani_real.teleop.loop import TeleopLoop, TeleopPhase
from dexmani_real.teleop.vr import VRController


class ControlCommitTests(unittest.TestCase):
    def setUp(self):
        self.now = 100
        self.clock = mock.patch("dexmani_real.teleop.loop.time.monotonic_ns", side_effect=lambda: self.now)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.hand_retargeter = mock.Mock()
        self.controller = VRController(None, self.hand_retargeter, None, np.eye(3))
        self.controller.target_timestamp_ns = 20
        self.controller.previous_arm_command = np.zeros(7)
        self.controller.previous_eef_target = Pose(np.zeros(3), np.array([1.0, 0, 0, 0]))
        self.result = ActionResult(
            np.full(7, 0.1), np.full(12, 0.2),
            eef_pose=Pose(np.ones(3), np.array([1.0, 0, 0, 0])),
        )
        self.controller.compute_target = mock.Mock(return_value=self.result)
        self.state = SimpleNamespace(
            can_move=mock.Mock(return_value=True), motion_lock=threading.RLock(),
            run_ended_id=0, run_ended_reason=RunEndReason.NONE,
            error_state=False, estop_request=False,
        )
        self.row = SimpleNamespace(
            control_timestamp_ns=10, observation_timestamp_ns=20, vr={"recv_ts_ns": 20}
        )
        self.loop = TeleopLoop.__new__(TeleopLoop)
        self.loop.run_id = 7
        self.loop.tick_index = 0
        self.loop.runtime_state = self.state
        self.loop.runtime = SimpleNamespace(teleop=SimpleNamespace(vr_stale_threshold_s=1))
        self.loop.controller = self.controller
        self.loop.robot = mock.Mock()
        self.loop.recorder = None
        self.loop.record_control_timing = mock.Mock()
        self.loop.record_error = mock.Mock()
        self.loop.abort_episode = mock.Mock()

    def assert_uncommitted(self):
        self.hand_retargeter.commit.assert_not_called()
        np.testing.assert_array_equal(self.controller.previous_arm_command, np.zeros(7))
        np.testing.assert_array_equal(self.controller.previous_eef_target.p, np.zeros(3))

    def test_commit_uses_the_actual_immutable_submitted_command(self):
        def mutate_candidate(command, **kwargs):
            self.result.hand_qpos[:] = 0.9
            self.result.arm_qpos[:] = 0.8

        self.loop.robot.send_action.side_effect = mutate_candidate
        self.loop.execute_control_step(self.row, 200, 300)
        command = self.loop.robot.send_action.call_args.args[0]
        committed = self.hand_retargeter.commit.call_args
        self.assertIs(committed.args[0], command.hand_qpos)
        self.assertEqual(committed.kwargs["timestamp_ns"], 20)
        np.testing.assert_array_equal(committed.args[0], np.full(12, 0.2))
        np.testing.assert_array_equal(self.controller.previous_arm_command, np.full(7, 0.1))
        self.assertFalse(command.hand_qpos.flags.writeable)

    def test_success_does_not_copy_discarded_solver_diagnostics(self):
        self.hand_retargeter.last_diagnostics.copy.side_effect = AssertionError("unused diagnostic copy")

        def compute(row):
            self.controller.hand_observation_id = 1
            return self.result

        self.controller.compute_target.side_effect = compute
        self.loop.execute_control_step(self.row, 200, 300)
        self.hand_retargeter.last_diagnostics.copy.assert_not_called()
        self.loop.record_control_timing.assert_not_called()
        self.hand_retargeter.commit.assert_called_once()

    def test_failure_keeps_actual_compute_and_submission_times(self):
        self.loop.record_control_timing = TeleopLoop.record_control_timing.__get__(self.loop)
        self.loop.termination_details = []

        def compute(row):
            self.now = 125
            return self.result

        def failed_send(*args, **kwargs):
            self.now = 150
            raise RobotError("SDK failed during submit")

        self.controller.compute_target.side_effect = compute
        self.loop.robot.send_action.side_effect = failed_send
        with self.assertRaisesRegex(RobotError, "SDK failed during submit"):
            self.loop.execute_control_step(self.row, 200, 300)
        timing = self.loop.termination_details[0]
        self.assertEqual(timing["compute_start_ns"], 100)
        self.assertEqual(timing["compute_end_ns"], 125)
        self.assertEqual(timing["submit_start_ns"], 125)
        self.assertNotIn("submit_end_ns", timing)
        self.assertEqual(timing["ended_ns"], 150)
        self.assert_uncommitted()

    def test_sdk_failure_does_not_advance_retarget_or_smoothing(self):
        self.loop.robot.send_action.side_effect = RobotError("hand SDK rejected")
        with self.assertRaisesRegex(RobotError, "hand SDK rejected"):
            self.loop.execute_control_step(self.row, 200, 300)
        self.assert_uncommitted()
        self.loop.abort_episode.assert_called_once()

    def test_revocation_after_send_does_not_commit(self):
        def revoke(command, **kwargs):
            self.state.can_move.return_value = False
            self.state.run_ended_id = command.run_id
            self.state.run_ended_reason = RunEndReason.OPERATOR

        self.loop.robot.send_action.side_effect = revoke
        self.loop.execute_control_step(self.row, 200, 300)
        self.assert_uncommitted()

    def test_final_authority_check_and_commit_hold_the_motion_lock(self):
        begin, attempted, revoked = threading.Event(), threading.Event(), threading.Event()

        def revoke():
            if begin.wait(1):
                attempted.set()
                if self.state.motion_lock.acquire(timeout=1):
                    try:
                        self.state.can_move.return_value = False
                        revoked.set()
                    finally:
                        self.state.motion_lock.release()

        def commit(*args, **kwargs):
            begin.set()
            self.assertTrue(attempted.wait(1))
            self.assertFalse(revoked.wait(0.02), "revocation raced between authority check and commit")

        self.hand_retargeter.commit.side_effect = commit
        thread = threading.Thread(target=revoke, daemon=True)
        thread.start()
        try:
            self.loop.execute_control_step(self.row, 200, 300)
        finally:
            begin.set()
            thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        self.assertTrue(revoked.is_set())
        self.hand_retargeter.commit.assert_called_once()

    def test_expired_before_submission_does_not_send_or_commit(self):
        self.now = 200
        self.loop.execute_control_step(self.row, 200, 300)
        self.loop.robot.send_action.assert_not_called()
        self.assert_uncommitted()
        self.loop.abort_episode.assert_called_once()

    def test_expired_after_submission_does_not_commit(self):
        def expire(command, **kwargs):
            self.now = 200

        self.loop.robot.send_action.side_effect = expire
        self.loop.execute_control_step(self.row, 200, 300)
        self.assert_uncommitted()
        self.loop.abort_episode.assert_called_once()


class SendBoundaryTests(unittest.TestCase):
    def test_revocation_between_actuators_prevents_hand_send(self):
        robot = Robot.__new__(Robot)
        robot.arm, robot.hand = mock.Mock(), mock.Mock()
        robot.stop_signal = threading.Event()
        revoked = False

        def check_run(run_id):
            if revoked:
                raise OperationCancelled("authority revoked")

        def submit(*args, **kwargs):
            nonlocal revoked
            revoked = True

        robot.check_run = check_run
        robot.arm.submit.side_effect = submit
        command = RobotCommand(7, np.zeros(7), np.zeros(12))
        with mock.patch("dexmani_real.robot.system.time.monotonic_ns", return_value=100):
            with self.assertRaisesRegex(OperationCancelled, "authority revoked"):
                robot._send_action(command, valid_until_ns=200, execute_until_ns=300)
        robot.hand._send_action.assert_not_called()
        self.assertTrue(robot.stop_signal.is_set())

    def test_expiry_between_actuators_prevents_hand_send(self):
        robot = Robot.__new__(Robot)
        robot.arm, robot.hand = mock.Mock(), mock.Mock()
        robot.stop_signal = threading.Event()
        robot.check_run = mock.Mock()
        command = RobotCommand(7, np.zeros(7), np.zeros(12))
        with mock.patch("dexmani_real.robot.system.time.monotonic_ns", side_effect=[100, 200]):
            with self.assertRaisesRegex(TimeoutError, "deadline expired"):
                robot._send_action(command, valid_until_ns=200, execute_until_ns=300)
        robot.hand._send_action.assert_not_called()
        self.assertTrue(robot.stop_signal.is_set())


class TeleopCleanupTests(unittest.TestCase):
    def test_cleanup_failure_preserves_first_error_and_drops_start_keys(self):
        loop = TeleopLoop.__new__(TeleopLoop)
        original = RuntimeError("control failed")
        loop.start_episode = mock.Mock(side_effect=original)
        loop.record_error = mock.Mock()
        loop.end_episode = mock.Mock(side_effect=OSError("writer close failed"))
        loop.keyboard = mock.Mock()
        loop.phase = TeleopPhase.ACTIVE
        with self.assertLogs("dexmani_real.teleop.loop", level="ERROR"):
            with self.assertRaises(RuntimeError) as raised:
                loop.run_episode(7)
        self.assertIs(raised.exception, original)
        loop.end_episode.assert_called_once()
        loop.keyboard.discard_events.assert_called_once_with("b", "h", "t")

    def test_close_does_not_repeat_a_timed_out_recording_decision(self):
        loop = TeleopLoop.__new__(TeleopLoop)
        loop.recorder = SimpleNamespace(
            finish_request=object(), writer_pending=True, finish=mock.Mock(),
        )
        with self.assertRaisesRegex(RecordingError, "has not exited"):
            loop.close()
        loop.recorder.writer_pending = False
        loop.close()
        loop.recorder.finish.assert_not_called()


class SourceDeadlineTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(
            arm=SimpleNamespace(feedback_max_age_s=1e-6),
            hand=SimpleNamespace(enabled=True, feedback_max_age_s=1e-6),
            camera=SimpleNamespace(max_frame_age_s=1e-6),
            teleop=SimpleNamespace(vr_stale_threshold_s=1e-6),
        )
        self.row = SimpleNamespace(
            arm={"timestamp_ns": np.array([900])}, hand={"timestamp_ns": np.array([900])},
            camera={"timestamp_ns": 100}, vr={"recv_ts_ns": 900},
            observation_timestamp_ns=1000,
            point_cloud=np.zeros((1, 3)), pointcloud_timestamp_ns=100,
        )

    def test_recording_camera_does_not_shorten_control_deadline(self):
        loop = TeleopLoop.__new__(TeleopLoop)
        loop.runtime = self.runtime
        loop.control_needs = ObservationNeeds(vr=True)
        loop.record_needs = ObservationNeeds(camera_fields=("rgb", "depth"))
        loop.recording_dir = "episodes"
        with mock.patch("dexmani_real.runtime.observation.time.monotonic_ns", return_value=1000):
            issues, deadline = loop.admit_observation(self.row)
        self.assertEqual(issues, [])
        self.assertEqual(deadline, 1900)

    def test_recent_snapshot_cannot_refresh_old_camera_or_cloud_source(self):
        self.runtime.camera.max_frame_age_s = 1e-7
        with mock.patch("dexmani_real.runtime.observation.time.monotonic_ns", return_value=1000):
            for needs, expected in (
                (ObservationNeeds(camera_fields=("rgb",)), "camera_source_stale"),
                (ObservationNeeds(point_cloud=True), "pointcloud_stale"),
            ):
                with self.subTest(needs=needs):
                    issues, deadline = check_observation(self.row, self.runtime, needs)
                    self.assertIn(expected, issues)
                    self.assertIsNone(deadline)


if __name__ == "__main__":
    unittest.main()
