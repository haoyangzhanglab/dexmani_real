"""真实按键代际与遥操作流程，设备、录制 I/O 和声音进程使用替身。"""

import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.config.real import RealConfig
from dexmani_real.control.actions import ActionResult
from dexmani_real.data.recorder import RecordingError
from dexmani_real.robot.commands import OperationCancelled, RobotError
from dexmani_real.runtime.keyboard import KeyEvent, Keyboard
from dexmani_real.runtime.state import RunEndReason, SafetyState, SessionState
from dexmani_real.teleop.audio import AudioFeedback
from dexmani_real.teleop.loop import TeleopLoop, TeleopPhase


class TeleopOperatorTests(unittest.TestCase):
    def setUp(self):
        self.state = SessionState()
        self.state.arm()
        self.keyboard = Keyboard(
            get_run_id=lambda: self.state.run_id,
            estop_callback=self.state.request_emergency_stop,
            bindings={
                "s": self.state.request_stop,
                "q": lambda: self.state.request_stop(reason=RunEndReason.QUIT),
            },
        )
        self.keyboard.active = True
        self.keyboard.listener = Mock()
        self.order = []
        self.robot = Mock()
        self.robot.stop.side_effect = lambda: self.order.append("stop")
        self.robot.prepare_streaming.side_effect = self.prepare
        self.controller = Mock(hand_retargeter=None)
        self.controller.rotation.as_matrix.return_value = np.eye(3)
        self.controller.compute_target.return_value = ActionResult(np.zeros(7), None)
        self.recorders = []
        self.loop = TeleopLoop(
            self.state, RealConfig(), self.robot,
            task_label="test", sensors=Mock(), controller=self.controller,
            recording_dir=Path("/tmp/unused-teleop-test"), max_duration_s=60,
            runtime_config_json="{}", keyboard=self.keyboard, audio=Mock(),
        )
        self.keyboard.bindings.update(c=self.loop.request_pause, d=self.loop.request_discard)
        self.loop.admit_observation = Mock(return_value=[])
        self.row = Mock(
            observation_timestamp_ns=time.monotonic_ns(),
            arm={"qpos": np.zeros((1, 7))}, hand=None,
            vr={"wrist_pos": np.zeros(3), "wrist_quat_wxyz": np.array([1, 0, 0, 0]),
                "recv_ts_ns": time.monotonic_ns()},
        )
        for name, kwargs in (
            ("EpisodeRecorder", {"side_effect": self.make_recorder}),
            ("snapshot_recording_metadata", {"return_value": {}}),
            ("read_observation", {"return_value": self.row}),
        ):
            replacement = patch(f"dexmani_real.teleop.loop.{name}", **kwargs)
            replacement.start()
            self.addCleanup(replacement.stop)

    def make_recorder(self, *args, **kwargs):
        recorder = Mock(
            start_attempted=True, published_path=None, submitted_frames=1,
            is_recording=True, finish_request=None,
        )
        recorder.start.side_effect = lambda **kwargs: self.order.append("start")
        recorder.finish.side_effect = lambda **kwargs: self.finish(**kwargs)
        self.recorders.append(recorder)
        return recorder

    def finish(self, **kwargs):
        self.order.append("finish")
        return Path("/tmp/unused-teleop-test/saved") if kwargs["save"] else None

    def prepare(self, run_id):
        if run_id != self.state.run_id:
            raise OperationCancelled("stale preparation")
        self.order.append("prepare")
        self.state.stop_signal.clear()

    def press(self, key):
        self.keyboard.handle_key(key, True)
        self.keyboard.handle_key(key, False)

    def start(self):
        self.loop.start_episode(self.state.run_id)
        self.assertIs(self.loop.phase, TeleopPhase.ACTIVE)

    def command(self, key):
        self.press(key)
        return self.loop.service_episode(0)

    def test_pause_keeps_capture_and_resume_saves_then_reanchors_new_segment(self):
        self.start()
        old_run = self.loop.run_id
        first = self.loop.recorder
        self.press("c")
        self.assertTrue(self.state.stop_signal.is_set())  # 无需等待控制线程处理按键。
        self.assertFalse(self.state.can_move(run_id=old_run))
        self.assertTrue(self.loop.service_episode(0))
        self.assertIs(self.loop.phase, TeleopPhase.PAUSED)
        self.loop.run_control_tick()
        first.finish.assert_not_called()
        first.append.assert_not_called()
        self.robot.send_action.assert_not_called()

        self.row.arm["qpos"] = np.ones((1, 7))
        self.order.clear()
        self.assertTrue(self.command("c"))
        self.assertEqual(self.order, ["stop", "finish", "start", "prepare"])
        self.assertIsNot(self.loop.recorder, first)
        self.assertEqual(first.finish.call_args.kwargs["technical_status"], "normal")
        self.assertIs(self.loop.phase, TeleopPhase.ACTIVE)
        self.assertGreater(self.loop.run_id, old_run)
        self.assertEqual(self.controller.reset.call_count, 2)
        self.assertEqual(self.loop.termination_details[-1]["arm_qpos"], [1] * 7)
        self.loop.execute_control_step(self.row, time.monotonic_ns() + 1_000_000_000)
        self.loop.audio.play.assert_called_with("resume", interrupt=False)

    def test_pause_during_submission_keeps_final_frame_in_old_recorder(self):
        self.start()
        first = self.loop.recorder

        def compute(*args, **kwargs):
            self.press("c")
            return ActionResult(np.zeros(7), None)

        self.controller.compute_target.side_effect = compute
        self.loop.execute_control_step(self.row, time.monotonic_ns() + 1_000_000_000)
        self.robot.send_action.assert_not_called()
        first.append.assert_called_once_with(self.row, None)
        first.finish.assert_not_called()
        self.assertFalse(self.loop.capture_failed)
        self.assertTrue(self.loop.service_episode(0))
        self.assertIs(self.loop.phase, TeleopPhase.PAUSED)

    def test_quit_requires_confirmation_and_can_be_cancelled_with_continue(self):
        self.start()
        first = self.loop.recorder
        self.assertTrue(self.command("q"))
        self.assertIs(self.loop.phase, TeleopPhase.QUIT_PENDING)
        self.assertFalse(self.state.quit_requested)
        first.finish.assert_not_called()
        self.assertTrue(self.command("b"))
        self.assertIs(self.loop.phase, TeleopPhase.ACTIVE)
        self.assertFalse(self.state.quit_requested)
        first.finish.assert_called_once()
        self.assertTrue(self.command("q"))
        self.assertFalse(self.command("q"))
        self.assertTrue(self.state.quit_requested)
        self.loop.close()
        self.recorders[-1].finish.assert_called_once()

    def test_pause_callback_before_event_enqueue_does_not_end_capture(self):
        self.start()
        event = KeyEvent("c", self.state.run_id)
        self.loop.request_pause()
        # 控制线程先看到了撤权，而键盘线程尚未发布事件。
        self.loop.execute_control_step(self.row, time.monotonic_ns() + 1_000_000_000)
        self.assertIsNone(self.loop.pending_termination_reason)
        self.assertTrue(self.loop.service_episode(0))
        self.keyboard.events.append(event)
        self.assertTrue(self.loop.service_episode(0))
        self.assertIs(self.loop.phase, TeleopPhase.PAUSED)
        self.loop.recorder.finish.assert_not_called()

    def test_cancelled_preparation_with_only_queued_home_still_ends(self):
        def cancelled(run_id):
            self.press("h")
            raise OperationCancelled("executor cancelled preparation")

        self.robot.prepare_streaming.side_effect = cancelled
        with patch("dexmani_real.teleop.loop.home_robot") as home:
            self.loop.run_episode(self.state.run_id)
        self.assertIs(self.loop.phase, TeleopPhase.IDLE)
        home.assert_not_called()
        self.recorders[0].finish.assert_called_once()

    def test_emergency_stop_during_preparation_keeps_fault_reason(self):
        def cancelled(run_id):
            self.state.request_emergency_stop()
            raise OperationCancelled("preparation interrupted by estop")

        self.robot.prepare_streaming.side_effect = cancelled
        self.loop.run_episode(self.state.run_id)
        self.assertEqual(self.recorders[0].finish.call_args.kwargs["reason"], "estop")
        self.assertEqual(self.recorders[0].finish.call_args.kwargs["technical_status"], "failed")
        self.robot.send_action.assert_not_called()

    def test_quit_during_preparation_does_not_start_or_exit_before_confirmation(self):
        def cancelled(run_id):
            self.press("q")
            raise OperationCancelled("preparation cancelled")

        self.robot.prepare_streaming.side_effect = cancelled
        self.loop.start_episode(self.state.run_id)
        self.assertTrue(self.loop.service_episode(0))
        self.assertIs(self.loop.phase, TeleopPhase.QUIT_PENDING)
        self.assertFalse(self.state.quit_requested)
        self.robot.send_action.assert_not_called()
        self.controller.reset.assert_not_called()
        self.assertFalse(self.command("q"))
        self.loop.close()
        self.assertEqual(self.recorders[0].finish.call_args.kwargs["technical_status"], "normal")

    def test_save_and_discard_while_paused(self):
        for key, save in (("s", True), ("d", False)):
            with self.subTest(key=key):
                self.start()
                self.command("c")
                recorder = self.loop.recorder
                self.assertFalse(self.command(key))
                self.loop.end_episode("operator")
                self.assertEqual(recorder.finish.call_args.kwargs["save"], save)
                self.assertIs(self.loop.phase, TeleopPhase.IDLE)

    def test_home_stops_and_saves_before_moving(self):
        self.start()
        old_run = self.loop.run_id
        self.order.clear()

        def home(state, robot, get_planner, *, run_id, cancel_requested):
            self.assertTrue(state.can_move(run_id=run_id, required_safety_state=SafetyState.ARMED))
            self.assertGreater(run_id, old_run)
            self.assertIsNone(self.loop.recorder)
            self.assertFalse(cancel_requested())
            self.order.append("home")

        with patch("dexmani_real.teleop.loop.home_robot", side_effect=home):
            self.command("h")
        self.assertEqual(self.order, ["stop", "finish", "home"])
        self.assertIs(self.loop.phase, TeleopPhase.IDLE)

    def test_stop_while_saving_cancels_home_and_stale_resume(self):
        self.start()

        def finish(**kwargs):
            self.press("s")
            return self.finish(**kwargs)

        self.loop.recorder.finish.side_effect = finish
        with patch("dexmani_real.teleop.loop.home_robot") as home:
            self.command("h")
        home.assert_not_called()

        self.start()
        self.command("c")
        stale = KeyEvent("c", self.state.run_id)
        self.state.request_stop()
        self.robot.prepare_streaming.reset_mock()
        self.loop.handle_commands([stale])
        self.robot.prepare_streaming.assert_not_called()
        self.assertIs(self.loop.phase, TeleopPhase.PAUSED)

    def test_recording_failure_prevents_resume(self):
        self.start()
        self.command("c")
        self.loop.recorder.finish.side_effect = RecordingError("writer failed")
        self.robot.prepare_streaming.reset_mock()
        with self.assertRaisesRegex(RecordingError, "writer failed"):
            self.command("c")
        self.robot.prepare_streaming.assert_not_called()
        self.assertTrue(self.state.error_state)
        self.assertTrue(self.state.stop_signal.is_set())

    def test_stop_failure_prevents_home(self):
        self.start()
        self.robot.stop.side_effect = RobotError("stop not confirmed")
        with patch("dexmani_real.teleop.loop.home_robot") as home:
            with self.assertRaisesRegex(RobotError, "stop not confirmed"):
                self.command("h")
        home.assert_not_called()
        self.assertTrue(self.state.error_state)

    def test_sequential_session_pauses_resumes_and_confirms_exit(self):
        # 在主循环轮询时输入；准备内部轮询不提前消费后续操作者动作。
        phases = [TeleopPhase.ACTIVE, TeleopPhase.PAUSED, TeleopPhase.ACTIVE, TeleopPhase.QUIT_PENDING]
        keys = iter(("c", "c", "q", "q"))
        poll = self.keyboard.poll

        def scripted_poll(timeout):
            if timeout > 0:
                self.assertIs(self.loop.phase, phases.pop(0))
                self.press(next(keys))
            return poll(0)

        self.keyboard.poll = scripted_poll
        self.loop.run_episode(self.state.run_id)
        self.assertFalse(phases)
        self.assertTrue(self.state.quit_requested)
        self.assertFalse(self.state.error_state)
        self.assertIs(self.loop.phase, TeleopPhase.IDLE)
        self.assertEqual(len(self.recorders), 2)
        for recorder in self.recorders:
            recorder.finish.assert_called_once()


class AudioFeedbackTests(unittest.TestCase):
    def test_background_playback_is_interruptible_and_reaped(self):
        started, killed = threading.Event(), threading.Event()
        first = Mock(returncode=-9)
        first.poll.return_value = None
        first.kill.side_effect = killed.set

        def communicate():
            started.set()
            if not killed.wait(2):
                raise AssertionError("playback was not interrupted")
            first.poll.return_value = -9
            return b"", b""

        first.communicate.side_effect = communicate
        second = Mock(returncode=0)
        second.communicate.return_value = (b"", b"")
        second.poll.return_value = 0
        with patch("dexmani_real.teleop.audio.shutil.which", return_value="/usr/bin/aplay"), \
                patch("dexmani_real.teleop.audio.subprocess.Popen", side_effect=[first, second]) as popen:
            audio = AudioFeedback()
            try:
                audio.play("begin")
                self.assertTrue(started.wait(1))
                audio.play("pause")
                self.assertTrue(audio.wait_until_idle(2))
            finally:
                audio.close()
        first.kill.assert_called_once()
        self.assertEqual(popen.call_count, 2)
        self.assertFalse(audio.worker.is_alive())

    def test_playback_failure_is_reported_and_does_not_block_control(self):
        proc = Mock(returncode=1)
        proc.communicate.return_value = (b"", b"audio device unavailable")
        proc.poll.return_value = 1
        with patch("dexmani_real.teleop.audio.shutil.which", return_value="/usr/bin/aplay"), \
                patch("dexmani_real.teleop.audio.subprocess.Popen", return_value=proc) as popen, \
                self.assertLogs("dexmani_real.teleop.audio", level="WARNING") as logs:
            audio = AudioFeedback()
            try:
                audio.play("begin")
                self.assertTrue(audio.wait_until_idle(2))
                audio.play("pause")
                self.assertTrue(audio.wait_until_idle(0))
            finally:
                audio.close()
        popen.assert_called_once()
        self.assertIn("audio device unavailable", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
