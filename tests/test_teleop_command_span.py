"""Offline tests for teleop backpressure-span accounting (task T2, V05/V19).

A teleop producer can hold a prepared-but-uncommitted candidate while the
ordered command FIFO is full. Every pause boundary that discards that
candidate must close the visible ``[WAIT]`` span with its own ``[DROP]``:
otherwise the span outlives its candidate, the next ``[WAIT]`` is suppressed,
and the following ``[RESUME]`` reports a wait that never resumed.

The span tests exercise the real tracker and close helper. Recording cases
drive the real Teleop loop, client and RecorderIO transaction against synthetic
inputs and temporary media; planner/input initialization is isolated from hardware.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

# Only a genuinely missing dependency may skip this module: a renamed or
# broken symbol must fail the suite rather than hide behind a skip.
try:
    from dexmani_real.control.publication import PublishWaitTracker
    from dexmani_real.teleop.control_loop.grid import close_publish_span

    _IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - environment guard
    PublishWaitTracker = None
    close_publish_span = None
    _IMPORT_ERROR = exc




class _TeleopRecordingHarness:
    """Real controller, client and finalizer; synthetic inputs and temporary media."""

    def __init__(self, case, *, max_frames=100):
        import tempfile
        from pathlib import Path
        from queue import Queue
        from unittest import mock
        from dexmani_real.config.experiment import resolve_experiment_config
        from dexmani_real.teleop.config import TeleopConfig
        from dexmani_real.recording.client import RecorderClient, StartRecording
        from dexmani_real.recording.io_worker import RecorderIOConfig, _RecorderIOSession
        from dexmani_real.ipc.schema import make_record_sample_dtype
        from test_deployment_evidence import _fake_shared
        from test_recording_preservation import _make_recorder, _RGB_SHAPE, _DEPTH_SHAPE, _CONTROL_HZ

        self.case = case
        tmp = tempfile.TemporaryDirectory()
        case.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.shared = _fake_shared()
        self.shared.set_heartbeat = lambda *args: None
        self.shared.set_ready = lambda *args: None
        self.shared.is_ready = lambda *args: True
        self.shared.recorder_consumed_sequence = SimpleNamespace(value=0)
        self.shared.record_result_q = Queue()
        self.messages = []
        self.starts = 0
        self.now = 10.0
        self.frame = 0
        self.controls = []
        self.snapshots = []
        self.recorder = _make_recorder(self.root, min_frames=1)
        self.recorder.max_frames = max_frames

        class Ring:
            dtype = make_record_sample_dtype(_RGB_SHAPE, _DEPTH_SHAPE)
            maxlen = 64
            latest_sequence = 0

            def __init__(self):
                self.frames = []

            def write(self, frame):
                self.frames.append(frame.copy())
                self.latest_sequence += 1
                return self.latest_sequence

            def read_sequence(self, sequence):
                return self.frames[sequence - 1], 0, sequence

        self.shared.record_sample_ring = Ring()
        self.session = _RecorderIOSession(
            self.shared, RecorderIOConfig(data_dir=str(self.root), max_frames=max_frames,
                                          control_hz=_CONTROL_HZ, min_frames=1), self.recorder)
        harness = self

        class ControlQueue(Queue):
            def put_nowait(self, message):
                harness.messages.append(message)
                if isinstance(message, StartRecording):
                    harness.starts += 1
                    # Calibration/resource setup only; the START owner and reply are real.
                    with mock.patch("dexmani_real.recording.io_worker._build_start_metadata",
                                    return_value={"task_label": "test", "episode_name": f"episode_{harness.starts}"}):
                        harness.session._handle_start(message)
                else:
                    super().put_nowait(message)

        self.shared.record_control_q = ControlQueue()
        self.client = RecorderClient(self.shared)
        runtime = resolve_experiment_config(cli_overrides={"policy.hand_enabled": False})
        self.config = TeleopConfig(runtime=runtime)

        def cleanup_writer():
            if self.recorder.is_recording:
                self.recorder.finish_episode(False, "test_cleanup")
            pending = self.session.pending_finalization
            if pending is not None:
                pending.thread.join(5)
                case.assertFalse(pending.thread.is_alive())
        case.addCleanup(cleanup_writer)

    def add_rows(self, count=2):
        import numpy as np
        from test_recording_preservation import _state, _action, _VR_FRAME
        for index in range(count):
            self.case.assertTrue(self.client.add_frame(
                _state(float(index + 1)), _action(), dict(_VR_FRAME), arm_qpos_sent=np.zeros(7)))

    def begin_finalization(self):
        stop = self.shared.record_control_q.get_nowait()
        self.session._handle_stop(stop)
        self.session._drain_samples()

    def finalize(self):
        if self.session.pending_finalization is None:
            self.begin_finalization()
        pending = self.session.pending_finalization
        self.case.assertIsNotNone(pending)
        pending.thread.join(5)
        self.case.assertFalse(pending.thread.is_alive())
        self.session._poll_finalization()

    def run(self, steps, *, decision=None):
        import contextlib
        import io
        import numpy as np
        from unittest import mock
        import dexmani_real.teleop.loop as loop
        harness = self

        class Rate:
            def wait(self):
                harness.now += 0.1
                harness.frame += 1
                if harness.frame > len(steps):
                    harness.shared.is_running.value = False
                    harness.controls = []
                else:
                    steps[harness.frame - 1](harness)

            def reset(self):
                pass

        def poll(timeout=0):
            if timeout and decision is not None:
                return decision(harness)
            controls, harness.controls = harness.controls, []
            return controls

        keyboard = SimpleNamespace(poll=poll, drain_signal=lambda *a: None,
                                   stop=lambda: None, healthy=True, estop_latched=False)
        arm = np.zeros(1, dtype=[("qpos", "f8", (7,))])
        clock = SimpleNamespace(monotonic=lambda: self.now, perf_counter=lambda: self.now,
                                monotonic_ns=lambda: int(self.now * 1e9))

        def grid(*args, **kwargs):
            self.snapshots.append((self.frame, kwargs["teleop_active"], kwargs["recording_active"],
                                   self.shared.safety_state.value, self.shared.run_generation.value))
            return SimpleNamespace(recording_active=kwargs["recording_active"],
                                   arm_feedback_error_count=0, hand_disconnected_at_s=None,
                                   pause_reason=None, pause_released=True, keep_running=True)

        patches = dict(
            _load_control_resources=mock.Mock(return_value=(mock.Mock(), mock.Mock(), mock.Mock(), self.client)),
            _start_keyboard=mock.Mock(return_value=keyboard), AudioFeedback=mock.Mock(),
            read_arm_state_causal=mock.Mock(return_value=arm), read_hand_state_causal=mock.Mock(return_value=None),
            read_vr_frame_causal=lambda *a: {"recv_ts_ns": int(self.now * 1e9),
                                           "wrist_pos": np.zeros(3), "wrist_quat_wxyz": np.array([1., 0., 0., 0.])},
            LoopRate=mock.Mock(return_value=Rate()), time=clock,
            run_control_grid_tick=grid,
        )
        self.output = io.StringIO()
        with contextlib.ExitStack() as stack:
            for name, value in patches.items():
                stack.enter_context(mock.patch.object(loop, name, value))
            stack.enter_context(mock.patch.object(loop.signal, "signal"))
            stack.enter_context(contextlib.redirect_stdout(self.output))
            loop.teleop_loop(self.shared, self.config)


class TeleopInterruptedRecordingTest(unittest.TestCase):
    def test_retained_terminal_is_unpublished_and_logs_confirmed_path(self):
        from dexmani_real.runtime.operator_input import OperatorCommand as Command
        from dexmani_real.runtime.safety import SafetyState
        h = _TeleopRecordingHarness(self)
        def reject(h):
            h.add_rows()
            h.shared.safety_state.value = SafetyState.ARMED
        with self.assertLogs("dexmani_real", level="INFO") as logs:
            h.run([lambda h: setattr(h, "controls", [Command.BEGIN]), reject,
                   lambda h: h.finalize(), lambda h: None])
        output = h.output.getvalue()
        self.assertEqual(output.count("录制未发布"), 1)
        self.assertNotIn("录制已丢弃", output)
        self.assertNotIn("已保存", output)
        retained = h.root / "incomplete_episode_1"
        self.assertTrue((retained / "data.h5").is_file())
        retention_lines = [line for line in logs.output if "partial staging retained at" in line]
        self.assertEqual(len(retention_lines), 1)
        self.assertIn(str(retained), retention_lines[0])
        self.assertFalse(h.client.poll_stop().done)

    def test_controller_interruptions_retain_buffered_source_rows(self):
        import json
        import h5py
        from dexmani_real.recording.client import StopRecording
        from dexmani_real.runtime.operator_input import OperatorCommand as Command
        from dexmani_real.runtime.safety import SafetyState

        for event, reason in (("shutdown", "policy_shutdown"), ("fault", "hardware_fault"),
                              ("esc", "estop"), ("arm_rejected", "arm_command_rejected")):
            with self.subTest(event=event):
                h = _TeleopRecordingHarness(self)
                def interrupt(h):
                    h.add_rows()
                    if event == "shutdown":
                        h.shared.is_running.value = False
                    elif event == "fault":
                        h.shared.error_state.value = True
                    elif event == "esc":
                        h.controls = [Command.EMERGENCY_STOP]
                    else:
                        h.shared.safety_state.value = SafetyState.ARMED
                h.run([lambda h: setattr(h, "controls", [Command.BEGIN]), interrupt])
                h.finalize()
                result = h.client.poll_stop()
                stops = [m for m in h.messages if isinstance(m, StopRecording)]
                self.assertEqual(len(stops), 1)
                self.assertEqual(result.reason, reason)
                self.assertFalse(result.saved)
                self.assertTrue(h.recorder.resources_released)
                retained = h.root / "incomplete_episode_1"
                self.assertTrue(retained.is_dir())
                note = json.loads((retained / "failure_note.json").read_text())
                self.assertEqual(note["reason"], reason)
                with h5py.File(retained / "data.h5", "r") as f:
                    self.assertEqual(f["meta"].attrs["num_frames"], 2)
                    self.assertEqual(list(f["timestamp"][:]), [1., 2.])
                self.assertFalse((h.root / "episode_1").exists())

    def test_explicit_save_discard_and_quit_decisions(self):
        from dexmani_real.recording.client import StopRecording
        from dexmani_real.runtime.operator_input import OperatorCommand as Command
        for decision, save, retain, reason in (
            ("D", False, False, "discard"), ("S", True, False, "manual"),
            ("Q_D", False, False, "discard"), ("Q_S", True, False, "manual"),
            ("Q_ESC", False, True, "estop"), ("Q_shutdown", False, True, "policy_shutdown"),
            ("Q_timeout", False, False, "quit_timeout"),
        ):
            with self.subTest(decision=decision):
                h = _TeleopRecordingHarness(self)
                def stop(h):
                    h.add_rows()
                    h.controls = [{"D": Command.DISCARD, "S": Command.STOP}.get(decision, Command.QUIT)]
                def answer(h):
                    if decision == "Q_shutdown":
                        h.shared.is_running.value = False
                    elif decision == "Q_timeout":
                        h.now += h.config.runtime.policy.quit_save_timeout_s
                    return {"Q_D": [Command.DISCARD], "Q_S": [Command.STOP],
                            "Q_ESC": [Command.EMERGENCY_STOP]}.get(decision, [])
                h.run([lambda h: setattr(h, "controls", [Command.BEGIN]), stop], decision=answer)
                h.finalize()
                result = h.client.poll_stop()
                stops = [m for m in h.messages if isinstance(m, StopRecording)]
                self.assertEqual(len(stops), 1, "finally must not replace an existing STOP")
                self.assertEqual((result.saved, result.reason), (save, reason))
                self.assertEqual(bool(list(h.root.glob("incomplete_*"))), retain)
                self.assertEqual((h.root / "episode_1").exists(), save)


class KeyboardReleaseLoopTest(unittest.TestCase):
    def _run(self, script):
        """Drive actual jog/retry/release code, replacing device reads and FIFO capacity."""
        import contextlib
        import io
        import numpy as np
        from unittest import mock
        import dexmani_real.teleop.keyboard_session as keyboard
        from dexmani_real.config.experiment import resolve_experiment_config
        from dexmani_real.control.action import ActionCandidate
        from dexmani_real.control.publication import PreparedCommand, PublishResult
        from dexmani_real.runtime.safety import SafetyState, CommittedCommand
        from test_deployment_evidence import _fake_shared

        runtime = resolve_experiment_config()
        shared = _fake_shared()
        state = SimpleNamespace(frame=0, now=10., step={})
        shared.get_heartbeat = lambda *a: state.now
        qpos = np.asarray(runtime.arm.home_qpos, dtype=np.float64)
        workspace = keyboard._workspace(runtime)
        pose = SimpleNamespace(p=workspace.mean(axis=1), q=np.array([1., 0., 0., 0.]))
        planner = mock.Mock()
        planner.kin.compute_eef_pose_world.return_value = pose
        planner.solve_teleop_ik.return_value = SimpleNamespace(success=True, qpos=qpos + .01)
        planner.ik_mgr.nearest_equivalent_qpos.side_effect = lambda q, ref: q.copy()
        prepared, attempts, published, snapshots = [], [], [], {}
        case = self

        class Rate:
            def wait(self):
                snapshots[state.frame] = (int(shared.safety_state.value), int(shared.run_generation.value))
                state.frame += 1
                case.assertLessEqual(state.frame, len(script) + 1)
                state.now += 1. / runtime.keyboard_teleop.control_hz
                state.step = script[state.frame - 1] if state.frame <= len(script) else {"keys": ("q",)}
                if state.step.get("revoke"):
                    keyboard.revoke_motion(shared, SafetyState.ARMED)

            def reset(self):
                pass

        keys = SimpleNamespace(healthy=True,
            is_pressed=lambda k: k in state.step.get("keys", ()),
            pressed_keys=lambda: tuple(state.step.get("keys", ())))

        def feedback(*args, **kwargs):
            return keyboard._KeyboardFeedback(
                arm_state={"last_cmd_generation": shared.run_generation.value,
                           "last_cmd_accepted_sequence": state.step.get("ack", 0)},
                arm_qpos_rad=qpos.copy(), hand_qpos_rad=None, issue=None)

        def prepare(shared, q, **kwargs):
            candidate = ActionCandidate(run_generation=int(shared.run_generation.value),
                                        arm_qpos=q.copy())
            prepared.append(candidate)
            return PreparedCommand(candidate=candidate)

        def publish(shared, candidate, **kwargs):
            attempts.append((state.frame, candidate))
            self.assertEqual(candidate.run_generation, shared.run_generation.value)
            self.assertEqual(shared.safety_state.value, SafetyState.RUNNING)
            if state.step.get("full", False):
                return PublishResult(False, reason=keyboard.PUBLISH_REASON_FIFO_FULL, fifo_depth=4)
            published.append((state.frame, candidate))
            return PublishResult(True, command=CommittedCommand(
                run_generation=candidate.run_generation, sequence=len(published)))

        def home(shared, runtime, planner, keys, current, **kwargs):
            keyboard.revoke_motion(shared, SafetyState.ARMED)
            return keyboard._keyboard_command_anchor(planner, current)

        with contextlib.ExitStack() as stack:
            patches = dict(read_initial_arm=mock.Mock(return_value={"qpos": qpos}),
                _read_keyboard_feedback=feedback, LoopRate=mock.Mock(return_value=Rate()),
                time=SimpleNamespace(monotonic=lambda: state.now),
                prepare_joint_command=prepare, publish_command=publish,
                _run_keyboard_home=home)
            for name, value in patches.items():
                stack.enter_context(mock.patch.object(keyboard, name, value))
            anchor = stack.enter_context(mock.patch.object(keyboard, "_keyboard_command_anchor",
                                                          wraps=keyboard._keyboard_command_anchor))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            logs = stack.enter_context(self.assertLogs("dexmani_real", level="INFO"))
            result = keyboard._run_control_loop(shared, runtime, planner, mock.Mock(), keys,
                                                mock.Mock(is_alive=lambda: True), None, hand_enabled=False)
        return SimpleNamespace(prepared=prepared, attempts=attempts, published=published,
                               snapshots=snapshots, logs="\n".join(logs.output), result=result,
                               anchors=anchor.call_count, ik_calls=planner.solve_teleop_ik.call_count)

    def test_full_release_drops_before_capacity_returns(self):
        r = self._run([{"keys": ("w",), "full": True}, {"full": True}, {}, {},
                       {"keys": ("w",)}, {}])
        self.assertEqual([f for f, c in r.attempts if c is r.prepared[0]], [1])
        self.assertEqual(len(r.published), 1)
        self.assertIs(r.published[0][1], r.prepared[1])
        self.assertEqual(r.logs.count("[DROP]"), 1)
        self.assertIn("reason=release", r.logs)
        self.assertGreaterEqual(r.anchors, 2)

    def test_unacked_predecessor_uses_original_release_timeout(self):
        from dexmani_real.runtime.safety import SafetyState
        r = self._run([{"keys": ("w",)}, {"keys": ("w",), "full": True}]
                      + [{"full": True}] * 8 + [{}])
        self.assertEqual(len(r.published), 1)
        self.assertIs(r.published[0][1], r.prepared[0])
        self.assertEqual(r.snapshots[4][0], SafetyState.RUNNING)
        self.assertEqual(r.snapshots[8][0], SafetyState.RUNNING)
        self.assertEqual(r.snapshots[9][0], SafetyState.ARMED)
        self.assertIn("final sequence=1 was not accepted", r.logs)
        self.assertEqual(r.logs.count("[DROP]"), 1)

    def test_accepted_or_absent_predecessor_releases_without_wait(self):
        from dexmani_real.runtime.safety import SafetyState
        for predecessor in (False, True):
            with self.subTest(predecessor=predecessor):
                prefix = [{"keys": ("w",)}] if predecessor else []
                r = self._run(prefix + [{"keys": ("w",), "full": True},
                                        {"ack": 1, "full": True}, {"ack": 1, "full": True}, {}])
                self.assertEqual(r.snapshots[len(prefix) + 3][0], SafetyState.ARMED)
                self.assertNotIn("was not accepted", r.logs)
                self.assertEqual(r.logs.count("[DROP]"), 1)

    def test_held_key_retries_same_candidate_once_without_new_ik(self):
        r = self._run([{"keys": ("w",), "full": True}] * 3 + [{"keys": ("w",)}])
        self.assertEqual(len(r.prepared), 1)
        self.assertEqual(r.ik_calls, 1)
        self.assertEqual(len(r.published), 1)
        self.assertTrue(all(c is r.prepared[0] for _, c in r.attempts))
        self.assertNotIn("[DROP]", r.logs)

    def test_short_release_retains_without_submitting_until_repress(self):
        r = self._run([{"keys": ("w",), "full": True}, {}, {"keys": ("w",)}])
        self.assertEqual([f for f, _ in r.attempts], [1, 3])
        self.assertEqual(len(r.prepared), 1)
        self.assertEqual(len(r.published), 1)
        self.assertIs(r.published[0][1], r.prepared[0])

    def test_terminal_home_and_epoch_boundaries_drop_once(self):
        for boundary in ("q", "esc", "r", "epoch"):
            with self.subTest(boundary=boundary):
                event = {"revoke": True, "keys": ("w",)} if boundary == "epoch" else {"keys": (boundary,)}
                r = self._run([{"keys": ("w",), "full": True}, event, {}, {"keys": ("w",)}])
                self.assertFalse(any(c is r.prepared[0] for _, c in r.published))
                self.assertEqual(r.logs.count("[DROP]"), 1)
                if boundary in ("r", "epoch"):
                    self.assertEqual(len(r.published), 1)
                    self.assertIs(r.published[0][1], r.prepared[1])


class TeleopCapacityOwnershipTest(unittest.TestCase):
    def test_old_terminal_arriving_on_first_poll_after_b_cannot_pause_it(self):
        from dexmani_real.runtime.operator_input import OperatorCommand as Command
        from dexmani_real.runtime.safety import SafetyState
        h = _TeleopRecordingHarness(self, max_frames=2)
        begin = lambda h: setattr(h, "controls", [Command.BEGIN])
        h.run([begin, lambda h: h.add_rows(), begin, lambda h: h.finalize(), lambda h: None])
        snapshots = {s[0]: s[1:] for s in h.snapshots}
        self.assertEqual(snapshots[4][:3], (True, False, SafetyState.RUNNING))
        self.assertEqual(snapshots[4], snapshots[5])
        self.assertEqual(h.output.getvalue().count("已达到最大录制时长"), 1)

    def test_old_pending_and_done_cannot_pause_new_unrecorded_run(self):
        import threading
        from unittest import mock
        from dexmani_real.recording.client import StopRecording
        from dexmani_real.runtime.operator_input import OperatorCommand as Command
        from dexmani_real.runtime.safety import SafetyState
        h = _TeleopRecordingHarness(self, max_frames=2)
        begin = lambda h: setattr(h, "controls", [Command.BEGIN])
        stop = lambda h: setattr(h, "controls", [Command.STOP])
        entered, release = threading.Event(), threading.Event()
        def capacity_with_active_finalizer(h):
            h.add_rows()
            close = h.recorder._camera_writer.close
            def blocked_close(*args, **kwargs):
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("test did not release finalizer")
                return close(*args, **kwargs)
            patch = mock.patch.object(h.recorder._camera_writer, "close", side_effect=blocked_close)
            patch.start()
            self.addCleanup(patch.stop)
            h.begin_finalization()
            self.assertTrue(entered.wait(2))
        def finish(h):
            self.assertTrue(h.session.pending_finalization.thread.is_alive())
            release.set()
            h.finalize()
        try:
            with self.assertLogs("dexmani_real", level="INFO") as logs:
                h.run([begin, capacity_with_active_finalizer, begin, lambda h: None,
                       finish, lambda h: None, stop,
                       begin, lambda h: h.add_rows(1), stop, lambda h: h.finalize(), lambda h: None])
        finally:
            release.set()
        snapshots = {s[0]: s[1:] for s in h.snapshots}
        # poll A/max_frames -> B -> repeated pending -> delayed terminal -> poll again.
        for frame in (4, 5, 6):
            self.assertEqual(snapshots[frame][:3], (True, False, SafetyState.RUNNING))
        self.assertEqual(snapshots[4][3], snapshots[5][3])
        self.assertEqual(snapshots[4][3], snapshots[6][3])
        self.assertEqual(snapshots[9][:3], (True, True, SafetyState.RUNNING))
        self.assertEqual(h.starts, 2)
        stops = [m for m in h.messages if isinstance(m, StopRecording)]
        self.assertEqual([(m.save, m.reason) for m in stops], [(True, "max_frames"), (True, "manual")])
        self.assertEqual(h.output.getvalue().count("已达到最大录制时长"), 1)
        records = [line for line in logs.output if "[RECORD] reason=" in line]
        self.assertEqual(len(records), 2)
        self.assertEqual(sum("reason=max_frames" in line for line in records), 1)
        self.assertTrue((h.root / "episode_1" / "data.h5").is_file())
        self.assertTrue((h.root / "episode_2" / "data.h5").is_file())
        self.assertFalse(h.client.poll_stop().done)

    def test_first_capacity_terminal_without_pending_observation_still_pauses(self):
        from dexmani_real.runtime.operator_input import OperatorCommand as Command
        from dexmani_real.runtime.safety import SafetyState
        h = _TeleopRecordingHarness(self, max_frames=2)
        def finish_capacity(h):
            h.add_rows()
            h.finalize()
            self.assertTrue(h.client.stop_pending)  # terminal is still queued
        with self.assertLogs("dexmani_real", level="INFO") as logs:
            h.run([lambda h: setattr(h, "controls", [Command.BEGIN]), finish_capacity, lambda h: None])
        self.assertEqual(h.snapshots[0][1:4], (False, False, SafetyState.ARMED))
        self.assertEqual(h.output.getvalue().count("已达到最大录制时长"), 1)
        self.assertEqual(sum("[RECORD] reason=max_frames" in line for line in logs.output), 1)
        self.assertFalse(h.client.poll_stop().done)


if __name__ == "__main__":
    unittest.main()
