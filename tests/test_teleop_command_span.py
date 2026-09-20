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


@unittest.skipIf(
    _IMPORT_ERROR is not None, f"dependencies unavailable: {_IMPORT_ERROR}"
)
class PauseBoundarySpanTest(unittest.TestCase):
    def _controller(self, action_id: int):
        return SimpleNamespace(
            pending_publish=SimpleNamespace(
                candidate=SimpleNamespace(action_id=action_id)
            )
        )

    def test_pause_boundary_drops_the_retained_candidate_visibly(self):
        tracker = PublishWaitTracker("teleop")
        tracker.note_full(8, 813)
        self.assertTrue(tracker.waiting)
        controller = self._controller(813)

        with self.assertLogs(
            "dexmani_real", level="WARNING"
        ) as logs:
            close_publish_span(
                controller, SimpleNamespace(fifo_wait=tracker), "pause_boundary"
            )

        text = "\n".join(logs.output)
        self.assertIn("[DROP] teleop action=813", text)
        self.assertIn("pause_boundary", text)
        self.assertEqual(text.count("[DROP]"), 1)
        self.assertIsNone(controller.pending_publish)
        self.assertFalse(tracker.waiting)
        # The span is closed, so the next successful commit reports no wait
        # that never resumed.
        with self.assertNoLogs("dexmani_real.control.publication", level="INFO"):
            tracker.note_committed()

    def test_boundary_without_a_candidate_still_closes_a_stale_span(self):
        tracker = PublishWaitTracker("teleop")
        tracker.note_full(8, 901)
        controller = SimpleNamespace(pending_publish=None)

        close_publish_span(
            controller, SimpleNamespace(fifo_wait=tracker), "pause_boundary"
        )
        self.assertFalse(tracker.waiting)
        # A later FULL span is announced again instead of being swallowed.
        with self.assertLogs("dexmani_real.control.publication", level="WARNING") as logs:
            tracker.note_full(8, 902)
        self.assertIn("keep_action=902", "\n".join(logs.output))

    def test_boundary_with_no_span_is_a_no_op(self):
        tracker = PublishWaitTracker("teleop")
        controller = SimpleNamespace(pending_publish=None)
        with self.assertNoLogs("dexmani_real.control.publication", level="WARNING"):
            close_publish_span(
                controller, SimpleNamespace(fifo_wait=tracker), "pause_release"
            )
        self.assertFalse(tracker.waiting)


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

    def finalize(self):
        stop = self.shared.record_control_q.get_nowait()
        self.session._handle_stop(stop)
        self.session._drain_samples()
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


if __name__ == "__main__":
    unittest.main()
