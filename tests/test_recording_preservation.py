"""Offline tests for raw-episode preservation contracts (task T1).

Covers the recorder transaction and the RecorderIO STOP classification:

* a ``camera_stall`` stop saves the technically complete collected prefix
  (below ``min_frames``) with its terminal reason instead of destroying it;
* a real writer/file corruption never publishes as valid raw — after all
  writers confirmed release, the owned partial staging is retained under an
  explicit ``incomplete_*`` name with a failure note;
* an explicit user discard (clean ``save=False``) stays destructive;
* RecorderIO no longer forces ``camera_stall`` into its error branch while
  genuine corruption reasons still override ``save=True``.

These tests exercise the real ``EpisodeRecorder``/``CameraStreamWriter``
transaction on tiny synthetic payloads. They never open hardware or an SDK.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from dexmani_real.recording.client import StopRecording
from dexmani_real.recording.frame import EpisodeFrame
from dexmani_real.recording.io_worker import (
    RecorderIOConfig,
    _RecorderIOSession,
)
from dexmani_real.recording.recorder import (
    EpisodeFinalizationError,
    EpisodeRecorder,
)
from dexmani_real.recording.frame import build_episode_frame
from dexmani_real.ipc.schema import ARM_STATE_DTYPE, HAND_STATE_DTYPE
from dexmani_real.recording.storage.camera_writer import CameraStreamWriterConfig
from dexmani_real.recording.storage.reader import EpisodeReader

_RGB_SHAPE = (8, 8, 3)
_DEPTH_SHAPE = (8, 8)
_CONTROL_HZ = 16.0
_ROT6D_IDENTITY = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
_VR_FRAME = {
    "wrist_pos": np.zeros(3, dtype=np.float64),
    "wrist_quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    "landmarks": np.zeros((21, 3), dtype=np.float64),
}


def _frame(timestamp_s: float) -> EpisodeFrame:
    arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
    hand = np.zeros(1, dtype=HAND_STATE_DTYPE)
    arm["connected"] = hand["connected"] = 1
    hand["tactile_aggregate_valid"] = hand["tactile_dense_valid"] = 1
    action = {
        "action_arm_joint_sent": np.zeros(7),
        "action_hand_joint": np.zeros(12),
        "action_arm_ee": np.concatenate((np.zeros(3), _ROT6D_IDENTITY)),
    }
    return build_episode_frame(arm, hand, action, _VR_FRAME, timestamp_s=timestamp_s)


def _make_recorder(data_dir: Path, *, min_frames: int = 50) -> EpisodeRecorder:
    return EpisodeRecorder(
        data_dir=str(data_dir),
        control_hz=_CONTROL_HZ,
        min_frames=min_frames,
        camera_writer_config=CameraStreamWriterConfig(
            rgb_shape=_RGB_SHAPE,
            depth_shape=_DEPTH_SHAPE,
            fps=_CONTROL_HZ,
            queue_size=8,
        ),
    )


def _add_frames(recorder: EpisodeRecorder, count: int) -> None:
    for index in range(count):
        added = recorder.add_frame(_frame(float(index + 1)))
        assert added, f"frame {index} was not accepted"


class RecorderPrefixPreservationTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_camera_stall_prefix_saves_below_min_frames(self):
        """A technically complete short prefix is a valid saved episode."""
        recorder = _make_recorder(self.root, min_frames=50)
        self.assertTrue(
            recorder.start_episode(task_label="t1", episode_name="episode_t1_stall")
        )
        _add_frames(recorder, 3)
        path = recorder.finish_episode(save=True, reason="camera_stall")
        self.assertEqual(path, str(self.root / "episode_t1_stall"))

        published = self.root / "episode_t1_stall"
        self.assertTrue(published.is_dir())
        self.assertTrue(recorder.last_finish_saved)
        self.assertTrue(recorder.resources_released)
        with EpisodeReader(published) as reader:
            meta = reader.h5f["meta"].attrs
            self.assertEqual(int(meta["num_frames"]), 3)
            self.assertEqual(str(meta["stop_reason"]), "camera_stall")
            self.assertFalse(bool(meta["min_frames_met"]))
            self.assertFalse(bool(meta["truncated"]))
            reader.require_valid(purpose="test")
        # No failure artifacts for a successful save.
        self.assertEqual(list(self.root.glob("incomplete_*")), [])
        self.assertEqual(list(self.root.glob("*.aborted.json")), [])

    def test_writer_corruption_retains_incomplete_staging(self):
        """A failed transaction keeps its closed partial staging, unpublished."""
        recorder = _make_recorder(self.root, min_frames=1)
        self.assertTrue(
            recorder.start_episode(task_label="t1", episode_name="episode_t1_fail")
        )
        _add_frames(recorder, 3)
        # Latch a real writer failure; close() must then raise.
        assert recorder._camera_writer is not None
        recorder._camera_writer._set_error("simulated encoder failure")

        with self.assertRaises(EpisodeFinalizationError):
            recorder.finish_episode(save=True, reason="manual")

        # Never published as a valid raw episode.
        self.assertFalse(recorder.last_finish_saved)
        self.assertFalse((self.root / "episode_t1_fail").exists())
        retained = self.root / "incomplete_episode_t1_fail"
        self.assertTrue(retained.is_dir(), "failed staging must be retained")
        note = json.loads((retained / "failure_note.json").read_text(encoding="utf-8"))
        self.assertEqual(note["status"], "incomplete")
        self.assertEqual(note["episode"], "episode_t1_fail")
        self.assertEqual(note["frame_count_before_failure"], 3)
        self.assertIn("simulated encoder failure", note["error"])
        # Failure provenance is also visible at the data-dir level.
        manifests = list(self.root.glob("episode_t1_fail*.aborted.json"))
        self.assertEqual(len(manifests), 1)
        self.assertTrue(recorder.resources_released)

    def test_automatic_failure_retains_prefix_when_finalization_succeeds(self):
        """An automatic failure keeps the collected prefix.

        The finalization itself completes cleanly here, so no exception marks
        the transaction: the failure identity carried by the caller is what
        separates an automatic failure (retain the closed staging) from an
        explicit operator discard (destructive).
        """
        recorder = _make_recorder(self.root, min_frames=1)
        self.assertTrue(
            recorder.start_episode(task_label="t1", episode_name="episode_t1_auto")
        )
        _add_frames(recorder, 3)
        path = recorder.finish_episode(
            save=False,
            reason="sample_ring_overflow",
            failure_note="RecorderIO sample_ring_overflow: sample ring full",
        )
        self.assertEqual(path, str(self.root / "episode_t1_auto"))
        self.assertFalse(recorder.last_finish_saved)
        self.assertFalse((self.root / "episode_t1_auto").exists())
        retained = self.root / "incomplete_episode_t1_auto"
        self.assertTrue(
            retained.is_dir(), "automatic-failure staging must be retained"
        )
        note = json.loads((retained / "failure_note.json").read_text(encoding="utf-8"))
        self.assertEqual(note["status"], "incomplete")
        self.assertEqual(note["frame_count_before_failure"], 3)
        self.assertIn("sample_ring_overflow", note["error"])
        self.assertTrue(recorder.resources_released)

    def test_zero_row_automatic_failure_keeps_no_empty_staging(self):
        """An empty staging is junk, not a partial prefix worth retaining."""
        recorder = _make_recorder(self.root, min_frames=1)
        self.assertTrue(
            recorder.start_episode(
                task_label="t1", episode_name="episode_t1_zero_fail"
            )
        )
        recorder.finish_episode(
            save=False, reason="sample_write_error", failure_note="write failed"
        )
        self.assertEqual(list(self.root.glob("incomplete_*")), [])
        self.assertEqual(list(self.root.glob(".tmp_*")), [])
        self.assertTrue(recorder.resources_released)

    def test_explicit_user_discard_stays_destructive(self):
        """A clean save=False discard removes staging and publishes nothing."""
        recorder = _make_recorder(self.root, min_frames=1)
        self.assertTrue(
            recorder.start_episode(
                task_label="t1", episode_name="episode_t1_discard"
            )
        )
        _add_frames(recorder, 3)
        recorder.finish_episode(save=False, reason="discard")

        self.assertFalse(recorder.last_finish_saved)
        self.assertFalse((self.root / "episode_t1_discard").exists())
        self.assertEqual(list(self.root.glob("incomplete_*")), [])
        self.assertEqual(list(self.root.glob(".tmp_*")), [])
        manifests = list(self.root.glob("episode_t1_discard*.aborted.json"))
        self.assertEqual(len(manifests), 1)
        payload = json.loads(manifests[0].read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "aborted")
        self.assertEqual(payload["reason"], "discard")
        self.assertTrue(recorder.resources_released)


    def test_zero_row_save_downgrades_to_clean_discard(self):
        """Zero source rows never masquerade as a published success."""
        recorder = _make_recorder(self.root, min_frames=1)
        self.assertTrue(
            recorder.start_episode(task_label="t1", episode_name="episode_t1_zero")
        )
        path = recorder.finish_episode(save=True, reason="camera_stall")
        self.assertEqual(path, str(self.root / "episode_t1_zero"))

        self.assertFalse(recorder.last_finish_saved)
        self.assertFalse((self.root / "episode_t1_zero").exists())
        # Clean downgrade: no incomplete staging, no failure manifest noise
        # beyond the ordinary aborted-episode record.
        self.assertEqual(list(self.root.glob("incomplete_*")), [])
        self.assertEqual(list(self.root.glob(".tmp_*")), [])
        manifests = list(self.root.glob("episode_t1_zero*.aborted.json"))
        self.assertEqual(len(manifests), 1)
        payload = json.loads(manifests[0].read_text(encoding="utf-8"))
        self.assertEqual(payload["reason"], "camera_stall")
        self.assertEqual(payload["frame_count_before_abort"], 0)
        self.assertTrue(recorder.resources_released)


class _FakeValue:
    def __init__(self, initial: int = 0) -> None:
        self.value = initial


class _FakeRing:
    maxlen = 64

    def __init__(self, latest_sequence: int) -> None:
        self._latest = int(latest_sequence)

    @property
    def latest_sequence(self) -> int:
        return self._latest

    def read_sequence(self, sequence: int):
        if 1 <= sequence <= self._latest:
            return ([b"record"], 0, sequence)
        return None


class _FakeShared:
    def __init__(self, latest_sequence: int) -> None:
        self.record_sample_ring = _FakeRing(latest_sequence)
        self.recorder_consumed_sequence = _FakeValue(0)


class _FakeRecorder:
    def __init__(self) -> None:
        self.is_recording = True
        self.frame_count = 3
        self.max_frames_reached = False
        self.episode_path = "episode_fake"
        self.camera_writer_error = None
        self.last_finish_saved = False
        self.finished: list[tuple[bool, str]] = []
        self.failure_notes: list[str] = []

    @property
    def resources_released(self) -> bool:
        return True

    def add_frame(self, frame: EpisodeFrame) -> bool:
        return True

    def finish_episode(self, save: bool, reason: str, *, failure_note: str = ""):
        self.finished.append((bool(save), reason))
        self.failure_notes.append(failure_note)
        self.is_recording = False
        self.last_finish_saved = bool(save)
        return self.episode_path


def _drain_stop(shared: _FakeShared, stop: StopRecording) -> _FakeRecorder:
    """Run one RecorderIO STOP→drain→finalization cycle on fakes."""
    config = RecorderIOConfig(
        data_dir="unused",
        max_frames=100,
        control_hz=_CONTROL_HZ,
        min_frames=1,
    )
    recorder = _FakeRecorder()
    session = _RecorderIOSession(shared, config, recorder, last_sample_sequence=0)
    frame = EpisodeFrame(timestamp_s=1.0, data={})
    with mock.patch(
        "dexmani_real.recording.io_worker.decode_record_sample", return_value=frame
    ):
        session._handle_stop(stop)
        session._drain_samples()
    pending = session.pending_finalization
    assert pending is not None and pending.thread is not None
    pending.thread.join(timeout=10.0)
    return recorder


class RecorderIOStopClassificationTest(unittest.TestCase):
    def test_camera_stall_stop_honors_save(self):
        """camera_stall is evidence degradation: the caller's save decision wins."""
        shared = _FakeShared(latest_sequence=3)
        recorder = _drain_stop(
            shared, StopRecording(save=True, reason="camera_stall", through_sequence=3)
        )
        self.assertEqual(recorder.finished, [(True, "camera_stall")])
        # An evidence-preserving stop carries no failure identity, so the
        # recorder would treat a non-publishing outcome as an explicit discard.
        self.assertEqual(recorder.failure_notes, [""])

    def test_corruption_reasons_still_override_save(self):
        """Genuine transport/writer corruption is never published as valid raw."""
        for reason in ("sample_ring_overflow", "camera_writer_error"):
            with self.subTest(reason=reason):
                shared = _FakeShared(latest_sequence=3)
                recorder = _drain_stop(
                    shared, StopRecording(save=True, reason=reason, through_sequence=3)
                )
                self.assertEqual(recorder.finished, [(False, reason)])
                # The automatic-failure identity reaches the transaction, which
                # is what makes the closed prefix staging retained rather than
                # destroyed (TASKBOOK T1: an automatic failure is never
                # conflated with an explicit operator discard).
                self.assertEqual(len(recorder.failure_notes), 1)
                self.assertIn(reason, recorder.failure_notes[0])


if __name__ == "__main__":
    unittest.main()

class RecorderFinalizerIsolationTest(unittest.TestCase):
    def test_real_loop_timeout_and_shutdown_preserve_active_writer(self):
        import threading
        from queue import Queue
        from types import SimpleNamespace
        import dexmani_real.recording.io_worker as io
        entered, release = threading.Event(), threading.Event()
        shared = SimpleNamespace(
            is_running=_FakeValue(True), error_state=_FakeValue(False),
            evidence_failed=_FakeValue(False), quit_requested=_FakeValue(False),
            run_generation=_FakeValue(7), recorder_consumed_sequence=_FakeValue(0),
            record_control_q=Queue(), record_result_q=Queue(),
            set_ready=lambda *a: None, set_heartbeat=lambda *a: None,
        )
        class Writer(_FakeRecorder):
            @property
            def resources_released(self):
                return release.is_set()
            def finish_episode(self, *args, **kwargs):
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("test writer was not released")
                return super().finish_episode(*args, **kwargs)
        recorder = Writer()
        create = io._RecorderIOSession.create
        sessions = []
        def prepare(*args):
            session = create(*args)
            sessions.append(session)
            session._begin_finalization(save=True, reason="operator")
            self.assertTrue(entered.wait(2))
            session.pending_finalization.started_monotonic_s = -1000
            return session
        try:
            with mock.patch.object(io, "_create_episode_recorder", return_value=recorder), mock.patch.object(io._RecorderIOSession, "create", side_effect=prepare):
                with self.assertRaisesRegex(RuntimeError, "recording failure"):
                    io.recorder_io_loop(shared, RecorderIOConfig(data_dir="unused", max_frames=2, control_hz=10, min_frames=1))
            self.assertTrue(shared.evidence_failed.value)
            self.assertFalse(shared.error_state.value)
            self.assertFalse(shared.quit_requested.value)
            self.assertEqual(shared.run_generation.value, 7)
            self.assertTrue(sessions[0].pending_finalization.thread.is_alive())
            self.assertFalse(sessions[0].pending_finalization.thread.daemon)
            self.assertFalse(recorder.resources_released)
        finally:
            release.set()
            for session in sessions:
                session.pending_finalization.thread.join(2)

    def test_unreleased_finished_writer_is_evidence_failure(self):
        from queue import Queue
        from types import SimpleNamespace
        import dexmani_real.recording.io_worker as io
        shared = SimpleNamespace(is_running=_FakeValue(True), error_state=_FakeValue(False),
            evidence_failed=_FakeValue(False), recorder_consumed_sequence=_FakeValue(0),
            record_control_q=Queue(), record_result_q=Queue(),
            set_ready=lambda *a: None, set_heartbeat=lambda *a: None)
        class Writer(_FakeRecorder):
            @property
            def resources_released(self):
                return False
        create = io._RecorderIOSession.create
        def prepare(*args):
            session = create(*args)
            session._begin_finalization(save=True, reason="operator")
            session.pending_finalization.thread.join(2)
            return session
        with mock.patch.object(io, "_create_episode_recorder", return_value=Writer()), mock.patch.object(io._RecorderIOSession, "create", side_effect=prepare):
            with self.assertRaisesRegex(RuntimeError, "recording failure"):
                io.recorder_io_loop(shared, RecorderIOConfig(data_dir="unused", max_frames=2, control_hz=10, min_frames=1))
        self.assertTrue(shared.evidence_failed.value)
        self.assertFalse(shared.error_state.value)

class RecorderCapacityPathTest(unittest.TestCase):
    def test_real_client_sample_stop_and_writer_prefix(self):
        from queue import Queue
        from types import SimpleNamespace
        from dexmani_real.ipc.schema import make_record_sample_dtype
        from dexmani_real.recording.client import RecorderClient
        from test_deployment_evidence import _RunnerTest, _fake_shared
        class Ring:
            dtype = make_record_sample_dtype(_RGB_SHAPE, _DEPTH_SHAPE)
            maxlen = 4
            latest_sequence = 0
            def __init__(self):
                self.frames = []
            def write(self, frame):
                self.frames.append(frame.copy())
                self.latest_sequence += 1
                return self.latest_sequence
            def read_sequence(self, sequence):
                return self.frames[sequence - 1], 0, sequence
        shared = _fake_shared()
        shared.record_sample_ring = Ring()
        shared.record_control_q, shared.record_result_q = Queue(), Queue()
        shared.recorder_consumed_sequence = _FakeValue(0)
        shared.set_heartbeat = lambda *a: None
        shared.is_recording.value = True
        client = RecorderClient(shared)
        client._recording, client._max_frames = True, 2
        with tempfile.TemporaryDirectory() as directory:
            recorder = _make_recorder(Path(directory), min_frames=1)
            self.assertTrue(recorder.start_episode(task_label="t1", episode_name="capacity"))
            session = _RecorderIOSession(shared, RecorderIOConfig(data_dir=directory,
                max_frames=2, control_hz=_CONTROL_HZ, min_frames=1), recorder)
            for index in range(2):
                self.assertTrue(client.add_frame(_frame(index + 1.)))
            self.assertFalse(client.is_recording)
            self.assertTrue(client.stop_pending)
            self.assertFalse(client.add_frame(_frame(3.)))
            stop = shared.record_control_q.get_nowait()
            self.assertEqual(stop.reason, "max_frames")
            session._handle_stop(stop)
            session._drain_samples()
            session.pending_finalization.thread.join(5)
            session._poll_finalization()
            runner = _RunnerTest()._runner(shared=shared, recorder=client, run_started_ns=1)
            self.assertTrue(runner._poll_recorder())
            self.assertEqual(runner.completed_trials, 0)
            self.assertEqual(runner.saved_episodes, 1)
            # Rapid replans after capacity exhaustion must not become a hidden
            # miss threshold or reopen the finished writer.
            from dexmani_real.deployment.runner import _RejectKind
            from collections import deque
            runner.run_generation = int(shared.run_generation.value)
            reference = np.ones(7)
            runner.previous_arm_command_qpos = reference
            for index in range(12):
                runner.actions = deque([np.zeros(19), np.zeros(19)])
                runner._handle_recoverable_miss("workspace" if index % 2 else "ik",
                    raw_action=np.zeros(19), reject_kind=_RejectKind.SAFETY if index % 2 else _RejectKind.IK)
                self.assertEqual(runner.completed_trials, 0)
                self.assertEqual(runner.run_started_ns, 1)
                self.assertIs(runner.previous_arm_command_qpos, reference)
            self.assertEqual(shared.record_sample_ring.latest_sequence, 2)
            runner._poll_recorder()
            self.assertFalse(client.join_stop().saved)
            self.assertEqual(runner.saved_episodes, 1)
            with EpisodeReader(str(Path(directory) / "capacity")) as reader:
                self.assertEqual(int(reader.h5f["meta"].attrs["num_frames"]), 2)
                reader.require_valid(purpose="test")

class RecorderShutdownExceptionTest(unittest.TestCase):
    def test_actual_shutdown_exception_does_not_set_motion_fault(self):
        from queue import Queue
        from types import SimpleNamespace
        import dexmani_real.recording.io_worker as io
        class Recorder(_FakeRecorder):
            is_recording = False
            @property
            def resources_released(self):
                raise OSError("writer close verification failed")
        recorder = Recorder()
        recorder.is_recording = False
        shared = SimpleNamespace(is_running=_FakeValue(True), error_state=_FakeValue(False),
            evidence_failed=_FakeValue(False), recorder_consumed_sequence=_FakeValue(0),
            record_control_q=Queue(), record_result_q=Queue(),
            set_ready=lambda *a: None, set_heartbeat=lambda *a: None)
        shared.record_control_q.put(object())  # actual step rejects unknown control
        with mock.patch.object(io, "_create_episode_recorder", return_value=recorder):
            with self.assertRaisesRegex(RuntimeError, "recording failure"):
                io.recorder_io_loop(shared, RecorderIOConfig(data_dir="unused",
                    max_frames=2, control_hz=10, min_frames=1))
        self.assertTrue(shared.evidence_failed.value)
        self.assertFalse(shared.error_state.value)


class InterruptedRetentionBoundaryTest(unittest.TestCase):
    def test_policy_automatic_stop_sends_retention_through_real_client(self):
        from queue import Queue
        from types import SimpleNamespace
        from dexmani_real.deployment.runner import PolicyRunner
        from dexmani_real.recording.client import RecorderClient
        shared = SimpleNamespace(record_control_q=Queue(),
                                 record_sample_ring=SimpleNamespace(latest_sequence=2))
        runner = PolicyRunner.__new__(PolicyRunner)
        runner._pending_stop_reason = None
        runner.recorder = RecorderClient(shared)
        runner.recorder._recording = True
        runner._stop_recording_capture(save=False, reason="hardware_fault")
        self.assertEqual(shared.record_control_q.get_nowait(),
                         StopRecording(False, "hardware_fault", 2, retain_partial=True))
        runner._stop_recording_capture(save=False, reason="policy_shutdown")
        self.assertTrue(shared.record_control_q.empty())

    def test_active_writer_stays_in_staging_until_confirmed_closed(self):
        import threading
        entered, release = threading.Event(), threading.Event()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = _make_recorder(root, min_frames=1)
            self.assertTrue(recorder.start_episode(task_label="test", episode_name="episode_active"))
            _add_frames(recorder, 2)
            self.assertEqual(len(recorder._pending_rows), 2)
            self.assertLess(2, recorder._flush_interval)
            staging = Path(recorder._temp_dir)
            close = recorder._camera_writer.close
            errors = []

            def blocked_close(*args, **kwargs):
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("test did not release writer")
                return close(*args, **kwargs)

            def finish():
                try:
                    recorder.finish_episode(False, "policy_shutdown", failure_note="controller interrupted")
                except Exception as exc:
                    errors.append(exc)

            with mock.patch.object(recorder._camera_writer, "close", side_effect=blocked_close):
                thread = threading.Thread(target=finish)
                thread.start()
                try:
                    self.assertTrue(entered.wait(2))
                    self.assertFalse(recorder.resources_released)
                    self.assertTrue(staging.is_dir())
                    self.assertEqual(list(root.glob("incomplete_*")), [])
                    self.assertFalse((root / "episode_active").exists())
                finally:
                    release.set()
                    thread.join(5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertTrue(recorder.resources_released)
            self.assertFalse(staging.exists())
            with EpisodeReader(root / "incomplete_episode_active") as reader:
                self.assertEqual(reader.h5f["meta"].attrs["num_frames"], 2)
                np.testing.assert_array_equal(reader.h5f["timestamp"][:], [1., 2.])

    def test_old_stop_signature_and_single_stop_intent_are_preserved(self):
        from queue import Queue
        from types import SimpleNamespace
        from dexmani_real.recording.client import RecorderClient
        old_message = StopRecording(False, "discard", 2)
        self.assertFalse(old_message.retain_partial)
        shared = SimpleNamespace(record_control_q=Queue(),
                                 record_sample_ring=SimpleNamespace(latest_sequence=2))
        client = RecorderClient(shared)
        client._recording = True
        client.stop_episode(False, "discard")
        client.stop_episode(False, "policy_shutdown", retain_partial=True)
        self.assertEqual(shared.record_control_q.get_nowait(), old_message)
        self.assertTrue(shared.record_control_q.empty())


class FrameSemanticsTest(unittest.TestCase):
    def test_owned_frame_preserves_submitted_joints_and_cartesian_intent(self):
        arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
        hand = np.zeros(1, dtype=HAND_STATE_DTYPE)
        submitted = np.arange(7, dtype=float)
        intent = np.arange(9, dtype=float) + 100
        action = {"action_arm_joint_sent": submitted,
                  "action_hand_joint": np.zeros(12), "action_arm_ee": intent}
        rgb = np.ones((2, 3, 3), np.uint8)
        frame = build_episode_frame(arm, hand, action, _VR_FRAME,
                                    timestamp_s=1.5, camera_frame={"rgb": rgb})
        submitted[:] = -1
        intent[:] = -1
        arm["qpos"] = 9
        rgb[:] = 0
        np.testing.assert_array_equal(frame.data["action_arm_joint_sent"], np.arange(7))
        np.testing.assert_array_equal(frame.data["action_arm_ee"], np.arange(9) + 100)
        np.testing.assert_array_equal(frame.data["arm_qpos"], np.zeros(7))
        np.testing.assert_array_equal(frame.camera_rgb, np.ones((2, 3, 3), np.uint8))
        self.assertEqual(frame.timestamp_s, 1.5)

    def test_invalid_tactile_is_nan_not_zero_contact(self):
        arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
        hand = np.zeros(1, dtype=HAND_STATE_DTYPE)
        hand["tactile_aggregate_valid"] = 1
        hand["tactile_aggregate"] = 3
        hand["tactile_dense_valid"] = 0
        action = {"action_arm_joint_sent": np.zeros(7),
                  "action_hand_joint": np.zeros(12), "action_arm_ee": np.zeros(9)}
        frame = build_episode_frame(arm, hand, action, _VR_FRAME, timestamp_s=1.)
        self.assertTrue(frame.data["hand_contact_valid"])
        np.testing.assert_array_equal(frame.data["hand_contact"], np.full((5, 3), 3))
        self.assertFalse(frame.data["hand_tactile_force_valid"])
        self.assertTrue(np.isnan(frame.data["hand_tactile_force"]).all())
