"""真实小尺寸 HDF5／视频与受控 writer 故障，不接入设备。"""

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from dexmani_real.data.recorder import AsyncEpisodeRecorder, RecordingError
from dexmani_real.data.raw.writer import RawEpisodeWriter
from dexmani_real.robot.commands import DispatchResult, DispatchStatus, RobotCommand
from dexmani_real.utils.atomic_io import atomic_publish
from tests.offline_fixtures import make_row, recording_metadata


class RecorderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="dexmani-recorder-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.command = RobotCommand(1, np.zeros(7), np.zeros(12))
        self.dispatch = DispatchResult(arm=DispatchStatus.SUBMITTED, hand=DispatchStatus.UNKNOWN)

    def new_recorder(self, name="capture"):
        return AsyncEpisodeRecorder(self.root / name, control_hz=10)

    def start(self, recorder):
        recorder.start_episode(task_label="offline", collection_source="teleop", metadata=recording_metadata())

    def test_two_independent_captures_and_zero_rows(self):
        captures = [self.new_recorder(), self.new_recorder()]
        self.assertFalse((self.root / "capture").exists())
        self.assertIsNot(captures[0].queue, captures[1].queue)
        destinations = []
        for index, recorder in enumerate(captures):
            self.start(recorder)
            if index:
                row = make_row()
                row.arm["qpos"] = 0.4
                recorder.add_frame(row, self.command, self.dispatch)
                row.arm["qpos"] = 9
                row.camera["depth"][:] = 9
            saved = recorder.save_episode("operator", stop_result="confirmed")
            destinations.append(saved)
            with h5py.File(saved / "data.h5") as data:
                self.assertEqual(data["meta"].attrs["num_frames"], index)
                self.assertEqual(data["meta"].attrs["termination_reason"], "operator")
                if index:
                    np.testing.assert_allclose(data["arm_qpos"][0], 0.4)
                    np.testing.assert_array_equal(data["depth"][0], 500)
                    np.testing.assert_array_equal(data["dispatch_status"][0], [5, 4])
            before = (saved / "data.h5").read_bytes()
            self.assertEqual(recorder.save_episode("changed", stop_result="unconfirmed"), saved)
            recorder.close()
            recorder.discard_episode()
            self.assertEqual((saved / "data.h5").read_bytes(), before)
            self.assertTrue((saved / "rgb.mp4").is_file())
            self.assertFalse(recorder.writer_pending)
            with self.assertRaises(RecordingError):
                self.start(recorder)
        self.assertNotEqual(*destinations)

    def test_start_validation_failure_is_single_attempt(self):
        recorder = self.new_recorder()
        with self.assertRaises(RecordingError):
            recorder.start_episode(task_label="", collection_source="teleop", metadata={})
        self.assertTrue(recorder.start_attempted)
        self.assertIsNone(recorder.thread)
        with self.assertRaises(RecordingError):
            self.start(recorder)
        with self.assertRaises(RecordingError):
            recorder.close()

    def test_start_open_and_thread_start_failures_retain_staging(self):
        for target in ("dexmani_real.data.raw.writer.RawEpisodeWriter.open", "threading.Thread.start"):
            with self.subTest(target=target):
                recorder = self.new_recorder(target.rsplit(".", 1)[-1])
                with patch(target, side_effect=OSError("START injected")):
                    with self.assertRaises(RecordingError):
                        self.start(recorder)
                self.assertTrue(recorder.staging_path.is_dir())
                with self.assertRaises(RecordingError):
                    recorder.save_episode("start_failed", stop_result="unconfirmed")
                evidence = json.loads((recorder.staging_path / "capture_failure.json").read_text())
                self.assertEqual(evidence["reason"], "start_failed")
                self.assertIsNone(recorder.published_path)
                self.assertFalse(recorder.writer_pending)

    def test_start_timeout_retains_live_writer(self):
        recorder = self.new_recorder()
        entered, release = threading.Event(), threading.Event()
        original = RawEpisodeWriter.open

        def blocked(writer, metadata):
            entered.set()
            if not release.wait(2):
                raise TimeoutError("test gate")
            original(writer, metadata)

        with patch.object(RawEpisodeWriter, "open", blocked), patch(
            "dexmani_real.data.recorder.RECORDER_START_TIMEOUT_S", 0.02,
        ), patch("dexmani_real.data.recorder.RECORDER_STOP_TIMEOUT_S", 0.02):
            try:
                with self.assertRaises(RecordingError):
                    self.start(recorder)
                self.assertTrue(entered.is_set())
                self.assertTrue(recorder.writer_pending)
                with self.assertRaises(RecordingError):
                    self.start(recorder)
            finally:
                release.set()
                recorder.thread.join(2)
        with self.assertRaises(RecordingError):
            recorder.close()
        self.assertTrue(recorder.staging_path.exists())
        self.assertIsNone(recorder.published_path)

    def test_queue_full_preserves_prefix_and_unwritten_dispatch(self):
        recorder = self.new_recorder()
        entered, release = threading.Event(), threading.Event()
        original = RawEpisodeWriter.append_frame

        def blocked(writer, *args):
            entered.set()
            if not release.wait(2):
                raise TimeoutError("test gate")
            original(writer, *args)

        with patch.object(RawEpisodeWriter, "append_frame", blocked):
            self.start(recorder)
            try:
                recorder.add_frame(make_row(), self.command, self.dispatch)
                self.assertTrue(entered.wait(1))
                for index in range(1, 17):
                    recorder.add_frame(make_row(1_000_000_000 + index * 100_000_000), self.command, self.dispatch)
                with self.assertRaisesRegex(RecordingError, "queue full"):
                    recorder.add_frame(make_row(2_700_000_000), self.command, self.dispatch)
            finally:
                release.set()
            saved = recorder.save_episode("queue_full", stop_result="unconfirmed")
        with h5py.File(saved / "data.h5") as data:
            self.assertEqual(data["meta"].attrs["num_frames"], 17)
            self.assertEqual(data["meta"].attrs["technical_status"], "failed")
            self.assertEqual(data["meta"].attrs["stop_result"], "unconfirmed")
            details = json.loads(data["meta/termination_details"][()])
            self.assertTrue(any(item.get("observation_ns") == 2_700_000_000 for item in details))
        with self.assertRaises(RecordingError):
            recorder.check_error()

    def test_io_failure_keeps_partial_dispatch_and_never_retries(self):
        for target in ("dexmani_real.data.raw.writer.RawEpisodeWriter.append_frame",
                       "dexmani_real.data.raw.video.VideoEncoder.write_frame"):
            with self.subTest(target=target):
                recorder = self.new_recorder(target.rsplit(".", 1)[-1])
                with patch(target, side_effect=OSError("I/O injected")) as failing:
                    self.start(recorder)
                    recorder.add_frame(make_row(), self.command, self.dispatch)
                    recorder.thread.join(2)
                    self.assertFalse(recorder.writer_pending)
                    with self.assertRaises(RecordingError):
                        recorder.save_episode("io_failed", stop_result="unconfirmed")
                    with self.assertRaises(RecordingError):
                        recorder.close()
                    self.assertEqual(failing.call_count, 1)
                self.assertIsNone(recorder.published_path)
                evidence = json.loads((recorder.staging_path / "capture_failure.json").read_text())
                self.assertEqual(evidence["reason"], "io_failed")
                self.assertEqual(evidence["unwritten_dispatches"][0]["dispatch_status"], [5, 4])
                staging = recorder.staging_path
                recorder.discard_episode()
                self.assertFalse(staging.exists())
                with self.assertRaises(RecordingError):
                    recorder.check_error()

    def test_publish_conflict_preserves_destination_and_staging(self):
        recorder = self.new_recorder()
        self.start(recorder)

        def conflict(source, destination, **kwargs):
            destination.mkdir()
            (destination / "existing").write_text("untouched")
            return atomic_publish(source, destination, **kwargs)

        with patch("dexmani_real.data.recorder.atomic_publish", side_effect=conflict) as publish:
            with self.assertRaises(RecordingError):
                recorder.save_episode("first_reason", stop_result="confirmed")
            with self.assertRaises(RecordingError):
                recorder.save_episode("second_reason", stop_result="unconfirmed")
            self.assertEqual(publish.call_count, 1)
        self.assertEqual((recorder.episode_dir / "existing").read_text(), "untouched")
        evidence = json.loads((recorder.staging_path / "capture_failure.json").read_text())
        self.assertEqual(evidence["reason"], "first_reason")
        self.assertEqual(evidence["stop_result"], "confirmed")

    def test_close_timeout_keeps_owner_and_first_end(self):
        recorder = self.new_recorder()
        entered, release = threading.Event(), threading.Event()
        original = RawEpisodeWriter.close
        calls = []

        def blocked(writer):
            calls.append("close")
            entered.set()
            if not release.wait(2):
                raise TimeoutError("test gate")
            original(writer)

        with patch.object(RawEpisodeWriter, "close", blocked), patch(
            "dexmani_real.data.recorder.RECORDER_STOP_TIMEOUT_S", 0.02,
        ):
            self.start(recorder)
            try:
                with self.assertRaises(RecordingError):
                    recorder.save_episode("original", stop_result="unconfirmed")
                self.assertTrue(entered.is_set())
                self.assertTrue(recorder.writer_pending)
                self.assertTrue(recorder.staging_path.exists())
                with self.assertRaises(RecordingError):
                    recorder.close()
            finally:
                release.set()
                recorder.thread.join(2)
        with self.assertRaises(RecordingError):
            recorder.close()
        self.assertEqual(calls, ["close"])
        self.assertIsNone(recorder.published_path)
        self.assertEqual(json.loads((recorder.staging_path / "capture_failure.json").read_text())["reason"], "original")
