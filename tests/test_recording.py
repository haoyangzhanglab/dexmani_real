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
from dexmani_real.robot.commands import RobotCommand
from dexmani_real.utils.atomic_io import atomic_publish
from tests.offline_fixtures import make_row, recording_metadata


class RecorderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="dexmani-recorder-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.command = RobotCommand(1, np.zeros(7), np.zeros(12))

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
                recorder.add_frame(row, self.command)
                row.arm["qpos"] = 9
                row.camera["depth"][:] = 9
            saved = recorder.save_episode("operator")
            destinations.append(saved)
            with h5py.File(saved / "data.h5") as data:
                self.assertEqual(data["meta"].attrs["num_frames"], index)
                self.assertEqual(data["meta"].attrs["termination_reason"], "operator")
                if index:
                    np.testing.assert_allclose(data["arm_qpos"][0], 0.4)
                    np.testing.assert_array_equal(data["depth"][0], 500)
                    np.testing.assert_array_equal(data["action_arm_joint_target"][0], self.command.arm_qpos)
                    self.assertNotIn("dispatch_status", data)
            before = (saved / "data.h5").read_bytes()
            self.assertEqual(recorder.save_episode("changed"), saved)
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
                    recorder.save_episode("start_failed")
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

    def test_queue_full_preserves_prefix_without_a_send_ledger(self):
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
                recorder.add_frame(make_row(), self.command)
                self.assertTrue(entered.wait(1))
                for index in range(1, 17):
                    recorder.add_frame(make_row(1_000_000_000 + index * 100_000_000), self.command)
                with self.assertRaisesRegex(RecordingError, "queue full"):
                    recorder.add_frame(make_row(2_700_000_000), self.command)
            finally:
                release.set()
            saved = recorder.save_episode("queue_full")
        with h5py.File(saved / "data.h5") as data:
            self.assertEqual(data["meta"].attrs["num_frames"], 17)
            self.assertEqual(data["meta"].attrs["technical_status"], "failed")
            self.assertNotIn("stop_result", data["meta"].attrs)
            details = json.loads(data["meta/termination_details"][()])
            self.assertTrue(any(item.get("stage") == "recording" for item in details))
        with self.assertRaises(RecordingError):
            recorder.check_error()

    def test_io_failure_keeps_staging_and_never_retries(self):
        for target in ("dexmani_real.data.raw.writer.RawEpisodeWriter.append_frame",
                       "dexmani_real.data.raw.video.VideoEncoder.write_frame"):
            with self.subTest(target=target):
                recorder = self.new_recorder(target.rsplit(".", 1)[-1])
                with patch(target, side_effect=OSError("I/O injected")) as failing:
                    self.start(recorder)
                    recorder.add_frame(make_row(), self.command)
                    recorder.thread.join(2)
                    self.assertFalse(recorder.writer_pending)
                    with self.assertRaises(RecordingError):
                        recorder.save_episode("io_failed")
                    with self.assertRaises(RecordingError):
                        recorder.close()
                    self.assertEqual(failing.call_count, 1)
                self.assertIsNone(recorder.published_path)
                evidence = json.loads((recorder.staging_path / "capture_failure.json").read_text())
                self.assertEqual(evidence["reason"], "io_failed")
                self.assertEqual(evidence["accepted_frames"], 1)
                self.assertNotIn("unwritten_dispatches", evidence)
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
                recorder.save_episode("first_reason")
            with self.assertRaises(RecordingError):
                recorder.save_episode("second_reason")
            self.assertEqual(publish.call_count, 1)
        self.assertEqual((recorder.episode_dir / "existing").read_text(), "untouched")
        evidence = json.loads((recorder.staging_path / "capture_failure.json").read_text())
        self.assertEqual(evidence["reason"], "first_reason")
        self.assertNotIn("stop_result", evidence)

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
                    recorder.save_episode("original")
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

    def test_partial_batch_and_encoder_close_failures_never_publish_or_retry(self):
        original_resize = h5py.Dataset.resize
        resize_calls = []

        def partial_resize(dataset, *args, **kwargs):
            resize_calls.append(dataset.name)
            if len(resize_calls) == 2:
                raise OSError("partial batch resize")
            return original_resize(dataset, *args, **kwargs)

        recorder = self.new_recorder("partial-batch")
        self.start(recorder)
        recorder.add_frame(make_row(), self.command)
        with patch.object(h5py.Dataset, "resize", partial_resize):
            with self.assertRaises(RecordingError):
                recorder.save_episode("operator")
            with self.assertRaises(RecordingError):
                recorder.close()
        self.assertEqual(len(resize_calls), 2)
        self.assertIsNone(recorder.published_path)
        self.assertTrue((recorder.staging_path / "capture_failure.json").is_file())

        from dexmani_real.data.raw.video import VideoEncoder
        original_close = VideoEncoder.close
        close_calls = []

        def fail_close(encoder):
            close_calls.append(encoder)
            original_close(encoder)
            raise OSError("encoder flush failed")

        recorder = self.new_recorder("close-failure")
        with patch.object(VideoEncoder, "close", fail_close):
            self.start(recorder)
            recorder.add_frame(make_row(), self.command)
            with self.assertRaises(RecordingError):
                recorder.save_episode("operator")
            with self.assertRaises(RecordingError):
                recorder.close()
        self.assertEqual(len(close_calls), 1)
        self.assertIsNone(recorder.published_path)
        self.assertTrue((recorder.staging_path / "capture_failure.json").is_file())
