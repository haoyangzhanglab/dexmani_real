import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np

from dexmani_real.data import recorder as recording
from dexmani_real.data.canonical.export import export_canonical
from dexmani_real.data.canonical.schema import ExportConfig
from dexmani_real.data.raw.reader import EpisodeReader
from dexmani_real.data.raw.writer import EpisodeWriter
from dexmani_real.geometry.camera import CameraIntrinsics, CameraSnapshot
from dexmani_real.perception.config import PointCloudConfig
from dexmani_real.robot.commands import RobotCommand
from dexmani_real.robot.xarm7.protocol import ARM_STATE_DTYPE
from dexmani_real.robot.xhand.feedback import HAND_STATE_DTYPE


def observation(index):
    stamp = 1_000_000_000 + index * 62_500_000
    arm, hand = np.zeros(1, ARM_STATE_DTYPE), np.zeros(1, HAND_STATE_DTYPE)
    arm["timestamp_ns"] = hand["timestamp_ns"] = stamp
    hand["tactile_aggregate_timestamp_ns"] = hand["tactile_dense_timestamp_ns"] = stamp
    arm["qpos"] = index * 0.001
    rgb = np.full((4, 4, 3), 100 + index % 100, dtype=np.uint8)
    depth = np.full((4, 4), 500, dtype=np.uint16)
    for array in (arm, hand, rgb, depth):
        array.flags.writeable = False
    return SimpleNamespace(
        arm=arm,
        hand=hand,
        camera=dict(rgb=rgb, depth=depth, timestamp_ns=stamp, ring_sequence=index),
        observation_timestamp_ns=stamp,
        control_timestamp_ns=stamp,
    )


class DataTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.camera = CameraSnapshot(
            "test",
            CameraIntrinsics(4, 4, 100.0, 100.0, 1.5, 1.5, "none", (0, 0, 0, 0, 0)),
            0.001,
            np.eye(4),
        )
        self.metadata = recording.snapshot_recording_metadata(
            self.camera,
            tactile_reset_metadata={},
            tactile_hold_max_age_s=0.1,
        )
        self.command = RobotCommand(1, np.zeros(7), np.zeros(12))
        self.config = ExportConfig(
            pointcloud=PointCloudConfig(
                num_points=16,
                workspace=(-1, -1, 0, 1, 1, 1),
                remove_table=False,
            )
        )
        errors = patch.object(recording.logger, "error")
        errors.start()
        self.addCleanup(errors.stop)

    def start(self):
        recorder = recording.EpisodeRecorder(self.root, control_hz=16)
        recorder.start(task_label="test", collection_source="teleop", metadata=self.metadata)
        self.addCleanup(self.stop, recorder)
        return recorder

    def stop(self, recorder):
        recorder.cancelled.set()
        recorder.thread.join(timeout=2)
        self.assertFalse(recorder.writer_pending)

    def write_episode(self, count=3):
        episode = self.root / "episode_1"
        episode.mkdir()
        writer = EpisodeWriter(episode, control_hz=16)
        try:
            writer.open(dict(self.metadata, task_label="test", collection_source="teleop"))
            for index in range(count):
                writer.append_frame(observation(index), self.command)
            writer.flush()
            writer.write_termination(
                "operator", [], expected_frames=count, technical_status="normal"
            )
        finally:
            writer.close()
        return episode

    def test_writer_roundtrip_across_batch_boundary_and_reader_closes_video(self):
        episode = self.write_episode(33)
        with EpisodeReader(episode) as reader:
            self.assertEqual(reader.num_frames, 33)
            self.assertNotIn("num_frames", reader.meta)
            reader.validate_completion()
            reader.validate_rows()
            np.testing.assert_allclose(reader["arm_qpos"][:, 0], np.arange(33) * 0.001)
            np.testing.assert_array_equal(reader["depth"][:], np.full((33, 4, 4), 500))
            self.assertEqual(len(list(reader.iter_rgb_frames())), 33)
            next(reader.iter_rgb_frames())
            camera = reader.read_camera()
            self.assertEqual(camera.color, self.camera.color)
            self.assertEqual(camera.depth_scale_m, 0.001)
            np.testing.assert_array_equal(camera.T_xarm_base_from_color, np.eye(4))
        self.assertFalse(reader.data_file.id.valid)
        with self.assertRaises((ValueError, KeyError)):
            reader["timestamp"]
        with self.assertRaises(AssertionError):
            reader.video_container.seek(0)

    def test_recorder_saves_only_the_first_finish_request(self):
        recorder = self.start()
        recorder.append(observation(0), self.command)
        path = recorder.finish(save=True, reason="operator", technical_status="normal")
        self.assertEqual(recorder.finish(save=False, reason="discard"), path)
        recorder.close()
        self.assertEqual(recorder.result()["status"], "published")
        with EpisodeReader(path) as reader:
            self.assertEqual(reader.num_frames, 1)
            self.assertEqual(reader.meta["termination_reason"], "operator")
            reader.validate_completion()

    def test_discard_is_preserved_on_repeated_close(self):
        recorder = self.start()
        recorder.append(observation(0), self.command)
        self.assertIsNone(recorder.finish(save=False, reason="discard"))
        recorder.close()
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertEqual(recorder.result()["status"], "not_published")

    def test_result_observes_publication_when_the_writer_finishes_during_snapshot(self):
        recorder = recording.EpisodeRecorder(self.root, control_hz=16)
        recorder.start_attempted = True
        target = self.root / "episode_1"

        def completed():
            recorder.published_path = target
            return False

        recorder.thread = SimpleNamespace(is_alive=completed)
        result = recorder.result()
        self.assertEqual(result["status"], "published")
        self.assertEqual(result["path"], str(target))

    def test_empty_capture_is_readable_but_not_trainable(self):
        recorder = self.start()
        path = recorder.finish(save=True, reason="operator", technical_status="normal")
        with EpisodeReader(path) as reader:
            self.assertEqual(reader.num_frames, 0)
        with self.assertLogs("dexmani_real.data.canonical.export", level="ERROR"):
            with self.assertRaisesRegex(ValueError, "no rows"):
                export_canonical(path, self.root / "empty.zarr", processing=self.config)
        self.assertFalse((self.root / "empty.zarr").exists())

    def test_complete_recording_can_describe_a_failed_control_episode(self):
        recorder = self.start()
        recorder.append(observation(0), self.command)
        path = recorder.finish(save=True, reason="stop_failed", technical_status="failed")
        self.assertTrue(path.is_dir())
        with EpisodeReader(path) as reader:
            with self.assertRaisesRegex(ValueError, "technical failure"):
                reader.validate_completion()

    def test_queue_overflow_preserves_staging_without_publishing(self):
        entered, release = threading.Event(), threading.Event()
        original = EpisodeWriter.append_frame

        def append(writer, *args):
            entered.set()
            if not release.wait(2):
                raise TimeoutError("test writer was not released")
            original(writer, *args)

        with patch.object(EpisodeWriter, "append_frame", append):
            recorder = self.start()
            try:
                recorder.append(observation(0), self.command)
                self.assertTrue(entered.wait(1))
                for index in range(1, 17):
                    recorder.append(observation(index), self.command)
                with self.assertRaisesRegex(recording.RecordingError, "queue full"):
                    recorder.append(observation(17), self.command)
            finally:
                release.set()
                recorder.thread.join(2)
        with self.assertRaisesRegex(recording.RecordingError, "queue full"):
            recorder.finish(save=True, reason="recording_failed")
        self.assertFalse(recorder.episode_dir.exists())
        self.assertTrue(recorder.staging_path.is_dir())
        self.assertFalse((recorder.staging_path / "capture_failure.json").exists())
        self.assertTrue(recorder.queue.empty())
        with EpisodeReader(recorder.staging_path) as reader:
            self.assertEqual(reader.num_frames, 1)
            with self.assertRaises(ValueError):
                reader.validate_completion()

    def test_write_failure_survives_close_failure_and_explicit_discard(self):
        original = EpisodeWriter.close
        write_error = OSError("disk failure")

        def close(writer):
            original(writer)
            raise OSError("close failure")

        with (
            patch.object(EpisodeWriter, "append_frame", side_effect=write_error),
            patch.object(EpisodeWriter, "close", close),
        ):
            recorder = self.start()
            recorder.append(observation(0), self.command)
            recorder.thread.join(2)
            with self.assertRaises(recording.RecordingError) as caught:
                recorder.finish(save=False, reason="discard")
        self.assertIs(caught.exception.__cause__, write_error)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_close_failure_prevents_publication(self):
        original = EpisodeWriter.close

        def close(writer):
            original(writer)
            raise OSError("video finalization failed")

        with patch.object(EpisodeWriter, "close", close):
            recorder = self.start()
            recorder.append(observation(0), self.command)
            with self.assertRaisesRegex(recording.RecordingError, "video finalization"):
                recorder.finish(save=True, reason="operator", technical_status="normal")
        self.assertFalse(recorder.episode_dir.exists())
        self.assertTrue(recorder.staging_path.is_dir())

    def test_startup_and_flush_failures_are_reported_without_publication(self):
        error = OSError("storage failed")
        with patch.object(EpisodeWriter, "open", side_effect=error):
            recorder = recording.EpisodeRecorder(self.root, control_hz=16)
            with self.assertRaises(recording.RecordingError) as caught:
                recorder.start(
                    task_label="test", collection_source="teleop", metadata=self.metadata
                )
            self.assertIs(caught.exception.__cause__, error)
            self.assertFalse(recorder.writer_pending)
            self.assertFalse(recorder.episode_dir.exists())
        with patch.object(EpisodeWriter, "flush", side_effect=error):
            recorder = self.start()
            recorder.append(observation(0), self.command)
            with self.assertRaises(recording.RecordingError) as caught:
                recorder.finish(save=True, reason="operator")
            self.assertIs(caught.exception.__cause__, error)
            self.assertFalse(recorder.episode_dir.exists())
            self.assertTrue(recorder.staging_path.is_dir())

    def test_full_finish_queue_is_a_pending_timeout(self):
        entered, release = threading.Event(), threading.Event()
        original = EpisodeWriter.append_frame

        def append(writer, *args):
            entered.set()
            if not release.wait(2):
                raise TimeoutError("test writer was not released")
            original(writer, *args)

        with (
            patch.object(EpisodeWriter, "append_frame", append),
            patch.object(recording, "RECORDER_STOP_TIMEOUT_S", 0.02),
        ):
            recorder = self.start()
            try:
                recorder.append(observation(0), self.command)
                self.assertTrue(entered.wait(1))
                for index in range(1, 17):
                    recorder.append(observation(index), self.command)
                with self.assertRaisesRegex(recording.RecordingError, "pending"):
                    recorder.finish(save=True, reason="operator")
                self.assertIsInstance(recorder.error, TimeoutError)
            finally:
                release.set()
                recorder.thread.join(2)
        self.assertFalse(recorder.episode_dir.exists())

    def test_timeout_cancels_uncommitted_publication_without_resetting_budget(self):
        entered, release = threading.Event(), threading.Event()
        original = EpisodeWriter.append_frame

        def append(writer, *args):
            entered.set()
            if not release.wait(2):
                raise TimeoutError("test writer was not released")
            original(writer, *args)

        with (
            patch.object(EpisodeWriter, "append_frame", append),
            patch.object(recording, "RECORDER_STOP_TIMEOUT_S", 0.02),
        ):
            recorder = self.start()
            try:
                recorder.append(observation(0), self.command)
                self.assertTrue(entered.wait(1))
                with self.assertRaisesRegex(recording.RecordingError, "pending"):
                    recorder.finish(save=True, reason="operator")
                deadline = recorder.stop_deadline
                with patch.object(recorder.thread, "join", wraps=recorder.thread.join) as join:
                    with self.assertRaisesRegex(recording.RecordingError, "pending"):
                        recorder.close()
                    self.assertEqual(join.call_args.kwargs["timeout"], 0)
                self.assertEqual(recorder.stop_deadline, deadline)
                self.assertEqual(recorder.result()["status"], "pending")
            finally:
                release.set()
                recorder.thread.join(2)
        self.assertIsNone(recorder.published_path)
        self.assertFalse(recorder.episode_dir.exists())
        self.assertEqual(recorder.result()["status"], "failed")

    def test_timeout_during_commit_reports_pending_then_actual_publication(self):
        entered, release = threading.Event(), threading.Event()
        original = recording.atomic_publish

        def publish(source, destination):
            entered.set()
            if not release.wait(2):
                raise TimeoutError("test publisher was not released")
            return original(source, destination)

        with (
            patch.object(recording, "atomic_publish", publish),
            patch.object(recording, "RECORDER_STOP_TIMEOUT_S", 0.03),
        ):
            recorder = self.start()
            try:
                with patch.object(
                    recorder.thread,
                    "join",
                    side_effect=lambda timeout: self.assertTrue(entered.wait(1)),
                ):
                    with self.assertRaisesRegex(recording.RecordingError, "pending"):
                        recorder.finish(save=True, reason="operator", technical_status="normal")
                self.assertTrue(entered.is_set())
                self.assertIsNone(recorder.error)
                self.assertEqual(recorder.result()["status"], "pending")
            finally:
                release.set()
                recorder.thread.join(2)
        recorder.close()
        self.assertEqual(recorder.result()["status"], "published")
        self.assertTrue(recorder.published_path.is_dir())

    def test_existing_episode_is_not_overwritten(self):
        path = self.write_episode()
        recorder = recording.EpisodeRecorder(self.root, control_hz=16)
        with self.assertRaises(recording.RecordingError) as caught:
            recorder.start(
                task_label="test",
                collection_source="teleop",
                metadata=self.metadata,
                episode_name=path.name,
            )
        self.assertIsInstance(caught.exception.__cause__, FileExistsError)
        with EpisodeReader(path) as reader:
            self.assertEqual(reader.num_frames, 3)

    def test_reader_rejects_corrupt_layout_timing_and_geometry(self):
        episode = self.write_episode()
        with h5py.File(episode / "data.h5", "r+") as data:
            data["timestamp"][1] = 0
        with EpisodeReader(episode) as reader:
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                reader.validate_rows()
        with h5py.File(episode / "data.h5", "r+") as data:
            del data["arm_qpos"]
            data.create_dataset("arm_qpos", data=np.zeros((3, 6)))
            matrix = np.eye(3)
            matrix[0, 1] = 0.1
            data["meta"].attrs["camera_color_intrinsics"] = matrix.ravel()
        with EpisodeReader(episode) as reader:
            with self.assertRaisesRegex(ValueError, "shape/dtype"):
                reader["arm_qpos"]
            with self.assertRaisesRegex(ValueError, "pinhole"):
                reader.read_camera()

    def test_export_rejects_missing_video_frames_and_nonfinite_targets(self):
        episode = self.write_episode()
        with patch.object(EpisodeReader, "iter_rgb_frames", return_value=iter([])):
            with self.assertLogs("dexmani_real.data.canonical.export", level="ERROR"):
                with self.assertRaisesRegex(ValueError, "RGB missing"):
                    export_canonical(episode, self.root / "missing.zarr", processing=self.config)
        with h5py.File(episode / "data.h5", "r+") as data:
            data["action_arm_joint_target"][0, 0] = np.nan
        with self.assertLogs("dexmani_real.data.canonical.export", level="ERROR"):
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                export_canonical(episode, self.root / "nan.zarr", processing=self.config)
        self.assertFalse((self.root / "missing.zarr").exists())
        self.assertFalse((self.root / "nan.zarr").exists())


if __name__ == "__main__":
    unittest.main()
