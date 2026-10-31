"""录制发布的所有权、收尾竞态与小 Raw 文件回归；不连接设备。"""

import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import av
import numpy as np

from dexmani_real.data.export import ExportConfig, discover_episode_dirs, export_zarr, training_prefix
from dexmani_real.data.raw import RawReader, RawWriter, snapshot_recording_metadata
from dexmani_real.data.recorder import EpisodeRecorder, RecordingError
from dexmani_real.geometry.camera import CameraIntrinsics, CameraSnapshot
from dexmani_real.runtime.observation import Observation
from dexmani_real.utils.io import publish


def raw_metadata():
    camera = CameraSnapshot(
        "test-camera", CameraIntrinsics(16, 16, 16., 16., 8., 8., "none", (0.,) * 5),
        .001, np.eye(4),
    )
    return snapshot_recording_metadata(camera, tactile_reset_metadata={}, tactile_hold_max_age_s=.2)


def raw_row(index):
    stamp = 1_000_000_000 + index * 62_500_000
    arm = np.zeros(1, dtype=[("qpos", "f8", (7,)), ("timestamp_ns", "i8")])
    hand = np.zeros(1, dtype=[
        ("qpos", "f8", (12,)), ("tactile_aggregate", "f4", (5, 3)),
        ("tactile_dense", "f4", (5, 120, 3)), ("timestamp_ns", "i8"),
        ("tactile_aggregate_timestamp_ns", "i8"), ("tactile_dense_timestamp_ns", "i8"),
    ])
    arm["qpos"] = index * .01
    hand["qpos"] = index * .02
    arm["timestamp_ns"] = stamp - 20
    for name in ("timestamp_ns", "tactile_aggregate_timestamp_ns", "tactile_dense_timestamp_ns"):
        hand[name] = stamp - 20
    return Observation(
        arm, hand,
        dict(rgb=np.full((16, 16, 3), index * 40, np.uint8),
             depth=np.full((16, 16), 500 + index, np.uint16),
             timestamp_ns=stamp - 30, ring_sequence=index + 1),
        None, None, stamp, control_timestamp_ns=stamp - 10,
    )


class _Writer:
    """只为可控阻塞测试提供 worker I/O 边界，真实格式另测。"""

    def __init__(self, *args, **kwargs):
        self.frame_count = 0
        self.closed = False

    def open(self, metadata):
        pass

    def append_frame(self, row, command):
        self.frame_count += 1

    def flush(self):
        pass

    def finish(self, *args, **kwargs):
        return self.frame_count, self.frame_count

    def close(self):
        self.closed = True


class RecorderOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.recorder = EpisodeRecorder(self.root, control_hz=16)

    def start(self):
        self.recorder.start(task_label="test", collection_source="policy_rollout", metadata={})

    def test_supervisor_publishes_once_after_worker_close(self):
        writer = _Writer()
        owner_thread = threading.get_ident()

        def checked_publish(source, destination):
            self.assertEqual(threading.get_ident(), owner_thread)
            self.assertTrue(writer.closed)
            self.assertFalse(self.recorder.writer_pending)
            return publish(source, destination)

        with patch("dexmani_real.data.recorder.RawWriter", return_value=writer), patch(
            "dexmani_real.data.recorder.publish", side_effect=checked_publish,
        ) as publishing:
            self.start()
            self.recorder.append(raw_row(0))
            result = self.recorder.finish(save=True, reason="stop", technical_status="normal")
            self.assertTrue(result.is_dir())
            self.assertEqual(self.recorder.result()["status"], "published")
            self.assertEqual(self.recorder.finish(save=False, reason="late discard"), result)
            publishing.assert_called_once()

    def test_discard_waits_for_close(self):
        writer = _Writer()
        with patch("dexmani_real.data.recorder.RawWriter", return_value=writer):
            self.start()
            staging = self.recorder.staging_path
            self.assertIsNone(self.recorder.finish(save=False, reason="discard"))
        self.assertTrue(writer.closed)
        self.assertFalse(staging.exists())
        self.assertEqual(self.recorder.result()["status"], "not_published")

    def test_late_open_observes_start_timeout_and_exits(self):
        release = threading.Event()
        writer = _Writer()
        writer.open = lambda metadata: release.wait(1)
        try:
            with patch("dexmani_real.data.recorder.RawWriter", return_value=writer), patch(
                "dexmani_real.data.recorder.RECORDER_START_TIMEOUT_S", .01,
            ), patch("dexmani_real.data.recorder.RECORDER_STOP_TIMEOUT_S", .01):
                with self.assertRaisesRegex(RecordingError, "START timed out"):
                    self.start()
                first_error = self.recorder.error
                self.assertTrue(self.recorder.writer_pending)
        finally:
            release.set()
            self.recorder.thread.join(1)
        self.assertFalse(self.recorder.writer_pending)
        self.assertTrue(writer.closed)
        self.assertIs(self.recorder.error, first_error)
        self.assertIsNone(self.recorder.published_path)
        self.assertTrue(self.recorder.staging_path.exists())

    def test_late_close_never_publishes_or_changes_first_error(self):
        for save in (True, False):
            with self.subTest(save=save):
                self.recorder = EpisodeRecorder(self.root, control_hz=16)
                release = threading.Event()
                writer = _Writer()
                writer.close = lambda: release.wait(1)
                try:
                    with patch("dexmani_real.data.recorder.RawWriter", return_value=writer), patch(
                        "dexmani_real.data.recorder.RECORDER_STOP_TIMEOUT_S", .02,
                    ):
                        self.start()
                        with self.assertRaisesRegex(RecordingError, "writer did not exit"):
                            self.recorder.finish(save=save, reason="stop", technical_status="normal")
                        first_error = self.recorder.error
                        self.assertEqual(self.recorder.result()["status"], "pending")
                finally:
                    release.set()
                    self.recorder.thread.join(1)
                self.assertIs(self.recorder.error, first_error)
                self.assertEqual(self.recorder.result()["status"], "failed")
                with self.assertRaises(RecordingError):
                    self.recorder.finish(save=True, reason="late completion")
                self.assertTrue(self.recorder.staging_path.exists())
                destination = self.recorder.staging_path.with_name(
                    self.recorder.staging_path.name.removeprefix(".tmp_")
                )
                self.assertFalse(destination.exists())

    def test_queue_full_keeps_first_failure_and_retains_staging(self):
        entered, release = threading.Event(), threading.Event()
        writer = _Writer()

        def blocked_append(row, command):
            entered.set()
            release.wait(1)
            writer.frame_count += 1

        writer.append_frame = blocked_append
        try:
            with patch("dexmani_real.data.recorder.RawWriter", return_value=writer), patch(
                "dexmani_real.data.recorder.RECORDER_STOP_TIMEOUT_S", .02,
            ):
                self.start()
                self.recorder.append(raw_row(0))
                self.assertTrue(entered.wait(1))
                for index in range(16):
                    self.recorder.append(raw_row(index + 1))
                with self.assertRaisesRegex(RecordingError, "queue full"):
                    self.recorder.append(raw_row(17))
                first_error = self.recorder.error
                with self.assertRaisesRegex(RecordingError, "queue full"):
                    self.recorder.finish(save=False, reason="discard")
        finally:
            release.set()
            self.recorder.thread.join(1)
        self.assertIs(self.recorder.error, first_error)
        self.assertTrue(self.recorder.staging_path.exists())
        self.assertIsNone(self.recorder.published_path)

    def test_open_append_finish_close_failures_never_publish(self):
        for operation in ("open", "append_frame", "finish", "close"):
            with self.subTest(operation=operation):
                self.recorder = EpisodeRecorder(self.root, control_hz=16)
                writer = _Writer()

                def fail(*args, **kwargs):
                    raise OSError(f"{operation} failed")

                setattr(writer, operation, fail)
                with patch("dexmani_real.data.recorder.RawWriter", return_value=writer):
                    if operation == "open":
                        with self.assertRaisesRegex(RecordingError, "open failed"):
                            self.start()
                    else:
                        self.start()
                        self.recorder.append(raw_row(0))
                        with self.assertRaisesRegex(RecordingError, f"{operation} failed"):
                            self.recorder.finish(save=False, reason="discard")
                self.assertFalse(self.recorder.writer_pending)
                self.assertTrue(self.recorder.staging_path.exists())
                self.assertIsNone(self.recorder.published_path)

    def test_supervisor_rejects_inconsistent_closed_counts(self):
        writer = _Writer()
        writer.finish = lambda *args, **kwargs: (0, 1)
        with patch("dexmani_real.data.recorder.RawWriter", return_value=writer):
            self.start()
            with self.assertRaisesRegex(RecordingError, "row count mismatch"):
                self.recorder.finish(save=True, reason="stop")
        self.assertTrue(writer.closed)
        self.assertTrue(self.recorder.staging_path.exists())

    def test_secondary_worker_close_error_is_visible_without_replacing_primary(self):
        writer = _Writer()

        def fail_append(*args):
            raise OSError("primary append failure")

        def fail_close():
            raise OSError("secondary close failure")

        writer.append_frame, writer.close = fail_append, fail_close
        with patch("dexmani_real.data.recorder.RawWriter", return_value=writer), \
                self.assertLogs("dexmani_real.data.recorder", level="ERROR") as recorded:
            self.start()
            self.recorder.append(raw_row(0))
            with self.assertRaisesRegex(RecordingError, "primary append failure"):
                self.recorder.finish(save=True, reason="failed")
        self.assertEqual(str(self.recorder.error), "primary append failure")
        self.assertIn("secondary close failure", "\n".join(recorded.output))
        self.assertIsNone(self.recorder.published_path)
        self.assertTrue(self.recorder.staging_path.exists())

    def test_finish_queue_timeout_cannot_publish_later(self):
        entered, release = threading.Event(), threading.Event()
        writer = _Writer()

        def blocked_append(row, command):
            entered.set()
            release.wait(1)
            writer.frame_count += 1

        writer.append_frame = blocked_append
        try:
            with patch("dexmani_real.data.recorder.RawWriter", return_value=writer), patch(
                "dexmani_real.data.recorder.RECORDER_STOP_TIMEOUT_S", .02,
            ):
                self.start()
                self.recorder.append(raw_row(0))
                self.assertTrue(entered.wait(1))
                for index in range(16):
                    self.recorder.append(raw_row(index + 1))
                with self.assertRaisesRegex(RecordingError, "finish queue timed out"):
                    self.recorder.finish(save=True, reason="stop", technical_status="normal")
                first_error = self.recorder.error
        finally:
            release.set()
            self.recorder.thread.join(1)
        self.assertIs(self.recorder.error, first_error)
        destination = self.recorder.staging_path.with_name(
            self.recorder.staging_path.name.removeprefix(".tmp_")
        )
        self.assertFalse(destination.exists())

    def test_publication_failure_is_not_retried(self):
        with patch("dexmani_real.data.recorder.RawWriter", return_value=_Writer()), patch(
            "dexmani_real.data.recorder.publish", side_effect=OSError("rename failed"),
        ) as publishing:
            self.start()
            with self.assertRaisesRegex(RecordingError, "rename failed"):
                self.recorder.finish(save=True, reason="stop")
            with self.assertRaisesRegex(RecordingError, "rename failed"):
                self.recorder.finish(save=False, reason="late discard")
            publishing.assert_called_once()
        self.assertTrue(self.recorder.staging_path.exists())


class RawRecordingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def record(self, name, status):
        recorder = EpisodeRecorder(self.root, control_hz=16)
        recorder.start(task_label="test", collection_source="policy_rollout",
                       metadata=raw_metadata(), episode_name=name)
        for index in range(3):
            row = raw_row(index)
            command = SimpleNamespace(arm_qpos=row.arm["qpos"][0] + .1,
                                      hand_qpos=row.hand["qpos"][0] + .1)
            recorder.append(row, command)
        path = recorder.finish(
            save=True, reason="stop" if status == "normal" else "hand_retarget_rejected",
            technical_status=status,
            details=[dict(final_loss=None, failure_reason="non-finite final loss")],
        )
        self.assertEqual(recorder.submitted_frames, 3)
        self.assertEqual(recorder.closed_counts, (3, 3))
        return path

    def test_normal_failed_and_consecutive_real_h5_video(self):
        for index, status in enumerate(("normal", "failed")):
            path = self.record(f"episode_{index}", status)
            with RawReader(path) as reader:
                self.assertEqual(reader.num_frames, 3)
                self.assertEqual(len(list(reader.iter_rgb_frames())), 3)
                np.testing.assert_array_equal(reader["arm_qpos"][:, 0], [0., .01, .02])
                np.testing.assert_allclose(reader["action_arm_joint_target"][:, 0], [.1, .11, .12])
                np.testing.assert_array_equal(reader["timing/rows"]["camera_ns"],
                                              [1_000_000_000 + i * 62_500_000 - 30 for i in range(3)])
                self.assertEqual(reader.meta["technical_status"], status)
                self.assertIsNone(json.loads(reader.data_file["meta/termination_details"][()])[0]["final_loss"])
                self.assertEqual(training_prefix(reader)["row_stop"], 3 if status == "normal" else 2)
        self.assertEqual(len(discover_episode_dirs(self.root)), 2)

    def test_export_api_and_cli_keep_the_explicit_temporary_request_name(self):
        from scripts import export_zarr as export_cli

        published = self.record("episode_published", "normal")
        alias = self.root / ".tmp_alias"
        alias.symlink_to(published, target_is_directory=True)
        output = self.root / "output.zarr"
        processing = ExportConfig()
        with self.assertRaisesRegex(ValueError, "unpublished"):
            export_zarr(alias, output, processing=processing)
        with patch.object(export_cli, "load_processing_config", return_value=processing), patch.object(
            export_cli, "tqdm",
        ), patch("dexmani_real.utils.log.configure_logging"):
            with self.assertRaisesRegex(ValueError, "unpublished"):
                export_cli.main([str(alias), "--output", str(output)])
        self.assertFalse(output.exists())

    def test_unpublished_explicit_path_and_symlink_are_rejected(self):
        path = self.root / ".tmp_episode_001"
        path.mkdir()
        alias = self.root / "episode_alias"
        alias.symlink_to(path, target_is_directory=True)
        published = self.root / "episode_published"
        published.mkdir()
        temporary_alias = self.root / ".tmp_alias"
        temporary_alias.symlink_to(published, target_is_directory=True)
        for target in (path, alias, temporary_alias):
            with self.subTest(target=target):
                with self.assertRaisesRegex(ValueError, "unpublished"):
                    RawReader(target)
                with self.assertRaisesRegex(ValueError, "unpublished"):
                    discover_episode_dirs(target)

    def test_real_writer_count_mismatch_retains_unpublished_data(self):
        writer = RawWriter(self.root, control_hz=16)
        try:
            writer.open(dict(raw_metadata(), task_label="test", collection_source="policy_rollout"))
            writer.append_frame(raw_row(0), None)
            with self.assertRaisesRegex(RuntimeError, "row count mismatch"):
                writer.finish("stop", (), expected_frames=2, technical_status="normal")
        finally:
            writer.close()

    def test_broken_video_is_a_read_error(self):
        path = self.record("episode_bad_rgb", "normal")
        (path / "rgb.mp4").write_bytes(b"broken mp4")
        with RawReader(path) as reader:
            with self.assertRaises(av.error.InvalidDataError):
                list(reader.iter_rgb_frames())


if __name__ == "__main__":
    unittest.main()
