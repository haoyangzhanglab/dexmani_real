import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import zarr

from fakes import FakeClock, camera_metadata, execution_snapshot, observation
from dexmani_real.data.canonical.export import CanonicalExporter
from dexmani_real.data.canonical.schema import ProcessingConfig, canonical_array_specs
from dexmani_real.data.raw.reader import EpisodeReader
from dexmani_real.data.raw.schema import RAW_FORMAT, encode_raw_row
from dexmani_real.data.raw.writer import RawEpisodeWriter
from dexmani_real.data.recorder import AsyncEpisodeRecorder, RecordingError
from dexmani_real.robot.commands import DispatchResult, DispatchStatus, RobotCommand
from dexmani_real.sensor.pointcloud.config import PointCloudConfig


def tree_hash(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


class DataContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.clock = FakeClock()

    def recorder(self):
        return AsyncEpisodeRecorder(self.root / "raw", control_hz=20)

    def start(self, recorder):
        recorder.start_episode(task_label="test", collection_source="policy_rollout",
                               metadata=camera_metadata(), arm_execution=execution_snapshot(),
                               arm_control_metadata={}, episode_name="episode_001")

    def test_zero_row_capture_is_published_with_termination(self):
        recorder = self.recorder()
        self.start(recorder)
        path = recorder.save_episode(reason="start_cancelled", arm_execution=execution_snapshot())
        with EpisodeReader(path) as reader:
            self.assertEqual(reader.num_frames, 0)
            self.assertEqual(reader.meta["termination_reason"], "start_cancelled")
            self.assertEqual(reader.meta["format"], RAW_FORMAT)
        before = tree_hash(path)
        recorder.close()
        self.assertEqual(tree_hash(path), before)

    def test_not_ready_writer_keeps_staging(self):
        recorder = self.recorder()
        with patch.object(RawEpisodeWriter, "open", side_effect=OSError("disk unavailable")):
            with self.assertRaises(RecordingError):
                self.start(recorder)
        self.assertFalse(recorder.is_recording)
        self.assertTrue(recorder.staging_path.is_dir())
        self.assertIsNone(recorder.published_path)
        with self.assertRaises(RecordingError):
            recorder.save_episode(reason="preparation_failed", arm_execution=execution_snapshot())

    def test_full_queue_never_retries_or_counts_rejected_frame(self):
        recorder = self.recorder()
        recorder.recording = True
        for _ in range(recorder.queue.maxsize):
            recorder.queue.put_nowait(object())
        with self.assertRaises(RecordingError):
            recorder.add_frame(observation(self.clock, camera=True))
        self.assertEqual(recorder.submitted_frames, 0)
        self.assertEqual(recorder.queue.qsize(), recorder.queue.maxsize)

    def test_write_encode_and_publish_failures_keep_staging(self):
        targets = (
            "dexmani_real.data.raw.writer.RawEpisodeWriter.append_frame",
            "dexmani_real.data.raw.video.VideoEncoder.write_frame",
            "dexmani_real.data.recorder.atomic_publish",
        )
        for index, target in enumerate(targets):
            with self.subTest(target=target):
                recorder = AsyncEpisodeRecorder(self.root / str(index), control_hz=20)
                self.start(recorder)
                with patch(target, side_effect=OSError("injected failure")):
                    recorder.add_frame(observation(self.clock, camera=True))
                    with self.assertRaises(RecordingError):
                        recorder.save_episode(reason="failure", arm_execution=execution_snapshot())
                self.assertTrue(recorder.staging_path.is_dir())
                self.assertFalse(recorder.episode_dir.exists())
                self.assertIsNone(recorder.published_path)
                self.assertFalse(recorder.thread.is_alive())

    def test_unconfirmed_stop_prevents_publication(self):
        recorder = self.recorder()
        self.start(recorder)
        with self.assertRaises(RecordingError):
            recorder.save_episode(reason="stop_failed", arm_execution=execution_snapshot(stopped=False))
        self.assertTrue(recorder.staging_path.is_dir())
        self.assertFalse(recorder.episode_dir.exists())

    def test_raw_missing_values_remain_nan_and_actions_are_submitted_targets(self):
        row = replace(observation(self.clock, camera=True), hand=None)
        command = RobotCommand(3, np.ones(7), np.ones(12))
        values = encode_raw_row(row, command,
                                DispatchResult(DispatchStatus.SUBMITTED, DispatchStatus.NOT_CALLED), 0)
        self.assertTrue(np.isnan(values["hand_current"]).all())
        self.assertTrue(np.isnan(values["action_hand_joint_target"]).all())
        np.testing.assert_array_equal(values["action_arm_joint_target"], np.ones(7))
        self.assertFalse(np.array_equal(values["arm_qpos"], values["action_arm_joint_target"]))

    def fixture(self, name, *, invalid=None):
        # Corruptions are constructed in unpublished synthetic staging only.
        staging = self.root / "raw" / (".tmp_" + name)
        staging.mkdir(parents=True)
        metadata = camera_metadata()
        transform = np.eye(4)
        transform[0, 3] = .1
        metadata["camera_T_xarm_base_from_color"] = transform.ravel()
        metadata.update(task_label="test", collection_source="teleop")
        writer = RawEpisodeWriter(staging, control_hz=20)
        try:
            writer.open(metadata, execution_snapshot(), {})
            for index in range(2):
                row = observation(self.clock, camera=True)
                command = RobotCommand(3, row.arm["qpos"][0] + .001, row.hand["qpos"][0])
                receipt = DispatchResult(DispatchStatus.SUBMITTED, DispatchStatus.CRC_UNCONFIRMED,
                                         self.clock.now, (3, index + 1))
                writer.append_frame(row, command, receipt, {"run_id": 3, "slot": index})
                self.clock.advance(.05)
            writer.flush()
            writer.write_execution_end(execution_snapshot(2))
            writer.write_termination("operator", [], None, expected_frames=2)
        finally:
            writer.close()
        if invalid:
            with h5py.File(staging / "data.h5", "r+") as data:
                if invalid == "missing":
                    del data["hand_current"]
                elif invalid == "nan":
                    data["hand_current"][1, 0] = np.nan
                elif invalid == "geometry":
                    del data["meta"].attrs["camera_T_xarm_base_from_color"]
                elif invalid == "dispatch":
                    data["dispatch_status"][1] = (0, 0)
                elif invalid == "depth_rows":
                    data["depth"].resize(1, axis=0)
                elif invalid == "sdk":
                    del data["arm_execution/end"]
                    data["arm_execution"].create_dataset("end", data=json.dumps(dict(execution_snapshot(2), first_accepted=1)))
                elif invalid == "empty_cloud":
                    data["depth"][:] = 0
                elif invalid == "timing":
                    data["timestamp"][1] = .2
        destination = staging.with_name(name)
        staging.rename(destination)
        return destination

    def exporter(self):
        cloud = PointCloudConfig(remove_table=False, num_points=4,
                                 workspace=(-1., -1., 0., 1., 1., 1.),
                                 depth_support_min_neighbors=0, edge_support_min_neighbors=0,
                                 outlier_min_neighbors=0, outlier_min_component_points=1)
        return CanonicalExporter(ProcessingConfig(pointcloud=cloud), chunk_frames=1)

    def test_full_export_preserves_15_fields_geometry_tactile_and_raw(self):
        source = self.fixture("episode_valid")
        before = tree_hash(source)
        target = self.root / "canonical.zarr"
        result = self.exporter().export(source, target)
        self.assertEqual(result["total_frames"], 2)
        store = zarr.open_group(str(target), mode="r")
        specs = canonical_array_specs(4, 4, 4)
        self.assertEqual(len(specs), 15)
        self.assertEqual(set(store["data"].array_keys()), set(specs))
        for name, (tail, dtype) in specs.items():
            self.assertEqual(store["data"][name].shape, (2, *tail))
            self.assertEqual(store["data"][name].dtype, dtype)
            self.assertTrue(np.isfinite(store["data"][name][:]).all())
        np.testing.assert_array_equal(store["data/finger_tactile_force"][:], np.full((2, 5, 3), 1.25))
        np.testing.assert_array_equal(store["data/taxel_tactile_force"][:], np.full((2, 5, 120, 3), 2.5))
        np.testing.assert_allclose(store["data/point_cloud"][:, :, 2], .5)
        expected = np.eye(4)[:3]
        expected[0, 3] = -.1
        np.testing.assert_allclose(store["data/camera_extrinsic"][0], expected.ravel())
        np.testing.assert_array_equal(store["meta/episode_ends"][:], [2])
        self.assertEqual(tree_hash(source), before)
        with self.assertRaises(FileExistsError):
            self.exporter().export(source, target)

    def test_bad_episodes_are_rejected_whole_without_raw_changes(self):
        self.fixture("episode_valid")
        for issue in ("missing", "nan", "geometry", "dispatch", "depth_rows", "sdk", "empty_cloud", "timing"):
            self.fixture("episode_" + issue, invalid=issue)
        before = tree_hash(self.root / "raw")
        result = self.exporter().export(self.root / "raw", self.root / "canonical.zarr")
        self.assertEqual(result["episode_count"], 1)
        self.assertEqual(result["total_frames"], 2)
        self.assertEqual(len(result["rejected_episodes"]), 8)
        self.assertTrue(all(result["rejected_episodes"].values()))
        self.assertEqual(tree_hash(self.root / "raw"), before)
