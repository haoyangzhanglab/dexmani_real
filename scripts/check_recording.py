"""用临时合成文件验证 Raw、Recorder 和 Canonical；不连接设备。"""

import json
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import h5py
import numpy as np
import zarr

from dexmani_real.config.experiment import ExperimentConfig
from dexmani_real.data.canonical.export import CanonicalExporter
from dexmani_real.data.canonical.processing import validate_export_episode
from dexmani_real.data.canonical.schema import ProcessingConfig, canonical_array_specs
from dexmani_real.data.raw.reader import EpisodeReader, RawDataError
from dexmani_real.data.raw.schema import TIMING_ROW_DTYPE, validate_fixed_rate_rows
from dexmani_real.data.raw.video import VideoEncoder
from dexmani_real.data.raw.writer import RawEpisodeWriter
from dexmani_real.data.recorder import AsyncEpisodeRecorder, RecordingError, snapshot_recording_metadata
from dexmani_real.replay.trajectory import load_trajectory
from dexmani_real.robot.commands import RobotCommand
from dexmani_real.robot.feedback import ARM_STATE_DTYPE, HAND_STATE_DTYPE
from dexmani_real.robot.model import XARM7_JOINT_NAMES, XHAND_SDK_JOINT_NAMES
from dexmani_real.runtime.observation import ObservationRow
from dexmani_real.sensor.camera.geometry import CameraIntrinsics, CameraSnapshot
from dexmani_real.sensor.pointcloud.config import PointCloudConfig


def recording_metadata(*, extrinsics=True, reset="not_requested"):
    camera = CameraSnapshot(
        "synthetic", CameraIntrinsics(16, 16, 100., 100., 7.5, 7.5, "none", (0.,) * 5),
        .001, np.eye(4) if extrinsics else None,
    )
    return snapshot_recording_metadata(
        camera, tactile_reset_metadata=dict(hand_tactile_reset_result=reset),
        tactile_hold_max_age_s=ExperimentConfig().hand.feedback_max_age_s,
    )


def recording_row(index):
    config = ExperimentConfig()
    stamp = 1_000_000_000 + index * 62_500_000
    arm, hand = np.zeros(1, ARM_STATE_DTYPE), np.zeros(1, HAND_STATE_DTYPE)
    arm["qpos"], arm["timestamp_ns"] = config.arm.home_qpos, stamp
    hand["qpos"] = np.deg2rad(config.hand.home_qpos_deg)
    for name in ("timestamp_ns", "tactile_aggregate_timestamp_ns", "tactile_dense_timestamp_ns"):
        hand[name] = stamp
    hand["tactile_aggregate"], hand["tactile_dense"] = index + 1, index + 2
    camera = dict(timestamp_ns=stamp, ring_sequence=index + 1,
                  rgb=np.full((16, 16, 3), index, np.uint8), depth=np.full((16, 16), 500, np.uint16))
    row = ObservationRow(arm, hand, camera, None, None, stamp, control_timestamp_ns=stamp)
    return row, RobotCommand(1, arm["qpos"][0], hand["qpos"][0])


def record_episode(root, name, *, frames=4, wait=False, extrinsics=True, reset="not_requested"):
    recorder = AsyncEpisodeRecorder(root, control_hz=16)
    recorder.start_episode(task_label="synthetic", collection_source="teleop", episode_name=name,
                           metadata=recording_metadata(extrinsics=extrinsics, reset=reset))
    for index in range(frames):
        row, command = recording_row(index)
        recorder.add_frame(row, None if wait and index == 1 else command)
    return recorder.save_episode(reason="operator", technical_status="failed" if wait else "normal")


class RecordingChecks(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="dexmani-recording-check-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_roundtrip_wait_prefix_zero_and_optional_extrinsics(self):
        for frames, wait, extrinsics in ((0, False, False), (4, True, True), (4, False, False)):
            path = record_episode(self.root, f"episode_{frames}_{wait}", frames=frames,
                                  wait=wait, extrinsics=extrinsics)
            with EpisodeReader(path) as reader:
                self.assertEqual(reader.num_frames, frames)
                self.assertEqual("camera_T_xarm_base_from_color" in reader.meta, extrinsics)
                reader.validate_rows()
                if wait:
                    with self.assertRaises(RawDataError):
                        reader.validate_completion()
                    self.assertTrue(np.isnan(reader["action_arm_joint_target"][1]).all())
                    np.testing.assert_array_equal(reader["action_arm_joint_target"][2], recording_row(2)[1].arm_qpos)
                else:
                    reader.validate_completion()
                if frames:
                    self.assertEqual(len(list(reader.iter_rgb_frames())), frames)
                    np.testing.assert_array_equal(reader["hand_contact"][:, 0, 0], np.arange(1, frames + 1))
                    np.testing.assert_array_equal(reader["hand_tactile_force"][:, 0, 0, 0], np.arange(2, frames + 2))

    def test_capture_reset_metadata_is_taken_per_capture(self):
        for index, reset in enumerate(("not_requested", "completed")):
            path = record_episode(self.root, f"episode_{index}", reset=reset)
            with EpisodeReader(path) as reader:
                self.assertEqual(reader.meta["hand_tactile_reset_result"], reset)

    def test_timing_layout_is_required_even_for_empty_captures(self):
        for frames in (0, 4):
            source = record_episode(self.root, f"source_{frames}", frames=frames)
            for defect in ("missing", "group", "dtype", "count"):
                with self.subTest(frames=frames, defect=defect):
                    path = Path(shutil.copytree(source, self.root / f"timing_{frames}_{defect}"))
                    with h5py.File(path / "data.h5", "r+") as data:
                        del data["timing/rows"]
                        if defect == "group":
                            data.create_group("timing/rows")
                        elif defect == "dtype":
                            data.create_dataset("timing/rows", shape=(frames,), dtype=np.int64)
                        elif defect == "count":
                            data.create_dataset("timing/rows", shape=(frames + 1,), dtype=TIMING_ROW_DTYPE)
                    with EpisodeReader(path) as reader, \
                         patch("dexmani_real.data.raw.reader.validate_fixed_rate_rows",
                               side_effect=AssertionError("read before layout admission")):
                        with self.assertRaisesRegex(RawDataError, "timing"):
                            reader.validate_rows(chunk_frames=1)

    def test_camera_geometry_is_checked_at_raw_consumption(self):
        source = record_episode(self.root, "source")
        changes = [
            ("camera_T_xarm_base_from_color", transform.ravel())
            for transform in (np.eye(3), np.full((4, 4), np.nan), np.diag([2., 1., 1., 1.]))
        ]
        changes.extend(
            (name, value)
            for name in ("camera_color_width", "camera_color_height")
            for value in (0, -1, True, 16., "16", [16], None)
        )
        for index, (name, value) in enumerate(changes):
            path = Path(shutil.copytree(source, self.root / f"camera_{index}"))
            with h5py.File(path / "data.h5", "r+") as data:
                del data["meta"].attrs[name]
                if value is not None:
                    data["meta"].attrs[name] = value
            with EpisodeReader(path) as reader:
                with self.assertRaisesRegex(RawDataError, "camera geometry"):
                    validate_export_episode(reader)

    def test_joint_layout_is_rejected_before_feature_construction(self):
        source = record_episode(self.root, "source")
        for index, (name, shape, dtype) in enumerate((
            ("arm_qpos", (4, 6), np.float64),
            ("hand_qpos", (4, 11), np.float64),
            ("hand_qpos", (5, 12), np.float64),
            ("hand_qpos", (4, 12), np.float32),
        )):
            path = Path(shutil.copytree(source, self.root / f"layout_{index}"))
            with h5py.File(path / "data.h5", "r+") as data:
                del data[name]
                data.create_dataset(name, shape=shape, dtype=dtype)
            with EpisodeReader(path) as reader:
                with self.assertRaisesRegex(RawDataError, name):
                    validate_export_episode(reader)

    def test_row_count_and_fixed_rate_boundaries(self):
        self.assertFalse(validate_fixed_rate_rows(np.array([0., .078125]), 16))
        self.assertTrue(validate_fixed_rate_rows(np.array([0., .078125001]), 16))
        writer = RawEpisodeWriter(self.root, control_hz=16)
        writer.open(recording_metadata())
        try:
            writer.append_frame(*recording_row(0))
            writer.flush()
            with self.assertRaisesRegex(RuntimeError, "row count mismatch"):
                writer.write_termination("operator", [], expected_frames=2, technical_status="normal")
        finally:
            writer.close()

    def test_codec_flush_error_still_closes_container(self):
        video = VideoEncoder(self.root / "unused.mp4", fps=16, width=16, height=16)
        container = Mock()
        video.container, video.stream = container, Mock()
        video.stream.encode.side_effect = OSError("synthetic codec flush failure")
        with self.assertRaises(OSError):
            video.close()
        container.close.assert_called_once()
        self.assertTrue(video.closed)

    def test_start_timeout_retains_writer_until_completion(self):
        entered, release = threading.Event(), threading.Event()
        opened = RawEpisodeWriter.open
        def blocked(writer, metadata):
            entered.set()
            if not release.wait(2):
                raise TimeoutError("synthetic START not released")
            opened(writer, metadata)
        recorder = AsyncEpisodeRecorder(self.root, control_hz=16)
        with patch.object(RawEpisodeWriter, "open", blocked), \
             patch("dexmani_real.data.recorder.RECORDER_START_TIMEOUT_S", .02), \
             patch("dexmani_real.data.recorder.RECORDER_STOP_TIMEOUT_S", .02):
            try:
                with self.assertRaisesRegex(RecordingError, "START timed out"):
                    recorder.start_episode(task_label="synthetic", collection_source="teleop", metadata=recording_metadata())
                self.assertTrue(entered.is_set())
                self.assertTrue(recorder.writer_pending)
                self.assertFalse(recorder.is_recording)
            finally:
                release.set()
                recorder.thread.join(2)
        self.assertFalse(recorder.writer_pending)
        with self.assertRaises(RecordingError):
            recorder.close()
        self.assertTrue(recorder.staging_path.exists())
        self.assertIsNone(recorder.published_path)

    def test_reader_rejects_joint_semantics(self):
        source = record_episode(self.root, "source")
        changes = (
            ("joint_units", "deg"), ("joint_units", None),
            ("arm_joint_names", json.dumps(XARM7_JOINT_NAMES[::-1])),
            ("hand_joint_names", json.dumps(XHAND_SDK_JOINT_NAMES[::-1])),
            ("arm_joint_names", "not JSON"), ("hand_joint_names", None),
        )
        for index, (name, value) in enumerate(changes):
            with self.subTest(name=name, value=value):
                path = Path(shutil.copytree(source, self.root / f"joint_{index}"))
                with h5py.File(path / "data.h5", "r+") as data:
                    del data["meta"].attrs[name]
                    if value is not None:
                        data["meta"].attrs[name] = value
                with self.assertRaises(RawDataError):
                    with EpisodeReader(path):
                        pass

    def test_tactile_semantics_only_gate_tactile_consumers(self):
        source = record_episode(self.root, "source")
        for index, (name, wrong) in enumerate((
            ("hand_tactile_processing", "zero_fill"), ("hand_tactile_units", "N"),
            ("hand_tactile_axes", "world"), ("hand_tactile_finger_order", "pinky,ring,middle,index,thumb"),
            ("hand_tactile_hold_max_age_s", 0.), ("hand_tactile_hold_max_age_s", np.nan),
        )):
            for missing in (False, True):
                path = Path(shutil.copytree(source, self.root / f"tactile_{index}_{missing}"))
                with h5py.File(path / "data.h5", "r+") as data:
                    del data["meta"].attrs[name]
                    if not missing:
                        data["meta"].attrs[name] = wrong
                # Only load a trajectory; never construct a replayer or connect devices.
                self.assertEqual(load_trajectory(path).num_frames, 4)
                with EpisodeReader(path) as reader:
                    for field in ("hand_contact", "hand_tactile_force"):
                        with self.assertRaises(RawDataError):
                            reader[field]
                    with self.assertRaises(RawDataError):
                        reader.validate_rows()

    def test_fifo_full_preserves_accepted_prefix(self):
        entered, release = threading.Event(), threading.Event()
        append = RawEpisodeWriter.append_frame
        def blocked(writer, row, command):
            entered.set()
            if not release.wait(2):
                raise TimeoutError("synthetic writer not released")
            append(writer, row, command)
        recorder = AsyncEpisodeRecorder(self.root, control_hz=16)
        with patch.object(RawEpisodeWriter, "append_frame", blocked):
            try:
                recorder.start_episode(task_label="synthetic", collection_source="teleop", metadata=recording_metadata())
                row, command = recording_row(0)
                recorder.add_frame(row, command)
                self.assertTrue(entered.wait(1))
                row.arm["qpos"], row.hand["qpos"] = np.nan, np.nan
                row.hand["tactile_aggregate"] = np.nan
                row.camera["depth"][:] = 0
                for index in range(1, 17):
                    recorder.add_frame(*recording_row(index))
                with self.assertRaisesRegex(RecordingError, "queue full"):
                    recorder.add_frame(*recording_row(17))
                release.set()
                path = recorder.save_episode(reason="recording_failure", technical_status="failed")
            finally:
                release.set()
                if recorder.thread is not None:
                    recorder.thread.join(2)
        self.assertFalse(recorder.writer_pending)
        with EpisodeReader(path) as reader:
            self.assertEqual(reader.num_frames, 17)
            self.assertEqual(reader.meta["technical_status"], "failed")
            self.assertEqual(len(list(reader.iter_rgb_frames())), 17)
            np.testing.assert_array_equal(reader["timestamp"][:], np.arange(17) / 16)
            original, _ = recording_row(0)
            np.testing.assert_array_equal(reader["arm_qpos"][0], original.arm["qpos"][0])
            np.testing.assert_array_equal(reader["hand_qpos"][0], original.hand["qpos"][0])
            np.testing.assert_array_equal(reader["hand_contact"][0], original.hand["tactile_aggregate"][0])
            np.testing.assert_array_equal(reader["depth"][0], original.camera["depth"])

    def test_io_failures_retain_staging_and_close_files(self):
        for stage in ("open", "append_frame", "flush", "close", "encode", "video_close", "publish"):
            with self.subTest(stage=stage):
                root = self.root / stage
                recorder = AsyncEpisodeRecorder(root, control_hz=16)
                target, method = (
                    (VideoEncoder, "write_frame") if stage == "encode" else
                    (VideoEncoder, "close") if stage == "video_close" else
                    (RawEpisodeWriter, stage) if stage != "publish" else (None, None)
                )
                if target is not None:
                    original = getattr(target, method)
                    def fail(owner, *args, **kwargs):
                        if stage in {"open", "close", "video_close"}:
                            original(owner, *args, **kwargs)
                        raise OSError(f"synthetic {stage} failure")
                    context = patch.object(target, method, fail)
                else:
                    context = patch("dexmani_real.data.recorder.atomic_publish", side_effect=OSError("synthetic publish failure"))
                with context:
                    if stage == "open":
                        with self.assertRaises(RecordingError):
                            recorder.start_episode(task_label="synthetic", collection_source="teleop", metadata=recording_metadata())
                    else:
                        recorder.start_episode(task_label="synthetic", collection_source="teleop", metadata=recording_metadata())
                        recorder.add_frame(*recording_row(0))
                    with self.assertRaises(RecordingError):
                        recorder.save_episode()
                self.assertFalse(recorder.writer_pending)
                self.assertIsNone(recorder.published_path)
                self.assertTrue(recorder.staging_path.is_dir())
                self.assertTrue((recorder.staging_path / "capture_failure.json").is_file())
                self.assertFalse(recorder.writer.meta.id.valid)
                self.assertTrue(recorder.writer.video.closed)

    def test_timeout_keeps_writer_owner_and_first_finish(self):
        entered, release = threading.Event(), threading.Event()
        append = RawEpisodeWriter.append_frame
        def blocked(writer, *args):
            entered.set()
            if not release.wait(2):
                raise TimeoutError("synthetic writer not released")
            append(writer, *args)
        recorder = AsyncEpisodeRecorder(self.root, control_hz=16)
        with patch.object(RawEpisodeWriter, "append_frame", blocked), \
             patch("dexmani_real.data.recorder.RECORDER_STOP_TIMEOUT_S", .02):
            try:
                recorder.start_episode(task_label="synthetic", collection_source="teleop", metadata=recording_metadata())
                recorder.add_frame(*recording_row(0))
                self.assertTrue(entered.wait(1))
                with self.assertRaises(RecordingError):
                    recorder.save_episode(reason="operator")
                first = recorder.finish_request
                self.assertTrue(recorder.writer_pending)
                self.assertTrue(recorder.staging_path.exists())
                with self.assertRaises(RecordingError):
                    recorder.close()
                self.assertIs(recorder.finish_request, first)
            finally:
                release.set()
                recorder.thread.join(2)
        self.assertFalse(recorder.writer_pending)
        self.assertIsNone(recorder.published_path)
        self.assertTrue(recorder.staging_path.exists())

    def test_explicit_discard_and_repeated_close(self):
        kept = record_episode(self.root, "kept")
        recorder = AsyncEpisodeRecorder(self.root, control_hz=16)
        recorder.start_episode(task_label="synthetic", collection_source="teleop", metadata=recording_metadata())
        recorder.add_frame(*recording_row(0))
        staging = recorder.staging_path
        recorder.discard_episode()
        with patch.object(recorder, "finish", side_effect=AssertionError("terminal writer re-finalized")):
            recorder.close()
        self.assertFalse(staging.exists())
        self.assertTrue(kept.exists())

    def test_canonical_full_modal_rejection_and_rollback(self):
        source = self.root / "raw"
        valid = record_episode(source, "episode_00")
        nan = Path(shutil.copytree(valid, source / "episode_01_nan"))
        extra = Path(shutil.copytree(valid, source / "episode_02_rgb"))
        drift = Path(shutil.copytree(valid, source / "episode_03_clock"))
        shutil.copytree(valid, source / "episode_04")
        record_episode(source, "episode_05_zero", frames=0)
        with h5py.File(nan / "data.h5", "r+") as data:
            data["action_arm_joint_target"][3, 0] = np.nan
        video = VideoEncoder(extra / "rgb.mp4", fps=16, width=16, height=16)
        video.open()
        try:
            for index in range(5):
                video.write_frame(recording_row(index)[0].camera["rgb"])
        finally:
            video.close()
        with h5py.File(drift / "data.h5", "r+") as data:
            data["timestamp"][:] += np.arange(4) * .006
            timing = data["timing/rows"][:]
            timing["observation_ns"] += np.arange(4) * 6_000_000
            data["timing/rows"][:] = timing
        before = {path: path.read_bytes() for path in source.glob("*/*") if path.is_file()}
        config = ProcessingConfig(pointcloud=PointCloudConfig(remove_table=False, num_points=16,
                                  workspace=(-1., -1., 0., 1., 1., 2.)))
        destination = self.root / "canonical.zarr"
        report = CanonicalExporter(config, chunk_frames=2).export(source, destination)
        self.assertEqual((report["episode_count"], report["total_frames"]), (2, 8))
        self.assertEqual(set(report["rejected_episodes"]),
                         {"episode_01_nan", "episode_02_rgb", "episode_03_clock", "episode_05_zero"})
        output = zarr.open_group(str(destination), mode="r")
        specs = canonical_array_specs(16, 16, 16, config.hand_pointcloud.num_points)
        self.assertEqual(set(output["data"]), set(specs))
        self.assertEqual(len(specs), 13)
        np.testing.assert_array_equal(output["meta/episode_ends"][:], [4, 8])
        for name, (tail, dtype) in specs.items():
            values = output[f"data/{name}"][:]
            self.assertEqual((values.shape, values.dtype), ((8, *tail), dtype))
            if values.dtype.kind == "f":
                self.assertTrue(np.isfinite(values).all())
        for path, original in before.items():
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
