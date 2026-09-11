"""Focused RecorderIO-boundary regressions for the raw transaction contract.

No hardware, no trained checkpoint.  These tests pin the owner-boundary contract:
RecorderIO announces capacity and its configured max_frames stop reason, the
rollout recording config threads capacity through it, and a real
EpisodeRecorder transaction publishes a structurally valid raw-v29 episode.
"""

from __future__ import annotations

import types
import unittest
from unittest import mock

import numpy as np

from dexmani_real.deployment.config import RolloutRecordingConfig
from dexmani_real.deployment.lifecycle import _rollout_recorder_config
from dexmani_real.recording.io_worker import RecorderIOConfig


class TestMaxFramesStopReason(unittest.TestCase):
    def test_default_reason_is_plain_max_frames(self):
        config = RecorderIOConfig(
            data_dir="/tmp/r", max_frames=10, control_hz=16.0, min_frames=1
        )
        self.assertEqual(config.max_frames_stop_reason, "max_frames")

    def test_empty_reason_is_rejected(self):
        with self.assertRaises(ValueError):
            RecorderIOConfig(
                data_dir="/tmp/r",
                max_frames=10,
                control_hz=16.0,
                min_frames=1,
                max_frames_stop_reason="   ",
            )

    def test_rollout_recorder_config_uses_default_max_frames_reason(self):
        runtime = types.SimpleNamespace(
            policy=types.SimpleNamespace(control_hz=16.0),
            camera=types.SimpleNamespace(writer_queue_size=8),
        )
        rollout = RolloutRecordingConfig(
            data_dir="/tmp/rollout_data",
            task_label="task",
            operator="op",
            max_running_s=30.0,
        )
        worker_config = types.SimpleNamespace(
            experiment="policy/task/exp",
            artifact="epoch_500-deployment.pt",
            inference_steps=2,
            seed=0,
        )
        config = _rollout_recorder_config(runtime, rollout, worker_config)
        self.assertEqual(config.max_frames_stop_reason, "max_frames")
        self.assertEqual(config.data_dir, "/tmp/rollout_data")
        self.assertEqual(
            config.provenance["checkpoint_name"], "epoch_500-deployment.pt"
        )
        self.assertEqual(config.provenance["inference_steps"], "2")


class TestSaveOutcome(unittest.TestCase):
    def test_save_commits_real_raw_transaction_with_technical_reason(self):
        """Exercise disk IO and codec validation, with no sensors or SDKs."""
        import tempfile
        from pathlib import Path

        import h5py

        from dexmani_real.recording.recorder import EpisodeRecorder
        from dexmani_real.recording.sample import EpisodeAction, build_episode_state
        from dexmani_real.recording.storage.camera_writer import (
            CameraStreamWriterConfig,
        )

        with tempfile.TemporaryDirectory() as directory:
            recorder = EpisodeRecorder(
                directory,
                max_frames=8,
                min_frames=1,
                control_hz=16.0,
                camera_writer_config=CameraStreamWriterConfig(
                    rgb_shape=(16, 16, 3), depth_shape=(16, 16), fps=16.0, queue_size=8
                ),
            )
            try:
                self.assertTrue(
                    recorder.start_episode(task_label="offline", operator="test")
                )
                episode_path = Path(recorder.episode_path)
                state = build_episode_state(None, None, timestamp_s=1.0)
                action = EpisodeAction(np.zeros(7), np.zeros(12))
                self.assertTrue(
                    recorder.add_frame(
                        state,
                        action,
                        {
                            "wrist_pos": np.full(3, np.nan),
                            "wrist_quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0]),
                            "landmarks": np.full((21, 3), np.nan),
                        },
                        signals={
                            "frame_status": 1,
                            "action_queued": False,
                            "observation_anchor_monotonic_ns": 1_000_000_000,
                        },
                        arm_qpos_sent=np.zeros(7),
                    )
                )
                self.assertEqual(
                    recorder.finish_episode(save=True, reason="operator"),
                    str(episode_path),
                )
                self.assertTrue(episode_path.is_dir())
                self.assertTrue((episode_path / "rgb.mp4").is_file())
                self.assertTrue((episode_path / "depth.h5").is_file())
                with h5py.File(episode_path / "data.h5", "r") as raw:
                    self.assertEqual(raw["meta"].attrs["schema_version"], 29)
                    self.assertEqual(raw["meta"].attrs["num_frames"], 1)
                    self.assertTrue(raw["meta"].attrs["success"])
                    self.assertEqual(raw["meta"].attrs["stop_reason"], "operator")
            finally:
                if recorder.is_recording:
                    recorder.finish_episode(save=False, reason="test_cleanup")


class TestExplicitEpisodeName(unittest.TestCase):
    """Episode naming contract: timestamp default, explicit exact names.

    Real ``EpisodeRecorder`` transactions on a temporary directory; no
    devices, no workers.  Explicit names are the policy multi-episode
    sequence (``episode_001``...); ``None`` must keep teleop naming intact.
    """

    @staticmethod
    def _recorder(directory):
        from dexmani_real.recording.recorder import EpisodeRecorder
        from dexmani_real.recording.storage.camera_writer import (
            CameraStreamWriterConfig,
        )

        return EpisodeRecorder(
            directory,
            max_frames=8,
            min_frames=1,
            control_hz=16.0,
            camera_writer_config=CameraStreamWriterConfig(
                rgb_shape=(16, 16, 3), depth_shape=(16, 16), fps=16.0, queue_size=8
            ),
        )

    @staticmethod
    def _add_one_frame(recorder):
        from dexmani_real.recording.sample import EpisodeAction, build_episode_state

        state = build_episode_state(None, None, timestamp_s=1.0)
        action = EpisodeAction(np.zeros(7), np.zeros(12))
        return recorder.add_frame(
            state,
            action,
            {
                "wrist_pos": np.full(3, np.nan),
                "wrist_quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0]),
                "landmarks": np.full((21, 3), np.nan),
            },
            signals={
                "frame_status": 1,
                "action_queued": False,
                "observation_anchor_monotonic_ns": 1_000_000_000,
            },
            arm_qpos_sent=np.zeros(7),
        )

    def test_none_keeps_timestamp_naming(self):
        import re
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            recorder = self._recorder(directory)
            try:
                self.assertTrue(recorder.start_episode())
                name = Path(recorder.episode_path).name
                self.assertRegex(name, re.compile(r"^episode_\d{8}_\d{6}$"))
            finally:
                if recorder.is_recording:
                    recorder.finish_episode(save=False, reason="test_cleanup")

    def test_explicit_name_publishes_exact_directory(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            recorder = self._recorder(directory)
            self.assertTrue(recorder.start_episode(episode_name="episode_001"))
            self.assertTrue(self._add_one_frame(recorder))
            published = Path(
                recorder.finish_episode(save=True, reason="operator")
            )
            self.assertEqual(published.name, "episode_001")
            self.assertEqual(published.parent, Path(directory))
            for filename in ("data.h5", "depth.h5", "rgb.mp4"):
                self.assertTrue((published / filename).is_file())

    def test_duplicate_explicit_name_fails_loudly_without_clobber(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            recorder = self._recorder(directory)
            self.assertTrue(recorder.start_episode(episode_name="episode_001"))
            self.assertTrue(self._add_one_frame(recorder))
            recorder.finish_episode(save=True, reason="operator")
            before = sorted(p.name for p in Path(directory).iterdir())
            with self.assertRaises(FileExistsError):
                recorder.start_episode(episode_name="episode_001")
            # A stale temporary directory blocks the name as well.
            (Path(directory) / ".tmp_episode_002").mkdir()
            with self.assertRaises(FileExistsError):
                recorder.start_episode(episode_name="episode_002")
            self.assertEqual(
                sorted(p.name for p in Path(directory).iterdir()),
                sorted(before + [".tmp_episode_002"]),
            )

    def test_invalid_explicit_names_rejected(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            recorder = self._recorder(directory)
            for name in ("../evil", "", "a/b", "a\\b", ".hidden", ".", ".."):
                with self.subTest(name=name):
                    with self.assertRaises(ValueError):
                        recorder.start_episode(episode_name=name)

    def test_second_episode_after_finalize_uses_next_name(self):
        import tempfile
        from pathlib import Path

        from dexmani_real.recording.storage.reader import EpisodeReader

        with tempfile.TemporaryDirectory() as directory:
            recorder = self._recorder(directory)
            for index in (1, 2):
                name = f"episode_{index:03d}"
                self.assertTrue(recorder.start_episode(episode_name=name))
                self.assertTrue(self._add_one_frame(recorder))
                published = Path(recorder.finish_episode(save=True, reason="operator"))
                self.assertEqual(published.name, name)
                with EpisodeReader(published) as reader:
                    self.assertEqual(reader.schema_version, 29)
                    self.assertEqual(reader.h5f["meta"].attrs["stop_reason"], "operator")


if __name__ == "__main__":
    unittest.main()
