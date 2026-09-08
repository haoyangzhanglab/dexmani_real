"""Focused RecorderIO-boundary regressions for the formal max_frames stop reason.

No hardware, no trained checkpoint.  These tests pin the owner-boundary contract:
RecorderIO auto-finalizes capacity exhaustion with the configured stop reason, and
the rollout recording config threads ``eval:invalid:max_frames`` through it. They
exercise the io_worker path directly rather than mocking the executor's intent.
"""

from __future__ import annotations

import types
import unittest
from unittest import mock

import numpy as np

import dexmani_real.recording.io_worker as io_worker
from dexmani_real.deployment.evaluation import (
    EVALUATION_MAX_FRAMES_STOP_REASON,
    RolloutRecordingConfig,
)
from dexmani_real.deployment.lifecycle import _evaluation_recorder_config
from dexmani_real.recording.io_worker import RecorderIOConfig, _RecorderIOSession


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

    def test_oversized_reason_is_rejected(self):
        with self.assertRaises(ValueError):
            RecorderIOConfig(
                data_dir="/tmp/r",
                max_frames=10,
                control_hz=16.0,
                min_frames=1,
                max_frames_stop_reason="x" * 600,
            )

    def test_evaluation_recorder_config_threads_formal_reason(self):
        runtime = types.SimpleNamespace(
            policy=types.SimpleNamespace(control_hz=16.0),
            camera=types.SimpleNamespace(writer_queue_size=8),
        )
        evaluation = RolloutRecordingConfig(
            data_dir="/tmp/eval_data",
            task_label="task",
            operator="op",
            max_running_s=30.0,
        )
        config = _evaluation_recorder_config(runtime, evaluation)
        self.assertEqual(
            config.max_frames_stop_reason, EVALUATION_MAX_FRAMES_STOP_REASON
        )


class TestAutoFinalizationReason(unittest.TestCase):
    def test_auto_finalization_uses_configured_reason(self):
        config = RecorderIOConfig(
            data_dir="/tmp/r",
            max_frames=10,
            control_hz=16.0,
            min_frames=1,
            max_frames_stop_reason=EVALUATION_MAX_FRAMES_STOP_REASON,
        )
        session = _RecorderIOSession.__new__(_RecorderIOSession)
        session.config = config
        session.recorder = mock.Mock()
        session.recorder.is_recording = True
        session.recorder.add_episode_frame.return_value = False
        session.recorder.max_frames_reached = True
        session.recorder.camera_writer_error = None
        session.recorder.frame_count = 3
        session.recorder.stop_episode.return_value = "/tmp/r/episode"
        session.active_generation = 1
        session.last_sample_sequence = 0
        session.sample_backlog_high_watermark = 0
        session.pending_finalization = None
        session.failure_count = 0

        ring = mock.Mock()
        ring.latest_sequence = 1
        ring.maxlen = 8
        record = np.zeros(1, dtype=[("generation", "<i8")])
        record["generation"] = 1
        ring.read_sequence.return_value = (record, 1000, 1)
        shared = mock.Mock()
        shared.record_sample_ring = ring
        shared.recorder_consumed_sequence = mock.Mock()
        session.shared = shared

        with mock.patch.object(
            io_worker, "decode_record_sample", return_value=mock.Mock()
        ):
            session._drain_samples()

        session.recorder.stop_episode.assert_called_once()
        self.assertEqual(
            session.recorder.stop_episode.call_args.kwargs["reason"],
            EVALUATION_MAX_FRAMES_STOP_REASON,
        )


class TestSaveOutcome(unittest.TestCase):
    def test_failure_commits_real_raw_v24_transaction(self):
        """Exercise disk IO and codec validation, with no sensors or SDKs."""
        import json
        import tempfile
        from pathlib import Path

        import h5py

        from dexmani_real.deployment.evaluation import (
            EvaluationOutcome,
            write_rollout_result,
        )
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
                            "held": True,
                            "action_queued": False,
                            "observation_id": 1,
                            "observation_anchor_monotonic_ns": 1_000_000_000,
                        },
                        control_run_generation=1,
                    )
                )
                recorder.stop_episode(save=True, reason="eval:failure:operator")
                self.assertTrue(recorder.join_stop(timeout=10.0), recorder.stop_error)
                self.assertTrue(episode_path.is_dir())
                self.assertTrue((episode_path / "rgb.mp4").is_file())
                self.assertTrue((episode_path / "depth.h5").is_file())
                with h5py.File(episode_path / "data.h5", "r") as raw:
                    self.assertEqual(raw["meta"].attrs["schema_version"], 24)
                    self.assertEqual(raw["meta"].attrs["num_frames"], 1)
                    self.assertTrue(raw["meta"].attrs["success"])
                    self.assertEqual(
                        raw["meta"].attrs["stop_reason"], "eval:failure:operator"
                    )
                config = RolloutRecordingConfig(
                    directory, "offline", "test", 60.0, provenance={"eval_seed": "1"}
                )
                result = write_rollout_result(
                    config,
                    episode_path,
                    outcome=EvaluationOutcome.FAILURE,
                    stop_reason="eval:failure:operator",
                    duration_s=1.0,
                    metrics={},
                    saved=True,
                )
                self.assertEqual(json.loads(result.read_text())["outcome"], "failure")
            finally:
                if recorder.is_recording:
                    recorder.stop_episode(save=False, reason="test_cleanup")
                recorder.join_stop(timeout=10.0)

    def test_task_failure_is_saved_and_result_is_failure(self):
        import json
        import tempfile
        from pathlib import Path
        from dexmani_real.deployment.evaluation import (
            EvaluationOutcome,
            write_rollout_result,
        )
        from dexmani_real.recording.client import RecorderClient, RecorderCommand

        shared = mock.Mock()
        client = RecorderClient(shared)
        client._recording = True
        with mock.patch.object(client, "_write_control") as write:
            client.stop_episode(save=True, reason="eval:failure:operator")
        write.assert_called_once_with(
            RecorderCommand.STOP, save=True, stop_reason="eval:failure:operator"
        )
        with tempfile.TemporaryDirectory() as directory:
            config = RolloutRecordingConfig(
                directory, "task", "me", 60.0, provenance={"eval_seed": "1"}
            )
            path = write_rollout_result(
                config,
                Path(directory) / "episode_test",
                outcome=EvaluationOutcome.FAILURE,
                stop_reason="operator_failure",
                duration_s=2.0,
                metrics={},
                saved=True,
            )
            payload = json.loads(path.read_text())
            self.assertEqual(payload["outcome"], "failure")
            self.assertTrue(payload["raw_saved"])
            self.assertEqual(payload["eval_seed"], 1)


if __name__ == "__main__":
    unittest.main()
