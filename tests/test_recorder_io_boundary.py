"""Focused RecorderIO-boundary regressions for the formal max_frames stop reason.

No hardware, no trained checkpoint.  These tests pin the owner-boundary contract:
RecorderIO auto-finalizes capacity exhaustion with the configured stop reason, and
the formal-evaluation config threads ``eval:invalid:max_frames`` through it.  They
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
    PolicyEvaluationConfig,
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
        evaluation = PolicyEvaluationConfig(
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


if __name__ == "__main__":
    unittest.main()
