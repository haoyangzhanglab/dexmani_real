"""Offline controller Queue ownership and publication boundary regressions."""

from queue import Empty, Queue
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from dexmani_real.ipc.schema import make_record_sample_dtype
from dexmani_real.recording.client import (
    RecorderClient,
    RecordingFinished,
    RecordingStarted,
    StartRecording,
    StopRecording,
)
from dexmani_real.recording.sample import EpisodeAction, build_episode_state


class SampleRing:
    dtype = make_record_sample_dtype((2, 2, 3), (2, 2))
    maxlen = 4

    def __init__(self):
        self.latest_sequence = 0

    def write(self, frame):
        self.latest_sequence += 1


def client():
    shared = SimpleNamespace(
        record_control_q=Queue(),
        record_result_q=Queue(),
        record_sample_ring=SampleRing(),
        recorder_consumed_sequence=SimpleNamespace(value=0),
        error_state=SimpleNamespace(value=False),
        is_running=SimpleNamespace(value=True),
        is_ready=lambda name: True,
        set_heartbeat=mock.Mock(),
    )
    return RecorderClient(shared)


def add(recorder):
    return recorder.add_frame(
        build_episode_state(None, None, timestamp_s=1),
        EpisodeAction(np.zeros(7), np.zeros(12)),
        {"wrist_pos": np.zeros(3), "wrist_quat_wxyz": np.array([1, 0, 0, 0]),
         "landmarks": np.zeros((21, 3))},
        arm_qpos_sent=np.zeros(7),
    )


def test_start_ack_precedes_production_and_stop_captures_last_commit():
    recorder = client()
    assert not add(recorder)
    recorder.shared.record_result_q.put(
        RecordingStarted("episode", 2, "eval:invalid:max_frames")
    )
    assert recorder.start_episode(task_label="task", operator="operator")
    assert recorder.shared.record_control_q.get_nowait() == StartRecording(
        "task", "operator", 1
    )
    assert add(recorder)
    assert add(recorder)
    assert not recorder.is_recording
    assert not add(recorder)
    assert recorder.shared.record_control_q.get_nowait() == StopRecording(
        True, "eval:invalid:max_frames", 2
    )
    assert recorder.stop_pending
    assert recorder.poll_stop().reason == "eval:invalid:max_frames"
    assert not recorder.start_episode()
    recorder.shared.record_result_q.put(
        RecordingFinished(True, "episode", 2, "eval:invalid:max_frames")
    )
    assert recorder.poll_stop().done
    assert not recorder.poll_stop().done
    recorder.shared.record_result_q.put(RecordingStarted("episode2", 2, "max_frames"))
    assert recorder.start_episode()
    assert recorder.shared.record_control_q.get_nowait().start_sequence == 3


def test_stop_revokes_production_before_sequence_snapshot():
    recorder = client()
    recorder._recording = True

    class BoundaryRing:
        @property
        def latest_sequence(self):
            assert not recorder.is_recording
            return 9

    recorder.shared.record_sample_ring = BoundaryRing()
    recorder.stop_episode(save=False, reason="discard")
    assert recorder.shared.record_control_q.get_nowait() == StopRecording(
        False, "discard", 9
    )


def test_full_ring_dooms_episode_without_overwrite():
    recorder = client()
    recorder._recording = True
    recorder.shared.record_sample_ring.latest_sequence = 4
    assert not add(recorder)
    assert recorder.shared.record_sample_ring.latest_sequence == 4
    assert recorder.shared.record_control_q.get_nowait() == StopRecording(
        False, "sample_ring_overflow", 4
    )


def test_start_timeout_latches_unavailable_without_cancel_or_retry():
    recorder = client()
    with mock.patch("dexmani_real.recording.client.RECORDER_START_TIMEOUT_S", 0):
        assert not recorder.start_episode()
    assert recorder.shared.error_state.value
    assert isinstance(recorder.shared.record_control_q.get_nowait(), StartRecording)
    assert not recorder.start_episode()
    with pytest.raises(Empty):
        recorder.shared.record_control_q.get_nowait()


def test_writer_failure_is_consumed_once_and_never_saved():
    recorder = client()
    recorder._recording = True
    recorder.shared.record_result_q.put(
        RecordingFinished(False, "episode", 1, "sample_write_error", "disk full")
    )
    result = recorder.poll_stop()
    assert result.done and result.error == "disk full" and not result.saved
    assert not recorder.is_recording
    assert recorder.camera_writer_error == "disk full"
    assert not recorder.poll_stop().done


def test_dead_worker_stop_timeout_surfaces_to_supervisor():
    recorder = client()
    recorder._recording = True
    recorder.stop_episode()
    result = recorder.join_stop(timeout=0)
    assert result.error and not result.done
    assert recorder.shared.error_state.value
    assert not recorder.start_episode()


def test_control_queue_full_aborts_without_blocking_or_restarting():
    recorder = client()
    recorder.shared.record_control_q = Queue(maxsize=1)
    recorder.shared.record_control_q.put("occupied")
    assert not recorder.start_episode()
    assert recorder.shared.error_state.value
    assert not recorder.start_episode()


def test_start_episode_wires_explicit_episode_name():
    recorder = client()
    recorder.shared.record_result_q.put(
        RecordingStarted("episode_001", 2, "max_frames")
    )
    assert recorder.start_episode(
        task_label="task", operator="op", episode_name="episode_001"
    )
    assert recorder.shared.record_control_q.get_nowait() == StartRecording(
        "task", "op", 1, "episode_001"
    )


def test_default_start_recording_keeps_three_positional_equality():
    assert StartRecording("task", "op", 1) == StartRecording("task", "op", 1, None)


def test_start_refusal_exposes_last_error():
    recorder = client()
    recorder.shared.record_result_q.put(
        RecordingFinished(
            False,
            None,
            0,
            "start_error",
            error="explicit episode name already exists: /x/episode_001",
        )
    )
    assert not recorder.start_episode(episode_name="episode_001")
    assert "already exists" in recorder.last_error
