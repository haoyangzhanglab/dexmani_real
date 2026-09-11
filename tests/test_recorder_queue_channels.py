"""Offline checks for the recorder queue and sample-ring IPC boundaries."""

from __future__ import annotations

import multiprocessing as mp
import os
from queue import Full

import numpy as np
import pytest

from dexmani_real.ipc.channels import RuntimeChannels, RuntimeChannelsConfig
from dexmani_real.ipc.schema import make_record_sample_dtype
from dexmani_real.recording.client import RecordingFinished, StartRecording


def _put_recording_command(queue) -> None:
    queue.put(StartRecording(task="fixture", operator="test", start_sequence=7))


def _tiny_channels_config() -> RuntimeChannelsConfig:
    return RuntimeChannelsConfig(
        camera_ring_maxlen=1,
        vr_ring_maxlen=1,
        arm_state_ring_maxlen=1,
        hand_state_ring_maxlen=1,
        coupled_cmd_ring_maxlen=1,
        record_sample_ring_maxlen=1,
        pointcloud_ring_maxlen=1,
        camera_rgb_shape=(2, 2, 3),
        camera_depth_shape=(2, 2),
        arm_home_q_maxsize=1,
    )


def test_sample_dtype_has_no_recorder_lifecycle_envelope():
    dtype = make_record_sample_dtype((2, 2, 3), (2, 2))

    assert "generation" not in dtype.names
    assert "sample_sequence" not in dtype.names
    assert dtype.names[0] == "timestamp"
    assert dtype.fields["timestamp"][0] == np.dtype("<f8")


def test_recorder_queues_are_bounded_and_spawn_compatible():
    context = mp.get_context("spawn")
    shared = RuntimeChannels.create(
        prefix=f"test_recorder_queue_{os.getpid()}",
        config=_tiny_channels_config(),
        mp_context=context,
    )
    child = context.Process(
        target=_put_recording_command,
        args=(shared.record_control_q,),
    )
    try:
        assert not hasattr(shared, "record_control_ring")
        assert not hasattr(shared, "record_status_ring")

        child.start()
        assert shared.record_control_q.get(timeout=5) == StartRecording(
            task="fixture", operator="test", start_sequence=7
        )
        child.join(timeout=5)
        assert child.exitcode == 0

        control_messages = [
            StartRecording(task="fixture", operator="test", start_sequence=i)
            for i in range(8)
        ]
        for message in control_messages:
            shared.record_control_q.put_nowait(message)
        with pytest.raises(Full):
            shared.record_control_q.put_nowait(
                StartRecording(task="overflow", operator="test", start_sequence=9)
            )
        for expected in control_messages:
            assert shared.record_control_q.get(timeout=5) == expected

        result = RecordingFinished(
            saved=True,
            path="/tmp/episode",
            frame_count=8,
            reason="manual",
        )
        shared.record_result_q.put_nowait(result)
        assert shared.record_result_q.get(timeout=5) == result
    finally:
        if child.is_alive():
            child.terminate()
            child.join(timeout=5)
        assert shared.close()
