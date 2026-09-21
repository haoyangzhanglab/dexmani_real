"""Real spawn/IPC and temporary media; no devices or model resources."""
import multiprocessing as mp
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from dexmani_real.recording.client import RecorderClient, RecordingFinished
from dexmani_real.recording.io_worker import RecorderIOConfig, _RecorderIOSession
from dexmani_real.recording.recorder import EpisodeFinalizationError
from dexmani_real.runtime.status import ExitReason
from dexmani_real.runtime.supervisor import supervisor_exit_reason
from test_recording_preservation import _make_recorder, _add_frames


def Shared(ctx):
    # The production allocator, including all synchronization choices.
    from uuid import uuid4
    from dexmani_real.ipc.channels import RuntimeChannels, RuntimeChannelsConfig
    return RuntimeChannels.create(
        prefix=f"recorder_test_{uuid4().hex}", mp_context=ctx,
        config=RuntimeChannelsConfig(camera_rgb_shape=(8, 8, 3),
                                     camera_depth_shape=(8, 8), camera_ring_maxlen=2),
    )


def close_in_child(shared, directory, entered, release, completed=None, exit_allowed=None):
    recorder = _make_recorder(Path(directory), min_frames=1)
    recorder.start_episode(task_label='test', episode_name='episode_spawn')
    _add_frames(recorder, 2)
    original = recorder._camera_writer.close
    def blocked_close(*args, **kwargs):
        entered.set()
        if not release.wait(15):
            raise RuntimeError('test did not release close')
        return original(*args, **kwargs)
    recorder._camera_writer.close = blocked_close
    session = _RecorderIOSession(shared, RecorderIOConfig(
        data_dir=directory, max_frames=10, control_hz=16, min_frames=1), recorder)
    session._begin_finalization(save=True, reason='manual')
    if completed is not None:
        completed.set()
        assert exit_allowed.wait(15)


@pytest.mark.parametrize('terminate', [False, True])
def test_real_process_close_has_one_fixed_deadline_and_preserves_staging(tmp_path, terminate):
    ctx = mp.get_context('spawn')
    shared = Shared(ctx)
    entered, release = ctx.Event(), ctx.Event()
    process = ctx.Process(name='recorder', target=close_in_child,
                          args=(shared, str(tmp_path), entered, release))
    process.start()
    try:
        assert entered.wait(10)
        deadline = shared.recorder_finish_deadline_ns.value
        assert deadline > time.monotonic_ns()
        assert not (tmp_path / 'episode_spawn').exists()
        assert list(tmp_path.glob('.tmp_*'))
        # Normal close suspends only recorder loop heartbeat, never hardware health.
        args = (shared, [process], {'recorder': 1000, 'arm': 0}, {'recorder': 1, 'arm': 1})
        assert supervisor_exit_reason(*args, service_process_names={'recorder'}) is ExitReason.NONE
        assert shared.recorder_finish_deadline_ns.value == deadline
        assert supervisor_exit_reason(shared, [process], {'recorder': 1000, 'arm': 2},
            {'recorder': 1, 'arm': 1}, service_process_names={'recorder'}) is ExitReason.HEARTBEAT_TIMEOUT
        if terminate:
            with patch('dexmani_real.runtime.supervisor.time.monotonic_ns', return_value=deadline):
                assert supervisor_exit_reason(*args, service_process_names={'recorder'}) is ExitReason.EVIDENCE_FAILURE
            assert shared.recorder_transport_failed.value
            process.terminate()
            process.join(5)
            assert not process.is_alive()
            client = RecorderClient(shared)
            # Simulate an unusable Queue after terminate; it must never be read.
            with patch.object(shared.record_result_q, 'get_nowait', side_effect=AssertionError('dead IPC read')):
                assert client.poll_stop().error
            assert not (tmp_path / 'episode_spawn').exists()
            assert list(tmp_path.glob('.tmp_*'))
        else:
            release.set()
            result = shared.record_result_q.get(timeout=10)
            assert isinstance(result, RecordingFinished)
            assert result.saved and not result.error and result.frame_count == 2
            process.join(5)
            assert process.exitcode == 0
            assert shared.recorder_finish_deadline_ns.value == 0
            assert (tmp_path / 'episode_spawn').is_dir()
            from queue import Empty
            with pytest.raises(Empty):
                shared.record_result_q.get_nowait()
    finally:
        # Never reacquire an Event lock after terminating its owner.
        if process.is_alive():
            process.terminate()
        process.join(5)
        assert shared.close()


def test_late_close_cannot_publish_complete(tmp_path):
    recorder = _make_recorder(tmp_path, min_frames=1)
    recorder.start_episode(task_label='test', episode_name='late')
    _add_frames(recorder, 2)
    with pytest.raises(EpisodeFinalizationError):
        recorder.finish_episode(True, 'manual', deadline_monotonic_ns=1)
    assert recorder.resources_released
    assert not recorder.last_finish_saved
    assert not (tmp_path / 'late').exists()
    assert (tmp_path / 'incomplete_late').is_dir()


def hold_recorder_access(shared, field, entered, release):
    from contextlib import nullcontext
    primitive = getattr(shared, field)
    # Force death inside the synchronized accessor if this production primitive
    # has a mutex. Raw production status has no lock to leave behind.
    with primitive.get_lock() if hasattr(primitive, "get_lock") else nullcontext():
        shared.set_heartbeat("recorder", time.monotonic())
        entered.set()
        release.wait(15)


def probe_critical_status(shared, done):
    shared.set_heartbeat("arm", 11.)
    shared.set_heartbeat("hand", 12.)
    assert shared.get_heartbeat("arm") == 11.
    assert shared.get_heartbeat("hand") == 12.
    shared.get_heartbeat("recorder")
    shared.is_ready("arm")
    shared.is_running.value
    shared.evidence_failed.value
    shared.camera_depth_scale.value
    shared.camera_serial.value
    shared.camera_geometry.value
    shared.recorder_consumed_sequence.value
    done.set()


@pytest.mark.parametrize("field", [
    "heartbeats", "ready_flags", "is_running", "evidence_failed",
    "camera_depth_scale", "camera_serial", "camera_geometry", "recorder_consumed_sequence",
])
def test_killed_recorder_cannot_poison_critical_status(field):
    ctx = mp.get_context("spawn")
    shared = Shared(ctx)
    entered, release, done = ctx.Event(), ctx.Event(), ctx.Event()
    recorder = ctx.Process(target=hold_recorder_access, args=(shared, field, entered, release))
    probe = ctx.Process(target=probe_critical_status, args=(shared, done))
    try:
        recorder.start()
        assert entered.wait(10)
        recorder.terminate()
        recorder.join(5)
        assert not recorder.is_alive()
        probe.start()
        assert done.wait(5), f"critical status blocked after recorder died in {field}"
        probe.join(5)
        assert probe.exitcode == 0
    finally:
        for process in (recorder, probe):
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(5)
        assert shared.close()


@pytest.mark.parametrize("join", [False, True])
def test_completed_result_survives_first_observation_after_deadline(tmp_path, join):
    ctx = mp.get_context("spawn")
    shared = Shared(ctx)
    entered, release, completed, exit_allowed = (ctx.Event() for _ in range(4))
    process = ctx.Process(name="recorder", target=close_in_child,
        args=(shared, str(tmp_path), entered, release, completed, exit_allowed))
    client = RecorderClient(shared)
    client._stop_requested = True
    try:
        process.start()
        assert entered.wait(10)
        deadline = shared.recorder_finish_deadline_ns.value
        client._finish_deadline_ns = deadline
        assert not client.start_episode()
        release.set()
        assert completed.wait(10)
        assert 0 < shared.recorder_completed_ns.value < deadline
        assert shared.record_result_q._reader.poll(5)  # readiness only; client is sole consumer
        with patch("dexmani_real.recording.client.time.monotonic_ns", return_value=deadline + 1):
            result = client.join_stop() if join else client.poll_stop()
            assert result.done and result.saved and result.frame_count == 2
            assert not shared.recorder_transport_failed.value
            assert not client.poll_stop().done
            assert not client.join_stop().saved
        assert not client.stop_pending
        assert not client.transport_unavailable
    finally:
        if process.is_alive():
            exit_allowed.set()
            process.join(5)
        if process.is_alive():
            process.terminate()
            process.join(5)
        assert shared.close()


def test_unfinished_deadline_disables_ipc_and_next_start():
    ctx = mp.get_context("spawn")
    shared = Shared(ctx)
    client = RecorderClient(shared)
    client._stop_requested = True
    client._finish_deadline_ns = 100
    try:
        with patch("dexmani_real.recording.client.time.monotonic_ns", return_value=100), \
             patch.object(shared.record_result_q, "get_nowait", side_effect=AssertionError("dead IPC read")):
            assert client.poll_stop().error
            assert shared.recorder_transport_failed.value
            assert not client.start_episode()
            assert client.join_stop().error
            # Even a subsequently visible completion cannot resurrect failed IPC.
            shared.recorder_completed_ns.value = 99
            assert client.poll_stop().error
    finally:
        assert shared.close()


def serve_two_episodes(shared, directory, ready, exit_allowed):
    from dexmani_real.recording.client import StartRecording, StopRecording
    recorder = _make_recorder(Path(directory), min_frames=1)
    session = _RecorderIOSession(shared, RecorderIOConfig(
        data_dir=directory, max_frames=10, control_hz=16, min_frames=1), recorder)
    shared.set_ready("recorder")
    ready.set()
    for index in range(2):
        start = shared.record_control_q.get(timeout=10)
        assert isinstance(start, StartRecording)
        with patch("dexmani_real.recording.io_worker._build_start_metadata",
                   return_value={"task_label": "test", "episode_name": f"episode_{index}"}):
            session._handle_start(start)
        _add_frames(recorder, 2)
        stop = shared.record_control_q.get(timeout=10)
        assert isinstance(stop, StopRecording)
        session._handle_stop(stop)
        session._drain_samples()
    assert exit_allowed.wait(15)


def test_late_result_then_next_start_and_shutdown_have_one_result_owner(tmp_path):
    from types import SimpleNamespace
    from dexmani_real.deployment.session import _wait_for_rollout_recording
    ctx = mp.get_context("spawn")
    shared = Shared(ctx)
    ready, exit_allowed = ctx.Event(), ctx.Event()
    process = ctx.Process(name="recorder", target=serve_two_episodes,
                         args=(shared, str(tmp_path), ready, exit_allowed))
    client = RecorderClient(shared)
    results = []
    try:
        process.start()
        assert ready.wait(10)
        for index in range(2):
            assert client.start_episode()
            assert shared.recorder_completed_ns.value == 0
            client.stop_episode()
            deadline = client._finish_deadline_ns
            assert not client.start_episode()
            assert shared.record_result_q._reader.poll(10)
            # The producer's completion fact may follow feeder availability.
            until = time.monotonic() + 5
            while not shared.recorder_completed_ns.value and time.monotonic() < until:
                time.sleep(.001)
            assert 0 < shared.recorder_completed_ns.value < deadline
            with patch("dexmani_real.recording.client.time.monotonic_ns", return_value=deadline + 1):
                if index == 0:
                    results.append(client.poll_stop())
                else:
                    shared.is_recording.value = True
                    # Exercise the completed-before-deadline-clear interleaving.
                    shared.recorder_finish_deadline_ns.value = deadline
                    def consume(_):
                        results.append(client.join_stop())
                        shared.is_recording.value = False
                    policy = SimpleNamespace(name="policy", exitcode=None, is_alive=lambda: True)
                    with patch("dexmani_real.deployment.session.time.sleep", side_effect=consume):
                        assert _wait_for_rollout_recording(shared, [process, policy],
                            heartbeat_timeouts_s={"recorder": 1}, service_process_names={"recorder"})
                assert not client.poll_stop().done
                assert not client.join_stop().saved
                assert not shared.recorder_transport_failed.value
        assert len(results) == 2 and all(result.saved for result in results)
        assert len({result.path for result in results}) == 2
    finally:
        if process.is_alive():
            exit_allowed.set()
            process.join(5)
        if process.is_alive():
            process.terminate()
            process.join(5)
        assert shared.close()


def test_dead_recorder_result_queue_is_never_read():
    ctx = mp.get_context("spawn")
    shared = Shared(ctx)
    entered, release = ctx.Event(), ctx.Event()
    process = ctx.Process(name="recorder", target=hold_recorder_access,
                         args=(shared, "heartbeats", entered, release))
    try:
        process.start()
        assert entered.wait(10)
        process.terminate()
        process.join(5)
        assert supervisor_exit_reason(shared, [process], {"recorder": 0}, {"recorder": 1},
            service_process_names={"recorder"}) is ExitReason.EVIDENCE_FAILURE
        with patch.object(shared.record_result_q, "get_nowait", side_effect=AssertionError("dead IPC")):
            assert RecorderClient(shared).poll_stop().error
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
        assert shared.close()
