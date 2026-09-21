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


class Shared:
    def __init__(self, ctx):
        for name in ('recorder_finish_deadline_ns', 'recorder_transport_failed',
                     'estop_request', 'error_state', 'quit_requested', 'session_failed'):
            setattr(self, name, ctx.Value('q', 0, lock=False))
        self.is_running = ctx.Value('b', True, lock=False)
        self.record_result_q = ctx.Queue()
        self.heartbeat = ctx.Value('d', 0, lock=False)

    def set_heartbeat(self, name, timestamp):
        self.heartbeat.value = timestamp


def close_in_child(shared, directory, entered, release):
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
        shared.record_result_q.cancel_join_thread()
        shared.record_result_q.close()


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
