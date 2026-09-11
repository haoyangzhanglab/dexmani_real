"""Recorder Queue boundaries and exact sample FIFO behavior, without hardware."""

from queue import Empty, Queue
from threading import Event
from types import SimpleNamespace
from unittest import mock
from uuid import uuid4

import numpy as np
import pytest

from dexmani_real.ipc.ring import SharedMemoryRingBuffer
from dexmani_real.ipc.schema import make_record_sample_dtype
from dexmani_real.recording.client import (
    RecordingFinished,
    RecordingStarted,
    StartRecording,
    StopRecording,
    RecorderClient,
)
from dexmani_real.recording.io_worker import (
    RecorderIOConfig,
    _RecorderIOSession,
    _build_start_metadata,
    _create_episode_recorder,
    recorder_io_loop,
)
from dexmani_real.recording.sample import EpisodeAction, build_episode_state
from dexmani_real.recording.storage.reader import EpisodeReader


class SampleRing:
    maxlen = 8

    def __init__(self):
        self.latest_sequence = 0
        self.rows = {}
        self.reads = []

    def write(self):
        self.latest_sequence += 1
        sample = np.zeros(1, dtype=make_record_sample_dtype((2, 2, 3), (2, 2)))
        sample["timestamp"] = self.latest_sequence
        self.rows[self.latest_sequence] = sample

    def read_sequence(self, sequence):
        self.reads.append(sequence)
        if sequence not in self.rows:
            return None
        return self.rows[sequence].copy(), 0, sequence


class Recorder:
    def __init__(self, max_frames=8):
        self.max_frames = max_frames
        self.is_recording = False
        self.frame_count = 0
        self.max_frames_reached = False
        self.camera_writer_error = None
        self.stop_error = None
        self.episode_path = "/offline/episode"
        self.rows = []
        self.stops = []
        self._finish_entered = Event()
        self._finish_release = Event()
        self._finish_release.set()
        self.resources_released = True

    def start_episode(self, **metadata):
        self.is_recording = True
        self.frame_count = 0
        self.max_frames_reached = False
        self.rows = []
        return True

    def add_episode_frame(self, frame):
        if self.frame_count >= self.max_frames:
            return False
        self.rows.append(frame)
        self.frame_count += 1
        self.max_frames_reached = self.frame_count == self.max_frames
        return not self.max_frames_reached

    @property
    def finished(self):
        return self._finish_release.is_set()

    @finished.setter
    def finished(self, value):
        if value:
            self._finish_release.set()
        else:
            self._finish_release.clear()

    def finish_episode(self, save=True, reason=""):
        self.stops.append((save, reason, self.frame_count))
        self._finish_entered.set()
        self._finish_release.wait()
        self.is_recording = False
        if self.stop_error:
            raise RuntimeError(self.stop_error)
        return self.episode_path


@pytest.fixture
def session():
    shared = SimpleNamespace(
        record_control_q=Queue(),
        record_result_q=Queue(),
        record_sample_ring=SampleRing(),
        recorder_consumed_sequence=SimpleNamespace(value=0),
        is_running=SimpleNamespace(value=True),
        error_state=SimpleNamespace(value=False),
        set_heartbeat=mock.Mock(),
        set_ready=mock.Mock(),
    )
    config = SimpleNamespace(
        max_frames=8,
        min_frames=1,
        camera_calibration=None,
        provenance={},
        max_frames_stop_reason="max_frames",
        poll_hz=128,
    )
    worker = _RecorderIOSession.create(shared, config, Recorder())
    with mock.patch(
        "dexmani_real.recording.io_worker._build_start_metadata", return_value={}
    ):
        try:
            yield worker
        finally:
            worker.recorder.finished = True
            if worker.pending_finalization is not None:
                thread = worker.pending_finalization.thread
                if thread.ident is not None:
                    thread.join(timeout=2)
                    assert not thread.is_alive()


def _step(session):
    session.step()
    pending = session.pending_finalization
    if (
        pending is not None
        and isinstance(session.recorder, Recorder)
        and session.recorder.finished
    ):
        pending.thread.join(timeout=2)
        session._poll_finalization()


def _start(session):
    session.shared.record_control_q.put(
        StartRecording(
            "task", "operator", session.shared.record_sample_ring.latest_sequence + 1
        )
    )
    _step(session)
    result = session.shared.record_result_q.get_nowait()
    assert isinstance(result, RecordingStarted)
    return result


def _assert_clean_worker_exit(session):
    session.shared.is_running.value = False
    with (
        mock.patch(
            "dexmani_real.recording.io_worker._create_episode_recorder",
            return_value=session.recorder,
        ),
        mock.patch(
            "dexmani_real.recording.io_worker._RecorderIOSession.create",
            return_value=session,
        ),
    ):
        recorder_io_loop(session.shared, session.config)
    assert not session.shared.error_state.value


def test_start_ack_precedes_samples(session):
    result = _start(session)
    assert result.max_frames == 8
    assert result.max_frames_stop_reason == "max_frames"
    assert session.recorder.is_recording
    assert not session.shared.record_sample_ring.reads
    assert session.shared.recorder_consumed_sequence.value == 0


def test_partial_start_failure_is_discarded_before_finished(session):
    session.recorder.finished = False

    def fail_after_open(**metadata):
        session.recorder.is_recording = True
        raise OSError("camera constructor failed")

    with mock.patch.object(
        session.recorder, "start_episode", side_effect=fail_after_open
    ):
        session.shared.record_control_q.put(StartRecording("task", "operator", 1))
        _step(session)
    assert session.recorder.stops == [(False, "start_error", 0)]
    with pytest.raises(Empty):
        session.shared.record_result_q.get_nowait()
    session.recorder.finished = True
    _step(session)
    result = session.shared.record_result_q.get_nowait()
    assert result.error == "camera constructor failed" and not result.saved


def test_start_failure_with_retained_resource_cannot_report_recovery(session):
    session.recorder.resources_released = False
    with mock.patch.object(
        session.recorder, "start_episode", side_effect=OSError("partial allocation")
    ):
        session.shared.record_control_q.put(StartRecording("task", "operator", 1))
        with pytest.raises(RuntimeError, match="start retained unreleased resources"):
            session.step()
    assert not session.shutdown()
    assert session.shared.record_result_q.empty()


def test_stop_drains_exact_boundary_before_finish_and_next_start(session):
    _start(session)
    ring = session.shared.record_sample_ring
    ring.write()
    ring.write()
    session.shared.record_control_q.put(StopRecording(True, "manual", 2))
    ring.write()  # An unexpected tail must not be included in this STOP.
    _step(session)
    assert ring.reads == [1, 2]
    assert session.recorder.stops == [(True, "manual", 2)]
    result = session.shared.record_result_q.get_nowait()
    assert isinstance(result, RecordingFinished) and result.saved
    assert result.frame_count == 2
    assert session.shared.recorder_consumed_sequence.value == 2
    _start(session)
    assert session.shared.recorder_consumed_sequence.value == 3
    ring.write()
    session.shared.record_control_q.put(StopRecording(False, "discard", 4))
    _step(session)
    result = session.shared.record_result_q.get_nowait()
    assert not result.saved and not result.error and result.frame_count == 1
    assert [row.timestamp_s for row in session.recorder.rows] == [4]


@pytest.mark.parametrize("damage", ["missing", "overflow", "decode", "write"])
def test_sample_failure_discards_then_releases_capacity_once(session, damage):
    _start(session)
    ring = session.shared.record_sample_ring
    ring.write()
    if damage == "missing":
        del ring.rows[1]
    if damage == "overflow":
        for _ in range(ring.maxlen):
            ring.write()
    decode = mock.patch(
        "dexmani_real.recording.io_worker.decode_record_sample",
        side_effect=ValueError("decode failure"),
    )
    write = mock.patch.object(
        session.recorder, "add_episode_frame", side_effect=OSError("write failure")
    )
    with (
        decode
        if damage == "decode"
        else (
            write
            if damage == "write"
            else mock.patch.object(session.shared, "set_ready")
        )
    ):
        _step(session)
    result = session.shared.record_result_q.get_nowait()
    assert result.error and not result.saved
    assert session.recorder.stops[0][0] is False
    assert session.shared.recorder_consumed_sequence.value == ring.latest_sequence
    assert not session.fatal
    session.shared.record_control_q.put(
        StopRecording(False, "late stop", ring.latest_sequence)
    )
    _step(session)
    with pytest.raises(Empty):
        session.shared.record_result_q.get_nowait()

    _start(session)
    ring.write()
    session.shared.record_control_q.put(
        StopRecording(True, "recovered", ring.latest_sequence)
    )
    _step(session)
    recovered = session.shared.record_result_q.get_nowait()
    assert recovered.saved and not recovered.error and recovered.frame_count == 1
    assert [row.timestamp_s for row in session.recorder.rows] == [ring.latest_sequence]
    _assert_clean_worker_exit(session)


def test_frame_decoded_before_shared_consumption_ack(session):
    _start(session)
    session.shared.record_sample_ring.write()
    from dexmani_real.recording.frame import decode_record_sample

    def decode(record):
        assert session.shared.recorder_consumed_sequence.value == 0
        return decode_record_sample(record)

    with mock.patch(
        "dexmani_real.recording.io_worker.decode_record_sample", side_effect=decode
    ):
        _step(session)
    assert session.shared.recorder_consumed_sequence.value == 1


def test_frame_arrays_are_owned_before_ack_releases_producer(session):
    _start(session)
    ring = session.shared.record_sample_ring
    ring.write()
    source = ring.rows[1]
    source["camera_present"] = True
    source["arm_qpos"] = 3
    source["camera_rgb"] = 17
    source["camera_depth"] = 23

    class OverwriteOnAck:
        value_before_ack = 0

        @property
        def value(self):
            return self.value_before_ack

        @value.setter
        def value(self, sequence):
            self.value_before_ack = sequence
            source["arm_qpos"] = 99
            source["camera_rgb"] = 99
            source["camera_depth"] = 99

    session.shared.recorder_consumed_sequence = OverwriteOnAck()
    # Return a shared view so the ring fake cannot mask a missing decode copy.
    with mock.patch.object(ring, "read_sequence", return_value=(source, 0, 1)):
        _step(session)
    frame = session.recorder.rows[0]
    np.testing.assert_array_equal(frame.data["arm_qpos"], np.full(7, 3))
    np.testing.assert_array_equal(frame.camera_rgb, np.full((2, 2, 3), 17))
    np.testing.assert_array_equal(frame.camera_depth, np.full((2, 2), 23))
    assert session.shared.recorder_consumed_sequence.value == 1


def test_pending_finalization_keeps_heartbeat_and_owns_shutdown(session):
    _start(session)
    ring = session.shared.record_sample_ring
    ring.write()
    session.recorder.finished = False
    session.shared.record_control_q.put(StopRecording(True, "manual", 1))
    _step(session)
    assert session.recorder._finish_entered.wait(timeout=2)
    heartbeat_count = session.shared.set_heartbeat.call_count
    ring.write()
    session.shared.record_control_q.put(StopRecording(False, "duplicate", 2))
    session.shared.is_running.value = False
    for _ in range(3):
        _step(session)
    assert session.shared.set_heartbeat.call_count == heartbeat_count + 3
    assert session.shared.record_control_q.empty()
    assert session.should_run
    assert ring.reads == [1]
    assert session.recorder.stops == [(True, "manual", 1)]
    with pytest.raises(Empty):
        session.shared.record_result_q.get_nowait()

    session.shared.record_control_q.put(StartRecording("task", "operator", 3))
    with pytest.raises(RuntimeError, match="previous recording is active"):
        _step(session)
    session.recorder.finished = True
    _step(session)
    result = session.shared.record_result_q.get_nowait()
    assert result.saved and result.frame_count == 1 and result.reason == "manual"
    assert not session.should_run
    _step(session)
    with pytest.raises(Empty):
        session.shared.record_result_q.get_nowait()


def test_queued_finalizer_result_waits_for_thread_exit_and_explicit_join(session):
    _start(session)
    queued = Event()
    release = Event()
    finish = session._finish_episode

    def finish_then_block(pending):
        finish(pending)
        queued.set()
        release.wait()

    try:
        with mock.patch.object(session, "_finish_episode", side_effect=finish_then_block):
            session.shared.record_control_q.put(StopRecording(True, "manual", 0))
            session.step()
            assert queued.wait(timeout=2)
        pending = session.pending_finalization
        assert pending is not None and not pending.results.empty()
        session.shared.is_running.value = False
        # A queued outcome does not permit reading any serializer state yet.
        with (
            mock.patch.object(
                Recorder, "is_recording", new_callable=mock.PropertyMock,
                side_effect=AssertionError("concurrent recorder read"), create=True,
            ),
            mock.patch.object(
                Recorder, "camera_writer_error", new_callable=mock.PropertyMock,
                side_effect=AssertionError("concurrent camera read"), create=True,
            ),
        ):
            assert session.should_run
            before = session.shared.set_heartbeat.call_count
            session.step()
            assert session.shared.set_heartbeat.call_count == before + 1
        assert session.shared.record_result_q.empty()
        session.shared.record_control_q.put(StartRecording("task", "operator", 1))
        with pytest.raises(RuntimeError, match="previous recording is active"):
            session.step()
        release.set()
        pending.thread.join(timeout=2)
        with mock.patch.object(pending.thread, "join", wraps=pending.thread.join) as join:
            send = session._send_result

            def require_reap(result):
                join.assert_called_once_with(timeout=0)
                send(result)

            with mock.patch.object(session, "_send_result", side_effect=require_reap):
                session.step()
        assert session.shared.record_result_q.get_nowait().saved
        assert session.pending_finalization is None
        session.shared.is_running.value = True
        _start(session)
    finally:
        release.set()


@pytest.mark.parametrize("failure", ["missing_result", "unexpected", "unreleased"])
def test_finalizer_contract_failure_is_fatal_without_finished(session, failure):
    _start(session)
    if failure == "missing_result":
        patch = mock.patch.object(session, "_finish_episode", return_value=None)
    elif failure == "unexpected":
        patch = mock.patch.object(
            session.recorder, "finish_episode", side_effect=ValueError("programming error")
        )
    else:
        session.recorder.resources_released = False
        patch = mock.patch.object(session.recorder, "finish_episode", return_value=None)
    session.shared.record_control_q.put(StopRecording(True, "manual", 0))
    with (
        patch,
        mock.patch(
            "dexmani_real.recording.io_worker._create_episode_recorder",
            return_value=session.recorder,
        ),
        mock.patch(
            "dexmani_real.recording.io_worker._RecorderIOSession.create",
            return_value=session,
        ),
    ):
        with pytest.raises(RuntimeError, match="recording failure"):
            recorder_io_loop(session.shared, session.config)
    assert session.shared.error_state.value
    assert session.pending_finalization is not None
    assert session.shared.record_result_q.empty()


def test_finalizer_start_failure_retains_pending_owner(session):
    _start(session)
    session.shared.record_control_q.put(StopRecording(True, "manual", 0))
    with mock.patch(
        "dexmani_real.recording.io_worker.threading.Thread.start",
        side_effect=RuntimeError("cannot launch finalizer"),
    ):
        with pytest.raises(RuntimeError, match="cannot launch"):
            session.step()
    pending = session.pending_finalization
    assert pending is not None
    assert session.recorder.is_recording
    assert not session.shutdown()
    assert session.pending_finalization is pending
    assert session.shared.record_result_q.empty()


def test_last_capacity_row_is_accepted_before_producer_stop(session):
    _start(session)
    ring = session.shared.record_sample_ring
    for _ in range(8):
        ring.write()
    _step(session)
    assert session.recorder.frame_count == 8 and session.recorder.is_recording
    assert not session.fatal
    session.shared.record_control_q.put(StopRecording(True, "max_frames", 8))
    _step(session)
    assert session.shared.record_result_q.get_nowait().saved


def test_idle_async_camera_error_is_finalized(session):
    _start(session)
    session.recorder.camera_writer_error = "encoder failed"
    _step(session)
    result = session.shared.record_result_q.get_nowait()
    assert result.error == "encoder failed" and not result.saved
    session.recorder.camera_writer_error = None
    _start(session)
    session.shared.record_sample_ring.write()
    session.shared.record_control_q.put(StopRecording(True, "recovered", 1))
    _step(session)
    result = session.shared.record_result_q.get_nowait()
    assert result.saved and not result.error and result.frame_count == 1
    _assert_clean_worker_exit(session)


def test_finalization_timeout_faults_session_without_premature_finished(session):
    _start(session)
    session.recorder.finished = False
    session.shared.record_control_q.put(StopRecording(True, "manual", 0))
    _step(session)
    pending = session.pending_finalization
    assert pending is not None
    pending.started_monotonic_s -= 61
    with pytest.raises(RuntimeError, match="finalization timed out"):
        _step(session)
    assert session.shared.error_state.value and session.fatal
    assert not session.shutdown()
    assert session.pending_finalization is pending
    session.shared.record_control_q.put(StartRecording("task", "operator", 1))
    with pytest.raises(RuntimeError, match="previous recording is active"):
        _step(session)
    session.recorder.finished = True
    pending.thread.join(timeout=2)
    with pytest.raises(RuntimeError, match="finalization timed out"):
        _step(session)
    with pytest.raises(Empty):
        session.shared.record_result_q.get_nowait()


def test_runtime_shutdown_discards_active_episode_and_ends(session):
    _start(session)
    session.shared.is_running.value = False
    _step(session)
    result = session.shared.record_result_q.get_nowait()
    assert not result.saved and result.reason == "runtime_shutdown"
    assert not session.should_run


def test_crashed_worker_signals_supervisor_and_reaps_recorder(session):
    with (
        mock.patch(
            "dexmani_real.recording.io_worker._create_episode_recorder",
            return_value=session.recorder,
        ),
        mock.patch(
            "dexmani_real.recording.io_worker._RecorderIOSession.create",
            return_value=session,
        ),
        mock.patch.object(session, "step", side_effect=OSError("worker failure")),
        mock.patch.object(
            session, "shutdown", wraps=session.shutdown
        ) as join,
    ):
        with pytest.raises(RuntimeError, match="recording failure"):
            recorder_io_loop(session.shared, session.config)
    assert session.shared.error_state.value
    join.assert_called_once()


def test_prior_recording_failure_allows_clean_worker_exit(session):
    _start(session)
    session.shared.record_sample_ring.write()
    session.shared.record_sample_ring.rows.clear()
    _step(session)
    assert session.shared.record_result_q.get_nowait().error
    _assert_clean_worker_exit(session)


def test_result_transport_failure_faults_worker_and_reaps_recorder(session):
    session.shared.record_control_q.put(StartRecording("task", "operator", 1))
    with (
        mock.patch(
            "dexmani_real.recording.io_worker._create_episode_recorder",
            return_value=session.recorder,
        ),
        mock.patch(
            "dexmani_real.recording.io_worker._RecorderIOSession.create",
            return_value=session,
        ),
        mock.patch.object(
            session.shared.record_result_q, "put", side_effect=OSError("broken queue")
        ),
        mock.patch.object(
            session, "shutdown", wraps=session.shutdown
        ) as join,
    ):
        with pytest.raises(RuntimeError, match="recording failure"):
            recorder_io_loop(session.shared, session.config)
    assert session.shared.error_state.value
    assert session.recorder.stops == [(False, "recorder_process_shutdown", 0)]
    join.assert_called_once()
    assert session.shared.record_result_q.empty()


def test_terminal_transport_failure_reaps_without_refinalizing_or_resending(session):
    _start(session)
    session.recorder.finished = False
    session.shared.record_control_q.put(StopRecording(True, "manual", 0))
    session.step()
    pending = session.pending_finalization
    assert session.recorder._finish_entered.wait(timeout=2)
    session.recorder.finished = True
    pending.thread.join(timeout=2)
    with (
        mock.patch.object(pending.thread, "join", wraps=pending.thread.join) as join,
        mock.patch.object(
            session.shared.record_result_q, "put", side_effect=OSError("broken result queue")
        ) as put,
        mock.patch(
            "dexmani_real.recording.io_worker._create_episode_recorder",
            return_value=session.recorder,
        ),
        mock.patch(
            "dexmani_real.recording.io_worker._RecorderIOSession.create",
            return_value=session,
        ),
    ):
        with pytest.raises(RuntimeError, match="recording failure"):
            recorder_io_loop(session.shared, session.config)
        assert join.call_args_list[0] == mock.call(timeout=0)
        assert join.call_count == 2  # Main poll, then the same shutdown owner.
        put.assert_called_once()
        assert isinstance(put.call_args.args[0], RecordingFinished)
    assert session.fatal and session.shared.error_state.value
    assert session.pending_finalization is pending
    assert not pending.thread.is_alive()
    assert session.recorder.stops == [(True, "manual", 0)]
    assert session.shared.record_result_q.empty()


def test_unreleased_resources_exit_nonzero_without_prior_failure(session):
    session.shared.is_running.value = False
    session.recorder.resources_released = False
    assert not session.fatal
    with (
        mock.patch(
            "dexmani_real.recording.io_worker._create_episode_recorder",
            return_value=session.recorder,
        ),
        mock.patch(
            "dexmani_real.recording.io_worker._RecorderIOSession.create",
            return_value=session,
        ),
    ):
        with pytest.raises(RuntimeError, match="recording failure"):
            recorder_io_loop(session.shared, session.config)
    assert session.shared.error_state.value
    assert session.shared.record_result_q.empty()



def test_client_shm_recorder_disk_roundtrip_at_capacity(tmp_path):
    ring = SharedMemoryRingBuffer(
        "recorder_fifo_test_" + uuid4().hex,
        dtype=make_record_sample_dtype((16, 16, 3), (16, 16)),
        maxlen=8,
    )
    shared = SimpleNamespace(
        record_control_q=Queue(),
        record_result_q=Queue(),
        record_sample_ring=ring,
        recorder_consumed_sequence=SimpleNamespace(value=0),
        is_running=SimpleNamespace(value=True),
        error_state=SimpleNamespace(value=False),
        set_heartbeat=mock.Mock(),
        is_ready=lambda name: True,
    )
    config = RecorderIOConfig(str(tmp_path), max_frames=2, control_hz=16, min_frames=1)
    recorder = _create_episode_recorder(shared, config)
    session = _RecorderIOSession.create(shared, config, recorder)
    client = RecorderClient(shared)
    put = shared.record_control_q.put
    poll = client.poll_stop

    def deliver_control(message, **kwargs):
        put(message, **kwargs)
        _step(session)

    def poll_result():
        _step(session)
        return poll()

    try:
        with (
            mock.patch(
                "dexmani_real.recording.io_worker._build_start_metadata",
                return_value={},
            ),
            mock.patch.object(
                shared.record_control_q, "put", side_effect=deliver_control
            ),
            mock.patch.object(client, "poll_stop", side_effect=poll_result),
        ):
            assert client.start_episode(task_label="offline")
            for index in range(2):
                assert client.add_frame(
                    build_episode_state(None, None, timestamp_s=1 + index / 16),
                    EpisodeAction(np.zeros(7), np.zeros(12)),
                    {
                        "wrist_pos": np.zeros(3),
                        "wrist_quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0]),
                        "landmarks": np.zeros((21, 3)),
                    },
                    camera_frame={
                        "rgb": np.full((16, 16, 3), index, np.uint8),
                        "depth": np.full((16, 16), index, np.uint16),
                    },
                    arm_qpos_sent=np.full(7, index, dtype=np.float64),
                )
            assert not client.is_recording
            result = client.join_stop(timeout=5)
            assert result.done and result.saved and not result.error
            assert result.reason == "max_frames"
            assert shared.recorder_consumed_sequence.value == ring.latest_sequence == 2
            with EpisodeReader(result.path) as reader:
                np.testing.assert_array_equal(reader.h5f["timestamp"][:], [1, 1.0625])
                np.testing.assert_array_equal(
                    reader.h5f["action_arm_joint_sent"][:, 0], [0, 1]
                )
                np.testing.assert_array_equal(reader.h5f["depth"][:, 0, 0], [0, 1])
    finally:
        assert session.shutdown()
        ring.close()
        ring.unlink()

def test_build_start_metadata_forwards_episode_name():
    shared = SimpleNamespace(
        camera_depth_scale=SimpleNamespace(value=0.0),
        camera_serial=SimpleNamespace(value=b""),
        camera_geometry=SimpleNamespace(value=b""),
    )
    with mock.patch(
        "dexmani_real.recording.io_worker._camera_geometry_from_shared",
        return_value=None,
    ):
        explicit = _build_start_metadata(
            shared,
            task_label="task",
            operator="op",
            episode_name="episode_003",
            calibration=None,
            provenance={},
        )
        default = _build_start_metadata(
            shared,
            task_label="task",
            operator="op",
            episode_name=None,
            calibration=None,
            provenance={},
        )
    assert explicit["episode_name"] == "episode_003"
    assert default["episode_name"] is None


def test_handle_start_forwards_episode_name(session):
    captured = {}

    def fake_metadata(shared, **kwargs):
        captured.update(kwargs)
        return {}

    with mock.patch(
        "dexmani_real.recording.io_worker._build_start_metadata",
        side_effect=fake_metadata,
    ):
        session.shared.record_control_q.put(
            StartRecording("task", "operator", 1, "episode_007")
        )
        _step(session)
    result = session.shared.record_result_q.get_nowait()
    assert isinstance(result, RecordingStarted)
    assert captured["episode_name"] == "episode_007"
