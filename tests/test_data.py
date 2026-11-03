"""调用真实 Raw/Canonical 与首帧选择链路；设备和感知模型使用替身。"""

import json
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import h5py
import numpy as np
import pytest
import zarr

from dexmani_real.config.perception import PointCloudConfig
from dexmani_real.config.real import RealConfig
from dexmani_real.data import recorder as recorder_module
from dexmani_real.data.export import ExportConfig, export_zarr
from dexmani_real.data.raw import RawReader, RawWriter, snapshot_recording_metadata
from dexmani_real.data.recorder import EpisodeRecorder, RecordingError
from dexmani_real.deployment.inference import MODEL_SHUTDOWN_TIMEOUT_S, PolicyWorker
from dexmani_real.deployment.rollout import Rollout
from dexmani_real.deployment.schedule import Prediction
from dexmani_real.geometry.camera import CameraIntrinsics, CameraSnapshot
from dexmani_real.robot.commands import RobotCommand
from dexmani_real.robot.xarm7.protocol import ARM_STATE_DTYPE
from dexmani_real.robot.xhand.feedback import HAND_STATE_DTYPE
from dexmani_real.runtime.observation import Observation
from dexmani_real.runtime.state import RunEndReason, SessionState


@pytest.fixture
def capture():
    camera = CameraSnapshot(
        "synthetic", CameraIntrinsics(32, 32, 200., 200., 15.5, 15.5, "none", (0.,) * 5),
        .001, np.eye(4),
    )
    metadata = snapshot_recording_metadata(
        camera, tactile_reset_metadata={}, tactile_hold_max_age_s=.5,
    )

    def row(index):
        stamp = 1_000_000_000 + index * 100_000_000
        arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
        hand = np.zeros(1, dtype=HAND_STATE_DTYPE)
        arm["timestamp_ns"] = stamp
        for name in ("timestamp_ns", "tactile_aggregate_timestamp_ns", "tactile_dense_timestamp_ns"):
            hand[name] = stamp
        hand["tactile_aggregate"] = np.arange(15).reshape(5, 3)
        hand["tactile_dense"] = np.arange(1800).reshape(5, 120, 3)
        return Observation(
            arm=arm, hand=hand,
            camera={"rgb": np.full((32, 32, 3), 80, np.uint8),
                    "depth": np.full((32, 32), 500, np.uint16),
                    "timestamp_ns": stamp, "ring_sequence": index // 2},
            vr={"wrist_pos": np.zeros(3), "wrist_quat_wxyz": np.array([1., 0., 0., 0.]),
                "landmarks": np.zeros((21, 3)), "recv_ts_ns": stamp, "sequence_id": index},
            point_cloud=None, observation_timestamp_ns=stamp + 1_000_000,
            control_timestamp_ns=stamp,
        )

    return metadata, row, RobotCommand(1, np.zeros(7), np.zeros(12))


def record(path, capture, *, frames=3, save=True, status="normal"):
    metadata, row, command = capture
    recorder = EpisodeRecorder(path, control_hz=10)
    recorder.start(task_label="fixture", collection_source="teleop", metadata=metadata)
    for index in range(frames):
        recorder.append(row(index), command)
    result = recorder.finish(save=save, reason="test", technical_status=status)
    return recorder, result


@pytest.mark.parametrize("frames,save,status", [(3, True, "normal"), (3, True, "failed"),
                                                   (0, True, "normal"), (3, False, "normal")])
def test_raw_publication_and_video(tmp_path, capture, frames, save, status):
    recorder, result = record(tmp_path, capture, frames=frames, save=save, status=status)
    assert not recorder.writer_pending
    assert recorder.written_frames == frames
    if frames and save:
        assert result == recorder.published_path
        with RawReader(result) as reader:
            assert reader.num_frames == frames
            assert reader.meta["technical_status"] == status
            assert not ({"hand_tactile_processing", "hand_tactile_axes", "vr_hand_side",
                         "vr_coordinate_axes", "vr_position_units", "vr_landmark_order",
                         "vr_timestamp_source"} & set(reader.meta))
            assert not ({"clock", "missing_value", "camera_timestamp_source"}
                        & set(reader.data_file["timing"].attrs))
            assert reader.meta["timestamp_source"] == "host_monotonic_ns"
            np.testing.assert_array_equal(reader["depth"][0], 500)
            assert len(list(reader.iter_rgb_frames())) == frames
    else:
        assert result is None
        assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("case,reason,row_index", [
    ("normal", None, None),
    ("jitter", None, None),
    ("gap", "control_interval_gap_or_drift", 1),
    ("nan", "nonfinite:action_arm_joint_target", 2),
    ("inf", "nonfinite:hand_tactile_force", 2),
    ("failed", "technical failure", None),
    ("tactile", "stale_source:tactile_dense_ns", 2),
    ("readout_between", None, None),
    ("readout_limit", "observation_crossed_control_period", 2),
    ("overlap", "overlapping_control_observations", 0),
    ("source_future", "invalid_source_clock:arm_ns", 2),
    ("source_missing", "invalid_source_clock:hand_ns", 2),
    ("source_regressed", "source_clock_regressed:camera_ns", 2),
    ("sequence_regressed", "camera_sequence_regressed", 2),
])
def test_training_admits_only_complete_episode(tmp_path, capture, case, reason, row_index):
    _, path = record(tmp_path / "raw", capture, status="failed" if case == "failed" else "normal")
    with h5py.File(path / "data.h5", "r+") as data:
        if case == "nan":
            data["action_arm_joint_target"][2] = np.full(7, np.nan)
        elif case == "inf":
            data["hand_tactile_force"][2] = np.full((5, 120, 3), np.inf)
        else:
            rows = data["timing/rows"][:]
            if case == "gap":
                rows["control_ns"][1:] += 100_000_000
                rows["observation_ns"][1:] += 100_000_000
            elif case == "jitter":
                # dt/4 边界仍接纳，来源值不随导出重采样。
                rows["control_ns"][1] += 25_000_000
                rows["observation_ns"][1] += 25_000_000
            elif case == "overlap":
                rows["observation_ns"][0] = rows["control_ns"][0] + 80_000_000
                rows["control_ns"][1] -= 25_000_000
            elif case.startswith("readout"):
                readout_ns = 30_000_000 if case == "readout_between" else 100_000_000
                rows["observation_ns"][2] = rows["control_ns"][2] + readout_ns
            elif case == "tactile":
                rows["tactile_dense_ns"] = rows["control_ns"][0]
                data["meta"].attrs["hand_tactile_hold_max_age_s"] = .15
            elif case == "source_future":
                rows["arm_ns"][2] = rows["observation_ns"][2] + 1
            elif case == "source_missing":
                rows["hand_ns"][2] = -1
            elif case == "source_regressed":
                rows["camera_ns"][2] = rows["camera_ns"][0]
            elif case == "sequence_regressed":
                rows["camera_sequence"] = [0, 1, 0]
            data["timing/rows"][:] = rows
    before = {file.name: file.read_bytes() for file in path.iterdir()}
    output = tmp_path / "canonical.zarr"
    processing = ExportConfig(pointcloud=PointCloudConfig(
        remove_table=False, num_points=32, workspace=(-1., -1., 0., 1., 1., 1.),
    ))
    if reason is not None:
        with pytest.raises(ValueError, match=reason) as caught:
            export_zarr(path, output, processing=processing)
        assert str(path) in str(caught.value)
        if row_index is not None:
            assert f"row {row_index}" in str(caught.value)
        assert not output.exists()
        assert bool(list(tmp_path.glob(".canonical.zarr.tmp-*"))) == (case in {"nan", "inf"})
    else:
        result = export_zarr(path, output, processing=processing)
        root = zarr.open_group(str(output), mode="r")
        assert result["total_frames"] == 3
        np.testing.assert_array_equal(root["meta/episode_ends"][:], [3])
        assert all(array.shape[0] == 3 for _, array in root["data"].arrays())
        quality = root.attrs["episode_quality"][0]
        assert set(quality) == {"max_control_grid_deviation_ns", "max_readout_ns", "mean_control_period_ns"}
        if case == "readout_between":
            assert quality["max_readout_ns"] == 30_000_000
        if case == "jitter":
            assert quality["max_control_grid_deviation_ns"] == 25_000_000
    assert before == {file.name: file.read_bytes() for file in path.iterdir()}


@pytest.mark.parametrize("writer_failure", [False, True])
@pytest.mark.parametrize("model_failure", ["pending", "error"])
def test_rollout_writer_finishes_independently_of_model(
    tmp_path, capture, monkeypatch, writer_failure, model_failure,
):
    from test_policy import make_session

    from dexmani_real.runtime.state import SafetyState

    metadata, row, command = capture
    entered, release = Event(), Event()
    events = []
    late_error = RuntimeError("late prediction failure")

    def predict(rows):
        entered.set()
        assert release.wait(5)
        if model_failure == "error":
            raise late_error
        return np.zeros((6, 19))

    worker = PolicyWorker(
        lambda: SimpleNamespace(predict=predict, close=lambda: events.append("model_close")),
        build_observation=lambda rows: rows,
    )
    state = SessionState()
    run_id = state.arm()
    state.stop_signal.clear()
    state.begin_motion(run_id=state.run_id)

    def stop():
        assert state.stop_signal.is_set()
        events.append("stop")

    episode = Rollout(
        state, RealConfig(), SimpleNamespace(control_dt_s=.1, n_obs_steps=1),
        run_id=run_id, robot=SimpleNamespace(stop=stop), observations=None, needs=None,
        sensors=None, action_mapper=None, model=worker, max_running_s=None,
        index=1, output_dir=tmp_path, record_raw=True, audio=Mock(),
    )
    episode.started_ns = 1
    episode.query = Prediction(0, None)
    recorder = episode.recorder
    writer_close = RawWriter.close

    def close_writer(writer):
        assert events == ["stop"] and (worker.future is not None)
        assert worker.future.running()
        writer_close(writer)
        events.append("writer_close")
        if writer_failure:
            raise OSError("writer close failed")

    monkeypatch.setattr(RawWriter, "close", close_writer)
    now = [0.0]
    monkeypatch.setattr(recorder_module, "time", SimpleNamespace(monotonic=lambda: now[0]))

    def prepare(poll):
        assert state.begin_motion(run_id=run_id) is not None
        worker.predict(())
        assert entered.wait(2)
        recorder.start(task_label="fixture", collection_source="policy_rollout",
                       metadata=metadata, episode_name="episode_001")
        recorder.append(row(0), command)
        join = recorder.thread.join

        def finish_join(timeout):
            assert timeout == recorder_module.RECORDER_STOP_TIMEOUT_S
            join(timeout=2)
            assert not recorder.writer_pending
            # 模拟 writer 消耗完整预算；模型仍应得到自己的完整等待预算。
            now[0] += recorder_module.RECORDER_STOP_TIMEOUT_S

        monkeypatch.setattr(recorder.thread, "join", finish_join)
        episode.stop("operator", run_end_reason=RunEndReason.OPERATOR)

    discard_pending = worker.discard_pending

    def retire(timeout):
        assert timeout == MODEL_SHUTDOWN_TIMEOUT_S
        assert events == ["stop", "writer_close"]
        assert recorder.result()["status"] == ("failed" if writer_failure else "published")
        events.append("model_retire")
        if model_failure == "error":
            release.set()
            worker.future.exception(timeout=2)
        discard_pending(0)  # 离线测试不实际等待完整的模型超时。

    monkeypatch.setattr(worker, "discard_pending", retire)
    episode.prepare = prepare
    session, _ = make_session(tmp_path)
    session.runtime_state = state
    session.model = worker
    session.poll_operator = Mock(return_value=run_id)
    session.robot.service_idle = Mock()
    session.action_mapper = session.camera_snapshot = None
    state.safety_state = SafetyState.ARMED
    factory = Mock(return_value=episode)
    monkeypatch.setattr('dexmani_real.deployment.session.Rollout', factory)
    try:
        worker.load()
        worker.future.result(timeout=2)
        worker.poll().result()
        expected = RecordingError if writer_failure else TimeoutError if model_failure == "pending" else RuntimeError
        with pytest.raises(expected) as caught:
            session.run_control_loop()
        assert session.episode is episode
        assert session.episodes == [episode.result]
        factory.assert_called_once()
        session.robot.service_idle.assert_called_once()
        assert events == ["stop", "writer_close", "model_retire"]
        assert [error["stage"] for error in episode.result.errors] == (
            ["recorder_finalize", "retired_model"] if writer_failure else ["retired_model"]
        )
        assert not episode.result.normal_termination
        assert (worker.future is not None) == (model_failure == "pending")
        assert episode.query is None
        if worker.future is not None:
            with pytest.raises(RuntimeError, match="unreclaimed"):
                worker.predict(())
        if writer_failure:
            assert caught.value is recorder.error
            assert episode.result.raw["status"] == "failed" and recorder.published_path is None
        else:
            assert episode.result.raw["status"] == "published" and recorder.error is None
            if model_failure == "error":
                assert caught.value is late_error
            with RawReader(recorder.published_path) as reader:
                assert reader.num_frames == 1
                reader.validate_completion()
                assert reader.meta["technical_status"] == "normal"
                np.testing.assert_array_equal(reader["action_arm_joint_target"][0], command.arm_qpos)
    finally:
        release.set()
        worker.close(state)
        assert worker.wait_closed(2) is True


@pytest.mark.parametrize("failure", ["writer", "queue"])
def test_finish_timeout_stays_failed_after_writer_exits(tmp_path, capture, monkeypatch, failure):
    entered, release = Event(), Event()
    append = RawWriter.append_frame

    def blocked_append(writer, *args):
        entered.set()
        assert release.wait(5)
        return append(writer, *args)

    monkeypatch.setattr(RawWriter, "append_frame", blocked_append)
    monkeypatch.setattr(recorder_module, "RECORDER_STOP_TIMEOUT_S", 0)
    metadata, row, command = capture
    recorder = EpisodeRecorder(tmp_path, control_hz=10)
    recorder.start(task_label="fixture", collection_source="teleop", metadata=metadata)
    try:
        recorder.append(row(0), command)
        assert entered.wait(5)
        if failure == "queue":
            for index in range(1, 17):
                recorder.append(row(index), command)
        message = "finish queue timed out" if failure == "queue" else "writer did not exit"
        with pytest.raises(RecordingError, match=message) as caught:
            recorder.finish(save=True, reason="test", technical_status="normal")
        assert recorder.writer_pending
    finally:
        release.set()
        recorder.thread.join(5)
    assert not recorder.writer_pending
    assert recorder.published_path is None and recorder.staging_path.is_dir()
    with pytest.raises(RecordingError) as repeated:
        recorder.finish(save=True, reason="test", technical_status="normal")
    assert repeated.value is caught.value


def test_interrupted_writer_wait_stays_pending_without_restarting_finish(tmp_path, capture, monkeypatch):
    from test_policy import make_rollout, make_session

    from dexmani_real.deployment.session import PolicySession

    entered, release = Event(), Event()
    append = RawWriter.append_frame
    def blocked_append(writer, *args):
        entered.set()
        assert release.wait(5)
        return append(writer, *args)
    monkeypatch.setattr(RawWriter, "append_frame", blocked_append)
    metadata, row, command = capture
    recorder = EpisodeRecorder(tmp_path / "raw", control_hz=10)
    recorder.start(task_label="fixture", collection_source="policy_rollout", metadata=metadata)
    recorder.append(row(0), command)
    join = recorder.thread.join
    interrupt = KeyboardInterrupt()
    interrupted_join = Mock(side_effect=interrupt)
    monkeypatch.setattr(recorder.thread, "join", interrupted_join)
    try:
        assert entered.wait(2)
        episode, _ = make_rollout(tmp_path)
        episode.recorder = recorder
        episode.prepare = lambda poll: episode.stop("operator", run_end_reason=RunEndReason.OPERATOR)
        with pytest.raises(KeyboardInterrupt) as caught:
            episode.run(poll_operator=lambda: None)
        assert caught.value is interrupt
        state = episode.runtime_state
        assert state.estop_request and state.emergency_signal.is_set() and state.stop_signal.is_set()
        assert episode.writer_pending and episode.result.raw["status"] == "pending"
        assert recorder.finish_request is not None and recorder.published_path is None
        session, events = make_session(tmp_path)
        session.runtime_state, session.episode, session.episodes = state, episode, [episode.result]
        session.save_results = PolicySession.save_results.__get__(session)
        assert session.close(failure=interrupt) == 1
        result = json.loads((tmp_path / "session_result.json").read_text())
        assert result["shutdown"] == "pending" and result["outcome"] == "fault"
        assert result["reason"] == "KeyboardInterrupt" and result["emergency_stop_requested"]
        assert result["episodes"][0]["raw"]["status"] == "pending"
        assert "model_close" in events and "audio_close" in events
        interrupted_join.assert_called_once()
    finally:
        release.set()
        join(2)
    assert not recorder.writer_pending
    with pytest.raises(RecordingError) as repeated:
        recorder.finish(save=True, reason="retry", technical_status="normal")
    assert repeated.value.__cause__ is interrupt
    interrupted_join.assert_called_once()
    assert recorder.published_path is None and recorder.staging_path.is_dir()



@pytest.mark.parametrize('period,frames,accepted', [
    (62_500_000, 160, True), (62_600_000, 160, True), (70_000_000, 160, False),
    (65_625_000, 3, True), (65_625_001, 3, False),
    (59_375_000, 3, True), (59_374_999, 3, False), (62_500_000, 1, True),
])
def test_export_mean_period_boundary(tmp_path, capture, period, frames, accepted, monkeypatch, caplog):
    from queue import Queue

    from dexmani_real.data.export import validate_export_episode
    # 批量构造磁盘 fixture，不以生产速度提交合成帧。
    monkeypatch.setattr(recorder_module, 'Queue', lambda **kw: Queue(maxsize=frames + 1))
    _, path = record(tmp_path / 'raw', capture, frames=frames)
    with h5py.File(path / 'data.h5', 'r+') as data:
        data['meta'].attrs['control_hz'] = 16
        rows = data['timing/rows'][:]
        stamps = 1_000_000_000 + np.arange(frames) * period
        for name in ('control_ns', 'arm_ns', 'hand_ns', 'tactile_aggregate_ns', 'tactile_dense_ns', 'camera_ns'):
            rows[name] = stamps
        rows['observation_ns'] = stamps + 1_000_000
        data['timing/rows'][:] = rows
    with RawReader(path) as reader:
        if not accepted:
            with pytest.raises(ValueError, match='mean_control_period_drift'):
                validate_export_episode(reader)
        else:
            _, quality = validate_export_episode(reader)
            assert quality['mean_control_period_ns'] == (period if frames > 1 else None)
            if frames == 1:
                assert 'single row, no sampling interval to verify' in caplog.text


@pytest.mark.parametrize('frames', [2, 4])
def test_rgb_count_mismatch_rejects_whole_episode(tmp_path, capture, frames):
    from dexmani_real.data.raw import _VideoWriter
    _, path = record(tmp_path / 'raw', capture)
    writer = _VideoWriter(path / 'rgb.mp4', fps=10, width=32, height=32)
    writer.open()
    try:
        for _ in range(frames):
            writer.write_frame(np.zeros((32, 32, 3), dtype=np.uint8))
    finally:
        writer.close()
    output = tmp_path / 'bad.zarr'
    with pytest.raises(ValueError):
        export_zarr(path, output, processing=ExportConfig(pointcloud=PointCloudConfig(
            remove_table=False, num_points=32, workspace=(-1., -1., 0., 1., 1., 1.),
        )))
    assert not output.exists()


def test_episode_discovery_allows_number_gaps(tmp_path):
    from dexmani_real.data.export import discover_episode_dirs
    for name in ('episode_002', 'episode_005'):
        (tmp_path / name).mkdir()
    assert [p.name for p in discover_episode_dirs(tmp_path)] == ['episode_002', 'episode_005']


def test_single_125ms_gap_at_16hz_rejects_episode(tmp_path, capture):
    from dexmani_real.data.export import validate_export_episode
    _, path = record(tmp_path / 'raw', capture)
    with h5py.File(path / 'data.h5', 'r+') as data:
        data['meta'].attrs['control_hz'] = 16
        rows = data['timing/rows'][:]
        rows['control_ns'] = [1_000_000_000, 1_062_500_000, 1_187_500_000]
        rows['observation_ns'] = rows['control_ns'] + 1_000_000
        data['timing/rows'][:] = rows
    with RawReader(path) as reader:
        with pytest.raises(ValueError, match='control_interval_gap_or_drift at row 2'):
            validate_export_episode(reader)


@pytest.mark.parametrize('stage', ['prepare', 'send'])
@pytest.mark.parametrize('key', ['s', 'c', 'd', 'q', 'esc'])
def test_teleop_blocking_stop_preserves_recorder_boundary(tmp_path, capture, monkeypatch, stage, key):
    from test_interaction import make_loop

    from dexmani_real.robot.commands import OperationCancelled
    from dexmani_real.teleop.loop import TeleopPhase
    loop = make_loop()
    metadata, row, command = capture
    clock = [1_000_000_000]
    monkeypatch.setattr('time.monotonic_ns', lambda: clock[0])
    loop.runtime.hand.feedback_max_age_s = .5
    loop.controller.rotation.as_matrix.return_value = np.eye(3)
    loop.recording_dir = tmp_path
    loop.runtime_config_json = "{}"
    monkeypatch.setattr('dexmani_real.teleop.loop.snapshot_recording_metadata', lambda *a, **kw: metadata.copy())
    monkeypatch.setattr('dexmani_real.teleop.loop.read_observation', lambda *a, **kw: row(0))
    loop.admit_observation = lambda row: []
    loop.controller.compute_target.return_value.hand_qpos = command.hand_qpos
    finished, captured = [], []
    finish = EpisodeRecorder.finish
    def finish_once(recorder, **kwargs):
        finished.append((recorder.submitted_frames, kwargs))
        return finish(recorder, **kwargs)
    monkeypatch.setattr(EpisodeRecorder, 'finish', finish_once)
    def interrupt(*args, **kwargs):
        loop.runtime_state.stop_signal.clear()
        recorder = loop.recorder
        captured.append(recorder)
        for name in ('b', 'h', 't', key):
            loop.keyboard.handle_key(name, True)
        assert loop.poll_blocking_commands()
        assert loop.recorder is recorder and recorder.finish_request is None
        assert not finished
        raise OperationCancelled('device revoked')
    if stage == 'prepare':
        loop.robot.prepare_streaming.side_effect = interrupt
    else:
        loop.robot.send_action.side_effect = interrupt
    loop.run_episode(loop.runtime_state.run_id)
    assert loop.phase is TeleopPhase.IDLE and loop.recorder is None
    assert len(finished) == 1 and finished[0][0] == (stage == 'send')
    assert finished[0][1]['save'] == (key != 'd')
    assert not loop.keyboard.poll(0)
    loop.robot.reset_tactile_sensors.assert_not_called()
    assert loop.robot.prepare_streaming.call_count == 1
    recorder = captured[0]
    if stage == 'send' and key != 'd':
        with RawReader(recorder.published_path) as reader:
            assert reader.num_frames == 1
            np.testing.assert_array_equal(reader['arm_qpos'][0], row(0).arm['qpos'][0])


@pytest.mark.parametrize('hz', [0, -1, float('nan'), float('inf'), 3e9])
def test_sampling_clock_rejects_invalid_period(hz):
    from dexmani_real.utils.timing import sampling_clock_ns
    with pytest.raises(ValueError):
        sampling_clock_ns(hz)


def test_sampling_clock_returns_integer_period():
    from dexmani_real.utils.timing import sampling_clock_ns
    assert type(sampling_clock_ns(16)) is int
    assert sampling_clock_ns(16) == 62_500_000


@pytest.mark.parametrize('field', ['arm_qpos', 'hand_qpos', 'action_arm_joint_target', 'action_hand_joint_target', 'hand_contact', 'hand_tactile_force'])
@pytest.mark.parametrize('bad_row', [0, 4])
def test_numeric_blocks_read_once_and_never_publish_bad_episode(tmp_path, capture, monkeypatch, field, bad_row):
    from dexmani_real.data import export as module
    from dexmani_real.data.raw import ROBOT_DATASET_SPECS
    _, path = record(tmp_path / 'raw', capture, frames=5)
    with h5py.File(path / 'data.h5', 'r+') as data:
        data[field][bad_row] = np.inf if bad_row else np.nan
    monkeypatch.setattr(module, 'MAX_BATCH_FRAMES', 2)
    reads = {name: [] for name in ROBOT_DATASET_SPECS}
    original = h5py.Dataset.__getitem__
    def read(dataset, key, *args, **kwargs):
        name = dataset.name.lstrip('/')
        if name in reads:
            reads[name].append(key)
        return original(dataset, key, *args, **kwargs)
    monkeypatch.setattr(h5py.Dataset, '__getitem__', read)
    with RawReader(path) as reader:
        module.validate_export_episode(reader)
    assert all(not count for count in reads.values())
    output = tmp_path / 'canonical.zarr'
    with pytest.raises(ValueError, match=f'nonfinite:{field} at row {bad_row}') as caught:
        export_zarr(path, output, processing=ExportConfig(pointcloud=PointCloudConfig(
            remove_table=False, num_points=32, workspace=(-1., -1., 0., 1., 1., 1.),
        )))
    assert str(path) in str(caught.value)
    expected = [slice(start, min(start + 2, 5)) for start in range(0, bad_row + 1, 2)]
    assert all(count == expected for count in reads.values())
    assert not output.exists()
    assert len(list(tmp_path.glob('.canonical.zarr.tmp-*'))) == 1


@pytest.mark.parametrize('failure', ['overflow', 'later_episode', 'derived'])
def test_late_conversion_failure_keeps_batch_unpublished(tmp_path, capture, monkeypatch, failure):
    from dexmani_real.data import export as module
    _, first = record(tmp_path / 'raw', capture)
    _, second = record(tmp_path / 'raw', capture)
    with h5py.File(second / 'data.h5', 'r+') as data:
        if failure == 'later_episode':
            data['hand_contact'][2] = np.nan
        elif failure == 'overflow':
            data['action_hand_joint_target'][2] = 1e300
    if failure == 'derived':
        geometry = module.hand_geometry
        def invalid(*args, **kwargs):
            result = geometry(*args, **kwargs)
            result['fingertip_xyz'][-1] = np.nan
            return result
        monkeypatch.setattr(module, 'hand_geometry', invalid)
    output = tmp_path / 'canonical.zarr'
    with np.errstate(over='ignore'), pytest.raises(ValueError, match='nonfinite'):
        export_zarr(first.parent, output, processing=ExportConfig(pointcloud=PointCloudConfig(
            remove_table=False, num_points=32, workspace=(-1., -1., 0., 1., 1., 1.),
        )))
    assert not output.exists()
    assert len(list(tmp_path.glob('.canonical.zarr.tmp-*'))) == 1


@pytest.mark.parametrize('field', ['arm_qpos', 'hand_tactile_force'])
@pytest.mark.parametrize('invalid', ['shape', 'dtype'])
def test_numeric_structure_rejected_before_staging(tmp_path, capture, field, invalid):
    _, path = record(tmp_path / 'raw', capture)
    with h5py.File(path / 'data.h5', 'r+') as data:
        values = data[field][:]
        del data[field]
        data[field] = values[:-1] if invalid == 'shape' else values.astype(np.int64)
    output = tmp_path / 'canonical.zarr'
    with pytest.raises(ValueError, match=f'{field}: incompatible shape/dtype'):
        export_zarr(path, output, processing=ExportConfig())
    assert not output.exists()
    assert not list(tmp_path.glob('.canonical.zarr.tmp-*'))


def test_rollout_reclaims_partially_opened_writer(tmp_path, capture, monkeypatch):
    from test_policy import make_rollout

    from dexmani_real.runtime.state import SafetyState

    episode, _ = make_rollout(tmp_path)
    episode.runtime_state.safety_state = SafetyState.ARMED
    episode.started_ns = 0
    episode.policy_info.task_name = 'fixture'
    episode.robot.hand = Mock()
    episode.robot.prepare_streaming = Mock()
    episode.robot.stop = Mock()
    episode.start_observation = Mock()
    recorder = episode.recorder = EpisodeRecorder(tmp_path, control_hz=10)
    metadata, _, _ = capture
    monkeypatch.setattr('dexmani_real.deployment.rollout.snapshot_recording_metadata',
                        lambda *args, **kwargs: metadata)
    primary = OSError('failure after opening Raw files')
    open_writer = RawWriter.open
    def fail_after_open(writer, metadata):
        open_writer(writer, metadata)
        raise primary
    monkeypatch.setattr(RawWriter, 'open', fail_after_open)
    finish = recorder.finish
    def finish_after_stop(**kwargs):
        episode.robot.stop.assert_called_once()
        return finish(**kwargs)
    monkeypatch.setattr(recorder, 'finish', finish_after_stop)
    with pytest.raises(RecordingError) as caught:
        episode.run(poll_operator=lambda: None)
    assert caught.value.__cause__ is primary
    assert recorder.start_attempted and not recorder.writer_pending
    assert not recorder.writer.data_file.id.valid
    assert recorder.writer.video.container is None
    assert episode.result.raw['status'] == 'failed'
    assert recorder.staging_path.is_dir() and recorder.published_path is None
    episode.start_observation.assert_not_called()


@pytest.mark.parametrize('cloud', [False, True])
def test_camera_ipc_pairing_with_real_pointcloud(monkeypatch, cloud):
    import multiprocessing as mp
    from dexmani_real.config.sensor import CameraConfig
    from dexmani_real.ipc.sensors import SensorChannels
    from dexmani_real.perception.pointcloud import PointCloudProcessor
    from dexmani_real.perception.worker import run_pointcloud_worker
    from dexmani_real.runtime.observation import ObservationNeeds, read_observation
    cfg = CameraConfig(width=32, height=32)
    camera = CameraSnapshot('test', CameraIntrinsics(32, 32, 200., 200., 15.5, 15.5, 'none', (0.,) * 5),
                            .001, np.eye(4))
    recipe = PointCloudConfig(remove_table=False, num_points=64, workspace=(-1., -1., 0., 1., 1., 1.))
    channels = SensorChannels(mp.get_context('spawn'), camera=cfg,
                              pointcloud_num_points=64 if cloud else None,
                              pointcloud_camera_fields=('rgb', 'depth') if cloud else ())
    rgb = np.full((32, 32, 3), 80, np.uint8)
    depth = np.full((32, 32), 500, np.uint16)
    sequence = channels.camera_ring.write(timestamp_ns=1234, rgb=rgb, depth=depth)
    if cloud:
        read = channels.camera_ring.read
        def one_frame():
            channels.is_running.value = False
            return read()
        monkeypatch.setattr(channels.camera_ring, 'read', one_frame)
        run_pointcloud_worker(channels, recipe, camera, None)
        expected = PointCloudProcessor(depth_scale_m=.001, intrinsics=camera.color,
                                       T_xarm_base_from_color=np.eye(4), table_plane_abcd=None,
                                       config=recipe).process(depth, rgb)
        assert expected.shape == (64, 6) and expected.dtype == np.float32
        # 新相机帧不得替换点云载荷中已经配对的图像。
        channels.camera_ring.write(timestamp_ns=9999, rgb=rgb + 1, depth=depth + 1)
    robot = SimpleNamespace(read_state=lambda: SimpleNamespace(arm=np.zeros(1, dtype=ARM_STATE_DTYPE), hand=None))
    row = read_observation(channels, robot, needs=ObservationNeeds(camera_fields=('rgb', 'depth'), point_cloud=cloud))
    assert row.camera['timestamp_ns'] == 1234 and row.camera['ring_sequence'] == sequence
    np.testing.assert_array_equal(row.camera['rgb'], rgb)
    np.testing.assert_array_equal(row.camera['depth'], depth)
    if cloud:
        np.testing.assert_array_equal(row.point_cloud, expected)
        assert row.pointcloud_timestamp_ns == 1234


def test_camera_worker_publishes_current_payload(monkeypatch):
    import multiprocessing as mp
    from dexmani_real.config.sensor import CameraConfig
    from dexmani_real.ipc.sensors import SensorChannels, read_camera_frame, snapshot_camera_parameters
    from dexmani_real.sensor import realsense
    config = CameraConfig(width=2, height=2)
    channels = SensorChannels(mp.get_context('spawn'), camera=config)
    intrinsics = CameraIntrinsics(2, 2, 200., 200., .5, .5, 'none', (0.,) * 5)
    rgb = np.full((2, 2, 3), 80, np.uint8)
    depth = np.full((2, 2), 500, np.uint16)
    def read():
        channels.is_running.value = False
        return realsense.RGBDFrame(rgb, depth, 41, 42, 1234)
    camera = SimpleNamespace(connect=Mock(), disconnect=Mock(), read=read, depth_scale_m=.001,
                             active_serial='test', color_intrinsics=intrinsics)
    monkeypatch.setattr(realsense, 'RealSenseCamera', lambda cfg: camera)
    realsense.run_camera(channels, config)
    camera.disconnect.assert_called_once_with()
    frame = read_camera_frame(channels)
    np.testing.assert_array_equal(frame['rgb'], rgb)
    np.testing.assert_array_equal(frame['depth'], depth)
    assert frame['timestamp_ns'] == 1234 and frame['ring_sequence'] == 1
    snapshot = snapshot_camera_parameters(channels, config, None, include_extrinsics=False)
    assert snapshot.serial == 'test' and snapshot.color == intrinsics
    assert snapshot.depth_scale_m == .001
