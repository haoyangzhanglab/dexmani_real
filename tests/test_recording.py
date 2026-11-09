"""录制失败不影响控制；真实文件与受控 writer 验证资源及时间契约。"""

import threading
from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest
from test_freshness import make_rollout, make_teleop, observation

from dexmani_real.data.recorder import EpisodeRecorder, RecordingError
from dexmani_real.runtime.observation import ObservationNeeds


@pytest.fixture
def record_raw(tmp_path, camera_snapshot):
    from dexmani_real.data.raw import snapshot_recording_metadata
    from dexmani_real.robot.commands import RobotCommand

    def record(*, stamps=(1_000_000_000, 1_050_000_000, 1_100_000_000), gaps=(), name="episode"):
        recorder = EpisodeRecorder(tmp_path, control_hz=20)
        recorder.start(task_label="test", collection_source="policy", episode_name=name,
                       metadata=snapshot_recording_metadata(camera_snapshot, tactile_reset_metadata={}))
        try:
            for i, stamp in enumerate(stamps):
                for _ in range(gaps.count(i)):
                    recorder.append(observation(stamp - 1), quality_issue="test_missing_row")
                row = observation(stamp)
                row.camera["depth"][:] = i + 1
                recorder.append(row, RobotCommand(row.arm["qpos"][0], row.hand["qpos"][0]))
            if len(stamps) in gaps:
                recorder.append(observation(stamps[-1] + 50_000_000), quality_issue="test_missing_tail")
        finally:
            path = recorder.finish(save=True, reason="operator")
        return path

    return record


@pytest.mark.parametrize("gaps,expected", [
    ((), [0]), ((0, 0), [0]), ((3,), [0]), ((0, 3), [0]),
    ((1,), [0, 1]), ((1, 1, 1), [0, 1]), ((1, 2), [0, 1, 2]),
])
def test_saved_segments_and_training_boundaries(record_raw, tmp_path, runtime, gaps, expected):
    from dexmani_policy.datasets.episode_store import EpisodeStore
    from dexmani_policy.datasets.sampler import SequenceSampler

    from dexmani_real.data.export import ExportConfig, export_zarr
    from dexmani_real.data.raw import RawReader
    from dexmani_real.replay.trajectory import load_trajectory

    path = record_raw(gaps=gaps)
    with RawReader(path) as reader:
        assert reader.segment_starts.tolist() == expected
    if len(expected) == 1:
        assert len(load_trajectory(path, runtime)) == 3
    output = tmp_path / "out.zarr"
    result = export_zarr(path, output, processing=ExportConfig(), fields=["joint_state", "action"])
    store = EpisodeStore(output)
    assert store.episode_ends.tolist() == expected[1:] + [3]
    assert result["episode_count"] == len(expected)
    sampler = SequenceSampler(store, 3, pad_before=2, pad_after=2,
                              episode_mask=np.ones(len(expected), dtype=bool))
    for i in range(len(sampler)):
        rows = sampler.sample_source_rows(i)
        assert len(set(np.searchsorted(store.episode_ends, rows, side="right"))) == 1


def test_jitter_is_single_segment(record_raw, tmp_path, runtime, caplog):
    from dexmani_real.data.export import ExportConfig, export_zarr
    from dexmani_real.replay.trajectory import load_trajectory

    path = record_raw(stamps=(1_000_000_000, 1_063_000_000, 1_100_000_000))
    result = export_zarr(path, tmp_path / "out.zarr", processing=ExportConfig(), fields=["action"])
    assert result["episode_count"] == 1
    assert result["total_frames"] == len(load_trajectory(path, runtime)) == 3
    assert "不重采样" in caplog.text


@pytest.mark.parametrize("marked", [False, True])
def test_legacy_recording_requires_evidence(record_raw, tmp_path, runtime, marked):
    import h5py

    from dexmani_real.data.export import ExportConfig, export_zarr
    from dexmani_real.replay.trajectory import load_trajectory

    path = record_raw(gaps=(0,) if marked else (), stamps=(1_000_000_000, 1_500_000_000))
    with h5py.File(path / "data.h5", "r+") as data:
        del data["timing/segment_starts"]
    if marked:
        with pytest.raises(ValueError, match="insufficient information"):
            load_trajectory(path, runtime)
        with pytest.raises(ValueError, match="insufficient information"):
            export_zarr(path, tmp_path / "out.zarr", processing=ExportConfig(), fields=["action"])
        assert not (tmp_path / "out.zarr").exists()
    else:
        assert len(load_trajectory(path, runtime)) == 2
        result = export_zarr(path, tmp_path / "out.zarr", processing=ExportConfig(), fields=["action"])
        assert result["episode_count"] == 1
    with h5py.File(path / "data.h5") as data:
        assert "timing/segment_starts" not in data


@pytest.mark.parametrize("starts", [[], 0, "bad", [1], [0, 0], [0, 2, 1], [0, 3], [0, -1], [0.0], [False], [[0]]])
def test_bad_segment_structure_rejected_before_devices(record_raw, tmp_path, runtime, monkeypatch, starts):
    import h5py

    from dexmani_real.data.export import ExportConfig, export_zarr
    from dexmani_real.replay.run import run_replay

    path = record_raw()
    with h5py.File(path / "data.h5", "r+") as data:
        del data["timing/segment_starts"]
        data["timing"].create_dataset("segment_starts", data=starts if isinstance(starts, str) else np.asarray(starts))
    robot = Mock(side_effect=AssertionError("hardware must not be constructed"))
    monkeypatch.setattr("dexmani_real.replay.run.Robot", robot)
    with pytest.raises(ValueError, match="segment_starts"):
        run_replay(path, runtime)
    robot.assert_not_called()
    output = tmp_path / "out.zarr"
    with pytest.raises(ValueError, match="segment_starts"):
        export_zarr(path, output, processing=ExportConfig(), fields=["action"])
    assert not output.exists()


@pytest.mark.parametrize("segment", [None, -1, 2])
def test_replay_selection_precedes_devices(record_raw, runtime, monkeypatch, segment):
    from dexmani_real.replay.run import run_replay

    path = record_raw(gaps=(2,))
    robot = Mock(side_effect=AssertionError("hardware must not be constructed"))
    monkeypatch.setattr("dexmani_real.replay.run.Robot", robot)
    with pytest.raises(ValueError, match=r"0: rows \[0, 2\).*1: rows \[2, 3\)"):
        run_replay(path, runtime, segment=segment)
    robot.assert_not_called()


@pytest.mark.parametrize("at_selected_start", [False, True])
def test_replay_executes_only_selected_segment(record_raw, runtime, clock, monkeypatch, at_selected_start):
    import h5py

    from dexmani_real.replay import run as replay

    path = record_raw(gaps=(2,))
    with h5py.File(path / "data.h5", "r+") as data:
        selected = data["action_arm_joint_target"][2]
        selected[0] += 0.2  # 与第一段起点差异超过原有 10 度限制。
        data["action_arm_joint_target"][2] = selected
    row = observation(clock.now_ns)
    if at_selected_start:
        row.arm["qpos"][0] = selected
    robot, keyboard = Mock(), Mock()
    robot.read_state.return_value = row
    keyboard.poll.return_value = ([], False)
    monkeypatch.setattr(replay, "Robot", Mock(return_value=robot))
    monkeypatch.setattr(replay, "Keyboard", Mock(return_value=keyboard))
    monkeypatch.setattr(replay.time, "sleep", lambda seconds: setattr(clock, "now_ns", clock.now_ns + round(seconds * 1e9)))
    if at_selected_start:
        replay.run_replay(path, runtime, segment=1)
        robot.send_action.assert_called_once()
        np.testing.assert_array_equal(robot.send_action.call_args.args[0].arm_qpos, selected)
    else:
        with pytest.raises(ValueError, match="start differs"):
            replay.run_replay(path, runtime, segment=1)
        robot.send_action.assert_not_called()
    robot.close.assert_called_once()
    keyboard.stop.assert_called_once()


def test_multiple_raws_decode_once_and_share_boundaries(record_raw, tmp_path, monkeypatch):
    import zarr

    from dexmani_real.data.export import ExportConfig, export_zarr
    from dexmani_real.data.raw import RawReader

    record_raw(gaps=(1,), name="episode_1")
    record_raw(gaps=(2,), name="episode_2")
    decoded = []
    original = RawReader.iter_rgb_frames
    def decode(reader):
        decoded.append(reader.path.name)
        yield from original(reader)
    monkeypatch.setattr(RawReader, "iter_rgb_frames", decode)
    output = tmp_path / "out.zarr"
    result = export_zarr(tmp_path, output, processing=ExportConfig(), fields=["rgb", "depth", "action", "joint_state"])
    assert result["raw_episode_count"] == 2
    assert result["episode_count"] == 4
    assert result["total_frames"] == 6
    data = zarr.open_group(str(output), mode="r")
    assert data["meta/episode_ends"][:].tolist() == [1, 3, 5, 6]
    assert data["data/depth"][:, 0, 0].tolist() == [1, 2, 3, 1, 2, 3]
    assert all(len(data["data"][field]) == 6 for field in data["data"])
    assert decoded == ["episode_1", "episode_2"]


def test_queue_full_segment_survives_async_writer(tmp_path, camera_snapshot, monkeypatch, caplog):
    from dexmani_real.data.raw import RawReader, RawWriter, snapshot_recording_metadata

    entered, release, drained = threading.Event(), threading.Event(), threading.Event()
    class GatedWriter(RawWriter):
        def append_frame(self, *item):
            if not entered.is_set():
                entered.set()
                assert release.wait(3)
            super().append_frame(*item)
            if self.frame_count + self.buffered == 17:
                drained.set()
    monkeypatch.setattr("dexmani_real.data.recorder.RawWriter", GatedWriter)
    recorder = EpisodeRecorder(tmp_path, control_hz=20)
    recorder.start(task_label="test", collection_source="policy",
                   metadata=snapshot_recording_metadata(camera_snapshot, tactile_reset_metadata={}))
    try:
        recorder.append(observation(1_000_000_000))
        assert entered.wait(3)
        for i in range(1, 19):
            recorder.append(observation(1_000_000_000 + i * 50_000_000))
        assert recorder.pending_segment_start
        release.set()
        assert drained.wait(3)
        recorder.append(observation(1_950_000_000))
        recorder.append(observation(2_000_000_000))
    finally:
        release.set()
        path = recorder.finish(save=True, reason="operator")
    with RawReader(path) as reader:
        assert reader.num_frames == 19
        assert reader.segment_starts.tolist() == [0, 17]
        assert reader["timing/rows"][:]["control_ns"].tolist() == [
            1_000_000_000 + i * 50_000_000 for i in (*range(17), 19, 20)
        ]
    assert sum("录制存在缺失" in record.message for record in caplog.records) == 1


def test_segment_indices_include_unflushed_rows(tmp_path, camera_snapshot):
    from dexmani_real.data.raw import RawReader, RawWriter, snapshot_recording_metadata

    writer = RawWriter(tmp_path, control_hz=20)
    try:
        writer.open(dict(snapshot_recording_metadata(camera_snapshot, tactile_reset_metadata={}),
                         collection_source="policy", task_label="test"))
        for i in range(40):
            writer.append_frame(observation(1_000_000_000 + i * 50_000_000), None,
                                segment_start=i in (0, 2, 31, 32, 39))
        writer.finish("operator", [])
    finally:
        writer.close()
    with RawReader(tmp_path) as reader:
        assert reader.num_frames == 40
        assert reader.segment_starts.tolist() == [0, 2, 31, 32, 39]


@pytest.mark.parametrize("mode", ["teleop", "sync", "async", "rtc"])
@pytest.mark.parametrize("skip", [False, True])
def test_scheduler_records_actual_skips(clock, runtime, tmp_path, camera_snapshot, monkeypatch, mode, skip):
    from dexmani_real.data.raw import RawReader, snapshot_recording_metadata
    from dexmani_real.deployment.schedule import ActionChunk

    runtime = replace(runtime, teleop=replace(runtime.teleop, control_hz=20),
                      execution=replace(runtime.execution, mode="sync" if mode == "teleop" else mode,
                                        action_steps=8, prefetch_steps=1))
    if mode == "teleop":
        loop = make_teleop(runtime, clock)
        loop.sensors.check = lambda **kw: True
        monkeypatch.setattr("dexmani_real.teleop.loop.read_observation", lambda *a, **kw: observation(clock.now_ns))
        tick = loop.run_control_tick
    else:
        loop = make_rollout(runtime, ObservationNeeds(), clock)
        loop.plan = ActionChunk(np.zeros((10, 19)), 0)
        monkeypatch.setattr(loop, "sample_observation", lambda: (observation(clock.now_ns), observation(clock.now_ns)))
        tick = loop.step
    recorder = loop.recorder = EpisodeRecorder(tmp_path, control_hz=20)
    recorder.start(task_label="test", collection_source="policy",
                   metadata=snapshot_recording_metadata(camera_snapshot, tactile_reset_metadata={}))
    stamps = [1_000_000_000, 1_110_000_000, 1_150_000_000] if skip else [1_000_000_000, 1_063_000_000, 1_100_000_000]
    try:
        for stamp in stamps:
            clock.now_ns = stamp
            tick()
        assert loop.robot.send_action.call_count == 3
    finally:
        path = recorder.finish(save=True, reason="operator")
    with RawReader(path) as reader:
        assert reader.num_frames == 3
        assert reader.segment_starts.tolist() == ([0, 1] if skip else [0])
        assert reader["timing/rows"][:]["control_ns"].tolist() == stamps


@pytest.mark.parametrize("partial_submission", [False, True])
def test_pause_segment_follows_last_attempt(clock, runtime, tmp_path, camera_snapshot, monkeypatch, partial_submission):
    from dexmani_real.data.raw import RawReader, snapshot_recording_metadata
    from dexmani_real.robot.commands import FeedbackUnavailable
    from dexmani_real.teleop.loop import TeleopPhase

    loop = make_teleop(runtime, clock)
    loop.sensors.check = lambda **kw: True
    monkeypatch.setattr("dexmani_real.teleop.loop.read_observation", lambda *a, **kw: observation(clock.now_ns))
    recorder = loop.recorder = EpisodeRecorder(tmp_path, control_hz=16)
    recorder.start(task_label="test", collection_source="policy",
                   metadata=snapshot_recording_metadata(camera_snapshot, tactile_reset_metadata={}))
    loop.robot.prepare_streaming.side_effect = lambda *a, **kw: loop.runtime_state.stop_signal.clear()
    try:
        loop.run_control_tick()
        if partial_submission:
            clock.now_ns += 62_500_000
            loop.robot.send_action.side_effect = FeedbackUnavailable("hand expired after arm submission")
            loop.run_control_tick()
            loop.robot.send_action.side_effect = None
        else:
            loop.pause("vr_stale")
        assert loop.phase is TeleopPhase.PAUSED
        assert loop.runtime_state.stop_signal.is_set()
        clock.now_ns += 62_500_000
        loop.run_control_tick()  # 准备完成，不提交。
        clock.now_ns += 62_500_000
        loop.run_control_tick()  # 新反馈，重新锚定并记录恢复后的第一行。
        assert loop.phase is TeleopPhase.ACTIVE
    finally:
        path = recorder.finish(save=True, reason="operator")
    with RawReader(path) as reader:
        assert reader.segment_starts.tolist() == [0, 2 if partial_submission else 1]
        assert reader.num_frames == (3 if partial_submission else 2)
        assert np.isfinite(reader["action_arm_joint_target"][:]).all()


def test_queue_full_does_not_stop_teleop(clock, runtime, tmp_path):
    loop = make_teleop(runtime, clock)
    loop.sensors.check = lambda **kw: True
    recorder = loop.recorder = EpisodeRecorder(tmp_path, control_hz=16)
    recorder.is_recording = True
    for _ in range(16):
        recorder.append(observation(clock.now_ns))
    loop.run_control_tick()
    assert recorder.recording_truncated_reason == "recording_queue_full"
    assert loop.pending_termination_reason is None
    assert not loop.runtime_state.stop_signal.is_set()
    recorder.queue.get_nowait()
    clock.now_ns += 62_500_000
    loop.run_control_tick()
    assert loop.robot.send_action.call_count == 2
    assert recorder.queue.qsize() == 16
    assert recorder.is_recording


@pytest.mark.parametrize("stage", ["open", "append", "close", "publish"])
def test_writer_failure_is_reclaimed_not_published(tmp_path, monkeypatch, stage):
    entered, release = threading.Event(), threading.Event()

    class Writer:
        frame_count = 1

        def __init__(self, *a, **k):
            pass

        def open(self, metadata):
            if stage == "open":
                raise OSError("open failed")

        def append_frame(self, *a):
            if stage == "append":
                raise OSError("write failed")

        def finish(self, *a, **k):
            pass

        def flush(self):
            pass

        def close(self):
            entered.set()
            assert release.wait(3)
            if stage == "close":
                raise OSError("close failed")

    monkeypatch.setattr("dexmani_real.data.recorder.RawWriter", Writer)
    if stage == "publish":
        monkeypatch.setattr(
            "dexmani_real.data.recorder.publish",
            Mock(side_effect=OSError("publish failed")),
        )
    recorder = EpisodeRecorder(tmp_path, control_hz=20)
    try:
        if stage == "open":
            with pytest.raises(RecordingError):
                recorder.start(
                    task_label="test", collection_source="policy", metadata={}
                )
        else:
            recorder.start(task_label="test", collection_source="policy", metadata={})
            recorder.append(observation(1_000_000_000))
        recorder.request_finish(save=True, reason="operator")
        assert entered.wait(3)
        assert not recorder.poll_finish()
        if stage in {"open", "append"}:
            recorder.service()  # 已报告 I/O 错误，仍须持有活线程。
        assert recorder.writer_pending
        expired = replace(recorder.finish_request, deadline=0)
        original, recorder.finish_request = recorder.finish_request, expired
        with pytest.raises(TimeoutError, match="did not exit"):
            recorder.poll_finish()
        recorder.finish_request = original
    finally:
        release.set()
        if recorder.thread is not None:
            recorder.thread.join(3)
    assert not recorder.writer_pending
    assert recorder.poll_finish()
    assert recorder.result()["status"] == "failed"
    assert recorder.result()["error"]
    assert recorder.staging_path.exists()
    assert not list(tmp_path.glob("episode_*"))


@pytest.mark.parametrize(
    "error", [ValueError("bug"), KeyboardInterrupt(), SystemExit(2)]
)
def test_recording_does_not_hide_program_errors(tmp_path, error):
    recorder = EpisodeRecorder(tmp_path, control_hz=20)
    recorder.store_error(OSError("disk"))
    recorder.store_error(error)
    with pytest.raises(type(error)):
        recorder.service()


@pytest.mark.parametrize("gap", [False, True])
def test_raw_video_alignment_and_gap_export(tmp_path, camera_snapshot, gap):
    import av

    from dexmani_real.config.real import RealConfig
    from dexmani_real.data.export import ExportConfig, export_zarr
    from dexmani_real.data.raw import RawReader, snapshot_recording_metadata
    from dexmani_real.replay.trajectory import load_trajectory
    from dexmani_real.robot.commands import RobotCommand

    recorder = EpisodeRecorder(tmp_path, control_hz=20)
    recorder.start(
        task_label="test",
        collection_source="policy",
        metadata=snapshot_recording_metadata(
            camera_snapshot, tactile_reset_metadata={}
        ),
    )
    stamps = [1_000_000_000, 1_050_000_000, 1_750_000_000 if gap else 1_100_000_000]
    for i, stamp in enumerate(stamps):
        row = observation(stamp)
        row.camera["rgb"][:] = i * 60
        row.camera["depth"][:] = i + 1
        if gap and i == 2:
            recorder.mark_gap("camera_source_stale")
        recorder.append(row, RobotCommand(row.arm["qpos"][0], row.hand["qpos"][0]))
    path = recorder.finish(save=True, reason="operator")
    assert recorder.result()["status"] == "published"
    assert not recorder.writer_pending
    with RawReader(path) as reader:
        assert reader["timing/rows"][:]["control_ns"].tolist() == stamps
        assert reader["depth"][:, 0, 0].tolist() == [1, 2, 3]
        assert len(list(reader.iter_rgb_frames())) == 3
    with av.open(str(path / "rgb.mp4")) as video:
        frames = list(video.decode(video=0))
        assert [round(float(f.pts * f.time_base) * 1e9) for f in frames] == [
            t - stamps[0] for t in stamps
        ]
        np.testing.assert_allclose(
            [f.to_ndarray(format="rgb24").mean() for f in frames], [0, 60, 120], atol=2
        )
    output = tmp_path / "out.zarr"
    import zarr

    result = export_zarr(path, output, processing=ExportConfig(), fields=["joint_state", "action", "rgb", "depth"])
    assert result["raw_episode_count"] == 1
    assert result["episode_count"] == (2 if gap else 1)
    assert result["total_frames"] == 3
    exported = zarr.open_group(str(output), mode="r")
    assert exported["meta/episode_ends"][:].tolist() == ([2, 3] if gap else [3])
    assert exported["data/depth"][:, 0, 0].tolist() == [1, 2, 3]
    np.testing.assert_allclose(exported["data/rgb"][:].mean(axis=(1, 2, 3)), [0, 60, 120], atol=2)
    with RawReader(path) as reader:
        assert reader.segment_starts.tolist() == ([0, 2] if gap else [0])
    if gap:
        with pytest.raises(ValueError, match="--segment"):
            load_trajectory(path, RealConfig())
        assert len(load_trajectory(path, RealConfig(), segment=0)) == 2
        assert len(load_trajectory(path, RealConfig(), segment=1)) == 1
    else:
        trajectory = load_trajectory(path, RealConfig())
        assert len(trajectory) == 3
        assert trajectory.control_hz == 20


@pytest.mark.parametrize("invalid", ["row_count", "semantics", "control_clock", "observation_clock"])
def test_export_and_replay_share_raw_validation(tmp_path, camera_snapshot, invalid):
    import h5py

    from dexmani_real.config.real import RealConfig
    from dexmani_real.data.export import ExportConfig, export_zarr
    from dexmani_real.data.raw import snapshot_recording_metadata
    from dexmani_real.replay.trajectory import load_trajectory
    from dexmani_real.robot.commands import RobotCommand

    recorder = EpisodeRecorder(tmp_path, control_hz=20)
    recorder.start(
        task_label="test", collection_source="policy",
        metadata=snapshot_recording_metadata(camera_snapshot, tactile_reset_metadata={}),
    )
    for stamp in (1_000_000_000, 1_050_000_000):
        row = observation(stamp)
        recorder.append(row, RobotCommand(row.arm["qpos"][0], row.hand["qpos"][0]))
    path = recorder.finish(save=True, reason="operator")
    with h5py.File(path / "data.h5", "r+") as raw:
        if invalid == "row_count":
            raw["action_arm_joint_target"].resize(1, axis=0)
            error = "incompatible shape"
        elif invalid == "semantics":
            raw["meta"].attrs["arm_action_semantics"] = "relative_joint_delta"
            error = "arm_action_semantics"
        else:
            rows = raw["timing/rows"][:]
            if invalid == "control_clock":
                rows["control_ns"][1] = rows["control_ns"][0]
                error = "control_clock_not_increasing"
            else:
                rows["observation_ns"][0] = rows["control_ns"][0] - 1
                error = "invalid_control_or_observation_clock"
            raw["timing/rows"][:] = rows
    with pytest.raises(ValueError, match=error):
        load_trajectory(path, RealConfig())
    output = tmp_path / "out.zarr"
    with pytest.raises(ValueError, match=error):
        export_zarr(path, output, processing=ExportConfig(), fields=["action"])
    assert not output.exists()


@pytest.mark.parametrize("required", [False, True])
def test_recording_worker_is_not_control_dependency(runtime, required):
    from dexmani_real.deployment.session import PolicySession
    from dexmani_real.robot.commands import RobotError
    from dexmani_real.robot.system import Robot
    from dexmani_real.runtime.lifecycle import SensorWorkers
    from dexmani_real.runtime.state import SessionState

    session = PolicySession.__new__(PolicySession)
    session.runtime_state = SessionState()
    session.keyboard = Mock()
    session.keyboard.check_health.return_value = True
    session.model_needs = ObservationNeeds(camera_fields=("rgb",) if required else ())
    session.sensors = SensorWorkers()
    camera = Mock(exitcode=1, pid=123)
    camera.name = "camera"
    camera.is_alive.return_value = False
    session.sensors.processes = [camera]
    robot = Robot(
        runtime.arm,
        runtime.hand,
        runtime_state=session.runtime_state,
        check_services=session.check_services,
    )
    if required:
        with pytest.raises(RobotError):
            robot.check()
        with pytest.raises(RuntimeError, match="exitcode=1"):
            session.sensors.shutdown(session.runtime_state)
    else:
        robot.check()
        assert not session.runtime_state.stop_signal.is_set()
        session.sensors.shutdown(session.runtime_state, optional_names=("camera",))
    assert not session.sensors.processes


def test_failed_writer_control_keeps_running(clock, runtime, tmp_path):
    loop = make_teleop(runtime, clock)
    loop.sensors.check = lambda **kw: True
    recorder = loop.recorder = EpisodeRecorder(tmp_path, control_hz=16)
    recorder.store_error(OSError("disk failed"))
    for _ in range(2):
        loop.run_control_tick()
        clock.now_ns += 62_500_000
    assert loop.robot.send_action.call_count == 2
    assert loop.pending_termination_reason is None
    assert not loop.runtime_state.stop_signal.is_set()
    assert recorder.result()["status"] == "failed"
    assert not recorder.is_recording
    assert loop.finish_recording()
    assert loop.recorder is None
    assert all(call.args[0] not in {"save", "discard"} for call in loop.audio.play.call_args_list)


def test_teleop_recorder_start_failure_can_retry(
    clock, runtime, tmp_path, camera_snapshot, monkeypatch
):
    from dexmani_real.teleop.loop import TeleopPhase

    loop = make_teleop(runtime, clock)
    loop.phase = TeleopPhase.IDLE
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    loop.recording_dir = blocked
    loop.camera_snapshot = camera_snapshot
    loop.max_duration_s = 10
    loop.robot.hand.tactile_reset_metadata = {}
    loop.controller.rotation.as_matrix.return_value = np.eye(3)
    monkeypatch.setattr(
        loop, "start_observation", lambda **kw: observation(clock.now_ns)
    )
    loop.run_episode(loop.runtime_state.cancel_epoch)
    assert loop.phase is TeleopPhase.IDLE
    assert loop.recorder is None
    assert not loop.runtime_state.quit_requested
    loop.robot.prepare_streaming.assert_not_called()
    loop.recording_dir = tmp_path / "valid"
    loop.robot.prepare_streaming.side_effect = lambda epoch, **kw: (
        loop.runtime_state.stop_signal.clear()
    )
    try:
        loop.start_episode(loop.runtime_state.cancel_epoch)
        assert loop.phase is TeleopPhase.ACTIVE
        assert loop.recorder.is_recording
        assert loop.pending_termination_reason is None
    finally:
        loop.request_episode_end("test_finished")
        loop.end_episode()
    assert loop.recorder is None


def test_policy_recorder_start_failure_is_episode_scoped(
    clock, runtime, tmp_path, camera_snapshot, monkeypatch
):
    rollout = make_rollout(runtime, ObservationNeeds(), clock, record_raw=True)
    rollout.started_ns = 0
    rollout.policy_info.task_name = "test"
    rollout.camera_snapshot = camera_snapshot
    rollout.robot.hand.tactile_reset_metadata = {}
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    rollout.output_dir = blocked
    monkeypatch.setattr(
        rollout, "start_observation", lambda *a, **kw: observation(clock.now_ns)
    )
    result = rollout.run(poll_operator=lambda: None)
    assert result.termination_reason == "preparation_failed"
    rollout.robot.prepare_streaming.assert_not_called()
    rollout.robot.send_action.assert_not_called()
    assert not rollout.runtime_state.quit_requested
    assert rollout.finish_recording()
    assert result.raw["status"] == "failed"
    assert rollout.recorder is None


def test_unmarked_time_gap_is_diagnostic(record_raw, caplog):
    from dexmani_real.data.raw import RawReader

    stamps = [1_000_000_000, 1_100_000_000]
    path = record_raw(stamps=stamps)
    with RawReader(path) as reader:
        assert reader.read_timing()["control_ns"].tolist() == stamps
    assert "不重采样" in caplog.text
    assert str(path) in caplog.text


def test_pointcloud_and_raw_use_independent_images(clock, runtime):
    import multiprocessing as mp

    from dexmani_real.config.sensor import CameraConfig
    from dexmani_real.ipc.sensors import SensorChannels

    rollout = make_rollout(
        runtime,
        ObservationNeeds(point_cloud=True, camera_fields=("rgb",)),
        clock,
        record_raw=True,
    )
    channels = SensorChannels(
        mp.get_context("spawn"),
        camera=CameraConfig(width=4, height=4),
        pointcloud_num_points=2,
        camera_fields=("rgb",),
    )
    rollout.sensors.channels = channels
    rollout.robot.read_state.return_value = observation(clock.now_ns)
    channels.camera_ring.write(
        timestamp_ns=clock.now_ns - 10_000_000,
        rgb=np.full((4, 4, 3), 20, np.uint8),
        depth=np.ones((4, 4), np.uint16),
    )
    channels.pointcloud_ring.write(
        timestamp_ns=clock.now_ns - 30_000_000,
        source_camera_sequence=9,
        point_cloud=np.ones((2, 6), np.float32),
        rgb=np.full((4, 4, 3), 10, np.uint8),
    )
    model, raw = rollout.read_observations()
    assert model.camera["ring_sequence"] == 9
    assert raw.camera["ring_sequence"] == 1
    assert model.camera["rgb"][0, 0, 0] == 10
    assert raw.camera["rgb"][0, 0, 0] == 20
    assert "depth" in raw.camera


@pytest.mark.parametrize("key", ["s", "d", "q", "esc"])
def test_operator_still_works_after_recording_failure(clock, runtime, tmp_path, key):
    from test_teleop_recovery import live_keyboard

    loop = make_teleop(runtime, clock)
    keyboard = live_keyboard(loop)
    loop.recorder = EpisodeRecorder(tmp_path, control_hz=16)
    loop.recorder.store_error(OSError("disk failed"))
    loop.recorder.service()
    assert loop.recorder.finish_request is not None
    keyboard.handle_key(key, True)
    assert not loop.service_episode(0)
    assert loop.pending_termination_reason == (
        "estop" if key == "esc" else "quit" if key == "q" else "operator"
    )
    assert loop.discard_requested == (key == "d")
    loop.robot.send_action.assert_not_called()


def test_writer_interrupt_is_not_hidden_by_earlier_program_error(tmp_path):
    recorder = EpisodeRecorder(tmp_path, control_hz=20)
    recorder.store_error(ValueError("writer bug"))
    recorder.store_error(KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        recorder.service()


@pytest.mark.parametrize("key", ["s", "d", "q", "esc"])
def test_operator_preempts_recording_start_failure(
    clock, runtime, tmp_path, camera_snapshot, monkeypatch, key
):
    from test_teleop_recovery import live_keyboard

    from dexmani_real.teleop.loop import TeleopPhase

    loop = make_teleop(runtime, clock)
    loop.phase = TeleopPhase.IDLE
    keyboard = live_keyboard(loop)
    loop.recording_dir = tmp_path
    loop.camera_snapshot = camera_snapshot
    loop.robot.hand.tactile_reset_metadata = {}
    loop.controller.rotation.as_matrix.return_value = np.eye(3)
    monkeypatch.setattr(loop, "start_observation", lambda **kw: observation(clock.now_ns))

    def fail_start(recorder, **kwargs):
        keyboard.handle_key(key, True)
        recorder.store_error(OSError("disk failed during startup"))
        recorder.check_error()

    monkeypatch.setattr(EpisodeRecorder, "start", fail_start)
    loop.run_episode(loop.runtime_state.cancel_epoch)
    assert loop.pending_termination_reason == (
        "estop" if key == "esc" else "quit" if key == "q" else "operator"
    )
    assert loop.discard_requested == (key == "d")
    assert loop.runtime_state.quit_requested == (key == "q")
    assert loop.runtime_state.estop_request == (key == "esc")
    loop.robot.prepare_streaming.assert_not_called()
    loop.robot.send_action.assert_not_called()
    assert loop.finish_recording()


def test_auxiliary_output_io_error_is_visible_but_program_error_propagates(runtime, tmp_path, caplog):
    from dexmani_real.deployment.session import PolicySession

    session = PolicySession(
        runtime, None, weights=None, inference_steps=None, output_root=tmp_path,
        device="cpu", seed=0, max_running_s=None, num_episodes=None,
        record_raw=False, camera_calibration_path=None,
    )
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    session.output_dir = blocked
    session.save_results()
    assert "辅助输出保存失败" in caplog.text
    assert not session.runtime_state.stop_signal.is_set()
    session.output_dir = tmp_path
    session.summary["duration_s"] = float("nan")
    with pytest.raises(ValueError, match="JSON compliant"):
        session.save_results()
    assert not (tmp_path / "session_result.json").exists()
