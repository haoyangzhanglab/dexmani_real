import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dexmani_real.data import recorder as recording
from dexmani_real.data.export import validate_export_episode
from dexmani_real.data.raw import (
    ROBOT_DATASET_SPECS,
    TIMING_ROW_DTYPE,
    require_published_raw,
)
from dexmani_real.runtime.observation import Observation


def test_queue_overflow_returns_reason_and_finish_drains_prefix(monkeypatch, tmp_path):
    entered, release = threading.Event(), threading.Event()

    class Writer:
        def __init__(self, *args, **kwargs):
            self.frame_count = 0
        def open(self, metadata):
            pass
        def append_frame(self, *args):
            entered.set()
            assert release.wait(3)
            self.frame_count += 1
        def finish(self, *args, **kwargs):
            self.reason = kwargs['recording_truncated_reason']
        def close(self):
            pass
        def flush(self):
            pass

    monkeypatch.setattr(recording, 'RawWriter', Writer)
    recorder = recording.EpisodeRecorder(tmp_path, control_hz=16)
    recorder.start(task_label='test', collection_source='teleop', metadata={})
    row = Observation(None, None, None, None, None, 1)
    try:
        recorder.append(row)
        assert entered.wait(3)
        for _ in range(16):
            recorder.append(row)
        assert recorder.append(row) == 'recording_queue_full'
        assert recorder.recording_truncated_reason == 'recording_queue_full'
        assert recorder.queue.qsize() == 16
        assert recorder.error is None
        recorder.append(row)
        assert recorder.queue.qsize() == 16
    finally:
        release.set()
        path = recorder.finish(save=True, reason='operator')
    assert path.is_dir()
    assert not recorder.writer_pending
    assert recorder.written_frames == 17
    assert recorder.writer.reason == 'recording_queue_full'


def reader_with_intervals(intervals, *, termination_reason='operator'):
    control = np.r_[1_000_000_000, 1_000_000_000 + np.cumsum(intervals)]
    rows = np.zeros(len(control), dtype=TIMING_ROW_DTYPE)
    for field in ('control_ns', 'observation_ns', 'arm_ns', 'hand_ns',
                  'tactile_aggregate_ns', 'tactile_dense_ns', 'camera_ns'):
        rows[field] = control
    rows['observation_ns'] += 1_000_000
    rows['camera_sequence'] = np.arange(len(rows))
    data = {name: np.zeros((len(rows), *shape), dtype=dtype)
            for name, (shape, dtype) in ROBOT_DATASET_SPECS.items()}
    data['timing/rows'] = rows
    data['depth'] = np.zeros((len(rows), 2, 2), dtype=np.uint16)

    class Reader:
        path = Path('/tmp/episode_test')
        num_frames = len(rows)
        control_hz = 16
        meta = {'task_label': 'test', 'collection_source': 'teleop', 'termination_reason': termination_reason}
        def require_fields(self, *args):
            pass
        def read_camera(self):
            return SimpleNamespace(color=SimpleNamespace(height=2, width=2, distortion_model='none'),
                                   T_xarm_base_from_color=np.eye(4))
        def validate_tactile_metadata(self):
            return .2
        def __getitem__(self, name):
            return data[name]
    return Reader()


@pytest.mark.parametrize('reason', ['operator', 'failed'])
def test_bounded_drift_export_preserves_termination_reason(reason):
    _, quality = validate_export_episode(reader_with_intervals([66_000_000] * 4, termination_reason=reason))
    assert quality['mean_control_period_ns'] == 66_000_000
    assert quality['termination_reason'] == reason


def test_actual_missing_period_is_reported(caplog):
    _, quality = validate_export_episode(reader_with_intervals([62_500_000, 125_000_000]))
    assert quality['max_control_interval_ns'] == 125_000_000
    assert 'jitter/gaps' in caplog.text


def test_export_rejects_staging_symlink(tmp_path):
    staging = tmp_path / '.tmp_episode'
    staging.mkdir()
    alias = tmp_path / 'episode_alias'
    alias.symlink_to(staging, target_is_directory=True)
    with pytest.raises(ValueError, match='unpublished Raw'):
        require_published_raw(alias)


@pytest.mark.parametrize('failure_stage', ['create', 'rows'])
def test_failed_export_closes_store_and_removes_only_its_staging(monkeypatch, tmp_path, failure_stage):
    from dexmani_real.config.perception import HandCloudConfig, PointCloudConfig
    from dexmani_real.data import export
    source = tmp_path / 'raw'
    episode = source / 'episode_one'
    episode.mkdir(parents=True)
    (episode / 'data.h5').write_bytes(b'raw sentinel')
    target = tmp_path / 'result.zarr'
    camera = SimpleNamespace(color=SimpleNamespace(height=2, width=2))
    class Reader:
        num_frames = 1
        control_hz = 16
        meta = {'task_label': 'test'}
        def __init__(self, path):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    monkeypatch.setattr(export, 'RawReader', Reader)
    monkeypatch.setattr(export, 'validate_export_episode', lambda reader: (camera, {}))
    monkeypatch.setattr(export, 'ArmFK', lambda: None)
    monkeypatch.setattr(export, 'HandFK', lambda: None)
    monkeypatch.setattr(export, 'HandPointCloud', lambda *args: None)
    closed = []
    monkeypatch.setattr(export.zarr, "DirectoryStore", lambda path: SimpleNamespace(close=lambda: closed.append(True)))
    def create_store(*args, **kwargs):
        if failure_stage == 'create':
            raise ValueError('store setup failed')
        return SimpleNamespace(), {}, 1
    monkeypatch.setattr(export, 'create_store', create_store)
    monkeypatch.setattr(export, 'iter_canonical_blocks', lambda *a, **kw: iter(()))
    with pytest.raises(ValueError, match='row count mismatch|store setup failed'):
        export.export_zarr(source, target,
                           processing=export.ExportConfig(PointCloudConfig(), HandCloudConfig()))
    assert closed == [True]
    assert not target.exists()
    assert not list(tmp_path.glob('.result.zarr.tmp-*'))
    assert (episode / 'data.h5').read_bytes() == b'raw sentinel'


@pytest.mark.parametrize('reason', ['operator', 'failed'])
def test_real_raw_container_termination_and_corruption_boundaries(tmp_path, reason):
    import h5py

    from dexmani_real.data.raw import RawReader, RawWriter, snapshot_recording_metadata
    from dexmani_real.geometry.camera import CameraIntrinsics, CameraSnapshot
    from dexmani_real.replay.trajectory import load_trajectory
    from dexmani_real.config.real import RealConfig
    from dexmani_real.robot.commands import RobotCommand
    from dexmani_real.robot.xarm7.protocol import ARM_STATE_DTYPE
    from dexmani_real.robot.xhand.feedback import HAND_STATE_DTYPE

    path = tmp_path / 'episode_real'
    path.mkdir()
    camera = CameraSnapshot('fake', CameraIntrinsics(4, 4, 4., 4., 2., 2., 'none', (0.,) * 5),
                            .001, np.eye(4))
    metadata = snapshot_recording_metadata(camera, tactile_reset_metadata={}, tactile_hold_max_age_s=.2)
    metadata.update(task_label='test', collection_source='teleop')
    writer = RawWriter(path, control_hz=16)
    try:
        writer.open(metadata)
        for i in range(3):
            stamp = 1_000_000_000 + i * 66_000_000
            arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
            hand = np.zeros(1, dtype=HAND_STATE_DTYPE)
            arm['timestamp_ns'] = stamp
            for field in ('timestamp_ns', 'tactile_aggregate_timestamp_ns', 'tactile_dense_timestamp_ns'):
                hand[field] = stamp
            row = Observation(arm, hand, dict(timestamp_ns=stamp, ring_sequence=i,
                              rgb=np.zeros((4, 4, 3), dtype=np.uint8), depth=np.ones((4, 4), dtype=np.uint16)),
                              dict(recv_ts_ns=stamp, sequence_id=i, landmarks=np.zeros((21, 3)), wrist_pos=np.zeros(3),
                                   wrist_quat_wxyz=np.array([1., 0, 0, 0])), None,
                              stamp + 1_000_000, control_timestamp_ns=stamp)
            writer.append_frame(row, RobotCommand(np.zeros(7), np.deg2rad(RealConfig().hand.home_qpos_deg)))
        writer.finish(reason, [])
    finally:
        writer.close()
    with RawReader(path) as reader:
        validate_export_episode(reader)
        assert len(list(reader.iter_rgb_frames())) == 3
    assert len(load_trajectory(path, RealConfig())) == 3
    with h5py.File(path / 'data.h5', 'r+') as data:
        data['action_arm_joint_target'][0, 0] = np.nan
    with pytest.raises(ValueError, match='finite targets'):
        load_trajectory(path, RealConfig())
    staging = path.with_name('.tmp_episode_real')
    path.rename(staging)
    with RawReader(staging) as reader:
        assert reader.num_frames == 3
    with pytest.raises(ValueError, match='finite targets'):
        load_trajectory(staging, RealConfig())


def test_writer_partial_start_failure_is_closed_and_retained(monkeypatch, tmp_path):
    closed = []
    class Writer:
        frame_count = 0
        def __init__(self, *args, **kwargs):
            pass
        def open(self, metadata):
            raise OSError('disk failed')
        def close(self):
            closed.append(True)
    monkeypatch.setattr(recording, 'RawWriter', Writer)
    recorder = recording.EpisodeRecorder(tmp_path, control_hz=16)
    with pytest.raises(recording.RecordingError, match='disk failed'):
        recorder.start(task_label='test', collection_source='teleop', metadata={})
    with pytest.raises(recording.RecordingError, match='disk failed'):
        recorder.finish(save=True, reason='failed')
    assert closed == [True]
    assert not recorder.writer_pending
    assert recorder.staging_path.is_dir()


@pytest.mark.parametrize('images', [1, 3])
def test_rgb_decoded_count_must_match_raw_rows(monkeypatch, images):
    from dexmani_real.config.perception import HandCloudConfig, PointCloudConfig
    from dexmani_real.data import export
    reader = reader_with_intervals([62_500_000])
    reader.iter_rgb_frames = lambda: (np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(images))
    camera = reader.read_camera()
    camera.depth_scale_m = .001
    monkeypatch.setattr(export, 'PointCloudProcessor', lambda **kw: SimpleNamespace(process=lambda *a: np.zeros((4, 6))))
    monkeypatch.setattr(export, 'camera_parameters_cv', lambda *args: {})
    monkeypatch.setattr(export, 'eef_poses', lambda joints, **kwargs: np.zeros((len(joints), 4, 4)))
    monkeypatch.setattr(export, 'encode_eef_pose', lambda poses: np.zeros((len(poses), 9)))
    monkeypatch.setattr(export, 'hand_geometry', lambda *a, **kw: {})
    config = export.ExportConfig(PointCloudConfig(num_points=4), HandCloudConfig())
    with pytest.raises(ValueError, match='RGB missing|unexpected RGB frame'):
        list(export.iter_canonical_blocks(reader, config, arm_fk=None, hand_fk=None,
                                         hand_cloud=None, chunk_frames=2, camera=camera))


def test_writer_timeout_keeps_live_thread_owned(monkeypatch, tmp_path):
    entered, release = threading.Event(), threading.Event()
    class Writer:
        frame_count = 0
        def __init__(self, *args, **kwargs):
            pass
        def open(self, metadata):
            pass
        def append_frame(self, *args):
            entered.set()
            assert release.wait(3)
        def flush(self):
            pass
        def close(self):
            pass
    monkeypatch.setattr(recording, 'RawWriter', Writer)
    monkeypatch.setattr(recording, 'RECORDER_STOP_TIMEOUT_S', .01)
    recorder = recording.EpisodeRecorder(tmp_path, control_hz=16)
    recorder.start(task_label='test', collection_source='teleop', metadata={})
    try:
        recorder.append(Observation(None, None, None, None, None, 1))
        assert entered.wait(3)
        with pytest.raises(recording.RecordingError, match='did not exit'):
            recorder.finish(save=True, reason='operator')
        assert recorder.writer_pending
        assert recorder.result()['status'] == 'pending'
    finally:
        release.set()
        recorder.thread.join(3)
    assert not recorder.writer_pending
    assert recorder.result()['status'] == 'failed'


def test_readout_longer_than_nominal_period_is_allowed_without_overlap():
    reader = reader_with_intervals([66_000_000] * 3)
    rows = reader['timing/rows']
    rows['observation_ns'] = rows['control_ns'] + 64_000_000
    _, quality = validate_export_episode(reader)
    assert quality['max_readout_ns'] == 64_000_000


@pytest.mark.parametrize('invalid', ['overlap', 'source_regression', 'source_future', 'sequence_regression'])
def test_export_rejects_invalid_time_relationships(invalid):
    reader = reader_with_intervals([66_000_000] * 3)
    rows = reader['timing/rows']
    if invalid == 'overlap':
        rows['observation_ns'][0] = rows['control_ns'][1]
    elif invalid == 'source_regression':
        rows['camera_ns'][1] = rows['camera_ns'][0] - 1
    elif invalid == 'source_future':
        rows['camera_ns'][0] = rows['observation_ns'][0] + 1
    else:
        rows['camera_sequence'][2] = 0
    if invalid == 'overlap':
        validate_export_episode(reader)
        return
    with pytest.raises(ValueError, match='regressed|invalid_source_clock'):
        validate_export_episode(reader)


def test_export_discovers_published_alias_under_temporary_workspace(tmp_path):
    from dexmani_real.data.export import discover_episode_dirs
    published = tmp_path / 'episode_published'
    published.mkdir()
    (published / 'data.h5').write_bytes(b'not opened')
    staging = tmp_path / '.tmp_parent'
    staging.mkdir()
    alias = staging / 'episode_alias'
    alias.symlink_to(published, target_is_directory=True)
    assert discover_episode_dirs(alias) == (published.resolve(),)
    assert (published / 'data.h5').read_bytes() == b'not opened'


def test_export_publishes_real_zarr_store_after_all_rows_are_written(monkeypatch, tmp_path):
    from dexmani_real.config.perception import HandCloudConfig, PointCloudConfig
    from dexmani_real.data import export
    source = tmp_path / '.tmp_workspace' / 'episode_one'
    source.mkdir(parents=True)
    (source / 'data.h5').write_bytes(b'raw preserved')
    target = tmp_path / 'result.zarr'
    camera = SimpleNamespace(color=SimpleNamespace(height=2, width=2), depth_scale_m=.001)

    class Reader:
        num_frames = 2
        control_hz = 16
        meta = {'task_label': 'test'}
        def __init__(self, path):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    monkeypatch.setattr(export, 'RawReader', Reader)
    monkeypatch.setattr(export, 'validate_export_episode', lambda reader: (camera, {'termination_reason': 'failed'}))
    monkeypatch.setattr(export, 'ArmFK', lambda: None)
    monkeypatch.setattr(export, 'HandFK', lambda: None)
    monkeypatch.setattr(export, 'HandPointCloud', lambda *args: None)
    config = export.ExportConfig(PointCloudConfig(num_points=4), HandCloudConfig())
    # 几何转换注入固定块，保留实际 Zarr 创建、追加、关闭与原子发布链。
    specs = export.canonical_array_specs(4, 2, 2, config.hand_pointcloud.num_points)
    block = {name: np.ones((2, *shape), dtype=dtype) for name, (shape, dtype) in specs.items()}
    monkeypatch.setattr(export, 'iter_canonical_blocks', lambda *a, **kw: iter((block,)))
    result = export.export_zarr(source.parent, target, processing=config)
    assert result['total_frames'] == 2
    store = export.zarr.DirectoryStore(str(target))
    try:
        root = export.zarr.open_group(store, mode='r')
        assert set(root['data']) == set(specs)
        assert root['data/action'].shape == (2, 19)
        np.testing.assert_array_equal(root['data/action'][:], block['action'])
        assert root.attrs['dt'] == 1 / 16
        assert root.attrs['episode_quality'] == [{'termination_reason': 'failed'}]
        assert root['meta/episode_ends'][:].tolist() == [2]
    finally:
        store.close()
    assert not list(tmp_path.glob('.result.zarr.tmp-*'))
    assert (source / 'data.h5').read_bytes() == b'raw preserved'


def test_export_episode_basename_gate(tmp_path):
    workspace = tmp_path / '.tmp_workspace'
    workspace.mkdir()
    published = workspace / 'episode_good'
    published.mkdir()
    require_published_raw(published)
    alias = workspace / 'episode_alias'
    alias.symlink_to(published, target_is_directory=True)
    require_published_raw(alias)
    staging = workspace / '.tmp_episode'
    staging.mkdir()
    with pytest.raises(ValueError, match='unpublished Raw'):
        require_published_raw(staging)


def test_batch_discovery_allows_temporary_workspace_name(tmp_path):
    from dexmani_real.data.export import discover_episode_dirs
    workspace = tmp_path / '.tmp_workspace'
    episode = workspace / 'episode_good'
    episode.mkdir(parents=True)
    assert discover_episode_dirs(workspace) == (episode,)


def test_zero_row_writer_is_closed_without_publication(monkeypatch, tmp_path):
    closed = []
    class Writer:
        frame_count = 0
        def __init__(self, *args, **kwargs):
            pass
        def open(self, metadata):
            pass
        def finish(self, *args, **kwargs):
            pass
        def close(self):
            closed.append(True)
    monkeypatch.setattr(recording, 'RawWriter', Writer)
    recorder = recording.EpisodeRecorder(tmp_path, control_hz=16)
    recorder.start(task_label='test', collection_source='teleop', metadata={})
    assert recorder.finish(save=True, reason='operator') is None
    assert closed == [True] and not recorder.writer_pending
    assert recorder.written_frames == 0 and recorder.staging_path is None


def test_finish_setup_interrupt_is_not_reported_as_completed(monkeypatch, tmp_path):
    closed = threading.Event()

    class Writer:
        frame_count = 0
        def __init__(self, *args, **kwargs):
            pass
        def open(self, metadata):
            pass
        def flush(self):
            pass
        def close(self):
            closed.set()

    monkeypatch.setattr(recording, 'RawWriter', Writer)
    recorder = recording.EpisodeRecorder(tmp_path, control_hz=16)
    recorder.start(task_label='test', collection_source='teleop', metadata={})

    def interrupted_clock():
        raise KeyboardInterrupt()

    # request 已建立，但尚未把 finish 放进队列；重入不能当作成功或续等待预算。
    with monkeypatch.context() as patch:
        patch.setattr(recording.time, 'monotonic', interrupted_clock)
        with pytest.raises(KeyboardInterrupt):
            recorder.finish(save=True, reason='operator')
        with pytest.raises(recording.RecordingError, match='KeyboardInterrupt'):
            recorder.finish(save=True, reason='operator')
    recorder.thread.join(1)
    assert closed.is_set() and not recorder.writer_pending
    assert recorder.staging_path.is_dir() and recorder.published_path is None
    assert recorder.result()['status'] == 'failed'


@pytest.mark.parametrize('stage', ['write', 'finish', 'close'])
def test_writer_failure_never_publishes_or_retries_finish(monkeypatch, tmp_path, stage):
    closes = []

    class Writer:
        frame_count = 0
        def __init__(self, *args, **kwargs):
            pass
        def open(self, metadata):
            pass
        def append_frame(self, *args):
            if stage == 'write':
                raise OSError('write failed')
            self.frame_count += 1
        def flush(self):
            pass
        def finish(self, *args, **kwargs):
            if stage == 'finish':
                raise OSError('finish failed')
        def close(self):
            closes.append(True)
            if stage == 'close':
                raise OSError('close failed')

    monkeypatch.setattr(recording, 'RawWriter', Writer)
    recorder = recording.EpisodeRecorder(tmp_path, control_hz=16)
    recorder.start(task_label='test', collection_source='teleop', metadata={})
    recorder.append(Observation(None, None, None, None, None, 1))
    for _ in range(2):
        with pytest.raises(recording.RecordingError, match=stage + ' failed'):
            recorder.finish(save=True, reason='operator')
    assert closes == [True]
    assert not recorder.writer_pending
    assert recorder.staging_path.is_dir() and recorder.published_path is None
    assert recorder.result()['status'] == 'failed'


@pytest.mark.parametrize('field', ['control_ns', 'observation_ns'])
@pytest.mark.parametrize('delta', [0, -1])
def test_export_rejects_nonincreasing_primary_clocks(field, delta):
    reader = reader_with_intervals([62_500_000] * 2)
    rows = reader['timing/rows']
    rows[field][1] = rows[field][0] + delta
    with pytest.raises(ValueError, match='clock'):
        validate_export_episode(reader)


def test_old_tactile_sources_export_with_warning(caplog):
    reader = reader_with_intervals([62_500_000] * 2)
    for field in ('tactile_aggregate_ns', 'tactile_dense_ns'):
        reader['timing/rows'][field] = 1
    validate_export_episode(reader)
    assert 'stale source' in caplog.text
