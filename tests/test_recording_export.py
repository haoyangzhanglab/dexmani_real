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


def raw_frames():
    from dexmani_real.config.real import RealConfig
    from dexmani_real.robot.commands import RobotCommand
    from dexmani_real.robot.xarm7.protocol import ARM_STATE_DTYPE
    from dexmani_real.robot.xhand.feedback import HAND_STATE_DTYPE

    for i in range(3):
        stamp = 1_000_000_000 + i * 66_000_000
        arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
        hand = np.zeros(1, dtype=HAND_STATE_DTYPE)
        arm['timestamp_ns'] = stamp
        for field in ('timestamp_ns', 'tactile_aggregate_timestamp_ns', 'tactile_dense_timestamp_ns'):
            hand[field] = stamp
        row = Observation(arm, hand, dict(timestamp_ns=stamp, ring_sequence=i,
                          rgb=np.zeros((16, 16, 3), dtype=np.uint8),
                          depth=np.full((16, 16), 500, dtype=np.uint16)),
                          dict(recv_ts_ns=stamp, sequence_id=i, landmarks=np.zeros((21, 3)),
                               wrist_pos=np.zeros(3), wrist_quat_wxyz=np.array([1., 0, 0, 0])),
                          None, stamp + 1_000_000, control_timestamp_ns=stamp)
        yield row, RobotCommand(np.zeros(7), np.deg2rad(RealConfig().hand.home_qpos_deg))


def raw_metadata(scale=.001):
    from dexmani_real.data.raw import snapshot_recording_metadata
    from dexmani_real.geometry.camera import CameraIntrinsics, CameraSnapshot
    camera = CameraSnapshot('fake', CameraIntrinsics(16, 16, 100., 100., 8., 8., 'none', (0.,) * 5),
                            scale, np.eye(4))
    return dict(snapshot_recording_metadata(camera, tactile_reset_metadata={}),
                task_label='test', collection_source='teleop')


def test_tactile_reset_results_are_json_encoded_at_raw_boundary(tmp_path):
    import json
    from dexmani_real.config.xhand import HandConfig
    from dexmani_real.data.raw import RawReader, decode_camera_metadata, snapshot_recording_metadata
    from dexmani_real.robot.xhand.driver import COMMUNICATION_CRC_ERROR_CODE, XHand

    hand = XHand(HandConfig())
    codes = iter([0, COMMUNICATION_CRC_ERROR_CODE, 0, 0, 0])
    hand.control = SimpleNamespace(reset_sensor=lambda *args: SimpleNamespace(
        error_code=next(codes), error_message=''))
    hand.connected_flag = True
    results = hand.reset_tactile_sensors()
    device_metadata = hand.tactile_reset_metadata
    metadata = snapshot_recording_metadata(
        decode_camera_metadata(raw_metadata()), tactile_reset_metadata=device_metadata,
    )
    assert device_metadata['hand_tactile_reset_result'] is results
    assert isinstance(metadata['hand_tactile_reset_result'], str)
    recorder = recording.EpisodeRecorder(tmp_path, control_hz=16)
    try:
        recorder.start(task_label='test', collection_source='teleop', metadata=metadata)
        for row, command in raw_frames():
            recorder.append(row, command)
        path = recorder.finish(save=True, reason='operator')
    finally:
        if recorder.writer_pending:
            recorder.finish(save=False, reason='test_cleanup')
    with RawReader(path) as reader:
        assert reader.meta['hand_tactile_reset_time_ns'] == device_metadata['hand_tactile_reset_time_ns']
        assert json.loads(reader.meta['hand_tactile_reset_result']) == {str(k): v for k, v in results.items()}
        assert reader.num_frames == 3


@pytest.fixture
def raw_episode(tmp_path):
    from dexmani_real.data.raw import RawWriter

    def make(name='episode_real', *, scale=.001, reason='operator'):
        path = tmp_path / name
        path.mkdir(parents=True)
        writer = RawWriter(path, control_hz=16)
        try:
            writer.open(raw_metadata(scale))
            for row, command in raw_frames():
                writer.append_frame(row, command)
            writer.finish(reason, [])
        finally:
            writer.close()
        return path
    return make


@pytest.mark.parametrize('reason', ['operator', 'failed'])
def test_real_raw_container_termination_and_corruption_boundaries(raw_episode, reason):
    import h5py
    from dexmani_real.data.raw import RawReader
    from dexmani_real.replay.trajectory import load_trajectory
    from dexmani_real.config.real import RealConfig

    path = raw_episode(reason=reason)
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


@pytest.mark.parametrize('field,value,dataset', [
    ('joint_units', 'deg', 'arm_qpos'),
    ('arm_joint_names', '[]', 'arm_qpos'),
    ('hand_joint_names', '[]', 'hand_qpos'),
    ('arm_action_semantics', 'next_step_arrival', 'action_arm_joint_target'),
    ('timestamp_source', 'wall_clock', 'timing/rows'),
])
def test_raw_metadata_is_checked_when_its_data_is_used(raw_episode, field, value, dataset):
    import h5py
    from dexmani_real.data.raw import RawReader
    path = raw_episode()
    with h5py.File(path / 'data.h5', 'r+') as data:
        data['meta'].attrs[field] = value
    with RawReader(path) as reader:
        assert len(list(reader.iter_rgb_frames())) == 3
        assert reader['depth'].shape == (3, 16, 16)
        if field in {'hand_joint_names', 'arm_action_semantics', 'timestamp_source'}:
            assert reader['arm_qpos'].shape == (3, 7)
        with pytest.raises(ValueError, match=field):
            reader[dataset]


@pytest.mark.parametrize('source', ['teleop', 'policy_rollout', 'imported'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_export_accepts_float_storage_and_preserves_collection_source(raw_episode, tmp_path, source, dtype):
    import h5py
    import zarr
    from dexmani_real.data.export import export_zarr
    path = raw_episode()
    with h5py.File(path / 'data.h5', 'r+') as data:
        data['meta'].attrs['collection_source'] = source
        for name in ROBOT_DATASET_SPECS:
            values = data[name][:].astype(dtype)
            del data[name]
            data.create_dataset(name, data=values)
    target = tmp_path / 'export.zarr'
    result = export_zarr(path, target, processing=export_processing())
    assert result['total_frames'] == 3
    assert result['episode_quality'][0]['collection_source'] == source
    with zarr.DirectoryStore(str(target)) as store:
        root = zarr.open_group(store, mode='r')
        for name in ('joint_state', 'action', 'finger_tactile_force', 'taxel_tactile_force'):
            assert root[f'data/{name}'].dtype == np.float32
            assert np.isfinite(root[f'data/{name}'][:]).all()


@pytest.mark.parametrize('damage', ['missing_action', 'tactile_overflow'])
def test_policy_export_still_rejects_invalid_numeric_rows(raw_episode, tmp_path, damage):
    import h5py
    from dexmani_real.data.export import export_zarr
    path = raw_episode()
    with h5py.File(path / 'data.h5', 'r+') as data:
        data['meta'].attrs['collection_source'] = 'policy_rollout'
        if damage == 'missing_action':
            data['action_arm_joint_target'][1, 0] = np.nan
        else:
            values = data['hand_contact'][:].astype(np.float64)
            values[1, 0, 0] = 1e100
            del data['hand_contact']
            data.create_dataset('hand_contact', data=values)
    target = tmp_path / 'bad.zarr'
    with np.errstate(over='ignore'), pytest.raises(ValueError, match='nonfinite.*row 1'):
        export_zarr(path, target, processing=export_processing())
    assert not target.exists()


@pytest.mark.parametrize('close_failure', [False, True])
def test_writer_partial_start_failure_is_closed_and_retained(monkeypatch, tmp_path, caplog, close_failure):
    import logging
    monkeypatch.setattr(logging.getLogger('dexmani_real'), 'handlers', [])
    monkeypatch.setattr(logging.getLogger('dexmani_real'), 'propagate', True)
    closed = []
    cause = ValueError('invalid metadata')
    failure = OSError('disk failed')
    cleanup_failure = OSError('disk failed')
    class Writer:
        frame_count = 0
        def __init__(self, *args, **kwargs):
            pass
        def open(self, metadata):
            raise failure from cause
        def close(self):
            closed.append(True)
            if close_failure:
                raise cleanup_failure
    monkeypatch.setattr(recording, 'RawWriter', Writer)
    recorder = recording.EpisodeRecorder(tmp_path, control_hz=16)
    with pytest.raises(recording.RecordingError, match='disk failed') as caught:
        recorder.start(task_label='test', collection_source='teleop', metadata={})
    assert caught.value.__cause__ is failure
    assert failure.__cause__ is cause
    with pytest.raises(recording.RecordingError, match='disk failed'):
        recorder.finish(save=True, reason='failed')
    assert closed == [True]
    assert not recorder.writer_pending
    assert recorder.staging_path.is_dir()
    assert recorder.result()['status'] == 'failed'
    reports = [record for record in caplog.records if record.exc_info]
    assert len(reports) == int(close_failure)
    if close_failure:
        assert reports[0].getMessage() == 'writer_close also failed'
        assert reports[0].exc_info[1] is cleanup_failure


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


def test_held_tactile_sources_export_with_age_statistics(caplog):
    reader = reader_with_intervals([62_500_000] * 2)
    for field in ('tactile_aggregate_ns', 'tactile_dense_ns'):
        reader['timing/rows'][field] = 1
    _, quality = validate_export_episode(reader)
    for channel in ('aggregate', 'dense'):
        assert quality[f'max_tactile_{channel}_age_ns'] == 1_125_999_999
    assert 'tactile held beyond 0.5 s' in caplog.text


def export_processing():
    from dexmani_real.config.perception import HandCloudConfig, PointCloudConfig
    from dexmani_real.data.export import ExportConfig
    return ExportConfig(PointCloudConfig(remove_table=False, num_points=16,
                        workspace=(-1., -1., 0., 1., 1., 2.)), HandCloudConfig())


@pytest.mark.parametrize('first,second,accepted', [
    (.001, float(np.float32(.001)), True),
    (float(np.float32(.001)), .001, True),
    (.001, .01, False),
])
def test_real_export_depth_scale_merge(raw_episode, tmp_path, first, second, accepted):
    import zarr
    from dexmani_real.data.export import export_zarr
    from dexmani_real.data.raw import RawReader
    paths = [raw_episode('raw/episode_001', scale=first),
             raw_episode('raw/episode_002', scale=second)]
    files = [p / name for p in paths for name in ('data.h5', 'rgb.mp4')]
    before = [p.read_bytes() for p in files]
    target = tmp_path / 'merged.zarr'
    if accepted:
        result = export_zarr(paths[0].parent, target, processing=export_processing())
        assert result['total_frames'] == 6
        with zarr.DirectoryStore(str(target)) as store:
            root = zarr.open_group(store, mode='r')
            assert root.attrs['depth_scale_m_per_unit'] == first
            assert root['meta/episode_ends'][:].tolist() == [3, 6]
            assert len(root['data']) == 13
            np.testing.assert_array_equal(root['data/depth'][:], np.full((6, 16, 16), 500))
        for path, scale in zip(paths, (first, second)):
            with RawReader(path) as reader:
                assert reader.meta['depth_scale'] == scale
    else:
        with pytest.raises(ValueError, match='episode_002.*depth_scale_m_per_unit'):
            export_zarr(paths[0].parent, target, processing=export_processing())
        assert not target.exists()
    assert not list(tmp_path.glob('.merged.zarr.tmp-*'))
    assert before == [p.read_bytes() for p in files]


@pytest.mark.parametrize('field,value', [
    ('hand_tactile_units', 'newton'),
    ('hand_tactile_finger_order', 'pinky,ring,middle,index,thumb'),
])
def test_real_tactile_semantics_still_rejected(raw_episode, tmp_path, field, value):
    import h5py
    from dexmani_real.data.raw import RawReader
    from dexmani_real.data.export import export_zarr
    path = raw_episode()
    with h5py.File(path / 'data.h5', 'r+') as data:
        data['meta'].attrs[field] = value
    with RawReader(path) as reader, pytest.raises(ValueError, match=field):
        reader['hand_contact']
    with pytest.raises(ValueError, match=field):
        export_zarr(path, tmp_path / 'bad.zarr', processing=export_processing())


@pytest.mark.parametrize('channel', ['aggregate', 'dense'])
@pytest.mark.parametrize('damage', ['zero', 'negative', 'future', 'regression', 'shape', 'nan', 'missing'])
def test_real_tactile_damage_still_rejected(raw_episode, tmp_path, channel, damage):
    import h5py
    from dexmani_real.data.export import export_zarr
    path = raw_episode()
    with h5py.File(path / 'data.h5', 'r+') as data:
        name = 'hand_contact' if channel == 'aggregate' else 'hand_tactile_force'
        if damage == 'shape':
            shape = list(data[name].shape)
            shape[-1] = 2
            del data[name]
            data.create_dataset(name, shape=shape, dtype=np.float32)
        elif damage == 'nan':
            data[name][1] = np.nan
        elif damage == 'missing':
            del data[name]
        else:
            rows = data['timing/rows'][:]
            name = f'tactile_{channel}_ns'
            rows[name][1] = dict(zero=0, negative=-1,
                                 future=rows['observation_ns'][1] + 1,
                                 regression=rows[name][0] - 1)[damage]
            data['timing/rows'][:] = rows
    with pytest.raises(ValueError, match='invalid_source_clock|source_clock_regressed|shape/dtype|nonfinite|missing fields'):
        export_zarr(path, tmp_path / 'bad.zarr', processing=export_processing())
    assert not (tmp_path / 'bad.zarr').exists()


@pytest.fixture
def teleop_recording(tmp_path):
    from dexmani_real.config.real import RealConfig
    from dexmani_real.runtime.state import SessionState
    from dexmani_real.teleop.loop import TeleopLoop, TeleopPhase

    sounds = []
    loop = TeleopLoop(SessionState(stop_signal=threading.Event(), emergency_signal=threading.Event()),
                     RealConfig(), SimpleNamespace(stop=lambda: None), task_label='test', sensors=None,
                     controller=SimpleNamespace(clear=lambda: None), recording_dir=tmp_path, max_duration_s=60,
                     runtime_config_json=None, keyboard=None, audio=SimpleNamespace(play=sounds.append))
    recorder = loop.recorder = recording.EpisodeRecorder(tmp_path, control_hz=16)
    try:
        recorder.start(task_label='test', collection_source='teleop', metadata=raw_metadata())
        loop.phase = TeleopPhase.ACTIVE
        yield loop, recorder, sounds
    finally:
        # 断言失败也要结束非 daemon writer，避免测试进程一直等候。
        if recorder.writer_pending:
            recorder.finish(save=False, reason='test_cleanup')


@pytest.mark.parametrize('runtime_config_json', [None, '{"collection": {"enabled": true}, "note": "实验"}'])
def test_teleop_start_archives_optional_config_in_real_hdf5(monkeypatch, tmp_path, runtime_config_json):
    import h5py
    from scipy.spatial.transform import Rotation
    from dexmani_real.config.real import RealConfig
    from dexmani_real.data.raw import decode_camera_metadata
    from dexmani_real.runtime.state import RunEndReason, SessionState
    from dexmani_real.teleop import loop as teleop

    metadata = raw_metadata()
    del metadata['camera_T_xarm_base_from_color']
    loop = teleop.TeleopLoop(
        SessionState(), RealConfig(),
        SimpleNamespace(prepare_streaming=lambda epoch: None, service_feedback=lambda: None,
                        stop=lambda: None,
                        hand=SimpleNamespace(tactile_reset_metadata={})),
        task_label='test', sensors=SimpleNamespace(channels=None),
        controller=SimpleNamespace(rotation=Rotation.identity(), hand_retargeter=None),
        recording_dir=tmp_path, max_duration_s=60, runtime_config_json=runtime_config_json,
        keyboard=None, audio=SimpleNamespace(play=lambda sound: None),
        camera_snapshot=decode_camera_metadata(metadata),
    )
    frames = list(raw_frames())
    def observe(*args, **kwargs):
        from dataclasses import replace
        row = frames[0][0]
        arm, hand = row.arm.copy(), row.hand.copy()
        arm['timestamp_ns'] = hand['timestamp_ns'] = teleop.time.monotonic_ns()
        return replace(row, arm=arm, hand=hand)
    monkeypatch.setattr(teleop, 'read_observation', observe)
    monkeypatch.setattr(loop, 'preparation_interrupted', lambda: False)
    monkeypatch.setattr(loop, 'admit_observation', lambda row: [])
    monkeypatch.setattr(loop, 'recording_issue', lambda row: None)
    monkeypatch.setattr(loop, 'try_begin', lambda row: None)
    try:
        loop.start_episode(loop.runtime_state.cancel_epoch)
        recorder = loop.recorder
        assert recorder.is_recording
        for row, command in frames:
            recorder.append(row, command)
        loop.request_episode_end('stop', run_end_reason=RunEndReason.OPERATOR)
        loop.finish_recording()
        with h5py.File(recorder.published_path / 'data.h5', 'r') as data:
            attrs = data['meta'].attrs
            assert ('runtime_config_json' in attrs) == (runtime_config_json is not None)
            if runtime_config_json is not None:
                assert attrs['runtime_config_json'] == runtime_config_json
            assert 'camera_T_xarm_base_from_color' not in attrs
            np.testing.assert_array_equal(attrs['vr_to_robot_rotation'], np.eye(3))
            assert attrs['termination_reason'] == 'operator'
            assert data['depth'].shape[0] == len(frames)
    finally:
        if loop.recorder is not None:
            loop.recorder.finish(save=False, reason='test_cleanup')


@pytest.mark.parametrize('case', ['save', 'discard', 'empty', 'truncated', 'cancelled', 'estop'])
def test_teleop_real_recording_finish_sounds(teleop_recording, case):
    from dexmani_real.data.raw import RawReader
    from dexmani_real.runtime.state import RunEndReason

    loop, recorder, sounds = teleop_recording
    if case != 'empty':
        for row, command in raw_frames():
            recorder.append(row, command)
    loop.request_episode_end('stop', run_end_reason=(
        RunEndReason.CANCELLED if case == 'cancelled' else RunEndReason.OPERATOR
    ))
    loop.discard_requested = case == 'discard'
    if case == 'truncated':
        recorder.append(row, command, quality_issue='recording_no_command')
    elif case == 'estop':
        loop.runtime_state.request_emergency_stop()
    loop.end_episode()
    loop.finish_recording()
    assert loop.recorder is None and not recorder.writer_pending
    if case in {'empty', 'discard'}:
        assert recorder.published_path is None and recorder.staging_path is None
    else:
        with RawReader(recorder.published_path) as reader:
            assert reader.num_frames == 3
            validate_export_episode(reader)
    assert ('save' in sounds) == (case == 'save')
    assert ('discard' in sounds) == (case == 'discard')
    assert ('emergency' in sounds) == (case == 'estop')


@pytest.mark.parametrize('error_type', [ValueError, KeyboardInterrupt])
def test_teleop_episode_error_propagates_before_prefix_publish(teleop_recording, monkeypatch, error_type):
    from dexmani_real.data.raw import RawReader
    from dexmani_real.teleop.loop import TeleopPhase

    loop, recorder, sounds = teleop_recording
    for row, command in raw_frames():
        recorder.append(row, command)
    failure = error_type('mapping failed')
    def stop():
        assert loop.runtime_state.estop_request == (error_type is KeyboardInterrupt)
    loop.robot.stop = stop
    def fail_tick():
        raise failure
    monkeypatch.setattr(loop, 'start_episode', lambda epoch: None)
    monkeypatch.setattr(loop, 'service_episode', lambda timeout: True)
    monkeypatch.setattr(loop, 'run_control_tick', fail_tick)
    with pytest.raises(error_type, match='mapping failed') as caught:
        loop.run_episode(loop.runtime_state.cancel_epoch)
    assert caught.value is failure
    assert loop.phase is TeleopPhase.IDLE and loop.recorder is recorder
    assert recorder.published_path is None and recorder.writer_pending
    assert not any('exception_type' in detail for detail in loop.termination_details)
    # 模拟 run_teleop 的主错误登记边界，再执行会话录制收尾。
    loop.record_error('session', caught.value)
    loop.finish_recording()
    assert loop.recorder is None and not recorder.writer_pending
    with RawReader(recorder.published_path) as reader:
        assert reader.num_frames == 3 and reader.meta['termination_reason'] == 'failed'
        validate_export_episode(reader)
    assert sum(detail.get('message') == 'mapping failed' for detail in loop.termination_details) == 1
    assert 'emergency' in sounds and 'save' not in sounds and 'discard' not in sounds


def test_teleop_stop_error_defers_prefix_publish(teleop_recording):
    from dexmani_real.data.raw import RawReader
    from dexmani_real.runtime.state import RunEndReason

    loop, recorder, sounds = teleop_recording
    for row, command in raw_frames():
        recorder.append(row, command)
    loop.request_episode_end('stop', run_end_reason=RunEndReason.OPERATOR)
    failure = OSError('stop failed')
    def fail_stop():
        raise failure
    loop.robot.stop = fail_stop
    with pytest.raises(OSError, match='stop failed') as caught:
        loop.end_episode()
    assert caught.value is failure
    assert loop.recorder is recorder and recorder.writer_pending
    loop.record_error('session', caught.value)
    loop.finish_recording()
    with RawReader(recorder.published_path) as reader:
        assert reader.num_frames == 3 and reader.meta['termination_reason'] == 'failed'
        validate_export_episode(reader)
    assert 'emergency' in sounds and 'save' not in sounds and 'discard' not in sounds


def test_teleop_writer_error_retains_staging(teleop_recording, monkeypatch):
    from dexmani_real.runtime.state import RunEndReason

    loop, recorder, sounds = teleop_recording
    for row, command in raw_frames():
        recorder.append(row, command)
    loop.request_episode_end('stop', run_end_reason=RunEndReason.OPERATOR)
    close = recorder.writer.close
    def fail_close():
        close()
        raise OSError('writer close failed')
    monkeypatch.setattr(recorder.writer, 'close', fail_close)
    with pytest.raises(recording.RecordingError, match='writer close failed') as caught:
        loop.end_episode()
    loop.record_error('session', caught.value)
    assert loop.recorder is recorder and not recorder.writer_pending
    assert recorder.published_path is None and recorder.staging_path.is_dir()
    with pytest.raises(recording.RecordingError, match='writer close failed'):
        loop.finish_recording()
    assert 'emergency' in sounds and 'save' not in sounds and 'discard' not in sounds


def test_feedback_holds_flow_through_real_recorder_to_full_zarr(tmp_path, caplog):
    from dataclasses import replace
    import zarr
    from dexmani_real.data.export import canonical_array_specs, export_zarr
    from dexmani_real.data.raw import RawReader
    from dexmani_real.robot.commands import RobotCommand
    from dexmani_real.robot.xhand.feedback import XHandFeedback, XHandState

    feedback = XHandFeedback()
    recorder = recording.EpisodeRecorder(tmp_path / 'raw', control_hz=4)
    recorder.start(task_label='test', collection_source='teleop', metadata=raw_metadata())
    template, _ = next(raw_frames())
    expected_aggregate, expected_dense, expected_arm, expected_actions = [], [], [], []
    expected_times = []
    # 两通道分别丢帧；关节继续更新，最后 dense 恢复，aggregate 继续保持。
    channel_values = [(2, 3), (None, 4), (5, None), (None, None), (None, None), (None, 6)]
    try:
        for i, (aggregate, dense) in enumerate(channel_values):
            stamp = 1_000_000_000 + i * 250_000_000
            sample = XHandState(
                qpos=np.full(12, i / 100),
                tactile_aggregate=None if aggregate is None else np.full((5, 3), aggregate, dtype=np.float32),
                tactile_dense=None if dense is None else np.full((5, 120, 3), dense, dtype=np.float32),
                timestamp_ns=stamp,
            )
            feedback.consume(sample)
            hand = feedback.snapshot()
            arm = template.arm.copy()
            arm['timestamp_ns'] = stamp
            arm['qpos'] = i / 100
            row = replace(template, arm=arm, hand=hand,
                          camera=dict(template.camera, timestamp_ns=stamp, ring_sequence=i),
                          observation_timestamp_ns=stamp + 1_000_000, control_timestamp_ns=stamp)
            command = RobotCommand(np.full(7, i / 50), np.full(12, i / 50))
            assert recorder.append(row, command) is None
            expected_arm.append(arm['qpos'][0].copy())
            expected_actions.append(np.full(19, i / 50))
            expected_aggregate.append(hand['tactile_aggregate'][0].copy())
            expected_dense.append(hand['tactile_dense'][0].copy())
            expected_times.append((int(hand['tactile_aggregate_timestamp_ns'][0]),
                                   int(hand['tactile_dense_timestamp_ns'][0])))
        path = recorder.finish(save=True, reason='operator')
    finally:
        if recorder.writer_pending:
            recorder.finish(save=False, reason='test_cleanup')
    with RawReader(path) as reader:
        assert reader.meta['hand_tactile_hold_policy'] == 'last_valid_until_recovery'
        assert len(list(reader.iter_rgb_frames())) == len(channel_values)
        rows = reader['timing/rows'][:]
        np.testing.assert_array_equal(rows['tactile_aggregate_ns'], np.asarray(expected_times)[:, 0])
        np.testing.assert_array_equal(rows['tactile_dense_ns'], np.asarray(expected_times)[:, 1])
        np.testing.assert_array_equal(reader['hand_contact'][:], expected_aggregate)
        np.testing.assert_array_equal(reader['hand_tactile_force'][:], expected_dense)
    target = tmp_path / 'held.zarr'
    processing = export_processing()
    result = export_zarr(path, target, processing=processing)
    assert result['total_frames'] == len(channel_values)
    quality = result['episode_quality'][0]
    assert quality['max_tactile_aggregate_age_ns'] == 751_000_000
    assert quality['max_tactile_dense_age_ns'] == 751_000_000
    with zarr.DirectoryStore(str(target)) as store:
        root = zarr.open_group(store, mode='r')
        arrays = root['data']
        specs = canonical_array_specs(processing.pointcloud.num_points, 16, 16,
                                      processing.hand_pointcloud.num_points)
        assert set(arrays) == set(specs) and len(arrays) == 13
        for name, (shape, dtype) in specs.items():
            assert arrays[name].shape == (len(channel_values), *shape)
            assert arrays[name].dtype == dtype
        np.testing.assert_array_equal(arrays['joint_state'][:, :7], np.asarray(expected_arm, dtype=np.float32))
        np.testing.assert_array_equal(arrays['action'][:], np.asarray(expected_actions, dtype=np.float32))
        np.testing.assert_array_equal(arrays['finger_tactile_force'][:], expected_aggregate)
        np.testing.assert_array_equal(arrays['taxel_tactile_force'][:], expected_dense)
        assert root['meta/episode_ends'][:].tolist() == [len(channel_values)]
        assert root.attrs['dt'] == .25
    assert '继续记录最近有效值及原来源时间' in caplog.text
    assert 'tactile held beyond 0.5 s' in caplog.text


@pytest.mark.parametrize('kind', ['teleop', 'policy'])
def test_recording_hold_keeps_control_input_age_checks(kind):
    from dataclasses import replace
    from dexmani_real.config.real import RealConfig
    from dexmani_real.runtime.observation import ObservationNeeds, check_observation
    from dexmani_real.teleop.loop import TeleopLoop
    from dexmani_real.deployment.rollout import Rollout

    runtime = RealConfig()
    row, _ = next(raw_frames())
    hand = row.hand.copy()
    hand['tactile_aggregate_timestamp_ns'] = 1
    hand['tactile_dense_timestamp_ns'] = 2
    hand['tactile_aggregate'] = 7
    hand['tactile_dense'] = 8
    row = replace(row, hand=hand)
    needs = ObservationNeeds(camera_fields=('rgb', 'depth'), required_tactile=('aggregate', 'dense'))
    owner = SimpleNamespace(sensors=SimpleNamespace(check=lambda **kw: True), runtime=runtime,
                            record_needs=needs)
    loop_class = TeleopLoop if kind == 'teleop' else Rollout
    assert loop_class.recording_issue(owner, row) is None
    for channel in ('aggregate', 'dense'):
        issues = check_observation(row, runtime, ObservationNeeds(required_tactile=(channel,)),
                                   now_ns=row.observation_timestamp_ns)
        assert issues == [f'hand_tactile_{channel}_timestamp_ns_unavailable_or_stale']


@pytest.mark.parametrize('kind', ['teleop', 'policy'])
def test_missing_dense_prevents_writer_and_allows_later_start(monkeypatch, tmp_path, kind):
    from dataclasses import replace
    from scipy.spatial.transform import Rotation
    from dexmani_real.config.real import RealConfig
    from dexmani_real.data.raw import decode_camera_metadata
    from dexmani_real.runtime.observation import ObservationNeeds
    from dexmani_real.runtime.state import SessionState
    from dexmani_real.teleop import loop as teleop
    from dexmani_real.deployment import rollout as policy

    state = SessionState()
    runtime = RealConfig()
    full, _ = next(raw_frames())
    missing = full.hand.copy()
    missing['tactile_dense'] = np.nan
    missing['tactile_dense_timestamp_ns'] = 0
    row = [replace(full, hand=missing)]
    clock = [full.observation_timestamp_ns]
    monkeypatch.setattr(teleop.time, 'monotonic_ns', lambda: clock[0])
    monkeypatch.setattr(teleop.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0] + 1_000_000_000))
    robot = SimpleNamespace(prepare_streaming=lambda epoch: state.stop_signal.clear(),
                            service_feedback=lambda: None, stop=lambda: state.stop_signal.set(),
                            hand=SimpleNamespace(tactile_reset_metadata={}))
    sensors = SimpleNamespace(channels=None, check=lambda **kw: True)
    needs = ObservationNeeds(camera_fields=('rgb', 'depth'), required_tactile=('aggregate', 'dense'))
    polls = []
    if kind == 'teleop':
        owner = teleop.TeleopLoop(
            state, runtime, robot, task_label='test', sensors=sensors,
            controller=SimpleNamespace(rotation=Rotation.identity(), hand_retargeter=None),
            recording_dir=tmp_path, max_duration_s=60, runtime_config_json=None,
            keyboard=None, audio=SimpleNamespace(play=lambda sound: None),
            camera_snapshot=decode_camera_metadata(raw_metadata()),
        )
        monkeypatch.setattr(teleop, 'read_observation', lambda *a, **kw: row[0])
        monkeypatch.setattr(owner, 'preparation_interrupted', lambda: polls.append(True) or False)
        monkeypatch.setattr(owner, 'admit_observation', lambda row: [])
        monkeypatch.setattr(owner, 'try_begin', lambda row: None)
        start = lambda: owner.start_episode(state.cancel_epoch)
    else:
        owner = policy.Rollout(
            state, runtime, SimpleNamespace(control_dt_s=.0625, n_obs_steps=1, task_name='test'),
            cancel_epoch=state.cancel_epoch, robot=robot, observations=SimpleNamespace(needs=ObservationNeeds()),
            record_needs=needs, sensors=sensors,
            action_mapper=None, model=None, max_running_s=None, index=1, output_dir=tmp_path,
            record_raw=True,
            camera_snapshot=decode_camera_metadata(raw_metadata()),
        )
        owner.read_observation = lambda: (row[0], None)
        start = lambda: owner.prepare(lambda: polls.append(True))
    start()
    assert owner.recorder is None and not list(tmp_path.iterdir())
    assert not state.quit_requested and not state.estop_request and len(polls) >= 2
    # 模拟结束准备回待命；新 B 接纳后，不需重新连接或重新加载模型。
    if kind == 'policy':
        owner.result.termination_reason = None
        owner.preparation_issue = None
    row[0] = full
    clock[0] = full.observation_timestamp_ns
    try:
        start()
        assert owner.recorder is not None and owner.recorder.is_recording
    finally:
        if owner.recorder is not None:
            owner.recorder.finish(save=False, reason='test_cleanup')


@pytest.mark.parametrize('kind', ['teleop', 'policy'])
def test_recording_old_tactile_still_submits_and_writes_every_tick(monkeypatch, tmp_path, kind):
    from dataclasses import replace
    from dexmani_real.config.real import RealConfig
    from dexmani_real.data.raw import RawReader
    from dexmani_real.deployment.rollout import Rollout
    from dexmani_real.robot.commands import RobotCommand
    from dexmani_real.runtime.observation import ObservationNeeds
    from dexmani_real.runtime.state import SessionState
    from dexmani_real.teleop import loop as teleop

    clock = [1_000_000_000]
    monkeypatch.setattr(teleop.time, 'monotonic_ns', lambda: clock[0])
    state, runtime, sends = SessionState(), RealConfig(), []
    robot = SimpleNamespace(send_action=lambda command, **kw: sends.append(command), check=lambda: None)
    sensors = SimpleNamespace(channels=None, check=lambda **kw: True)
    audio = SimpleNamespace(play=lambda sound: None)
    recorder = recording.EpisodeRecorder(tmp_path, control_hz=16)
    source = 'teleop' if kind == 'teleop' else 'policy_rollout'
    recorder.start(task_label='test', collection_source=source, metadata=raw_metadata())
    current = [None]
    if kind == 'teleop':
        controller = SimpleNamespace(
            compute_target=lambda row, **kw: SimpleNamespace(
                arm_qpos=np.zeros(7), hand_qpos=np.zeros(12), rejection_reason=None, eef_pose=None),
            commit=lambda *a, **kw: None,
        )
        owner = teleop.TeleopLoop(state, runtime, robot, task_label='test', sensors=sensors,
                                  controller=controller, recording_dir=tmp_path, max_duration_s=60,
                                  runtime_config_json=None, keyboard=None, audio=audio)
        owner.capture_deadline_ns = clock[0] + 10_000_000_000
        owner.first_target_pending = True
        owner.cancel_epoch = state.cancel_epoch
        monkeypatch.setattr(teleop, 'read_observation', lambda *a, **kw: current[0])
        tick = owner.run_control_tick
    else:
        needs = ObservationNeeds(camera_fields=('rgb', 'depth'), required_tactile=('aggregate', 'dense'))
        owner = Rollout(state, runtime, SimpleNamespace(control_dt_s=.0625, n_obs_steps=1),
                        cancel_epoch=state.cancel_epoch, robot=robot, observations=SimpleNamespace(project=lambda row: {}, needs=ObservationNeeds()),
                        record_needs=needs, sensors=sensors,
                        action_mapper=None, model=SimpleNamespace(poll=lambda: None, future=None),
                        max_running_s=None, index=1, output_dir=tmp_path, record_raw=True)
        owner.execution = replace(runtime.execution, action_steps=4)
        owner.started_ns = owner.next_step_ns = clock[0]
        owner.sync_actions = np.zeros((4, 19))
        owner.read_observation = lambda: (current[0], None)
        owner.map_command = lambda *args: (RobotCommand(np.zeros(7), np.zeros(12)), None)
        tick = owner.step
    owner.recorder = recorder
    try:
        for row, _ in raw_frames():
            hand = row.hand.copy()
            hand['tactile_aggregate_timestamp_ns'] = 1
            hand['tactile_dense_timestamp_ns'] = 2
            hand['tactile_aggregate'] = 7
            hand['tactile_dense'] = 8
            current[0] = replace(row, hand=hand)
            clock[0] = row.observation_timestamp_ns
            tick()
        assert len(sends) == 3 and not state.stop_signal.is_set()
        assert recorder.recording_truncated_reason is None
        path = recorder.finish(save=True, reason='operator')
        with RawReader(path) as reader:
            assert reader.num_frames == 3
            np.testing.assert_array_equal(reader['timing/rows'][:]['tactile_aggregate_ns'], [1] * 3)
            np.testing.assert_array_equal(reader['timing/rows'][:]['tactile_dense_ns'], [2] * 3)
            assert np.all(reader['hand_contact'][:] == 7)
            assert np.all(reader['hand_tactile_force'][:] == 8)
    finally:
        if recorder.writer_pending:
            recorder.finish(save=False, reason='test_cleanup')


def test_unrecorded_tactile_policy_waits_for_initial_value_after_reset(monkeypatch):
    from dataclasses import replace
    from dexmani_real.config.real import RealConfig
    from dexmani_real.deployment import rollout as policy
    from dexmani_real.runtime.observation import ObservationNeeds
    from dexmani_real.runtime.state import SessionState

    full, _ = next(raw_frames())
    hand = full.hand.copy()
    hand['tactile_dense_timestamp_ns'] = 0
    hand['tactile_dense'] = np.nan
    samples = iter(((replace(full, hand=hand), 'model_input_dense_unavailable'), (full, None)))
    clock = [full.observation_timestamp_ns]
    monkeypatch.setattr(policy.time, 'monotonic_ns', lambda: clock[0])
    monkeypatch.setattr(policy.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0] + 5_000_000))
    state = SessionState()
    owner = policy.Rollout(
        state, RealConfig(), SimpleNamespace(control_dt_s=.0625, n_obs_steps=1),
        cancel_epoch=state.cancel_epoch, robot=SimpleNamespace(service_feedback=lambda: None),
        observations=SimpleNamespace(needs=ObservationNeeds(required_tactile=('dense',))),
        record_needs=ObservationNeeds(), sensors=None, action_mapper=None, model=None,
        max_running_s=None, index=1, output_dir=None, record_raw=False,
    )
    owner.read_observation = lambda: next(samples)
    polls = []
    assert owner.start_observation(lambda: polls.append(True)) is full
    assert len(polls) == 2 and owner.recorder is None and owner.preparation_issue is None


@pytest.mark.parametrize('kind', ['teleop', 'policy'])
@pytest.mark.parametrize('outcome', ['recover', 'stop', 'quit', 'estop', 'writer_failure',
                                   'camera_failure', 'no_feedback'])
@pytest.mark.parametrize('writer_delay_ns', [10_000_000, 600_000_000])
def test_preparation_keeps_robot_stopped_through_writer_and_refreshes_inputs(
    monkeypatch, tmp_path, kind, outcome, writer_delay_ns,
):
    from dataclasses import replace
    from scipy.spatial.transform import Rotation
    from dexmani_real.config.real import RealConfig
    from dexmani_real.data.raw import decode_camera_metadata
    from dexmani_real.deployment import rollout as policy
    from dexmani_real.robot.commands import OperationCancelled
    from dexmani_real.robot.system import Robot
    from dexmani_real.robot.xhand.driver import COMMUNICATION_CRC_ERROR_CODE, XHand
    from dexmani_real.runtime.observation import ObservationNeeds
    from dexmani_real.runtime.state import SessionState
    from dexmani_real.teleop import loop as teleop

    full, _ = next(raw_frames())
    clock = [full.observation_timestamp_ns]
    monkeypatch.setattr(teleop.time, 'monotonic_ns', lambda: clock[0])
    monkeypatch.setattr(teleop.time, 'sleep', lambda seconds: clock.__setitem__(
        0, clock[0] + int(seconds * 1e9)))
    state, runtime = SessionState(), RealConfig()
    events, anchors, submitted = [], [], []
    writer_started = [False]
    dropped = [False]
    robot = Robot(runtime.arm, runtime.hand, runtime_state=state)
    robot.connected = robot.hand_stop_required = True

    def arm_snapshot():
        frame = full.arm.copy()
        frame['timestamp_ns'] = clock[0]
        return frame

    def prepare(epoch, *, service):
        assert writer_started[0] and not submitted
        assert not state.estop_request and not state.quit_requested
        events.append('streaming')
        state.stop_signal.clear()
        service()

    robot.arm = SimpleNamespace(
        check_error=lambda: None, snapshot=arm_snapshot,
        request_stop=lambda: events.append('stop'), wait_stopped=lambda: None,
        prepare=prepare, submit=lambda *a, **kw: submitted.append(a),
    )

    def read_sdk_state(*args):
        if writer_started[0]:
            events.append(('sample', state.stop_signal.is_set()))
            if not dropped[0] or outcome == 'no_feedback':
                dropped[0] = True
                return SimpleNamespace(error_code=COMMUNICATION_CRC_ERROR_CODE), None
        force = SimpleNamespace(fx=1., fy=1., fz=1.)
        payload = SimpleNamespace(
            finger_state=[SimpleNamespace(id=i, position=.2 if writer_started[0] else .1,
                                           commboard_err=0, jonitboard_err=0, tipboard_err=0)
                          for i in range(12)],
            sensor_data=[SimpleNamespace(calc_force=force, raw_force=[force] * 120)
                         for _ in range(5)],
        )
        return SimpleNamespace(error_code=0, error_message=''), payload

    robot.hand = XHand(runtime.hand)
    robot.hand.control = SimpleNamespace(read_state=read_sdk_state)
    robot.hand.connected_flag = True
    robot.hand.stop = lambda *a, **kw: None
    robot.hand._send_action = lambda *a: submitted.append(a)
    robot.hand_feedback.consume(robot.hand.get_state())
    sensors = SimpleNamespace(
        channels=None, check=lambda **kw: not (writer_started[0] and outcome == 'camera_failure'),
    )

    def observe(*args, **kwargs):
        latest = robot.read_state()
        return replace(full, arm=latest.arm, hand=latest.hand,
                       camera=dict(full.camera, timestamp_ns=clock[0]),
                       vr=dict(full.vr, recv_ts_ns=clock[0]),
                       observation_timestamp_ns=clock[0], control_timestamp_ns=clock[0])

    monkeypatch.setattr(teleop, 'read_observation', observe)
    monkeypatch.setattr(policy, 'read_observation', observe)
    open_writer = recording.RawWriter.open

    def delayed_open(writer, metadata):
        assert state.stop_signal.is_set() and not robot.hand_stop_required
        assert 'streaming' not in events and not submitted
        writer_started[0] = True
        events.append('writer')
        clock[0] += writer_delay_ns  # include a wait exceeding the unchanged 0.5 s active TTL
        if outcome == 'writer_failure':
            raise OSError('writer setup failed')
        open_writer(writer, metadata)
        if outcome in ('stop', 'quit', 'estop'):
            getattr(state, {'stop': 'request_stop', 'quit': 'request_quit',
                            'estop': 'request_emergency_stop'}[outcome])()

    monkeypatch.setattr(recording.RawWriter, 'open', delayed_open)
    audio = SimpleNamespace(play=lambda sound: None)
    if kind == 'teleop':
        owner = teleop.TeleopLoop(
            state, runtime, robot, task_label='test', sensors=sensors,
            controller=SimpleNamespace(rotation=Rotation.identity(), hand_retargeter=None,
                                       reset=lambda row: anchors.append(row), clear=lambda: None),
            recording_dir=tmp_path, max_duration_s=60, runtime_config_json=None,
            keyboard=None, audio=audio, camera_snapshot=decode_camera_metadata(raw_metadata()),
        )
        owner.poll_blocking_commands = lambda: False
        start = lambda: owner.start_episode(state.cancel_epoch)
    else:
        owner = policy.Rollout(
            state, runtime, SimpleNamespace(control_dt_s=.0625, n_obs_steps=1, task_name='test'),
            cancel_epoch=state.cancel_epoch, robot=robot, observations=SimpleNamespace(needs=ObservationNeeds()),
            record_needs=ObservationNeeds(camera_fields=('rgb', 'depth'),
                                           required_tactile=('aggregate', 'dense')),
            sensors=sensors, action_mapper=None, model=None, max_running_s=None,
            index=1, output_dir=tmp_path, record_raw=True,
            camera_snapshot=decode_camera_metadata(raw_metadata()),
        )
        start = lambda: owner.prepare(lambda: None)
    try:
        if outcome == 'writer_failure':
            with pytest.raises(recording.RecordingError, match='writer setup failed'):
                start()
        elif outcome == 'camera_failure':
            with pytest.raises(RuntimeError, match='recording camera worker exited'):
                start()
        elif kind == 'policy' and outcome in ('stop', 'quit', 'estop'):
            with pytest.raises(OperationCancelled):
                start()
        else:
            start()
        assert not submitted
        active = owner.phase is teleop.TeleopPhase.ACTIVE if kind == 'teleop' else bool(owner.started_ns)
        assert active == (outcome == 'recover')
        if outcome == 'recover':
            streaming = events.index('streaming')
            assert events.index('writer') < streaming
            assert events[:streaming].count(('sample', True)) >= 2
            assert robot.hand.counts['read_crc'] == 1
            assert int(robot.hand_feedback.snapshot()['timestamp_ns'][0]) > full.observation_timestamp_ns
            if kind == 'teleop':
                assert len(anchors) == 1
                np.testing.assert_allclose(anchors[0].hand['qpos'][0], .2)
        else:
            assert 'streaming' not in events and not anchors
        if outcome == 'no_feedback':
            assert not state.quit_requested and not state.estop_request
    finally:
        robot.stop()
        recorder = owner.recorder
        if recorder is not None:
            if outcome == 'writer_failure':
                with pytest.raises(recording.RecordingError, match='writer setup failed'):
                    recorder.finish(save=False, reason='preparation_failed')
            else:
                assert recorder.finish(save=True, reason='start_cancelled') is None
            assert not recorder.writer_pending and recorder.written_frames == 0
            assert recorder.published_path is None
    assert not list(tmp_path.glob('episode_*'))
