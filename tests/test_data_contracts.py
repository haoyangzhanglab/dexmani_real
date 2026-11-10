"""用临时 RGB-D 样本验证现有 Raw 发布、导出和回放读取契约。"""

import contextlib
import io
import json
import logging
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import zarr

from dexmani_real.config.real import RealConfig
from dexmani_real.data.export import ExportConfig, export_zarr, select_episodes
from dexmani_real.data.hoi import HOI_FIELDS, validate_hoi, write_hoi
from dexmani_real.data.raw import (
    RawReader,
    camera_metadata,
    read_episode_outcome,
    require_published_raw,
    write_episode_outcome,
)
from dexmani_real.data.recorder import (
    EpisodeRecorder,
    RecordingError,
    format_recording_result,
)
from dexmani_real.geometry.camera import CameraIntrinsics, CameraSnapshot
from dexmani_real.replay.trajectory import load_trajectory
from dexmani_real.robot.commands import RobotCommand
from dexmani_real.robot.xarm7.protocol import ARM_STATE_DTYPE
from dexmani_real.robot.xhand.feedback import HAND_STATE_DTYPE
from dexmani_real.runtime.observation import Observation


class DataContracts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='dexmani_test_', dir='/tmp')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.runtime = RealConfig()
        self.arm = (np.array(self.runtime.arm.joint_limit_lower) + self.runtime.arm.joint_limit_upper)/2
        self.hand = (np.array(self.runtime.hand.qpos_min_rad) + self.runtime.hand.qpos_max_rad)/2
        camera = CameraSnapshot('offline', CameraIntrinsics(16, 16, 10., 10., 8., 8., 'none', (0, 0, 0, 0, 0)), 0.001, None)
        self.metadata = camera_metadata(camera)

    def row(self, index):
        control = 10**10 + index * 100_000_000
        arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
        hand = np.zeros(1, dtype=HAND_STATE_DTYPE)
        arm['qpos'], hand['qpos'] = self.arm, self.hand
        arm['timestamp_ns'] = control-1_000_000
        for name in ('timestamp_ns', 'tactile_aggregate_timestamp_ns', 'tactile_dense_timestamp_ns'):
            hand[name] = control-1_000_000
        hand['tactile_aggregate'], hand['tactile_dense'] = 0.5, 0.25
        camera = {'timestamp_ns': control-2_000_000, 'ring_sequence': index+1,
                      'rgb': np.full((16, 16, 3), index, np.uint8), 'depth': np.full((16, 16), 100+index, np.uint16)}
        vr = {'wrist_pos': np.zeros(3), 'wrist_quat_wxyz': np.array([1., 0., 0., 0.]),
                  'landmarks': np.zeros((21, 3)), 'recv_ts_ns': control-3_000_000, 'sequence_id': index+1}
        return Observation(arm, hand, camera, vr, None, control+1_000_000, control_timestamp_ns=control)

    def record(self, source, *, command=True):
        recorder = EpisodeRecorder(self.root, control_hz=10)
        recorder.start(task_label='offline', collection_source=source, metadata=self.metadata, episode_name=source)
        try:
            with self.assertRaisesRegex(ValueError, 'unpublished Raw'):
                require_published_raw(recorder.staging_path)
            for index in range(4):
                if index == 2:
                    recorder.mark_gap()
                target = RobotCommand(self.arm, self.hand) if command else None
                self.assertIsNone(recorder.append(self.row(index), target))
        finally:
            published = recorder.finish(save=True, reason='operator')
        self.assertEqual(recorder.result()['status'], 'published')
        return published

    def test_raw_fields_timing_and_segments(self):
        expected = {
            'arm_qpos': ((4, 7), 'float64'), 'hand_qpos': ((4, 12), 'float64'),
            'hand_contact': ((4, 5, 3), 'float32'), 'hand_tactile_force': ((4, 5, 120, 3), 'float32'),
            'action_arm_joint_target': ((4, 7), 'float64'), 'action_hand_joint_target': ((4, 12), 'float64'),
            'depth': ((4, 16, 16), 'uint16'),
        }
        for source in ('policy_rollout', 'teleop'):
            with self.subTest(source=source), RawReader(self.record(source)) as reader:
                specs = dict(expected)
                if source == 'teleop':
                    specs.update(vr_wrist_pos=((4, 3), 'float64'), vr_wrist_quat_wxyz=((4, 4), 'float64'),
                                 vr_landmarks=((4, 21, 3), 'float64'), vr_recv_ts_ns=((4,), 'uint64'),
                                 vr_sequence_id=((4,), 'uint64'))
                self.assertEqual(reader.fields, set(specs) | {'rgb'})
                for name, (shape, dtype) in specs.items():
                    self.assertEqual(reader[name].shape, shape)
                    self.assertEqual(reader[name].dtype, np.dtype(dtype))
                timing = reader.read_timing()
                self.assertEqual(timing.dtype.names, ('observation_ns', 'control_ns', 'arm_ns', 'hand_ns',
                                                      'tactile_aggregate_ns', 'tactile_dense_ns', 'camera_ns', 'camera_sequence'))
                self.assertTrue(all(timing.dtype[n] == np.dtype('<i8') for n in timing.dtype.names))
                for index in range(4):
                    control = 10**10 + index*100_000_000
                    self.assertEqual(tuple(timing[index]), (control+1_000_000, control,
                                     control-1_000_000, control-1_000_000, control-1_000_000,
                                     control-1_000_000, control-2_000_000, index+1))
                np.testing.assert_array_equal(reader.segment_starts, [0, 2])
                self.assertEqual(len(list(reader.iter_rgb_frames())), 4)
                self.assertEqual(reader.meta['arm_action_semantics'], 'q_goal_submission_attempt')
                self.assertEqual(reader.meta['joint_units'], 'rad')
                self.assertEqual(reader.meta['hand_tactile_scale_divisor'], 255.)

    def test_export_and_replay_boundaries(self):
        path = self.record('policy_rollout')
        with self.assertRaisesRegex(ValueError, 'select one continuous segment'):
            load_trajectory(path, self.runtime)
        for segment in (0, 1):
            trajectory = load_trajectory(path, self.runtime, segment=segment)
            self.assertEqual(len(trajectory), 2)
            self.assertEqual(trajectory.control_hz, 10)
            np.testing.assert_array_equal(trajectory.action_arm_joint, np.tile(self.arm, (2, 1)))
        with self.assertRaises(ValueError):
            load_trajectory(path, self.runtime, segment=2)
        result = export_zarr(path, self.root/'output.zarr', processing=ExportConfig(),
                             fields=('joint_state', 'action', 'finger_tactile_force', 'taxel_tactile_force', 'rgb', 'depth'))
        self.assertEqual((result['raw_episode_count'], result['episode_count'], result['total_frames']), (1, 2, 4))
        store = zarr.open(str(self.root/'output.zarr'), mode='r')
        try:
            np.testing.assert_array_equal(store['meta/episode_ends'][:], [2, 4])
            for name, shape, dtype in (('joint_state', (4, 19), 'float32'), ('action', (4, 19), 'float32'),
                                       ('rgb', (4, 16, 16, 3), 'uint8'), ('depth', (4, 16, 16), 'uint16')):
                self.assertEqual(store['data/'+name].shape, shape)
                self.assertEqual(store['data/'+name].dtype, np.dtype(dtype))
            np.testing.assert_array_equal(store['data/taxel_tactile_force'][:], np.full((4, 5, 120, 3), 0.25, np.float32))
            self.assertEqual(store.attrs['dt'], 0.1)
        finally:
            store.store.close()

    def test_broken_hoi_only_blocks_hoi_consumers(self):
        path = self.record('policy_rollout')
        with h5py.File(path / 'data.h5', 'r+') as data:
            data.create_group('hoi')
        with patch('dexmani_real.data.raw.validate_hoi', wraps=validate_hoi) as validate:
            with RawReader(path) as reader:
                self.assertEqual(reader.meta['task_label'], 'offline')
                self.assertTrue(set(HOI_FIELDS).issubset(reader.fields))
                reader.require_fields('arm_qpos', 'rgb', 'depth')
                self.assertEqual(reader['arm_qpos'].shape, (4, 7))
                self.assertEqual(reader['depth'].shape, (4, 16, 16))
                reader.read_timing()
                self.assertEqual(len(list(reader.iter_rgb_frames())), 4)
                validate.assert_not_called()
                with self.assertRaises(KeyError):
                    _ = reader.hoi_metadata
                for name in HOI_FIELDS:
                    with self.subTest(field=name):
                        with self.assertRaises(KeyError):
                            reader.require_fields(name)
                        with self.assertRaises(KeyError):
                            reader[name]
            self.assertEqual(len(load_trajectory(path, self.runtime, segment=0)), 2)
            export_zarr(path, self.root / 'ordinary.zarr', processing=ExportConfig(), fields=('joint_state',))
            with self.assertRaises(ValueError):
                export_zarr(path, self.root / 'hoi.zarr', processing=ExportConfig(), fields=('object_pointcloud',))
        self.assertFalse((self.root / 'hoi.zarr').exists())

    def test_hoi_validation_is_cached_and_preserves_task_identity(self):
        path = self.record('policy_rollout')
        manifest = {'task_name': 'offline', 'object_names': ['object'], 'template_sha256': 'a' * 64}
        write_hoi(path / 'data.h5',
                  {'points_local': np.zeros((1024, 3), np.float32), 'offsets': np.array([0, 1024])},
                  manifest, np.tile(np.eye(4), (4, 1, 1, 1)), np.ones((4, 1), bool),
                  np.ones((4, 1), np.float32), np.eye(4))
        with patch('dexmani_real.data.raw.validate_hoi', wraps=validate_hoi) as validate, \
                RawReader(path) as reader:
            reader.require_fields('arm_qpos')
            validate.assert_not_called()
            reader.require_fields(*HOI_FIELDS)
            self.assertEqual(reader.hoi_metadata, manifest)
            for name in HOI_FIELDS:
                reader[name]
                reader.require_fields(name)
            validate.assert_called_once()
        with h5py.File(path / 'data.h5', 'r+') as data:
            data['meta'].attrs['task_label'] = 'other'
        with RawReader(path) as reader, self.assertRaisesRegex(ValueError, 'task_name differs'):
            reader['object_pointcloud']
        with h5py.File(path / 'data.h5', 'r+') as data:
            del data['meta'].attrs['task_label']
        with RawReader(path) as reader, self.assertRaisesRegex(ValueError, 'requires /meta task_label'):
            _ = reader.hoi_metadata
        with h5py.File(path / 'data.h5', 'r+') as data:
            data['meta'].attrs['task_label'] = 'offline'
            del data['hoi/object_pointcloud']
            data['hoi'].create_dataset('object_pointcloud', shape=(4, 1024, 3), dtype='f8')
        with RawReader(path) as reader, self.assertRaisesRegex(ValueError, 'shape/dtype'):
            reader.require_fields('object_pointcloud')

    def test_absent_and_staging_hoi_are_not_published(self):
        path = self.record('policy_rollout')
        for staging in (False, True):
            if staging:
                with h5py.File(path / 'data.h5', 'r+') as data:
                    data.create_group('_hoi_staging')
            with self.subTest(staging=staging), RawReader(path) as reader, \
                    patch('dexmani_real.data.raw.validate_hoi') as validate:
                self.assertIsNone(reader.hoi_metadata)
                self.assertTrue(set(HOI_FIELDS).isdisjoint(reader.fields))
                with self.assertRaisesRegex(ValueError, 'missing fields'):
                    reader.require_fields('object_pointcloud')
                with self.assertRaises(KeyError):
                    reader['object_pointcloud']
                validate.assert_not_called()

    def test_no_attempt_nan_rejected_by_export_and_replay(self):
        path = self.record('policy_rollout', command=False)
        with RawReader(path) as reader:
            self.assertTrue(np.isnan(reader['action_arm_joint_target'][:]).all())
        with self.assertRaisesRegex(ValueError, 'finite targets'):
            load_trajectory(path, self.runtime, segment=0)
        with self.assertRaises(ValueError):
            export_zarr(path, self.root/'failed.zarr', processing=ExportConfig(), fields=('action',))
        self.assertFalse((self.root/'failed.zarr').exists())

    def test_writer_failure_never_publishes(self):
        recorder = EpisodeRecorder(self.root, control_hz=10)
        with patch('dexmani_real.data.recorder.RawWriter.open', side_effect=OSError('disk failure')), \
                self.assertLogs('dexmani_real.data.recorder', level='ERROR'):
            with self.assertRaisesRegex(RecordingError, 'disk failure'):
                recorder.start(task_label='offline', collection_source='policy_rollout', metadata=self.metadata)
            recorder.finish(save=True, reason='failed')
        self.assertEqual(recorder.result()['status'], 'failed')
        self.assertIsNone(recorder.published_path)
        self.assertTrue(recorder.staging_path.is_dir())

    def test_invalid_segment_and_clock_rejected(self):
        path = self.record('policy_rollout')
        with h5py.File(path/'data.h5', 'r+') as data:
            data['timing/segment_starts'][:] = [0, 0]
        with RawReader(path) as reader, self.assertRaisesRegex(ValueError, 'invalid timing/segment_starts'):
            _ = reader.segment_starts
        with h5py.File(path/'data.h5', 'r+') as data:
            data['timing/segment_starts'][:] = [0, 2]
            timing = data['timing/rows'][:]
            timing['control_ns'][1] = timing['control_ns'][0]
            data['timing/rows'][:] = timing
        with RawReader(path) as reader, self.assertRaisesRegex(ValueError, 'control_clock_not_increasing'):
            reader.read_timing()

    def test_exclude_names_are_checked_before_selection(self):
        for name in ('episode_a', 'episode_b'):
            (self.root / name).mkdir()
        selected, excluded = select_episodes(self.root, ('episode_a', 'episode_a'))
        self.assertEqual([p.name for p in selected], ['episode_b'])
        self.assertEqual(excluded, {'episode_a'})
        with self.assertRaisesRegex(ValueError, "unknown excluded episode.*'bad_a'.*'bad_b'"):
            select_episodes(self.root, ('episode_a', 'bad_b', 'bad_a'))
        with self.assertRaisesRegex(ValueError, 'no selected episodes'):
            select_episodes(self.root, ('episode_a', 'episode_b'))

    def test_outcome_sidecar_and_explicit_replacement(self):
        from scripts.label_episode import main as label
        from scripts.visualize_episode import main as info

        # 合成样本包含调度断点：截断不妨碍人工评价。
        path = self.record('policy_rollout')
        original = {name: (path / name).read_bytes() for name in ('data.h5', 'rgb.mp4')}
        self.assertEqual(read_episode_outcome(path), {'outcome': 'unknown'})
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(info([str(path), '--info']), 0)
        self.assertIn('Task outcome: unknown', output.getvalue())
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(label([str(path), 'failure', '--note', '抓取后滑落']), 0)
        annotation = read_episode_outcome(path)
        self.assertEqual(annotation['note'], '抓取后滑落')
        self.assertTrue(annotation['annotated_at_utc'].endswith('Z'))
        self.assertEqual(datetime.fromisoformat(annotation['annotated_at_utc'].replace('Z', '+00:00')).tzinfo,
                         timezone.utc)
        with self.assertRaises(FileExistsError):
            write_episode_outcome(path, 'success')
        self.assertEqual(read_episode_outcome(path), annotation)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(label([str(path), 'success', '--overwrite']), 0)
        self.assertNotIn('note', read_episode_outcome(path))
        with contextlib.redirect_stdout(io.StringIO()) as output, \
                patch('scripts.visualize_episode.EpisodeVisualizer', side_effect=AssertionError('viewer initialized')):
            self.assertEqual(info([str(path), '--info']), 0)
        self.assertIn('Task outcome: success', output.getvalue())
        for name, data in original.items():
            self.assertEqual((path / name).read_bytes(), data)

    def test_invalid_outcome_is_not_unknown(self):
        path = self.root
        for content, message in (
            ('{', 'Expecting property name'),
            ('[]', 'outcome must be'),
            ('{}', 'outcome must be'),
            ('{"outcome":"typo","annotated_at_utc":"2026-11-10T08:00:00Z"}', 'outcome must be'),
            ('{"outcome":"success","annotated_at_utc":"2026-11-10T08:00:00Z","note":2}', 'note must be'),
        ):
            with self.subTest(content=content):
                (path / 'outcome.json').write_text(content)
                with self.assertRaisesRegex(ValueError, message):
                    read_episode_outcome(path)
        with self.assertRaises(ValueError):
            write_episode_outcome(path, 'typo', overwrite=True)

    def test_read_outcome_requires_episode_directory(self):
        path = self.root / 'missing'
        with self.assertRaises(NotADirectoryError):
            read_episode_outcome(path)
        path.write_text('not a directory')
        with self.assertRaises(NotADirectoryError):
            read_episode_outcome(path)
        self.assertEqual(read_episode_outcome(self.root), {'outcome': 'unknown'})

    def test_outcome_requires_valid_utc_timestamp(self):
        path = self.root / 'outcome.json'
        path.write_text(json.dumps({'outcome': 'success'}))
        with self.assertRaisesRegex(ValueError, 'annotated_at_utc'):
            read_episode_outcome(self.root)
        for timestamp in (None, 123, '', 'invalid', '2026-02-30T08:00:00Z',
                          '2026-11-10', '2026-11-10T08:00:00', '2026-11-10T08:00:00+08:00'):
            with self.subTest(timestamp=timestamp):
                path.write_text(json.dumps({'outcome': 'success', 'annotated_at_utc': timestamp}))
                with self.assertRaisesRegex(ValueError, 'annotated_at_utc'):
                    read_episode_outcome(self.root)
        for timestamp in ('2026-11-10T08:00:00Z', '2026-11-10T08:00:00+00:00',
                          '2026-11-10T08:00:00.123456Z'):
            with self.subTest(timestamp=timestamp):
                annotation = {'outcome': 'success', 'annotated_at_utc': timestamp, 'note': '完成'}
                path.write_text(json.dumps(annotation))
                self.assertEqual(read_episode_outcome(self.root), annotation)

    def test_info_rejects_invalid_outcome_before_visualization(self):
        from scripts.visualize_episode import main as info

        # 显式 unknown 也必须有合法时间戳，不能等同于缺少 Sidecar。
        (self.root / 'outcome.json').write_text(json.dumps({'outcome': 'unknown'}))
        with contextlib.redirect_stdout(io.StringIO()) as output, \
                patch('scripts.visualize_episode.EpisodeVisualizer') as viewer, \
                self.assertRaisesRegex(ValueError, 'annotated_at_utc'):
            info([str(self.root), '--info'])
        viewer.assert_not_called()
        self.assertNotIn('Task outcome: unknown', output.getvalue())

    def test_outcome_requires_published_raw_structure(self):
        missing = self.root / 'missing'
        with self.assertRaises(NotADirectoryError):
            write_episode_outcome(missing, 'success')
        missing.mkdir()
        with self.assertRaises(OSError):
            write_episode_outcome(missing, 'success')
        path = self.record('policy_rollout')
        staging = path.rename(path.with_name('.tmp_episode'))
        with self.assertRaisesRegex(ValueError, 'unpublished Raw'):
            write_episode_outcome(staging, 'success')
        alias = self.root / 'episode_alias'
        alias.symlink_to(staging, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'unpublished Raw'):
            write_episode_outcome(alias, 'success')
        staging.rename(path)
        with h5py.File(path / 'data.h5', 'r+') as data:
            data['meta'].attrs['termination_reason'] = 'recording_incomplete'
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            write_episode_outcome(path, 'success')
        with h5py.File(path / 'data.h5', 'r+') as data:
            data['meta'].attrs['termination_reason'] = 'operator'
        (path / 'rgb.mp4').unlink()
        with self.assertRaisesRegex(ValueError, 'missing fields'):
            write_episode_outcome(path, 'success')
        self.assertFalse((path / 'outcome.json').exists())

    def test_outcome_is_independent_of_modality_quality(self):
        path = self.record('teleop')
        with h5py.File(path / 'data.h5', 'r+') as data:
            data.create_group('hoi')
            for name in ('arm_qpos', 'hand_tactile_force', 'vr_landmarks'):
                data[name].resize(3, axis=0)
            del data['depth']
            del data['meta'].attrs['hand_tactile_scale_divisor']
            del data['meta'].attrs['camera_color_intrinsics']
        original = {name: (path / name).read_bytes() for name in ('data.h5', 'rgb.mp4')}
        with patch.object(RawReader, 'read_camera', side_effect=AssertionError('camera read')), \
                patch.object(RawReader, 'read_timing', side_effect=AssertionError('timing scan')), \
                patch.object(RawReader, 'iter_rgb_frames', side_effect=AssertionError('video decode')), \
                patch('dexmani_real.data.raw.validate_hoi', side_effect=AssertionError('HOI validation')):
            write_episode_outcome(path, 'success')
        self.assertEqual(read_episode_outcome(path)['outcome'], 'success')
        for name, content in original.items():
            self.assertEqual((path / name).read_bytes(), content)

    def test_outcome_requires_core_metadata_and_rows(self):
        path = self.record('policy_rollout')
        for name in ('task_label', 'control_hz', 'collection_source', 'termination_reason'):
            with h5py.File(path / 'data.h5', 'r+') as data:
                value = data['meta'].attrs[name]
                del data['meta'].attrs[name]
            with self.subTest(missing=name), self.assertRaises(KeyError):
                write_episode_outcome(path, 'success')
            with h5py.File(path / 'data.h5', 'r+') as data:
                data['meta'].attrs[name] = value
        with h5py.File(path / 'data.h5', 'r+') as data:
            data['timing/segment_starts'][:] = [0, 0]
        with self.assertRaisesRegex(ValueError, 'invalid timing/segment_starts'):
            write_episode_outcome(path, 'success')
        with h5py.File(path / 'data.h5', 'r+') as data:
            data['timing/segment_starts'][:] = [0, 2]
            del data['timing/rows']
            data['timing'].create_dataset('rows', data=np.zeros(4))
        with self.assertRaisesRegex(ValueError, 'incompatible shape/dtype'):
            write_episode_outcome(path, 'success')
        with h5py.File(path / 'data.h5', 'r+') as data:
            del data['timing/rows']
            data['timing'].create_dataset('rows', data=np.zeros(0))
        with self.assertRaisesRegex(ValueError, 'no rows'):
            write_episode_outcome(path, 'success')
        self.assertFalse((path / 'outcome.json').exists())

    def test_pending_and_inaccessible_staging_formatting(self):
        raw = {'status': 'pending', 'path': None, 'staging_path': str(self.root), 'row_count': 3,
               'error': 'writer close delayed', 'recording_truncated_reason': None}
        with patch('dexmani_real.data.recorder.os.access', return_value=False):
            level, text = format_recording_result(raw)
        self.assertEqual(level, logging.WARNING)
        self.assertIn('尚无最终录制结果', text)
        self.assertIn('Staging：不可用', text)
        self.assertIn('writer close delayed', text)
        self.assertNotIn('按 B', text)
        self.assertNotIn('保存成功', text)

    def test_unpublished_result_does_not_infer_discard(self):
        recorder = EpisodeRecorder(self.root, control_hz=10)
        recorder.staging_path = self.root
        level, text = format_recording_result(recorder.result())
        self.assertEqual(level, logging.INFO)
        self.assertIn('Raw 未发布', text)
        self.assertNotIn('丢弃', text)
        self.assertNotIn('未录到数据', text)

    def test_unreadable_outcome_is_not_unknown(self):
        path = self.root / 'outcome.json'
        path.symlink_to(self.root / 'missing.json')
        with self.assertRaises(FileNotFoundError):
            read_episode_outcome(self.root)
        path.unlink()
        path.mkdir()
        with self.assertRaises(IsADirectoryError):
            read_episode_outcome(self.root)
        with (patch('pathlib.Path.open', side_effect=PermissionError('denied')),
              self.assertRaises(PermissionError)):
            read_episode_outcome(self.root)
