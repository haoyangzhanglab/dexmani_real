"""用临时 RGB-D 样本验证现有 Raw 发布、导出和回放读取契约。"""

import contextlib
import io
import json
import logging
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import zarr

from dexmani_real.config.real import RealConfig
from dexmani_real.data.export import ExportConfig, export_zarr, select_episodes
from dexmani_real.data.hoi import write_hoi
from dexmani_real.data.raw import (
    RawReader,
    RawWriter,
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
        self.vr_metadata = {
            'vr_to_robot_rotation': [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]],
            'hand_retargeting_calibration_json': json.dumps({
                'bone_lengths': (np.arange(20).reshape(5, 4) / 1000 + .02).tolist(),
                'palm_width': .08,
            }),
        }

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
        quat = np.array([4., 1., -2., 3. + index])
        vr = {'wrist_pos': np.array([.1, -.2, .3]) + index,
                  'wrist_quat_wxyz': quat / np.linalg.norm(quat),
                  'landmarks': (np.arange(63).reshape(21, 3) + index * 100) / 1000,
                  'recv_ts_ns': control-3_000_000, 'sequence_id': index * 3 + 7}
        return Observation(arm, hand, camera, vr, None, control+1_000_000, control_timestamp_ns=control)

    def record(self, source, *, command=True):
        recorder = EpisodeRecorder(self.root, control_hz=10)
        metadata = self.metadata | self.vr_metadata if source == 'teleop' else self.metadata
        recorder.start(task_label='offline', collection_source=source, metadata=metadata, episode_name=source)
        try:
            with self.assertRaisesRegex(ValueError, 'unpublished Raw'):
                require_published_raw(recorder.staging_path)
            for index in range(4):
                if index == 2:
                    recorder.mark_gap()
                target = RobotCommand(self.arm, self.hand) if command else None
                row = self.row(index)
                if source == 'policy_rollout':
                    row = replace(row, vr=None)
                self.assertIsNone(recorder.append(row, target))
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
                self.assertEqual(reader.meta['collection_source'], source)
                self.assertEqual(reader.meta['termination_reason'], 'operator')
                self.assertEqual(reader.meta['recording_truncated_reason'], 'control_slots_skipped')
                self.assertNotIn('meta/termination_details', reader.data_file)
                if source == 'teleop':
                    for index in range(4):
                        for name, value in self.row(index).vr.items():
                            np.testing.assert_array_equal(reader[f'vr_{name}'][index], value)
                    np.testing.assert_array_equal(reader.meta['vr_to_robot_rotation'],
                                                  self.vr_metadata['vr_to_robot_rotation'])
                    self.assertEqual(json.loads(reader.meta['hand_retargeting_calibration_json']),
                                     json.loads(self.vr_metadata['hand_retargeting_calibration_json']))
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
        with RawReader(path) as reader:
            self.assertEqual(reader.meta['task_label'], 'offline')
            self.assertNotIn('object_pointcloud', reader.fields)
            reader.require_fields('arm_qpos', 'rgb', 'depth')
            self.assertEqual(reader['arm_qpos'].shape, (4, 7))
            self.assertEqual(reader['depth'].shape, (4, 16, 16))
            reader.read_timing()
            self.assertEqual(len(list(reader.iter_rgb_frames())), 4)
            with self.assertRaisesRegex(ValueError, 'missing fields'):
                reader.require_fields('object_pointcloud')
            with self.assertRaises(KeyError):
                reader['object_pointcloud']
        self.assertEqual(len(load_trajectory(path, self.runtime, segment=0)), 2)
        export_zarr(path, self.root / 'ordinary.zarr', processing=ExportConfig(), fields=('joint_state',))
        with self.assertRaises(ValueError):
            export_zarr(path, self.root / 'hoi.zarr', processing=ExportConfig(), fields=('object_pointcloud',))
        self.assertFalse((self.root / 'hoi.zarr').exists())

    def test_malformed_hoi_rejected_before_visualizer_start(self):
        from scripts.visualize_episode import EpisodeVisualizer
        path = self.record('policy_rollout')
        for shape, dtype in ((None, None), ((4, 1, 3), 'f4'), ((3, 1024, 3), 'f4'), ((4, 1024, 3), 'f8')):
            with self.subTest(shape=shape, dtype=dtype):
                with h5py.File(path / 'data.h5', 'r+') as data:
                    group = data.create_group('hoi')
                    if shape is not None:
                        group.create_dataset('object_pointcloud', shape, dtype=dtype)
                error = 'missing fields' if shape is None else 'shape/dtype'
                with RawReader(path) as reader:
                    self.assertEqual(reader['arm_qpos'].shape, (4, 7))
                    with self.assertRaisesRegex(ValueError, error):
                        reader.require_fields('object_pointcloud')
                        reader['object_pointcloud']
                rerun = MagicMock()
                with patch.dict(sys.modules, {'rerun': rerun}), self.assertRaisesRegex(ValueError, error):
                    EpisodeVisualizer(str(path), table_plane_path=None)
                rerun.init.assert_not_called()
                with h5py.File(path / 'data.h5', 'r+') as data:
                    del data['hoi']

    def test_annotation_stops_failed_episode_and_deduplicates(self):
        from scripts.cordvip.annotate_episodes import annotate_episodes
        failed, good = self.record('policy_rollout'), self.record('teleop')
        for path in (failed, good):
            with h5py.File(path / 'data.h5', 'r+') as data:
                data['meta'].attrs['camera_T_xarm_base_from_color'] = np.eye(4).ravel()
        for path in (failed, good):
            with h5py.File(path / 'data.h5', 'r+') as data:
                rows = data['timing/rows'][:]
                rows['camera_sequence'][1] = rows['camera_sequence'][0]
                rows['camera_ns'][1] = rows['camera_ns'][0]
                data['timing/rows'][:] = rows
        template = {'points_local': np.zeros((1024, 3), np.float32),
                    'offsets': np.array([0, 1024]), 'object_names': np.array(['object'])}
        from cordvip_geometry.tracking import TrackingFailure
        valid = {'object': np.eye(4)}
        invalid = TrackingFailure('object: depth lost')
        tracker = MagicMock()
        tracker.step.side_effect = [valid, invalid, valid, valid, valid]
        with patch('scripts.cordvip.annotate_episodes.load_task_template', return_value=template), \
                patch('scripts.cordvip.annotate_episodes.task_objects', return_value=[{'name': 'object', 'prompt': 'object'}]), \
                patch('scripts.cordvip.annotate_episodes.SceneTracker.open') as opened, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            opened.return_value.__enter__.return_value = tracker
            annotate_episodes([failed, good], self.root)
            self.assertIn('首次失败行 2', output.getvalue())
            self.assertEqual(tracker.step.call_count, 5)
            tracker.reset.assert_called_once()
            with h5py.File(failed / 'data.h5') as data:
                self.assertNotIn('hoi', data)
            with h5py.File(good / 'data.h5') as data:
                self.assertEqual(set(data['hoi']), {'object_pointcloud'})
                self.assertEqual(data['hoi/object_pointcloud'].shape, (4, 1024, 3))
            annotate_episodes([good], self.root)
            opened.assert_called_once()
            self.assertEqual(tracker.step.call_count, 5)
            for error in (RuntimeError('model failed'), OSError('IO failed')):
                tracker.step.side_effect = error
                with self.assertRaises(type(error)):
                    annotate_episodes([failed], self.root)

    def test_annotation_deduplicates_before_depth_read_but_consumes_rgb(self):
        from scripts.cordvip.annotate_episodes import annotate_episodes
        module = 'scripts.cordvip.annotate_episodes.'
        camera = CameraSnapshot('offline', CameraIntrinsics(16, 16, 10., 10., 8., 8.,
                                'none', (0, 0, 0, 0, 0)), 0.001, np.eye(4))
        template = {'object_names': np.array(['object'])}
        for scenario, error in (('success', None), ('time_conflict', '来源时间冲突'),
                                ('short_rgb', 'ended before row 2'),
                                ('extra_rgb', 'more frames'), ('bad_repeat_rgb', 'grid invalid at row 1'),
                                ('bad_depth', 'Raw depth')):
            with self.subTest(scenario=scenario):
                timing = np.array([(10, 100), (10, 100), (11, 200)],
                                  dtype=[('camera_sequence', 'i8'), ('camera_ns', 'i8')])
                if scenario == 'time_conflict':
                    timing['camera_ns'][1] += 1
                depth = MagicMock(shape=(3, 16, 16), dtype=np.dtype('uint16'))
                if scenario == 'bad_depth':
                    depth.dtype = np.dtype('float32')
                depth.__getitem__.side_effect = lambda index: np.full((16, 16), 100 + index, np.uint16)
                consumed = []
                def images():
                    count = 2 if scenario == 'short_rgb' else 4 if scenario == 'extra_rgb' else 3
                    for index in range(count):
                        consumed.append(index)
                        dtype = np.float32 if scenario == 'bad_repeat_rgb' and index == 1 else np.uint8
                        yield np.full((16, 16, 3), index, dtype)
                reader = MagicMock(num_frames=3, data_file={})
                reader.__enter__.return_value = reader
                reader.read_camera.return_value = camera
                reader.__getitem__.side_effect = {'depth': depth, 'timing/rows': timing}.__getitem__
                reader.iter_rgb_frames.return_value = images()
                tracker = MagicMock()
                first, second = np.eye(4), np.eye(4)
                second[0, 3] = 1
                tracker.step.side_effect = [{'object': first}, {'object': second}]
                with (patch(module + 'RawReader', return_value=reader),
                      patch(module + 'require_published_raw'),
                      patch(module + 'load_task_template', return_value=template),
                      patch(module + 'task_objects', return_value=[{'name': 'object', 'prompt': 'object'}]),
                      patch(module + 'SceneTracker.open') as opened,
                      patch(module + 'write_hoi') as publish,
                      contextlib.redirect_stdout(io.StringIO())):
                    opened.return_value.__enter__.return_value = tracker
                    if error:
                        with self.assertRaisesRegex(ValueError, error):
                            annotate_episodes([self.root / 'episode'], self.root)
                        publish.assert_not_called()
                        if scenario == 'bad_depth':
                            opened.assert_not_called()
                    else:
                        annotate_episodes([self.root / 'episode'], self.root)
                        self.assertEqual(consumed, [0, 1, 2])
                        self.assertEqual([c.args[0] for c in depth.__getitem__.call_args_list], [0, 2])
                        self.assertEqual(tracker.step.call_count, 2)
                        tracker.detect_masks.assert_called_once()
                        publish.assert_called_once()
                        poses = publish.call_args.args[2]
                        self.assertEqual(poses.shape, (3, 1, 4, 4))
                        np.testing.assert_array_equal(poses[:, 0], [first, first, second])
                        for call, index in zip(tracker.step.call_args_list, (0, 2)):
                            frame = call.args[0]
                            np.testing.assert_array_equal(frame.rgb, np.full((16, 16, 3), index, np.uint8))
                            self.assertEqual(frame.depth_m.dtype, np.float32)
                            np.testing.assert_array_equal(frame.depth_m,
                                np.full((16, 16), 100 + index, np.float32) * np.float32(camera.depth_scale_m))

    def test_joint_hoi_export_preserves_rows_across_chunks_and_episodes(self):
        template = {'points_local': np.zeros((1024, 3), np.float32), 'offsets': [0, 1024]}
        for episode_index, (source, frames) in enumerate((('policy_rollout', 131), ('teleop', 3))):
            recorded = self.record(source)
            path = recorded.rename(self.root / f'episode_{episode_index}')
            with h5py.File(path / 'data.h5', 'r+') as data:
                rows = np.resize(data['timing/rows'][:], frames)
                for name in rows.dtype.names:
                    rows[name] += (np.arange(frames) // 4) * (4 if name == 'camera_sequence' else 400_000_000)
                del data['timing/rows']
                data['timing'].create_dataset('rows', data=rows)
                del data['timing/segment_starts']
                data['timing'].create_dataset('segment_starts', data=[0])
                for name in ('arm_qpos', 'hand_qpos', 'action_arm_joint_target', 'action_hand_joint_target'):
                    values = np.resize(data[name][:], (frames, data[name].shape[1]))
                    del data[name]
                    data.create_dataset(name, data=values)
            poses = np.tile(np.eye(4), (frames, 1, 1, 1))
            poses[:, 0, 0, 3] = episode_index * 1000 + np.arange(frames)
            write_hoi(path / 'data.h5', template, poses, np.eye(4))
        destination = self.root / 'combined.zarr'
        result = export_zarr(self.root, destination, processing=ExportConfig(hoi_max_skew_s=.01),
                             fields=('object_pointcloud', 'joint_state', 'action'))
        self.assertEqual(result['total_frames'], 134)
        store = zarr.open_group(str(destination), mode='r')
        try:
            np.testing.assert_array_equal(store['meta/episode_ends'][:], [131, 134])
            for key in ('object_pointcloud', 'joint_state', 'action'):
                self.assertEqual(store['data/' + key].shape[0], 134)
            np.testing.assert_array_equal(store['data/object_pointcloud'][:, 0, 0],
                                          np.r_[np.arange(131), 1000 + np.arange(3)])
        finally:
            store.store.close()

    def test_minimal_hoi_export_and_skew(self):
        path = self.record('policy_rollout')
        template = {'points_local': np.zeros((1024, 3), np.float32), 'offsets': np.array([0, 1024])}
        write_hoi(path / 'data.h5', template, np.tile(np.eye(4), (4, 1, 1, 1)), np.eye(4))
        fields = ('object_pointcloud', 'joint_state', 'action')
        export_zarr(path, self.root / 'default.zarr', processing=ExportConfig(), fields=fields)
        export_zarr(path, self.root / 'valid.zarr', processing=ExportConfig(hoi_max_skew_s=.001), fields=fields)
        with self.assertRaisesRegex(ValueError, 'skew.*row 0'):
            export_zarr(path, self.root / 'over.zarr', processing=ExportConfig(hoi_max_skew_s=.000999999), fields=fields)
        with h5py.File(path / 'data.h5', 'r+') as data:
            self.assertEqual(set(data['hoi']), {'object_pointcloud'})
            self.assertFalse(dict(data['hoi'].attrs))
            rows = data['timing/rows'][:]
            rows['camera_ns'][0] -= 100_000_000
            data['timing/rows'][:] = rows
        with self.assertRaisesRegex(ValueError, 'skew.*row 0'):
            export_zarr(path, self.root / 'skew.zarr', processing=ExportConfig(hoi_max_skew_s=.01), fields=fields)

    def test_hoi_visualization_and_nonfinite_export(self):
        from scripts.visualize_episode import EpisodeVisualizer

        path = self.record('policy_rollout')
        template = {'points_local': np.zeros((1024, 3), np.float32), 'offsets': [0, 1024]}
        write_hoi(path / 'data.h5', template, np.tile(np.eye(4), (4, 1, 1, 1)), np.eye(4))
        for value in (1., np.nan, np.inf):
            with self.subTest(value=value):
                with h5py.File(path / 'data.h5', 'r+') as data:
                    data['hoi/object_pointcloud'][0, 0, 0] = value
                with patch.dict(sys.modules, {'rerun': MagicMock(), 'rerun.blueprint': MagicMock()}), \
                        patch('scripts.visualize_episode.ArmFK'), \
                        patch('scripts.visualize_episode.HandFK'), \
                        patch('scripts.visualize_episode.HandPointCloud'), \
                        patch.object(EpisodeVisualizer, 'log_hand_geometry'), \
                        patch('scripts.visualize_episode.log_points') as points:
                    viewer = EpisodeVisualizer(str(path), table_plane_path=None)
                    try:
                        rgb = next(viewer.reader.iter_rgb_frames())
                        if np.isfinite(value):
                            viewer.log_step(0, rgb)
                            self.assertEqual(points.call_args.args[0], 'object_pointcloud')
                            expected = np.zeros((1024, 3), np.float32)
                            expected[0, 0] = value
                            np.testing.assert_array_equal(points.call_args.args[1], expected)
                        else:
                            with self.assertRaisesRegex(ValueError, 'nonfinite.*row 0'):
                                viewer.log_step(0, rgb)
                    finally:
                        viewer.close()
                if not np.isfinite(value):
                    with self.assertRaisesRegex(ValueError, 'nonfinite.*row 0'):
                        export_zarr(path, self.root / f'{value}.zarr', processing=ExportConfig(),
                                    fields=('object_pointcloud',))

    def test_hoi_write_failure_remains_unpublished(self):
        path = self.record('policy_rollout')
        template = {'points_local': np.zeros((1024, 3), np.float32), 'offsets': [0, 1024]}
        poses = np.tile(np.eye(4), (4, 1, 1, 1))
        poses[2, 0, 0, 3] = np.nan
        with self.assertRaisesRegex(ValueError, '非有限'):
            write_hoi(path / 'data.h5', template, poses, np.eye(4))
        with RawReader(path) as reader:
            self.assertIn('_hoi_staging', reader.data_file)
            self.assertNotIn('hoi', reader.data_file)
            self.assertNotIn('object_pointcloud', reader.fields)
        poses[2, 0, 0, 3] = 0
        write_hoi(path / 'data.h5', template, poses, np.eye(4))
        with RawReader(path) as reader:
            self.assertNotIn('_hoi_staging', reader.data_file)
            self.assertEqual(set(reader.data_file['hoi']), {'object_pointcloud'})
            np.testing.assert_array_equal(reader['object_pointcloud'][:], np.zeros((4, 1024, 3), np.float32))
        with self.assertRaises(FileExistsError):
            write_hoi(path / 'data.h5', template, poses, np.eye(4))

    def test_absent_and_staging_hoi_are_not_published(self):
        path = self.record('policy_rollout')
        for staging in (False, True):
            if staging:
                with h5py.File(path / 'data.h5', 'r+') as data:
                    data.create_group('_hoi_staging').create_dataset('object_pointcloud', (4, 1024, 3), dtype='f4')
            with self.subTest(staging=staging), RawReader(path) as reader:
                self.assertNotIn('object_pointcloud', reader.fields)
                with self.assertRaisesRegex(ValueError, 'missing fields'):
                    reader.require_fields('object_pointcloud')
                with self.assertRaises(KeyError):
                    reader['object_pointcloud']
            with self.assertRaisesRegex(ValueError, 'missing fields'):
                export_zarr(path, self.root / f'staging_{staging}.zarr', processing=ExportConfig(),
                            fields=('object_pointcloud',))

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

    def test_finish_and_close_failure_never_publish(self):
        close = RawWriter.close

        def fail_close(writer):
            close(writer)
            raise OSError('close failure')

        for stage in ('finish', 'close'):
            with self.subTest(stage=stage):
                recorder = EpisodeRecorder(self.root, control_hz=10)
                failure = fail_close if stage == 'close' else OSError('finish failure')
                with patch.object(RawWriter, stage, autospec=True, side_effect=failure), \
                        self.assertLogs('dexmani_real.data.recorder', level='ERROR'):
                    recorder.start(task_label='offline', collection_source='policy_rollout',
                                   metadata=self.metadata, episode_name=stage)
                    try:
                        recorder.append(replace(self.row(0), vr=None))
                    finally:
                        recorder.finish(save=True, reason='operator')
                self.assertEqual(recorder.result()['status'], 'failed')
                self.assertIsNone(recorder.published_path)
                self.assertTrue(recorder.staging_path.is_dir())
                self.assertFalse(recorder.writer_pending)

    def test_first_finish_request_controls_save_and_discard(self):
        for save in (True, False):
            with self.subTest(save=save):
                recorder = EpisodeRecorder(self.root, control_hz=10)
                recorder.start(task_label='offline', collection_source='policy_rollout',
                               metadata=self.metadata, episode_name=f'finish_{save}')
                staging = recorder.staging_path
                try:
                    recorder.append(replace(self.row(0), vr=None))
                    request = recorder.request_finish(save=save, reason='operator')
                    deadline = request.deadline
                    self.assertIs(recorder.request_finish(save=not save, reason='quit'), request)
                    self.assertEqual((request.save, request.reason, request.deadline),
                                     (save, 'operator', deadline))
                finally:
                    recorder.finish(save=not save, reason='quit')
                self.assertFalse(recorder.writer_pending)
                self.assertTrue(recorder.queue.empty())
                self.assertFalse(staging.exists())
                self.assertEqual(recorder.result()['status'], 'published' if save else 'not_published')
                if save:
                    with RawReader(recorder.published_path) as reader:
                        self.assertEqual(reader.num_frames, 1)
                        self.assertEqual(reader.meta['termination_reason'], 'operator')
                        self.assertNotIn('recording_truncated_reason', reader.meta)
                        self.assertNotIn('meta/termination_details', reader.data_file)
                else:
                    self.assertIsNone(recorder.published_path)

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
        self.assertEqual(annotation, {'outcome': 'failure', 'note': '抓取后滑落'})
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
            ('{"outcome":"typo"}', 'outcome must be'),
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

    def test_outcome_ignores_extra_fields(self):
        for extra in ({}, {'annotated_at_utc': '2026-11-10T08:00:00+08:00'},
                      {'annotated_at_utc': 'invalid', 'note': 2, 'extra': []}):
            annotation = {'outcome': 'success', **extra}
            (self.root / 'outcome.json').write_text(json.dumps(annotation))
            self.assertEqual(read_episode_outcome(self.root), annotation)

    def test_info_rejects_invalid_outcome_before_visualization(self):
        from scripts.visualize_episode import main as info

        (self.root / 'outcome.json').write_text(json.dumps({'outcome': 'invalid'}))
        with contextlib.redirect_stdout(io.StringIO()) as output, \
                patch('scripts.visualize_episode.EpisodeVisualizer') as viewer, \
                self.assertRaisesRegex(ValueError, 'outcome must be'):
            info([str(self.root), '--info'])
        viewer.assert_not_called()
        self.assertNotIn('Task outcome: unknown', output.getvalue())

    def test_outcome_only_requires_existing_directory(self):
        path = self.root / 'episode'
        with self.assertRaises(NotADirectoryError):
            write_episode_outcome(path, 'success')
        path.mkdir()
        with patch('dexmani_real.data.raw.RawReader', side_effect=AssertionError('Raw opened')), \
                patch('dexmani_real.data.raw.require_published_raw', side_effect=AssertionError('Raw checked')):
            write_episode_outcome(path, 'success')
        self.assertEqual(read_episode_outcome(path), {'outcome': 'success'})

    def test_pending_staging_formatting(self):
        raw = {'status': 'pending', 'path': None, 'staging_path': str(self.root), 'row_count': 3,
               'error': 'writer close delayed', 'recording_truncated_reason': None}
        level, text = format_recording_result(raw)
        self.assertEqual(level, logging.WARNING)
        self.assertIn('尚无最终录制结果', text)
        self.assertIn('Staging：' + raw['staging_path'], text)
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
        path.mkdir()
        with self.assertRaises(IsADirectoryError):
            read_episode_outcome(self.root)
        with (patch('pathlib.Path.open', side_effect=PermissionError('denied')),
              self.assertRaises(PermissionError)):
            read_episode_outcome(self.root)
