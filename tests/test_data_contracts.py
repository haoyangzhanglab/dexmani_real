"""用临时 RGB-D 样本验证现有 Raw 发布、导出和回放读取契约。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import zarr

from dexmani_real.config.real import RealConfig
from dexmani_real.data.export import ExportConfig, export_zarr
from dexmani_real.data.raw import RawReader, camera_metadata, require_published_raw
from dexmani_real.data.recorder import EpisodeRecorder, RecordingError
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
