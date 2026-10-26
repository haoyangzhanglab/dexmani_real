"""真实 Raw timing、RGB 解码及 CordViP 消费契约。"""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
from raw_fixture import raw_row, write_raw_episode

from cordvip_geometry.integrations.dexmani import RawHandCloudProvider, iter_raw_episode
from dexmani_real.data.canonical.processing import validate_export_episode
from dexmani_real.data.raw.reader import EpisodeReader, RawDataError
from dexmani_real.data.raw.schema import TIMING_ROW_DTYPE, encode_timing_row
from dexmani_real.replay.trajectory import load_trajectory


class RawReaderTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path = write_raw_episode(self.root)
        self.timing = np.array([encode_timing_row(raw_row(i)) for i in range(2)], dtype=TIMING_ROW_DTYPE)

    def test_selected_timing_fields_keep_order_int64_and_exact_values(self):
        with EpisodeReader(self.path) as reader:
            reader.validate_completion()
            reader.validate_rows(chunk_frames=1)
            for fields in (('camera_sequence', 'camera_ns'), ['camera_ns', 'camera_sequence'], ('arm_ns',)):
                with self.subTest(fields=fields):
                    rows = reader.read_timing_rows(fields)
                    self.assertEqual(rows.dtype.names, tuple(fields))
                    self.assertEqual(rows.shape, (2,))
                    for name in fields:
                        self.assertEqual(rows.dtype[name], np.dtype('<i8'))
                        np.testing.assert_array_equal(rows[name], self.timing[name])

    def test_missing_clocks_are_diagnostic_data_not_training_admission(self):
        with h5py.File(self.path / 'data.h5', 'r+') as data:
            rows = data['timing/rows'][:]
            rows['camera_ns'] = -1
            data['timing/rows'][:] = rows
        with EpisodeReader(self.path) as reader:
            actual = reader.read_timing_rows(('camera_ns', 'camera_sequence'))
            np.testing.assert_array_equal(actual['camera_ns'], [-1, -1])
            np.testing.assert_array_equal(actual['camera_sequence'], self.timing['camera_sequence'])
            with self.assertRaisesRegex(RawDataError, 'invalid consumed source clock: camera_ns'):
                reader.validate_rows()

    def test_invalid_field_selection_is_a_parameter_error(self):
        with EpisodeReader(self.path) as reader:
            for fields in ('camera_ns', None, {'camera_ns'}, (1,), ('camera_ns', None)):
                with self.subTest(fields=fields), self.assertRaises(TypeError):
                    reader.read_timing_rows(fields)
            for fields in ((), [], ('unknown',), ('camera_ns', 'camera_ns')):
                with self.subTest(fields=fields):
                    with self.assertRaises(ValueError) as raised:
                        reader.read_timing_rows(fields)
                    self.assertNotIsInstance(raised.exception, RawDataError)

    def test_timing_structure_errors_are_shared_with_strict_validation(self):
        for defect in ('missing', 'group', 'row_count', 'rank', 'dtype'):
            with self.subTest(defect=defect):
                path = self.root / defect
                shutil.copytree(self.path, path)
                with h5py.File(path / 'data.h5', 'r+') as data:
                    del data['timing/rows']
                    if defect == 'group':
                        data.create_group('timing/rows')
                    elif defect != 'missing':
                        shape = {'row_count': (1,), 'rank': (2, 1), 'dtype': (2,)}[defect]
                        dtype = np.dtype('int64') if defect == 'dtype' else TIMING_ROW_DTYPE
                        data.create_dataset('timing/rows', shape=shape, dtype=dtype)
                with EpisodeReader(path) as reader:
                    for read in (lambda: reader.read_timing_rows(('camera_ns',)), reader.validate_rows):
                        with self.assertRaisesRegex(RawDataError, 'timing/rows'):
                            read()

    def test_closed_reader_rejects_reads_even_after_caches_are_populated(self):
        reader = EpisodeReader(self.path)
        self.addCleanup(reader.close)
        reader.fields
        reader['arm_qpos']
        reader.validate_tactile_metadata()
        timing = reader.read_timing_rows(('camera_ns',))
        images = reader.iter_rgb_frames()
        self.addCleanup(images.close)
        next(images)
        reader.close()
        for read in (
            lambda: reader.meta, lambda: reader.fields, lambda: reader['arm_qpos'],
            lambda: reader.read_timing_rows(('camera_ns',)), reader.validate_completion,
            reader.validate_tactile_metadata, reader.validate_rows,
            lambda: next(reader.iter_rgb_frames()),
        ):
            with self.subTest(read=read), self.assertRaisesRegex(RuntimeError, 'Raw reader is closed'):
                read()
        np.testing.assert_array_equal(timing['camera_ns'], self.timing['camera_ns'])

    def test_missing_dataset_reports_raw_data_error(self):
        with EpisodeReader(self.path) as reader:
            with self.assertRaisesRegex(RawDataError, 'missing_dataset'):
                reader['missing_dataset']

    def test_export_and_replay_admission_remain_separate_from_diagnostic_reads(self):
        for status in ('normal', 'failed'):
            with self.subTest(status=status):
                with h5py.File(self.path / 'data.h5', 'r+') as data:
                    data['meta'].attrs['technical_status'] = status
                    data['meta'].attrs['collection_source'] = 'teleop'
                with EpisodeReader(self.path) as reader:
                    np.testing.assert_array_equal(
                        reader.read_timing_rows(('camera_ns',))['camera_ns'], self.timing['camera_ns'],
                    )
                    if status == 'failed':
                        with self.assertRaisesRegex(RawDataError, 'technical failure'):
                            validate_export_episode(reader)
                    else:
                        camera, _, _ = validate_export_episode(reader)
                        self.assertEqual(camera.intrinsics.width, 16)
                with patch('dexmani_real.replay.trajectory.compute_eef_pose_history_xarm_base') as fk:
                    fk.return_value = np.zeros((2, 9))
                    if status == 'failed':
                        with self.assertRaisesRegex(RawDataError, 'technical failure'):
                            load_trajectory(self.path)
                        fk.assert_not_called()
                    else:
                        trajectory = load_trajectory(self.path)
                        self.assertEqual((trajectory.num_frames, trajectory.fps), (2, 10))
                        fk.assert_called_once()
                # 诊断消费者允许查看失败产物，不代表通过训练或物理回放准入。
                self.assertEqual(len(list(iter_raw_episode(self.path))), 2)

    def test_cordvip_consumes_rgb_video_and_true_source_identity(self):
        stream = iter_raw_episode(self.path)
        try:
            first = next(stream)
            frames = [first, *stream]
        finally:
            stream.close()
        self.assertEqual(len(frames), 2)
        for index, frame in enumerate(frames):
            self.assertEqual(frame.row_index, index)
            self.assertEqual(frame.rgb.shape, (16, 16, 3))
            self.assertEqual(frame.rgb.dtype, np.uint8)
            self.assertEqual(frame.source_id, f'raw:{self.path}:camera:{17 + index}')
            self.assertEqual(frame.timestamp_ns, int(self.timing['camera_ns'][index]))
            self.assertEqual(frame.clock, 'host_monotonic_oldest_rgbd_receive')
            np.testing.assert_array_equal(
                frame.depth_m, raw_row(index).camera['depth'].astype(np.float32) * np.float32(.001),
            )

    def test_hand_cloud_uses_real_reader_and_keeps_missing_clock_unverified(self):
        for missing_camera_clock in (False, True):
            with self.subTest(missing_camera_clock=missing_camera_clock):
                if missing_camera_clock:
                    with h5py.File(self.path / 'data.h5', 'r+') as data:
                        rows = data['timing/rows'][:]
                        rows['camera_ns'] = -1
                        data['timing/rows'][:] = rows
                # 仅替换 FK/mesh 数值开销；Reader、相机元数据及视频解码都是真实链路。
                with (
                    patch('dexmani_real.geometry.kinematics.arm_fk.ArmFK') as arm_fk,
                    patch('cordvip_geometry.hand_cloud.HandCloudModel') as hand_model,
                ):
                    arm_fk.return_value.compute.return_value = (np.zeros(3), np.array([1, 0, 0, 0, 1, 0]))
                    hand_model.return_value.metadata = {}
                    hand_model.return_value.transform.return_value = {'points': np.zeros((4, 3))}
                    provider = RawHandCloudProvider(self.path, num_points=4)
                fields = ('arm_ns', 'hand_ns', 'camera_ns', 'camera_sequence')
                self.assertEqual(provider._timing.dtype.names, fields)
                expected = self.timing.copy()
                if missing_camera_clock:
                    expected['camera_ns'] = -1
                for name in fields:
                    np.testing.assert_array_equal(provider._timing[name], expected[name])
                for index, frame in enumerate(iter_raw_episode(self.path)):
                    cloud = provider.for_frame(frame)
                    self.assertEqual(cloud['camera_timestamp_ns'], expected['camera_ns'][index])
                    self.assertEqual(cloud['arm_timestamp_ns'], expected['arm_ns'][index])
                    self.assertEqual(cloud['hand_timestamp_ns'], expected['hand_ns'][index])
                    self.assertEqual(cloud['timing_available'], not missing_camera_clock)
                    if missing_camera_clock:
                        self.assertIsNone(cloud['skew_ms'])
                        self.assertEqual(cloud['clock'], 'raw_observation_relative')
                        self.assertEqual(frame.timestamp_ns, index * 100_000_000)
                    else:
                        self.assertEqual(cloud['skew_ms'], 80 / 1e6)
