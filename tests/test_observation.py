"""来源时钟、相机/点云身份和触觉缓存，不把处理时钟当成采样时间。"""

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.config import ExperimentConfig
from dexmani_real.ipc.schema import (
    CAMERA_FRAME_HEADER_DTYPE,
    make_pointcloud_frame_dtype,
)
from dexmani_real.robot.feedback import ARM_STATE_DTYPE, HAND_STATE_DTYPE
from dexmani_real.robot.xhand.feedback import XHandFeedback
from dexmani_real.runtime.observation import (
    ObservationNeeds,
    ObservationRow,
    check_observation,
    read_observation,
)


class ObservationTests(unittest.TestCase):
    def test_arm_and_hand_keep_distinct_timeout_endpoints(self):
        cfg = ExperimentConfig()
        cfg = replace(cfg, arm=replace(cfg.arm, feedback_max_age_s=1), hand=replace(cfg.hand, feedback_max_age_s=1))
        arm, hand = np.zeros(1, dtype=ARM_STATE_DTYPE), np.zeros(1, dtype=HAND_STATE_DTYPE)
        arm['timestamp_ns'] = hand['timestamp_ns'] = 100
        row = ObservationRow(arm, None, None, None, None, 1_000_000_100)
        issues, deadline = check_observation(row, cfg, ObservationNeeds(), require_hand=False, now_ns=1_000_000_100)
        self.assertEqual(issues, [])
        self.assertEqual(deadline, 1_000_000_101)
        row = replace(row, hand=hand)
        issues, _ = check_observation(row, cfg, ObservationNeeds(), now_ns=1_000_000_100)
        self.assertIn('hand_timestamp_ns_unavailable_or_stale', issues)

    def test_cloud_camera_sequence_and_original_time_survive_repeated_reads(self):
        frame = np.zeros(1, dtype=make_pointcloud_frame_dtype(8))
        frame['source_camera_sequence'], frame['timestamp_ns'] = 42, 100
        frame['processing_started_ns'] = 900
        header = np.zeros((), dtype=CAMERA_FRAME_HEADER_DTYPE)
        header['timestamp_ns'] = 100
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        channels = SimpleNamespace(pointcloud_ring=Mock(), camera_ring=Mock())
        channels.pointcloud_ring.read_latest.return_value = (frame, 1000, 5)
        channels.camera_ring.read_fields.return_value = ({'header': header, 'rgb': rgb}, 1000, 42)
        arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
        arm['timestamp_ns'] = 100
        robot = SimpleNamespace(read_state=lambda: SimpleNamespace(arm=arm, hand=None))
        needs = ObservationNeeds(camera='rgb', point_cloud=True)
        with patch('dexmani_real.runtime.observation.time.monotonic_ns', side_effect=[1000, 1001, 2000, 2001]):
            first = read_observation(channels, robot, needs=needs)
            second = read_observation(channels, robot, needs=needs)
        channels.camera_ring.read_fields.assert_called_with(('header', 'rgb'), sequence=42)
        self.assertEqual((first.pointcloud_timestamp_ns, second.pointcloud_timestamp_ns), (100, 100))
        self.assertEqual(second.camera['timestamp_ns'], 100)
        self.assertEqual((first.observation_timestamp_ns, second.observation_timestamp_ns), (1001, 2001))
        self.assertFalse(second.arm.flags.writeable)
        self.assertFalse(second.point_cloud.flags.writeable)
        self.assertFalse(second.camera['rgb'].flags.writeable)
        with self.assertRaises(TypeError):
            second.camera['timestamp_ns'] = 2000
        channels.camera_ring.read_fields.return_value = None
        unmatched = read_observation(channels, robot, needs=needs)
        issues, _ = check_observation(unmatched, ExperimentConfig(), needs, require_hand=False, now_ns=2001)
        self.assertIn('camera_source_unavailable_or_unmatched', issues)

    def test_tactile_hold_keeps_original_vector_clock_and_expires(self):
        config = replace(ExperimentConfig().hand, feedback_max_age_s=.1)
        feedback = XHandFeedback(config)
        start = 1_000_000_000
        feedback.reset(start)
        sample = SimpleNamespace(qpos=np.zeros(12), tactile_aggregate=np.ones((5, 3)),
                                 tactile_dense=np.ones((5, 120, 3)), commboard_err=[0], jointboard_err=[0], tipboard_err=[0])
        first = feedback.consume(sample, source_ns=start, now_ns=start)
        sample.tactile_aggregate[0, 0] = np.nan
        with self.assertLogs('dexmani_real.robot.xhand.feedback', level='WARNING'):
            second = feedback.consume(sample, source_ns=start+10, now_ns=start+10)
            held = feedback.consume(None, source_ns=start+50_000_000, now_ns=start+50_000_000)
        self.assertIs(held, second)
        np.testing.assert_array_equal(held['tactile_aggregate'][0, 0], first['tactile_aggregate'][0, 0])
        self.assertEqual(held['tactile_aggregate_timestamp_ns'][0], start)
        self.assertEqual(held['timestamp_ns'][0], start+10)
        self.assertFalse(held.flags.writeable)
        with self.assertRaisesRegex(RuntimeError, 'feedback age reached'):
            feedback.consume(None, source_ns=start+100_000_010, now_ns=start+100_000_010)
