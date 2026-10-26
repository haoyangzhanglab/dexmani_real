"""来源时钟、相机/点云身份和触觉缓存，不把处理时钟当成采样时间。"""

import multiprocessing as mp
import unittest
from concurrent.futures import CancelledError
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.config import ExperimentConfig
from dexmani_real.calibration.camera.session import detect_aruco_stable
from dexmani_real.ipc.ring import SharedMemoryRingBuffer
from dexmani_real.ipc.schema import (
    CAMERA_FRAME_HEADER_DTYPE,
    make_camera_frame_dtype,
    make_pointcloud_frame_dtype,
)
from dexmani_real.robot.feedback import ARM_STATE_DTYPE, HAND_STATE_DTYPE
from dexmani_real.robot.xhand.feedback import XHandFeedback
from dexmani_real.runtime.observation import (
    ObservationNeeds,
    ObservationRow,
    check_observation,
    read_camera_frame,
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


class CameraCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.ring = SharedMemoryRingBuffer(
            mp.get_context('spawn'), make_camera_frame_dtype((2, 2, 3), (2, 2)), 5,
        )
        self.channels = SimpleNamespace(camera_ring=self.ring)

    def publish(self, stamp, color_number, pixel=0):
        return self.ring.write(
            header=np.array((stamp, stamp + 7, 200, color_number), dtype=CAMERA_FRAME_HEADER_DTYPE),
            rgb=np.full((2, 2, 3), pixel, dtype=np.uint8), depth=np.ones((2, 2), dtype=np.uint16),
        )

    def detect(self, frames, *, n_frames=1, check_capture=None, result=None):
        frames = iter(frames)
        read_fields = self.ring.read_fields
        clock = [10.0]

        def read(*args, **kwargs):
            frame = next(frames, None)
            if frame is not None:
                self.publish(*frame)
            return read_fields(*args, **kwargs)

        def sleep(seconds):
            clock[0] += seconds

        with (
            patch.object(self.ring, 'read_fields', side_effect=read) as reads,
            patch('dexmani_real.calibration.camera.session.time.monotonic', side_effect=lambda: clock[0]),
            patch('dexmani_real.calibration.camera.session.time.monotonic_ns', return_value=10_000_000_000),
            patch('dexmani_real.calibration.camera.session.time.sleep', side_effect=sleep),
            patch('dexmani_real.calibration.camera.session.detect_aruco_pose', return_value=result) as detector,
        ):
            self.detector = detector
            value = detect_aruco_stable(
                self.channels, np.eye(3), np.zeros(5), detector=None,
                marker_corners=np.zeros((4, 3)), target_id=None, max_frame_age_s=.5,
                n_frames=n_frames, not_before_ns=8_000_000_000,
                check_capture=check_capture if check_capture is not None else lambda: None,
            )
        for call in reads.call_args_list:
            self.assertEqual(call.args, (('header', 'rgb'),))
        return value

    def test_public_reader_preserves_device_identity_and_source_clock(self):
        self.assertIsNone(read_camera_frame(self.channels))
        color_number = 2**63 + 3
        sequence = self.publish(123, color_number)
        for payload, fields in (('rgbd', {'rgb', 'depth'}), ('rgb', {'rgb'}), ('depth', {'depth'})):
            with self.subTest(payload=payload):
                frame = read_camera_frame(self.channels, sequence, payload=payload)
                self.assertEqual(frame['color_frame_number'], color_number)
                self.assertIs(type(frame['color_frame_number']), int)
                self.assertEqual(frame['timestamp_ns'], 123)
                self.assertEqual(frame['ring_sequence'], sequence)
                self.assertEqual(set(frame) - {'timestamp_ns', 'ring_sequence', 'color_frame_number'}, fields)
        header = self.ring.read_fields(('header',))[0]['header']
        self.assertEqual((int(header['timestamp_ns']), int(header['received_ns'])), (123, 130))

    def test_calibration_admits_only_fresh_post_barrier_distinct_rgb_frames(self):
        frames = [
            (8_000_000_000, 10, 1),  # 屏障端点。
            (9_000_000_000, 11, 2),  # 屏障之后但来源过期。
            (9_600_000_000, 20, 3),
            (9_700_000_000, 20, 4),  # 新发布和新 timestamp 不能替代 RGB 身份。
            (9_550_000_000, 21, 5),  # 新身份仍不能越过来源时间递增检查。
            (9_800_000_000, 21, 6),
        ]
        result = self.detect(frames, n_frames=2, result=(np.zeros(3), np.ones(3)))
        np.testing.assert_array_equal(result, (np.zeros(3), np.ones(3)))
        self.assertEqual([int(call.args[0][0, 0, 0]) for call in self.detector.call_args_list], [3, 6])
        self.assertEqual(self.ring.latest_sequence, 6)

    def test_detection_failure_returns_none(self):
        self.assertIsNone(self.detect([(9_600_000_000, 20, 3)]))
        self.detector.assert_called_once()

    def test_cancel_before_read_before_detection_and_after_detection_propagates(self):
        for cancelled_at in (1, 2, 3):
            with self.subTest(cancelled_at=cancelled_at):
                check = Mock(side_effect=[None] * (cancelled_at - 1) + [CancelledError('capture cancelled')])
                with self.assertRaisesRegex(CancelledError, 'capture cancelled'):
                    self.detect([(9_600_000_000, 20, 3)], check_capture=check,
                                result=(np.zeros(3), np.ones(3)))
                self.assertEqual(self.detector.call_count, int(cancelled_at == 3))

    def test_missing_and_republished_rgb_frames_time_out(self):
        with self.assertRaisesRegex(TimeoutError, 'fresh calibration image unavailable'):
            self.detect([])
        self.detector.assert_not_called()
        with self.assertRaisesRegex(TimeoutError, 'fresh calibration image unavailable'):
            self.detect([(9_600_000_000, 20, 3), (9_700_000_000, 20, 4)], n_frames=2,
                        result=(np.zeros(3), np.ones(3)))
        self.detector.assert_called_once()
