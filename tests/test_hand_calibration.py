import contextlib
import io
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.config.teleop import HandRetargetingConfig, TeleopConfig
from dexmani_real.config.xhand import HandConfig
from dexmani_real.teleop.retargeting.adaptive import AdaptiveHandRetargeter
from dexmani_real.teleop.retargeting.landmarks import (
    HandCalibration, hand_lengths, prepare_landmarks,
)
from dexmani_real.teleop.retargeting.objective import HandObjective
from dexmani_real.teleop.run import calibrate_hand


def hand_frame():
    return np.array([
        [0, 0, 0],
        [.005, .035, .035], [.01, .05, .05], [.015, .06, .065], [.02, .07, .08],
        [0, .03, .075], [0, .03, .11], [0, .03, .135], [0, .03, .155],
        [0, 0, .085], [0, 0, .12], [0, 0, .145], [0, 0, .165],
        [0, -.018, .08], [0, -.018, .112], [0, -.018, .135], [0, -.018, .155],
        [0, -.04, .065], [0, -.04, .092], [0, -.04, .11], [0, -.04, .127],
    ])


class HandCalibrationTests(unittest.TestCase):
    def test_median_measurement_and_original_scatter_rejections(self):
        config = HandRetargetingConfig()
        base = hand_frame()
        measurements = [hand_lengths(prepare_landmarks(base * (1 + i * .001)))
                        for i in range(-12, 13)]
        calibration = HandCalibration.from_measurements(measurements, config)
        expected = hand_lengths(prepare_landmarks(base))
        np.testing.assert_array_equal(calibration.bone_lengths, expected[:15].reshape(5, 3))
        self.assertEqual(calibration.palm_width, expected[15])
        self.assertEqual(calibration.sample_count, 25)
        with self.assertRaisesRegex(ValueError, "25 frames"):
            HandCalibration.from_measurements(measurements[:24], config)
        unstable = [expected * scale for scale in np.linspace(.5, 1.5, 25)]
        with self.assertRaisesRegex(ValueError, "unstable"):
            HandCalibration.from_measurements(unstable, config)
        outlier = [expected.copy() for _ in range(24)] + [expected * 1.5]
        with self.assertRaisesRegex(ValueError, "beyond the calibrated tolerance"):
            HandCalibration.from_measurements(outlier, config)

    def test_collection_reuses_only_fresh_valid_measurements(self):
        retargeter = Mock()
        loop = SimpleNamespace(
            controller=SimpleNamespace(hand_retargeter=retargeter),
            runtime=SimpleNamespace(
                hand_retargeting=replace(HandRetargetingConfig(), calibration_samples=3),
                teleop=TeleopConfig(),
            ),
            keyboard=SimpleNamespace(check_health=Mock(return_value=True)),
            robot=SimpleNamespace(service_idle=Mock()),
            sensors=SimpleNamespace(check=Mock(return_value=True), channels=object()),
            poll_blocking_commands=Mock(return_value=False),
        )
        base = hand_frame()
        frames = [
            dict(ring_sequence=1, recv_ts_ns=2_000_000_000, landmarks=base),
            dict(ring_sequence=1, recv_ts_ns=2_050_000_000, landmarks=base),
            dict(ring_sequence=2, recv_ts_ns=1_900_000_000, landmarks=base),
            dict(ring_sequence=2, recv_ts_ns=2_020_000_000, landmarks=base),
            dict(ring_sequence=2, recv_ts_ns=2_300_000_000, landmarks=np.zeros((21, 3))),
            dict(ring_sequence=3, recv_ts_ns=2_500_000_000, landmarks=base),
            dict(ring_sequence=4, recv_ts_ns=2_600_000_000, landmarks=base),
        ]
        clocks = [2_100_000_000, 2_200_000_000, 2_300_000_000, 2_700_000_000,
                  2_750_000_000, 2_800_000_000, 2_850_000_000]
        with patch("dexmani_real.teleop.run.read_vr_frame", side_effect=frames), \
                patch("dexmani_real.teleop.run.time.monotonic", return_value=0), \
                patch("dexmani_real.teleop.run.time.monotonic_ns", side_effect=clocks), \
                patch("dexmani_real.teleop.run.time.sleep"), \
                patch("dexmani_real.teleop.retargeting.landmarks.prepare_landmarks", wraps=prepare_landmarks) as prepare, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(calibrate_hand(loop))
        self.assertEqual(prepare.call_count, 4)  # 3 个有效帧及 1 次早拒，无最终重算。
        retargeter.calibrate.assert_called_once()
        measurements = retargeter.calibrate.call_args.args[0]
        self.assertEqual(np.asarray(measurements).shape, (3, 16))
        np.testing.assert_array_equal(measurements, np.tile(hand_lengths(prepare_landmarks(base)), (3, 1)))
        self.assertFalse(np.shares_memory(measurements[0], base))

    def test_fixed_robot_geometry_and_target_gradient(self):
        retargeter = AdaptiveHandRetargeter(HandRetargetingConfig(), HandConfig(), control_hz=16)
        points = prepare_landmarks(hand_frame())
        retargeter.calibrate([hand_lengths(points)] * 25)
        targets = retargeter.calibration.targets(
            points, retargeter.robot_points, retargeter.config,
            robot_segments=retargeter.robot_segments,
            robot_palm_width=retargeter.robot_palm_width,
            thumb_base_xy=retargeter.thumb_base_xy,
            thumb_base_radius=retargeter.thumb_base_radius,
        )
        np.testing.assert_array_equal(
            retargeter.robot_segments, np.linalg.norm(np.diff(retargeter.robot_points, axis=1), axis=2),
        )
        qpos = np.array([.5, .2, .4, .05, .4, .6, .5, .4, .3, .8, .4, .7])
        objective = HandObjective(retargeter.fk, retargeter.frame_ids, targets,
                                  qpos - .02, 1 / 16, retargeter.config)
        gradient = np.empty(12)
        objective(qpos, gradient)
        epsilon = 1e-6
        numerical = np.empty(12)
        for index in range(12):
            offset = np.zeros(12)
            offset[index] = epsilon
            numerical[index] = (objective(qpos + offset, np.empty(0))
                                - objective(qpos - offset, np.empty(0))) / (2 * epsilon)
        np.testing.assert_allclose(gradient, numerical, atol=1e-7, rtol=1e-6)

    def test_rejected_commit_preserves_command_and_time(self):
        retargeter = AdaptiveHandRetargeter.__new__(AdaptiveHandRetargeter)
        retargeter.limits = np.tile([0., 1.], (12, 1))
        retargeter.previous_command = np.full(12, .5)
        retargeter.command_timestamp_ns = 100
        for command, stamp in ((np.full(12, np.nan), 101), (np.full(12, 1.01), 101),
                               (np.full(12, .6), 99)):
            with self.subTest(stamp=stamp, command=command):
                with self.assertRaises(ValueError):
                    retargeter.commit(command, timestamp_ns=stamp)
                np.testing.assert_array_equal(retargeter.previous_command, np.full(12, .5))
                self.assertEqual(retargeter.command_timestamp_ns, 100)


if __name__ == "__main__":
    unittest.main()
