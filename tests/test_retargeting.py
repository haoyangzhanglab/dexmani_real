"""重定向的数学边界与失败诊断；不连接设备。"""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.config.teleop import HandRetargetingConfig
from dexmani_real.config.xhand import HandConfig
from dexmani_real.geometry.kinematics import HandFK
from dexmani_real.robot.model import XHAND_FINGERTIP_LINK_NAMES
from dexmani_real.robot.xhand.driver import XHand
from dexmani_real.teleop.retargeting.adaptive import AdaptiveHandRetargeter, FINGER_FRAME_NAMES
from dexmani_real.teleop.retargeting.landmarks import HandTargets
from dexmani_real.teleop.retargeting.objective import HandObjective
from dexmani_real.utils.io import write_json


class RetargetingTests(unittest.TestCase):
    def test_zero_weights_and_time_domains(self):
        config = HandRetargetingConfig()
        for name in (
            "shape_weight", "tip_weight", "direction_weight", "pinch_weight",
            "proximal_direction_weight", "damping_s", "smoothing_tau_s",
        ):
            with self.subTest(name=name):
                replace(config, **{name: 0}).validate()
                with self.assertRaises(ValueError):
                    replace(config, **{name: -1}).validate()
                with self.assertRaises(ValueError):
                    replace(config, **{name: float("nan")}).validate()
        for name in (
            "calibration_timeout_s", "huber_delta_m", "direction_huber_delta",
            "pinch_start_m", "pinch_full_m", "max_joint_speed_rad_s",
            "max_solve_time_s", "ftol_abs",
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                replace(config, **{name: 0}).validate()

    def test_zero_tip_weight_keeps_direction_and_pinch_gradients(self):
        fk = HandFK()
        names = [
            name for pair, tip in zip(FINGER_FRAME_NAMES, XHAND_FINGERTIP_LINK_NAMES)
            for name in (*pair, tip)
        ]
        frame_ids = [fk.model.getFrameId(name) for name in names]
        qpos = np.array([.5, .2, .4, .05, .4, .6, .5, .4, .3, .8, .4, .7])
        points = fk.forward(qpos + .03, frame_ids)[:, :3, 3].reshape(5, 3, 3)
        directions = np.diff(points, axis=1)
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
        targets = HandTargets(
            points, directions, np.array([.3, .5, .7, .9]),
            np.linalg.norm(points[1:, 2] - points[0, 2], axis=1) * .95, True,
        )
        config = replace(HandRetargetingConfig(), tip_weight=0)
        objective = HandObjective(fk, frame_ids, targets, qpos - .02, 1 / 16, config)
        gradient = np.empty(12)
        loss = objective(qpos, gradient)
        epsilon = 1e-6
        finite_difference = np.empty(12)
        for index in range(12):
            offset = np.zeros(12)
            offset[index] = epsilon
            finite_difference[index] = (
                objective(qpos + offset, np.empty(0))
                - objective(qpos - offset, np.empty(0))
            ) / (2 * epsilon)
        np.testing.assert_allclose(gradient, finite_difference, atol=1e-7, rtol=1e-6)
        without_directions_or_pinch = HandObjective(
            fk, frame_ids, targets, qpos - .02, 1 / 16,
            replace(config, direction_weight=0, proximal_direction_weight=0, pinch_weight=0),
        )
        position_gradient = np.empty(12)
        position_loss = without_directions_or_pinch(qpos, position_gradient)
        self.assertGreater(loss, position_loss)
        self.assertGreater(np.max(np.abs(gradient - position_gradient)), .1)

    def test_nonfinite_loss_rejects_without_advancing_command_and_can_be_saved(self):
        for initial, final in ((float("nan"), 1.), (float("inf"), 1.),
                               (1., float("nan")), (1., float("inf"))):
            with self.subTest(initial=initial, final=final):
                retargeter = AdaptiveHandRetargeter.__new__(AdaptiveHandRetargeter)
                retargeter.config = HandRetargetingConfig()
                retargeter.nominal_dt_s = 1 / 16
                retargeter.calibration = Mock()
                retargeter.calibration.targets.return_value = SimpleNamespace(thumb_base_observable=True)
                retargeter.previous_command = np.full(12, .5)
                retargeter.command_timestamp_ns = 1_000_000_000
                retargeter.robot_points = retargeter.thumb_cmc = retargeter.fk = None
                retargeter.frame_ids = []
                retargeter.limits = np.tile([0., 1.], (12, 1))
                retargeter.optimizer = Mock()
                retargeter.optimizer.optimize.side_effect = lambda seed: seed
                retargeter.optimizer.last_optimize_result.return_value = 1
                retargeter.optimizer.get_numevals.return_value = 1
                objective = Mock(side_effect=[initial, final])
                with patch("dexmani_real.teleop.retargeting.adaptive.prepare_landmarks"), patch(
                    "dexmani_real.teleop.retargeting.adaptive.HandObjective", return_value=objective,
                ):
                    self.assertIsNone(retargeter.retarget(None, timestamp_ns=1_062_500_000))
                self.assertIsNotNone(retargeter.rejection_reason)
                np.testing.assert_array_equal(retargeter.previous_command, np.full(12, .5))
                self.assertEqual(retargeter.command_timestamp_ns, 1_000_000_000)
                key = "initial_loss" if not np.isfinite(initial) else "final_loss"
                self.assertIsNone(retargeter.last_diagnostics[key])
                self.assertTrue(retargeter.last_diagnostics["failure_reason"])
                with tempfile.TemporaryDirectory() as directory:
                    path = write_json(retargeter.last_diagnostics, Path(directory) / "failure.json")
                    self.assertIsNone(json.loads(path.read_text())[key])


class XHandConnectionTests(unittest.TestCase):
    def test_optional_identity_failures_are_outside_connect(self):
        hand = XHand(replace(HandConfig(), comm_type="ethercat"))
        hand.control = Mock()
        hand.control.list_hands_id.return_value = [hand.cfg.device_id]
        for name in ("get_sdk_version", "get_hand_type", "get_serial_number"):
            getattr(hand.control, name).side_effect = RuntimeError("optional identity unavailable")
        hand.open_device = Mock()
        hand.initialize_command = Mock(return_value=123)
        with patch.dict("sys.modules", {"xhand_controller": SimpleNamespace(xhand_control=object())}):
            self.assertEqual(hand.connect(), 123)
        hand.initialize_command.assert_called_once()
        for name in ("get_sdk_version", "get_hand_type", "get_serial_number"):
            getattr(hand.control, name).assert_not_called()


if __name__ == "__main__":
    unittest.main()
