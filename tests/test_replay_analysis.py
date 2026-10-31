import contextlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from dexmani_real.replay import run
from dexmani_real.replay.joint_replay import ReplayResult, ReplayStatus
from dexmani_real.replay.metrics import compute_metrics
from dexmani_real.replay.trajectory import Trajectory
from scripts.evaluate_replay import main as evaluate_replay


class ReplayAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.arm = np.zeros((2, 7))
        self.hand = np.zeros((2, 12))
        self.trajectory = Trajectory(
            "source_episode", 16.0, "test", self.arm, self.hand, self.arm, self.hand
        )
        self.measurements = {"arm_qpos": self.arm, "hand_qpos": self.hand}

    def run_without_hardware(self, output, *, save_error=None):
        state = SimpleNamespace(
            run_signal=None, stop_signal=None, emergency_signal=None, motion_lock=None,
            error_state=False, estop_request=False, arm=mock.Mock(),
            request_emergency_stop=mock.Mock(), request_quit=mock.Mock(),
            request_stop=mock.Mock(),
        )
        runtime = SimpleNamespace(
            arm=None, hand=None,
            environment=SimpleNamespace(table=SimpleNamespace(enabled=False)),
        )
        outcome = ReplayResult(ReplayStatus.COMPLETED, self.measurements)
        with contextlib.ExitStack() as stack:
            for name in (
                "configure_logging", "validate_robot_config", "validate_control_rate",
                "validate_trajectory", "HomePlanner", "Robot", "Keyboard", "wait_for_home",
            ):
                stack.enter_context(mock.patch.object(run, name))
            stack.enter_context(mock.patch.object(run, "SessionState", return_value=state))
            stack.enter_context(mock.patch.object(run, "close_runtime", return_value=True))
            replay = stack.enter_context(mock.patch.object(run, "JointReplay"))
            replay.return_value.run.return_value = outcome
            analysis = stack.enter_context(mock.patch(
                "dexmani_real.replay.metrics.compute_metrics", side_effect=RuntimeError("FK unavailable")
            ))
            if save_error is not None:
                stack.enter_context(mock.patch.object(run, "save_replay_data", side_effect=save_error))
            result = run.run_replay(self.trajectory, runtime, output_dir=output)
            analysis.assert_not_called()
        return result

    def test_control_result_and_samples_do_not_depend_on_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = self.run_without_hardware(output)
            self.assertIs(result.status, ReplayStatus.COMPLETED)
            summary = json.loads((output / "session_result.json").read_text())
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["trajectory_status"], "completed")
            self.assertFalse((output / "metrics.json").exists())
            with np.load(output / "replay_data.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["arm_qpos"], self.arm)

    def test_sample_write_failure_is_still_a_fault(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_without_hardware(directory, save_error=OSError("disk full"))
            self.assertIs(result.status, ReplayStatus.FAULT)
            self.assertIn("disk full", result.reason)
            summary = json.loads((Path(directory) / "session_result.json").read_text())
            self.assertEqual(summary["trajectory_status"], "completed")

    def test_failed_offline_analysis_leaves_control_result_intact(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            control_result = output / "session_result.json"
            control_result.write_text('{"status":"completed","trajectory_status":"completed"}')
            original = control_result.read_bytes()
            np.savez(output / "replay_data.npz", **self.measurements)
            with mock.patch(
                "dexmani_real.replay.trajectory.load_trajectory", return_value=self.trajectory
            ), mock.patch(
                "dexmani_real.replay.metrics.compute_metrics", side_effect=RuntimeError("FK unavailable")
            ):
                with self.assertRaisesRegex(RuntimeError, "FK unavailable"):
                    evaluate_replay(["--episode", "source_episode", "--replay", directory])
            self.assertEqual(control_result.read_bytes(), original)
            self.assertFalse((output / "metrics.json").exists())

    def test_invalid_external_measurements_are_rejected(self):
        for arm, hand in (
            (np.full((2, 7), np.nan), self.hand),
            (self.arm, np.zeros((1, 12))),
            (self.arm, np.zeros((2, 11))),
        ):
            with self.subTest(arm_shape=arm.shape, hand_shape=hand.shape):
                with self.assertRaises(ValueError):
                    compute_metrics(self.trajectory, {"arm_qpos": arm, "hand_qpos": hand})

    def test_identical_measured_trajectories_have_zero_error(self):
        metrics = compute_metrics(self.trajectory, self.measurements)
        self.assertEqual(metrics["matching_frames"], 2)
        self.assertEqual(metrics["arm_joint"]["pooled_rmse_deg"], 0)
        self.assertEqual(metrics["hand_joint"]["pooled_rmse_deg"], 0)
        self.assertEqual(metrics["eef_position"]["max_error_mm"], 0)
        self.assertLessEqual(metrics["eef_orientation"]["max_error_deg"], 2e-6)


if __name__ == "__main__":
    unittest.main()
