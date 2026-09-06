"""Pure regressions for replay consistency evaluation."""

from __future__ import annotations

import unittest

import numpy as np

from dexmani_real.replay.evaluation import compute_metrics


class ReplayEvaluationTest(unittest.TestCase):
    @staticmethod
    def _tracking_lag_frames(
        original_arm_qpos: np.ndarray,
        replay_arm_qpos: np.ndarray,
    ) -> int:
        frame_count = original_arm_qpos.shape[0]
        metrics = compute_metrics(
            original_arm_qpos=original_arm_qpos,
            replay_arm_qpos=replay_arm_qpos,
            original_arm_ee=None,
            replay_arm_ee_pos=np.zeros((frame_count, 3), dtype=np.float64),
            replay_arm_ee_rot6d=np.zeros((frame_count, 6), dtype=np.float64),
            fps=16.0,
        )
        return metrics.tracking_lag_frames

    def test_constant_identical_trajectory_has_zero_tracking_lag(self) -> None:
        trajectory = np.zeros((40, 7), dtype=np.float64)

        self.assertEqual(self._tracking_lag_frames(trajectory, trajectory), 0)

    def test_known_delayed_trajectory_reports_its_delay(self) -> None:
        delay_frames = 3
        original = np.arange(40, dtype=np.float64)[:, None] + np.arange(7)
        replayed = np.full_like(original, np.nan)
        replayed[delay_frames:] = original[:-delay_frames]

        self.assertEqual(self._tracking_lag_frames(original, replayed), delay_frames)

    def test_exact_rmse_tie_prefers_smallest_absolute_lag(self) -> None:
        # A period-three trajectory has exact matches at -2 and +1 frames.
        # The minimum-RMSE tie must choose +1 because abs(+1) < abs(-2).
        original = np.tile((np.arange(42, dtype=np.float64) % 3)[:, None], (1, 7))
        replayed = np.roll(original, shift=1, axis=0)

        self.assertEqual(self._tracking_lag_frames(original, replayed), 1)


if __name__ == "__main__":
    unittest.main()
