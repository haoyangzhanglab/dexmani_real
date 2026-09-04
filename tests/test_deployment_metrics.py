"""Offline contracts for the bounded learned-policy diagnostics."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from dexmani_real.deployment.metrics import PolicyStats


class DeploymentMetricsTest(unittest.TestCase):
    def test_snapshot_keeps_only_control_quality_diagnostics(self) -> None:
        stats = PolicyStats()
        stats.observe_inference_latency_ms(4.0)
        stats.observe_observation_age_ms(2.0)
        stats.observe_observation_skew_ms(0.5)
        stats.observe_schedule_lateness_ms(1.0)
        stats.observe_publication_interval_ms(62.5)
        stats.safety_rejection_count = 2
        stats.command_progress_timeout_count = 1
        stats.ik_rejection_count = 3
        stats.stale_prediction_count = 4

        self.assertEqual(
            stats.snapshot(),
            {
                "inference_latency_ms": 4.0,
                "observation_age_ms": 2.0,
                "observation_skew_ms": 0.5,
                "schedule_lateness_ms": 1.0,
                "publication_interval_ms": 62.5,
                "safety_rejection_count": 2,
                "command_progress_timeout_count": 1,
                "ik_rejection_count": 3,
                "stale_prediction_count": 4,
            },
        )

    def test_flush_resets_counts_but_retains_bounded_recent_timings(self) -> None:
        stats = PolicyStats()
        stats.observe_publication_interval_ms(1.0)
        stats.safety_rejection_count = 1

        with patch("dexmani_real.deployment.metrics.logger") as logger:
            stats.flush(prefix="executor metrics", debug=True)

        logger.debug.assert_called_once()
        self.assertEqual(stats.safety_rejection_count, 0)
        self.assertEqual(stats.snapshot()["publication_interval_ms"], 1.0)


if __name__ == "__main__":
    unittest.main()
