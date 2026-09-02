"""Offline contracts for bounded learned-policy episode diagnostics."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from dexmani_real.deployment.metrics import (
    ENDPOINTS_PUBLISHED,
    PLAN_AGE_MS,
    Metrics,
)


class DeploymentMetricsTest(unittest.TestCase):
    def test_episode_counters_and_timings_survive_flush(self) -> None:
        metrics = Metrics()
        metrics.begin_episode(generation=3, started_monotonic_ns=1)
        metrics.increment(ENDPOINTS_PUBLISHED, 2)
        metrics.observe_timing(PLAN_AGE_MS, 1.0)

        metrics.flush()

        metrics.increment(ENDPOINTS_PUBLISHED)
        metrics.observe_timing(PLAN_AGE_MS, 5.0)

        self.assertEqual(metrics.snapshot()[ENDPOINTS_PUBLISHED], 1)
        self.assertEqual(
            metrics.episode_snapshot(),
            {
                ENDPOINTS_PUBLISHED: 3,
                f"{PLAN_AGE_MS}_samples": 2,
                f"{PLAN_AGE_MS}_p50": 1.0,
                f"{PLAN_AGE_MS}_p95": 5.0,
                f"{PLAN_AGE_MS}_p99": 5.0,
            },
        )

    def test_episode_timing_summary_is_bounded(self) -> None:
        metrics = Metrics()
        metrics.begin_episode(generation=3, started_monotonic_ns=1)

        for value in range(300):
            metrics.observe_timing(PLAN_AGE_MS, float(value))

        summary = metrics.episode_snapshot()
        self.assertEqual(summary[f"{PLAN_AGE_MS}_samples"], 256)
        self.assertEqual(summary[f"{PLAN_AGE_MS}_p50"], 171.0)
        self.assertEqual(summary[f"{PLAN_AGE_MS}_p95"], 287.0)
        self.assertEqual(summary[f"{PLAN_AGE_MS}_p99"], 297.0)

    def test_episode_summary_logs_once_without_receipt_artifact(self) -> None:
        metrics = Metrics()
        metrics.begin_episode(generation=7, started_monotonic_ns=1)
        metrics.increment(ENDPOINTS_PUBLISHED)

        with patch("dexmani_real.deployment.metrics.logger") as logger:
            metrics.log_episode_summary(status="STOPPED", reason="operator stop")
            metrics.log_episode_summary(status="STOPPED", reason="operator stop")

        logger.info.assert_called_once()
        message, generation, status, reason, duration_s, rendered = (
            logger.info.call_args.args
        )
        self.assertIn("episode summary", message)
        self.assertEqual(generation, 7)
        self.assertEqual(status, "STOPPED")
        self.assertEqual(reason, "operator stop")
        self.assertGreaterEqual(duration_s, 0.0)
        self.assertIn(f"{ENDPOINTS_PUBLISHED}=1", rendered)


if __name__ == "__main__":
    unittest.main()
