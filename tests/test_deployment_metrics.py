"""Pure bounded observability and shadow-receipt contracts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dexmani_real.deployment.metrics import (
    Metrics,
    execute_run_receipt_json,
    shadow_run_receipt_json,
)
from dexmani_real.utils.log import write_json_receipt


class DeploymentMetricsTest(unittest.TestCase):
    def test_timing_window_is_bounded_and_uses_nearest_rank_quantiles(self) -> None:
        metrics = Metrics()
        for value in range(260):
            metrics.observe_timing("inference_ms", float(value))

        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["inference_ms_samples"], 256)
        self.assertEqual(snapshot["inference_ms_p50"], 131.0)
        self.assertEqual(snapshot["inference_ms_p95"], 247.0)
        self.assertEqual(snapshot["inference_ms_p99"], 257.0)

    def test_flush_resets_interval_counter_but_not_run_receipt_totals(self) -> None:
        metrics = Metrics()
        metrics.begin_run()
        metrics.increment("endpoints_shadow_validated", 2)
        metrics.observe("plan_age_ms", 5.0)
        metrics.observe_timing("usable_horizon_ms", 10.0)
        metrics.flush()

        self.assertEqual(metrics.snapshot()["endpoints_shadow_validated"], 0)
        run_snapshot = metrics.run_snapshot()
        self.assertEqual(run_snapshot["endpoints_shadow_validated"], 2)
        self.assertEqual(run_snapshot["usable_horizon_ms_samples"], 1)

        metrics.begin_run()
        self.assertNotIn("plan_age_ms", metrics.run_snapshot())
        self.assertNotIn("usable_horizon_ms_samples", metrics.run_snapshot())
        self.assertNotIn("endpoints_shadow_validated", metrics.snapshot())

    def test_shadow_receipt_exposes_zero_write_invariant(self) -> None:
        receipt = json.loads(
            shadow_run_receipt_json(
                run_generation=7,
                reason="operator stop",
                coupled_command_start_sequence=13,
                coupled_command_end_sequence=13,
                metrics={"endpoints_shadow_validated": 4, "inference_ms_p95": 9.5},
            )
        )

        self.assertEqual(receipt["execution_mode"], "shadow")
        self.assertEqual(receipt["coupled_command_writes"], 0)
        self.assertTrue(receipt["zero_coupled_command_writes"])
        self.assertEqual(receipt["metrics"]["endpoints_shadow_validated"], 4)

        violation = json.loads(
            shadow_run_receipt_json(
                run_generation=7,
                reason="shadow coupled-command write detected",
                coupled_command_start_sequence=13,
                coupled_command_end_sequence=14,
                metrics={"shadow_coupled_write_violations": 1},
            )
        )
        self.assertEqual(violation["coupled_command_writes"], 1)
        self.assertFalse(violation["zero_coupled_command_writes"])

    def test_shadow_receipt_rejects_backward_sequence_and_nonfinite_metric(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "sequences"):
            shadow_run_receipt_json(
                run_generation=1,
                reason="stop",
                coupled_command_start_sequence=2,
                coupled_command_end_sequence=1,
                metrics={},
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            shadow_run_receipt_json(
                run_generation=1,
                reason="stop",
                coupled_command_start_sequence=2,
                coupled_command_end_sequence=2,
                metrics={"inference_ms_p95": float("nan")},
            )

    def test_execute_receipt_exposes_one_publication_and_acknowledgement(self) -> None:
        receipt = json.loads(
            execute_run_receipt_json(
                run_generation=8,
                reason="H4 publication bound reached",
                coupled_command_start_sequence=20,
                coupled_command_end_sequence=21,
                max_published_endpoints=1,
                acknowledgement_timeout_s=2.0,
                acknowledged_action_id=41,
                completed=True,
                metrics={
                    "coupled_command_writes": 1,
                    "execute_acknowledged": 1,
                },
            )
        )

        self.assertEqual(receipt["execution_mode"], "execute")
        self.assertEqual(receipt["coupled_command_writes"], 1)
        self.assertTrue(receipt["within_publication_bound"])
        self.assertEqual(receipt["acknowledged_action_id"], 41)
        self.assertTrue(receipt["completed"])
        self.assertEqual(receipt["outcome"], "completed")

    def test_execute_receipt_rejects_non_h4_bound(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one"):
            execute_run_receipt_json(
                run_generation=8,
                reason="stop",
                coupled_command_start_sequence=20,
                coupled_command_end_sequence=20,
                max_published_endpoints=2,
                acknowledgement_timeout_s=2.0,
                acknowledged_action_id=None,
                completed=False,
                metrics={},
            )

    def test_h4_receipt_is_written_as_one_json_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_json_receipt(directory, '{"outcome":"completed"}')
            self.assertEqual(path.parent, Path(directory))
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")), {"outcome": "completed"}
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
