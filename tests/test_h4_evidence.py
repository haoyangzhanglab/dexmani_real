"""Offline sealing checks for H4 runtime evidence."""

from __future__ import annotations

import json
import runpy
import tempfile
import unittest
from pathlib import Path


class H4EvidenceSealTest(unittest.TestCase):
    def test_completed_runtime_receipt_and_terminal_log_are_sealed(self) -> None:
        main = runpy.run_path(
            str(Path(__file__).resolve().parents[1] / "examples/seal_h4_evidence.py")
        )["main"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt_path = root / "h4_execute.json"
            receipt_path.write_text(
                json.dumps(
                    {
                        "execution_mode": "execute",
                        "completed": True,
                        "max_published_endpoints": 1,
                        "coupled_command_writes": 1,
                        "within_publication_bound": True,
                        "acknowledged_action_id": 7,
                        "run_generation": 4,
                        "metrics": {"physical_home_completed": 1},
                        "timeline_monotonic_ns": {
                            "run_started": 10,
                            "first_publication": 20,
                            "last_publication": 20,
                            "receipt_emitted": 30,
                        },
                    }
                ),
                encoding="utf-8",
            )
            terminal_log = root / "terminal.log"
            terminal_log.write_text(
                "\n".join(
                    (
                        "hand_loop: startup reset-home command accepted",
                        "hand: home command accepted (action_id=3)",
                        "arm: home path selected=canonical milestones=1",
                        "arm: home reached",
                        "operator: physical home sequence completed; press B to start",
                        "coordinator_loop: RUNNING (run_generation=4)",
                        "coordinator: execute published action_id=7 (1/1); awaiting worker acknowledgement",
                        "coordinator: policy run stopped: H4 publication bound reached",
                        "arm_loop: exited (servo_calls=1, duplicate_skips=0)",
                        "hand_loop: exited (sdk_send_attempts=1, exact_target_accepts=1, crc_unconfirmed=0, duplicate_skips=0, sdk_rejections=0)",
                        "── Session End ──",
                        "safety=DISARMED supervisor_normal=True execute_completed=True clean=True",
                    )
                ),
                encoding="utf-8",
            )
            operator_record = root / "operator.json"
            operator_record.write_text(
                json.dumps(
                    {
                        "operator": "tester",
                        "scene": "clear workspace",
                        "e_stop_ready": True,
                        "authorization": "bounded H4",
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                main(
                    [
                        "--runtime-receipt",
                        str(receipt_path),
                        "--terminal-log",
                        str(terminal_log),
                        "--operator-record",
                        str(operator_record),
                        "--output-dir",
                        str(root),
                    ]
                ),
                0,
            )
            manifests = list(root.glob("h4_evidence_*.json"))
            self.assertEqual(len(manifests), 1)
            manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["worker_command_counts"]["arm"]["sdk_servo_calls"], 1
            )
            self.assertEqual(
                manifest["worker_command_counts"]["hand"]["exact_target_accepts"], 1
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
