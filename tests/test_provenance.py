"""Offline tests for raw-episode workflow provenance classification."""

from __future__ import annotations

import unittest

from dexmani_real.dataset.provenance import (
    read_provenance_workflow,
    supports_fixed_dt_teleop,
)


class ProvenanceWorkflowTest(unittest.TestCase):
    def test_supports_fixed_dt_teleop(self):
        self.assertTrue(supports_fixed_dt_teleop(None))
        self.assertTrue(supports_fixed_dt_teleop(""))
        self.assertTrue(supports_fixed_dt_teleop("teleop"))
        self.assertFalse(supports_fixed_dt_teleop("policy_eval"))
        self.assertFalse(supports_fixed_dt_teleop("unknown_workflow"))

    def test_read_provenance_workflow_absent(self):
        self.assertIsNone(read_provenance_workflow(None))
        self.assertIsNone(read_provenance_workflow({}))
        self.assertIsNone(read_provenance_workflow({"task_label": "x"}))

    def test_read_provenance_workflow_normalizes(self):
        self.assertEqual(
            read_provenance_workflow({"provenance_workflow": "policy_eval"}),
            "policy_eval",
        )
        self.assertEqual(
            read_provenance_workflow({"provenance_workflow": "  teleop  "}),
            "teleop",
        )
        self.assertIsNone(read_provenance_workflow({"provenance_workflow": ""}))
        self.assertIsNone(read_provenance_workflow({"provenance_workflow": "   "}))
        self.assertEqual(
            read_provenance_workflow({"provenance_workflow": b"policy_eval"}),
            "policy_eval",
        )


if __name__ == "__main__":
    unittest.main()
