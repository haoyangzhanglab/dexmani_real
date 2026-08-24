from __future__ import annotations

import json
import unittest

from tools.check_architecture import (
    BASELINE_PATH,
    FORBIDDEN_EDGES,
    analyze_repository,
    check_against_baseline,
)


class ArchitectureBoundaryTest(unittest.TestCase):
    def test_ipc_cannot_depend_on_offline_data_package(self) -> None:
        self.assertIn(("ipc", "data"), FORBIDDEN_EDGES)

    def test_import_metrics_do_not_regress(self) -> None:
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        errors = check_against_baseline(analyze_repository(), baseline)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
