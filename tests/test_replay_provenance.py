"""Raw replay rejects irregular policy_eval timing without touching hardware."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


try:
    from dexmani_real.replay.trajectory import (
        load_trajectory,
    )
    from tests.raw_episode_fixture import build_raw_episode

    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment guard
    load_trajectory = None
    build_raw_episode = None
    _IMPORT_ERROR = exc


@unittest.skipIf(
    _IMPORT_ERROR is not None,
    f"replay runtime dependencies unavailable: {_IMPORT_ERROR}",
)
class ReplayProvenanceTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _build(self, workflow: str | None) -> Path:
        return build_raw_episode(
            self.root / (workflow or "legacy"),
            provenance_workflow=workflow,
        )

    def test_policy_eval_rejected_at_admission(self):
        episode = self._build("policy_eval")
        with self.assertRaises(ValueError) as ctx:
            load_trajectory(str(episode))
        self.assertIn(
            "policy_eval rollout contains synchronous irregular timing",
            str(ctx.exception),
        )
        self.assertIn("fixed-rate", str(ctx.exception))

    def test_unknown_workflow_fail_closed(self):
        episode = self._build("foo")
        with self.assertRaises(ValueError) as ctx:
            load_trajectory(str(episode))
        self.assertIn("unsupported provenance_workflow", str(ctx.exception))

    def test_legacy_absent_workflow_proceeds(self):
        trajectory = load_trajectory(str(self._build(None)))
        self.assertEqual(trajectory.num_frames, 3)
        self.assertGreater(trajectory.fps, 0.0)

    def test_explicit_teleop_proceeds(self):
        trajectory = load_trajectory(str(self._build("teleop")))
        self.assertEqual(trajectory.num_frames, 3)



if __name__ == "__main__":
    unittest.main()
