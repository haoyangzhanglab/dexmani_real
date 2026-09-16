"""Offline integration tests for the policy_eval processing provenance gate.

These exercise the real ``analyze_episode`` admission boundary against a
minimal-but-valid schema-v29 raw episode fixture (``tests.raw_episode_fixture``).
They never open hardware or an SDK, but they do import the processing runtime
(numpy, scipy, pinocchio, ...); when those are unavailable the suite is skipped
rather than reporting false passes.

The gate under test is the *admission* classifier: a ``policy_eval`` (or any
unknown explicit) ``provenance_workflow`` must fail closed before any semantic
relabeling, RGB decode, FK, or output publication, while the current teleop
path (absent or explicit ``teleop``) must keep passing through.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py

try:
    from dexmani_real.dataset.processing import analyze_episode
    from dexmani_real.recording.storage.reader import EpisodeReader
    from tests.raw_episode_fixture import build_raw_episode

    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment guard
    analyze_episode = None
    EpisodeReader = None
    build_raw_episode = None
    _IMPORT_ERROR = exc


@unittest.skipIf(
    _IMPORT_ERROR is not None,
    f"processing runtime dependencies unavailable: {_IMPORT_ERROR}",
)
class ProcessingProvenanceTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _analyze(self, workflow: str | None):
        episode = build_raw_episode(
            self.root / (workflow or "legacy"),
            provenance_workflow=workflow,
        )
        with EpisodeReader(episode) as reader:
            # ``config`` is unused by admission; the gate is config-independent.
            return analyze_episode(reader, config=None)

    def test_policy_eval_rejected_at_admission(self):
        with self.assertRaises(ValueError) as ctx:
            self._analyze("policy_eval")
        self.assertIn("policy_eval rollout has synchronous", str(ctx.exception))

    def test_unknown_workflow_fail_closed(self):
        with self.assertRaises(ValueError) as ctx:
            self._analyze("foo")
        self.assertIn("unsupported provenance_workflow", str(ctx.exception))

    def test_legacy_absent_workflow_proceeds(self):
        self.assertTrue(self._analyze(None).accepted)

    def test_explicit_teleop_proceeds(self):
        self.assertTrue(self._analyze("teleop").accepted)

    def test_policy_eval_rejection_precedes_downstream_admission(self):
        # Corrupt a semantic field that downstream admission would otherwise
        # reject (non-increasing timestamps). The provenance gate must fire
        # first, before the dataset/semantic checks ever run.
        episode = build_raw_episode(
            self.root / "policy_eval_corrupt", provenance_workflow="policy_eval"
        )
        with h5py.File(episode / "data.h5", "r+") as data:
            data["timestamp"][:] = 0.0
        with EpisodeReader(episode) as reader:
            with self.assertRaises(ValueError) as ctx:
                analyze_episode(reader, config=None)
        self.assertIn("policy_eval rollout has synchronous", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
