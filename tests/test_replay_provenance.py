"""Offline integration tests for the policy_eval replay provenance gate.

These exercise the real ``load_trajectory`` raw-admission boundary (and its
processed-entry funnel) against a minimal-but-valid schema-v29 raw episode
fixture (``tests.raw_episode_fixture``). They never open hardware or an SDK,
but they do import the replay/planning runtime (scipy, pinocchio, ...); when
those are unavailable the suite is skipped rather than reporting false passes.

The contract under test: ``policy_eval`` (or any unknown explicit)
``provenance_workflow`` must fail closed for fixed-rate physical replay, while
the current teleop path (absent or explicit ``teleop``) keeps entering the
normal replay load path. The processed-entry path has no second provenance
classifier — its ``source_path`` funnels into the same raw gate.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py

try:
    import dexmani_real.replay.trajectory as traj_module
    from dexmani_real.replay.trajectory import (
        load_processed_trajectory,
        load_trajectory,
    )
    from tests.raw_episode_fixture import build_raw_episode

    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment guard
    traj_module = None
    load_trajectory = None
    load_processed_trajectory = None
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

    def test_processed_entry_cannot_bypass_raw_gate(self):
        # A processed artifact funnels to load_trajectory(source_path); there is
        # no second provenance classifier on the processed side. We stub only
        # the orthogonal processed-schema validator (validate_processed_hdf5),
        # whose full multimodal payload is disproportionate to build here, and
        # leave load_trajectory's raw provenance gate fully real.
        raw_source = self._build("policy_eval")
        processed = self.root / "processed.h5"
        with h5py.File(processed, "w") as artifact:
            artifact.attrs["source_path"] = str(raw_source)
        with mock.patch.object(
            traj_module,
            "validate_processed_hdf5",
            return_value={"path": "processed.h5", "frames": 3, "keys": []},
        ):
            with mock.patch.object(
                traj_module, "load_trajectory", wraps=traj_module.load_trajectory
            ) as load_spy:
                with self.assertRaises(ValueError) as ctx:
                    load_processed_trajectory(str(processed))
            # Provenance enforcement must be delegated to the raw loader.
            load_spy.assert_called_once_with(str(raw_source))
        # The rejection must come from the RAW gate, not the processed path.
        self.assertIn(
            "policy_eval rollout contains synchronous irregular timing",
            str(ctx.exception),
        )


if __name__ == "__main__":
    unittest.main()
