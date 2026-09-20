"""Offline admission tests: technical checks only, no automatic quality gate.

Task T1 removed the "more than four consecutive FRAME_IK_FAIL rows reject the
whole episode" rule from ``analyze_episode``. Technically valid IK-hold rows
are source rows and stay admitted; the only whole-episode quality selection is
explicit operator curation (annotation ``include: false``). The provenance and
corruption gates are covered by ``test_processing_provenance.py``.

Like the provenance suite, this imports the processing runtime (numpy, scipy,
pinocchio, ...); when those are unavailable the suite is skipped rather than
reporting false passes.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py

try:
    from dexmani_real.dataset.contracts import EpisodeAnnotation
    from dexmani_real.dataset.processing import analyze_episode
    from dexmani_real.recording.storage.reader import EpisodeReader
    from tests.raw_episode_fixture import build_raw_episode

    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment guard
    EpisodeAnnotation = None
    analyze_episode = None
    EpisodeReader = None
    build_raw_episode = None
    _IMPORT_ERROR = exc

_FRAME_IK_FAIL = 2


@unittest.skipIf(
    _IMPORT_ERROR is not None,
    f"processing runtime dependencies unavailable: {_IMPORT_ERROR}",
)
class IKHoldAdmissionTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _build_ik_hold_episode(self, num_frames: int = 8) -> Path:
        """Build a valid fixture whose every row is an IK-failure hold."""
        episode = build_raw_episode(
            self.root / "ik_hold", num_frames=num_frames
        )
        with h5py.File(episode / "data.h5", "r+") as data:
            data["flag_frame_status"][:] = _FRAME_IK_FAIL
        return episode

    def test_persistent_ik_hold_rows_are_admitted(self):
        """>4 consecutive FRAME_IK_FAIL rows no longer reject the episode."""
        episode = self._build_ik_hold_episode(num_frames=8)
        with EpisodeReader(episode) as reader:
            decision = analyze_episode(reader, config=None)
        self.assertTrue(decision.accepted)
        self.assertIsNone(decision.rejected_reason)
        self.assertEqual(decision.source_frames, 8)
        # Source-row mapping is preserved: every row stays processable.
        self.assertEqual(decision.processed_frames, 8)

    def test_explicit_annotation_still_excludes(self):
        """Quality selection remains an explicit operator decision."""
        episode = self._build_ik_hold_episode(num_frames=8)
        with EpisodeReader(episode) as reader:
            decision = analyze_episode(
                reader,
                config=None,
                annotation=EpisodeAnnotation(include=False),
            )
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.rejected_reason, "excluded by annotation")
        self.assertEqual(decision.processed_frames, 0)


if __name__ == "__main__":
    unittest.main()
