"""Offline tests for teleop backpressure-span accounting (task T2, V05/V19).

A teleop producer can hold a prepared-but-uncommitted candidate while the
ordered command FIFO is full. Every pause boundary that discards that
candidate must close the visible ``[WAIT]`` span with its own ``[DROP]``:
otherwise the span outlives its candidate, the next ``[WAIT]`` is suppressed,
and the following ``[RESUME]`` reports a wait that never resumed.

The real ``PublishWaitTracker`` and the real ``close_publish_span`` helper are
exercised against a minimal controller/resources pair; no hardware, SDK, or
planner is involved.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

# Only a genuinely missing dependency may skip this module: a renamed or
# broken symbol must fail the suite rather than hide behind a skip.
try:
    from dexmani_real.control.publication import PublishWaitTracker
    from dexmani_real.teleop.control_loop.grid import close_publish_span

    _IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - environment guard
    PublishWaitTracker = None
    close_publish_span = None
    _IMPORT_ERROR = exc


@unittest.skipIf(
    _IMPORT_ERROR is not None, f"dependencies unavailable: {_IMPORT_ERROR}"
)
class PauseBoundarySpanTest(unittest.TestCase):
    def _controller(self, action_id: int):
        return SimpleNamespace(
            pending_publish=SimpleNamespace(
                candidate=SimpleNamespace(action_id=action_id)
            )
        )

    def test_pause_boundary_drops_the_retained_candidate_visibly(self):
        tracker = PublishWaitTracker("teleop")
        tracker.note_full(8, 813)
        self.assertTrue(tracker.waiting)
        controller = self._controller(813)

        with self.assertLogs(
            "dexmani_real.teleop.control_loop.grid", level="WARNING"
        ) as logs:
            close_publish_span(
                controller, SimpleNamespace(fifo_wait=tracker), "pause_boundary"
            )

        text = "\n".join(logs.output)
        self.assertIn("[DROP] teleop action=813", text)
        self.assertIn("pause_boundary", text)
        self.assertIsNone(controller.pending_publish)
        self.assertFalse(tracker.waiting)
        # The span is closed, so the next successful commit reports no wait
        # that never resumed.
        with self.assertNoLogs("dexmani_real.control.publication", level="INFO"):
            tracker.note_committed()

    def test_boundary_without_a_candidate_still_closes_a_stale_span(self):
        tracker = PublishWaitTracker("teleop")
        tracker.note_full(8, 901)
        controller = SimpleNamespace(pending_publish=None)

        close_publish_span(
            controller, SimpleNamespace(fifo_wait=tracker), "pause_boundary"
        )
        self.assertFalse(tracker.waiting)
        # A later FULL span is announced again instead of being swallowed.
        with self.assertLogs("dexmani_real.control.publication", level="WARNING") as logs:
            tracker.note_full(8, 902)
        self.assertIn("keep_action=902", "\n".join(logs.output))

    def test_boundary_with_no_span_is_a_no_op(self):
        tracker = PublishWaitTracker("teleop")
        controller = SimpleNamespace(pending_publish=None)
        with self.assertNoLogs("dexmani_real.control.publication", level="WARNING"):
            close_publish_span(
                controller, SimpleNamespace(fifo_wait=tracker), "pause_release"
            )
        self.assertFalse(tracker.waiting)


if __name__ == "__main__":
    unittest.main()
