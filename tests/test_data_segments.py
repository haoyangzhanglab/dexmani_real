"""Offline checks for source-contiguous policy episode boundaries."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from dexmani_real.data.clean import _quality_summary
from dexmani_real.data.contracts import EpisodeDecision, OutputProfile


class DataSegmentTest(unittest.TestCase):
    def test_decision_maps_source_gap_to_compact_segment_end(self) -> None:
        decision = EpisodeDecision(
            source_path=Path("episode/data.h5"),
            source_frames=6,
            profile=OutputProfile.JOINT,
            selected_indices=np.asarray([0, 1, 4, 5], dtype=np.int64),
            keep_mask=np.asarray([1, 1, 0, 0, 1, 1], dtype=bool),
            drop_reason_bits=np.asarray([0, 0, 1, 1, 0, 0], dtype=np.uint64),
            drop_reason_names=("invalid",),
            hard_reason_counts={"invalid": 2},
            boundary_counts={},
            selected_frames=4,
            quality={},
            source_gap_findings=({"source_row_before": 1, "source_row_after": 4},),
        )

        np.testing.assert_array_equal(decision.segment_ends, [2, 4])
        self.assertEqual(decision.to_dict()["selected_segment_ends"], [2, 4])

    def test_full_windows_never_cross_source_segments(self) -> None:
        arrays = {
            "action_arm": np.zeros((6, 7), dtype=np.float64),
            "action_hand": np.zeros((6, 12), dtype=np.float64),
            "tracking_error": np.zeros(6, dtype=np.float64),
        }

        quality = _quality_summary(
            arrays,
            np.asarray([0, 1, 4, 5], dtype=np.int64),
            grid_dt_s=0.1,
            tracking_error_warn_rad=0.2,
            horizon=3,
            segment_ends=np.asarray([2, 4], dtype=np.int64),
        )

        self.assertEqual(quality["full_window_count"], 0)


if __name__ == "__main__":
    unittest.main()
