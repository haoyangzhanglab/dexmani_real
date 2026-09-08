"""Offline regressions for the payload-independent causal tactile selector.

No hardware and no HDF5 input: the selector consumes pure provenance arrays.
These tests pin causal selection (latest proven source <= reference), the skew
bound, the provenance gates (fresh/calibrated/unit/source match), non-monotonic
source clocks, the strict prefix restriction (a later persisted row can never
repair an earlier observation), and forward-fill duplication.  Run with:

    python -m unittest discover -s tests -p 'test_tactile_selector.py'
"""

from __future__ import annotations

import unittest

import numpy as np

from dexmani_real.dataset.clean import select_tactile_rows_to_references

_MAX_SKEW_S = 0.1


def _provenance(
    source_ns: list[int],
    *,
    hand_source_ns: list[int] | None = None,
    fresh: list[bool] | None = None,
    calibrated: list[bool] | None = None,
    unit_code: list[int] | None = None,
    references: list[int] | None = None,
) -> dict[str, np.ndarray]:
    count = len(source_ns)
    return {
        "hand_source_monotonic_ns": np.asarray(
            source_ns if hand_source_ns is None else hand_source_ns, dtype=np.int64
        ),
        "tactile_source_monotonic_ns": np.asarray(source_ns, dtype=np.int64),
        "tactile_fresh": np.asarray(
            [True] * count if fresh is None else fresh, dtype=bool
        ),
        "tactile_calibrated": np.asarray(
            [True] * count if calibrated is None else calibrated, dtype=bool
        ),
        "tactile_unit_code": np.asarray(
            [0] * count if unit_code is None else unit_code, dtype=np.int64
        ),
        "reference_monotonic_ns": np.asarray(
            source_ns if references is None else references, dtype=np.int64
        ),
    }


def _select(kwargs: dict[str, np.ndarray]) -> np.ndarray:
    return select_tactile_rows_to_references(
        **kwargs, max_observation_skew_s=_MAX_SKEW_S
    )


class TestTactileSelectorCausality(unittest.TestCase):
    def test_identity_selection_when_source_equals_reference(self) -> None:
        selected = _select(_provenance([1_000, 2_000, 3_000]))
        np.testing.assert_array_equal(selected, np.array([0, 1, 2]))

    def test_selects_latest_source_not_after_reference(self) -> None:
        # Row 2's reference sits after rows 0 and 1 are proven; row 1 has the
        # newest source at or before the reference.
        selected = _select(
            _provenance(
                [100, 200, 300],
                references=[100, 200, 250],
            )
        )
        np.testing.assert_array_equal(selected, np.array([0, 1, 1]))

    def test_skew_bound_rejects_old_source(self) -> None:
        # Row 1's reference is 150ms after the only proven source (100ms cap).
        selected = _select(
            _provenance(
                [100, 200],
                references=[100, 100 + int(1.5 * _MAX_SKEW_S * 1e9)],
                fresh=[True, False],
            )
        )
        self.assertEqual(int(selected[0]), 0)
        self.assertEqual(int(selected[1]), -1)

    def test_skew_bound_accepts_source_at_the_bound(self) -> None:
        reference = 100 + int(_MAX_SKEW_S * 1e9)
        selected = _select(
            _provenance([100, 200], references=[100, reference], fresh=[True, False])
        )
        self.assertEqual(int(selected[1]), 0)


class TestTactileSelectorProvenanceGates(unittest.TestCase):
    def test_not_fresh_rejected(self) -> None:
        selected = _select(_provenance([100], fresh=[False], references=[100]))
        np.testing.assert_array_equal(selected, np.array([-1]))

    def test_not_calibrated_rejected(self) -> None:
        selected = _select(_provenance([100], calibrated=[False], references=[100]))
        np.testing.assert_array_equal(selected, np.array([-1]))

    def test_nonzero_unit_code_rejected(self) -> None:
        selected = _select(_provenance([100], unit_code=[3], references=[100]))
        np.testing.assert_array_equal(selected, np.array([-1]))

    def test_hand_source_mismatch_rejected(self) -> None:
        selected = _select(
            _provenance([100], hand_source_ns=[999], references=[100])
        )
        np.testing.assert_array_equal(selected, np.array([-1]))

    def test_nonpositive_source_rejected(self) -> None:
        selected = _select(_provenance([0], references=[100]))
        np.testing.assert_array_equal(selected, np.array([-1]))

    def test_nonpositive_reference_rejected(self) -> None:
        selected = _select(_provenance([100], references=[0]))
        np.testing.assert_array_equal(selected, np.array([-1]))


class TestTactileSelectorOrdering(unittest.TestCase):
    def test_duplicate_source_uses_latest_proven_persisted_row(self) -> None:
        selected = _select(
            _provenance(
                [300, 100, 300, 300],
                references=[0, 350, 350, 350],
                fresh=[True, True, True, False],
            )
        )
        np.testing.assert_array_equal(selected, [-1, 0, 2, 2])

    def test_non_monotonic_source_timestamps(self) -> None:
        # Source clock steps backward at row 1.  Selection is timestamp-based:
        # the reference at row 1 (350) must select the newest proven source
        # at or before it (300 at row 0), not the latest persisted row.
        selected = _select(
            _provenance([300, 100], references=[300, 350])
        )
        np.testing.assert_array_equal(selected, np.array([0, 0]))

    def test_later_persisted_row_cannot_repair_earlier_observation(self) -> None:
        # Row 0's reference (1000) is after row 1's source timestamp (900),
        # but row 1 was not yet persisted at row 0, so row 0 stays invalid.
        selected = _select(
            _provenance(
                [0, 900],
                fresh=[False, True],
                references=[1000, 1000],
            )
        )
        np.testing.assert_array_equal(selected, np.array([-1, 1]))

    def test_forward_fill_duplicates_the_source_row(self) -> None:
        # Rows 2 and 3 are not fresh; their references stay within skew of the
        # row-1 source, so both forward-fill to the same raw row.
        selected = _select(
            _provenance(
                [100, 200, 300, 400],
                fresh=[True, True, False, False],
                references=[100, 200, 240, 260],
            )
        )
        np.testing.assert_array_equal(selected, np.array([0, 1, 1, 1]))

    def test_reference_before_any_proven_source(self) -> None:
        selected = _select(_provenance([500], references=[400]))
        np.testing.assert_array_equal(selected, np.array([-1]))


class TestTactileSelectorInputValidation(unittest.TestCase):
    def test_mismatched_provenance_shapes_rejected(self) -> None:
        kwargs = _provenance([100, 200])
        kwargs["tactile_fresh"] = np.asarray([True], dtype=bool)
        with self.assertRaises(ValueError):
            _select(kwargs)

    def test_invalid_skew_rejected(self) -> None:
        kwargs = _provenance([100])
        with self.assertRaises(ValueError):
            select_tactile_rows_to_references(**kwargs, max_observation_skew_s=0.0)
        with self.assertRaises(ValueError):
            select_tactile_rows_to_references(**kwargs, max_observation_skew_s=np.nan)

    def test_output_dtype_and_shape(self) -> None:
        selected = _select(_provenance([100, 200, 300]))
        self.assertEqual(selected.dtype, np.int64)
        self.assertEqual(selected.shape, (3,))


def test_randomized_small_arrays_match_prefix_scan():
    rng = np.random.default_rng(20260908)
    for count in range(65):
        for _ in range(8):
            # Small timestamp domain deliberately produces duplicates and reversals.
            sources = rng.integers(-2, 13, size=count) * 1_000_000
            hand_sources = sources + rng.choice([0, 0, 0, 1], size=count)
            fresh = rng.random(count) < 0.8
            calibrated = rng.random(count) < 0.8
            units = rng.choice([0, 0, 0, 1, 3], size=count)
            references = rng.integers(-2, 18, size=count) * 1_000_000
            max_skew_ns = int(rng.integers(1, 8)) * 1_000_000
            expected = np.full(count, -1, dtype=np.int64)
            for row, reference in enumerate(references):
                candidates = [
                    (int(sources[candidate]), candidate)
                    for candidate in range(row + 1)
                    if fresh[candidate]
                    and calibrated[candidate]
                    and units[candidate] == 0
                    and hand_sources[candidate] == sources[candidate]
                    and 0 < sources[candidate] <= reference
                    and reference - sources[candidate] <= max_skew_ns
                ]
                if candidates:
                    expected[row] = max(candidates)[1]
            actual = select_tactile_rows_to_references(
                hand_sources,
                sources,
                fresh,
                calibrated,
                units,
                references,
                max_observation_skew_s=max_skew_ns / 1e9,
            )
            np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
