"""Offline regressions for the visualizer EEF and fingertip helpers.

No Rerun GUI is started and no hardware is touched: these tests import the two
example visualizer modules by path and pin their pure render-admission helpers
— finite raw ``arm_ee`` / processed ``eef_pose`` rows yield the position and
finite ``fingertip`` rows yield the (5,3) positions, while invalid rows yield
None (the clear-entity path that prevents stale spheres), and the processed
time-series labels are the frozen ``ee_x..ee_r5`` set.  Run with:

    python -m unittest discover -s tests -p 'test_visualize_eef_helpers.py'
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_example_module(name: str):
    path = _REPO_ROOT / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"examples_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_raw_viz = _load_example_module("visualize_episode")
_processed_viz = _load_example_module("visualize_episode_processed")


def _valid_eef_row() -> np.ndarray:
    row = np.zeros(9)
    row[:3] = (0.3, -0.1, 0.25)
    row[3:9] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    return row


class TestRawEefHelper(unittest.TestCase):
    def test_finite_row_returns_position(self) -> None:
        position = _raw_viz._eef_position_or_none(_valid_eef_row())
        self.assertIsNotNone(position)
        np.testing.assert_allclose(position, (0.3, -0.1, 0.25))

    def test_nan_sentinel_row_returns_none(self) -> None:
        self.assertIsNone(_raw_viz._eef_position_or_none(np.full(9, np.nan)))

    def test_partially_invalid_row_returns_none(self) -> None:
        row = _valid_eef_row()
        row[1] = np.inf
        self.assertIsNone(_raw_viz._eef_position_or_none(row))

    def test_wrong_shape_returns_none(self) -> None:
        self.assertIsNone(_raw_viz._eef_position_or_none(np.zeros(8)))
        self.assertIsNone(_raw_viz._eef_position_or_none(np.zeros((1, 9))))


def _valid_fingertip_row() -> np.ndarray:
    return np.arange(15, dtype=np.float32).reshape(5, 3) * 0.01


class TestFingertipHelpers(unittest.TestCase):
    """Both scripts must clear the entity instead of leaving stale spheres."""

    def test_raw_finite_row_returns_positions(self) -> None:
        row = _valid_fingertip_row()
        positions = _raw_viz._fingertip_positions_or_none(row)
        self.assertIsNotNone(positions)
        np.testing.assert_array_equal(positions, row)
        self.assertEqual(positions.dtype, np.float32)

    def test_raw_nan_sentinel_row_returns_none(self) -> None:
        self.assertIsNone(
            _raw_viz._fingertip_positions_or_none(np.full((5, 3), np.nan))
        )

    def test_raw_partially_invalid_row_returns_none(self) -> None:
        row = _valid_fingertip_row()
        row[2, 1] = np.inf
        self.assertIsNone(_raw_viz._fingertip_positions_or_none(row))

    def test_raw_wrong_shape_returns_none(self) -> None:
        self.assertIsNone(_raw_viz._fingertip_positions_or_none(np.zeros((4, 3))))
        self.assertIsNone(
            _raw_viz._fingertip_positions_or_none(np.zeros((5, 3, 1)))
        )

    def test_processed_finite_row_returns_positions(self) -> None:
        row = _valid_fingertip_row()
        positions = _processed_viz._fingertip_positions_or_none(row)
        self.assertIsNotNone(positions)
        np.testing.assert_array_equal(positions, row)

    def test_processed_invalid_rows_return_none(self) -> None:
        self.assertIsNone(
            _processed_viz._fingertip_positions_or_none(np.full((5, 3), np.nan))
        )
        self.assertIsNone(_processed_viz._fingertip_positions_or_none(np.zeros((5, 2))))


class TestProcessedEefHelper(unittest.TestCase):
    def test_finite_row_returns_position(self) -> None:
        position = _processed_viz._eef_position_or_none(_valid_eef_row())
        self.assertIsNotNone(position)
        np.testing.assert_allclose(position, (0.3, -0.1, 0.25))

    def test_invalid_rows_return_none(self) -> None:
        self.assertIsNone(_processed_viz._eef_position_or_none(np.full(9, np.nan)))
        self.assertIsNone(_processed_viz._eef_position_or_none(np.zeros(8)))

    def test_series_labels_for_eef_pose(self) -> None:
        labels = _processed_viz._series_labels("eef_pose", 9)
        self.assertEqual(
            labels,
            ["ee_x", "ee_y", "ee_z", "ee_r0", "ee_r1", "ee_r2", "ee_r3", "ee_r4", "ee_r5"],
        )

    def test_visual_constants(self) -> None:
        # EEF sphere is larger than fingertip spheres and distinctly colored.
        self.assertNotIn(_processed_viz._EEF_COLOR, _processed_viz._FINGERTIP_COLORS)
        self.assertNotIn(_raw_viz._EEF_COLOR, _raw_viz._FINGERTIP_COLORS)


if __name__ == "__main__":
    unittest.main()
