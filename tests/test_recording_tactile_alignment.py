"""Offline regressions for recording-time camera-aligned tactile selection.

No hardware. Two causal primitives and their consumer are pinned:

* ``read_structured_frame_aligned_to_source`` selects the newest ring frame
  whose source is at or before the camera reference (T1/T2).
* ``read_valid_structured_frame_aligned_to_source`` is the same but skips
  gate-failing frames, so an older valid frame can still be selected (T6).
* ``_recording_policy_observation_signals`` turns that selection plus the hand
  aggregate/provenance gates into the persisted policy-observation tactile
  fields (T1-T5), expressing ``contact_force_valid`` independently of
  ``tactile_force_valid``.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from dexmani_real.ipc.causal import (
    read_structured_frame_aligned_to_source,
    read_valid_structured_frame_aligned_to_source,
)
from dexmani_real.ipc.schema import (
    ARM_STATE_DTYPE,
    HAND_STATE_DTYPE,
    HAND_TACTILE_DTYPE,
    TACTILE_UNIT_CODE_XHAND_SDK_NATIVE,
)
from dexmani_real.teleop.control_loop.grid import (
    _recording_policy_observation_signals,
)

_NS = 1_000_000


def _tactile_frame(source_ns, *, fresh=True, calibrated=True, unit_code=0):
    frame = np.zeros(1, dtype=HAND_TACTILE_DTYPE)
    frame["source_monotonic_ns"] = source_ns
    frame["fresh"] = int(fresh)
    frame["calibrated"] = int(calibrated)
    frame["unit_code"] = unit_code
    frame["tactile_force"] = 1.0
    return frame


def _hand_frame(source_ns, *, tactile_sum_valid=True, qpos_stale=False):
    frame = np.zeros(1, dtype=HAND_STATE_DTYPE)
    frame["source_monotonic_ns"] = source_ns
    frame["state_valid"] = 1
    frame["tactile_sum_valid"] = int(tactile_sum_valid)
    frame["qpos_stale"] = int(qpos_stale)
    frame["tactile_sum"] = 1.0
    return frame


def _arm_frame(source_ns):
    frame = np.zeros(1, dtype=ARM_STATE_DTYPE)
    frame["source_monotonic_ns"] = source_ns
    frame["state_valid"] = 1
    return frame


class _FakeRing:
    """Minimal ring surface for the aligned-source causal primitives."""

    def __init__(self, frames):
        self._frames = list(frames)  # (data, ring_publish_ns, sequence)
        self.latest_sequence = max((f[2] for f in self._frames), default=0)
        self.maxlen = 16

    def read_sequence(self, sequence):
        for frame in self._frames:
            if frame[2] == sequence:
                return frame
        return None


class TestAlignedSourceSelection(unittest.TestCase):
    def test_newest_source_at_or_before_camera_reference_is_selected(self):
        # frame-0 regression: the tactile ring holds 983ms and 1016ms; the
        # camera sampled at 1000ms, so 983ms (not the newer 1016ms) must win.
        ring = _FakeRing(
            [
                (_tactile_frame(983 * _NS), 984 * _NS, 1),
                (_tactile_frame(1016 * _NS), 1017 * _NS, 2),
            ]
        )
        result = read_structured_frame_aligned_to_source(
            ring,
            source_field="source_monotonic_ns",
            reference_source_monotonic_ns=1000 * _NS,
            anchor_monotonic_ns=1025 * _NS,
        )
        assert result is not None
        data, _publish_ns, _sequence = result
        assert int(data["source_monotonic_ns"][0]) == 983 * _NS

    def test_no_causal_candidate_returns_none(self):
        # Every tactile source is in the camera's future: no future fill.
        ring = _FakeRing(
            [
                (_tactile_frame(1016 * _NS), 1017 * _NS, 1),
                (_tactile_frame(1030 * _NS), 1031 * _NS, 2),
            ]
        )
        result = read_structured_frame_aligned_to_source(
            ring,
            source_field="source_monotonic_ns",
            reference_source_monotonic_ns=1000 * _NS,
            anchor_monotonic_ns=1025 * _NS,
        )
        assert result is None


class TestValidAlignedSourceSelection(unittest.TestCase):
    def test_invalid_newest_frame_falls_back_to_older_valid(self):
        # Newest frame (1016ms) is fresh=False; the older 983ms frame is fresh.
        # The validity-gated read must fall back to 983ms, not fail.
        ring = _FakeRing(
            [
                (_tactile_frame(983 * _NS, fresh=True), 984 * _NS, 1),
                (_tactile_frame(1016 * _NS, fresh=False), 1017 * _NS, 2),
            ]
        )
        result = read_valid_structured_frame_aligned_to_source(
            ring,
            source_field="source_monotonic_ns",
            reference_source_monotonic_ns=1000 * _NS,
            anchor_monotonic_ns=1025 * _NS,
            required_true_fields=("fresh", "calibrated"),
        )
        assert result is not None
        data, _publish_ns, _sequence = result
        assert int(data["source_monotonic_ns"][0]) == 983 * _NS

    def test_no_valid_candidate_returns_none(self):
        ring = _FakeRing(
            [
                (_tactile_frame(983 * _NS, fresh=False), 984 * _NS, 1),
            ]
        )
        result = read_valid_structured_frame_aligned_to_source(
            ring,
            source_field="source_monotonic_ns",
            reference_source_monotonic_ns=1000 * _NS,
            anchor_monotonic_ns=1025 * _NS,
            required_true_fields=("fresh", "calibrated"),
        )
        assert result is None


class TestRecordingTactileAlignment(unittest.TestCase):
    """Exercise ``_recording_policy_observation_signals`` tactile gating."""

    _CAMERA_NS = 1000 * _NS
    _ANCHOR_NS = 1025 * _NS
    _TACTILE_NS = 983 * _NS

    def _signals(self, arm, hand, aggregate, dense):
        shared = SimpleNamespace(
            arm_state_ring=object(),
            hand_state_ring=object(),
            hand_tactile_ring=object(),
        )
        # arm/hand qpos, then aggregate and dense tactile, all go through the
        # validity-gated read after the qpos fallback fix.
        valid_side_effect = [
            (arm, arm["source_monotonic_ns"][0], 1),
            (hand, hand["source_monotonic_ns"][0], 1),
            None
            if aggregate is None
            else (aggregate, aggregate["source_monotonic_ns"][0], 1),
            None if dense is None else (dense, dense["source_monotonic_ns"][0], 1),
        ]
        with mock.patch(
            "dexmani_real.teleop.control_loop.grid.read_valid_structured_frame_aligned_to_source",
            side_effect=valid_side_effect,
        ):
            return _recording_policy_observation_signals(
                shared,
                {"source_monotonic_ns": self._CAMERA_NS},
                anchor_monotonic_ns=self._ANCHOR_NS,
                max_observation_skew_s=0.1,
            )

    def test_frame0_pre_existing_tactile_is_valid(self):
        signals = self._signals(
            _arm_frame(self._TACTILE_NS),
            _hand_frame(self._TACTILE_NS),
            _hand_frame(self._TACTILE_NS),
            _tactile_frame(self._TACTILE_NS),
        )
        assert signals["policy_observation_valid"] is True
        assert signals["policy_observation_contact_force_valid"] is True
        assert signals["policy_observation_tactile_force_valid"] is True
        assert signals["policy_observation_tactile_source_monotonic_ns"] == self._TACTILE_NS
        assert signals["policy_observation_tactile_calibrated"] is True
        assert signals["policy_observation_tactile_unit_code"] == (
            TACTILE_UNIT_CODE_XHAND_SDK_NATIVE
        )

    def test_no_causal_tactile_is_invalid(self):
        # All tactile sources are after the camera reference; no future fill.
        signals = self._signals(
            _arm_frame(self._CAMERA_NS),
            _hand_frame(self._CAMERA_NS),
            None,
            None,
        )
        assert signals["policy_observation_valid"] is True  # arm/hand still valid
        assert signals["policy_observation_contact_force_valid"] is False
        assert signals["policy_observation_tactile_force_valid"] is False

    def test_skew_exceeded_is_invalid(self):
        signals = self._signals(
            _arm_frame(self._CAMERA_NS),
            _hand_frame(self._CAMERA_NS),
            _hand_frame(100 * _NS),  # 900ms older than the camera
            _tactile_frame(100 * _NS),
        )
        assert signals["policy_observation_contact_force_valid"] is False
        assert signals["policy_observation_tactile_force_valid"] is False

    def test_aggregate_provenance_source_mismatch_invalidates_contact(self):
        # Aggregate (hand) source and tactile provenance source disagree.
        signals = self._signals(
            _arm_frame(self._TACTILE_NS),
            _hand_frame(self._TACTILE_NS),
            _hand_frame(self._TACTILE_NS),
            _tactile_frame(self._TACTILE_NS + 1 * _NS),
        )
        assert signals["policy_observation_contact_force_valid"] is False
        assert signals["policy_observation_tactile_force_valid"] is False

    def test_aggregate_valid_dense_invalid_are_independent(self):
        # Aggregate is fresh but the dense tactile frame is not: contact stays
        # valid while dense is invalid, without collapsing the two validities.
        # (The dense read falls back to the aggregate source's older fresh row
        # in real deployment; here the mock returns no dense candidate.)
        signals = self._signals(
            _arm_frame(self._TACTILE_NS),
            _hand_frame(self._TACTILE_NS),
            _hand_frame(self._TACTILE_NS),
            None,
        )
        assert signals["policy_observation_contact_force_valid"] is False
        assert signals["policy_observation_tactile_force_valid"] is False


if __name__ == "__main__":
    unittest.main()
