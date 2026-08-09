from __future__ import annotations

import numpy as np
import pytest

from dexmani_real.policy.observation import CausalFrame, SnapshotBuilder
from dexmani_real.policy.runtime import ModalitySpec, ObservationSpec
from dexmani_real.teleop.hand_retarget import XHandRetargeter, adaptive_retargeting_xhand, validate_landmarks
from dexmani_real.utils.signal_utils import ema_smooth_pose


def test_snapshot_builder_is_causal_and_pads_missing_history() -> None:
    spec = ObservationSpec(
        (
            ModalitySpec(
                "arm",
                (1,),
                "float64",
                history_length=3,
                max_age_s=0.2,
                max_skew_s=0.1,
                producer_hz=30.0,
            ),
        ),
        control_hz=10.0,
    )
    builder = SnapshotBuilder(spec, session_generation=9)
    anchor = 1_000_000_000
    frames = [
        CausalFrame(np.array([1.0]), 1, 850_000_000, 860_000_000),
        CausalFrame(np.array([2.0]), 2, 950_000_000, 960_000_000),
        # Captured before the anchor but unavailable until afterwards: causal
        # grid construction must not back-date it into this snapshot.
        CausalFrame(np.array([98.0]), 4, 980_000_000, 1_010_000_000),
        CausalFrame(np.array([99.0]), 3, 1_010_000_000, 1_020_000_000),
    ]

    snapshot = builder.build(anchor_monotonic_ns=anchor, frames={"arm": frames})

    np.testing.assert_array_equal(snapshot.values["arm"], [[0.0], [1.0], [2.0]])
    np.testing.assert_array_equal(snapshot.valid_history_mask["arm"], [False, True, True])
    np.testing.assert_allclose(snapshot.source_age_s["arm"][1:], [0.05, 0.05])
    np.testing.assert_array_equal(snapshot.receive_monotonic_ns["arm"][1:], [860_000_000, 960_000_000])
    assert np.isnan(snapshot.source_age_s["arm"][0])
    assert 99.0 not in snapshot.values["arm"]
    assert 98.0 not in snapshot.values["arm"]
    assert snapshot.session_generation == 9


def test_snapshot_builder_invalidates_cross_modal_skew_per_history_slot() -> None:
    spec = ObservationSpec(
        (
            ModalitySpec("fast", (1,), "float64", history_length=2, max_skew_s=0.02),
            ModalitySpec("slow", (1,), "float64", history_length=2, max_skew_s=0.02),
        ),
        control_hz=10.0,
    )
    anchor = 1_000_000_000
    snapshot = SnapshotBuilder(spec, session_generation=1).build(
        anchor_monotonic_ns=anchor,
        frames={
            "fast": [
                CausalFrame(np.array([1.0]), 1, 895_000_000, 896_000_000),
                CausalFrame(np.array([2.0]), 2, 995_000_000, 996_000_000),
            ],
            "slow": [
                CausalFrame(np.array([3.0]), 1, 890_000_000, 891_000_000),
                CausalFrame(np.array([4.0]), 2, 950_000_000, 951_000_000),
            ],
        },
    )

    np.testing.assert_array_equal(snapshot.valid_history_mask["fast"], [True, True])
    np.testing.assert_array_equal(snapshot.valid_history_mask["slow"], [True, False])
    np.testing.assert_allclose(snapshot.source_skew_s["slow"], [0.005, 0.045])
    np.testing.assert_array_equal(snapshot.values["slow"], [[3.0], [0.0]])


def test_snapshot_builder_validates_derived_ring_capacity() -> None:
    spec = ObservationSpec((ModalitySpec("camera", (2,), "float32", history_length=4, producer_hz=30.0),))
    with pytest.raises(ValueError, match="capacity"):
        SnapshotBuilder.validate_ring_capacities(spec, {"camera": 4})
    SnapshotBuilder.validate_ring_capacities(spec, {"camera": spec.modalities[0].required_ring_capacity})


def test_snapshot_builder_allows_nan_only_in_invalid_nan_padding() -> None:
    spec = ObservationSpec(
        (
            ModalitySpec(
                "arm",
                (2,),
                "float64",
                history_length=2,
                padding="invalid_nan",
            ),
        )
    )
    anchor = 1_000_000_000
    snapshot = SnapshotBuilder(spec, session_generation=1).build(
        anchor_monotonic_ns=anchor,
        frames={"arm": [CausalFrame(np.array([1.0, 2.0]), 1, anchor, anchor)]},
    )

    assert np.isnan(snapshot.values["arm"][0]).all()
    np.testing.assert_array_equal(snapshot.values["arm"][1], [1.0, 2.0])
    np.testing.assert_array_equal(snapshot.valid_history_mask["arm"], [False, True])

    malformed = [CausalFrame(np.array([np.nan, 2.0]), 1, anchor, anchor)]
    invalid_snapshot = SnapshotBuilder(spec, session_generation=1).build(
        anchor_monotonic_ns=anchor,
        frames={"arm": malformed},
    )
    assert not np.any(invalid_snapshot.valid_history_mask["arm"])


def _valid_landmarks() -> np.ndarray:
    points = np.zeros((21, 3), dtype=np.float64)
    points[1:5] = [[0.01, 0.01, 0.0], [0.02, 0.02, 0.0], [0.03, 0.03, 0.0], [0.04, 0.04, 0.0]]
    for base, x in ((5, 0.03), (9, 0.01), (13, -0.02), (17, -0.04)):
        points[base : base + 4] = [[x, 0.03 + 0.015 * index, 0.0] for index in range(4)]
    return points


def test_landmark_gate_rejects_zero_collinear_nan_and_short_bones() -> None:
    valid = _valid_landmarks()
    assert validate_landmarks(valid) == (True, "")

    for invalid in (np.zeros((21, 3)), np.full((21, 3), np.nan)):
        accepted, _ = validate_landmarks(invalid)
        assert not accepted
    collinear = valid.copy()
    collinear[17] = 2.0 * collinear[5]
    assert not validate_landmarks(collinear)[0]
    short_bone = valid.copy()
    short_bone[8] = short_bone[7] + np.array([0.0, 0.001, 0.0])
    assert not validate_landmarks(short_bone)[0]


def test_pinky_scaling_uses_each_original_parent_child_segment() -> None:
    points = _valid_landmarks()
    raw_segments = np.diff(points[17:21], axis=0)
    scaled = adaptive_retargeting_xhand(points)
    scaled_segments = np.diff(scaled[17:21], axis=0)

    ratios = np.linalg.norm(scaled_segments, axis=1) / np.linalg.norm(raw_segments, axis=1)
    np.testing.assert_allclose(ratios, ratios[0])
    np.testing.assert_array_equal(points, _valid_landmarks())


def test_dexpilot_reset_uses_true_sdk_to_internal_inverse_mapping() -> None:
    class _Filter:
        def reset(self) -> None:
            return None

    class _Optimizer:
        projected = np.ones(12, dtype=bool)
        idx_pin2target = np.arange(12)

    class _Retargeter:
        filter = _Filter()
        optimizer = _Optimizer()
        last_qpos = np.zeros(12, dtype=np.float32)

        def reset(self) -> None:
            return None

    retargeter = XHandRetargeter.__new__(XHandRetargeter)
    retargeter.retargeter = _Retargeter()
    retargeter._hand_ema_state = np.ones(12)
    retargeter.retargeted_joint_order = np.array([3, 4, 5, 0, 1, 2, 8, 9, 10, 11, 6, 7])
    retargeter.inverse_retargeted_joint_order = np.argsort(retargeter.retargeted_joint_order)
    measured_sdk = np.arange(12, dtype=np.float64)

    retargeter.reset(measured_sdk)

    np.testing.assert_array_equal(
        retargeter.retargeter.last_qpos,
        measured_sdk[retargeter.inverse_retargeted_joint_order].astype(np.float32),
    )


def test_pose_ema_takes_short_arc_across_plus_minus_pi() -> None:
    eps = 0.01
    previous = np.array([np.cos((np.pi - eps) / 2), 0.0, 0.0, np.sin((np.pi - eps) / 2)])
    target = np.array([np.cos((-np.pi + eps) / 2), 0.0, 0.0, np.sin((-np.pi + eps) / 2)])

    _, smoothed = ema_smooth_pose(np.zeros(3), target, np.zeros(3), previous, 1.0, 0.5)

    assert abs(float(np.dot(smoothed, previous))) > 0.999
    assert np.isclose(np.linalg.norm(smoothed), 1.0)
