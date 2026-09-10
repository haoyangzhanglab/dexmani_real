"""Offline regressions for run-start warm-up and visual-frame reuse.

These pin the deployment temporal contract (§4.3 warm-start, §5 camera reuse):
the first complete post-run observation edge-repeats into the leading history
slots, a healthy recent visual frame may be reused across adjacent grid slots,
and future / middle-gap data still fails the window.
"""

import numpy as np

from dexmani_real.deployment.inference.observation import (
    FrameWindow,
    RgbFrame,
    _align_state_history_to_reference_ns,
    _select_camera_control_grid,
    _select_control_grid_reference_ns,
)

_T0_NS = 10**15
_DT_NS = 62_500_000  # 16 Hz


def _frame_window(sources_ns, *, shape=(7,)) -> FrameWindow:
    sources = np.asarray(sources_ns, dtype=np.uint64)
    n = sources.size
    return FrameWindow(
        values=np.zeros((n, *shape)),
        source_sequence=np.arange(1, n + 1, dtype=np.uint64),
        source_monotonic_ns=sources,
        publish_monotonic_ns=sources + 1_000_000,
        valid_mask=np.ones(n, dtype=np.uint8),
    )


def test_reference_ns_fills_warmup_with_edge_repeat():
    # At run start (latest_tick=0) with horizon=3, the three leading references
    # all collapse to the earliest tick (run_started).
    reference, logical_step = _select_control_grid_reference_ns(
        run_started_ns=_T0_NS,
        anchor_ns=_T0_NS + 1_000_000,
        history_len=3,
        step_dt_ns=_DT_NS,
    )
    np.testing.assert_array_equal(reference, [_T0_NS, _T0_NS, _T0_NS])
    assert logical_step == _T0_NS


def test_reference_ns_steady_state_is_unchanged():
    reference, logical_step = _select_control_grid_reference_ns(
        run_started_ns=_T0_NS,
        anchor_ns=_T0_NS + 5 * _DT_NS + 1_000_000,
        history_len=3,
        step_dt_ns=_DT_NS,
    )
    np.testing.assert_array_equal(
        reference, [_T0_NS + 3 * _DT_NS, _T0_NS + 4 * _DT_NS, _T0_NS + 5 * _DT_NS]
    )
    assert logical_step == _T0_NS + 5 * _DT_NS


def test_align_state_edge_repeats_first_post_run_observation():
    window = _frame_window([_T0_NS + 5_000_000])
    reference = np.array([_T0_NS, _T0_NS, _T0_NS], dtype=np.uint64)
    aligned = _align_state_history_to_reference_ns(
        window, reference, max_skew_ns=100_000_000, run_started_ns=_T0_NS
    )
    assert aligned is not None
    np.testing.assert_array_equal(
        aligned.source_monotonic_ns,
        [_T0_NS + 5_000_000, _T0_NS + 5_000_000, _T0_NS + 5_000_000],
    )


def test_align_state_middle_gap_still_fails():
    window = _frame_window([_T0_NS + 10_000_000, _T0_NS + 80_000_000])
    reference = np.array([_T0_NS + 40_000_000], dtype=np.uint64)
    aligned = _align_state_history_to_reference_ns(
        window, reference, max_skew_ns=20_000_000, run_started_ns=_T0_NS
    )
    assert aligned is None


def test_align_state_future_source_still_fails():
    # A source newer than the (non-leading) logical step must not be accepted.
    window = _frame_window([_T0_NS + 5 * _DT_NS + 5_000_000])
    reference = np.array([_T0_NS + 4 * _DT_NS], dtype=np.uint64)
    aligned = _align_state_history_to_reference_ns(
        window, reference, max_skew_ns=100_000_000, run_started_ns=_T0_NS
    )
    assert aligned is None


def _rgb_frame(source_ns, sequence) -> RgbFrame:
    return RgbFrame(
        values=np.zeros((2, 2, 3), dtype=np.uint8),
        source_camera_sequence=sequence,
        source_monotonic_ns=source_ns,
        publish_monotonic_ns=source_ns + 1_000_000,
        camera_generation=1,
    )


def test_camera_edge_repeats_single_run_start_frame():
    frame = _rgb_frame(_T0_NS + 5_000_000, 1)
    selected, logical_step = _select_camera_control_grid(
        (frame,),
        run_started_ns=_T0_NS,
        anchor_ns=_T0_NS + 1_000_000,
        history_len=3,
        step_dt_ns=_DT_NS,
        max_grid_lag_ns=100_000_000,
    )
    assert logical_step == _T0_NS
    assert len(selected) == 3
    assert all(f.source_camera_sequence == 1 for f in selected)


def test_camera_reuses_recent_healthy_frame_across_adjacent_ticks():
    # C0 at T0+5ms and C1 at T0+60ms (causal to the T0+DT logical step); the
    # window is [C0, C1]. When only C0 is resident, both slots reuse C0.
    c0 = _rgb_frame(_T0_NS + 5_000_000, 1)
    c1 = _rgb_frame(_T0_NS + 60_000_000, 2)
    selected, logical_step = _select_camera_control_grid(
        (c0, c1),
        run_started_ns=_T0_NS,
        anchor_ns=_T0_NS + _DT_NS + 1_000_000,
        history_len=2,
        step_dt_ns=_DT_NS,
        max_grid_lag_ns=100_000_000,
    )
    assert logical_step == _T0_NS + _DT_NS
    assert [f.source_camera_sequence for f in selected] == [1, 2]

    selected, _ = _select_camera_control_grid(
        (c0,),
        run_started_ns=_T0_NS,
        anchor_ns=_T0_NS + _DT_NS + 1_000_000,
        history_len=2,
        step_dt_ns=_DT_NS,
        max_grid_lag_ns=100_000_000,
    )
    assert [f.source_camera_sequence for f in selected] == [1, 1]


def test_camera_future_source_still_fails():
    # A frame newer than the logical step must not be edge-repeated in
    # steady state (its desired_ns is past run start).
    future = _rgb_frame(_T0_NS + 5 * _DT_NS + 5_000_000, 1)
    selected, _ = _select_camera_control_grid(
        (future,),
        run_started_ns=_T0_NS,
        anchor_ns=_T0_NS + 5 * _DT_NS + 1_000_000,
        history_len=2,
        step_dt_ns=_DT_NS,
        max_grid_lag_ns=100_000_000,
    )
    assert selected == ()
