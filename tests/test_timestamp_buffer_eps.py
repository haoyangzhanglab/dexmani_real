"""get_accumulate_timestamp_idxs slot-assignment boundary tests (eps semantics).

Pins the behavior the 16Hz migration relies on: EpisodeRecorder constructs its
buffer with eps=0.5 (round-to-nearest, tolerating ±dt/2 scheduling jitter),
while the library default stays eps=1e-5 (floor).
"""

from __future__ import annotations

from dexmani_real.recording.timestamp_buffer import get_accumulate_timestamp_idxs

DT = 1.0 / 16.0  # 62.5ms grid


def test_eps_half_absorbs_negative_jitter():
    """eps=0.5: samples up to dt/2 early still land in their own slot."""
    jitter = -0.02  # -20ms < dt/2 = 31.25ms
    ts = [k * DT + jitter for k in range(4)]
    local, global_, nxt = get_accumulate_timestamp_idxs(ts, start_time=0.0, dt=DT, eps=0.5)
    assert global_ == [0, 1, 2, 3]
    assert local == [0, 1, 2, 3]
    assert nxt == 4


def test_eps_half_rounds_beyond_half_period_to_next_slot():
    """eps=0.5: a sample more than dt/2 late belongs to the next slot."""
    local, global_, nxt = get_accumulate_timestamp_idxs([0.0, 1 * DT + 0.6 * DT], start_time=0.0, dt=DT, eps=0.5)
    # Second sample (t=1.6*dt) rounds to slot 2; slot 1 is back-filled by it.
    assert global_ == [0, 1, 2]
    assert local == [0, 1, 1]
    assert nxt == 3


def test_legacy_eps_floor_shifts_on_negative_jitter():
    """eps=1e-5 (library default): -20ms jitter drops the first sample and
    shifts every later sample one slot late — the dup-drop bug eps=0.5 fixes."""
    jitter = -0.02
    ts = [k * DT + jitter for k in range(4)]
    local, global_, nxt = get_accumulate_timestamp_idxs(ts, start_time=0.0, dt=DT, eps=1e-5)
    assert nxt == 3  # one slot short: sample 0 fell below the grid and was dropped
    assert global_ == [0, 1, 2]
    assert local == [1, 2, 3]  # every slot filled by the *next* sample (shifted)


def test_fast_source_same_window_first_wins():
    """Two samples in one window: the first is kept, the later one is dropped."""
    local, global_, nxt = get_accumulate_timestamp_idxs([0.0, 0.01], start_time=0.0, dt=DT, eps=0.5)
    assert global_ == [0]
    assert local == [0]
    assert nxt == 1


def test_missed_slot_backfilled_by_next_sample():
    """A skipped window is back-filled by the NEXT arriving sample (not held)."""
    local, global_, nxt = get_accumulate_timestamp_idxs([0.0, 2 * DT], start_time=0.0, dt=DT, eps=0.5)
    assert global_ == [0, 1, 2]
    assert local == [0, 1, 1]  # slot 1 carries sample 1 (the later one)
    assert nxt == 3
