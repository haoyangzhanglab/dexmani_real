"""RateManager absolute-deadline scheduling tests."""

from __future__ import annotations

import time

from dexmani_real.utils.rate_manager import RateManager


def test_no_drift_with_jittery_work():
    """Absolute deadlines: per-tick work jitter must not accumulate as drift."""
    hz = 100.0
    n = 50
    rm = RateManager(hz)
    t0 = time.perf_counter()
    for i in range(n):
        time.sleep(0.002 if i % 2 else 0.006)  # jittery work, always < period
        rm.wait()
    elapsed = time.perf_counter() - t0
    # With the old relative scheme, drift ≈ sum of scheduling latencies (grows
    # with n).  Absolute schedule: total stays locked to n * period.
    assert abs(elapsed - n / hz) < 0.005, f"drift {elapsed - n / hz:+.4f}s over {n} ticks"


def test_small_overrun_absorbed_on_grid():
    """Overrun < 1 period: next deadline stays on the absolute grid."""
    hz = 50.0
    rm = RateManager(hz)
    t0 = time.perf_counter()
    time.sleep(0.03)  # overrun one tick by ~10ms (< 20ms period)
    rm.wait()  # returns immediately (overdue)
    rm.wait()  # must wait only until the *grid* deadline (2 periods from t0)
    elapsed = time.perf_counter() - t0
    assert abs(elapsed - 2 / hz) < 0.005, f"grid not held: {elapsed:.4f}s vs {2 / hz:.4f}s"


def test_reanchor_after_long_block():
    """Overrun >= 1 period: re-anchor to now, no catch-up burst."""
    hz = 50.0
    rm = RateManager(hz)
    time.sleep(0.1)  # miss ~5 slots
    rm.wait()  # overdue → re-anchor
    t0 = time.perf_counter()
    rm.wait()  # must be a full fresh period, not an immediate catch-up return
    elapsed = time.perf_counter() - t0
    assert elapsed > 0.5 / hz, f"catch-up burst detected: wait returned in {elapsed * 1000:.1f}ms"
    # Upper bound: re-anchor is now + ONE period (now+2*period would be a bug)
    assert elapsed < 1.5 / hz, f"re-anchor overshoot: {elapsed * 1000:.1f}ms (expected ≈{1000 / hz:.0f}ms)"


def test_reset_clears_deadline():
    hz = 50.0
    rm = RateManager(hz)
    time.sleep(0.1)
    rm.reset()
    t0 = time.perf_counter()
    rm.wait()
    elapsed = time.perf_counter() - t0
    assert abs(elapsed - 1 / hz) < 0.005
