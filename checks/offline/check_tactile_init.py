"""H4: explicit tactile initialization — split out of connect(), degrade-not-fatal.

Covers doc §6.1 item 15 (Phase 4a): the connection only opens the device and
seeds the command history; tactile reset/bias is an explicit worker-invoked
``initialize_tactile()`` step.  ``hand_ready`` is set only after that step has
completed (or explicitly degraded) *and* the first valid hand frame has been
published.  A tactile failure degrades to ``calibrated=False`` without blocking
joint control — it is never a startup failure.

Runs against ``FakeHand`` and a lightweight shared-storage stand-in — no hardware.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401  (repo root on sys.path)
from _fakes import FakeHand

import dexmani_real.robot.xhand as xhand_mod
from dexmani_real.robot import hand_process as hp
from dexmani_real.robot.safety import SafetyState


class _Flag:
    def __init__(self, value) -> None:
        self.value = value


class _RecorderRing:
    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []

    def write(self, frame) -> None:
        self.frames.append(np.asarray(frame).copy())

    def read_latest(self):
        return None


class _FakeShared:
    def __init__(self, *, is_running: bool = True) -> None:
        self.error_state = _Flag(False)
        self.is_running = _Flag(is_running)
        self.estop_request = _Flag(False)
        self.safety_state = _Flag(int(SafetyState.ARMED))
        self.run_generation = _Flag(1)
        self.hand_state_ring = _RecorderRing()
        self.hand_tactile_ring = _RecorderRing()
        self.hand_cmd_ring = _RecorderRing()
        self._ready: set[str] = set()
        self._heartbeats: dict[str, float] = {}

    def set_heartbeat(self, name: str, value: float) -> None:
        self._heartbeats[name] = value

    def set_ready(self, name: str) -> None:
        self._ready.add(name)

    def is_ready(self, name: str) -> bool:
        return name in self._ready


def _wait_until(predicate, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition not met within timeout")


def _run_until_ready(hand: FakeHand, shared: _FakeShared) -> threading.Thread:
    """Start hand_loop in a thread, wait for hand_ready, return the thread."""
    original = xhand_mod.XHand
    xhand_mod.XHand = lambda config=None: hand
    try:
        thread = threading.Thread(
            target=hp.hand_loop, args=(shared, hp.HandProcessConfig()), daemon=True
        )
        thread.start()
        _wait_until(lambda: shared.is_ready("hand"))
    finally:
        xhand_mod.XHand = original
    return thread


def _stop(thread: threading.Thread, shared: _FakeShared) -> None:
    shared.is_running.value = False
    thread.join(timeout=5.0)
    assert not thread.is_alive(), "hand_loop failed to exit after shutdown"


def _test_order_and_ready() -> None:
    hand = FakeHand()
    shared = _FakeShared()
    thread = _run_until_ready(hand, shared)
    try:
        # The explicit tactile step runs after connect and before the first read.
        i_connect = hand.first_call_index("connect")
        i_tactile = hand.first_call_index("initialize_tactile")
        i_get_state = hand.first_call_index("get_state")
        assert 0 <= i_connect < i_tactile < i_get_state, hand.call_order()
        # The first valid frame is published before ready (consumers wait on
        # hand_ready expecting the ring to already hold a frame).
        assert len(shared.hand_state_ring.frames) >= 1
        # A successful tactile init leaves the driver calibrated.
        assert hand.tactile_calibrated is True
        assert shared.hand_tactile_ring.frames
        tactile = shared.hand_tactile_ring.frames[0]
        assert int(tactile["fresh"][0]) == 1
        assert int(tactile["calibrated"][0]) == 1
        assert shared.error_state.value is False
    finally:
        _stop(thread, shared)


def _test_tactile_failure_degrades() -> None:
    # A failed initialize_tactile degrades (calibrated=False) but still reaches
    # ready — it never blocks joint control or latches a startup fault.
    hand = FakeHand()
    hand.fail("initialize_tactile")
    shared = _FakeShared()
    thread = _run_until_ready(hand, shared)
    try:
        assert hand.tactile_calibrated is False
        assert shared.error_state.value is False
        assert len(shared.hand_state_ring.frames) >= 1
    finally:
        _stop(thread, shared)

    # A raising initialize_tactile is swallowed by the worker and degrades too.
    hand2 = FakeHand()
    hand2.raise_on("initialize_tactile", RuntimeError("vendor reset refused"))
    shared2 = _FakeShared()
    thread2 = _run_until_ready(hand2, shared2)
    try:
        assert hand2.tactile_calibrated is False
        assert shared2.error_state.value is False
        assert len(shared2.hand_state_ring.frames) >= 1
    finally:
        _stop(thread2, shared2)


def _test_invalid_payload_publishes_invalidation() -> None:
    # Joint feedback still reaches ready, while malformed tactile immediately
    # replaces any older valid ring entry with fresh=0/calibrated=0.
    hand = FakeHand(tactile_calibrated=True, tactile_valid=False)
    shared = _FakeShared()
    thread = _run_until_ready(hand, shared)
    try:
        assert shared.error_state.value is False
        assert shared.hand_state_ring.frames
        assert int(shared.hand_state_ring.frames[0]["state_valid"][0]) == 1
        _wait_until(lambda: bool(shared.hand_tactile_ring.frames))
        tactile = shared.hand_tactile_ring.frames[-1]
        assert int(tactile["fresh"][0]) == 0
        assert int(tactile["calibrated"][0]) == 0
        np.testing.assert_array_equal(tactile["tactile_force"][0], 0.0)
    finally:
        _stop(thread, shared)


def _test_source_structural() -> None:
    hp_src = Path(hp.__file__).read_text()
    # The explicit tactile step precedes the first frame publish and hand_ready.
    tactile_idx = hp_src.index("hand.initialize_tactile()")
    frame_idx = hp_src.index("hand_state_ring.write(_frame0)")
    ready_idx = hp_src.index('set_ready("hand")')
    assert tactile_idx < frame_idx < ready_idx

    xhand_src = Path(xhand_mod.__file__).read_text()
    # initialize_tactile is a real method; connect() no longer does tactile reset
    # (its init path only seeds the command history).
    assert "def initialize_tactile" in xhand_src
    assert "def _init_hand_state" in xhand_src


def main() -> int:
    _test_order_and_ready()
    _test_tactile_failure_degrades()
    _test_invalid_payload_publishes_invalidation()
    _test_source_structural()

    print("check_tactile_init: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
