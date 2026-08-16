"""H2: hand worker lifecycle — single cleanup path, idempotent disconnect, board-error logging.

Covers doc §6.1 items 9/10/11/17/18/19 for the Phase 2 lifecycle 收口:

  - item 9:  ``source_monotonic_ns`` is host accept time (``time.monotonic_ns()``
    at the moment the read completed), not a device-side timestamp.
  - item 10/11: every init failure reaches the single ``finally``; ``disconnect``
    is idempotent (guards on a non-None ``control``, whose close clears the
    reference) and a raising ``disconnect`` is swallowed, never double-closed.
  - item 17: board-error appear/change/disappear is logged per-joint with hex
    values; repeated identical values never spam the log.
  - item 18: ``clear_local_error()`` is gone; the send/error/read counters keep
    their original trigger thresholds (success resets, failure accumulates).
  - item 19: normal shutdown, e-stop, and capturable init exceptions all reach
    the same ``_safe_disconnect``; no clean exit latches a fault.

Everything runs against ``FakeHand`` and a lightweight shared-storage stand-in —
no hardware, no real ``SharedStorage``.
"""

from __future__ import annotations

import logging
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
from dexmani_real.utils.schema import HAND_JOINT_SHAPE


class _Flag:
    def __init__(self, value) -> None:
        self.value = value


class _RecorderRing:
    """Minimal ring stand-in: records every write, reports no command available."""

    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []

    def write(self, frame) -> None:
        self.frames.append(np.asarray(frame).copy())

    def read_latest(self):
        return None  # no hand command is ever queued in these checks


class _FakeShared:
    """The subset of SharedStorage ``hand_loop`` touches (no ``hand_device_identity``)."""

    def __init__(self, *, is_running: bool = True, estop: bool = False) -> None:
        self.error_state = _Flag(False)
        self.is_running = _Flag(is_running)
        self.estop_request = _Flag(estop)
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


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _wait_until(predicate, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition not met within timeout")


def _run_hand_loop(hand: FakeHand, shared: _FakeShared) -> None:
    """Run ``hand_loop`` synchronously with ``hand`` as the constructed driver."""
    original = xhand_mod.XHand
    xhand_mod.XHand = lambda config=None: hand
    try:
        hp.hand_loop(shared, hp.HandProcessConfig())
    finally:
        xhand_mod.XHand = original


def _test_safe_disconnect() -> None:
    # None -> True without touching a device.
    assert hp._safe_disconnect(None) is True

    # Success -> True, exactly one call.
    hand = FakeHand()
    assert hp._safe_disconnect(hand) is True
    assert hand.call_order() == ["disconnect"], hand.call_order()

    # A raising disconnect is swallowed and reported as False, exactly one call.
    bad = FakeHand()
    bad.raise_on("disconnect", RuntimeError("vendor disconnect refused"))
    assert hp._safe_disconnect(bad) is False
    assert bad.call_order() == ["disconnect"], bad.call_order()


def _test_board_error_transitions() -> None:
    names = ("commboard_err", "jointboard_err", "tipboard_err")
    zero = {n: np.zeros(HAND_JOINT_SHAPE, dtype=np.int32) for n in names}

    cap = _Capture()
    hp.logger.addHandler(cap)
    try:
        # Appear: one joint goes 0 -> 0x2 in commboard.
        appeared = {n: zero[n].copy() for n in names}
        appeared["commboard_err"][3] = 2
        nxt = hp._log_board_error_transitions(zero, appeared)
        assert len(cap.records) == 1, f"expected 1 transition log, got {len(cap.records)}"
        assert "commboard_err" in cap.records[0].getMessage()

        # Change + disappear: [3] 2->0 (disappear) and [5] 0->7 (appear).
        changed = {n: nxt[n].copy() for n in names}
        changed["commboard_err"][3] = 0
        changed["commboard_err"][5] = 7
        before = len(cap.records)
        nxt = hp._log_board_error_transitions(nxt, changed)
        assert len(cap.records) == before + 2, "change and disappear must both log"

        # Repeat identical values: no new logs (no steady-state spam).
        before = len(cap.records)
        nxt = hp._log_board_error_transitions(nxt, changed)
        assert len(cap.records) == before, "repeated identical values must not spam"

        # The returned dict holds fresh copies — mutating the input does not alias.
        assert nxt["commboard_err"] is not changed["commboard_err"]
        changed["commboard_err"][0] = 999
        assert nxt["commboard_err"][0] == 0, "returned dict must be a copy"
    finally:
        hp.logger.removeHandler(cap)


def _test_init_failure_disconnects() -> None:
    # Connect failure (fatal): latch error_state, finally disconnects exactly once.
    hand = FakeHand()
    hand.fail("connect")
    shared = _FakeShared()
    _run_hand_loop(hand, shared)
    assert shared.error_state.value is True, "fatal connect failure must latch error_state"
    assert hand.call_order() == ["connect", "disconnect"], hand.call_order()

    # Initial get_state exception (fatal): latch error_state, finally disconnects once.
    hand2 = FakeHand()
    hand2.raise_on("get_state", RuntimeError("SDK read failed"))
    shared2 = _FakeShared()
    _run_hand_loop(hand2, shared2)
    assert shared2.error_state.value is True, "fatal init exception must latch error_state"
    assert hand2.call_order() == [
        "connect", "initialize_tactile", "get_state", "disconnect"
    ], hand2.call_order()


def _test_loop_shutdown_and_estop() -> None:
    original = xhand_mod.XHand

    # Normal shutdown: reach ready, publish frame0 (host monotonic source), exit on
    # is_running=False, disconnect exactly once, no fault latched.
    hand = FakeHand()
    shared = _FakeShared(is_running=True)
    xhand_mod.XHand = lambda config=None: hand
    try:
        thread = threading.Thread(
            target=hp.hand_loop, args=(shared, hp.HandProcessConfig()), daemon=True
        )
        thread.start()
        _wait_until(lambda: shared.is_ready("hand"))
        assert len(shared.hand_state_ring.frames) >= 1, "frame0 must be published before ready"
        src_ns = int(shared.hand_state_ring.frames[0]["source_monotonic_ns"][0])
        assert abs(src_ns - time.monotonic_ns()) < 5_000_000_000, (
            "source_monotonic_ns must be host accept time, not device time"
        )
        shared.is_running.value = False
        thread.join(timeout=5.0)
        assert not thread.is_alive(), "hand_loop failed to exit after shutdown"
    finally:
        xhand_mod.XHand = original
    assert hand.call_order().count("disconnect") == 1, hand.call_order()
    assert shared.error_state.value is False, "clean shutdown must not latch fault"

    # e-stop: the loop breaks on estop_request and still disconnects once.
    hand2 = FakeHand()
    shared2 = _FakeShared(is_running=True, estop=True)
    xhand_mod.XHand = lambda config=None: hand2
    try:
        thread2 = threading.Thread(
            target=hp.hand_loop, args=(shared2, hp.HandProcessConfig()), daemon=True
        )
        thread2.start()
        _wait_until(lambda: shared2.is_ready("hand"))
        thread2.join(timeout=5.0)
        assert not thread2.is_alive(), "hand_loop failed to exit on e-stop"
    finally:
        xhand_mod.XHand = original
    assert hand2.call_order().count("disconnect") == 1, hand2.call_order()
    assert shared2.error_state.value is False


def _test_source_structural() -> None:
    src = Path(hp.__file__).read_text()
    # item 9: source_monotonic_ns is host accept time.
    assert "_initial_source_ns = time.monotonic_ns()" in src
    assert '_frame0["source_monotonic_ns"][0] = _initial_source_ns' in src
    # item 18: clear_local_error removed; recovery is counter-only.
    assert "clear_local_error" not in src
    # item 19: the single finally routes every exit through _safe_disconnect.
    assert "_safe_disconnect(hand)" in src

    # item 10/11: disconnect guards on a non-None control whose close clears it,
    # so a repeated disconnect is a no-op rather than a second freed-handle access.
    xhand_src = Path(xhand_mod.__file__).read_text()
    assert "if self.control is not None:" in xhand_src
    assert "self.control = None" in xhand_src


def main() -> int:
    _test_safe_disconnect()
    _test_board_error_transitions()
    _test_init_failure_disconnects()
    _test_loop_shutdown_and_estop()
    _test_source_structural()

    print("check_hand_lifecycle: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
