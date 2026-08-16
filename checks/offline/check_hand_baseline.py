"""H0: hand worker baseline — driver surface, schema, transport order, watchdog.

Freezes the pre-existing hand-worker contract before any Phase 2–4 behaviour
change, so later commits can prove they only changed what they claim to change:

  - ``HAND_STATE_DTYPE`` field meanings (including the reserved ``qpos_stale``
    compatibility bit, ``last_cmd_seq`` = last successful ``send_action``, and
    the per-joint board error registers).
  - the first valid hand frame is published before ``hand_ready`` (consumers
    wait on ``hand_ready`` and expect the ring to already hold a frame).
  - ``send_command`` writes the arm endpoint before the hand endpoint (the arm
    is an ordered queue; the hand is a latest-wins seqlock).
  - the three hand-worker watchdog counters share ``RetryCounter`` semantics
    (``triggered`` iff ``count >= max_consecutive``) with the thresholds
    ``send_err_watchdog_count`` (30) and ``error_state_watchdog_frames`` (5).

Pure assertions — no behaviour change, no hardware.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401  (repo root on sys.path)
from _fakes import FakeHand

from dexmani_real.config.defaults import arm as arm_defaults
from dexmani_real.config.defaults import hand as hand_defaults
from dexmani_real.policy.runtime import ActionCandidate
from dexmani_real.policy.safety import CommandPublishStatus, send_command
from dexmani_real.robot.hand_process import HandProcessConfig
from dexmani_real.robot.safety import SafetyState
from dexmani_real.utils.retry import RetryCounter
from dexmani_real.utils.schema import (
    HAND_CONTACT_SHAPE,
    HAND_DOF,
    HAND_JOINT_SHAPE,
    HAND_STATE_DTYPE,
    HAND_TACTILE_SUM_SHAPE,
)


def _mid(low: tuple[float, ...], high: tuple[float, ...]) -> np.ndarray:
    lo = np.asarray(low, dtype=np.float64)
    hi = np.asarray(high, dtype=np.float64)
    return (lo + hi) / 2.0


# -- minimal transport-recording shared for the send_command order test -------
class _Flag:
    def __init__(self, value):
        self.value = value


class _RecordingArmQueue:
    def __init__(self, order: list[str]) -> None:
        self._order = order

    def put(self, frame, block: bool = True, timeout: float | None = None) -> None:
        self._order.append("arm")


class _RecordingHandRing:
    def __init__(self, order: list[str]) -> None:
        self._order = order

    def write(self, frame) -> None:
        self._order.append("hand")


class _FakeShared:
    def __init__(self, order: list[str]) -> None:
        self.estop_request = _Flag(False)
        self.error_state = _Flag(False)
        self.is_running = _Flag(True)
        self.safety_state = _Flag(int(SafetyState.ARMED))
        self.action_lead_time_s = 0.05
        self.arm_action_q = _RecordingArmQueue(order)
        self.hand_cmd_ring = _RecordingHandRing(order)


def _candidate(arm_qpos: np.ndarray | None, hand_qpos: np.ndarray | None) -> ActionCandidate:
    now = time.monotonic_ns()
    return ActionCandidate(
        observation_id=1,
        run_generation=1,
        created_monotonic_ns=now,
        target_monotonic_ns=now + 50_000_000,
        valid_until_monotonic_ns=now + 500_000_000,
        action_id=1,
        arm_qpos=arm_qpos,
        hand_qpos=hand_qpos,
    )


def _test_schema() -> None:
    # Field order is the contract: any rename/reorder is a breaking change.
    expected = (
        "qpos",
        "current",
        "tactile_sum",
        "tactile_contact",
        "error_state",
        "connected",
        "qpos_stale",
        "last_cmd_seq",
        "last_cmd_qpos",
        "commboard_err",
        "jointboard_err",
        "tipboard_err",
        "source_monotonic_ns",
        "publish_monotonic_ns",
        "state_valid",
        "send_healthy",
        "read_healthy",
        "timestamp",
    )
    assert HAND_STATE_DTYPE.names == expected, HAND_STATE_DTYPE.names

    assert HAND_STATE_DTYPE["qpos"].shape == HAND_JOINT_SHAPE
    assert HAND_STATE_DTYPE["current"].shape == HAND_JOINT_SHAPE
    assert HAND_STATE_DTYPE["last_cmd_qpos"].shape == HAND_JOINT_SHAPE
    assert HAND_STATE_DTYPE["tactile_sum"].shape == HAND_TACTILE_SUM_SHAPE
    assert HAND_STATE_DTYPE["tactile_contact"].shape == HAND_CONTACT_SHAPE
    for name in ("commboard_err", "jointboard_err", "tipboard_err"):
        assert HAND_STATE_DTYPE[name].shape == HAND_JOINT_SHAPE, name

    fields = HAND_STATE_DTYPE.fields
    # `qpos_stale` is a reserved compatibility bit (always 0 in the worker);
    # freshness comes from source timestamp + read_healthy/state_valid.
    assert fields["qpos_stale"][0] == np.dtype("<u1")
    assert fields["last_cmd_seq"][0] == np.dtype("<u8")
    for name in ("error_state", "connected", "state_valid", "send_healthy", "read_healthy"):
        assert fields[name][0] == np.dtype("<u1"), name
    for name in ("commboard_err", "jointboard_err", "tipboard_err"):
        assert HAND_STATE_DTYPE[name].base == np.dtype("<i4"), name


def _test_first_frame_before_ready() -> None:
    from dexmani_real.robot import hand_process as hand_process_mod

    src = Path(hand_process_mod.__file__).read_text()
    write_idx = src.index("hand_state_ring.write(_frame0)")
    ready_idx = src.index('set_ready("hand")')
    assert write_idx < ready_idx, (
        "initial hand frame must be published before hand_ready "
        "(consumers wait on hand_ready expecting the ring to already hold a frame)"
    )


def _test_send_command_order(arm_mid: np.ndarray, hand_mid: np.ndarray) -> None:
    # Coupled: the arm endpoint is enqueued before the hand endpoint is written.
    order: list[str] = []
    result = send_command(
        _FakeShared(order), _candidate(arm_mid, hand_mid), prepare_timeout_s=1.0
    )
    assert result.status == CommandPublishStatus.PUBLISHED, result
    assert order == ["arm", "hand"], order

    # Arm-only publishes only the arm endpoint.
    order = []
    result = send_command(
        _FakeShared(order), _candidate(arm_mid, None), prepare_timeout_s=1.0
    )
    assert result.status == CommandPublishStatus.PUBLISHED, result
    assert order == ["arm"], order

    # Hand-only publishes only the hand endpoint.
    order = []
    result = send_command(
        _FakeShared(order), _candidate(None, hand_mid), prepare_timeout_s=1.0
    )
    assert result.status == CommandPublishStatus.PUBLISHED, result
    assert order == ["hand"], order


def _test_watchdog_semantics() -> None:
    # All three hand-worker counters share RetryCounter trigger semantics:
    # triggered iff count >= max_consecutive, and it keeps counting past the
    # threshold (callers decide whether to stop escalating).
    c = RetryCounter(max_consecutive=3, label="hand_send")
    assert not c.triggered
    c.inc()
    c.inc()
    assert not c.triggered, "triggered below threshold"
    c.inc()
    assert c.triggered, "triggered at threshold"
    c.inc()
    assert c.triggered, "stays triggered past threshold"
    c.reset()
    assert not c.triggered and c.count == 0

    # Freeze the resolved thresholds.
    cfg = HandProcessConfig()
    assert cfg.send_err_watchdog_frames == hand_defaults.send_err_watchdog_count
    assert cfg.send_err_watchdog_frames == 30
    assert cfg.error_state_watchdog_frames == 5


def _test_fake_hand(hand_mid: np.ndarray) -> None:
    hand = FakeHand(qpos=hand_mid)
    assert hand.connect() is True and hand.connected_flag
    state = hand.get_state(force_update=True)
    assert state.qpos.shape == HAND_JOINT_SHAPE
    assert np.allclose(state.qpos, hand_mid)
    assert state.tactile_sum.shape == HAND_TACTILE_SUM_SHAPE
    assert state.tactile_contact.shape == HAND_CONTACT_SHAPE
    for name in ("commboard_err", "jointboard_err", "tipboard_err"):
        assert getattr(state, name).shape == HAND_JOINT_SHAPE

    target = hand_mid + 0.001
    assert hand.send_action(target) is True
    assert np.allclose(hand.last_qpos_cmd, target)

    # A failed send must not advance last_qpos_cmd (driver only advances on SDK success).
    hand.fail("send_action")
    assert hand.send_action(hand_mid) is False
    assert np.allclose(hand.last_qpos_cmd, target)

    hand.raise_on("get_state", RuntimeError("SDK read failed"))
    try:
        hand.get_state()
    except RuntimeError:
        pass
    else:
        raise AssertionError("raise_on must propagate the configured exception")

    assert hand.call_order() == [
        "connect", "get_state", "send_action", "send_action", "get_state"
    ], hand.call_order()
    hand.disconnect()
    assert not hand.connected_flag


def main() -> int:
    arm_mid = _mid(arm_defaults.joint_limit_lower, arm_defaults.joint_limit_upper)
    hand_mid = _mid(hand_defaults.qpos_min_rad, hand_defaults.qpos_max_rad)

    _test_schema()
    _test_first_frame_before_ready()
    _test_send_command_order(arm_mid, hand_mid)
    _test_watchdog_semantics()
    _test_fake_hand(hand_mid)

    print("check_hand_baseline: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
