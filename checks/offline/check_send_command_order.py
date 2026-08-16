"""H6: transport write semantics — arm-then-hand is non-atomic by design, hand write cannot fail.

Covers doc §6.1 item 22 (Phase 4c) by *confirming the invariant* rather than
building a new fault path.  ``send_command`` publishes the arm endpoint into the
bounded queue first, then overwrites the latest-wins hand seqlock.  That ordering
is only safe because the hand ring write cannot fail:

  - ``arm_action_q`` is a bounded ``mp.Queue`` (maxsize=2) whose ``put`` raises
    ``Full`` on backpressure; ``send_command`` catches it and returns
    ``ARM_QUEUE_FULL`` *before* the hand is written.
  - ``hand_cmd_ring`` is a ``SharedMemoryRingBuffer`` seqlock whose ``write``
    returns the new sequence number and has no error-return / backpressure
    channel, so there is deliberately no second write-failure path.

If the hand transport ever changes to one whose write can fail, a coordinated
stop/fault path would be required (not a silent ``PUBLISHED``).  This check pins
that contract with fake transports — no hardware.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from queue import Full

import numpy as np

import _bootstrap  # noqa: F401  (repo root on sys.path)

from dexmani_real.config.defaults import arm as arm_defaults
from dexmani_real.config.defaults import hand as hand_defaults
from dexmani_real.policy.runtime import ActionCandidate
from dexmani_real.policy.safety import CommandPublishStatus, send_command
from dexmani_real.robot.safety import SafetyState


def _mid(low: tuple[float, ...], high: tuple[float, ...]) -> np.ndarray:
    return (np.asarray(low, dtype=np.float64) + np.asarray(high, dtype=np.float64)) / 2.0


class _Flag:
    def __init__(self, value) -> None:
        self.value = value


class _RecordingArmQueue:
    """Bounded queue stand-in: records the arm write, raises ``Full`` when told."""

    def __init__(self, order: list[str]) -> None:
        self._order = order
        self.full = False

    def put(self, frame, block: bool = True, timeout: float | None = None) -> None:
        if self.full:
            raise Full()
        self._order.append("arm")


class _RecordingHandRing:
    """Seqlock ring stand-in: write always succeeds and returns a sequence number."""

    def __init__(self, order: list[str]) -> None:
        self._order = order

    def write(self, frame) -> int:
        self._order.append("hand")
        return 1  # a sequence number, not an error status


class _FakeShared:
    def __init__(self, order: list[str], *, arm_full: bool = False) -> None:
        self.estop_request = _Flag(False)
        self.error_state = _Flag(False)
        self.is_running = _Flag(True)
        self.safety_state = _Flag(int(SafetyState.ARMED))
        self.action_lead_time_s = 0.05
        self.arm_action_q = _RecordingArmQueue(order)
        self.arm_action_q.full = arm_full
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


def _test_publish_order(arm_mid: np.ndarray, hand_mid: np.ndarray) -> None:
    # Coupled action: arm endpoint enqueued before the hand endpoint is written.
    order: list[str] = []
    result = send_command(_FakeShared(order), _candidate(arm_mid, hand_mid), prepare_timeout_s=1.0)
    assert result.status == CommandPublishStatus.PUBLISHED, result
    assert order == ["arm", "hand"], order

    # The hand write's sequence-number return is ignored — it is not a status.
    order = []
    result = send_command(_FakeShared(order), _candidate(None, hand_mid), prepare_timeout_s=1.0)
    assert result.status == CommandPublishStatus.PUBLISHED, result
    assert order == ["hand"], order


def _test_arm_backpressure_before_hand(arm_mid: np.ndarray, hand_mid: np.ndarray) -> None:
    # A full arm queue fails fast: the hand is never written, and the caller gets
    # a typed rejection (not a phantom PUBLISHED, not a half-committed action).
    order: list[str] = []
    result = send_command(
        _FakeShared(order, arm_full=True), _candidate(arm_mid, hand_mid), prepare_timeout_s=1.0
    )
    assert result.status == CommandPublishStatus.ARM_QUEUE_FULL, result
    assert order == [], order


def _test_hand_write_cannot_fail() -> None:
    import dexmani_real.policy.safety as safety_mod
    import dexmani_real.shm.ring_buffer as ring_mod
    import dexmani_real.shm.shared_storage as storage_mod

    safety_src = Path(safety_mod.__file__).read_text()
    # The arm queue is the only bounded/failable transport: its backpressure is
    # caught, and the hand write below has no failure handling.
    assert "except Full:" in safety_src
    # The invariant is pinned in a comment (doc §6.1 item 22 is met by
    # confirming the invariant, not by adding a rollback path).
    assert "arm-then-hand ordering is" in safety_src
    assert "non-atomic by design" in safety_src

    # The hand ring write returns a sequence number (int), not an error status,
    # so there is no error-return channel for a caller to check.
    ring_src = Path(ring_mod.__file__).read_text()
    assert "def write(self, data: np.ndarray) -> int:" in ring_src

    # The arm queue is bounded (maxsize=2) and therefore raises ``Full``.
    storage_src = Path(storage_mod.__file__).read_text()
    assert "arm_action_q_maxsize: int = 2" in storage_src
    assert "ctx.Queue(maxsize=cfg.arm_action_q_maxsize)" in storage_src


def main() -> int:
    arm_mid = _mid(arm_defaults.joint_limit_lower, arm_defaults.joint_limit_upper)
    hand_mid = _mid(hand_defaults.qpos_min_rad, hand_defaults.qpos_max_rad)

    _test_publish_order(arm_mid, hand_mid)
    _test_arm_backpressure_before_hand(arm_mid, hand_mid)
    _test_hand_write_cannot_fail()

    print("check_send_command_order: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
