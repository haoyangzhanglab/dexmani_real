"""Phase-1 remaining expiry check: hand feedback source-timestamp freshness.

Locks the missing half of ``_hand_feedback_snapshot`` (doc §6.1 item 4): a
coupled publish must reject hand feedback whose source timestamp is missing
(``<= 0``), in the future, or older than ``hand_feedback_max_age_s`` — the
same fail-closed boundary that already checks connectivity, hardware error,
state validity, and command/state I/O health.

The pure predicate ``validate_hand_feedback`` is exercised directly for the
three timestamp branches plus the finite-``max_age_s`` contract, and the
end-to-end ``validate_and_send_candidate`` path is exercised to prove the
rejection surfaces as ``HAND_FEEDBACK_UNHEALTHY`` with a descriptive detail
while a fresh frame still publishes.
"""

from __future__ import annotations

import sys
import time
from queue import Empty

import numpy as np

import _bootstrap  # noqa: F401  (repo root on sys.path)
from _fakes import make_arm_state_frame, make_hand_state_frame

from dexmani_real.config.defaults import arm as arm_defaults
from dexmani_real.config.defaults import hand as hand_defaults
from dexmani_real.policy.safety import (
    CommandPublishStatus,
    SafetyGate,
    build_action_candidate,
    validate_and_send_candidate,
)
from dexmani_real.robot.safety import SafetyState, transition
from dexmani_real.shm.shared_storage import SharedStorage
from dexmani_real.utils.hand_health import validate_hand_feedback


def _mid(low: tuple[float, ...], high: tuple[float, ...]) -> np.ndarray:
    lo = np.asarray(low, dtype=np.float64)
    hi = np.asarray(high, dtype=np.float64)
    return (lo + hi) / 2.0


def _drain_arm_queue(shared: SharedStorage, settle_s: float = 0.2) -> int:
    """Pop every queued arm endpoint; return the count drained.

    ``send_command`` writes through a ``multiprocessing.Queue`` whose feeder
    thread flushes asynchronously, so ``get_nowait`` immediately after a
    successful publish can transiently see nothing. Poll briefly so the
    positive case is deterministic without slowing the negative cases.
    """
    deadline = time.monotonic() + settle_s
    count = 0
    while True:
        try:
            shared.arm_action_q.get_nowait()
            count += 1
        except Empty:
            if time.monotonic() >= deadline:
                return count
            time.sleep(0.005)


def _predicate_args(qpos: np.ndarray, **overrides) -> dict:
    now_ns = time.monotonic_ns()
    args = dict(
        connected=True,
        error_state=False,
        state_valid=True,
        send_healthy=True,
        read_healthy=True,
        source_monotonic_ns=now_ns,
        now_monotonic_ns=now_ns,
        max_age_s=1.0,
        qpos=qpos,
    )
    args.update(overrides)
    return args


def main() -> int:
    arm_mid = _mid(arm_defaults.joint_limit_lower, arm_defaults.joint_limit_upper)
    hand_mid = _mid(hand_defaults.qpos_min_rad, hand_defaults.qpos_max_rad)

    # ── pure predicate: timestamp branches + max_age contract ──
    now_ns = time.monotonic_ns()
    assert validate_hand_feedback(**_predicate_args(hand_mid)) is None, "fresh feedback is valid"

    stale = validate_hand_feedback(
        **_predicate_args(hand_mid, source_monotonic_ns=now_ns - int(2.0 * 1e9))
    )
    assert stale is not None and "stale" in stale, stale

    future = validate_hand_feedback(
        **_predicate_args(hand_mid, source_monotonic_ns=now_ns + int(5.0 * 1e9))
    )
    assert future is not None and "future" in future, future

    missing = validate_hand_feedback(**_predicate_args(hand_mid, source_monotonic_ns=0))
    assert missing is not None and "no source timestamp" in missing, missing

    for bad_age in (0.0, -1.0, float("nan"), float("inf")):
        try:
            validate_hand_feedback(**_predicate_args(hand_mid, max_age_s=bad_age))
        except ValueError:
            pass
        else:
            raise AssertionError(f"max_age_s={bad_age!r} must raise ValueError")

    # ── end-to-end: expiry surfaces as HAND_FEEDBACK_UNHEALTHY ──
    gate = SafetyGate(
        arm_joint_lower_rad=arm_defaults.joint_limit_lower,
        arm_joint_upper_rad=arm_defaults.joint_limit_upper,
        hand_joint_lower_rad=hand_defaults.qpos_min_rad,
        hand_joint_upper_rad=hand_defaults.qpos_max_rad,
    )
    shared = SharedStorage.create(prefix="check_hand_feedback_expiry")
    try:
        assert transition(shared, SafetyState.ARMED)
        shared.arm_state_ring.write(make_arm_state_frame(arm_mid, connected=1, state_valid=1))

        def _publish(hand_frame: np.ndarray):
            shared.hand_state_ring.write(hand_frame)
            candidate = build_action_candidate(shared, arm_mid, hand_mid)
            assert candidate is not None
            return validate_and_send_candidate(
                shared, candidate, gate=gate, hand_feedback_max_age_s=1.0
            )

        # no-timestamp frame → reject
        result = _publish(
            make_hand_state_frame(hand_mid, source_monotonic_ns=0)
        )
        assert result.status == CommandPublishStatus.HAND_FEEDBACK_UNHEALTHY, result
        assert "no source timestamp" in result.detail, result.detail
        assert _drain_arm_queue(shared) == 0, "rejected feedback must not write transport"

        # future frame → reject
        result = _publish(
            make_hand_state_frame(
                hand_mid, source_monotonic_ns=time.monotonic_ns() + int(5.0 * 1e9)
            )
        )
        assert result.status == CommandPublishStatus.HAND_FEEDBACK_UNHEALTHY, result
        assert "future" in result.detail, result.detail

        # stale frame → reject
        result = _publish(
            make_hand_state_frame(
                hand_mid, source_monotonic_ns=time.monotonic_ns() - int(2.0 * 1e9)
            )
        )
        assert result.status == CommandPublishStatus.HAND_FEEDBACK_UNHEALTHY, result
        assert "stale" in result.detail, result.detail

        # fresh frame → publish
        shared.hand_state_ring.write(make_hand_state_frame(hand_mid))
        result = _publish(make_hand_state_frame(hand_mid))
        assert result.status == CommandPublishStatus.PUBLISHED, result
        assert _drain_arm_queue(shared) == 1, "fresh coupled feedback must publish one endpoint"
    finally:
        shared.close()

    print("check_hand_feedback_expiry: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
