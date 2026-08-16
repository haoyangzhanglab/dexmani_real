"""P3: the reusable candidate build + validate/send publication boundary.

Locks the two helpers extracted from ``_safe_joint_publish`` so teleop,
keyboard/replay, and the learned-policy coordinator share one candidate
construction + gate/transport tail:

  - ``build_action_candidate`` stamps a monotonic action_id, the target/validity
    timestamps, and copied joint targets; an invalid observation anchor yields
    ``None`` and never allocates a command.
  - ``validate_and_send_candidate`` reads authoritative arm/hand feedback,
    runs ``SafetyGate.validate`` plus coupled-hand preflight, and publishes via
    ``send_command``; rejection or unhealthy feedback yields a typed failure
    and writes no transport.
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
from dexmani_real.policy.safety import (CommandPublishStatus, SafetyGate,
                                        build_action_candidate,
                                        validate_and_send_candidate)
from dexmani_real.robot.safety import SafetyState, transition
from dexmani_real.shm.shared_storage import SharedStorage


def _mid(low: tuple[float, ...], high: tuple[float, ...]) -> np.ndarray:
    lo = np.asarray(low, dtype=np.float64)
    hi = np.asarray(high, dtype=np.float64)
    return (lo + hi) / 2.0


def _drain_arm_queue(shared: SharedStorage, settle_s: float = 0.2) -> int:
    """Pop every queued arm endpoint; return the count drained.

    ``send_command`` writes through a ``multiprocessing.Queue`` whose feeder
    thread flushes asynchronously, so ``get_nowait`` immediately after a
    successful publish can transiently see nothing.  Poll briefly so the
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


def main() -> int:
    arm_mid = _mid(arm_defaults.joint_limit_lower, arm_defaults.joint_limit_upper)
    hand_mid = _mid(hand_defaults.qpos_min_rad, hand_defaults.qpos_max_rad)
    gate = SafetyGate(
        arm_joint_lower_rad=arm_defaults.joint_limit_lower,
        arm_joint_upper_rad=arm_defaults.joint_limit_upper,
        hand_joint_lower_rad=hand_defaults.qpos_min_rad,
        hand_joint_upper_rad=hand_defaults.qpos_max_rad,
    )

    shared = SharedStorage.create(prefix="check_candidate_publication")
    try:
        assert transition(shared, SafetyState.ARMED)
        shared.arm_command_seq.value = 100
        shared.run_generation.value = 7

        # ── build_action_candidate: field stamping (no feedback needed) ──
        now_ns = 1_000_000_000
        candidate = build_action_candidate(shared, arm_mid, hand_mid, now_ns=now_ns)
        assert candidate is not None
        assert candidate.action_id == 101, "action_id must advance arm_command_seq"
        assert candidate.observation_id == 101, "observation_id defaults to action_id"
        assert candidate.run_generation == 7, "run_generation must come from shared"
        assert candidate.created_monotonic_ns == now_ns
        assert candidate.target_monotonic_ns == now_ns + int(shared.action_lead_time_s * 1e9)
        assert candidate.valid_until_monotonic_ns == now_ns + int(0.5 * 1e9)
        assert candidate.is_hold is False
        np.testing.assert_array_equal(candidate.arm_qpos, arm_mid)
        np.testing.assert_array_equal(candidate.hand_qpos, hand_mid)

        # observation_id / is_hold / anchor overrides
        with_obs = build_action_candidate(
            shared,
            arm_mid,
            None,
            is_hold=True,
            observation_id=42,
            observation_anchor_monotonic_ns=now_ns - 1,
            now_ns=now_ns,
        )
        assert with_obs is not None
        assert with_obs.observation_id == 42
        assert with_obs.is_hold is True
        assert with_obs.hand_qpos is None

        # invalid observation anchor -> None, no command allocated
        assert (
            build_action_candidate(
                shared, arm_mid, hand_mid, observation_anchor_monotonic_ns=0, now_ns=now_ns
            )
            is None
        ), "non-positive anchor must be rejected"
        assert (
            build_action_candidate(
                shared, arm_mid, hand_mid, observation_anchor_monotonic_ns=now_ns + 1, now_ns=now_ns
            )
            is None
        ), "future anchor must be rejected"

        # ── centralized runtime gate runs before feedback/transport ──
        candidate = build_action_candidate(shared, arm_mid, None)
        assert candidate is not None
        shared.estop_request.value = True
        result = validate_and_send_candidate(
            shared, candidate, gate=gate, hand_feedback_max_age_s=1.0
        )
        assert result.status == CommandPublishStatus.ESTOP_REQUESTED, result
        shared.estop_request.value = False

        shared.error_state.value = True
        result = validate_and_send_candidate(
            shared, candidate, gate=gate, hand_feedback_max_age_s=1.0
        )
        assert result.status == CommandPublishStatus.STICKY_FAULT, result
        shared.error_state.value = False

        shared.is_running.value = False
        result = validate_and_send_candidate(
            shared, candidate, gate=gate, hand_feedback_max_age_s=1.0
        )
        assert result.status == CommandPublishStatus.RUNTIME_STOPPED, result
        shared.is_running.value = True

        assert transition(shared, SafetyState.DISARMED)
        result = validate_and_send_candidate(
            shared, candidate, gate=gate, hand_feedback_max_age_s=1.0
        )
        assert result.status == CommandPublishStatus.SAFETY_STATE_GATED, result
        assert transition(shared, SafetyState.ARMED)
        assert _drain_arm_queue(shared) == 0, "runtime gate must not write transport"

        # ── validate_and_send_candidate: empty arm feedback ──
        candidate = build_action_candidate(shared, arm_mid, hand_mid)
        assert candidate is not None
        result = validate_and_send_candidate(
            shared, candidate, gate=gate, hand_feedback_max_age_s=1.0, prepare_timeout_s=0.1
        )
        assert result.status == CommandPublishStatus.ARM_FEEDBACK_UNAVAILABLE, result
        assert _drain_arm_queue(shared) == 0, "no transport write on unavailable feedback"

        # ── coupled command requires fully healthy hand feedback ──
        shared.arm_state_ring.write(make_arm_state_frame(arm_mid, connected=1, state_valid=1))
        candidate = build_action_candidate(shared, arm_mid, hand_mid)
        assert candidate is not None
        result = validate_and_send_candidate(
            shared, candidate, gate=gate, hand_feedback_max_age_s=1.0, prepare_timeout_s=0.1
        )
        assert result.status == CommandPublishStatus.HAND_FEEDBACK_UNAVAILABLE, result
        assert _drain_arm_queue(shared) == 0, "missing hand feedback must block the arm endpoint"

        shared.hand_state_ring.write(
            make_hand_state_frame(hand_mid, connected=1, state_valid=1, send_healthy=0)
        )
        candidate = build_action_candidate(shared, arm_mid, hand_mid)
        assert candidate is not None
        result = validate_and_send_candidate(
            shared, candidate, gate=gate, hand_feedback_max_age_s=1.0, prepare_timeout_s=0.1
        )
        assert result.status == CommandPublishStatus.HAND_FEEDBACK_UNHEALTHY, result
        assert _drain_arm_queue(shared) == 0, "unhealthy hand must block the arm endpoint"

        # ── validate_and_send_candidate: healthy publish path ──
        shared.hand_state_ring.write(make_hand_state_frame(hand_mid, connected=1, state_valid=1))
        candidate = build_action_candidate(shared, arm_mid, hand_mid)
        assert candidate is not None
        sent = validate_and_send_candidate(
            shared, candidate, gate=gate, hand_feedback_max_age_s=1.0, prepare_timeout_s=0.1
        )
        assert sent.status == CommandPublishStatus.PUBLISHED, sent
        published = sent.candidate
        assert published is not None
        assert published.action_id == candidate.action_id
        assert _drain_arm_queue(shared) == 1, "one arm endpoint must be queued"
        assert shared.hand_cmd_ring.read_latest() is not None, "one hand endpoint must be written"

        # ── centralized hand delta preflight rejects before arm enqueue ──
        hand_step = hand_mid.copy()
        hand_step[0] += 0.01
        candidate = build_action_candidate(shared, arm_mid, hand_step)
        assert candidate is not None
        result = validate_and_send_candidate(
            shared,
            candidate,
            gate=gate,
            hand_feedback_max_age_s=1.0,
            prepare_timeout_s=0.1,
            previous_hand_qpos=hand_mid,
            hand_mechanical_lower_rad=np.asarray(
                hand_defaults.mechanical_qpos_min_rad, dtype=np.float64
            ),
            hand_mechanical_upper_rad=np.asarray(
                hand_defaults.mechanical_qpos_max_rad, dtype=np.float64
            ),
            hand_max_delta_rad=0.001,
        )
        assert result.status == CommandPublishStatus.HAND_PREFLIGHT_REJECTED, result
        assert _drain_arm_queue(shared) == 0, "hand preflight must precede arm enqueue"

        # ── validate_and_send_candidate: typed gate rejection, no write ──
        bad_arm = np.asarray(arm_defaults.joint_limit_upper, dtype=np.float64) + 10.0
        candidate = build_action_candidate(shared, bad_arm, None)
        assert candidate is not None
        result = validate_and_send_candidate(
            shared, candidate, gate=gate, hand_feedback_max_age_s=1.0, prepare_timeout_s=0.1
        )
        assert result.status == CommandPublishStatus.GATE_REJECTED, result
        assert _drain_arm_queue(shared) == 0, "gate rejection must not write transport"
    finally:
        shared.close()

    print("check_candidate_publication: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
