"""A11: coupled ``wait_applied`` acknowledgement semantics.

Exercises arm-only and coupled apply acknowledgement, pre-publication hand
health rejection, and hand supersession. A coupled action requires arm
``last_cmd_seq >= action_id`` plus healthy hand ``last_cmd_seq == action_id``;
hand ``> action_id`` fails immediately without waiting.
"""

from __future__ import annotations

import sys
from queue import Empty

import numpy as np

import _bootstrap  # noqa: F401  (repo root on sys.path)
from _fakes import make_arm_state_frame, make_hand_state_frame

from dexmani_real.config.defaults import arm as arm_defaults
from dexmani_real.config.defaults import hand as hand_defaults
from dexmani_real.policy.safety import (CommandPublishStatus, SafetyGate,
                                        publish_joint_targets)
from dexmani_real.robot.safety import SafetyState, transition
from dexmani_real.shm.shared_storage import SharedStorage
from dexmani_real.utils.schema import ARM_JOINT_SHAPE, HAND_JOINT_SHAPE


def _mid(low: tuple[float, ...], high: tuple[float, ...]) -> np.ndarray:
    lo = np.asarray(low, dtype=np.float64)
    hi = np.asarray(high, dtype=np.float64)
    return (lo + hi) / 2.0


def _drain_arm_queue(shared: SharedStorage) -> None:
    while True:
        try:
            shared.arm_action_q.get_nowait()
        except Empty:
            return


def main() -> int:
    arm_mid = _mid(arm_defaults.joint_limit_lower, arm_defaults.joint_limit_upper)
    hand_mid = _mid(hand_defaults.qpos_min_rad, hand_defaults.qpos_max_rad)
    gate = SafetyGate(
        arm_joint_lower_rad=arm_defaults.joint_limit_lower,
        arm_joint_upper_rad=arm_defaults.joint_limit_upper,
        hand_joint_lower_rad=hand_defaults.qpos_min_rad,
        hand_joint_upper_rad=hand_defaults.qpos_max_rad,
    )

    shared = SharedStorage.create(prefix="check_coupled_ack")
    try:
        assert transition(shared, SafetyState.ARMED)
        shared.arm_command_seq.value = 100

        # ── 1. arm-only success (action_id = 101) ──────────────────────
        shared.arm_state_ring.write(make_arm_state_frame(arm_mid, last_cmd_seq=101))
        result = publish_joint_targets(
            shared, arm_mid, None, safety_gate=gate, wait_applied=True
        )
        assert result.status == CommandPublishStatus.APPLIED, result
        assert result.candidate is not None and result.candidate.hand_qpos is None
        _drain_arm_queue(shared)

        # ── 2. arm+hand success (action_id = 102) ──────────────────────
        shared.arm_state_ring.write(make_arm_state_frame(arm_mid, last_cmd_seq=102))
        shared.hand_state_ring.write(
            make_hand_state_frame(hand_mid, last_cmd_seq=102)
        )
        result = publish_joint_targets(
            shared, arm_mid, hand_mid, safety_gate=gate, wait_applied=True
        )
        assert result.status == CommandPublishStatus.APPLIED, result
        assert result.candidate is not None and result.candidate.hand_qpos is not None
        _drain_arm_queue(shared)

        # ── 2b. arm+hand with an unhealthy hand is never published ─────
        # Full hand health is checked before arm enqueue, so a board-faulted
        # hand cannot create a coupled arm/hand mismatch.
        shared.arm_state_ring.write(make_arm_state_frame(arm_mid, last_cmd_seq=103))
        shared.hand_state_ring.write(
            make_hand_state_frame(hand_mid, last_cmd_seq=103, error_state=1)
        )
        result = publish_joint_targets(
            shared, arm_mid, hand_mid, safety_gate=gate, wait_applied=True
        )
        assert result.status == CommandPublishStatus.HAND_FEEDBACK_UNHEALTHY, result
        _drain_arm_queue(shared)

        # ── 3. hand-superseded fails immediately (action_id = 104) ─────
        shared.arm_state_ring.write(make_arm_state_frame(arm_mid, last_cmd_seq=104))
        shared.hand_state_ring.write(
            make_hand_state_frame(hand_mid, last_cmd_seq=105)  # > 104 → superseded
        )
        result = publish_joint_targets(
            shared, arm_mid, hand_mid, safety_gate=gate, wait_applied=True
        )
        assert result.status == CommandPublishStatus.ACK_SUPERSEDED, result
        _drain_arm_queue(shared)
    finally:
        shared.close()

    print("check_coupled_ack: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
