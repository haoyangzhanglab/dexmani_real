"""Offline behavior tests for the controller and worker safety boundaries."""

from __future__ import annotations

import threading
from unittest import mock

import numpy as np

from dexmani_real.control.action import ActionCandidate
from dexmani_real.control.hand_homing import publish_hand_home_and_wait_accepted
from dexmani_real.control.publication import (
    PUBLISH_REASON_EXPIRED,
    PUBLISH_REASON_GENERATION,
    build_action_candidate,
    command_publishability_reason,
)
from dexmani_real.control.safety_gate import GateRejectCode, SafetyGate
from dexmani_real.robot.command_validation import (
    check_worker_arm_target,
    check_worker_hand_target,
)
from dexmani_real.runtime.safety import (
    CoupledCommandTicket,
    SafetyState,
    coupled_command_ticket_allows_execution,
    coupled_command_ticket_is_current,
)


class _Value:
    def __init__(self, value):
        self.value = value


class _Counter(_Value):
    def __init__(self, value):
        super().__init__(value)
        self._lock = threading.Lock()

    def get_lock(self):
        return self._lock


class _Ring:
    def __init__(self, latest_sequence=0):
        self.latest_sequence = latest_sequence


class _Shared:
    """Small in-memory stand-in for the runtime permit and command identity."""

    def __init__(self, *, state=SafetyState.RUNNING, generation=7, sequence=4):
        self.motion_lock = threading.RLock()
        self.safety_state = _Value(int(state))
        self.run_generation = _Value(generation)
        self.arm_command_seq = _Counter(0)
        self.coupled_cmd_ring = _Ring(sequence)
        self.estop_request = _Value(False)
        self.error_state = _Value(False)
        self.is_running = _Value(True)


def _candidate(*, arm=None, hand=None, valid_until=1_000):
    return ActionCandidate(
        observation_id=1,
        run_generation=7,
        action_id=1,
        created_monotonic_ns=100,
        scheduled_target_monotonic_ns=100,
        target_monotonic_ns=100,
        valid_until_monotonic_ns=valid_until,
        arm_qpos=np.zeros(7) if arm is None else np.asarray(arm, dtype=np.float64),
        hand_qpos=np.zeros(12) if hand is None else np.asarray(hand, dtype=np.float64),
    )


def _gate(**kwargs):
    return SafetyGate(
        arm_joint_lower_rad=(-1.0,) * 7,
        arm_joint_upper_rad=(1.0,) * 7,
        hand_joint_lower_rad=(-1.0,) * 12,
        hand_joint_upper_rad=(1.0,) * 12,
        **kwargs,
    )


def _validate(gate, candidate):
    return gate.validate(
        candidate,
        current_arm_qpos=np.zeros(7),
        current_hand_qpos=np.zeros(12),
    )


def test_safety_gate_rejects_malformed_targets_as_invalid():
    wrong_shape = _validate(_gate(), _candidate(arm=np.zeros(6)))
    nonfinite = _validate(_gate(), _candidate(hand=np.full(12, np.nan)))

    assert not wrong_shape.accepted
    assert wrong_shape.code is GateRejectCode.INVALID_TARGET
    assert not nonfinite.accepted
    assert nonfinite.code is GateRejectCode.INVALID_TARGET


def test_safety_gate_rejects_operational_arm_and_hand_limits():
    arm = np.zeros(7)
    arm[0] = 1.1
    arm_result = _validate(_gate(), _candidate(arm=arm))

    hand = np.zeros(12)
    hand[0] = 1.1
    hand_result = _validate(_gate(), _candidate(hand=hand))

    assert not arm_result.accepted
    assert arm_result.code is GateRejectCode.ARM_JOINT_LIMIT
    assert not hand_result.accepted
    assert hand_result.code is GateRejectCode.HAND_JOINT_LIMIT


def test_safety_gate_rejects_hand_delta():
    hand = np.full(12, 0.2)
    result = _validate(
        _gate(max_hand_delta_rad=0.1),
        _candidate(hand=hand),
    )

    assert not result.accepted
    assert result.code is GateRejectCode.HAND_DELTA_LIMIT


def test_safety_gate_fails_closed_for_workspace_and_collision():
    workspace = _validate(
        _gate(workspace_check=lambda _start, _end: False),
        _candidate(),
    )
    collision = _validate(
        _gate(collision_check=lambda _a0, _a1, _h0, _h1: False),
        _candidate(),
    )

    def _workspace_failure(_start, _end):
        raise RuntimeError("workspace unavailable")

    def _collision_failure(_a0, _a1, _h0, _h1):
        raise RuntimeError("collision model unavailable")

    workspace_failure = _validate(
        _gate(workspace_check=_workspace_failure),
        _candidate(),
    )
    collision_failure = _validate(
        _gate(collision_check=_collision_failure),
        _candidate(),
    )

    assert workspace.code is GateRejectCode.WORKSPACE
    assert collision.code is GateRejectCode.COLLISION_TRANSITION
    assert workspace_failure.code is GateRejectCode.WORKSPACE_CHECK_FAILED
    assert collision_failure.code is GateRejectCode.COLLISION_CHECK_FAILED


def test_builder_rejects_future_policy_endpoint_and_expired_delivery():
    shared = _Shared()
    future = build_action_candidate(
        shared,
        np.zeros(7),
        np.zeros(12),
        now_ns=100,
        scheduled_target_monotonic_ns=101,
    )
    expired = build_action_candidate(
        shared,
        np.zeros(7),
        np.zeros(12),
        now_ns=100,
        valid_until_monotonic_ns=99,
    )

    assert future is None
    assert expired is None


def test_worker_arm_guard_rejects_nonfinite_mechanical_and_jump():
    lower = np.full(7, -1.0)
    upper = np.full(7, 1.0)
    previous = np.zeros(7)

    nonfinite = np.zeros(7)
    nonfinite[0] = np.nan
    mechanical = np.zeros(7)
    mechanical[0] = 1.1
    jump = np.zeros(7)
    jump[0] = 0.2

    assert (
        check_worker_arm_target(
            nonfinite,
            previous_target_qpos_rad=previous,
            joint_limit_lower_rad=lower,
            joint_limit_upper_rad=upper,
            max_command_jump_rad=1.0,
        )
        == "non-finite target"
    )
    assert (
        check_worker_arm_target(
            mechanical,
            previous_target_qpos_rad=previous,
            joint_limit_lower_rad=lower,
            joint_limit_upper_rad=upper,
            max_command_jump_rad=1.0,
        )
        == "joint limit violation"
    )
    assert (
        check_worker_arm_target(
            jump,
            previous_target_qpos_rad=previous,
            joint_limit_lower_rad=lower,
            joint_limit_upper_rad=upper,
            max_command_jump_rad=0.1,
        )
        == "command jump limit violation"
    )


def test_worker_hand_guard_rejects_nonfinite_and_mechanical_limit():
    lower = np.full(12, -1.0)
    upper = np.full(12, 1.0)
    nonfinite = np.zeros(12)
    nonfinite[0] = np.inf
    mechanical = np.zeros(12)
    mechanical[0] = 1.1

    assert (
        check_worker_hand_target(
            nonfinite,
            mechanical_lower_rad=lower,
            mechanical_upper_rad=upper,
        )
        == "non-finite target"
    )
    assert (
        check_worker_hand_target(
            mechanical,
            mechanical_lower_rad=lower,
            mechanical_upper_rad=upper,
        )
        == "mechanical joint limit violation"
    )


def test_ticket_authority_and_expiry_are_checked_before_execution():
    shared = _Shared()
    current_ticket = CoupledCommandTicket(
        run_generation=7,
        ring_sequence=4,
        valid_until_monotonic_ns=10**18,
    )
    superseded_ticket = CoupledCommandTicket(
        run_generation=7,
        ring_sequence=3,
        valid_until_monotonic_ns=10**18,
    )
    expired_ticket = CoupledCommandTicket(
        run_generation=7,
        ring_sequence=4,
        valid_until_monotonic_ns=1,
    )

    assert coupled_command_ticket_is_current(shared, ticket=current_ticket)
    assert coupled_command_ticket_allows_execution(shared, ticket=current_ticket)
    assert not coupled_command_ticket_is_current(shared, ticket=superseded_ticket)
    assert not coupled_command_ticket_allows_execution(shared, ticket=expired_ticket)

    shared.run_generation.value = 8
    assert not coupled_command_ticket_is_current(shared, ticket=current_ticket)


def test_publishability_rejects_generation_mismatch_and_expiry():
    generation_mismatch_shared = _Shared(generation=8)
    generation_mismatch = _candidate(valid_until=10**18)
    expired = _candidate(valid_until=1)

    assert (
        command_publishability_reason(generation_mismatch_shared, generation_mismatch)
        == PUBLISH_REASON_GENERATION
    )
    assert command_publishability_reason(_Shared(), expired) == PUBLISH_REASON_EXPIRED


def test_hand_home_rejects_invalid_target_before_publish_side_effect():
    invalid_home = np.full(12, 2.0)
    with mock.patch("dexmani_real.control.hand_homing.publish_command") as publish:
        try:
            publish_hand_home_and_wait_accepted(
                object(),
                invalid_home,
                command_lower_rad=np.full(12, -1.0),
                command_upper_rad=np.full(12, 1.0),
                mechanical_lower_rad=np.full(12, -2.0),
                mechanical_upper_rad=np.full(12, 2.0),
                hand_feedback_max_age_s=0.1,
            )
        except ValueError as exc:
            assert "operational joint limits" in str(exc)
        else:
            raise AssertionError("invalid hand home was accepted")
    publish.assert_not_called()
