"""Offline regressions for learned-policy coordinator failure semantics."""

from __future__ import annotations

import threading
import unittest

import numpy as np

from dexmani_real.control.action import ActionCandidate
from dexmani_real.control.publication import (
    CommandPublishResult,
    CommandPublishStatus,
    PolicyEndpointDisposition,
    classify_policy_endpoint_disposition,
)
from dexmani_real.control.safety_gate import GateRejectCode, SafetyGate
from dexmani_real.deployment.action_buffer import ActionBuffer, BufferedPlan, PushStatus
from dexmani_real.deployment.contracts import JointActionChunk
from dexmani_real.deployment.coordinator import (
    _AcknowledgementAction,
    _classify_acknowledgement,
    _command_watchdog_abort_reason,
    _end_policy_run,
    _PendingAcknowledgement,
)
from dexmani_real.runtime.safety import CoupledCommandTicket, SafetyState


class _Value:
    def __init__(self, value: int | bool) -> None:
        self.value = value


class _FakeShared:
    def __init__(self) -> None:
        self.is_running = _Value(True)
        self.error_state = _Value(False)
        self.estop_request = _Value(False)
        self.physical_home_completed = _Value(True)
        self.safety_state = _Value(int(SafetyState.RUNNING))
        self.run_generation = _Value(5)
        self.run_started_monotonic_ns = _Value(100)
        self.active_coupled_command_sequence = _Value(9)
        self.motion_lock = threading.Lock()


def _candidate(*, run_generation: int = 1, arm_value: float = 0.0) -> ActionCandidate:
    arm_qpos = np.zeros(7, dtype=np.float64)
    arm_qpos[0] = arm_value
    return ActionCandidate(
        observation_id=1,
        run_generation=run_generation,
        action_id=1,
        created_monotonic_ns=100,
        scheduled_target_monotonic_ns=90,
        target_monotonic_ns=100,
        valid_until_monotonic_ns=300,
        arm_qpos=arm_qpos,
        hand_qpos=np.zeros(12, dtype=np.float64),
    )


def _plan(
    *,
    run_generation: int,
    observation_id: int = 1,
    plan_id: int = 1,
    deadline_ns: int = 100,
) -> BufferedPlan:
    return BufferedPlan(
        plan_id=plan_id,
        run_generation=run_generation,
        observation_id=observation_id,
        observation_anchor_ns=20,
        observation_latest_source_ns=10,
        inference_finished_ns=30,
        deadline_ns=deadline_ns,
        chunk=JointActionChunk(
            arm_qpos=np.zeros((1, 7), dtype=np.float64),
            hand_qpos=np.zeros((1, 12), dtype=np.float64),
            target_monotonic_ns=np.array([50], dtype=np.uint64),
            valid_mask=np.array([1], dtype=np.uint8),
        ),
    )


def _pending_acknowledgement() -> _PendingAcknowledgement:
    return _PendingAcknowledgement(
        candidate=_candidate(),
        ticket=CoupledCommandTicket(run_generation=1, ring_sequence=2),
        published_monotonic_ns=100,
        deadline_monotonic_ns=200,
    )


class PolicyCoordinatorRegressionTest(unittest.TestCase):
    def test_wrong_generation_plan_is_rejected(self) -> None:
        action_buffer = ActionBuffer(max_buffered_plans=2)
        action_buffer.reset(run_generation=2)

        result = action_buffer.push(_plan(run_generation=1), now_ns=40)

        self.assertIs(result.status, PushStatus.WRONG_GENERATION)
        self.assertEqual(action_buffer.plan_count, 0)

    def test_stale_plan_is_rejected(self) -> None:
        action_buffer = ActionBuffer(max_buffered_plans=2)
        action_buffer.reset(run_generation=1)
        self.assertTrue(
            action_buffer.push(
                _plan(run_generation=1, observation_id=2, plan_id=2),
                now_ns=40,
            ).accepted
        )

        stale_identity = action_buffer.push(
            _plan(run_generation=1, observation_id=1, plan_id=1),
            now_ns=40,
        )
        expired = action_buffer.push(
            _plan(
                run_generation=1,
                observation_id=3,
                plan_id=3,
                deadline_ns=40,
            ),
            now_ns=40,
        )

        self.assertIs(stale_identity.status, PushStatus.STALE_IDENTITY)
        self.assertIs(expired.status, PushStatus.DEADLINE_CLOSED)

    def test_safety_gate_rejection_discards_motion_endpoint(self) -> None:
        gate = SafetyGate(
            arm_joint_lower_rad=(-1.0,) * 7,
            arm_joint_upper_rad=(1.0,) * 7,
            hand_joint_lower_rad=(-1.0,) * 12,
            hand_joint_upper_rad=(1.0,) * 12,
        )
        candidate = _candidate(arm_value=2.0)

        gate_result = gate.validate(
            candidate,
            current_arm_qpos=np.zeros(7, dtype=np.float64),
            current_hand_qpos=np.zeros(12, dtype=np.float64),
            run_generation=1,
        )
        publication_result = CommandPublishResult(
            CommandPublishStatus.GATE_REJECTED,
            candidate=candidate,
            gate_code=gate_result.code,
        )

        self.assertFalse(gate_result.accepted)
        self.assertIs(gate_result.code, GateRejectCode.ARM_JOINT_LIMIT)
        self.assertIs(
            classify_policy_endpoint_disposition(
                publication_result,
                hand_limit_nesting_valid=True,
            ),
            PolicyEndpointDisposition.DISCARD_MOTION,
        )

    def test_ack_timeout_faults_at_deadline(self) -> None:
        waiting = _classify_acknowledgement(
            _pending_acknowledgement(),
            CommandPublishStatus.ACK_PENDING,
            poll_started_monotonic_ns=199,
            observed_monotonic_ns=200,
        )
        deadline_timeout = _classify_acknowledgement(
            _pending_acknowledgement(),
            CommandPublishStatus.ACK_PENDING,
            poll_started_monotonic_ns=200,
            observed_monotonic_ns=201,
        )
        late_applied = _classify_acknowledgement(
            _pending_acknowledgement(),
            CommandPublishStatus.APPLIED,
            poll_started_monotonic_ns=199,
            observed_monotonic_ns=201,
        )
        applied = _classify_acknowledgement(
            _pending_acknowledgement(),
            CommandPublishStatus.APPLIED,
            poll_started_monotonic_ns=199,
            observed_monotonic_ns=200,
        )

        self.assertIs(waiting.action, _AcknowledgementAction.WAIT)
        self.assertIs(
            deadline_timeout.action,
            _AcknowledgementAction.FAULT_TIMEOUT,
        )
        self.assertEqual(
            deadline_timeout.reason,
            "arm/hand acknowledgement timeout",
        )
        self.assertIs(late_applied.action, _AcknowledgementAction.FAULT_TIMEOUT)
        self.assertEqual(
            late_applied.reason,
            "worker acknowledgement arrived after deadline",
        )
        self.assertIs(applied.action, _AcknowledgementAction.APPLIED)
        self.assertEqual(applied.latency_ms, 0.0001)

    def test_worker_ack_rejection_faults(self) -> None:
        for status in (
            CommandPublishStatus.ACK_SUPERSEDED,
            CommandPublishStatus.ARM_FEEDBACK_UNHEALTHY,
            CommandPublishStatus.STICKY_FAULT,
        ):
            with self.subTest(status=status):
                decision = _classify_acknowledgement(
                    _pending_acknowledgement(),
                    status,
                    poll_started_monotonic_ns=150,
                    observed_monotonic_ns=151,
                )

                self.assertIs(
                    decision.action,
                    _AcknowledgementAction.FAULT_REJECTED,
                )
                self.assertIn(status.value, decision.reason)

    def test_operator_stop_revokes_motion_and_home_authorization(self) -> None:
        shared = _FakeShared()
        generation_before_stop = int(shared.run_generation.value)

        _end_policy_run(shared, "operator stop", abort=False)

        self.assertEqual(shared.safety_state.value, int(SafetyState.ARMED))
        self.assertGreater(shared.run_generation.value, generation_before_stop)
        self.assertEqual(shared.active_coupled_command_sequence.value, 0)
        self.assertEqual(shared.run_started_monotonic_ns.value, 0)
        self.assertFalse(shared.physical_home_completed.value)

    def test_first_command_timeout_is_distinct_from_initial_wait(self) -> None:
        before_deadline = _command_watchdog_abort_reason(
            now_monotonic_ns=200,
            run_started_monotonic_ns=100,
            last_valid_command_monotonic_ns=None,
            first_command_timeout_ns=100,
            command_silence_timeout_ns=50,
        )
        after_deadline = _command_watchdog_abort_reason(
            now_monotonic_ns=201,
            run_started_monotonic_ns=100,
            last_valid_command_monotonic_ns=None,
            first_command_timeout_ns=100,
            command_silence_timeout_ns=50,
        )

        self.assertIsNone(before_deadline)
        self.assertEqual(after_deadline, "first command timeout")

    def test_command_silence_timeout_starts_at_last_valid_command(self) -> None:
        before_deadline = _command_watchdog_abort_reason(
            now_monotonic_ns=200,
            run_started_monotonic_ns=1,
            last_valid_command_monotonic_ns=150,
            first_command_timeout_ns=10,
            command_silence_timeout_ns=50,
        )
        after_deadline = _command_watchdog_abort_reason(
            now_monotonic_ns=201,
            run_started_monotonic_ns=1,
            last_valid_command_monotonic_ns=150,
            first_command_timeout_ns=10,
            command_silence_timeout_ns=50,
        )

        self.assertIsNone(before_deadline)
        self.assertEqual(after_deadline, "command silence timeout")


if __name__ == "__main__":
    unittest.main()
