"""Offline checks for non-blocking coupled-command publication and fencing."""

from __future__ import annotations

import threading
import time
import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.config.defaults import hand as hand_defaults
from dexmani_real.control.action import ActionCandidate
from dexmani_real.control.publication import (
    CommandPublishResult,
    CommandPublishStatus,
    poll_coupled_command_acknowledgement,
    publish_joint_targets,
    send_command,
    validate_and_send_candidate,
)
from dexmani_real.control.safety_gate import GateResult, SafetyGate
from dexmani_real.ipc.schema import COUPLED_COMMAND_DTYPE
from dexmani_real.runtime.safety import (
    CoupledCommandTicket,
    SafetyState,
    begin_motion,
    cancel_coupled_command_if_current,
    coupled_command_ticket_allows_execution,
    coupled_command_ticket_is_current,
    publish_coupled_command_if_motion_permitted,
    revoke_motion,
    transition,
)


class _Value:
    def __init__(self, value: int) -> None:
        self.value = value


class _Ring:
    def __init__(self) -> None:
        self.sequence = 0
        self.latest: np.ndarray | None = None

    def write(self, frame: np.ndarray) -> int:
        self.sequence += 1
        self.latest = frame.copy()
        return self.sequence


def _shared(*, state: SafetyState = SafetyState.RUNNING) -> SimpleNamespace:
    return SimpleNamespace(
        safety_state=_Value(int(state)),
        run_generation=_Value(7),
        run_started_monotonic_ns=_Value(1),
        motion_lock=threading.RLock(),
        coupled_cmd_ring=_Ring(),
        active_coupled_command_sequence=_Value(0),
        is_running=_Value(1),
        error_state=_Value(0),
        estop_request=_Value(0),
    )


def _frame(
    action_id: int,
    *,
    arm_present: bool = True,
    hand_present: bool = True,
) -> np.ndarray:
    frame = np.zeros(1, dtype=COUPLED_COMMAND_DTYPE)
    frame["run_generation"][0] = 7
    frame["action_id"][0] = action_id
    frame["arm_present"][0] = int(arm_present)
    frame["hand_present"][0] = int(hand_present)
    return frame


class CoupledCommandPublicationTest(unittest.TestCase):
    def test_publication_immediately_activates_one_coherent_record(self) -> None:
        shared = _shared()
        ticket = publish_coupled_command_if_motion_permitted(
            shared,
            expected_run_generation=7,
            frame=_frame(1),
        )
        assert ticket is not None

        self.assertEqual(shared.coupled_cmd_ring.latest["action_id"][0], 1)
        self.assertTrue(coupled_command_ticket_is_current(shared, ticket=ticket))

    def test_newer_record_atomically_supersedes_older_ticket(self) -> None:
        shared = _shared()
        first = publish_coupled_command_if_motion_permitted(
            shared,
            expected_run_generation=7,
            frame=_frame(1),
        )
        second = publish_coupled_command_if_motion_permitted(
            shared,
            expected_run_generation=7,
            frame=_frame(2),
        )
        assert first is not None and second is not None

        self.assertFalse(coupled_command_ticket_is_current(shared, ticket=first))
        self.assertTrue(coupled_command_ticket_is_current(shared, ticket=second))

    def test_execution_check_includes_runtime_fault_flags(self) -> None:
        shared = _shared()
        ticket = publish_coupled_command_if_motion_permitted(
            shared,
            expected_run_generation=7,
            frame=_frame(1),
        )
        assert ticket is not None
        self.assertTrue(coupled_command_ticket_allows_execution(shared, ticket=ticket))

        shared.estop_request.value = 1

        self.assertTrue(coupled_command_ticket_is_current(shared, ticket=ticket))
        self.assertFalse(coupled_command_ticket_allows_execution(shared, ticket=ticket))

    def test_revoke_invalidates_active_ticket(self) -> None:
        shared = _shared()
        ticket = publish_coupled_command_if_motion_permitted(
            shared,
            expected_run_generation=7,
            frame=_frame(1),
        )
        assert ticket is not None

        self.assertTrue(revoke_motion(shared, SafetyState.ARMED))
        self.assertEqual(shared.run_generation.value, 8)
        self.assertFalse(coupled_command_ticket_is_current(shared, ticket=ticket))

    def test_begin_motion_fails_closed_on_sticky_fault(self) -> None:
        shared = _shared(state=SafetyState.ARMED)
        shared.error_state.value = 1

        self.assertFalse(begin_motion(shared))
        self.assertEqual(shared.safety_state.value, int(SafetyState.ARMED))
        self.assertEqual(shared.run_generation.value, 7)

    def test_legacy_running_to_armed_transition_revokes(self) -> None:
        shared = _shared()
        ticket = publish_coupled_command_if_motion_permitted(
            shared,
            expected_run_generation=7,
            frame=_frame(1),
        )
        assert ticket is not None

        self.assertTrue(transition(shared, SafetyState.ARMED))
        self.assertEqual(shared.run_generation.value, 8)
        self.assertFalse(coupled_command_ticket_is_current(shared, ticket=ticket))

    def test_cancellation_never_revokes_a_newer_ticket(self) -> None:
        shared = _shared()
        first = publish_coupled_command_if_motion_permitted(
            shared,
            expected_run_generation=7,
            frame=_frame(1),
        )
        second = publish_coupled_command_if_motion_permitted(
            shared,
            expected_run_generation=7,
            frame=_frame(2),
        )
        assert first is not None and second is not None

        self.assertFalse(cancel_coupled_command_if_current(shared, ticket=first))
        self.assertEqual(shared.run_generation.value, 7)
        self.assertTrue(coupled_command_ticket_is_current(shared, ticket=second))

    def test_cancellation_revokes_its_current_ticket(self) -> None:
        shared = _shared()
        ticket = publish_coupled_command_if_motion_permitted(
            shared,
            expected_run_generation=7,
            frame=_frame(1, arm_present=False, hand_present=True),
        )
        assert ticket is not None

        self.assertTrue(cancel_coupled_command_if_current(shared, ticket=ticket))
        self.assertEqual(shared.run_generation.value, 8)
        self.assertFalse(coupled_command_ticket_is_current(shared, ticket=ticket))

    def test_send_command_does_not_wait_for_workers(self) -> None:
        shared = _shared()
        now_ns = time.monotonic_ns()
        candidate = ActionCandidate(
            observation_id=1,
            run_generation=7,
            action_id=1,
            created_monotonic_ns=now_ns,
            target_monotonic_ns=now_ns,
            scheduled_target_monotonic_ns=now_ns,
            valid_until_monotonic_ns=now_ns + 500_000_000,
            arm_qpos=np.zeros(7),
        )

        result = send_command(shared, candidate)

        self.assertEqual(result.status, CommandPublishStatus.PUBLISHED)
        self.assertIsNotNone(result.ticket)
        self.assertEqual(
            int(shared.coupled_cmd_ring.latest["created_monotonic_ns"][0]),
            candidate.created_monotonic_ns,
        )
        self.assertLessEqual(
            int(shared.coupled_cmd_ring.latest["created_monotonic_ns"][0]),
            int(shared.coupled_cmd_ring.latest["target_monotonic_ns"][0]),
        )

    def test_validated_execute_candidate_preserves_worker_delivery_margin(
        self,
    ) -> None:
        shared = _shared()
        now_ns = 10_000_000_000
        candidate = ActionCandidate(
            observation_id=1,
            run_generation=7,
            action_id=1,
            created_monotonic_ns=now_ns,
            target_monotonic_ns=now_ns,
            scheduled_target_monotonic_ns=now_ns,
            valid_until_monotonic_ns=now_ns + 8_500_000,
            arm_qpos=np.zeros(7),
        )
        gate = Mock()
        gate.validate.return_value = GateResult(True)
        arm_feedback = SimpleNamespace(qpos=np.zeros(7), last_cmd_seq=0)

        with (
            patch(
                "dexmani_real.control.publication._arm_feedback_snapshot",
                return_value=(arm_feedback, None),
            ),
            patch(
                "dexmani_real.control.publication.time.monotonic_ns",
                return_value=now_ns,
            ),
        ):
            result = validate_and_send_candidate(
                shared,
                candidate,
                gate=gate,
                arm_feedback_max_age_s=0.5,
                hand_feedback_max_age_s=0.5,
                execution_mode="execute",
                minimum_delivery_window_s=1.0 / 16.0,
            )

        self.assertEqual(
            result.status,
            CommandPublishStatus.TEMPORAL_WINDOW_CLOSED,
        )
        self.assertEqual(result.detail, "insufficient worker delivery window")
        gate.validate.assert_called_once()
        self.assertEqual(shared.coupled_cmd_ring.sequence, 0)
        self.assertIsNone(shared.coupled_cmd_ring.latest)

    def test_shadow_validates_full_candidate_without_coupled_command_write(
        self,
    ) -> None:
        shared = _shared()
        now_ns = time.monotonic_ns()
        candidate = ActionCandidate(
            observation_id=1,
            run_generation=7,
            action_id=1,
            created_monotonic_ns=now_ns,
            target_monotonic_ns=now_ns,
            scheduled_target_monotonic_ns=now_ns,
            valid_until_monotonic_ns=now_ns + 500_000_000,
            arm_qpos=np.zeros(7),
        )
        gate = Mock()
        gate.validate.return_value = GateResult(True)
        arm_feedback = SimpleNamespace(qpos=np.zeros(7), last_cmd_seq=0)

        with (
            patch(
                "dexmani_real.control.publication._arm_feedback_snapshot",
                return_value=(arm_feedback, None),
            ),
            patch(
                "dexmani_real.control.publication.publish_coupled_command_if_motion_permitted"
            ) as publish_coupled,
        ):
            result = validate_and_send_candidate(
                shared,
                candidate,
                gate=gate,
                arm_feedback_max_age_s=0.5,
                hand_feedback_max_age_s=0.5,
                execution_mode="shadow",
            )

        self.assertEqual(result.status, CommandPublishStatus.SHADOW_VALIDATED)
        self.assertTrue(result.succeeded)
        self.assertIs(result.candidate, candidate)
        self.assertIsNone(result.ticket)
        gate.validate.assert_called_once()
        publish_coupled.assert_not_called()
        self.assertEqual(shared.coupled_cmd_ring.sequence, 0)
        self.assertIsNone(shared.coupled_cmd_ring.latest)

    def test_execute_publishes_one_fully_validated_coupled_record_to_fake_ring(
        self,
    ) -> None:
        """Exercise the H4 publication tail without workers or hardware SDKs."""
        shared = _shared()
        now_ns = time.monotonic_ns()
        hand_qpos = (
            np.asarray(hand_defaults.qpos_min_rad, dtype=np.float64)
            + np.asarray(hand_defaults.qpos_max_rad, dtype=np.float64)
        ) / 2.0
        candidate = ActionCandidate(
            observation_id=1,
            run_generation=7,
            action_id=1,
            created_monotonic_ns=now_ns,
            target_monotonic_ns=now_ns,
            scheduled_target_monotonic_ns=now_ns,
            valid_until_monotonic_ns=now_ns + 500_000_000,
            arm_qpos=np.zeros(7),
            hand_qpos=hand_qpos,
        )
        gate = SafetyGate(
            arm_joint_lower_rad=tuple(np.full(7, -1.0)),
            arm_joint_upper_rad=tuple(np.full(7, 1.0)),
            hand_joint_lower_rad=tuple(hand_defaults.qpos_min_rad),
            hand_joint_upper_rad=tuple(hand_defaults.qpos_max_rad),
        )
        arm_feedback = SimpleNamespace(qpos=np.zeros(7), last_cmd_seq=0)
        hand_feedback = SimpleNamespace(qpos=hand_qpos, accepted_target_action_id=0)

        with (
            patch(
                "dexmani_real.control.publication._arm_feedback_snapshot",
                return_value=(arm_feedback, None),
            ),
            patch(
                "dexmani_real.control.publication.read_hand_feedback",
                return_value=(hand_feedback, None),
            ),
        ):
            result = validate_and_send_candidate(
                shared,
                candidate,
                gate=gate,
                arm_feedback_max_age_s=0.5,
                hand_feedback_max_age_s=0.5,
                hand_mechanical_lower_rad=np.asarray(
                    hand_defaults.mechanical_qpos_min_rad
                ),
                hand_mechanical_upper_rad=np.asarray(
                    hand_defaults.mechanical_qpos_max_rad
                ),
                execution_mode="execute",
            )

        self.assertEqual(result.status, CommandPublishStatus.PUBLISHED)
        self.assertIs(result.candidate, candidate)
        self.assertIsNotNone(result.ticket)
        self.assertEqual(shared.coupled_cmd_ring.sequence, 1)
        assert shared.coupled_cmd_ring.latest is not None
        self.assertEqual(shared.coupled_cmd_ring.latest["action_id"][0], 1)
        self.assertEqual(shared.coupled_cmd_ring.latest["arm_present"][0], 1)
        self.assertEqual(shared.coupled_cmd_ring.latest["hand_present"][0], 1)
        np.testing.assert_array_equal(
            shared.coupled_cmd_ring.latest["arm_qpos"][0], candidate.arm_qpos
        )
        np.testing.assert_array_equal(
            shared.coupled_cmd_ring.latest["hand_qpos"][0], candidate.hand_qpos
        )

    def test_policy_hand_endpoint_is_feedback_rate_shaped_and_revalidated(
        self,
    ) -> None:
        """A contact-blocked learned endpoint becomes one exact IPC command."""
        shared = _shared()
        now_ns = time.monotonic_ns()
        hand_low = np.asarray(hand_defaults.qpos_min_rad, dtype=np.float64)
        hand_high = np.asarray(hand_defaults.qpos_max_rad, dtype=np.float64)
        measured_hand_qpos = hand_low.copy()
        raw_hand_qpos = measured_hand_qpos + np.minimum(
            0.4,
            (hand_high - hand_low) / 2.0,
        )
        self.assertTrue(np.all(raw_hand_qpos < hand_high))
        expected_hand_qpos = measured_hand_qpos + np.minimum(
            raw_hand_qpos - measured_hand_qpos,
            0.3,
        )
        self.assertTrue(np.any(raw_hand_qpos - measured_hand_qpos > 0.3))
        candidate = ActionCandidate(
            observation_id=1,
            run_generation=7,
            action_id=1,
            created_monotonic_ns=now_ns,
            target_monotonic_ns=now_ns,
            scheduled_target_monotonic_ns=now_ns,
            valid_until_monotonic_ns=now_ns + 500_000_000,
            arm_qpos=np.zeros(7),
            hand_qpos=raw_hand_qpos,
        )
        gate = SafetyGate(
            arm_joint_lower_rad=tuple(np.full(7, -1.0)),
            arm_joint_upper_rad=tuple(np.full(7, 1.0)),
            hand_joint_lower_rad=tuple(hand_low),
            hand_joint_upper_rad=tuple(hand_high),
        )
        arm_feedback = SimpleNamespace(qpos=np.zeros(7), last_cmd_seq=0)
        hand_feedback = SimpleNamespace(
            qpos=measured_hand_qpos,
            accepted_target_action_id=0,
        )

        with (
            patch(
                "dexmani_real.control.publication._arm_feedback_snapshot",
                return_value=(arm_feedback, None),
            ),
            patch(
                "dexmani_real.control.publication.read_hand_feedback",
                return_value=(hand_feedback, None),
            ),
            patch.object(gate, "validate", wraps=gate.validate) as validate,
        ):
            result = validate_and_send_candidate(
                shared,
                candidate,
                gate=gate,
                arm_feedback_max_age_s=0.5,
                hand_feedback_max_age_s=0.5,
                hand_mechanical_lower_rad=np.asarray(
                    hand_defaults.mechanical_qpos_min_rad
                ),
                hand_mechanical_upper_rad=np.asarray(
                    hand_defaults.mechanical_qpos_max_rad
                ),
                hand_command_max_delta_rad_per_tick=0.3,
                execution_mode="execute",
            )

        self.assertEqual(result.status, CommandPublishStatus.PUBLISHED)
        assert result.candidate is not None
        np.testing.assert_array_equal(
            result.candidate.hand_qpos,
            expected_hand_qpos,
        )
        self.assertEqual(validate.call_count, 2)
        first_candidate = validate.call_args_list[0].args[0]
        second_candidate = validate.call_args_list[1].args[0]
        np.testing.assert_array_equal(first_candidate.hand_qpos, raw_hand_qpos)
        np.testing.assert_array_equal(
            second_candidate.hand_qpos,
            expected_hand_qpos,
        )
        assert shared.coupled_cmd_ring.latest is not None
        np.testing.assert_array_equal(
            shared.coupled_cmd_ring.latest["hand_qpos"][0],
            expected_hand_qpos,
        )

    def test_execute_generation_change_after_gate_does_not_write(self) -> None:
        """A post-gate lifecycle revocation wins over a stale H4 candidate."""
        shared = _shared()
        now_ns = time.monotonic_ns()
        candidate = ActionCandidate(
            observation_id=1,
            run_generation=7,
            action_id=1,
            created_monotonic_ns=now_ns,
            target_monotonic_ns=now_ns,
            scheduled_target_monotonic_ns=now_ns,
            valid_until_monotonic_ns=now_ns + 500_000_000,
            arm_qpos=np.zeros(7),
        )
        gate = Mock()

        def revoke_generation(*_args, **_kwargs) -> GateResult:
            shared.run_generation.value = 8
            return GateResult(True)

        gate.validate.side_effect = revoke_generation
        arm_feedback = SimpleNamespace(qpos=np.zeros(7), last_cmd_seq=0)

        with patch(
            "dexmani_real.control.publication._arm_feedback_snapshot",
            return_value=(arm_feedback, None),
        ):
            result = validate_and_send_candidate(
                shared,
                candidate,
                gate=gate,
                arm_feedback_max_age_s=0.5,
                hand_feedback_max_age_s=0.5,
                execution_mode="execute",
            )

        self.assertEqual(result.status, CommandPublishStatus.RUN_GENERATION_GATED)
        self.assertEqual(shared.coupled_cmd_ring.sequence, 0)
        self.assertIsNone(shared.coupled_cmd_ring.latest)

    def test_h4_acknowledgement_requires_both_arm_and_hand_workers(self) -> None:
        """The H4 coordinator may finish only after the exact coupled ticket."""
        shared = _shared()
        shared.active_coupled_command_sequence.value = 3
        now_ns = time.monotonic_ns()
        candidate = ActionCandidate(
            observation_id=1,
            run_generation=7,
            action_id=5,
            created_monotonic_ns=now_ns,
            target_monotonic_ns=now_ns,
            scheduled_target_monotonic_ns=now_ns,
            valid_until_monotonic_ns=now_ns + 500_000_000,
            arm_qpos=np.zeros(7),
            hand_qpos=np.zeros(12),
        )
        ticket = CoupledCommandTicket(7, 3)
        arm_feedback = SimpleNamespace(qpos=np.zeros(7), last_cmd_seq=5)
        hand_pending = SimpleNamespace(qpos=np.zeros(12), accepted_target_action_id=0)
        hand_applied = SimpleNamespace(qpos=np.zeros(12), accepted_target_action_id=5)

        with (
            patch(
                "dexmani_real.control.publication._arm_feedback_snapshot",
                return_value=(arm_feedback, None),
            ),
            patch(
                "dexmani_real.control.publication.read_hand_feedback",
                side_effect=[(hand_pending, None), (hand_applied, None)],
            ),
        ):
            pending = poll_coupled_command_acknowledgement(
                shared,
                candidate,
                ticket=ticket,
                arm_feedback_max_age_s=0.5,
                hand_feedback_max_age_s=0.5,
            )
            applied = poll_coupled_command_acknowledgement(
                shared,
                candidate,
                ticket=ticket,
                arm_feedback_max_age_s=0.5,
                hand_feedback_max_age_s=0.5,
            )

        self.assertEqual(pending.status, CommandPublishStatus.ACK_PENDING)
        self.assertEqual(applied.status, CommandPublishStatus.APPLIED)

    def test_policy_hand_roundoff_is_canonicalized_before_strict_gate(self) -> None:
        shared = _shared()
        now_ns = time.monotonic_ns()
        hand_qpos = np.asarray(hand_defaults.qpos_min_rad, dtype=np.float64)
        hand_qpos[5] -= 3.9791546155298896e-8
        candidate = ActionCandidate(
            observation_id=1,
            run_generation=7,
            action_id=1,
            created_monotonic_ns=now_ns,
            target_monotonic_ns=now_ns,
            scheduled_target_monotonic_ns=now_ns,
            valid_until_monotonic_ns=now_ns + 500_000_000,
            arm_qpos=np.zeros(7),
            hand_qpos=hand_qpos,
        )
        gate = SafetyGate(
            arm_joint_lower_rad=tuple(np.full(7, -1.0)),
            arm_joint_upper_rad=tuple(np.full(7, 1.0)),
            hand_joint_lower_rad=tuple(hand_defaults.qpos_min_rad),
            hand_joint_upper_rad=tuple(hand_defaults.qpos_max_rad),
        )
        arm_feedback = SimpleNamespace(qpos=np.zeros(7), last_cmd_seq=0)
        hand_feedback = SimpleNamespace(qpos=hand_qpos, accepted_target_action_id=0)

        with (
            patch(
                "dexmani_real.control.publication._arm_feedback_snapshot",
                return_value=(arm_feedback, None),
            ),
            patch(
                "dexmani_real.control.publication.read_hand_feedback",
                return_value=(hand_feedback, None),
            ),
        ):
            result = validate_and_send_candidate(
                shared,
                candidate,
                gate=gate,
                arm_feedback_max_age_s=0.5,
                hand_feedback_max_age_s=0.5,
                hand_mechanical_lower_rad=np.asarray(
                    hand_defaults.mechanical_qpos_min_rad
                ),
                hand_mechanical_upper_rad=np.asarray(
                    hand_defaults.mechanical_qpos_max_rad
                ),
                canonicalize_policy_hand_roundoff=True,
                execution_mode="shadow",
            )

        self.assertEqual(result.status, CommandPublishStatus.SHADOW_VALIDATED)
        self.assertTrue(result.hand_roundoff_canonicalized)
        assert result.candidate is not None
        self.assertEqual(result.candidate.hand_qpos[5], hand_defaults.qpos_min_rad[5])
        self.assertEqual(shared.coupled_cmd_ring.sequence, 0)

    def test_wait_applied_exits_when_ticket_loses_ownership(self) -> None:
        shared = _shared()
        now_ns = time.monotonic_ns()
        candidate = ActionCandidate(
            observation_id=1,
            run_generation=7,
            action_id=1,
            created_monotonic_ns=now_ns,
            target_monotonic_ns=now_ns,
            scheduled_target_monotonic_ns=now_ns,
            valid_until_monotonic_ns=now_ns + 500_000_000,
            arm_qpos=np.zeros(7),
        )
        published = CommandPublishResult(
            CommandPublishStatus.PUBLISHED,
            candidate=candidate,
            ticket=CoupledCommandTicket(7, 3),
        )
        arm_feedback = SimpleNamespace(qpos=np.zeros(7), last_cmd_seq=0)

        with (
            patch(
                "dexmani_real.control.publication.check_runtime_gate",
                return_value=None,
            ),
            patch(
                "dexmani_real.control.publication.build_action_candidate",
                return_value=candidate,
            ),
            patch(
                "dexmani_real.control.publication.validate_and_send_candidate",
                return_value=published,
            ),
            patch(
                "dexmani_real.control.publication._arm_feedback_snapshot",
                return_value=(arm_feedback, None),
            ),
            patch(
                "dexmani_real.control.publication.coupled_command_ticket_is_current",
                return_value=False,
            ),
        ):
            result = publish_joint_targets(
                shared,
                np.zeros(7),
                safety_gate=cast(Any, object()),
                wait_applied=True,
                apply_timeout_s=10.0,
                arm_feedback_max_age_s=0.5,
                hand_feedback_max_age_s=0.5,
            )

        self.assertEqual(result.status, CommandPublishStatus.ACK_SUPERSEDED)

    def test_wait_timeout_revokes_its_still_current_ticket(self) -> None:
        shared = _shared()
        shared.active_coupled_command_sequence.value = 3
        now_ns = time.monotonic_ns()
        candidate = ActionCandidate(
            observation_id=1,
            run_generation=7,
            action_id=1,
            created_monotonic_ns=now_ns,
            target_monotonic_ns=now_ns,
            scheduled_target_monotonic_ns=now_ns,
            valid_until_monotonic_ns=now_ns + 500_000_000,
            arm_qpos=np.zeros(7),
        )
        published = CommandPublishResult(
            CommandPublishStatus.PUBLISHED,
            candidate=candidate,
            ticket=CoupledCommandTicket(7, 3),
        )

        with (
            patch(
                "dexmani_real.control.publication.check_runtime_gate",
                return_value=None,
            ),
            patch(
                "dexmani_real.control.publication.build_action_candidate",
                return_value=candidate,
            ),
            patch(
                "dexmani_real.control.publication.validate_and_send_candidate",
                return_value=published,
            ),
        ):
            result = publish_joint_targets(
                shared,
                np.zeros(7),
                safety_gate=cast(Any, object()),
                wait_applied=True,
                apply_timeout_s=1e-9,
                arm_feedback_max_age_s=0.5,
                hand_feedback_max_age_s=0.5,
            )

        self.assertEqual(result.status, CommandPublishStatus.ACK_TIMEOUT)
        self.assertEqual(shared.run_generation.value, 8)
        self.assertEqual(shared.active_coupled_command_sequence.value, 0)


if __name__ == "__main__":
    unittest.main()
