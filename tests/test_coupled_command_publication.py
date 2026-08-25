"""Offline checks for non-blocking coupled-command publication and fencing."""

from __future__ import annotations

import threading
import time
import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import numpy as np

from dexmani_real.control.action import ActionCandidate
from dexmani_real.control.publication import (
    CommandPublishResult,
    CommandPublishStatus,
    publish_joint_targets,
    send_command,
)
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
            target_monotonic_ns=now_ns + 50_000_000,
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

    def test_wait_applied_exits_when_ticket_loses_ownership(self) -> None:
        shared = _shared()
        now_ns = time.monotonic_ns()
        candidate = ActionCandidate(
            observation_id=1,
            run_generation=7,
            action_id=1,
            created_monotonic_ns=now_ns,
            target_monotonic_ns=now_ns,
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
