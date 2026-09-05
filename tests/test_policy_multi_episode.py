"""Offline contracts for repeated policy episodes in one process."""

from __future__ import annotations

import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from dexmani_real.deployment.operator import run_operator_control
from dexmani_real.runtime.safety import (
    CoupledCommandTicket,
    SafetyState,
    StopRequest,
    begin_motion,
    begin_requested_motion,
    cancel_coupled_command_if_current,
    coupled_command_ticket_allows_execution,
    coupled_command_ticket_is_current,
    request_policy_start,
    request_policy_stop,
    revoke_motion,
)
from dexmani_real.teleop.keyboard import ControlSignal


class _Value:
    def __init__(self, value: int | bool) -> None:
        self.value = value


class _FakeShared:
    def __init__(self) -> None:
        self.is_running = _Value(True)
        self.error_state = _Value(False)
        self.estop_request = _Value(False)
        self.quit_requested = _Value(False)
        self.start_request = _Value(False)
        self.stop_request = _Value(int(StopRequest.NONE))
        self.physical_home_completed = _Value(False)
        self.safety_state = _Value(int(SafetyState.ARMED))
        self.run_generation = _Value(0)
        self.run_started_monotonic_ns = _Value(0)
        self.coupled_cmd_ring = SimpleNamespace(latest_sequence=0)
        self.motion_lock = threading.Lock()


class _FakeKeyboard:
    def __init__(self, shared: _FakeShared, batches: list[list[ControlSignal]]) -> None:
        self._shared = shared
        self._batches = list(batches)
        self.started = False
        self.stopped = False

    @property
    def healthy(self) -> bool:
        return True

    @property
    def estop_latched(self) -> bool:
        return False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def poll(self, *, timeout: float) -> list[ControlSignal]:
        del timeout
        if self._batches:
            return self._batches.pop(0)
        self._shared.is_running.value = False
        return []

    def drain_signal(self, _signal: ControlSignal) -> int:
        return 0


class PolicyMultiEpisodeTest(unittest.TestCase):
    def test_newer_stop_prevents_pending_begin(self) -> None:
        shared = _FakeShared()
        self.assertTrue(request_policy_start(shared, require_physical_home=False))
        self.assertTrue(request_policy_stop(shared))

        self.assertIsNone(begin_requested_motion(shared))
        self.assertEqual(shared.safety_state.value, int(SafetyState.ARMED))
        self.assertFalse(shared.start_request.value)
        self.assertEqual(shared.stop_request.value, int(StopRequest.OPERATOR))

    def test_begin_requires_prior_stop_acknowledgement(self) -> None:
        shared = _FakeShared()
        self.assertTrue(request_policy_stop(shared))
        self.assertFalse(request_policy_start(shared, require_physical_home=False))
        shared.stop_request.value = int(StopRequest.NONE)
        self.assertTrue(request_policy_start(shared, require_physical_home=False))

        epoch = begin_requested_motion(shared)
        self.assertIsNotNone(epoch)
        self.assertEqual(shared.safety_state.value, int(SafetyState.RUNNING))
        self.assertFalse(shared.start_request.value)
        self.assertEqual(shared.stop_request.value, int(StopRequest.NONE))

    def test_physical_begin_requires_home_inside_request_lock(self) -> None:
        shared = _FakeShared()
        self.assertFalse(request_policy_start(shared, require_physical_home=True))
        shared.physical_home_completed.value = True
        self.assertTrue(request_policy_start(shared, require_physical_home=True))

    def test_second_episode_advances_generation_and_invalidates_old_ticket(
        self,
    ) -> None:
        shared = _FakeShared()
        self.assertTrue(begin_motion(shared))
        first_generation = int(shared.run_generation.value)

        old_ticket = CoupledCommandTicket(
            run_generation=first_generation,
            ring_sequence=7,
            valid_until_monotonic_ns=time.monotonic_ns() + 1_000_000_000,
        )
        shared.coupled_cmd_ring.latest_sequence = old_ticket.ring_sequence
        self.assertTrue(coupled_command_ticket_is_current(shared, ticket=old_ticket))

        self.assertTrue(revoke_motion(shared, SafetyState.ARMED))
        self.assertFalse(coupled_command_ticket_is_current(shared, ticket=old_ticket))
        self.assertTrue(begin_motion(shared))
        second_generation = int(shared.run_generation.value)
        self.assertGreater(second_generation, first_generation)

        self.assertFalse(coupled_command_ticket_is_current(shared, ticket=old_ticket))

    def test_stop_revokes_active_ticket_before_the_sdk_boundary(self) -> None:
        shared = _FakeShared()
        self.assertTrue(begin_motion(shared))
        active_ticket = CoupledCommandTicket(
            run_generation=int(shared.run_generation.value),
            ring_sequence=7,
            valid_until_monotonic_ns=time.monotonic_ns() + 1_000_000_000,
        )
        shared.coupled_cmd_ring.latest_sequence = active_ticket.ring_sequence
        self.assertTrue(
            coupled_command_ticket_allows_execution(
                shared,
                ticket=active_ticket,
            )
        )

        self.assertTrue(request_policy_stop(shared))

        self.assertFalse(
            coupled_command_ticket_allows_execution(
                shared,
                ticket=active_ticket,
            )
        )

    def test_cancel_only_invalidates_the_latest_current_ticket(self) -> None:
        shared = _FakeShared()
        self.assertTrue(begin_motion(shared))
        generation = int(shared.run_generation.value)
        older_ticket = CoupledCommandTicket(generation, 6, 10**18)
        newer_ticket = CoupledCommandTicket(generation, 7, 10**18)
        shared.coupled_cmd_ring.latest_sequence = newer_ticket.ring_sequence

        self.assertFalse(
            cancel_coupled_command_if_current(shared, ticket=older_ticket)
        )
        self.assertEqual(shared.run_generation.value, generation)
        self.assertTrue(
            coupled_command_ticket_is_current(shared, ticket=newer_ticket)
        )

        self.assertTrue(
            cancel_coupled_command_if_current(shared, ticket=newer_ticket)
        )
        self.assertEqual(shared.run_generation.value, generation + 1)
        self.assertFalse(
            coupled_command_ticket_is_current(shared, ticket=newer_ticket)
        )

    def test_physical_operator_allows_home_before_each_episode(self) -> None:
        shared = _FakeShared()
        keyboard = _FakeKeyboard(
            shared,
            [
                [ControlSignal.HOME],
                [ControlSignal.BEGIN],
                [ControlSignal.STOP],
                [ControlSignal.HOME],
                [ControlSignal.BEGIN],
            ],
        )
        home = Mock(side_effect=(True, True))

        with (
            patch(
                "dexmani_real.deployment.operator.KeyboardHandler",
                return_value=keyboard,
            ),
            patch("dexmani_real.deployment.operator._home", home),
        ):
            run_operator_control(
                shared,
                SimpleNamespace(),
                Mock(),
                stop_event=threading.Event(),
                execute=True,
            )

        self.assertEqual(home.call_count, 2)
        self.assertTrue(shared.physical_home_completed.value)
        self.assertTrue(shared.start_request.value)
        self.assertEqual(shared.stop_request.value, int(StopRequest.NONE))
        self.assertTrue(keyboard.started)
        self.assertTrue(keyboard.stopped)

    def test_physical_begin_requires_a_successful_home_sequence(self) -> None:
        shared = _FakeShared()
        keyboard = _FakeKeyboard(
            shared,
            [[ControlSignal.HOME], [ControlSignal.BEGIN]],
        )
        home = Mock(return_value=False)

        with (
            patch(
                "dexmani_real.deployment.operator.KeyboardHandler",
                return_value=keyboard,
            ),
            patch("dexmani_real.deployment.operator._home", home),
        ):
            run_operator_control(
                shared,
                SimpleNamespace(),
                Mock(),
                stop_event=threading.Event(),
                execute=True,
            )

        home.assert_called_once()
        self.assertFalse(shared.physical_home_completed.value)
        self.assertFalse(shared.start_request.value)

    def test_home_is_rejected_while_running(self) -> None:
        shared = _FakeShared()
        shared.safety_state.value = int(SafetyState.RUNNING)
        keyboard = _FakeKeyboard(shared, [[ControlSignal.HOME]])
        home = Mock(return_value=True)

        with (
            patch(
                "dexmani_real.deployment.operator.KeyboardHandler",
                return_value=keyboard,
            ),
            patch("dexmani_real.deployment.operator._home", home),
        ):
            run_operator_control(
                shared,
                SimpleNamespace(),
                Mock(),
                stop_event=threading.Event(),
                execute=True,
            )

        home.assert_not_called()
        self.assertFalse(shared.physical_home_completed.value)

    def test_home_begin_same_batch_cannot_start_an_episode(self) -> None:
        for signals in (
            [ControlSignal.HOME, ControlSignal.BEGIN],
            [ControlSignal.BEGIN, ControlSignal.HOME],
        ):
            with self.subTest(signals=signals):
                shared = _FakeShared()
                keyboard = _FakeKeyboard(shared, [signals])

                with (
                    patch(
                        "dexmani_real.deployment.operator.KeyboardHandler",
                        return_value=keyboard,
                    ),
                    patch(
                        "dexmani_real.deployment.operator._home",
                        return_value=True,
                    ),
                ):
                    run_operator_control(
                        shared,
                        SimpleNamespace(),
                        Mock(),
                        stop_event=threading.Event(),
                        execute=True,
                    )

                self.assertFalse(shared.start_request.value)

    def test_stop_during_completed_home_cannot_restore_home_authorization(self) -> None:
        shared = _FakeShared()
        keyboard = _FakeKeyboard(shared, [[ControlSignal.HOME]])

        def complete_after_stop(*_args: object, **_kwargs: object) -> bool:
            shared.stop_request.value = int(StopRequest.OPERATOR)
            return True

        with (
            patch(
                "dexmani_real.deployment.operator.KeyboardHandler",
                return_value=keyboard,
            ),
            patch(
                "dexmani_real.deployment.operator._home",
                side_effect=complete_after_stop,
            ),
        ):
            run_operator_control(
                shared,
                SimpleNamespace(),
                Mock(),
                stop_event=threading.Event(),
                execute=True,
            )

        self.assertFalse(shared.physical_home_completed.value)


if __name__ == "__main__":
    unittest.main()
