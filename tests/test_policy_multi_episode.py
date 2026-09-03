"""Offline contracts for repeated policy episodes in one process."""

from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.deployment.action_buffer import ActionBuffer, BufferedPlan, PushStatus
from dexmani_real.deployment.contracts import JointActionChunk
from dexmani_real.deployment.operator import run_operator_control
from dexmani_real.runtime.safety import (
    CoupledCommandTicket,
    SafetyState,
    StopRequest,
    begin_motion,
    coupled_command_ticket_is_current,
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
        self.active_coupled_command_sequence = _Value(0)
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
            batch = self._batches.pop(0)
            if not self._batches:
                self._shared.is_running.value = False
            return batch
        self._shared.is_running.value = False
        return []

    def drain_signal(self, _signal: ControlSignal) -> int:
        return 0


def _old_plan(run_generation: int) -> BufferedPlan:
    return BufferedPlan(
        plan_id=1,
        run_generation=run_generation,
        observation_id=1,
        observation_anchor_ns=20,
        observation_latest_source_ns=10,
        inference_finished_ns=30,
        deadline_ns=100,
        chunk=JointActionChunk(
            arm_qpos=np.zeros((1, 7), dtype=np.float64),
            hand_qpos=np.zeros((1, 12), dtype=np.float64),
            target_monotonic_ns=np.array([50], dtype=np.uint64),
            valid_mask=np.array([1], dtype=np.uint8),
        ),
    )


class PolicyMultiEpisodeTest(unittest.TestCase):
    def test_second_episode_advances_generation_and_invalidates_old_work(self) -> None:
        shared = _FakeShared()
        self.assertTrue(begin_motion(shared))
        first_generation = int(shared.run_generation.value)

        old_ticket = CoupledCommandTicket(
            run_generation=first_generation,
            ring_sequence=7,
        )
        shared.active_coupled_command_sequence.value = old_ticket.ring_sequence
        self.assertTrue(coupled_command_ticket_is_current(shared, ticket=old_ticket))

        action_buffer = ActionBuffer(max_buffered_plans=2)
        old_plan = _old_plan(first_generation)
        action_buffer.reset(run_generation=first_generation)
        self.assertTrue(action_buffer.push(old_plan, now_ns=40).accepted)

        self.assertTrue(revoke_motion(shared, SafetyState.ARMED))
        self.assertFalse(coupled_command_ticket_is_current(shared, ticket=old_ticket))
        self.assertTrue(begin_motion(shared))
        second_generation = int(shared.run_generation.value)
        self.assertGreater(second_generation, first_generation)

        action_buffer.reset(run_generation=second_generation)
        self.assertIs(
            action_buffer.push(old_plan, now_ns=40).status,
            PushStatus.WRONG_GENERATION,
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
