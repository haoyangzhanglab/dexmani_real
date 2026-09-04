"""Offline regressions for hand startup, shadow publication, and home commands."""

from __future__ import annotations

import sys
import threading
import time
import types
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.config.defaults import hand as hand_defaults
from dexmani_real.control import hand_home, publication
from dexmani_real.control.action import ActionCandidate
from dexmani_real.ipc.schema import (
    COUPLED_COMMAND_DTYPE,
    HAND_CONTACT_SHAPE,
    HAND_JOINT_SHAPE,
    HAND_TACTILE_FORCE_SHAPE,
    HAND_TACTILE_SUM_SHAPE,
)
from dexmani_real.robot.hand_worker import _limited_hand_setpoint, hand_loop
from dexmani_real.runtime.safety import CoupledCommandTicket, SafetyState


class _FakeRing:
    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())


class _StartupShared:
    def __init__(self) -> None:
        self.error_state = SimpleNamespace(value=False)
        self.is_running = SimpleNamespace(value=False)
        self.hand_state_ring = _FakeRing()
        self.hand_tactile_ring = _FakeRing()
        self.set_heartbeat = Mock()
        self.set_ready = Mock()


def _initial_hand_state() -> SimpleNamespace:
    return SimpleNamespace(
        qpos=np.zeros(HAND_JOINT_SHAPE, dtype=np.float64),
        current_ma=np.zeros(HAND_JOINT_SHAPE, dtype=np.float64),
        tactile_sum=np.zeros(HAND_TACTILE_SUM_SHAPE, dtype=np.float64),
        tactile_sum_valid=False,
        tactile_contact=np.zeros(HAND_CONTACT_SHAPE, dtype=bool),
        tactile_force=np.zeros(HAND_TACTILE_FORCE_SHAPE, dtype=np.float64),
        tactile_valid=False,
        commboard_err=np.zeros(HAND_JOINT_SHAPE, dtype=np.int32),
        jointboard_err=np.zeros(HAND_JOINT_SHAPE, dtype=np.int32),
        tipboard_err=np.zeros(HAND_JOINT_SHAPE, dtype=np.int32),
    )


class HandWorkerStartupTest(unittest.TestCase):
    def test_startup_publishes_live_state_without_command(self) -> None:
        shared = _StartupShared()
        state = _initial_hand_state()

        class FakeXHand:
            instance: "FakeXHand | None" = None

            def __init__(self, _config: object) -> None:
                type(self).instance = self
                self.connect = Mock()
                self.calibrate_tactile = Mock()
                self.get_state = Mock(return_value=state)
                self.reset_home = Mock()
                self.send_action = Mock()
                self.disconnect = Mock()
                self.is_connected = True
                self.tactile_calibrated = False

        fake_xhand_module = types.ModuleType("dexmani_real.robot.xhand")
        fake_xhand_module.XHand = FakeXHand
        fake_xhand_module.XHandSendStatus = object
        config = SimpleNamespace(
            loop_hz=30.0,
            qpos_min_rad=np.full(HAND_JOINT_SHAPE, -1.0),
            qpos_max_rad=np.full(HAND_JOINT_SHAPE, 1.0),
            mechanical_qpos_min_rad=np.full(HAND_JOINT_SHAPE, -1.0),
            mechanical_qpos_max_rad=np.full(HAND_JOINT_SHAPE, 1.0),
        )

        with patch.dict(sys.modules, {fake_xhand_module.__name__: fake_xhand_module}):
            hand_loop(shared, config)

        hand = FakeXHand.instance
        assert hand is not None
        hand.connect.assert_called_once_with()
        hand.calibrate_tactile.assert_called_once_with()
        hand.get_state.assert_called_once_with()
        hand.reset_home.assert_not_called()
        hand.send_action.assert_not_called()
        hand.disconnect.assert_called_once_with()
        shared.set_ready.assert_called_once_with("hand")
        self.assertEqual(len(shared.hand_state_ring.frames), 1)
        self.assertEqual(len(shared.hand_tactile_ring.frames), 1)
        np.testing.assert_array_equal(
            shared.hand_state_ring.frames[0]["qpos"][0], state.qpos
        )
        self.assertEqual(
            int(
                shared.hand_state_ring.frames[0][
                    "accepted_target_monotonic_ns"
                ][0]
            ),
            0,
        )


class HandWorkerRampingTest(unittest.TestCase):
    def test_running_advances_from_last_accepted_setpoint_under_contact(self) -> None:
        measured = np.zeros(HAND_JOINT_SHAPE, dtype=np.float64)
        target = measured.copy()
        target[0] = 0.301454

        first = _limited_hand_setpoint(
            target,
            measured_qpos=measured,
            last_sdk_accepted_qpos=measured,
            is_running=True,
            max_delta_rad_per_tick=0.3,
        )
        second = _limited_hand_setpoint(
            target,
            measured_qpos=measured,
            last_sdk_accepted_qpos=first,
            is_running=True,
            max_delta_rad_per_tick=0.3,
        )

        self.assertAlmostEqual(first[0], 0.3)
        np.testing.assert_array_equal(second, target)

    def test_armed_homing_remains_bounded_from_measurement(self) -> None:
        measured = np.zeros(HAND_JOINT_SHAPE, dtype=np.float64)
        target = measured.copy()
        target[0] = 0.301454
        prior_command = target.copy()

        bounded = _limited_hand_setpoint(
            target,
            measured_qpos=measured,
            last_sdk_accepted_qpos=prior_command,
            is_running=False,
            max_delta_rad_per_tick=0.3,
        )

        self.assertAlmostEqual(bounded[0], 0.3)
        self.assertFalse(np.array_equal(bounded, target))

    def test_running_reaches_exact_target_after_crc_with_static_measurement(self) -> None:
        now_ns = time.monotonic_ns()
        command = np.zeros(1, dtype=COUPLED_COMMAND_DTYPE)
        command["run_generation"][0] = 1
        command["action_id"][0] = 1
        command["created_monotonic_ns"][0] = now_ns
        command["scheduled_target_monotonic_ns"][0] = now_ns
        command["target_monotonic_ns"][0] = now_ns
        command["valid_until_monotonic_ns"][0] = now_ns + 1_000_000_000
        command["hand_present"][0] = 1
        target = np.asarray(hand_defaults.qpos_min_rad).copy()
        target[0] += 0.301454
        command["hand_qpos"][0] = target

        class CommandRing:
            def read_latest(self) -> tuple[np.ndarray, int, int]:
                return command, now_ns, 1

        class FakeRate:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                pass

            def wait(self) -> None:
                pass

        shared = SimpleNamespace(
            error_state=SimpleNamespace(value=False),
            is_running=SimpleNamespace(value=True),
            estop_request=SimpleNamespace(value=False),
            safety_state=SimpleNamespace(value=int(SafetyState.RUNNING)),
            run_generation=SimpleNamespace(value=1),
            active_coupled_command_sequence=SimpleNamespace(value=1),
            motion_lock=threading.Lock(),
            coupled_cmd_ring=CommandRing(),
            hand_state_ring=_FakeRing(),
            hand_tactile_ring=_FakeRing(),
            set_heartbeat=Mock(),
            set_ready=Mock(),
        )
        state = _initial_hand_state()
        state.qpos = np.asarray(hand_defaults.qpos_min_rad).copy()

        class SendStatus:
            ACCEPTED = object()
            CRC_UNCONFIRMED = object()
            REJECTED = object()

        class FakeXHand:
            instance: "FakeXHand | None" = None

            def __init__(self, _config: object) -> None:
                type(self).instance = self
                self.connect = Mock()
                self.calibrate_tactile = Mock()
                self.get_state = Mock(return_value=state)
                self.disconnect = Mock()
                self.is_connected = True
                self.tactile_calibrated = False
                self.sent: list[np.ndarray] = []
                self.send_statuses = (
                    SendStatus.CRC_UNCONFIRMED,
                    SendStatus.ACCEPTED,
                    SendStatus.ACCEPTED,
                )

            def send_action(self, action: np.ndarray) -> object:
                self.sent.append(action.copy())
                if len(self.sent) == 3:
                    shared.is_running.value = False
                return self.send_statuses[len(self.sent) - 1]

        fake_xhand_module = types.ModuleType("dexmani_real.robot.xhand")
        fake_xhand_module.XHand = FakeXHand
        fake_xhand_module.XHandSendStatus = SendStatus
        config = SimpleNamespace(
            loop_hz=30.0,
            qpos_min_rad=np.asarray(hand_defaults.qpos_min_rad),
            qpos_max_rad=np.asarray(hand_defaults.qpos_max_rad),
            mechanical_qpos_min_rad=np.asarray(
                hand_defaults.mechanical_qpos_min_rad
            ),
            mechanical_qpos_max_rad=np.asarray(
                hand_defaults.mechanical_qpos_max_rad
            ),
            hand_max_delta_rad_per_tick=0.3,
        )

        with (
            patch.dict(sys.modules, {fake_xhand_module.__name__: fake_xhand_module}),
            patch("dexmani_real.robot.hand_worker.LoopRate", FakeRate),
        ):
            hand_loop(shared, config)

        hand = FakeXHand.instance
        assert hand is not None
        self.assertEqual(len(hand.sent), 3)
        self.assertAlmostEqual(hand.sent[0][0], 0.3)
        np.testing.assert_array_equal(hand.sent[1], hand.sent[0])
        np.testing.assert_array_equal(hand.sent[2], command["hand_qpos"][0])


class CandidatePublicationTest(unittest.TestCase):
    def _candidate(self) -> ActionCandidate:
        now_ns = time.monotonic_ns()
        return ActionCandidate(
            observation_id=1,
            run_generation=7,
            action_id=1,
            created_monotonic_ns=now_ns,
            scheduled_target_monotonic_ns=now_ns,
            target_monotonic_ns=now_ns,
            valid_until_monotonic_ns=now_ns + 1_000_000_000,
            arm_qpos=np.zeros(7, dtype=np.float64),
        )

    def _validate_hand_shaped_candidate(
        self,
        candidate: ActionCandidate,
        gate: SimpleNamespace,
    ) -> publication.CommandPublishResult:
        hand_feedback_qpos = np.asarray(
            hand_defaults.qpos_min_rad, dtype=np.float64
        )
        arm_feedback = publication._ArmFeedbackSnapshot(
            qpos=np.zeros(7, dtype=np.float64), last_cmd_seq=0
        )
        hand_feedback = publication._HandFeedbackSnapshot(
            qpos=hand_feedback_qpos,
            accepted_target_action_id=0,
        )

        with (
            patch.object(publication, "check_runtime_gate", return_value=None),
            patch.object(
                publication,
                "_arm_feedback_snapshot",
                return_value=(arm_feedback, None),
            ),
            patch.object(
                publication,
                "read_hand_feedback",
                return_value=(hand_feedback, None),
            ),
            patch.object(
                publication,
                "read_motion_permit",
                return_value=SimpleNamespace(run_generation=7),
            ),
            patch.object(publication, "_validate_command_delivery", return_value=None),
        ):
            return publication.validate_and_send_candidate(
                object(),
                candidate,
                gate=gate,
                arm_feedback_max_age_s=0.1,
                hand_feedback_max_age_s=0.1,
                hand_command_max_delta_rad_per_tick=0.3,
                execute=False,
            )

    def test_execute_false_never_invokes_publication(self) -> None:
        candidate = self._candidate()
        gate = SimpleNamespace(
            validate=Mock(return_value=SimpleNamespace(accepted=True))
        )
        arm_feedback = publication._ArmFeedbackSnapshot(
            qpos=np.zeros(7, dtype=np.float64), last_cmd_seq=0
        )

        with (
            patch.object(publication, "check_runtime_gate", return_value=None),
            patch.object(
                publication,
                "_arm_feedback_snapshot",
                return_value=(arm_feedback, None),
            ),
            patch.object(
                publication,
                "read_motion_permit",
                return_value=SimpleNamespace(run_generation=7),
            ),
            patch.object(publication, "_validate_command_delivery", return_value=None),
            patch.object(publication, "send_command") as send_command,
            patch.object(
                publication, "publish_coupled_command_if_motion_permitted"
            ) as publish_coupled_command,
        ):
            result = publication.validate_and_send_candidate(
                object(),
                candidate,
                gate=gate,
                arm_feedback_max_age_s=0.1,
                hand_feedback_max_age_s=0.1,
                execute=False,
            )

        self.assertIs(result.status, publication.CommandPublishStatus.VALIDATED)
        send_command.assert_not_called()
        publish_coupled_command.assert_not_called()

    def test_delivery_keeps_one_full_policy_tick_for_workers(self) -> None:
        candidate = self._candidate()
        now_ns = candidate.target_monotonic_ns

        with (
            patch.object(publication, "check_runtime_gate", return_value=None),
            patch.object(
                publication,
                "read_motion_permit",
                return_value=SimpleNamespace(run_generation=7),
            ),
            patch.object(publication.time, "monotonic_ns", return_value=now_ns),
        ):
            closed = publication._validate_command_delivery(
                object(),
                replace(
                    candidate,
                    valid_until_monotonic_ns=now_ns + 62_500_000,
                ),
                check_is_running=True,
                minimum_delivery_window_s=0.0625,
            )
            open_window = publication._validate_command_delivery(
                object(),
                replace(
                    candidate,
                    valid_until_monotonic_ns=now_ns + 62_500_001,
                ),
                check_is_running=True,
                minimum_delivery_window_s=0.0625,
            )

        assert closed is not None
        self.assertIs(
            closed.status,
            publication.CommandPublishStatus.TEMPORAL_WINDOW_CLOSED,
        )
        self.assertIsNone(open_window)

    def test_execute_true_invokes_publication_seam(self) -> None:
        candidate = self._candidate()
        gate = SimpleNamespace(
            validate=Mock(return_value=SimpleNamespace(accepted=True))
        )
        arm_feedback = publication._ArmFeedbackSnapshot(
            qpos=np.zeros(7, dtype=np.float64), last_cmd_seq=0
        )
        ticket = CoupledCommandTicket(run_generation=7, ring_sequence=1)
        published = publication.CommandPublishResult(
            publication.CommandPublishStatus.PUBLISHED,
            candidate=candidate,
            ticket=ticket,
        )

        with (
            patch.object(publication, "check_runtime_gate", return_value=None),
            patch.object(
                publication,
                "_arm_feedback_snapshot",
                return_value=(arm_feedback, None),
            ),
            patch.object(
                publication,
                "read_motion_permit",
                return_value=SimpleNamespace(run_generation=7),
            ),
            patch.object(publication, "send_command", return_value=published) as send,
        ):
            result = publication.validate_and_send_candidate(
                object(),
                candidate,
                gate=gate,
                arm_feedback_max_age_s=0.1,
                hand_feedback_max_age_s=0.1,
                execute=True,
            )

        self.assertIs(result.status, publication.CommandPublishStatus.PUBLISHED)
        self.assertEqual(result.ticket, ticket)
        send.assert_called_once()

    def test_unchanged_hand_rate_shape_does_not_repeat_safety_gate(self) -> None:
        candidate = self._candidate()
        hand_qpos = np.asarray(hand_defaults.qpos_min_rad, dtype=np.float64)
        candidate = replace(candidate, hand_qpos=hand_qpos)
        gate = SimpleNamespace(
            hand_low=hand_qpos,
            hand_high=np.asarray(hand_defaults.qpos_max_rad, dtype=np.float64),
            validate=Mock(return_value=SimpleNamespace(accepted=True)),
        )
        arm_feedback = publication._ArmFeedbackSnapshot(
            qpos=np.zeros(7, dtype=np.float64), last_cmd_seq=0
        )
        hand_feedback = publication._HandFeedbackSnapshot(
            qpos=hand_qpos.copy(), accepted_target_action_id=0
        )

        with (
            patch.object(publication, "check_runtime_gate", return_value=None),
            patch.object(
                publication,
                "_arm_feedback_snapshot",
                return_value=(arm_feedback, None),
            ),
            patch.object(
                publication,
                "read_hand_feedback",
                return_value=(hand_feedback, None),
            ),
            patch.object(
                publication,
                "read_motion_permit",
                return_value=SimpleNamespace(run_generation=7),
            ),
            patch.object(publication, "_validate_command_delivery", return_value=None),
        ):
            result = publication.validate_and_send_candidate(
                object(),
                candidate,
                gate=gate,
                arm_feedback_max_age_s=0.1,
                hand_feedback_max_age_s=0.1,
                hand_command_max_delta_rad_per_tick=0.3,
                execute=False,
            )

        self.assertIs(result.status, publication.CommandPublishStatus.VALIDATED)
        gate.validate.assert_called_once()

    def test_changed_hand_rate_shape_is_revalidated(self) -> None:
        candidate = self._candidate()
        hand_qpos = np.asarray(hand_defaults.qpos_min_rad, dtype=np.float64)
        hand_qpos[0] = 0.6
        candidate = replace(candidate, hand_qpos=hand_qpos)
        gate = SimpleNamespace(
            hand_low=np.asarray(hand_defaults.qpos_min_rad, dtype=np.float64),
            hand_high=np.asarray(hand_defaults.qpos_max_rad, dtype=np.float64),
            max_hand_delta_rad=np.ones(HAND_JOINT_SHAPE, dtype=np.float64),
            collision_check=None,
            validate=Mock(return_value=SimpleNamespace(accepted=True)),
        )

        result = self._validate_hand_shaped_candidate(candidate, gate)

        self.assertIs(result.status, publication.CommandPublishStatus.VALIDATED)
        self.assertEqual(gate.validate.call_count, 2)
        np.testing.assert_array_equal(
            gate.validate.call_args_list[-1].args[0].hand_qpos,
            np.asarray(hand_defaults.qpos_min_rad, dtype=np.float64)
            + np.array([0.3] + [0.0] * 11),
        )

    def test_changed_hand_rate_shape_skips_policy_gate_revalidation(self) -> None:
        candidate = self._candidate()
        hand_qpos = np.asarray(hand_defaults.qpos_min_rad, dtype=np.float64)
        hand_qpos[0] = 0.6
        candidate = replace(candidate, hand_qpos=hand_qpos)
        gate = SimpleNamespace(
            hand_low=np.asarray(hand_defaults.qpos_min_rad, dtype=np.float64),
            hand_high=np.asarray(hand_defaults.qpos_max_rad, dtype=np.float64),
            max_hand_delta_rad=None,
            collision_check=None,
            validate=Mock(return_value=SimpleNamespace(accepted=True)),
        )

        result = self._validate_hand_shaped_candidate(candidate, gate)

        self.assertIs(result.status, publication.CommandPublishStatus.VALIDATED)
        gate.validate.assert_called_once()

    def test_ack_requires_both_arm_and_hand_workers(self) -> None:
        candidate = self._candidate()
        candidate = ActionCandidate(
            **{
                **candidate.__dict__,
                "hand_qpos": np.zeros(HAND_JOINT_SHAPE, dtype=np.float64),
            }
        )
        ticket = CoupledCommandTicket(run_generation=7, ring_sequence=1)
        arm_feedback = publication._ArmFeedbackSnapshot(
            qpos=np.zeros(7, dtype=np.float64),
            last_cmd_seq=candidate.action_id,
            last_cmd_accepted_monotonic_ns=110,
        )
        hand_pending = publication._HandFeedbackSnapshot(
            qpos=np.zeros(HAND_JOINT_SHAPE, dtype=np.float64),
            accepted_target_action_id=0,
        )
        hand_applied = publication._HandFeedbackSnapshot(
            qpos=np.zeros(HAND_JOINT_SHAPE, dtype=np.float64),
            accepted_target_action_id=candidate.action_id,
            accepted_target_monotonic_ns=120,
        )

        with (
            patch.object(publication, "check_runtime_gate", return_value=None),
            patch.object(
                publication,
                "_arm_feedback_snapshot",
                return_value=(arm_feedback, None),
            ),
            patch.object(
                publication,
                "coupled_command_ticket_is_current",
                return_value=True,
            ),
            patch.object(
                publication,
                "read_hand_feedback",
                side_effect=((hand_pending, None), (hand_applied, None)),
            ),
        ):
            pending = publication.poll_coupled_command_acknowledgement(
                object(),
                candidate,
                ticket=ticket,
                arm_feedback_max_age_s=0.1,
                hand_feedback_max_age_s=0.1,
            )
            applied = publication.poll_coupled_command_acknowledgement(
                object(),
                candidate,
                ticket=ticket,
                arm_feedback_max_age_s=0.1,
                hand_feedback_max_age_s=0.1,
            )

        self.assertIs(pending.status, publication.CommandPublishStatus.ACK_PENDING)
        self.assertEqual(pending.detail, "awaiting hand(last_action_id=0)")
        self.assertEqual(pending.arm_accepted_monotonic_ns, 110)
        self.assertIsNone(pending.hand_accepted_monotonic_ns)
        self.assertIs(applied.status, publication.CommandPublishStatus.APPLIED)
        self.assertEqual(applied.arm_accepted_monotonic_ns, 110)
        self.assertEqual(applied.hand_accepted_monotonic_ns, 120)


class HandHomePublicationTest(unittest.TestCase):
    def test_legal_home_allows_feedback_outside_command_bounds(self) -> None:
        target = np.deg2rad(
            np.asarray(hand_defaults.home_qpos_deg, dtype=np.float64)
        )
        measured = target.copy()
        measured[4] = -0.03
        pending = publication._HandFeedbackSnapshot(
            qpos=measured,
            accepted_target_action_id=0,
        )
        applied = publication._HandFeedbackSnapshot(
            qpos=measured,
            accepted_target_action_id=8,
        )
        candidate = SimpleNamespace(action_id=8)
        ticket = CoupledCommandTicket(run_generation=7, ring_sequence=1)
        published = publication.CommandPublishResult(
            publication.CommandPublishStatus.PUBLISHED,
            candidate=candidate,
            ticket=ticket,
        )

        with (
            patch.object(hand_home, "check_runtime_gate", return_value=None),
            patch.object(
                hand_home,
                "read_hand_feedback",
                side_effect=((pending, None), (applied, None)),
            ),
            patch.object(
                hand_home,
                "build_action_candidate",
                return_value=candidate,
            ),
            patch.object(hand_home, "send_command", return_value=published) as send,
        ):
            accepted = hand_home.publish_hand_home_and_wait_applied(
                object(),
                target,
                command_lower_rad=np.asarray(hand_defaults.qpos_min_rad),
                command_upper_rad=np.asarray(hand_defaults.qpos_max_rad),
                mechanical_lower_rad=np.asarray(
                    hand_defaults.mechanical_qpos_min_rad
                ),
                mechanical_upper_rad=np.asarray(
                    hand_defaults.mechanical_qpos_max_rad
                ),
                hand_feedback_max_age_s=0.1,
                verbose=False,
            )

        self.assertTrue(accepted)
        send.assert_called_once()


if __name__ == "__main__":
    unittest.main()
