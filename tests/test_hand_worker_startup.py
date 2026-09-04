"""Offline regressions for hand startup, shadow publication, and home commands."""

from __future__ import annotations

import sys
import time
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.config.defaults import hand as hand_defaults
from dexmani_real.control import hand_home, publication
from dexmani_real.control.action import ActionCandidate
from dexmani_real.ipc.schema import (
    HAND_CONTACT_SHAPE,
    HAND_JOINT_SHAPE,
    HAND_TACTILE_FORCE_SHAPE,
    HAND_TACTILE_SUM_SHAPE,
)
from dexmani_real.robot.hand_worker import hand_loop
from dexmani_real.runtime.safety import CoupledCommandTicket


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
