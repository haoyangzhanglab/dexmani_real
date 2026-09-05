"""Offline regressions for hand startup, shadow publication, and home commands."""

from __future__ import annotations

import sys
import threading
import time
import types
import unittest
from collections.abc import Callable
from dataclasses import replace
from itertools import count
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.config.defaults import hand as hand_defaults
from dexmani_real.control import hand_home, publication
from dexmani_real.control.action import ActionCandidate
from dexmani_real.control.safety_gate import SafetyGate
from dexmani_real.ipc.schema import (
    COUPLED_COMMAND_DTYPE,
    HAND_CONTACT_SHAPE,
    HAND_JOINT_SHAPE,
    HAND_TACTILE_FORCE_SHAPE,
    HAND_TACTILE_SUM_SHAPE,
)
from dexmani_real.robot.hand_worker import _limited_hand_setpoint, hand_loop
from dexmani_real.runtime.safety import CoupledCommandTicket, SafetyState
from dexmani_real.utils.limits import (
    POLICY_HAND_ENDPOINT_ROUNDOFF_TOLERANCE_RAD,
    limit_hand_target_delta,
)


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
            hand_loop(shared, config, 1.0)

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
            int(shared.hand_state_ring.frames[0]["accepted_target_monotonic_ns"][0]),
            0,
        )

    def test_sustained_state_read_failure_faults_worker(self) -> None:
        shared = _StartupShared()
        shared.is_running.value = True
        shared.estop_request = SimpleNamespace(value=False)
        state = _initial_hand_state()

        class FakeXHand:
            instance: "FakeXHand | None" = None

            def __init__(self, _config: object) -> None:
                type(self).instance = self
                self.connect = Mock()
                self.calibrate_tactile = Mock()
                self.get_state = Mock(side_effect=[state, None, None, None])
                self.disconnect = Mock()
                self.is_connected = True
                self.tactile_calibrated = False

        class FakeRate:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                pass

            def wait(self) -> None:
                pass

        fake_xhand_module = types.ModuleType("dexmani_real.robot.xhand")
        fake_xhand_module.XHand = FakeXHand
        fake_xhand_module.XHandSendStatus = object
        config = SimpleNamespace(
            loop_hz=30.0,
            mechanical_qpos_min_rad=np.full(HAND_JOINT_SHAPE, -1.0),
            mechanical_qpos_max_rad=np.full(HAND_JOINT_SHAPE, 1.0),
        )
        monotonic_values = count(1.0, 0.1)

        with (
            patch.dict(sys.modules, {fake_xhand_module.__name__: fake_xhand_module}),
            patch("dexmani_real.robot.hand_worker.LoopRate", FakeRate),
            patch(
                "dexmani_real.robot.hand_worker.time.monotonic",
                side_effect=lambda: next(monotonic_values),
            ),
            self.assertRaisesRegex(RuntimeError, "hand state reads failed"),
        ):
            hand_loop(shared, config, 0.15)

        self.assertTrue(shared.error_state.value)
        stale_frames = [
            frame for frame in shared.hand_state_ring.frames if frame["qpos_stale"][0]
        ]
        self.assertEqual(len(stale_frames), 2)
        hand = FakeXHand.instance
        assert hand is not None
        hand.disconnect.assert_called_once_with()


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

    def test_running_reaches_exact_target_after_crc_with_static_measurement(
        self,
    ) -> None:
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

            @property
            def latest_sequence(self) -> int:
                return 1

        class FakeRate:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                self.waits_after_exact_target = 0

            def wait(self) -> None:
                hand = FakeXHand.instance
                if hand is not None and len(hand.sent) == 3:
                    self.waits_after_exact_target += 1
                    if self.waits_after_exact_target == 2:
                        shared.is_running.value = False

        shared = SimpleNamespace(
            error_state=SimpleNamespace(value=False),
            is_running=SimpleNamespace(value=True),
            estop_request=SimpleNamespace(value=False),
            safety_state=SimpleNamespace(value=int(SafetyState.RUNNING)),
            run_generation=SimpleNamespace(value=1),
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
                return self.send_statuses[len(self.sent) - 1]

        fake_xhand_module = types.ModuleType("dexmani_real.robot.xhand")
        fake_xhand_module.XHand = FakeXHand
        fake_xhand_module.XHandSendStatus = SendStatus
        config = SimpleNamespace(
            loop_hz=30.0,
            qpos_min_rad=np.asarray(hand_defaults.qpos_min_rad),
            qpos_max_rad=np.asarray(hand_defaults.qpos_max_rad),
            mechanical_qpos_min_rad=np.asarray(hand_defaults.mechanical_qpos_min_rad),
            mechanical_qpos_max_rad=np.asarray(hand_defaults.mechanical_qpos_max_rad),
            hand_max_delta_rad_per_tick=0.3,
        )

        with (
            patch.dict(sys.modules, {fake_xhand_module.__name__: fake_xhand_module}),
            patch("dexmani_real.robot.hand_worker.LoopRate", FakeRate),
        ):
            hand_loop(shared, config, 1.0)

        hand = FakeXHand.instance
        assert hand is not None
        self.assertEqual(len(hand.sent), 3)
        self.assertAlmostEqual(hand.sent[0][0], 0.3)
        np.testing.assert_array_equal(hand.sent[1], hand.sent[0])
        np.testing.assert_array_equal(hand.sent[2], command["hand_qpos"][0])

    def test_invalid_commands_never_invoke_hand_sdk(self) -> None:
        now_ns = time.monotonic_ns()
        command = np.zeros(1, dtype=COUPLED_COMMAND_DTYPE)
        command["run_generation"][0] = 1
        command["action_id"][0] = 1
        command["created_monotonic_ns"][0] = now_ns - 2_000_000
        command["scheduled_target_monotonic_ns"][0] = now_ns - 2_000_000
        command["target_monotonic_ns"][0] = now_ns - 1_000_000
        command["valid_until_monotonic_ns"][0] = now_ns - 1
        command["hand_present"][0] = 1
        command["hand_qpos"][0] = np.asarray(hand_defaults.qpos_min_rad)

        class CommandRing:
            def read_latest(self) -> tuple[np.ndarray, int, int]:
                return command, now_ns, 1

            @property
            def latest_sequence(self) -> int:
                return 1

        class FakeRate:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                pass

            def wait(self) -> None:
                shared.is_running.value = False

        shared = SimpleNamespace(
            error_state=SimpleNamespace(value=False),
            is_running=SimpleNamespace(value=True),
            estop_request=SimpleNamespace(value=False),
            safety_state=SimpleNamespace(value=int(SafetyState.RUNNING)),
            run_generation=SimpleNamespace(value=1),
            motion_lock=threading.Lock(),
            coupled_cmd_ring=CommandRing(),
            hand_state_ring=_FakeRing(),
            hand_tactile_ring=_FakeRing(),
            set_heartbeat=Mock(),
            set_ready=Mock(),
        )
        state = _initial_hand_state()

        class FakeXHand:
            instance: "FakeXHand | None" = None

            def __init__(self, _config: object) -> None:
                type(self).instance = self
                self.connect = Mock()
                self.calibrate_tactile = Mock()
                self.get_state = Mock(return_value=state)
                self.send_action = Mock()
                self.disconnect = Mock()
                self.is_connected = True
                self.tactile_calibrated = False

        fake_xhand_module = types.ModuleType("dexmani_real.robot.xhand")
        fake_xhand_module.XHand = FakeXHand
        fake_xhand_module.XHandSendStatus = object
        config = SimpleNamespace(
            loop_hz=30.0,
            qpos_min_rad=np.asarray(hand_defaults.qpos_min_rad),
            qpos_max_rad=np.asarray(hand_defaults.qpos_max_rad),
            mechanical_qpos_min_rad=np.asarray(hand_defaults.mechanical_qpos_min_rad),
            mechanical_qpos_max_rad=np.asarray(hand_defaults.mechanical_qpos_max_rad),
            hand_max_delta_rad_per_tick=0.3,
        )

        with (
            patch.dict(sys.modules, {fake_xhand_module.__name__: fake_xhand_module}),
            patch("dexmani_real.robot.hand_worker.LoopRate", FakeRate),
        ):
            hand_loop(shared, config, 1.0)

        hand = FakeXHand.instance
        assert hand is not None
        hand.send_action.assert_not_called()

        lower = np.asarray(hand_defaults.mechanical_qpos_min_rad, dtype=np.float64)
        invalid_targets = (
            np.array([np.nan] + [0.0] * 11),
            np.array([np.inf] + [0.0] * 11),
            lower + np.array([0.0, 0.0, -0.1] + [0.0] * 9),
        )
        for target in invalid_targets:
            with self.subTest(target=target):
                valid_now_ns = time.monotonic_ns()
                for field in (
                    "created_monotonic_ns",
                    "scheduled_target_monotonic_ns",
                    "target_monotonic_ns",
                ):
                    command[field][0] = valid_now_ns
                command["valid_until_monotonic_ns"][0] = valid_now_ns + 1_000_000_000
                command["hand_qpos"][0] = target
                shared.is_running.value = True
                shared.error_state.value = False

                with (
                    patch.dict(
                        sys.modules, {fake_xhand_module.__name__: fake_xhand_module}
                    ),
                    patch("dexmani_real.robot.hand_worker.LoopRate", FakeRate),
                ):
                    hand_loop(shared, config, 1.0)

                hand = FakeXHand.instance
                assert hand is not None
                hand.send_action.assert_not_called()


class CandidatePublicationTest(unittest.TestCase):
    def _candidate(
        self,
        *,
        arm_qpos: np.ndarray | None = None,
        hand_qpos: np.ndarray | None = None,
    ) -> ActionCandidate:
        now_ns = time.monotonic_ns()
        return ActionCandidate(
            observation_id=1,
            run_generation=7,
            action_id=1,
            created_monotonic_ns=now_ns,
            scheduled_target_monotonic_ns=now_ns,
            target_monotonic_ns=now_ns,
            valid_until_monotonic_ns=now_ns + 1_000_000_000,
            arm_qpos=(np.zeros(7, dtype=np.float64) if arm_qpos is None else arm_qpos),
            hand_qpos=hand_qpos,
        )

    def test_publication_reserves_the_requested_worker_window(self) -> None:
        candidate = self._candidate()
        now_ns = candidate.target_monotonic_ns
        shared = SimpleNamespace()

        with (
            patch.object(publication, "motion_rejection_reason", return_value=""),
            patch.object(
                publication,
                "read_motion_permit",
                return_value=SimpleNamespace(run_generation=7),
            ),
            patch.object(publication.time, "monotonic_ns", return_value=now_ns),
        ):
            closed = publication.command_publishability_reason(
                shared,
                replace(candidate, valid_until_monotonic_ns=now_ns + 62_500_000),
                minimum_delivery_window_s=0.0625,
            )
            open_window = publication.command_publishability_reason(
                shared,
                replace(candidate, valid_until_monotonic_ns=now_ns + 62_500_001),
                minimum_delivery_window_s=0.0625,
            )

        self.assertEqual(closed, publication.PUBLISH_REASON_EXPIRED)
        self.assertEqual(open_window, "")

    def test_stale_generation_never_reaches_command_ring(self) -> None:
        candidate = self._candidate()
        with patch.object(
            publication,
            "publish_coupled_command_if_motion_permitted",
            return_value=(None, publication.PUBLISH_REASON_GENERATION),
        ) as publish_coupled:
            result = publication.publish_command(object(), candidate)

        self.assertFalse(result.published)
        self.assertEqual(result.reason, publication.PUBLISH_REASON_GENERATION)
        publish_coupled.assert_called_once()

    def test_valid_command_publishes_without_polling_acceptance(self) -> None:
        candidate = self._candidate()
        ticket = CoupledCommandTicket(
            run_generation=7,
            ring_sequence=1,
            valid_until_monotonic_ns=10**18,
        )
        with (
            patch.object(publication, "_command_publishability_reason") as query,
            patch.object(
                publication,
                "publish_coupled_command_if_motion_permitted",
                return_value=(ticket, ""),
            ) as publish_coupled,
            patch.object(publication, "wait_command_accepted") as wait,
        ):
            result = publication.publish_command(object(), candidate)

        self.assertTrue(result.published)
        self.assertEqual(result.ticket, ticket)
        publish_coupled.assert_called_once()
        query.assert_not_called()
        wait.assert_not_called()

    def test_locked_publication_reports_each_concurrent_rejection(self) -> None:
        class CaptureRing:
            def __init__(self) -> None:
                self.writes = 0

            def write(self, _frame: np.ndarray) -> int:
                self.writes += 1
                return self.writes

        class MutateOnAtomicEntry:
            def __init__(
                self,
                shared: SimpleNamespace,
                clock: dict[str, int],
                *,
                locked_now_ns: int,
                mutate: Callable[[SimpleNamespace], None],
            ) -> None:
                self.entries = 0
                self._shared = shared
                self._clock = clock
                self._locked_now_ns = locked_now_ns
                self._mutate = mutate

            def __enter__(self) -> "MutateOnAtomicEntry":
                self.entries += 1
                if self.entries == 1:
                    self._clock["now_ns"] = self._locked_now_ns
                    self._mutate(self._shared)
                return self

            def __exit__(self, *_args: object) -> None:
                return None

        candidate = replace(
            self._candidate(),
            created_monotonic_ns=100,
            scheduled_target_monotonic_ns=100,
            target_monotonic_ns=100,
            valid_until_monotonic_ns=1_000,
        )
        cases = (
            (
                "runtime stopped",
                900,
                0.0,
                lambda shared: setattr(shared.is_running, "value", False),
                publication.PUBLISH_REASON_RUNTIME_STOPPED,
            ),
            (
                "fault",
                900,
                0.0,
                lambda shared: setattr(shared.error_state, "value", True),
                publication.PUBLISH_REASON_FAULT,
            ),
            (
                "estop",
                900,
                0.0,
                lambda shared: setattr(shared.estop_request, "value", True),
                publication.PUBLISH_REASON_ESTOP,
            ),
            (
                "generation",
                900,
                0.0,
                lambda shared: setattr(shared.run_generation, "value", 8),
                publication.PUBLISH_REASON_GENERATION,
            ),
            (
                "safety state",
                900,
                0.0,
                lambda shared: setattr(
                    shared.safety_state, "value", int(SafetyState.ARMED)
                ),
                f"{publication.PUBLISH_REASON_SAFETY_STATE}: expected RUNNING, got ARMED",
            ),
            (
                "deadline",
                1_000,
                0.0,
                lambda _shared: None,
                publication.PUBLISH_REASON_EXPIRED,
            ),
            (
                "delivery window",
                950,
                0.00000005,
                lambda _shared: None,
                publication.PUBLISH_REASON_EXPIRED,
            ),
        )
        for (
            label,
            locked_now_ns,
            minimum_delivery_window_s,
            mutate,
            expected_reason,
        ) in cases:
            with self.subTest(rejection=label):
                clock = {"now_ns": 900}
                ring = CaptureRing()
                shared = SimpleNamespace(
                    is_running=SimpleNamespace(value=True),
                    error_state=SimpleNamespace(value=False),
                    estop_request=SimpleNamespace(value=False),
                    safety_state=SimpleNamespace(value=int(SafetyState.RUNNING)),
                    run_generation=SimpleNamespace(value=7),
                    coupled_cmd_ring=ring,
                )
                lock = MutateOnAtomicEntry(
                    shared,
                    clock,
                    locked_now_ns=locked_now_ns,
                    mutate=mutate,
                )
                shared.motion_lock = lock

                with (
                    patch.object(
                        publication.time,
                        "monotonic_ns",
                        side_effect=lambda: clock["now_ns"],
                    ),
                    patch(
                        "dexmani_real.runtime.safety.time.monotonic_ns",
                        side_effect=lambda: clock["now_ns"],
                    ),
                ):
                    result = publication.publish_command(
                        shared,
                        candidate,
                        required_safety_state=SafetyState.RUNNING,
                        minimum_delivery_window_s=minimum_delivery_window_s,
                    )

                self.assertFalse(result.published)
                self.assertEqual(result.reason, expected_reason)
                self.assertEqual(ring.writes, 0)
                self.assertEqual(lock.entries, 1)

    def test_valid_policy_hand_target_is_shaped_before_one_final_gate(self) -> None:
        hand_qpos = np.asarray(hand_defaults.qpos_min_rad, dtype=np.float64)
        hand_qpos[0] += 0.6
        candidate = self._candidate(hand_qpos=hand_qpos)
        measured_hand = np.asarray(hand_defaults.qpos_min_rad, dtype=np.float64)
        gate = SimpleNamespace(
            hand_low=np.asarray(hand_defaults.qpos_min_rad, dtype=np.float64),
            hand_high=np.asarray(hand_defaults.qpos_max_rad, dtype=np.float64),
            validate=Mock(return_value=SimpleNamespace(accepted=True)),
        )
        arm_feedback = publication._ArmFeedbackSnapshot(
            qpos=np.zeros(7, dtype=np.float64),
            accepted_action_id=0,
        )
        hand_feedback = publication._HandFeedbackSnapshot(
            qpos=measured_hand,
            accepted_action_id=0,
        )

        with (
            patch.object(
                publication,
                "_read_arm_feedback",
                return_value=(arm_feedback, "", None),
            ),
            patch.object(
                publication,
                "read_hand_feedback",
                return_value=(hand_feedback, "", None),
            ),
        ):
            result = publication.prepare_command(
                object(),
                candidate,
                gate=gate,
                arm_feedback_max_age_s=0.1,
                hand_feedback_max_age_s=0.1,
                hand_mechanical_lower_rad=hand_defaults.mechanical_qpos_min_rad,
                hand_mechanical_upper_rad=hand_defaults.mechanical_qpos_max_rad,
                hand_command_max_delta_rad_per_tick=0.3,
                canonicalize_policy_hand_roundoff=True,
            )

        self.assertTrue(result.accepted)
        gate.validate.assert_called_once()
        assert result.candidate is not None
        expected = measured_hand + np.array([0.3] + [0.0] * 11)
        np.testing.assert_array_equal(result.candidate.hand_qpos, expected)
        validated_candidate = gate.validate.call_args.args[0]
        np.testing.assert_array_equal(validated_candidate.hand_qpos, expected)

    def _prepare_policy_hand(
        self, raw_hand: np.ndarray, measured_hand: np.ndarray
    ) -> publication.PreparedCommand:
        lower = np.asarray(hand_defaults.qpos_min_rad, dtype=np.float64)
        upper = np.asarray(hand_defaults.qpos_max_rad, dtype=np.float64)
        gate = SafetyGate(
            arm_joint_lower_rad=(-1.0,) * 7,
            arm_joint_upper_rad=(1.0,) * 7,
            hand_joint_lower_rad=tuple(lower),
            hand_joint_upper_rad=tuple(upper),
        )
        arm_feedback = publication._ArmFeedbackSnapshot(
            qpos=np.zeros(7), accepted_action_id=0
        )
        hand_feedback = publication._HandFeedbackSnapshot(
            qpos=measured_hand, accepted_action_id=0
        )
        with (
            patch.object(
                publication,
                "_read_arm_feedback",
                return_value=(arm_feedback, "", None),
            ),
            patch.object(
                publication,
                "read_hand_feedback",
                return_value=(hand_feedback, "", None),
            ),
        ):
            return publication.prepare_command(
                object(),
                self._candidate(hand_qpos=raw_hand),
                gate=gate,
                arm_feedback_max_age_s=0.1,
                hand_feedback_max_age_s=0.1,
                hand_mechanical_lower_rad=hand_defaults.mechanical_qpos_min_rad,
                hand_mechanical_upper_rad=hand_defaults.mechanical_qpos_max_rad,
                hand_command_max_delta_rad_per_tick=0.3,
                canonicalize_policy_hand_roundoff=True,
            )

    def test_policy_raw_hand_violation_is_rejected_before_safe_shaping(self) -> None:
        lower = np.asarray(hand_defaults.qpos_min_rad, dtype=np.float64)
        upper = np.asarray(hand_defaults.qpos_max_rad, dtype=np.float64)
        measured = lower.copy()
        measured[2] += 0.5
        raw = lower.copy()
        raw[2] -= 0.01
        shaped = limit_hand_target_delta(raw, measured, 0.3)
        self.assertTrue(np.all((lower <= shaped) & (shaped <= upper)))

        prepared = self._prepare_policy_hand(raw, measured)
        self.assertFalse(prepared.accepted)
        self.assertIn("operational joint limits", prepared.reason)

    def test_policy_raw_hand_target_slightly_beyond_tolerance_is_rejected(
        self,
    ) -> None:
        lower = np.asarray(hand_defaults.qpos_min_rad, dtype=np.float64)
        raw = lower.copy()
        raw[2] -= 2.0 * POLICY_HAND_ENDPOINT_ROUNDOFF_TOLERANCE_RAD

        prepared = self._prepare_policy_hand(raw, lower)

        self.assertFalse(prepared.accepted)
        self.assertIn("operational joint limits", prepared.reason)

    def test_policy_float32_endpoint_roundoff_is_canonicalized_exactly(self) -> None:
        lower = np.asarray(hand_defaults.qpos_min_rad, dtype=np.float64)
        raw = lower.copy()
        raw[2] = np.float64(np.float32(lower[2]))
        self.assertLess(raw[2], lower[2])

        prepared = self._prepare_policy_hand(raw, lower)
        self.assertTrue(prepared.accepted)
        assert prepared.candidate is not None
        self.assertEqual(prepared.candidate.hand_qpos[2], lower[2])

    def test_policy_shaped_hand_mechanical_violation_is_rejected(self) -> None:
        lower = np.asarray(hand_defaults.qpos_min_rad, dtype=np.float64)
        raw = lower.copy()
        raw[0] += 0.6
        measured = lower.copy()
        measured[0] = -1.0

        prepared = self._prepare_policy_hand(raw, measured)

        self.assertFalse(prepared.accepted)
        self.assertIs(prepared.gate_code, publication.GateRejectCode.HAND_JOINT_LIMIT)

    def test_blocking_acceptance_requires_both_workers(self) -> None:
        ticket = CoupledCommandTicket(
            run_generation=7,
            ring_sequence=1,
            valid_until_monotonic_ns=10**18,
        )
        arm_feedback = publication._ArmFeedbackSnapshot(
            qpos=np.zeros(7, dtype=np.float64),
            accepted_action_id=1,
            accepted_monotonic_ns=110,
        )
        hand_pending = publication._HandFeedbackSnapshot(
            qpos=np.zeros(HAND_JOINT_SHAPE, dtype=np.float64),
            accepted_action_id=0,
        )
        hand_accepted = publication._HandFeedbackSnapshot(
            qpos=np.zeros(HAND_JOINT_SHAPE, dtype=np.float64),
            accepted_action_id=1,
            accepted_monotonic_ns=120,
        )

        with (
            patch.object(publication, "motion_rejection_reason", return_value=""),
            patch.object(
                publication,
                "_read_arm_feedback",
                return_value=(arm_feedback, "", None),
            ),
            patch.object(
                publication,
                "read_hand_feedback",
                side_effect=(
                    (hand_pending, "", None),
                    (hand_accepted, "", None),
                ),
            ),
            patch.object(
                publication,
                "coupled_command_ticket_is_current",
                return_value=True,
            ),
            patch.object(publication.time, "sleep"),
        ):
            accepted = publication.wait_command_accepted(
                object(),
                ticket=ticket,
                action_id=1,
                wait_for_arm=True,
                wait_for_hand=True,
                timeout_s=0.1,
                arm_feedback_max_age_s=0.1,
                hand_feedback_max_age_s=0.1,
            )

        self.assertTrue(accepted.accepted)


class HandHomePublicationTest(unittest.TestCase):
    def test_legal_home_allows_feedback_outside_command_bounds(self) -> None:
        target = np.deg2rad(np.asarray(hand_defaults.home_qpos_deg, dtype=np.float64))
        measured = target.copy()
        measured[4] = -0.03
        feedback = publication._HandFeedbackSnapshot(
            qpos=measured,
            accepted_action_id=0,
        )
        candidate = SimpleNamespace(action_id=8)
        ticket = CoupledCommandTicket(
            run_generation=7,
            ring_sequence=1,
            valid_until_monotonic_ns=10**18,
        )

        with (
            patch.object(hand_home, "motion_rejection_reason", return_value=""),
            patch.object(
                hand_home,
                "read_hand_feedback",
                return_value=(feedback, "", None),
            ),
            patch.object(
                hand_home,
                "build_action_candidate",
                return_value=candidate,
            ),
            patch.object(
                hand_home,
                "publish_command",
                return_value=publication.PublishResult(True, ticket=ticket),
            ) as publish,
            patch.object(
                hand_home,
                "wait_command_accepted",
                return_value=publication.AcceptanceResult(True),
            ) as wait,
        ):
            accepted = hand_home.publish_hand_home_and_wait_accepted(
                object(),
                target,
                command_lower_rad=np.asarray(hand_defaults.qpos_min_rad),
                command_upper_rad=np.asarray(hand_defaults.qpos_max_rad),
                mechanical_lower_rad=np.asarray(hand_defaults.mechanical_qpos_min_rad),
                mechanical_upper_rad=np.asarray(hand_defaults.mechanical_qpos_max_rad),
                hand_feedback_max_age_s=0.1,
                verbose=False,
            )

        self.assertTrue(accepted)
        publish.assert_called_once()
        wait.assert_called_once()


if __name__ == "__main__":
    unittest.main()
