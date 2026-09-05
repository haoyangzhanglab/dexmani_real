"""Behavior regressions for the thin learned-policy executor."""

from __future__ import annotations

import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.config.runtime import resolve_runtime_config
from dexmani_real.control.action import ActionCandidate
from dexmani_real.control.publication import (
    PUBLISH_REASON_GENERATION,
    PreparedCommand,
    PublishResult,
)
from dexmani_real.control.safety_gate import GateRejectCode, SafetyGate
from dexmani_real.deployment.config import PolicyDeploymentConfig
from dexmani_real.deployment.contracts import Prediction
from dexmani_real.deployment.executor import (
    PolicyExecutor,
    _advance_control_grid_ns,
    _build_policy_planner,
    _build_policy_safety_gate,
    _build_policy_workspace_check,
    _clip_policy_arm_action,
    _command_watchdog_reason,
    _CommandProgress,
    _end_policy_run,
    _physical_start_pose_rejection,
    _prediction_source_deadline_ns,
    decode_policy_action,
)
from dexmani_real.deployment.metrics import PolicyStats
from dexmani_real.runtime.safety import (
    CoupledCommandTicket,
    SafetyState,
    StopRequest,
    request_policy_stop,
)


class _Value:
    def __init__(self, value: int | bool) -> None:
        self.value = value
        self._lock = threading.Lock()

    def get_lock(self) -> threading.Lock:
        return self._lock


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
        self.arm_command_seq = _Value(0)
        self.motion_lock = threading.Lock()
        self.inference_request = threading.Event()
        self.heartbeats: list[tuple[str, float]] = []

    def set_heartbeat(self, name: str, timestamp_s: float) -> None:
        self.heartbeats.append((name, timestamp_s))


def _policy_spec(action_key: str = "action") -> SimpleNamespace:
    return SimpleNamespace(
        action_key=action_key,
        control_action_dim=21 if action_key == "action_ee" else 19,
    )


def _prediction(
    *,
    generation: int = 1,
    source_ns: int = 10,
    logical_ns: int = 100,
    steps: int = 3,
    action_dim: int = 19,
) -> Prediction:
    return Prediction(
        run_generation=generation,
        source_monotonic_ns=source_ns,
        logical_step_monotonic_ns=logical_ns,
        actions=np.zeros((steps, action_dim), dtype=np.float64),
    )


def _candidate(action_id: int = 1) -> ActionCandidate:
    return ActionCandidate(
        observation_id=action_id,
        run_generation=1,
        action_id=action_id,
        created_monotonic_ns=100,
        scheduled_target_monotonic_ns=100,
        target_monotonic_ns=100,
        valid_until_monotonic_ns=200,
        arm_qpos=np.zeros(7, dtype=np.float64),
        hand_qpos=np.zeros(12, dtype=np.float64),
    )


def _executor(
    *,
    mode: str = "sync",
    action_key: str = "action",
    max_action_steps: int | None = None,
    max_running_s: float | None = None,
    runtime_data: dict[str, object] | None = None,
    execute: bool = False,
) -> PolicyExecutor:
    with (
        patch(
            "dexmani_real.deployment.executor._build_policy_safety_gate",
            return_value=Mock(),
        ),
        patch(
            "dexmani_real.deployment.executor._build_policy_planner",
            return_value=Mock(),
        ),
    ):
        return PolicyExecutor(
            _FakeShared(),
            resolve_runtime_config(data=runtime_data),
            _policy_spec(action_key),
            PolicyDeploymentConfig(
                inference_mode=mode,
                max_action_steps=max_action_steps,
            ),
            execute=execute,
            max_running_s=max_running_s,
        )


def _policy_step_executor(
    *, hand_max_action_jump_rad: float = 0.3, prediction_steps: int = 8
) -> PolicyExecutor:
    executor = _executor(
        runtime_data={"policy": {"hand_max_action_jump_rad": hand_max_action_jump_rad}},
        execute=True,
    )
    runtime = executor.runtime
    executor.gate = SafetyGate(
        arm_joint_lower_rad=tuple(runtime.arm.joint_limit_lower),
        arm_joint_upper_rad=tuple(runtime.arm.joint_limit_upper),
        hand_joint_lower_rad=tuple(runtime.hand.qpos_min_rad),
        hand_joint_upper_rad=tuple(runtime.hand.qpos_max_rad),
        max_hand_delta_rad=runtime.policy.hand_max_action_jump_rad,
        endpoint_delta_tolerance_rad=runtime.policy.endpoint_delta_tolerance_rad,
    )
    now_ns = time.monotonic_ns()
    executor.shared.safety_state.value = int(SafetyState.RUNNING)
    executor.shared.run_generation.value = 1
    executor.run_generation = 1
    executor.run_started_ns = now_ns
    executor.progress.reset(1)
    executor.progress.observe(
        generation=1,
        arm_action_id=0,
        hand_action_id=0,
        now_ns=now_ns,
        timeout_ns=executor.command_progress_timeout_ns,
    )
    executor.max_source_age_ns = 10**12
    executor.active_prediction = _prediction(
        source_ns=now_ns,
        logical_ns=now_ns,
        steps=prediction_steps,
    )
    executor.schedule_base_ns = now_ns
    return executor


def _attempt_policy_step(
    executor: PolicyExecutor,
    *,
    arm_target: np.ndarray,
    hand_target: np.ndarray,
    measured_hand: np.ndarray,
) -> Mock:
    executor._decode_due_action = Mock(
        return_value=((arm_target.copy(), hand_target.copy()), None)
    )
    arm_feedback = SimpleNamespace(qpos=np.zeros(7, dtype=np.float64))
    hand_feedback = SimpleNamespace(qpos=measured_hand.copy())
    publication_ns = time.monotonic_ns()
    publish = Mock(
        return_value=PublishResult(
            True,
            ticket=CoupledCommandTicket(
                run_generation=1,
                ring_sequence=1,
                valid_until_monotonic_ns=publication_ns + 10**9,
                published_monotonic_ns=publication_ns,
            ),
        )
    )
    with (
        patch(
            "dexmani_real.control.publication._read_arm_feedback",
            return_value=(arm_feedback, "", None),
        ),
        patch(
            "dexmani_real.control.publication.read_hand_feedback",
            return_value=(hand_feedback, "", None),
        ),
        patch("dexmani_real.deployment.executor.publish_command", publish),
    ):
        executor._publish_due_action(
            np.zeros(19, dtype=np.float64),
            scheduled_target_ns=publication_ns,
            due_ns=publication_ns,
        )
    return publish


class PolicyExecutorBehaviorTest(unittest.TestCase):
    def test_joint_action_decodes_exact_arm_and_hand_targets(self) -> None:
        runtime = resolve_runtime_config()
        action = np.zeros(19, dtype=np.float64)
        action[:7] = 0.1
        action[7:] = np.arange(12, dtype=np.float64) / 20.0

        arm, hand, reason = decode_policy_action(
            action,
            _policy_spec(),
            np.zeros(7, dtype=np.float64),
            previous_arm_command_qpos=None,
            planner=None,
        )

        np.testing.assert_allclose(arm, action[:7])
        np.testing.assert_array_equal(hand, action[7:])
        self.assertEqual(reason, "")

    def test_ee_action_uses_ik_and_failure_remains_a_step_rejection(self) -> None:
        runtime = resolve_runtime_config()
        action = np.zeros(21, dtype=np.float64)
        action[3] = 1.0
        action[7] = 1.0
        expected = np.full(7, 0.2, dtype=np.float64)
        planner = Mock()
        planner.solve_teleop_ik.return_value = SimpleNamespace(
            success=True, qpos=expected, reason=""
        )

        arm, hand, reason = decode_policy_action(
            action,
            _policy_spec("action_ee"),
            np.zeros(7, dtype=np.float64),
            previous_arm_command_qpos=None,
            planner=planner,
        )
        np.testing.assert_array_equal(arm, expected)
        np.testing.assert_array_equal(hand, action[9:])
        self.assertEqual(reason, "")

        planner.solve_teleop_ik.return_value = SimpleNamespace(
            success=False, qpos=None, reason="no solution"
        )
        arm, _hand, reason = decode_policy_action(
            action,
            _policy_spec("action_ee"),
            np.zeros(7, dtype=np.float64),
            previous_arm_command_qpos=None,
            planner=planner,
        )
        self.assertIsNone(arm)
        self.assertEqual(reason, "no solution")

    def test_ee_action_rejects_degenerate_rot6d_during_decode(self) -> None:
        runtime = resolve_runtime_config()
        action = np.zeros(21, dtype=np.float64)
        planner = Mock()

        arm, hand, reason = decode_policy_action(
            action,
            _policy_spec("action_ee"),
            np.zeros(7, dtype=np.float64),
            previous_arm_command_qpos=None,
            planner=planner,
        )

        self.assertIsNone(arm)
        np.testing.assert_array_equal(hand, action[9:])
        self.assertIn("EE IK failed: ValueError", reason)
        planner.solve_teleop_ik.assert_not_called()

    def test_workspace_check_rejects_without_clipping(self) -> None:
        runtime = resolve_runtime_config()
        arm_fk = Mock()
        bounds = np.asarray(runtime.policy.workspace.as_tuple(), dtype=np.float64)
        inside = np.mean(bounds, axis=1)
        arm_fk.compute.return_value = (inside, np.zeros(4))
        with patch("dexmani_real.deployment.executor.make_arm_fk", return_value=arm_fk):
            check = _build_policy_workspace_check(runtime)
        self.assertTrue(check(np.zeros(7), np.full(7, 0.04)))

        arm_fk.compute.return_value = (np.array([10.0, 0.0, 0.0]), np.zeros(4))
        self.assertFalse(check(np.zeros(7), np.zeros(7)))

    def test_joint_gate_does_not_construct_the_ee_planner(self) -> None:
        runtime = resolve_runtime_config()
        arm_fk = Mock()
        arm_fk.compute.return_value = (np.zeros(3), np.zeros(4))
        with (
            patch("dexmani_real.deployment.executor.make_arm_fk", return_value=arm_fk),
            patch(
                "dexmani_real.deployment.executor.XArm7MotionPlanner"
            ) as planner_factory,
        ):
            gate = _build_policy_safety_gate(runtime)
        self.assertIsNone(gate.collision_check)
        planner_factory.assert_not_called()

        with patch(
            "dexmani_real.deployment.executor.XArm7MotionPlanner"
        ) as planner_factory:
            _build_policy_planner(runtime)
        self.assertFalse(
            planner_factory.call_args.kwargs["teleop_profile"].check_self_collision
        )

    def test_policy_arm_clip_is_independent_of_teleop_and_worker_guards(self) -> None:
        defaults = resolve_runtime_config()
        self.assertAlmostEqual(
            defaults.arm.max_servo_command_jump_rad, np.deg2rad(20.0)
        )
        self.assertAlmostEqual(
            defaults.policy.teleop_arm_max_delta_rad_per_tick, np.deg2rad(8.0)
        )
        self.assertAlmostEqual(
            defaults.policy.arm_action_delta_clip_rad, np.deg2rad(20.0)
        )
        self.assertEqual(defaults.policy.hand_max_action_jump_rad, 1.0)
        runtime = resolve_runtime_config(
            data={
                "arm": {"max_servo_command_jump_rad": 0.4},
                "policy": {
                    "teleop_arm_max_delta_rad_per_tick": 0.05,
                    "arm_action_delta_clip_rad": 0.2,
                    "hand_max_action_jump_rad": 0.45,
                },
            }
        )
        with patch(
            "dexmani_real.deployment.executor._build_policy_workspace_check",
            return_value=lambda _start, _end: True,
        ):
            gate = _build_policy_safety_gate(runtime)

        np.testing.assert_array_equal(gate.max_hand_delta_rad, np.full(12, 0.45))
        self.assertNotEqual(
            runtime.arm.max_servo_command_jump_rad,
            runtime.policy.arm_action_delta_clip_rad,
        )

    def test_policy_arm_spike_clip_is_wrap_aware_and_admits_before_clip(self) -> None:
        runtime = resolve_runtime_config()
        limit = runtime.policy.arm_action_delta_clip_rad
        reference = np.zeros(7, dtype=np.float64)
        target = reference.copy()
        target[0] = np.deg2rad(30.0)

        clipped_target, clipped, reason = _clip_policy_arm_action(
            target, reference, runtime
        )

        self.assertEqual(reason, "")
        self.assertTrue(clipped)
        assert clipped_target is not None
        self.assertAlmostEqual(clipped_target[0], limit)

        boundary_target = reference.copy()
        boundary_target[0] = limit
        exact_target, clipped, reason = _clip_policy_arm_action(
            boundary_target, reference, runtime
        )
        self.assertEqual(reason, "")
        self.assertFalse(clipped)
        np.testing.assert_array_equal(exact_target, boundary_target)

        wrap_reference = reference.copy()
        wrap_reference[0] = 6.1
        wrap_target = reference.copy()
        wrap_target[0] = -0.1
        canonical, clipped, reason = _clip_policy_arm_action(
            wrap_target, wrap_reference, runtime
        )
        self.assertEqual(reason, "")
        self.assertFalse(clipped)
        assert canonical is not None
        self.assertAlmostEqual(canonical[0], 2.0 * np.pi - 0.1)

        invalid = reference.copy()
        invalid[1] = runtime.arm.joint_limit_upper[1] + 0.1
        rejected, clipped, reason = _clip_policy_arm_action(invalid, reference, runtime)
        self.assertIsNone(rejected)
        self.assertFalse(clipped)
        self.assertEqual(reason, "arm joint limit violation")

    def test_ee_policy_clips_before_workspace_segment_admission(self) -> None:
        executor = _executor(action_key="action_ee", execute=True)
        runtime = executor.runtime
        measured_arm = np.zeros(7, dtype=np.float64)
        measured_hand = (
            np.asarray(runtime.hand.qpos_min_rad, dtype=np.float64)
            + np.asarray(runtime.hand.qpos_max_rad, dtype=np.float64)
        ) / 2.0
        arm_state = {
            "connected": True,
            "error_code": 0,
            "state_valid": True,
            "source_monotonic_ns": time.monotonic_ns(),
            "qpos": measured_arm,
            "qvel": np.zeros(7, dtype=np.float64),
        }
        workspace_segments: list[tuple[np.ndarray, np.ndarray]] = []

        def workspace_check(start: np.ndarray, end: np.ndarray) -> bool:
            workspace_segments.append((start.copy(), end.copy()))
            return True

        executor.gate = SafetyGate(
            arm_joint_lower_rad=tuple(runtime.arm.joint_limit_lower),
            arm_joint_upper_rad=tuple(runtime.arm.joint_limit_upper),
            hand_joint_lower_rad=tuple(runtime.hand.qpos_min_rad),
            hand_joint_upper_rad=tuple(runtime.hand.qpos_max_rad),
            workspace_check=workspace_check,
            max_hand_delta_rad=runtime.policy.hand_max_action_jump_rad,
        )
        ik_target = measured_arm.copy()
        ik_target[0] = np.deg2rad(30.0)
        executor.ee_planner.solve_teleop_ik.return_value = SimpleNamespace(
            success=True, qpos=ik_target, reason=""
        )
        executor.shared.safety_state.value = int(SafetyState.RUNNING)
        executor.shared.run_generation.value = 1
        executor.run_generation = 1
        executor.run_started_ns = time.monotonic_ns()
        executor.progress.reset(1)
        executor.progress.observe(
            generation=1,
            arm_action_id=0,
            hand_action_id=0,
            now_ns=executor.run_started_ns,
            timeout_ns=executor.command_progress_timeout_ns,
        )
        prediction_ns = time.monotonic_ns()
        executor.active_prediction = _prediction(
            source_ns=prediction_ns, logical_ns=prediction_ns, action_dim=21
        )
        executor.max_source_age_ns = 10**12
        action = np.zeros(21, dtype=np.float64)
        action[3] = 1.0
        action[7] = 1.0
        action[9:] = measured_hand
        publication_ns = time.monotonic_ns()
        publish = Mock(
            return_value=PublishResult(
                True,
                ticket=CoupledCommandTicket(
                    run_generation=1,
                    ring_sequence=1,
                    valid_until_monotonic_ns=publication_ns + 10**9,
                    published_monotonic_ns=publication_ns,
                ),
            )
        )
        with (
            patch(
                "dexmani_real.deployment.executor.read_arm_state_dict",
                return_value=arm_state,
            ),
            patch(
                "dexmani_real.control.publication._read_arm_feedback",
                return_value=(SimpleNamespace(qpos=measured_arm), "", None),
            ),
            patch(
                "dexmani_real.control.publication.read_hand_feedback",
                return_value=(SimpleNamespace(qpos=measured_hand), "", None),
            ),
            patch("dexmani_real.deployment.executor.publish_command", publish),
        ):
            executor._publish_due_action(
                action,
                scheduled_target_ns=publication_ns,
                due_ns=publication_ns,
            )

        publish.assert_called_once()
        self.assertEqual(executor.stats.arm_action_clip_count, 1)
        self.assertEqual(len(workspace_segments), 1)
        np.testing.assert_array_equal(workspace_segments[0][0], measured_arm)
        self.assertAlmostEqual(
            workspace_segments[0][1][0], runtime.policy.arm_action_delta_clip_rad
        )
        published = publish.call_args.args[1]
        np.testing.assert_array_equal(published.arm_qpos, workspace_segments[0][1])

    def test_shadow_and_execute_advance_the_same_accepted_references(self) -> None:
        execute = _policy_step_executor()
        shadow = _policy_step_executor()
        shadow.execute = False
        runtime = execute.runtime
        hand = (
            np.asarray(runtime.hand.qpos_min_rad, dtype=np.float64)
            + np.asarray(runtime.hand.qpos_max_rad, dtype=np.float64)
        ) / 2.0
        arm_targets = (
            np.full(7, 0.1, dtype=np.float64),
            np.full(7, 0.2, dtype=np.float64),
        )
        hand_targets = (hand, hand + 0.1)

        with patch(
            "dexmani_real.deployment.executor.command_publishability_reason",
            return_value="",
        ):
            for arm_target, hand_target in zip(arm_targets, hand_targets):
                _attempt_policy_step(
                    execute,
                    arm_target=arm_target,
                    hand_target=hand_target,
                    measured_hand=hand,
                )
                _attempt_policy_step(
                    shadow,
                    arm_target=arm_target,
                    hand_target=hand_target,
                    measured_hand=hand,
                )

        np.testing.assert_array_equal(
            shadow.previous_arm_command_qpos, execute.previous_arm_command_qpos
        )
        np.testing.assert_array_equal(
            shadow.previous_hand_command_qpos, execute.previous_hand_command_qpos
        )

    def test_shadow_publication_rejection_does_not_advance_references(self) -> None:
        executor = _policy_step_executor()
        executor.execute = False
        previous_arm = np.full(7, 0.1, dtype=np.float64)
        previous_hand = (
            np.asarray(executor.runtime.hand.qpos_min_rad, dtype=np.float64)
            + np.asarray(executor.runtime.hand.qpos_max_rad, dtype=np.float64)
        ) / 2.0
        executor.previous_arm_command_qpos = previous_arm.copy()
        executor.previous_hand_command_qpos = previous_hand.copy()

        with patch(
            "dexmani_real.deployment.executor.command_publishability_reason",
            return_value=PUBLISH_REASON_GENERATION,
        ):
            _attempt_policy_step(
                executor,
                arm_target=np.full(7, 0.2, dtype=np.float64),
                hand_target=previous_hand + 0.1,
                measured_hand=previous_hand,
            )

        np.testing.assert_array_equal(executor.previous_arm_command_qpos, previous_arm)
        np.testing.assert_array_equal(
            executor.previous_hand_command_qpos, previous_hand
        )

    def test_policy_hand_target_is_published_exactly_without_teleop_shaping(
        self,
    ) -> None:
        executor = _policy_step_executor(hand_max_action_jump_rad=0.5)
        lower = np.asarray(executor.runtime.hand.qpos_min_rad, dtype=np.float64)
        upper = np.asarray(executor.runtime.hand.qpos_max_rad, dtype=np.float64)
        measured = (lower + upper) / 2.0
        target = measured.copy()
        target[0] += 0.4

        publish = _attempt_policy_step(
            executor,
            arm_target=np.zeros(7, dtype=np.float64),
            hand_target=target,
            measured_hand=measured,
        )

        publish.assert_called_once()
        published = publish.call_args.args[1]
        np.testing.assert_array_equal(published.hand_qpos, target)
        self.assertEqual(executor.runtime.hand.hand_max_delta_rad_per_tick, 0.3)

    def test_first_policy_hand_action_uses_measured_feedback(self) -> None:
        executor = _policy_step_executor()
        lower = np.asarray(executor.runtime.hand.qpos_min_rad, dtype=np.float64)
        upper = np.asarray(executor.runtime.hand.qpos_max_rad, dtype=np.float64)
        measured = (lower + upper) / 2.0
        target = measured.copy()
        target[0] += 0.31

        publish = _attempt_policy_step(
            executor,
            arm_target=np.zeros(7, dtype=np.float64),
            hand_target=target,
            measured_hand=measured,
        )

        publish.assert_not_called()
        self.assertIsNone(executor.previous_hand_command_qpos)

    def test_later_policy_hand_action_uses_previous_published_target(self) -> None:
        executor = _policy_step_executor()
        lower = np.asarray(executor.runtime.hand.qpos_min_rad, dtype=np.float64)
        upper = np.asarray(executor.runtime.hand.qpos_max_rad, dtype=np.float64)
        previous = (lower + upper) / 2.0
        first_publish = _attempt_policy_step(
            executor,
            arm_target=np.zeros(7, dtype=np.float64),
            hand_target=previous,
            measured_hand=previous,
        )
        first_publish.assert_called_once()

        measured_lag = previous.copy()
        measured_lag[0] -= 0.2
        target = previous.copy()
        target[0] += 0.2
        self.assertGreater(abs(target[0] - measured_lag[0]), 0.3)
        second_publish = _attempt_policy_step(
            executor,
            arm_target=np.zeros(7, dtype=np.float64),
            hand_target=target,
            measured_hand=measured_lag,
        )

        second_publish.assert_called_once()
        published = second_publish.call_args.args[1]
        np.testing.assert_array_equal(published.hand_qpos, target)
        np.testing.assert_array_equal(executor.previous_hand_command_qpos, target)

    def test_rejected_policy_step_does_not_advance_hand_reference(self) -> None:
        executor = _policy_step_executor()
        lower = np.asarray(executor.runtime.hand.qpos_min_rad, dtype=np.float64)
        upper = np.asarray(executor.runtime.hand.qpos_max_rad, dtype=np.float64)
        accepted = (lower + upper) / 2.0
        _attempt_policy_step(
            executor,
            arm_target=np.zeros(7, dtype=np.float64),
            hand_target=accepted,
            measured_hand=accepted,
        )

        rejected = accepted.copy()
        rejected[0] += 0.31
        rejected_publish = _attempt_policy_step(
            executor,
            arm_target=np.zeros(7, dtype=np.float64),
            hand_target=rejected,
            measured_hand=accepted,
        )
        rejected_publish.assert_not_called()
        np.testing.assert_array_equal(executor.previous_hand_command_qpos, accepted)

        next_target = accepted.copy()
        next_target[0] -= 0.1
        self.assertGreater(abs(next_target[0] - rejected[0]), 0.3)
        next_publish = _attempt_policy_step(
            executor,
            arm_target=np.zeros(7, dtype=np.float64),
            hand_target=next_target,
            measured_hand=rejected,
        )
        next_publish.assert_called_once()

    def test_policy_hand_reference_continues_across_prediction_chunks(self) -> None:
        executor = _policy_step_executor(prediction_steps=1)
        lower = np.asarray(executor.runtime.hand.qpos_min_rad, dtype=np.float64)
        upper = np.asarray(executor.runtime.hand.qpos_max_rad, dtype=np.float64)
        chunk_a_last = (lower + upper) / 2.0
        _attempt_policy_step(
            executor,
            arm_target=np.zeros(7, dtype=np.float64),
            hand_target=chunk_a_last,
            measured_hand=chunk_a_last,
        )
        self.assertIsNone(executor.active_prediction)

        now_ns = time.monotonic_ns()
        executor.active_prediction = _prediction(
            source_ns=now_ns, logical_ns=now_ns, steps=2
        )
        executor.schedule_base_ns = now_ns
        measured_lag = chunk_a_last.copy()
        measured_lag[0] -= 0.2
        chunk_b_first = chunk_a_last.copy()
        chunk_b_first[0] += 0.2
        publish = _attempt_policy_step(
            executor,
            arm_target=np.zeros(7, dtype=np.float64),
            hand_target=chunk_b_first,
            measured_hand=measured_lag,
        )

        publish.assert_called_once()
        np.testing.assert_array_equal(
            executor.previous_hand_command_qpos, chunk_b_first
        )

    def test_hand_jump_rejects_the_whole_coupled_policy_step(self) -> None:
        executor = _policy_step_executor()
        lower = np.asarray(executor.runtime.hand.qpos_min_rad, dtype=np.float64)
        upper = np.asarray(executor.runtime.hand.qpos_max_rad, dtype=np.float64)
        measured = (lower + upper) / 2.0
        unsafe_hand = measured.copy()
        unsafe_hand[0] += 0.31

        publish = _attempt_policy_step(
            executor,
            arm_target=np.full(7, 0.1, dtype=np.float64),
            hand_target=unsafe_hand,
            measured_hand=measured,
        )

        publish.assert_not_called()
        self.assertIsNone(executor.previous_arm_command_qpos)
        self.assertIsNone(executor.previous_hand_command_qpos)

    def test_sync_prediction_is_sequential_and_requests_next_at_completion(
        self,
    ) -> None:
        executor = _executor()
        executor.run_generation = 1
        executor.active_prediction = _prediction(steps=2)
        executor.schedule_base_ns = 100

        self.assertIsNotNone(executor._next_due_action(100))
        executor._record_terminal_step(successful=True, candidate=_candidate(1))
        self.assertEqual(executor.step_index, 1)
        self.assertFalse(executor.shared.inference_request.is_set())

        self.assertIsNotNone(executor._next_due_action(100 + executor.step_dt_ns))
        executor._record_terminal_step(successful=True, candidate=_candidate(2))
        self.assertIsNone(executor.active_prediction)
        self.assertTrue(executor.shared.inference_request.is_set())

    def test_sync_timeline_starts_when_prediction_becomes_active(self) -> None:
        executor = _executor()
        executor.run_generation = 1
        prediction = _prediction(logical_ns=10_000)
        with patch(
            "dexmani_real.deployment.executor.read_latest_prediction",
            return_value=(prediction, 1),
        ):
            self.assertTrue(executor._ingest_latest_prediction(200))
        self.assertEqual(executor.schedule_base_ns, 200)

    def test_async_ingest_skips_past_steps_and_newer_replaces_suffix(self) -> None:
        executor = _executor(mode="async")
        executor.run_generation = 1
        executor.step_dt_ns = 10
        first = _prediction(logical_ns=100)
        with patch(
            "dexmani_real.deployment.executor.read_latest_prediction",
            return_value=(first, 1),
        ):
            self.assertTrue(executor._ingest_latest_prediction(111))
        self.assertEqual(executor.step_index, 2)

        newer = _prediction(logical_ns=105)
        with patch(
            "dexmani_real.deployment.executor.read_latest_prediction",
            return_value=(newer, 2),
        ):
            self.assertTrue(executor._ingest_latest_prediction(111))
        self.assertIs(executor.active_prediction, newer)
        self.assertEqual(executor.step_index, 1)

    def test_async_expired_prediction_does_not_replace_future_suffix(self) -> None:
        executor = _executor(mode="async")
        executor.run_generation = 1
        executor.step_dt_ns = 10
        current = _prediction(logical_ns=200)
        executor.active_prediction = current
        executor.schedule_base_ns = 200
        expired = _prediction(logical_ns=100)
        with patch(
            "dexmani_real.deployment.executor.read_latest_prediction",
            return_value=(expired, 2),
        ):
            executor._ingest_latest_prediction(131)
        self.assertIs(executor.active_prediction, current)

    def test_async_late_slot_and_newer_suffix_publish_at_most_once_per_tick(
        self,
    ) -> None:
        """A late terminal slot cannot cause catch-up after a new prediction."""
        executor = _executor(mode="async")
        shared = executor.shared
        shared.safety_state.value = int(SafetyState.RUNNING)
        shared.run_generation.value = 1
        executor.run_generation = 1
        executor.run_started_ns = 1
        executor.step_dt_ns = 10
        executor.max_source_age_ns = 1_000_000
        executor._observe_worker_progress = Mock(return_value=True)
        first = _prediction(source_ns=1, logical_ns=100, steps=5)
        newer = _prediction(source_ns=1, logical_ns=120, steps=5)
        published = Mock()

        def finish_late_slot(
            _action: np.ndarray,
            *,
            scheduled_target_ns: int,
            due_ns: int,
        ) -> None:
            self.assertEqual(scheduled_target_ns, 100)
            published()
            executor._consume_control_slot(due_ns, terminal_ns=125)
            executor._advance_prediction()

        executor._publish_due_action = Mock(side_effect=finish_late_slot)
        with (
            patch(
                "dexmani_real.deployment.executor.read_latest_prediction",
                side_effect=((first, 1), (newer, 2)),
            ),
            patch(
                "dexmani_real.deployment.executor.time.monotonic_ns", return_value=125
            ),
        ):
            executor._run_active_tick(100)
            executor._run_active_tick(125)

        self.assertEqual(published.call_count, 1)
        self.assertIs(executor.active_prediction, newer)
        self.assertEqual(executor.step_index, 1)
        self.assertEqual(executor.next_command_due_ns, 135)

    def test_control_grid_never_catches_up_after_late_terminal(self) -> None:
        self.assertEqual(_advance_control_grid_ns(100, 109, 10), 110)
        self.assertEqual(_advance_control_grid_ns(100, 110, 10), 120)
        self.assertEqual(_advance_control_grid_ns(100, 125, 10), 135)

        executor = _executor()
        executor.step_dt_ns = 10
        executor.active_prediction = _prediction(steps=2)
        executor.schedule_base_ns = 100
        executor.next_command_due_ns = 110
        self.assertIsNone(executor._next_due_action(109))

    def test_terminal_rejections_count_toward_action_limit(self) -> None:
        executor = _executor(max_action_steps=2)
        executor.active_prediction = _prediction()
        executor.schedule_base_ns = 100
        executor._finish_episode = Mock()

        executor._record_terminal_step(successful=False, candidate=None)
        self.assertEqual(executor.episode_steps, 1)
        executor._record_terminal_step(successful=False, candidate=None)
        self.assertEqual(executor.episode_steps, 2)
        executor._finish_episode.assert_called_once_with(
            "action_step_limit", aborted=False
        )

    def test_physical_action_limit_waits_for_both_worker_watermarks(self) -> None:
        executor = _executor(max_action_steps=1)
        executor.execute = True
        executor.active_prediction = _prediction()
        executor.schedule_base_ns = 100
        candidate = _candidate()

        executor._record_terminal_step(successful=True, candidate=candidate)

        self.assertEqual(executor.pending_truncation_action_id, candidate.action_id)
        self.assertEqual(executor.step_index, 0)
        self.assertIsNotNone(executor.active_prediction)

    def test_running_limit_is_executor_owned_and_ends_without_fault(self) -> None:
        executor = _executor(max_running_s=0.001)
        executor.run_generation = 1
        executor.run_started_ns = 100
        executor.shared.safety_state.value = int(SafetyState.RUNNING)
        executor.shared.run_generation.value = 1
        executor._observe_worker_progress = Mock(return_value=True)
        executor._finish_episode = Mock()

        executor._run_active_tick(1_000_101)

        executor._finish_episode.assert_called_once_with(
            "run time limit", aborted=False
        )
        self.assertFalse(executor.shared.error_state.value)

    def test_operator_stop_fences_and_clears_active_prediction(self) -> None:
        executor = _executor()
        shared = executor.shared
        shared.safety_state.value = int(SafetyState.RUNNING)
        shared.run_generation.value = 1
        executor.run_generation = 1
        executor.run_started_ns = 100
        executor.active_prediction = _prediction()

        self.assertTrue(request_policy_stop(shared))
        executor._handle_run_boundary()

        self.assertEqual(shared.safety_state.value, int(SafetyState.ARMED))
        self.assertIsNone(executor.active_prediction)
        self.assertIsNone(executor.run_started_ns)

    def test_stop_generation_race_ends_normally_without_fault(self) -> None:
        executor = _executor()
        shared = executor.shared
        executor.run_generation = 1
        executor.run_started_ns = 100
        shared.safety_state.value = int(SafetyState.ARMED)
        shared.run_generation.value = 2
        shared.stop_request.value = int(StopRequest.OPERATOR)
        executor._fault = Mock()

        executor._run_active_tick(101)

        executor._fault.assert_not_called()
        executor._handle_run_boundary()
        self.assertIsNone(executor.run_started_ns)
        self.assertFalse(shared.error_state.value)

    def test_unexplained_generation_change_remains_fail_closed(self) -> None:
        executor = _executor()
        shared = executor.shared
        executor.run_generation = 1
        executor.run_started_ns = 100
        shared.safety_state.value = int(SafetyState.ARMED)
        shared.run_generation.value = 2
        executor._fault = Mock()

        executor._run_active_tick(101)

        executor._fault.assert_called_once_with(
            "RUNNING generation changed outside an episode boundary"
        )

    def test_quit_keeps_executor_alive_until_supervisor_stops_runtime(self) -> None:
        executor = _executor()
        shared = executor.shared
        shared.quit_requested.value = True
        shared.safety_state.value = int(SafetyState.RUNNING)
        shared.run_generation.value = 1
        executor.run_generation = 1
        executor.run_started_ns = 100
        rate = Mock()
        rate.wait.side_effect = lambda: setattr(shared.is_running, "value", False)

        with patch("dexmani_real.deployment.executor.LoopRate", return_value=rate):
            executor.run()

        self.assertEqual(shared.safety_state.value, int(SafetyState.ARMED))
        self.assertIsNone(executor.run_started_ns)
        self.assertEqual(len(shared.heartbeats), 1)
        self.assertEqual(shared.heartbeats[0][0], "policy")
        rate.wait.assert_called_once_with()

    def test_new_begin_resets_execution_state_and_generation(self) -> None:
        executor = _executor()
        shared = executor.shared
        executor.active_prediction = _prediction(generation=1)
        executor.step_index = 2
        executor.previous_arm_command_qpos = np.ones(7, dtype=np.float64)
        executor.previous_hand_command_qpos = np.ones(12, dtype=np.float64)
        shared.start_request.value = True

        executor._start_requested_episode()

        self.assertEqual(shared.safety_state.value, int(SafetyState.RUNNING))
        self.assertEqual(executor.run_generation, shared.run_generation.value)
        self.assertIsNone(executor.active_prediction)
        self.assertEqual(executor.step_index, 0)
        self.assertIsNone(executor.previous_arm_command_qpos)
        self.assertIsNone(executor.previous_hand_command_qpos)
        self.assertTrue(shared.inference_request.is_set())

    def test_safety_rejection_is_terminal_but_checker_failure_faults(self) -> None:
        executor = _executor()
        executor._reject_due_step = Mock()
        candidate = _candidate()

        executor._handle_preparation_rejection(
            PreparedCommand(reason="workspace", gate_code=GateRejectCode.WORKSPACE),
            candidate,
            100,
        )
        executor._reject_due_step.assert_called_once_with(100, "workspace")
        self.assertFalse(executor.shared.error_state.value)
        self.assertEqual(executor.stats.safety_rejection_count, 1)

        executor._fault = Mock()
        executor._handle_preparation_rejection(
            PreparedCommand(
                reason="workspace checker failed",
                gate_code=GateRejectCode.WORKSPACE_CHECK_FAILED,
                fatal=True,
            ),
            candidate,
            100,
        )
        executor._fault.assert_called_once()

    def test_physical_begin_rechecks_measured_home_pose(self) -> None:
        runtime = resolve_runtime_config()
        shared = _FakeShared()
        shared.physical_home_completed.value = True
        state = {
            "connected": True,
            "error_code": 0,
            "state_valid": True,
            "source_monotonic_ns": time.monotonic_ns(),
            "qpos": np.asarray(runtime.arm.home_qpos, dtype=np.float64),
            "qvel": np.zeros(7, dtype=np.float64),
        }
        with patch(
            "dexmani_real.deployment.executor.read_arm_state_dict", return_value=state
        ):
            self.assertIsNone(
                _physical_start_pose_rejection(shared, runtime, execute=True)
            )
            state["qpos"] = state["qpos"].copy()
            state["qpos"][0] += 1.0
            self.assertIn(
                "not at the training start pose",
                _physical_start_pose_rejection(shared, runtime, execute=True),
            )

    def test_source_deadline_and_command_silence_are_bounded(self) -> None:
        self.assertEqual(
            _prediction_source_deadline_ns(
                _prediction(source_ns=10), max_source_age_ns=20
            ),
            30,
        )
        self.assertIsNone(
            _command_watchdog_reason(
                now_ns=200,
                run_started_ns=100,
                last_valid_command_ns=None,
                first_command_timeout_ns=100,
                command_silence_timeout_ns=50,
            )
        )
        self.assertEqual(
            _command_watchdog_reason(
                now_ns=201,
                run_started_ns=100,
                last_valid_command_ns=None,
                first_command_timeout_ns=100,
                command_silence_timeout_ns=50,
            ),
            "first command timeout",
        )

    def test_stale_prediction_never_publishes_a_physical_command(self) -> None:
        executor = _executor()
        executor.execute = True
        executor.run_generation = 1
        executor.run_started_ns = 100
        executor.shared.safety_state.value = int(SafetyState.RUNNING)
        executor.shared.run_generation.value = 1
        executor.active_prediction = _prediction(source_ns=100, logical_ns=100)
        executor.schedule_base_ns = 100
        executor.max_source_age_ns = 10
        executor.progress.reset(1)
        executor.progress.observe(
            generation=1,
            arm_action_id=0,
            hand_action_id=0,
            now_ns=100,
            timeout_ns=50,
        )

        with (
            patch(
                "dexmani_real.deployment.executor.read_latest_prediction",
                return_value=None,
            ),
            patch("dexmani_real.deployment.executor.publish_command") as publish,
        ):
            executor._run_active_tick(111)

        publish.assert_not_called()
        self.assertIsNone(executor.active_prediction)

    def test_latest_wins_progress_allows_skipped_ids_and_stall_faults(self) -> None:
        progress = _CommandProgress()
        progress.reset(1)
        self.assertIsNone(
            progress.observe(
                generation=1,
                arm_action_id=4,
                hand_action_id=4,
                now_ns=100,
                timeout_ns=50,
            )
        )
        progress.record_publication(5, 110)
        self.assertIsNone(
            progress.observe(
                generation=1,
                arm_action_id=8,
                hand_action_id=9,
                now_ns=120,
                timeout_ns=50,
            )
        )
        self.assertTrue(progress.covers(5))

        progress.record_publication(10, 130)
        self.assertEqual(
            progress.observe(
                generation=1,
                arm_action_id=8,
                hand_action_id=9,
                now_ns=181,
                timeout_ns=50,
            ),
            "arm worker command progress timeout",
        )

    def test_continuous_publication_cannot_mask_stalled_worker_progress(self) -> None:
        progress = _CommandProgress()
        progress.reset(1)
        self.assertIsNone(
            progress.observe(
                generation=1,
                arm_action_id=0,
                hand_action_id=0,
                now_ns=100,
                timeout_ns=50,
            )
        )
        progress.record_publication(1, 110)
        progress.record_publication(2, 130)

        self.assertEqual(
            progress.observe(
                generation=1,
                arm_action_id=2,
                hand_action_id=0,
                now_ns=161,
                timeout_ns=50,
            ),
            "hand worker command progress timeout",
        )

    def test_intermediate_hand_acceptance_keeps_outstanding_target_healthy(
        self,
    ) -> None:
        progress = _CommandProgress()
        progress.reset(1)
        self.assertIsNone(
            progress.observe(
                generation=1,
                arm_action_id=0,
                hand_action_id=0,
                hand_setpoint_accepted_ns=90,
                now_ns=100,
                timeout_ns=50,
            )
        )
        progress.record_publication(1, 110)

        self.assertIsNone(
            progress.observe(
                generation=1,
                arm_action_id=1,
                hand_action_id=0,
                hand_setpoint_accepted_ns=150,
                now_ns=170,
                timeout_ns=50,
            )
        )
        self.assertFalse(progress.covers(1))
        self.assertEqual(
            progress.observe(
                generation=1,
                arm_action_id=1,
                hand_action_id=0,
                hand_setpoint_accepted_ns=150,
                now_ns=201,
                timeout_ns=50,
            ),
            "hand worker command progress timeout",
        )

    def test_old_generation_hand_timestamp_does_not_refresh_new_wait(self) -> None:
        progress = _CommandProgress()
        progress.reset(2)
        self.assertIsNone(
            progress.observe(
                generation=2,
                arm_action_id=1,
                hand_action_id=1,
                hand_setpoint_accepted_ns=1_000,
                now_ns=1_010,
                timeout_ns=50,
            )
        )
        progress.record_publication(2, 1_020)

        self.assertEqual(
            progress.observe(
                generation=2,
                arm_action_id=2,
                hand_action_id=1,
                hand_setpoint_accepted_ns=1_000,
                now_ns=1_071,
                timeout_ns=50,
            ),
            "hand worker command progress timeout",
        )

    def test_future_hand_progress_timestamp_is_rejected(self) -> None:
        progress = _CommandProgress()
        progress.reset(1)

        self.assertEqual(
            progress.observe(
                generation=1,
                arm_action_id=0,
                hand_action_id=0,
                hand_setpoint_accepted_ns=101,
                now_ns=100,
                timeout_ns=50,
            ),
            "hand SDK setpoint progress is in the future",
        )

    def test_worker_acceptance_stall_escalates_to_runtime_fault(self) -> None:
        executor = _executor()
        executor.execute = True
        executor.run_generation = 1
        executor.command_progress_timeout_ns = 50
        executor.progress.reset(1)
        executor.progress.observe(
            generation=1,
            arm_action_id=0,
            hand_action_id=0,
            now_ns=100,
            timeout_ns=50,
        )
        executor.progress.record_publication(1, 110)
        executor._fault = Mock()
        with patch(
            "dexmani_real.deployment.executor._read_command_progress",
            return_value=(0, 0, 0, None),
        ):
            self.assertFalse(executor._observe_worker_progress(161))
        executor._fault.assert_called_once_with(
            "arm worker command progress timeout",
        )
        self.assertEqual(executor.stats.command_progress_timeout_count, 1)

    def test_episode_failure_returns_to_armed_without_global_fault(self) -> None:
        shared = _FakeShared()
        shared.safety_state.value = int(SafetyState.RUNNING)
        stats = PolicyStats()

        _end_policy_run(
            shared,
            "first command timeout",
            stats=stats,
            aborted=True,
        )

        self.assertEqual(shared.safety_state.value, int(SafetyState.ARMED))
        self.assertFalse(shared.error_state.value)
        self.assertFalse(shared.physical_home_completed.value)


if __name__ == "__main__":
    unittest.main()
