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
from dexmani_real.control.publication import PreparedCommand
from dexmani_real.control.safety_gate import GateRejectCode
from dexmani_real.deployment.config import PolicyDeploymentConfig
from dexmani_real.deployment.contracts import Prediction
from dexmani_real.deployment.executor import (
    PolicyExecutor,
    _advance_control_grid_ns,
    _build_policy_planner,
    _build_policy_safety_gate,
    _build_policy_workspace_check,
    _command_watchdog_reason,
    _CommandProgress,
    _end_policy_run,
    _physical_start_pose_rejection,
    _prediction_source_deadline_ns,
    decode_policy_action,
)
from dexmani_real.deployment.metrics import PolicyStats
from dexmani_real.runtime.safety import SafetyState, StopRequest, request_policy_stop


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
            resolve_runtime_config(),
            _policy_spec(action_key),
            PolicyDeploymentConfig(
                inference_mode=mode,
                max_action_steps=max_action_steps,
            ),
            execute=False,
            max_running_s=max_running_s,
        )


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
            runtime=runtime,
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
            runtime=runtime,
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
            runtime=runtime,
        )
        self.assertIsNone(arm)
        self.assertEqual(reason, "no solution")

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
            executor._advance_prediction(None)

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
        shared.start_request.value = True

        executor._start_requested_episode()

        self.assertEqual(shared.safety_state.value, int(SafetyState.RUNNING))
        self.assertEqual(executor.run_generation, shared.run_generation.value)
        self.assertIsNone(executor.active_prediction)
        self.assertEqual(executor.step_index, 0)
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
            return_value=(0, 0, None),
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
