"""Offline regressions for learned-policy coordinator failure semantics."""

from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.control.action import ActionCandidate
from dexmani_real.control.publication import (
    CommandPublishResult,
    CommandPublishStatus,
    build_action_candidate,
)
from dexmani_real.control.safety_gate import GateRejectCode, SafetyGate
from dexmani_real.deployment.contracts import ActionChunk
from dexmani_real.deployment.coordinator import (
    PolicyEndpointDisposition,
    _advance_endpoint_due_ns,
    _AsyncExecution,
    _build_policy_planner,
    _build_policy_safety_gate,
    _build_policy_workspace_check,
    _chunk_source_deadline_ns,
    _chunk_source_is_stale,
    _command_watchdog_abort_reason,
    _CommandProgressWatchdog,
    _end_policy_run,
    _EpisodeActionSteps,
    _newer_async_execution,
    _record_terminal_before_finalize,
    _scheduled_endpoint_due_ns,
    _selected_async_target_is_due,
    _SyncExecution,
    classify_policy_endpoint_disposition,
)
from dexmani_real.deployment.metrics import (
    EPISODE_ACTION_STEPS,
    SAFETY_REJECTED_STEPS,
    SUCCESSFUL_ACTION_STEPS,
    Metrics,
)
from dexmani_real.deployment.timing import (
    first_future_step_index,
    next_periodic_deadline_ns,
)
from dexmani_real.runtime.safety import SafetyState, revoke_motion


class _Value:
    def __init__(self, value: int | bool) -> None:
        self.value = value


class _LockedValue(_Value):
    def __init__(self, value: int) -> None:
        super().__init__(value)
        self._lock = threading.Lock()

    def get_lock(self) -> threading.Lock:
        return self._lock


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


def _action_chunk(
    *,
    chunk_id: int = 1,
    run_generation: int = 1,
    logical_start_ns: int = 20,
    num_steps: int = 2,
    source_ns: int = 10,
) -> ActionChunk:
    return ActionChunk(
        chunk_id=chunk_id,
        run_generation=run_generation,
        observation_id=1,
        observation_anchor_monotonic_ns=max(30, logical_start_ns),
        observation_latest_source_monotonic_ns=source_ns,
        observation_logical_step_monotonic_ns=logical_start_ns,
        inference_started_monotonic_ns=max(30, logical_start_ns),
        inference_finished_monotonic_ns=max(40, logical_start_ns),
        num_steps=num_steps,
        arm_present=True,
        ee_present=False,
        hand_present=True,
        arm_qpos=np.zeros((num_steps, 7), dtype=np.float64),
        hand_qpos=np.zeros((num_steps, 12), dtype=np.float64),
    )


class PolicyCoordinatorRegressionTest(unittest.TestCase):
    def test_joint_policy_gate_uses_arm_fk_without_constructing_planner(self) -> None:
        config = SimpleNamespace(
            workspace_bounds=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
            ik_max_pose_error_pos_m=0.008,
            ik_max_pose_error_rot_rad=0.08,
            arm_joint_lower_rad=(-1.0,) * 7,
            arm_joint_upper_rad=(1.0,) * 7,
            hand_joint_lower_rad=(-1.0,) * 12,
            hand_joint_upper_rad=(1.0,) * 12,
            arm_max_delta_rad_per_tick=1.0,
            hand_max_delta_rad_per_tick=None,
            endpoint_delta_tolerance_rad=0.0,
        )
        arm_fk = Mock()
        arm_fk.compute.return_value = (np.zeros(3, dtype=np.float64), np.zeros(6))

        with (
            patch(
                "dexmani_real.deployment.coordinator.make_arm_fk",
                return_value=arm_fk,
            ),
            patch(
                "dexmani_real.deployment.coordinator.XArm7MotionPlanner"
            ) as planner_factory,
        ):
            gate = _build_policy_safety_gate(config)
        result = gate.validate(
            _candidate(),
            current_arm_qpos=np.zeros(7, dtype=np.float64),
            current_hand_qpos=np.zeros(12, dtype=np.float64),
            run_generation=1,
        )

        self.assertTrue(result.accepted)
        self.assertIsNone(gate.collision_check)
        arm_fk.compute.assert_called_once()
        planner_factory.assert_not_called()

        with patch(
            "dexmani_real.deployment.coordinator.XArm7MotionPlanner"
        ) as planner_factory:
            _build_policy_planner(config)

        kwargs = planner_factory.call_args.kwargs
        self.assertFalse(kwargs["teleop_profile"].check_self_collision)
        self.assertFalse(kwargs["hand_dof"])

    def test_policy_workspace_check_interpolates_and_fails_closed(self) -> None:
        config = SimpleNamespace(
            workspace_bounds=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
        )
        arm_fk = Mock()
        arm_fk.compute.return_value = (np.zeros(3, dtype=np.float64), np.zeros(6))

        with patch(
            "dexmani_real.deployment.coordinator.make_arm_fk",
            return_value=arm_fk,
        ):
            workspace_check = _build_policy_workspace_check(config)

        self.assertTrue(
            workspace_check(np.zeros(7, dtype=np.float64), np.full(7, 0.04))
        )
        self.assertEqual(arm_fk.compute.call_count, 3)

        arm_fk.compute.reset_mock()
        arm_fk.compute.return_value = (
            np.array([1.002, 0.0, 0.0], dtype=np.float64),
            np.zeros(6),
        )
        self.assertFalse(
            workspace_check(
                np.zeros(7, dtype=np.float64), np.zeros(7, dtype=np.float64)
            )
        )

        arm_fk.compute.side_effect = RuntimeError("FK failed")
        self.assertFalse(
            workspace_check(
                np.zeros(7, dtype=np.float64), np.zeros(7, dtype=np.float64)
            )
        )

    def test_ee_planner_is_deferred_until_requested(self) -> None:
        config = SimpleNamespace(
            workspace_bounds=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
            ik_max_pose_error_pos_m=0.008,
            ik_max_pose_error_rot_rad=0.08,
            arm_joint_lower_rad=(-1.0,) * 7,
            arm_joint_upper_rad=(1.0,) * 7,
            hand_joint_lower_rad=(-1.0,) * 12,
            hand_joint_upper_rad=(1.0,) * 12,
            arm_max_delta_rad_per_tick=1.0,
            hand_max_delta_rad_per_tick=None,
            endpoint_delta_tolerance_rad=0.0,
        )
        arm_fk = Mock()
        arm_fk.compute.return_value = (np.zeros(3, dtype=np.float64), np.zeros(6))

        with (
            patch(
                "dexmani_real.deployment.coordinator.make_arm_fk",
                return_value=arm_fk,
            ),
            patch(
                "dexmani_real.deployment.coordinator.XArm7MotionPlanner"
            ) as planner_factory,
        ):
            _build_policy_safety_gate(config)
            planner_factory.assert_not_called()
            _build_policy_planner(config)

        planner_factory.assert_called_once()

    def test_candidate_keeps_chunk_generation_after_revoke(self) -> None:
        shared = _FakeShared()
        shared.arm_command_seq = _LockedValue(0)
        chunk_generation = int(shared.run_generation.value)
        self.assertTrue(revoke_motion(shared, SafetyState.ARMED))

        candidate = build_action_candidate(
            shared,
            np.zeros(7, dtype=np.float64),
            np.zeros(12, dtype=np.float64),
            run_generation=chunk_generation,
            now_ns=100,
            scheduled_target_monotonic_ns=100,
        )

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.run_generation, chunk_generation)
        self.assertNotEqual(candidate.run_generation, shared.run_generation.value)

    def test_action_step_limit_counts_applied_and_rejected_terminals(self) -> None:
        max_one = _EpisodeActionSteps(1)
        self.assertTrue(max_one.record_successful())
        self.assertEqual(max_one.episode_action_steps, 1)
        self.assertEqual(max_one.successful_action_steps, 1)
        self.assertEqual(max_one.safety_rejected_steps, 0)

        max_three = _EpisodeActionSteps(3)
        self.assertFalse(max_three.record_successful())
        self.assertFalse(max_three.record_safety_rejected())
        self.assertTrue(max_three.record_successful())
        self.assertEqual(max_three.episode_action_steps, 3)
        self.assertEqual(max_three.successful_action_steps, 2)
        self.assertEqual(max_three.safety_rejected_steps, 1)

    def test_action_limit_stops_before_chunk_suffix_and_resets_next_episode(
        self,
    ) -> None:
        steps = _EpisodeActionSteps(3)
        attempted = 0
        for _ in range(8):
            attempted += 1
            if steps.record_successful():
                break
        self.assertEqual(attempted, 3)
        self.assertEqual(steps.episode_action_steps, 3)

        steps.reset()
        self.assertEqual(steps.episode_action_steps, 0)
        self.assertEqual(steps.successful_action_steps, 0)
        self.assertEqual(steps.safety_rejected_steps, 0)

    def test_limit_at_chunk_boundary_does_not_finalize_or_request_next(self) -> None:
        inference_request = threading.Event()
        at_limit = _EpisodeActionSteps(1)

        self.assertTrue(
            _record_terminal_before_finalize(
                at_limit.record_successful,
                inference_request.set,
            )
        )
        self.assertFalse(inference_request.is_set())

        below_limit = _EpisodeActionSteps(2)
        self.assertFalse(
            _record_terminal_before_finalize(
                below_limit.record_successful,
                inference_request.set,
            )
        )
        self.assertTrue(inference_request.is_set())

    def test_last_physical_publication_records_progress_before_limit_wait(self) -> None:
        progress = _CommandProgressWatchdog()
        progress.reset(run_generation=1)
        progress.observe(
            run_generation=1,
            arm_accepted_action_id=0,
            hand_accepted_action_id=0,
            now_monotonic_ns=100,
            timeout_ns=50,
        )
        progress.record_publication(
            run_generation=1,
            action_id=1,
            published_monotonic_ns=110,
        )
        execution = _SyncExecution(
            _action_chunk(num_steps=1),
            chunk_start_monotonic_ns=100,
        )
        inference_request = threading.Event()

        def finalize_chunk_tail() -> None:
            self.assertTrue(execution.finalize_current())
            inference_request.set()

        reached_limit = _record_terminal_before_finalize(
            _EpisodeActionSteps(1).record_successful,
            finalize_chunk_tail,
        )

        self.assertTrue(reached_limit)
        self.assertFalse(inference_request.is_set())
        self.assertEqual(execution.step_index, 0)
        self.assertFalse(progress.latest_publication_is_accepted)

    def test_defer_and_stale_are_not_terminal_action_counts(self) -> None:
        steps = _EpisodeActionSteps(1)
        for disposition in (
            PolicyEndpointDisposition.DEFER_TRANSIENT,
            PolicyEndpointDisposition.DISCARD_STALE,
        ):
            self.assertNotIn(
                disposition,
                {
                    PolicyEndpointDisposition.COMMIT,
                    PolicyEndpointDisposition.DISCARD_MOTION,
                },
            )
        self.assertEqual(steps.episode_action_steps, 0)
        self.assertFalse(steps.limit_reached)

    def test_action_counter_summary_always_contains_all_three_fields(self) -> None:
        metrics = Metrics()
        metrics.begin_episode(generation=1, started_monotonic_ns=1)
        metrics.increment(EPISODE_ACTION_STEPS, 0)
        metrics.increment(SUCCESSFUL_ACTION_STEPS, 0)
        metrics.increment(SAFETY_REJECTED_STEPS, 0)
        metrics.increment(EPISODE_ACTION_STEPS)
        metrics.increment(SUCCESSFUL_ACTION_STEPS)

        snapshot = metrics.episode_snapshot()
        self.assertEqual(snapshot[EPISODE_ACTION_STEPS], 1)
        self.assertEqual(snapshot[SUCCESSFUL_ACTION_STEPS], 1)
        self.assertEqual(snapshot[SAFETY_REJECTED_STEPS], 0)

    def test_successful_publication_and_motion_reject_are_terminal(self) -> None:
        steps = _EpisodeActionSteps(3)
        candidate = _candidate()

        shadow = CommandPublishResult(CommandPublishStatus.VALIDATED, candidate)
        self.assertIs(
            classify_policy_endpoint_disposition(
                shadow,
            ),
            PolicyEndpointDisposition.COMMIT,
        )
        self.assertFalse(steps.record_successful())

        published = CommandPublishResult(
            CommandPublishStatus.PUBLISHED,
            candidate,
        )
        self.assertIs(
            classify_policy_endpoint_disposition(
                published,
            ),
            PolicyEndpointDisposition.COMMIT,
        )
        # Successful IPC publication consumes the control slot immediately.
        self.assertFalse(steps.record_successful())

        rejected = CommandPublishResult(
            CommandPublishStatus.GATE_REJECTED,
            candidate,
            gate_code=GateRejectCode.ARM_JOINT_LIMIT,
        )
        self.assertIs(
            classify_policy_endpoint_disposition(
                rejected,
            ),
            PolicyEndpointDisposition.DISCARD_MOTION,
        )
        self.assertTrue(steps.record_safety_rejected())
        self.assertEqual(steps.successful_action_steps, 2)
        self.assertEqual(steps.safety_rejected_steps, 1)

    def test_first_future_step_index_includes_exact_target(self) -> None:
        cases = (
            (99, 0),
            (100, 0),
            (101, 1),
            (110, 1),
            (111, 2),
            (120, 2),
            (121, None),
        )
        for now_ns, expected in cases:
            with self.subTest(now_ns=now_ns):
                self.assertEqual(
                    first_future_step_index(100, 10, now_ns, 3),
                    expected,
                )

    def test_absolute_async_cadence_skips_missed_periods(self) -> None:
        self.assertEqual(next_periodic_deadline_ns(100, 20, 100), 120)
        self.assertEqual(next_periodic_deadline_ns(100, 20, 119), 120)
        self.assertEqual(next_periodic_deadline_ns(100, 20, 145), 160)

    def test_newer_async_chunk_replaces_but_expired_chunk_is_dropped(self) -> None:
        current = _newer_async_execution(
            None,
            _action_chunk(chunk_id=1, logical_start_ns=100, num_steps=3),
            run_generation=1,
            now_ns=100,
            step_dt_ns=10,
        )
        self.assertIsNotNone(current)
        assert current is not None

        replacement = _newer_async_execution(
            current,
            _action_chunk(chunk_id=2, logical_start_ns=100, num_steps=3),
            run_generation=1,
            now_ns=111,
            step_dt_ns=10,
        )
        self.assertIsNot(replacement, current)
        assert replacement is not None
        self.assertEqual(replacement.chunk.chunk_id, 2)
        self.assertEqual(replacement.step_index, 2)

        self.assertIs(
            _newer_async_execution(
                replacement,
                _action_chunk(chunk_id=3, logical_start_ns=100, num_steps=3),
                run_generation=1,
                now_ns=121,
                step_dt_ns=10,
            ),
            replacement,
        )
        self.assertIsNone(
            _newer_async_execution(
                None,
                _action_chunk(chunk_id=3, logical_start_ns=100, num_steps=3),
                run_generation=1,
                now_ns=121,
                step_dt_ns=10,
            )
        )

    def test_bounded_async_wait_can_ingest_newer_chunk_before_due(self) -> None:
        current = _AsyncExecution(
            _action_chunk(chunk_id=1, logical_start_ns=100, num_steps=3),
            step_index=1,
        )
        self.assertTrue(current.advance_to_first_future(now_ns=105, step_dt_ns=10))
        self.assertEqual(current.scheduled_target_ns(10), 110)

        replacement = _newer_async_execution(
            current,
            _action_chunk(chunk_id=2, logical_start_ns=105, num_steps=3),
            run_generation=1,
            now_ns=105,
            step_dt_ns=10,
        )
        self.assertIsNot(replacement, current)
        assert replacement is not None
        self.assertEqual(replacement.chunk.chunk_id, 2)

    def test_async_non_aligned_wake_retains_selected_due_endpoint(self) -> None:
        execution = _AsyncExecution(
            _action_chunk(logical_start_ns=100, num_steps=3),
            step_index=1,
        )

        # A 110 ns target selected before sleeping remains due at a 111 ns
        # wake-up instead of being skipped to 120 ns.
        target_ns = execution.scheduled_target_ns(10)
        self.assertTrue(
            _selected_async_target_is_due(
                target_ns=target_ns,
                step_dt_ns=10,
                now_ns=111,
            )
        )
        self.assertTrue(execution.advance_to_first_future(now_ns=111, step_dt_ns=10))
        self.assertEqual(execution.step_index, 1)
        self.assertFalse(
            _selected_async_target_is_due(
                target_ns=target_ns,
                step_dt_ns=10,
                now_ns=120,
            )
        )

        self.assertTrue(execution.advance_to_first_future(now_ns=120, step_dt_ns=10))
        self.assertEqual(execution.step_index, 2)

    def test_source_deadline_is_exact_and_rejects_uint64_overflow(self) -> None:
        self.assertEqual(
            _chunk_source_deadline_ns(_action_chunk(), max_source_age_ns=10),
            20,
        )
        with self.assertRaisesRegex(ValueError, "must be positive"):
            _chunk_source_deadline_ns(_action_chunk(), max_source_age_ns=0)

        uint64_max = int(np.iinfo(np.uint64).max)
        overflowing = _action_chunk(
            logical_start_ns=uint64_max - 2,
            source_ns=uint64_max - 3,
        )
        with self.assertRaisesRegex(ValueError, "exceeds uint64"):
            _chunk_source_deadline_ns(overflowing, max_source_age_ns=4)

    def test_sync_execution_is_sequential_and_rebases_timing(self) -> None:
        execution = _SyncExecution(
            _action_chunk(),
            chunk_start_monotonic_ns=100,
        )

        self.assertEqual(execution.scheduled_target_ns(10), 100)
        self.assertFalse(execution.finalize_current())
        self.assertEqual(execution.step_index, 1)
        self.assertEqual(execution.scheduled_target_ns(10), 110)
        self.assertTrue(execution.finalize_current())

    def test_small_endpoint_jitter_does_not_accumulate(self) -> None:
        step_dt_ns = 10
        endpoint_due_ns = 100
        for index in range(600):
            terminal_ns = endpoint_due_ns + (index % 3)
            endpoint_due_ns = _advance_endpoint_due_ns(
                endpoint_due_ns,
                terminal_ns,
                step_dt_ns,
            )

        self.assertEqual(endpoint_due_ns, 100 + 600 * step_dt_ns)

    def test_full_missed_slot_reanchors_without_catch_up(self) -> None:
        self.assertEqual(_advance_endpoint_due_ns(100, 109, 10), 110)
        self.assertEqual(_advance_endpoint_due_ns(100, 110, 10), 120)
        self.assertEqual(_advance_endpoint_due_ns(100, 125, 10), 135)

    def test_sync_chunk_boundary_reuses_next_absolute_slot(self) -> None:
        next_due_ns = _advance_endpoint_due_ns(100, 101, 10)
        next_chunk = _SyncExecution(
            _action_chunk(chunk_id=2),
            chunk_start_monotonic_ns=107,
        )

        self.assertEqual(next_chunk.scheduled_target_ns(10), 107)
        self.assertEqual(
            _scheduled_endpoint_due_ns(
                next_chunk.scheduled_target_ns(10),
                next_due_ns,
            ),
            110,
        )

    def test_sync_publication_advances_without_waiting_for_worker_progress(
        self,
    ) -> None:
        execution = _SyncExecution(_action_chunk(), chunk_start_monotonic_ns=100)
        self.assertFalse(execution.finalize_current())
        self.assertEqual(execution.step_index, 1)

    def test_async_publication_advances_without_waiting_for_worker_progress(
        self,
    ) -> None:
        execution = _AsyncExecution(
            _action_chunk(logical_start_ns=100),
            step_index=0,
        )
        self.assertFalse(execution.finalize_current())
        self.assertEqual(execution.step_index, 1)

    def test_sync_terminal_dispositions_and_source_staleness(self) -> None:
        candidate = _candidate()
        results = (
            (
                CommandPublishResult(CommandPublishStatus.VALIDATED, candidate),
                PolicyEndpointDisposition.COMMIT,
            ),
            (
                CommandPublishResult(
                    CommandPublishStatus.GATE_REJECTED,
                    candidate,
                    gate_code=GateRejectCode.ARM_JOINT_LIMIT,
                ),
                PolicyEndpointDisposition.DISCARD_MOTION,
            ),
            (
                CommandPublishResult(
                    CommandPublishStatus.GATE_REJECTED,
                    candidate,
                    gate_code=GateRejectCode.COLLISION_TRANSITION,
                ),
                PolicyEndpointDisposition.ABORT_FATAL,
            ),
            (
                CommandPublishResult(
                    CommandPublishStatus.ARM_FEEDBACK_UNAVAILABLE,
                    candidate,
                ),
                PolicyEndpointDisposition.DEFER_TRANSIENT,
            ),
        )
        for result, expected in results:
            with self.subTest(expected=expected):
                self.assertIs(
                    classify_policy_endpoint_disposition(
                        result,
                    ),
                    expected,
                )

        chunk = _action_chunk()
        self.assertFalse(
            _chunk_source_is_stale(
                chunk,
                now_monotonic_ns=20,
                max_source_age_ns=10,
            )
        )
        self.assertTrue(
            _chunk_source_is_stale(
                chunk,
                now_monotonic_ns=21,
                max_source_age_ns=10,
            )
        )

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
            ),
            PolicyEndpointDisposition.DISCARD_MOTION,
        )

    def test_worker_action_id_jumps_are_healthy_progress(self) -> None:
        progress = _CommandProgressWatchdog()
        progress.reset(run_generation=1)
        self.assertIsNone(
            progress.observe(
                run_generation=1,
                arm_accepted_action_id=4,
                hand_accepted_action_id=4,
                now_monotonic_ns=100,
                timeout_ns=50,
            )
        )
        progress.record_publication(
            run_generation=1,
            action_id=5,
            published_monotonic_ns=110,
        )

        self.assertIsNone(
            progress.observe(
                run_generation=1,
                arm_accepted_action_id=8,
                hand_accepted_action_id=9,
                now_monotonic_ns=120,
                timeout_ns=50,
            )
        )
        self.assertTrue(progress.latest_publication_is_accepted)

    def test_one_stalled_worker_times_out(self) -> None:
        progress = _CommandProgressWatchdog()
        progress.reset(run_generation=1)
        progress.observe(
            run_generation=1,
            arm_accepted_action_id=4,
            hand_accepted_action_id=4,
            now_monotonic_ns=100,
            timeout_ns=50,
        )
        progress.record_publication(
            run_generation=1,
            action_id=5,
            published_monotonic_ns=110,
        )

        reason = progress.observe(
            run_generation=1,
            arm_accepted_action_id=5,
            hand_accepted_action_id=4,
            now_monotonic_ns=161,
            timeout_ns=50,
        )

        self.assertEqual(reason, "hand worker command progress timeout")

    def test_new_publication_does_not_hide_existing_stall(self) -> None:
        progress = _CommandProgressWatchdog()
        progress.reset(run_generation=1)
        progress.observe(
            run_generation=1,
            arm_accepted_action_id=0,
            hand_accepted_action_id=0,
            now_monotonic_ns=100,
            timeout_ns=50,
        )
        progress.record_publication(
            run_generation=1,
            action_id=1,
            published_monotonic_ns=110,
        )
        progress.record_publication(
            run_generation=1,
            action_id=2,
            published_monotonic_ns=140,
        )

        reason = progress.observe(
            run_generation=1,
            arm_accepted_action_id=0,
            hand_accepted_action_id=0,
            now_monotonic_ns=161,
            timeout_ns=50,
        )

        self.assertEqual(reason, "arm worker command progress timeout")

    def test_progress_reset_requires_new_generation_publication(self) -> None:
        progress = _CommandProgressWatchdog()
        progress.reset(run_generation=1)
        progress.observe(
            run_generation=1,
            arm_accepted_action_id=4,
            hand_accepted_action_id=4,
            now_monotonic_ns=100,
            timeout_ns=50,
        )
        progress.record_publication(
            run_generation=1,
            action_id=5,
            published_monotonic_ns=110,
        )
        progress.reset(run_generation=2)

        self.assertFalse(progress.latest_publication_is_accepted)
        self.assertIsNone(progress.latest_published_action_id)
        progress.observe(
            run_generation=2,
            arm_accepted_action_id=5,
            hand_accepted_action_id=5,
            now_monotonic_ns=120,
            timeout_ns=50,
        )
        progress.record_publication(
            run_generation=2,
            action_id=6,
            published_monotonic_ns=130,
        )
        self.assertFalse(progress.latest_publication_is_accepted)

    def test_action_limit_waits_for_last_publication_but_not_rejection(self) -> None:
        progress = _CommandProgressWatchdog()
        progress.reset(run_generation=1)
        progress.observe(
            run_generation=1,
            arm_accepted_action_id=0,
            hand_accepted_action_id=0,
            now_monotonic_ns=100,
            timeout_ns=50,
        )
        progress.record_publication(
            run_generation=1,
            action_id=1,
            published_monotonic_ns=110,
        )
        self.assertTrue(_EpisodeActionSteps(1).record_successful())
        self.assertFalse(progress.latest_publication_is_accepted)
        progress.observe(
            run_generation=1,
            arm_accepted_action_id=1,
            hand_accepted_action_id=1,
            now_monotonic_ns=120,
            timeout_ns=50,
        )
        self.assertTrue(progress.latest_publication_is_accepted)
        self.assertTrue(_EpisodeActionSteps(1).record_safety_rejected())

    def test_operator_stop_revokes_motion_and_home_authorization(self) -> None:
        shared = _FakeShared()
        generation_before_stop = int(shared.run_generation.value)

        _end_policy_run(shared, "operator stop", abort=False)

        self.assertEqual(shared.safety_state.value, int(SafetyState.ARMED))
        self.assertGreater(shared.run_generation.value, generation_before_stop)
        self.assertEqual(shared.active_coupled_command_sequence.value, 0)
        self.assertEqual(shared.run_started_monotonic_ns.value, 0)
        self.assertFalse(shared.physical_home_completed.value)

    def test_operator_stop_summarizes_motion_already_revoked_by_keyboard(self) -> None:
        shared = _FakeShared()
        self.assertTrue(revoke_motion(shared, SafetyState.ARMED))
        generation_after_keyboard_stop = int(shared.run_generation.value)
        metrics = Mock()

        _end_policy_run(
            shared,
            "operator stop",
            abort=False,
            metrics=metrics,
        )

        self.assertEqual(shared.safety_state.value, int(SafetyState.ARMED))
        self.assertEqual(shared.run_generation.value, generation_after_keyboard_stop)
        metrics.log_episode_summary.assert_called_once_with(
            status="STOPPED",
            reason="operator stop",
        )

    def test_action_limit_uses_truncated_summary_without_abort(self) -> None:
        shared = _FakeShared()
        metrics = Mock()

        _end_policy_run(
            shared,
            "action_step_limit",
            abort=False,
            metrics=metrics,
            summary_status="TRUNCATED",
        )

        self.assertEqual(shared.safety_state.value, int(SafetyState.ARMED))
        metrics.log_episode_summary.assert_called_once_with(
            status="TRUNCATED",
            reason="action_step_limit",
        )

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
