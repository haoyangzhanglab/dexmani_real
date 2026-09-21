"""Offline tests for single-owner projection and recoverable misses (task T3).

Covers:
* V09 one soft-jump clip at the projection owner, canonicalization, float64
  round-off guard, hard-bound preservation, and the reject-style producer
  validator used by keyboard/calibration jogs;
* V10 an ordinary IK/workspace miss drops only the unpublished chunk suffix —
  the trial, the committed prefix, and the continuity reference survive —
  while post-projection invariant violations become session failures and are
  never swallowed as recoverable misses.

The runner-level cases import the deployment runtime (numpy, scipy,
pinocchio, ...); when those are unavailable the suite is skipped rather than
reporting false passes.
"""

from __future__ import annotations

import unittest
from collections import deque
from types import SimpleNamespace

import numpy as np

from dexmani_real.robot.projection import (
    ARM_COMMAND_JUMP_REJECTION,
    project_arm_command,
    project_hand_command,
    validate_arm_command,
)

# Only a genuinely missing dependency may skip this module: a renamed or
# broken symbol must fail the suite rather than hide behind a skip.
try:
    from dexmani_real.robot.commands import PreparedCommand
    from dexmani_real.robot.commands import GateRejectCode
    from dexmani_real.deployment.runner import (
        _RejectKind,
        _build_policy_workspace_check,
    )
    from dexmani_real.deployment.runner import PolicyRunner

    _IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - environment guard
    PreparedCommand = None
    GateRejectCode = None
    _RejectKind = None
    _build_policy_workspace_check = None
    PolicyRunner = None
    _IMPORT_ERROR = exc

_LOWER = np.array([-6.283, -2.059, -6.283, -0.192, -6.283, -1.693, -6.283])
_UPPER = np.array([6.283, 2.094, 6.283, 3.927, 6.283, 3.142, 6.283])
_JUMP = 0.349


class ProjectionOwnerTest(unittest.TestCase):
    """Pure projection contract — no IPC, no hardware."""

    def test_big_jump_is_clipped_once_to_the_bound(self):
        reference = np.zeros(7)
        target = np.full(7, 3.0)
        projected = project_arm_command(
            target,
            reference,
            joint_lower_rad=_LOWER,
            joint_upper_rad=_UPPER,
            max_command_jump_rad=_JUMP,
        )
        delta = np.abs(projected - reference)
        self.assertTrue(np.all(delta <= _JUMP))
        self.assertTrue(np.allclose(delta, _JUMP))
        # One clip: re-projecting the result is a fixed point.
        again = project_arm_command(
            projected,
            reference,
            joint_lower_rad=_LOWER,
            joint_upper_rad=_UPPER,
            max_command_jump_rad=_JUMP,
        )
        self.assertTrue(np.array_equal(again, projected))

    def test_reported_projection_names_the_truncation(self):
        """A real truncation is reported for the producer's visible [CLIP] line."""
        from dexmani_real.robot.projection import project_arm_command_reported

        reference = np.zeros(7)
        target = np.full(7, 3.0)
        projected, report = project_arm_command_reported(
            target,
            reference,
            joint_lower_rad=_LOWER,
            joint_upper_rad=_UPPER,
            max_command_jump_rad=_JUMP,
        )
        self.assertTrue(report.clipped)
        self.assertEqual(report.joint, 0)
        self.assertAlmostEqual(report.max_abs_delta_rad, 3.0)
        self.assertTrue(np.array_equal(projected, np.full(7, _JUMP)))
        # A step inside the bound is not a truncation and stays silent.
        _, quiet = project_arm_command_reported(
            reference + 0.01,
            reference,
            joint_lower_rad=_LOWER,
            joint_upper_rad=_UPPER,
            max_command_jump_rad=_JUMP,
        )
        self.assertFalse(quiet.clipped)
        self.assertEqual(quiet.joint, -1)

    def test_roundoff_guard_keeps_strict_bound(self):
        # reference + clip can round beyond the strict float64 bound; the
        # nextafter guard must keep |delta| <= limit exactly.
        reference = np.full(7, 0.123456789012345678)
        target = reference + 10.0
        projected = project_arm_command(
            target,
            reference,
            joint_lower_rad=_LOWER,
            joint_upper_rad=_UPPER,
            max_command_jump_rad=0.07,
        )
        self.assertTrue(np.all(np.abs(projected - reference) <= 0.07))
        self.assertTrue(np.all(projected >= _LOWER) and np.all(projected <= _UPPER))

    def test_canonicalization_prefers_nearest_equivalent(self):
        # Joint 1 has a >2π span: 6.0 rad is equivalent to 6.0-2π ≈ -0.283,
        # which is far nearer to the -6.0 reference than 6.0 itself.
        reference = np.zeros(7)
        reference[0] = -6.0
        target = np.zeros(7)
        target[0] = 6.0
        projected = project_arm_command(
            target,
            reference,
            joint_lower_rad=_LOWER,
            joint_upper_rad=_UPPER,
            max_command_jump_rad=10.0,
        )
        self.assertLess(abs(projected[0] - (6.0 - 2.0 * np.pi)), 1e-9)

    def test_invalid_reference_or_target_raises(self):
        with self.assertRaises(ValueError):
            project_arm_command(
                np.zeros(7),
                _UPPER + 1.0,  # reference outside joint limits
                joint_lower_rad=_LOWER,
                joint_upper_rad=_UPPER,
                max_command_jump_rad=_JUMP,
            )
        with self.assertRaises(ValueError):
            project_arm_command(
                np.full(7, np.nan),
                np.zeros(7),
                joint_lower_rad=_LOWER,
                joint_upper_rad=_UPPER,
                max_command_jump_rad=_JUMP,
            )

    def test_hand_projection_clips_into_command_box(self):
        hand = np.linspace(-5.0, 5.0, 12)
        projected = project_hand_command(hand, qpos_min_rad=-0.7, qpos_max_rad=0.9)
        self.assertTrue(np.all(projected >= -0.7) and np.all(projected <= 0.9))
        with self.assertRaises(ValueError):
            project_hand_command(np.zeros(11), qpos_min_rad=-0.7, qpos_max_rad=0.9)

    def test_reject_style_producer_validator(self):
        ok = validate_arm_command(
            np.full(7, 0.1),
            np.zeros(7),
            joint_lower_rad=_LOWER,
            joint_upper_rad=_UPPER,
            max_command_jump_rad=_JUMP,
        )
        self.assertIsNone(ok)
        jump = validate_arm_command(
            np.full(7, 0.5),
            np.zeros(7),
            joint_lower_rad=_LOWER,
            joint_upper_rad=_UPPER,
            max_command_jump_rad=_JUMP,
        )
        self.assertEqual(jump, ARM_COMMAND_JUMP_REJECTION)
        limit = validate_arm_command(
            np.full(7, 2.5),
            np.full(7, 2.4),
            joint_lower_rad=_LOWER,
            joint_upper_rad=_UPPER,
            max_command_jump_rad=_JUMP,
        )
        self.assertEqual(limit, "joint limit violation")
        finite = validate_arm_command(
            np.full(7, np.inf),
            np.zeros(7),
            joint_lower_rad=_LOWER,
            joint_upper_rad=_UPPER,
            max_command_jump_rad=_JUMP,
        )
        self.assertEqual(finite, "non-finite target")


@unittest.skipIf(
    _IMPORT_ERROR is not None,
    f"deployment runtime dependencies unavailable: {_IMPORT_ERROR}",
)
class RecoverableMissTest(unittest.TestCase):
    """V10: ordinary misses replan in-trial; contract breaks do not."""

    def _runner(self):
        runner = PolicyRunner.__new__(PolicyRunner)
        runner.run_generation = 7
        runner.last_publication_ns = None
        runner.previous_arm_command_qpos = np.full(7, 0.25)
        runner.actions = deque([np.array([1.0]), np.array([2.0])])
        runner._pending_dispatch = None
        runner.run_started_ns = 12345
        runner.recorder = None
        runner.shared = None
        from unittest import mock

        runner._invalidate_rollout = mock.Mock(return_value=True)
        runner._request_failed_session_shutdown = mock.Mock()
        return runner

    def test_ik_miss_keeps_trial_prefix_and_reference(self):
        runner = self._runner()
        runner._handle_recoverable_miss(
            "ik_no_solution",
            reject_kind=_RejectKind.IK,
            raw_action=np.array([0.0]),
        )
        # The trial continues; only the unpublished suffix was dropped.
        self.assertEqual(runner.run_started_ns, 12345)
        self.assertEqual(list(runner.actions), [])
        # New predictions still anchor behind the last committed command.
        self.assertTrue(
            np.array_equal(runner.previous_arm_command_qpos, np.full(7, 0.25))
        )
        runner._invalidate_rollout.assert_not_called()
        runner._request_failed_session_shutdown.assert_not_called()

    def test_workspace_gate_rejection_is_recoverable(self):
        runner = self._runner()
        prepared = PreparedCommand(
            reason="workspace", gate_code=GateRejectCode.WORKSPACE
        )
        runner._handle_preparation_rejection(prepared, raw_action=np.array([0.0]))
        self.assertEqual(runner.run_started_ns, 12345)
        runner._invalidate_rollout.assert_not_called()

    def test_post_projection_invariant_break_is_session_failure(self):
        runner = self._runner()
        prepared = PreparedCommand(
            reason="arm joint limit violation",
            gate_code=GateRejectCode.ARM_JOINT_LIMIT,
        )
        runner._handle_preparation_rejection(prepared, raw_action=np.array([0.0]))
        runner._invalidate_rollout.assert_called_once()
        runner._request_failed_session_shutdown.assert_called_once()

    def test_failed_check_is_session_failure_not_recoverable(self):
        runner = self._runner()
        prepared = PreparedCommand(
            reason="workspace check failed",
            gate_code=GateRejectCode.WORKSPACE_CHECK_FAILED,
            fatal=True,
        )
        runner._handle_preparation_rejection(prepared, raw_action=np.array([0.0]))
        runner._invalidate_rollout.assert_called_once()
        runner._request_failed_session_shutdown.assert_called_once()


@unittest.skipIf(
    _IMPORT_ERROR is not None,
    f"deployment runtime dependencies unavailable: {_IMPORT_ERROR}",
)
class FullRetryKeepsCandidateTest(unittest.TestCase):
    """V05: a FULL commit keeps the identical prepared candidate."""

    def test_full_never_reprepares_or_advances_state(self):
        from unittest import mock

        import dexmani_real.deployment.runner as executor_module
        from dexmani_real.robot.commands import ActionCandidate
        from dexmani_real.robot.commands import PublishResult

        runner = PolicyRunner.__new__(PolicyRunner)
        candidate = ActionCandidate(
            run_generation=7, arm_qpos=np.full(7, 0.1)
        )
        runner._pending_dispatch = candidate
        runner.shared = SimpleNamespace()
        runner._running_generation_is_live = lambda: True
        runner._running_time_expired = lambda now_ns: False
        runner.execute = True
        runner.last_publication_ns = 1_000_000
        runner.actions = deque([np.array([1.0])])
        runner._prepare_dispatch_candidate = mock.Mock(
            side_effect=AssertionError("FULL retry must not re-prepare")
        )
        full = PublishResult(
            False, reason="command fifo full", fifo_depth=8
        )
        with mock.patch.object(executor_module, "publish_command", return_value=full):
            runner._dispatch_action(np.array([0.5]))

        runner._prepare_dispatch_candidate.assert_not_called()
        self.assertIs(runner._pending_dispatch, candidate)
        self.assertEqual(len(runner.actions), 1)  # head not popped
        self.assertEqual(runner.last_publication_ns, 1_000_000)  # cadence kept


@unittest.skipIf(
    _IMPORT_ERROR is not None,
    f"deployment runtime dependencies unavailable: {_IMPORT_ERROR}",
)
class EndpointWorkspaceCheckTest(unittest.TestCase):
    """The joint-policy workspace critic is an endpoint check."""

    def _runtime(self, center: np.ndarray, half_width: float):
        workspace = SimpleNamespace(
            as_tuple=lambda: (
                (center[0] - half_width, center[0] + half_width),
                (center[1] - half_width, center[1] + half_width),
                (center[2] - half_width, center[2] + half_width),
            )
        )
        return SimpleNamespace(policy=SimpleNamespace(workspace=workspace))

    def test_endpoint_inside_and_outside(self):
        from dexmani_real.planning.kinematics.arm_fk import make_arm_fk

        fk = make_arm_fk()
        inside_qpos = np.zeros(7)
        position, _ = fk.compute(inside_qpos)
        check = _build_policy_workspace_check(
            self._runtime(np.asarray(position, dtype=np.float64), 0.05)
        )
        self.assertTrue(check(inside_qpos, inside_qpos))
        outside_qpos = np.array([0.0, 0.6, 0.0, 0.6, 0.0, 0.6, 0.0])
        outside_position, _ = fk.compute(outside_qpos)
        if np.max(np.abs(np.asarray(outside_position) - position)) > 0.05:
            self.assertFalse(check(inside_qpos, outside_qpos))
        else:  # pragma: no cover - geometry guard
            self.skipTest("chosen endpoint is not outside the tiny workspace")


class PolicyEndpointClipReportingTest(unittest.TestCase):
    """Real decode/projection/prepare/dispatch; only devices and FIFO capacity are fake."""

    def _runner_and_action(self, key="action", *, arm_clip=False, hand_clip=False, tiny=False):
        import threading
        from unittest import mock
        from dexmani_real.config.experiment import resolve_experiment_config
        from dexmani_real.robot.commands import (CommandFeedbackSnapshot)
        from dexmani_real.robot.commands import SafetyGate
        from dexmani_real.runtime.safety import SafetyState
        from test_deployment_evidence import _fake_shared
        runtime = resolve_experiment_config()
        runner = PolicyRunner.__new__(PolicyRunner)
        runner.runtime = runtime
        runner.shared = _fake_shared(safety_state=SafetyState.RUNNING)
        lock = threading.RLock()
        runner.run_generation = runner.shared.run_generation.value
        runner.previous_arm_command_qpos = np.asarray(runtime.arm.home_qpos).copy()
        arm = runner.previous_arm_command_qpos.copy()
        if arm_clip:
            arm[0] += 1.
        low, high = np.asarray(runtime.hand.qpos_min_rad), np.asarray(runtime.hand.qpos_max_rad)
        hand = (low + high) / 2
        if hand_clip:
            hand[3] = np.nextafter(high[3], np.inf) if tiny else high[3] + .25
        runner.policy_spec = SimpleNamespace(action_key=key)
        runner.ee_planner = mock.Mock()
        runner.ee_planner.solve_teleop_ik.return_value = SimpleNamespace(success=True, qpos=arm.copy())
        action = np.concatenate((arm, hand)) if key == "action" else np.concatenate(
            (np.zeros(3), [1., 0., 0., 0., 1., 0.], hand))
        runner.gate = SafetyGate(arm_joint_lower_rad=runtime.arm.joint_limit_lower,
            arm_joint_upper_rad=runtime.arm.joint_limit_upper,
            hand_joint_lower_rad=runtime.hand.qpos_min_rad, hand_joint_upper_rad=runtime.hand.qpos_max_rad)
        # Deliberately unrelated measured hand: correction must use decoded target.
        feedback = CommandFeedbackSnapshot(arm_qpos=runner.previous_arm_command_qpos.copy(),
            arm_source_monotonic_ns=1, hand_qpos=low.copy(), hand_source_monotonic_ns=1)
        runner._session_failure = mock.Mock()
        runner._record_rollout_tick = mock.Mock()
        runner._pending_dispatch = None
        runner._running_generation_is_live = lambda: True
        runner._running_time_expired = lambda now: False
        runner.execute = True
        runner.last_publication_ns = None
        runner.session_publication_count = 0
        runner.actions = deque([action])
        return runner, action, feedback, arm, hand

    def test_real_candidate_logs_one_combined_line_and_preserves_values(self):
        from unittest import mock
        import dexmani_real.deployment.runner as executor
        for key in ("action", "action_ee"):
            for arm_clip, hand_clip in ((False, True), (True, False), (True, True), (False, False)):
                with self.subTest(key=key, arm=arm_clip, hand=hand_clip):
                    runner, action, feedback, arm, hand = self._runner_and_action(
                        key, arm_clip=arm_clip, hand_clip=hand_clip)
                    original = action.copy()
                    with mock.patch.object(executor, "read_command_feedback", return_value=(feedback, "", None)):
                        context = self.assertLogs(executor.logger.name, level="INFO") if arm_clip or hand_clip else self.assertNoLogs(executor.logger.name, level="INFO")
                        with context as logs:
                            candidate = runner._prepare_dispatch_candidate(action)
                    self.assertIsNotNone(candidate)
                    runner._session_failure.assert_not_called()
                    np.testing.assert_array_equal(action, original)
                    np.testing.assert_array_equal(candidate.arm_qpos, project_arm_command(
                        arm, runner.previous_arm_command_qpos,
                        joint_lower_rad=runner.runtime.arm.joint_limit_lower,
                        joint_upper_rad=runner.runtime.arm.joint_limit_upper,
                        max_command_jump_rad=runner.runtime.arm.max_servo_command_jump_rad))
                    np.testing.assert_array_equal(candidate.hand_qpos, np.clip(
                        hand, runner.runtime.hand.qpos_min_rad, runner.runtime.hand.qpos_max_rad))
                    if arm_clip or hand_clip:
                        lines = [line for line in logs.output if "[CLIP]" in line]
                        self.assertEqual(len(lines), 1)
                        self.assertEqual("arm_max_delta_rad=" in lines[0], arm_clip)
                        self.assertEqual("hand_max_correction_rad=" in lines[0], hand_clip)
                        if hand_clip:
                            self.assertIn("hand_joint=3", lines[0])
                            self.assertIn("hand_max_correction_rad=0.25", lines[0])

    def test_tiny_nonzero_hand_correction_is_visible(self):
        from unittest import mock
        import dexmani_real.deployment.runner as executor
        runner, action, feedback, _, hand = self._runner_and_action(hand_clip=True, tiny=True)
        with mock.patch.object(executor, "read_command_feedback", return_value=(feedback, "", None)):
            with self.assertLogs(executor.logger.name, level="INFO") as logs:
                candidate = runner._prepare_dispatch_candidate(action)
        line, = [line for line in logs.output if "[CLIP]" in line]
        magnitude = float(line.split("hand_max_correction_rad=")[1].split()[0])
        self.assertGreater(magnitude, 0.)
        self.assertEqual(magnitude, hand[3] - candidate.hand_qpos[3])

    def test_full_retries_project_and_report_only_once(self):
        from unittest import mock
        import dexmani_real.deployment.runner as executor
        from dexmani_real.robot.commands import PublishResult, PUBLISH_REASON_FIFO_FULL
        runner, action, feedback, _, _ = self._runner_and_action(hand_clip=True)
        original = action.copy()
        results = [PublishResult(False, reason=PUBLISH_REASON_FIFO_FULL, fifo_depth=4)] * 3
        results.append(PublishResult(True, command=SimpleNamespace(published_monotonic_ns=100)))
        with mock.patch.object(executor, "read_command_feedback", return_value=(feedback, "", None)), \
             mock.patch.object(executor, "_project_policy_targets", wraps=executor._project_policy_targets) as project, \
             mock.patch.object(executor, "publish_command", side_effect=results) as publish:
            with self.assertLogs(executor.logger.name, level="INFO") as logs:
                for index in range(4):
                    runner._dispatch_action(action)
                    if index < 3:
                        self.assertIsNone(runner.last_publication_ns)
        self.assertEqual(project.call_count, 1)
        candidates = [call.args[1] for call in publish.call_args_list]
        self.assertTrue(all(c is candidates[0] for c in candidates))
        self.assertEqual(sum("[CLIP]" in line for line in logs.output), 1)
        self.assertEqual(runner.session_publication_count, 1)
        self.assertEqual(len(runner.actions), 0)
        self.assertIsNone(runner._pending_dispatch)
        np.testing.assert_array_equal(action, original)

    def test_invalid_hand_shape_and_nan_still_fail_contract(self):
        from unittest import mock
        import dexmani_real.deployment.runner as executor
        for key in ("action", "action_ee"):
            for invalid in ("shape", "nan"):
                with self.subTest(key=key, invalid=invalid):
                    runner, action, feedback, _, _ = self._runner_and_action(key)
                    if invalid == "shape":
                        action = action[:-1]
                    else:
                        action[-1] = np.nan
                    with mock.patch.object(executor, "read_command_feedback", return_value=(feedback, "", None)):
                        with self.assertNoLogs(executor.logger.name, level="INFO"):
                            self.assertIsNone(runner._prepare_dispatch_candidate(action))
                    runner._session_failure.assert_called_once()
                    self.assertIn("projection invariant violation", runner._session_failure.call_args.args[0])


class PublicPolicyArtifactBoundaryTest(unittest.TestCase):
    def test_load_returns_public_runtime_and_pins_exact_artifact(self):
        from unittest import mock
        import sys
        from types import ModuleType
        from dexmani_real.deployment.runner import _load_policy_runtime

        config = SimpleNamespace(experiment="p/t/e", device="cpu", seed=0,
                                 artifact="pinned.pt", inference_steps=8, spec=object())
        loaded = SimpleNamespace(spec=config.spec, close=mock.Mock())
        public = ModuleType("dexmani_policy.deployment")
        public.load_experiment = mock.Mock(return_value=loaded)
        with mock.patch.dict(sys.modules, {"dexmani_policy.deployment": public}):
            self.assertIs(_load_policy_runtime(config), loaded)
        public.load_experiment.assert_called_once_with(
            "p/t/e", device="cpu", seed=0, artifact="pinned.pt", inference_steps=8)
        loaded.close.assert_not_called()

    def test_changed_spec_closes_loaded_runtime_before_rejecting(self):
        from unittest import mock
        import sys
        from types import ModuleType
        from dexmani_real.deployment.runner import _load_policy_runtime

        config = SimpleNamespace(experiment="p/t/e", device="cpu", seed=0,
                                 artifact="pinned.pt", inference_steps=8, spec=object())
        loaded = SimpleNamespace(spec=object(), close=mock.Mock())
        public = ModuleType("dexmani_policy.deployment")
        public.load_experiment = mock.Mock(return_value=loaded)
        with mock.patch.dict(sys.modules, {"dexmani_policy.deployment": public}):
            with self.assertRaises(RuntimeError):
                _load_policy_runtime(config)
        loaded.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
