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

from dexmani_real.control.projection import (
    ARM_COMMAND_JUMP_REJECTION,
    project_arm_command,
    project_hand_command,
    validate_arm_command,
)

try:
    from dexmani_real.control.publication import PreparedCommand
    from dexmani_real.control.safety_gate import GateRejectCode
    from dexmani_real.deployment.executor import (
        _RejectKind,
        _build_policy_workspace_check,
    )
    from dexmani_real.deployment.executor import PolicyRunner
    from dexmani_real.deployment.metrics import PolicyStats

    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment guard
    PreparedCommand = None
    GateRejectCode = None
    _RejectKind = None
    _build_policy_workspace_check = None
    PolicyRunner = None
    PolicyStats = None
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
        runner.observation_id = 3
        runner.chunk_action_index = 1
        runner.last_publication_ns = None
        runner.previous_arm_command_qpos = np.full(7, 0.25)
        runner.chunk_sources = {}
        runner.actions = deque([np.array([1.0]), np.array([2.0])])
        runner._pending_dispatch = None
        runner._fifo_wait = SimpleNamespace(waiting=False, note_dropped=lambda reason: None)
        runner.stats = PolicyStats()
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
        self.assertEqual(runner.chunk_action_index, 0)
        # New predictions still anchor behind the last committed command.
        self.assertTrue(
            np.array_equal(runner.previous_arm_command_qpos, np.full(7, 0.25))
        )
        self.assertEqual(runner.stats.rejection_reasons.get("ik_no_solution"), 1)
        runner._invalidate_rollout.assert_not_called()
        runner._request_failed_session_shutdown.assert_not_called()

    def test_workspace_gate_rejection_is_recoverable(self):
        runner = self._runner()
        prepared = PreparedCommand(
            reason="workspace", gate_code=GateRejectCode.WORKSPACE
        )
        runner._handle_preparation_rejection(prepared, raw_action=np.array([0.0]))
        self.assertEqual(runner.run_started_ns, 12345)
        self.assertEqual(runner.stats.safety_rejection_count, 1)
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

        import dexmani_real.deployment.executor as executor_module
        from dexmani_real.control.action import ActionCandidate
        from dexmani_real.control.publication import PublishResult

        runner = PolicyRunner.__new__(PolicyRunner)
        candidate = ActionCandidate(
            run_generation=7, action_id=41, arm_qpos=np.full(7, 0.1)
        )
        runner._pending_dispatch = candidate
        runner.shared = SimpleNamespace()
        runner._running_generation_is_live = lambda: True
        runner._running_time_expired = lambda now_ns: False
        runner.execute = True
        runner.last_publication_ns = 1_000_000
        runner.actions = deque([np.array([1.0])])
        runner.chunk_action_index = 0
        observed: dict = {}
        runner._fifo_wait = SimpleNamespace(
            waiting=False,
            note_full=lambda depth, keep: observed.update(depth=depth, keep=keep),
            note_committed=lambda: observed.update(committed=True),
            note_dropped=lambda reason: observed.update(dropped=reason),
        )
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
        self.assertEqual(runner.chunk_action_index, 0)
        self.assertEqual(observed.get("keep"), 41)
        self.assertNotIn("committed", observed)


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


if __name__ == "__main__":
    unittest.main()
