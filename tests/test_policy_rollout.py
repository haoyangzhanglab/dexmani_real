"""Deterministic offline regressions for the learned-policy rollout semantics.

No hardware, no trained checkpoint, no GPU.  These tests pin the public Policy
contract boundary, timestamp-based scheduling, the explicit IK-vs-SAFETY
attribution, reject-only arm admission, and the control-first formal-eval
recording semantics.  Run with:

    python -m unittest discover -s tests -p 'test_policy_rollout.py'
"""

from __future__ import annotations

import types
import unittest
from unittest import mock

import numpy as np

import dexmani_real.deployment.executor as executor_mod
from dexmani_real.deployment.config import validate_policy_runtime_compatibility
from dexmani_real.deployment.evaluation import EVALUATION_MAX_FRAMES_STOP_REASON
from dexmani_real.deployment.executor import (
    PolicyExecutor,
    _CommandProgress,
    _EvaluationEvidenceIssueKind,
    _PendingEvaluationTermination,
    _RejectKind,
    _rejection_ik_metadata,
    _validate_policy_arm_action,
    decode_policy_action,
)
from dexmani_real.deployment.metrics import PolicyStats
from dexmani_real.deployment.prediction import Prediction
from dexmani_real.deployment.timing import first_future_step_index

# xArm7 joint bounds (rad), matching the defaults documented in
# docs/action_clip_mechanisms.md §3.1.
ARM_LOWER = np.array(
    (-6.28318530718, -2.059, -6.28318530718, -0.19198,
     -6.28318530718, -1.69297, -6.28318530718)
)
ARM_UPPER = np.array(
    (6.28318530718, 2.0944, 6.28318530718, 3.927,
     6.28318530718, 3.14159265359, 6.28318530718)
)
_ARM_JUMP_RAD = float(np.deg2rad(20.0))


class _Field:
    def __init__(self, name, shape, dtype, semantics=None):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.semantics = semantics if semantics is not None else {}


def _fake_policy_spec(action_key="action", control_dt_s=1.0 / 16.0, **overrides):
    """A current PolicySpec with no temporal_ensemble_coeff field."""
    kwargs = dict(
        action_key=action_key,
        action_dim=21 if action_key == "action_ee" else 19,
        control_action_dim=21 if action_key == "action_ee" else 19,
        horizon=16,
        n_obs_steps=2,
        n_action_steps=16,
        observation_fields=(_Field("joint_state", (19,), "float32"),),
        control_dt_s=control_dt_s,
        requires_hand=True,
        rgb_preprocessing=None,
    )
    kwargs.update(overrides)
    return types.SimpleNamespace(**kwargs)


def _fake_runtime(**overrides):
    runtime = types.SimpleNamespace()
    runtime.policy = types.SimpleNamespace(
        control_hz=16.0,
        arm_max_action_jump_rad=_ARM_JUMP_RAD,
        hand_max_action_jump_rad=1.0,
        endpoint_delta_tolerance_rad=1e-12,
        action_validity_s=0.5,
    )
    runtime.arm = types.SimpleNamespace(
        joint_limit_lower=ARM_LOWER,
        joint_limit_upper=ARM_UPPER,
    )
    runtime.safety = types.SimpleNamespace(heartbeat_timeouts={"arm": 1.0, "hand": 1.0})
    runtime.environment = types.SimpleNamespace(
        table=types.SimpleNamespace(enabled=False, plane_abcd=None)
    )
    for key, value in overrides.items():
        setattr(runtime, key, value)
    return runtime


def _fake_arm_state(qpos=None):
    qpos = np.zeros(7) if qpos is None else np.asarray(qpos, dtype=np.float64)
    return {
        "connected": True,
        "error_code": 0,
        "state_valid": True,
        "source_monotonic_ns": 1,
        "qpos": qpos,
        "qvel": np.zeros(7),
    }


class TestPolicyContract(unittest.TestCase):
    def test_spec_without_temporal_field_is_accepted(self):
        # The external PolicySpec no longer declares temporal_ensemble_coeff.
        validate_policy_runtime_compatibility(_fake_policy_spec(), _fake_runtime())

    def test_requires_hand_still_enforced(self):
        spec = _fake_policy_spec(requires_hand=False)
        with self.assertRaises(ValueError):
            validate_policy_runtime_compatibility(spec, _fake_runtime())

    def test_control_dt_still_enforced(self):
        spec = _fake_policy_spec(control_dt_s=0.1)
        with self.assertRaises(ValueError):
            validate_policy_runtime_compatibility(spec, _fake_runtime())

    def test_control_action_dim_still_enforced(self):
        spec = _fake_policy_spec(control_action_dim=7)
        with self.assertRaises(ValueError):
            validate_policy_runtime_compatibility(spec, _fake_runtime())


class TestScheduling(unittest.TestCase):
    def test_partial_stale_skips_prefix(self):
        self.assertEqual(first_future_step_index(10, 1, 15, 10), 5)

    def test_whole_stale_discards(self):
        self.assertIsNone(first_future_step_index(10, 1, 200, 10))

    def test_future_grid_starts_at_zero(self):
        self.assertEqual(first_future_step_index(100, 1, 50, 10), 0)

    def test_source_age_no_longer_discards_executable_prediction(self):
        # Change B: a prediction whose source is arbitrarily old but whose
        # logical grid still has an executable step must not be dropped.
        step_dt_ns = int(round(1.0 / 16.0 * 1e9))
        now_ns = 10_000_000_000
        num_steps = 16
        prediction = Prediction(
            run_generation=0,
            source_monotonic_ns=1,  # far older than any source-age threshold
            logical_step_monotonic_ns=now_ns - step_dt_ns,
            actions=np.zeros((num_steps, 19)),
        )
        executor = PolicyExecutor.__new__(PolicyExecutor)
        executor.active_prediction = prediction
        executor.schedule_base_ns = prediction.logical_step_monotonic_ns
        executor.step_index = 0
        executor.sync_mode = False
        executor.step_dt_ns = step_dt_ns
        executor.next_command_due_ns = None
        executor.stats = PolicyStats()
        due = executor._next_due_action(now_ns)
        self.assertIsNotNone(due)
        self.assertFalse(hasattr(executor_mod, "_prediction_source_deadline_ns"))


class TestArmRejectOnly(unittest.TestCase):
    def test_within_jump_limit_is_canonical_unchanged(self):
        runtime = _fake_runtime()
        reference = np.zeros(7)
        target = reference + 0.1
        canonical, reason = _validate_policy_arm_action(target, reference, runtime)
        self.assertIsNone(reason)
        self.assertTrue(np.array_equal(canonical, target))

    def test_beyond_jump_limit_rejects_without_clipping(self):
        runtime = _fake_runtime()
        reference = np.zeros(7)
        target = reference + 0.5  # 0.5 rad > 20 deg
        canonical, reason = _validate_policy_arm_action(target, reference, runtime)
        self.assertIsNone(canonical)
        self.assertIsNotNone(reason)

    def test_joint_limit_violation_rejects(self):
        runtime = _fake_runtime()
        reference = np.zeros(7)
        target = reference.copy()
        target[3] = 5.0  # joint 3 upper bound is 3.927
        canonical, reason = _validate_policy_arm_action(target, reference, runtime)
        self.assertIsNone(canonical)
        self.assertIsNotNone(reason)


class TestRejectAttribution(unittest.TestCase):
    def _executor_for_decode(self, spec, planner=None):
        executor = PolicyExecutor.__new__(PolicyExecutor)
        executor.shared = mock.Mock()
        executor.runtime = _fake_runtime()
        executor.policy_spec = spec
        executor.stats = PolicyStats()
        executor.previous_arm_command_qpos = None
        executor.ee_planner = planner
        return executor

    def test_joint_admission_failure_is_safety(self):
        executor = self._executor_for_decode(_fake_policy_spec(action_key="action"))
        with mock.patch.object(
            executor_mod, "read_arm_state_dict", return_value=_fake_arm_state()
        ), mock.patch.object(executor_mod, "diagnose_arm_feedback", return_value=None):
            action = np.zeros(19)
            action[:7] = 0.5  # arm jump beyond 20 deg
            decoded, kind, reason = executor._decode_due_action(action)
        self.assertIsNone(decoded)
        self.assertIs(kind, _RejectKind.SAFETY)
        self.assertEqual(executor.stats.safety_rejection_count, 1)

    def test_ee_ik_failure_is_ik(self):
        spec = _fake_policy_spec(action_key="action_ee")
        planner = types.SimpleNamespace(
            solve_teleop_ik=lambda *a, **k: types.SimpleNamespace(
                success=False, qpos=None, reason="no solution"
            )
        )
        executor = self._executor_for_decode(spec, planner=planner)
        with mock.patch.object(
            executor_mod, "read_arm_state_dict", return_value=_fake_arm_state()
        ), mock.patch.object(executor_mod, "diagnose_arm_feedback", return_value=None):
            action = np.zeros(21)
            action[3:9] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)  # identity rot6d
            decoded, kind, reason = executor._decode_due_action(action)
        self.assertIsNone(decoded)
        self.assertIs(kind, _RejectKind.IK)
        self.assertEqual(executor.stats.ik_rejection_count, 1)

    def test_rejection_frame_maps_kind_to_frame_status(self):
        executor = PolicyExecutor.__new__(PolicyExecutor)
        executor.policy_spec = _fake_policy_spec(action_key="action")
        inputs = mock.Mock()
        inputs.state = mock.Mock(arm_qpos=np.zeros(7), hand_qpos=np.zeros(12))
        action = np.zeros(19)
        with mock.patch.object(executor, "_evaluation_hold_action", return_value=mock.Mock()), \
             mock.patch.object(executor, "_evaluation_raw_action_parts", return_value=(np.zeros(7), np.zeros(12))), \
             mock.patch.object(executor, "_record_evaluation_frame") as record_frame:
            record_frame.return_value = True
            executor._record_evaluation_rejection(
                inputs, action, ik_attempted=False, ik_ok=False, kind=_RejectKind.SAFETY
            )
            safety_signals = record_frame.call_args.kwargs["signals"]
            executor._record_evaluation_rejection(
                inputs, action, ik_attempted=True, ik_ok=False, kind=_RejectKind.IK
            )
            ik_signals = record_frame.call_args.kwargs["signals"]
        self.assertTrue(safety_signals["flag_safety_reject"])
        self.assertEqual(
            safety_signals["frame_status"], executor_mod._EVALUATION_FRAME_SAFETY_REJECT
        )
        self.assertFalse(ik_signals["flag_safety_reject"])
        self.assertEqual(
            ik_signals["frame_status"], executor_mod._EVALUATION_FRAME_IK_FAIL
        )


class TestEvaluationInvalidNotFault(unittest.TestCase):
    def test_request_invalid_finishes_without_fault(self):
        executor = PolicyExecutor.__new__(PolicyExecutor)
        executor.run_started_ns = 1
        executor.pending_evaluation_termination = None
        executor.progress = _CommandProgress()
        with mock.patch.object(executor, "_finish_evaluation_episode") as finish, \
             mock.patch.object(executor, "_fault") as fault:
            result = executor._request_evaluation_invalid(
                "missing evidence", stop_reason="eval:invalid:x", recorder_save=True
            )
        self.assertTrue(result)
        finish.assert_called_once_with(
            "missing evidence",
            stop_reason="eval:invalid:x",
            recorder_save=True,
            aborted=True,
        )
        fault.assert_not_called()

    def test_command_evidence_missing_invalidates_saving_prefix(self):
        executor = PolicyExecutor.__new__(PolicyExecutor)
        executor.recorder = mock.Mock()
        executor.stats = PolicyStats()
        executor.progress = _CommandProgress()
        candidate = mock.Mock(action_id=42)
        with mock.patch.object(
            executor, "_build_evaluation_frame_inputs",
            return_value=(None, "camera: missing evidence", _EvaluationEvidenceIssueKind.CAMERA_INVALID),
        ), mock.patch.object(executor, "_fault") as fault, \
             mock.patch.object(executor, "_request_evaluation_invalid") as req:
            executor._record_evaluation_command_evidence(np.zeros(19), candidate, anchor_ns=0)
        fault.assert_not_called()
        req.assert_called_once()
        self.assertTrue(req.call_args.kwargs["recorder_save"])
        self.assertEqual(req.call_args.kwargs["wait_for_action_id"], 42)

    def test_command_evidence_control_fault_raises_fault(self):
        executor = PolicyExecutor.__new__(PolicyExecutor)
        executor.recorder = mock.Mock()
        executor.stats = PolicyStats()
        executor.progress = _CommandProgress()
        candidate = mock.Mock(action_id=42)
        with mock.patch.object(
            executor, "_build_evaluation_frame_inputs",
            return_value=(None, "hardware: arm feedback unhealthy", _EvaluationEvidenceIssueKind.CONTROL_FAULT),
        ), mock.patch.object(executor, "_fault") as fault, \
             mock.patch.object(executor, "_request_evaluation_invalid") as req:
            executor._record_evaluation_command_evidence(np.zeros(19), candidate, anchor_ns=0)
        fault.assert_called_once()
        req.assert_not_called()

    def test_add_frame_failure_invalidates_without_save(self):
        executor = PolicyExecutor.__new__(PolicyExecutor)
        executor.recorder = mock.Mock()
        executor.stats = PolicyStats()
        executor.progress = _CommandProgress()
        candidate = mock.Mock(action_id=42)
        with mock.patch.object(
            executor, "_build_evaluation_frame_inputs",
            return_value=(mock.Mock(), "", None),
        ), mock.patch.object(executor, "_record_evaluation_command", return_value=False), \
             mock.patch.object(executor, "_request_evaluation_invalid") as req:
            executor._record_evaluation_command_evidence(np.zeros(19), candidate, anchor_ns=0)
        req.assert_called_once()
        self.assertFalse(req.call_args.kwargs["recorder_save"])

    def test_max_frames_is_invalid_not_failure(self):
        executor = PolicyExecutor.__new__(PolicyExecutor)
        executor.recorder = mock.Mock()
        executor.recorder.stop_pending = False
        executor.recorder.poll_stop.return_value = mock.Mock(
            phase=executor_mod.RecorderPhase.FINALIZING,
            reason=EVALUATION_MAX_FRAMES_STOP_REASON,
            done=False,
            error=None,
        )
        executor.run_started_ns = 1
        executor.pending_evaluation_termination = None
        executor.progress = _CommandProgress()
        with mock.patch.object(executor, "_finish_evaluation_episode") as finish:
            executor._poll_evaluation_recorder()
        finish.assert_called_once()
        self.assertEqual(finish.call_args.kwargs["stop_reason"], EVALUATION_MAX_FRAMES_STOP_REASON)


class TestInitialSampleGate(unittest.TestCase):
    def test_sync_requests_inference_only_after_initial_sample(self):
        executor = PolicyExecutor.__new__(PolicyExecutor)
        executor.recorder = mock.Mock()
        executor.stats = PolicyStats()
        executor.sync_mode = True
        executor.shared = mock.Mock()
        executor.evaluation_initial_sample_pending = True
        executor.evaluation_initial_deadline_ns = 10**18  # far future
        inputs = mock.Mock()
        with mock.patch.object(
            executor, "_build_evaluation_frame_inputs",
            return_value=(inputs, "", None),
        ), mock.patch.object(executor, "_evaluation_hold_action", return_value=mock.Mock()), \
             mock.patch.object(executor, "_record_evaluation_frame", return_value=True):
            ok = executor._record_initial_evaluation_sample(now_ns=1)
        self.assertTrue(ok)
        self.assertFalse(executor.evaluation_initial_sample_pending)
        executor.shared.inference_request.set.assert_called_once()

    def test_waiting_for_initial_sample_does_not_request_inference(self):
        executor = PolicyExecutor.__new__(PolicyExecutor)
        executor.recorder = mock.Mock()
        executor.stats = PolicyStats()
        executor.sync_mode = True
        executor.shared = mock.Mock()
        executor.evaluation_initial_sample_pending = True
        executor.evaluation_initial_deadline_ns = 10**18
        with mock.patch.object(
            executor, "_build_evaluation_frame_inputs",
            return_value=(None, "hardware: waiting", _EvaluationEvidenceIssueKind.RETRYABLE),
        ), mock.patch.object(executor, "_fault") as fault:
            ok = executor._record_initial_evaluation_sample(now_ns=1)
        self.assertFalse(ok)
        self.assertTrue(executor.evaluation_initial_sample_pending)
        fault.assert_not_called()
        executor.shared.inference_request.set.assert_not_called()

    def test_control_fault_during_initial_sample_faults(self):
        executor = PolicyExecutor.__new__(PolicyExecutor)
        executor.recorder = mock.Mock()
        executor.stats = PolicyStats()
        executor.evaluation_initial_sample_pending = True
        executor.evaluation_initial_deadline_ns = 10**18
        with mock.patch.object(
            executor, "_build_evaluation_frame_inputs",
            return_value=(None, "hardware: arm feedback unhealthy", _EvaluationEvidenceIssueKind.CONTROL_FAULT),
        ), mock.patch.object(executor, "_fault") as fault, \
             mock.patch.object(executor, "_request_evaluation_invalid") as req:
            executor._record_initial_evaluation_sample(now_ns=1)
        fault.assert_called_once()
        req.assert_not_called()

    def test_eval_invalid_during_initial_sample_discards_without_fault(self):
        executor = PolicyExecutor.__new__(PolicyExecutor)
        executor.recorder = mock.Mock()
        executor.stats = PolicyStats()
        executor.evaluation_initial_sample_pending = True
        executor.evaluation_initial_deadline_ns = 10**18
        with mock.patch.object(
            executor, "_build_evaluation_frame_inputs",
            return_value=(None, "camera: health=UNKNOWN", _EvaluationEvidenceIssueKind.CAMERA_INVALID),
        ), mock.patch.object(executor, "_fault") as fault, \
             mock.patch.object(executor, "_request_evaluation_invalid") as req:
            executor._record_initial_evaluation_sample(now_ns=1)
        fault.assert_not_called()
        req.assert_called_once()
        self.assertFalse(req.call_args.kwargs["recorder_save"])

    def test_initial_evidence_timeout_is_invalid_not_failure(self):
        executor = PolicyExecutor.__new__(PolicyExecutor)
        executor.shared = mock.Mock()
        executor.run_started_ns = 0
        executor.run_generation = 7
        executor.evaluation_initial_sample_pending = True
        executor.max_running_ns = 1000
        snapshot = mock.Mock(generation=7)
        with mock.patch.object(executor_mod, "read_run_state_snapshot", return_value=snapshot), \
             mock.patch.object(executor, "_finish_evaluation_episode") as finish:
            executor._run_active_tick(now_ns=2000)
        finish.assert_called_once()
        self.assertEqual(
            finish.call_args.kwargs["stop_reason"], "eval:invalid:initial_evidence_timeout"
        )
        self.assertFalse(finish.call_args.kwargs["recorder_save"])


class TestPolicyStatsEvidenceTiming(unittest.TestCase):
    def test_empty_stats_have_no_evaluation_keys(self):
        snapshot = PolicyStats().snapshot()
        self.assertNotIn("evaluation_state_build_ms", snapshot)
        self.assertNotIn("evaluation_record_ms", snapshot)
        self.assertNotIn("evaluation_state_build_max_ms", snapshot)
        self.assertNotIn("evaluation_record_max_ms", snapshot)

    def test_observe_sets_latest_and_window_max(self):
        stats = PolicyStats()
        for value in (1.0, 3.0, 2.0):
            stats.observe_evaluation_state_build_ms(value)
        stats.observe_evaluation_record_ms(0.5)
        snapshot = stats.snapshot()
        self.assertEqual(snapshot["evaluation_state_build_ms"], 2.0)  # latest
        self.assertEqual(snapshot["evaluation_state_build_max_ms"], 3.0)  # max
        self.assertEqual(snapshot["evaluation_record_ms"], 0.5)
        self.assertEqual(snapshot["evaluation_record_max_ms"], 0.5)

    def test_non_finite_sample_is_rejected(self):
        stats = PolicyStats()
        with self.assertRaises(ValueError):
            stats.observe_evaluation_record_ms(float("nan"))

    def test_deques_are_bounded(self):
        stats = PolicyStats()
        for i in range(512):
            stats.observe_evaluation_state_build_ms(float(i))
        self.assertEqual(len(stats.evaluation_state_build_ms), 256)


class TestPreparedCommandUnavailableParity(unittest.TestCase):
    def _assert_unavailable_is_noop(self, recorder):
        executor = PolicyExecutor.__new__(PolicyExecutor)
        executor.recorder = recorder
        prepared = mock.Mock(unavailable=True, fatal=False)
        with mock.patch.object(executor, "_fault") as fault, \
             mock.patch.object(executor, "_reject_due_step") as reject, \
             mock.patch.object(executor, "_record_evaluation_rejection_evidence") as evidence:
            executor._handle_preparation_rejection(
                prepared, mock.Mock(), 0, raw_action=np.zeros(19)
            )
        fault.assert_not_called()
        reject.assert_not_called()
        evidence.assert_not_called()

    def test_unavailable_without_recorder_is_noop(self):
        self._assert_unavailable_is_noop(None)

    def test_unavailable_with_recorder_is_noop(self):
        self._assert_unavailable_is_noop(mock.Mock())


class TestRejectionIKMetadata(unittest.TestCase):
    def test_joint_safety_reject_never_attempts_ik(self):
        self.assertEqual(_rejection_ik_metadata(False, _RejectKind.SAFETY), (False, False))

    def test_ee_ik_failure_marks_attempted_not_ok(self):
        self.assertEqual(_rejection_ik_metadata(True, _RejectKind.IK), (True, False))

    def test_ee_safety_after_ik_marks_ok(self):
        self.assertEqual(_rejection_ik_metadata(True, _RejectKind.SAFETY), (True, True))


class TestPendingEvaluationTermination(unittest.TestCase):
    def _outstanding_executor(self):
        executor = PolicyExecutor.__new__(PolicyExecutor)
        executor.run_started_ns = 1
        executor.pending_evaluation_termination = None
        executor.progress = _CommandProgress()
        executor.progress.generation = 0
        executor.progress.latest_published_action_id = 42
        executor.progress.arm_accepted_action_id = 41
        executor.progress.hand_accepted_action_id = 41
        return executor

    def test_request_latches_when_not_covered(self):
        executor = self._outstanding_executor()
        with mock.patch.object(executor, "_finish_evaluation_episode") as finish:
            result = executor._request_evaluation_invalid(
                "ev", stop_reason="eval:invalid:x", recorder_save=True, wait_for_action_id=42
            )
        self.assertFalse(result)
        self.assertIsNotNone(executor.pending_evaluation_termination)
        self.assertEqual(executor.pending_evaluation_termination.wait_for_action_id, 42)
        finish.assert_not_called()

    def test_drain_finishes_once_when_covered(self):
        executor = self._outstanding_executor()
        executor.pending_evaluation_termination = _PendingEvaluationTermination(
            "ev", "eval:invalid:x", True, 42
        )
        executor.progress.arm_accepted_action_id = 42
        executor.progress.hand_accepted_action_id = 42
        with mock.patch.object(executor, "_finish_evaluation_episode") as finish:
            drained = executor._drain_pending_evaluation_termination()
        self.assertTrue(drained)
        finish.assert_called_once()
        self.assertEqual(finish.call_args.kwargs["stop_reason"], "eval:invalid:x")

    def test_merge_recorder_integrity_wins_and_keeps_wait(self):
        executor = self._outstanding_executor()
        executor.pending_evaluation_termination = _PendingEvaluationTermination(
            "ev1", "eval:invalid:a", True, 42
        )
        with mock.patch.object(executor, "_finish_evaluation_episode") as finish:
            result = executor._request_evaluation_invalid(
                "ev2", stop_reason="eval:invalid:recorder_fault",
                recorder_save=False, wait_for_action_id=42,
            )
        self.assertFalse(result)
        pending = executor.pending_evaluation_termination
        self.assertFalse(pending.recorder_save)
        self.assertEqual(pending.stop_reason, "eval:invalid:recorder_fault")
        self.assertEqual(pending.wait_for_action_id, 42)
        finish.assert_not_called()

    def test_sticky_save_never_upgrades(self):
        executor = PolicyExecutor.__new__(PolicyExecutor)
        executor.pending_evaluation_termination = _PendingEvaluationTermination(
            "ev", "eval:invalid:x", False, 42
        )
        self.assertFalse(executor._sticky_recorder_save(True))
        executor.pending_evaluation_termination = None
        self.assertTrue(executor._sticky_recorder_save(True))

    def test_operator_stop_preserves_automatic_invalid(self):
        executor = PolicyExecutor.__new__(PolicyExecutor)
        executor.pending_evaluation_termination = _PendingEvaluationTermination(
            "ev", "eval:invalid:recorder_fault", True, 42
        )
        with mock.patch.object(executor, "_finish_episode") as finish_ep:
            executor._finish_evaluation_episode(
                "operator stop", stop_reason="eval:success:operator", recorder_save=True
            )
        self.assertEqual(
            finish_ep.call_args.kwargs["evaluation_stop_reason"], "eval:invalid:recorder_fault"
        )

    def test_sticky_save_false_across_hardware_termination(self):
        executor = PolicyExecutor.__new__(PolicyExecutor)
        executor.pending_evaluation_termination = _PendingEvaluationTermination(
            "ev", "eval:invalid:recorder_fault", False, 42
        )
        with mock.patch.object(executor, "_finish_episode") as finish_ep:
            executor._finish_evaluation_episode(
                "hardware fault", stop_reason="eval:invalid:hardware_fault", recorder_save=True
            )
        self.assertFalse(finish_ep.call_args.kwargs["recorder_save"])
        self.assertEqual(
            finish_ep.call_args.kwargs["evaluation_stop_reason"], "eval:invalid:hardware_fault"
        )


class TestCommandProgressDrain(unittest.TestCase):
    def test_hand_intermediate_progress_keeps_watchdog_alive(self):
        progress = _CommandProgress()
        progress.generation = 0
        progress.latest_published_action_id = 42
        progress.arm_accepted_action_id = 42
        progress.hand_accepted_action_id = 41
        progress.arm_last_progress_ns = 1000
        progress.hand_last_progress_ns = 1000
        progress.hand_last_sdk_setpoint_accepted_ns = 1000

        reason = progress.observe(
            generation=0, arm_action_id=42, hand_action_id=41,
            hand_setpoint_accepted_ns=2900, now_ns=3000, timeout_ns=500,
        )
        self.assertIsNone(reason)
        self.assertEqual(progress.hand_last_progress_ns, 2900)

    def test_stalled_hand_progress_times_out(self):
        progress = _CommandProgress()
        progress.generation = 0
        progress.latest_published_action_id = 42
        progress.arm_accepted_action_id = 42
        progress.hand_accepted_action_id = 41
        progress.arm_last_progress_ns = 1000
        progress.hand_last_progress_ns = 2900
        progress.hand_last_sdk_setpoint_accepted_ns = 2900

        reason = progress.observe(
            generation=0, arm_action_id=42, hand_action_id=41,
            hand_setpoint_accepted_ns=2900, now_ns=4000, timeout_ns=500,
        )
        self.assertEqual(reason, "hand worker command progress timeout")


class TestPendingInvalidPrecedence(unittest.TestCase):
    def _executor(self, arm_accepted, hand_accepted):
        executor = PolicyExecutor.__new__(PolicyExecutor)
        executor.shared = mock.Mock()
        executor.run_started_ns = 0
        executor.run_generation = 7
        executor.evaluation_initial_sample_pending = False
        executor.pending_evaluation_termination = _PendingEvaluationTermination(
            "ev", "eval:invalid:camera_evidence", True, 42
        )
        executor.pending_truncation_action_id = 42
        executor.progress = _CommandProgress()
        executor.progress.generation = 0
        executor.progress.arm_accepted_action_id = arm_accepted
        executor.progress.hand_accepted_action_id = hand_accepted
        executor.progress.latest_published_action_id = 42
        return executor

    def test_pending_invalid_beats_pending_truncation(self):
        executor = self._executor(42, 42)
        snapshot = mock.Mock(generation=7)
        with mock.patch.object(executor_mod, "read_run_state_snapshot", return_value=snapshot), \
             mock.patch.object(executor, "_observe_worker_progress", return_value=True), \
             mock.patch.object(executor, "_finish_evaluation_episode") as finish:
            executor._run_active_tick(now_ns=100)
        finish.assert_called_once()
        self.assertEqual(finish.call_args.kwargs["stop_reason"], "eval:invalid:camera_evidence")

    def test_pending_returns_before_ingest(self):
        executor = self._executor(41, 41)
        snapshot = mock.Mock(generation=7)
        with mock.patch.object(executor_mod, "read_run_state_snapshot", return_value=snapshot), \
             mock.patch.object(executor, "_observe_worker_progress", return_value=True), \
             mock.patch.object(executor, "_ingest_latest_prediction") as ingest:
            executor._run_active_tick(now_ns=100)
        ingest.assert_not_called()


class TestTerminalRejectedStepOrdering(unittest.TestCase):
    def _rejection_executor(self):
        executor = PolicyExecutor.__new__(PolicyExecutor)
        executor.shared = mock.Mock()
        executor.shared.error_state.value = False
        executor.policy_spec = _fake_policy_spec(action_key="action_ee")
        executor.deployment = types.SimpleNamespace(max_action_steps=1)
        executor.episode_steps = 0
        executor.active_prediction = mock.Mock()
        executor.stats = PolicyStats()
        executor.run_started_ns = 1
        executor.pending_evaluation_termination = None
        executor.recorder = mock.Mock()
        executor.previous_arm_command_qpos = None
        executor.next_command_due_ns = None
        executor.step_dt_ns = int(round(1e9 / 16))
        return executor

    def test_rejection_evidence_before_step_limit_finish(self):
        executor = self._rejection_executor()
        calls = []
        with mock.patch.object(
            executor, "_decode_due_action",
            return_value=(None, _RejectKind.SAFETY, "jump"),
        ), mock.patch.object(executor, "_record_evaluation_rejection_evidence") as evidence, \
             mock.patch.object(executor, "_finish_action_step_limit") as finish_limit, \
             mock.patch.object(executor, "_fault") as fault:
            evidence.side_effect = lambda *a, **k: calls.append("evidence")
            finish_limit.side_effect = lambda: calls.append("finish")
            executor._publish_due_action(np.zeros(21), scheduled_target_ns=0, due_ns=0)
        self.assertEqual(calls, ["evidence", "finish"])
        fault.assert_not_called()

    def test_pending_invalid_skips_step_limit_finish(self):
        executor = self._rejection_executor()

        def latch_pending(*args, **kwargs):
            executor.pending_evaluation_termination = _PendingEvaluationTermination(
                "ev", "eval:invalid:camera_evidence", True, 42
            )

        with mock.patch.object(
            executor, "_decode_due_action",
            return_value=(None, _RejectKind.SAFETY, "jump"),
        ), mock.patch.object(
            executor, "_record_evaluation_rejection_evidence", side_effect=latch_pending
        ), mock.patch.object(executor, "_finish_action_step_limit") as finish_limit:
            executor._publish_due_action(np.zeros(21), scheduled_target_ns=0, due_ns=0)
        finish_limit.assert_not_called()


class TestRecorderPollPendingBoundary(unittest.TestCase):
    def test_recorder_poll_latches_pending_but_continues_boundary(self):
        executor = PolicyExecutor.__new__(PolicyExecutor)
        executor.recorder = mock.Mock()
        executor.recorder.stop_pending = False
        executor.recorder.poll_stop.return_value = mock.Mock(
            phase=executor_mod.RecorderPhase.ERROR, done=False, error="boom", reason=None,
        )
        executor.run_started_ns = 1
        executor.pending_evaluation_termination = None
        executor.progress = _CommandProgress()
        executor.progress.generation = 0
        executor.progress.latest_published_action_id = 42
        executor.progress.arm_accepted_action_id = 41
        executor.progress.hand_accepted_action_id = 41
        with mock.patch.object(executor, "_finish_evaluation_episode") as finish:
            result = executor._poll_evaluation_recorder()
        self.assertTrue(result)
        self.assertIsNotNone(executor.pending_evaluation_termination)
        finish.assert_not_called()


class TestEventAnchor(unittest.TestCase):
    def test_command_evidence_uses_given_anchor(self):
        executor = PolicyExecutor.__new__(PolicyExecutor)
        executor.recorder = mock.Mock()
        executor.stats = PolicyStats()
        executor.progress = _CommandProgress()
        candidate = mock.Mock(action_id=42)
        with mock.patch.object(
            executor, "_build_evaluation_frame_inputs", return_value=(mock.Mock(), "", None)
        ) as build, mock.patch.object(
            executor, "_record_evaluation_command", return_value=True
        ):
            executor._record_evaluation_command_evidence(
                np.zeros(19), candidate, anchor_ns=12345
            )
        build.assert_called_once_with(12345)

    def test_rejection_evidence_uses_given_anchor(self):
        executor = PolicyExecutor.__new__(PolicyExecutor)
        executor.recorder = mock.Mock()
        executor.stats = PolicyStats()
        executor.progress = _CommandProgress()
        with mock.patch.object(
            executor, "_build_evaluation_frame_inputs", return_value=(mock.Mock(), "", None)
        ) as build, mock.patch.object(
            executor, "_record_evaluation_rejection", return_value=True
        ):
            executor._record_evaluation_rejection_evidence(
                np.zeros(19), kind=_RejectKind.IK, ik_attempted=True, ik_ok=False, anchor_ns=999
            )
        build.assert_called_once_with(999)


if __name__ == "__main__":
    unittest.main()
