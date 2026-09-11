"""Deterministic offline regressions for the learned-policy rollout semantics.

No hardware, no trained checkpoint, no GPU.  These tests pin the public Policy
contract boundary, timestamp-based scheduling, the explicit IK-vs-SAFETY
attribution, reject-only arm admission, and the publish-then-record rollout
semantics.  Run with:

    python -m unittest discover -s tests -p 'test_policy_rollout.py'
"""

from __future__ import annotations

import threading
import types
import unittest
from dataclasses import replace
from unittest import mock

import numpy as np

import dexmani_real.deployment.executor as executor_mod
from dexmani_real.deployment.config import (
    RolloutRecordingConfig,
    validate_policy_runtime_compatibility,
)
from dexmani_real.deployment.executor import (
    PolicyExecutor,
    _CommandProgress,
    _RejectKind,
    _validate_policy_arm_action,
    decode_policy_action,
)
from dexmani_real.deployment.inference.worker import serialize_prediction
from dexmani_real.deployment.metrics import PolicyStats
from dexmani_real.deployment.prediction import Prediction
from dexmani_real.deployment.timing import first_future_step_index
from dexmani_real.recording.client import RecorderStopResult
from dexmani_real.runtime.safety import SafetyState, StopRequest, request_policy_stop

# xArm7 joint bounds (rad), matching the defaults documented in
# docs/action_clip_mechanisms.md §3.1.
ARM_LOWER = np.array(
    (
        -6.28318530718,
        -2.059,
        -6.28318530718,
        -0.19198,
        -6.28318530718,
        -1.69297,
        -6.28318530718,
    )
)
ARM_UPPER = np.array(
    (
        6.28318530718,
        2.0944,
        6.28318530718,
        3.927,
        6.28318530718,
        3.14159265359,
        6.28318530718,
    )
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
        n_action_steps=8,
        chunk_size=15,
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
    def test_periodic_deadline_does_not_accumulate_inference_latency(self):
        from dexmani_real.deployment.timing import next_periodic_deadline_ns

        start = 1_000_000_000
        period = 8 * 62_500_000
        deadline = start
        for cycle in range(4):
            finished = deadline + 200_000_000
            deadline = next_periodic_deadline_ns(deadline, period, finished)
            self.assertEqual(deadline, start + (cycle + 1) * period)
        # A slow prediction skips missed query slots instead of catching up.
        self.assertEqual(
            next_periodic_deadline_ns(start, period, start + 2 * period + 1),
            start + 3 * period,
        )

    def test_partial_stale_skips_prefix(self):
        self.assertEqual(first_future_step_index(10, 1, 15, 10), 6)

    def test_equal_start_is_stale(self):
        self.assertEqual(first_future_step_index(10, 2, 10, 3), 1)
        self.assertIsNone(first_future_step_index(10, 2, 14, 3))

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
            inference_latency_ms=12.5,
            observation_age_ms=3.25,
            observation_skew_ms=0.125,
        )
        executor = PolicyExecutor.__new__(PolicyExecutor)
        executor.active_prediction = prediction
        executor.schedule_base_ns = prediction.logical_step_monotonic_ns
        executor.step_index = 0
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
        with (
            mock.patch.object(
                executor_mod, "read_arm_state_dict", return_value=_fake_arm_state()
            ),
            mock.patch.object(executor_mod, "diagnose_arm_feedback", return_value=None),
        ):
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
        with (
            mock.patch.object(
                executor_mod, "read_arm_state_dict", return_value=_fake_arm_state()
            ),
            mock.patch.object(executor_mod, "diagnose_arm_feedback", return_value=None),
        ):
            action = np.zeros(21)
            action[3:9] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)  # identity rot6d
            decoded, kind, reason = executor._decode_due_action(action)
        self.assertIsNone(decoded)
        self.assertIs(kind, _RejectKind.IK)
        self.assertEqual(executor.stats.ik_rejection_count, 1)

    def test_rejection_frame_maps_kind_to_frame_status(self):
        executor = PolicyExecutor.__new__(PolicyExecutor)
        executor.policy_spec = _fake_policy_spec(action_key="action")
        inputs = [mock.Mock(), {}, {}]
        with (
            mock.patch.object(
                executor, "_recorded_hold_action", return_value=mock.Mock()
            ),
            mock.patch.object(executor, "_record_frame") as record_frame,
        ):
            record_frame.return_value = True
            executor._record_rejection(inputs, kind=_RejectKind.SAFETY)
            safety_signals = record_frame.call_args.kwargs["signals"]
            executor._record_rejection(inputs, kind=_RejectKind.IK)
            ik_signals = record_frame.call_args.kwargs["signals"]
        self.assertFalse(safety_signals["action_queued"])
        self.assertEqual(
            safety_signals["frame_status"], executor_mod._RECORD_FRAME_SAFETY_REJECT
        )
        self.assertFalse(ik_signals["action_queued"])
        self.assertEqual(ik_signals["frame_status"], executor_mod._RECORD_FRAME_IK_FAIL)


class TestPreparedCommandUnavailableParity(unittest.TestCase):
    def _assert_unavailable_is_noop(self, recorder):
        executor = PolicyExecutor.__new__(PolicyExecutor)
        executor.recorder = recorder
        prepared = mock.Mock(unavailable=True, fatal=False)
        with (
            mock.patch.object(executor, "_fault") as fault,
            mock.patch.object(executor, "_reject_due_step") as reject,
            mock.patch.object(executor, "_record_rollout_tick") as evidence,
        ):
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
            generation=0,
            arm_action_id=42,
            hand_action_id=41,
            hand_setpoint_accepted_ns=2900,
            now_ns=3000,
            timeout_ns=500,
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
            generation=0,
            arm_action_id=42,
            hand_action_id=41,
            hand_setpoint_accepted_ns=2900,
            now_ns=4000,
            timeout_ns=500,
        )
        self.assertEqual(reason, "hand worker command progress timeout")


class TestEvalSeed(unittest.TestCase):
    def test_nonnegative_integer_seed(self):
        from dexmani_real.deployment.config import InferenceWorkerConfig

        for seed in (0, 1):
            self.assertEqual(
                InferenceWorkerConfig(
                    "dp/task/exp", "cpu", None, seed, "ckpt.pt", 10
                ).seed,
                seed,
            )
        for seed in (-1, True, 1.0):
            with self.assertRaises(ValueError):
                InferenceWorkerConfig(
                    "dp/task/exp", "cpu", None, seed, "ckpt.pt", 10
                )

    def test_artifact_and_inference_steps_validation(self):
        from dexmani_real.deployment.config import InferenceWorkerConfig

        base = ("dp/task/exp", "cpu", None, 0)
        for artifact in ("ckpt.pt", "epoch_500-deployment.pt"):
            InferenceWorkerConfig(*base, artifact, 10)
        for artifact in ("", "../evil.pt", "/abs/x.pt", "sub/x.pt"):
            with self.assertRaises(ValueError):
                InferenceWorkerConfig(*base, artifact, 10)
        for steps in (0, -1, True, 1.0):
            with self.assertRaises(ValueError):
                InferenceWorkerConfig(*base, "ckpt.pt", steps)

    def test_cli_seed(self):
        from examples.run_policy import _parser

        parser = _parser()
        for command in ("run", "shadow"):
            self.assertEqual(parser.parse_args([command, "dp/task/exp"]).eval_seed, 0)
        required = [
            "eval",
            "dp/task/exp",
            "--max-duration",
            "60",
            "--task-label",
            "task",
            "--operator",
            "me",
        ]
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            parser.parse_args(required)
        self.assertEqual(
            parser.parse_args(required + ["--eval-seed", "1"]).eval_seed, 1
        )

    def test_removed_cli_options_are_rejected(self):
        from examples.run_policy import _parser

        for option in (
            "--inference-mode",
            "--runtime-config",
            "--max-action-steps",
            "--output-dir",
        ):
            with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
                _parser().parse_args(["run", "dp/task/exp", option, "1"])
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            _parser().parse_args(["shadow", "dp/task/exp", "--max-duration", "60"])

    def test_loader_passes_seed_and_reset(self):
        from dexmani_real.deployment.config import InferenceWorkerConfig
        from dexmani_real.deployment.inference.worker import _load_inference_runtime
        import dexmani_policy.deployment as policy_api

        spec = _fake_policy_spec()
        loaded = mock.Mock()
        loaded.spec = spec
        with mock.patch.object(
            policy_api, "load_experiment", return_value=loaded
        ) as load:
            runtime = _load_inference_runtime(
                InferenceWorkerConfig("dp/task/exp", "cpu", spec, 1, "ckpt.pt", 7)
            )
            runtime.reset_episode()
        load.assert_called_once_with(
            "dp/task/exp",
            device="cpu",
            seed=1,
            artifact="ckpt.pt",
            inference_steps=7,
        )
        loaded.reset_episode.assert_called_once_with()


class TestFutureChunkBoundary(unittest.TestCase):
    def test_check_smokes_production_api_and_rejects_bad_chunks(self):
        from examples.run_policy import _check_action_chunk

        spec = _fake_policy_spec()
        policy = mock.Mock()
        policy.predict_action_chunk.return_value = np.zeros((15, 19), np.float64)
        _check_action_chunk(policy, spec)
        policy.predict.assert_not_called()
        policy.reset_episode.assert_called_once_with()
        observation = policy.predict_action_chunk.call_args.args[0]
        self.assertEqual(observation["joint_state"].shape, (2, 19))
        for actions in (
            np.zeros((8, 19)),
            np.zeros((15, 19), np.float32),
            np.full((15, 19), np.nan),
        ):
            policy.predict_action_chunk.return_value = actions
            with self.assertRaises(RuntimeError):
                _check_action_chunk(policy, spec)

    def test_adapter_and_prediction_transport_keep_full_chunk(self):
        from dexmani_real.deployment.inference.dexmani_policy import (
            DexManiPolicyAdapter,
        )
        from dexmani_real.deployment.inference.worker import serialize_prediction
        from dexmani_real.deployment.executor import prediction_from_record

        spec = _fake_policy_spec()
        chunk = np.arange(15 * 19, dtype=np.float64).reshape(15, 19)
        loaded = mock.Mock()
        loaded.spec = spec
        loaded.predict_action_chunk.return_value = chunk
        observation = types.SimpleNamespace(
            arrays={"joint_state": np.zeros((2, 19), np.float32)}
        )
        actions = DexManiPolicyAdapter(loaded, spec).predict_action_chunk(observation)
        loaded.predict_action_chunk.assert_called_once_with(observation.arrays)
        loaded.predict.assert_not_called()
        prediction = Prediction(1, 10, 20, actions, 12.5, 3.25, 0.125)
        frame = serialize_prediction(prediction)
        chunk[:] = -1
        restored = prediction_from_record(frame[0])
        self.assertEqual(restored.num_steps, spec.chunk_size)
        self.assertEqual(restored.actions.dtype, np.float64)
        self.assertEqual(restored.actions[-1, -1], 284)

    def test_full_chunk_capacity_is_checked(self):
        from dexmani_real.ipc.schema import MAX_PREDICTION_STEPS

        with self.assertRaisesRegex(
            ValueError, "Policy chunk_size exceeds Real IPC capacity"
        ):
            validate_policy_runtime_compatibility(
                _fake_policy_spec(chunk_size=MAX_PREDICTION_STEPS + 1), _fake_runtime()
            )

    def test_prediction_ipc_rejects_malformed_dimensions(self):
        from dexmani_real.ipc.schema import MAX_PREDICTION_STEPS

        frame = serialize_prediction(
            Prediction(1, 10, 20, np.zeros((15, 19)), 12.5, 3.25, 0.125)
        )
        for field, value in (
            ("num_steps", 0),
            ("num_steps", MAX_PREDICTION_STEPS + 1),
            ("action_dim", 0),
            ("action_dim", 22),
        ):
            with self.subTest(field=field, value=value):
                malformed = frame.copy()
                malformed[field][0] = value
                with self.assertRaises(ValueError):
                    executor_mod.prediction_from_record(malformed[0])


class TestPredictionTiming(unittest.TestCase):
    def test_inference_attaches_exact_observation_and_duration(self):
        import dexmani_real.deployment.inference.worker as worker
        from dexmani_real.config.defaults import PolicyParams
        from dexmani_real.deployment.config import InferenceWorkerConfig

        shared = mock.Mock()
        shared.is_running.value = True
        shared.run_generation.value = 2
        shared.error_state.value = False
        shared.estop_request.value = False
        clock_ns = 2_000_000_000
        observation = types.SimpleNamespace(
            latest_source_monotonic_ns=1_995_000_000,
            logical_step_monotonic_ns=clock_ns,
            anchor_monotonic_ns=clock_ns,
            arm_history=types.SimpleNamespace(
                valid_mask=np.array([1]),
                source_monotonic_ns=np.array([1_994_000_000]),
            ),
            hand_history=types.SimpleNamespace(
                valid_mask=np.array([1]),
                source_monotonic_ns=np.array([1_995_000_000]),
            ),
        )
        actions = np.arange(15 * 19, dtype=np.float64).reshape(15, 19)
        runtime = mock.Mock()
        runtime.warmup.return_value = [0.0] * 5
        policy_observation = object()

        def predict(value):
            nonlocal clock_ns
            self.assertIs(value, policy_observation)
            clock_ns += 12_500_000
            shared.is_running.value = False
            return actions

        runtime.predict_action_chunk.side_effect = predict
        with (
            mock.patch.object(worker, "_load_inference_runtime", return_value=runtime),
            mock.patch.object(worker, "build_fingertip_runtime", return_value=None),
            mock.patch.object(
                worker,
                "read_run_state_snapshot",
                return_value=types.SimpleNamespace(
                    generation=2,
                    state=SafetyState.RUNNING,
                    started_monotonic_ns=1_000_000_000,
                ),
            ),
            mock.patch.object(worker, "_build_observation", return_value=observation),
            mock.patch.object(
                worker, "_to_policy_observation", return_value=policy_observation
            ),
            mock.patch.object(
                worker.time, "monotonic_ns", side_effect=lambda: clock_ns
            ),
        ):
            worker.inference_loop(
                shared,
                PolicyParams(),
                InferenceWorkerConfig("fake", "cpu", _fake_policy_spec(), 0, "ckpt.pt", 10),
            )
        shared.prediction_ring.write.assert_called_once()
        prediction = executor_mod.prediction_from_record(
            shared.prediction_ring.write.call_args.args[0][0]
        )
        self.assertEqual(prediction.inference_latency_ms, 12.5)
        self.assertEqual(prediction.observation_age_ms, 5.0)
        self.assertEqual(prediction.observation_skew_ms, 1.0)
        self.assertEqual(
            prediction.source_monotonic_ns, observation.latest_source_monotonic_ns
        )
        self.assertEqual(
            prediction.logical_step_monotonic_ns, observation.logical_step_monotonic_ns
        )
        np.testing.assert_array_equal(prediction.actions, actions)
        runtime.close.assert_called_once()

    def test_timing_round_trip(self):
        prediction = Prediction(
            1, 10, 20, np.zeros((15, 19)), 12.123456789012345, np.float64(3.2), 0
        )
        restored = executor_mod.prediction_from_record(
            serialize_prediction(prediction)[0]
        )
        for name in (
            "inference_latency_ms",
            "observation_age_ms",
            "observation_skew_ms",
        ):
            self.assertEqual(getattr(restored, name), getattr(prediction, name))

    def test_stats_filter_bad_diagnostics_and_preserve_counters(self):
        stats = PolicyStats(
            inference_latency_ms=float("nan"),
            observation_age_ms=-1.0,
            observation_skew_ms=1.25,
            schedule_lateness_ms=float("inf"),
            publication_interval_ms=0.0,
            skipped_prefix_steps=-1,
            safety_rejection_count=2,
            command_progress_timeout_count=1,
            ik_rejection_count=3,
        )
        metrics = stats.snapshot()
        self.assertNotIn("inference_latency_ms", metrics)
        self.assertNotIn("observation_age_ms", metrics)
        self.assertNotIn("schedule_lateness_ms", metrics)
        self.assertNotIn("skipped_prefix_steps", metrics)
        self.assertEqual(metrics["observation_skew_ms"], 1.25)
        self.assertEqual(metrics["publication_interval_ms"], 0.0)
        self.assertEqual(metrics["safety_rejection_count"], 2)
        self.assertEqual(metrics["command_progress_timeout_count"], 1)
        self.assertEqual(metrics["ik_rejection_count"], 3)


class TestFutureTail(unittest.TestCase):
    def executor(self):
        executor = PolicyExecutor.__new__(PolicyExecutor)
        executor.shared = mock.Mock()
        executor.policy_spec = _fake_policy_spec()
        executor.run_generation = 1
        executor.last_seen_prediction_sequence = None
        executor.active_prediction = None
        executor.step_dt_ns = 62_500_000
        executor.next_command_due_ns = None
        executor.stats = PolicyStats()
        return executor

    def test_timing_ignores_repeated_reads_and_old_generations(self):
        executor = self.executor()
        prediction = Prediction(1, 1, 2, np.zeros((15, 19)), 12.5, 3.25, 0.125)
        ring = executor.shared.prediction_ring
        ring.read_latest.return_value = (serialize_prediction(prediction), 1, 1)
        executor._ingest_latest_prediction(1_000_000_000)
        self.assertIsNone(executor.active_prediction)
        self.assertEqual(executor.stats.snapshot()["inference_latency_ms"], 12.5)
        changed = replace(prediction, inference_latency_ms=99.0)
        # A repeated sequence must not overwrite the latest sample.
        ring.read_latest.return_value = (serialize_prediction(changed), 1, 1)
        executor._ingest_latest_prediction(1_000_000_000)
        ring.read_latest.return_value = (
            serialize_prediction(replace(changed, run_generation=0)),
            1,
            2,
        )
        executor._ingest_latest_prediction(1_000_000_000)
        self.assertEqual(executor.stats.snapshot()["inference_latency_ms"], 12.5)

    def test_old_tail_survives_next_inference_until_new_plan(self):
        executor = self.executor()
        start = 1_000_000_000
        chunk = np.repeat(np.arange(15, dtype=np.float64)[:, None], 19, axis=1)
        old = Prediction(1, start, start, chunk, 12.5, 3.25, 0.125)
        with mock.patch.object(
            executor_mod, "read_latest_prediction", return_value=(old, 1)
        ):
            executor._ingest_latest_prediction(start - 1)
            # Consume A normally, including its tail while B is still pending.
            executor.episode_steps = 0
            for index in range(15):
                now = start + index * executor.step_dt_ns - 1
                executor._ingest_latest_prediction(now)
                due = executor._next_due_action(now)
                self.assertIsNotNone(due)
                self.assertEqual(due[0][0], index)
                executor._consume_control_slot(due[2], now)
                with mock.patch.object(
                    executor_mod.time, "monotonic_ns", return_value=now
                ):
                    executor._commit_terminal_step()
            self.assertIsNone(executor.active_prediction)
        new = Prediction(
            1, start, start + 8 * executor.step_dt_ns, chunk + 100, 12.5, 3.25, 0.125
        )
        with mock.patch.object(
            executor_mod, "read_latest_prediction", return_value=(new, 2)
        ):
            executor._ingest_latest_prediction(start + 10 * executor.step_dt_ns + 1)
        self.assertIs(executor.active_prediction, new)
        self.assertEqual(executor.step_index, 3)

    def test_whole_stale_and_old_generation_do_not_replace_future_plan(self):
        executor = self.executor()
        chunk = np.zeros((15, 19))
        active = Prediction(1, 1, 2_000_000_000, chunk, 12.5, 3.25, 0.125)
        executor.active_prediction = active
        for seq, prediction in enumerate(
            (
                Prediction(1, 1, 2, chunk, 12.5, 3.25, 0.125),
                Prediction(0, 1, 3_000_000_000, chunk, 12.5, 3.25, 0.125),
            ),
            1,
        ):
            with mock.patch.object(
                executor_mod, "read_latest_prediction", return_value=(prediction, seq)
            ):
                executor._ingest_latest_prediction(1_000_000_000)
            self.assertIs(executor.active_prediction, active)

    def test_delayed_executor_skips_elapsed_targets_without_burst(self):
        executor = self.executor()
        start = 1_000_000_000
        dt = executor.step_dt_ns
        chunk = np.repeat(np.arange(15, dtype=np.float64)[:, None], 19, axis=1)
        executor.active_prediction = Prediction(
            1, start, start, chunk, 12.5, 3.25, 0.125
        )
        executor.schedule_base_ns = start
        executor.step_index = 0
        now = start + 3 * dt
        due = executor._next_due_action(now)
        self.assertEqual(due[0][0], 4)
        self.assertGreater(due[1], now)
        executor._consume_control_slot(due[2], now)
        self.assertIsNone(executor._next_due_action(now + 1))
        self.assertIsNone(executor._next_due_action(start + 14 * dt))
        self.assertIsNone(executor.active_prediction)


class TestRecordedRolloutLifecycle(unittest.TestCase):
    def executor(self, *, num_episodes=1):
        shared = types.SimpleNamespace(motion_lock=threading.RLock())
        for name, value in dict(
            is_running=True,
            error_state=False,
            estop_request=False,
            quit_requested=False,
            start_request=True,
            stop_request=int(StopRequest.NONE),
            safety_state=int(SafetyState.ARMED),
            run_generation=1,
            run_started_monotonic_ns=0,
            physical_home_completed=True,
            is_recording=False,
        ).items():
            setattr(shared, name, types.SimpleNamespace(value=value))
        runtime = _fake_runtime()
        runtime.policy.first_command_timeout_s = 10.0
        runtime.policy.max_command_silence_s = 10.0
        runtime.policy.command_progress_timeout_s = 1.0
        runtime.policy.executor_poll_hz = 128.0
        runtime.camera = types.SimpleNamespace(max_frame_age_s=1.0)
        runtime.hand = types.SimpleNamespace(
            mechanical_qpos_min_rad=np.full(12, -2.0),
            mechanical_qpos_max_rad=np.full(12, 2.0),
            T_eef_handbase_pos_xyz=(0.0, 0.0, 0.0),
            T_eef_handbase_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
        config = RolloutRecordingConfig(
            "/tmp/rollout-test",
            "task",
            "me",
            60.0,
            num_episodes,
        )
        with mock.patch.object(executor_mod, "_build_policy_safety_gate"):
            executor = PolicyExecutor(
                shared,
                runtime,
                _fake_policy_spec(),
                execute=True,
                max_running_s=60.0,
                recording_config=config,
            )
        recorder = mock.Mock()
        recorder.stop_pending = False
        recorder.is_recording = False
        recorder.start_episode.return_value = True
        recorder.poll_stop.return_value = RecorderStopResult(done=False)
        executor.recorder = recorder
        shared.arm_state_ring = mock.Mock()
        shared.hand_state_ring = mock.Mock()
        shared.hand_tactile_ring = mock.Mock()
        return executor

    def begin(self, executor):
        with mock.patch.object(
            executor_mod, "_physical_start_pose_rejection", return_value=None
        ):
            executor._start_requested_episode()

    def test_start_ack_is_only_recording_barrier(self):
        executor = self.executor()
        self.begin(executor)
        self.assertEqual(executor.shared.safety_state.value, SafetyState.RUNNING)
        executor.recorder.add_frame.assert_not_called()
        executor.progress.arm_accepted_action_id = 0
        executor.progress.hand_accepted_action_id = 0
        with (
            mock.patch.object(executor, "_observe_worker_progress", return_value=True),
            mock.patch.object(
                executor, "_ingest_latest_prediction", return_value=True
            ) as ingest,
        ):
            executor._run_active_tick(executor.run_started_ns + 1)
        ingest.assert_called_once()

    def test_start_failure_never_begins_motion(self):
        executor = self.executor()
        executor.recorder.start_episode.return_value = False
        self.begin(executor)
        self.assertEqual(executor.shared.safety_state.value, SafetyState.ARMED)
        self.assertIsNone(executor.run_started_ns)

    def test_cancel_after_start_ack_waits_for_recorder_terminal_status(self):
        executor = self.executor()
        with mock.patch.object(
            executor_mod,
            "_physical_start_pose_rejection",
            side_effect=[None, "home invalidated"],
        ):
            executor._start_requested_episode()
        self.assertIsNone(executor.run_started_ns)
        self.assertTrue(executor.shared.is_recording.value)
        executor.recorder.stop_episode.assert_called_once_with(
            save=False, reason="start_recheck_failed"
        )
        executor._complete_recording(executor_mod.RecorderStopResult(done=True))
        self.assertFalse(executor.shared.is_recording.value)

    def test_timeout_termination_reasons(self):
        for event in ("timeout", "first_command_timeout", "command_silence_timeout"):
            with self.subTest(event=event):
                executor = self.executor()
                self.begin(executor)
                start = executor.run_started_ns
                if event == "timeout":
                    now = start + executor.max_running_ns
                elif event == "first_command_timeout":
                    now = start + executor.first_command_timeout_ns + 1
                else:
                    executor.last_valid_command_ns = start + 1
                    now = start + executor.command_silence_timeout_ns + 2
                with mock.patch.object(
                    executor, "_observe_worker_progress", return_value=True
                ):
                    executor._run_active_tick(now)
                executor.recorder.stop_episode.assert_called_once_with(
                    save=True, reason=event
                )
                self.assertEqual(executor._pending_stop_reason, event)
                self.assertEqual(executor.shared.safety_state.value, SafetyState.ARMED)

    def test_shared_fault_paths_use_technical_reasons(self):
        for event in ("hardware_fault", "estop", "recorder_fault", "max_frames"):
            with self.subTest(event=event):
                executor = self.executor()
                self.begin(executor)
                if event == "hardware_fault":
                    executor._fault("offline fault")
                elif event == "estop":
                    executor.shared.estop_request.value = True
                    executor._handle_run_boundary()
                else:
                    executor.recorder.stop_pending = True
                    executor.recorder.poll_stop.return_value = RecorderStopResult(
                        done=False,
                        reason="max_frames" if event == "max_frames" else "unexpected",
                    )
                    executor._poll_recorder()
                self.assertEqual(executor._pending_stop_reason, event)
                if event in {"hardware_fault", "recorder_fault"}:
                    self.assertTrue(executor.shared.error_state.value)
                else:
                    self.assertFalse(executor.shared.error_state.value)
                expected_state = (
                    SafetyState.FAULT
                    if event in {"hardware_fault", "estop"}
                    else SafetyState.ARMED
                )
                self.assertEqual(executor.shared.safety_state.value, expected_state)

    def test_malformed_ipc_timing_is_passive_diagnostic(self):
        executor = self.executor()
        self.begin(executor)
        now = executor.run_started_ns
        frame = serialize_prediction(
            Prediction(
                executor.run_generation,
                now,
                now,
                np.zeros((15, 19)),
                12.5,
                3.25,
                0.125,
            )
        )
        frame["inference_latency_ms"][0] = np.nan
        executor.shared.prediction_ring = mock.Mock()
        executor.shared.prediction_ring.read_latest.return_value = (frame, now, 1)
        self.assertTrue(executor._ingest_latest_prediction(now))
        self.assertEqual(executor.shared.safety_state.value, SafetyState.RUNNING)
        self.assertIsNotNone(executor.active_prediction)
        self.assertNotIn("inference_latency_ms", executor.stats.snapshot())

    def test_recording_failure_fences_without_worker_acceptance(self):
        executor = self.executor()
        self.begin(executor)
        generation = executor.shared.run_generation.value
        executor.progress.latest_published_action_id = 42
        executor.progress.arm_accepted_action_id = 41
        executor.progress.hand_accepted_action_id = 40

        def stopped(**kwargs):
            self.assertEqual(executor.shared.safety_state.value, SafetyState.ARMED)
            self.assertGreater(executor.shared.run_generation.value, generation)
            self.assertFalse(kwargs["save"])

        executor.recorder.stop_episode.side_effect = stopped
        with mock.patch.object(
            executor.progress,
            "covers",
            create=True,
            side_effect=AssertionError("acceptance fence"),
        ):
            executor._invalidate_rollout(
                "missing frame", stop_reason="recording_failure", recorder_save=False
            )
        self.assertFalse(executor.shared.error_state.value)
        self.assertIsNone(executor.active_prediction)
        self.assertEqual(executor._pending_stop_reason, "recording_failure")

    def test_terminal_storage_error_fails_closed(self):
        executor = self.executor()
        self.begin(executor)
        executor.recorder.poll_stop.return_value = RecorderStopResult(
            done=True,
            error="disk full",
            path="/tmp/rollout-test/episode",
            saved=False,
        )
        executor._poll_recorder()
        self.assertFalse(executor.shared.is_recording.value)
        self.assertTrue(executor.shared.error_state.value)
        self.assertIsNone(executor._pending_stop_reason)

    def test_operator_stop_saves_and_returns_armed(self):
        executor = self.executor()
        self.begin(executor)
        request_policy_stop(executor.shared)
        executor._handle_run_boundary()
        self.assertEqual(executor.shared.safety_state.value, SafetyState.ARMED)
        self.assertTrue(executor.shared.is_running.value)
        self.assertTrue(executor.recorder.stop_episode.call_args.kwargs["save"])
        self.assertEqual(
            executor.recorder.stop_episode.call_args.kwargs["reason"], "operator"
        )
        self.assertEqual(executor._pending_stop_reason, "operator")
        self.assertEqual(executor.shared.stop_request.value, StopRequest.NONE)

    def test_timeout_quit_and_estop_stop_reasons(self):
        for event, expected_reason in (
            ("timeout", "timeout"),
            ("quit", "quit"),
            ("estop", "estop"),
        ):
            with self.subTest(event=event):
                executor = self.executor()
                self.begin(executor)
                if event == "timeout":
                    with mock.patch.object(
                        executor, "_observe_worker_progress", return_value=True
                    ):
                        executor._run_active_tick(
                            executor.run_started_ns + executor.max_running_ns
                        )
                else:
                    getattr(
                        executor.shared,
                        "quit_requested" if event == "quit" else "estop_request",
                    ).value = True
                    executor._handle_run_boundary()
                self.assertEqual(executor._pending_stop_reason, expected_reason)
                self.assertIsNone(executor.active_prediction)
                if event == "estop":
                    self.assertEqual(
                        executor.shared.safety_state.value, SafetyState.FAULT
                    )

    def test_terminal_storage_status_is_consumed_once(self):
        executor = self.executor()
        self.begin(executor)
        executor._finish_episode("failure")
        executor.recorder.poll_stop.return_value = RecorderStopResult(done=False)
        executor._poll_recorder()
        self.assertTrue(executor.shared.is_recording.value)
        executor.recorder.poll_stop.return_value = RecorderStopResult(
            done=True,
            saved=True,
            path="/tmp/rollout-test/episode",
        )
        executor._poll_recorder()
        self.assertFalse(executor.shared.is_recording.value)
        self.assertIsNone(executor._pending_stop_reason)
        self.assertFalse(executor.shared.error_state.value)

    def test_target_expiring_during_ik_is_not_published(self):
        executor = self.executor()
        self.begin(executor)
        executor.active_prediction = Prediction(
            executor.run_generation, 1, 100, np.zeros((15, 19)), 12.5, 3.25, 0.125
        )
        candidate = types.SimpleNamespace(arm_qpos=np.zeros(7), hand_qpos=np.zeros(12))
        with (
            mock.patch.object(
                executor,
                "_decode_due_action",
                return_value=((np.zeros(7), np.zeros(12)), None, None),
            ),
            mock.patch.object(
                executor_mod, "build_action_candidate", return_value=candidate
            ),
            mock.patch.object(
                executor_mod,
                "prepare_command",
                return_value=types.SimpleNamespace(accepted=True, candidate=candidate),
            ),
            mock.patch.object(executor_mod.time, "monotonic_ns", return_value=100),
            mock.patch.object(executor, "_advance_prediction") as advance,
            mock.patch.object(executor_mod, "publish_command") as publish,
        ):
            executor._publish_due_action(
                np.zeros(19), scheduled_target_ns=100, due_ns=90
            )
        publish.assert_not_called()
        advance.assert_called_once_with()
        self.assertFalse(executor.shared.error_state.value)

    def test_recording_read_failure_occurs_after_command_publication(self):
        executor = self.executor()
        self.begin(executor)
        now = executor.run_started_ns + executor.step_dt_ns
        executor.progress.reset(executor.run_generation)
        executor.progress.arm_accepted_action_id = 0
        executor.progress.hand_accepted_action_id = 0
        executor.active_prediction = Prediction(
            executor.run_generation, now, now, np.zeros((15, 19)), 12.5, 3.25, 0.125
        )
        executor.schedule_base_ns = now
        candidate = types.SimpleNamespace(
            action_id=1,
            arm_qpos=np.zeros(7),
            hand_qpos=np.zeros(12),
            run_generation=executor.run_generation,
        )
        order = []
        published = executor_mod.PublishResult(
            True, ticket=types.SimpleNamespace(published_monotonic_ns=now)
        )

        def publish(*args, **kwargs):
            order.append("publish")
            return published

        def recording_read(*args, **kwargs):
            order.append("record")
            raise RuntimeError("camera transport broken")

        with (
            mock.patch.object(
                executor,
                "_decode_due_action",
                return_value=((np.zeros(7), np.zeros(12)), None, None),
            ),
            mock.patch.object(
                executor_mod, "build_action_candidate", return_value=candidate
            ),
            mock.patch.object(
                executor_mod,
                "prepare_command",
                return_value=types.SimpleNamespace(accepted=True, candidate=candidate),
            ),
            mock.patch.object(executor_mod, "publish_command", side_effect=publish),
            mock.patch.object(
                executor_mod, "read_causal_structured_frame", side_effect=recording_read
            ),
        ):
            executor._publish_due_action(
                np.zeros(19), scheduled_target_ns=now, due_ns=now
            )
        self.assertEqual(order, ["publish", "record"])
        self.assertEqual(executor.episode_steps, 1)
        self.assertEqual(executor._pending_stop_reason, "recording_failure")
        self.assertTrue(executor.shared.error_state.value)
        self.assertEqual(executor.shared.safety_state.value, SafetyState.ARMED)

    def test_ordinary_held_row_needs_no_initial_evidence_transaction(self):
        from dexmani_real.ipc.schema import ARM_STATE_DTYPE, HAND_STATE_DTYPE

        executor = self.executor()
        self.begin(executor)
        now = executor.next_record_ns
        source_ns = now - 1
        arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
        hand = np.zeros(1, dtype=HAND_STATE_DTYPE)
        for frame in (arm, hand):
            frame["source_monotonic_ns"] = source_ns
            frame["state_valid"] = True
        camera = dict(
            source_monotonic_ns=source_ns,
            camera_health=0,
            ring_sequence=1,
            publish_monotonic_ns=source_ns,
            receive_monotonic_ns=source_ns,
        )
        state = types.SimpleNamespace(
            arm_qpos=np.zeros(7),
            hand_qpos=np.zeros(12),
        )
        with (
            mock.patch.object(
                executor_mod,
                "read_causal_structured_frame",
                side_effect=[(arm, source_ns, 1), (hand, source_ns, 1), None],
            ),
            mock.patch.object(
                executor_mod, "read_camera_frame_causal", return_value=camera
            ),
            mock.patch.object(executor_mod, "build_episode_state", return_value=state),
            mock.patch.object(
                executor_mod, "read_hand_contact_causal", return_value=None
            ),
            mock.patch.object(executor, "_record_frame", return_value=True) as record,
        ):
            executor._record_rollout_tick(now)
        record.assert_called_once()
        self.assertEqual(
            record.call_args.kwargs["signals"]["frame_status"],
            executor_mod._RECORD_FRAME_HELD,
        )
        self.assertFalse(record.call_args.kwargs["signals"]["action_queued"])
        self.assertEqual(executor.shared.safety_state.value, SafetyState.RUNNING)

    def test_lifecycle_waits_for_result_not_arm_hand_acceptance(self):
        from dexmani_real.deployment.lifecycle import _wait_for_rollout_recording

        executor = self.executor()
        executor.shared.is_recording.value = True
        owners = [
            types.SimpleNamespace(name=name, is_alive=lambda: True)
            for name in ("policy", "recorder")
        ]
        with mock.patch(
            "dexmani_real.deployment.lifecycle.time.sleep",
            side_effect=lambda _: setattr(executor.shared.is_recording, "value", False),
        ):
            self.assertTrue(_wait_for_rollout_recording(executor.shared, owners))
        executor.shared.is_recording.value = True
        owners[0].is_alive = lambda: False
        self.assertFalse(_wait_for_rollout_recording(executor.shared, owners))

    def test_no_begin_is_noop(self):
        executor = PolicyExecutor.__new__(PolicyExecutor)
        executor.shared = types.SimpleNamespace(
            start_request=types.SimpleNamespace(value=False)
        )
        with mock.patch.object(executor_mod, "begin_requested_motion") as begin:
            executor._start_requested_episode()
        begin.assert_not_called()

    def test_second_episode_uses_next_name_after_finalization(self):
        executor = self.executor(num_episodes=2)
        self.begin(executor)
        executor._finish_episode("stop", stop_reason="operator")
        executor.recorder.poll_stop.return_value = RecorderStopResult(
            done=True, saved=True, path="/tmp/rollout-test/episode_001"
        )
        executor._poll_recorder()
        self.assertEqual(executor.completed_episodes, 1)
        # A second episode needs a fresh physical home, then a fresh B.
        executor.shared.physical_home_completed.value = True
        executor.shared.start_request.value = True
        self.begin(executor)
        self.assertEqual(
            executor.recorder.start_episode.call_args.kwargs["episode_name"],
            "episode_002",
        )

    def test_auto_quit_only_after_nth_finalization(self):
        executor = self.executor(num_episodes=2)
        for index in range(2):
            self.begin(executor)
            executor._finish_episode("stop", stop_reason="operator")
            executor.recorder.poll_stop.return_value = RecorderStopResult(
                done=True,
                saved=True,
                path=f"/tmp/rollout-test/episode_{index + 1:03d}",
            )
            executor._poll_recorder()
            self.assertEqual(executor.completed_episodes, index + 1)
            if index == 0:
                self.assertFalse(executor.shared.quit_requested.value)
            else:
                self.assertTrue(executor.shared.quit_requested.value)

    def test_unsaved_episode_does_not_increment(self):
        executor = self.executor(num_episodes=2)
        self.begin(executor)
        executor._finish_episode("stop", stop_reason="operator")
        executor.recorder.poll_stop.return_value = RecorderStopResult(
            done=True, saved=False, error=None, path=None
        )
        executor._poll_recorder()
        self.assertEqual(executor.completed_episodes, 0)
        self.assertFalse(executor.shared.quit_requested.value)
        self.assertFalse(executor.shared.error_state.value)

    def test_storage_error_fails_closed_without_increment(self):
        executor = self.executor()
        self.begin(executor)
        executor._finish_episode("stop", stop_reason="operator")
        executor.recorder.poll_stop.return_value = RecorderStopResult(
            done=True, saved=False, error="disk full", path=None
        )
        executor._poll_recorder()
        self.assertEqual(executor.completed_episodes, 0)
        self.assertTrue(executor.shared.error_state.value)
        self.assertFalse(executor.shared.quit_requested.value)

    def test_episode_end_clears_physical_home(self):
        executor = self.executor()
        self.begin(executor)
        executor.shared.physical_home_completed.value = True
        executor._finish_episode("stop", stop_reason="operator")
        self.assertFalse(executor.shared.physical_home_completed.value)

    def test_execute_false_dry_run_begin_and_stop(self):
        shared = types.SimpleNamespace(motion_lock=threading.RLock())
        for name, value in dict(
            is_running=True,
            error_state=False,
            estop_request=False,
            quit_requested=False,
            start_request=True,
            stop_request=int(StopRequest.NONE),
            safety_state=int(SafetyState.ARMED),
            run_generation=1,
            run_started_monotonic_ns=0,
            physical_home_completed=True,
            is_recording=False,
        ).items():
            setattr(shared, name, types.SimpleNamespace(value=value))
        runtime = _fake_runtime()
        runtime.policy.first_command_timeout_s = 10.0
        runtime.policy.max_command_silence_s = 10.0
        runtime.policy.command_progress_timeout_s = 1.0
        runtime.policy.executor_poll_hz = 128.0
        with mock.patch.object(executor_mod, "_build_policy_safety_gate"):
            executor = PolicyExecutor(
                shared,
                runtime,
                _fake_policy_spec(),
                execute=False,
                max_running_s=None,
            )
        with mock.patch.object(
            executor_mod, "_physical_start_pose_rejection", return_value=None
        ):
            executor._start_requested_episode()
        self.assertEqual(executor.shared.safety_state.value, SafetyState.RUNNING)
        request_policy_stop(executor.shared)
        executor._handle_run_boundary()
        self.assertEqual(executor.shared.stop_request.value, StopRequest.NONE)
        self.assertFalse(executor.shared.error_state.value)


if __name__ == "__main__":
    unittest.main()
