"""Offline contracts for the parallel latest-wins flat prediction transport."""

from __future__ import annotations

import os
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.config.runtime import resolve_runtime_config
from dexmani_real.deployment.config import PolicyDeploymentConfig, PolicyWorkerConfig
from dexmani_real.deployment.contracts import Prediction
from dexmani_real.deployment.executor import (
    prediction_from_record,
    read_latest_prediction,
)
from dexmani_real.deployment.worker import (
    _clear_sync_request_for_inactive_snapshot,
    _consume_sync_request,
    inference_loop,
    publish_prediction,
    serialize_prediction,
)
from dexmani_real.ipc.channels import RuntimeChannels, RuntimeChannelsConfig, new_frame
from dexmani_real.ipc.schema import (
    COUPLED_COMMAND_DTYPE,
    MAX_PREDICTION_STEPS,
    PREDICTION_DTYPE,
)
from dexmani_real.robot.arm_worker import _handle_servo_command, _LoopState
from dexmani_real.runtime.safety import (
    PUBLISH_REASON_EXPIRED,
    PUBLISH_REASON_FAULT,
    PUBLISH_REASON_SAFETY_STATE,
    CoupledCommandTicket,
    SafetyState,
    begin_motion,
    coupled_command_ticket_allows_execution,
    publish_coupled_command_if_motion_permitted,
    read_run_state_snapshot,
    transition,
)


def _prediction(
    *,
    run_generation: int = 7,
    num_steps: int = 2,
) -> Prediction:
    return Prediction(
        run_generation=run_generation,
        source_monotonic_ns=100,
        logical_step_monotonic_ns=200,
        actions=np.arange(num_steps * 19, dtype=np.float64).reshape(num_steps, 19),
    )


class PredictionContractTest(unittest.TestCase):
    def test_joint_round_trip_preserves_metadata_and_owns_actions(self) -> None:
        source = _prediction()
        frame = serialize_prediction(source)

        self.assertEqual(frame.dtype, PREDICTION_DTYPE)
        self.assertNotIn("target_monotonic_ns", frame.dtype.names)
        self.assertNotIn("chunk_id", frame.dtype.names)
        self.assertNotIn("observation_id", frame.dtype.names)
        restored = prediction_from_record(frame[0])

        self.assertEqual(restored.run_generation, 7)
        self.assertEqual(restored.source_monotonic_ns, 100)
        self.assertEqual(restored.logical_step_monotonic_ns, 200)
        np.testing.assert_array_equal(restored.actions, source.actions)
        self.assertFalse(restored.actions.flags.writeable)
        frame["actions"][0, 0, 0] = -100.0
        self.assertNotEqual(restored.actions[0, 0], -100.0)

    def test_ee_round_trip_is_exact(self) -> None:
        source = Prediction(
            run_generation=2,
            source_monotonic_ns=10,
            logical_step_monotonic_ns=20,
            actions=np.arange(21, dtype=np.float64).reshape(1, 21),
        )
        restored = prediction_from_record(serialize_prediction(source)[0])
        np.testing.assert_array_equal(restored.actions, source.actions)

    def test_rejects_generation_shape_capacity_and_time_order(self) -> None:
        cases = (
            {"run_generation": -1},
            {"actions": np.zeros((2, 22), dtype=np.float64)},
            {"actions": np.zeros((MAX_PREDICTION_STEPS + 1, 19), dtype=np.float64)},
            {
                "source_monotonic_ns": 301,
                "logical_step_monotonic_ns": 200,
            },
        )
        base = dict(_prediction().__dict__)
        for overrides in cases:
            with self.subTest(overrides=overrides):
                values = dict(base)
                values.update(overrides)
                with self.assertRaises((TypeError, ValueError)):
                    Prediction(**values)

    def test_record_rejects_unsupported_action_dimension(self) -> None:
        frame = serialize_prediction(_prediction())
        frame["action_dim"][0] = 20

        with self.assertRaisesRegex(ValueError, "action_dim"):
            prediction_from_record(frame[0])


class PredictionRuntimeChannelsTest(unittest.TestCase):
    def test_policy_publication_requires_running_at_atomic_boundary(self) -> None:
        prefix = f"dexmani_test_policy_gate_{os.getpid()}_{time.monotonic_ns()}"
        shared = RuntimeChannels.create(
            prefix=prefix,
            config=RuntimeChannelsConfig(camera_ring_maxlen=1),
            camera_rgb_shape=(2, 2, 3),
            camera_depth_shape=(2, 2),
        )
        try:
            self.assertTrue(transition(shared, SafetyState.ARMED))
            frame = new_frame(COUPLED_COMMAND_DTYPE)
            now_ns = time.monotonic_ns()
            frame["run_generation"][0] = shared.run_generation.value
            frame["action_id"][0] = 1
            frame["arm_present"][0] = 1
            frame["created_monotonic_ns"][0] = now_ns
            frame["scheduled_target_monotonic_ns"][0] = now_ns
            frame["target_monotonic_ns"][0] = now_ns
            frame["valid_until_monotonic_ns"][0] = now_ns + 1_000_000_000
            ticket, rejection_reason = publish_coupled_command_if_motion_permitted(
                shared,
                expected_run_generation=int(shared.run_generation.value),
                frame=frame,
                required_state=SafetyState.RUNNING,
            )
            self.assertIsNone(ticket)
            self.assertEqual(
                rejection_reason,
                f"{PUBLISH_REASON_SAFETY_STATE}: expected RUNNING, got ARMED",
            )
            self.assertIsNone(shared.coupled_cmd_ring.read_latest())

            self.assertTrue(begin_motion(shared))
            frame["run_generation"][0] = shared.run_generation.value
            ticket, rejection_reason = publish_coupled_command_if_motion_permitted(
                shared,
                expected_run_generation=int(shared.run_generation.value),
                frame=frame,
                required_state=SafetyState.RUNNING,
            )
            self.assertIsNotNone(ticket)
            self.assertEqual(rejection_reason, "")
            assert ticket is not None
            self.assertGreater(ticket.published_monotonic_ns, 0)
        finally:
            self.assertTrue(shared.close())

    def test_newer_command_revokes_the_older_worker_execution_ticket(self) -> None:
        prefix = f"dexmani_test_command_supersede_{os.getpid()}_{time.monotonic_ns()}"
        shared = RuntimeChannels.create(
            prefix=prefix,
            config=RuntimeChannelsConfig(camera_ring_maxlen=1),
            camera_rgb_shape=(2, 2, 3),
            camera_depth_shape=(2, 2),
        )
        try:
            self.assertTrue(transition(shared, SafetyState.ARMED))
            self.assertTrue(begin_motion(shared))
            generation = int(shared.run_generation.value)
            frame = new_frame(COUPLED_COMMAND_DTYPE)
            now_ns = time.monotonic_ns()
            frame["run_generation"][0] = generation
            frame["arm_present"][0] = 1
            frame["hand_present"][0] = 1
            frame["created_monotonic_ns"][0] = now_ns
            frame["scheduled_target_monotonic_ns"][0] = now_ns
            frame["target_monotonic_ns"][0] = now_ns
            frame["valid_until_monotonic_ns"][0] = now_ns + 1_000_000_000

            frame["action_id"][0] = 1
            older_ticket, rejection_reason = (
                publish_coupled_command_if_motion_permitted(
                    shared,
                    expected_run_generation=generation,
                    frame=frame,
                    required_state=SafetyState.RUNNING,
                )
            )
            self.assertIsNotNone(older_ticket)
            self.assertEqual(rejection_reason, "")
            assert older_ticket is not None
            self.assertTrue(
                coupled_command_ticket_allows_execution(
                    shared,
                    ticket=older_ticket,
                )
            )

            frame["action_id"][0] = 2
            newer_ticket, rejection_reason = (
                publish_coupled_command_if_motion_permitted(
                    shared,
                    expected_run_generation=generation,
                    frame=frame,
                    required_state=SafetyState.RUNNING,
                )
            )
            self.assertIsNotNone(newer_ticket)
            self.assertEqual(rejection_reason, "")
            assert newer_ticket is not None

            self.assertFalse(
                coupled_command_ticket_allows_execution(
                    shared,
                    ticket=older_ticket,
                )
            )
            self.assertTrue(
                coupled_command_ticket_allows_execution(
                    shared,
                    ticket=newer_ticket,
                )
            )
        finally:
            self.assertTrue(shared.close())

    def test_atomic_permit_and_expiry_checks_fence_publication_and_execution(
        self,
    ) -> None:
        """Permit loss or deadline expiry between prechecks cannot reach an SDK."""

        class CaptureRing:
            def __init__(self) -> None:
                self.writes = 0

            def write(self, _frame: np.ndarray) -> int:
                self.writes += 1
                return self.writes

        ring = CaptureRing()
        shared = SimpleNamespace(
            is_running=_Value(True),
            error_state=_Value(False),
            estop_request=_Value(False),
            safety_state=_Value(int(SafetyState.RUNNING)),
            run_generation=_Value(4),
            active_coupled_command_sequence=_Value(0),
            coupled_cmd_ring=ring,
        )

        class PermitRevokedOnLockEntry:
            def __enter__(self) -> "PermitRevokedOnLockEntry":
                shared.error_state.value = True
                return self

            def __exit__(self, *_args: object) -> None:
                return None

        frame = new_frame(COUPLED_COMMAND_DTYPE)
        frame["run_generation"][0] = 4
        frame["action_id"][0] = 1
        frame["arm_present"][0] = 1
        frame["valid_until_monotonic_ns"][0] = 250
        shared.motion_lock = PermitRevokedOnLockEntry()
        ticket, rejection_reason = publish_coupled_command_if_motion_permitted(
            shared,
            expected_run_generation=4,
            frame=frame,
            required_state=SafetyState.RUNNING,
        )
        self.assertIsNone(ticket)
        self.assertEqual(rejection_reason, PUBLISH_REASON_FAULT)
        self.assertEqual(ring.writes, 0)

        shared.error_state.value = False
        shared.motion_lock = threading.Lock()
        with patch("dexmani_real.runtime.safety.time.monotonic_ns", return_value=250):
            ticket, rejection_reason = publish_coupled_command_if_motion_permitted(
                shared,
                expected_run_generation=4,
                frame=frame,
                required_state=SafetyState.RUNNING,
            )
        self.assertIsNone(ticket)
        self.assertEqual(rejection_reason, PUBLISH_REASON_EXPIRED)
        self.assertEqual(ring.writes, 0)

        frame["valid_until_monotonic_ns"][0] = 300
        with patch("dexmani_real.runtime.safety.time.monotonic_ns", return_value=251):
            ticket, rejection_reason = publish_coupled_command_if_motion_permitted(
                shared,
                expected_run_generation=4,
                frame=frame,
                required_state=SafetyState.RUNNING,
            )
        self.assertIsNotNone(ticket)
        self.assertEqual(rejection_reason, "")
        assert ticket is not None
        with patch("dexmani_real.runtime.safety.time.monotonic_ns", return_value=300):
            self.assertFalse(
                coupled_command_ticket_allows_execution(shared, ticket=ticket)
            )

    def test_single_slot_ring_and_inference_event(self) -> None:
        prefix = f"dexmani_test_prediction_{os.getpid()}_{time.monotonic_ns()}"
        shared = RuntimeChannels.create(
            prefix=prefix,
            config=RuntimeChannelsConfig(camera_ring_maxlen=1),
            camera_rgb_shape=(2, 2, 3),
            camera_depth_shape=(2, 2),
        )
        try:
            self.assertEqual(shared.prediction_ring.maxlen, 1)
            self.assertFalse(shared.inference_request.is_set())
            shared.inference_request.set()
            self.assertTrue(shared.inference_request.wait(timeout=0.01))
            shared.inference_request.clear()
            self.assertFalse(shared.inference_request.is_set())

            shared.prediction_ring.write(serialize_prediction(_prediction()))
            shared.prediction_ring.write(
                serialize_prediction(_prediction(run_generation=8))
            )
            restored = read_latest_prediction(shared)
            self.assertIsNotNone(restored)
            assert restored is not None
            prediction, ring_sequence = restored
            self.assertEqual(ring_sequence, 2)
            self.assertEqual(prediction.run_generation, 8)
        finally:
            self.assertTrue(shared.close())

    def test_prediction_ring_capacity_is_fixed(self) -> None:
        with self.assertRaisesRegex(ValueError, "must remain exactly 1"):
            RuntimeChannelsConfig(prediction_ring_maxlen=2)


class ArmWorkerDeadlineGuardTest(unittest.TestCase):
    def test_deadline_crossed_during_validation_prevents_servo(self) -> None:
        command = new_frame(COUPLED_COMMAND_DTYPE)
        command["run_generation"][0] = 1
        command["action_id"][0] = 1
        command["created_monotonic_ns"][0] = 100
        command["scheduled_target_monotonic_ns"][0] = 100
        command["target_monotonic_ns"][0] = 100
        command["valid_until_monotonic_ns"][0] = 200
        command["arm_present"][0] = 1

        arm = SimpleNamespace(servo=Mock())
        st = _LoopState(
            cfg=SimpleNamespace(
                joint_limit_lower=np.full(7, -1.0),
                joint_limit_upper=np.full(7, 1.0),
                max_servo_command_jump_rad=1.0,
            ),
            arm=arm,
            frame=None,
            last_target=np.zeros(7),
            last_measured_qpos=np.zeros(7),
            last_command_generation=1,
        )
        shared = SimpleNamespace(
            is_running=_Value(True),
            error_state=_Value(False),
            estop_request=_Value(False),
            safety_state=_Value(int(SafetyState.RUNNING)),
            run_generation=_Value(1),
            active_coupled_command_sequence=_Value(1),
            motion_lock=threading.RLock(),
        )
        ticket = CoupledCommandTicket(
            run_generation=1,
            ring_sequence=1,
            valid_until_monotonic_ns=200,
        )

        with (
            patch(
                "dexmani_real.runtime.safety.time.monotonic_ns",
                side_effect=(150, 150, 150, 200),
            ) as safety_clock,
        ):
            _handle_servo_command(st, shared, command, ticket)

        self.assertEqual(safety_clock.call_count, 4)
        arm.servo.assert_not_called()
        self.assertFalse(shared.error_state.value)


class _Value:
    def __init__(self, value: int | bool) -> None:
        self.value = value


class _CapturePredictionRing:
    def __init__(self, shared: object) -> None:
        self.shared = shared
        self.frames: list[np.ndarray] = []

    def write(self, frame: np.ndarray) -> int:
        self.frames.append(frame.copy())
        self.shared.is_running.value = False
        return len(self.frames)


class _SyncInferenceShared:
    def __init__(self) -> None:
        self.is_running = _Value(True)
        self.error_state = _Value(False)
        self.estop_request = _Value(False)
        self.safety_state = _Value(2)
        self.run_generation = _Value(7)
        self.run_started_monotonic_ns = _Value(1)
        self.stop_request = _Value(0)
        self.motion_lock = threading.Lock()
        self.inference_request = threading.Event()
        self.prediction_ring = _CapturePredictionRing(self)
        self.heartbeats: list[float] = []
        self.ready = False

    def set_heartbeat(self, _name: str, value: float) -> None:
        self.heartbeats.append(value)

    def set_ready(self, _name: str) -> None:
        self.ready = True


class _FakePolicyRuntime:
    def __init__(self, *, warmup_s: float = 2.0, output_steps: int = 2) -> None:
        self.predict_calls = 0
        self.reset_calls = 0
        self.closed = False
        self.warmup_s = warmup_s
        self.output_steps = output_steps

    def warmup(self, *, samples: int) -> tuple[float, ...]:
        return (self.warmup_s,) * samples

    def reset_episode(self) -> None:
        self.reset_calls += 1

    def predict(self, _observation: object) -> np.ndarray:
        self.predict_calls += 1
        return np.zeros((self.output_steps, 19), dtype=np.float64)

    def close(self) -> None:
        self.closed = True


def _policy_spec() -> SimpleNamespace:
    return SimpleNamespace(
        action_key="action",
        action_dim=19,
        control_action_dim=19,
        horizon=4,
        n_obs_steps=2,
        n_action_steps=2,
        observation_fields=(
            SimpleNamespace(name="joint_state", shape=(19,), dtype="float32"),
            SimpleNamespace(name="point_cloud", shape=(1024, 6), dtype="float32"),
        ),
        control_dt_s=0.0625,
        requires_hand=True,
    )


class SyncInferenceLoopTest(unittest.TestCase):
    def test_run_snapshot_cannot_mix_armed_epoch_with_running_state(self) -> None:
        shared = _SyncInferenceShared()
        shared.safety_state.value = int(SafetyState.ARMED)
        shared.run_generation.value = 6
        shared.run_started_monotonic_ns.value = 0
        base_lock = threading.Lock()

        class TransitionAfterReadLock:
            def __enter__(self):
                base_lock.acquire()
                return self

            def __exit__(self, *_args):
                shared.run_generation.value = 7
                shared.run_started_monotonic_ns.value = 123
                shared.safety_state.value = int(SafetyState.RUNNING)
                base_lock.release()

        shared.motion_lock = TransitionAfterReadLock()
        snapshot = read_run_state_snapshot(shared)

        self.assertIs(snapshot.state, SafetyState.ARMED)
        self.assertEqual(snapshot.generation, 6)
        self.assertEqual(snapshot.started_monotonic_ns, 0)
        self.assertEqual(shared.safety_state.value, int(SafetyState.RUNNING))

    def test_old_armed_snapshot_cannot_clear_new_generation_request(self) -> None:
        shared = _SyncInferenceShared()
        shared.safety_state.value = int(SafetyState.ARMED)
        shared.run_generation.value = 6
        observed_generation = int(shared.run_generation.value)

        # Model begin_motion() and its first request after the worker captured
        # the older ARMED snapshot.
        shared.run_generation.value = 7
        shared.safety_state.value = int(SafetyState.RUNNING)
        shared.inference_request.set()

        self.assertFalse(
            _clear_sync_request_for_inactive_snapshot(
                shared,
                observed_generation=observed_generation,
            )
        )
        self.assertTrue(shared.inference_request.is_set())
        self.assertIsNone(
            _consume_sync_request(
                shared,
                observed_generation=observed_generation,
            )
        )
        self.assertTrue(shared.inference_request.is_set())

    def test_request_produces_one_generation_fenced_prediction(self) -> None:
        shared = _SyncInferenceShared()
        shared.inference_request.set()
        runtime = _FakePolicyRuntime()
        spec = _policy_spec()
        worker_config = PolicyWorkerConfig("fake/model", "cpu", spec)

        def observation(*_args: object, **kwargs: object) -> SimpleNamespace:
            anchor_ns = int(kwargs["anchor_ns"])
            return SimpleNamespace(
                arm_history=SimpleNamespace(values=np.zeros((2, 7))),
                hand_history=SimpleNamespace(values=np.zeros((2, 12))),
                pointcloud_history=(object(), object()),
                rgb_history=(),
                logical_step_monotonic_ns=anchor_ns - 1,
                latest_source_monotonic_ns=anchor_ns - 2,
            )

        with (
            patch(
                "dexmani_real.deployment.worker._load_inference_runtime",
                return_value=runtime,
            ),
            patch(
                "dexmani_real.deployment.worker._build_observation",
                side_effect=observation,
            ),
            patch(
                "dexmani_real.deployment.worker.observation_timing_ms",
                return_value=(0.0, 0.0),
            ),
            patch(
                "dexmani_real.deployment.worker._to_policy_observation",
                return_value=object(),
            ),
        ):
            inference_loop(
                shared,
                resolve_runtime_config().policy,
                worker_config,
                PolicyDeploymentConfig(inference_mode="sync"),
            )

        self.assertTrue(shared.ready)
        self.assertFalse(shared.inference_request.is_set())
        self.assertEqual(runtime.predict_calls, 1)
        self.assertTrue(runtime.closed)
        self.assertEqual(len(shared.prediction_ring.frames), 1)
        prediction = prediction_from_record(shared.prediction_ring.frames[0][0])
        self.assertEqual(prediction.run_generation, 7)
        self.assertEqual(prediction.num_steps, 2)

    def test_async_inference_publishes_directly_to_prediction_ring(self) -> None:
        shared = _SyncInferenceShared()
        runtime = _FakePolicyRuntime(warmup_s=0.001, output_steps=4)
        spec = _policy_spec()
        spec.n_action_steps = 4
        spec.horizon = 6

        def observation(*_args: object, **kwargs: object) -> SimpleNamespace:
            anchor_ns = int(kwargs["anchor_ns"])
            return SimpleNamespace(
                arm_history=SimpleNamespace(values=np.zeros((2, 7))),
                hand_history=SimpleNamespace(values=np.zeros((2, 12))),
                pointcloud_history=(object(), object()),
                rgb_history=(),
                logical_step_monotonic_ns=anchor_ns - 1,
                latest_source_monotonic_ns=anchor_ns - 2,
            )

        with (
            patch(
                "dexmani_real.deployment.worker._load_inference_runtime",
                return_value=runtime,
            ),
            patch(
                "dexmani_real.deployment.worker._build_observation",
                side_effect=observation,
            ),
            patch(
                "dexmani_real.deployment.worker.observation_timing_ms",
                return_value=(0.0, 0.0),
            ),
            patch(
                "dexmani_real.deployment.worker._to_policy_observation",
                return_value=object(),
            ),
        ):
            inference_loop(
                shared,
                resolve_runtime_config().policy,
                PolicyWorkerConfig("fake/model", "cpu", spec),
                PolicyDeploymentConfig(inference_mode="async"),
            )

        self.assertEqual(runtime.predict_calls, 1)
        self.assertEqual(len(shared.prediction_ring.frames), 1)
        prediction = prediction_from_record(shared.prediction_ring.frames[0][0])
        self.assertEqual(prediction.run_generation, 7)
        self.assertEqual(prediction.num_steps, 4)

    def test_publish_rejects_generation_change(self) -> None:
        shared = _SyncInferenceShared()
        shared.run_generation.value = 8

        self.assertFalse(publish_prediction(shared, _prediction(run_generation=7)))
        self.assertEqual(shared.prediction_ring.frames, [])


if __name__ == "__main__":
    unittest.main()
