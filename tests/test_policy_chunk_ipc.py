"""Offline contracts for the parallel latest-wins ActionChunk transport."""

from __future__ import annotations

import os
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from dexmani_real.config.runtime import resolve_runtime_config
from dexmani_real.deployment.config import PolicyDeploymentConfig, PolicyWorkerConfig
from dexmani_real.deployment.contracts import ActionChunk, PolicyPrediction
from dexmani_real.deployment.coordinator import (
    action_chunk_from_record,
    read_latest_action_chunk,
)
from dexmani_real.deployment.worker import (
    _clear_sync_request_for_inactive_snapshot,
    _consume_sync_request,
    inference_loop,
    publish_action_chunk,
    serialize_action_chunk,
)
from dexmani_real.ipc.channels import RuntimeChannels, RuntimeChannelsConfig, new_frame
from dexmani_real.ipc.schema import (
    COUPLED_COMMAND_DTYPE,
    MAX_POLICY_CHUNK_STEPS,
    POLICY_CHUNK_DTYPE,
)
from dexmani_real.runtime.safety import (
    SafetyState,
    begin_motion,
    publish_coupled_command_if_motion_permitted,
    read_run_state_snapshot,
    transition,
)


def _joint_chunk(
    *,
    chunk_id: int = 3,
    run_generation: int = 7,
    num_steps: int = 2,
) -> ActionChunk:
    return ActionChunk(
        chunk_id=chunk_id,
        run_generation=run_generation,
        observation_id=11,
        observation_anchor_monotonic_ns=300,
        observation_latest_source_monotonic_ns=100,
        observation_logical_step_monotonic_ns=200,
        inference_started_monotonic_ns=300,
        inference_finished_monotonic_ns=400,
        num_steps=num_steps,
        arm_present=True,
        ee_present=False,
        hand_present=True,
        arm_qpos=np.arange(num_steps * 7, dtype=np.float64).reshape(num_steps, 7),
        hand_qpos=np.arange(num_steps * 12, dtype=np.float64).reshape(num_steps, 12),
    )


class ActionChunkContractTest(unittest.TestCase):
    def test_joint_round_trip_preserves_generation_and_owns_arrays(self) -> None:
        source = _joint_chunk()
        frame = serialize_action_chunk(source)

        self.assertEqual(frame.dtype, POLICY_CHUNK_DTYPE)
        self.assertNotIn("target_monotonic_ns", frame.dtype.names)
        self.assertNotIn("valid_mask", frame.dtype.names)
        restored = action_chunk_from_record(frame[0])

        self.assertEqual(restored.chunk_id, 3)
        self.assertEqual(restored.run_generation, 7)
        np.testing.assert_array_equal(restored.arm_qpos, source.arm_qpos)
        np.testing.assert_array_equal(restored.hand_qpos, source.hand_qpos)
        self.assertFalse(restored.arm_qpos.flags.writeable)
        frame["arm_qpos"][0, 0, 0] = -100.0
        self.assertNotEqual(restored.arm_qpos[0, 0], -100.0)

    def test_ee_presence_round_trip(self) -> None:
        chunk = ActionChunk(
            chunk_id=1,
            run_generation=2,
            observation_id=3,
            observation_anchor_monotonic_ns=30,
            observation_latest_source_monotonic_ns=10,
            observation_logical_step_monotonic_ns=20,
            inference_started_monotonic_ns=30,
            inference_finished_monotonic_ns=40,
            num_steps=1,
            arm_present=False,
            ee_present=True,
            hand_present=False,
            arm_qpos=None,
            hand_qpos=None,
            ee_pos=np.ones((1, 3), dtype=np.float64),
            ee_rot6d=np.ones((1, 6), dtype=np.float64),
        )

        restored = action_chunk_from_record(serialize_action_chunk(chunk)[0])

        self.assertTrue(restored.is_ee)
        self.assertIsNone(restored.arm_qpos)
        self.assertIsNone(restored.hand_qpos)
        np.testing.assert_array_equal(restored.ee_pos, chunk.ee_pos)
        np.testing.assert_array_equal(restored.ee_rot6d, chunk.ee_rot6d)

    def test_rejects_generation_shape_presence_capacity_and_time_order(self) -> None:
        cases = (
            {"run_generation": -1},
            {"arm_qpos": np.zeros((2, 6), dtype=np.float64)},
            {"arm_present": True, "ee_present": True},
            {"hand_present": False},
            {"num_steps": MAX_POLICY_CHUNK_STEPS + 1},
            {
                "observation_latest_source_monotonic_ns": 301,
                "observation_logical_step_monotonic_ns": 200,
            },
        )
        base = dict(_joint_chunk().__dict__)
        for overrides in cases:
            with self.subTest(overrides=overrides):
                values = dict(base)
                values.update(overrides)
                with self.assertRaises((TypeError, ValueError)):
                    ActionChunk(**values)

    def test_record_rejects_non_binary_presence(self) -> None:
        frame = serialize_action_chunk(_joint_chunk())
        frame["arm_present"][0] = 2

        with self.assertRaisesRegex(ValueError, "arm_present must be 0 or 1"):
            action_chunk_from_record(frame[0])


class ActionChunkRuntimeChannelsTest(unittest.TestCase):
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
            frame["run_generation"][0] = shared.run_generation.value
            frame["action_id"][0] = 1
            frame["arm_present"][0] = 1
            self.assertIsNone(
                publish_coupled_command_if_motion_permitted(
                    shared,
                    expected_run_generation=int(shared.run_generation.value),
                    frame=frame,
                    required_state=SafetyState.RUNNING,
                )
            )
            self.assertIsNone(shared.coupled_cmd_ring.read_latest())

            self.assertTrue(begin_motion(shared))
            frame["run_generation"][0] = shared.run_generation.value
            ticket = publish_coupled_command_if_motion_permitted(
                shared,
                expected_run_generation=int(shared.run_generation.value),
                frame=frame,
                required_state=SafetyState.RUNNING,
            )
            self.assertIsNotNone(ticket)
        finally:
            self.assertTrue(shared.close())

    def test_single_slot_ring_and_inference_event(self) -> None:
        prefix = f"dexmani_test_chunk_{os.getpid()}_{time.monotonic_ns()}"
        shared = RuntimeChannels.create(
            prefix=prefix,
            config=RuntimeChannelsConfig(camera_ring_maxlen=1),
            camera_rgb_shape=(2, 2, 3),
            camera_depth_shape=(2, 2),
        )
        try:
            self.assertEqual(shared.policy_chunk_ring.maxlen, 1)
            self.assertFalse(shared.inference_request.is_set())
            shared.inference_request.set()
            self.assertTrue(shared.inference_request.wait(timeout=0.01))
            shared.inference_request.clear()
            self.assertFalse(shared.inference_request.is_set())

            shared.policy_chunk_ring.write(serialize_action_chunk(_joint_chunk()))
            shared.policy_chunk_ring.write(
                serialize_action_chunk(_joint_chunk(chunk_id=4))
            )
            restored = read_latest_action_chunk(shared)
            self.assertIsNotNone(restored)
            assert restored is not None
            self.assertEqual(restored.chunk_id, 4)
            self.assertEqual(restored.run_generation, 7)
        finally:
            self.assertTrue(shared.close())

    def test_chunk_ring_capacity_is_fixed(self) -> None:
        with self.assertRaisesRegex(ValueError, "must remain exactly 1"):
            RuntimeChannelsConfig(policy_chunk_ring_maxlen=2)


class _Value:
    def __init__(self, value: int | bool) -> None:
        self.value = value


class _CaptureChunkRing:
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
        self.motion_lock = threading.Lock()
        self.action_control_hz = 16.0
        self.inference_request = threading.Event()
        self.policy_chunk_ring = _CaptureChunkRing(self)
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

    def predict(self, _observation: object) -> PolicyPrediction:
        self.predict_calls += 1
        return PolicyPrediction(
            arm_qpos=np.zeros((self.output_steps, 7), dtype=np.float64),
            hand_qpos=np.zeros((self.output_steps, 12), dtype=np.float64),
        )

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

    def test_request_produces_one_generation_fenced_chunk(self) -> None:
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
        self.assertEqual(len(shared.policy_chunk_ring.frames), 1)
        chunk = action_chunk_from_record(shared.policy_chunk_ring.frames[0][0])
        self.assertEqual(chunk.run_generation, 7)
        self.assertEqual(chunk.num_steps, 2)

    def test_async_inference_publishes_directly_to_action_chunk_ring(self) -> None:
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
        self.assertEqual(len(shared.policy_chunk_ring.frames), 1)
        chunk = action_chunk_from_record(shared.policy_chunk_ring.frames[0][0])
        self.assertEqual(chunk.run_generation, 7)
        self.assertEqual(chunk.num_steps, 4)

    def test_publish_rejects_generation_change(self) -> None:
        shared = _SyncInferenceShared()
        shared.run_generation.value = 8

        self.assertFalse(publish_action_chunk(shared, _joint_chunk(run_generation=7)))
        self.assertEqual(shared.policy_chunk_ring.frames, [])


if __name__ == "__main__":
    unittest.main()
