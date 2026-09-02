"""Offline contract tests for the public Policy-to-Real NumPy boundary."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from dexmani_real.deployment.config import (
    DeploymentConfig,
    PolicyRuntimeConfig,
    resolve_policy_runtime_config,
)
from dexmani_real.deployment.observation import (
    FrameWindow,
    ObservationBatch,
    PointCloudFrame,
)
from dexmani_real.integrations.dexmani_policy import DexManiPolicyRuntime
from dexmani_real.ipc.schema import MAX_POLICY_CHUNK_STEPS


def _spec(**changes: object) -> SimpleNamespace:
    values = {
        "action_key": "action",
        "action_dim": 19,
        "control_action_dim": 19,
        "horizon": 16,
        "n_obs_steps": 2,
        "n_action_steps": 8,
        "sensor_modalities": ("joint_state", "point_cloud"),
        "point_cloud_num_points": 1024,
        "point_cloud_feature_dim": 6,
        "control_dt_s": 0.0625,
        "requires_hand": True,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _resolved_config(spec: SimpleNamespace | None = None) -> PolicyRuntimeConfig:
    policy_spec = spec or _spec()
    runtime = SimpleNamespace(
        policy=SimpleNamespace(
            control_hz=16.0,
            arm_max_delta_rad_per_tick=0.1,
            endpoint_delta_tolerance_rad=0.01,
        ),
        pointcloud=SimpleNamespace(num_points=1024),
        environment=SimpleNamespace(
            table=SimpleNamespace(enabled=False, plane_abcd=(0.0, 0.0, 1.0, 0.0))
        ),
        hand=SimpleNamespace(hand_max_delta_rad_per_tick=0.3),
    )
    return resolve_policy_runtime_config(
        experiment="dp3/pick/example",
        task_name="pick",
        policy_spec=policy_spec,
        runtime_config=runtime,
        device="cpu",
    ).runtime


def _observation(config: PolicyRuntimeConfig) -> ObservationBatch:
    count = config.deployment.observation_horizon
    source = np.arange(2, count + 2, dtype=np.uint64)
    published = source + 10
    sequence = np.arange(1, count + 1, dtype=np.uint64)
    valid = np.ones(count, dtype=np.uint8)
    arm = FrameWindow(
        values=np.zeros((count, 7), dtype=np.float64),
        source_sequence=sequence,
        source_monotonic_ns=source,
        publish_monotonic_ns=published,
        valid_mask=valid,
    )
    hand = FrameWindow(
        values=np.zeros((count, 12), dtype=np.float64),
        source_sequence=sequence,
        source_monotonic_ns=source,
        publish_monotonic_ns=published,
        valid_mask=valid,
    )
    points = np.zeros((config.deployment.pointcloud_num_points, 6), dtype=np.float32)
    clouds = tuple(
        PointCloudFrame(
            values=points,
            source_camera_sequence=int(index),
            source_monotonic_ns=int(source_ns),
            publish_monotonic_ns=int(published_ns),
            camera_generation=1,
        )
        for index, source_ns, published_ns in zip(
            sequence, source, published, strict=True
        )
    )
    return ObservationBatch(
        observation_id=1,
        run_generation=1,
        run_started_monotonic_ns=1,
        anchor_monotonic_ns=int(published[-1]) + 1,
        latest_source_monotonic_ns=int(source[-1]),
        logical_step_monotonic_ns=int(source[-1]),
        arm_history=arm,
        hand_history=hand,
        pointcloud=clouds[-1],
        pointcloud_history=clouds,
    )


class _FakeLoadedPolicy:
    def __init__(self, spec: SimpleNamespace, output: np.ndarray) -> None:
        self.info = SimpleNamespace(task_name="pick")
        self.spec = spec
        self.output = output
        self.observation = None

    def warmup(self, *, samples: int) -> tuple[float, ...]:
        return (0.01,) * samples

    def reset_episode(self) -> None:
        pass

    def predict(self, observation: object) -> np.ndarray:
        self.observation = observation
        return self.output

    def close(self) -> None:
        pass


class PolicyCompatibilityTest(unittest.TestCase):
    def test_fixed_real_contract_rejections(self) -> None:
        cases = (
            _spec(requires_hand=False),
            _spec(sensor_modalities=("joint_state",)),
            _spec(point_cloud_feature_dim=3),
            _spec(n_action_steps=MAX_POLICY_CHUNK_STEPS + 1),
            _spec(control_dt_s=0.05),
            _spec(point_cloud_num_points=2048),
        )
        for policy_spec in cases:
            with self.subTest(policy_spec=policy_spec), self.assertRaises(ValueError):
                _resolved_config(policy_spec)

    def test_command_ack_timeout_cannot_outlive_command_validity(self) -> None:
        with self.assertRaises(ValueError):
            DeploymentConfig(
                experiment="fake/model",
                task_name="fake",
                hand_enabled=True,
                action_validity_s=0.5,
                command_acknowledgement_timeout_s=0.6,
            )


class PolicyAdapterTest(unittest.TestCase):
    def test_joint_action_split_and_numpy_observation(self) -> None:
        spec = _spec()
        config = _resolved_config(spec)
        output = np.arange(8 * 19, dtype=np.float64).reshape(8, 19)
        loaded = _FakeLoadedPolicy(spec, output)
        adapter = DexManiPolicyRuntime(loaded, config)

        prediction = adapter.predict(_observation(config))

        np.testing.assert_array_equal(prediction.arm_qpos, output[:, :7])
        np.testing.assert_array_equal(prediction.hand_qpos, output[:, 7:])
        self.assertEqual(set(loaded.observation), {"joint_state", "point_cloud"})
        self.assertEqual(loaded.observation["joint_state"].dtype, np.float32)

    def test_ee_action_split_and_rotation_validation(self) -> None:
        spec = _spec(action_key="action_ee", action_dim=21, control_action_dim=21)
        config = _resolved_config(spec)
        output = np.zeros((8, 21), dtype=np.float64)
        output[:, 0] = 0.2
        output[:, 3] = 1.0
        output[:, 7] = 1.0
        adapter = DexManiPolicyRuntime(_FakeLoadedPolicy(spec, output), config)
        prediction = adapter.predict(_observation(config))
        self.assertEqual(prediction.ee_pos.shape, (8, 3))
        self.assertEqual(prediction.ee_rot6d.shape, (8, 6))
        self.assertEqual(prediction.hand_qpos.shape, (8, 12))

        output[:, 3:9] = 0.0
        with self.assertRaises(ValueError):
            adapter.predict(_observation(config))

    def test_rejects_invalid_policy_output(self) -> None:
        spec = _spec()
        config = _resolved_config(spec)
        invalid = (
            np.zeros((7, 19), dtype=np.float64),
            np.zeros((8, 19), dtype=np.float32),
            np.full((8, 19), np.nan, dtype=np.float64),
        )
        for output in invalid:
            with self.subTest(shape=output.shape, dtype=output.dtype):
                adapter = DexManiPolicyRuntime(_FakeLoadedPolicy(spec, output), config)
                with self.assertRaises(ValueError):
                    adapter.predict(_observation(config))

    def test_rejects_history_length_and_camera_generation(self) -> None:
        spec = _spec()
        config = _resolved_config(spec)
        adapter = DexManiPolicyRuntime(
            _FakeLoadedPolicy(spec, np.zeros((8, 19), dtype=np.float64)), config
        )
        observation = _observation(config)
        object.__setattr__(
            observation, "pointcloud_history", observation.pointcloud_history[:1]
        )
        with self.assertRaises(ValueError):
            adapter.predict(observation)

        observation = _observation(config)
        second = observation.pointcloud_history[1]
        mismatched = PointCloudFrame(
            values=second.values,
            source_camera_sequence=second.source_camera_sequence,
            source_monotonic_ns=second.source_monotonic_ns,
            publish_monotonic_ns=second.publish_monotonic_ns,
            camera_generation=2,
        )
        object.__setattr__(
            observation,
            "pointcloud_history",
            (observation.pointcloud_history[0], mismatched),
        )
        with self.assertRaises(ValueError):
            adapter.predict(observation)


if __name__ == "__main__":
    unittest.main()
