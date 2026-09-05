"""Offline contract tests for the public Policy-to-Real NumPy boundary."""

from __future__ import annotations

import pickle
import unittest
from types import SimpleNamespace

import numpy as np

from dexmani_real.config.runtime import resolve_runtime_config
from dexmani_real.deployment.config import (
    PolicyWorkerConfig,
    validate_policy_runtime_compatibility,
)
from dexmani_real.deployment.observation import PolicyObservation
from dexmani_real.integrations.dexmani_policy import DexManiPolicyRuntime
from dexmani_real.ipc.schema import MAX_PREDICTION_STEPS


def _field(name: str, shape: tuple[int, ...], dtype: str) -> SimpleNamespace:
    return SimpleNamespace(name=name, shape=shape, dtype=dtype)


def _spec(**changes: object) -> SimpleNamespace:
    values = {
        "action_key": "action",
        "action_dim": 19,
        "control_action_dim": 19,
        "horizon": 16,
        "n_obs_steps": 2,
        "n_action_steps": 8,
        "observation_fields": (
            _field("joint_state", (19,), "float32"),
            _field("point_cloud", (1024, 6), "float32"),
        ),
        "control_dt_s": 0.0625,
        "requires_hand": True,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _resolved_runtime(spec: SimpleNamespace | None = None):
    policy_spec = spec or _spec()
    runtime = resolve_runtime_config()
    validate_policy_runtime_compatibility(policy_spec, runtime)
    return runtime


def _observation(spec: SimpleNamespace) -> PolicyObservation:
    count = spec.n_obs_steps
    arrays = {
        field.name: np.zeros((count, *field.shape), dtype=np.dtype(field.dtype))
        for field in spec.observation_fields
    }
    return PolicyObservation(
        observation_id=1,
        run_generation=1,
        anchor_monotonic_ns=20,
        latest_source_monotonic_ns=10,
        logical_step_monotonic_ns=15,
        arrays=arrays,
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
            _spec(observation_fields=(_field("point_cloud", (1024, 6), "float32"),)),
            _spec(
                observation_fields=(
                    _field("joint_state", (19,), "float32"),
                    _field("point_cloud", (1024, 3), "float32"),
                )
            ),
            _spec(n_action_steps=MAX_PREDICTION_STEPS + 1),
            _spec(control_dt_s=0.05),
            _spec(
                observation_fields=(
                    _field("joint_state", (19,), "float32"),
                    _field("point_cloud", (2048, 6), "float32"),
                )
            ),
        )
        for policy_spec in cases:
            with self.subTest(policy_spec=policy_spec), self.assertRaises(ValueError):
                _resolved_runtime(policy_spec)

    def test_command_progress_timeout_cannot_outlive_command_validity(self) -> None:
        with self.assertRaises(ValueError):
            resolve_runtime_config(
                data={
                    "policy": {
                        "action_validity_s": 0.5,
                        "command_progress_timeout_s": 0.6,
                    }
                }
            )
        with self.assertRaises(TypeError):
            resolve_runtime_config(
                data={"policy": {"command_acknowledgement_timeout_s": 0.5}}
            )

    def test_runtime_timing_is_strictly_resolved_and_worker_config_pickles(
        self,
    ) -> None:
        with self.assertRaises(TypeError):
            resolve_runtime_config(data={"policy": {"unknown_timing_s": 1.0}})
        timing_fields = (
            "max_input_age_s",
            "max_observation_skew_s",
            "max_grid_lag_s",
            "max_source_to_command_age_s",
            "max_command_silence_s",
            "action_validity_s",
            "command_progress_timeout_s",
            "first_command_timeout_s",
        )
        for name in timing_fields:
            with self.subTest(name=name), self.assertRaises(ValueError):
                resolve_runtime_config(data={"policy": {name: 0.0}})

        spec = _spec()
        config = PolicyWorkerConfig(
            experiment="dp3/pick/example", device="cpu", spec=spec
        )
        restored = pickle.loads(pickle.dumps(config))
        self.assertEqual(restored.seed, 0)
        self.assertEqual(restored.spec.__dict__, spec.__dict__)


class PolicyAdapterTest(unittest.TestCase):
    def test_rejects_inspect_load_spec_drift(self) -> None:
        inspected_spec = _spec()
        loaded_spec = _spec(n_action_steps=7)
        loaded = _FakeLoadedPolicy(
            loaded_spec,
            np.zeros((7, 19), dtype=np.float64),
        )
        with self.assertRaisesRegex(
            RuntimeError, "PolicySpec changed between inspect and load"
        ):
            DexManiPolicyRuntime(loaded, inspected_spec)

    def test_joint_action_remains_flat_and_observation_is_numpy(self) -> None:
        spec = _spec()
        _resolved_runtime(spec)
        output = np.arange(8 * 19, dtype=np.float64).reshape(8, 19)
        loaded = _FakeLoadedPolicy(spec, output)
        adapter = DexManiPolicyRuntime(loaded, spec)

        actions = adapter.predict(_observation(spec))

        self.assertIs(actions, output)
        np.testing.assert_array_equal(actions, output)
        self.assertEqual(set(loaded.observation), {"joint_state", "point_cloud"})
        self.assertEqual(loaded.observation["joint_state"].dtype, np.float32)

    def test_ee_action_remains_flat(self) -> None:
        spec = _spec(action_key="action_ee", action_dim=21, control_action_dim=21)
        _resolved_runtime(spec)
        output = np.zeros((8, 21), dtype=np.float64)
        output[:, 0] = 0.2
        output[:, 3] = 1.0
        output[:, 7] = 1.0
        adapter = DexManiPolicyRuntime(_FakeLoadedPolicy(spec, output), spec)
        actions = adapter.predict(_observation(spec))
        self.assertIs(actions, output)
        np.testing.assert_array_equal(actions, output)


if __name__ == "__main__":
    unittest.main()
