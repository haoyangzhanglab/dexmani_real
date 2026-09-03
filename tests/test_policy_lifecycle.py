"""Offline startup-order regression tests for policy deployment."""

from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from dexmani_real.config.runtime import resolve_runtime_config
from dexmani_real.deployment.config import PolicyWorkerConfig
from dexmani_real.deployment.lifecycle import run_policy_deployment
from dexmani_real.deployment.worker import inference_loop
from dexmani_real.runtime.safety import SafetyState
from dexmani_real.runtime.workers import WorkerSpec


class _Value:
    def __init__(self, value: int | bool) -> None:
        self.value = value


class _FakeRuntimeChannels:
    """Minimal lifecycle state without shared memory or worker processes."""

    def __init__(self) -> None:
        self.error_state = _Value(False)
        self.estop_request = _Value(False)
        self.is_running = _Value(True)
        self.safety_state = _Value(int(SafetyState.DISARMED))
        self.run_generation = _Value(0)
        self.run_started_monotonic_ns = _Value(0)
        self.active_coupled_command_sequence = _Value(0)
        self.motion_lock = threading.Lock()
        self._ready_names: set[str] = set()
        self._heartbeats: dict[str, float] = {}

    def set_heartbeat(self, name: str, timestamp_s: float) -> None:
        self._heartbeats[name] = timestamp_s

    def set_ready(self, name: str) -> None:
        self._ready_names.add(name)

    def is_ready(self, name: str) -> bool:
        return name in self._ready_names

    def close(self) -> bool:
        return True


class _FakeProcess:
    """Synchronous process double that records a child startup failure."""

    def __init__(self, spec: WorkerSpec) -> None:
        self.name = spec.name
        self._target = spec.target
        self._args = spec.args
        self.started = False
        self.exitcode: int | None = None
        self._alive = False

    def start(self) -> None:
        self.started = True
        self._alive = True
        try:
            self._target(*self._args)
        except BaseException:
            self.exitcode = 1
        else:
            self.exitcode = 0
        finally:
            self._alive = False

    def is_alive(self) -> bool:
        return self._alive


def _policy_spec() -> SimpleNamespace:
    return SimpleNamespace(
        action_key="action",
        action_dim=19,
        control_action_dim=19,
        horizon=16,
        n_obs_steps=2,
        n_action_steps=8,
        sensor_modalities=("joint_state", "point_cloud"),
        point_cloud_num_points=1024,
        point_cloud_feature_dim=6,
        control_dt_s=0.0625,
        requires_hand=True,
    )


def _policy_worker_config(spec: SimpleNamespace) -> PolicyWorkerConfig:
    return PolicyWorkerConfig(
        experiment="fake/model",
        device="cpu",
        spec=spec,
    )


class PolicyLifecycleStartupTest(unittest.TestCase):
    def test_model_failure_blocks_all_non_inference_workers(self) -> None:
        shared = _FakeRuntimeChannels()
        runtime = resolve_runtime_config()
        spec = _policy_spec()
        worker_config = _policy_worker_config(spec)
        specs = [
            WorkerSpec(name, _unexpected_worker_start, (), ready_name=name)
            for name in ("arm", "camera", "pointcloud")
        ]
        specs.extend(
            (
                WorkerSpec(
                    "inference",
                    inference_loop,
                    (shared, runtime.policy, worker_config),
                    ready_name="inference",
                ),
                WorkerSpec("policy", _unexpected_worker_start, (), ready_name="policy"),
                WorkerSpec("hand", _unexpected_worker_start, (), ready_name="hand"),
            )
        )
        processes: list[_FakeProcess] = []

        def build_fake_processes(
            _context: object, worker_specs: list[WorkerSpec]
        ) -> list[_FakeProcess]:
            processes.extend(_FakeProcess(spec) for spec in worker_specs)
            return processes

        shutdown = Mock(return_value=object())
        with (
            patch(
                "dexmani_real.deployment.lifecycle.RuntimeChannels.create",
                return_value=shared,
            ),
            patch(
                "dexmani_real.deployment.lifecycle.RuntimeChannelsConfig.from_runtime",
                return_value=object(),
            ) as channel_config,
            patch(
                "dexmani_real.deployment.lifecycle.build_policy_worker_specs",
                return_value=specs,
            ),
            patch(
                "dexmani_real.deployment.lifecycle.build_processes",
                side_effect=build_fake_processes,
            ),
            patch("dexmani_real.deployment.lifecycle.shutdown_processes", shutdown),
            patch(
                "dexmani_real.deployment.worker._load_inference_runtime",
                side_effect=RuntimeError("fake model restore failed"),
            ),
        ):
            result = run_policy_deployment(runtime, spec, worker_config, False)

        by_name = {process.name: process for process in processes}
        self.assertEqual(result, 1)
        self.assertFalse(shared.is_ready("inference"))
        self.assertTrue(shared.error_state.value)
        self.assertEqual(shared.safety_state.value, int(SafetyState.FAULT))
        self.assertTrue(by_name["inference"].started)
        for name in ("arm", "hand", "camera", "pointcloud", "policy"):
            with self.subTest(worker=name):
                self.assertFalse(by_name[name].started)
        shutdown.assert_called_once_with(
            shared,
            [by_name["inference"]],
            graceful_timeout_s=float(runtime.safety.shutdown_timeout_s),
        )
        channel_config.assert_called_once_with(
            runtime,
            pointcloud_num_points=spec.point_cloud_num_points,
            camera_requested=True,
            pointcloud_requested=True,
            observation_horizon=spec.n_obs_steps,
            observation_dt_s=spec.control_dt_s,
            max_input_age_s=runtime.policy.max_input_age_s,
            max_observation_skew_s=runtime.policy.max_observation_skew_s,
            max_grid_lag_s=runtime.policy.max_grid_lag_s,
        )


def _unexpected_worker_start() -> None:
    raise AssertionError("non-inference worker must not start before inference ready")


if __name__ == "__main__":
    unittest.main()
