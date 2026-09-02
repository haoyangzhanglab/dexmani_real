"""Offline startup-order regression tests for policy deployment."""

from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from dexmani_real.deployment.config import DeploymentConfig, PolicyRuntimeConfig
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
        self.execute_completed = _Value(False)
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


def _policy_runtime_config() -> PolicyRuntimeConfig:
    deployment = DeploymentConfig(
        experiment="fake/model",
        device="cpu",
        task_name="fake",
        action_dim=19,
        control_action_dim=19,
        hand_enabled=True,
    )
    return PolicyRuntimeConfig(
        deployment=deployment,
        control_dt_s=deployment.control_dt_s,
        point_cloud_frame="xarm_base",
        point_cloud_color_source="fake",
        point_cloud_policy_id="fake",
        point_cloud_table_plane_abcd_json="null",
        point_cloud_sampling="fake",
        point_cloud_transform="fake",
        hand_acknowledged=True,
    )


class PolicyLifecycleStartupTest(unittest.TestCase):
    def test_model_failure_blocks_all_non_inference_workers(self) -> None:
        shared = _FakeRuntimeChannels()
        runtime = SimpleNamespace(
            policy=SimpleNamespace(control_hz=16.0),
            safety=SimpleNamespace(
                readiness_timeouts_s={
                    name: 1.0
                    for name in (
                        "arm",
                        "camera",
                        "pointcloud",
                        "inference",
                        "policy",
                        "hand",
                    )
                },
                shutdown_timeout_s=1.0,
            ),
        )
        policy_runtime_config = _policy_runtime_config()
        specs = [
            WorkerSpec(name, _unexpected_worker_start, (), ready_name=name)
            for name in ("arm", "camera", "pointcloud")
        ]
        specs.extend(
            (
                WorkerSpec(
                    "inference",
                    inference_loop,
                    (shared, policy_runtime_config),
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
            ),
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
            result = run_policy_deployment(runtime, policy_runtime_config)

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
            shared, [by_name["inference"]], graceful_timeout_s=1.0
        )


def _unexpected_worker_start() -> None:
    raise AssertionError("non-inference worker must not start before inference ready")


if __name__ == "__main__":
    unittest.main()
