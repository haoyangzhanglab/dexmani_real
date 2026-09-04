"""Offline startup-order regression tests for policy deployment."""

from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from dexmani_real.config.runtime import resolve_runtime_config
from dexmani_real.deployment.config import PolicyDeploymentConfig, PolicyWorkerConfig
from dexmani_real.deployment.lifecycle import (
    build_policy_worker_specs,
    run_policy_deployment,
)
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
        observation_fields=(
            SimpleNamespace(name="joint_state", shape=(19,), dtype="float32"),
            SimpleNamespace(name="point_cloud", shape=(1024, 6), dtype="float32"),
        ),
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
    def test_deployment_mode_reaches_inference_and_coordinator(self) -> None:
        runtime = resolve_runtime_config()
        spec = _policy_spec()
        worker_config = _policy_worker_config(spec)
        deployment = PolicyDeploymentConfig(
            inference_mode="async",
            max_action_steps=3,
        )

        specs = build_policy_worker_specs(
            object(),
            runtime,
            spec,
            worker_config,
            execute=False,
            deployment_config=deployment,
        )

        by_name = {worker.name: worker for worker in specs}
        self.assertIs(by_name["arm"].args[1], runtime.arm)
        self.assertIs(by_name["inference"].args[-2], deployment)
        self.assertIsNone(by_name["inference"].args[-1])
        self.assertEqual(by_name["policy"].args[-1].inference_mode, "async")
        self.assertEqual(by_name["policy"].args[-1].max_action_steps, 3)
        self.assertEqual(
            by_name["policy"].args[-1].coordinator_hz,
            runtime.policy.coordinator_hz,
        )
        self.assertEqual(
            by_name["policy"].args[-1].command_progress_timeout_s,
            runtime.policy.command_progress_timeout_s,
        )

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
            ) as create_channels,
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
        final_config = create_channels.call_args.kwargs["config"]
        self.assertTrue(final_config.camera_requested)
        self.assertTrue(final_config.pointcloud_requested)
        self.assertEqual(
            final_config.pointcloud_num_points,
            spec.observation_fields[1].shape[0],
        )
        self.assertEqual(
            (
                final_config.arm_state_ring_maxlen,
                final_config.hand_state_ring_maxlen,
                final_config.hand_tactile_ring_maxlen,
                final_config.camera_ring_maxlen,
                final_config.pointcloud_ring_maxlen,
            ),
            (14, 14, 14, 11, 11),
        )


def _unexpected_worker_start() -> None:
    raise AssertionError("non-inference worker must not start before inference ready")


if __name__ == "__main__":
    unittest.main()
