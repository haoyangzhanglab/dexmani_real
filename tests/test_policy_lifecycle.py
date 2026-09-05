"""Offline startup-order regression tests for policy deployment."""

from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from dexmani_real.config.runtime import resolve_runtime_config
from dexmani_real.deployment.config import (
    PolicyDeploymentConfig,
    PolicyWorkerConfig,
    validate_policy_runtime_compatibility,
)
from dexmani_real.deployment.lifecycle import (
    build_policy_worker_specs,
    run_policy_deployment,
)
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


class _ControllableProcess:
    """Non-spawning process double for lifecycle ordering tests."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.started = False
        self._alive = False

    def is_alive(self) -> bool:
        return self._alive


class _StartFailProcess(_ControllableProcess):
    """Process double whose start can fail before the child is launched."""

    def __init__(self, name: str, events: list[str], *, fail: bool = False) -> None:
        super().__init__(name)
        self._events = events
        self._fail = fail

    def start(self) -> None:
        self._events.append(f"start:{self.name}")
        if self._fail:
            raise RuntimeError(f"could not start {self.name}")
        self.started = True
        self._alive = True


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
    def test_deployment_mode_reaches_inference_and_executor(self) -> None:
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
            max_running_s=1.5,
        )

        by_name = {worker.name: worker for worker in specs}
        self.assertIs(by_name["arm"].args[1], runtime.arm)
        self.assertIs(by_name["inference"].args[-2], deployment)
        self.assertIsNone(by_name["inference"].args[-1])
        self.assertIs(by_name["policy"].args[1], runtime)
        self.assertIs(by_name["policy"].args[2], spec)
        self.assertIs(by_name["policy"].args[3], deployment)
        self.assertFalse(by_name["policy"].args[4])
        self.assertEqual(by_name["policy"].args[5], 1.5)
        self.assertIsNone(by_name["policy"].ready_name)

    def test_model_failure_blocks_all_non_inference_workers(self) -> None:
        shared = _FakeRuntimeChannels()
        runtime = resolve_runtime_config()
        spec = _policy_spec()
        worker_config = _policy_worker_config(spec)
        processes: list[_FakeProcess] = []
        startup_order: list[str] = []

        def build_fake_processes(
            _context: object, worker_specs: list[WorkerSpec]
        ) -> list[_FakeProcess]:
            processes.extend(_FakeProcess(spec) for spec in worker_specs)
            return processes

        def validate_once(policy_spec: object, resolved_runtime: object) -> None:
            startup_order.append("validate")
            validate_policy_runtime_compatibility(policy_spec, resolved_runtime)

        def create_channels_side_effect(**_kwargs: object) -> _FakeRuntimeChannels:
            startup_order.append("create")
            return shared

        shutdown = Mock(return_value=object())
        with (
            patch(
                "dexmani_real.deployment.lifecycle.RuntimeChannels.create",
                side_effect=create_channels_side_effect,
            ) as create_channels,
            patch(
                "dexmani_real.deployment.lifecycle.validate_policy_runtime_compatibility",
                side_effect=validate_once,
            ) as validate_compatibility,
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
        self.assertEqual(startup_order, ["validate", "create"])
        validate_compatibility.assert_called_once_with(spec, runtime)
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

    def test_ready_worker_death_before_armed_faults_deployment(self) -> None:
        shared = _FakeRuntimeChannels()
        runtime = resolve_runtime_config()
        spec = _policy_spec()
        worker_config = _policy_worker_config(spec)
        specs = (
            WorkerSpec(
                "inference", _unexpected_worker_start, (), ready_name="inference"
            ),
            WorkerSpec("arm", _unexpected_worker_start, (), ready_name="arm"),
            WorkerSpec("hand", _unexpected_worker_start, (), ready_name="hand"),
            WorkerSpec("policy", _unexpected_worker_start, ()),
        )
        processes = [_ControllableProcess(worker.name) for worker in specs]
        by_name = {process.name: process for process in processes}
        specs_by_name = {worker.name: worker for worker in specs}

        def start_fake(selected: list[_ControllableProcess]) -> None:
            if by_name["inference"].started and by_name["inference"] not in selected:
                by_name["inference"]._alive = False
            for process in selected:
                process.started = True
                process._alive = True
                ready_name = specs_by_name[process.name].ready_name
                if ready_name is not None:
                    shared.set_ready(ready_name)

        shutdown = Mock(return_value=object())
        with (
            patch(
                "dexmani_real.deployment.lifecycle.RuntimeChannels.create",
                return_value=shared,
            ),
            patch(
                "dexmani_real.deployment.lifecycle.build_policy_worker_specs",
                return_value=specs,
            ),
            patch(
                "dexmani_real.deployment.lifecycle.build_processes",
                return_value=processes,
            ),
            patch("dexmani_real.deployment.lifecycle.start_processes", start_fake),
            patch("dexmani_real.deployment.lifecycle.shutdown_processes", shutdown),
        ):
            result = run_policy_deployment(runtime, spec, worker_config, False)

        self.assertEqual(result, 1)
        self.assertTrue(shared.error_state.value)
        self.assertEqual(shared.safety_state.value, int(SafetyState.FAULT))
        self.assertTrue(by_name["inference"].started)
        for name in ("arm", "hand", "policy"):
            with self.subTest(worker=name):
                self.assertTrue(by_name[name].started)
        shutdown.assert_called_once_with(
            shared,
            [by_name[name] for name in ("inference", "arm", "hand", "policy")],
            graceful_timeout_s=float(runtime.safety.shutdown_timeout_s),
        )

    def test_partial_remaining_start_failure_shuts_down_only_started_children(
        self,
    ) -> None:
        shared = _FakeRuntimeChannels()
        runtime = resolve_runtime_config()
        spec = _policy_spec()
        worker_config = _policy_worker_config(spec)
        specs = (
            WorkerSpec(
                "inference", _unexpected_worker_start, (), ready_name="inference"
            ),
            WorkerSpec("arm", _unexpected_worker_start, (), ready_name="arm"),
            WorkerSpec("hand", _unexpected_worker_start, (), ready_name="hand"),
            WorkerSpec("policy", _unexpected_worker_start, ()),
        )
        events: list[str] = []
        processes = [
            _StartFailProcess("inference", events),
            _StartFailProcess("arm", events),
            _StartFailProcess("hand", events, fail=True),
            _StartFailProcess("policy", events),
        ]
        by_name = {process.name: process for process in processes}

        def close_channels() -> bool:
            events.append("close")
            return True

        shared.close = close_channels  # type: ignore[method-assign]

        def verified_shutdown(
            channels: _FakeRuntimeChannels,
            selected: list[_StartFailProcess],
            **_kwargs: object,
        ) -> object:
            self.assertIs(channels, shared)
            self.assertEqual(
                selected,
                [by_name["inference"], by_name["arm"]],
            )
            self.assertNotIn("close", events)
            events.append("shutdown")
            channels.close()
            return object()

        with (
            patch(
                "dexmani_real.deployment.lifecycle.RuntimeChannels.create",
                return_value=shared,
            ),
            patch(
                "dexmani_real.deployment.lifecycle.build_policy_worker_specs",
                return_value=specs,
            ),
            patch(
                "dexmani_real.deployment.lifecycle.build_processes",
                return_value=processes,
            ),
            patch(
                "dexmani_real.deployment.lifecycle.wait_subsystem_ready",
                return_value=True,
            ) as wait_ready,
            patch(
                "dexmani_real.deployment.lifecycle.shutdown_processes",
                side_effect=verified_shutdown,
            ) as shutdown,
        ):
            result = run_policy_deployment(runtime, spec, worker_config, False)

        self.assertEqual(result, 1)
        self.assertEqual(
            events,
            [
                "start:inference",
                "start:arm",
                "start:hand",
                "shutdown",
                "close",
            ],
        )
        self.assertTrue(by_name["inference"].started)
        self.assertTrue(by_name["arm"].started)
        self.assertFalse(by_name["hand"].started)
        self.assertFalse(by_name["policy"].started)
        wait_ready.assert_called_once()
        shutdown.assert_called_once_with(
            shared,
            [by_name["inference"], by_name["arm"]],
            graceful_timeout_s=float(runtime.safety.shutdown_timeout_s),
        )

    def test_lifecycle_supervises_policy_heartbeat(self) -> None:
        shared = _FakeRuntimeChannels()
        runtime = resolve_runtime_config()
        spec = _policy_spec()
        worker_config = _policy_worker_config(spec)
        specs = (
            WorkerSpec(
                "inference", _unexpected_worker_start, (), ready_name="inference"
            ),
            WorkerSpec("arm", _unexpected_worker_start, (), ready_name="arm"),
            WorkerSpec("hand", _unexpected_worker_start, (), ready_name="hand"),
            WorkerSpec("policy", _unexpected_worker_start, ()),
        )
        processes = [_ControllableProcess(worker.name) for worker in specs]
        by_name = {process.name: process for process in processes}
        specs_by_name = {worker.name: worker for worker in specs}

        def start_fake(selected: list[_ControllableProcess]) -> None:
            for process in selected:
                process.started = True
                process._alive = True
                ready_name = specs_by_name[process.name].ready_name
                if ready_name is not None:
                    shared.set_ready(ready_name)

        def clean_shutdown(*_args: object, **_kwargs: object) -> SimpleNamespace:
            shared.is_running.value = False
            shared.safety_state.value = int(SafetyState.DISARMED)
            return SimpleNamespace(
                exits=[
                    SimpleNamespace(exitcode=0, escalation="graceful")
                    for _process in processes
                ],
                shared_closed=True,
            )

        supervisor = Mock(return_value=("explicit quit", True))
        with (
            patch(
                "dexmani_real.deployment.lifecycle.RuntimeChannels.create",
                return_value=shared,
            ),
            patch(
                "dexmani_real.deployment.lifecycle.build_policy_worker_specs",
                return_value=specs,
            ),
            patch(
                "dexmani_real.deployment.lifecycle.build_processes",
                return_value=processes,
            ),
            patch("dexmani_real.deployment.lifecycle.start_processes", start_fake),
            patch("dexmani_real.deployment.lifecycle.run_operator_control"),
            patch("dexmani_real.deployment.lifecycle.run_supervisor", supervisor),
            patch(
                "dexmani_real.deployment.lifecycle.shutdown_processes",
                side_effect=clean_shutdown,
            ),
        ):
            result = run_policy_deployment(runtime, spec, worker_config, False)

        self.assertEqual(result, 0)
        heartbeat_timeouts = supervisor.call_args.kwargs["heartbeat_timeouts_s"]
        self.assertIn("policy", heartbeat_timeouts)
        self.assertEqual(
            set(heartbeat_timeouts), {"inference", "arm", "hand", "policy"}
        )


def _unexpected_worker_start() -> None:
    raise AssertionError("non-inference worker must not start before inference ready")


if __name__ == "__main__":
    unittest.main()
