"""Offline lifecycle contracts for supervision and verified shutdown."""

from __future__ import annotations

import sys
import threading
import time
import types
import unittest
from itertools import count
from types import SimpleNamespace
from unittest.mock import Mock, patch

from dexmani_real.calibration.camera.session import run_camera_calibration
from dexmani_real.config.defaults import CameraParams
from dexmani_real.runtime.safety import SafetyState
from dexmani_real.runtime.status import ExitReason
from dexmani_real.runtime.supervisor import run_supervisor, wait_subsystem_ready
from dexmani_real.runtime.workers import (
    WorkerSpec,
    shutdown_processes_verified,
    supervisor_exit_reason,
)
from dexmani_real.sensor.camera_worker import CameraLoopConfig, camera_loop


class _Value:
    def __init__(self, value: int | bool) -> None:
        self.value = value


class _Shared:
    def __init__(self, *, quit_requested: bool = False) -> None:
        self.is_running = _Value(True)
        self.quit_requested = _Value(quit_requested)
        self.estop_request = _Value(False)
        self.error_state = _Value(False)
        self.safety_state = _Value(int(SafetyState.ARMED))
        self.run_generation = _Value(0)
        self.run_started_monotonic_ns = _Value(0)
        self.motion_lock = threading.Lock()
        self._ready: set[str] = set()
        self._heartbeats: dict[str, float] = {}
        self.closed = False

    def get_heartbeat(self, name: str) -> float:
        return self._heartbeats.get(name, 0.0)

    def is_ready(self, name: str) -> bool:
        return name in self._ready

    def close(self) -> bool:
        self.closed = True
        return True


class _Process:
    def __init__(
        self,
        name: str,
        *,
        alive: bool = True,
        exitcode: int | None = None,
        terminate_exitcode: int = -15,
    ) -> None:
        self.name = name
        self._alive = alive
        self.exitcode = exitcode
        self._terminate_exitcode = terminate_exitcode
        self.terminate_called = False
        self.kill_called = False

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: float) -> None:
        del timeout

    def terminate(self) -> None:
        self.terminate_called = True
        self._alive = False
        self.exitcode = self._terminate_exitcode

    def kill(self) -> None:
        self.kill_called = True
        self._alive = False
        self.exitcode = -9


class RuntimeSupervisionTest(unittest.TestCase):
    def test_sustained_camera_read_failure_faults_worker(self) -> None:
        class FakeRealSenseConfig:
            def __init__(self, **kwargs: object) -> None:
                self.frame_queue_capacity = kwargs["frame_queue_capacity"]

        class FakeRealSense:
            instance: "FakeRealSense | None" = None

            def __init__(self, config: FakeRealSenseConfig) -> None:
                type(self).instance = self
                self.config = config
                self.active_serial = "fake-camera"
                self.read = Mock(side_effect=RuntimeError("read failed"))
                self.disconnect = Mock()

            def connect(self) -> bool:
                return True

            def get_depth_scale(self) -> float:
                return 0.001

            def get_device_info(self) -> dict[str, str]:
                return {"firmware": "fake"}

            def get_geometry(self) -> SimpleNamespace:
                return SimpleNamespace(to_dict=lambda: {})

            def get_active_profiles(self) -> list[object]:
                return []

            def get_l515_depth_option_snapshot(self) -> dict[str, object]:
                return {}

        fake_realsense = types.ModuleType("dexmani_real.sensor.realsense")
        fake_realsense.RealSense = FakeRealSense
        fake_realsense.RealSenseConfig = FakeRealSenseConfig
        fake_realsense.L515DepthConfig = lambda **_kwargs: object()
        shared = SimpleNamespace(
            camera_depth_scale=_Value(0.0),
            camera_serial=_Value(b""),
            camera_firmware=_Value(b""),
            camera_sdk_version=_Value(b""),
            camera_geometry=_Value(b""),
            camera_profile=_Value(b""),
            is_running=_Value(True),
            safety_state=_Value(int(SafetyState.DISARMED)),
            is_recording=_Value(False),
            camera_requested=_Value(False),
            error_state=_Value(False),
            set_heartbeat=Mock(),
        )
        monotonic_values = count(1.0, 0.06)

        with (
            patch.dict(sys.modules, {fake_realsense.__name__: fake_realsense}),
            patch("dexmani_real.sensor.camera_worker.time.sleep"),
            patch(
                "dexmani_real.sensor.camera_worker.time.monotonic",
                side_effect=lambda: next(monotonic_values),
            ),
        ):
            camera_loop(
                shared,
                CameraLoopConfig(
                    max_frame_age_s=0.05,
                    read_failure_timeout_s=0.1,
                ),
            )

        self.assertTrue(shared.error_state.value)
        camera = FakeRealSense.instance
        assert camera is not None
        self.assertEqual(camera.read.call_count, 3)
        camera.disconnect.assert_called_once_with()

    def test_camera_deadlines_reject_non_finite_and_misordered_values(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf"), 0.0):
            with self.subTest(config="runtime", field="max_frame_age_s", value=value):
                with self.assertRaises(ValueError):
                    CameraParams(max_frame_age_s=value)
            with self.subTest(
                config="runtime", field="recording_stall_abort_s", value=value
            ):
                with self.assertRaises(ValueError):
                    CameraParams(recording_stall_abort_s=value)
            with self.subTest(config="worker", field="max_frame_age_s", value=value):
                with self.assertRaises(ValueError):
                    CameraLoopConfig(max_frame_age_s=value)
            with self.subTest(
                config="worker", field="read_failure_timeout_s", value=value
            ):
                with self.assertRaises(ValueError):
                    CameraLoopConfig(read_failure_timeout_s=value)

        with self.assertRaises(ValueError):
            CameraParams(max_frame_age_s=0.25, recording_stall_abort_s=0.25)
        with self.assertRaises(ValueError):
            CameraLoopConfig(max_frame_age_s=0.25, read_failure_timeout_s=0.25)

    def test_camera_calibration_start_error_reaps_started_worker(self) -> None:
        shared = SimpleNamespace(
            estop_request=_Value(False),
            error_state=_Value(False),
            safety_state=_Value(int(SafetyState.DISARMED)),
            close=Mock(return_value=True),
        )
        process = SimpleNamespace(pid=None)
        runtime = SimpleNamespace(
            arm=SimpleNamespace(loop_hz=100.0),
            safety=SimpleNamespace(
                readiness_timeouts_s={"arm": 1.0},
                shutdown_timeout_s=0.1,
            ),
        )

        def fail_after_start(processes: list[SimpleNamespace]) -> None:
            processes[0].pid = 123
            raise RuntimeError("start failed")

        with (
            patch(
                "dexmani_real.calibration.camera.session._build_planner_and_gate",
                return_value=(object(), object(), object()),
            ),
            patch("dexmani_real.calibration.camera.session.mp.get_context"),
            patch(
                "dexmani_real.calibration.camera.session.RuntimeChannelsConfig.from_runtime"
            ),
            patch(
                "dexmani_real.calibration.camera.session.RuntimeChannels.create",
                return_value=shared,
            ),
            patch(
                "dexmani_real.calibration.camera.session.build_processes",
                return_value=[process],
            ),
            patch(
                "dexmani_real.calibration.camera.session.start_processes",
                side_effect=fail_after_start,
            ),
            patch(
                "dexmani_real.calibration.camera.session.shutdown_processes",
                return_value=SimpleNamespace(exits=(), shared_closed=True),
            ) as shutdown,
        ):
            with self.assertRaisesRegex(RuntimeError, "start failed"):
                run_camera_calibration(
                    runtime,
                    camera_serial=None,
                    hand_geometry="absent",
                )

        shutdown.assert_called_once()
        self.assertEqual(shutdown.call_args.args[:2], (shared, [process]))
        shared.close.assert_not_called()

    def test_camera_calibration_build_error_closes_unowned_channels(self) -> None:
        shared = SimpleNamespace(close=Mock(return_value=True))
        runtime = SimpleNamespace(
            arm=SimpleNamespace(loop_hz=100.0),
            safety=SimpleNamespace(
                readiness_timeouts_s={"arm": 1.0},
                shutdown_timeout_s=0.1,
            ),
        )
        with (
            patch(
                "dexmani_real.calibration.camera.session._build_planner_and_gate",
                return_value=(object(), object(), object()),
            ),
            patch("dexmani_real.calibration.camera.session.mp.get_context"),
            patch(
                "dexmani_real.calibration.camera.session.RuntimeChannelsConfig.from_runtime"
            ),
            patch(
                "dexmani_real.calibration.camera.session.RuntimeChannels.create",
                return_value=shared,
            ),
            patch(
                "dexmani_real.calibration.camera.session.build_processes",
                side_effect=RuntimeError("build failed"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "build failed"):
                run_camera_calibration(
                    runtime,
                    camera_serial=None,
                    hand_geometry="absent",
                )

        shared.close.assert_called_once_with()

    def test_exit_reason_prioritizes_death_and_timeout_before_quit(self) -> None:
        shared = _Shared(quit_requested=True)
        dead = _Process("camera", alive=False, exitcode=0)
        self.assertIs(
            supervisor_exit_reason(shared, [dead], {}, {}),
            ExitReason.WORKER_DEATH,
        )

        live = _Process("arm")
        self.assertIs(
            supervisor_exit_reason(
                shared,
                [live],
                {"arm": float("inf")},
                {"arm": 1.0},
            ),
            ExitReason.HEARTBEAT_TIMEOUT,
        )

        self.assertIs(
            supervisor_exit_reason(shared, [live], {}, {}),
            ExitReason.EXPLICIT_QUIT,
        )

    def test_stalled_policy_heartbeat_is_a_runtime_timeout(self) -> None:
        shared = _Shared()

        self.assertIs(
            supervisor_exit_reason(
                shared,
                [_Process("policy")],
                {"policy": float("inf")},
                {"policy": 1.0},
            ),
            ExitReason.HEARTBEAT_TIMEOUT,
        )

    def test_supervisor_allows_heartbeat_subset_of_processes(self) -> None:
        shared = _Shared(quit_requested=True)
        shared._heartbeats["arm"] = time.monotonic()
        reason, normal = run_supervisor(
            shared,
            [_Process("arm"), _Process("policy")],
            heartbeat_timeouts_s={"arm": 1.0},
            supervisor_hz=10.0,
        )

        self.assertEqual(reason, "explicit quit")
        self.assertTrue(normal)

    def test_readiness_checks_only_requested_subsystems(self) -> None:
        shared = _Shared()
        shared._ready.update({"inference", "arm"})
        processes = [_Process("inference"), _Process("arm"), _Process("policy")]

        self.assertTrue(
            wait_subsystem_ready(
                shared,
                [
                    (
                        WorkerSpec("inference", lambda: None, (), "inference"),
                        processes[0],
                    ),
                    (WorkerSpec("arm", lambda: None, (), "arm"), processes[1]),
                ],
                {"inference": 0.1, "arm": 0.1},
                monitored_processes=processes,
            )
        )

    def test_sticky_ready_does_not_mask_dead_worker(self) -> None:
        shared = _Shared()
        shared._ready.add("inference")

        self.assertFalse(
            wait_subsystem_ready(
                shared,
                [
                    (
                        WorkerSpec("inference", lambda: None, (), "inference"),
                        _Process("inference", alive=False, exitcode=1),
                    )
                ],
                {"inference": 0.1},
            )
        )

    def test_readiness_fails_on_sticky_fault_and_timeout(self) -> None:
        worker = WorkerSpec("arm", lambda: None, (), "arm")
        process = _Process("arm")

        faulted = _Shared()
        faulted.error_state.value = True
        self.assertFalse(
            wait_subsystem_ready(faulted, [(worker, process)], {"arm": 0.1})
        )

        timed_out = _Shared()
        self.assertFalse(
            wait_subsystem_ready(timed_out, [(worker, process)], {"arm": 1e-6})
        )

    def test_shutdown_report_records_verified_escalation_and_shared_close(self) -> None:
        shared = _Shared()
        graceful = _Process("inference", alive=False, exitcode=0)
        terminated = _Process("arm")

        report = shutdown_processes_verified(
            shared,
            [graceful, terminated],
            graceful_timeout_s=0.0,
        )

        self.assertFalse(shared.is_running.value)
        self.assertTrue(terminated.terminate_called)
        self.assertTrue(report.shared_closed)
        self.assertEqual(
            [(item.name, item.exitcode, item.escalation) for item in report.exits],
            [("inference", 0, "graceful"), ("arm", -15, "terminate")],
        )
        self.assertFalse(hasattr(report, "clean"))
        self.assertTrue(shared.closed)
        self.assertTrue(shared.error_state.value)
        self.assertEqual(shared.safety_state.value, int(SafetyState.FAULT))

    def test_ipc_cleanup_error_after_verified_stop_is_not_a_physical_fault(
        self,
    ) -> None:
        class CleanupErrorShared(_Shared):
            def close(self) -> bool:
                self.closed = True
                return False

        shared = CleanupErrorShared()
        stopped = _Process("arm", alive=False, exitcode=0)

        report = shutdown_processes_verified(
            shared,
            [stopped],
            graceful_timeout_s=0.0,
            disarm_if_clean=True,
        )

        self.assertFalse(report.shared_closed)
        self.assertTrue(shared.closed)
        self.assertFalse(shared.error_state.value)
        self.assertEqual(shared.safety_state.value, int(SafetyState.DISARMED))


if __name__ == "__main__":
    unittest.main()
