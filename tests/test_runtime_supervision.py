"""Offline lifecycle contracts for supervision and verified shutdown."""

from __future__ import annotations

import threading
import time
import unittest

from dexmani_real.runtime.safety import SafetyState
from dexmani_real.runtime.status import ExitReason
from dexmani_real.runtime.supervisor import run_supervisor, wait_subsystem_ready
from dexmani_real.runtime.workers import (
    shutdown_processes_verified,
    supervisor_exit_reason,
)


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
        self.active_coupled_command_sequence = _Value(0)
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
            ["arm", "policy"],
            ["arm"],
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
                [("inference", 0.1), ("arm", 0.1)],
                processes,
            )
        )

    def test_sticky_ready_does_not_mask_dead_worker(self) -> None:
        shared = _Shared()
        shared._ready.add("inference")

        self.assertFalse(
            wait_subsystem_ready(
                shared,
                [("inference", 0.1)],
                [_Process("inference", alive=False, exitcode=1)],
            )
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


if __name__ == "__main__":
    unittest.main()
