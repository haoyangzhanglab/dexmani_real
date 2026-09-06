"""Offline regressions for teleoperation shutdown and recording admission."""

from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.config.experiment import resolve_experiment_config
from dexmani_real.runtime.operator_input import OperatorCommand
from dexmani_real.runtime.safety import SafetyState
from dexmani_real.runtime.status import ExitReason
from dexmani_real.runtime.supervisor import supervisor_exit_reason
from dexmani_real.teleop.config import TeleopConfig
from dexmani_real.teleop.loop import teleop_loop
from dexmani_real.teleop.session import run_teleop_experiment


class _Value:
    def __init__(self, value: bool | int) -> None:
        self.value = value


class _QuitWaitShared:
    """Minimal teleop-loop state without shared memory or a worker process."""

    def __init__(self) -> None:
        self.error_state = _Value(False)
        self.estop_request = _Value(False)
        self.is_running = _Value(True)
        self.quit_requested = _Value(False)
        self.is_recording = _Value(False)
        self.safety_state = _Value(int(SafetyState.ARMED))
        self.run_generation = _Value(0)
        self.run_started_monotonic_ns = _Value(0)
        self.motion_lock = threading.Lock()
        self._heartbeats: dict[str, float] = {}
        self._ready_names: set[str] = set()

    def set_heartbeat(self, name: str, timestamp_s: float) -> None:
        self._heartbeats[name] = timestamp_s

    def set_ready(self, name: str) -> None:
        self._ready_names.add(name)


class _QuitWaitRate:
    """Make the parent shutdown after the child has remained alive one tick."""

    def __init__(self, shared: _QuitWaitShared) -> None:
        self.shared = shared
        self.wait_calls = 0

    def wait(self) -> None:
        self.wait_calls += 1
        if self.wait_calls == 3:
            self.shared.is_running.value = False

    def reset(self) -> None:
        pass


class _Keyboard:
    def __init__(self) -> None:
        self._poll_count = 0

    def poll(self, *, timeout: float) -> list[OperatorCommand]:
        self._poll_count += 1
        if self._poll_count <= 2:
            return [OperatorCommand.QUIT]
        return []

    def stop(self) -> None:
        pass


class _Audio:
    def play(self, _name: str) -> None:
        pass

    def wait_until_idle(self, *, timeout_s: float) -> bool:
        return True

    def close(self) -> None:
        pass


class TeleopLifecycleTest(unittest.TestCase):
    def test_quit_waits_for_parent_shutdown_and_preserves_explicit_quit(self) -> None:
        runtime = resolve_experiment_config(
            data={"policy": {"hand_enabled": False, "recording_enabled": False}}
        )
        config = TeleopConfig.from_runtime(runtime)
        shared = _QuitWaitShared()
        rate = _QuitWaitRate(shared)
        controller = SimpleNamespace(
            hand_enabled=False,
            prev_hand_qpos=np.zeros(12, dtype=np.float64),
            clear_reference=Mock(),
        )

        with (
            patch(
                "dexmani_real.teleop.loop._load_control_resources",
                return_value=(Mock(), Mock(), Mock(), None),
            ),
            patch(
                "dexmani_real.teleop.loop._load_hand_kinematics", return_value=None
            ),
            patch("dexmani_real.teleop.loop._start_keyboard", return_value=_Keyboard()),
            patch("dexmani_real.teleop.loop.AudioFeedback", return_value=_Audio()),
            patch("dexmani_real.teleop.loop.TeleopController", return_value=controller),
            patch("dexmani_real.teleop.loop.LoopRate", return_value=rate),
            patch("dexmani_real.teleop.loop.read_arm_state_causal", return_value=None),
            patch("dexmani_real.teleop.loop.read_hand_state_causal", return_value=None),
            patch("dexmani_real.teleop.loop.signal.signal"),
        ):
            teleop_loop(shared, config)

        self.assertEqual(rate.wait_calls, 3)
        self.assertTrue(shared.quit_requested.value)
        self.assertIn("policy", shared._heartbeats)
        self.assertIs(
            supervisor_exit_reason(
                shared,
                [SimpleNamespace(exitcode=None)],
                {},
                {},
            ),
            ExitReason.EXPLICIT_QUIT,
        )

    def test_recording_without_hand_rejects_before_any_startup(self) -> None:
        runtime = resolve_experiment_config(
            data={"policy": {"hand_enabled": False, "recording_enabled": True}}
        )

        with (
            patch(
                "dexmani_real.teleop.session.load_vr_transform",
                side_effect=AssertionError("startup preflight must not run"),
            ) as load_vr_transform,
            patch("dexmani_real.teleop.session.RuntimeChannels.create") as create_channels,
        ):
            result = run_teleop_experiment(runtime, allow_no_hand=True)

        self.assertEqual(result, 1)
        load_vr_transform.assert_not_called()
        create_channels.assert_not_called()


if __name__ == "__main__":
    unittest.main()
