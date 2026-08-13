"""Thread helpers and test-config builders shared by the harness tests."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

from dexmani_real.config.runtime import resolve_runtime_config
from dexmani_real.teleop.config import TeleopConfig


def run_in_thread(target: Callable[..., Any], *args: Any) -> threading.Thread:
    """Run *target* as a daemon thread and return it (does not join)."""
    thread = threading.Thread(target=target, args=args, daemon=True)
    thread.start()
    return thread


def stop_loop(shared: Any, thread: threading.Thread, *, timeout_s: float = 8.0) -> None:
    """Request clean shutdown and confirm the loop thread exits."""
    shared.is_running.value = False
    thread.join(timeout=timeout_s)
    assert not thread.is_alive(), f"loop did not exit within {timeout_s}s"


def wait_until(
    predicate: Callable[[], bool],
    *,
    timeout_s: float = 8.0,
    interval_s: float = 0.01,
    description: str = "condition",
) -> None:
    """Poll *predicate* until true, or fail with a descriptive timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval_s)
    raise AssertionError(f"timed out waiting for {description}")


def make_teleop_config(
    *,
    hand_enabled: bool = False,
    recording_enabled: bool = False,
    **overrides: Any,
) -> TeleopConfig:
    """Build a TeleopConfig with the requested runtime overrides."""
    merged: dict[str, Any] = {
        "policy.recording_enabled": recording_enabled,
        "policy.hand_enabled": hand_enabled,
    }
    merged.update(overrides)
    runtime = resolve_runtime_config(cli_overrides=merged)
    return TeleopConfig.from_runtime(runtime, task_label="harness", operator="harness")
