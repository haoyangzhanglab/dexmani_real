"""Shared xArm SDK surface for ``arm_loop.py`` (servo loop) and ``homing.py``.

Holds the resolved ``ArmLoopConfig``, the live-read primitives, and the
controller state-transition leaf helpers (``enter_mode0``/``enter_mode6``/
``stop_controller``).  Neither consumer imports the other for these, so the
dependency graph stays acyclic; none touch ``SharedStorage`` or policy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from dexmani_real.config.defaults import arm, safety


@dataclass
class ArmLoopConfig:
    """Mode 6 joint online trajectory planning configuration."""

    joint_max_speed_rad_per_s: float = field(
        default_factory=lambda: arm.max_joint_velocity_rad_per_s
    )
    joint_max_acc_rad_per_s2: float = field(
        default_factory=lambda: arm.max_joint_acceleration_rad_per_s2
    )
    arm_loop_hz: float = field(default_factory=lambda: arm.loop_hz)

    joint_limit_lower: tuple[float, ...] = field(
        default_factory=lambda: arm.joint_limit_lower
    )
    joint_limit_upper: tuple[float, ...] = field(
        default_factory=lambda: arm.joint_limit_upper
    )

    tracking_error_warn_rad: float = field(
        default_factory=lambda: arm.tracking_error_warn_rad
    )

    arm_ip: str = field(default_factory=lambda: arm.ip)

    home_qpos: tuple[float, ...] = field(default_factory=lambda: arm.home_qpos)

    collision_sensitivity: int = field(
        default_factory=lambda: arm.collision_sensitivity
    )
    max_consecutive_arm_health_failures: int = field(
        default_factory=lambda: safety.max_consecutive_arm_health_failures
    )

    expected_axis: int = field(default_factory=lambda: arm.expected_axis)
    device_profile: str | None = field(default_factory=lambda: arm.device_profile)

    homing_convergence_rad: float = field(
        default_factory=lambda: arm.homing.convergence_rad
    )
    homing_step_interval_s: float = field(
        default_factory=lambda: arm.homing.step_interval_s
    )
    homing_max_speed_rad_per_s: float = field(
        default_factory=lambda: np.deg2rad(arm.homing.max_speed_deg_s)
    )
    homing_target_timeout_s: float = field(
        default_factory=lambda: arm.homing.target_timeout_s
    )
    homing_velocity_convergence_rad_s: float = field(
        default_factory=lambda: arm.homing.velocity_convergence_rad_s
    )
    homing_dwell_s: float = field(default_factory=lambda: arm.homing.dwell_s)

    tcp_load_mass_kg: float = field(default_factory=lambda: arm.tcp_load_mass_kg)
    tcp_load_cog_mm: tuple[float, float, float] = field(
        default_factory=lambda: arm.tcp_load_cog_mm
    )

    @classmethod
    def from_runtime(cls, runtime: Any) -> "ArmLoopConfig":
        cfg = runtime.arm
        return cls(
            joint_max_speed_rad_per_s=float(
                np.deg2rad(cfg.max_joint_velocity_deg_per_s)
            ),
            joint_max_acc_rad_per_s2=float(
                np.deg2rad(cfg.max_joint_acceleration_deg_per_s2)
            ),
            arm_loop_hz=float(cfg.loop_hz),
            joint_limit_lower=tuple(cfg.joint_limit_lower),
            joint_limit_upper=tuple(cfg.joint_limit_upper),
            tracking_error_warn_rad=float(cfg.tracking_error_warn_rad),
            arm_ip=str(cfg.ip),
            home_qpos=tuple(cfg.home_qpos),
            collision_sensitivity=int(cfg.collision_sensitivity),
            max_consecutive_arm_health_failures=int(
                runtime.safety.max_consecutive_arm_health_failures
            ),
            expected_axis=int(cfg.expected_axis),
            device_profile=cfg.device_profile,
            homing_convergence_rad=float(cfg.homing.convergence_rad),
            homing_step_interval_s=float(cfg.homing.step_interval_s),
            homing_max_speed_rad_per_s=float(np.deg2rad(cfg.homing.max_speed_deg_s)),
            homing_target_timeout_s=float(cfg.homing.target_timeout_s),
            homing_velocity_convergence_rad_s=float(
                cfg.homing.velocity_convergence_rad_s
            ),
            homing_dwell_s=float(cfg.homing.dwell_s),
            tcp_load_mass_kg=float(cfg.tcp_load_mass_kg),
            tcp_load_cog_mm=tuple(cfg.tcp_load_cog_mm),
        )


def _require_sdk_ok(operation: str, code: Any) -> None:
    """Raise when an xArm setter reports failure without raising."""
    if not isinstance(code, (int, np.integer)) or int(code) != 0:
        raise RuntimeError(f"{operation} failed with SDK code {code!r}")


def _read_live_error_code(arm_api: Any) -> int:
    """Synchronous live error code; raise on failure (never the cached value)."""
    code, values = arm_api.get_err_warn_code()
    _require_sdk_ok("get_err_warn_code", code)
    return int(values[0])


# Vendor ``ControllerErrorCodeMap`` titles plus a recovery action — a startup
# controller error is never cleared implicitly (see arm_loop).
_CONTROLLER_ERROR_HELP: dict[int, str] = {
    1: "Emergency Stop button pressed — release the E-stop button, then re-enable the robot",
    2: "Emergency IO of the control box triggered — ground the 2 EI pins, then re-enable the robot",
    3: "Three-state switch E-stop pressed — release the three-state switch, then re-enable the robot",
    10: "servo motor error",
    21: "kinematic error",
    22: "self-collision error",
    23: "joints angle exceed limit",
    24: "speed exceeds limit",
    31: "collision caused abnormal current",
}


def describe_controller_error(code: int) -> str:
    """Return a human-readable description for a controller error code."""
    code = int(code)
    if 11 <= code <= 17:
        return f"servo motor {code - 10} error"
    return _CONTROLLER_ERROR_HELP.get(code, "controller error")


@dataclass(frozen=True)
class LiveStateError:
    """Synchronous controller read: state and the (error, warn) register."""

    state: int
    error_code: int
    warn_code: int


@dataclass(frozen=True)
class StopResult:
    """Outcome of a :func:`stop_controller` attempt."""

    confirmed: bool
    reason: str


def read_live_state_and_error(arm_api: Any) -> LiveStateError:
    """Synchronously read controller state and the (error, warn) register."""
    code, state = arm_api.get_state()
    _require_sdk_ok("get_state", code)
    code, values = arm_api.get_err_warn_code()
    _require_sdk_ok("get_err_warn_code", code)
    return LiveStateError(
        state=int(state), error_code=int(values[0]), warn_code=int(values[1])
    )


def _wait_controller_ready(
    arm_api: Any,
    *,
    expected_mode: int,
    on_poll: Callable[[], None] | None,
    timeout_s: float,
) -> int:
    """Bounded wait for error==0, a movable state, and a settled mode.

    ``mode``/``connected`` are cached report attributes (no synchronous
    ``get_mode`` read), so a repeated read is one observation; ``on_poll``
    keeps the caller's heartbeat fresh while this helper sleeps.
    """
    deadline = time.monotonic() + timeout_s
    last: LiveStateError | None = None
    while time.monotonic() < deadline:
        if on_poll is not None:
            on_poll()
        last = read_live_state_and_error(arm_api)
        if (
            bool(getattr(arm_api, "connected", True))
            and last.error_code == 0
            and int(getattr(arm_api, "mode", 6)) == expected_mode
            and int(last.state) in (0, 1, 2)
        ):
            return last.state
        time.sleep(0.03)
    raise RuntimeError(
        f"controller postcondition failed: expected mode={expected_mode} "
        f"error=0 movable-state, got state={last.state if last else None} "
        f"error={last.error_code if last else None}"
    )


def _enter_mode(
    arm_api: Any, mode: int, *, on_poll: Callable[[], None] | None = None
) -> None:
    """Enter a controller mode and wait for a movable state; raise on failure."""
    _require_sdk_ok(f"set_mode({mode})", arm_api.set_mode(mode))
    _require_sdk_ok(f"set_state(0) after Mode {mode}", arm_api.set_state(0))
    _wait_controller_ready(arm_api, expected_mode=mode, on_poll=on_poll, timeout_s=1.0)


def enter_mode0(arm_api: Any, *, on_poll: Callable[[], None] | None = None) -> None:
    """Enter Mode 0 and wait for a movable state; raise on failure."""
    _enter_mode(arm_api, 0, on_poll=on_poll)


def enter_mode6(arm_api: Any, *, on_poll: Callable[[], None] | None = None) -> None:
    """Enter Mode 6 and wait for a movable state; raise on failure."""
    _enter_mode(arm_api, 6, on_poll=on_poll)


def issue_mode_enter(arm_api: Any, mode: int) -> None:
    """Issue the Mode-enter setters without blocking on the postcondition.

    Split from :func:`_enter_mode` for the non-blocking Mode-6 entry: two
    ms-scale SDK RPCs (``set_mode`` + ``set_state(0)``) are issued here, and the
    movable-state postcondition is confirmed on later ticks via
    :func:`mode_enter_ready`.  Raises on a setter failure (fail-closed).
    """
    _require_sdk_ok(f"set_mode({mode})", arm_api.set_mode(mode))
    _require_sdk_ok(f"set_state(0) after Mode {mode}", arm_api.set_state(0))


def mode_enter_ready(arm_api: Any, expected_mode: int) -> bool:
    """One non-blocking probe of the Mode-enter movable-state postcondition.

    Mirrors a single iteration of :func:`_wait_controller_ready` without the
    sleep or deadline: True only when connected, error==0, the cached mode
    matches, and the controller state is movable.  A transient SDK failure is
    caught and reported as not-ready so the caller retries next tick.
    """
    try:
        last = read_live_state_and_error(arm_api)
    except Exception:
        return False
    return (
        bool(getattr(arm_api, "connected", True))
        and last.error_code == 0
        and int(getattr(arm_api, "mode", 6)) == expected_mode
        and int(last.state) in (0, 1, 2)
    )


def stop_controller(
    arm_api: Any,
    *,
    emergency: bool = False,
    on_poll: Callable[[], None] | None = None,
    timeout_s: float = 1.0,
) -> StopResult:
    """Request State 4 and confirm it; a latched error must not block the stop
    confirmation (``emergency=True`` also calls ``emergency_stop`` first)."""
    if emergency:
        try:
            arm_api.emergency_stop()
        except Exception:
            pass
    try:
        _require_sdk_ok("set_state(4)", arm_api.set_state(4))
    except Exception:
        pass
    deadline = time.monotonic() + timeout_s
    last_state: int | None = None
    while time.monotonic() < deadline:
        if on_poll is not None:
            on_poll()
        try:
            code, state = arm_api.get_state()
            _require_sdk_ok("get_state", code)
        except Exception:
            time.sleep(0.03)
            continue
        last_state = int(state)
        if last_state == 4:
            return StopResult(confirmed=True, reason="")
        time.sleep(0.03)
    return StopResult(
        confirmed=False, reason=f"state-4 not confirmed (last={last_state})"
    )


# ``joint_speed_limit`` / ``joint_acc_limit`` are the firmware's *persisted*
# config registers (set_joint_maxacc + save_conf), not hardware bounds.  The
# Mode 6 hard clamps (speed ≤ π, acc ≤ 20) live in the SDK and are enforced by
# ``ArmParams.__post_init__``.
