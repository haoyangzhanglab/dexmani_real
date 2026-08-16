"""Shared xArm SDK surface for the arm servo loop and homing execution.

Holds the pieces both ``arm_loop.py`` (servo loop) and ``homing.py`` (homing
execution) need without either owning them: the resolved ``ArmLoopConfig``, the
live-read primitives, and the controller state-transition leaf helpers
(``enter_mode0`` / ``enter_mode6`` / ``stop_controller``).  ``arm_loop`` imports
from here, and so does ``homing``; neither imports the other for these, so the
dependency graph stays acyclic.  None of these helpers read or write
``SharedStorage`` or make policy decisions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from dexmani_real.config.defaults import arm, safety
from dexmani_real.utils.schema import ARM_JOINT_SHAPE


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
    serial_number: str | None = field(default_factory=lambda: arm.serial_number)
    min_firmware: tuple[int, ...] | None = field(
        default_factory=lambda: arm.min_firmware
    )

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

    def __post_init__(self) -> None:
        lower = np.asarray(self.joint_limit_lower, dtype=np.float64)
        upper = np.asarray(self.joint_limit_upper, dtype=np.float64)
        home = np.asarray(self.home_qpos, dtype=np.float64)
        if (
            lower.shape != ARM_JOINT_SHAPE
            or upper.shape != ARM_JOINT_SHAPE
            or home.shape != ARM_JOINT_SHAPE
        ):
            raise ValueError(
                f"arm loop joint limits/home must have shape {ARM_JOINT_SHAPE}"
            )
        if not np.all(np.isfinite(np.concatenate((lower, upper, home)))) or np.any(
            lower > upper
        ):
            raise ValueError("arm loop joint limits/home must be finite and ordered")
        if self.max_consecutive_arm_health_failures <= 0:
            raise ValueError("max_consecutive_arm_health_failures must be positive")
        if not isinstance(self.expected_axis, int) or self.expected_axis <= 0:
            raise ValueError("expected_axis must be a positive integer")
        if self.device_profile is not None and not self.device_profile:
            raise ValueError("device_profile must be non-empty when set")
        if self.serial_number is not None and not self.serial_number:
            raise ValueError("serial_number must be non-empty when set")
        if self.min_firmware is not None and (
            not isinstance(self.min_firmware, tuple)
            or not self.min_firmware
            or any(not isinstance(v, int) or v < 0 for v in self.min_firmware)
        ):
            raise ValueError(
                "min_firmware must be a non-empty tuple of non-negative integers"
            )
        timing = (
            self.joint_max_speed_rad_per_s,
            self.joint_max_acc_rad_per_s2,
            self.arm_loop_hz,
            self.tracking_error_warn_rad,
            self.homing_convergence_rad,
            self.homing_step_interval_s,
            self.homing_max_speed_rad_per_s,
            self.homing_target_timeout_s,
            self.homing_velocity_convergence_rad_s,
            self.homing_dwell_s,
        )
        if not all(np.isfinite(value) and value > 0 for value in timing):
            raise ValueError(
                "arm loop motion/homing parameters must be finite and positive"
            )
        if not self.arm_ip or not (0 <= self.collision_sensitivity <= 5):
            raise ValueError("arm loop IP/collision sensitivity is invalid")
        if not np.isfinite(self.tcp_load_mass_kg) or self.tcp_load_mass_kg <= 0:
            raise ValueError("arm loop tcp_load_mass_kg must be finite and positive")
        cog = np.asarray(self.tcp_load_cog_mm, dtype=np.float64)
        if cog.shape != (3,) or not np.all(np.isfinite(cog)):
            raise ValueError("arm loop tcp_load_cog_mm must be a finite (3,) vector")

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
            serial_number=cfg.serial_number,
            min_firmware=tuple(cfg.min_firmware) if cfg.min_firmware is not None else None,
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
    """Return the live controller error code; raise if the live read fails.

    Unlike the cached ``arm.error_code`` property (updated by a background
    report thread), this synchronously reads ``get_err_warn_code``.  Control
    decisions that follow a setter failure or a homing run must use this and
    treat a raise as a fault — never fall back to the cached value.
    """
    code, values = arm_api.get_err_warn_code()
    _require_sdk_ok("get_err_warn_code", code)
    return int(values[0])


# ── Controller state transitions (leaf helpers shared by arm_loop + homing) ──
#
# ``get_state()`` / ``get_err_warn_code()`` are synchronous reads; ``arm.mode``
# and ``arm.connected`` are report-cache attributes (SDK 1.18.4 has no
# synchronous ``get_mode()``).  The two are deliberately separate result types:
# never present a combined read as one atomic live snapshot.


@dataclass(frozen=True)
class LiveStateError:
    """Synchronous controller read: state and the (error, warn) register."""

    state: int
    error_code: int
    warn_code: int


@dataclass(frozen=True)
class ReportSnapshot:
    """Cached report attributes: connected flag and controller mode."""

    connected: bool
    mode: int


@dataclass(frozen=True)
class StopResult:
    """Outcome of a :func:`stop_controller` attempt."""

    confirmed: bool
    reason: str


def controller_state_allows_motion(state: Any) -> bool:
    """Return whether a controller state permits motion (0/1/2).

    State 2 is a legal idle state but not the only ready state; the fixed SDK
    accepts 0/1/2 as "not suspended/stopped".  Unknown states fail closed.
    """
    return int(state) in (0, 1, 2)


def read_live_state_and_error(arm_api: Any) -> LiveStateError:
    """Synchronously read controller state and the (error, warn) register."""
    code, state = arm_api.get_state()
    _require_sdk_ok("get_state", code)
    code, values = arm_api.get_err_warn_code()
    _require_sdk_ok("get_err_warn_code", code)
    return LiveStateError(
        state=int(state), error_code=int(values[0]), warn_code=int(values[1])
    )


def read_report_mode_and_connection(arm_api: Any) -> ReportSnapshot:
    """Read the cached ``connected``/``mode`` report attributes."""
    connected = bool(getattr(arm_api, "connected", True))
    mode = int(getattr(arm_api, "mode", 6))
    return ReportSnapshot(connected=connected, mode=mode)


def _wait_controller_ready(
    arm_api: Any,
    *,
    expected_mode: int,
    on_poll: Callable[[], None] | None,
    timeout_s: float,
) -> int:
    """Bounded wait for ``error==0``, a movable state, and a settled mode.

    A repeated read of the same cached ``mode`` is a single observation, not
    several independent live samples; only the (possibly updated) report value
    at each poll is evaluated.  ``on_poll`` keeps the caller's heartbeat fresh
    while this helper sleeps.
    """
    deadline = time.monotonic() + timeout_s
    last: LiveStateError | None = None
    while time.monotonic() < deadline:
        if on_poll is not None:
            on_poll()
        last = read_live_state_and_error(arm_api)
        report = read_report_mode_and_connection(arm_api)
        if (
            report.connected
            and last.error_code == 0
            and report.mode == expected_mode
            and controller_state_allows_motion(last.state)
        ):
            return last.state
        time.sleep(0.03)
    raise RuntimeError(
        f"controller postcondition failed: expected mode={expected_mode} "
        f"error=0 movable-state, got state={last.state if last else None} "
        f"error={last.error_code if last else None}"
    )


def enter_mode0(arm_api: Any, *, on_poll: Callable[[], None] | None = None) -> None:
    """Enter Mode 0 and wait for a movable state; raise on failure."""
    _require_sdk_ok("set_mode(0)", arm_api.set_mode(0))
    _require_sdk_ok("set_state(0) after Mode 0", arm_api.set_state(0))
    _wait_controller_ready(
        arm_api, expected_mode=0, on_poll=on_poll, timeout_s=1.0
    )


def enter_mode6(arm_api: Any, *, on_poll: Callable[[], None] | None = None) -> None:
    """Enter Mode 6 and wait for a movable state; raise on failure."""
    _require_sdk_ok("set_mode(6)", arm_api.set_mode(6))
    _require_sdk_ok("set_state(0)", arm_api.set_state(0))
    _wait_controller_ready(
        arm_api, expected_mode=6, on_poll=on_poll, timeout_s=1.0
    )


def stop_controller(
    arm_api: Any,
    *,
    emergency: bool = False,
    on_poll: Callable[[], None] | None = None,
    timeout_s: float = 1.0,
) -> StopResult:
    """Request State 4 and confirm it, without requiring a cleared error.

    ``emergency=True`` first calls ``arm.emergency_stop()`` (no integer return
    code in SDK 1.18.4; only exceptions are caught).  A failed ``set_state(4)``
    does not skip the bounded synchronous ``get_state()`` confirmation: a
    latched controller error must not prevent confirming the physical stop.
    """
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


# ── Connect-time validation (config vs. the SDK report surface) ──
#
# ``set_servo_angle`` silently clamps speed to [0.0001, π] rad/s and mvacc to
# [0.01, 20] rad/s² (SDK 1.18.4), so a configured value outside the effective
# intersection of those hard clamps and the reported device limits would be
# rewritten without any error.  Refuse such values at connect instead of
# letting metadata claim an execution value the firmware cannot honor.

SPEED_COMMAND_CLAMP_RAD_PER_S: tuple[float, float] = (0.0001, float(np.pi))
ACC_COMMAND_CLAMP_RAD_PER_S2: tuple[float, float] = (0.01, 20.0)


def _binding_device_limit(value: Any) -> float | None:
    """Return the most restrictive positive finite reported device limit.

    ``joint_speed_limit`` / ``joint_acc_limit`` are per-joint report lists; the
    binding constraint is their minimum.  Returns ``None`` when the report is
    absent or unusable so the caller falls back to the SDK hard clamp only.
    """
    if value is None:
        return None
    if np.isscalar(value):
        value = [value]
    try:
        finite = [float(v) for v in value if np.isfinite(float(v)) and float(v) > 0.0]
    except (TypeError, ValueError):
        return None
    return min(finite) if finite else None


def validate_command_dynamics_intersection(
    *,
    config_speed_rad_per_s: float,
    config_acc_rad_per_s2: float,
    device_speed_limits: Any,
    device_acc_limits: Any,
) -> str | None:
    """Return a reason string when the configured Mode 6 dynamics would be
    silently clamped, or ``None`` when they fit the effective command range.

    The effective upper bound is ``min(SDK hard clamp, reported device limit)``;
    a missing device report falls back to the SDK hard clamp alone.  Both below
    the lower clamp and above the effective upper bound are refused.
    """
    speed_upper = SPEED_COMMAND_CLAMP_RAD_PER_S[1]
    acc_upper = ACC_COMMAND_CLAMP_RAD_PER_S2[1]
    dev_speed = _binding_device_limit(device_speed_limits)
    dev_acc = _binding_device_limit(device_acc_limits)
    if dev_speed is not None:
        speed_upper = min(speed_upper, dev_speed)
    if dev_acc is not None:
        acc_upper = min(acc_upper, dev_acc)
    if not (
        SPEED_COMMAND_CLAMP_RAD_PER_S[0] <= config_speed_rad_per_s <= speed_upper
    ):
        return (
            f"configured speed {config_speed_rad_per_s} rad/s outside the effective "
            f"command range [{SPEED_COMMAND_CLAMP_RAD_PER_S[0]}, {speed_upper}]"
        )
    if not (
        ACC_COMMAND_CLAMP_RAD_PER_S2[0] <= config_acc_rad_per_s2 <= acc_upper
    ):
        return (
            f"configured mvacc {config_acc_rad_per_s2} rad/s² outside the effective "
            f"command range [{ACC_COMMAND_CLAMP_RAD_PER_S2[0]}, {acc_upper}]"
        )
    return None


def validate_device_identity(
    *,
    axis: int,
    device_type: str,
    serial_number: str,
    firmware: tuple[int, ...],
    expected_axis: int,
    expected_serial: str | None,
    min_firmware: tuple[int, ...] | None,
    device_profile: str | None,
) -> str | None:
    """Return a reason string when the reported device identity fails the
    configured expectations, or ``None`` when it passes.

    ``firmware`` is compared as an integer tuple (``version_number``), never as
    a string.  ``device_profile=None`` performs no type check — the model is not
    guessed without on-site confirmation (doc §8.5).  ``expected_serial=None``
    and ``min_firmware=None`` skip their checks.
    """
    if axis != expected_axis:
        return f"device reports {axis} axes, expected {expected_axis}"
    if expected_serial is not None and serial_number != expected_serial:
        return (
            f"serial number mismatch: got {serial_number!r}, "
            f"expected {expected_serial!r}"
        )
    if min_firmware is not None and firmware and firmware < min_firmware:
        return f"firmware {firmware} below minimum {min_firmware}"
    if device_profile is not None and device_type and device_type != device_profile:
        return (
            f"device type {device_type!r} does not match profile {device_profile!r}"
        )
    return None
