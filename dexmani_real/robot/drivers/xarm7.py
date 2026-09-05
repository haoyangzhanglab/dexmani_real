"""Thin xArm7 hardware driver — the only xArm SDK surface in the codebase.

One class (:class:`XArm7`) wraps connect / read / servo / home / stop / close.
Error handling is fail-fast: no retry counters, no last-known fallbacks, no
stop confirmation — the firmware is the final backstop.

:meth:`XArm7.home` drives a collision-validated waypoint path in Mode 0 and
raises on failure (a plain :class:`RuntimeError`) or on a runtime interruption
(:class:`HomeAborted`, so the caller can stop without latching a fault).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from dexmani_real.config.defaults import ArmParams
from dexmani_real.robot.model import ARM_JOINT_SHAPE
from dexmani_real.utils.log import (
    capture_native_stdout,
    extract_native_diagnostics,
    get_logger,
)

logger = get_logger(__name__)


# Vendor ``ControllerErrorCodeMap`` titles plus a recovery action — a startup
# controller error is never cleared implicitly.
_CONTROLLER_ERROR_HELP: dict[int, str] = {
    1: "Emergency Stop button pressed — release the E-stop button, then re-enable the robot",
    2: "Emergency IO of the control box triggered — ground the two EI pins, then re-enable the robot",
    3: "Three-state switch E-stop pressed — release the three-state switch, then re-enable the robot",
    10: "servo motor error",
    21: "kinematic error",
    22: "self-collision error",
    23: "joints angle exceed limit",
    24: "speed exceeds limit",
    31: "collision caused abnormal current",
}


@dataclass(frozen=True)
class LiveStateError:
    """Synchronous controller read: state and the (error, warn) register."""

    state: int
    error_code: int
    warn_code: int


def _check_sdk_return_code(code: Any, operation: str) -> None:
    """Raise when an SDK call reports a non-zero return code."""
    if not isinstance(code, (int, np.integer)) or int(code) != 0:
        raise RuntimeError(f"{operation} failed with SDK code {code!r}")


def describe_controller_error(code: int) -> str:
    """Return a human-readable description for a controller error code."""
    code = int(code)
    if 11 <= code <= 17:
        return f"servo motor {code - 10} error"
    return _CONTROLLER_ERROR_HELP.get(code, "controller error")


def read_live_state_and_error(arm_api: Any) -> LiveStateError:
    """Synchronously read controller state and the (error, warn) register."""
    code, state = arm_api.get_state()
    _check_sdk_return_code(code, "get_state")
    code, values = arm_api.get_err_warn_code()
    _check_sdk_return_code(code, "get_err_warn_code")
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


def _enter_mode(arm_api: Any, mode: int, *, on_poll: Callable[[], None] | None = None) -> None:
    """Enter a controller mode and wait for a movable state; raise on failure."""
    _check_sdk_return_code(arm_api.set_mode(mode), f"set_mode({mode})")
    _check_sdk_return_code(arm_api.set_state(0), f"set_state(0) after Mode {mode}")
    _wait_controller_ready(arm_api, expected_mode=mode, on_poll=on_poll, timeout_s=1.0)


def _decode_joint_states(
    code: Any, states: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate position, velocity, and SDK effort at the driver boundary."""
    _check_sdk_return_code(code, "get_joint_states")
    if not isinstance(states, (list, tuple)) or len(states) < 3:
        raise RuntimeError("get_joint_states(num=3) must return position, velocity, and effort")
    qpos = np.asarray(states[0], dtype=np.float64)[: ARM_JOINT_SHAPE[0]].copy()
    qvel = np.asarray(states[1], dtype=np.float64)[: ARM_JOINT_SHAPE[0]].copy()
    tau = np.asarray(states[2], dtype=np.float64)[: ARM_JOINT_SHAPE[0]].copy()
    if any(v.shape != ARM_JOINT_SHAPE or not np.all(np.isfinite(v)) for v in (qpos, qvel, tau)):
        raise RuntimeError("get_joint_states returned invalid qpos/qvel/tau shape or non-finite values")
    return qpos, qvel, tau


def _estimate_segment_timeout_s(
    start: np.ndarray, target: np.ndarray, cfg: ArmParams
) -> float:
    """Deadline for one firmware-planned home milestone, including settle time."""
    delta_rad = float(np.max(np.abs(np.asarray(target) - np.asarray(start))))
    nominal_s = delta_rad / max(cfg.homing.max_speed_rad_per_s, 1e-6)
    return max(
        cfg.homing.target_timeout_s,
        2.0 * nominal_s + cfg.homing.target_timeout_s,
    )


class HomeAborted(RuntimeError):
    """A HOME was interrupted by a runtime signal (e-stop/shutdown/state change).

    Raised from :meth:`XArm7.home` when ``abort_check`` reports a reason.  The
    caller treats it as a clean interruption (best-effort stop, no sticky
    fault), unlike a plain :class:`RuntimeError` from a hardware/SDK failure.
    """


class XArm7:
    """xArm7 driver — the single owner of the controller connection.

    Mode 6 is entered once at the end of :meth:`connect` and held for the
    whole runtime; the only other mode switches are the HOME-path helpers.
    Failures raise: the arm worker's top-level handler latches the sticky
    error, and cleanup does a best-effort stop.
    """

    def __init__(self, cfg: ArmParams) -> None:
        self.cfg = cfg
        self._api: Any = None


    def connect(self, *, on_poll: Callable[[], None] | None = None) -> None:
        """One-shot initialization ending in servo Mode 6.

        Sequence: XArmAPI → axis-count check → controller error check →
        motion_enable → Mode 0 → collision/TCP/acceleration configuration →
        Mode 6 → state 0.
        """
        sdk_connect_output = None
        try:
            with capture_native_stdout() as sdk_connect_output:
                from xarm.wrapper import XArmAPI

                logging.getLogger("origin.print").setLevel(logging.WARNING)
                self._api = XArmAPI(self.cfg.ip, is_radian=True)
        except Exception as exc:
            vendor_detail = (
                sdk_connect_output.text if sdk_connect_output is not None else ""
            )
            raise RuntimeError(
                f"connect failed: {exc}"
                + (f"; vendor output:\n{vendor_detail}" if vendor_detail else "")
            ) from exc
        sdk_diagnostics = extract_native_diagnostics(sdk_connect_output.text)
        if sdk_diagnostics:
            logger.warning(
                "xArm SDK initialization diagnostics:\n%s", "\n".join(sdk_diagnostics)
            )

        self._wait_for_axis_report(on_poll=on_poll)
        self._check_controller_error()
        _check_sdk_return_code(self._api.motion_enable(True), "startup motion_enable")
        self.enter_mode0(on_poll=on_poll)
        self._apply_config()
        self.enter_mode6(on_poll=on_poll)

    def stop(self) -> None:
        """Best-effort State-4 stop; fire-and-forget by design.

        Deliberately no confirmation polling: the firmware is the final
        backstop, and cleanup must never block on proving State 4.
        """
        if self._api is None:
            return
        try:
            self._api.set_state(4)
        except Exception:
            logger.warning("xarm7: set_state(4) failed during stop", exc_info=True)

    def close(self) -> None:
        """Best-effort disconnect."""
        if self._api is None:
            return
        try:
            self._api.disconnect()
        except Exception:
            logger.warning("xarm7: disconnect failed during cleanup", exc_info=True)


    def read(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read joint positions/velocities/efforts [rad, rad/s, Nm]; raise on any failure."""
        code, states = self._api.get_joint_states(is_radian=True, num=3)
        return _decode_joint_states(code, states)

    def servo(self, qpos: np.ndarray) -> int:
        """Send one Mode-6 servo endpoint [rad].

        Returns the SDK return code (0 = accepted); the caller owns the
        non-zero escalation.
        """
        return int(
            self._api.set_servo_angle(
                angle=qpos,
                is_radian=True,
                speed=self.cfg.max_joint_velocity_rad_per_s,
                mvacc=self.cfg.max_joint_acceleration_rad_per_s2,
                wait=False,
            )
        )

    def home(
        self,
        waypoints: np.ndarray,
        final_qpos: np.ndarray,
        *,
        on_poll: Callable[[], None] | None = None,
        feedback_callback: (
            Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None] | None
        ) = None,
        abort_check: Callable[[], str | None] | None = None,
    ) -> None:
        """Drive a collision-validated waypoint path to ``final_qpos`` [rad].

        Blocks the caller: the controller runs each Mode-0 point-to-point move
        (the path was already densely collision-checked by the planner).  The
        controller is left in Mode 6 on a normal return; any failure or
        interruption raises, and the caller's fail-fast handler owns recovery.
        """
        waypoints = np.asarray(waypoints, dtype=np.float64)
        final_qpos = np.asarray(final_qpos, dtype=np.float64)
        if (
            waypoints.ndim != 2
            or waypoints.shape[1:] != ARM_JOINT_SHAPE
            or not np.all(np.isfinite(waypoints))
        ):
            raise ValueError("invalid home waypoint array")
        if final_qpos.shape != ARM_JOINT_SHAPE or not np.all(np.isfinite(final_qpos)):
            raise ValueError("invalid home final_qpos")
        lower = np.asarray(self.cfg.joint_limit_lower, dtype=np.float64)
        upper = np.asarray(self.cfg.joint_limit_upper, dtype=np.float64)
        if len(waypoints) > 0 and not np.all((waypoints >= lower) & (waypoints <= upper)):
            raise ValueError("home waypoint violates joint limits")
        if len(waypoints) > 0 and float(np.max(np.abs(waypoints[-1] - final_qpos))) > 1e-6:
            raise ValueError("final milestone does not match canonical home")

        def _raise_abort() -> None:
            if abort_check is not None:
                reason = abort_check()
                if reason is not None:
                    raise HomeAborted(reason)

        def _converged(target: np.ndarray, q: np.ndarray, v: np.ndarray, tol_rad: float) -> bool:
            return (
                float(np.max(np.abs(q - target))) <= tol_rad
                and float(np.max(np.abs(v)))
                <= self.cfg.homing.velocity_convergence_rad_s
            )

        def _publish(target: np.ndarray, q: np.ndarray, v: np.ndarray, t: np.ndarray) -> None:
            if feedback_callback is not None:
                feedback_callback(q.copy(), v.copy(), t.copy(), target.copy())

        def _dwell(target: np.ndarray) -> None:
            stable_since = time.monotonic()
            while time.monotonic() - stable_since < self.cfg.homing.dwell_s:
                _raise_abort()
                if on_poll is not None:
                    on_poll()
                time.sleep(
                    min(self.cfg.homing.step_interval_s, self.cfg.homing.dwell_s)
                )
                q, v, t = self.read()
                if not _converged(target, q, v, self.cfg.homing.convergence_rad):
                    raise RuntimeError("home dwell interrupted by position/velocity")
                _publish(target, q, v, t)

        _raise_abort()
        qpos, qvel, _tau = self.read()

        if len(waypoints) == 0:
            if not _converged(
                final_qpos, qpos, qvel, self.cfg.homing.convergence_rad
            ):
                raise RuntimeError("empty home path while away from canonical home")
            _dwell(final_qpos)
            return

        if (
            float(np.max(np.abs(qpos - waypoints[0])))
            > self.cfg.homing.convergence_rad
        ):
            raise RuntimeError("current state moved too far from planned path start")

        targets = waypoints[1:]
        if len(targets) == 0:
            _dwell(final_qpos)
            return

        self.enter_mode0(on_poll=on_poll)
        milestone_tol = min(self.cfg.homing.convergence_rad, np.deg2rad(0.5))
        current = qpos
        for index, target in enumerate(targets, start=1):
            _raise_abort()
            if on_poll is not None:
                on_poll()
            code = int(
                self._api.set_servo_angle(
                    angle=target,
                    is_radian=True,
                    speed=self.cfg.homing.max_speed_rad_per_s,
                    mvacc=self.cfg.max_joint_acceleration_rad_per_s2,
                    wait=False,
                    radius=None,
                )
            )
            if code != 0:
                raise RuntimeError(f"home milestone {index} rejected (SDK code={code})")

            deadline = time.monotonic() + _estimate_segment_timeout_s(current, target, self.cfg)
            stable_since: float | None = None
            q = current
            while time.monotonic() < deadline:
                _raise_abort()
                if on_poll is not None:
                    on_poll()
                q, v, t = self.read()
                if self.read_live_error_code() != 0:
                    raise RuntimeError(f"controller error at home milestone {index}")
                _publish(target, q, v, t)
                if _converged(target, q, v, milestone_tol):
                    if stable_since is None:
                        stable_since = time.monotonic()
                    if time.monotonic() - stable_since >= self.cfg.homing.dwell_s:
                        break
                else:
                    stable_since = None
                time.sleep(self.cfg.homing.step_interval_s)
            else:
                error = np.abs(q - target)
                joint = int(np.argmax(error))
                raise RuntimeError(
                    f"home milestone {index} convergence timeout "
                    f"(J{joint + 1} error={np.rad2deg(error[joint]):.2f}deg)"
                )
            current = q.copy()

        final_error = float(np.max(np.abs(current - final_qpos)))
        if final_error > self.cfg.homing.convergence_rad:
            raise RuntimeError(f"home final error {np.rad2deg(final_error):.2f}deg")
        self.enter_mode6(on_poll=on_poll)

    def emergency_stop(self) -> None:
        """Best-effort emergency stop (requests State 4 without cutting power)."""
        try:
            self._api.emergency_stop()
        except Exception:
            logger.warning(
                "xarm7: emergency_stop call failed; cleanup will enforce state 4",
                exc_info=True,
            )


    def enter_mode0(self, *, on_poll: Callable[[], None] | None = None) -> None:
        """Enter Mode 0 (MoveJoint) and wait for a movable state; raise on failure."""
        _enter_mode(self._api, 0, on_poll=on_poll)

    def enter_mode6(self, *, on_poll: Callable[[], None] | None = None) -> None:
        """Enter servo Mode 6 and wait for a movable state; raise on failure."""
        _enter_mode(self._api, 6, on_poll=on_poll)

    def read_live_error_code(self) -> int:
        """Synchronous live error code; raise on failure (never the cached value)."""
        code, values = self._api.get_err_warn_code()
        _check_sdk_return_code(code, "get_err_warn_code")
        return int(values[0])


    @property
    def api(self) -> Any:
        """Raw XArmAPI handle; the homing path drives milestones through it."""
        return self._api

    @property
    def axis(self) -> int:
        return int(getattr(self._api, "axis", 0) or 0)

    @property
    def mode(self) -> int:
        value = getattr(self._api, "mode", None)
        return int(value) if value is not None else 0

    @property
    def state(self) -> Any:
        return getattr(self._api, "state", None)

    @property
    def error_code(self) -> int:
        return int(getattr(self._api, "error_code", 0) or 0)


    def _wait_for_axis_report(self, *, on_poll: Callable[[], None] | None) -> None:
        """Wait for the report thread's axis count and validate it."""
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if self.axis > 0:
                break
            if on_poll is not None:
                on_poll()
            time.sleep(0.05)
        if self.axis != self.cfg.expected_axis:
            raise RuntimeError(
                f"device reports {self.axis} axes, expected {self.cfg.expected_axis}"
            )

    def _check_controller_error(self) -> None:
        """Refuse to proceed on a pre-existing controller error; warn is diagnostic."""
        live = read_live_state_and_error(self._api)  # raises on SDK failure
        if live.error_code != 0:
            raise RuntimeError(
                f"startup controller error C{live.error_code} (warn={live.warn_code}, "
                f"state={live.state}): {describe_controller_error(live.error_code)} — "
                "refusing to clear"
            )
        if live.warn_code != 0:
            logger.warning(
                "xarm7: startup controller warn=%d (diagnostic only)", live.warn_code
            )

    def _apply_config(self) -> None:
        """Apply collision sensitivity, TCP load, and the Mode 6 joint acc cap."""
        _check_sdk_return_code(
            self._api.set_collision_sensitivity(self.cfg.collision_sensitivity),
            "set_collision_sensitivity",
        )
        _check_sdk_return_code(
            self._api.set_tcp_load(
                weight=self.cfg.tcp_load_mass_kg,
                center_of_gravity=list(self.cfg.tcp_load_cog_mm),
            ),
            "set_tcp_load",
        )
        _check_sdk_return_code(
            self._api.set_joint_maxacc(
                self.cfg.max_joint_acceleration_rad_per_s2,
                is_radian=True,
            ),
            "set_joint_maxacc",
        )
