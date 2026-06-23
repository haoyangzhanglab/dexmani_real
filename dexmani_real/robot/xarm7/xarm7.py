"""xArm7 7-DOF robot arm hardware driver.

Two control modes (ref: BunnyVisionPro xarm7_ability.py):

  Mode 1 (servo, default): position servo via set_servo_angle_j.
      Controller calls send_action(qpos) → direct position command.

  Mode 4 (velocity, PID inner loop): velocity control via vc_set_joint_velocity.
      Controller calls send_action(qpos) → stores target.
      250Hz inner thread: reads target → PID → velocity clip → hardware.

Mode 4 produces smoother motion because the PID converts position error
into continuous velocity, eliminating step-jump artifacts.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from xarm.wrapper import XArmAPI

from dexmani_real.log import get_logger
from dexmani_real.robot._connection_state import ConnectionStateMixin
from dexmani_real.utils.array_utils import nan_array, safe_resize
from dexmani_real.utils.rate_limiter import RateLimiter
from dexmani_real.utils.serialization import from_dict_helper

logger = get_logger(__name__)


# ===========================================================================
# SDK "mode may be incorrect" warning — belt-and-suspenders suppression
# ===========================================================================
# The xArm SDK's base.py:2169-2170 emits a WARNING when vc_set_joint_velocity
# is called while self.mode (SDK-cached) != 4.  The root cause is fixed by the
# _velocity_control_active gate in _pid_loop_impl + _set_mode, which ensures
# velocity commands are only sent when the arm is actually in mode 4.
#
# This Python-level suppression remains as defense-in-depth for any edge case
# where the SDK cached mode is briefly stale after a mode switch.
# ===========================================================================


def _suppress_sdk_mode_warnings(arm: Any) -> None:
    """Suppress the xArm SDK's mode-check warning at the Python logging layer.

    Called AFTER XArmAPI() creation (SDK sets up logging during init).
    This is belt-and-suspenders — the primary fix is the mode gate in the
    PID inner loop that prevents incorrect vc_set_joint_velocity calls.
    """
    import logging
    import warnings

    # Walk registered loggers + scan arm instance for Logger attributes
    suppressed = set()
    for logger_name in list(logging.root.manager.loggerDict):
        if "xarm" in logger_name.lower() or "sdk" in logger_name.lower():
            lg = logging.getLogger(logger_name)
            lg.setLevel(logging.ERROR)
            for h in lg.handlers:
                h.setLevel(logging.ERROR)
            for f in lg.filters:
                lg.removeFilter(f)
            suppressed.add(logger_name)
    for attr in dir(arm):
        try:
            obj = getattr(arm, attr)
        except (AttributeError, RuntimeError):
            continue
        if isinstance(obj, logging.Logger):
            obj.setLevel(logging.ERROR)
            for h in obj.handlers:
                h.setLevel(logging.ERROR)
    if not suppressed:
        for name in ("SDK", "xarm", "xarm.wrapper.xarm_api", "xarm.xarm"):
            logging.getLogger(name).setLevel(logging.ERROR)
    logger.debug("SDK warning suppression: %d loggers silenced", len(suppressed))

    # Warnings module
    warnings.filterwarnings("ignore", message=".*mode may be incorrect.*")
    warnings.filterwarnings("ignore", message=r".*The mode may be incorrect.*")


# ===========================================================================
# PID Controller (ref: BunnyVisionPro xarm7_ability.py PIDController)
# ===========================================================================


class PIDController:
    """Per-joint PID controller for joint-space velocity control.

    Operates on 7-DOF joint position error::

        vel = Kp * err + Kd * (err - prev_err) / dt + Ki * cum_err

    Integral term (Ki) is disabled by default to prevent windup in
    teleoperation where the target is continuously changing.

    ref: BunnyVisionPro xarm7_ability.py:11-36
    """

    def __init__(
        self,
        kp: np.ndarray,
        ki: np.ndarray | None = None,
        kd: np.ndarray | None = None,
    ) -> None:
        self.kp = np.asarray(kp, dtype=np.float64)
        self.ki = (
            np.asarray(ki, dtype=np.float64)
            if ki is not None
            else np.zeros_like(self.kp)
        )
        self.kd = (
            np.asarray(kd, dtype=np.float64)
            if kd is not None
            else self.kp / 20.0
        )
        self._prev_err: np.ndarray | None = None
        self._cum_err: np.ndarray = np.zeros_like(self.kp)

    def reset(self) -> None:
        """Clear error history (call on mode switch or arm reset)."""
        self._prev_err = None
        self._cum_err = np.zeros_like(self.kp)

    def control(self, err: np.ndarray, dt: float) -> np.ndarray:
        """Compute velocity command from position error.

        Args:
            err: Joint position error (target - current), shape (7,).
            dt: Time step in seconds (inner loop period).

        Returns:
            Joint velocity command, shape (7,).
        """
        err = np.asarray(err, dtype=np.float64)
        if self._prev_err is None:
            self._prev_err = err.copy()

        # P + D + I
        value = (
            self.kp * err
            + self.kd * (err - self._prev_err) / dt
            + self.ki * self._cum_err
        )

        self._prev_err = err.copy()
        self._cum_err += dt * err

        return value


@dataclass
class XArm7Config:
    ip: str = "192.168.1.111"
    dt: float = 1.0 / 50.0
    init_qpos: np.ndarray = field(
        default_factory=lambda: np.deg2rad([-30, -45, 0, 20, -180, 25, 0])
    )
    qpos_min: np.ndarray = field(
        default_factory=lambda: np.deg2rad([-360, -118, -360, -11, -360, -97, -360])
    )
    qpos_max: np.ndarray = field(
        default_factory=lambda: np.deg2rad([360, 120, 360, 225, 360, 180, 360])
    )
    max_qvel: np.ndarray = field(
        default_factory=lambda: np.deg2rad([90, 90, 90, 90, 120, 120, 150])
    )
    reset_speed: float = np.deg2rad(20)
    reset_acc: float = np.deg2rad(180)
    # NOTE: use_delta_limit (step bottleneck) and clip_joint_limit (range clamp)
    # are kept as separate flags even though both default to True and there is no
    # known use case for enabling only one. The separation preserves explicit
    # per-behavior control — hardware-range protection and per-step-speed limiting
    # are conceptually independent concerns.
    use_delta_limit: bool = True
    clip_joint_limit: bool = True
    # Threshold-based convergence gate (ref: BunnyVisionPro teleop_bimanual_xarm7_ability.py:144-175).
    # When > 0, velocity is limited to 30% of pid_max_vel until ALL joint errors
    # drop below this threshold (rad), then immediately released to 100%.
    # Pure state-driven — no time-based ramp, fully deterministic & reproducible.
    # When <= 0, soft-start is disabled (skip directly to full speed).
    # Reduced from 2.0°→1.0° for dexterous teleop: faster convergence detection
    # while still preventing initial snap on IDLE→TELEOP transitions.
    pid_convergence_threshold_rad: float = np.deg2rad(1.0)
    # When use_servo_control=False, a 250Hz inner thread runs PID + velocity control
    # via xArm Mode 4 (vc_set_joint_velocity). When True, uses the
    # existing Mode 1 position servo (backward compatible).
    # Default changed to False (PID velocity mode) for smoother motion.
    use_servo_control: bool = False
    # Mode 6 (joint-space trajectory planning): when True, uses xArm controller
    # firmware's built-in trajectory smoother instead of host-side PID.
    # Offloads trajectory smoothing to the controller, reducing host CPU load.
    inner_control_dt: float = 1.0 / 250.0  # 250 Hz inner loop
    pid_kp: np.ndarray = field(
        default_factory=lambda: np.array([7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0])
    )  # Uniform Kp — preserves Cartesian trajectory shape in velocity mode.
    # Raised from 5.0→7.0 for dexterous teleop: improves tracking responsiveness
    # (faster error correction) while maintaining straight-line path fidelity.
    # The xArm internal servo handles inertia-dependent torque.
    pid_kd: np.ndarray = field(
        default_factory=lambda: np.array([0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35])
    )  # Kd = Kp / 20
    pid_max_vel: np.ndarray = field(
        default_factory=lambda: np.array([1.2, 1.2, 1.2, 1.2, 1.6, 1.6, 2.0])
    )  # rad/s, ~80% of max_qvel — balances PID overshoot headroom with speed parity
    # Prior value [0.8, 0.8, 0.8, 0.8, 1.0, 1.0, 1.5] was ~50% of max_qvel,
    # directly inherited from BunnyVisionPro (which had no servo-mode baseline).
    # Raised to match the hardware capability established by max_qvel.

    # Collision detection params — prevent false C31 triggering
    # tcp_load_kg: XHand weight (kg); non-zero enables correct dynamics torque estimation
    tcp_load_kg: float = 1.2
    # tcp_load_cog_mm: load center of gravity [x, y, z] mm, relative to flange frame
    tcp_load_cog_mm: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 80.0]
    )
    # collision_sensitivity: 0=disabled, 1=least sensitive, 5=most sensitive. Factory default is 3.
    # NOTE: Set to 0 (disabled) for teleoperation — even with correct TCP load configured,
    # rapid teleop motions can still trigger false C31. 0 is the most reliable choice.
    # ref: ufactory_teleop uf_robot.py:137
    collision_sensitivity: int = 0



class XArm7(ConnectionStateMixin):
    def __init__(self, config: XArm7Config):
        super().__init__()
        self.config = config
        self.arm: XArmAPI | None = None

        self.last_sdk_error_code: int = 0  # SDK-level error code for C31/C32 recovery

        self.last_qpos_cmd: np.ndarray | None = None
        self.last_cmd_time: float | None = None
        self.last_joint_limit_clipped = False
        self.last_delta_limited = False
        self._vel_ramp_start: float | None = None  # velocity soft-start timer (perf_counter)
        self._pid_converged: bool = False  # threshold-based convergence flag (B3)

        # ── PID inner-loop state (velocity control mode) ──
        self._arm_thread: threading.Thread | None = None
        self._arm_lock: threading.Lock = threading.Lock()
        self._arm_pos_target: np.ndarray | None = None
        self._arm_should_stop: threading.Event = threading.Event()
        # Gate flag: when False, the PID inner loop skips velocity commands.
        # Managed by _set_mode() — cleared when mode transitions away from 4,
        # set when mode returns to 4.  Prevents calling vc_set_joint_velocity
        # during mode-0 blocking moves (reset/return_to_home), which would
        # flood the SDK's "mode may be incorrect" warning at 250 Hz.
        self._velocity_control_active: bool = True
        if not self.config.use_servo_control:
            self._arm_pid = PIDController(
                kp=self.config.pid_kp,
                kd=self.config.pid_kd,
            )

    def connect(self) -> bool:
        if self.connected_flag and self.arm is not None:
            return True

        try:
            self.arm = XArmAPI(self.config.ip, is_radian=True)
        except (OSError, ConnectionError, RuntimeError) as e:
            self.error_state = True
            self.last_error_message = f"XArmAPI init failed: {e}"
            return False

        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(True)

        # Suppress SDK "mode may be incorrect" warnings when using velocity
        # control mode (Mode 4).  The xArm SDK's base.py:2170 performs a
        # conservative mode check and emits a WARNING on every vc_set_joint_velocity
        # call (250 Hz), flooding the log.  The arm operates correctly; this is
        # a false-positive diagnostic.
        #
        # This warning may come from Python logging or directly from the C/C++
        # SDK layer (fprintf to stderr).  Multi-layer suppression:
        #   1. Python logging: walk all loggers with "xarm"/"SDK" in name,
        #      plus scan the XArmAPI instance for Logger attributes
        #   2. Python warnings module
        #   3. sys.stderr filter (activated in PID inner loop thread)
        _suppress_sdk_mode_warnings(self.arm)

        # Full init sequence (mode switch, collision params, verification)
        # ref: ufactory_teleop uf_robot.py:139-187
        self.robot_init()

        state = self.get_state()
        if np.all(np.isfinite(state["qpos"])):
            self.last_qpos_cmd = state["qpos"].copy()
        else:
            self.last_qpos_cmd = self.config.init_qpos.copy()

        self.last_cmd_time = time.time()
        self.connected_flag = True
        self.error_state = False

        # Start PID inner thread for velocity control mode
        if not self.config.use_servo_control:
            self._arm_should_stop.clear()
            self._arm_pos_target = self.last_qpos_cmd
            self._vel_ramp_start = time.perf_counter()  # activate velocity soft-start
            self._arm_thread = threading.Thread(
                target=self._internal_control_arm_qpos,
                name="xarm7_pid_inner",
                daemon=True,
            )
            self._arm_thread.start()
            logger.info(
                "PID inner loop started at %.0f Hz (mode 4, velocity control)",
                1.0 / self.config.inner_control_dt,
            )

        return True

    def disconnect(self) -> None:
        # Stop PID inner thread before disconnecting hardware
        self._stop_inner_thread()
        if self.arm is not None:
            self.arm.disconnect()
        self.connected_flag = False

    def is_connected(self) -> bool:
        return self.arm is not None and self.connected_flag and not self.error_state

    def is_error(self) -> bool:
        if self.arm is None:
            return True
        if not self.connected_flag:
            return True
        if self.error_state:
            return True
        if self.arm.error_code != 0:
            return True
        return False

    def clear_error(self) -> bool:
        if self.arm is None:
            return False
        # Re-run full init sequence (clean, mode switch, collision params)
        self.robot_init()
        # Reset PID to prevent error accumulation from pre-error state
        if not self.config.use_servo_control:
            self._arm_pid.reset()
            with self._arm_lock:
                self._arm_pos_target = self._read_qpos()
        # Clear error state AFTER robot_init (which may set it on init failure)
        self.error_state = False
        self.last_error_message = ""
        return self.arm.error_code == 0

    def stop(self) -> bool:
        if self.arm is None:
            return False
        # Send zero velocity before stopping (velocity control mode)
        if not self.config.use_servo_control:
            try:
                self.arm.vc_set_joint_velocity(np.zeros(7, dtype=np.float64))
            except (RuntimeError, OSError):
                pass
        self._stop_inner_thread()
        code = self.arm.set_state(4)
        self.error_state = True
        return code == 0

    def get_state(self, full: bool = False) -> dict[str, Any]:
        code, states = self.arm.get_joint_states(is_radian=True, num=3)
        if code == 0:
            qpos = self._array7(states[0])
            qvel = self._array7(states[1])
            tau = self._array7(states[2])
        else:
            qpos = self._read_qpos()
            qvel = nan_array(7)
            tau = nan_array(7)

        state: dict[str, Any] = {
            "qpos": qpos,
            "qvel": qvel,
            "tau": tau,
            "timestamp": time.time(),
        }

        if full:
            state.update({
                "mode": self.arm.mode,
                "state": self.arm.state,
                "connected": self.arm.connected,
                "error_code": self.arm.error_code,
                "warn_code": self.arm.warn_code,
                "cartesian_position": self.get_position(),
                "cartesian_position_aa": self.get_position_aa(),
                "cmd_num": self.arm.cmd_num,
                "servo_codes": getattr(self.arm, "servo_codes", None),
                "temperatures": self._array7(getattr(self.arm, "temperatures", None)),
                "currents": self._array7(getattr(self.arm, "currents", None)),
                "voltages": self._array7(getattr(self.arm, "voltages", None)),
                "motor_enable_states": self._array7(getattr(self.arm, "motor_enable_states", None)),
                "motor_brake_states": self._array7(getattr(self.arm, "motor_brake_states", None)),
                "connected_flag": self.connected_flag,
                "error_state": self.error_state,
                "last_error_message": self.last_error_message,
                "last_action_code": self.last_action_code,
                "last_sdk_error_code": self.last_sdk_error_code,
                "last_joint_limit_clipped": self.last_joint_limit_clipped,
                "last_delta_limited": self.last_delta_limited,
            })
        return state

    # Action sending

    def send_action(self, action: np.ndarray) -> bool:
        """Send joint position command to the arm.

        Two modes (controlled by config.use_servo_control):

        **Servo mode (mode 1, default)**: Direct position command.
        Clips joint limits → bottleneck velocity limit → set_servo_angle_j.

        **Velocity mode (mode 4, PID inner loop)**: Store target only.
        The 250Hz inner thread reads the target, runs PID, clips velocity,
        and sends vc_set_joint_velocity. Returns True if target was stored
        (the inner thread handles error reporting).
        """
        if self.arm is None:
            self.error_state = True
            self.last_error_message = "arm not connected"
            return False

        # ── SDK-level pre-checks (ref: ufactory_teleop uf_robot.py:197-200) ──
        # send_action is called at 50 Hz — catching errors here reduces invalid
        # commands before the slower is_error() polling in the control loop.
        if not self.arm.connected:
            self.error_state = True
            self.last_error_message = "SDK reports arm not connected"
            return False
        if self.arm.error_code != 0:
            self.last_sdk_error_code = self.arm.error_code
            self.error_state = True
            self.last_error_message = f"SDK error code: {self.arm.error_code}"
            return False

        target_qpos = np.asarray(action, dtype=np.float64).reshape(7)
        target_qpos = self._limit_joint_range(target_qpos)

        if self.config.use_servo_control:
            # ── Servo mode (mode 1): direct position command ──
            qpos_cmd = self._limit_joint_step(target_qpos)

            if self.arm.mode != 1:
                self._set_mode(1)

            code = self.arm.set_servo_angle_j(
                angles=qpos_cmd.tolist(), is_radian=True
            )
            self.last_action_code = code

            if code == 0:
                self.last_qpos_cmd = qpos_cmd.copy()
                self.last_cmd_time = time.time()
                return True

            # Refresh SDK error/warn codes immediately
            try:
                _, _, sdk_err, sdk_warn = self.arm.get_err_warn_code()
            except (RuntimeError, OSError):
                sdk_err, sdk_warn = -1, -1

            self.last_sdk_error_code = int(sdk_err)
            self.error_state = True
            self.last_error_message = (
                f"set_servo_angle_j failed: code={code}, "
                f"sdk_err={sdk_err}, sdk_warn={sdk_warn}"
            )
            return False

        else:
            # ── Velocity mode (mode 4): store target for PID inner loop ──
            if self.arm.mode != 4:
                self._set_mode(4)

            with self._arm_lock:
                self._arm_pos_target = target_qpos

            self.last_qpos_cmd = target_qpos.copy()
            self.last_cmd_time = time.time()
            self.last_action_code = 0
            return True

    def reset(self, target: np.ndarray | None = None) -> bool:
        qpos = self.config.init_qpos if target is None else np.asarray(target, dtype=np.float64).reshape(7)
        self._set_mode(0)  # position control mode for blocking move
        code = self.arm.set_servo_angle(
            angle=qpos.tolist(),
            speed=self.config.reset_speed,
            mvacc=self.config.reset_acc,
            is_radian=True,
            wait=True,
        )
        # Restore control mode (1 = servo, 4 = velocity)
        mode = 4 if not self.config.use_servo_control else 1
        self._set_mode(mode)

        state = self.get_state()
        if np.all(np.isfinite(state["qpos"])):
            self.last_qpos_cmd = state["qpos"].copy()
        else:
            self.last_qpos_cmd = qpos.copy()
        self.last_cmd_time = time.time()
        self.last_action_code = code

        # Reset PID error accumulation and sync target to current position
        if not self.config.use_servo_control:
            self._arm_pid.reset()
            with self._arm_lock:
                self._arm_pos_target = self.last_qpos_cmd.copy()

        return code == 0

    # Cartesian pose queries

    def get_position(self) -> np.ndarray:
        """Get Cartesian pose as (x, y, z, roll, pitch, yaw).

        Returns:
            Array of shape (6,) in meters / radians, or NaN on failure.
        """
        if self.arm is None:
            return nan_array(6)
        code, pos = self.arm.get_position(is_radian=True)
        if code != 0:
            return nan_array(6)
        return np.asarray(pos, dtype=np.float64)

    def get_position_aa(self) -> np.ndarray:
        """Get Cartesian pose as (x, y, z, rx, ry, rz) axis-angle.

        Returns:
            Array of shape (6,) in meters / radians, or NaN on failure.
        """
        if self.arm is None:
            return nan_array(6)
        code, pos = self.arm.get_position_aa(is_radian=True)
        if code != 0:
            return nan_array(6)
        return np.asarray(pos, dtype=np.float64)

    # Soft-start

    def clear_target(self) -> None:
        """Clear the PID target (set to None) for natural deceleration.

        When the PID inner loop sees None, it sends zero velocity rather
        than holding the last position.  Used by the controller during
        PAUSED, soft-deceleration, and EMERGENCY_STOP transitions.

        Only meaningful in velocity control mode (use_servo_control=False).
        In servo mode, this is a no-op — position hold is the only option.
        """
        if not self.config.use_servo_control:
            with self._arm_lock:
                self._arm_pos_target = None

    def reset_soft_start(self) -> None:
        """Reset soft-start ramp counter. Call on TELEOP entry.

        Ensures the soft-start speed ramp always applies to the first
        teleop motion, regardless of idle duration since connect().
        Resets both servo (position) and PID (velocity) soft-start.
        """
        self._pid_converged = False  # reset convergence state for new teleop session
        if not self.config.use_servo_control:
            self._vel_ramp_start = time.perf_counter()


    def _clip_arm_velocity(self, arm_qvel: np.ndarray) -> np.ndarray:
        """Bottleneck-scale joint velocities to per-joint limits.

        Same proportional-scaling approach as _limit_joint_step, but applied
        to velocity output rather than position delta.  When any joint exceeds
        its max velocity, ALL joints are scaled by the same factor to preserve
        the joint-space trajectory shape.

        Convergence-threshold soft-start: limits velocity to 30% of pid_max_vel
        until ALL joint errors drop below pid_convergence_threshold_rad, then
        immediately releases to 100%.  The time-based ramp previously following
        convergence was redundant: once errors < 1°, PID output (~0.12 rad/s)
        is already well below the 30% floor (~0.36 rad/s), so the ramp never
        actually clipped.  This is now a pure state-driven gate — deterministic
        and reproducible across restarts.

        Activated on connect(), reset(), clear_error(), and reset_soft_start().
        When pid_convergence_threshold_rad <= 0, soft-start is disabled.

        ref: BunnyVisionPro xarm7_ability.py clip_arm_velocity()
        """
        # ── Convergence-threshold gate (ref: BunnyVisionPro init speed reduction) ──
        if self._vel_ramp_start is not None:
            if self.config.pid_convergence_threshold_rad > 0 and not self._pid_converged:
                # Not yet converged — limit to 30% to prevent initial snap
                effective_limit = self.config.pid_max_vel * 0.3
            else:
                # Converged (or threshold disabled) — release to full speed.
                # No time-based ramp: once errors < threshold, PID output is
                # naturally below the 30% floor, so the ramp was a no-op.
                self._vel_ramp_start = None  # hot path zero-overhead
                effective_limit = self.config.pid_max_vel
        else:
            effective_limit = self.config.pid_max_vel

        velocity_overshoot = np.abs(arm_qvel) / effective_limit
        max_overshoot = np.max(velocity_overshoot)
        if max_overshoot > 1.0 + 1e-4:
            safe_velocity = arm_qvel / max_overshoot
            bottleneck_joint = int(np.argmax(velocity_overshoot))
            logger.debug(
                "Velocity bottleneck: joint-%d overshoot=%.2f",
                bottleneck_joint + 1,
                max_overshoot,
            )
            return safe_velocity
        return arm_qvel

    def _internal_control_arm_qpos(self) -> None:
        """250Hz inner control loop: PID position→velocity→hardware.

        Runs on a dedicated daemon thread.  Reads the latest target from
        _arm_pos_target (set by send_action at 50Hz), computes joint-space
        position error, runs PID to produce velocity, clips to per-joint
        limits, and sends vc_set_joint_velocity.

        Errors (C31/C32, connection loss) are flagged via error_state so
        the controller's is_error() check catches them on the next cycle.
        """
        dt = float(self.config.inner_control_dt)
        rate_limiter = RateLimiter(1.0 / dt)
        logger.info("PID inner loop running at %.0f Hz", 1.0 / dt)
        self._pid_loop_impl(dt, rate_limiter)

    def _pid_loop_impl(self, dt: float, rate_limiter: RateLimiter) -> None:
        while not self._arm_should_stop.is_set():
            rate_limiter.wait()

            if self.arm is None or not self.connected_flag:
                continue

            # ── Velocity control mode gate ──
            # During mode transitions (reset, return_to_home), the arm temporarily
            # leaves velocity control mode (mode 4) for blocking position moves
            # (mode 0).  Calling vc_set_joint_velocity when mode != 4 triggers the
            # SDK's "mode may be incorrect" warning (base.py:2169-2170) and is
            # incorrect API usage — velocity commands are undefined in position mode.
            #
            # This gate is a code-level fix, NOT output suppression:
            #   - _velocity_control_active: False during intentional mode switches
            #   - self.arm.mode != 4: belt-and-suspenders, catches stale SDK cache
            #     after set_mode(4) before the next status report updates _mode.
            if not self._velocity_control_active or self.arm.mode != 4:
                continue

            # Read latest target (under lock)
            with self._arm_lock:
                target = self._arm_pos_target
            if target is None:
                # None-sentinel: send zero velocity for natural deceleration.
                # Ref: T-Rex arm_hand_control.py action_buffer None → stop command.
                # Used by controller during PAUSED / soft-deceleration / emergency
                # to let the PID inner loop decelerate smoothly rather than
                # abruptly holding position.
                try:
                    self.arm.vc_set_joint_velocity(np.zeros(7, dtype=np.float64))
                except (RuntimeError, OSError):
                    pass
                continue

            # Read current hardware position
            try:
                code, xarm_state = self.arm.get_joint_states(is_radian=True)
            except (RuntimeError, OSError) as e:
                logger.error("PID inner: get_joint_states failed: %s", e)
                self.error_state = True
                self.last_error_message = f"PID inner get_joint_states: {e}"
                continue

            if code != 0:
                logger.error(
                    "PID inner: arm error code=%d, disabling velocity control", code
                )
                self.error_state = True
                self.last_error_message = f"PID inner arm error code={code}"
                continue

            arm_current_qpos = np.asarray(xarm_state[0], dtype=np.float64)
            if arm_current_qpos.shape[0] < 7:
                continue

            # Joint-space position error → PID → velocity
            error = target[:7] - arm_current_qpos[:7]

            # Threshold-based convergence check (B3)
            if self.config.pid_convergence_threshold_rad > 0 and not self._pid_converged:
                if np.all(np.abs(error) < self.config.pid_convergence_threshold_rad):
                    self._pid_converged = True
                    logger.info(
                        "PID converged: all joint errors < %.3f rad",
                        self.config.pid_convergence_threshold_rad,
                    )

            qvel = self._arm_pid.control(error, dt)
            safe_qvel = self._clip_arm_velocity(qvel)

            # Send velocity command to hardware
            try:
                vc_code = self.arm.vc_set_joint_velocity(safe_qvel.tolist())
            except (RuntimeError, OSError) as e:
                logger.error("PID inner: vc_set_joint_velocity failed: %s", e)
                self.error_state = True
                self.last_error_message = f"PID inner vc_set_joint_velocity: {e}"
                continue

            if vc_code != 0:
                # Refresh SDK error codes for diagnosis
                try:
                    _, _, sdk_err, sdk_warn = self.arm.get_err_warn_code()
                except (RuntimeError, OSError):
                    sdk_err, sdk_warn = -1, -1
                logger.error(
                    "PID inner: vc_set_joint_velocity code=%d, "
                    "sdk_err=%s, sdk_warn=%s",
                    vc_code, sdk_err, sdk_warn,
                )
                self.error_state = True
                self.last_error_message = (
                    f"PID inner vc_set_joint_velocity: code={vc_code}"
                )

    def _stop_inner_thread(self) -> None:
        """Signal the PID inner thread to stop and wait for it to exit."""
        if self._arm_thread is None or not self._arm_thread.is_alive():
            return
        self._arm_should_stop.set()
        self._arm_thread.join(timeout=2.0)
        if self._arm_thread.is_alive():
            logger.warning("PID inner thread did not exit within timeout")
        else:
            logger.info("PID inner thread stopped")

    def _set_mode(self, mode: int):
        """Transition arm to target control mode with safety guards.

        Uses Mode 0 (idle) as an intermediate state to prevent the xArm
        controller's internal state machine from entering undefined states
        during direct mode switches.

        Sequence: idle → target → verify (ref: BunnyVisionPro xarm7_ability.py:163-167,
        ufactory_teleop uf_robot.py:124-125,210-216).

        Gate-keeps _velocity_control_active: cleared on entry (stops PID inner
        loop from calling vc_set_joint_velocity during the mode transition),
        restored on exit iff target mode is 4.
        """
        # ── Gate: suspend velocity commands during mode transition ──
        # The PID inner loop (250 Hz) must not call vc_set_joint_velocity while
        # the arm is in a non-velocity mode.  Without this gate, every call during
        # the mode-0 intermediate step triggers the SDK's base.py:2169-2170 warning.
        prev_active = self._velocity_control_active
        if not self.config.use_servo_control:
            self._velocity_control_active = False

        # Step 1: enter idle mode for safe state machine transition
        self.arm.set_mode(0)
        self.arm.set_state(0)
        time.sleep(0.05)

        # Step 2: switch to target mode with double set_state for reliability
        self.arm.set_mode(mode)
        self.arm.set_state(0)
        time.sleep(0.05)
        self.arm.set_state(0)

        # Step 3: verify no latent errors after mode switch
        # xArm controller can silently fail — mode switch reports success but
        # internal error code is non-zero (ref: ufactory_teleop uf_robot.py:147-149).
        _, err_warn = self.arm.get_err_warn_code()
        if err_warn[0] != 0:
            self.error_state = True
            self.last_error_message = (
                f"_set_mode({mode}) post-check failed: err_warn={err_warn}"
            )

        # ── Gate: resume velocity commands only when back in velocity mode ──
        if not self.config.use_servo_control:
            if mode == 4:
                self._velocity_control_active = True
                self._vel_ramp_start = time.perf_counter()  # reset velocity soft-start

    def robot_init(self) -> None:
        """Full initialization sequence for the xArm7 controller.

        Encapsulates the complete init ritual required after connect() and
        after C31/C32 error recovery.  Safe to call at any time while connected.

        Sequence (ref: ufactory_teleop uf_robot.py:139-187):
          1. clean_error + clean_warn
          2. motion_enable(True)
          3. _set_mode() with Mode-0 transition + error verification (A2/A3)
          4. set_collision_sensitivity(0)
          5. set_tcp_load(...)
          6. get_err_warn_code() final verification
          7. reset soft-start
        """
        if self.arm is None:
            return

        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(True)

        mode = 4 if not self.config.use_servo_control else 1
        self._set_mode(mode)

        self._configure_collision_params()

        # Final error/warning verification (ref: ufactory_teleop uf_robot.py:147-149)
        _, err_warn = self.arm.get_err_warn_code()
        if err_warn[0] != 0:
            self.error_state = True
            self.last_error_message = (
                f"robot_init post-check failed: err_warn={err_warn}"
            )

        # Reset soft-start so the next teleop motion gets the full ramp
        self.reset_soft_start()

    def _configure_collision_params(self) -> None:
        """Set TCP load and collision sensitivity to prevent false C31 triggering.

        C31 (Collision Caused Abnormal Current) detection mechanism:
          - xArm controller estimates theoretical joint torques via dynamics model
          - Compares actual torque (motor current) against theoretical torque
          - Deviation exceeds threshold → C31 emergency stop

        Without load configured, dynamics model assumes 0kg → theoretical torque is
        severely underestimated → the torque needed to drive XHand (~1.2kg) is
        misclassified as a collision.
        """
        if self.arm is None:
            return
        cfg = self.config

        try:
            code = self.arm.set_tcp_load(
                cfg.tcp_load_kg,
                list(cfg.tcp_load_cog_mm),
            )
            if code != 0:
                self.last_error_message = f"set_tcp_load failed: code={code}"
        except RuntimeError as e:
            self.last_error_message = f"set_tcp_load exception: {e}"

        try:
            code = self.arm.set_collision_sensitivity(cfg.collision_sensitivity)
            if code != 0:
                self.last_error_message = f"set_collision_sensitivity failed: code={code}"
        except RuntimeError as e:
            self.last_error_message = f"set_collision_sensitivity exception: {e}"

    def _read_qpos(self) -> np.ndarray:
        code, qpos = self.arm.get_servo_angle(is_radian=True)
        if code != 0:
            return nan_array(7)
        return self._array7(qpos)

    def _limit_joint_range(self, qpos: np.ndarray) -> np.ndarray:
        # XArm7 variant: same np.clip logic as XHand._limit_joint_range but with
        # different clipping targets (arm joint ranges vs hand finger ranges).
        if not self.config.clip_joint_limit:
            self.last_joint_limit_clipped = False
            return qpos
        clipped = np.clip(qpos, self.config.qpos_min, self.config.qpos_max)
        self.last_joint_limit_clipped = not np.allclose(qpos, clipped)
        return clipped

    def _limit_joint_step(self, target_qpos: np.ndarray) -> np.ndarray:
        """Limit per-step joint motion using proportional (bottleneck) scaling.

        Uses a scalar scaling approach: when any joint exceeds its individual
        speed limit, ALL joints are scaled by the same factor. This preserves
        the relative joint-space trajectory (and approximately the Cartesian
        trajectory), unlike per-joint independent clipping which distorts the
        motion path.

        Reference is the hardware position (not previous command), so that
        tracking lag does not cause command compounding.

        Soft-start (ref: ufactory_teleop uf_robot.py L206): first N frames
        use a linear ramp from soft_start_speed_rad_s to per-joint max_qvel,
        eliminating the speed jump that a hard switch would cause. After the
        ramp period, per-joint max_qvel limits apply at full speed.

        ref: BunnyVisionPro xarm7_ability.py clip_arm_next_qpos() — scalar
        scaling with hardware position as reference.

        See also: XHand._limit_joint_step (xhand.py) — per-joint independent
        clipping (different strategy for hand finger joints).
        """
        if not self.config.use_delta_limit:
            self.last_delta_limited = False
            return target_qpos

        now = time.time()

        # Read current hardware position — the ground-truth reference for
        # how far the robot will actually move.
        hw_qpos = self._read_qpos()

        if self.last_qpos_cmd is None:
            if np.all(np.isfinite(hw_qpos)):
                self.last_qpos_cmd = hw_qpos.copy()
            else:
                self.last_qpos_cmd = self.config.init_qpos.copy()
        if self.last_cmd_time is None:
            self.last_cmd_time = now

        dt_raw = now - self.last_cmd_time
        dt = min(max(dt_raw, self.config.dt), self.config.dt * 10)
        # dt floor: prevents divide-by-zero / infinite speed when commands
        #   arrive faster than expected.
        # dt ceiling (10× dt): if the control loop stalls (GC, system pause),
        #   caps the allowed step to 10 frames of motion (~0.2 s @ 50 Hz).
        #   Without this, a 500 ms gap would allow a 45-75° jump.

        max_step = self.config.max_qvel * dt

        # Use hardware position as the delta reference.  When the hardware has
        # not yet reached the previous command (tracking lag), the delta from
        # hardware to target is larger than from last_cmd to target — measuring
        # from hardware catches this and clips the actual robot motion.
        if np.all(np.isfinite(hw_qpos)):
            ref = hw_qpos
        else:
            ref = self.last_qpos_cmd

        delta = target_qpos - ref

        # Proportional (bottleneck) scaling — same factor applied to all joints.
        # Normalize each joint's delta by its individual speed limit, then find
        # the bottleneck joint. If any joint exceeds its limit, scale ALL joints
        # proportionally to preserve the trajectory shape.
        normalized = np.abs(delta) / max_step
        max_ratio = np.max(normalized)
        if max_ratio > 1.0:
            delta = delta / max_ratio
        self.last_delta_limited = max_ratio > 1.0
        return ref + delta

    @staticmethod
    def _array7(value) -> np.ndarray:
        return safe_resize(value, 7)


