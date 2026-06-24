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

    Operates on 7-DOF joint position error using **derivative on measurement**
    (not derivative on error)::

        vel = Kp * err - Kd * (current - prev_current) / dt + Ki * cum_err

    **Derivative on measurement (F1)**: the D term reacts only to changes in
    the process variable (joint position), NOT changes in the setpoint (target).
    This eliminates the velocity spike that derivative-on-error produces when
    the target steps at 50 Hz — the dominant source of audible/visible jerk in
    the original implementation.  The P term alone handles setpoint tracking;
    the D term provides identical oscillation damping without reacting to steps.

    Integral term (Ki) is disabled by default to prevent windup in
    teleoperation where the target is continuously changing.

    **Anti-windup**: when ``max_vel`` is provided to ``control()``, the
    integral accumulator ``_cum_err`` is clamped per-joint so that the
    total I contribution never exceeds the velocity limit.  Without this
    a sustained small error with non-zero Ki would cause unbounded
    integral growth, leading to severe overshoot when the error changes
    sign.  The clamp is conservative: ``|Ki * cum_err| <= max_vel``.

    ref: BunnyVisionPro xarm7_ability.py:11-36 (original D-on-error)
         Franklin Feedback Control ch.6 — derivative-on-measurement
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
        self._prev_current: np.ndarray | None = None  # F1: D-on-measurement
        self._cum_err: np.ndarray = np.zeros_like(self.kp)
        self._ki_nonzero = np.any(self.ki != 0.0)  # fast-path gate

    def reset(self) -> None:
        """Clear error history (call on mode switch or arm reset)."""
        self._prev_err = None
        self._prev_current = None  # F1
        self._cum_err = np.zeros_like(self.kp)

    def control(
        self,
        err: np.ndarray,
        dt: float,
        max_vel: np.ndarray | None = None,
        *,
        current: np.ndarray | None = None,  # F1: for D-on-measurement
    ) -> np.ndarray:
        """Compute velocity command from position error.

        Args:
            err: Joint position error (target - current), shape (7,).
            dt: Time step in seconds (inner loop period).
            max_vel: Per-joint velocity limit for anti-windup clamping.
            current: Current joint position (process variable) for
                     derivative-on-measurement.  Required for the D term
                     to avoid reacting to setpoint steps.

        Returns:
            Joint velocity command, shape (7,).
        """
        err = np.asarray(err, dtype=np.float64)
        if self._prev_err is None:
            self._prev_err = err.copy()

        # Integral accumulation with anti-windup clamping when Ki is active.
        if self._ki_nonzero:
            self._cum_err += dt * err
            if max_vel is not None:
                i_contrib = self.ki * self._cum_err
                i_max = np.abs(max_vel)
                i_clipped = np.clip(i_contrib, -i_max, i_max)
                self._cum_err = np.where(self.ki != 0, i_clipped / self.ki, self._cum_err)

        # F1: Derivative on measurement — D term uses -d(current)/dt
        # instead of d(error)/dt.  This gives identical oscillation damping
        # without reacting to setpoint steps (the 50Hz target update).
        if current is not None and self._prev_current is not None:
            d_term = -self.kd * (current - self._prev_current) / dt
        else:
            # Fallback: derivative on error (backward compatible)
            d_term = self.kd * (err - self._prev_err) / dt

        # P + D + I
        value = (
            self.kp * err
            + d_term
            + self.ki * self._cum_err
        )

        self._prev_err = err.copy()
        if current is not None:
            self._prev_current = np.asarray(current, dtype=np.float64).copy()

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
    # Default 0 (disabled): with conservative pid_max_vel (~50% max_qvel),
    # the two-phase soft-start is unnecessary — the bottleneck scaling alone
    # produces smooth velocity profiles without stair-step gating.
    pid_convergence_threshold_rad: float = 0.0
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
        default_factory=lambda: np.array([0.8, 0.8, 0.8, 0.8, 1.0, 1.0, 1.5])
    )  # rad/s, ~50% of max_qvel — conservative cap for smooth teleop motion.
    # Aligned with BunnyVisionPro's default.  Lower max_vel removes the need for
    # complex soft-start logic: natural PID damping + bottleneck scaling produce
    # smooth velocity profiles without multi-phase ramp gating.

    # ── Integral term (C1 — replay precision) ──
    # Per-joint integral gain for steady-state error elimination.
    # Default 0.0 for teleop (prevents integral windup with continuously
    # moving targets).  Set to ~0.1 × Kp during replay for accurate
    # trajectory tracking.  Anti-windup clamping is built into PIDController
    # (clamps |Ki * cum_err| <= max_vel per joint).
    pid_ki: np.ndarray = field(
        default_factory=lambda: np.zeros(7)
    )

    # ── Inner-loop target interpolation (A1 — smoothness) ──
    # When True, the PID inner loop linearly interpolates between consecutive
    # 50 Hz targets at 250 Hz, eliminating the stair-step pattern in the
    # position error signal.  Adds ~4ms effective latency (half the 8ms
    # inter-target gap) for significantly smoother velocity profiles.
    inner_target_interpolation: bool = True

    # ── Soft-start linear ramp (A2 — smooth engagement) ──
    # Duration (seconds) of a linear velocity ramp from 0% → 100% when
    # pid_convergence_threshold_rad > 0 and convergence has been reached,
    # or when pid_convergence_threshold_rad <= 0 (simplified single-phase).
    # 0 = no ramp, full speed immediately (backward compatible).
    # 0.3s default (reduced from 0.5): with conservative max_vel (~50% of
    # max_qvel), the arm engages gently without needing a long ramp.
    soft_start_ramp_duration: float = 0.3

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
        # A1: Target interpolation — stores the previous target + timestamps
        # so the inner loop can linearly interpolate between 50 Hz updates
        # at 250 Hz, eliminating the stair-step error pattern.
        self._arm_pos_target_prev: np.ndarray | None = None
        self._arm_pos_target_ts: float = 0.0
        self._arm_pos_target_prev_ts: float = 0.0
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
                ki=self.config.pid_ki,  # C1: integral term for replay precision
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

        # H1: Preserve error_state from robot_init() — don't unconditionally
        # overwrite.  If the firmware has residual error codes (e.g. C31
        # collision not fully recovered), the controller must not enter
        # normal operation.
        if not self.error_state:
            self.connected_flag = True
        else:
            logger.error("connect(): robot_init detected hardware error, aborting")
            return False

        # Start PID inner thread for velocity control mode
        if not self.config.use_servo_control:
            self._arm_should_stop.clear()
            self._arm_pos_target = self.last_qpos_cmd
            # A1: reset interpolation state on fresh connection
            self._arm_pos_target_prev = None
            self._arm_pos_target_ts = time.perf_counter()
            self._arm_pos_target_prev_ts = 0.0
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
        else:
            # L3: servo mode (mode 1) is superseded by velocity mode (mode 4)
            # with the PID inner loop.  Velocity mode provides smoother motion,
            # soft-start ramp, and None-sentinel deceleration.  Servo mode is
            # retained for backward compatibility but may be removed in a
            # future release.
            logger.warning(
                "Servo control mode (mode 1) is deprecated — "
                "set use_servo_control=False for velocity PID mode (mode 4)"
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
                qpos = self._read_qpos()
                # H2 (layer 1): guard against NaN from failed get_servo_angle.
                # _read_qpos() returns nan_array(7) on error - feeding NaN into
                # the PID target produces NaN velocity that passes through
                # _clip_arm_velocity (NaN > 1.0 is False per IEEE 754).
                if np.all(np.isfinite(qpos)):
                    self._arm_pos_target = qpos
                else:
                    fallback = (
                        self.last_qpos_cmd.copy()
                        if self.last_qpos_cmd is not None
                        else self.config.init_qpos.copy()
                    )
                    logger.warning(
                        "clear_error(): qpos read returned NaN - "
                        "using fallback target (last_cmd=%s)",
                        self.last_qpos_cmd is not None,
                    )
                    self._arm_pos_target = fallback
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

            now = time.perf_counter()
            with self._arm_lock:
                # A1: save previous target for linear interpolation in inner loop
                if self.config.inner_target_interpolation and self._arm_pos_target is not None:
                    self._arm_pos_target_prev = self._arm_pos_target.copy()
                    self._arm_pos_target_prev_ts = self._arm_pos_target_ts
                else:
                    self._arm_pos_target_prev = None
                self._arm_pos_target = target_qpos
                self._arm_pos_target_ts = now

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
                # A1: clear interpolation state so the inner loop doesn't
                # try to interpolate from a stale previous target after resume.
                self._arm_pos_target_prev = None

    def reset_soft_start(self) -> None:
        """Reset soft-start ramp counter. Call on TELEOP entry.

        Ensures the soft-start speed ramp always applies to the first
        teleop motion, regardless of idle duration since connect().
        Resets both servo (position) and PID (velocity) soft-start.
        """
        self._pid_converged = False  # reset convergence state for new teleop session
        if not self.config.use_servo_control:
            self._vel_ramp_start = time.perf_counter()

    def set_pid_ki(self, ki: np.ndarray | float) -> None:
        """Set integral gain at runtime (C1 — replay precision).

        During teleop, Ki=0 prevents integral windup with continuously
        moving targets.  During replay, a small Ki (~0.1 × Kp ≈ 0.5-0.7)
        eliminates steady-state tracking error for faithful trajectory
        reproduction.  The built-in anti-windup clamps |Ki*cum_err| ≤
        max_vel per joint.

        Args:
            ki: Per-joint integral gains (7,) or scalar broadcast to all joints.
        """
        if not self.config.use_servo_control:
            ki_arr = np.broadcast_to(np.asarray(ki, dtype=np.float64), 7).copy()
            self._arm_pid.ki = ki_arr
            self._arm_pid._ki_nonzero = np.any(ki_arr != 0.0)
            self._arm_pid.reset()  # clear accumulated integral
            logger.info("PID Ki set to %s", ki_arr)

    def _clip_arm_velocity(self, arm_qvel: np.ndarray) -> np.ndarray:
        """Bottleneck-scale joint velocities to per-joint limits.

        Proportional-scaling: when any joint exceeds its max velocity, ALL
        joints are scaled by the same factor to preserve trajectory shape.

        Soft-start (activated on connect/reset/clear_error/reset_soft_start):
          - When pid_convergence_threshold_rad > 0 (two-phase):
              Phase 1: 30% hard limit until all errors < threshold
              Phase 2: linear ramp 0%→100% over soft_start_ramp_duration
          - When pid_convergence_threshold_rad <= 0 (simplified):
              Single linear ramp 0%→100% over soft_start_ramp_duration.
              If soft_start_ramp_duration <= 0, skip straight to full speed.

        ref: BunnyVisionPro xarm7_ability.py clip_arm_velocity()
        """
        # H2 (layer 3): NaN entry guard — NaN velocity passes through
        # unclipped because all comparisons with NaN return False per
        # IEEE 754 (NaN > 1.0 → False, bypasses bottleneck scaling).
        # Return zero velocity instead of forwarding NaN to hardware.
        if not np.all(np.isfinite(arm_qvel)):
            return np.zeros(7, dtype=np.float64)

        if self._vel_ramp_start is not None:
            thr = self.config.pid_convergence_threshold_rad
            ramp_dur = self.config.soft_start_ramp_duration

            if thr > 0 and not self._pid_converged:
                # Phase 1 (two-phase mode): 30% hard limit before convergence
                effective_limit = self.config.pid_max_vel * 0.3
            elif ramp_dur <= 0:
                # No ramp — release to full speed immediately
                self._vel_ramp_start = None
                effective_limit = self.config.pid_max_vel
            else:
                # Simplified ramp: 0% → 100% over soft_start_ramp_duration
                elapsed = time.perf_counter() - self._vel_ramp_start
                ramp_progress = min(elapsed / ramp_dur, 1.0)
                effective_limit = self.config.pid_max_vel * ramp_progress
                if ramp_progress >= 1.0:
                    self._vel_ramp_start = None  # ramp complete
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
        # H3 (fix A): wrap the entire PID inner loop with try/except so that
        # an unhandled exception doesn't silently kill the daemon thread.
        # The outer try catches fatal init errors; the inner try catches
        # per-iteration errors (IndexError, TypeError, ValueError from
        # unexpected xarm_state format, target type mismatch, PID math).
        # The controller's thread-alive monitor (H3 fix B) detects a dead
        # thread and escalates to E-Stop within ~1 second.
        try:
            while not self._arm_should_stop.is_set():
                try:
                    rate_limiter.wait()

                    if self.arm is None or not self.connected_flag:
                        continue

                    # --- Velocity control mode gate ---
                    # During mode transitions (reset, return_to_home), the arm
                    # temporarily leaves velocity control mode (mode 4) for
                    # blocking position moves (mode 0).  Calling
                    # vc_set_joint_velocity when mode != 4 triggers the SDK's
                    # "mode may be incorrect" warning and is incorrect API usage.
                    if not self._velocity_control_active or self.arm.mode != 4:
                        continue

                    # Read latest target + interpolation state (under lock)
                    with self._arm_lock:
                        target = self._arm_pos_target
                        if self.config.inner_target_interpolation:
                            target_prev = self._arm_pos_target_prev
                            target_ts = self._arm_pos_target_ts
                            target_prev_ts = self._arm_pos_target_prev_ts
                        else:
                            target_prev = None
                    if target is None:
                        # None-sentinel: send zero velocity for natural
                        # deceleration.  Used by controller during PAUSED /
                        # soft-deceleration / emergency to let the PID inner
                        # loop decelerate smoothly.
                        try:
                            self.arm.vc_set_joint_velocity(
                                np.zeros(7, dtype=np.float64)
                            )
                        except (RuntimeError, OSError):
                            pass
                        continue

                    # H2 (layer 2): NaN target guard - NaN in PID error
                    # produces NaN velocity that bypasses _clip_arm_velocity
                    # (all NaN comparisons return False per IEEE 754).
                    if not np.all(np.isfinite(target)):
                        logger.error(
                            "PID inner: target is NaN, sending zero velocity"
                        )
                        try:
                            self.arm.vc_set_joint_velocity(
                                np.zeros(7, dtype=np.float64)
                            )
                        except (RuntimeError, OSError):
                            pass
                        continue

                    # A1: Target interpolation — linearly interpolate between
                    # consecutive 50 Hz targets at 250 Hz to eliminate the
                    # stair-step pattern in the position error signal.
                    # Without this, the PID sees step changes every 5th tick
                    # (~4 ms gap), producing higher-frequency content in the
                    # velocity command.  The interpolation adds at most ~4 ms
                    # effective latency (half the 8 ms inter-target gap).
                    if (
                        self.config.inner_target_interpolation
                        and target_prev is not None
                        and np.all(np.isfinite(target_prev))
                    ):
                        now = time.perf_counter()
                        gap = target_ts - target_prev_ts
                        if gap > 0 and gap < 0.1:  # sanity: gap must be < 100ms
                            t = (now - target_prev_ts) / gap
                            if t < 1.0:
                                # Linear interpolation between prev and current
                                target = target_prev + (
                                    target - target_prev
                                ) * min(max(t, 0.0), 1.0)
                            # else t >= 1.0: use current target directly

                    # Read current hardware position
                    try:
                        code, xarm_state = self.arm.get_joint_states(
                            is_radian=True
                        )
                    except (RuntimeError, OSError) as e:
                        logger.error(
                            "PID inner: get_joint_states failed: %s", e
                        )
                        self.error_state = True
                        self.last_error_message = (
                            f"PID inner get_joint_states: {e}"
                        )
                        continue

                    if code != 0:
                        logger.error(
                            "PID inner: arm error code=%d, "
                            "disabling velocity control",
                            code,
                        )
                        self.error_state = True
                        self.last_error_message = (
                            f"PID inner arm error code={code}"
                        )
                        continue

                    arm_current_qpos = np.asarray(
                        xarm_state[0], dtype=np.float64
                    )
                    if arm_current_qpos.shape[0] < 7:
                        continue

                    # Joint-space position error -> PID -> velocity
                    error = target[:7] - arm_current_qpos[:7]

                    # Threshold-based convergence check (B3)
                    thr = self.config.pid_convergence_threshold_rad
                    if thr > 0 and not self._pid_converged:
                        if np.all(np.abs(error) < thr):
                            self._pid_converged = True
                            logger.info(
                                "PID converged: all joint errors < %.3f rad",
                                thr,
                            )

                    qvel = self._arm_pid.control(
                        error, dt,
                        max_vel=self.config.pid_max_vel,
                        current=arm_current_qpos,  # F1: D-on-measurement
                    )
                    safe_qvel = self._clip_arm_velocity(qvel)

                    # Send velocity command to hardware
                    try:
                        vc_code = self.arm.vc_set_joint_velocity(
                            safe_qvel.tolist()
                        )
                    except (RuntimeError, OSError) as e:
                        logger.error(
                            "PID inner: vc_set_joint_velocity failed: %s", e
                        )
                        self.error_state = True
                        self.last_error_message = (
                            f"PID inner vc_set_joint_velocity: {e}"
                        )
                        continue

                    if vc_code != 0:
                        # Refresh SDK error codes for diagnosis
                        try:
                            _, _, sdk_err, sdk_warn = (
                                self.arm.get_err_warn_code()
                            )
                        except (RuntimeError, OSError):
                            sdk_err, sdk_warn = -1, -1
                        logger.error(
                            "PID inner: vc_set_joint_velocity code=%d, "
                            "sdk_err=%s, sdk_warn=%s",
                            vc_code,
                            sdk_err,
                            sdk_warn,
                        )
                        self.error_state = True
                        self.last_error_message = (
                            f"PID inner vc_set_joint_velocity: code={vc_code}"
                        )
                except Exception:
                    logger.exception(
                        "PID inner loop iteration failed"
                    )
                    self.error_state = True
                    self.last_error_message = (
                        "PID inner loop unhandled exception"
                    )
                    # Attempt to send zero velocity before exiting the loop
                    try:
                        self.arm.vc_set_joint_velocity(
                            np.zeros(7, dtype=np.float64)
                        )
                    except Exception:
                        pass
                    break
        except Exception:
            logger.exception("PID inner loop fatal error")
            self.error_state = True
            self.last_error_message = "PID inner loop fatal exception"
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

        Thread safety (L2): this method is intentionally single-threaded —
        called only from the main thread (connect, robot_init, clear_error).
        The _velocity_control_active gate + time.sleep barriers are the
        coordination mechanism with the PID daemon thread; no additional
        lock is needed.
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
        else:
            # AF1: only reset soft-start on clean init.  When the post-check
            # detects errors, skip so that connect() (H1) can correctly abort.
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


