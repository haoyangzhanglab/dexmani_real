"""Arm servo loop — Mode 6 joint online trajectory planning for xArm7.

Primary entry point: ``arm_loop(shared)`` — mp.Process target, reads
SharedStorage.arm_action_q, writes arm_state_ring. Communicates exclusively
through SharedStorage (no direct SDK access from other processes).

Mode 6: firmware performs online trajectory replanning with configurable
speed/acceleration limits. No inner-loop interpolation — commands forwarded
directly and firmware handles all trajectory smoothing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dexmani_real.config.defaults import arm
from dexmani_real.planning.path_utils import wrap_nearest_equivalent
from dexmani_real.utils.log import ThrottledWarner, get_logger
from dexmani_real.utils.rate_manager import RateManager

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════


@dataclass
class ArmLoopConfig:
    """Configuration for arm_loop (Mode 6: joint online trajectory planning).

    Defaults sourced from :data:`~dexmani_real.config.defaults.arm` singleton.
    """

    joint_max_speed_rad_per_s: float = field(default_factory=lambda: arm.max_joint_velocity_rad_per_s)
    joint_max_acc_rad_per_s2: float = field(default_factory=lambda: arm.max_joint_acceleration_rad_per_s2)
    arm_loop_hz: float = field(default_factory=lambda: arm.loop_hz)

    # Joint limits — sourced from arm singleton via shared_storage re-exports.
    joint_limit_lower: tuple[float, ...] = field(default_factory=lambda: arm.joint_limit_lower)
    joint_limit_upper: tuple[float, ...] = field(default_factory=lambda: arm.joint_limit_upper)

    # Tracking error warning threshold (rad). Diagnostic only.
    tracking_error_warn_rad: float = field(default_factory=lambda: arm.tracking_error_warn_rad)

    # Arm connection (single source of truth for IP).
    arm_ip: str = field(default_factory=lambda: arm.ip)

    # Home position — sourced from arm singleton via shared_storage re-exports.
    home_qpos: tuple[float, ...] = field(default_factory=lambda: arm.home_qpos)

    # Collision sensitivity level (0-5, 1 = most sensitive).
    collision_sensitivity: int = field(default_factory=lambda: arm.collision_sensitivity)

    # Homing parameters for _planned_homing.
    homing_convergence_rad: float = field(default_factory=lambda: arm.homing.convergence_rad)
    homing_step_interval_s: float = field(default_factory=lambda: arm.homing.step_interval_s)
    homing_max_speed_rad_per_s: float = field(default_factory=lambda: np.deg2rad(arm.homing.max_speed_deg_s))
    homing_target_timeout_s: float = field(default_factory=lambda: arm.homing.target_timeout_s)

    # ── Velocity feedforward (application-level compensation for Mode 6 servo lag) ──
    # Per-joint lead gain (seconds).  Adds cmd_vel * lead_gain to the position
    # command — the anticipatory target gives Mode 6's trajectory planner a
    # "head start" that compensates for its lack of velocity feedforward.
    #   J5 (wrist roll, highest inertia):        0.06 s  (~2 servo ticks @30Hz)
    #   J2 (shoulder lift), J6 (wrist pitch):    0.04 s
    #   J1/J3/J4/J7 (lighter joints):            0.03 s  (~1 servo tick @30Hz)
    # Set to None to disable feedforward entirely.
    feedforward_lead_gain: tuple[float, ...] | None = field(default_factory=lambda: (
        0.03,  # J1
        0.04,  # J2
        0.03,  # J3
        0.03,  # J4
        0.06,  # J5
        0.04,  # J6
        0.03,  # J7
    ))

    # Max absolute correction per joint (rad).  Clamps the feedforward term to
    # prevent overshoot on noisy velocity estimates.  0.05 rad ≈ 2.9°.
    feedforward_max_correction_rad: float = 0.05


# Controller errors that indicate a problematic target rather than a hardware fault.
_RECOVERABLE_ERRORS: frozenset[int] = arm.recoverable_errors
_RECOVERY_MAX: int = 30  # consecutive recoveries before FAULT escalation (1s @ 30Hz)

# Velocity feedforward: number of ticks to skip after startup or HOME before
# enabling compensation.  3 ticks @ 30 Hz ≈ 100 ms — enough for the arm state
# ring to be populated and any homing transient to settle.
_FF_SKIP_TICKS: int = 3


# ═══════════════════════════════════════════════════════════════════
# arm_loop (mp.Process target)
# ═══════════════════════════════════════════════════════════════════


def arm_loop(shared, config: ArmLoopConfig | None = None) -> None:
    """Arm process entry point — reads SharedStorage.arm_action_q, servos arm.

    Designed as an mp.Process target. Communicates exclusively through
    SharedStorage (no RPC, no side channels).

    Features velocity feedforward compensation (configurable via
    ``ArmLoopConfig.feedforward_lead_gain``): adds ``cmd_vel * lead_gain``
    to the position command to compensate for Mode 6 servo lag.  Per-joint
    direction guard prevents overshoot; joint-limit clip after feedforward
    is defense-in-depth against C31 triggers.
    """
    from queue import Empty

    from dexmani_real.shm.shared_storage import HOME_SENTINEL, ARM_STATE_DTYPE, new_frame
    from dexmani_real.robot.safety import SafetyState, transition
    from dexmani_real.planning.kinematics import ArmFK
    from dexmani_real import ASSET_DIR

    _tracking_warn = ThrottledWarner(interval_s=5.0)
    _fk_warn = ThrottledWarner(interval_s=5.0)
    _consecutive_recoveries = 0
    _consecutive_state_errors = 0
    cfg = config or ArmLoopConfig()

    HOME_QPOS = np.array(cfg.home_qpos, dtype=np.float64)

    # ── Velocity feedforward state ──
    _ff_enabled = (
        cfg.feedforward_lead_gain is not None
        and any(g != 0.0 for g in cfg.feedforward_lead_gain)
    )
    _ff_lead_gain = (
        np.array(cfg.feedforward_lead_gain, dtype=np.float64)
        if cfg.feedforward_lead_gain is not None
        else np.zeros(7, dtype=np.float64)
    )
    _current_raw_target: np.ndarray | None = None  # uncompensated target, updated on new-action ticks only
    _prev_raw_target: np.ndarray | None = None  # previous new-action target, for velocity estimation
    _prev_target_time: float | None = None
    _last_cmd_vel: np.ndarray | None = None  # 7-DOF (rad/s), reused on hold ticks
    _ff_skip_count: int = _FF_SKIP_TICKS  # skip first N frames after startup/HOME

    # Pre-convert joint limits for fast clamping (feedforward safety net).
    _joint_lo = np.array(cfg.joint_limit_lower, dtype=np.float64)
    _joint_hi = np.array(cfg.joint_limit_upper, dtype=np.float64)

    # ── URDF-consistent FK (replaces arm.get_position_aa) ──
    # xArm firmware uses a different EEF coordinate definition than our URDF.
    # Using Pinocchio FK ensures all downstream consumers (IK, recording,
    # display) share a single coordinate system.
    _urdf_path = str(ASSET_DIR / "robots" / "xhand" / "xarm7_xhand_collision.urdf")
    _arm_fk = ArmFK(_urdf_path)

    try:
        from xarm.wrapper import XArmAPI
        arm = XArmAPI(cfg.arm_ip, is_radian=True)
    except Exception as e:
        logger.error("arm_loop: connect failed: %s", e)
        shared.error_state.value = True
        return

    try:
        arm.clean_error()
        arm.clean_warn()
        arm.motion_enable(True)
        arm.set_mode(6)
        arm.set_state(0)
        arm.set_collision_sensitivity(cfg.collision_sensitivity)
        arm.set_joint_maxacc(cfg.joint_max_acc_rad_per_s2, is_radian=True)
        if getattr(arm, "mode", -1) != 6:
            logger.error("arm_loop: failed to set mode 6")
            shared.error_state.value = True
            _disconnect_arm(arm)
            return
    except Exception as e:
        logger.error("arm_loop: init failed: %s", e)
        shared.error_state.value = True
        _disconnect_arm(arm)
        return

    # Seed last_qpos — FAIL if initial state unreadable (safety: never cmd HOME_QPOS blind).
    try:
        code, states = arm.get_joint_states(is_radian=True, num=1)
        if code == 0 and len(states) > 0:
            last_qpos = np.asarray(states[0], dtype=np.float64)[:7].copy()
        else:
            logger.error("arm_loop: cannot read initial joint states (code=%d)", code)
            shared.error_state.value = True
            _disconnect_arm(arm)
            return
    except Exception as e:
        logger.error("arm_loop: joint states read failed: %s", e)
        shared.error_state.value = True
        _disconnect_arm(arm)
        return
    last_target = last_qpos.copy()

    # Publish initial state BEFORE arm_ready — consumers wait on arm_ready and
    # expect the ring to already contain a valid frame.  Without this, there is
    # a one-tick window where arm_ready is set but arm_state_ring is empty.
    try:
        eef_pos_init, eef_rot6d_init = _arm_fk.compute(last_qpos)
    except Exception:
        eef_pos_init = np.zeros(3, dtype=np.float64)
        eef_rot6d_init = np.zeros(6, dtype=np.float64)
    _frame0 = new_frame(ARM_STATE_DTYPE)
    _frame0["qpos"][0] = last_qpos
    _frame0["qvel"][0] = np.zeros(7, dtype=np.float64)
    _frame0["tau"][0] = np.zeros(7, dtype=np.float64)
    _frame0["eef_pos"][0] = eef_pos_init
    _frame0["eef_rot6d"][0] = eef_rot6d_init
    _frame0["error_code"][0] = 0
    _frame0["connected"][0] = 1
    _frame0["mode"][0] = 6
    _frame0["tracking_err"][0] = 0.0
    _frame0["timestamp"][0] = time.monotonic()
    shared.arm_state_ring.write(_frame0)

    # Write heartbeat BEFORE ready signal — prevents false FAULT on startup
    # (same pattern as vr_loop).  Main's supervisor checks heartbeats immediately
    # after all ready events.
    shared.arm_heartbeat_s.value = time.monotonic()
    shared.arm_ready.set()
    logger.info("arm_loop: ready (Mode 6, ip=%s, hz=%.0f)", cfg.arm_ip, cfg.arm_loop_hz)

    limiter = RateManager(cfg.arm_loop_hz)
    while shared.is_running.value:
        # Heartbeat — written even when holding position (proves we're alive)
        shared.arm_heartbeat_s.value = time.monotonic()

        if shared.estop_request.value:
            try:
                arm.set_state(4)
            except Exception:
                logger.warning("arm_loop: estop set_state(4) failed", exc_info=True)
            break

        # Safety state gate — only process commands in ARMED or RUNNING.
        # When gated (DISARMED or FAULT), skip action read + servo but continue
        # to publish state (for monitoring) and rate-limit normally.
        _safety = shared.safety_state.value
        if _safety in (SafetyState.ARMED, SafetyState.RUNNING):

            # Read action from queue (non-blocking — rate limiter controls cadence)
            try:
                action = shared.arm_action_q.get(timeout=0.0)
            except Empty:
                action = None

            # HOME sentinel (tuple: (HOME_SENTINEL, waypoints_or_None))
            if isinstance(action, tuple) and len(action) == 2 and action[0] == HOME_SENTINEL:
                _waypoints = action[1]
                logger.info("arm_loop: HOME sentinel — planned homing (%d waypoints)",
                            len(_waypoints) if _waypoints is not None else 0)
                _planned_homing(arm, _waypoints, HOME_QPOS, cfg, shared=shared)
                last_qpos = HOME_QPOS.copy()
                last_target = HOME_QPOS.copy()
                # Reset feedforward state — skip first N frames after home
                # to avoid transient compensation from the jump to home_qpos.
                _current_raw_target = None
                _prev_raw_target = None
                _prev_target_time = None
                _last_cmd_vel = None
                _ff_skip_count = _FF_SKIP_TICKS
                continue

            # Servo
            _new_action = action is not None and isinstance(action, dict)
            if _new_action:
                target = np.asarray(action.get("qpos", last_target), dtype=np.float64).ravel()[:7]
                if np.all(np.isfinite(target)):
                    # Wrap equivalent joints (J1/J3/J5/J7) to the same 2π band
                    # as the physical arm position, so Mode 6 always takes the
                    # shortest angular path.  Without this, a target on the
                    # opposite band (e.g. +3.1 rad when the arm is at -3.1 rad)
                    # can cause the firmware to rotate the joint through ~2π
                    # → "关节转大圈" during teleop.
                    # Ref: canonicalize_qpos in ik_candidates.py does the same
                    # wrapping during IK, but this is defense-in-depth for any
                    # edge case where the IK result slips onto the wrong band.
                    target = wrap_nearest_equivalent(
                        target, last_qpos,
                        cfg.joint_limit_lower, cfg.joint_limit_upper,
                    )
                    last_target = target.copy()

            # ── Velocity feedforward compensation ──
            # Compute command velocity from successive targets and add a lead
            # term to the position command: compensated = target + cmd_vel × lead_gain.
            # Mode 6 lacks velocity feedforward — the anticipatory position target
            # gives the firmware's trajectory planner a "head start".
            #
            # arm_loop runs at 30Hz but new targets arrive at 16Hz.  On hold ticks
            # (no new action) we reuse the last velocity estimate AND the last
            # uncompensated target (_current_raw_target).  Critically, we do NOT
            # snapshot last_target as the base — it carries compensation from the
            # previous tick, and using it would accumulate lead on every hold tick
            # (verified: 5 hold ticks → +28.7° drift).
            _loop_time = time.monotonic()

            if _new_action:
                _current_raw_target = last_target.copy()  # uncompensated snapshot

            if _ff_enabled and _ff_skip_count <= 0 and _current_raw_target is not None:
                if _new_action and _prev_raw_target is not None and _prev_target_time is not None:
                    _dt = max(_loop_time - _prev_target_time, 1e-6)
                    _cmd_vel = (_current_raw_target - _prev_raw_target) / _dt  # 7-DOF (rad/s)
                    _last_cmd_vel = _cmd_vel
                # else: hold tick — reuse _last_cmd_vel from previous new-action tick

                if _last_cmd_vel is not None:
                    _pos_err = _current_raw_target - last_qpos  # shrinks as arm approaches target
                    # Per-joint guard: only compensate joints where the arm is
                    # chasing the target in the same direction as cmd velocity.
                    # Prevents overshoot when a joint has passed the target while
                    # others are still catching up.
                    _guard = (_last_cmd_vel * _pos_err) > 0  # shape (7,) bool
                    _lead = np.where(
                        _guard,
                        _last_cmd_vel * _ff_lead_gain,
                        np.float64(0.0),
                    )
                    _lead = np.clip(
                        _lead,
                        -cfg.feedforward_max_correction_rad,
                        cfg.feedforward_max_correction_rad,
                    )
                    # mypy narrows _current_raw_target correctly but np.where
                    # dtype inference interacts poorly with the type guard.
                    _compensated: np.ndarray = _current_raw_target + _lead.astype(np.float64)
                    last_target = _compensated

            # ── Joint-limit safety clamp (defense-in-depth) ──
            # Feedforward can push last_target up to ±0.05 rad beyond the
            # IK target.  Clamp to hardware limits so we never trigger C31
            # (joint limit) on a compensated command.
            last_target = np.clip(last_target, _joint_lo, _joint_hi)

            if _new_action:
                _prev_raw_target = _current_raw_target  # for next velocity estimate
                _prev_target_time = _loop_time

            if _ff_skip_count > 0:
                _ff_skip_count -= 1

            try:
                code = arm.set_servo_angle(angle=last_target, is_radian=True,
                                           speed=cfg.joint_max_speed_rad_per_s, mvacc=cfg.joint_max_acc_rad_per_s2, wait=False)
                if code != 0:
                    err_code = getattr(arm, "error_code", 0)
                    if err_code in _RECOVERABLE_ERRORS:
                        # C22/C24/C31 — recoverable arm errors.
                        # Standard recovery: clean error → ready state → re-enter Mode 6.
                        # set_state(0) MUST come before set_mode(6) — the arm needs to
                        # be in ready state before the mode change is accepted.
                        arm.clean_error()
                        arm.set_state(0)
                        arm.set_mode(6)
                        _consecutive_recoveries += 1
                        if _consecutive_recoveries > _RECOVERY_MAX:
                            logger.error("arm_loop: %d consecutive recoveries — escalating to FAULT", _consecutive_recoveries)
                            shared.error_state.value = True
                            transition(shared, SafetyState.FAULT)
                            break
                    elif err_code != 0:
                        logger.error("arm_loop: set_servo_angle code=%d err=%d — non-recoverable", code, err_code)
                        shared.error_state.value = True
                        transition(shared, SafetyState.FAULT)
                        break
                    else:
                        # code != 0 but err_code == 0 (e.g. ERR_CODE=1 transient error,
                        # mode drop).  Same recovery as above; do NOT FAULT on a
                        # single failure — transient comm glitches can self-resolve.
                        # The _RECOVERY_MAX counter gates escalation.
                        logger.warning("arm_loop: set_servo_angle code=%d (no arm error) — attempting mode recovery", code)
                        arm.clean_error()
                        arm.set_state(0)
                        arm.set_mode(6)
                        _consecutive_recoveries += 1
                        if _consecutive_recoveries > _RECOVERY_MAX:
                            logger.error("arm_loop: %d consecutive recoveries — escalating to FAULT", _consecutive_recoveries)
                            shared.error_state.value = True
                            transition(shared, SafetyState.FAULT)
                            break
            except Exception:
                logger.warning("arm_loop: set_servo_angle failed", exc_info=True)
                _consecutive_recoveries += 1
                if _consecutive_recoveries > _RECOVERY_MAX:
                    logger.error("arm_loop: %d consecutive exceptions — escalating to FAULT", _consecutive_recoveries)
                    shared.error_state.value = True
                    transition(shared, SafetyState.FAULT)
                    break
            else:
                # code == 0: successful send — reset recovery streak
                _consecutive_recoveries = 0

        # Read state
        arm_connected = True
        _state_ts = time.monotonic()  # capture BEFORE FK — timestamp = joint read time
        try:
            code, states = arm.get_joint_states(is_radian=True, num=3)
            if code == 0 and len(states) > 0:
                qpos = np.asarray(states[0], dtype=np.float64)[:7]
                qvel = np.asarray(states[1], dtype=np.float64)[:7] if len(states) > 1 else np.zeros(7)
                tau = np.asarray(states[2], dtype=np.float64)[:7] if len(states) > 2 else np.zeros(7)
                last_qpos = qpos.copy()
            else:
                qpos, qvel, tau = last_qpos.copy(), np.zeros(7), np.zeros(7)
                arm_connected = False
        except Exception:
            logger.warning("arm_loop: get_joint_states failed", exc_info=True)
            qpos, qvel, tau = last_qpos.copy(), np.zeros(7), np.zeros(7)
            arm_connected = False

        # FK: Pinocchio URDF-consistent FK (NOT arm firmware get_position_aa).
        # The xArm firmware uses a different EEF coordinate definition; using
        # URDF FK ensures consumers (IK, recording, display) share one system.
        try:
            eef_pos, eef_rot6d = _arm_fk.compute(qpos)
        except Exception:
            _fk_warn("arm_loop: Pinocchio FK failed — publishing zero EEF")
            eef_pos = np.zeros(3, dtype=np.float64)
            eef_rot6d = np.zeros(6, dtype=np.float64)

        # Compute tracking error
        tracking_err = float(np.max(np.abs(qpos - last_target)))

        if tracking_err > cfg.tracking_error_warn_rad:
            _tracking_warn("arm_loop: tracking_err=%.3f_rad threshold=%.3f_rad", tracking_err, cfg.tracking_error_warn_rad)

        # Error handling
        try:
            error_code = arm.error_code
        except Exception:
            error_code = 0
            arm_connected = False

        if error_code in _RECOVERABLE_ERRORS:
            _consecutive_state_errors += 1
            if _consecutive_state_errors > _RECOVERY_MAX:
                logger.error("arm_loop: %d consecutive state-read errors — escalating to FAULT", _consecutive_state_errors)
                shared.error_state.value = True
                transition(shared, SafetyState.FAULT)
                break
            try:
                arm.clean_error()
                arm.set_mode(6)
                arm.set_state(0)
            except Exception:
                pass
        elif error_code != 0:
            shared.error_state.value = True
            transition(shared, SafetyState.FAULT)
            break
        else:
            _consecutive_state_errors = 0

        # Publish state
        frame = new_frame(ARM_STATE_DTYPE)
        frame["qpos"][0] = qpos
        frame["qvel"][0] = qvel
        frame["tau"][0] = tau
        frame["eef_pos"][0] = eef_pos
        frame["eef_rot6d"][0] = eef_rot6d
        frame["error_code"][0] = int(error_code)
        frame["connected"][0] = 1 if arm_connected else 0
        frame["mode"][0] = 6
        frame["tracking_err"][0] = tracking_err
        frame["timestamp"][0] = _state_ts
        shared.arm_state_ring.write(frame)

        # Rate limit
        limiter.wait()

    # Cleanup
    try:
        arm.set_state(4)
        arm.disconnect()
    except Exception:
        logger.warning("arm_loop: cleanup failed", exc_info=True)
    logger.info("arm_loop: exited")


def _disconnect_arm(arm: Any) -> None:
    """Disconnect arm safely, ignoring errors."""
    try:
        arm.disconnect()
    except Exception:
        pass


def _planned_homing(arm: Any, waypoints: np.ndarray | None, home_qpos: np.ndarray,
                    cfg: ArmLoopConfig | None = None, *, shared: Any = None) -> None:
    """Execute planned waypoints, then converge to exact home_qpos.

    When *waypoints* is ``None`` or empty, falls back to joint-space linear
    interpolation (the old ``_simple_homing`` path).

    Writes heartbeat to ``shared.arm_heartbeat_s`` during execution so that
    the homing sequence does not trigger a false FAULT timeout.
    """
    _cfg = cfg or ArmLoopConfig()

    try:
        code, states = arm.get_joint_states(is_radian=True, num=1)
        if code == 0 and len(states) > 0:
            current = np.asarray(states[0], dtype=np.float64)[:7]
        else:
            return
    except Exception:
        return

    # ── Wrap home_qpos to nearest equivalent of current position ──
    # Prevents Stage 2 from taking the long way around for equivalent joints
    # (J1/J3/J5/J7 on xArm7, 720° range).  Wrapping home→current keeps all
    # interpolation waypoints and the final set_servo_angle target in the arm's
    # current encoder band.
    _home = wrap_nearest_equivalent(
        home_qpos, current,
        _cfg.joint_limit_lower, _cfg.joint_limit_upper,
    )

    if np.max(np.abs(current - _home)) < _cfg.homing_convergence_rad:
        return

    # ── Stage 1: execute planned waypoints (collision-safe path) ──
    if waypoints is not None and len(waypoints) > 0:
        for _wp in waypoints:
            if shared is not None:
                if not shared.is_running.value:
                    return
                shared.arm_heartbeat_s.value = time.monotonic()
            try:
                arm.set_servo_angle(angle=_wp, is_radian=True, wait=False)
            except Exception:
                break
            time.sleep(_cfg.homing_step_interval_s)

    # ── Stage 2: converge to exact home_qpos (fine positioning) ──
    if shared is not None and not shared.is_running.value:
        return

    # Re-read current position — Stage 1 may have moved the arm.
    try:
        code, states = arm.get_joint_states(is_radian=True, num=1)
        if code == 0 and len(states) > 0:
            current = np.asarray(states[0], dtype=np.float64)[:7]
    except Exception:
        pass

    # Re-wrap after Stage 1 — the arm may have settled in a different band.
    _home = wrap_nearest_equivalent(
        home_qpos, current,
        _cfg.joint_limit_lower, _cfg.joint_limit_upper,
    )

    if np.max(np.abs(current - _home)) < _cfg.homing_convergence_rad:
        return

    # Compute step count from max joint delta and configured speed (30°/s default).
    _max_delta_rad = float(np.max(np.abs(_home - current)))
    _total_time_s = _max_delta_rad / _cfg.homing_max_speed_rad_per_s
    steps = max(int(_total_time_s / _cfg.homing_step_interval_s), 10)
    for i in range(1, steps + 1):
        if shared is not None:
            if not shared.is_running.value:
                break
            shared.arm_heartbeat_s.value = time.monotonic()
        wp = current + (i / steps) * (_home - current)
        try:
            arm.set_servo_angle(angle=wp, is_radian=True, wait=False)
        except Exception:
            break
        time.sleep(_cfg.homing_step_interval_s)

    # Final: send canonical home_qpos to align encoder values.
    # Stage 2 converged to _home (shortest equivalent path).  If _home and
    # home_qpos are in different ±360° bands, this final set_servo_angle
    # triggers a deliberate full rotation to the canonical band at 90°/s —
    # acceptable because the arm is already at the physical home pose.
    # Mode 6 firmware handles trajectory interpolation internally.
    _align_speed = np.deg2rad(90.0)
    try:
        arm.set_servo_angle(angle=home_qpos, is_radian=True,
                           speed=_align_speed, wait=False)
    except Exception:
        pass
