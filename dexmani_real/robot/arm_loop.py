"""Arm servo loop — Mode 6 joint online trajectory planning for xArm7.

Primary entry point: ``arm_loop(shared)`` — mp.Process target using SharedStorage.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dexmani_real import ASSET_DIR
from dexmani_real.config.defaults import arm, safety
from dexmani_real.planning.kinematics import ArmFK
from dexmani_real.planning.path_utils import wrap_nearest_equivalent
from dexmani_real.robot.safety import SafetyState, transition
from dexmani_real.shm.shared_storage import ARM_STATE_DTYPE, HOME_SENTINEL, new_frame
from dexmani_real.utils.log import ThrottledWarner, get_logger
from dexmani_real.utils.rate_manager import RateManager

logger = get_logger(__name__)


@dataclass
class ArmLoopConfig:
    """Mode 6 joint online trajectory planning configuration."""

    joint_max_speed_rad_per_s: float = field(default_factory=lambda: arm.max_joint_velocity_rad_per_s)
    joint_max_acc_rad_per_s2: float = field(default_factory=lambda: arm.max_joint_acceleration_rad_per_s2)
    arm_loop_hz: float = field(default_factory=lambda: arm.loop_hz)

    joint_limit_lower: tuple[float, ...] = field(default_factory=lambda: arm.joint_limit_lower)
    joint_limit_upper: tuple[float, ...] = field(default_factory=lambda: arm.joint_limit_upper)

    tracking_error_warn_rad: float = field(default_factory=lambda: arm.tracking_error_warn_rad)

    arm_ip: str = field(default_factory=lambda: arm.ip)

    home_qpos: tuple[float, ...] = field(default_factory=lambda: arm.home_qpos)

    collision_sensitivity: int = field(default_factory=lambda: arm.collision_sensitivity)

    homing_convergence_rad: float = field(default_factory=lambda: arm.homing.convergence_rad)
    homing_step_interval_s: float = field(default_factory=lambda: arm.homing.step_interval_s)
    homing_max_speed_rad_per_s: float = field(default_factory=lambda: np.deg2rad(arm.homing.max_speed_deg_s))
    homing_target_timeout_s: float = field(default_factory=lambda: arm.homing.target_timeout_s)


# Controller errors that indicate a problematic target rather than a hardware fault.
_RECOVERABLE_ERRORS: frozenset[int] = arm.recoverable_errors
_RECOVERY_MAX: int = safety.max_consecutive_recoveries  # consecutive recoveries before FAULT escalation (1s @ 30Hz)


def arm_loop(shared, config: ArmLoopConfig | None = None) -> None:
    """Arm process entry point — reads arm_action_q, servos arm via Mode 6.

    mp.Process target communicating exclusively through SharedStorage.
    """
    from queue import Empty

    _tracking_warn = ThrottledWarner(interval_s=5.0)
    _fk_warn = ThrottledWarner(interval_s=5.0)
    _state_read_warn = ThrottledWarner(interval_s=5.0)
    _consecutive_recoveries = 0
    _consecutive_state_errors = 0
    _tracking_err_count = 0
    cfg = config or ArmLoopConfig()

    HOME_QPOS = np.array(cfg.home_qpos, dtype=np.float64)

    # URDF-consistent FK (replaces arm.get_position_aa). xArm firmware uses a
    # different EEF coordinate definition — Pinocchio FK ensures all consumers
    # share a single coordinate system.
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
        # Torque-based collision detection (level 1). Detects impacts but may
        # miss slow contact. Primary table protection: self-collision + z-clearance.
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
    _frame = new_frame(ARM_STATE_DTYPE)
    _frame["qpos"][0] = last_qpos
    _frame["qvel"][0] = np.zeros(7, dtype=np.float64)
    _frame["tau"][0] = np.zeros(7, dtype=np.float64)
    _frame["eef_pos"][0] = eef_pos_init
    _frame["eef_rot6d"][0] = eef_rot6d_init
    _frame["error_code"][0] = 0
    _frame["connected"][0] = 1
    _frame["mode"][0] = getattr(arm, "mode", 6)
    _frame["tracking_err"][0] = 0.0
    _frame["timestamp"][0] = time.monotonic()
    shared.arm_state_ring.write(_frame)

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
            break

        # Safety state gate — only process commands in ARMED or RUNNING.
        # When gated (DISARMED or FAULT), skip action read + servo but continue
        # to publish state (for monitoring) and rate-limit normally.
        _safety = shared.safety_state.value
        if _safety in (SafetyState.ARMED, SafetyState.RUNNING):

            try:
                action = shared.arm_action_q.get(timeout=0.0)
            except Empty:
                action = None

            # HOME sentinel (tuple: (HOME_SENTINEL, waypoints_or_None))
            if isinstance(action, tuple) and len(action) == 2 and action[0] == HOME_SENTINEL:
                _waypoints = action[1]
                logger.info(
                    "arm_loop: HOME sentinel — planned homing (%d waypoints)",
                    len(_waypoints) if _waypoints is not None else 0,
                )
                _planned_homing(arm, _waypoints, HOME_QPOS, cfg, shared=shared)
                # Read actual state after homing — arm may still be settling
                # (all servo commands use wait=False)
                try:
                    code, states = arm.get_joint_states(is_radian=True, num=1)
                    if code == 0 and len(states) > 0:
                        last_qpos = np.asarray(states[0], dtype=np.float64)[:7].copy()
                    else:
                        last_qpos = HOME_QPOS.copy()
                except Exception:
                    last_qpos = HOME_QPOS.copy()
                last_target = last_qpos.copy()
                # Publish post-homing state so consumers see the final position
                try:
                    eef_pos_home, eef_rot6d_home = _arm_fk.compute(last_qpos)
                except Exception:
                    eef_pos_home = np.zeros(3, dtype=np.float64)
                    eef_rot6d_home = np.zeros(6, dtype=np.float64)
                _frame_home = new_frame(ARM_STATE_DTYPE)
                _frame_home["qpos"][0] = last_qpos
                _frame_home["qvel"][0] = np.zeros(7, dtype=np.float64)
                _frame_home["tau"][0] = np.zeros(7, dtype=np.float64)
                _frame_home["eef_pos"][0] = eef_pos_home
                _frame_home["eef_rot6d"][0] = eef_rot6d_home
                _frame_home["error_code"][0] = 0
                _frame_home["connected"][0] = 1
                _frame_home["mode"][0] = getattr(arm, "mode", 6)
                _frame_home["tracking_err"][0] = 0.0
                _frame_home["timestamp"][0] = time.monotonic()
                shared.arm_state_ring.write(_frame_home)
                continue

            _new_action = action is not None and isinstance(action, dict)
            if _new_action:
                target = np.asarray(action.get("qpos", last_target), dtype=np.float64).ravel()[:7]
                if np.all(np.isfinite(target)):
                    # Wrap equivalent joints to the same 2π band for shortest
                    # path. Mismatched bands cause the joint to rotate full
                    # circle (~2π) — defense-in-depth for IK edge cases.
                    target = wrap_nearest_equivalent(
                        target,
                        last_qpos,
                        cfg.joint_limit_lower,
                        cfg.joint_limit_upper,
                    )
                    last_target = target

            try:
                code = arm.set_servo_angle(
                    angle=last_target,
                    is_radian=True,
                    speed=cfg.joint_max_speed_rad_per_s,
                    mvacc=cfg.joint_max_acc_rad_per_s2,
                    wait=False,
                )
                if code != 0:
                    err_code = getattr(arm, "error_code", 0)
                    if err_code in _RECOVERABLE_ERRORS:
                        # C22/C24/C31 — recoverable. Clean error → ready state →
                        # re-enter Mode 6. set_state(0) MUST precede set_mode(6).
                        arm.clean_error()
                        arm.set_state(0)
                        arm.set_mode(6)
                        _consecutive_recoveries += 1
                        if _consecutive_recoveries > _RECOVERY_MAX:
                            logger.error(
                                "arm_loop: %d consecutive recoveries — escalating to FAULT", _consecutive_recoveries
                            )
                            shared.error_state.value = True
                            transition(shared, SafetyState.FAULT)
                            break
                    elif err_code != 0:
                        logger.error("arm_loop: set_servo_angle code=%d err=%d — non-recoverable", code, err_code)
                        shared.error_state.value = True
                        transition(shared, SafetyState.FAULT)
                        break
                    else:
                        # code != 0, err_code == 0 — transient glitch; same recovery, _RECOVERY_MAX gates escalation.
                        logger.warning(
                            "arm_loop: set_servo_angle code=%d (no arm error) — attempting mode recovery", code
                        )
                        arm.clean_error()
                        arm.set_state(0)
                        arm.set_mode(6)
                        _consecutive_recoveries += 1
                        if _consecutive_recoveries > _RECOVERY_MAX:
                            logger.error(
                                "arm_loop: %d consecutive recoveries — escalating to FAULT", _consecutive_recoveries
                            )
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

        arm_connected = True
        try:
            code, states = arm.get_joint_states(is_radian=True, num=3)
            if code == 0 and len(states) > 0:
                _state_ts = time.monotonic()  # timestamp = post-read time
                qpos = np.asarray(states[0], dtype=np.float64)[:7]
                qvel = np.asarray(states[1], dtype=np.float64)[:7] if len(states) > 1 else np.zeros(7)
                tau = np.asarray(states[2], dtype=np.float64)[:7] if len(states) > 2 else np.zeros(7)
                last_qpos = qpos.copy()
            else:
                _state_ts = time.monotonic()
                _state_read_warn("arm_loop: get_joint_states returned code=%d", code)
                qpos, qvel, tau = last_qpos.copy(), np.zeros(7), np.zeros(7)
                arm_connected = False
        except Exception:
            _state_ts = time.monotonic()
            logger.warning("arm_loop: get_joint_states failed", exc_info=True)
            qpos, qvel, tau = last_qpos.copy(), np.zeros(7), np.zeros(7)
            arm_connected = False

        # Pinocchio URDF-consistent FK (see note above).
        try:
            eef_pos, eef_rot6d = _arm_fk.compute(qpos)
        except Exception:
            _fk_warn("arm_loop: Pinocchio FK failed — publishing zero EEF")
            eef_pos = np.zeros(3, dtype=np.float64)
            eef_rot6d = np.zeros(6, dtype=np.float64)

        tracking_err = float(np.max(np.abs(qpos - last_target)))

        if tracking_err > cfg.tracking_error_warn_rad:
            _tracking_err_count += 1
            if _tracking_err_count >= 3:
                _tracking_warn(
                    "arm_loop: tracking_err=%.3f_rad threshold=%.3f_rad", tracking_err, cfg.tracking_error_warn_rad
                )
        else:
            _tracking_err_count = 0

        # arm.error_code is an SDK cached property (background report thread
        # ~every 200ms), not a per-access network call.
        try:
            error_code = arm.error_code
        except Exception:
            error_code = 0
            arm_connected = False

        if error_code in _RECOVERABLE_ERRORS:
            _consecutive_state_errors += 1
            if _consecutive_state_errors > _RECOVERY_MAX:
                logger.error(
                    "arm_loop: %d consecutive state-read errors — escalating to FAULT", _consecutive_state_errors
                )
                shared.error_state.value = True
                transition(shared, SafetyState.FAULT)
                break
            try:
                arm.clean_error()
                arm.set_state(0)
                arm.set_mode(6)
            except Exception:
                logger.warning("arm_loop: state-read recovery failed", exc_info=True)
        elif error_code != 0:
            shared.error_state.value = True
            transition(shared, SafetyState.FAULT)
            break
        else:
            _consecutive_state_errors = 0

        # Publish state
        _frame["qpos"][0] = qpos
        _frame["qvel"][0] = qvel
        _frame["tau"][0] = tau
        _frame["eef_pos"][0] = eef_pos
        _frame["eef_rot6d"][0] = eef_rot6d
        _frame["error_code"][0] = int(error_code)
        _frame["connected"][0] = 1 if arm_connected else 0
        _frame["mode"][0] = getattr(arm, "mode", 6)
        _frame["tracking_err"][0] = tracking_err
        _frame["timestamp"][0] = _state_ts
        shared.arm_state_ring.write(_frame)

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


def _planned_homing(
    arm: Any,
    waypoints: np.ndarray | None,
    home_qpos: np.ndarray,
    cfg: ArmLoopConfig | None = None,
    *,
    shared: Any = None,
) -> None:
    """Execute planned waypoints, then converge to exact home_qpos.

    Falls back to joint-space linear interpolation when waypoints is None.
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

    # Wrap home_qpos to nearest equivalent of current position — prevents
    # the long way around for equivalent joints (J1/J3/J5/J7, 720° range).
    _home = wrap_nearest_equivalent(
        home_qpos,
        current,
        _cfg.joint_limit_lower,
        _cfg.joint_limit_upper,
    )

    if np.max(np.abs(current - _home)) < _cfg.homing_convergence_rad:
        return

    # Stage 1: execute planned waypoints (collision-safe path).
    if waypoints is not None and len(waypoints) > 0:
        for _wp in waypoints:
            if shared is not None:
                if not shared.is_running.value or shared.safety_state.value == SafetyState.FAULT:
                    return
                shared.arm_heartbeat_s.value = time.monotonic()
            try:
                arm.set_servo_angle(angle=_wp, is_radian=True, wait=False)
            except Exception:
                break
            time.sleep(_cfg.homing_step_interval_s)
    elif waypoints is not None and len(waypoints) == 0:
        # Empty (0,7) array = sentinel from plan_joint_home_path: no safe path
        # to home exists.  Hold position — do NOT fall back to raw linear
        # interpolation (which has zero collision checking).
        logger.warning("_planned_homing: no safe path to home — holding position")
        return

    # Stage 2: converge to exact home_qpos (fine positioning). Linear interpolation
    # without collision check — safe because Stage 1 brings arm close to home.
    if shared is not None and (not shared.is_running.value or shared.safety_state.value == SafetyState.FAULT):
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
        home_qpos,
        current,
        _cfg.joint_limit_lower,
        _cfg.joint_limit_upper,
    )

    if np.max(np.abs(current - _home)) < _cfg.homing_convergence_rad:
        return

    # Compute step count from max joint delta and configured speed (30°/s default).
    _max_delta_rad = float(np.max(np.abs(_home - current)))
    _total_time_s = _max_delta_rad / _cfg.homing_max_speed_rad_per_s
    steps = max(int(_total_time_s / _cfg.homing_step_interval_s), 10)
    for i in range(1, steps + 1):
        if shared is not None:
            if not shared.is_running.value or shared.safety_state.value == SafetyState.FAULT:
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
        arm.set_servo_angle(angle=home_qpos, is_radian=True, speed=_align_speed, wait=False)
    except Exception:
        pass
