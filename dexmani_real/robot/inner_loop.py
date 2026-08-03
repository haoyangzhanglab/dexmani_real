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

import numpy as np
from scipy.spatial.transform import Rotation

from dexmani_real.config.defaults import arm
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate_manager import RateManager
from dexmani_real.utils.throttle import ThrottledWarner

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════


@dataclass
class ArmInnerLoopConfig:
    """Configuration for ArmInnerLoop (Mode 6: joint online trajectory planning).

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

    # Homing parameters for _simple_homing.
    homing_convergence_rad: float = field(default_factory=lambda: arm.homing.convergence_rad)
    homing_step_count: int = field(default_factory=lambda: arm.homing.step_count)
    homing_step_interval_s: float = field(default_factory=lambda: arm.homing.step_interval_s)
    homing_target_timeout_s: float = field(default_factory=lambda: arm.homing.target_timeout_s)


# Controller errors that indicate a problematic target rather than a hardware fault.
_RECOVERABLE_ERRORS: frozenset[int] = arm.recoverable_errors
_RECOVERY_MAX: int = 30  # consecutive recoveries before FAULT escalation (1s @ 30Hz)


# ═══════════════════════════════════════════════════════════════════
# arm_loop (mp.Process target)
# ═══════════════════════════════════════════════════════════════════


def arm_loop(shared, config: ArmInnerLoopConfig | None = None) -> None:
    """Arm process entry point — reads SharedStorage.arm_action_q, servos arm.

    Designed as an mp.Process target. Communicates exclusively through
    SharedStorage (no RPC, no side channels).
    """
    from queue import Empty

    from dexmani_real.shm.shared_storage import HOME_SENTINEL, ARM_STATE_DTYPE, new_frame
    from dexmani_real.robot.safety import SafetyState, transition

    _tracking_warn = ThrottledWarner(interval_s=5.0)
    _fk_warn = ThrottledWarner(interval_s=5.0)
    _consecutive_recoveries = 0
    _consecutive_state_errors = 0
    cfg = config or ArmInnerLoopConfig()

    HOME_QPOS = np.array(cfg.home_qpos, dtype=np.float64)

    try:
        from xarm.wrapper import XArmAPI
        arm = XArmAPI(cfg.arm_ip, is_radian=True)
    except Exception as e:
        logger.error("arm_loop: connect failed: %s", e)
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
            _disconnect_arm(arm)
            return
    except Exception as e:
        logger.error("arm_loop: init failed: %s", e)
        _disconnect_arm(arm)
        return

    # Seed last_qpos — FAIL if initial state unreadable (safety: never cmd HOME_QPOS blind).
    try:
        code, states = arm.get_joint_states(is_radian=True, num=1)
        if code == 0 and len(states) > 0:
            last_qpos = np.asarray(states[0], dtype=np.float64)[:7].copy()
        else:
            logger.error("arm_loop: cannot read initial joint states (code=%d)", code)
            _disconnect_arm(arm)
            return
    except Exception as e:
        logger.error("arm_loop: joint states read failed: %s", e)
        _disconnect_arm(arm)
        return
    last_target = last_qpos.copy()

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

            # HOME sentinel → homing
            if action == HOME_SENTINEL:
                logger.info("arm_loop: HOME sentinel — executing homing")
                _simple_homing(arm, HOME_QPOS, cfg, shared=shared)
                last_qpos = HOME_QPOS.copy()
                last_target = HOME_QPOS.copy()
                continue

            # Servo
            if action is not None and isinstance(action, dict):
                target = np.asarray(action.get("qpos", last_target), dtype=np.float64).ravel()[:7]
                if np.all(np.isfinite(target)):
                    last_target = target.copy()

            try:
                code = arm.set_servo_angle(angle=last_target, is_radian=True,
                                           speed=cfg.joint_max_speed_rad_per_s, mvacc=cfg.joint_max_acc_rad_per_s2, wait=False)
                if code != 0:
                    err_code = getattr(arm, "error_code", 0)
                    if err_code in _RECOVERABLE_ERRORS:
                        arm.clean_error()
                        arm.set_mode(6)
                        arm.set_state(0)
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
                        # code != 0 but no arm error (e.g. mode drop, code 9).
                        # Attempt mode recovery; if it fails, escalate to FAULT.
                        logger.warning("arm_loop: set_servo_angle code=%d (no arm error) — attempting mode recovery", code)
                        arm.clean_error()
                        arm.set_mode(6)
                        arm.set_state(0)
                        _consecutive_recoveries += 1
                        if _consecutive_recoveries > _RECOVERY_MAX:
                            logger.error("arm_loop: %d consecutive recoveries — escalating to FAULT", _consecutive_recoveries)
                            shared.error_state.value = True
                            transition(shared, SafetyState.FAULT)
                            break
                        if getattr(arm, "mode", -1) != 6:
                            logger.error("arm_loop: mode recovery failed (mode=%d)", getattr(arm, "mode", -1))
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

        # FK: read actual EEF pose from arm controller
        try:
            fk_code, fk_pose = arm.get_position_aa(is_radian=True)
            if fk_code == 0 and len(fk_pose) >= 6:
                eef_pos = np.asarray(fk_pose[:3], dtype=np.float64) / 1000.0  # mm → m
                rx, ry, rz = float(fk_pose[3]), float(fk_pose[4]), float(fk_pose[5])
                R = Rotation.from_rotvec([rx, ry, rz]).as_matrix()
                eef_rot6d = np.concatenate([R[:, 0], R[:, 1]]).astype(np.float64)
            else:
                _fk_warn("arm_loop: FK failed code=%d — publishing zero EEF", fk_code)
                eef_pos = np.zeros(3, dtype=np.float64)
                eef_rot6d = np.zeros(6, dtype=np.float64)
        except Exception:
            _fk_warn("arm_loop: FK failed — publishing zero EEF")
            eef_pos = np.zeros(3, dtype=np.float64)
            eef_rot6d = np.zeros(6, dtype=np.float64)

        # Compute tracking error
        tracking_err = float(np.max(np.abs(qpos - last_target)))

        if tracking_err > cfg.tracking_error_warn_rad:
            _tracking_warn("arm_loop: tracking error %.3f rad > threshold %.3f rad", tracking_err, cfg.tracking_error_warn_rad)

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


def _disconnect_arm(arm) -> None:
    """Disconnect arm safely, ignoring errors."""
    try:
        arm.disconnect()
    except Exception:
        pass


def _simple_homing(arm, home_qpos: np.ndarray, cfg: ArmInnerLoopConfig | None = None, *, shared=None) -> None:
    """Simple joint-space linear interpolation to home.

    Writes heartbeat to ``shared.arm_heartbeat_s`` during execution so that
    the 2 s homing sequence does not trigger a false FAULT timeout (1 s).
    """
    _cfg = cfg or ArmInnerLoopConfig()

    try:
        code, states = arm.get_joint_states(is_radian=True, num=1)
        if code == 0 and len(states) > 0:
            current = np.asarray(states[0], dtype=np.float64)[:7]
        else:
            return
    except Exception:
        return

    if np.max(np.abs(current - home_qpos)) < _cfg.homing_convergence_rad:
        return

    steps = _cfg.homing_step_count
    for i in range(1, steps + 1):
        if shared is not None:
            if not shared.is_running.value:
                break
            shared.arm_heartbeat_s.value = time.monotonic()
        wp = current + (i / steps) * (home_qpos - current)
        try:
            arm.set_servo_angle(angle=wp, is_radian=True, wait=False)
        except Exception:
            break
        time.sleep(_cfg.homing_step_interval_s)

    try:
        arm.set_servo_angle(angle=home_qpos, is_radian=True, wait=False)
    except Exception:
        pass
