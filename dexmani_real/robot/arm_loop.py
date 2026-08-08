"""Arm servo loop — Mode 6 joint online trajectory planning for xArm7.

Primary entry point: ``arm_loop(shared)`` — mp.Process target using SharedStorage.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from dexmani_real import ASSET_DIR
from dexmani_real.config.defaults import arm, safety
from dexmani_real.planning.kinematics import ArmFK
from dexmani_real.planning.path_utils import wrap_nearest_equivalent
from dexmani_real.robot.safety import SafetyState
from dexmani_real.shm.shared_storage import ARM_STATE_DTYPE, HOME_SENTINEL, HomeRequest, HomeResult, new_frame
from dexmani_real.utils.log import ThrottledWarner, get_logger
from dexmani_real.utils.rate_manager import RateManager
from dexmani_real.utils.retry import RetryCounter

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


# Controller errors: C24 is recoverable; C22/C31 are immediate collision faults.
_RECOVERABLE_ERRORS: frozenset[int] = arm.recoverable_errors
_COLLISION_FAULT_ERRORS: frozenset[int] = arm.collision_fault_errors
_RECOVERY_MAX: int = safety.max_consecutive_recoveries  # consecutive recoveries before FAULT escalation (1s @ 30Hz)


def _require_sdk_ok(operation: str, code: Any) -> None:
    """Raise when an xArm setter reports failure without raising."""
    if not isinstance(code, (int, np.integer)) or int(code) != 0:
        raise RuntimeError(f"{operation} failed with SDK code {code!r}")


def _parse_arm_action_metadata(action: dict[str, Any], received_s: float) -> tuple[int, float, bool]:
    """Return validated ``(sequence, created_s, is_hold)`` metadata.

    Legacy/externally constructed actions remain executable, but receive a
    sequence of zero and a zero-queue-age timestamp instead of publishing
    invalid latency diagnostics.
    """
    try:
        command_seq = int(action.get("command_seq", 0))
    except (TypeError, ValueError, OverflowError):
        command_seq = 0
    if command_seq < 0:
        command_seq = 0

    try:
        created_s = float(action.get("created_monotonic_s", received_s))
    except (TypeError, ValueError, OverflowError):
        created_s = received_s
    if not np.isfinite(created_s) or created_s <= 0.0 or created_s > received_s:
        created_s = received_s

    return command_seq, created_s, bool(action.get("is_hold", False))


def _latch_collision_fault(shared: Any, arm_api: Any, error_code: int) -> None:
    details: Any = None
    if error_code == 31 and hasattr(arm_api, "get_c31_error_info"):
        try:
            code, info = arm_api.get_c31_error_info()
            if code == 0:
                details = info
        except Exception:
            logger.warning("arm_loop: failed to read C31 diagnostics", exc_info=True)
    if error_code == 31 and isinstance(details, (list, tuple, np.ndarray)) and len(details) >= 3:
        try:
            servo_id = int(details[0])
            theoretical_tau = float(details[1])
            actual_tau = float(details[2])
        except (TypeError, ValueError, OverflowError):
            logger.error("arm_loop: collision fault C31 detected; details=%s", details)
        else:
            logger.error(
                "arm_loop: collision fault C31 detected; servo_id=%d "
                "theoretical_tau=%.3f actual_tau=%.3f delta_tau=%.3f",
                servo_id,
                theoretical_tau,
                actual_tau,
                actual_tau - theoretical_tau,
            )
    else:
        logger.error("arm_loop: collision fault C%d detected; details=%s", error_code, details)
    shared.error_state.value = True


def arm_loop(shared, config: ArmLoopConfig | None = None) -> None:
    """Arm process entry point — reads arm_action_q, servos arm via Mode 6.

    mp.Process target communicating exclusively through SharedStorage.
    """
    from queue import Empty

    _tracking_warn = ThrottledWarner(interval_s=5.0)
    _fk_warn = ThrottledWarner(interval_s=5.0)
    _state_read_warn = ThrottledWarner(interval_s=5.0)
    _recovery_counter = RetryCounter(max_consecutive=_RECOVERY_MAX, label="arm_servo")
    _state_error_counter = RetryCounter(max_consecutive=_RECOVERY_MAX, label="arm_state")
    _tracking_err_count = 0
    cfg = config or ArmLoopConfig()

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

    # 3-retry connect recovery (ref: pi-r2-flow xarm_sdk.py:61-81).
    # Sticky error states (post-collision, E-Stop, power-on race) may need
    # multiple clear cycles.  Transition through mode 0 first — some firmware
    # releases require this intermediate state.
    _CONNECT_MAX_RETRIES = 3
    for _attempt in range(_CONNECT_MAX_RETRIES):
        try:
            arm.clean_error()
            arm.clean_warn()
            arm.motion_enable(True)
            time.sleep(0.3)
            arm.set_mode(0)
            time.sleep(0.1)
            arm.set_state(0)
            time.sleep(0.3)
            arm.set_mode(6)
            time.sleep(0.1)
            # Live read via get_err_warn_code() — more reliable than the cached
            # .error_code property (background report thread, ~200ms refresh).
            try:
                _rc, _codes = arm.get_err_warn_code()
                _err = _codes[0] if _rc == 0 else 1
            except Exception:
                _err = getattr(arm, "error_code", 0) or 0
            _state = getattr(arm, "state", -1)
            if _err == 0 and _state == 2:  # 2 = ready-to-move
                break
            logger.warning(
                "arm_loop: connect recovery attempt %d/%d: err=%s state=%s",
                _attempt + 1,
                _CONNECT_MAX_RETRIES,
                _err,
                _state,
            )
            time.sleep(0.5)
        except Exception:
            logger.warning(
                "arm_loop: connect recovery attempt %d/%d raised exception",
                _attempt + 1,
                _CONNECT_MAX_RETRIES,
                exc_info=True,
            )
            time.sleep(0.5)
    else:
        logger.error("arm_loop: connect recovery failed after %d attempts", _CONNECT_MAX_RETRIES)
        shared.error_state.value = True
        _disconnect_arm(arm)
        return

    # Post-recovery configuration (only after successful mode-6 transition).
    try:
        _require_sdk_ok("set_collision_sensitivity", arm.set_collision_sensitivity(cfg.collision_sensitivity))
        # TCP load: XHand (1.1 kg). COG in tool-flange frame (link_eef) from
        # URDF weighted-COM of all end-effector links; flange_joint2 corrected
        # 0.043→0.033 m per physical measurement.
        _require_sdk_ok("set_tcp_load", arm.set_tcp_load(weight=1.1, center_of_gravity=[16.3, 7.9, 109.5]))
        # Torque-based collision detection (level 1, least-sensitive enabled
        # setting). Keep this firmware backstop enabled during intentional contact.
        _require_sdk_ok("set_joint_maxacc", arm.set_joint_maxacc(cfg.joint_max_acc_rad_per_s2, is_radian=True))
    except Exception as e:
        logger.error("arm_loop: post-recovery config failed: %s", e)
        shared.error_state.value = True
        _disconnect_arm(arm)
        return

    # Seed last_qpos — retry transient comm failures (ref: pi-r2-flow
    # control_utils.py:181-192 get_obs_retry).  A single failed read during
    # startup is not a reason to abort the process.
    _STATE_READ_MAX_RETRIES = 10
    for _attempt in range(_STATE_READ_MAX_RETRIES):
        try:
            code, states = arm.get_joint_states(is_radian=True, num=1)
            if code == 0 and len(states) > 0:
                last_qpos = np.asarray(states[0], dtype=np.float64)[:7].copy()
                break
            logger.warning(
                "arm_loop: initial joint state read attempt %d/%d: code=%d",
                _attempt + 1,
                _STATE_READ_MAX_RETRIES,
                code,
            )
        except Exception:
            logger.warning(
                "arm_loop: initial joint state read attempt %d/%d raised exception",
                _attempt + 1,
                _STATE_READ_MAX_RETRIES,
                exc_info=True,
            )
        time.sleep(0.1)
    else:
        logger.error("arm_loop: cannot read initial joint states after %d attempts", _STATE_READ_MAX_RETRIES)
        shared.error_state.value = True
        _disconnect_arm(arm)
        return
    last_target = last_qpos.copy()
    last_cmd_seq = 0
    last_cmd_created_s = 0.0
    last_cmd_received_s = 0.0
    last_cmd_applied_s = 0.0
    last_cmd_queue_latency_s = 0.0
    last_cmd_apply_latency_s = 0.0
    last_cmd_sdk_duration_s = 0.0
    last_cmd_is_hold = False

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
    _frame["last_cmd_seq"][0] = last_cmd_seq
    _frame["last_cmd_created_s"][0] = last_cmd_created_s
    _frame["last_cmd_received_s"][0] = last_cmd_received_s
    _frame["last_cmd_applied_s"][0] = last_cmd_applied_s
    _frame["last_cmd_queue_latency_s"][0] = last_cmd_queue_latency_s
    _frame["last_cmd_apply_latency_s"][0] = last_cmd_apply_latency_s
    _frame["last_cmd_sdk_duration_s"][0] = last_cmd_sdk_duration_s
    _frame["last_cmd_is_hold"][0] = int(last_cmd_is_hold)
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
            # SDK emergency_stop() is the fastest path to kill motor power.
            # Fall back to set_state(4) in cleanup if the SDK method is
            # unavailable or fails (belt-and-suspenders per Xarm7-).
            try:
                arm.emergency_stop()
            except Exception:
                pass
            break

        # Safety state gate — only process commands in ARMED or RUNNING.
        # When gated (DISARMED or FAULT), skip action read + servo but continue
        # to publish state (for monitoring) and rate-limit normally.
        _safety = shared.safety_state.value
        if _safety in (SafetyState.ARMED, SafetyState.RUNNING) and not shared.error_state.value:

            _action_received_s = 0.0
            try:
                action = shared.arm_action_q.get(timeout=0.0)
                _action_received_s = time.monotonic()
            except Empty:
                action = None

            # HOME sentinel carries a collision-validated path and a request ID.
            # Execution is feedback-driven; completion is acknowledged only
            # after fresh controller state converges to the canonical target.
            if isinstance(action, tuple) and len(action) == 2 and action[0] == HOME_SENTINEL:
                _request = action[1]
                if not isinstance(_request, HomeRequest):
                    logger.error("arm_loop: rejecting malformed HOME request")
                    continue
                logger.info(
                    "arm_loop: HOME sentinel — planned homing (%d validated milestones)",
                    len(_request.waypoints),
                )

                def _publish_homing_feedback(
                    qpos: np.ndarray,
                    qvel: np.ndarray,
                    tau: np.ndarray,
                    target: np.ndarray,
                ) -> None:
                    """Keep the state ring fresh while homing owns the SDK loop."""
                    _state_ts = time.monotonic()
                    try:
                        eef_pos, eef_rot6d = _arm_fk.compute(qpos)
                    except Exception:
                        _fk_warn("arm_loop: Pinocchio FK failed during homing — publishing zero EEF")
                        eef_pos = np.zeros(3, dtype=np.float64)
                        eef_rot6d = np.zeros(6, dtype=np.float64)
                    try:
                        error_code = int(getattr(arm, "error_code", 0) or 0)
                    except Exception:
                        error_code = 0
                    _frame["qpos"][0] = qpos
                    _frame["qvel"][0] = qvel
                    _frame["tau"][0] = tau
                    _frame["eef_pos"][0] = eef_pos
                    _frame["eef_rot6d"][0] = eef_rot6d
                    _frame["error_code"][0] = error_code
                    _frame["connected"][0] = 1
                    _frame["mode"][0] = getattr(arm, "mode", 6)
                    _frame["tracking_err"][0] = float(np.max(np.abs(qpos - target)))
                    _frame["timestamp"][0] = _state_ts
                    shared.arm_state_ring.write(_frame)

                _home_started_s = time.monotonic()
                _home_result = _planned_homing(
                    arm,
                    _request,
                    cfg,
                    shared=shared,
                    feedback_callback=_publish_homing_feedback,
                )
                try:
                    shared.arm_home_result_q.put(_home_result, timeout=0.2)
                except Exception:
                    logger.error("arm_loop: failed to publish HOME result", exc_info=True)
                if _home_result.final_qpos.shape == (7,) and np.all(np.isfinite(_home_result.final_qpos)):
                    last_qpos = _home_result.final_qpos.copy()
                    last_target = last_qpos.copy()
                if _home_result.success:
                    logger.info("arm_loop: HOME complete in %.2fs", time.monotonic() - _home_started_s)
                elif shared.is_running.value:
                    logger.error("arm_loop: HOME failed — %s", _home_result.reason)
                    shared.error_state.value = True
                continue

            _new_action = action is not None and isinstance(action, dict)
            _accepted_action_metadata: tuple[int, float, float, bool] | None = None
            if _new_action:
                target = np.asarray(action.get("qpos", last_target), dtype=np.float64).ravel()[:7]
                if np.all(np.isfinite(target)):
                    # Wrap equivalent joints to the same 2π band for shortest
                    # path. Mismatched bands cause the joint to rotate full
                    # circle (~2π) — defense-in-depth for IK edge cases.
                    try:
                        target = wrap_nearest_equivalent(
                            target, last_qpos, cfg.joint_limit_lower, cfg.joint_limit_upper
                        )
                    except ValueError:
                        logger.warning("arm_loop: invalid target joint vector — holding", exc_info=True)
                    else:
                        low = np.asarray(cfg.joint_limit_lower, dtype=np.float64)
                        high = np.asarray(cfg.joint_limit_upper, dtype=np.float64)
                        if np.all((target >= low) & (target <= high)):
                            last_target = target
                            _command_seq, _created_s, _is_hold = _parse_arm_action_metadata(action, _action_received_s)
                            _accepted_action_metadata = (
                                _command_seq,
                                _created_s,
                                _action_received_s,
                                _is_hold,
                            )
                        else:
                            logger.warning("arm_loop: no limit-valid equivalent target — holding")

            _sdk_started_s = time.monotonic()
            try:
                code = arm.set_servo_angle(
                    angle=last_target,
                    is_radian=True,
                    speed=cfg.joint_max_speed_rad_per_s,
                    mvacc=cfg.joint_max_acc_rad_per_s2,
                    wait=False,
                )
                _sdk_finished_s = time.monotonic()
                if code != 0:
                    err_code = getattr(arm, "error_code", 0)
                    if err_code in _COLLISION_FAULT_ERRORS:
                        _latch_collision_fault(shared, arm, err_code)
                        break
                    elif err_code in _RECOVERABLE_ERRORS:
                        # C24 speed-limit error — bounded recovery. Clean error + warn → ready state →
                        # re-enter Mode 6. set_state(0) MUST precede set_mode(6).
                        arm.clean_error()
                        arm.clean_warn()
                        arm.set_state(0)
                        arm.set_mode(6)
                        _recovery_counter.inc()
                        if _recovery_counter.triggered:
                            logger.error(
                                "arm_loop: %d consecutive recoveries — escalating to FAULT", _recovery_counter.count
                            )
                            shared.error_state.value = True
                            break
                    elif err_code != 0:
                        logger.error("arm_loop: set_servo_angle code=%d err=%d — non-recoverable", code, err_code)
                        shared.error_state.value = True
                        break
                    else:
                        # code != 0, err_code == 0 — transient glitch; same recovery, _RECOVERY_MAX gates escalation.
                        logger.warning(
                            "arm_loop: set_servo_angle code=%d (no arm error) — attempting mode recovery", code
                        )
                        arm.clean_error()
                        arm.clean_warn()
                        arm.set_state(0)
                        arm.set_mode(6)
                        _recovery_counter.inc()
                        if _recovery_counter.triggered:
                            logger.error(
                                "arm_loop: %d consecutive recoveries — escalating to FAULT", _recovery_counter.count
                            )
                            shared.error_state.value = True
                            break
            except Exception:
                _sdk_finished_s = time.monotonic()
                logger.warning("arm_loop: set_servo_angle failed", exc_info=True)
                _recovery_counter.inc()
                if _recovery_counter.triggered:
                    logger.error("arm_loop: %d consecutive exceptions — escalating to FAULT", _recovery_counter.count)
                    shared.error_state.value = True
                    break
            else:
                if code == 0:
                    _recovery_counter.reset()
                    if _accepted_action_metadata is not None:
                        (
                            last_cmd_seq,
                            last_cmd_created_s,
                            last_cmd_received_s,
                            last_cmd_is_hold,
                        ) = _accepted_action_metadata
                        last_cmd_applied_s = _sdk_finished_s
                        last_cmd_queue_latency_s = max(0.0, last_cmd_received_s - last_cmd_created_s)
                        last_cmd_apply_latency_s = max(0.0, last_cmd_applied_s - last_cmd_created_s)
                        last_cmd_sdk_duration_s = max(0.0, _sdk_finished_s - _sdk_started_s)

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

        if error_code in _COLLISION_FAULT_ERRORS:
            _latch_collision_fault(shared, arm, error_code)
            break
        elif error_code in _RECOVERABLE_ERRORS:
            _state_error_counter.inc()
            if _state_error_counter.triggered:
                logger.error(
                    "arm_loop: %d consecutive state-read errors — escalating to FAULT", _state_error_counter.count
                )
                shared.error_state.value = True
                break
            try:
                arm.clean_error()
                arm.clean_warn()
                arm.set_state(0)
                arm.set_mode(6)
            except Exception:
                logger.warning("arm_loop: state-read recovery failed", exc_info=True)
        elif error_code != 0:
            shared.error_state.value = True
            break
        else:
            _state_error_counter.reset()

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
        _frame["last_cmd_seq"][0] = last_cmd_seq
        _frame["last_cmd_created_s"][0] = last_cmd_created_s
        _frame["last_cmd_received_s"][0] = last_cmd_received_s
        _frame["last_cmd_applied_s"][0] = last_cmd_applied_s
        _frame["last_cmd_queue_latency_s"][0] = last_cmd_queue_latency_s
        _frame["last_cmd_apply_latency_s"][0] = last_cmd_apply_latency_s
        _frame["last_cmd_sdk_duration_s"][0] = last_cmd_sdk_duration_s
        _frame["last_cmd_is_hold"][0] = int(last_cmd_is_hold)
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
    request: HomeRequest,
    cfg: ArmLoopConfig | None = None,
    *,
    shared: Any = None,
    feedback_callback: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None] | None = None,
) -> HomeResult:
    """Execute collision-validated milestones with the firmware joint planner.

    The caller densely checks every joint-space segment for collision, but only
    the sparse segment endpoints cross the process boundary.  Homing temporarily
    enters Mode 0 and uses unblended ``MoveJoint`` commands so the controller,
    rather than this process, owns the point-to-point trajectory.  Normal Mode 6
    teleoperation is restored before returning from healthy paths; E-stop,
    shutdown, and controller-fault paths stop instead.  Completion is based
    only on fresh encoder feedback; no state is fabricated on SDK read failure.
    """
    _cfg = cfg or ArmLoopConfig()

    def _result(success: bool, reason: str, qpos: np.ndarray) -> HomeResult:
        return HomeResult(
            request_id=request.request_id,
            success=success,
            reason=reason,
            final_qpos=np.asarray(qpos, dtype=np.float64).copy(),
            completed_at_s=time.monotonic(),
        )

    def _shared_abort_reason() -> str | None:
        if shared is None:
            return None
        if not shared.is_running.value:
            return "shutdown requested"
        if shared.estop_request.value:
            return "e-stop requested"
        if shared.error_state.value:
            return "sticky error_state set during homing"
        if shared.safety_state.value == SafetyState.FAULT:
            return "FAULT during homing"
        return None

    waypoints = np.asarray(request.waypoints, dtype=np.float64)
    home_qpos = np.asarray(request.final_qpos, dtype=np.float64)
    if not isinstance(request.request_id, (int, np.integer)) or int(request.request_id) <= 0:
        return _result(False, "invalid request_id", np.full(7, np.nan))
    if waypoints.ndim != 2 or waypoints.shape[1:] != (7,) or not np.all(np.isfinite(waypoints)):
        return _result(False, "invalid waypoint array", np.full(7, np.nan))
    if home_qpos.shape != (7,) or not np.all(np.isfinite(home_qpos)):
        return _result(False, "invalid final_qpos", np.full(7, np.nan))
    if not np.isfinite(request.execution_timeout_s) or request.execution_timeout_s <= 0.0:
        return _result(False, "invalid execution timeout", np.full(7, np.nan))
    _lower = np.asarray(_cfg.joint_limit_lower, dtype=np.float64)
    _upper = np.asarray(_cfg.joint_limit_upper, dtype=np.float64)
    if len(waypoints) > 0 and not np.all((waypoints >= _lower) & (waypoints <= _upper)):
        return _result(False, "waypoint violates joint limits", np.full(7, np.nan))
    if len(waypoints) > 0 and float(np.max(np.abs(waypoints[-1] - home_qpos))) > 1e-6:
        return _result(False, "final milestone does not match canonical home", np.full(7, np.nan))

    try:
        code, states = arm.get_joint_states(is_radian=True, num=3)
        if code == 0 and len(states) > 0:
            current = np.asarray(states[0], dtype=np.float64)[:7]
        else:
            return _result(False, f"initial state read failed (code={code})", np.full(7, np.nan))
    except Exception:
        logger.warning("_planned_homing: initial state read raised", exc_info=True)
        return _result(False, "initial state read raised", np.full(7, np.nan))
    if current.shape != (7,) or not np.all(np.isfinite(current)):
        return _result(False, "initial state is invalid", np.full(7, np.nan))

    if len(waypoints) == 0:
        if float(np.max(np.abs(current - home_qpos))) <= _cfg.homing_convergence_rad:
            return _result(True, "already at canonical home", current)
        return _result(False, "empty path while away from canonical home", current)
    if float(np.max(np.abs(current - waypoints[0]))) > np.deg2rad(2.0):
        return _result(False, "current state moved too far from planned path start", current)

    _execution_targets = waypoints[1:]
    if len(_execution_targets) == 0:
        _final_error = float(np.max(np.abs(current - home_qpos)))
        if _final_error <= _cfg.homing_convergence_rad:
            return _result(True, "already at canonical home", current)
        return _result(False, f"final error {np.rad2deg(_final_error):.2f}deg", current)
    _preflight_abort = _shared_abort_reason()
    if _preflight_abort is not None:
        return _result(False, _preflight_abort, current)

    def _execute_mode0_milestones() -> HomeResult:
        nonlocal current
        _overall_deadline = time.monotonic() + request.execution_timeout_s
        _milestone_tol = min(_cfg.homing_convergence_rad, np.deg2rad(0.5))
        _stable_required = 2

        for _target_index, _target in enumerate(_execution_targets, start=1):
            if shared is not None:
                _abort_reason = _shared_abort_reason()
                if _abort_reason is not None:
                    return _result(False, _abort_reason, current)
                shared.arm_heartbeat_s.value = time.monotonic()
            if time.monotonic() >= _overall_deadline:
                return _result(
                    False,
                    f"overall timeout before milestone {_target_index}/{len(_execution_targets)}",
                    current,
                )

            _segment_start = current.copy()
            _segment_started_s = time.monotonic()
            try:
                _code = arm.set_servo_angle(
                    angle=_target,
                    is_radian=True,
                    speed=_cfg.homing_max_speed_rad_per_s,
                    mvacc=_cfg.joint_max_acc_rad_per_s2,
                    wait=False,
                    radius=None,
                )
            except Exception:
                logger.warning("_planned_homing: milestone send failed", exc_info=True)
                return _result(False, f"milestone {_target_index} send raised", current)
            if _code != 0:
                return _result(False, f"milestone {_target_index} rejected (SDK code={_code})", current)

            _segment_timeout_s = _estimate_homing_segment_timeout_s(_segment_start, _target, _cfg)
            _segment_deadline = min(_overall_deadline, _segment_started_s + _segment_timeout_s)
            _stable = 0
            while time.monotonic() < _segment_deadline:
                if shared is not None:
                    _abort_reason = _shared_abort_reason()
                    if _abort_reason is not None:
                        return _result(False, _abort_reason, current)
                    shared.arm_heartbeat_s.value = time.monotonic()
                try:
                    _state_code, _states = arm.get_joint_states(is_radian=True, num=3)
                except Exception:
                    logger.warning("_planned_homing: milestone state read raised", exc_info=True)
                    return _result(False, f"state read raised at milestone {_target_index}", current)
                if _state_code != 0 or len(_states) == 0:
                    return _result(
                        False,
                        f"state read failed at milestone {_target_index} (code={_state_code})",
                        current,
                    )
                current = np.asarray(_states[0], dtype=np.float64)[:7]
                if current.shape != (7,) or not np.all(np.isfinite(current)):
                    return _result(False, f"invalid state at milestone {_target_index}", current)
                qvel = np.asarray(_states[1], dtype=np.float64)[:7] if len(_states) > 1 else np.zeros(7)
                tau = np.asarray(_states[2], dtype=np.float64)[:7] if len(_states) > 2 else np.zeros(7)
                try:
                    _controller_error = int(getattr(arm, "error_code", 0) or 0)
                except Exception:
                    _controller_error = 0
                if _controller_error != 0:
                    return _result(
                        False,
                        f"controller error C{_controller_error} at milestone {_target_index}",
                        current,
                    )
                if feedback_callback is not None:
                    try:
                        feedback_callback(current.copy(), qvel.copy(), tau.copy(), _target.copy())
                    except Exception:
                        logger.warning("_planned_homing: feedback publication failed", exc_info=True)
                if float(np.max(np.abs(current - _target))) <= _milestone_tol:
                    _stable += 1
                    if _stable >= _stable_required:
                        break
                else:
                    _stable = 0
                time.sleep(_cfg.homing_step_interval_s)
            else:
                _error = np.abs(current - _target)
                _joint = int(np.argmax(_error))
                _elapsed_s = time.monotonic() - _segment_started_s
                if time.monotonic() >= _overall_deadline:
                    _timeout_kind = "overall timeout"
                else:
                    _timeout_kind = "convergence timeout"
                return _result(
                    False,
                    f"{_timeout_kind} at milestone {_target_index}/{len(_execution_targets)} "
                    f"after {_elapsed_s:.2f}s (J{_joint + 1} error={np.rad2deg(_error[_joint]):.2f}deg)",
                    current,
                )

        _final_error = float(np.max(np.abs(current - home_qpos)))
        if _final_error > _cfg.homing_convergence_rad:
            return _result(False, f"final error {np.rad2deg(_final_error):.2f}deg", current)
        return _result(True, "canonical home reached", current)

    # Mode 6 is designed for continuously changing online targets and its
    # per-joint velocity profiles need not be synchronous.  A planned homing
    # path instead uses Mode 0 MoveJoint.  Explicitly restore Mode 6 after
    # healthy entry/execution failures so the worker never silently changes
    # semantics; global-stop and controller-fault paths remain stopped.
    _mode_switch_attempted = False
    try:
        logger.info(
            "arm_loop: homing entering Mode 0 MoveJoint (%d motion milestones, speed=%.1fdeg/s)",
            len(_execution_targets),
            np.rad2deg(_cfg.homing_max_speed_rad_per_s),
        )
        _mode_switch_attempted = True
        _require_sdk_ok("set_mode(0)", arm.set_mode(0))
        _require_sdk_ok("set_state(0) after Mode 0", arm.set_state(0))
    except Exception as exc:
        logger.error("_planned_homing: failed to enter Mode 0", exc_info=True)
        _home_result = _result(False, f"Mode 0 entry failed: {exc}", current)
    else:
        _home_result = _execute_mode0_milestones()

    _restore_error: Exception | None = None
    _post_homing_abort = _shared_abort_reason()
    try:
        _controller_error_after_home = int(getattr(arm, "error_code", 0) or 0)
    except Exception:
        _controller_error_after_home = 0
    _restore_mode6 = _post_homing_abort is None and _controller_error_after_home == 0
    if _mode_switch_attempted and _restore_mode6:
        try:
            _require_sdk_ok("restore set_mode(6)", arm.set_mode(6))
            _require_sdk_ok("restore set_state(0)", arm.set_state(0))
        except Exception as exc:
            _restore_error = exc
            logger.error("_planned_homing: failed to restore Mode 6", exc_info=True)
    elif _mode_switch_attempted:
        _stop_reason = _post_homing_abort or f"controller error C{_controller_error_after_home}"
        try:
            _require_sdk_ok("stop after interrupted homing", arm.set_state(4))
        except Exception as exc:
            _restore_error = exc
            logger.error("_planned_homing: failed to stop after interrupted homing", exc_info=True)
        if _home_result.success:
            _home_result = _result(False, f"homing interrupted after convergence: {_stop_reason}", current)
    if _restore_error is not None:
        _operation = "Mode 6 restore" if _restore_mode6 else "safe stop"
        return _result(False, f"{_home_result.reason}; {_operation} failed: {_restore_error}", current)
    if _restore_mode6:
        logger.info("arm_loop: homing restored Mode 6")
    return _home_result


def _estimate_homing_segment_timeout_s(start: np.ndarray, target: np.ndarray, cfg: ArmLoopConfig) -> float:
    """Deadline for one firmware-planned milestone, including settle time."""
    delta_rad = float(np.max(np.abs(np.asarray(target) - np.asarray(start))))
    nominal_s = delta_rad / max(cfg.homing_max_speed_rad_per_s, 1e-6)
    return max(cfg.homing_target_timeout_s, 2.0 * nominal_s + cfg.homing_target_timeout_s)
