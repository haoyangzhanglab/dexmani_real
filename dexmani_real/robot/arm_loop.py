"""Arm servo loop — Mode 6 joint online trajectory planning for xArm7.

Primary entry point: ``arm_loop(shared)`` — mp.Process target using SharedStorage.
"""

from __future__ import annotations

import json
import logging
import time
from queue import Empty
from typing import Any, Callable

import numpy as np

from dexmani_real import ASSET_DIR
from dexmani_real.planning.kinematics import ArmFK
from dexmani_real.policy.safety import worker_validate_arm
from dexmani_real.robot.arm_sdk import (
    ArmLoopConfig,
    _read_live_error_code,
    _require_sdk_ok,
    describe_controller_error,
    enter_mode0,
    enter_mode6,
    read_live_state_and_error,
    stop_controller,
    validate_command_dynamics_intersection,
    validate_device_identity,
)
from dexmani_real.robot.homing import run_planned_homing
from dexmani_real.robot.safety import SafetyState
from dexmani_real.shm.shared_storage import (
    HOME_SENTINEL,
    HomeOutcome,
    HomeRequest,
    HomeResult,
    new_frame,
)
from dexmani_real.utils.log import (
    ThrottledWarner,
    capture_native_stdout,
    extract_native_diagnostics,
    get_logger,
)
from dexmani_real.utils.rate_manager import RateManager
from dexmani_real.utils.retry import RetryCounter
from dexmani_real.utils.schema import (
    ARM_COMMAND_DTYPE,
    ARM_JOINT_SHAPE,
    ARM_STATE_DTYPE,
)

logger = get_logger(__name__)


def _parse_arm_action_metadata(
    action: Any, received_s: float
) -> tuple[int, float, bool]:
    """Return ``(sequence, created_s, is_hold)`` for a fixed command frame."""
    if (
        isinstance(action, np.ndarray)
        and action.shape == (1,)
        and action.dtype == ARM_COMMAND_DTYPE
    ):
        command_seq = int(action["action_id"][0])
        created_s = int(action["created_monotonic_ns"][0]) / 1e9
        if not np.isfinite(created_s) or created_s <= 0.0 or created_s > received_s:
            created_s = received_s
        return command_seq, created_s, bool(action["is_hold"][0])
    return 0, received_s, False


def _take_next_current_arm_action(
    action_q: Any, *, expected_run_generation: int
) -> tuple[Any, float] | None:
    """Drain invalidated endpoints and return ``(item, received_s)``.

    ``received_s`` is sampled immediately after ``get()`` returns so queue
    latency is measured from the true dequeue time, not after validation.
    Generation checks can discard only endpoints still in this queue; an
    endpoint already accepted by the SDK remains owned by Mode 6 firmware.
    """
    while True:
        try:
            queued_action = action_q.get(timeout=0.0)
        except Empty:
            return None
        received_s = time.monotonic()
        now_ns = time.monotonic_ns()
        if isinstance(queued_action, np.ndarray):
            if worker_validate_arm(
                queued_action,
                expected_run_generation=expected_run_generation,
                now_monotonic_ns=now_ns,
            ):
                return queued_action, received_s
            logger.info(
                "arm_loop: discarded malformed, stale-generation, or expired command"
            )
            continue
        return queued_action, received_s


def _decode_joint_state_feedback(code: Any, states: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate position, velocity, and SDK effort at the worker boundary."""
    _require_sdk_ok("get_joint_states", code)
    if not isinstance(states, (list, tuple)) or len(states) < 3:
        raise RuntimeError("get_joint_states(num=3) must return position, velocity, and effort")
    qpos = np.asarray(states[0], dtype=np.float64)[: ARM_JOINT_SHAPE[0]].copy()
    qvel = np.asarray(states[1], dtype=np.float64)[: ARM_JOINT_SHAPE[0]].copy()
    tau = np.asarray(states[2], dtype=np.float64)[: ARM_JOINT_SHAPE[0]].copy()
    for name, value in (("qpos", qpos), ("qvel", qvel), ("tau", tau)):
        if value.shape != ARM_JOINT_SHAPE or not np.all(np.isfinite(value)):
            raise RuntimeError(
                f"get_joint_states returned invalid {name}: shape={value.shape}"
            )
    return qpos, qvel, tau


def _update_state_read_watchdog(counter: RetryCounter, *, succeeded: bool) -> bool:
    """Update the consecutive feedback-failure counter and report escalation."""
    if succeeded:
        counter.reset()
        return False
    counter.inc()
    return counter.triggered


def _compute_command_latency(
    *,
    created_s: float,
    received_s: float,
    applied_s: float,
    sdk_started_s: float,
) -> tuple[float, float, float]:
    """Return ``(queue_latency, apply_latency, sdk_duration)`` from the three
    monotonic samples, clamped to non-negative.

    ``received_s`` is the dequeue time and ``applied_s`` the successful SDK
    return time, so queue latency excludes the SDK call and apply latency
    includes it — they are distinct quantities, never conflated.
    """
    return (
        max(0.0, received_s - created_s),
        max(0.0, applied_s - created_s),
        max(0.0, applied_s - sdk_started_s),
    )


_MODE_DRIFT_TIMEOUT_S = 1.0  # bounded wall-clock window for a cached-mode mismatch


def _advance_mode_drift(
    *,
    monitoring: bool,
    report_mode: int,
    expected_mode: int,
    feedback_healthy: bool,
    mismatch_since_s: float | None,
    now_s: float,
    timeout_s: float,
) -> tuple[float | None, bool]:
    """Advance the cached-mode drift window; return ``(mismatch_since_s, fault)``.

    A repeated read of the same unchanged cached ``mode`` is one observation,
    not a per-tick count: the mismatch must persist past ``timeout_s`` of
    wall-clock while joint feedback stays healthy before ``fault`` becomes True.
    ``monitoring`` gates on STREAMING, so an expected Mode 0/6 transition and a
    single stale read are never treated as drift.
    """
    if not monitoring or report_mode == expected_mode:
        return None, False
    if mismatch_since_s is None:
        return now_s, False
    if feedback_healthy and now_s - mismatch_since_s >= timeout_s:
        return mismatch_since_s, True
    return mismatch_since_s, False


_CONTROLLER_ERROR_LABELS: dict[int, str] = {
    22: "self-collision",
    24: "speed limit exceeded",
    31: "collision current",
}


def latch_arm_fault(
    shared: Any,
    arm: Any,
    reason: str,
    *,
    api_code: int | None = None,
    controller_error: int | None = None,
    on_poll: Callable[[], None] | None = None,
) -> None:
    """Latch the single sticky arm fault and stop the controller.

    Writes the sticky ``error_state`` first — before the stop attempt — so a
    fault is never lost even if stopping raises.  ``api_code`` (SDK setter
    return) and ``controller_error`` (live controller register) are distinct
    namespaces and are logged separately.  Never calls ``clean_error`` and
    never writes ``safety_state`` (Main/policy owns it).
    """
    shared.error_state.value = True
    label = (
        _CONTROLLER_ERROR_LABELS.get(controller_error, "")
        if controller_error is not None
        else ""
    )
    logger.error(
        "arm_loop: latching fault: %s (api_code=%s, controller_error=%s%s, "
        "mode=%s, state=%s)",
        reason,
        api_code,
        controller_error,
        f" {label}" if label else "",
        getattr(arm, "mode", None),
        getattr(arm, "state", None),
    )
    if controller_error == 31 and hasattr(arm, "get_c31_error_info"):
        # C31 diagnostics (servo id / theoretical vs actual torque) are pure
        # log output, never a control branch.
        try:
            code, info = arm.get_c31_error_info()
        except Exception:
            logger.warning("arm_loop: failed to read C31 diagnostics", exc_info=True)
        else:
            if (
                code == 0
                and isinstance(info, (list, tuple, np.ndarray))
                and len(info) >= 3
            ):
                try:
                    servo_id = int(info[0])
                    theoretical_tau = float(info[1])
                    actual_tau = float(info[2])
                except (TypeError, ValueError, OverflowError):
                    logger.error("arm_loop: C31 diagnostics: %s", info)
                else:
                    logger.error(
                        "arm_loop: C31 diagnostics: servo_id=%d theoretical_tau=%.3f "
                        "actual_tau=%.3f delta_tau=%.3f",
                        servo_id,
                        theoretical_tau,
                        actual_tau,
                        actual_tau - theoretical_tau,
                    )
            else:
                logger.error("arm_loop: C31 diagnostics: %s", info)
    stop = stop_controller(arm, on_poll=on_poll)
    if not stop.confirmed:
        logger.error("arm_loop: fault stop not confirmed: %s", stop.reason)


def _request_still_current_and_armed(shared: Any, request: HomeRequest) -> bool:
    """True when the request's generation is current and safety is ARMED."""
    return (
        int(shared.run_generation.value) == request.run_generation
        and int(shared.safety_state.value) == int(SafetyState.ARMED)
        and not shared.error_state.value
    )


def _with_outcome(result: HomeResult, outcome: HomeOutcome, reason: str) -> HomeResult:
    """Rebuild ``result`` with a new terminal outcome and reason."""
    return HomeResult(
        request_id=result.request_id,
        outcome=outcome,
        reason=reason,
        final_qpos=np.asarray(result.final_qpos, dtype=np.float64).copy(),
        completed_at_s=time.monotonic(),
    )


def _finalize_home_result(
    shared: Any,
    arm: Any,
    request: HomeRequest,
    provisional: HomeResult,
    *,
    on_poll: Callable[[], None],
) -> tuple[HomeResult, bool]:
    """Apply the single controller finalizer for a finished HOME.

    The worker that owns ``arm_action_q`` is the only party that stops/restores
    the controller and publishes the terminal outcome.  Returns
    ``(terminal, accept_motion)``:

    * FAILED → sticky ``error_state`` + stop; never accept motion.
    * CANCELLED → stop; a failed stop confirmation upgrades to FAILED + sticky.
    * SUCCESS → re-check generation/ARMED, re-enter Mode 6; any failure
      upgrades to FAILED + sticky + state-4.

    Only a confirmed SUCCESS Mode-6 restore yields ``accept_motion=True``.
    """
    outcome = provisional.outcome

    if outcome is HomeOutcome.FAILED:
        shared.error_state.value = True
        stop_controller(arm, on_poll=on_poll)
        return provisional, False

    if outcome is HomeOutcome.CANCELLED:
        stop = stop_controller(arm, on_poll=on_poll)
        if not stop.confirmed:
            shared.error_state.value = True
            return (
                _with_outcome(
                    provisional,
                    HomeOutcome.FAILED,
                    f"{provisional.reason}; stop not confirmed: {stop.reason}",
                ),
                False,
            )
        return provisional, False

    if not _request_still_current_and_armed(shared, request):
        shared.error_state.value = True
        stop_controller(arm, on_poll=on_poll)
        return (
            _with_outcome(
                provisional,
                HomeOutcome.FAILED,
                "generation/safety boundary changed before Mode 6 restore",
            ),
            False,
        )

    try:
        enter_mode6(arm, on_poll=on_poll)
    except Exception as exc:
        shared.error_state.value = True
        stop_controller(arm, on_poll=on_poll)
        return (
            _with_outcome(
                provisional, HomeOutcome.FAILED, f"Mode 6 restore failed: {exc}"
            ),
            False,
        )
    return provisional, True


def arm_loop(shared, config: ArmLoopConfig | None = None) -> None:
    """Arm process entry point — reads arm_action_q, servos arm via Mode 6.

    mp.Process target communicating exclusively through SharedStorage.
    """
    _tracking_warn = ThrottledWarner(interval_s=5.0)
    _fk_warn = ThrottledWarner(interval_s=5.0)
    cfg = config or ArmLoopConfig()
    _state_error_counter = RetryCounter(
        max_consecutive=cfg.max_consecutive_arm_health_failures, label="arm_state"
    )
    _fk_error_counter = RetryCounter(
        max_consecutive=cfg.max_consecutive_arm_health_failures, label="arm_fk"
    )
    _tracking_err_count = 0
    logger.debug("arm_loop: LOADING")

    def _publish_startup_fault(detail: str) -> None:
        logger.error("arm_loop: %s", detail)

    def _heartbeat() -> None:
        shared.set_heartbeat("arm", time.monotonic())

    # URDF-consistent FK (replaces arm.get_position_aa). xArm firmware uses a
    # different EEF coordinate definition — Pinocchio FK ensures all consumers
    # share a single coordinate system.
    _urdf_path = str(ASSET_DIR / "robots" / "xhand" / "xarm7_xhand_collision.urdf")
    try:
        _arm_fk = ArmFK(_urdf_path)
    except Exception:
        logger.error("arm_loop: ArmFK initialization failed", exc_info=True)
        _publish_startup_fault("ArmFK initialization failed")
        shared.error_state.value = True
        return

    sdk_connect_output = None
    try:
        # The vendor SDK prints a multi-line banner directly to stdout and its
        # ``origin.print`` logger announces every ready/not-ready transition.
        # Our worker validates and reports those states itself, so suppress the
        # successful duplicate chatter while retaining SDK WARNING/ERROR logs.
        with capture_native_stdout() as sdk_connect_output:
            from xarm.wrapper import XArmAPI

            logging.getLogger("origin.print").setLevel(logging.WARNING)
            arm = XArmAPI(cfg.arm_ip, is_radian=True)
        sdk_diagnostics = extract_native_diagnostics(sdk_connect_output.text)
        if sdk_diagnostics:
            logger.warning(
                "xArm SDK initialization diagnostics:\n%s", "\n".join(sdk_diagnostics)
            )
    except Exception as e:
        vendor_detail = (
            sdk_connect_output.text if sdk_connect_output is not None else ""
        )
        logger.error(
            "arm_loop: connect failed: %s%s",
            e,
            f"; vendor output:\n{vendor_detail}" if vendor_detail else "",
        )
        _publish_startup_fault("SDK connect failed")
        shared.error_state.value = True
        return

    # Device identity: wait (bounded) for the SDK report thread to populate the
    # identity attributes, then sample + validate them once.  Sampling at
    # connection time would permanently record "unavailable" placeholders.
    _identity_deadline = time.monotonic() + 3.0
    while time.monotonic() < _identity_deadline:
        if isinstance(getattr(arm, "axis", None), int) and int(getattr(arm, "axis", 0) or 0) > 0:
            break
        _heartbeat()
        time.sleep(0.05)
    _identity_axis = int(getattr(arm, "axis", 0) or 0)
    _identity_type = str(getattr(arm, "device_type", "") or "")
    _identity_sn = str(getattr(arm, "sn", "") or "")
    _identity_firmware = tuple(getattr(arm, "version_number", ()) or ())

    _identity_error = validate_device_identity(
        axis=_identity_axis,
        device_type=_identity_type,
        serial_number=_identity_sn,
        firmware=_identity_firmware,
        expected_axis=cfg.expected_axis,
        expected_serial=cfg.serial_number,
        min_firmware=cfg.min_firmware,
        device_profile=cfg.device_profile,
    )
    if _identity_error is not None:
        logger.error("arm_loop: device identity validation failed: %s", _identity_error)
        _publish_startup_fault(_identity_error)
        shared.error_state.value = True
        stop_controller(arm, on_poll=_heartbeat)
        _disconnect_arm(arm)
        return

    if hasattr(shared, "arm_device_identity"):
        _identity_firmware_str = (
            ".".join(str(v) for v in _identity_firmware)
            if _identity_firmware
            else str(getattr(arm, "version", "unavailable") or "unavailable")
        )
        identity = {
            "axis": _identity_axis,
            "device_type": _identity_type or "unavailable",
            "model": cfg.device_profile or _identity_type or "unavailable",
            "serial_number": _identity_sn or "unavailable",
            "firmware_version": _identity_firmware_str,
        }
        encoded_identity = json.dumps(
            identity, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        shared.arm_device_identity.value = encoded_identity[:1023].ljust(1024, b"\x00")

    # Connect-time dynamics intersection: the configured Mode 6 speed/mvacc must
    # fit within the SDK hard clamps AND the reported device limits, or the
    # firmware would silently rewrite them.  Refuse before touching the motion
    # controller so metadata never claims an execution value it cannot honor.
    _dynamics_mismatch = validate_command_dynamics_intersection(
        config_speed_rad_per_s=cfg.joint_max_speed_rad_per_s,
        config_acc_rad_per_s2=cfg.joint_max_acc_rad_per_s2,
        device_speed_limits=getattr(arm, "joint_speed_limit", None),
        device_acc_limits=getattr(arm, "joint_acc_limit", None),
    )
    if _dynamics_mismatch is not None:
        logger.error("arm_loop: %s", _dynamics_mismatch)
        _publish_startup_fault(_dynamics_mismatch)
        shared.error_state.value = True
        stop_controller(arm, on_poll=_heartbeat)
        _disconnect_arm(arm)
        return

    # Connect/configure while the application remains DISARMED. Motion mode is
    # entered only on the ARMED edge below.  Startup never clears a pre-existing
    # controller error: read + record it first, and if non-zero, stop and exit
    # without publishing readiness.
    try:
        _live = read_live_state_and_error(arm)
    except Exception:
        logger.error("arm_loop: startup live error/warn read failed", exc_info=True)
        _publish_startup_fault("startup live error/warn read failed")
        shared.error_state.value = True
        _disconnect_arm(arm)
        return
    if _live.error_code != 0:
        logger.error(
            "arm_loop: startup controller error C%d (warn=%d, state=%d): %s — "
            "refusing to clear",
            _live.error_code,
            _live.warn_code,
            _live.state,
            describe_controller_error(_live.error_code),
        )
        _publish_startup_fault(
            f"startup controller error C{_live.error_code}: "
            f"{describe_controller_error(_live.error_code)}"
        )
        shared.error_state.value = True
        stop_controller(arm, on_poll=_heartbeat)
        _disconnect_arm(arm)
        return
    if _live.warn_code != 0:
        logger.warning(
            "arm_loop: startup controller warn=%d (diagnostic only)", _live.warn_code
        )

    # A single non-retried state-change pass: motion-enable then Mode 0/State 0.
    # ``enter_mode0`` verifies the controller reaches a movable, error-free
    # state.  State changes are never retried; only the bounded reads below
    # (initial joint state) may retry.
    try:
        _require_sdk_ok("startup motion_enable", arm.motion_enable(True))
        enter_mode0(arm, on_poll=_heartbeat)
    except Exception:
        logger.error(
            "arm_loop: startup motion/mode configuration failed", exc_info=True
        )
        _publish_startup_fault("controller configuration failed")
        shared.error_state.value = True
        _disconnect_arm(arm)
        return

    # Post-connect configuration. Every setter return code is authoritative.
    try:
        _require_sdk_ok(
            "set_collision_sensitivity",
            arm.set_collision_sensitivity(cfg.collision_sensitivity),
        )
        # TCP load (XHand) mass/COG from runtime config (see ArmParams).
        _require_sdk_ok(
            "set_tcp_load",
            arm.set_tcp_load(
                weight=cfg.tcp_load_mass_kg,
                center_of_gravity=list(cfg.tcp_load_cog_mm),
            ),
        )
        # Controller-global joint acceleration cap for Mode 6 trajectory
        # generation — not collision detection (that is set_collision_sensitivity
        # above).
        _require_sdk_ok(
            "set_joint_maxacc",
            arm.set_joint_maxacc(cfg.joint_max_acc_rad_per_s2, is_radian=True),
        )
    except Exception as e:
        logger.error("arm_loop: post-recovery config failed: %s", e)
        _publish_startup_fault("controller configuration failed")
        shared.error_state.value = True
        _disconnect_arm(arm)
        return

    # A single failed startup read is not enough to declare the controller dead.
    _STATE_READ_MAX_RETRIES = 10
    for _attempt in range(_STATE_READ_MAX_RETRIES):
        try:
            code, states = arm.get_joint_states(is_radian=True, num=3)
            last_qpos, initial_qvel, initial_effort = _decode_joint_state_feedback(
                code, states
            )
            break
        except Exception:
            logger.warning(
                "arm_loop: initial joint state read attempt %d/%d raised exception",
                _attempt + 1,
                _STATE_READ_MAX_RETRIES,
                exc_info=True,
            )
        time.sleep(0.1)
    else:
        logger.error(
            "arm_loop: cannot read initial joint states after %d attempts",
            _STATE_READ_MAX_RETRIES,
        )
        _publish_startup_fault("initial feedback unavailable")
        shared.error_state.value = True
        _disconnect_arm(arm)
        return
    last_target = last_qpos.copy()
    # Ready means connected/configured and physically STOPPED, not Mode-6
    # motion-enabled. Confirm state 4 before exposing arm_ready.
    _startup_stop = stop_controller(arm, on_poll=_heartbeat)
    if not _startup_stop.confirmed:
        logger.error(
            "arm_loop: failed to enter confirmed DISARMED state: %s",
            _startup_stop.reason,
        )
        _publish_startup_fault("confirmed state-4 startup stop failed")
        shared.error_state.value = True
        _disconnect_arm(arm)
        return
    last_cmd_seq = 0
    last_cmd_created_s = 0.0
    last_cmd_received_s = 0.0
    last_cmd_applied_s = 0.0
    last_cmd_queue_latency_s = 0.0
    last_cmd_apply_latency_s = 0.0
    last_cmd_sdk_duration_s = 0.0
    last_cmd_is_hold = False
    accepts_motion_commands = False
    last_safety_state = int(SafetyState.DISARMED)
    last_state_source_ns = time.monotonic_ns()
    terminal_feedback_detail: str | None = None
    _mode_mismatch_since_s: float | None = None

    # Publish initial state BEFORE arm_ready — consumers wait on arm_ready and
    # expect the ring to already contain a valid frame.  Without this, there is
    # a one-tick window where arm_ready is set but arm_state_ring is empty.
    try:
        eef_pos_init, eef_rot6d_init = _arm_fk.compute(last_qpos)
    except Exception:
        logger.error("arm_loop: initial ArmFK computation failed", exc_info=True)
        _publish_startup_fault("initial ArmFK computation failed")
        shared.error_state.value = True
        _disconnect_arm(arm)
        return
    _frame = new_frame(ARM_STATE_DTYPE)
    _frame["qpos"][0] = last_qpos
    _frame["qvel"][0] = initial_qvel
    # SDK current-estimated effort; precise SI unit is unverified.
    _frame["tau"][0] = initial_effort
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
    _frame["source_monotonic_ns"][0] = last_state_source_ns
    _frame["publish_monotonic_ns"][0] = time.monotonic_ns()
    _frame["state_valid"][0] = 1
    _frame["timestamp"][0] = last_state_source_ns / 1e9
    shared.arm_state_ring.write(_frame)

    # Write heartbeat BEFORE ready signal — prevents false FAULT on startup
    # (same pattern as vr_loop).  Main's supervisor checks heartbeats immediately
    # after all ready events.
    shared.set_heartbeat("arm", time.monotonic())
    shared.set_ready("arm")
    logger.debug("arm_loop: READY")
    logger.info(
        "arm_loop: ready and DISARMED (state=4, ip=%s, hz=%.0f)",
        cfg.arm_ip,
        cfg.arm_loop_hz,
    )

    limiter = RateManager(cfg.arm_loop_hz, label="arm")

    stopped_cleanly = False
    try:
        while shared.is_running.value:
            # Heartbeat continues even when no new endpoint is being consumed.
            shared.set_heartbeat("arm", time.monotonic())

            if shared.estop_request.value:
                # SDK emergency_stop() requests an immediate stop and waits for
                # controller State 4; it does not cut motor power.  Cleanup still
                # re-confirms State 4 (belt-and-suspenders per XArm7).
                try:
                    arm.emergency_stop()
                except Exception:
                    logger.warning(
                        "arm_loop: emergency_stop call failed; cleanup will enforce state 4",
                        exc_info=True,
                    )
                break

            # Safety-state/controller lifecycle edges. DISARMED/FAULT always map to
            # confirmed controller state 4; ARMED enters Mode 6 then state 0 and
            # confirms the live postcondition before accepting commands.
            # When gated (DISARMED or FAULT), skip action read + servo but continue
            # to publish state (for monitoring) and rate-limit normally.
            _safety = shared.safety_state.value
            if _safety in (SafetyState.DISARMED, SafetyState.FAULT):
                if accepts_motion_commands or last_safety_state not in (
                    SafetyState.DISARMED,
                    SafetyState.FAULT,
                ):
                    _safe_stop = stop_controller(arm, on_poll=_heartbeat)
                    if not _safe_stop.confirmed:
                        logger.error(
                            "arm_loop: failed to confirm safe stop: %s",
                            _safe_stop.reason,
                        )
                        shared.error_state.value = True
                        break
                    accepts_motion_commands = False
            elif (
                _safety in (SafetyState.ARMED, SafetyState.RUNNING)
                and not accepts_motion_commands
                and not shared.error_state.value
            ):
                try:
                    enter_mode6(arm, on_poll=_heartbeat)
                except Exception:
                    logger.error(
                        "arm_loop: failed ARMED Mode-6 postcondition", exc_info=True
                    )
                    shared.error_state.value = True
                    stop_controller(arm, on_poll=_heartbeat)
                    break
                accepts_motion_commands = True
            last_safety_state = int(_safety)
            if _safety in (SafetyState.ARMED, SafetyState.RUNNING) and not shared.error_state.value:
                _dequeued = _take_next_current_arm_action(
                    shared.arm_action_q,
                    expected_run_generation=int(shared.run_generation.value),
                )
                if _dequeued is None:
                    action = None
                    action_received_s = 0.0
                else:
                    action, action_received_s = _dequeued

                # HOME sentinel — collision-validated path with request ID.
                if (
                    isinstance(action, tuple)
                    and len(action) == 2
                    and action[0] == HOME_SENTINEL
                ):
                    _request = action[1]
                    if not isinstance(_request, HomeRequest):
                        logger.error("arm_loop: rejecting malformed HOME request")
                        continue
                    if int(shared.run_generation.value) != _request.run_generation:
                        # A superseded request never started: publish a clean
                        # cancellation and stay in the current mode (no stop).
                        logger.warning(
                            "arm_loop: discarding stale-generation HOME request"
                        )
                        _stale = HomeResult(
                            request_id=_request.request_id,
                            outcome=HomeOutcome.CANCELLED,
                            reason="run generation changed before homing started",
                            final_qpos=np.full(
                                ARM_JOINT_SHAPE, np.nan, dtype=np.float64
                            ),
                            completed_at_s=time.monotonic(),
                        )
                        try:
                            shared.arm_home_result_q.put(_stale, timeout=0.2)
                        except Exception:
                            logger.error(
                                "arm_loop: failed to publish stale HOME result",
                                exc_info=True,
                            )
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
                        nonlocal last_state_source_ns, terminal_feedback_detail
                        last_state_source_ns = time.monotonic_ns()
                        fk_valid = True
                        try:
                            eef_pos, eef_rot6d = _arm_fk.compute(qpos)
                        except Exception:
                            _fk_warn(
                                "arm_loop: Pinocchio FK failed during homing — publishing invalid EEF"
                            )
                            eef_pos = np.full(3, np.nan, dtype=np.float64)
                            eef_rot6d = np.full(6, np.nan, dtype=np.float64)
                            fk_valid = False
                            _fk_error_counter.inc()
                            if _fk_error_counter.triggered:
                                terminal_feedback_detail = "persistent ArmFK failure"
                                shared.error_state.value = True
                        else:
                            _fk_error_counter.reset()
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
                        _frame["source_monotonic_ns"][0] = last_state_source_ns
                        _frame["publish_monotonic_ns"][0] = time.monotonic_ns()
                        _frame["state_valid"][0] = int(fk_valid)
                        _frame["timestamp"][0] = last_state_source_ns / 1e9
                        shared.arm_state_ring.write(_frame)

                    accepts_motion_commands = False
                    _home_started_s = time.monotonic()
                    _provisional = run_planned_homing(
                        arm,
                        _request,
                        cfg,
                        shared=shared,
                        feedback_callback=_publish_homing_feedback,
                    )
                    _terminal, _accept = _finalize_home_result(
                        shared, arm, _request, _provisional, on_poll=_heartbeat
                    )
                    if _terminal.final_qpos.shape == ARM_JOINT_SHAPE and np.all(
                        np.isfinite(_terminal.final_qpos)
                    ):
                        last_qpos = _terminal.final_qpos.copy()
                        last_target = last_qpos.copy()
                    try:
                        shared.arm_home_result_q.put(_terminal, timeout=0.2)
                    except Exception:
                        logger.error(
                            "arm_loop: failed to publish HOME result", exc_info=True
                        )
                        shared.error_state.value = True
                        accepts_motion_commands = False
                        stop_controller(arm, on_poll=_heartbeat)
                        break
                    accepts_motion_commands = _accept
                    if _terminal.outcome is HomeOutcome.SUCCESS:
                        logger.info(
                            "arm_loop: HOME complete in %.2fs",
                            time.monotonic() - _home_started_s,
                        )
                    else:
                        logger.error(
                            "arm_loop: HOME %s — %s",
                            _terminal.outcome.name,
                            _terminal.reason,
                        )
                    continue

                elif action is not None and not isinstance(action, tuple):
                    if not worker_validate_arm(
                        action,
                        expected_run_generation=int(shared.run_generation.value),
                        now_monotonic_ns=time.monotonic_ns(),
                    ):
                        logger.info("arm_loop: discarded malformed, stale-generation, or expired command")
                    elif not accepts_motion_commands:
                        logger.warning("arm_loop: discarded endpoint while controller motion is disabled")
                    else:
                        target = np.asarray(action["qpos_cmd"][0], dtype=np.float64)
                        # 2π-canonicalized by the producer (IK-stage
                        # canonicalize_qpos at planning/ik.py:147 and :454 against
                        # arm_state_ring current_qpos); this worker no longer wraps.
                        last_target = target.copy()
                        _sdk_started_s = time.monotonic()
                        try:
                            code = arm.set_servo_angle(
                                angle=target,
                                is_radian=True,
                                speed=cfg.joint_max_speed_rad_per_s,
                                mvacc=cfg.joint_max_acc_rad_per_s2,
                                wait=False,
                            )
                        except Exception as exc:
                            latch_arm_fault(
                                shared,
                                arm,
                                f"set_servo_angle raised: {exc}",
                                on_poll=_heartbeat,
                            )
                            break
                        if code == 0:
                            _applied_s = time.monotonic()
                            last_cmd_seq, last_cmd_created_s, last_cmd_is_hold = (
                                _parse_arm_action_metadata(action, action_received_s)
                            )
                            last_cmd_received_s = action_received_s
                            last_cmd_applied_s = _applied_s
                            (
                                last_cmd_queue_latency_s,
                                last_cmd_apply_latency_s,
                                last_cmd_sdk_duration_s,
                            ) = _compute_command_latency(
                                created_s=last_cmd_created_s,
                                received_s=action_received_s,
                                applied_s=_applied_s,
                                sdk_started_s=_sdk_started_s,
                            )
                        else:
                            # A non-zero setter return is a terminal command
                            # failure even when the live controller error is
                            # 0 — record both namespaces, never continue.
                            try:
                                err_code = _read_live_error_code(arm)
                            except Exception:
                                latch_arm_fault(
                                    shared,
                                    arm,
                                    f"set_servo_angle failed (code={code}); "
                                    "live error read failed",
                                    api_code=code,
                                    on_poll=_heartbeat,
                                )
                                break
                            latch_arm_fault(
                                shared,
                                arm,
                                f"set_servo_angle failed (code={code})",
                                api_code=code,
                                controller_error=err_code,
                                on_poll=_heartbeat,
                            )
                            break
                elif action is not None:
                    logger.error(
                        "arm_loop: rejecting malformed action queue item %r", action
                    )

            arm_connected = True
            state_read_succeeded = False
            try:
                code, states = arm.get_joint_states(is_radian=True, num=3)
                qpos, qvel, tau = _decode_joint_state_feedback(code, states)
                last_state_source_ns = time.monotonic_ns()
                last_qpos = qpos.copy()
                state_read_succeeded = True
            except Exception:
                logger.warning("arm_loop: get_joint_states failed", exc_info=True)
                qpos, qvel, tau = (
                    last_qpos.copy(),
                    np.zeros(ARM_JOINT_SHAPE),
                    np.zeros(ARM_JOINT_SHAPE),
                )
                arm_connected = False
            state_read_fault = _update_state_read_watchdog(
                _state_error_counter,
                succeeded=state_read_succeeded,
            )

            # Pinocchio URDF-consistent FK (see note above). EEF is mandatory for
            # every consumer, so a failed derived calculation invalidates the whole
            # frame instead of fabricating a finite zero pose.
            fk_valid = True
            try:
                eef_pos, eef_rot6d = _arm_fk.compute(qpos)
            except Exception:
                _fk_warn("arm_loop: Pinocchio FK failed — publishing invalid EEF")
                eef_pos = np.full(3, np.nan, dtype=np.float64)
                eef_rot6d = np.full(6, np.nan, dtype=np.float64)
                fk_valid = False
                _fk_error_counter.inc()
            else:
                _fk_error_counter.reset()
            fk_fault = _fk_error_counter.triggered

            tracking_err = float(np.max(np.abs(qpos - last_target)))

            if tracking_err > cfg.tracking_error_warn_rad:
                _tracking_err_count += 1
                if _tracking_err_count >= 3:
                    _tracking_warn(
                        "arm_loop: tracking_err=%.3f_rad threshold=%.3f_rad",
                        tracking_err,
                        cfg.tracking_error_warn_rad,
                    )
            else:
                _tracking_err_count = 0

            # The cached error code drives a fault decision, so confirm it
            # against the live controller register before acting.
            try:
                error_code = arm.error_code
            except Exception:
                error_code = 0
                arm_connected = False
            if error_code != 0:
                try:
                    error_code = _read_live_error_code(arm)
                except Exception:
                    latch_arm_fault(
                        shared,
                        arm,
                        "live controller error read failed",
                        on_poll=_heartbeat,
                    )
                    break

            if error_code != 0:
                latch_arm_fault(
                    shared,
                    arm,
                    f"controller error C{error_code}",
                    controller_error=error_code,
                    on_poll=_heartbeat,
                )
                break

            # Mode-drift gate (STREAMING only).  The cached ``arm.mode`` has no
            # synchronous getter, so a persistent mismatch must outlive a short
            # bounded wall-clock window while joint feedback is still healthy —
            # a single stale read or an expected Mode 0/6 transition is not drift.
            _report_mode = int(getattr(arm, "mode", 6) or 6)
            _mode_mismatch_since_s, _mode_drifted = _advance_mode_drift(
                monitoring=accepts_motion_commands,
                report_mode=_report_mode,
                expected_mode=6,
                feedback_healthy=arm_connected,
                mismatch_since_s=_mode_mismatch_since_s,
                now_s=time.monotonic(),
                timeout_s=_MODE_DRIFT_TIMEOUT_S,
            )
            if _mode_drifted:
                latch_arm_fault(
                    shared,
                    arm,
                    f"controller mode drift (cached mode={_report_mode}, expected 6)",
                    on_poll=_heartbeat,
                )
                break

            # Publish state
            _frame["qpos"][0] = qpos
            _frame["qvel"][0] = qvel
            _frame["tau"][0] = tau
            _frame["eef_pos"][0] = eef_pos
            _frame["eef_rot6d"][0] = eef_rot6d
            _frame["error_code"][0] = int(error_code)
            _frame["connected"][0] = 1 if arm_connected else 0
            _frame["mode"][0] = _report_mode
            _frame["tracking_err"][0] = tracking_err
            _frame["last_cmd_seq"][0] = last_cmd_seq
            _frame["last_cmd_created_s"][0] = last_cmd_created_s
            _frame["last_cmd_received_s"][0] = last_cmd_received_s
            _frame["last_cmd_applied_s"][0] = last_cmd_applied_s
            _frame["last_cmd_queue_latency_s"][0] = last_cmd_queue_latency_s
            _frame["last_cmd_apply_latency_s"][0] = last_cmd_apply_latency_s
            _frame["last_cmd_sdk_duration_s"][0] = last_cmd_sdk_duration_s
            _frame["last_cmd_is_hold"][0] = int(last_cmd_is_hold)
            _frame["source_monotonic_ns"][0] = last_state_source_ns
            _frame["publish_monotonic_ns"][0] = time.monotonic_ns()
            _frame["state_valid"][0] = int(arm_connected and fk_valid)
            _frame["timestamp"][0] = last_state_source_ns / 1e9
            shared.arm_state_ring.write(_frame)

            if state_read_fault:
                terminal_feedback_detail = "persistent get_joint_states failure"
                shared.error_state.value = True
                logger.error(
                    "arm_loop: %d consecutive feedback-read failures — latching global fault",
                    _state_error_counter.count,
                )
                break
            if fk_fault:
                terminal_feedback_detail = "persistent ArmFK failure"
                shared.error_state.value = True
                logger.error(
                    "arm_loop: %d consecutive FK failures — latching global fault",
                    _fk_error_counter.count,
                )
                break

            # Rate limit
            limiter.wait()
    finally:
        # Cleanup: best-effort state 4 + disconnect, even if the loop raised.
        # A fault exit leaves a non-zero controller error; confirm the stop
        # without requiring the error to be cleared (it is already latched).
        _cleanup_stop = stop_controller(arm, on_poll=_heartbeat)
        if _cleanup_stop.confirmed:
            stopped_cleanly = True
        else:
            logger.warning(
                "arm_loop: cleanup failed: %s", _cleanup_stop.reason
            )
            shared.error_state.value = True
        _disconnect_arm(arm)
        if terminal_feedback_detail is not None:
            logger.error("arm_loop: %s", terminal_feedback_detail)
        elif stopped_cleanly:
            logger.debug("arm_loop: STOPPED")
        else:
            logger.error("arm_loop: state-4 cleanup failed")
        logger.info("arm_loop: exited")


def _disconnect_arm(arm: Any) -> None:
    """Disconnect arm and report best-effort cleanup failures."""
    try:
        arm.disconnect()
    except Exception:
        logger.warning("arm disconnect failed during cleanup", exc_info=True)
