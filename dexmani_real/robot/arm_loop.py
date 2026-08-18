"""Arm servo loop — Mode 6 joint online trajectory planning for xArm7.

``arm_loop(shared)`` is the mp.Process entry point (SharedStorage only). It calls
``_startup`` to connect, configure, and reach a confirmed-stop READY state, then
runs a fixed-rate loop where each iteration is ``_step``: ``_transition_safety`` →
``_consume_command`` (HOME or servo) → ``_observe_and_publish``, all sharing a
single ``_LoopState``.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import partial
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
    issue_mode_enter,
    mode_enter_ready,
    read_live_state_and_error,
    stop_controller,
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
    """Drain invalidated endpoints; ``received_s`` is sampled right after ``get()``.

    Generation checks discard only queued endpoints — one already accepted by
    the SDK belongs to Mode 6 firmware.
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
    if any(v.shape != ARM_JOINT_SHAPE or not np.all(np.isfinite(v)) for v in (qpos, qvel, tau)):
        raise RuntimeError("get_joint_states returned invalid qpos/qvel/tau shape or non-finite values")
    return qpos, qvel, tau


_MODE_DRIFT_TIMEOUT_S = 1.0  # bounded wall-clock window for a cached-mode mismatch
_MODE_ENTER_TIMEOUT_S = 1.0  # deadline for the non-blocking Mode-6 confirm probe


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

    The mismatch must persist past ``timeout_s`` while joint feedback stays
    healthy before ``fault`` becomes True; ``monitoring`` gates on STREAMING so
    an expected Mode 0/6 transition or a single stale read is never drift.
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
    """Latch the sticky arm fault and stop the controller.

    Writes ``error_state`` before the stop attempt so a fault is never lost if
    stopping raises.  ``api_code`` (SDK return) and ``controller_error`` (live
    register) are logged separately; never writes ``safety_state`` (Main owns it).
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
    """Finalize a finished HOME; return ``(terminal, accept_motion)``.

    FAILED → sticky + stop; CANCELLED → stop (unconfirmed stop upgrades to
    FAILED); SUCCESS → re-check generation/ARMED, re-enter Mode 6.  Only a
    confirmed SUCCESS Mode-6 restore yields ``accept_motion=True``.
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


@dataclass(frozen=True)
class _CmdState:
    """Metadata of the last command accepted by the SDK."""

    seq: int
    created_s: float
    received_s: float
    applied_s: float
    queue_latency_s: float
    apply_latency_s: float
    sdk_duration_s: float
    is_hold: bool

    @classmethod
    def idle(cls) -> "_CmdState":
        return cls(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False)


def _write_arm_frame(
    shared: Any,
    frame: Any,
    *,
    qpos: np.ndarray,
    qvel: np.ndarray,
    tau: np.ndarray,
    eef_pos: np.ndarray,
    eef_rot6d: np.ndarray,
    error_code: int,
    connected: bool,
    mode: int,
    accepts_motion_commands: bool,
    tracking_err: float,
    cmd: _CmdState,
    source_ns: int,
    state_valid: bool,
) -> None:
    """Publish one fully-populated ARM_STATE frame to ``arm_state_ring``."""
    frame["qpos"][0] = qpos
    frame["qvel"][0] = qvel
    frame["tau"][0] = tau
    frame["eef_pos"][0] = eef_pos
    frame["eef_rot6d"][0] = eef_rot6d
    frame["error_code"][0] = int(error_code)
    frame["connected"][0] = 1 if connected else 0
    frame["mode"][0] = mode
    frame["accepts_motion_commands"][0] = 1 if accepts_motion_commands else 0
    frame["tracking_err"][0] = tracking_err
    frame["last_cmd_seq"][0] = cmd.seq
    frame["last_cmd_created_s"][0] = cmd.created_s
    frame["last_cmd_received_s"][0] = cmd.received_s
    frame["last_cmd_applied_s"][0] = cmd.applied_s
    frame["last_cmd_queue_latency_s"][0] = cmd.queue_latency_s
    frame["last_cmd_apply_latency_s"][0] = cmd.apply_latency_s
    frame["last_cmd_sdk_duration_s"][0] = cmd.sdk_duration_s
    frame["last_cmd_is_hold"][0] = int(cmd.is_hold)
    frame["source_monotonic_ns"][0] = source_ns
    frame["publish_monotonic_ns"][0] = time.monotonic_ns()
    frame["state_valid"][0] = int(state_valid)
    frame["timestamp"][0] = source_ns / 1e9
    shared.arm_state_ring.write(frame)


class _Flow(Enum):
    """Per-iteration control flow: fall through, skip to next, or exit the loop."""

    PROCEED = auto()
    NEXT = auto()
    EXIT = auto()


@dataclass
class _ModeTransition:
    """In-flight non-blocking Mode-6 entry (issue + per-tick confirm)."""

    deadline_s: float


@dataclass
class _LoopState:
    """Loop-carried state shared by the per-iteration block functions."""

    cfg: ArmLoopConfig
    arm: Any
    fk: ArmFK
    frame: Any
    last_qpos: np.ndarray
    last_target: np.ndarray
    state_err_counter: RetryCounter
    fk_err_counter: RetryCounter
    tracking_warn: ThrottledWarner
    fk_warn: ThrottledWarner
    last_cmd: _CmdState = field(default_factory=_CmdState.idle)
    accepts_motion_commands: bool = False
    mode_transition: _ModeTransition | None = None
    last_safety_state: int = field(default_factory=lambda: int(SafetyState.DISARMED))
    last_state_source_ns: int = field(default_factory=time.monotonic_ns)
    mode_mismatch_since_s: float | None = None
    terminal_feedback_detail: str | None = None
    tracking_err_count: int = 0


def _make_fk() -> ArmFK:
    """Build the URDF-consistent FK (replaces ``arm.get_position_aa``)."""
    urdf_path = str(ASSET_DIR / "robots" / "xhand" / "xarm7_xhand_collision.urdf")
    return ArmFK(urdf_path)


def _connect_arm(cfg: ArmLoopConfig) -> Any:
    """Connect to the controller, suppressing the SDK's stdout banner.

    The vendor SDK prints a banner and its ``origin.print`` logger announces
    every ready/not-ready transition; this worker validates and reports those
    states itself, so the duplicate chatter is suppressed while WARNING/ERROR
    logs are retained.
    """
    sdk_connect_output = None
    try:
        with capture_native_stdout() as sdk_connect_output:
            from xarm.wrapper import XArmAPI

            logging.getLogger("origin.print").setLevel(logging.WARNING)
            arm = XArmAPI(cfg.arm_ip, is_radian=True)
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
    return arm


def _validate_identity(shared: Any, arm: Any, cfg: ArmLoopConfig) -> None:
    """Wait for the report thread's axis count, validate it, and publish identity."""
    heartbeat = lambda: shared.set_heartbeat("arm", time.monotonic())
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if isinstance(getattr(arm, "axis", None), int) and int(getattr(arm, "axis", 0) or 0) > 0:
            break
        heartbeat()
        time.sleep(0.05)
    axis = int(getattr(arm, "axis", 0) or 0)
    if axis != cfg.expected_axis:
        raise RuntimeError(f"device reports {axis} axes, expected {cfg.expected_axis}")
    if not hasattr(shared, "arm_device_identity"):
        return
    device_type = str(getattr(arm, "device_type", "") or "")
    sn = str(getattr(arm, "sn", "") or "")
    firmware = tuple(getattr(arm, "version_number", ()) or ())
    firmware_str = (
        ".".join(str(v) for v in firmware)
        if firmware
        else str(getattr(arm, "version", "unavailable") or "unavailable")
    )
    identity = {
        "axis": axis,
        "device_type": device_type or "unavailable",
        "model": cfg.device_profile or device_type or "unavailable",
        "serial_number": sn or "unavailable",
        "firmware_version": firmware_str,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    shared.arm_device_identity.value = encoded[:1023].ljust(1024, b"\x00")


def _check_startup_error(arm: Any) -> None:
    """Refuse to proceed on a pre-existing controller error; warn is diagnostic."""
    live = read_live_state_and_error(arm)  # raises on SDK failure
    if live.error_code != 0:
        raise RuntimeError(
            f"startup controller error C{live.error_code} (warn={live.warn_code}, "
            f"state={live.state}): {describe_controller_error(live.error_code)} — "
            "refusing to clear"
        )
    if live.warn_code != 0:
        logger.warning(
            "arm_loop: startup controller warn=%d (diagnostic only)", live.warn_code
        )


def _enable_and_mode0(arm: Any, heartbeat: Callable[[], None]) -> None:
    """Enable motion and enter Mode 0 (movable, error-free state)."""
    _require_sdk_ok("startup motion_enable", arm.motion_enable(True))
    enter_mode0(arm, on_poll=heartbeat)


def _apply_config(arm: Any, cfg: ArmLoopConfig) -> None:
    """Apply collision sensitivity, TCP load, and the Mode 6 joint acc cap."""
    _require_sdk_ok(
        "set_collision_sensitivity",
        arm.set_collision_sensitivity(cfg.collision_sensitivity),
    )
    _require_sdk_ok(
        "set_tcp_load",
        arm.set_tcp_load(
            weight=cfg.tcp_load_mass_kg,
            center_of_gravity=list(cfg.tcp_load_cog_mm),
        ),
    )
    _require_sdk_ok(
        "set_joint_maxacc",
        arm.set_joint_maxacc(cfg.joint_max_acc_rad_per_s2, is_radian=True),
    )


def _read_initial_state(arm: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the initial joint states with bounded retries; raise on exhaustion."""
    max_retries = 10
    for attempt in range(max_retries):
        try:
            code, states = arm.get_joint_states(is_radian=True, num=3)
            return _decode_joint_state_feedback(code, states)
        except Exception:
            logger.warning(
                "arm_loop: initial joint state read attempt %d/%d raised exception",
                attempt + 1,
                max_retries,
                exc_info=True,
            )
        time.sleep(0.1)
    raise RuntimeError(f"cannot read initial joint states after {max_retries} attempts")


def _confirm_stopped(arm: Any, heartbeat: Callable[[], None]) -> None:
    """Confirm a physically stopped state (4); raise if not confirmed."""
    stop = stop_controller(arm, on_poll=heartbeat)
    if not stop.confirmed:
        raise RuntimeError(f"confirmed state-4 startup stop failed: {stop.reason}")


def _startup(shared: Any, cfg: ArmLoopConfig) -> _LoopState | None:
    """Connect, configure, and reach a confirmed-stop READY state.

    Returns a fully-initialized ``_LoopState`` (initial frame already published,
    ``arm`` ready-signal set) or ``None`` with ``error_state`` latched on any
    failure.  A failure after connect always does a best-effort state-4 stop
    before disconnect.
    """
    heartbeat = lambda: shared.set_heartbeat("arm", time.monotonic())
    logger.debug("arm_loop: LOADING")
    arm = None
    try:
        fk = _make_fk()
        arm = _connect_arm(cfg)
        _validate_identity(shared, arm, cfg)
        _check_startup_error(arm)
        _enable_and_mode0(arm, heartbeat)
        _apply_config(arm, cfg)
        qpos, qvel, tau = _read_initial_state(arm)
        _confirm_stopped(arm, heartbeat)
        eef_pos, eef_rot = fk.compute(qpos)
        st = _LoopState(
            cfg=cfg,
            arm=arm,
            fk=fk,
            frame=new_frame(ARM_STATE_DTYPE),
            last_qpos=qpos.copy(),
            last_target=qpos.copy(),
            state_err_counter=RetryCounter(
                cfg.max_consecutive_arm_health_failures, label="arm_state"
            ),
            fk_err_counter=RetryCounter(
                cfg.max_consecutive_arm_health_failures, label="arm_fk"
            ),
            tracking_warn=ThrottledWarner(interval_s=5.0),
            fk_warn=ThrottledWarner(interval_s=5.0),
        )
        # Publish the initial frame before the ready signal so consumers never
        # see an empty ring.
        _write_arm_frame(
            shared,
            st.frame,
            qpos=qpos,
            qvel=qvel,
            tau=tau,
            eef_pos=eef_pos,
            eef_rot6d=eef_rot,
            error_code=0,
            connected=True,
            mode=getattr(arm, "mode", 6),
            accepts_motion_commands=False,
            tracking_err=0.0,
            cmd=st.last_cmd,
            source_ns=st.last_state_source_ns,
            state_valid=True,
        )
        shared.set_heartbeat("arm", time.monotonic())  # heartbeat before ready
        shared.set_ready("arm")
        logger.debug("arm_loop: READY")
        logger.info(
            "arm_loop: ready and DISARMED (state=4, ip=%s, hz=%.0f)",
            cfg.arm_ip,
            cfg.arm_loop_hz,
        )
        return st
    except Exception as exc:
        logger.exception("arm_loop: startup failed: %s", exc)
        shared.error_state.value = True
        if arm is not None:
            stop_controller(arm, on_poll=heartbeat)
            _disconnect_arm(arm)
        return None


def _transition_safety(st: _LoopState, shared: Any, safety: Any) -> _Flow:
    """Stop on DISARMED/FAULT or issue a non-blocking Mode-6 entry on ARMED/RUNNING.

    The Mode-6 entry is asynchronous: ``issue_mode_enter`` sends the two
    ms-scale setters once, then ``_observe_and_publish`` confirms the movable
    postcondition on later ticks via ``mode_enter_ready``.  This keeps the loop
    publishing a truthful ``accepts_motion_commands=0`` frame every tick for the
    whole window instead of blocking ~100ms-1s and draining nothing.
    """
    heartbeat = lambda: shared.set_heartbeat("arm", time.monotonic())
    if safety in (SafetyState.DISARMED, SafetyState.FAULT):
        if st.accepts_motion_commands or st.last_safety_state not in (
            SafetyState.DISARMED,
            SafetyState.FAULT,
        ):
            safe_stop = stop_controller(st.arm, on_poll=heartbeat)
            if not safe_stop.confirmed:
                logger.error(
                    "arm_loop: failed to confirm safe stop: %s", safe_stop.reason
                )
                shared.error_state.value = True
                return _Flow.EXIT
            st.accepts_motion_commands = False
        st.mode_transition = None
    elif (
        safety in (SafetyState.ARMED, SafetyState.RUNNING)
        and not st.accepts_motion_commands
        and not shared.error_state.value
    ):
        if st.mode_transition is None:
            try:
                issue_mode_enter(st.arm, 6)
            except Exception:
                logger.error("arm_loop: failed ARMED Mode-6 issue", exc_info=True)
                shared.error_state.value = True
                stop_controller(st.arm, on_poll=heartbeat)
                return _Flow.EXIT
            st.mode_transition = _ModeTransition(
                deadline_s=time.monotonic() + _MODE_ENTER_TIMEOUT_S
            )
    st.last_safety_state = int(safety)
    return _Flow.PROCEED


def _publish_homing_feedback(
    st: _LoopState,
    shared: Any,
    qpos: np.ndarray,
    qvel: np.ndarray,
    tau: np.ndarray,
    target: np.ndarray,
) -> None:
    """Publish a homing-milestone frame; an FK failure invalidates the EEF."""
    st.last_state_source_ns = time.monotonic_ns()
    fk_valid = True
    try:
        eef_pos, eef_rot6d = st.fk.compute(qpos)
    except Exception:
        st.fk_warn("arm_loop: Pinocchio FK failed during homing — publishing invalid EEF")
        eef_pos = np.full(3, np.nan, dtype=np.float64)
        eef_rot6d = np.full(6, np.nan, dtype=np.float64)
        fk_valid = False
        st.fk_err_counter.inc()
        if st.fk_err_counter.triggered:
            st.terminal_feedback_detail = "persistent ArmFK failure"
            shared.error_state.value = True
    else:
        st.fk_err_counter.reset()
    try:
        error_code = int(getattr(st.arm, "error_code", 0) or 0)
    except Exception:
        error_code = 0
    _write_arm_frame(
        shared,
        st.frame,
        qpos=qpos,
        qvel=qvel,
        tau=tau,
        eef_pos=eef_pos,
        eef_rot6d=eef_rot6d,
        error_code=error_code,
        connected=True,
        mode=getattr(st.arm, "mode", 6),
        accepts_motion_commands=st.accepts_motion_commands,
        tracking_err=float(np.max(np.abs(qpos - target))),
        cmd=st.last_cmd,
        source_ns=st.last_state_source_ns,
        state_valid=bool(fk_valid),
    )


def _handle_home(st: _LoopState, shared: Any, request: Any) -> _Flow:
    """Run planned homing for a HOME request and publish the terminal result."""
    heartbeat = lambda: shared.set_heartbeat("arm", time.monotonic())
    if not isinstance(request, HomeRequest):
        logger.error("arm_loop: rejecting malformed HOME request")
        return _Flow.NEXT
    if int(shared.run_generation.value) != request.run_generation:
        # A superseded request never started: publish a clean cancellation and
        # stay in the current mode (no stop).
        logger.warning("arm_loop: discarding stale-generation HOME request")
        stale = HomeResult(
            request_id=request.request_id,
            outcome=HomeOutcome.CANCELLED,
            reason="run generation changed before homing started",
            final_qpos=np.full(ARM_JOINT_SHAPE, np.nan, dtype=np.float64),
            completed_at_s=time.monotonic(),
        )
        try:
            shared.arm_home_result_q.put(stale, timeout=0.2)
        except Exception:
            logger.error(
                "arm_loop: failed to publish stale HOME result", exc_info=True
            )
        return _Flow.NEXT
    logger.info(
        "arm_loop: HOME sentinel — planned homing (%d validated milestones)",
        len(request.waypoints),
    )
    st.accepts_motion_commands = False
    st.mode_transition = None  # homing runs in Mode 0; cancel any in-flight Mode-6 entry
    home_started_s = time.monotonic()
    provisional = run_planned_homing(
        st.arm,
        request,
        st.cfg,
        shared=shared,
        feedback_callback=partial(_publish_homing_feedback, st, shared),
    )
    terminal, accept = _finalize_home_result(
        shared, st.arm, request, provisional, on_poll=heartbeat
    )
    st.accepts_motion_commands = accept
    if terminal.final_qpos.shape == ARM_JOINT_SHAPE and np.all(
        np.isfinite(terminal.final_qpos)
    ):
        st.last_qpos = terminal.final_qpos.copy()
        st.last_target = st.last_qpos.copy()

    # A confirmed Mode-6 restore must be visible in the ring *before* the
    # terminal result unblocks the producer.  ``enter_mode6`` (inside
    # ``_finalize_home_result``) restores Mode 6, but the last published frame is
    # still the homing milestone (mode=0) and this tick skips the normal
    # observe-and-publish.  If a fire-and-forget producer publishes on the "home
    # complete" signal before the next tick refreshes the ring, the shared
    # snapshot's ``mode != 6`` gate mis-rejects it.  Refresh the frame now so the
    # completion signal and the Mode-6 readiness change are atomic to producers.
    observe_flow: _Flow | None = None
    if terminal.outcome is HomeOutcome.SUCCESS:
        observe_flow = _observe_and_publish(st, shared)
    try:
        shared.arm_home_result_q.put(terminal, timeout=0.2)
    except Exception:
        logger.error("arm_loop: failed to publish HOME result", exc_info=True)
        shared.error_state.value = True
        st.accepts_motion_commands = False
        stop_controller(st.arm, on_poll=heartbeat)
        return _Flow.EXIT
    if observe_flow is _Flow.EXIT:
        return observe_flow
    if terminal.outcome is HomeOutcome.SUCCESS:
        logger.info(
            "arm_loop: HOME complete in %.2fs",
            time.monotonic() - home_started_s,
        )
    else:
        logger.error(
            "arm_loop: HOME %s — %s", terminal.outcome.name, terminal.reason
        )
    return _Flow.NEXT


def _handle_servo_command(
    st: _LoopState, shared: Any, action: Any, action_received_s: float
) -> _Flow:
    """Validate and servo a normal endpoint command; latch a fault on failure."""
    heartbeat = lambda: shared.set_heartbeat("arm", time.monotonic())
    if not worker_validate_arm(
        action,
        expected_run_generation=int(shared.run_generation.value),
        now_monotonic_ns=time.monotonic_ns(),
    ):
        logger.info("arm_loop: discarded malformed, stale-generation, or expired command")
        return _Flow.PROCEED
    if not st.accepts_motion_commands:
        logger.warning("arm_loop: discarded endpoint while controller motion is disabled")
        return _Flow.PROCEED
    target = np.asarray(action["qpos_cmd"][0], dtype=np.float64)
    st.last_target = target.copy()  # target is 2π-canonicalized by the producer; no wrap
    sdk_started_s = time.monotonic()
    try:
        code = st.arm.set_servo_angle(
            angle=target,
            is_radian=True,
            speed=st.cfg.joint_max_speed_rad_per_s,
            mvacc=st.cfg.joint_max_acc_rad_per_s2,
            wait=False,
        )
    except Exception as exc:
        latch_arm_fault(
            shared, st.arm, f"set_servo_angle raised: {exc}", on_poll=heartbeat
        )
        return _Flow.EXIT
    if code == 0:
        applied_s = time.monotonic()
        seq, created_s, is_hold = _parse_arm_action_metadata(action, action_received_s)
        st.last_cmd = _CmdState(
            seq,
            created_s,
            action_received_s,
            applied_s,
            max(0.0, action_received_s - created_s),
            max(0.0, applied_s - created_s),
            max(0.0, applied_s - sdk_started_s),
            is_hold,
        )
        return _Flow.PROCEED
    # Non-zero setter return is terminal even if the cached error is 0.
    try:
        err_code = _read_live_error_code(st.arm)
    except Exception:
        latch_arm_fault(
            shared,
            st.arm,
            f"set_servo_angle failed (code={code}); live error read failed",
            api_code=code,
            on_poll=heartbeat,
        )
        return _Flow.EXIT
    latch_arm_fault(
        shared,
        st.arm,
        f"set_servo_angle failed (code={code})",
        api_code=code,
        controller_error=err_code,
        on_poll=heartbeat,
    )
    return _Flow.EXIT


def _consume_command(st: _LoopState, shared: Any, safety: Any) -> _Flow:
    """Dequeue and dispatch one command (HOME sentinel or a servo endpoint)."""
    if safety not in (SafetyState.ARMED, SafetyState.RUNNING) or shared.error_state.value:
        return _Flow.PROCEED
    dequeued = _take_next_current_arm_action(
        shared.arm_action_q,
        expected_run_generation=int(shared.run_generation.value),
    )
    if dequeued is None:
        return _Flow.PROCEED
    action, action_received_s = dequeued
    if isinstance(action, tuple) and len(action) == 2 and action[0] == HOME_SENTINEL:
        return _handle_home(st, shared, action[1])
    if action is not None and not isinstance(action, tuple):
        return _handle_servo_command(st, shared, action, action_received_s)
    if action is not None:
        logger.error("arm_loop: rejecting malformed action queue item %r", action)
    return _Flow.PROCEED


def _observe_and_publish(st: _LoopState, shared: Any) -> _Flow:
    """Read state + FK, confirm errors, gate mode drift, publish, check watchdogs."""
    heartbeat = lambda: shared.set_heartbeat("arm", time.monotonic())

    # Confirm an in-flight non-blocking Mode-6 entry before the normal observe.
    # While transitioning, ``accepts_motion_commands`` stays False so producers
    # gate on it; a transient mid-transition controller error is the probe's to
    # interpret ("not ready"), not a fault, so the fault-path checks below are
    # skipped for this tick.
    transitioning = st.mode_transition is not None
    if transitioning:
        if mode_enter_ready(st.arm, 6):
            st.accepts_motion_commands = True
            st.mode_transition = None
            transitioning = False
        elif time.monotonic() >= st.mode_transition.deadline_s:
            try:
                _transition_error = _read_live_error_code(st.arm)
            except Exception:
                _transition_error = None
            latch_arm_fault(
                shared,
                st.arm,
                f"Mode 6 postcondition not confirmed within {_MODE_ENTER_TIMEOUT_S:.1f}s",
                controller_error=_transition_error,
                on_poll=heartbeat,
            )
            return _Flow.EXIT

    arm_connected = True
    state_read_succeeded = False
    try:
        code, states = st.arm.get_joint_states(is_radian=True, num=3)
        qpos, qvel, tau = _decode_joint_state_feedback(code, states)
        st.last_state_source_ns = time.monotonic_ns()
        st.last_qpos = qpos.copy()
        state_read_succeeded = True
    except Exception:
        logger.warning("arm_loop: get_joint_states failed", exc_info=True)
        qpos, qvel, tau = (
            st.last_qpos.copy(),
            np.zeros(ARM_JOINT_SHAPE),
            np.zeros(ARM_JOINT_SHAPE),
        )
        arm_connected = False
    if state_read_succeeded:
        st.state_err_counter.reset()
    else:
        st.state_err_counter.inc()
    state_read_fault = st.state_err_counter.triggered

    # FK is mandatory for every consumer: a failure invalidates the whole frame
    # (NaN EEF) instead of fabricating a zero pose.
    fk_valid = True
    try:
        eef_pos, eef_rot6d = st.fk.compute(qpos)
    except Exception:
        st.fk_warn("arm_loop: Pinocchio FK failed — publishing invalid EEF")
        eef_pos = np.full(3, np.nan, dtype=np.float64)
        eef_rot6d = np.full(6, np.nan, dtype=np.float64)
        fk_valid = False
        st.fk_err_counter.inc()
    else:
        st.fk_err_counter.reset()
    fk_fault = st.fk_err_counter.triggered

    tracking_err = float(np.max(np.abs(qpos - st.last_target)))
    if tracking_err > st.cfg.tracking_error_warn_rad:
        st.tracking_err_count += 1
        if st.tracking_err_count >= 3:
            st.tracking_warn(
                "arm_loop: tracking_err=%.3f_rad threshold=%.3f_rad",
                tracking_err,
                st.cfg.tracking_error_warn_rad,
            )
    else:
        st.tracking_err_count = 0

    # Confirm the cached error against the live register before faulting.
    try:
        error_code = st.arm.error_code
    except Exception:
        error_code = 0
        arm_connected = False
    if error_code != 0 and not transitioning:
        try:
            error_code = _read_live_error_code(st.arm)
        except Exception:
            latch_arm_fault(
                shared, st.arm, "live controller error read failed", on_poll=heartbeat
            )
            return _Flow.EXIT
    if error_code != 0 and not transitioning:
        latch_arm_fault(
            shared,
            st.arm,
            f"controller error C{error_code}",
            controller_error=error_code,
            on_poll=heartbeat,
        )
        return _Flow.EXIT
    if transitioning:
        # A transient mid-transition error is the confirm probe's to interpret;
        # mask it from the published frame so consumers (teleop home admission,
        # health checks) don't misread a transition-settle error as a fault.
        error_code = 0

    # Mode-drift gate (STREAMING only): the cached mode has no sync getter, so
    # a mismatch must persist while feedback stays healthy before faulting.
    # The published mode must stay truthful (0 in Mode 0, 6 in Mode 6) so
    # consumers can gate on "arm is actually in servo Mode 6".  Do NOT coerce a
    # falsy Mode 0 up to 6: that would make the ring lie about the transition
    # and hide a genuine drift to Mode 0 from this gate.
    _cached_mode = getattr(st.arm, "mode", None)
    report_mode = int(_cached_mode) if _cached_mode is not None else 0
    st.mode_mismatch_since_s, mode_drifted = _advance_mode_drift(
        monitoring=st.accepts_motion_commands,
        report_mode=report_mode,
        expected_mode=6,
        feedback_healthy=arm_connected,
        mismatch_since_s=st.mode_mismatch_since_s,
        now_s=time.monotonic(),
        timeout_s=_MODE_DRIFT_TIMEOUT_S,
    )
    if mode_drifted:
        latch_arm_fault(
            shared,
            st.arm,
            f"controller mode drift (cached mode={report_mode}, expected 6)",
            on_poll=heartbeat,
        )
        return _Flow.EXIT

    # Publish state
    _write_arm_frame(
        shared,
        st.frame,
        qpos=qpos,
        qvel=qvel,
        tau=tau,
        eef_pos=eef_pos,
        eef_rot6d=eef_rot6d,
        error_code=error_code,
        connected=arm_connected,
        mode=report_mode,
        accepts_motion_commands=st.accepts_motion_commands,
        tracking_err=tracking_err,
        cmd=st.last_cmd,
        source_ns=st.last_state_source_ns,
        state_valid=bool(arm_connected and fk_valid),
    )

    if state_read_fault:
        st.terminal_feedback_detail = "persistent get_joint_states failure"
        shared.error_state.value = True
        logger.error(
            "arm_loop: %d consecutive feedback-read failures — latching global fault",
            st.state_err_counter.count,
        )
        return _Flow.EXIT
    if fk_fault:
        st.terminal_feedback_detail = "persistent ArmFK failure"
        shared.error_state.value = True
        logger.error(
            "arm_loop: %d consecutive FK failures — latching global fault",
            st.fk_err_counter.count,
        )
        return _Flow.EXIT
    return _Flow.PROCEED


def _step(st: _LoopState, shared: Any) -> _Flow:
    """Run one iteration; ``safety`` is read once and shared across the blocks."""
    safety = shared.safety_state.value
    flow = _transition_safety(st, shared, safety)
    if flow is _Flow.EXIT:
        return flow
    if st.mode_transition is not None:
        # Non-blocking Mode-6 entry in flight: drain nothing this tick, publish a
        # not-ready (accepts=0) frame, and confirm readiness in place.
        return _observe_and_publish(st, shared)
    flow = _consume_command(st, shared, safety)
    if flow is not _Flow.PROCEED:
        return flow
    return _observe_and_publish(st, shared)


def arm_loop(shared, config: ArmLoopConfig | None = None) -> None:
    """Arm process entry point — reads arm_action_q, servos arm via Mode 6.

    mp.Process target communicating exclusively through SharedStorage.
    """
    cfg = config or ArmLoopConfig()
    st = _startup(shared, cfg)
    if st is None:
        return
    heartbeat = lambda: shared.set_heartbeat("arm", time.monotonic())
    limiter = RateManager(st.cfg.arm_loop_hz, label="arm")
    stopped_cleanly = False
    try:
        while shared.is_running.value:
            shared.set_heartbeat("arm", time.monotonic())
            if shared.estop_request.value:
                # emergency_stop() requests State 4 without cutting motor power.
                try:
                    st.arm.emergency_stop()
                except Exception:
                    logger.warning(
                        "arm_loop: emergency_stop call failed; cleanup will enforce state 4",
                        exc_info=True,
                    )
                break
            flow = _step(st, shared)
            if flow is _Flow.EXIT:
                break
            if flow is _Flow.NEXT:
                continue
            limiter.wait()
    finally:
        # Cleanup: best-effort state 4 + disconnect (confirm without clearing
        # the already-latched error).
        cleanup_stop = stop_controller(st.arm, on_poll=heartbeat)
        if cleanup_stop.confirmed:
            stopped_cleanly = True
        else:
            logger.warning("arm_loop: cleanup failed: %s", cleanup_stop.reason)
            shared.error_state.value = True
        _disconnect_arm(st.arm)
        if st.terminal_feedback_detail is not None:
            logger.error("arm_loop: %s", st.terminal_feedback_detail)
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
