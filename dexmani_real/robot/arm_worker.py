"""Arm worker — Mode 6 joint online trajectory planning for xArm7.

``arm_loop(shared)`` is the mp.Process entry point (RuntimeChannels only). It
connects and enters servo Mode 6 once at startup, then runs a fixed-rate
loop: consume at most one command (HOME or servo), observe, publish.

Mode 6 is held for the whole runtime (re-entered only by the HOME path);
DISARMED/ARMED/RUNNING are software lifecycle states and never switch the
controller mode.  DISARMED means "publish no servo commands" — the arm holds
its position (software disarm).

Error handling is fail-fast: any hardware/SDK failure raises to the single
top-level handler, which latches ``error_state``; cleanup always does a
best-effort stop + disconnect.  No retry counters, no last-known fallbacks,
no error-classification framework.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from functools import partial
from queue import Empty
from typing import Any

import numpy as np

from dexmani_real.config.defaults import ArmParams
from dexmani_real.ipc.channels import new_frame
from dexmani_real.ipc.schema import ARM_STATE_DTYPE, COUPLED_COMMAND_DTYPE
from dexmani_real.robot.command_validation import check_worker_arm_command
from dexmani_real.robot.xarm7 import HomeAborted, XArm7
from dexmani_real.runtime.safety import (
    CoupledCommandTicket,
    SafetyState,
    coupled_command_ticket_allows_execution,
    read_motion_permit,
)
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate import LoopRate

logger = get_logger(__name__)


@dataclass(frozen=True)
class _CmdState:
    """Metadata of the last command accepted by the SDK."""

    seq: int
    is_hold: bool

    @classmethod
    def idle(cls) -> "_CmdState":
        return cls(0, False)


@dataclass
class _LoopState:
    """Loop-carried state shared by the per-iteration functions.

    Kept minimal by design: no error counters or health trackers — hardware
    failures fail fast at the worker top level.
    """

    cfg: ArmParams
    arm: XArm7
    frame: Any
    last_target: np.ndarray
    last_measured_qpos: np.ndarray
    last_command_generation: int
    last_cmd: _CmdState = field(default_factory=_CmdState.idle)
    last_processed_ring_sequence: int = 0
    servo_call_count: int = 0
    duplicate_command_skip_count: int = 0
    last_state_source_ns: int = field(default_factory=time.monotonic_ns)


def _read_latest_arm_command(
    shared: Any,
) -> tuple[Any, CoupledCommandTicket] | None:
    """Read the newest arm-present record; the handler owns safety checks."""
    result = shared.coupled_cmd_ring.read_latest()
    if result is None:
        return None
    command, _published_ns, ring_sequence = result
    if not bool(command["arm_present"][0]):
        return None
    ticket = CoupledCommandTicket(
        run_generation=int(command["run_generation"][0]),
        ring_sequence=int(ring_sequence),
    )
    return command, ticket


def _home_abort_reason(shared: Any, generation: int) -> str | None:
    """Return why an in-progress HOME must stop, or ``None`` to continue."""
    if not shared.is_running.value:
        return "shutdown requested"
    if shared.estop_request.value:
        return "e-stop requested"
    if shared.error_state.value:
        return "sticky error_state set during homing"
    if int(shared.safety_state.value) == int(SafetyState.FAULT):
        return "FAULT during homing"
    if int(shared.safety_state.value) != int(SafetyState.ARMED):
        return "safety state is not ARMED during homing"
    if int(shared.run_generation.value) != generation:
        return "run generation changed during homing"
    return None


def _mode6_restore_allowed(shared: Any) -> bool:
    """Whether a cleanly-aborted HOME may restore servo Mode 6.

    Mode 6 is held for the whole runtime (software disarm), so it is restored
    after any non-faulted interruption — including DISARMED, where the arm
    simply holds position.  Skipped on a latched error, an active e-stop,
    FAULT, or an already-stopped runtime.
    """
    return (
        bool(shared.is_running.value)
        and int(shared.safety_state.value) != int(SafetyState.FAULT)
        and not shared.error_state.value
        and not shared.estop_request.value
    )


def _write_arm_frame(
    shared: Any,
    frame: Any,
    *,
    qpos: np.ndarray,
    qvel: np.ndarray,
    tau: np.ndarray,
    error_code: int,
    tracking_err: float,
    connected: bool = True,
    cmd: _CmdState,
    source_ns: int,
    state_valid: bool,
) -> None:
    """Publish one fully-populated ARM_STATE frame to ``arm_state_ring``."""
    frame["qpos"][0] = qpos
    frame["qvel"][0] = qvel
    frame["tau"][0] = tau
    frame["error_code"][0] = int(error_code)
    frame["connected"][0] = 1 if connected else 0
    frame["tracking_err"][0] = tracking_err
    frame["last_cmd_seq"][0] = cmd.seq
    frame["last_cmd_is_hold"][0] = int(cmd.is_hold)
    frame["source_monotonic_ns"][0] = source_ns
    frame["publish_monotonic_ns"][0] = time.monotonic_ns()
    frame["state_valid"][0] = int(state_valid)
    shared.arm_state_ring.write(frame)


def _publish_identity(shared: Any, arm: XArm7, cfg: ArmParams) -> None:
    """Publish device identity (the axis count is validated by ``XArm7.connect``)."""
    if not hasattr(shared, "arm_device_identity"):
        return
    api = arm.api
    device_type = str(getattr(api, "device_type", "") or "")
    sn = str(getattr(api, "sn", "") or "")
    firmware = tuple(getattr(api, "version_number", ()) or ())
    firmware_str = (
        ".".join(str(v) for v in firmware)
        if firmware
        else str(getattr(api, "version", "unavailable") or "unavailable")
    )
    identity = {
        "axis": arm.axis,
        "device_type": device_type or "unavailable",
        "model": cfg.device_profile or device_type or "unavailable",
        "serial_number": sn or "unavailable",
        "firmware_version": firmware_str,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    shared.arm_device_identity.value = encoded[:1023].ljust(1024, b"\x00")


def _startup(shared: Any, arm: XArm7, cfg: ArmParams) -> _LoopState:
    """Connect, enter Mode 6 once, publish the initial frame, signal ready.

    Returns a fully-initialized ``_LoopState``.  Any failure raises to the
    worker's top-level handler (which latches ``error_state``); cleanup does
    the best-effort stop + disconnect.
    """
    heartbeat = lambda: shared.set_heartbeat("arm", time.monotonic())
    logger.debug("arm_loop: LOADING")
    arm.connect(on_poll=heartbeat)
    _publish_identity(shared, arm, cfg)
    qpos, qvel, tau = arm.read()
    st = _LoopState(
        cfg=cfg,
        arm=arm,
        frame=new_frame(ARM_STATE_DTYPE),
        last_target=qpos.copy(),
        last_measured_qpos=qpos.copy(),
        last_command_generation=int(shared.run_generation.value),
    )
    # Publish the initial frame before signaling ready.
    _write_arm_frame(
        shared,
        st.frame,
        qpos=qpos,
        qvel=qvel,
        tau=tau,
        error_code=0,
        tracking_err=0.0,
        cmd=st.last_cmd,
        source_ns=st.last_state_source_ns,
        state_valid=True,
    )
    shared.set_heartbeat("arm", time.monotonic())  # heartbeat before ready
    shared.set_ready("arm")
    logger.debug("arm_loop: READY")
    logger.info(
        "arm_loop: ready and DISARMED (Mode 6 held, software disarm; ip=%s, hz=%.0f)",
        cfg.ip,
        cfg.loop_hz,
    )
    return st


def _publish_homing_feedback(
    st: _LoopState,
    shared: Any,
    qpos: np.ndarray,
    qvel: np.ndarray,
    tau: np.ndarray,
    target: np.ndarray,
) -> None:
    """Publish a homing-milestone frame."""
    st.last_state_source_ns = time.monotonic_ns()
    error_code = st.arm.error_code
    _write_arm_frame(
        shared,
        st.frame,
        qpos=qpos,
        qvel=qvel,
        tau=tau,
        error_code=error_code,
        tracking_err=float(np.max(np.abs(qpos - target))),
        cmd=st.last_cmd,
        source_ns=st.last_state_source_ns,
        state_valid=True,
    )


def _handle_home(st: _LoopState, shared: Any, request: tuple) -> None:
    """Run planned homing for a queued ``(waypoints, final_qpos, generation)``.

    Blocks the worker: the arm drives the collision-validated milestones in
    Mode 0, then restores Mode 6.  A stale request (its generation advanced
    after planning) is discarded.  A clean runtime interruption (e-stop,
    shutdown, DISARM, generation change) stops the controller and restores
    Mode 6 without faulting; any other failure raises into the top-level
    handler, which latches ``error_state``.
    """
    waypoints, final_qpos, generation = request
    if int(shared.run_generation.value) != generation:
        logger.warning("arm_loop: discarding stale-generation HOME request")
        return
    logger.info(
        "arm_loop: HOME — planned homing (%d validated milestones)",
        len(waypoints),
    )
    heartbeat = lambda: shared.set_heartbeat("arm", time.monotonic())
    try:
        st.arm.home(
            waypoints,
            final_qpos,
            on_poll=heartbeat,
            feedback_callback=partial(_publish_homing_feedback, st, shared),
            abort_check=lambda: _home_abort_reason(shared, generation),
        )
    except HomeAborted as exc:
        logger.warning("arm_loop: HOME aborted — %s", exc)
        st.arm.stop()
        if _mode6_restore_allowed(shared):
            st.arm.enter_mode6(on_poll=heartbeat)
        return
    st.last_target = np.asarray(final_qpos, dtype=np.float64).copy()
    st.last_command_generation = int(generation)
    logger.info("arm_loop: HOME complete")


def _handle_servo_command(
    st: _LoopState,
    shared: Any,
    action: Any,
    ticket: CoupledCommandTicket,
) -> None:
    """Validate and servo one endpoint command.

    Fail-fast: a raised SDK exception or a non-zero return propagates to the
    worker's top-level handler, which latches ``error_state`` and stops the
    controller in cleanup.
    """
    if ticket.ring_sequence <= st.last_processed_ring_sequence:
        st.duplicate_command_skip_count += 1
        return

    # A coupled-ring endpoint is an event, not a level-triggered setpoint. Mark
    # it consumed before validation so a superseded or rejected snapshot cannot
    # be retried on every worker tick. A newer ring sequence remains eligible.
    st.last_processed_ring_sequence = ticket.ring_sequence
    permit = read_motion_permit(shared)
    command_generation = int(action["run_generation"][0])
    if command_generation != permit.run_generation or not (
        coupled_command_ticket_allows_execution(shared, ticket=ticket)
    ):
        return
    jump_reference = (
        st.last_target
        if command_generation == st.last_command_generation
        else st.last_measured_qpos
    )
    issue = check_worker_arm_command(
        action,
        expected_run_generation=permit.run_generation,
        now_monotonic_ns=time.monotonic_ns(),
        joint_limit_lower_rad=np.asarray(st.cfg.joint_limit_lower, dtype=np.float64),
        joint_limit_upper_rad=np.asarray(st.cfg.joint_limit_upper, dtype=np.float64),
        previous_command_qpos_rad=jump_reference,
        max_command_jump_rad=st.cfg.max_servo_command_jump_rad,
    )
    # A newer command or a stop/fault may arrive while validation is running.
    # Do not execute or fault on a snapshot that no longer owns the slot.
    if not coupled_command_ticket_allows_execution(shared, ticket=ticket):
        return
    if issue is not None:
        if issue.fault:
            raise RuntimeError(
                "unsafe servo action_id="
                f"{int(action['action_id'][0])}: {issue.reason}"
            )
        logger.info(
            "arm_loop: discarded action_id=%d: %s",
            int(action["action_id"][0]),
            issue.reason,
        )
        return
    target = np.asarray(action["arm_qpos"][0], dtype=np.float64)
    code = st.arm.servo(target)
    if code != 0:
        # Non-zero setter return is terminal even if the cached error is 0.
        raise RuntimeError(f"set_servo_angle failed (SDK code={code})")
    st.servo_call_count += 1
    st.last_target = target.copy()  # producer owns 2π canonicalization
    st.last_command_generation = command_generation
    st.last_cmd = _CmdState(
        int(action["action_id"][0]),
        bool(action["is_hold"][0]),
    )


def _step(st: _LoopState, shared: Any) -> bool:
    """Run one iteration; return True when the worker must exit.

    Apply at most one command per tick — a queued HOME request takes priority
    over the newest servo endpoint — then observe + publish.  Motion is software-disarmed:
    outside ARMED/RUNNING (or on ``error_state``) nothing is consumed, while
    observation keeps publishing every tick.
    """
    permit = read_motion_permit(shared)
    if permit.allows_motion and not shared.error_state.value:
        try:
            home_request = shared.arm_home_q.get(timeout=0.0)
        except Empty:
            home_request = None
        if home_request is not None:
            _handle_home(st, shared, home_request)
        else:
            latest = _read_latest_arm_command(shared)
            if latest is not None:
                command, ticket = latest
                _handle_servo_command(st, shared, command, ticket)
    return _observe_and_publish(st, shared)


def _observe_and_publish(st: _LoopState, shared: Any) -> bool:
    """Read state, check the controller error, publish.

    Returns True when the worker must exit; failures raise instead.  A failed
    state read or a non-zero controller error raises to the worker's single
    top-level handler.
    """
    qpos, qvel, tau = st.arm.read()
    st.last_measured_qpos = qpos.copy()
    st.last_state_source_ns = time.monotonic_ns()

    tracking_err = float(np.max(np.abs(qpos - st.last_target)))

    error_code = st.arm.error_code
    if error_code != 0:
        raise RuntimeError(f"controller error C{error_code}")

    _write_arm_frame(
        shared,
        st.frame,
        qpos=qpos,
        qvel=qvel,
        tau=tau,
        error_code=error_code,
        tracking_err=tracking_err,
        cmd=st.last_cmd,
        source_ns=st.last_state_source_ns,
        state_valid=True,
    )
    return False


def arm_loop(shared, config: ArmParams | None = None) -> None:
    """Arm process entry point — applies coupled servo endpoints via Mode 6.

    mp.Process target communicating exclusively through RuntimeChannels.  The
    single fail-fast boundary: any SDK/hardware failure raises into the
    top-level handler below, which latches ``error_state``; cleanup always
    does a best-effort stop + disconnect.
    """
    cfg = config if config is not None else ArmParams()
    arm = XArm7(cfg)
    st: _LoopState | None = None
    try:
        st = _startup(shared, arm, cfg)
        limiter = LoopRate(cfg.loop_hz, label="arm")
        while shared.is_running.value:
            shared.set_heartbeat("arm", time.monotonic())
            if shared.estop_request.value:
                # Best-effort: cleanup enforces the final state-4 stop.
                arm.emergency_stop()
                break
            if _step(st, shared):
                break
            limiter.wait()
    except Exception:
        shared.error_state.value = True
        logger.exception("arm_loop: worker failed")
    finally:
        # Cleanup requests a best-effort state-4 stop and disconnect; firmware is the backstop.
        arm.stop()
        arm.close()
        if st is None:
            logger.info("arm_loop: exited before loop startup")
        else:
            logger.info(
                "arm_loop: exited (servo_calls=%d, duplicate_skips=%d)",
                st.servo_call_count,
                st.duplicate_command_skip_count,
            )
