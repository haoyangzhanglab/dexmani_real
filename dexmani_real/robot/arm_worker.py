"""Arm worker — Mode 6 joint online trajectory planning for xArm7.

``arm_loop(shared, config)`` is the mp.Process entry point (RuntimeChannels only). It
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

The worker validates only the HARD boundary (finite targets inside physical
joint limits) at the SDK fence; the soft command-jump bound is owned once by
the producers through ``control/projection.py`` and is never re-rejected here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import partial
from queue import Empty
from typing import Any

import numpy as np

from dexmani_real.config.defaults import ArmParams
from dexmani_real.ipc.channels import new_frame
from dexmani_real.ipc.command_stream import CommandStreamConsumer
from dexmani_real.ipc.schema import ARM_STATE_DTYPE
from dexmani_real.robot.command_validation import check_worker_arm_target
from dexmani_real.robot.drivers.xarm7 import HomeAborted, XArm7, describe_controller_error
from dexmani_real.runtime.safety import (
    SafetyState,
    StopRequest,
    coupled_command_may_cross_sdk,
    read_motion_permit,
)
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate import LoopRate

logger = get_logger(__name__)


@dataclass(frozen=True)
class _CmdState:
    """Metadata of the last command accepted by the SDK."""

    is_hold: bool
    accepted_monotonic_ns: int
    generation: int = 0
    accepted_sequence: int = 0

    @classmethod
    def idle(cls) -> "_CmdState":
        return cls(False, 0, 0, 0)


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
    consumer: CommandStreamConsumer
    last_cmd: _CmdState = field(default_factory=_CmdState.idle)
    last_state_source_ns: int = field(default_factory=time.monotonic_ns)


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
    frame["last_cmd_accepted_monotonic_ns"][0] = cmd.accepted_monotonic_ns
    frame["last_cmd_generation"][0] = cmd.generation
    frame["last_cmd_accepted_sequence"][0] = cmd.accepted_sequence
    frame["last_cmd_is_hold"][0] = int(cmd.is_hold)
    frame["source_monotonic_ns"][0] = source_ns
    frame["publish_monotonic_ns"][0] = time.monotonic_ns()
    frame["state_valid"][0] = int(state_valid)
    shared.arm_state_ring.write(frame)


def _startup(shared: Any, arm: XArm7, cfg: ArmParams) -> _LoopState:
    """Connect, enter Mode 6 once, publish the initial frame, signal ready.

    Returns a fully-initialized ``_LoopState``.  Any failure raises to the
    worker's top-level handler (which latches ``error_state``); cleanup does
    the best-effort stop + disconnect.
    """
    heartbeat = lambda: shared.set_heartbeat("arm", time.monotonic())
    logger.debug("arm_loop: LOADING")
    arm.connect(on_poll=heartbeat)
    qpos, qvel, tau = arm.read()
    # Attach this worker as the arm consumer of the ordered command FIFO
    # before signalling ready, so the publisher's capacity watermark sees an
    # attached consumer from the first committable command on.
    consumer = CommandStreamConsumer(shared, shared.arm_cmd_consumed_sequence)
    st = _LoopState(
        cfg=cfg,
        arm=arm,
        frame=new_frame(ARM_STATE_DTYPE),
        last_target=qpos.copy(),
        consumer=consumer,
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
    logger.info("arm_loop: HOME complete")
    # Publish only after the driver's settle and mode-restoration lifecycle.
    # S/Q and generation changes must win over a late HOME completion.
    with shared.motion_lock:
        if (
            _home_abort_reason(shared, generation) is None
            and not shared.quit_requested.value
            and int(shared.stop_request.value) == int(StopRequest.NONE)
        ):
            shared.arm_home_completed_generation.value = int(generation)


def _handle_servo_command(
    st: _LoopState,
    shared: Any,
    action: Any,
    sequence: int,
) -> None:
    """Validate the hard boundary and servo one ordered endpoint command.

    The soft command-jump bound was already applied once by the producer's
    projection; the worker keeps only the hard SDK-boundary validation and
    never re-rejects the same soft threshold. Fail-fast: a raised SDK
    exception or a non-zero return propagates to the worker's top-level
    handler, which latches ``error_state`` and stops the controller in
    cleanup. The FIFO cursor advances only after explicit SDK acceptance. At
    most one send happens per tick — the worker never drains the queue to
    catch up.
    """
    command_generation = int(action["run_generation"][0])
    target = np.asarray(action["arm_qpos"][0], dtype=np.float64)
    issue = check_worker_arm_target(
        target,
        joint_limit_lower_rad=np.asarray(st.cfg.joint_limit_lower, dtype=np.float64),
        joint_limit_upper_rad=np.asarray(st.cfg.joint_limit_upper, dtype=np.float64),
    )
    # This is the sole command-authority fence and the final operation before
    # an otherwise valid target crosses the xArm SDK boundary. A transient
    # same-generation fence miss retries this record on the next tick; any
    # real revocation advanced the generation and resyncs the cursor.
    if not coupled_command_may_cross_sdk(shared, run_generation=command_generation):
        return
    if issue is not None:
        raise RuntimeError(f"unsafe servo sequence={sequence}: {issue}")
    code = st.arm.servo(target)
    if code != 0:
        # Setter code 1 only means that the controller has an error. Read the
        # live register here so the terminal fault reports the physical cause.
        controller_error = st.arm.read_live_error_code()
        if controller_error != 0:
            raise RuntimeError(
                f"set_servo_angle failed (SDK code={code}, controller C{controller_error}: "
                f"{describe_controller_error(controller_error)})"
            )
        raise RuntimeError(f"set_servo_angle failed (SDK code={code})")
    accepted_monotonic_ns = time.monotonic_ns()
    st.last_target = target.copy()  # producer owns 2π canonicalization
    st.last_cmd = _CmdState(
        bool(action["is_hold"][0]),
        accepted_monotonic_ns,
        generation=command_generation,
        accepted_sequence=sequence,
    )
    # The SDK accepted this endpoint: the record is fully processed and its
    # FIFO slot is released to the publisher's capacity watermark.
    st.consumer.advance()


def _consume_one_arm_command(st: _LoopState, shared: Any, permit_generation: int) -> None:
    """Consume at most one ordered FIFO record for the arm this tick."""
    consumer = st.consumer
    consumer.resync_if_stale_generation(permit_generation)
    if consumer.generation != permit_generation:
        return  # The locked resync observed a newer epoch; reread next tick.
    record = consumer.next_record()
    if record is None:
        return  # EMPTY is a wait, never a fault
    command, sequence = record
    if not bool(command["arm_present"][0]):
        # An absent actuator advances only its consumer; no SDK involvement
        # and no acceptance is implied.
        consumer.advance()
        return
    _handle_servo_command(st, shared, command, sequence)


def _step(st: _LoopState, shared: Any, limiter: LoopRate) -> bool:
    """Run one iteration; return True when the worker must exit.

    Apply at most one command per tick — a queued HOME request takes priority
    over the next ordered servo endpoint — then observe + publish.  Motion is software-disarmed:
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
            # HOME is intentionally blocking; begin a fresh worker schedule
            # instead of reporting its Mode 0/6 transition as loop overrun.
            limiter.reset()
        else:
            _consume_one_arm_command(st, shared, permit.run_generation)
    return _observe_and_publish(st, shared)


def _observe_and_publish(st: _LoopState, shared: Any) -> bool:
    """Read state, check the controller error, publish.

    Returns True when the worker must exit; failures raise instead.  A failed
    state read or a non-zero controller error raises to the worker's single
    top-level handler.
    """
    qpos, qvel, tau = st.arm.read()
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


def arm_loop(shared: Any, config: ArmParams) -> None:
    """Arm process entry point — applies coupled servo endpoints via Mode 6.

    mp.Process target communicating exclusively through RuntimeChannels.  The
    single fail-fast boundary: any SDK/hardware failure raises into the
    top-level handler below, which latches ``error_state``; cleanup always
    does a best-effort stop + disconnect.
    """
    cfg = config
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
            if _step(st, shared, limiter):
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
            logger.info("arm_loop: exited")
