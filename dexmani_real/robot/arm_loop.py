"""Arm servo loop — Mode 6 joint online trajectory planning for xArm7.

``arm_loop(shared)`` is the mp.Process entry point (SharedStorage only). It
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

from dexmani_real.config.runtime import ArmLoopConfig
from dexmani_real.policy.safety import worker_validate_arm
from dexmani_real.robot.xarm7 import HomeAborted, XArm7
from dexmani_real.robot.safety import SafetyState
from dexmani_real.shm.shared_storage import new_frame
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate_manager import RateManager
from dexmani_real.utils.schema import (
    ARM_COMMAND_DTYPE,
    ARM_STATE_DTYPE,
)

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

    cfg: ArmLoopConfig
    arm: XArm7
    frame: Any
    last_target: np.ndarray
    last_cmd: _CmdState = field(default_factory=_CmdState.idle)
    last_state_source_ns: int = field(default_factory=time.monotonic_ns)


def _parse_arm_action_metadata(action: Any) -> tuple[int, bool]:
    """Return ``(sequence, is_hold)`` for a fixed command frame."""
    if (
        isinstance(action, np.ndarray)
        and action.shape == (1,)
        and action.dtype == ARM_COMMAND_DTYPE
    ):
        return int(action["action_id"][0]), bool(action["is_hold"][0])
    return 0, False


def _read_latest_arm_command(shared: Any, armed_at_seq: int) -> Any | None:
    """Return the newest servo endpoint worth applying, or ``None``.

    The transport is latest-wins: history is never replayed, so an endpoint is
    applied only when it was created after motion was armed and is not stale.
    """
    result = shared.arm_cmd_ring.read_latest()
    if result is None:
        return None
    command = result[0]
    if not worker_validate_arm(
        command,
        armed_at_seq=armed_at_seq,
        now_monotonic_ns=time.monotonic_ns(),
    ):
        return None
    return command


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


def _publish_identity(shared: Any, arm: XArm7, cfg: ArmLoopConfig) -> None:
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
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    shared.arm_device_identity.value = encoded[:1023].ljust(1024, b"\x00")


def _startup(shared: Any, arm: XArm7, cfg: ArmLoopConfig) -> _LoopState:
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
        cfg.arm_ip,
        cfg.arm_loop_hz,
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


def _handle_servo_command(st: _LoopState, shared: Any, action: Any) -> None:
    """Validate and servo one endpoint command.

    Fail-fast: a raised SDK exception or a non-zero return propagates to the
    worker's top-level handler, which latches ``error_state`` and stops the
    controller in cleanup.
    """
    target = np.asarray(action["qpos_cmd"][0], dtype=np.float64)
    st.last_target = target.copy()  # target is 2π-canonicalized by the producer; no wrap
    code = st.arm.servo(target)
    if code != 0:
        # Non-zero setter return is terminal even if the cached error is 0.
        raise RuntimeError(f"set_servo_angle failed (SDK code={code})")
    seq, is_hold = _parse_arm_action_metadata(action)
    st.last_cmd = _CmdState(seq, is_hold)


def _step(st: _LoopState, shared: Any) -> bool:
    """Run one iteration; return True when the worker must exit.

    Apply at most one command per tick — a queued HOME request takes priority
    over the newest servo endpoint — then observe + publish.  Motion is software-disarmed:
    outside ARMED/RUNNING (or on ``error_state``) nothing is consumed, while
    observation keeps publishing every tick.
    """
    safety = shared.safety_state.value
    if (
        safety in (SafetyState.ARMED, SafetyState.RUNNING)
        and not shared.error_state.value
    ):
        try:
            home_request = shared.arm_home_q.get(timeout=0.0)
        except Empty:
            home_request = None
        if home_request is not None:
            _handle_home(st, shared, home_request)
        else:
            command = _read_latest_arm_command(
                shared, int(shared.arm_armed_at_seq.value)
            )
            if command is not None:
                _handle_servo_command(st, shared, command)
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


def arm_loop(shared, config: ArmLoopConfig | None = None) -> None:
    """Arm process entry point — applies arm_cmd_ring endpoints via Mode 6.

    mp.Process target communicating exclusively through SharedStorage.  The
    single fail-fast boundary: any SDK/hardware failure raises into the
    top-level handler below, which latches ``error_state``; cleanup always
    does a best-effort stop + disconnect.
    """
    cfg = config or ArmLoopConfig()
    arm = XArm7(cfg)
    try:
        st = _startup(shared, arm, cfg)
        limiter = RateManager(cfg.arm_loop_hz, label="arm")
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
        logger.info("arm_loop: exited")
