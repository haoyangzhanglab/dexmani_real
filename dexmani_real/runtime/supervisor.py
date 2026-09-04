"""Process readiness, health supervision, and verified shutdown."""

from __future__ import annotations

import sys
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from dexmani_real.ipc.channels import RuntimeChannels
from dexmani_real.runtime.safety import SafetyState, transition
from dexmani_real.utils.feedback import validate_hand_feedback
from dexmani_real.utils.log import get_logger

if TYPE_CHECKING:
    from dexmani_real.runtime.workers import ShutdownReport

logger = get_logger(__name__)

_READY_POLL_INTERVAL_S = 0.2


def shutdown_processes(
    shared: RuntimeChannels,
    procs: list[Any],
    *,
    graceful_timeout_s: float = 5.0,
    disarm_if_clean: bool = False,
) -> ShutdownReport:
    """Stop workers and return their verified post-join safety state."""
    from dexmani_real.runtime.workers import shutdown_processes_verified

    report = shutdown_processes_verified(
        shared,
        procs,
        graceful_timeout_s=graceful_timeout_s,
        disarm_if_clean=disarm_if_clean,
    )
    if report.exits:
        print(
            "  shutdown: "
            + "  ".join(
                f"{item.name}={item.escalation}:{item.exitcode}"
                for item in report.exits
            )
        )
    return report


def run_supervisor(
    shared: RuntimeChannels,
    procs: list[Any],
    proc_names: list[str],
    heartbeat_names: list[str],
    *,
    status_interval_s: float = 30.0,
    heartbeat_timeouts_s: dict[str, float] | None = None,
    supervisor_hz: float | None = None,
) -> tuple[str, bool]:
    """Run the standard supervisor loop with resolved heartbeat settings.

    Returns ``(exit_reason, normal_exit)``.  *exit_reason* describes why the
    supervisor stopped; *normal_exit* is True for user-requested clean exits
    (Q key or KeyboardInterrupt), False for faults.

    The caller should have already transitioned to ARMED before calling this
    and must handle shutdown + DISARMED transition after it returns.
    """
    from dexmani_real.config.defaults import safety
    from dexmani_real.runtime.status import ExitReason
    from dexmani_real.runtime.workers import supervisor_exit_reason

    start_time = time.monotonic()
    last_status_s = start_time
    exit_reason = "unknown"
    normal_exit = False
    loop_hz = float(safety.supervisor_hz if supervisor_hz is None else supervisor_hz)
    if not np.isfinite(loop_hz) or loop_hz <= 0:
        raise ValueError("supervisor_hz must be finite and positive")
    if not np.isfinite(status_interval_s) or status_interval_s <= 0:
        raise ValueError("status_interval_s must be finite and positive")
    configured_timeouts = (
        safety.heartbeat_timeouts
        if heartbeat_timeouts_s is None
        else heartbeat_timeouts_s
    )
    if len(procs) != len(proc_names) or len(set(proc_names)) != len(proc_names):
        raise ValueError("proc_names must contain one unique name per process")
    if len(set(heartbeat_names)) != len(heartbeat_names):
        raise ValueError("heartbeat_names must be unique")
    extra_heartbeats = set(heartbeat_names) - set(proc_names)
    if extra_heartbeats:
        raise ValueError(
            "heartbeat_names must name running processes; "
            f"unknown={sorted(extra_heartbeats)}"
        )
    missing_timeouts = set(heartbeat_names) - set(configured_timeouts)
    if missing_timeouts:
        raise ValueError(f"missing heartbeat timeouts for {sorted(missing_timeouts)}")
    timeouts = {name: float(configured_timeouts[name]) for name in heartbeat_names}
    if any(not np.isfinite(timeout) or timeout <= 0 for timeout in timeouts.values()):
        raise ValueError("heartbeat timeouts must be finite and positive")
    try:
        while True:
            heartbeat_timestamps = {
                name: shared.get_heartbeat(name) for name in heartbeat_names
            }
            now = time.monotonic()
            heartbeat_ages = {
                name: (
                    now - timestamp_s
                    if np.isfinite(timestamp_s) and 0.0 < timestamp_s <= now
                    else float("inf")
                )
                for name, timestamp_s in heartbeat_timestamps.items()
            }
            reason = supervisor_exit_reason(shared, procs, heartbeat_ages, timeouts)
            if reason is ExitReason.ESTOP:
                exit_reason = "e-stop requested"
                transition(shared, SafetyState.FAULT)
                break
            if reason is ExitReason.STICKY_FAULT:
                exit_reason = "error_state set"
                transition(shared, SafetyState.FAULT)
                break
            if reason is ExitReason.WORKER_DEATH:
                dead_names = [
                    process.name for process in procs if process.exitcode is not None
                ]
                exit_reason = f"process died: {dead_names}"
                transition(shared, SafetyState.FAULT)
                break
            if reason is ExitReason.HEARTBEAT_TIMEOUT:
                stale = [
                    name for name, age in heartbeat_ages.items() if age > timeouts[name]
                ]
                exit_reason = f"heartbeat timeout: {stale}"
                transition(shared, SafetyState.FAULT)
                break
            if reason is ExitReason.EXPLICIT_QUIT:
                normal_exit = True
                exit_reason = "explicit quit"
                break

            if now - last_status_s >= status_interval_s:
                runtime_m = (now - start_time) / 60.0
                safety_state = shared.safety_state.value
                heartbeat_text = ", ".join(
                    f"{name}={heartbeat_ages[name]:.1f}s" for name in heartbeat_names
                )
                print(
                    f"  [supervisor]  runtime={runtime_m:.1f}min  safety={safety_state}  hb_age=({heartbeat_text})",
                    flush=True,
                )
                last_status_s = now

            time.sleep(1.0 / loop_hz)

    except KeyboardInterrupt:
        exit_reason = "KeyboardInterrupt"
        normal_exit = True
        shared.is_running.value = False

    print(f"  [supervisor exit] reason={exit_reason}", flush=True)
    return exit_reason, normal_exit


def wait_subsystem_ready(
    shared: RuntimeChannels,
    ready_checks: list[tuple[str, float]],
    procs: list[Any],
) -> bool:
    """Wait for each ``(name, timeout_s)`` subsystem to become ready.

    Checks ``error_state`` and process liveness on every poll tick.
    Returns True if all subsystems are ready, False if any fail.

    The caller is responsible for printing pre-wait user messages
    (e.g. "put on Quest headset") before calling this function.
    """
    for name, timeout in ready_checks:
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError(
                f"readiness timeout for {name!r} must be finite and positive"
            )
        deadline = time.monotonic() + timeout
        ready = False
        failure_logged = False
        while time.monotonic() < deadline:
            if shared.error_state.value:
                logger.error("subsystem=%s init failed: error_state set", name)
                failure_logged = True
                break
            if not all(p.is_alive() for p in procs):
                dead_names = [p.name for p in procs if not p.is_alive()]
                logger.error(
                    "subsystem=%s init failed: process(es) %s exited prematurely",
                    name,
                    dead_names,
                )
                failure_logged = True
                break
            if shared.is_ready(name):
                ready = True
                break
            time.sleep(_READY_POLL_INTERVAL_S)
        if not ready and not failure_logged:
            logger.error("subsystem=%s ready_timeout=%ds", name, timeout)
        if not ready:
            return False
    return True


def _hand_feedback_issue(hand_data: Any, *, max_age_s: float) -> str | None:
    """Delegated fail-closed health check over one hand state record.

    Single source of truth for "is this hand feedback usable" in the supervisor,
    matching ``control.publication.read_hand_feedback`` and the teleop predicates.
    Returns the rejection reason, or ``None`` when healthy.
    """
    return validate_hand_feedback(
        connected=bool(hand_data["connected"][0]),
        state_valid=bool(hand_data["state_valid"][0]),
        source_monotonic_ns=int(hand_data["source_monotonic_ns"][0]),
        now_monotonic_ns=time.monotonic_ns(),
        max_age_s=max_age_s,
        qpos=np.asarray(hand_data["qpos"][0], dtype=np.float64),
    )


def print_health_summary(
    shared: RuntimeChannels, *, hand_feedback_max_age_s: float | None = None
) -> None:
    """Print a pre-flight health summary from ring data (arm, hand, VR, camera)."""
    if hand_feedback_max_age_s is None:
        from dexmani_real.config.defaults import safety

        hand_feedback_max_age_s = float(safety.heartbeat_timeouts["hand"])
    print("\n── Health Check ──")

    arm_result = shared.arm_state_ring.read_latest()
    if arm_result is not None:
        arm_data, _, _ = arm_result
        arm_connected = bool(arm_data["connected"][0])
        arm_error = int(arm_data["error_code"][0])
        arm_state_valid = bool(arm_data["state_valid"][0])
        arm_qpos = np.asarray(arm_data["qpos"][0], dtype=np.float64)
        arm_qpos_ok = int(np.all(np.isfinite(arm_qpos)))
        arm_ok = (
            arm_connected and arm_error == 0 and arm_state_valid and bool(arm_qpos_ok)
        )
        print(
            f"  arm   {'OK' if arm_ok else 'FAIL':>4s}  connected={int(arm_connected)}  "
            f"valid={int(arm_state_valid)}  error={arm_error}  qpos_ok={arm_qpos_ok}"
        )
    else:
        print("  arm   ----  (no data yet)")

    hand_result = shared.hand_state_ring.read_latest()
    if hand_result is not None:
        hand_data, _, _ = hand_result
        hand_connected = bool(hand_data["connected"][0])
        hand_state_valid = bool(hand_data["state_valid"][0])
        hand_qpos = np.asarray(hand_data["qpos"][0], dtype=np.float64)
        hand_qpos_ok = int(np.all(np.isfinite(hand_qpos)))
        hand_ok = (
            _hand_feedback_issue(hand_data, max_age_s=hand_feedback_max_age_s) is None
        )
        print(
            f"  hand  {'OK' if hand_ok else 'FAIL':>4s}  connected={int(hand_connected)}  "
            f"valid={int(hand_state_valid)}  "
            f"qpos_ok={hand_qpos_ok}"
        )
    else:
        print("  hand  ----  (no data yet)")

    vr_result = shared.vr_ring.read_latest()
    if vr_result is not None:
        vr_data, _, _ = vr_result
        vr_age_s = (
            (time.monotonic_ns() - int(vr_data["local_recv_ns"][0])) / 1e9
            if vr_data["local_recv_ns"][0] > 0
            else -1
        )
        print(
            f"  vr     OK   age={vr_age_s:.1f}s  seq={int(vr_data['sequence_id'][0])}"
        )
    else:
        print("  vr    ----  (no data yet)")

    cam_serial_bytes = shared.camera_serial.value.rstrip(b"\x00")
    if cam_serial_bytes:
        print(f"  cam    OK   serial={cam_serial_bytes.decode()}")
    elif shared.get_heartbeat("camera") > 0:
        print("  cam    OK   serial=unknown")
    else:
        print("  cam   ----  (no data yet)")

    print("──")
    sys.stdout.flush()
