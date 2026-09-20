"""Process readiness, health supervision, and verified shutdown."""

from __future__ import annotations

import math
import sys
import time
from collections.abc import Collection, Iterable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from dexmani_real.ipc.channels import RuntimeChannels
from dexmani_real.runtime.safety import (
    SafetyState,
    RunEndReason,
    revoke_motion,
    read_run_state_snapshot,
    revoke_motion_if_generation,
    transition,
)
from dexmani_real.runtime.status import ExitReason
from dexmani_real.utils.feedback import validate_hand_feedback
from dexmani_real.utils.log import get_logger

if TYPE_CHECKING:
    from dexmani_real.runtime.processes import ProcessSpec, ShutdownReport

logger = get_logger(__name__)

_READY_POLL_INTERVAL_S = 0.2


def shutdown_processes(
    shared: RuntimeChannels,
    procs: list[Any],
    *,
    graceful_timeout_s: float = 5.0,
    disarm_if_clean: bool = False,
    service_process_names: Collection[str] = (),
) -> ShutdownReport:
    """Stop workers and return their verified post-join safety state."""
    from dexmani_real.runtime.processes import shutdown_processes_verified

    report = shutdown_processes_verified(
        shared,
        procs,
        graceful_timeout_s=graceful_timeout_s,
        disarm_if_clean=disarm_if_clean,
        service_process_names=service_process_names,
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


def supervisor_exit_reason(
    shared: Any,
    processes: Iterable[Any],
    heartbeat_ages_s: Mapping[str, float],
    heartbeat_timeouts_s: Mapping[str, float],
    *,
    service_process_names: Collection[str] = (),
) -> ExitReason:
    """Apply the fixed safety-first supervisor priority.

    Critical faults take precedence over terminal session failure and Q.
    Optional service death/heartbeat loss is evidence-only and cannot mask Q;
    the caller records it while continuing the entire control run plan.
    """
    if bool(shared.estop_request.value):
        return ExitReason.ESTOP
    if bool(shared.error_state.value):
        return ExitReason.STICKY_FAULT
    stopped = [process for process in processes if process.exitcode is not None]
    critical_stopped = [
        process for process in stopped if process.name not in service_process_names
    ]
    service_stopped = [
        process for process in stopped if process.name in service_process_names
    ]
    explicit_quit = bool(shared.quit_requested.value) or not bool(
        shared.is_running.value
    )
    if critical_stopped:
        return ExitReason.WORKER_DEATH
    critical_heartbeat_timeout = False
    service_heartbeat_timeout = False
    for name, timeout in heartbeat_timeouts_s.items():
        age_s = float(heartbeat_ages_s.get(name, float("inf")))
        timeout_s = float(timeout)
        if (
            not math.isfinite(age_s)
            or age_s < 0.0
            or not math.isfinite(timeout_s)
            or timeout_s <= 0.0
            or age_s > timeout_s
        ):
            if name in service_process_names:
                service_heartbeat_timeout = True
            else:
                critical_heartbeat_timeout = True
    if critical_heartbeat_timeout:
        return ExitReason.HEARTBEAT_TIMEOUT
    if bool(shared.session_failed.value):
        return ExitReason.SERVICE_FAILURE
    if explicit_quit:
        return ExitReason.EXPLICIT_QUIT
    if service_stopped or service_heartbeat_timeout:
        return ExitReason.EVIDENCE_FAILURE
    return ExitReason.NONE


def run_supervisor(
    shared: RuntimeChannels,
    procs: list[Any],
    *,
    status_interval_s: float = 30.0,
    heartbeat_timeouts_s: Mapping[str, float],
    supervisor_hz: float | None = None,
    service_process_names: Collection[str] = (),
    max_running_s: float | None = None,
) -> tuple[str, bool]:
    """Run the standard supervisor loop with resolved heartbeat settings.

    Returns ``(exit_reason, normal_exit)``.  *exit_reason* describes why the
    supervisor stopped; *normal_exit* is True for requested clean exits
    (Q key, episode target reached, KeyboardInterrupt) or a session/service
    failure, False for a critical fault.

    Terminal ``session_failed`` ends the session. Optional evidence service
    failure only latches ``evidence_failed`` through ARMED and RUNNING; all
    started process handles remain owned by the final verified shutdown.

    ``max_running_s`` is the parent-side run budget for one RUNNING epoch —
    the same budget the policy child applies between its own polls. The
    supervisor enforces it even while a blocking predict prevents the child
    from checking: it revokes motion only after re-verifying the snapshot's
    generation under the lock, so an expired timeout can never revoke a
    newer trial. The policy process itself is supervised through
    is_alive/exitcode, not through a loop-heartbeat deadline.

    The caller should have already transitioned to ARMED before calling this
    and must handle shutdown + DISARMED transition after it returns.
    """
    from dexmani_real.config.defaults import safety
    start_time = time.monotonic()
    last_status_s = start_time
    exit_reason = "unknown"
    normal_exit = False
    loop_hz = float(safety.supervisor_hz if supervisor_hz is None else supervisor_hz)
    if not np.isfinite(loop_hz) or loop_hz <= 0:
        raise ValueError("supervisor_hz must be finite and positive")
    if not np.isfinite(status_interval_s) or status_interval_s <= 0:
        raise ValueError("status_interval_s must be finite and positive")
    process_names = [process.name for process in procs]
    if any(not name for name in process_names) or len(set(process_names)) != len(
        process_names
    ):
        raise ValueError("processes must have unique non-empty names")
    extra_heartbeats = set(heartbeat_timeouts_s) - set(process_names)
    if extra_heartbeats:
        raise ValueError(
            "heartbeat timeouts must name running processes; "
            f"unknown={sorted(extra_heartbeats)}"
        )
    timeouts = {name: float(timeout) for name, timeout in heartbeat_timeouts_s.items()}
    if any(not np.isfinite(timeout) or timeout <= 0 for timeout in timeouts.values()):
        raise ValueError("heartbeat timeouts must be finite and positive")
    if max_running_s is not None and (
        not np.isfinite(float(max_running_s)) or float(max_running_s) <= 0.0
    ):
        raise ValueError("max_running_s must be finite and positive, or None")
    max_running_ns = (
        None if max_running_s is None else int(float(max_running_s) * 1e9)
    )
    service_failure_deferred = False
    try:
        while True:
            if max_running_ns is not None:
                snapshot = read_run_state_snapshot(shared)
                if (
                    snapshot.state is SafetyState.RUNNING
                    and snapshot.started_monotonic_ns > 0
                    and time.monotonic_ns() - int(snapshot.started_monotonic_ns)
                    >= max_running_ns
                    and revoke_motion_if_generation(shared, snapshot.generation, reason=RunEndReason.TIMEOUT)
                ):
                    logger.warning(
                        "[SUPERVISOR] run budget %.1fs exceeded — motion revoked "
                        "(generation=%d); the control owner ends the trial",
                        float(max_running_s),
                        snapshot.generation,
                    )
            heartbeat_timestamps = {
                name: shared.get_heartbeat(name) for name in timeouts
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
            reason = supervisor_exit_reason(
                shared,
                procs,
                heartbeat_ages,
                timeouts,
                service_process_names=service_process_names,
            )
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
            if reason is ExitReason.EVIDENCE_FAILURE:
                shared.evidence_failed.value = True
                if not service_failure_deferred:
                    service_failure_deferred = True
                    logger.error("[EVIDENCE] optional service unavailable; control run plan continues")
            if reason is ExitReason.SERVICE_FAILURE:
                normal_exit = True
                exit_reason = "terminal session failure"
                revoke_motion(shared, reason=RunEndReason.POLICY_FAILURE)
                break
            if reason is ExitReason.EXPLICIT_QUIT:
                normal_exit = True
                exit_reason = "shutdown requested"
                revoke_motion(shared, reason=RunEndReason.QUIT)
                break

            if now - last_status_s >= status_interval_s:
                runtime_m = (now - start_time) / 60.0
                safety_state = shared.safety_state.value
                heartbeat_text = ", ".join(
                    f"{name}={heartbeat_ages[name]:.1f}s" for name in timeouts
                )
                logger.debug(
                    "runtime=%.1fmin safety=%s hb_age=(%s)",
                    runtime_m,
                    safety_state,
                    heartbeat_text,
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
    workers: Iterable[tuple["ProcessSpec", Any]],
    readiness_timeouts_s: Mapping[str, float],
    *,
    monitored_processes: Iterable[Any] | None = None,
) -> bool:
    """Wait boundedly for worker readiness while supervising startup health.

    Worker specs own readiness names, while the runtime configuration owns their
    timeouts. ``monitored_processes`` lets staged startup keep earlier workers in
    the same fault/liveness barrier.

    The caller is responsible for printing pre-wait user messages
    (e.g. "put on Quest headset") before calling this function.
    """
    worker_pairs = list(workers)
    monitored = (
        [process for _spec, process in worker_pairs]
        if monitored_processes is None
        else list(monitored_processes)
    )
    ready_names = [
        spec.ready_name
        for spec, _process in worker_pairs
        if spec.ready_name is not None
    ]
    if len(set(ready_names)) != len(ready_names):
        raise ValueError("worker readiness names must be unique")

    for name in ready_names:
        if name not in readiness_timeouts_s:
            raise ValueError(f"missing readiness timeout for {name!r}")
        timeout = float(readiness_timeouts_s[name])
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
            if not all(process.is_alive() for process in monitored):
                dead_names = [
                    process.name for process in monitored if not process.is_alive()
                ]
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

    if shared.error_state.value:
        logger.error("startup failed after readiness: error_state set")
        return False
    dead_names = [process.name for process in monitored if not process.is_alive()]
    if dead_names:
        logger.error(
            "startup failed after readiness: process(es) %s exited", dead_names
        )
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
            (time.monotonic_ns() - int(vr_data["recv_ts_ns"][0])) / 1e9
            if vr_data["recv_ts_ns"][0] > 0
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


def start_evidence_services(shared, worker_pairs, readiness_timeouts_s, *, critical_processes, started_processes) -> bool:
    """Prepare optional evidence once, retaining every started cleanup handle.

    A failed optional service disables capture, not control readiness. Earlier
    critical workers are still monitored during each bounded preparation.
    """
    available = True
    for spec, process in worker_pairs:
        try:
            process.start()
        except Exception:
            # Some start implementations can raise after acquiring a PID.
            if process.pid is not None and process not in started_processes:
                started_processes.append(process)
            available = False
            shared.evidence_failed.value = True
            logger.error("[EVIDENCE] %s failed to start", spec.name, exc_info=True)
        else:
            started_processes.append(process)
            ready = wait_subsystem_ready(
                shared, [(spec, process)], readiness_timeouts_s,
                monitored_processes=[*critical_processes, process],
            )
            if not ready:
                available = False
                shared.evidence_failed.value = True
            logger.info("evidence %s: %s", spec.name, "ready" if ready else "unavailable")
        if shared.error_state.value or shared.estop_request.value or any(
            not process.is_alive() for process in critical_processes
        ):
            raise RuntimeError("critical startup failure during evidence preparation")
    return available
