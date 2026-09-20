"""Learned-policy deployment lifecycle.

Composes the runtime primitives (``ProcessSpec`` +
``build_processes``/``start_processes``/``wait_subsystem_ready``/
``run_supervisor``/``shutdown_processes``) into the policy workflow — resolve
config -> create ``RuntimeChannels`` -> load/warm up the policy child ->
policy READY -> spawn hardware/recording workers -> readiness -> ARMED ->
supervise -> verified shutdown. The policy child owns model/CUDA and synchronous
action dispatch; supervisor heartbeats and readiness cover the existing lifecycle.

There is no VR worker. A validate-only session starts the camera only when
the explicit observation contract contains ``point_cloud`` or ``rgb``.
Physical recorded sessions attempt to start camera and RecorderIO for raw
evidence. Optional evidence startup failure permits unrecorded control;
camera payload stays out of a state-only policy observation.

"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import threading
import time
from dataclasses import replace
from typing import Any

from dexmani_real.config.experiment import ExperimentConfig
from dexmani_real.deployment.config import (
    FingertipAssemblerConfig,
    PolicyRuntimeConfig,
    RolloutRecordingConfig,
    validate_max_running_s,
    validate_num_trials,
    validate_policy_runtime_compatibility,
)
from dexmani_real.deployment.executor import policy_runner_loop
from dexmani_real.deployment.operator import build_home_planner, run_operator_control
from dexmani_real.ipc.channels import RuntimeChannels, RuntimeChannelsConfig
from dexmani_real.recording.io_worker import RecorderIOConfig, recorder_io_loop
from dexmani_real.recording.client import RECORDER_STOP_TIMEOUT_S
from dexmani_real.robot.arm_worker import arm_loop
from dexmani_real.robot.hand_worker import hand_loop
from dexmani_real.runtime.processes import (
    ProcessSpec,
    ShutdownReport,
    build_processes,
    start_processes,
    stop_processes_verified,
)
from dexmani_real.runtime.safety import SafetyState, require_transition
from dexmani_real.runtime.supervisor import (
    run_supervisor,
    shutdown_processes,
    wait_subsystem_ready,
    start_evidence_services,
)
from dexmani_real.sensor.camera.worker import CameraLoopConfig, camera_loop
from dexmani_real.sensor.pointcloud_worker import PointCloudLoopConfig, pointcloud_loop
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


_OBSERVATION_READ_MARGIN = 2
_ROLLOUT_RECORDER_FRAME_MARGIN = 4


def _observation_field_names(policy_spec: Any) -> tuple[str, ...]:
    return tuple(field.name for field in policy_spec.observation_fields)


def _requires_pointcloud(policy_spec: Any) -> bool:
    return "point_cloud" in _observation_field_names(policy_spec)


def _requires_camera(policy_spec: Any) -> bool:
    requested = _observation_field_names(policy_spec)
    return "point_cloud" in requested or "rgb" in requested


def _session_result_facts(shared, report, *, recording_enabled: bool, normal_exit: bool):
    """Compute independent evidence, cleanup and overall 0/1 outcome facts."""
    cleanup_ok = report is not None and report.shared_closed and all(
        item.exitcode is not None for item in report.exits
    )
    recording_status = (
        "failed" if shared.evidence_failed.value else
        "no evidence failure observed" if recording_enabled else "not recording"
    )
    clean_exit = bool(
        cleanup_ok and normal_exit
        and all(item.exitcode == 0 and item.escalation == "graceful" for item in report.exits)
        and not shared.error_state.value and not shared.estop_request.value
        and not shared.session_failed.value and not shared.evidence_failed.value
        and int(shared.safety_state.value) == int(SafetyState.DISARMED)
    )
    return recording_status, "clean" if cleanup_ok else "incomplete-or-failed", clean_exit


def _report_session_end(shared, report, *, recording_enabled: bool, normal_exit: bool, exit_reason: str) -> bool:
    """Report evidence and verified cleanup even on exceptional teardown."""
    recording_status, cleanup_status, clean_exit = _session_result_facts(
        shared, report, recording_enabled=recording_enabled, normal_exit=normal_exit,
    )
    print("\n── Session End ──")
    print(f"  control_reason  = {exit_reason}")
    print(f"  recording_status= {recording_status}")
    print(f"  cleanup_status  = {cleanup_status}")
    print(f"  safety={SafetyState(int(shared.safety_state.value)).name}  "
          f"supervisor_normal={normal_exit}  clean_exit={clean_exit}")
    if report is not None:
        print(f"  process_exits   = {report.exits}")
    print("──")
    return clean_exit


def _service_process_names(
    policy_spec: Any, recording_config: RolloutRecordingConfig | None
) -> set[str]:
    """Optional evidence processes: failure affects the result, not the run plan.

    Recorder is a service whenever recording is requested; camera is a service
    only when it was started solely for recording evidence (i.e. the policy
    observation contract itself does not require it). Pointcloud is never a
    service: it is always part of the policy observation contract.
    """
    names: set[str] = set()
    if recording_config is not None:
        names.add("recorder")
        if not _requires_camera(policy_spec):
            names.add("camera")
    return names


def _requires_hand_sensor(policy_spec: Any) -> bool:
    # eef_pose derives from arm qpos only and never triggers the hand sensor.
    requested = set(_observation_field_names(policy_spec))
    return bool(
        requested
        & {
            "joint_state",
            "contact_force",
            "fingertip_points",
            "tactile_force",
        }
    )


def _rollout_recorder_config(
    runtime: ExperimentConfig,
    rollout: RolloutRecordingConfig,
    worker_config: PolicyRuntimeConfig,
    max_running_s: float,
    num_trials: int,
) -> RecorderIOConfig:
    """Build the recorder capacity contract for one recorded rollout session.

    The frame capacity is DERIVED from the run-owner trial budget; recording
    does not own or re-validate the run plan. The recorder-owned
    ``provenance_*`` attributes carry the resolved experimental conditions
    (pinned artifact, effective inference steps, seed, budget) so each
    published raw episode is self-describing without any git or checksum
    provenance.
    """
    control_hz = float(runtime.policy.control_hz)
    max_frames = (
        math.ceil(float(max_running_s) * control_hz)
        + _ROLLOUT_RECORDER_FRAME_MARGIN
    )
    return RecorderIOConfig(
        data_dir=rollout.data_dir,
        max_frames=max_frames,
        control_hz=control_hz,
        min_frames=1,
        writer_queue_size=int(runtime.camera.writer_queue_size),
        provenance={
            "workflow": "policy_eval",
            "policy_selector": worker_config.experiment,
            "checkpoint_name": worker_config.artifact,
            "inference_steps": str(worker_config.inference_steps),
            "n_action_steps": str(worker_config.spec.n_action_steps),
            "seed": str(worker_config.seed),
            "max_running_s": f"{float(max_running_s):.17g}",
            "num_trials": str(int(num_trials)),
        },
    )


def _wait_for_rollout_recording(shared: RuntimeChannels, processes: list[Any]) -> bool:
    """Allow the ordinary recorder transaction to finish before shutdown.

    Motion must already be fenced. Only the recorder and its policy owner
    need to remain alive; no arm/hand acceptance is involved in finalization.
    """
    owners = [
        process for process in processes if process.name in {"policy", "recorder"}
    ]
    deadline = time.monotonic() + RECORDER_STOP_TIMEOUT_S
    while bool(shared.is_recording.value):
        if (
            len(owners) != 2
            or time.monotonic() >= deadline
            or not all(p.is_alive() for p in owners)
        ):
            return False
        time.sleep(0.01)
    return True


def _compute_policy_observation_ring_capacities(
    runtime: ExperimentConfig,
    policy_spec: Any,
    channels_config: RuntimeChannelsConfig,
) -> dict[str, int]:
    """Return deployment-owned history capacities for a policy observation grid.

    Capacity = source rate x history span, plus one predecessor control period
    and a small read margin. This is a STORAGE COVERAGE assumption so one
    causal window stays resident for sequence-addressed reads — it is NOT an
    admission deadline: temporal admission is decided per query from source
    causality against the query anchor alone.
    """
    observation_horizon = policy_spec.n_obs_steps
    observation_dt_s = policy_spec.control_dt_s
    if (
        observation_horizon is None
        or isinstance(observation_horizon, bool)
        or int(observation_horizon) <= 0
    ):
        raise ValueError("deployment observation_horizon must be positive")
    if (
        observation_dt_s is None
        or not math.isfinite(float(observation_dt_s))
        or float(observation_dt_s) <= 0.0
    ):
        raise ValueError("deployment observation_dt_s must be finite and positive")

    horizon = int(observation_horizon)
    # Observation history lives on the policy control grid. Camera FPS only
    # determines source frames inside that span, not model temporal spacing.
    coverage_s = horizon * float(observation_dt_s)
    arm_state_ring_maxlen = max(
        channels_config.arm_state_ring_maxlen,
        math.ceil(float(runtime.arm.loop_hz) * coverage_s) + _OBSERVATION_READ_MARGIN,
    )
    hand_state_ring_maxlen = max(
        channels_config.hand_state_ring_maxlen,
        math.ceil(float(runtime.hand.loop_hz) * coverage_s)
        + _OBSERVATION_READ_MARGIN,
    )
    capacities = {
        "arm_state_ring_maxlen": arm_state_ring_maxlen,
        "hand_state_ring_maxlen": hand_state_ring_maxlen,
    }
    if channels_config.camera_requested:
        capacities["camera_ring_maxlen"] = max(
            channels_config.camera_ring_maxlen,
            math.ceil(float(runtime.camera.fps) * coverage_s)
            + _OBSERVATION_READ_MARGIN,
        )
    if channels_config.pointcloud_requested:
        capacities["pointcloud_ring_maxlen"] = max(
            channels_config.pointcloud_ring_maxlen,
            math.ceil(float(runtime.camera.fps) * coverage_s)
            + _OBSERVATION_READ_MARGIN,
        )
    return capacities


def build_policy_worker_specs(
    shared: RuntimeChannels,
    runtime: ExperimentConfig,
    policy_spec: Any,
    worker_config: PolicyRuntimeConfig,
    *,
    execute: bool,
    max_running_s: float | None = None,
    num_trials: int = 1,
    recording_config: RolloutRecordingConfig | None = None,
) -> list[ProcessSpec]:
    """Build the workers required by the explicit deployment contract.

    Each spec owns its process and readiness names. ``ready_name`` exists only
    for workers with initialization that must complete before use.
    """
    if not isinstance(worker_config, PolicyRuntimeConfig):
        raise TypeError("worker_config must be a PolicyRuntimeConfig")
    max_running_s = validate_max_running_s(max_running_s)
    num_trials = validate_num_trials(num_trials)
    if recording_config is not None:
        if not isinstance(recording_config, RolloutRecordingConfig):
            raise TypeError("recording_config must be a RolloutRecordingConfig")
        if not execute:
            raise ValueError("recorded rollout requires execute=True")
        if max_running_s is None:
            raise ValueError(
                "recorded rollout requires an explicit max_running_s run budget"
            )
    pointcloud_requested = _requires_pointcloud(policy_spec)
    camera_requested = _requires_camera(policy_spec) or recording_config is not None
    fingertip_config = (
        FingertipAssemblerConfig.from_runtime(runtime)
        if "fingertip_points" in _observation_field_names(policy_spec)
        else None
    )
    pointcloud_config = (
        PointCloudLoopConfig.from_runtime(
            runtime,
            num_points=next(
                field.shape[0]
                for field in policy_spec.observation_fields
                if field.name == "point_cloud"
            ),
        )
        if pointcloud_requested
        else None
    )
    specs: list[ProcessSpec] = [
        ProcessSpec(
            "arm",
            arm_loop,
            (shared, runtime.arm),
            ready_name="arm",
        ),
    ]
    if camera_requested:
        specs.append(
            ProcessSpec(
                "camera",
                camera_loop,
                (
                    shared,
                    CameraLoopConfig.from_runtime(runtime),
                    _requires_camera(policy_spec),
                ),
                ready_name="camera",
            )
        )
    if pointcloud_requested:
        specs.append(
            ProcessSpec(
                "pointcloud",
                pointcloud_loop,
                (
                    shared,
                    pointcloud_config,
                ),
                ready_name="pointcloud",
            )
        )
    if recording_config is not None:
        assert max_running_s is not None
        specs.append(
            ProcessSpec(
                "recorder",
                recorder_io_loop,
                (
                    shared,
                    _rollout_recorder_config(
                        runtime,
                        recording_config,
                        worker_config,
                        max_running_s,
                        num_trials,
                    ),
                ),
                ready_name="recorder",
            )
        )
    specs.append(
        ProcessSpec(
            "policy",
            policy_runner_loop,
            (
                shared,
                runtime,
                worker_config,
                execute,
                max_running_s,
                num_trials,
                recording_config,
                fingertip_config,
            ),
            ready_name="policy",
        )
    )
    if policy_spec.requires_hand or _requires_hand_sensor(policy_spec):
        specs.append(
            ProcessSpec(
                "hand",
                hand_loop,
                (
                    shared,
                    runtime.hand,
                    float(runtime.policy.hand_disconnect_timeout_s),
                ),
                ready_name="hand",
            )
        )
    return specs


def run_policy_deployment(
    runtime: ExperimentConfig,
    policy_spec: Any,
    worker_config: PolicyRuntimeConfig,
    execute: bool,
    *,
    prefix: str | None = None,
    max_running_s: float | None = None,
    num_trials: int = 1,
    recording_config: RolloutRecordingConfig | None = None,
) -> int:
    """Run a persistent multi-trial policy deployment lifecycle and return its exit code.

    ``execute=False`` validates candidates without publication;
    ``execute=True`` enables coupled arm/hand publication. The policy runner
    must load and warm up before any hardware process is started. The runtime
    follows ``DISARMED -> hardware readiness -> ARMED -> supervision ->
    verified shutdown``. Trials and the per-trial budget are run-owner
    configuration; recording is evidence and never gates the run plan.
    """
    if not isinstance(runtime, ExperimentConfig):
        raise TypeError("runtime must be an ExperimentConfig")
    if not isinstance(execute, bool):
        raise TypeError("execute must be a boolean")
    if recording_config is not None:
        if not isinstance(recording_config, RolloutRecordingConfig):
            raise TypeError("recording_config must be a RolloutRecordingConfig")
        if not execute:
            raise ValueError("recorded rollout requires execute=True")
    validate_policy_runtime_compatibility(policy_spec, runtime)
    if not isinstance(worker_config, PolicyRuntimeConfig):
        raise TypeError("worker_config must be a PolicyRuntimeConfig")
    if worker_config.spec is not policy_spec:
        raise ValueError("worker PolicySpec must be the validated lifecycle PolicySpec")
    max_running_s = validate_max_running_s(max_running_s)
    num_trials = validate_num_trials(num_trials)
    if recording_config is not None and max_running_s is None:
        raise ValueError(
            "recorded rollout requires an explicit max_running_s run budget"
        )
    logger.debug(
        "policy deployment: experiment=%s device=%s seed=%s execute=%s",
        worker_config.experiment,
            worker_config.device,
        worker_config.seed,
        execute,
    )

    ctx = mp.get_context("spawn")
    pointcloud_requested = _requires_pointcloud(policy_spec)
    camera_requested = _requires_camera(policy_spec) or recording_config is not None
    service_process_names = _service_process_names(policy_spec, recording_config)
    channel_config = RuntimeChannelsConfig.from_runtime(
        runtime,
        pointcloud_num_points=(
            next(
                field.shape[0]
                for field in policy_spec.observation_fields
                if field.name == "point_cloud"
            )
            if pointcloud_requested
            else runtime.pointcloud.num_points
        ),
        camera_requested=camera_requested,
        pointcloud_requested=pointcloud_requested,
    )
    shared = RuntimeChannels.create(
        prefix=prefix or f"dexmani_policy_{os.getpid()}",
        config=replace(
            channel_config,
            **_compute_policy_observation_ring_capacities(
                runtime, policy_spec, channel_config
            ),
        ),
        mp_context=ctx,
    )
    specs: list[ProcessSpec] = []
    procs: list[Any] = []
    started_procs: list[Any] = []
    shutdown_report: ShutdownReport | None = None
    operator_thread: threading.Thread | None = None
    operator_stop: threading.Event | None = None
    summary_emitted = False
    normal_exit = False
    exit_reason = "startup failed"
    try:
        specs = build_policy_worker_specs(
            shared,
            runtime,
            policy_spec,
            worker_config,
            execute=execute,
            max_running_s=max_running_s,
            num_trials=num_trials,
            recording_config=recording_config,
        )
        procs = build_processes(ctx, specs)
        require_transition(shared, SafetyState.DISARMED)

        timeouts = runtime.safety.readiness_timeouts_s
        spec_processes = list(zip(specs, procs))
        policy_pairs = [
            (spec, process)
            for spec, process in spec_processes
            if spec.ready_name == "policy"
        ]
        if len(policy_pairs) != 1:
            raise RuntimeError("deployment requires exactly one policy runner")
        _policy_spec, policy_process = policy_pairs[0]
        start_processes([policy_process])
        started_procs.append(policy_process)
        if not wait_subsystem_ready(
            shared,
            policy_pairs,
            timeouts,
            monitored_processes=started_procs,
        ):
            shared.error_state.value = True
            require_transition(shared, SafetyState.FAULT)
            shutdown_report = shutdown_processes(
                shared,
                started_procs,
                graceful_timeout_s=float(runtime.safety.shutdown_timeout_s),
            )
            return 1
        print("  policy: ready", flush=True)

        remaining_pairs = [
            (spec, process)
            for spec, process in spec_processes
            if process is not policy_process
        ]
        critical_pairs = [
            (spec, process)
            for spec, process in remaining_pairs
            if spec.name not in service_process_names
        ]
        service_pairs = [
            (spec, process)
            for spec, process in remaining_pairs
            if spec.name in service_process_names
        ]
        # policy is already started/ready; it is critical like every other
        # non-service process for the purposes of the service readiness wait.
        critical_procs = [policy_process] + [
            process for _spec, process in critical_pairs
        ]
        # Register each successful start immediately.  If a later Process.start()
        # raises, verified shutdown must still stop every earlier child before IPC
        # is closed or unlinked. Critical workers start (and become ready)
        # before any experiment service is started.
        for _spec, process in critical_pairs:
            start_processes([process])
            started_procs.append(process)
        if not wait_subsystem_ready(
            shared,
            critical_pairs,
            timeouts,
            monitored_processes=started_procs,
        ):
            shared.error_state.value = True
            require_transition(shared, SafetyState.FAULT)
            shutdown_report = shutdown_processes(
                shared,
                started_procs,
                graceful_timeout_s=float(runtime.safety.shutdown_timeout_s),
                service_process_names=service_process_names,
            )
            return 1

        for spec, _process in critical_pairs:
            if spec.ready_name is not None:
                print(f"  {spec.ready_name}: ready", flush=True)

        evidence_ready = start_evidence_services(
            shared, service_pairs, timeouts, critical_processes=critical_procs,
            started_processes=started_procs,
        )

        require_transition(shared, SafetyState.ARMED)
        print(
            f"\nControl subsystems ready — safety=ARMED({int(SafetyState.ARMED)})",
            flush=True,
        )
        if not evidence_ready:
            print("  Evidence unavailable; trials can run without recording", flush=True)
        home_planner = build_home_planner(runtime) if execute else None
        home_status = "return hand + arm home before B" if home_planner else "disabled"
        print(
            "  [B] begin   [S] stop/save   [Q] quit   [ESC] e-stop   "
            f"[H] {home_status}",
            flush=True,
        )

        operator_stop = threading.Event()
        operator_thread = threading.Thread(
            target=run_operator_control,
            args=(shared, runtime, home_planner),
            kwargs={
                "stop_event": operator_stop,
                "execute": execute,
            },
            name="policy-operator",
            daemon=True,
        )
        operator_thread.start()

        # The policy child is supervised through is_alive/exitcode plus the
        # parent-side run budget below — a normal blocking inference must
        # never trip a loop-heartbeat deadline. Every started I/O worker keeps
        # its existing heartbeat supervision (the timeout lookup below only
        # covers processes that actually run).
        heartbeat_names = {"arm", "hand", "camera", "pointcloud", "recorder"}
        heartbeat_timeouts = {
            process.name: float(runtime.safety.heartbeat_timeouts[process.name])
            for process in started_procs
            if process.name in heartbeat_names
        }
        exit_reason, normal_exit = run_supervisor(
            shared,
            started_procs,
            heartbeat_timeouts_s=heartbeat_timeouts,
            supervisor_hz=float(runtime.safety.supervisor_hz),
            service_process_names=service_process_names,
            max_running_s=max_running_s,
        )

        # Stop user input before finalization. E-stop remains latched and the
        # software fence is applied before any disk wait.
        if operator_stop is not None:
            operator_stop.set()
        require_transition(
            shared, SafetyState.ARMED if normal_exit else SafetyState.FAULT
        )
        if normal_exit:
            shared.quit_requested.value = True
        if recording_config is not None and not _wait_for_rollout_recording(
            shared, started_procs
        ):
            logger.error("rollout recording did not finalize before shutdown")
            shared.evidence_failed.value = True

        if operator_stop is not None:
            operator_stop.set()
        if operator_thread is not None:
            shared.is_running.value = False
            operator_thread.join(timeout=float(runtime.safety.shutdown_timeout_s))
            if operator_thread.is_alive():
                raise RuntimeError(
                    "operator control did not stop; RuntimeChannels cannot be closed"
                )
            operator_thread = None

        shutdown_report = shutdown_processes(
            shared,
            started_procs,
            graceful_timeout_s=float(runtime.safety.shutdown_timeout_s),
            disarm_if_clean=normal_exit,
            service_process_names=service_process_names,
        )
        clean_exit = _report_session_end(
            shared, shutdown_report, recording_enabled=recording_config is not None,
            normal_exit=normal_exit, exit_reason=exit_reason,
        )
        summary_emitted = True
        return 0 if clean_exit else 1

    except Exception:
        logger.error("policy deployment failed", exc_info=True)
        shared.error_state.value = True
        require_transition(shared, SafetyState.FAULT)
        return 1
    finally:
        try:
            if operator_stop is not None:
                operator_stop.set()
            operator_alive = False
            if operator_thread is not None:
                operator_thread.join(timeout=float(runtime.safety.shutdown_timeout_s))
                operator_alive = operator_thread.is_alive()
                if operator_alive:
                    logger.critical(
                        "operator thread remains alive; leaving RuntimeChannels linked"
                    )
            if shutdown_report is None:
                if started_procs:
                    try:
                        if operator_alive:
                            stop_processes_verified(
                                shared,
                                started_procs,
                                graceful_timeout_s=float(runtime.safety.shutdown_timeout_s),
                            )
                        else:
                            shutdown_report = shutdown_processes(
                                shared,
                                started_procs,
                                graceful_timeout_s=float(runtime.safety.shutdown_timeout_s),
                                service_process_names=service_process_names,
                            )
                    except RuntimeError:
                        logger.critical(
                            "child process remains alive; leaving RuntimeChannels linked",
                            exc_info=True,
                        )
                        raise
                elif not operator_alive:
                    try:
                        closed = bool(shared.close())
                        shutdown_report = ShutdownReport((), shared_closed=closed)
                        if not closed:
                            logger.error("RuntimeChannels cleanup was incomplete")
                    except Exception:
                        logger.warning("RuntimeChannels cleanup failed", exc_info=True)
        finally:
            if not summary_emitted:
                _report_session_end(
                    shared, shutdown_report, recording_enabled=recording_config is not None,
                    normal_exit=False, exit_reason=exit_reason,
                )
