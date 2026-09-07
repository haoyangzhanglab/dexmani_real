"""Learned-policy deployment lifecycle.

Composes the runtime primitives (``ProcessSpec`` +
``build_processes``/``start_processes``/``wait_subsystem_ready``/
``run_supervisor``/``shutdown_processes``) into the policy workflow — resolve
config -> create ``RuntimeChannels`` -> start the Policy-owned inference
process -> spawn arm (+ optional hand and RGB-D / point-cloud workers) ->
executor -> readiness -> ARMED -> supervise -> verified shutdown. There is
no second health mechanism: supervisor heartbeats cover the policy executor,
actuator, and inference workers, while readiness covers asynchronous startup
only.

There is no VR worker. Ordinary deployment starts the camera only when the
explicit observation contract contains ``point_cloud`` or ``rgb``. Formal
evaluation additionally starts camera and RecorderIO for audit evidence, while
keeping camera payload out of a state-only policy observation.

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
    FIXED_POLICY_RUNTIME_TARGET,
    FingertipAssemblerConfig,
    InferenceWorkerConfig,
    PolicyDeploymentConfig,
    validate_max_running_s,
    validate_policy_runtime_compatibility,
)
from dexmani_real.deployment.evaluation import (
    EVALUATION_MAX_FRAMES_STOP_REASON,
    PolicyEvaluationConfig,
)
from dexmani_real.deployment.executor import policy_executor_loop
from dexmani_real.deployment.inference.worker import inference_loop
from dexmani_real.deployment.operator import build_home_planner, run_operator_control
from dexmani_real.ipc.channels import RuntimeChannels, RuntimeChannelsConfig
from dexmani_real.recording.io_worker import RecorderIOConfig, recorder_io_loop
from dexmani_real.recording.client import RecorderCommand, RecorderPhase
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
)
from dexmani_real.sensor.camera.worker import CameraLoopConfig, camera_loop
from dexmani_real.sensor.pointcloud_worker import PointCloudLoopConfig, pointcloud_loop
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


_OBSERVATION_READ_MARGIN = 2
_EVALUATION_RECORDER_FRAME_MARGIN = 4
_EVALUATION_RECORDER_STOP_ACK_TIMEOUT_S = 1.0
_EVALUATION_RECORDER_STOP_ACK_POLL_S = 0.01


def _observation_field_names(policy_spec: Any) -> tuple[str, ...]:
    return tuple(field.name for field in policy_spec.observation_fields)


def _requires_pointcloud(policy_spec: Any) -> bool:
    return "point_cloud" in _observation_field_names(policy_spec)


def _requires_camera(policy_spec: Any) -> bool:
    requested = _observation_field_names(policy_spec)
    return "point_cloud" in requested or "rgb" in requested


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


def _evaluation_recorder_config(
    runtime: ExperimentConfig,
    evaluation: PolicyEvaluationConfig,
) -> RecorderIOConfig:
    """Build the recorder-only capacity contract for one formal eval session."""
    control_hz = float(runtime.policy.control_hz)
    max_frames = (
        math.ceil(float(evaluation.max_running_s) * control_hz)
        + _EVALUATION_RECORDER_FRAME_MARGIN
    )
    return RecorderIOConfig(
        data_dir=evaluation.data_dir,
        max_frames=max_frames,
        control_hz=control_hz,
        min_frames=1,
        writer_queue_size=int(runtime.camera.writer_queue_size),
        provenance=evaluation.provenance,
        max_frames_stop_reason=EVALUATION_MAX_FRAMES_STOP_REASON,
    )


def _evaluation_recorder_phase(shared: RuntimeChannels) -> RecorderPhase | None:
    """Read the latest recorder phase without assigning it controller meaning."""
    result = shared.record_status_ring.read_latest()
    if result is None:
        return None
    try:
        return RecorderPhase(int(result[0][0]["phase"]))
    except (TypeError, ValueError):
        return None


def _evaluation_recorder_stop_is_queued(shared: RuntimeChannels) -> bool:
    """Return whether RecorderIO has a STOP/finalize decision for this session."""
    try:
        control_result = shared.record_control_ring.read_latest()
        if control_result is not None:
            command = int(control_result[0][0]["command"])
            if command == int(RecorderCommand.STOP):
                return True
        phase = _evaluation_recorder_phase(shared)
    except Exception:
        logger.warning("could not inspect formal-eval recorder stop state", exc_info=True)
        return False
    return phase in {
        RecorderPhase.FINALIZING,
        RecorderPhase.COMPLETED,
        RecorderPhase.ERROR,
    }


def _wait_for_evaluation_recorder_stop(shared: RuntimeChannels) -> bool:
    """Give executor a bounded chance to queue STOP before process shutdown.

    The wait covers only the small cross-process handoff from executor to
    RecorderIO.  It never waits for HDF5/video finalization; RecorderIO owns
    that bounded transaction after the STOP control has been published.
    """
    phase = _evaluation_recorder_phase(shared)
    needs_ack = bool(shared.is_recording.value) or phase is RecorderPhase.RECORDING
    if not needs_ack:
        return True
    deadline = time.monotonic() + _EVALUATION_RECORDER_STOP_ACK_TIMEOUT_S
    while time.monotonic() < deadline:
        if _evaluation_recorder_stop_is_queued(shared):
            return True
        time.sleep(_EVALUATION_RECORDER_STOP_ACK_POLL_S)
    return _evaluation_recorder_stop_is_queued(shared)


def _compute_policy_observation_ring_capacities(
    runtime: ExperimentConfig,
    policy_spec: Any,
    channels_config: RuntimeChannelsConfig,
) -> dict[str, int]:
    """Return deployment-owned history capacities for a policy observation grid."""
    observation_horizon = policy_spec.n_obs_steps
    observation_dt_s = policy_spec.control_dt_s
    max_input_age_s = runtime.policy.max_input_age_s
    max_observation_skew_s = runtime.policy.max_observation_skew_s
    max_grid_lag_s = runtime.policy.max_grid_lag_s
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
    if not math.isfinite(float(max_input_age_s)) or float(max_input_age_s) <= 0.0:
        raise ValueError("max_input_age_s must be finite and positive")
    if (
        not math.isfinite(float(max_observation_skew_s))
        or float(max_observation_skew_s) < 0.0
    ):
        raise ValueError("max_observation_skew_s must be finite and non-negative")
    if not math.isfinite(float(max_grid_lag_s)) or float(max_grid_lag_s) < 0.0:
        raise ValueError("max_grid_lag_s must be finite and non-negative")

    horizon = int(observation_horizon)
    # Observation history lives on the policy control grid. Camera FPS only
    # determines source frames inside that span, not model temporal spacing.
    history_span_s = (horizon - 1) * float(observation_dt_s)
    visual_span_s = history_span_s + float(max_grid_lag_s) + float(max_input_age_s)
    state_span_s = (
        visual_span_s + float(max_observation_skew_s)
        if channels_config.camera_requested
        else history_span_s + float(max_input_age_s) + float(max_observation_skew_s)
    )
    arm_state_ring_maxlen = max(
        channels_config.arm_state_ring_maxlen,
        math.ceil(float(runtime.arm.loop_hz) * state_span_s) + _OBSERVATION_READ_MARGIN,
    )
    hand_state_ring_maxlen = max(
        channels_config.hand_state_ring_maxlen,
        math.ceil(float(runtime.hand.loop_hz) * state_span_s)
        + _OBSERVATION_READ_MARGIN,
    )
    capacities = {
        "arm_state_ring_maxlen": arm_state_ring_maxlen,
        "hand_state_ring_maxlen": hand_state_ring_maxlen,
        "hand_tactile_ring_maxlen": hand_state_ring_maxlen,
    }
    if channels_config.camera_requested:
        capacities["camera_ring_maxlen"] = max(
            channels_config.camera_ring_maxlen,
            math.ceil(float(runtime.camera.fps) * visual_span_s)
            + _OBSERVATION_READ_MARGIN,
        )
    if channels_config.pointcloud_requested:
        capacities["pointcloud_ring_maxlen"] = max(
            channels_config.pointcloud_ring_maxlen,
            math.ceil(float(runtime.camera.fps) * visual_span_s)
            + _OBSERVATION_READ_MARGIN,
        )
    return capacities


def build_policy_worker_specs(
    shared: RuntimeChannels,
    runtime: ExperimentConfig,
    policy_spec: Any,
    worker_config: InferenceWorkerConfig,
    *,
    execute: bool,
    deployment_config: PolicyDeploymentConfig | None = None,
    max_running_s: float | None = None,
    evaluation_config: PolicyEvaluationConfig | None = None,
) -> list[ProcessSpec]:
    """Build the workers required by the explicit deployment contract.

    Each spec owns its process and readiness names. ``ready_name`` exists only
    for workers whose asynchronous initialization can fail; executor liveness
    is enough.
    """
    if not isinstance(worker_config, InferenceWorkerConfig):
        raise TypeError("worker_config must be an InferenceWorkerConfig")
    deployment = deployment_config or PolicyDeploymentConfig()
    if not isinstance(deployment, PolicyDeploymentConfig):
        raise TypeError("deployment_config must be a PolicyDeploymentConfig")
    max_running_s = validate_max_running_s(max_running_s)
    if evaluation_config is not None:
        if not isinstance(evaluation_config, PolicyEvaluationConfig):
            raise TypeError("evaluation_config must be a PolicyEvaluationConfig")
        if not execute:
            raise ValueError("formal policy evaluation requires execute=True")
        if max_running_s != evaluation_config.max_running_s:
            raise ValueError(
                "formal evaluation max_running_s must match its evaluation contract"
            )
    pointcloud_requested = _requires_pointcloud(policy_spec)
    camera_requested = _requires_camera(policy_spec) or evaluation_config is not None
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
                (shared, CameraLoopConfig.from_runtime(runtime)),
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
    if evaluation_config is not None:
        specs.append(
            ProcessSpec(
                "recorder",
                recorder_io_loop,
                (shared, _evaluation_recorder_config(runtime, evaluation_config)),
                ready_name="recorder",
            )
        )
    specs.extend(
        [
            ProcessSpec(
                "inference",
                inference_loop,
                (shared, runtime.policy, worker_config, deployment, fingertip_config),
                ready_name="inference",
            ),
            ProcessSpec(
                "policy",
                policy_executor_loop,
                (
                    shared,
                    runtime,
                    policy_spec,
                    deployment,
                    execute,
                    max_running_s,
                    evaluation_config,
                ),
            ),
        ]
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
    worker_config: InferenceWorkerConfig,
    execute: bool,
    *,
    deployment_config: PolicyDeploymentConfig | None = None,
    prefix: str | None = None,
    max_running_s: float | None = None,
    evaluation_config: PolicyEvaluationConfig | None = None,
) -> int:
    """Run a multi-episode policy deployment lifecycle and return its exit code.

    ``execute=False`` validates candidates without publication;
    ``execute=True`` enables coupled arm/hand publication. The inference worker
    must load successfully before any hardware process is started. The runtime then
    follows ``DISARMED -> hardware readiness -> ARMED -> supervision ->
    verified shutdown``.
    """
    if not isinstance(runtime, ExperimentConfig):
        raise TypeError("runtime must be an ExperimentConfig")
    if not isinstance(execute, bool):
        raise TypeError("execute must be a boolean")
    if evaluation_config is not None:
        if not isinstance(evaluation_config, PolicyEvaluationConfig):
            raise TypeError("evaluation_config must be a PolicyEvaluationConfig")
        if not execute:
            raise ValueError("formal policy evaluation requires execute=True")
    validate_policy_runtime_compatibility(policy_spec, runtime)
    if not isinstance(worker_config, InferenceWorkerConfig):
        raise TypeError("worker_config must be an InferenceWorkerConfig")
    if worker_config.spec is not policy_spec:
        raise ValueError("worker PolicySpec must be the validated lifecycle PolicySpec")
    deployment = deployment_config or PolicyDeploymentConfig()
    if not isinstance(deployment, PolicyDeploymentConfig):
        raise TypeError("deployment_config must be a PolicyDeploymentConfig")
    max_running_s = validate_max_running_s(max_running_s)
    if evaluation_config is not None:
        if (
            max_running_s is not None
            and max_running_s != evaluation_config.max_running_s
        ):
            raise ValueError(
                "formal evaluation max_running_s must match its evaluation contract"
            )
        max_running_s = evaluation_config.max_running_s
    logger.info(
        "policy deployment: experiment=%s runtime=%s device=%s seed=0 execute=%s mode=%s",
        worker_config.experiment,
        FIXED_POLICY_RUNTIME_TARGET,
        worker_config.device,
        execute,
        deployment.inference_mode,
    )

    ctx = mp.get_context("spawn")
    pointcloud_requested = _requires_pointcloud(policy_spec)
    camera_requested = _requires_camera(policy_spec) or evaluation_config is not None
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
    try:
        specs = build_policy_worker_specs(
            shared,
            runtime,
            policy_spec,
            worker_config,
            execute=execute,
            deployment_config=deployment,
            max_running_s=max_running_s,
            evaluation_config=evaluation_config,
        )
        procs = build_processes(ctx, specs)
        require_transition(shared, SafetyState.DISARMED)

        timeouts = runtime.safety.readiness_timeouts_s
        spec_processes = list(zip(specs, procs))
        inference_pairs = [
            (spec, process)
            for spec, process in spec_processes
            if spec.ready_name == "inference"
        ]
        if len(inference_pairs) != 1:
            raise RuntimeError("deployment requires exactly one inference worker")
        _inference_spec, inference_process = inference_pairs[0]
        start_processes([inference_process])
        started_procs.append(inference_process)
        if not wait_subsystem_ready(
            shared,
            inference_pairs,
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
        print("  inference: ready", flush=True)

        remaining_pairs = [
            (spec, process)
            for spec, process in spec_processes
            if process is not inference_process
        ]
        remaining_procs = [process for _spec, process in remaining_pairs]
        # Register each successful start immediately.  If a later Process.start()
        # raises, verified shutdown must still stop every earlier child before IPC
        # is closed or unlinked.
        for process in remaining_procs:
            start_processes([process])
            started_procs.append(process)
        if not wait_subsystem_ready(
            shared,
            remaining_pairs,
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

        for spec, _process in remaining_pairs:
            if spec.ready_name is not None:
                print(f"  {spec.ready_name}: ready", flush=True)

        require_transition(shared, SafetyState.ARMED)
        print(
            f"\nAll subsystems ready — safety=ARMED({int(SafetyState.ARMED)})",
            flush=True,
        )
        home_planner = build_home_planner(runtime) if execute else None
        home_status = "return hand + arm home before B" if home_planner else "disabled"
        if evaluation_config is not None:
            print(
                "  [B] begin eval   [S] success   [C] failure   [D] invalid   "
                f"[Q] quit   [ESC] e-stop   [H] {home_status}",
                flush=True,
            )
        else:
            print(
                "  [B] start run   [S] stop run   [Q] quit   [ESC] e-stop   "
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
                "evaluation": evaluation_config is not None,
            },
            name="policy-operator",
            daemon=True,
        )
        operator_thread.start()

        heartbeat_names = {"arm", "hand", "inference", "policy"}
        if evaluation_config is not None:
            heartbeat_names.update({"camera", "pointcloud", "recorder"})
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
        )

        if evaluation_config is not None and not _wait_for_evaluation_recorder_stop(
            shared
        ):
            # Motion has already been fenced by the supervisor's fault/stop
            # path.  Do not block on finalization, but make an unacknowledged
            # formal recorder transaction visible as a failed session.
            logger.error(
                "formal eval recorder STOP was not queued before shutdown"
            )
            shared.error_state.value = True

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
        )
        worker_exit_clean = all(
            item.exitcode == 0 and item.escalation == "graceful"
            for item in shutdown_report.exits
        )
        clean_exit = (
            normal_exit
            and worker_exit_clean
            and shutdown_report.shared_closed
            and not bool(shared.error_state.value)
            and not bool(shared.estop_request.value)
            and int(shared.safety_state.value) == int(SafetyState.DISARMED)
        )
        safety_name = SafetyState(int(shared.safety_state.value)).name
        print(f"\n── Session End ──")
        print(
            f"  exit_reason={exit_reason}  safety={safety_name}  "
            f"supervisor_normal={normal_exit}  clean={clean_exit}"
        )
        print("──")
        return 0 if clean_exit else 1

    except Exception:
        logger.error("policy deployment failed", exc_info=True)
        shared.error_state.value = True
        require_transition(shared, SafetyState.FAULT)
        return 1
    finally:
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
                        shutdown_processes(
                            shared,
                            started_procs,
                            graceful_timeout_s=float(runtime.safety.shutdown_timeout_s),
                        )
                except RuntimeError:
                    logger.critical(
                        "child process remains alive; leaving RuntimeChannels linked",
                        exc_info=True,
                    )
                    raise
            elif not operator_alive:
                try:
                    if not shared.close():
                        logger.error("RuntimeChannels cleanup was incomplete")
                except Exception:
                    logger.warning("RuntimeChannels cleanup failed", exc_info=True)
