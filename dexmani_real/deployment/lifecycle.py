"""Learned-policy deployment lifecycle.

Composes the runtime primitives (``WorkerSpec`` +
``build_processes``/``start_processes``/``wait_subsystem_ready``/
``run_supervisor``/``shutdown_processes``) into the policy workflow — resolve
config -> create ``RuntimeChannels`` -> start the Policy-owned inference
process -> spawn arm (+ optional hand and RGB-D / point-cloud workers) ->
executor -> readiness -> ARMED -> supervise -> verified shutdown. There is
no second health mechanism: supervisor heartbeats cover the policy executor,
actuator, and inference workers, while readiness covers asynchronous startup
only.

There is no VR worker or recorder. The camera worker is included whenever the
explicit observation contract contains ``point_cloud`` or ``rgb``; the
point-cloud worker is included only for ``point_cloud``.

"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import threading
from dataclasses import replace
from typing import Any

from dexmani_real.config.runtime import ResolvedRuntimeConfig
from dexmani_real.deployment.config import (
    FIXED_POLICY_RUNTIME_TARGET,
    FingertipAssemblerConfig,
    PolicyDeploymentConfig,
    PolicyWorkerConfig,
    policy_observation_fields,
    validate_max_running_s,
    validate_policy_runtime_compatibility,
)
from dexmani_real.deployment.executor import policy_executor_loop
from dexmani_real.deployment.operator import build_home_planner, run_operator_control
from dexmani_real.deployment.worker import inference_loop
from dexmani_real.ipc.channels import RuntimeChannels, RuntimeChannelsConfig
from dexmani_real.robot.arm_worker import arm_loop
from dexmani_real.robot.hand_worker import hand_loop
from dexmani_real.runtime.safety import SafetyState, require_transition, transition
from dexmani_real.runtime.supervisor import (
    run_supervisor,
    shutdown_processes,
    wait_subsystem_ready,
)
from dexmani_real.runtime.workers import (
    ShutdownReport,
    WorkerSpec,
    build_processes,
    start_processes,
    stop_processes_verified,
)
from dexmani_real.sensor.camera_worker import CameraLoopConfig, camera_loop
from dexmani_real.sensor.pointcloud_worker import PointCloudLoopConfig, pointcloud_loop
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


_OBSERVATION_READ_MARGIN = 2


def _observation_field_names(policy_spec: Any) -> tuple[str, ...]:
    return tuple(field.name for field in policy_observation_fields(policy_spec))


def _requires_pointcloud(policy_spec: Any) -> bool:
    return "point_cloud" in _observation_field_names(policy_spec)


def _requires_camera(policy_spec: Any) -> bool:
    requested = _observation_field_names(policy_spec)
    return "point_cloud" in requested or "rgb" in requested


def _requires_hand_sensor(policy_spec: Any) -> bool:
    requested = set(_observation_field_names(policy_spec))
    return bool(requested & {"joint_state", "contact_force", "fingertip_points"})


def _compute_policy_observation_ring_capacities(
    runtime: ResolvedRuntimeConfig,
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
        else history_span_s
        + float(max_input_age_s)
        + float(max_observation_skew_s)
    )
    arm_state_ring_maxlen = max(
        channels_config.arm_state_ring_maxlen,
        math.ceil(float(runtime.arm.loop_hz) * state_span_s)
        + _OBSERVATION_READ_MARGIN,
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
    runtime: ResolvedRuntimeConfig,
    policy_spec: Any,
    worker_config: PolicyWorkerConfig,
    *,
    execute: bool,
    deployment_config: PolicyDeploymentConfig | None = None,
    max_running_s: float | None = None,
) -> list[WorkerSpec]:
    """Build the workers required by the explicit deployment contract.

    Readiness order is the single source of truth for build order and for the
    readiness names derived from it. ``ready_name`` exists only for workers
    whose asynchronous initialization can fail; executor liveness is enough.
    """
    validate_policy_runtime_compatibility(policy_spec, runtime)
    if not isinstance(worker_config, PolicyWorkerConfig):
        raise TypeError("worker_config must be a PolicyWorkerConfig")
    if worker_config.spec is not policy_spec:
        raise ValueError("worker PolicySpec must be the validated lifecycle PolicySpec")
    deployment = deployment_config or PolicyDeploymentConfig()
    if not isinstance(deployment, PolicyDeploymentConfig):
        raise TypeError("deployment_config must be a PolicyDeploymentConfig")
    max_running_s = validate_max_running_s(max_running_s)
    pointcloud_requested = _requires_pointcloud(policy_spec)
    camera_requested = _requires_camera(policy_spec)
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
                for field in policy_observation_fields(policy_spec)
                if field.name == "point_cloud"
            ),
        )
        if pointcloud_requested
        else None
    )
    specs: list[WorkerSpec] = [
        WorkerSpec(
            "arm",
            arm_loop,
            (shared, runtime.arm),
            ready_name="arm",
        ),
    ]
    if camera_requested:
        specs.append(
            WorkerSpec(
                "camera",
                camera_loop,
                (shared, CameraLoopConfig.from_runtime(runtime)),
                ready_name="camera",
            )
        )
    if pointcloud_requested:
        specs.append(
            WorkerSpec(
                "pointcloud",
                pointcloud_loop,
                (
                    shared,
                    pointcloud_config,
                ),
                ready_name="pointcloud",
            )
        )
    specs.extend(
        [
            WorkerSpec(
                "inference",
                inference_loop,
                (shared, runtime.policy, worker_config, deployment, fingertip_config),
                ready_name="inference",
            ),
            WorkerSpec(
                "policy",
                policy_executor_loop,
                (
                    shared,
                    runtime,
                    policy_spec,
                    deployment,
                    execute,
                    max_running_s,
                ),
            ),
        ]
    )
    if policy_spec.requires_hand or _requires_hand_sensor(policy_spec):
        specs.append(
            WorkerSpec(
                "hand",
                hand_loop,
                (shared, runtime.hand),
                ready_name="hand",
            )
        )
    return specs


def run_policy_deployment(
    runtime: ResolvedRuntimeConfig,
    policy_spec: Any,
    worker_config: PolicyWorkerConfig,
    execute: bool,
    *,
    deployment_config: PolicyDeploymentConfig | None = None,
    prefix: str | None = None,
    max_running_s: float | None = None,
) -> int:
    """Run a multi-episode policy deployment lifecycle and return its exit code.

    ``execute=False`` validates candidates without publication;
    ``execute=True`` enables coupled arm/hand publication. The inference worker
    must load successfully before any hardware process is started. The runtime then
    follows ``DISARMED -> hardware readiness -> ARMED -> supervision ->
    verified shutdown``.
    """
    if not isinstance(runtime, ResolvedRuntimeConfig):
        raise TypeError("runtime must be a ResolvedRuntimeConfig")
    if not isinstance(execute, bool):
        raise TypeError("execute must be a boolean")
    validate_policy_runtime_compatibility(policy_spec, runtime)
    if not isinstance(worker_config, PolicyWorkerConfig):
        raise TypeError("worker_config must be a PolicyWorkerConfig")
    if worker_config.spec is not policy_spec:
        raise ValueError("worker PolicySpec must be the validated lifecycle PolicySpec")
    deployment = deployment_config or PolicyDeploymentConfig()
    if not isinstance(deployment, PolicyDeploymentConfig):
        raise TypeError("deployment_config must be a PolicyDeploymentConfig")
    max_running_s = validate_max_running_s(max_running_s)
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
    camera_requested = _requires_camera(policy_spec)
    channel_config = RuntimeChannelsConfig.from_runtime(
        runtime,
        pointcloud_num_points=(
            next(
                field.shape[0]
                for field in policy_observation_fields(policy_spec)
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
    specs: list[WorkerSpec] = []
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
        inference_spec, inference_process = inference_pairs[0]
        if inference_spec.ready_name is None:
            raise RuntimeError("inference worker requires a readiness name")
        start_processes([inference_process])
        started_procs.append(inference_process)
        inference_ready = [
            (
                inference_spec.ready_name,
                float(timeouts[inference_spec.ready_name]),
            )
        ]
        if not wait_subsystem_ready(shared, inference_ready, started_procs):
            shared.error_state.value = True
            require_transition(shared, SafetyState.FAULT)
            shutdown_report = shutdown_processes(
                shared,
                started_procs,
                graceful_timeout_s=float(runtime.safety.shutdown_timeout_s),
            )
            return 1
        print("  inference: ready", flush=True)

        # A ready flag is one-shot state. Recheck the process immediately before
        # starting hardware so a post-ready inference crash cannot open devices.
        if shared.error_state.value or not inference_process.is_alive():
            logger.error("inference exited or faulted after reporting ready")
            shared.error_state.value = True
            require_transition(shared, SafetyState.FAULT)
            shutdown_report = shutdown_processes(
                shared,
                started_procs,
                graceful_timeout_s=float(runtime.safety.shutdown_timeout_s),
            )
            return 1

        remaining_pairs = [
            (spec, process)
            for spec, process in spec_processes
            if process is not inference_process
        ]
        remaining_procs = [process for _spec, process in remaining_pairs]
        start_processes(remaining_procs)
        started_procs.extend(remaining_procs)
        ready_checks: list[tuple[str, float]] = []
        for spec, _process in remaining_pairs:
            if spec.ready_name is not None:
                ready_checks.append((spec.ready_name, float(timeouts[spec.ready_name])))
        if not wait_subsystem_ready(shared, ready_checks, started_procs):
            shared.error_state.value = True
            require_transition(shared, SafetyState.FAULT)
            shutdown_report = shutdown_processes(
                shared,
                started_procs,
                graceful_timeout_s=float(runtime.safety.shutdown_timeout_s),
            )
            return 1

        for name, _timeout in ready_checks:
            print(f"  {name}: ready", flush=True)

        require_transition(shared, SafetyState.ARMED)
        print(
            f"\nAll subsystems ready — safety=ARMED({int(SafetyState.ARMED)})",
            flush=True,
        )
        home_planner = build_home_planner(runtime) if execute else None
        home_status = "return hand + arm home before B" if home_planner else "disabled"
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
            },
            name="policy-operator",
            daemon=True,
        )
        operator_thread.start()

        process_names = [spec.name for spec in specs]
        heartbeat_names = [
            spec.name
            for spec in specs
            if spec.name in {"arm", "hand", "inference", "policy"}
        ]
        exit_reason, normal_exit = run_supervisor(
            shared,
            started_procs,
            process_names,
            heartbeat_names,
            heartbeat_timeouts_s=dict(runtime.safety.heartbeat_timeouts),
            supervisor_hz=float(runtime.safety.supervisor_hz),
        )

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
                        shared.error_state.value = True
                        transition(shared, SafetyState.FAULT)
                except Exception:
                    logger.warning("RuntimeChannels cleanup failed", exc_info=True)
                    shared.error_state.value = True
                    transition(shared, SafetyState.FAULT)
