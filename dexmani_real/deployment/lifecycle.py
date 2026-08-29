"""Learned-policy deployment lifecycle.

Composes the runtime primitives (``WorkerSpec`` +
``build_processes``/``start_processes``/``wait_subsystem_ready``/
``run_supervisor``/``shutdown_processes``) into the policy workflow — resolve
config -> create ``RuntimeChannels`` -> start the artifact-verified inference
process -> spawn arm (+ optional hand and RGB-D / point-cloud workers) ->
coordinator -> readiness -> ARMED -> supervise -> verified shutdown. There is
no second health mechanism: the supervisor's heartbeat/readiness slots already
carry ``arm``/``hand``/``camera``/``pointcloud``/``inference``/``policy``.

There is no VR worker or recorder. Camera and point-cloud workers are included
only when the explicit observation contract contains ``point_cloud``.

Also owns the one-time startup provenance log line (commit hashes +
checkpoint SHA-256) via ``sha256_file`` and
``log_deployment_provenance``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing as mp
import os
import threading
from pathlib import Path
from typing import Any

from dexmani_real.config.runtime import ArmLoopConfig, ResolvedRuntimeConfig
from dexmani_real.deployment.config import DeploymentConfig, PolicyRuntimeConfig
from dexmani_real.deployment.coordinator import CoordinatorConfig, coordinator_loop
from dexmani_real.deployment.observation import parse_observation_fields
from dexmani_real.deployment.operator import build_home_planner, run_operator_control
from dexmani_real.deployment.run_identity import RealSourceIdentity
from dexmani_real.deployment.worker import inference_loop
from dexmani_real.ipc.channels import RuntimeChannels, RuntimeChannelsConfig
from dexmani_real.robot.arm_worker import arm_loop
from dexmani_real.robot.hand_worker import hand_loop
from dexmani_real.runtime.safety import SafetyState, require_transition, transition
from dexmani_real.runtime.supervisor import (
    run_supervisor,
    shutdown_processes,
    validate_max_running_s,
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


def _prepare_execute_receipt_dir() -> str:
    """Create the physical-execution receipt directory before hardware starts."""
    receipt_dir = Path(
        os.environ.get(
            "DEXMANI_RECEIPT_DIR",
            str(Path.home() / ".dexmani" / "receipts"),
        )
    )
    receipt_dir.mkdir(parents=True, exist_ok=True)
    if not receipt_dir.is_dir():
        raise RuntimeError(f"execute receipt path is not a directory: {receipt_dir}")
    return str(receipt_dir)


def _requires_pointcloud(deployment: DeploymentConfig) -> bool:
    return "point_cloud" in parse_observation_fields(deployment.observation_fields)


def sha256_file(path: str | Path) -> str:
    """Return the hex SHA-256 of a file's contents ("" when unreadable/missing).

    Best-effort: logs the checkpoint hash "if available"; an
    unreadable file logs empty rather than failing startup (the PolicyRuntime
    load is the authoritative check for a bad checkpoint).
    """
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return ""


def log_deployment_provenance(
    logger: logging.Logger,
    *,
    deployment: DeploymentConfig,
    runtime_sha256: str,
    dexmani_commit: str = "",
    model_commit: str = "",
    checkpoint_sha256: str = "",
) -> None:
    """Log one structured provenance line (no RuntimeChannels write).

    Provenance is a one-time startup log line, never a shared-memory payload:
    the full resolved config must not enter high-frequency IPC. Commit hashes
    are optional; absence logs ``unknown`` rather than fabricating a value.
    """
    logger.info(
        "deployment provenance: dexmani_commit=%s model_commit=%s "
        "runtime_target=%s observation_fields=%s pointcloud_num_points=%d checkpoint=%s "
        "checkpoint_sha256=%s runtime_sha256=%s",
        dexmani_commit or "unknown",
        model_commit or "unknown",
        deployment.runtime_target,
        deployment.observation_fields,
        deployment.pointcloud_num_points,
        deployment.checkpoint or "",
        checkpoint_sha256 or "",
        runtime_sha256,
    )


def build_policy_worker_specs(
    shared: RuntimeChannels,
    runtime: ResolvedRuntimeConfig,
    policy_runtime_config: PolicyRuntimeConfig,
    *,
    execute_receipt_dir: str | None = None,
    execute_receipt_provenance_json: str | None = None,
) -> list[WorkerSpec]:
    """Build the workers required by the explicit deployment contract.

    Readiness order is the single source of truth for build order and for the
    readiness/heartbeat names derived from it. ``ready_name`` mirrors the
    process ``name`` for every worker (the coordinator reuses the existing
    ``policy`` control-source slot, the inference worker the new ``inference``
    slot).
    """
    deployment = policy_runtime_config.deployment
    coordinator_config = CoordinatorConfig.from_runtime(
        deployment,
        runtime,
        execution_mode=policy_runtime_config.execution_mode,
        h4_execute_bounds=policy_runtime_config.h4_execute_bounds,
        task_execute_bounds=policy_runtime_config.task_execute_bounds,
        execute_receipt_dir=execute_receipt_dir,
        execute_receipt_provenance_json=execute_receipt_provenance_json,
    )
    pointcloud_requested = _requires_pointcloud(deployment)
    pointcloud_config = (
        PointCloudLoopConfig.from_runtime(
            runtime,
            num_points=deployment.pointcloud_num_points,
        )
        if pointcloud_requested
        else None
    )
    specs: list[WorkerSpec] = [
        WorkerSpec(
            "arm",
            arm_loop,
            (shared, ArmLoopConfig.from_runtime(runtime)),
            ready_name="arm",
        ),
    ]
    if pointcloud_requested:
        specs.extend(
            [
                WorkerSpec(
                    "camera",
                    camera_loop,
                    (shared, CameraLoopConfig.from_runtime(runtime)),
                    ready_name="camera",
                ),
                WorkerSpec(
                    "pointcloud",
                    pointcloud_loop,
                    (
                        shared,
                        pointcloud_config,
                    ),
                    ready_name="pointcloud",
                ),
            ]
        )
    specs.extend(
        [
            WorkerSpec(
                "inference",
                inference_loop,
                (shared, policy_runtime_config),
                ready_name="inference",
            ),
            WorkerSpec(
                "policy",
                coordinator_loop,
                (shared, coordinator_config),
                ready_name="policy",
            ),
        ]
    )
    if deployment.hand_enabled:
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
    policy_runtime_config: PolicyRuntimeConfig,
    *,
    prefix: str | None = None,
    max_running_s: float | None = None,
    real_source: RealSourceIdentity | None = None,
    invocation_argv: tuple[str, ...] | None = None,
) -> int:
    """Run one policy deployment lifecycle and return its exit code.

    A frozen ``shadow`` projection, bounded H4 ``execute``, or independently
    bounded ``task`` projection reaches this lifecycle. The inference worker must load
    successfully before any hardware process is started. The runtime then
    follows ``DISARMED -> hardware readiness -> ARMED -> supervision ->
    verified shutdown``.
    """
    if not isinstance(policy_runtime_config, PolicyRuntimeConfig):
        raise TypeError("policy_runtime_config must be a PolicyRuntimeConfig")
    if policy_runtime_config.execution_mode == "shadow":
        max_running_s = validate_max_running_s(max_running_s)
        if policy_runtime_config.h4_execute_bounds is not None:
            raise ValueError("shadow lifecycle must not carry H4 execute bounds")
    else:
        if max_running_s is not None:
            raise ValueError(
                "physical execute duration belongs to its immutable bounds"
            )
        execute_bounds = policy_runtime_config.physical_execute_bounds
        if execute_bounds is None:
            raise ValueError("physical execute lifecycle requires explicit bounds")
        max_running_s = execute_bounds.max_running_s
    deployment = policy_runtime_config.deployment
    if deployment.hand_enabled and not policy_runtime_config.hand_acknowledged:
        raise ValueError("deployment with hand targets requires --hand")
    execute_receipt_provenance_json: str | None = None
    if policy_runtime_config.execution_mode in {"execute", "task"}:
        if real_source is None or real_source.availability != "available":
            raise ValueError("physical execute requires resolved Real source identity")
        if real_source.dirty != "false":
            raise ValueError("physical execute requires a clean Real source identity")
        artifact = policy_runtime_config.artifact
        if artifact is None:
            raise ValueError("physical execute requires a resolved policy artifact")
        execute_receipt_provenance_json = json.dumps(
            {
                "artifact": {
                    "checkpoint_sha256": artifact.checkpoint_sha256_from_index,
                    "checkpoint": artifact.checkpoint_path.name,
                    "index_sha256": artifact.index_sha256,
                },
                "invocation_argv": list(invocation_argv or ()),
                "real_source": {
                    "commit": real_source.commit,
                    "dirty": real_source.dirty,
                    "python_tree_sha256": real_source.python_tree_sha256,
                },
                "runtime": {"config_sha256": runtime.sha256},
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    log_deployment_provenance(
        logger,
        deployment=deployment,
        runtime_sha256=runtime.sha256,
        dexmani_commit=((real_source.commit or "") if real_source is not None else ""),
        checkpoint_sha256=(
            sha256_file(deployment.checkpoint) if deployment.checkpoint else ""
        ),
    )

    execute_receipt_dir = (
        _prepare_execute_receipt_dir()
        if policy_runtime_config.execution_mode in {"execute", "task"}
        else None
    )

    ctx = mp.get_context("spawn")
    pointcloud_requested = _requires_pointcloud(deployment)
    shared = RuntimeChannels.create(
        prefix=prefix or f"dexmani_policy_{os.getpid()}",
        config=RuntimeChannelsConfig.from_runtime(
            runtime,
            pointcloud_num_points=deployment.pointcloud_num_points,
            pointcloud_requested=pointcloud_requested,
            observation_horizon=deployment.observation_horizon,
            observation_dt_s=1.0 / float(runtime.policy.control_hz),
            max_input_age_s=deployment.max_input_age_s,
            max_observation_skew_s=deployment.max_observation_skew_s,
            max_grid_lag_s=deployment.max_grid_lag_s,
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
            policy_runtime_config,
            execute_receipt_dir=execute_receipt_dir,
            execute_receipt_provenance_json=execute_receipt_provenance_json,
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
        home_planner = (
            build_home_planner(runtime)
            if policy_runtime_config.execution_mode in {"execute", "task"}
            else None
        )
        home_status = "return hand + arm home before B" if home_planner else "disabled"
        print(
            "  [B] start run   [S] stop run   [Q] quit   [ESC] e-stop   "
            f"[H] {home_status}",
            flush=True,
        )

        operator_stop = threading.Event()
        operator_thread = threading.Thread(
            target=run_operator_control,
            args=(shared, runtime, deployment, home_planner),
            kwargs={
                "stop_event": operator_stop,
                "execution_mode": policy_runtime_config.execution_mode,
            },
            name="policy-operator",
            daemon=True,
        )
        operator_thread.start()

        process_names = [spec.name for spec in specs]
        heartbeat_names = process_names
        exit_reason, normal_exit = run_supervisor(
            shared,
            started_procs,
            process_names,
            heartbeat_names,
            heartbeat_timeouts_s=dict(runtime.safety.heartbeat_timeouts),
            supervisor_hz=float(runtime.safety.supervisor_hz),
            max_running_s=max_running_s,
            exit_after_run_stops=(
                policy_runtime_config.execution_mode in {"execute", "task"}
            ),
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
        execute_completed = bool(shared.execute_completed.value)
        clean_exit = normal_exit and shutdown_report.clean
        if policy_runtime_config.execution_mode in {"execute", "task"}:
            # A clean cancellation is operationally safe, but it is not a
            # successful H4 execute.  Keep the process status unambiguous for
            # runbooks and any calling automation.
            clean_exit = clean_exit and execute_completed
        safety_name = (
            SafetyState(shutdown_report.safety_state).name
            if shutdown_report.safety_state is not None
            else "UNKNOWN"
        )
        print(f"\n── Session End ──")
        print(
            f"  exit_reason={exit_reason}  safety={safety_name}  "
            f"supervisor_normal={normal_exit}  execute_completed={execute_completed}  "
            f"clean={clean_exit}"
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
