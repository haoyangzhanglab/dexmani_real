"""Learned-policy deployment lifecycle.

Composes the runtime primitives (``WorkerSpec`` +
``build_processes``/``start_processes``/``wait_subsystem_ready``/
``run_supervisor``/``shutdown_processes``) into the policy workflow — resolve
config -> create ``SharedStorage`` -> spawn arm (+ optional hand and RGB-D /
point-cloud workers) -> inference -> coordinator -> readiness -> ARMED ->
supervise -> verified shutdown. There is
no second health mechanism: the supervisor's heartbeat/readiness slots already
carry ``arm``/``hand``/``camera``/``pointcloud``/``inference``/``policy``.

There is no VR worker or recorder. Camera and point-cloud workers are included
only when the explicit observation contract contains ``point_cloud``.

Also owns the one-time startup provenance log line (commit hashes +
checkpoint/model-config SHA-256) via ``sha256_file`` and
``log_deployment_provenance``.
"""

from __future__ import annotations

import hashlib
import logging
import multiprocessing as mp
import os
import threading
from pathlib import Path
from typing import Any

from dexmani_real.config.runtime import ArmLoopConfig, ResolvedRuntimeConfig
from dexmani_real.deployment.config import DeploymentConfig
from dexmani_real.deployment.coordinator import CoordinatorConfig, coordinator_loop
from dexmani_real.deployment.observation import parse_observation_fields
from dexmani_real.deployment.operator import build_home_planner, run_operator_control
from dexmani_real.deployment.worker import inference_loop
from dexmani_real.robot.arm_loop import arm_loop
from dexmani_real.robot.hand_process import hand_loop
from dexmani_real.robot.safety import SafetyState, require_transition, transition
from dexmani_real.runtime.processes import (
    ShutdownReport,
    WorkerSpec,
    build_processes,
    start_processes,
)
from dexmani_real.runtime.supervisor import (
    run_supervisor,
    shutdown_processes,
    wait_subsystem_ready,
)
from dexmani_real.sensor.camera_process import CameraLoopConfig, camera_loop
from dexmani_real.sensor.pointcloud_process import PointCloudLoopConfig, pointcloud_loop
from dexmani_real.shm.shared_storage import SharedStorage, SharedStorageConfig
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


def _requires_pointcloud(deployment: DeploymentConfig) -> bool:
    return "point_cloud" in parse_observation_fields(deployment.observation_fields)


def sha256_file(path: str | Path) -> str:
    """Return the hex SHA-256 of a file's contents ("" when unreadable/missing).

    Best-effort: logs the checkpoint/model-config hash "if available"; an
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
    model_config_sha256: str = "",
) -> None:
    """Log one structured provenance line (no SharedStorage write).

    Provenance is a one-time startup log line, never a shared-memory payload:
    the full resolved config must not enter high-frequency IPC. Commit hashes
    are optional; absence logs ``unknown`` rather than fabricating a value.
    """
    logger.info(
        "deployment provenance: dexmani_commit=%s model_commit=%s "
        "runtime_target=%s observation_fields=%s pointcloud_num_points=%d checkpoint=%s "
        "checkpoint_sha256=%s model_config_sha256=%s runtime_sha256=%s",
        dexmani_commit or "unknown",
        model_commit or "unknown",
        deployment.runtime_target,
        deployment.observation_fields,
        deployment.pointcloud_num_points,
        deployment.checkpoint or "",
        checkpoint_sha256 or "",
        model_config_sha256 or "",
        runtime_sha256,
    )


def build_policy_worker_specs(
    shared: SharedStorage,
    runtime: ResolvedRuntimeConfig,
    deployment: DeploymentConfig,
) -> list[WorkerSpec]:
    """Build the workers required by the explicit deployment contract.

    Readiness order is the single source of truth for build order and for the
    readiness/heartbeat names derived from it. ``ready_name`` mirrors the
    process ``name`` for every worker (the coordinator reuses the existing
    ``policy`` control-source slot, the inference worker the new ``inference``
    slot).
    """
    coordinator_config = CoordinatorConfig.from_runtime(deployment, runtime)
    pointcloud_requested = _requires_pointcloud(deployment)
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
                        PointCloudLoopConfig.from_runtime(
                            runtime,
                            num_points=deployment.pointcloud_num_points,
                        ),
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
                (shared, deployment),
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
    deployment: DeploymentConfig,
    *,
    prefix: str | None = None,
) -> int:
    """Run one policy deployment lifecycle and return its exit code.

    Mirrors the collect_teleop order: ``build -> DISARMED -> start ->
    wait_subsystem_ready -> ARMED -> run_supervisor -> shutdown``, minus the VR
    transform/provenance/recording preflight the joint-only workflow does not
    need.
    """
    log_deployment_provenance(
        logger,
        deployment=deployment,
        runtime_sha256=runtime.sha256,
        checkpoint_sha256=(
            sha256_file(deployment.checkpoint) if deployment.checkpoint else ""
        ),
        model_config_sha256=(
            sha256_file(deployment.model_config_path)
            if deployment.model_config_path
            else ""
        ),
    )

    ctx = mp.get_context("spawn")
    pointcloud_requested = _requires_pointcloud(deployment)
    shared = SharedStorage.create(
        prefix=prefix or f"dexmani_policy_{os.getpid()}",
        config=SharedStorageConfig.from_runtime(
            runtime,
            pointcloud_num_points=deployment.pointcloud_num_points,
            pointcloud_requested=pointcloud_requested,
        ),
        mp_context=ctx,
    )
    specs: list[WorkerSpec] = []
    procs: list[Any] = []
    shutdown_report: ShutdownReport | None = None
    operator_thread: threading.Thread | None = None
    operator_stop: threading.Event | None = None
    try:
        specs = build_policy_worker_specs(shared, runtime, deployment)
        procs = build_processes(ctx, specs)
        require_transition(shared, SafetyState.DISARMED)
        start_processes(procs)

        timeouts = runtime.safety.readiness_timeouts_s
        ready_checks = [
            (spec.ready_name, float(timeouts[spec.ready_name]))
            for spec in specs
            if spec.ready_name
        ]
        if not wait_subsystem_ready(shared, ready_checks, procs):
            shared.error_state.value = True
            require_transition(shared, SafetyState.FAULT)
            shutdown_report = shutdown_processes(
                shared,
                procs,
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
        print(
            "  [B] start run   [S] stop run   [H] home   [Q] quit   [ESC] e-stop",
            flush=True,
        )

        home_planner = build_home_planner(runtime)
        operator_stop = threading.Event()
        operator_thread = threading.Thread(
            target=run_operator_control,
            args=(shared, runtime, deployment, home_planner),
            kwargs={"stop_event": operator_stop},
            name="policy-operator",
            daemon=True,
        )
        operator_thread.start()

        process_names = [spec.name for spec in specs]
        heartbeat_names = process_names
        exit_reason, normal_exit = run_supervisor(
            shared,
            procs,
            process_names,
            heartbeat_names,
            heartbeat_timeouts_s=dict(runtime.safety.heartbeat_timeouts),
            supervisor_hz=float(runtime.safety.supervisor_hz),
        )

        shutdown_report = shutdown_processes(
            shared,
            procs,
            graceful_timeout_s=float(runtime.safety.shutdown_timeout_s),
            disarm_if_clean=normal_exit,
        )
        clean_exit = normal_exit and shutdown_report.clean
        safety_name = (
            SafetyState(shutdown_report.safety_state).name
            if shutdown_report.safety_state is not None
            else "UNKNOWN"
        )
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
        if operator_thread is not None:
            operator_thread.join(timeout=1.0)
        if shutdown_report is None:
            started = [process for process in procs if process.pid is not None]
            if started:
                try:
                    shutdown_processes(
                        shared,
                        started,
                        graceful_timeout_s=float(runtime.safety.shutdown_timeout_s),
                    )
                except RuntimeError:
                    logger.critical(
                        "child process remains alive; leaving SharedStorage linked",
                        exc_info=True,
                    )
                    raise
            else:
                try:
                    if not shared.close():
                        shared.error_state.value = True
                        transition(shared, SafetyState.FAULT)
                except Exception:
                    logger.warning("SharedStorage cleanup failed", exc_info=True)
                    shared.error_state.value = True
                    transition(shared, SafetyState.FAULT)
