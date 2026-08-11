"""Learned-policy experiment setup, supervision, and shutdown.

This is a hardware-affecting entry point.  It starts DISARMED, validates model
resources and live observation history, then requires an explicit B key before
entering RUNNING.  C/S/D return to an applied coordinated hold; ESC requests
the global e-stop and Q shuts down.
"""

from __future__ import annotations

import argparse
import dataclasses
import multiprocessing as mp
import os
import time
from typing import Any

import numpy as np
import yaml

from dexmani_real.config.runtime import ResolvedRuntimeConfig, resolve_runtime_config
from dexmani_real.policy.inference_process import InferenceWorkerTransport, inference_loop
from dexmani_real.policy.learned_coordinator import learned_policy_loop
from dexmani_real.policy.observation_sources import with_observation_capacities
from dexmani_real.policy.spec import PolicySpec
from dexmani_real.policy.tensor_block import ObservationTensorBlock
from dexmani_real.robot.arm_loop import ArmLoopConfig, arm_loop
from dexmani_real.robot.hand_process import HandProcessConfig, hand_loop
from dexmani_real.robot.safety import SafetyState, require_transition, transition
from dexmani_real.runtime.supervisor import run_supervisor, shutdown_processes, wait_subsystem_ready
from dexmani_real.sensor.camera_process import CameraLoopConfig, camera_loop
from dexmani_real.sensor.vr_receiver_process import VRReceiverConfig, vr_loop
from dexmani_real.shm.shared_storage import SharedStorage, SharedStorageConfig
from dexmani_real.teleop.keyboard import validate_arm_feedback, validate_hand_feedback
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


def _required_capabilities(inference: PolicySpec, runtime: Any) -> tuple[bool, bool, bool]:
    modality_names = {modality.name for modality in inference.observation.modalities}
    needs_camera = any(name.startswith("camera_") for name in modality_names)
    needs_vr = any(name.startswith("vr_") for name in modality_names)
    policy_requires_hand = "hand" in inference.actuators or any(name.startswith("hand_") for name in modality_names)
    hand_feedback_enabled = bool(runtime.policy.hand_enabled)
    if policy_requires_hand and not hand_feedback_enabled:
        raise ValueError("PolicySpec requires the hand capability but resolved runtime disables it")
    return needs_camera, needs_vr, hand_feedback_enabled


def _feedback_preflight_issues(
    shared: SharedStorage,
    runtime: ResolvedRuntimeConfig,
    *,
    hand_feedback_enabled: bool,
    now_monotonic_ns: int | None = None,
) -> list[str]:
    """Validate the measured geometry that deployment will use after arming."""
    issues: list[str] = []
    if shared.error_state.value:
        issues.append("sticky error_state is set")
    if shared.estop_request.value:
        issues.append("e-stop is requested")

    arm_result = shared.arm_state_ring.read_latest()
    if arm_result is None:
        issues.append("arm feedback is unavailable")
    else:
        arm = arm_result[0][0]
        arm_now_ns = time.monotonic_ns() if now_monotonic_ns is None else int(now_monotonic_ns)
        arm_issue = validate_arm_feedback(
            connected=bool(arm["connected"]),
            state_valid=bool(arm["state_valid"]),
            source_monotonic_ns=int(arm["source_monotonic_ns"]),
            now_monotonic_ns=arm_now_ns,
            max_age_s=float(runtime.policy.arm_state_stale_threshold_s),
            qpos=np.asarray(arm["qpos"]),
            qvel=np.asarray(arm["qvel"]),
            eef_pos=np.asarray(arm["eef_pos"]),
            eef_rot6d=np.asarray(arm["eef_rot6d"]),
        )
        if arm_issue is not None:
            issues.append(arm_issue)
        error_code = int(arm["error_code"])
        if error_code != 0:
            issues.append(f"arm controller error C{error_code}")

    if hand_feedback_enabled:
        hand_result = shared.hand_state_ring.read_latest()
        if hand_result is None:
            issues.append("hand feedback is unavailable")
        else:
            hand = hand_result[0][0]
            hand_now_ns = time.monotonic_ns() if now_monotonic_ns is None else int(now_monotonic_ns)
            hand_issue = validate_hand_feedback(
                connected=bool(hand["connected"]),
                error_state=bool(hand["error_state"]),
                qpos_stale=bool(hand["qpos_stale"]),
                state_valid=bool(hand["state_valid"]),
                send_healthy=bool(hand["send_healthy"]),
                read_healthy=bool(hand["read_healthy"]),
                source_monotonic_ns=int(hand["source_monotonic_ns"]),
                now_monotonic_ns=hand_now_ns,
                max_age_s=float(runtime.safety.heartbeat_timeouts["hand"]),
                qpos=np.asarray(hand["qpos"]),
            )
            if hand_issue is not None:
                issues.append(hand_issue)

    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DexMani policy deployment experiment")
    parser.add_argument("--policy", required=True, help="Policy adapter and resource YAML")
    parser.add_argument("--config", default=None, help="Experiment config YAML")
    parser.add_argument(
        "--no-hand",
        action="store_true",
        help="Do not start XHand; use only when absent or secured at configured home (PolicySpec must be arm-only).",
    )
    parser.add_argument("--print-config", action="store_true", help="Validate and print runtime/PolicySpec, then exit")
    args = parser.parse_args(argv)

    try:
        inference = PolicySpec.from_yaml(args.policy)
        runtime = resolve_runtime_config(
            yaml_path=args.config,
            cli_overrides={"policy.hand_enabled": False if args.no_hand else None},
        )
    except (KeyError, OSError, TypeError, UnicodeError, ValueError, yaml.YAMLError) as exc:
        parser.error(f"invalid policy or experiment config: {exc}")
    if not args.no_hand and not bool(runtime.policy.hand_enabled):
        parser.error(
            "runtime config disables hand feedback; pass --no-hand explicitly to confirm the fixed-home assumption"
        )
    try:
        needs_camera, needs_vr, hand_feedback_enabled = _required_capabilities(inference, runtime)
    except ValueError as exc:
        parser.error(str(exc))
    if not np.isclose(
        inference.observation.control_hz,
        float(runtime.policy.control_hz),
        rtol=0.0,
        atol=1e-12,
    ):
        parser.error("PolicySpec control_hz must match the resolved runtime policy.control_hz")

    if args.print_config:
        print(runtime.canonical_yaml, end="")
        print(
            yaml.safe_dump(
                {
                    "policy_sha256": inference.sha256,
                    "adapter_module": inference.adapter_module,
                    "hardware_deployable": inference.hardware_deployable,
                    "resources": dict(inference.resource_hashes),
                    "actuators": inference.actuators,
                    "observations": [dataclasses.asdict(item) for item in inference.observation.modalities],
                    "action": dataclasses.asdict(inference.action),
                },
                allow_unicode=True,
                sort_keys=True,
            ),
            end="",
        )
        return 0

    if not inference.hardware_deployable:
        parser.error("PolicySpec is marked offline-only and cannot start hardware workers")

    try:
        return run_policy_deployment(
            inference,
            runtime,
            needs_camera=needs_camera,
            needs_vr=needs_vr,
            hand_feedback_enabled=hand_feedback_enabled,
            fixed_hand_home_acknowledged=bool(args.no_hand),
        )
    except Exception:
        logger.error("learned-policy startup failed before lifecycle ownership was established", exc_info=True)
        return 1


def run_policy_deployment(
    inference: PolicySpec,
    runtime: ResolvedRuntimeConfig,
    *,
    needs_camera: bool,
    needs_vr: bool,
    hand_feedback_enabled: bool,
    fixed_hand_home_acknowledged: bool,
) -> int:
    """Run one validated learned-policy deployment lifecycle."""
    if not inference.hardware_deployable:
        raise ValueError("PolicySpec is marked offline-only and cannot start hardware workers")
    if not np.isclose(
        inference.observation.control_hz,
        float(runtime.policy.control_hz),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("PolicySpec control_hz must match the resolved runtime policy.control_hz")
    expected_capabilities = _required_capabilities(inference, runtime)
    supplied_capabilities = (bool(needs_camera), bool(needs_vr), bool(hand_feedback_enabled))
    if supplied_capabilities != expected_capabilities:
        raise ValueError("deployment capabilities must match PolicySpec and the resolved runtime")
    runtime_hand_enabled = bool(runtime.policy.hand_enabled)
    if not runtime_hand_enabled and fixed_hand_home_acknowledged is not True:
        raise ValueError("fixed hand-home deployment requires explicit acknowledgement")

    ctx = mp.get_context("spawn")
    storage_config = with_observation_capacities(
        SharedStorageConfig.from_runtime(runtime),
        inference.observation,
    )
    prefix = f"dexmani_learned_{os.getpid()}"
    shared = SharedStorage.create(prefix=prefix, config=storage_config, mp_context=ctx)
    tensor_block: ObservationTensorBlock | None = None
    processes: list[Any] = []
    shared_closed = False
    try:
        tensor_block = ObservationTensorBlock.create(f"{prefix}_observation", inference.observation)
        shared.camera_observation_required.value = needs_camera
        arm_config = ArmLoopConfig.from_runtime(runtime)
        hand_config = HandProcessConfig.from_runtime(runtime) if hand_feedback_enabled else None
        camera_config = CameraLoopConfig.from_runtime(runtime)
        vr_config = VRReceiverConfig.from_runtime(runtime)
        inference_transport = InferenceWorkerTransport.from_shared(shared)
        processes = [
            ctx.Process(
                target=inference_loop,
                args=(inference_transport, tensor_block, inference),
                name="inference",
                daemon=False,
            ),
            ctx.Process(
                target=learned_policy_loop,
                args=(shared, runtime, inference, tensor_block),
                name="policy",
                daemon=False,
            ),
            ctx.Process(target=arm_loop, args=(shared, arm_config), name="arm", daemon=False),
        ]
        if hand_config is not None:
            processes.append(ctx.Process(target=hand_loop, args=(shared, hand_config), name="hand", daemon=False))
        if needs_camera:
            processes.append(ctx.Process(target=camera_loop, args=(shared, camera_config), name="camera", daemon=False))
        if needs_vr:
            processes.append(ctx.Process(target=vr_loop, args=(shared, vr_config), name="vr", daemon=False))

        require_transition(shared, SafetyState.DISARMED)
        for process in processes:
            process.start()

        ready_timeouts_s = dict(runtime.safety.readiness_timeouts_s)
        ready_checks: list[tuple[str, Any, float]] = [("arm", shared.arm_ready, ready_timeouts_s["arm"])]
        if hand_feedback_enabled:
            ready_checks.append(("hand", shared.hand_ready, ready_timeouts_s["hand"]))
        if needs_camera:
            ready_checks.append(("camera", shared.camera_ready, ready_timeouts_s["camera"]))
        if needs_vr:
            ready_checks.append(("vr", shared.vr_ready, ready_timeouts_s["vr"]))
        ready_checks.extend(
            [
                ("inference", shared.inference_ready, ready_timeouts_s["inference"]),
                ("policy", shared.policy_ready, ready_timeouts_s["policy"]),
            ]
        )
        if not wait_subsystem_ready(shared, ready_checks, processes):
            shared.error_state.value = True
            require_transition(shared, SafetyState.FAULT)
            shutdown_report = shutdown_processes(
                shared,
                processes,
                graceful_timeout_s=float(runtime.safety.shutdown_timeout_s),
            )
            shared_closed = shutdown_report.shared_closed
            return 1

        feedback_issues = _feedback_preflight_issues(
            shared,
            runtime,
            hand_feedback_enabled=hand_feedback_enabled,
        )
        if feedback_issues:
            logger.error("learned-policy feedback preflight failed: %s", "; ".join(feedback_issues))
            shared.error_state.value = True
            require_transition(shared, SafetyState.FAULT)
            shutdown_report = shutdown_processes(
                shared,
                processes,
                graceful_timeout_s=float(runtime.safety.shutdown_timeout_s),
            )
            shared_closed = shutdown_report.shared_closed
            return 1

        require_transition(shared, SafetyState.ARMED)
        print("Learned policy ready and ARMED. B=run C/S/D=hold Q=quit ESC=e-stop", flush=True)

        names = [process.name for process in processes]
        heartbeat_fields = {
            "inference": shared.inference_heartbeat_s,
            "policy": shared.policy_heartbeat_s,
            "arm": shared.arm_heartbeat_s,
        }
        if hand_feedback_enabled:
            heartbeat_fields["hand"] = shared.hand_heartbeat_s
        if needs_camera:
            heartbeat_fields["camera"] = shared.camera_heartbeat_s
        if needs_vr:
            heartbeat_fields["vr"] = shared.vr_heartbeat_s
        exit_reason, normal_exit = run_supervisor(
            shared,
            processes,
            names,
            heartbeat_fields,
            heartbeat_timeouts_s=dict(runtime.safety.heartbeat_timeouts),
            supervisor_hz=float(runtime.safety.supervisor_hz),
        )
        shutdown_report = shutdown_processes(
            shared,
            processes,
            graceful_timeout_s=float(runtime.safety.shutdown_timeout_s),
            disarm_if_clean=normal_exit,
        )
        shared_closed = shutdown_report.shared_closed
        clean_exit = normal_exit and shutdown_report.clean
        if normal_exit and not clean_exit:
            logger.error("verified shutdown invalidated the clean supervisor exit: %s", shutdown_report)
        print(f"Deployment ended: {exit_reason}; clean={clean_exit}", flush=True)
        return 0 if clean_exit else 1
    except Exception:
        logger.error("learned-policy deployment failed", exc_info=True)
        shared.error_state.value = True
        require_transition(shared, SafetyState.FAULT)
        return 1
    finally:
        started = [process for process in processes if process.pid is not None]
        if any(process.is_alive() for process in started):
            try:
                shutdown_report = shutdown_processes(
                    shared,
                    started,
                    graceful_timeout_s=float(runtime.safety.shutdown_timeout_s),
                )
                shared_closed = shutdown_report.shared_closed
            except RuntimeError:
                logger.critical("child remains alive; leaving all shared memory linked", exc_info=True)
                raise
        if not any(process.is_alive() for process in started):
            if tensor_block is not None:
                try:
                    tensor_block.close()
                    tensor_block.unlink()
                except FileNotFoundError:
                    pass
                except Exception:
                    logger.warning("observation tensor cleanup failed", exc_info=True)
            if not shared_closed:
                try:
                    shared_closed = bool(shared.close())
                    if not shared_closed:
                        shared.error_state.value = True
                        transition(shared, SafetyState.FAULT)
                        logger.error("SharedStorage cleanup was incomplete")
                except Exception:
                    logger.warning("SharedStorage cleanup failed", exc_info=True)
                    shared.error_state.value = True
                    transition(shared, SafetyState.FAULT)
