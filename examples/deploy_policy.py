#!/usr/bin/env python3
"""Experimental learned-policy deployment entry point.

Starts an isolated inference worker alongside the policy coordinator, arm,
and optional hand/camera/VR subsystems.  The PolicySpec YAML binds the adapter
module, observation modalities, action contract, and resource hashes;
``hardware_deployable: true`` is required for live runs.

Safety: DISARMED → ARMED → RUNNING/FAULT state machine with heartbeat
supervision across all enabled capabilities.

The full deployment lifecycle lives here rather than in the ``dexmani_real``
package — that keeps the package focused on reusable library code and avoids
accumulating entry-point logic.

Usage:
    python examples/deploy_policy.py --policy POLICY.yaml [--config PATH] [--no-hand] [--print-config]
Controls:
    B=run  C/S/D=hold  Q=quit  ESC=estop
"""

from __future__ import annotations

import argparse
import dataclasses
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from dexmani_real.config.runtime import ResolvedRuntimeConfig, resolve_runtime_config
from dexmani_real.policy.inference_process import InferenceWorkerTransport
from dexmani_real.policy.inference_process import inference_loop as _inference_loop
from dexmani_real.policy.learned_coordinator import learned_policy_loop as _learned_policy_loop
from dexmani_real.policy.observation_sources import with_observation_capacities
from dexmani_real.policy.spec import PolicySpec
from dexmani_real.policy.tensor_block import ObservationTensorBlock
from dexmani_real.robot.arm_loop import ArmLoopConfig
from dexmani_real.robot.arm_loop import arm_loop as _arm_loop
from dexmani_real.robot.hand_process import HandProcessConfig
from dexmani_real.robot.hand_process import hand_loop as _hand_loop
from dexmani_real.robot.safety import SafetyState, require_transition, transition
from dexmani_real.runtime.supervisor import (
    run_supervisor,
    shutdown_processes,
    wait_subsystem_ready,
)
from dexmani_real.sensor.camera_process import CameraLoopConfig
from dexmani_real.sensor.camera_process import camera_loop as _camera_loop
from dexmani_real.sensor.vr_receiver_process import VRReceiverConfig
from dexmani_real.sensor.vr_receiver_process import vr_loop as _vr_loop
from dexmani_real.shm.shared_storage import SharedStorage, SharedStorageConfig
from dexmani_real.teleop.keyboard import validate_arm_feedback, validate_hand_feedback
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Capability detection
# ---------------------------------------------------------------------------


def _required_capabilities(inference: PolicySpec) -> tuple[bool, bool, bool]:
    """Return ``(needs_camera, needs_vr, needs_hand)`` derived purely from the PolicySpec."""
    modality_names = {modality.name for modality in inference.observation.modalities}
    needs_camera = any(name.startswith("camera_") for name in modality_names)
    needs_vr = any(name.startswith("vr_") for name in modality_names)
    needs_hand = "hand" in inference.actuators or any(name.startswith("hand_") for name in modality_names)
    return needs_camera, needs_vr, needs_hand


# ---------------------------------------------------------------------------
# Preflight validation
# ---------------------------------------------------------------------------


def _feedback_preflight_issues(
    shared: SharedStorage,
    runtime: ResolvedRuntimeConfig,
    *,
    hand_feedback_enabled: bool,
    needs_camera: bool = False,
    needs_vr: bool = False,
    now_s: float | None = None,
    now_monotonic_ns: int | None = None,
) -> list[str]:
    """Validate fresh, finite feedback before Main permits ARMED."""
    issues: list[str] = []
    if shared.error_state.value:
        issues.append("sticky error_state is set")
    if shared.estop_request.value:
        issues.append("e-stop is requested")

    # --- heartbeats -------------------------------------------------------
    enabled_heartbeats: list[str] = ["inference", "policy", "arm"]
    if hand_feedback_enabled:
        enabled_heartbeats.append("hand")
    if needs_camera:
        enabled_heartbeats.append("camera")
    if needs_vr:
        enabled_heartbeats.append("vr")
    heartbeat_timeouts = runtime.safety.heartbeat_timeouts
    for name in enabled_heartbeats:
        last_s = shared.get_heartbeat(name)
        current_s = time.monotonic() if now_s is None else now_s
        if (
            not np.isfinite(last_s)
            or last_s <= 0
            or last_s > current_s
            or current_s - last_s > float(heartbeat_timeouts[name])
        ):
            issues.append(f"{name} heartbeat is missing or stale")

    # --- arm feedback ----------------------------------------------------
    arm_result = shared.arm_state_ring.read_latest()
    if arm_result is None:
        issues.append("arm feedback is unavailable")
    else:
        arm, _timestamp_ns, _sequence = arm_result
        current_ns = time.monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns
        arm_issue = validate_arm_feedback(
            connected=bool(arm["connected"][0]),
            state_valid=bool(arm["state_valid"][0]),
            source_monotonic_ns=int(arm["source_monotonic_ns"][0]),
            now_monotonic_ns=current_ns,
            max_age_s=float(runtime.policy.arm_state_stale_threshold_s),
            qpos=arm["qpos"][0],
            qvel=arm["qvel"][0],
            eef_pos=arm["eef_pos"][0],
            eef_rot6d=arm["eef_rot6d"][0],
        )
        if arm_issue is not None:
            issues.append(arm_issue)
        error_code = int(arm["error_code"][0])
        if error_code != 0:
            issues.append(f"arm controller error C{error_code}")

    # --- hand feedback ----------------------------------------------------
    if hand_feedback_enabled:
        hand_result = shared.hand_state_ring.read_latest()
        if hand_result is None:
            issues.append("hand feedback is unavailable")
        else:
            hand, _timestamp_ns, _sequence = hand_result
            current_ns = time.monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns
            hand_issue = validate_hand_feedback(
                connected=bool(hand["connected"][0]),
                error_state=bool(hand["error_state"][0]),
                state_valid=bool(hand["state_valid"][0]),
                send_healthy=bool(hand["send_healthy"][0]),
                read_healthy=bool(hand["read_healthy"][0]),
                source_monotonic_ns=int(hand["source_monotonic_ns"][0]),
                now_monotonic_ns=current_ns,
                max_age_s=float(heartbeat_timeouts["hand"]),
                qpos=hand["qpos"][0],
            )
            if hand_issue is not None:
                issues.append(hand_issue)

    # --- VR feedback ------------------------------------------------------
    if needs_vr:
        vr_result = shared.vr_ring.read_latest()
        if vr_result is None:
            issues.append("VR feedback is unavailable")
        else:
            vr, _timestamp_ns, _sequence = vr_result
            local_recv_ns = int(vr["local_recv_ns"][0])
            current_ns = time.monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns
            vr_ok = (
                np.all(np.isfinite(vr["wrist_pos"][0]))
                and np.all(np.isfinite(vr["wrist_quat_wxyz"][0]))
                and np.all(np.isfinite(vr["landmarks"][0]))
                and 0 < local_recv_ns <= current_ns
                and current_ns - local_recv_ns <= int(float(runtime.policy.vr_mapping.stale_threshold_s) * 1e9)
            )
            if not vr_ok:
                issues.append("VR feedback is invalid or stale")

    # --- camera feedback --------------------------------------------------
    if needs_camera:
        camera_result = shared.camera_ring.read_latest()
        if camera_result is None:
            issues.append("camera frame is unavailable")
        else:
            header = camera_result[0]
            source_ns = int(header["source_monotonic_ns"][0])
            current_ns = time.monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns
            camera_ok = (
                int(header["camera_health"][0]) == 0
                and 0 < source_ns <= current_ns
                and current_ns - source_ns <= int(float(runtime.camera.max_frame_age_s) * 1e9)
            )
            if not camera_ok:
                issues.append("camera frame is unhealthy or stale")

    return issues


# ---------------------------------------------------------------------------
# Session header
# ---------------------------------------------------------------------------


def _print_session_header(
    runtime: ResolvedRuntimeConfig,
    inference: PolicySpec,
    *,
    hand_feedback_enabled: bool,
    needs_camera: bool,
    needs_vr: bool,
) -> None:
    process_labels = ["inference", "policy", "arm"]
    if hand_feedback_enabled:
        process_labels.append("hand")
    if needs_camera:
        process_labels.append("camera")
    if needs_vr:
        process_labels.append("vr")
    session_meta = [
        f"policy={inference.sha256[:12]}",
        f"adapter={inference.adapter_module}",
        f"hand={'ON' if hand_feedback_enabled else 'OFF'}",
        f"config={runtime.sha256[:12]}",
    ]
    print("=" * 60)
    print("  DexMani Learned Policy Deployment")
    print(f"  procs: {' | '.join(process_labels)}")
    print(f"  {'  '.join(session_meta)}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Experimental DexMani learned-policy deployment")
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

    needs_camera, needs_vr, needs_hand = _required_capabilities(inference)
    hand_feedback_enabled = bool(runtime.policy.hand_enabled)
    if needs_hand and not hand_feedback_enabled:
        parser.error("PolicySpec requires hand capability but resolved runtime disables it")

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


# ---------------------------------------------------------------------------
# Deployment lifecycle
# ---------------------------------------------------------------------------


def run_policy_deployment(
    inference: PolicySpec,
    runtime: ResolvedRuntimeConfig,
    *,
    needs_camera: bool,
    needs_vr: bool,
    hand_feedback_enabled: bool,
    fixed_hand_home_acknowledged: bool,
) -> int:
    """Run one validated learned-policy deployment lifecycle.

    All structural checks (hardware_deployable, control_hz, capability
    compatibility) are performed by *main()* before this function is called.
    """
    runtime_hand_enabled = bool(runtime.policy.hand_enabled)
    if not runtime_hand_enabled and not fixed_hand_home_acknowledged:
        raise ValueError("fixed hand-home deployment requires explicit acknowledgement")

    ctx = mp.get_context("spawn")
    storage_config = with_observation_capacities(
        SharedStorageConfig.from_runtime(runtime, enable_inference=True),
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
        camera_config = CameraLoopConfig.from_runtime(runtime) if needs_camera else None
        vr_config = VRReceiverConfig.from_runtime(runtime) if needs_vr else None
        inference_transport = InferenceWorkerTransport.from_shared(shared)
        processes = [
            ctx.Process(
                target=_inference_loop,
                args=(inference_transport, tensor_block, inference),
                name="inference",
                daemon=False,
            ),
            ctx.Process(
                target=_learned_policy_loop,
                args=(shared, runtime, inference, tensor_block),
                name="policy",
                daemon=False,
            ),
            ctx.Process(target=_arm_loop, args=(shared, arm_config), name="arm", daemon=False),
        ]
        if hand_config is not None:
            processes.append(ctx.Process(target=_hand_loop, args=(shared, hand_config), name="hand", daemon=False))
        if needs_camera:
            processes.append(ctx.Process(target=_camera_loop, args=(shared, camera_config), name="camera", daemon=False))
        if needs_vr:
            processes.append(ctx.Process(target=_vr_loop, args=(shared, vr_config), name="vr", daemon=False))

        _print_session_header(
            runtime,
            inference,
            hand_feedback_enabled=hand_feedback_enabled,
            needs_camera=needs_camera,
            needs_vr=needs_vr,
        )

        require_transition(shared, SafetyState.DISARMED)
        for process in processes:
            process.start()

        ready_timeouts_s = dict(runtime.safety.readiness_timeouts_s)
        ready_checks: list[tuple[str, float]] = [("arm", ready_timeouts_s["arm"])]
        if hand_feedback_enabled:
            ready_checks.append(("hand", ready_timeouts_s["hand"]))
        if needs_camera:
            ready_checks.append(("camera", ready_timeouts_s["camera"]))
        if needs_vr:
            ready_checks.append(("vr", ready_timeouts_s["vr"]))
        ready_checks.extend(
            [
                ("inference", ready_timeouts_s["inference"]),
                ("policy", ready_timeouts_s["policy"]),
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
            needs_camera=needs_camera,
            needs_vr=needs_vr,
        )
        if feedback_issues:
            for issue in feedback_issues:
                logger.error("learned-policy feedback preflight failed: %s", issue)
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
        heartbeat_names = ["inference", "policy", "arm"]
        if hand_feedback_enabled:
            heartbeat_names.append("hand")
        if needs_camera:
            heartbeat_names.append("camera")
        if needs_vr:
            heartbeat_names.append("vr")
        exit_reason, normal_exit = run_supervisor(
            shared,
            processes,
            names,
            heartbeat_names,
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


if __name__ == "__main__":
    raise SystemExit(main())
