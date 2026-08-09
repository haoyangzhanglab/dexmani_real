#!/usr/bin/env python3
"""Manifest-driven learned policy runtime for xArm7 and optional XHand.

This is a hardware-affecting entry point.  It starts DISARMED, validates model
resources and live observation history, then requires an explicit B key before
entering RUNNING.  C/S/D return to an applied coordinated hold; ESC requests
the global e-stop and Q shuts down.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dexmani_real.config.runtime import resolve_runtime_config
from dexmani_real.policy.inference_process import InferenceConfig, InferenceWorkerTransport, inference_loop
from dexmani_real.policy.learned_coordinator import learned_policy_loop
from dexmani_real.policy.observation_sources import with_observation_capacities
from dexmani_real.policy.tensor_block import ObservationTensorBlock
from dexmani_real.robot.arm_loop import ArmLoopConfig, arm_loop
from dexmani_real.robot.hand_process import HandProcessConfig, hand_loop
from dexmani_real.robot.safety import SafetyState, transition
from dexmani_real.sensor.camera_process import CameraLoopConfig, camera_loop
from dexmani_real.sensor.vr_receiver_process import VRReceiverConfig, vr_loop
from dexmani_real.shm.shared_storage import (
    SharedStorage,
    SharedStorageConfig,
    run_supervisor,
    shutdown_processes,
    wait_subsystem_ready,
)
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manifest-driven DexMani learned-policy runtime")
    parser.add_argument("--manifest", required=True, help="Model/backend resource manifest JSON")
    parser.add_argument("--config", default=None, help="Runtime config JSON")
    parser.add_argument("--no-hand", action="store_true", help="Disable XHand; manifest must be arm-only")
    parser.add_argument("--print-config", action="store_true", help="Validate and print config/manifest, then exit")
    args = parser.parse_args()

    inference = InferenceConfig.from_manifest(args.manifest)
    runtime = resolve_runtime_config(
        json_path=args.config,
        cli_overrides={"policy.hand_enabled": False if args.no_hand else None},
    )
    modality_names = {modality.name for modality in inference.observation_spec.modalities}
    needs_camera = any(name.startswith("camera_") for name in modality_names)
    needs_vr = any(name.startswith("vr_") for name in modality_names)
    needs_hand = "hand" in inference.actuators or any(name.startswith("hand_") for name in modality_names)
    if needs_hand and not bool(runtime.policy.hand_enabled):
        parser.error("manifest requires the hand capability but resolved runtime disables it")

    if args.print_config:
        print(json.dumps(json.loads(runtime.canonical_json), indent=2, ensure_ascii=False, sort_keys=True))
        print(
            json.dumps(
                {
                    "manifest_sha256": inference.manifest_sha256,
                    "resources": dict(inference.resource_hashes),
                    "actuators": inference.actuators,
                    "modalities": [dataclasses.asdict(item) for item in inference.observation_spec.modalities],
                    "action": dataclasses.asdict(inference.action_spec),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    ctx = mp.get_context("spawn")
    storage_config = with_observation_capacities(
        SharedStorageConfig.from_runtime(runtime),
        inference.observation_spec,
    )
    prefix = f"dexmani_learned_{os.getpid()}"
    shared = SharedStorage.create(prefix=prefix, config=storage_config, mp_context=ctx)
    tensor_block = ObservationTensorBlock.create(f"{prefix}_observation", inference.observation_spec)
    shared.camera_observation_required.value = needs_camera
    processes: list[Any] = []
    try:
        arm_config = ArmLoopConfig.from_runtime(runtime)
        hand_config = HandProcessConfig.from_runtime(runtime)
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
        if needs_hand:
            processes.append(ctx.Process(target=hand_loop, args=(shared, hand_config), name="hand", daemon=False))
        if needs_camera:
            processes.append(ctx.Process(target=camera_loop, args=(shared, camera_config), name="camera", daemon=False))
        if needs_vr:
            processes.append(ctx.Process(target=vr_loop, args=(shared, vr_config), name="vr", daemon=False))

        transition(shared, SafetyState.DISARMED)
        for process in processes:
            process.start()

        ready_checks: list[tuple[str, Any, float]] = [("arm", shared.arm_ready, 15.0)]
        if needs_hand:
            ready_checks.append(("hand", shared.hand_ready, 15.0))
        if needs_camera:
            ready_checks.append(("camera", shared.camera_ready, 15.0))
        if needs_vr:
            ready_checks.append(("vr", shared.vr_ready, 120.0))
        ready_checks.extend(
            [
                ("inference", shared.inference_ready, 60.0),
                ("policy", shared.policy_ready, 120.0),
            ]
        )
        if not wait_subsystem_ready(shared, ready_checks, processes):
            shutdown_processes(shared, processes)
            return

        transition(shared, SafetyState.ARMED)
        print("Learned policy ready and ARMED. B=run C/S/D=hold Q=quit ESC=e-stop", flush=True)

        names = [process.name for process in processes]
        heartbeat_fields = {
            "inference": shared.inference_heartbeat_s,
            "policy": shared.policy_heartbeat_s,
            "arm": shared.arm_heartbeat_s,
        }
        if needs_hand:
            heartbeat_fields["hand"] = shared.hand_heartbeat_s
        if needs_camera:
            heartbeat_fields["camera"] = shared.camera_heartbeat_s
        if needs_vr:
            heartbeat_fields["vr"] = shared.vr_heartbeat_s
        now_s = time.monotonic()
        for heartbeat in heartbeat_fields.values():
            if heartbeat.value == 0.0:
                heartbeat.value = now_s
        run_supervisor(
            shared,
            processes,
            names,
            heartbeat_fields,
            heartbeat_timeouts_s=dict(runtime.safety.heartbeat_timeouts),
            supervisor_hz=float(runtime.safety.supervisor_hz),
        )
        if shared.safety_state.value != int(SafetyState.FAULT):
            transition(shared, SafetyState.DISARMED)
        shutdown_processes(shared, processes)
    finally:
        started = [process for process in processes if process.pid is not None]
        if any(process.is_alive() for process in started):
            try:
                shutdown_processes(shared, started)
            except RuntimeError:
                logger.critical("child remains alive; leaving all shared memory linked", exc_info=True)
        if not any(process.is_alive() for process in started):
            shared.close()
            try:
                tensor_block.close()
                tensor_block.unlink()
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    main()
