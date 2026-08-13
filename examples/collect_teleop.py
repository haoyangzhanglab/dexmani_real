#!/usr/bin/env python3
"""VR teleoperation and data-collection entry point.

Spawn-only capability model: VR, teleop, arm, measured hand (unless explicitly
disabled), camera, and RecorderIO (only when recording is enabled).
Safety: DISARMED/ARMED/RUNNING/FAULT state machine + enabled-capability heartbeats.
Recording: one causal configured-rate grid published as HDF5 schema v16.

The full experiment lifecycle lives here rather than in the ``dexmani_real``
package — that keeps the package focused on reusable library code and avoids
accumulating entry-point logic.

Usage:
    python examples/collect_teleop.py [--task T] [--operator O] [--acc A] [--speed S]
                     [--no-hand] [--no-record] [--config PATH] [--print-config]
Controls:
    B=teleop(+record when enabled)  C=pause  S=stop/save  D=discard  H=home  Q=quit  ESC=estop
"""

from __future__ import annotations

import argparse
import hashlib
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from dexmani_real import ASSET_DIR
from dexmani_real.config.runtime import ResolvedRuntimeConfig, resolve_runtime_config
from dexmani_real.recording.io_process import RecorderIOConfig, recorder_io_loop
from dexmani_real.robot.arm_loop import ArmLoopConfig
from dexmani_real.robot.arm_loop import arm_loop as _arm_loop
from dexmani_real.robot.hand_process import HandProcessConfig
from dexmani_real.robot.hand_process import hand_loop as _hand_loop
from dexmani_real.robot.safety import SafetyState, require_transition, transition
from dexmani_real.runtime.session import ManagedProcessGroup
from dexmani_real.runtime.supervisor import (
    print_health_summary,
    run_supervisor,
    wait_subsystem_ready,
)
from dexmani_real.sensor.camera_process import CameraLoopConfig
from dexmani_real.sensor.camera_process import camera_loop as _camera_loop
from dexmani_real.sensor.vr_receiver_process import VRReceiverConfig
from dexmani_real.sensor.vr_receiver_process import vr_loop as _vr_loop
from dexmani_real.shm.shared_storage import SharedStorage, SharedStorageConfig
from dexmani_real.teleop.config import TeleopConfig
from dexmani_real.teleop.keyboard import validate_arm_feedback, validate_hand_feedback
from dexmani_real.teleop.loop import teleop_loop
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


def _resource_provenance(repo_root: Path) -> tuple[tuple[str, str], ...]:
    """Hash static planning/calibration resources without importing a device SDK."""
    resources = {
        "arm_hand_collision_urdf_sha256": ASSET_DIR / "robots" / "xhand" / "xarm7_xhand_collision.urdf",
        "arm_hand_urdf_sha256": ASSET_DIR / "robots" / "xhand" / "xarm7_xhand_right.urdf",
        "arm_hand_srdf_sha256": ASSET_DIR / "robots" / "xhand" / "xarm7_xhand.srdf",
        "camera_calibration_sha256": repo_root / "dexmani_real" / "config" / "cameras.json",
        "vr_heading_calibration_sha256": repo_root / "dexmani_real" / "config" / "vr_transform.json",
    }
    result: list[tuple[str, str]] = []
    for name, path in resources.items():
        if not path.is_file():
            raise FileNotFoundError(f"required experiment resource is missing: {path}")
        result.append((name, hashlib.sha256(path.read_bytes()).hexdigest()))
    return tuple(sorted(result))


def _positive_float(value: str) -> float:
    """Argparse type: positive float."""
    v = float(value)
    if not math.isfinite(v) or v <= 0:
        raise argparse.ArgumentTypeError(f"must be finite and > 0, got {value}")
    return v


def _preflight_health_issues(
    shared: SharedStorage,
    runtime: Any,
    *,
    hand_enabled: bool,
    recording_enabled: bool,
    now_s: float | None = None,
    now_ns: int | None = None,
) -> list[str]:
    """Validate fresh, finite feedback before Main permits ARMED."""
    issues: list[str] = []
    if shared.error_state.value:
        issues.append("sticky error_state is set")
    if shared.estop_request.value:
        issues.append("e-stop is requested")

    # The teleop loop reuses the "policy" heartbeat/ready slots in SharedStorage
    # (shared with the learned-policy deployment path).
    enabled_heartbeats = {
        "arm": shared.arm_heartbeat_s,
        "vr": shared.vr_heartbeat_s,
        "policy": shared.policy_heartbeat_s,
    }
    if hand_enabled:
        enabled_heartbeats["hand"] = shared.hand_heartbeat_s
    if recording_enabled:
        enabled_heartbeats["camera"] = shared.camera_heartbeat_s
        enabled_heartbeats["recorder"] = shared.recorder_heartbeat_s
    heartbeat_timeouts = runtime.safety.heartbeat_timeouts
    for name, heartbeat in enabled_heartbeats.items():
        last_s = float(heartbeat.value)
        current_s = time.monotonic() if now_s is None else now_s
        if (
            not np.isfinite(last_s)
            or last_s <= 0
            or last_s > current_s
            or current_s - last_s > float(heartbeat_timeouts[name])
        ):
            issues.append(f"{name} heartbeat is missing or stale")

    arm_result = shared.arm_state_ring.read_latest()
    if arm_result is None:
        issues.append("arm feedback is unavailable")
    else:
        arm, _timestamp_ns, _sequence = arm_result
        current_ns = time.monotonic_ns() if now_ns is None else now_ns
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
        if int(arm["error_code"][0]) != 0:
            issues.append(f"arm controller error C{int(arm['error_code'][0])}")

    if hand_enabled:
        hand_result = shared.hand_state_ring.read_latest()
        if hand_result is None:
            issues.append("hand feedback is unavailable")
        else:
            hand, _timestamp_ns, _sequence = hand_result
            current_ns = time.monotonic_ns() if now_ns is None else now_ns
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

    vr_result = shared.vr_ring.read_latest()
    if vr_result is None:
        issues.append("VR hand feedback is unavailable")
    else:
        vr, _timestamp_ns, _sequence = vr_result
        local_recv_ns = int(vr["local_recv_ns"][0])
        current_ns = time.monotonic_ns() if now_ns is None else now_ns
        vr_ok = (
            np.all(np.isfinite(vr["wrist_pos"][0]))
            and np.all(np.isfinite(vr["wrist_quat_wxyz"][0]))
            and np.all(np.isfinite(vr["landmarks"][0]))
            and 0 < local_recv_ns <= current_ns
            and current_ns - local_recv_ns <= int(float(runtime.policy.vr_mapping.stale_threshold_s) * 1e9)
        )
        if not vr_ok:
            issues.append("VR hand feedback is invalid or stale")

    if recording_enabled:
        camera_result = shared.camera_ring.read_latest()
        if camera_result is None:
            issues.append("camera frame is unavailable")
        else:
            header = camera_result[0]
            source_ns = int(header["source_monotonic_ns"][0])
            current_ns = time.monotonic_ns() if now_ns is None else now_ns
            camera_ok = (
                int(header["camera_health"][0]) == 0
                and 0 < source_ns <= current_ns
                and current_ns - source_ns <= int(float(runtime.camera.max_frame_age_s) * 1e9)
            )
            if not camera_ok:
                issues.append("camera frame is unhealthy or stale")
    return issues


def _print_session_header(
    runtime: ResolvedRuntimeConfig,
    *,
    task_label: str,
    operator: str,
    hand_enabled: bool,
    recording_enabled: bool,
) -> None:
    process_labels = ["vr", "policy", "arm"]
    if recording_enabled:
        process_labels.extend(("camera", "recorder"))
    if hand_enabled:
        process_labels.append("hand")
    session_meta = []
    if task_label:
        session_meta.append(f"task={task_label}")
    if operator:
        session_meta.append(f"operator={operator}")
    session_meta.extend(
        (
            f"acc={float(runtime.arm.max_joint_acceleration_deg_per_s2)}deg/s2",
            f"speed={float(runtime.arm.max_joint_velocity_deg_per_s)}deg/s",
            f"hand={'ON' if hand_enabled else 'OFF'}",
            f"record={'ON' if recording_enabled else 'OFF'}",
            f"config={runtime.sha256[:12]}",
        )
    )
    print("=" * 60)
    print("  DexMani VR Teleop — xArm7 + XHand")
    print(f"  procs: {' | '.join(process_labels)}")
    print(f"  {'  '.join(session_meta)}")
    print("=" * 60)


def _build_processes(
    shared: SharedStorage,
    runtime: ResolvedRuntimeConfig,
    context: Any,
    *,
    repo_root: Path,
    task_label: str,
    operator: str,
    provenance: tuple[tuple[str, str], ...],
    hand_enabled: bool,
    recording_enabled: bool,
) -> list[Any]:
    policy_config = TeleopConfig.from_runtime(
        runtime,
        task_label=task_label,
        operator=operator,
        hand_urdf_path=str(ASSET_DIR / "robots" / "xhand" / "xhand_right.urdf"),
    )
    processes = [
        context.Process(
            target=_vr_loop, args=(shared, VRReceiverConfig.from_runtime(runtime)), name="vr", daemon=False
        ),
        context.Process(target=teleop_loop, args=(shared, policy_config), name="policy", daemon=False),
        context.Process(
            target=_arm_loop,
            args=(shared, ArmLoopConfig.from_runtime(runtime)),
            name="arm",
            daemon=False,
        ),
    ]
    if recording_enabled:
        camera_config = CameraLoopConfig.from_runtime(runtime)
        recorder_config = RecorderIOConfig(
            data_dir=str(repo_root / policy_config.runtime.policy.episodes_dir),
            max_frames=int(round(policy_config.runtime.policy.max_record_duration_s * policy_config.runtime.policy.control_hz)),
            control_hz=policy_config.runtime.policy.control_hz,
            min_frames=int(round(policy_config.runtime.policy.min_record_duration_s * policy_config.runtime.policy.control_hz)),
            resolved_config_sha256=runtime.sha256,
            provenance=provenance,
            writer_queue_size=int(runtime.camera.writer_queue_size),
        )
        processes.append(
            context.Process(target=_camera_loop, args=(shared, camera_config), name="camera", daemon=False)
        )
        processes.append(
            context.Process(target=recorder_io_loop, args=(shared, recorder_config), name="recorder", daemon=False)
        )
    if hand_enabled:
        processes.append(
            context.Process(
                target=_hand_loop,
                args=(shared, HandProcessConfig.from_runtime(runtime)),
                name="hand",
                daemon=False,
            )
        )
    return processes


def _readiness_checks(
    shared: SharedStorage,
    runtime: ResolvedRuntimeConfig,
    *,
    hand_enabled: bool,
    recording_enabled: bool,
) -> list[tuple[str, Any, float]]:
    timeouts = runtime.safety.readiness_timeouts_s
    checks = [
        ("arm", shared.arm_ready, float(timeouts["arm"])),
        ("vr", shared.vr_ready, float(timeouts["vr"])),
        ("policy", shared.policy_ready, float(timeouts["policy"])),
    ]
    if recording_enabled:
        checks.extend(
            (
                ("camera", shared.camera_ready, float(timeouts["camera"])),
                ("recorder", shared.recorder_ready, float(timeouts["recorder"])),
            )
        )
    if hand_enabled:
        checks.append(("hand", shared.hand_ready, float(timeouts["hand"])))
    return checks


def _heartbeat_fields(
    shared: SharedStorage,
    *,
    hand_enabled: bool,
    recording_enabled: bool,
) -> dict[str, Any]:
    fields = {
        "arm": shared.arm_heartbeat_s,
        "policy": shared.policy_heartbeat_s,
        "vr": shared.vr_heartbeat_s,
    }
    if recording_enabled:
        fields["camera"] = shared.camera_heartbeat_s
        fields["recorder"] = shared.recorder_heartbeat_s
    if hand_enabled:
        fields["hand"] = shared.hand_heartbeat_s
    return fields


def main(argv: list[str] | None = None) -> int:
    """Spawn enabled child capabilities, supervise them, and clean up."""
    parser = argparse.ArgumentParser(description="VR Teleop xArm7 + XHand with recording")
    parser.add_argument("--task", type=str, default="", help="Task label for recording metadata")
    parser.add_argument("--operator", type=str, default="", help="Operator name for recording metadata")
    parser.add_argument(
        "--acc", type=_positive_float, default=None, help="Joint max acceleration (°/s²; defaults to YAML/defaults)"
    )
    parser.add_argument(
        "--speed", type=_positive_float, default=None, help="Joint max speed (°/s; YAML/defaults if omitted)"
    )
    parser.add_argument(
        "--no-hand",
        action="store_true",
        help="Do not start XHand; use only when the physical hand is absent or secured.",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Run VR teleoperation without recording; camera and RecorderIO are not started.",
    )
    parser.add_argument("--config", type=str, default=None, help="YAML file with experiment overrides")
    parser.add_argument("--print-config", action="store_true", help="Print all config values and exit")
    args = parser.parse_args(argv)

    try:
        runtime = resolve_runtime_config(
            yaml_path=args.config,
            cli_overrides={
                "arm.max_joint_acceleration_deg_per_s2": args.acc,
                "arm.max_joint_velocity_deg_per_s": args.speed,
                "policy.hand_enabled": False if args.no_hand else None,
                "policy.recording_enabled": False if args.no_record else None,
            },
        )
    except (KeyError, OSError, TypeError, UnicodeError, ValueError, yaml.YAMLError) as exc:
        parser.error(f"invalid experiment config: {exc}")
    if not bool(runtime.policy.hand_enabled) and not args.no_hand:
        parser.error("policy.hand_enabled=false requires explicit --no-hand confirmation")
    if args.print_config:
        print(runtime.canonical_yaml, end="")
        print(f"sha256={runtime.sha256}")
        return 0

    try:
        return run_teleop_experiment(
            runtime,
            task_label=args.task,
            operator=args.operator,
            allow_no_hand=args.no_hand,
        )
    except Exception:
        logger.error("teleoperation startup failed before lifecycle ownership was established", exc_info=True)
        return 1


def run_teleop_experiment(
    runtime: ResolvedRuntimeConfig,
    *,
    task_label: str = "",
    operator: str = "",
    allow_no_hand: bool = False,
) -> int:
    """Run one resolved teleoperation experiment lifecycle."""
    hand_enabled = bool(runtime.policy.hand_enabled)
    recording_enabled = bool(runtime.policy.recording_enabled)
    if not hand_enabled and not allow_no_hand:
        logger.error("disabled hand requires explicit allow_no_hand acknowledgement")
        return 1

    repo_root = Path(__file__).resolve().parents[1]
    vr_transform_path = repo_root / "dexmani_real" / "config" / "vr_transform.json"
    if not vr_transform_path.is_file():
        print(f"Preflight failed: VR transform is missing: {vr_transform_path}")
        return 1
    try:
        provenance = _resource_provenance(repo_root) if recording_enabled else ()
    except (FileNotFoundError, OSError) as exc:
        print(f"Preflight failed: {exc}")
        return 1

    _print_session_header(
        runtime,
        task_label=task_label,
        operator=operator,
        hand_enabled=hand_enabled,
        recording_enabled=recording_enabled,
    )

    ctx = mp.get_context("spawn")
    shared = SharedStorage.create(
        prefix=f"dexmani_collect_{os.getpid()}",
        config=SharedStorageConfig.from_runtime(runtime),
        mp_context=ctx,
    )
    procs: list[Any] = []
    group: ManagedProcessGroup | None = None
    shared_closed = False
    try:
        procs = _build_processes(
            shared,
            runtime,
            ctx,
            repo_root=repo_root,
            task_label=task_label,
            operator=operator,
            provenance=provenance,
            hand_enabled=hand_enabled,
            recording_enabled=recording_enabled,
        )
        group = ManagedProcessGroup(
            shared,
            procs,
            graceful_timeout_s=float(runtime.safety.shutdown_timeout_s),
        )
        require_transition(shared, SafetyState.DISARMED)
        group.start()

        ready_checks = _readiness_checks(
            shared,
            runtime,
            hand_enabled=hand_enabled,
            recording_enabled=recording_enabled,
        )

        for name, _event, timeout in ready_checks:
            if name == "vr":
                print(f"\n  Waiting for VR connection (up to {timeout}s) — put on Quest headset...", flush=True)

        if not wait_subsystem_ready(shared, ready_checks, procs):
            shared.error_state.value = True
            require_transition(shared, SafetyState.FAULT)
            shutdown_report = group.shutdown()
            shared_closed = shutdown_report.shared_closed
            return 1

        for name, _event, _timeout in ready_checks:
            if name == "vr":
                print(f"  VR connected", flush=True)
            else:
                print(f"  {name}: ready", flush=True)

        print_health_summary(shared)
        health_issues = _preflight_health_issues(
            shared,
            runtime,
            hand_enabled=hand_enabled,
            recording_enabled=recording_enabled,
        )
        if health_issues:
            for issue in health_issues:
                logger.error("preflight health failed: %s", issue)
            shared.error_state.value = True
            require_transition(shared, SafetyState.FAULT)
            shutdown_report = group.shutdown()
            shared_closed = shutdown_report.shared_closed
            return 1

        require_transition(shared, SafetyState.ARMED)
        begin_label = "teleop+record" if recording_enabled else "teleop"
        print(
            f"\nAll subsystems ready — safety=ARMED({int(SafetyState.ARMED)})\n"
            f"Controls: B={begin_label}  C=pause  S=stop  D=discard  H=home  Q=quit  ESC=estop\n"
        )

        process_names = [process.name for process in procs]
        heartbeat_fields = _heartbeat_fields(
            shared,
            hand_enabled=hand_enabled,
            recording_enabled=recording_enabled,
        )

        start_time = time.monotonic()
        exit_reason, normal_exit = run_supervisor(
            shared,
            procs,
            process_names,
            heartbeat_fields,
            heartbeat_timeouts_s=dict(runtime.safety.heartbeat_timeouts),
            supervisor_hz=float(runtime.safety.supervisor_hz),
        )

        shutdown_report = group.shutdown(disarm_if_clean=normal_exit)
        shared_closed = shutdown_report.shared_closed
        clean_exit = normal_exit and shutdown_report.clean
        if normal_exit and not clean_exit:
            logger.error("verified shutdown invalidated the clean supervisor exit: %s", shutdown_report)

        runtime_m = (time.monotonic() - start_time) / 60.0
        safety_name = (
            SafetyState(shutdown_report.safety_state).name if shutdown_report.safety_state is not None else "UNKNOWN"
        )
        print(f"\n── Session End ──")
        print(
            f"  exit_reason={exit_reason}  runtime={runtime_m:.1f}min  safety={safety_name}  "
            f"supervisor_normal={normal_exit}  clean={clean_exit}"
        )
        print("──")
        return 0 if clean_exit else 1

    except Exception:
        logger.error("teleoperation experiment failed", exc_info=True)
        shared.error_state.value = True
        require_transition(shared, SafetyState.FAULT)
        return 1
    finally:
        # RecorderIO may still be validating and publishing an episode transaction.
        if group is not None:
            try:
                shutdown_report = group.shutdown()
                shared_closed = shutdown_report.shared_closed
            except RuntimeError:
                logger.critical("child process remains alive; leaving SharedStorage linked", exc_info=True)
                raise
        if group is None and not shared_closed:
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
