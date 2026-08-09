#!/usr/bin/env python3
"""VR teleop xArm7 + XHand with recording — canonical data-collection entry point.

Spawn-only capability model: camera, VR, PolicyCoordinator, arm, optional
hand, and optional RecorderIO.
Safety: DISARMED/ARMED/RUNNING/FAULT state machine + enabled-capability heartbeats.
Recording: additive HDF5 schema v15 via TimestampAlignedBuffer → EpisodeRecorder.

Usage:
    python examples/real/vr_teleop_hand_record.py [--task T] [--operator O] [--acc A] [--speed S]
                     [--no-hand] [--no-record] [--config PATH] [--print-config]
Controls:
    B=teleop(+record when enabled)  C=pause  S=stop/save  D=discard  H=home  Q=quit  ESC=estop
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any

# Ensure repo root is on sys.path before imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dexmani_real import ASSET_DIR
from dexmani_real.config.runtime import resolve_runtime_config
from dexmani_real.policy.vr_teleop_policy import PolicyConfig, policy_loop
from dexmani_real.recording.io_process import RecorderIOConfig, recorder_io_loop
from dexmani_real.robot.arm_loop import ArmLoopConfig
from dexmani_real.robot.arm_loop import arm_loop as _arm_loop
from dexmani_real.robot.hand_process import HandProcessConfig
from dexmani_real.robot.hand_process import hand_loop as _hand_loop
from dexmani_real.robot.safety import SafetyState, transition
from dexmani_real.sensor.camera_process import CameraLoopConfig
from dexmani_real.sensor.camera_process import camera_loop as _camera_loop
from dexmani_real.sensor.vr_receiver_process import VRReceiverConfig
from dexmani_real.sensor.vr_receiver_process import vr_loop as _vr_loop
from dexmani_real.shm.shared_storage import (
    SharedStorage,
    SharedStorageConfig,
    print_health_summary,
    run_supervisor,
    shutdown_processes,
    wait_subsystem_ready,
)
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
        if path.is_file():
            result.append((name, hashlib.sha256(path.read_bytes()).hexdigest()))
        else:
            result.append((name, "missing"))
    return tuple(sorted(result))


def _positive_float(value: str) -> float:
    """Argparse type: positive float."""
    v = float(value)
    if v <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {value}")
    return v


def main() -> None:
    """Spawn enabled child capabilities, supervise them, and clean up."""
    parser = argparse.ArgumentParser(description="VR Teleop xArm7 + XHand with recording")
    parser.add_argument("--task", type=str, default="", help="Task label for recording metadata")
    parser.add_argument("--operator", type=str, default="", help="Operator name for recording metadata")
    parser.add_argument(
        "--acc", type=_positive_float, default=None, help="Joint max acceleration (°/s²; defaults to JSON/defaults)"
    )
    parser.add_argument(
        "--speed", type=_positive_float, default=None, help="Joint max speed (°/s; JSON/defaults if omitted)"
    )
    parser.add_argument("--no-hand", action="store_true", help="Disable hand retargeting (hold-position only)")
    parser.add_argument("--no-record", action="store_true", help="Run VR teleoperation without RecorderIO")
    parser.add_argument("--config", type=str, default=None, help="JSON file with nested runtime config overrides")
    parser.add_argument("--print-config", action="store_true", help="Print all config values and exit")
    args = parser.parse_args()

    runtime = resolve_runtime_config(
        json_path=args.config,
        cli_overrides={
            "arm.max_joint_acceleration_deg_per_s2": args.acc,
            "arm.max_joint_velocity_deg_per_s": args.speed,
            "policy.hand_enabled": False if args.no_hand else None,
            "policy.recording_enabled": False if args.no_record else None,
        },
    )
    hand_enabled = bool(runtime.policy.hand_enabled)
    recording_enabled = bool(runtime.policy.recording_enabled)
    effective_acc = float(runtime.arm.max_joint_acceleration_deg_per_s2)
    effective_speed = float(runtime.arm.max_joint_velocity_deg_per_s)

    if args.print_config:
        print(json.dumps(json.loads(runtime.canonical_json), indent=2, ensure_ascii=False, sort_keys=True))
        print(f"sha256={runtime.sha256}")
        sys.exit(0)

    _proc_labels = ["camera", "vr", "policy", "arm"]
    if recording_enabled:
        _proc_labels.append("recorder")
    if hand_enabled:
        _proc_labels.append("hand")
    print("=" * 60)
    print("  DexMani VR Teleop — xArm7 + XHand")
    print(f"  procs: {' | '.join(_proc_labels)}")
    _meta = []
    if args.task:
        _meta.append(f"task={args.task}")
    if args.operator:
        _meta.append(f"operator={args.operator}")
    _meta.append(f"acc={effective_acc}deg/s2")
    _meta.append(f"speed={effective_speed}deg/s")
    _meta.append(f"hand={'ON' if hand_enabled else 'OFF'}")
    _meta.append(f"record={'ON' if recording_enabled else 'OFF'}")
    _meta.append(f"config={runtime.sha256[:12]}")
    print(f"  {'  '.join(_meta)}")
    print("=" * 60)

    # 1. SharedStorage
    ctx = mp.get_context("spawn")
    shared = SharedStorage.create(
        prefix="dexmani",
        config=SharedStorageConfig.from_runtime(runtime),
        mp_context=ctx,
    )
    procs: list[Any] = []
    try:
        # 2. Policy config
        policy_cfg = PolicyConfig.from_runtime(
            runtime,
            task_label=args.task,
            operator=args.operator,
            hand_urdf_path=str(ASSET_DIR / "robots" / "xhand" / "xhand_right.urdf"),
        )

        # 2b. Arm config (must match PolicyConfig speed/acc)
        arm_cfg = ArmLoopConfig.from_runtime(runtime)
        camera_cfg = CameraLoopConfig.from_runtime(runtime)
        vr_cfg = VRReceiverConfig.from_runtime(runtime)
        hand_cfg = HandProcessConfig.from_runtime(runtime)
        repo_root = Path(__file__).resolve().parents[2]
        recorder_cfg = (
            RecorderIOConfig(
                data_dir=str(repo_root / policy_cfg.episodes_dir),
                max_frames=int(round(policy_cfg.max_record_seconds * policy_cfg.control_hz)),
                control_hz=policy_cfg.control_hz,
                min_frames=int(round(policy_cfg.min_record_seconds * policy_cfg.control_hz)),
                resolved_config_json=runtime.canonical_json,
                resolved_config_sha256=runtime.sha256,
                provenance=_resource_provenance(repo_root),
                writer_queue_size=int(runtime.camera.writer_queue_size),
            )
            if recording_enabled
            else None
        )

        # 3. Spawn
        procs = [
            ctx.Process(target=_camera_loop, args=(shared, camera_cfg), name="cam", daemon=False),
            ctx.Process(target=_vr_loop, args=(shared, vr_cfg), name="vr", daemon=False),
            ctx.Process(target=policy_loop, args=(shared, policy_cfg), name="pol", daemon=False),
            ctx.Process(target=_arm_loop, args=(shared, arm_cfg), name="arm", daemon=False),
        ]
        if recorder_cfg is not None:
            procs.append(ctx.Process(target=recorder_io_loop, args=(shared, recorder_cfg), name="rec", daemon=False))
        if hand_enabled:
            procs.append(ctx.Process(target=_hand_loop, args=(shared, hand_cfg), name="hand", daemon=False))
        for p in procs:
            p.start()

        # 4. Wait for ready
        # vr_loop defers vr_ready until the first HTS event arrives (not just TCP
        # connect), so the 120 s timeout here doubles as the "put on headset" grace
        # period.  Arm/camera/hand are ready within seconds.
        transition(shared, SafetyState.DISARMED)

        _ready_checks: list[tuple[str, Any, float]] = [
            ("arm", shared.arm_ready, 15),
            ("camera", shared.camera_ready, 15),
            ("vr", shared.vr_ready, 120),
            ("policy", shared.policy_ready, 60),
        ]
        if recording_enabled:
            _ready_checks.append(("recorder", shared.recorder_ready, 15))
        if hand_enabled:
            _ready_checks.append(("hand", shared.hand_ready, 15))

        # VR-specific pre-wait message (caller responsibility — wait_subsystem_ready
        # is a pure polling loop).
        for name, _ev, timeout in _ready_checks:
            if name == "vr":
                print(f"\n  Waiting for VR connection (up to {timeout}s) — put on Quest headset...", flush=True)

        if not wait_subsystem_ready(shared, _ready_checks, procs):
            shutdown_processes(shared, procs, graceful_timeout_s=65.0)
            return

        for name, _ev, _timeout in _ready_checks:
            if name == "vr":
                print(f"  VR connected", flush=True)
            else:
                print(f"  {name}: ready", flush=True)

        # 4b. Pre-flight health summary (ring-based, no direct hardware access)
        print_health_summary(shared)

        # All subsystems ready — transition to ARMED.
        transition(shared, SafetyState.ARMED)
        _begin_label = "teleop+record" if recording_enabled else "teleop"
        print(
            f"\nAll subsystems ready — safety=ARMED({int(SafetyState.ARMED)})\n"
            f"Controls: B={_begin_label}  C=pause  S=stop  D=discard  H=home  Q=quit  ESC=estop\n"
        )

        # 5. Supervisor (heartbeat + process monitor)
        _proc_names = ["camera", "vr", "policy", "arm"]
        if recording_enabled:
            _proc_names.append("recorder")
        if hand_enabled:
            _proc_names.append("hand")
        _heartbeat_fields = {
            "arm": shared.arm_heartbeat_s,
            "policy": shared.policy_heartbeat_s,
            "vr": shared.vr_heartbeat_s,
            "camera": shared.camera_heartbeat_s,
        }
        if recording_enabled:
            _heartbeat_fields["recorder"] = shared.recorder_heartbeat_s
        if hand_enabled:
            _heartbeat_fields["hand"] = shared.hand_heartbeat_s

        # Seed any heartbeats still at 0.0 (processes still initializing) to
        # current time so they get the full timeout window from ARMED, not from
        # process start.  policy_loop is the main beneficiary — it does heavy
        # init (collision model, hand retargeter) after all Ready events but
        # before its main loop starts ticking the heartbeat.
        _now = time.monotonic()
        for _hb in _heartbeat_fields.values():
            if _hb.value == 0.0:
                _hb.value = _now

        _start_time = time.monotonic()
        _exit_reason, normal_exit = run_supervisor(
            shared,
            procs,
            _proc_names,
            _heartbeat_fields,
            heartbeat_timeouts_s=dict(runtime.safety.heartbeat_timeouts),
            supervisor_hz=float(runtime.safety.supervisor_hz),
        )

        transition(shared, SafetyState.DISARMED)

        shutdown_processes(shared, procs, graceful_timeout_s=65.0)

        # Exit summary
        _runtime_m = (time.monotonic() - _start_time) / 60.0
        _safety_name = SafetyState(shared.safety_state.value).name
        print(f"\n── Session End ──")
        print(f"  exit_reason={_exit_reason}  runtime={_runtime_m:.1f}min  safety={_safety_name}  normal={normal_exit}")
        print("──")

    finally:
        # RecorderIO may need up to 60 s to drain, validate, fsync, and rename.
        # Never unlink IPC while any successfully started child can still use it.
        started = [process for process in procs if process.pid is not None]
        if any(process.is_alive() for process in started):
            try:
                shutdown_processes(shared, started, graceful_timeout_s=65.0)
            except RuntimeError:
                logger.critical("child process remains alive; leaving SharedStorage linked", exc_info=True)
        if not any(process.is_alive() for process in started):
            try:
                shared.close()
            except Exception:
                logger.warning("SharedStorage cleanup failed", exc_info=True)


if __name__ == "__main__":
    main()
