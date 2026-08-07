#!/usr/bin/env python3
"""VR teleop xArm7 + XHand with recording — canonical data-collection entry point.

5-process SharedStorage model: camera, VR, policy, arm (Mode 6), hand.
Safety: DISARMED/ARMED/RUNNING/FAULT state machine + 5 per-process heartbeats.
Recording: HDF5 schema v11 via TimestampAlignedBuffer → EpisodeRecorder.

Usage:
    python examples/real/vr_teleop_hand_record.py [--task T] [--operator O] [--acc A] [--speed S] [--no-hand]
Controls:
    B=teleop+record  C=pause  S=save  D=discard  H=home  Q=quit  ESC=estop
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

# Ensure repo root is on sys.path before imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dexmani_real import ASSET_DIR
from dexmani_real.config.defaults import arm, camera, hand, policy, safety, vr
from dexmani_real.policy.vr_teleop_policy import PolicyConfig, policy_loop
from dexmani_real.robot.arm_loop import ArmLoopConfig
from dexmani_real.robot.arm_loop import arm_loop as _arm_loop
from dexmani_real.robot.hand_process import hand_loop as _hand_loop
from dexmani_real.robot.safety import SafetyState, transition
from dexmani_real.sensor.camera_process import camera_loop as _camera_loop
from dexmani_real.sensor.vr_receiver_process import vr_loop as _vr_loop
from dexmani_real.shm.shared_storage import (
    SharedStorage,
    hand_home_converge,
    print_health_summary,
    run_supervisor,
    send_arm_home,
    shutdown_processes,
    wait_subsystem_ready,
)
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Spawn 5 child processes, monitor, clean up."""
    parser = argparse.ArgumentParser(description="VR Teleop xArm7 + XHand with recording")
    parser.add_argument("--task", type=str, default="", help="Task label for recording metadata")
    parser.add_argument("--operator", type=str, default="", help="Operator name for recording metadata")
    parser.add_argument("--acc", type=float, default=900.0, help="Joint max acceleration (°/s², default: 900)")
    parser.add_argument("--speed", type=float, default=120.0, help="Joint max speed (°/s, default: 120)")
    parser.add_argument("--no-hand", action="store_true", help="Disable hand retargeting (hold-position only)")
    parser.add_argument("--config", type=str, default=None, help="JSON file with config overrides (flat fields only)")
    parser.add_argument("--print-config", action="store_true", help="Print all config values and exit")
    args = parser.parse_args()

    if args.config:
        from dexmani_real.config.defaults import load_config_json

        load_config_json(args.config)

    if args.print_config:
        import dataclasses

        for label, obj in (
            ("arm", arm),
            ("hand", hand),
            ("policy", policy),
            ("vr", vr),
            ("safety", safety),
            ("camera", camera),
        ):
            print(f"[{label}]")
            for k, v in dataclasses.asdict(obj).items():
                print(f"  {k} = {v}")
        sys.exit(0)

    _procs = ["camera", "vr", "policy", "arm"]
    if not args.no_hand:
        _procs.append("hand")
    print("=" * 60)
    print("  DexMani VR Teleop — xArm7 + XHand")
    print(f"  procs: {' | '.join(_procs)}")
    _meta = []
    if args.task:
        _meta.append(f"task={args.task}")
    if args.operator:
        _meta.append(f"operator={args.operator}")
    _meta.append(f"acc={args.acc}deg/s2")
    _meta.append(f"speed={args.speed}deg/s")
    _meta.append(f"hand={'OFF' if args.no_hand else 'ON'}")
    print(f"  {'  '.join(_meta)}")
    print("=" * 60)

    # 1. SharedStorage
    shared = SharedStorage.create(prefix="dexmani")

    # 2. Policy config
    policy_cfg = PolicyConfig(
        task_label=args.task,
        operator=args.operator,
        joint_max_acc_deg_s2=args.acc,
        joint_max_speed_deg_s=args.speed,
        hand_urdf_path=str(ASSET_DIR / "robots" / "xhand" / "xhand_right.urdf"),
    )
    if args.no_hand:
        policy_cfg.hand_enabled = False

    # 2b. Arm config (must match PolicyConfig speed/acc)
    arm_cfg = ArmLoopConfig(
        joint_max_speed_rad_per_s=float(np.deg2rad(args.speed)),
        joint_max_acc_rad_per_s2=float(np.deg2rad(args.acc)),
    )

    # 3. Spawn
    procs: list[mp.Process] = [
        mp.Process(target=_camera_loop, args=(shared,), name="cam", daemon=True),
        mp.Process(target=_vr_loop, args=(shared,), name="vr", daemon=True),
        mp.Process(target=policy_loop, args=(shared, policy_cfg), name="pol", daemon=True),
        mp.Process(target=_arm_loop, args=(shared, arm_cfg), name="arm", daemon=True),
    ]
    if not args.no_hand:
        procs.append(mp.Process(target=_hand_loop, args=(shared,), name="hand", daemon=True))
    for p in procs:
        p.start()

    # 4. Wait for ready
    # vr_loop defers vr_ready until the first HTS event arrives (not just TCP
    # connect), so the 120 s timeout here doubles as the "put on headset" grace
    # period.  Arm/camera/hand are ready within seconds.
    transition(shared, SafetyState.DISARMED)

    _ready_checks: list[tuple[str, object, float]] = [
        ("arm", shared.arm_ready, 15),
        ("camera", shared.camera_ready, 15),
        ("vr", shared.vr_ready, 120),
    ]
    if not args.no_hand:
        _ready_checks.append(("hand", shared.hand_ready, 15))

    # VR-specific pre-wait message (caller responsibility — wait_subsystem_ready
    # is a pure polling loop).
    for name, _ev, timeout in _ready_checks:
        if name == "vr":
            print(f"\n  Waiting for VR connection (up to {timeout}s) — put on Quest headset...", flush=True)

    if not wait_subsystem_ready(shared, _ready_checks, procs):
        shutdown_processes(shared, procs)
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
    print(
        f"\nAll subsystems ready — safety=ARMED({int(SafetyState.ARMED)})\n"
        "Controls: B=teleop+record  C=pause  S=save  D=discard  H=home  Q=quit  ESC=estop\n"
    )

    # 5. Supervisor (heartbeat + process monitor)
    _proc_names = ["camera", "vr", "policy", "arm"]
    if not args.no_hand:
        _proc_names.append("hand")
    _heartbeat_fields = {
        "arm": shared.arm_heartbeat_s,
        "policy": shared.policy_heartbeat_s,
        "vr": shared.vr_heartbeat_s,
        "camera": shared.camera_heartbeat_s,
    }
    if not args.no_hand:
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
        shared, procs, _proc_names, _heartbeat_fields,
    )

    transition(shared, SafetyState.DISARMED)

    # Post-loop: offer return_home on normal exit.
    # Skip when Policy initiated the quit (quit_requested=True) — Policy
    # already handled the [H] return_home / [Q] quit prompt while all
    # processes were still alive.
    if normal_exit and not shared.error_state.value and not shared.quit_requested.value:
        _post_loop_home(shared)

    shutdown_processes(shared, procs)

    # Exit summary
    _runtime_m = (time.monotonic() - _start_time) / 60.0
    _final_safety = shared.safety_state.value
    print(f"\n── Session End ──")
    print(
        f"  exit_reason={_exit_reason}  runtime={_runtime_m:.1f}min  "
        f"safety={_final_safety}  normal={normal_exit}"
    )
    print("──")


def _post_loop_home(shared: SharedStorage) -> None:
    """Offer return_home after normal exit — hand first, then arm."""
    from dexmani_real.teleop.keyboard import ControlSignal, KeyboardHandler

    HAND_HOME_QPOS = np.deg2rad(np.array(hand.home_qpos_deg, dtype=np.float64))

    kb = KeyboardHandler()
    kb.start()
    try:
        print("\n[H] return_home  [Q] quit  (60s timeout)")
        _deadline = time.perf_counter() + 60.0
        while time.perf_counter() < _deadline:
            for sig in kb.poll(timeout=0.1):
                if sig == ControlSignal.HOME:
                    print("  H: return_home")

                    # Step 1: Hand home first
                    # hand_loop is still running (is_running not set yet).
                    hand_home_converge(shared, HAND_HOME_QPOS, heartbeat=False, check_is_running=False, verbose=True)

                    # Step 2: Arm home — no planner available post-exit.
                    _home_qpos = np.array(arm.home_qpos, dtype=np.float64)
                    send_arm_home(
                        shared, _home_qpos,
                        planner=None, heartbeat=False, converge_timeout_s=10.0, verbose=True,
                    )
                    print("  [Q] quit")
                if sig in (ControlSignal.QUIT, ControlSignal.EMERGENCY_STOP):
                    break
            else:
                continue
            break
        else:
            print("  timeout — auto exit")
    finally:
        kb.stop()


if __name__ == "__main__":
    main()
