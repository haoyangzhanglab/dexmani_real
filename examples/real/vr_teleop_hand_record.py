#!/usr/bin/env python3
"""VR teleop xArm7 + XHand with recording — canonical data-collection entry point.

5 child processes + thin Main orchestrator. All data exchange through SharedStorage.

**This is the PRIMARY data-collection entry point.** Supports:
- Arm teleop via VR wrist → IK (EMA + workspace clamp + delta clamp)
- Hand teleop via VR landmarks → DexPilot NLP retargeting
- Safety: DISARMED/ARMED/RUNNING/FAULT state machine + 5 per-process heartbeats
- Recording: TimestampAlignedBuffer → HDF5 (schema v11, arm+hand+EEF+camera)
- Voice feedback (TTS audio prompts for headset-blind operation)
- Camera metadata (intrinsics, depth_scale, serial) propagated to HDF5 /meta

Architecture:
    Main (~200 lines) — spawns 5 processes, monitors is_running + heartbeats
      │
      ├─ CameraProcess ──camera_ring──┐
      ├─ VRProcess ──────vr_ring─────┤
      │                               ▼
      ├─ PolicyProcess ──arm_action_q──→ arm_loop
      │                ──hand_cmd_ring─→ hand_loop
      │                ◄──arm_state_ring, hand_state_ring, hand_tactile_ring
      │                owns EpisodeRecorder (single-clock recording)
      │
      ├─ arm_loop (Mode 6, 30Hz) — reads arm_action_q, servos xArm7
      └─ hand_loop (30Hz) — reads hand_cmd_ring, servos XHand

Usage:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate real_robot
    python examples/real/vr_teleop_hand_record.py
    python examples/real/vr_teleop_hand_record.py --task pick_place --operator alice
    python examples/real/vr_teleop_hand_record.py --acc 900 --speed 120
    python examples/real/vr_teleop_hand_record.py --no-hand

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

from dexmani_real.config.defaults import arm, safety
from dexmani_real.policy.vr_teleop_policy import PolicyConfig, policy_loop
from dexmani_real.robot.inner_loop import ArmInnerLoopConfig, arm_loop as _arm_loop
from dexmani_real.robot.hand_process import hand_loop as _hand_loop
from dexmani_real.sensor.camera_process import camera_loop as _camera_loop
from dexmani_real.sensor.vr_receiver_process import vr_loop as _vr_loop
from dexmani_real.shm.shared_storage import HOME_SENTINEL, SharedStorage
from dexmani_real.robot.safety import SafetyState, transition
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def main() -> None:
    """Spawn 5 child processes, monitor, clean up."""
    parser = argparse.ArgumentParser(description="VR Teleop xArm7 + XHand with recording")
    parser.add_argument("--task", type=str, default="", help="Task label for recording metadata")
    parser.add_argument("--operator", type=str, default="", help="Operator name for recording metadata")
    parser.add_argument("--acc", type=float, default=900.0, help="Joint max acceleration (°/s², default: 900)")
    parser.add_argument("--speed", type=float, default=120.0, help="Joint max speed (°/s, default: 120)")
    parser.add_argument("--no-hand", action="store_true", help="Disable hand retargeting (hold-position only)")
    args = parser.parse_args()

    print("=" * 60)
    print("VR Teleop xArm7 + XHand (canonical entry point)")
    print("5 processes: camera | vr | policy | arm | hand")
    if args.task:
        print(f"  task: {args.task}")
    if args.operator:
        print(f"  operator: {args.operator}")
    print(f"  acc: {args.acc}°/s²  speed: {args.speed}°/s  hand: {'OFF' if args.no_hand else 'ON'}")
    print("=" * 60)

    # ── 1. SharedStorage ──
    shared = SharedStorage.create(prefix="dexmani")

    # ── 2. Policy config ──
    policy_cfg = PolicyConfig(
        task_label=args.task,
        operator=args.operator,
        joint_max_acc_deg_s2=args.acc,
        joint_max_speed_deg_s=args.speed,
    )
    if args.no_hand:
        policy_cfg.hand_enabled = False

    # ── 2b. Arm config (must match PolicyConfig speed/acc) ──
    arm_cfg = ArmInnerLoopConfig(
        joint_max_speed_rad_per_s=float(np.deg2rad(args.speed)),
        joint_max_acc_rad_per_s2=float(np.deg2rad(args.acc)),
    )

    # ── 3. Spawn ──
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

    # ── 4. Wait for ready ──
    transition(shared, SafetyState.DISARMED)

    _ready_checks: list[tuple[str, object, float]] = [
        ("arm", shared.arm_ready, 15),
        ("camera", shared.camera_ready, 15),
        ("vr", shared.vr_ready, 15),
    ]
    if not args.no_hand:
        _ready_checks.append(("hand", shared.hand_ready, 15))
    for name, ev, timeout in _ready_checks:
        if not ev.wait(timeout=timeout):
            logger.error("%s startup failed: ready-event timeout after %ds", name, timeout)
            shared.is_running.value = False
            _shutdown(procs, shared)
            return

    # ── 4b. Pre-flight health summary (ring-based, no direct hardware access) ──
    _print_health_summary(shared)

    # All subsystems ready — transition to ARMED.
    transition(shared, SafetyState.ARMED)
    print(f"\nAll subsystems ready — 安全状态: ARMED ({int(SafetyState.ARMED)})")
    print("Controls: B=teleop+record C=pause S=save D=discard H=home Q=quit ESC=estop\n")

    # ── 5. Supervisor (heartbeat + process monitor) ──
    _all_names = ["arm", "policy", "vr", "camera"]
    if not args.no_hand:
        _all_names.append("hand")
    PROC_NAMES = _all_names
    HEARTBEAT_FIELDS = {
        "arm": shared.arm_heartbeat_s,
        "policy": shared.policy_heartbeat_s,
        "vr": shared.vr_heartbeat_s,
        "camera": shared.camera_heartbeat_s,
    }
    if not args.no_hand:
        HEARTBEAT_FIELDS["hand"] = shared.hand_heartbeat_s

    normal_exit = False
    try:
        while True:
            # 0. Normal exit — policy set is_running=False (Q key)
            if not shared.is_running.value:
                normal_exit = True
                logger.info("Normal exit (is_running=False)")
                break

            # 1. Process aliveness
            for p, name in zip(procs, PROC_NAMES):
                if not p.is_alive():
                    if normal_exit:
                        logger.info("%s process exited (normal shutdown)", name)
                    else:
                        logger.error("%s process died — FAULT", name)
                        transition(shared, SafetyState.FAULT)
                    break
            if shared.safety_state.value == int(SafetyState.FAULT):
                break

            # 2. Error state (sticky latch from arm/hand)
            if shared.error_state.value:
                logger.error("error_state set — transitioning to FAULT")
                transition(shared, SafetyState.FAULT)
                break

            # 3. Heartbeat timeouts
            now = time.monotonic()
            for name in PROC_NAMES:
                last_hb = float(HEARTBEAT_FIELDS[name].value)
                age_s = now - last_hb if last_hb > 0 else float("inf")
                timeout_s = float(safety.heartbeat_timeouts[name])
                if age_s > timeout_s:
                    logger.error("%s heartbeat timeout: %.1fs > %.1fs — FAULT", name, age_s, timeout_s)
                    transition(shared, SafetyState.FAULT)
                    break
            if shared.safety_state.value == int(SafetyState.FAULT):
                break

            time.sleep(0.1)  # 10Hz supervisor

    except KeyboardInterrupt:
        normal_exit = True
        shared.is_running.value = False
    finally:
        transition(shared, SafetyState.DISARMED)

        # ── Post-loop: offer return_home on normal exit ──
        if normal_exit and not shared.error_state.value:
            _post_loop_home(shared)

        _shutdown(procs, shared)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def _print_health_summary(shared: SharedStorage) -> None:
    """Print a pre-flight style health summary from ring data."""
    print("\n── 硬件健康检查 ──")

    # Arm
    arm_result = shared.arm_state_ring.read_latest()
    if arm_result is not None:
        arm_data, _, _ = arm_result
        arm_connected = bool(arm_data["connected"][0])
        arm_error = int(arm_data["error_code"][0])
        arm_qpos = np.asarray(arm_data["qpos"][0], dtype=np.float64)
        arm_ok = arm_connected and arm_error == 0 and np.all(np.isfinite(arm_qpos))
        print(f"  arm:  {'OK' if arm_ok else 'FAIL'}  connected={arm_connected}  error={arm_error}  "
              f"qpos_finite={'OK' if np.all(np.isfinite(arm_qpos)) else 'FAIL'}")
    else:
        print("  arm:  (no data yet)")

    # Hand
    hand_result = shared.hand_state_ring.read_latest()
    if hand_result is not None:
        hand_data, _, _ = hand_result
        hand_connected = bool(hand_data["connected"][0])
        hand_error = bool(hand_data["error_state"][0])
        hand_qpos_stale = bool(hand_data["qpos_stale"][0])
        hand_qpos = np.asarray(hand_data["qpos"][0], dtype=np.float64)
        hand_ok = hand_connected and not hand_error and np.all(np.isfinite(hand_qpos))
        stale_note = " (stale!)" if hand_qpos_stale else ""
        print(f"  hand: {'OK' if hand_ok else 'FAIL'}  connected={hand_connected}  "
              f"error={hand_error}  qpos_finite={'OK' if np.all(np.isfinite(hand_qpos)) else 'FAIL'}{stale_note}")
    else:
        print("  hand: (no data yet)")

    # VR
    vr_result = shared.vr_ring.read_latest()
    if vr_result is not None:
        vr_data, _, _ = vr_result
        vr_age_s = (time.monotonic_ns() - int(vr_data["local_recv_ns"][0])) / 1e9 if vr_data["local_recv_ns"][0] > 0 else -1
        print(f"  vr:   OK  age={vr_age_s:.1f}s  seq={int(vr_data['sequence_id'][0])}")
    else:
        print("  vr:   (no data yet)")

    # Camera
    cam_serial_bytes = bytes(shared.camera_serial).rstrip(b"\x00")
    if cam_serial_bytes:
        print(f"  cam:  OK  serial={cam_serial_bytes.decode()}")
    elif shared.camera_heartbeat_s.value > 0:
        print("  cam:  OK (serial unknown)")
    else:
        print("  cam:  (no data yet)")

    print("──")
    sys.stdout.flush()


def _post_loop_home(shared: SharedStorage) -> None:
    """Offer return_home via HOME_SENTINEL after normal exit."""
    from dexmani_real.teleop.control.keyboard import ControlSignal, KeyboardHandler

    kb = KeyboardHandler()
    kb.start()
    try:
        print("\n按 H 执行 return_home，或按 Q 直接退出...")
        _deadline = time.perf_counter() + 60.0
        while time.perf_counter() < _deadline:
            for sig in kb.poll(timeout=0.1):
                if sig == ControlSignal.HOME:
                    print("\nH: return_home")
                    try:
                        shared.arm_action_q.put(HOME_SENTINEL, timeout=2.0)
                    except Exception:
                        print("  ⚠ Arm queue full — arm may have already exited")
                        break
                    # Wait for arm to pick up and execute homing (~8s typical).
                    _home_wait = time.monotonic() + 30.0
                    while time.monotonic() < _home_wait:
                        _hb = shared.arm_heartbeat_s.value
                        if _hb > 0 and time.monotonic() - _hb < 3.0:
                            time.sleep(0.5)
                        else:
                            break
                    print("按 Q 退出...")
                if sig in (ControlSignal.QUIT, ControlSignal.EMERGENCY_STOP):
                    break
            else:
                continue
            break
        else:
            print("\n  超时，自动退出")
    finally:
        kb.stop()


def _shutdown(procs: list, shared: SharedStorage) -> None:
    """Graceful shutdown: signal → join → terminate."""
    shared.is_running.value = False
    for p in procs:
        p.join(timeout=5)
    for p in procs:
        if p.is_alive():
            p.terminate()
            p.join(timeout=1)
    shared.close()
    print("Shutdown complete")


# ═══════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
