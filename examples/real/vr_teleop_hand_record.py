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
      ├─ camera_loop ──camera_ring──┐
      ├─ vr_loop ────────vr_ring─────┤
      │                               ▼
      ├─ policy_loop ───arm_action_q──→ arm_loop
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
    HOME_SENTINEL,
    SharedStorage,
    read_arm_state,
    read_hand_state,
    write_hand_cmd,
)
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
    parser.add_argument("--config", type=str, default=None, help="JSON file with config overrides (flat fields only)")
    parser.add_argument("--print-config", action="store_true", help="Print all config values and exit")
    args = parser.parse_args()

    if args.config:
        from dexmani_real.config.defaults import load_config_json

        load_config_json(args.config)

    if args.print_config:
        import dataclasses

        for label, obj in (
            ("arm", arm), ("hand", hand), ("policy", policy),
            ("vr", vr), ("safety", safety), ("camera", camera),
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

    # ── 1. SharedStorage ──
    shared = SharedStorage.create(prefix="dexmani")

    # ── 2. Policy config ──
    policy_cfg = PolicyConfig(
        task_label=args.task,
        operator=args.operator,
        joint_max_acc_deg_s2=args.acc,
        joint_max_speed_deg_s=args.speed,
        hand_urdf_path=str(ASSET_DIR / "robots" / "xhand" / "xhand_right.urdf"),
    )
    if args.no_hand:
        policy_cfg.hand_enabled = False

    # ── 2b. Arm config (must match PolicyConfig speed/acc) ──
    arm_cfg = ArmLoopConfig(
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
    # vr_loop defers vr_ready until the first HTS event arrives (not just TCP
    # connect), so the 120 s timeout here doubles as the "put on headset" grace
    # period.  Arm/camera/hand are ready within seconds.
    transition(shared, SafetyState.DISARMED)

    _ready_checks: list[tuple[str, mp.synchronize.Event, float]] = [
        ("arm", shared.arm_ready, 15),
        ("camera", shared.camera_ready, 15),
        ("vr", shared.vr_ready, 120),
    ]
    if not args.no_hand:
        _ready_checks.append(("hand", shared.hand_ready, 15))

    for name, ev, timeout in _ready_checks:
        if name == "vr":
            print(f"\n  Waiting for VR connection (up to {timeout}s) — put on Quest headset...", flush=True)
        _ready_deadline = time.monotonic() + timeout
        _ready_ok = False
        _already_logged = False
        while time.monotonic() < _ready_deadline:
            if ev.is_set():
                _ready_ok = True
                break
            if shared.error_state.value:
                logger.error("subsystem=%s init failed: error_state set", name)
                _already_logged = True
                break
            # If any spawned process exits during ready-check, abort immediately
            if not all(p.is_alive() for p in procs):
                logger.error("subsystem=%s init failed: a process exited prematurely", name)
                _already_logged = True
                break
            time.sleep(0.2)
        if not _ready_ok and not _already_logged:
            logger.error("subsystem=%s ready_timeout=%ds", name, timeout)
        if not _ready_ok:
            shared.is_running.value = False
            _shutdown(procs, shared)
            return
        if name == "vr":
            print(f"  VR connected", flush=True)
        else:
            print(f"  {name}: ready", flush=True)

    # ── 4b. Pre-flight health summary (ring-based, no direct hardware access) ──
    _print_health_summary(shared)

    # All subsystems ready — transition to ARMED.
    transition(shared, SafetyState.ARMED)
    print(f"\nAll subsystems ready — safety=ARMED({int(SafetyState.ARMED)})\n"
          "Controls: B=teleop+record  C=pause  S=save  D=discard  H=home  Q=quit  ESC=estop\n")

    # ── 5. Supervisor (heartbeat + process monitor) ──
    _all_names = ["camera", "vr", "policy", "arm"]
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

    # Seed any heartbeats still at 0.0 (processes still initializing) to
    # current time so they get the full timeout window from ARMED, not from
    # process start.  policy_loop is the main beneficiary — it does heavy
    # init (collision model, hand retargeter) after all Ready events but
    # before its main loop starts ticking the heartbeat.
    _now = time.monotonic()
    for _hb in HEARTBEAT_FIELDS.values():
        if _hb.value == 0.0:
            _hb.value = _now

    _start_time = time.monotonic()
    _last_status_s = _start_time
    _exit_reason = "unknown"
    normal_exit = False
    try:
        while True:
            # 0. Normal exit — policy set is_running=False (Q key)
            if not shared.is_running.value:
                normal_exit = True
                _exit_reason = "is_running=False (Q key)"
                logger.info("exit_reason=%s", _exit_reason)
                break

            # 1. Process aliveness
            for p, name in zip(procs, PROC_NAMES):
                if not p.is_alive():
                    if normal_exit:
                        logger.info("process=%s exit=normal", name)
                    else:
                        _exit_reason = f"process={name} died"
                        logger.error("%s — FAULT", _exit_reason)
                        transition(shared, SafetyState.FAULT)
                    break
            if shared.safety_state.value == int(SafetyState.FAULT):
                # FAULT was set by a subprocess (arm_loop/hand_loop).
                # Diagnose the root cause from available flags.
                if shared.error_state.value:
                    _exit_reason = "error_state set (subprocess)"
                elif shared.estop_request.value:
                    _exit_reason = "e-stop (subprocess)"
                else:
                    _exit_reason = "FAULT set by subprocess"
                logger.error("%s — FAULT", _exit_reason)
                break

            # 2. Error state (sticky latch from arm/hand)
            if shared.error_state.value:
                _exit_reason = "error_state set"
                logger.error("%s — FAULT", _exit_reason)
                transition(shared, SafetyState.FAULT)
                break

            # 3. Heartbeat timeouts
            now = time.monotonic()
            for name in PROC_NAMES:
                last_hb = float(HEARTBEAT_FIELDS[name].value)
                age_s = now - last_hb if last_hb > 0 else float("inf")
                timeout_s = float(safety.heartbeat_timeouts[name])
                if age_s > timeout_s:
                    _exit_reason = f"heartbeat={name} timeout={age_s:.1f}s>{timeout_s:.1f}s"
                    logger.error("%s — FAULT", _exit_reason)
                    transition(shared, SafetyState.FAULT)
                    break
            if shared.safety_state.value == int(SafetyState.FAULT):
                break

            # 4. Periodic supervisor heartbeat (~every 30s)
            if now - _last_status_s >= 30.0:
                _runtime_m = (now - _start_time) / 60.0
                _safety = shared.safety_state.value
                _hb_ages = ", ".join(f"{n}={now - float(HEARTBEAT_FIELDS[n].value):.1f}s"
                                    for n in PROC_NAMES)
                print(f"  [supervisor]  runtime={_runtime_m:.1f}min  safety={_safety}  hb_age=({_hb_ages})", flush=True)
                _last_status_s = now

            time.sleep(0.1)  # 10Hz supervisor

    except KeyboardInterrupt:
        _exit_reason = "KeyboardInterrupt"
        normal_exit = True
        shared.is_running.value = False
    finally:
        transition(shared, SafetyState.DISARMED)

        # ── Post-loop: offer return_home on normal exit ──
        # Skip when Policy initiated the quit (quit_requested=True) — Policy
        # already handled the [H] return_home / [Q] quit prompt while all
        # processes were still alive.
        if normal_exit and not shared.error_state.value and not shared.quit_requested.value:
            _post_loop_home(shared)

        _shutdown(procs, shared)

        # ── Exit summary ──
        _runtime_m = (time.monotonic() - _start_time) / 60.0
        _final_safety = shared.safety_state.value
        print(f"\n── Session End ──")
        print(f"  exit_reason={_exit_reason}  runtime={_runtime_m:.1f}min  "
              f"safety={_final_safety}  normal={normal_exit}")
        print("──")


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def _print_health_summary(shared: SharedStorage) -> None:
    """Print a pre-flight style health summary from ring data."""
    print("\n── Health Check ──")

    # Arm
    arm_result = shared.arm_state_ring.read_latest()
    if arm_result is not None:
        arm_data, _, _ = arm_result
        arm_connected = bool(arm_data["connected"][0])
        arm_error = int(arm_data["error_code"][0])
        arm_qpos = np.asarray(arm_data["qpos"][0], dtype=np.float64)
        arm_qpos_ok = int(np.all(np.isfinite(arm_qpos)))
        arm_ok = arm_connected and arm_error == 0 and bool(arm_qpos_ok)
        print(f"  arm   {'OK' if arm_ok else 'FAIL':>4s}  connected={int(arm_connected)}  "
              f"error={arm_error}  qpos_ok={arm_qpos_ok}")
    else:
        print("  arm   ----  (no data yet)")

    # Hand
    hand_result = shared.hand_state_ring.read_latest()
    if hand_result is not None:
        hand_data, _, _ = hand_result
        hand_connected = bool(hand_data["connected"][0])
        hand_error = bool(hand_data["error_state"][0])
        hand_qpos_stale = bool(hand_data["qpos_stale"][0])
        hand_qpos = np.asarray(hand_data["qpos"][0], dtype=np.float64)
        hand_qpos_ok = int(np.all(np.isfinite(hand_qpos)))
        hand_ok = hand_connected and not hand_error and bool(hand_qpos_ok)
        stale_note = " stale=1" if hand_qpos_stale else ""
        print(f"  hand  {'OK' if hand_ok else 'FAIL':>4s}  connected={int(hand_connected)}  "
              f"error={int(hand_error)}  qpos_ok={hand_qpos_ok}{stale_note}")
    else:
        print("  hand  ----  (no data yet)")

    # VR
    vr_result = shared.vr_ring.read_latest()
    if vr_result is not None:
        vr_data, _, _ = vr_result
        vr_age_s = (time.monotonic_ns() - int(vr_data["local_recv_ns"][0])) / 1e9 if vr_data["local_recv_ns"][0] > 0 else -1
        print(f"  vr     OK   age={vr_age_s:.1f}s  seq={int(vr_data['sequence_id'][0])}")
    else:
        print("  vr    ----  (no data yet)")

    # Camera
    cam_serial_bytes = shared.camera_serial.value.rstrip(b"\x00")
    if cam_serial_bytes:
        print(f"  cam    OK   serial={cam_serial_bytes.decode()}")
    elif shared.camera_heartbeat_s.value > 0:
        print("  cam    OK   serial=unknown")
    else:
        print("  cam   ----  (no data yet)")

    print("──")
    sys.stdout.flush()


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

                    # ── Step 1: Hand home first ──
                    # hand_loop is still running (is_running not set yet).
                    _hand_home_tol = np.deg2rad(5.0)
                    _hand_home_deadline = time.monotonic() + 5.0
                    _hand_home_reached = False
                    while time.monotonic() < _hand_home_deadline:
                        write_hand_cmd(shared, HAND_HOME_QPOS)
                        _hs = read_hand_state(shared)
                        if _hs is not None:
                            _current = np.asarray(_hs["qpos"][0], dtype=np.float64)
                            if np.all(np.isfinite(_current)):
                                if float(np.max(np.abs(_current - HAND_HOME_QPOS))) < _hand_home_tol:
                                    _hand_home_reached = True
                                    break
                        time.sleep(0.05)
                    if _hand_home_reached:
                        print("  hand: home reached", flush=True)
                    else:
                        print("  hand: home settle timeout — proceeding", flush=True)

                    # ── Step 2: Arm home ──
                    # Uses None waypoints: arm_loop's _planned_homing falls back
                    # to joint-space linear interpolation.  On Ctrl+C the arm
                    # process typically exits before we reach here — best-effort only.
                    try:
                        shared.arm_action_q.put((HOME_SENTINEL, None), timeout=2.0)
                    except Exception:
                        print("  arm_action_q full — arm may have already exited")
                        break
                    # Wait for arm to converge to home_qpos (joint-position check).
                    # On Ctrl+C the arm process may have already exited — best-effort
                    # polling with a short timeout.
                    _home_qpos = np.array(arm.home_qpos, dtype=np.float64)
                    _home_tol = np.deg2rad(2.0)
                    _home_deadline = time.monotonic() + 10.0
                    _home_reached = False
                    while time.monotonic() < _home_deadline:
                        _as = read_arm_state(shared)
                        if _as is not None:
                            _q = np.asarray(_as["qpos"][0], dtype=np.float64)
                            if np.all(np.isfinite(_q)):
                                if float(np.max(np.abs(_q - _home_qpos))) < _home_tol:
                                    _home_reached = True
                                    break
                        time.sleep(0.1)
                    if _home_reached:
                        print("  arm: home reached", flush=True)
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


def _shutdown(procs: list, shared: SharedStorage) -> None:
    """Graceful shutdown: signal -> join -> terminate."""
    shared.is_running.value = False
    _status: list[str] = []
    for p in procs:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
            p.join(timeout=1)
            _status.append(f"{p.name}=term")
        else:
            _status.append(f"{p.name}=ok")
    shared.close()
    print(f"  shutdown: {'  '.join(_status)}")


# ═══════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
