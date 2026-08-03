#!/usr/bin/env python3
"""VR teleop xArm7 + XHand with recording — simplified architecture (Phase 2.3).

.. attention::
   **DEPRECATED** — use **vr_teleop_hand_record.py** as the canonical entry point.
   This file is kept for reference; new features are added to hand_record.py.

5 child processes + thin Main orchestrator. All data exchange through SharedStorage.

Supports:
- Arm teleop via VR wrist → IK (EMA + workspace clamp + delta clamp)
- Hand teleop via VR landmarks → DexPilot NLP retargeting (P4 Step 1, 2026-08-02)
- Safety: DISARMED/ARMED/RUNNING/FAULT state machine + 5 per-process heartbeats
- Recording: TimestampAlignedBuffer → HDF5 (schema v11, arm+hand+EEF+camera)

Legacy entry points (deprecated):
- vr_teleop_arm_only_record.py → replaced by this script
- vr_teleop_arm_only.py → replaced by this script
- vr_teleop_hand_record.py → hand retargeting now available here; legacy kept
  for voice feedback + advanced features not yet ported

Usage:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate real_robot
    python examples/real/vr_teleop_arm_only_record_plus.py

Controls:
    B=teleop+record  C=pause  S=save  D=discard  H=home  Q=quit  ESC=estop
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path before imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def main() -> None:
    """Spawn 5 child processes, wait, clean up."""
    import argparse

    from dexmani_real.policy.vr_teleop_policy import PolicyConfig, policy_loop
    from dexmani_real.robot.inner_loop import ArmInnerLoopConfig, arm_loop as _arm_loop
    from dexmani_real.robot.hand_process import hand_loop as _hand_loop
    from dexmani_real.sensor.camera_process import camera_loop as _camera_loop
    from dexmani_real.sensor.vr_receiver_process import vr_loop as _vr_loop
    from dexmani_real.shm.shared_storage import SharedStorage
    from dexmani_real.utils.log import get_logger

    logger = get_logger(__name__)

    parser = argparse.ArgumentParser(description="VR Teleop xArm7 + XHand with recording")
    parser.add_argument("--task", type=str, default="", help="Task label for recording metadata")
    parser.add_argument("--operator", type=str, default="", help="Operator name for recording metadata")
    args = parser.parse_args()

    print("=" * 60)
    print("VR Teleop xArm7 + XHand (simplified architecture)")
    print("5 processes: camera | vr | policy | arm | hand")
    print("=" * 60)

    # ── 1. SharedStorage ──
    shared = SharedStorage.create(prefix="dexmani")

    # ── 2. Policy config ──
    policy_cfg = PolicyConfig(task_label=args.task, operator=args.operator)

    # ── 3. Spawn ──
    procs = [
        mp.Process(target=_camera_loop, args=(shared,), name="cam", daemon=True),
        mp.Process(target=_vr_loop, args=(shared,), name="vr", daemon=True),
        mp.Process(target=policy_loop, args=(shared, policy_cfg), name="pol", daemon=True),
        mp.Process(target=_arm_loop, args=(shared, ArmInnerLoopConfig()), name="arm", daemon=True),
        mp.Process(target=_hand_loop, args=(shared,), name="hand", daemon=True),
    ]
    for p in procs:
        p.start()

    # ── 4. Wait for ready ──
    from dexmani_real.robot.safety import SafetyState, transition
    from dexmani_real.shm.shared_storage import HEARTBEAT_TIMEOUTS, HOME_SENTINEL

    transition(shared, SafetyState.DISARMED)

    for name, ev, timeout in [
        ("arm", shared.arm_ready, 15),
        ("hand", shared.hand_ready, 15),
        ("camera", shared.camera_ready, 15),
        ("vr", shared.vr_ready, 15),
    ]:
        if not ev.wait(timeout=timeout):
            logger.error("%s startup failed: ready-event timeout after %ds", name, timeout)
            shared.is_running.value = False
            _shutdown(procs, shared)
            return

    # All subsystems ready — transition to ARMED (hardware connected, ready for teleop).
    transition(shared, SafetyState.ARMED)
    print(f"\nAll subsystems ready — 安全状态: ARMED ({int(SafetyState.ARMED)})")
    print("Controls: B=teleop+record C=pause H=home Q=quit ESC=estop\n")

    # ── 5. Supervisor (heartbeat + process monitor) ──
    PROC_NAMES = ["arm", "hand", "policy", "vr", "camera"]
    HEARTBEAT_FIELDS = {
        "arm": shared.arm_heartbeat_s,
        "hand": shared.hand_heartbeat_s,
        "policy": shared.policy_heartbeat_s,
        "vr": shared.vr_heartbeat_s,
        "camera": shared.camera_heartbeat_s,
    }

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
                        # Policy exited after setting is_running=False — expected
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
                timeout_s = float(HEARTBEAT_TIMEOUTS[name])
                if age_s > timeout_s:
                    logger.error("%s heartbeat timeout: %.1fs > %.1fs — FAULT", name, age_s, timeout_s)
                    transition(shared, SafetyState.FAULT)
                    break
            if shared.safety_state.value == int(SafetyState.FAULT):
                break

            time.sleep(0.1)  # 10Hz supervisor — faster detection than legacy 0.5s
    except KeyboardInterrupt:
        normal_exit = True
        shared.is_running.value = False
    finally:
        transition(shared, SafetyState.DISARMED)

        # ── Post-loop: offer return_home on normal exit ──
        if normal_exit and not shared.error_state.value:
            from dexmani_real.teleop.control.keyboard import ControlSignal, KeyboardHandler
            kb = KeyboardHandler()
            kb.start()
            try:
                print("\n按 H 执行 return_home，或按 Q 直接退出...")
                _post_deadline = time.perf_counter() + 60.0
                while time.perf_counter() < _post_deadline:
                    for sig in kb.poll(timeout=0.1):
                        if sig == ControlSignal.HOME:
                            print("\nH: return_home")
                            shared.arm_action_q.put(HOME_SENTINEL)
                            # Wait for arm to pick up and execute homing (~8s typical)
                            _home_wait = time.perf_counter() + 20.0
                            while time.perf_counter() < _home_wait:
                                if shared.arm_heartbeat_s.value > 0:
                                    time.sleep(0.5)
                                else:
                                    break
                            print("按 Q 退出...")
                        if sig == ControlSignal.QUIT or sig == ControlSignal.EMERGENCY_STOP:
                            break
                    else:
                        continue
                    break
                else:
                    print("\n  超时，自动退出")
            finally:
                kb.stop()

        _shutdown(procs, shared)


def _shutdown(procs: list, shared) -> None:
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


if __name__ == "__main__":
    main()
