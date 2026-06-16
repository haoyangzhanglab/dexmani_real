"""Test real XHand teleop via Quest VR hand tracking + dex-retargeting.

Hardware prerequisites:
  - XHand connected via RS485 (default /dev/ttyUSB0)
  - Quest 3/Pro connected via USB-C cable (or WiFi — see VR_TRANSPORT below)

Quest 有线连接设置 (USB-C):
  1. Quest 通过 USB-C 线连接 PC
  2. PC 端执行: adb reverse tcp:8000 tcp:8000
  3. Quest 端 HTS app 以 TCP Server 模式运行在端口 8000
  4. 本脚本使用 tcp_client 模式连接 127.0.0.1:8000

Quest WiFi 无线连接:
  1. Quest 和 PC 在同一局域网
  2. 本脚本使用 tcp_server 模式，PC 监听 0.0.0.0:8000
  3. HTS app 以 TCP Client 模式连接 PC 的 IP:8000

Usage:
  source /home/zhy/anaconda3/etc/profile.d/conda.sh && conda activate real
  cd dexmani_real/example
  python test_quest_hand_teleop.py
"""

from __future__ import annotations

import time

import numpy as np

from dexmani_real.robot.xhand import JOINT_NAMES, XHand, XHandConfig
from dexmani_real.teleop.hand_retarget import XHandRetargeter
from dexmani_real.teleop.quest_hand_tracker import QuestHandTracker
from dexmani_real.utils.hand_utils import OPERATOR2MANO_RIGHT, estimate_frame_from_hand_points
from dexmani_real.utils.rate_limiter import RateLimiter

np.set_printoptions(precision=3, suppress=True, linewidth=120)

# ── VR 连接配置 ────────────────────────────────────────────────
# 有线 USB:  transport="tcp_client"  host="127.0.0.1"  (需先 adb reverse tcp:8000 tcp:8000)
# 无线 WiFi: transport="tcp_server"  host="0.0.0.0"    (Quest 主动连 PC)
# 无线 WiFi: transport="udp"         host="0.0.0.0"    port=9000

VR_TRANSPORT = "tcp_client"
VR_HOST = "127.0.0.1"
VR_PORT = 8000
VR_HAND_SIDE = "right"
VR_OUTPUT_FRAME = "flu"

# ── XHand 配置 ─────────────────────────────────────────────────

XHAND_COMM_TYPE = "RS485"
XHAND_DEVICE = "/dev/ttyUSB0"

# ── 控制参数 ───────────────────────────────────────────────────

CONTROL_HZ = 50.0
STATUS_INTERVAL = 2.0  # print status every N seconds


def test_quest_hand_teleop() -> None:
    print("=" * 60)
    print("  Quest VR → XHand Teleop Test")
    print("=" * 60)

    # ── 1. VR tracker ──────────────────────────────────────────
    print(f"\n[1/3] Starting VR tracker ({VR_TRANSPORT}://{VR_HOST}:{VR_PORT})...")
    tracker = QuestHandTracker(
        transport=VR_TRANSPORT,
        host=VR_HOST,
        port=VR_PORT,
        hand_side=VR_HAND_SIDE,
        output_frame=VR_OUTPUT_FRAME,
        max_frame_age_s=0.20,
        verbose=True,
    )
    tracker.connect()

    if not tracker.started:
        raise RuntimeError("QuestHandTracker failed to start. Is HTS streaming?")

    # ── 2. XHand ───────────────────────────────────────────────
    print(f"\n[2/3] Connecting XHand ({XHAND_COMM_TYPE}:{XHAND_DEVICE})...")
    hand_config = XHandConfig(
        comm_type=XHAND_COMM_TYPE,
        device_name=XHAND_DEVICE,
    )
    xhand = XHand(hand_config)

    if not xhand.connect():
        tracker.disconnect()
        raise RuntimeError(f"XHand connect failed: {xhand.last_error_message}")

    # Print initial state
    state = xhand.get_state(full=True)
    print(f"  Connected. qpos={state['qpos']}")

    if xhand.is_error():
        print(f"  [WARN] Hand has errors: commboard={state.get('commboard_err')}")

    # ── 3. Retargeter ──────────────────────────────────────────
    print("\n[3/3] Loading dex-retargeting model...")
    retargeter = XHandRetargeter(debug_adapters=False)
    print("  Ready.")

    # ── Main loop ───────────────────────────────────────────────
    limiter = RateLimiter(CONTROL_HZ)
    last_status_ts = time.monotonic()
    frame_count = 0
    retarget_count = 0
    retarget_fail_count = 0
    no_frame_count = 0
    last_qpos = state["qpos"].copy()

    print("\n" + "=" * 60)
    print("  Teleop running. Press Ctrl+C to stop.")
    print(f"  Control rate: {CONTROL_HZ} Hz")
    print("=" * 60 + "\n")

    try:
        while True:
            # ── VR frame ──
            frame = tracker.get_latest()
            if frame is None:
                no_frame_count += 1
                if no_frame_count == 1 or no_frame_count % 50 == 0:
                    print(f"[VR] no frame (x{no_frame_count})")
                limiter.wait()
                continue

            no_frame_count = 0
            frame_count += 1

            # ── Retarget ──
            landmarks = frame["landmarks"]  # (21, 3)
            wrist_rot = estimate_frame_from_hand_points(landmarks)  # (3, 3)
            mano_landmarks = landmarks @ wrist_rot @ OPERATOR2MANO_RIGHT  # (21, 3)

            target_qpos = retargeter.retarget(mano_landmarks)

            if target_qpos is None:
                retarget_fail_count += 1
                print(f"[retarget] failed (x{retarget_fail_count}), using last action")
                target_qpos = last_qpos
            else:
                retarget_count += 1
                last_qpos = target_qpos

            # ── Safety check ──
            if xhand.is_error():
                print("[SAFETY] XHand error detected, attempting clear...")
                xhand.clear_error()
                if xhand.is_error():
                    print("[SAFETY] Unrecoverable error, stopping.")
                    xhand.stop()
                    break

            # ── Send action ──
            ok = xhand.send_action(target_qpos)
            if not ok:
                print(f"[XHand] send_action failed: {xhand.last_error_message}")
                if xhand.is_error():
                    break

            # ── Status print ──
            now = time.monotonic()
            if now - last_status_ts >= STATUS_INTERVAL:
                last_status_ts = now
                tracker_status = tracker.get_status()
                qpos = xhand.get_state()["qpos"]
                age_ms = tracker_status.get("frame_age_s", 0) * 1000
                clipped = "CLIP" if xhand.last_joint_limit_clipped else ""
                dlimited = "DLIM" if xhand.last_delta_limited else ""
                flags = " ".join(f for f in [clipped, dlimited] if f) or "ok"
                print(
                    f"[t={now:.1f}] frames={frame_count} "
                    f"retarget_ok={retarget_count} fail={retarget_fail_count} "
                    f"vr_age={age_ms:.0f}ms "
                    f"seq={frame['sequence_id']} "
                    f"qpos={qpos} "
                    f"safety_flags=[{flags}]"
                )

            # ── Rate limit ──
            limiter.wait()

    except KeyboardInterrupt:
        print("\n\n[Stopping] KeyboardInterrupt received.")

    finally:
        print("[Cleanup] Resetting XHand to home...")
        xhand.reset()
        time.sleep(1.0)
        xhand.disconnect()
        tracker.disconnect()

        print(f"\n[Summary] frames={frame_count} retarget={retarget_count} retarget_fail={retarget_fail_count}")
        print("  Done.")


if __name__ == "__main__":
    test_quest_hand_teleop()
