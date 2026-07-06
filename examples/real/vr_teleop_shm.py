#!/usr/bin/env python3
"""真机 VR 遥操作 — SharedMemory VR 通道 (零拷贝, 最低延迟)。

架构:
    Meta Quest (HTS app) ──TCP──→ VRReceiverProcess ──SharedMemory──→ TeleopController
                                     (独立进程, 隔离 HTS SDK)          (主进程, 50Hz)
                                         │                                │
                                   shm.write_vr_frame()            _vr_shm.read_latest_vr()

    相比 ZMQ PUB/SUB 方案:
      - 延迟: ~1μs (SHM) vs ~1-2ms (ZMQ TCP 序列化/反序列化)
      - 零拷贝: numpy array 直接写入 ring buffer
      - 无网络栈开销

用法:
    # 1. Quest USB 有线: adb reverse tcp:8000 tcp:8000
    # 2. 启动 VR receiver:
    source /home/zhy/anaconda3/etc/profile.d/conda.sh && conda activate real
    python examples/real/vr_teleop_shm.py

控制:
    B    开始遥操作 (自动录制)
    C    暂停/恢复 (toggle, 保持当前位置)
    S    停止录制 (自动保存)
    H    归位 (return-to-home)
    Q    退出
    ESC  急停
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import time

import numpy as np

from dexmani_real import ASSET_DIR
from dexmani_real.utils.log import get_logger
from dexmani_real.planning import (
    PlanningProfile,
    Pose,
    TeleopProfile,
    XArm7MotionPlanner,
    XArm7PlannerConfig,
)
from dexmani_real.planning.collision_config import CollisionConfig
from dexmani_real.recording.collection_config import CollectionConfig
from dexmani_real.recording.episode_recorder import EpisodeRecorder
from dexmani_real.robot.interface import RobotInterface, RobotInterfaceConfig
from dexmani_real.robot.preflight import preflight_check, print_preflight
from dexmani_real.robot.xarm7 import XArm7Config
from dexmani_real.sensor.vr_receiver_process import VRReceiverConfig, VRReceiverProcess
from dexmani_real.teleop.control.keyboard import ControlSignal, KeyboardHandler
from dexmani_real.teleop.core.controller import TeleopController, TeleopControllerConfig
from dexmani_real.teleop.vr.arm_mapper import ArmWristMapper
from dexmani_real.teleop.vr.hand_retarget import XHandRetargeter

logger = get_logger(__name__)


def main():
    print("=" * 60)
    print("VR 遥操作 (SharedMemory 通道)")
    print("=" * 60)

    # ── 1. VR Receiver (独立进程, SharedMemory 写入端) ──
    vr_config = VRReceiverConfig(
        transport="tcp_server",
        host="0.0.0.0",
        port=8000,
        hand_side="right",
        output_frame="flu",
        max_frame_age_s=0.20,
    )
    vr_receiver = VRReceiverProcess(config=vr_config)
    vr_receiver.start()
    print(f"VRReceiverProcess 已启动 ({vr_config.transport}://{vr_config.host}:{vr_config.port})")

    # ── 2. Planner ──
    arm_config = XArm7Config()
    urdf_path = str(ASSET_DIR / "robots" / "xhand" / "xarm7_xhand_collision.urdf")
    srdf_path = str(ASSET_DIR / "robots" / "xhand" / "xarm7_xhand.srdf")

    collision = CollisionConfig(
        table_z_world=0.0,
        hand_extension_below_eef=0.076,
        hand_safe_margin=0.03,
    )

    planner = XArm7MotionPlanner(
        XArm7PlannerConfig(
            urdf_path=urdf_path,
            srdf_path=srdf_path,
            base_pose_world=Pose(
                p=np.array([0.0, 0.0, 0.0]),
                q=np.array([np.cos(np.pi / 12), 0.0, 0.0, np.sin(np.pi / 12)]),
            ),
            collision=collision,
        ),
        planning_profile=PlanningProfile(),
        teleop_profile=TeleopProfile(
            teleop_dt=0.02,
            use_position_ik=True,
            max_pose_error_pos_m=0.02,
            max_pose_error_rot_rad=np.deg2rad(5.0),
        ),
    )

    # ── 3. Robot ──
    robot = RobotInterface(
        RobotInterfaceConfig(
            arm=arm_config,
            collision=collision,
            hand_urdf_path=str(ASSET_DIR / "robots" / "xhand" / "xhand_right.urdf"),
        ),
        kinematics=planner.kin,
        planner=planner,
    )

    # ── 4. Connect & Pre-Flight ──
    print("\n连接硬件...")
    result = robot.connect()
    print(f"  arm:  {'OK' if result.get('arm') else 'FAIL'}")
    print(f"  hand: {'OK' if result.get('hand') else 'FAIL (降级运行)'}")

    if not result.get("arm"):
        print("arm 连接失败，退出")
        vr_receiver.stop()
        return

    report = preflight_check(robot)
    print_preflight(report)
    if not report.passed:
        print("Pre-Flight 检查失败，退出")
        robot.disconnect()
        vr_receiver.stop()
        return

    # ── 5. Mapper + Retargeter ──
    arm_mapper = ArmWristMapper(robot, planner)
    retargeter = XHandRetargeter()

    # ── 6. Recorder (optional) ──
    recorder = EpisodeRecorder(
        data_dir="episodes",
        max_frames=3000,
    )

    # ── 7. Controller (SharedMemory VR path) ──
    cfg = TeleopControllerConfig(
        target_hz=50.0,
        ema_alpha_arm=1.0,  # LeFranX-style simple EMA (1.0 = no smoothing, pass-through)
        dry_run=False,
        use_shm_vr=True,  # ← 零拷贝 SHM 路径
        collection_config=CollectionConfig(),
    )

    controller = TeleopController(
        robot=robot,
        arm_mapper=arm_mapper,
        retargeter=retargeter,
        planner=planner,
        cfg=cfg,
        recorder=recorder,
    )

    # ── 8. Run ──
    print("\n等待 VR 帧... (确保 Quest 已连接并启动 HTS App)")
    print("  Q=退出")
    kb = KeyboardHandler()
    kb.start()
    try:
        startup_deadline = time.perf_counter() + 120.0
        while time.perf_counter() < startup_deadline:
            for sig in kb.poll(timeout=0.0):
                if sig in (ControlSignal.QUIT, ControlSignal.EMERGENCY_STOP):
                    print(f"\n{sig.value}: 退出")
                    vr_receiver.stop()
                    robot.disconnect()
                    return
            frame = vr_receiver.read_latest()
            if frame is not None:
                print(f"  收到首帧 seq={frame.get('sequence_id', '?')} — 就绪")
                break
            time.sleep(0.5)
        else:
            print("  VR 帧超时 (120s) — 退出")
            vr_receiver.stop()
            robot.disconnect()
            return
    finally:
        kb.stop()

    print("\n控制: B=开始 C=暂停 S=停止 H=归位 Q=退出 ESC=急停\n")
    controller.run()

    # ── 9. Cleanup ──
    vr_receiver.stop()
    robot.disconnect()
    print("Done.")


if __name__ == "__main__":
    main()
