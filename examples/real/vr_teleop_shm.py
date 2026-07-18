#!/usr/bin/env python3
"""真机 VR 遥操作 — SharedMemory VR 通道 (零拷贝, 最低延迟)。

架构:
    Meta Quest (HTS app) ──TCP──→ VRReceiverProcess ──SharedMemory──→ TeleopController
                                     (独立进程, 隔离 HTS SDK)          (主进程, CTRL_HZ 决策; 臂内环 50Hz)
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
from dexmani_real.utils.signal_utils import alpha_from_tau, tau_from_alpha
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
from dexmani_real.robot.xhand import XHandConfig
from dexmani_real.sensor.vr_receiver_process import VRReceiverConfig, VRReceiverProcess
from dexmani_real.teleop.control.keyboard import ControlSignal, KeyboardHandler
from dexmani_real.teleop.core.controller import TeleopController, TeleopControllerConfig
from dexmani_real.teleop.vr.arm_mapper import ArmWristMapper
from dexmani_real.teleop.vr.hand_retarget import XHandRetargeter

logger = get_logger(__name__)

# ── 控制频率: 单点定义, 其余常量全部由此派生 ──
# 决策/录制 @ CTRL_HZ; 臂内环保持 50Hz (Mode 6 固件在线规划, 16Hz 直发无插值)。
CTRL_HZ = 16.0
REF_HZ = 50.0  # 滤波/步长参数的原调参频率 (换算保持时间常数/每秒速率不变)
HAND_MAX_QVEL_DEG_S = 90.0  # 手关节限速 → per-send delta clip = deg2rad(90)/CTRL_HZ ≈ 0.098 rad
# 回退注意: 这是"再语义化"而非纯换算 —— 旧默认 XHandConfig.max_delta_rad=0.3 rad/step
# (@50Hz ≙ 859°/s 防尖峰门)。CTRL_HZ 改回 50 会得到 0.031 rad (紧 9.6x)，
# 完整回退需同时删除下方 XHandConfig(max_delta_rad=...) 派生行恢复库默认。


def main():
    print("=" * 60)
    print("VR 遥操作 (SharedMemory 通道)")
    print("=" * 60)

    # ── 0. 任务标签 (写入 HDF5 /meta 与 sidecar, 供多任务数据集过滤) ──
    task_label = input("任务标签 task_label (回车=teleop): ").strip() or "teleop"
    operator = input("操作者 operator (回车=空): ").strip()
    print(f"  task_label={task_label!r} operator={operator!r}")

    # ── 1. VR Receiver (独立进程, SharedMemory 写入端) ──
    vr_config = VRReceiverConfig(
        transport="tcp_server",
        host="0.0.0.0",
        port=8000,
        hand_side="right",
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
            use_position_ik=True,
            max_pose_error_pos_m=0.02,
            max_pose_error_rot_rad=np.deg2rad(5.0),
            # 1°/frame @50Hz — 换算保持 °/s 不变
            nullspace_step_size_deg=1.0 * (REF_HZ / CTRL_HZ),
        ),
    )

    # ── 3. Robot ──
    robot = RobotInterface(
        RobotInterfaceConfig(
            arm=arm_config,
            # 手限速 90°/s 的速度语义: per-send clip 随 CTRL_HZ 派生 (E3)
            hand=XHandConfig(max_delta_rad=float(np.deg2rad(HAND_MAX_QVEL_DEG_S)) / CTRL_HZ),
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
    arm_mapper = ArmWristMapper()
    # LPFilter α=0.6 @50Hz (τ≈22ms) → τ 不变换算到 CTRL_HZ (≈0.94 @16Hz)
    retargeter = XHandRetargeter(low_pass_alpha=alpha_from_tau(tau_from_alpha(0.6, 1.0 / REF_HZ), 1.0 / CTRL_HZ))

    # ── 6. Recorder (optional) ──
    recorder = EpisodeRecorder(
        data_dir="episodes",
        max_frames=int(round(60.0 * CTRL_HZ)),  # 60s 上限
        control_hz=CTRL_HZ,
        min_frames=int(round(1.0 * CTRL_HZ)),  # ≥1s 才算有效 episode
    )

    # ── 7. Controller (SharedMemory VR path) ──
    cfg = TeleopControllerConfig(
        target_hz=CTRL_HZ,
        ema_alpha_pos=1.0,
        ema_alpha_rot=1.0,  # Cartesian EMA pass-through for SHM raw path
        dry_run=False,
        use_shm_vr=True,  # ← 零拷贝 SHM 路径
        collection_config=CollectionConfig(
            task_label=task_label,
            operator=operator,
            min_frames=int(round(1.0 * CTRL_HZ)),
            skip_initial_frames=int(np.ceil(0.2 * CTRL_HZ)),  # 跳过 begin 过渡 ≥0.2s (ceil: 16Hz→4 帧)
        ),
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
