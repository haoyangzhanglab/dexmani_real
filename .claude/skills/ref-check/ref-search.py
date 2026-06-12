#!/usr/bin/env python3
"""跨参考项目搜索工具。

按 P1→P2→P3 优先级在 6 个参考项目中搜索相关代码文件。
用法:
  python ref-search.py <module> [--keyword <kw>] [--priority P1|P2|P3] [--list-mapping]
  python ref-search.py robot        # 搜索 robot 模块的所有参考
  python ref-search.py ik --keyword "SDLS"  # 在所有参考中搜索含 SDLS 的文件
  python ref-search.py --list-mapping       # 列出 CLAUDE.md 中的静态映射表
"""

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ── 参考项目配置 ──────────────────────────────────────────────

@dataclass
class RefProject:
    name: str
    path: str
    priority: int  # 1, 2, 3
    remote: str = ""

REF_PROJECTS: list[RefProject] = [
    RefProject("ManiUniCon",     "/home/zhy/Desktop/ManiUniCon",          1),
    RefProject("LeFranX",        "/home/zhy/Desktop/LeFranX",             1),
    RefProject("BunnyVisionPro", "/home/zhy/Desktop/BunnyVisionPro",      2),
    RefProject("DexUMI",         "/home/zhy/Desktop/DexUMI",              2),
    RefProject("Open-Teach",     "/home/zhy/Desktop/Open-Teach",          2),
    RefProject("Bidex_Manus_Teleop", "/home/zhy/Desktop/Bidex_Manus_Teleop", 3),
]

# ── 静态模块映射（从 CLAUDE.md Section 0.5.3 提取）────────────

MODULE_MAPPING: dict[str, list[dict]] = {
    "robot": [
        {"p": 1, "src": "ManiUniCon",     "path": "maniunicon/robot_interface/xarm6_robotiq.py", "note": "xArm6 接口，与 xArm7 SDK 最接近"},
        {"p": 1, "src": "ManiUniCon",     "path": "maniunicon/robot_interface/base.py",          "note": "RobotInterface 基类设计"},
        {"p": 1, "src": "LeFranX",        "path": "src/lerobot/robots/franka_fer_xhand/",        "note": "Franka+XHand 复合接口设计"},
        {"p": 1, "src": "LeFranX",        "path": "src/lerobot/robots/xhand/xhand.py",           "note": "XHand 接口参考"},
        {"p": 2, "src": "BunnyVisionPro", "path": "real_control/xarm7_ability.py",               "note": "xArm7 真机控制（PID、servo 模式、Ability Hand 集成）"},
        {"p": 2, "src": "DexUMI",         "path": "dexumi/hand_sdk/dexhand.py",                   "note": "DexterousHand/ExoDexterousHand ABC 设计"},
        {"p": 2, "src": "DexUMI",         "path": "dexumi/hand_sdk/xhand/hand_api_cls.py",        "note": "XHand SDK 封装（后台读取线程 + joint clip + 触觉）"},
        {"p": 2, "src": "DexUMI",         "path": "dexumi/real_env/common/ur5.py",                "note": "UR5 servoL 伺服控制（ZMQ Server/Client 分离模式）"},
        {"p": 2, "src": "Open-Teach",     "path": "openteach/robot/robot.py",                    "note": "RobotWrapper ABC 抽象模式（多机器人接口）"},
        {"p": 2, "src": "Open-Teach",     "path": "openteach/robot/franka.py",                   "note": "Franka 驱动实现参考"},
    ],
    "controller": [
        {"p": 1, "src": "LeFranX",        "path": "src/lerobot/teleoperators/franka_fer_xhand_vr/", "note": "Franka+XHand+VR 遥操作主循环"},
        {"p": 1, "src": "LeFranX",        "path": "src/lerobot/teleoperators/franka_fer_vr/arm_ik_processor.py", "note": "Arm IK 处理流程"},
        {"p": 1, "src": "ManiUniCon",     "path": "maniunicon/policies/quest.py",                    "note": "Quest VR 遥操作策略"},
        {"p": 1, "src": "ManiUniCon",     "path": "maniunicon/utils/quest_controller.py",            "note": "Quest 控制器工具"},
        {"p": 2, "src": "BunnyVisionPro", "path": "real_control/teleop_bimanual_xarm7_ability.py",   "note": "xArm7+Ability Hand 双手遥操作主循环（含录制、状态机、键盘控制）"},
        {"p": 2, "src": "DexUMI",         "path": "real_script/teleoperation/teleoperation.py",       "note": "外骨骼遥操作主循环（iPhone手腕追踪 + 外骨骼编码器）"},
        {"p": 2, "src": "DexUMI",         "path": "dexumi/real_env/common/pose_trajectory_interpolator.py", "note": "位姿轨迹插值器（速度限制 + schedule_waypoint）"},
        {"p": 2, "src": "Open-Teach",     "path": "openteach/components/operators/operator.py",     "note": "Operator 模式：retargeting → streaming 控制循环"},
        {"p": 2, "src": "Open-Teach",     "path": "openteach/components/operators/allegro.py",      "note": "手部 operator 实现（移动平均滤波 + 关节限位）"},
    ],
    "teleop": [
        {"p": 1, "src": "LeFranX",        "path": "src/lerobot/teleoperators/xhand_vr/",                "note": "XHand VR 重定向（vr_hand_detector_adapter.py）"},
        {"p": 1, "src": "LeFranX",        "path": "src/lerobot/teleoperators/vr_router_manager.py",     "note": "VR 路由管理"},
        {"p": 1, "src": "ManiUniCon",     "path": "third_party/oculus_reader/",                         "note": "Oculus/Quest 数据读取"},
        {"p": 2, "src": "BunnyVisionPro", "path": "examples/retargeting/retargeting.py",                "note": "dex-retargeting 手部重定向（OPERATOR2MANO/AVP 坐标变换）"},
        {"p": 2, "src": "BunnyVisionPro", "path": "bunny_teleop/bimanual_teleop_client.py",             "note": "ZMQ-based 双手遥操作 client/server 模式"},
        {"p": 2, "src": "BunnyVisionPro", "path": "bunny_teleop/init_config.py",                        "note": "双手对齐模式配置（BimanualAlignmentMode）"},
        {"p": 2, "src": "DexUMI",         "path": "dexumi/encoder/encoder.py",                          "note": "外骨骼关节编码器（UART + 电压→角度 + XHand/Inspire双实现）"},
        {"p": 2, "src": "DexUMI",         "path": "dexumi/encoder/UARTReader.py",                       "note": "UART 串口读取基类（后台线程 + 环形缓冲区）"},
        {"p": 2, "src": "DexUMI",         "path": "dexumi/hand_sdk/xhand/hand_api_cls.py",              "note": "ExoXhandSDK：外骨骼编码器→电机值预测"},
        {"p": 2, "src": "DexUMI",         "path": "dexumi/camera/iphone_camera.py",                     "note": "iPhone ARKit 6-DoF 手腕位姿追踪"},
        {"p": 2, "src": "Open-Teach",     "path": "openteach/components/detector/oculus.py",            "note": "Oculus/Quest 手部关键点检测"},
        {"p": 2, "src": "Open-Teach",     "path": "openteach/robot/allegro/allegro_retargeters.py",     "note": "Kinematic retargeting（关节限位 + 移动平均滤波）"},
        {"p": 3, "src": "Bidex_Manus_Teleop", "path": "python/minimal_example.py",                      "note": "Manus 数据手套 ZMQ 数据解析（skeleton + ergonomics）"},
        {"p": 3, "src": "Bidex_Manus_Teleop", "path": "ros2/telekinesis/telekinesis/leap_ik.py",       "note": "PyBullet IK 指尖重定向（glove fingertip → LEAP hand joint）"},
        {"p": 3, "src": "Bidex_Manus_Teleop", "path": "ros2/glove/glove/read_and_send_zmq.py",         "note": "Manus glove ZMQ → ROS2 桥接模式"},
    ],
    "ipc": [
        {"p": 1, "src": "ManiUniCon",     "path": "maniunicon/utils/shared_memory/",               "note": "完整 shared memory 实现（ring buffer、queue、ndarray）"},
        {"p": 1, "src": "LeFranX",        "path": "src/lerobot/teleoperators/vr_router_manager.py", "note": "VR 数据路由模式"},
        {"p": 2, "src": "BunnyVisionPro", "path": "bunny_teleop/bimanual_teleop_client.py",         "note": "ZMQ + threading 多进程遥操作数据传递"},
        {"p": 2, "src": "DexUMI",         "path": "dexumi/real_env/common/base.py",                 "note": "ZMQServerBase/ZMQClientBase 通用IPC框架（PUB/SUB+ROUTER/DEALER）"},
        {"p": 2, "src": "DexUMI",         "path": "dexumi/real_env/ring_buffer.py",                 "note": "线程安全 RingBuffer（deque + RLock）"},
    ],
    "recording": [
        {"p": 1, "src": "LeFranX",        "path": "scripts/dual_robot/dual_vr_record.py",                     "note": "VR 录制流程"},
        {"p": 1, "src": "ManiUniCon",     "path": "maniunicon/utils/replay_buffer.py",                         "note": "Replay buffer 设计"},
        {"p": 1, "src": "ManiUniCon",     "path": "maniunicon/sensors/replay.py",                              "note": "传感器数据回放/录制"},
        {"p": 2, "src": "BunnyVisionPro", "path": "real_control/teleop_bimanual_xarm7_ability.py",             "note": "HDF5 录制集成在遥操作主循环中的完整模式"},
        {"p": 2, "src": "DexUMI",         "path": "dexumi/data_recording/record_manager.py",                   "note": "RecorderManager 模式（多Recorder编排+episode生命周期）"},
        {"p": 2, "src": "DexUMI",         "path": "dexumi/data_recording/video_recorder.py",                   "note": "视频录制器（多相机源+帧队列+录制/流分离线程）"},
        {"p": 2, "src": "DexUMI",         "path": "dexumi/data_recording/numeric_recorder.py",                 "note": "数值数据录制器（Zarr+后台录制线程+episode管理）"},
        {"p": 2, "src": "DexUMI",         "path": "real_script/data_collection/record_exoskeleton.py",         "note": "外骨骼数据采集主脚本（多传感器同步录制完整流程）"},
        {"p": 2, "src": "Open-Teach",     "path": "data_collect.py",                                           "note": "多进程组件式数据采集 pipeline"},
    ],
    "deploy": [
        {"p": 1, "src": "LeFranX",    "path": "src/lerobot/scripts/server/robot_client.py",         "note": "策略客户端"},
        {"p": 1, "src": "LeFranX",    "path": "src/lerobot/scripts/server/policy_server.py",        "note": "策略服务端"},
        {"p": 1, "src": "LeFranX",    "path": "scripts/dual_robot/dual_robot_deploy_act.py",        "note": "ACT 策略部署"},
        {"p": 1, "src": "ManiUniCon", "path": "maniunicon/customize/act_wrapper/chunk_wrapper.py",  "note": "Action chunk 包装"},
        {"p": 1, "src": "ManiUniCon", "path": "maniunicon/customize/obs_wrapper/",                  "note": "观测构建包装"},
        {"p": 1, "src": "ManiUniCon", "path": "maniunicon/policies/torch_model.py",                 "note": "Torch 模型加载与推理"},
        {"p": 2, "src": "DexUMI",     "path": "dexumi/real_env/common/policy.py",                   "note": "PolicyServer/Client通用框架（ZMQ IPC+obs_config验证）"},
        {"p": 2, "src": "DexUMI",     "path": "dexumi/real_env/dexumi_policy.py",                   "note": "DexUMIPolicySever（Diffusion Policy加载+推理+归一化）"},
        {"p": 2, "src": "DexUMI",     "path": "dexumi/real_env/real_policy.py",                     "note": "RealPolicy：Diffusion Policy推理（visual obs预处理+action chunk）"},
        {"p": 2, "src": "DexUMI",     "path": "real_script/eval_policy/eval_xhand.py",              "note": "XHand 策略评估脚本（真实硬件部署）"},
        {"p": 2, "src": "Open-Teach", "path": "deploy_server.py",                                   "note": "策略部署服务端（多机器人支持）"},
    ],
    "sensor": [
        {"p": 1, "src": "ManiUniCon", "path": "maniunicon/sensors/realsense.py",   "note": "RealSense 驱动"},
        {"p": 2, "src": "DexUMI",     "path": "dexumi/camera/camera.py",            "note": "Camera ABC + FrameData/FrameNumericData dataclass"},
        {"p": 2, "src": "DexUMI",     "path": "dexumi/camera/realsense_camera.py",  "note": "RealSense 相机实现"},
        {"p": 2, "src": "DexUMI",     "path": "dexumi/camera/oak_camera.py",        "note": "OAK-D 立体相机实现"},
        {"p": 2, "src": "DexUMI",     "path": "dexumi/encoder/fsr.py",              "note": "FSR 力传感器驱动"},
        {"p": 2, "src": "DexUMI",     "path": "dexumi/encoder/xhand_tactile.py",    "note": "XHand 指尖触觉传感器读取"},
        {"p": 2, "src": "Open-Teach", "path": "robot_camera.py",                    "note": "相机传感器抽象"},
        {"p": 2, "src": "Open-Teach", "path": "fish_eye_camera.py",                 "note": "Fish-eye 相机处理"},
    ],
    "planner": [
        {"p": 1, "src": "ManiUniCon",     "path": "maniunicon/utils/ik_solver.py",                              "note": "IK solver 实现"},
        {"p": 1, "src": "ManiUniCon",     "path": "maniunicon/utils/pose_trajectory_interpolator.py",            "note": "位姿轨迹插值"},
        {"p": 1, "src": "ManiUniCon",     "path": "maniunicon/utils/ruckig_utils.py",                            "note": "Ruckig 轨迹生成（可选参考）"},
        {"p": 1, "src": "LeFranX",        "path": "src/lerobot/teleoperators/franka_fer_vr/arm_ik_processor.py", "note": "IK 处理流程"},
        {"p": 2, "src": "DexUMI",         "path": "dexumi/real_env/common/pose_trajectory_interpolator.py",      "note": "PoseTrajectoryInterpolator（scipy Slerp+速度限制+schedule_waypoint）"},
        {"p": 2, "src": "DexUMI",         "path": "dexumi/real_env/common/motor_trajectory_interpolator.py",     "note": "电机轨迹插值（外骨骼手部回放用）"},
        {"p": 3, "src": "Bidex_Manus_Teleop", "path": "ros2/telekinesis/telekinesis/leap_ik.py",                "note": "PyBullet SDLS IK（glove fingertip → robot hand joint）"},
    ],
    "utils": [
        {"p": 1, "src": "ManiUniCon", "path": "maniunicon/utils/filter.py",                 "note": "滤波器实现"},
        {"p": 1, "src": "ManiUniCon", "path": "maniunicon/utils/math_utils.py",              "note": "数学工具"},
        {"p": 1, "src": "ManiUniCon", "path": "maniunicon/utils/pcd_utils.py",               "note": "点云工具"},
        {"p": 1, "src": "ManiUniCon", "path": "maniunicon/utils/timestamp_accumulator.py",   "note": "时间戳对齐"},
        {"p": 2, "src": "DexUMI",     "path": "dexumi/common/precise_sleep.py",              "note": "高精度 sleep/wait（hybrid sleep+spin 最小化 jitter）"},
        {"p": 2, "src": "DexUMI",     "path": "dexumi/common/frame_manager.py",              "note": "FrameRateContext：速率限制器上下文管理器"},
        {"p": 2, "src": "DexUMI",     "path": "dexumi/common/utility/matrix.py",             "note": "矩阵/位姿变换工具（relative/invert transformation）"},
        {"p": 2, "src": "DexUMI",     "path": "dexumi/common/data.py",                       "note": "通用数据结构定义"},
        {"p": 2, "src": "Open-Teach", "path": "openteach/utils/vectorops.py",                "note": "向量运算工具"},
        {"p": 2, "src": "Open-Teach", "path": "openteach/utils/network.py",                  "note": "ZMQ publisher/subscriber 网络工具"},
        {"p": 2, "src": "Open-Teach", "path": "openteach/utils/timer.py",                    "note": "控制循环计时器"},
        {"p": 3, "src": "Bidex_Manus_Teleop", "path": "steamvr/triad_openvr-master/triad_openvr.py", "note": "SteamVR/OpenVR 追踪器数据读取"},
    ],
}

# ── 本项目已知的不采纳清单 ──────────────────────────────────────

NOT_ADOPT: dict[str, str] = {
    "libfranka + Ruckig C++ server": "xArm7 内置伺服控制 (LeFranX)",
    "geofik + Brent 解析式 IK": "已有 MPlib 数值 IK + 碰撞检测 (LeFranX)",
    "LeRobot Parquet": "使用 HDF5 (LeFranX)",
    "LeRobot draccus CLI": "不引入该框架 (LeFranX)",
    "Hydra 配置管理": "使用 @dataclass (Open-Teach)",
    "ROS/ROS2 通信层": "使用 multiprocessing + shared memory (Open-Teach / Bidex)",
    "Vision Pro 专用 API": "本项目使用 Quest 3 (BunnyVisionPro)",
    "Allegro Hand 专用 retargeting": "手型差异，仅参考滤波/限位模式 (Open-Teach)",
    "Manus Core C++ SDK 直连": "不依赖 Manus 数据手套 (Bidex)",
    "Zarr 存储格式": "本项目使用 HDF5 (DexUMI)",
    "ZMQ IPC 通信": "本项目使用 multiprocessing + shared memory (DexUMI)",
    "UR5 RTDE servoL 控制": "xArm7 使用 set_servo_angle_j() (DexUMI)",
    "iPhone ARKit 手腕追踪": "本项目使用 Quest 3 VR (DexUMI)",
    "外骨骼编码器手指追踪": "本项目使用 VR hand tracking + dex-retargeting (DexUMI)",
    "ExoDexterousHand ABC 接口": "VR 遥操作不需要 encoder→motor 映射 (DexUMI)",
}


# ── 命令实现 ──────────────────────────────────────────────────

def list_mapping(module: str | None = None):
    """列出静态模块映射。"""
    modules = [module] if module else sorted(MODULE_MAPPING.keys())
    for mod in modules:
        if mod not in MODULE_MAPPING:
            print(f"\n## {mod}/ — 无静态映射，请用 --keyword 动态搜索")
            continue
        entries = MODULE_MAPPING[mod]
        print(f"\n## {mod}/ — {len(entries)} 条参考:")
        print(f"  {'P':<3} {'来源':<22} {'路径'}")
        print(f"  {'-'*3} {'-'*22} {'-'*50}")
        for e in sorted(entries, key=lambda x: (x["p"], x["src"])):
            full = os.path.join(REF_PROJECTS_BY_NAME[e["src"]].path, e["path"])
            exists = "✓" if os.path.exists(full) else "✗"
            print(f"  P{e['p']}  {e['src']:<22} {e['path']:<50} {exists} {e['note']}")


def search_keyword(modules: list[str] | None, keyword: str, max_priority: int = 3):
    """在参考项目中按优先级搜索含 keyword 的 .py 文件。

    搜索策略:
    1. 先在文件名中匹配 keyword
    2. 再在文件内容中 grep keyword（限制结果数避免噪音）
    """
    projects = [p for p in REF_PROJECTS if p.priority <= max_priority]
    found_files: list[tuple[int, str, str, str]] = []  # (priority, project, rel_path, match_type)

    for proj in projects:
        if not os.path.isdir(proj.path):
            continue

        # 1. 文件名匹配
        try:
            result = subprocess.run(
                ["find", proj.path, "-type", "f", "-name", "*.py",
                 "-path", f"*{keyword}*"],
                capture_output=True, text=True, timeout=30
            )
            for line in result.stdout.strip().split("\n"):
                if line:
                    rel = os.path.relpath(line, proj.path)
                    found_files.append((proj.priority, proj.name, rel, "filename"))
        except subprocess.TimeoutExpired:
            pass

        # 2. 内容 grep（限制深度，避免搜索 node_modules/.git 等）
        try:
            result = subprocess.run(
                ["grep", "-rl", "--include=*.py", "-m", "5", keyword, proj.path],
                capture_output=True, text=True, timeout=60
            )
            for line in result.stdout.strip().split("\n")[:10]:  # 最多 10 个结果
                if line:
                    rel = os.path.relpath(line, proj.path)
                    # 避免重复
                    if not any(f[2] == rel and f[1] == proj.name for f in found_files):
                        found_files.append((proj.priority, proj.name, rel, "grep"))
        except subprocess.TimeoutExpired:
            print(f"  [WARN] grep timeout in {proj.name}", file=sys.stderr)

    # 输出
    found_files.sort(key=lambda x: (x[0], x[1], x[2]))
    if not found_files:
        print(f"未在参考项目中找到与 '{keyword}' 相关的 .py 文件")
        return

    print(f"\n搜索 '{keyword}' — 找到 {len(found_files)} 个文件:\n")
    cur_p = 0
    for p, proj, path, mtype in found_files:
        if p != cur_p:
            cur_p = p
            print(f"  ── P{cur_p} ──")
        tag = "[文件名]" if mtype == "filename" else "[内容]"
        print(f"  P{p} {proj:<22} {tag} {path}")

    # 检查不采纳清单
    for na_pattern, na_reason in NOT_ADOPT.items():
        if keyword.lower() in na_pattern.lower():
            print(f"\n  ⚠ 注意：'{na_pattern}' 已标记为不采纳 — {na_reason}")


def check_exists(module: str):
    """检查某模块的所有静态映射文件是否存在。"""
    if module not in MODULE_MAPPING:
        print(f"模块 '{module}' 不在静态映射表中")
        return
    missing = []
    for e in MODULE_MAPPING[module]:
        proj = REF_PROJECTS_BY_NAME[e["src"]]
        full = os.path.join(proj.path, e["path"])
        if not os.path.exists(full):
            missing.append((e["p"], e["src"], e["path"]))
    if missing:
        print(f"\n{module}/ 映射中有 {len(missing)} 个路径不存在:")
        for p, src, path in missing:
            print(f"  P{p} {src}: {path}")
    else:
        print(f"{module}/ 全部 {len(MODULE_MAPPING[module])} 个映射路径存在")


# ── main ──────────────────────────────────────────────────────

REF_PROJECTS_BY_NAME = {p.name: p for p in REF_PROJECTS}


def main():
    parser = argparse.ArgumentParser(description="跨参考项目搜索工具")
    parser.add_argument("module", nargs="?", help="模块名 (robot/controller/teleop/...)")
    parser.add_argument("--keyword", "-k", help="在参考项目中 grep 的关键词")
    parser.add_argument("--priority", "-p", type=int, choices=[1, 2, 3], default=3,
                        help="搜索的最高优先级 (默认 3)")
    parser.add_argument("--list-mapping", "-l", action="store_true", help="列出静态映射表")
    parser.add_argument("--check-exists", "-c", action="store_true", help="检查映射路径是否存在")

    args = parser.parse_args()

    if args.list_mapping:
        list_mapping(args.module)
        return

    if args.check_exists and args.module:
        check_exists(args.module)
        return

    if not args.module and not args.keyword:
        parser.print_help()
        print("\n示例:")
        print("  python ref-search.py robot                    # 列出 robot 模块映射")
        print("  python ref-search.py --list-mapping           # 列出全部模块映射")
        print("  python ref-search.py ik --keyword SDLS        # 搜索 SDLS IK 实现")
        print("  python ref-search.py --keyword xarm7 -p 2     # 在 P1+P2 中搜索 xarm7")
        print("  python ref-search.py teleop --check-exists    # 检查 teleop 映射文件是否存在")
        return

    if args.module and args.keyword:
        search_keyword([args.module], args.keyword, args.priority)
    elif args.keyword:
        search_keyword(None, args.keyword, args.priority)
    elif args.module:
        list_mapping(args.module)


if __name__ == "__main__":
    main()
