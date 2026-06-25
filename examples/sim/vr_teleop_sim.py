#!/usr/bin/env python3
"""VR 遥操作 xArm7+XHand 仿真脚本（SAPIEN 可视化 + HDF5 数据录制 + RGBD 相机）。

将 VR 手部追踪 → arm mapping → IK → hand retarget → 仿真执行 → 数据录制串联起来。
支持 `--dummy` 模式（无需 VR 头显即可测试），默认启用 EEF RGBD 相机。

用法:
    # Dummy 模式（无需 VR 头显即可测试）
    python examples/sim/vr_teleop_sim.py --dummy

    # 真实 Quest VR 模式
    python examples/sim/vr_teleop_sim.py

    # VR 帧模拟器（正弦轨迹，无需 Quest 设备）— Phase 3.3
    python examples/sim/vr_teleop_sim.py --dummy-vr-sinusoidal

    # 禁用相机（默认启用）
    python examples/sim/vr_teleop_sim.py --dummy --no-camera

    # 无头模式 + 指定数据目录
    python examples/sim/vr_teleop_sim.py --dummy --headless --data-dir ./my_episodes

    # 预录制缓冲区 + 文件分类路由
    python examples/sim/vr_teleop_sim.py --dummy --pre-record-duration 3.0 \\
        --success-dir ./data/success --failure-dir ./data/failure

键位:
    B   - Begin:  开始遥操作 + 录制（重置 mapper，开始 episode）
    C   - Pause:  暂停/恢复遥操作。暂停时冻结 EEF，
                  恢复时自动重新标定 mapper（抵消暂停期间的漂移）。
    H   - Home:   规划路径回 home → IDLE（停止录制并丢弃）
    S   - Stop:   停止录制 → 自动保存 → IDLE
    Q   - Quit:   结束遥操作 → 自动保存 → IDLE；
                  IDLE 下需双击确认退出（2 秒内）
    ESC - 紧急停止（仅 Q 可退出）

状态机:
    IDLE ──B──→ TELEOP_RECORDING  (start_episode, reset_mapper)
    TELEOP_RECORDING ──H──→ return_home → IDLE (discard)
    TELEOP_RECORDING ──C──→ PAUSED (冻结 EEF)
    PAUSED ──C──→ TELEOP_RECORDING (自动 re-anchor mapper，恢复录制)
    PAUSED ──H──→ return_home → IDLE (discard)
    TELEOP_RECORDING ──S──→ IDLE (auto-save episode)
    TELEOP_RECORDING ──Q──→ IDLE (auto-save episode)
    PAUSED ──S──→ IDLE (auto-save episode)
    PAUSED ──Q──→ IDLE (auto-save episode)
    IDLE ──Q──→ 双击确认退出（2 秒内再按确认，其他按键取消）
    ESC (any) ──→ ESTOP (仅 Q 可退出)

Phase 7 新增特性:
    - CollectionLoop: sidecar JSON
    - VRFrameSimulator: --dummy-vr-sinusoidal 正弦手腕轨迹
    - cbreak 键盘: termios + select 替代 pynput（与 KeyboardHandler 同技术栈）
    - 追踪安全: |q_actual - q_cmd| 偏差监控
    - 速度限制: bottleneck scaling 平滑

设计参考:
    - TeleopController (teleop/core/controller.py): 状态机、_compute_action
    - keyboard_teleop_sim.py: SAPIEN viewer、execute_return_home
    - EpisodeRecorder + CollectionLoop: HDF5 录制生命周期
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import sapien.core as sapien

from dexmani_real import ASSET_DIR
from dexmani_real.planning import (
    PlanningProfile,
    TeleopProfile,
    XArm7MotionPlanner,
    XArm7PlannerConfig,
)
from dexmani_real.planning.pose_utils import quat_wxyz_to_rot6d
from dexmani_real.planning.types import Pose  # used in workspace clamp wrapper
from dexmani_real.recording import EpisodeRecorder
from dexmani_real.recording.collection_config import CollectionConfig
from dexmani_real.recording.collection_loop import CollectionLoop
from dexmani_real.robot.types import RobotAction, RobotState
from dexmani_real.simulation import SimRobotConfig, SimRobotInterface
from dexmani_real.simulation.constructor import add_light, create_viewer
from dexmani_real.teleop.core.pipeline import TeleopPipeline
from dexmani_real.teleop.vr.arm_mapper import ArmWristMapper
from dexmani_real.teleop.vr.dummy_tracker import DummyTracker
from dexmani_real.teleop.vr.hand_retarget import XHandRetargeter
from dexmani_real.teleop.control.keyboard import ControlSignal, KeyboardHandler
from dexmani_real.teleop.vr.vr_tracker import QuestHandTracker, VRFrameSimulator
from dexmani_real.utils.rate_limiter import RateLimiter

from examples._test_utils import execute_dense_path, interpolate_waypoints, settle_at_target

# ═══════════════════════════════════════════════════════════════════════════════
# 模块常量
# ═══════════════════════════════════════════════════════════════════════════════

CTRL_HZ = 50.0                     # 与 TeleopControllerConfig.target_hz 一致
CTRL_DT = 1.0 / CTRL_HZ
PHYSICS_STEPS_PER_WP = 20
PHYSICS_STEPS_PER_TICK = 5          # 240Hz → 48Hz effective
INTERP_MAX_STEP_RAD = np.deg2rad(2.0)
VR_FRAME_MAX_AGE_S = 0.2            # 仿真容忍度高于真机（0.1s）：dummy VR 无网络延迟
DEFAULT_DATA_DIR = "./recordings"

# 工作空间边界（world frame，与 RobotInterfaceConfig 默认值保持一致）
WORKSPACE_BOUNDS = np.array([
    [0.28, 0.70],   # x [min, max] m
    [-0.40, 0.40],  # y [min, max] m
    [0.02, 0.55],   # z [min, max] m
])

# ArmWristMapper EEF delta 边界（robot base frame）
EEF_DELTA_BOUNDS = np.array([
    [-0.30, 0.30],
    [-0.30, 0.30],
    [-0.30, 0.30],
])

# VR → robot base 旋转矩阵（FLU → FLU-aligned）
VR_TO_BASE_ROT = np.eye(3)

# 循环超限告警阈值：tick 耗时超过目标周期的 150% 时发出警告。
OVERRUN_WARN_RATIO = 1.5

# 速度限制：每关节最大速度 (deg/s → rad/s)
# 仿真模式下较宽松，真机模式使用 xarm7.py 中的配置值
MAX_QVEL_RAD_S = np.deg2rad([180, 180, 180, 180, 180, 180, 180])

# 追踪安全：command-vs-actual 偏差阈值 (rad)
# 仿真模式下仅记录警告（无真实硬件风险）
TRACKING_DIVERGENCE_THRESHOLD_RAD = 5.0

# Home 关节角
ARM_HOME_QPOS = np.array(
    [-np.pi / 6, -np.pi / 4, 0, np.deg2rad(20), -np.pi, np.deg2rad(25), 0],
    dtype=np.float64,
)
HAND_HOME_QPOS = np.zeros(12, dtype=np.float64)

# ── 仿真 RGBD 相机配置 ──
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_FOVY_RAD = np.deg2rad(60.0)
CAMERA_NEAR = 0.01
CAMERA_FAR = 3.0
CAMERA_EEF_OFFSET_POS = np.array([0.05, 0.0, -0.03], dtype=np.float64)
CAMERA_EEF_OFFSET_QUAT_WXYZ = np.array([0.966, 0.0, 0.259, 0.0], dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# cbreak 键盘输入（与 KeyboardHandler 同技术栈: termios + select）
# ═══════════════════════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════════════════════
# 工作空间工具
# ═══════════════════════════════════════════════════════════════════════════════


def is_in_workspace(pos: np.ndarray) -> bool:
    """检查位置是否在预定义的工作空间边界内。"""
    return bool(
        WORKSPACE_BOUNDS[0, 0] <= pos[0] <= WORKSPACE_BOUNDS[0, 1]
        and WORKSPACE_BOUNDS[1, 0] <= pos[1] <= WORKSPACE_BOUNDS[1, 1]
        and WORKSPACE_BOUNDS[2, 0] <= pos[2] <= WORKSPACE_BOUNDS[2, 1]
    )


def clamp_to_workspace(pos: np.ndarray) -> np.ndarray:
    """将位置夹紧到工作空间边界内。"""
    return np.clip(pos, WORKSPACE_BOUNDS[:, 0], WORKSPACE_BOUNDS[:, 1])


# ═══════════════════════════════════════════════════════════════════════════════
# 仿真状态 → RobotState/RobotAction 构造（供 CollectionLoop / EpisodeRecorder 使用）
# ═══════════════════════════════════════════════════════════════════════════════


def build_robot_state(sim_state: dict) -> RobotState:
    """从 SimRobotInterface.get_state() 构造 RobotState。"""
    eef_quat = sim_state["eef_quat_wxyz"]
    return RobotState(
        arm_qpos=sim_state["arm_qpos"],
        arm_qvel=np.zeros(7, dtype=np.float64),
        arm_tau=np.zeros(7, dtype=np.float64),
        eef_pos=sim_state["eef_pos"],
        eef_quat_wxyz=eef_quat,
        eef_rot6d=quat_wxyz_to_rot6d(eef_quat),
        hand_qpos=sim_state["hand_qpos"],
        hand_tactile_sum=np.zeros((5, 3), dtype=np.float64),
        fingertip_pos=np.zeros((5, 3), dtype=np.float64),
        arm_connected=True,
        hand_connected=True,
        timestamp=sim_state["timestamp"],
    )


def build_robot_action(
    arm_cmd: np.ndarray, hand_cmd: np.ndarray, target_eef_pos: np.ndarray | None = None,
) -> RobotAction:
    """从关节命令构造 RobotAction。"""
    return RobotAction(
        arm_qpos_cmd=arm_cmd,
        hand_qpos_cmd=hand_cmd,
        target_eef_pos=target_eef_pos,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 速度限制（Phase 2.1 — bottleneck scaling，与 XArm7._limit_joint_step 等价）
# ═══════════════════════════════════════════════════════════════════════════════


def velocity_limited_step(
    target: np.ndarray,
    prev_cmd: np.ndarray,
    max_velocities: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Bottleneck scaling: limit per-joint step to max_velocities * dt.

    与 XArm7._limit_joint_step() 算法等价。
    超限时等比缩放所有关节，保持轨迹形状。
    """
    step = target - prev_cmd
    max_step = max_velocities * dt
    ratio = np.max(np.abs(step) / np.maximum(max_step, 1e-8))
    if ratio > 1.0:
        return prev_cmd + step / ratio
    return target


# ═══════════════════════════════════════════════════════════════════════════════
# 仿真 RGBD 相机
# ═══════════════════════════════════════════════════════════════════════════════


def setup_ee_camera(
    scene: sapien.Scene,
    robot: "sapien.Articulation",
    width: int = CAMERA_WIDTH,
    height: int = CAMERA_HEIGHT,
    fovy: float = CAMERA_FOVY_RAD,
    near: float = CAMERA_NEAR,
    far: float = CAMERA_FAR,
) -> tuple[sapien.render.RenderCameraComponent, np.ndarray]:
    """在 EEF link 上挂载仿真 RGBD 相机（eye-in-hand）。"""
    eef_link = robot.find_link_by_name("custom_eef_link")
    if eef_link is None:
        raise RuntimeError("custom_eef_link not found in robot model")

    eef_cam_pose = sapien.Pose(
        p=CAMERA_EEF_OFFSET_POS,
        q=CAMERA_EEF_OFFSET_QUAT_WXYZ,
    )

    cam = scene.add_mounted_camera(
        name="ee_camera",
        mount=eef_link,
        pose=eef_cam_pose,
        width=width,
        height=height,
        fovy=fovy,
        near=near,
        far=far,
    )

    K = np.array([
        [cam.fx, 0, cam.cx],
        [0, cam.fy, cam.cy],
        [0, 0, 1],
    ], dtype=np.float64)

    return cam, K


def capture_camera_frame(
    cam: sapien.render.RenderCameraComponent,
) -> dict[str, np.ndarray | float] | None:
    """拍摄一帧 RGBD 图像。"""
    try:
        cam.take_picture()

        color = cam.get_picture("Color")  # (H, W, 4) float32
        if color is None:
            return None
        rgb = (color[..., :3].clip(0, 1) * 255).astype(np.uint8)

        pos = cam.get_picture("Position")  # (H, W, 4) float32
        if pos is None:
            return None

        cam_entity = cam.get_entity()
        if cam_entity is None:
            return None
        cam_pose = cam_entity.get_pose()
        cam_pos_world = np.asarray(cam_pose.p, dtype=np.float64)
        cam_rot_world = np.asarray(
            cam_pose.to_transformation_matrix()[:3, :3], dtype=np.float64,
        )

        world_xyz = np.asarray(pos[..., :3], dtype=np.float64)
        cam_rel = world_xyz - cam_pos_world
        cam_z_axis = cam_rot_world[:, 2]
        depth = np.abs(np.dot(cam_rel, cam_z_axis))
        depth = np.where(np.all(world_xyz == 0, axis=-1), np.nan, depth)
        depth = depth.astype(np.float32)

        return {
            "rgb": rgb,
            "depth": depth,
            "timestamp": time.perf_counter(),
        }
    except (RuntimeError, ValueError, AttributeError) as e:
        print(f"[Camera] capture failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 回 home 路径执行
# ═══════════════════════════════════════════════════════════════════════════════


def execute_return_home(
    sim: SimRobotInterface,
    planner: XArm7MotionPlanner,
    home_eef: Pose,
    viewer: sapien.Viewer | None = None,
) -> bool:
    """规划并执行从当前位置回 home 的路径。"""
    current_qpos = sim.get_full_qpos()[:7]
    result = planner.plan_path(home_eef, current_qpos)
    if not result.success or result.qpos_path is None:
        print(f"  return_home PLAN FAILED: {result.reason}")
        return False

    path = result.qpos_path
    full = np.vstack([path, ARM_HOME_QPOS])
    dense_joint = interpolate_waypoints(full, INTERP_MAX_STEP_RAD)
    if not any(planner.has_self_collision(q) for q in dense_joint):
        path = full

    dense = interpolate_waypoints(path, INTERP_MAX_STEP_RAD)
    hand_qpos = sim.get_full_qpos()[7:]
    execute_dense_path(sim, dense, viewer, physics_steps_per_wp=PHYSICS_STEPS_PER_WP)
    settle_at_target(
        sim, dense[-1, :7], hand_qpos,
        max_iter=3, physics_steps_per_wp=PHYSICS_STEPS_PER_WP,
    )

    final_qpos = sim.get_full_qpos()[:7]
    joint_err = float(np.max(np.abs(final_qpos - ARM_HOME_QPOS)))
    print(f"  return_home OK  max_joint_err={np.rad2deg(joint_err):.2f}deg")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VR Teleop — xArm7+XHand SAPIEN 仿真 + HDF5 数据录制",
    )
    parser.add_argument(
        "--dummy", action="store_true",
        help="使用 DummyTracker（无需 VR 头显即可测试）",
    )
    parser.add_argument(
        "--dummy-vr-sinusoidal", action="store_true",
        help="使用 VRFrameSimulator 正弦手腕轨迹（Phase 3.3，无需 Quest 设备）",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="无头模式（不创建 viewer 窗口）",
    )
    parser.add_argument(
        "--data-dir", type=str, default=DEFAULT_DATA_DIR,
        help=f"episode 数据输出目录（默认: {DEFAULT_DATA_DIR}）",
    )
    parser.add_argument(
        "--no-camera", action="store_true",
        help="禁用 EEF RGBD 相机（默认启用，headless 模式下自动禁用）",
    )
    parser.add_argument(
        "--camera-width", type=int, default=CAMERA_WIDTH,
        help=f"相机图像宽度（默认: {CAMERA_WIDTH}）",
    )
    parser.add_argument(
        "--camera-height", type=int, default=CAMERA_HEIGHT,
        help=f"相机图像高度（默认: {CAMERA_HEIGHT}）",
    )
    args = parser.parse_args()

    # ── VR Tracker 初始化 ──
    if args.dummy_vr_sinusoidal:
        tracker: DummyTracker | QuestHandTracker | VRFrameSimulator = VRFrameSimulator(
            hz=CTRL_HZ,
            amplitude_m=0.15,
            center_pos=(0.45, 0.0, 0.30),
        )
        print("[VR] VRFrameSimulator initialized (sinusoidal trajectory, no headset required)")
    elif args.dummy:
        tracker = DummyTracker()
        tracker.connect()
        print("[VR] DummyTracker initialized (no headset required)")
    else:
        tracker = QuestHandTracker(
            transport="tcp_server",
            host="0.0.0.0",
            port=8000,
            hand_side="right",
            output_frame="flu",
            max_frame_age_s=VR_FRAME_MAX_AGE_S,
        )
        if not tracker.connect():
            print(f"[VR] QuestHandTracker connect FAILED: {tracker.last_error}")
            print("  Diagnostic: 确保 Quest 上 hand_tracking_sdk 以 TCP Client 模式运行")
            print("  使用 --dummy 可在无 VR 头显时测试")
            sys.exit(1)
        print("[VR] QuestHandTracker connected")

    # ── 仿真初始化 ──
    sim_config = SimRobotConfig(
        headless=args.headless,
        arm_home_qpos=ARM_HOME_QPOS.copy(),
    )
    sim = SimRobotInterface(sim_config)
    if not sim.connect():
        raise RuntimeError(f"Sim connect failed: {sim.last_error_message}")

    root_pose = sim.robot.model.get_root_pose()
    planner = XArm7MotionPlanner(
        XArm7PlannerConfig(
            urdf_path=str(ASSET_DIR / "robots" / "xhand" / "xarm7_xhand_collision.urdf"),
            srdf_path=str(ASSET_DIR / "robots" / "xhand" / "xarm7_xhand_collision_mplib.srdf"),
            base_pose_world=Pose(p=np.array(root_pose.p), q=np.array(root_pose.q)),
        ),
        planning_profile=PlanningProfile(
            max_waypoint_delta_deg=360.0,
            max_ik_delta_deg=(180,) * 7,
            max_pose_error_rot_rad=np.deg2rad(5.0),
            num_random_ik_seeds=30,
            rrt_time_limit=2.0,
            num_rrt_attempts=2,
        ),
        teleop_profile=TeleopProfile(
            teleop_dt=CTRL_DT,
            max_ik_jump_deg=(15, 15, 15, 15, 15, 15, 15),
            max_pose_error_pos_m=0.05,
            max_pose_error_rot_rad=np.deg2rad(5.0),
            differential_ik_max_pos_step_m=0.05,
        ),
    )

    home_eef = planner.compute_eef_pose_world(ARM_HOME_QPOS)

    # 复位到 home
    sim.reset()
    for _ in range(5):
        sim._step_physics(n=10)

    # ── VR Mapper 初始化 ──
    arm_mapper = ArmWristMapper(
        pos_scale=1.0,
        rot_scale=1.0,
        vr_to_base_rot=VR_TO_BASE_ROT,
        eef_delta_bounds=EEF_DELTA_BOUNDS,
        max_delta_rot_rad=1.0,
    )

    hand_retargeter = XHandRetargeter(
        hand_type="right",
        retargeting_type="dexpilot",
        enable_ref_adapter=True,
    )

    # ── TeleopPipeline (shared action computation with real controller) ──
    pipeline = TeleopPipeline(
        arm_mapper, hand_retargeter, planner,
        ema_alpha_arm=1.0,  # No EMA smoothing — use direct IK result in simulation
    )

    # ── Episode Recorder + CollectionLoop 初始化 ──
    recorder = EpisodeRecorder(data_dir=args.data_dir)
    collection_config = CollectionConfig(
        task_label="teleop",
        operator="",
        save_sidecar_json=True,
    )
    collection = CollectionLoop(recorder, collection_config)

    # ── Viewer ──
    viewer: sapien.Viewer | None = None
    if not args.headless:
        add_light(sim.scene)
        viewer = create_viewer(sim.scene, sapien.Pose(
            [0.784, 0.027, 0.630],
            [0.005, -0.233, 0.001, 0.973],
        ))

    # ── 仿真 RGBD 相机（默认启用，headless 或 --no-camera 时禁用）──
    ee_camera: sapien.render.RenderCameraComponent | None = None
    camera_K: np.ndarray | None = None
    camera_enabled = not args.no_camera and not args.headless
    if camera_enabled:
        try:
            ee_camera, camera_K = setup_ee_camera(
                sim.scene, sim.robot.model,
                width=args.camera_width,
                height=args.camera_height,
            )
            print(
                f"[Camera] EEF RGBD camera ready: "
                f"{args.camera_width}x{args.camera_height} "
                f"fx={camera_K[0,0]:.1f} fy={camera_K[1,1]:.1f}"
            )
        except (RuntimeError, ValueError, AttributeError) as e:
            print(f"[Camera] setup failed: {e}，已禁用")
    elif args.no_camera:
        print("[Camera] 已通过 --no-camera 禁用")

    # ── 控制状态变量 ──
    state = "IDLE"               # IDLE | TELEOP_RECORDING | PAUSED
    rate_limiter = RateLimiter(CTRL_HZ)
    prev_arm_cmd = sim.get_full_qpos()[:7].copy()
    prev_hand_cmd = sim.get_full_qpos()[7:].copy()
    ik_fail_total = 0          # 累计 IK 失败次数（只增不减，用于 ik_rate 计算）
    ik_fail_consecutive = 0    # 连续 IK 失败次数（成功时重置，用于诊断）
    retarget_fail_total = 0
    retarget_fail_consecutive = 0
    stale_frame_count = 0
    episode_tick_count = 0       # Phase 1.2: 当前 episode 内的 tick 计数
    episode_idx = 0
    last_status_time = time.perf_counter()
    # VR 丢帧计时（用于软减速）
    lost_since_ns: int | None = None
    # 追踪安全（Phase 1.1）
    consecutive_divergence = 0
    # Idle quit double-tap confirmation (Opt 2)
    idle_quit_pending_ts = 0.0

    print("=" * 60)
    print("VR Teleop — xArm7+XHand SAPIEN 仿真 + 数据录制")
    print(f"  Mode:       {'VRFrameSimulator' if args.dummy_vr_sinusoidal else ('Dummy (no headset)' if args.dummy else 'Quest VR')}")
    print(f"  Data dir:   {args.data_dir}")
    print(f"  Home EEF:   pos={np.round(home_eef.p, 3)}")
    print("  B: 开始遥操作+录制  |  C: 暂停/恢复  |  H: 回 home")
    print("  Q: 退出/丢弃         |  S: 停止/保存   |  ESC: 紧急停止")
    print("=" * 60)

    # ═══════════════════════════════════════════════════════════════════════
    # 主循环（cbreak 键盘 + 状态机）
    # ═══════════════════════════════════════════════════════════════════════

    try:
        with KeyboardHandler() as kb:
            while True:
                tick_start = time.perf_counter()

                # ── Viewer 关闭检测 ──
                if viewer is not None and viewer.closed:
                    print("\n[Viewer] 窗口已关闭，退出...")
                    break

                # ── 键盘状态机（6 键统一方案：B/C/S/H/Q/ESC）──
                for sig in kb.poll(timeout=0.0):
                    # Reset idle quit pending on any non-QUIT signal (Opt 2)
                    if sig != ControlSignal.QUIT:
                        idle_quit_pending_ts = 0.0

                    # ── ESC: 紧急停止（任何状态）──
                    if sig == ControlSignal.EMERGENCY_STOP:
                        if state != "ESTOP":
                            print("\n=== STATE: → EMERGENCY_STOP ===")
                            print("[ESC] Emergency stop — 冻结运动，仅 Q 可退出")
                            if collection.is_recording:
                                collection.stop_episode(
                                    success=False, reason="emergency_stop",
                                    classification="failure",
                                )
                            state = "ESTOP"
                        continue

                    # ── ESTOP guard: 仅 Q 可退出 ──
                    if state == "ESTOP":
                        if sig == ControlSignal.QUIT:
                            print("=== STATE: ESTOP → EXIT ===")
                            raise KeyboardInterrupt
                        else:
                            print(f"[ESTOP] 仅 Q 可退出，忽略: {sig.value}")
                        continue

                    # ── Q: 上下文重载（停止录制 / 丢弃 / 退出）──
                    if sig == ControlSignal.QUIT:
                        if state in ("TELEOP_RECORDING", "PAUSED"):
                            # Auto-save → IDLE (matches TeleopController behavior)
                            ik_rate = 1.0 - min(
                                ik_fail_total / max(episode_tick_count, 1), 1.0,
                            )
                            vr_drop = stale_frame_count / max(episode_tick_count, 1)
                            classification = (
                                "failure" if ik_fail_total > 10
                                else "partial" if ik_fail_total > 0
                                else "success"
                            )
                            path = collection.stop_episode(
                                success=(classification != "failure"),
                                classification=classification,
                                ik_success_rate=round(ik_rate, 4),
                                vr_drop_rate=round(vr_drop, 4),
                            )
                            old_state = state
                            state = "IDLE"
                            if path:
                                print(f"[Save] Episode saved to {path}")
                            print(f"\n=== STATE: {old_state} → IDLE (auto-saved) ===")
                            print(f"[Recorder] 录制已停止 ({collection.frame_count} 帧)")
                            print(f"  classification={classification} ik_rate={ik_rate:.2%} vr_drop={vr_drop:.2%}")
                            print("[State] IDLE — 按 B 开始新的遥操作")
                        else:
                            # IDLE: double-tap Q to confirm exit (Opt 2)
                            now = time.perf_counter()
                            if idle_quit_pending_ts > 0 and (now - idle_quit_pending_ts) < 2.0:
                                print("\n=== STATE: IDLE → EXIT (double-tap confirmed) ===")
                                raise KeyboardInterrupt
                            else:
                                idle_quit_pending_ts = now
                                print("[QUIT] 再按一次 Q 确认退出（2 秒内有效，其他按键取消）")

                    # ── B: Begin — 开始遥操作 + 录制（合并原 T+R）──
                    if sig == ControlSignal.BEGIN:
                        if state == "IDLE":
                            frame = tracker.get_latest()
                            sim_state = sim.get_state()
                            if frame is not None:
                                # 重置 arm mapper（锚定 VR↔EEF 参考系）
                                arm_mapper.reset(
                                    wrist_pos=frame["wrist_pos"],
                                    wrist_quat_wxyz=frame["wrist_quat_wxyz"],
                                    eef_pos=sim_state["eef_pos"],
                                    eef_quat_wxyz=sim_state["eef_quat_wxyz"],
                                )
                                # 开始录制 episode（CollectionLoop 管理）
                                episode_idx += 1
                                collection.start_episode(
                                    task_label="teleop", operator="",
                                    camera_K=camera_K,
                                )
                                state = "TELEOP_RECORDING"
                                ik_fail_total = 0
                                ik_fail_consecutive = 0
                                retarget_fail_total = 0
                                retarget_fail_consecutive = 0
                                stale_frame_count = 0
                                episode_tick_count = 0
                                consecutive_divergence = 0
                                lost_since_ns = None
                                print(
                                    f"\n=== STATE: IDLE → TELEOP_RECORDING ==="
                                    f"\n[State] TELEOP_RECORDING  episode=#{episode_idx}"
                                    f"  EEF={np.round(sim_state['eef_pos'], 3)}"
                                )
                            else:
                                print("[State] 无法获取 VR 帧，请确保 tracker 已连接")

                    # ── H: Home — 规划路径回 home → IDLE（对齐 controller.py）──
                    if sig == ControlSignal.HOME:
                        if state in ("TELEOP_RECORDING", "PAUSED"):
                            print(f"\n=== STATE: {state} → HOME ===")
                            print("[Home] 规划回 home...")
                            # Stop recording if active (discard for home)
                            if collection.is_recording:
                                collection.stop_episode(success=False, reason="home",
                                                        classification="failure")
                            execute_return_home(sim, planner, home_eef, viewer)
                            prev_arm_cmd = sim.get_full_qpos()[:7].copy()
                            prev_hand_cmd = sim.get_full_qpos()[7:].copy()
                            ik_fail_total = 0
                            ik_fail_consecutive = 0
                            retarget_fail_total = 0
                            retarget_fail_consecutive = 0
                            stale_frame_count = 0
                            episode_tick_count = 0
                            consecutive_divergence = 0
                            lost_since_ns = None
                            state = "IDLE"
                            print("=== STATE: HOME → IDLE ===")
                            print("[State] IDLE — 按 B 开始新的遥操作")
                        elif state == "IDLE":
                            print("\n=== STATE: IDLE → HOME ===")
                            print("[Home] 规划回 home...")
                            execute_return_home(sim, planner, home_eef, viewer)
                            prev_arm_cmd = sim.get_full_qpos()[:7].copy()
                            prev_hand_cmd = sim.get_full_qpos()[7:].copy()
                            print("=== STATE: HOME → IDLE ===")

                    # ── C: Pause / Resume — 冻结/恢复遥操作 ──
                    if sig == ControlSignal.PAUSE:
                        if state == "TELEOP_RECORDING":
                            state = "PAUSED"
                            print(
                                f"\n=== STATE: TELEOP_RECORDING → PAUSED ==="
                                f"\n[Pause] 遥操作已暂停  |  EEF 冻结  |  "
                                f"录制暂停 ({collection.frame_count} 帧)"
                                f"\n        按 C 恢复，按 H 回 home，按 Q 退出"
                            )
                        elif state == "PAUSED":
                            # 恢复：重新锚定 mapper，抵消暂停期间的 VR 漂移
                            frame = tracker.get_latest()
                            sim_state = sim.get_state()
                            if frame is not None:
                                arm_mapper.reset(
                                    wrist_pos=frame["wrist_pos"],
                                    wrist_quat_wxyz=frame["wrist_quat_wxyz"],
                                    eef_pos=sim_state["eef_pos"],
                                    eef_quat_wxyz=sim_state["eef_quat_wxyz"],
                                )
                                state = "TELEOP_RECORDING"
                                ik_fail_total = 0
                                ik_fail_consecutive = 0
                                retarget_fail_total = 0
                                retarget_fail_consecutive = 0
                                stale_frame_count = 0
                                consecutive_divergence = 0
                                lost_since_ns = None
                                print(
                                    f"\n=== STATE: PAUSED → TELEOP_RECORDING ==="
                                    f"\n[Resume] 遥操作已恢复  |  Mapper 已重新锚定  |  "
                                    f"EEF={np.round(sim_state['eef_pos'], 3)}"
                                )
                            else:
                                lost_msg = ""
                                if lost_since_ns is not None:
                                    lost_dur = (time.perf_counter_ns() - lost_since_ns) * 1e-9
                                    lost_msg = f" (VR 丢失 {lost_dur:.1f}s)"
                                print(f"[Resume] 无法获取 VR 帧，恢复失败{lost_msg}")
                                print("         检查头显连接 / HTS SDK。按 H 回 home，Q 停止。")
                    # ── S: Stop — 停止录制 → 自动保存 → IDLE ──
                    if sig == ControlSignal.STOP:
                        if state in ("TELEOP_RECORDING", "PAUSED"):
                            # Auto-save → IDLE (matches TeleopController behavior)
                            ik_rate = 1.0 - min(
                                ik_fail_total / max(episode_tick_count, 1), 1.0,
                            )
                            vr_drop = stale_frame_count / max(episode_tick_count, 1)
                            classification = (
                                "failure" if ik_fail_total > 10
                                else "partial" if ik_fail_total > 0
                                else "success"
                            )
                            path = collection.stop_episode(
                                success=(classification != "failure"),
                                classification=classification,
                                ik_success_rate=round(ik_rate, 4),
                                vr_drop_rate=round(vr_drop, 4),
                            )
                            old_state = state
                            state = "IDLE"
                            if path:
                                print(f"[Save] Episode saved to {path}")
                            print(f"\n=== STATE: {old_state} → IDLE (auto-saved) ===")
                            print(f"[Recorder] 录制已停止 ({collection.frame_count} 帧)")
                            print(f"  classification={classification}")
                            print("[State] IDLE — 按 B 开始新的遥操作")

                # ═══════════════════════════════════════════════════════════
                # TELEOP_RECORDING 控制 tick
                # ═══════════════════════════════════════════════════════════

                if state == "TELEOP_RECORDING":
                    episode_tick_count += 1

                    # 1. 读取 VR 帧
                    frame = tracker.get_latest()

                    if frame is None:
                        # ── VR 帧过期处理 ──
                        stale_frame_count += 1
                        now_ns = time.perf_counter_ns()
                        if lost_since_ns is None:
                            lost_since_ns = now_ns
                        lost_duration_s = (now_ns - lost_since_ns) * 1e-9

                        if stale_frame_count == 1:
                            print(f"[VR] 帧过期或不可用 (age > {VR_FRAME_MAX_AGE_S}s)，"
                                  f"软减速中...")

                        # Hold current position
                        sim_state = sim.get_state()
                        arm_blend, hand_blend = TeleopPipeline.soft_deceleration(
                            sim_state["arm_qpos"], sim_state["hand_qpos"],
                        )
                        full_cmd = np.concatenate([arm_blend, hand_blend])
                        sim.robot.balance_passive_force()
                        sim.robot.apply_action(full_cmd)
                        sim._step_physics(n=PHYSICS_STEPS_PER_TICK)

                        # 更新上一帧命令为衰减后的值（保证连续性）
                        prev_arm_cmd = arm_blend.copy()
                        prev_hand_cmd = hand_blend.copy()

                    else:
                        # ── VR 帧有效：正常控制 ──
                        lost_since_ns = None      # 重置丢帧计时
                        stale_frame_count = 0
                        sim_state = sim.get_state()

                        # 2. 计算遥操作命令（使用 TeleopPipeline — 与真机共享逻辑）
                        action, status = pipeline.compute_action(
                            vr_frame=frame,
                            current_arm_qpos=sim_state["arm_qpos"],
                            current_hand_qpos=sim_state["hand_qpos"],
                            prev_arm_cmd=prev_arm_cmd,
                            prev_hand_cmd=prev_hand_cmd,
                            check_workspace=is_in_workspace,
                            clamp_workspace_pos=clamp_to_workspace,
                            last_arm_cmd=prev_arm_cmd,
                        )
                        arm_cmd = action.arm_qpos_cmd
                        hand_cmd = action.hand_qpos_cmd
                        target_eef_pos = action.target_eef_pos

                        # 更新失败计数（用于状态打印 + episode 元数据）
                        # ik_fail_total/retarget_fail_total: 累计总数，只增不减，用于速率计算
                        # ik_fail_consecutive/retarget_fail_consecutive: 连续失败，成功时重置
                        if not status["ik_ok"]:
                            ik_fail_total += 1
                            ik_fail_consecutive += 1
                        else:
                            ik_fail_consecutive = 0
                        if not status["retarget_ok"]:
                            retarget_fail_total += 1
                            retarget_fail_consecutive += 1
                        else:
                            retarget_fail_consecutive = 0

                        # ── 速度限制（Phase 2.1 — bottleneck scaling）──
                        arm_cmd = velocity_limited_step(
                            arm_cmd, prev_arm_cmd, MAX_QVEL_RAD_S, CTRL_DT,
                        )

                        # 3. 仿真相机帧捕获（可选）
                        camera_frame = None
                        T_base_eef = None
                        if ee_camera is not None:
                            camera_frame = capture_camera_frame(ee_camera)
                            try:
                                eef_link = sim.robot.model.find_link_by_name("custom_eef_link")
                                if eef_link is not None:
                                    eef_pose = eef_link.get_entity_pose()
                                    T_base_eef = np.eye(4, dtype=np.float64)
                                    T_base_eef[:3, 3] = np.asarray(eef_pose.p, dtype=np.float64)
                                    T_base_eef[:3, :3] = np.asarray(
                                        eef_pose.to_transformation_matrix()[:3, :3],
                                        dtype=np.float64,
                                    )
                            except (RuntimeError, AttributeError) as e:
                                print(f"[Camera] T_base_eef 计算失败: {e}")

                        # 4. 录制帧（使用 CollectionLoop）
                        if collection.is_recording:
                            try:
                                robot_state = build_robot_state(sim_state)
                                robot_action = build_robot_action(
                                    arm_cmd, hand_cmd, target_eef_pos,
                                )
                                collection.record_frame(
                                    state=robot_state,
                                    action=robot_action,
                                    vr_frame=frame,
                                    camera_frame=camera_frame,
                                    T_base_eef=T_base_eef,
                                )
                            except (ValueError, OSError) as e:
                                print(f"[Recorder] record_frame 失败: {e}")

                        # 5. 应用动作到仿真
                        full_cmd = np.concatenate([arm_cmd, hand_cmd])
                        sim.robot.balance_passive_force()
                        sim.robot.apply_action(full_cmd)
                        sim._step_physics(n=PHYSICS_STEPS_PER_TICK)

                        # ── 追踪安全（Phase 1.1 — 仿真仅记录警告）──
                        actual_qpos = sim.get_full_qpos()[:7]
                        tracking_err = np.max(np.abs(actual_qpos - arm_cmd))
                        if tracking_err > TRACKING_DIVERGENCE_THRESHOLD_RAD:
                            consecutive_divergence += 1
                            if consecutive_divergence == 1:
                                print(
                                    f"[SAFETY] Tracking divergence: "
                                    f"max_err={tracking_err:.1f}rad "
                                    f"(sim — warning only)"
                                )
                            if consecutive_divergence >= 3:
                                print(
                                    f"[SAFETY] Tracking divergence persistent "
                                    f"({consecutive_divergence} frames) — "
                                    f"would E-Stop on real hardware"
                                )
                        else:
                            consecutive_divergence = 0

                        # 更新上一帧命令
                        prev_arm_cmd = arm_cmd.copy()
                        prev_hand_cmd = hand_cmd.copy()

                elif state == "PAUSED":
                    # PAUSED: 冻结 EEF（PD 保持当前位置），不读取 VR、不录制
                    full_cmd = np.concatenate([prev_arm_cmd, prev_hand_cmd])
                    sim.robot.balance_passive_force()
                    sim.robot.apply_action(full_cmd)
                    sim._step_physics(n=PHYSICS_STEPS_PER_TICK)

                elif state == "ESTOP":
                    # ESTOP: 冻结 EEF，仅推进物理步进，仅 Q 可退出
                    full_cmd = np.concatenate([prev_arm_cmd, prev_hand_cmd])
                    sim.robot.balance_passive_force()
                    sim.robot.apply_action(full_cmd)
                    sim._step_physics(n=PHYSICS_STEPS_PER_TICK)

                elif state == "IDLE":
                    # IDLE: 仅推进物理（PD 保持当前位置）
                    sim._step_physics(n=1)

                # ── 渲染 ──
                if viewer is not None:
                    sim.scene.update_render()
                    viewer.render()

                # ── 定期状态打印（每 2 秒）──
                now = time.perf_counter()
                if now - last_status_time > 2.0:
                    sim_state = sim.get_state()
                    if state == "TELEOP_RECORDING":
                        rec_frames = collection.frame_count
                        ep_elapsed = now - tick_start + episode_tick_count * CTRL_DT  # approximate
                        # Use actual episode duration based on tick count
                        ep_dur = episode_tick_count * CTRL_DT
                        actual_fps = episode_tick_count / max(ep_dur, 0.001)
                        status = (
                            f"[REC] EEF={np.round(sim_state['eef_pos'], 3)}"
                            f"  frames={rec_frames}  tick={episode_tick_count}"
                            f"  ep_t={ep_dur:.1f}s  fps={actual_fps:.1f}"
                        )
                        if ik_fail_consecutive > 0:
                            status += f"  IK_fail(consec)={ik_fail_consecutive}"
                        if retarget_fail_consecutive > 0:
                            status += f"  retarget_fail(consec)={retarget_fail_consecutive}"
                        if stale_frame_count > 0:
                            lost_s = (time.perf_counter_ns() - (lost_since_ns or 0)) * 1e-9
                            status += f"  stale={stale_frame_count} (lost={lost_s:.1f}s)"
                    elif state == "PAUSED":
                        rec_frames = collection.frame_count
                        ep_dur = episode_tick_count * CTRL_DT
                        # VR health info
                        frame = tracker.get_latest()
                        vr_status = ""
                        if frame is None:
                            if lost_since_ns is not None:
                                lost_dur = (time.perf_counter_ns() - lost_since_ns) * 1e-9
                                vr_status = f"  VR丢失={lost_dur:.1f}s"
                            else:
                                vr_status = "  VR: 无信号"
                        else:
                            vr_status = "  VR: OK"
                        status = (
                            f"[PAUSED] EEF={np.round(sim_state['eef_pos'], 3)}"
                            f"  frames={rec_frames}  ep_t={ep_dur:.1f}s"
                            f"{vr_status}"
                            f"  |  C: 恢复  H: 回 home  Q: 退出"
                        )
                    elif state == "ESTOP":
                        status = "[ESTOP] 紧急停止 — 仅 Q 可退出"
                    else:
                        status = f"[IDLE] EEF={np.round(sim_state['eef_pos'], 3)}"
                    print(status)
                    last_status_time = now

                # ── 循环超限检测 ──
                tick_elapsed_ms = (time.perf_counter() - tick_start) * 1000.0
                target_ms = CTRL_DT * 1000.0
                if tick_elapsed_ms > target_ms * OVERRUN_WARN_RATIO:
                    print(
                        f"[Overrun] tick={tick_elapsed_ms:.1f}ms "
                        f"target={target_ms:.1f}ms "
                        f"(IK 耗时异常或系统负载过高)"
                    )

                # ── 频率限制 (50Hz) ──
                rate_limiter.wait()

    except KeyboardInterrupt:
        print("\n\n[Stopping] User interrupt.")

    finally:
        # ── 清理（始终执行）──
        print("\nCleaning up...")

        # 如果还在录制中（异常退出），尝试保存
        if collection.is_recording:
            try:
                path = collection.stop_episode(
                    success=False, reason="abnormal_exit",
                    classification="failure",
                )
                if path:
                    print(f"[Recorder] 异常退出，episode 已保存至 {path}")
            except (ValueError, OSError):
                pass

        tracker.disconnect() if hasattr(tracker, 'disconnect') else None
        sim.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
