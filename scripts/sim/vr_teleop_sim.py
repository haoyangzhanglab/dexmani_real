#!/usr/bin/env python3
"""VR 遥操作 xArm7+XHand 仿真脚本（SAPIEN 可视化 + HDF5 数据录制 + RGBD 相机）。

将 VR 手部追踪 → arm mapping → IK → hand retarget → 仿真执行 → 数据录制串联起来。
支持 `--dummy` 模式（无需 VR 头显即可测试），默认启用 EEF RGBD 相机。

用法:
    # Dummy 模式（无需 VR 头显即可测试）
    python scripts/sim/vr_teleop_sim.py --dummy

    # 真实 Quest VR 模式
    python scripts/sim/vr_teleop_sim.py

    # 禁用相机（默认启用）
    python scripts/sim/vr_teleop_sim.py --dummy --no-camera

    # 无头模式 + 指定数据目录
    python scripts/sim/vr_teleop_sim.py --dummy --headless --data-dir ./my_episodes

键位:
    B   - Begin:  开始遥操作（重置 mapper，开始录制 episode）
    R   - Return: 规划路径回 home（保持在当前状态）
    C   - Pause:  暂停/恢复遥操作。暂停时冻结 EEF、暂停录制，
                  恢复时自动重新标定 mapper（抵消暂停期间的漂移）。
                  这是遥操作中最实用的功能——接电话、思考策略、
                  调整坐姿时按 C 暂停，回来后按 C 继续。
    Q   - Quit:   结束遥操作 → 提示保存/丢弃

    S   - Save:   保存录制的 episode 到 HDF5（仅在 SAVE_PROMPT 下）
    N   - No save: 丢弃当前 episode（仅在 SAVE_PROMPT 下）

状态机:
    IDLE ──B──→ TELEOP_RECORDING  (start_episode, reset_mapper)
    TELEOP_RECORDING ──R──→ return_home (保持在 TELEOP_RECORDING)
    TELEOP_RECORDING ──C──→ PAUSED (冻结 EEF，暂停录制)
    PAUSED ──C──→ TELEOP_RECORDING (自动 re-anchor mapper，恢复录制)
    PAUSED ──R──→ return_home (保持在 PAUSED)
    TELEOP_RECORDING ──Q──→ SAVE_PROMPT (stop_episode)
    PAUSED ──Q──→ SAVE_PROMPT (stop_episode)
    SAVE_PROMPT ──S──→ IDLE (save episode to disk)
    SAVE_PROMPT ──N──→ IDLE (discard episode)

设计参考:
    - TeleopController (teleop/core/controller.py): 状态机、_compute_action、
      hand retarget with MANO frame transform、soft deceleration on VR stale、
      workspace clamp+re-IK、loop overrun detection
    - keyboard_teleop_sim.py: SAPIEN viewer、GlobalKeyState、execute_return_home
    - EpisodeRecorder (recording/episode_recorder.py): HDF5 录制生命周期
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import sapien.core as sapien
from pynput import keyboard as pynput_keyboard

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
from dexmani_real.robot.types import RobotAction, RobotState
from dexmani_real.simulation import SimRobotConfig, SimRobotInterface
from dexmani_real.simulation.constructor import add_light, create_viewer
from dexmani_real.teleop.core.pipeline import TeleopPipeline
from dexmani_real.teleop.vr.arm_mapper import ArmWristMapper
from dexmani_real.teleop.vr.dummy_tracker import DummyTracker
from dexmani_real.teleop.vr.hand_retarget import XHandRetargeter
from dexmani_real.teleop.vr.vr_tracker import QuestHandTracker
from dexmani_real.utils.rate_limiter import RateLimiter

from scripts._test_utils import execute_dense_path, interpolate_waypoints, settle_at_target

# ═══════════════════════════════════════════════════════════════════════════════
# 模块常量
# ═══════════════════════════════════════════════════════════════════════════════

CTRL_HZ = 50.0
CTRL_DT = 1.0 / CTRL_HZ
PHYSICS_STEPS_PER_WP = 20
PHYSICS_STEPS_PER_TICK = 5   # 240Hz → 48Hz effective
INTERP_MAX_STEP_RAD = np.deg2rad(2.0)
VR_FRAME_MAX_AGE_S = 0.2
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
# 参考 BunnyVisionPro wait_until_next_control_signal。
OVERRUN_WARN_RATIO = 1.5

# Home 关节角
ARM_HOME_QPOS = np.array(
    [-np.pi / 6, -np.pi / 4, 0, np.deg2rad(20), -np.pi, np.deg2rad(25), 0],
    dtype=np.float64,
)
HAND_HOME_QPOS = np.zeros(12, dtype=np.float64)

# ── 仿真 RGBD 相机配置 ──
# 相机挂载在 EEF link 上（eye-in-hand），位姿偏移使相机朝向前下方以观察手部工作区域
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_FOVY_RAD = np.deg2rad(60.0)   # 垂直 FOV（标准视场角）
CAMERA_NEAR = 0.01
CAMERA_FAR = 3.0
# 相机在 EEF 坐标系下的位姿偏移
# EEF 坐标系: x-forward, y-left, z-up (FLU)
# 相机放在 EEF 前方 5cm、下方 3cm 处，向下倾斜约 30 度
CAMERA_EEF_OFFSET_POS = np.array([0.05, 0.0, -0.03], dtype=np.float64)
CAMERA_EEF_OFFSET_QUAT_WXYZ = np.array([0.966, 0.0, 0.259, 0.0], dtype=np.float64)  # 绕 Y 轴 -30°


# ═══════════════════════════════════════════════════════════════════════════════
# GlobalKeyState — 非阻塞键盘监听
# ═══════════════════════════════════════════════════════════════════════════════


class GlobalKeyState:
    """非阻塞全局键盘状态追踪（后台 listener 线程）。"""

    def __init__(self) -> None:
        self._pressed: set = set()
        self._lock = threading.Lock()
        self._listener = pynput_keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release,
        )
        self._listener.daemon = True
        self._listener.start()

    def _on_press(self, key) -> None:
        with self._lock:
            ch = getattr(key, "char", None)
            self._pressed.add(ch.lower() if ch else key)

    def _on_release(self, key) -> None:
        with self._lock:
            ch = getattr(key, "char", None)
            self._pressed.discard(ch.lower() if ch else key)

    def is_pressed(self, key) -> bool:
        if isinstance(key, str) and len(key) == 1:
            key = key.lower()
        with self._lock:
            return key in self._pressed

    def clear(self, key) -> None:
        with self._lock:
            self._pressed.discard(key)

    def stop(self) -> None:
        try:
            self._listener.stop()
        except Exception:
            pass


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
# 仿真状态 → RobotState/RobotAction 构造（供 EpisodeRecorder 使用）
# ═══════════════════════════════════════════════════════════════════════════════


def build_robot_state(sim_state: dict) -> RobotState:
    """从 SimRobotInterface.get_state() 构造 RobotState。

    仿真中不存在真实扭矩/电流/触觉等硬件量，使用合理默认值填充。
    """
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
    """在 EEF link 上挂载仿真 RGBD 相机（eye-in-hand）。

    Args:
        scene: SAPIEN 场景
        robot: XArm7XHand 的 SAPIEN articulation model
        width, height: 图像分辨率
        fovy: 垂直 FOV (rad)
        near, far: 近/远裁剪面 (m)

    Returns:
        (camera, K): RenderCameraComponent + 3×3 相机内参矩阵
    """
    eef_link = robot.find_link_by_name("custom_eef_link")
    if eef_link is None:
        raise RuntimeError("custom_eef_link not found in robot model")

    # 相机在 EEF 坐标系下的偏移位姿
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

    # 内参矩阵
    K = np.array([
        [cam.fx, 0, cam.cx],
        [0, cam.fy, cam.cy],
        [0, 0, 1],
    ], dtype=np.float64)

    return cam, K


def capture_camera_frame(
    cam: sapien.render.RenderCameraComponent,
) -> dict[str, np.ndarray | float] | None:
    """拍摄一帧 RGBD 图像。

    Args:
        cam: SAPIEN 相机组件

    Returns:
        {"rgb": (H,W,3) uint8, "depth": (H,W) float32, "timestamp": float} 或 None
    """
    try:
        cam.take_picture()

        # RGB: float32 RGBA [0,1] → uint8 RGB [0,255]
        color = cam.get_picture("Color")  # (H, W, 4) float32
        if color is None:
            return None
        rgb = (color[..., :3].clip(0, 1) * 255).astype(np.uint8)

        # Depth: 从 Position 纹理提取相机坐标系下的 Z 分量
        # Position 纹理: (H, W, 4) float32，前 3 通道为世界空间 XYZ
        pos = cam.get_picture("Position")  # (H, W, 4) float32
        if pos is None:
            return None

        # 相机世界位姿
        cam_entity = cam.get_entity()
        if cam_entity is None:
            return None
        cam_pose = cam_entity.get_pose()
        cam_pos_world = np.asarray(cam_pose.p, dtype=np.float64)       # (3,)
        cam_rot_world = np.asarray(
            cam_pose.to_transformation_matrix()[:3, :3], dtype=np.float64,
        )  # (3,3)

        # 世界坐标 → 相机坐标 → Z 分量即为深度
        world_xyz = np.asarray(pos[..., :3], dtype=np.float64)  # (H, W, 3)
        cam_rel = world_xyz - cam_pos_world                     # (H, W, 3)
        # depth = cam_rel 在相机 Z 轴上的投影量（前向为正）
        cam_z_axis = cam_rot_world[:, 2]  # (3,) 相机的 Z 轴在世界系下的方向
        depth = np.abs(np.dot(cam_rel, cam_z_axis))  # (H, W) — 标量深度

        # 用 NaN 标记无效区域（Position 纹理全零 = 无穷远/无命中）
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
    """规划并执行从当前位置回 home 的路径。

    Returns:
        True 如果成功到达 home，False 如果规划失败。
    """
    current_qpos = sim.get_full_qpos()[:7]
    result = planner.plan_path(home_eef, current_qpos)
    if not result.success or result.qpos_path is None:
        print(f"  return_home PLAN FAILED: {result.reason}")
        return False

    path = result.qpos_path
    # 如果最后一段（path → HOME）无碰撞，直接追加 HOME 避免 deceleration 迟疑
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

    keys = GlobalKeyState()

    # ── VR Tracker 初始化 ──
    if args.dummy:
        tracker: DummyTracker | QuestHandTracker = DummyTracker()
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
            max_ik_jump_deg=(8, 8, 8, 8, 12, 12, 18),
            max_pose_error_pos_m=0.01,
            max_pose_error_rot_rad=np.deg2rad(5.0),
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

    # ── Episode Recorder 初始化 ──
    recorder = EpisodeRecorder(data_dir=args.data_dir)

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
    # else: headless mode, camera auto-disabled (no render context)

    # ── 控制状态变量 ──
    state = "IDLE"               # IDLE | TELEOP_RECORDING | PAUSED | SAVE_PROMPT
    rate_limiter = RateLimiter(CTRL_HZ)
    prev_arm_cmd = sim.get_full_qpos()[:7].copy()
    prev_hand_cmd = sim.get_full_qpos()[7:].copy()
    ik_fail_count = 0
    retarget_fail_count = 0
    stale_frame_count = 0
    episode_idx = 0
    last_episode_path: str | None = None  # set by Q handler, read by S/N handlers
    last_status_time = time.perf_counter()
    # VR 丢帧计时（用于软减速）
    lost_since_ns: int | None = None  # 第一次丢帧的时间戳 (perf_counter_ns)

    print("=" * 60)
    print("VR Teleop — xArm7+XHand SAPIEN 仿真 + 数据录制")
    print(f"  Mode:       {'Dummy (no headset)' if args.dummy else 'Quest VR'}")
    print(f"  Data dir:   {args.data_dir}")
    print(f"  Home EEF:   pos={np.round(home_eef.p, 3)}")
    print("  B: 开始遥操作+录制  |  Q: 结束  |  R: 回 home")
    print("  C: 暂停/恢复         |  S: 保存  |  N: 丢弃")
    print("=" * 60)

    # ═══════════════════════════════════════════════════════════════════════
    # 主循环
    # ═══════════════════════════════════════════════════════════════════════

    try:
        while True:
            tick_start = time.perf_counter()

            # ── Viewer 关闭检测 ──
            if viewer is not None and viewer.closed:
                print("\n[Viewer] 窗口已关闭，退出...")
                break

            # ── 键盘状态机 ──

            # Q: 退出（在不同状态下行为不同）
            if keys.is_pressed('q') or keys.is_pressed('Q'):
                keys.clear('q')
                keys.clear('Q')
                if state in ("TELEOP_RECORDING", "PAUSED"):
                    # 停止录制，进入保存提示
                    path = recorder.stop_episode(success=True)
                    state = "SAVE_PROMPT"
                    print(f"\n[Recorder] 录制已停止 ({recorder.frame_count} 帧)")
                    print("  S: 保存 episode  |  N: 丢弃  |  Q: 完全退出")
                    last_episode_path = path
                elif state == "SAVE_PROMPT":
                    # 丢弃并完全退出
                    if last_episode_path is not None:
                        try:
                            Path(last_episode_path).unlink(missing_ok=True)
                            print(f"[Discard] 已删除 {last_episode_path}")
                        except OSError:
                            pass
                    break
                else:
                    # IDLE: 直接退出
                    break

            # B: Begin — 开始遥操作 + 录制
            if keys.is_pressed('b') or keys.is_pressed('B'):
                keys.clear('b')
                keys.clear('B')
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
                        # 开始录制 episode
                        episode_idx += 1
                        recorder.start_episode(
                            task_label="teleop", operator="",
                            camera_K=camera_K,
                        )
                        state = "TELEOP_RECORDING"
                        ik_fail_count = 0
                        retarget_fail_count = 0
                        stale_frame_count = 0
                        lost_since_ns = None
                        print(
                            f"\n[State] TELEOP_RECORDING  episode=#{episode_idx}"
                            f"  EEF={np.round(sim_state['eef_pos'], 3)}"
                        )
                    else:
                        print("[State] 无法获取 VR 帧，请确保 tracker 已连接")
                elif state == "SAVE_PROMPT":
                    print("[State] 请先按 S 保存或 N 丢弃当前 episode")

            # R: Return home（保持在当前状态）
            if keys.is_pressed('r') or keys.is_pressed('R'):
                keys.clear('r')
                keys.clear('R')
                if state in ("TELEOP_RECORDING", "PAUSED"):
                    state_before = state
                    print("\n[Return] 规划回 home...")
                    execute_return_home(sim, planner, home_eef, viewer)
                    prev_arm_cmd = sim.get_full_qpos()[:7].copy()
                    prev_hand_cmd = sim.get_full_qpos()[7:].copy()
                    ik_fail_count = 0
                    retarget_fail_count = 0
                    lost_since_ns = None
                    if state_before == "PAUSED":
                        print("[Return] 已回 home，仍在暂停中（按 C 恢复）")
                    else:
                        print("[Return] 已回 home，继续遥操作 + 录制")
                elif state == "IDLE":
                    print("\n[Return] 规划回 home...")
                    execute_return_home(sim, planner, home_eef, viewer)
                    prev_arm_cmd = sim.get_full_qpos()[:7].copy()
                    prev_hand_cmd = sim.get_full_qpos()[7:].copy()
                elif state == "SAVE_PROMPT":
                    print("[Return] 请先按 S 保存或 N 丢弃当前 episode")

            # C: Pause / Resume — 冻结/恢复遥操作
            #     暂停：冻结 EEF（PD 保持当前位置），暂停向 episode 写入帧。
            #     恢复：自动重新标定 arm mapper（抵消暂停期间的漂移），
            #           继续录制。典型使用场景：接电话、思考策略、调整坐姿。
            if keys.is_pressed('c') or keys.is_pressed('C'):
                keys.clear('c')
                keys.clear('C')
                if state == "TELEOP_RECORDING":
                    state = "PAUSED"
                    print(
                        f"\n[Pause] 遥操作已暂停  |  EEF 冻结  |  "
                        f"录制暂停 ({recorder.frame_count} 帧)"
                        f"\n        按 C 恢复，按 R 回 home，按 Q 退出"
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
                        ik_fail_count = 0
                        retarget_fail_count = 0
                        stale_frame_count = 0
                        lost_since_ns = None
                        print(
                            f"\n[Resume] 遥操作已恢复  |  Mapper 已重新锚定  |  "
                            f"EEF={np.round(sim_state['eef_pos'], 3)}"
                        )
                    else:
                        print("[Resume] 无法获取 VR 帧，恢复失败")
                elif state == "SAVE_PROMPT":
                    print("[Pause] 请先按 S 保存或 N 丢弃当前 episode")

            # S: Save episode（仅在 SAVE_PROMPT 下）
            if keys.is_pressed('s') or keys.is_pressed('S'):
                keys.clear('s')
                keys.clear('S')
                if state == "SAVE_PROMPT":
                    if last_episode_path is not None:
                        print(f"[Save] Episode saved to {last_episode_path}")
                    else:
                        print("[Save] Episode saved (no path returned)")
                    last_episode_path = None
                    state = "IDLE"
                    print("[State] IDLE — 按 B 开始新的遥操作")
                # In TELEOP_RECORDING, S is intentionally NOT handled —
                # the user must press Q first to stop, then S to save.

            # N: No save — 丢弃 episode（仅在 SAVE_PROMPT 下）
            if keys.is_pressed('n') or keys.is_pressed('N'):
                keys.clear('n')
                keys.clear('N')
                if state == "SAVE_PROMPT":
                    if last_episode_path is not None:
                        try:
                            Path(last_episode_path).unlink(missing_ok=True)
                            print(f"[Discard] Deleted {last_episode_path}")
                        except OSError:
                            print(f"[Discard] Failed to delete {last_episode_path}")
                    last_episode_path = None
                    state = "IDLE"
                    print("[State] IDLE — 按 B 开始新的遥操作")

            # ═══════════════════════════════════════════════════════════════
            # TELEOP_RECORDING 控制 tick
            # ═══════════════════════════════════════════════════════════════

            if state == "TELEOP_RECORDING":
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

                    # Hold current position (soft deceleration simplified to instant hold)
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

                    # 更新失败计数（用于状态打印）
                    if not status["ik_ok"]:
                        ik_fail_count += 1
                    else:
                        ik_fail_count = 0
                    if not status["retarget_ok"]:
                        retarget_fail_count += 1
                    else:
                        retarget_fail_count = 0

                    # 3. 仿真相机帧捕获（可选）
                    camera_frame = None
                    T_base_eef = None
                    if ee_camera is not None:
                        camera_frame = capture_camera_frame(ee_camera)
                        # 计算 T_base_eef 用于相机外参（eye-in-hand）
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

                    # 4. 录制帧（在应用动作前录制当前状态 + 即将发送的命令）
                    if recorder.is_recording:
                        try:
                            robot_state = build_robot_state(sim_state)
                            robot_action = build_robot_action(
                                arm_cmd, hand_cmd, target_eef_pos,
                            )
                            recorder.add_frame(
                                state=robot_state,
                                action=robot_action,
                                vr_frame=frame,
                                camera_frame=camera_frame,
                                T_base_eef=T_base_eef,
                            )
                        except (ValueError, OSError) as e:
                            print(f"[Recorder] add_frame 失败: {e}")

                    # 5. 应用动作到仿真
                    full_cmd = np.concatenate([arm_cmd, hand_cmd])
                    sim.robot.balance_passive_force()
                    sim.robot.apply_action(full_cmd)
                    sim._step_physics(n=PHYSICS_STEPS_PER_TICK)

                    # 更新上一帧命令
                    prev_arm_cmd = arm_cmd.copy()
                    prev_hand_cmd = hand_cmd.copy()

            elif state == "PAUSED":
                # PAUSED: 冻结 EEF（PD 保持当前位置），不读取 VR、不录制
                #         保持与 TELEOP_RECORDING 相同的物理步数以维持渲染流畅
                full_cmd = np.concatenate([prev_arm_cmd, prev_hand_cmd])
                sim.robot.balance_passive_force()
                sim.robot.apply_action(full_cmd)
                sim._step_physics(n=PHYSICS_STEPS_PER_TICK)

            elif state == "IDLE" or state == "SAVE_PROMPT":
                # IDLE / SAVE_PROMPT: 仅推进物理（PD 保持当前位置）
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
                    rec_frames = recorder.frame_count if recorder.is_recording else 0
                    status = (
                        f"[REC] EEF={np.round(sim_state['eef_pos'], 3)}"
                        f"  frames={rec_frames}"
                    )
                    if ik_fail_count > 0:
                        status += f"  IK_fail={ik_fail_count}"
                    if retarget_fail_count > 0:
                        status += f"  retarget_fail={retarget_fail_count}"
                    if stale_frame_count > 0:
                        lost_s = (time.perf_counter_ns() - (lost_since_ns or 0)) * 1e-9
                        status += f"  stale={stale_frame_count} (lost={lost_s:.1f}s)"
                elif state == "PAUSED":
                    rec_frames = recorder.frame_count if recorder.is_recording else 0
                    status = (
                        f"[PAUSED] EEF={np.round(sim_state['eef_pos'], 3)}"
                        f"  frames={rec_frames}"
                        f"  |  C: 恢复  R: 回 home  Q: 退出"
                    )
                elif state == "SAVE_PROMPT":
                    status = "[SAVE_PROMPT] S: 保存  |  N: 丢弃  |  Q: 退出"
                else:
                    status = f"[IDLE] EEF={np.round(sim_state['eef_pos'], 3)}"
                print(status)
                last_status_time = now

            # ── 循环超限检测（参考 BunnyVisionPro overrun detection）──
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

    finally:
        # ── 清理（始终执行）──
        print("\nCleaning up...")

        # 如果还在录制中（异常退出），尝试保存
        if recorder.is_recording:
            try:
                path = recorder.stop_episode(success=False)
                if path:
                    print(f"[Recorder] 异常退出，episode 已保存至 {path}")
            except (ValueError, OSError):
                pass

        keys.stop()
        tracker.disconnect()
        sim.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
