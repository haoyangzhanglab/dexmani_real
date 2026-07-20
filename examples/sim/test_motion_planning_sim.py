#!/usr/bin/env python3
"""Pick-and-Place episode loop (仿真模型 + SAPIEN 可视化) — sim-to-real 验证。

通过随机目标物位姿的连续抓取-放置 episode 验证规划/IK/碰撞检测管线的
端到端可靠性与 sim-to-real 一致性。

用法:
    conda activate real
    python examples/sim/test_motion_planning_sim.py [--headless] [--seed SEED] [--episodes N]

本文件还保留了 ~1000 行未被 main() 调用的参考代码 (ik_test, plan_and_execute,
return_to_home_sim, sweep_z_min 等) — 这些函数提供 IK 成功率统计、单路径规划
执行、归位测试、z_min 扫描等独立测试能力, 可供手动调用或重组。

真机对应入口: examples/real/test_motion_planning_real.py (按序跑 Test1-Test5:
solve_ik, solve_teleop_ik, plan_path, 硬件执行, IK 自碰撞)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import time
from dataclasses import dataclass, field

import numpy as np
import sapien.core as sapien

from dexmani_real import ASSET_DIR
from dexmani_real.planning import (
    CollisionConfig,
    PlanningProfile,
    Pose,
    TeleopProfile,
    XArm7MotionPlanner,
    XArm7PlannerConfig,
)
from dexmani_real.planning.collision_model import CollisionModel
from dexmani_real.planning.path_utils import interpolate_waypoints
from dexmani_real.planning.pose_utils import angular_dist_rad, quat_multiply
from dexmani_real.planning.pose_utils import quat_wxyz_to_rotmat as quat_to_rotmat
from dexmani_real.planning.pose_utils import random_quat_full_so3, random_quat_multi_axis, random_quat_within_angle
from dexmani_real.simulation import SimRobotConfig, SimRobotInterface, execute_dense_path, settle_at_target
from dexmani_real.simulation.constructor import add_light, create_viewer

# ═══════════════════════════════════════════════════════════════════
# Local test utilities (lightweight — not worth a shared module)
# ═══════════════════════════════════════════════════════════════════


@dataclass
class IKStats:
    """Aggregate IK test statistics."""

    ok: int
    total: int = 0
    pos_errs_mm: list[float] = field(default_factory=list)
    rot_errs_deg: list[float] = field(default_factory=list)
    max_dq_deg: list[float] = field(default_factory=list)


def build_target_pose(
    pos: np.ndarray,
    home_quat: np.ndarray,
    rng: "np.random.RandomState | None" = None,
    *,
    rot_mode: str = "single_axis",
    rot_max_deg: float = 30.0,
    rot_axis1_deg: float = 45.0,
    rot_axis2_deg: float = 30.0,
) -> Pose:
    """Build a target EEF pose with optional random rotation."""
    quat = home_quat
    if rng is None:
        return Pose(p=pos, q=quat)
    if rot_mode == "full_so3":
        quat = random_quat_full_so3(rng)
    elif rot_mode == "multi_axis":
        delta_q = random_quat_multi_axis(rng, rot_axis1_deg, rot_axis2_deg)
        quat = quat_multiply(delta_q, home_quat)
    elif rot_mode == "single_axis" and rot_max_deg > 0:
        quat = quat_multiply(random_quat_within_angle(rng, rot_max_deg), home_quat)
    return Pose(p=pos, q=quat)


# ═══════════════════════════════════════════════ 配置

NUM_SAMPLES = 90  # 默认路径规划采样点（stratified 15 区域）
NUM_IK_SAMPLES = 200  # 默认 IK 采样点
HEADLESS = False
SEED = 123

# --comprehensive 模式（已废弃：现为默认行为，保留 flag 向后兼容）
COMPREHENSIVE_NUM_SAMPLES = 90
COMPREHENSIVE_NUM_IK = 200

# 采样空间（world frame，与真机 workspace 一致）
SAMPLE_X = (0.28, 0.70)
SAMPLE_Y = (-0.40, 0.40)
SAMPLE_Z = (0.02, 0.55)

# Z 偏置：低区 [0.02, 0.20] 权重放大（穿桌高风险区）
Z_LOW_WEIGHT = 6.0  # 低区相对权重（增强近桌面测试密度）
Z_LOW_RANGE = (0.02, 0.20)
Z_MID_RANGE = (0.20, 0.36)
Z_HIGH_RANGE = (0.36, 0.55)

# 姿态随机化配置
ROT_MODE = "multi_axis"  # "fixed" | "single_axis" | "multi_axis" | "full_so3"
ROT_MAX_DEG = 60.0  # 最大旋转角度（单轴模式）
ROT_AXIS1_DEG = 45.0  # 第一轴最大角度（多轴模式）
ROT_AXIS2_DEG = 30.0  # 第二轴最大角度（多轴模式）

# ── 桌面碰撞几何（与 SAPIEN add_base_components 的 table 对齐）──
TABLE_CENTER = (0.4, 0.0, -0.5)  # table actor 位置 (constructor.py:100)
TABLE_HALF = (0.5, 1.0, 0.5)  # half_size (constructor.py:97-98)
TABLE_TOP_Z = TABLE_CENTER[2] + TABLE_HALF[2]  # = 0.0 桌面上表面
# 桌面碰撞检测使用 Pinocchio FK 直接计算五指指尖世界坐标。
# home 手型下 pinky_tip 比 EEF 低 7.6cm（不是拇指的 11.3cm — 旧值是中间关节 ID 39 的 pinky_link2）。

# ── 统一碰撞配置（通过 CollisionConfig 管理，替代之前分散的常量）──
# 旧常量映射（仅作参考，不与 CollisionConfig 重复赋值）：
#   HAND_EXTENSION_BELOW_EEF → collision_config.hand_extension_below_eef (0.076)
#   HAND_SAFE_MARGIN         → collision_config.hand_safe_margin (0.03)
#   DESK_SAFE_Z              → collision_config.desk_safe_z (0.106)
#   REJECT_EPSILON           → FingertipDeskSafety._epsilon (0.001)
collision_config = CollisionConfig(
    table_z_world=TABLE_TOP_Z,
    hand_extension_below_eef=0.076,
    hand_safe_margin=0.03,
)
# ── 统一碰撞模型（CollisionModel FCL 替代 DESK_SAFE_Z + FingertipDeskSafety）──
# 不再使用 EEF 级固定阈值或 FK 指尖 Z 检测。
# CollisionModel 支持 19-DOF 全模型 + table box obstacle 的 mesh 级碰撞检测。
_env_cm: "CollisionModel | None" = None  # set by main()


def _get_cm() -> CollisionModel:
    assert _env_cm is not None, "CollisionModel not initialized — call main() first"
    return _env_cm


# ═══════════════════════════════════════════════ CollisionModel 桌面碰撞 wrapper
# 以下函数保持原有签名（兼容所有调用者），内部委托给 CollisionModel FCL。


def check_path_desk_safety(
    _planner,
    path: np.ndarray,
    step_rad: float = 0.02,
) -> tuple[bool, float, int]:
    """Check env collision along arm path using CollisionModel FCL (replaces FK Z)."""
    cm = _get_cm()
    path_arm = path[:, :7] if path.ndim == 2 and path.shape[1] > 7 else path
    for i in range(len(path_arm) - 1):
        if not cm.check_segment_env_collision_free(path_arm[i], path_arm[i + 1], step_rad):
            return False, 0.0, i
    return True, float("inf"), -1


def check_hand_desk_clearance(
    _planner,
    qpos: np.ndarray,
) -> tuple[bool, float, str]:
    """Check env collision at single qpos using CollisionModel FCL (replaces FK Z)."""
    cm = _get_cm()
    q = np.asarray(qpos, dtype=np.float64)
    in_collision = cm.check_env_collision(q)
    return not in_collision, 0.0, "fcl" if in_collision else "ok"


def check_hand_desk_clearance_sim(sim: SimRobotInterface) -> tuple[bool, float, str]:
    """Sim FK → CollisionModel: check current sim arm qpos against table.

    Uses planner's 7-DOF CollisionModel (collision URDF, hand fixed at home).
    Home hand (open, max extension) is the most conservative check.
    """
    cm = _get_cm()
    arm = sim.get_full_qpos()[:7]  # planner cm is 7-DOF
    in_collision = cm.check_env_collision(arm)
    return not in_collision, 0.0, "fcl" if in_collision else "ok"


def check_path_desk_safety_sim(
    _sim,
    arm_path: np.ndarray,
    step_rad: float = 0.02,
) -> tuple[bool, float, int]:
    """CollisionModel segment check using current sim hand (replaces FK Z)."""
    cm = _get_cm()
    path_arm = arm_path[:, :7] if arm_path.ndim == 2 and arm_path.shape[1] > 7 else arm_path
    for i in range(len(path_arm) - 1):
        if not cm.check_segment_env_collision_free(path_arm[i], path_arm[i + 1], step_rad):
            return False, 0.0, i
    return True, float("inf"), -1


RANDOMIZE_HAND = True  # 是否随机化手部关节（默认开启，测试不同手型下的桌面碰撞）
HAND_RANDOM_RANGE_DEG = 30.0  # 手部关节随机范围 ±30°（相对 home=0°）
NUM_HAND_JOINTS = 12  # 手部 DOF（xhand_right.urdf 中 12 个 revolute 关节）

# 手部关节索引（user order: arm7 + hand12，hand12 顺序见 register_joint_names）
# [0]thumb_bend [1]thumb_rota1 [2]thumb_rota2
# [3]index_bend [4]index_j1  [5]index_j2
# [6]mid_j1     [7]mid_j2
# [8]ring_j1    [9]ring_j2
# [10]pinky_j1  [11]pinky_j2
# 同一手指内 j1(基关节) 和 j2(尖关节) 联动弯曲；四指(index/mid/ring/pinky) 协同

# 抓取类型手部姿态生成器
# 每种抓取类型定义手指弯曲程度、拇指位置、噪声幅度
GRASP_TYPES = {
    "power": {  # 力量抓取（柱状）：四指全弯 60-90°，拇指包覆
        "finger_curl": (60, 90),  # (min_deg, max_deg) 四指基关节弯曲范围
        "finger_tip_scale": 0.8,  # 尖关节 = 基关节 × scale
        "thumb_bend": (60, 90),  # 拇指弯曲
        "thumb_spread": (10, 30),  # 拇指外展（rota1）
        "noise": 10,  # 关节噪声 ±deg
    },
    "pinch": {  # 精确捏取（指尖）：拇指+食指微弯对捏，其余三指卷起
        "finger_curl": (20, 50),
        "finger_tip_scale": 0.5,
        "thumb_bend": (30, 60),
        "thumb_spread": (20, 45),
        "noise": 5,
    },
    "tripod": {  # 三指捏取（笔握）：拇+食+中，无名+小指卷起
        "finger_curl": (30, 60),
        "finger_tip_scale": 0.6,
        "thumb_bend": (30, 60),
        "thumb_spread": (15, 40),
        "noise": 8,
    },
    "open": {  # 张开手掌：五指展开，微弯
        "finger_curl": (0, 20),
        "finger_tip_scale": 0.3,
        "thumb_bend": (0, 20),
        "thumb_spread": (30, 60),
        "noise": 15,
    },
    "hook": {  # 钩状抓取：基关节中立，尖关节强弯
        "finger_curl": (0, 15),
        "finger_tip_scale": 1.5,  # 尖关节比基关节弯更多
        "thumb_bend": (0, 15),
        "thumb_spread": (10, 30),
        "noise": 8,
    },
    "spherical": {  # 球形抓取：五指均匀弯曲，拇指对握
        "finger_curl": (30, 70),
        "finger_tip_scale": 0.7,
        "thumb_bend": (40, 70),
        "thumb_spread": (25, 50),
        "noise": 12,
    },
}
GRASP_TYPE_NAMES = list(GRASP_TYPES.keys())

# --test-desk 模式配置（聚焦桌面碰撞测试）
TEST_DESK_NUM_SAMPLES = 30
TEST_DESK_Z_RANGE = (0.02, 0.18)  # 桌面附近 Z 范围
TEST_DESK_X_RANGE = (0.28, 0.70)
TEST_DESK_Y_RANGE = (-0.35, 0.35)

# 分层采样区域（Z低区重点覆盖：超低/很低/低 三档共 9 个子区域 vs 原来的 6 个）
STRATIFIED_REGIONS = [
    # (label, x_range, y_range, z_range, weight)
    # ── Z 超低区 [0.02, 0.06]：紧贴桌面，极高风险（权重 ×6）──
    ("ultra_low_a", (0.28, 0.40), (-0.30, 0.30), (0.02, 0.06), 6.0),
    ("ultra_low_b", (0.40, 0.55), (-0.40, 0.40), (0.02, 0.06), 6.0),
    ("ultra_low_c", (0.55, 0.70), (-0.40, 0.40), (0.02, 0.06), 6.0),
    # ── Z 很低区 [0.06, 0.12]：高风险（权重 ×5）──
    ("very_low_a", (0.28, 0.40), (-0.30, 0.30), (0.06, 0.12), 5.0),
    ("very_low_b", (0.40, 0.55), (-0.40, 0.40), (0.06, 0.12), 5.0),
    ("very_low_c", (0.55, 0.70), (-0.40, 0.40), (0.06, 0.12), 5.0),
    # ── Z 低区 [0.12, 0.20]：中等风险（权重 ×4）──
    ("low_a", (0.28, 0.40), (-0.30, 0.30), (0.12, 0.20), 4.0),
    ("low_b", (0.40, 0.55), (-0.40, 0.40), (0.12, 0.20), 4.0),
    ("low_c", (0.55, 0.70), (-0.40, 0.40), (0.12, 0.20), 4.0),
    # ── Z 中区 [0.20, 0.36]：正常操作区 ──
    ("near_mid", (0.28, 0.40), (-0.30, 0.30), (0.20, 0.36), 1.0),
    ("mid_mid", (0.40, 0.55), (-0.40, 0.40), (0.20, 0.36), 1.0),
    ("far_mid", (0.55, 0.70), (-0.40, 0.40), (0.20, 0.36), 1.0),
    # ── Z 高区 [0.36, 0.55]：安全区 ──
    ("near_high", (0.28, 0.40), (-0.30, 0.30), (0.36, 0.55), 0.5),
    ("mid_high", (0.40, 0.55), (-0.40, 0.40), (0.36, 0.55), 0.5),
    ("far_high", (0.55, 0.70), (-0.40, 0.40), (0.36, 0.55), 0.5),
]

PHYSICS_STEPS_PER_WP = 20
INTERP_MAX_STEP_RAD = np.deg2rad(2.0)
MARKER_RADIUS = 0.015
RANDOM_ROT_DEG = 30.0

# ── return_to_home 参数（与 RobotInterface 保持一致）──
_HOME_JOINT_THRESHOLD_RAD = np.deg2rad(1.0)  # 视为已归位的关节偏差
_PHASE1_CONVERGE_THRESHOLD_RAD = np.deg2rad(3.0)  # Phase 1 收敛阈值
_PHASE2_MIN_DELTA_RAD = np.deg2rad(0.5)  # Phase 2 跳过的关节偏差下限
_PHASE2_MAX_STEP_RAD = np.deg2rad(1.0)  # Phase 2 关节空间最大步长
_DIRECT_LIFT_Z_M = 0.15  # 安全抬升高度
_DIRECT_LIFT_SLEEP_STEPS = 6  # 抬升后稳定步数 (0.3s ÷ 0.05s per step)
_RESIDUAL_ERROR_MAX_DEG = 10.0  # 残余误差上限（超过此值返回失败）

# ═══════════════════════════════════════════════ 数学工具


# ═══════════════════════════════════════════════ 仿真执行


def smooth_drive_to_target(
    sim: SimRobotInterface,
    target_full_qpos: np.ndarray,
    viewer: sapien.Viewer | None = None,
    max_iter: int = 60,
    converge_threshold_rad: float = np.deg2rad(0.1),
    label: str = "",
) -> float:
    """通过 PD 控制器平滑驱动 arm+hand 到目标位姿（非传送）。

    与 set_qpos 的瞬时传送不同，此函数通过 apply_action 设置驱动目标，
    然后逐步推进物理，PD 控制器会平滑地将关节驱动到目标位置。
    viewer 中可以看到关节的平滑运动。

    Returns: final max joint error (rad)
    """
    for step in range(max_iter):
        sim.robot.balance_passive_force()
        sim.robot.apply_action(target_full_qpos)
        sim._step_physics(n=PHYSICS_STEPS_PER_WP)
        if viewer is not None:
            sim.scene.update_render()
            viewer.render()
        current = sim.robot.get_qpos()
        err = float(np.max(np.abs(current[: len(target_full_qpos)] - target_full_qpos)))
        if err < converge_threshold_rad:
            return err
    current = sim.robot.get_qpos()
    return float(np.max(np.abs(current[: len(target_full_qpos)] - target_full_qpos)))


def animated_reset_to_home(
    sim: SimRobotInterface,
    home_qpos: np.ndarray,
    viewer: sapien.Viewer | None = None,
    max_step_rad: float = np.deg2rad(1.0),
    planner: XArm7MotionPlanner | None = None,
) -> float:
    """Smooth joint-space animation to home, with collision checking.

    Replaces teleport sim.reset().  Interpolates current → home linearly.
    If planner is provided, checks the joint path for env/self collisions
    before execution.  If the direct joint path collides (e.g. hand near
    desk → joint-space shortcut through desk), falls back to a two-step
    safe path: lift Z first, then go to home.

    Returns: final max joint error (rad)
    """
    current_qpos = sim.get_full_qpos()[:7].copy()
    hand_qpos = sim.get_full_qpos()[7:].copy()

    dist = float(np.max(np.abs(current_qpos - home_qpos)))
    if dist < np.deg2rad(0.1):
        return settle_at_target(sim, home_qpos, hand_qpos)

    n = max(2, int(np.ceil(dist / max_step_rad)) + 1)
    joint_path = np.array([current_qpos + (k / (n - 1)) * (home_qpos - current_qpos) for k in range(n)])

    # ── Desk safety check: if direct joint path dips below desk, use 2-step safe path ──
    effective_path = joint_path
    if planner is not None:
        desk_safe, min_z, _ = check_path_desk_safety(planner, joint_path)
        if not desk_safe:
            # Direct joint path would dip below desk — try two-step: lift → home
            home_ik = _safe_two_step_home(sim, planner, home_qpos, viewer, max_step_rad)
            if home_ik is not None:
                err = settle_at_target(sim, home_qpos, hand_qpos)
                return err
            # Two-step failed too — fall through with original path (last resort)

    for wp in effective_path:
        if viewer is not None and viewer.closed:
            break
        sim.robot.balance_passive_force()
        sim.robot.apply_action(np.concatenate([wp, hand_qpos]))
        sim._step_physics(n=PHYSICS_STEPS_PER_WP)
        if viewer is not None:
            sim.scene.update_render()
            viewer.render()

    return settle_at_target(sim, home_qpos, hand_qpos)


def _safe_two_step_home(
    sim: SimRobotInterface,
    planner: XArm7MotionPlanner,
    home_qpos: np.ndarray,
    viewer: sapien.Viewer | None,
    max_step_rad: float,
) -> np.ndarray | None:
    """Two-step safe path to home: lift EEF Z → then joint-space to home.

    Step A: teleop IK to same XY but higher Z (above desk)
    Step B: joint-space path to home_qpos (now safe because EEF is high)

    Returns the executed lift qpos if successful, None otherwise.
    """
    current_qpos = sim.get_full_qpos()[:7].copy()
    hand_qpos = sim.get_full_qpos()[7:].copy()
    current_pose = planner.compute_eef_pose_world(current_qpos)

    # Step A: Lift Z to safe level above table
    safe_z = max(TABLE_TOP_Z + 0.15, current_pose.p[2] + _DIRECT_LIFT_Z_M)
    safe_z = min(safe_z, 0.55)  # clamp to workspace ceiling
    lift_pose = Pose(
        p=np.array([current_pose.p[0], current_pose.p[1], safe_z], dtype=np.float64),
        q=current_pose.q.copy(),
    )
    lift_result = planner.solve_teleop_ik(lift_pose, current_qpos, current_qpos)
    if not lift_result.success or lift_result.qpos is None:
        return None

    lift_qpos = lift_result.qpos

    # Execute lift
    lift_path = np.array([current_qpos, lift_qpos])
    lift_dense = _dense_interpolate_sim(lift_path, max_step_rad)
    for wp in lift_dense:
        if viewer is not None and viewer.closed:
            return None
        sim.robot.balance_passive_force()
        sim.robot.apply_action(np.concatenate([wp, hand_qpos]))
        sim._step_physics(n=PHYSICS_STEPS_PER_WP)
        if viewer is not None:
            sim.scene.update_render()
            viewer.render()

    # Step B: Joint-space to home (now safe — EEF is high above desk)
    current = sim.get_full_qpos()[:7]
    dist = float(np.max(np.abs(current - home_qpos)))
    n = max(2, int(np.ceil(dist / max_step_rad)) + 1)
    home_path = np.array([current + (k / (n - 1)) * (home_qpos - current) for k in range(n)])
    for wp in home_path:
        if viewer is not None and viewer.closed:
            return None
        sim.robot.balance_passive_force()
        sim.robot.apply_action(np.concatenate([wp, hand_qpos]))
        sim._step_physics(n=PHYSICS_STEPS_PER_WP)
        if viewer is not None:
            sim.scene.update_render()
            viewer.render()

    return lift_qpos


def append_joint_goal(
    planner: XArm7MotionPlanner,
    path: np.ndarray,
    goal: np.ndarray,
) -> np.ndarray:
    """在路径末尾追加 goal_qpos，插值并用 segment-based 碰撞检测。

    使用 check_path_collisions() (0.02 rad 步长 segment 插值)，而非逐点
    has_self_collision，与 interface._safe_joint_path 检测精度一致。
    不安全则返回原 path。
    """
    full = np.vstack([path, goal])
    if planner.planning_profile.check_self_collision:
        seg_result = planner.check_path_collisions(full)
        if seg_result.get("path_self_collision"):
            return path
    return full


# ═══════════════════════════════════════════════ 指尖验证


@dataclass
class FingertipCheck:
    """指尖位置验证结果。"""

    sapien_fk_delta_mm: float  # SAPIEN vs Pinocchio FK 最大偏差
    tip_eef_local: np.ndarray  # (5,3) EEF 局部坐标系下的指尖偏移


def check_fingertips(sim: SimRobotInterface) -> FingertipCheck:
    """验证指尖位置：SAPIEN 物理 vs Pinocchio FK 一致性 + EEF 局部偏移。"""
    robot = sim.robot
    full_qpos = sim.get_full_qpos()
    names = robot.fingertip_link_names

    tips_sapien = robot.get_link_poses(names)  # (5,7) [x,y,z,w,x,y,z]
    tips_fk = robot.forward_kinematics(full_qpos, target_link_names=names)

    max_delta = float(max(np.linalg.norm(tips_sapien[i, :3] - tips_fk[i, :3]) for i in range(len(names))))

    eef_pose = robot.get_eef_pose()
    eef_p, eef_q = np.array(eef_pose.p), np.array(eef_pose.q)
    eef_R_inv = quat_to_rotmat(eef_q).T
    tips_local = np.array([eef_R_inv @ (tips_sapien[i, :3] - eef_p) for i in range(len(names))])

    return FingertipCheck(sapien_fk_delta_mm=max_delta * 1000, tip_eef_local=tips_local)


# ═══════════════════════════════════════════════ return_to_home (镜像 RobotInterface)


def _dense_interpolate_sim(path: np.ndarray, max_step_rad: float = np.deg2rad(1.0)) -> np.ndarray:
    """Densify sparse joint path (mirrors interface._dense_interpolate)."""
    if len(path) <= 1:
        return path
    dense = [path[0]]
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        n = int(np.ceil(float(np.max(np.abs(b - a))) / max_step_rad))
        for k in range(1, n + 1):
            dense.append(a + (k / n) * (b - a))
    return np.array(dense, dtype=np.float64)


def _safe_joint_path_sim(
    planner: XArm7MotionPlanner,
    start: np.ndarray,
    goal: np.ndarray,
    max_step_rad: float = _PHASE2_MAX_STEP_RAD,
) -> np.ndarray | None:
    """Linear interpolation start→goal with segment-based collision check.

    Mirrors RobotInterface._safe_joint_path().
    Dense interpolation at max_step_rad resolution for smooth execution,
    then validates with planner.check_path_collisions() (0.02 rad internal step).
    Returns None if unsafe or planner unavailable (cannot verify safety).
    """
    dist = float(np.max(np.abs(goal - start)))
    n = max(2, int(np.ceil(dist / max_step_rad)) + 1)
    path = np.array([start + (k / (n - 1)) * (goal - start) for k in range(n)])

    profile = planner.planning_profile
    if profile.check_self_collision:
        result = planner.check_path_collisions(path)
        if result.get("path_self_collision"):
            return None
    # Desk safety: geometric Z check (no MPlib point cloud needed)
    desk_safe, min_z, _ = check_path_desk_safety(planner, path)
    if not desk_safe:
        return None
    return path


def _lift_eef_z_safe_sim(
    planner: XArm7MotionPlanner,
    sim: SimRobotInterface,
    current_qpos: np.ndarray,
    ws_bounds: np.ndarray,
) -> bool:
    """Safety lift: move EEF upward to clear desk.

    Mirrors RobotInterface._lift_eef_z_safe().
    Targets workspace Z midpoint or at least 0.15 m above current Z,
    clamped to workspace Z max minus 2 cm margin.
    """
    if not np.all(np.isfinite(current_qpos)):
        return False
    current_pose = planner.compute_eef_pose_world(current_qpos)
    ws_z_mid = float(np.mean(ws_bounds[2]))
    ws_z_max = float(ws_bounds[2, 1])
    target_z = max(float(current_pose.p[2]) + _DIRECT_LIFT_Z_M, ws_z_mid)
    target_z = min(target_z, ws_z_max - 0.02)

    lift_pose = Pose(
        p=np.array([current_pose.p[0], current_pose.p[1], target_z], dtype=np.float64),
        q=current_pose.q.copy(),
    )

    lift_result = planner.solve_teleop_ik(lift_pose, current_qpos, current_qpos)
    if not lift_result.success or lift_result.qpos is None:
        print(f"  [_lift_eef_z_safe] Safety lift IK failed: {lift_result.reason}")
        return False
    # Verify lift target with hand FK (not just EEF Z)
    lift_safe, lift_z, lift_name = check_hand_desk_clearance(planner, lift_result.qpos)
    if not lift_safe:
        print(f"  [_lift_eef_z_safe] Lift result {lift_name} z={lift_z:.3f}m < desk, skipping")
        return False

    # Execute lift in sim
    hand = sim.get_full_qpos()[7:]
    sim.robot.balance_passive_force()
    sim.robot.apply_action(np.concatenate([lift_result.qpos, hand]))
    sim._step_physics(n=_DIRECT_LIFT_SLEEP_STEPS)
    return True


def return_to_home_sim(
    planner: XArm7MotionPlanner,
    sim: SimRobotInterface,
    home_qpos: np.ndarray,
    ws_bounds: np.ndarray,
    viewer: sapien.Viewer | None = None,
) -> dict:
    """Sim return_to_home — mirrors RobotInterface.return_to_home() algorithm.

    Two-phase:
      Phase 1: plan_path(home_eef) → dense execution with collision check
      Phase 2: Joint-space interpolation current→home with segment collision check
      Phase 1 failure fallback: _lift_eef_z_safe → direct reset

    Returns dict with:
        success: bool
        phase1_completed: bool
        phase2_executed: bool
        lift_used: bool
        final_err_deg: float
        final_pos_err_mm: float
    """
    home_eef = planner.compute_eef_pose_world(home_qpos)
    hand_qpos = sim.get_full_qpos()[7:]
    current_qpos = sim.get_full_qpos()[:7].copy()

    if not np.all(np.isfinite(current_qpos)):
        print("  [return_to_home_sim] Invalid qpos, falling back to animated reset")
        animated_reset_to_home(sim, home_qpos, viewer)
        smooth_drive_to_target(
            sim, np.concatenate([home_qpos, np.zeros(NUM_HAND_JOINTS)]), viewer, max_iter=60, label="hand_home_fallback"
        )
        final_qpos = sim.get_full_qpos()[:7]
        err = float(np.rad2deg(np.max(np.abs(final_qpos - home_qpos))))
        return {
            "success": err < _RESIDUAL_ERROR_MAX_DEG,
            "phase1_completed": False,
            "phase2_executed": False,
            "lift_used": False,
            "final_err_deg": err,
            "final_pos_err_mm": 0.0,
        }

    # Already at home?
    if float(np.max(np.abs(current_qpos - home_qpos))) < _HOME_JOINT_THRESHOLD_RAD:
        # Arm already home — still restore hand to zero
        smooth_drive_to_target(
            sim, np.concatenate([home_qpos, np.zeros(NUM_HAND_JOINTS)]), viewer, max_iter=60, label="hand_home_already"
        )
        print("  [return_to_home_sim] Already at home (hand restored)")
        return {
            "success": True,
            "phase1_completed": False,
            "phase2_executed": False,
            "lift_used": False,
            "final_err_deg": 0.0,
            "final_pos_err_mm": 0.0,
        }

    # ── Pre-check: fingertip Z desk clearance ──
    # Check actual fingertip positions (FK), not just EEF position.
    # Hand extends 7.6cm below EEF (pinky tip) — EEF Z check alone misses hand-desk collision.
    start_env_ok = True
    start_safe, start_z, start_name = check_hand_desk_clearance(planner, current_qpos)
    if not start_safe:
        start_env_ok = False
        print(f"  [return_to_home_sim] {start_name} z={start_z:.3f}m < desk+margin, " f"forcing safety lift")
    else:
        # Check joint path to home for fingertip Z violations
        joint_delta = float(np.max(np.abs(current_qpos - home_qpos)))
        if joint_delta > np.deg2rad(5.0):
            test_n = min(10, int(np.ceil(joint_delta / np.deg2rad(5.0))) + 1)
            for k in range(1, test_n):
                alpha = k / test_n
                test_q = current_qpos + alpha * (home_qpos - current_qpos)
                safe, z, name = check_hand_desk_clearance(planner, test_q)
                if not safe:
                    start_env_ok = False
                    print(
                        f"  [return_to_home_sim] Joint path to home: {name} dips below desk "
                        f"(α={alpha:.2f}, z={z:.3f}m), forcing safety lift"
                    )
                    break

    # ── Phase 1: EEF Cartesian path ──
    phase1_completed = False
    if start_env_ok:
        try:
            result = planner.plan_path(home_eef, current_qpos)
        except RuntimeError as e:
            print(f"  [return_to_home_sim] plan_path error: {e}")
            result = None
    else:
        result = None

    if result is not None and result.success and result.qpos_path is not None and len(result.qpos_path) > 0:
        # Segment-based collision verification: self (MPlib FCL) + desk (geometric Z)
        self_check = planner.check_path_collisions(result.qpos_path)
        desk_safe, min_z, viol_idx = check_path_desk_safety(planner, result.qpos_path)
        if self_check.get("path_self_collision"):
            print(f"  [return_to_home_sim] Phase 1 path has self-collision, falling back")
        elif not desk_safe:
            print(
                f"  [return_to_home_sim] Phase 1 path dips below desk "
                f"(fingertip_z_min={min_z:.3f}m, waypoint {viol_idx}), falling back"
            )
        else:
            # Dense interpolation (1° step, mirrors interface._dense_interpolate)
            dense_path = _dense_interpolate_sim(result.qpos_path)

            # Execute waypoints (with viewer rendering for visual continuity)
            exec_ok = True
            for wp in dense_path:
                if viewer is not None and viewer.closed:
                    exec_ok = False
                    break
                sim.robot.balance_passive_force()
                sim.robot.apply_action(np.concatenate([wp, hand_qpos]))
                sim._step_physics(n=PHYSICS_STEPS_PER_WP)
                if viewer is not None:
                    sim.scene.update_render()
                    viewer.render()

            # Check convergence to path endpoint (closed-loop PD settling)
            if exec_ok:
                err = settle_at_target(
                    sim, result.qpos_path[-1], hand_qpos, converge_threshold_rad=_PHASE1_CONVERGE_THRESHOLD_RAD
                )
                phase1_completed = err < _PHASE1_CONVERGE_THRESHOLD_RAD
                print(
                    f"  [return_to_home_sim] Phase 1: {'converged' if phase1_completed else 'timeout'} "
                    f"(err={np.rad2deg(err):.2f}deg, src={result.source})"
                )
            else:
                print(f"  [return_to_home_sim] Phase 1: execution interrupted")
    elif not start_env_ok:
        print(f"  [return_to_home_sim] Phase 1 SKIPPED: desk collision risk detected in pre-check")
    else:
        reason = result.reason if (result is not None and hasattr(result, "reason")) else "planner error"
        print(f"  [return_to_home_sim] Phase 1 plan FAILED: {reason}")

    # ── Phase 2: Joint-space homing ──
    phase2_executed = False
    if phase1_completed:
        current_qpos = sim.get_full_qpos()[:7]
        joint_delta = float(np.max(np.abs(current_qpos - home_qpos)))
        if joint_delta > _PHASE2_MIN_DELTA_RAD:
            joint_path = _safe_joint_path_sim(planner, current_qpos, home_qpos)
            if joint_path is not None:
                for wp in joint_path:
                    if viewer is not None and viewer.closed:
                        break
                    sim.robot.balance_passive_force()
                    sim.robot.apply_action(np.concatenate([wp, hand_qpos]))
                    sim._step_physics(n=PHYSICS_STEPS_PER_WP)
                    if viewer is not None:
                        sim.scene.update_render()
                        viewer.render()
                phase2_executed = True
                print(f"  [return_to_home_sim] Phase 2: executed ({len(joint_path)} waypoints)")
            else:
                print(f"  [return_to_home_sim] Phase 2: joint path unsafe/unverifiable, skipped")
        else:
            print(f"  [return_to_home_sim] Phase 2: already within {np.rad2deg(_PHASE2_MIN_DELTA_RAD):.1f}deg, skipped")
    else:
        # Phase 1 failed — safety lift EEF to clear desk (mirrors _lift_eef_z_safe)
        post_phase1_qpos = sim.get_full_qpos()[:7]
        lift_used = _lift_eef_z_safe_sim(planner, sim, post_phase1_qpos, ws_bounds)
        if lift_used:
            print(f"  [return_to_home_sim] Phase 1 failure → safety lift executed")
        else:
            print(f"  [return_to_home_sim] Phase 1 failure → safety lift skipped/FAILED")

    # ── Final: collision-aware animated reset to home ──
    # Pass planner so animated_reset_to_home can detect desk collisions and
    # fall back to 2-step safe path (lift Z → joint space to home)
    settle_err_rad = animated_reset_to_home(sim, home_qpos, viewer, planner=planner)

    # ── Restore hand to home (zero) position ──
    # Use smooth_drive_to_target (checks full 19-DOF convergence) instead of
    # settle_at_target (arm-only check) so the hand actually reaches zero.
    smooth_drive_to_target(
        sim, np.concatenate([home_qpos, np.zeros(NUM_HAND_JOINTS)]), viewer, max_iter=60, label="hand_home"
    )

    final_qpos = sim.get_full_qpos()[:7]
    final_hand = sim.get_full_qpos()[7:]
    hand_err_deg = float(np.rad2deg(np.max(np.abs(final_hand))))
    err_rad = float(np.max(np.abs(final_qpos - home_qpos)))
    err_deg = float(np.rad2deg(err_rad))
    final_eef = planner.compute_eef_pose_world(final_qpos)
    pos_err_mm = float(np.linalg.norm(final_eef.p - home_eef.p)) * 1000

    if err_rad > np.deg2rad(_RESIDUAL_ERROR_MAX_DEG):
        print(f"  [return_to_home_sim] INCOMPLETE: residual error {err_deg:.1f}° > {_RESIDUAL_ERROR_MAX_DEG}°")
        return {
            "success": False,
            "phase1_completed": phase1_completed,
            "phase2_executed": phase2_executed,
            "lift_used": not phase1_completed,
            "final_err_deg": err_deg,
            "final_pos_err_mm": pos_err_mm,
        }

    print(
        f"  [return_to_home_sim] CONVERGED: err={err_deg:.2f}deg  pos={pos_err_mm:.1f}mm  "
        f"hand={hand_err_deg:.1f}deg"
    )
    return {
        "success": True,
        "phase1_completed": phase1_completed,
        "phase2_executed": phase2_executed,
        "lift_used": not phase1_completed,
        "final_err_deg": err_deg,
        "final_pos_err_mm": pos_err_mm,
    }


def hand_randomize(rng: np.random.RandomState | None = None) -> np.ndarray:
    """生成随机手部关节角（12 DOF，对称分布于 home=0° 附近）。

    Args:
        rng: 随机数生成器，None 时使用全局 numpy.random

    Returns: (12,) array of hand joint angles in radians
    """
    if rng is None:
        rng = np.random
    max_rad = np.deg2rad(HAND_RANDOM_RANGE_DEG)
    return rng.uniform(-max_rad, max_rad, NUM_HAND_JOINTS)


def hand_grasp_pose(
    rng: np.random.RandomState | None = None,
    grasp_type: str | None = None,
) -> tuple[np.ndarray, str]:
    """生成类抓取姿态的手部关节角，模拟真实抓取动作。

    与 hand_randomize() 的独立随机不同，此函数根据抓取类型生成具有
    手指协同联动（同一手指内 j1/j2 联动 + 四指协同弯曲）的合理手型。

    抓取类型：
      "power"     — 力量抓取，四指强弯，拇指包覆（桌面风险：中）
      "pinch"     — 精确捏取，拇指+食指对捏（桌面风险：低，手指不延伸）
      "tripod"    — 三指捏取，拇+食+中（桌面风险：中低）
      "open"      — 张开手掌，五指伸展（桌面风险：高，手指延伸最多）
      "hook"      — 钩状抓取，尖关节强弯（桌面风险：中，指背可能触桌）
      "spherical" — 球形抓取，五指均匀弯曲（桌面风险：中）

    Args:
        rng: 随机数生成器
        grasp_type: 抓取类型名，None 时随机选择

    Returns: (hand_qpos_12d, grasp_type_name)
    """
    if rng is None:
        rng = np.random

    if grasp_type is None:
        grasp_type = rng.choice(GRASP_TYPE_NAMES)

    cfg = GRASP_TYPES[grasp_type]
    noise = cfg["noise"]
    tip_scale = cfg["finger_tip_scale"]

    # Joint indices in 12-DOF user order:
    # 0:thumb_bend  1:thumb_rota1  2:thumb_rota2
    # 3:index_bend  4:index_j1     5:index_j2
    # 6:mid_j1      7:mid_j2
    # 8:ring_j1     9:ring_j2
    # 10:pinky_j1   11:pinky_j2

    q = np.zeros(NUM_HAND_JOINTS)

    # ── 拇指 ──
    tb_min, tb_max = cfg["thumb_bend"]
    q[0] = np.deg2rad(rng.uniform(tb_min, tb_max))  # thumb_bend
    ts_min, ts_max = cfg["thumb_spread"]
    # thumb_rota1: spread/opposition; rota2: fine rotation
    if grasp_type == "open":
        q[1] = np.deg2rad(rng.uniform(ts_min, ts_max))  # spread out
        q[2] = np.deg2rad(rng.uniform(-15, 15))  # neutral
    elif grasp_type == "pinch":
        q[1] = np.deg2rad(rng.uniform(ts_min, ts_max))  # oppose to index
        q[2] = np.deg2rad(rng.uniform(-10, 10))
    else:
        q[1] = np.deg2rad(rng.uniform(ts_min, ts_max))
        q[2] = np.deg2rad(rng.uniform(-20, 20))
    # Add noise to thumb
    q[0] += np.deg2rad(rng.uniform(-noise, noise))
    q[1] += np.deg2rad(rng.uniform(-noise * 0.5, noise * 0.5))

    # ── 四指 (index/mid/ring/pinky) 协同弯曲 ──
    # 基关节弯曲度：共享均值 + 各指独立偏差
    fc_min, fc_max = cfg["finger_curl"]
    base_curl_deg = rng.uniform(fc_min, fc_max)

    # 四指基关节索引：(index_bend=3, mid_j1=6, ring_j1=8, pinky_j1=10)
    # 注意：index 有 bend_joint(3) + j1(4) + j2(5) 三个关节
    #       mid/ring/pinky 各有 j1 + j2 两个关节
    base_indices = [3, 4, 6, 8, 10]  # 基关节
    tip_indices = [5, 7, 9, 11]  # 对应尖关节 (index_j2, mid_j2, ring_j2, pinky_j2)

    # 四指协同：pinky < ring < mid < index 弯曲递减（尺侧更弯）
    finger_bias = np.array([-5, 0, 5, 10, 15])  # index_bend, index_j1, mid, ring, pinky 的偏置

    for idx_in, base_idx in enumerate(base_indices):
        curl = base_curl_deg + finger_bias[idx_in] + rng.uniform(-noise, noise)
        q[base_idx] = np.deg2rad(np.clip(curl, -10, 100))

    # 尖关节 = 基关节 × tip_scale（联动弯曲）
    for tip_idx, base_idx in zip(tip_indices, [4, 6, 8, 10]):  # index_j1→j2, mid_j1→j2, ring_j1→j2, pinky_j1→j2
        base_val_deg = np.rad2deg(q[base_idx])
        tip_val = base_val_deg * tip_scale + rng.uniform(-noise, noise)
        q[tip_idx] = np.deg2rad(np.clip(tip_val, -10, 100))

    # index_j2 对应 index_j1 (index[4])
    # But wait: index has bend_joint(3) + joint1(4) + joint2(5)
    # The tip_indices = [5, 7, 9, 11] map to base_indices [4, 6, 8, 10]
    # So index_j2(5) ← index_j1(4), mid_j2(7) ← mid_j1(6), etc.
    # However, the finger_bias and base_curl were applied to ALL base_indices including index_bend(3).
    # We need index_j2 to follow index_j1(4), not index_bend(3).
    # Let me re-check: base_indices = [3, 4, 6, 8, 10]
    #                    tip_indices  = [5, 7, 9, 11]
    # The mapping tip→base is: 5→4(index_j2←index_j1), 7→6, 9→8, 11→10. Correct!

    # Clamp all to valid range
    q = np.clip(q, np.deg2rad(-30), np.deg2rad(100))

    return q, grasp_type


def plan_safe_descent(
    planner: XArm7MotionPlanner,
    sim: SimRobotInterface,
    target_eef: Pose,
    viewer: sapien.Viewer | None = None,
) -> tuple[np.ndarray | None, str]:
    """对低 Z 目标使用两阶段安全下降：先平移到目标上方安全高度，再垂直下降。

    模拟真机遥操作的"XY 对齐 → Z 下降"流程，避免 screw 直线路径穿过桌面。

    Returns: (joint_path, description) — joint_path 包含 arm waypoints 或 None
    """
    current_arm = sim.get_full_qpos()[:7].copy()
    current_pose = planner.compute_eef_pose_world(current_arm)
    hand_qpos = sim.get_full_qpos()[7:].copy()

    # Stage 1: plan to waypoint above target (same XY, safe Z)
    safe_z = max(target_eef.p[2] + 0.10, TABLE_TOP_Z + collision_config.hand_safe_margin + 0.05)
    safe_z = min(safe_z, 0.55)
    above_pose = Pose(
        p=np.array([target_eef.p[0], target_eef.p[1], safe_z], dtype=np.float64),
        q=target_eef.q.copy(),
    )

    # Try plan_path to the above waypoint
    try:
        stage1_result = planner.plan_path(above_pose, current_arm)
    except RuntimeError:
        return None, "stage1 plan_path error"

    if stage1_result is None or not stage1_result.success or stage1_result.qpos_path is None:
        return None, f"stage1 plan failed: {stage1_result.reason if stage1_result else 'None'}"

    # Verify stage1 path is desk-safe
    desk_safe, min_z, _ = check_path_desk_safety(planner, stage1_result.qpos_path)
    if not desk_safe:
        return None, f"stage1 path dips below desk (z_min={min_z:.3f}m)"

    # Stage 2: plan from above to target (short vertical descent)
    stage1_end = stage1_result.qpos_path[-1]
    try:
        stage2_result = planner.plan_path(target_eef, stage1_end)
    except RuntimeError:
        return None, "stage2 plan_path error"

    if stage2_result is None or not stage2_result.success or stage2_result.qpos_path is None:
        return None, f"stage2 plan failed: {stage2_result.reason if stage2_result else 'None'}"

    # Verify stage2 path is also desk-safe (should be a short vertical motion)
    desk_safe2, min_z2, _ = check_path_desk_safety(planner, stage2_result.qpos_path)
    if not desk_safe2:
        return None, f"stage2 path dips below desk (z_min={min_z2:.3f}m)"

    # Combine paths: stage1 (excluding last wp which overlaps with stage2 start)
    full_path = np.vstack([stage1_result.qpos_path[:-1], stage2_result.qpos_path])

    # ── Execute ──
    for wp in full_path:
        if viewer is not None and viewer.closed:
            return full_path, "viewer closed"
        sim.robot.balance_passive_force()
        sim.robot.apply_action(np.concatenate([wp, hand_qpos]))
        sim._step_physics(n=PHYSICS_STEPS_PER_WP)
        if viewer is not None:
            sim.scene.update_render()
            viewer.render()

    settle_at_target(sim, full_path[-1], hand_qpos)

    desc = (
        f"safe_descent: above_z={safe_z:.3f}m → target_z={target_eef.p[2]:.3f}m, "
        f"wp={len(full_path)} ({len(stage1_result.qpos_path)}+{len(stage2_result.qpos_path)})"
    )
    return full_path, desc


# ═══════════════════════════════════════════════ 路径规划测试


def plan_and_execute(
    planner: XArm7MotionPlanner,
    sim: SimRobotInterface,
    target_eef: Pose,
    viewer: sapien.Viewer | None = None,
    label: str = "",
    joint_goal: np.ndarray | None = None,
    check_desk_with_sim: bool = False,
) -> bool:
    """plan_path(target_eef) → 执行 → 验证 EEF 精度 + 指尖正确性。

    Args:
        check_desk_with_sim: 若 True，执行后用 sim FK 做精确指尖碰撞检测
                            （用于手部随机化场景，planner FK 预检仍保留作快速筛选）
    """
    current_qpos = sim.get_full_qpos()[:7]
    current_eef = planner.compute_eef_pose_world(current_qpos)
    dist = float(np.linalg.norm(target_eef.p - current_eef.p))

    joint_info = ""
    if joint_goal is not None:
        jd = float(np.max(np.abs(np.rad2deg(current_qpos - joint_goal))))
        joint_info = f"  joint_delta={jd:.1f}deg"

    print(f"  [{label}] {np.round(current_eef.p, 3)} → {np.round(target_eef.p, 3)}  " f"dist={dist:.3f}m{joint_info}")

    t0 = time.perf_counter()
    result = planner.plan_path(target_eef, current_qpos)
    if not result.success or result.qpos_path is None:
        print(f"  [{label}] PLAN FAILED: {result.reason}")
        return False

    r = result.report
    path = result.qpos_path
    print(
        f"  [{label}] plan: src={result.source}  wp={r.get('num_waypoints','?')}  "
        f"len={r.get('joint_path_length',0):.2f}rad  t={time.perf_counter()-t0:.3f}s"
    )

    # ── 桌面安全预检查（planner FK，固定手型保守估计）──
    # 如果手部已随机化 (check_desk_with_sim=True)，planner FK 预检可能偏保守
    # （collision URDF 手型可能比实际手型更低），保留作快速筛选但不作为最终依据。
    desk_safe, min_z, viol_idx = check_path_desk_safety(planner, path)
    used_safe_descent = False

    if not desk_safe:
        # ── 路径有桌面碰撞 → 尝试安全下降（模拟真机遥操作）──
        print(f"  [{label}] 🔽 path has env collision → trying safe descent")
        descent_path, descent_desc = plan_safe_descent(planner, sim, target_eef, viewer)
        if descent_path is not None:
            print(f"  [{label}] ✅ safe descent: {descent_desc}")
            path = descent_path
            used_safe_descent = True
            desk_safe = True  # override: path is now safe

        if not desk_safe and not check_desk_with_sim:
            print(f"  [{label}] ❌ DESK COLLISION (CollisionModel FCL): waypoint {viol_idx}")
            return False
        elif not desk_safe:
            print(f"  [{label}] ⚠️  CollisionModel预检不安全 → 执行后用 sim FK 确认")

    if not used_safe_descent:
        if joint_goal is not None:
            path = append_joint_goal(planner, path, joint_goal)

    dense = interpolate_waypoints(path, INTERP_MAX_STEP_RAD)
    print(f"  [{label}] exec {len(dense)} wp")

    if not used_safe_descent:
        # Normal execution (safe_descent already executed the path)
        tips_before = check_fingertips(sim)
        hand_qpos = sim.get_full_qpos()[7:]
        execute_dense_path(sim, dense, viewer, physics_steps_per_wp=PHYSICS_STEPS_PER_WP)
        settle_at_target(sim, dense[-1, :7], hand_qpos)

    if used_safe_descent:
        # safe_descent already executed the path; need tips_before for drift check
        tips_before = check_fingertips(sim)
    # ── CollisionModel 桌面碰撞确认（用于手部随机化场景）──
    desk_ok = True
    if check_desk_with_sim:
        sim_desk_safe, _sim_z, sim_viol = check_path_desk_safety_sim(sim, path)
        if not sim_desk_safe:
            print(f"  [{label}] ❌ DESK COLLISION (CollisionModel FCL): waypoint {sim_viol}")
            desk_ok = False

    # 验证
    final_qpos = sim.get_full_qpos()[:7]
    final_eef = planner.compute_eef_pose_world(final_qpos)
    pos_err = float(np.linalg.norm(final_eef.p - target_eef.p))
    rot_err = angular_dist_rad(final_eef.q, target_eef.q)
    ok = pos_err < 0.05

    joint_str = ""
    if joint_goal is not None:
        max_joint_err = float(np.max(np.abs(final_qpos - joint_goal)))
        joint_str = f"  max_joint_err={np.rad2deg(max_joint_err):.2f}deg"
        ok = ok and max_joint_err < np.deg2rad(5.0)

    tips_after = check_fingertips(sim)
    tip_drift_mm = float(np.max(np.abs(tips_after.tip_eef_local - tips_before.tip_eef_local))) * 1000
    tip_ok = tips_after.sapien_fk_delta_mm < 1.0 and tip_drift_mm < 2.0
    ok = ok and tip_ok and desk_ok

    print(
        f"  [{label}] pos_err={pos_err:.4f}m  rot_err={np.rad2deg(rot_err):.2f}deg{joint_str}  "
        f"tip_s2fk={tips_after.sapien_fk_delta_mm:.2f}mm tip_drift={tip_drift_mm:.2f}mm  "
        f"{'desk!' if not desk_ok else ''}  "
        f"[{'OK' if ok else 'FAIL'}]"
    )
    return ok


# ═══════════════════════════════════════════════ IK 测试


def _run_ik_loop(
    planner: XArm7MotionPlanner,
    targets: list[Pose],
    init_qpos: np.ndarray,
    chained: bool,
) -> IKStats:
    """对 targets 列表逐一调 solve_ik()，统计结果。

    chained=False: 每次从 init_qpos 起算。chained=True: seed 用上次 IK 解。
    """
    ok, pos_errs, rot_errs = 0, [], []
    seed = init_qpos.copy()
    for target in targets:
        r = planner.solve_ik(target, seed)
        if not r.success or r.qpos is None:
            pos_errs.append(np.nan)
            rot_errs.append(np.nan)
            continue
        ok += 1
        if chained:
            seed = r.qpos.copy()
        eef = planner.compute_eef_pose_world(r.qpos)
        pos_errs.append(float(np.linalg.norm(eef.p - target.p)))
        rot_errs.append(angular_dist_rad(eef.q, target.q))
    return IKStats(ok=ok, pos_errs_mm=[e * 1000 for e in pos_errs], rot_errs_deg=np.rad2deg(rot_errs).tolist())


def print_ik_stats(label: str, stats: IKStats) -> None:
    pos = np.array(stats.pos_errs_mm)
    rot = np.array(stats.rot_errs_deg)
    valid = ~np.isnan(pos)
    pos_v = pos[valid] if valid.any() else np.array([np.inf])
    rot_v = rot[valid] if valid.any() else np.array([np.inf])
    total = len(pos)
    rate = f"{stats.ok}/{total} ({100*stats.ok/total:.1f}%)" if total else "0"
    print(
        f"  [{label}] success_rate={rate}  "
        f"pos_err: avg={np.mean(pos_v):.1f}mm  max={np.max(pos_v):.1f}mm  "
        f"rot_err: avg={np.mean(rot_v):.2f}deg  max={np.max(rot_v):.2f}deg"
    )


def ik_test(
    planner: XArm7MotionPlanner,
    sim: SimRobotInterface,
    home_qpos: np.ndarray,
    num_samples: int = 50,
    rng: np.random.RandomState | None = None,
) -> dict[str, IKStats]:
    """独立 IK 测试：对随机 EEF 位姿调 solve_ik()，FK 往返验证。"""
    if rng is None:
        rng = np.random.RandomState(SEED)

    home_eef = planner.compute_eef_pose_world(home_qpos)
    positions = np.column_stack(
        [
            rng.uniform(*SAMPLE_X, num_samples),
            rng.uniform(*SAMPLE_Y, num_samples),
            sample_z_biased(rng, num_samples) if num_samples > 20 else rng.uniform(*SAMPLE_Z, num_samples),
        ]
    )
    targets = [
        build_target_pose(
            positions[i],
            home_eef.q,
            rng,
            rot_mode=ROT_MODE,
            rot_max_deg=ROT_MAX_DEG,
            rot_axis1_deg=ROT_AXIS1_DEG,
            rot_axis2_deg=ROT_AXIS2_DEG,
        )
        for i in range(num_samples)
    ]

    return {
        "fresh": _run_ik_loop(planner, targets, home_qpos, chained=False),
        "chained": _run_ik_loop(planner, targets, home_qpos, chained=True),
    }


# ═══════════════════════════════════════════════ Z偏置采样


def sample_z_biased(rng: np.random.RandomState, n: int) -> np.ndarray:
    """Z 偏置采样：低区 [0.02, 0.20] 权重放大。

    危险区（低 Z，穿桌风险）分配更多采样点，与真机测试中重点关注
    低姿态安全性的需求一致。

    Returns: (n,) array of Z values
    """
    weights = np.array([Z_LOW_WEIGHT, 1.0, 1.0])
    weights /= weights.sum()

    z_low = rng.uniform(*Z_LOW_RANGE, n)
    z_mid = rng.uniform(*Z_MID_RANGE, n)
    z_high = rng.uniform(*Z_HIGH_RANGE, n)

    # 按权重从三个区间中随机选择
    choices = rng.choice(3, size=n, p=weights)
    z = np.where(choices == 0, z_low, np.where(choices == 1, z_mid, z_high))
    return z


# ═══════════════════════════════════════════════ 可视化


def place_marker(scene: sapien.Scene, pos: np.ndarray) -> sapien.Actor:
    builder = scene.create_actor_builder()
    builder.add_sphere_visual(radius=MARKER_RADIUS, material=(1.0, 0.2, 0.2))
    m = builder.build_kinematic(name="target_marker")
    m.set_pose(sapien.Pose(p=pos))
    return m


def _setup_viewer(sim: SimRobotInterface) -> sapien.Viewer | None:
    if HEADLESS:
        return None
    add_light(sim.scene)
    return create_viewer(
        sim.scene,
        sapien.Pose(
            [0.784212, 0.0267081, 0.630188],
            [0.00493842, -0.232841, 0.00108951, 0.972502],
        ),
    )


# ═══════════════════════════════════════════════ z_min 网格搜索


def sweep_z_min(
    sim_config: dict,
    headless: bool = True,
    num_samples: int = 30,
    seed: int = SEED,
    with_objects: bool = False,
) -> None:
    """网格搜索最优 hand_safe_margin，平衡安全性与可达性。

    对 margin ∈ [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06] m 逐值测试：
      - 运行 --test-desk 模式（30 个低 Z 目标）
      - 统计：碰撞次数、可达目标数、IK 成功率

    输出 trade-off 表格，帮助选择最佳 z_min。
    """
    margins = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    results = []

    for margin in margins:
        # Build a CollisionConfig with this margin
        cfg = collision_config.with_overrides(hand_safe_margin=margin)
        desk_safe_z = cfg.desk_safe_z

        print(f"\n{'='*60}")
        print(
            f"Sweep: margin={margin:.2f}m  desk_safe_z={desk_safe_z:.3f}m  "
            f"fingertip_threshold={cfg.fingertip_threshold:.3f}m"
        )
        print(f"{'='*60}")

        # Run single test with this config
        stat = _run_desk_test(
            collision_cfg=cfg,
            headless=headless,
            num_samples=num_samples,
            seed=seed,
            with_objects=with_objects,
        )
        results.append((margin, desk_safe_z, stat))

    # Print summary table
    print(f"\n{'='*70}")
    print(f"{'margin(m)':<10} {'desk_safe_z(m)':<15} {'collisions':<12} " f"{'reachable':<12} {'rate':<8}")
    print(f"{'-'*10} {'-'*15} {'-'*12} {'-'*12} {'-'*8}")
    for margin, desk_safe_z, stat in results:
        collisions_str = f"{stat['desk_collisions']}/{stat['total']}"
        reachable_str = f"{stat['reachable']}/{stat['total']}"
        rate_str = f"{stat['rate']:.0f}%"
        print(f"{margin:<10.2f} {desk_safe_z:<15.3f} {collisions_str:<12} " f"{reachable_str:<12} {rate_str:<8}")
    print(f"{'='*70}")
    print("  collisions = sim FK detected fingertip-desk collisions")
    print("  reachable = successfully planned and executed targets")
    print("  rate = reachable / total * 100%")


def _run_desk_test(
    collision_cfg: CollisionConfig,
    headless: bool = True,
    num_samples: int = 30,
    seed: int = SEED,
    with_objects: bool = False,
) -> dict:
    """Run a single --test-desk configuration and return statistics."""
    rng = np.random.RandomState(seed)

    sim = SimRobotInterface(SimRobotConfig(headless=headless))
    if not sim.connect():
        return {"desk_collisions": 0, "reachable": 0, "total": 0, "rate": 0.0}

    root_pose = sim.robot.model.get_root_pose()
    planner = XArm7MotionPlanner(
        XArm7PlannerConfig(
            urdf_path=str(ASSET_DIR / "robots" / "xhand" / "xarm7_xhand_collision.urdf"),
            srdf_path=str(ASSET_DIR / "robots" / "xhand" / "xarm7_xhand.srdf"),
            base_pose_world=Pose(p=np.array(root_pose.p), q=np.array(root_pose.q)),
            workspace_bounds=np.array([[0.0, 0.75], [-0.5, 0.5], [0.0, 0.6]], dtype=np.float64),
            collision=collision_cfg,
        ),
        planning_profile=PlanningProfile(
            max_waypoint_delta_deg=360.0,
            max_ik_delta_deg=(180,) * 7,
            max_pose_error_rot_rad=np.deg2rad(5.0),
            num_random_ik_seeds=30,
            rrt_time_limit=2.0,
            num_rrt_attempts=2,
            random_seed=seed,
            check_env_collision=True,  # enable FK desk safety in validate_path
        ),
        teleop_profile=TeleopProfile(
            check_self_collision=True,
            max_pose_error_pos_m=0.01,
            max_pose_error_rot_rad=np.deg2rad(5.0),
        ),
    )

    home_qpos = sim.config.arm_home_qpos.copy()
    home_eef = planner.compute_eef_pose_world(home_qpos)
    home_quat = home_eef.q.copy()

    # Add table-top objects if requested
    table_objects = []
    if with_objects:
        table_objects = _spawn_table_objects(sim, rng, n_objects=3)

    sim.reset()
    for _ in range(5):
        sim._step_physics(n=10)

    # Desk test targets: low Z range
    target_positions = np.column_stack(
        [
            rng.uniform(*TEST_DESK_X_RANGE, num_samples),
            rng.uniform(*TEST_DESK_Y_RANGE, num_samples),
            rng.uniform(*TEST_DESK_Z_RANGE, num_samples),
        ]
    )

    reachable = 0
    desk_collisions = 0

    for i, pos in enumerate(target_positions):
        # Randomize hand
        hand_qpos, _grasp_type = hand_grasp_pose(rng)
        target_full = np.concatenate([sim.get_full_qpos()[:7], hand_qpos])
        smooth_drive_to_target(sim, target_full, None, max_iter=30, label=f"sweep_{i+1}")

        target_pose = build_target_pose(
            pos,
            home_quat,
            rng,
            rot_mode=ROT_MODE,
            rot_max_deg=ROT_MAX_DEG,
            rot_axis1_deg=ROT_AXIS1_DEG,
            rot_axis2_deg=ROT_AXIS2_DEG,
        )
        current_qpos = sim.get_full_qpos()[:7]

        result = planner.plan_path(target_pose, current_qpos)
        if not result.success or result.qpos_path is None:
            continue

        # Execute path
        path = result.qpos_path
        hand = sim.get_full_qpos()[7:]
        for wp in path:
            sim.robot.balance_passive_force()
            sim.robot.apply_action(np.concatenate([wp, hand]))
            sim._step_physics(n=PHYSICS_STEPS_PER_WP)

        settle_at_target(sim, path[-1, :7], hand)

        # CollisionModel desk check
        desk_safe, _, _ = check_hand_desk_clearance_sim(sim)
        if not desk_safe:
            desk_collisions += 1
            continue

        reachable += 1

    # Clean up table objects
    for obj in table_objects:
        sim.scene.remove_actor(obj)

    sim.disconnect()

    total = num_samples
    rate = (reachable / total * 100) if total > 0 else 0.0
    return {
        "desk_collisions": desk_collisions,
        "reachable": reachable,
        "total": total,
        "rate": rate,
    }


def _spawn_table_objects(
    sim: SimRobotInterface,
    rng: np.random.RandomState,
    n_objects: int = 3,
) -> list:
    """Spawn random objects (box/cylinder) on the table surface.

    Objects are placed within the table footprint at random heights (5-15 cm).
    Returns list of created actors for later cleanup.
    """
    actors = []
    for _ in range(n_objects):
        obj_type = rng.choice(["box", "cylinder"])
        # Random position on table surface
        x = rng.uniform(TABLE_CENTER[0] - TABLE_HALF[0] + 0.1, TABLE_CENTER[0] + TABLE_HALF[0] - 0.1)
        y = rng.uniform(TABLE_CENTER[1] - TABLE_HALF[1] + 0.1, TABLE_CENTER[1] + TABLE_HALF[1] - 0.1)
        # Random height 5-15 cm
        height = rng.uniform(0.05, 0.15)
        z = TABLE_TOP_Z + height / 2  # center Z of the object

        builder = sim.scene.create_actor_builder()
        if obj_type == "box":
            size = rng.uniform(0.03, 0.08)
            builder.add_box_visual(
                half_size=[size, size, height / 2],
                material=(0.3, 0.6, 0.3),  # greenish
            )
            builder.add_box_collision(
                half_size=[size, size, height / 2],
            )
        else:  # cylinder
            radius = rng.uniform(0.02, 0.05)
            builder.add_cylinder_visual(
                radius=radius,
                half_length=height / 2,
                material=(0.3, 0.3, 0.6),  # bluish
            )
            builder.add_cylinder_collision(
                radius=radius,
                half_length=height / 2,
            )

        actor = builder.build_kinematic(name=f"table_obj_{_}")
        actor.set_pose(sapien.Pose(p=[x, y, z]))
        actors.append(actor)

    if actors:
        print(f"  Spawned {len(actors)} table-top objects (heights 5-15 cm)")

    return actors


# ═══════════════════════════════════════════════ Pick-and-Place 场景


@dataclass
class EpisodeResult:
    """单次 Pick-and-Place episode 结果."""

    episode: int = 0
    success: bool = False
    cube_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cube_size: float = 0.05
    total_holds: int = 0
    collision_warnings: int = 0
    duration_s: float = 0.0
    failure_reason: str = ""
    phases_completed: list[str] = field(default_factory=list)


# ── 立方块生成区域 ──
CUBE_X_RANGE = (0.35, 0.70)
CUBE_Y_RANGE = (-0.30, 0.30)
CUBE_SIZE_RANGE = (0.03, 0.08)  # 3-8 cm 正方块
REGION_CENTER = np.array([0.525, 0.0], dtype=np.float64)  # 区域中心 XY

# ── Pick-and-Place 运动参数 ──
APPROACH_Z_OFFSET = 0.12  # 接近阶段：目标物体上方高度 (m)
GRASP_Z_OFFSET = 0.01  # 抓取阶段：物体顶部上方间隙 (m)
GRASP_Z_MIN = 0.12  # 抓取 EEF 最低 Z（避免手指穿透桌面）
LIFT_Z_OFFSET = 0.15  # 抬起阶段：抓取后抬升高度 (m)
PLACE_Z_OFFSET = 0.02  # 放置阶段：桌面/物体顶部上方间隙 (m)
SAFE_RETURN_Z = 0.25  # 返回安全位置的高度 (m)

# ── 碰撞 HOLD 参数 ──
HOLD_DURATION_S = 0.15  # 碰撞检测到后的暂停时间
MAX_CONSECUTIVE_HOLDS = 4  # 同一 waypoint 最大连续 hold 次数（超过则 skip）
MAX_TOTAL_HOLDS = 20  # 单 episode 最大 hold 总次数（超过则 abort）

# ── 手型选择 ──
GRASP_TYPES_ALL = ["hook", "open"]  # 排除 pinch/power（拇指延伸致物理穿桌）


def spawn_pick_cube(
    sim: SimRobotInterface,
    rng: np.random.RandomState,
) -> dict:
    """在桌面指定区域随机生成一个正方块（SAPIEN kinematic actor）。"""
    half = float(rng.uniform(*CUBE_SIZE_RANGE)) / 2
    cx = float(rng.uniform(*CUBE_X_RANGE))
    cy = float(rng.uniform(*CUBE_Y_RANGE))
    cz = TABLE_TOP_Z + half

    color = tuple(float(x) for x in rng.uniform(0.1, 0.95, 3))

    builder = sim.scene.create_actor_builder()
    builder.add_box_visual(half_size=[half, half, half], material=color)
    builder.add_box_collision(half_size=[half, half, half])
    actor = builder.build_kinematic(name=f"pick_cube_{rng.randint(0, 9999)}")
    actor.set_pose(sapien.Pose(p=[cx, cy, cz]))

    return {
        "actor": actor,
        "half_size": half,
        "center_pos": (cx, cy, cz),
        "color": color,
    }


def execute_path_with_collision_hold(
    sim: SimRobotInterface,
    path_arm: np.ndarray,
    cm: CollisionModel,
    hand_qpos: np.ndarray,
    viewer: sapien.Viewer | None = None,
    *,
    episode_idx: int = 0,
    label: str = "",
    cube_actor: sapien.Actor | None = None,
    is_grasping: bool = False,
    eef_to_cube_offset: np.ndarray | None = None,
    check_env: bool | str = True,  # True=full HOLD, "warn"=env→warn-only, False=self-only
) -> tuple[bool, int, int]:
    """执行 arm 关节路径，逐 waypoint 碰撞检测 + HOLD 行为。

    先检查再执行（与遥操作 HOLD 模式一致）：
      1. cm.check_teleop_collision(wp) → 碰撞检测
      2. 碰撞 → [HOLD] sleep → retry 同一 waypoint
      3. 连续 > MAX_CONSECUTIVE_HOLDS → [SKIP]
      4. 总计 > MAX_TOTAL_HOLDS → [ABORT]
      5. 无碰撞 → execute + step_physics

    Returns: (completed, total_holds, collision_warnings)
    """
    total_holds = 0
    collision_warnings = 0
    consecutive_holds = 0
    wp_idx = 0

    while wp_idx < len(path_arm):
        wp = path_arm[wp_idx]

        # ── 碰撞检测：先检查后执行 ──
        # True: self+env → HOLD; "warn": self→HOLD, env→日志; False: self-only
        if check_env is True:
            has_self, has_env = cm.check_teleop_collision(wp)
        elif check_env == "warn":
            has_self = cm.check_self_collision(wp)
            has_env = cm.check_env_collision(wp)  # Tier2 FCL, 比 Tier1 更精确
        else:
            has_self = cm.check_self_collision(wp)
            has_env = False

        if has_self or has_env:
            parts = []
            if has_self:
                parts.append("self")
            if has_env:
                parts.append("env")
            reason = "+".join(parts)
            collision_warnings += 1

            # ── warn 模式：仅 env → 日志警告，非阻塞 ──
            if check_env == "warn" and has_env and not has_self:
                print(
                    f"  ⚠️  ep={episode_idx} {label} wp={wp_idx+1}/{len(path_arm)} "
                    f"env near table (expected, non-blocking)"
                )
                collision_warnings -= 1  # 不计入 collision counter
                consecutive_holds = 0
                # 继续执行（fall through to execution below — 不 sleep/continue）

            # ── self / strict env → HOLD ──
            elif has_self or (check_env is True and has_env):
                total_holds += 1
                consecutive_holds += 1

                print(
                    f"  [HOLD] ep={episode_idx} {label} wp={wp_idx+1}/{len(path_arm)} "
                    f"reason={reason:<6s} consecutive={consecutive_holds} total={total_holds}"
                )

                if total_holds > MAX_TOTAL_HOLDS:
                    print(
                        f"  [ABORT] ep={episode_idx} too many total holds "
                        f"({total_holds} > {MAX_TOTAL_HOLDS}) during {label}"
                    )
                    return False, total_holds, collision_warnings

                if consecutive_holds > MAX_CONSECUTIVE_HOLDS:
                    print(
                        f"  [SKIP] ep={episode_idx} skipping wp {wp_idx+1} "
                        f"after {consecutive_holds} consecutive holds"
                    )
                    wp_idx += 1
                    consecutive_holds = 0
                    continue

                time.sleep(HOLD_DURATION_S)
                continue

        # ── 无碰撞：执行 waypoint ──
        consecutive_holds = 0
        full_target = np.concatenate([wp, hand_qpos])
        sim.robot.balance_passive_force()
        sim.robot.apply_action(full_target)
        sim._step_physics(n=PHYSICS_STEPS_PER_WP)

        if is_grasping and cube_actor is not None:
            eef_pose = sim.robot.get_eef_pose()
            offset = eef_to_cube_offset if eef_to_cube_offset is not None else np.zeros(3)
            cube_p = np.array(eef_pose.p) + offset
            cube_actor.set_pose(sapien.Pose(p=cube_p))

        if viewer is not None and wp_idx % 4 == 0:
            sim.scene.update_render()
            viewer.render()

        wp_idx += 1

    return True, total_holds, collision_warnings


def pick_and_place_episode(
    sim: SimRobotInterface,
    planner: XArm7MotionPlanner,
    cm: CollisionModel,
    home_qpos: np.ndarray,
    home_quat: np.ndarray,
    cube_info: dict,
    episode_idx: int,
    viewer: sapien.Viewer | None = None,
    rng: np.random.RandomState | None = None,
) -> EpisodeResult:
    """执行一次完整的 Pick-and-Place episode。

    流程: APPROACH → DESCEND → GRASP → LIFT → TRANSPORT → PLACE → RELEASE → RETURN
    """
    t_start = time.perf_counter()
    result = EpisodeResult(
        episode=episode_idx,
        cube_pos=cube_info["center_pos"],
        cube_size=cube_info["half_size"] * 2,
    )

    cube_actor = cube_info["actor"]
    cube_half = cube_info["half_size"]
    cube_cx, cube_cy, cube_cz = cube_info["center_pos"]
    cube_top_z = cube_cz + cube_half

    # ── 选择手型（check 无碰撞才使用）──
    arm_at_home = sim.get_full_qpos()[:7].copy()
    grasp_hand_qpos: np.ndarray | None = None
    grasp_type: str = "open"
    for _ in range(10):
        candidate_type = rng.choice(GRASP_TYPES_ALL) if rng else "hook"
        candidate_qpos, _ = hand_grasp_pose(rng, grasp_type=candidate_type)
        cm.set_hand_qpos(candidate_qpos)
        has_self, has_env = cm.check_teleop_collision(arm_at_home)
        if not has_self and not has_env:
            grasp_hand_qpos = candidate_qpos
            grasp_type = candidate_type
            break
    else:
        grasp_hand_qpos = np.zeros(12, dtype=np.float64)
        grasp_type = "open"
        cm.set_hand_qpos(grasp_hand_qpos)
        print(f"  ⚠️  All hand poses in collision at home, fallback to open-hand")

    sim.robot.balance_passive_force()
    sim.robot.apply_action(np.concatenate([arm_at_home, grasp_hand_qpos]))
    sim._step_physics(n=5)
    hand_qpos = grasp_hand_qpos.copy()

    # ── 计算各阶段目标位姿（Z 钳位确保手指不穿桌面）──
    grasp_z = max(cube_top_z + GRASP_Z_OFFSET, TABLE_TOP_Z + GRASP_Z_MIN)
    approach_z = cube_top_z + APPROACH_Z_OFFSET

    approach_pose = Pose(p=np.array([cube_cx, cube_cy, approach_z], dtype=np.float64), q=home_quat.copy())
    grasp_pose = Pose(p=np.array([cube_cx, cube_cy, grasp_z], dtype=np.float64), q=home_quat.copy())
    lift_pose = Pose(p=np.array([cube_cx, cube_cy, grasp_z + LIFT_Z_OFFSET], dtype=np.float64), q=home_quat.copy())
    transport_pose = Pose(
        p=np.array([REGION_CENTER[0], REGION_CENTER[1], grasp_z + LIFT_Z_OFFSET], dtype=np.float64), q=home_quat.copy()
    )
    place_pose = Pose(
        p=np.array(
            [
                REGION_CENTER[0],
                REGION_CENTER[1],
                max(TABLE_TOP_Z + cube_half * 2 + PLACE_Z_OFFSET, TABLE_TOP_Z + GRASP_Z_MIN, 0.15),
            ],
            dtype=np.float64,
        ),
        q=home_quat.copy(),
    )

    phases = [
        ("approach", approach_pose),
        ("descend", grasp_pose),
        ("lift+transport", transport_pose),
        ("place", place_pose),
    ]

    print(f"\n{'─'*55}")
    print(
        f"  🎯 Episode {episode_idx+1}: cube=({cube_cx:.3f}, {cube_cy:.3f}, {cube_cz:.3f}) "
        f"size={cube_half*2*100:.1f}cm  hand={grasp_type}"
    )
    print(f"{'─'*55}")

    is_grasping = False
    eef_to_cube_offset: np.ndarray | None = None
    current_qpos = arm_at_home.copy()

    # ── 闭合手型 (power grip): 低噪声确保视觉一致性 ──
    close_hand_qpos, close_grasp_type = hand_grasp_pose(rng, grasp_type="power")
    close_hand_qpos = np.clip(close_hand_qpos, np.deg2rad(-10), np.deg2rad(90))

    for phase_name, target_pose in phases:
        # ── lift+transport / place: 使用直接 IK + joint-space 插值 ──
        # plan_path 的保守手部碰撞模型 (home=全伸展) 会在近桌面位姿误杀所有 IK 候选。
        # 这些阶段是短距离垂直运动，joint-space 插值可安全替代规划。
        if phase_name in ("lift+transport", "place"):
            ik_result = planner.solve_teleop_ik(target_pose, current_qpos, current_qpos)
            if not ik_result.success or ik_result.qpos is None:
                result.failure_reason = f"{phase_name} IK failed: {ik_result.reason}"
                print(f"  ❌ {phase_name} IK: {ik_result.reason}")
                result.duration_s = time.perf_counter() - t_start
                return result
            # Joint-space linear path (no collision check needed — moving upward)
            joint_delta = float(np.max(np.abs(ik_result.qpos - current_qpos)))
            n_wp = max(2, int(np.ceil(joint_delta / INTERP_MAX_STEP_RAD)) + 1)
            path = np.array([current_qpos + (k / (n_wp - 1)) * (ik_result.qpos - current_qpos) for k in range(n_wp)])
            interp_label = f"ik({n_wp}wp)"
        else:
            try:
                plan_result = planner.plan_path(target_pose, current_qpos)
            except RuntimeError as e:
                result.failure_reason = f"{phase_name} plan_path error: {e}"
                print(f"  ❌ {phase_name}: plan_path exception: {e}")
                result.duration_s = time.perf_counter() - t_start
                return result

            if not plan_result.success or plan_result.qpos_path is None:
                result.failure_reason = f"{phase_name} plan failed: {plan_result.reason}"
                print(f"  ❌ {phase_name}: {plan_result.reason}")
                result.duration_s = time.perf_counter() - t_start
                return result

            path = plan_result.qpos_path
            interp_label = plan_result.report.get("num_waypoints", "?")

        dense = interpolate_waypoints(path, INTERP_MAX_STEP_RAD)

        # 桌面是已知静态障碍物，各阶段有意接近/远离它。
        # 自碰撞 → HOLD（危险），环境碰撞 → WARNING 日志（非阻塞）。
        _check_env = "warn"
        completed, holds, warnings = execute_path_with_collision_hold(
            sim,
            dense,
            cm,
            hand_qpos,
            viewer,
            episode_idx=episode_idx + 1,
            label=phase_name,
            cube_actor=cube_actor if is_grasping else None,
            is_grasping=is_grasping,
            eef_to_cube_offset=eef_to_cube_offset if is_grasping else None,
            check_env=_check_env,
        )

        result.total_holds += holds
        result.collision_warnings += warnings
        result.phases_completed.append(phase_name)

        if not completed:
            result.failure_reason = f"{phase_name} aborted: {holds} holds, {warnings} warnings"
            print(f"  ⚠️  {phase_name} ABORTED (holds={holds} warnings={warnings})")
            result.duration_s = time.perf_counter() - t_start
            return result

        current_qpos = dense[-1].copy()

        # ── DESCEND → 闭合手部抓取立方块 ──
        if phase_name == "descend" and not is_grasping:
            # 平滑动画：open → close (power grip 包裹物体)
            _n_close = 20
            for _step in range(_n_close):
                alpha = (_step + 1) / _n_close
                blend_hand = hand_qpos * (1.0 - alpha) + close_hand_qpos * alpha
                current_arm = sim.get_full_qpos()[:7]
                sim.robot.balance_passive_force()
                sim.robot.apply_action(np.concatenate([current_arm, blend_hand]))
                sim._step_physics(n=PHYSICS_STEPS_PER_WP)
                if viewer is not None and _step % 4 == 0:
                    sim.scene.update_render()
                    viewer.render()
            hand_qpos = close_hand_qpos.copy()
            cm.set_hand_qpos(hand_qpos)

            is_grasping = True
            eef_pose = sim.robot.get_eef_pose()
            cube_p = np.array([cube_cx, cube_cy, cube_cz], dtype=np.float64)
            eef_to_cube_offset = cube_p - np.array(eef_pose.p)
            print(f"  ✋ CLOSE: {close_grasp_type} grip, cube attached " f"offset={np.round(eef_to_cube_offset, 3)}")

        # ── PLACE → 张开放置并释放立方块 ──
        if phase_name == "place":
            is_grasping = False
            place_p = np.array([REGION_CENTER[0], REGION_CENTER[1], TABLE_TOP_Z + cube_half], dtype=np.float64)
            cube_actor.set_pose(sapien.Pose(p=place_p))

            # 平滑动画：close → open (释放物体)
            _n_open = 15
            for _step in range(_n_open):
                alpha = (_step + 1) / _n_open
                blend_hand = hand_qpos * (1.0 - alpha) + grasp_hand_qpos * alpha
                current_arm = sim.get_full_qpos()[:7]
                sim.robot.balance_passive_force()
                sim.robot.apply_action(np.concatenate([current_arm, blend_hand]))
                sim._step_physics(n=PHYSICS_STEPS_PER_WP)
                if viewer is not None and _step % 3 == 0:
                    sim.scene.update_render()
                    viewer.render()
            hand_qpos = grasp_hand_qpos.copy()
            cm.set_hand_qpos(hand_qpos)
            print(f"  📍 OPEN: cube released at ({place_p[0]:.3f}, {place_p[1]:.3f}, {place_p[2]:.3f})")

        settle_at_target(sim, current_qpos, hand_qpos)
        print(f"  ✅ {phase_name:<15s} {interp_label}wp  holds={holds}  warn={warnings}")

    # ── RETURN to safe position ──
    print(f"  🔄 return: moving to safe position...")
    current_qpos = sim.get_full_qpos()[:7].copy()
    current_pose = planner.compute_eef_pose_world(current_qpos)
    safe_return_pose = Pose(
        p=np.array(
            [current_pose.p[0], current_pose.p[1], max(current_pose.p[2] + 0.10, SAFE_RETURN_Z)], dtype=np.float64
        ),
        q=home_quat.copy(),
    )

    try:
        return_result = planner.plan_path(safe_return_pose, current_qpos)
        if return_result.success and return_result.qpos_path is not None:
            return_dense = interpolate_waypoints(return_result.qpos_path, INTERP_MAX_STEP_RAD)
            completed, holds, warnings = execute_path_with_collision_hold(
                sim,
                return_dense,
                cm,
                hand_qpos,
                viewer,
                episode_idx=episode_idx + 1,
                label="return",
            )
            result.total_holds += holds
            result.collision_warnings += warnings
            result.phases_completed.append("return")
            if not completed:
                print(f"  ⚠️  return ABORTED (holds={holds})")
            else:
                settle_at_target(sim, return_dense[-1], hand_qpos)
                print(f"  ✅ return   holds={holds}")
    except RuntimeError:
        print(f"  ⚠️  return plan failed, using animated reset")
        animated_reset_to_home(sim, home_qpos, viewer)

    result.success = len(result.phases_completed) >= 5
    result.duration_s = time.perf_counter() - t_start
    status = "✅ SUCCESS" if result.success else f"❌ FAILED: {result.failure_reason}"
    print(f"  {status}  holds={result.total_holds}  warnings={result.collision_warnings}  t={result.duration_s:.1f}s")
    return result


# ═══════════════════════════════════════════════ 主流程


def main():
    import argparse

    p = argparse.ArgumentParser(description="Pick-and-Place 抓取放置仿真测试")
    p.add_argument("--headless", action="store_true", help="无头模式（无 GUI）")
    p.add_argument("--seed", type=int, default=SEED, help="随机种子")
    p.add_argument("--episodes", type=int, default=20, help="episode 数量")
    args = p.parse_args()

    seed = args.seed
    headless = HEADLESS or args.headless
    num_episodes = args.episodes
    rng = np.random.RandomState(seed)

    print("=" * 70)
    print(f"  Pick-and-Place 抓取放置仿真测试 — {num_episodes} episodes")
    print("=" * 70)
    print(f"  桌面区域: x{CUBE_X_RANGE} y{CUBE_Y_RANGE}")
    print(f"  立方块:   {CUBE_SIZE_RANGE[0]*100:.0f}-{CUBE_SIZE_RANGE[1]*100:.0f}cm")
    print(f"  区域中心: ({REGION_CENTER[0]:.3f}, {REGION_CENTER[1]:.3f})")
    print(f"  手型: {GRASP_TYPES_ALL}")
    print(f"  碰撞HOLD: {HOLD_DURATION_S}s × max {MAX_CONSECUTIVE_HOLDS} consecutive, {MAX_TOTAL_HOLDS} total")

    sim = SimRobotInterface(SimRobotConfig(headless=headless))
    if not sim.connect():
        print(f"ERROR: connect failed: {sim.last_error_message}", file=sys.stderr)
        return

    root_pose = sim.robot.model.get_root_pose()
    planner = XArm7MotionPlanner(
        XArm7PlannerConfig(
            urdf_path=str(ASSET_DIR / "robots" / "xhand" / "xarm7_xhand_collision.urdf"),
            srdf_path=str(ASSET_DIR / "robots" / "xhand" / "xarm7_xhand.srdf"),
            base_pose_world=Pose(p=np.array(root_pose.p), q=np.array(root_pose.q)),
            workspace_bounds=np.array([[0.0, 0.75], [-0.5, 0.5], [0.0, 0.6]], dtype=np.float64),
        ),
        planning_profile=PlanningProfile(
            max_waypoint_delta_deg=360.0,
            max_ik_delta_deg=(180,) * 7,
            max_pose_error_rot_rad=np.deg2rad(5.0),
            num_random_ik_seeds=50,
            rrt_time_limit=2.0,
            num_rrt_attempts=2,
            random_seed=seed,
        ),
        teleop_profile=TeleopProfile(check_self_collision=True),
    )

    home_qpos = sim.config.arm_home_qpos.copy()
    home_eef = planner.compute_eef_pose_world(home_qpos)
    home_quat = home_eef.q.copy()

    # ── 注册 CollisionModel + 桌面障碍物 ──
    global _env_cm
    _env_cm = planner.collision_model
    cm = _env_cm

    table_world = np.array([TABLE_CENTER[0], TABLE_CENTER[1], TABLE_TOP_Z], dtype=np.float64)
    table_in_urdf = root_pose.inv() * sapien.Pose(table_world)
    cm.add_table(
        table_height=float(table_in_urdf.p[2]),
        x_center=float(table_in_urdf.p[0]),
        half_x=TABLE_HALF[0],
        half_y=TABLE_HALF[1],
        half_z=TABLE_HALF[2],
    )
    print(
        f"  CollisionModel: {cm.nq}-DOF, {cm._collision_model.ngeoms} geometries, "
        f"{len(cm._collision_model.collisionPairs)} pairs"
    )
    print(f"  桌面障碍物: table @ z_top={TABLE_TOP_Z:.1f}m")

    sim.reset()
    for _ in range(5):
        sim._step_physics(n=10)

    viewer = _setup_viewer(sim) if not headless else None

    print(f"\n{'='*70}")
    print(f"  开始 {num_episodes} 次 Pick-and-Place 循环")
    print(f"{'='*70}")

    episode_results: list[EpisodeResult] = []
    t_total_start = time.perf_counter()

    for ep in range(num_episodes):
        cube_info = spawn_pick_cube(sim, rng)
        cube_half = cube_info["half_size"]
        cube_cx, cube_cy, _ = cube_info["center_pos"]

        print(
            f"\n  🧊 Episode {ep+1}/{num_episodes}: "
            f"cube @ ({cube_cx:.3f}, {cube_cy:.3f}) size={cube_half*2*100:.1f}cm"
        )

        ep_result = pick_and_place_episode(sim, planner, cm, home_qpos, home_quat, cube_info, ep, viewer, rng)
        episode_results.append(ep_result)

        if cube_info["actor"] is not None:
            try:
                sim.scene.remove_actor(cube_info["actor"])
            except Exception:
                pass

        print(f"  🔄 resetting to home...")
        sim.reset()
        for _ in range(5):
            sim._step_physics(n=10)
        sim.robot.balance_passive_force()
        sim.robot.apply_action(np.concatenate([home_qpos, np.zeros(12, dtype=np.float64)]))
        sim._step_physics(n=10)

    # ── 汇总报告 ──
    t_total = time.perf_counter() - t_total_start
    n_success = sum(1 for r in episode_results if r.success)
    n_failed = num_episodes - n_success

    print(f"\n{'='*70}")
    print(f"  Pick-and-Place 汇总报告")
    print(f"{'='*70}")
    print(f"\n  📊 总体统计:")
    print(f"    Episodes:        {num_episodes}")
    print(f"    成功:            {n_success} ({n_success/max(num_episodes,1)*100:.0f}%)")
    print(f"    失败:            {n_failed} ({n_failed/max(num_episodes,1)*100:.0f}%)")

    total_holds = sum(r.total_holds for r in episode_results)
    total_warnings = sum(r.collision_warnings for r in episode_results)
    episodes_with_holds = sum(1 for r in episode_results if r.total_holds > 0)

    print(f"\n  🛡️  碰撞 HOLD 统计:")
    print(f"    总 HOLD 次数:    {total_holds}")
    print(f"    总碰撞预警:      {total_warnings}")
    print(f"    触发 HOLD:       {episodes_with_holds}/{num_episodes}")
    print(f"    平均 HOLD/ep:    {total_holds/max(num_episodes,1):.1f}")
    print(f"    最大 HOLD/ep:    {max((r.total_holds for r in episode_results), default=0)}")

    if n_failed > 0:
        print(f"\n  ❌ 失败原因:")
        reasons: dict[str, int] = {}
        for r in episode_results:
            if not r.success:
                reason = r.failure_reason or "unknown"
                if "plan fail" in reason.lower():
                    reason = "planning failure"
                elif "abort" in reason.lower():
                    reason = "collision abort"
                reasons[reason] = reasons.get(reason, 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason:<25s} {count}")

    print(f"\n  📋 阶段完成率:")
    for phase in ["approach", "descend", "lift+transport", "place", "return"]:
        completed = sum(1 for r in episode_results if phase in r.phases_completed)
        pct = completed / max(num_episodes, 1) * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"    {phase:<10s} {bar} {completed}/{num_episodes} ({pct:.0f}%)")

    durations = [r.duration_s for r in episode_results]
    print(f"\n  ⏱️  性能:")
    print(f"    总耗时:          {t_total:.1f}s")
    print(f"    平均/episode:    {np.mean(durations):.1f}s" if durations else "    N/A")
    print(f"    最快/最慢:       {min(durations):.1f}s / {max(durations):.1f}s" if durations else "    N/A")

    print(f"\n  📝 Episode 明细:")
    print(f"  {'#':>3s}  {'cube_pos':>18s}  {'size':>5s}  {'holds':>6s}  {'dur':>6s}  {'result'}")
    print(f"  {'─'*3}  {'─'*18}  {'─'*5}  {'─'*6}  {'─'*6}  {'─'*8}")
    for r in episode_results:
        pos_str = f"({r.cube_pos[0]:.2f},{r.cube_pos[1]:.2f})"
        status = "✅" if r.success else f"❌ {r.failure_reason[:25]}"
        print(
            f"  {r.episode+1:>3d}  {pos_str:>18s}  {r.cube_size*100:>4.1f}cm  "
            f"{r.total_holds:>5d}  {r.duration_s:>5.1f}s  {status}"
        )

    cm.clear_obstacles()
    if viewer and not viewer.closed:
        print("\nClose viewer to exit...")
        while not viewer.closed:
            sim.scene.update_render()
            viewer.render()
    sim.disconnect()
    print(f"\n{'='*70}")
    print(f"  ✅ Pick-and-Place 测试完成 ({n_success}/{num_episodes} success)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
