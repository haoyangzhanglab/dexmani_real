#!/usr/bin/env python3
"""Workspace 随机采样 — 测试 Planner 路径规划（仿真模型 + SAPIEN 可视化）。

用法:
    conda activate real
    python scripts/sim/test_motion_planning_sim.py

修改顶部常量控制测试。

测试流程:
    1. 随机采样 N 个 EEF 位姿（位置+姿态），plan_path → 执行 → 验证
    2. return_home: plan_path(home_eef) + 关节归位（含碰撞检测）
    3. IK 独立测试：solve_ik() 成功率 + FK 往返误差
"""

from __future__ import annotations

# sys.path修正：脚本已从dexmani_real/example/移至scripts/
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))


import sys
import time
from dataclasses import dataclass

import numpy as np
import sapien.core as sapien

from dexmani_real import ASSET_DIR
from dexmani_real.planning import (
    CollisionConfig, PlanningProfile, Pose, TeleopProfile, XArm7MotionPlanner, XArm7PlannerConfig,
)
from dexmani_real.simulation import SimRobotConfig, SimRobotInterface
from dexmani_real.simulation.constructor import add_light, create_viewer

# ═══════════════════════════════════════════════ 配置

NUM_SAMPLES = 30           # 默认路径规划采样点
NUM_IK_SAMPLES = 100       # 默认 IK 采样点
HEADLESS = False
SEED = 123

# --comprehensive 模式配置
COMPREHENSIVE_NUM_SAMPLES = 60
COMPREHENSIVE_NUM_IK = 200

# 采样空间（world frame，与真机 workspace 一致）
SAMPLE_X = (0.28, 0.70)
SAMPLE_Y = (-0.40, 0.40)
SAMPLE_Z = (0.02, 0.55)

# Z 偏置：低区 [0.02, 0.20] 权重放大（穿桌高风险区）
Z_LOW_WEIGHT = 4.0         # 低区相对权重
Z_LOW_RANGE = (0.02, 0.20)
Z_MID_RANGE = (0.20, 0.36)
Z_HIGH_RANGE = (0.36, 0.55)

# 姿态随机化配置
ROT_MODE = "multi_axis"    # "fixed" | "single_axis" | "multi_axis" | "full_so3"
ROT_MAX_DEG = 60.0          # 最大旋转角度（单轴模式）
ROT_AXIS1_DEG = 45.0        # 第一轴最大角度（多轴模式）
ROT_AXIS2_DEG = 30.0        # 第二轴最大角度（多轴模式）

# ── 桌面碰撞几何（与 SAPIEN add_base_components 的 table 对齐）──
TABLE_CENTER = (0.4, 0.0, -0.5)  # table actor 位置 (constructor.py:100)
TABLE_HALF = (0.5, 1.0, 0.5)     # half_size (constructor.py:97-98)
TABLE_TOP_Z = TABLE_CENTER[2] + TABLE_HALF[2]  # = 0.0 桌面上表面
# 桌面碰撞检测使用 Pinocchio FK 直接计算五指指尖世界坐标。
# home 手型下 pinky_tip 比 EEF 低 7.6cm（不是拇指的 11.3cm — 旧值是中间关节 ID 39 的 pinky_link2）。

# ── 统一碰撞配置（通过 CollisionConfig 管理，替代之前分散的常量）──
# 旧常量映射（仅作参考，不与 CollisionConfig 重复赋值）：
#   HAND_EXTENSION_BELOW_EEF → collision_config.hand_extension_below_eef (0.076)
#   HAND_SAFE_MARGIN         → collision_config.hand_safe_margin (0.03)
#   DESK_SAFE_Z              → collision_config.desk_safe_z (0.106)
#   REJECT_BELOW_DESK_Z      → collision_config.reject_below_desk_z
#   REJECT_EPSILON           → FingertipDeskSafety._epsilon (0.001)
collision_config = CollisionConfig(
    table_z_world=TABLE_TOP_Z,
    hand_extension_below_eef=0.076,
    hand_safe_margin=0.03,
    reject_below_desk_z=True,
)
# 向后兼容别名（test_motion_planning_sim.py 内大量引用这些旧名，逐步迁移）
DESK_SAFE_Z = collision_config.desk_safe_z
REJECT_BELOW_DESK_Z = collision_config.reject_below_desk_z

# 手部随机化配置
RANDOMIZE_HAND = False          # 是否随机化手部关节（测试不同手型下的桌面碰撞）
HAND_RANDOM_RANGE_DEG = 30.0    # 手部关节随机范围 ±30°（相对 home=0°）
NUM_HAND_JOINTS = 12            # 手部 DOF（xhand_right.urdf 中 12 个 revolute 关节）

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
    "power": {      # 力量抓取（柱状）：四指全弯 60-90°，拇指包覆
        "finger_curl": (60, 90),   # (min_deg, max_deg) 四指基关节弯曲范围
        "finger_tip_scale": 0.8,   # 尖关节 = 基关节 × scale
        "thumb_bend": (60, 90),    # 拇指弯曲
        "thumb_spread": (10, 30),  # 拇指外展（rota1）
        "noise": 10,               # 关节噪声 ±deg
    },
    "pinch": {      # 精确捏取（指尖）：拇指+食指微弯对捏，其余三指卷起
        "finger_curl": (20, 50),
        "finger_tip_scale": 0.5,
        "thumb_bend": (30, 60),
        "thumb_spread": (20, 45),
        "noise": 5,
    },
    "tripod": {     # 三指捏取（笔握）：拇+食+中，无名+小指卷起
        "finger_curl": (30, 60),
        "finger_tip_scale": 0.6,
        "thumb_bend": (30, 60),
        "thumb_spread": (15, 40),
        "noise": 8,
    },
    "open": {       # 张开手掌：五指展开，微弯
        "finger_curl": (0, 20),
        "finger_tip_scale": 0.3,
        "thumb_bend": (0, 20),
        "thumb_spread": (30, 60),
        "noise": 15,
    },
    "hook": {       # 钩状抓取：基关节中立，尖关节强弯
        "finger_curl": (0, 15),
        "finger_tip_scale": 1.5,   # 尖关节比基关节弯更多
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

# 分层采样区域（Z低区重点覆盖：穿桌高风险区分配更多区域）
STRATIFIED_REGIONS = [
    # (label, x_range, y_range, z_range, weight)
    # ── Z 低区 [0.02, 0.20]：重点覆盖（权重 ×4），共 6 个子区域 ──
    ("near_low_a",   (0.28, 0.40), (-0.30, 0.30), (0.02, 0.10), 4.0),
    ("near_low_b",   (0.28, 0.40), (-0.30, 0.30), (0.10, 0.20), 4.0),
    ("mid_low_a",    (0.40, 0.55), (-0.40, 0.40), (0.02, 0.10), 4.0),
    ("mid_low_b",    (0.40, 0.55), (-0.40, 0.40), (0.10, 0.20), 4.0),
    ("far_low_a",    (0.55, 0.70), (-0.40, 0.40), (0.02, 0.10), 4.0),
    ("far_low_b",    (0.55, 0.70), (-0.40, 0.40), (0.10, 0.20), 4.0),
    # ── Z 中区 [0.20, 0.36]：正常操作区 ──
    ("near_mid",     (0.28, 0.40), (-0.30, 0.30), (0.20, 0.36), 1.0),
    ("mid_mid",      (0.40, 0.55), (-0.40, 0.40), (0.20, 0.36), 1.0),
    ("far_mid",      (0.55, 0.70), (-0.40, 0.40), (0.20, 0.36), 1.0),
    # ── Z 高区 [0.36, 0.55]：安全区 ──
    ("near_high",    (0.28, 0.40), (-0.30, 0.30), (0.36, 0.55), 0.5),
    ("mid_high",     (0.40, 0.55), (-0.40, 0.40), (0.36, 0.55), 0.5),
    ("far_high",     (0.55, 0.70), (-0.40, 0.40), (0.36, 0.55), 0.5),
]

PHYSICS_STEPS_PER_WP = 20
INTERP_MAX_STEP_RAD = np.deg2rad(2.0)
MARKER_RADIUS = 0.015
RANDOM_ROT_DEG = 30.0

# ── return_to_home 参数（与 RobotInterface 保持一致）──
_HOME_JOINT_THRESHOLD_RAD = np.deg2rad(1.0)      # 视为已归位的关节偏差
_PHASE1_CONVERGE_THRESHOLD_RAD = np.deg2rad(3.0)  # Phase 1 收敛阈值
_PHASE2_MIN_DELTA_RAD = np.deg2rad(0.5)            # Phase 2 跳过的关节偏差下限
_PHASE2_MAX_STEP_RAD = np.deg2rad(1.0)             # Phase 2 关节空间最大步长
_DIRECT_LIFT_Z_M = 0.15                             # 安全抬升高度
_DIRECT_LIFT_SLEEP_STEPS = 6                        # 抬升后稳定步数 (0.3s ÷ 0.05s per step)
_RESIDUAL_ERROR_MAX_DEG = 10.0                      # 残余误差上限（超过此值返回失败）

# ═══════════════════════════════════════════════ 数学工具


def angular_dist_rad(q1: np.ndarray, q2: np.ndarray) -> float:
    return float(2 * np.arccos(np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0)))


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """wxyz 四元数 Hamilton 乘积。"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_to_rotmat(q_wxyz: np.ndarray) -> np.ndarray:
    """wxyz 四元数 → 3x3 旋转矩阵。"""
    w, x, y, z = q_wxyz
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y,     2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y],
    ])


def random_quat_within_angle(rng: np.random.RandomState, max_deg: float) -> np.ndarray:
    """均匀随机旋转，角度 ≤ max_deg。返回 wxyz 四元数。"""
    axis = rng.randn(3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(0, np.deg2rad(max_deg))
    half = angle / 2
    return np.array([np.cos(half), axis[0] * np.sin(half),
                     axis[1] * np.sin(half), axis[2] * np.sin(half)])


def random_quat_full_so3(rng: np.random.RandomState) -> np.ndarray:
    """均匀采样 SO(3) 全空间随机四元数（wxyz）。"""
    # Marsaglia 方法：均匀分布在单位球面
    u = rng.uniform(0, 1, 3)
    q = np.array([
        np.sqrt(1 - u[0]) * np.sin(2 * np.pi * u[1]),
        np.sqrt(1 - u[0]) * np.cos(2 * np.pi * u[1]),
        np.sqrt(u[0]) * np.sin(2 * np.pi * u[2]),
        np.sqrt(u[0]) * np.cos(2 * np.pi * u[2]),
    ])
    q /= np.linalg.norm(q)
    return q


def random_quat_multi_axis(
    rng: np.random.RandomState, max_deg1: float = 45.0, max_deg2: float = 30.0,
) -> np.ndarray:
    """绕两个独立随机轴依次旋转，产生更丰富的姿态分布。

    先绕 axis1 旋转 angle1 ∈ [0, max_deg1]，再绕 axis2 旋转 angle2 ∈ [0, max_deg2]，
    合成旋转 = R2 * R1。比单轴旋转覆盖更大的 SO(3) 子集。
    """
    # Axis 1: random direction
    a1 = rng.randn(3)
    a1 /= np.linalg.norm(a1)
    angle1 = rng.uniform(0, np.deg2rad(max_deg1))
    half1 = angle1 / 2
    q1 = np.array([np.cos(half1), a1[0] * np.sin(half1),
                   a1[1] * np.sin(half1), a1[2] * np.sin(half1)])

    # Axis 2: orthogonal to axis 1 (or random)
    a2 = rng.randn(3)
    a2 -= a1 * np.dot(a2, a1)  # orthogonalize
    norm = np.linalg.norm(a2)
    if norm < 1e-10:
        a2 = np.array([-a1[1], a1[0], 0.0]) if abs(a1[0]) > 1e-10 else np.array([1.0, 0.0, 0.0])
        a2 -= a1 * np.dot(a2, a1)
    a2 /= np.linalg.norm(a2)
    angle2 = rng.uniform(0, np.deg2rad(max_deg2))
    half2 = angle2 / 2
    q2 = np.array([np.cos(half2), a2[0] * np.sin(half2),
                   a2[1] * np.sin(half2), a2[2] * np.sin(half2)])

    return quat_multiply(q2, q1)  # R2 * R1


def build_target_pose(
    pos: np.ndarray, home_quat: np.ndarray, rng: np.random.RandomState | None = None,
) -> Pose:
    """构建目标位姿，支持多种姿态随机化模式。

    Modes:
      "fixed"       — 保持 home 姿态不变
      "single_axis" — 绕随机轴旋转 ≤ ROT_MAX_DEG
      "multi_axis"  — 绕两个独立轴旋转 (≤ ROT_AXIS1_DEG, ≤ ROT_AXIS2_DEG)
      "full_so3"    — SO(3) 均匀采样（IK 可达率会降低）
    """
    quat = home_quat
    if rng is None:
        return Pose(p=pos, q=quat)

    if ROT_MODE == "full_so3":
        quat = random_quat_full_so3(rng)
    elif ROT_MODE == "multi_axis":
        delta_q = random_quat_multi_axis(rng, ROT_AXIS1_DEG, ROT_AXIS2_DEG)
        quat = quat_multiply(delta_q, home_quat)
    elif ROT_MODE == "single_axis" and ROT_MAX_DEG > 0:
        quat = quat_multiply(random_quat_within_angle(rng, ROT_MAX_DEG), home_quat)
    # else: "fixed" — keep home quat

    return Pose(p=pos, q=quat)


def interpolate_waypoints(path: np.ndarray, max_step: float = INTERP_MAX_STEP_RAD) -> np.ndarray:
    """对稀疏关节路径线性插值，每步关节变化 ≤ max_step rad。"""
    if len(path) <= 1:
        return path
    dense = [path[0]]
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        n = int(np.ceil(float(np.max(np.abs(b - a))) / max_step))
        for k in range(1, n + 1):
            dense.append(a + (k / n) * (b - a))
    return np.array(dense, dtype=np.float64)

# ═══════════════════════════════════════════════ 仿真执行


def execute_dense_path(
    sim: SimRobotInterface, dense: np.ndarray, viewer: sapien.Viewer | None = None,
) -> bool:
    """执行已插值的稠密关节路径，(N,7) arm-only。hand 保持不变。"""
    assert dense.ndim == 2 and dense.shape[1] == 7
    hand = sim.get_full_qpos()[7:]
    for wp in dense:
        if viewer is not None and viewer.closed:
            return False
        sim.robot.balance_passive_force()
        sim.robot.apply_action(np.concatenate([wp, hand]))
        sim._step_physics(n=PHYSICS_STEPS_PER_WP)
        if viewer is not None:
            sim.scene.update_render()
            viewer.render()
    return True


def settle_at_target(
    sim: SimRobotInterface, target_arm: np.ndarray, hand_qpos: np.ndarray,
    max_iter: int = 30, converge_threshold_rad: float = np.deg2rad(0.05),
) -> float:
    """闭环收敛：迭代 PD 控制直到关节误差 < converge_threshold_rad。

    对应真机 arm.reset(wait=True) 的阻塞等待行为 — 不停留在固定轮数，
    而是持续驱动直到实际关节角收敛到目标值。

    Returns: final max joint error (rad)
    """
    for _ in range(max_iter):
        sim.robot.balance_passive_force()
        sim.robot.apply_action(np.concatenate([target_arm, hand_qpos]))
        sim._step_physics(n=PHYSICS_STEPS_PER_WP)
        current = sim.get_full_qpos()[:7]
        err = float(np.max(np.abs(current - target_arm)))
        if err < converge_threshold_rad:
            return err
    # Max iterations reached — return current error
    current = sim.get_full_qpos()[:7]
    return float(np.max(np.abs(current - target_arm)))


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
        err = float(np.max(np.abs(current[:len(target_full_qpos)] - target_full_qpos)))
        if err < converge_threshold_rad:
            return err
    current = sim.robot.get_qpos()
    return float(np.max(np.abs(current[:len(target_full_qpos)] - target_full_qpos)))


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
    joint_path = np.array([current_qpos + (k / (n - 1)) * (home_qpos - current_qpos)
                           for k in range(n)])

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

    # Step A: Lift Z to at least desk_safe level
    safe_z = max(collision_config.desk_safe_z + 0.05, current_pose.p[2] + _DIRECT_LIFT_Z_M)
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
    home_path = np.array([current + (k / (n - 1)) * (home_qpos - current)
                          for k in range(n)])
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
    planner: XArm7MotionPlanner, path: np.ndarray, goal: np.ndarray,
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
    sapien_fk_delta_mm: float     # SAPIEN vs Pinocchio FK 最大偏差
    tip_eef_local: np.ndarray     # (5,3) EEF 局部坐标系下的指尖偏移


def check_fingertips(sim: SimRobotInterface) -> FingertipCheck:
    """验证指尖位置：SAPIEN 物理 vs Pinocchio FK 一致性 + EEF 局部偏移。"""
    robot = sim.robot
    full_qpos = sim.get_full_qpos()
    names = robot.fingertip_link_names

    tips_sapien = robot.get_link_poses(names)  # (5,7) [x,y,z,w,x,y,z]
    tips_fk = robot.forward_kinematics(full_qpos, target_link_names=names)

    max_delta = float(max(np.linalg.norm(tips_sapien[i, :3] - tips_fk[i, :3])
                          for i in range(len(names))))

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
        final_qpos = sim.get_full_qpos()[:7]
        err = float(np.rad2deg(np.max(np.abs(final_qpos - home_qpos))))
        return {"success": err < _RESIDUAL_ERROR_MAX_DEG, "phase1_completed": False,
                "phase2_executed": False, "lift_used": False, "final_err_deg": err,
                "final_pos_err_mm": 0.0}

    # Already at home?
    if float(np.max(np.abs(current_qpos - home_qpos))) < _HOME_JOINT_THRESHOLD_RAD:
        print("  [return_to_home_sim] Already at home")
        return {"success": True, "phase1_completed": False, "phase2_executed": False,
                "lift_used": False, "final_err_deg": 0.0, "final_pos_err_mm": 0.0}

    # ── Pre-check: fingertip Z desk clearance ──
    # Check actual fingertip positions (FK), not just EEF position.
    # Hand extends 7.6cm below EEF (pinky tip) — EEF Z check alone misses hand-desk collision.
    start_env_ok = True
    start_safe, start_z, start_name = check_hand_desk_clearance(planner, current_qpos)
    if not start_safe:
        start_env_ok = False
        print(f"  [return_to_home_sim] {start_name} z={start_z:.3f}m < desk+margin, "
              f"forcing safety lift")
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
                    print(f"  [return_to_home_sim] Joint path to home: {name} dips below desk "
                          f"(α={alpha:.2f}, z={z:.3f}m), forcing safety lift")
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
            print(f"  [return_to_home_sim] Phase 1 path dips below desk "
                  f"(fingertip_z_min={min_z:.3f}m, waypoint {viol_idx}), falling back")
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
                err = settle_at_target(sim, result.qpos_path[-1], hand_qpos,
                                       converge_threshold_rad=_PHASE1_CONVERGE_THRESHOLD_RAD)
                phase1_completed = err < _PHASE1_CONVERGE_THRESHOLD_RAD
                print(f"  [return_to_home_sim] Phase 1: {'converged' if phase1_completed else 'timeout'} "
                      f"(err={np.rad2deg(err):.2f}deg, src={result.source})")
            else:
                print(f"  [return_to_home_sim] Phase 1: execution interrupted")
    elif not start_env_ok:
        print(f"  [return_to_home_sim] Phase 1 SKIPPED: desk collision risk detected in pre-check")
    else:
        reason = result.reason if (result is not None and hasattr(result, 'reason')) else "planner error"
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

    final_qpos = sim.get_full_qpos()[:7]
    err_rad = float(np.max(np.abs(final_qpos - home_qpos)))
    err_deg = float(np.rad2deg(err_rad))
    final_eef = planner.compute_eef_pose_world(final_qpos)
    pos_err_mm = float(np.linalg.norm(final_eef.p - home_eef.p)) * 1000

    if err_rad > np.deg2rad(_RESIDUAL_ERROR_MAX_DEG):
        print(f"  [return_to_home_sim] INCOMPLETE: residual error {err_deg:.1f}° > {_RESIDUAL_ERROR_MAX_DEG}°")
        return {"success": False, "phase1_completed": phase1_completed,
                "phase2_executed": phase2_executed, "lift_used": not phase1_completed,
                "final_err_deg": err_deg, "final_pos_err_mm": pos_err_mm}

    print(f"  [return_to_home_sim] CONVERGED: err={err_deg:.2f}deg  pos={pos_err_mm:.1f}mm")
    return {"success": True, "phase1_completed": phase1_completed,
            "phase2_executed": phase2_executed, "lift_used": not phase1_completed,
            "final_err_deg": err_deg, "final_pos_err_mm": pos_err_mm}


# ═══════════════════════════════════════════════ 桌面碰撞检测（FK 手部几何 Z 检测）

# 不使用 MPlib 全局点云（污染 IK solver）。改用 Pinocchio FK 直接计算
# 五指指尖的世界坐标，检查最低指尖 Z 是否高于桌面。比 EEF Z 检测更精确。
# SAPIEN 物理桌面仍提供物理碰撞反馈。

# 指尖 link ID 信息现在由 CollisionConfig 统一管理。


def _min_fingertip_z(planner: XArm7MotionPlanner, qpos: np.ndarray) -> tuple[float, str]:
    """计算给定 arm 构型下五指指尖的最低 world Z 坐标（委托给 planner.desk_safety）。"""
    if planner.desk_safety is not None:
        return planner.desk_safety.min_fingertip_z(qpos)
    # Fallback (should not happen when collision config is set)
    return float("inf"), ""


def check_eef_desk_clearance(
    planner: XArm7MotionPlanner, qpos: np.ndarray,
) -> tuple[bool, float]:
    """检查给定构型下 EEF 是否在桌面之上安全高度（委托给 CollisionConfig）。"""
    eef_pose = planner.compute_eef_pose_world(qpos)
    eef_z = float(eef_pose.p[2])
    return eef_z >= DESK_SAFE_Z, eef_z


def check_hand_desk_clearance(
    planner: XArm7MotionPlanner, qpos: np.ndarray,
) -> tuple[bool, float, str]:
    """检查给定构型下手指是否在桌面之上（委托给 planner.desk_safety）。"""
    if planner.desk_safety is not None:
        return planner.desk_safety.check_hand_desk_clearance(qpos)
    # Fallback: EEF-level check
    eef_pose = planner.compute_eef_pose_world(qpos)
    eef_z = float(eef_pose.p[2])
    return eef_z >= DESK_SAFE_Z, eef_z, "eef_only"


def check_path_desk_safety(
    planner: XArm7MotionPlanner, path: np.ndarray, step_rad: float = 0.05,
) -> tuple[bool, float, int]:
    """FK 手指桌面安全检查：沿关节路径密集采样（委托给 planner.desk_safety）。

    Returns: (safe, min_fingertip_z, first_violation_index)
    """
    if planner.desk_safety is not None:
        return planner.desk_safety.check_path_desk_safety(path, step_rad)
    # Fallback: simple EEF Z check (coarse)
    fingertip_threshold = collision_config.fingertip_threshold
    path_arm = path[:, :7] if path.ndim == 2 and path.shape[1] > 7 else path
    min_z = float("inf")
    for i in range(len(path_arm)):
        eef_z = float(planner.compute_eef_pose_world(path_arm[i]).p[2])
        if eef_z < min_z:
            min_z = eef_z
        if eef_z <= fingertip_threshold:
            return False, min_z, i
    return True, min_z, -1


# ═══════════════════════════════════════════════ Sim FK 碰撞检测（支持手部随机化）

# planner FK（固定手型）用于规划阶段的快速预检，sim FK（实际手型）
# 用于执行后的精确碰撞检测。当 RANDOMIZE_HAND=True 时，路径规划使用
# 固定手型的 planner FK 做预检（保守估计），sim FK 做最终确认。


def _min_fingertip_z_sim(sim: SimRobotInterface) -> tuple[float, str]:
    """使用 sim 的 SAPIEN Pinocchio FK 计算五指指尖最低 world Z。

    与 _min_fingertip_z(planner, qpos) 不同，此函数使用 sim 的实际手部
    关节角（支持手部随机化），而非 collision URDF 的固定手型。
    """
    full_qpos = sim.get_full_qpos()  # (19,) arm(7) + hand(12)
    names = sim.robot.fingertip_link_names
    tips = sim.robot.forward_kinematics(full_qpos, target_link_names=names)  # (5,7)
    min_z = float("inf")
    min_name = ""
    for i, name in enumerate(names):
        z = float(tips[i, 2])
        if z < min_z:
            min_z = z
            min_name = name.split("right_hand_")[-1]
    return min_z, min_name


_SIM_FK_EPSILON = 0.001  # FK 浮点比较容差（1mm）


def check_hand_desk_clearance_sim(sim: SimRobotInterface) -> tuple[bool, float, str]:
    """Sim FK 版本的手指桌面碰撞检查（支持手部随机化）。

    使用 sim 的实际手部关节角计算指尖位置，比 planner FK 更精确。
    """
    min_z, min_name = _min_fingertip_z_sim(sim)
    safe = min_z > collision_config.fingertip_threshold - _SIM_FK_EPSILON
    return safe, min_z, min_name


def check_path_desk_safety_sim(
    sim: SimRobotInterface,
    arm_path: np.ndarray,
    step_rad: float = 0.05,
) -> tuple[bool, float, int]:
    """Sim FK 版本的路径桌面安全检查（支持手部随机化）。

    对于 arm-only 路径 (N,7)，用当前 sim 手部关节角拼接完整 qpos，
    通过 SAPIEN Pinocchio FK 计算每个采样点的指尖位置。

    Returns: (safe, min_fingertip_z, first_violation_index)
    """
    if arm_path.ndim != 2 or arm_path.shape[1] > 7:
        arm_path = arm_path[:, :7]

    hand_qpos = sim.get_full_qpos()[7:].copy()
    fingertip_threshold = collision_config.fingertip_threshold - _SIM_FK_EPSILON
    names = sim.robot.fingertip_link_names

    min_z = float("inf")
    min_name = ""

    for i in range(len(arm_path) - 1):
        a, b = arm_path[i], arm_path[i + 1]
        dist = float(np.max(np.abs(b - a)))
        n = max(1, int(np.ceil(dist / step_rad)))
        for k in range(n + 1):
            alpha = k / max(n, 1)
            q = a + alpha * (b - a)
            full = np.concatenate([q, hand_qpos])
            tips = sim.robot.forward_kinematics(full, target_link_names=names)
            for tip_idx, name in enumerate(names):
                z = float(tips[tip_idx, 2])
                if z < min_z:
                    min_z = z
                    min_name = name.split("right_hand_")[-1]
                if z <= fingertip_threshold:
                    # Compute EEF Z for diagnostic
                    eef = sim.robot.forward_kinematics(full, target_link_names=["custom_eef_link"])
                    eef_z = float(eef[0, 2])
                    print(f"  [desk_safety_sim] {name.split('right_hand_')[-1]}={z:.3f}m "
                          f"≤ {fingertip_threshold:.3f}m "
                          f"(segment {i}, α={alpha:.2f}, EEF_z={eef_z:.3f}m)")
                    return False, min_z, i
    return True, min_z, -1


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
    q[0] = np.deg2rad(rng.uniform(tb_min, tb_max))          # thumb_bend
    ts_min, ts_max = cfg["thumb_spread"]
    # thumb_rota1: spread/opposition; rota2: fine rotation
    if grasp_type == "open":
        q[1] = np.deg2rad(rng.uniform(ts_min, ts_max))       # spread out
        q[2] = np.deg2rad(rng.uniform(-15, 15))              # neutral
    elif grasp_type == "pinch":
        q[1] = np.deg2rad(rng.uniform(ts_min, ts_max))       # oppose to index
        q[2] = np.deg2rad(rng.uniform(-10, 10))
    else:
        q[1] = np.deg2rad(rng.uniform(ts_min, ts_max))
        q[2] = np.deg2rad(rng.uniform(-20, 20))
    # Add noise to thumb
    q[0] += np.deg2rad(rng.uniform(-noise, noise))
    q[1] += np.deg2rad(rng.uniform(-noise*0.5, noise*0.5))

    # ── 四指 (index/mid/ring/pinky) 协同弯曲 ──
    # 基关节弯曲度：共享均值 + 各指独立偏差
    fc_min, fc_max = cfg["finger_curl"]
    base_curl_deg = rng.uniform(fc_min, fc_max)

    # 四指基关节索引：(index_bend=3, mid_j1=6, ring_j1=8, pinky_j1=10)
    # 注意：index 有 bend_joint(3) + j1(4) + j2(5) 三个关节
    #       mid/ring/pinky 各有 j1 + j2 两个关节
    base_indices = [3, 4, 6, 8, 10]     # 基关节
    tip_indices = [5, 7, 9, 11]          # 对应尖关节 (index_j2, mid_j2, ring_j2, pinky_j2)

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
    safe_z = max(DESK_SAFE_Z + 0.02, target_eef.p[2] + collision_config.hand_extension_below_eef + collision_config.hand_safe_margin + 0.03)
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

    desc = f"safe_descent: above_z={safe_z:.3f}m → target_z={target_eef.p[2]:.3f}m, " \
           f"wp={len(full_path)} ({len(stage1_result.qpos_path)}+{len(stage2_result.qpos_path)})"
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

    print(f"  [{label}] {np.round(current_eef.p, 3)} → {np.round(target_eef.p, 3)}  "
          f"dist={dist:.3f}m{joint_info}")

    t0 = time.perf_counter()
    result = planner.plan_path(target_eef, current_qpos)
    if not result.success or result.qpos_path is None:
        print(f"  [{label}] PLAN FAILED: {result.reason}")
        return False

    r = result.report
    path = result.qpos_path
    print(f"  [{label}] plan: src={result.source}  wp={r.get('num_waypoints','?')}  "
          f"len={r.get('joint_path_length',0):.2f}rad  t={time.perf_counter()-t0:.3f}s")

    # ── 桌面安全预检查（planner FK，固定手型保守估计）──
    # 如果手部已随机化 (check_desk_with_sim=True)，planner FK 预检可能偏保守
    # （collision URDF 手型可能比实际手型更低），保留作快速筛选但不作为最终依据。
    desk_safe, min_z, viol_idx = check_path_desk_safety(planner, path)
    used_safe_descent = False

    if not desk_safe:
        # ── 对低 Z 目标尝试安全下降（模拟真机遥操作）──
        if target_eef.p[2] < DESK_SAFE_Z:
            print(f"  [{label}] 🔽 screw path unsafe → trying safe descent")
            descent_path, descent_desc = plan_safe_descent(planner, sim, target_eef, viewer)
            if descent_path is not None:
                print(f"  [{label}] ✅ safe descent: {descent_desc}")
                path = descent_path
                used_safe_descent = True
                desk_safe = True  # override: path is now safe

        if not desk_safe and not check_desk_with_sim:
            # Planner FK says unsafe AND we're not doing sim FK override → reject
            wp0_eef = planner.compute_eef_pose_world(path[0])
            fingertip_threshold = collision_config.fingertip_threshold
            print(f"  [{label}] ❌ DESK COLLISION (planner FK): fingertip z_min={min_z:.3f}m "
                  f"< safe={fingertip_threshold:.3f}m (waypoint {viol_idx})")
            print(f"  [{label}]     path shape={path.shape}  wp0_eef_z={wp0_eef.p[2]:.3f}m  "
                  f"DESK_SAFE_Z(EEF)={DESK_SAFE_Z:.3f}m")
            return False
        elif not desk_safe:
            # Planner FK says unsafe but we'll verify with sim FK after execution
            print(f"  [{label}] ⚠️  planner FK预检不安全 → 执行后用 sim FK 确认")

    if not used_safe_descent:
        if joint_goal is not None:
            path = append_joint_goal(planner, path, joint_goal)

    dense = interpolate_waypoints(path)
    print(f"  [{label}] exec {len(dense)} wp")

    if not used_safe_descent:
        # Normal execution (safe_descent already executed the path)
        tips_before = check_fingertips(sim)
        hand_qpos = sim.get_full_qpos()[7:]
        execute_dense_path(sim, dense, viewer)
        settle_at_target(sim, dense[-1, :7], hand_qpos)

    if used_safe_descent:
        # safe_descent already executed the path; need tips_before for drift check
        tips_before = check_fingertips(sim)
    # ── Sim FK 桌面碰撞确认（用于手部随机化场景）──
    desk_ok = True
    if check_desk_with_sim:
        # 对执行后的实际路径做 sim FK 碰撞检测（考虑实际手型）
        sim_desk_safe, sim_z, sim_viol = check_path_desk_safety_sim(sim, path)
        if not sim_desk_safe:
            fingertip_threshold = collision_config.fingertip_threshold
            print(f"  [{label}] ❌ DESK COLLISION (sim FK): fingertip z_min={sim_z:.3f}m "
                  f"< safe={fingertip_threshold:.3f}m (waypoint {sim_viol})")
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

    print(f"  [{label}] pos_err={pos_err:.4f}m  rot_err={np.rad2deg(rot_err):.2f}deg{joint_str}  "
          f"tip_s2fk={tips_after.sapien_fk_delta_mm:.2f}mm tip_drift={tip_drift_mm:.2f}mm  "
          f"{'desk!' if not desk_ok else ''}  "
          f"[{'OK' if ok else 'FAIL'}]")
    return ok

# ═══════════════════════════════════════════════ IK 测试


@dataclass
class IKStats:
    ok: int
    pos_errs: list[float]
    rot_errs: list[float]


def _run_ik_loop(
    planner: XArm7MotionPlanner, targets: list[Pose], init_qpos: np.ndarray,
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
    return IKStats(ok=ok, pos_errs=pos_errs, rot_errs=rot_errs)


def print_ik_stats(label: str, stats: IKStats) -> None:
    pos = np.array(stats.pos_errs)
    rot = np.array(stats.rot_errs)
    valid = ~np.isnan(pos)
    pos_v = pos[valid] if valid.any() else np.array([np.inf])
    rot_v = rot[valid] if valid.any() else np.array([np.inf])
    total = len(pos)
    rate = f"{stats.ok}/{total} ({100*stats.ok/total:.1f}%)" if total else "0"
    print(f"  [{label}] success_rate={rate}  "
          f"pos_err: avg={np.mean(pos_v)*1000:.1f}mm  max={np.max(pos_v)*1000:.1f}mm  "
          f"rot_err: avg={np.rad2deg(np.mean(rot_v)):.2f}deg  max={np.rad2deg(np.max(rot_v)):.2f}deg")


def ik_test(
    planner: XArm7MotionPlanner, sim: SimRobotInterface, home_qpos: np.ndarray,
    num_samples: int = 50, rng: np.random.RandomState | None = None,
) -> dict[str, IKStats]:
    """独立 IK 测试：对随机 EEF 位姿调 solve_ik()，FK 往返验证。"""
    if rng is None:
        rng = np.random.RandomState(SEED)

    home_eef = planner.compute_eef_pose_world(home_qpos)
    positions = np.column_stack([
        rng.uniform(*SAMPLE_X, num_samples),
        rng.uniform(*SAMPLE_Y, num_samples),
        sample_z_biased(rng, num_samples) if num_samples > 20
        else rng.uniform(*SAMPLE_Z, num_samples),
    ])
    targets = [build_target_pose(positions[i], home_eef.q, rng) for i in range(num_samples)]

    return {
        "fresh":   _run_ik_loop(planner, targets, home_qpos,            chained=False),
        "chained": _run_ik_loop(planner, targets, home_qpos,            chained=True),
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
    z = np.where(choices == 0, z_low,
         np.where(choices == 1, z_mid, z_high))
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
    return create_viewer(sim.scene, sapien.Pose(
        [0.784212, 0.0267081, 0.630188],
        [0.00493842, -0.232841, 0.00108951, 0.972502],
    ))

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
        print(f"Sweep: margin={margin:.2f}m  desk_safe_z={desk_safe_z:.3f}m  "
              f"fingertip_threshold={cfg.fingertip_threshold:.3f}m")
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
    print(f"{'margin(m)':<10} {'desk_safe_z(m)':<15} {'collisions':<12} "
          f"{'reachable':<12} {'rate':<8}")
    print(f"{'-'*10} {'-'*15} {'-'*12} {'-'*12} {'-'*8}")
    for margin, desk_safe_z, stat in results:
        collisions_str = f"{stat['desk_collisions']}/{stat['total']}"
        reachable_str = f"{stat['reachable']}/{stat['total']}"
        rate_str = f"{stat['rate']:.0f}%"
        print(f"{margin:<10.2f} {desk_safe_z:<15.3f} {collisions_str:<12} "
              f"{reachable_str:<12} {rate_str:<8}")
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
    from dexmani_real.planning import FingertipDeskSafety

    rng = np.random.RandomState(seed)

    sim = SimRobotInterface(SimRobotConfig(headless=headless))
    if not sim.connect():
        return {"desk_collisions": 0, "reachable": 0, "total": 0, "rate": 0.0}

    root_pose = sim.robot.model.get_root_pose()
    planner = XArm7MotionPlanner(
        XArm7PlannerConfig(
            urdf_path=str(ASSET_DIR / "robots" / "xhand" / "xarm7_xhand_collision.urdf"),
            srdf_path=str(ASSET_DIR / "robots" / "xhand" / "xarm7_xhand_collision_mplib.srdf"),
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
    target_positions = np.column_stack([
        rng.uniform(*TEST_DESK_X_RANGE, num_samples),
        rng.uniform(*TEST_DESK_Y_RANGE, num_samples),
        rng.uniform(*TEST_DESK_Z_RANGE, num_samples),
    ])

    reachable = 0
    desk_collisions = 0

    for i, pos in enumerate(target_positions):
        # Randomize hand
        hand_qpos, _grasp_type = hand_grasp_pose(rng)
        target_full = np.concatenate([sim.get_full_qpos()[:7], hand_qpos])
        smooth_drive_to_target(sim, target_full, None, max_iter=30,
                               label=f"sweep_{i+1}")

        target_pose = build_target_pose(pos, home_quat, rng)
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

        # Sim FK desk check
        _, min_z, _ = check_hand_desk_clearance_sim(sim)
        fingertip_threshold = collision_cfg.fingertip_threshold
        if min_z <= fingertip_threshold - _SIM_FK_EPSILON:
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
    sim: SimRobotInterface, rng: np.random.RandomState, n_objects: int = 3,
) -> list:
    """Spawn random objects (box/cylinder) on the table surface.

    Objects are placed within the table footprint at random heights (5-15 cm).
    Returns list of created actors for later cleanup.
    """
    actors = []
    for _ in range(n_objects):
        obj_type = rng.choice(["box", "cylinder"])
        # Random position on table surface
        x = rng.uniform(TABLE_CENTER[0] - TABLE_HALF[0] + 0.1,
                        TABLE_CENTER[0] + TABLE_HALF[0] - 0.1)
        y = rng.uniform(TABLE_CENTER[1] - TABLE_HALF[1] + 0.1,
                        TABLE_CENTER[1] + TABLE_HALF[1] - 0.1)
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
                radius=radius, half_length=height / 2,
                material=(0.3, 0.3, 0.6),  # bluish
            )
            builder.add_cylinder_collision(
                radius=radius, half_length=height / 2,
            )

        actor = builder.build_kinematic(name=f"table_obj_{_}")
        actor.set_pose(sapien.Pose(p=[x, y, z]))
        actors.append(actor)

    if actors:
        print(f"  Spawned {len(actors)} table-top objects (heights 5-15 cm)")

    return actors


# ═══════════════════════════════════════════════ 主流程


def main():
    import argparse
    p = argparse.ArgumentParser(description="Workspace 路径规划仿真测试")
    p.add_argument("--comprehensive", action="store_true",
                   help=f"全面模式：{COMPREHENSIVE_NUM_SAMPLES} 采样点 "
                        f"(含分层采样) + {COMPREHENSIVE_NUM_IK} IK 采样")
    p.add_argument("--test-desk", action="store_true",
                   help="桌面碰撞专项测试：聚焦 Z∈[0.02,0.18] 区域，关闭 EEF 预过滤，"
                        "手部随机化，用 sim FK 精确检测手指穿桌")
    p.add_argument("--randomize-hand", action="store_true",
                   help=f"手部关节随机化 (±{HAND_RANDOM_RANGE_DEG}°)，测试不同手型下的桌面碰撞")
    p.add_argument("--optimize-z-min", action="store_true",
                   help="网格搜索最优 hand_safe_margin（需与 --test-desk 联用），"
                        "输出安全性与可达性 trade-off 表格")
    p.add_argument("--with-objects", action="store_true",
                   help="在桌面上添加随机物体（box/cylinder，高度 5-15cm），"
                        "测试物体场景下的碰撞安全性")
    p.add_argument("--ci", action="store_true", help="CI 快速模式 (5 samples, headless)")
    p.add_argument("--headless", action="store_true", help="无头模式（无 GUI）")
    p.add_argument("--seed", type=int, default=SEED, help="随机种子")
    args = p.parse_args()

    num_samples = 5 if args.ci else (COMPREHENSIVE_NUM_SAMPLES if args.comprehensive else NUM_SAMPLES)
    num_ik = 20 if args.ci else (COMPREHENSIVE_NUM_IK if args.comprehensive else NUM_IK_SAMPLES)
    seed = args.seed
    headless = HEADLESS or args.headless or args.ci

    # 桌面碰撞专项模式
    test_desk = args.test_desk
    randomize_hand = args.randomize_hand or test_desk  # test-desk 强制手部随机化

    # --optimize-z-min: run grid search and exit
    if args.optimize_z_min:
        if not test_desk:
            print("NOTE: --optimize-z-min implies --test-desk mode")
            test_desk = True
        sweep_z_min(
            sim_config={},
            headless=headless or True,  # optimize always headless for speed
            num_samples=TEST_DESK_NUM_SAMPLES,
            seed=seed,
            with_objects=args.with_objects,
        )
        return

    if test_desk:
        num_samples = TEST_DESK_NUM_SAMPLES
        num_ik = min(num_ik, 30)  # IK 测试在 desk 模式下不重要

    rng = np.random.RandomState(seed)

    sim = SimRobotInterface(SimRobotConfig(headless=headless))
    if not sim.connect():
        print(f"ERROR: connect failed: {sim.last_error_message}", file=sys.stderr)
        return

    # Planner: 坐标系对齐 sim root_pose
    root_pose = sim.robot.model.get_root_pose()
    planner = XArm7MotionPlanner(
        XArm7PlannerConfig(
            urdf_path=str(ASSET_DIR / "robots" / "xhand" / "xarm7_xhand_collision.urdf"),
            srdf_path=str(ASSET_DIR / "robots" / "xhand" / "xarm7_xhand_collision_mplib.srdf"),
            base_pose_world=Pose(p=np.array(root_pose.p), q=np.array(root_pose.q)),
            # Workspace bounds for _lift_eef_z_safe (mirrors RobotInterface)
            workspace_bounds=np.array([[0.0, 0.75], [-0.5, 0.5], [0.0, 0.6]], dtype=np.float64),
            # Unified collision config (geometric FK desk safety)
            collision=collision_config,
        ),
        planning_profile=PlanningProfile(
            max_waypoint_delta_deg=360.0,
            max_ik_delta_deg=(180,) * 7,
            max_pose_error_rot_rad=np.deg2rad(5.0),
            num_random_ik_seeds=30,
            rrt_time_limit=2.0,
            num_rrt_attempts=2,
            random_seed=seed,
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
    assert float(np.linalg.norm(home_eef.p - np.array(sim.robot.eef_home_pose.p))) < 1e-6

    # ── 桌面碰撞保护 ──
    # 不使用 MPlib 全局点云（会污染 IK/planning，使成功率从 100% 跌到 53%）。
    # 改用几何 Z 检测 + SAPIEN 物理桌面（与真机 WorkspaceSafety 方式一致）。
    # planner.planning_profile.check_env_collision 保持 False（默认）。
    # DESK_SAFE_Z = 0.106m 是 EEF 级保守预过滤，真实碰撞检测使用指尖 FK。
    # Apply collision_config override for --with-objects mode
    if args.with_objects and test_desk:
        collision_config.table_object_max_height = 0.15  # max object height for safety calc
        print(f"  Objects mode: table_object_max_height={collision_config.table_object_max_height:.2f}m, "
              f"eef_safe_z_with_objects={collision_config.eef_safe_z_with_objects:.3f}m")

    reject_below = REJECT_BELOW_DESK_Z and not test_desk  # test-desk 模式关闭预过滤
    print(f"Desk safety: SAPIEN physics + Z guard (DESK_SAFE_Z={DESK_SAFE_Z:.3f}m, "
          f"fingertip_safe={collision_config.fingertip_threshold:.3f}m)")
    print(f"  Table: X∈[{TABLE_CENTER[0]-TABLE_HALF[0]:.1f}, {TABLE_CENTER[0]+TABLE_HALF[0]:.1f}]  "
          f"Y∈[{TABLE_CENTER[1]-TABLE_HALF[1]:.1f}, {TABLE_CENTER[1]+TABLE_HALF[1]:.1f}]  "
          f"z_top={TABLE_TOP_Z:.1f}")
    print(f"  Hand extension: pinky_tip={collision_config.hand_extension_below_eef*1000:.0f}mm below EEF (home hand)")
    if test_desk:
        print(f"  Mode: DESK TEST — EEF pre-filter OFF, hand randomized ±{HAND_RANDOM_RANGE_DEG}°, sim FK verify")
    elif randomize_hand:
        print(f"  Hand: randomized ±{HAND_RANDOM_RANGE_DEG}°")

    viewer = _setup_viewer(sim) if not headless else None

    # 复位到 home
    sim.reset()
    for _ in range(5):
        sim._step_physics(n=10)
    if viewer:
        for _ in range(5):
            sim.scene.update_render()
            viewer.render()

    # ── 桌面物体交互测试（--with-objects）──
    table_objects = []
    if args.with_objects and test_desk:
        table_objects = _spawn_table_objects(sim, rng, n_objects=3)
        # Reset to let objects settle in physics
        for _ in range(5):
            sim._step_physics(n=10)
        print()

    # ── 路径规划测试 ──
    if test_desk:
        # 桌面碰撞专项：Z 集中在 [0.02, 0.18]，X/Y 全 workspace
        target_positions = np.column_stack([
            rng.uniform(*TEST_DESK_X_RANGE, num_samples),
            rng.uniform(*TEST_DESK_Y_RANGE, num_samples),
            rng.uniform(*TEST_DESK_Z_RANGE, num_samples),
        ])
        z_low_pct = 100 * np.mean(target_positions[:, 2] <= Z_LOW_RANGE[1])
        sampling_mode = f"desk-test Z∈{TEST_DESK_Z_RANGE} (Z-low {z_low_pct:.0f}%)"
    elif args.comprehensive:
        # 分层加权采样：低 Z 区权重 ×4 分配更多采样点
        region_weights = np.array([r[4] for r in STRATIFIED_REGIONS])
        region_weights /= region_weights.sum()
        region_counts = np.round(region_weights * num_samples).astype(int)
        # 修正 rounding 导致的偏差
        diff = num_samples - region_counts.sum()
        if diff > 0:
            idx = np.argmax(region_weights)
            region_counts[idx] += diff
        elif diff < 0:
            for i in np.argsort(region_counts)[::-1]:
                if region_counts[i] > 1 and diff < 0:
                    region_counts[i] -= 1
                    diff += 1
                if diff == 0:
                    break

        positions_list = []
        for i, (label, xr, yr, zr, _) in enumerate(STRATIFIED_REGIONS):
            n = region_counts[i]
            if n > 0:
                positions_list.append(np.column_stack([
                    rng.uniform(*xr, n),
                    rng.uniform(*yr, n),
                    rng.uniform(*zr, n),
                ]))
        target_positions = np.vstack(positions_list)
        rng.shuffle(target_positions)
        z_low_pct = 100 * np.mean((target_positions[:, 2] >= Z_LOW_RANGE[0]) &
                                   (target_positions[:, 2] <= Z_LOW_RANGE[1]))
        sampling_mode = (f"stratified × {len(STRATIFIED_REGIONS)} regions "
                         f"(Z-low {z_low_pct:.0f}%)")
    else:
        # 默认模式：Z 偏置采样（低区权重 ×{Z_LOW_WEIGHT}）
        target_positions = np.column_stack([
            rng.uniform(*SAMPLE_X, num_samples),
            rng.uniform(*SAMPLE_Y, num_samples),
            sample_z_biased(rng, num_samples),
        ])
        z_low_pct = 100 * np.mean((target_positions[:, 2] >= Z_LOW_RANGE[0]) &
                                   (target_positions[:, 2] <= Z_LOW_RANGE[1]))
        sampling_mode = f"Z-biased (low weight ×{Z_LOW_WEIGHT:.0f}, Z-low {z_low_pct:.0f}%)"

    rot_info = f"  rot={ROT_MODE}"
    if ROT_MODE == "multi_axis":
        rot_info += f" (axis1≤{ROT_AXIS1_DEG}° axis2≤{ROT_AXIS2_DEG}°)"
    elif ROT_MODE == "single_axis":
        rot_info += f" (≤{ROT_MAX_DEG}°)"
    print(f"{'='*60}")
    print(f"Workspace 路径规划 — {num_samples} 采样点 ({sampling_mode}) + return_home")
    print(f"  home EEF: {np.round(home_eef.p, 4)}  quat={np.round(home_quat, 4)}")
    print(f"  空间: x{SAMPLE_X} y{SAMPLE_Y} z{SAMPLE_Z}{rot_info}  seed={seed}")
    if test_desk:
        print(f"  手部: 随机化 ±{HAND_RANDOM_RANGE_DEG}°  sim FK 验证")
    print(f"{'='*60}\n")

    t_start = time.perf_counter()
    ok_count = 0
    skipped_desk = 0
    desk_collisions = 0
    start_unsafe_lifts = 0  # 起点手部不安全，需抬升

    for i, pos in enumerate(target_positions):
        if viewer and viewer.closed:
            break

        # ── 桌面安全预过滤 ──
        # DESK_SAFE_Z 是 EEF 级保守估计：home 手型下最低指尖比 EEF 低 7.6cm。
        # 如果手部已随机化，手型可能更"收紧"（手指比 home 更靠上），
        # EEF Z < DESK_SAFE_Z 不意味着一定碰撞 — 由 sim FK 做最终判定。
        # test-desk 模式关闭预过滤以允许测试低 Z 目标。
        if reject_below and pos[2] < DESK_SAFE_Z:
            skipped_desk += 1
            print(f"  [{i+1:2d}/{num_samples}] ⏭️  SKIP: z={pos[2]:.3f}m < DESK_SAFE_Z={DESK_SAFE_Z:.3f}m")
            continue

        # ── 手部随机化（抓取姿态，PD 平滑驱动）──
        hand_note = ""
        hand_qpos = None
        grasp_type_name = ""
        if randomize_hand:
            hand_qpos, grasp_type_name = hand_grasp_pose(rng)
            target_full = np.concatenate([sim.get_full_qpos()[:7], hand_qpos])
            # 通过 PD 控制器平滑驱动手部到随机目标位姿
            smooth_drive_to_target(sim, target_full, viewer, max_iter=40,
                                   label=f"grasp_{i+1}")
            # Log hand fingertip Z and extension (BOTH from sim FK, same reference frame)
            hand_safe, hand_min_z, hand_min_name = check_hand_desk_clearance_sim(sim)
            eef_sim = sim.robot.forward_kinematics(sim.get_full_qpos(), target_link_names=["custom_eef_link"])
            eef_z_sim = float(eef_sim[0, 2])
            hand_ext = eef_z_sim - hand_min_z
            hand_note = f" hand:{hand_min_name}={hand_min_z:.3f}m ext={hand_ext*1000:.0f}mm"
            print(f"  [{i+1:2d}/{num_samples}] 🖐️  {grasp_type_name}: {hand_min_name}={hand_min_z:.3f}m "
                  f"(ext={hand_ext*1000:.0f}mm) "
                  f"({'OK' if hand_safe else '⚠️BELOW DESK'})")

        # ── 起点安全检查（sim FK，考虑随机化手型）──
        start_collision_type = ""  # "" | "start" | "path"
        if randomize_hand:
            start_safe, start_z, start_name = check_hand_desk_clearance_sim(sim)
            if not start_safe:
                # 多层抬升策略：
                # Stage 1: 保持当前随机手型，抬升 EEF
                # Stage 2: 如果仍不安全，手部回 home（延伸更小），抬升 EEF
                start_unsafe_lifts += 1
                current_arm = sim.get_full_qpos()[:7]
                current_pose = planner.compute_eef_pose_world(current_arm)
                lift_success = False

                for stage in range(2):
                    if stage == 1:
                        # Stage 2: reset hand to home (7.6cm extension, minimal) via smooth PD
                        home_hand = np.zeros(NUM_HAND_JOINTS)
                        target_full = np.concatenate([sim.get_full_qpos()[:7], home_hand])
                        smooth_drive_to_target(sim, target_full, viewer, max_iter=30,
                                               label=f"hand_home_{i+1}")
                        hand_qpos = home_hand  # update for later use

                    # Compute lift target: raise EEF Z to clear desk + hand extension
                    current_arm = sim.get_full_qpos()[:7]
                    current_pose = planner.compute_eef_pose_world(current_arm)
                    _, hand_z, _ = check_hand_desk_clearance_sim(sim)
                    eef_sim = sim.robot.forward_kinematics(sim.get_full_qpos(), target_link_names=["custom_eef_link"])
                    eef_z = float(eef_sim[0, 2])
                    hand_ext = eef_z - hand_z
                    # Need EEF high enough: eef_z > TABLE_TOP_Z + margin + hand_ext + extra
                    target_eef_z = collision_config.fingertip_threshold + hand_ext + 0.03
                    target_eef_z = max(target_eef_z, current_pose.p[2] + _DIRECT_LIFT_Z_M)
                    target_eef_z = min(target_eef_z, 0.55)

                    lift_pose = Pose(
                        p=np.array([current_pose.p[0], current_pose.p[1], target_eef_z], dtype=np.float64),
                        q=current_pose.q.copy(),
                    )
                    stage_tag = f"S{stage+1}" + ("(home_hand)" if stage == 1 else "")

                    # Try teleop IK first (fast, for small lifts)
                    lift_ik = planner.solve_teleop_ik(lift_pose, current_arm, current_arm)
                    if lift_ik.success and lift_ik.qpos is not None:
                        # Smooth PD-driven lift (arm + hand)
                        target_full = np.concatenate([lift_ik.qpos, hand_qpos])
                        smooth_drive_to_target(sim, target_full, viewer, max_iter=40,
                                               label=f"lift_{i+1}_{stage_tag}")
                        _, lift_z, lift_name = check_hand_desk_clearance_sim(sim)
                        lifted_eef_z = float(planner.compute_eef_pose_world(lift_ik.qpos).p[2])
                        if lift_z > collision_config.fingertip_threshold:
                            print(f"  [{i+1:2d}/{num_samples}] ⬆️  start unsafe → lift {stage_tag} OK: "
                                  f"EEF z={lifted_eef_z:.3f}m, {lift_name}={lift_z:.3f}m")
                            lift_success = True
                            break
                        else:
                            print(f"  [{i+1:2d}/{num_samples}] ⬆️  start unsafe → lift {stage_tag}: "
                                  f"EEF z={lifted_eef_z:.3f}m, {lift_name}={lift_z:.3f}m (STILL LOW, "
                                  f"need ext<{lifted_eef_z - collision_config.table_z_world - collision_config.hand_safe_margin:.3f}m)")
                        continue  # try next stage if still low
                    else:
                        # Teleop IK rejected (jump too large) → try plan_path for multi-waypoint lift
                        try:
                            plan_result = planner.plan_path(lift_pose, current_arm)
                        except RuntimeError:
                            plan_result = None
                        if plan_result is not None and plan_result.success and plan_result.qpos_path is not None:
                            # Execute planned path waypoints
                            hand = hand_qpos
                            for wp in plan_result.qpos_path:
                                if viewer is not None and viewer.closed:
                                    break
                                sim.robot.balance_passive_force()
                                sim.robot.apply_action(np.concatenate([wp, hand]))
                                sim._step_physics(n=PHYSICS_STEPS_PER_WP)
                                if viewer is not None:
                                    sim.scene.update_render()
                                    viewer.render()
                            # Settle
                            settle_at_target(sim, plan_result.qpos_path[-1], hand,
                                             converge_threshold_rad=_PHASE1_CONVERGE_THRESHOLD_RAD)
                            _, lift_z, lift_name = check_hand_desk_clearance_sim(sim)
                            lifted_eef_z = float(planner.compute_eef_pose_world(plan_result.qpos_path[-1]).p[2])
                            if lift_z > collision_config.fingertip_threshold:
                                print(f"  [{i+1:2d}/{num_samples}] ⬆️  start unsafe → plan_path lift {stage_tag} OK: "
                                      f"EEF z={lifted_eef_z:.3f}m, {lift_name}={lift_z:.3f}m")
                                lift_success = True
                                break
                            else:
                                print(f"  [{i+1:2d}/{num_samples}] ⬆️  start unsafe → plan_path lift {stage_tag}: "
                                      f"EEF z={lifted_eef_z:.3f}m, {lift_name}={lift_z:.3f}m (STILL LOW)")
                                continue
                        else:
                            reason = plan_result.reason if plan_result else "plan error"
                            print(f"  [{i+1:2d}/{num_samples}] ⬆️  start unsafe → lift {stage_tag} "
                                  f"plan_path FAILED: {reason}")
                            continue  # try next stage

                if not lift_success:
                    print(f"  [{i+1:2d}/{num_samples}] ⬆️  ALL lift stages FAILED — continuing with unsafe start")

        target_pose = build_target_pose(pos, home_quat, rng)
        marker = place_marker(sim.scene, pos) if viewer else None
        if viewer:
            sim.scene.update_render()
            viewer.render()

        ok = plan_and_execute(planner, sim, target_pose, viewer, f"{i+1:2d}/{num_samples}",
                              check_desk_with_sim=randomize_hand)
        if ok:
            ok_count += 1
        elif randomize_hand:
            # Determine failure type: start vs path collision
            _, final_min_z, final_min_name = check_hand_desk_clearance_sim(sim)
            fingertip_threshold = collision_config.fingertip_threshold
            if final_min_z <= fingertip_threshold:
                desk_collisions += 1
        if marker:
            sim.scene.remove_actor(marker)

    # ── return_home (mirrors RobotInterface.return_to_home()) ──
    home_report = {}
    if not viewer or not viewer.closed:
        print(f"\n{'='*60}")
        print("return_home (mirrors RobotInterface 两阶段归航)")
        print("=" * 60)
        # Workspace bounds for safety lift (matches RobotInterface config)
        ws_bounds = np.array([[0.0, 0.75], [-0.5, 0.5], [0.0, 0.6]], dtype=np.float64)
        home_report = return_to_home_sim(planner, sim, home_qpos, ws_bounds, viewer)
        home_ok = home_report.get("success", False)
    else:
        home_ok = False

    # ── IK 独立测试 ──
    if num_ik > 0:
        ik_report = ik_test(planner, sim, home_qpos, num_samples=num_ik, rng=rng)
        print(f"\n{'='*60}\nIK solve_ik 独立测试 ({num_ik} 采样)")
        print_ik_stats("fresh  ", ik_report["fresh"])
        print_ik_stats("chained", ik_report["chained"])

    print(f"\n{'='*60}")
    collision_str = f"  desk_collisions={desk_collisions}" if randomize_hand else ""
    lift_str = f"  start_lifts={start_unsafe_lifts}" if randomize_hand else ""
    print(f"targets_ok={ok_count}/{num_samples}  skipped_desk={skipped_desk}{collision_str}{lift_str}  "
          f"return_home={'OK' if home_ok else 'FAIL'}  "
          f"total={time.perf_counter()-t_start:.1f}s")
    if home_report:
        print(f"  home phase1={home_report.get('phase1_completed')}  "
              f"phase2={home_report.get('phase2_executed')}  "
              f"lift={home_report.get('lift_used')}  "
              f"err={home_report.get('final_err_deg', '?'):.2f}deg  "
              f"pos={home_report.get('final_pos_err_mm', '?'):.1f}mm")

    if viewer and not viewer.closed:
        print("\nClose viewer to exit...")
        while not viewer.closed:
            sim.scene.update_render()
            viewer.render()

    sim.disconnect()
    print("Done.")


if __name__ == "__main__":
    main()
