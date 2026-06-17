#!/usr/bin/env python3
"""真机运动规划测试 — 路径规划 + IK 验证 + 硬件执行。

用法:
    source /home/zhy/anaconda3/etc/profile.d/conda.sh && conda activate real
    python test_motion_planning_real.py

测试流程:
    0. 连接硬件 + Pre-Flight 安全检查
    1. solve_ik() 批量 FK 往返验证 (不移动)
    2. solve_teleop_ik() 批量遥操作 IK 验证 (不移动)
    3. plan_path() 规划成功率 (不执行)
    4. 安全 waypoints 路径规划 + 硬件执行 + return_home

如需修改测试参数，编辑本文件顶部的 DEFAULT_* 常量。
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass

import numpy as np

from dexmani_real import ASSET_DIR
from dexmani_real.planning import (
    PlanningProfile,
    Pose,
    TeleopProfile,
    XArm7MotionPlanner,
    XArm7PlannerConfig,
)
from dexmani_real.robot.xarm7 import XArm7, XArm7Config

# ═══════════════════════════════════════════════ 配置

DEFAULT_NUM_SAMPLES = 8
DEFAULT_NUM_IK = 50
DEFAULT_SEED = 42
RANDOM_ROT_DEG = 30.0

SAMPLE_X = (0.28, 0.70)
SAMPLE_Y = (-0.40, 0.40)
SAMPLE_Z = (0.02, 0.55)

SIMULATE_PATHS = False  # Set True to validate paths in SAPIEN before real-arm exec

SAFE_WAYPOINTS = [
    {"pos": (0.45,  0.00, 0.33), "label": "center"},
    {"pos": (0.30,  0.00, 0.35), "label": "near_center"},
    {"pos": (0.65,  0.00, 0.30), "label": "far_center"},
    {"pos": (0.45,  0.30, 0.30), "label": "right"},
    {"pos": (0.45, -0.30, 0.30), "label": "left"},
    {"pos": (0.45,  0.00, 0.53), "label": "high_center"},
    {"pos": (0.45,  0.00, 0.13), "label": "low_center"},
    {"pos": (0.60,  0.25, 0.45), "label": "far_right_high"},
    {"pos": (0.60, -0.25, 0.15), "label": "far_left_low"},
    {"pos": (0.30,  0.20, 0.15), "label": "near_right_low"},
]

INTERP_MAX_STEP_RAD = np.deg2rad(1.0)
ARM_DT = 1.0 / 30.0
CONVERGE_THRESHOLD_RAD = np.deg2rad(3.0)       # Phase 1 收敛阈值
HOME_AT_THRESHOLD_RAD = np.deg2rad(1.0)        # 视为已归位的关节偏差
PHASE2_MIN_DELTA_RAD = np.deg2rad(0.5)         # Phase 2 跳过的关节偏差下限
PHASE2_MAX_STEP_RAD = np.deg2rad(2.0)          # Phase 2 关节空间最大步长
POS_ERR_THRESHOLD_M = 0.03                     # waypoint 终点位置误差阈值 (m)


# ═══════════════════════════════════════════════ 数学工具


def angular_dist_rad(q1: np.ndarray, q2: np.ndarray) -> float:
    return float(2 * np.arccos(np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0)))


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def random_quat_within_angle(rng: np.random.RandomState, max_deg: float) -> np.ndarray:
    axis = rng.randn(3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(0, np.deg2rad(max_deg))
    half = angle / 2
    return np.array([np.cos(half), axis[0] * np.sin(half),
                     axis[1] * np.sin(half), axis[2] * np.sin(half)])


def build_target_pose(
    pos: np.ndarray, home_quat: np.ndarray, rng: np.random.RandomState | None = None,
) -> Pose:
    quat = home_quat
    if RANDOM_ROT_DEG > 0 and rng is not None:
        quat = quat_multiply(random_quat_within_angle(rng, RANDOM_ROT_DEG), home_quat)
    return Pose(p=pos, q=quat)


def interpolate_waypoints(path: np.ndarray, max_step: float = INTERP_MAX_STEP_RAD) -> np.ndarray:
    if len(path) <= 1:
        return path
    dense = [path[0]]
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        n = int(np.ceil(float(np.max(np.abs(b - a))) / max_step))
        for k in range(1, n + 1):
            dense.append(a + (k / n) * (b - a))
    return np.array(dense, dtype=np.float64)


# ═══════════════════════════════════════════════ 数据结构


@dataclass
class IKStats:
    ok: int
    total: int
    pos_errs_mm: list[float]
    rot_errs_deg: list[float]
    max_dq_deg: list[float]


@dataclass
class PathPlanStats:
    ok: int
    total: int
    pos_errs_mm: list[float]
    rot_errs_deg: list[float]
    path_lengths_rad: list[float]
    num_waypoints: list[int]
    reasons: list[str]


def ik_stats_empty() -> IKStats:
    return IKStats(ok=0, total=0, pos_errs_mm=[], rot_errs_deg=[], max_dq_deg=[])


# ═══════════════════════════════════════════════ Planner


def create_planner(seed: int = DEFAULT_SEED) -> XArm7MotionPlanner:
    urdf_path = str(ASSET_DIR / "robots" / "xhand" / "xarm7_xhand_collision.urdf")
    srdf_path = str(ASSET_DIR / "robots" / "xhand" / "xarm7_xhand_collision_mplib.srdf")

    return XArm7MotionPlanner(
        XArm7PlannerConfig(
            urdf_path=urdf_path,
            srdf_path=srdf_path,
            # 30° yaw offset: robot base → world (quat: w=cos(15°), x=0, y=0, z=sin(15°))
            base_pose_world=Pose(p=[0,0,0], q=[np.cos(np.pi/12), 0, 0, np.sin(np.pi/12)]),
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
            teleop_dt=0.02,
            max_ik_jump_deg=(30, 30, 30, 30, 45, 45, 60),
            max_pose_error_pos_m=0.01,
            max_pose_error_rot_rad=np.deg2rad(5.0),
        ),
    )


# ═══════════════════════════════════════════════ Pre-Flight


def preflight_check(arm: XArm7) -> bool:
    """Pre-Flight 安全检查 (hardware-safety.md)。"""
    print("Pre-Flight 检查:")
    all_ok = True

    ok = arm.is_connected()
    print(f"  [{'OK' if ok else 'FAIL'}] arm 连接")
    all_ok = all_ok and ok

    ok = not arm.is_error()
    print(f"  [{'OK' if ok else 'FAIL'}] arm 无错误 ({arm.last_error_message})" if not ok else f"  [OK] arm 无错误")
    all_ok = all_ok and ok

    state = arm.get_state()
    qpos = np.asarray(state["qpos"], dtype=np.float64)
    ok = bool(np.all(np.isfinite(qpos)))
    print(f"  [{'OK' if ok else 'FAIL'}] 关节角度有效")
    all_ok = all_ok and ok

    if ok:
        config = arm.config
        in_range = bool(np.all(qpos >= config.qpos_min) and np.all(qpos <= config.qpos_max))
        print(f"  [{'OK' if in_range else 'FAIL'}] 关节在限位内")
        all_ok = all_ok and in_range

    print(f"  结果: {'全部通过' if all_ok else '检查失败'}\n")
    return all_ok


# ═══════════════════════════════════════════════ Test 1: solve_ik()


def test_solve_ik(
    planner: XArm7MotionPlanner,
    home_qpos: np.ndarray,
    num_samples: int,
    rng: np.random.RandomState,
) -> dict[str, IKStats]:
    """solve_ik() 批量 FK 往返验证 (不移动硬件)。"""
    home_eef = planner.compute_eef_pose_world(home_qpos)
    positions = np.column_stack([
        rng.uniform(*SAMPLE_X, num_samples),
        rng.uniform(*SAMPLE_Y, num_samples),
        rng.uniform(*SAMPLE_Z, num_samples),
    ])
    targets = [build_target_pose(positions[i], home_eef.q, rng) for i in range(num_samples)]

    fresh = ik_stats_empty()
    fresh.total = num_samples
    for target in targets:
        r = planner.solve_ik(target, home_qpos)
        if r.success and r.qpos is not None:
            fresh.ok += 1
            eef = planner.compute_eef_pose_world(r.qpos)
            fresh.pos_errs_mm.append(float(np.linalg.norm(eef.p - target.p)) * 1000)
            fresh.rot_errs_deg.append(np.rad2deg(angular_dist_rad(eef.q, target.q)))
            fresh.max_dq_deg.append(float(np.max(np.abs(np.rad2deg(r.qpos - home_qpos)))))

    chained = ik_stats_empty()
    chained.total = num_samples
    seed = home_qpos.copy()
    for target in targets:
        r = planner.solve_ik(target, seed)
        if r.success and r.qpos is not None:
            chained.ok += 1
            seed = r.qpos.copy()
            eef = planner.compute_eef_pose_world(r.qpos)
            chained.pos_errs_mm.append(float(np.linalg.norm(eef.p - target.p)) * 1000)
            chained.rot_errs_deg.append(np.rad2deg(angular_dist_rad(eef.q, target.q)))
            chained.max_dq_deg.append(float(np.max(np.abs(np.rad2deg(r.qpos - home_qpos)))))

    return {"fresh": fresh, "chained": chained}


# ═══════════════════════════════════════════════ Test 2: solve_teleop_ik()


def test_solve_teleop_ik(
    planner: XArm7MotionPlanner,
    home_qpos: np.ndarray,
    num_samples: int,
    rng: np.random.RandomState,
) -> IKStats:
    """solve_teleop_ik() 批量随机游走验证 (不移动硬件)。"""
    stats = ik_stats_empty()
    home_eef = planner.compute_eef_pose_world(home_qpos)

    positions = [home_eef.p.copy()]
    current = home_eef.p.copy()
    for _ in range(num_samples):
        step = rng.uniform(-0.05, 0.05, 3)
        new_pos = current + step
        new_pos = np.clip(new_pos, [SAMPLE_X[0], SAMPLE_Y[0], SAMPLE_Z[0]],
                          [SAMPLE_X[1], SAMPLE_Y[1], SAMPLE_Z[1]])
        positions.append(new_pos)
        current = new_pos

    stats.total = num_samples
    prev_qpos = home_qpos.copy()
    prev_cmd = home_qpos.copy()

    for pos in positions[1:]:
        target = build_target_pose(pos, home_eef.q)
        r = planner.solve_teleop_ik(target, prev_qpos, prev_cmd)

        if r.success and r.qpos is not None:
            stats.ok += 1
            eef = planner.compute_eef_pose_world(r.qpos)
            stats.pos_errs_mm.append(float(np.linalg.norm(eef.p - target.p)) * 1000)
            stats.rot_errs_deg.append(np.rad2deg(angular_dist_rad(eef.q, target.q)))
            stats.max_dq_deg.append(float(np.max(np.abs(np.rad2deg(r.qpos - prev_cmd)))))
            prev_qpos = r.qpos.copy()
            prev_cmd = r.qpos.copy()

    return stats


# ═══════════════════════════════════════════════ Test 3: plan_path() (plan only)


def test_plan_path(
    planner: XArm7MotionPlanner,
    home_qpos: np.ndarray,
    num_samples: int,
    rng: np.random.RandomState,
) -> PathPlanStats:
    """plan_path() 规划成功率 (不执行)。"""
    stats = PathPlanStats(
        ok=0, total=num_samples, pos_errs_mm=[], rot_errs_deg=[],
        path_lengths_rad=[], num_waypoints=[], reasons=[],
    )

    home_eef = planner.compute_eef_pose_world(home_qpos)
    positions = np.column_stack([
        rng.uniform(*SAMPLE_X, num_samples),
        rng.uniform(*SAMPLE_Y, num_samples),
        rng.uniform(*SAMPLE_Z, num_samples),
    ])

    for i in range(num_samples):
        target = build_target_pose(positions[i], home_eef.q, rng)
        r = planner.plan_path(target, home_qpos)

        if r.success and r.qpos_path is not None:
            stats.ok += 1
            final_eef = planner.compute_eef_pose_world(r.qpos_path[-1])
            stats.pos_errs_mm.append(float(np.linalg.norm(final_eef.p - target.p)) * 1000)
            stats.rot_errs_deg.append(np.rad2deg(angular_dist_rad(final_eef.q, target.q)))
            stats.path_lengths_rad.append(r.report.get("joint_path_length", 0.0))
            stats.num_waypoints.append(r.report.get("num_waypoints", 0))
        else:
            stats.reasons.append(r.reason or "unknown")

    return stats


# ═══════════════════════════════════════════════ 打印


def _arr_stats(values: list[float], fmt: str = ".1f") -> str:
    if not values:
        return "N/A"
    arr = np.array(values)
    return f"avg={np.mean(arr):{fmt}}  max={np.max(arr):{fmt}}"


def print_ik_stats(label: str, stats: IKStats):
    rate = f"{stats.ok}/{stats.total} ({100*stats.ok/max(stats.total,1):.1f}%)"
    print(f"  [{label}] success={rate}")
    if stats.pos_errs_mm:
        print(f"    pos_err_mm:  {_arr_stats(stats.pos_errs_mm, '.1f')}")
        print(f"    rot_err_deg: {_arr_stats(stats.rot_errs_deg, '.2f')}")
        print(f"    max_dq_deg:  {_arr_stats(stats.max_dq_deg, '.1f')}")


def print_path_stats(stats: PathPlanStats):
    rate = f"{stats.ok}/{stats.total} ({100*stats.ok/max(stats.total,1):.1f}%)"
    print(f"  plan_path  success={rate}")
    if stats.pos_errs_mm:
        print(f"    pos_err_mm:     {_arr_stats(stats.pos_errs_mm, '.1f')}")
        print(f"    rot_err_deg:    {_arr_stats(stats.rot_errs_deg, '.2f')}")
        print(f"    path_len_rad:   {_arr_stats(stats.path_lengths_rad, '.2f')}")
        print(f"    num_waypoints:  {_arr_stats([float(n) for n in stats.num_waypoints], '.0f')}")
    if stats.reasons:
        for reason, count in Counter(stats.reasons).most_common(3):
            print(f"    fail: {reason[:80]} x{count}")


# ═══════════════════════════════════════════════ 首次运动递增验证


def incremental_motion_check(arm: XArm7, planner: XArm7MotionPlanner, home_qpos: np.ndarray) -> bool:
    """Pre-Flight 首次运动递增验证 (hardware-safety.md).

    Step 1: 发送当前关节角度 (stay in place)
    Step 2: 发送小幅度关节运动 (+2°)
    Step 3: 复位到 home
    """
    print("首次运动递增验证...")

    current_qpos = np.asarray(arm.get_state()["qpos"], dtype=np.float64)

    # Step 1: stay in place
    print("  Step 1: stay in place...")
    if not arm.send_action(current_qpos):
        print("    FAILED: send_action(stay) failed")
        return False
    time.sleep(0.3)
    qpos_after = np.asarray(arm.get_state()["qpos"], dtype=np.float64)
    max_delta = float(np.max(np.abs(qpos_after - current_qpos)))
    if max_delta > np.deg2rad(2.0):
        print(f"    FAILED: arm moved {np.rad2deg(max_delta):.2f}deg (stay in place)")
        return False
    print(f"    OK (delta={np.rad2deg(max_delta):.2f}deg)")

    # Step 2: small motion (+2°)
    print("  Step 2: small motion (+2deg)...")
    small_target = current_qpos + np.deg2rad(2.0)
    if not arm.send_action(small_target):
        print("    FAILED: send_action(small) failed")
        return False
    time.sleep(0.3)
    qpos_after = np.asarray(arm.get_state()["qpos"], dtype=np.float64)
    delta = np.rad2deg(np.abs(qpos_after - current_qpos))
    if np.all(delta < 0.5):
        print(f"    FAILED: arm barely moved (delta={np.round(delta, 1)}deg)")
        return False
    print(f"    OK (delta={np.round(delta, 1)}deg)")

    # Step 3: return to home
    print("  Step 3: return to home...")
    arm.reset(home_qpos)
    time.sleep(0.5)
    qpos_after = np.asarray(arm.get_state()["qpos"], dtype=np.float64)
    home_err = float(np.max(np.abs(qpos_after - home_qpos)))
    if home_err > CONVERGE_THRESHOLD_RAD:
        print(f"    FAILED: home error {np.rad2deg(home_err):.2f}deg")
        return False
    print(f"    OK (home_err={np.rad2deg(home_err):.2f}deg)")

    print("  递增运动验证通过\n")
    return True


# ═══════════════════════════════════════════════ Waypoint IK 预验证


def validate_waypoint_ik(
    planner: XArm7MotionPlanner,
    home_qpos: np.ndarray,
    waypoints: list[dict],
) -> list[dict]:
    """过滤 IK 可达的 waypoints。"""
    home_eef = planner.compute_eef_pose_world(home_qpos)
    valid = []
    for wp in waypoints:
        pos = np.array(wp["pos"], dtype=np.float64)
        target = Pose(p=pos, q=home_eef.q)
        r = planner.solve_ik(target, home_qpos)
        if r.success and r.qpos is not None:
            valid.append(wp)
        else:
            print(f"  skip [{wp['label']}]: IK unreachable ({r.reason or 'unknown'})")
    return valid


# ═══════════════════════════════════════════════ Test 4: 硬件执行


def execute_path_on_arm(arm: XArm7, path: np.ndarray) -> tuple[bool, np.ndarray | None, dict]:
    """执行稠密关节路径，返回 (ok, actual_qpos_at_samples, tracking_stats)。

    每 send_action 后抽样读取实际关节角（每 5 步读一次，避免过多 SDK 调用），
    与规划路径对比计算跟踪精度。
    """
    dense = interpolate_waypoints(path)
    n = len(dense)
    sample_step = max(1, n // 20)  # 至少 20 个采样点
    actual_samples = []
    planned_samples = []

    for i, wp in enumerate(dense):
        if arm.is_error():
            print("    arm error during exec")
            return False, None, {}
        if not arm.send_action(wp):
            print(f"    send_action failed: {arm.last_error_message}")
            return False, None, {}
        time.sleep(ARM_DT)

        if i % sample_step == 0 or i == n - 1:
            actual_qpos = np.asarray(arm.get_state()["qpos"], dtype=np.float64)
            if np.all(np.isfinite(actual_qpos)):
                actual_samples.append(actual_qpos)
                planned_samples.append(wp)

    time.sleep(0.5)

    if not actual_samples:
        return True, None, {}

    actual_arr = np.array(actual_samples)      # (S, 7)
    planned_arr = np.array(planned_samples)     # (S, 7)
    joint_errs_deg = np.rad2deg(np.abs(actual_arr - planned_arr))  # (S, 7)
    per_joint_mean = joint_errs_deg.mean(axis=0)
    per_joint_max = joint_errs_deg.max(axis=0)
    overall_mean = float(joint_errs_deg.mean())
    overall_max = float(joint_errs_deg.max())
    worst_joint = int(np.argmax(per_joint_max))

    return True, actual_arr, {
        "joint_mean_deg": overall_mean,
        "joint_max_deg": overall_max,
        "per_joint_mean": per_joint_mean,
        "per_joint_max": per_joint_max,
        "worst_joint": worst_joint,
        "num_samples": len(actual_samples),
    }


def run_waypoint_test(
    planner: XArm7MotionPlanner,
    arm: XArm7,
    waypoint: dict,
    home_eef: Pose,
    rng: np.random.RandomState | None = None,
) -> bool:
    """单个 waypoint: plan_path → execute → verify → return_home。"""
    label = waypoint["label"]
    pos = np.array(waypoint["pos"], dtype=np.float64)
    target = build_target_pose(pos, home_eef.q, rng)

    current_qpos = np.asarray(arm.get_state()["qpos"], dtype=np.float64)
    dist = float(np.linalg.norm(target.p - planner.compute_eef_pose_world(current_qpos).p))

    rot_deg = np.rad2deg(angular_dist_rad(target.q, home_eef.q)) if rng is not None else 0.0

    rot_str = f"  rot={rot_deg:.0f}deg" if rot_deg > 1 else ""
    print(f"\n  [{label}] {np.round(current_qpos[:3], 2)} → {np.round(pos, 3)}{rot_str}  dist={dist:.3f}m")

    t0 = time.perf_counter()
    result = planner.plan_path(target, current_qpos)

    if not result.success or result.qpos_path is None:
        print(f"  [{label}] PLAN FAILED: {result.reason}")
        return False

    r = result.report
    num_waypoints = r.get("num_waypoints", "?")
    path_len = r.get("joint_path_length", 0)
    plan_t = time.perf_counter() - t0
    print(f"  [{label}] plan: src={result.source}  wp={num_waypoints}  "
          f"len={path_len:.2f}rad  t={plan_t:.3f}s")

    # 碰撞检测验证（使用 planner 内置 FCL）
    collision_report = validate_path_collisions(planner, result.qpos_path)
    if not collision_report["ok"]:
        print(f"  [{label}] COLLISION WARNING: {collision_report['errors']}")

    # SAPIEN 仿真验证（可选，较重）
    if SIMULATE_PATHS:
        sim_report = simulate_path_in_sapien(result.qpos_path, arm.config.init_qpos)
        if not sim_report["ok"]:
            print(f"  [{label}] SIM WARNING: {sim_report['errors']}")
            if sim_report["warnings"]:
                print(f"           sim warnings: {sim_report['warnings']}")

    exec_ok, _, track = execute_path_on_arm(arm, result.qpos_path)
    if not exec_ok:
        print(f"  [{label}] EXEC FAILED")
        return False

    # ── 路径跟踪精度 ──
    if track:
        jm = track["joint_mean_deg"]
        jx = track["joint_max_deg"]
        pjm = track["per_joint_mean"]
        pjx = track["per_joint_max"]
        wj = track["worst_joint"]
        ns = track["num_samples"]
        print(f"  [{label}] track ({ns} samples): "
              f"joint_mean={jm:.2f}deg  joint_max={jx:.2f}deg")
        print(f"           per_joint_mean: {np.round(pjm, 2)}")
        print(f"           per_joint_max : {np.round(pjx, 2)}  worst=J{wj}")

    # ── 终点精度 ──
    final_qpos = np.asarray(arm.get_state()["qpos"], dtype=np.float64)
    final_eef = planner.compute_eef_pose_world(final_qpos)
    pos_err = float(np.linalg.norm(final_eef.p - target.p))
    rot_err = angular_dist_rad(final_eef.q, target.q)

    max_joint_err = float(np.max(np.abs(final_qpos - result.qpos_path[-1])))
    ok = pos_err < POS_ERR_THRESHOLD_M and max_joint_err < CONVERGE_THRESHOLD_RAD

    print(f"  [{label}] final: pos_err={pos_err:.4f}m  rot_err={np.rad2deg(rot_err):.2f}deg  "
          f"joint_err={np.rad2deg(max_joint_err):.2f}deg  [{'OK' if ok else 'FAIL'}]")

    return ok


def validate_path_collisions(
    planner: XArm7MotionPlanner, qpos_path: np.ndarray
) -> dict:
    """Validate a planned joint path for self- and environment collisions.

    Returns dict with:
        ok: bool
        errors: list[str]
        checked: int — number of waypoints checked
    """
    profile = planner.planning_profile
    errors = []
    checked = 0
    for i, q in enumerate(qpos_path):
        q_7 = np.asarray(q, dtype=np.float64).ravel()[:7]
        if profile.check_self_collision and planner.has_self_collision(q_7):
            errors.append(f"waypoint {i}: self-collision")
        if profile.check_env_collision and planner.has_env_collision(q_7):
            errors.append(f"waypoint {i}: env-collision")
        checked += 1
    return {"ok": len(errors) == 0, "errors": errors, "checked": checked}


def simulate_path_in_sapien(
    qpos_path: np.ndarray, home_qpos: np.ndarray
) -> dict:
    """Validate a planned joint path in SAPIEN simulation.

    NOTE: Heavy dependency (sapien). Only call when full physics verification
    is needed.  For fast collision checking, use validate_path_collisions().

    Returns dict with:
        ok: bool
        errors: list[str]
        warnings: list[str]
    """
    import sapien.core as sapien
    from dexmani_real.simulation.constructor import setup_scene
    from dexmani_real.simulation.xarm7_xhand import XArm7_XHand

    report: dict = {"ok": True, "errors": [], "warnings": []}
    scene = None
    try:
        scene = setup_scene(time_step=1.0 / 240.0)
        robot = XArm7_XHand(
            scene,
            disable_self_collision=False,
            arm_home_qpos=np.asarray(home_qpos, dtype=np.float64).ravel()[:7],
        )
    except Exception as e:
        report["ok"] = False
        report["errors"].append(f"Sim setup failed: {e}")
        return report

    try:
        qlimits = robot.qlimits
        for i, qpos_arm7 in enumerate(qpos_path):
            full_qpos = np.concatenate([
                np.asarray(qpos_arm7, dtype=np.float64).ravel()[:7],
                np.zeros(12, dtype=np.float64),
            ])

            # Joint limit check
            if np.any(full_qpos[:7] < qlimits[:7, 0]) or np.any(full_qpos[:7] > qlimits[:7, 1]):
                report["warnings"].append(f"Waypoint {i}: joint limit violation")

            robot.set_qpos(full_qpos)
            robot.balance_passive_force()

            # Step physics to let contacts settle
            for _ in range(3):
                scene.step()

            # Check self-collision
            contacts = scene.get_contacts()
            if contacts is not None and len(contacts) > 0:
                report["errors"].append(f"Waypoint {i}: {len(contacts)} contacts detected")
                report["ok"] = False
    except Exception as e:
        report["ok"] = False
        report["errors"].append(f"Path sim error: {e}")

    return report


def _fallback_reset(arm: XArm7, home_qpos: np.ndarray) -> float:
    """紧急恢复：stop → clear_error → arm.reset(home_qpos)。

    仅在 arm.reset() 直接调用失败时使用。通过 E-stop 清空所有 pending
    命令，重置 arm 状态，再用阻塞式 set_servo_angle(wait=True) 归位。

    Returns: max joint error from home_qpos (deg)
    """
    print("  [_fallback_reset] emergency stop + clear...")
    arm.stop()
    time.sleep(0.5)
    arm.clear_error()
    time.sleep(0.3)

    print("  [_fallback_reset] resetting to home (blocking)...")
    ok = arm.reset(home_qpos)
    if not ok:
        print(f"  [_fallback_reset] reset failed ({arm.last_error_message}), retrying...")
        arm.stop()
        time.sleep(0.3)
        arm.clear_error()
        time.sleep(0.3)
        arm.reset(home_qpos)

    time.sleep(0.5)
    final = np.asarray(arm.get_state()["qpos"], dtype=np.float64)
    err = float(np.rad2deg(np.max(np.abs(final - home_qpos))))
    print(f"  [_fallback_reset] final error: {err:.2f}deg")
    return err


def safe_return_home(arm: XArm7, planner: XArm7MotionPlanner, home_qpos: np.ndarray,
                     home_eef: Pose, dt: float = ARM_DT) -> float:
    """安全归位：plan_path 碰撞预警 + 阻塞式 arm.reset() 执行。

    设计原则：
    - arm.reset() 使用 set_servo_angle(wait=True)，不经过伺服缓冲区
    - plan_path 仅做碰撞安全预警（~0.01s），不改变执行路径
    - _fallback_reset() 仅在 arm.reset() 直接失败时调用

    Returns: max joint error from home_qpos (deg)
    """
    current_qpos = np.asarray(arm.get_state()["qpos"], dtype=np.float64)

    if not np.all(np.isfinite(current_qpos)):
        return _fallback_reset(arm, home_qpos)

    # 已在 home（joint 偏差 < 1°）
    if float(np.max(np.abs(current_qpos - home_qpos))) < HOME_AT_THRESHOLD_RAD:
        return float(np.rad2deg(np.max(np.abs(current_qpos - home_qpos))))

    # 碰撞安全预警（plan_path 内部已做 FCL 碰撞检测）
    result = planner.plan_path(home_eef, current_qpos)
    if result.success:
        print(f"  [safe_return_home] plan OK (src={result.source})")
    else:
        print(f"  [safe_return_home] plan WARNING: {result.reason}")

    # 轻量清理 + 阻塞式归位（不走 servo 缓冲区，无 code=1 风险）
    arm.clear_error()
    ok = arm.reset(home_qpos)
    time.sleep(0.3)

    if not ok or arm.is_error():
        print("  [safe_return_home] direct reset failed, falling back to _fallback_reset()")
        return _fallback_reset(arm, home_qpos)

    final = np.asarray(arm.get_state()["qpos"], dtype=np.float64)
    err = float(np.rad2deg(np.max(np.abs(final - home_qpos))))

    # 残余误差微调（arm.reset 后通常 < 0.1°，此处仅为安全网）
    if err > 0.5:
        for _ in range(3):
            arm.send_action(home_qpos)
            time.sleep(0.1)
        time.sleep(0.2)
        final = np.asarray(arm.get_state()["qpos"], dtype=np.float64)
        err = float(np.rad2deg(np.max(np.abs(final - home_qpos))))

    return err


# ═══════════════════════════════════════════════ 主流程


def main():
    num_samples = DEFAULT_NUM_SAMPLES
    num_ik = DEFAULT_NUM_IK
    seed = DEFAULT_SEED

    rng = np.random.RandomState(seed)
    arm_config = XArm7Config()
    home_qpos = arm_config.init_qpos.copy()

    print("=" * 60)
    print("真机运动规划测试")
    print(f"  num_samples={num_samples}  num_ik={num_ik}  seed={seed}")
    print(f"  home_qpos: {np.round(np.rad2deg(home_qpos), 1)} deg")
    print("=" * 60)

    arm = XArm7(arm_config)
    print("\n连接硬件...")
    if not arm.connect():
        print(f"arm 连接失败: {arm.last_error_message}")
        return

    try:
        if not preflight_check(arm):
            print("Pre-Flight 检查失败，退出")
            return

        # Planner
        planner = create_planner(seed)
        home_eef = planner.compute_eef_pose_world(home_qpos)

        # 初始归位（安全归位：plan_path + 关节精调 + 碰撞检测）
        print("初始归位...")
        init_err = safe_return_home(arm, planner, home_qpos, home_eef)
        print(f"初始归位误差: {init_err:.2f}deg")
        if init_err > 5.0:
            print("归位误差过大，fallback 到 arm.reset()")
            arm.reset(home_qpos)
            time.sleep(1.0)
        print(f"home EEF: {np.round(home_eef.p, 4)}m  quat={np.round(home_eef.q, 4)}")

        # 首次运动递增验证
        if not incremental_motion_check(arm, planner, home_qpos):
            print("递增运动验证失败，退出")
            return

        # Waypoint IK 预验证
        print(f"Waypoint IK 预验证 ({len(SAFE_WAYPOINTS)} 个)...")
        safe_waypoints = validate_waypoint_ik(planner, home_qpos, SAFE_WAYPOINTS)
        if not safe_waypoints:
            print("所有 waypoints IK 不可达，退出")
            return
        print(f"  {len(safe_waypoints)}/{len(SAFE_WAYPOINTS)} 个可达\n")

        # ══ 1. solve_ik() ══
        t0 = time.perf_counter()
        print(f"\n{'='*60}")
        print(f"Test 1: solve_ik() — {num_ik} 采样 FK 往返验证")
        ik_results = test_solve_ik(planner, home_qpos, num_ik, rng)
        print_ik_stats("fresh  ", ik_results["fresh"])
        print_ik_stats("chained", ik_results["chained"])
        print(f"  time: {time.perf_counter()-t0:.1f}s")

        # ══ 2. solve_teleop_ik() ══
        t0 = time.perf_counter()
        print(f"\n{'='*60}")
        print(f"Test 2: solve_teleop_ik() — {num_ik} 步随机游走")
        teleop_stats = test_solve_teleop_ik(planner, home_qpos, num_ik, rng)
        print_ik_stats("teleop", teleop_stats)
        print(f"  time: {time.perf_counter()-t0:.1f}s")

        # ══ 3. plan_path() ══
        t0 = time.perf_counter()
        print(f"\n{'='*60}")
        print(f"Test 3: plan_path() — {num_samples} 采样 (仅规划)")
        path_stats = test_plan_path(planner, home_qpos, num_samples, rng)
        print_path_stats(path_stats)
        print(f"  time: {time.perf_counter()-t0:.1f}s")

        # ══ 4. 硬件 waypoint 执行 ══
        print(f"\n{'='*60}")
        print(f"Test 4: 硬件 waypoint 执行 — {len(safe_waypoints)} 个目标")
        print("=" * 60)

        ok_count = 0
        for wp in safe_waypoints:
            if arm.is_error():
                print("arm error, 中止")
                break
            ok = run_waypoint_test(planner, arm, wp, home_eef, rng)
            if ok:
                ok_count += 1

        print(f"\n{'='*60}")
        print(f"Waypoints: {ok_count}/{len(safe_waypoints)} OK")

        # ══ 最终归位 ══
        print(f"\n{'='*60}")
        print("最终归位...")
        final_err = safe_return_home(arm, planner, home_qpos, home_eef)
        print(f"最终归位误差: {final_err:.2f}deg")
        if final_err > 5.0:
            print("最终归位误差过大，fallback 到 arm.reset()")
            arm.reset(home_qpos)
            time.sleep(1.0)
            final_retry = np.asarray(arm.get_state()["qpos"], dtype=np.float64)
            print(f"额外 reset 后误差: {np.rad2deg(np.max(np.abs(final_retry - home_qpos))):.2f}deg")

        # ══ Summary ══
        print(f"\n{'='*60}")
        print("Summary:")
        print(f"  solve_ik       : {ik_results['fresh'].ok}/{ik_results['fresh'].total} "
              f"fresh, {ik_results['chained'].ok}/{ik_results['chained'].total} chained")
        print(f"  solve_teleop_ik: {teleop_stats.ok}/{teleop_stats.total}")
        print(f"  plan_path      : {path_stats.ok}/{path_stats.total}")
        print(f"  live_waypoints : {ok_count}/{len(safe_waypoints)}")
        print("Done.")

    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
