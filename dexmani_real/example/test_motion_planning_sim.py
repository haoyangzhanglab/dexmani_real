#!/usr/bin/env python3
"""Workspace 随机采样 — 测试 Planner 路径规划（仿真模型 + SAPIEN 可视化）。

用法:
    conda activate real
    python test_planner_path.py

修改顶部常量控制测试。

测试流程:
    1. 随机采样 N 个 EEF 位姿（位置+姿态），plan_path → 执行 → 验证
    2. return_home: plan_path(home_eef) + 关节归位（含碰撞检测）
    3. IK 独立测试：solve_ik() 成功率 + FK 往返误差
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import sapien.core as sapien

from dexmani_real import ASSET_DIR
from dexmani_real.planner import PlanningProfile, Pose, XArm7MotionPlanner, XArm7PlannerConfig
from dexmani_real.robot.model import SimRobotConfig, SimRobotInterface
from dexmani_real.robot.model.constructor import add_light, create_viewer

# ═══════════════════════════════════════════════ 配置

NUM_SAMPLES = 10
NUM_IK_SAMPLES = 50
HEADLESS = False
SEED = 123

SAMPLE_X = (0.3, 0.7)
SAMPLE_Y = (-0.4, 0.4)
SAMPLE_Z = (0.0, 0.5)

PHYSICS_STEPS_PER_WP = 20
INTERP_MAX_STEP_RAD = np.deg2rad(2.0)
MARKER_RADIUS = 0.015
RANDOM_ROT_DEG = 30.0

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


def build_target_pose(
    pos: np.ndarray, home_quat: np.ndarray, rng: np.random.RandomState | None = None,
) -> Pose:
    """在 home 姿态附近随机旋转，构建目标位姿。"""
    quat = home_quat
    if RANDOM_ROT_DEG > 0 and rng is not None:
        quat = quat_multiply(random_quat_within_angle(rng, RANDOM_ROT_DEG), home_quat)
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


def settle_at_target(sim: SimRobotInterface, target_arm: np.ndarray, hand_qpos: np.ndarray) -> None:
    """在最终 arm 目标上稳定 PD 控制器，消除 tracking error。"""
    for _ in range(3):
        sim.robot.balance_passive_force()
        sim.robot.apply_action(np.concatenate([target_arm, hand_qpos]))
        sim._step_physics(n=PHYSICS_STEPS_PER_WP)


def append_joint_goal(
    planner: XArm7MotionPlanner, path: np.ndarray, goal: np.ndarray,
) -> np.ndarray:
    """在路径末尾追加 goal_qpos，插值并逐点检查自碰撞。不安全则返回原 path。"""
    full = np.vstack([path, goal])
    dense = interpolate_waypoints(full)
    if planner.planning_profile.check_self_collision:
        if any(planner.has_self_collision(q) for q in dense):
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

# ═══════════════════════════════════════════════ 路径规划测试


def plan_and_execute(
    planner: XArm7MotionPlanner,
    sim: SimRobotInterface,
    target_eef: Pose,
    viewer: sapien.Viewer | None = None,
    label: str = "",
    joint_goal: np.ndarray | None = None,
) -> bool:
    """plan_path(target_eef) → 执行 → 验证 EEF 精度 + 指尖正确性。"""
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

    if joint_goal is not None:
        path = append_joint_goal(planner, path, joint_goal)

    dense = interpolate_waypoints(path)
    print(f"  [{label}] exec {len(dense)} wp")

    tips_before = check_fingertips(sim)
    hand_qpos = sim.get_full_qpos()[7:]
    execute_dense_path(sim, dense, viewer)
    settle_at_target(sim, dense[-1, :7], hand_qpos)

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
    ok = ok and tip_ok

    print(f"  [{label}] pos_err={pos_err:.4f}m  rot_err={np.rad2deg(rot_err):.2f}deg{joint_str}  "
          f"tip_s2fk={tips_after.sapien_fk_delta_mm:.2f}mm tip_drift={tip_drift_mm:.2f}mm  "
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
        rng.uniform(*SAMPLE_Z, num_samples),
    ])
    targets = [build_target_pose(positions[i], home_eef.q, rng) for i in range(num_samples)]

    return {
        "fresh":   _run_ik_loop(planner, targets, home_qpos,            chained=False),
        "chained": _run_ik_loop(planner, targets, home_qpos,            chained=True),
    }

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

# ═══════════════════════════════════════════════ 主流程


def main():
    rng = np.random.RandomState(SEED)

    sim = SimRobotInterface(SimRobotConfig(headless=HEADLESS))
    if not sim.connect():
        raise RuntimeError(f"connect failed: {sim.last_error_message}")

    # Planner: 坐标系对齐 sim root_pose
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
            random_seed=SEED,
        ),
    )

    home_qpos = sim.config.arm_home_qpos.copy()
    home_eef = planner.compute_eef_pose_world(home_qpos)
    home_quat = home_eef.q.copy()
    assert float(np.linalg.norm(home_eef.p - np.array(sim.robot.eef_home_pose.p))) < 1e-6

    viewer = _setup_viewer(sim)

    # 复位到 home
    sim.reset()
    for _ in range(5):
        sim._step_physics(n=10)
    if viewer:
        for _ in range(5):
            sim.scene.update_render()
            viewer.render()

    # ── 路径规划测试 ──
    target_positions = np.column_stack([
        rng.uniform(*SAMPLE_X, NUM_SAMPLES),
        rng.uniform(*SAMPLE_Y, NUM_SAMPLES),
        rng.uniform(*SAMPLE_Z, NUM_SAMPLES),
    ])

    rot_info = f"  rot≤{RANDOM_ROT_DEG}deg" if RANDOM_ROT_DEG > 0 else "  rot=fixed"
    print(f"{'='*60}")
    print(f"Workspace 路径规划 — {NUM_SAMPLES} 采样点 + return_home")
    print(f"  home EEF: {np.round(home_eef.p, 4)}  quat={np.round(home_quat, 4)}")
    print(f"  空间: x{SAMPLE_X} y{SAMPLE_Y} z{SAMPLE_Z}{rot_info}  seed={SEED}")
    print(f"{'='*60}\n")

    t_start = time.perf_counter()
    ok_count = 0

    for i, pos in enumerate(target_positions):
        if viewer and viewer.closed:
            break

        target_pose = build_target_pose(pos, home_quat, rng)
        marker = place_marker(sim.scene, pos) if viewer else None
        if viewer:
            sim.scene.update_render()
            viewer.render()

        ok = plan_and_execute(planner, sim, target_pose, viewer, f"{i+1:2d}/{NUM_SAMPLES}")
        if ok:
            ok_count += 1
        if marker:
            sim.scene.remove_actor(marker)

    # ── return_home ──
    home_ok = False
    if not viewer or not viewer.closed:
        print(f"\n{'='*60}\nreturn_home")
        home_ok = plan_and_execute(planner, sim, home_eef, viewer, "return_home", joint_goal=home_qpos)

    # ── IK 独立测试 ──
    if NUM_IK_SAMPLES > 0:
        ik_report = ik_test(planner, sim, home_qpos, num_samples=NUM_IK_SAMPLES, rng=rng)
        print(f"\n{'='*60}\nIK solve_ik 独立测试 ({NUM_IK_SAMPLES} 采样)")
        print_ik_stats("fresh  ", ik_report["fresh"])
        print_ik_stats("chained", ik_report["chained"])

    print(f"\n{'='*60}")
    print(f"targets_ok={ok_count}/{NUM_SAMPLES}  return_home={'OK' if home_ok else 'FAIL'}  "
          f"total={time.perf_counter()-t_start:.1f}s")

    if viewer and not viewer.closed:
        print("\nClose viewer to exit...")
        while not viewer.closed:
            sim.scene.update_render()
            viewer.render()

    sim.disconnect()
    print("Done.")


if __name__ == "__main__":
    main()
