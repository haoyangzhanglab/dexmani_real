#!/usr/bin/env python3
"""仿真键盘遥操作 xArm7 — 通过 SimRobotInterface 在 SAPIEN 中控制虚拟机械臂。

用法:
    conda activate real
    python examples/sim/keyboard_teleop_sim.py

    # 无头模式（不创建 viewer 窗口）
    python examples/sim/keyboard_teleop_sim.py --headless

控制:
    移动 EEF (base 坐标系):
      W/S       X 前后
      A/D       Y 左右
      ↑/↓       Z 上下
      I/K       Pitch (Y 旋转)
      ←/→       Roll  (X 旋转)
      J/L       Yaw   (Z 旋转)
    Q          退出主循环
    R          return_home (规划路径回 home)
    ESC        紧急停止 + 退出

安全:
    - workspace 软墙 (到达边界拒绝移动)
    - EEF 速度限制 (替代关节空间 bottleneck scaling)
    - EEF 跟踪误差监控 (位置 + 姿态, target-vs-actual)
    - solve_teleop_ik 迭代 DLS + MPlib 位置 IK 兜底
    - 关节跟踪 divergence 监控 (cmd-vs-actual, 安全兜底)

设计参考:
    - keyboard_teleop_real.py (真机版): 按键映射、安全机制
    - vr_teleop_sim.py (仿真 VR 版): 仿真接口、路径执行
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import sapien.core as sapien

from dexmani_real import ASSET_DIR
from dexmani_real.planning import PlanningProfile, Pose, TeleopProfile, XArm7MotionPlanner, XArm7PlannerConfig
from dexmani_real.simulation import SimRobotConfig, SimRobotInterface
from dexmani_real.simulation.constructor import add_light, setup_scene
from dexmani_real.teleop.control.keyboard import GlobalKeyState
from dexmani_real.utils.rate_manager import RateManager
from scipy.spatial.transform import Rotation as R

try:
    from pynput import keyboard  # type: ignore[import-untyped]
except ImportError:
    raise ImportError("pynput is required for keyboard input. Install with: pip install pynput")

from dexmani_real.planning.path_utils import interpolate_waypoints
from dexmani_real.planning.pose_utils import angular_dist_rad, compute_pose_error, quat_multiply
from dexmani_real.simulation import execute_dense_path, settle_at_target

# ═══════════════════════════════════════════════ 配置

CTRL_HZ = 50.0
CTRL_DT = 1.0 / CTRL_HZ
DELTA_POS = 0.005  # 每次按键 EEF 平移量 (m)
DELTA_RPY = 0.02  # 每次按键 EEF 旋转量 (rad)
INTERP_MAX_STEP_RAD = np.deg2rad(2.0)
PHYSICS_STEPS_PER_TICK = 5  # 240Hz → ~48Hz effective
PHYSICS_STEPS_PER_WP = 20
CONVERGE_THRESHOLD_RAD = np.deg2rad(0.05)

# 工作空间边界（world frame）
WORKSPACE_BOUNDS = np.array(
    [
        [0.28, 0.72],  # x [min, max] m
        [-0.45, 0.45],  # y [min, max] m
        [0.05, 0.55],  # z [min, max] m
    ],
    dtype=np.float64,
)

# Home 关节角
ARM_HOME_QPOS = np.deg2rad([-30.0, -1.9, 0.0, 13.5, -180.0, 74.7, 0.0]).astype(np.float64)

# ═══════════════════════════════════════════════ 速度限制
# EEF 空间速度限制 (替代关节空间限制 — 更贴合遥操作语义)
# 关节空间安全兜底阈值远高于正常运动，仅在 IK 产生病理解时触发。
MAX_EEF_LIN_VEL = 0.5  # m/s  (EEF 最大线速度)
MAX_EEF_ANG_VEL = np.deg2rad(180.0)  # rad/s (EEF 最大角速度)
MAX_QVEL_SAFETY_RAD_S = np.deg2rad([360.0] * 7)

# EEF 跟踪误差阈值
EEF_POS_ERROR_WARN_M = 0.03  # 位置误差告警 (3cm)
EEF_ROT_ERROR_WARN_RAD = np.deg2rad(10.0)  # 姿态误差告警 (10°)
EEF_POS_ERROR_CRITICAL_M = 0.08  # 位置误差安全触发 (8cm)
EEF_ROT_ERROR_CRITICAL_RAD = np.deg2rad(30.0)  # 姿态误差安全触发 (30°)
MAX_EEF_DIVERGENCE_CONSEC = 5

# 关节跟踪误差阈值 (硬件 PD 跟踪保护)
TRACKING_DIVERGENCE_THRESHOLD_RAD = 5.0

# 循环超限告警阈值
OVERRUN_WARN_RATIO = 1.5


# ═══════════════════════════════════════════════ 姿态工具




# ═══════════════════════════════════════════════ EEF 速度限制


def eef_speed_limited_target(
    actual_pos: np.ndarray,
    actual_quat: np.ndarray,
    desired_pos: np.ndarray,
    desired_quat: np.ndarray,
    max_lin_vel: float,
    max_ang_vel: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Clamp desired EEF target so actual EEF velocity limits are respected.

    Uses the current *actual* EEF pose (not the accumulated target) as the
    reference, so the speed-limited target is always reachable within one
    control step.  Position is clamped by linear distance; orientation is
    clamped via SLERP (spherical linear interpolation).

    Returns: (limited_pos, limited_quat) — both in world frame (wxyz quat).
    """
    # ── Position: clamp linear displacement ──
    pos_delta = desired_pos - actual_pos
    pos_dist = float(np.linalg.norm(pos_delta))
    max_pos_step = max_lin_vel * dt
    if pos_dist > max_pos_step:
        pos_delta = pos_delta / pos_dist * max_pos_step
    limited_pos = actual_pos + pos_delta

    # ── Orientation: clamp rotation angle via SLERP ──
    rot_dist = angular_dist_rad(desired_quat, actual_quat)
    max_rot_step = max_ang_vel * dt
    if rot_dist > max_rot_step:
        t = max_rot_step / rot_dist
        # q_err = actual^{-1} * desired  (rotation from actual to desired)
        q_actual_conj = np.array(
            [actual_quat[0], -actual_quat[1], -actual_quat[2], -actual_quat[3]],
            dtype=np.float64,
        )
        q_err = quat_multiply(q_actual_conj, desired_quat)
        if q_err[0] < 0:  # shortest path
            q_err = -q_err
        sin_half = float(np.linalg.norm(q_err[1:]))
        if sin_half > 1e-12:
            angle = 2.0 * np.arctan2(sin_half, q_err[0])
            axis = q_err[1:] / sin_half
            half = angle * t / 2.0
            q_partial = np.array(
                [np.cos(half), axis[0] * np.sin(half), axis[1] * np.sin(half), axis[2] * np.sin(half)],
                dtype=np.float64,
            )
        else:
            q_partial = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        limited_quat = quat_multiply(actual_quat, q_partial)
    else:
        limited_quat = desired_quat.copy()

    return limited_pos, limited_quat


def _joint_safety_clamp(
    target_qpos: np.ndarray,
    prev_cmd: np.ndarray,
    max_velocities: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Joint-space bottleneck scaling — safety backstop only (very permissive).

    Uses 360°/s limits, far above normal teleop speeds.  Only triggers when
    IK produces a pathological solution (e.g. near singularity).
    """
    step = target_qpos - prev_cmd
    max_step = max_velocities * dt
    ratio = np.max(np.abs(step) / np.maximum(max_step, 1e-8))
    if ratio > 1.0:
        return prev_cmd + step / ratio
    return target_qpos


# ═══════════════════════════════════════════════ workspace 工具


def is_in_workspace(pos: np.ndarray) -> bool:
    return bool(
        WORKSPACE_BOUNDS[0, 0] <= pos[0] <= WORKSPACE_BOUNDS[0, 1]
        and WORKSPACE_BOUNDS[1, 0] <= pos[1] <= WORKSPACE_BOUNDS[1, 1]
        and WORKSPACE_BOUNDS[2, 0] <= pos[2] <= WORKSPACE_BOUNDS[2, 1]
    )


# ═══════════════════════════════════════════════ 归位


def do_return_home(
    sim: SimRobotInterface,
    planner: XArm7MotionPlanner,
    home_eef: Pose,
    home_qpos: np.ndarray,
    viewer: sapien.Viewer | None = None,
) -> bool:
    """规划并执行从当前位置回 home 的碰撞安全路径。

    与 vr_teleop_sim.py 的 execute_return_home() 一致。
    Phase 1: plan_path(home EEF) → 稠密执行
    Phase 2: 关节空间收敛到精确 home_qpos
    """
    current_qpos = sim.get_full_qpos()[:7]

    if float(np.max(np.abs(current_qpos - home_qpos))) < np.deg2rad(0.5):
        return True

    result = planner.plan_path(home_eef, current_qpos)
    if not result.success or result.qpos_path is None:
        print(f"  return_home PLAN FAILED: {result.reason}")
        return False

    path = result.qpos_path
    dense = interpolate_waypoints(path, INTERP_MAX_STEP_RAD)

    # 碰撞检测：如果稠密路径无自碰撞，附加 home_qpos 终点
    if not any(planner.has_self_collision(q[:7]) for q in dense):
        full_path = np.vstack([dense, home_qpos.reshape(1, -1)])
        dense = interpolate_waypoints(full_path, INTERP_MAX_STEP_RAD)

    execute_dense_path(sim, dense, viewer, physics_steps_per_wp=PHYSICS_STEPS_PER_WP)
    err = settle_at_target(
        sim,
        home_qpos,
        np.zeros(12),
        max_iter=15,
        converge_threshold_rad=CONVERGE_THRESHOLD_RAD,
        physics_steps_per_wp=PHYSICS_STEPS_PER_WP,
    )

    final_qpos = sim.get_full_qpos()[:7]
    joint_err = float(np.max(np.abs(final_qpos - home_qpos)))
    print(f"  return_home OK  max_joint_err={np.rad2deg(joint_err):.2f}deg (settle={np.rad2deg(err):.2f}deg)")
    return True


# ═══════════════════════════════════════════════ 主循环


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Keyboard Teleop — xArm7 SAPIEN 仿真")
    parser.add_argument("--headless", action="store_true", help="无头模式（不创建 viewer 窗口）")
    args = parser.parse_args()

    print("=" * 60)
    print("仿真键盘遥操作 xArm7")
    print(f"  DELTA_POS={DELTA_POS * 1000:.0f}mm  DELTA_RPY={np.rad2deg(DELTA_RPY):.1f}deg  CTRL_HZ={CTRL_HZ}Hz")
    print(f"  EEF max_lin_vel={MAX_EEF_LIN_VEL:.1f}m/s  max_ang_vel={np.rad2deg(MAX_EEF_ANG_VEL):.0f}deg/s")
    print(f"  workspace: x{WORKSPACE_BOUNDS[0]} y{WORKSPACE_BOUNDS[1]} z{WORKSPACE_BOUNDS[2]}")
    print("=" * 60)

    # ── 1. 仿真初始化 ──
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
            srdf_path=str(ASSET_DIR / "robots" / "xhand" / "xarm7_xhand.srdf"),
            base_pose_world=Pose(p=np.array(root_pose.p), q=np.array(root_pose.q)),
        ),
        planning_profile=PlanningProfile(
            max_waypoint_delta_deg=360.0,
            max_ik_delta_deg=(180,) * 7,
            max_pose_error_rot_rad=np.deg2rad(5.0),
            num_random_ik_seeds=30,
            rrt_time_limit=2.0,
            num_rrt_attempts=2,
            check_self_collision=True,
        ),
        teleop_profile=TeleopProfile(
            max_pose_error_pos_m=0.02,
            max_pose_error_rot_rad=np.deg2rad(5.0),
        ),
    )

    home_eef = planner.compute_eef_pose_world(ARM_HOME_QPOS)

    # 复位到 home
    sim.reset()
    for _ in range(5):
        sim._step_physics(n=10)

    # 初始化碰撞模型手部姿态（消除 set_hand_qpos 未调用警告）
    planner.collision_model.set_hand_qpos(sim.get_full_qpos()[7:])

    # ── 2. Viewer ──
    viewer: sapien.Viewer | None = None
    if not args.headless:
        add_light(sim.scene)
        from dexmani_real.simulation.constructor import create_viewer

        viewer = create_viewer(
            sim.scene,
            sapien.Pose(
                [0.784, 0.027, 0.630],
                [0.005, -0.233, 0.001, 0.973],
            ),
        )

    # ── 3. 当前状态 ──
    sim_state = sim.get_state()
    target_pos = np.asarray(sim_state["eef_pos"], dtype=np.float64).copy()
    target_quat = np.asarray(sim_state["eef_quat_wxyz"], dtype=np.float64).copy()
    prev_arm_cmd = sim.get_full_qpos()[:7].copy()

    print(f"\n初始状态:")
    print(f"  arm_qpos:  {np.round(np.rad2deg(prev_arm_cmd), 1)} deg")
    print(f"  eef_pos:   {np.round(target_pos, 4)} m")
    print(f"  eef_quat:  {np.round(target_quat, 4)}")
    print(f"  home EEF:  {np.round(home_eef.p, 4)} m")

    # ── 4. 键盘 ──
    keys = GlobalKeyState()
    keys.start()
    print("\n键盘控制已启动，按 Q 退出\n")

    # ── 5. 状态变量 ──
    limiter = RateManager(CTRL_HZ)
    running = True
    wall_warned = [False, False, False]
    last_wall_time = 0.0
    loop_count = 0
    ik_fail_consecutive = 0
    max_consecutive_errors = 10
    consecutive_divergence = 0
    start_time = time.perf_counter()
    prev_eef_pos = None
    ik_method = "-"
    eef_pos_err = 0.0
    eef_rot_err = 0.0
    consecutive_eef_divergence = 0
    last_status_time = time.perf_counter()

    try:
        while running:
            limiter.wait()
            loop_count += 1
            tick_start = time.perf_counter()

            # ── Viewer 关闭检测 ──
            if viewer is not None and viewer.closed:
                print("\n[Viewer] 窗口已关闭，退出...")
                break

            # ── 退出 / 急停 ──
            if keys.is_pressed("esc"):
                print("\nESC: emergency_stop")
                running = False
                break
            if keys.is_pressed("q"):
                print("\nQ: 退出")
                running = False
                break

            # ── 归位 ──
            if keys.is_pressed("r"):
                print("\nR: return_home (path planned)...")
                do_return_home(sim, planner, home_eef, ARM_HOME_QPOS, viewer)
                # Snap target to exact home (not slightly-off sim reading)
                target_pos = home_eef.p.copy()
                target_quat = home_eef.q.copy()
                prev_arm_cmd = ARM_HOME_QPOS.copy()
                consecutive_divergence = 0
                ik_fail_consecutive = 0
                # Fall through to render step below — no continue

            # ── EEF 目标增量 ──
            dx = np.zeros(3, dtype=np.float64)
            if keys.is_pressed("w"):
                dx[0] += DELTA_POS
            if keys.is_pressed("s"):
                dx[0] -= DELTA_POS
            if keys.is_pressed("a"):
                dx[1] -= DELTA_POS
            if keys.is_pressed("d"):
                dx[1] += DELTA_POS
            if keys.is_pressed("up"):
                dx[2] += DELTA_POS
            if keys.is_pressed("down"):
                dx[2] -= DELTA_POS

            drpy = np.zeros(3, dtype=np.float64)
            if keys.is_pressed("left"):
                drpy[0] += DELTA_RPY
            if keys.is_pressed("right"):
                drpy[0] -= DELTA_RPY
            if keys.is_pressed("i"):
                drpy[1] += DELTA_RPY
            if keys.is_pressed("k"):
                drpy[1] -= DELTA_RPY
            if keys.is_pressed("j"):
                drpy[2] -= DELTA_RPY
            if keys.is_pressed("l"):
                drpy[2] += DELTA_RPY

            # ── 周期性状态打印 ──
            now = time.perf_counter()
            if now - last_status_time > 2.0:
                sim_state = sim.get_state()
                eef_pos = np.asarray(sim_state["eef_pos"], dtype=np.float64)
                eef_quat = np.asarray(sim_state["eef_quat_wxyz"], dtype=np.float64)
                if prev_eef_pos is not None:
                    vel = np.linalg.norm(eef_pos - prev_eef_pos) / max(now - last_status_time, 1e-3)
                else:
                    vel = 0.0
                prev_eef_pos = eef_pos.copy()

                # EEF 跟踪误差 (pos + rot)
                eef_pos_err, eef_rot_err = compute_pose_error(
                    Pose(p=target_pos, q=target_quat),
                    Pose(p=eef_pos, q=eef_quat),
                )

                elapsed = now - start_time
                status = (
                    f"[T+{elapsed:.1f}s f={loop_count}] "
                    f"eef={np.round(eef_pos, 3)}m  target={np.round(target_pos, 3)}  "
                    f"v={vel:.2f}m/s  ik={ik_method}"
                    f"  err_pos={eef_pos_err*100:.1f}cm err_rot={np.rad2deg(eef_rot_err):.1f}°"
                )
                if ik_fail_consecutive > 0:
                    status += f"  IK_fail(consec)={ik_fail_consecutive}"
                # EEF tracking warning (non-critical)
                if eef_pos_err > EEF_POS_ERROR_WARN_M or eef_rot_err > EEF_ROT_ERROR_WARN_RAD:
                    status += "  ⚠ EEF_drift"
                print(status, flush=True)
                last_status_time = now

            # ── 有输入则计算 control ──
            has_input = not (np.all(dx == 0) and np.all(drpy == 0))
            if has_input:
                # 软墙: 方向感知 — 允许移回工作空间
                new_pos = target_pos + dx
                for axis in range(3):
                    if dx[axis] == 0:
                        continue
                    lo, hi = WORKSPACE_BOUNDS[axis]
                    cur = target_pos[axis]
                    new = new_pos[axis]
                    if lo <= new <= hi:
                        target_pos[axis] = new
                        if wall_warned[axis] and lo + 0.01 <= new <= hi - 0.01:
                            wall_warned[axis] = False
                    elif (cur < lo and dx[axis] > 0) or (cur > hi and dx[axis] < 0):
                        # Moving back toward workspace → allow, clamp to boundary
                        target_pos[axis] = float(np.clip(new, lo, hi))
                    else:
                        # Moving further outside → reject
                        if not wall_warned[axis] or now - last_wall_time > 3.0:
                            names = ["x", "y", "z"]
                            print(f"  ⚠ {names[axis]} 轴到达边界 [{lo:.2f}, {hi:.2f}]")
                            wall_warned[axis] = True
                            last_wall_time = now

                if np.any(drpy != 0):
                    dq = R.from_euler('xyz', drpy).as_quat(scalar_first=True)
                    target_quat = quat_multiply(dq, target_quat)

                # ── EEF 速度限制 + IK ──
                # 以当前实际 EEF 姿态为参考，限制目标 EEF 增量，确保 IK
                # 永远求解一个在单步内可达的姿态（避免全伸展时目标漂移）。
                sim_state = sim.get_state()
                actual_pos = np.asarray(sim_state["eef_pos"], dtype=np.float64)
                actual_quat = np.asarray(sim_state["eef_quat_wxyz"], dtype=np.float64)

                ik_target_pos, ik_target_quat = eef_speed_limited_target(
                    actual_pos,
                    actual_quat,
                    target_pos,
                    target_quat,
                    MAX_EEF_LIN_VEL,
                    MAX_EEF_ANG_VEL,
                    CTRL_DT,
                )
                # Keep accumulated target synced to the limited pose — no drift.
                target_pos = ik_target_pos
                target_quat = ik_target_quat

                current_arm_qpos = sim.get_full_qpos()[:7]
                target_pose = Pose(p=ik_target_pos, q=ik_target_quat)
                ik_result = planner.solve_teleop_ik(target_pose, current_arm_qpos, prev_arm_cmd)

                if ik_result.success and ik_result.qpos is not None:
                    ik_fail_consecutive = 0
                    ik_method = "diff"
                    # Joint-space safety backstop (360°/s — catches pathological IK only)
                    arm_cmd = _joint_safety_clamp(
                        ik_result.qpos,
                        prev_arm_cmd,
                        MAX_QVEL_SAFETY_RAD_S,
                        CTRL_DT,
                    )
                    prev_arm_cmd = arm_cmd.copy()
                else:
                    ik_fail_consecutive += 1
                    reason = getattr(ik_result, "reason", "") or "unknown"
                    if ik_fail_consecutive <= 1 or ik_fail_consecutive % 50 == 0:
                        print(f"  ⚡ IK fail (#{ik_fail_consecutive}): {reason}", flush=True)
                    target_pos = actual_pos.copy()
                    target_quat = actual_quat.copy()
                    arm_cmd = current_arm_qpos.copy()
                    prev_arm_cmd = arm_cmd.copy()
                    ik_method = "held"
            else:
                # Idle: hold current position via PD
                ik_method = "idle"
                ik_fail_consecutive = 0
                arm_cmd = prev_arm_cmd

            # ── 应用动作（每 tick 都执行，确保物理持续推进）──
            hand_qpos = sim.get_full_qpos()[7:]
            sim.robot.balance_passive_force()
            sim.robot.apply_action(np.concatenate([arm_cmd, hand_qpos]))
            sim._step_physics(n=PHYSICS_STEPS_PER_TICK)

            # ── 追踪安全 (关节空间: PD 跟踪保护) ──
            actual_qpos = sim.get_full_qpos()[:7]
            tracking_err = np.max(np.abs(actual_qpos - arm_cmd))
            if tracking_err > TRACKING_DIVERGENCE_THRESHOLD_RAD:
                consecutive_divergence += 1
                if consecutive_divergence == 1:
                    print(f"  [SAFETY] Joint tracking divergence: max_err={tracking_err:.1f}rad")
                if consecutive_divergence >= 3:
                    print("  [SAFETY] Persistent joint tracking divergence — stopping")
                    running = False
                    break
            else:
                consecutive_divergence = 0

            # ── EEF 跟踪安全 (姿态空间: target-vs-actual) ──
            if eef_pos_err > EEF_POS_ERROR_CRITICAL_M or eef_rot_err > EEF_ROT_ERROR_CRITICAL_RAD:
                consecutive_eef_divergence += 1
                if consecutive_eef_divergence == 1:
                    print(
                        f"  [SAFETY] EEF tracking error: "
                        f"pos={eef_pos_err*100:.1f}cm rot={np.rad2deg(eef_rot_err):.1f}°"
                    )
                if consecutive_eef_divergence >= MAX_EEF_DIVERGENCE_CONSEC:
                    print("  [SAFETY] Persistent EEF tracking divergence — stopping")
                    running = False
                    break
            else:
                consecutive_eef_divergence = 0

            # ── 渲染（每 tick 都执行，保持 viewer 响应）──
            if viewer is not None:
                sim.scene.update_render()
                viewer.render()

            # ── 超限检测 ──
            tick_elapsed_ms = (time.perf_counter() - tick_start) * 1000.0
            target_ms = CTRL_DT * 1000.0
            if tick_elapsed_ms > target_ms * OVERRUN_WARN_RATIO:
                print(f"[Overrun] tick={tick_elapsed_ms:.1f}ms target={target_ms:.1f}ms")

    finally:
        keys.stop()
        time.sleep(0.05)
        sim.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
