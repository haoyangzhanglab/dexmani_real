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
    - solve_teleop_ik 迭代 DLS + MPlib 位置 IK 兜底
    - tracking divergence 监控 (cmd-vs-actual 偏差)
    - 速度限制 (bottleneck scaling)

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
from dexmani_real.planning import (
    PlanningProfile,
    Pose,
    TeleopProfile,
    XArm7MotionPlanner,
    XArm7PlannerConfig,
)
from dexmani_real.simulation import SimRobotConfig, SimRobotInterface
from dexmani_real.simulation.constructor import add_light, setup_scene
from dexmani_real.utils.rate_limiter import RateLimiter

from dexmani_real.planning.path_utils import interpolate_waypoints
from dexmani_real.planning.pose_utils import quat_multiply
from dexmani_real.simulation import execute_dense_path, settle_at_target

# ═══════════════════════════════════════════════ 配置

CTRL_HZ = 50.0
CTRL_DT = 1.0 / CTRL_HZ
DELTA_POS = 0.005     # 每次按键 EEF 平移量 (m)
DELTA_RPY = 0.02      # 每次按键 EEF 旋转量 (rad)
INTERP_MAX_STEP_RAD = np.deg2rad(2.0)
PHYSICS_STEPS_PER_TICK = 5        # 240Hz → ~48Hz effective
PHYSICS_STEPS_PER_WP = 20
CONVERGE_THRESHOLD_RAD = np.deg2rad(0.05)

# 工作空间边界（world frame）
WORKSPACE_BOUNDS = np.array([
    [0.28, 0.72],    # x [min, max] m
    [-0.45, 0.45],   # y [min, max] m
    [0.05, 0.55],    # z [min, max] m
], dtype=np.float64)

# Home 关节角
ARM_HOME_QPOS = np.deg2rad([-30.0, -1.9, 0.0, 13.5, -180.0, 74.7, 0.0]).astype(np.float64)

# 速度限制 (rad/s)
MAX_QVEL_RAD_S = np.deg2rad([180, 180, 180, 180, 180, 180, 180])
TRACKING_DIVERGENCE_THRESHOLD_RAD = 5.0

# 循环超限告警阈值
OVERRUN_WARN_RATIO = 1.5


# ═══════════════════════════════════════════════ 键盘输入 (pynput)


class GlobalKeyState:
    """非阻塞键盘状态追踪 (pynput, 线程安全)。

    与 keyboard_teleop_real.py 中的实现一致。
    """

    def __init__(self):
        self._keys: set[str] = set()
        self._running = True
        self._thread = None
        self._listener: "keyboard.Listener | None" = None  # type: ignore[name-defined]

    def _run(self):
        from pynput import keyboard

        def on_press(key):
            try:
                if hasattr(key, "char") and key.char is not None:
                    self._keys.add(key.char.lower())
                elif key == keyboard.Key.esc:
                    self._keys.add("esc")
                elif key == keyboard.Key.up:
                    self._keys.add("up")
                elif key == keyboard.Key.down:
                    self._keys.add("down")
                elif key == keyboard.Key.left:
                    self._keys.add("left")
                elif key == keyboard.Key.right:
                    self._keys.add("right")
            except Exception:
                pass

        def on_release(key):
            try:
                if hasattr(key, "char") and key.char is not None:
                    self._keys.discard(key.char.lower())
                elif key == keyboard.Key.esc:
                    self._keys.discard("esc")
                elif key == keyboard.Key.up:
                    self._keys.discard("up")
                elif key == keyboard.Key.down:
                    self._keys.discard("down")
                elif key == keyboard.Key.left:
                    self._keys.discard("left")
                elif key == keyboard.Key.right:
                    self._keys.discard("right")
            except Exception:
                pass

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()
        while self._running:
            time.sleep(0.1)
        self._listener.stop()
        self._listener = None

    def stop(self):
        self._running = False

    def start(self):
        import threading

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def is_pressed(self, key: str) -> bool:
        return key in self._keys

    @property
    def any_pressed(self) -> bool:
        return len(self._keys) > 0


# ═══════════════════════════════════════════════ 姿态工具


def rpy_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """RPY (rad) → wxyz 四元数。"""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


# ═══════════════════════════════════════════════ 速度限制


def velocity_limited_step(
    target: np.ndarray, prev_cmd: np.ndarray,
    max_velocities: np.ndarray, dt: float,
) -> np.ndarray:
    """Bottleneck scaling: limit per-joint step to max_velocities * dt."""
    step = target - prev_cmd
    max_step = max_velocities * dt
    ratio = np.max(np.abs(step) / np.maximum(max_step, 1e-8))
    if ratio > 1.0:
        return prev_cmd + step / ratio
    return target


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
        sim, home_qpos, np.zeros(12),
        max_iter=15, converge_threshold_rad=CONVERGE_THRESHOLD_RAD,
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
            teleop_dt=CTRL_DT,
            use_position_ik=True,           # MPlib 兜底: DLS 迭代不收敛时接管
            max_pose_error_pos_m=0.02,
            max_pose_error_rot_rad=np.deg2rad(5.0),
            differential_ik_max_pos_step_m=0.05,
        ),
    )

    home_eef = planner.compute_eef_pose_world(ARM_HOME_QPOS)

    # 复位到 home
    sim.reset()
    for _ in range(5):
        sim._step_physics(n=10)

    # 初始化碰撞模型手部姿态（消除 set_hand_qpos 未调用警告）
    planner.collision_model.set_hand_qpos(sim.get_full_qpos()[7:])

    # 注册桌面障碍物 — 匹配 SAPIEN 场景中 constructor.py 的 table actor
    # (中心 [0.4, 0, -0.5], half_size [0.5, 1.0, 0.5] → 桌面顶部 z=0)
    planner.collision_model.add_table(
        table_height=0.0,
        x_center=0.4,
        half_x=0.5,
        half_y=1.0,
        half_z=0.04,
    )

    # ── 2. Viewer ──
    viewer: sapien.Viewer | None = None
    if not args.headless:
        add_light(sim.scene)
        from dexmani_real.simulation.constructor import create_viewer

        viewer = create_viewer(sim.scene, sapien.Pose(
            [0.784, 0.027, 0.630],
            [0.005, -0.233, 0.001, 0.973],
        ))

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
    limiter = RateLimiter(CTRL_HZ)
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
            if keys.is_pressed("w"):    dx[0] += DELTA_POS
            if keys.is_pressed("s"):    dx[0] -= DELTA_POS
            if keys.is_pressed("a"):    dx[1] -= DELTA_POS
            if keys.is_pressed("d"):    dx[1] += DELTA_POS
            if keys.is_pressed("up"):   dx[2] += DELTA_POS
            if keys.is_pressed("down"): dx[2] -= DELTA_POS

            drpy = np.zeros(3, dtype=np.float64)
            if keys.is_pressed("left"):  drpy[0] += DELTA_RPY
            if keys.is_pressed("right"): drpy[0] -= DELTA_RPY
            if keys.is_pressed("i"):     drpy[1] += DELTA_RPY
            if keys.is_pressed("k"):     drpy[1] -= DELTA_RPY
            if keys.is_pressed("j"):     drpy[2] -= DELTA_RPY
            if keys.is_pressed("l"):     drpy[2] += DELTA_RPY

            # ── 周期性状态打印 ──
            now = time.perf_counter()
            if now - last_status_time > 2.0:
                sim_state = sim.get_state()
                eef = np.round(np.asarray(sim_state["eef_pos"], dtype=np.float64), 3)
                if prev_eef_pos is not None:
                    vel = np.linalg.norm(np.asarray(sim_state["eef_pos"]) - prev_eef_pos) / max(now - last_status_time, 1e-3)
                else:
                    vel = 0.0
                prev_eef_pos = np.asarray(sim_state["eef_pos"], dtype=np.float64).copy()
                elapsed = now - start_time
                status = (
                    f"[T+{elapsed:.1f}s f={loop_count}] "
                    f"eef={eef}m  target={np.round(target_pos, 3)}  "
                    f"v={vel:.2f}m/s  ik={ik_method}"
                )
                if ik_fail_consecutive > 0:
                    status += f"  IK_fail(consec)={ik_fail_consecutive}"
                print(status, flush=True)
                last_status_time = now

            # ── 有输入则计算 control ──
            has_input = not (np.all(dx == 0) and np.all(drpy == 0))
            if has_input:
                # 软墙: 逐轴拒绝边界外移动
                new_pos = target_pos + dx
                for axis in range(3):
                    if dx[axis] == 0:
                        continue
                    lo, hi = WORKSPACE_BOUNDS[axis]
                    if lo <= new_pos[axis] <= hi:
                        target_pos[axis] = new_pos[axis]
                        if wall_warned[axis] and lo + 0.01 <= new_pos[axis] <= hi - 0.01:
                            wall_warned[axis] = False
                    else:
                        if not wall_warned[axis] or now - last_wall_time > 3.0:
                            names = ["x", "y", "z"]
                            print(f"  ⚠ {names[axis]} 轴到达边界 [{lo:.2f}, {hi:.2f}]")
                            wall_warned[axis] = True
                            last_wall_time = now

                if np.any(drpy != 0):
                    dq = rpy_to_quat_wxyz(drpy[0], drpy[1], drpy[2])
                    target_quat = quat_multiply(dq, target_quat)

                # IK
                sim_state = sim.get_state()
                current_arm_qpos = sim.get_full_qpos()[:7]
                target_pose = Pose(p=target_pos, q=target_quat)
                ik_result = planner.solve_teleop_ik(target_pose, current_arm_qpos, prev_arm_cmd)

                if ik_result.success and ik_result.qpos is not None:
                    ik_fail_consecutive = 0
                    ik_method = "diff"
                    arm_cmd = velocity_limited_step(
                        ik_result.qpos, prev_arm_cmd, MAX_QVEL_RAD_S, CTRL_DT,
                    )
                    prev_arm_cmd = arm_cmd.copy()
                else:
                    ik_fail_consecutive += 1
                    target_pos = np.asarray(sim_state["eef_pos"], dtype=np.float64).copy()
                    target_quat = np.asarray(sim_state["eef_quat_wxyz"], dtype=np.float64).copy()
                    # Hold current position — don't keep driving toward the last
                    # successful command that led into collision.
                    arm_cmd = current_arm_qpos.copy()
                    prev_arm_cmd = arm_cmd.copy()
                    ik_method = "held"
            else:
                # Idle: hold current position via PD
                arm_cmd = prev_arm_cmd

            # ── 应用动作（每 tick 都执行，确保物理持续推进）──
            hand_qpos = sim.get_full_qpos()[7:]
            sim.robot.balance_passive_force()
            sim.robot.apply_action(np.concatenate([arm_cmd, hand_qpos]))
            sim._step_physics(n=PHYSICS_STEPS_PER_TICK)

            # ── 追踪安全 ──
            actual_qpos = sim.get_full_qpos()[:7]
            tracking_err = np.max(np.abs(actual_qpos - arm_cmd))
            if tracking_err > TRACKING_DIVERGENCE_THRESHOLD_RAD:
                consecutive_divergence += 1
                if consecutive_divergence == 1:
                    print(f"  [SAFETY] Tracking divergence: max_err={tracking_err:.1f}rad")
                if consecutive_divergence >= 3:
                    print("  [SAFETY] Persistent tracking divergence — stopping")
                    running = False
                    break
            else:
                consecutive_divergence = 0

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
