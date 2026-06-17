#!/usr/bin/env python3
"""键盘遥操作 xArm7 仿真模型（SAPIEN 可视化）。

用法:
    conda activate real
    python keyboard_teleop_sim.py

键位:
    w/s     EEF x +/- (前/后)
    a/d     EEF y -/+ (左/右)
    ↑/↓     EEF z +/- (上/下)
    ←/→     roll -/+
    i/k     pitch -/+
    j/l     yaw +/-
    r       规划路径回 home（退出循环后仍可用）
    q       退出遥操作循环（再按一次完全退出）
"""

from __future__ import annotations

# sys.path修正：脚本已从dexmani_real/example/移至scripts/
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))


import threading
import time

import numpy as np
import sapien.core as sapien
from pynput import keyboard as pynput_keyboard
from transforms3d import euler

from dexmani_real import ASSET_DIR
from dexmani_real.planning import (
    PlanningProfile, Pose, TeleopProfile, XArm7MotionPlanner, XArm7PlannerConfig,
)
from dexmani_real.simulation import SimRobotConfig, SimRobotInterface
from dexmani_real.simulation.constructor import add_light, create_viewer

# ═══════════════════════════════════════════════ 配置

HEADLESS = False
DELTA_POS = 0.005       # 每次按键平移增量 (m)
DELTA_RPY = 0.02        # 每次按键旋转增量 (rad)
PHYSICS_STEPS_PER_WP = 20
INTERP_MAX_STEP_RAD = np.deg2rad(2.0)
CTRL_DT = 0.02          # 控制周期 (s)，50Hz

WORKSPACE_BOUNDS = np.array([
    [0.28, 0.70],   # x [min, max] m  — 在所有 y/z 下 IK 100% 可达
    [-0.40, 0.40],  # y [min, max] m
    [0.02, 0.55],   # z [min, max] m
])


# ═══════════════════════════════════════════════ 键盘状态


class GlobalKeyState:
    """非阻塞全局键盘状态追踪（后台 listener 线程）。"""

    def __init__(self):
        self._pressed: set = set()
        self._lock = threading.Lock()
        self._listener = pynput_keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self._listener.daemon = True
        self._listener.start()

    def _on_press(self, key):
        with self._lock:
            ch = getattr(key, "char", None)
            self._pressed.add(ch.lower() if ch else key)

    def _on_release(self, key):
        with self._lock:
            ch = getattr(key, "char", None)
            self._pressed.discard(ch.lower() if ch else key)

    def is_pressed(self, key) -> bool:
        """检查字符键 ('w') 或特殊键 (Key.up) 是否被按下。"""
        if isinstance(key, str) and len(key) == 1:
            key = key.lower()
        with self._lock:
            return key in self._pressed

    def clear(self, key):
        with self._lock:
            self._pressed.discard(key)

    def stop(self):
        try:
            self._listener.stop()
        except Exception:
            pass


# ═══════════════════════════════════════════════ 仿真辅助


def interpolate_waypoints(path: np.ndarray, max_step: float = INTERP_MAX_STEP_RAD) -> np.ndarray:
    """对稀疏关节路径线性插值。"""
    if len(path) <= 1:
        return path
    dense = [path[0]]
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        n = int(np.ceil(float(np.max(np.abs(b - a))) / max_step))
        for k in range(1, n + 1):
            dense.append(a + (k / n) * (b - a))
    return np.array(dense, dtype=np.float64)


def execute_dense_path(
    sim: SimRobotInterface, dense: np.ndarray, viewer: sapien.Viewer | None = None,
) -> bool:
    """执行已插值的稠密关节路径，(N,7) arm-only。"""
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
    """在目标 arm 关节上稳定 PD 控制器。"""
    for _ in range(3):
        sim.robot.balance_passive_force()
        sim.robot.apply_action(np.concatenate([target_arm, hand_qpos]))
        sim._step_physics(n=PHYSICS_STEPS_PER_WP)


def execute_return_home(
    sim: SimRobotInterface,
    planner: XArm7MotionPlanner,
    home_qpos: np.ndarray,
    home_eef: Pose,
    viewer: sapien.Viewer | None = None,
) -> bool:
    """规划并执行 return_home 路径。"""
    current_qpos = sim.get_full_qpos()[:7]
    result = planner.plan_path(home_eef, current_qpos)
    if not result.success or result.qpos_path is None:
        print(f"  return_home PLAN FAILED: {result.reason}")
        return False

    path = result.qpos_path
    # 追加关节归位目标
    full = np.vstack([path, home_qpos])
    dense_joint = interpolate_waypoints(full)
    if not any(planner.has_self_collision(q) for q in dense_joint):
        path = full

    dense = interpolate_waypoints(path)
    hand_qpos = sim.get_full_qpos()[7:]
    execute_dense_path(sim, dense, viewer)
    settle_at_target(sim, dense[-1, :7], hand_qpos)

    final_qpos = sim.get_full_qpos()[:7]
    joint_err = float(np.max(np.abs(final_qpos - home_qpos)))
    print(f"  return_home OK  max_joint_err={np.rad2deg(joint_err):.2f}deg")
    return True


# ═══════════════════════════════════════════════ 主循环


def main():
    keys = GlobalKeyState()

    # ── 初始化仿真 ──
    sim = SimRobotInterface(SimRobotConfig(headless=HEADLESS))
    if not sim.connect():
        raise RuntimeError(f"sim connect failed: {sim.last_error_message}")

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
            max_ik_jump_deg=(8, 8, 8, 8, 12, 12, 18),  # tight: reject IK branch switches (keyboard EEF moves 5mm/step → ~0.5° per joint)
            max_pose_error_pos_m=0.01,
            max_pose_error_rot_rad=np.deg2rad(5.0),
            
        ),
    )

    home_qpos = sim.config.arm_home_qpos.copy()
    home_eef = planner.compute_eef_pose_world(home_qpos)

    # 复位
    sim.reset()
    for _ in range(5):
        sim._step_physics(n=10)

    # ── 可视化 ──
    viewer = None
    if not HEADLESS:
        add_light(sim.scene)
        viewer = create_viewer(sim.scene, sapien.Pose(
            [0.784, 0.027, 0.630],
            [0.005, -0.233, 0.001, 0.973],
        ))

    # ── 初始化目标位姿 ──
    target_pos = np.array(home_eef.p, dtype=np.float64)
    target_rpy = np.zeros(3, dtype=np.float64)
    hand_qpos = sim.get_full_qpos()[7:].copy()
    prev_qpos_cmd = sim.get_full_qpos()[:7].copy()
    wall_warned = [False, False, False]  # 每个轴撞墙时打印一次警告
    last_wall_time = 0.0

    print("=" * 60)
    print("Keyboard Teleop — xArm7 仿真")
    print(f"  home EEF: {np.round(home_eef.p, 3)}")
    print("  w/s: x+/-  a/d: y-/+  ↑/↓: z+/-")
    print("  ←/→: roll  i/k: pitch  j/l: yaw")
    print("  r: return_home  q: 退出循环")
    print("=" * 60)

    # ═══════════════════════════════════════════ 遥操作主循环

    while True:
        if viewer is not None and viewer.closed:
            break
        if keys.is_pressed('q'):
            break

        # ── return_home ──
        if keys.is_pressed('r'):
            keys.clear('r')
            execute_return_home(sim, planner, home_qpos, home_eef, viewer)
            target_pos = np.array(home_eef.p, dtype=np.float64)
            target_rpy = np.zeros(3)
            prev_qpos_cmd = sim.get_full_qpos()[:7].copy()
            continue

        # ── 累积平移增量（软墙：越界方向拒绝）──
        dx = np.zeros(3, dtype=np.float64)
        if keys.is_pressed('w'):
            dx[0] += DELTA_POS
        if keys.is_pressed('s'):
            dx[0] -= DELTA_POS
        if keys.is_pressed('a'):
            dx[1] -= DELTA_POS
        if keys.is_pressed('d'):
            dx[1] += DELTA_POS
        if keys.is_pressed(pynput_keyboard.Key.up):
            dx[2] += DELTA_POS
        if keys.is_pressed(pynput_keyboard.Key.down):
            dx[2] -= DELTA_POS

        new_pos = target_pos + dx
        for axis in range(3):
            if dx[axis] == 0:
                continue
            lo, hi = WORKSPACE_BOUNDS[axis]
            if lo <= new_pos[axis] <= hi:
                target_pos[axis] = new_pos[axis]
            else:
                now = time.perf_counter()
                if not wall_warned[axis] or now - last_wall_time > 3.0:
                    names = ["x", "y", "z"]
                    print(f"  ⚠ {names[axis]} 轴到达边界 [{lo:.2f}, {hi:.2f}]")
                    wall_warned[axis] = True
                    last_wall_time = now

        # ── 累积旋转增量 ──
        if keys.is_pressed(pynput_keyboard.Key.left):
            target_rpy[0] -= DELTA_RPY
        if keys.is_pressed(pynput_keyboard.Key.right):
            target_rpy[0] += DELTA_RPY
        if keys.is_pressed('i'):
            target_rpy[1] -= DELTA_RPY
        if keys.is_pressed('k'):
            target_rpy[1] += DELTA_RPY
        if keys.is_pressed('j'):
            target_rpy[2] += DELTA_RPY
        if keys.is_pressed('l'):
            target_rpy[2] -= DELTA_RPY

        # ── IK ──
        target_quat = np.array(euler.euler2quat(*target_rpy, axes="sxyz"), dtype=np.float64)
        target_pose = Pose(p=target_pos, q=target_quat)
        current_qpos = sim.get_full_qpos()[:7]

        ik_result = planner.solve_teleop_ik(target_pose, current_qpos, prev_qpos_cmd)
        if ik_result.success and ik_result.qpos is not None:
            prev_qpos_cmd = ik_result.qpos.copy()
            sim.robot.balance_passive_force()
            sim.robot.apply_action(np.concatenate([ik_result.qpos, hand_qpos]))
            sim._step_physics(n=PHYSICS_STEPS_PER_WP)
        # IK 失败时保持当前位置（不发送命令）

        if viewer is not None:
            sim.scene.update_render()
            viewer.render()

        time.sleep(CTRL_DT)

    # ═══════════════════════════════════════════ 退出后等待 return_home

    keys.clear('q')
    keys.clear('r')
    time.sleep(0.15)

    if viewer is None or not viewer.closed:
        print("\n遥操作已退出。")
        print("  r: 规划路径回 home  |  q: 完全退出")
        while True:
            if viewer is not None and viewer.closed:
                break
            if keys.is_pressed('q'):
                break
            if keys.is_pressed('r'):
                keys.clear('r')
                execute_return_home(sim, planner, home_qpos, home_eef, viewer)
                print("  r: 再次回 home  |  q: 完全退出")
            if viewer is not None:
                sim.scene.update_render()
                viewer.render()
            time.sleep(0.1)

    # ── 清理 ──
    keys.stop()
    sim.disconnect()
    print("Done.")


if __name__ == "__main__":
    main()
