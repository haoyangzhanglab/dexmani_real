#!/usr/bin/env python3
"""真机键盘遥操作 xArm7 — 通过 RobotInterface 控制真实硬件。

用法:
    source /home/zhy/anaconda3/etc/profile.d/conda.sh && conda activate real
    python keyboard_teleop_real.py

控制:
    移动 EEF (base 坐标系):
      W/S       X 前后
      A/D       Y 左右
      ↑/↓       Z 上下
      I/K       Pitch (Y 旋转)
      ←/→       Roll  (X 旋转)
      J/L       Yaw   (Z 旋转)
    Q          退出主循环 (不归位，退出后可继续按 R 归位)
    R          return_home (保持循环)
    ESC        急停 (emergency_stop + 退出)

安全:
    - 启动时执行 Pre-Flight 检查清单
    - 软墙 workspace 安全 (到达边界拒绝移动)
    - solve_teleop_ik 保证分支连续性
    - EMA 平滑抑制抖动
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import termios

import time
import traceback
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
from dexmani_real.robot.interface import RobotAction, RobotInterface, RobotInterfaceConfig
from dexmani_real.robot.xarm7 import XArm7Config
from dexmani_real.planning.collision_config import CollisionConfig

from scripts._test_utils import quat_multiply

# ═══════════════════════════════════════════════ 配置

CTRL_DT = 0.02       # 50Hz
DELTA_POS = 0.005    # 每次按键 EEF 平移量 (m)
DELTA_RPY = 0.02     # 每次按键 EEF 旋转量 (rad)
EMA_ALPHA = 0.7      # EMA 平滑系数 (geometric_fk 下 IK 确定性, 无随机 seed 抖动; 0.7兼顾响应速度与平滑)

WORKSPACE_BOUNDS = np.array([
    [0.28, 0.72],    # x [min, max] m
    [-0.45, 0.45],   # y [min, max] m
    [0.05, 0.5],    # z [min, max] m
], dtype=np.float64)

# 碰撞检测配置 — geometric_fk 使用 Pinocchio FK 指尖 Z 检测桌面
# 零成本，不污染 IK（MPlib 点云会使 IK 成功率从 100% → 53%）
COLLISION_CONFIG = CollisionConfig(
    table_z_world=0.0,
    hand_extension_below_eef=0.076,   # pinky_tip 在 home 位 EEF 下方距离
    hand_safe_margin=0.03,            # 指尖到桌面最小安全间距
    env_collision_mode="geometric_fk",
)

# ═══════════════════════════════════════════════ 键盘输入


class GlobalKeyState:
    """非阻塞键盘状态追踪 (pynput, 线程安全)。"""

    def __init__(self):
        self._keys: set[str] = set()
        self._running = True
        self._thread = None
        self._listener: "keyboard.Listener | None" = None

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


# ═══════════════════════════════════════════════ 安全


@dataclass
class PreFlightReport:
    passed: bool
    checks: list[tuple[str, bool | None, str]]  # (name, ok, detail)  ok: True=pass, False=fail, None=warn


def preflight_check(robot: RobotInterface) -> PreFlightReport:
    """Pre-Flight 检查清单 (hardware-safety.md)。"""
    checks: list[tuple[str, bool, str]] = []

    # 连接检查
    ok = robot.arm.is_connected()
    checks.append(("arm 连接", ok, "" if ok else "arm 未连接"))

    ok = not robot.arm.is_error()
    checks.append(("arm 无错误", ok, "" if ok else robot.arm.last_error_message))

    # 状态检查
    state = robot.arm.get_state()
    qpos = np.asarray(state["qpos"], dtype=np.float64)
    has_nan = not np.all(np.isfinite(qpos))
    checks.append(("关节角度有效", not has_nan, "含 NaN" if has_nan else ""))

    if not has_nan:
        config = robot.config.arm
        in_range = bool(np.all(qpos >= config.qpos_min) and np.all(qpos <= config.qpos_max))
        checks.append(("关节在限位内", in_range, f"qpos={np.round(np.rad2deg(qpos), 1)}deg" if not in_range else ""))

    # Hand (降级允许)
    hand_ok = robot.hand.is_connected()
    checked_val = hand_ok if hand_ok else None
    checks.append(("hand 连接", checked_val, "" if hand_ok else "降级运行 (arm only)"))

    if hand_ok:
        hand_state = robot.hand.get_state(full=True)
        hand_errs = hand_state.get("commboard_err", np.zeros(12))
        hand_ok2 = bool(np.all(hand_errs == 0))
        checked_val = hand_ok2 if hand_ok2 else None
        checks.append(("hand 通信正常", checked_val, f"commboard_err={hand_errs}" if not hand_ok2 else ""))

    passed = all(ok is not False for _, ok, _ in checks)
    return PreFlightReport(passed=passed, checks=checks)


def print_preflight(report: PreFlightReport):
    print("\nPre-Flight 检查:")
    for name, ok, detail in report.checks:
        if ok is True:
            status = "OK"
        elif ok is False:
            status = "FAIL"
        else:
            status = "WARN"
        detail_str = f"  ({detail})" if detail else ""
        print(f"  [{status}] {name}{detail_str}")
    has_warnings = any(ok is None for _, ok, _ in report.checks)
    if report.passed:
        result = "通过 (有告警)" if has_warnings else "全部通过"
    else:
        result = "检查失败，中止操作"
    print(f"\n结果: {result}\n")


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



# ═══════════════════════════════════════════════ 主循环


class RateLimiter:
    def __init__(self, target_hz: float):
        self.dt = 1.0 / target_hz
        self.last_wake = time.perf_counter()

    def wait(self):
        elapsed = time.perf_counter() - self.last_wake
        sleep_time = self.dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        self.last_wake = time.perf_counter()


def do_return_home(robot: RobotInterface, planner: XArm7MotionPlanner):
    """执行 return_home。"""
    print("return_home ...", flush=True)
    try:
        ok = robot.return_to_home(use_planning=True)
        print(f"  {'OK' if ok else 'FAIL'}")
    except Exception:
        traceback.print_exc()
        print("  return_to_home 异常，尝试 emergency_stop")
        robot.emergency_stop()


def main():
    print("=" * 60)
    print("真机键盘遥操作 xArm7")
    print(f"  DELTA_POS={DELTA_POS*1000:.0f}mm  DELTA_RPY={np.rad2deg(DELTA_RPY):.1f}deg  CTRL_DT={CTRL_DT}s")
    print(f"  workspace: x{WORKSPACE_BOUNDS[0]} y{WORKSPACE_BOUNDS[1]} z{WORKSPACE_BOUNDS[2]}")
    print("=" * 60)

    # ── 1. 创建 Planner（在 RobotInterface 之前）──
    arm_config = XArm7Config()

    urdf_path = str(ASSET_DIR / "robots" / "xhand" / "xarm7_xhand_collision.urdf")
    srdf_path = str(ASSET_DIR / "robots" / "xhand" / "xarm7_xhand_collision_mplib.srdf")

    planner = XArm7MotionPlanner(
        XArm7PlannerConfig(
            urdf_path=urdf_path,
            srdf_path=srdf_path,
            base_pose_world=Pose(p=np.array([0.0, 0.0, 0.0]), q=np.array([np.cos(np.pi / 12), 0.0, 0.0, np.sin(np.pi / 12)])),
            collision=COLLISION_CONFIG,
        ),
        planning_profile=PlanningProfile(check_self_collision=False),
        teleop_profile=TeleopProfile(
            teleop_dt=CTRL_DT,
            max_ik_jump_deg=(8, 8, 8, 8, 12, 12, 18),  # tight: reject IK branch switches (keyboard EEF moves 5mm/step → ~0.5° per joint)
            max_pose_error_pos_m=0.01,
            max_pose_error_rot_rad=np.deg2rad(5.0),
        ),
    )

    # ── 2. 连接硬件 ──
    robot = RobotInterface(
        RobotInterfaceConfig(
            arm=arm_config,
            collision=COLLISION_CONFIG,
            hand_urdf_path=str(ASSET_DIR / "robots" / "xhand" / "xhand_right.urdf"),
        ),
        kinematics=planner.kin,
        planner=planner,
    )

    print("\n连接硬件...")
    result = robot.connect()
    print(f"  arm:  {'OK' if result.get('arm') else 'FAIL'}")
    print(f"  hand: {'OK' if result.get('hand') else 'FAIL (降级运行)'}")

    if not result.get("arm"):
        print("arm 连接失败，退出")
        return

    # 重置 soft-start 计数器 — 确保前 20 帧有限速保护
    robot.arm.reset_soft_start()

    # ── 3. Pre-Flight 检查 ──
    report = preflight_check(robot)
    print_preflight(report)
    if not report.passed:
        print("Pre-Flight 检查失败，退出")
        robot.disconnect()
        return

    # ── 4. 获取当前状态 ──
    state = robot.get_state()
    home_qpos = arm_config.init_qpos.copy()
    prev_qpos_cmd = state.arm_qpos.copy()

    if not np.all(np.isfinite(prev_qpos_cmd)):
        print("当前关节角度无效，使用 init_qpos")
        prev_qpos_cmd = home_qpos.copy()

    target_pos = state.eef_pos.copy()
    target_quat = state.eef_quat_wxyz.copy()

    print(f"\n初始状态:")
    print(f"  arm_qpos:  {np.round(np.rad2deg(state.arm_qpos), 1)} deg")
    print(f"  eef_pos:   {np.round(state.eef_pos, 4)} m")
    print(f"  eef_quat:  {np.round(state.eef_quat_wxyz, 4)}")

    # ── 5. 键盘输入 ──
    keys = GlobalKeyState()
    keys.start()
    print("\n键盘控制已启动，按 Q 退出")

    # ── 6. 主循环 ──
    limiter = RateLimiter(1.0 / CTRL_DT)
    running = True
    wall_warned = [False, False, False]
    last_wall_time = 0.0
    loop_count = 0
    error_count = 0
    max_consecutive_errors = 10

    print("\n进入遥操作循环...\n")

    # Disable terminal echo so WASD keystrokes don't appear in the terminal
    fd = sys.stdin.fileno()
    old_termios = termios.tcgetattr(fd)
    new_termios = termios.tcgetattr(fd)
    new_termios[3] &= ~termios.ECHO
    termios.tcsetattr(fd, termios.TCSANOW, new_termios)

    try:
        while running:
            limiter.wait()
            loop_count += 1

            # ── 退出/急停 ──
            if keys.is_pressed("esc"):
                print("\nESC: emergency_stop")
                robot.emergency_stop()
                running = False
                break

            if keys.is_pressed("q"):
                print("\nQ: 退出")
                running = False
                break

            if keys.is_pressed("r"):
                print("\nR: return_home")
                do_return_home(robot, planner)
                # 重置目标为当前状态
                state = robot.get_state()
                if np.all(np.isfinite(state.arm_qpos)):
                    prev_qpos_cmd = state.arm_qpos.copy()
                    target_pos = state.eef_pos.copy()
                    target_quat = state.eef_quat_wxyz.copy()
                continue

            # ── 读状态 ──
            try:
                state = robot.get_state()
            except Exception as e:
                error_count += 1
                print(f"  get_state 异常: {e}")
                if error_count > max_consecutive_errors:
                    print("连续错误过多，急停退出")
                    robot.emergency_stop()
                    running = False
                    break
                continue

            error_count = 0

            # ── 安全检查 ──
            if robot.arm.is_error():
                # Check SDK-level error code. send_action() now calls
                # get_err_warn_code() on failure, so last_sdk_error_code
                # is fresh; arm.error_code may still be stale from async
                # report callback. Check both.
                arm_code = robot.arm.arm.error_code if robot.arm.arm else 0
                sdk_code = robot.arm.last_sdk_error_code

                # Code 22 = ControllerError (C31 collision / C32 overcurrent).
                # Often false-positive when hand is disconnected (TCP load
                # mismatch) or at workspace boundaries. Attempt recovery
                # before escalating to emergency_stop.
                if arm_code == 22 or sdk_code == 22:
                    print("  ⚠ ControllerError 22 (C31/C32)，清除错误并保持位置", flush=True)
                    robot.arm.clear_error()
                    # Re-read state to reset targets to actual position
                    state = robot.get_state()
                    if np.all(np.isfinite(state.arm_qpos)):
                        target_pos = state.eef_pos.copy()
                        target_quat = state.eef_quat_wxyz.copy()
                        prev_qpos_cmd = state.arm_qpos.copy()
                    error_count = 0
                    continue
                print(f"arm 错误: {robot.arm.last_error_message}")
                robot.emergency_stop()
                running = False
                break

            if not np.all(np.isfinite(state.arm_qpos)):
                error_count += 1
                continue

            # ── EEF 目标增量 ──
            dx = np.zeros(3, dtype=np.float64)

            # 平移: WASD + ↑↓
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

            # 旋转: I/K + ←→ + JL
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

            # 无输入则保持
            if np.all(dx == 0) and np.all(drpy == 0):
                continue

            # ── 软墙: 逐轴拒绝边界外移动，离开时重置警告 ──
            new_pos = target_pos + dx
            for axis in range(3):
                if dx[axis] == 0:
                    continue
                lo, hi = WORKSPACE_BOUNDS[axis]
                if lo <= new_pos[axis] <= hi:
                    target_pos[axis] = new_pos[axis]
                    if wall_warned[axis] and lo + 0.01 <= new_pos[axis] <= hi - 0.01:
                        wall_warned[axis] = False  # 离开边界重置
                else:
                    now = time.perf_counter()
                    if not wall_warned[axis] or now - last_wall_time > 3.0:
                        names = ["x", "y", "z"]
                        print(f"  ⚠ {names[axis]} 轴到达边界 [{lo:.2f}, {hi:.2f}]")
                        wall_warned[axis] = True
                        last_wall_time = now

            # 旋转: dq = drpy_to_quat ⊗ target_quat
            if np.any(drpy != 0):
                dq = rpy_to_quat_wxyz(drpy[0], drpy[1], drpy[2])
                target_quat = quat_multiply(dq, target_quat)

            # ── IK ──
            target_pose = Pose(p=target_pos, q=target_quat)
            ik_result = planner.solve_teleop_ik(target_pose, state.arm_qpos, prev_qpos_cmd)

            if not ik_result.success or ik_result.qpos is None:
                # hold_on_failure：原地不动，回退 target 到实际 EEF 避免累积跳变
                target_pos = state.eef_pos.copy()
                target_quat = state.eef_quat_wxyz.copy()
                continue

            # ── EMA 平滑 + 发送 ──
            arm_cmd = EMA_ALPHA * ik_result.qpos + (1 - EMA_ALPHA) * prev_qpos_cmd
            prev_qpos_cmd = arm_cmd.copy()

            action = RobotAction(
                arm_qpos_cmd=arm_cmd,
                hand_qpos_cmd=state.hand_qpos.copy(),
                target_eef_pos=target_pos.copy(),
            )

            send_result = robot.send_action(action)

            # DEBUG: 每 50 帧 (1s), 使用 post-clip 值对比 pre-clip IK
            if loop_count % 50 == 0:
                post_clip = send_result.get("arm_cmd")
                qpos_delta = (post_clip - state.arm_qpos) if post_clip is not None else np.zeros(7)
                clipped = not np.allclose(arm_cmd, post_clip, atol=1e-4) if post_clip is not None else False
                post_str = f"post={np.round(np.rad2deg(post_clip),1)}" if post_clip is not None else "post=N/A"
                clip_tag = " CLIPPED!" if clipped else " ok"
                print(
                    f"  [DBG {loop_count}] cur={np.round(np.rad2deg(state.arm_qpos),1)} "
                    f"ik={np.round(np.rad2deg(ik_result.qpos),1)} "
                    f"{post_str}{clip_tag} "
                    f"Δ={np.round(np.rad2deg(qpos_delta),1)} "
                    f"eef→{np.round(target_pos,3)}")
            if not send_result.get("arm_ok"):
                error_count += 1
                if error_count > max_consecutive_errors:
                    print("连续发送失败，急停退出")
                    robot.emergency_stop()
                    running = False
                    break

            # 状态打印 (每 100 帧)
            if loop_count % 100 == 0:
                eef = state.eef_pos
                in_ws = "OK" if np.all(
                    (WORKSPACE_BOUNDS[:, 0] <= eef) & (eef <= WORKSPACE_BOUNDS[:, 1])
                ) else "EDGE"
                print(f"  [{loop_count}] eef={np.round(eef, 3)}m  ws={in_ws}", flush=True)

    finally:
        keys.stop()
        time.sleep(0.05)  # let listener thread exit cleanly
        termios.tcflush(fd, termios.TCIFLUSH)  # drain buffered keystrokes
        termios.tcsetattr(fd, termios.TCSADRAIN, old_termios)

        # ── 7. 退出 ──
        print("\n退出主循环")

        # 退出后仍可接收 return_home
        print("\n按 R 执行 return_home，或按 Q 直接退出...")
        while True:
            if keys.is_pressed("r"):
                do_return_home(robot, planner)
                print("按 Q 退出...")
            if keys.is_pressed("q") or keys.is_pressed("esc"):
                break
            time.sleep(0.1)

        robot.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
