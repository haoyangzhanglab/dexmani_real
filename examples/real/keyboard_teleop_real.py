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
    - solve_teleop_ik 迭代 DLS (ref: BunnyVisionPro) + MPlib 位置 IK 兜底 (use_position_ik=True)
    - 速度平滑由 PID 内环 _clip_arm_velocity 统一负责
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
from dexmani_real.robot.pid_process import PIDProcess
from dexmani_real.robot.xarm7 import XArm7Config
from dexmani_real.shm.pid_channels import PIDStateChannel, PIDTargetChannel
from dexmani_real.planning.collision_config import CollisionConfig

from examples._test_utils import quat_multiply

# ═══════════════════════════════════════════════ 配置

CTRL_DT = 0.02       # 50Hz
DELTA_POS = 0.005    # 每次按键 EEF 平移量 (m) — 用于 idle→移动过渡
DELTA_RPY = 0.02     # 每次按键 EEF 旋转量 (rad)
TARGET_LEAD_MAX = 0.03  # target 领先 arm 的最大距离 (m)，超过则限速
# 外环键盘脚本只负责累积 target + 安全边界，不做滤波/裁剪。
# Target Lead Governor (MAX_LEAD / chase_pos) 已删除 —
# BunnyVisionPro 不存在该机制，VR 目标位姿直接入 IK。

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


def do_return_home(
    robot: RobotInterface,
    planner: XArm7MotionPlanner,
    pid_process: PIDProcess | None = None,
    pid_target: PIDTargetChannel | None = None,
) -> PIDProcess | None:
    """执行 return_home（停止 PID 进程 → 归位 → 重启 PID 进程）。

    Returns:
        新的 PIDProcess 实例（已启动），或 None（如果原 pid_process 为 None）。
    """
    print("return_home ...", flush=True)
    try:
        # Stop PID process before return_to_home to avoid mode conflicts
        # (PID process uses mode 4 velocity control; return_to_home uses mode 1 position servo)
        if pid_process is not None and pid_process.is_alive():
            if pid_target is not None:
                pid_target.write(None)  # signal deceleration
            pid_process.stop()
            print("  PID process stopped for return-to-home")

        ok = robot.return_to_home()
        print(f"  {'OK' if ok else 'FAIL'}")

        # Restart PID process
        if pid_process is not None:
            new_pid = PIDProcess()
            new_pid.start()
            print("  PID process restarted")
            return new_pid
        return None
    except Exception:
        traceback.print_exc()
        print("  return_to_home 异常，尝试 emergency_stop")
        robot.emergency_stop()
        return None


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
            use_position_ik=True,           # MPlib 兜底: DLS 迭代不收敛时接管
            max_pose_error_pos_m=0.02,      # 2cm FK gate (tight enough for teleop)
            max_pose_error_rot_rad=np.deg2rad(5.0),  # 5° FK gate
            differential_ik_max_pos_step_m=0.05,  # 5cm final-step cap (ref: BVP DLS)
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

    # ── 3. Pre-Flight 检查 ──
    report = preflight_check(robot)
    print_preflight(report)
    if not report.passed:
        print("Pre-Flight 检查失败，退出")
        robot.disconnect()
        return

    # ── 3.5. 启动 PID 进程隔离 ──
    # PIDProcess 拥有独立 XArmAPI 连接，Main 通过共享内存通信
    #   PIDTargetChannel: Main → PID (arm 目标位置)
    #   PIDStateChannel:  PID → Main (arm 当前状态)
    pid_target = PIDTargetChannel(create=True)
    pid_state = PIDStateChannel(create=True)
    pid_process = PIDProcess()
    pid_process.start()
    print("PID 进程启动中...")
    sys.stdout.flush()

    # 轮询等待 PID 进程就绪（XArmAPI 连接 + 模式切换需数秒）
    startup_deadline = time.perf_counter() + 30.0
    arm_qpos = None
    while time.perf_counter() < startup_deadline:
        arm_qpos, error_state, pid_ts = pid_state.read()
        if not error_state and pid_ts > 0 and np.all(np.isfinite(arm_qpos)) and not np.all(arm_qpos == 0):
            print("PID 进程已就绪 (250Hz velocity control)")
            break
        if error_state:
            print("  PID 进程错误，重试中...")
        time.sleep(0.1)

    if arm_qpos is None or np.all(arm_qpos == 0):
        # Fallback: read from robot directly
        print("PID 进程启动超时，降级为直接读取")
        arm_qpos = None
    state = robot.get_state(arm_qpos=arm_qpos)
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
    # ── 追踪安全 (Phase 1.1) — cmd-vs-actual 偏差监控 ──
    consecutive_divergence = 0
    TRACKING_DIVERGENCE_THRESHOLD_RAD = 5.0
    # ── 周期性摘要变量 ──
    start_time = time.perf_counter()
    prev_eef_pos = None  # type: np.ndarray | None
    ik_method = "-"

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
                # 记录旧 PID 最后的 timestamp，等待新 PID 写入 fresh data
                _old_ts = pid_state.read()[2]
                new_pid = do_return_home(robot, planner, pid_process, pid_target)
                if new_pid is not None:
                    pid_process = new_pid
                # 轮询等待新 PID 进程写入第一帧（pid_ts 发生变化）
                startup_deadline = time.perf_counter() + 30.0
                while time.perf_counter() < startup_deadline:
                    arm_qpos, error_state, pid_ts = pid_state.read()
                    if not error_state and pid_ts != _old_ts and np.all(np.isfinite(arm_qpos)) and not np.all(arm_qpos == 0):
                        state = robot.get_state(arm_qpos=arm_qpos)
                        prev_qpos_cmd = state.arm_qpos.copy()
                        target_pos = state.eef_pos.copy()
                        target_quat = state.eef_quat_wxyz.copy()
                        consecutive_divergence = 0
                        error_count = 0  # reset stale counter on successful restart
                        print("  PID 进程重启就绪")
                        break
                    time.sleep(0.1)
                else:
                    # 超时：降级为直接读取
                    print("  PID 重启超时，降级为直接读取")
                    state = robot.get_state()
                    if np.all(np.isfinite(state.arm_qpos)):
                        prev_qpos_cmd = state.arm_qpos.copy()
                        target_pos = state.eef_pos.copy()
                        target_quat = state.eef_quat_wxyz.copy()
                    consecutive_divergence = 0
                    error_count = 0
                continue

            # ── 读状态 (从 PID 进程获取 arm_qpos，robot.get_state 获取 hand/EFF FK) ──
            try:
                arm_qpos, error_state, pid_ts = pid_state.read()
                now = time.perf_counter()

                # PID 进程存活检测: 100ms 无更新 → 异常
                if error_state or (now - pid_ts) > 0.1:
                    print(f"  PID 进程异常: error={error_state} stale_ms={(now-pid_ts)*1000:.0f}")
                    if error_count > 3:
                        print("PID 连续异常，急停退出")
                        robot.emergency_stop()
                        running = False
                        break
                    error_count += 1
                    continue

                state = robot.get_state(arm_qpos=arm_qpos)
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

            # ── 周期性摘要 ──

            if loop_count % 50 == 0:
                elapsed = time.perf_counter() - start_time
                if prev_eef_pos is not None:
                    vel = np.linalg.norm(state.eef_pos - prev_eef_pos) / (50 * CTRL_DT)
                else:
                    vel = 0.0
                prev_eef_pos = state.eef_pos.copy()
                print(
                    f"[T+{elapsed:.1f}s f={loop_count}] "
                    f"eef={np.round(state.eef_pos, 3)}m  "
                    f"target={np.round(target_pos, 3)}  "
                    f"v={vel:.2f}m/s  "
                    f"ik={ik_method}  err={error_count}",
                    flush=True,
                )

            # 无输入 → 重置 target 到当前 EEF 位置（避免累积偏移导致反向运动）
            if np.all(dx == 0) and np.all(drpy == 0):
                target_pos = state.eef_pos.copy()
                target_quat = state.eef_quat_wxyz.copy()
                prev_qpos_cmd = state.arm_qpos.copy()
                continue

            # ── target lead 限幅: 不让 target_pos 领先实际 EEF 太远 ──
            lead = np.linalg.norm(target_pos - state.eef_pos)
            if lead > TARGET_LEAD_MAX:
                target_pos = state.eef_pos + (target_pos - state.eef_pos) * (TARGET_LEAD_MAX / lead)

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

            # ── IK ──（target_pos 直接作为 IK 输入，ref: BunnyVisionPro）
            target_pose = Pose(p=target_pos, q=target_quat)
            ik_result = planner.solve_teleop_ik(target_pose, state.arm_qpos, prev_qpos_cmd)

            if not ik_result.success or ik_result.qpos is None:
                # hold_on_failure: 同时回退 target_pos 和 target_quat。
                # DLS 和 MPlib 都失败说明 target pose 从当前构型不可达，
                # 直接 snap 回实际 EEF 状态给 IK 一个可行的恢复起点。
                target_pos = state.eef_pos.copy()
                target_quat = state.eef_quat_wxyz.copy()
                ik_method = "held"
                continue

            ik_method = "diff"
            # ── 发送 ──
            # Arm: 写入 PIDTargetChannel → PIDProcess 250Hz 速度控制
            #      平滑/jerk限幅/软启动由 PID 内环统一负责
            # Hand: 直接通过 robot.send_action() 发送
            prev_qpos_cmd = ik_result.qpos.copy()
            arm_cmd = ik_result.qpos  # 直接透传

            pid_target.write(arm_cmd)

            action = RobotAction(
                arm_qpos_cmd=arm_cmd,
                hand_qpos_cmd=state.hand_qpos.copy(),
                target_eef_pos=target_pos.copy(),
            )
            robot.send_action(action)  # hand only

            # ── 追踪安全: |q_actual - q_cmd| 偏差监控 ──
            if np.all(np.isfinite(state.arm_qpos)):
                tracking_err = np.max(np.abs(state.arm_qpos - arm_cmd))
                if tracking_err > TRACKING_DIVERGENCE_THRESHOLD_RAD:
                    consecutive_divergence += 1
                    print(
                        f"  [SAFETY] Tracking divergence: max_err={tracking_err:.1f}rad "
                        f"(frame {consecutive_divergence}/3)"
                    )
                    if consecutive_divergence >= 3:
                        print("  [SAFETY] Emergency stop — persistent tracking divergence")
                        robot.emergency_stop()
                        running = False
                        break
                else:
                    consecutive_divergence = 0



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
                new_pid = do_return_home(robot, planner, pid_process, pid_target)
                if new_pid is not None:
                    pid_process = new_pid
                print("按 Q 退出...")
            if keys.is_pressed("q") or keys.is_pressed("esc"):
                break
            time.sleep(0.1)

        # ── PID 进程清理 ──
        if pid_process is not None and pid_process.is_alive():
            if pid_target is not None:
                pid_target.write(None)  # signal deceleration
            pid_process.stop()
            print("PID 进程已停止")

        # ── 共享内存清理 ──
        if pid_target is not None:
            pid_target.unlink()
            pid_target.close()
        if pid_state is not None:
            pid_state.unlink()
            pid_state.close()

        robot.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
