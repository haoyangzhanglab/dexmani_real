#!/usr/bin/env python3
"""真机 VR 遥操作 xArm7 (仅机械臂，灵巧手可降级跳过)。

机械臂通过 VR wrist pose 控制 EEF 位姿，灵巧手在不可用时自动降级跳过。

架构:
    Meta Quest (HTS app) ──TCP──→ VRReceiverProcess ──SharedMemory──→ 主循环
                                     (独立进程, 隔离 HTS SDK)          (主进程, 50Hz)

用法:
    # 1. Quest USB 有线: adb reverse tcp:8000 tcp:8000
    # 2. 启动:
    source /home/zhy/anaconda3/etc/profile.d/conda.sh && conda activate real
    python examples/real/vr_teleop_arm_only.py

控制:
    T    开始/重置遥操作 (记录当前 wrist→EEF 映射)
    R    return_home (归位)
    Q    退出
    ESC  急停
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import termios
import threading
import time
import traceback

import numpy as np

from dexmani_real import ASSET_DIR
from dexmani_real.utils.log import get_logger
from dexmani_real.planning import (
    PlanningProfile,
    Pose,
    TeleopProfile,
    XArm7MotionPlanner,
    XArm7PlannerConfig,
)
from dexmani_real.planning.collision_config import CollisionConfig
from dexmani_real.robot.inner_loop import ArmInnerLoop
from dexmani_real.robot.interface import RobotAction, RobotInterface, RobotInterfaceConfig
from dexmani_real.robot.preflight import PreFlightReport, preflight_check, print_preflight
from dexmani_real.robot.xarm7 import XArm7Config
from dexmani_real.sensor.vr_receiver_process import VRReceiverConfig, VRReceiverProcess
from dexmani_real.teleop.vr.arm_mapper import ArmWristMapper
from dexmani_real.utils.array_utils import nan_array
from dexmani_real.utils.rate_limiter import RateLimiter

logger = get_logger(__name__)

# ═══════════════════════════════════════════════ 配置

CTRL_DT = 0.02           # 50Hz
HOME_DT = 0.04           # 归位 waypoint 间隔 (s)

WORKSPACE_BOUNDS = np.array([
    [0.28, 0.72],    # x [min, max] m
    [-0.45, 0.45],   # y [min, max] m
    [0.05, 0.5],     # z [min, max] m
], dtype=np.float64)

COLLISION_CONFIG = CollisionConfig(
    table_z_world=0.0,
    hand_extension_below_eef=0.076,
    hand_safe_margin=0.03,
)

# VR wrist → EEF 映射参数
VR_POS_SCALE = 1.0          # 位置缩放 (1.0 = 1:1)
VR_ROT_SCALE = 1.0          # 旋转缩放 (1.0 = 1:1)
VR_MAX_DELTA_ROT_RAD = 1.0  # 每帧旋转增量上限 (~57°)
VR_STALE_THRESHOLD_S = 0.5  # VR 帧超时阈值 (过期则保持当前位置)


# ═══════════════════════════════════════════════ 键盘输入 (非阻塞)


class GlobalKeyState:
    """非阻塞键盘状态追踪 (pynput, 线程安全) — 用于触发键检测."""

    def __init__(self):
        self._keys: set[str] = set()
        self._running = True
        self._thread: threading.Thread | None = None
        self._listener = None

    def _run(self):
        from pynput import keyboard

        def on_press(key):
            try:
                if hasattr(key, "char") and key.char is not None:
                    self._keys.add(key.char.lower())
                elif key == keyboard.Key.esc:
                    self._keys.add("esc")
            except Exception:
                pass

        def on_release(key):
            try:
                if hasattr(key, "char") and key.char is not None:
                    self._keys.discard(key.char.lower())
                elif key == keyboard.Key.esc:
                    self._keys.discard("esc")
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
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def is_pressed(self, key: str) -> bool:
        return key in self._keys

    def consume(self, key: str) -> bool:
        """检测并消费按键 (防重复触发)."""
        if key in self._keys:
            self._keys.discard(key)
            return True
        return False


# ═══════════════════════════════════════════════ 归位


def do_return_home(
    robot: RobotInterface,
    planner: XArm7MotionPlanner,
    arm_inner: ArmInnerLoop,
) -> ArmInnerLoop:
    """归位: 停止内环 → 规划+执行 → 重启内环."""
    print("return_home ...", flush=True)
    try:
        arm_inner.set_target(None)
        arm_inner.stop()
        print("  Arm 内环线程已停止")

        ok = robot.return_to_home(home_dt=HOME_DT)
        print(f"  {'OK' if ok else 'FAIL'}")

        new_inner = ArmInnerLoop()
        new_inner.start()
        print("  Arm 内环线程已重启")
        return new_inner
    except Exception:
        traceback.print_exc()
        print("  return_to_home 异常，尝试 emergency_stop")
        arm_inner.set_target(None)
        arm_inner.stop()
        robot.emergency_stop()
        raise


# ═══════════════════════════════════════════════ 主循环


def main():
    print("=" * 60)
    print("VR 遥操作 xArm7 (仅机械臂)")
    print(f"  VR scale: pos={VR_POS_SCALE}  rot={VR_ROT_SCALE}")
    print(f"  workspace: x{WORKSPACE_BOUNDS[0]} y{WORKSPACE_BOUNDS[1]} z{WORKSPACE_BOUNDS[2]}")
    print("=" * 60)

    # ── 1. VR Receiver (独立进程) ──
    vr_config = VRReceiverConfig(
        transport="tcp_server",
        host="0.0.0.0",
        port=8000,
        hand_side="right",
        output_frame="flu",
        max_frame_age_s=0.20,
    )
    vr_receiver = VRReceiverProcess(config=vr_config)
    vr_receiver.start()
    print(f"VRReceiverProcess 已启动 ({vr_config.transport}://{vr_config.host}:{vr_config.port})")

    # ── 2. Planner ──
    arm_config = XArm7Config()
    urdf_path = str(ASSET_DIR / "robots" / "xhand" / "xarm7_xhand_collision.urdf")
    srdf_path = str(ASSET_DIR / "robots" / "xhand" / "xarm7_xhand.srdf")

    planner = XArm7MotionPlanner(
        XArm7PlannerConfig(
            urdf_path=urdf_path,
            srdf_path=srdf_path,
            base_pose_world=Pose(
                p=np.array([0.0, 0.0, 0.0]),
                q=np.array([np.cos(np.pi / 12), 0.0, 0.0, np.sin(np.pi / 12)]),
            ),
            collision=COLLISION_CONFIG,
        ),
        planning_profile=PlanningProfile(),
        teleop_profile=TeleopProfile(
            teleop_dt=CTRL_DT,
            use_position_ik=True,
            max_pose_error_pos_m=0.02,
            max_pose_error_rot_rad=np.deg2rad(5.0),
            differential_ik_max_pos_step_m=0.05,
        ),
    )

    # ── 3. Robot ──
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
    print(f"  hand: {'OK' if result.get('hand') else 'FAIL (降级运行 — 仅机械臂控制)'}")

    if not result.get("arm"):
        print("arm 连接失败，退出")
        vr_receiver.stop()
        return

    hand_available = bool(result.get("hand"))

    # ── 4. Pre-Flight ──
    report = preflight_check(robot)
    print_preflight(report)
    if not report.passed:
        print("Pre-Flight 检查失败，退出")
        robot.disconnect()
        vr_receiver.stop()
        return

    # ── 5. ArmInnerLoop (250Hz position servo) ──
    arm_inner = ArmInnerLoop()
    arm_inner.start()
    print("Arm 内环线程启动中...")
    sys.stdout.flush()

    if not arm_inner.wait_ready(timeout=30.0):
        print("Arm 内环线程启动超时，降级为直接读取")
        arm_qpos = None
    else:
        print("Arm 内环线程已就绪 (250Hz position servo)")
        arm_qpos, error_state, _ = arm_inner.get_state()

    if arm_qpos is None or not np.all(np.isfinite(arm_qpos)) or np.all(arm_qpos == 0):
        arm_qpos = None

    state = robot.get_state(arm_qpos=arm_qpos)
    home_qpos = arm_config.init_qpos.copy()
    prev_qpos_cmd = state.arm_qpos.copy()

    if not np.all(np.isfinite(prev_qpos_cmd)):
        print("当前关节角度无效，使用 init_qpos")
        prev_qpos_cmd = home_qpos.copy()

    print(f"\n初始状态:")
    print(f"  arm_qpos:  {np.round(np.rad2deg(state.arm_qpos), 1)} deg")
    print(f"  eef_pos:   {np.round(state.eef_pos, 4)} m")
    print(f"  eef_quat:  {np.round(state.eef_quat_wxyz, 4)}")

    # ── 6. VR Arm Mapper ──
    # vr_to_base_rot: VR 坐标系 (FLU: x=forward, y=left, z=up)
    # → 机械臂 base 坐标系 (x=forward, y=left, z=up)
    # FLU 与 base 对齐时为单位阵，如实际方向不同在此调整。
    arm_mapper = ArmWristMapper(
        pos_scale=VR_POS_SCALE,
        rot_scale=VR_ROT_SCALE,
        max_delta_rot_rad=VR_MAX_DELTA_ROT_RAD,
    )

    # ── 7. 等待 VR 首帧 ──
    print("\n等待 VR 帧...")
    startup_deadline = time.perf_counter() + 30.0
    while time.perf_counter() < startup_deadline:
        frame = vr_receiver.read_latest()
        if frame is not None:
            print(f"  收到首帧 seq={frame.get('sequence_id', '?')} — 就绪")
            break
        time.sleep(0.1)
    else:
        print("  VR 帧超时 — 退出")
        arm_inner.set_target(None)
        arm_inner.stop()
        robot.disconnect()
        vr_receiver.stop()
        return

    # ── 8. Keyboard input ──
    keys = GlobalKeyState()
    keys.start()
    print("\n控制: T=开始遥操作 R=归位 Q=退出 ESC=急停")
    print("等待按键 T 开始遥操作...\n")

    # ── 9. Main loop ──
    limiter = RateLimiter(1.0 / CTRL_DT)
    running = True
    teleop_active = False
    loop_count = 0
    error_count = 0
    max_consecutive_errors = 10
    start_time = time.perf_counter()
    prev_eef_pos: np.ndarray | None = None
    ik_method = "-"

    # Disable terminal echo for clean output
    fd = sys.stdin.fileno()
    old_termios = termios.tcgetattr(fd)
    new_termios = termios.tcgetattr(fd)
    new_termios[3] &= ~termios.ECHO
    termios.tcsetattr(fd, termios.TCSANOW, new_termios)

    def _emergency_stop():
        """停止内环 + 急停."""
        nonlocal running, teleop_active
        teleop_active = False
        if arm_inner.is_alive:
            arm_inner.set_target(None)
            arm_inner.stop()
        robot.emergency_stop()
        running = False

    try:
        while running:
            limiter.wait()
            loop_count += 1

            # ── 按键处理 ──
            if keys.consume("esc"):
                print("\nESC: emergency_stop")
                _emergency_stop()
                break

            if keys.consume("q"):
                print("\nQ: 退出")
                running = False
                break

            if keys.consume("r"):
                print("\nR: return_home")
                arm_inner = do_return_home(robot, planner, arm_inner)
                teleop_active = False
                arm_mapper.clear()
                if arm_inner.wait_ready(timeout=30.0):
                    arm_qpos, error_state, _ = arm_inner.get_state()
                    if not error_state and np.all(np.isfinite(arm_qpos)) and not np.all(arm_qpos == 0):
                        state = robot.get_state(arm_qpos=arm_qpos)
                        prev_qpos_cmd = state.arm_qpos.copy()
                        error_count = 0
                        print("  Arm 内环线程重启就绪")
                else:
                    print("  Arm 内环重启超时，降级为直接读取")
                    state = robot.get_state()
                    if np.all(np.isfinite(state.arm_qpos)):
                        prev_qpos_cmd = state.arm_qpos.copy()
                    error_count = 0
                continue

            if keys.consume("t"):
                # 开始/重置遥操作: 记录当前 wrist → EEF 映射
                frame = vr_receiver.read_latest()
                if frame is None:
                    print("\nT: 无 VR 帧，无法开始遥操作")
                    continue
                state = robot.get_state(arm_qpos=arm_inner.get_state()[0] if arm_inner.is_alive else None)
                arm_mapper.reset(
                    wrist_pos=frame["wrist_pos"],
                    wrist_quat_wxyz=frame["wrist_quat_wxyz"],
                    eef_pos=state.eef_pos,
                    eef_quat_wxyz=state.eef_quat_wxyz,
                )
                teleop_active = True
                error_count = 0
                print(f"\nT: 遥操作开始 (wrist→EEF 映射已记录)")
                print(f"  wrist_ref={np.round(frame['wrist_pos'], 3)}")
                print(f"  eef_ref=  {np.round(state.eef_pos, 3)}")
                continue

            # ── 读取 ArmInnerLoop 状态 ──
            try:
                arm_qpos, error_state, _inner_ts = arm_inner.get_state()

                if error_state:
                    print(f"  Arm 内环异常: error_state=True")
                    error_count += 1
                    if error_count > 3:
                        print("Arm 内环连续异常，急停退出")
                        _emergency_stop()
                        break
                    continue

                state = robot.get_state(arm_qpos=arm_qpos)
            except Exception as e:
                error_count += 1
                print(f"  get_state 异常: {e}")
                if error_count > max_consecutive_errors:
                    print("连续错误过多，急停退出")
                    _emergency_stop()
                    break
                continue

            # ── Arm error check ──
            if robot.arm.is_error():
                arm_code = robot.arm.arm.error_code if robot.arm.arm else 0
                sdk_code = robot.arm.last_sdk_error_code

                if arm_code == 22 or sdk_code == 22:
                    print("  ⚠ ControllerError 22 (C31/C32)，清除错误并保持位置", flush=True)
                    robot.arm.clear_error()
                    state = robot.get_state()
                    if np.all(np.isfinite(state.arm_qpos)):
                        prev_qpos_cmd = state.arm_qpos.copy()
                    error_count = 0
                    continue
                print(f"arm 错误: C{arm_code}")
                _emergency_stop()
                break

            if not np.all(np.isfinite(state.arm_qpos)):
                error_count += 1
                continue

            error_count = 0

            # ── 读取 VR 帧 ──
            vr_frame = vr_receiver.read_latest()
            vr_stale = vr_frame is None or (time.monotonic_ns() - vr_frame.get("local_recv_ns", 0)) > VR_STALE_THRESHOLD_S * 1e9

            # ── Periodic status ──
            if loop_count % 50 == 0:
                elapsed = time.perf_counter() - start_time
                if prev_eef_pos is not None:
                    vel = np.linalg.norm(state.eef_pos - prev_eef_pos) / (50 * CTRL_DT)
                else:
                    vel = 0.0
                prev_eef_pos = state.eef_pos.copy()
                vr_age = vr_receiver.frame_age_s() if vr_frame is not None else -1
                print(
                    f"[T+{elapsed:.1f}s f={loop_count}] "
                    f"eef={np.round(state.eef_pos, 3)}m  "
                    f"v={vel:.2f}m/s  "
                    f"teleop={'ON' if teleop_active else 'OFF'}  "
                    f"vr_age={vr_age:.3f}s  "
                    f"ik={ik_method}  "
                    f"err={error_count}",
                    flush=True,
                )

            # ── 非遥操作模式: 保持当前位置 ──
            if not teleop_active or vr_stale:
                if not teleop_active:
                    # No action needed — hold current arm position via inner loop
                    pass
                elif vr_stale:
                    if loop_count % 50 == 0:
                        print("  ⚠ VR 帧过期，保持当前位置")
                prev_qpos_cmd = state.arm_qpos.copy()
                continue

            # ── VR wrist → EEF target pose ──
            mapped = arm_mapper.map(vr_frame["wrist_pos"], vr_frame["wrist_quat_wxyz"])
            if mapped is None:
                ik_method = "no_map"
                continue

            target_pos = mapped["pos"]
            target_quat = mapped["quat_wxyz"]

            # ── Workspace clamp ──
            for axis in range(3):
                lo, hi = WORKSPACE_BOUNDS[axis]
                target_pos[axis] = np.clip(target_pos[axis], lo, hi)

            # ── IK solve ──
            target_pose = Pose(p=target_pos, q=target_quat)
            if np.all(np.isfinite(state.hand_qpos)):
                planner.set_hand_qpos(state.hand_qpos)
            ik_result = planner.solve_teleop_ik(target_pose, state.arm_qpos, prev_qpos_cmd)

            if not ik_result.success or ik_result.qpos is None:
                ik_method = "fail"
                continue

            ik_method = "ok"
            prev_qpos_cmd = ik_result.qpos.copy()
            arm_cmd = ik_result.qpos

            # ── Send to ArmInnerLoop ──
            arm_inner.set_target(arm_cmd)

            # ── Hand: hold position (skip if hand unavailable) ──
            if hand_available:
                hand_cmd = state.hand_qpos.copy() if np.all(np.isfinite(state.hand_qpos)) else np.zeros(12, dtype=np.float64)
            else:
                hand_cmd = np.zeros(12, dtype=np.float64)

            action = RobotAction(
                arm_qpos_cmd=arm_cmd,
                hand_qpos_cmd=hand_cmd,
                target_eef_pos=target_pos.copy(),
            )
            robot.send_action(action)  # hand only (arm is via inner loop)

    finally:
        keys.stop()
        time.sleep(0.05)
        termios.tcflush(fd, termios.TCIFLUSH)
        termios.tcsetattr(fd, termios.TCSADRAIN, old_termios)

        print("\n退出主循环")

        # Post-loop: offer return_home
        print("\n按 R 执行 return_home，或按 Q 直接退出...")
        while True:
            if keys.is_pressed("r"):
                arm_inner = do_return_home(robot, planner, arm_inner)
                print("按 Q 退出...")
            if keys.is_pressed("q") or keys.is_pressed("esc"):
                break
            time.sleep(0.1)

        # ── Cleanup ──
        if arm_inner.is_alive:
            arm_inner.set_target(None)
            arm_inner.stop()
            print("Arm 内环线程已停止")

        robot.disconnect()
        vr_receiver.stop()
        print("Done.")


if __name__ == "__main__":
    main()
