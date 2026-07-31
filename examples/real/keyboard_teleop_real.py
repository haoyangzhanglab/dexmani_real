#!/usr/bin/env python3
"""真机键盘遥操作 xArm7 — 复用 ArmInnerLoop + PreFlight + RobotInterface。

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

共享组件:
    ArmInnerLoop   — 内环线程 (robot/inner_loop.py, 30Hz, mode 6 — firmware trajectory planning)
    PreFlight      — 硬件就绪检查 (robot/preflight.py)
    RateManager   — 高精度频率限制 (utils/rate_manager.py)
    RobotInterface — arm/hand 统一接口 (robot/interface.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import termios
import time

import numpy as np

from dexmani_real import ASSET_DIR
from dexmani_real.planning import PlanningProfile, Pose, TeleopProfile, XArm7MotionPlanner, XArm7PlannerConfig

from dexmani_real.planning.pose_utils import quat_multiply
from dexmani_real.robot.arm_process import ArmServo, do_return_home, make_arm_servo
from dexmani_real.teleop.control.keyboard import GlobalKeyState
from dexmani_real.robot.inner_loop import ArmInnerLoopConfig
from dexmani_real.robot.interface import RobotAction, RobotInterface, RobotInterfaceConfig
from dexmani_real.robot.preflight import PreFlightReport, preflight_check, print_preflight
from dexmani_real.robot.validate import validate_action
from dexmani_real.robot.xarm7 import XArm7Config
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate_manager import RateManager
from dexmani_real.utils.signal_utils import EMA_ALPHA_POS, EMA_ALPHA_ROT, ema_smooth_pose
from scipy.spatial.transform import Rotation as R

try:
    from pynput import keyboard  # type: ignore[import-untyped]
except ImportError:
    raise ImportError("pynput is required for keyboard input. Install with: pip install pynput")

logger = get_logger(__name__)

# ═══════════════════════════════════════════════ 配置

CTRL_DT = 0.02  # 50Hz
DELTA_POS = 0.005  # 每次按键 EEF 平移量 (m)
DELTA_RPY = 0.02  # 每次按键 EEF 旋转量 (rad)
# EMA_ALPHA_POS / EMA_ALPHA_ROT imported from dexmani_real.utils.signal_utils

# Cartesian P-term: amplifies position error before IK to reduce steady-state
# tracking lag.  At Kp=0.0 the arm lags ~50mm behind the target at 250 mm/s
# (open-loop time constant τ ≈ 0.2 s).  Kp=0.3 reduces the steady-state error
# by ~23 % (to ~38 mm); Kp=0.5 by ~33 % (to ~33 mm).  Higher values risk
# overshoot on direction reversals.  Set 0.0 for pure open-loop behaviour.
CARTESIAN_KP = 0.0  # conservative default; try 0.3–0.5 for less lag

# Mode 6 online trajectory planning — firmware replans trajectory with
# configurable speed/acc limits. No inner-loop interpolation.
INNER_LOOP_CFG = ArmInnerLoopConfig()
HOME_DT = 0.04  # 归位 waypoint 间隔 (s): ~25°/s (默认 0.02→~50°/s，减半保安全)

# ── Motion tracing: 追踪纯 +X 运动时的位置变化管线 ──
TRACE_MOTION = True  # 启用运动追踪
TRACE_FRAME_INTERVAL = 10  # 每 N 帧打印一次 (避免刷屏)

WORKSPACE_BOUNDS = np.array(
    [
        [0.28, 0.72],  # x [min, max] m
        [-0.45, 0.45],  # y [min, max] m
        [0.05, 0.5],  # z [min, max] m
    ],
    dtype=np.float64,
)


# ═══════════════════════════════════════════════ 姿态工具




# ═══════════════════════════════════════════════ 辅助函数


def _print_motion_trace(
    loop_count: int,
    dx: np.ndarray,
    target_pos: np.ndarray,
    ik_target_pos: np.ndarray,
    eef_pos: np.ndarray,
    ik_fk_pos: np.ndarray,
    ik_fk_quat: np.ndarray,
    ik_target_quat: np.ndarray,
    ik_result,
    arm_qpos: np.ndarray,
    report: dict,
) -> None:
    """Print pure-axis (+X or -X) motion trace — target → EMA → IK → FK pipeline."""
    pos_error_mm = float(np.linalg.norm(ik_target_pos - ik_fk_pos) * 1000)
    pos_error_per_axis_mm = (ik_target_pos - ik_fk_pos) * 1000
    dot = float(min(np.abs(np.dot(ik_target_quat, ik_fk_quat)), 1.0))
    rot_error_deg = float(np.rad2deg(2.0 * np.arccos(dot)))
    raw_lead_mm = (target_pos - eef_pos) * 1000
    ema_lead_mm = (ik_target_pos - eef_pos) * 1000
    z_shift_mm = float((ik_fk_pos[2] - ik_target_pos[2]) * 1000)

    print(
        f"\n{'─'*60}"
        f"\n[TRACE #{loop_count}] 纯轴运动 + Cartesian EMA + P-term (Kp={CARTESIAN_KP})"
        f"\n{'─'*60}"
        f"\n  dx:          {np.array2string(dx * 1000, precision=1, suppress_small=True)} mm"
        f"\n  raw target:  {np.array2string(target_pos * 1000, precision=1, suppress_small=True)} mm"
        f"\n  EMA→IK:      {np.array2string(ik_target_pos * 1000, precision=1, suppress_small=True)} mm"
        f"\n  eef:         {np.array2string(eef_pos * 1000, precision=1, suppress_small=True)} mm"
        f"\n  raw lead:    {np.array2string(raw_lead_mm, precision=1, suppress_small=True)} mm"
        f"\n  EMA lead:    {np.array2string(ema_lead_mm, precision=1, suppress_small=True)} mm"
        f"\n  IK FK:       {np.array2string(ik_fk_pos * 1000, precision=1, suppress_small=True)} mm"
        f"\n  IK err:      pos={pos_error_mm:.1f}mm  per_axis={np.array2string(pos_error_per_axis_mm, precision=1, suppress_small=True)} mm  rot={rot_error_deg:.2f}deg"
        f"\n  IK Z off:    {z_shift_mm:+.1f}mm"
        f"\n  IK: {report.get('method', '?')} seed={report.get('seed', '?')} attempts={report.get('attempts', '?')}"
        f"\n  jnt Δ: {np.array2string(np.rad2deg(ik_result.qpos - arm_qpos), precision=2, suppress_small=True)} deg"
        f"\n{'─'*60}",
        flush=True,
    )


def _wall_check(
    axis: int,
    target_pos: np.ndarray,
    workspace_bounds: np.ndarray,
    wall_warned: list,
    wall_timers: list,
) -> None:
    """Debounced workspace-boundary warning — one independent 3 s cooldown per axis."""
    lo, hi = workspace_bounds[axis]
    if target_pos[axis] <= lo or target_pos[axis] >= hi:
        now = time.perf_counter()
        if not wall_warned[axis] or now - wall_timers[axis] > 3.0:
            names = ["x", "y", "z"]
            print(f"  ⚠ {names[axis]} 轴到达边界 [{lo:.2f}, {hi:.2f}]")
            wall_warned[axis] = True
            wall_timers[axis] = now


# ═══════════════════════════════════════════════ 主循环


def main():
    print("=" * 60)
    print("真机键盘遥操作 xArm7")
    print(
        f"  DELTA_POS={DELTA_POS*1000:.0f}mm  DELTA_RPY={np.rad2deg(DELTA_RPY):.1f}deg  CTRL_DT={CTRL_DT}s  EMA_POS={EMA_ALPHA_POS} EMA_ROT={EMA_ALPHA_ROT}"
    )
    print(f"  workspace: x{WORKSPACE_BOUNDS[0]} y{WORKSPACE_BOUNDS[1]} z{WORKSPACE_BOUNDS[2]}")
    print("=" * 60)

    # ── 1. Planner ──
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

        ),
        planning_profile=PlanningProfile(
            max_waypoint_delta_deg=360.0,
        ),
        teleop_profile=TeleopProfile(
            max_pose_error_pos_m=0.02,
            max_pose_error_rot_rad=np.deg2rad(5.0),
        ),
    )

    # ── 2. Robot connection ──
    robot = RobotInterface(
        RobotInterfaceConfig(
            arm=arm_config,

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

    # ── 3. Pre-Flight ── (shared: robot/preflight.py)
    report = preflight_check(robot)
    print_preflight(report)
    if not report.passed:
        print("Pre-Flight 检查失败，退出")
        robot.disconnect()
        return

    # ── 4. Start ArmInnerLoop (shared: robot/inner_loop.py) ──
    arm_inner = make_arm_servo(cfg=INNER_LOOP_CFG)
    robot.set_arm_servo(arm_inner)
    arm_inner.start()
    print("Arm 内环线程启动中...")
    sys.stdout.flush()

    if not arm_inner.wait_ready(timeout=30.0):
        print("Arm 内环线程启动超时，降级为直接读取")
        arm_qpos = None
    else:
        print("Arm 内环线程已就绪 (30Hz online trajectory planning, passthrough)")
        arm_qpos, error_state, _ = arm_inner.get_state()

    if arm_qpos is None or not np.all(np.isfinite(arm_qpos)):
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

    # ── 5. Keyboard input ──
    keys = GlobalKeyState()
    keys.start()
    print("\n键盘控制已启动，按 Q 退出")

    # ── 6. Main loop ──
    limiter = RateManager(1.0 / CTRL_DT)
    running = True
    wall_warned = [False, False, False]
    wall_timers = [0.0, 0.0, 0.0]  # per-axis debounce (independent 3 s cooldown)
    loop_count = 0
    error_count = 0
    max_consecutive_errors = 10
    consecutive_divergence = 0
    TRACKING_DIVERGENCE_THRESHOLD_RAD = 5.0
    start_time = time.perf_counter()
    prev_eef_pos: np.ndarray | None = None
    ik_outcome = "-"
    ik_fail_count = 0
    _last_ik_fail_reason = ""
    _last_ik_fail_time = 0.0

    # Cartesian EMA state (same smoothing as TeleopPipeline)
    _prev_ema_pos: np.ndarray | None = None
    _prev_ema_quat: np.ndarray | None = None

    def _emergency_stop():
        """Stop inner loop first, then arm+hand, to avoid SDK error spam."""
        nonlocal running
        if arm_inner.is_alive:
            arm_inner.set_target(None)
            arm_inner.stop()
        robot.emergency_stop()
        running = False

    print("\n进入遥操作循环...\n")

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
                _emergency_stop()
                break

            if keys.is_pressed("q"):
                print("\nQ: 退出")
                running = False
                break

            if keys.is_pressed("r"):
                print("\nR: return_home")
                arm_inner = do_return_home(robot, arm_inner, INNER_LOOP_CFG)
                # Wait for new inner loop to be ready
                if arm_inner.wait_ready(timeout=30.0):
                    arm_qpos, error_state, _ = arm_inner.get_state()
                    if not error_state and np.all(np.isfinite(arm_qpos)):
                        state = robot.get_state(arm_qpos=arm_qpos)
                        prev_qpos_cmd = state.arm_qpos.copy()
                        target_pos = state.eef_pos.copy()
                        target_quat = state.eef_quat_wxyz.copy()
                        _prev_ema_pos = _prev_ema_quat = None  # reset EMA on re-engage
                        consecutive_divergence = 0
                        error_count = 0
                        print("  Arm 内环线程重启就绪")
                else:
                    print("  Arm 内环重启超时，降级为直接读取")
                    state = robot.get_state()
                    if np.all(np.isfinite(state.arm_qpos)):
                        prev_qpos_cmd = state.arm_qpos.copy()
                        target_pos = state.eef_pos.copy()
                        target_quat = state.eef_quat_wxyz.copy()
                        _prev_ema_pos = _prev_ema_quat = None  # reset EMA on re-engage
                    consecutive_divergence = 0
                    error_count = 0
                prev_eef_pos = None
                ik_outcome = "-"
                limiter.reset()
                continue

            # ── Read state from inner loop ──
            try:
                arm_qpos, error_state, _inner_ts, arm_qvel, arm_tau = arm_inner.get_state_and_dynamics()
                state = robot.get_state(arm_qpos=arm_qpos, arm_qvel=arm_qvel, arm_tau=arm_tau)

                if error_state:
                    print(f"  Arm 内环异常: error_state=True")
                    if error_count > 3:
                        print("Arm 内环连续异常，急停退出")
                        _emergency_stop()
                        break
                    error_count += 1
                    continue
            except Exception as e:
                error_count += 1
                print(f"  get_state 异常: {e}")
                if error_count > max_consecutive_errors:
                    print("连续错误过多，急停退出")
                    _emergency_stop()
                    break
                continue

            error_count = 0

            # ── Safety: arm error ──
            if robot.arm.is_connected() and robot.arm.is_error():
                arm_code = robot.arm.arm.error_code if robot.arm.arm else 0
                sdk_code = robot.arm.last_sdk_error_code

                if arm_code == 22 or sdk_code == 22:
                    print("  ⚠ ControllerError 22 (C31/C32)，清除错误并保持位置", flush=True)
                    robot.arm.clear_error()
                    state = robot.get_state()
                    if np.all(np.isfinite(state.arm_qpos)):
                        target_pos = state.eef_pos.copy()
                        target_quat = state.eef_quat_wxyz.copy()
                        prev_qpos_cmd = state.arm_qpos.copy()
                    error_count = 0
                    continue
                print(f"arm 错误: C{arm_code}")
                _emergency_stop()
                break

            if not np.all(np.isfinite(state.arm_qpos)):
                error_count += 1
                continue

            # ── EEF target delta from keys ──
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

            # ── Periodic status ──
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
                    f"v={vel:.2f}m/s  ik={ik_outcome}  err={error_count}",
                    flush=True,
                )

            # No input → snap target to EEF, reset EMA state
            if np.all(dx == 0) and np.all(drpy == 0):
                target_pos = state.eef_pos.copy()
                target_quat = state.eef_quat_wxyz.copy()
                prev_qpos_cmd = state.arm_qpos.copy()
                _prev_ema_pos = _prev_ema_quat = None  # reset EMA on re-engage
                continue

            # ── Incremental target ──
            # target_pos accumulates keyboard deltas independently.
            # Uncommanded axes keep their last-set value → no cross-axis drift.
            for axis in range(3):
                if dx[axis] != 0:
                    target_pos[axis] += dx[axis]

            # Workspace boundary: clamp target to valid range
            target_pos = np.clip(target_pos, WORKSPACE_BOUNDS[:, 0], WORKSPACE_BOUNDS[:, 1])
            for axis in range(3):
                _wall_check(axis, target_pos, WORKSPACE_BOUNDS, wall_warned, wall_timers)

            if np.any(drpy != 0):
                dq = R.from_euler('xyz', drpy).as_quat(scalar_first=True)
                target_quat = quat_multiply(dq, target_quat)

            # ── Cartesian EMA (before IK, same as TeleopPipeline) ──
            # Smooths target trajectory to prevent IK discontinuities from
            # abrupt keypress changes.  At α=0.8 the EMA steady-state lag
            # is ~(1-α)/α × Δ ≈ 1.25 mm — negligible; the dominant 50 mm
            # lead comes from arm tracking dynamics, which the EMA does not
            # address (see CARTESIAN_KP below for that).
            if _prev_ema_pos is not None:
                ik_target_pos, ik_target_quat = ema_smooth_pose(
                    target_pos,
                    target_quat,
                    _prev_ema_pos,
                    _prev_ema_quat,
                    EMA_ALPHA_POS,
                    EMA_ALPHA_ROT,
                )
            else:
                ik_target_pos, ik_target_quat = target_pos.copy(), target_quat.copy()
            _prev_ema_pos = ik_target_pos.copy()
            _prev_ema_quat = ik_target_quat.copy()

            # ── Cartesian P-term: amplify position error → reduce tracking lag ──
            # Without feedback the arm lags ~50 mm behind the target at 250 mm/s
            # (open-loop time constant τ ≈ 0.2 s).  Adding Kp × pos_error to the
            # IK target effectively reduces the time constant by 1/(1+Kp):
            #   Kp=0.0 → 50 mm lag   Kp=0.3 → ~38 mm (−23 %)
            #   Kp=0.5 → ~33 mm (−33 %)   Kp=1.0 → ~25 mm (−50 %)
            # Also counteracts Z-axis coupling during pure-X motion.
            if CARTESIAN_KP > 0:
                pos_error = target_pos - state.eef_pos
                if float(np.linalg.norm(pos_error)) > 0.003:  # 3 mm deadband
                    ik_target_pos = ik_target_pos + CARTESIAN_KP * pos_error
                    ik_target_pos = np.clip(
                        ik_target_pos, WORKSPACE_BOUNDS[:, 0], WORKSPACE_BOUNDS[:, 1],
                    )

            # ── IK solve (on EMA-smoothed target) ──
            target_pose = Pose(p=ik_target_pos, q=ik_target_quat)
            if np.all(np.isfinite(state.hand_qpos)):
                planner.set_hand_qpos(state.hand_qpos)
            ik_result = planner.solve_teleop_ik(target_pose, state.arm_qpos, prev_qpos_cmd)

            if not ik_result.success or ik_result.qpos is None:
                ik_fail_count += 1
                reason = getattr(ik_result, "reason", "") or "unknown"
                now = time.perf_counter()
                if reason != _last_ik_fail_reason or now - _last_ik_fail_time > 1.0:
                    print(f"  ⚡ IK fail (#{ik_fail_count}): {reason}", flush=True)
                    _last_ik_fail_reason = reason
                    _last_ik_fail_time = now
                target_pos = state.eef_pos.copy()
                target_quat = state.eef_quat_wxyz.copy()
                _prev_ema_pos = _prev_ema_quat = None  # reset EMA on IK failure
                ik_outcome = "held"
                continue

            ik_outcome = "ok"
            prev_qpos_cmd = ik_result.qpos.copy()
            arm_cmd = ik_result.qpos

            # ── Motion Trace: 纯轴运动管线诊断 ──
            if (
                TRACE_MOTION
                and loop_count % TRACE_FRAME_INTERVAL == 0
                and dx[0] != 0
                and dx[1] == 0
                and dx[2] == 0
                and np.all(drpy == 0)
            ):
                ik_fk_pose = planner.kin.compute_eef_pose_world(ik_result.qpos)
                _print_motion_trace(
                    loop_count=loop_count,
                    dx=dx,
                    target_pos=target_pos,
                    ik_target_pos=ik_target_pos,
                    eef_pos=state.eef_pos,
                    ik_fk_pos=ik_fk_pose.p,
                    ik_fk_quat=ik_fk_pose.q,
                    ik_target_quat=ik_target_quat,
                    ik_result=ik_result,
                    arm_qpos=state.arm_qpos,
                    report=getattr(ik_result, "report", {}) or {},
                )

            # ── Send ──
            # Arm: via inner loop (30Hz, mode 6 — firmware trajectory planning)
            # Hand: via robot.send_action() (hold position)
            action = RobotAction(
                arm_qpos_cmd=arm_cmd,
                hand_qpos_cmd=state.hand_qpos.copy(),
            )

            # ── Pre-send gate: 力矩/温度/软限位 (与 controller.py:346 一致) ──
            action_valid, fail_reason = validate_action(
                robot,
                action,
                actual_arm_qvel=state.arm_qvel,
                actual_arm_tau=state.arm_tau,
            )
            if not action_valid:
                print(f"  [SAFETY] Pre-send gate: {fail_reason} — 跳过本帧", flush=True)
                continue

            arm_inner.set_target(action.arm_qpos_cmd)
            robot.send_action(action)  # hand only

            # ── Tracking safety ──
            # Compare actual position against the delta-clamped command the inner
            # loop actually sent to the SDK (not the raw IK output, which may
            # have been clamped by ArmInnerLoop's per-step delta limit).
            if np.all(np.isfinite(state.arm_qpos)):
                sent_cmd = arm_inner.last_sent_cmd
                tracking_err = np.max(np.abs(state.arm_qpos - sent_cmd)) if sent_cmd is not None else 0.0
                if tracking_err > TRACKING_DIVERGENCE_THRESHOLD_RAD:
                    consecutive_divergence += 1
                    print(
                        f"  [SAFETY] Tracking divergence: max_err={tracking_err:.1f}rad "
                        f"(frame {consecutive_divergence}/3)"
                    )
                    if consecutive_divergence >= 3:
                        print("  [SAFETY] Emergency stop — persistent tracking divergence")
                        _emergency_stop()
                        break
                else:
                    consecutive_divergence = 0

    finally:
        # Restore terminal first (pynput uses evdev, not termios — no conflict).
        time.sleep(0.05)
        termios.tcflush(fd, termios.TCIFLUSH)
        termios.tcsetattr(fd, termios.TCSADRAIN, old_termios)

        print("\n退出主循环")

        # Post-loop: offer return_home (keys listener still alive)
        print("\n按 R 执行 return_home，或按 Q 直接退出...")
        while True:
            if keys.is_pressed("r"):
                arm_inner = do_return_home(robot, arm_inner, INNER_LOOP_CFG)
                print("按 Q 退出...")
            if keys.is_pressed("q") or keys.is_pressed("esc"):
                break
            time.sleep(0.1)

        keys.stop()

        # ── Cleanup ──
        if arm_inner.is_alive:
            arm_inner.set_target(None)
            arm_inner.stop()
            print("Arm 内环线程已停止")

        robot.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
