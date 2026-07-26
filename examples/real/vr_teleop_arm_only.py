#!/usr/bin/env python3
"""真机 VR 遥操作 xArm7 (仅机械臂，灵巧手可降级跳过)。

机械臂通过 VR wrist pose 控制 EEF 位姿，灵巧手在不可用时自动降级跳过。

架构:
    Meta Quest (HTS app) ──TCP──→ VRReceiverProcess ──SharedMemory──→ 主循环
                                     (独立进程, 隔离 HTS SDK)          (主进程, CTRL_HZ 决策; 臂内环 30Hz)

用法:
    # 1. Quest USB 有线: adb reverse tcp:8000 tcp:8000
    # 2. 启动:
    source /home/zhy/anaconda3/etc/profile.d/conda.sh && conda activate real
    python examples/real/vr_teleop_arm_only.py

控制:
    B    开始遥操作 + 录制 (记录当前 wrist→EEF 映射)
    C    暂停/恢复 (toggle, 保持当前位置, 录制继续)
    S    停止录制 (自动保存)
    H    return_home (归位, 自动保存)
    Q    退出
    ESC  急停
"""

from __future__ import annotations

import atexit
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy.spatial.transform import Rotation

from dexmani_real import ASSET_DIR
from dexmani_real.planning import PlanningProfile, Pose, TeleopProfile, XArm7MotionPlanner, XArm7PlannerConfig
from dexmani_real.planning.collision_config import CollisionConfig
from dexmani_real.planning.pose_utils import quat_wxyz_to_rot6d, wxyz_to_xyzw
from dexmani_real.recording.episode_recorder import EpisodeRecorder
from dexmani_real.robot.arm_process import ArmServo, make_arm_servo
from dexmani_real.robot.inner_loop import ArmInnerLoopConfig
from dexmani_real.robot.interface import RobotAction, RobotInterface, RobotInterfaceConfig
from dexmani_real.robot.preflight import PreFlightReport, preflight_check, print_preflight
from dexmani_real.robot.validate import validate_action
from dexmani_real.robot.xarm7 import XArm7Config
from dexmani_real.sensor.vr_receiver_process import VRReceiverConfig, VRReceiverProcess
from dexmani_real.teleop.control.keyboard import ControlSignal, KeyboardHandler
from dexmani_real.teleop.vr.arm_mapper import ArmWristMapper
from dexmani_real.utils.array_utils import nan_array
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate_manager import RateManager
from dexmani_real.utils.signal_utils import EMA_ALPHA_POS, EMA_ALPHA_ROT, ema_smooth_pose
from dexmani_real.utils.trajectory_logger import TrajectoryLogger

logger = get_logger(__name__)

# ═══════════════════════════════════════════════ Keyboard (pynput, 全局捕获)


# ═══════════════════════════════════════════════ TrajectoryLogger 已提取至 dexmani_real.utils.trajectory_logger


# ═══════════════════════════════════════════════ 配置

# ── 控制频率: 单点定义, 其余常量全部由此派生 ──
# 决策/录制 @ CTRL_HZ; 臂内环保持 30Hz (Mode 6 固件在线规划, 直发无插值)。
CTRL_HZ = 16.0
CTRL_DT = 1.0 / CTRL_HZ
REF_HZ = 50.0  # 零空间步长参数的原调参频率
STATUS_EVERY = int(round(1.0 * CTRL_HZ))  # 状态打印节流 (~1Hz)
HOME_DT = 0.04  # 归位 waypoint 间隔 (s)

WORKSPACE_BOUNDS = np.array(
    [
        [0.24, 0.72],  # x [min, max] m
        [-0.50, 0.50],  # y [min, max] m
        [0.05, 0.5],  # z [min, max] m
    ],
    dtype=np.float64,
)

COLLISION_CONFIG = CollisionConfig(
    table_z_world=0.0,
    hand_extension_below_eef=0.076,
    hand_safe_margin=0.03,
)

# VR wrist → EEF 映射参数
VR_POS_SCALE = 1.0  # 位置缩放 (1.0 = 1:1)
VR_ROT_SCALE = 1.0  # 旋转缩放 (1.0 = 1:1)
VR_MAX_DELTA_ROT_RAD = 3.0  # 距复位点总旋转增量上限 (~172°, 由 ArmWristMapper.max_delta_rot_rad 使用)
VR_STALE_THRESHOLD_S = 0.5  # VR 帧超时阈值 (过期则保持当前位置)

# 笛卡尔位姿 EMA 平滑 (IK 前, 唯一平滑级, 匹配 sim TeleopPipeline)
# 参数定义在 dexmani_real.utils.signal_utils (EMA_ALPHA_POS/EMA_ALPHA_ROT)

# Mode 6 online trajectory planning — firmware respects speed/accel limits (120°/s, 500°/s²).
ARM_MAX_SPEED_DEG_S = 120.0  # 首次上机需低速验收 (C22/C24 与 tracking 告警频次)
_INNER_CFG = ArmInnerLoopConfig(joint_max_speed=float(np.deg2rad(ARM_MAX_SPEED_DEG_S)))
ARM_CMD_MAX_STEP_RAD = float(np.deg2rad(ARM_MAX_SPEED_DEG_S)) * CTRL_DT  # 命令级限速步长 (0.131 rad/拍 @120°/s,16Hz)


# ═══════════════════════════════════════════════ 归位


def do_return_home(
    robot: RobotInterface,
    planner: XArm7MotionPlanner,
    arm_inner: ArmServo,
) -> ArmServo:
    """归位: 停止内环 → 规划+执行 → 重启内环."""
    print("return_home ...", flush=True)
    try:
        arm_inner.set_target(None)
        arm_inner.stop()
        print("  Arm 内环线程已停止")

        ok = robot.return_to_home(home_dt=HOME_DT)
        print(f"  {'OK' if ok else 'FAIL'}")

        new_inner = make_arm_servo(cfg=_INNER_CFG)
        robot.set_arm_servo(new_inner)
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


def record_held_frame(recorder, state, hold_arm, vr_frame, *, ik_ok: bool) -> None:
    """录制 held 帧: 跳过发送的帧仍占用栅格槽位, 如实标记 flags (防止回填伪造 ik_ok/held)."""
    if vr_frame is None:  # VR 过期/丢失 — NaN 位姿如实标记无数据
        vr_frame = {
            "wrist_pos": np.full(3, np.nan),
            "wrist_quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0]),
            "landmarks": np.full((21, 3), np.nan),
        }
    hand = state.hand_qpos.copy() if np.all(np.isfinite(state.hand_qpos)) else np.zeros(12, dtype=np.float64)
    action = RobotAction(arm_qpos_cmd=hold_arm, hand_qpos_cmd=hand)
    recorder.add_frame(state, action, vr_frame, signals={"ik_ok": ik_ok, "retarget_ok": False, "held": True})  # retarget_ok=False: hand retargeting not wired


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
        hand_side="both",  # "both" needed for HeadFrame (heading calibration)
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
            use_position_ik=True,
            max_pose_error_pos_m=0.02,
            max_pose_error_rot_rad=np.deg2rad(5.0),
            # 1°/frame @50Hz — 换算保持 °/s 不变
            nullspace_step_size_deg=1.0 * (REF_HZ / CTRL_HZ),
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

    # 清除上次运行可能遗留的错误状态 (C31/C32 等)
    robot.arm.clear_error()

    hand_available = bool(result.get("hand"))

    # ── 4. Pre-Flight ──
    report = preflight_check(robot)
    print_preflight(report)
    if not report.passed:
        print("Pre-Flight 检查失败，退出")
        robot.disconnect()
        vr_receiver.stop()
        return

    # ── 5. ArmInnerLoop (30Hz online trajectory planning) ──
    arm_inner = make_arm_servo(cfg=_INNER_CFG)
    robot.set_arm_servo(arm_inner)
    arm_inner.start()
    print("Arm 内环线程启动中...")
    sys.stdout.flush()

    if not arm_inner.wait_ready(timeout=30.0):
        print("Arm 内环线程启动超时，降级为直接读取")
        arm_qpos = None
    else:
        print("Arm 内环线程已就绪 (30Hz online trajectory planning)")
        arm_qpos, error_state, _ = arm_inner.get_state()

    if arm_qpos is None or not np.all(np.isfinite(arm_qpos)):
        arm_qpos = None

    state = robot.get_state(arm_qpos=arm_qpos)
    home_qpos = arm_config.init_qpos.copy()
    prev_qpos_cmd = state.arm_qpos.copy()

    if not np.all(np.isfinite(prev_qpos_cmd)):
        print("当前关节角度无效，使用 init_qpos")
        prev_qpos_cmd = home_qpos.copy()

    ema_prev_pos: np.ndarray | None = None  # 笛卡尔 EMA 状态 (IK 前)
    ema_prev_quat: np.ndarray | None = None

    print(f"\n初始状态:")
    print(f"  arm_qpos:  {np.round(np.rad2deg(state.arm_qpos), 1)} deg")
    print(f"  eef_pos:   {np.round(state.eef_pos, 4)} m")
    print(f"  eef_quat:  {np.round(state.eef_quat_wxyz, 4)}")

    # ── 6. VR Arm Mapper ──
    # vr_to_base_rot = I: FLU delta 直接当 world delta 用
    # base_to_world_rot = I: 不做额外旋转
    arm_mapper = ArmWristMapper(
        pos_scale=VR_POS_SCALE,
        rot_scale=VR_ROT_SCALE,
        max_delta_rot_rad=VR_MAX_DELTA_ROT_RAD,
    )

    # ── 7. Recorder ──
    recorder = EpisodeRecorder(
        data_dir="episodes",
        max_frames=int(round(90.0 * CTRL_HZ)),  # 90s 上限
        control_hz=CTRL_HZ,
        min_frames=int(round(1.0 * CTRL_HZ)),  # ≥1s 才算有效 episode
    )

    # ── 7b. Trajectory logger (wrist + EEF motion debug) ──
    traj_logger = TrajectoryLogger()
    _traj_save_dir = Path("trajectories")
    _traj_save_dir.mkdir(parents=True, exist_ok=True)
    _traj_path = str(_traj_save_dir / f"traj_{time.strftime('%Y%m%d_%H%M%S')}.npz")

    # ── 8. Keyboard ──
    kb = KeyboardHandler()
    kb.start()
    atexit.register(kb.stop)

    # ── 9. 等待 VR 首帧 ──
    print("\n等待 VR 帧... (确保 Quest 已连接并启动 HTS App)")
    print("  Q=退出")
    startup_deadline = time.perf_counter() + 120.0
    last_diag_ts = 0.0
    while time.perf_counter() < startup_deadline:
        startup_sigs = {s for s in kb.poll(timeout=0.0)}
        if ControlSignal.QUIT in startup_sigs or ControlSignal.EMERGENCY_STOP in startup_sigs:
            print("\nQ/ESC: 退出")
            arm_inner.set_target(None)
            arm_inner.stop()
            robot.disconnect()
            vr_receiver.stop()
            return
        frame = vr_receiver.read_latest()
        if frame is not None:
            print(f"  收到首帧 seq={frame.get('sequence_id', '?')} — 就绪")
            break
        # Periodic diagnostic (every 5s)
        now = time.perf_counter()
        if now - last_diag_ts > 5.0:
            last_diag_ts = now
            s = vr_receiver.get_stats()
            print(
                f"  [diag] recv={s['received_frames']} ignored={s['ignored_events']} "
                f"err={s['error_frames']} "
                f"sdk_lines={s['sdk_lines_received']} "
                f"sdk_parse_err={s['sdk_parse_errors']} "
                f"sdk_dropped={s['sdk_dropped_lines']} "
                f"running={s['running']} crashed={s['crashed']}"
            )
        time.sleep(0.5)
    else:
        print("  VR 帧超时 (120s) — 退出")
        arm_inner.set_target(None)
        arm_inner.stop()
        robot.disconnect()
        vr_receiver.stop()
        return

    # ── 10. 键盘就绪 (pynput 全局捕获) ──

    print("\n控制: B=开始遥操作+录制 C=暂停 S=停止录制 H=归位 Q=退出 ESC=急停")
    print("等待按键 B 开始遥操作...\n")

    # ── 11. Main loop ──
    limiter = RateManager(CTRL_HZ)  # 绝对期限调度 — tick 锁定录制时间栅格
    running = True
    teleop_active = False
    recording_active = False
    loop_count = 0
    error_count = 0
    max_consecutive_errors = 10
    start_time = time.perf_counter()
    prev_eef_pos: np.ndarray | None = None
    ik_method = "-"

    def _stop_recording(save: bool):
        """停止录制. save=True 保存, save=False 丢弃."""
        nonlocal recording_active
        if recording_active:
            if save:
                n_frames = recorder.frame_count  # capture before stop_episode() resets it
                print("  保存中…", flush=True)
                path = recorder.stop_episode(success=True)
                recorder.join_stop(timeout=60.0)  # 落盘完成后才报结果
                if path:
                    if recorder.stop_error:
                        print(f"  ⚠ 保存失败 ({recorder.stop_error}): {path}  — 文件可能不完整")
                    else:
                        print(f"  录制已保存: {path}  ({n_frames} 帧)")
            else:
                recorder.stop_episode(success=False)
                # 注意: 与 record/record_plus 不同，此入口丢弃不删除文件 —
                # h5 以 success=False 留盘。join 确保文件完整关闭。
                recorder.join_stop(timeout=60.0)
            recording_active = False

    def _emergency_stop():
        """停止内环 + 停止录制 + 急停."""
        print("[TRACE] _emergency_stop() called from:", flush=True)
        traceback.print_stack()
        nonlocal running, teleop_active, recording_active
        teleop_active = False
        if recording_active:
            recorder.stop_episode(success=False)
            recording_active = False
        if arm_inner.is_alive:
            arm_inner.set_target(None)
            arm_inner.stop()
        robot.emergency_stop()
        robot.arm.clear_error()
        running = False

    try:
        while running:
            limiter.wait()
            loop_count += 1

            # ── 按键处理 (KeyboardHandler, 与 sim 一致) ──
            skip_rest = False
            for sig in kb.poll(timeout=0.0):
                if sig == ControlSignal.EMERGENCY_STOP:
                    print("\nESC: emergency_stop")
                    print("[TRACE] emergency_stop triggered from keyboard handler:", flush=True)
                    traceback.print_stack()
                    _emergency_stop()
                    break

                elif sig == ControlSignal.QUIT:
                    print("\nQ: 退出")
                    _stop_recording(save=False)
                    running = False
                    break

                elif sig == ControlSignal.HOME:
                    print("\nH: return_home")
                    _stop_recording(save=True)
                    arm_inner = do_return_home(robot, planner, arm_inner)
                    teleop_active = False
                    arm_mapper.clear()
                    if arm_inner.wait_ready(timeout=30.0):
                        arm_qpos, error_state, _ = arm_inner.get_state()
                        if not error_state and np.all(np.isfinite(arm_qpos)):
                            state = robot.get_state(arm_qpos=arm_qpos)
                            prev_qpos_cmd = state.arm_qpos.copy()
                            ema_prev_pos = ema_prev_quat = None
                            error_count = 0
                            print("  Arm 内环线程重启就绪")
                    else:
                        print("  Arm 内环重启超时，降级为直接读取")
                        state = robot.get_state()
                        if np.all(np.isfinite(state.arm_qpos)):
                            prev_qpos_cmd = state.arm_qpos.copy()
                            ema_prev_pos = ema_prev_quat = None
                        error_count = 0
                    skip_rest = True

                elif sig == ControlSignal.STOP:
                    print("\nS: 停止录制")
                    _stop_recording(save=True)
                    teleop_active = False
                    skip_rest = True

                elif sig == ControlSignal.PAUSE:
                    teleop_active = not teleop_active
                    state_str = "暂停" if not teleop_active else "恢复"
                    print(f"\nC: {state_str}遥操作 (录制{'继续' if recording_active else '已停止'})")
                    if teleop_active:
                        # 恢复时重新建立 wrist→EEF 映射，避免跳跃
                        frame = vr_receiver.read_latest()
                        if frame is not None:
                            state = robot.get_state(arm_qpos=arm_inner.get_state()[0] if arm_inner.is_alive else None)
                            arm_mapper.reset(
                                wrist_pos=frame["wrist_pos"],
                                wrist_quat_wxyz=frame["wrist_quat_wxyz"],
                                eef_pos=state.eef_pos,
                                eef_quat_wxyz=state.eef_quat_wxyz,
                            )
                    skip_rest = True

                elif sig == ControlSignal.BEGIN:
                    # 开始/重置遥操作: 启动录制 + 记录 wrist→EEF 映射
                    frame = vr_receiver.read_latest()
                    if frame is None:
                        print("\nB: 无 VR 帧，无法开始遥操作")
                        skip_rest = True
                        continue
                    # 如果已在录制，先停止旧 episode
                    _stop_recording(save=recording_active)
                    if not recorder.start_episode():
                        print("  ⚠ 无法开始录制（上一 episode 仍在写盘）")
                        skip_rest = True
                        continue
                    recording_active = True
                    state = robot.get_state(arm_qpos=arm_inner.get_state()[0] if arm_inner.is_alive else None)

                    # Heading calibration: align user's facing direction → robot +X
                    head_q = frame.get("head_quat_wxyz")
                    head_p = frame.get("head_pos")
                    head_ok = (
                        head_q is not None
                        and head_p is not None
                        and np.any(np.isfinite(head_q))
                        and np.any(np.array(head_p) != 0)  # non-zero = HeadFrame received
                    )
                    if head_ok:
                        arm_mapper.set_heading(head_q)
                        # Print heading direction for debugging
                        head_rot = Rotation.from_quat(np.roll(np.asarray(head_q), -1))  # wxyz → xyzw
                        fwd = head_rot.apply(np.array([1.0, 0.0, 0.0]))
                        print(f"  heading: forward_2d=[{fwd[0]:.3f}, {fwd[1]:.3f}]")
                    else:
                        print("  heading: head pose unavailable, keeping default (I)")

                    arm_mapper.reset(
                        wrist_pos=frame["wrist_pos"],
                        wrist_quat_wxyz=frame["wrist_quat_wxyz"],
                        eef_pos=state.eef_pos,
                        eef_quat_wxyz=state.eef_quat_wxyz,
                    )
                    teleop_active = True
                    error_count = 0
                    print(f"\nB: 遥操作+录制开始 (wrist→EEF 映射已记录)  episode={recorder.frame_count}")
                    print(f"  wrist_ref={np.round(frame['wrist_pos'], 3)}")
                    print(f"  eef_ref=  {np.round(state.eef_pos, 3)}")
                    skip_rest = True

            if not running:
                break
            if skip_rest:
                continue

            # ── 读取 ArmInnerLoop 状态 ──
            try:
                arm_qpos, error_state, _inner_ts, arm_qvel, arm_tau = arm_inner.get_state_and_dynamics()
                state = robot.get_state(arm_qpos=arm_qpos, arm_qvel=arm_qvel, arm_tau=arm_tau)

                if error_state:
                    print(f"  Arm 内环异常: error_state=True")
                    error_count += 1
                    if error_count > 3:
                        print("Arm 内环连续异常，急停退出")
                        _emergency_stop()
                        break
                    continue
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
                        ema_prev_pos = ema_prev_quat = None
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
            vr_stale = (
                vr_frame is None
                or (time.monotonic_ns() - vr_frame.get("local_recv_ns", 0)) > VR_STALE_THRESHOLD_S * 1e9
            )

            # ── Periodic status ──
            if loop_count % STATUS_EVERY == 0:
                elapsed = time.perf_counter() - start_time
                if prev_eef_pos is not None:
                    vel = np.linalg.norm(state.eef_pos - prev_eef_pos) / (STATUS_EVERY * CTRL_DT)
                else:
                    vel = 0.0
                prev_eef_pos = state.eef_pos.copy()
                vr_age = vr_receiver.frame_age_s() if vr_frame is not None else -1
                # DEBUG: show wrist delta direction vs actual EEF motion
                if teleop_active and vr_frame is not None and arm_mapper.is_ready():
                    _wrist_delta = vr_frame["wrist_pos"] - arm_mapper.wrist_pos0
                    _eef_delta = state.eef_pos - arm_mapper.eef_pos0
                    print(
                        f"[T+{elapsed:.1f}s f={loop_count}] "
                        f"eef={np.round(state.eef_pos, 3)}m  "
                        f"v={vel:.2f}m/s  "
                        f"wrist_d={np.round(_wrist_delta, 3)}  "
                        f"eef_d={np.round(_eef_delta, 3)}  "
                        f"ik={ik_method}",
                        flush=True,
                    )
                else:
                    print(
                        f"[T+{elapsed:.1f}s f={loop_count}] "
                        f"eef={np.round(state.eef_pos, 3)}m  "
                        f"v={vel:.2f}m/s  "
                        f"teleop={'ON' if teleop_active else 'OFF'}  "
                        f"rec={'ON' if recording_active else 'OFF'}  "
                        f"vr_age={vr_age:.3f}s  "
                        f"ik={ik_method}  "
                        f"err={error_count}",
                        flush=True,
                    )

                # ── VR wrist raw pose logging (FLU frame, quat wxyz) ──
                if vr_frame is not None:
                    _wp = vr_frame["wrist_pos"]
                    _wq = vr_frame["wrist_quat_wxyz"]
                    _w_euler = np.rad2deg(Rotation.from_quat(wxyz_to_xyzw(_wq)).as_euler("xyz", degrees=False))
                    print(
                        f"  [VR raw] wrist_pos(FLU)={np.round(_wp, 4)}m  "
                        f"wrist_quat_wxyz={np.round(_wq, 4)}  "
                        f"wrist_euler_xyz={np.round(_w_euler, 1)}°",
                        flush=True,
                    )

            # ── 非遥操作模式: 保持当前位置 ──
            if not teleop_active or vr_stale:
                if not teleop_active:
                    # No action needed — hold current arm position via inner loop
                    pass
                elif vr_stale:
                    if loop_count % STATUS_EVERY == 0:
                        print("  ⚠ VR 帧过期，保持当前位置")
                prev_qpos_cmd = state.arm_qpos.copy()
                ema_prev_pos = ema_prev_quat = None  # 重置笛卡尔 EMA, 恢复时无跳变
                if recording_active:
                    record_held_frame(recorder, state, prev_qpos_cmd.copy(), vr_frame, ik_ok=True)
                continue

            # ── VR wrist → EEF target pose ──
            mapped = arm_mapper.map(vr_frame["wrist_pos"], vr_frame["wrist_quat_wxyz"])
            if mapped is None:
                ik_method = "no_map"
                if recording_active:
                    hold_arm = prev_qpos_cmd.copy() if prev_qpos_cmd is not None else state.arm_qpos.copy()
                    record_held_frame(recorder, state, hold_arm, vr_frame, ik_ok=True)
                continue

            target_pos = mapped["pos"]
            target_quat = mapped["quat_wxyz"]

            # ── Cartesian pose EMA (IK 前, 唯一平滑级) ──
            # 首帧 seed, 后续帧 EMA. IK 失败时冻结 EMA 状态,
            # 防止向不可达目标累积漂移.
            if ema_prev_pos is not None:
                target_pos, target_quat = ema_smooth_pose(
                    target_pos,
                    target_quat,
                    ema_prev_pos,
                    ema_prev_quat,
                    EMA_ALPHA_POS,
                    EMA_ALPHA_ROT,
                )

            # ── Workspace clamp (EMA 之后, 作为最终安全门) ──
            target_pos_before_clamp = target_pos.copy()
            for axis in range(3):
                lo, hi = WORKSPACE_BOUNDS[axis]
                target_pos[axis] = np.clip(target_pos[axis], lo, hi)

            # ── IK solve ──
            target_pose = Pose(p=target_pos, q=target_quat)
            if np.all(np.isfinite(state.hand_qpos)):
                planner.set_hand_qpos(state.hand_qpos)
            ik_result = planner.solve_teleop_ik(target_pose, state.arm_qpos, prev_qpos_cmd)

            # ── Trajectory debug record (before continue on fail) ──
            _wrist_d = vr_frame["wrist_pos"] - arm_mapper.wrist_pos0 if arm_mapper.is_ready() else None
            _eef_d = state.eef_pos - arm_mapper.eef_pos0 if arm_mapper.is_ready() else None
            traj_logger.append(
                t=time.perf_counter() - start_time,
                wrist_pos=vr_frame["wrist_pos"],
                wrist_quat_wxyz=vr_frame["wrist_quat_wxyz"],
                target_pos=target_pos,
                target_quat_wxyz=target_quat,
                actual_eef_pos=state.eef_pos,
                actual_eef_quat_wxyz=state.eef_quat_wxyz,
                arm_qpos_actual=state.arm_qpos,
                ik_ok=ik_result.success and ik_result.qpos is not None,
                wrist_delta=_wrist_d,
                eef_delta=_eef_d,
                target_pos_before_clamp=target_pos_before_clamp,
            )

            if not ik_result.success or ik_result.qpos is None:
                ik_method = "fail"
                if recording_active:
                    hold_arm = prev_qpos_cmd.copy() if prev_qpos_cmd is not None else state.arm_qpos.copy()
                    record_held_frame(recorder, state, hold_arm, vr_frame, ik_ok=False)
                continue

            ik_method = "ok"
            # IK success: update EMA state so the next frame's smoothing starts
            # from a reachable target.  Freezing during failures prevents drift.
            ema_prev_pos = target_pos.copy()
            ema_prev_quat = target_quat.copy()
            # 平滑已在 IK 前 (笛卡尔 EMA) 完成; IK 输出按 joint_max_speed×dt 截步长后下发 —
            # 快腕旋时 IK 目标可超前可达状态 >50° (实测 max 57.5°), 截步长使 action 标签保持动力学可达
            arm_cmd = np.asarray(ik_result.qpos, dtype=np.float64)
            arm_cmd = prev_qpos_cmd + np.clip(arm_cmd - prev_qpos_cmd, -ARM_CMD_MAX_STEP_RAD, ARM_CMD_MAX_STEP_RAD)

            # ── Hand: hold position (skip if hand unavailable) ──
            if hand_available:
                hand_cmd = (
                    state.hand_qpos.copy() if np.all(np.isfinite(state.hand_qpos)) else np.zeros(12, dtype=np.float64)
                )
            else:
                hand_cmd = np.zeros(12, dtype=np.float64)

            action = RobotAction(
                arm_qpos_cmd=arm_cmd,
                hand_qpos_cmd=hand_cmd,
                target_eef_pos=target_pos.copy(),
                target_eef_rot6d=quat_wxyz_to_rot6d(target_quat),
            )

            # ── Pre-send gate: 力矩/温度/软限位 (与 controller.py:346 一致) ──
            action_valid, fail_reason = validate_action(
                robot,
                action,
                actual_arm_qpos=arm_qpos,
                actual_arm_qvel=state.arm_qvel,
                actual_arm_tau=state.arm_tau,
                actual_hand_current=state.hand_current,
            )
            if not action_valid:
                print(f"  [SAFETY] Pre-send gate: {fail_reason} — 跳过本帧", flush=True)
                if recording_active:
                    hold_arm = prev_qpos_cmd.copy() if prev_qpos_cmd is not None else state.arm_qpos.copy()
                    record_held_frame(recorder, state, hold_arm, vr_frame, ik_ok=True)
                continue

            prev_qpos_cmd = arm_cmd.copy()  # only after gate passes (held frames use last-good command)

            # ── Send to ArmInnerLoop ──
            arm_inner.set_target(action.arm_qpos_cmd)
            robot.send_action(action)  # hand only (arm is via inner loop)

            # ── 录制帧 ──
            if recording_active:
                sig = {"ik_ok": True, "retarget_ok": False, "held": False}  # retarget_ok=False: hand retargeting not wired (arm-only teleop)
                if not recorder.add_frame(state, action, vr_frame, signals=sig):
                    # max_frames reached or error
                    if recorder.max_frames_reached:
                        print(f"\n  达到 max_frames={recorder.max_frames}，自动停止录制")
                        _stop_recording(save=True)
                        teleop_active = False

    finally:
        # 确保录制已停止
        if recording_active:
            recorder.stop_episode(success=False)
        # 无条件等待后台 flush 完成（急停/H 保存的 stop 线程也在此收口）。
        # 必须在交互 prompt 之前 — 否则 flush 被 prompt 劫持、二次 Ctrl-C 可截断 h5。
        _join_deadline = time.perf_counter() + 60.0
        while True:
            try:
                recorder.join_stop(timeout=max(1.0, _join_deadline - time.perf_counter()))
                break
            except KeyboardInterrupt:
                if time.perf_counter() > _join_deadline:
                    print("\n  ⚠ 写盘超时", flush=True)
                    break
                print("\n  ⚠ 正在等待写盘完成，请勿中断…", flush=True)
        if recorder.stop_error:
            print(f"  ⚠ 后台写盘失败: {recorder.stop_error}", flush=True)

        # ── 保存轨迹 debug 数据 ──
        if len(traj_logger) > 0:
            try:
                saved = traj_logger.save(_traj_path)
                print(f"\n轨迹已保存: {saved}  ({len(traj_logger)} 帧)")
            except (OSError, ValueError) as e:
                print(f"\n轨迹保存失败: {e}")

        print("\n退出主循环")

        # Post-loop: offer return_home
        print("\n按 H 执行 return_home，或按 Q 直接退出...")
        while True:
            post_sigs = {s for s in kb.poll(timeout=0.1)}
            if ControlSignal.HOME in post_sigs:
                try:
                    arm_inner = do_return_home(robot, planner, arm_inner)
                except Exception:
                    traceback.print_exc()
                    print("  return_home 失败，继续退出")
                print("按 Q 退出...")
            if ControlSignal.QUIT in post_sigs or ControlSignal.EMERGENCY_STOP in post_sigs:
                break

        kb.stop()

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
