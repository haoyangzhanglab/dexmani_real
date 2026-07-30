#!/usr/bin/env python3
"""真机 VR 遥操作 xArm7 + XHand (臂+手协同) + 数据录制 + 语音播报。

相比 vr_teleop_arm_only_record_plus，额外接入 VR 手部 landmarks → XHand 关节
retargeting (DexPilot NLP 优化)，录制时记录 retarget_ok / retarget_fail 信号。

机械臂通过 VR wrist pose 控制 EEF 位姿，灵巧手通过 VR 21 点 landmarks 控制
12-DOF 关节。手不可用时自动降级为 hold-position。

架构:
    Meta Quest (HTS app) ──TCP──→ VRReceiverProcess ──SharedMemory──→ 主循环
                                     (独立进程, 隔离 HTS SDK)          (主进程, CTRL_HZ 决策; 臂内环 30Hz)

用法:
    # 1. Quest USB 有线: adb reverse tcp:8000 tcp:8000
    # 2. EtherCAT 权限 (手部): sudo setcap cap_net_raw+ep <python_path>
    # 3. 启动:
    source /home/zhy/anaconda3/etc/profile.d/conda.sh && conda activate real
    python examples/real/vr_teleop_hand_record.py

控制:
    B    开始遥操作 + 录制 (记录当前 wrist→EEF 映射, 重置手部 retargeter)
    C    暂停/恢复 (toggle, 保持当前位置, 录制继续)
    S    停止录制 (自动保存)
    D    丢弃录制 (Discard, 不保存)
    H    return_home (归位, 自动保存)
    Q    退出
    ESC  急停
"""

from __future__ import annotations

import atexit
import gc
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy.spatial.transform import Rotation

from dexmani_real import ASSET_DIR
from dexmani_real.sensor.camera_process import create_camera_session
from dexmani_real.planning import PlanningProfile, Pose, TeleopProfile, XArm7MotionPlanner, XArm7PlannerConfig

from dexmani_real.planning.pose_utils import normalize_quat_wxyz, quat_wxyz_to_rot6d, wxyz_to_xyzw
from dexmani_real.recording.episode_recorder import EpisodeRecorder
from dexmani_real.robot.arm_process import ArmServo, do_return_home, make_arm_servo
from dexmani_real.robot.inner_loop import ArmInnerLoopConfig
from dexmani_real.robot.interface import RobotAction, RobotInterface, RobotInterfaceConfig
from dexmani_real.robot.preflight import preflight_check, print_preflight
from dexmani_real.robot.validate import validate_action
from dexmani_real.robot.xarm7 import XArm7Config

from dexmani_real.sensor.vr_receiver_process import VRReceiverConfig, VRReceiverProcess
from dexmani_real.teleop.control.audio_feedback import AudioFeedback
from dexmani_real.teleop.control.keyboard import ControlSignal, KeyboardHandler
from dexmani_real.teleop.vr.arm_mapper import ArmWristMapper
from dexmani_real.teleop.vr.hand_retarget import XHandRetargeter
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.loop_timing import StageTimer
from dexmani_real.utils.rate_manager import RateManager
from dexmani_real.utils.signal_utils import EMA_ALPHA_POS, EMA_ALPHA_ROT, ema_smooth_pose

logger = get_logger(__name__)

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
        [0.28, 0.72],  # x [min, max] m
        [-0.50, 0.50],  # y [min, max] m
        [0.05, 0.5],  # z [min, max] m
    ],
    dtype=np.float64,
)

# VR wrist → EEF 映射参数
VR_POS_SCALE = 1.0  # 位置缩放 (1.0 = 1:1)
VR_ROT_SCALE = 1.0  # 旋转缩放 (1.0 = 1:1)
VR_MAX_DELTA_ROT_RAD = 3.0  # 距复位点总旋转增量上限 (~172°, 由 ArmWristMapper.max_delta_rot_rad 使用)
VR_STALE_THRESHOLD_S = 0.5  # VR 帧超时阈值 (过期则保持当前位置)

# 笛卡尔位姿 EMA 平滑 (IK 前, 唯一平滑级, 匹配 sim TeleopPipeline)
# 参数定义在 dexmani_real.utils.signal_utils (EMA_ALPHA_POS/EMA_ALPHA_ROT)

# ── 手部 retargeting ──
HAND_RETARGETING_TYPE = "dexpilot"  # DexPilot NLP 优化 (YAML: assets/retargeting/xhand_right_dexpilot.yml)

# ── 元数据 (写入 HDF5 /meta, 用于下游数据集管理) ──
TASK_LABEL = ""
OPERATOR = ""

# Mode 6 online trajectory planning — firmware respects speed/accel limits.
# acc 900°/s²: 从默认 500 提升以降低跟踪误差 (sim 验证 acc→1146 改善 32%; 真机 900 保守步进)
ARM_MAX_SPEED_DEG_S = 120.0
ARM_MAX_ACC_DEG_S2 = 900.0
_INNER_CFG = ArmInnerLoopConfig(
    joint_max_speed=float(np.deg2rad(ARM_MAX_SPEED_DEG_S)),
    joint_max_acc=float(np.deg2rad(ARM_MAX_ACC_DEG_S2)),
    loop_period=1.0 / 30.0,  # 30Hz — Mode 6 firmware handles interpolation
)
ARM_CMD_MAX_STEP_RAD = float(np.deg2rad(ARM_MAX_SPEED_DEG_S)) * CTRL_DT  # 命令级限速步长 (0.131 rad/拍 @120°/s,16Hz)

# IK output safety gate: reject IK solutions with joint-space delta from physical
# arm position exceeding this threshold.  Must be > ARM_CMD_MAX_STEP_RAD (0.131)
# to allow normal tracking lag, < anomalous hard cap (0.50 rad).
# Default disabled — enable after empirical validation (no false rejects in normal teleop).
IK_OUTPUT_SAFETY_THRESHOLD_RAD = 0.45
IK_OUTPUT_GATE_ENABLED = False

# VR wrist quaternion continuity gate: reject frames where the angular distance
# between consecutive wrist quaternions exceeds this threshold.  Human wrist max
# is ~500°/s (31°/frame @ 16 Hz) for elite athletes; 45°/frame (~720°/s) gives
# 45% margin and catches VR tracking glitches (typically 100°+/frame).
VR_WRIST_QUAT_JUMP_THRESHOLD_RAD: float = np.deg2rad(45)

# Frame quality codes for HDF5 flag_frame_status field (schema v11).
# Written alongside the legacy flags for backward-compatible filtering.
_FRAME_OK = 0
_FRAME_HELD = 1
_FRAME_IK_FAIL = 2
_FRAME_SAFETY_REJECT = 3
_FRAME_RETARGET_FAIL = 4  # IK succeeded but hand retargeting failed



# ═══════════════════════════════════════════════ 录制辅助函数


def record_held_frame(
    recorder,
    state,
    hold_arm,
    vr_frame,
    cam,
    *,
    ik_ok: bool,
    retarget_ok: bool = False,
    arm_qpos_sent: np.ndarray | None = None,
    hold_hand: np.ndarray | None = None,
    safety_reject: bool = False,
    diagnostics: dict | None = None,
    frame_status: int = _FRAME_HELD,
    ik_attempted: bool = False,
    target_eef_pos: np.ndarray | None = None,
    target_eef_rot6d: np.ndarray | None = None,
) -> None:
    """录制 held 帧: 跳过发送的帧仍占用栅格槽位, 如实标记 flags.

    arm_qpos_sent: 上一帧实发值 (arm_inner.last_sent_cmd) — held 帧不发送新目标,
    但记录侧应反映"上一帧实发值"而非零, 否则 --source=sent replay 会发送错误指令。

    target_eef_pos / target_eef_rot6d: 最后有效目标位姿 — held 帧无新目标,
    但记录此值可保持 action_arm_ee 列的连续性, 避免 replay 遇到 NaN。
    调用方传入上一帧成功 solve 的值 (或 None, 写 NaN)。
    """
    if vr_frame is None:  # VR 过期/丢失 — NaN 位姿如实标记无数据
        vr_frame = {
            "wrist_pos": np.full(3, np.nan),
            "wrist_quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0]),
            "landmarks": np.full((21, 3), np.nan),
        }
    hand = (
        hold_hand.copy()
        if hold_hand is not None
        else (state.hand_qpos.copy() if np.all(np.isfinite(state.hand_qpos)) else np.zeros(12, dtype=np.float64))
    )
    action = RobotAction(
        arm_qpos_cmd=hold_arm,
        hand_qpos_cmd=hand,
        target_eef_pos=target_eef_pos.copy() if target_eef_pos is not None else None,
        target_eef_rot6d=target_eef_rot6d.copy() if target_eef_rot6d is not None else None,
    )
    recorder.add_frame(
        state,
        action,
        vr_frame,
        camera_frame=cam,
        arm_qpos_sent=arm_qpos_sent,
        signals={
            "ik_ok": ik_ok,
            "ik_attempted": ik_attempted,
            "retarget_ok": retarget_ok,
            "held": True,
            "flag_safety_reject": safety_reject,
            "frame_status": frame_status,
        },
        diagnostics=diagnostics,
    )


# ═══════════════════════════════════════════════ 手部命令计算


def compute_hand_command(
    retargeter: XHandRetargeter | None,
    vr_frame: dict,
    prev_hand_cmd: np.ndarray,
    hand_available: bool,
) -> tuple[np.ndarray, bool]:
    """Compute hand joint command from VR landmarks via DexPilot retargeting.

    Returns (hand_cmd, retarget_ok).  On failure or hand unavailable,
    returns prev_hand_cmd unchanged with retarget_ok=False.
    """
    if not hand_available:
        return prev_hand_cmd.copy(), False

    if retargeter is None:
        return prev_hand_cmd.copy(), False

    landmarks = vr_frame.get("landmarks") if vr_frame is not None else None
    if landmarks is None or landmarks.shape != (21, 3):
        return prev_hand_cmd.copy(), False
    if not np.all(np.isfinite(landmarks)):
        return prev_hand_cmd.copy(), False

    try:
        target = retargeter.retarget(landmarks)
        if target is not None and len(target) == 12:
            return np.asarray(target, dtype=np.float64), True
    except Exception:
        logger.warning("Hand retargeting failed — holding position", exc_info=True)

    return prev_hand_cmd.copy(), False


# ═══════════════════════════════════════════════ 主循环


def main():
    print("=" * 60)
    print("VR 遥操作 xArm7 + XHand (臂+手协同)")
    print(f"  VR scale: pos={VR_POS_SCALE}  rot={VR_ROT_SCALE}")
    print(f"  hand retargeting: {HAND_RETARGETING_TYPE}")
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
        ),
        planning_profile=PlanningProfile(),
        teleop_profile=TeleopProfile(
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
            hand_urdf_path=str(ASSET_DIR / "robots" / "xhand" / "xhand_right.urdf"),
        ),
        kinematics=planner.kin,
        planner=planner,
    )

    print("\n连接硬件...")
    result = robot.connect()
    print(f"  arm:  {'OK' if result.get('arm') else 'FAIL'}")
    print(f"  hand: {'OK' if result.get('hand') else 'FAIL (降级运行 — 手部 hold-position 仅)'}")

    if not result.get("arm"):
        print("arm 连接失败，退出")
        vr_receiver.stop()
        return

    # 清除上次运行可能遗留的错误状态 (C31/C32 等)
    robot.arm.clear_error()

    hand_available = bool(result.get("hand"))
    if not hand_available:
        print("  ⚠ 手部不可用 — 本集手部命令将 hold-position (不控制手指运动)")

    # ── 3b. Hand Retargeter (VR landmarks → XHand 12-DOF joints) ──
    hand_retargeter: XHandRetargeter | None = None
    if hand_available:
        try:
            hand_retargeter = XHandRetargeter(
                hand_type="right",
                retargeting_type=HAND_RETARGETING_TYPE,
            )
            print(f"  Hand retargeter 就绪 (type={HAND_RETARGETING_TYPE})")
        except Exception as e:
            logger.warning("Hand retargeter 初始化失败: %s — 降级为 hold-position", e)
            hand_available = False
            hand_retargeter = None

    def _reset_hand_retargeter() -> None:
        if hand_retargeter is not None:
            try:
                hand_retargeter.reset()
            except Exception:
                pass

    # ── 4. Pre-Flight ──
    report = preflight_check(robot)
    print_preflight(report)
    if not report.passed:
        print("Pre-Flight 检查失败，退出")
        robot.disconnect()
        vr_receiver.stop()
        return

    # ── 5. ArmInnerLoop (30Hz online trajectory planning, crash-isolated subprocess) ──
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

    prev_hand_cmd = (
        state.hand_qpos.copy() if np.all(np.isfinite(state.hand_qpos)) else np.zeros(12, dtype=np.float64)
    )

    ema_prev_pos: np.ndarray | None = None  # 笛卡尔 EMA 状态 (IK 前)
    ema_prev_quat: np.ndarray | None = None
    _last_target_eef_pos = np.full(3, np.nan)  # 最后有效目标位姿 (held 帧写此值)
    _last_target_eef_rot6d = np.full(6, np.nan)

    print(f"\n初始状态:")
    print(f"  arm_qpos:  {np.round(np.rad2deg(state.arm_qpos), 1)} deg")
    print(f"  eef_pos:   {np.round(state.eef_pos, 4)} m")
    print(f"  eef_quat:  {np.round(state.eef_quat_wxyz, 4)}")
    if hand_available:
        print(f"  hand_qpos: {np.round(np.rad2deg(state.hand_qpos), 1)} deg")

    # ── 6. VR Arm Mapper ──
    arm_mapper = ArmWristMapper(
        pos_scale=VR_POS_SCALE,
        rot_scale=VR_ROT_SCALE,
        max_delta_rot_rad=VR_MAX_DELTA_ROT_RAD,
    )

    # ── 7. Recorder ──
    # Resolve data_dir relative to repo root (not CWD) so episodes always land
    # under the repository regardless of where the script was launched from.
    _repo_root = Path(__file__).resolve().parents[2]
    recorder = EpisodeRecorder(
        data_dir=str(_repo_root / "episodes"),
        max_frames=int(round(60.0 * CTRL_HZ)),  # 60s 上限
        control_hz=CTRL_HZ,
        min_frames=int(round(1.0 * CTRL_HZ)),  # ≥1s 才算有效 episode
        arm_sent_stream=True,  # schema v9: 记录实发的 arm 指令 (plan §6 P1)
    )

    # ── 7b. 主循环分段计时 (1Hz 聚合打印, 定位超预算去向) ──
    stage_timer = StageTimer(window=STATUS_EVERY)

    # ── 8. Keyboard ──
    kb = KeyboardHandler()
    kb.start()
    atexit.register(kb.stop)

    # ── 8b. Audio feedback (voice prompts for headset-blind operator) ──
    audio = AudioFeedback()

    # ── 8c. Camera (RealSense, 独立进程, 共享内存零拷贝, 可降级) ──
    session = create_camera_session()

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
            session.stop()
            return
        frame = vr_receiver.read_latest()
        if frame is not None:
            print(f"  收到首帧 seq={frame.get('sequence_id', '?')} — 就绪")
            break
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
        session.stop()
        return

    # ── 10. 键盘就绪 ──

    print("\n控制: B=开始遥操作+录制 C=暂停 S=保存 D=丢弃 H=归位 Q=退出 ESC=急停")
    print("等待按键 B 开始遥操作...\n")

    # ── 11. Main loop ──
    limiter = RateManager(CTRL_HZ)  # 绝对期限调度 — tick 锁定录制时间栅格
    running = True
    teleop_active = False
    recording_active = False
    recording_paused = False  # C 暂停时置 True，暂停期间不写 held 帧
    teleop_hold_for_audio = False  # B 按下后等待 begin 音频播完才允许运动
    loop_count = 0
    error_count = 0
    max_consecutive_errors = 10
    recover_count = 0  # 连续可恢复错误 (C22/C24) 清错次数
    start_time = time.perf_counter()
    prev_eef_pos: np.ndarray | None = None
    ik_method = "-"
    _last_vr_wrist_quat: np.ndarray | None = None  # VR quat continuity tracking
    _vr_raw_printed = False  # suppress [VR raw] after first status tick

    def _stop_recording(save: bool, *, triggered_by: ControlSignal | None = None):
        """停止录制. save=True 保存, save=False 丢弃.

        Non-blocking: spawns the stop daemon and returns immediately.
        Completion (save/discard result) is reported by the poll block in
        the main loop via recorder.poll_stop().
        """
        nonlocal recording_active
        if recording_active:
            if save:
                print("  保存中…", flush=True)
                recorder.stop_episode(success=True)
            else:
                recorder.stop_episode(success=False)
            recording_active = False
            limiter.reset()
            if triggered_by is not None:
                kb.drain_signal(triggered_by)

    def _emergency_stop():
        """停止内环 + 停止录制 + 急停."""
        print("[TRACE] _emergency_stop() called from:", flush=True)
        traceback.print_stack()
        nonlocal running, teleop_active, recording_active
        teleop_active = False
        if recording_active:
            recorder.stop_episode(success=False)
            recording_active = False
            limiter.reset()
        if arm_inner.is_alive:
            arm_inner.set_target(None)
            arm_inner.stop()
        robot.emergency_stop()
        robot.arm.clear_error()
        running = False

    def _build_diag(ik_solve_time_ms, target_pos_before_clamp, vr_frame, **extra):
        """构建 held/fail 帧的诊断字典，消除 6 处重复."""
        hq = vr_frame.get("head_quat_wxyz") if vr_frame is not None else None
        diag = {
            "tracking_error": arm_inner.tracking_error,
            "ik_solve_time_ms": ik_solve_time_ms,
            "target_pos_before_clamp": target_pos_before_clamp,
            "head_quat_wxyz": hq if hq is not None else np.full(4, np.nan),
        }
        diag.update(extra)
        return diag

    # Cancel function for do_return_home's convergence loop.
    # Drains the keyboard buffer without blocking; returns True when
    # QUIT or EMERGENCY_STOP is pending so do_return_home can abort
    # its convergence wait early.  Non-cancel signals (B/C/D/S/H) are
    # silently consumed — the user can re-press after return_home
    # completes (these keys are meaningless during homing anyway).
    def _return_home_cancel_fn() -> bool:
        sigs = kb.poll(timeout=0.0)
        return any(s in (ControlSignal.QUIT, ControlSignal.EMERGENCY_STOP) for s in sigs)

    gc.disable()
    try:
        while running:
            _timing_line = stage_timer.tick()
            if _timing_line:
                print(f"  [timing] {_timing_line}")
            limiter.wait()
            stage_timer.mark("wait")

            # ── Non-blocking save completion poll (16 Hz) ──
            _stop_result = recorder.poll_stop()
            if _stop_result.done and _stop_result.path is not None:
                if _stop_result.error:
                    print(f"  ⚠ 保存失败 ({_stop_result.error}): {_stop_result.path}  — 文件可能不完整")
                elif _stop_result.success:
                    print(f"  录制已保存: {_stop_result.path}  ({_stop_result.frame_count} 帧)")
                gc.collect()

            loop_count += 1

            # ── 按键处理 ──
            skip_rest = False
            for sig in kb.poll(timeout=0.0):
                if sig == ControlSignal.EMERGENCY_STOP:
                    print("\nESC: emergency_stop")
                    audio.play("emergency")
                    print("[TRACE] emergency_stop triggered from keyboard handler:", flush=True)
                    traceback.print_stack()
                    _emergency_stop()
                    break

                elif sig == ControlSignal.QUIT:
                    print("\nQ: 退出")
                    audio.play("quit")

                    if recording_active:
                        audio.play("quit_save_prompt")
                        print("  [S] 保存并退出  [D] 丢弃并退出  [H] 保存并归位 (30s 超时默认丢弃)")

                        decision: bool | None = None
                        do_home: bool = False
                        deadline = time.perf_counter() + 30.0
                        while time.perf_counter() < deadline:
                            for post_sig in kb.poll(timeout=0.1):
                                if post_sig == ControlSignal.STOP:
                                    decision = True
                                    break
                                if post_sig == ControlSignal.DISCARD:
                                    decision = False
                                    break
                                if post_sig == ControlSignal.HOME:
                                    decision = True
                                    do_home = True
                                    break
                                if post_sig == ControlSignal.EMERGENCY_STOP:
                                    audio.play("emergency")
                                    _emergency_stop()
                                    running = False
                                    decision = None
                                    break
                            if decision is not None:
                                break
                            if not running:
                                break

                        if decision is True:
                            audio.play("save")
                            _stop_recording(save=True, triggered_by=ControlSignal.STOP)
                            print("  已保存")
                        elif decision is False:
                            audio.play("discard")
                            _stop_recording(save=False, triggered_by=ControlSignal.DISCARD)
                            print("  已丢弃")
                        elif running:
                            audio.play("discard")
                            _stop_recording(save=False)
                            print("  超时，默认丢弃")

                        if do_home and running:
                            audio.play("home")
                            arm_inner = do_return_home(robot, arm_inner, _INNER_CFG, cancel_fn=_return_home_cancel_fn)
                            teleop_active = False
                            arm_mapper.clear()
                            _reset_hand_retargeter()
                            if arm_inner.wait_ready(timeout=30.0):
                                arm_qpos, error_state, _ = arm_inner.get_state()
                                if not error_state and np.all(np.isfinite(arm_qpos)):
                                    state = robot.get_state(arm_qpos=arm_qpos)
                                    prev_qpos_cmd = state.arm_qpos.copy()
                                    prev_hand_cmd = (
                                        state.hand_qpos.copy()
                                        if np.all(np.isfinite(state.hand_qpos))
                                        else np.zeros(12, dtype=np.float64)
                                    )
                                    ema_prev_pos = ema_prev_quat = None
                                    error_count = 0
                            else:
                                state = robot.get_state()
                                if np.all(np.isfinite(state.arm_qpos)):
                                    prev_qpos_cmd = state.arm_qpos.copy()
                                    ema_prev_pos = ema_prev_quat = None
                                error_count = 0
                            # Drain HOME auto-repeat accumulated during
                            # do_return_home.
                            kb.poll(timeout=0.0)
                    else:
                        _stop_recording(save=False, triggered_by=ControlSignal.QUIT)

                    running = False
                    break

                elif sig == ControlSignal.HOME:
                    print("\nH: return_home")
                    audio.play("home")

                    # If already at home, skip expensive do_return_home
                    # (defense-in-depth against auto-repeat HOME leaking past
                    # the post-home drain).
                    if arm_qpos is not None and np.all(np.isfinite(arm_qpos)):
                        if np.max(np.abs(arm_qpos - home_qpos)) < np.deg2rad(2.0):
                            print("  已在 home 位置，跳过归位")
                            _stop_recording(save=True, triggered_by=ControlSignal.HOME)
                            skip_rest = True
                            continue

                    _stop_recording(save=True, triggered_by=ControlSignal.HOME)
                    arm_inner = do_return_home(robot, arm_inner, _INNER_CFG, cancel_fn=_return_home_cancel_fn)
                    teleop_active = False
                    arm_mapper.clear()
                    _reset_hand_retargeter()
                    if arm_inner.wait_ready(timeout=30.0):
                        arm_qpos, error_state, _ = arm_inner.get_state()
                        if not error_state and np.all(np.isfinite(arm_qpos)):
                            state = robot.get_state(arm_qpos=arm_qpos)
                            prev_qpos_cmd = state.arm_qpos.copy()
                            prev_hand_cmd = (
                                state.hand_qpos.copy()
                                if np.all(np.isfinite(state.hand_qpos))
                                else np.zeros(12, dtype=np.float64)
                            )
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
                    # Drain HOME auto-repeat accumulated during do_return_home
                    # (3-4s blocking).  Blanket drain is safe here: HOME
                    # auto-repeat is unwanted (arm already home), ESC bypasses
                    # debounce at ~60Hz, and any other signal can be re-pressed.
                    kb.poll(timeout=0.0)
                    skip_rest = True

                elif sig == ControlSignal.STOP:
                    print("\nS: 停止录制")
                    audio.play("save")
                    _stop_recording(save=True, triggered_by=ControlSignal.STOP)
                    teleop_active = False
                    skip_rest = True

                elif sig == ControlSignal.DISCARD:
                    print("\nD: 丢弃录制")
                    audio.play("discard")
                    _stop_recording(save=False, triggered_by=ControlSignal.DISCARD)
                    teleop_active = False
                    skip_rest = True

                elif sig == ControlSignal.PAUSE:
                    teleop_active = not teleop_active
                    recording_paused = not teleop_active  # 同步暂停/恢复录制
                    state_str = "暂停" if not teleop_active else "恢复"
                    print(f"\nC: {state_str}遥操作 (录制{'暂停' if recording_paused else '继续'})")
                    if teleop_active:
                        audio.play("resume")
                        # 恢复时重新建立 wrist→EEF 映射，避免跳跃
                        frame = vr_receiver.read_latest()
                        if frame is not None:
                            state = robot.get_state(
                                arm_qpos=arm_inner.get_state()[0] if arm_inner.is_alive else None
                            )
                            arm_mapper.reset(
                                wrist_pos=frame["wrist_pos"],
                                wrist_quat_wxyz=frame["wrist_quat_wxyz"],
                                eef_pos=state.eef_pos,
                                eef_quat_wxyz=state.eef_quat_wxyz,
                            )
                            # 手部: 重置 retargeter 以清除 SLSQP warm-start / LPFilter 状态
                            _reset_hand_retargeter()
                            audio.play("calibrated")
                    else:
                        audio.play("pause")
                    skip_rest = True

                elif sig == ControlSignal.BEGIN:
                    # 开始/重置遥操作: 启动录制 + 记录 wrist→EEF 映射
                    frame = vr_receiver.read_latest()
                    if frame is None:
                        print("\nB: 无 VR 帧，无法开始遥操作")
                        skip_rest = True
                        continue
                    if session.crashed:
                        print("  ⚠ 相机进程已退出 — 本集降级为只录关节/EEF")
                    # 如果已在录制，先停止旧 episode
                    _stop_recording(save=recording_active, triggered_by=ControlSignal.BEGIN)
                    gc.collect()
                    _record_cfg = session.pointcloud_meta
                    _record_cfg.update({
                        "ema_alpha_pos": EMA_ALPHA_POS,
                        "ema_alpha_rot": EMA_ALPHA_ROT,
                        "joint_max_acc": float(ARM_MAX_ACC_DEG_S2),
                        "joint_max_speed": ARM_MAX_SPEED_DEG_S,
                        "inner_loop_hz": float(1.0 / _INNER_CFG.loop_period),
                        "tracking_error_adaptive_max_rad": _INNER_CFG.tracking_error_adaptive_max_rad,
                        "tracking_error_anomaly_cap_rad": _INNER_CFG.tracking_error_anomaly_cap_rad,
                        "hand_available": hand_available,
                        "hand_retargeting_type": HAND_RETARGETING_TYPE if hand_available else "none",
                    })
                    if not recorder.start_episode(
                        task_label=TASK_LABEL,
                        operator=OPERATOR,
                        depth_scale=session.depth_scale,
                        calib=session.calib,
                        camera_name=session.resolve_name(),
                        camera_K=session.camera_K,
                        record_config=_record_cfg,
                    ):
                        print("  ⚠ 无法开始录制（上一 episode 仍在写盘）")
                        skip_rest = True
                        continue
                    # Drain B auto-repeat accumulated during start_episode's
                    # join_stop (previous save daemon may still be flushing).
                    kb.drain_signal(ControlSignal.BEGIN)
                    recording_active = True
                    state = robot.get_state(arm_qpos=arm_inner.get_state()[0] if arm_inner.is_alive else None)

                    # Heading calibration
                    head_q = frame.get("head_quat_wxyz")
                    head_p = frame.get("head_pos")
                    head_ok = (
                        head_q is not None
                        and head_p is not None
                        and np.any(np.isfinite(head_q))
                        and np.any(np.array(head_p) != 0)
                    )
                    if head_ok:
                        arm_mapper.set_heading(head_q)
                        head_rot = Rotation.from_quat(np.roll(np.asarray(head_q), -1))
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
                    # 手部: 重置 retargeter (SLSQP warm-start seed, LPFilter EMA, DexPilot projection)
                    _reset_hand_retargeter()
                    audio.play("calibrated")
                    teleop_active = True
                    recording_paused = False
                    error_count = 0
                    # Seed continuity gate from B-press frame
                    _last_vr_wrist_quat = normalize_quat_wxyz(
                        np.asarray(frame["wrist_quat_wxyz"], dtype=np.float64)
                    )
                    # Seed hand command from current state so first frame delta is zero
                    prev_hand_cmd = (
                        state.hand_qpos.copy()
                        if np.all(np.isfinite(state.hand_qpos))
                        else np.zeros(12, dtype=np.float64)
                    )
                    audio.play("begin")
                    teleop_hold_for_audio = True
                    print(f"\nB: 遥操作+录制开始 (wrist→EEF 映射已记录, 手部 retargeting 已重置)  episode={recorder.frame_count}")
                    print(f"  wrist_ref={np.round(frame['wrist_pos'], 3)}")
                    print(f"  eef_ref=  {np.round(state.eef_pos, 3)}")
                    skip_rest = True

            if not running:
                break
            if skip_rest:
                continue

            # ── 读取 ArmInnerLoop 状态 (一次加锁, 避免 qpos vs qvel/tau 偏差) ──
            try:
                arm_qpos, error_state, _inner_ts, arm_qvel, arm_tau = arm_inner.get_state_and_dynamics()
                state = robot.get_state(arm_qpos=arm_qpos, arm_qvel=arm_qvel, arm_tau=arm_tau)

                if error_state:
                    print(f"  Arm 内环异常: error_state=True")
                    error_count += 1
                    if error_count > 3:
                        print("Arm 内环连续异常，急停退出")
                        audio.play("emergency")
                        _emergency_stop()
                        break
                    continue
            except Exception as e:
                error_count += 1
                print(f"  get_state 异常: {e}")
                if error_count > max_consecutive_errors:
                    print("连续错误过多，急停退出")
                    audio.play("emergency")
                    _emergency_stop()
                    break
                continue
            stage_timer.mark("state")

            # ── Arm error check ──
            if robot.arm.is_connected() and robot.arm.is_error():
                arm_code = robot.arm.arm.error_code if robot.arm.arm else 0
                sdk_code = robot.arm.last_sdk_error_code

                if arm_code in (22, 24) or sdk_code in (22, 24):
                    code = arm_code or sdk_code
                    recover_count += 1
                    if recover_count > 5:
                        print(f"  C{code} 连续恢复超过 5 次，急停退出")
                        audio.play("emergency")
                        _emergency_stop()
                        break
                    print(
                        f"  ⚠ ControllerError {code} ({'自碰撞' if code == 22 else '速度超限'})，清除错误并保持位置",
                        flush=True,
                    )
                    robot.arm.clear_error()
                    state = robot.get_state()
                    if np.all(np.isfinite(state.arm_qpos)):
                        prev_qpos_cmd = state.arm_qpos.copy()
                        prev_hand_cmd = (
                            state.hand_qpos.copy()
                            if np.all(np.isfinite(state.hand_qpos))
                            else np.zeros(12, dtype=np.float64)
                        )
                        ema_prev_pos = ema_prev_quat = None
                    error_count = 0
                    continue
                print(f"arm 错误: C{arm_code}")
                audio.play("emergency")
                _emergency_stop()
                break

            if not np.all(np.isfinite(state.arm_qpos)):
                error_count += 1
                continue

            error_count = 0
            recover_count = 0

            # ── 读取 VR 帧 ──
            vr_frame = vr_receiver.read_latest()
            vr_stale = (
                vr_frame is None
                or (time.monotonic_ns() - vr_frame.get("local_recv_ns", 0)) > VR_STALE_THRESHOLD_S * 1e9
            )
            stage_timer.mark("vr")

            # ── 相机帧 ──
            cam = session.poll_latest_frame()
            stage_timer.mark("cam")

            # ── Periodic status ──
            if loop_count % STATUS_EVERY == 0:
                elapsed = time.perf_counter() - start_time
                if prev_eef_pos is not None:
                    vel = np.linalg.norm(state.eef_pos - prev_eef_pos) / (STATUS_EVERY * CTRL_DT)
                else:
                    vel = 0.0
                prev_eef_pos = state.eef_pos.copy()
                vr_age = vr_receiver.frame_age_s() if vr_frame is not None else -1
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

                if vr_frame is not None and not _vr_raw_printed:
                    _wp = vr_frame["wrist_pos"]
                    _wq = vr_frame["wrist_quat_wxyz"]
                    _w_euler = np.rad2deg(Rotation.from_quat(wxyz_to_xyzw(_wq)).as_euler("xyz", degrees=False))
                    print(
                        f"  [VR raw] wrist_pos(FLU)={np.round(_wp, 4)}m  "
                        f"wrist_quat_wxyz={np.round(_wq, 4)}  "
                        f"wrist_euler_xyz={np.round(_w_euler, 1)}°",
                        flush=True,
                    )
                    _vr_raw_printed = True
            stage_timer.mark("print")

            # ── 非遥操作模式: 保持当前位置 ──
            if not teleop_active or vr_stale:
                if not teleop_active:
                    pass
                elif vr_stale:
                    if loop_count % STATUS_EVERY == 0:
                        print("  ⚠ VR 帧过期，保持当前位置")
                prev_qpos_cmd = state.arm_qpos.copy()
                ema_prev_pos = ema_prev_quat = None
                if recording_active and not recording_paused:
                    record_held_frame(
                        recorder,
                        state,
                        prev_qpos_cmd.copy(),
                        vr_frame,
                        cam,
                        ik_ok=False,
                        retarget_ok=False,
                        arm_qpos_sent=arm_inner.last_sent_cmd,
                        hold_hand=prev_hand_cmd.copy(),
                        diagnostics=_build_diag(0.0, np.full(3, np.nan), vr_frame),
                        target_eef_pos=_last_target_eef_pos,
                        target_eef_rot6d=_last_target_eef_rot6d,
                    )
                # Keep the hand cmd ring fresh during pause / VR stale —
                # the arm inner loop holds position on its own, but the hand
                # child's cmd_stale_hold_s (500 ms) fires noisy warnings if
                # we stop sending entirely.  Sending the current hand position
                # as an explicit hold matches the IK-fail path (:1130-1137).
                if hand_available:
                    robot.send_action(RobotAction(
                        arm_qpos_cmd=prev_qpos_cmd.copy(),
                        hand_qpos_cmd=prev_hand_cmd.copy(),
                    ))
                continue

            # ── 音频等待: B 按下后等待 begin 音频播完再响应运动 ──
            if teleop_hold_for_audio:
                if audio.is_playing:
                    prev_qpos_cmd = state.arm_qpos.copy()
                    # Keep the hand cmd ring fresh during audio hold —
                    # the hand child's cmd_stale_hold_s (500ms) fires
                    # false-positive warnings if we stop sending entirely.
                    # Match the pattern in the pause/VR-stale path (L969-973).
                    if hand_available:
                        robot.send_action(RobotAction(
                            arm_qpos_cmd=prev_qpos_cmd.copy(),
                            hand_qpos_cmd=prev_hand_cmd.copy(),
                        ))
                    continue
                ema_prev_pos = ema_prev_quat = None
                teleop_hold_for_audio = False

            # ── VR wrist quaternion continuity gate ──
            _vr_wrist_quat = normalize_quat_wxyz(np.asarray(vr_frame["wrist_quat_wxyz"], dtype=np.float64))
            if _last_vr_wrist_quat is not None:
                _qd = min(np.abs(np.dot(_vr_wrist_quat, _last_vr_wrist_quat)), 1.0)
                _angle = 2.0 * np.arccos(_qd)
                if _angle > VR_WRIST_QUAT_JUMP_THRESHOLD_RAD:
                    logger.warning(
                        "VR wrist quaternion spike: %.1f°/frame > %.1f°/frame "
                        "threshold — holding position",
                        np.rad2deg(_angle),
                        np.rad2deg(VR_WRIST_QUAT_JUMP_THRESHOLD_RAD),
                    )
                    if recording_active:
                        hold_arm = prev_qpos_cmd.copy() if prev_qpos_cmd is not None else state.arm_qpos.copy()
                        record_held_frame(
                            recorder, state, hold_arm, vr_frame, cam, ik_ok=False, retarget_ok=False,
                            arm_qpos_sent=arm_inner.last_sent_cmd, hold_hand=prev_hand_cmd.copy(),
                            diagnostics=_build_diag(0.0, np.full(3, np.nan), vr_frame,
                                                    vr_quat_jump_rejected=_angle),
                            target_eef_pos=_last_target_eef_pos,
                            target_eef_rot6d=_last_target_eef_rot6d,
                        )
                    arm_inner.set_target(prev_qpos_cmd.copy() if prev_qpos_cmd is not None else state.arm_qpos.copy())
                    ik_method = "vr_jump"
                    continue
            _last_vr_wrist_quat = _vr_wrist_quat.copy()

            # ── VR wrist → EEF target pose ──
            mapped = arm_mapper.map(vr_frame["wrist_pos"], vr_frame["wrist_quat_wxyz"])
            if mapped is None:
                ik_method = "no_map"
                if recording_active:
                    hold_arm = prev_qpos_cmd.copy() if prev_qpos_cmd is not None else state.arm_qpos.copy()
                    record_held_frame(
                        recorder, state, hold_arm, vr_frame, cam, ik_ok=False, retarget_ok=False,
                        arm_qpos_sent=arm_inner.last_sent_cmd, hold_hand=prev_hand_cmd.copy(),
                        diagnostics=_build_diag(0.0, np.full(3, np.nan), vr_frame),
                        target_eef_pos=_last_target_eef_pos,
                        target_eef_rot6d=_last_target_eef_rot6d,
                    )
                arm_inner.set_target(prev_qpos_cmd.copy() if prev_qpos_cmd is not None else state.arm_qpos.copy())
                continue

            target_pos = mapped["pos"]
            target_quat = mapped["quat_wxyz"]

            # ── Cartesian pose EMA (IK 前, 唯一平滑级) ──
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
            stage_timer.mark("map")

            # ── IK solve ──
            target_pose = Pose(p=target_pos, q=target_quat)
            if np.all(np.isfinite(state.hand_qpos)):
                planner.set_hand_qpos(state.hand_qpos)
            _ik_t0 = time.perf_counter()
            ik_result = planner.solve_teleop_ik(target_pose, state.arm_qpos, prev_qpos_cmd)
            ik_solve_time_ms = (time.perf_counter() - _ik_t0) * 1000.0
            stage_timer.mark("ik")

            # ── Hand: compute via retargeting from VR landmarks ──
            hand_cmd, retarget_ok = compute_hand_command(
                hand_retargeter,
                vr_frame,
                prev_hand_cmd,
                hand_available,
            )
            stage_timer.mark("hand")

            if not ik_result.success or ik_result.qpos is None:
                ik_method = "fail"
                if recording_active:
                    hold_arm = prev_qpos_cmd.copy() if prev_qpos_cmd is not None else state.arm_qpos.copy()
                    record_held_frame(
                        recorder, state, hold_arm, vr_frame, cam, ik_ok=False, retarget_ok=retarget_ok,
                        arm_qpos_sent=arm_inner.last_sent_cmd, hold_hand=hand_cmd.copy(),
                        diagnostics=_build_diag(ik_solve_time_ms, target_pos_before_clamp, vr_frame),
                        frame_status=_FRAME_IK_FAIL, ik_attempted=True,
                        target_eef_pos=target_pos.copy(),
                        target_eef_rot6d=quat_wxyz_to_rot6d(target_quat),
                    )
                arm_inner.set_target(prev_qpos_cmd.copy())
                # Hand retargeting is independent of arm IK — send hand
                # even when IK fails so the hand cmd ring doesn't go stale.
                if hand_available:
                    robot.send_action(RobotAction(
                        arm_qpos_cmd=prev_qpos_cmd.copy(),
                        hand_qpos_cmd=hand_cmd,
                    ))
                    prev_hand_cmd = hand_cmd.copy()
                stage_timer.mark("send")
                continue

            # ── IK output safety gate ──
            if IK_OUTPUT_GATE_ENABLED:
                _seed = prev_qpos_cmd if prev_qpos_cmd is not None else state.arm_qpos
                _ik_joint_delta = float(np.max(np.abs(ik_result.qpos - _seed)))
                if _ik_joint_delta > IK_OUTPUT_SAFETY_THRESHOLD_RAD:
                    logger.warning(
                        "IK output rejected: joint delta %.3f rad > threshold %.3f rad "
                        "(seed vs ik_result.qpos) — holding position",
                        _ik_joint_delta,
                        IK_OUTPUT_SAFETY_THRESHOLD_RAD,
                    )
                    if recording_active:
                        hold_arm = prev_qpos_cmd.copy() if prev_qpos_cmd is not None else state.arm_qpos.copy()
                        record_held_frame(
                            recorder, state, hold_arm, vr_frame, cam, ik_ok=False, retarget_ok=retarget_ok,
                            arm_qpos_sent=arm_inner.last_sent_cmd, hold_hand=hand_cmd.copy(),
                            diagnostics=_build_diag(ik_solve_time_ms, target_pos_before_clamp, vr_frame,
                                                    ik_joint_delta_rejected=_ik_joint_delta),
                            ik_attempted=True,
                            target_eef_pos=target_pos.copy(),
                            target_eef_rot6d=quat_wxyz_to_rot6d(target_quat),
                        )
                    arm_inner.set_target(prev_qpos_cmd.copy())
                    # IK output rejected (joint delta too large) — arm holds
                    # but hand retargeting is independent and should still send.
                    if hand_available:
                        robot.send_action(RobotAction(
                            arm_qpos_cmd=prev_qpos_cmd.copy(),
                            hand_qpos_cmd=hand_cmd,
                        ))
                        prev_hand_cmd = hand_cmd.copy()
                    stage_timer.mark("send")
                    continue

            ik_method = "ok"
            # Update EMA state on IK success
            ema_prev_pos = target_pos.copy()
            ema_prev_quat = target_quat.copy()
            _last_target_eef_pos = target_pos.copy()
            _last_target_eef_rot6d = quat_wxyz_to_rot6d(target_quat)

            arm_cmd = np.asarray(ik_result.qpos, dtype=np.float64)
            arm_cmd = prev_qpos_cmd + np.clip(arm_cmd - prev_qpos_cmd, -ARM_CMD_MAX_STEP_RAD, ARM_CMD_MAX_STEP_RAD)

            action = RobotAction(
                arm_qpos_cmd=arm_cmd,
                hand_qpos_cmd=hand_cmd,
                target_eef_pos=target_pos.copy(),
                target_eef_rot6d=quat_wxyz_to_rot6d(target_quat),
            )

            # ── Pre-send gate: 力矩/温度/软限位 ──
            action_valid, fail_reason = validate_action(
                robot,
                action,
                actual_arm_qvel=state.arm_qvel,
                actual_arm_tau=state.arm_tau,
            )
            if not action_valid:
                print(f"  [SAFETY] Pre-send gate: {fail_reason} — 跳过本帧", flush=True)
                if recording_active:
                    hold_arm = prev_qpos_cmd.copy() if prev_qpos_cmd is not None else state.arm_qpos.copy()
                    record_held_frame(
                        recorder, state, hold_arm, vr_frame, cam, ik_ok=False, retarget_ok=retarget_ok,
                        arm_qpos_sent=arm_inner.last_sent_cmd, hold_hand=hand_cmd.copy(),
                        safety_reject=True,
                        diagnostics=_build_diag(ik_solve_time_ms, target_pos_before_clamp, vr_frame),
                        frame_status=_FRAME_SAFETY_REJECT, ik_attempted=True,
                        target_eef_pos=target_pos.copy(),
                        target_eef_rot6d=quat_wxyz_to_rot6d(target_quat),
                    )
                # NOTE: 安全拒绝不发送 hand 命令。与 IK 失败路径不同，
                # 安全拒绝表示硬件异常（力矩/温度/错误状态），此时应停止
                # 所有命令发送。IK 失败仅是规划问题，手部 retargeting
                # 独立于臂部 IK，故继续发送手部命令。
                arm_inner.set_target(prev_qpos_cmd.copy())
                stage_timer.mark("send")
                continue

            prev_qpos_cmd = arm_cmd.copy()
            prev_hand_cmd = hand_cmd.copy()

            # Snapshot sent command for recording temporal alignment
            sent_cmd = arm_cmd.copy()

            # ── Send ──
            arm_inner.set_target(action.arm_qpos_cmd)
            robot.send_action(action)  # hand via SHM subprocess (or direct XHand)
            stage_timer.mark("send")

            # ── 录制帧 ──
            if recording_active:
                # 手部 retargeting 失败但 IK 成功 → 单独标记, 不丢弃整帧
                if not retarget_ok and hand_available:
                    _frame_status = _FRAME_RETARGET_FAIL
                else:
                    _frame_status = _FRAME_OK

                sig = {
                    "ik_ok": ik_result.success and ik_result.qpos is not None,
                    "ik_attempted": True,
                    "retarget_ok": retarget_ok,
                    "held": False,
                    "flag_safety_reject": False,
                    "frame_status": _frame_status,
                }
                head_quat = vr_frame.get("head_quat_wxyz")
                diagnostics = {
                    "tracking_error": arm_inner.tracking_error,
                    "ik_solve_time_ms": ik_solve_time_ms,
                    "target_pos_before_clamp": target_pos_before_clamp,
                    "head_quat_wxyz": head_quat if head_quat is not None else np.full(4, np.nan),
                }
                ok = recorder.add_frame(
                    state, action, vr_frame, camera_frame=cam, signals=sig, arm_qpos_sent=sent_cmd,
                    diagnostics=diagnostics,
                )
                if not ok and recorder.max_frames_reached:
                    print(f"\n  达到 max_frames={recorder.max_frames}，自动停止录制")
                    audio.play("save")
                    _stop_recording(save=True)
                    teleop_active = False
            stage_timer.mark("rec")

    finally:
        gc.enable()
        if recording_active:
            recorder.stop_episode(success=False)
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

        print("\n退出主循环")

        print("\n按 H 执行 return_home，或按 Q 直接退出...")
        _post_deadline = time.perf_counter() + 60.0
        while time.perf_counter() < _post_deadline:
            post_sigs = {s for s in kb.poll(timeout=0.1)}
            if ControlSignal.HOME in post_sigs:
                # Early return: skip do_return_home if already at home
                # (matches main-loop H handler guard at line 646-651).
                if arm_inner.is_alive:
                    _qpos, _err, __ = arm_inner.get_state()
                    if _qpos is not None and not _err and np.all(np.isfinite(_qpos)):
                        if np.max(np.abs(_qpos - home_qpos)) < np.deg2rad(2.0):
                            print("  已在 home 位置，跳过归位")
                            print("按 Q 退出...")
                            continue
                print("\nH: return_home")
                audio.play("home")
                try:
                    arm_inner = do_return_home(robot, arm_inner, _INNER_CFG, cancel_fn=_return_home_cancel_fn)
                except Exception:
                    traceback.print_exc()
                    print("  return_home 失败，继续退出")
                print("按 Q 退出...")
            if ControlSignal.QUIT in post_sigs or ControlSignal.EMERGENCY_STOP in post_sigs:
                if ControlSignal.EMERGENCY_STOP in post_sigs:
                    arm_inner.emergency_stop()
                    robot.emergency_stop()
                break
        else:
            print("\n  超时，自动退出")

        kb.stop()

        if arm_inner.is_alive:
            arm_inner.set_target(None)
            arm_inner.stop()
            print("Arm 内环线程已停止")

        session.stop()

        robot.disconnect()
        vr_receiver.stop()

        audio.play("end")
        time.sleep(2.0)

        print("Done.")


if __name__ == "__main__":
    main()
