#!/usr/bin/env python3
"""Keyboard teleop xArm7 — SharedStorage-based architecture.

Uses arm_loop process (Mode 6, 30Hz) for arm control via SharedStorage.

Usage:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate real_robot
    python examples/real/keyboard_teleop_real.py

Controls:
    Move EEF (world frame):
      W/S       X forward/back
      A/D       Y left/right
      ↑/↓       Z up/down
      I/K       Pitch (Y rotation)
      ←/→       Roll  (X rotation)
      J/L       Yaw   (Z rotation)
    Q          quit
    R          return_home
    ESC        emergency stop
"""

from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import termios
import time
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R

from dexmani_real.planning import PlanningProfile, Pose, TeleopProfile, XArm7MotionPlanner
from dexmani_real.planning.path_utils import plan_joint_home_path
from dexmani_real.planning.pose_utils import quat_multiply
from dexmani_real.teleop.keyboard import GlobalKeyState
from dexmani_real.robot.arm_loop import arm_loop as _arm_loop, ArmLoopConfig
from dexmani_real.robot.hand_process import hand_loop as _hand_loop
from dexmani_real.shm.shared_storage import SharedStorage, HOME_SENTINEL
from dexmani_real.robot.safety import SafetyState, transition
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate_manager import RateManager
from dexmani_real.config.defaults import policy
from dexmani_real.utils.signal_utils import ema_smooth_pose

try:
    from pynput import keyboard  # type: ignore[import-untyped]
except ImportError:
    raise ImportError("pynput is required for keyboard input. Install with: pip install pynput")

logger = get_logger(__name__)

# ═══════════════════════════════════════════════ 配置


@dataclass
class KeyboardTeleopConfig:
    """Keyboard teleop tuning parameters. Edit defaults here — no CLI needed."""

    ctrl_hz: float = 30.0  # control loop rate, matches arm_loop to avoid queue backpressure
    delta_pos: float = 0.008  # EEF translation per keypress (m) → 240 mm/s @ 30Hz
    delta_rpy: float = 0.03  # EEF rotation per keypress (rad) → 1.7°/frame, 51°/s @ 30Hz

    # Cartesian P-term: amplifies position error before IK to reduce steady-state
    # tracking lag.  At Kp=0.0 the arm lags ~50mm behind the target at 250 mm/s
    # (open-loop time constant τ ≈ 0.2 s).  Kp=0.3 reduces the steady-state error
    # by ~23 % (to ~38 mm); Kp=0.5 by ~33 % (to ~33 mm).  Higher values risk
    # overshoot on direction reversals.  Set 0.0 for pure open-loop behaviour.
    cartesian_kp: float = 0.0  # conservative default; try 0.3–0.5 for less lag

    home_dt: float = 0.04  # homing waypoint interval (s): ~25°/s

    # ── Motion tracing: 追踪纯 +X 运动时的位置变化管线 ──
    trace_motion: bool = True
    trace_frame_interval: int = 10  # print every N frames


# Singleton config (edit defaults in the dataclass above).
_cfg = KeyboardTeleopConfig()

# Workspace bounds in WORLD frame (defined in defaults.py as world-frame coordinates).
# The robot base sits at base_pose_world (30° Z rotation) within this world.
WORKSPACE_BOUNDS = policy.workspace.as_array()


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
    z_shift_mm_raw = float((ik_fk_pos[2] - ik_target_pos[2]) * 1000)
    z_shift_mm = 0.0 if abs(z_shift_mm_raw) < 0.05 else z_shift_mm_raw  # suppress -0.0/+0.0 flip

    print(
        f"\n{'─'*60}"
        f"\n[TRACE #{loop_count}] 纯轴运动 + Cartesian EMA + P-term (Kp={_cfg.cartesian_kp})"
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
        f"  _cfg.delta_pos={_cfg.delta_pos*1000:.0f}mm  _cfg.delta_rpy={np.rad2deg(_cfg.delta_rpy):.1f}deg  (1.0 / _cfg.ctrl_hz)={(1.0 / _cfg.ctrl_hz)}s  EMA_POS={policy.ema.alpha_pos} EMA_ROT={policy.ema.alpha_rot}"
    )
    print(f"  workspace (world): x{WORKSPACE_BOUNDS[0]} y{WORKSPACE_BOUNDS[1]} z{WORKSPACE_BOUNDS[2]}")
    print("=" * 60)

    # ── 1. Planner ──
    planner = XArm7MotionPlanner.create_default(
        planning_profile=PlanningProfile(max_waypoint_delta_deg=360.0),
        teleop_profile=TeleopProfile(
            max_pose_error_pos_m=0.02,
            max_pose_error_rot_rad=np.deg2rad(5.0),
        ),
    )

    # ── 2. SharedStorage + subprocesses ──
    shared = SharedStorage.create(prefix="dexmani_kb")
    arm_loop_cfg = ArmLoopConfig()
    arm_proc = mp.Process(target=_arm_loop, args=(shared, arm_loop_cfg), name="arm-kb", daemon=True)
    hand_proc = mp.Process(target=_hand_loop, args=(shared,), name="hand-kb", daemon=True)
    arm_proc.start()
    hand_proc.start()

    if not shared.arm_ready.wait(timeout=15):
        print("Arm 进程启动超时，退出")
        _shutdown_kb(shared, arm_proc, hand_proc)
        return
    shared.hand_ready.wait(timeout=15)  # optional — degrade gracefully

    transition(shared, SafetyState.ARMED)

    # Read initial state from rings.
    _arm_result = shared.arm_state_ring.read_latest()
    if _arm_result is None:
        print("无法读取 arm 状态，退出")
        _shutdown_kb(shared, arm_proc, hand_proc)
        return
    _arm_data, _, _ = _arm_result
    arm_qpos = np.asarray(_arm_data["qpos"][0], dtype=np.float64)
    eef_pos = np.asarray(_arm_data["eef_pos"][0], dtype=np.float64)
    eef_rot6d = np.asarray(_arm_data["eef_rot6d"][0], dtype=np.float64)
    arm_connected = bool(_arm_data["connected"][0])
    if not arm_connected or not np.all(np.isfinite(arm_qpos)):
        print(f"Arm 状态无效: connected={arm_connected}, qpos_finite={np.all(np.isfinite(arm_qpos))}")
        _shutdown_kb(shared, arm_proc, hand_proc)
        return

    prev_qpos_cmd = arm_qpos.copy()
    # Initialize target in WORLD frame (consistent with keyboard operator's perspective).
    # Pinocchio FK gives base-frame EEF; base_to_world_pose rotates by ~30° around Z.
    _eef_pin = planner.compute_eef_pose_base(arm_qpos)
    _eef_world = planner.base_to_world_pose(_eef_pin)
    target_pos = _eef_world.p.copy()
    target_quat = _eef_world.q.copy()

    print(f"  arm:  OK  connected={arm_connected}")
    print(f"\n初始状态:")
    print(f"  arm_qpos:  {np.round(np.rad2deg(arm_qpos), 1)} deg")
    print(f"  EEF (base):  {np.round(eef_pos, 4)} m")
    print(f"  EEF (world): {np.round(target_pos, 4)} m  ← IK target (Pinocchio FK → world)")
    print(f"  eef_quat:  {np.round(target_quat, 4)}")

    # ── 5. Keyboard input ──
    keys = GlobalKeyState()
    keys.start()
    print("\n键盘控制已启动，按 Q 退出")

    # ── 6. Main loop ──
    limiter = RateManager(1.0 / (1.0 / _cfg.ctrl_hz))
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

    # Cartesian EMA state (same smoothing as vr_teleop_policy)
    _prev_ema_pos: np.ndarray | None = None
    _prev_ema_quat: np.ndarray | None = None

    def _emergency_stop():
        """Set estop flag — arm_loop detects and stops."""
        nonlocal running
        shared.estop_request.value = True
        shared.is_running.value = False
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
                # Plan collision-safe path to home (same as VR policy)
                _home_qpos = np.array(arm_loop_cfg.home_qpos, dtype=np.float64)
                _waypoints = plan_joint_home_path(arm_qpos, _home_qpos, planner)
                if _waypoints is not None and len(_waypoints) > 0:
                    print(f"  planned homing: {len(_waypoints)} waypoints (路径安全无碰撞)")
                elif _waypoints is not None and len(_waypoints) == 0:
                    # Should not happen (plan_joint_home_path returns None when at home),
                    # but guard defensively.
                    print(f"  planned homing: already close to home, skipping")
                else:
                    print(f"  plan_joint_home_path returned None — falling back to joint-space interpolation")
                shared.arm_action_q.put((HOME_SENTINEL, _waypoints))
                _prev_ema_pos = _prev_ema_quat = None
                consecutive_divergence = 0
                error_count = 0
                # Wait for homing to complete, then refresh state from ring.
                time.sleep(5.0)  # generous — arm_loop homing takes ~3-4s
                _arm_result = shared.arm_state_ring.read_latest()
                if _arm_result is not None:
                    _ad, _, _ = _arm_result
                    _qpos_home = np.asarray(_ad["qpos"][0], dtype=np.float64)
                    prev_qpos_cmd = _qpos_home.copy()
                    # World-frame target at home position
                    _eef_pin = planner.compute_eef_pose_base(_qpos_home)
                    _eef_world = planner.base_to_world_pose(_eef_pin)
                    target_pos = _eef_world.p.copy()
                    target_quat = _eef_world.q.copy()
                prev_eef_pos = None
                ik_outcome = "home"
                limiter.reset()
                continue

            # ── Read state from arm_state_ring ──
            _arm_result = shared.arm_state_ring.read_latest()
            if _arm_result is None:
                error_count += 1
                if error_count > max_consecutive_errors:
                    print("连续arm state丢失，急停退出")
                    _emergency_stop()
                    break
                continue
            _arm_data, _, _ = _arm_result
            arm_qpos = np.asarray(_arm_data["qpos"][0], dtype=np.float64)
            arm_qvel = np.asarray(_arm_data["qvel"][0], dtype=np.float64)
            arm_connected = bool(_arm_data["connected"][0])
            arm_error_code = int(_arm_data["error_code"][0])
            eef_pos = np.asarray(_arm_data["eef_pos"][0], dtype=np.float64)
            eef_rot6d = np.asarray(_arm_data["eef_rot6d"][0], dtype=np.float64)

            if not arm_connected:
                error_count += 1
                if error_count > 3:
                    print("Arm disconnected, 急停退出")
                    _emergency_stop()
                    break
                continue

            error_count = 0

            # ── Safety: arm error ──
            if arm_error_code != 0:
                if arm_error_code in (22, 24, 31):
                    # arm_loop auto-clears these — just log and continue
                    if loop_count % 50 == 0:
                        print(f"  ⚠ Arm error C{arm_error_code} (arm_loop auto-recovering)", flush=True)
                else:
                    print(f"arm 非可恢复错误: C{arm_error_code}")
                    _emergency_stop()
                    break

            if not np.all(np.isfinite(arm_qpos)):
                error_count += 1
                continue

            # ── EEF target delta from keys ──
            dx = np.zeros(3, dtype=np.float64)
            if keys.is_pressed("w"):
                dx[0] += _cfg.delta_pos
            if keys.is_pressed("s"):
                dx[0] -= _cfg.delta_pos
            if keys.is_pressed("a"):
                dx[1] -= _cfg.delta_pos
            if keys.is_pressed("d"):
                dx[1] += _cfg.delta_pos
            if keys.is_pressed("up"):
                dx[2] += _cfg.delta_pos
            if keys.is_pressed("down"):
                dx[2] -= _cfg.delta_pos

            drpy = np.zeros(3, dtype=np.float64)
            if keys.is_pressed("left"):
                drpy[0] += _cfg.delta_rpy
            if keys.is_pressed("right"):
                drpy[0] -= _cfg.delta_rpy
            if keys.is_pressed("i"):
                drpy[1] += _cfg.delta_rpy
            if keys.is_pressed("k"):
                drpy[1] -= _cfg.delta_rpy
            if keys.is_pressed("j"):
                drpy[2] -= _cfg.delta_rpy
            if keys.is_pressed("l"):
                drpy[2] += _cfg.delta_rpy

            # ── Periodic status (suppressed when idle — no keys pressed) ──
            _is_idle = np.all(dx == 0) and np.all(drpy == 0)
            if loop_count % 50 == 0:
                # Always track velocity baseline (even when idle) so the first
                # non-idle status line gets an accurate speed estimate.
                if prev_eef_pos is not None:
                    vel = np.linalg.norm(eef_pos - prev_eef_pos) / (50 * (1.0 / _cfg.ctrl_hz))
                else:
                    vel = 0.0
                prev_eef_pos = eef_pos.copy()
                if not _is_idle:
                    elapsed = time.perf_counter() - start_time
                    _eef_world_status = planner.base_to_world_pose(Pose(p=eef_pos, q=np.array([1.,0.,0.,0.])))
                    print(
                        f"[T+{elapsed:.1f}s f={loop_count}] "
                        f"eef_w={np.round(_eef_world_status.p, 3)}m  "
                        f"target_w={np.round(target_pos, 3)}  "
                        f"v={vel:.2f}m/s  ik={ik_outcome}  err={error_count}",
                        flush=True,
                    )

            # No input → snap target to current world-frame EEF, reset EMA state
            if _is_idle:
                _eef_pin = planner.compute_eef_pose_base(arm_qpos)
                _eef_world = planner.base_to_world_pose(_eef_pin)
                target_pos = _eef_world.p.copy()
                target_quat = _eef_world.q.copy()
                prev_qpos_cmd = arm_qpos.copy()
                _prev_ema_pos = _prev_ema_quat = None  # reset EMA on re-engage
                continue

            # ── Incremental target (world frame) ──
            # Keyboard deltas accumulate in world frame — the operator's
            # intuitive forward/left/up directions.
            for axis in range(3):
                if dx[axis] != 0:
                    target_pos[axis] += dx[axis]

            # Workspace boundary: direct clamp in world frame.
            # WORKSPACE_BOUNDS is the world-frame AABB of the base workspace box.
            target_pos = np.clip(target_pos, WORKSPACE_BOUNDS[:, 0], WORKSPACE_BOUNDS[:, 1])
            for axis in range(3):
                _wall_check(axis, target_pos, WORKSPACE_BOUNDS, wall_warned, wall_timers)

            if np.any(drpy != 0):
                dq = R.from_euler('xyz', drpy).as_quat(scalar_first=True)
                target_quat = quat_multiply(dq, target_quat)

            # ── Cartesian EMA (before IK, same as vr_teleop_policy pipeline) ──
            # Smooths target trajectory to prevent IK discontinuities from
            # abrupt keypress changes.  At α=0.8 the EMA steady-state lag
            # is ~(1-α)/α × Δ ≈ 1.25 mm — negligible; the dominant 50 mm
            # lead comes from arm tracking dynamics, which the EMA does not
            # address (see _cfg.cartesian_kp below for that).
            if _prev_ema_pos is not None:
                ik_target_pos, ik_target_quat = ema_smooth_pose(
                    target_pos,
                    target_quat,
                    _prev_ema_pos,
                    _prev_ema_quat,
                    policy.ema.alpha_pos,
                    policy.ema.alpha_rot,
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
            if _cfg.cartesian_kp > 0:
                _eef_world_p = planner.base_to_world_pose(Pose(p=eef_pos, q=np.array([1.,0.,0.,0.]))).p
                pos_error = target_pos - _eef_world_p
                if float(np.linalg.norm(pos_error)) > 0.003:  # 3 mm deadband
                    ik_target_pos = ik_target_pos + _cfg.cartesian_kp * pos_error
                    ik_target_pos = np.clip(
                        ik_target_pos, WORKSPACE_BOUNDS[:, 0], WORKSPACE_BOUNDS[:, 1],
                    )

            # ── IK solve (on EMA-smoothed world-frame target) ──
            target_pose_world = Pose(p=ik_target_pos, q=ik_target_quat)
            # Hand qpos from ring (if available — keyboard teleop doesn't need hand)
            _hand_result = shared.hand_state_ring.read_latest()
            if _hand_result is not None:
                _hd, _, _ = _hand_result
                _hand_qpos = np.asarray(_hd["qpos"][0], dtype=np.float64)
                if np.all(np.isfinite(_hand_qpos)):
                    planner.set_hand_qpos(_hand_qpos)
            ik_result = planner.solve_teleop_ik(target_pose_world, arm_qpos, prev_qpos_cmd)

            if not ik_result.success or ik_result.qpos is None:
                ik_fail_count += 1
                reason = getattr(ik_result, "reason", "") or "unknown"
                now = time.perf_counter()
                if reason != _last_ik_fail_reason or now - _last_ik_fail_time > 1.0:
                    print(f"  ⚡ IK fail (#{ik_fail_count}): {reason}", flush=True)
                    _last_ik_fail_reason = reason
                    _last_ik_fail_time = now
                # Snap target to current world-frame EEF
                _eef_pin = planner.compute_eef_pose_base(arm_qpos)
                _eef_world = planner.base_to_world_pose(_eef_pin)
                target_pos = _eef_world.p.copy()
                target_quat = _eef_world.q.copy()
                _prev_ema_pos = _prev_ema_quat = None  # reset EMA on IK failure
                ik_outcome = "held"
                continue

            ik_outcome = "ok"
            prev_qpos_cmd = ik_result.qpos.copy()
            arm_cmd = ik_result.qpos

            # ── Motion Trace: 纯轴运动管线诊断 ──
            if (
                _cfg.trace_motion
                and loop_count % _cfg.trace_frame_interval == 0
                and dx[0] != 0
                and dx[1] == 0
                and dx[2] == 0
                and np.all(drpy == 0)
            ):
                ik_fk_pose_world = planner.kin.compute_eef_pose_world(ik_result.qpos)
                _eef_world_trace = planner.base_to_world_pose(Pose(p=eef_pos, q=np.array([1.,0.,0.,0.])))
                _print_motion_trace(
                    loop_count=loop_count,
                    dx=dx,
                    target_pos=target_pos,
                    ik_target_pos=ik_target_pos,
                    eef_pos=_eef_world_trace.p,
                    ik_fk_pos=ik_fk_pose_world.p,
                    ik_fk_quat=ik_fk_pose_world.q,
                    ik_target_quat=ik_target_quat,
                    ik_result=ik_result,
                    arm_qpos=arm_qpos,
                    report=getattr(ik_result, "report", {}) or {},
                )

            # ── Send via SharedStorage ──
            # Arm: via arm_action_q (arm_loop reads and servos)
            # NaN gate (inline — same as policy_loop)
            if not np.all(np.isfinite(arm_cmd)):
                continue
            if shared.safety_state.value == SafetyState.FAULT:
                continue

            shared.arm_action_q.put({"qpos": arm_cmd.copy()})

            # ── Tracking safety ──
            if np.all(np.isfinite(arm_qpos)):
                sent_cmd = arm_cmd.copy()
                tracking_err = np.max(np.abs(arm_qpos - sent_cmd))
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
                # Read current arm state for path planning
                _arm_result = shared.arm_state_ring.read_latest()
                if _arm_result is not None:
                    _ad, _, _ = _arm_result
                    _arm_qpos = np.asarray(_ad["qpos"][0], dtype=np.float64)
                    _home_qpos = np.array(arm_loop_cfg.home_qpos, dtype=np.float64)
                    _waypoints = plan_joint_home_path(_arm_qpos, _home_qpos, planner)
                    if _waypoints is not None and len(_waypoints) > 0:
                        print(f"  planned homing: {len(_waypoints)} waypoints (路径安全无碰撞)")
                    elif _waypoints is not None and len(_waypoints) == 0:
                        print(f"  planned homing: already close to home, skipping")
                    else:
                        print(f"  plan_joint_home_path returned None — falling back to joint-space interpolation")
                    shared.arm_action_q.put((HOME_SENTINEL, _waypoints))
                else:
                    shared.arm_action_q.put((HOME_SENTINEL, None))
                time.sleep(5.0)
                print("按 Q 退出...")
            if keys.is_pressed("q") or keys.is_pressed("esc"):
                break
            time.sleep(0.1)

        keys.stop()

        # ── Cleanup ──
        _shutdown_kb(shared, arm_proc, hand_proc)
        print("Done.")


def _shutdown_kb(shared: SharedStorage, arm_proc: mp.Process, hand_proc: mp.Process) -> None:
    """Graceful shutdown for keyboard teleop."""
    shared.is_running.value = False
    for p in (arm_proc, hand_proc):
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
            p.join(timeout=1)
    shared.close()


if __name__ == "__main__":
    main()
