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

from dexmani_real.config.defaults import arm, policy
from dexmani_real.planning import PlanningProfile, Pose, TeleopProfile, XArm7MotionPlanner
from dexmani_real.planning.path_utils import plan_joint_home_path, wrap_nearest_equivalent
from dexmani_real.planning.pose_utils import quat_multiply
from dexmani_real.robot.arm_loop import ArmLoopConfig
from dexmani_real.robot.arm_loop import arm_loop as _arm_loop
from dexmani_real.robot.hand_process import hand_loop as _hand_loop
from dexmani_real.robot.safety import SafetyState, transition
from dexmani_real.shm.shared_storage import HOME_SENTINEL, SharedStorage, shutdown_processes, wait_for_arm_home, wait_subsystem_ready
from dexmani_real.teleop.keyboard import GlobalKeyState, eef_delta_from_keys
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate_manager import RateManager
from dexmani_real.utils.signal_utils import ema_smooth_pose

try:
    from pynput import keyboard  # type: ignore[import-untyped]
except ImportError:
    raise ImportError("pynput is required for keyboard input. Install with: pip install pynput")

logger = get_logger(__name__)

# ═══════════════════════════════════════════════ Config


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

    # ── Motion tracing: track position pipeline during pure-axis motion ──
    trace_motion: bool = True
    trace_frame_interval: int = 10  # print every N frames


# Singleton config (edit defaults in the dataclass above).
_cfg = KeyboardTeleopConfig()

# Workspace bounds in WORLD frame (defined in defaults.py as world-frame coordinates).
# Base frame = world frame (identity transform).
WORKSPACE_BOUNDS = policy.workspace.as_array()


# ═══════════════════════════════════════════════ Helpers


def _print_motion_trace(
    loop_count: int,
    dx: np.ndarray,
    target_pos: np.ndarray,
    ik_target_pos: np.ndarray,
    eef_pos: np.ndarray,
    ik_fk_pos: np.ndarray,
    ik_fk_quat: np.ndarray,
    ik_target_quat: np.ndarray,
    report: dict,
) -> None:
    """Print pure-axis (+X or -X) motion trace — target → EMA → IK → FK pipeline."""
    pos_error_mm = float(np.linalg.norm(ik_target_pos - ik_fk_pos) * 1000)
    dot = float(min(np.abs(np.dot(ik_target_quat, ik_fk_quat)), 1.0))
    rot_error_deg = float(np.rad2deg(2.0 * np.arccos(dot)))
    raw_lead_mm = float(np.linalg.norm(target_pos - eef_pos) * 1000)
    ema_lead_mm = float(np.linalg.norm(ik_target_pos - eef_pos) * 1000)
    z_shift_mm = float((ik_fk_pos[2] - ik_target_pos[2]) * 1000)

    _fmt_vec = lambda v: f"({v[0]:+.0f},{v[1]:+.0f},{v[2]:+.0f})"
    print(
        f"[TRACE f={loop_count}] "
        f"dx={_fmt_vec(dx * 1000)} mm  "
        f"raw={_fmt_vec(target_pos * 1000)}  EMA={_fmt_vec(ik_target_pos * 1000)}  eef={_fmt_vec(eef_pos * 1000)} mm  "
        f"lead raw={raw_lead_mm:.0f} EMA={ema_lead_mm:.0f} mm  "
        f"err pos={pos_error_mm:.1f} rot={rot_error_deg:.2f} deg  "
        f"Z-off={z_shift_mm:+.1f} mm  "
        f"IK={report.get('method', '?')} att={', '.join(report.get('attempts', ['?']))}",
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
            print(f"  [WARN] {names[axis]}-axis at boundary [{lo:.2f}, {hi:.2f}] m", flush=True)
            wall_warned[axis] = True
            wall_timers[axis] = now


# ═══════════════════════════════════════════════ Main Loop


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Keyboard Teleop xArm7")
    parser.add_argument("--no-hand", action="store_true", help="Disable hand (no hand_loop spawned)")
    _args = parser.parse_args()

    _dt = 1.0 / _cfg.ctrl_hz
    _hand_enabled = not _args.no_hand
    print("=" * 60)
    print("  Keyboard Teleop — xArm7 (SharedStorage)")
    print(
        f"  step  pos={_cfg.delta_pos*1000:.0f} mm  rot={np.rad2deg(_cfg.delta_rpy):.1f} deg  dt={_dt*1000:.0f} ms  rate={_cfg.ctrl_hz:.0f} Hz"
    )
    print(f"  EMA   pos={policy.ema.alpha_pos:.2f}  rot={policy.ema.alpha_rot:.2f}")
    print(f"  Kp    cartesian={_cfg.cartesian_kp:.1f}")
    print(f"  WS    x{WORKSPACE_BOUNDS[0]}  y{WORKSPACE_BOUNDS[1]}  z{WORKSPACE_BOUNDS[2]}")
    print(f"  Hand  {'ON' if _hand_enabled else 'OFF'}")
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
    arm_proc.start()
    hand_proc: mp.Process | None = None
    if _hand_enabled:
        hand_proc = mp.Process(target=_hand_loop, args=(shared,), name="hand-kb", daemon=True)
        hand_proc.start()

    _procs = [arm_proc]
    if hand_proc is not None:
        _procs.append(hand_proc)
    if not wait_subsystem_ready(shared, [("arm", shared.arm_ready, 15)], _procs):
        shutdown_processes(shared, _procs)
        return
    if _hand_enabled and hand_proc is not None:
        shared.hand_ready.wait(timeout=15)  # optional — degrade gracefully

    transition(shared, SafetyState.ARMED)

    # Read initial state from rings.
    _arm_result = shared.arm_state_ring.read_latest()
    if _arm_result is None:
        logger.error("Cannot read initial arm state from ring — exiting")
        shutdown_processes(shared, [arm_proc] if hand_proc is None else [arm_proc, hand_proc])
        return
    _arm_data, _, _ = _arm_result
    arm_qpos = np.asarray(_arm_data["qpos"][0], dtype=np.float64)
    eef_pos = np.asarray(_arm_data["eef_pos"][0], dtype=np.float64)
    eef_rot6d = np.asarray(_arm_data["eef_rot6d"][0], dtype=np.float64)
    arm_connected = bool(_arm_data["connected"][0])
    if not arm_connected or not np.all(np.isfinite(arm_qpos)):
        logger.error("Arm state invalid: connected=%s, qpos_finite=%s", arm_connected, np.all(np.isfinite(arm_qpos)))
        shutdown_processes(shared, [arm_proc] if hand_proc is None else [arm_proc, hand_proc])
        return

    prev_qpos_cmd = arm_qpos.copy()
    # Initialize target in WORLD frame (consistent with keyboard operator's perspective).
    # Pinocchio FK gives base-frame EEF; base_to_world_pose is identity (base = world).
    _eef_pin = planner.compute_eef_pose_base(arm_qpos)
    _eef_world = planner.base_to_world_pose(_eef_pin)
    target_pos = _eef_world.p.copy()
    target_quat = _eef_world.q.copy()

    print(f"  arm     connected  qpos={np.round(np.rad2deg(arm_qpos), 1)} deg")
    print(f"  EEF     base=({eef_pos[0]:.3f},{eef_pos[1]:.3f},{eef_pos[2]:.3f}) m")
    print(f"  target  world=({target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f}) m  (Pinocchio FK -> world)")
    print(f"  quat    wx={target_quat[0]:.4f}  xyz=({target_quat[1]:.4f},{target_quat[2]:.4f},{target_quat[3]:.4f})")

    # ── 5. Keyboard input ──
    keys = GlobalKeyState()
    keys.start()
    print("\nKeyboard control active — [Q] quit")

    # ── 6. Main loop ──
    limiter = RateManager(1.0 / (1.0 / _cfg.ctrl_hz))
    running = True
    wall_warned = [False, False, False]
    wall_timers = [0.0, 0.0, 0.0]  # per-axis debounce (independent 3 s cooldown)
    loop_count = 0
    error_count = 0
    total_state_errors = 0  # cumulative arm state read failures
    max_consecutive_errors = 10
    _status_interval = 50  # frames between active status prints
    _idle_interval = 150  # frames between idle heartbeat prints
    consecutive_divergence = 0
    TRACKING_DIVERGENCE_THRESHOLD_RAD = 5.0
    start_time = time.perf_counter()
    prev_eef_pos: np.ndarray | None = None
    ik_outcome = "-"
    ik_fail_count = 0
    _last_ik_fail_reason = ""
    _last_ik_fail_time = 0.0
    _homed_during_session = False  # skip redundant post-loop home prompt

    # Cartesian EMA state (same smoothing as vr_teleop_policy)
    _prev_ema_pos: np.ndarray | None = None
    _prev_ema_quat: np.ndarray | None = None

    def _emergency_stop():
        """Set estop flag — arm_loop detects and stops."""
        nonlocal running
        shared.estop_request.value = True
        shared.is_running.value = False
        running = False

    print("\nEntering teleop loop...\n")

    fd = sys.stdin.fileno()
    old_termios = termios.tcgetattr(fd)
    new_termios = termios.tcgetattr(fd)
    new_termios[3] &= ~termios.ECHO
    termios.tcsetattr(fd, termios.TCSANOW, new_termios)

    try:
        while running:
            limiter.wait()
            loop_count += 1

            # ── Exit / estop ──
            if keys.is_pressed("esc"):
                print("\n[ESC] emergency stop", flush=True)
                _emergency_stop()
                break

            if keys.is_pressed("q"):
                print("\n[Q] quit", flush=True)
                running = False
                break

            if keys.is_pressed("r"):
                _homed_during_session = True
                elapsed = time.perf_counter() - start_time
                print(f"\n[T+{elapsed:.0f}s f={loop_count}] [R] return_home", flush=True)
                # Plan collision-safe path to home (same as VR policy)
                _home_qpos = np.array(arm_loop_cfg.home_qpos, dtype=np.float64)
                _waypoints = plan_joint_home_path(
                    arm_qpos, _home_qpos, planner, table_z_surface_m=arm.table_z_surface_m
                )
                if _waypoints is not None and len(_waypoints) > 0:
                    print(f"  home  path={len(_waypoints)} waypoints  collision-free", flush=True)
                elif _waypoints is not None and len(_waypoints) == 0:
                    print(f"  home  NO SAFE PATH — holding position", flush=True)
                else:
                    print(f"  home  already close to home", flush=True)
                shared.arm_action_q.put((HOME_SENTINEL, _waypoints))
                _prev_ema_pos = _prev_ema_quat = None
                consecutive_divergence = 0
                error_count = 0
                # Wait for homing to converge, then refresh state from ring.
                _home_arr = np.array(arm_loop_cfg.home_qpos, dtype=np.float64)
                _converged = wait_for_arm_home(shared, _home_arr, timeout_s=20.0)
                if not _converged:
                    print("  home  wait timeout — continuing", flush=True)
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
                total_state_errors += 1
                if error_count > max_consecutive_errors:
                    logger.error("Consecutive arm state read failures — emergency stop")
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
                total_state_errors += 1
                if error_count > 3:
                    logger.error("Arm disconnected — emergency stop")
                    _emergency_stop()
                    break
                continue

            error_count = 0

            # ── Safety: arm error ──
            if arm_error_code != 0:
                if arm_error_code in (22, 24, 31):
                    # arm_loop auto-clears these — just log and continue
                    if loop_count % _status_interval == 0:
                        logger.warning("Arm error C%d (arm_loop auto-recovering)", arm_error_code)
                else:
                    logger.error("Arm unrecoverable error C%d — emergency stop", arm_error_code)
                    _emergency_stop()
                    break

            if not np.all(np.isfinite(arm_qpos)):
                error_count += 1
                total_state_errors += 1
                continue

            # ── EEF target delta from keys ──
            dx, drpy = eef_delta_from_keys(keys, _cfg.delta_pos, _cfg.delta_rpy)

            # ── Periodic status (suppressed when idle — no keys pressed) ──
            _is_idle = np.all(dx == 0) and np.all(drpy == 0)
            if loop_count % _status_interval == 0:
                # Always track velocity baseline (even when idle) so the first
                # non-idle status line gets an accurate speed estimate.
                if prev_eef_pos is not None:
                    vel = np.linalg.norm(eef_pos - prev_eef_pos) / (_status_interval * _dt)
                else:
                    vel = 0.0
                prev_eef_pos = eef_pos.copy()
                if not _is_idle:
                    elapsed = time.perf_counter() - start_time
                    _eef_world_status = planner.base_to_world_pose(Pose(p=eef_pos, q=np.array([1.0, 0.0, 0.0, 0.0])))
                    _tw = _eef_world_status.p
                    print(
                        f"[T+{elapsed:.0f}s f={loop_count}] "
                        f"eef_w=({_tw[0]:.3f},{_tw[1]:.3f},{_tw[2]:.3f}) m  "
                        f"target_w=({target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f}) m  "
                        f"v={vel:.3f} m/s  ik={ik_outcome}",
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
                if loop_count % _idle_interval == 0:
                    elapsed = time.perf_counter() - start_time
                    print(
                        f"[idle T+{elapsed:.0f}s]  eef_w=({target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f}) m",
                        flush=True,
                    )
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
                dq = R.from_euler("xyz", drpy).as_quat(scalar_first=True)
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
                _eef_world_p = planner.base_to_world_pose(Pose(p=eef_pos, q=np.array([1.0, 0.0, 0.0, 0.0]))).p
                pos_error = target_pos - _eef_world_p
                if float(np.linalg.norm(pos_error)) > 0.003:  # 3 mm deadband
                    ik_target_pos = ik_target_pos + _cfg.cartesian_kp * pos_error
                    ik_target_pos = np.clip(
                        ik_target_pos,
                        WORKSPACE_BOUNDS[:, 0],
                        WORKSPACE_BOUNDS[:, 1],
                    )

            # ── IK solve (on EMA-smoothed world-frame target) ──
            target_pose_world = Pose(p=ik_target_pos, q=ik_target_quat)
            # Hand qpos from ring (for collision checking — optional)
            if _hand_enabled:
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
                    logger.warning("IK fail #%d: %s", ik_fail_count, reason)
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

            # ── Motion Trace: pure-axis pipeline diagnostics ──
            if (
                _cfg.trace_motion
                and loop_count % _cfg.trace_frame_interval == 0
                and dx[0] != 0
                and dx[1] == 0
                and dx[2] == 0
                and np.all(drpy == 0)
            ):
                ik_fk_pose_world = planner.kin.compute_eef_pose_world(ik_result.qpos)
                _eef_world_trace = planner.base_to_world_pose(Pose(p=eef_pos, q=np.array([1.0, 0.0, 0.0, 0.0])))
                _print_motion_trace(
                    loop_count=loop_count,
                    dx=dx,
                    target_pos=target_pos,
                    ik_target_pos=ik_target_pos,
                    eef_pos=_eef_world_trace.p,
                    ik_fk_pos=ik_fk_pose_world.p,
                    ik_fk_quat=ik_fk_pose_world.q,
                    ik_target_quat=ik_target_quat,
                    report=getattr(ik_result, "report", {}) or {},
                )

            # ── Send via SharedStorage ──
            # Arm: via arm_action_q (arm_loop reads and servos)
            # NaN gate (inline — same as policy_loop)
            if not np.all(np.isfinite(arm_cmd)):
                continue
            if shared.safety_state.value == int(SafetyState.FAULT):
                continue

            shared.arm_action_q.put({"qpos": arm_cmd.copy()})

            # ── Tracking safety ──
            if np.all(np.isfinite(arm_qpos)):
                sent_cmd = arm_cmd.copy()
                tracking_err = np.max(np.abs(arm_qpos - sent_cmd))
                if tracking_err > TRACKING_DIVERGENCE_THRESHOLD_RAD:
                    consecutive_divergence += 1
                    logger.warning(
                        "Tracking divergence: max_err=%.1f rad  frame=%d/3",
                        tracking_err,
                        consecutive_divergence,
                    )
                    if consecutive_divergence >= 3:
                        logger.error("Persistent tracking divergence — emergency stop")
                        _emergency_stop()
                        break
                else:
                    consecutive_divergence = 0

    finally:
        # Restore terminal first (pynput uses evdev, not termios — no conflict).
        time.sleep(0.05)
        termios.tcflush(fd, termios.TCIFLUSH)
        termios.tcsetattr(fd, termios.TCSADRAIN, old_termios)

        # ── Exit summary ──
        elapsed_total = time.perf_counter() - start_time
        ik_ok_count = loop_count - ik_fail_count
        print()
        print("=" * 60)
        print(
            f"  Session ended  elapsed={elapsed_total:.0f}s  frames={loop_count}  "
            f"ik_ok={ik_ok_count}  ik_fail={ik_fail_count}  state_errs={total_state_errors}"
        )
        print("=" * 60)

        # Post-loop: offer return_home (keys listener still alive).
        # Skip the prompt if the arm was already homed during the session and is
        # still near home — avoids confusing the operator with a redundant prompt.
        _near_home = False
        if _homed_during_session:
            _arm_result = shared.arm_state_ring.read_latest()
            if _arm_result is not None:
                _ad, _, _ = _arm_result
                _qpos = np.asarray(_ad["qpos"][0], dtype=np.float64)
                _home_qpos_arr = np.array(arm_loop_cfg.home_qpos, dtype=np.float64)
                if np.all(np.isfinite(_qpos)):
                    # Wrap home_qpos to nearest equivalent band before comparing
                    # (J1/J3/J5/J7 have 720° range; encoder may report on any band).
                    _wrapped_home = wrap_nearest_equivalent(
                        _home_qpos_arr,
                        _qpos,
                        arm_loop_cfg.joint_limit_lower,
                        arm_loop_cfg.joint_limit_upper,
                    )
                    _near_home = float(np.max(np.abs(_qpos - _wrapped_home))) < 0.05  # ~3°

        if _near_home:
            print("\nAlready at home — [Q] quit")
            while True:
                if keys.is_pressed("q") or keys.is_pressed("esc"):
                    break
                time.sleep(0.1)
        else:
            print("\n[R] return_home  [Q] quit")
            while True:
                if keys.is_pressed("r"):
                    # Read current arm state for path planning
                    _arm_result = shared.arm_state_ring.read_latest()
                    if _arm_result is not None:
                        _ad, _, _ = _arm_result
                        _arm_qpos = np.asarray(_ad["qpos"][0], dtype=np.float64)
                        _home_qpos = np.array(arm_loop_cfg.home_qpos, dtype=np.float64)
                        _waypoints = plan_joint_home_path(
                            _arm_qpos, _home_qpos, planner, table_z_surface_m=arm.table_z_surface_m
                        )
                        if _waypoints is not None and len(_waypoints) > 0:
                            print(f"  home  path={len(_waypoints)} waypoints  collision-free", flush=True)
                        elif _waypoints is not None and len(_waypoints) == 0:
                            print(f"  home  NO SAFE PATH — holding position", flush=True)
                        else:
                            print(f"  home  already close to home", flush=True)
                        shared.arm_action_q.put((HOME_SENTINEL, _waypoints))
                    else:
                        # Arm state ring has no valid frame — can't plan a path.
                        # _planned_homing will read the current position from the
                        # arm SDK directly and do its own wrapping + interpolation.
                        print("  home  arm state unavailable — falling back to SDK-based homing", flush=True)
                        shared.arm_action_q.put((HOME_SENTINEL, None))
                    _home_arr = np.array(arm_loop_cfg.home_qpos, dtype=np.float64)
                    _converged = wait_for_arm_home(shared, _home_arr, timeout_s=20.0)
                    if not _converged:
                        print("  home  wait timeout — continuing", flush=True)
                    print("[Q] quit")
                if keys.is_pressed("q") or keys.is_pressed("esc"):
                    break
                time.sleep(0.1)

        keys.stop()

        # ── Cleanup ──
        shutdown_processes(shared, [arm_proc] if hand_proc is None else [arm_proc, hand_proc])
        print("Shutdown complete.")



if __name__ == "__main__":
    main()
