#!/usr/bin/env python3
"""Keyboard teleop xArm7 — SharedStorage-based architecture.

Uses arm_loop process (Mode 6, 30Hz) for arm control via SharedStorage.

Usage:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate real_robot
    python examples/real/keyboard_teleop_real.py
    # Default: probe XHand, then fall back to arm-only if no slave is present.
    # --no-hand skips probing; --require-hand restores fail-closed startup.

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

import numpy as np
from scipy.spatial.transform import Rotation as R

from dexmani_real.config.runtime import resolve_runtime_config
from dexmani_real.planning import PlanningProfile, Pose, TeleopProfile, XArm7MotionPlanner
from dexmani_real.planning.path_utils import wrap_nearest_equivalent
from dexmani_real.planning.pose_utils import quat_multiply
from dexmani_real.policy.action_protocol import (
    ActionSafetyGateConfig,
    planner_action_safety_gate,
    publish_joint_targets,
)
from dexmani_real.robot.arm_loop import ArmLoopConfig
from dexmani_real.robot.arm_loop import arm_loop as _arm_loop
from dexmani_real.robot.hand_process import HandProcessConfig
from dexmani_real.robot.hand_process import hand_loop as _hand_loop
from dexmani_real.robot.safety import SafetyState, transition
from dexmani_real.shm.shared_storage import (
    SharedStorage,
    SharedStorageConfig,
    send_arm_home,
    shutdown_processes,
    wait_subsystem_ready,
)
from dexmani_real.teleop.keyboard import (
    GlobalKeyState,
    MotionActivityLatch,
    MotionTraceSample,
    ReleaseMotionTracer,
    eef_delta_from_keys,
)
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate_manager import RateManager
from dexmani_real.utils.signal_utils import ema_smooth_pose

try:
    from pynput import keyboard  # type: ignore[import-untyped]
except ImportError:
    raise ImportError("pynput is required for keyboard input. Install with: pip install pynput") from None

logger = get_logger(__name__)

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
    state_age_s: float,
    last_cmd_seq: int,
    queue_latency_s: float,
    apply_latency_s: float,
    sdk_duration_s: float,
    last_cmd_is_hold: bool,
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
        f"cmd={last_cmd_seq}{'H' if last_cmd_is_hold else ''} "
        f"age={state_age_s * 1000:.0f} q={queue_latency_s * 1000:.1f} "
        f"apply={apply_latency_s * 1000:.1f} sdk={sdk_duration_s * 1000:.1f} ms  "
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


def _runtime_health_error(
    shared: SharedStorage,
    arm_proc: object,
    hand_proc: object | None,
    *,
    hand_required: bool,
    heartbeat_timeouts_s: dict[str, float],
) -> str | None:
    """Return a fail-closed runtime health error, or ``None`` when healthy."""
    if shared.error_state.value:
        return "sticky error_state set by a worker"
    if shared.safety_state.value == int(SafetyState.FAULT):
        return "safety state is FAULT"
    if not getattr(arm_proc, "is_alive")():
        return "arm worker exited"
    if hand_required and (hand_proc is None or not getattr(hand_proc, "is_alive")()):
        return "hand worker exited"

    now = time.monotonic()
    heartbeats = [("arm", shared.arm_heartbeat_s)]
    if hand_required:
        heartbeats.append(("hand", shared.hand_heartbeat_s))
    for name, heartbeat in heartbeats:
        last_s = float(heartbeat.value)
        age_s = now - last_s if last_s > 0.0 else float("inf")
        if age_s > float(heartbeat_timeouts_s[name]):
            return f"{name} heartbeat stale ({age_s:.2f}s)"
    return None


# ═══════════════════════════════════════════════ Main Loop


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Keyboard Teleop xArm7")
    _hand_group = parser.add_mutually_exclusive_group()
    _hand_group.add_argument("--no-hand", action="store_true", help="Skip XHand probing and run arm-only")
    _hand_group.add_argument("--require-hand", action="store_true", help="Abort unless XHand becomes ready")
    parser.add_argument("--config", type=Path, default=None, help="Runtime JSON; CLI capability flags take precedence")
    _args = parser.parse_args()

    runtime = resolve_runtime_config(
        json_path=_args.config,
        cli_overrides={
            "policy.hand_enabled": False if _args.no_hand else True if _args.require_hand else None,
        },
    )
    _cfg = runtime.keyboard_teleop
    arm_runtime = runtime.arm
    hand_runtime = runtime.hand
    policy_runtime = runtime.policy
    heartbeat_timeouts_s = dict(runtime.safety.heartbeat_timeouts)
    workspace_bounds = np.array(
        [
            [policy_runtime.workspace.x_min, policy_runtime.workspace.x_max],
            [policy_runtime.workspace.y_min, policy_runtime.workspace.y_max],
            [policy_runtime.workspace.z_min, policy_runtime.workspace.z_max],
        ],
        dtype=np.float64,
    )

    _dt = 1.0 / _cfg.control_hz
    _hand_enabled = bool(runtime.policy.hand_enabled)
    _hand_required = bool(_args.require_hand)
    _hand_mode = "OFF" if not _hand_enabled else ("REQUIRED" if _hand_required else "OPTIONAL")
    print("=" * 60)
    print("  Keyboard Teleop — xArm7 (SharedStorage)")
    print(
        f"  step  pos={_cfg.delta_pos_m*1000:.0f} mm  rot={np.rad2deg(_cfg.delta_rpy_rad):.1f} deg  "
        f"dt={_dt*1000:.0f} ms  rate={_cfg.control_hz:.0f} Hz"
    )
    print(f"  EMA   pos={policy_runtime.ema.alpha_pos:.2f}  rot={policy_runtime.ema.alpha_rot:.2f}")
    print(f"  Kp    cartesian={_cfg.cartesian_kp:.1f}")
    if _cfg.release_trace_enabled:
        print(
            f"  Trace release={_cfg.release_trace_pre_frames}+{_cfg.release_trace_post_frames} frames  "
            f"cooldown={_cfg.release_trace_cooldown_s:.1f}s"
        )
    print(f"  WS    x{workspace_bounds[0]}  y{workspace_bounds[1]}  z{workspace_bounds[2]}")
    print(f"  Hand  {_hand_mode}")
    print(f"  Config {runtime.sha256[:12]}")
    print("=" * 60)

    # ── 1. Planner ──
    planner = XArm7MotionPlanner.create_default(
        planning_profile=PlanningProfile(max_waypoint_delta_deg=360.0),
        teleop_profile=TeleopProfile(
            max_pose_error_pos_m=0.02,
            max_pose_error_rot_rad=np.deg2rad(5.0),
        ),
        static_boxes=tuple(runtime.environment.static_boxes),
    )
    # These experimental entries may run without XHand hardware. Keep the
    # 19-DOF collision model conservative and deterministic by explicitly
    # using the configured open-hand pose until measured state is available.
    _assumed_hand_qpos = np.deg2rad(np.asarray(hand_runtime.home_qpos_deg, dtype=np.float64))
    planner.set_hand_qpos(_assumed_hand_qpos)
    planner.workspace_bounds = workspace_bounds.copy()
    action_safety_gate = planner_action_safety_gate(
        ActionSafetyGateConfig(
            arm_joint_lower_rad=tuple(arm_runtime.joint_limit_lower),
            arm_joint_upper_rad=tuple(arm_runtime.joint_limit_upper),
            hand_joint_lower_rad=tuple(hand_runtime.qpos_min_rad),
            hand_joint_upper_rad=tuple(hand_runtime.qpos_max_rad),
            arm_max_velocity_rad_s=float(np.deg2rad(arm_runtime.max_joint_velocity_deg_per_s)),
            hand_max_velocity_rad_s=(
                float(hand_runtime.max_delta_rad) * _cfg.control_hz
                if hand_runtime.max_delta_rad is not None
                else float(np.deg2rad(180.0))
            ),
            require_geometry_checks=True,
        ),
        planner=planner,
        table_z_surface_m=float(arm_runtime.table_z_surface_m),
        hand_safety_margin_m=float(arm_runtime.hand_safety_margin_m),
    )

    # ── 2. SharedStorage + subprocesses ──
    ctx = mp.get_context("spawn")
    shared = SharedStorage.create(
        prefix="dexmani_kb",
        config=SharedStorageConfig.from_runtime(runtime),
        mp_context=ctx,
    )
    arm_loop_cfg = ArmLoopConfig.from_runtime(runtime)
    arm_proc = ctx.Process(target=_arm_loop, args=(shared, arm_loop_cfg), name="arm-kb", daemon=False)
    arm_proc.start()
    hand_proc: object | None = None
    _procs: list[object] = [arm_proc]
    if not wait_subsystem_ready(shared, [("arm", shared.arm_ready, 15)], [arm_proc]):
        shutdown_processes(shared, _procs)
        return

    if _hand_enabled:
        _hand_cfg = HandProcessConfig.from_runtime(runtime, startup_failure_is_fatal=_hand_required)
        hand_proc = ctx.Process(target=_hand_loop, args=(shared, _hand_cfg), name="hand-kb", daemon=False)
        getattr(hand_proc, "start")()
        _procs.append(hand_proc)

        _hand_deadline_s = time.monotonic() + 15.0
        while (
            not shared.hand_ready.is_set() and getattr(hand_proc, "is_alive")() and time.monotonic() < _hand_deadline_s
        ):
            time.sleep(0.1)

        if not shared.hand_ready.is_set():
            if getattr(hand_proc, "is_alive")():
                logger.error("XHand startup did not finish within 15s — aborting to avoid an indeterminate worker")
                shutdown_processes(shared, _procs)
                return
            if _hand_required:
                logger.error("XHand is required but its worker exited before ready")
                shutdown_processes(shared, _procs)
                return
            _hand_enabled = False
            logger.warning(
                "XHand unavailable — continuing arm-only with configured open-hand collision geometry. "
                "If a physical hand is mounted, secure it in that pose or use --require-hand."
            )
            print("  Hand  unavailable → ARM-ONLY (open-hand collision model)")
        else:
            _hand_result = shared.hand_state_ring.read_latest()
            if _hand_result is None:
                logger.error("XHand ready event has no state frame — aborting before ARMED")
                shutdown_processes(shared, _procs)
                return
            _hand_data, _, _ = _hand_result
            _initial_hand_qpos = np.asarray(_hand_data["qpos"][0], dtype=np.float64)
            if not bool(_hand_data["connected"][0]) or not np.all(np.isfinite(_initial_hand_qpos)):
                logger.error("XHand initial state is disconnected or invalid — aborting before ARMED")
                shutdown_processes(shared, _procs)
                return
            planner.set_hand_qpos(_initial_hand_qpos)
            print("  Hand  connected → measured posture enabled for collision checks")

    if shared.error_state.value:
        logger.error("A required worker failed during startup — aborting before ARMED")
        shutdown_processes(shared, _procs)
        return

    if not _hand_enabled and _args.no_hand:
        print("  Hand  disabled → ARM-ONLY (open-hand collision model)")

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
    limiter = RateManager(_cfg.control_hz)
    motion_latch = MotionActivityLatch()
    release_tracer = (
        ReleaseMotionTracer(
            pre_frames=_cfg.release_trace_pre_frames,
            post_frames=_cfg.release_trace_post_frames,
            cooldown_s=_cfg.release_trace_cooldown_s,
        )
        if _cfg.release_trace_enabled
        else None
    )
    _last_translation_direction: np.ndarray | None = None
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
    ik_attempt_count = 0
    ik_ok_count = 0
    ik_fail_count = 0
    _last_ik_fail_reason = ""
    _last_ik_fail_time = 0.0
    _homed_during_session = False  # skip redundant post-loop home prompt
    _faulted = False
    _r_was_pressed = False

    # Seed EMA at the held target. Resetting it to None on every release makes
    # the next keypress bypass smoothing for one frame (an 8 mm position step),
    # followed by a smaller EMA step; that velocity discontinuity is visible as
    # a repeated start/stop twitch during short keyboard taps.
    _prev_ema_pos: np.ndarray | None = target_pos.copy()
    _prev_ema_quat: np.ndarray | None = target_quat.copy()

    def _emergency_stop():
        """Set estop flag — arm_loop detects and stops."""
        nonlocal running
        shared.estop_request.value = True
        transition(shared, SafetyState.FAULT)
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

            _health_error = _runtime_health_error(
                shared,
                arm_proc,
                hand_proc,
                hand_required=_hand_enabled,
                heartbeat_timeouts_s=heartbeat_timeouts_s,
            )
            if _health_error is not None:
                logger.error("Runtime health failure: %s", _health_error)
                transition(shared, SafetyState.FAULT)
                _faulted = True
                running = False
                break

            # ── Exit / estop ──
            if keys.is_pressed("esc"):
                print("\n[ESC] emergency stop", flush=True)
                _emergency_stop()
                break

            if keys.is_pressed("q"):
                print("\n[Q] quit", flush=True)
                running = False
                break

            _r_pressed = keys.is_pressed("r")
            _home_requested = _r_pressed and not _r_was_pressed
            _r_was_pressed = _r_pressed
            if _home_requested:
                elapsed = time.perf_counter() - start_time
                print(f"\n[T+{elapsed:.0f}s f={loop_count}] [R] return_home", flush=True)
                _home_qpos = np.array(arm_loop_cfg.home_qpos, dtype=np.float64)
                _home_ok = send_arm_home(
                    shared,
                    _home_qpos,
                    planner=planner,
                    table_z_surface_m=float(arm_runtime.table_z_surface_m),
                    current_qpos=arm_qpos,
                    heartbeat=False,
                    verbose=True,
                )
                _homed_during_session = _homed_during_session or _home_ok
                motion_latch.reset()
                if release_tracer is not None:
                    release_tracer.reset()
                _last_translation_direction = None
                consecutive_divergence = 0
                error_count = 0
                # Refresh state from ring after homing.
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
                    _prev_ema_pos = target_pos.copy()
                    _prev_ema_quat = target_quat.copy()
                prev_eef_pos = None
                ik_outcome = "home" if _home_ok else "home_failed"
                if shared.error_state.value or shared.safety_state.value == int(SafetyState.FAULT):
                    transition(shared, SafetyState.FAULT)
                    _faulted = True
                    running = False
                    break
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

            _arm_state_age_s = time.monotonic() - float(_arm_data["timestamp"][0])
            if _arm_state_age_s > 0.5:
                error_count += 1
                total_state_errors += 1
                if error_count > 3:
                    logger.error("Arm state stale for %.2fs — entering FAULT", _arm_state_age_s)
                    transition(shared, SafetyState.FAULT)
                    _faulted = True
                    running = False
                    break
                continue

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
                if arm_error_code == 24:
                    # arm_loop performs bounded C24 recovery.
                    if loop_count % _status_interval == 0:
                        logger.warning("Arm error C%d (arm_loop auto-recovering)", arm_error_code)
                elif arm_error_code in (22, 31):
                    logger.error("Arm collision fault C%d — entering FAULT", arm_error_code)
                    transition(shared, SafetyState.FAULT)
                    _faulted = True
                    running = False
                    break
                else:
                    logger.error("Arm unrecoverable error C%d — emergency stop", arm_error_code)
                    _emergency_stop()
                    break

            if not np.all(np.isfinite(arm_qpos)):
                error_count += 1
                total_state_errors += 1
                continue

            # ── EEF target delta from keys ──
            dx, drpy = eef_delta_from_keys(keys, _cfg.delta_pos_m, _cfg.delta_rpy_rad)

            # ── Periodic status (suppressed when idle — no keys pressed) ──
            _is_idle = np.all(dx == 0) and np.all(drpy == 0)
            _motion_active = not _is_idle
            _release_edge = motion_latch.update(_motion_active)
            _translation_norm = float(np.linalg.norm(dx))
            if _translation_norm > 0.0:
                _last_translation_direction = dx / _translation_norm
            elif _motion_active:
                # Rotation-only activity must not inherit an old translation
                # direction and produce a misleading release trace.
                _last_translation_direction = None

            # High-rate release window. This is read-only instrumentation: the
            # sample is aligned to the arm state timestamp and the last queued
            # joint target, and never publishes a control action. Avoid the
            # extra FK/delta work during unrelated idle intervals.
            _release_sample_needed = release_tracer is not None and (
                release_tracer.active or _translation_norm > 0.0 or _release_edge
            )
            if release_tracer is not None and _release_sample_needed:
                _eef_world_diag = planner.base_to_world_pose(Pose(p=eef_pos, q=np.array([1.0, 0.0, 0.0, 0.0]))).p
                try:
                    _command_world_diag = planner.kin.compute_eef_pose_world(prev_qpos_cmd).p
                    _qpos_error_diag = float(np.max(np.abs(planner.ik_mgr.compute_qpos_delta(prev_qpos_cmd, arm_qpos))))
                except (ValueError, RuntimeError):
                    logger.warning("Release-motion diagnostic FK/delta failed", exc_info=True)
                    _command_world_diag = target_pos.copy()
                    _qpos_error_diag = float(np.max(np.abs(prev_qpos_cmd - arm_qpos)))
                _release_sample = MotionTraceSample(
                    frame=loop_count,
                    timestamp_s=float(_arm_data["timestamp"][0]),
                    input_active=_motion_active,
                    eef_pos_m=_eef_world_diag,
                    command_pos_m=_command_world_diag,
                    qpos_error_rad=_qpos_error_diag,
                    qvel_peak_rad_s=float(np.max(np.abs(arm_qvel))),
                    state_age_s=_arm_state_age_s,
                    queue_latency_s=float(_arm_data["last_cmd_queue_latency_s"][0]),
                    apply_latency_s=float(_arm_data["last_cmd_apply_latency_s"][0]),
                )
                _release_lines = release_tracer.observe(
                    _release_sample,
                    release_edge=_release_edge,
                    translation_direction=_last_translation_direction,
                )
                if _release_lines:
                    # One terminal write after capture avoids injecting stdout
                    # latency into each measured 30 Hz control interval.
                    print("\n".join(_release_lines), flush=True)

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

            # No input → keep the last accepted target. arm_loop already
            # resends its last target every Mode 6 tick, so enqueueing a
            # duplicate HOLD here only consumes an ordered FIFO slot. A rapid
            # re-press would then execute one stale no-motion command before
            # the new command, producing a 33 ms start/stop notch.
            if _is_idle:
                if _release_edge:
                    # Keep the Cartesian accumulator consistent with that joint
                    # target. Snapping to measured state would command a stale
                    # backward step; keeping an empty EMA state would bypass
                    # smoothing on the next press.
                    _hold_pose_world = planner.kin.compute_eef_pose_world(prev_qpos_cmd)
                    target_pos = _hold_pose_world.p.copy()
                    target_quat = _hold_pose_world.q.copy()
                    _prev_ema_pos = target_pos.copy()
                    _prev_ema_quat = target_quat.copy()
                    if _cfg.trace_motion:
                        print(f"[HOLD f={loop_count}] key-release local-last-target (queue unchanged)", flush=True)
                if loop_count % _idle_interval == 0:
                    elapsed = time.perf_counter() - start_time
                    _idle_pose_world = planner.kin.compute_eef_pose_world(arm_qpos)
                    _idle_eef = _idle_pose_world.p
                    print(
                        f"[idle T+{elapsed:.0f}s]  "
                        f"eef_w=({_idle_eef[0]:.3f},{_idle_eef[1]:.3f},{_idle_eef[2]:.3f}) m",
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
            # workspace_bounds is the world-frame AABB of the base workspace box.
            target_pos = np.clip(target_pos, workspace_bounds[:, 0], workspace_bounds[:, 1])
            for axis in range(3):
                _wall_check(axis, target_pos, workspace_bounds, wall_warned, wall_timers)

            if np.any(drpy != 0):
                dq = R.from_euler("xyz", drpy).as_quat(scalar_first=True)
                target_quat = quat_multiply(dq, target_quat)

            # ── Cartesian EMA (before IK, same as vr_teleop_policy pipeline) ──
            # Smooths target trajectory to prevent IK discontinuities from
            # abrupt keypress changes.  At the configured alpha=0.60, the
            # theoretical ramp lag is 5.3 mm at the 8 mm step. The much larger
            # observed lead is therefore downstream tracking dynamics, not EMA
            # alone.
            if _prev_ema_pos is not None and _prev_ema_quat is not None:
                ik_target_pos, ik_target_quat = ema_smooth_pose(
                    target_pos,
                    target_quat,
                    _prev_ema_pos,
                    _prev_ema_quat,
                    policy_runtime.ema.alpha_pos,
                    policy_runtime.ema.alpha_rot,
                )
            else:
                ik_target_pos, ik_target_quat = target_pos.copy(), target_quat.copy()
            _prev_ema_pos = ik_target_pos.copy()
            _prev_ema_quat = ik_target_quat.copy()

            # ── Cartesian P-term: amplify position error → reduce tracking lag ──
            # The log showed about 60 mm following distance at the configured
            # 0.24 m/s command. Cartesian gain remains disabled by default: the
            # new timestamps should be measured before tuning closed-loop gain,
            # especially because Mode 6 already smooths the joint trajectory.
            if _cfg.cartesian_kp > 0:
                _eef_world_p = planner.base_to_world_pose(Pose(p=eef_pos, q=np.array([1.0, 0.0, 0.0, 0.0]))).p
                pos_error = target_pos - _eef_world_p
                if float(np.linalg.norm(pos_error)) > 0.003:  # 3 mm deadband
                    ik_target_pos = ik_target_pos + _cfg.cartesian_kp * pos_error
                    ik_target_pos = np.clip(
                        ik_target_pos,
                        workspace_bounds[:, 0],
                        workspace_bounds[:, 1],
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
            ik_attempt_count += 1
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
                _prev_ema_pos = target_pos.copy()
                _prev_ema_quat = target_quat.copy()
                ik_outcome = "held"
                continue

            ik_outcome = "ok"
            ik_ok_count += 1
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
                    state_age_s=_arm_state_age_s,
                    last_cmd_seq=int(_arm_data["last_cmd_seq"][0]),
                    queue_latency_s=float(_arm_data["last_cmd_queue_latency_s"][0]),
                    apply_latency_s=float(_arm_data["last_cmd_apply_latency_s"][0]),
                    sdk_duration_s=float(_arm_data["last_cmd_sdk_duration_s"][0]),
                    last_cmd_is_hold=bool(_arm_data["last_cmd_is_hold"][0]),
                )

            # ── Send via SharedStorage ──
            # Arm: via arm_action_q (arm_loop reads and servos)
            # NaN gate (inline — same as policy_loop)
            if not np.all(np.isfinite(arm_cmd)):
                continue
            if shared.safety_state.value == int(SafetyState.FAULT):
                continue

            published_candidate = publish_joint_targets(
                shared,
                arm_cmd,
                prepare_timeout_s=0.06,
                dt_s=_dt,
                safety_gate=action_safety_gate,
            )
            if published_candidate is None:
                logger.error("keyboard teleop: arm prepare/commit failed")
                shared.error_state.value = True
                break
            assert published_candidate.arm_qpos is not None
            arm_cmd = np.asarray(published_candidate.arm_qpos, dtype=np.float64)
            prev_qpos_cmd = arm_cmd.copy()

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
        print()
        print("=" * 60)
        print(
            f"  Session ended  elapsed={elapsed_total:.0f}s  frames={loop_count}  "
            f"ik_attempts={ik_attempt_count}  ik_ok={ik_ok_count}  "
            f"ik_fail={ik_fail_count}  state_errs={total_state_errors}"
        )
        print("=" * 60)

        # Post-loop: offer return_home (keys listener still alive).
        # Skip the prompt if the arm was already homed during the session and is
        # still near home — avoids confusing the operator with a redundant prompt.
        _near_home = False
        _home_prompt_allowed = (
            not _faulted
            and not shared.error_state.value
            and not shared.estop_request.value
            and shared.is_running.value
            and shared.safety_state.value != int(SafetyState.FAULT)
        )
        if _home_prompt_allowed and _homed_during_session:
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

        if not _home_prompt_allowed:
            print("\nFAULT/E-stop active — return_home is disabled; shutting down")
        elif _near_home:
            print("\nAlready at home — [Q] quit")
            while True:
                _health_error = _runtime_health_error(
                    shared,
                    arm_proc,
                    hand_proc,
                    hand_required=_hand_enabled,
                    heartbeat_timeouts_s=heartbeat_timeouts_s,
                )
                if _health_error is not None:
                    logger.error("Runtime health failure while waiting to quit: %s", _health_error)
                    transition(shared, SafetyState.FAULT)
                    break
                if keys.is_pressed("q") or keys.is_pressed("esc"):
                    break
                time.sleep(0.1)
        else:
            print("\n[R] return_home  [Q] quit")
            _post_r_was_pressed = keys.is_pressed("r")
            while True:
                _health_error = _runtime_health_error(
                    shared,
                    arm_proc,
                    hand_proc,
                    hand_required=_hand_enabled,
                    heartbeat_timeouts_s=heartbeat_timeouts_s,
                )
                if _health_error is not None:
                    logger.error("Runtime health failure in post-loop prompt: %s", _health_error)
                    transition(shared, SafetyState.FAULT)
                    break
                _post_r_pressed = keys.is_pressed("r")
                _post_home_requested = _post_r_pressed and not _post_r_was_pressed
                _post_r_was_pressed = _post_r_pressed
                if _post_home_requested:
                    _home_qpos = np.array(arm_loop_cfg.home_qpos, dtype=np.float64)
                    _home_ok = send_arm_home(
                        shared,
                        _home_qpos,
                        planner=planner,
                        table_z_surface_m=float(arm_runtime.table_z_surface_m),
                        heartbeat=False,
                        verbose=True,
                    )
                    if shared.error_state.value or shared.safety_state.value == int(SafetyState.FAULT):
                        transition(shared, SafetyState.FAULT)
                        break
                    if not _home_ok:
                        print("  arm: return_home was not executed; inspect the reason above")
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
