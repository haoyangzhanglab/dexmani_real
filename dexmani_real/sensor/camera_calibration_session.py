"""Interactive xArm7/RealSense lifecycle for ArUco eye-to-hand calibration.

Computes T_world_camera by detecting ArUco markers on the end-effector from a
fixed tripod-mounted camera and solving the hand-eye transform across five
OpenCV algorithms.

Results are written to ``dexmani_real/config/cameras.json``, compatible with
the ``CameraCalib`` config loader.

Hardware preparation:

  1. Print an ArUco 7x7_50 marker (ID=1), size 98.2 mm × 98.2 mm.
  2. Attach the marker flat on the end-effector, facing the camera.
  3. Fix the RealSense camera on a tripod, covering the workspace.
  4. Ensure conda environment has: pyrealsense2, opencv-python, scipy.

Usage::

    conda activate real_robot
    python examples/calibrate_camera.py [--serial SERIAL] [--config YAML]

Controls:

  WASD / arrows     translate EEF
  ← →              roll (about X)
  I / K            pitch (about Y)
  J / L            yaw (about Z)
  SPACE            capture calibration sample (requires ArUco detection)
  BACKSPACE         undo last sample
  X                 delete worst-residual frame (after ENTER evaluation)
  ENTER             compute calibration and write cameras.json (min 10 samples)
  R                 return home (collision-safe path)
  Q                 quit (discard data)
  ESC               emergency stop (FAULT)

XHand is optional, but ``--hand-geometry`` is a physical assertion used for
collision checks: pass ``absent`` only when it is not mounted, or
``secured-home`` only when an installed hand is physically fixed at its
configured home pose.  The assertion never disables collision checking.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

from dexmani_real.config.runtime import ArmLoopConfig, ResolvedRuntimeConfig
from dexmani_real.planning import Pose, TeleopProfile, XArm7MotionPlanner
from dexmani_real.planning.kinematics import make_arm_fk
from dexmani_real.planning.pose_utils import quat_multiply, rot6d_to_quat_wxyz
from dexmani_real.policy.safety import (
    SafetyGate,
    advance_run_generation,
    planner_action_safety_gate,
    publish_joint_targets,
)
from dexmani_real.robot.arm_loop import arm_loop
from dexmani_real.robot.homing import ArmHomeConfig, execute_arm_home
from dexmani_real.robot.safety import SafetyState, require_transition, transition
from dexmani_real.runtime.processes import WorkerSpec, build_processes, start_processes
from dexmani_real.runtime.supervisor import shutdown_processes, wait_subsystem_ready
from dexmani_real.sensor.camera_calibration import (
    ARUCO_DICT,
    ARUCO_DICT_NAME,
    CAMERA_CALIBRATION_PATH,
    ArucoConfig,
    CalibrationConfig,
    CalibrationSamples,
    calibrate_and_select,
    detect_aruco_pose,
    draw_calibration_overlay,
    marker_corners_3d,
    save_camera_calibration,
)
from dexmani_real.shm.shared_storage import (
    SharedStorage,
    SharedStorageConfig,
    read_arm_state_dict,
)
from dexmani_real.teleop.keyboard import GlobalKeyState, eef_delta_from_keys
from dexmani_real.utils.hand_health import validate_arm_feedback
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate_manager import RateManager

logger = get_logger(__name__)

_WINDOW_NAME = "ArUco Calibration"
_INITIAL_STATE_POLL_S = 0.05
_IK_WARNING_INTERVAL_S = 1.0
_BOUNDARY_WARN_INTERVAL_S = 2.0

_CAMERA_WIDTH = 640
_CAMERA_HEIGHT = 480
_CAMERA_FPS = 30
_CAMERA_WARMUP_FRAMES = 30


def _detect_aruco_stable(
    pipeline: Any,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
    *,
    marker_size_m: float,
    target_id: int | None,
    n_frames: int = 5,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Capture N frames and return median ArUco pose for noise reduction."""
    rvecs_all: list[np.ndarray] = []
    tvecs_all: list[np.ndarray] = []
    for _ in range(n_frames):
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        image = np.asanyarray(color_frame.get_data())
        result = detect_aruco_pose(
            image,
            intrinsics,
            distortion,
            marker_size_m=marker_size_m,
            target_id=target_id,
        )
        if result is not None:
            rvecs_all.append(result[0])
            tvecs_all.append(result[1])

    if len(rvecs_all) < max(1, n_frames // 2):
        return None
    return np.median(rvecs_all, axis=0), np.median(tvecs_all, axis=0)


def _start_camera(serial: str | None = None) -> tuple[Any, str, np.ndarray, np.ndarray]:
    """Start RealSense color stream and return (pipeline, serial, K, dist)."""
    import pyrealsense2 as rs  # type: ignore[import-not-found]

    pipeline = rs.pipeline()
    rs_config = rs.config()
    if serial:
        rs_config.enable_device(serial)
    rs_config.enable_stream(
        rs.stream.color, _CAMERA_WIDTH, _CAMERA_HEIGHT, rs.format.bgr8, _CAMERA_FPS
    )
    started = False
    try:
        profile = pipeline.start(rs_config)
        started = True
        device = profile.get_device()
        serial = device.get_info(rs.camera_info.serial_number)
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        intrinsics = np.array(
            [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]],
            dtype=np.float64,
        )
        distortion = np.array(intr.coeffs, dtype=np.float64)

        # Warm-up: first few frames may have unstable auto-exposure.
        for _ in range(_CAMERA_WARMUP_FRAMES):
            pipeline.wait_for_frames()
    except Exception:
        if started:
            try:
                pipeline.stop()
            except Exception:
                logger.error(
                    "camera cleanup after startup failure failed", exc_info=True
                )
        raise

    return pipeline, serial, intrinsics, distortion


def _workspace_bounds(runtime: ResolvedRuntimeConfig) -> np.ndarray:
    w = runtime.policy.workspace
    return np.array(
        [[w.x_min, w.x_max], [w.y_min, w.y_max], [w.z_min, w.z_max]], dtype=np.float64
    )


def _build_planner_and_gate(
    runtime: ResolvedRuntimeConfig,
) -> tuple[XArm7MotionPlanner, SafetyGate, np.ndarray]:
    workspace = _workspace_bounds(runtime)
    planner = XArm7MotionPlanner.create_default(
        teleop_profile=TeleopProfile(
            max_pose_error_pos_m=float(runtime.keyboard_teleop.ik_max_pose_error_pos_m),
            max_pose_error_rot_rad=float(
                runtime.keyboard_teleop.ik_max_pose_error_rot_rad
            ),
        ),
        static_boxes=tuple(runtime.environment.static_boxes),
        table=runtime.environment.table,
    )
    planner.workspace_bounds = workspace.copy()
    planner.set_hand_qpos(
        np.deg2rad(np.asarray(runtime.hand.home_qpos_deg, dtype=np.float64))
    )
    gate = planner_action_safety_gate(
        planner=planner,
        arm_joint_lower_rad=tuple(runtime.arm.joint_limit_lower),
        arm_joint_upper_rad=tuple(runtime.arm.joint_limit_upper),
        hand_joint_lower_rad=tuple(runtime.hand.qpos_min_rad),
        hand_joint_upper_rad=tuple(runtime.hand.qpos_max_rad),
    )
    return planner, gate, workspace


def _read_initial_arm(
    shared: SharedStorage, runtime: ResolvedRuntimeConfig
) -> dict[str, Any] | None:
    deadline_s = time.monotonic() + float(runtime.safety.readiness_timeouts_s["arm"])
    while time.monotonic() < deadline_s:
        state = read_arm_state_dict(shared)
        if state is not None:
            issue = validate_arm_feedback(
                connected=state["connected"],
                state_valid=state["state_valid"],
                source_monotonic_ns=state["source_monotonic_ns"],
                now_monotonic_ns=time.monotonic_ns(),
                max_age_s=float(runtime.policy.arm_state_stale_threshold_s),
                qpos=state["qpos"],
                qvel=state["qvel"],
            )
            if issue is None and state["error_code"] == 0:
                return state
        time.sleep(_INITIAL_STATE_POLL_S)
    return None


def _set_fault(shared: SharedStorage, reason: str, *, estop: bool = False) -> None:
    logger.error("Calibration fault: %s", reason)
    if estop:
        shared.estop_request.value = True
    shared.error_state.value = True
    transition(shared, SafetyState.FAULT)


def _runtime_issue(
    shared: SharedStorage, arm_process: Any, heartbeat_timeout_s: float
) -> str | None:
    if shared.estop_request.value:
        return "e-stop is requested"
    if shared.error_state.value:
        return "a worker set the sticky error latch"
    if int(shared.safety_state.value) == int(SafetyState.FAULT):
        return "safety state is FAULT"
    if not arm_process.is_alive():
        return "arm worker exited"
    heartbeat_s = shared.get_heartbeat("arm")
    now_s = time.monotonic()
    age_s = now_s - heartbeat_s
    if (
        not np.isfinite(heartbeat_s)
        or heartbeat_s <= 0.0
        or heartbeat_s > now_s
        or age_s > heartbeat_timeout_s
    ):
        return f"arm heartbeat stale ({age_s:.2f}s)"
    return None


def _eef_rpy_from_rot6d(rot6d: np.ndarray) -> np.ndarray:
    """Convert arm EEF rot6d to RPY (rad) via the canonical library path."""
    q_wxyz = rot6d_to_quat_wxyz(rot6d)
    rpy = Rotation.from_quat(np.roll(q_wxyz, -1)).as_euler("xyz", degrees=False)
    return np.asarray(rpy, dtype=np.float64)


def _capture_calibration_sample(
    shared: SharedStorage,
    runtime: ResolvedRuntimeConfig,
    pipeline: Any,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
    samples: CalibrationSamples,
    aruco_config: ArucoConfig,
) -> None:
    """Capture one synchronized marker/arm observation into ``samples``."""
    print(f"\n  [{len(samples) + 1}] capturing ArUco pose...", end=" ", flush=True)
    try:
        aruco_pose = _detect_aruco_stable(
            pipeline,
            intrinsics,
            distortion,
            marker_size_m=aruco_config.marker_size_m,
            target_id=aruco_config.target_id,
            n_frames=aruco_config.capture_frames,
        )
    except Exception as exc:
        logger.warning("capture failed", exc_info=True)
        print(f"FAILED — {exc}, skipped")
        return
    if aruco_pose is None:
        print("FAILED — marker not detected, skipped")
        return

    arm_state = read_arm_state_dict(shared)
    if arm_state is None:
        print("FAILED — arm state unavailable, skipped")
        return
    feedback_issue = validate_arm_feedback(
        connected=arm_state["connected"],
        state_valid=arm_state["state_valid"],
        source_monotonic_ns=arm_state["source_monotonic_ns"],
        now_monotonic_ns=time.monotonic_ns(),
        max_age_s=float(runtime.policy.arm_state_stale_threshold_s),
        qpos=np.asarray(arm_state["qpos"], dtype=np.float64),
        qvel=arm_state["qvel"],
    )
    if feedback_issue is not None:
        print(f"FAILED — {feedback_issue}, skipped")
        return

    marker_rvec, marker_tvec = aruco_pose
    eef_pos_base_m, eef_rot6d_base = make_arm_fk().compute(
        np.asarray(arm_state["qpos"], dtype=np.float64)
    )
    eef_rpy_base_rad = _eef_rpy_from_rot6d(eef_rot6d_base)
    samples.append(
        eef_pos_base_m,
        eef_rpy_base_rad,
        marker_rvec,
        marker_tvec,
    )
    print(
        f"OK (total {len(samples)})  EE={np.round(eef_pos_base_m, 3)}m  "
        f"marker_dist={np.linalg.norm(marker_tvec):.3f}m"
    )


def _solve_and_save_calibration(
    samples: CalibrationSamples,
    planner: XArm7MotionPlanner,
    camera_serial: str,
    config: CalibrationConfig,
) -> np.ndarray | None:
    """Solve, report, quality-gate, and explicitly persist accepted samples."""
    sample_count = len(samples)
    if sample_count < config.min_samples:
        print(
            f"  need at least {config.min_samples} samples, have {sample_count} "
            "— keep collecting"
        )
        return None
    print(f"\n  computing hand-eye calibration ({sample_count} samples, 5 methods)...")
    try:
        T_base_camera, method, errors_mm, errors_deg, method_table = (
            calibrate_and_select(*samples.solver_inputs())
        )
    except Exception as exc:
        logger.warning("solve failed", exc_info=True)
        print(f"FAILED — {exc}, skipped")
        return None

    T_world_base = np.eye(4, dtype=np.float64)
    T_world_base[:3, :3] = Rotation.from_quat(
        np.roll(np.asarray(planner.kin.base_pose_world.q, dtype=np.float64), -1),
    ).as_matrix()
    T_world_base[:3, 3] = np.asarray(
        planner.kin.base_pose_world.p,
        dtype=np.float64,
    )
    T_world_camera = T_world_base @ T_base_camera
    position_std_mm = float(errors_mm.std())
    rotation_std_deg = float(errors_deg.std())
    samples.set_residuals(errors_mm)

    print("  method consistency (mm, lower is better):")
    for name, score_mm in method_table:
        selected = "  ← selected" if name == method else ""
        score_text = "  FAILED" if np.isnan(score_mm) else f"{score_mm:7.1f}"
        print(f"    {name:11s} {score_text}{selected}")
    print(f"  quality ({method}, T_ee_marker consistency):")
    print(
        f"    position mean={errors_mm.mean():.1f}mm "
        f"max={errors_mm.max():.1f}mm std={position_std_mm:.1f}mm"
    )
    print(
        f"    rotation mean={errors_deg.mean():.2f}° "
        f"max={errors_deg.max():.2f}° std={rotation_std_deg:.2f}°"
    )
    worst_index = int(np.argmax(errors_mm))
    print("  per-frame residuals (mm, larger = more suspicious):")
    for index, residual_mm in enumerate(errors_mm):
        bar = "█" * min(
            30,
            int(residual_mm / max(errors_mm.max(), 1e-9) * 30),
        )
        flag = "  ← worst, press X to remove" if index == worst_index else ""
        print(f"    #{index + 1:2d} {residual_mm:6.1f} {bar}{flag}")
    print(f"  T_world_camera position: {np.round(T_world_camera[:3, 3], 4)}m")

    rejection_reasons: list[str] = []
    if position_std_mm > config.max_consistency_std_mm:
        rejection_reasons.append(
            f"pos std={position_std_mm:.1f}mm > {config.max_consistency_std_mm:.1f}mm"
        )
    if rotation_std_deg > config.max_consistency_rot_std_deg:
        rejection_reasons.append(
            f"rot std={rotation_std_deg:.2f}° > "
            f"{config.max_consistency_rot_std_deg:.1f}°"
        )
    if rejection_reasons:
        print(
            f"  REJECTED (quality gate: {'; '.join(rejection_reasons)}) "
            "— increase rotation variety and retry"
        )
        return None

    try:
        save_camera_calibration(
            T_world_camera,
            camera_serial,
            CAMERA_CALIBRATION_PATH,
        )
    except Exception as exc:
        logger.warning("save failed", exc_info=True)
        print(f"FAILED — {exc}, skipped")
        return None
    print(
        f"  ACCEPTED ({method}, pos std={position_std_mm:.1f}mm, "
        f"rot std={rotation_std_deg:.2f}°)"
    )
    return T_world_camera


@dataclass
class _CalibrationLoopState:
    """Mutable operator and motion state for one calibration control loop."""

    samples: CalibrationSamples
    current_qpos: np.ndarray
    previous_command: np.ndarray
    target_pos: np.ndarray
    target_quat: np.ndarray
    calibration_saved: bool = False
    home_key_down: bool = False
    motion_active: bool = False
    frame: int = 0
    last_ik_warning_s: float = 0.0
    blocked_keys: tuple[str, ...] | None = None
    last_boundary_warning_s: float = 0.0

    @classmethod
    def from_arm_state(
        cls, planner: XArm7MotionPlanner, arm_state: dict[str, Any]
    ) -> "_CalibrationLoopState":
        current_qpos = np.asarray(arm_state["qpos"], dtype=np.float64)
        pose = planner.kin.compute_eef_pose_world(current_qpos)
        return cls(
            samples=CalibrationSamples(),
            current_qpos=current_qpos,
            previous_command=current_qpos.copy(),
            target_pos=pose.p.copy(),
            target_quat=pose.q.copy(),
        )


@dataclass(frozen=True)
class _CalibrationArmFeedback:
    qpos: np.ndarray | None
    issue: str = ""
    error_code: int = 0


class _HomeKeyOutcome(str, Enum):
    IDLE = "idle"
    COMPLETED = "completed"
    FAULT = "fault"


def _show_calibration_preview(
    pipeline: Any,
    detector: cv2.aruco.ArucoDetector,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
    samples: CalibrationSamples,
    calib_cfg: CalibrationConfig,
    aruco_cfg: ArucoConfig,
    marker_corners: np.ndarray,
    previous_image: np.ndarray | None,
) -> np.ndarray | None:
    """Poll and display the newest camera preview without blocking control."""
    frames = pipeline.poll_for_frames()
    color_frame = frames.get_color_frame() if frames else None
    display_image = previous_image
    if color_frame:
        image = np.asanyarray(color_frame.get_data()).copy()
        display_image, _ = draw_calibration_overlay(
            image,
            detector,
            intrinsics,
            distortion,
            n_samples=len(samples),
            min_samples=calib_cfg.min_samples,
            target_id=aruco_cfg.target_id,
            marker_corners=marker_corners,
            marker_size_m=aruco_cfg.marker_size_m,
        )
    if display_image is not None:
        cv2.imshow(_WINDOW_NAME, display_image)
    cv2.waitKey(1)
    return display_image


def _handle_calibration_sample_events(
    shared: SharedStorage,
    runtime: ResolvedRuntimeConfig,
    planner: XArm7MotionPlanner,
    pipeline: Any,
    serial: str,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
    keys: GlobalKeyState,
    state: _CalibrationLoopState,
    calib_cfg: CalibrationConfig,
    aruco_cfg: ArucoConfig,
) -> None:
    """Drain edge-triggered capture, undo, reject, and solve events."""
    event = keys.pop_event()
    while event is not None:
        if event == "space":
            _capture_calibration_sample(
                shared,
                runtime,
                pipeline,
                intrinsics,
                distortion,
                state.samples,
                aruco_cfg,
            )
        elif event == "backspace":
            if state.samples.pop_last():
                print(f"  undone, {len(state.samples)} remaining")
            else:
                print("  (no samples to undo)")
        elif event == "x":
            removed = state.samples.pop_worst()
            if removed is None:
                print(
                    "  (press ENTER first to evaluate quality, then X to remove worst)"
                )
            else:
                index, residual_mm = removed
                print(
                    f"  removed worst frame #{index + 1} "
                    f"(residual {residual_mm:.1f}mm), {len(state.samples)} remaining "
                    "— press ENTER to recompute"
                )
        elif event == "enter":
            transform = _solve_and_save_calibration(
                state.samples,
                planner,
                serial,
                calib_cfg,
            )
            state.calibration_saved = transform is not None
        event = keys.pop_event()


def _read_calibration_arm_feedback(
    shared: SharedStorage, runtime: ResolvedRuntimeConfig
) -> _CalibrationArmFeedback:
    arm_state = read_arm_state_dict(shared)
    if arm_state is None:
        return _CalibrationArmFeedback(None, "arm state is unavailable")
    qpos = np.asarray(arm_state["qpos"], dtype=np.float64)
    issue = validate_arm_feedback(
        connected=arm_state["connected"],
        state_valid=arm_state["state_valid"],
        source_monotonic_ns=arm_state["source_monotonic_ns"],
        now_monotonic_ns=time.monotonic_ns(),
        max_age_s=float(runtime.policy.arm_state_stale_threshold_s),
        qpos=qpos,
        qvel=arm_state["qvel"],
    )
    return _CalibrationArmFeedback(
        None if issue is not None else qpos,
        issue or "",
        int(arm_state["error_code"]),
    )


def _publish_calibration_quit_hold(
    shared: SharedStorage,
    runtime: ResolvedRuntimeConfig,
    safety_gate: SafetyGate,
    current_qpos: np.ndarray,
    *,
    calibration_saved: bool,
) -> int:
    """Invalidate queued motion, publish measured hold, and return exit status."""
    advance_run_generation(shared)
    published = publish_joint_targets(
        shared,
        current_qpos,
        is_hold=True,
        prepare_timeout_s=float(runtime.policy.action_prepare_timeout_s),
        safety_gate=safety_gate,
        wait_applied=True,
        apply_timeout_s=float(runtime.policy.action_apply_timeout_s),
        hand_feedback_max_age_s=float(runtime.safety.heartbeat_timeouts["hand"]),
    )
    if not published.succeeded:
        _set_fault(
            shared,
            f"measured quit hold was not applied: {published.reason}",
        )
        return 1
    if int(shared.safety_state.value) == int(SafetyState.RUNNING):
        require_transition(shared, SafetyState.ARMED)
    return 0 if calibration_saved else 2


def _handle_calibration_home_key(
    shared: SharedStorage,
    runtime: ResolvedRuntimeConfig,
    planner: XArm7MotionPlanner,
    keys: GlobalKeyState,
    rate: RateManager,
    state: _CalibrationLoopState,
) -> _HomeKeyOutcome:
    """Handle one return-home key edge and re-anchor the motion state."""
    home_pressed = keys.is_pressed("r")
    if not home_pressed:
        state.home_key_down = False
        return _HomeKeyOutcome.IDLE
    if state.home_key_down:
        return _HomeKeyOutcome.IDLE
    state.home_key_down = True

    if int(shared.safety_state.value) == int(SafetyState.RUNNING):
        require_transition(shared, SafetyState.ARMED)
    home_result = execute_arm_home(
        shared,
        np.asarray(runtime.arm.home_qpos, dtype=np.float64),
        planner=planner,
        config=ArmHomeConfig.from_runtime(
            runtime,
            publish_policy_heartbeat=False,
        ),
        table_z_surface_m=float(runtime.arm.table_z_surface_m),
        current_qpos=state.current_qpos,
        estop_requested=lambda: keys.is_pressed("esc") or not keys.healthy,
        progress=lambda message: print(f"  {message}", flush=True),
    )
    if shared.estop_request.value:
        _set_fault(shared, "operator e-stop during homing")
        return _HomeKeyOutcome.FAULT
    refreshed = _read_initial_arm(shared, runtime)
    if refreshed is None:
        _set_fault(shared, "fresh arm feedback unavailable after homing")
        return _HomeKeyOutcome.FAULT

    state.current_qpos = np.asarray(refreshed["qpos"], dtype=np.float64)
    state.previous_command = state.current_qpos.copy()
    fresh_pose = planner.kin.compute_eef_pose_world(state.current_qpos)
    state.target_pos = fresh_pose.p.copy()
    state.target_quat = fresh_pose.q.copy()
    if not home_result.succeeded:
        print("  WARNING: return-home request was not executed")
    state.motion_active = False
    rate.reset()
    return _HomeKeyOutcome.COMPLETED


def _log_workspace_clipping(
    desired_pos: np.ndarray,
    clipped_pos: np.ndarray,
    last_warning_s: float,
) -> float:
    clipped = np.abs(desired_pos - clipped_pos) > 1e-9
    if not np.any(clipped):
        return last_warning_s
    parts: list[str] = []
    for axis_index, axis_name in enumerate(("x", "y", "z")):
        if clipped[axis_index]:
            side = "⁺" if desired_pos[axis_index] > clipped_pos[axis_index] else "⁻"
            parts.append(f"{axis_name}{side}{clipped_pos[axis_index]:.3f}")
    now_s = time.monotonic()
    if now_s - last_warning_s >= _BOUNDARY_WARN_INTERVAL_S:
        logger.warning("Workspace boundary: %s", " ".join(parts))
        return now_s
    return last_warning_s


def _run_calibration_motion_tick(
    shared: SharedStorage,
    runtime: ResolvedRuntimeConfig,
    planner: XArm7MotionPlanner,
    safety_gate: SafetyGate,
    workspace: np.ndarray,
    keys: GlobalKeyState,
    state: _CalibrationLoopState,
    calib_cfg: CalibrationConfig,
) -> None:
    """Translate held motion keys into one gated arm command or an idle hold."""
    dx, drpy = eef_delta_from_keys(keys, calib_cfg.delta_pos_m, calib_cfg.delta_rpy_rad)
    moving = bool(np.any(dx != 0.0) or np.any(drpy != 0.0))
    active_keys = keys.pressed_keys()
    if state.blocked_keys is not None and active_keys == state.blocked_keys:
        return
    state.blocked_keys = None

    if moving and not state.motion_active:
        require_transition(shared, SafetyState.RUNNING)
    elif not moving and state.motion_active:
        require_transition(shared, SafetyState.ARMED)
        held_pose = planner.kin.compute_eef_pose_world(state.previous_command)
        state.target_pos = held_pose.p.copy()
        state.target_quat = held_pose.q.copy()
    state.motion_active = moving

    if not moving:
        state.previous_command = state.current_qpos.copy()
        idle_interval = int(runtime.keyboard_teleop.idle_interval_frames)
        if state.frame % idle_interval == 0:
            measured_pose = planner.kin.compute_eef_pose_world(state.current_qpos)
            print(
                f"[f={state.frame}] samples={len(state.samples)} "
                f"eef={np.round(measured_pose.p, 3)}m",
                flush=True,
            )
        return

    workspace_margin_m = float(runtime.keyboard_teleop.workspace_command_margin_m)
    command_low = workspace[:, 0] + workspace_margin_m
    command_high = workspace[:, 1] - workspace_margin_m
    desired_pos = state.target_pos + dx
    state.target_pos = np.clip(desired_pos, command_low, command_high)
    state.last_boundary_warning_s = _log_workspace_clipping(
        desired_pos,
        state.target_pos,
        state.last_boundary_warning_s,
    )
    if np.any(drpy != 0.0):
        delta_quat = Rotation.from_euler("xyz", drpy).as_quat(scalar_first=True)
        state.target_quat = quat_multiply(delta_quat, state.target_quat)

    ik_result = planner.solve_teleop_ik(
        Pose(p=state.target_pos, q=state.target_quat),
        state.current_qpos,
        state.previous_command,
    )
    if not ik_result.success or ik_result.qpos is None:
        now_s = time.monotonic()
        if now_s - state.last_ik_warning_s >= _IK_WARNING_INTERVAL_S:
            logger.warning("IK rejected target: %s", ik_result.reason or "unknown")
            state.last_ik_warning_s = now_s
        measured_pose = planner.kin.compute_eef_pose_world(state.current_qpos)
        state.target_pos = measured_pose.p.copy()
        state.target_quat = measured_pose.q.copy()
        return

    published = publish_joint_targets(
        shared,
        ik_result.qpos,
        prepare_timeout_s=float(runtime.policy.action_prepare_timeout_s),
        safety_gate=safety_gate,
        wait_applied=True,
        apply_timeout_s=float(runtime.policy.action_apply_timeout_s),
        hand_feedback_max_age_s=float(runtime.safety.heartbeat_timeouts["hand"]),
    )
    candidate = published.candidate
    if not published.succeeded or candidate is None or candidate.arm_qpos is None:
        logger.warning(
            "arm motion command rejected (%s) — blocked until keys change",
            published.reason,
        )
        state.blocked_keys = active_keys
        return
    state.previous_command = np.asarray(candidate.arm_qpos, dtype=np.float64).copy()

    if state.frame % calib_cfg.status_interval_frames == 0:
        measured_pose = planner.kin.compute_eef_pose_world(state.current_qpos)
        print(
            f"[f={state.frame}] samples={len(state.samples)} "
            f"eef={np.round(measured_pose.p, 3)}m "
            f"target={np.round(state.target_pos, 3)}m",
            flush=True,
        )


def _run_calibration(
    shared: SharedStorage,
    runtime: ResolvedRuntimeConfig,
    planner: XArm7MotionPlanner,
    safety_gate: SafetyGate,
    workspace: np.ndarray,
    arm_process: Any,
    camera_serial: str | None,
    calib_cfg: CalibrationConfig,
    aruco_cfg: ArucoConfig,
) -> int:
    """Own camera, keyboard, and GUI resources for one calibration session."""
    state = _read_initial_arm(shared, runtime)
    if state is None:
        _set_fault(shared, "initial arm feedback is unavailable or unhealthy")
        return 1

    pipeline: Any | None = None
    keys = GlobalKeyState(
        suppress_echo=True,
        estop_callback=lambda: _set_fault(
            shared, "operator e-stop callback", estop=True
        ),
    )
    keys_started = False
    window_created = False
    try:
        pipeline, serial, intrinsics, distortion = _start_camera(camera_serial)
        print(f"  Camera serial: {serial}")
        print(
            f"  Intrinsics: fx={intrinsics[0, 0]:.1f} "
            f"fy={intrinsics[1, 1]:.1f} ({_CAMERA_WIDTH}x{_CAMERA_HEIGHT})"
        )
        keys.start()
        keys_started = True
        cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        window_created = True
        return _run_calibration_control_loop(
            shared,
            runtime,
            planner,
            safety_gate,
            workspace,
            arm_process,
            pipeline,
            serial,
            intrinsics,
            distortion,
            keys,
            state,
            calib_cfg,
            aruco_cfg,
        )
    except KeyboardInterrupt:
        _set_fault(shared, "KeyboardInterrupt")
        return 130
    except Exception as exc:
        logger.error("calibration session failed", exc_info=True)
        _set_fault(shared, f"calibration session failed: {exc}")
        return 1
    finally:
        if keys_started:
            try:
                keys.stop()
            except Exception:
                logger.error("keyboard listener cleanup failed", exc_info=True)
        if window_created:
            try:
                cv2.destroyWindow(_WINDOW_NAME)
            except Exception:
                logger.error("calibration window cleanup failed", exc_info=True)
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                logger.error("camera cleanup failed", exc_info=True)


def _run_calibration_control_loop(
    shared: SharedStorage,
    runtime: ResolvedRuntimeConfig,
    planner: XArm7MotionPlanner,
    safety_gate: SafetyGate,
    workspace: np.ndarray,
    arm_process: Any,
    pipeline: Any,
    serial: str,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
    keys: GlobalKeyState,
    initial_state: dict[str, Any],
    calib_cfg: CalibrationConfig,
    aruco_cfg: ArucoConfig,
) -> int:
    """Run control logic while borrowing already-started session resources."""
    heartbeat_timeout = float(runtime.safety.heartbeat_timeouts["arm"])
    state = _CalibrationLoopState.from_arm_state(planner, initial_state)
    marker_corners = marker_corners_3d(aruco_cfg.marker_size_m)
    preview_detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(ARUCO_DICT),
        cv2.aruco.DetectorParameters(),
    )
    rate = RateManager(
        float(runtime.keyboard_teleop.control_hz), label="camera_calibration"
    )

    print(
        f"\n  ArUco: {ARUCO_DICT_NAME} ID={aruco_cfg.target_id} "
        f"size={aruco_cfg.marker_size_m * 1000:.1f}mm"
    )
    print(
        "  Controls: WASD/arrows move, ←→/I/J/K/L rotate, SPACE capture, ENTER calibrate"
    )
    print(f"  Preview window: {_WINDOW_NAME} (green=detected, red=not found)")

    display_image: np.ndarray | None = None
    while shared.is_running.value:
        rate.wait()
        state.frame += 1
        display_image = _show_calibration_preview(
            pipeline,
            preview_detector,
            intrinsics,
            distortion,
            state.samples,
            calib_cfg,
            aruco_cfg,
            marker_corners,
            display_image,
        )
        _handle_calibration_sample_events(
            shared,
            runtime,
            planner,
            pipeline,
            serial,
            intrinsics,
            distortion,
            keys,
            state,
            calib_cfg,
            aruco_cfg,
        )

        if keys.is_pressed("esc"):
            _set_fault(shared, "operator e-stop", estop=True)
            return 1
        if not keys.healthy:
            _set_fault(shared, "keyboard listener exited", estop=True)
            return 1
        issue = _runtime_issue(shared, arm_process, heartbeat_timeout)
        if issue is not None:
            _set_fault(shared, issue)
            return 1

        quit_requested = keys.is_pressed("q")
        feedback = _read_calibration_arm_feedback(shared, runtime)
        if feedback.error_code != 0:
            _set_fault(shared, f"arm controller error C{feedback.error_code}")
            return 1
        if feedback.qpos is None:
            if quit_requested:
                _set_fault(
                    shared,
                    f"cannot publish measured quit hold: {feedback.issue}",
                )
                return 1
            continue
        state.current_qpos = feedback.qpos

        if quit_requested:
            return _publish_calibration_quit_hold(
                shared,
                runtime,
                safety_gate,
                state.current_qpos,
                calibration_saved=state.calibration_saved,
            )

        home_outcome = _handle_calibration_home_key(
            shared,
            runtime,
            planner,
            keys,
            rate,
            state,
        )
        if home_outcome is _HomeKeyOutcome.FAULT:
            return 1
        if home_outcome is _HomeKeyOutcome.COMPLETED:
            continue

        _run_calibration_motion_tick(
            shared,
            runtime,
            planner,
            safety_gate,
            workspace,
            keys,
            state,
            calib_cfg,
        )
    return 0


def run_camera_calibration(
    runtime: ResolvedRuntimeConfig,
    *,
    camera_serial: str | None,
    hand_geometry: str,
    calibration_config: CalibrationConfig | None = None,
    aruco_config: ArucoConfig | None = None,
) -> int:
    """Own the arm worker, shared state, camera session, and bounded cleanup."""
    if hand_geometry not in {"absent", "secured-home"}:
        raise ValueError("hand_geometry must be 'absent' or 'secured-home'")
    calib_cfg = calibration_config or CalibrationConfig()
    aruco_cfg = aruco_config or ArucoConfig()
    planner, safety_gate, workspace = _build_planner_and_gate(runtime)
    print(f"  XHand: not required ({hand_geometry} geometry used for collision checks)")

    ctx = mp.get_context("spawn")
    shared = SharedStorage.create(
        prefix=f"dexmani_calib_{os.getpid()}",
        config=SharedStorageConfig.from_runtime(runtime),
        mp_context=ctx,
    )
    specs = [
        WorkerSpec(
            "arm-calib",
            arm_loop,
            (shared, ArmLoopConfig.from_runtime(runtime)),
            ready_name="arm",
        )
    ]
    processes = build_processes(ctx, specs)
    start_processes(processes)
    arm_process = processes[0]
    arm_timeout_s = float(runtime.safety.readiness_timeouts_s["arm"])
    if not wait_subsystem_ready(shared, [("arm", arm_timeout_s)], processes):
        _set_fault(shared, "arm worker did not become ready")
        shutdown_processes(shared, processes)
        return 1

    initial_state = _read_initial_arm(shared, runtime)
    if initial_state is None:
        _set_fault(shared, "initial arm feedback is unavailable or unhealthy")
        shutdown_processes(shared, processes)
        return 1

    require_transition(shared, SafetyState.ARMED)
    print(f"  arm worker ready (Mode 6, {runtime.arm.loop_hz}Hz)")

    exit_code = 1
    try:
        exit_code = _run_calibration(
            shared,
            runtime,
            planner,
            safety_gate,
            workspace,
            arm_process,
            camera_serial,
            calib_cfg,
            aruco_cfg,
        )
    finally:
        started = [p for p in processes if p.pid is not None]
        if started:
            try:
                clean_exit = exit_code == 0
                shutdown_report = shutdown_processes(
                    shared,
                    started,
                    graceful_timeout_s=float(runtime.safety.shutdown_timeout_s),
                    disarm_if_clean=clean_exit,
                )
                if clean_exit and not shutdown_report.clean:
                    logger.error(
                        "verified shutdown invalidated the clean control exit: %s",
                        shutdown_report,
                    )
                    exit_code = 1
            except RuntimeError:
                logger.critical(
                    "child process remains alive; leaving SharedStorage linked",
                    exc_info=True,
                )
                exit_code = 1
        else:
            try:
                if not shared.close():
                    _set_fault(shared, "SharedStorage cleanup was incomplete")
                    exit_code = 1
            except Exception:
                _set_fault(shared, "SharedStorage cleanup failed")
                exit_code = 1

    print(f"  calibration session exit code: {exit_code}")
    return exit_code
