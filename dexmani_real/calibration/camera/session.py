"""Interactive xArm7/RealSense lifecycle for ArUco eye-to-hand calibration.

This module owns worker topology, RealSense/GUI interaction, sample capture,
and cleanup. ``camera_calibration_control`` owns the arm-motion state machine,
gated command publication, quit hold, and homing actions.

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
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

from dexmani_real.calibration.camera.control import (
    CalibrationLoopState,
    HomeKeyOutcome,
    handle_calibration_home_key,
    publish_calibration_quit_hold,
    read_calibration_arm_feedback,
    read_initial_arm,
    run_calibration_motion_tick,
    set_calibration_fault,
)
from dexmani_real.calibration.camera.solver import (
    ARUCO_DICT,
    ARUCO_DICT_NAME,
    CAMERA_CALIBRATION_PATH,
    ArucoConfig,
    CalibrationConfig,
    CalibrationSamples,
    calibrate_and_select,
    detect_aruco_pose,
    draw_calibration_overlay,
    eef_rpy_from_rot6d,
    marker_corners_3d,
    save_camera_calibration,
)
from dexmani_real.config.runtime import ResolvedRuntimeConfig
from dexmani_real.control.safety_gate import SafetyGate, planner_action_safety_gate
from dexmani_real.ipc.channels import (
    RuntimeChannels,
    RuntimeChannelsConfig,
    read_arm_state_dict,
)
from dexmani_real.planning import TeleopProfile, XArm7MotionPlanner
from dexmani_real.planning.arm_fk import make_arm_fk
from dexmani_real.robot.arm_worker import arm_loop
from dexmani_real.runtime.safety import SafetyState, require_transition
from dexmani_real.runtime.supervisor import shutdown_processes, wait_subsystem_ready
from dexmani_real.runtime.workers import WorkerSpec, build_processes, start_processes
from dexmani_real.teleop.keyboard import GlobalKeyState
from dexmani_real.utils.feedback import validate_arm_feedback
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate import LoopRate

logger = get_logger(__name__)

_WINDOW_NAME = "ArUco Calibration"
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


def _runtime_issue(
    shared: RuntimeChannels, arm_process: Any, heartbeat_timeout_s: float
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


def _capture_calibration_sample(
    shared: RuntimeChannels,
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
        error_code=arm_state["error_code"],
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
    eef_rpy_base_rad = eef_rpy_from_rot6d(eef_rot6d_base)
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
    shared: RuntimeChannels,
    runtime: ResolvedRuntimeConfig,
    planner: XArm7MotionPlanner,
    pipeline: Any,
    serial: str,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
    keys: GlobalKeyState,
    state: CalibrationLoopState,
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


def _run_calibration(
    shared: RuntimeChannels,
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
    state = read_initial_arm(shared, runtime)
    if state is None:
        set_calibration_fault(shared, "initial arm feedback is unavailable or unhealthy")
        return 1

    pipeline: Any | None = None
    keys = GlobalKeyState(
        suppress_echo=True,
        estop_callback=lambda: set_calibration_fault(
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
        set_calibration_fault(shared, "KeyboardInterrupt")
        return 130
    except Exception as exc:
        logger.error("calibration session failed", exc_info=True)
        set_calibration_fault(shared, f"calibration session failed: {exc}")
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
    shared: RuntimeChannels,
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
    state = CalibrationLoopState.from_arm_state(planner, initial_state)
    marker_corners = marker_corners_3d(aruco_cfg.marker_size_m)
    preview_detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(ARUCO_DICT),
        cv2.aruco.DetectorParameters(),
    )
    rate = LoopRate(
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
            set_calibration_fault(shared, "operator e-stop", estop=True)
            return 1
        if not keys.healthy:
            set_calibration_fault(shared, "keyboard listener exited", estop=True)
            return 1
        issue = _runtime_issue(shared, arm_process, heartbeat_timeout)
        if issue is not None:
            set_calibration_fault(shared, issue)
            return 1

        quit_requested = keys.is_pressed("q")
        feedback = read_calibration_arm_feedback(shared, runtime)
        if feedback.error_code != 0:
            set_calibration_fault(shared, f"arm controller error C{feedback.error_code}")
            return 1
        if feedback.qpos is None:
            if quit_requested:
                set_calibration_fault(
                    shared,
                    f"cannot publish measured quit hold: {feedback.issue}",
                )
                return 1
            continue
        state.current_qpos = feedback.qpos

        if quit_requested:
            return publish_calibration_quit_hold(
                shared,
                runtime,
                safety_gate,
                state.current_qpos,
                calibration_saved=state.calibration_saved,
            )

        home_outcome = handle_calibration_home_key(
            shared,
            runtime,
            planner,
            keys,
            rate,
            state,
        )
        if home_outcome is HomeKeyOutcome.FAULT:
            return 1
        if home_outcome is HomeKeyOutcome.COMPLETED:
            continue

        run_calibration_motion_tick(
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
    shared = RuntimeChannels.create(
        prefix=f"dexmani_calib_{os.getpid()}",
        config=RuntimeChannelsConfig.from_runtime(runtime),
        mp_context=ctx,
    )
    specs = [
        WorkerSpec(
            "arm-calib",
            arm_loop,
            (shared, runtime.arm),
            ready_name="arm",
        )
    ]
    processes = build_processes(ctx, specs)
    start_processes(processes)
    arm_process = processes[0]
    arm_timeout_s = float(runtime.safety.readiness_timeouts_s["arm"])
    if not wait_subsystem_ready(shared, [("arm", arm_timeout_s)], processes):
        set_calibration_fault(shared, "arm worker did not become ready")
        shutdown_processes(shared, processes)
        return 1

    initial_state = read_initial_arm(shared, runtime)
    if initial_state is None:
        set_calibration_fault(shared, "initial arm feedback is unavailable or unhealthy")
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
                    "child process remains alive; leaving RuntimeChannels linked",
                    exc_info=True,
                )
                exit_code = 1
        else:
            try:
                if not shared.close():
                    set_calibration_fault(shared, "RuntimeChannels cleanup was incomplete")
                    exit_code = 1
            except Exception:
                set_calibration_fault(shared, "RuntimeChannels cleanup failed")
                exit_code = 1

    print(f"  calibration session exit code: {exit_code}")
    return exit_code
