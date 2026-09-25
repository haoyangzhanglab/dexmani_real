"""Interactive ArUco eye-to-hand calibration with xArm7 and a fixed RealSense camera.

Estimates T_world_camera from an end-effector marker using five OpenCV hand-eye
methods and saves accepted results to ``dexmani_real/calibration/state/cameras.json``.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

from dexmani_real.calibration import CAMERAS_PATH
from dexmani_real.calibration.camera.motion import (
    CalibrationLoopState,
    HomeKeyOutcome,
    finish_calibration_motion,
    handle_calibration_home_key,
    read_initial_arm,
    run_calibration_motion_tick,
    set_calibration_fault,
)
from dexmani_real.calibration.camera.solver import (
    ARUCO_DICT,
    ARUCO_DICT_NAME,
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
from dexmani_real.config.experiment import ExperimentConfig
from dexmani_real.ipc.channels import (
    RuntimeChannels,
    RuntimeChannelsConfig,
    read_arm_state_dict,
)
from dexmani_real.planning import XArm7MotionPlanner
from dexmani_real.planning.kinematics.arm_fk import make_arm_fk
from dexmani_real.planning.kinematics.ik import make_online_ik_config
from dexmani_real.robot.arm_worker import run_arm_worker
from dexmani_real.runtime.observation import read_camera_frame, sample_is_fresh
from dexmani_real.runtime.operator_input import KeyboardInput
from dexmani_real.runtime.processes import shutdown_processes_verified
from dexmani_real.runtime.safety import SafetyState, require_transition
from dexmani_real.runtime.supervisor import wait_subsystem_ready
from dexmani_real.sensor.camera.worker import run_camera_worker
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate import LoopRate

logger = get_logger(__name__)

_WINDOW_NAME = "ArUco Calibration"
_CAMERA_WIDTH = 640
_CAMERA_HEIGHT = 480
_CAMERA_FPS = 30


def _detect_aruco_stable(
    shared: RuntimeChannels,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
    *,
    marker_size_m: float,
    target_id: int | None,
    max_frame_age_s: float,
    n_frames: int = 5,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Capture N frames and return median ArUco pose for noise reduction."""
    rvecs_all: list[np.ndarray] = []
    tvecs_all: list[np.ndarray] = []
    last_stamp = 0
    for _ in range(n_frames):
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            frame = read_camera_frame(shared)
            if (
                frame
                and frame["timestamp_ns"] > last_stamp
                and sample_is_fresh(frame["timestamp_ns"], max_frame_age_s)
            ):
                break
            time.sleep(0.01)
        else:
            raise RuntimeError("fresh calibration image unavailable")
        last_stamp = frame["timestamp_ns"]
        image = cv2.cvtColor(frame["rgb"], cv2.COLOR_RGB2BGR)
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


def _build_planner(
    runtime: ExperimentConfig,
) -> tuple[XArm7MotionPlanner, np.ndarray]:
    workspace = runtime.policy.workspace.as_array()
    planner = XArm7MotionPlanner.create_default(
        online_ik_profile=make_online_ik_config(
            runtime,
            max_pose_error_pos_m=float(runtime.keyboard_teleop.ik_max_pose_error_pos_m),
            max_pose_error_rot_rad=float(runtime.keyboard_teleop.ik_max_pose_error_rot_rad),
        ),
        static_boxes=tuple(runtime.environment.static_boxes),
        table=runtime.environment.table,
    )
    planner.workspace_bounds = workspace.copy()
    planner.set_hand_qpos(np.deg2rad(np.asarray(runtime.hand.home_qpos_deg, dtype=np.float64)))
    return planner, workspace


def _read_stationary_calibration_arm_state(
    shared: RuntimeChannels,
    runtime: ExperimentConfig,
) -> tuple[dict[str, Any] | None, str]:
    """Read fresh arm feedback using the homing stationary bound."""
    arm_state = read_arm_state_dict(shared)
    if arm_state is None:
        return None, "arm state unavailable"
    if not sample_is_fresh(arm_state["timestamp_ns"], runtime.arm.feedback_max_age_s):
        return None, "arm feedback stale"
    max_velocity_rad_s = float(np.max(np.abs(np.asarray(arm_state["qvel"]))))
    velocity_convergence_rad_s = float(runtime.arm.homing.velocity_convergence_rad_s)
    if max_velocity_rad_s > velocity_convergence_rad_s:
        return (
            None,
            "arm velocity exceeds stationary bound "
            f"({max_velocity_rad_s:.4f}rad/s > "
            f"{velocity_convergence_rad_s:.4f}rad/s)",
        )
    return arm_state, ""


def _calibration_capture_metadata(
    *,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
    method: str,
    sample_count: int,
    position_errors_mm: np.ndarray,
    rotation_errors_deg: np.ndarray,
) -> dict[str, object]:
    """Build finite JSON camera diagnostics for the captured sample."""
    intrinsic_matrix = np.asarray(intrinsics, dtype=np.float64)
    distortion_values = np.asarray(distortion, dtype=np.float64)
    position_errors = np.asarray(position_errors_mm, dtype=np.float64)
    rotation_errors = np.asarray(rotation_errors_deg, dtype=np.float64)
    if intrinsic_matrix.shape != (3, 3) or not np.all(np.isfinite(intrinsic_matrix)):
        raise ValueError("calibration intrinsics must be finite shape (3, 3)")
    if distortion_values.ndim != 1 or not np.all(np.isfinite(distortion_values)):
        raise ValueError("calibration distortion must be a finite vector")
    if (
        type(sample_count) is not int
        or sample_count <= 0
        or position_errors.shape != (sample_count,)
        or rotation_errors.shape != (sample_count,)
        or not np.all(np.isfinite(position_errors))
        or not np.all(np.isfinite(rotation_errors))
    ):
        raise ValueError("calibration residuals must be finite per-sample vectors")
    if not isinstance(method, str) or not method.strip():
        raise ValueError("calibration method must be non-empty")

    def _summary(values: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "max": float(values.max()),
        }

    return {
        "width": _CAMERA_WIDTH,
        "height": _CAMERA_HEIGHT,
        "fps": _CAMERA_FPS,
        "intrinsics": intrinsic_matrix.tolist(),
        "distortion": distortion_values.tolist(),
        "method": method,
        "sample_count": sample_count,
        "position_error_mm": _summary(position_errors),
        "rotation_error_deg": _summary(rotation_errors),
        "calibrated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _solve_and_save_calibration(
    samples: CalibrationSamples,
    planner: XArm7MotionPlanner,
    camera_serial: str,
    config: CalibrationConfig,
    *,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray | None:
    """Solve calibration and save results that pass the quality checks."""
    sample_count = len(samples)
    if sample_count < config.min_samples:
        print(
            f"  need at least {config.min_samples} samples, have {sample_count} — keep collecting"
        )
        return None
    print(f"\n  computing hand-eye calibration ({sample_count} samples, 5 methods)...")
    try:
        T_base_camera, method, errors_mm, errors_deg, method_table = calibrate_and_select(
            *samples.solver_inputs()
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
            f"rot std={rotation_std_deg:.2f}° > {config.max_consistency_rot_std_deg:.1f}°"
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
            CAMERAS_PATH,
            calibration_capture=_calibration_capture_metadata(
                intrinsics=intrinsics,
                distortion=distortion,
                method=method,
                sample_count=sample_count,
                position_errors_mm=errors_mm,
                rotation_errors_deg=errors_deg,
            ),
        )
    except Exception as exc:
        logger.warning("save failed", exc_info=True)
        print(f"FAILED — {exc}, skipped")
        return None
    print(
        f"  ACCEPTED ({method}, pos std={position_std_mm:.1f}mm, rot std={rotation_std_deg:.2f}°)"
    )
    return T_world_camera


class CameraCalibrationSession:
    """Own interactive calibration context, keyboard and preview lifetime."""

    def __init__(
        self, shared, runtime, planner, workspace, arm_process, calibration_config, aruco_config
    ):
        self.shared = shared
        self.runtime = runtime
        self.planner = planner
        self.workspace = workspace
        self.arm_process = arm_process
        self.calibration_config = calibration_config
        self.aruco_config = aruco_config

    def _runtime_issue(self):
        if self.shared.estop_request.value:
            return "e-stop is requested"
        if self.shared.error_state.value:
            return "a worker set the sticky error latch"
        if int(self.shared.safety_state.value) == int(SafetyState.FAULT):
            return "safety state is FAULT"
        if not self.arm_process.is_alive():
            return "arm worker exited"
        arm = read_arm_state_dict(self.shared)
        if arm is None or not sample_is_fresh(
            arm["timestamp_ns"], self.runtime.arm.feedback_max_age_s
        ):
            return "arm feedback stale"
        if self.shared.workflow_failed.value:
            return "camera acquisition failed"
        return None

    def _capture_sample(self):
        """Append one marker/arm observation only when the arm stayed stationary."""
        print(f"\n  [{len(self.state.samples) + 1}] capturing ArUco pose...", end=" ", flush=True)
        arm_before, feedback_issue = _read_stationary_calibration_arm_state(
            self.shared, self.runtime
        )
        if arm_before is None:
            print(f"FAILED — before capture: {feedback_issue}, skipped")
            return
        try:
            aruco_pose = _detect_aruco_stable(
                self.shared,
                self.intrinsics,
                self.distortion,
                marker_size_m=self.aruco_config.marker_size_m,
                target_id=self.aruco_config.target_id,
                max_frame_age_s=self.runtime.camera.max_frame_age_s,
                n_frames=self.aruco_config.capture_frames,
            )
        except Exception as exc:
            logger.warning("capture failed", exc_info=True)
            print(f"FAILED — {exc}, skipped")
            return

        arm_after, feedback_issue = _read_stationary_calibration_arm_state(
            self.shared, self.runtime
        )
        if arm_after is None:
            print(f"FAILED — after capture: {feedback_issue}, skipped")
            return
        arm_before_qpos = np.asarray(arm_before["qpos"], dtype=np.float64)
        arm_after_qpos = np.asarray(arm_after["qpos"], dtype=np.float64)
        drift_rad = float(np.max(np.abs(arm_after_qpos - arm_before_qpos)))
        convergence_rad = float(self.runtime.arm.homing.convergence_rad)
        if drift_rad > convergence_rad:
            print(
                "FAILED — arm moved during capture "
                f"({drift_rad:.4f}rad > {convergence_rad:.4f}rad), skipped"
            )
            return
        if aruco_pose is None:
            print("FAILED — marker not detected, skipped")
            return

        marker_rvec, marker_tvec = aruco_pose
        eef_pos_base_m, eef_rot6d_base = make_arm_fk().compute(arm_after_qpos)
        eef_rpy_base_rad = eef_rpy_from_rot6d(eef_rot6d_base)
        self.state.samples.append(
            eef_pos_base_m,
            eef_rpy_base_rad,
            marker_rvec,
            marker_tvec,
        )
        print(
            f"OK (total {len(self.state.samples)})  EE={np.round(eef_pos_base_m, 3)}m  "
            f"marker_dist={np.linalg.norm(marker_tvec):.3f}m"
        )

    def _show_preview(self):
        """Poll and display the newest camera preview without blocking control."""
        frame = read_camera_frame(self.shared)
        if frame and sample_is_fresh(frame["timestamp_ns"], self.runtime.camera.max_frame_age_s):
            image = cv2.cvtColor(frame["rgb"], cv2.COLOR_RGB2BGR)
            self.display_image, _ = draw_calibration_overlay(
                image,
                self.preview_detector,
                self.intrinsics,
                self.distortion,
                n_samples=len(self.state.samples),
                min_samples=self.calibration_config.min_samples,
                target_id=self.aruco_config.target_id,
                marker_corners=self.marker_corners,
                marker_size_m=self.aruco_config.marker_size_m,
            )
        if self.display_image is not None:
            cv2.imshow(_WINDOW_NAME, self.display_image)
        cv2.waitKey(1)

    def _handle_sample_events(self):
        """Drain edge-triggered capture, undo, reject, and solve events."""
        event = self.keys.pop_event()
        while event is not None:
            if event == "space":
                self._capture_sample()
            elif event == "backspace":
                if self.state.samples.pop_last():
                    print(f"  undone, {len(self.state.samples)} remaining")
                else:
                    print("  (no samples to undo)")
            elif event == "x":
                removed = self.state.samples.pop_worst()
                if removed is None:
                    print("  (press ENTER first to evaluate quality, then X to remove worst)")
                else:
                    index, residual_mm = removed
                    print(
                        f"  removed worst frame #{index + 1} "
                        f"(residual {residual_mm:.1f}mm), {len(self.state.samples)} remaining "
                        "— press ENTER to recompute"
                    )
            elif event == "enter":
                transform = _solve_and_save_calibration(
                    self.state.samples,
                    self.planner,
                    self.serial,
                    self.calibration_config,
                    intrinsics=self.intrinsics,
                    distortion=self.distortion,
                )
                self.state.calibration_saved = transform is not None
            event = self.keys.pop_event()

    def run(self) -> int:
        initial_state = read_initial_arm(self.shared, self.runtime)
        if initial_state is None:
            set_calibration_fault(self.shared, "initial arm feedback is unavailable or unhealthy")
            return 1

        self.keys = KeyboardInput(
            suppress_echo=True,
            capture_commands=False,
            capture_raw_events=True,
            repeat_estop_callback=True,
            estop_callback=lambda: set_calibration_fault(
                self.shared, "operator e-stop callback", estop=True
            ),
        )
        keys_started = False
        window_created = False
        try:
            self.serial = self.shared.camera_serial.value.decode()
            geometry = json.loads(self.shared.camera_geometry.value.decode())["color"]
            self.intrinsics = np.array(
                [
                    [geometry["fx"], 0, geometry["ppx"]],
                    [0, geometry["fy"], geometry["ppy"]],
                    [0, 0, 1],
                ],
                dtype=np.float64,
            )
            self.distortion = np.asarray(geometry["distortion_coeffs"], dtype=np.float64)
            print(f"  Camera serial: {self.serial}")
            print(
                f"  Intrinsics: fx={self.intrinsics[0, 0]:.1f} "
                f"fy={self.intrinsics[1, 1]:.1f} ({_CAMERA_WIDTH}x{_CAMERA_HEIGHT})"
            )
            self.keys.start()
            keys_started = True
            cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
            window_created = True
            return self._run_control_loop(initial_state)
        except KeyboardInterrupt:
            set_calibration_fault(self.shared, "KeyboardInterrupt")
            return 130
        except Exception as exc:
            logger.error("calibration session failed", exc_info=True)
            set_calibration_fault(self.shared, f"calibration session failed: {exc}")
            return 1
        finally:
            if keys_started:
                try:
                    self.keys.stop()
                except Exception:
                    logger.error("keyboard listener cleanup failed", exc_info=True)
            if window_created:
                try:
                    cv2.destroyWindow(_WINDOW_NAME)
                except Exception:
                    logger.error("calibration window cleanup failed", exc_info=True)

    def _run_control_loop(self, initial_state):
        self.state = CalibrationLoopState.from_arm_state(initial_state)
        self.marker_corners = marker_corners_3d(self.aruco_config.marker_size_m)
        self.preview_detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(ARUCO_DICT),
            cv2.aruco.DetectorParameters(),
        )
        self.rate = LoopRate(
            float(self.runtime.keyboard_teleop.control_hz), label="camera_calibration"
        )

        print(
            f"\n  ArUco: {ARUCO_DICT_NAME} ID={self.aruco_config.target_id} "
            f"size={self.aruco_config.marker_size_m * 1000:.1f}mm"
        )
        print("  Controls: WASD/arrows move, ←→/I/J/K/L rotate, SPACE capture, ENTER calibrate")
        print(f"  Preview window: {_WINDOW_NAME} (green=detected, red=not found)")

        self.display_image: np.ndarray | None = None
        while self.shared.is_running.value:
            self.rate.wait()
            self.state.frame += 1
            self._show_preview()
            self._handle_sample_events()

            if self.keys.is_pressed("esc"):
                set_calibration_fault(self.shared, "operator e-stop", estop=True)
                return 1
            if not self.keys.healthy:
                set_calibration_fault(self.shared, "keyboard listener exited", estop=True)
                return 1
            issue = self._runtime_issue()
            if issue is not None:
                set_calibration_fault(self.shared, issue)
                return 1

            if self.keys.is_pressed("q"):
                return finish_calibration_motion(
                    self.shared, calibration_saved=self.state.calibration_saved
                )
            self.state.current_qpos = read_arm_state_dict(self.shared)["qpos"]

            home_outcome = handle_calibration_home_key(
                self.shared,
                self.runtime,
                self.planner,
                self.keys,
                self.rate,
                self.state,
            )
            if home_outcome is HomeKeyOutcome.FAULT:
                return 1
            if home_outcome is HomeKeyOutcome.COMPLETED:
                continue

            run_calibration_motion_tick(
                self.shared,
                self.runtime,
                self.planner,
                self.workspace,
                self.keys,
                self.state,
                self.calibration_config,
            )
        return 0


def run_camera_calibration(
    runtime: ExperimentConfig,
    *,
    hand_geometry: str,
    calibration_config: CalibrationConfig | None = None,
    aruco_config: ArucoConfig | None = None,
) -> int:
    """Run interactive camera calibration with bounded cleanup.

    ``hand_geometry`` is an operator physical-state assertion: ``absent`` means no
    XHand is mounted; ``secured-home`` means it is physically fixed at home.
    IK endpoint and return-home checks use the fixed-home XHand envelope for both
    values, so ``absent`` retains conservative hand geometry.
    """
    if hand_geometry not in {"absent", "secured-home"}:
        raise ValueError("hand_geometry must be 'absent' or 'secured-home'")
    calib_cfg = calibration_config or CalibrationConfig()
    aruco_cfg = aruco_config or ArucoConfig()
    planner, workspace = _build_planner(runtime)
    if hand_geometry == "absent":
        print(
            "  XHand: absent (operator assertion); IK/home geometry conservatively "
            "retains the fixed-home XHand envelope"
        )
    else:
        print(
            "  XHand: mounted and secured at configured home (operator assertion); "
            "IK/home geometry uses the fixed-home XHand envelope"
        )

    ctx = mp.get_context("spawn")
    shared = RuntimeChannels.create(
        prefix=f"dexmani_calib_{os.getpid()}",
        config=RuntimeChannelsConfig.from_runtime(runtime),
        mp_context=ctx,
    )
    processes: list[Any] = []
    exit_code = 1
    try:
        processes = [
            ctx.Process(name="arm", target=run_arm_worker, args=(shared, runtime.arm)),
            ctx.Process(
                name="camera",
                target=run_camera_worker,
                args=(shared, runtime.camera),
            ),
        ]
        for process in processes:
            process.start()
        arm_process = processes[0]
        for process in processes:
            if not wait_subsystem_ready(
                shared, process, runtime.safety.readiness_timeouts_s[process.name]
            ):
                raise RuntimeError(f"{process.name} startup failed")

        initial_state = read_initial_arm(shared, runtime)
        if initial_state is None:
            set_calibration_fault(shared, "initial arm feedback is unavailable or unhealthy")
            return 1

        require_transition(shared, SafetyState.ARMED)
        print(f"  arm worker ready (Mode 6, {runtime.arm.loop_hz}Hz)")

        exit_code = CameraCalibrationSession(
            shared,
            runtime,
            planner,
            workspace,
            arm_process,
            calib_cfg,
            aruco_cfg,
        ).run()
    finally:
        started = [p for p in processes if p.pid is not None]
        if started:
            try:
                clean_exit = exit_code == 0
                shutdown_report = shutdown_processes_verified(
                    shared,
                    started,
                    graceful_timeout_s=float(runtime.safety.shutdown_timeout_s),
                    disarm_if_clean=clean_exit,
                )
                worker_exit_clean = all(
                    item.exitcode == 0 and item.escalation == "graceful"
                    for item in shutdown_report.exits
                )
                shutdown_clean = (
                    worker_exit_clean
                    and shutdown_report.shared_closed
                    and not bool(shared.error_state.value)
                    and not bool(shared.estop_request.value)
                    and int(shared.safety_state.value) == int(SafetyState.DISARMED)
                )
                if clean_exit and not shutdown_clean:
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
                    logger.error("RuntimeChannels cleanup was incomplete")
                    exit_code = 1
            except Exception:
                logger.error("RuntimeChannels cleanup failed", exc_info=True)
                exit_code = 1

    print(f"  calibration session exit code: {exit_code}")
    return exit_code
