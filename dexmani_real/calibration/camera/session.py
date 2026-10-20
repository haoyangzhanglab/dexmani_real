"""Interactive ArUco eye-to-hand calibration; publish accepted transforms to cameras.json."""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import time
from concurrent.futures import CancelledError
from datetime import datetime, timezone

import cv2
import numpy as np
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

from dexmani_real.calibration import CAMERAS_PATH
from dexmani_real.calibration.camera.solver import (
    ARUCO_DICT,
    ARUCO_DICT_NAME,
    ArucoConfig,
    CalibrationConfig,
    CalibrationSamples,
    HandEyeCalibrator,
    detect_aruco_pose,
    draw_calibration_overlay,
    eef_rpy_from_rot6d,
    marker_corners_3d,
    save_camera_calibration,
)
from dexmani_real.config.experiment import ExperimentConfig, validate_keyboard_workspace
from dexmani_real.ipc.channels import SensorChannels
from dexmani_real.planning import XArm7MotionPlanner
from dexmani_real.planning.admission import TargetAdmission
from dexmani_real.planning.kinematics.arm_fk import make_arm_fk
from dexmani_real.planning.kinematics.ik import IKFailureKind, OnlineIKSolver, make_online_ik_solver
from dexmani_real.robot.commands import DispatchError
from dexmani_real.robot.home import build_home_planner, execute_arm_home
from dexmani_real.robot.robot import DexManiRobot
from dexmani_real.runtime.observation import read_camera_frame, sample_is_fresh
from dexmani_real.runtime.operator_input import KeyboardInput
from dexmani_real.runtime.processes import SensorProcesses, shutdown_local_runtime
from dexmani_real.runtime.state import RunEndReason, SafetyState, RuntimeState
from dexmani_real.sensor.camera.worker import run_camera_worker
from dexmani_real.teleop.jog import (
    any_jog_key_held,
    compute_cartesian_jog_delta,
    jog_motion_requested,
    propose_cartesian_jog_pose,
    submit_jog_target,
)
from dexmani_real.utils.log import configure_logging
from dexmani_real.utils.timing import LoopRate

logger = logging.getLogger(__name__)

_WINDOW_NAME = "ArUco Calibration"
_IDLE_STATUS_INTERVAL_FRAMES = 150


def detect_aruco_stable(
    runtime_state: RuntimeState,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
    *,
    detector: cv2.aruco.ArucoDetector,
    marker_corners: np.ndarray,
    target_id: int | None,
    max_frame_age_s: float,
    n_frames: int = 5,
    cancel_requested=lambda: False,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Capture N frames and return SO(3) mean rotation and median translation for noise reduction."""
    rvecs_all: list[np.ndarray] = []
    tvecs_all: list[np.ndarray] = []
    last_stamp = 0
    for _ in range(n_frames):
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if cancel_requested():
                raise CancelledError("calibration capture cancelled")
            frame = read_camera_frame(runtime_state)
            if (
                frame
                and frame["timestamp_ns"] > last_stamp
                and sample_is_fresh(frame["timestamp_ns"], max_frame_age_s)
            ):
                break
            time.sleep(0.01)
        else:
            raise TimeoutError("fresh calibration image unavailable")
        last_stamp = frame["timestamp_ns"]
        if cancel_requested():
            raise CancelledError("calibration capture cancelled")
        image = cv2.cvtColor(frame["rgb"], cv2.COLOR_RGB2BGR)
        result = detect_aruco_pose(
            image,
            intrinsics,
            distortion,
            detector=detector,
            marker_corners=marker_corners,
            target_id=target_id,
        )
        if cancel_requested():
            raise CancelledError("calibration capture cancelled")
        if result is not None:
            rvecs_all.append(result[0])
            tvecs_all.append(result[1])

    if len(rvecs_all) < max(1, n_frames // 2):
        return None
    return Rotation.from_rotvec(np.asarray(rvecs_all)).mean().as_rotvec(), np.median(
        tvecs_all, axis=0
    )


def build_planner(
    runtime: ExperimentConfig,
) -> tuple[XArm7MotionPlanner, OnlineIKSolver]:
    """Share arm kinematics while keeping HOME and online collision scenes separate."""
    planner = build_home_planner(runtime)
    home_hand_qpos = np.deg2rad(np.asarray(runtime.hand.home_qpos_deg, dtype=np.float64))
    planner.set_hand_qpos(home_hand_qpos)
    solver = make_online_ik_solver(
        runtime,
        kin=planner.kin,
        max_pose_error_pos_m=float(runtime.keyboard_teleop.ik_max_pose_error_pos_m),
        max_pose_error_rot_rad=float(runtime.keyboard_teleop.ik_max_pose_error_rot_rad),
    )
    # Online jog permits table contact; both scenes retain the declared fixed-home hand.
    solver.collision_model.set_hand_qpos(home_hand_qpos)
    return planner, solver


def read_stationary_calibration_arm_state(
    runtime: ExperimentConfig,
    robot,
) -> tuple[np.void | None, str]:
    """Read fresh arm feedback using the homing stationary bound."""
    arm_state = robot.read_state().arm[0]
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


def calibration_capture_metadata(
    *,
    width: int,
    height: int,
    requested_fps: int,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
    method: str,
    sample_count: int,
    position_errors_mm: np.ndarray,
    rotation_errors_deg: np.ndarray,
) -> dict[str, object]:
    """Summarize solver residuals and camera geometry for publication."""

    def summary(values: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(values.mean()),
            "rms": float(np.sqrt(np.mean(values**2))),
            "max": float(values.max()),
        }

    return {
        "width": width,
        "height": height,
        "requested_fps": requested_fps,
        "intrinsics": intrinsics.tolist(),
        "distortion": distortion.tolist(),
        "method": method,
        "sample_count": sample_count,
        "position_error_mm": summary(position_errors_mm),
        "rotation_error_deg": summary(rotation_errors_deg),
        "calibrated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def solve_calibration(
    samples: CalibrationSamples,
    config: CalibrationConfig,
    *,
    width: int,
    height: int,
    requested_fps: int,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
):
    """Compute a qualified T_base_camera candidate; the session owns publication."""
    sample_count = len(samples)
    print(f"\n  computing hand-eye calibration ({sample_count} samples, 5 methods)...")
    try:
        result = HandEyeCalibrator(config).solve(samples)
    except (ValueError, cv2.error) as exc:
        logger.warning("solve failed", exc_info=True)
        print(f"FAILED — {exc}, skipped")
        return None

    errors_mm, errors_deg = result.position_errors_mm, result.rotation_errors_deg
    method = result.method
    T_base_camera = result.T_base_camera
    samples.residuals_mm = errors_mm.copy()

    print("  method consistency (mm, lower is better):")
    for name, score_mm in result.method_scores_mm:
        selected = "  ← selected" if name == method else ""
        score_text = "  FAILED" if np.isnan(score_mm) else f"{score_mm:7.1f}"
        print(f"    {name:11s} {score_text}{selected}")
    print(f"  quality ({method}, T_ee_marker consistency):")
    print(
        f"    position mean={errors_mm.mean():.1f}mm "
        f"max={errors_mm.max():.1f}mm rms={result.position_rms_mm:.1f}mm"
    )
    print(
        f"    rotation mean={errors_deg.mean():.2f}° "
        f"max={errors_deg.max():.2f}° rms={result.rotation_rms_deg:.2f}°"
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
    print(f"  T_base_camera position: {np.round(T_base_camera[:3, 3], 4)}m")

    metadata = calibration_capture_metadata(
        width=width,
        height=height,
        requested_fps=requested_fps,
        intrinsics=intrinsics,
        distortion=distortion,
        method=method,
        sample_count=sample_count,
        position_errors_mm=errors_mm,
        rotation_errors_deg=errors_deg,
    )
    metadata["excitation"] = result.excitation
    return T_base_camera, metadata


def log_workspace_clipping(
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
    if now_s - last_warning_s >= 2.0:
        logger.warning("Workspace boundary: %s", " ".join(parts))
        return now_s
    return last_warning_s


class CameraCalibrationSession:
    """Own calibration samples, motion authority, devices, keyboard and preview lifetime."""

    def __init__(
        self,
        runtime: ExperimentConfig,
        *,
        hand_geometry: str,
        calibration_config: CalibrationConfig | None = None,
        aruco_config: ArucoConfig | None = None,
        output_path=CAMERAS_PATH,
    ):
        self.runtime = runtime
        self.hand_geometry = hand_geometry
        self.calibration_config = calibration_config
        self.aruco_config = aruco_config
        self.output_path = output_path

    def check_services(self):
        return self.sensors.check() and (self.keys is None or self.keys.healthy)

    def fault(self, reason, *, estop=False, run_end_reason=RunEndReason.HARDWARE_FAULT):
        logger.error("Calibration fault: %s", reason)
        if estop:
            self.runtime_state.request_emergency_stop()
        else:
            self.runtime_state.mark_error()
        self.runtime_state.revoke_motion(SafetyState.FAULT, reason=run_end_reason)

    def runtime_issue(self, arm):
        if self.runtime_state.estop_request:
            return "e-stop is requested"
        if self.runtime_state.error_state:
            return "runtime error latch is set"
        if self.runtime_state.safety_state == SafetyState.FAULT:
            return "safety state is FAULT"
        if not sample_is_fresh(arm["timestamp_ns"], self.runtime.arm.feedback_max_age_s):
            return "arm feedback stale"
        return None

    def cancel_requested(self):
        return bool(
            self.runtime_state.quit_requested
            or self.runtime_state.estop_request
            or self.runtime_state.error_state
            or not self.runtime_state.running
            or not self.keys.healthy
            or self.keys.is_pressed("q")
            or self.keys.is_pressed("esc")
            or not self.camera_process.is_alive()
        )

    def capture_sample(self):
        """Append one marker/arm observation only when the arm stayed stationary."""
        if self.cancel_requested():
            return
        print(f"\n  [{len(self.samples) + 1}] capturing ArUco pose...", end=" ", flush=True)
        arm_before, feedback_issue = read_stationary_calibration_arm_state(self.runtime, self.robot)
        if arm_before is None:
            print(f"FAILED — before capture: {feedback_issue}, skipped")
            return
        try:
            aruco_pose = detect_aruco_stable(
                self.runtime_state,
                self.intrinsics,
                self.distortion,
                detector=self.capture_detector,
                marker_corners=self.marker_corners,
                target_id=self.aruco_config.target_id,
                max_frame_age_s=self.runtime.camera.max_frame_age_s,
                n_frames=self.aruco_config.capture_frames,
                cancel_requested=self.cancel_requested,
            )
        except CancelledError:
            print("CANCELLED")
            return
        except (TimeoutError, ValueError, cv2.error) as exc:
            logger.warning("capture failed", exc_info=True)
            print(f"FAILED — {exc}, skipped")
            return

        if self.cancel_requested():
            return
        arm_after, feedback_issue = read_stationary_calibration_arm_state(self.runtime, self.robot)
        if self.cancel_requested():
            return
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
        if self.cancel_requested():
            return
        self.samples.append(
            eef_pos_base_m,
            eef_rpy_base_rad,
            marker_rvec,
            marker_tvec,
        )
        print(
            f"OK (total {len(self.samples)})  EE={np.round(eef_pos_base_m, 3)}m  "
            f"marker_dist={np.linalg.norm(marker_tvec):.3f}m"
        )

    def show_preview(self):
        """Poll and display the newest camera preview without blocking control."""
        frame = read_camera_frame(self.runtime_state)
        if frame and sample_is_fresh(frame["timestamp_ns"], self.runtime.camera.max_frame_age_s):
            image = cv2.cvtColor(frame["rgb"], cv2.COLOR_RGB2BGR)
            self.display_image = draw_calibration_overlay(
                image,
                self.preview_detector,
                self.intrinsics,
                self.distortion,
                n_samples=len(self.samples),
                min_samples=self.calibration_config.min_samples,
                target_id=self.aruco_config.target_id,
                marker_corners=self.marker_corners,
                marker_size_m=self.aruco_config.marker_size_m,
            )
        if self.display_image is not None:
            cv2.imshow(_WINDOW_NAME, self.display_image)
        cv2.waitKey(1)

    def handle_sample_events(self):
        """Drain edge-triggered capture, undo, reject, and solve events."""
        event = self.keys.pop_event()
        while event is not None:
            if self.cancel_requested():
                return
            self.robot.check()
            if event == "space":
                self.capture_sample()
            elif event == "backspace":
                if self.samples.pop_last():
                    print(f"  undone, {len(self.samples)} remaining")
                else:
                    print("  (no samples to undo)")
            elif event == "x":
                removed = self.samples.pop_worst()
                if removed is None:
                    print("  (press ENTER first to evaluate quality, then X to remove worst)")
                else:
                    index, residual_mm = removed
                    print(
                        f"  removed worst frame #{index + 1} "
                        f"(residual {residual_mm:.1f}mm), {len(self.samples)} remaining "
                        "— press ENTER to recompute"
                    )
            elif event == "enter":
                self.runtime_state.revoke_motion()
                self.robot.stop()
                self.command_qpos = self.command_pose = None
                epoch = self.runtime_state.run_id

                def cancelled():
                    return self.cancel_requested() or self.runtime_state.run_id != epoch

                candidate = solve_calibration(
                    self.samples,
                    self.calibration_config,
                    width=self.camera_width,
                    height=self.camera_height,
                    requested_fps=self.runtime.camera.fps,
                    intrinsics=self.intrinsics,
                    distortion=self.distortion,
                )
                if candidate is not None and not cancelled():
                    transform, metadata = candidate
                    try:
                        save_camera_calibration(
                            transform,
                            self.serial,
                            self.output_path,
                            calibration_capture=metadata,
                            cancelled=cancelled,
                        )
                    except CancelledError:
                        print("  calibration publication cancelled")
                    else:
                        self.calibration_saved = True
                        print("  calibration saved")
            event = self.keys.pop_event()

    def run(self) -> int:
        self.prepare()
        exit_code = 1
        try:
            self.robot.connect()
            self.camera_process = self.ctx.Process(
                name="camera",
                target=run_camera_worker,
                args=(self.runtime_state.sensors, self.runtime.camera),
            )
            self.sensors.start(
                self.camera_process,
                ready=self.runtime_state.sensors.camera_ready,
                timeout_s=self.runtime.safety.camera_ready_timeout_s,
            )
            self.runtime_state.arm()
            print("  local arm connected (owner process; Mode 1 prepared on jog)")
            initial_state = self.robot.read_state().arm[0]

            self.keys = KeyboardInput(
                jog_change_callback=self.runtime_state.jog_input_changed,
                quit_callback=self.runtime_state.request_quit,
                capture_commands=False,
                capture_raw_events=True,
                estop_callback=lambda: self.fault("operator e-stop callback", estop=True),
            )
            self.serial = self.runtime_state.sensors.camera_serial.value.decode()
            geometry = json.loads(self.runtime_state.sensors.camera_geometry.value.decode())[
                "color"
            ]
            self.camera_width, self.camera_height = geometry["width"], geometry["height"]
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
                f"fy={self.intrinsics[1, 1]:.1f} ({self.camera_width}x{self.camera_height})"
            )
            self.keys.start()
            cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
            self.window_created = True
            exit_code = self.run_control_loop(initial_state)
        except KeyboardInterrupt:
            self.fault("KeyboardInterrupt", estop=True)
            exit_code = 130
        except Exception as exc:
            logger.exception("calibration session failed")
            self.fault(
                f"calibration session failed: {exc}",
                run_end_reason=RunEndReason.POLICY_FAILURE,
            )
        finally:
            if not self.close():
                exit_code = 1
        print(f"  calibration session exit code: {exit_code}")
        return exit_code

    def close(self) -> bool:
        clean = shutdown_local_runtime(
            self.runtime_state,
            self.robot,
            sensors=self.sensors,
            keyboard=self.keys,
            timeout_s=self.runtime.safety.shutdown_timeout_s,
        )
        if self.window_created:
            try:
                cv2.destroyWindow(_WINDOW_NAME)
                self.window_created = False
            except Exception:
                logger.exception("calibration window cleanup failed")
                clean = False
        return clean

    def run_control_loop(self, initial_state):
        self.current_qpos = initial_state["qpos"]
        self.feedback_timestamp_ns = int(initial_state["timestamp_ns"])
        self.marker_corners = marker_corners_3d(self.aruco_config.marker_size_m)
        self.preview_detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(ARUCO_DICT),
            cv2.aruco.DetectorParameters(),
        )
        capture_params = cv2.aruco.DetectorParameters()
        capture_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.capture_detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(ARUCO_DICT),
            capture_params,
        )
        self.rate = LoopRate(
            float(self.runtime.keyboard_teleop.control_hz),
            label="camera_calibration",
        )

        print(
            f"\n  ArUco: {ARUCO_DICT_NAME} ID={self.aruco_config.target_id} "
            f"size={self.aruco_config.marker_size_m * 1000:.1f}mm"
        )
        print("  Controls: WASD/arrows move, ←→/I/J/K/L rotate, SPACE capture, ENTER calibrate")
        print(f"  Preview window: {_WINDOW_NAME} (green=detected, red=not found)")

        self.display_image: np.ndarray | None = None
        while self.runtime_state.running:
            self.rate.wait()
            self.frame += 1

            if self.keys.is_pressed("esc"):
                self.fault("operator e-stop", estop=True)
                return 1
            if not self.keys.healthy:
                self.fault("keyboard listener exited", estop=True)
                return 1
            if not self.camera_process.is_alive():
                logger.error("camera worker exited: %s", self.camera_process.exitcode)
                self.fault("camera worker exited")
                return 1

            if self.runtime_state.quit_requested or self.keys.is_pressed("q"):
                self.runtime_state.revoke_motion(reason=RunEndReason.QUIT)
                self.robot.stop()
                return 0 if self.calibration_saved else 2
            self.show_preview()
            self.handle_sample_events()
            if self.cancel_requested():
                continue
            # Capture/solve can block; jog must use feedback read afterwards.
            arm_state = self.robot.read_state().arm[0]
            issue = self.runtime_issue(arm_state)
            if issue is not None:
                self.fault(issue)
                return 1
            self.current_qpos = arm_state["qpos"]
            self.feedback_timestamp_ns = int(arm_state["timestamp_ns"])

            if self.handle_home_key():
                if self.runtime_state.safety_state == SafetyState.FAULT:
                    return 1
                continue
            self.motion_tick()
        return 0

    def handle_home_key(self):
        """Handle one return-home key edge and re-anchor the motion state."""
        home_pressed = self.keys.is_pressed("r")
        if not home_pressed:
            self.home_key_down = False
            return False
        if self.home_key_down:
            return False
        self.home_key_down = True
        self.command_qpos = None
        self.command_pose = None

        if self.runtime_state.safety_state == SafetyState.RUNNING:
            if not self.runtime_state.revoke_motion(SafetyState.ARMED):
                self.fault("failed to stop calibration motion before home")
                return True
        self.robot.stop()
        home_result = execute_arm_home(
            self.runtime_state,
            np.asarray(self.runtime.arm.home_qpos, dtype=np.float64),
            robot=self.robot,
            planner=self.home_planner,
            config=self.runtime.arm,
            estop_requested=lambda: self.keys.is_pressed("esc") or not self.keys.healthy,
            cancel_requested=lambda: bool(self.runtime_state.quit_requested),
            progress=lambda message: print(f"  {message}", flush=True),
        )
        if self.runtime_state.estop_request:
            self.fault("operator e-stop during homing")
            return True
        refreshed = self.robot.read_state().arm[0]

        self.current_qpos = np.asarray(refreshed["qpos"], dtype=np.float64)
        self.feedback_timestamp_ns = int(refreshed["timestamp_ns"])
        if not home_result.ok:
            print("  WARNING: return-home request was not executed")
        if not any_jog_key_held(self.keys.pressed_keys()):
            self.blocked_until_release = False
        self.rate.reset()
        return True

    def motion_tick(self):
        """Propose a jog target and commit it only after successful dispatch."""
        with self.runtime_state.motion_lock:
            safety_state, epoch = self.runtime_state.safety_state, self.runtime_state.run_id
        if (
            self.runtime_state.error_state
            or self.runtime_state.estop_request
            or not self.runtime_state.running
            or safety_state not in (SafetyState.ARMED, SafetyState.RUNNING)
        ):
            self.command_qpos = None
            self.command_pose = None
            self.fault("unexpected calibration motion state")
            return
        if self.blocked_until_release:
            if self.robot.stop_required:
                self.runtime_state.revoke_motion()
                self.robot.stop()
                self.command_qpos = self.command_pose = None
            if not any_jog_key_held(self.keys.pressed_keys()):
                self.blocked_until_release = False
            return
        keyboard = self.runtime.keyboard_teleop
        pressed_keys, generation = self.keys.snapshot()
        moving = jog_motion_requested(pressed_keys)
        if not moving:
            if self.robot.stop_required:
                self.runtime_state.revoke_motion()
                self.robot.stop()
                self.command_qpos = self.command_pose = None
            if self.frame % _IDLE_STATUS_INTERVAL_FRAMES == 0:
                measured_pose = self.solver.kin.compute_eef_pose_base(self.current_qpos)
                print(
                    f"[f={self.frame}] samples={len(self.samples)} "
                    f"eef={np.round(measured_pose.p, 3)}m",
                    flush=True,
                )
            return

        if safety_state == SafetyState.ARMED:
            self.command_qpos = self.command_pose = None
            if self.robot.stop_required:
                self.robot.stop()
            try:
                self.robot.prepare_streaming(epoch)
            except DispatchError as exc:
                if not exc.revoked:
                    raise
                self.robot.stop()
                self.command_qpos = self.command_pose = None
                return
            # Mode preparation may block; IK must use feedback read afterwards.
            refreshed = self.robot.read_state().arm[0]
            self.current_qpos = refreshed["qpos"]
            self.feedback_timestamp_ns = int(refreshed["timestamp_ns"])
            if self.keys.snapshot()[1] != generation:
                return
        baseline_qpos = self.command_qpos
        baseline_pose = self.command_pose
        if baseline_qpos is None:
            baseline_qpos = self.current_qpos.copy()
            baseline_pose = self.solver.kin.compute_eef_pose_base(self.current_qpos)
        workspace_margin_m = float(self.runtime.keyboard_teleop.workspace_command_margin_m)
        command_low = self.workspace[:, 0] + workspace_margin_m
        command_high = self.workspace[:, 1] - workspace_margin_m
        dx, drpy = compute_cartesian_jog_delta(
            pressed_keys, keyboard.delta_pos_m, keyboard.delta_rpy_rad
        )
        proposed_pose, desired_pos, changed = propose_cartesian_jog_pose(
            baseline_pose, dx, drpy, command_low, command_high
        )
        self.last_boundary_warning_s = log_workspace_clipping(
            desired_pos,
            proposed_pose.p,
            self.last_boundary_warning_s,
        )
        if not changed:
            self.blocked_until_release = True
            return

        ik_result = self.solver.solve(
            proposed_pose,
            self.current_qpos,
            baseline_qpos,
        )
        if ik_result.failure_kind == IKFailureKind.INVALID_OUTPUT:
            raise RuntimeError(f"online IK technical failure: {ik_result.reason}")
        if not ik_result.success or ik_result.qpos is None:
            now_s = time.monotonic()
            if now_s - self.last_ik_warning_s >= 1.0:
                logger.warning("IK rejected target: %s", ik_result.reason or "unknown")
                self.last_ik_warning_s = now_s
            # Rejection does not renew the previous reference's bounded input lifetime.
            self.blocked_until_release = True
            return

        rejection = self.admission.rejection(
            ik_result.qpos,
            self.current_qpos,
            baseline_qpos,
            collision_hand_qpos=np.deg2rad(self.runtime.hand.home_qpos_deg),
        )
        if rejection is not None:
            self.blocked_until_release = True
            logger.warning("calibration target rejected: %s", rejection)
            return
        q_cmd = ik_result.qpos
        try:
            self.robot.arm.update_health(upstream_until_ns=2**63 - 1)
            committed = submit_jog_target(
                self.keys, generation, self.runtime_state, self.robot, q_cmd, epoch,
                valid_until_ns=self.feedback_timestamp_ns
                + int(self.runtime.arm.feedback_max_age_s * 1e9)
                + 1,
                execute_until_ns=self.feedback_timestamp_ns
                + int(self.runtime.control.vr_mapping.stale_threshold_s * 1e9),
            )
        except DispatchError as exc:
            if not exc.revoked:
                self.runtime_state.revoke_motion(
                    SafetyState.FAULT, reason=RunEndReason.HARDWARE_FAULT
                )
                raise
            logger.info("calibration motion interrupted: dispatch=%s", exc.result)
            self.robot.stop()
            self.command_qpos = self.command_pose = None
            return
        if committed is None:
            return
        self.command_pose = proposed_pose
        self.command_qpos = q_cmd.copy()

        if self.frame % self.calibration_config.status_interval_frames == 0:
            measured_pose = self.solver.kin.compute_eef_pose_base(self.current_qpos)
            print(
                f"[f={self.frame}] samples={len(self.samples)} "
                f"eef={np.round(measured_pose.p, 3)}m "
                f"target={np.round(proposed_pose.p, 3)}m",
                flush=True,
            )

    def prepare(self):
        """Absent and secured-home declarations both retain the conservative fixed-home hand envelope."""
        configure_logging()
        if self.hand_geometry not in {"absent", "secured-home"}:
            raise ValueError("hand_geometry must be 'absent' or 'secured-home'")
        if self.runtime.control.hand_enabled:
            raise ValueError(
                "camera calibration requires explicit hand_enabled=false and secured/absent hand"
            )
        from dexmani_real.config.experiment import resolve_runtime_table, validate_robot_config

        validate_robot_config(self.runtime)
        self.runtime = resolve_runtime_table(self.runtime)
        workspace = validate_keyboard_workspace(self.runtime)
        self.runtime.camera.validate()
        calib_cfg = self.calibration_config or CalibrationConfig()
        aruco_cfg = self.aruco_config or ArucoConfig()
        planner, solver = build_planner(self.runtime)
        if self.hand_geometry == "absent":
            print(
                "  XHand: absent (operator assertion); IK/home geometry conservatively "
                "retains the fixed-home XHand envelope"
            )
        else:
            print(
                "  XHand: mounted and secured at configured home (operator assertion); "
                "IK/home geometry uses the fixed-home XHand envelope"
            )

        self.home_planner, self.solver = planner, solver
        self.admission = TargetAdmission(
            self.runtime, solver.collision_model, dynamic_rejection=solver.dynamic_rejection
        )
        self.workspace = workspace
        self.calibration_config = calib_cfg
        self.aruco_config = aruco_cfg
        self.ctx = mp.get_context("spawn")
        self.runtime_state = RuntimeState(SensorChannels(self.ctx, camera=self.runtime.camera))
        self.sensors = SensorProcesses(self.runtime_state)
        self.keys = None
        self.window_created = False
        self.robot = DexManiRobot(
            self.runtime_state, self.runtime, check_services=self.check_services
        )
        self.samples = CalibrationSamples()
        self.command_qpos = self.command_pose = None
        self.calibration_saved = False
        self.home_key_down = False
        self.frame = 0
        self.last_ik_warning_s = self.last_boundary_warning_s = 0.0
        self.blocked_until_release = False
