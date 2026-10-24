"""Interactive ArUco eye-to-hand calibration; publish accepted transforms to cameras.json."""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import time
from concurrent.futures import CancelledError

import cv2
import numpy as np
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

from dexmani_real.calibration import CAMERAS_PATH
from dexmani_real.calibration.camera.aruco import (
    ARUCO_DICT,
    ARUCO_DICT_NAME,
    ArucoConfig,
    detect_aruco_pose,
    draw_calibration_overlay,
    marker_corners_3d,
)
from dexmani_real.calibration.camera.extrinsics import (
    calibration_capture_metadata,
    save_camera_calibration,
)
from dexmani_real.calibration.camera.hand_eye import (
    CalibrationConfig,
    CalibrationSamples,
    solve_hand_eye,
)
from dexmani_real.config.experiment import (
    ExperimentConfig,
    resolve_table_plane,
    validate_keyboard_workspace,
    validate_robot_config,
)
from dexmani_real.ipc.channels import SensorChannels
from dexmani_real.planning.home_planner import HomePlanner
from dexmani_real.planning.kinematics.ik import OnlineIKConfig, OnlineIKSolver, make_online_ik_solver
from dexmani_real.planning.kinematics.pose import rot6d_to_quat_wxyz
from dexmani_real.planning.target import TargetChecker
from dexmani_real.robot.commands import OperationCancelled
from dexmani_real.robot.robot import DexManiRobot
from dexmani_real.runtime.home import build_home_planner, execute_arm_home
from dexmani_real.runtime.jog import JogController, jog_motion_requested
from dexmani_real.runtime.observation import read_camera_frame, service_feedback
from dexmani_real.runtime.operator_input import KeyboardInput
from dexmani_real.runtime.processes import SensorProcesses, shutdown_local_runtime
from dexmani_real.runtime.state import RunEndReason, SafetyState, RuntimeState
from dexmani_real.sensor.camera.worker import run_camera_worker
from dexmani_real.utils.log import configure_logging
from dexmani_real.utils.timing import LoopRate, sample_is_fresh

logger = logging.getLogger(__name__)

WINDOW_NAME = "ArUco Calibration"
IDLE_STATUS_INTERVAL_FRAMES = 150


def eef_rpy_from_rot6d(rot6d: np.ndarray) -> np.ndarray:
    """Convert an arm EEF rot6d orientation to xyz RPY in radians."""
    quat_wxyz = rot6d_to_quat_wxyz(rot6d)
    rpy_rad = Rotation.from_quat(np.roll(quat_wxyz, -1)).as_euler("xyz", degrees=False)
    return np.asarray(rpy_rad, dtype=np.float64)


def detect_aruco_stable(
    channels: SensorChannels,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
    *,
    detector: cv2.aruco.ArucoDetector,
    marker_corners: np.ndarray,
    target_id: int | None,
    max_frame_age_s: float,
    n_frames: int,
    not_before_ns: int,
    check_capture,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Aggregate post-barrier RGB poses using an SO(3) mean and median translation."""
    rvecs_all: list[np.ndarray] = []
    tvecs_all: list[np.ndarray] = []
    last_stamp = not_before_ns
    last_sequence = 0
    last_color_frame = -1
    for _ in range(n_frames):
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            check_capture()
            frame = read_camera_frame(channels)
            if (
                frame
                and frame["timestamp_ns"] > last_stamp
                and frame["ring_sequence"] > last_sequence
                and frame["color_frame_number"] > last_color_frame
                and sample_is_fresh(frame["timestamp_ns"], max_frame_age_s)
            ):
                break
            time.sleep(0.01)
        else:
            raise TimeoutError("fresh calibration image unavailable")
        last_stamp = frame["timestamp_ns"]
        last_sequence = frame["ring_sequence"]
        last_color_frame = frame["color_frame_number"]
        check_capture()
        image = cv2.cvtColor(frame["rgb"], cv2.COLOR_RGB2BGR)
        result = detect_aruco_pose(
            image,
            intrinsics,
            distortion,
            detector=detector,
            marker_corners=marker_corners,
            target_id=target_id,
        )
        check_capture()
        if result is not None:
            rvecs_all.append(result[0])
            tvecs_all.append(result[1])

    if len(rvecs_all) < max(1, n_frames // 2):
        return None
    return Rotation.from_rotvec(np.asarray(rvecs_all)).mean().as_rotvec(), np.median(
        tvecs_all, axis=0
    )


def build_planner(
    runtime: ExperimentConfig, *, table_plane_abcd,
) -> tuple[HomePlanner, OnlineIKSolver]:
    """Build HOME geometry and online IK with their separate collision scenes."""
    planner = build_home_planner(runtime.environment, table_plane_abcd=table_plane_abcd)
    home_hand_qpos = np.deg2rad(np.asarray(runtime.hand.home_qpos_deg, dtype=np.float64))
    planner.set_hand_qpos(home_hand_qpos)
    solver = make_online_ik_solver(
        OnlineIKConfig(
            max_pose_error_pos_m=runtime.keyboard_teleop.ik_max_pose_error_pos_m,
            max_pose_error_rot_rad=runtime.keyboard_teleop.ik_max_pose_error_rot_rad,
            operational_joint_lower_rad=tuple(runtime.arm.joint_limit_lower),
            operational_joint_upper_rad=tuple(runtime.arm.joint_limit_upper),
        ),
        static_boxes=runtime.environment.static_boxes,
    )
    # Online jog permits table contact; both scenes retain the declared fixed-home hand.
    solver.collision_model.set_hand_qpos(home_hand_qpos)
    return planner, solver


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
    """Compute a qualified T_base_from_camera candidate; the session owns publication."""
    sample_count = len(samples)
    print(f"\n  computing hand-eye calibration ({sample_count} samples, 5 methods)...")
    try:
        result = solve_hand_eye(samples, config)
    except (ValueError, cv2.error) as exc:
        logger.warning("solve failed", exc_info=True)
        print(f"FAILED — {exc}, skipped")
        return None

    errors_mm, errors_deg = result.position_errors_mm, result.rotation_errors_deg
    method = result.method
    T_base_from_camera = result.T_base_from_camera
    samples.residuals_mm = errors_mm.copy()

    print("  method consistency (mm, lower is better):")
    for name, score_mm in result.method_scores_mm:
        selected = "  ← selected" if name == method else ""
        score_text = "  FAILED" if np.isnan(score_mm) else f"{score_mm:7.1f}"
        print(f"    {name:11s} {score_text}{selected}")
    print(f"  quality ({method}, T_eef_from_marker consistency):")
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
    print(f"  T_base_from_camera position: {np.round(T_base_from_camera[:3, 3], 4)}m")

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
    return T_base_from_camera, metadata


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
        return self.keys is not None and self.keys.healthy and self.sensors.check()

    def fault(self, reason, *, estop=False, run_end_reason=RunEndReason.HARDWARE_FAULT):
        if estop:
            self.runtime_state.request_emergency_stop()
        else:
            self.runtime_state.mark_error()
        self.runtime_state.revoke_motion(SafetyState.FAULT, reason=run_end_reason)
        logger.error("Calibration fault: %s", reason)

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
            or self.runtime_state.safety_state == SafetyState.FAULT
            or not self.runtime_state.running
            or not self.keys.healthy
            or self.keys.is_pressed("q")
            or self.keys.is_pressed("esc")
            or not self.camera_process.is_alive()
        )

    def stop_jog(self):
        """Revoke the target and require a physical key release before another jog."""
        revoked = self.runtime_state.revoke_motion()
        try:
            self.robot.stop()
        finally:
            self.jog.clear_reference()
            self.jog.blocked_until_release = True
        return revoked and not self.cancel_requested()

    def read_capture_feedback(self):
        """Read one arm snapshot while servicing capture cancellation and device faults."""
        if self.cancel_requested():
            raise CancelledError("calibration capture cancelled")
        arm = self.robot.read_state().arm[0]
        issue = self.runtime_issue(arm)
        if issue is not None:
            self.fault(issue)
        if self.cancel_requested():
            raise CancelledError("calibration capture cancelled")
        return arm

    def read_capture_arm_state(self, after_ns):
        """Confirm standstill from feedback acquired after the stop or sampling barrier."""
        deadline_ns = time.monotonic_ns() + int(self.runtime.arm.feedback_max_age_s * 1e9)
        while time.monotonic_ns() < deadline_ns:
            arm = self.read_capture_feedback()
            if arm["timestamp_ns"] <= after_ns:
                time.sleep(0.01)
                continue
            speed_rad_s = float(np.max(np.abs(arm["qvel"])))
            bound_rad_s = self.runtime.arm.homing.velocity_convergence_rad_s
            if speed_rad_s > bound_rad_s:
                return None, (
                    "arm velocity exceeds stationary bound "
                    f"({speed_rad_s:.4f}rad/s > {bound_rad_s:.4f}rad/s)"
                )
            return arm, ""
        raise TimeoutError("new arm feedback unavailable after calibration capture barrier")

    def capture_sample(self):
        """Append one marker/arm observation only when the arm stayed stationary."""
        if self.cancel_requested():
            return
        if not self.stop_jog():
            return
        print(f"\n  [{len(self.samples) + 1}] capturing ArUco pose...", end=" ", flush=True)
        try:
            arm_before, feedback_issue = self.read_capture_arm_state(time.monotonic_ns())
            if arm_before is None:
                print(f"FAILED — before capture: {feedback_issue}, skipped")
                return
            aruco_pose = detect_aruco_stable(
                self.sensors.channels,
                self.intrinsics,
                self.distortion,
                detector=self.capture_detector,
                marker_corners=self.marker_corners,
                target_id=self.aruco_config.target_id,
                max_frame_age_s=self.runtime.camera.max_frame_age_s,
                n_frames=self.aruco_config.capture_frames,
                not_before_ns=time.monotonic_ns(),
                check_capture=self.read_capture_feedback,
            )
            arm_after, feedback_issue = self.read_capture_arm_state(time.monotonic_ns())
        except CancelledError:
            print("CANCELLED")
            return
        except (TimeoutError, ValueError, cv2.error) as exc:
            logger.warning("capture failed", exc_info=True)
            print(f"FAILED — {exc}, skipped")
            return

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
        eef_pos_base_m, eef_rot6d_base = self.target_checker.target_fk.compute(arm_after_qpos)
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
        frame = read_camera_frame(self.sensors.channels)
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
            cv2.imshow(WINDOW_NAME, self.display_image)
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
                if not self.stop_jog():
                    return
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
            self.keys = KeyboardInput(
                quit_callback=self.runtime_state.request_quit,
                capture_commands=False,
                capture_raw_events=True,
                estop_callback=self.runtime_state.request_emergency_stop,
            )
            self.keys.start()
            if not self.keys.healthy:
                raise RuntimeError("keyboard stop listener unavailable")
            self.robot.input_signal = self.keys.generation_signal
            self.robot.connect()
            self.camera_process = self.ctx.Process(
                name="camera",
                target=run_camera_worker,
                args=(self.sensors.channels, self.runtime.camera),
            )
            self.sensors.start(
                self.camera_process,
                ready=self.sensors.channels.camera_ready,
                timeout_s=self.runtime.safety.camera_ready_timeout_s,
                may_start=lambda: self.keys.healthy and self.runtime_state.requests_allow_motion(),
            )
            self.robot.check()
            self.runtime_state.arm()
            print("  local arm connected (owner process; Mode 1 prepared on jog)")
            initial_state = self.robot.read_state().arm[0]

            self.serial = self.sensors.channels.camera_serial.value.decode()
            color_intrinsics = json.loads(self.sensors.channels.camera_intrinsics.value.decode())
            self.camera_width, self.camera_height = color_intrinsics["width"], color_intrinsics["height"]
            self.intrinsics = np.array(
                [
                    [color_intrinsics["fx"], 0, color_intrinsics["ppx"]],
                    [0, color_intrinsics["fy"], color_intrinsics["ppy"]],
                    [0, 0, 1],
                ],
                dtype=np.float64,
            )
            self.distortion = np.asarray(color_intrinsics["distortion_coeffs"], dtype=np.float64)
            print(f"  Camera serial: {self.serial}")
            print(
                f"  Intrinsics: fx={self.intrinsics[0, 0]:.1f} "
                f"fy={self.intrinsics[1, 1]:.1f} ({self.camera_width}x{self.camera_height})"
            )
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
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
                cv2.destroyWindow(WINDOW_NAME)
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
            self.runtime.keyboard_teleop.control_hz,
            label="camera_calibration",
        )

        print(
            f"\n  ArUco: {ARUCO_DICT_NAME} ID={self.aruco_config.target_id} "
            f"size={self.aruco_config.marker_size_m * 1000:.1f}mm"
        )
        print("  Controls: WASD/arrows move, ←→/I/J/K/L rotate, SPACE capture, ENTER calibrate")
        print("  Jog key changes stop and re-prepare motion; release all jog keys after a rejected target")
        print(f"  Preview window: {WINDOW_NAME} (green=detected, red=not found)")

        self.display_image: np.ndarray | None = None
        while self.runtime_state.running:
            self.rate.wait()
            self.frame += 1

            if self.runtime_state.estop_request or self.keys.is_pressed("esc"):
                self.fault("operator e-stop", estop=True)
                return 1
            if (
                self.runtime_state.error_state
                or self.runtime_state.safety_state == SafetyState.FAULT
            ):
                self.fault("runtime fault during calibration")
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
        self.jog.clear_reference()

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
            service=lambda: service_feedback(self.robot, self.runtime),
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
        self.jog.release_if_idle(self.keys.pressed_keys())
        self.rate.reset()
        return True

    def motion_tick(self):
        """构造 jog 目标，仅在公共提交正常返回后更新参考。"""
        with self.runtime_state.motion_lock:
            safety_state, epoch = self.runtime_state.safety_state, self.runtime_state.run_id
        if (
            self.runtime_state.error_state
            or self.runtime_state.estop_request
            or not self.runtime_state.running
            or safety_state not in (SafetyState.ARMED, SafetyState.RUNNING)
        ):
            self.jog.clear_reference()
            self.fault("unexpected calibration motion state")
            return
        if self.jog.blocked_until_release:
            if self.robot.stop_required:
                self.runtime_state.revoke_motion()
                self.robot.stop()
                self.jog.clear_reference()
            self.jog.release_if_idle(self.keys.pressed_keys())
            return
        keyboard = self.runtime.keyboard_teleop
        pressed_keys, generation = self.keys.snapshot()
        moving = jog_motion_requested(pressed_keys)
        if not moving:
            if self.robot.stop_required:
                self.runtime_state.revoke_motion()
                self.robot.stop()
                self.jog.clear_reference()
            if self.frame % IDLE_STATUS_INTERVAL_FRAMES == 0:
                measured_pose = self.solver.kin.compute_eef_pose_base(self.current_qpos)
                print(
                    f"[f={self.frame}] samples={len(self.samples)} "
                    f"eef={np.round(measured_pose.p, 3)}m",
                    flush=True,
                )
            return

        if safety_state == SafetyState.ARMED:
            self.jog.clear_reference()
            if self.robot.stop_required:
                self.robot.stop()
            try:
                self.robot.prepare_streaming(epoch, service=lambda: service_feedback(self.robot, self.runtime))
            except OperationCancelled:
                self.robot.stop()
                self.jog.clear_reference()
                return
            # Mode preparation may block; IK must use feedback read afterwards.
            refreshed = self.robot.read_state().arm[0]
            self.current_qpos = refreshed["qpos"]
            self.feedback_timestamp_ns = int(refreshed["timestamp_ns"])
            if self.keys.snapshot()[1] != generation:
                return
        proposed_pose, desired_pos, ik_result = self.jog.propose(
            pressed_keys, self.current_qpos,
            hand_qpos=np.deg2rad(self.runtime.hand.home_qpos_deg),
        )
        self.last_boundary_warning_s = log_workspace_clipping(
            desired_pos,
            proposed_pose.p,
            self.last_boundary_warning_s,
        )
        if ik_result is None:
            return
        if self.jog.blocked_until_release:
            now_s = time.monotonic()
            if now_s - self.last_ik_warning_s >= 1.0:
                logger.warning("IK rejected target: %s", ik_result.reason or "unknown")
                self.last_ik_warning_s = now_s
            return

        try:
            self.robot.arm.update_health(health_until_ns=min(
                time.monotonic_ns() + int(self.runtime.arm.feedback_max_age_s * 1e9),
                self.feedback_timestamp_ns + int(self.runtime.arm.feedback_max_age_s * 1e9) + 1,
            ))
            committed = self.jog.submit(
                proposed_pose, ik_result.qpos, self.keys, generation,
                self.runtime_state, self.robot, epoch,
                valid_until_ns=self.feedback_timestamp_ns
                + int(self.runtime.arm.feedback_max_age_s * 1e9)
                + 1,
                execute_until_ns=self.feedback_timestamp_ns
                + int(keyboard.target_max_age_s * 1e9),
            )
        except (OperationCancelled, TimeoutError) as exc:
            logger.info("jog interrupted: %s", exc)
            self.robot.stop()
            self.jog.clear_reference()
            return
        if not committed:
            return

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
        if self.runtime.hand.enabled:
            raise ValueError(
                "camera calibration requires hand.enabled=false and secured/absent hand"
            )
        validate_robot_config(self.runtime)
        self.table_plane_abcd = resolve_table_plane(
            self.runtime.environment.table, required=self.runtime.environment.table.enabled,
        )
        workspace = validate_keyboard_workspace(self.runtime)
        self.runtime.camera.validate()
        calib_cfg = self.calibration_config or CalibrationConfig()
        aruco_cfg = self.aruco_config or ArucoConfig()
        planner, solver = build_planner(self.runtime, table_plane_abcd=self.table_plane_abcd)
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
        self.target_checker = TargetChecker(
            solver.collision_model,
            joint_lower_rad=self.runtime.arm.joint_limit_lower,
            joint_upper_rad=self.runtime.arm.joint_limit_upper,
            workspace=self.runtime.environment.workspace.as_array(),
            workspace_margin_m=self.runtime.environment.eef_workspace_margin_m,
        )
        self.calibration_config = calib_cfg
        self.aruco_config = aruco_cfg
        self.ctx = mp.get_context("spawn")
        self.runtime_state = RuntimeState()
        self.sensors = SensorProcesses(
            SensorChannels(self.ctx, camera=self.runtime.camera)
        )
        self.keys = None
        self.window_created = False
        self.robot = DexManiRobot(
            self.runtime.arm, self.runtime.hand, run_signal=self.runtime_state.run_signal,
            stop_signal=self.runtime_state.stop_signal,
            emergency_signal=self.runtime_state.emergency_signal,
            signal_lock=self.runtime_state.motion_lock,
            check_services=self.check_services,
        )
        self.samples = CalibrationSamples()
        self.jog = JogController(solver, self.target_checker, self.runtime.keyboard_teleop, workspace)
        self.calibration_saved = False
        self.home_key_down = False
        self.frame = 0
        self.last_ik_warning_s = self.last_boundary_warning_s = 0.0
