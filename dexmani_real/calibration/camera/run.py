"""键盘移动、静止采样与 eye-to-hand 标定。"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from concurrent.futures import CancelledError
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import cv2
import numpy as np

from dexmani_real.calibration import CAMERAS_PATH
from dexmani_real.calibration.camera.aruco import (
    ARUCO_DICT_NAME,
    MARKER_ID,
    MARKER_SIZE_M,
    MarkerSampler,
)
from dexmani_real.calibration.camera.extrinsics import save_camera_calibration
from dexmani_real.calibration.camera.hand_eye import HandEyeResult, solve_hand_eye
from dexmani_real.calibration.table_plane import resolve_table_plane
from dexmani_real.control.constraints import MotionConstraints
from dexmani_real.control.homing import HomePlanner
from dexmani_real.control.ik import IKSolver
from dexmani_real.geometry.pose import rot6d_to_rotmat
from dexmani_real.ipc.sensors import (
    SensorChannels,
    read_camera_frame,
    snapshot_camera_parameters,
)
from dexmani_real.robot.system import Robot
from dexmani_real.runtime.home import home_arm
from dexmani_real.runtime.jog import CartesianJog
from dexmani_real.runtime.keyboard import Keyboard
from dexmani_real.runtime.lifecycle import (
    CAMERA_READY_TIMEOUT_S,
    SHUTDOWN_TIMEOUT_S,
    SensorWorkers,
    close_runtime,
)
from dexmani_real.runtime.observation import service_feedback
from dexmani_real.runtime.preflight import (
    jog_bounds,
    validate_robot_config,
)
from dexmani_real.runtime.state import RunEndReason, SafetyState, SessionState
from dexmani_real.sensor.realsense import run_camera
from dexmani_real.utils.log import configure_logging
from dexmani_real.utils.timing import Rate, sample_is_fresh

if TYPE_CHECKING:
    from dexmani_real.config import ExperimentConfig

logger = logging.getLogger(__name__)

WINDOW_NAME = "ArUco Calibration"
IDLE_STATUS_INTERVAL_FRAMES = 150
MOTION_STATUS_INTERVAL_FRAMES = 50


def build_planner(
    runtime: ExperimentConfig,
    *,
    table_plane_abcd,
) -> tuple[HomePlanner, IKSolver]:
    """HOME 路径与在线 IK 使用各自的碰撞场景。"""
    planner = HomePlanner(runtime.environment, runtime.arm, table_plane_abcd=table_plane_abcd)
    home_hand_qpos = np.deg2rad(np.asarray(runtime.hand.home_qpos_deg, dtype=np.float64))
    planner.set_hand_qpos(home_hand_qpos)
    constraints = MotionConstraints(runtime.arm, runtime.environment)
    solver = IKSolver(
        constraints,
        position_tolerance=runtime.keyboard_teleop.ik_max_pose_error_pos_m,
        rotation_tolerance=runtime.keyboard_teleop.ik_max_pose_error_rot_rad,
    )
    return planner, solver


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


class CameraCalibration:
    def __init__(
        self,
        runtime: ExperimentConfig,
        *,
        hand_geometry: str,
        output_path=CAMERAS_PATH,
    ):
        self.runtime = runtime
        self.hand_geometry = hand_geometry
        self.output_path = output_path
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []
        self.result: HandEyeResult | None = None
        self.calibration_saved = False
        self.window_created = False
        self.display_image: np.ndarray | None = None
        self.preview_sequence = 0
        self.frame = 0
        self.last_ik_warning_s = self.last_boundary_warning_s = 0.0

    def check_services(self):
        return self.keys.healthy and self.sensors.check()

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
            or not self.sensors.check()
        )

    def read_capture_feedback(self):
        if self.cancel_requested():
            raise CancelledError("calibration capture cancelled")
        arm = self.robot.read_state().arm[0]
        issue = self.runtime_issue(arm)
        if issue is not None:
            self.fault(issue)
        if self.cancel_requested():
            raise CancelledError("calibration capture cancelled")
        return arm

    def capture_sample(self):
        if self.cancel_requested():
            return
        if not self.jog.pause(self.robot, self.runtime_state) or self.cancel_requested():
            return
        print(f"\n  [{len(self.samples) + 1}] capturing ArUco pose...", end=" ", flush=True)
        try:
            sample = self.sampler.capture(
                self.sensors.channels,
                self.read_capture_feedback,
                camera_config=self.runtime.camera,
                arm_config=self.runtime.arm,
            )
        except CancelledError:
            print("CANCELLED")
            return
        except (TimeoutError, ValueError, cv2.error) as exc:
            print(f"FAILED — {exc}, skipped")
            return
        if self.cancel_requested():
            return
        arm_qpos, marker_pose = sample
        position, orientation = self.constraints.target_fk.forward(arm_qpos)
        eef_pose = np.eye(4)
        eef_pose[:3, :3] = rot6d_to_rotmat(orientation)
        eef_pose[:3, 3] = position
        if self.cancel_requested():
            return
        self.samples.append((eef_pose, marker_pose))
        self.result = None
        print(f"OK (total {len(self.samples)})  EE={np.round(position, 3)}m")

    def show_preview(self):
        frame = read_camera_frame(self.sensors.channels, fields=("rgb",))
        if (
            frame
            and frame["ring_sequence"] > self.preview_sequence
            and sample_is_fresh(frame["timestamp_ns"], self.runtime.camera.max_frame_age_s)
        ):
            image = cv2.cvtColor(frame["rgb"], cv2.COLOR_RGB2BGR)
            self.display_image = self.sampler.preview(image, len(self.samples))
            self.preview_sequence = frame["ring_sequence"]
        if self.display_image is not None:
            cv2.imshow(WINDOW_NAME, self.display_image)
        cv2.waitKey(1)

    def solve_and_save(self):
        if not self.jog.pause(self.robot, self.runtime_state) or self.cancel_requested():
            return
        epoch = self.runtime_state.run_id

        def cancelled():
            return self.cancel_requested() or self.runtime_state.run_id != epoch

        self.result = None
        try:
            if not self.samples:
                raise ValueError("no calibration samples")
            eef_poses, marker_poses = (np.stack(poses) for poses in zip(*self.samples))
            result = solve_hand_eye(eef_poses, marker_poses)
        except (ValueError, cv2.error) as exc:
            print(f"Calibration failed: {exc}")
            return
        if cancelled():
            return
        self.result = result
        print(
            f"{result.method}: position RMS={result.position_rms_mm:.2f}mm, "
            f"rotation RMS={result.rotation_rms_deg:.2f}°, worst sample=#{result.worst_sample + 1}"
        )
        if not result.quality_ok:
            print(
                "Calibration exceeds RMS limits; press X to remove the worst sample and recapture"
            )
            return
        metadata = {
            "width": self.camera_width,
            "height": self.camera_height,
            "intrinsics": self.intrinsics.tolist(),
            "distortion": self.distortion.tolist(),
            "method": result.method,
            "sample_count": len(self.samples),
            "position_rms_mm": result.position_rms_mm,
            "rotation_rms_deg": result.rotation_rms_deg,
            "calibrated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        try:
            save_camera_calibration(
                result.T_base_from_camera,
                self.serial,
                self.output_path,
                calibration_capture=metadata,
                cancelled=cancelled,
            )
        except CancelledError:
            print("Calibration publication cancelled")
            return
        self.calibration_saved = True
        print(f"Calibration saved → {self.output_path} (serial: {self.serial})")

    def handle_sample_events(self, events):
        for event in events:
            if self.cancel_requested():
                return
            self.robot.check()
            if event == "space":
                self.capture_sample()
            elif event == "backspace" and self.samples:
                self.samples.pop()
                self.result = None
                print(f"Undone, {len(self.samples)} samples remaining")
            elif event == "x":
                if self.result is None:
                    print("Press ENTER to evaluate before removing the worst sample")
                else:
                    worst = self.result.worst_sample
                    self.samples.pop(worst)
                    self.result = None
                    print(f"Removed sample #{worst + 1}; press ENTER to recompute")
            elif event == "enter":
                self.solve_and_save()

    def run(self) -> int:
        self.prepare()
        exit_code = 1
        try:
            self.keys.start()
            if not self.keys.healthy:
                raise RuntimeError("keyboard stop listener unavailable")
            self.robot.input_signal = self.keys.generation_signal
            self.robot.connect()
            self.sensors.start(
                mp.get_context("spawn").Process(
                    name="camera", target=run_camera,
                    args=(self.sensors.channels, self.runtime.camera),
                ),
                ready=self.sensors.channels.camera_ready,
                timeout_s=CAMERA_READY_TIMEOUT_S,
                may_start=lambda: self.keys.healthy and self.runtime_state.requests_allow_motion(),
            )
            self.robot.check()
            self.runtime_state.arm()
            print("  local arm connected (owner process; Mode 1 prepared on jog)")
            camera = snapshot_camera_parameters(self.sensors.channels, self.runtime.camera, None)
            self.serial = camera.serial
            self.camera_width, self.camera_height = camera.color.width, camera.color.height
            self.intrinsics = camera.color.matrix()
            self.distortion = np.asarray(camera.color.distortion_coeffs)
            print(f"  Camera serial: {self.serial}")
            print(
                f"  Intrinsics: fx={self.intrinsics[0, 0]:.1f} "
                f"fy={self.intrinsics[1, 1]:.1f} ({self.camera_width}x{self.camera_height})"
            )
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
            self.window_created = True
            exit_code = self.run_control_loop()
        except KeyboardInterrupt:
            self.fault("KeyboardInterrupt", estop=True)
            exit_code = 130
        except Exception as exc:
            self.fault(
                f"calibration session failed: {exc}",
                run_end_reason=RunEndReason.POLICY_FAILURE,
            )
            logger.exception("calibration session failed")
        finally:
            if not self.close():
                exit_code = 1
        print(f"  calibration session exit code: {exit_code}")
        return exit_code

    def close(self) -> bool:
        clean = close_runtime(
            self.runtime_state,
            self.robot,
            sensors=self.sensors,
            keyboard=self.keys,
            deadline=time.monotonic() + SHUTDOWN_TIMEOUT_S,
        )
        if self.window_created:
            try:
                cv2.destroyWindow(WINDOW_NAME)
                self.window_created = False
            except cv2.error:
                logger.exception("calibration window cleanup failed")
                clean = False
        return clean

    def run_control_loop(self):
        self.sampler: MarkerSampler = MarkerSampler(self.intrinsics, self.distortion)
        self.rate = Rate(self.runtime.keyboard_teleop.control_hz)

        print(f"\n  ArUco: {ARUCO_DICT_NAME} ID={MARKER_ID} size={MARKER_SIZE_M * 1000:.1f}mm")
        print("  Controls: WASD/arrows move, ←→/I/J/K/L rotate, SPACE capture, ENTER calibrate")
        print(
            "  Key changes invalidate pending targets; release all jog keys after a rejected target"
        )
        print(f"  Preview window: {WINDOW_NAME} (green=detected, red=not found)")

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
            if not self.sensors.check():
                self.fault("camera worker exited")
                return 1

            if self.runtime_state.quit_requested or self.keys.is_pressed("q"):
                self.runtime_state.revoke_motion(reason=RunEndReason.QUIT)
                self.robot.stop()
                return 0 if self.calibration_saved else 2
            self.show_preview()
            events = self.keys.poll(timeout=0)
            self.handle_sample_events(events)
            if self.cancel_requested():
                continue
            if "r" in events:
                self.run_home()
                if self.runtime_state.safety_state == SafetyState.FAULT:
                    return 1
                continue
            self.motion_tick()
        return 0

    def run_home(self):
        self.jog.stop(self.robot, self.runtime_state)
        home_result = home_arm(
            self.runtime_state,
            self.robot,
            self.home_planner,
            service=lambda: service_feedback(self.robot, self.runtime),
            cancel_requested=self.cancel_requested,
        )
        if self.runtime_state.estop_request:
            self.fault("operator e-stop during homing")
            return
        if not home_result.ok:
            print("  WARNING: return-home request was not executed")
        self.jog.release_if_idle(self.keys.snapshot()[0])
        self.keys.discard_events("r")
        self.rate.reset()

    def motion_tick(self):
        step = self.jog.step(
            self.robot, self.keys, self.runtime_state, self.runtime, deadline_source="arm",
        )
        if step.issues:
            self.fault("; ".join(step.issues))
            return
        if step.pose is not None:
            self.last_boundary_warning_s = log_workspace_clipping(
                step.desired_position, step.pose.p, self.last_boundary_warning_s,
            )
        if step.ik_result is not None and step.ik_result.qpos is None:
            now_s = time.monotonic()
            if now_s - self.last_ik_warning_s >= 1.0:
                logger.warning("IK rejected target: %s", step.ik_result.reason or "unknown")
                self.last_ik_warning_s = now_s
        interval = MOTION_STATUS_INTERVAL_FRAMES if step.committed else IDLE_STATUS_INTERVAL_FRAMES
        if step.observation is not None and self.frame % interval == 0:
            measured_pose = self.solver.compute_eef_pose_base(step.observation.arm["qpos"][0])
            target = f" target={np.round(step.pose.p, 3)}m" if step.committed else ""
            print(
                f"[f={self.frame}] samples={len(self.samples)} "
                f"eef={np.round(measured_pose.p, 3)}m{target}", flush=True,
            )

    def prepare(self):
        """两种手部声明均使用保守的固定 HOME 碰撞包络。"""
        configure_logging()
        if self.hand_geometry not in {"absent", "secured-home"}:
            raise ValueError("hand_geometry must be 'absent' or 'secured-home'")
        if self.runtime.hand.enabled:
            raise ValueError(
                "camera calibration requires hand.enabled=false and secured/absent hand"
            )
        validate_robot_config(self.runtime)
        table_plane_abcd = resolve_table_plane(
            self.runtime.environment.table,
            required=self.runtime.environment.table.enabled,
        )
        bounds = jog_bounds(self.runtime)
        self.runtime.camera.validate()
        planner, solver = build_planner(self.runtime, table_plane_abcd=table_plane_abcd)
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
        self.constraints = solver.constraints
        ctx = mp.get_context("spawn")
        self.runtime_state = SessionState()
        self.sensors = SensorWorkers(SensorChannels(ctx, camera=self.runtime.camera))
        self.keys = Keyboard(
            estop_callback=self.runtime_state.request_emergency_stop,
            bindings={'q': self.runtime_state.request_quit},
        )
        self.robot: Robot = Robot(
            self.runtime.arm,
            self.runtime.hand,
            run_signal=self.runtime_state.run_signal,
            stop_signal=self.runtime_state.stop_signal,
            emergency_signal=self.runtime_state.emergency_signal,
            signal_lock=self.runtime_state.motion_lock,
            check_services=self.check_services,
        )
        self.jog: CartesianJog = CartesianJog(solver, self.runtime.keyboard_teleop, bounds)
