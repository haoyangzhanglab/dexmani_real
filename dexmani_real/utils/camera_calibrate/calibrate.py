"""键盘移动、静止采样与 eye-to-hand 标定。"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from concurrent.futures import CancelledError
from datetime import datetime, timezone
from typing import TYPE_CHECKING, cast

import cv2
import numpy as np

from dexmani_real.config.calibration_data import CAMERAS_PATH
from dexmani_real.control.constraints import MotionConstraints
from dexmani_real.control.homing import HomePlanner
from dexmani_real.control.ik import IKSolver
from dexmani_real.geometry.pose import rot6d_to_rotmat
from dexmani_real.ipc.sensors import (
    SensorChannels,
    read_camera_frame,
    snapshot_camera_parameters,
)
from dexmani_real.robot.commands import OperationCancelled, RobotError
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
from dexmani_real.runtime.preflight import (
    jog_bounds,
    validate_robot_config,
)
from dexmani_real.runtime.state import RunEndReason, SafetyState, SessionState
from dexmani_real.sensor.realsense import run_camera
from dexmani_real.utils.camera_calibrate.aruco import (
    ARUCO_DICT_NAME,
    MARKER_ID,
    MARKER_SIZE_M,
    ArucoSampler,
)
from dexmani_real.utils.camera_calibrate.extrinsics import save_camera_calibration
from dexmani_real.utils.camera_calibrate.hand_eye import (
    MIN_SAMPLES,
    HandEyeResult,
    solve_hand_eye,
)
from dexmani_real.utils.log import configure_logging
from dexmani_real.utils.table_plane import load_table_plane
from dexmani_real.utils.timing import Rate, sample_is_fresh
from dexmani_real.utils.visualization.image import ImageViewer, draw_aruco, draw_text

if TYPE_CHECKING:
    from dexmani_real.config.real import RealConfig

logger = logging.getLogger(__name__)

WINDOW_NAME = "ArUco Calibration"


class CameraCalibrator:
    def __init__(
        self,
        runtime: RealConfig,
        *,
        output_path=CAMERAS_PATH,
    ):
        self.runtime = runtime
        self.output_path = output_path
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []
        self.result: HandEyeResult | None = None
        self.calibration_saved = False
        self.preview: ImageViewer | None = None
        self.display_image: np.ndarray | None = None
        self.preview_sequence = 0
        self.preview_timestamp_ns = 0
        self.marker_detected = False
        self.status = "SPACE=capture a stationary pose"

    def check_services(self):
        return self.keys.check_health() and self.sensors.check()

    def fault(self, reason, *, estop=False, run_end_reason=RunEndReason.HARDWARE_FAULT):
        if estop:
            self.runtime_state.request_emergency_stop()
        else:
            self.runtime_state.mark_error()
        self.runtime_state.revoke_motion(SafetyState.FAULT, reason=run_end_reason)
        logger.error("Calibration fault: %s", reason)

    def cancel_requested(self) -> bool:
        # 所有等待阶段（包括 HOME）使用同一取消入口，窗口关闭仅请求退出一次。
        if (
            not self.runtime_state.quit_requested
            and self.preview is not None
            and "closed" in self.preview.poll_events()
        ):
            self.runtime_state.request_quit()
        return bool(
            self.runtime_state.quit_requested
            or self.runtime_state.estop_request
            or self.runtime_state.error_state
            or self.runtime_state.safety_state == SafetyState.FAULT
            or not self.runtime_state.running
            or self.keys.is_pressed("q")
            or self.keys.is_pressed("esc")
        )

    def read_capture_feedback(self):
        if self.cancel_requested():
            raise CancelledError("calibration capture cancelled")
        arm = self.robot.read_state().arm[0]
        if not sample_is_fresh(arm["timestamp_ns"], self.runtime.arm.feedback_max_age_s):
            self.fault("arm feedback stale")
        if self.cancel_requested():
            raise CancelledError("calibration capture cancelled")
        return arm

    def capture_sample(self):
        if self.cancel_requested():
            return
        if not self.jog.pause(self.robot, self.runtime_state) or self.cancel_requested():
            return
        self.status = "CAPTURING: waiting for stationary arm and fresh images"
        self.show_preview()
        if self.cancel_requested():
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
            self.status = "Capture cancelled"
            print("CANCELLED")
            return
        except (TimeoutError, ValueError, cv2.error) as exc:
            self.status = f"Capture failed: {exc}"
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
        self.robot.check()
        self.samples.append((eef_pose, marker_pose))
        self.result = None
        self.status = "Sample captured; ENTER=solve when enough poses are collected"
        print(f"OK (total {len(self.samples)})  EE={np.round(position, 3)}m")

    def show_preview(self):
        if self.cancel_requested():
            return
        frame = read_camera_frame(self.sensors.channels, fields=("rgb",))
        if (
            frame
            and frame["ring_sequence"] > self.preview_sequence
            and sample_is_fresh(frame["timestamp_ns"], self.runtime.camera.max_frame_age_s)
        ):
            image = cv2.cvtColor(frame["rgb"], cv2.COLOR_RGB2BGR)
            detection = self.sampler.detect(frame["rgb"])
            draw_aruco(
                image, detection.corners, detection.ids, detection.pose,
                intrinsics=self.sampler.intrinsics, distortion=self.sampler.distortion,
                axis_length=MARKER_SIZE_M / 2,
            )
            self.marker_detected = detection.pose is not None
            self.display_image = image
            self.preview_sequence = frame["ring_sequence"]
            self.preview_timestamp_ns = frame["timestamp_ns"]
        fresh = sample_is_fresh(self.preview_timestamp_ns, self.runtime.camera.max_frame_age_s)
        image = (
            self.display_image.copy() if self.display_image is not None
            else np.zeros((self.camera.color.height, self.camera.color.width, 3), dtype=np.uint8)
        )
        marker = "STALE" if not fresh else "OK" if self.marker_detected else "NOT FOUND"
        draw_text(
            image, [f"MARKER {marker}"],
            color=(0, 200, 0) if fresh and self.marker_detected else (0, 0, 255),
        )
        draw_text(image, [
            f"samples={len(self.samples)}/{MIN_SAMPLES}  {self.status}",
            "SPACE=capture ENTER=solve/save BACKSPACE=undo X=remove worst",
            "WASD/Up/Down=move Left/Right/IJKL=rotate R=home Q=quit Esc=stop",
        ], origin=(10, 50))
        cast(ImageViewer, self.preview).show(image)
        self.cancel_requested()

    def solve_and_save(self):
        if not self.jog.pause(self.robot, self.runtime_state) or self.cancel_requested():
            return
        epoch = self.runtime_state.run_id

        def cancelled():
            self.robot.check()
            return self.cancel_requested() or self.runtime_state.run_id != epoch

        self.result = None
        self.status = "SOLVING calibration"
        self.show_preview()
        if cancelled():
            return
        try:
            if not self.samples:
                raise ValueError("no calibration samples")
            eef_poses, marker_poses = (np.stack(poses) for poses in zip(*self.samples))
            result = solve_hand_eye(eef_poses, marker_poses)
        except (ValueError, cv2.error) as exc:
            self.status = f"Solve failed: {exc}"
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
            self.status = (
                f"Quality failed: {result.position_rms_mm:.2f}mm / "
                f"{result.rotation_rms_deg:.2f}deg; X=remove worst"
            )
            print(
                "Calibration exceeds RMS limits; press X to remove the worst sample and recapture"
            )
            return
        metadata = {
            "width": self.camera.color.width,
            "height": self.camera.color.height,
            "intrinsics": self.camera.color.matrix().tolist(),
            "distortion": list(self.camera.color.distortion_coeffs),
            "method": result.method,
            "sample_count": len(self.samples),
            "position_rms_mm": result.position_rms_mm,
            "rotation_rms_deg": result.rotation_rms_deg,
            "calibrated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        try:
            save_camera_calibration(
                result.T_base_from_camera,
                self.camera.serial,
                self.output_path,
                calibration_capture=metadata,
                cancelled=cancelled,
            )
        except CancelledError:
            self.status = "Publication cancelled"
            print("Calibration publication cancelled")
            return
        self.calibration_saved = True
        self.status = f"SAVED: {result.method} {result.position_rms_mm:.2f}mm / {result.rotation_rms_deg:.2f}deg"
        print(f"Calibration saved → {self.output_path} (serial: {self.camera.serial})")

    def handle_sample_events(self, events):
        for event in events:
            if self.cancel_requested():
                return
            self.robot.check()
            if event in ("space", "enter"):
                try:
                    if event == "space":
                        self.capture_sample()
                    else:
                        self.solve_and_save()
                finally:
                    self.keys.discard_events("space", "enter", "backspace", "x", "r")
                return  # 同一批次余下的一次性操作已基于旧场景。
            elif event == "backspace" and self.samples:
                self.samples.pop()
                self.result = None
                self.status = "Sample removed; ENTER=recompute"
                print(f"Undone, {len(self.samples)} samples remaining")
            elif event == "x":
                if self.result is None:
                    self.status = "ENTER=solve before removing worst sample"
                    print("Press ENTER to evaluate before removing the worst sample")
                else:
                    worst = self.result.worst_sample
                    self.samples.pop(worst)
                    self.result = None
                    self.status = f"Removed sample #{worst + 1}; ENTER=recompute"
                    print(f"Removed sample #{worst + 1}; press ENTER to recompute")

    def run(self) -> int:
        self.prepare()
        exit_code = 1
        try:
            self.keys.start()
            if not self.keys.check_health():
                raise RuntimeError("keyboard stop listener unavailable")
            self.robot.connect()
            self.sensors.start(
                mp.get_context("spawn").Process(
                    name="camera", target=run_camera,
                    args=(self.sensors.channels, self.runtime.camera),
                ),
                ready=self.sensors.channels.camera_ready,
                timeout_s=CAMERA_READY_TIMEOUT_S,
                may_start=lambda: self.keys.check_health() and self.runtime_state.requests_allow_motion(),
            )
            self.robot.check()
            self.runtime_state.arm()
            print("  local arm connected (owner process)")
            self.camera = snapshot_camera_parameters(self.sensors.channels, self.runtime.camera, None)
            print(f"  Camera serial: {self.camera.serial}")
            print(f"  Resolution: {self.camera.color.width}x{self.camera.color.height}")
            self.preview = ImageViewer(WINDOW_NAME)
            exit_code = self.run_control_loop()
        except KeyboardInterrupt:
            self.fault("KeyboardInterrupt", estop=True)
            exit_code = 130
        except OperationCancelled as exc:
            logger.info("Calibration cancelled: %s", exc)
            if self.runtime_state.quit_requested:
                exit_code = 0 if self.calibration_saved else 2
        except Exception as exc:
            self.fault(
                f"calibration session failed: {exc}",
                run_end_reason=(
                    RunEndReason.HARDWARE_FAULT
                    if isinstance(exc, RobotError)
                    else RunEndReason.POLICY_FAILURE
                ),
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
        if self.preview is not None:
            try:
                self.preview.close()
            except cv2.error:
                logger.exception("calibration window cleanup failed")
                clean = False
        return clean

    def run_control_loop(self):
        self.sampler = ArucoSampler(
            self.camera.color.matrix(), np.asarray(self.camera.color.distortion_coeffs),
        )
        self.jog.prepare(self.robot, self.runtime_state)
        self.rate = Rate(self.runtime.keyboard_teleop.control_hz)

        print(f"\n  ArUco: {ARUCO_DICT_NAME} ID={MARKER_ID} size={MARKER_SIZE_M * 1000:.1f}mm")
        print("  Controls: WASD/arrows move, ←→/I/J/K/L rotate, SPACE capture, ENTER calibrate")
        print(
            "  Release jog keys to brake and hold; release all jog keys after a rejected target"
        )
        print(f"  Preview window: {WINDOW_NAME} (green=detected, red=not found)")

        while self.runtime_state.running:
            self.rate.wait(service=self.robot.service_idle)
            self.robot.service_feedback()

            if self.runtime_state.estop_request or self.keys.is_pressed("esc"):
                self.fault("operator e-stop", estop=True)
                return 1
            if (
                self.runtime_state.error_state
                or self.runtime_state.safety_state == SafetyState.FAULT
            ):
                self.fault("runtime fault during calibration")
                return 1
            if not self.keys.check_health():
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
            if self.cancel_requested():
                continue
            if "r" in events:
                self.run_home()
                if self.runtime_state.safety_state == SafetyState.FAULT:
                    return 1
                continue
            self.handle_sample_events(events)
            if self.cancel_requested():
                continue
            self.jog.step(self.robot, self.keys, self.runtime_state, self.runtime)
        return 0

    def run_home(self):
        logger.info("Returning home...")
        try:
            self.jog.stop(self.robot, self.runtime_state)
            home_result = home_arm(
                self.runtime_state,
                self.robot,
                self.home_planner,
                cancel_requested=self.cancel_requested,
            )
            if self.runtime_state.estop_request:
                self.fault("operator e-stop during homing")
                return
            if not home_result.ok:
                if home_result.interrupted:
                    logger.info("HOME cancelled: %s", home_result.reason)
                else:
                    logger.warning("HOME failed: %s", home_result.reason)
        finally:
            self.keys.discard_events("space", "enter", "backspace", "x", "r")
            self.rate.reset()

    def prepare(self) -> None:
        configure_logging()
        if self.runtime.hand.enabled:
            raise ValueError(
                "camera calibration requires hand.enabled=false and secured/absent hand"
            )
        validate_robot_config(self.runtime)
        table_plane_abcd = (
            load_table_plane(self.runtime.environment.table)
            if self.runtime.environment.table.enabled else None
        )
        bounds = jog_bounds(self.runtime)
        self.runtime.camera.validate()
        print("  XHand must be removed or secured at HOME; collision checks use the fixed-HOME envelope")

        # HOME 路径与在线 IK 使用各自的碰撞场景。
        self.home_planner = HomePlanner(
            self.runtime.environment, self.runtime.arm, table_plane_abcd=table_plane_abcd,
        )
        self.home_planner.set_hand_qpos(np.deg2rad(self.runtime.hand.home_qpos_deg))
        self.constraints = MotionConstraints(self.runtime.arm, self.runtime.environment)
        solver = IKSolver(
            self.constraints,
            position_tolerance=self.runtime.keyboard_teleop.ik_max_pose_error_pos_m,
            rotation_tolerance=self.runtime.keyboard_teleop.ik_max_pose_error_rot_rad,
        )
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
