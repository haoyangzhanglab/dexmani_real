"""Resource lifecycle and stationary sampling for camera calibration."""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from dexmani_real.calibration.aruco import (
    DEFAULT_ARUCO_CONFIG,
    ArucoConfig,
    capture_stable_pose,
    create_detector,
    draw_overlay,
)
from dexmani_real.calibration.camera_device import CameraStream, start_camera_stream
from dexmani_real.ipc.schema import ARM_JOINT_SHAPE
from dexmani_real.planning.pose_utils import rot6d_to_quat_wxyz
from dexmani_real.robot.arm_loop import ArmLoopConfig
from dexmani_real.robot.arm_loop import arm_loop as _arm_loop
from dexmani_real.robot.safety import SafetyState, require_transition, transition
from dexmani_real.runtime.supervisor import shutdown_processes, wait_subsystem_ready
from dexmani_real.shm.shared_storage import SharedStorage, SharedStorageConfig, read_arm_state_dict
from dexmani_real.teleop.keyboard import GlobalKeyState, validate_arm_feedback
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate_manager import RateManager

logger = get_logger(__name__)

WINDOW_NAME = "ArUco Calibration"


@dataclass(frozen=True)
class CameraSessionConfig:
    """Physical and timing parameters for one interactive session."""

    delta_pos_m: float = 0.008
    delta_rpy_rad: float = 0.03
    target_lead_max_m: float = 0.03
    min_samples: int = 10
    capture_max_state_age_s: float = 0.25
    capture_max_qvel_rad_s: float = 0.03
    capture_max_qpos_delta_rad: float = 0.003
    duplicate_position_m: float = 0.015
    duplicate_rotation_deg: float = 5.0
    workspace_y_limit_m: float = 0.45
    ik_position_tolerance_m: float = 0.02
    ik_rotation_tolerance_deg: float = 5.0
    status_interval_frames: int = 50
    wall_warning_interval_s: float = 3.0
    initial_state_attempts: int = 30
    initial_state_poll_s: float = 0.05

    def __post_init__(self) -> None:
        positive = (
            self.delta_pos_m,
            self.delta_rpy_rad,
            self.target_lead_max_m,
            self.capture_max_state_age_s,
            self.capture_max_qvel_rad_s,
            self.capture_max_qpos_delta_rad,
            self.duplicate_position_m,
            self.duplicate_rotation_deg,
            self.workspace_y_limit_m,
            self.ik_position_tolerance_m,
            self.ik_rotation_tolerance_deg,
            self.wall_warning_interval_s,
            self.initial_state_poll_s,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("camera session scales and timing values must be finite and positive")
        if self.min_samples < 3 or self.status_interval_frames <= 0 or self.initial_state_attempts <= 0:
            raise ValueError("min_samples must be at least 3 and iteration counts must be positive")


DEFAULT_CAMERA_SESSION_CONFIG = CameraSessionConfig()


@dataclass(frozen=True)
class CapturedSample:
    marker_rvec_camera: np.ndarray
    marker_tvec_camera_m: np.ndarray
    eef_position_base_m: np.ndarray
    eef_rpy_base_rad: np.ndarray


class CalibrationEventHandler(Protocol):
    @property
    def sample_count(self) -> int: ...

    def handle(self, event: str, session: "CameraCalibrationSession") -> None: ...


def validate_capture_state(
    state: Mapping[str, Any] | None,
    *,
    now_monotonic_ns: int,
    config: CameraSessionConfig = DEFAULT_CAMERA_SESSION_CONFIG,
) -> str | None:
    """Return why an arm state is unsuitable for paired capture, if any."""
    if state is None:
        return "arm state unavailable"
    expected_shapes = {"qpos": ARM_JOINT_SHAPE, "qvel": ARM_JOINT_SHAPE, "eef_pos": (3,), "eef_rot6d": (6,)}
    try:
        arrays = {name: np.asarray(state[name], dtype=np.float64) for name in expected_shapes}
        connected = bool(state["connected"])
        state_valid = bool(state["state_valid"])
        error_code = int(state["error_code"])
        source_ns = int(state["source_monotonic_ns"])
    except (KeyError, TypeError, ValueError):
        return "arm feedback is malformed"
    if any(arrays[name].shape != shape for name, shape in expected_shapes.items()):
        return "arm feedback has invalid shapes"
    if not connected or not state_valid or not all(np.all(np.isfinite(value)) for value in arrays.values()):
        return "arm feedback disconnected or invalid"
    if error_code != 0:
        return f"arm controller error C{error_code}"
    age_ns = int(now_monotonic_ns) - source_ns
    if source_ns <= 0 or age_ns < 0:
        return "arm feedback timestamp is missing or from the future"
    age_s = age_ns * 1e-9
    if age_s > config.capture_max_state_age_s:
        return f"arm feedback stale ({age_s:.2f}s)"
    if float(np.max(np.abs(arrays["qvel"]))) > config.capture_max_qvel_rad_s:
        return "arm is still moving"
    return None


class CameraCalibrationSession:
    """Own arm, camera, keyboard, GUI, and their verified teardown."""

    def __init__(
        self,
        runtime: Any,
        planner: Any,
        action_safety_gate: Any,
        workspace_bounds_m: np.ndarray,
        selected_serial: str,
        *,
        hand_geometry: str,
        config: CameraSessionConfig = DEFAULT_CAMERA_SESSION_CONFIG,
        aruco_config: ArucoConfig = DEFAULT_ARUCO_CONFIG,
    ) -> None:
        bounds = np.asarray(workspace_bounds_m, dtype=np.float64)
        if bounds.shape != (3, 2) or not np.all(np.isfinite(bounds)) or np.any(bounds[:, 0] > bounds[:, 1]):
            raise ValueError("workspace_bounds_m must be finite (3, 2) lower/upper bounds")
        if hand_geometry not in {"absent", "secured-home"}:
            raise ValueError("hand_geometry must be 'absent' or 'secured-home'")
        self.runtime = runtime
        self.planner = planner
        self.action_safety_gate = action_safety_gate
        self.workspace_bounds_m = bounds.copy()
        self.selected_serial = selected_serial
        self.hand_geometry = hand_geometry
        self.config = config
        self.aruco_config = aruco_config
        self.control_hz = float(runtime.arm.loop_hz)
        self.shared: SharedStorage | None = None
        self.arm_config: ArmLoopConfig | None = None
        self.arm_process: Any | None = None
        self.camera: CameraStream | None = None
        self.keys: GlobalKeyState | None = None
        self.initial_arm_state: dict[str, Any] | None = None
        self.preview_detector: Any | None = None
        self.display_image: np.ndarray | None = None
        self.running = False
        self.exit_code = 1
        self.exit_reason = "session did not start"
        self._previous_delete_pressed = False
        self._window_open = False
        self._armed_once = False
        self._closed = False

    @property
    def serial(self) -> str:
        if self.camera is None:
            raise RuntimeError("camera stream is not ready")
        return self.camera.serial

    @property
    def capture_metadata(self) -> dict[str, Any]:
        if self.camera is None:
            raise RuntimeError("camera stream is not ready")
        metadata = dict(self.camera.capture_metadata)
        metadata["hand_geometry"] = self.hand_geometry
        return metadata

    def _wait_for_initial_arm_state(self) -> dict[str, Any]:
        assert self.shared is not None
        max_age_s = float(self.runtime.policy.arm_state_stale_threshold_s)
        for _ in range(self.config.initial_state_attempts):
            state = read_arm_state_dict(self.shared)
            issue: str | None = "arm state unavailable"
            if state is not None:
                issue = validate_arm_feedback(
                    connected=state["connected"],
                    state_valid=state["state_valid"],
                    source_monotonic_ns=state["source_monotonic_ns"],
                    now_monotonic_ns=time.monotonic_ns(),
                    max_age_s=max_age_s,
                    qpos=state["qpos"],
                    qvel=state["qvel"],
                    eef_pos=state["eef_pos"],
                    eef_rot6d=state["eef_rot6d"],
                )
            if state is not None and issue is None and int(state["error_code"]) == 0:
                return state
            time.sleep(self.config.initial_state_poll_s)
        raise RuntimeError("无法从 arm_state_ring 读取初始状态")

    def start(self) -> None:
        if self.shared is not None or self._closed:
            raise RuntimeError("camera calibration session cannot be restarted")
        ctx = mp.get_context("spawn")
        try:
            self.shared = SharedStorage.create(
                prefix=f"dexmani_calib_{os.getpid()}",
                config=SharedStorageConfig.from_runtime(self.runtime),
                mp_context=ctx,
            )
            if int(self.shared.safety_state.value) != int(SafetyState.DISARMED):
                raise RuntimeError("camera calibration must start DISARMED")
            self.arm_config = ArmLoopConfig.from_runtime(self.runtime)
            self.arm_process = ctx.Process(
                target=_arm_loop,
                args=(self.shared, self.arm_config),
                name="arm-calib",
                daemon=False,
            )
            self.arm_process.start()
            ready_timeout_s = float(self.runtime.safety.readiness_timeouts_s["arm"])
            if not wait_subsystem_ready(
                self.shared,
                [("arm", self.shared.arm_ready, ready_timeout_s)],
                [self.arm_process],
            ):
                raise RuntimeError("arm worker did not become ready")
            print(f"  ✓ arm_loop 就绪 (SharedStorage, DISARMED, {self.control_hz:g}Hz)")
            self.initial_arm_state = self._wait_for_initial_arm_state()

            print("\n启动 RealSense 相机...")
            self.camera = start_camera_stream(self.selected_serial, self.runtime.camera, self.runtime.sha256)
            self.keys = GlobalKeyState(suppress_echo=True, estop_callback=self.latch_estop)
            self.keys.start()
            if not self.keys.healthy:
                raise RuntimeError("keyboard listener exited during startup")
            print("\n  键盘控制就绪 ←")
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
            self._window_open = True
            self.preview_detector = create_detector(self.aruco_config, refine_corners=False)
            health_issue = self.arm_health_issue()
            if health_issue is not None or not self.keys.healthy:
                raise RuntimeError(
                    f"session became unhealthy before arming: {health_issue or 'keyboard listener exited'}"
                )
            if int(self.shared.safety_state.value) != int(SafetyState.DISARMED):
                raise RuntimeError("safety state changed before calibration became ready")
            require_transition(self.shared, SafetyState.ARMED)
            self._armed_once = True
            self.running = True
            self.exit_reason = "session running"
            print("  预览窗口已打开（绿=已检测，红=未检测）")
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.exit_code == 0 and self._armed_once:
            health_issue = self.arm_health_issue()
            if health_issue is not None:
                self.latch_fault(f"fault observed during shutdown: {health_issue}")
        if self.shared is not None:
            healthy = not self.shared.error_state.value and not self.shared.estop_request.value
            armed = int(self.shared.safety_state.value) in (int(SafetyState.ARMED), int(SafetyState.RUNNING))
            if healthy and armed:
                try:
                    require_transition(self.shared, SafetyState.DISARMED)
                except RuntimeError:
                    self.shared.error_state.value = True
                    transition(self.shared, SafetyState.FAULT)
                    logger.error("Failed to disarm camera calibration before cleanup", exc_info=True)
                post_disarm_issue = self.arm_health_issue()
                if post_disarm_issue is not None:
                    self.latch_fault(f"fault observed while disarming: {post_disarm_issue}")

        cleanup: list[tuple[str, Any]] = []
        if self.keys is not None:
            cleanup.append(("keyboard", self.keys.stop))
        if self._window_open:
            cleanup.append(("OpenCV windows", cv2.destroyAllWindows))
        if self.camera is not None:
            cleanup.append(("RealSense pipeline", self.camera.pipeline.stop))
        for name, operation in cleanup:
            try:
                operation()
            except Exception:
                logger.warning("calibration cleanup failed: %s", name, exc_info=True)

        if self.shared is None:
            return
        if self.arm_process is not None and self.arm_process.pid is not None:
            report = shutdown_processes(
                self.shared,
                [self.arm_process],
                graceful_timeout_s=float(self.runtime.safety.shutdown_timeout_s),
            )
            if not report.shared_closed:
                self.latch_fault("SharedStorage cleanup was incomplete")
            elif self.exit_code == 0 and not report.clean:
                self.latch_fault("arm worker did not shut down cleanly")
        else:
            if not self.shared.close():
                self.latch_fault("SharedStorage cleanup was incomplete")

    def latch_estop(self, reason: str = "e-stop requested") -> None:
        if self.shared is None:
            return
        self.shared.estop_request.value = True
        transition(self.shared, SafetyState.FAULT)
        self.shared.is_running.value = False
        self.running = False
        self.exit_code = 1
        self.exit_reason = reason

    def latch_fault(self, message: str) -> None:
        if self.shared is None:
            return
        logger.error("Camera calibration fault: %s", message)
        self.shared.error_state.value = True
        transition(self.shared, SafetyState.FAULT)
        self.running = False
        self.exit_code = 1
        self.exit_reason = message

    def arm_health_issue(self) -> str | None:
        if self.shared is None or self.arm_process is None:
            return "arm session is not started"
        if self.shared.estop_request.value:
            return "e-stop requested"
        if self.shared.error_state.value or int(self.shared.safety_state.value) == int(SafetyState.FAULT):
            return "shared arm fault"
        if self.arm_process.exitcode is not None or not self.arm_process.is_alive():
            return f"arm worker exited (exitcode={self.arm_process.exitcode})"
        heartbeat_s = float(self.shared.arm_heartbeat_s.value)
        now_s = time.monotonic()
        if not np.isfinite(heartbeat_s) or heartbeat_s <= 0.0:
            return "arm heartbeat is missing or non-finite"
        heartbeat_age_s = now_s - heartbeat_s
        timeout_s = float(self.runtime.safety.heartbeat_timeouts["arm"])
        if heartbeat_age_s < 0.0 or heartbeat_age_s > timeout_s:
            return f"arm heartbeat age {heartbeat_age_s:.2f}s exceeds {timeout_s:.2f}s"
        return None

    def can_publish_calibration(self) -> bool:
        active = self.shared is not None and int(self.shared.safety_state.value) in (
            int(SafetyState.ARMED),
            int(SafetyState.RUNNING),
        )
        return active and self.arm_health_issue() is None and self.keys is not None and self.keys.healthy

    def read_capture_state(self) -> tuple[dict[str, Any] | None, str | None]:
        assert self.shared is not None
        state = read_arm_state_dict(self.shared)
        issue = validate_capture_state(state, now_monotonic_ns=time.monotonic_ns(), config=self.config)
        return (state if issue is None else None), issue

    def _capture_abort_requested(self) -> bool:
        return (
            self.keys is None
            or not self.keys.healthy
            or self.keys.is_pressed("esc")
            or self.arm_health_issue() is not None
        )

    def capture_sample(self, previous: CapturedSample | None) -> tuple[CapturedSample | None, str | None]:
        if self.camera is None or self.keys is None:
            raise RuntimeError("camera calibration session is not ready")
        before, issue = self.read_capture_state()
        if issue is not None or before is None:
            return None, issue
        marker_pose = capture_stable_pose(
            self.camera.pipeline,
            self.camera.intrinsics,
            self.camera.distortion,
            self.aruco_config,
            abort_requested=self._capture_abort_requested,
        )
        if marker_pose is None:
            return None, "marker burst missing, dispersed, or interrupted"
        after, issue = self.read_capture_state()
        if issue is not None or after is None:
            return None, issue
        qpos_delta = float(
            np.max(
                np.abs(
                    self.planner.ik_mgr.compute_qpos_delta(
                        np.asarray(after["qpos"]),
                        np.asarray(before["qpos"]),
                    )
                )
            )
        )
        if not np.isfinite(qpos_delta):
            return None, "arm joint displacement is non-finite"
        if qpos_delta > self.config.capture_max_qpos_delta_rad:
            return None, f"arm moved during capture ({np.rad2deg(qpos_delta):.2f}deg)"

        position = 0.5 * (np.asarray(before["eef_pos"]) + np.asarray(after["eef_pos"]))
        quaternions_xyzw = np.stack(
            [
                np.roll(rot6d_to_quat_wxyz(np.asarray(before["eef_rot6d"])), -1),
                np.roll(rot6d_to_quat_wxyz(np.asarray(after["eef_rot6d"])), -1),
            ]
        )
        orientation = Rotation.from_quat(quaternions_xyzw).mean()
        if previous is not None:
            position_delta = float(np.linalg.norm(position - previous.eef_position_base_m))
            previous_rotation = Rotation.from_euler("xyz", previous.eef_rpy_base_rad)
            rotation_delta_deg = float(np.rad2deg((previous_rotation.inv() * orientation).magnitude()))
            if (
                position_delta < self.config.duplicate_position_m
                and rotation_delta_deg < self.config.duplicate_rotation_deg
            ):
                return None, "pose duplicates the previous sample"
        marker_rvec, marker_tvec = marker_pose
        return (
            CapturedSample(
                marker_rvec_camera=np.asarray(marker_rvec, dtype=np.float64),
                marker_tvec_camera_m=np.asarray(marker_tvec, dtype=np.float64),
                eef_position_base_m=np.asarray(position, dtype=np.float64),
                eef_rpy_base_rad=np.asarray(orientation.as_euler("xyz", degrees=False), dtype=np.float64),
            ),
            None,
        )

    def preview(self, sample_count: int) -> None:
        if self.camera is None or self.preview_detector is None:
            raise RuntimeError("camera preview is not ready")
        frames = self.camera.pipeline.poll_for_frames()
        color_frame = frames.get_color_frame() if frames else None
        if color_frame:
            image = np.asanyarray(color_frame.get_data()).copy()
            self.display_image, _detected = draw_overlay(
                image,
                self.preview_detector,
                self.camera.intrinsics,
                self.camera.distortion,
                sample_count=sample_count,
                minimum_samples=self.config.min_samples,
                config=self.aruco_config,
            )
        if self.display_image is not None:
            cv2.imshow(WINDOW_NAME, self.display_image)
        cv2.waitKey(1)

    def handle_events(self, handler: CalibrationEventHandler) -> None:
        assert self.keys is not None
        event = self.keys.pop_event()
        delete_pressed = self.keys.is_pressed("x")
        if event is None and delete_pressed and not self._previous_delete_pressed:
            event = "x"
        self._previous_delete_pressed = delete_pressed
        while event is not None:
            if self.keys.is_pressed("esc"):
                self.latch_estop()
                return
            handler.handle(event, self)
            event = self.keys.pop_event()

    def _control_loop(self, handler: CalibrationEventHandler) -> None:
        from dexmani_real.calibration.camera_motion import CameraMotionController

        assert self.shared is not None and self.keys is not None and self.initial_arm_state is not None
        motion = CameraMotionController(self, self.initial_arm_state)
        limiter = RateManager(self.control_hz)
        while self.running:
            limiter.wait()
            if self.keys.is_pressed("esc"):
                print("\nESC: emergency_stop")
                self.latch_estop()
                break
            if not self.keys.healthy:
                logger.error("Keyboard listener exited during camera calibration")
                self.latch_estop("keyboard listener exited")
                break
            health_issue = self.arm_health_issue()
            if health_issue is not None:
                self.latch_fault(health_issue)
                break
            if self.keys.is_pressed("q"):
                print("\nQ: 退出")
                self.running = False
                self.exit_code = 0
                self.exit_reason = "operator quit"
                break
            self.preview(handler.sample_count)
            self.handle_events(handler)
            if not self.running:
                break
            if not self.keys.healthy:
                self.latch_estop("keyboard listener exited")
                break
            health_issue = self.arm_health_issue()
            if health_issue is not None:
                self.latch_fault(health_issue)
                break
            if motion.handle_home():
                continue
            motion.step(handler.sample_count)

    def run(self, handler: CalibrationEventHandler) -> int:
        try:
            self.start()
            self._control_loop(handler)
            if self.exit_reason == "session running":
                self.exit_code = 1
                self.exit_reason = "control loop stopped unexpectedly"
        except KeyboardInterrupt:
            print("\n\nKeyboardInterrupt — 退出")
            faulted = self.shared is not None and (
                self.shared.error_state.value
                or self.shared.estop_request.value
                or int(self.shared.safety_state.value) == int(SafetyState.FAULT)
            )
            if not faulted:
                self.exit_code = 0
                self.exit_reason = "KeyboardInterrupt"
        finally:
            self.close()
        return self.exit_code


__all__ = [
    "CameraCalibrationSession",
    "CameraSessionConfig",
    "CapturedSample",
    "DEFAULT_CAMERA_SESSION_CONFIG",
    "validate_capture_state",
]
