"""在同一窗口点动机械臂、预览 ArUco 并采样标定相机。"""

from __future__ import annotations

import time
from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from dexmani_real.control.collision import CollisionModel
from dexmani_real.robot.commands import (
    FeedbackUnavailable,
    OperationCancelled,
    validate_hand_target,
)
from dexmani_real.robot.model import XHAND_RATED_QPOS_MAX_RAD, XHAND_RATED_QPOS_MIN_RAD
from dexmani_real.runtime.keyboard import JOG_MOTION_KEYS
from dexmani_real.sensor.realsense import RealSenseCamera
from dexmani_real.teleop.keyboard import KeyboardTeleop
from dexmani_real.utils.camera_calibrate.aruco import (
    ARUCO_DICT_NAME,
    MARKER_ID,
    MARKER_SIZE_M,
    ArucoSampler,
    SampleRejected,
)
from dexmani_real.utils.camera_calibrate.extrinsics import save_camera_calibration
from dexmani_real.utils.camera_calibrate.hand_eye import MIN_SAMPLES, solve_hand_eye
from dexmani_real.utils.timing import sample_is_fresh
from dexmani_real.utils.visualization.image import ImageViewer

if TYPE_CHECKING:
    from dexmani_real.config.real import RealConfig

WINDOW_POLL_NS = 16_000_000


def _marker_preview(rgb: np.ndarray, sampler: ArucoSampler) -> tuple[np.ndarray, str]:
    preview = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    corners, pose = sampler.detect(rgb)
    if corners is None:
        return preview, "NOT FOUND"
    color = (0, 200, 0) if pose is not None else (0, 160, 255)
    cv2.polylines(preview, [np.rint(corners).astype(np.int32)], True, color, 2, cv2.LINE_AA)
    if pose is None:
        return preview, "POSE REJECTED"
    rvec, tvec = pose
    cv2.drawFrameAxes(preview, sampler.intrinsics, sampler.distortion, rvec, tvec, MARKER_SIZE_M / 2)
    return preview, "OK"


class CameraCalibrator(KeyboardTeleop):
    def __init__(self, runtime: RealConfig, *, output_path: str | Path):
        super().__init__(replace(runtime, hand=replace(runtime.hand, enabled=False)))
        self.output_path = Path(output_path).expanduser().resolve()
        self.camera = RealSenseCamera(self.runtime.camera, depth_enabled=False, auto_exposure_priority=None)
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []
        self.worst_sample: int | None = None
        self.calibration_saved = False
        self.preview_timestamp_ns = 0
        self.next_window_poll_ns = 0
        self.marker_status = "WAITING FOR RGB"
        self.status = "Collect stationary poses with varied rotations."

    def run(self) -> int:
        self.runtime.camera.validate()
        try:
            with ExitStack() as resources:
                self.resources = resources
                resources.callback(self.camera.disconnect)
                exit_code = super().run()
        except KeyboardInterrupt:
            return 130
        if exit_code == 0 and not self.calibration_saved:
            print("标定结束，未保存外参。")
            return 2
        return exit_code

    def start(self) -> None:
        hand_qpos = validate_hand_target(
            self.runtime.hand.home_qpos_rad, XHAND_RATED_QPOS_MIN_RAD, XHAND_RATED_QPOS_MAX_RAD,
        )
        static_boxes = self.runtime.environment.static_boxes
        for box in static_boxes:
            box.validate()
        # 标定点动的碰撞检查使用固定在 HOME 的手部模型。
        collision = CollisionModel(static_boxes=static_boxes, margin_m=0.0)
        self.jog.target_rejection = lambda qpos: collision.rejection(qpos, hand_qpos)
        print(f"ArUco {ARUCO_DICT_NAME}, ID={MARKER_ID}, 边长 {MARKER_SIZE_M * 1000:.1f} mm")
        print("将标记刚性固定在末端。拆除 XHand 或将其固定在配置的 HOME 姿态。")
        print("Space 停稳采样，Enter 求解，Backspace 撤销，X 剔除最差样本。")
        print(f"标定合格后更新外参文件：{self.output_path}")
        self.camera.connect()
        self.sampler = ArucoSampler(self.camera.color_intrinsics)
        self.preview = np.zeros((*self.runtime.camera.image_shape, 3), dtype=np.uint8)
        self.viewer = ImageViewer("ArUco Camera Calibration")
        self.resources.callback(self.viewer.close)
        self.show_preview()
        super().start()

    def poll_window(self, *, force: bool = False) -> None:
        now_ns = time.monotonic_ns()
        if not self.runtime_state.quit_requested and (force or now_ns >= self.next_window_poll_ns):
            self.next_window_poll_ns = now_ns + WINDOW_POLL_NS
            if "closed" in self.viewer.poll_events():
                self.runtime_state.request_quit()

    def check_services(self) -> bool:
        self.poll_window()
        return super().check_services()

    def show_preview(self, status: str | None = None) -> None:
        # 预览不能用 SDK 的阻塞等待，避免它耗尽点动命令的租期。
        frame = self.camera.read(timeout_s=0)
        if frame is not None:
            self.preview, self.marker_status = _marker_preview(frame.rgb, self.sampler)
            self.preview_timestamp_ns = frame.timestamp_ns
        marker = self.marker_status
        if self.preview_timestamp_ns and not sample_is_fresh(
            self.preview_timestamp_ns, self.runtime.camera.max_frame_age_s, time.monotonic_ns(),
        ):
            marker = "STALE"
        image = self.preview.copy()
        lines = (
            f"Marker: {marker}   Samples: {len(self.samples)}/{MIN_SAMPLES}",
            "WASD/arrows/IJKL: jog   R: HOME   Q: quit   Esc: stop",
            "Space: sample   Enter: solve   Backspace: undo   X: remove worst",
            self.status if status is None else status,
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_width = max(cv2.getTextSize(line, font, 0.48, 1)[0][0] for line in lines)
        scale = 0.48 * min(1.0, max(1, image.shape[1] - 16) / text_width)
        for row, line in enumerate(lines):
            origin = (8, 22 + row * 22)
            cv2.putText(image, line, origin, font, scale, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(image, line, origin, font, scale, (255, 255, 255), 1, cv2.LINE_AA)
        self.viewer.show(image)

    def check_operation(self, cancel_epoch: int, *, force_window_poll: bool = False) -> None:
        if force_window_poll:
            self.poll_window(force=True)
        self.robot.check()
        state = self.runtime_state
        if state.cancel_epoch != cancel_epoch or not state.requests_allow_motion():
            raise OperationCancelled("camera calibration operation cancelled")

    def prepare_jog(self, cancel_epoch: int) -> None:
        # 等待上一次操作期间按住的点动键释放，再恢复发送新目标。
        while self.keys.snapshot() & JOG_MOTION_KEYS:
            self.check_operation(cancel_epoch)
            self.show_preview("Release jog keys to continue.")
            time.sleep(0.01)
        self.check_operation(cancel_epoch, force_window_poll=True)
        super().prepare_jog(cancel_epoch)

    def handle_operation(self, events: list[str]) -> bool:
        state = self.runtime_state
        if "r" in events:
            self.poll_window(force=True)
            if state.requests_allow_motion():
                self.show_preview("Returning to HOME. See terminal.")
            return super().handle_operation(events)

        for key in events:
            if key in {"space", "enter"}:
                if state.stop_signal.is_set():
                    continue
                if key == "enter" and len(self.samples) < MIN_SAMPLES:
                    self.status = f"Need at least {MIN_SAMPLES} valid samples."
                    print(f"至少需要 {MIN_SAMPLES} 组有效样本，当前 {len(self.samples)} 组。", flush=True)
                    continue
                cancel_epoch = self.keys.accept_operation(state)
                if cancel_epoch is not None:
                    self.robot.check_run()
                    self.robot.stop()
                    self.check_operation(cancel_epoch)
                    if key == "space":
                        self.capture_sample(cancel_epoch)
                    else:
                        self.solve_and_save(cancel_epoch)
                    self.prepare_jog(cancel_epoch)
                    self.tick_due_ns = time.monotonic_ns() + self.period_ns
                return True
            elif key == "backspace" and self.samples:
                self.samples.pop()
                self.worst_sample = None
                self.status = "Last sample removed. Enter to recompute."
                print(f"已撤销末次采样，还剩 {len(self.samples)} 组。")
            elif key == "x":
                if self.worst_sample is None:
                    self.status = "Enter to solve before removing the worst sample."
                else:
                    worst = self.worst_sample
                    self.samples.pop(worst)
                    self.worst_sample = None
                    self.status = f"Removed sample #{worst + 1}. Enter to recompute."
                    print(f"已移除样本 #{worst + 1}，按 Enter 重算。")
        self.show_preview("Jog stopped. Press R for HOME." if state.stop_signal.is_set() else None)
        return False

    def capture_sample(self, cancel_epoch: int) -> None:
        def read_arm() -> np.void:
            self.check_operation(cancel_epoch)
            arm = self.robot.read_state().arm[0]
            if not sample_is_fresh(arm["timestamp_ns"], self.runtime.arm.feedback_max_age_s, time.monotonic_ns()):
                raise FeedbackUnavailable("calibration arm feedback unavailable/stale")
            return arm

        self.show_preview("Capturing stationary pose...")
        try:
            qpos, marker_pose = self.sampler.capture(
                self.camera.read, read_arm, camera_config=self.runtime.camera, arm_config=self.runtime.arm,
            )
        except SampleRejected as exc:
            print(f"采样拒绝：{exc}", flush=True)
            self.status = "Sample rejected. See terminal."
            return
        eef = self.jog.solver.fk.forward(qpos)
        self.check_operation(cancel_epoch, force_window_poll=True)
        self.samples.append((eef, marker_pose))
        self.worst_sample = None
        self.status = f"Captured sample #{len(self.samples)}."
        print(f"已采样 #{len(self.samples)}", flush=True)

    def solve_and_save(self, cancel_epoch: int) -> None:
        self.worst_sample = None
        self.show_preview("Solving hand-eye calibration...")
        eef, markers = (np.stack(poses) for poses in zip(*self.samples))
        try:
            result = solve_hand_eye(eef, markers)
        except ValueError as exc:
            print(f"求解失败：{exc}", flush=True)
            self.status = "Solve failed. See terminal."
            return
        self.check_operation(cancel_epoch)
        self.worst_sample = result.worst_sample
        residual = f"{result.position_rms_mm:.2f} mm / {result.rotation_rms_deg:.2f} deg"
        worst = result.worst_sample + 1
        print(f"{result.method}: RMS {residual}，最差样本 #{worst}", flush=True)
        if not result.quality_ok:
            self.status = f"High residual: {residual}   Worst #{worst}"
            print("外参质量未达标。按 X 移除最差样本，补采后按 Enter 重算。", flush=True)
            return

        intrinsics = self.camera.color_intrinsics
        metadata = {
            "width": intrinsics.width,
            "height": intrinsics.height,
            "intrinsics": intrinsics.matrix().tolist(),
            "distortion": list(intrinsics.distortion_coeffs),
            "method": result.method,
            "sample_count": len(self.samples),
            "position_rms_mm": result.position_rms_mm,
            "rotation_rms_deg": result.rotation_rms_deg,
            "calibrated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        save_camera_calibration(
            result.T_base_from_camera, self.camera.active_serial, self.output_path, calibration_capture=metadata,
            check_abort=lambda: self.check_operation(cancel_epoch, force_window_poll=True),
        )
        self.calibration_saved = True
        self.status = f"Saved: {result.method}   RMS {residual}   Worst #{worst}"
        print(f"外参已保存：{self.output_path}（相机 {self.camera.active_serial}）", flush=True)
