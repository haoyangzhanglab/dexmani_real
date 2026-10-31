from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from dexmani_real.config.sensor import CameraConfig
from dexmani_real.geometry.camera import CameraIntrinsics
from dexmani_real.utils.log import configure_logging

if TYPE_CHECKING:
    import pyrealsense2 as rs

logger = logging.getLogger(__name__)
WARMUP_FRAMES = 10
SOURCE_STALL_TIMEOUT_NS = 2_000_000_000


@dataclass(frozen=True)
class RGBDFrame:
    """深度对齐到 color 网格，时间为较早通道的首次主机接收时刻。"""

    rgb: np.ndarray
    depth: np.ndarray
    depth_frame_number: int
    color_frame_number: int
    timestamp_ns: int


class RealSenseCamera:
    def __init__(self, config: CameraConfig, *, auto_exposure_priority: float | None = 0.0):
        if auto_exposure_priority is not None and not 0 <= auto_exposure_priority <= 1:
            raise ValueError("auto_exposure_priority must be in [0, 1] or None")
        self.config = config
        self.auto_exposure_priority = auto_exposure_priority
        self.active_serial: str
        self.pipeline: rs.pipeline | None = None
        self.condition = threading.Condition()
        self.pending_frame = None
        self.callback_error: Exception | None = None
        self.last_identity: tuple[int, int] | None = None
        self.received_ns = [0, 0]

    def connect(self):
        import pyrealsense2 as rs

        if self.pipeline is not None:
            raise RuntimeError("RealSense is already connected")
        context = rs.context()
        devices = list(context.query_devices())
        if self.config.serial is None:
            if len(devices) != 1:
                raise RuntimeError("Expected one RealSense camera; configure a serial otherwise")
            self.active_serial = devices[0].get_info(rs.camera_info.serial_number)
        else:
            self.active_serial = self.config.serial
            if not any(
                d.get_info(rs.camera_info.serial_number) == self.active_serial for d in devices
            ):
                raise RuntimeError(f"No RealSense camera with serial={self.active_serial}")

        config = rs.config()
        config.enable_device(self.active_serial)
        for stream, pixel_format in (
            (rs.stream.depth, rs.format.z16),
            (rs.stream.color, rs.format.bgr8),
        ):
            config.enable_stream(
                stream, self.config.width, self.config.height, pixel_format, self.config.fps
            )

        try:
            # 独立 SDK context 保留 L515 的发现隔离。
            with self.condition:
                self.pending_frame = self.callback_error = self.last_identity = None
                self.received_ns = [time.monotonic_ns()] * 2
            self.pipeline = rs.pipeline()
            profile = self.pipeline.start(config, self.receive_frame)
            device = profile.get_device()
            depth_sensor = device.first_depth_sensor()
            color_sensor = device.first_color_sensor()
            name = device.get_info(rs.camera_info.name)
            product_line = device.get_info(rs.camera_info.product_line)
            if product_line == "L500" or "L515" in name.upper():
                preset, confidence = self.config.l515_visual_preset, self.config.l515_confidence_threshold
                if type(preset) is not int or not 0 <= preset <= 5:
                    raise ValueError("camera l515_visual_preset must be an integer in [0, 5]")
                if confidence is not None and (type(confidence) is not int or not 0 <= confidence <= 3):
                    raise ValueError("camera l515_confidence_threshold must be in [0, 3] or null")
                depth_sensor.set_option(rs.option.visual_preset, self.config.l515_visual_preset)
                time.sleep(0.5)
                if self.config.l515_confidence_threshold is not None:
                    depth_sensor.set_option(
                        rs.option.confidence_threshold, self.config.l515_confidence_threshold
                    )
            else:
                for option, value in (
                    (rs.option.emitter_enabled, 1.0),
                    (rs.option.visual_preset, 3.0),
                ):
                    if depth_sensor.supports(option):
                        depth_sensor.set_option(option, value)
            if self.auto_exposure_priority is not None:
                option = rs.option.auto_exposure_priority
                if color_sensor.supports(option):
                    color_sensor.set_option(option, self.auto_exposure_priority)

            self.depth_scale_m = float(depth_sensor.get_depth_scale())
            if not np.isfinite(self.depth_scale_m) or self.depth_scale_m <= 0:
                raise ValueError("RealSense depth scale must be finite and positive")
            color = (
                profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            )
            self.color_intrinsics = CameraIntrinsics(
                width=color.width,
                height=color.height,
                fx=color.fx,
                fy=color.fy,
                ppx=color.ppx,
                ppy=color.ppy,
                distortion_model=str(color.model),
                distortion_coeffs=tuple(color.coeffs),
            )
            self.aligner = rs.align(rs.stream.color)
            for _ in range(WARMUP_FRAMES):
                self.read()
        except BaseException:
            try:
                self.disconnect()
            except BaseException:
                logger.exception("RealSense startup cleanup failed")
            raise

    def receive_frame(self, queued):
        """SDK 回调入口；帧和首次主机接收时间一起进入有界最新帧槽。"""
        received = time.monotonic_ns()
        try:
            frames = queued.as_frameset()
            depth, color = frames.get_depth_frame(), frames.get_color_frame()
            if not depth or not color:
                raise RuntimeError("RealSense frameset is missing depth or RGB")
            identity = (depth.get_frame_number(), color.get_frame_number())
            with self.condition:
                if self.callback_error is not None:
                    return
                for channel, name in enumerate(("depth", "RGB")):
                    number = identity[channel]
                    if number < 0 or (
                        self.last_identity is not None and number < self.last_identity[channel]
                    ):
                        raise RuntimeError(
                            f"RealSense {name} frame identity is invalid or moved backward"
                        )
                    if self.last_identity is None or number != self.last_identity[channel]:
                        self.received_ns[channel] = received
                if identity == self.last_identity:
                    return
                frames.keep()
                self.pending_frame = frames, identity, tuple(self.received_ns)
                self.last_identity = identity
                self.condition.notify()
        except Exception as error:
            # Python 回调异常不能留在 SDK 线程中；交给设备 owner 的 read() 报告。
            with self.condition:
                if self.callback_error is None:
                    self.callback_error = error
                self.pending_frame = None
                self.condition.notify()

    def read(self) -> RGBDFrame | None:
        with self.condition:
            self.condition.wait_for(
                lambda: self.pending_frame is not None or self.callback_error is not None,
                timeout=0.3,
            )
            if self.callback_error is not None:
                raise self.callback_error
            pending, self.pending_frame = self.pending_frame, None
            now = time.monotonic_ns()
            if pending is None:
                if now - min(self.received_ns) >= SOURCE_STALL_TIMEOUT_NS:
                    raise TimeoutError("RealSense stopped producing RGB-D frames")
                return None

        frames, identity, received_ns = pending
        for name, received in zip(("depth", "RGB"), received_ns):
            if now - received >= SOURCE_STALL_TIMEOUT_NS:
                raise TimeoutError(f"RealSense stopped producing new {name}")

        aligned = self.aligner.process(frames).get_depth_frame()
        if not aligned:
            raise RuntimeError("Failed to align RealSense depth to color")
        # SDK 缓冲在返回后可能复用，像素必须拥有独立内存。
        depth_array = np.array(aligned.get_data(), dtype=np.uint16, copy=True, order="C")
        rgb = np.array(
            np.asarray(frames.get_color_frame().get_data())[..., ::-1],
            dtype=np.uint8, copy=True, order="C",
        )
        return RGBDFrame(rgb, depth_array, *identity, min(received_ns))

    def disconnect(self):
        # stop 失败时保留句柄，让资源所有者能继续收尾。
        if self.pipeline is not None:
            # stop 等待在途回调退出，不能持有回调也要取得的 condition。
            self.pipeline.stop()
            self.pipeline = None
            with self.condition:
                self.pending_frame = None


def run_camera(shared, config: CameraConfig):
    configure_logging()
    camera = RealSenseCamera(config)
    camera.connect()
    try:
        shared.camera_depth_scale.value = camera.depth_scale_m
        shared.camera_serial.value = camera.active_serial.encode()
        shared.camera_intrinsics.value = json.dumps(camera.color_intrinsics.to_dict()).encode()
        while shared.is_running.value:
            frame = camera.read()
            if frame is None:
                continue
            if (
                frame.rgb.shape != (config.height, config.width, 3)
                or frame.depth.shape != (config.height, config.width)
            ):
                raise ValueError("camera pixels must match the configured RGB-D shape")
            shared.camera_ring.write(
                timestamp_ns=frame.timestamp_ns,
                color_frame_number=frame.color_frame_number,
                rgb=frame.rgb,
                depth=frame.depth,
            )
            shared.camera_ready.set()
    finally:
        shared.camera_ready.clear()
        camera.disconnect()
