"""RealSense D400/L515 driver with RGB-D aligned to the color grid."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pyrealsense2 as rs

from dexmani_real.geometry.camera import CameraIntrinsics
from dexmani_real.sensor.camera.config import CameraParams

__all__ = ["RealSenseCamera", "RGBDFrame"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RGBDFrame:
    rgb: np.ndarray
    # Aligned depth uses color-grid intrinsics and the color-camera frame.
    depth_aligned_to_color_raw: np.ndarray
    depth_frame_number: int
    color_frame_number: int
    # Host monotonic queue-return time, before alignment/copying.
    timestamp_ns: int


class RealSenseCamera:
    def __init__(self, config: CameraParams, *, auto_exposure_priority: float | None = 0.0) -> None:
        if auto_exposure_priority is not None and (
            isinstance(auto_exposure_priority, bool)
            or not isinstance(auto_exposure_priority, (int, float))
            or not np.isfinite(auto_exposure_priority)
            or not 0.0 <= auto_exposure_priority <= 1.0
        ):
            raise ValueError("auto_exposure_priority must be None or a finite value in [0, 1]")
        self.config = config
        self.auto_exposure_priority = auto_exposure_priority
        self.active_serial: str | None = config.serial

        self.pipeline: rs.pipeline | None = None
        self.frame_queue: rs.frame_queue | None = None
        self.profile: rs.pipeline_profile | None = None

        self.depth_scale: float | None = None
        self.color_intrinsics: CameraIntrinsics | None = None
        self.depth_to_color_aligner: rs.align | None = None

    def connect(self) -> None:
        """Start once; failures require cleanup and an explicit new connection."""
        if self.pipeline is not None:
            raise RuntimeError("RealSense pipeline is already owned; disconnect before connecting")
        try:
            context = rs.context()
            devices = list(context.query_devices())
            if not devices:
                raise RuntimeError("No RealSense camera found.")
            if self.config.serial is None:
                if len(devices) != 1:
                    raise RuntimeError(
                        "Multiple RealSense cameras found; configure an explicit serial."
                    )
                device = devices[0]
                self.active_serial = self.get_device_info_value(
                    device, rs.camera_info.serial_number
                )
                if not self.active_serial:
                    raise RuntimeError("RealSense camera does not expose a serial number.")
            else:
                device = next(
                    (
                        device
                        for device in devices
                        if self.get_device_info_value(device, rs.camera_info.serial_number)
                        == self.config.serial
                    ),
                    None,
                )
                if device is None:
                    raise RuntimeError(
                        f"No RealSense camera found with serial={self.config.serial}."
                    )
            rs_config = self.create_rs_config()
            # Use a separate SDK context for L515 discovery isolation.
            self.pipeline = rs.pipeline()
            # Favor fresh feedback over retaining old SDK frames.
            self.frame_queue = rs.frame_queue(2)
            self.profile = self.pipeline.start(rs_config, self.frame_queue)
            sensor = self.profile.get_device().first_depth_sensor()
            if self.is_l515_device(device):
                self.apply_l515_depth_config(sensor)
            else:
                self.apply_d400_depth_config(sensor)
            self.apply_color_config()
            self.depth_scale = float(sensor.get_depth_scale())
            color_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
            self.color_intrinsics = self.intrinsics_from_sdk(color_profile.get_intrinsics())
            self.depth_to_color_aligner = rs.align(rs.stream.color)
            for _ in range(self.config.warmup_frames):
                # Initialize alignment and owned copies before live publication.
                self.read(timeout_ms=5000)
        except BaseException:
            try:
                self.disconnect()
            except BaseException:
                logger.exception("RealSense startup cleanup failed; pipeline remains owned")
            raise

    def apply_l515_depth_config(self, sensor) -> None:
        """Apply one factory preset and an optional confidence override."""
        cfg = self.config

        if not sensor.supports(rs.option.visual_preset):
            raise RuntimeError("connected L515 does not expose visual_preset")
        sensor.set_option(rs.option.visual_preset, float(cfg.l515_visual_preset))
        time.sleep(0.5)
        base_preset = sensor.get_option(rs.option.visual_preset)
        if not np.isclose(base_preset, float(cfg.l515_visual_preset), atol=1e-6):
            raise RuntimeError(
                "L515 visual preset readback mismatch: "
                f"requested={cfg.l515_visual_preset}, actual={base_preset}"
            )

        if cfg.l515_confidence_threshold is not None:
            if not sensor.supports(rs.option.confidence_threshold):
                raise RuntimeError(
                    "configured L515 confidence override is unsupported by this device"
                )
            sensor.set_option(
                rs.option.confidence_threshold,
                float(cfg.l515_confidence_threshold),
            )

            confidence = sensor.get_option(rs.option.confidence_threshold)
            if not np.isclose(confidence, float(cfg.l515_confidence_threshold), atol=1e-6):
                raise RuntimeError(
                    "L515 confidence readback mismatch: "
                    f"requested={cfg.l515_confidence_threshold}, actual={confidence}"
                )

    def get_color_sensor(self) -> Any:
        if self.profile is None:
            raise RuntimeError("RealSense profile is unavailable for color settings")
        device = self.profile.get_device()
        for sensor in device.query_sensors():
            for profile in sensor.get_stream_profiles():
                try:
                    if profile.stream_type() == rs.stream.color:
                        return sensor
                except RuntimeError:
                    continue
        raise RuntimeError("no color sensor with a color stream profile found")

    def apply_color_config(self) -> None:
        """Apply color options; priority 0 limits exposure to the frame period and None preserves defaults."""
        priority = self.auto_exposure_priority
        if priority is None:
            return
        try:
            sensor = self.get_color_sensor()
        except RuntimeError as exc:
            logger.warning(
                "color sensor unavailable; auto_exposure_priority not applied: %s",
                exc,
            )
            return
        option = rs.option.auto_exposure_priority
        try:
            if not sensor.supports(option):
                logger.warning(
                    "color sensor does not expose auto_exposure_priority; left at device default"
                )
                return
            sensor.set_option(option, float(priority))
            readback = float(sensor.get_option(option))
        except (RuntimeError, OSError) as exc:
            logger.warning("auto_exposure_priority could not be set: %s", exc)
            return
        if not np.isclose(readback, float(priority), atol=1e-6):
            logger.warning(
                "auto_exposure_priority readback mismatch: requested=%s, actual=%s",
                float(priority),
                readback,
            )
        else:
            logger.debug("color auto_exposure_priority set to %s (0=OFF)", readback)

    def apply_d400_depth_config(self, sensor) -> None:
        """Enable the D400 IR emitter and available High Accuracy preset for textureless depth."""
        options: list[tuple[rs.option, float]] = []
        if sensor.supports(rs.option.emitter_enabled):
            options.append((rs.option.emitter_enabled, 1.0))
        if sensor.supports(rs.option.visual_preset):
            options.append((rs.option.visual_preset, 3.0))

        for option, value in options:
            try:
                sensor.set_option(option, value)
                logger.debug("D400 set_option(%s) applied: %s", option, value)
            except (RuntimeError, OSError) as error:
                logger.warning("D400 set_option(%s) failed: %s", option, error)

    def disconnect(self) -> None:
        # A failed stop retains ownership so callers can report or retry cleanup.
        if self.pipeline is not None:
            self.pipeline.stop()
        self.pipeline = None
        self.frame_queue = None
        self.profile = None
        self.depth_scale = None
        self.color_intrinsics = None
        self.depth_to_color_aligner = None

    def create_rs_config(self) -> rs.config:
        width, height = self.config.width, self.config.height

        rs_config = rs.config()
        rs_config.enable_device(self.active_serial)
        rs_config.enable_stream(rs.stream.depth, width, height, rs.format.z16, self.config.fps)
        rs_config.enable_stream(
            rs.stream.color,
            width,
            height,
            rs.format.bgr8,
            self.config.fps,
        )
        return rs_config

    @staticmethod
    def is_l515_device(device: rs.device) -> bool:
        name = RealSenseCamera.get_device_info_value(device, rs.camera_info.name).upper()
        product_line = RealSenseCamera.get_device_info_value(device, rs.camera_info.product_line)
        return product_line == "L500" or "L515" in name

    @staticmethod
    def intrinsics_from_sdk(intrinsics: Any) -> CameraIntrinsics:
        coefficients = tuple(float(value) for value in intrinsics.coeffs)
        return CameraIntrinsics(
            width=int(intrinsics.width),
            height=int(intrinsics.height),
            fx=float(intrinsics.fx),
            fy=float(intrinsics.fy),
            ppx=float(intrinsics.ppx),
            ppy=float(intrinsics.ppy),
            distortion_model=str(intrinsics.model),
            distortion_coeffs=cast(tuple[float, float, float, float, float], coefficients),
        )

    def read(self, timeout_ms: int = 5000) -> RGBDFrame:
        if self.frame_queue is None or self.depth_to_color_aligner is None:
            raise RuntimeError("RealSense is not connected. Call connect() first.")

        queued_frame = self.frame_queue.wait_for_frame(timeout_ms)
        # Timestamp queue return before frameset recovery, alignment, or owned copies.
        wait_return_monotonic_ns = time.monotonic_ns()
        frames = queued_frame.as_frameset()
        if not frames:
            raise RuntimeError("RealSense frame queue returned a non-frameset frame.")
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame:
            raise RuntimeError("Failed to get depth frame.")
        if not color_frame:
            raise RuntimeError("Failed to get color frame.")

        aligned_frames = self.depth_to_color_aligner.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        if not aligned_depth_frame:
            raise RuntimeError("Failed to align depth to the color stream.")
        aligned_depth_raw = np.array(
            aligned_depth_frame.get_data(), dtype=np.uint16, copy=True, order="C"
        )

        # Force an owned copy: ascontiguousarray may retain SDK buffers that expire on return.
        bgr = np.asarray(color_frame.get_data(), dtype=np.uint8)
        rgb = np.array(bgr[..., ::-1], dtype=np.uint8, copy=True, order="C")

        # Device frame numbers expose stalls/drops that a local counter would hide.
        depth_frame_number = int(depth_frame.get_frame_number())
        color_frame_number = int(color_frame.get_frame_number())
        return RGBDFrame(
            rgb=rgb,
            depth_aligned_to_color_raw=aligned_depth_raw,
            depth_frame_number=depth_frame_number,
            color_frame_number=color_frame_number,
            timestamp_ns=wait_return_monotonic_ns,
        )

    def get_color_intrinsics(self) -> CameraIntrinsics:
        if self.color_intrinsics is None:
            raise RuntimeError("RealSense color intrinsics are unavailable.")
        return self.color_intrinsics

    def get_depth_scale(self) -> float:
        if self.depth_scale is None:
            raise RuntimeError("RealSense is not connected or depth_scale is unavailable.")
        return float(self.depth_scale)

    def get_device_info(self) -> dict:
        if self.profile is None:
            raise RuntimeError("RealSense is not connected.")
        return self.device_info(self.profile.get_device())

    @staticmethod
    def device_info(device: rs.device) -> dict[str, str]:
        return {
            name: RealSenseCamera.get_device_info_value(device, key)
            for name, key in (
                ("name", rs.camera_info.name),
                ("serial", rs.camera_info.serial_number),
                ("firmware", rs.camera_info.firmware_version),
                ("product_line", rs.camera_info.product_line),
            )
        }

    @staticmethod
    def get_device_info_value(device: rs.device, key: rs.camera_info) -> str:
        try:
            return str(device.get_info(key)) if device.supports(key) else ""
        except RuntimeError:
            return ""

    @staticmethod
    def list_cameras() -> list[dict[str, str]]:
        context = rs.context()
        return [RealSenseCamera.device_info(device) for device in context.query_devices()]
