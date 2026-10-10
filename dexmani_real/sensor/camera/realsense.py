"""RealSense D400/L515 driver with native and depth-to-color RGB-D payloads."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, cast

__all__ = [
    "RealSenseCamera",
    "RGBDFrame",
]

import numpy as np
import pyrealsense2 as rs

from dexmani_real.config.hardware import CameraParams
from dexmani_real.sensor.camera.geometry import CameraIntrinsics, RGBDGeometry


logger = logging.getLogger(__name__)


_L515_OPTION_NAMES = (
    "visual_preset",
    "confidence_threshold",
    "laser_power",
    "receiver_gain",
    "noise_filtering",
    "zero_order_enabled",
    "invalidation_bypass",
)


@dataclass(frozen=True)
class RGBDFrame:
    rgb: np.ndarray
    # Native depth is retained for measurement provenance. Production point
    # clouds use the aligned payload with ``geometry.aligned_depth_to_color``.
    depth_raw: np.ndarray
    # ``rs.align(depth -> color)`` samples are deprojected with color intrinsics
    # and transformed from the color-camera frame.
    depth_aligned_to_color_raw: np.ndarray | None
    depth_frame_number: int
    color_frame_number: int
    depth_device_timestamp_s: float
    color_device_timestamp_s: float
    depth_timestamp_domain: int
    color_timestamp_domain: int
    depth_source_ns: int
    color_source_ns: int
    # Host queue-return time, before alignment/copying; distinct from source time.
    timestamp_ns: int
    depth_scale: float
    serial: str | None


def _source_timestamp_ns(timestamp_s, domain, *, received_ns, realtime_ns):
    """Map SDK global/system time to host monotonic; 0 rejects an invalid clock.

    Hardware-only timestamps have no known host offset, so retain the explicit
    receive-time origin for that domain. Never estimate an offset from arrivals:
    that would hide camera/USB queue latency. Domain and SDK time travel with RGB-D.
    """
    if not np.isfinite(timestamp_s) or timestamp_s <= 0:
        return 0
    if domain == int(rs.timestamp_domain.hardware_clock):
        return received_ns
    if domain not in (int(rs.timestamp_domain.global_time), int(rs.timestamp_domain.system_time)):
        return 0
    source_ns = received_ns + round(timestamp_s * 1e9) - realtime_ns
    return source_ns if 0 < source_ns <= received_ns else 0


class RealSenseCamera:
    def __init__(self, config: CameraParams, *, auto_exposure_priority: float | None = 0.0) -> None:
        config.validate()
        if auto_exposure_priority is not None and (
            isinstance(auto_exposure_priority, bool)
            or not isinstance(auto_exposure_priority, (int, float))
            or not np.isfinite(auto_exposure_priority)
            or not 0.0 <= auto_exposure_priority <= 1.0
        ):
            raise ValueError("auto_exposure_priority must be None or a finite value in [0, 1]")
        self.config = config
        self.auto_exposure_priority = auto_exposure_priority
        self.context = None
        self.active_serial: str | None = config.serial
        self.active_is_l515 = False

        self.pipeline: rs.pipeline | None = None
        self.frame_queue: rs.frame_queue | None = None
        self.profile: rs.pipeline_profile | None = None

        self.depth_scale: float | None = None
        self.geometry: RGBDGeometry | None = None
        self._depth_to_color_aligner: rs.align | None = None

    def connect(self) -> bool:
        """Open the pipeline; return True on success or if already connected."""
        if self.pipeline is not None:
            return True

        return self._open_pipeline()

    def _open_pipeline(self) -> bool:
        """Discover device, start + warm up the pipeline.

        Returns True on success. On any failure the pipeline is torn down (via
        disconnect) so a started-but-unusable pipeline is never left holding the
        device — otherwise the next open would hit "Device or resource busy".
        """
        self.context = rs.context()
        try:
            self.active_serial = self.config.serial or self._find_default_serial_in_context()
            device = self._find_device_by_serial_in_context(self.active_serial)
            self.active_is_l515 = self.is_l515_device(device)
        except (RuntimeError, OSError) as e:
            logger.warning("connect() failed at device discovery / config: %s", e)
            return False

        rs_config = self.create_rs_config()
        try:
            self._start_pipeline(rs_config)
        except (RuntimeError, OSError) as e:
            logger.warning("connect() failed at pipeline start: %s", e)
            self.disconnect()
            return False
        try:
            self._setup_pipeline_post_start()
        except (RuntimeError, OSError) as e:
            logger.warning("connect() failed at post-start setup: %s", e)
            self.disconnect()
            return False
        try:
            self._warmup_pipeline()
        except (RuntimeError, OSError) as e:
            logger.warning("connect() failed at warmup: %s", e)
            self.disconnect()
            return False
        return True

    def _apply_depth_config(self) -> None:
        """Apply L515 or D400 depth settings after pipeline start."""
        if self.profile is None:
            return

        if self.active_is_l515:
            self._apply_l515_depth_config()
        else:
            self._apply_d400_depth_config()

    def _apply_l515_depth_config(self) -> None:
        """Apply one factory preset and an optional confidence override."""
        cfg = self.config

        if self.profile is None:
            raise RuntimeError("RealSense profile is unavailable for L515 settings")
        sensor = self.profile.get_device().first_depth_sensor()
        if not sensor.supports(rs.option.visual_preset):
            raise RuntimeError("connected L515 does not expose visual_preset")
        sensor.set_option(rs.option.visual_preset, float(cfg.l515_visual_preset))
        time.sleep(0.5)
        base_readbacks = self._read_l515_option_snapshot(sensor)
        base_preset = base_readbacks["visual_preset"]
        if base_preset is None or not np.isclose(base_preset, float(cfg.l515_visual_preset), atol=1e-6):
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

        final_readbacks = self._read_l515_option_snapshot(sensor)
        final_confidence = final_readbacks["confidence_threshold"]
        if cfg.l515_confidence_threshold is not None and (
            final_confidence is None
            or not np.isclose(final_confidence, float(cfg.l515_confidence_threshold), atol=1e-6)
        ):
            raise RuntimeError(
                "L515 confidence readback mismatch: "
                f"requested={cfg.l515_confidence_threshold}, actual={final_confidence}"
            )
        option_snapshot = {
            "base_visual_preset": int(cfg.l515_visual_preset),
            "base_readbacks": base_readbacks,
            "confidence_override": cfg.l515_confidence_threshold,
            "final_readbacks": final_readbacks,
        }
        logger.debug("L515 depth option snapshot: %s", option_snapshot)

    @staticmethod
    def _read_l515_option_snapshot(sensor: Any) -> dict[str, float | None]:
        """Read only supported, firmware-owned L515 options."""
        snapshot: dict[str, float | None] = {}
        for name in _L515_OPTION_NAMES:
            option = getattr(rs.option, name, None)
            if option is None:
                snapshot[name] = None
                continue
            try:
                snapshot[name] = (
                    float(sensor.get_option(option)) if sensor.supports(option) else None
                )
            except (RuntimeError, OSError):
                snapshot[name] = None
        return snapshot

    def get_color_sensor(self) -> Any:
        """Return the active RGB sensor for driver setup and diagnostics."""
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

    def _apply_color_config(self) -> None:
        """Apply color sensor options after pipeline start.

        Priority 0 caps auto exposure at the frame period to preserve FPS;
        it does not disable auto exposure. None leaves the device default.
        """
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

    def _setup_pipeline_post_start(self) -> None:
        """Configure active-device options and immutable native geometry."""
        if self.profile is None:
            raise RuntimeError("Pipeline profile is unavailable after start.")

        self._apply_depth_config()
        self._apply_color_config()

        self.set_global_time()
        self.depth_scale = float(self.profile.get_device().first_depth_sensor().get_depth_scale())
        self.update_geometry_from_profile()
        self._depth_to_color_aligner = rs.align(rs.stream.color)

    def _apply_d400_depth_config(self) -> None:
        """Apply D400 depth settings after pipeline start.

        D400 (D415/D435/D455): enable the IR emitter (stereo dot projector)
        for robust depth on textureless surfaces, and set the High Accuracy
        visual preset when available.
        """
        if self.profile is None:
            return
        try:
            sensor = self.profile.get_device().first_depth_sensor()
        except RuntimeError:
            return
        if sensor is None:
            return

        options: list[tuple[rs.option, float]] = []
        if sensor.supports(rs.option.emitter_enabled):
            options.append((rs.option.emitter_enabled, 1.0))
        if sensor.supports(rs.option.visual_preset):
            options.append((rs.option.visual_preset, 3.0))

        for option, value in options:
            try:
                sensor.set_option(option, value)
            except (RuntimeError, OSError) as error:
                logger.warning("D400 set_option(%s) failed: %s", option, error)

        if options:
            logger.info("D400 depth config applied: %s", [o.name for o, _ in options])

    def _start_pipeline(self, rs_config: rs.config) -> None:
        """Create and start a fresh pipeline.

        A separate context avoids L515 resolve errors from discovery handles.
        The SDK queue decouples delivery from alignment/copying with one SDK owner.
        """
        self.pipeline = rs.pipeline()
        rs_config.resolve(rs.pipeline_wrapper(self.pipeline))
        self.frame_queue = rs.frame_queue(int(self.config.frame_queue_capacity))
        self.profile = self.pipeline.start(rs_config, self.frame_queue)

    def _warmup_pipeline(self) -> None:
        """Warm up frame processing for ``warmup_frames``, restarting if necessary."""
        if self.pipeline is None or self.frame_queue is None:
            raise RuntimeError("Pipeline is unavailable during warmup.")

        warmup_frames = max(int(self.config.warmup_frames), 0)
        if warmup_frames == 0:
            return

        max_restarts = 3
        for attempt in range(max_restarts + 1):
            try:
                for _ in range(warmup_frames):
                    # Initialize alignment and image copies before live publication.
                    self.read(timeout_ms=5000)
                return
            except RuntimeError:
                if attempt >= max_restarts:
                    raise

                delay = 3.0 * (attempt + 1)
                logger.warning(
                    "L515 warmup failed; restarting pipeline after %.0f s (attempt %d/%d).",
                    delay,
                    attempt + 1,
                    max_restarts,
                )
                try:
                    self.pipeline.stop()
                except RuntimeError:
                    logger.warning("RealSense pipeline stop failed before restart", exc_info=True)
                time.sleep(delay)
                self._start_pipeline(self.create_rs_config())
                # Reapply Short Range and rebuild active-profile state after every restart.
                self._setup_pipeline_post_start()
                time.sleep(1.0)

    def disconnect(self) -> None:
        """Close the pipeline; repeated calls are safe."""
        if self.pipeline is None:
            return
        try:
            self.pipeline.stop()
        except RuntimeError:
            logger.warning("RealSense pipeline stop failed during disconnect", exc_info=True)
        finally:
            self.pipeline = None
            self.frame_queue = None
            self.profile = None
            self.depth_scale = None
            self.geometry = None
            self._depth_to_color_aligner = None

    def create_rs_config(self) -> rs.config:
        width, height = self.config.width, self.config.height

        rs_config = rs.config()
        rs_config.enable_device(self.active_serial)
        rs_config.enable_stream(
            rs.stream.depth, width, height, rs.format.z16, self.config.fps
        )
        rs_config.enable_stream(
            rs.stream.color,
            width,
            height,
            rs.format.bgr8,
            self.config.fps,
        )
        return rs_config

    def set_global_time(self) -> None:
        if self.profile is None:
            return
        for sensor in self.profile.get_device().query_sensors():
            try:
                if sensor.supports(rs.option.global_time_enabled):
                    sensor.set_option(rs.option.global_time_enabled, 1)
            except RuntimeError:
                logger.warning("RealSense global-time option could not be enabled", exc_info=True)

    @staticmethod
    def is_l515_device(device: rs.device) -> bool:
        name = RealSenseCamera.get_device_info_value(device, rs.camera_info.name).upper()
        product_line = RealSenseCamera.get_device_info_value(device, rs.camera_info.product_line)
        return product_line == "L500" or "L515" in name

    @staticmethod
    def _intrinsics_from_sdk(intrinsics: Any) -> CameraIntrinsics:
        coefficients = tuple(float(value) for value in intrinsics.coeffs)
        if len(coefficients) != 5:
            raise RuntimeError("RealSense intrinsics did not provide five distortion coefficients")
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

    @staticmethod
    def _transform_from_sdk_extrinsics(extrinsics: Any) -> np.ndarray:
        """Decode SDK rotation storage and verify the chosen convention."""
        raw_rotation = np.asarray(extrinsics.rotation, dtype=np.float64).reshape(3, 3)
        translation = np.asarray(extrinsics.translation, dtype=np.float64)
        points = np.vstack((np.zeros(3), np.eye(3)))
        oracle = np.asarray(
            [rs.rs2_transform_point_to_point(extrinsics, point.tolist()) for point in points],
            dtype=np.float64,
        )
        for rotation in (raw_rotation.T, raw_rotation):
            transformed = points @ rotation.T + translation
            if np.allclose(transformed, oracle, rtol=0.0, atol=1e-7):
                transform = np.eye(4, dtype=np.float64)
                transform[:3, :3] = rotation
                transform[:3, 3] = translation
                return transform
        raise RuntimeError("failed to decode RealSense extrinsic rotation convention")

    def update_geometry_from_profile(self) -> None:
        """Read immutable native stream calibration from the active profile."""
        if self.profile is None:
            raise RuntimeError("RealSense is not connected.")
        depth_profile = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
        color_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.geometry = RGBDGeometry(
            depth=self._intrinsics_from_sdk(depth_profile.get_intrinsics()),
            color=self._intrinsics_from_sdk(color_profile.get_intrinsics()),
            T_color_from_depth=self._transform_from_sdk_extrinsics(
                depth_profile.get_extrinsics_to(color_profile)
            ),
        )

    def read(self, timeout_ms: int = 5000) -> RGBDFrame:
        if self.pipeline is None or self.frame_queue is None:
            raise RuntimeError("RealSense is not connected. Call connect() first.")
        if self.depth_scale is None:
            raise RuntimeError("RealSense depth_scale is unavailable.")

        # Capture native frames for provenance, then construct the aligned
        # depth-to-color payload used by geometry-sensitive point-cloud work.
        queued_frame = self.frame_queue.wait_for_frame(timeout_ms)
        # Timestamp immediately after the queue wait returns, before frameset
        # recovery or any array ownership copies.
        wait_return_monotonic_ns = time.monotonic_ns()
        # Sample monotonic first: scheduling between these calls can only make
        # the mapped source older. SDK global_time is in OS realtime, not monotonic.
        wait_return_realtime_ns = time.time_ns()
        frames = queued_frame.as_frameset()
        if not frames:
            raise RuntimeError("RealSense frame queue returned a non-frameset frame.")
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame:
            raise RuntimeError("Failed to get depth frame.")
        if not color_frame:
            raise RuntimeError("Failed to get color frame.")

        aligned_depth_raw: np.ndarray | None = None
        if self._depth_to_color_aligner is not None:
            aligned_frames = self._depth_to_color_aligner.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            if not aligned_depth_frame:
                raise RuntimeError("Failed to align depth to the color stream.")
            aligned_depth_raw = np.array(
                aligned_depth_frame.get_data(), dtype=np.uint16, copy=True, order="C"
            )

        # ``ascontiguousarray`` may return an SDK-backed view. Use ``array``
        # with copy=True because the frame becomes invalid after this method.
        depth_raw = np.array(depth_frame.get_data(), dtype=np.uint16, copy=True, order="C")

        # The frame keeps the SDK buffer alive until this owned RGB copy completes.
        bgr = np.asarray(color_frame.get_data(), dtype=np.uint8)
        rgb = np.array(bgr[..., ::-1], dtype=np.uint8, copy=True, order="C")

        # Preserve the device-provided frame number.  Unlike a local counter,
        # this exposes device/pipeline stalls and dropped frames end-to-end.
        depth_frame_number = int(depth_frame.get_frame_number())
        depth_timestamp_s = float(depth_frame.get_timestamp()) * 1e-3
        depth_timestamp_domain = int(depth_frame.get_frame_timestamp_domain())
        color_timestamp_s = float(color_frame.get_timestamp()) * 1e-3
        color_timestamp_domain = int(color_frame.get_frame_timestamp_domain())
        color_frame_number = int(color_frame.get_frame_number())
        frame = RGBDFrame(
            rgb=rgb,
            depth_raw=depth_raw,
            depth_aligned_to_color_raw=aligned_depth_raw,
            depth_frame_number=depth_frame_number,
            color_frame_number=color_frame_number,
            depth_device_timestamp_s=depth_timestamp_s,
            color_device_timestamp_s=color_timestamp_s,
            depth_timestamp_domain=depth_timestamp_domain,
            color_timestamp_domain=color_timestamp_domain,
            depth_source_ns=_source_timestamp_ns(
                depth_timestamp_s, depth_timestamp_domain,
                received_ns=wait_return_monotonic_ns, realtime_ns=wait_return_realtime_ns,
            ),
            color_source_ns=_source_timestamp_ns(
                color_timestamp_s, color_timestamp_domain,
                received_ns=wait_return_monotonic_ns, realtime_ns=wait_return_realtime_ns,
            ),
            timestamp_ns=wait_return_monotonic_ns,
            depth_scale=float(self.depth_scale),
            serial=self.active_serial,
        )

        return frame

    def get_geometry(self) -> RGBDGeometry:
        if self.geometry is None:
            raise RuntimeError("RealSense native RGB-D geometry is unavailable.")
        return RGBDGeometry.from_dict(self.geometry.to_dict())

    def get_depth_scale(self) -> float:
        if self.depth_scale is None:
            raise RuntimeError("RealSense is not connected or depth_scale is unavailable.")
        return float(self.depth_scale)

    def get_device_info(self) -> dict:
        if self.profile is None:
            raise RuntimeError("RealSense is not connected.")
        device = self.profile.get_device()
        info = {}
        for name, key in [
            ("name", rs.camera_info.name),
            ("serial", rs.camera_info.serial_number),
            ("firmware", rs.camera_info.firmware_version),
            ("product_line", rs.camera_info.product_line),
        ]:
            try:
                info[name] = device.get_info(key) if device.supports(key) else ""
            except RuntimeError:
                info[name] = ""
        return info

    @staticmethod
    def get_device_info_value(device: rs.device, key: rs.camera_info) -> str:
        try:
            return str(device.get_info(key)) if device.supports(key) else ""
        except RuntimeError:
            return ""

    def _find_device_by_serial_in_context(self, serial: str) -> rs.device:
        for device in self.context.query_devices():
            device_serial = self.get_device_info_value(device, rs.camera_info.serial_number)
            if device_serial == serial:
                return device
        raise RuntimeError(f"No RealSense camera found with serial={serial}.")

    def _find_default_serial_in_context(self) -> str:
        devices = self.context.query_devices()
        if len(devices) == 0:
            raise RuntimeError("No RealSense camera found.")
        if len(devices) != 1:
            serials = [
                self.get_device_info_value(device, rs.camera_info.serial_number)
                for device in devices
            ]
            raise RuntimeError(
                "Multiple RealSense cameras found; configure an explicit serial "
                f"instead of relying on discovery order (connected={serials})."
            )
        serial = self.get_device_info_value(devices[0], rs.camera_info.serial_number)
        if not serial:
            raise RuntimeError("The first RealSense camera does not expose a serial number.")
        return serial

    @staticmethod
    def list_cameras() -> list[dict[str, str]]:
        context = rs.context()
        cameras = []
        for device in context.query_devices():
            item = {}
            for name, key in [
                ("serial", rs.camera_info.serial_number),
                ("name", rs.camera_info.name),
                ("firmware", rs.camera_info.firmware_version),
                ("product_line", rs.camera_info.product_line),
            ]:
                try:
                    item[name] = device.get_info(key) if device.supports(key) else ""
                except RuntimeError:
                    item[name] = ""
            cameras.append(item)
        return cameras
