"""RealSense D400/L515 camera driver — streaming, alignment, point clouds."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "RealSense",
    "RealSenseConfig",
    "CameraFrame",
    "L515DepthConfig",
    "AlignMode",
    "remove_l515_mixed_edge_depth",
    "_normalize_align_mode",
]

import numpy as np
import pyrealsense2 as rs

from dexmani_real.utils.log import get_logger
from dexmani_real.utils.pointcloud_utils import (
    PointCloudConfig,
    depth_valid_ratio,
    intrinsics_to_dict,
    intrinsics_to_matrix,
    intrinsics_to_vector,
    make_depth_vis,
    make_rays,
    rgbd_to_pointcloud,
    vis_point_cloud,
)

logger = get_logger(__name__)


AlignMode = Literal["depth_to_color", "color_to_depth", "none"]
ALIGN_MODE_ALIASES = {
    "depth_to_color": "depth_to_color",
    "color_to_depth": "color_to_depth",
    "none": "none",
    "color": "depth_to_color",
    "depth": "color_to_depth",
}


@dataclass(frozen=True)
class L515DepthConfig:
    """
    L515-only depth preset converted from the previous exported JSON.

    It is applied through rs.serializable_device(...).load_json(...).
    D400 cameras such as D435/D455 will not use this config.
    """

    enabled: bool = True
    # Full exported JSON mixes Short Range with many hand-tuned values.
    # Keep it disabled by default; the active device receives Short Range
    # again after pipeline.start(), which is the state that actually matters.
    load_json_before_stream: bool = False

    visual_preset: int = 5
    depth_units: float = 0.000250000011874363
    depth_offset: float = 4.5
    min_distance: int = 190

    laser_power: int = 70
    receiver_gain: int = 12
    confidence_threshold: int = 2
    digital_gain: int = 2

    noise_filtering: int = 50
    # rs.option.noise_filtering runtime scale is 0-6 (distinct from the JSON
    # "Noise Filtering" value above); used by the set_option fallback.
    noise_filtering_runtime: int = 3
    noise_estimation: float = 0.0
    pre_processing_sharpening: float = 0.0
    post_processing_sharpening: int = 1

    alternate_ir: float = 0.0
    enable_ir_reflectivity: float = 0.0
    enable_max_usable_range: float = 0.0
    error_polling_enabled: int = 1
    frames_queue_size: int = 16
    freefall_detection_enabled: int = 1
    global_time_enabled: float = 0.0
    host_performance: float = 0.0
    inter_cam_sync_mode: float = 0.0
    invalidation_bypass: float = 0.0
    reset_camera_accuracy_health: float = 0.0
    sensor_mode: float = 0.0
    trigger_camera_accuracy_health: float = 0.0

    def to_json_dict(self, depth_resolution: tuple[int, int], fps: int) -> dict[str, Any]:
        depth_width, depth_height = depth_resolution
        return {
            "Alternate IR": self.alternate_ir,
            "Confidence Threshold": self.confidence_threshold,
            "Depth Offset": self.depth_offset,
            "Depth Units": self.depth_units,
            "Digital Gain": self.digital_gain,
            "Enable IR Reflectivity": self.enable_ir_reflectivity,
            "Enable Max Usable Range": self.enable_max_usable_range,
            "Error Polling Enabled": self.error_polling_enabled,
            "Frames Queue Size": self.frames_queue_size,
            "Freefall Detection Enabled": self.freefall_detection_enabled,
            "Global Time Enabled": self.global_time_enabled,
            "Host Performance": self.host_performance,
            "Inter Cam Sync Mode": self.inter_cam_sync_mode,
            "Invalidation Bypass": self.invalidation_bypass,
            "Laser Power": self.laser_power,
            "Min Distance": self.min_distance,
            "Noise Estimation": self.noise_estimation,
            "Noise Filtering": self.noise_filtering,
            "Post Processing Sharpening": self.post_processing_sharpening,
            "Pre Processing Sharpening": self.pre_processing_sharpening,
            "Receiver Gain": self.receiver_gain,
            "Reset Camera Accuracy Health": self.reset_camera_accuracy_health,
            "Sensor Mode": self.sensor_mode,
            "Trigger Camera Accuracy Health": self.trigger_camera_accuracy_health,
            "Visual Preset": self.visual_preset,
            "stream-depth-format": "Z16",
            "stream-fps": str(int(fps)),
            "stream-height": str(int(depth_height)),
            "stream-width": str(int(depth_width)),
        }

    def to_json_string(self, depth_resolution: tuple[int, int], fps: int) -> str:
        return json.dumps(self.to_json_dict(depth_resolution, fps))


def apply_l515_depth_config(
    device: rs.device,
    l515_config: L515DepthConfig | None,
    depth_resolution: tuple[int, int],
    fps: int,
) -> str:
    """Apply the L515 depth preset to a device (works before or after streaming).

    Tries the full JSON preset via ``serializable_device.load_json`` first. On
    hosts whose kernel uvcvideo backend rejects L515 XU controls ("Device or
    resource busy"), load_json and visual_preset are unavailable — fall back to
    the depth options that DO work over plain UVC (confidence_threshold,
    min_distance, noise_filtering, laser_power, receiver_gain). See
    docs/l515_backend_preset_fix.md.

    Returns "json", "fallback", "failed", or "disabled".
    """
    if l515_config is None or not l515_config.enabled:
        return "disabled"

    try:
        rs.serializable_device(device).load_json(l515_config.to_json_string(depth_resolution, fps))
        return "json"
    except (RuntimeError, OSError) as error:
        logger.warning(
            "L515 preset (load_json) not applied (%s); applying the UVC-safe subset "
            "via set_option. Host UVC limitation, not a config error — see "
            "docs/l515_backend_preset_fix.md.",
            error,
        )

    # Fallback: options that work over plain UVC. noise_filtering uses the runtime
    # 0-6 scale (noise_filtering_runtime), not the JSON value. visual_preset and
    # digital_gain need the (blocked/flaky) XU path and are not retried here.
    try:
        depth_sensor = device.first_depth_sensor()
    except (RuntimeError, OSError) as error:
        logger.warning("L515 set_option fallback skipped (no depth sensor): %s", error)
        return "failed"

    applied = False
    for option, value in (
        (rs.option.confidence_threshold, l515_config.confidence_threshold),
        (rs.option.min_distance, l515_config.min_distance),
        (rs.option.noise_filtering, l515_config.noise_filtering_runtime),
        (rs.option.laser_power, l515_config.laser_power),
        (rs.option.receiver_gain, l515_config.receiver_gain),
    ):
        try:
            if depth_sensor.supports(option):
                depth_sensor.set_option(option, float(value))
                applied = True
        except (RuntimeError, OSError) as error:
            logger.warning("L515 set_option(%s) failed: %s", option, error)
    return "fallback" if applied else "failed"


@dataclass(frozen=True)
class RealSenseConfig:
    camera_name: str = "realsense"
    serial: str | None = None
    depth_resolution: tuple[int, int] = (640, 480)
    color_resolution: tuple[int, int] = (640, 480)
    fps: int = 30
    enable_color: bool = True
    align_mode: AlignMode = "depth_to_color"
    depth_hole_filling: bool = False
    enable_sdk_spatial_filter: bool = False
    # Conservative L515 edge-ramp filter applied only for point-cloud generation.
    # It removes intermediate depths between a local foreground/background pair,
    # while preserving both actual surfaces. No hole filling is performed.
    enable_l515_flying_pixel_filter: bool = True
    l515_edge_jump_threshold_m: float = 0.020
    l515_edge_surface_margin_m: float = 0.004
    l515_edge_filter_radius: int = 2
    l515_edge_min_valid_neighbors: int = 6
    # Safety valve: if the 2-D filter would remove too much valid depth, reject
    # its output and use the original aligned depth for this frame.
    l515_edge_max_removed_ratio: float = 0.08
    enable_global_time: bool = True
    warmup_frames: int = 10
    frame_name: str | None = None
    l515_depth_config: L515DepthConfig | None = field(default_factory=L515DepthConfig)

    def __post_init__(self) -> None:
        # object.__setattr__ bypasses frozen=True in __post_init__ so we can
        # normalize/normalize assign immutable fields after construction.
        mode = _normalize_align_mode(self.align_mode)
        object.__setattr__(self, "align_mode", mode)
        if mode != "none" and not self.enable_color:
            raise ValueError("alignment requires enable_color=True.")
        if self.frame_name is None:
            frame_name = "camera_color_optical" if mode == "depth_to_color" else "camera_depth_optical"
            object.__setattr__(self, "frame_name", frame_name)


@dataclass(frozen=True)
class CameraFrame:
    rgb: np.ndarray | None
    depth: np.ndarray
    depth_raw: np.ndarray
    timestamp: float
    host_time: float
    frame_id: int
    K: np.ndarray
    intr: np.ndarray
    intrinsics_info: dict
    depth_scale: float
    camera_name: str
    serial: str | None
    align_mode: AlignMode
    frame_name: str

    def to_dict(self) -> dict:
        return {
            "rgb": self.rgb,
            "depth": self.depth,
            "depth_raw": self.depth_raw,
            "timestamp": self.timestamp,
            "host_time": self.host_time,
            "frame_id": self.frame_id,
            "K": self.K,
            "intr": self.intr,
            "intrinsics_info": self.intrinsics_info,
            "depth_scale": self.depth_scale,
            "camera_name": self.camera_name,
            "serial": self.serial,
            "align_mode": self.align_mode,
            "frame_name": self.frame_name,
            "meta": {
                "rgb_order": "RGB",
                "rgb_dtype": "uint8" if self.rgb is not None else None,
                "depth_unit": "m",
                "depth_dtype": "float32",
                "depth_raw_unit": "raw_z16",
                "depth_raw_dtype": "uint16",
            },
        }


def _normalize_align_mode(mode: str) -> AlignMode:
    key = str(mode).lower()
    if key not in ALIGN_MODE_ALIASES:
        valid = ", ".join(ALIGN_MODE_ALIASES.keys())
        raise ValueError(f"align_mode must be one of: {valid}.")
    return ALIGN_MODE_ALIASES[key]  # type: ignore[return-value]


def remove_l515_mixed_edge_depth(
    depth_m: np.ndarray,
    *,
    jump_threshold_m: float = 0.020,
    surface_margin_m: float = 0.004,
    radius: int = 2,
    min_valid_neighbors: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove L515 mixed-depth ramps at object boundaries.

    A true foreground/background edge contains two valid surfaces. L515 flying
    pixels typically occupy intermediate depths between those surfaces. This
    filter keeps values close to the local minimum or maximum and invalidates
    only intermediate values. It is intentionally conservative and suitable
    for depth already aligned to RGB.

    Returns:
        filtered_depth_m: float32 depth in metres; rejected pixels are zero.
        removed_mask: boolean mask of pixels invalidated by this filter.
    """
    if radius < 1:
        raise ValueError("radius must be >= 1")
    if jump_threshold_m <= 0:
        raise ValueError("jump_threshold_m must be > 0")
    if surface_margin_m < 0:
        raise ValueError("surface_margin_m must be >= 0")
    if 2.0 * surface_margin_m >= jump_threshold_m:
        raise ValueError(
            "surface_margin_m must be less than half jump_threshold_m"
        )

    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "OpenCV is required for L515 mixed-edge filtering."
        ) from error

    depth = np.ascontiguousarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return depth.copy(), np.zeros(depth.shape, dtype=bool)

    ksize = 2 * int(radius) + 1
    kernel = np.ones((ksize, ksize), dtype=np.uint8)

    # Invalid pixels must not become local foreground/background surfaces.
    min_input = np.where(valid, depth, np.float32(1e6))
    max_input = np.where(valid, depth, np.float32(-1e6))
    local_min = cv2.erode(min_input, kernel, borderType=cv2.BORDER_REPLICATE)
    local_max = cv2.dilate(max_input, kernel, borderType=cv2.BORDER_REPLICATE)

    valid_count = cv2.boxFilter(
        valid.astype(np.float32),
        ddepth=-1,
        ksize=(ksize, ksize),
        normalize=False,
        borderType=cv2.BORDER_CONSTANT,
    )

    local_span = local_max - local_min
    removed_mask = (
        valid
        & (valid_count >= float(min_valid_neighbors))
        & (local_span > float(jump_threshold_m))
        & (depth > local_min + float(surface_margin_m))
        & (depth < local_max - float(surface_margin_m))
    )

    filtered = depth.copy()
    filtered[removed_mask] = 0.0
    return filtered, removed_mask


class RealSense:
    def __init__(self, config: RealSenseConfig = RealSenseConfig()) -> None:
        self.config = config
        # Device discovery, option access, and streaming must share one context.
        # This avoids competing device handles and intermittent power-state errors.
        self.context = rs.context()
        self.active_serial: str | None = config.serial
        self.active_is_l515 = False

        self.pipeline: rs.pipeline | None = None
        self.profile: rs.pipeline_profile | None = None
        self.aligner: rs.align | None = None
        self.hole_filling_filter: rs.hole_filling_filter | None = None
        self.to_disparity: rs.disparity_transform | None = None
        self.spatial_filter: rs.spatial_filter | None = None
        self.to_depth: rs.disparity_transform | None = None

        self.depth_scale: float | None = None
        self.K: np.ndarray | None = None
        self.intr: np.ndarray | None = None
        self.intrinsics_info: dict | None = None
        self.rays_cache: dict[tuple[int, int, str], Any] = {}

        self.frame_id = 0
        self.last_frame: CameraFrame | None = None

    def connect(self) -> bool:
        """Open RealSense pipeline. Returns True on success.

        Canonical lifecycle method per CLAUDE.md Section 2.3.
        Idempotent: calling on an already-connected camera returns True.

        On L515 a started pipeline can still fail to expose depth-stream
        intrinsics (VGA/XGA), which makes rs.align throw on every frame. Only a
        hardware_reset() reloads them — pipeline stop/start does not — so after
        opening we verify the depth intrinsics and self-heal with a one-shot
        reset if they are missing.
        """
        if self.pipeline is not None:
            return True

        if not self._open_pipeline():
            return False
        if self._depth_intrinsics_available():
            return True

        logger.warning(
            "Depth stream exposes no intrinsics (L515 bad state) — "
            "hardware_reset() and reconnecting once."
        )
        self._hardware_reset_and_wait()
        if not self._open_pipeline():
            return False
        if not self._depth_intrinsics_available():
            logger.error("Depth intrinsics still missing after hardware_reset — giving up.")
            self.disconnect()
            return False
        return True

    def _open_pipeline(self) -> bool:
        """Discover device, apply model config, start + warm up the pipeline.

        Returns True on success. On any failure the pipeline is torn down (via
        disconnect) so a started-but-unusable pipeline is never left holding the
        device — otherwise the next open would hit "Device or resource busy".
        """
        try:
            self.active_serial = self.config.serial or self._find_default_serial_in_context()
            device = self._find_device_by_serial_in_context(self.active_serial)
            self.active_is_l515 = self.is_l515_device(device)
            self.load_model_specific_config(device)
        except (RuntimeError, OSError) as e:
            logger.warning("connect() failed at device discovery / config: %s", e)
            return False

        rs_config = self.create_rs_config()
        try:
            self._start_pipeline(rs_config)
            self._setup_pipeline_post_start()
            self._warmup_pipeline()
        except (RuntimeError, OSError) as e:
            logger.warning("connect() failed at pipeline start / warmup: %s", e)
            self.disconnect()
            return False
        return True

    def _depth_intrinsics_available(self) -> bool:
        """Whether the started depth stream actually exposes intrinsics.

        The L515 can stream depth yet report "intrinsics for resolution W,H
        doesn't exist" for VGA/XGA, which breaks rs.align. The calibration is
        intact; a hardware_reset() reloads it.
        """
        if self.profile is None:
            return False
        try:
            self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
            return True
        except RuntimeError:
            return False

    def _hardware_reset_and_wait(self, settle_s: float = 12.0) -> None:
        """Reset the device and wait for it to re-enumerate.

        Only a hardware_reset reloads L515 depth intrinsics; pipeline stop/start
        does not. Captures the device handle before tearing down the pipeline.
        """
        device = None
        if self.profile is not None:
            try:
                device = self.profile.get_device()
            except RuntimeError:
                device = None
        self.disconnect()
        if device is None and self.active_serial:
            try:
                device = self._find_device_by_serial_in_context(self.active_serial)
            except RuntimeError:
                device = None
        if device is not None:
            try:
                device.hardware_reset()
                logger.info("hardware_reset() issued; waiting %.0fs for re-enumeration.", settle_s)
            except RuntimeError as e:
                logger.warning("hardware_reset() failed: %s", e)
        time.sleep(settle_s)
        # Use a fresh context after USB re-enumeration.
        self.context = rs.context()

    def _apply_l515_runtime_preset(self) -> None:
        """Set and verify Short Range on the active streaming device.

        L515 may report that a pre-start JSON load succeeded and still reset the
        visual preset during ``pipeline.start()``. Therefore the final preset is
        always applied after start through the active pipeline profile.
        """
        if (
            not self.active_is_l515
            or self.profile is None
            or self.config.l515_depth_config is None
            or not self.config.l515_depth_config.enabled
        ):
            return

        sensor = self.profile.get_device().first_depth_sensor()
        option = rs.option.visual_preset
        target = float(self.config.l515_depth_config.visual_preset)
        if not sensor.supports(option):
            raise RuntimeError("L515 depth sensor does not expose visual_preset.")

        sensor.set_option(option, target)
        time.sleep(0.5)
        actual = float(sensor.get_option(option))
        try:
            description = sensor.get_option_value_description(option, actual)
        except RuntimeError:
            description = "unknown"

        if not np.isclose(actual, target, atol=1e-6):
            raise RuntimeError(
                "L515 runtime preset verification failed: "
                f"requested={target}, actual={actual} ({description})."
            )

        logger.info("L515 runtime preset verified: %.0f (%s)", actual, description)

    def _setup_pipeline_post_start(self) -> None:
        """Configure active-device options, alignment, filters, and intrinsics."""
        if self.profile is None:
            raise RuntimeError("Pipeline profile is unavailable after start.")

        self._apply_l515_runtime_preset()
        self.aligner = self.create_aligner()
        self.hole_filling_filter = rs.hole_filling_filter(2) if self.config.depth_hole_filling else None

        # Optional SDK spatial filter chain (disparity domain → edge-preserving).
        # Attenuates L515 edge flying pixels at negligible CPU cost (C++ SDK).
        if self.config.enable_sdk_spatial_filter:
            self.to_disparity = rs.disparity_transform(True)
            self.spatial_filter = rs.spatial_filter()
            self.spatial_filter.set_option(rs.option.filter_magnitude, 2)
            self.spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
            self.spatial_filter.set_option(rs.option.filter_smooth_delta, 20)
            self.spatial_filter.set_option(rs.option.holes_fill, 0)  # never invent depth
            self.to_depth = rs.disparity_transform(False)
        else:
            self.to_disparity = None
            self.spatial_filter = None
            self.to_depth = None

        self.set_global_time()
        self.depth_scale = float(self.profile.get_device().first_depth_sensor().get_depth_scale())
        self.update_intrinsics_from_profile()
        self.frame_id = 0
        self.last_frame = None
        self.rays_cache.clear()

    def _start_pipeline(self, rs_config: rs.config) -> None:
        """Create and start a fresh pipeline with the shared context."""
        self.pipeline = rs.pipeline(self.context)
        rs_config.resolve(rs.pipeline_wrapper(self.pipeline))
        self.profile = self.pipeline.start(rs_config)

    def _warmup_pipeline(self) -> None:
        """Consume exactly ``warmup_frames`` frames, restarting if necessary."""
        if self.pipeline is None:
            raise RuntimeError("Pipeline is unavailable during warmup.")

        warmup_frames = max(int(self.config.warmup_frames), 0)
        if warmup_frames == 0:
            return

        max_restarts = 3
        for attempt in range(max_restarts + 1):
            try:
                for _ in range(warmup_frames):
                    self.pipeline.wait_for_frames(5000)
                return
            except RuntimeError:
                if attempt >= max_restarts:
                    raise

                delay = 3.0 * (attempt + 1)
                logger.warning(
                    "L515 warmup timed out; restarting pipeline after %.0f s "
                    "(attempt %d/%d).",
                    delay,
                    attempt + 1,
                    max_restarts,
                )
                try:
                    self.pipeline.stop()
                except RuntimeError:
                    pass
                time.sleep(delay)
                self._start_pipeline(self.create_rs_config())
                # Reapply Short Range and rebuild active-profile state after every restart.
                self._setup_pipeline_post_start()
                time.sleep(1.0)

    def disconnect(self) -> None:
        """Close RealSense pipeline.

        Canonical lifecycle method per CLAUDE.md Section 2.3.
        Idempotent: calling on an already-disconnected camera is a no-op.
        """
        if self.pipeline is None:
            return
        try:
            self.pipeline.stop()
        except RuntimeError:
            pass
        finally:
            self.pipeline = None
            self.profile = None
            self.aligner = None
            self.hole_filling_filter = None
            self.depth_scale = None
            self.K = None
            self.intr = None
            self.intrinsics_info = None
            self.rays_cache.clear()
            self.last_frame = None

    def __enter__(self) -> "RealSense":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.disconnect()

    def create_rs_config(self) -> rs.config:
        depth_width, depth_height = self.config.depth_resolution
        color_width, color_height = self.config.color_resolution

        rs_config = rs.config()
        rs_config.enable_device(self.active_serial)
        rs_config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, self.config.fps)
        if self.config.enable_color:
            rs_config.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, self.config.fps)
        return rs_config

    def create_aligner(self) -> rs.align | None:
        if self.config.align_mode == "none":
            return None
        target = rs.stream.color if self.config.align_mode == "depth_to_color" else rs.stream.depth
        return rs.align(target)

    def set_global_time(self) -> None:
        if not self.config.enable_global_time or self.profile is None:
            return
        for sensor in self.profile.get_device().query_sensors():
            try:
                if sensor.supports(rs.option.global_time_enabled):
                    sensor.set_option(rs.option.global_time_enabled, 1)
            except RuntimeError:
                pass

    def load_model_specific_config(self, device: rs.device) -> None:
        if not self.is_l515_device(device):
            return
        cfg = self.config.l515_depth_config
        if cfg is not None and cfg.enabled and cfg.load_json_before_stream:
            self.load_l515_depth_config(device)

    def load_l515_depth_config(self, device: rs.device) -> None:
        apply_l515_depth_config(
            device,
            self.config.l515_depth_config,
            depth_resolution=self.config.depth_resolution,
            fps=self.config.fps,
        )

    @staticmethod
    def is_l515_device(device: rs.device) -> bool:
        name = RealSense.get_device_info_value(device, rs.camera_info.name).upper()
        product_line = RealSense.get_device_info_value(device, rs.camera_info.product_line)
        return product_line == "L500" or "L515" in name

    def update_intrinsics_from_profile(self) -> None:
        if self.profile is None:
            raise RuntimeError("RealSense is not connected.")
        stream = rs.stream.color if self.config.align_mode == "depth_to_color" else rs.stream.depth
        video_profile = self.profile.get_stream(stream).as_video_stream_profile()
        self.set_intrinsics(video_profile.get_intrinsics())

    def update_intrinsics_from_depth_frame(self, depth_frame: rs.depth_frame) -> None:
        video_profile = depth_frame.get_profile().as_video_stream_profile()
        self.set_intrinsics(video_profile.get_intrinsics())

    def set_intrinsics(self, intrinsics: Any) -> None:
        K = intrinsics_to_matrix(intrinsics)
        if self.K is None or not np.allclose(K, self.K):
            self.K = K
            self.intr = intrinsics_to_vector(K)
            self.intrinsics_info = intrinsics_to_dict(intrinsics)
            self.rays_cache.clear()

    def read(self, timeout_ms: int = 5000) -> CameraFrame:
        if self.pipeline is None:
            raise RuntimeError("RealSense is not connected. Call connect() first.")
        if self.depth_scale is None:
            raise RuntimeError("RealSense depth_scale is unavailable.")

        frames = self.pipeline.wait_for_frames(timeout_ms)
        if self.aligner is not None:
            frames = self.aligner.process(frames)

        host_time = time.time()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame() if self.config.enable_color else None
        if not depth_frame:
            raise RuntimeError("Failed to get depth frame.")
        if self.config.enable_color and not color_frame:
            raise RuntimeError("Failed to get color frame.")

        # Optional SDK spatial filter (disparity domain, edge-preserving).
        # Must run *before* hole filling so the hole filler works on
        # already-cleaned depth.
        if self.to_disparity is not None:
            depth_frame = self.to_disparity.process(depth_frame)
            depth_frame = self.spatial_filter.process(depth_frame)  # type: ignore[union-attr]
            depth_frame = self.to_depth.process(depth_frame)
            depth_frame = depth_frame.as_depth_frame()

        if self.hole_filling_filter is not None:
            depth_frame = self.hole_filling_filter.process(depth_frame).as_depth_frame()

        self.update_intrinsics_from_depth_frame(depth_frame)
        if self.K is None or self.intr is None or self.intrinsics_info is None:
            raise RuntimeError("RealSense intrinsics are unavailable.")

        depth_raw = np.ascontiguousarray(np.asanyarray(depth_frame.get_data()))
        depth = depth_raw.astype(np.float32) * float(self.depth_scale)

        rgb = None
        if color_frame is not None:
            bgr = np.asanyarray(color_frame.get_data())
            rgb = np.ascontiguousarray(bgr[..., ::-1])

        self.frame_id += 1
        frame = CameraFrame(
            rgb=rgb,
            depth=depth,
            depth_raw=depth_raw,
            timestamp=float(depth_frame.get_timestamp()) * 1e-3,
            host_time=host_time,
            frame_id=self.frame_id,
            K=self.K.copy(),
            intr=self.intr.copy(),
            intrinsics_info=dict(self.intrinsics_info),
            depth_scale=float(self.depth_scale),
            camera_name=self.config.camera_name,
            serial=self.active_serial,
            align_mode=self.config.align_mode,
            frame_name=str(self.config.frame_name),
        )
        self.last_frame = frame
        return frame

    def get_state(
        self,
        *,
        mode: Literal["rgbd", "pointcloud", "full"] = "full",
        pcd_config: PointCloudConfig | None = None,
        T_out_camera: np.ndarray | None = None,
        timeout_ms: int = 5000,
    ) -> dict:
        if mode not in ("rgbd", "pointcloud", "full"):
            raise ValueError("mode must be one of: 'rgbd', 'pointcloud', 'full'.")

        frame = self.read(timeout_ms=timeout_ms)
        obs = frame.to_dict()
        config = pcd_config or PointCloudConfig()
        obs["meta"].update(
            {
                "valid_depth_ratio": depth_valid_ratio(frame.depth, config.min_depth, config.max_depth),
                "pointcloud_frame": "out" if T_out_camera is not None else frame.frame_name,
            }
        )

        if mode == "pointcloud":
            obs.pop("rgb", None)
            obs.pop("depth", None)
            obs.pop("depth_raw", None)

        if mode in ("pointcloud", "full"):
            pointcloud = self.pointcloud_from_frame(frame, config, T_out_camera=T_out_camera)
            obs["pointcloud"] = pointcloud
            obs["meta"].update(
                {
                    "pointcloud_format": "xyzrgb",
                    "pointcloud_xyz_unit": "m",
                    "pointcloud_rgb_range": [0.0, 1.0],
                    "pointcloud_dtype": "float32",
                    "point_count": int(pointcloud.shape[0]),
                    "workspace": list(config.workspace) if config.workspace is not None else None,
                    "npoints": config.npoints,
                    "sampling": config.sampling,
                    "min_depth": config.min_depth,
                    "max_depth": config.max_depth,
                }
            )
        return obs

    def _filter_l515_depth_for_pointcloud(
        self,
        depth: np.ndarray,
        K: np.ndarray,
    ) -> np.ndarray:
        """Conservatively invalidate L515 mixed-depth edge ramps."""
        del K  # Kept in the signature for API stability and future models.
        if not self.active_is_l515 or not self.config.enable_l515_flying_pixel_filter:
            return depth

        filtered, removed_mask = remove_l515_mixed_edge_depth(
            depth,
            jump_threshold_m=self.config.l515_edge_jump_threshold_m,
            surface_margin_m=self.config.l515_edge_surface_margin_m,
            radius=self.config.l515_edge_filter_radius,
            min_valid_neighbors=self.config.l515_edge_min_valid_neighbors,
        )

        valid_count = int(np.count_nonzero(np.isfinite(depth) & (depth > 0)))
        removed_count = int(np.count_nonzero(removed_mask))
        removed_ratio = removed_count / max(valid_count, 1)

        if removed_ratio > self.config.l515_edge_max_removed_ratio:
            logger.warning(
                "Rejecting L515 edge filter for this frame: removed %.2f%% "
                "of valid depth (limit %.2f%%).",
                100.0 * removed_ratio,
                100.0 * self.config.l515_edge_max_removed_ratio,
            )
            return depth

        if removed_count > 0:
            logger.debug(
                "L515 edge filter removed %d/%d valid pixels (%.2f%%).",
                removed_count,
                valid_count,
                100.0 * removed_ratio,
            )
        return np.ascontiguousarray(filtered, dtype=np.float32)

    def pointcloud_from_frame(
        self,
        frame: CameraFrame,
        config: PointCloudConfig | None = None,
        *,
        T_out_camera: np.ndarray | None = None,
    ) -> np.ndarray:
        depth_for_pointcloud = self._filter_l515_depth_for_pointcloud(frame.depth, frame.K)
        return rgbd_to_pointcloud(
            depth=depth_for_pointcloud,
            K=frame.K,
            rgb=frame.rgb if frame.align_mode != "none" else None,
            config=config or PointCloudConfig(),
            T_out_camera=T_out_camera,
        )

    def get_intrinsics(self) -> np.ndarray:
        if self.K is None:
            raise RuntimeError("RealSense is not connected or intrinsics are unavailable.")
        return self.K.copy()

    def get_intrinsics_info(self) -> dict:
        if self.intrinsics_info is None:
            raise RuntimeError("RealSense is not connected or intrinsics info is unavailable.")
        return dict(self.intrinsics_info)

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

    def get_rays(self, shape: tuple[int, int], device: str = "cpu") -> np.ndarray:
        if self.K is None:
            raise RuntimeError("RealSense is not connected or intrinsics are unavailable.")
        height, width = int(shape[0]), int(shape[1])
        key = (height, width, str(device))
        if key not in self.rays_cache:
            self.rays_cache[key] = make_rays(height, width, self.K, device=device)
        return self.rays_cache[key]

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
        serial = self.get_device_info_value(devices[0], rs.camera_info.serial_number)
        if not serial:
            raise RuntimeError("The first RealSense camera does not expose a serial number.")
        return serial

    @staticmethod
    def find_device_by_serial(serial: str) -> rs.device:
        """Backward-compatible one-shot device lookup using a temporary context."""
        context = rs.context()
        for device in context.query_devices():
            device_serial = RealSense.get_device_info_value(device, rs.camera_info.serial_number)
            if device_serial == serial:
                return device
        raise RuntimeError(f"No RealSense camera found with serial={serial}.")

    def find_default_serial(self) -> str:
        """Return the first serial visible to this camera's shared context."""
        return self._find_default_serial_in_context()

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

