"""RealSense D400/L515 camera driver — streaming, alignment, point clouds."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "RealSense",
    "RealSenseConfig",
    "CameraFrame",
    "L515DepthConfig",
    "AlignMode",
    "_normalize_align_mode",
]

import numpy as np
import pyrealsense2 as rs

from dexmani_real.utils.log import get_logger
from dexmani_real.utils.pointcloud_utils import (
    intrinsics_to_dict,
    intrinsics_to_matrix,
    intrinsics_to_vector,
    make_rays,
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
    """L515-only depth settings, applied via set_option after pipeline start.

    Ten writable L515 options are exposed, plus the read-only depth-offset
    calibration for verification. load_json (XU control path) fails silently
    on the stock uvcvideo kernel, so set_option is the sole config path.

    Tuning strategy (2026-08-07): Short Range preset (5) for the close-range
    dexterous-manipulation workspace (0.25-0.85 m).  The factory Short Range
    parameter table optimises MEMS timing, noise estimation, and confidence
    mapping for < 1 m — then we override only what needs fine-tuning on top.
    D400 cameras ignore this config.
    """

    enabled: bool = True

    # --- visual preset (applied FIRST — loads the factory base table) ---
    visual_preset: int = 5  # L500: 5 = Short Range (< 1 m); was 3 = Low Ambient

    # --- explicit overrides (applied AFTER preset, flips label to 0=Custom) ---
    laser_power: int = 100  # 0-100, full power (MEMS eye-safe at all levels)
    receiver_gain: int = 9  # 8-18; numerically higher = *lower* actual gain.
    # At close range the reflected signal is strong — lower gain (higher number)
    # reduces shot noise with plenty of margin on dark/absorbing surfaces.
    confidence_threshold: int = 1  # 0-3 firmware confidence cull; 1 = keep more
    # pixels (thin fingertip structures survive).  Short Range preset already
    # has a tighter confidence mapping than Low Ambient.
    noise_filtering: int = 1  # 0-6; 1 = light temporal smoothing.  Short Range's
    # native noise floor is ~2-3× lower at 0.5 m than Low Ambient, so we can
    # back off the filter and preserve fine edge detail.
    min_distance: int = 150  # mm; was 190.  15 cm gives headroom for the hand
    # operating close to the camera without clipping valid near-field depth.

    # --- gains & sharpening (newly exposed — were at hardware defaults) ---
    digital_gain: int = 1  # 1-2; post-ADC digital amplification.  1 = no extra
    # gain → less noise amplification.  Strong close-range signal makes the
    # extra 6 dB unnecessary.
    depth_offset: float = 4.5  # mm; expected read-only per-unit calibration.
    # This is verified at startup, not written and not used as a tuning knob.
    post_processing_sharpening: int = 2  # 0-3; edge-enhancement on the firmware
    # depth output.  2 = moderate sharpening — crisper object boundaries without
    # the ringing artefacts that 3 can produce on thin geometry.
    pre_processing_sharpening: int = 0  # 0-5; pre-sharpening amplifies sensor
    # noise before the noise filter runs — keep off.
    noise_estimation: float | None = None  # 0-4100; None = leave at the Short
    # Range preset's factory value.  Explicit override only for custom tuning.


@dataclass(frozen=True)
class RealSenseConfig:
    camera_name: str = "realsense"
    serial: str | None = None
    depth_resolution: tuple[int, int] = (640, 480)
    color_resolution: tuple[int, int] = (640, 480)
    fps: int = 30
    enable_color: bool = True
    align_mode: AlignMode = "depth_to_color"
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


def _normalize_align_mode(mode: str) -> AlignMode:
    key = str(mode).lower()
    if key not in ALIGN_MODE_ALIASES:
        valid = ", ".join(ALIGN_MODE_ALIASES.keys())
        raise ValueError(f"align_mode must be one of: {valid}.")
    return ALIGN_MODE_ALIASES[key]  # type: ignore[return-value]


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

        self.depth_scale: float | None = None
        self.K: np.ndarray | None = None
        self.intr: np.ndarray | None = None
        self.intrinsics_info: dict | None = None
        self.rays: np.ndarray | None = None
        self.frame_id = 0

    def connect(self) -> bool:
        """Open RealSense pipeline. Returns True on success.

        Canonical lifecycle method per CLAUDE.md Section 2.3.
        Idempotent: calling on an already-connected camera returns True.
        """
        if self.pipeline is not None:
            return True

        return self._open_pipeline()

    def _open_pipeline(self) -> bool:
        """Discover device, start + warm up the pipeline.

        Returns True on success. On any failure the pipeline is torn down (via
        disconnect) so a started-but-unusable pipeline is never left holding the
        device — otherwise the next open would hit "Device or resource busy".
        """
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
        """Apply depth settings via set_option at pipeline post-start.

        Device-type dispatch: L515 gets the calibrated preset (visual_preset,
        laser, gain, confidence, noise, min_distance); D400 gets emitter
        enabled and High Accuracy preset.
        """
        if self.profile is None:
            return

        if self.active_is_l515:
            self._apply_l515_depth_config()
        else:
            self._apply_d400_depth_config()

    def _apply_l515_depth_config(self) -> None:
        """Apply L515 depth settings via set_option after pipeline start.

        The only config path: load_json (XU controls) fails silently on the
        stock uvcvideo kernel — no error, but hardware reads back preset
        defaults. set_option is reliable. Order matters: visual_preset first
        (loads the factory base parameter set, including internals with no
        corresponding rs.option), then the explicit overrides — which flips
        the preset label to 0 (Custom); that is the expected final state.
        """
        cfg = self.config.l515_depth_config
        if cfg is None or not cfg.enabled:
            return

        if self.profile is None:
            logger.warning("RealSense: profile is None — cannot apply depth config")
            return
        sensor = self.profile.get_device().first_depth_sensor()

        options: list[tuple[rs.option, float]] = [
            (rs.option.visual_preset, float(cfg.visual_preset)),
            (rs.option.laser_power, float(cfg.laser_power)),
            (rs.option.receiver_gain, float(cfg.receiver_gain)),
            (rs.option.confidence_threshold, float(cfg.confidence_threshold)),
            (rs.option.noise_filtering, float(cfg.noise_filtering)),
            (rs.option.min_distance, float(cfg.min_distance)),
            (rs.option.digital_gain, float(cfg.digital_gain)),
            (rs.option.post_processing_sharpening, float(cfg.post_processing_sharpening)),
            (rs.option.pre_processing_sharpening, float(cfg.pre_processing_sharpening)),
        ]
        if cfg.noise_estimation is not None:
            options.append((rs.option.noise_estimation, float(cfg.noise_estimation)))

        for option, value in options:
            try:
                if sensor.supports(option):
                    sensor.set_option(option, value)
            except (RuntimeError, OSError) as error:
                logger.warning("L515 set_option(%s) failed: %s", option, error)

        # Verify with a read-back sentinel. The preset label itself always
        # flips to 0 (Custom) once individual options are overridden, so
        # receiver_gain is checked instead.
        time.sleep(0.5)
        actual_gain = float(sensor.get_option(rs.option.receiver_gain))
        # depth_offset is a per-unit calibration constant and read-only on the
        # connected L515. Verify/read it instead of issuing a failing setter.
        actual_depth_offset = (
            float(sensor.get_option(rs.option.depth_offset))
            if sensor.supports(rs.option.depth_offset)
            else float("nan")
        )
        logger.info(
            "L515 depth config applied (set_option): preset_base=%d, laser=%d, "
            "gain=%d, conf=%d, noise=%d, min_dist=%d, digital_gain=%d, "
            "sharpening(post=%d, pre=%d), depth_offset_readback=%.1f, noise_est=%s",
            int(cfg.visual_preset),
            int(cfg.laser_power),
            int(actual_gain),
            int(cfg.confidence_threshold),
            int(cfg.noise_filtering),
            int(cfg.min_distance),
            int(cfg.digital_gain),
            int(cfg.post_processing_sharpening),
            int(cfg.pre_processing_sharpening),
            actual_depth_offset,
            "preset" if cfg.noise_estimation is None else str(cfg.noise_estimation),
        )
        if not np.isclose(actual_gain, float(cfg.receiver_gain), atol=1e-6):
            logger.warning(
                "L515 receiver_gain read-back mismatch: requested=%d, actual=%.0f.",
                int(cfg.receiver_gain),
                actual_gain,
            )
        if np.isfinite(actual_depth_offset) and not np.isclose(actual_depth_offset, float(cfg.depth_offset), atol=1e-6):
            logger.warning(
                "L515 read-only depth_offset mismatch: expected=%.1f, actual=%.1f",
                float(cfg.depth_offset),
                actual_depth_offset,
            )

    def _setup_pipeline_post_start(self) -> None:
        """Configure active-device options, alignment, filters, and intrinsics."""
        if self.profile is None:
            raise RuntimeError("Pipeline profile is unavailable after start.")

        self._apply_depth_config()
        self.aligner = self.create_aligner()

        self.set_global_time()
        self.depth_scale = float(self.profile.get_device().first_depth_sensor().get_depth_scale())
        self.update_intrinsics_from_profile()
        self.frame_id = 0

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
            # RS2_RS400_VISUAL_PRESET_HIGH_ACCURACY = 3
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

        Uses its own internal context (rs.pipeline() without argument) — sharing
        the discovery context via rs.pipeline(self.context) causes "Couldn't
        resolve requests" on L515 when the context still holds device handles
        from the earlier query_devices() calls.
        """
        self.pipeline = rs.pipeline()
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
                    "L515 warmup timed out; restarting pipeline after %.0f s " "(attempt %d/%d).",
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
            self.depth_scale = None
            self.K = None
            self.intr = None
            self.intrinsics_info = None

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
            # Unit rays precomputed once per intrinsics change (edge-LUT
            # pattern): per-frame deprojection is then a single multiply,
            # points_cam = rays * depth_m[..., None].
            self.rays = make_rays(int(intrinsics.height), int(intrinsics.width), K).numpy()

    def get_rays(self) -> np.ndarray:
        """(H, W, 3) float32 unit rays matching the output frame geometry."""
        if self.rays is None:
            raise RuntimeError("RealSense is not connected or intrinsics are unavailable.")
        return self.rays

    def read(self, timeout_ms: int = 5000, *, compute_depth: bool = True) -> CameraFrame:
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

        self.update_intrinsics_from_depth_frame(depth_frame)
        if self.K is None or self.intr is None or self.intrinsics_info is None:
            raise RuntimeError("RealSense intrinsics are unavailable.")

        depth_raw = np.ascontiguousarray(np.asanyarray(depth_frame.get_data()))
        if compute_depth:
            depth = depth_raw.astype(np.float32) * float(self.depth_scale)
        else:
            depth = depth_raw  # skip float32 allocation (~1.2 MB/frame); SHM path only needs raw

        rgb = None
        if color_frame is not None:
            bgr = np.asanyarray(color_frame.get_data())
            rgb = np.ascontiguousarray(bgr[..., ::-1])

        # Increment frame_id on every successful read.
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

        return frame

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
