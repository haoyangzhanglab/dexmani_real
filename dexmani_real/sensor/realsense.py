"""RealSense D400/L515 camera driver — streaming, alignment, point clouds."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
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
    DepthValidityConfig,
    build_edge_threshold_lut,
    compute_depth_edge_mask,
    compute_depth_valid_mask,
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

    Exactly the subset settable over plain UVC. load_json (XU control path)
    fails silently on the stock uvcvideo kernel — no error, hardware reads
    back preset defaults — so it is not used. JSON-only parameters
    (digital_gain, depth_units, depth_offset, ...) stay at hardware defaults.
    Defaults below were validated on SN f1382055 (2026-07-15): Low Ambient
    base + conf=2 + gain=12 -> 95% valid depth. D400 cameras ignore this.
    """

    enabled: bool = True
    visual_preset: int = 3  # L500 runtime enum: 3 = Low Ambient Light (base)
    min_distance: int = 190  # mm
    laser_power: int = 100
    receiver_gain: int = 12  # 8-18; numerically higher = lower actual gain
    confidence_threshold: int = 2  # firmware confidence cull, 0-3
    noise_filtering: int = 2  # runtime scale 0-6


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
    # L515-only image-domain depth validity gate (confidence + IR). When set, the
    # confidence and IR streams are enabled alongside depth, and invalid raw-depth
    # pixels are zeroed BEFORE alignment (confidence/IR are registered to the raw
    # depth frame; rs.align only warps depth). None = disabled (no extra streams).
    depth_validity: DepthValidityConfig | None = None

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
        self.last_frame: CameraFrame | None = None
        self._validity_warned = False
        # Runtime copy of depth_validity with confidence_min pre-shifted into the
        # RAW8 upper-nibble domain: the SDK unpacks CNF4 there with a zero lower
        # nibble, so comparing raw bytes against (min << 4) is exactly equivalent
        # to unpacking and saves a full-frame shift per read.
        validity = config.depth_validity
        if validity is not None and validity.confidence_min is not None:
            validity = replace(validity, confidence_min=validity.confidence_min << 4)
        self._validity_rt = validity
        self._edge_t_lut: np.ndarray | None = None
        # Per-instance scratch buffers for compute_depth_edge_mask (avoids
        # per-frame temp allocations; populated lazily on first frame).
        self._edge_scratch: dict = {}

    def connect(self) -> bool:
        """Open RealSense pipeline. Returns True on success.

        Canonical lifecycle method per CLAUDE.md Section 2.3.
        Idempotent: calling on an already-connected camera returns True.

        On L515 two classes of bad device state are self-healed with a one-shot
        hardware reset:

        1. Depth stream exposes no intrinsics (VGA/XGA) — rs.align throws on
           every frame. Pipeline stop/start does not reload them.
        2. Pipeline start fails — the device stream engine may be stuck.
        """
        if self.pipeline is not None:
            return True

        if self._open_pipeline():
            if self._depth_intrinsics_available():
                return True
            logger.warning(
                "Depth stream exposes no intrinsics (L515 bad state) — "
                "hardware_reset() and reconnecting once."
            )
        elif self.active_is_l515:
            logger.warning(
                "Pipeline start failed on L515 — hardware_reset() and retrying once."
            )
        else:
            return False

        self._hardware_reset_and_wait()
        if not self._open_pipeline():
            return False
        if not self._depth_intrinsics_available():
            logger.error("Depth intrinsics still missing after hardware_reset — giving up.")
            self.disconnect()
            return False
        return True

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
        if (
            not self.active_is_l515
            or self.profile is None
            or cfg is None
            or not cfg.enabled
        ):
            return

        sensor = self.profile.get_device().first_depth_sensor()

        options: list[tuple[rs.option, float]] = [
            (rs.option.visual_preset, float(cfg.visual_preset)),
            (rs.option.laser_power, float(cfg.laser_power)),
            (rs.option.receiver_gain, float(cfg.receiver_gain)),
            (rs.option.confidence_threshold, float(cfg.confidence_threshold)),
            (rs.option.min_distance, float(cfg.min_distance)),
            (rs.option.noise_filtering, float(cfg.noise_filtering)),
        ]

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
        logger.info(
            "L515 depth config applied (set_option): preset_base=%d, laser=%d, "
            "gain=%d, conf=%d, noise=%d, min_dist=%d",
            int(cfg.visual_preset),
            int(cfg.laser_power),
            int(actual_gain),
            int(cfg.confidence_threshold),
            int(cfg.noise_filtering),
            int(cfg.min_distance),
        )
        if not np.isclose(actual_gain, float(cfg.receiver_gain), atol=1e-6):
            logger.warning(
                "L515 receiver_gain read-back mismatch: requested=%d, actual=%.0f.",
                int(cfg.receiver_gain),
                actual_gain,
            )

    def _setup_pipeline_post_start(self) -> None:
        """Configure active-device options, alignment, filters, and intrinsics."""
        if self.profile is None:
            raise RuntimeError("Pipeline profile is unavailable after start.")

        self._apply_l515_depth_config()
        self.aligner = self.create_aligner()

        self.set_global_time()
        self.depth_scale = float(self.profile.get_device().first_depth_sensor().get_depth_scale())
        self.update_intrinsics_from_profile()
        if self._validity_rt is not None and self._validity_rt.edge is not None:
            # Rebuilt on every (re)start — depth_scale is only known here.
            self._edge_t_lut = build_edge_threshold_lut(self.depth_scale, self._validity_rt.edge)
        self.frame_id = 0
        self.last_frame = None

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
            self.depth_scale = None
            self.K = None
            self.intr = None
            self.intrinsics_info = None
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
        if self.active_is_l515 and self.config.depth_validity is not None:
            validity = self.config.depth_validity
            # Confidence/IR are pixel-registered to the raw depth stream (same sensor);
            # only pull the streams the configured sub-checks actually need.
            if validity.confidence_min is not None:
                rs_config.enable_stream(rs.stream.confidence, depth_width, depth_height, rs.format.raw8, self.config.fps)
            if validity.ir_min is not None or validity.ir_saturation is not None:
                rs_config.enable_stream(rs.stream.infrared, depth_width, depth_height, rs.format.y8, self.config.fps)
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

    def _apply_depth_validity(self, frames: rs.composite_frame) -> None:
        """Zero invalid raw-depth pixels (confidence/IR/edge gate) BEFORE alignment.

        L515 confidence/IR frames are pixel-registered to the raw depth frame and
        rs.align only warps depth, so the whole gate must run pre-align — zeroed
        pixels do not project through alignment. The discontinuity gate runs after
        the confidence/IR zeroing, so rejected pixels are already excluded from
        its neighbourhoods.
        """
        validity = self._validity_rt
        if validity is None or not self.active_is_l515:
            return
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            return

        depth = np.asanyarray(depth_frame.get_data())
        if not depth.flags.writeable:
            self._warn_validity_once("depth frame buffer is not writable — validity gate skipped.")
            return

        confidence = self._get_stream_data(frames, rs.stream.confidence, depth.shape)
        ir = self._get_stream_data(frames, rs.stream.infrared, depth.shape)
        if confidence is not None or ir is not None:
            valid = compute_depth_valid_mask(depth, confidence=confidence, ir=ir, config=validity)
            # In-place multiply by the bool mask instead of fancy-index zeroing
            # (~50x faster on XGA: vectorized multiply vs scatter write).
            np.multiply(depth, valid, out=depth, casting="unsafe")

        if validity.edge is not None and self._edge_t_lut is not None:
            edge = compute_depth_edge_mask(
                depth,
                self._edge_t_lut,
                dilate_px=validity.edge.dilate_px,
                scratch=self._edge_scratch,
            )
            np.multiply(depth, ~edge, out=depth, casting="unsafe")

    def _get_stream_data(
        self, frames: rs.composite_frame, stream: rs.stream, depth_shape: tuple[int, ...]
    ) -> np.ndarray | None:
        frame = frames.first_or_default(stream)
        if not frame:
            return None
        data = np.asanyarray(frame.get_data())
        if data.shape != depth_shape:
            self._warn_validity_once(f"{stream} shape {data.shape} != depth shape {depth_shape} — stream ignored.")
            return None
        return data

    def _warn_validity_once(self, message: str) -> None:
        if not self._validity_warned:
            self._validity_warned = True
            logger.warning("Depth validity gate: %s", message)

    def read(self, timeout_ms: int = 5000, *, compute_depth: bool = True) -> CameraFrame:
        if self.pipeline is None:
            raise RuntimeError("RealSense is not connected. Call connect() first.")
        if self.depth_scale is None:
            raise RuntimeError("RealSense depth_scale is unavailable.")

        frames = self.pipeline.wait_for_frames(timeout_ms)
        self._apply_depth_validity(frames)
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

