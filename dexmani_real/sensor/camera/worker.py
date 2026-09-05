"""RealSense worker that publishes frames to ``RuntimeChannels.camera_ring``."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

from dexmani_real.config.defaults import camera
from dexmani_real.runtime.safety import SafetyState
from dexmani_real.utils.log import ThrottledWarner, get_logger

if TYPE_CHECKING:
    import numpy as np

    from dexmani_real.ipc.channels import RuntimeChannels

logger = get_logger(__name__)


class CameraHealth(IntEnum):
    OK = 0
    CLOCK_RESET = 1
    DUPLICATE = 2
    # A recovered current frame is not invalid merely because device frame
    # numbers skipped earlier frames.
    FRAME_GAP = 3
    # Current device-to-host delay above the clock mapper's lower envelope
    # exceeded the freshness budget; this is not an SDK queue depth.
    DELIVERY_DELAY = 4


_READ_FAILURE_BACKOFF_S = 0.05


@dataclass(frozen=True)
class CameraLoopConfig:
    serial: str | None = field(default_factory=lambda: camera.serial)
    width: int = field(default_factory=lambda: camera.width)
    height: int = field(default_factory=lambda: camera.height)
    fps: int = field(default_factory=lambda: camera.fps)
    warmup_frames: int = field(default_factory=lambda: camera.warmup_frames)
    max_frame_age_s: float = field(default_factory=lambda: camera.max_frame_age_s)
    read_failure_timeout_s: float = field(
        default_factory=lambda: camera.recording_stall_abort_s
    )
    frame_gap_stall_threshold: int = field(
        default_factory=lambda: camera.frame_gap_stall_threshold
    )
    l515_visual_preset: int = field(default_factory=lambda: camera.l515_visual_preset)
    l515_confidence_threshold: int | None = field(
        default_factory=lambda: camera.l515_confidence_threshold
    )
    frame_queue_capacity: int = field(
        default_factory=lambda: camera.frame_queue_capacity
    )

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.fps <= 0:
            raise ValueError("camera dimensions/rates must be positive")
        if (
            not math.isfinite(self.max_frame_age_s)
            or self.max_frame_age_s <= 0
            or not math.isfinite(self.read_failure_timeout_s)
            or self.read_failure_timeout_s <= self.max_frame_age_s
        ):
            raise ValueError(
                "camera frame age and read-failure thresholds must be finite and "
                "positive, with read-failure timeout greater than max frame age"
            )
        if self.warmup_frames < 0:
            raise ValueError("warmup_frames must be non-negative")
        if self.frame_gap_stall_threshold < 0:
            raise ValueError("frame_gap_stall_threshold must be >= 0")
        if (
            not isinstance(self.l515_visual_preset, int)
            or isinstance(self.l515_visual_preset, bool)
            or not 0 <= self.l515_visual_preset <= 5
        ):
            raise ValueError("l515_visual_preset must be an integer in [0, 5]")
        if self.l515_confidence_threshold is not None and (
            not isinstance(self.l515_confidence_threshold, int)
            or isinstance(self.l515_confidence_threshold, bool)
            or not 0 <= self.l515_confidence_threshold <= 3
        ):
            raise ValueError("l515_confidence_threshold must be in [0, 3] or None")
        if self.frame_queue_capacity <= 0:
            raise ValueError("frame_queue_capacity must be positive")

    @property
    def resolved_frame_gap_stall_threshold(self) -> int:
        """Skipped-frame count above which a recovered-frame gap is logged.

        Camera acquisition now drains at the device frame rate. A frame-number
        gap remains useful telemetry, but does not invalidate the recovered
        current frame; freshness is decided from its timestamps and payload.
        """
        if self.frame_gap_stall_threshold > 0:
            return self.frame_gap_stall_threshold
        return 1

    @classmethod
    def from_runtime(cls, runtime: object) -> "CameraLoopConfig":
        cam = getattr(runtime, "camera")
        return cls(
            serial=cam.serial,
            width=int(cam.width),
            height=int(cam.height),
            fps=int(cam.fps),
            warmup_frames=int(cam.warmup_frames),
            max_frame_age_s=float(cam.max_frame_age_s),
            read_failure_timeout_s=float(cam.recording_stall_abort_s),
            frame_gap_stall_threshold=int(cam.frame_gap_stall_threshold),
            l515_visual_preset=int(cam.l515_visual_preset),
            l515_confidence_threshold=(
                None
                if cam.l515_confidence_threshold is None
                else int(cam.l515_confidence_threshold)
            ),
            frame_queue_capacity=int(cam.frame_queue_capacity),
        )


def pack_camera_frame(
    rgb: "np.ndarray",
    depth_raw: "np.ndarray",
    depth_device_timestamp_s: float,
    color_device_timestamp_s: float | None,
    depth_frame_number: int,
    color_frame_number: int | None,
    pc_valid_depth_ratio: float = 0.0,
    camera_health: int = 0,
    source_monotonic_ns: int = 0,
    camera_generation: int = 0,
    frame_gap: int = 0,
    clock_reset: bool = False,
    duplicate: bool = False,
    backlog_s: float = 0.0,
    wait_return_monotonic_ns: int = 0,
    payload_ready_monotonic_ns: int = 0,
    depth_timestamp_domain: int = 0,
    color_timestamp_domain: int | None = None,
) -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    """Pack one depth-to-color aligned RGB-D frame with explicit timing."""
    import numpy as np

    from dexmani_real.ipc.schema import CAMERA_FRAME_HEADER_DTYPE

    rgb_arr = np.ascontiguousarray(rgb, dtype=np.uint8)
    depth_arr = np.ascontiguousarray(depth_raw, dtype=np.uint16)
    if rgb_arr.ndim != 3 or rgb_arr.shape[2] != 3:
        raise ValueError(f"RGB frame must have shape (H, W, 3), got {rgb_arr.shape}")
    if depth_arr.ndim != 2:
        raise ValueError(f"depth frame must have shape (H, W), got {depth_arr.shape}")
    numeric: tuple[float, ...] = (
        depth_device_timestamp_s,
        pc_valid_depth_ratio,
        backlog_s,
    )
    if color_device_timestamp_s is not None:
        numeric = (*numeric, color_device_timestamp_s)
    if (
        not all(np.isfinite(value) for value in numeric)
        or depth_device_timestamp_s < 0
        or backlog_s < 0
    ):
        raise ValueError(
            "camera frame timestamps/ratios/backlog must be finite and non-negative"
        )
    if not 0.0 <= pc_valid_depth_ratio <= 1.0:
        raise ValueError("pc_valid_depth_ratio must be in [0, 1]")
    integers = (
        depth_frame_number,
        source_monotonic_ns,
        camera_generation,
        wait_return_monotonic_ns,
        payload_ready_monotonic_ns,
    )
    if any(int(value) < 0 for value in integers) or frame_gap < 0:
        raise ValueError("camera frame identifiers/counts must be non-negative")
    if not 0 <= int(depth_timestamp_domain) <= 255:
        raise ValueError("depth timestamp_domain must fit uint8")
    if (
        color_timestamp_domain is not None
        and not 0 <= int(color_timestamp_domain) <= 254
    ):
        raise ValueError("color timestamp_domain must fit uint8")
    if color_frame_number is not None and int(color_frame_number) < 0:
        raise ValueError("color_frame_number must be non-negative")
    if int(camera_health) not in {int(item) for item in CameraHealth}:
        raise ValueError("camera_health is not a known CameraHealth value")
    receive_monotonic_ns = int(wait_return_monotonic_ns)
    payload_ready_ns = int(payload_ready_monotonic_ns)
    if receive_monotonic_ns <= 0 or payload_ready_ns <= 0:
        raise ValueError("camera timing stages must be positive")
    if source_monotonic_ns > receive_monotonic_ns:
        raise ValueError("camera source time cannot be later than host receive time")
    if payload_ready_ns < receive_monotonic_ns:
        raise ValueError("camera payload readiness cannot precede host receive time")

    header = np.zeros(1, dtype=CAMERA_FRAME_HEADER_DTYPE)
    header["depth_device_timestamp_s"] = np.float64(depth_device_timestamp_s)
    header["color_device_timestamp_s"] = np.float64(
        np.nan if color_device_timestamp_s is None else color_device_timestamp_s
    )
    header["source_monotonic_ns"] = np.uint64(source_monotonic_ns)
    header["receive_monotonic_ns"] = np.uint64(receive_monotonic_ns)
    header["payload_ready_monotonic_ns"] = np.uint64(payload_ready_ns)
    header["depth_timestamp_domain"] = np.uint8(depth_timestamp_domain)
    header["color_timestamp_domain"] = np.uint8(
        255 if color_timestamp_domain is None else color_timestamp_domain
    )
    header["camera_generation"] = np.uint64(camera_generation)
    header["depth_frame_number"] = np.uint64(depth_frame_number)
    header["color_frame_number"] = np.uint64(
        0 if color_frame_number is None else color_frame_number
    )
    header["frame_gap"] = np.uint32(frame_gap)
    header["clock_reset"] = np.uint8(clock_reset)
    header["duplicate"] = np.uint8(duplicate)
    header["backlog_s"] = np.float64(backlog_s)
    header["pc_valid_depth_ratio"] = np.float32(pc_valid_depth_ratio)
    header["camera_health"] = np.uint8(camera_health)

    header["rgb_size"] = np.uint64(rgb_arr.nbytes)
    header["depth_size"] = np.uint64(depth_arr.nbytes)
    header["rgb_shape_h"] = np.uint32(rgb_arr.shape[0])
    header["rgb_shape_w"] = np.uint32(rgb_arr.shape[1])
    header["rgb_shape_c"] = np.uint32(rgb_arr.shape[2])
    header["depth_shape_h"] = np.uint32(depth_arr.shape[0])
    header["depth_shape_w"] = np.uint32(depth_arr.shape[1])

    return header, rgb_arr, depth_arr


def camera_loop(shared: "RuntimeChannels", config: CameraLoopConfig) -> None:
    """Run RealSense camera → write frames directly to ``shared.camera_ring``.

    Designed as an ``mp.Process`` target. Runs the camera capture loop directly
    in this process — no subprocess spawn.

    On init failure, logs the error and returns without setting the camera
    ready flag — Main detects this via ready timeout.
    """
    import numpy as np

    _logger = get_logger("camera_loop")
    if not isinstance(config, CameraLoopConfig):
        raise TypeError("camera_loop requires a CameraLoopConfig")
    cfg = config
    failed = False
    _logger.debug("camera_loop: LOADING")

    # Limit library thread pools so process-level scheduling remains predictable.
    try:
        import cv2

        cv2.setNumThreads(1)
    except ImportError:
        pass

    cam = None
    try:
        from dexmani_real.sensor.camera.realsense import (
            L515DepthConfig,
            RealSenseCamera,
            RealSenseCameraConfig,
        )

        rs_config = RealSenseCameraConfig(
            camera_name="realsense",
            serial=cfg.serial,
            depth_resolution=(cfg.width, cfg.height),
            color_resolution=(cfg.width, cfg.height),
            fps=cfg.fps,
            warmup_frames=cfg.warmup_frames,
            frame_queue_capacity=cfg.frame_queue_capacity,
            l515_depth_config=L515DepthConfig(
                visual_preset=cfg.l515_visual_preset,
                confidence_threshold=cfg.l515_confidence_threshold,
            ),
        )
        cam = RealSenseCamera(rs_config)

        if not cam.connect():
            _logger.error("camera_loop: RealSense connect failed")
            failed = True
            return

        shared.camera_depth_scale.value = float(cam.get_depth_scale())
        _serial_raw = str(cam.active_serial or "")
        shared.camera_serial.value = _serial_raw[:31].ljust(32, "\x00").encode()
        _device_info = cam.get_device_info()
        _firmware = str(_device_info.get("firmware", ""))
        shared.camera_firmware.value = _firmware[:63].ljust(64, "\x00").encode()
        try:
            import pyrealsense2 as rs

            _sdk_version = str(getattr(rs, "__version__", "unknown"))
        except Exception:
            _sdk_version = "unknown"
        shared.camera_sdk_version.value = _sdk_version[:63].ljust(64, "\x00").encode()
        _geometry_json = json.dumps(cam.get_geometry().to_dict(), separators=(",", ":"))
        _geometry_payload = _geometry_json.encode("utf-8")
        if len(_geometry_payload) >= 2048:
            raise RuntimeError("camera geometry exceeds shared metadata capacity")
        shared.camera_geometry.value = _geometry_payload.ljust(2048, b"\x00")
        _profile_json = json.dumps(
            {
                "streams": cam.get_active_profiles(),
                "payload_mode": "depth_to_color_aligned_rgbd",
                "depth_payload_frame": "color",
                "native_depth_retained_by_driver": True,
                "frame_queue_capacity": cam.config.frame_queue_capacity,
                "l515_depth_options": cam.get_l515_depth_option_snapshot(),
                "geometry": json.loads(_geometry_json),
            },
            separators=(",", ":"),
        )
        _profile_payload = _profile_json.encode("utf-8")
        if len(_profile_payload) >= 2048:
            raise RuntimeError(
                "camera profile and L515 option snapshot exceed shared metadata capacity"
            )
        shared.camera_profile.value = _profile_payload.ljust(2048, b"\x00")

        ready_published = False
        read_failure_started_s: float | None = None
        frame_gap_warn = ThrottledWarner(interval_s=5.0, logger=_logger)

        while shared.is_running.value:
            _publish_payload = (
                not ready_published
                or int(shared.safety_state.value) == int(SafetyState.DISARMED)
                or bool(shared.is_recording.value)
                or bool(shared.camera_requested.value)
            )
            try:
                frame = cam.read(timeout_ms=300, compute_depth=False)
            except (RuntimeError, OSError):
                now_s = time.monotonic()
                if read_failure_started_s is None:
                    read_failure_started_s = now_s
                _logger.warning("camera_loop: frame read failed", exc_info=True)
                # ``wait_for_frames`` normally provides the device-rate pacing.
                # Back off only after a failure so repeated errors cannot spin.
                shared.set_heartbeat("camera", now_s)
                if now_s - read_failure_started_s >= cfg.read_failure_timeout_s:
                    shared.error_state.value = True
                    raise RuntimeError(
                        "camera frame reads failed for "
                        f"{now_s - read_failure_started_s:.3f}s"
                    )
                time.sleep(_READ_FAILURE_BACKOFF_S)
                continue
            read_failure_started_s = None
            shared.set_heartbeat("camera", time.monotonic())

            if _publish_payload:
                if frame.rgb is None or frame.depth_aligned_to_color_raw is None:
                    raise RuntimeError(
                        "camera configured with color must publish aligned RGB-D"
                    )
                pc_valid_depth_ratio = float(
                    np.count_nonzero(frame.depth_aligned_to_color_raw)
                    / frame.depth_aligned_to_color_raw.size
                )
                if frame.clock_reset:
                    camera_health = CameraHealth.CLOCK_RESET
                elif frame.duplicate:
                    camera_health = CameraHealth.DUPLICATE
                elif frame.backlog_s > cfg.max_frame_age_s:
                    camera_health = CameraHealth.DELIVERY_DELAY
                else:
                    camera_health = CameraHealth.OK
                if frame.frame_gap > cfg.resolved_frame_gap_stall_threshold:
                    frame_gap_warn(
                        "camera_loop: device frame gap=%d (current frame retained; threshold=%d)",
                        frame.frame_gap,
                        cfg.resolved_frame_gap_stall_threshold,
                    )
                try:
                    header, rgb, depth = pack_camera_frame(
                        frame.rgb,
                        frame.depth_aligned_to_color_raw,
                        frame.depth_device_timestamp_s,
                        frame.color_device_timestamp_s,
                        frame.depth_frame_number,
                        frame.color_frame_number,
                        pc_valid_depth_ratio=pc_valid_depth_ratio,
                        camera_health=int(camera_health),
                        source_monotonic_ns=frame.source_monotonic_ns,
                        camera_generation=frame.camera_generation,
                        frame_gap=frame.frame_gap,
                        clock_reset=frame.clock_reset,
                        duplicate=frame.duplicate,
                        backlog_s=frame.backlog_s,
                        wait_return_monotonic_ns=frame.wait_return_monotonic_ns,
                        payload_ready_monotonic_ns=frame.payload_ready_monotonic_ns,
                        depth_timestamp_domain=frame.depth_timestamp_domain,
                        color_timestamp_domain=frame.color_timestamp_domain,
                    )
                    shared.camera_ring.write(header, rgb, depth)
                    if not ready_published:
                        ready_published = True
                        shared.set_ready("camera")
                        _logger.debug("camera_loop: READY")
                        _logger.info(
                            "camera_loop: ready after first verified frame @ device %.1f Hz",
                            cfg.fps,
                        )
                except Exception:
                    failed = True
                    shared.error_state.value = True
                    _logger.exception(
                        "camera_loop: frame publication failed; latching runtime fault"
                    )
                    return
    except Exception:
        failed = True
        _logger.exception("camera_loop: crashed")
    finally:
        if cam is not None:
            try:
                cam.disconnect()
            except Exception:
                failed = True
                _logger.warning("camera_loop: disconnect failed", exc_info=True)
        if not failed:
            _logger.debug("camera_loop: STOPPED")
        _logger.info("camera_loop: exited")
