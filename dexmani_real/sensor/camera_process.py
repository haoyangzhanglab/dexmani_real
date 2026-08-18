"""RealSense process that publishes frames to ``SharedStorage.camera_ring``."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Literal, cast

from dexmani_real.config.defaults import camera, policy
from dexmani_real.robot.safety import SafetyState
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate_manager import RateManager

if TYPE_CHECKING:
    import numpy as np

    from dexmani_real.shm.shared_storage import SharedStorage

logger = get_logger(__name__)


class CameraHealth(IntEnum):
    OK = 0
    CLOCK_RESET = 1
    DUPLICATE = 2
    FRAME_GAP = 3
    BACKLOG = 4


@dataclass(frozen=True)
class CameraLoopConfig:
    serial: str | None = field(default_factory=lambda: camera.serial)
    width: int = field(default_factory=lambda: camera.width)
    height: int = field(default_factory=lambda: camera.height)
    fps: int = field(default_factory=lambda: camera.fps)
    align_mode: Literal["depth_to_color", "color_to_depth", "none"] = field(default_factory=lambda: camera.align_mode)
    warmup_frames: int = field(default_factory=lambda: camera.warmup_frames)
    pointcloud_num_points: int = field(default_factory=lambda: camera.pointcloud_num_points)
    publish_hz: float = field(default_factory=lambda: policy.control_hz)
    max_frame_age_s: float = field(default_factory=lambda: camera.max_frame_age_s)
    frame_gap_stall_threshold: int = field(default_factory=lambda: camera.frame_gap_stall_threshold)
    desk_plane: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if (
            self.width <= 0
            or self.height <= 0
            or self.fps <= 0
            or not math.isfinite(self.publish_hz)
            or not math.isfinite(self.max_frame_age_s)
            or self.publish_hz <= 0
            or self.max_frame_age_s <= 0
        ):
            raise ValueError("camera dimensions/rates must be positive")
        if self.pointcloud_num_points <= 0:
            raise ValueError("pointcloud_num_points must be positive")
        if self.frame_gap_stall_threshold < 0:
            raise ValueError("frame_gap_stall_threshold must be >= 0 (0 = derive from fps/publish_hz)")

    @property
    def resolved_frame_gap_stall_threshold(self) -> int:
        """Skipped-frame count above which a read gap is flagged as a stall.

        A read throttled to ``publish_hz`` below the device ``fps`` inherently
        skips ~``fps/publish_hz`` frames per read; only a gap beyond that
        ceiling (plus jitter) is a real pipeline stall.  A positive configured
        threshold overrides the derived value.
        """
        if self.frame_gap_stall_threshold > 0:
            return self.frame_gap_stall_threshold
        return int(math.ceil(self.fps / self.publish_hz))

    @classmethod
    def from_runtime(cls, runtime: object) -> "CameraLoopConfig":
        cam = getattr(runtime, "camera")
        pol = getattr(runtime, "policy")
        # The resolved environment.table.plane_abcd is the single desk-plane
        # source of truth (already refreshed from plane_path at resolve time);
        # consume it here instead of letting the pointcloud processor auto-load
        # its own hardcoded desk_plane.json.  A disabled table means "no desk
        # removal" rather than "re-read the calibration file".
        table = getattr(getattr(runtime, "environment"), "table")
        desk_plane = (
            cast(tuple[float, float, float, float], tuple(float(v) for v in table.plane_abcd))
            if table.enabled
            else None
        )
        return cls(
            serial=cam.serial,
            width=int(cam.width),
            height=int(cam.height),
            fps=int(cam.fps),
            align_mode=cam.align_mode,
            warmup_frames=int(cam.warmup_frames),
            pointcloud_num_points=int(cam.pointcloud_num_points),
            publish_hz=float(pol.control_hz),
            max_frame_age_s=float(cam.max_frame_age_s),
            frame_gap_stall_threshold=int(cam.frame_gap_stall_threshold),
            desk_plane=desk_plane,
        )


def pack_camera_frame(
    rgb: "np.ndarray",
    depth_raw: "np.ndarray",
    timestamp: float,
    capture_monotonic_s: float,
    frame_id: int,
    pc_valid_depth_ratio: float = 0.0,
    camera_health: int = 0,
    source_monotonic_ns: int = 0,
    camera_generation: int = 0,
    frame_gap: int = 0,
    clock_reset: bool = False,
    duplicate: bool = False,
    backlog_s: float = 0.0,
) -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    """Pack camera frame attributes into (header, rgb_bytes, depth_bytes)."""
    import numpy as np

    from dexmani_real.utils.schema import CAMERA_FRAME_HEADER_DTYPE

    rgb_arr = np.ascontiguousarray(rgb, dtype=np.uint8)
    depth_arr = np.ascontiguousarray(depth_raw, dtype=np.uint16)
    if rgb_arr.ndim != 3 or rgb_arr.shape[2] != 3:
        raise ValueError(f"RGB frame must have shape (H, W, 3), got {rgb_arr.shape}")
    if depth_arr.ndim != 2 or depth_arr.shape != rgb_arr.shape[:2]:
        raise ValueError(f"depth frame must match RGB H/W, got {depth_arr.shape} and {rgb_arr.shape}")
    numeric = (timestamp, capture_monotonic_s, pc_valid_depth_ratio, backlog_s)
    if not all(np.isfinite(value) for value in numeric) or capture_monotonic_s < 0 or backlog_s < 0:
        raise ValueError("camera frame timestamps/ratios/backlog must be finite and non-negative")
    if not 0.0 <= pc_valid_depth_ratio <= 1.0:
        raise ValueError("pc_valid_depth_ratio must be in [0, 1]")
    integers = (
        frame_id,
        source_monotonic_ns,
        camera_generation,
    )
    if any(int(value) < 0 for value in integers) or frame_gap < 0:
        raise ValueError("camera frame identifiers/counts must be non-negative")
    if int(camera_health) not in {int(item) for item in CameraHealth}:
        raise ValueError("camera_health is not a known CameraHealth value")
    receive_monotonic_ns = max(0, round(capture_monotonic_s * 1e9))
    if source_monotonic_ns > receive_monotonic_ns:
        raise ValueError("camera source time cannot be later than host receive time")

    header = np.zeros(1, dtype=CAMERA_FRAME_HEADER_DTYPE)
    header["timestamp"] = np.float64(timestamp)
    header["capture_monotonic_s"] = np.float64(capture_monotonic_s)
    header["source_monotonic_ns"] = np.uint64(source_monotonic_ns)
    header["receive_monotonic_ns"] = np.uint64(receive_monotonic_ns)
    header["camera_generation"] = np.uint64(camera_generation)
    header["frame_number"] = np.uint64(frame_id)
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


def camera_loop(shared: "SharedStorage", config: CameraLoopConfig | None = None) -> None:
    """Run RealSense camera → write frames directly to ``shared.camera_ring``.

    Designed as an ``mp.Process`` target. Runs the camera capture loop directly
    in this process — no subprocess spawn.

    On init failure, logs the error and returns without setting the camera
    ready flag — Main detects this via ready timeout.
    """
    import numpy as np

    _logger = get_logger("camera_loop")
    cfg = config or CameraLoopConfig()
    failed = False
    _logger.debug("camera_loop: LOADING")

    # Calibration and aligned depth must use the same color-optical frame.
    # Reject incompatible alignment before opening the device SDK.
    if cfg.align_mode != "depth_to_color":
        _logger.error(
            "camera_loop: align_mode=%r is unsupported for recording; "
            "the color-optical calibration requires 'depth_to_color'",
            cfg.align_mode,
        )
        _logger.info("camera_loop: exited")
        return

    # Thread pool limit.
    # OpenCV/NumPy default to multi-threading on many-core machines, competing
    # for CPU with the configured policy loop. We rely on process-level parallelism
    # (arm/hand/camera each in its own process), not per-library thread pools.
    try:
        import cv2

        cv2.setNumThreads(1)
    except ImportError:
        pass

    cam = None
    try:
        # Create and connect camera.
        from dexmani_real.sensor.realsense import RealSense, RealSenseConfig

        rs_config = RealSenseConfig(
            camera_name="realsense",
            serial=cfg.serial,
            depth_resolution=(cfg.width, cfg.height),
            color_resolution=(cfg.width, cfg.height),
            fps=cfg.fps,
            align_mode=cfg.align_mode,
            warmup_frames=cfg.warmup_frames,
        )
        cam = RealSense(rs_config)

        if not cam.connect():
            _logger.error("camera_loop: RealSense connect failed")
            failed = True
            return

        # Publish metadata.
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
        _profile_json = json.dumps(
            {
                "streams": cam.get_active_profiles(),
                "align_mode": cam.config.align_mode,
                "common_viewport": "color",
                "output_optical_frame": cam.config.frame_name,
                "output_intrinsics": cam.get_intrinsics_info(),
            },
            separators=(",", ":"),
        )
        shared.camera_profile.value = _profile_json[:2047].ljust(2048, "\x00").encode()
        if cam.K is not None:
            shared.camera_K[:] = cam.K.flatten().tolist()

        # Publish point-cloud derivation metadata.
        # The point cloud is no longer computed in this loop: it is derived at
        # the consumption boundary (offline in data_processing, online in the
        # deployment observation adapter) from the recorded depth.  We still
        # publish the resolved processor parameters so both derivation sites can
        # reconstruct an identical point cloud from the same depth/intrinsics/
        # extrinsics (train/inference distribution consistency).
        try:
            from dexmani_real.config.camera_calib import CameraCalib
            from dexmani_real.sensor.pointcloud_processor import PointCloudProcessorConfig

            calib = CameraCalib()
            cam_name = calib.resolve_name_by_serial(str(cam.active_serial))
            _t_world_camera = calib.get_extrinsics(cam_name)
            pc_kwargs: dict[str, Any] = {
                "num_points": cfg.pointcloud_num_points,
                "desk_plane": cfg.desk_plane,
            }
            if cfg.desk_plane is None:
                # No resolved plane (table disabled): fall back to workspace
                # z_min rather than the processor's own desk_plane.json auto-load.
                pc_kwargs["desk_plane_path"] = ""
            pc_config = PointCloudProcessorConfig(**pc_kwargs)
            _pc_meta = json.dumps(pc_config.to_meta_dict(), separators=(",", ":"))
            shared.camera_pointcloud_config.value = _pc_meta[:2047].ljust(2048, "\x00").encode()
            _logger.info(
                "camera_loop: pointcloud derivation config published, T pos=%s",
                _t_world_camera[:3, 3].round(3).tolist(),
            )
        except Exception:
            _logger.warning("camera_loop: pointcloud derivation config DISABLED", exc_info=True)
            shared.camera_pointcloud_config.value = "{}".ljust(2048, "\x00").encode()

        # Main capture loop.
        rate_mgr = RateManager(cfg.publish_hz)
        ready_published = False

        while shared.is_running.value:
            # Publish startup/recording RGB-D payloads; publish metadata-only
            # frames during steady operation to keep the camera ring current.
            _publish_payload = (
                not ready_published
                or int(shared.safety_state.value) == int(SafetyState.DISARMED)
                or bool(shared.is_recording.value)
            )
            # Read frame.
            try:
                frame = cam.read(timeout_ms=300, compute_depth=False)
            except (RuntimeError, OSError):
                _logger.warning("camera_loop: frame read failed", exc_info=True)
                # This heartbeat represents worker liveness.  Source freshness
                # is carried by capture_monotonic_s and enforced by Policy.
                shared.set_heartbeat("camera", time.monotonic())
                # Maintain target rate even on read failure so a persistent
                # error doesn't turn into a tight loop.
                rate_mgr.wait()
                continue
            shared.set_heartbeat("camera", time.monotonic())

            # Write to the shared ring.
            if _publish_payload:
                pc_valid_depth_ratio = float(np.count_nonzero(frame.depth_raw) / frame.depth_raw.size)
                if frame.clock_reset:
                    camera_health = CameraHealth.CLOCK_RESET
                elif frame.duplicate:
                    camera_health = CameraHealth.DUPLICATE
                elif frame.frame_gap > cfg.resolved_frame_gap_stall_threshold:
                    camera_health = CameraHealth.FRAME_GAP
                elif frame.backlog_s > cfg.max_frame_age_s:
                    camera_health = CameraHealth.BACKLOG
                else:
                    camera_health = CameraHealth.OK
                try:
                    header, rgb, depth = pack_camera_frame(
                        frame.rgb,  # type: ignore[arg-type]
                        frame.depth_raw,
                        frame.timestamp,
                        frame.capture_monotonic_s,
                        frame.frame_id,
                        pc_valid_depth_ratio=pc_valid_depth_ratio,
                        camera_health=int(camera_health),
                        source_monotonic_ns=frame.source_monotonic_ns,
                        camera_generation=frame.camera_generation,
                        frame_gap=frame.frame_gap,
                        clock_reset=frame.clock_reset,
                        duplicate=frame.duplicate,
                        backlog_s=frame.backlog_s,
                    )
                    shared.camera_ring.write(header, rgb, depth)
                    if not ready_published:
                        ready_published = True
                        shared.set_ready("camera")
                        _logger.debug("camera_loop: READY")
                        _logger.info("camera_loop: ready after first verified frame @ %.1f Hz", cfg.publish_hz)
                except Exception:
                    _logger.warning("camera_loop: ring write failed", exc_info=True)

            # Maintain the target rate.
            rate_mgr.wait()

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
