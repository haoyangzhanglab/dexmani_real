"""RealSense worker that publishes frames to ``RuntimeChannels.camera_ring``.

The worker owns device truth: it publishes each frame with its health class
(clock reset, duplicate, skipped-frame telemetry, finite-but-slow delivery)
and it owns the source-stall latch — failed reads OR the absence of any new
valid source frame for the device-owned stall budget fail closed, so a stale
resident frame downstream can never mask a dead required sensor. Duplicates
and clock resets never refresh that progress counter.
"""

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
    # exceeded the freshness budget; this is not an SDK queue depth. The
    # payload is still valid and causally ordered — finite-but-slow delivery,
    # distinct from an invalid clock or payload.
    DELIVERY_DELAY = 4


# Payload-valid health classes for downstream admission: OK, skipped-frame
# telemetry, and finite-but-slow delivery all carry a valid, causally ordered
# payload; CLOCK_RESET (invalid clock) and DUPLICATE (no new source) do not.
# Raw enum meanings are unchanged — this only classifies admission.
PAYLOAD_VALID_CAMERA_HEALTH = frozenset(
    {
        int(CameraHealth.OK),
        int(CameraHealth.FRAME_GAP),
        int(CameraHealth.DELIVERY_DELAY),
    }
)


_READ_FAILURE_BACKOFF_S = 0.05


@dataclass(frozen=True)
class CameraLoopConfig:
    serial: str | None = field(default_factory=lambda: camera.serial)
    width: int = field(default_factory=lambda: camera.width)
    height: int = field(default_factory=lambda: camera.height)
    fps: int = field(default_factory=lambda: camera.fps)
    warmup_frames: int = field(default_factory=lambda: camera.warmup_frames)
    max_frame_age_s: float = field(default_factory=lambda: camera.max_frame_age_s)
    # Device-owned stall budget covering both failed reads and the absence of
    # any new valid source frame. It is never derived from model latency or
    # recording-quality parameters.
    source_stall_timeout_s: float = field(
        default_factory=lambda: camera.source_stall_timeout_s
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
            or not math.isfinite(self.source_stall_timeout_s)
            or self.source_stall_timeout_s <= self.max_frame_age_s
        ):
            raise ValueError(
                "camera frame age and source-stall thresholds must be finite and "
                "positive, with the stall timeout greater than max frame age"
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
            source_stall_timeout_s=float(cam.source_stall_timeout_s),
            frame_gap_stall_threshold=int(cam.frame_gap_stall_threshold),
            l515_visual_preset=int(cam.l515_visual_preset),
            l515_confidence_threshold=(
                None
                if cam.l515_confidence_threshold is None
                else int(cam.l515_confidence_threshold)
            ),
            frame_queue_capacity=int(cam.frame_queue_capacity),
        )


def _camera_health(
    *, clock_reset: bool, duplicate: bool, backlog_s: float, max_age_s: float
) -> CameraHealth:
    """Classify live admission; skipped older frames do not invalidate this one."""
    if clock_reset:
        return CameraHealth.CLOCK_RESET
    if duplicate:
        return CameraHealth.DUPLICATE
    if not math.isfinite(backlog_s) or not 0.0 <= backlog_s <= max_age_s:
        return CameraHealth.DELIVERY_DELAY
    return CameraHealth.OK


def pack_camera_frame(
    rgb: "np.ndarray",
    depth_raw: "np.ndarray",
    *,
    depth_frame_number: int,
    color_frame_number: int,
    camera_health: int,
    source_monotonic_ns: int,
    receive_monotonic_ns: int,
    camera_generation: int,
) -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    """Pack the live camera admission fields and fixed RGB-D layout."""
    import numpy as np

    from dexmani_real.ipc.schema import CAMERA_FRAME_HEADER_DTYPE

    rgb_arr = np.ascontiguousarray(rgb, dtype=np.uint8)
    depth_arr = np.ascontiguousarray(depth_raw, dtype=np.uint16)
    if rgb_arr.ndim != 3 or rgb_arr.shape[2] != 3:
        raise ValueError(f"RGB frame must have shape (H, W, 3), got {rgb_arr.shape}")
    if depth_arr.ndim != 2:
        raise ValueError(f"depth frame must have shape (H, W), got {depth_arr.shape}")
    if not 0 < source_monotonic_ns <= receive_monotonic_ns:
        raise ValueError("camera requires positive source time no later than receive")
    if min(depth_frame_number, color_frame_number, camera_generation) < 0:
        raise ValueError("camera frame identifiers must be non-negative")
    if int(camera_health) not in {int(item) for item in CameraHealth}:
        raise ValueError("camera_health is not a known CameraHealth value")

    header = np.zeros(1, dtype=CAMERA_FRAME_HEADER_DTYPE)
    header["source_monotonic_ns"] = source_monotonic_ns
    header["receive_monotonic_ns"] = receive_monotonic_ns
    header["camera_generation"] = camera_generation
    header["depth_frame_number"] = depth_frame_number
    header["color_frame_number"] = color_frame_number
    header["camera_health"] = camera_health
    header["rgb_size"] = rgb_arr.nbytes
    header["depth_size"] = depth_arr.nbytes
    header["rgb_shape_h"] = rgb_arr.shape[0]
    header["rgb_shape_w"] = rgb_arr.shape[1]
    header["rgb_shape_c"] = rgb_arr.shape[2]
    header["depth_shape_h"] = depth_arr.shape[0]
    header["depth_shape_w"] = depth_arr.shape[1]
    return header, rgb_arr, depth_arr


def camera_loop(
    shared: "RuntimeChannels",
    config: CameraLoopConfig,
    latch_runtime_fault: bool = True,
) -> None:
    """Run RealSense camera → write frames directly to ``shared.camera_ring``.

    Designed as an ``mp.Process`` target. Runs the camera capture loop directly
    in this process — no subprocess spawn.

    On init failure, logs the error and returns without setting the camera
    ready flag — Main detects this via ready timeout.

    ``latch_runtime_fault`` distinguishes policy-input camera (default, a
    control-critical failure) from a camera started only for recording
    evidence: when ``False``, persistent read/publication failures mark
    ``evidence_failed`` instead of the physical ``error_state`` fault latch.
    """
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
        _geometry_json = json.dumps(cam.get_geometry().to_dict(), separators=(",", ":"))
        _geometry_payload = _geometry_json.encode("utf-8")
        if len(_geometry_payload) >= 2048:
            raise RuntimeError("camera geometry exceeds shared metadata capacity")
        shared.camera_geometry.value = _geometry_payload.ljust(2048, b"\x00")
        ready_published = False
        read_failure_started_s: float | None = None
        last_new_source_s = time.monotonic()
        frame_gap_warn = ThrottledWarner(interval_s=5.0, logger=_logger)
        invalid_timing_warn = ThrottledWarner(interval_s=5.0, logger=_logger)

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
                if now_s - read_failure_started_s >= cfg.source_stall_timeout_s:
                    if latch_runtime_fault:
                        shared.error_state.value = True
                    else:
                        shared.evidence_failed.value = True
                    raise RuntimeError(
                        "camera frame reads failed for "
                        f"{now_s - read_failure_started_s:.3f}s"
                    )
                time.sleep(_READ_FAILURE_BACKOFF_S)
                continue
            read_failure_started_s = None
            now_s = time.monotonic()
            shared.set_heartbeat("camera", now_s)

            # Producer-owned source truth: only a genuinely new frame with a
            # usable device stamp advances the stream. Duplicates and invalid
            # clocks never refresh it, so a required sensor that stopped
            # producing cannot be masked forever by an old resident frame.
            # ``backlog_s`` is finite and non-negative by construction (a
            # clamped host/device difference), so the remaining device-stamp
            # class this rejects is an unusable source timestamp — exactly the
            # rule the consumers apply before they will read a frame.
            timing_valid = (
                math.isfinite(frame.backlog_s)
                and frame.backlog_s >= 0.0
                and frame.source_monotonic_ns > 0
            )
            if not frame.duplicate and not frame.clock_reset and timing_valid:
                last_new_source_s = now_s
            elif now_s - last_new_source_s >= cfg.source_stall_timeout_s:
                if latch_runtime_fault:
                    shared.error_state.value = True
                else:
                    shared.evidence_failed.value = True
                raise RuntimeError(
                    "camera source stalled (no new valid frame) for "
                    f"{now_s - last_new_source_s:.3f}s"
                )

            if _publish_payload:
                if frame.rgb is None or frame.depth_aligned_to_color_raw is None:
                    raise RuntimeError(
                        "camera configured with color must publish aligned RGB-D"
                    )
                if not timing_valid:
                    # Invalid device timing is never published as a valid
                    # frame, and never conflated with finite-but-slow delivery.
                    invalid_timing_warn(
                        "camera_loop: dropped frame with invalid device timing "
                        "(backlog_s=%r)",
                        frame.backlog_s,
                    )
                    continue
                camera_health = _camera_health(
                    clock_reset=frame.clock_reset,
                    duplicate=frame.duplicate,
                    backlog_s=frame.backlog_s,
                    max_age_s=cfg.max_frame_age_s,
                )
                if camera_health is CameraHealth.DUPLICATE:
                    # A frameset can reuse the previous depth sample. It adds no
                    # new depth observation: keep the resident RGB-D sample and
                    # its original timestamps instead of publishing a duplicate.
                    # Persistent repeats still age out at the consumers. Clock
                    # resets take precedence above and must remain visible.
                    continue
                if frame.frame_gap > cfg.resolved_frame_gap_stall_threshold:
                    frame_gap_warn(
                        "camera_loop: skipped_frames=%d threshold=%d "
                        "depth_frame=%d color_frame=%s generation=%d "
                        "health=%s(%d) backlog_ms=%.3f "
                        "(publishing current frame with reported health)",
                        frame.frame_gap,
                        cfg.resolved_frame_gap_stall_threshold,
                        frame.depth_frame_number,
                        frame.color_frame_number,
                        frame.camera_generation,
                        camera_health.name,
                        int(camera_health),
                        frame.backlog_s * 1e3,
                    )
                try:
                    header, rgb, depth = pack_camera_frame(
                        frame.rgb,
                        frame.depth_aligned_to_color_raw,
                        depth_frame_number=frame.depth_frame_number,
                        color_frame_number=frame.color_frame_number or 0,
                        camera_health=int(camera_health),
                        source_monotonic_ns=frame.source_monotonic_ns,
                        receive_monotonic_ns=frame.wait_return_monotonic_ns,
                        camera_generation=frame.camera_generation,
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
                    if latch_runtime_fault:
                        shared.error_state.value = True
                        _logger.exception(
                            "camera_loop: frame publication failed; latching runtime fault"
                        )
                    else:
                        shared.evidence_failed.value = True
                        _logger.exception(
                            "camera_loop: frame publication failed; failing session"
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
        if failed:
            if latch_runtime_fault:
                shared.error_state.value = True
            else:
                shared.evidence_failed.value = True
        else:
            _logger.debug("camera_loop: STOPPED")
        _logger.info("camera_loop: exited")
